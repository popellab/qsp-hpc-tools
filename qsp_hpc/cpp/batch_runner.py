"""Run a batch of C++ QSP simulations and write one MATLAB-compatible Parquet.

This layer sits between the single-sim CppRunner (M3) and the top-level
CppSimulator (M5). Given a `theta_matrix` of parameter samples plus the
priors column names, it fans out over a ProcessPoolExecutor, collects
trajectories, and writes a Parquet whose schema matches what the MATLAB
pipeline has been emitting — so downstream caching / test-stat
derivation code keeps working without edits.

Schema (one row per simulation):
    simulation_id:  int64     # zero-indexed within this batch
    status:         int64     # 0 = success, 1 = qsp_sim failure
    time:           list<float64>   # length n_times (same for all rows)
    param:<name>:   float64   # one column per priors-CSV param
    <species>:      list<float64>   # one column per qsp_sim species
    <compartment>:  list<float64>   # v2 binaries: one per compartment
    <rule>:         list<float64>   # v2 binaries: one per assignment rule

The compartment and rule columns are emitted as bare names (no prefix)
so calibration-target functions can read them via ``species_dict[name]``
just like the MATLAB SimBiology output.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import (
    BrokenExecutor,
    ProcessPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qsp_hpc.cpp.evolve_cache import EvolveCache
from qsp_hpc.cpp.param_xml import ParamNotFoundError
from qsp_hpc.cpp.qsth import theta_hash_for_xml, wire_hash
from qsp_hpc.cpp.runner import CppRunner, QspSimError, SimResult

logger = logging.getLogger(__name__)


STATUS_OK = 0
STATUS_FAILED = 1
STATUS_FUTURE_TIMEOUT = 2  # worker future never completed (pool broken / hung worker)


POOL_MANIFEST_FILENAME = "pool_manifest.json"
POOL_MANIFEST_SCHEMA = "thin-v1"


def write_pool_manifest(
    pool_dir: Path | str,
    template_defaults: dict[str, float],
    sampled_params: Sequence[str],
) -> Path:
    """Write ``pool_manifest.json`` with template defaults + sampled set.

    One manifest per pool dir, written once (idempotent — existing
    manifests are left alone). Downstream consumers (derive_test_stats
    worker, CppSimulator local cache loader) fall back to these defaults
    when a parameter's ``param:{name}`` column is missing from the thin
    parquet — see #23.

    Layout::

        {
            "schema_version": "thin-v1",
            "template_defaults": {"A": 1.5, "B": 2.0, ...},  # EVERY model param
            "sampled_params": ["A", "C", ...]                # subset varied
        }

    The ``sampled_params`` list is informational — writers put those
    columns into the parquet; readers can trust the parquet columns for
    sampled values and the manifest for everything else.
    """
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pool_dir / POOL_MANIFEST_FILENAME
    if manifest_path.exists():
        return manifest_path
    payload = {
        "schema_version": POOL_MANIFEST_SCHEMA,
        "template_defaults": {str(k): float(v) for k, v in template_defaults.items()},
        "sampled_params": list(sampled_params),
    }
    # Atomic write so a partial read never happens: a parallel cal-target
    # evaluator could race a first-run write otherwise. Every SLURM array
    # task races to write this on first run of a fresh pool, so the tmp
    # filename MUST be unique per-process — a single shared
    # "pool_manifest.json.tmp" lets task 0 rename its tmp to the final
    # path and leaves task 1 without a tmp to rename (FileNotFoundError,
    # #hit on SBI smoke 2026-04-17). os.getpid() is enough even across
    # different hosts because two tasks writing simultaneously into the
    # same scratch dir with the same PID would need the same PID to be
    # reused within milliseconds — negligible. Content is identical
    # across writers so "last writer wins" is safe.
    tmp_path = manifest_path.with_suffix(f".json.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        tmp_path.replace(manifest_path)
    finally:
        # If replace() succeeded the tmp file no longer exists (renamed);
        # on failure we want to clean up to avoid littering the pool dir.
        if tmp_path.exists():
            tmp_path.unlink()
    return manifest_path


def load_pool_manifest(pool_dir: Path | str) -> dict | None:
    """Load ``pool_manifest.json`` from a pool dir, or None if absent.

    Pre-#23 pools have no manifest — parquets carry the full param set
    inline, so callers treat None as "no fallback needed".
    """
    pool_dir = Path(pool_dir)
    manifest_path = pool_dir / POOL_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    with open(manifest_path) as fh:
        return json.load(fh)


@dataclass
class BatchResult:
    """Summary returned after a batch completes."""

    parquet_path: Path
    n_sims: int
    n_failed: int
    species_names: list[str]
    n_times: int
    compartment_names: list[str] | None = None
    rule_names: list[str] | None = None
    # QSEP shard this batch wrote into the persistent evolve cache —
    # one shard holding the post-evolve states for every theta this batch
    # computed on a cache miss (#90). None when the cache was off, every
    # theta hit, or no evolve states were captured (trajectory-dump mode,
    # or no healthy_state_yaml).
    evolve_shard_path: Path | None = None


@dataclass
class FusedScenarioSpec:
    """One scenario inside a fused multi-scenario batch (#90 Phase 2).

    A fused batch resolves each theta's post-evolve ODE state **once**
    and runs every scenario from it. Each scenario still gets its own
    Parquet (``output_path``) so per-scenario pools / test_stats trees
    stay independently cacheable — fusion is task-execution-level, not
    storage-level.

    ``start_index`` is the global ``sample_index`` below which this
    scenario is skipped: a partially-cached scenario (pool already at
    depth ``n_X``) only needs the deficit tail, so a theta with
    ``sample_index < start_index`` is not run for it even though the
    fused array spans the union of all scenarios' deficits. See the
    issue #90 "partial misses & per-scenario deficits" design note.
    """

    name: str
    output_path: Path
    scenario_yaml: str | Path | None = None
    drug_metadata_yaml: str | Path | None = None
    start_index: int = 0


# --- Worker (module-level so ProcessPoolExecutor can pickle it) -------------

# Each worker process holds one CppRunner for its lifetime; we initialize it
# in an executor `initializer` and stash it as a module global. The hot path
# then avoids repaying ParamXMLRenderer parse + binary-path validation on
# every sim.
_WORKER_RUNNER: CppRunner | None = None
_WORKER_WORKDIR: Path | None = None
# Optional: when the batch was constructed with evolve_cache_root, every
# worker holds a read-only EvolveCache over the same namespace (binary +
# healthy_state). Per-sim the worker checks the cache: a hit runs the
# scenario from the cached post-evolve state via --initial-state; a miss
# evolves once and carries the new state back to the parent for
# write-through (#90). The namespace's append-only QSEP shards make this
# NFS-safe — no shared writable store, no cross-task locking.
_WORKER_EVOLVE_CACHE: EvolveCache | None = None


def _worker_init(
    binary_path: str,
    template_path: str,
    subtree: str | None,
    workdir: str,
    default_timeout_s: float,
    scenario_yaml: str | None,
    drug_metadata_yaml: str | None,
    healthy_state_yaml: str | None,
    evolve_cache_root: str | None,
) -> None:
    global _WORKER_RUNNER, _WORKER_WORKDIR, _WORKER_EVOLVE_CACHE
    # Configure the qsp_hpc parent logger inside the worker so descendant
    # loggers (qsp_hpc.cpp.evolve_cache, this module) emit to stdout even
    # when the pool was created with the "spawn" start method (Python 3.14+
    # on Linux). Under "fork" this no-ops since the logger is already
    # configured by the parent cpp_batch_worker import — setup_logger is
    # idempotent. See #34 / #36.
    from qsp_hpc.utils.logging_config import setup_logger

    setup_logger("qsp_hpc", verbose=True)

    _WORKER_RUNNER = CppRunner(
        binary_path=binary_path,
        template_path=template_path,
        subtree=subtree,
        default_timeout_s=default_timeout_s,
        scenario_yaml=scenario_yaml,
        drug_metadata_yaml=drug_metadata_yaml,
        healthy_state_yaml=healthy_state_yaml,
    )
    # Per-process subdir so concurrent workers don't fight over the same
    # `failed/` folder or race on UUID collisions (unlikely but cheap).
    _WORKER_WORKDIR = Path(workdir) / f"worker_{os.getpid()}"
    _WORKER_WORKDIR.mkdir(parents=True, exist_ok=True)

    if evolve_cache_root is not None and healthy_state_yaml is not None:
        # Each worker scans the namespace once here (footer-only reads —
        # see EvolveCache.load); a manifest, when present, bounds that to
        # uncompacted shards. Read-only — write-through happens in the
        # parent, which packs every worker's misses into one shard.
        _WORKER_EVOLVE_CACHE = EvolveCache.for_run(
            evolve_cache_root,
            binary_path=binary_path,
            healthy_state_yaml=healthy_state_yaml,
        ).load()
        logger.info(
            "worker %d: evolve cache ENABLED, namespace_dir=%s (%d cached theta state(s))",
            os.getpid(),
            _WORKER_EVOLVE_CACHE.namespace_dir,
            len(_WORKER_EVOLVE_CACHE),
        )
    else:
        _WORKER_EVOLVE_CACHE = None
        logger.info(
            "worker %d: evolve cache DISABLED " "(evolve_cache_root=%s, healthy_state_yaml=%s)",
            os.getpid(),
            evolve_cache_root,
            healthy_state_yaml,
        )


def _run_one_in_worker(
    sim_id: int,
    sample_index: int,
    params: dict[str, float],
    t_end_days: float,
    min_cadence_hours: float,
    timeout_s: float | None,
    evolve_trajectory_dir: str | None = None,
    evolve_trajectory_dt_days: float | None = None,
) -> tuple[
    int,
    int,
    np.ndarray | None,
    np.ndarray | None,
    list[str] | None,
    list[str] | None,
    list[str] | None,
    str | None,
    str | None,
    bytes | None,
]:
    """Return (sim_id, status, trajectory, species, comps, rules, err,
    theta_hash, evolve_blob).

    ``evolve_trajectory_dir`` / ``evolve_trajectory_dt_days`` are the
    *effective* values resolved by the caller (CppBatchRunner.run()) —
    per-call overrides take priority over __init__ defaults. ``None``
    here means "no dump for this sim".

    Evolve cache (#90): when a cache is wired (and there is an evolve
    phase and no trajectory dump), the worker checks it per theta. A hit
    runs the scenario from the cached post-evolve state via
    ``--initial-state``. A miss evolves once via ``--dump-state``, runs
    the scenario from that state, and carries the new QSTH blob back to
    the parent (``theta_hash`` + ``evolve_blob``) for write-through into
    a cache shard. ``theta_hash`` / ``evolve_blob`` are non-None *only*
    for a miss this worker computed — a hit, or a disabled cache, leaves
    both None. The blob is returned even when the scenario sim then
    fails: a completed evolve is reusable regardless of the outcome.
    """
    assert _WORKER_RUNNER is not None, "_worker_init must be called first"
    assert _WORKER_WORKDIR is not None
    _worker_t0 = time.time()
    logging.getLogger(__name__).debug(
        "worker pid=%d start sim_id=%d sample_index=%d",
        os.getpid(),
        sim_id,
        sample_index,
    )
    # theta_hash / evolve_blob carry a write-through payload (#90) back to
    # the parent: they are populated only when this worker evolved a theta
    # on a cache MISS. Bound before the try so the except path can still
    # return the blob when the evolve succeeded but the scenario sim then
    # failed — a completed evolve is reusable either way.
    theta_hash: str | None = None
    evolve_blob: bytes | None = None
    try:
        # Resolve the post-evolve ODE state for this theta.
        #
        # The evolve cache is skipped when ``evolve_trajectory_dir`` is
        # set: cached-state mode loads a post-evolve snapshot, which by
        # design skips the burn-in, and honoring the cache would silently
        # produce zero trajectory binaries (#69's worker-side
        # "trajectories take priority over cache" half).
        #
        #   - cache wired + evolve phase: get → a hit runs the scenario
        #     from the cached state; a miss evolves once via --dump-state,
        #     runs the scenario from that state, and carries the blob back
        #     for write-through. A non-None cache implies healthy_state.
        #   - no cache: a plain full evolve+scenario run.
        evolve_state_path: Path | None = None
        params_hash: str | None = None
        if _WORKER_EVOLVE_CACHE is not None and evolve_trajectory_dir is None:
            xml = _WORKER_RUNNER._renderer.render(params)
            th = theta_hash_for_xml(xml)
            params_hash = wire_hash(th)
            evolve_state_path = _WORKER_WORKDIR / f"{th[:16]}.evolve_state.bin"
            cached = _WORKER_EVOLVE_CACHE.get(th)
            if cached is not None:
                # HIT — materialize the cached state for --initial-state.
                evolve_state_path.write_bytes(cached)
            else:
                # MISS — evolve once via --dump-state, then run from it.
                # The blob travels back to the parent for write-through;
                # theta_hash stays set so the parent files it correctly.
                _WORKER_RUNNER.dump_evolve_state(
                    params=params,
                    params_hash=params_hash,
                    state_out=evolve_state_path,
                    workdir=_WORKER_WORKDIR,
                    timeout_s=timeout_s,
                )
                theta_hash = th
                evolve_blob = evolve_state_path.read_bytes()
        # Burn-in trajectory dump path (per-sim, keyed by sample_index so
        # the assembler can join back to theta rows). Only set when the
        # batch enabled it AND we're not in cached-state mode (the cache
        # path skips burn-in entirely).
        traj_path: Path | None = None
        traj_dt: float | None = None
        if evolve_trajectory_dir is not None and evolve_state_path is None:
            traj_path = Path(evolve_trajectory_dir) / f"sim_{sample_index:09d}.bin"
            traj_dt = evolve_trajectory_dt_days
        try:
            result: SimResult = _WORKER_RUNNER.run_one(
                params=params,
                t_end_days=t_end_days,
                min_cadence_hours=min_cadence_hours,
                workdir=_WORKER_WORKDIR,
                timeout_s=timeout_s,
                evolve_state_path=evolve_state_path,
                params_hash=params_hash,
                evolve_trajectory_path=traj_path,
                evolve_trajectory_dt_days=traj_dt,
            )
        finally:
            # evolve_state_path is a per-theta tmp file (a materialized
            # cache hit, or a fresh --dump-state); drop it so the worker
            # workdir doesn't accumulate one file per distinct theta seen.
            if evolve_state_path is not None:
                Path(evolve_state_path).unlink(missing_ok=True)
        logging.getLogger(__name__).debug(
            "worker pid=%d done  sim_id=%d sample_index=%d (%.2fs, n_times=%d)",
            os.getpid(),
            sim_id,
            sample_index,
            time.time() - _worker_t0,
            int(result.trajectory.shape[0]) if result.trajectory is not None else 0,
        )
        return (
            sim_id,
            STATUS_OK,
            result.trajectory,
            result.time_days,
            result.species_names,
            result.compartment_names,
            result.rule_names,
            None,
            theta_hash,
            evolve_blob,
        )
    except (QspSimError, ParamNotFoundError) as e:
        # Per-worker FAIL is logged at debug — the collector loop emits a
        # single aggregate "sim N failed" warning per failure (see run()),
        # which is the non-verbose failure report.
        logging.getLogger(__name__).debug(
            "worker pid=%d FAIL sim_id=%d sample_index=%d (%.2fs): %s",
            os.getpid(),
            sim_id,
            sample_index,
            time.time() - _worker_t0,
            str(e)[:120],
        )
        # theta_hash / evolve_blob are populated when the evolve completed
        # before the scenario sim failed — still emitted so the pack keeps
        # a reusable evolve state for that theta.
        return (
            sim_id,
            STATUS_FAILED,
            None,
            None,
            None,
            None,
            None,
            str(e),
            theta_hash,
            evolve_blob,
        )


def _run_one_fused_in_worker(
    sim_id: int,
    sample_index: int,
    params: dict[str, float],
    t_end_days: float,
    min_cadence_hours: float,
    timeout_s: float | None,
    scenario_specs: list[tuple[str, str | None, str | None]],
) -> tuple[int, str | None, bytes | None, list[tuple]]:
    """Evolve one theta once, run every requested scenario from that state.

    The fused multi-scenario worker (#90 Phase 2). ``scenario_specs`` is
    the subset of scenarios this theta participates in —
    ``(name, scenario_yaml, drug_metadata_yaml)`` tuples; the parent has
    already filtered out scenarios whose ``start_index`` excludes this
    theta. ``scenario_yaml`` is None for an undosed scenario.

    Returns ``(sim_id, theta_hash, evolve_blob, results)``:
      - ``theta_hash`` / ``evolve_blob`` carry the evolve-cache
        write-through payload — non-None only when a persistent cache is
        wired AND this worker evolved the theta on a cache miss.
      - ``results`` is a list aligned with ``scenario_specs``, each entry
        ``(name, status, trajectory, time_days, species, comps, rules, err)``.

    The evolve is resolved exactly once — a persistent-cache hit
    materializes the cached post-evolve state, a miss (or a wired-off
    cache) runs ``--dump-state`` once. Every scenario then runs from that
    single state via ``--initial-state``. A failed evolve fails every
    scenario for this theta; a failed scenario sim fails only itself
    (the evolve state stays reusable, so its write-through payload is
    still returned).
    """
    assert _WORKER_RUNNER is not None, "_worker_init must be called first"
    assert _WORKER_WORKDIR is not None
    _worker_t0 = time.time()
    theta_hash: str | None = None
    evolve_blob: bytes | None = None
    evolve_state_path: Path | None = None
    results: list[tuple] = []
    try:
        # Resolve the post-evolve ODE state for this theta exactly once.
        # params_hash is needed for every scenario's --initial-state call,
        # so render + hash unconditionally (cheap relative to a sim).
        xml = _WORKER_RUNNER._renderer.render(params)
        th = theta_hash_for_xml(xml)
        params_hash = wire_hash(th)
        evolve_state_path = _WORKER_WORKDIR / f"{th[:16]}.{sim_id}.evolve_state.bin"
        cached = _WORKER_EVOLVE_CACHE.get(th) if _WORKER_EVOLVE_CACHE is not None else None
        if cached is not None:
            # HIT — materialize the cached state for --initial-state.
            evolve_state_path.write_bytes(cached)
        else:
            # MISS (or no persistent cache) — evolve once via --dump-state.
            # Fusion shares this one evolve across every scenario even
            # when the persistent cache is off; the cache only adds
            # cross-run reuse on top.
            _WORKER_RUNNER.dump_evolve_state(
                params=params,
                params_hash=params_hash,
                state_out=evolve_state_path,
                workdir=_WORKER_WORKDIR,
                timeout_s=timeout_s,
            )
            if _WORKER_EVOLVE_CACHE is not None:
                theta_hash = th
                evolve_blob = evolve_state_path.read_bytes()
        try:
            for name, scen_yaml, drug_yaml in scenario_specs:
                try:
                    res: SimResult = _WORKER_RUNNER.run_one(
                        params=params,
                        t_end_days=t_end_days,
                        min_cadence_hours=min_cadence_hours,
                        workdir=_WORKER_WORKDIR,
                        timeout_s=timeout_s,
                        evolve_state_path=evolve_state_path,
                        params_hash=params_hash,
                        scenario_yaml=scen_yaml,
                        drug_metadata_yaml=drug_yaml,
                    )
                    results.append(
                        (
                            name,
                            STATUS_OK,
                            res.trajectory,
                            res.time_days,
                            res.species_names,
                            res.compartment_names,
                            res.rule_names,
                            None,
                        )
                    )
                except (QspSimError, ParamNotFoundError) as e:
                    logging.getLogger(__name__).debug(
                        "worker pid=%d FAIL sim_id=%d scenario=%s: %s",
                        os.getpid(),
                        sim_id,
                        name,
                        str(e)[:120],
                    )
                    results.append((name, STATUS_FAILED, None, None, None, None, None, str(e)))
        finally:
            evolve_state_path.unlink(missing_ok=True)
        logging.getLogger(__name__).debug(
            "worker pid=%d done fused sim_id=%d sample_index=%d (%d scenario(s), %.2fs)",
            os.getpid(),
            sim_id,
            sample_index,
            len(scenario_specs),
            time.time() - _worker_t0,
        )
        return (sim_id, theta_hash, evolve_blob, results)
    except (QspSimError, ParamNotFoundError) as e:
        # The evolve itself failed (XML render or --dump-state) — no state
        # to run any scenario from. theta_hash / evolve_blob stay None
        # (nothing reusable was produced). Mark every scenario failed.
        if evolve_state_path is not None:
            evolve_state_path.unlink(missing_ok=True)
        logging.getLogger(__name__).debug(
            "worker pid=%d FAIL fused evolve sim_id=%d: %s",
            os.getpid(),
            sim_id,
            str(e)[:120],
        )
        err = str(e)
        results = [
            (name, STATUS_FAILED, None, None, None, None, None, err)
            for (name, _s, _d) in scenario_specs
        ]
        return (sim_id, theta_hash, evolve_blob, results)


# --- Public batch runner ----------------------------------------------------


class CppBatchRunner:
    """Run many sims in parallel; emit one MATLAB-schema Parquet."""

    def __init__(
        self,
        binary_path: str | Path,
        template_path: str | Path,
        subtree: str | None = "QSP",
        default_timeout_s: float = 120.0,
        scenario_yaml: str | Path | None = None,
        drug_metadata_yaml: str | Path | None = None,
        healthy_state_yaml: str | Path | None = None,
        evolve_cache_root: str | Path | None = None,
        evolve_trajectory_dir: str | Path | None = None,
        evolve_trajectory_dt_days: float | None = None,
    ):
        # Validate eagerly so callers fail fast, before we fork workers.
        probe = CppRunner(
            binary_path=binary_path,
            template_path=template_path,
            subtree=subtree,
            default_timeout_s=default_timeout_s,
            scenario_yaml=scenario_yaml,
            drug_metadata_yaml=drug_metadata_yaml,
            healthy_state_yaml=healthy_state_yaml,
        )
        self.binary_path = probe.binary_path
        self.template_path = Path(template_path).resolve()
        self.subtree = subtree
        self.default_timeout_s = default_timeout_s
        self.scenario_yaml = probe.scenario_yaml
        self.drug_metadata_yaml = probe.drug_metadata_yaml
        self.healthy_state_yaml = probe.healthy_state_yaml
        self.parameter_names = probe.parameter_names
        # evolve_cache_root is the root of the persistent, theta-keyed
        # evolve cache (#90). Workers open the namespace under it derived
        # from (binary, healthy_state) and reuse post-evolve ODE states
        # across scenarios and runs. Only meaningful when healthy_state
        # is set — without an evolve phase there is nothing to cache.
        # When the caller asked for a cache but gave no healthy_state,
        # that's a config mismatch worth surfacing (see #34: silent
        # disable produced "0 blobs written" with no log trail);
        # otherwise the CppSimulator / submit_cpp_jobs callers that pass
        # it unconditionally don't have to check.
        if evolve_cache_root is not None and probe.healthy_state_yaml is None:
            logger.warning(
                "evolve_cache_root=%s ignored: no healthy_state_yaml " "(no evolve phase to cache)",
                evolve_cache_root,
            )
            evolve_cache_root = None
        self.evolve_cache_root = Path(evolve_cache_root).resolve() if evolve_cache_root else None
        # Burn-in trajectory dump dir (passed to qsp_sim as
        # --evolve-trajectory-out per sim). Same caveat as evolve_cache:
        # only meaningful when healthy_state_yaml is set, and disabled
        # at the worker layer for sims that go through evolve_cache
        # (no burn-in to dump in cached-state mode).
        if evolve_trajectory_dir is not None and probe.healthy_state_yaml is None:
            logger.warning(
                "evolve_trajectory_dir=%s ignored: no healthy_state_yaml "
                "(no burn-in phase to dump)",
                evolve_trajectory_dir,
            )
            evolve_trajectory_dir = None
        self.evolve_trajectory_dir = (
            Path(evolve_trajectory_dir).resolve() if evolve_trajectory_dir else None
        )
        self.evolve_trajectory_dt_days = evolve_trajectory_dt_days
        # Cache template defaults for the pool manifest (#23): only
        # sampled params land as parquet columns, and non-sampled
        # defaults live in pool_manifest.json alongside the batch
        # parquets. cal-target functions that reach for a non-sampled
        # parameter (e.g. rho_collagen) get it from the manifest
        # fallback in derive_test_stats_worker, not from a broadcast
        # parquet column.
        self.template_defaults = probe.template_defaults

    def run(
        self,
        theta_matrix: np.ndarray,
        param_names: Sequence[str],
        t_end_days: float,
        min_cadence_hours: float,
        output_path: str | Path,
        scenario: str = "default",
        seed: int = 0,
        workdir: str | Path | None = None,
        max_workers: int | None = None,
        per_sim_timeout_s: float | None = None,
        sample_indices: np.ndarray | None = None,
        evolve_trajectory_dir: str | Path | None = None,
        evolve_trajectory_dt_days: float | None = None,
    ) -> BatchResult:
        """Execute a batch and write the Parquet.

        Args:
            theta_matrix: shape (n_sims, n_params). Row i is the parameter
                vector for sim i.
            param_names: length n_params; the priors-CSV column names that
                line up with theta_matrix's columns.
            t_end_days, min_cadence_hours: passed through to qsp_sim
                (--t-end-days and --min-cadence-hours respectively;
                under v3 the output cadence is solver-native with the
                given hours value as a floor on inter-row spacing).
            output_path: Parquet destination. Parent dirs created.
            scenario, seed: metadata embedded in the Parquet filename
                schema used elsewhere in the codebase. Not written into
                the file itself (Parquet layout is scenario-agnostic).
            workdir: scratch dir for per-sim XML + binary outputs.
                Defaults to a sibling of `output_path`.
            max_workers: process-pool size. Default = CPU count.
            per_sim_timeout_s: overrides default_timeout_s for this batch.
            sample_indices: optional length-n_sims int64 array. Row i's
                global theta-pool index, written as the ``sample_index``
                Parquet column. When ``None`` (local/test use), falls
                back to ``arange(n_sims)`` — which is only valid for
                single-batch runs, not for multi-scenario alignment.

        Returns:
            BatchResult with the written path, counts, and schema info.
        """
        n_sims, n_params = theta_matrix.shape
        if len(param_names) != n_params:
            raise ValueError(
                f"theta_matrix has {n_params} columns but {len(param_names)} "
                f"param_names were given"
            )
        if sample_indices is not None and len(sample_indices) != n_sims:
            raise ValueError(
                f"sample_indices has {len(sample_indices)} entries "
                f"but theta_matrix has {n_sims} rows"
            )

        unknown = set(param_names) - self.parameter_names
        if unknown:
            raise ParamNotFoundError(
                f"{len(unknown)} priors column(s) not in XML template: " f"{sorted(unknown)[:10]}"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if workdir is None:
            workdir = output_path.parent / f".workdir_{output_path.stem}"
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        # Submit.
        logger.info(
            "Starting batch: %d sims × %d params, %d workers, scenario=%s seed=%d",
            n_sims,
            n_params,
            max_workers or os.cpu_count(),
            scenario,
            seed,
        )
        trajectories: list[np.ndarray | None] = [None] * n_sims
        time_arrays: list[np.ndarray | None] = [None] * n_sims
        statuses: list[int] = [STATUS_FAILED] * n_sims
        errors: list[str | None] = [None] * n_sims
        species_names: list[str] | None = None
        compartment_names: list[str] | None = None
        rule_names: list[str] | None = None

        # Effective evolve-trajectory dump for this batch: per-call args
        # (passed to .run()) override the __init__ defaults. ``None`` here
        # means no dump for any sim in this batch.
        effective_traj_dir = (
            evolve_trajectory_dir
            if evolve_trajectory_dir is not None
            else self.evolve_trajectory_dir
        )
        effective_traj_dt = (
            evolve_trajectory_dt_days
            if evolve_trajectory_dt_days is not None
            else self.evolve_trajectory_dt_days
        )
        if effective_traj_dir is not None and self.healthy_state_yaml is None:
            logger.warning(
                "evolve_trajectory_dir=%s ignored: no healthy_state_yaml "
                "(no burn-in phase to dump)",
                effective_traj_dir,
            )
            effective_traj_dir = None
        if effective_traj_dir is not None:
            Path(effective_traj_dir).mkdir(parents=True, exist_ok=True)
            logger.info(
                "Evolve trajectory dump: ENABLED (dir=%s, evolve_dt_days=%s)",
                effective_traj_dir,
                effective_traj_dt if effective_traj_dt else "(spec step_days)",
            )
        traj_dir_str = str(effective_traj_dir) if effective_traj_dir else None

        # Persistent evolve cache (#90): when wired, every worker reads
        # post-evolve ODE states from a theta-keyed namespace and runs a
        # cache hit via --initial-state; a miss evolves once and travels
        # back here to be written through as one QSEP shard. The cache is
        # incompatible with trajectory-dump mode (that dumps the burn-in
        # itself — there is no post-evolve snapshot to cache), so it is
        # disabled when effective_traj_dir is set. #34: announce the
        # resolved state so "did the evolve cache engage?" is answerable
        # from the SLURM stdout alone (worker-side _worker_init logs may
        # not reach stdout under the spawn start method).
        evolve_cache: EvolveCache | None = None
        if (
            self.evolve_cache_root is not None
            and self.healthy_state_yaml is not None
            and effective_traj_dir is None
        ):
            evolve_cache = EvolveCache.for_run(
                self.evolve_cache_root,
                binary_path=self.binary_path,
                healthy_state_yaml=self.healthy_state_yaml,
            )
            logger.info(
                "Evolve-cache: ENABLED (namespace=%s, dir=%s)",
                evolve_cache.namespace,
                evolve_cache.namespace_dir,
            )
        else:
            logger.info(
                "Evolve-cache: DISABLED (evolve_cache_root=%s, healthy_state_yaml=%s, "
                "trajectory_dump=%s)",
                self.evolve_cache_root,
                self.healthy_state_yaml,
                effective_traj_dir is not None,
            )
        # Workers get the cache root only when the cache is actually
        # active; None keeps _worker_init from scanning a namespace it
        # would never use (trajectory-dump mode).
        worker_evolve_cache_root = str(self.evolve_cache_root) if evolve_cache is not None else None
        # (theta_hash, QSTH blob) pairs for thetas this batch evolved on a
        # cache miss; written through as one shard after the parquet.
        evolve_miss_entries: list[tuple[str, bytes]] = []

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(
                str(self.binary_path),
                str(self.template_path),
                self.subtree,
                str(workdir),
                self.default_timeout_s,
                str(self.scenario_yaml) if self.scenario_yaml else None,
                str(self.drug_metadata_yaml) if self.drug_metadata_yaml else None,
                str(self.healthy_state_yaml) if self.healthy_state_yaml else None,
                worker_evolve_cache_root,
            ),
        ) as pool:
            # sample_indices is the canonical theta-pool index for each row;
            # pass it through to workers so per-sim trajectory dump filenames
            # use the global index (not the local sim_id) and downstream
            # assemblers can join back to theta cleanly.
            if sample_indices is not None:
                _sample_idx = np.asarray(sample_indices, dtype=np.int64)
            else:
                _sample_idx = np.arange(n_sims, dtype=np.int64)
            futures = []
            sim_id_to_sample_idx: dict = {}
            for i in range(n_sims):
                params = {name: float(theta_matrix[i, j]) for j, name in enumerate(param_names)}
                fut = pool.submit(
                    _run_one_in_worker,
                    i,
                    int(_sample_idx[i]),
                    params,
                    t_end_days,
                    min_cadence_hours,
                    per_sim_timeout_s,
                    traj_dir_str,
                    effective_traj_dt,
                )
                futures.append(fut)
                sim_id_to_sample_idx[id(fut)] = (i, int(_sample_idx[i]))

            # Per-future wait timeout. Bound on top of per_sim_timeout_s
            # so a worker process death (broken pool) doesn't hang
            # as_completed forever — without this, a SIGSEGV-killed worker
            # leaves its future unfulfilled and the main process waits on
            # the SLURM time_limit. Slack added to absorb pickling /
            # post-sim parquet-write time inside _run_one_in_worker.
            future_wait_s = float(per_sim_timeout_s or 300.0) + 60.0
            n_completed = 0
            n_timeouts = 0
            n_pool_broken = 0
            broken_pool = False
            try:
                for fut in as_completed(futures, timeout=future_wait_s * (n_sims + 1)):
                    try:
                        (
                            sim_id,
                            status,
                            traj,
                            t_days,
                            sp,
                            comps,
                            rules,
                            err,
                            theta_hash,
                            evolve_blob,
                        ) = fut.result(timeout=future_wait_s)
                    except FuturesTimeoutError:
                        # Future never completed within future_wait_s.
                        # Worker may be hung in a kernel call (e.g.
                        # blocked I/O on apopel1) or in C-level qsp_sim
                        # that ignored its own timeout_s. Mark the sim
                        # failed and continue.
                        sid_pair = sim_id_to_sample_idx.get(id(fut))
                        sid_str = (
                            f"sim_id={sid_pair[0]} sample_index={sid_pair[1]}"
                            if sid_pair
                            else "(unknown sim)"
                        )
                        logger.warning(
                            "Future-level timeout (%.0fs) on %s — worker may be hung; "
                            "marking failed and continuing.",
                            future_wait_s,
                            sid_str,
                        )
                        n_timeouts += 1
                        if sid_pair is not None:
                            statuses[sid_pair[0]] = STATUS_FUTURE_TIMEOUT
                            errors[sid_pair[0]] = f"future-level timeout after {future_wait_s:.0f}s"
                        continue
                    except BrokenExecutor as e:
                        # Worker died (SIGSEGV / OOM / unhandled C++ abort). The
                        # ProcessPoolExecutor flips to broken state; every other
                        # future is now unfulfillable. Log + bail out so the
                        # remaining futures don't sit in as_completed forever.
                        logger.error(
                            "ProcessPool broken: %s. %d/%d sims completed before "
                            "the worker died. Remaining futures will not resolve; "
                            "exiting the collection loop.",
                            e,
                            n_completed,
                            n_sims,
                        )
                        n_pool_broken = 1
                        broken_pool = True
                        break

                    n_completed += 1
                    # Capture an evolve-state blob this worker computed on
                    # a cache miss — kept even for STATUS_FAILED sims,
                    # since a completed evolve is reusable regardless of
                    # the scenario outcome.
                    if theta_hash is not None and evolve_blob is not None:
                        evolve_miss_entries.append((theta_hash, evolve_blob))
                    statuses[sim_id] = status
                    if status == STATUS_OK:
                        trajectories[sim_id] = traj
                        time_arrays[sim_id] = t_days
                        if species_names is None:
                            species_names = sp
                            compartment_names = comps or []
                            rule_names = rules or []
                    else:
                        errors[sim_id] = err
                        logger.warning("sim %d failed: %s", sim_id, err)

                    # Progress heartbeat: one line every 5% of n_sims so a
                    # hang post-sim shows up as "stopped at 80%" not silence.
                    # A logfile-friendly ASCII bar (no \r spam) stands in for
                    # a tqdm progress bar, which doesn't survive nohup well.
                    every = max(1, n_sims // 20)
                    if n_completed % every == 0 or n_completed == n_sims:
                        n_ok = sum(1 for s in statuses if s == STATUS_OK)
                        n_bad = sum(
                            1 for s in statuses if s not in (STATUS_OK, STATUS_FUTURE_TIMEOUT, None)
                        )
                        frac = n_completed / n_sims
                        filled = int(round(frac * 30))
                        bar = "#" * filled + "-" * (30 - filled)
                        logger.info(
                            "[%s] %3.0f%%  %d/%d sims (%d ok, %d failed, %d future-timeout)",
                            bar,
                            frac * 100,
                            n_completed,
                            n_sims,
                            n_ok,
                            n_bad,
                            n_timeouts,
                        )
            except FuturesTimeoutError:
                logger.error(
                    "as_completed-level timeout: %d/%d sims completed before the "
                    "outer collector wait expired. Bailing out of the collection "
                    "loop and proceeding with what we have.",
                    n_completed,
                    n_sims,
                )
                broken_pool = True

            if broken_pool or n_timeouts:
                logger.warning(
                    "Collection loop summary: completed=%d, future_timeouts=%d, "
                    "pool_broken=%d (out of %d sims).",
                    n_completed,
                    n_timeouts,
                    n_pool_broken,
                    n_sims,
                )

        n_failed = sum(1 for s in statuses if s != STATUS_OK)
        if species_names is None:
            # Every sim failed. We still need a schema to write — raise
            # rather than fabricate one, because an empty-species Parquet
            # would silently break downstream consumers expecting columns
            # by name.
            raise QspSimError(
                f"All {n_sims} sims failed; cannot infer species schema.\n"
                f"First error: {next((e for e in errors if e), '(none)')}"
            )

        # mypy/pyright: _run_one_in_worker initializes all three when
        # status is OK, so these are non-None by this point.
        assert compartment_names is not None
        assert rule_names is not None

        # Under v3 (CV_ONE_STEP), each sim has its own non-uniform time
        # vector. n_times reported here is the first successful sim's
        # row count for backward-compat metadata; the parquet writer
        # uses each sim's own time array via time_arrays.
        first_ok = next(i for i, t in enumerate(trajectories) if t is not None)
        n_times = trajectories[first_ok].shape[0]

        n_ok = sum(1 for s in statuses if s == STATUS_OK)
        logger.info(
            "Collection loop done: %d/%d ok. Writing batch parquet (this is the "
            "step that lights up shared filesystem I/O — if the task hangs after "
            "this log line, the parquet write is the suspect).",
            n_ok,
            n_sims,
        )
        _write_t0 = time.time()
        parquet_path = _write_batch_parquet(
            output_path=output_path,
            theta_matrix=theta_matrix,
            param_names=list(param_names),
            statuses=statuses,
            trajectories=trajectories,
            time_arrays=time_arrays,
            species_names=species_names,
            compartment_names=compartment_names,
            rule_names=rule_names,
            t_end_days=t_end_days,
            min_cadence_hours=min_cadence_hours,
            sample_indices=sample_indices,
        )
        logger.info(
            "Batch parquet write done in %.1fs",
            time.time() - _write_t0,
        )

        # Evolve-cache write-through (#90): pack every theta this batch
        # evolved on a miss into one QSEP shard. Written after the parquet
        # so a shard-write failure can never cost the batch its results —
        # the cache is a speedup for *later* runs, not load-bearing for
        # this one. A run where every theta hit the cache writes no shard.
        written_evolve_shard: Path | None = None
        if evolve_cache is not None and evolve_miss_entries:
            try:
                shard_writer = evolve_cache.writer()
                for theta_hash, blob in evolve_miss_entries:
                    shard_writer.add(theta_hash, blob)
                written_evolve_shard = shard_writer.flush()
                logger.info(
                    "Evolve-cache write-through: shard %s (%d/%d theta(s) evolved this batch)",
                    written_evolve_shard,
                    len(evolve_miss_entries),
                    n_sims,
                )
            except Exception as e:  # noqa: BLE001 — the cache is an optimization
                logger.error(
                    "Evolve-cache shard write failed (%s) — batch parquet is "
                    "unaffected; later runs will re-evolve these thetas.",
                    e,
                )
        elif evolve_cache is not None:
            logger.info(
                "Evolve-cache write-through: nothing to write " "(all %d theta(s) hit the cache)",
                n_sims,
            )

        # Fold accumulated shards into the manifest once enough have piled
        # up — bounds the per-task footer scan future readers pay at load
        # (#90 Phase 4). Best-effort: the compaction is atomic + idempotent,
        # and a failure only means later readers scan a few more shards.
        if evolve_cache is not None:
            try:
                evolve_cache.maybe_compact()
            except Exception as e:  # noqa: BLE001 — maintenance is best-effort
                logger.error("Evolve-cache maybe_compact failed (%s) — harmless", e)

        logger.info(
            "Batch complete: %d/%d succeeded, wrote %s (cols: %d species + %d comps + %d rules)",
            n_sims - n_failed,
            n_sims,
            parquet_path,
            len(species_names),
            len(compartment_names),
            len(rule_names),
        )
        return BatchResult(
            parquet_path=parquet_path,
            n_sims=n_sims,
            n_failed=n_failed,
            species_names=species_names,
            n_times=n_times,
            compartment_names=compartment_names,
            rule_names=rule_names,
            evolve_shard_path=written_evolve_shard,
        )

    def run_fused(
        self,
        theta_matrix: np.ndarray,
        param_names: Sequence[str],
        t_end_days: float,
        min_cadence_hours: float,
        scenarios: Sequence[FusedScenarioSpec],
        sample_indices: np.ndarray,
        workdir: str | Path | None = None,
        max_workers: int | None = None,
        per_sim_timeout_s: float | None = None,
    ) -> list[BatchResult | None]:
        """Run a fused multi-scenario batch — evolve once per theta, run
        every scenario from that state (#90 Phase 2).

        For each theta in ``theta_matrix`` the post-``evolve_to_diagnosis``
        ODE state is resolved exactly once (persistent-cache hit, or one
        ``--dump-state`` evolve), then **every** scenario in ``scenarios``
        whose ``start_index`` admits this theta is run from it via
        ``--initial-state``. One Parquet is written per scenario at its
        ``output_path`` — per-scenario pools / test_stats stay
        independently cacheable; fusion only amortizes the shared evolve.

        The runner MUST be scenario-agnostic (constructed with
        ``scenario_yaml=None``): each scenario's YAML is supplied per
        :class:`FusedScenarioSpec`, so an undosed scenario (``scenario_yaml
        =None``) is honored as such rather than inheriting a runner
        default.

        Args:
            theta_matrix: shape (n_sims, n_params); row i is one theta.
            param_names: priors-CSV column names aligned with the columns.
            t_end_days, min_cadence_hours: passed through to qsp_sim.
            scenarios: the fused scenario set. Non-empty.
            sample_indices: length-n_sims int64 array — row i's global
                theta-pool index. Required: per-scenario ``start_index``
                filtering keys off it.
            workdir: scratch dir for per-sim XML + binary outputs.
            max_workers: process-pool size. Default = CPU count.
            per_sim_timeout_s: per-simulation timeout override.

        Returns:
            A list aligned with ``scenarios``; entry k is the
            :class:`BatchResult` for ``scenarios[k]``, or ``None`` when
            that scenario had no thetas in this chunk (its ``start_index``
            excluded every row).
        """
        n_sims, n_params = theta_matrix.shape
        if len(param_names) != n_params:
            raise ValueError(
                f"theta_matrix has {n_params} columns but {len(param_names)} "
                f"param_names were given"
            )
        if len(sample_indices) != n_sims:
            raise ValueError(
                f"sample_indices has {len(sample_indices)} entries "
                f"but theta_matrix has {n_sims} rows"
            )
        if not scenarios:
            raise ValueError("run_fused requires at least one scenario")
        if self.scenario_yaml is not None:
            raise ValueError(
                "run_fused requires a scenario-agnostic CppBatchRunner "
                "(construct with scenario_yaml=None); per-scenario YAMLs "
                "are supplied via FusedScenarioSpec"
            )
        if self.healthy_state_yaml is None:
            raise ValueError(
                "run_fused requires healthy_state_yaml — fusion amortizes "
                "the evolve_to_diagnosis burn-in, and there is no burn-in "
                "without a healthy state"
            )
        unknown = set(param_names) - self.parameter_names
        if unknown:
            raise ParamNotFoundError(
                f"{len(unknown)} priors column(s) not in XML template: " f"{sorted(unknown)[:10]}"
            )
        for s in scenarios:
            if s.scenario_yaml is not None and s.drug_metadata_yaml is None:
                raise ValueError(
                    f"fused scenario {s.name!r}: scenario_yaml requires " f"drug_metadata_yaml"
                )

        sample_idx = np.asarray(sample_indices, dtype=np.int64)
        if workdir is None:
            workdir = Path(scenarios[0].output_path).parent / ".workdir_fused"
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        for s in scenarios:
            Path(s.output_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting fused batch: %d sims × %d params, %d scenario(s), %d workers",
            n_sims,
            n_params,
            len(scenarios),
            max_workers or os.cpu_count(),
        )
        for s in scenarios:
            logger.info(
                "  fused scenario %s: start_index=%d scenario_yaml=%s",
                s.name,
                s.start_index,
                s.scenario_yaml,
            )

        # Persistent evolve cache (#90 Phase 1) — composes with fusion:
        # fusion shares the evolve across scenarios within this run, the
        # cache shares it across runs. A miss travels back here for a
        # single write-through shard.
        evolve_cache: EvolveCache | None = None
        if self.evolve_cache_root is not None and self.healthy_state_yaml is not None:
            evolve_cache = EvolveCache.for_run(
                self.evolve_cache_root,
                binary_path=self.binary_path,
                healthy_state_yaml=self.healthy_state_yaml,
            )
            logger.info(
                "Fused evolve-cache: ENABLED (namespace=%s, dir=%s)",
                evolve_cache.namespace,
                evolve_cache.namespace_dir,
            )
        else:
            logger.info("Fused evolve-cache: DISABLED (evolve still shared across scenarios)")
        worker_evolve_cache_root = str(self.evolve_cache_root) if evolve_cache is not None else None

        # collected[sim_id] -> results list from _run_one_fused_in_worker.
        collected: dict[int, list[tuple]] = {}
        evolve_miss_entries: list[tuple[str, bytes]] = []

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(
                str(self.binary_path),
                str(self.template_path),
                self.subtree,
                str(workdir),
                self.default_timeout_s,
                None,  # scenario_yaml — fused workers are scenario-agnostic
                None,  # drug_metadata_yaml
                str(self.healthy_state_yaml),
                worker_evolve_cache_root,
            ),
        ) as pool:
            futures = []
            fut_meta: dict = {}
            for i in range(n_sims):
                abs_idx = int(sample_idx[i])
                specs_for_i = [
                    (
                        s.name,
                        str(s.scenario_yaml) if s.scenario_yaml is not None else None,
                        (str(s.drug_metadata_yaml) if s.drug_metadata_yaml is not None else None),
                    )
                    for s in scenarios
                    if abs_idx >= s.start_index
                ]
                if not specs_for_i:
                    # No scenario admits this theta — nothing to run. (Only
                    # reachable if the caller's fused range dips below every
                    # start_index; normally min(start_index) bounds it.)
                    continue
                params = {name: float(theta_matrix[i, j]) for j, name in enumerate(param_names)}
                fut = pool.submit(
                    _run_one_fused_in_worker,
                    i,
                    abs_idx,
                    params,
                    t_end_days,
                    min_cadence_hours,
                    per_sim_timeout_s,
                    specs_for_i,
                )
                futures.append(fut)
                fut_meta[id(fut)] = (i, abs_idx, [s[0] for s in specs_for_i])

            future_wait_s = float(per_sim_timeout_s or 300.0) + 60.0
            n_completed = 0
            n_total = len(futures)
            try:
                for fut in as_completed(futures, timeout=future_wait_s * (n_total + 1)):
                    i, abs_idx, names = fut_meta[id(fut)]
                    try:
                        sim_id, theta_hash, evolve_blob, results = fut.result(timeout=future_wait_s)
                    except FuturesTimeoutError:
                        logger.warning(
                            "Future-level timeout (%.0fs) on fused sim_id=%d "
                            "sample_index=%d — marking all scenarios failed.",
                            future_wait_s,
                            i,
                            abs_idx,
                        )
                        collected[i] = [
                            (
                                name,
                                STATUS_FUTURE_TIMEOUT,
                                None,
                                None,
                                None,
                                None,
                                None,
                                f"future-level timeout after {future_wait_s:.0f}s",
                            )
                            for name in names
                        ]
                        n_completed += 1
                        continue
                    except BrokenExecutor as e:
                        logger.error(
                            "ProcessPool broken: %s. %d/%d fused sims completed; "
                            "remaining sims will be marked failed.",
                            e,
                            n_completed,
                            n_total,
                        )
                        break
                    collected[sim_id] = results
                    if theta_hash is not None and evolve_blob is not None:
                        evolve_miss_entries.append((theta_hash, evolve_blob))
                    n_completed += 1
                    every = max(1, n_total // 20)
                    if n_completed % every == 0 or n_completed == n_total:
                        logger.info("fused: %d/%d sims completed", n_completed, n_total)
            except FuturesTimeoutError:
                logger.error(
                    "as_completed-level timeout: %d/%d fused sims completed — "
                    "proceeding with what we have.",
                    n_completed,
                    n_total,
                )

        # Schema (species / compartments / rules) is binary-determined, so
        # it is identical across scenarios — borrow it from the first
        # successful sim anywhere in the batch.
        global_species: list[str] | None = None
        global_comps: list[str] = []
        global_rules: list[str] = []
        for results in collected.values():
            for entry in results:
                _name, status, _traj, _t, sp, comp, rule, _err = entry
                if status == STATUS_OK and sp is not None:
                    global_species = sp
                    global_comps = comp or []
                    global_rules = rule or []
                    break
            if global_species is not None:
                break
        if global_species is None:
            raise QspSimError(
                f"All fused sims failed across all {len(scenarios)} scenario(s); "
                f"cannot infer species schema.\n"
                f"First error: "
                f"{next((e for r in collected.values() for (*_, e) in r if e), '(none)')}"
            )

        # Assemble one Parquet per scenario. Each scenario keeps only the
        # rows its start_index admits — a partially-cached scenario writes
        # just its deficit tail, byte-identical to a per-scenario top-up.
        written_evolve_shard: Path | None = None
        if evolve_cache is not None and evolve_miss_entries:
            try:
                shard_writer = evolve_cache.writer()
                for theta_hash, blob in evolve_miss_entries:
                    shard_writer.add(theta_hash, blob)
                written_evolve_shard = shard_writer.flush()
                logger.info(
                    "Fused evolve-cache write-through: shard %s (%d theta(s) evolved)",
                    written_evolve_shard,
                    len(evolve_miss_entries),
                )
            except Exception as e:  # noqa: BLE001 — the cache is an optimization
                logger.error("Fused evolve-cache shard write failed (%s)", e)
        elif evolve_cache is not None:
            logger.info("Fused evolve-cache write-through: nothing to write (all hit)")

        # Bound future readers' per-task footer scan (#90 Phase 4) — see
        # the matching call in run(). Best-effort, atomic + idempotent.
        if evolve_cache is not None:
            try:
                evolve_cache.maybe_compact()
            except Exception as e:  # noqa: BLE001 — maintenance is best-effort
                logger.error("Fused evolve-cache maybe_compact failed (%s) — harmless", e)

        batch_results: list[BatchResult | None] = []
        for spec in scenarios:
            included = [i for i in range(n_sims) if int(sample_idx[i]) >= spec.start_index]
            if not included:
                logger.info(
                    "fused scenario %s: no thetas in this chunk "
                    "(start_index=%d > every sample_index) — no parquet written",
                    spec.name,
                    spec.start_index,
                )
                batch_results.append(None)
                continue
            statuses: list[int] = []
            trajectories: list[np.ndarray | None] = []
            time_arrays: list[np.ndarray | None] = []
            for i in included:
                entry = next((r for r in collected.get(i, []) if r[0] == spec.name), None)
                if entry is None:
                    # Sim never returned a result for this scenario (broken
                    # pool before it ran). Mark failed.
                    statuses.append(STATUS_FAILED)
                    trajectories.append(None)
                    time_arrays.append(None)
                else:
                    _name, status, traj, t_days, *_rest = entry
                    statuses.append(status)
                    trajectories.append(traj if status == STATUS_OK else None)
                    time_arrays.append(t_days if status == STATUS_OK else None)
            out_path = Path(spec.output_path)
            parquet_path = _write_batch_parquet(
                output_path=out_path,
                theta_matrix=theta_matrix[included],
                param_names=list(param_names),
                statuses=statuses,
                trajectories=trajectories,
                time_arrays=time_arrays,
                species_names=global_species,
                compartment_names=global_comps,
                rule_names=global_rules,
                t_end_days=t_end_days,
                min_cadence_hours=min_cadence_hours,
                sample_indices=sample_idx[included],
            )
            n_failed = sum(1 for s in statuses if s != STATUS_OK)
            first_ok = next((k for k, s in enumerate(statuses) if s == STATUS_OK), None)
            n_times = trajectories[first_ok].shape[0] if first_ok is not None else 1
            logger.info(
                "fused scenario %s: %d/%d ok, wrote %s",
                spec.name,
                len(included) - n_failed,
                len(included),
                parquet_path,
            )
            batch_results.append(
                BatchResult(
                    parquet_path=parquet_path,
                    n_sims=len(included),
                    n_failed=n_failed,
                    species_names=global_species,
                    n_times=n_times,
                    compartment_names=global_comps,
                    rule_names=global_rules,
                    evolve_shard_path=written_evolve_shard,
                )
            )
        return batch_results


# --- Parquet writer ---------------------------------------------------------


def _write_batch_parquet(
    output_path: Path,
    theta_matrix: np.ndarray,
    param_names: list[str],
    statuses: list[int],
    trajectories: list[np.ndarray | None],
    time_arrays: list[np.ndarray | None],
    species_names: list[str],
    compartment_names: list[str],
    rule_names: list[str],
    t_end_days: float,
    min_cadence_hours: float,
    sample_indices: np.ndarray | None = None,
) -> Path:
    """Build one pyarrow Table matching MATLAB's Parquet schema, write it.

    Trajectory columns are laid out in the order
    ``[species..., compartments..., rules...]`` (matching the binary
    body layout). Each is emitted as a bare column name so downstream
    code reads them via ``species_dict[name]`` uniformly.

    Under qsp-codegen v3 (CV_ONE_STEP), each successful sim has its own
    non-uniform time vector — the writer threads these through as
    ``time_arrays`` rather than reconstructing one shared axis from a
    fixed dt. Failed sims get a NaN-padded single-row time/state.

    Only **sampled** model parameters land as ``param:<name>`` columns
    (one per entry in ``param_names``). Non-sampled template defaults
    live in the pool's sidecar ``pool_manifest.json`` and are injected
    by readers when a calibration-target function asks for a parameter
    that isn't in the sampled set. See #23 — the prior behavior
    broadcast every template default into every row, tripling parquet
    width and creating a recurring source of "which set do I want?"
    ambiguity between callers.
    """
    n_sims = len(statuses)

    # Per-sim time vectors (variable length under v3). Failed sims get
    # a single-NaN placeholder so the parquet column shape is well-defined.
    time_lists: list[list[float]] = []
    nan_rows: list[np.ndarray] = []
    for traj, t_days in zip(trajectories, time_arrays):
        if traj is None or t_days is None:
            time_lists.append([float("nan")])
            nan_rows.append(np.array([np.nan], dtype=np.float64))
        else:
            time_lists.append(np.asarray(t_days, dtype=np.float64).tolist())
            nan_rows.append(np.full(traj.shape[0], np.nan, dtype=np.float64))

    # sample_index is the GLOBAL theta-pool position (same across all
    # scenarios for a given patient/draw); simulation_id is the LOCAL
    # position within this batch. Downstream multi-scenario alignment
    # (sbi_runner.py) intersects on sample_index. When the caller doesn't
    # provide one — e.g. local runs or tests — fall back to arange, which
    # is only correct for a single-batch, single-scenario pool.
    if sample_indices is None:
        sample_indices_arr = np.arange(n_sims, dtype=np.int64)
    else:
        sample_indices_arr = np.asarray(sample_indices, dtype=np.int64)
    columns: dict[str, pa.Array] = {
        "sample_index": pa.array(sample_indices_arr),
        "simulation_id": pa.array(np.arange(n_sims, dtype=np.int64)),
        "status": pa.array(np.asarray(statuses, dtype=np.int64)),
        "time": pa.array(time_lists, type=pa.list_(pa.float64())),
    }
    for j, name in enumerate(param_names):
        columns[f"param:{name}"] = pa.array(theta_matrix[:, j].astype(np.float64))

    # Trajectory columns in the same order they appear in the v2 binary:
    # species first, then compartments, then assignment rules. Indexing
    # is positional — column k corresponds to trajectory[:, k].
    all_trajectory_names = list(species_names) + list(compartment_names) + list(rule_names)
    for k, name in enumerate(all_trajectory_names):
        per_sim_series: list[list[float]] = []
        for sim_idx, traj in enumerate(trajectories):
            if traj is None:
                per_sim_series.append(nan_rows[sim_idx].tolist())
            else:
                per_sim_series.append(traj[:, k].tolist())
        columns[name] = pa.array(per_sim_series, type=pa.list_(pa.float64()))

    table = pa.Table.from_pydict(columns)
    pq.write_table(table, str(output_path), compression="snappy")
    return output_path


def batch_filename(
    batch_index: int,
    n_sims: int,
    scenario: str,
    seed: int,
    timestamp: datetime | None = None,
) -> str:
    """Produce the filename the existing SimulationPoolManager expects."""
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"batch_{batch_index:03d}_{ts}_{n_sims}sims_seed{seed}.parquet"
