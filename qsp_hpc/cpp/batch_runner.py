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

from qsp_hpc.cpp.evolve_cache import CppEvolveCache, theta_hash_for_xml, wire_hash
from qsp_hpc.cpp.evolve_pack import EvolveStatePackReader, EvolveStatePackWriter
from qsp_hpc.cpp.param_xml import ParamNotFoundError
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
    # Per-task QSEP evolve-state pack written when evolve_pack emission is
    # active (#86). None when emission was off or no evolve states were
    # captured (e.g. trajectory-dump mode, or no healthy_state_yaml).
    evolve_pack_path: Path | None = None


# --- Worker (module-level so ProcessPoolExecutor can pickle it) -------------

# Each worker process holds one CppRunner for its lifetime; we initialize it
# in an executor `initializer` and stash it as a module global. The hot path
# then avoids repaying ParamXMLRenderer parse + binary-path validation on
# every sim.
_WORKER_RUNNER: CppRunner | None = None
_WORKER_WORKDIR: Path | None = None
# Optional: when the batch was constructed with evolve_cache_root, workers
# hold a CppEvolveCache keyed on the same healthy_state_yaml / qsp_sim
# binary. Workers lookup via the cache per-sim; the fcntl lock inside
# get_or_build serializes builds across workers hitting the same theta.
_WORKER_EVOLVE_CACHE: CppEvolveCache | None = None
# Optional: when the batch was constructed with evolve_pack_read_path,
# workers hold a read-only EvolveStatePackReader over a prior scenario's
# per-task QSEP pack. Per-sim, the worker looks up this theta's evolve
# state and runs the scenario from it via --initial-state; a miss falls
# back to a normal full evolve+scenario run (#86).
_WORKER_EVOLVE_PACK_READER: EvolveStatePackReader | None = None


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
    evolve_pack_read_path: str | None = None,
) -> None:
    global _WORKER_RUNNER, _WORKER_WORKDIR, _WORKER_EVOLVE_CACHE
    global _WORKER_EVOLVE_PACK_READER
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
        _WORKER_EVOLVE_CACHE = CppEvolveCache(
            cache_root=evolve_cache_root,
            renderer=_WORKER_RUNNER._renderer,
            runner=_WORKER_RUNNER,
        )
        logger.info(
            "worker %d: evolve cache ENABLED, cache_dir=%s",
            os.getpid(),
            _WORKER_EVOLVE_CACHE.cache_dir,
        )
    else:
        _WORKER_EVOLVE_CACHE = None
        logger.info(
            "worker %d: evolve cache DISABLED " "(evolve_cache_root=%s, healthy_state_yaml=%s)",
            os.getpid(),
            evolve_cache_root,
            healthy_state_yaml,
        )

    if evolve_pack_read_path is not None:
        _WORKER_EVOLVE_PACK_READER = EvolveStatePackReader(evolve_pack_read_path)
        logger.info(
            "worker %d: evolve-pack consume ENABLED, %d states from %s",
            os.getpid(),
            len(_WORKER_EVOLVE_PACK_READER),
            evolve_pack_read_path,
        )
    else:
        _WORKER_EVOLVE_PACK_READER = None


def _run_one_in_worker(
    sim_id: int,
    sample_index: int,
    params: dict[str, float],
    t_end_days: float,
    min_cadence_hours: float,
    timeout_s: float | None,
    evolve_trajectory_dir: str | None = None,
    evolve_trajectory_dt_days: float | None = None,
    emit_evolve_pack: bool = False,
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

    ``emit_evolve_pack``: when True (and there is an evolve phase and no
    trajectory dump), the worker evolves via ``--dump-state``, captures
    the QSTH blob, and runs the scenario from it via ``--initial-state``.
    ``theta_hash`` + ``evolve_blob`` carry that blob back to the parent
    for the per-task QSEP pack (#86); both are None when emission is off.
    The blob is returned even when the scenario sim fails — a completed
    evolve is reusable regardless of the scenario outcome.
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
    # theta_hash / evolve_blob carry the per-task pack payload (#86) back
    # to the parent. Bound before the try so the except path can still
    # return a blob captured by a successful evolve when the scenario sim
    # afterwards fails — a completed evolve is reusable either way.
    theta_hash: str | None = None
    evolve_blob: bytes | None = None
    try:
        # Resolve the post-evolve ODE state for this theta. The evolve
        # modes below are all skipped when ``evolve_trajectory_dir`` is
        # set: cached-state mode loads a post-evolve snapshot, which by
        # design skips the burn-in entirely, and honoring it would
        # silently produce zero trajectory binaries (#69's worker-side
        # "trajectories take priority over cache" half).
        #
        #   - emit_evolve_pack: evolve via --dump-state, capture the QSTH
        #     blob for the per-task QSEP pack, then run the scenario from
        #     it via --initial-state. No shared store — the parent packs
        #     the blobs, so there is no concurrent-writer NFS hazard (#86).
        #   - evolve cache (LMDB): get_or_build amortizes the evolve
        #     across scenarios sharing a theta.
        #   - neither: a plain full evolve+scenario run.
        evolve_state_path: Path | None = None
        params_hash: str | None = None
        # emit_evolve_pack is checked first so the healthy_state_yaml
        # attribute access short-circuits away in the non-emission path
        # (the worker runner is always a real CppRunner there, but test
        # stubs need not carry every attribute).
        if (
            emit_evolve_pack
            and evolve_trajectory_dir is None
            and _WORKER_RUNNER.healthy_state_yaml is not None
        ):
            xml = _WORKER_RUNNER._renderer.render(params)
            theta_hash = theta_hash_for_xml(xml)
            params_hash = wire_hash(theta_hash)
            evolve_state_path = _WORKER_WORKDIR / f"{theta_hash[:16]}.evolve_state.bin"
            _WORKER_RUNNER.dump_evolve_state(
                params=params,
                params_hash=params_hash,
                state_out=evolve_state_path,
                workdir=_WORKER_WORKDIR,
                timeout_s=timeout_s,
            )
            evolve_blob = evolve_state_path.read_bytes()
        elif _WORKER_EVOLVE_PACK_READER is not None and evolve_trajectory_dir is None:
            # Consume: look this theta's evolve state up in a prior
            # scenario's QSEP pack and run the scenario from it via
            # --initial-state. A miss leaves evolve_state_path None → a
            # normal full evolve+scenario fallback runs (correct, just
            # not amortized — e.g. an upstream task failed to pack this
            # chunk). theta_hash/evolve_blob stay None: consume emits
            # nothing back to the parent.
            xml = _WORKER_RUNNER._renderer.render(params)
            _th = theta_hash_for_xml(xml)
            blob = _WORKER_EVOLVE_PACK_READER.get(_th)
            if blob is not None:
                params_hash = wire_hash(_th)
                evolve_state_path = _WORKER_WORKDIR / f"{_th[:16]}.evolve_state.bin"
                evolve_state_path.write_bytes(blob)
        elif _WORKER_EVOLVE_CACHE is not None and evolve_trajectory_dir is None:
            evolve_state_path, params_hash = _WORKER_EVOLVE_CACHE.get_or_build(
                params,
                workdir=_WORKER_WORKDIR,
                timeout_s=timeout_s,
            )
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
            # The blob is materialized fresh per call from LMDB; drop it
            # so the worker workdir doesn't accumulate one file per
            # distinct theta this worker has seen.
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
        evolve_pack_path: str | Path | None = None,
        evolve_pack_read_path: str | Path | None = None,
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
        # evolve_cache_root is propagated to workers. Only meaningful when
        # healthy_state_yaml is set — without an evolve phase there is
        # nothing to cache. When the caller explicitly asked for a cache
        # but didn't provide a healthy_state, that's a config mismatch
        # worth surfacing (see #34: silent disable produced "0 blobs
        # written" with no log trail); otherwise the CppSimulator /
        # submit_cpp_jobs callers that pass it unconditionally don't have
        # to check.
        if evolve_cache_root is not None and probe.healthy_state_yaml is None:
            logger.warning(
                "evolve_cache_root=%s ignored: no healthy_state_yaml " "(no evolve phase to cache)",
                evolve_cache_root,
            )
            evolve_cache_root = None
        # evolve_pack_path: per-task QSEP pack emission (#86). When set,
        # every sim evolves via --dump-state, the QSTH blob is captured,
        # and run() writes one pack for the whole batch. Supersedes the
        # LMDB cache for this batch — emission is the NFS-safe successor,
        # so a caller passing both gets emission (the LMDB write path is
        # the thing being retired). Like the cache, it needs an evolve
        # phase to have anything to pack.
        if evolve_pack_path is not None and probe.healthy_state_yaml is None:
            logger.warning(
                "evolve_pack_path=%s ignored: no healthy_state_yaml " "(no evolve phase to pack)",
                evolve_pack_path,
            )
            evolve_pack_path = None
        if evolve_pack_path is not None and evolve_cache_root is not None:
            logger.info(
                "evolve_pack_path set — superseding evolve_cache_root=%s for this "
                "batch (per-task pack emission replaces the shared LMDB cache)",
                evolve_cache_root,
            )
            evolve_cache_root = None
        self.evolve_pack_path = Path(evolve_pack_path).resolve() if evolve_pack_path else None
        # evolve_pack_read_path: consume a prior scenario's QSEP pack (#86).
        # Mutually exclusive with evolve_pack_path — a batch emits OR
        # consumes, never both. Needs a healthy_state_yaml so a pack miss
        # has a full-evolve fallback.
        if evolve_pack_path is not None and evolve_pack_read_path is not None:
            raise ValueError(
                "evolve_pack_path (emit) and evolve_pack_read_path (consume) are "
                "mutually exclusive — a batch does one or the other"
            )
        if evolve_pack_read_path is not None and probe.healthy_state_yaml is None:
            logger.warning(
                "evolve_pack_read_path=%s ignored: no healthy_state_yaml "
                "(a pack miss would have no evolve fallback)",
                evolve_pack_read_path,
            )
            evolve_pack_read_path = None
        self.evolve_pack_read_path = (
            Path(evolve_pack_read_path).resolve() if evolve_pack_read_path else None
        )
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
        evolve_pack_path: str | Path | None = None,
        evolve_pack_read_path: str | Path | None = None,
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
        # #34: announce resolved cache state so "did the evolve cache
        # engage?" is answerable from the SLURM stdout alone. Worker-side
        # wiring runs inside _worker_init — those logs depend on the child
        # process having the qsp_hpc logger configured, which isn't
        # guaranteed under the spawn start method.
        if self.evolve_cache_root is not None:
            logger.info(
                "Evolve-cache: ENABLED (root=%s, healthy_state_yaml=%s)",
                self.evolve_cache_root,
                self.healthy_state_yaml,
            )
        else:
            logger.info(
                "Evolve-cache: DISABLED (healthy_state_yaml=%s)",
                self.healthy_state_yaml,
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

        # Effective per-task evolve-pack emission (#86): per-call arg
        # overrides the __init__ default. Emission needs an evolve phase
        # and is incompatible with trajectory-dump mode (that dumps the
        # burn-in itself; there is no post-evolve snapshot to pack).
        effective_evolve_pack_path = (
            Path(evolve_pack_path) if evolve_pack_path is not None else self.evolve_pack_path
        )
        emit_evolve_pack = (
            effective_evolve_pack_path is not None
            and self.healthy_state_yaml is not None
            and effective_traj_dir is None
        )
        if effective_evolve_pack_path is not None and not emit_evolve_pack:
            logger.warning(
                "evolve_pack_path=%s not emitted: %s",
                effective_evolve_pack_path,
                (
                    "no healthy_state_yaml (no evolve phase)"
                    if self.healthy_state_yaml is None
                    else "trajectory-dump mode active (burn-in is dumped, not packed)"
                ),
            )
        if emit_evolve_pack:
            logger.info("Evolve-pack emission: ENABLED (path=%s)", effective_evolve_pack_path)
        # Collected (theta_hash, QSTH blob) pairs from workers; packed once
        # after the collection loop.
        evolve_pack_entries: list[tuple[str, bytes]] = []

        # Effective per-task evolve-pack consumption (#86): workers read a
        # prior scenario's QSEP pack and run from it via --initial-state,
        # falling back to a full evolve on a miss. Mutually exclusive with
        # emission; skipped under trajectory-dump mode.
        effective_evolve_pack_read_path = (
            Path(evolve_pack_read_path)
            if evolve_pack_read_path is not None
            else self.evolve_pack_read_path
        )
        consume_evolve_pack = (
            effective_evolve_pack_read_path is not None
            and self.healthy_state_yaml is not None
            and effective_traj_dir is None
        )
        if emit_evolve_pack and consume_evolve_pack:
            raise ValueError(
                "evolve-pack emit and consume are mutually exclusive — a batch "
                "does one or the other"
            )
        if consume_evolve_pack:
            logger.info("Evolve-pack consume: ENABLED (read=%s)", effective_evolve_pack_read_path)

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
                str(self.evolve_cache_root) if self.evolve_cache_root else None,
                (str(effective_evolve_pack_read_path) if consume_evolve_pack else None),
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
                    emit_evolve_pack,
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
                    # Capture the evolve-state blob for the per-task pack —
                    # kept even for STATUS_FAILED sims, since a completed
                    # evolve is reusable regardless of the scenario outcome.
                    if theta_hash is not None and evolve_blob is not None:
                        evolve_pack_entries.append((theta_hash, evolve_blob))
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

        # Per-task evolve-state pack (#86). Written after the parquet so a
        # pack failure can never cost the batch its results — emission is a
        # speedup for *later* scenarios, not load-bearing for this one.
        written_evolve_pack: Path | None = None
        if emit_evolve_pack and evolve_pack_entries:
            try:
                writer = EvolveStatePackWriter()
                for theta_hash, blob in evolve_pack_entries:
                    writer.add(theta_hash, blob)
                written_evolve_pack = writer.write(effective_evolve_pack_path)
                logger.info(
                    "Evolve-pack written: %s (%d/%d evolve states captured)",
                    written_evolve_pack,
                    len(evolve_pack_entries),
                    n_sims,
                )
            except Exception as e:  # noqa: BLE001 — pack is an optimization
                logger.error(
                    "Evolve-pack write failed (%s) — batch parquet is unaffected; "
                    "later scenarios will re-evolve instead of reusing the pack.",
                    e,
                )
        elif emit_evolve_pack:
            logger.warning("Evolve-pack emission was ON but 0 evolve states were captured.")

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
            evolve_pack_path=written_evolve_pack,
        )


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
