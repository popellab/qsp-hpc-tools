"""Run a batch of C++ QSP simulations and write long-form Parquet.

This layer sits between the single-sim CppRunner (M3) and the top-level
CppSimulator (M5). Given a `theta_matrix` of parameter samples plus the
priors column names, it fans out over a ProcessPoolExecutor, collects
trajectories, and writes two Parquet files per chunk:

  ``{stem}.trajectory.parquet`` — long-form trajectory rows::
      sample_index:  int64
      time:          float64
      species:       dictionary<string>   # categorical, includes
                                          # species + compartments + rules
      value:         float64

  ``{stem}.params.parquet`` — sidecar with one row per sample_index::
      sample_index:  int64
      simulation_id: int64    # zero-indexed within this batch
      status:        int64    # 0 = success, 1 = qsp_sim failure
      param:<name>:  float64  # one column per priors-CSV param

The compartment and rule entries appear in the ``species`` dict alongside
true species so calibration-target functions can read them via
``species_dict[name]`` uniformly. Failed sims contribute zero trajectory
rows; their status survives in the params sidecar.

The long-form layout halves on-disk size vs the prior wide-form
list-typed schema (zstd-compressed categorical species column) and
enables column projection at read time via predicate pushdown on
``species`` — see ``notes/architecture/local_observable_eval_plan.md``
Open Q5 for the 2026-05-07 benchmark.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qsp_hpc.cpp.evolve_cache import CppEvolveCache
from qsp_hpc.cpp.param_xml import ParamNotFoundError
from qsp_hpc.cpp.runner import CppRunner, QspSimError, SimResult

logger = logging.getLogger(__name__)


STATUS_OK = 0
STATUS_FAILED = 1


POOL_MANIFEST_FILENAME = "pool_manifest.json"
POOL_MANIFEST_SCHEMA = "subpool-v2"
SUBPOOL_KINDS: tuple[str, ...] = ("training", "ppc")


def subpool_dir(pool_dir: Path | str, kind: str) -> Path:
    """Return ``{pool_dir}/{kind}`` after validating ``kind``.

    Layer-3 sub-pool layout (D1 in
    ``notes/architecture/local_observable_eval_plan.md``): each
    ``pool_id`` carries a ``training/`` and a ``ppc/`` sub-pool, each
    with its own manifest.
    """
    if kind not in SUBPOOL_KINDS:
        raise ValueError(f"subpool kind must be one of {SUBPOOL_KINDS}; got {kind!r}")
    return Path(pool_dir) / kind


def write_pool_manifest(
    pool_dir: Path | str,
    kind: str,
    template_defaults: dict[str, float],
    sampled_params: Sequence[str],
) -> Path:
    """Write ``{pool_dir}/{kind}/pool_manifest.json`` (subpool-v2).

    The sub-pool manifest is the single source of truth for both
    ``HPCSession.reserve_sample_index_range``-allocated ``reservations``
    and the SLURM-side ``template_defaults`` + ``sampled_params`` snapshot
    a downstream reader needs to resolve unsampled ``param:*`` columns.
    Concurrent SLURM array tasks all race to write this on first run of
    a fresh pool; the merge below is idempotent so second+ writers no-op.

    Layout::

        {
            "schema_version": "subpool-v2",
            "kind": "training",                              # or "ppc"
            "reservations": [{"start": 0, "end": N, "ts": ...}, ...],
            "template_defaults": {"A": 1.5, "B": 2.0, ...},  # EVERY model param
            "sampled_params": ["A", "C", ...]                # subset varied
        }

    Merge semantics: if the manifest already exists with
    ``reservations`` populated by ``HPCSession`` (training-side
    SLURM-tasks land *after* reservation broadcast), this call adds the
    ``template_defaults`` and ``sampled_params`` keys without touching
    ``reservations``. If ``template_defaults`` is already present, the
    call is a no-op (first writer wins; content is identical across
    SLURM array tasks).
    """
    sub_dir = subpool_dir(pool_dir, kind)
    sub_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sub_dir / POOL_MANIFEST_FILENAME

    if manifest_path.exists():
        try:
            with open(manifest_path) as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            payload = {}
        if "template_defaults" in payload:
            return manifest_path
    else:
        payload = {}

    payload.setdefault("schema_version", POOL_MANIFEST_SCHEMA)
    payload.setdefault("kind", kind)
    payload.setdefault("reservations", [])
    payload["template_defaults"] = {str(k): float(v) for k, v in template_defaults.items()}
    payload["sampled_params"] = list(sampled_params)

    # Atomic write so a partial read never happens: a parallel cal-target
    # evaluator could race a first-run write otherwise. Every SLURM array
    # task races to write this on first run of a fresh pool, so the tmp
    # filename MUST be unique per-process — a single shared
    # "pool_manifest.json.tmp" lets task 0 rename its tmp to the final
    # path and leaves task 1 without a tmp to rename (FileNotFoundError,
    # hit on SBI smoke 2026-04-17). os.getpid() is enough even across
    # different hosts because two tasks writing simultaneously into the
    # same scratch dir with the same PID would need the same PID to be
    # reused within milliseconds — negligible.
    tmp_path = manifest_path.with_suffix(f".json.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        tmp_path.replace(manifest_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return manifest_path


def load_pool_manifest(pool_dir: Path | str, kind: str) -> dict | None:
    """Load ``{pool_dir}/{kind}/pool_manifest.json``, or None if absent.

    Pre-subpool-v2 pools (legacy ``{pool_dir}/pool_manifest.json`` at the
    pool root, schema ``thin-v1``) are not migrated — the local-eval
    rollout (D6) is a hard cutover; older pools must be re-simulated.
    """
    manifest_path = subpool_dir(pool_dir, kind) / POOL_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    with open(manifest_path) as fh:
        return json.load(fh)


@dataclass
class BatchResult:
    """Summary returned after a batch completes.

    The two paths point at the long-form trajectory parquet and the
    params sidecar respectively. ``parquet_path`` is retained as an
    alias for ``trajectory_path`` to keep test scaffolding migrating
    one piece at a time, but new callers should prefer the explicit
    field names.
    """

    trajectory_path: Path
    params_path: Path
    n_sims: int
    n_failed: int
    species_names: list[str]
    n_times: int
    compartment_names: list[str] | None = None
    rule_names: list[str] | None = None

    @property
    def parquet_path(self) -> Path:  # legacy alias — prefer trajectory_path
        return self.trajectory_path


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
]:
    """Return (sim_id, status, trajectory, species, comps, rules, err).

    ``evolve_trajectory_dir`` / ``evolve_trajectory_dt_days`` are the
    *effective* values resolved by the caller (CppBatchRunner.run()) —
    per-call overrides take priority over __init__ defaults. ``None``
    here means "no dump for this sim".
    """
    assert _WORKER_RUNNER is not None, "_worker_init must be called first"
    assert _WORKER_WORKDIR is not None
    try:
        # If the evolve cache is active, materialize the post-evolve state
        # for this theta (build-if-missing) and pass it to qsp_sim via
        # --initial-state. Scenarios sharing a theta amortize the evolve
        # across all runs; the first one pays ~0.5s, the rest ~0ms.
        #
        # Skip the cache when ``evolve_trajectory_dir`` is set: cached-state
        # mode loads a post-evolve snapshot, which by design skips the
        # burn-in phase entirely. Honoring the cache here would silently
        # produce zero trajectory binaries even when the caller asked for
        # them (caught locally on a 50-sim smoke; this is the worker-side
        # half of #69's "trajectories take priority over cache" semantics).
        evolve_state_path: Path | None = None
        params_hash: str | None = None
        if _WORKER_EVOLVE_CACHE is not None and evolve_trajectory_dir is None:
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
        return (
            sim_id,
            STATUS_OK,
            result.trajectory,
            result.time_days,
            result.species_names,
            result.compartment_names,
            result.rule_names,
            None,
        )
    except (QspSimError, ParamNotFoundError) as e:
        return sim_id, STATUS_FAILED, None, None, None, None, None, str(e)


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
            for i in range(n_sims):
                params = {name: float(theta_matrix[i, j]) for j, name in enumerate(param_names)}
                futures.append(
                    pool.submit(
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
                )
            for fut in as_completed(futures):
                sim_id, status, traj, t_days, sp, comps, rules, err = fut.result()
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

        trajectory_path, params_path = _write_batch_parquet(
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
            "Batch complete: %d/%d succeeded, wrote %s + %s "
            "(species in dict: %d sp + %d comps + %d rules)",
            n_sims - n_failed,
            n_sims,
            trajectory_path.name,
            params_path.name,
            len(species_names),
            len(compartment_names),
            len(rule_names),
        )
        return BatchResult(
            trajectory_path=trajectory_path,
            params_path=params_path,
            n_sims=n_sims,
            n_failed=n_failed,
            species_names=species_names,
            n_times=n_times,
            compartment_names=compartment_names,
            rule_names=rule_names,
        )


# --- Parquet writer ---------------------------------------------------------


def _derive_chunk_paths(output_path: Path) -> tuple[Path, Path]:
    """Map a legacy single-parquet ``output_path`` to the long-form pair.

    ``chunk_NNN.parquet`` → (``chunk_NNN.trajectory.parquet``,
    ``chunk_NNN.params.parquet``). A path without a ``.parquet`` suffix
    is treated as the stem.
    """
    stem = output_path.name
    if stem.endswith(".parquet"):
        stem = stem[: -len(".parquet")]
    return (
        output_path.parent / f"{stem}.trajectory.parquet",
        output_path.parent / f"{stem}.params.parquet",
    )


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
) -> tuple[Path, Path]:
    """Emit the long-form trajectory + params sidecar pair for this batch.

    Returns ``(trajectory_path, params_path)``. Trajectory rows are
    keyed by ``sample_index``; the species column is dictionary-encoded
    and combines true species, compartments, and assignment rules
    (positionally aligned with the binary body layout's column order:
    species first, then compartments, then rules). Failed sims
    contribute zero trajectory rows; the params sidecar always carries
    one row per sim so consumers can look up ``status`` / ``param:*``
    by ``sample_index``.

    Under qsp-codegen v3 (CV_ONE_STEP), each successful sim has its own
    non-uniform time vector. The long-form layout absorbs heterogeneous
    per-sim time grids natively — no NaN padding needed — and carries
    the per-sim time vector verbatim into the ``time`` column.

    Only **sampled** model parameters appear as ``param:<name>`` columns
    in the params sidecar. Non-sampled template defaults live in the
    pool's ``pool_manifest.json`` and are injected by readers when a
    calibration-target function asks for a parameter outside the
    sampled set (issue #23 convention, preserved).
    """
    n_sims = len(statuses)
    trajectory_path, params_path = _derive_chunk_paths(output_path)

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

    # --- Params sidecar -----------------------------------------------------
    params_columns: dict[str, pa.Array] = {
        "sample_index": pa.array(sample_indices_arr),
        "simulation_id": pa.array(np.arange(n_sims, dtype=np.int64)),
        "status": pa.array(np.asarray(statuses, dtype=np.int64)),
    }
    for j, name in enumerate(param_names):
        params_columns[f"param:{name}"] = pa.array(theta_matrix[:, j].astype(np.float64))
    params_table = pa.Table.from_pydict(params_columns)
    pq.write_table(params_table, str(params_path), compression="zstd")

    # --- Long-form trajectory ----------------------------------------------
    # Column dictionary order matches the binary body layout.
    all_trajectory_names = list(species_names) + list(compartment_names) + list(rule_names)
    n_cols = len(all_trajectory_names)

    # Allocate one block per (sim, species_col) pair, then concatenate.
    # Failed sims contribute nothing here — their status is in the sidecar.
    sample_idx_blocks: list[np.ndarray] = []
    time_blocks: list[np.ndarray] = []
    species_idx_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []

    for i in range(n_sims):
        traj = trajectories[i]
        t_days = time_arrays[i]
        if traj is None or t_days is None or statuses[i] != STATUS_OK:
            continue
        n_t = traj.shape[0]
        if n_t == 0:
            continue
        sidx = int(sample_indices_arr[i])
        t_arr = np.asarray(t_days, dtype=np.float64)
        for k in range(n_cols):
            sample_idx_blocks.append(np.full(n_t, sidx, dtype=np.int64))
            time_blocks.append(t_arr)
            species_idx_blocks.append(np.full(n_t, k, dtype=np.int32))
            value_blocks.append(traj[:, k].astype(np.float64, copy=False))

    if sample_idx_blocks:
        sample_idx_flat = np.concatenate(sample_idx_blocks)
        time_flat = np.concatenate(time_blocks)
        species_idx_flat = np.concatenate(species_idx_blocks)
        value_flat = np.concatenate(value_blocks)
    else:
        # All-failed batch — emit an empty trajectory parquet with the
        # right schema. Consumers must tolerate this (it's already the
        # "raise QspSimError" path above for fully-failed batches).
        sample_idx_flat = np.empty(0, dtype=np.int64)
        time_flat = np.empty(0, dtype=np.float64)
        species_idx_flat = np.empty(0, dtype=np.int32)
        value_flat = np.empty(0, dtype=np.float64)

    species_dict = pa.DictionaryArray.from_arrays(
        pa.array(species_idx_flat, type=pa.int32()),
        pa.array(all_trajectory_names, type=pa.string()),
    )

    trajectory_table = pa.table(
        {
            "sample_index": pa.array(sample_idx_flat, type=pa.int64()),
            "time": pa.array(time_flat, type=pa.float64()),
            "species": species_dict,
            "value": pa.array(value_flat, type=pa.float64()),
        }
    )
    pq.write_table(trajectory_table, str(trajectory_path), compression="zstd")

    return trajectory_path, params_path


def load_long_form_chunk_as_wide(
    trajectory_path: Path,
    params_path: Path,
) -> "pd.DataFrame":  # noqa: F821 — pandas imported lazily
    """Inverse of :func:`_write_batch_parquet`: read the long-form pair and
    return the wide-form ``sim_df`` schema that
    :func:`qsp_hpc.batch.derive_test_stats_worker.compute_test_statistics_batch`
    expects.

    Output columns: ``sample_index``, ``simulation_id``, ``status``, ``time``
    (list[float] per row), one column per species/compartment/rule (also
    list[float] per row), and ``param:<name>`` columns from the sidecar.

    Bridge code: this exists so the local PPC path
    (``CppSimulator._simulate_with_parameters_local``) and any other
    consumer that hasn't been migrated to the long-form view can keep
    working through the Layer 4 transition. Once observable evaluation
    moves onto the runner via ``evaluate_targets_to_x``, this helper
    can be deleted alongside ``derive_test_stats_worker``.

    Failed sims (status != 0) keep their sidecar row but appear with empty
    ``time`` and per-species lists, matching the legacy convention.
    """
    import pandas as pd

    params_df = pd.read_parquet(params_path)
    traj_table = pq.read_table(trajectory_path)

    # Decode the dictionary-encoded `species` column once. The value
    # column is plain float64; sample_index/time too.
    species_arr = traj_table.column("species").to_pylist()
    sample_idx_arr = traj_table.column("sample_index").to_numpy()
    time_arr = traj_table.column("time").to_numpy()
    value_arr = traj_table.column("value").to_numpy()

    species_names = [s for s in dict.fromkeys(species_arr).keys()]

    # Group rows by sample_index → species → ordered (time, value).
    # Within a chunk all (sample, species) entries share the same
    # per-sim time vector (the writer iterates n_cols against one
    # t_arr per sim), so we extract one time vector per sample_index.
    per_sample: dict[int, dict[str, list[float]]] = {}
    per_sample_time: dict[int, list[float]] = {}

    # Single pass: bucket into per-sample dicts. This is O(N) and
    # avoids the O(N²) sort-then-groupby pandas path.
    for sidx_raw, sp, t, v in zip(sample_idx_arr, species_arr, time_arr, value_arr):
        sidx = int(sidx_raw)
        bucket = per_sample.setdefault(sidx, {})
        lst = bucket.setdefault(sp, [])
        lst.append(float(v))
        if sp == species_names[0]:
            per_sample_time.setdefault(sidx, []).append(float(t))

    # Build wide-form rows in params_df order.
    out: dict[str, list] = {
        "sample_index": [],
        "simulation_id": [],
        "status": [],
        "time": [],
    }
    for sp in species_names:
        out[sp] = []
    param_cols = [c for c in params_df.columns if c.startswith("param:")]
    for c in param_cols:
        out[c] = []

    for _, prow in params_df.iterrows():
        sidx = int(prow["sample_index"])
        out["sample_index"].append(sidx)
        out["simulation_id"].append(
            int(prow["simulation_id"]) if "simulation_id" in params_df.columns else sidx
        )
        out["status"].append(int(prow["status"]) if "status" in params_df.columns else 0)
        sp_data = per_sample.get(sidx, {})
        out["time"].append(per_sample_time.get(sidx, []))
        for sp in species_names:
            out[sp].append(sp_data.get(sp, []))
        for c in param_cols:
            out[c].append(float(prow[c]))

    return pd.DataFrame(out)


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
