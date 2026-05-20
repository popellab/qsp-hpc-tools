"""C++ QSP simulation worker for HPC (SLURM array tasks).

Each SLURM array task processes a chunk of simulations:

1. Reads config JSON and parameter CSV
2. Slices to this task's chunk via ``SLURM_ARRAY_TASK_ID``
3. Runs :class:`CppBatchRunner` (ProcessPoolExecutor within the task)
4. Saves one Parquet to the simulation pool directory

Usage (invoked by the generated SLURM script)::

    python -m qsp_hpc.batch.cpp_batch_worker batch_jobs/input/cpp_job_config.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qsp_hpc.batch.test_stats_compute import (
    build_test_stat_registry,
    derive_chunk_to_csv,
)
from qsp_hpc.cpp.batch_runner import CppBatchRunner, load_pool_manifest, write_pool_manifest
from qsp_hpc.utils.logging_config import setup_logger
from qsp_hpc.utils.model_structure_units import load_units_from_model_structure

# Plain getLogger at module scope so `import cpp_batch_worker` (tests,
# subagents, etc.) doesn't mutate the root-logger-adjacent state.
# Handler wiring happens inside main() — the actual script entry — via
# setup_logger, which installs a stdout handler on the qsp_hpc parent
# logger and sets propagate=False (so descendants still emit without
# climbing to root, but root-handler-based captures like pytest's
# caplog aren't fighting for the same child messages).
logger = logging.getLogger(__name__)


def _resolve_max_workers(config_value: int | None) -> int | None:
    """Resolve the ProcessPoolExecutor worker count with SLURM awareness.

    Precedence:
      1. explicit config override (``max_workers`` in the job config)
      2. ``SLURM_CPUS_PER_TASK`` environment variable (cgroup-correct on HPC)
      3. ``None`` → downstream falls back to ``os.cpu_count()``

    The SLURM fallback exists because Python 3.11's ``os.cpu_count()``
    returns the NODE's physical core count, not the cgroup allocation —
    so a default-None on a 64-core Rockfish node spawns 64 workers when
    SLURM actually granted 1 CPU. Python 3.13+'s
    ``os.process_cpu_count()`` respects the cgroup, but we can't assume
    a modern interpreter on every cluster.
    """
    if config_value is not None:
        return config_value
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return int(slurm)
    return None


def _run_inline_derive(
    config: dict,
    chunk_parquet: Path,
    pool_dir: Path,
    batch_subdir: str,
    array_idx: int,
) -> None:
    """Derive test stats for this chunk into the pool's test_stats tree.

    Replaces the chained derivation SLURM job for the hot path: every
    array task computes its own shard immediately after writing its
    trajectory parquet, so derive runs N-way parallel across the array
    instead of serially in one downstream job.

    Output: ``pool_dir/test_stats/<hash>/<batch_subdir>/chunk_NNN_*.csv``.
    The per-batch subdir prevents collision across submissions sharing
    one pool, since each submission writes into its own ``batch_*/``
    sim dir already. ``combine_test_stats_chunks`` walks recursively so
    flat (cold-path) and nested (inline) layouts coexist.

    No-ops silently when ``test_stats_csv`` is absent from the config —
    that's the "sims-only" submission path (caller will trigger
    cold-path derive later).
    """
    test_stats_csv = config.get("test_stats_csv")
    test_stats_hash = config.get("test_stats_hash")
    if not test_stats_csv or not test_stats_hash:
        logger.info("Inline derive skipped: no test_stats_csv / test_stats_hash in config")
        return

    t0 = time.time()
    test_stats_csv_path = Path(test_stats_csv)
    test_stats_df = pd.read_csv(test_stats_csv_path)
    logger.info("Inline derive: %d test statistics loaded", len(test_stats_df))

    registry = build_test_stat_registry(test_stats_df)

    model_structure_file = config.get("model_structure_file")
    if model_structure_file and Path(model_structure_file).exists():
        species_units = load_units_from_model_structure(Path(model_structure_file))
    else:
        species_units = {}

    pool_manifest = load_pool_manifest(pool_dir)
    template_defaults: dict[str, float] | None = None
    if pool_manifest is not None:
        template_defaults = {
            str(k): float(v) for k, v in pool_manifest.get("template_defaults", {}).items()
        }

    aux_samples_csv = config.get("aux_samples_csv")
    auxiliary_units = config.get("auxiliary_units") or {}
    aux_by_sample_index: dict[int, dict[str, float]] = {}
    if aux_samples_csv and Path(aux_samples_csv).exists():
        aux_df = pd.read_csv(aux_samples_csv)
        if "sample_index" in aux_df.columns:
            aux_names = [c for c in aux_df.columns if c != "sample_index"]
            for row in aux_df.itertuples(index=False):
                sid = int(getattr(row, "sample_index"))
                aux_by_sample_index[sid] = {name: float(getattr(row, name)) for name in aux_names}

    output_dir = pool_dir / "test_stats" / test_stats_hash / batch_subdir
    n_sims = derive_chunk_to_csv(
        chunk_parquet=chunk_parquet,
        output_dir=output_dir,
        chunk_idx=array_idx,
        test_stats_df=test_stats_df,
        test_stat_registry=registry,
        species_units=species_units,
        template_defaults=template_defaults,
        aux_by_sample_index=aux_by_sample_index,
        auxiliary_units=auxiliary_units,
    )
    logger.info(
        "Inline derive complete: %d sims in %.1fs, wrote %s",
        n_sims,
        time.time() - t0,
        output_dir,
    )


def run_chunk(config: dict, array_idx: int) -> None:
    """Execute one array-task's chunk of simulations.

    Factored out of :func:`main` so it can be called from tests without
    going through ``sys.argv``.
    """
    binary_path = config["binary_path"]
    template_path = config["template_path"]
    subtree = config.get("subtree", "QSP")
    param_csv = config["param_csv"]
    n_simulations = config["n_simulations"]
    seed = config["seed"]
    jobs_per_chunk = config["jobs_per_chunk"]
    t_end_days = config["t_end_days"]
    min_cadence_hours = config["min_cadence_hours"]
    pool_id = config["simulation_pool_id"]
    pool_base = config["simulation_pool_path"]
    scenario = config.get("scenario", "default")
    max_workers = _resolve_max_workers(config.get("max_workers"))
    logger.info("ProcessPool max_workers resolved to %s", max_workers)
    per_sim_timeout_s = config.get("per_sim_timeout_s", 300.0)
    scenario_yaml = config.get("scenario_yaml")
    drug_metadata_yaml = config.get("drug_metadata_yaml")
    healthy_state_yaml = config.get("healthy_state_yaml")
    # M13: shared evolve-to-diagnosis cache on scratch. First task to hit
    # a given theta runs evolve + writes the QSTH blob under an advisory
    # fcntl lock; every other task (including future scenario arrays) for
    # the same theta skips evolve via --initial-state. None disables.
    evolve_cache_root = config.get("evolve_cache_root")
    # Per-task evolve-pack (#86). ``evolve_pack_dir`` + ``evolve_pack_mode``
    # together drive the NFS-safe successor to the LMDB evolve cache:
    #   - mode "emit"   (default): this task writes its post-evolve QSTH
    #     blobs to ``{evolve_pack_dir}/chunk_NNN.qsep``.
    #   - mode "consume": this task reads ``{evolve_pack_dir}/chunk_NNN.qsep``
    #     (a prior scenario's pack at the same index — the chunking is
    #     identical across scenarios sharing a theta pool) and runs from
    #     each evolve state via --initial-state, skipping the burn-in.
    # Either way the per-task file is chunk-NNN-named so pack ↔ parquet
    # pair by array index. No shared writable store.
    evolve_pack_dir = config.get("evolve_pack_dir")
    evolve_pack_mode = config.get("evolve_pack_mode", "emit")
    # samples_start_offset: when the caller hoisted the FULL theta pool to
    # a shared remote CSV (samples_csv_remote=...) but this submit is a
    # top-up needing only rows ``[existing : N)``, the offset shifts the
    # per-chunk slice into the right region. ``n_simulations`` here is
    # the deficit count, not the pool size; the absolute slice each task
    # reads is ``[offset + start : offset + end]``.
    samples_start_offset = int(config.get("samples_start_offset", 0))

    # #34: announce the cache-wiring inputs at INFO in the parent process.
    # These are also implicitly checked by CppBatchRunner/_worker_init, but
    # a partial config (e.g. evolve_cache_root set but healthy_state_yaml
    # None) silently disables the cache — and the SLURM task still exits 0.
    # Logging here pins down "0 blobs written" to a specific cause without
    # needing a debug rerun.
    logger.info(
        "Evolve-cache config: healthy_state_yaml=%s evolve_cache_root=%s "
        "evolve_pack_dir=%s evolve_pack_mode=%s",
        healthy_state_yaml,
        evolve_cache_root,
        evolve_pack_dir,
        evolve_pack_mode if evolve_pack_dir else "(n/a)",
    )

    start = array_idx * jobs_per_chunk
    end = min(start + jobs_per_chunk, n_simulations)
    chunk_size = end - start

    if chunk_size <= 0:
        logger.info(
            "No simulations for task %d (start=%d >= n=%d)",
            array_idx,
            start,
            n_simulations,
        )
        return

    logger.info("Processing sims [%d, %d) — %d simulations", start, end, chunk_size)

    params_df = pd.read_csv(param_csv)
    # Peel off sample_index (first column, written by CppSimulator
    # ._write_params_csv — mirrors MATLAB's load_parameter_samples_csv.m).
    # Without this, rows entering CppBatchRunner would look like an extra
    # integer-valued "param" and fail the XML template lookup. The index
    # is forwarded to runner.run so the written parquet carries it, which
    # downstream multi-scenario alignment relies on.
    abs_lo = samples_start_offset + start
    abs_hi = samples_start_offset + end
    if "sample_index" in params_df.columns:
        sample_index_all = params_df["sample_index"].astype(np.int64).values
        sample_indices_chunk = sample_index_all[abs_lo:abs_hi]
        params_df = params_df.drop(columns=["sample_index"])
    else:
        # Legacy / test CSVs that predate the sample_index convention —
        # fall back to positional global index (samples_start_offset +
        # task_offset + local idx).
        sample_indices_chunk = np.arange(abs_lo, abs_hi, dtype=np.int64)
    param_names = list(params_df.columns)
    theta_chunk = params_df.iloc[abs_lo:abs_hi].values.astype(np.float64)

    logger.info(
        "Loaded %d parameters: %s%s",
        len(param_names),
        param_names[:5],
        "..." if len(param_names) > 5 else "",
    )

    pool_dir = Path(pool_base) / pool_id
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Write to a per-submission batch subdir (issue #43 option A: no
    # combine step). Each submission deposits one
    # ``batch_{ts}_{scenario}_seed{S}/`` subdir containing one
    # ``chunk_NNN.parquet`` per array task. The derive worker walks
    # subdirs instead of a consolidated parquet. The old
    # ``.staging/{array_job_id}/`` layout + downstream combine job are
    # gone.
    #
    # ``batch_subdir`` comes from submit_cpp_jobs (hpc_job_manager) so
    # every task in the array — and every retry submission — agrees on
    # the same directory. Fallback to a SLURM_ARRAY_JOB_ID-anchored name
    # keeps unit tests + ad-hoc runs working without the orchestrator.
    batch_subdir_override = config.get("batch_subdir")
    if batch_subdir_override:
        batch_dir = pool_dir / batch_subdir_override
    else:
        array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
        batch_dir = pool_dir / f"batch_{array_job_id}_{scenario}_seed{seed}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    filename = f"chunk_{array_idx:03d}.parquet"
    output_path = batch_dir / filename

    # Per-task evolve-pack file: one QSEP per array task, named to match
    # the chunk parquet so pack <-> parquet pair by index (#86). The same
    # chunk_NNN.qsep path is the emit *target* or the consume *source*
    # depending on evolve_pack_mode; only one is non-None per task.
    evolve_pack_path = None  # emit target
    evolve_pack_read_path = None  # consume source
    if evolve_pack_dir:
        _per_task_qsep = Path(evolve_pack_dir) / f"chunk_{array_idx:03d}.qsep"
        if evolve_pack_mode == "consume":
            evolve_pack_read_path = _per_task_qsep
        else:
            evolve_pack_path = _per_task_qsep

    t0 = time.time()

    runner = CppBatchRunner(
        binary_path=binary_path,
        template_path=template_path,
        subtree=subtree,
        default_timeout_s=per_sim_timeout_s,
        scenario_yaml=scenario_yaml,
        drug_metadata_yaml=drug_metadata_yaml,
        healthy_state_yaml=healthy_state_yaml,
        evolve_cache_root=evolve_cache_root,
    )

    # #23: pool_manifest.json lives at the POOL dir (one per pool), not
    # the per-submission staging dir. Every array task races to write it
    # on first run of a fresh pool; write_pool_manifest is idempotent so
    # the second+ writers no-op. Defaults come from the XML template the
    # CppBatchRunner probed at init time, so the snapshot always matches
    # the binary this submission ran against.
    write_pool_manifest(pool_dir, runner.template_defaults, param_names)

    result = runner.run(
        theta_matrix=theta_chunk,
        param_names=param_names,
        sample_indices=sample_indices_chunk,
        t_end_days=t_end_days,
        min_cadence_hours=min_cadence_hours,
        output_path=output_path,
        scenario=scenario,
        seed=seed,
        max_workers=max_workers,
        per_sim_timeout_s=per_sim_timeout_s,
        evolve_pack_path=evolve_pack_path,
        evolve_pack_read_path=evolve_pack_read_path,
    )

    elapsed = time.time() - t0
    logger.info(
        "Task %d complete: %d/%d succeeded in %.1fs, wrote %s%s",
        array_idx,
        result.n_sims - result.n_failed,
        result.n_sims,
        elapsed,
        output_path,
        f" + evolve pack {result.evolve_pack_path}" if result.evolve_pack_path else "",
    )

    _run_inline_derive(
        config=config,
        chunk_parquet=output_path,
        pool_dir=pool_dir,
        batch_subdir=batch_dir.name,
        array_idx=array_idx,
    )


def main() -> None:
    # Wire the qsp_hpc parent logger + this module's logger only when
    # running as a script (i.e. SLURM worker invocation). Descendant
    # loggers (qsp_hpc.cpp.evolve_cache etc.) propagate up to qsp_hpc
    # and land in SLURM .out/.err. See #36 (observability) + #34.
    setup_logger("qsp_hpc", verbose=True)
    setup_logger(__name__, verbose=True)

    if len(sys.argv) != 2:
        print("Usage: python -m qsp_hpc.batch.cpp_batch_worker <config.json>", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]
    array_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    logger.info("C++ QSP Simulation Worker")
    logger.info("Node: %s", os.getenv("SLURMD_NODENAME", "localhost"))
    logger.info("Job ID: %s", os.getenv("SLURM_JOB_ID", "local"))
    logger.info("Array Task ID: %d", array_idx)

    with open(config_file) as f:
        config = json.load(f)

    run_chunk(config, array_idx)


if __name__ == "__main__":
    main()
