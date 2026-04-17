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
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qsp_hpc.cpp.batch_runner import CppBatchRunner
from qsp_hpc.utils.logging_config import setup_logger

logger = setup_logger(__name__, verbose=True)


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
    dt_days = config["dt_days"]
    pool_id = config["simulation_pool_id"]
    pool_base = config["simulation_pool_path"]
    scenario = config.get("scenario", "default")
    max_workers = config.get("max_workers")
    per_sim_timeout_s = config.get("per_sim_timeout_s", 300.0)
    scenario_yaml = config.get("scenario_yaml")
    drug_metadata_yaml = config.get("drug_metadata_yaml")
    healthy_state_yaml = config.get("healthy_state_yaml")

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
    if "sample_index" in params_df.columns:
        sample_index_all = params_df["sample_index"].astype(np.int64).values
        sample_indices_chunk = sample_index_all[start:end]
        params_df = params_df.drop(columns=["sample_index"])
    else:
        # Legacy / test CSVs that predate the sample_index convention —
        # fall back to positional global index (task_offset + local idx).
        sample_indices_chunk = np.arange(start, end, dtype=np.int64)
    param_names = list(params_df.columns)
    theta_chunk = params_df.iloc[start:end].values.astype(np.float64)

    logger.info(
        "Loaded %d parameters: %s%s",
        len(param_names),
        param_names[:5],
        "..." if len(param_names) > 5 else "",
    )

    pool_dir = Path(pool_base) / pool_id
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Write to a per-submission staging subdir. A downstream combine task
    # (cpp_combine_batch_worker) consolidates all chunks into a single
    # pool-level batch parquet, matching MATLAB's "one file per
    # submission" layout. This avoids task-id sharding in the pool dir
    # and lets partial top-up (n_hpc < n) work cleanly: each top-up is
    # simply one additional batch file scanned by the pool loader.
    #
    # SLURM_ARRAY_JOB_ID is shared by every task in the array, so all
    # chunks from one submission land in the same staging dir. Using
    # "local" as the fallback keeps unit tests runnable without SLURM.
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    staging_dir = pool_dir / ".staging" / str(array_job_id)
    staging_dir.mkdir(parents=True, exist_ok=True)
    filename = f"chunk_{array_idx:03d}.parquet"
    output_path = staging_dir / filename

    t0 = time.time()

    runner = CppBatchRunner(
        binary_path=binary_path,
        template_path=template_path,
        subtree=subtree,
        default_timeout_s=per_sim_timeout_s,
        scenario_yaml=scenario_yaml,
        drug_metadata_yaml=drug_metadata_yaml,
        healthy_state_yaml=healthy_state_yaml,
    )

    result = runner.run(
        theta_matrix=theta_chunk,
        param_names=param_names,
        sample_indices=sample_indices_chunk,
        t_end_days=t_end_days,
        dt_days=dt_days,
        output_path=output_path,
        scenario=scenario,
        seed=seed,
        max_workers=max_workers,
        per_sim_timeout_s=per_sim_timeout_s,
    )

    elapsed = time.time() - t0
    logger.info(
        "Task %d complete: %d/%d succeeded in %.1fs, wrote %s",
        array_idx,
        result.n_sims - result.n_failed,
        result.n_sims,
        elapsed,
        output_path,
    )


def main() -> None:
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
