"""Simple batch worker runner - works locally or on HPC.

This module provides a thin wrapper around batch_worker.m that just:
1. Writes input files (params.csv, job_config.json)
2. Sets environment variables
3. Runs batch_worker.m
4. Returns results

All workflow logic lives in batch_worker.m - no duplication!
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from qsp_hpc.utils.logging_config import setup_logger


def run_batch_worker(
    params: np.ndarray,
    param_names: list[str],
    model_script: str,
    project_root: Path,
    seed: int = 2025,
    dosing: Optional[Dict[str, Any]] = None,
    sim_config: Optional[Dict[str, Any]] = None,
    matlab_path: str = "matlab",
    simulation_pool_path: Optional[Path] = None,
    simulation_pool_id: str = "local_pool",
    verbose: bool = False,
) -> Path:
    """
    Run batch_worker.m and return path to output parquet file.

    This is a thin wrapper that just sets up the environment and calls batch_worker.m.
    All simulation logic lives in batch_worker.m (same for local and HPC).

    Args:
        params: Parameter array (n_sims, n_params)
        param_names: List of parameter names
        model_script: MATLAB model script name
        project_root: Project root directory (must contain batch_jobs/)
        seed: Random seed
        dosing: Optional dosing configuration dict with 'drugs' list and drug-specific params.
                Passed to MATLAB schedule_dosing() function.
        sim_config: Optional simulation configuration dict with solver settings and time span.
        matlab_path: Path to MATLAB executable
        simulation_pool_path: Where to write parquet files (default: batch_jobs/simulation_pool)
        simulation_pool_id: Subdirectory name within simulation_pool_path
        verbose: Enable verbose logging

    Returns:
        Path to output parquet file

    Raises:
        RuntimeError: If MATLAB execution fails
    """
    logger = setup_logger(__name__, verbose=verbose)

    # Validate inputs
    if params.ndim == 1:
        params = params.reshape(1, -1)
    n_sims = params.shape[0]

    # Set up directory structure (same as HPC)
    batch_jobs_dir = project_root / "batch_jobs"
    input_dir = batch_jobs_dir / "input"
    output_dir = batch_jobs_dir / "output"

    # Use provided simulation_pool_path or default to batch_jobs/simulation_pool
    if simulation_pool_path is None:
        pool_dir = batch_jobs_dir / "simulation_pool"
    else:
        pool_dir = simulation_pool_path

    # Create directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old pool files
    pool_subdir = pool_dir / simulation_pool_id
    if pool_subdir.exists():
        import shutil

        shutil.rmtree(pool_subdir)

    # Write parameters CSV
    param_csv = input_dir / "params.csv"
    params_df = pd.DataFrame(params, columns=param_names)
    params_df.to_csv(param_csv, index=False)
    logger.debug(f"Wrote params: {param_csv}")

    # Write job config (same format as HPC)
    config = {
        "model_script": model_script,
        "param_csv": str((input_dir / "params.csv").relative_to(project_root)),
        "n_simulations": int(n_sims),
        "seed": int(seed),
        "jobs_per_chunk": int(n_sims),
        "save_full_simulations": True,
        "simulation_pool_id": simulation_pool_id,
    }

    if dosing is not None:
        config["dosing"] = dosing
    if sim_config is not None:
        config["sim_config"] = sim_config

    config_json = input_dir / "job_config.json"
    with open(config_json, "w") as f:
        json.dump(config, f, indent=2)
    logger.debug(f"Wrote config: {config_json}")

    # Set environment variables (same as HPC SLURM script)
    env = os.environ.copy()
    env["SLURM_ARRAY_TASK_ID"] = "0"
    env["SLURM_JOB_ID"] = "local"
    env["SLURMD_NODENAME"] = "localhost"
    env["SIMULATION_POOL_PATH"] = str(pool_dir.absolute())

    # Set HPC_VENV_PATH to current Python environment (for write_species_parquet.py)
    env["HPC_VENV_PATH"] = sys.prefix  # Points to current venv or Python installation

    # Get MATLAB scripts path
    import qsp_hpc.matlab

    matlab_dir = Path(qsp_hpc.matlab.get_matlab_path())

    # Run batch_worker.m (exactly as HPC does)
    matlab_cmd = f"addpath('{matlab_dir.absolute()}'); batch_worker(); exit;"

    logger.info(f"Running batch_worker.m ({n_sims} simulations)...")
    logger.debug(f"Working directory: {project_root.absolute()}")
    logger.debug(f"Pool directory: {pool_dir.absolute()}")

    result = subprocess.run(
        [matlab_path, "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
        env=env,
        cwd=project_root.absolute(),
    )

    # Always show MATLAB output (includes per-simulation status)
    if result.stdout:
        for line in result.stdout.split("\n"):
            if line.strip():
                logger.info(f"  {line}")

    # Check for MATLAB execution errors
    if result.returncode != 0:
        logger.error("MATLAB execution failed!")
        logger.error("MATLAB stdout:")
        for line in result.stdout.split("\n"):
            if line.strip():
                logger.error(f"  {line}")
        logger.error("MATLAB stderr:")
        for line in result.stderr.split("\n"):
            if line.strip():
                logger.error(f"  {line}")
        raise RuntimeError(f"MATLAB execution failed with return code {result.returncode}")

    # Find output parquet file
    logger.debug(f"Looking for parquet files in: {pool_dir / simulation_pool_id}")
    parquet_files = list((pool_dir / simulation_pool_id).glob("batch_*.parquet"))

    if not parquet_files:
        # Show all MATLAB output to help diagnose the issue
        logger.error("MATLAB did not produce expected parquet file!")
        logger.error(f"Expected location: {pool_dir / simulation_pool_id}")
        logger.error("MATLAB stdout:")
        for line in result.stdout.split("\n"):
            if line.strip():
                logger.error(f"  {line}")
        if result.stderr:
            logger.error("MATLAB stderr:")
            for line in result.stderr.split("\n"):
                if line.strip():
                    logger.error(f"  {line}")

        # Check if directory exists and what files are there
        pool_subdir = pool_dir / simulation_pool_id
        if pool_subdir.exists():
            all_files = list(pool_subdir.glob("*"))
            logger.error(f"Files found in {pool_subdir}: {[f.name for f in all_files]}")
        else:
            logger.error(f"Directory does not exist: {pool_subdir}")

        raise RuntimeError(
            f"MATLAB did not produce parquet file in {pool_dir / simulation_pool_id}\n"
            f"See MATLAB output above for details."
        )

    parquet_file = parquet_files[0]
    logger.info(f"✓ Simulation complete: {parquet_file.name}")

    return parquet_file
