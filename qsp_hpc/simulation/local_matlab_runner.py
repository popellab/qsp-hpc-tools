#!/usr/bin/env python3
"""
Local MATLAB Runner for QSP Simulations

Provides utilities for running QSP simulations locally using MATLAB,
without requiring HPC infrastructure. Reuses HPC batch_worker.m and
test stats derivation code for consistency.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from qsp_hpc.batch.derive_test_stats_worker import (
    build_test_stat_registry,
    compute_test_statistics_batch,
)
from qsp_hpc.utils.logging_config import get_logger, log_operation

logger = get_logger(__name__)


def create_local_matlab_runner(
    model_script: str,
    priors_csv: Path,
    test_stats_csv: Path,
    project_root: Optional[Path] = None,
    project_name: str = "local_sim",
    model_version: str = "v1",
    scenario: str = "default",
    dose_schedule: Optional[Dict[str, Any]] = None,
    sim_config: Optional[Dict[str, Any]] = None,
    matlab_path: str = "matlab",
    verbose: bool = False,
) -> Callable[[np.ndarray, Optional[int]], np.ndarray]:
    """
    Create a local MATLAB runner function for QSP simulations.

    This factory function creates a callable that runs MATLAB simulations
    locally using the HPC batch_worker.m and test stats derivation code.

    Uses the same directory structure as HPC:
    {project_root}/projects/{project_name}/batch_jobs/input/
    {project_root}/projects/{project_name}/batch_jobs/output/

    Args:
        model_script: MATLAB model script name (e.g., 'immune_oncology_model_PDAC')
        priors_csv: Path to priors CSV (needed for parameter names)
        test_stats_csv: Path to test statistics CSV (defines what to extract)
        project_root: Path to project root directory (containing startup.m and projects/)
        project_name: Project identifier (e.g., 'pdac_2025') used for organizing batch files
                     under projects/{project_name}/batch_jobs/
        model_version: Descriptive version name for logging
        scenario: Scenario name for therapy protocol
        dose_schedule: Optional dose schedule configuration
        sim_config: Optional simulation configuration (solver, time, tolerances)
        matlab_path: Path to MATLAB executable (default: 'matlab' from PATH)
        verbose: Enable verbose logging

    Returns:
        Callable that takes (n_sims, n_params) array and returns (n_sims, n_test_stats) array

    Example:
        >>> runner = create_local_matlab_runner(
        ...     model_script='my_model',
        ...     priors_csv=Path('priors.csv'),
        ...     test_stats_csv=Path('test_stats.csv'),
        ...     project_root=Path('.'),
        ...     project_name='my_project'
        ... )
        >>> params = np.random.rand(5, 10)  # 5 sims, 10 params
        >>> test_stats = runner(params, seed=42)     # (5, n_test_stats) array
    """
    # Load parameter names from priors CSV
    priors_df = pd.read_csv(priors_csv)
    param_names = priors_df["name"].tolist()

    # Get absolute path to MATLAB batch worker script
    matlab_dir = Path(__file__).parent.parent / "matlab"
    worker_script = matlab_dir / "batch_worker.m"

    if not worker_script.exists():
        raise FileNotFoundError(
            f"Batch worker script not found: {worker_script}\n"
            f"Expected location: {worker_script.absolute()}"
        )

    def run_matlab_simulation(params: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Run MATLAB simulation locally with given parameters.

        Uses HPC batch_worker.m and derive_test_stats_worker.py for consistency.

        Args:
            params: (n_sims, n_params) array of parameter values
            seed: Optional random seed for reproducibility

        Returns:
            (n_sims, n_test_stats) array of test statistics
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        n_sims = params.shape[0]
        n_params = params.shape[1]

        if n_params != len(param_names):
            raise ValueError(
                f"Parameter array has {n_params} columns but priors define "
                f"{len(param_names)} parameters"
            )

        with log_operation(logger, f"Local MATLAB simulation ({n_sims} sims)"):
            # Use actual project directory structure (like HPC does)
            # batch_worker expects to run from project_root with projects/{project_name}/batch_jobs/
            if project_root is None:
                raise ValueError(
                    "project_root is required for local simulation. "
                    "Pass project_root=Path('.') if running from project directory."
                )

            # Create batch_jobs in actual project structure (mimics HPC)
            # On HPC: {remote_project_path}/projects/{project_name}/batch_jobs/
            # Locally: {project_root}/projects/{project_name}/batch_jobs/
            project_dir = project_root / "projects" / project_name
            batch_jobs_dir = project_dir / "batch_jobs"
            input_dir = batch_jobs_dir / "input"
            output_dir = batch_jobs_dir / "output"

            # Create directories if they don't exist
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create temp directory only for pool output
            with tempfile.TemporaryDirectory(prefix="qsp_local_") as temp_dir:
                temp_path = Path(temp_dir)

                # Write parameters to CSV
                param_csv = input_dir / "params.csv"
                params_df = pd.DataFrame(params, columns=param_names)
                params_df.to_csv(param_csv, index=False)

                # Set up simulation pool directory for Parquet output
                pool_dir = temp_path / "simulation_pool"
                pool_dir.mkdir()

                # Write job config JSON (for batch_worker.m)
                import json

                config = {
                    "model_script": model_script,
                    "param_csv": str((input_dir / "params.csv").relative_to(project_root)),
                    "test_stats_csv": str(test_stats_csv.absolute()),
                    "n_simulations": int(n_sims),
                    "seed": int(seed) if seed is not None else 2025,
                    "jobs_per_chunk": int(n_sims),  # Process all sims in one chunk
                    "save_full_simulations": True,  # Enable Parquet saving
                    "simulation_pool_id": "local_pool",  # Pool subdirectory name
                }

                if dose_schedule is not None:
                    config["dose_schedule"] = dose_schedule
                if sim_config is not None:
                    config["sim_config"] = sim_config

                config_json = input_dir / "job_config.json"
                with open(config_json, "w") as f:
                    json.dump(config, f, indent=2)

                # Set SLURM environment variables (batch_worker expects these)
                env = os.environ.copy()
                env["SLURM_ARRAY_TASK_ID"] = "0"  # Single chunk, index 0
                env["SLURM_JOB_ID"] = "local"
                env["SLURMD_NODENAME"] = "localhost"
                env["SIMULATION_POOL_PATH"] = str(pool_dir.absolute())

                # Prepare MATLAB command (mimicking HPC SLURM script)
                # On HPC: cd to project, then run matlab with batch_worker
                # Here: set cwd to project_root so startup.m is available
                # batch_worker will find projects/{project_name}/batch_jobs/ from current dir
                matlab_cmd = (
                    f"addpath('{matlab_dir.absolute()}'); "
                    f"batch_worker('{project_name}'); "
                    f"exit;"
                )

                # Set working directory for MATLAB (like HPC does with cd before matlab)
                matlab_cwd = project_root.absolute() if project_root is not None else None

                # Run MATLAB
                logger.debug(f"Running MATLAB worker: {worker_script.name}")
                logger.debug(f"Config: {config_json}")
                logger.debug(f"Pool dir: {pool_dir}")
                if matlab_cwd:
                    logger.debug(f"MATLAB working directory: {matlab_cwd}")

                try:
                    result = subprocess.run(
                        [matlab_path, "-batch", matlab_cmd],
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout
                        env=env,
                        cwd=matlab_cwd,  # Start MATLAB from project directory (like HPC)
                    )

                    # Always show MATLAB stdout for debugging
                    logger.info("MATLAB stdout:")
                    for line in result.stdout.split("\n"):
                        if line.strip():
                            logger.info(f"  {line}")

                    if result.returncode != 0:
                        logger.error("MATLAB stderr:")
                        for line in result.stderr.split("\n"):
                            if line.strip():
                                logger.error(f"  {line}")
                        raise RuntimeError(
                            f"MATLAB execution failed with return code {result.returncode}"
                        )

                    # Find the Parquet file that batch_worker created
                    # Format: batch_{array_idx}_{timestamp}_{n_sims}sims_seed{seed}.parquet
                    parquet_files = list((pool_dir / "local_pool").glob("batch_*.parquet"))

                    if not parquet_files:
                        raise RuntimeError(
                            f"MATLAB did not produce Parquet file in {pool_dir / 'local_pool'}\n"
                            f"Check MATLAB output above for errors."
                        )

                    # Use the first (should be only) Parquet file
                    parquet_file = parquet_files[0]
                    logger.debug(f"Reading species data from: {parquet_file}")

                    # Read species data from Parquet
                    species_df = pd.read_parquet(parquet_file)
                    logger.debug(
                        f"Loaded species data: {len(species_df)} sims × "
                        f"{len(species_df.columns)} columns"
                    )

                    # Load test stats CSV and build function registry
                    logger.debug(f"Loading test stats functions from {test_stats_csv}")
                    test_stats_df = pd.read_csv(test_stats_csv)

                    # Build test stat registry (from derive_test_stats_worker)
                    test_stat_registry = build_test_stat_registry(test_stats_df)

                    # Compute test statistics (from derive_test_stats_worker)
                    logger.debug(f"Computing {len(test_stats_df)} test statistics...")
                    test_stats = compute_test_statistics_batch(
                        species_df, test_stats_df, test_stat_registry
                    )

                    # Validate shape
                    if test_stats.shape[0] != n_sims:
                        raise RuntimeError(
                            f"Expected {n_sims} rows in output, got {test_stats.shape[0]}"
                        )

                    logger.debug(
                        f"Computed test stats: {test_stats.shape[0]} sims × "
                        f"{test_stats.shape[1]} stats"
                    )

                    return test_stats

                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        "MATLAB simulation timed out after 5 minutes. "
                        "Check model configuration or increase timeout."
                    )
                except Exception as e:
                    logger.error(f"Local MATLAB simulation failed: {e}")
                    raise

    return run_matlab_simulation


def write_parameter_csv(params: np.ndarray, param_names: list[str], output_path: Path) -> None:
    """
    Write parameters to CSV file in MATLAB-compatible format.

    Args:
        params: (n_sims, n_params) array of parameter values
        param_names: List of parameter names for header
        output_path: Path to output CSV file

    Format:
        - Header row: parameter names
        - Data rows: parameter values (one simulation per row)
    """
    if params.ndim == 1:
        params = params.reshape(1, -1)

    df = pd.DataFrame(params, columns=param_names)
    df.to_csv(output_path, index=False)
    logger.debug(f"Wrote {len(df)} parameter samples to {output_path}")
