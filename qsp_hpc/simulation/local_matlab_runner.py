#!/usr/bin/env python3
"""
Local MATLAB Runner for QSP Simulations

Provides utilities for running QSP simulations locally using MATLAB,
without requiring HPC infrastructure. Useful for testing, debugging,
and small-scale simulations.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from qsp_hpc.utils.logging_config import get_logger, log_operation

logger = get_logger(__name__)


def create_local_matlab_runner(
    model_script: str,
    priors_csv: Path,
    test_stats_csv: Path,
    model_version: str = "v1",
    scenario: str = "default",
    dose_schedule: Optional[Dict[str, Any]] = None,
    sim_config: Optional[Dict[str, Any]] = None,
    matlab_path: str = "matlab",
    verbose: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a local MATLAB runner function for QSP simulations.

    This factory function creates a callable that can run MATLAB simulations
    locally, suitable for use with QSPSimulator's matlab_runner parameter.

    Args:
        model_script: MATLAB model script name (e.g., 'immune_oncology_model_PDAC')
        priors_csv: Path to priors CSV (needed for parameter names)
        test_stats_csv: Path to test statistics CSV (defines what to extract)
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
        ...     test_stats_csv=Path('test_stats.csv')
        ... )
        >>> params = np.random.rand(5, 10)  # 5 sims, 10 params
        >>> test_stats = runner(params)     # (5, n_test_stats) array
    """
    # Load parameter names from priors CSV
    priors_df = pd.read_csv(priors_csv)
    param_names = priors_df["name"].tolist()

    # Get absolute path to MATLAB worker script
    matlab_dir = Path(__file__).parent.parent / "matlab"
    worker_script = matlab_dir / "local_worker.m"

    if not worker_script.exists():
        raise FileNotFoundError(
            f"Local worker script not found: {worker_script}\n"
            f"Expected location: {worker_script.absolute()}"
        )

    def run_matlab_simulation(params: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Run MATLAB simulation locally with given parameters.

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
            # Create temp directory for this simulation
            with tempfile.TemporaryDirectory(prefix="qsp_local_") as temp_dir:
                temp_path = Path(temp_dir)

                # Write parameters to CSV
                param_csv = temp_path / "params.csv"
                params_df = pd.DataFrame(params, columns=param_names)
                params_df.to_csv(param_csv, index=False)

                # Write config JSON
                config = {
                    "model_script": model_script,
                    "param_csv": str(param_csv.absolute()),
                    "test_stats_csv": str(test_stats_csv.absolute()),
                    "n_simulations": int(n_sims),
                    "seed": int(seed) if seed is not None else 2025,
                    "model_version": model_version,
                    "scenario": scenario,
                }

                if dose_schedule is not None:
                    config["dose_schedule"] = dose_schedule
                if sim_config is not None:
                    config["sim_config"] = sim_config

                config_json = temp_path / "config.json"
                import json

                with open(config_json, "w") as f:
                    json.dump(config, f, indent=2)

                # Prepare MATLAB command
                output_csv = temp_path / "test_stats.csv"
                matlab_cmd = (
                    f"cd('{matlab_dir.absolute()}'); "
                    f"local_worker('{config_json.absolute()}', '{output_csv.absolute()}'); "
                    f"exit;"
                )

                # Run MATLAB
                logger.debug(f"Running MATLAB worker: {worker_script.name}")
                logger.debug(f"Config: {config_json}")
                logger.debug(f"Output: {output_csv}")

                try:
                    result = subprocess.run(
                        [
                            matlab_path,
                            "-batch",
                            matlab_cmd,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout
                    )

                    if verbose or result.returncode != 0:
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

                    # Read test statistics
                    if not output_csv.exists():
                        raise RuntimeError(
                            f"MATLAB did not produce output file: {output_csv}\n"
                            f"Check MATLAB output above for errors."
                        )

                    test_stats = pd.read_csv(output_csv, header=None).values

                    # Validate shape
                    if test_stats.shape[0] != n_sims:
                        raise RuntimeError(
                            f"Expected {n_sims} rows in output, got {test_stats.shape[0]}"
                        )

                    logger.debug(
                        f"Extracted test stats: {test_stats.shape[0]} sims × "
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


def write_parameter_csv(params: np.ndarray, param_names: List[str], output_path: Path) -> None:
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
