#!/usr/bin/env python3
"""
QSP Simulator for Simulation-Based Inference (SBI)

This module provides a simulator function that interfaces between Python SBI workflows
and MATLAB QSP models. It generates parameter samples from priors, runs QSP model
simulations via MATLAB, and returns model observables for training neural density estimators.

The simulator handles:
  - Parameter sampling from torch distributions
  - CSV generation for MATLAB interface
  - MATLAB batch execution (local or HPC)
  - Results caching and retrieval
  - Observable extraction and formatting

Usage:
    from qsp_hpc.simulation.qsp_simulator import qsp_simulator

    # For use with sbi.inference.simulate_for_sbi
    theta, x = simulate_for_sbi(qsp_simulator, prior, num_simulations)
"""

import csv
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from qsp_hpc.batch.batch_utils import calculate_batch_split
from qsp_hpc.batch.hpc_job_manager import MissingOutputError, RemoteCommandError, SubmissionError
from qsp_hpc.constants import HASH_PREFIX_LENGTH, JOB_QUEUE_TIMEOUT, SLURM_REGISTRATION_DELAY
from qsp_hpc.simulation.simulation_pool import SimulationPoolManager
from qsp_hpc.utils.logging_config import (
    create_child_logger,
    format_config,
    log_operation,
    setup_logger,
)


class QSPSimulatorError(RuntimeError):
    """Raised when simulator orchestration fails (wraps lower-level errors)."""


def _run_worker_batch(
    worker_id: int,
    worker_params: np.ndarray,
    worker_seed: int,
    param_names: List[str],
    model_script: str,
    project_root: Path,
    dosing: Optional[Dict[str, Any]],
    sim_config: Dict[str, Any],
    matlab_path: str,
    temp_path: Path,
) -> Path:
    """
    Run a single worker batch for parallel execution.

    This is a module-level function to enable proper pickling with ProcessPoolExecutor.
    """
    from qsp_hpc.simulation.batch_runner import run_batch_worker

    worker_pool_id = f"worker_{worker_id}"
    worker_pool_path = temp_path / worker_pool_id
    worker_pool_path.mkdir(exist_ok=True)
    worker_work_dir = temp_path / f"work_{worker_id}"
    worker_work_dir.mkdir(exist_ok=True)

    return run_batch_worker(
        params=worker_params,
        param_names=param_names,
        model_script=model_script,
        project_root=project_root,
        seed=worker_seed,
        dosing=dosing,
        sim_config=sim_config,
        matlab_path=matlab_path,
        simulation_pool_path=temp_path,
        simulation_pool_id=worker_pool_id,
        verbose=False,  # Reduce noise from workers
        work_dir=worker_work_dir,
    )


class QSPSimulator:
    """
    QSP Simulator for SBI workflows.

    This class wraps MATLAB QSP model execution and provides a callable interface
    compatible with sbi.inference.simulate_for_sbi.

    Attributes:
        test_stats_csv: Path to test statistics CSV (defines observables)
        priors_csv: Path to priors CSV (defines parameter names)
        model_script: MATLAB model script name (optional)
        model_version: Descriptive version name for simulation pooling
        cache_dir: Directory for caching results
        seed: Random seed for reproducibility
        param_names: Parameter names (loaded from priors CSV)
    """

    def __init__(
        self,
        priors_csv: Union[str, Path],
        test_stats_csv: Optional[Union[str, Path]] = None,
        calibration_targets: Optional[Union[str, Path]] = None,
        species_units_file: Optional[Union[str, Path]] = None,
        model_script: str = "",
        model_version: str = "v1",
        model_description: str = "",
        scenario: str = "default",
        cache_dir: Union[str, Path] = "cache/sbi_simulations",
        seed: int = 2025,
        cache_sampling_seed: Optional[int] = None,
        max_tasks: int = 10,
        poll_interval: int = 30,
        max_wait_time: Optional[int] = None,
        verbose: bool = False,
        pool: Optional["SimulationPoolManager"] = None,
        job_manager: Any = None,
        local_only: bool = False,
        project_root: Optional[Union[str, Path]] = None,
        matlab_path: str = "matlab",
        n_local_workers: int = 1,
        accelerate: bool = False,
    ):
        """
        Initialize QSP simulator.

        Args:
            priors_csv: Path to priors CSV defining parameter names and distributions
            test_stats_csv: Path to test statistics CSV defining observables (optional).
                           If None, simulations run but no test statistics are derived.
                           Use this for QC checks or when you only need raw species data.
            calibration_targets: Path to directory of calibration target YAML files
                                (from qsp-llm-workflows). Alternative to test_stats_csv.
                                Cannot be used together with test_stats_csv.
            species_units_file: Path to species_units.json mapping species names to unit strings.
                               Required if test_stats_csv uses Pint units in model_output_code.
            model_script: MATLAB model script name (e.g., 'immune_oncology_model_PDAC')
            model_version: Descriptive version name (e.g., 'baseline_gvax')
            model_description: Brief description of model configuration
            scenario: Scenario name for therapy protocol (e.g., 'gvax', 'control', 'gvax_anti_pd1')
            cache_dir: Directory for caching simulation results (base directory for all pools)
            seed: Random seed for reproducibility (used for new simulations)
            cache_sampling_seed: Fixed seed for reproducible cache sampling (default: None, uses seed)
                                When set, ensures consistent correlations across runs by sampling
                                the same subset from cache. Use different from seed to get diverse
                                parameter sets while maintaining reproducible analysis.
            max_tasks: Maximum number of parallel SLURM array tasks (default: 10)
                      Actual tasks may be less if fewer simulations needed
            poll_interval: Seconds between job status checks (default: 30)
            max_wait_time: Maximum wait time in seconds before timeout (default: None, no timeout)
            verbose: If True, print detailed progress information (default: False)
        """
        # Validate mutually exclusive observables sources
        if test_stats_csv is not None and calibration_targets is not None:
            raise ValueError("Provide test_stats_csv OR calibration_targets, not both")

        # Handle calibration_targets: load YAMLs, serialize to temp CSV for downstream use
        self._calibration_targets_dir = None
        self._temp_csv = None
        if calibration_targets is not None:
            from qsp_hpc.calibration import load_calibration_targets

            cal_dir = Path(calibration_targets)
            self._calibration_targets_dir = cal_dir
            self._test_stats_df = load_calibration_targets(cal_dir)
            # Serialize to temp CSV for internal use + HPC upload
            self._temp_csv = Path(tempfile.mktemp(suffix=".csv"))
            self._test_stats_df.to_csv(self._temp_csv, index=False)
            test_stats_csv = self._temp_csv

        self.test_stats_csv = Path(test_stats_csv) if test_stats_csv is not None else None
        self.species_units_file = (
            Path(species_units_file) if species_units_file is not None else None
        )
        self.priors_csv = Path(priors_csv)
        self.model_script = model_script
        self.model_version = model_version
        self.model_description = model_description
        self.scenario = scenario
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        self.cache_sampling_seed = cache_sampling_seed if cache_sampling_seed is not None else seed
        self.max_tasks = max_tasks
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time
        self.verbose = verbose
        self.local_only = local_only
        self.project_root = Path(project_root) if project_root is not None else None
        self.matlab_path = matlab_path
        self.n_local_workers = n_local_workers
        self.accelerate = accelerate

        if self.test_stats_csv is not None and not self.test_stats_csv.exists():
            raise FileNotFoundError(f"Test statistics CSV not found: {self.test_stats_csv}")

        if not self.priors_csv.exists():
            raise FileNotFoundError(f"Priors CSV not found: {self.priors_csv}")

        # Load scenario config from YAML (infer path from scenario name)
        self.sim_config, self.dosing = self._load_scenario_config()

        # Load parameter names from priors CSV
        with open(self.priors_csv, "r") as f:
            reader = csv.DictReader(f)
            self.param_names = [row["name"] for row in reader]

        # Initialize SimulationPoolManager for local caching (only if test_stats_csv is provided)
        # Pool requires test_stats_csv for config hashing; skip pooling for QC-only runs
        if pool is not None:
            self.pool = pool
        elif self.test_stats_csv is not None:
            pool_kwargs = dict(
                cache_dir=cache_dir,
                model_version=model_version,
                model_description=model_description,
                priors_csv=priors_csv,
                model_script=model_script,
                scenario=scenario,
            )
            if self._calibration_targets_dir is not None:
                pool_kwargs["calibration_targets"] = self._calibration_targets_dir
            else:
                pool_kwargs["test_stats_csv"] = test_stats_csv
            self.pool = SimulationPoolManager(**pool_kwargs)
        else:
            self.pool = None  # No pooling for QC-only simulations

        # Store job_manager for lazy initialization (don't create until needed)
        self._job_manager = job_manager  # May be None or injected for testing

        # Set up logging with scenario-specific hierarchical logger
        base_logger = setup_logger(__name__, verbose=self.verbose)
        self.logger = create_child_logger(base_logger, self.scenario)

        # Log initialization details (safely handle mock pools and None values)
        self.logger.info(f"Initializing QSP simulator for scenario: {self.scenario}")

        # Handle pool directory logging
        if self.pool is None:
            pool_dir_str = "(no pooling - QC mode)"
            config_hash_str = "(n/a)"
        else:
            try:
                pool_dir = self.pool.pool_dir
                pool_dir_str = str(
                    pool_dir.relative_to(Path.cwd())
                    if pool_dir.is_relative_to(Path.cwd())
                    else pool_dir
                )
            except (AttributeError, TypeError):
                pool_dir_str = "(mock pool)"

            try:
                config_hash_str = self.pool.config_hash[:8] + "..."
            except (AttributeError, TypeError):
                config_hash_str = "(mock)"

        # Handle test_stats_csv logging
        if self.test_stats_csv is not None:
            test_stats_str = str(
                self.test_stats_csv.relative_to(Path.cwd())
                if self.test_stats_csv.is_relative_to(Path.cwd())
                else self.test_stats_csv
            )
        else:
            test_stats_str = "(none - QC mode)"

        config_info = {
            "test_stats_csv": test_stats_str,
            "priors_csv": str(
                self.priors_csv.relative_to(Path.cwd())
                if self.priors_csv.is_relative_to(Path.cwd())
                else self.priors_csv
            ),
            "model_version": self.model_version,
            "model_script": self.model_script or "(default)",
            "scenario": self.scenario,
            "pool_directory": pool_dir_str,
            "config_hash": config_hash_str,
            "seed": self.seed,
            "cache_sampling_seed": self.cache_sampling_seed,
        }
        for line in format_config(config_info):
            self.logger.info(line)

        # Log what's included in the config hash (helps understand cache invalidation)
        self.logger.debug(
            "Config hash includes: priors CSV, test stats CSV, model script, model version, scenario"
        )

        # Random number generator for sampling from pool (use fixed seed for reproducibility)
        self.rng = np.random.default_rng(self.cache_sampling_seed)
        # RNG for parameter generation; seeded once so successive batches differ
        self.param_rng = np.random.default_rng(self.seed)

    def __repr__(self) -> str:
        """Return string representation of simulator."""
        return (
            f"QSPSimulator(\n"
            f"  parameters={len(self.param_names)},\n"
            f"  model_version='{self.model_version}',\n"
            f"  scenario='{self.scenario}',\n"
            f"  test_stats_csv='{self.test_stats_csv}',\n"
            f"  priors_csv='{self.priors_csv}',\n"
            f"  model_script='{self.model_script}',\n"
            f"  max_tasks={self.max_tasks},\n"
            f"  seed={self.seed},\n"
            f"  cache_sampling_seed={self.cache_sampling_seed}\n"
            f")"
        )

    def _load_scenario_config(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Load scenario configuration from YAML file.

        Infers YAML path from scenario name: scenarios/{scenario}.yaml
        relative to project_root.

        Returns:
            Tuple of (sim_config, dosing) dicts, or (None, None) if YAML not found.
        """
        import yaml

        # Default config if no YAML found
        default_sim_config = {
            "start_time": 0,
            "stop_time": 30,
            "time_units": "day",
            "solver": "sundials",
            "abs_tolerance": 1e-9,
            "rel_tolerance": 1e-6,
        }

        # Set up temporary logger for this method (main logger not yet initialized)
        import logging

        temp_logger = logging.getLogger(__name__)

        # Infer YAML path from scenario name
        if self.project_root is None:
            temp_logger.warning(
                "⚠️  No project_root specified - using DEFAULT sim_config (stop_time=30 days, no dosing)"
            )
            return default_sim_config, None

        scenario_yaml = self.project_root / "scenarios" / f"{self.scenario}.yaml"

        if not scenario_yaml.exists():
            temp_logger.warning(
                f"⚠️  Scenario YAML not found: {scenario_yaml}\n"
                f"    Using DEFAULT sim_config (stop_time=30 days, no dosing)"
            )
            return default_sim_config, None

        # Load YAML
        with open(scenario_yaml, "r") as f:
            scenario_data = yaml.safe_load(f)

        # Extract sim_config (use defaults for missing fields)
        sim_config = scenario_data.get("sim_config", {})
        missing_fields = []
        for key, value in default_sim_config.items():
            if key not in sim_config:
                sim_config[key] = value
                missing_fields.append(key)

        if missing_fields:
            temp_logger.warning(
                f"⚠️  Scenario YAML missing sim_config fields: {missing_fields} - using defaults"
            )

        # Extract dosing config
        dosing = scenario_data.get("dosing", None)
        if dosing is None:
            temp_logger.warning(
                "⚠️  Scenario YAML has no 'dosing' section - simulations will run without treatment"
            )

        return sim_config, dosing

    @property
    def job_manager(self):
        """
        Lazy-load HPCJobManager only when needed.

        This avoids requiring HPC configuration for local-only operations
        and makes unit testing easier by deferring config file loading.

        Returns:
            HPCJobManager instance (created on first access if not injected)

        Raises:
            FileNotFoundError: If config file doesn't exist (when actually needed)
        """
        if self._job_manager is None and not self.local_only:
            from qsp_hpc.batch.hpc_job_manager import HPCJobManager

            self._job_manager = HPCJobManager(verbose=self.verbose)
        return self._job_manager

    def _info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def _debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def _warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def _error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def _generate_parameters(self, n_samples: int) -> np.ndarray:
        """
        Generate parameter samples from priors.

        Args:
            n_samples: Number of parameter sets to generate

        Returns:
            numpy array of shape (n_samples, num_params)
        """
        import pandas as pd

        # Load priors CSV
        priors_df = pd.read_csv(self.priors_csv)
        param_names = priors_df["name"].tolist()
        dist_types = priors_df["distribution"].tolist()
        dist_param1 = priors_df["dist_param1"].values
        dist_param2 = priors_df["dist_param2"].values

        # Generate samples
        samples = np.zeros((n_samples, len(param_names)))
        for i in range(len(param_names)):
            if dist_types[i] == "lognormal":
                samples[:, i] = self.param_rng.lognormal(
                    mean=dist_param1[i], sigma=dist_param2[i], size=n_samples
                )
            else:
                raise ValueError(f"Unsupported distribution: {dist_types[i]}")

        return samples  # type: ignore[no-any-return]

    def _format_number(self, value: float) -> str:
        """
        Format a number for display in the test statistics table.

        Uses scientific notation for very large/small numbers,
        otherwise shows up to 3 significant figures.

        Args:
            value: The number to format

        Returns:
            Formatted string (max 12 chars)
        """
        if np.isnan(value):
            return "—"

        # For very small or very large numbers, use scientific notation
        if abs(value) < 0.001 or abs(value) >= 1e6:
            return f"{value:.2e}"

        # For moderate numbers, use up to 3 significant figures
        if abs(value) >= 100:
            return f"{value:.1f}"
        elif abs(value) >= 10:
            return f"{value:.2f}"
        elif abs(value) >= 1:
            return f"{value:.3f}"
        else:
            return f"{value:.4f}"

    def _format_log_ratio(self, log_ratio: float) -> str:
        """
        Format a log10 ratio for display.

        Shows the log10 ratio with sign (+1 = 10x higher, -1 = 10x lower).

        Args:
            log_ratio: The log10(simulated/observed) ratio

        Returns:
            Formatted string (max 12 chars)
        """
        return f"{log_ratio:+.2f}"

    def compute_test_statistics_table(
        self, test_stats: np.ndarray, test_stats_df: pd.DataFrame, n_sims: int
    ) -> pd.DataFrame:
        """
        Compute test statistics comparison table.

        This method computes summary statistics comparing simulated test statistics
        to observed values, including median, percentage difference, 95% confidence
        intervals, and coverage status.

        Args:
            test_stats: Computed test statistics array (n_sims, n_test_stats)
            test_stats_df: DataFrame with test statistics metadata (must have
                'test_statistic_id' and 'median' columns)
            n_sims: Number of simulations

        Returns:
            DataFrame with columns:
                - test_statistic_id: str, identifier for the test statistic
                - median: float, median of simulated values (NaN if all sims failed)
                - observed: float, observed value from CSV
                - log_ratio: float, log10(median/observed) - symmetric measure where
                    +1 means 10x higher, -1 means 10x lower (NaN if undefined)
                - ci_lower: float, 2.5th percentile (NaN if n_sims=1)
                - ci_upper: float, 97.5th percentile (NaN if n_sims=1)
                - covers_observed: bool, True if observed in [ci_lower, ci_upper]
                    (False if n_sims=1 or observed is NaN)
        """
        rows = []

        for idx, row in test_stats_df.iterrows():
            test_stat_id = row["test_statistic_id"]
            observed_value = row.get("median", np.nan)

            # Compute value (median if multiple sims, single value if n_sims=1)
            if n_sims == 1:
                computed_value = test_stats[0, idx]
                ci_lower = np.nan
                ci_upper = np.nan
            else:
                # Use nanmedian to ignore NaN values and be robust to outliers
                computed_value = np.nanmedian(test_stats[:, idx])
                # Get all individual values for this test statistic
                individual_values = test_stats[:, idx]
                # Compute 95% CI (2.5th and 97.5th percentiles)
                ci_lower = np.nanpercentile(individual_values, 2.5)
                ci_upper = np.nanpercentile(individual_values, 97.5)

            # Calculate log10 ratio (symmetric: +1 means 10x higher, -1 means 10x lower)
            if (
                not np.isnan(computed_value)
                and not np.isnan(observed_value)
                and observed_value > 0
                and computed_value > 0
            ):
                log_ratio = np.log10(computed_value / observed_value)
            else:
                log_ratio = np.nan

            # Check 95% CI coverage (only for multiple sims with valid observed)
            if (
                n_sims > 1
                and not np.isnan(ci_lower)
                and not np.isnan(ci_upper)
                and not np.isnan(observed_value)
            ):
                covers_observed = ci_lower <= observed_value <= ci_upper
            else:
                covers_observed = False

            rows.append(
                {
                    "test_statistic_id": test_stat_id,
                    "median": computed_value,
                    "observed": observed_value,
                    "log_ratio": log_ratio,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "covers_observed": covers_observed,
                }
            )

        return pd.DataFrame(rows)

    def _log_test_statistics_table(
        self, test_stats: np.ndarray, test_stats_df: pd.DataFrame, n_sims: int
    ) -> None:
        """
        Log test statistics in a detailed table format.

        Shows passing tests (95% CI covers observed) first, then failing tests.

        Args:
            test_stats: Computed test statistics array (n_sims, n_test_stats)
            test_stats_df: DataFrame with test statistics metadata
            n_sims: Number of simulations
        """
        # Compute the statistics table
        stats_table = self.compute_test_statistics_table(test_stats, test_stats_df, n_sims)

        self.logger.info("")
        self.logger.info(f"Test Statistics ({n_sims} simulation{'s' if n_sims > 1 else ''}):")

        # Separate passing and failing tests (only for n_sims > 1)
        if n_sims > 1:
            # Valid rows have non-NaN observed and CI values
            valid_mask = ~stats_table["observed"].isna() & ~stats_table["ci_lower"].isna()
            # Fail if: CI doesn't cover observed OR |log_ratio| > 2 (>100x difference)
            large_diff_mask = stats_table["log_ratio"].abs() > 2
            passing_table = stats_table[
                valid_mask & stats_table["covers_observed"] & ~large_diff_mask
            ]
            failing_table = stats_table[
                valid_mask & (~stats_table["covers_observed"] | large_diff_mask)
            ]
            invalid_table = stats_table[~valid_mask]
        else:
            passing_table = stats_table
            failing_table = pd.DataFrame()
            invalid_table = pd.DataFrame()

        def log_table_rows(table: pd.DataFrame, is_multi_sim: bool) -> None:
            """Log rows for a statistics table."""
            for _, row in table.iterrows():
                test_stat_id = row["test_statistic_id"]
                computed_value = row["median"]
                observed_value = row["observed"]
                log_ratio = row["log_ratio"]
                ci_lower = row["ci_lower"]
                ci_upper = row["ci_upper"]
                covers_observed = row["covers_observed"]

                # Format computed value
                if np.isnan(computed_value):
                    computed_str = "NaN"
                    observed_str = self._format_number(observed_value)
                    diff_str = "—"
                    coverage_str = "—"
                    ci_str = "—"
                else:
                    computed_str = self._format_number(computed_value)
                    observed_str = self._format_number(observed_value)

                    # Format log ratio
                    if not np.isnan(log_ratio):
                        diff_str = self._format_log_ratio(log_ratio)
                    else:
                        diff_str = "—"

                    # Format 95% CI coverage (only for multiple sims)
                    if is_multi_sim and not np.isnan(ci_lower) and not np.isnan(observed_value):
                        coverage_str = "✓" if covers_observed else "✗"
                        ci_str = (
                            f"[{self._format_number(ci_lower)}, {self._format_number(ci_upper)}]"
                        )
                    else:
                        coverage_str = "—"
                        ci_str = "—"

                # Truncate test_stat_id if too long
                display_id = test_stat_id if len(test_stat_id) <= 35 else test_stat_id[:32] + "..."

                if is_multi_sim:
                    self.logger.info(
                        f"│ {display_id:<35} │ {computed_str:>12} │ {observed_str:>12} │ {diff_str:>12} │ {coverage_str:^6} │ {ci_str:<26} │"
                    )
                else:
                    self.logger.info(
                        f"│ {display_id:<35} │ {computed_str:>12} │ {observed_str:>12} │ {diff_str:>12} │"
                    )

        # Log passing tests table
        if n_sims > 1:
            self.logger.info("")
            self.logger.info(
                f"Passing ({len(passing_table)} tests - 95% CI covers observed, <100x diff):"
            )
            self.logger.info(
                "┌─────────────────────────────────────┬──────────────┬──────────────┬──────────────┬────────┬────────────────────────────┐"
            )
            self.logger.info(
                "│ Test Statistic                      │ Median       │ Observed     │ log₁₀(M/O)   │ 95% CI │ [2.5%, 97.5%]              │"
            )
            self.logger.info(
                "├─────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────┼────────────────────────────┤"
            )
            log_table_rows(passing_table, is_multi_sim=True)
            self.logger.info(
                "└─────────────────────────────────────┴──────────────┴──────────────┴──────────────┴────────┴────────────────────────────┘"
            )

            # Log failing tests table (if any)
            if len(failing_table) > 0:
                self.logger.info("")
                self.logger.info(
                    f"Failing ({len(failing_table)} tests - 95% CI misses observed OR >100x diff):"
                )
                self.logger.info(
                    "┌─────────────────────────────────────┬──────────────┬──────────────┬──────────────┬────────┬────────────────────────────┐"
                )
                self.logger.info(
                    "│ Test Statistic                      │ Median       │ Observed     │ log₁₀(M/O)   │ 95% CI │ [2.5%, 97.5%]              │"
                )
                self.logger.info(
                    "├─────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────┼────────────────────────────┤"
                )
                log_table_rows(failing_table, is_multi_sim=True)
                self.logger.info(
                    "└─────────────────────────────────────┴──────────────┴──────────────┴──────────────┴────────┴────────────────────────────┘"
                )

            # Log invalid tests (NaN observed or CI) if any
            if len(invalid_table) > 0:
                self.logger.info("")
                self.logger.info(f"Invalid ({len(invalid_table)} tests - missing observed or CI):")
                self.logger.info(
                    "┌─────────────────────────────────────┬──────────────┬──────────────┬──────────────┬────────┬────────────────────────────┐"
                )
                self.logger.info(
                    "│ Test Statistic                      │ Median       │ Observed     │ log₁₀(M/O)   │ 95% CI │ [2.5%, 97.5%]              │"
                )
                self.logger.info(
                    "├─────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────┼────────────────────────────┤"
                )
                log_table_rows(invalid_table, is_multi_sim=True)
                self.logger.info(
                    "└─────────────────────────────────────┴──────────────┴──────────────┴──────────────┴────────┴────────────────────────────┘"
                )
        else:
            # Single simulation - show all in one table
            self.logger.info(
                "┌─────────────────────────────────────┬──────────────┬──────────────┬──────────────┐"
            )
            self.logger.info(
                "│ Test Statistic                      │ Median       │ Observed     │ log₁₀(M/O)   │"
            )
            self.logger.info(
                "├─────────────────────────────────────┼──────────────┼──────────────┼──────────────┤"
            )
            log_table_rows(stats_table, is_multi_sim=False)
            self.logger.info(
                "└─────────────────────────────────────┴──────────────┴──────────────┴──────────────┘"
            )

        # Summary - compute from the stats table
        total_stats = len(stats_table)
        nan_count = stats_table["median"].isna().sum()
        success_count = total_stats - nan_count
        # Count tests with |log_ratio| > 1 (i.e., >10x difference)
        large_diff_count = (stats_table["log_ratio"].abs() > 1).sum()

        # Coverage stats (only for n_sims > 1)
        if n_sims > 1:
            # Only count rows with valid observed values for coverage
            valid_coverage_mask = ~stats_table["observed"].isna() & ~stats_table["ci_lower"].isna()
            covered_count = (stats_table.loc[valid_coverage_mask, "covers_observed"]).sum()
            not_covered_count = valid_coverage_mask.sum() - covered_count
        else:
            covered_count = 0
            not_covered_count = 0

        self.logger.info("")
        summary_parts = [f"✓ Computed {success_count}/{total_stats} test statistics"]
        if nan_count > 0:
            summary_parts.append(f"{nan_count} NaN")
        if large_diff_count > 0:
            summary_parts.append(f"{large_diff_count} >10x diff from observed")
        if n_sims > 1 and (covered_count + not_covered_count) > 0:
            summary_parts.append(
                f"{covered_count}/{covered_count + not_covered_count} 95% CI cover observed"
            )
        self.logger.info(" (".join(summary_parts) + (")" if len(summary_parts) > 1 else ""))

    def log_test_statistics_summary(self, test_stats: np.ndarray) -> None:
        """
        Log test statistics summary table (public API).

        Shows passing tests (95% CI covers observed) first, then failing tests.
        Useful for inspecting results after running simulations.

        Args:
            test_stats: Computed test statistics array (n_sims, n_test_stats)

        Example:
            >>> params, test_stats = simulator.run_local_simulation(n_sims=100)
            >>> simulator.log_test_statistics_summary(test_stats)
        """
        if self.test_stats_csv is None:
            raise ValueError("No test_stats_csv configured for this simulator")

        test_stats_df = pd.read_csv(self.test_stats_csv)
        n_sims = test_stats.shape[0]
        self._log_test_statistics_table(test_stats, test_stats_df, n_sims)

    def print_test_statistic(self, test_stat_id: str) -> None:
        """
        Pretty print a test statistic function for inspection.

        Args:
            test_stat_id: ID of the test statistic to print
        """
        test_stats_df = pd.read_csv(self.test_stats_csv)

        # Find the test statistic
        row = test_stats_df[test_stats_df["test_statistic_id"] == test_stat_id]
        if len(row) == 0:
            print(f"❌ Test statistic '{test_stat_id}' not found")
            print("\nAvailable test statistics:")
            for tid in test_stats_df["test_statistic_id"]:
                print(f"  - {tid}")
            return

        row = row.iloc[0]

        # Print header
        print(f"\n{'='*80}")
        print(f"Test Statistic: {test_stat_id}")
        print(f"{'='*80}")

        # Print metadata
        print(f"\nRequired Species: {row.get('required_species', '—')}")

        observed_median = row.get("median", np.nan)
        if not np.isnan(observed_median):
            print(f"Observed Median: {observed_median:.6g}")

        observed_iqr = row.get("iqr", np.nan)
        if not np.isnan(observed_iqr):
            print(f"Observed IQR: {observed_iqr:.6g}")

        units = row.get("units", "")
        if units:
            print(f"Units: {units}")

        # Print function code
        print("\nFunction Code:")
        print(f"{'-'*80}")
        function_code = row.get("model_output_code", "")
        if function_code:
            print(function_code)
        else:
            print("(no function code)")
        print(f"{'-'*80}\n")

    def list_test_statistics(self) -> None:
        """Print a list of all available test statistics."""
        test_stats_df = pd.read_csv(self.test_stats_csv)

        print(f"\nTest Statistics ({len(test_stats_df)} total):")
        print(f"{'='*80}")

        for _, row in test_stats_df.iterrows():
            test_stat_id = row["test_statistic_id"]
            required_species = row.get("required_species", "—")
            observed_median = row.get("median", np.nan)

            print(f"\n  {test_stat_id}")
            print(f"    Required: {required_species}")
            if not np.isnan(observed_median):
                print(f"    Observed: {observed_median:.6g}")
        print()

    def _log_parameters_table(self, param_values: np.ndarray, param_names: list[str]) -> None:
        """
        Log parameter values in a detailed table format.

        Args:
            param_values: Parameter values array (n_params,)
            param_names: List of parameter names
        """
        self.logger.info("")
        self.logger.info("Parameters:")
        self.logger.info("┌─────────────────────────────────────┬──────────────┐")
        self.logger.info("│ Parameter                           │ Value        │")
        self.logger.info("├─────────────────────────────────────┼──────────────┤")

        for param_name, param_value in zip(param_names, param_values):
            # Format value
            value_str = f"{param_value:.6g}"

            # Truncate param_name if too long
            display_name = param_name if len(param_name) <= 35 else param_name[:32] + "..."

            self.logger.info(f"│ {display_name:<35} │ {value_str:>12} │")

        self.logger.info("└─────────────────────────────────────┴──────────────┘")

    def run_local_simulation(
        self,
        n_sims: int = 1,
        seed: Optional[int] = None,
        matlab_path: str = "matlab",
        n_workers: int = 1,
        accelerate: bool = False,
    ) -> Tuple[np.ndarray, Union[np.ndarray, Path]]:
        """
        Run simulations locally using MATLAB (no HPC required).

        This method is useful for testing, debugging, and small-scale simulations.
        It samples parameters from the prior, runs MATLAB simulations locally,
        and returns both parameters and test statistics (or parquet path if no
        test_stats_csv was provided).

        Args:
            n_sims: Number of simulations to run (default: 1)
            seed: Random seed for parameter sampling (default: use simulator's seed)
            matlab_path: Path to MATLAB executable (default: 'matlab' from PATH)
            n_workers: Number of parallel MATLAB processes (default: 1, no parallelization)
            accelerate: Use sbioaccelerate for faster simulations (default: False)

        Returns:
            Tuple of (params, result):
            - params: numpy array of shape (n_sims, num_params)
            - result: If test_stats_csv was provided, numpy array of shape (n_sims, num_test_stats).
                     If test_stats_csv was None, Path to parquet file with raw species data.

        Raises:
            RuntimeError: If MATLAB execution fails or local worker script not found
            ValueError: If parameter dimensions don't match

        Example:
            >>> # With test statistics
            >>> simulator = QSPSimulator(
            ...     priors_csv='priors.csv',
            ...     test_stats_csv='test_stats.csv',
            ...     model_script='my_model'
            ... )
            >>> params, test_stats = simulator.run_local_simulation(n_sims=5, seed=123)
            >>> print(f"Test stats shape: {test_stats.shape}")

            >>> # Parallel execution with acceleration
            >>> params, test_stats = simulator.run_local_simulation(
            ...     n_sims=100, seed=123, n_workers=4, accelerate=True
            ... )
        """
        import pandas as pd

        from qsp_hpc.utils.logging_config import log_section

        # Use provided seed or fall back to simulator's seed
        if seed is None:
            seed = self.seed

        # Adjust n_workers if we have fewer sims than workers
        n_workers = min(n_workers, n_sims)

        # Log the operation
        with log_section(
            self.logger,
            f"Local MATLAB Simulation: {self.scenario}",
            separator_width=80,
        ):
            config_info = {
                "n_simulations": n_sims,
                "seed": seed,
                "model_script": self.model_script or "(default)",
                "model_version": self.model_version,
                "scenario": self.scenario,
                "test_stats_csv": str(self.test_stats_csv) if self.test_stats_csv else "(none)",
                "priors_csv": str(self.priors_csv),
                "n_workers": n_workers,
                "accelerate": accelerate,
            }
            for line in format_config(config_info):
                self.logger.info(line)

            # Sample parameters from prior
            self.logger.info(f"Sampling {n_sims} parameter sets from prior (seed={seed})")

            # Create temporary RNG with the specified seed
            temp_rng = np.random.default_rng(seed)

            # Generate parameters using the temporary RNG
            priors_df = pd.read_csv(self.priors_csv)
            param_names = priors_df["name"].tolist()
            dist_types = priors_df["distribution"].tolist()
            dist_param1 = priors_df["dist_param1"].values
            dist_param2 = priors_df["dist_param2"].values

            params = np.zeros((n_sims, len(param_names)))
            for i in range(len(param_names)):
                if dist_types[i] == "lognormal":
                    params[:, i] = temp_rng.lognormal(
                        mean=dist_param1[i], sigma=dist_param2[i], size=n_sims
                    )
                else:
                    raise ValueError(f"Unsupported distribution: {dist_types[i]}")

            self.logger.info(f"✓ Generated {params.shape[0]} × {params.shape[1]} parameter matrix")

            # Log parameter table
            if n_sims > 0:
                self._log_parameters_table(params[0], param_names)

            # Run batch_worker.m locally
            from qsp_hpc.simulation.batch_runner import run_batch_worker

            self.logger.info("Running batch_worker.m locally")
            self.logger.info(f"  sim_config: stop_time={self.sim_config.get('stop_time', 30)} days")
            self.logger.info(f"  accelerate: {accelerate}")
            if self.dosing and self.dosing.get("drugs"):
                self.logger.info(f"  dosing: {self.dosing.get('drugs')}")
            else:
                self.logger.info("  dosing: none (baseline)")

            # Prepare sim_config with acceleration flag
            sim_config_with_accel = dict(self.sim_config) if self.sim_config else {}
            sim_config_with_accel["accelerate_model"] = accelerate

            if n_workers == 1:
                # Single worker - run all simulations in one batch
                parquet_file = run_batch_worker(
                    params=params,
                    param_names=param_names,
                    model_script=self.model_script or "default_model",
                    project_root=self.project_root,
                    seed=seed,
                    dosing=self.dosing,
                    sim_config=sim_config_with_accel,
                    matlab_path=matlab_path,
                    verbose=self.verbose,
                )
            else:
                # Parallel execution - split across workers
                from concurrent.futures import ProcessPoolExecutor, as_completed

                self.logger.info(f"  Splitting {n_sims} sims across {n_workers} parallel workers")

                # Split parameters into chunks
                chunk_size = n_sims // n_workers
                remainder = n_sims % n_workers
                chunks = []
                start = 0
                for i in range(n_workers):
                    end = start + chunk_size + (1 if i < remainder else 0)
                    chunks.append((i, params[start:end], seed + i * 10000))
                    start = end

                self.logger.info(f"  Chunk sizes: {[c[1].shape[0] for c in chunks]}")

                # Create temp directory for worker outputs
                with tempfile.TemporaryDirectory(prefix="qsp_parallel_") as temp_dir:
                    temp_path = Path(temp_dir)

                    # Run workers in parallel using module-level function (picklable)
                    parquet_files = []
                    with ProcessPoolExecutor(max_workers=n_workers) as executor:
                        futures = {
                            executor.submit(
                                _run_worker_batch,
                                wid,
                                wparams,
                                wseed,
                                param_names,
                                self.model_script or "default_model",
                                self.project_root,
                                self.dosing,
                                sim_config_with_accel,
                                matlab_path,
                                temp_path,
                            ): wid
                            for wid, wparams, wseed in chunks
                        }

                        for future in as_completed(futures):
                            worker_id = futures[future]
                            try:
                                pq_file = future.result()
                                parquet_files.append((worker_id, pq_file))
                                self.logger.info(f"  Worker {worker_id} complete")
                            except Exception as e:
                                self.logger.error(f"  Worker {worker_id} failed: {e}")
                                raise

                    # Combine parquet files
                    self.logger.info("  Combining results from all workers...")
                    dfs = []
                    for worker_id, pq_file in sorted(parquet_files, key=lambda x: x[0]):
                        df = pd.read_parquet(pq_file)
                        # Renumber patient_id to be globally unique
                        base_id = sum(chunks[i][1].shape[0] for i in range(worker_id))
                        if "patient_id" in df.columns:
                            df["patient_id"] = df["patient_id"] + base_id
                        dfs.append(df)

                    combined_df = pd.concat(dfs, ignore_index=True)

                    # Write combined output
                    output_dir = (
                        self.project_root / "batch_jobs" / "simulation_pool" / "parallel_combined"
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    parquet_file = output_dir / f"batch_{n_sims}sims_combined.parquet"
                    combined_df.to_parquet(parquet_file)

                    self.logger.info(f"  ✓ Combined {n_sims} simulations")

            # If no test_stats_csv, return parquet path for raw species access
            if self.test_stats_csv is None:
                self.logger.info(f"✓ Simulations complete. Parquet file: {parquet_file}")
                return params, parquet_file

            # Derive test stats from parquet (same as HPC derivation worker)
            import json

            from qsp_hpc.batch.derive_test_stats_worker import (
                build_test_stat_registry,
                compute_test_statistics_batch,
            )

            self.logger.info("Deriving test statistics from simulation data")
            species_df = pd.read_parquet(parquet_file)
            test_stats_df = pd.read_csv(self.test_stats_csv)
            test_stat_registry = build_test_stat_registry(test_stats_df)

            # Load species units (required for Pint-aware test statistics)
            if self.species_units_file is not None and self.species_units_file.exists():
                with open(self.species_units_file, "r") as f:
                    species_units = json.load(f)
                self.logger.info(f"Loaded units for {len(species_units)} species")
            else:
                species_units = {}
                self.logger.warning(
                    "No species_units_file provided - using dimensionless for all species"
                )

            test_stats = compute_test_statistics_batch(
                species_df, test_stats_df, test_stat_registry, species_units
            )

            # Log detailed test statistics table
            self._log_test_statistics_table(test_stats, test_stats_df, n_sims)

            return params, test_stats

    def _download_and_add_to_pool(
        self, hpc_pool_path: str, test_stats_hash: str, num_simulations: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Download test statistics from HPC and add to SimulationPoolManager.

        Args:
            hpc_pool_path: Path to HPC pool directory
            test_stats_hash: Hash of test statistics configuration
            num_simulations: Number of simulations to download

        Returns:
            Tuple of (params, observables)
        """
        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_cache_dir = Path(temp_dir)

            with log_operation(self.logger, "Downloading test statistics from HPC"):
                try:
                    params, test_stats = self.job_manager.download_test_stats(
                        hpc_pool_path, test_stats_hash, temp_cache_dir
                    )
                except (MissingOutputError, RemoteCommandError) as exc:
                    raise QSPSimulatorError(
                        f"Failed to download test statistics from HPC: {exc}"
                    ) from exc

            if params is None:
                raise RuntimeError(
                    "HPC has test statistics but no parameters. "
                    "This indicates older simulation data that is no longer supported. "
                    "Please re-run simulations."
                )

            # Replace local pool for this scenario with the full HPC data
            for batch in self.pool._scan_batches(scenario=self.scenario):
                batch_path = self.pool.pool_dir / batch["filename"]
                batch_path.unlink(missing_ok=True)
                self.logger.info(f"Removed stale batch: {batch['filename']}")
            self.logger.info(f"Adding {params.shape[0]} simulations to local pool")
            self.pool.add_batch(
                params_matrix=params,
                observables_matrix=test_stats,
                seed=self.seed,
                scenario=self.scenario,
            )

        # Load from pool (handles sampling if needed)
        params_out, obs_out = self.pool.load_simulations(
            n_requested=num_simulations, scenario=self.scenario, random_state=self.rng
        )

        self.logger.info(f"✓ Returning {params_out.shape[0]} simulations")
        return params_out, obs_out

    def _check_hpc_simulations(
        self, num_simulations: int, priors_hash: str, verbose: bool = False
    ) -> Tuple[bool, str, int]:
        """Check HPC for existing full simulations."""
        # Construct pool path with scenario suffix (must match save location)
        hpc_pool_id = f"{self.model_version}_{priors_hash[:HASH_PREFIX_LENGTH]}_{self.scenario}"
        hpc_pool_path = f"{self.job_manager.config.simulation_pool_path}/{hpc_pool_id}"
        self._debug(f"HPC pool: {hpc_pool_id}")

        # Check directly using result_collector
        has_full_sims = self.job_manager.result_collector.check_pool_directory_exists(hpc_pool_path)
        n_available = (
            self.job_manager.result_collector.count_pool_simulations(hpc_pool_path)
            if has_full_sims
            else 0
        )
        has_full_sims = n_available >= num_simulations

        if has_full_sims:
            self._info(f"Found {n_available} HPC simulations (sufficient)")
        elif n_available > 0:
            self._info(f"Found {n_available}/{num_simulations} HPC simulations (need more)")
        else:
            if verbose:
                self._debug("No HPC simulations found")

        return has_full_sims, hpc_pool_path, n_available

    def _derive_test_statistics(
        self, hpc_pool_path: str, test_stats_hash: str, num_simulations: int, verbose: bool = False
    ) -> None:
        """Derive test statistics from full simulations on HPC."""
        # Validate that derived test stats match requested simulation count
        try:
            has_test_stats = self.job_manager.check_hpc_test_stats(
                hpc_pool_path, test_stats_hash, expected_n_sims=num_simulations
            )
        except (RemoteCommandError, MissingOutputError) as exc:
            raise QSPSimulatorError(f"Failed checking test stats on HPC: {exc}") from exc

        if has_test_stats:
            self.logger.info("✓ Test statistics already derived")
            return

        # Need to derive test statistics
        self.logger.info("Submitting derivation job to HPC...")

        # Launch derivation job (with selective batch processing)
        job_id = self.job_manager.submit_derivation_job(
            hpc_pool_path,
            str(self.test_stats_csv),
            test_stats_hash,
            species_units_file=str(self.species_units_file) if self.species_units_file else None,
            num_simulations=num_simulations,
        )
        self.logger.info(f"Derivation job submitted: {job_id}")

        # Wait for derivation job to complete
        with log_operation(self.logger, "Waiting for derivation job", log_start=False):
            self._wait_for_completion([job_id], num_simulations)

        # Verify that test stats were actually created
        has_test_stats_now = self.job_manager.check_hpc_test_stats(hpc_pool_path, test_stats_hash)

        if not has_test_stats_now:
            # Derivation failed - check logs
            log_path = f"{self.job_manager.config.remote_project_path}/batch_jobs/logs"
            log_cmd = f'ls -lt "{log_path}"/qsp_derive_*.err 2>/dev/null | head -3'
            status, log_output = self.job_manager.transport.exec(log_cmd)

            if status == 0 and log_output.strip():
                self.logger.error("Recent derivation error logs:")
                self.logger.error(f"  {log_output.strip()}")

            raise RuntimeError(
                f"Derivation job completed but did not produce test statistics. "
                f"Check HPC logs at: {log_path}/qsp_derive_*.err"
            )

    def _download_derived_test_stats(
        self, hpc_pool_path: str, test_stats_hash: str, num_simulations: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Download derived test statistics and parameters from HPC.

        Args:
            hpc_pool_path: Path to HPC pool directory
            test_stats_hash: Hash of test statistics configuration
            num_simulations: Number of simulations expected

        Returns:
            Tuple of (params, test_stats):
            - params: Numpy array of parameters (n_sims x n_params)
            - test_stats: Numpy array of test statistics (n_sims x n_observables)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_cache_dir = Path(temp_dir)

            with log_operation(self.logger, "Downloading derived test statistics"):
                params, test_stats = self.job_manager.download_test_stats(
                    hpc_pool_path, test_stats_hash, temp_cache_dir
                )

            if test_stats is None:
                raise RuntimeError(
                    "Failed to download derived test statistics from HPC. "
                    "Check derivation job logs."
                )

            if params is None:
                raise RuntimeError(
                    "Failed to download parameters from HPC. " "Check derivation job output."
                )

            if test_stats.shape[0] != params.shape[0]:
                raise RuntimeError(
                    f"Row mismatch: {params.shape[0]} params vs {test_stats.shape[0]} test stats"
                )

            if test_stats.shape[0] != num_simulations:
                self.logger.info(
                    f"Downloaded {test_stats.shape[0]} simulations from HPC "
                    f"(requested {num_simulations} new, pool has accumulated more)"
                )

            return params, test_stats

    def _run_new_simulations(self, num_simulations: int) -> None:
        """
        Run new simulations on HPC and add to pool.

        The HPC derivation processes ALL batches in the pool (old + new),
        so we download the full params + test_stats and replace the local
        pool with the complete set.

        Results are added to the pool, not returned directly.
        Caller should use pool.load_simulations() to get results.

        Args:
            num_simulations: Number of simulations to run
        """
        self._info(
            f"Running {num_simulations} new simulations on HPC (scenario='{self.scenario}')..."
        )

        theta_np, samples_csv = self._stage_parameters_to_csv(num_simulations)

        try:
            params_np, observables_np = self._run_matlab_simulation(samples_csv, num_simulations)
            # Use the HPC-returned params (covers full pool: old + new batches)
            # instead of the locally-generated theta_np, to stay aligned with
            # the derived test statistics which span all batches.
            self._update_pool_with_results(params_np, observables_np)
        finally:
            Path(samples_csv).unlink(missing_ok=True)

    def _stage_parameters_to_csv(self, num_simulations: int) -> Tuple[np.ndarray, str]:
        """Generate parameters and stage to a temporary CSV."""
        theta_np = self._generate_parameters(num_simulations)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            samples_csv = tmp.name
            writer = csv.writer(tmp)
            writer.writerow(self.param_names)
            for row in theta_np:
                writer.writerow(row)
        return theta_np, samples_csv

    def _update_pool_with_results(self, params: np.ndarray, observables: np.ndarray) -> None:
        """Replace local pool for this scenario with the full HPC results."""
        # Remove old batches for this scenario before adding the complete set
        for batch in self.pool._scan_batches(scenario=self.scenario):
            batch_path = self.pool.pool_dir / batch["filename"]
            batch_path.unlink(missing_ok=True)
            self.logger.info(f"Removed stale batch: {batch['filename']}")
        self._info(f"Adding {params.shape[0]} simulations to pool (scenario='{self.scenario}')")
        self.pool.add_batch(
            params_matrix=params,
            observables_matrix=observables,
            seed=self.seed,
            scenario=self.scenario,
        )
        self._info(f"Complete: {observables.shape[0]} simulations finished")

    def __call__(self, batch_size: Union[int, Tuple[int, ...]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run QSP simulations and return parameter-observable pairs.

        Workflow:
        1. Check SimulationPoolManager for cached test statistics (scenario-specific)
        2. Check HPC for derived test statistics (scenario-specific)
        3. If not derived, check HPC for full simulations and derive
        4. If no full sims, run simulations on HPC, then derive
        5. Download and add to SimulationPoolManager, return results

        Args:
            batch_size: Number of simulations to run (int or tuple of ints)
                       If tuple, computes product of dimensions (e.g., (10,) -> 10, (10, 5) -> 50)

        Returns:
            Tuple of (params, observables):
            - params: numpy array of shape (batch_size, num_params)
            - observables: numpy array of shape (batch_size, num_observables)
        """
        # Handle both int and tuple batch_size (BayesFlow passes tuple)
        if isinstance(batch_size, tuple):
            num_simulations = int(np.prod(batch_size))
        else:
            num_simulations = int(batch_size)

        # Log simulation request
        self.logger.info(f"Simulation request: {num_simulations} simulations (seed={self.seed})")

        # 1. Check SimulationPoolManager for cached test statistics (fast path)
        n_available = self.pool.get_available_simulations(scenario=self.scenario)

        if n_available >= num_simulations:
            self.logger.info(f"✓ Using local pool: {n_available} simulations available")
            params, observables = self.pool.load_simulations(
                n_requested=num_simulations, scenario=self.scenario, random_state=self.rng
            )
            self.logger.info(f"Returning {params.shape[0]} simulations from local pool")
            return params, observables

        if n_available > 0:
            self.logger.info(
                f"Local pool has {n_available}/{num_simulations} simulations - checking HPC"
            )
        else:
            self.logger.info("No local simulations - checking HPC")

        if self.local_only:
            raise QSPSimulatorError(
                f"Local-only mode enabled but only {n_available}/{num_simulations} simulations available."
            )

        # Compute hash keys for HPC lookups
        priors_hash = self._compute_priors_hash()
        test_stats_hash = self._compute_test_stats_hash()

        # Get HPC pool path (scenario-specific)
        hpc_pool_path = f"{self.job_manager.config.simulation_pool_path}/{self.model_version}_{priors_hash[:HASH_PREFIX_LENGTH]}_{self.scenario}"
        self.logger.info(f"HPC pool path: {hpc_pool_path}")

        # 2. Check HPC for derived test statistics
        self.logger.info("Checking HPC for pre-derived test statistics...")
        try:
            has_test_stats = self.job_manager.check_hpc_test_stats(
                hpc_pool_path, test_stats_hash, expected_n_sims=num_simulations
            )
        except (RemoteCommandError, MissingOutputError) as exc:
            raise QSPSimulatorError(f"Failed checking HPC test stats: {exc}") from exc

        if has_test_stats:
            self.logger.info("✓ Found pre-derived test statistics on HPC")
            try:
                return self._download_and_add_to_pool(
                    hpc_pool_path, test_stats_hash, num_simulations
                )
            except Exception as exc:
                raise QSPSimulatorError(f"Failed downloading test stats from HPC: {exc}") from exc
        else:
            self.logger.info("No pre-derived test statistics found")

        # 3. Check HPC for full simulations (construct path explicitly to include scenario)
        self.logger.info("Checking HPC for full simulations...")
        try:
            # Use same path construction as when saving
            hpc_full_sim_path = f"{self.job_manager.config.simulation_pool_path}/{self.model_version}_{priors_hash[:HASH_PREFIX_LENGTH]}_{self.scenario}"
            has_full_sims = self.job_manager.result_collector.check_pool_directory_exists(
                hpc_full_sim_path
            )
            if has_full_sims:
                n_hpc_available = self.job_manager.result_collector.count_pool_simulations(
                    hpc_full_sim_path
                )
            else:
                n_hpc_available = 0
            has_full_sims = n_hpc_available >= num_simulations
        except (RemoteCommandError, MissingOutputError) as exc:
            raise QSPSimulatorError(f"Failed checking HPC full simulations: {exc}") from exc

        if has_full_sims:
            self.logger.info(f"✓ Found {n_hpc_available} full simulations on HPC (sufficient)")
            self.logger.info("Deriving test statistics from full simulations...")
            try:
                self._derive_test_statistics(
                    hpc_pool_path, test_stats_hash, num_simulations, self.verbose
                )
                return self._download_and_add_to_pool(
                    hpc_pool_path, test_stats_hash, num_simulations
                )
            except (RemoteCommandError, MissingOutputError, SubmissionError) as exc:
                raise QSPSimulatorError(
                    f"Failed deriving/downloading test stats from HPC: {exc}"
                ) from exc
        elif n_hpc_available > 0:
            self.logger.info(f"Found {n_hpc_available}/{num_simulations} full simulations on HPC")
        else:
            self.logger.info("No full simulations found on HPC")

        # 4. Run new full simulations on HPC
        n_needed = num_simulations - n_hpc_available
        if n_needed > 0:
            self.logger.info(f"Generating {n_needed} new simulations on HPC...")
            try:
                # Run new simulations and add to pool
                self._run_new_simulations(n_needed)

                # Now load the full requested amount from pool (old + new)
                self.logger.info(
                    f"Loading {num_simulations} total simulations from pool ({n_needed} newly generated)"
                )
                params, observables = self.pool.load_simulations(
                    n_requested=num_simulations, scenario=self.scenario, random_state=self.rng
                )
                self.logger.info(f"✓ Returning {params.shape[0]} simulations")
                return params, observables
            except (RemoteCommandError, MissingOutputError, SubmissionError) as exc:
                raise QSPSimulatorError(f"Failed running new simulations on HPC: {exc}") from exc

        # If we have some HPC sims but not enough, derive from what we have
        self.logger.info(
            f"Deriving test statistics from {n_hpc_available} available HPC simulations"
        )
        try:
            self._derive_test_statistics(
                hpc_pool_path, test_stats_hash, n_hpc_available, self.verbose
            )
            return self._download_and_add_to_pool(hpc_pool_path, test_stats_hash, n_hpc_available)
        except (RemoteCommandError, MissingOutputError, SubmissionError) as exc:
            raise QSPSimulatorError(f"Failed deriving/downloading partial HPC sims: {exc}") from exc

    def simulate_with_parameters(
        self, theta: np.ndarray, pool_suffix: str = "posterior_predictive"
    ) -> np.ndarray:
        """
        Run QSP simulations for given parameter samples with persistent caching.

        Uses a theta-hashed suffix pool so that identical parameter matrices
        produce cache hits without row-by-row matching. Each unique theta gets
        its own isolated pool; count-based lookup is safe because the hash
        guarantees the pool only contains sims for this exact theta.

        Workflow:
        1. Compute theta_hash from the parameter matrix
        2. Check local suffix pool (count-based — hash guarantees correctness)
        3. If not local_only: check HPC suffix pool for derived test stats
        4. If HPC hit: download, add to local suffix pool, return
        5. Run new sims via _run_matlab_simulation (with pool ID override)
        6. Add results to local suffix pool
        7. Return observables

        Args:
            theta: Parameter matrix (n_samples, n_params)
            pool_suffix: Suffix for pool identification (default: 'posterior_predictive')
                        Combined with theta hash to create unique pool per parameter set

        Returns:
            Test statistics array (n_samples, n_test_stats)
        """
        n_samples = theta.shape[0]
        self._info(f"QSP Simulator ({pool_suffix}): {n_samples} samples ({self.model_version})")

        # 1. Compute theta hash for pool isolation
        theta_hash = hashlib.sha256(theta.tobytes()).hexdigest()[:HASH_PREFIX_LENGTH]
        pool_scenario = f"{pool_suffix}_{theta_hash}"
        self._info(f"Theta hash: {theta_hash} → pool scenario: {pool_scenario}")

        # Get or create the suffix pool for this theta
        suffix_pool = self._get_or_create_suffix_pool(pool_scenario)

        # 2. Check local suffix pool (count-based — hash guarantees correctness)
        n_available = suffix_pool.get_available_simulations(scenario=pool_scenario)
        if n_available >= n_samples:
            self._info(f"Local suffix pool hit: {n_available} simulations available")
            _, observables = suffix_pool.load_simulations(
                n_requested=n_samples, scenario=pool_scenario, random_state=self.rng
            )
            return observables

        # 3. Check HPC for derived test stats in the suffix pool
        if not self.local_only:
            hpc_pool_id = self._compute_hpc_pool_id(scenario_override=pool_scenario)
            hpc_pool_path = f"{self.job_manager.config.simulation_pool_path}/{hpc_pool_id}"
            test_stats_hash = self._compute_test_stats_hash()

            try:
                has_test_stats = self.job_manager.check_hpc_test_stats(
                    hpc_pool_path, test_stats_hash, expected_n_sims=n_samples
                )
            except (RemoteCommandError, MissingOutputError):
                has_test_stats = False

            if has_test_stats:
                self._info("HPC suffix pool hit — downloading test statistics")
                with tempfile.TemporaryDirectory() as temp_dir:
                    params, test_stats = self.job_manager.download_test_stats(
                        hpc_pool_path, test_stats_hash, Path(temp_dir)
                    )
                if params is not None and test_stats is not None:
                    suffix_pool.add_batch(
                        params_matrix=params,
                        observables_matrix=test_stats,
                        seed=self.seed,
                        scenario=pool_scenario,
                    )
                    self._info(f"Added {params.shape[0]} sims to local suffix pool from HPC")
                    _, observables = suffix_pool.load_simulations(
                        n_requested=n_samples,
                        scenario=pool_scenario,
                        random_state=self.rng,
                    )
                    return observables

        # 4. Run new simulations with the provided parameter values
        self._info(f"Running {n_samples} new simulations...")

        if self.local_only:
            # Local mode: run MATLAB on this machine with parallel workers
            return self._run_local_with_params(theta, pool_scenario, suffix_pool)

        # HPC mode: submit to cluster
        # Create temporary CSV with provided parameters
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            samples_csv = tmp.name
            writer = csv.writer(tmp)
            writer.writerow(self.param_names)
            for row in theta:
                writer.writerow(row)

        try:
            # Run MATLAB simulations — use suffix pool ID so HPC also persists separately
            hpc_pool_id = self._compute_hpc_pool_id(scenario_override=pool_scenario)
            hpc_params, observables = self._run_matlab_simulation(
                samples_csv, n_samples, simulation_pool_id_override=hpc_pool_id
            )

            # 5. Persist to local suffix pool (use HPC params to stay aligned)
            suffix_pool.add_batch(
                params_matrix=hpc_params,
                observables_matrix=observables,
                seed=self.seed,
                scenario=pool_scenario,
            )

            self._info(f"Complete: {observables.shape[0]} simulations persisted to suffix pool")
            return observables

        finally:
            Path(samples_csv).unlink(missing_ok=True)

    def _run_local_with_params(
        self, theta: np.ndarray, pool_scenario: str, suffix_pool
    ) -> np.ndarray:
        """
        Run simulations locally with specific parameter values, derive test stats,
        cache in suffix pool, and return observables.

        Mirrors _run_matlab_simulation but uses local MATLAB processes instead of HPC.

        Args:
            theta: Parameter matrix (n_samples, n_params) in original space
            pool_scenario: Scenario name for suffix pool caching
            suffix_pool: SimulationPoolManager instance for caching

        Returns:
            Test statistics array (n_samples, n_test_stats)
        """
        import json

        import pandas as pd

        from qsp_hpc.batch.derive_test_stats_worker import (
            build_test_stat_registry,
            compute_test_statistics_batch,
        )
        from qsp_hpc.simulation.batch_runner import run_batch_worker

        n_samples = theta.shape[0]
        n_workers = min(self.n_local_workers, n_samples)

        sim_config = dict(self.sim_config) if self.sim_config else {}
        sim_config["accelerate_model"] = self.accelerate

        if n_workers <= 1:
            parquet_file = run_batch_worker(
                params=theta,
                param_names=self.param_names,
                model_script=self.model_script or "default_model",
                project_root=self.project_root,
                seed=self.seed,
                dosing=self.dosing,
                sim_config=sim_config,
                matlab_path=self.matlab_path,
                verbose=self.verbose,
            )
            species_df = pd.read_parquet(parquet_file)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            self._info(f"Splitting {n_samples} sims across {n_workers} local workers")
            chunk_size = n_samples // n_workers
            remainder = n_samples % n_workers
            chunks = []
            start = 0
            for i in range(n_workers):
                end = start + chunk_size + (1 if i < remainder else 0)
                chunks.append((i, theta[start:end], self.seed + i * 10000))
                start = end

            with tempfile.TemporaryDirectory(prefix="qsp_local_swp_") as temp_dir:
                temp_path = Path(temp_dir)
                parquet_files = []
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(
                            _run_worker_batch,
                            wid,
                            wparams,
                            wseed,
                            self.param_names,
                            self.model_script or "default_model",
                            self.project_root,
                            self.dosing,
                            sim_config,
                            self.matlab_path,
                            temp_path,
                        ): wid
                        for wid, wparams, wseed in chunks
                    }
                    for future in as_completed(futures):
                        wid = futures[future]
                        parquet_files.append((wid, future.result()))
                        self._info(f"Local worker {wid} complete")

                dfs = []
                for wid, pq_file in sorted(parquet_files, key=lambda x: x[0]):
                    df = pd.read_parquet(pq_file)
                    base_id = sum(chunks[i][1].shape[0] for i in range(wid))
                    if "patient_id" in df.columns:
                        df["patient_id"] = df["patient_id"] + base_id
                    dfs.append(df)
                species_df = pd.concat(dfs, ignore_index=True)

        # Derive test statistics
        test_stats_df = pd.read_csv(self.test_stats_csv)
        test_stat_registry = build_test_stat_registry(test_stats_df)

        species_units = {}
        if self.species_units_file is not None and self.species_units_file.exists():
            with open(self.species_units_file, "r") as f:
                species_units = json.load(f)

        observables = compute_test_statistics_batch(
            species_df, test_stats_df, test_stat_registry, species_units
        )

        # Cache in suffix pool
        suffix_pool.add_batch(
            params_matrix=theta,
            observables_matrix=observables,
            seed=self.seed,
            scenario=pool_scenario,
        )
        self._info(f"Complete: {observables.shape[0]} local simulations persisted to suffix pool")
        return observables

    def _compute_priors_hash(self) -> str:
        """
        Compute hash of priors configuration (excludes test statistics).

        Returns:
            SHA256 hash (full hex string)
        """
        hasher = hashlib.sha256()

        # Hash priors CSV content
        priors_content = self.priors_csv.read_text()
        hasher.update(priors_content.encode("utf-8"))

        # Hash model script name
        hasher.update(self.model_script.encode("utf-8"))

        # Hash model version
        hasher.update(self.model_version.encode("utf-8"))

        return hasher.hexdigest()

    def _compute_test_stats_hash(self) -> str:
        """
        Compute hash of test statistics CSV.

        Returns:
            SHA256 hash (full hex string)
        """
        test_stats_content = self.test_stats_csv.read_text()
        return hashlib.sha256(test_stats_content.encode("utf-8")).hexdigest()

    def _compute_hpc_pool_id(self, scenario_override: Optional[str] = None) -> str:
        """
        Compute the HPC pool identifier string.

        Args:
            scenario_override: If provided, use this instead of self.scenario
                              (e.g., for suffix pools like 'posterior_predictive_a3f7c2e1')

        Returns:
            Pool ID string like '{model_version}_{priors_hash[:8]}_{scenario}'
        """
        priors_hash = self._compute_priors_hash()
        scenario = scenario_override if scenario_override is not None else self.scenario
        return f"{self.model_version}_{priors_hash[:HASH_PREFIX_LENGTH]}_{scenario}"

    def _get_or_create_suffix_pool(self, pool_scenario: str) -> "SimulationPoolManager":
        """
        Lazily create and cache a SimulationPoolManager for a suffix pool.

        Each unique pool_scenario gets its own local directory under cache_dir.

        Args:
            pool_scenario: Scenario string for the suffix pool
                          (e.g., 'posterior_predictive_a3f7c2e1')

        Returns:
            SimulationPoolManager for the suffix pool
        """
        if not hasattr(self, "_suffix_pools"):
            self._suffix_pools: Dict[str, SimulationPoolManager] = {}
        if pool_scenario not in self._suffix_pools:
            pool_kwargs = dict(
                cache_dir=self.cache_dir,
                model_version=self.model_version,
                model_description=self.model_description,
                priors_csv=self.priors_csv,
                model_script=self.model_script,
                scenario=pool_scenario,
            )
            if self._calibration_targets_dir is not None:
                pool_kwargs["calibration_targets"] = self._calibration_targets_dir
            else:
                pool_kwargs["test_stats_csv"] = self.test_stats_csv
            self._suffix_pools[pool_scenario] = SimulationPoolManager(**pool_kwargs)
        return self._suffix_pools[pool_scenario]

    def _run_matlab_simulation(
        self,
        samples_csv: str,
        num_simulations: int,
        simulation_pool_id_override: Optional[str] = None,
    ) -> np.ndarray:
        """
        Run MATLAB simulation for parameter samples on HPC.

        By default, full simulations go to the shared pool (no suffix).
        When simulation_pool_id_override is provided, simulations go to
        a separate HPC pool (e.g., for posterior predictive checks).

        Uses Python job submission via HPCJobManager for efficient batch execution.

        Args:
            samples_csv: Path to CSV file with parameter samples
            num_simulations: Number of simulations
            simulation_pool_id_override: If provided, use this as the HPC pool ID
                                        instead of the default scenario-based ID

        Returns:
            Numpy array of observables (num_simulations x num_observables)
        """
        # Validate SSH connection before submitting jobs
        self._validate_hpc_connection()

        # Compute simulation pool ID
        if simulation_pool_id_override is not None:
            simulation_pool_id = simulation_pool_id_override
        else:
            simulation_pool_id = self._compute_hpc_pool_id()

        # Log HPC save path
        hpc_save_path = f"{self.job_manager.config.simulation_pool_path}/{simulation_pool_id}"
        self.logger.info(f"   → HPC save path: {hpc_save_path} (scenario='{self.scenario}')")

        # Calculate optimal split across tasks
        jobs_per_chunk, n_tasks = calculate_batch_split(num_simulations, self.max_tasks)
        self.logger.info(
            f"   → Splitting {num_simulations} simulations into {n_tasks} tasks ({jobs_per_chunk} sims/task)"
        )

        # Submit jobs via Python (no MATLAB startup!)
        job_info = self.job_manager.submit_jobs(
            samples_csv=samples_csv,
            test_stats_csv=str(self.test_stats_csv),
            model_script=self.model_script,
            num_simulations=num_simulations,
            seed=self.seed,
            jobs_per_chunk=jobs_per_chunk,
            skip_sync=False,  # Sync codebase first
            save_full_simulations=True,  # Enable full simulation saving
            simulation_pool_id=simulation_pool_id,  # Pool ID for HPC storage
            sim_config=self.sim_config,  # Simulation settings (stop_time, solver, etc.)
            dosing=self.dosing,  # Treatment dosing schedule (if any)
        )

        # Wait for MATLAB simulation jobs to complete (saves full sims to parquet)
        self._wait_for_completion(job_info.job_ids, num_simulations)

        # Clean up state file from MATLAB jobs
        Path(job_info.state_file).unlink(missing_ok=True)

        # Now derive test statistics from the saved parquet files using Python
        # This is the correct flow: MATLAB saves full sims → Python computes test stats
        self.logger.info("Deriving test statistics from full simulations...")
        test_stats_hash = self._compute_test_stats_hash()
        self._derive_test_statistics(hpc_save_path, test_stats_hash, num_simulations)

        # Download the derived test statistics AND parameters from HPC.
        # The derivation processes ALL batches in the pool (old + new), so we
        # return the full set to keep params/observables aligned.
        params_matrix, observables_matrix = self._download_derived_test_stats(
            hpc_save_path, test_stats_hash, num_simulations
        )

        return params_matrix, observables_matrix

    def extract_trajectory_grid(
        self,
        species_list: Union[list, str] = "all",
        time_grid: Union[list, str] = "daily",
        output_subdir: str = "trajectory_grid",
        local_dest: Optional[Path] = None,
        scenario_override: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Extract dense species × timepoint grid from HPC simulation pool.

        Computes the HPC pool path from the current simulator config, submits
        a SLURM trajectory grid extraction job, waits for completion, and
        downloads the resulting Parquet DataFrame.

        Args:
            species_list: Species to extract — list of names or "all"
            time_grid: Timepoints in days — list of floats or "daily"
            output_subdir: Subdirectory name within pool for output
            local_dest: Local directory to download grid into.
                        Defaults to cache_dir / "trajectory_grids" / pool_id
            scenario_override: If provided, use this pool scenario instead of self.scenario

        Returns:
            Tuple of (grid_df, meta):
                grid_df: DataFrame with columns like "V_T.CD8__t14.0", one row per simulation
                meta: Dict with species list, time grid, shape info
        """
        self._validate_hpc_connection()

        # Compute HPC pool path
        pool_id = self._compute_hpc_pool_id(scenario_override=scenario_override)
        hpc_pool_path = f"{self.job_manager.config.simulation_pool_path}/{pool_id}"

        # Resolve stop_time from sim_config
        stop_time = 21.0
        if self.sim_config and "stop_time" in self.sim_config:
            stop_time = float(self.sim_config["stop_time"])

        scenario_name = scenario_override or self.scenario or "default"

        # Submit trajectory grid extraction job
        self.logger.info(f"Submitting trajectory grid extraction for pool {pool_id}...")
        job_id = self.job_manager.submit_trajectory_grid_job(
            pool_path=hpc_pool_path,
            species_list=species_list,
            time_grid=time_grid,
            output_subdir=output_subdir,
            scenario_name=scenario_name,
            stop_time=stop_time,
        )

        # Wait for completion (single SLURM task)
        self._wait_for_completion([job_id], 1)

        # Download results
        if local_dest is None:
            local_dest = self.cache_dir / "trajectory_grids" / pool_id
        local_dest = Path(local_dest)

        grid_df, meta = self.job_manager.download_trajectory_grid(
            pool_path=hpc_pool_path,
            output_subdir=output_subdir,
            local_dest=local_dest,
        )

        self.logger.info(
            f"Trajectory grid ready: {grid_df.shape[0]} sims × {grid_df.shape[1]} features"
        )
        return grid_df, meta

    def _validate_hpc_connection(self) -> None:
        """
        Validate SSH connection to HPC cluster (fast fail).

        Raises:
            RuntimeError: If SSH connection cannot be established
        """
        try:
            # Validate SSH connection (typically 1-2s, but allow up to 15s for slow connections)
            self.job_manager.validate_ssh_connection(timeout=15)

        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except Exception as e:
            raise RuntimeError(f"HPC connection validation failed: {e}")

    def _wait_for_completion(self, job_ids: List[str], num_simulations: int):
        """
        Wait for HPC jobs to complete with progress updates.

        Args:
            job_ids: List of SLURM job IDs to monitor
            num_simulations: Expected number of simulations
        """
        # Give SLURM time to register the jobs before first check
        time.sleep(SLURM_REGISTRATION_DELAY)

        start_time = time.time()
        max_tasks_seen = 0  # Track the maximum number of tasks we've seen

        while True:
            # Check status of all jobs
            total_status = {"completed": 0, "running": 0, "pending": 0, "failed": 0}

            for job_id in job_ids:
                try:
                    status = self.job_manager.check_job_status(job_id)
                    for key in total_status:
                        total_status[key] += status[key]
                except Exception as e:
                    if self.verbose:
                        self.logger.warning(f"Could not check status for job {job_id}: {e}")

            # Calculate progress
            total_tasks = sum(total_status.values())
            max_tasks_seen = max(max_tasks_seen, total_tasks)

            if total_tasks > 0:
                completed_pct = (total_status["completed"] / total_tasks) * 100
            else:
                completed_pct = 0

            # Show progress after every check
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

            self.logger.info(
                f"  {total_status['completed']}/{total_tasks} done ({completed_pct:.1f}%) | "
                f"Running: {total_status['running']}, Pending: {total_status['pending']}, "
                f"Failed: {total_status['failed']} | {elapsed_str}"
            )

            # Check if all jobs completed or failed
            active_jobs = total_status["running"] + total_status["pending"]

            # Break conditions:
            # 1. We've seen tasks and they're all done (completed + failed = total)
            if total_tasks > 0 and active_jobs == 0:
                if total_status["failed"] > 0:
                    print(f"  Warning: {total_status['failed']} task(s) failed")
                break
            # 2. No tasks visible but we previously saw them (they completed and disappeared)
            elif total_tasks == 0 and max_tasks_seen > 0 and elapsed > 30:
                if self.verbose:
                    print(f"  All {max_tasks_seen} jobs completed (no longer visible in queue)")
                break
            # 3. Waited a while and never saw any tasks (possible monitoring failure, proceed anyway)
            elif total_tasks == 0 and elapsed > JOB_QUEUE_TIMEOUT:
                self.logger.warning(
                    f"No jobs visible in queue after {JOB_QUEUE_TIMEOUT}s - proceeding to download"
                )
                break

            # Check timeout
            if self.max_wait_time and (time.time() - start_time) > self.max_wait_time:
                raise TimeoutError(
                    f"Job monitoring timeout after {self.max_wait_time}s. "
                    f"Jobs may still be running on HPC."
                )

            # Wait before next check
            time.sleep(self.poll_interval)


def get_observed_data(
    test_stats_csv: Optional[Union[str, Path]] = None,
    calibration_targets: Optional[Union[str, Path]] = None,
    value_column: str = "median",
) -> Dict[str, np.ndarray]:
    """
    Extract observed data from test statistics CSV or calibration target YAMLs.

    Accepts either a CSV path or a directory of calibration target YAMLs.
    Exactly one must be provided.

    Args:
        test_stats_csv: Path to test statistics CSV file
        calibration_targets: Path to directory of calibration target YAML files
        value_column: Column name to use for observed values (default: 'median')

    Returns:
        Dictionary with observable names as keys and 2D numpy arrays as values.
        Each array has shape (1, 1) for compatibility with BayesFlow workflow.

    Example:
        obs = get_observed_data(test_stats_csv='cache/test_stats.csv')
        obs = get_observed_data(calibration_targets='calibration_targets/')
        posterior_samples = workflow.sample(conditions=obs, num_samples=1000)
    """
    import pandas as pd

    if test_stats_csv is not None and calibration_targets is not None:
        raise ValueError("Provide test_stats_csv OR calibration_targets, not both")

    if test_stats_csv is None and calibration_targets is None:
        raise ValueError("Must provide either test_stats_csv or calibration_targets")

    if calibration_targets is not None:
        from qsp_hpc.calibration import load_calibration_targets

        df = load_calibration_targets(Path(calibration_targets))
    else:
        test_stats_csv = Path(test_stats_csv)
        if not test_stats_csv.exists():
            raise FileNotFoundError(f"Test statistics CSV not found: {test_stats_csv}")
        df = pd.read_csv(test_stats_csv)

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' not found in test statistics CSV. "
            f"Available columns: {', '.join(df.columns)}"
        )

    if "test_statistic_id" not in df.columns:
        raise ValueError(
            f"Column 'test_statistic_id' not found in test statistics CSV. "
            f"Available columns: {', '.join(df.columns)}"
        )

    # Extract observable names and values
    observable_names = df["test_statistic_id"].tolist()
    observed_values = df[value_column].values

    # Build dictionary with 2D arrays (1, 1) for each observable
    obs_dict = {}
    for i, obs_name in enumerate(observable_names):
        obs_dict[obs_name] = observed_values[i : i + 1].reshape(1, 1)

    return obs_dict


def qsp_simulator(
    priors_csv: Union[str, Path],
    test_stats_csv: Optional[Union[str, Path]] = None,
    calibration_targets: Optional[Union[str, Path]] = None,
    model_script: str = "",
    model_version: str = "v1",
    model_description: str = "",
    scenario: str = "default",
    cache_dir: Union[str, Path] = "cache/sbi_simulations",
    seed: int = 2025,
    cache_sampling_seed: Optional[int] = None,
    max_tasks: int = 10,
    poll_interval: int = 30,
    max_wait_time: Optional[int] = None,
    verbose: bool = False,
) -> QSPSimulator:
    """
    Create a QSP simulator for SBI workflows.

    This function returns a callable simulator that can be used with
    sbi.inference.simulate_for_sbi to generate training data for neural
    density estimators.

    Simulations are cached in pools organized by model version, configuration, and scenario.
    Multiple runs with the same configuration will accumulate and reuse simulations
    from the pool, enabling efficient iteration and exploration.

    Args:
        priors_csv: Path to priors CSV defining parameter names and distributions
        test_stats_csv: Path to test statistics CSV defining observables
        calibration_targets: Path to directory of calibration target YAML files
                            (alternative to test_stats_csv; cannot use both)
        model_script: MATLAB model script name (e.g., 'immune_oncology_model_PDAC')
        model_version: Descriptive version name (e.g., 'baseline_gvax')
        model_description: Brief description of model configuration
        scenario: Scenario name for therapy protocol (e.g., 'gvax', 'control', 'gvax_anti_pd1')
        cache_dir: Base directory for simulation pools (default: cache/sbi_simulations)
        seed: Random seed for reproducibility (used for new simulations)
        cache_sampling_seed: Fixed seed for reproducible cache sampling (default: None, uses seed)
                            Set to fixed value to ensure consistent correlations across runs
        max_tasks: Maximum number of parallel SLURM array tasks (default: 10)
        poll_interval: Seconds between job status checks on HPC (default: 30)
        max_wait_time: Maximum wait time in seconds before timeout (default: None, no timeout)
        verbose: If True, print detailed progress information (default: False)

    Returns:
        Callable simulator compatible with simulate_for_sbi

    Example:
        from sbi.inference import simulate_for_sbi
        from qsp_hpc.simulation.qsp_simulator import qsp_simulator
        from qsp_hpc.priors.load_sbi_priors import load_prior

        # Load prior
        priors_csv = 'projects/pdac_2025/cache/pdac_sbi_priors.csv'
        prior = load_prior(priors_csv)

        # Create simulator with calibration target YAMLs
        simulator = qsp_simulator(
            priors_csv=priors_csv,
            calibration_targets='calibration_targets/',
            model_script='immune_oncology_model_PDAC',
            model_version='baseline_gvax',
            model_description='PDAC baseline: 8 params, 12 obs, GVAX',
            max_tasks=20,
            poll_interval=60  # Check every minute
        )

        # Generate training data (will reuse from pool if available)
        theta, x = simulate_for_sbi(simulator, prior, num_simulations=100)
    """
    return QSPSimulator(
        test_stats_csv=test_stats_csv,
        calibration_targets=calibration_targets,
        priors_csv=priors_csv,
        model_script=model_script,
        model_version=model_version,
        model_description=model_description,
        scenario=scenario,
        cache_dir=cache_dir,
        seed=seed,
        cache_sampling_seed=cache_sampling_seed,
        max_tasks=max_tasks,
        poll_interval=poll_interval,
        max_wait_time=max_wait_time,
        verbose=verbose,
    )
