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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from qsp_hpc.batch.batch_utils import calculate_batch_split
from qsp_hpc.batch.hpc_job_manager import MissingOutputError, RemoteCommandError, SubmissionError
from qsp_hpc.constants import HASH_PREFIX_LENGTH, JOB_QUEUE_TIMEOUT, SLURM_REGISTRATION_DELAY
from qsp_hpc.simulation.simulation_pool import SimulationPoolManager
from qsp_hpc.utils.logging_config import setup_logger


class QSPSimulatorError(RuntimeError):
    """Raised when simulator orchestration fails (wraps lower-level errors)."""


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
        test_stats_csv: Union[str, Path],
        priors_csv: Union[str, Path],
        project_name: str,
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
        matlab_runner: Optional[Callable[..., np.ndarray]] = None,
        local_only: bool = False,
    ):
        """
        Initialize QSP simulator.

        Args:
            test_stats_csv: Path to test statistics CSV defining observables
            priors_csv: Path to priors CSV defining parameter names and distributions
            project_name: Project identifier (e.g., 'pdac_2025') used for organizing HPC files
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
        self.test_stats_csv = Path(test_stats_csv)
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
        self.project_name = project_name

        if not self.test_stats_csv.exists():
            raise FileNotFoundError(f"Test statistics CSV not found: {self.test_stats_csv}")

        if not self.priors_csv.exists():
            raise FileNotFoundError(f"Priors CSV not found: {self.priors_csv}")

        # Load parameter names from priors CSV
        with open(self.priors_csv, "r") as f:
            reader = csv.DictReader(f)
            self.param_names = [row["name"] for row in reader]

        # Initialize SimulationPoolManager for local caching
        self.pool = pool or SimulationPoolManager(
            cache_dir=cache_dir,
            model_version=model_version,
            model_description=model_description,
            priors_csv=priors_csv,
            test_stats_csv=test_stats_csv,
            model_script=model_script,
        )

        # Store job_manager for lazy initialization (don't create until needed)
        self._job_manager = job_manager  # May be None or injected for testing

        # Optional injected MATLAB runner for testing/mocking
        self._matlab_runner = matlab_runner

        # Set up logging
        self.logger = setup_logger(__name__, verbose=self.verbose)

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

            self._info("Downloading test statistics from HPC...")
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

            # Add to SimulationPoolManager
            self._info(f"Adding {params.shape[0]} simulations to pool (scenario='{self.scenario}')")
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

        self._info(f"Complete: Returning {params_out.shape[0]} samples")
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
            self._info("Test statistics already derived")
            return

        # Need to derive test statistics
        self._info("Deriving test statistics from full simulations...")

        # Launch derivation job (with selective batch processing)
        job_id = self.job_manager.submit_derivation_job(
            hpc_pool_path,
            str(self.test_stats_csv),
            test_stats_hash,
            project_name=self.project_name,
            num_simulations=num_simulations,
        )

        # Wait for derivation job to complete
        self._info("Waiting for derivation to complete...")
        self._wait_for_completion([job_id], num_simulations)
        self._info("Derivation complete")

        # Verify that test stats were actually created
        has_test_stats_now = self.job_manager.check_hpc_test_stats(hpc_pool_path, test_stats_hash)

        if not has_test_stats_now:
            # Derivation failed - check logs
            log_path = f"{self.job_manager.config.remote_project_path}/projects/{self.project_name}/batch_jobs/logs"
            log_cmd = f'ls -lt "{log_path}"/qsp_derive_*.err 2>/dev/null | head -3'
            status, log_output = self.job_manager.transport.exec(log_cmd)

            if status == 0 and log_output.strip():
                self._error("Recent derivation error logs:")
                self._error(f"  {log_output.strip()}")

            raise RuntimeError(
                f"Derivation job completed but did not produce test statistics. "
                f"Check HPC logs at: {log_path}/qsp_derive_*.err"
            )

    def _run_new_simulations(self, num_simulations: int) -> None:
        """
        Run new simulations on HPC and add to pool.

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
            observables_np = self._run_matlab_simulation(samples_csv, num_simulations)
            self._update_pool_with_results(theta_np, observables_np)
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
        """Add completed simulations to the local pool."""
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

        self._info(
            f"QSP Simulator: {num_simulations} simulations (scenario='{self.scenario}', version={self.model_version})"
        )

        # 1. Check SimulationPoolManager for cached test statistics (fast path)
        n_available = self.pool.get_available_simulations(scenario=self.scenario)
        if n_available >= num_simulations:
            if self.verbose:
                self._info(f"  Found {n_available} simulations in local pool (sufficient)")
            params, observables = self.pool.load_simulations(
                n_requested=num_simulations, scenario=self.scenario, random_state=self.rng
            )
            return params, observables

        if self.local_only:
            raise QSPSimulatorError(
                f"Local-only mode enabled but only {n_available}/{num_simulations} simulations available."
            )

        if self.verbose:
            self._info(
                f"  Local pool has {n_available}/{num_simulations} simulations - checking HPC"
            )

        # Compute hash keys for HPC lookups
        priors_hash = self._compute_priors_hash()
        test_stats_hash = self._compute_test_stats_hash()

        # Get HPC pool path (scenario-specific)
        hpc_pool_path = f"{self.job_manager.config.simulation_pool_path}/{self.model_version}_{priors_hash[:HASH_PREFIX_LENGTH]}_{self.scenario}"

        # 2. Check HPC for derived test statistics
        try:
            has_test_stats = self.job_manager.check_hpc_test_stats(
                hpc_pool_path, test_stats_hash, expected_n_sims=num_simulations
            )
        except (RemoteCommandError, MissingOutputError) as exc:
            raise QSPSimulatorError(f"Failed checking HPC test stats: {exc}") from exc

        if has_test_stats:
            self._info("Found derived test statistics on HPC")
            try:
                return self._download_and_add_to_pool(
                    hpc_pool_path, test_stats_hash, num_simulations
                )
            except Exception as exc:
                raise QSPSimulatorError(f"Failed downloading test stats from HPC: {exc}") from exc

        # 3. Check HPC for full simulations (construct path explicitly to include scenario)
        try:
            # Use same path construction as when saving (line 463)
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
            self._info(
                f"Found {n_hpc_available} full simulations on HPC - deriving test statistics"
            )
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

        # 4. Run new full simulations on HPC
        n_needed = num_simulations - n_hpc_available
        if n_needed > 0:
            self._info(f"Running {n_needed} new simulations on HPC (scenario='{self.scenario}')")
            try:
                # Run new simulations and add to pool
                self._run_new_simulations(n_needed)

                # Now load the full requested amount from pool (old + new)
                self._info(
                    f"Loading {num_simulations} simulations from pool after adding {n_needed} new ones"
                )
                params, observables = self.pool.load_simulations(
                    n_requested=num_simulations, scenario=self.scenario, random_state=self.rng
                )
                self._info(
                    f"  Loaded {params.shape[0]} parameter sets and {observables.shape[0]} observable sets"
                )
                return params, observables
            except (RemoteCommandError, MissingOutputError, SubmissionError) as exc:
                raise QSPSimulatorError(f"Failed running new simulations on HPC: {exc}") from exc

        # If we have some HPC sims but not enough, derive from what we have
        self._info(f"Deriving test statistics from {n_hpc_available} HPC simulations")
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
        Run QSP simulations for given parameter samples with caching.

        This method provides the same caching and pooling benefits as __call__(),
        but accepts pre-specified parameter values (e.g., posterior samples).

        Workflow:
        1. Check local cache for this parameter set + pool suffix
        2. Check HPC for full simulations in this pool
        3. Check HPC for derived test statistics
        4. Download and cache if available
        5. Run new simulations only if needed

        Args:
            theta: Parameter matrix (n_samples, n_params)
            pool_suffix: Suffix for pool identification (default: 'posterior_predictive')
                        This creates a separate pool for posterior predictive sims
                        while still allowing reuse across multiple PPC runs

        Returns:
            Test statistics array (n_samples, n_test_stats)
        """
        n_samples = theta.shape[0]
        self._info(f"QSP Simulator ({pool_suffix}): {n_samples} samples ({self.model_version})")

        # Run new simulations with the provided parameter values
        #
        # NOTE: We do NOT check the shared HPC pool here because this function is called
        # with SPECIFIC parameter values (e.g., posterior samples) that must be simulated
        # exactly as provided. The shared pool contains simulations from prior sampling
        # which have different parameter values and cannot be reused.
        #
        # Use __call__(n_samples) instead if you want to sample from prior and reuse existing.
        self._info(f"Running {n_samples} new simulations...")

        # Create temporary CSV with provided parameters
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            samples_csv = tmp.name
            writer = csv.writer(tmp)
            writer.writerow(self.param_names)
            for row in theta:
                writer.writerow(row)

        try:
            # Run MATLAB simulations - they go to shared pool (no suffix)
            observables = self._run_matlab_simulation(samples_csv, n_samples)

            # Caching handled by SimulationPoolManager

            self._info(f"Complete: {observables.shape[0]} simulations finished")
            return observables

        finally:
            Path(samples_csv).unlink(missing_ok=True)

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

    def _run_matlab_simulation(self, samples_csv: str, num_simulations: int) -> np.ndarray:
        """
        Run MATLAB simulation for parameter samples on HPC.

        Full simulations are ALWAYS saved to the shared pool (no suffix).
        This allows training, prior PPCs, and posterior PPCs to reuse the same
        expensive full simulations. Test statistics are cached separately by caller.

        Uses Python job submission via HPCJobManager for efficient batch execution.

        Args:
            samples_csv: Path to CSV file with parameter samples
            num_simulations: Number of simulations

        Returns:
            Numpy array of observables (num_simulations x num_observables)
        """
        # Validate SSH connection before submitting jobs
        self._validate_hpc_connection()

        # Compute simulation pool ID for full simulations
        # IMPORTANT: Full simulations always go to the SHARED pool (no suffix)
        # This allows training, prior PPCs, and posterior PPCs to reuse the same expensive simulations
        priors_hash = self._compute_priors_hash()
        simulation_pool_id = (
            f"{self.model_version}_{priors_hash[:HASH_PREFIX_LENGTH]}_{self.scenario}"
        )

        # NOTE: We do NOT append pool_suffix here - full sims are shared
        # The suffix is only used for local test stats caching

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
            project_name=self.project_name,
            seed=self.seed,
            jobs_per_chunk=jobs_per_chunk,
            skip_sync=False,  # Sync codebase first
            save_full_simulations=True,  # Enable full simulation saving
            simulation_pool_id=simulation_pool_id,  # Pool ID for HPC storage
        )

        # Wait for jobs to complete
        self._wait_for_completion(job_info.job_ids, num_simulations)

        # Collect results via Python (no MATLAB needed!)
        observables_matrix = self.job_manager.collect_results(state_file=job_info.state_file)

        return observables_matrix

    def _validate_hpc_connection(self) -> None:
        """
        Validate SSH connection to HPC cluster (fast fail).

        Raises:
            RuntimeError: If SSH connection cannot be established
        """
        try:
            # Validate SSH connection (fast - should return in 1-2s)
            self.job_manager.validate_ssh_connection(timeout=5)

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
    test_stats_csv: Union[str, Path], value_column: str = "mean"
) -> Dict[str, np.ndarray]:
    """
    Extract observed data from test statistics CSV as dictionary.

    Args:
        test_stats_csv: Path to test statistics CSV file
        value_column: Column name to use for observed values (default: 'mean')

    Returns:
        Dictionary with observable names as keys and 2D numpy arrays as values.
        Each array has shape (1, 1) for compatibility with BayesFlow workflow.

    Example:
        obs = get_observed_data('projects/pdac_2025/cache/test_stats.csv')
        posterior_samples = workflow.sample(conditions=obs, num_samples=1000)
    """
    import pandas as pd

    test_stats_csv = Path(test_stats_csv)
    if not test_stats_csv.exists():
        raise FileNotFoundError(f"Test statistics CSV not found: {test_stats_csv}")

    # Read CSV
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
    test_stats_csv: Union[str, Path],
    priors_csv: Union[str, Path],
    model_script: str = "",
    model_version: str = "v1",
    model_description: str = "",
    scenario: str = "default",
    cache_dir: Union[str, Path] = "cache/sbi_simulations",
    project_name: Optional[str] = None,
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
        test_stats_csv: Path to test statistics CSV defining observables
        priors_csv: Path to priors CSV defining parameter names and distributions
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

        # Create simulator (for HPC with max 20 parallel tasks)
        simulator = qsp_simulator(
            test_stats_csv='projects/pdac_2025/cache/test_stats.csv',
            priors_csv=priors_csv,
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
        priors_csv=priors_csv,
        model_script=model_script,
        model_version=model_version,
        model_description=model_description,
        scenario=scenario,
        cache_dir=cache_dir,
        project_name=project_name,
        seed=seed,
        cache_sampling_seed=cache_sampling_seed,
        max_tasks=max_tasks,
        poll_interval=poll_interval,
        max_wait_time=max_wait_time,
        verbose=verbose,
    )
