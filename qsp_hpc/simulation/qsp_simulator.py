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
import tempfile
import subprocess
import numpy as np
import time
import re
import yaml
import hashlib
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List
from scipy.io import loadmat


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
        model_script: str = '',
        model_version: str = 'v1',
        model_description: str = '',
        scenario: str = 'default',
        cache_dir: Union[str, Path] = 'projects/pdac_2025/cache/sbi_simulations',
        seed: int = 2025,
        cache_sampling_seed: Optional[int] = None,
        max_tasks: int = 10,
        poll_interval: int = 30,
        max_wait_time: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize QSP simulator.

        Args:
            test_stats_csv: Path to test statistics CSV defining observables
            priors_csv: Path to priors CSV defining parameter names and distributions
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
        self.batch_config = None  # Loaded lazily when needed

        if not self.test_stats_csv.exists():
            raise FileNotFoundError(f"Test statistics CSV not found: {self.test_stats_csv}")

        if not self.priors_csv.exists():
            raise FileNotFoundError(f"Priors CSV not found: {self.priors_csv}")

        # Load parameter names from priors CSV
        # Load parameter names from priors CSV (inline to avoid dependencies)
        import csv
        with open(self.priors_csv, 'r') as f:
            reader = csv.DictReader(f)
            self.param_names = [row['name'] for row in reader]

        # Initialize SimulationPoolManager for local caching
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager
        self.pool = SimulationPoolManager(
            cache_dir=cache_dir,
            model_version=model_version,
            model_description=model_description,
            priors_csv=priors_csv,
            test_stats_csv=test_stats_csv,
            model_script=model_script
        )

        # Random number generator for sampling from pool (use fixed seed for reproducibility)
        self.rng = np.random.default_rng(self.cache_sampling_seed)

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
        param_names = priors_df['name'].tolist()
        dist_types = priors_df['distribution'].tolist()
        dist_param1 = priors_df['dist_param1'].values
        dist_param2 = priors_df['dist_param2'].values

        # Create RNG for parameter generation (use self.seed for new simulations)
        rng = np.random.default_rng(self.seed)

        # Generate samples
        samples = np.zeros((n_samples, len(param_names)))
        for i in range(len(param_names)):
            if dist_types[i] == 'lognormal':
                samples[:, i] = rng.lognormal(
                    mean=dist_param1[i],
                    sigma=dist_param2[i],
                    size=n_samples
                )
            else:
                raise ValueError(f"Unsupported distribution: {dist_types[i]}")

        return samples

    def _download_and_add_to_pool(
        self,
        hpc_pool_path: str,
        test_stats_hash: str,
        num_simulations: int
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
        from qsp_hpc.batch.hpc_job_manager import HPCJobManager
        job_manager = HPCJobManager()

        # Create temporary directory for download
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_cache_dir = Path(temp_dir)

            print(f"Downloading test statistics from HPC...")
            params, test_stats = job_manager.download_test_stats(
                hpc_pool_path, test_stats_hash, temp_cache_dir
            )

            if params is None:
                raise RuntimeError(
                    f"HPC has test statistics but no parameters. "
                    f"This indicates older simulation data that is no longer supported. "
                    f"Please re-run simulations."
                )

            # Add to SimulationPoolManager
            print(f"Adding {params.shape[0]} simulations to pool (scenario='{self.scenario}')")
            self.pool.add_batch(
                params_matrix=params,
                observables_matrix=test_stats,
                seed=self.seed,
                scenario=self.scenario
            )

        # Load from pool (handles sampling if needed)
        params_out, obs_out = self.pool.load_simulations(
            n_requested=num_simulations,
            scenario=self.scenario,
            random_state=self.rng
        )

        print(f"Complete: Returning {params_out.shape[0]} samples")
        return params_out, obs_out

    def _check_hpc_simulations(
        self,
        num_simulations: int,
        priors_hash: str,
        verbose: bool = False
    ) -> Tuple[bool, str, int]:
        """Check HPC for existing full simulations."""
        from qsp_hpc.batch.hpc_job_manager import HPCJobManager
        job_manager = HPCJobManager()

        hpc_pool_id = f"{self.model_version}_{priors_hash[:8]}"
        print(f"HPC pool: {hpc_pool_id}")

        has_full_sims, hpc_pool_path, n_available = job_manager.check_hpc_full_simulations(
            self.model_version, priors_hash, num_simulations
        )

        if has_full_sims:
            print(f"Found {n_available} HPC simulations (sufficient)")
        elif n_available > 0:
            print(f"Found {n_available}/{num_simulations} HPC simulations (need more)")
        else:
            if verbose:
                print(f"No HPC simulations found")

        return has_full_sims, hpc_pool_path, n_available


    def _derive_test_statistics(
        self,
        hpc_pool_path: str,
        test_stats_hash: str,
        num_simulations: int,
        verbose: bool = False
    ):
        """Derive test statistics from full simulations on HPC."""
        from qsp_hpc.batch.hpc_job_manager import HPCJobManager
        job_manager = HPCJobManager()

        # Validate that derived test stats match requested simulation count
        has_test_stats = job_manager.check_hpc_test_stats(
            hpc_pool_path, test_stats_hash, expected_n_sims=num_simulations
        )

        if has_test_stats:
            print(f"Test statistics already derived")
            return

        # Need to derive test statistics
        print(f"Deriving test statistics from full simulations...")

        # Launch derivation job
        job_id = job_manager.submit_derivation_job(
            hpc_pool_path,
            str(self.test_stats_csv),
            test_stats_hash
        )

        # Wait for derivation job to complete
        print(f"Waiting for derivation to complete...")
        self._wait_for_completion([job_id], num_simulations)
        print(f"Derivation complete")

        # Verify that test stats were actually created
        has_test_stats_now = job_manager.check_hpc_test_stats(hpc_pool_path, test_stats_hash)

        if not has_test_stats_now:
            # Derivation failed - check logs
            project_name = 'pdac_2025'  # Extract from self if needed
            log_path = f"{job_manager.config.remote_project_path}/projects/{project_name}/batch_jobs/logs"
            log_cmd = f'ls -lt "{log_path}"/qsp_derive_*.err 2>/dev/null | head -3'
            status, log_output = job_manager._ssh_exec(log_cmd)

            if status == 0 and log_output.strip():
                print(f"Recent derivation error logs:")
                print(f"  {log_output.strip()}")

            raise RuntimeError(
                f"Derivation job completed but did not produce test statistics. "
                f"Check HPC logs at: {log_path}/qsp_derive_*.err"
            )


    def _run_new_simulations(
        self,
        num_simulations: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run new simulations on HPC and add to pool.

        Args:
            num_simulations: Number of simulations to run

        Returns:
            Tuple of (params, observables)
        """
        print(f"Running {num_simulations} new simulations on HPC (scenario='{self.scenario}')...")

        # Generate parameters for new simulations
        theta_np = self._generate_parameters(num_simulations)

        # Create temporary CSV file for parameter samples
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            samples_csv = tmp.name
            writer = csv.writer(tmp)
            writer.writerow(self.param_names)
            for row in theta_np:
                writer.writerow(row)

        try:
            # Run MATLAB simulations (with full output saving enabled)
            # This runs simulations, derives test statistics, and saves to HPC
            observables_np = self._run_matlab_simulation(samples_csv, num_simulations)

            # Add to SimulationPoolManager
            print(f"Adding {theta_np.shape[0]} simulations to pool (scenario='{self.scenario}')")
            self.pool.add_batch(
                params_matrix=theta_np,
                observables_matrix=observables_np,
                seed=self.seed,
                scenario=self.scenario
            )

            print(f"Complete: {observables_np.shape[0]} simulations finished")

            return theta_np, observables_np

        finally:
            # Clean up temporary file
            Path(samples_csv).unlink(missing_ok=True)

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

        print(f"QSP Simulator: {num_simulations} simulations (scenario='{self.scenario}', version={self.model_version})")

        # 1. Check SimulationPoolManager for cached test statistics (fast path)
        n_available = self.pool.get_available_simulations(scenario=self.scenario)
        if n_available >= num_simulations:
            if self.verbose:
                print(f"  Found {n_available} simulations in local pool (sufficient)")
            params, observables = self.pool.load_simulations(
                n_requested=num_simulations,
                scenario=self.scenario,
                random_state=self.rng
            )
            print(f"Using {num_simulations} cached samples from pool")
            return params, observables

        if self.verbose:
            print(f"  Local pool has {n_available}/{num_simulations} simulations - checking HPC")

        # Compute hash keys for HPC lookups
        priors_hash = self._compute_priors_hash()
        test_stats_hash = self._compute_test_stats_hash()

        # Get HPC pool path (scenario-specific)
        from qsp_hpc.batch.hpc_job_manager import HPCJobManager
        job_manager = HPCJobManager()
        hpc_pool_path = f"{job_manager.config.simulation_pool_path}/{self.model_version}_{priors_hash[:8]}_{self.scenario}"

        # 2. Check HPC for derived test statistics
        has_test_stats = job_manager.check_hpc_test_stats(
            hpc_pool_path, test_stats_hash, expected_n_sims=num_simulations
        )

        if has_test_stats:
            print(f"Found derived test statistics on HPC")
            return self._download_and_add_to_pool(hpc_pool_path, test_stats_hash, num_simulations)

        # 3. Check HPC for full simulations
        has_full_sims, _, n_hpc_available = job_manager.check_hpc_full_simulations(
            f"{self.model_version}_{self.scenario}", priors_hash, num_simulations
        )

        if has_full_sims:
            print(f"Found {n_hpc_available} full simulations on HPC - deriving test statistics")
            self._derive_test_statistics(hpc_pool_path, test_stats_hash, num_simulations, self.verbose)
            return self._download_and_add_to_pool(hpc_pool_path, test_stats_hash, num_simulations)

        # 4. Run new full simulations on HPC
        n_needed = num_simulations - n_hpc_available
        if n_needed > 0:
            print(f"Running {n_needed} new simulations on HPC (scenario='{self.scenario}')")
            params, observables = self._run_new_simulations(n_needed)
            return params, observables

        # If we have some HPC sims but not enough, derive from what we have
        print(f"Deriving test statistics from {n_hpc_available} HPC simulations")
        self._derive_test_statistics(hpc_pool_path, test_stats_hash, n_hpc_available, self.verbose)
        return self._download_and_add_to_pool(hpc_pool_path, test_stats_hash, n_hpc_available)

    def simulate_with_parameters(
        self,
        theta: np.ndarray,
        pool_suffix: str = 'posterior_predictive'
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
        print(f"QSP Simulator ({pool_suffix}): {n_samples} samples ({self.model_version})")

        # Compute hash keys
        priors_hash = self._compute_priors_hash()
        test_stats_hash = self._compute_test_stats_hash()

        # Create pool ID with suffix to separate posterior predictive from training
        pool_id_base = f"{self.model_version}_{priors_hash[:8]}_{pool_suffix}"

        # 1. Check local cache
        local_cache_key = hashlib.sha256(
            (priors_hash + test_stats_hash + self.model_version + pool_suffix).encode()
        ).hexdigest()[:16]
        local_cache_dir = self.cache_dir / f"{pool_id_base}_{local_cache_key}"
        local_cache_file = local_cache_dir / "test_stats.npy"
        local_params_file = local_cache_dir / "params.npy"

        # Check local cache using helper (returns only test stats, not params)
        cached = self._check_local_cache(n_samples, local_cache_file, local_params_file, self.verbose)
        if cached is not None:
            _, cached_test_stats = cached  # Unpack (params, test_stats)
            print(f"Using {n_samples} cached samples")
            return cached_test_stats

        # 2. Check HPC for existing simulations
        from qsp_hpc.batch.hpc_job_manager import HPCJobManager
        job_manager = HPCJobManager()

        pool_path = f"{job_manager.config.simulation_pool_path}/{pool_id_base}"
        has_full_sims, _, n_available = job_manager.check_hpc_full_simulations(
            self.model_version + f"_{pool_suffix}", priors_hash, n_samples
        )

        if has_full_sims:
            if self.verbose:
                print(f"HPC has {n_available} simulations (sufficient)")

            # Check/derive test statistics
            has_test_stats = job_manager.check_hpc_test_stats(
                pool_path, test_stats_hash, expected_n_sims=n_samples
            )

            if not has_test_stats:
                print(f"Deriving test statistics from full simulations...")
                job_id = job_manager.submit_derivation_job(
                    pool_path, str(self.test_stats_csv), test_stats_hash
                )
                self._wait_for_completion([job_id], n_samples)

            # Download test stats
            print(f"Downloading from HPC...")
            params, test_stats = job_manager.download_test_stats(
                pool_path, test_stats_hash, local_cache_dir
            )

            # Cache locally
            self._save_to_cache(params, test_stats, local_cache_dir, self.verbose)

            # Sample if needed
            if test_stats.shape[0] > n_samples:
                if self.verbose:
                    print(f"Sampling {n_samples} from {test_stats.shape[0]} available")
                indices = self.rng.choice(test_stats.shape[0], size=n_samples, replace=False)
                test_stats = test_stats[indices]

            print(f"Complete: Returning {test_stats.shape[0]} samples")
            return test_stats

        # 3. Run new simulations
        print(f"Running {n_samples} new simulations...")

        # Create temporary CSV with provided parameters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            samples_csv = tmp.name
            writer = csv.writer(tmp)
            writer.writerow(self.param_names)
            for row in theta:
                writer.writerow(row)

        try:
            # Run MATLAB simulations with pooling enabled
            observables = self._run_matlab_simulation(samples_csv, n_samples, pool_suffix=pool_suffix)

            # Cache locally
            self._save_to_cache(theta, observables, local_cache_dir, self.verbose)

            print(f"Complete: {observables.shape[0]} simulations finished")
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
        hasher.update(priors_content.encode('utf-8'))

        # Hash model script name
        hasher.update(self.model_script.encode('utf-8'))

        # Hash model version
        hasher.update(self.model_version.encode('utf-8'))

        return hasher.hexdigest()

    def _compute_test_stats_hash(self) -> str:
        """
        Compute hash of test statistics CSV.

        Returns:
            SHA256 hash (full hex string)
        """
        test_stats_content = self.test_stats_csv.read_text()
        return hashlib.sha256(test_stats_content.encode('utf-8')).hexdigest()

    def _run_matlab_simulation(
        self,
        samples_csv: str,
        num_simulations: int,
        pool_suffix: str = ''
    ) -> np.ndarray:
        """
        Run MATLAB simulation for parameter samples on HPC.

        Uses Python job submission via HPCJobManager for efficient batch execution.

        NOTE: Caching is handled by SimulationPoolManager in __call__().
              This method only executes simulations.

        Args:
            samples_csv: Path to CSV file with parameter samples
            num_simulations: Number of simulations
            pool_suffix: Optional suffix to append to simulation pool ID
                        (used for posterior predictive simulations)

        Returns:
            Numpy array of observables (num_simulations x num_observables)
        """
        # Validate SSH connection before submitting jobs
        self._validate_hpc_connection()

        # Import HPC job manager
        from qsp_hpc.batch.hpc_job_manager import HPCJobManager

        # Create job manager (loads from global config)
        job_manager = HPCJobManager()

        # Extract project name from test_stats_csv path
        # Expected format: projects/{project_name}/...
        project_name = 'pdac_2025'  # Default
        test_stats_str = str(self.test_stats_csv)
        if 'projects/' in test_stats_str:
            parts = test_stats_str.split('projects/')[1].split('/')
            if parts:
                project_name = parts[0]

        # Compute simulation pool ID for full simulations (scenario-specific)
        priors_hash = self._compute_priors_hash()
        simulation_pool_id = f"{self.model_version}_{priors_hash[:8]}_{self.scenario}"
        if pool_suffix:
            simulation_pool_id = f"{simulation_pool_id}_{pool_suffix}"

        # Print HPC save path
        hpc_save_path = f"{job_manager.config.simulation_pool_path}/{simulation_pool_id}"
        print(f"   → HPC save path: {hpc_save_path} (scenario='{self.scenario}')")

        # Calculate optimal split across tasks
        from qsp_hpc.batch.batch_utils import calculate_batch_split
        jobs_per_chunk, n_tasks = calculate_batch_split(num_simulations, self.max_tasks)
        print(f"   → Splitting {num_simulations} simulations into {n_tasks} tasks ({jobs_per_chunk} sims/task)")

        # Submit jobs via Python (no MATLAB startup!)
        job_info = job_manager.submit_jobs(
            samples_csv=samples_csv,
            test_stats_csv=str(self.test_stats_csv),
            model_script=self.model_script,
            num_simulations=num_simulations,
            project_name=project_name,
            seed=self.seed,
            jobs_per_chunk=jobs_per_chunk,
            skip_sync=False,  # Sync codebase first
            save_full_simulations=True,  # Enable full simulation saving
            simulation_pool_id=simulation_pool_id  # Pool ID for HPC storage
        )

        # Wait for jobs to complete
        self._wait_for_completion(job_info.job_ids, num_simulations)

        # Collect results via Python (no MATLAB needed!)
        observables_matrix = job_manager.collect_results(
            state_file=job_info.state_file
        )

        return observables_matrix

    def _load_batch_config(self) -> Dict:
        """
        Load batch configuration from ~/.config/qsp-hpc/credentials.yaml.

        Returns:
            Dictionary with SSH and cluster configuration
        """
        if self.batch_config is not None:
            return self.batch_config

        config_file = Path.home() / '.config' / 'qsp-hpc' / 'credentials.yaml'
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration not found at {config_file}\n"
                "Please run 'qsp-hpc setup' to configure HPC connection."
            )

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Extract SSH settings
        ssh_config = config.get('ssh', {})
        self.batch_config = {
            'ssh_host': ssh_config.get('host', ''),
            'ssh_user': ssh_config.get('user', ''),
            'ssh_key': ssh_config.get('key', ''),
        }

        if not self.batch_config['ssh_host']:
            raise ValueError(
                "SSH host not configured in credentials.yaml\n"
                "Please run 'qsp-hpc setup' to configure HPC connection."
            )

        return self.batch_config

    def _validate_hpc_connection(self):
        """
        Validate SSH connection to HPC cluster (fast fail).

        Raises:
            RuntimeError: If SSH connection cannot be established
        """
        try:
            # Import HPC job manager
            from qsp_hpc.batch.hpc_job_manager import HPCJobManager

            # Create job manager (loads from global config)
            job_manager = HPCJobManager()

            # Validate SSH connection (fast - should return in 1-2s)
            job_manager.validate_ssh_connection(timeout=5)

        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except Exception as e:
            raise RuntimeError(f"HPC connection validation failed: {e}")

    def _check_job_status(self, job_id: str) -> Dict[str, int]:
        """
        Check status of SLURM job array via SSH.

        Args:
            job_id: SLURM job ID

        Returns:
            Dictionary with counts: {' completed': N, 'running': N, 'pending': N, 'failed': N}
        """
        config = self._load_batch_config()

        # Build SSH command base
        ssh_base = ['ssh']
        if config['ssh_key']:
            ssh_base.extend(['-i', config['ssh_key']])
        ssh_base.extend([
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=no'
        ])
        # Handle SSH config alias (user is empty) vs direct connection (user@host)
        if config['ssh_user']:
            ssh_base.append(f"{config['ssh_user']}@{config['ssh_host']}")
        else:
            ssh_base.append(config['ssh_host'])

        # Initialize status
        status = {
            'completed': 0,
            'running': 0,
            'pending': 0,
            'failed': 0
        }

        try:
            # First check squeue for active jobs (most reliable for running/pending)
            # Use --array flag to get one entry per array task
            ssh_cmd = ssh_base + [f'squeue -j {job_id} --array --format="%i %T" --noheader 2>/dev/null']
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse squeue output (active jobs only)
            # Each line is "job_id state"
            if result.returncode == 0 and result.stdout.strip():
                lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        state_upper = parts[1].upper()
                        if 'RUNNING' in state_upper:
                            status['running'] += 1
                        elif 'PENDING' in state_upper:
                            status['pending'] += 1

            # Always check sacct for completed/failed jobs (not just when queue is empty)
            # Completed jobs disappear from squeue, so we need to check sacct in parallel
            # Use simple sacct query without time filter (job ID is enough to limit scope)
            ssh_cmd = ssh_base + [
                f'sacct -j {job_id} --format=JobID,State --noheader --parsable2'
            ]
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout.strip():
                lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]

                # Parse "jobid|state" format
                for line in lines:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        job_part = parts[0]
                        state = parts[1]

                        # Only count main array tasks (format: 12345_0, 12345_1, ...)
                        # Skip: main job (12345), sub-steps (12345_0.batch, 12345_0.extern)
                        if '_' in job_part and '.' not in job_part:
                            state_upper = state.upper()
                            if 'COMPLETED' in state_upper:
                                status['completed'] += 1
                            elif 'FAILED' in state_upper or 'CANCELLED' in state_upper or 'TIMEOUT' in state_upper:
                                status['failed'] += 1
                            # Don't double-count running/pending from sacct if already counted in squeue
                            # (sacct may show them in RUNNING state even if not in squeue yet)

            return status

        except subprocess.TimeoutExpired:
            raise RuntimeError("SSH connection timeout while checking job status")
        except Exception as e:
            raise RuntimeError(f"Failed to check job status: {e}")

    def _wait_for_completion(self, job_ids: List[str], num_simulations: int):
        """
        Wait for HPC jobs to complete with progress updates.

        Args:
            job_ids: List of SLURM job IDs to monitor
            num_simulations: Expected number of simulations
        """
        # Give SLURM a few seconds to register the jobs before first check
        time.sleep(5)

        start_time = time.time()
        max_tasks_seen = 0  # Track the maximum number of tasks we've seen

        while True:
            # Check status of all jobs
            total_status = {
                'completed': 0,
                'running': 0,
                'pending': 0,
                'failed': 0
            }

            for job_id in job_ids:
                try:
                    status = self._check_job_status(job_id)
                    for key in total_status:
                        total_status[key] += status[key]
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not check status for job {job_id}: {e}")

            # Calculate progress
            total_tasks = sum(total_status.values())
            max_tasks_seen = max(max_tasks_seen, total_tasks)

            if total_tasks > 0:
                completed_pct = (total_status['completed'] / total_tasks) * 100
            else:
                completed_pct = 0

            # Show progress after every check
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

            print(f"  {total_status['completed']}/{total_tasks} done ({completed_pct:.1f}%) | "
                  f"Running: {total_status['running']}, Pending: {total_status['pending']}, "
                  f"Failed: {total_status['failed']} | {elapsed_str}")

            # Check if all jobs completed or failed
            active_jobs = total_status['running'] + total_status['pending']

            # Break conditions:
            # 1. We've seen tasks and they're all done (completed + failed = total)
            if total_tasks > 0 and active_jobs == 0:
                if total_status['failed'] > 0:
                    print(f"  Warning: {total_status['failed']} task(s) failed")
                break
            # 2. No tasks visible but we previously saw them (they completed and disappeared)
            elif total_tasks == 0 and max_tasks_seen > 0 and elapsed > 30:
                if self.verbose:
                    print(f"  All {max_tasks_seen} jobs completed (no longer visible in queue)")
                break
            # 3. Waited a while and never saw any tasks (possible monitoring failure, proceed anyway)
            elif total_tasks == 0 and elapsed > 120:
                print("  No jobs visible in queue after 2 minutes - proceeding to download")
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
    test_stats_csv: Union[str, Path],
    value_column: str = 'mean'
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

    if 'test_statistic_id' not in df.columns:
        raise ValueError(
            f"Column 'test_statistic_id' not found in test statistics CSV. "
            f"Available columns: {', '.join(df.columns)}"
        )

    # Extract observable names and values
    observable_names = df['test_statistic_id'].tolist()
    observed_values = df[value_column].values

    # Build dictionary with 2D arrays (1, 1) for each observable
    obs_dict = {}
    for i, obs_name in enumerate(observable_names):
        obs_dict[obs_name] = observed_values[i:i+1].reshape(1, 1)

    return obs_dict


def qsp_simulator(
    test_stats_csv: Union[str, Path],
    priors_csv: Union[str, Path],
    model_script: str = '',
    model_version: str = 'v1',
    model_description: str = '',
    scenario: str = 'default',
    cache_dir: Union[str, Path] = 'projects/pdac_2025/cache/sbi_simulations',
    seed: int = 2025,
    cache_sampling_seed: Optional[int] = None,
    max_tasks: int = 10,
    poll_interval: int = 30,
    max_wait_time: Optional[int] = None,
    verbose: bool = False
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
        seed=seed,
        cache_sampling_seed=cache_sampling_seed,
        max_tasks=max_tasks,
        poll_interval=poll_interval,
        max_wait_time=max_wait_time,
        verbose=verbose
    )
