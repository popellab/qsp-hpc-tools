#!/usr/bin/env python3
"""
Simulation Pool Manager for QSP-SBI Workflows

This module manages accumulation and reuse of QSP simulations across training runs.
Simulations are pooled by configuration (priors, observables, model version) rather than
by specific batch parameters (num_simulations, seed), enabling flexible reuse.

Supports multi-scenario workflows where the same parameter sets are evaluated under
different therapy protocols or conditions.

Architecture:
    cache/sbi_simulations/
    └── {model_version}_{config_hash[:8]}/
        ├── batch_YYYYMMDD_HHMMSS_{scenario}_Nsims_seedN.mat
        └── ...

Filename format: batch_{timestamp}_{scenario}_{n_sims}sims_seed{seed}.mat

Key Features:
    - Content-based hashing: Pools are identified by priors, observables, and model version
    - Scenario support: Pool simulations for multiple scenarios independently
    - Batch accumulation: Multiple simulation runs are accumulated into pools
    - Flexible reuse: Load arbitrary number of simulations from any scenario
    - Manifest-free: All metadata encoded in filenames for simplicity

Usage:
    pool = SimulationPoolManager(
        cache_dir='cache/sbi_simulations',
        model_version='baseline_gvax',
        model_description='PDAC baseline: 8 params, 12 obs',
        priors_csv='cache/priors.csv',
        test_stats_csv='cache/test_stats.csv',
        model_script='immune_oncology_model_PDAC'
    )

    # Check availability for a scenario
    available = pool.get_available_simulations(scenario='gvax')

    # Load simulations for a specific scenario
    params, obs = pool.load_simulations(n_requested=1000, scenario='gvax')

    # Add new batch for a scenario
    pool.add_batch(new_params, new_obs, seed=42, scenario='gvax')

    # Load multiple scenarios jointly (for multi-scenario SBI)
    scenarios_data = pool.load_multi_scenario(['gvax', 'gvax_anti_pd1'], n_requested=1000)
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.io import loadmat, savemat

from qsp_hpc.constants import HASH_PREFIX_LENGTH
from qsp_hpc.utils.logging_config import create_child_logger, setup_logger


class SimulationPoolManager:
    """
    Manages simulation pools for QSP-SBI workflows with multi-scenario support.

    Pools are identified by configuration hash (priors + observables + model version)
    rather than specific batch parameters, enabling accumulation and reuse across runs.

    Scenarios are tracked via filename patterns, allowing independent pooling per scenario
    while sharing the same parameter space.

    Attributes:
        cache_dir: Base cache directory
        model_version: Descriptive version name (e.g., 'baseline_gvax')
        model_description: Brief description of model configuration
        priors_csv: Path to priors CSV file
        test_stats_csv: Path to test statistics CSV file
        model_script: MATLAB model script name
        config_hash: Hash of configuration (computed from file contents)
        pool_dir: Directory for this specific pool

    Filename Pattern:
        batch_{timestamp}_{scenario}_{n_sims}sims_seed{seed}.mat
        Example: batch_20250114_120530_gvax_1000sims_seed42.mat
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        model_version: str,
        model_description: str,
        priors_csv: Union[str, Path],
        test_stats_csv: Optional[Union[str, Path]] = None,
        calibration_targets: Optional[Union[str, Path, list]] = None,
        model_script: str = "",
        scenario: str = "default",
        submodel_priors_yaml: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize simulation pool manager.

        Args:
            cache_dir: Base cache directory for all simulation pools
            model_version: Descriptive version name (e.g., 'baseline_gvax')
            model_description: Brief description of model configuration
            priors_csv: Path to priors CSV file
            test_stats_csv: Path to test statistics CSV file (mutually exclusive
                           with calibration_targets)
            calibration_targets: Path to directory of calibration target YAML files
                                (mutually exclusive with test_stats_csv)
            model_script: MATLAB model script name
            scenario: Scenario name (e.g., 'baseline_no_treatment', 'gvax_standard_regimen')
            submodel_priors_yaml: Path to submodel_priors.yaml with fitted marginals
                                  and copula correlations (overrides CSV for matched params)
        """
        if test_stats_csv is not None and calibration_targets is not None:
            raise ValueError("Provide test_stats_csv OR calibration_targets, not both")
        if test_stats_csv is None and calibration_targets is None:
            raise ValueError("Must provide either test_stats_csv or calibration_targets")

        self.cache_dir = Path(cache_dir)
        self.model_version = model_version
        self.model_description = model_description
        self.priors_csv = Path(priors_csv)
        self.submodel_priors_yaml = (
            Path(submodel_priors_yaml) if submodel_priors_yaml is not None else None
        )
        self.model_script = model_script
        self.scenario = scenario
        self.seed = seed
        self._calibration_targets_dir = None

        if calibration_targets is not None:
            from qsp_hpc.calibration.yaml_loader import _resolve_yaml_dirs

            # Normalized to List[Path] so the multi-dir form (literature +
            # mechanistic-prior parallel trees) is supported uniformly.
            self._calibration_targets_dir = _resolve_yaml_dirs(calibration_targets)
            self.test_stats_csv = None  # Not used when calibration_targets provided
        else:
            self.test_stats_csv = Path(test_stats_csv)
            if not self.test_stats_csv.exists():
                raise FileNotFoundError(f"Test statistics CSV not found: {self.test_stats_csv}")

        # Verify files exist
        if not self.priors_csv.exists():
            raise FileNotFoundError(f"Priors CSV not found: {self.priors_csv}")

        # Compute configuration hash
        self.config_hash = self._compute_config_hash()

        # Pool directory: {model_version}_{config_hash[:HASH_PREFIX_LENGTH]}_{scenario}
        # This makes local pools consistent with HPC pool structure
        pool_name = f"{model_version}_{self.config_hash[:HASH_PREFIX_LENGTH]}_{scenario}"
        self.pool_dir = self.cache_dir / pool_name

        # Ensure pool directory exists
        pool_existed = self.pool_dir.exists()
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        # Batch filename pattern for parsing
        # Format: batch_{timestamp}_{scenario}_{n_sims}sims_seed{seed}.mat
        # Note: scenario can contain underscores, so we use non-greedy match
        self.batch_pattern = re.compile(r"batch_(\d{8}_\d{6})_(.+?)_(\d+)sims_seed(\d+)\.mat")

        # Setup logger with pool name context
        base_logger = setup_logger(__name__)
        self.logger = create_child_logger(base_logger, model_version)

        # Log pool initialization
        if pool_existed:
            self.logger.info(f"Using existing pool: {self.pool_dir.name}")
        else:
            self.logger.info(f"Creating new pool: {self.pool_dir}")

        # Log pool configuration (verbose)
        self.logger.debug(f"  Config hash: {self.config_hash[:16]}...")
        self.logger.debug(f"  Priors: {self.priors_csv}")
        if self._calibration_targets_dir is not None:
            self.logger.debug(f"  Calibration targets: {self._calibration_targets_dir}")
        else:
            self.logger.debug(f"  Test stats: {self.test_stats_csv}")

    def _compute_config_hash(self) -> str:
        """
        Compute pool-id hash via shared :func:`compute_pool_id_hash`.

        Hash includes only inputs that affect raw simulation outputs:
        priors CSV, submodel priors YAML, model script, seed. Test
        statistics live in a per-hash subdir (``test_stats/<hash>/``)
        and scenario is the pool dir suffix — neither participates here.
        ``model_version`` was retired from the hash in #56 (the human-
        readable directory prefix still uses it).
        """
        from qsp_hpc.utils.hash_utils import compute_pool_id_hash

        return compute_pool_id_hash(
            priors_csv=self.priors_csv,
            model_script=self.model_script,
            submodel_priors_yaml=self.submodel_priors_yaml,
            seed=self.seed,
        )

    def _scan_batches(self, scenario: Optional[str] = None) -> List[Dict]:
        """
        Scan pool directory for batch files matching optional scenario filter.

        Args:
            scenario: Optional scenario name to filter by (default: all scenarios)

        Returns:
            List of batch info dicts with keys: filename, timestamp, scenario, n_sims, seed
        """
        batches = []

        for batch_file in sorted(self.pool_dir.glob("batch_*.mat")):
            match = self.batch_pattern.match(batch_file.name)
            if not match:
                continue

            timestamp, file_scenario, n_sims, seed = match.groups()

            # Filter by scenario if specified
            if scenario is not None and file_scenario != scenario:
                continue

            batches.append(
                {
                    "filename": batch_file.name,
                    "filepath": batch_file,
                    "timestamp": timestamp,
                    "scenario": file_scenario,
                    "n_sims": int(n_sims),
                    "seed": int(seed),
                }
            )

        return batches

    def list_scenarios(self) -> List[str]:
        """
        List all scenarios present in the pool.

        Returns:
            Sorted list of unique scenario names
        """
        batches = self._scan_batches()
        scenarios = sorted(set(b["scenario"] for b in batches))
        return scenarios

    def get_available_simulations(self, scenario: Optional[str] = None) -> int:
        """
        Get total number of simulations available in pool.

        Args:
            scenario: Optional scenario name to filter by (default: all scenarios)

        Returns:
            Total number of simulations across matching batches
        """
        batches = self._scan_batches(scenario=scenario)
        return sum(b["n_sims"] for b in batches)

    def load_simulations(
        self,
        n_requested: int,
        scenario: Optional[str] = None,
        random_state: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load simulations from pool by aggregating batches for a specific scenario.

        If pool has more simulations than requested, randomly samples subset.
        If pool has fewer simulations than requested, returns all available.

        Args:
            n_requested: Number of simulations to load
            scenario: Scenario name to load (required for scenario-aware loading)
            random_state: Random generator for sampling (default: creates new one)

        Returns:
            params_matrix: Parameter values (n_loaded, n_params)
            observables_matrix: Observable values (n_loaded, n_observables)
        """
        if scenario is None:
            raise ValueError("scenario parameter is required for load_simulations")

        if random_state is None:
            random_state = np.random.default_rng()

        # Scan for batches matching this scenario
        batches = self._scan_batches(scenario=scenario)

        if len(batches) == 0:
            raise ValueError(f"No simulations found for scenario '{scenario}'")

        # Load all batches
        all_params = []
        all_observables = []

        for batch in batches:
            batch_file = batch["filepath"]
            if not batch_file.exists():
                self.logger.warning(f"Batch file not found: {batch_file}")
                continue

            # Load batch data with validation
            try:
                mat_data = loadmat(batch_file)
            except Exception as exc:  # pragma: no cover - exercised in tests
                raise ValueError(f"Failed to load batch file {batch_file}: {exc}") from exc

            if "params_matrix" not in mat_data or "observables_matrix" not in mat_data:
                raise ValueError(f"Batch file missing required keys: {batch_file}")

            params = mat_data["params_matrix"]
            observables = mat_data["observables_matrix"]

            # Ensure 2D shape
            if params.ndim == 1:
                params = params.reshape(1, -1)
            if observables.ndim == 1:
                observables = observables.reshape(1, -1)

            if params.shape[0] != observables.shape[0]:
                raise ValueError(
                    f"Batch {batch_file} has mismatched rows: "
                    f"params {params.shape[0]} vs obs {observables.shape[0]}"
                )

            all_params.append(params)
            all_observables.append(observables)

        # Concatenate all batches
        if not all_params:
            raise ValueError(f"No readable simulations found for scenario '{scenario}'")

        params_all = np.vstack(all_params)
        observables_all = np.vstack(all_observables)

        n_loaded = params_all.shape[0]

        # Sample subset if we have more than requested
        if n_loaded > n_requested:
            self.logger.info(
                f"Sampling {n_requested} from {n_loaded} available simulations (scenario={scenario})"
            )
            indices = random_state.choice(n_loaded, size=n_requested, replace=False)
            params_all = params_all[indices]
            observables_all = observables_all[indices]
            n_loaded = n_requested
        elif n_loaded < n_requested:
            self.logger.warning(
                f"Only {n_loaded} simulations available (requested {n_requested}, scenario={scenario})"
            )
        else:
            self.logger.info(f"Loaded {n_loaded} simulations from pool (scenario={scenario})")

        return params_all, observables_all

    def load_multi_scenario(
        self,
        scenarios: List[str],
        n_requested: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load simulations for multiple scenarios jointly.

        Useful for multi-scenario SBI where you want to fit parameters across
        different therapy protocols or conditions.

        Args:
            scenarios: List of scenario names to load
            n_requested: Number of simulations to load per scenario
            random_state: Random generator for sampling (default: creates new one)

        Returns:
            Dictionary mapping scenario name -> (params_matrix, observables_matrix)
        """
        if random_state is None:
            random_state = np.random.default_rng()

        results = {}
        for scenario in scenarios:
            params, obs = self.load_simulations(
                n_requested=n_requested, scenario=scenario, random_state=random_state
            )
            results[scenario] = (params, obs)

        return results

    def add_batch(
        self, params_matrix: np.ndarray, observables_matrix: np.ndarray, seed: int, scenario: str
    ) -> str:
        """
        Add new batch of simulations to pool for a specific scenario.

        Args:
            params_matrix: Parameter values (n_sims, n_params)
            observables_matrix: Observable values (n_sims, n_observables)
            seed: Random seed used for this batch
            scenario: Scenario name (e.g., 'gvax', 'gvax_anti_pd1', 'control')

        Returns:
            batch_filename: Name of saved batch file
        """
        # Ensure 2D shape
        if params_matrix.ndim == 1:
            params_matrix = params_matrix.reshape(1, -1)
        if observables_matrix.ndim == 1:
            observables_matrix = observables_matrix.reshape(1, -1)

        n_sims = params_matrix.shape[0]

        # Generate batch filename with timestamp and scenario
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = f"batch_{timestamp}_{scenario}_{n_sims}sims_seed{seed}.mat"
        batch_file = self.pool_dir / batch_filename

        # Save batch to .mat file with scenario metadata
        savemat(
            batch_file,
            {
                "params_matrix": params_matrix,
                "observables_matrix": observables_matrix,
                "metadata": {
                    "n_sims": n_sims,
                    "seed": seed,
                    "scenario": scenario,
                    "timestamp": timestamp,
                },
            },
        )

        # Log batch addition with details
        self.logger.info("Adding new batch to pool:")
        self.logger.info(f"  File: {batch_filename}")
        self.logger.info(f"  Simulations: {n_sims}")

        # Get total simulations in pool for this scenario after adding
        total_sims = self.get_available_simulations(scenario=scenario)
        n_batches = len(self._scan_batches(scenario=scenario))
        self.logger.info(
            f"  Pool now contains {n_batches} batches, {total_sims} total sims (scenario={scenario})"
        )

        return batch_filename

    def get_pool_info(self, scenario: Optional[str] = None) -> Dict:
        """
        Get summary information about this pool.

        Args:
            scenario: Optional scenario name to filter by (default: all scenarios)

        Returns:
            Dictionary with pool metadata
        """
        batches = self._scan_batches(scenario=scenario)
        scenarios = self.list_scenarios()

        # Scenario-specific info
        scenario_info = {}
        for scen in scenarios:
            scen_batches = self._scan_batches(scenario=scen)
            scenario_info[scen] = {
                "n_batches": len(scen_batches),
                "total_simulations": sum(b["n_sims"] for b in scen_batches),
            }

        return {
            "pool_dir": str(self.pool_dir),
            "model_version": self.model_version,
            "model_description": self.model_description,
            "config_hash": self.config_hash,
            "scenarios": scenarios,
            "scenario_info": scenario_info,
            "total_simulations": sum(b["n_sims"] for b in batches),
            "n_batches": len(batches),
        }

    @classmethod
    def list_pools(cls, cache_dir: Union[str, Path]) -> List[Dict]:
        """
        List all simulation pools in cache directory.

        Scans directories for batch files and extracts info from filenames.

        Args:
            cache_dir: Base cache directory

        Returns:
            List of pool info dictionaries with basic metadata
        """
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            return []

        pools = []
        batch_pattern = re.compile(r"batch_(\d{8}_\d{6})_(.+?)_(\d+)sims_seed(\d+)\.mat")

        for pool_dir in sorted(cache_dir.iterdir()):
            if not pool_dir.is_dir():
                continue

            # Scan for batch files
            batch_files = list(pool_dir.glob("batch_*.mat"))
            if len(batch_files) == 0:
                continue

            # Extract scenarios and count simulations
            scenarios = set()
            total_sims = 0
            for batch_file in batch_files:
                match = batch_pattern.match(batch_file.name)
                if match:
                    _, scenario, n_sims, _ = match.groups()
                    scenarios.add(scenario)
                    total_sims += int(n_sims)

            # Parse pool directory name: {model_version}_{config_hash[:8]}
            pool_name = pool_dir.name
            parts = pool_name.rsplit("_", 1)
            model_version = parts[0] if len(parts) == 2 else pool_name
            config_hash = parts[1] if len(parts) == 2 else "unknown"

            pool_info = {
                "pool_dir": str(pool_dir),
                "pool_name": pool_name,
                "model_version": model_version,
                "config_hash": config_hash,
                "scenarios": sorted(scenarios),
                "total_simulations": total_sims,
                "n_batches": len(batch_files),
            }
            pools.append(pool_info)

        return pools
