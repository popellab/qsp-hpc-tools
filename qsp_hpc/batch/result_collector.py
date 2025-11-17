#!/usr/bin/env python3
"""
Result Collection and Parsing

Handles downloading, parsing, and aggregating simulation results from HPC.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from qsp_hpc.utils.logging_config import setup_logger


class MissingOutputError(RuntimeError):
    """Raised when expected remote output artifacts are missing."""
    pass


class ResultCollector:
    """
    Handles result collection and parsing from HPC.

    Responsibilities:
    - Check for simulation pools on HPC
    - Download and combine result files
    - Parse Parquet and CSV files
    - Count available simulations
    """

    def __init__(self, config, transport, verbose: bool = False):
        """
        Initialize result collector.

        Args:
            config: BatchConfig with paths
            transport: SSHTransport instance
            verbose: Enable verbose logging
        """
        self.config = config
        self.transport = transport
        self.verbose = verbose
        self.logger = setup_logger(__name__, verbose=verbose)

    def check_pool_directory_exists(self, pool_path: str) -> bool:
        """Check if simulation pool directory exists on HPC."""
        status, _ = self.transport.exec(f'test -d "{pool_path}" && echo "exists"')
        return status == 0

    def count_pool_simulations(self, pool_path: str) -> int:
        """
        Count number of simulations in an HPC pool directory.

        Counts Parquet files and sums their simulation counts from metadata.
        """
        if not self.check_pool_directory_exists(pool_path):
            return 0

        # List Parquet files and count simulations
        count_script = f"""
cd "{pool_path}" || exit 1

# Count from filenames (batch_TIMESTAMP_SCENARIO_NNNsims_seedSSS.parquet)
total=0
for f in batch_*.parquet; do
    if [[ "$f" =~ batch_[0-9]+_[0-9]+_[^_]+_([0-9]+)sims_seed[0-9]+\\.parquet ]]; then
        n="${{BASH_REMATCH[1]}}"
        total=$((total + n))
    fi
done

echo "N_SIMS:$total"
"""

        status, output = self.transport.exec(count_script)

        n_available = 0
        try:
            if status == 0 and output.strip():
                # Extract N_SIMS value
                for line in output.split('\n'):
                    if line.startswith('N_SIMS:'):
                        n_available = int(line.split(':')[1].strip())
                        self.logger.debug(f"Counted from filenames: {n_available} simulations")
                        break
            else:
                self.logger.debug("Could not parse output format")

        except (ValueError, IndexError, KeyError) as e:
            # Handle parsing errors gracefully - likely means no valid simulations
            self.logger.warning(f"Error parsing simulation count: {e}")
            n_available = 0
        except Exception as e:
            # Unexpected error - log and re-raise
            self.logger.error(f"Unexpected error checking HPC pool: {e}")
            raise

        return n_available

    def check_hpc_full_simulations(
        self,
        model_version: str,
        priors_hash: str,
        num_simulations: int
    ) -> Tuple[bool, str, int]:
        """
        Check HPC for existing full simulation results.

        Returns:
            Tuple of (has_sufficient, pool_path, n_available)
        """
        # Construct pool path
        pool_id = f"{model_version}_{priors_hash[:8]}"
        pool_path = f"{self.config.simulation_pool_path}/{pool_id}"

        # Check if directory exists
        if not self.check_pool_directory_exists(pool_path):
            return False, pool_path, 0

        # Count available simulations
        n_available = self.count_pool_simulations(pool_path)

        has_sufficient = n_available >= num_simulations
        return has_sufficient, pool_path, n_available

    def check_hpc_test_stats(
        self,
        pool_path: str,
        test_stats_hash: str,
        expected_n_sims: Optional[int] = None
    ) -> bool:
        """
        Check if derived test statistics exist on HPC.

        Args:
            pool_path: Path to simulation pool
            test_stats_hash: Hash of test statistics configuration
            expected_n_sims: Expected number of simulations (for validation)

        Returns:
            True if test stats exist and are valid
        """
        # Check for combined params and test_stats files
        params_file = f"{pool_path}/test_stats_{test_stats_hash[:8]}_params.csv"
        stats_file = f"{pool_path}/test_stats_{test_stats_hash[:8]}.csv"

        # Check if both files exist
        check_cmd = f'test -f "{params_file}" && test -f "{stats_file}" && echo "exists"'
        status, output = self.transport.exec(check_cmd)

        if status != 0 or 'exists' not in output:
            return False

        # If expected count specified, validate it
        if expected_n_sims is not None:
            count_cmd = f'wc -l < "{stats_file}"'
            status, output = self.transport.exec(count_cmd)

            if status == 0 and output.strip().isdigit():
                # Subtract 1 for header line
                n_lines = int(output.strip()) - 1
                if n_lines < expected_n_sims:
                    self.logger.warning(
                        f"Test stats file has {n_lines} rows, expected {expected_n_sims}"
                    )
                    return False

        return True

    def download_test_stats(
        self,
        pool_path: str,
        test_stats_hash: str,
        local_cache_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Download derived test statistics from HPC.

        Args:
            pool_path: Path to simulation pool on HPC
            test_stats_hash: Hash of test statistics configuration
            local_cache_dir: Local directory to save files

        Returns:
            Tuple of (params, test_stats) as numpy arrays
        """
        local_cache_dir = Path(local_cache_dir)
        local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Download params and test stats CSV files
        params_file = f"{pool_path}/test_stats_{test_stats_hash[:8]}_params.csv"
        stats_file = f"{pool_path}/test_stats_{test_stats_hash[:8]}.csv"

        local_params = local_cache_dir / "params.csv"
        local_stats = local_cache_dir / "test_stats.csv"

        self.transport.download(params_file, str(local_cache_dir))
        self.transport.download(stats_file, str(local_cache_dir))

        # Rename downloaded files
        (local_cache_dir / f"test_stats_{test_stats_hash[:8]}_params.csv").rename(local_params)
        (local_cache_dir / f"test_stats_{test_stats_hash[:8]}.csv").rename(local_stats)

        # Load into numpy arrays
        params = np.loadtxt(local_params, delimiter=',', skiprows=1)
        test_stats = np.loadtxt(local_stats, delimiter=',', skiprows=1)

        return params, test_stats
