#!/usr/bin/env python3
"""
Result Collection and Parsing

Handles downloading, parsing, and aggregating simulation results from HPC.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

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
        check_dir_cmd = f'test -d "{pool_path}" && echo "exists" || echo "not_found"'
        status, output = self.transport.exec(check_dir_cmd)

        if self.verbose:
            self.logger.debug(f"Directory check result: {output.strip()}")

        return "not_found" not in output

    def count_pool_simulations(self, pool_path: str) -> int:
        """Count simulations in a remote pool.

        Prefers parquet footer metadata (cheap O(1) per file) over the
        legacy ``_{N}sims_`` filename token, which silently overreported
        when array-task chunk drops left the consolidated batch shorter
        than the requested N (#21). Manifest.json takes precedence when
        present (MATLAB pools historically wrote one).

        Pool layouts recognised:
          - Current (#43 option A): ``{pool}/batch_*/chunk_*.parquet`` —
            array tasks write chunks straight into per-submission subdirs.
          - Legacy (pre-#43): flat ``{pool}/batch_*.parquet`` produced
            by the retired combine worker.
          - MATLAB: ``{pool}/batch_*_{N}sims_*.mat``.
        """
        # The python one-liner walks batch_*/chunk_*.parquet AND flat
        # batch_*.parquet, sums num_rows from each file's footer, and
        # prints ``N_SIMS:<sum>``. Footer reads don't materialise rows,
        # so this stays O(num_files), not O(rows).
        py_count = (
            "import sys, glob; "
            "import pyarrow.parquet as pq; "
            "files = sorted(glob.glob('batch_*/chunk_*.parquet')) "
            "+ sorted(glob.glob('batch_*.parquet')); "
            "total = sum(pq.read_metadata(f).num_rows for f in files); "
            "print(f'N_FILES:{len(files)}'); "
            "print(f'N_SIMS:{total}')"
        )
        venv_python = f"{self.config.hpc_venv_path}/bin/python"
        count_cmd = f"""
            cd "{pool_path}" 2>/dev/null || exit 1

            if [ -f manifest.json ]; then
                echo "MANIFEST_FOUND"
                cat manifest.json
            elif ls batch_*/chunk_*.parquet >/dev/null 2>&1 || ls batch_*.parquet >/dev/null 2>&1; then
                echo "COUNTING_PARQUET_METADATA"
                "{venv_python}" -c "{py_count}"
            else
                # No parquets — fall back to the legacy MATLAB .mat path
                # which still encodes N in the filename.
                echo "COUNTING_MAT_FILES"
                ls batch_*.mat 2>/dev/null | wc -l | awk '{{print "N_FILES:" $1}}'
                ls batch_*.mat 2>/dev/null | \
                grep -oE '[0-9]+sims' | \
                sed 's/sims//' | \
                awk '{{sum+=$1}} END {{print "N_SIMS:" sum}}'
            fi
        """
        status, output = self.transport.exec(count_cmd)

        if self.verbose:
            self.logger.debug("Count command output:")
            for line in output.strip().split("\n"):
                self.logger.debug(f"  {line}")

        if status != 0:
            if self.verbose:
                self.logger.debug(f"Failed to count simulations (status={status})")
            return 0

        n_available = 0

        try:
            # Check if we got manifest
            if "MANIFEST_FOUND" in output:
                # Extract JSON (everything after MANIFEST_FOUND line)
                lines = output.split("\n")
                manifest_start = lines.index("MANIFEST_FOUND") + 1
                manifest_json = "\n".join(lines[manifest_start:])

                manifest = json.loads(manifest_json)
                n_available = manifest.get("total_simulations", 0)
                if self.verbose:
                    self.logger.debug(f"Parsed manifest: {n_available} simulations")

            elif "COUNTING_PARQUET_METADATA" in output or "COUNTING_MAT_FILES" in output:
                source = (
                    "parquet metadata" if "COUNTING_PARQUET_METADATA" in output else "filenames"
                )
                for line in output.split("\n"):
                    if line.startswith("N_SIMS:"):
                        n_available = int(line.split(":")[1].strip())
                        self.logger.debug(f"Counted from {source}: {n_available} simulations")
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
        self, model_version: str, priors_hash: str, n_requested: int
    ) -> Tuple[bool, str, int]:
        """
        Check HPC for existing full simulation results.

        Args:
            model_version: Model version string (e.g., 'baseline_pdac')
            priors_hash: Hash of priors + model script + model version
            n_requested: Number of simulations requested

        Returns:
            Tuple of (has_enough, pool_path, n_available)
        """
        pool_name = f"{model_version}_{priors_hash[:8]}"
        pool_path = f"{self.config.simulation_pool_path}/{pool_name}"

        if self.verbose:
            self.logger.debug(f"Checking pool directory: {pool_path}")

        # Check if pool directory exists
        if not self.check_pool_directory_exists(pool_path):
            if self.verbose:
                self.logger.debug("Pool directory does not exist on HPC")
            return False, pool_path, 0

        # Count simulations in pool
        n_available = self.count_pool_simulations(pool_path)

        has_enough = n_available >= n_requested

        if self.verbose:
            self.logger.debug(
                f"Found {n_available} simulations, need {n_requested}: {'sufficient' if has_enough else 'insufficient'}"
            )

        return has_enough, pool_path, n_available

    def check_hpc_test_stats(
        self, pool_path: str, test_stats_hash: str, expected_n_sims: Optional[int] = None
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

        if status != 0 or "exists" not in output:
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
        self, pool_path: str, test_stats_hash: str, local_cache_dir: Path
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
        params = np.loadtxt(local_params, delimiter=",", skiprows=1)
        test_stats = np.loadtxt(local_stats, delimiter=",", skiprows=1)

        return params, test_stats
