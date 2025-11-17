#!/usr/bin/env python3
"""
HPC File Transfer Operations

Handles all file transfer operations between local machine and HPC cluster,
including codebase syncing, file uploads, and directory setup.
"""

import json
import subprocess
import tempfile
import time
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Any
from qsp_hpc.utils.logging_config import setup_logger


class HPCFileTransfer:
    """
    Handles file transfer operations to/from HPC cluster.

    Responsibilities:
    - Sync codebase with rsync
    - Setup Python virtual environment on HPC
    - Upload job configurations, parameters, test statistics
    - Create remote directories
    """

    def __init__(self, config, transport, verbose: bool = False):
        """
        Initialize file transfer handler.

        Args:
            config: BatchConfig with paths and settings
            transport: SSHTransport instance
            verbose: Enable verbose logging
        """
        self.config = config
        self.transport = transport
        self.verbose = verbose
        self.logger = setup_logger(__name__, verbose=verbose)

        # Rsync exclusion patterns
        self.rsync_exclude_patterns = [
            '.git',
            '*.mat',
            '*.asv',
            '*.m~',
            '*.pdf',
            '__pycache__',
            '*.pyc',
            '.DS_Store',
            'venv/',
            '.venv/',
            'projects/*/batch_jobs/output/',
            'projects/*/cache/',
        ]

    def sync_codebase(self, skip_sync: bool = False) -> None:
        """
        Sync codebase to HPC using rsync.

        Args:
            skip_sync: If True, skip syncing (for testing)
        """
        if skip_sync:
            return

        start_time = time.time()

        # Build rsync command
        local_root = Path.cwd()
        remote_target = f"{self.config.ssh_user}@{self.config.ssh_host}:{self.config.remote_project_path}"

        rsync_cmd = ['rsync', '-avz', '--delete']

        # Add SSH key if specified
        if self.config.ssh_key:
            rsync_cmd.extend(['-e', f'ssh -i {self.config.ssh_key}'])

        # Add exclusion patterns
        for pattern in self.rsync_exclude_patterns:
            rsync_cmd.extend(['--exclude', pattern])

        rsync_cmd.append(f'{local_root}/')
        rsync_cmd.append(remote_target)

        # Execute rsync
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)

        elapsed = time.time() - start_time
        if self.verbose:
            self.logger.info(f"Codebase synced ({elapsed:.1f}s)")

    def ensure_hpc_venv(self) -> None:
        """
        Ensure Python virtual environment is set up on HPC.

        Creates venv at configured hpc_venv_path if it doesn't exist and installs
        required packages for Parquet I/O and test statistics derivation.
        """
        if self.verbose:
            self.logger.info(f"Checking HPC Python environment at {self.config.hpc_venv_path}...")

        # Check if venv exists
        status, _ = self.transport.exec(f'test -d "{self.config.hpc_venv_path}" && echo "exists"')

        if status == 0:
            if self.verbose:
                self.logger.info("HPC venv already configured")
            return

        self.logger.info("Setting up HPC Python environment (first time only)...")

        # Run setup script on HPC
        setup_script = f"""
cd "{self.config.remote_project_path}"
bash scripts/hpc/setup_hpc_venv.sh
"""
        status, output = self.transport.exec(setup_script, timeout=300)  # 5 min timeout

        if status != 0:
            self.logger.warning("venv setup had issues (but may still work)")
            if self.verbose:
                self.logger.debug(f"Output: {output}")
        else:
            self.logger.info("HPC Python environment configured")

    def setup_remote_directories(self, project_name: str) -> None:
        """Create necessary directories on HPC for batch jobs."""
        dirs = [
            f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/input",
            f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/output",
            f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/logs",
        ]

        for dir_path in dirs:
            self.transport.exec(f'mkdir -p "{dir_path}"')

    def upload_job_config(
        self,
        model_script: str,
        num_simulations: int,
        seed: int,
        jobs_per_chunk: int,
        save_full_simulations: bool,
        simulation_pool_id: str,
        project_name: str
    ) -> None:
        """Upload job configuration JSON."""
        start_time = time.time()

        config_data = {
            'model_script': model_script,
            'num_simulations': num_simulations,
            'seed': seed,
            'jobs_per_chunk': jobs_per_chunk,
            'save_full_simulations': save_full_simulations,
            'simulation_pool_id': simulation_pool_id,
            'simulation_pool_path': self.config.simulation_pool_path
        }

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f, indent=2)
            temp_file = f.name

        try:
            remote_input_dir = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/input"
            remote_file = f"{remote_input_dir}/job_config.json"

            self.transport.upload(temp_file, remote_file)
            if self.verbose:
                elapsed = time.time() - start_time
                self.logger.debug(f"Job config uploaded ({elapsed:.1f}s)")
        finally:
            Path(temp_file).unlink()

    def upload_parameter_csv(self, csv_path: str, project_name: str) -> None:
        """Upload parameter samples CSV."""
        start_time = time.time()

        remote_input_dir = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/input"
        remote_file = f"{remote_input_dir}/params.csv"

        self.transport.upload(csv_path, remote_file)
        if self.verbose:
            elapsed = time.time() - start_time
            self.logger.debug(f"Parameters uploaded ({elapsed:.1f}s)")

    def upload_test_statistics(self, test_stats_csv: str, project_name: str) -> None:
        """Upload test statistics CSV and extract embedded functions as tarball."""
        start_time = time.time()

        remote_input_dir = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/input"

        # Upload CSV file
        remote_csv = f"{remote_input_dir}/test_stats.csv"
        self.transport.upload(test_stats_csv, remote_csv)

        # Check if CSV has embedded functions
        import csv
        try:
            with open(test_stats_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Check if any row has a 'function_code' column
            if rows and 'function_code' in rows[0]:
                # Extract functions to temp directory
                temp_dir = Path(tempfile.mkdtemp())
                functions_dir = temp_dir / 'test_stat_functions'
                functions_dir.mkdir()

                n_functions = 0
                try:
                    for i, row in enumerate(rows):
                        if row.get('function_code'):
                            func_file = functions_dir / f"test_stat_{i}.m"
                            func_file.write_text(row['function_code'])
                            n_functions += 1

                    # Create tarball
                    tarball_path = temp_dir / 'test_stat_functions.tar.gz'
                    with tarfile.open(tarball_path, 'w:gz') as tar:
                        tar.add(functions_dir, arcname='test_stat_functions')

                    # Upload and extract on HPC
                    remote_tarball = f"{remote_input_dir}/test_stat_functions.tar.gz"
                    self.transport.upload(str(tarball_path), remote_tarball)

                    # Extract on remote
                    extract_cmd = f'cd "{remote_input_dir}" && tar -xzf test_stat_functions.tar.gz && rm test_stat_functions.tar.gz'
                    self.transport.exec(extract_cmd)

                    if self.verbose:
                        elapsed = time.time() - start_time
                        self.logger.debug(f"Test statistics uploaded ({n_functions} functions, {elapsed:.1f}s)")

                finally:
                    # Clean up temp directory
                    shutil.rmtree(temp_dir)
            else:
                if self.verbose:
                    elapsed = time.time() - start_time
                    self.logger.debug(f"Test statistics uploaded ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Failed to upload test statistics: {e}")
            self.logger.debug(f"Time elapsed before error: {elapsed:.1f}s")
            raise  # Re-raise the exception after logging
