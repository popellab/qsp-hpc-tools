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
from typing import Dict, Any, Optional
from qsp_hpc.utils.logging_config import setup_logger
from qsp_hpc.utils.security import validate_project_name, safe_shell_quote


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

        # Build remote target (handle SSH config aliases where user is optional)
        if self.config.ssh_user:
            remote_target = f"{self.config.ssh_user}@{self.config.ssh_host}:{self.config.remote_project_path}"
        else:
            remote_target = f"{self.config.ssh_host}:{self.config.remote_project_path}"

        rsync_cmd = ['rsync', '-avz', '--delete']

        # Add SSH key if specified (with proper quoting)
        if self.config.ssh_key:
            rsync_cmd.extend(['-e', f'ssh -i "{self.config.ssh_key}"'])

        # Add exclusion patterns
        for pattern in self.rsync_exclude_patterns:
            rsync_cmd.extend(['--exclude', pattern])

        rsync_cmd.append(f'{local_root}/')
        rsync_cmd.append(remote_target)

        # Execute rsync
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"rsync failed (rc={result.returncode}): {result.stderr or result.stdout}"
            )

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
        self.logger.info(f"Installing qsp-hpc-tools from: {self.config.qsp_hpc_tools_source}")

        # Run inline setup commands
        setup_script = f"""
set -e

echo "Creating venv at {self.config.hpc_venv_path}..."
uv venv --python 3.11 {self.config.hpc_venv_path}

echo "Installing qsp-hpc-tools from {self.config.qsp_hpc_tools_source}..."
uv pip install --python {self.config.hpc_venv_path}/bin/python "{self.config.qsp_hpc_tools_source}"

echo "Verifying installation..."
{self.config.hpc_venv_path}/bin/python -c "import qsp_hpc; print('✓ qsp-hpc-tools installed')"
{self.config.hpc_venv_path}/bin/python -c "import numpy, pandas, pyarrow; print('✓ Dependencies available')"

echo "Python venv setup complete!"
"""
        status, output = self.transport.exec(setup_script, timeout=300)  # 5 min timeout

        if status != 0:
            self.logger.warning("venv setup had issues (but may still work)")
            if self.verbose:
                self.logger.debug(f"Output: {output}")
        else:
            self.logger.info("HPC Python environment configured")

    def setup_remote_directories(self, project_name: str) -> None:
        """Create necessary directories on remote cluster and clean old files."""
        # Validate project name to prevent command injection
        project_name = validate_project_name(project_name)

        remote_root = (self.config.remote_project_path or "").strip()
        if not remote_root or remote_root in {"/", ".", "//"}:
            raise ValueError("remote_project_path must be set to a non-root directory before deleting files")

        remote_base = f"{remote_root}/projects/{project_name}/batch_jobs"
        dirs = ['input', 'output', 'scripts', 'logs']

        for d in dirs:
            remote_dir = f"{remote_base}/{d}"
            # Remove all files in directory, then recreate it
            # Use safe_shell_quote to prevent injection even though project_name is validated
            safe_dir = safe_shell_quote(remote_dir)
            self.transport.exec(f'rm -rf {safe_dir} && mkdir -p {safe_dir}')

    def upload_job_config(
        self,
        test_stats_csv: str,
        model_script: str,
        num_simulations: int,
        seed: int,
        jobs_per_chunk: int,
        project_name: str,
        save_full_simulations: bool = True,
        simulation_pool_id: Optional[str] = None
    ) -> None:
        """Create and upload job configuration as JSON."""
        start_time = time.time()

        # Create job config dictionary
        job_config = {
            'project_name': project_name,
            'n_simulations': num_simulations,
            'seed': seed,
            'jobs_per_chunk': jobs_per_chunk,
            'model_script': model_script,
            'test_stats_csv': f'projects/{project_name}/batch_jobs/input/test_stats.csv',  # Remote path
            'param_csv': f'projects/{project_name}/batch_jobs/input/params.csv',  # Remote path
            'save_full_simulations': save_full_simulations,
            'simulation_pool_id': simulation_pool_id
        }

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(job_config, f, indent=2)
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

        # Load CSV to extract embedded functions
        try:
            import pandas as pd

            test_stats_df = pd.read_csv(test_stats_csv)

            if 'model_output_code' in test_stats_df.columns and len(test_stats_df) > 0:
                # Create temp directory for functions
                temp_dir = Path(tempfile.mkdtemp())

                try:
                    # Extract all functions to temp directory
                    n_functions = 0
                    for idx, row in test_stats_df.iterrows():
                        test_stat_id = row['test_statistic_id']
                        func_code = row['model_output_code']

                        # Write function to temp directory
                        func_file = temp_dir / f"test_stat_{test_stat_id}.m"
                        func_file.write_text(func_code)
                        n_functions += 1

                    # Create tarball of all functions
                    tarball_path = temp_dir / "test_stat_functions.tar.gz"
                    with tarfile.open(tarball_path, 'w:gz') as tar:
                        for func_file in temp_dir.glob("test_stat_*.m"):
                            tar.add(func_file, arcname=func_file.name)

                    # Upload tarball
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
