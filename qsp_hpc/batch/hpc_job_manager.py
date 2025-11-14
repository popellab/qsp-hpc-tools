#!/usr/bin/env python3
"""
HPC Job Manager for QSP Simulations

This module handles all HPC job submission, monitoring, and result collection
for QSP simulations. It replaces the MATLAB-based batch_execute.m workflow
with pure Python for faster job submission and better error handling.

Key features:
- Fast SSH connection validation (1-2s)
- Rsync codebase syncing with exclusions
- SLURM job submission and management
- Job state tracking (Python pickle format)
- Result collection and aggregation

Usage:
    from qsp_hpc.hpc_job_manager import HPCJobManager

    manager = HPCJobManager(batch_config)
    manager.validate_ssh_connection()  # Fast validation

    job_info = manager.submit_jobs(
        samples_csv='path/to/samples.csv',
        model_config='path/to/model_config.mat',
        num_simulations=100,
        project_name='pdac_2025'
    )

    # Job info contains: job_ids, state_file
    # Use with qsp_simulator for monitoring
"""

import subprocess
import yaml
import pickle
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class JobInfo:
    """Information about submitted HPC jobs."""
    job_ids: List[str]
    state_file: str
    n_jobs: int
    n_simulations: int
    project_name: str
    submission_time: str


@dataclass
class BatchConfig:
    """HPC batch configuration."""
    ssh_host: str
    ssh_user: str
    simulation_pool_path: str
    hpc_venv_path: str
    ssh_key: str = ''
    remote_project_path: str = ''
    partition: str = 'shared'
    time_limit: str = '20:00'
    memory_per_job: str = '2G'
    matlab_module: str = 'matlab/R2024a'
    jobs_per_chunk: int = 20


class HPCJobManager:
    """
    Manages HPC job submission and result collection for QSP simulations.

    This class provides a Python-based alternative to MATLAB's batch_execute.m,
    eliminating MATLAB startup overhead and enabling faster job submission.
    """

    def __init__(self, config: Union[Dict, BatchConfig, None] = None, verbose: bool = False):
        """
        Initialize HPC job manager.

        Args:
            config: Batch configuration dict, BatchConfig object, or None to load from YAML
            verbose: If True, print detailed progress information (default: False)
        """
        if config is None:
            config = self._load_config_from_yaml()
        elif isinstance(config, dict):
            config = BatchConfig(**config)

        self.config = config
        self.verbose = verbose

        # Rsync exclusion patterns (from batch_execute.m)
        self.rsync_exclude_patterns = [
            '.git',
            '*.mat',
            '*.asv',
            '*.m~',
            '*.pdf',
            '*.xlsx',
            'venv/',
            'env/',
            '.venv/',
            'scratch/',
            'projects/*/batch_jobs/',
            'projects/*/cache/',
            '.DS_Store',
            '.vscode/',
            '.idea/',
            '*.swp',
            '*.swo',
            '__pycache__/',
            'node_modules/',
            'batch_credentials.yaml',
            'slurm_config.yaml'
        ]

    def _load_config_from_yaml(self) -> BatchConfig:
        """Load configuration from batch_credentials.yaml."""
        config_file = Path('batch_credentials.yaml')
        if not config_file.exists():
            raise FileNotFoundError(
                "batch_credentials.yaml not found. Required for HPC job submission."
            )

        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)

        ssh = yaml_config.get('ssh', {})
        cluster = yaml_config.get('cluster', {})

        # Require simulation_pool_path (no fallback)
        simulation_pool_path = ssh.get('simulation_pool_path')
        if not simulation_pool_path:
            raise ValueError(
                "ssh.simulation_pool_path must be specified in batch_credentials.yaml. "
                "This is the HPC path where simulation pools are stored (e.g., '/home/username/data/qsp_simulations')"
            )

        # Require hpc_venv_path (no fallback)
        hpc_venv_path = ssh.get('hpc_venv_path')
        if not hpc_venv_path:
            raise ValueError(
                "ssh.hpc_venv_path must be specified in batch_credentials.yaml. "
                "This is the HPC Python virtual environment path (e.g., '/home/username/qspio_venv')"
            )

        return BatchConfig(
            ssh_host=ssh.get('host', ''),
            ssh_user=ssh.get('user', ''),
            simulation_pool_path=simulation_pool_path,
            hpc_venv_path=hpc_venv_path,
            ssh_key=ssh.get('key', ''),
            remote_project_path=ssh.get('remote_project_path', ''),
            partition=cluster.get('partition', 'shared'),
            time_limit=cluster.get('time_limit', '01:00:00'),  # Default 1 hour
            memory_per_job=cluster.get('memory_per_job', '4G'),  # Default 4G
            matlab_module=cluster.get('matlab_module', 'matlab/R2024a')
        )

    def validate_ssh_connection(self, timeout: int = 5) -> bool:
        """
        Quickly validate SSH connection to HPC cluster.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connection successful

        Raises:
            RuntimeError: If SSH connection fails
        """

        try:
            status, output = self._ssh_exec('echo "SSH_OK"', timeout=timeout)

            if status == 0 and 'SSH_OK' in output:
                return True
            else:
                raise RuntimeError(f"SSH connection failed: {output}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SSH connection timeout after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"SSH connection error: {e}")

    def _ssh_exec(self, command: str, timeout: Optional[int] = 30) -> Tuple[int, str]:
        """
        Execute command on HPC cluster via SSH.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds

        Returns:
            Tuple of (return_code, output)
        """
        ssh_cmd = ['ssh']

        if self.config.ssh_key:
            ssh_cmd.extend(['-i', self.config.ssh_key])

        # Add connection options
        ssh_cmd.extend([
            '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=5',
            '-o', 'StrictHostKeyChecking=no',  # Trust host keys automatically
            '-o', 'BatchMode=yes'  # Prevent interactive prompts
        ])

        # Use SSH config alias if no user specified, otherwise use user@host
        if self.config.ssh_user:
            ssh_cmd.append(f"{self.config.ssh_user}@{self.config.ssh_host}")
        else:
            ssh_cmd.append(self.config.ssh_host)
        ssh_cmd.append(command)

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return result.returncode, result.stdout + result.stderr

    def _ssh_upload(self, local_path: str, remote_path: str) -> None:
        """Upload file to HPC cluster via SCP."""
        scp_cmd = ['scp']

        if self.config.ssh_key:
            scp_cmd.extend(['-i', self.config.ssh_key])

        # Use SSH config alias if no user specified, otherwise use user@host
        if self.config.ssh_user:
            remote_target = f"{self.config.ssh_user}@{self.config.ssh_host}:{remote_path}"
        else:
            remote_target = f"{self.config.ssh_host}:{remote_path}"

        scp_cmd.extend([
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'BatchMode=yes',
            local_path,
            remote_target
        ])

        subprocess.run(scp_cmd, check=True, capture_output=True)

    def _ssh_download(self, remote_path: str, local_dir: str) -> None:
        """Download file from HPC cluster via SCP."""
        scp_cmd = ['scp']

        if self.config.ssh_key:
            scp_cmd.extend(['-i', self.config.ssh_key])

        # Use SSH config alias if no user specified, otherwise use user@host
        if self.config.ssh_user:
            remote_source = f"{self.config.ssh_user}@{self.config.ssh_host}:{remote_path}"
        else:
            remote_source = f"{self.config.ssh_host}:{remote_path}"

        scp_cmd.extend([
            '-o', 'StrictHostKeyChecking=no',
            remote_source,
            local_dir
        ])

        subprocess.run(scp_cmd, check=True, capture_output=True)

    def ensure_hpc_venv(self) -> None:
        """
        Ensure Python virtual environment is set up on HPC.

        Creates venv at configured hpc_venv_path if it doesn't exist and installs
        required packages for Parquet I/O and test statistics derivation.
        """
        if self.verbose:
            print(f"Checking HPC Python environment at {self.config.hpc_venv_path}...")

        # Check if venv exists
        status, _ = self._ssh_exec(f'test -d "{self.config.hpc_venv_path}" && echo "exists"')

        if status == 0:
            if self.verbose:
                print(f"HPC venv already configured")
            return

        print(f"Setting up HPC Python environment (first time only)...")

        # Run setup script on HPC
        setup_script = f"""
cd "{self.config.remote_project_path}"
bash scripts/setup_hpc_venv.sh
"""
        status, output = self._ssh_exec(setup_script, timeout=300)  # 5 min timeout

        if status != 0:
            print(f"Warning: venv setup had issues (but may still work)")
            if self.verbose:
                print(f"Output: {output}")
        else:
            print(f"HPC Python environment configured")

    def sync_codebase(self, skip_sync: bool = False) -> None:
        """
        Sync codebase to HPC using rsync.

        Args:
            skip_sync: If True, skip syncing (for testing)
        """
        if skip_sync:
            return

        start_time = time.time()

        local_root = Path.cwd()
        remote_path = self.config.remote_project_path

        # Ensure remote directory exists
        self._ssh_exec(f'mkdir -p {remote_path}')

        # Build rsync command
        rsync_cmd = [
            'rsync', '-az', '--delete',
            '-e'
        ]

        # SSH options
        ssh_opts = 'ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=5'
        if self.config.ssh_key:
            ssh_opts = f'ssh -i {self.config.ssh_key} -o ServerAliveInterval=30 -o ServerAliveCountMax=5'

        rsync_cmd.append(ssh_opts)

        # Add exclusions
        for pattern in self.rsync_exclude_patterns:
            rsync_cmd.extend(['--exclude', pattern])

        # Use SSH config alias if no user specified, otherwise use user@host
        if self.config.ssh_user:
            remote_target = f'{self.config.ssh_user}@{self.config.ssh_host}:{remote_path}/'
        else:
            remote_target = f'{self.config.ssh_host}:{remote_path}/'

        # Add source and destination
        rsync_cmd.append(f'{local_root}/')
        rsync_cmd.append(remote_target)

        # Execute rsync
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"Codebase synced ({elapsed:.1f}s)")

    def submit_jobs(
        self,
        samples_csv: str,
        test_stats_csv: str,
        model_script: str,
        num_simulations: int,
        project_name: str,
        seed: int = 2025,
        jobs_per_chunk: Optional[int] = None,
        skip_sync: bool = False,
        save_full_simulations: bool = False,
        simulation_pool_id: Optional[str] = None
    ) -> JobInfo:
        """
        Submit batch jobs to HPC cluster.

        Args:
            samples_csv: Path to CSV file with parameter samples
            test_stats_csv: Path to test statistics CSV (defines scenario/observables)
            model_script: MATLAB model script name (e.g., 'immune_oncology_model_PDAC')
            num_simulations: Number of simulations
            project_name: Project name (e.g., 'pdac_2025')
            seed: Random seed
            jobs_per_chunk: Simulations per job (default from config)
            skip_sync: Skip codebase sync (for testing)

        Returns:
            JobInfo object with job IDs and state file
        """
        if jobs_per_chunk is None:
            jobs_per_chunk = self.config.jobs_per_chunk

        from qsp_hpc.batch.batch_utils import calculate_num_tasks
        n_jobs = calculate_num_tasks(num_simulations, jobs_per_chunk)

        # Sync codebase
        self.sync_codebase(skip_sync=skip_sync)

        # Ensure Python venv is set up on HPC
        self.ensure_hpc_venv()

        # Setup remote directories
        self._setup_remote_directories(project_name)

        # Create and upload job config (JSON)
        self._upload_job_config(
            test_stats_csv=test_stats_csv,
            model_script=model_script,
            num_simulations=num_simulations,
            seed=seed,
            jobs_per_chunk=jobs_per_chunk,
            project_name=project_name,
            save_full_simulations=save_full_simulations,
            simulation_pool_id=simulation_pool_id
        )

        # Upload parameter CSV
        self._upload_parameter_csv(samples_csv, project_name)

        # Upload test statistics CSV and functions
        self._upload_test_statistics(test_stats_csv, project_name)

        # Submit SLURM job
        job_id = self._submit_slurm_job(n_jobs, project_name)

        # Save job state
        job_info = JobInfo(
            job_ids=[job_id],
            state_file='',  # Will be set below
            n_jobs=n_jobs,
            n_simulations=num_simulations,
            project_name=project_name,
            submission_time=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        state_file = self._save_job_state(job_info, project_name)
        job_info.state_file = state_file

        return job_info

    def _setup_remote_directories(self, project_name: str) -> None:
        """Create necessary directories on remote cluster and clean old files."""
        remote_base = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs"
        dirs = ['input', 'output', 'scripts', 'logs']

        for d in dirs:
            remote_dir = f"{remote_base}/{d}"
            # Remove all files in directory, then recreate it
            self._ssh_exec(f'rm -rf "{remote_dir}" && mkdir -p "{remote_dir}"')

    def _upload_job_config(
        self,
        test_stats_csv: str,
        model_script: str,
        num_simulations: int,
        seed: int,
        jobs_per_chunk: int,
        project_name: str,
        save_full_simulations: bool = True,
        simulation_pool_id: str = None
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

            self._ssh_upload(temp_file, remote_file)
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"Job config uploaded ({elapsed:.1f}s)")
        finally:
            Path(temp_file).unlink()

    def _upload_parameter_csv(self, csv_path: str, project_name: str) -> None:
        """Upload parameter samples CSV."""
        start_time = time.time()

        remote_input_dir = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/input"
        remote_file = f"{remote_input_dir}/params.csv"

        self._ssh_upload(csv_path, remote_file)
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"Parameters uploaded ({elapsed:.1f}s)")

    def _upload_test_statistics(self, test_stats_csv: str, project_name: str) -> None:
        """Upload test statistics CSV and extract embedded functions as tarball."""
        start_time = time.time()

        remote_input_dir = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/input"

        # Upload CSV file
        remote_csv = f"{remote_input_dir}/test_stats.csv"
        self._ssh_upload(test_stats_csv, remote_csv)

        # Load CSV to extract embedded functions
        try:
            import pandas as pd
            import tarfile
            import shutil

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
                    self._ssh_upload(str(tarball_path), remote_tarball)

                    # Extract on remote
                    extract_cmd = f'cd "{remote_input_dir}" && tar -xzf test_stat_functions.tar.gz && rm test_stat_functions.tar.gz'
                    self._ssh_exec(extract_cmd)

                    if self.verbose:
                        elapsed = time.time() - start_time
                        print(f"Test statistics uploaded ({n_functions} functions, {elapsed:.1f}s)")

                finally:
                    # Clean up temp directory
                    shutil.rmtree(temp_dir)
            else:
                if self.verbose:
                    elapsed = time.time() - start_time
                    print(f"Test statistics uploaded ({elapsed:.1f}s)")

        except Exception as e:
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"Test statistics uploaded ({elapsed:.1f}s)")

    def _submit_slurm_job(self, n_jobs: int, project_name: str) -> str:
        """Generate and submit SLURM array job."""
        start_time = time.time()

        # Log SLURM configuration
        print(f"   SLURM Configuration:")
        print(f"     Partition: {self.config.partition}")
        print(f"     Time limit: {self.config.time_limit}")
        print(f"     Memory per job: {self.config.memory_per_job}")
        print(f"     Array size: {n_jobs} tasks")

        # Generate SLURM script
        script_content = self._generate_slurm_script(n_jobs, project_name)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            # Upload script
            remote_script_dir = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/scripts"
            remote_script = f"{remote_script_dir}/qsp_batch_job.sh"

            self._ssh_upload(temp_script, remote_script)

            # Submit job
            status, output = self._ssh_exec(f'sbatch "{remote_script}"')

            if status != 0:
                raise RuntimeError(f"SLURM submission failed: {output}")

            # Extract job ID
            import re
            match = re.search(r'Submitted batch job (\d+)', output)
            if not match:
                raise RuntimeError(f"Could not parse job ID from: {output}")

            job_id = match.group(1)
            elapsed = time.time() - start_time
            print(f"   🚀 Job {job_id} ({n_jobs} tasks, {elapsed:.1f}s)")

            return job_id

        finally:
            Path(temp_script).unlink()

    def _generate_slurm_script(self, n_jobs: int, project_name: str) -> str:
        """Generate SLURM batch script."""
        project_path = self.config.remote_project_path
        log_dir = f"{project_path}/projects/{project_name}/batch_jobs/logs"

        script = f"""#!/bin/bash
#SBATCH --job-name=qsp_batch
#SBATCH --partition={self.config.partition}
#SBATCH --time={self.config.time_limit}
#SBATCH --mem={self.config.memory_per_job}
#SBATCH --array=0-{n_jobs-1}
#SBATCH --output={log_dir}/qsp_batch_%A_%a.out
#SBATCH --error={log_dir}/qsp_batch_%A_%a.err

echo "Starting QSP batch job at $(date)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

# Export HPC paths for MATLAB scripts to use
export HPC_VENV_PATH="{self.config.hpc_venv_path}"
export SIMULATION_POOL_PATH="{self.config.simulation_pool_path}"

module load {self.config.matlab_module}
cd "{project_path}"
matlab -nodisplay -nodesktop -nosplash -r "batch_worker('{project_name}'); exit"
echo "Job completed at $(date)"
"""
        return script

    def _save_job_state(self, job_info: JobInfo, project_name: str) -> str:
        """Save job state to file."""
        state_dir = Path(f"projects/{project_name}/batch_jobs")
        state_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        state_file = state_dir / f"job_state_{timestamp}.pkl"

        with open(state_file, 'wb') as f:
            pickle.dump(asdict(job_info), f)

        return str(state_file)

    def collect_results(self, state_file: str) -> np.ndarray:
        """
        Collect results from completed HPC jobs.

        Args:
            state_file: Path to job state file

        Returns:
            Numpy array of observables (test statistics)
        """
        # Load job state
        with open(state_file, 'rb') as f:
            job_state = pickle.load(f)

        project_name = job_state['project_name']

        # Combine chunks on HPC
        self._combine_chunks_remotely(project_name)

        # Download combined results
        observables = self._download_combined_results(project_name)

        # Clean up state file
        Path(state_file).unlink(missing_ok=True)

        return observables

    def _combine_chunks_remotely(self, project_name: str) -> None:
        """Combine chunk CSV files on HPC."""
        remote_output = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/output"

        # Check that chunk files exist
        check_cmd = f'cd "{remote_output}" && ls chunk_*_test_stats.csv 2>/dev/null | wc -l'
        status, output = self._ssh_exec(check_cmd)
        num_chunks = int(output.strip()) if output.strip().isdigit() else 0

        if num_chunks == 0:
            raise RuntimeError(
                f"No chunk output files found in {remote_output}. "
                "Jobs may have failed or not produced output files."
            )

        # Combine test stats
        combine_cmd = f'cd "{remote_output}" && cat chunk_*_test_stats.csv > combined_test_stats.csv'
        status, output = self._ssh_exec(combine_cmd)

        if status != 0:
            raise RuntimeError(f"Failed to combine test stats: {output}")

    def _download_combined_results(self, project_name: str) -> np.ndarray:
        """Download and load combined results."""
        remote_output = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/output"

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Download combined CSV
            remote_file = f"{remote_output}/combined_test_stats.csv"
            self._ssh_download(remote_file, str(temp_dir))

            # Load CSV
            local_file = temp_dir / "combined_test_stats.csv"
            observables = np.loadtxt(local_file, delimiter=',', ndmin=2)
            # Ensure 2D shape (num_simulations, num_observables)
            if observables.ndim == 1:
                observables = observables.reshape(1, -1)

            return observables

        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)

    def _check_pool_directory_exists(self, pool_path: str) -> bool:
        """Check if simulation pool directory exists on HPC."""
        check_dir_cmd = f'test -d "{pool_path}" && echo "exists" || echo "not_found"'
        status, output = self._ssh_exec(check_dir_cmd)

        if self.verbose:
            print(f"Directory check result: {output.strip()}")

        return 'not_found' not in output

    def _count_pool_simulations(self, pool_path: str) -> int:
        """Count number of simulations in pool from manifest or filenames."""
        count_cmd = f'''
            cd "{pool_path}" 2>/dev/null || exit 1

            if [ -f manifest.json ]; then
                echo "MANIFEST_FOUND"
                cat manifest.json
            else
                echo "COUNTING_FILES"
                ls batch_*.parquet 2>/dev/null | wc -l | awk '{{print "N_FILES:" $1}}'
                ls batch_*.parquet 2>/dev/null | \
                grep -oE '[0-9]+sims' | \
                sed 's/sims//' | \
                awk '{{sum+=$1}} END {{print "N_SIMS:" sum}}'
            fi
        '''
        status, output = self._ssh_exec(count_cmd)

        if self.verbose:
            print(f"Count command output:")
            for line in output.strip().split('\n'):
                print(f"  {line}")

        if status != 0:
            if self.verbose:
                print(f"Failed to count simulations (status={status})")
            return 0

        n_available = 0

        try:
            # Check if we got manifest
            if 'MANIFEST_FOUND' in output:
                # Extract JSON (everything after MANIFEST_FOUND line)
                lines = output.split('\n')
                manifest_start = lines.index('MANIFEST_FOUND') + 1
                manifest_json = '\n'.join(lines[manifest_start:])

                import json
                manifest = json.loads(manifest_json)
                n_available = manifest.get('total_simulations', 0)
                if self.verbose:
                    print(f"Parsed manifest: {n_available} simulations")

            elif 'COUNTING_FILES' in output:
                # Extract N_SIMS value
                for line in output.split('\n'):
                    if line.startswith('N_SIMS:'):
                        n_available = int(line.split(':')[1].strip())
                        if self.verbose:
                            print(f"Counted from filenames: {n_available} simulations")
                        break
            else:
                if self.verbose:
                    print(f"Could not parse output format")

        except Exception as e:
            if self.verbose:
                print(f"Error parsing count: {e}")
            n_available = 0

        return n_available

    def check_hpc_full_simulations(
        self,
        model_version: str,
        priors_hash: str,
        n_requested: int
    ) -> Tuple[bool, str, int]:
        """
        Check if HPC has enough full simulations in persistent storage.

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
            print(f"Checking pool directory: {pool_path}")

        # Check if pool directory exists
        if not self._check_pool_directory_exists(pool_path):
            if self.verbose:
                print(f"Pool directory does not exist on HPC")
            return False, pool_path, 0

        # Count simulations in pool
        n_available = self._count_pool_simulations(pool_path)

        has_enough = n_available >= n_requested

        if self.verbose:
            print(f"Found {n_available} simulations, need {n_requested}: {'sufficient' if has_enough else 'insufficient'}")

        return has_enough, pool_path, n_available

    def _combine_chunks_on_hpc(self, test_stats_dir: str) -> None:
        """Combine test statistics chunks on HPC using Python script."""
        if self.verbose:
            print(f"Combining chunk files on HPC...")

        # Upload combine script (small file, quick to upload)
        local_script = Path('metadata/combine_test_stats_chunks.py')
        remote_script = f"{test_stats_dir}/combine_chunks.py"
        self._ssh_upload(str(local_script), remote_script)

        # Run combine script using HPC venv
        combine_cmd = f'{self.config.hpc_venv_path}/bin/python "{remote_script}" "{test_stats_dir}"'
        status, output = self._ssh_exec(combine_cmd, timeout=60)

        if self.verbose:
            print(f"HPC combine output:")
            for line in output.strip().split('\n'):
                print(f"  {line}")

        if status != 0:
            raise RuntimeError(f"Failed to combine chunks on HPC: {output}")

    def _download_combined_files(
        self,
        test_stats_dir: str,
        local_dest: Path
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Download and load combined test statistics and parameters from HPC."""
        local_dest.mkdir(parents=True, exist_ok=True)

        # Download test stats
        remote_test_stats_file = f"{test_stats_dir}/combined_test_stats.csv"
        if self.verbose:
            print(f"Downloading combined test stats...")
        self._ssh_download(remote_test_stats_file, str(local_dest))

        # Check for combined params
        check_params_cmd = f'test -f "{test_stats_dir}/combined_params.csv" && echo "exists"'
        status_params, output_params = self._ssh_exec(check_params_cmd)
        has_params = (status_params == 0 and 'exists' in output_params)

        params = None
        if has_params:
            remote_params_file = f"{test_stats_dir}/combined_params.csv"
            if self.verbose:
                print(f"Downloading combined params...")
            self._ssh_download(remote_params_file, str(local_dest))

        # Rename downloaded files
        downloaded_test_stats = local_dest / "combined_test_stats.csv"
        local_test_stats_file = local_dest / "test_stats.csv"

        if downloaded_test_stats.exists():
            downloaded_test_stats.rename(local_test_stats_file)

        if has_params:
            downloaded_params = local_dest / "combined_params.csv"
            local_params_file = local_dest / "params.csv"

            if downloaded_params.exists():
                downloaded_params.rename(local_params_file)

                # Load params
                import pandas as pd
                params_df = pd.read_csv(local_params_file)
                params = params_df.values

                # Ensure 2D shape
                if params.ndim == 1:
                    params = params.reshape(1, -1)

                if self.verbose:
                    print(f"Downloaded parameters: {params.shape}")

        # Load test stats using pandas (handles NaN/empty values properly)
        import pandas as pd
        test_stats_df = pd.read_csv(local_test_stats_file, header=None)
        test_stats = test_stats_df.values

        # Ensure 2D shape
        if test_stats.ndim == 1:
            test_stats = test_stats.reshape(1, -1)

        if self.verbose:
            print(f"Downloaded test statistics: {test_stats.shape}")

        return params, test_stats

    def check_hpc_test_stats(
        self,
        pool_path: str,
        test_stats_hash: str,
        expected_n_sims: Optional[int] = None
    ) -> bool:
        """
        Check if HPC has derived test statistics for given configuration.

        Args:
            pool_path: Path to simulation pool on HPC
            test_stats_hash: Hash of test statistics CSV
            expected_n_sims: Expected number of simulations (if provided, validates count)

        Returns:
            True if derived test statistics exist and match expected count
        """
        test_stats_dir = f"{pool_path}/test_stats/{test_stats_hash}"

        # Check if test stats directory exists and has both params and test stats chunk files
        # (derivation worker creates chunk_XXX_params.csv and chunk_XXX_test_stats.csv files)
        check_cmd = f'''
            test -d "{test_stats_dir}" || exit 1
            echo "TEST_STATS_CHUNKS:$(ls "{test_stats_dir}"/chunk_*_test_stats.csv 2>/dev/null | wc -l)"
            echo "PARAMS_CHUNKS:$(ls "{test_stats_dir}"/chunk_*_params.csv 2>/dev/null | wc -l)"
        '''
        status, output = self._ssh_exec(check_cmd)

        if status != 0:
            return False

        # Parse chunk counts
        try:
            n_test_stats_chunks = 0
            n_params_chunks = 0
            for line in output.strip().split('\n'):
                if 'TEST_STATS_CHUNKS:' in line:
                    n_test_stats_chunks = int(line.split(':')[1])
                elif 'PARAMS_CHUNKS:' in line:
                    n_params_chunks = int(line.split(':')[1])

            # Both must have at least one chunk
            if n_test_stats_chunks == 0:
                print(f"   No test stats chunks found")
                return False

            # Params chunks may not exist for older datasets (backward compatibility)
            if n_params_chunks == 0:
                print(f"  Warning:  No params chunks found (older format without parameters)")
            else:
                print(f"   Found {n_test_stats_chunks} test stats chunks and {n_params_chunks} params chunks")

        except Exception as e:
            print(f"   Error parsing chunk counts: {e}")
            return False

        # If expected count provided, validate that derived test stats match pool size
        if expected_n_sims is not None:
            # Count rows in combined test stats (or combine chunks first if needed)
            count_cmd = f'''
                cd "{test_stats_dir}" 2>/dev/null || exit 1

                # Check if combined file exists, otherwise combine chunks
                if [ ! -f combined_test_stats.csv ]; then
                    cat chunk_*_test_stats.csv > combined_test_stats.csv 2>/dev/null
                fi

                # Count lines in combined file
                if [ -f combined_test_stats.csv ]; then
                    wc -l < combined_test_stats.csv
                else
                    echo "0"
                fi
            '''
            status, output = self._ssh_exec(count_cmd)

            if status == 0:
                try:
                    n_derived = int(output.strip())
                    if n_derived != expected_n_sims:
                        print(f"  Warning:  Derived test stats count mismatch: {n_derived} vs {expected_n_sims} expected")
                        print(f"   Will re-derive from complete pool")
                        # Delete old derived test stats so they get re-derived
                        self._ssh_exec(f'rm -rf "{test_stats_dir}"')
                        return False
                    else:
                        print(f"   Derived test stats count matches: {n_derived} simulations")
                except ValueError:
                    pass

        return True

    def submit_derivation_job(
        self,
        pool_path: str,
        test_stats_csv: str,
        test_stats_hash: str,
        project_name: str = 'pdac_2025'
    ) -> str:
        """
        Submit SLURM job to derive test statistics from full simulations.

        Args:
            pool_path: Path to simulation pool on HPC (e.g., {simulation_pool_path}/baseline_pdac_abc12345)
            test_stats_csv: Local path to test statistics CSV
            test_stats_hash: Hash of test statistics CSV
            project_name: Project name for logging

        Returns:
            SLURM job ID
        """
        print(f"   Submitting test statistics derivation job...")

        # Ensure Python venv is set up
        self.ensure_hpc_venv()

        # Create persistent directory for derivation inputs (in batch_jobs)
        derivation_dir = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/derivation"
        self._ssh_exec(f'mkdir -p "{derivation_dir}"')

        # Upload test statistics CSV
        remote_test_stats_csv = f"{derivation_dir}/test_stats_{test_stats_hash[:8]}.csv"
        self._ssh_upload(test_stats_csv, remote_test_stats_csv)

        # Upload test_stat_functions.py
        local_test_stat_funcs = Path('metadata/test_stat_functions.py')
        remote_test_stat_funcs = f"{derivation_dir}/test_stat_functions.py"
        self._ssh_upload(str(local_test_stat_funcs), remote_test_stat_funcs)

        # Expand $HOME in pool_path (Python won't expand shell variables)
        # Get the actual home directory from HPC
        status, home_dir = self._ssh_exec('echo $HOME')
        home_dir = home_dir.strip()
        expanded_pool_path = pool_path.replace('$HOME', home_dir)

        print(f"   Expanded pool path: {expanded_pool_path}")

        # Create derivation config JSON
        config = {
            'simulation_pool_dir': expanded_pool_path,
            'test_stats_csv': remote_test_stats_csv,
            'output_dir': expanded_pool_path,
            'test_stats_hash': test_stats_hash
        }

        # Write config locally then upload
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config, f, indent=2)
            temp_config = f.name

        remote_config = f"{derivation_dir}/derive_config_{test_stats_hash[:8]}.json"
        self._ssh_upload(temp_config, remote_config)
        Path(temp_config).unlink()

        # Count number of Parquet files (one job per file)
        status, output = self._ssh_exec(f'ls "{pool_path}"/batch_*.parquet | wc -l')
        n_batches = int(output.strip()) if output.strip().isdigit() else 1

        # Log SLURM configuration for derivation job
        print(f"   SLURM Derivation Configuration:")
        print(f"     Partition: {self.config.partition}")
        print(f"     Time limit: 01:00:00 (fixed for derivation)")
        print(f"     Memory per job: 4G (fixed for derivation)")
        print(f"     Array size: {n_batches} tasks")

        # Generate SLURM script
        slurm_script = self._generate_derivation_slurm_script(
            pool_path, remote_config, derivation_dir, n_batches, project_name
        )

        # Upload SLURM script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(slurm_script)
            temp_slurm = f.name

        remote_slurm = f"{derivation_dir}/derive_job_{test_stats_hash[:8]}.sh"
        self._ssh_upload(temp_slurm, remote_slurm)
        Path(temp_slurm).unlink()

        # Submit SLURM job
        status, output = self._ssh_exec(f'sbatch "{remote_slurm}"')

        if status != 0:
            raise RuntimeError(f"SLURM derivation job submission failed: {output}")

        # Extract job ID
        import re
        match = re.search(r'Submitted batch job (\d+)', output)
        if not match:
            raise RuntimeError(f"Could not parse job ID from: {output}")

        job_id = match.group(1)
        print(f"   🚀 Derivation job {job_id} ({n_batches} tasks)")

        return job_id

    def _generate_derivation_slurm_script(
        self,
        pool_path: str,
        config_file: str,
        derivation_dir: str,
        n_batches: int,
        project_name: str
    ) -> str:
        """Generate SLURM script for test statistics derivation."""
        project_path = self.config.remote_project_path
        log_dir = f"{project_path}/projects/{project_name}/batch_jobs/logs"

        script = f"""#!/bin/bash
#SBATCH --job-name=qsp_derive
#SBATCH --partition={self.config.partition}
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --array=0-{n_batches-1}
#SBATCH --output={log_dir}/qsp_derive_%A_%a.out
#SBATCH --error={log_dir}/qsp_derive_%A_%a.err

echo "Starting test statistics derivation at $(date)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

# Activate Python virtual environment
VENV_DIR="{self.config.hpc_venv_path}"
if [ -d "$VENV_DIR" ]; then
    echo "Activating venv: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "WARNING: Virtual environment not found at $VENV_DIR"
    echo "Please run: bash scripts/setup_hpc_venv.sh"
    exit 1
fi

# Add project root and derivation dir to Python path
export PYTHONPATH="{project_path}:{derivation_dir}:$PYTHONPATH"

cd "{project_path}"
python metadata/batch/derive_test_stats_worker.py "{config_file}"

echo "Derivation completed at $(date)"
"""
        return script

    def download_test_stats(
        self,
        pool_path: str,
        test_stats_hash: str,
        local_dest: Path
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Download and combine derived parameters and test statistics from HPC.

        Args:
            pool_path: Path to simulation pool on HPC (may contain $HOME)
            test_stats_hash: Hash of test statistics CSV
            local_dest: Local destination directory

        Returns:
            Tuple of (params, test_stats):
            - params: Numpy array of parameters (n_sims x n_params) or None if not available
            - test_stats: Numpy array of test statistics (n_sims x n_test_stats)
        """
        # Expand $HOME if present (needed for scp)
        if '$HOME' in pool_path:
            status, home_dir = self._ssh_exec('echo $HOME')
            pool_path = pool_path.replace('$HOME', home_dir.strip())
            if self.verbose:
                print(f"Expanded pool path: {pool_path}")

        test_stats_dir = f"{pool_path}/test_stats/{test_stats_hash}"

        if self.verbose:
            print(f"Test stats directory: {test_stats_dir}")

        # Check directory exists
        check_cmd = f'test -d "{test_stats_dir}" && ls -la "{test_stats_dir}" || echo "DIRECTORY_NOT_FOUND"'
        status, output = self._ssh_exec(check_cmd)

        if self.verbose:
            print(f"Directory listing:")
            for line in output.strip().split('\n')[:10]:  # Show first 10 lines
                print(f"  {line}")

        if 'DIRECTORY_NOT_FOUND' in output:
            # Determine likely project name for better error message
            project_name = 'pdac_2025'  # Could be passed as parameter if needed
            log_path = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/logs"

            raise RuntimeError(
                f"Test statistics directory not found on HPC: {test_stats_dir}\n"
                f"This suggests the derivation job failed. Check logs on HPC:\n"
                f"  {log_path}/qsp_derive_*.err"
            )

        # Combine chunks on HPC
        self._combine_chunks_on_hpc(test_stats_dir)

        # Download combined files and load
        return self._download_combined_files(test_stats_dir, local_dest)

    def download_latest_parquet_batch(
        self,
        pool_path: str,
        local_dest: Path,
        n_files: int = 1
    ) -> List[Path]:
        """
        Download the most recent Parquet batch file(s) from HPC simulation pool.

        Args:
            pool_path: Path to simulation pool on HPC (may contain $HOME)
            local_dest: Local destination directory
            n_files: Number of most recent files to download (default: 1)

        Returns:
            List of local paths to downloaded Parquet files
        """
        # Expand $HOME if present
        if '$HOME' in pool_path:
            status, home_dir = self._ssh_exec('echo $HOME')
            pool_path = pool_path.replace('$HOME', home_dir.strip())

        print(f"   Downloading {n_files} most recent Parquet batch(es) from HPC...")
        print(f"   Pool path: {pool_path}")

        # List Parquet files sorted by modification time (most recent first)
        list_cmd = f'ls -t "{pool_path}"/batch_*.parquet 2>/dev/null | head -{n_files}'
        status, output = self._ssh_exec(list_cmd)

        if status != 0 or not output.strip():
            raise RuntimeError(f"No Parquet files found in {pool_path}")

        parquet_files = output.strip().split('\n')
        print(f"   Found {len(parquet_files)} recent file(s)")

        # Create local destination
        local_dest.mkdir(parents=True, exist_ok=True)

        # Download each file
        downloaded_files = []
        for remote_file in parquet_files:
            remote_file = remote_file.strip()
            if not remote_file:
                continue

            filename = Path(remote_file).name
            print(f"   Downloading {filename}...")

            self._ssh_download(remote_file, str(local_dest))
            local_file = local_dest / filename
            downloaded_files.append(local_file)

        print(f"   Downloaded {len(downloaded_files)} Parquet file(s)")

        return downloaded_files

    def parse_parquet_simulations(
        self,
        parquet_file: Path,
        species_of_interest: Optional[List[str]] = None,
        max_simulations: Optional[int] = None
    ) -> Dict:
        """
        Parse Parquet file containing full simulation data.

        Args:
            parquet_file: Path to local Parquet file
            species_of_interest: Optional list of species to extract (default: all)
            max_simulations: Optional limit on number of simulations to load

        Returns:
            Dictionary containing:
            - 'n_simulations': Number of simulations
            - 'time': Time vector (n_timepoints,)
            - 'simulations': Dict mapping species name to array (n_sims, n_timepoints)
            - 'species_names': List of species names
            - 'simulation_ids': Array of simulation IDs
            - 'statuses': Array of simulation statuses
        """
        print(f"   Parsing Parquet file: {parquet_file.name}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for Parquet parsing. Install with: pip install pandas pyarrow")

        # Read Parquet file
        df = pd.read_parquet(parquet_file)

        print(f"   Loaded {len(df)} simulations")
        print(f"   Columns: {len(df.columns)} ({df.columns[0]}, {df.columns[1]}, ...)")

        # Extract metadata columns
        simulation_ids = df['simulation_id'].values
        statuses = df['status'].values

        # Filter to successful simulations only
        success_mask = (statuses == 1)
        n_successful = np.sum(success_mask)

        if n_successful == 0:
            raise ValueError(f"No successful simulations found in {parquet_file}")

        print(f"   {n_successful}/{len(df)} simulations successful")

        # Apply max_simulations limit
        if max_simulations is not None and n_successful > max_simulations:
            print(f"   Limiting to first {max_simulations} successful simulations")
            # Get indices of successful simulations
            success_indices = np.where(success_mask)[0]
            selected_indices = success_indices[:max_simulations]
            success_mask = np.zeros(len(df), dtype=bool)
            success_mask[selected_indices] = True
            n_successful = max_simulations

        # Extract time vector (from first successful simulation)
        first_success_idx = np.where(success_mask)[0][0]
        time = np.array(df.iloc[first_success_idx]['time'])

        print(f"   Time points: {len(time)} ({time[0]:.1f} to {time[-1]:.1f})")

        # Get species columns (exclude metadata: simulation_id, status, time)
        metadata_cols = {'simulation_id', 'status', 'time'}
        species_names = [col for col in df.columns if col not in metadata_cols]

        print(f"   Species: {len(species_names)} total")

        # Filter species if requested
        if species_of_interest is not None:
            # Map species names (replace dots with underscores)
            species_map = {name.replace('.', '_'): name for name in species_names}

            selected_species = []
            for requested_species in species_of_interest:
                # Try exact match first
                if requested_species in species_names:
                    selected_species.append(requested_species)
                # Try with underscore mapping
                elif requested_species in species_map:
                    selected_species.append(species_map[requested_species])
                else:
                    print(f"  Warning:  Warning: Species '{requested_species}' not found")

            if not selected_species:
                raise ValueError(f"None of the requested species found in Parquet file")

            species_names = selected_species
            print(f"   Selected {len(species_names)} species")

        # Extract simulation data for each species
        simulations = {}
        for species_name in species_names:
            # Extract time series for all successful simulations
            species_data = []

            for idx in np.where(success_mask)[0]:
                trajectory = np.array(df.iloc[idx][species_name])
                species_data.append(trajectory)

            # Stack into array (n_sims, n_timepoints)
            species_array = np.array(species_data)
            simulations[species_name] = species_array

        print(f"   Extracted {len(species_names)} species x {n_successful} simulations")

        return {
            'n_simulations': n_successful,
            'time': time,
            'simulations': simulations,
            'species_names': species_names,
            'simulation_ids': simulation_ids[success_mask],
            'statuses': statuses[success_mask]
        }


# Convenience function
def create_hpc_manager(config_file: str = 'batch_credentials.yaml') -> HPCJobManager:
    """
    Create HPC job manager from config file.

    Args:
        config_file: Path to batch credentials YAML

    Returns:
        HPCJobManager instance
    """
    return HPCJobManager()
