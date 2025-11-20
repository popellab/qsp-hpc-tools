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

import pickle
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from qsp_hpc.batch.hpc_file_transfer import HPCFileTransfer
from qsp_hpc.batch.result_collector import MissingOutputError, ResultCollector
from qsp_hpc.batch.slurm_job_submitter import SLURMJobSubmitter, SubmissionError  # noqa: F401
from qsp_hpc.utils.logging_config import format_config, log_operation, setup_logger
from qsp_hpc.utils.security import (
    build_safe_ssh_command,
    validate_project_name,
)


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
    ssh_key: str = ""
    remote_project_path: str = ""
    partition: str = "shared"
    time_limit: str = "20:00"
    memory_per_job: str = "2G"
    matlab_module: str = "matlab/R2024a"
    jobs_per_chunk: int = 20
    strict_host_key_checking: bool = True  # Security: verify SSH host keys by default
    qsp_hpc_tools_source: str = "git+ssh://git@github.com/jeliason/qsp-hpc-tools.git"


class RemoteCommandError(RuntimeError):
    """Raised when a remote command fails."""

    def __init__(self, command: str, returncode: int, output: str):
        self.command = command
        self.returncode = returncode
        self.output = output
        super().__init__(f"Command failed (rc={returncode}): {command}\n{output}")


class SSHTransport:
    """Thin SSH/SCP transport layer to allow swapping/mocking."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self._warned_about_host_key_checking = False

    def _build_ssh_target(self) -> str:
        if self.config.ssh_user:
            return f"{self.config.ssh_user}@{self.config.ssh_host}"
        return self.config.ssh_host

    def _warn_insecure_ssh(self):
        """Warn user once about disabled host key checking."""
        if not self._warned_about_host_key_checking and not self.config.strict_host_key_checking:
            import warnings

            warnings.warn(
                "SSH host key verification is DISABLED. This makes connections vulnerable to "
                "man-in-the-middle attacks. Set 'strict_host_key_checking: true' in credentials.yaml "
                "and add the host key to ~/.ssh/known_hosts for better security.",
                category=UserWarning,
                stacklevel=3,
            )
            self._warned_about_host_key_checking = True

    def exec(self, command: str, timeout: Optional[int] = 30) -> Tuple[int, str]:
        """
        Execute command on remote host via SSH.

        Args:
            command: Shell command to execute (should be pre-escaped if needed)
            timeout: Timeout in seconds

        Returns:
            Tuple of (return_code, combined_output)

        Raises:
            subprocess.TimeoutExpired: If command exceeds timeout

        Note:
            This method does not automatically escape arguments. Use build_safe_ssh_command()
            from qsp_hpc.utils.security for safe command construction.
        """
        self._warn_insecure_ssh()

        ssh_cmd = ["ssh"]

        if self.config.ssh_key:
            ssh_cmd.extend(["-i", self.config.ssh_key])

        ssh_cmd.extend(
            [
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=5",
                "-o",
                f'StrictHostKeyChecking={"yes" if self.config.strict_host_key_checking else "no"}',
                "-o",
                "BatchMode=yes",
            ]
        )

        ssh_cmd.append(self._build_ssh_target())
        ssh_cmd.append(command)

        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)

        return result.returncode, result.stdout + result.stderr

    def upload(self, local_path: str, remote_path: str) -> None:
        """
        Upload file to remote host via SCP.

        Args:
            local_path: Local file path
            remote_path: Remote destination path

        Raises:
            subprocess.CalledProcessError: If upload fails
        """
        self._warn_insecure_ssh()

        scp_cmd = ["scp"]

        if self.config.ssh_key:
            scp_cmd.extend(["-i", self.config.ssh_key])

        remote_target = f"{self._build_ssh_target()}:{remote_path}"

        scp_cmd.extend(
            [
                "-o",
                f'StrictHostKeyChecking={"yes" if self.config.strict_host_key_checking else "no"}',
                "-o",
                "BatchMode=yes",
                local_path,
                remote_target,
            ]
        )

        try:
            subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RemoteCommandError(
                f"scp upload to {remote_path}", exc.returncode, exc.stderr or str(exc)
            ) from exc

    def download(self, remote_path: str, local_dir: str) -> None:
        """
        Download file from remote host via SCP.

        Args:
            remote_path: Remote file path
            local_dir: Local destination directory

        Raises:
            subprocess.CalledProcessError: If download fails
        """
        self._warn_insecure_ssh()

        scp_cmd = ["scp"]

        if self.config.ssh_key:
            scp_cmd.extend(["-i", self.config.ssh_key])

        remote_source = f"{self._build_ssh_target()}:{remote_path}"

        scp_cmd.extend(
            [
                "-o",
                f'StrictHostKeyChecking={"yes" if self.config.strict_host_key_checking else "no"}',
                remote_source,
                local_dir,
            ]
        )

        try:
            subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RemoteCommandError(
                f"scp download {remote_path}", exc.returncode, exc.stderr or str(exc)
            ) from exc


class HPCJobManager:
    """
    Manages HPC job submission and result collection for QSP simulations.

    This class provides a Python-based alternative to MATLAB's batch_execute.m,
    eliminating MATLAB startup overhead and enabling faster job submission.
    """

    def __init__(
        self,
        config: Union[Dict, BatchConfig, None] = None,
        verbose: bool = False,
        transport: Optional[SSHTransport] = None,
    ):
        """
        Initialize HPC job manager.

        Args:
            config: Batch configuration dict, BatchConfig object, or None to load from YAML
            verbose: If True, print detailed progress information (default: False)
        """
        if config is None:
            config = self._load_config_from_yaml()
        elif isinstance(config, dict):
            # Normalize common fields (e.g., expand ssh_key)
            cfg_copy = dict(config)
            if cfg_copy.get("ssh_key"):
                cfg_copy["ssh_key"] = str(Path(cfg_copy["ssh_key"]).expanduser())
            config = BatchConfig(**cfg_copy)

        self.config = config
        self.verbose = verbose
        self.logger = setup_logger(__name__, verbose=verbose)
        self.transport = transport or SSHTransport(self.config)

        # Initialize component classes (Composition over inheritance)
        self.slurm_submitter = SLURMJobSubmitter(self.config, self.transport, verbose)
        self.file_transfer = HPCFileTransfer(self.config, self.transport, verbose)
        self.result_collector = ResultCollector(self.config, self.transport, verbose)

    def _load_config_from_yaml(self) -> BatchConfig:
        """
        Load configuration from ~/.config/qsp-hpc/credentials.yaml

        Run 'qsp-hpc setup' to create this configuration file.
        """
        global_config_file = Path.home() / ".config" / "qsp-hpc" / "credentials.yaml"
        project_config_file = Path.cwd() / ".qsp-hpc" / "credentials.yaml"

        if not global_config_file.exists():
            raise FileNotFoundError(
                f"Configuration not found at {global_config_file}\n"
                "Please run 'qsp-hpc setup' to configure HPC connection."
            )

        with open(global_config_file, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

        # Layer project-specific overrides if present
        if project_config_file.exists():
            with open(project_config_file, "r") as f:
                project_cfg = yaml.safe_load(f) or {}
            yaml_config = self._merge_config_dicts(yaml_config, project_cfg)

        return self._parse_config_dict(yaml_config, source=global_config_file)

    @staticmethod
    def _parse_config_dict(cfg: Dict, source: Optional[Path] = None) -> BatchConfig:
        """Parse and validate a credentials dict into BatchConfig."""
        if not cfg:
            raise ValueError(
                f"Configuration file {source} is empty.\n"
                "Please run 'qsp-hpc setup' to configure HPC connection."
            )

        ssh = cfg.get("ssh", {})
        cluster = cfg.get("cluster", {})
        paths = cfg.get("paths", {})
        slurm = cfg.get("slurm", {})
        package = cfg.get("package", {})

        # Validate required SSH fields
        ssh_host = ssh.get("host", "").strip()
        if not ssh_host:
            raise ValueError(
                "ssh.host must be specified in credentials.yaml\n"
                "Please run 'qsp-hpc setup' to configure HPC connection."
            )

        # Validate required paths
        simulation_pool_path = paths.get("simulation_pool_path", "").strip()
        hpc_venv_path = paths.get("hpc_venv_path", "").strip()

        if not simulation_pool_path:
            raise ValueError(
                "paths.simulation_pool_path must be specified in credentials.yaml\n"
                "Please run 'qsp-hpc setup' to configure HPC connection."
            )
        if not hpc_venv_path:
            raise ValueError(
                "paths.hpc_venv_path must be specified in credentials.yaml\n"
                "Please run 'qsp-hpc setup' to configure HPC connection."
            )

        # Validate and expand SSH key path
        ssh_key = ssh.get("key", "").strip()
        if ssh_key:
            ssh_key_path = Path(ssh_key).expanduser()
            # Only validate if file should exist (not empty string)
            if not ssh_key_path.exists():
                raise ValueError(
                    f"SSH key file not found: {ssh_key_path}\n"
                    f"Please check ssh.key in credentials.yaml or run 'qsp-hpc setup'."
                )
            ssh_key = str(ssh_key_path)

        # Validate SLURM time limit format (HH:MM:SS or DD-HH:MM:SS)
        time_limit = slurm.get("time_limit", "01:00:00")
        if (
            not isinstance(time_limit, str)
            or not time_limit.replace("-", "").replace(":", "").isdigit()
        ):
            raise ValueError(
                f"Invalid SLURM time_limit format: {time_limit}\n"
                "Expected format: HH:MM:SS or DD-HH:MM:SS"
            )

        # Validate memory format (e.g., '4G', '512M')
        memory_per_job = slurm.get("mem_per_cpu", "4G")
        if not isinstance(memory_per_job, str) or memory_per_job[-1].upper() not in [
            "K",
            "M",
            "G",
            "T",
        ]:
            raise ValueError(
                f"Invalid memory format: {memory_per_job}\n"
                "Expected format: <number><unit> (e.g., '4G', '512M')"
            )

        return BatchConfig(
            ssh_host=ssh_host,
            ssh_user=ssh.get("user", "").strip(),
            simulation_pool_path=simulation_pool_path,
            hpc_venv_path=hpc_venv_path,
            ssh_key=ssh_key,
            remote_project_path=paths.get("remote_base_dir", "").strip(),
            partition=slurm.get("partition", "shared").strip(),
            time_limit=time_limit,
            memory_per_job=memory_per_job,
            matlab_module=cluster.get("matlab_module", "matlab/R2024a").strip(),
            strict_host_key_checking=ssh.get(
                "strict_host_key_checking", True
            ),  # Default to True for security
            qsp_hpc_tools_source=package.get(
                "qsp_hpc_tools_source", "git+ssh://git@github.com/jeliason/qsp-hpc-tools.git"
            ).strip(),
        )

    @staticmethod
    def _merge_config_dicts(base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base config dict."""
        if not override:
            return base

        merged = dict(base)
        for key, val in override.items():
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = HPCJobManager._merge_config_dicts(merged[key], val)
            else:
                merged[key] = val
        return merged

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
            status, output = self.transport.exec('echo "SSH_OK"', timeout=timeout)

            if status == 0 and "SSH_OK" in output:
                return True
            else:
                raise RuntimeError(f"SSH connection failed: {output}")

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"SSH connection timeout after {timeout}s") from exc
        except Exception as exc:
            raise RuntimeError(f"SSH connection error: {exc}") from exc

    def ensure_hpc_venv(self) -> None:
        """
        Ensure Python virtual environment is set up on HPC.

        Creates venv at configured hpc_venv_path if it doesn't exist and installs
        required packages for Parquet I/O and test statistics derivation.
        """
        return self.file_transfer.ensure_hpc_venv()

    def sync_codebase(self, skip_sync: bool = False) -> None:
        """
        Sync codebase to HPC using rsync.

        Args:
            skip_sync: If True, skip syncing (for testing)
        """
        return self.file_transfer.sync_codebase(skip_sync)

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
        simulation_pool_id: Optional[str] = None,
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

        Raises:
            SecurityError: If project_name contains invalid characters
        """
        # Validate project name for security (prevent path traversal/command injection)
        project_name = validate_project_name(project_name)

        if jobs_per_chunk is None:
            jobs_per_chunk = self.config.jobs_per_chunk

        from qsp_hpc.batch.batch_utils import calculate_num_tasks

        n_jobs = calculate_num_tasks(num_simulations, jobs_per_chunk)

        # Log job submission details
        self.logger.info("Preparing HPC job submission:")
        job_config = {
            "project": project_name,
            "simulations": num_simulations,
            "array_tasks": n_jobs,
            "sims_per_task": jobs_per_chunk,
            "model_script": model_script,
            "seed": seed,
        }
        for line in format_config(job_config):
            self.logger.info(line)

        # Sync codebase
        with log_operation(self.logger, "Syncing codebase to HPC", log_start=not skip_sync):
            self.sync_codebase(skip_sync=skip_sync)

        # Ensure Python venv is set up on HPC
        self.ensure_hpc_venv()

        # Setup remote directories
        self._setup_remote_directories(project_name)

        # Create and upload job config (JSON)
        self.logger.info("Uploading job configuration and inputs...")
        self._upload_job_config(
            test_stats_csv=test_stats_csv,
            model_script=model_script,
            num_simulations=num_simulations,
            seed=seed,
            jobs_per_chunk=jobs_per_chunk,
            project_name=project_name,
            save_full_simulations=save_full_simulations,
            simulation_pool_id=simulation_pool_id,
        )

        # Upload parameter CSV
        self._upload_parameter_csv(samples_csv, project_name)

        # Upload test statistics CSV and functions
        self._upload_test_statistics(test_stats_csv, project_name)

        # Submit SLURM job
        self.logger.info(f"Submitting SLURM array job with {n_jobs} tasks...")
        job_id = self._submit_slurm_job(n_jobs, project_name)
        self.logger.info(f"✓ Job submitted: {job_id}")

        # Save job state
        job_info = JobInfo(
            job_ids=[job_id],
            state_file="",  # Will be set below
            n_jobs=n_jobs,
            n_simulations=num_simulations,
            project_name=project_name,
            submission_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        state_file = self._save_job_state(job_info, project_name)
        job_info.state_file = state_file

        return job_info

    def _setup_remote_directories(self, project_name: str) -> None:
        """Create necessary directories on remote cluster and clean old files."""
        return self.file_transfer.setup_remote_directories(project_name)

    def _upload_job_config(
        self,
        test_stats_csv: str,
        model_script: str,
        num_simulations: int,
        seed: int,
        jobs_per_chunk: int,
        project_name: str,
        save_full_simulations: bool = True,
        simulation_pool_id: Optional[str] = None,
    ) -> None:
        """Create and upload job configuration as JSON."""
        return self.file_transfer.upload_job_config(
            test_stats_csv,
            model_script,
            num_simulations,
            seed,
            jobs_per_chunk,
            project_name,
            save_full_simulations,
            simulation_pool_id,
        )

    def _upload_parameter_csv(self, csv_path: str, project_name: str) -> None:
        """Upload parameter samples CSV."""
        return self.file_transfer.upload_parameter_csv(csv_path, project_name)

    def _upload_test_statistics(self, test_stats_csv: str, project_name: str) -> None:
        """Upload test statistics CSV and extract embedded functions as tarball."""
        return self.file_transfer.upload_test_statistics(test_stats_csv, project_name)

    def _submit_slurm_job(self, n_jobs: int, project_name: str) -> str:
        """Generate and submit SLURM array job."""
        return self.slurm_submitter.submit_job(n_jobs, project_name)

    def _generate_slurm_script(self, n_jobs: int, project_name: str) -> str:
        """Generate SLURM batch script."""
        return self.slurm_submitter._generate_slurm_script(n_jobs, project_name)

    def _save_job_state(self, job_info: JobInfo, project_name: str) -> str:
        """Save job state to file."""
        # Prefer local storage for state to avoid writing to remote-only paths
        state_root = (
            Path(self.config.remote_project_path)
            if (self.config.remote_project_path and Path(self.config.remote_project_path).exists())
            else Path.cwd()
        )

        base_dir = state_root / f"projects/{project_name}/batch_jobs"
        base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        state_file = base_dir / f"job_state_{timestamp}.pkl"

        try:
            with open(state_file, "wb") as f:
                pickle.dump(asdict(job_info), f)
        except OSError as exc:  # pragma: no cover - hard to trigger in tests
            raise RuntimeError(f"Failed to write job state to {state_file}: {exc}") from exc

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
        with open(state_file, "rb") as f:
            job_state = pickle.load(f)

        project_name = job_state["project_name"]

        # Combine chunks on HPC
        self._combine_chunks_remotely(project_name)

        # Download combined results
        observables = self._download_combined_results(project_name)

        # Clean up state file
        Path(state_file).unlink(missing_ok=True)

        return observables

    def _combine_chunks_remotely(self, project_name: str) -> None:
        """Combine chunk CSV files on HPC."""
        # Validate project name for security
        project_name = validate_project_name(project_name)

        remote_output = (
            f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/output"
        )

        # Check that chunk files exist - using safe command construction
        check_cmd = build_safe_ssh_command(
            ["sh", "-c", "ls chunk_*_test_stats.csv 2>/dev/null | wc -l"], cwd=remote_output
        )
        status, output = self.transport.exec(check_cmd)
        num_chunks = int(output.strip()) if output.strip().isdigit() else 0

        if num_chunks == 0:
            raise MissingOutputError(
                f"No chunk output files found in {remote_output}. "
                "Jobs may have failed or not produced output files."
            )

        # Combine test stats - using safe command construction
        combine_cmd = build_safe_ssh_command(
            ["sh", "-c", "cat chunk_*_test_stats.csv > combined_test_stats.csv"], cwd=remote_output
        )
        status, output = self.transport.exec(combine_cmd)

        if status != 0:
            raise RemoteCommandError(combine_cmd, status, output)

    def _download_combined_results(self, project_name: str) -> np.ndarray:
        """Download and load combined results."""
        remote_output = (
            f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/output"
        )

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Download combined CSV
            remote_file = f"{remote_output}/combined_test_stats.csv"
            self.transport.download(remote_file, str(temp_dir))

            # Load CSV
            local_file = temp_dir / "combined_test_stats.csv"
            if not local_file.exists():
                raise MissingOutputError(f"Combined results not found locally: {local_file}")

            observables = np.loadtxt(local_file, delimiter=",", ndmin=2)
            # Ensure 2D shape (num_simulations, num_observables)
            if observables.ndim == 1:
                observables = observables.reshape(1, -1)

            return observables  # type: ignore[no-any-return]

        finally:
            # Clean up temp directory
            import shutil

            shutil.rmtree(temp_dir)

    def _check_pool_directory_exists(self, pool_path: str) -> bool:
        """Check if simulation pool directory exists on HPC."""
        return self.result_collector.check_pool_directory_exists(pool_path)

    def _count_pool_simulations(self, pool_path: str) -> int:
        """Count number of simulations in pool from manifest or filenames."""
        return self.result_collector.count_pool_simulations(pool_path)

    def check_hpc_full_simulations(
        self, model_version: str, priors_hash: str, n_requested: int
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
        return self.result_collector.check_hpc_full_simulations(
            model_version, priors_hash, n_requested
        )

    def _combine_chunks_on_hpc(self, test_stats_dir: str) -> None:
        """Combine test statistics chunks on HPC using installed Python script."""
        if self.verbose:
            self.logger.info("Combining chunk files on HPC...")

        # Run combine script from installed qsp-hpc-tools package
        combine_cmd = f'{self.config.hpc_venv_path}/bin/python -m qsp_hpc.batch.combine_test_stats_chunks "{test_stats_dir}"'
        status, output = self.transport.exec(combine_cmd, timeout=60)

        if self.verbose:
            self.logger.info("HPC combine output:")
            for line in output.strip().split("\n"):
                self.logger.info(f"  {line}")

        if status != 0:
            raise RuntimeError(f"Failed to combine chunks on HPC: {output}")

    def _download_combined_files(
        self, test_stats_dir: str, local_dest: Path
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Download and load combined test statistics and parameters from HPC."""
        local_dest.mkdir(parents=True, exist_ok=True)

        # Download test stats
        remote_test_stats_file = f"{test_stats_dir}/combined_test_stats.csv"
        if self.verbose:
            self.logger.info("Downloading combined test stats...")
        self.transport.download(remote_test_stats_file, str(local_dest))

        # Check for combined params
        check_params_cmd = f'test -f "{test_stats_dir}/combined_params.csv" && echo "exists"'
        status_params, output_params = self.transport.exec(check_params_cmd)
        has_params = status_params == 0 and "exists" in output_params

        params = None
        if has_params:
            remote_params_file = f"{test_stats_dir}/combined_params.csv"
            if self.verbose:
                self.logger.info("Downloading combined params...")
            self.transport.download(remote_params_file, str(local_dest))

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
                    self.logger.info(f"Downloaded parameters: {params.shape}")

        # Load test stats using pandas (handles NaN/empty values properly)
        import pandas as pd

        test_stats_df = pd.read_csv(local_test_stats_file, header=None)
        test_stats = test_stats_df.values

        # Ensure 2D shape
        if test_stats.ndim == 1:
            test_stats = test_stats.reshape(1, -1)

        if self.verbose:
            self.logger.info(f"Downloaded test statistics: {test_stats.shape}")

        return params, test_stats

    def check_hpc_test_stats(
        self, pool_path: str, test_stats_hash: str, expected_n_sims: Optional[int] = None
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
        check_cmd = f"""
            test -d "{test_stats_dir}" || exit 1
            echo "TEST_STATS_CHUNKS:$(ls "{test_stats_dir}"/chunk_*_test_stats.csv 2>/dev/null | wc -l)"
            echo "PARAMS_CHUNKS:$(ls "{test_stats_dir}"/chunk_*_params.csv 2>/dev/null | wc -l)"
        """
        status, output = self.transport.exec(check_cmd)

        if status != 0:
            return False

        # Parse chunk counts
        try:
            n_test_stats_chunks = 0
            n_params_chunks = 0
            for line in output.strip().split("\n"):
                if "TEST_STATS_CHUNKS:" in line:
                    n_test_stats_chunks = int(line.split(":")[1])
                elif "PARAMS_CHUNKS:" in line:
                    n_params_chunks = int(line.split(":")[1])

            # Both must have at least one chunk
            if n_test_stats_chunks == 0:
                self.logger.info("   No test stats chunks found")
                return False

            # Params chunks may not exist for older datasets (backward compatibility)
            if n_params_chunks == 0:
                self.logger.info(
                    "  Warning:  No params chunks found (older format without parameters)"
                )
            else:
                self.logger.info(
                    f"   Found {n_test_stats_chunks} test stats chunks and {n_params_chunks} params chunks"
                )

        except Exception as e:
            self.logger.info(f"   Error parsing chunk counts: {e}")
            return False

        # If expected count provided, validate that derived test stats match pool size
        if expected_n_sims is not None:
            # Count rows in combined test stats (or combine chunks first if needed)
            count_cmd = f"""
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
            """
            status, output = self.transport.exec(count_cmd)

            if status == 0:
                try:
                    n_derived = int(output.strip())
                    if n_derived < expected_n_sims:
                        # Not enough derived - need to derive more
                        self.logger.info(
                            f"  Warning:  Not enough derived test stats: {n_derived} < {expected_n_sims} needed"
                        )
                        self.logger.info("   Will derive additional batches")
                        # Delete old derived test stats so they get re-derived
                        self.transport.exec(f'rm -rf "{test_stats_dir}"')
                        return False
                    elif n_derived > expected_n_sims:
                        # Have more than needed - that's fine!
                        self.logger.info(
                            f"   Found {n_derived} derived sims (need {expected_n_sims}) - using existing"
                        )
                    else:
                        # Exact match
                        self.logger.info(
                            f"   Derived test stats count matches: {n_derived} simulations"
                        )
                except ValueError:
                    pass

        return True

    def _calculate_batches_needed(
        self, pool_path: str, num_simulations: Optional[int] = None
    ) -> int:
        """
        Calculate how many Parquet batches need to be processed to get num_simulations.

        If num_simulations is None, returns all batches (old behavior).

        Args:
            pool_path: Path to simulation pool on HPC
            num_simulations: Number of simulations needed (None = all batches)

        Returns:
            Number of batches to process
        """
        # Count total batches
        status, output = self.transport.exec(
            f'ls "{pool_path}"/batch_*.parquet 2>/dev/null | wc -l'
        )
        total_batches = int(output.strip()) if output.strip().isdigit() else 0

        if total_batches == 0:
            self.logger.warning(f"No Parquet batches found in {pool_path}")
            return 0

        # If num_simulations not specified, derive all batches
        if num_simulations is None:
            self.logger.debug(
                f"No simulation count specified - will derive all {total_batches} batches"
            )
            return total_batches

        # Count total simulations across all batches
        total_sims = self.result_collector.count_pool_simulations(pool_path)

        if total_sims == 0:
            self.logger.warning("Could not count simulations in pool - deriving all batches")
            return total_batches

        # Calculate average simulations per batch
        avg_sims_per_batch = total_sims / total_batches

        # Calculate batches needed (round up)
        import math

        batches_needed = math.ceil(num_simulations / avg_sims_per_batch)

        # Cap at total batches available
        batches_needed = min(batches_needed, total_batches)

        self.logger.info(
            f"   Pool has {total_sims} sims in {total_batches} batches "
            f"(~{avg_sims_per_batch:.1f} sims/batch)"
        )
        self.logger.info(
            f"   Need {num_simulations} sims → will derive first {batches_needed} batches"
        )

        return batches_needed

    def submit_derivation_job(
        self,
        pool_path: str,
        test_stats_csv: str,
        test_stats_hash: str,
        project_name: str = "pdac_2025",
        num_simulations: Optional[int] = None,
    ) -> str:
        """
        Submit SLURM job to derive test statistics from full simulations.

        Only derives the minimum number of batches needed to satisfy num_simulations,
        rather than processing the entire pool.

        Args:
            pool_path: Path to simulation pool on HPC (e.g., {simulation_pool_path}/baseline_pdac_abc12345)
            test_stats_csv: Local path to test statistics CSV
            test_stats_hash: Hash of test statistics CSV
            project_name: Project name for logging
            num_simulations: Number of simulations needed (None = derive all batches)

        Returns:
            SLURM job ID
        """
        self.logger.info("Preparing derivation job:")
        self.logger.info(f"  Pool: {pool_path}")
        self.logger.info(f"  Test stats hash: {test_stats_hash[:8]}...")

        # Ensure Python venv is set up
        self.ensure_hpc_venv()

        # Create persistent directory for derivation inputs (in batch_jobs)
        derivation_dir = (
            f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/derivation"
        )
        self.transport.exec(f'mkdir -p "{derivation_dir}"')

        # Upload test statistics CSV
        # The CSV now contains Python function code in the python_function column,
        # eliminating the need for separate test_stat_functions.py files
        remote_test_stats_csv = f"{derivation_dir}/test_stats_{test_stats_hash[:8]}.csv"
        self.logger.info("Uploading test statistics CSV to HPC...")
        self.transport.upload(test_stats_csv, remote_test_stats_csv)

        # Expand $HOME in pool_path (Python won't expand shell variables)
        # Get the actual home directory from HPC
        status, home_dir = self.transport.exec("echo $HOME")
        home_dir = home_dir.strip()
        expanded_pool_path = pool_path.replace("$HOME", home_dir)

        # Create derivation config JSON
        config = {
            "simulation_pool_dir": expanded_pool_path,
            "test_stats_csv": remote_test_stats_csv,
            "output_dir": expanded_pool_path,
            "test_stats_hash": test_stats_hash,
        }

        # Write config locally then upload
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(config, f, indent=2)
            temp_config = f.name

        remote_config = f"{derivation_dir}/derive_config_{test_stats_hash[:8]}.json"
        self.transport.upload(temp_config, remote_config)
        Path(temp_config).unlink()

        # Calculate how many batches to derive (selective derivation)
        n_batches = self._calculate_batches_needed(pool_path, num_simulations)

        if n_batches == 0:
            raise ValueError(f"No Parquet batches found in {pool_path}")

        # Submit derivation job via slurm_submitter (eliminates code duplication)
        job_id = self.slurm_submitter.submit_derivation_job(
            pool_path=pool_path,
            test_stats_config=remote_config,
            derivation_dir=derivation_dir,
            n_batches=n_batches,
            project_name=project_name,
        )

        self.logger.info(f"   🚀 Derivation job {job_id} ({n_batches} tasks)")
        return job_id

    def download_test_stats(
        self, pool_path: str, test_stats_hash: str, local_dest: Path
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
        if "$HOME" in pool_path:
            status, home_dir = self.transport.exec("echo $HOME")
            pool_path = pool_path.replace("$HOME", home_dir.strip())
            if self.verbose:
                self.logger.info(f"Expanded pool path: {pool_path}")

        test_stats_dir = f"{pool_path}/test_stats/{test_stats_hash}"

        if self.verbose:
            self.logger.info(f"Test stats directory: {test_stats_dir}")

        # Check directory exists
        check_cmd = (
            f'test -d "{test_stats_dir}" && ls -la "{test_stats_dir}" || echo "DIRECTORY_NOT_FOUND"'
        )
        status, output = self.transport.exec(check_cmd)

        if self.verbose:
            self.logger.info("Directory listing:")
            for line in output.strip().split("\n")[:10]:  # Show first 10 lines
                self.logger.info(f"  {line}")

        if "DIRECTORY_NOT_FOUND" in output:
            # Determine likely project name for better error message
            project_name = "pdac_2025"  # Could be passed as parameter if needed
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
        self, pool_path: str, local_dest: Path, n_files: int = 1
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
        if "$HOME" in pool_path:
            status, home_dir = self.transport.exec("echo $HOME")
            pool_path = pool_path.replace("$HOME", home_dir.strip())

        self.logger.info(f"   Downloading {n_files} most recent Parquet batch(es) from HPC...")
        self.logger.info(f"   Pool path: {pool_path}")

        # List Parquet files sorted by modification time (most recent first)
        list_cmd = f'ls -t "{pool_path}"/batch_*.parquet 2>/dev/null | head -{n_files}'
        status, output = self.transport.exec(list_cmd)

        if status != 0 or not output.strip():
            raise RuntimeError(f"No Parquet files found in {pool_path}")

        parquet_files = output.strip().split("\n")
        self.logger.info(f"   Found {len(parquet_files)} recent file(s)")

        # Create local destination
        local_dest.mkdir(parents=True, exist_ok=True)

        # Download each file
        downloaded_files = []
        for remote_file in parquet_files:
            remote_file = remote_file.strip()
            if not remote_file:
                continue

            filename = Path(remote_file).name
            self.logger.info(f"   Downloading {filename}...")

            self.transport.download(remote_file, str(local_dest))
            local_file = local_dest / filename
            downloaded_files.append(local_file)

        self.logger.info(f"   Downloaded {len(downloaded_files)} Parquet file(s)")

        return downloaded_files

    def check_job_status(self, job_id: str) -> Dict[str, int]:
        """
        Check status of SLURM job array.

        Args:
            job_id: SLURM job ID

        Returns:
            Dictionary with counts: {'completed': N, 'running': N, 'pending': N, 'failed': N}
        """
        status = {"completed": 0, "running": 0, "pending": 0, "failed": 0}

        # Check squeue for active jobs (running/pending)
        squeue_cmd = f'squeue -j {job_id} --array --format="%i %T" --noheader 2>/dev/null'
        returncode, output = self.transport.exec(squeue_cmd)

        if returncode == 0 and output.strip():
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    state_upper = parts[1].upper()
                    if "RUNNING" in state_upper:
                        status["running"] += 1
                    elif "PENDING" in state_upper:
                        status["pending"] += 1

        # Check sacct for completed/failed jobs
        sacct_cmd = f"sacct -j {job_id} --format=JobID,State --noheader --parsable2"
        returncode, output = self.transport.exec(sacct_cmd)

        if returncode == 0 and output.strip():
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            for line in lines:
                parts = line.split("|")
                if len(parts) >= 2:
                    job_part = parts[0]
                    state = parts[1]

                    # Only count main array tasks (format: 12345_0, 12345_1, ...)
                    # Skip: main job (12345), sub-steps (12345_0.batch, 12345_0.extern)
                    if "_" in job_part and "." not in job_part:
                        state_upper = state.upper()
                        if "COMPLETED" in state_upper:
                            status["completed"] += 1
                        elif (
                            "FAILED" in state_upper
                            or "CANCELLED" in state_upper
                            or "TIMEOUT" in state_upper
                        ):
                            status["failed"] += 1

        return status

    def parse_parquet_simulations(
        self,
        parquet_file: Path,
        species_of_interest: Optional[List[str]] = None,
        max_simulations: Optional[int] = None,
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
        self.logger.info(f"   Parsing Parquet file: {parquet_file.name}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas required for Parquet parsing. Install with: pip install pandas pyarrow"
            )

        # Read Parquet file
        df = pd.read_parquet(parquet_file)

        self.logger.info(f"   Loaded {len(df)} simulations")
        self.logger.info(f"   Columns: {len(df.columns)} ({df.columns[0]}, {df.columns[1]}, ...)")

        # Extract metadata columns
        simulation_ids = df["simulation_id"].values
        statuses = df["status"].values

        # Filter to successful simulations only
        success_mask = statuses == 1
        n_successful: int = int(np.sum(success_mask))

        if n_successful == 0:
            raise ValueError(f"No successful simulations found in {parquet_file}")

        self.logger.info(f"   {n_successful}/{len(df)} simulations successful")

        # Apply max_simulations limit
        if max_simulations is not None and n_successful > max_simulations:
            self.logger.info(f"   Limiting to first {max_simulations} successful simulations")
            # Get indices of successful simulations
            success_indices = np.where(success_mask)[0]
            selected_indices = success_indices[:max_simulations]
            success_mask = np.zeros(len(df), dtype=bool)
            success_mask[selected_indices] = True
            n_successful = max_simulations

        # Extract time vector (from first successful simulation)
        first_success_idx = np.where(success_mask)[0][0]
        time = np.array(df.iloc[first_success_idx]["time"])

        self.logger.info(f"   Time points: {len(time)} ({time[0]:.1f} to {time[-1]:.1f})")

        # Get species columns (exclude metadata: simulation_id, status, time)
        metadata_cols = {"simulation_id", "status", "time"}
        species_names = [col for col in df.columns if col not in metadata_cols]

        self.logger.info(f"   Species: {len(species_names)} total")

        # Filter species if requested
        if species_of_interest is not None:
            # Map species names (replace dots with underscores)
            species_map = {name.replace(".", "_"): name for name in species_names}

            selected_species = []
            for requested_species in species_of_interest:
                # Try exact match first
                if requested_species in species_names:
                    selected_species.append(requested_species)
                # Try with underscore mapping
                elif requested_species in species_map:
                    selected_species.append(species_map[requested_species])
                else:
                    self.logger.info(
                        f"  Warning:  Warning: Species '{requested_species}' not found"
                    )

            if not selected_species:
                raise ValueError("None of the requested species found in Parquet file")

            species_names = selected_species
            self.logger.info(f"   Selected {len(species_names)} species")

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

        self.logger.info(f"   Extracted {len(species_names)} species x {n_successful} simulations")

        return {
            "n_simulations": n_successful,
            "time": time,
            "simulations": simulations,
            "species_names": species_names,
            "simulation_ids": simulation_ids[success_mask],
            "statuses": statuses[success_mask],
        }


# Convenience function
def create_hpc_manager() -> HPCJobManager:
    """
    Create HPC job manager using global configuration.

    Configuration is loaded from ~/.config/qsp-hpc/credentials.yaml
    Run 'qsp-hpc setup' to configure.

    Returns:
        HPCJobManager instance
    """
    return HPCJobManager()
