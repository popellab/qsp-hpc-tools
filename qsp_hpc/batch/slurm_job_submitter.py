#!/usr/bin/env python3
"""
SLURM Job Submission and Management

Handles all SLURM-specific operations including script generation,
job submission, and status monitoring.
"""

import re
import tempfile
import time
from pathlib import Path
from typing import Optional
from qsp_hpc.utils.logging_config import setup_logger


class SubmissionError(RuntimeError):
    """Raised when SLURM submission cannot be parsed or accepted."""
    pass


class SLURMJobSubmitter:
    """
    Handles SLURM job submission and script generation.

    Responsibilities:
    - Generate SLURM batch scripts
    - Submit jobs via SSH transport
    - Parse job IDs from submission output
    - Generate derivation job scripts
    """

    def __init__(self, config, transport, verbose: bool = False):
        """
        Initialize SLURM job submitter.

        Args:
            config: BatchConfig with SLURM settings
            transport: SSHTransport instance for remote execution
            verbose: Enable verbose logging
        """
        self.config = config
        self.transport = transport
        self.verbose = verbose
        self.logger = setup_logger(__name__, verbose=verbose)

    def submit_job(self, n_jobs: int, project_name: str) -> str:
        """
        Generate and submit SLURM array job.

        Args:
            n_jobs: Number of array tasks
            project_name: Project identifier

        Returns:
            Job ID string

        Raises:
            SubmissionError: If job submission fails or job ID cannot be parsed
        """
        start_time = time.time()

        # Log SLURM configuration
        self.logger.info("SLURM Configuration:")
        self.logger.info(f"  Partition: {self.config.partition}")
        self.logger.info(f"  Time limit: {self.config.time_limit}")
        self.logger.info(f"  Memory per job: {self.config.memory_per_job}")
        self.logger.info(f"  Array size: {n_jobs} tasks")

        # Generate SLURM script
        script_content = self._generate_slurm_script(n_jobs, project_name)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            # Upload script to HPC
            remote_script = f"{self.config.remote_project_path}/projects/{project_name}/batch_jobs/job_script.sh"
            self.transport.upload(temp_script, remote_script)

            # Submit job
            submit_cmd = f'cd "{self.config.remote_project_path}" && sbatch "{remote_script}"'
            returncode, output = self.transport.exec(submit_cmd, timeout=30)

            if returncode != 0:
                raise SubmissionError(f"sbatch failed (rc={returncode}): {output}")

            # Extract job ID
            match = re.search(r'Submitted batch job (\d+)', output)
            if not match:
                raise SubmissionError(f"Could not parse job ID from: {output}")

            job_id = match.group(1)
            elapsed = time.time() - start_time
            self.logger.info(f"🚀 Job {job_id} ({n_jobs} tasks, {elapsed:.1f}s)")

            return job_id

        finally:
            Path(temp_script).unlink()

    def _generate_slurm_script(self, n_jobs: int, project_name: str) -> str:
        """Generate SLURM batch script for simulation jobs."""
        return f"""#!/bin/bash
#SBATCH --job-name=qsp_batch
#SBATCH --partition={self.config.partition}
#SBATCH --time={self.config.time_limit}
#SBATCH --mem-per-cpu={self.config.memory_per_job}
#SBATCH --array=1-{n_jobs}
#SBATCH --output={self.config.remote_project_path}/projects/{project_name}/batch_jobs/logs/qsp_%A_%a.out
#SBATCH --error={self.config.remote_project_path}/projects/{project_name}/batch_jobs/logs/qsp_%A_%a.err

# Load MATLAB module
module load {self.config.matlab_module}

# Activate Python virtual environment
source {self.config.hpc_venv_path}/bin/activate

# Run batch worker
cd {self.config.remote_project_path}
matlab -nodisplay -nosplash -batch "addpath('matlab'); batch_worker('$SLURM_ARRAY_TASK_ID', '{project_name}')"
"""

    def submit_derivation_job(
        self,
        pool_path: str,
        test_stats_config: str,
        derivation_dir: str,
        n_batches: int,
        project_name: str
    ) -> str:
        """
        Submit SLURM job to derive test statistics from full simulations.

        Args:
            pool_path: Path to simulation pool on HPC
            test_stats_config: Path to test statistics config on HPC
            derivation_dir: Directory for derivation outputs
            n_batches: Number of array tasks (one per Parquet file)
            project_name: Project identifier

        Returns:
            Job ID string
        """
        # Log configuration
        self.logger.info("SLURM Derivation Configuration:")
        self.logger.info(f"  Partition: {self.config.partition}")
        self.logger.info("  Time limit: 01:00:00 (fixed for derivation)")
        self.logger.info("  Memory per job: 4G (fixed for derivation)")
        self.logger.info(f"  Array size: {n_batches} tasks")

        # Generate SLURM script
        script_content = self._generate_derivation_slurm_script(
            pool_path, test_stats_config, derivation_dir, n_batches, project_name
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            # Upload and submit
            remote_script = f"{derivation_dir}/derive_script.sh"
            self.transport.upload(temp_script, remote_script)

            submit_cmd = f'cd "{self.config.remote_project_path}" && sbatch "{remote_script}"'
            returncode, output = self.transport.exec(submit_cmd, timeout=30)

            if returncode != 0:
                raise SubmissionError(f"Derivation job submission failed: {output}")

            # Extract job ID
            match = re.search(r'Submitted batch job (\d+)', output)
            if not match:
                raise SubmissionError(f"Could not parse job ID from: {output}")

            return match.group(1)

        finally:
            Path(temp_script).unlink()

    def _generate_derivation_slurm_script(
        self,
        pool_path: str,
        test_stats_config: str,
        derivation_dir: str,
        n_batches: int,
        project_name: str
    ) -> str:
        """Generate SLURM script for test statistics derivation."""
        return f"""#!/bin/bash
#SBATCH --job-name=qsp_derive
#SBATCH --partition={self.config.partition}
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-{n_batches}
#SBATCH --output={self.config.remote_project_path}/projects/{project_name}/batch_jobs/logs/qsp_derive_%A_%a.out
#SBATCH --error={self.config.remote_project_path}/projects/{project_name}/batch_jobs/logs/qsp_derive_%A_%a.err

# Activate Python virtual environment
source {self.config.hpc_venv_path}/bin/activate

# Run derivation worker
cd {self.config.remote_project_path}
python3 -m qsp_hpc.batch.derive_test_stats_worker \\
    --pool-path "{pool_path}" \\
    --config "{test_stats_config}" \\
    --output-dir "{derivation_dir}" \\
    --task-id $SLURM_ARRAY_TASK_ID
"""
