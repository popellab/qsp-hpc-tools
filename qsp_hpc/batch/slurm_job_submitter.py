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

    def submit_job(self, n_jobs: int) -> str:
        """
        Generate and submit SLURM array job.

        Args:
            n_jobs: Number of array tasks

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
        script_content = self._generate_slurm_script(n_jobs)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            # Upload script
            remote_script_dir = f"{self.config.remote_project_path}/batch_jobs/scripts"
            remote_script = f"{remote_script_dir}/qsp_batch_job.sh"

            self.transport.upload(temp_script, remote_script)

            # Submit job
            returncode, output = self.transport.exec(f'sbatch "{remote_script}"', timeout=30)

            if returncode != 0:
                raise SubmissionError(f"SLURM submission failed: {output}")

            # Extract job ID
            match = re.search(r"Submitted batch job (\d+)", output)
            if not match:
                raise SubmissionError(f"Could not parse job ID from: {output}")

            job_id = match.group(1)
            elapsed = time.time() - start_time
            self.logger.info(f"🚀 Job {job_id} ({n_jobs} tasks, {elapsed:.1f}s)")

            return job_id

        finally:
            Path(temp_script).unlink()

    def _generate_slurm_script(self, n_jobs: int) -> str:
        """Generate SLURM batch script."""
        project_path = self.config.remote_project_path
        log_dir = f"{project_path}/batch_jobs/logs"

        script = f"""#!/bin/bash
#SBATCH --job-name=qsp_batch
#SBATCH --partition={self.config.partition}
#SBATCH --time={self.config.time_limit}
#SBATCH --mem={self.config.memory_per_job}
#SBATCH --cpus-per-task={self.config.cpus_per_task}
#SBATCH --array=0-{n_jobs-1}
#SBATCH --output={log_dir}/qsp_batch_%A_%a.out
#SBATCH --error={log_dir}/qsp_batch_%A_%a.err

echo "Starting QSP batch job at $(date)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

# Export HPC paths for MATLAB scripts to use
export HPC_VENV_PATH="{self.config.hpc_venv_path}"
export SIMULATION_POOL_PATH="{self.config.simulation_pool_path}"
# MATLAB_WORKERS=0 (or unset) -> serial loop; >0 -> parpool(N) + parfor
export MATLAB_WORKERS="{self.config.matlab_workers}"

# Get MATLAB scripts path from installed qsp-hpc-tools package
MATLAB_PATH=$("{self.config.hpc_venv_path}/bin/python" -c "import qsp_hpc.matlab; print(qsp_hpc.matlab.get_matlab_path())")

module load {self.config.matlab_module}
cd "{project_path}"
matlab -nodisplay -nodesktop -nosplash -r "addpath('$MATLAB_PATH'); batch_worker(); exit"
echo "Job completed at $(date)"
"""
        return script

    def submit_cpp_job(
        self,
        n_jobs: int,
        cpus_per_task: int = 1,
        memory: str = "4G",
    ) -> str:
        """Submit a C++ simulation array job.

        Generates a SLURM script that activates the Python venv and runs
        :mod:`qsp_hpc.batch.cpp_batch_worker` — no MATLAB module is loaded.

        Args:
            n_jobs: Number of array tasks.
            cpus_per_task: CPUs per task (controls CppBatchRunner parallelism).
            memory: Memory per task (e.g. ``"4G"``).

        Returns:
            Job ID string.
        """
        self.logger.info("C++ SLURM Configuration:")
        self.logger.info(f"  Partition: {self.config.partition}")
        self.logger.info(f"  Time limit: {self.config.time_limit}")
        self.logger.info(f"  Memory: {memory}")
        self.logger.info(f"  CPUs/task: {cpus_per_task}")
        self.logger.info(f"  Array size: {n_jobs} tasks")

        script_content = self._generate_cpp_slurm_script(n_jobs, cpus_per_task, memory)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            remote_script_dir = f"{self.config.remote_project_path}/batch_jobs/scripts"
            remote_script = f"{remote_script_dir}/qsp_cpp_batch_job.sh"

            self.transport.upload(temp_script, remote_script)

            returncode, output = self.transport.exec(f'sbatch "{remote_script}"', timeout=30)

            if returncode != 0:
                raise SubmissionError(f"C++ SLURM submission failed: {output}")

            match = re.search(r"Submitted batch job (\d+)", output)
            if not match:
                raise SubmissionError(f"Could not parse job ID from: {output}")

            job_id = match.group(1)
            self.logger.info(f"Job {job_id} ({n_jobs} C++ tasks)")
            return job_id

        finally:
            Path(temp_script).unlink()

    def _generate_cpp_slurm_script(self, n_jobs: int, cpus_per_task: int, memory: str) -> str:
        """Generate SLURM script for C++ simulation array job."""
        project_path = self.config.remote_project_path
        log_dir = f"{project_path}/batch_jobs/logs"

        return f"""#!/bin/bash
#SBATCH --job-name=qsp_cpp_batch
#SBATCH --partition={self.config.partition}
#SBATCH --time={self.config.time_limit}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --array=0-{n_jobs - 1}
#SBATCH --output={log_dir}/qsp_cpp_%A_%a.out
#SBATCH --error={log_dir}/qsp_cpp_%A_%a.err

echo "Starting C++ QSP batch job at $(date)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

source "{self.config.hpc_venv_path}/bin/activate"

cd "{project_path}"
python3 -m qsp_hpc.batch.cpp_batch_worker batch_jobs/input/cpp_job_config.json
echo "Job completed at $(date)"
"""

    def submit_derivation_job(
        self,
        pool_path: str,
        test_stats_config: str,
        derivation_dir: str,
    ) -> str:
        """
        Submit SLURM job to derive test statistics from full simulations.

        Submits a single task that processes all batches sequentially.
        This is more efficient than array jobs for derivation since:
        - Derivation is I/O bound (reading Parquet) and fast
        - Avoids SLURM scheduling overhead for many small tasks
        - Single task processes all batches in one job

        Args:
            pool_path: Path to simulation pool on HPC
            test_stats_config: Path to test statistics config on HPC
            derivation_dir: Directory for derivation outputs

        Returns:
            Job ID string
        """
        # Log configuration
        self.logger.info("SLURM Derivation Configuration:")
        self.logger.info(f"  Partition: {self.config.partition}")
        self.logger.info("  Time limit: 00:15:00 (fixed for derivation)")
        self.logger.info("  Memory: 4G (fixed for derivation)")
        self.logger.info("  Single task (processes all batches)")

        # Generate SLURM script
        script_content = self._generate_derivation_slurm_script(
            pool_path, test_stats_config, derivation_dir
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
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
            match = re.search(r"Submitted batch job (\d+)", output)
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
    ) -> str:
        """Generate SLURM script for test statistics derivation."""
        log_dir = f"{self.config.remote_project_path}/batch_jobs/logs"
        return f"""#!/bin/bash
#SBATCH --job-name=qsp_derive
#SBATCH --partition={self.config.partition}
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --output={log_dir}/qsp_derive_%j.out
#SBATCH --error={log_dir}/qsp_derive_%j.err

# Activate Python virtual environment
source {self.config.hpc_venv_path}/bin/activate

# Run derivation worker with config JSON (processes all batches)
cd {self.config.remote_project_path}
python3 -m qsp_hpc.batch.derive_test_stats_worker "{test_stats_config}"
"""

    def submit_trajectory_grid_job(
        self,
        grid_config: str,
        derivation_dir: str,
    ) -> str:
        """Submit SLURM job to extract trajectory grids from full simulations.

        Args:
            grid_config: Path to trajectory grid config JSON on HPC
            derivation_dir: Directory for job scripts/logs

        Returns:
            Job ID string
        """
        self.logger.info("SLURM Trajectory Grid Configuration:")
        self.logger.info(f"  Partition: {self.config.partition}")
        self.logger.info("  Time limit: 00:15:00 (fixed for grid extraction)")
        self.logger.info("  Memory: 8G")

        log_dir = f"{self.config.remote_project_path}/batch_jobs/logs"
        script_content = f"""#!/bin/bash
#SBATCH --job-name=qsp_traj_grid
#SBATCH --partition={self.config.partition}
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --output={log_dir}/qsp_traj_grid_%j.out
#SBATCH --error={log_dir}/qsp_traj_grid_%j.err

# Activate Python virtual environment
source {self.config.hpc_venv_path}/bin/activate

# Run trajectory grid worker
cd {self.config.remote_project_path}
python3 -m qsp_hpc.batch.derive_trajectory_grid_worker "{grid_config}"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            remote_script = f"{derivation_dir}/traj_grid_script.sh"
            self.transport.upload(temp_script, remote_script)

            submit_cmd = f'cd "{self.config.remote_project_path}" && sbatch "{remote_script}"'
            returncode, output = self.transport.exec(submit_cmd, timeout=30)

            if returncode != 0:
                raise SubmissionError(f"Trajectory grid job submission failed: {output}")

            match = re.search(r"Submitted batch job (\d+)", output)
            if not match:
                raise SubmissionError(f"Could not parse job ID from: {output}")

            return match.group(1)

        finally:
            Path(temp_script).unlink()
