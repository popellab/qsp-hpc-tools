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
set -e
set -o pipefail

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
        array_spec: str | None = None,
        config_path: str | None = None,
        dependency: str | None = None,
        script_name: str = "qsp_cpp_batch_job.sh",
    ) -> str:
        """Submit a C++ simulation array job.

        Generates a SLURM script that activates the Python venv and runs
        :mod:`qsp_hpc.batch.cpp_batch_worker` — no MATLAB module is loaded.

        Args:
            n_jobs: Number of array tasks. Used for the default
                ``--array=0-{n_jobs-1}`` spec and for logging. When
                ``array_spec`` is provided, ``n_jobs`` is purely cosmetic.
            cpus_per_task: CPUs per task (controls CppBatchRunner parallelism).
            memory: Memory per task (e.g. ``"4G"``).
            array_spec: Override the SLURM array spec string. Accepts the
                sparse syntax (``"7,22,41-45"``) for retrying a specific
                set of task indices without resubmitting the full range
                (#29). ``None`` → default range ``0-{n_jobs-1}``.
            config_path: Override the cpp_job_config.json path that the
                worker reads. Used by retry submissions to pass a
                staging-dir-overriding config without clobbering the
                original. ``None`` → default
                ``batch_jobs/input/cpp_job_config.json``.
            dependency: Optional SLURM ``--dependency=...`` expression.
                Used by retry submissions to wait on the original array.
            script_name: Name of the sbatch script file uploaded to
                ``batch_jobs/scripts/``. Retry submissions pass a distinct
                name so the original script isn't overwritten.

        Returns:
            Job ID string.
        """
        self.logger.info("C++ SLURM Configuration:")
        self.logger.info(f"  Partition: {self.config.partition}")
        self.logger.info(f"  Time limit: {self.config.time_limit}")
        self.logger.info(f"  Memory: {memory}")
        self.logger.info(f"  CPUs/task: {cpus_per_task}")
        if array_spec is not None:
            self.logger.info(f"  Array spec: {array_spec}")
        else:
            self.logger.info(f"  Array size: {n_jobs} tasks")
        if dependency:
            self.logger.info(f"  Dependency: {dependency}")

        script_content = self._generate_cpp_slurm_script(
            n_jobs,
            cpus_per_task,
            memory,
            array_spec=array_spec,
            config_path=config_path,
            dependency=dependency,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            remote_script_dir = f"{self.config.remote_project_path}/batch_jobs/scripts"
            remote_script = f"{remote_script_dir}/{script_name}"

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

    def _generate_cpp_slurm_script(
        self,
        n_jobs: int,
        cpus_per_task: int,
        memory: str,
        array_spec: str | None = None,
        config_path: str | None = None,
        dependency: str | None = None,
    ) -> str:
        """Generate SLURM script for C++ simulation array job."""
        project_path = self.config.remote_project_path
        log_dir = f"{project_path}/batch_jobs/logs"
        array_directive = array_spec if array_spec is not None else f"0-{n_jobs - 1}"
        worker_config = config_path or "batch_jobs/input/cpp_job_config.json"
        dependency_directive = f"#SBATCH --dependency={dependency}\n" if dependency else ""

        # Runtime modules: qsp_sim is dynamically linked against GCC 13's
        # libstdc++ (GLIBCXX_3.4.32) and Boost 1.83.  These need to be on the
        # shared-library search path when SLURM workers invoke the binary.
        modules = self.config.cpp_runtime_modules or ""
        module_load = (
            f"module purge && module load {modules}"
            if modules
            else "# no runtime modules configured"
        )
        return f"""#!/bin/bash
#SBATCH --job-name=qsp_cpp_batch
#SBATCH --partition={self.config.partition}
#SBATCH --time={self.config.time_limit}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --array={array_directive}
#SBATCH --output={log_dir}/qsp_cpp_%A_%a.out
#SBATCH --error={log_dir}/qsp_cpp_%A_%a.err
{dependency_directive}set -e
set -o pipefail

echo "Starting C++ QSP batch job at $(date)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

{module_load}

source "{self.config.hpc_venv_path}/bin/activate"

cd "{project_path}"
python3 -m qsp_hpc.batch.cpp_batch_worker {worker_config}
echo "Job completed at $(date)"
"""

    def submit_derivation_job(
        self,
        pool_path: str,
        test_stats_config: str,
        derivation_dir: str,
        dependency: str | None = None,
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
            dependency: Optional SLURM dependency expression (e.g.
                ``"afterok:12345"``) — when set, derivation waits for the
                upstream job to complete before starting. Used by
                :meth:`HPCJobManager.submit_cpp_jobs` to chain derivation
                after a C++ array.

        Returns:
            Job ID string
        """
        # Log configuration
        self.logger.info("SLURM Derivation Configuration:")
        self.logger.info(f"  Partition: {self.config.partition}")
        self.logger.info("  Time limit: 00:15:00 (fixed for derivation)")
        self.logger.info("  Memory: 4G (fixed for derivation)")
        self.logger.info("  Single task (processes all batches)")
        if dependency:
            self.logger.info(f"  Dependency: {dependency}")

        # Generate SLURM script
        script_content = self._generate_derivation_slurm_script(
            pool_path, test_stats_config, derivation_dir, dependency=dependency
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
        dependency: str | None = None,
    ) -> str:
        """Generate SLURM script for test statistics derivation."""
        log_dir = f"{self.config.remote_project_path}/batch_jobs/logs"
        dependency_directive = f"#SBATCH --dependency={dependency}\n" if dependency else ""
        return f"""#!/bin/bash
#SBATCH --job-name=qsp_derive
#SBATCH --partition={self.config.partition}
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --output={log_dir}/qsp_derive_%j.out
#SBATCH --error={log_dir}/qsp_derive_%j.err
{dependency_directive}set -e
set -o pipefail

# Activate Python virtual environment
source {self.config.hpc_venv_path}/bin/activate

# Run derivation worker with config JSON (processes all batches)
cd {self.config.remote_project_path}
python3 -m qsp_hpc.batch.derive_test_stats_worker "{test_stats_config}"
"""

    def submit_combine_batch_job(
        self,
        combine_config: str,
        combine_dir: str,
        dependency: str | None = None,
    ) -> str:
        """Submit SLURM job to combine array-task chunk parquets into one batch.

        Runs :mod:`qsp_hpc.batch.cpp_combine_batch_worker` as a single task
        chained ``afterok`` to the C++ array. Concatenates
        ``{pool}/.staging/{ARRAY_JOB_ID}/chunk_*.parquet`` into one
        ``batch_*.parquet`` in the pool dir (matching MATLAB's layout)
        and removes the staging dir.

        Args:
            combine_config: Path to combine config JSON on HPC.
            combine_dir: Directory for the temp sbatch script on HPC.
            dependency: SLURM dependency expression (``afterok:<array>``).
                Required so combine only runs after the array succeeds.

        Returns:
            Job ID string.
        """
        self.logger.info("SLURM Combine-Batch Configuration:")
        self.logger.info(f"  Partition: {self.config.partition}")
        self.logger.info("  Time limit: 00:10:00 (fixed for combine)")
        self.logger.info("  Memory: 4G (fixed for combine)")
        self.logger.info("  Single task (consolidates staging → batch parquet)")
        if dependency:
            self.logger.info(f"  Dependency: {dependency}")

        script_content = self._generate_combine_slurm_script(combine_config, dependency=dependency)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            remote_script = f"{combine_dir}/combine_script.sh"
            self.transport.upload(temp_script, remote_script)

            submit_cmd = f'cd "{self.config.remote_project_path}" && sbatch "{remote_script}"'
            returncode, output = self.transport.exec(submit_cmd, timeout=30)

            if returncode != 0:
                raise SubmissionError(f"Combine job submission failed: {output}")

            match = re.search(r"Submitted batch job (\d+)", output)
            if not match:
                raise SubmissionError(f"Could not parse job ID from: {output}")

            return match.group(1)
        finally:
            Path(temp_script).unlink()

    def _generate_combine_slurm_script(
        self,
        combine_config: str,
        dependency: str | None = None,
    ) -> str:
        """Generate SLURM script for the cpp combine-batch worker."""
        log_dir = f"{self.config.remote_project_path}/batch_jobs/logs"
        dependency_directive = f"#SBATCH --dependency={dependency}\n" if dependency else ""
        return f"""#!/bin/bash
#SBATCH --job-name=qsp_combine
#SBATCH --partition={self.config.partition}
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --output={log_dir}/qsp_combine_%j.out
#SBATCH --error={log_dir}/qsp_combine_%j.err
{dependency_directive}set -e
set -o pipefail

source {self.config.hpc_venv_path}/bin/activate

cd {self.config.remote_project_path}
python3 -m qsp_hpc.batch.cpp_combine_batch_worker "{combine_config}"
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
set -e
set -o pipefail

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
