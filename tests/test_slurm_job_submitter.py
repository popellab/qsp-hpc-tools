#!/usr/bin/env python3
"""
Tests for SLURM Job Submitter.

Tests SLURM script generation, job submission, and error handling.
"""

from unittest.mock import Mock

import pytest

from qsp_hpc.batch.hpc_job_manager import BatchConfig
from qsp_hpc.batch.slurm_job_submitter import SLURMJobSubmitter, SubmissionError


@pytest.fixture
def mock_config():
    """Create mock BatchConfig."""
    return BatchConfig(
        ssh_host="hpc.example.edu",
        ssh_user="testuser",
        ssh_key="~/.ssh/id_rsa",
        remote_project_path="/home/testuser/qsp-projects",
        hpc_venv_path="/home/testuser/.venv/hpc-qsp",
        simulation_pool_path="/scratch/testuser/simulations",
        partition="normal",
        time_limit="04:00:00",
        memory_per_job="4G",
        matlab_module="matlab/R2024a",
    )


@pytest.fixture
def mock_transport():
    """Create mock SSH transport."""
    transport = Mock()
    transport.exec = Mock()
    transport.upload = Mock()
    return transport


@pytest.fixture
def submitter(mock_config, mock_transport):
    """Create SLURMJobSubmitter instance."""
    return SLURMJobSubmitter(config=mock_config, transport=mock_transport, verbose=False)


class TestJobSubmission:
    """Tests for submit_job method."""

    def test_submit_job_success(self, submitter, mock_transport):
        """Test successful job submission."""
        # Mock successful submission
        mock_transport.exec.return_value = (0, "Submitted batch job 12345\n")

        job_id = submitter.submit_job(n_jobs=10)

        assert job_id == "12345"

        # Verify upload was called
        assert mock_transport.upload.called

        # Verify sbatch was executed
        assert mock_transport.exec.called
        call_args = mock_transport.exec.call_args[0][0]
        assert "sbatch" in call_args

    def test_submit_job_submission_failure(self, submitter, mock_transport):
        """Test handling of SLURM submission failure."""
        # Mock failed submission
        mock_transport.exec.return_value = (1, "sbatch: error: Unable to allocate resources\n")

        with pytest.raises(SubmissionError, match="SLURM submission failed"):
            submitter.submit_job(n_jobs=10)

    def test_submit_job_cannot_parse_job_id(self, submitter, mock_transport):
        """Test handling when job ID cannot be parsed."""
        # Mock submission with unexpected output
        mock_transport.exec.return_value = (0, "Some unexpected output\n")

        with pytest.raises(SubmissionError, match="Could not parse job ID"):
            submitter.submit_job(n_jobs=10)

    def test_submit_job_generates_correct_script(self, submitter, mock_transport):
        """Test that generated script contains correct SLURM directives."""
        mock_transport.exec.return_value = (0, "Submitted batch job 12345\n")

        # Capture the uploaded script content
        uploaded_files = []

        def capture_upload(local_path, remote_path):
            with open(local_path, "r") as f:
                uploaded_files.append((remote_path, f.read()))

        mock_transport.upload.side_effect = capture_upload

        submitter.submit_job(n_jobs=10)

        # Check that script was uploaded
        assert len(uploaded_files) == 1
        remote_path, script_content = uploaded_files[0]

        # Verify script content
        assert "#SBATCH --partition=normal" in script_content
        assert "#SBATCH --time=04:00:00" in script_content
        assert "#SBATCH --mem=4G" in script_content
        assert "#SBATCH --array=0-9" in script_content  # 0-indexed, 10 tasks
        assert "module load matlab/R2024a" in script_content
        assert "batch_worker()" in script_content

    def test_submit_job_verbose_logging(self, mock_config, mock_transport):
        """Test that verbose mode enables debug logging."""
        submitter_verbose = SLURMJobSubmitter(
            config=mock_config, transport=mock_transport, verbose=True
        )

        mock_transport.exec.return_value = (0, "Submitted batch job 12345\n")

        job_id = submitter_verbose.submit_job(n_jobs=5)

        assert job_id == "12345"


class TestDerivationJobSubmission:
    """Tests for submit_derivation_job method."""

    def test_submit_derivation_job_success(self, submitter, mock_transport):
        """Test successful derivation job submission."""
        mock_transport.exec.return_value = (0, "Submitted batch job 67890\n")

        job_id = submitter.submit_derivation_job(
            pool_path="/scratch/pool",
            test_stats_config="/scratch/test_stats.csv",
            derivation_dir="/scratch/derivation",
        )

        assert job_id == "67890"

        # Verify upload was called
        assert mock_transport.upload.called

        # Verify sbatch was executed
        assert mock_transport.exec.called

    def test_submit_derivation_job_failure(self, submitter, mock_transport):
        """Test handling of derivation job submission failure."""
        mock_transport.exec.return_value = (1, "sbatch: error: Invalid partition\n")

        with pytest.raises(SubmissionError, match="Derivation job submission failed"):
            submitter.submit_derivation_job(
                pool_path="/scratch/pool",
                test_stats_config="/scratch/test_stats.csv",
                derivation_dir="/scratch/derivation",
            )

    def test_submit_derivation_job_cannot_parse_id(self, submitter, mock_transport):
        """Test handling when derivation job ID cannot be parsed."""
        mock_transport.exec.return_value = (0, "Unexpected output\n")

        with pytest.raises(SubmissionError, match="Could not parse job ID"):
            submitter.submit_derivation_job(
                pool_path="/scratch/pool",
                test_stats_config="/scratch/test_stats.csv",
                derivation_dir="/scratch/derivation",
            )

    def test_submit_derivation_job_generates_single_task_script(self, submitter, mock_transport):
        """Test that derivation submits single task, not array job.

        Regression test: derivation should use a single SLURM task that
        processes all batches, not one array task per batch file.
        """
        mock_transport.exec.return_value = (0, "Submitted batch job 67890\n")

        uploaded_files = []

        def capture_upload(local_path, remote_path):
            with open(local_path, "r") as f:
                uploaded_files.append((remote_path, f.read()))

        mock_transport.upload.side_effect = capture_upload

        submitter.submit_derivation_job(
            pool_path="/scratch/pool",
            test_stats_config="/scratch/test_stats.csv",
            derivation_dir="/scratch/derivation",
        )

        # Verify script content
        assert len(uploaded_files) == 1
        remote_path, script_content = uploaded_files[0]

        assert "#SBATCH --job-name=qsp_derive" in script_content
        assert "#SBATCH --partition=normal" in script_content
        assert "#SBATCH --time=00:15:00" in script_content  # Fixed time for derivation
        assert "#SBATCH --mem=4G" in script_content  # Fixed memory (not --mem-per-cpu)
        # Should NOT have array directive - single task processes all batches
        assert "#SBATCH --array" not in script_content
        assert "source /home/testuser/.venv/hpc-qsp/bin/activate" in script_content
        assert "qsp_hpc.batch.derive_test_stats_worker" in script_content
        assert '"/scratch/test_stats.csv"' in script_content  # Config JSON path


class TestScriptGeneration:
    """Tests for internal script generation methods."""

    def test_generate_slurm_script(self, submitter):
        """Test SLURM script generation."""
        script = submitter._generate_slurm_script(n_jobs=20)

        # Check SLURM directives
        assert "#SBATCH --job-name=qsp_batch" in script
        assert "#SBATCH --partition=normal" in script
        assert "#SBATCH --time=04:00:00" in script
        assert "#SBATCH --mem=4G" in script
        assert "#SBATCH --array=0-19" in script  # 0-indexed, 20 tasks

        # Check log file paths
        assert "/home/testuser/qsp-projects/batch_jobs/logs" in script

        # Check MATLAB execution
        assert "module load matlab/R2024a" in script
        assert "batch_worker()" in script

        # Check environment exports
        assert 'export HPC_VENV_PATH="/home/testuser/.venv/hpc-qsp"' in script
        assert 'export SIMULATION_POOL_PATH="/scratch/testuser/simulations"' in script

    def test_generate_derivation_slurm_script_single_task(self, submitter):
        """Test derivation SLURM script generates single task, not array.

        Regression test: derivation was incorrectly using array jobs with
        one task per batch file. Now uses single task that processes all batches.
        """
        script = submitter._generate_derivation_slurm_script(
            pool_path="/scratch/test_pool",
            test_stats_config="/scratch/test_stats.csv",
            derivation_dir="/scratch/derive_output",
        )

        # Check SLURM directives
        assert "#SBATCH --job-name=qsp_derive" in script
        assert "#SBATCH --partition=normal" in script
        assert "#SBATCH --time=00:15:00" in script
        assert "#SBATCH --mem=4G" in script  # Not --mem-per-cpu
        # Should NOT have array directive
        assert "#SBATCH --array" not in script

        # Check Python virtual environment activation
        assert "source /home/testuser/.venv/hpc-qsp/bin/activate" in script

        # Check Python command
        assert "python3 -m qsp_hpc.batch.derive_test_stats_worker" in script
        assert '"/scratch/test_stats.csv"' in script  # Config JSON path


class TestDerivationWorkerCompatibility:
    """Test SLURM script matches derivation worker expectations."""

    def test_slurm_script_passes_config_json_not_cli_args(self, submitter):
        """Test SLURM script passes config JSON path, not command-line arguments.

        Regression test for bug where worker expects single config JSON argument
        but SLURM script was passing --pool-path, --config, etc.
        """
        script = submitter._generate_derivation_slurm_script(
            pool_path="/scratch/pool",
            test_stats_config="/scratch/config.json",
            derivation_dir="/scratch/derive",
        )

        # Should pass config as single positional argument
        assert 'derive_test_stats_worker "/scratch/config.json"' in script

        # Should NOT use command-line flags
        assert "--pool-path" not in script
        assert "--config" not in script
        assert "--output-dir" not in script


class TestFailFastOnErrors:
    """Generated sbatch scripts must exit non-zero when inner commands fail.

    Without `set -e`, a failing python/matlab worker leaves the trailing
    `echo "Job completed"` as the script's exit status, so SLURM reports
    COMPLETED and downstream `afterok` dependencies run on garbage data.
    """

    def test_matlab_array_script_sets_errexit(self, submitter):
        script = submitter._generate_slurm_script(n_jobs=5)
        assert "set -e" in script
        assert "set -o pipefail" in script

    def test_cpp_array_script_sets_errexit(self, submitter):
        script = submitter._generate_cpp_slurm_script(n_jobs=5, cpus_per_task=1, memory="4G")
        assert "set -e" in script
        assert "set -o pipefail" in script

    def test_derivation_script_sets_errexit(self, submitter):
        script = submitter._generate_derivation_slurm_script(
            pool_path="/scratch/pool",
            test_stats_config="/scratch/config.json",
            derivation_dir="/scratch/derive",
        )
        assert "set -e" in script
        assert "set -o pipefail" in script

    def test_combine_script_sets_errexit(self, submitter):
        script = submitter._generate_combine_slurm_script(combine_config="/scratch/combine.json")
        assert "set -e" in script
        assert "set -o pipefail" in script


class TestCppArraySpec:
    """Sparse `--array=...` spec + config override (#29 retry infrastructure)."""

    def test_default_array_spec_is_contiguous_range(self, submitter):
        script = submitter._generate_cpp_slurm_script(n_jobs=5, cpus_per_task=1, memory="4G")
        assert "#SBATCH --array=0-4" in script

    def test_custom_array_spec_preserved_verbatim(self, submitter):
        """Sparse retry: pass a comma/range spec and expect it emitted
        as-is into the sbatch directive. The worker's row-slice addressing
        means sparse task ids hit the right params CSV rows without any
        CSV rewriting."""
        script = submitter._generate_cpp_slurm_script(
            n_jobs=50,
            cpus_per_task=1,
            memory="4G",
            array_spec="7,15,22-25,41",
        )
        assert "#SBATCH --array=7,15,22-25,41" in script
        # n_jobs is purely cosmetic when array_spec is provided.
        assert "#SBATCH --array=0-49" not in script

    def test_custom_worker_config_path_used_in_script(self, submitter):
        """Retry submissions upload a distinct config JSON that pins the
        staging_dir to the original array's dir. The sbatch script must
        invoke the worker with that config path."""
        script = submitter._generate_cpp_slurm_script(
            n_jobs=5,
            cpus_per_task=1,
            memory="4G",
            config_path="batch_jobs/input/cpp_retry_config_7654321.json",
        )
        assert "cpp_batch_worker batch_jobs/input/cpp_retry_config_7654321.json" in script
        # Default path must NOT appear when an override is provided.
        assert "cpp_batch_worker batch_jobs/input/cpp_job_config.json" not in script

    def test_dependency_directive_when_set(self, submitter):
        script = submitter._generate_cpp_slurm_script(
            n_jobs=3,
            cpus_per_task=1,
            memory="4G",
            dependency="afterany:7654321",
        )
        assert "#SBATCH --dependency=afterany:7654321" in script

    def test_no_dependency_directive_by_default(self, submitter):
        script = submitter._generate_cpp_slurm_script(n_jobs=3, cpus_per_task=1, memory="4G")
        assert "--dependency=" not in script


class TestSubmissionError:
    """Tests for SubmissionError exception."""

    def test_submission_error_creation(self):
        """Test creating SubmissionError."""
        error = SubmissionError("Test error")
        assert isinstance(error, RuntimeError)
        assert str(error) == "Test error"

    def test_submission_error_raising(self):
        """Test raising SubmissionError."""
        with pytest.raises(SubmissionError) as exc_info:
            raise SubmissionError("Job failed")

        assert "Job failed" in str(exc_info.value)
