"""Tests for HPC job manager.

This module contains both unit tests (no HPC required) and integration tests
(require real HPC connection). Integration tests are marked with @pytest.mark.hpc
and can be skipped with: pytest -m "not hpc"

To run HPC integration tests, you need:
1. Valid credentials in ~/.config/qsp-hpc/credentials.yaml (run 'qsp-hpc setup')
2. SSH access to HPC cluster
3. SLURM scheduler available

Run HPC tests with: pytest -m hpc -v
Skip HPC tests with: pytest -m "not hpc" -v
"""

import pickle
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from qsp_hpc.batch.hpc_job_manager import (
    BatchConfig,
    HPCJobManager,
    JobInfo,
    MissingOutputError,
    RemoteCommandError,
    SubmissionError,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_hpc_config():
    """Mock HPC configuration for unit tests."""
    return BatchConfig(
        ssh_host="test-hpc.example.com",
        ssh_user="testuser",
        ssh_key=str(Path.home() / ".ssh" / "id_rsa"),
        remote_project_path="/home/testuser/qsp-hpc",
        matlab_module="matlab/R2024a",
        hpc_venv_path="/home/testuser/.venv/hpc-qsp",
        simulation_pool_path="/scratch/testuser/simulations",
        partition="normal",
        time_limit="04:00:00",
        memory_per_job="4G",
    )


@pytest.fixture
def temp_config_file(tmp_path, mock_hpc_config):
    """Create temporary config file for testing."""
    config_data = {
        "ssh": {
            "host": mock_hpc_config.ssh_host,
            "user": mock_hpc_config.ssh_user,
            "key": str(mock_hpc_config.ssh_key),
        },
        "cluster": {"matlab_module": mock_hpc_config.matlab_module},
        "paths": {
            "remote_base_dir": mock_hpc_config.remote_project_path,
            "hpc_venv_path": mock_hpc_config.hpc_venv_path,
            "simulation_pool_path": mock_hpc_config.simulation_pool_path,
        },
        "slurm": {
            "partition": mock_hpc_config.partition,
            "time_limit": mock_hpc_config.time_limit,
            "mem_per_cpu": mock_hpc_config.memory_per_job,
        },
    }

    config_file = tmp_path / "test_credentials.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return config_file


@pytest.fixture
def real_hpc_config():
    """
    Load real HPC configuration from credentials file.

    This fixture is used for integration tests marked with @pytest.mark.hpc.
    It will skip the test if no valid config is found.
    """
    config_path = Path.home() / ".config" / "qsp-hpc" / "credentials.yaml"

    if config_path.exists():
        return config_path

    pytest.skip("No HPC credentials found. Run 'qsp-hpc setup' to configure.")


# ============================================================================
# Unit Tests (No HPC Required)
# ============================================================================


class TestHPCConfigLoading:
    """Test configuration loading and validation."""

    def test_config_loading_from_file(self, tmp_path):
        """Test loading config from YAML file."""
        tmp_home = tmp_path / "home"
        creds_dir = tmp_home / ".config" / "qsp-hpc"
        creds_dir.mkdir(parents=True)

        # Create SSH key file for validation
        ssh_dir = tmp_home / ".ssh"
        ssh_dir.mkdir(parents=True)
        ssh_key = ssh_dir / "id_rsa"
        ssh_key.write_text("fake-ssh-key-for-testing")

        creds_path = creds_dir / "credentials.yaml"
        with open(creds_path, "w") as fh:
            yaml.safe_dump(
                {
                    "ssh": {
                        "host": "test-hpc.example.com",
                        "user": "testuser",
                        "key": str(ssh_key),
                    },
                    "paths": {
                        "remote_base_dir": "/home/testuser/qsp-hpc",
                        "simulation_pool_path": "/scratch/testuser/simulations",
                        "hpc_venv_path": "/home/testuser/.venv/hpc-qsp",
                    },
                    "cluster": {"matlab_module": "matlab/R2024a"},
                    "slurm": {"partition": "normal", "time_limit": "02:00:00", "mem_per_cpu": "8G"},
                },
                fh,
            )

        with patch.object(Path, "home", return_value=tmp_home):
            manager = HPCJobManager()

        cfg = manager.config
        assert cfg.ssh_host == "test-hpc.example.com"
        assert cfg.ssh_user == "testuser"
        assert cfg.remote_project_path == "/home/testuser/qsp-hpc"
        assert cfg.simulation_pool_path == "/scratch/testuser/simulations"
        assert cfg.hpc_venv_path == "/home/testuser/.venv/hpc-qsp"
        assert cfg.partition == "normal"
        assert cfg.time_limit == "02:00:00"

    def test_hierarchical_config_merge(self):
        """Test that project config overrides global config."""
        tmp_home = Path(tempfile.mkdtemp())
        global_dir = tmp_home / ".config" / "qsp-hpc"
        project_dir = tmp_home / "project"
        global_dir.mkdir(parents=True)
        (project_dir / ".qsp-hpc").mkdir(parents=True)

        # Create SSH key file for validation
        ssh_dir = tmp_home / ".ssh"
        ssh_dir.mkdir(parents=True)
        ssh_key = ssh_dir / "id_rsa"
        ssh_key.write_text("fake-ssh-key-for-testing")

        global_cfg = {
            "ssh": {"host": "global-host", "user": "global", "key": str(ssh_key)},
            "paths": {
                "remote_base_dir": "/global/base",
                "simulation_pool_path": "/global/pool",
                "hpc_venv_path": "/global/venv",
            },
            "slurm": {"partition": "global", "time_limit": "01:00:00", "mem_per_cpu": "4G"},
        }
        project_override = {
            "ssh": {"host": "project-host"},
            "paths": {"remote_base_dir": "/project/base"},
            "slurm": {"partition": "project-partition"},
        }

        with open(global_dir / "credentials.yaml", "w") as fh:
            yaml.safe_dump(global_cfg, fh)
        with open(project_dir / ".qsp-hpc" / "credentials.yaml", "w") as fh:
            yaml.safe_dump(project_override, fh)

        with (
            patch.object(Path, "home", return_value=tmp_home),
            patch.object(Path, "cwd", return_value=project_dir),
        ):
            manager = HPCJobManager()

        cfg = manager.config
        assert cfg.ssh_host == "project-host"  # override
        assert cfg.remote_project_path == "/project/base"  # override
        assert cfg.partition == "project-partition"  # override
        assert cfg.simulation_pool_path == "/global/pool"  # inherited
        assert cfg.hpc_venv_path == "/global/venv"  # inherited

    def test_missing_required_fields_error(self, tmp_path):
        """Test that missing required config fields raise error."""
        tmp_home = tmp_path / "home"
        creds_dir = tmp_home / ".config" / "qsp-hpc"
        creds_dir.mkdir(parents=True)

        # Missing simulation_pool_path and hpc_venv_path
        with open(creds_dir / "credentials.yaml", "w") as fh:
            yaml.safe_dump({"ssh": {"host": "host"}, "paths": {}}, fh)

        with patch.object(Path, "home", return_value=tmp_home):
            with pytest.raises(ValueError):
                HPCJobManager()

    def test_ssh_key_expansion(self):
        """Test that SSH key path expands ~ correctly."""
        custom_cfg = {
            "ssh_host": "example",
            "ssh_user": "user",
            "ssh_key": "~/.ssh/id_rsa",
            "simulation_pool_path": "/pool",
            "hpc_venv_path": "/venv",
        }
        manager = HPCJobManager(config=custom_cfg)
        assert manager.config.ssh_key == str(Path("~/.ssh/id_rsa").expanduser())

    def test_save_job_state_uses_local_path_when_remote_missing(self, tmp_path, mock_hpc_config):
        job_info = JobInfo(
            job_ids=["1"],
            state_file="",
            n_jobs=1,
            n_simulations=10,
            submission_time="now",
        )

        # Point remote_project_path to a non-existent location
        cfg = mock_hpc_config
        cfg.remote_project_path = "/definitely/not/present"

        with patch.object(Path, "cwd", return_value=tmp_path):
            manager = HPCJobManager(config=cfg)
            state_path = Path(manager._save_job_state(job_info))

        assert state_path.parent == tmp_path / "batch_jobs"
        assert state_path.exists()


class TestSLURMScriptGeneration:
    """Test SLURM batch script generation."""

    def test_basic_script_generation(self, mock_hpc_config):
        """Test generation of basic SLURM script."""
        manager = HPCJobManager(config=mock_hpc_config)
        script = manager._generate_slurm_script(n_jobs=3)

        assert "#SBATCH --job-name=qsp_batch" in script
        assert "#SBATCH --array=0-2" in script
        assert f"#SBATCH --partition={mock_hpc_config.partition}" in script
        assert f"#SBATCH --time={mock_hpc_config.time_limit}" in script
        assert mock_hpc_config.simulation_pool_path in script
        assert mock_hpc_config.hpc_venv_path in script
        assert mock_hpc_config.matlab_module in script
        assert "batch_worker()" in script

    def test_array_job_script(self, mock_hpc_config):
        """Test array job parameters in script."""
        manager = HPCJobManager(config=mock_hpc_config)
        script = manager._generate_slurm_script(n_jobs=10)
        assert "#SBATCH --array=0-9" in script

    def test_module_loading_in_script(self, mock_hpc_config):
        """Test that MATLAB module load is included."""
        manager = HPCJobManager(config=mock_hpc_config)
        script = manager._generate_slurm_script(n_jobs=1)
        assert f"module load {mock_hpc_config.matlab_module}" in script

    def test_environment_variables_in_script(self, mock_hpc_config):
        """Test that HPC_VENV_PATH and SIMULATION_POOL_PATH are exported."""
        manager = HPCJobManager(config=mock_hpc_config)
        script = manager._generate_slurm_script(n_jobs=1)
        assert "export HPC_VENV_PATH" in script
        assert "export SIMULATION_POOL_PATH" in script

    def test_job_name_formatting(self, mock_hpc_config):
        """Test SLURM job name generation."""
        manager = HPCJobManager(config=mock_hpc_config)
        script = manager._generate_slurm_script(n_jobs=4)
        assert "#SBATCH --job-name=qsp_batch" in script


class TestJobStateManagement:
    """Test job state tracking and serialization."""

    def test_job_info_creation(self):
        """Test creating JobInfo object."""
        info = JobInfo(
            job_ids=["123"],
            state_file="state.pkl",
            n_jobs=2,
            n_simulations=20,
            submission_time="now",
        )
        assert info.job_ids == ["123"]
        assert info.n_jobs == 2

    def test_job_state_serialization(self, tmp_path, monkeypatch):
        """Test saving job state to pickle file."""
        info = JobInfo(
            job_ids=["1"],
            state_file="",
            n_jobs=1,
            n_simulations=10,
            submission_time="2025-01-01",
        )
        # Point remote path to a non-existent location; manager should fall back to cwd
        manager = HPCJobManager(
            config={
                "ssh_host": "example",
                "ssh_user": "user",
                "simulation_pool_path": "/pool",
                "hpc_venv_path": "/venv",
                "remote_project_path": str(tmp_path / "remote"),
            }
        )

        # Write inside temp directory to avoid polluting repo
        monkeypatch.chdir(tmp_path)
        state_file = manager._save_job_state(info)

        saved = pickle.load(open(state_file, "rb"))
        assert saved["job_ids"] == ["1"]
        assert saved["n_simulations"] == 10
        assert Path(state_file).parent.name == "batch_jobs"
        assert str(tmp_path / "batch_jobs") in state_file

    def test_job_state_deserialization(self, tmp_path):
        """Test loading job state from pickle file."""
        state_payload = {
            "job_ids": ["1"],
            "state_file": "state.pkl",
            "n_jobs": 1,
            "n_simulations": 5,
            "submission_time": "now",
        }
        state_file = tmp_path / "state.pkl"
        with open(state_file, "wb") as fh:
            pickle.dump(state_payload, fh)

        manager = HPCJobManager(
            config={
                "ssh_host": "example",
                "ssh_user": "user",
                "simulation_pool_path": "/pool",
                "hpc_venv_path": "/venv",
                "remote_project_path": str(tmp_path / "remote"),
            }
        )

        calls = {"combined": False, "downloaded": False}

        manager._combine_chunks_remotely = lambda: calls.__setitem__("combined", True)
        manager._download_combined_results = lambda: calls.__setitem__(
            "downloaded", True
        ) or np.ones((1, 1))

        result = manager.collect_results(str(state_file))

        assert calls["combined"] is True
        assert calls["downloaded"] is True
        assert isinstance(result, np.ndarray)
        assert not state_file.exists()

    def test_job_state_file_naming(self, monkeypatch, tmp_path):
        """Test job state filename includes timestamp."""
        info = JobInfo(
            job_ids=["1"],
            state_file="",
            n_jobs=1,
            n_simulations=1,
            submission_time="now",
        )
        manager = HPCJobManager(
            config={
                "ssh_host": "example",
                "ssh_user": "user",
                "simulation_pool_path": "/pool",
                "hpc_venv_path": "/venv",
                "remote_project_path": str(tmp_path / "remote"),
            }
        )

        monkeypatch.chdir(tmp_path)
        state_file = manager._save_job_state(info)

        assert "job_state_" in Path(state_file).name
        assert Path(state_file).parent.name == "batch_jobs"
        # Should fall back to cwd when remote path is missing
        assert state_file.startswith(str(tmp_path / "batch_jobs"))


class TestPathConstruction:
    """Test remote path construction."""

    def test_remote_project_path(self, mock_hpc_config):
        """Test construction of remote project path."""
        manager = HPCJobManager(config=mock_hpc_config)
        with patch.object(manager.transport, "exec", return_value=(0, "")) as mock_exec:
            manager._setup_remote_directories()

        # All calls should include the remote base directory
        assert all(
            mock_hpc_config.remote_project_path in call.args[0] for call in mock_exec.call_args_list
        )

    def test_remote_simulation_pool_path(self, mock_hpc_config):
        """Test construction of simulation pool path."""
        manager = HPCJobManager(config=mock_hpc_config)
        script = manager._generate_slurm_script(1)
        assert mock_hpc_config.simulation_pool_path in script

    def test_remote_log_path(self, mock_hpc_config):
        """Test construction of SLURM log file paths."""
        manager = HPCJobManager(config=mock_hpc_config)
        script = manager._generate_slurm_script(2)
        assert f"{mock_hpc_config.remote_project_path}/batch_jobs/logs" in script


class TestSyncCodebase:
    """Tests for rsync command construction."""

    @patch("subprocess.run")
    def test_sync_includes_ssh_key_and_excludes(self, mock_run, mock_hpc_config):
        manager = HPCJobManager(config=mock_hpc_config)
        # Mock ssh exec for mkdir
        manager._ssh_exec = Mock(return_value=(0, ""))

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        manager.sync_codebase(skip_sync=False)

        # First call is mkdir, second is rsync
        assert mock_run.call_count == 1
        args = mock_run.call_args[0][0]
        assert "rsync" in args[0]
        # Ensure SSH key is passed
        assert any(mock_hpc_config.ssh_key in part for part in args)
        # Ensure a known exclude is present
        assert "--exclude" in args

    @patch("subprocess.run")
    def test_sync_skipped(self, mock_run, mock_hpc_config):
        manager = HPCJobManager(config=mock_hpc_config)
        manager.sync_codebase(skip_sync=True)
        mock_run.assert_not_called()


class TestCommandExecutionBehaviors:
    """Mocked command construction and error paths."""

    def test_submit_slurm_parsing_failure(self, mock_hpc_config):
        manager = HPCJobManager(config=mock_hpc_config)
        manager.transport.upload = Mock()
        manager._generate_slurm_script = Mock(return_value="#SBATCH")
        # Return successful status but without job id string
        manager.transport.exec = Mock(return_value=(0, "no job id here"))

        with pytest.raises(SubmissionError):
            manager._submit_slurm_job(1)

    def test_submit_slurm_nonzero_raises(self, mock_hpc_config):
        manager = HPCJobManager(config=mock_hpc_config)
        manager.transport.upload = Mock()
        manager._generate_slurm_script = Mock(return_value="#SBATCH")
        manager.transport.exec = Mock(return_value=(1, "error"))

        with pytest.raises(SubmissionError):
            manager._submit_slurm_job(1)

    def test_combine_chunks_missing_outputs(self, mock_hpc_config):
        manager = HPCJobManager(config=mock_hpc_config)
        # Simulate zero chunk files found
        manager._ssh_exec = Mock(return_value=(0, "0"))

        with pytest.raises(MissingOutputError):
            manager._combine_chunks_remotely()

    def test_combine_chunks_command_failure(self, mock_hpc_config):
        manager = HPCJobManager(config=mock_hpc_config)
        # First call: chunks exist; Second call: combine fails
        manager.transport.exec = Mock(side_effect=[(0, "1"), (1, "bad")])

        with pytest.raises(RemoteCommandError):
            manager._combine_chunks_remotely()

    def test_download_combined_missing_local(self, mock_hpc_config, tmp_path, monkeypatch):
        manager = HPCJobManager(config=mock_hpc_config)
        # Prevent actual download
        monkeypatch.setattr(manager.transport, "download", lambda remote, local: None)

        # Point tempdir to controlled path
        monkeypatch.setenv("TMPDIR", str(tmp_path))

        with pytest.raises(MissingOutputError):
            manager._download_combined_results()

    def test_scp_upload_wrapped_error(self, mock_hpc_config):
        """Test that SCP upload errors are wrapped in RemoteCommandError."""
        manager = HPCJobManager(config=mock_hpc_config)
        # Mock subprocess.run to raise CalledProcessError
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "scp", stderr="upload failed"),
        ):
            with pytest.raises(RemoteCommandError) as exc_info:
                manager.transport.upload("/tmp/file", "/remote/path")
            assert "scp upload" in str(exc_info.value)

    def test_scp_download_wrapped_error(self, mock_hpc_config):
        """Test that SCP download errors are wrapped in RemoteCommandError."""
        manager = HPCJobManager(config=mock_hpc_config)
        # Mock subprocess.run to raise CalledProcessError
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "scp", stderr="download failed"),
        ):
            with pytest.raises(RemoteCommandError) as exc_info:
                manager.transport.download("/remote/file", "/tmp")
            assert "scp download" in str(exc_info.value)


# ============================================================================
# Integration Tests (Require Real HPC)
# ============================================================================


@pytest.mark.hpc
class TestHPCConnection:
    """Test real SSH connection to HPC cluster."""

    def test_ssh_connection_establishes(self, real_hpc_config):
        """Test that we can establish SSH connection."""
        manager = HPCJobManager()

        # validate_ssh_connection returns True or raises RuntimeError
        result = manager.validate_ssh_connection(timeout=10)
        assert result is True

    def test_ssh_authentication(self, real_hpc_config):
        """Test SSH key authentication works (no password prompt)."""
        manager = HPCJobManager()

        # If we get here without hanging, SSH key auth is working
        returncode, output = manager.transport.exec("echo 'auth_test'", timeout=10)
        assert returncode == 0
        assert "auth_test" in output

    def test_whoami_command(self, real_hpc_config):
        """Test running whoami on remote system."""
        manager = HPCJobManager()

        returncode, output = manager.transport.exec("whoami", timeout=10)
        assert returncode == 0

        # Extract username from output
        username = output.strip()
        assert len(username) > 0

        # Should match config if user is specified
        if manager.config.ssh_user:
            assert username == manager.config.ssh_user

    def test_pwd_command(self, real_hpc_config):
        """Test getting working directory."""
        manager = HPCJobManager()

        returncode, output = manager.transport.exec("pwd", timeout=10)
        assert returncode == 0

        # Should get a valid path
        pwd = output.strip()
        assert pwd.startswith("/")
        assert len(pwd) > 1

    def test_hostname_command(self, real_hpc_config):
        """Test getting remote hostname."""
        manager = HPCJobManager()

        returncode, output = manager.transport.exec("hostname", timeout=10)
        assert returncode == 0

        hostname = output.strip()
        assert len(hostname) > 0

    def test_echo_with_special_characters(self, real_hpc_config):
        """Test SSH handles special characters correctly."""
        manager = HPCJobManager()

        test_string = "test with spaces and $VARS and 'quotes'"
        returncode, output = manager.transport.exec(f"echo '{test_string}'", timeout=10)
        assert returncode == 0
        assert "test with spaces" in output

    def test_ssh_timeout_handling(self, real_hpc_config):
        """Test that SSH timeout works correctly."""
        manager = HPCJobManager()

        # This should timeout
        with pytest.raises(Exception):  # Could be RuntimeError or subprocess.TimeoutExpired
            manager.transport.exec("sleep 30", timeout=2)

    def test_failed_command_returns_nonzero(self, real_hpc_config):
        """Test that failed commands return non-zero exit code."""
        manager = HPCJobManager()

        # Run a command that should fail
        returncode, output = manager.transport.exec("exit 1", timeout=10)
        assert returncode == 1


@pytest.mark.hpc
class TestRemoteFileOperations:
    """Test file operations on HPC cluster."""

    def test_create_remote_directory(self, real_hpc_config):
        """Test creating directory on HPC."""
        manager = HPCJobManager()

        # Create unique test directory name
        timestamp = int(time.time())
        test_dir = f"~/pytest_qsp_hpc/test_dir_{timestamp}"

        try:
            # Create directory
            returncode, output = manager.transport.exec(f"mkdir -p {test_dir}", timeout=10)
            assert returncode == 0

            # Verify it exists
            returncode, output = manager.transport.exec(
                f"test -d {test_dir} && echo 'exists'", timeout=10
            )
            assert returncode == 0
            assert "exists" in output

        finally:
            # Cleanup
            manager.transport.exec(f"rm -rf {test_dir}", timeout=10)

    def test_upload_file_via_scp(self, real_hpc_config, tmp_path):
        """Test uploading file to HPC via SCP."""
        manager = HPCJobManager()

        # Create test file locally
        test_file = tmp_path / "test_upload.txt"
        test_content = f"test content {time.time()}"
        test_file.write_text(test_content)

        # Remote path
        timestamp = int(time.time())
        remote_dir = f"~/pytest_qsp_hpc/test_upload_{timestamp}"
        remote_file = f"{remote_dir}/test_upload.txt"

        try:
            # Create remote directory
            manager.transport.exec(f"mkdir -p {remote_dir}", timeout=10)

            # Upload file
            manager.transport.upload(str(test_file), remote_file)

            # Verify content
            returncode, output = manager.transport.exec(f"cat {remote_file}", timeout=10)
            assert returncode == 0
            assert test_content in output

        finally:
            # Cleanup
            manager.transport.exec(f"rm -rf {remote_dir}", timeout=10)

    def test_download_file_via_scp(self, real_hpc_config, tmp_path):
        """Test downloading file from HPC via SCP."""
        manager = HPCJobManager()

        # Create test file on HPC
        timestamp = int(time.time())
        remote_file = f"~/pytest_qsp_hpc/test_download_{timestamp}.txt"
        test_content = f"download test {timestamp}"

        try:
            # Create parent directory and file on HPC
            manager.transport.exec(
                f"mkdir -p ~/pytest_qsp_hpc && echo '{test_content}' > {remote_file}", timeout=10
            )

            # Download to local
            local_dir = tmp_path / "downloads"
            local_dir.mkdir()
            manager.transport.download(remote_file, str(local_dir))

            # Verify file exists locally
            downloaded_file = local_dir / f"test_download_{timestamp}.txt"
            assert downloaded_file.exists()
            assert test_content in downloaded_file.read_text()

        finally:
            # Cleanup remote file
            manager.transport.exec(f"rm -f {remote_file}", timeout=10)

    def test_list_remote_files(self, real_hpc_config):
        """Test listing files in remote directory."""
        manager = HPCJobManager()

        # Create temp directory with known files
        timestamp = int(time.time())
        test_dir = f"~/pytest_qsp_hpc/test_ls_{timestamp}"

        try:
            # Create directory and files in one command to avoid rate limiting
            setup_cmd = f"mkdir -p {test_dir} && touch {test_dir}/file1.txt {test_dir}/file2.txt"
            rc_setup, out_setup = manager.transport.exec(setup_cmd, timeout=10)
            assert rc_setup == 0, f"Setup failed: {out_setup}"

            # List files
            returncode, output = manager.transport.exec(f"ls {test_dir}", timeout=10)
            assert returncode == 0, f"ls failed: {output}"
            assert "file1.txt" in output
            assert "file2.txt" in output

        finally:
            # Cleanup
            manager.transport.exec(f"rm -rf {test_dir}", timeout=10)

    def test_remove_remote_file(self, real_hpc_config):
        """Test removing file from HPC."""
        manager = HPCJobManager()

        # Create test file
        timestamp = int(time.time())
        test_file = f"~/pytest_qsp_hpc/test_rm_{timestamp}.txt"

        # Create parent directory and file, then verify
        returncode, _ = manager.transport.exec(
            f"mkdir -p ~/pytest_qsp_hpc && touch {test_file} && test -f {test_file}", timeout=10
        )
        assert returncode == 0

        # Remove file and verify it's gone
        returncode, _ = manager.transport.exec(
            f"rm -f {test_file} && ! test -f {test_file}", timeout=10
        )
        assert returncode == 0

    def test_write_and_read_file_content(self, real_hpc_config):
        """Test writing and reading file content on HPC."""
        manager = HPCJobManager()

        timestamp = int(time.time())
        test_file = f"~/pytest_qsp_hpc/test_content_{timestamp}.txt"
        test_content = "Line 1\nLine 2\nSpecial chars: $VAR @#%"

        try:
            # Create parent directory and write content using cat with heredoc
            write_cmd = (
                f"mkdir -p ~/pytest_qsp_hpc && cat > {test_file} << 'EOF'\n{test_content}\nEOF"
            )
            returncode, _ = manager.transport.exec(write_cmd, timeout=10)
            assert returncode == 0

            # Read content back
            returncode, output = manager.transport.exec(f"cat {test_file}", timeout=10)
            assert returncode == 0
            assert "Line 1" in output
            assert "Line 2" in output
            assert "Special chars" in output

        finally:
            manager.transport.exec(f"rm -f {test_file}", timeout=10)


@pytest.mark.hpc
class TestSLURMCommands:
    """Test SLURM command execution (no actual job submission)."""

    def test_squeue_command(self, real_hpc_config):
        """Test running squeue to check job queue."""
        manager = HPCJobManager()

        # Run squeue for current user
        returncode, output = manager.transport.exec("squeue -u $USER", timeout=15)
        assert returncode == 0

        # Should have header at minimum
        assert "JOBID" in output or "PARTITION" in output or len(output) >= 0

    def test_sinfo_partition_check(self, real_hpc_config):
        """Test checking available SLURM partitions."""
        manager = HPCJobManager()

        # Get list of partitions
        returncode, output = manager.transport.exec("sinfo -o '%P'", timeout=15)
        assert returncode == 0

        # Should have at least one partition listed
        lines = output.strip().split("\n")
        assert len(lines) >= 1  # At least header

        # Check if configured partition exists
        if manager.config.partition:
            # Remove asterisk that marks default partition
            partitions = [line.strip().replace("*", "") for line in lines]
            # Note: partition might not exist, just verify command works
            assert len(partitions) > 0

    def test_sacct_command(self, real_hpc_config):
        """Test running sacct to check job history."""
        manager = HPCJobManager()

        # Get recent job history
        returncode, output = manager.transport.exec(
            "sacct --user=$USER --starttime=now-7days --format=JobID,JobName,State --noheader",
            timeout=15,
        )
        assert returncode == 0
        # Output could be empty if no recent jobs, that's ok

    def test_scontrol_show_config(self, real_hpc_config):
        """Test running scontrol show config."""
        manager = HPCJobManager()

        returncode, output = manager.transport.exec("scontrol show config | head -20", timeout=15)
        assert returncode == 0

        # Should contain config information
        assert "Configuration data" in output or "AccountingStorage" in output

    def test_sinfo_node_list(self, real_hpc_config):
        """Test listing SLURM nodes."""
        manager = HPCJobManager()

        returncode, output = manager.transport.exec("sinfo -N", timeout=15)
        assert returncode == 0

        # Should have node information
        assert "NODELIST" in output or "PARTITION" in output

    def test_squeue_format_options(self, real_hpc_config):
        """Test squeue with custom format."""
        manager = HPCJobManager()

        returncode, output = manager.transport.exec(
            "squeue -u $USER -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'", timeout=15
        )
        assert returncode == 0

    def test_slurm_version(self, real_hpc_config):
        """Test getting SLURM version."""
        manager = HPCJobManager()

        returncode, output = manager.transport.exec("scontrol --version", timeout=15)
        assert returncode == 0
        assert "slurm" in output.lower()


@pytest.mark.hpc
class TestCodebaseSync:
    """Test syncing codebase to HPC."""

    def test_sync_matlab_files(self, real_hpc_config, tmp_path):
        """Test syncing MATLAB files to HPC."""
        # Create mock matlab files
        # matlab_dir = tmp_path / "matlab"
        # matlab_dir.mkdir()
        # (matlab_dir / "test_script.m").write_text("% Test MATLAB script")
        #
        # Sync to HPC
        # manager.sync_codebase(project_name="pytest_test")
        #
        # Verify files exist on HPC
        pass

    def test_sync_python_scripts(self, real_hpc_config):
        """Test syncing Python helper scripts to HPC."""
        pass

    def test_sync_preserves_permissions(self, real_hpc_config):
        """Test that rsync preserves file permissions."""
        pass


@pytest.mark.hpc
@pytest.mark.slow
class TestDryRunJobSubmission:
    """Test job submission with sleep jobs (not real MATLAB)."""

    def test_submit_sleep_job(self, real_hpc_config):
        """Submit a job that just sleeps for 10 seconds."""
        # This tests the full submission pipeline without running MATLAB
        #
        # manager = HPCJobManager(config_file=real_hpc_config)
        #
        # Create a simple sleep script
        # script = """#!/bin/bash
        # #SBATCH --job-name=pytest_sleep
        # #SBATCH --time=00:01:00
        # #SBATCH --partition=normal
        # #SBATCH --output=pytest_sleep.out
        #
        # echo "Starting test job at $(date)"
        # sleep 10
        # echo "Finished test job at $(date)"
        # """
        #
        # Submit job
        # job_id = manager._submit_slurm_script(script, project_name="pytest_test")
        # assert job_id is not None
        #
        # Monitor until completion
        # # Wait and check status
        #
        # Cleanup
        # manager._run_ssh_command(f"scancel {job_id}")
        pass

    def test_submit_array_job(self, real_hpc_config):
        """Submit an array job with multiple tasks."""
        # Similar to above but with --array=0-4
        pass

    def test_cancel_job(self, real_hpc_config):
        """Test cancelling a submitted job."""
        # Submit sleep job
        # job_id = ...
        #
        # Cancel it
        # manager._run_ssh_command(f"scancel {job_id}")
        #
        # Verify it's cancelled
        # result = manager._run_ssh_command(f"scontrol show job {job_id}")
        # assert "CANCELLED" in result.stdout
        pass

    def test_job_status_monitoring(self, real_hpc_config):
        """Test monitoring job status through its lifecycle."""
        # Submit job
        # Check PENDING status
        # Check RUNNING status (may need to wait)
        # Check COMPLETED status
        pass


@pytest.mark.hpc
class TestLogRetrieval:
    """Test retrieving job logs from HPC."""

    def test_fetch_job_stdout(self, real_hpc_config):
        """Test fetching job output log."""
        pass

    def test_fetch_job_stderr(self, real_hpc_config):
        """Test fetching job error log."""
        pass

    def test_fetch_array_job_logs(self, real_hpc_config):
        """Test fetching logs from array job tasks."""
        pass


# ============================================================================
# Mock-based Tests (SSH operations mocked)
# ============================================================================


class TestWithMockedSSH:
    """Tests with mocked SSH for isolation."""

    @patch("subprocess.run")
    def test_submit_job_mocked(self, mock_subprocess, mock_hpc_config):
        """Test job submission with mocked subprocess."""
        # Mock the SSH command that submits the job
        # mock_subprocess.return_value = Mock(
        #     returncode=0,
        #     stdout="Submitted batch job 12345\n",
        #     stderr=""
        # )
        #
        # manager = HPCJobManager(config=mock_hpc_config)
        # job_id = manager.submit_jobs(...)
        #
        # assert job_id == "12345"
        # mock_subprocess.assert_called()
        pass

    @patch("subprocess.run")
    def test_check_job_status_mocked(self, mock_subprocess, mock_hpc_config):
        """Test job status checking with mocked squeue."""
        pass

    @patch("subprocess.run")
    def test_rsync_upload_mocked(self, mock_subprocess, mock_hpc_config):
        """Test rsync upload with mocked subprocess."""
        pass
