#!/usr/bin/env python3
"""
Tests for CLI commands.

Tests the qsp-hpc CLI interface including setup wizard, connection testing,
configuration display, and log viewing.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from qsp_hpc.batch.hpc_job_manager import BatchConfig
from qsp_hpc.cli import cli, info, logs, setup, test


@pytest.fixture
def cli_runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a temporary config file."""
    config_dir = tmp_path / ".config" / "qsp-hpc"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "credentials.yaml"

    config_data = {
        "ssh": {
            "host": "hpc.example.edu",
            "user": "testuser",
            "key": "~/.ssh/id_rsa",
        },
        "cluster": {
            "matlab_module": "matlab/R2024a",
        },
        "paths": {
            "remote_base_dir": "/home/testuser/qsp-projects",
            "hpc_venv_path": "/home/testuser/.venv/hpc-qsp",
            "simulation_pool_path": "/scratch/testuser/simulations",
        },
        "slurm": {
            "partition": "normal",
            "time_limit": "04:00:00",
            "mem_per_cpu": "4G",
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return config_file


class TestSetupCommand:
    """Tests for 'qsp-hpc setup' command."""

    def test_setup_creates_new_config(self, cli_runner, tmp_path, monkeypatch):
        """Test setup wizard creates new config file with correct values."""
        # Mock home directory to use tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock SSH connection test to succeed
        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.validate_ssh_connection = Mock()
            mock_manager.transport.exec = Mock(return_value=(0, "slurm 23.02.1"))
            mock_mgr_cls.return_value = mock_manager

            # Provide input for all prompts
            result = cli_runner.invoke(
                setup,
                input=(
                    "hpc.example.edu\n"  # SSH host
                    "testuser\n"  # SSH user
                    "~/.ssh/id_rsa\n"  # SSH key
                    "normal\n"  # Partition
                    "04:00:00\n"  # Time limit
                    "4G\n"  # Memory
                    "data\n"  # Data base directory name
                    "/home/testuser/qsp-projects\n"  # Base dir
                    "/home/testuser/.venv/hpc-qsp\n"  # Venv path
                    "/scratch/testuser/simulations\n"  # Sim pool
                    "matlab/R2024a\n"  # MATLAB module
                    "n\n"  # Don't create dirs
                    "n\n"  # Don't setup Python venv
                ),
            )

            # Should complete successfully
            assert result.exit_code == 0

            # Verify config file was created
            config_file = tmp_path / ".config" / "qsp-hpc" / "credentials.yaml"
            assert config_file.exists()

            # Verify config contains correct values
            with open(config_file) as f:
                config = yaml.safe_load(f)

            assert config["ssh"]["host"] == "hpc.example.edu"
            assert config["ssh"]["user"] == "testuser"
            assert config["slurm"]["partition"] == "normal"
            assert config["slurm"]["time_limit"] == "04:00:00"
            assert config["paths"]["remote_base_dir"] == "/home/testuser/qsp-projects"

    def test_setup_cancel_on_overwrite(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test setup cancels when user declines to overwrite existing config."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Get original config content
        with open(mock_config_file) as f:
            original_config = yaml.safe_load(f)

        result = cli_runner.invoke(setup, input="n\n")  # Decline overwrite

        # Should exit cleanly
        assert result.exit_code == 0

        # Config file should remain unchanged
        with open(mock_config_file) as f:
            current_config = yaml.safe_load(f)
        assert current_config == original_config

    def test_setup_ssh_connection_failure(self, cli_runner, tmp_path, monkeypatch):
        """Test setup handles SSH connection failure gracefully."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.validate_ssh_connection = Mock(side_effect=Exception("Connection refused"))
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(
                setup,
                input=(
                    "hpc.example.edu\n"
                    "testuser\n"
                    "~/.ssh/id_rsa\n"
                    "n\n"  # Don't continue after connection failure
                ),
            )

            # Should show the connection error
            assert "Connection refused" in result.output

    def test_setup_detects_ssh_config(self, cli_runner, tmp_path, monkeypatch):
        """Test setup uses SSH config values correctly."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Create mock SSH config
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        ssh_config = ssh_dir / "config"
        ssh_config.write_text(
            """
Host hpc-cluster
    HostName hpc.example.edu
    User myuser

Host other-host
    HostName other.example.com
"""
        )

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.validate_ssh_connection = Mock()
            mock_manager.transport.exec = Mock(return_value=(0, "slurm 23.02.1"))
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(
                setup,
                input=(
                    "hpc-cluster\n"  # Use SSH config alias
                    "\n"  # Empty user (use SSH config)
                    "\n"  # Empty key (use SSH config)
                    "normal\n"
                    "04:00:00\n"
                    "4G\n"
                    "data\n"
                    "/home/testuser/qsp-projects\n"
                    "/home/testuser/.venv/hpc-qsp\n"
                    "/scratch/testuser/simulations\n"
                    "matlab/R2024a\n"
                    "n\n"
                    "n\n"
                ),
            )

            assert result.exit_code == 0

            # Verify config uses SSH alias
            config_file = tmp_path / ".config" / "qsp-hpc" / "credentials.yaml"
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert config["ssh"]["host"] == "hpc-cluster"

    def test_setup_creates_remote_directories(self, cli_runner, tmp_path, monkeypatch):
        """Test setup creates missing remote directories when requested."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.validate_ssh_connection = Mock()
            # Mock exec calls for: SLURM check, whoami, dir checks, mkdirs, MATLAB
            mock_manager.transport.exec = Mock(
                side_effect=[
                    (0, "slurm 23.02.1"),  # SLURM version check
                    (0, "testuser"),  # whoami
                    (1, ""),  # Base dir doesn't exist
                    (1, ""),  # Venv doesn't exist
                    (1, ""),  # Sim pool doesn't exist
                    (0, ""),  # mkdir base dir success
                    (0, ""),  # mkdir venv success
                    (0, ""),  # mkdir sim pool success
                    (0, "MODULE_OK"),  # MATLAB module check
                ]
            )
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(
                setup,
                input=(
                    "hpc.example.edu\n"
                    "testuser\n"
                    "~/.ssh/id_rsa\n"
                    "normal\n"
                    "04:00:00\n"
                    "4G\n"
                    "data\n"
                    "/home/testuser/qsp-projects\n"
                    "/home/testuser/.venv/hpc-qsp\n"
                    "/scratch/testuser/simulations\n"
                    "matlab/R2024a\n"
                    "y\n"  # Yes, create directories
                    "n\n"  # No, don't setup Python venv
                ),
            )

            assert result.exit_code == 0

            # Verify mkdir commands were called (3 directories)
            mkdir_calls = [
                call for call in mock_manager.transport.exec.call_args_list if "mkdir" in str(call)
            ]
            assert len(mkdir_calls) == 3


class TestTestCommand:
    """Tests for 'qsp-hpc test' command."""

    def test_test_command_no_config(self, cli_runner, tmp_path, monkeypatch):
        """Test 'test' command fails gracefully when no config exists."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = cli_runner.invoke(test)

        # Should fail with exit code 1
        assert result.exit_code == 1

    def test_test_command_successful(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'test' command with successful connection."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
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
            mock_manager.validate_ssh_connection = Mock()
            mock_manager.transport.exec = Mock(
                side_effect=[
                    (0, "testuser"),  # whoami
                    (0, "slurm 23.02.1"),  # scontrol --version
                    (0, "PARTITION\nnormal\n"),  # sinfo partitions
                    (0, "OK"),  # module load MATLAB
                    (0, ""),  # test -d venv
                    (0, ""),  # test -d sim pool
                ]
            )
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(test)

            # Should succeed
            assert result.exit_code == 0

    def test_test_command_ssh_failure(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'test' command with SSH connection failure."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
                ssh_host="hpc.example.edu",
                ssh_user="testuser",
                ssh_key="~/.ssh/id_rsa",
                remote_project_path="/home/testuser/qsp-projects",
                hpc_venv_path="/home/testuser/.venv/hpc-qsp",
                simulation_pool_path="/scratch/testuser/simulations",
            )
            mock_manager.validate_ssh_connection = Mock(side_effect=Exception("Connection timeout"))
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(test)

            assert result.exit_code == 1
            assert "Connection timeout" in result.output

    def test_test_command_partition_not_found(
        self, cli_runner, tmp_path, monkeypatch, mock_config_file
    ):
        """Test 'test' command warns if partition not found."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
                ssh_host="hpc.example.edu",
                ssh_user="testuser",
                ssh_key="~/.ssh/id_rsa",
                remote_project_path="/home/testuser/qsp-projects",
                hpc_venv_path="/home/testuser/.venv/hpc-qsp",
                simulation_pool_path="/scratch/testuser/simulations",
                partition="nonexistent",
                time_limit="04:00:00",
                memory_per_job="4G",
                matlab_module="matlab/R2024a",
            )
            mock_manager.validate_ssh_connection = Mock()
            mock_manager.transport.exec = Mock(
                side_effect=[
                    (0, "testuser"),  # whoami
                    (0, "slurm 23.02.1"),  # scontrol --version
                    (0, "PARTITION\nnormal\nshort\n"),  # sinfo (no 'nonexistent')
                    (0, "OK"),  # MATLAB
                    (0, ""),  # venv
                    (0, ""),  # sim pool
                ]
            )
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(test)

            # Should complete but show partition warning
            assert result.exit_code == 0
            assert "nonexistent" in result.output  # Shows the missing partition name


class TestInfoCommand:
    """Tests for 'qsp-hpc info' command."""

    def test_info_command_no_config(self, cli_runner, tmp_path, monkeypatch):
        """Test 'info' command fails gracefully when no config exists."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = cli_runner.invoke(info)

        # Should fail
        assert result.exit_code == 1

    def test_info_command_hides_secrets(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'info' command hides SSH key by default (security feature)."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
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
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(info)

            assert result.exit_code == 0
            # Verify secrets are hidden
            assert "~/.ssh/id_rsa" not in result.output  # SSH key path should be masked
            # Verify non-secret info is shown
            assert "hpc.example.edu" in result.output
            assert "testuser" in result.output

    def test_info_command_shows_secrets(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'info' command shows SSH key with --show-secrets flag."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
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
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(info, ["--show-secrets"])

            assert result.exit_code == 0
            assert "~/.ssh/id_rsa" in result.output  # Key path shown

    def test_info_command_shows_all_config_sections(
        self, cli_runner, tmp_path, monkeypatch, mock_config_file
    ):
        """Test 'info' command displays all configuration values."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
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
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(info)

            assert result.exit_code == 0
            # Verify all config values are displayed
            assert "normal" in result.output  # Partition
            assert "04:00:00" in result.output  # Time limit
            assert "matlab/R2024a" in result.output  # MATLAB module
            assert "/home/testuser/qsp-projects" in result.output  # Paths


class TestLogsCommand:
    """Tests for 'qsp-hpc logs' command."""

    def test_logs_command_no_config(self, cli_runner, tmp_path, monkeypatch):
        """Test 'logs' command fails gracefully when no config exists."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = cli_runner.invoke(logs, ["--job-id", "12345"])

        # Should fail
        assert result.exit_code == 1

    def test_logs_command_no_arguments(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'logs' command requires job ID."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = cli_runner.invoke(logs)

        # Should fail - missing required job-id
        assert result.exit_code != 0

    def test_logs_command_with_job_id(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'logs' command retrieves logs for specific job ID."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.config = Mock()
            mock_manager.config.remote_project_path = "/home/testuser/qsp-projects"
            mock_manager.transport.exec = Mock(
                side_effect=[
                    (0, "/home/testuser/qsp-projects/job123/slurm-12345.out"),  # find log file
                    (0, "Task 1 completed\nTask 2 running\n"),  # tail log content
                ]
            )
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(logs, ["--job-id", "12345"])

            assert result.exit_code == 0
            assert "Task 1 completed" in result.output
            assert "Task 2 running" in result.output

    def test_logs_command_with_task_id(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'logs' command retrieves logs for specific array task."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.config = Mock()
            mock_manager.config.remote_project_path = "/home/testuser/qsp-projects"
            mock_manager.transport.exec = Mock(
                side_effect=[
                    (
                        0,
                        "/home/testuser/qsp-projects/job123/slurm-12345_3.out",
                    ),  # find log with task ID
                    (0, "Array task 3 output\n"),  # tail log content
                ]
            )
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(logs, ["--job-id", "12345", "--task-id", "3"])

            assert result.exit_code == 0
            assert "Array task 3" in result.output

    def test_logs_command_log_not_found(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'logs' command handles missing log files gracefully."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager") as mock_mgr_cls:
            mock_manager = Mock()
            mock_manager.config = Mock()
            mock_manager.config.remote_project_path = "/home/testuser/qsp-projects"
            mock_manager.transport.exec = Mock(return_value=(0, ""))  # No log file found
            mock_mgr_cls.return_value = mock_manager

            result = cli_runner.invoke(logs, ["--job-id", "99999"])

            # Should complete (not crash) even when log not found
            assert result.exit_code == 0


class TestCLIVersion:
    """Tests for CLI version and help."""

    def test_cli_version(self, cli_runner):
        """Test --version flag exists."""
        result = cli_runner.invoke(cli, ["--version"])
        # Exit code may be 0 (success) or 1 (package not installed in dev mode)
        # Either is acceptable - we're just testing the flag exists
        assert result.exit_code in [0, 1]

    def test_cli_help(self, cli_runner):
        """Test --help flag shows available commands."""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Verify all commands are listed
        assert "setup" in result.output
        assert "test" in result.output
        assert "info" in result.output
        assert "logs" in result.output
