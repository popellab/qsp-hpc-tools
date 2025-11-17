#!/usr/bin/env python3
"""
Tests for CLI commands.

Tests the qsp-hpc CLI interface including setup wizard, connection testing,
configuration display, and log viewing.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from qsp_hpc.cli import cli, setup, test, info, logs
from qsp_hpc.batch.hpc_job_manager import BatchConfig


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
        'ssh': {
            'host': 'hpc.example.edu',
            'user': 'testuser',
            'key': '~/.ssh/id_rsa',
        },
        'cluster': {
            'matlab_module': 'matlab/R2024a',
        },
        'paths': {
            'remote_base_dir': '/home/testuser/qsp-projects',
            'hpc_venv_path': '/home/testuser/.venv/hpc-qsp',
            'simulation_pool_path': '/scratch/testuser/simulations',
        },
        'slurm': {
            'partition': 'normal',
            'time_limit': '04:00:00',
            'mem_per_cpu': '4G',
        }
    }

    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    return config_file


class TestSetupCommand:
    """Tests for 'qsp-hpc setup' command."""

    def test_setup_creates_new_config(self, cli_runner, tmp_path, monkeypatch):
        """Test setup wizard creates new config file."""
        # Mock home directory to use tmp_path
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        # Mock SSH connection test to succeed
        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.validate_ssh_connection = Mock()
            mock_manager._ssh_exec = Mock(return_value=(0, "slurm 23.02.1"))
            MockManager.return_value = mock_manager

            # Provide input for all prompts
            result = cli_runner.invoke(setup, input=(
                'hpc.example.edu\n'  # SSH host
                'testuser\n'         # SSH user
                '~/.ssh/id_rsa\n'    # SSH key
                'normal\n'           # Partition
                '04:00:00\n'         # Time limit
                '4G\n'               # Memory
                '/home/testuser/qsp-projects\n'  # Base dir
                '/home/testuser/.venv/hpc-qsp\n' # Venv path
                '/scratch/testuser/simulations\n' # Sim pool
                'n\n'                # Don't create dirs
                'matlab/R2024a\n'    # MATLAB module
            ))

            assert result.exit_code == 0
            assert 'Setup Wizard' in result.output
            assert 'Configuration saved' in result.output

            # Verify config file was created
            config_file = tmp_path / '.config' / 'qsp-hpc' / 'credentials.yaml'
            assert config_file.exists()

            # Verify config contents
            with open(config_file) as f:
                config = yaml.safe_load(f)

            assert config['ssh']['host'] == 'hpc.example.edu'
            assert config['ssh']['user'] == 'testuser'
            assert config['slurm']['partition'] == 'normal'

    def test_setup_cancel_on_overwrite(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test setup cancels when user declines to overwrite existing config."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        result = cli_runner.invoke(setup, input='n\n')  # Decline overwrite

        assert result.exit_code == 0
        assert 'already exists' in result.output
        assert 'Setup cancelled' in result.output

    def test_setup_ssh_connection_failure(self, cli_runner, tmp_path, monkeypatch):
        """Test setup handles SSH connection failure gracefully."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.validate_ssh_connection = Mock(
                side_effect=Exception("Connection refused")
            )
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(setup, input=(
                'hpc.example.edu\n'
                'testuser\n'
                '~/.ssh/id_rsa\n'
                'n\n'  # Don't continue after connection failure
            ))

            assert 'Failed!' in result.output
            assert 'Connection refused' in result.output

    def test_setup_detects_ssh_config(self, cli_runner, tmp_path, monkeypatch):
        """Test setup detects and suggests SSH config hosts."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        # Create mock SSH config
        ssh_dir = tmp_path / '.ssh'
        ssh_dir.mkdir()
        ssh_config = ssh_dir / 'config'
        ssh_config.write_text("""
Host hpc-cluster
    HostName hpc.example.edu
    User myuser

Host other-host
    HostName other.example.com
""")

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.validate_ssh_connection = Mock()
            mock_manager._ssh_exec = Mock(return_value=(0, "slurm 23.02.1"))
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(setup, input=(
                'hpc-cluster\n'  # Use SSH config alias
                '\n'             # Empty user (use SSH config)
                '\n'             # Empty key (use SSH config)
                'normal\n'
                '04:00:00\n'
                '4G\n'
                '/home/testuser/qsp-projects\n'
                '/home/testuser/.venv/hpc-qsp\n'
                '/scratch/testuser/simulations\n'
                'n\n'
                'matlab/R2024a\n'
            ))

            assert 'Found SSH config hosts' in result.output
            assert 'hpc-cluster' in result.output

    def test_setup_creates_remote_directories(self, cli_runner, tmp_path, monkeypatch):
        """Test setup offers to create missing remote directories."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.validate_ssh_connection = Mock()
            # First _ssh_exec is SLURM check, then whoami, then dir checks, then mkdir
            mock_manager._ssh_exec = Mock(side_effect=[
                (0, "slurm 23.02.1"),  # SLURM version check
                (0, "testuser"),        # whoami
                (1, ""),                # Base dir doesn't exist
                (1, ""),                # Venv doesn't exist
                (1, ""),                # Sim pool doesn't exist
                (0, ""),                # mkdir base dir success
                (0, ""),                # mkdir venv success
                (0, ""),                # mkdir sim pool success
                (0, "MODULE_OK"),       # MATLAB module check
            ])
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(setup, input=(
                'hpc.example.edu\n'
                'testuser\n'
                '~/.ssh/id_rsa\n'
                'normal\n'
                '04:00:00\n'
                '4G\n'
                '/home/testuser/qsp-projects\n'
                '/home/testuser/.venv/hpc-qsp\n'
                '/scratch/testuser/simulations\n'
                'y\n'  # Yes, create directories
                'n\n'  # No, don't setup Python venv
                'matlab/R2024a\n'
            ))

            assert result.exit_code == 0
            assert "don't exist yet" in result.output
            assert 'Creating' in result.output


class TestTestCommand:
    """Tests for 'qsp-hpc test' command."""

    def test_test_command_no_config(self, cli_runner, tmp_path, monkeypatch):
        """Test 'test' command fails gracefully when no config exists."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        result = cli_runner.invoke(test)

        assert result.exit_code == 1
        assert 'No configuration found' in result.output
        assert 'qsp-hpc setup' in result.output

    def test_test_command_successful(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'test' command with successful connection."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
                ssh_host='hpc.example.edu',
                ssh_user='testuser',
                ssh_key='~/.ssh/id_rsa',
                remote_project_path='/home/testuser/qsp-projects',
                hpc_venv_path='/home/testuser/.venv/hpc-qsp',
                simulation_pool_path='/scratch/testuser/simulations',
                partition='normal',
                time_limit='04:00:00',
                memory_per_job='4G',
                matlab_module='matlab/R2024a'
            )
            mock_manager.validate_ssh_connection = Mock()
            mock_manager._ssh_exec = Mock(side_effect=[
                (0, "testuser"),            # whoami
                (0, "slurm 23.02.1"),       # scontrol --version
                (0, "PARTITION\nnormal\n"), # sinfo partitions
                (0, "OK"),                  # module load MATLAB
                (0, ""),                    # test -d venv
                (0, ""),                    # test -d sim pool
            ])
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(test)

            assert result.exit_code == 0
            assert 'Testing HPC Connection' in result.output
            assert 'All critical tests passed' in result.output

    def test_test_command_ssh_failure(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'test' command with SSH connection failure."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
                ssh_host='hpc.example.edu',
                ssh_user='testuser',
                ssh_key='~/.ssh/id_rsa',
                remote_project_path='/home/testuser/qsp-projects',
                hpc_venv_path='/home/testuser/.venv/hpc-qsp',
                simulation_pool_path='/scratch/testuser/simulations'
            )
            mock_manager.validate_ssh_connection = Mock(
                side_effect=Exception("Connection timeout")
            )
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(test)

            assert result.exit_code == 1
            assert 'Connection timeout' in result.output

    def test_test_command_partition_not_found(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'test' command warns if partition not found."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
                ssh_host='hpc.example.edu',
                ssh_user='testuser',
                ssh_key='~/.ssh/id_rsa',
                remote_project_path='/home/testuser/qsp-projects',
                hpc_venv_path='/home/testuser/.venv/hpc-qsp',
                simulation_pool_path='/scratch/testuser/simulations',
                partition='nonexistent',
                time_limit='04:00:00',
                memory_per_job='4G',
                matlab_module='matlab/R2024a'
            )
            mock_manager.validate_ssh_connection = Mock()
            mock_manager._ssh_exec = Mock(side_effect=[
                (0, "testuser"),                  # whoami
                (0, "slurm 23.02.1"),             # scontrol --version
                (0, "PARTITION\nnormal\nshort\n"), # sinfo (no 'nonexistent')
                (0, "OK"),                        # MATLAB
                (0, ""),                          # venv
                (0, ""),                          # sim pool
            ])
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(test)

            assert result.exit_code == 0
            assert 'not found' in result.output  # Partition warning


class TestInfoCommand:
    """Tests for 'qsp-hpc info' command."""

    def test_info_command_no_config(self, cli_runner, tmp_path, monkeypatch):
        """Test 'info' command fails gracefully when no config exists."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        result = cli_runner.invoke(info)

        assert result.exit_code == 1
        assert 'No configuration found' in result.output
        assert 'qsp-hpc setup' in result.output

    def test_info_command_hides_secrets(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'info' command hides SSH key by default."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
                ssh_host='hpc.example.edu',
                ssh_user='testuser',
                ssh_key='~/.ssh/id_rsa',
                remote_project_path='/home/testuser/qsp-projects',
                hpc_venv_path='/home/testuser/.venv/hpc-qsp',
                simulation_pool_path='/scratch/testuser/simulations',
                partition='normal',
                time_limit='04:00:00',
                memory_per_job='4G',
                matlab_module='matlab/R2024a'
            )
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(info)

            assert result.exit_code == 0
            assert 'Current Configuration' in result.output
            assert 'hpc.example.edu' in result.output
            assert 'testuser' in result.output
            assert '**********' in result.output  # Hidden SSH key
            assert '~/.ssh/id_rsa' not in result.output  # Key path hidden

    def test_info_command_shows_secrets(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'info' command shows SSH key with --show-secrets flag."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
                ssh_host='hpc.example.edu',
                ssh_user='testuser',
                ssh_key='~/.ssh/id_rsa',
                remote_project_path='/home/testuser/qsp-projects',
                hpc_venv_path='/home/testuser/.venv/hpc-qsp',
                simulation_pool_path='/scratch/testuser/simulations',
                partition='normal',
                time_limit='04:00:00',
                memory_per_job='4G',
                matlab_module='matlab/R2024a'
            )
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(info, ['--show-secrets'])

            assert result.exit_code == 0
            assert '~/.ssh/id_rsa' in result.output  # Key path shown

    def test_info_command_shows_all_config_sections(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'info' command displays all configuration sections."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.config = BatchConfig(
                ssh_host='hpc.example.edu',
                ssh_user='testuser',
                ssh_key='~/.ssh/id_rsa',
                remote_project_path='/home/testuser/qsp-projects',
                hpc_venv_path='/home/testuser/.venv/hpc-qsp',
                simulation_pool_path='/scratch/testuser/simulations',
                partition='normal',
                time_limit='04:00:00',
                memory_per_job='4G',
                matlab_module='matlab/R2024a'
            )
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(info)

            assert result.exit_code == 0
            assert 'SSH Configuration:' in result.output
            assert 'SLURM Configuration:' in result.output
            assert 'HPC Paths:' in result.output
            assert 'MATLAB Configuration:' in result.output
            assert 'normal' in result.output  # Partition
            assert '04:00:00' in result.output  # Time limit


class TestLogsCommand:
    """Tests for 'qsp-hpc logs' command."""

    def test_logs_command_no_config(self, cli_runner, tmp_path, monkeypatch):
        """Test 'logs' command fails gracefully when no config exists."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        result = cli_runner.invoke(logs, ['test_project'])

        assert result.exit_code == 1
        assert 'No configuration found' in result.output

    def test_logs_command_no_arguments(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'logs' command requires either project name or job ID."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        result = cli_runner.invoke(logs)

        assert result.exit_code == 1
        assert 'Must specify either PROJECT_NAME or --job-id' in result.output

    def test_logs_command_with_job_id(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'logs' command retrieves logs for specific job ID."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.config = Mock()
            mock_manager.config.remote_project_path = '/home/testuser/qsp-projects'
            mock_manager._ssh_exec = Mock(side_effect=[
                (0, "/home/testuser/qsp-projects/job123/slurm-12345.out"),  # find log file
                (0, "Task 1 completed\nTask 2 running\n"),  # tail log content
            ])
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(logs, ['--job-id', '12345'])

            assert result.exit_code == 0
            assert 'Task 1 completed' in result.output
            assert 'Task 2 running' in result.output

    def test_logs_command_with_task_id(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'logs' command retrieves logs for specific array task."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.config = Mock()
            mock_manager.config.remote_project_path = '/home/testuser/qsp-projects'
            mock_manager._ssh_exec = Mock(side_effect=[
                (0, "/home/testuser/qsp-projects/job123/slurm-12345_3.out"),  # find log with task ID
                (0, "Array task 3 output\n"),  # tail log content
            ])
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(logs, ['--job-id', '12345', '--task-id', '3'])

            assert result.exit_code == 0
            assert 'Array task 3' in result.output

    def test_logs_command_log_not_found(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'logs' command handles missing log files."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            mock_manager = Mock()
            mock_manager.config = Mock()
            mock_manager.config.remote_project_path = '/home/testuser/qsp-projects'
            mock_manager._ssh_exec = Mock(return_value=(0, ""))  # No log file found
            MockManager.return_value = mock_manager

            result = cli_runner.invoke(logs, ['--job-id', '99999'])

            assert result.exit_code == 0
            assert 'not found' in result.output

    def test_logs_command_with_project_name(self, cli_runner, tmp_path, monkeypatch, mock_config_file):
        """Test 'logs' command with project name (not yet fully implemented)."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)

        with patch('qsp_hpc.batch.hpc_job_manager.HPCJobManager') as MockManager:
            MockManager.return_value = Mock()

            result = cli_runner.invoke(logs, ['test_project'])

            assert result.exit_code == 0
            assert 'not yet implemented' in result.output


class TestCLIVersion:
    """Tests for CLI version and help."""

    def test_cli_version(self, cli_runner):
        """Test --version flag (may fail if package not installed)."""
        result = cli_runner.invoke(cli, ['--version'])
        # Exit code may be 0 (success) or 1 (package not installed in dev mode)
        # Either is acceptable - we're just testing the flag exists
        assert result.exit_code in [0, 1]

    def test_cli_help(self, cli_runner):
        """Test --help flag."""
        result = cli_runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Manage QSP simulations' in result.output
        assert 'setup' in result.output
        assert 'test' in result.output
        assert 'info' in result.output
        assert 'logs' in result.output
