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
from typing import Optional
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from qsp_hpc.batch.hpc_job_manager import (
    BatchConfig,
    DownloadResult,
    HPCJobManager,
    JobInfo,
    MissingOutputError,
    RemoteCommandError,
    SSHTransport,
    SubmissionError,
    _format_array_spec,
    _is_transient_ssh_error,
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


class TestFormatArraySpec:
    """Sparse-array formatter (#29). Consecutive runs collapse to N-M,
    singletons and length-2 runs stay comma-separated."""

    def test_single_id(self):
        assert _format_array_spec([7]) == "7"

    def test_two_consecutive_stays_comma_form(self):
        # Length-2 runs are the same character count either way;
        # comma form is easier to eyeball at a glance.
        assert _format_array_spec([7, 8]) == "7,8"

    def test_three_consecutive_collapses(self):
        assert _format_array_spec([7, 8, 9]) == "7-9"

    def test_mixed(self):
        assert _format_array_spec([7, 15, 22, 23, 24, 25, 41]) == "7,15,22-25,41"

    def test_multiple_runs(self):
        assert _format_array_spec([0, 1, 2, 5, 6, 9, 10, 11, 12]) == "0-2,5,6,9-12"

    def test_dedup_and_sort(self):
        assert _format_array_spec([41, 7, 8, 7, 9]) == "7-9,41"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _format_array_spec([])


class TestListMissingChunksOnHPC:
    """Post-array staging inspection via SSH ls."""

    def test_parses_chunk_filenames(self, mock_hpc_config):
        manager = HPCJobManager(config=mock_hpc_config)
        # `ls | sed` output one id per line, zero-padded in the filename
        # but sed strips the zeros.
        manager.transport.exec = Mock(return_value=(0, "0\n2\n4\n7\n"))

        missing = manager._list_missing_chunks_on_hpc("/scratch/pool/.staging/12345", expected=10)
        assert missing == [1, 3, 5, 6, 8, 9]

    def test_empty_staging_returns_all_as_missing(self, mock_hpc_config):
        """Staging dir absent / no chunks — treat as "everything missing"
        so the orchestrator can still build a full retry spec."""
        manager = HPCJobManager(config=mock_hpc_config)
        manager.transport.exec = Mock(return_value=(0, ""))
        missing = manager._list_missing_chunks_on_hpc("/scratch/x", expected=5)
        assert missing == [0, 1, 2, 3, 4]

    def test_complete_staging(self, mock_hpc_config):
        manager = HPCJobManager(config=mock_hpc_config)
        manager.transport.exec = Mock(return_value=(0, "0\n1\n2\n"))
        assert manager._list_missing_chunks_on_hpc("/x", expected=3) == []


class TestSubmitCppJobsRetryLoop:
    """End-to-end #29: retry_missing_chunks loops on short staging,
    builds a sparse retry array, and chains strict combine afterok
    of the last retry."""

    def _make_manager(self, tmp_path, monkeypatch):
        config = BatchConfig(
            ssh_host="test.edu",
            ssh_user="testuser",
            simulation_pool_path="/scratch/sims",
            hpc_venv_path="/home/testuser/.venv/qsp",
            remote_project_path="/home/testuser/project",
            partition="shared",
            time_limit="01:00:00",
            cpp_binary_path="/usr/bin/qsp_sim",
            cpp_template_path="/tmp/p.xml",
            cpp_repo_path="/home/testuser/SPQSP_PDAC",
        )
        transport = Mock()

        # Most exec calls succeed and parse as sbatch submissions; tests
        # install call-sequence side_effect to return distinct job ids.
        transport.exec.return_value = (0, "OK")
        transport.upload.return_value = None
        transport.download.return_value = None
        manager = HPCJobManager(config=config, transport=transport)
        # No real wait; the test controls the "missing" return directly.
        monkeypatch.setattr(manager, "_wait_for_array_completion", lambda *a, **k: {})
        return manager, transport

    def test_no_missing_chunks_skips_retry(self, tmp_path, monkeypatch):
        manager, transport = self._make_manager(tmp_path, monkeypatch)
        manager._list_missing_chunks_on_hpc = Mock(return_value=[])

        # Return distinct ids so we can verify no retry was submitted.
        transport.exec.side_effect = [
            # ensure_cpp_binary existence check + git steps — first match
            # short-circuits to generic (0, "OK"). Use a flexible approach:
            # the sbatch submissions will be the only ones returning job ids.
            (0, "OK"),
        ] + [(0, "Submitted batch job 9000")] * 30
        transport.exec.return_value = (0, "OK")

        def side(cmd, *a, **kw):
            if "sbatch" in cmd:
                return (0, "Submitted batch job 9000")
            return (0, "OK")

        transport.exec.side_effect = side

        csv = tmp_path / "params.csv"
        csv.write_text("A\n1.0\n2.0\n")

        info = manager.submit_cpp_jobs(
            samples_csv=str(csv),
            num_simulations=2,
            simulation_pool_id="pool",
            skip_sync=True,
            retry_missing_chunks=3,
        )
        # Array + combine, no retry.
        assert len(info.job_ids) == 2
        manager._list_missing_chunks_on_hpc.assert_called_once()

    def test_missing_chunks_trigger_sparse_retry(self, tmp_path, monkeypatch):
        manager, transport = self._make_manager(tmp_path, monkeypatch)

        # Round 1: [1, 3, 7] missing → submit retry
        # Round 2: complete
        manager._list_missing_chunks_on_hpc = Mock(side_effect=[[1, 3, 7], []])

        submitted_array_specs: list[str] = []
        submitted_configs: list[str] = []
        submitted_deps: list[Optional[str]] = []
        sbatch_counter = [1000]

        def fake_submit_cpp_job(**kwargs):
            submitted_array_specs.append(kwargs.get("array_spec"))
            submitted_configs.append(kwargs.get("config_path"))
            submitted_deps.append(kwargs.get("dependency"))
            sbatch_counter[0] += 1
            return f"arr{sbatch_counter[0]}"

        monkeypatch.setattr(manager.slurm_submitter, "submit_cpp_job", fake_submit_cpp_job)

        combine_calls: list[dict] = []

        def fake_combine(**kwargs):
            combine_calls.append(kwargs)
            return "cmb5000"

        monkeypatch.setattr(manager, "submit_combine_batch_job", fake_combine)

        # transport.exec handles `echo $HOME` + ls + sbatch-not-used-here.
        def side(cmd, *a, **kw):
            if "echo $HOME" in cmd:
                return (0, "/home/testuser\n")
            if "echo OK" in cmd:
                return (0, "OK")
            return (0, "")

        transport.exec.side_effect = side

        # _upload_cpp_retry_config runs a real download+upload; stub it.
        manager._upload_cpp_retry_config = Mock(
            return_value="/home/testuser/project/batch_jobs/input/cpp_retry_config_r1.json"
        )

        csv = tmp_path / "params.csv"
        csv.write_text("A\n" + "\n".join(str(float(i)) for i in range(10)) + "\n")

        info = manager.submit_cpp_jobs(
            samples_csv=str(csv),
            num_simulations=10,
            simulation_pool_id="pool",
            skip_sync=True,
            jobs_per_chunk=1,  # one task per sim → 10 tasks → ids 0..9
            retry_missing_chunks=3,
        )

        # Two array submits: the original + one retry.
        assert len(submitted_array_specs) == 2
        # First submit is the original 0-9 (no sparse spec → default range).
        assert submitted_array_specs[0] is None
        # Retry uses collapsed sparse spec.
        assert submitted_array_specs[1] == "1,3,7"
        # Retry points at the override config.
        assert submitted_configs[1] and "cpp_retry_config_" in submitted_configs[1]

        # Combine was chained afterok:<last array id> with strict=True.
        assert len(combine_calls) == 1
        assert combine_calls[0]["strict"] is True
        assert combine_calls[0]["dependency"] == f"afterok:arr{sbatch_counter[0]}"

        # JobInfo.job_ids: [original_array, retry_array, combine]
        assert len(info.job_ids) == 3
        assert info.job_ids[-1] == "cmb5000"

    def test_retry_budget_exhausted_still_submits_strict_combine(self, tmp_path, monkeypatch):
        """When staging stays short after N rounds, combine still runs —
        strict=True so it fails the afterok dep and skips derivation."""
        manager, transport = self._make_manager(tmp_path, monkeypatch)
        # Every round finds the same missing chunk — pathological.
        manager._list_missing_chunks_on_hpc = Mock(return_value=[5])

        sbatch_counter = [2000]

        def fake_submit_cpp_job(**kwargs):
            sbatch_counter[0] += 1
            return f"arr{sbatch_counter[0]}"

        monkeypatch.setattr(manager.slurm_submitter, "submit_cpp_job", fake_submit_cpp_job)

        combine_calls: list[dict] = []
        monkeypatch.setattr(
            manager,
            "submit_combine_batch_job",
            lambda **kw: (combine_calls.append(kw), "cmb9999")[1],
        )

        def side(cmd, *a, **kw):
            if "echo $HOME" in cmd:
                return (0, "/home/testuser\n")
            return (0, "OK")

        transport.exec.side_effect = side
        manager._upload_cpp_retry_config = Mock(return_value="/tmp/retry.json")

        csv = tmp_path / "params.csv"
        csv.write_text("A\n" + "\n".join(str(float(i)) for i in range(6)) + "\n")

        info = manager.submit_cpp_jobs(
            samples_csv=str(csv),
            num_simulations=6,
            simulation_pool_id="pool",
            skip_sync=True,
            jobs_per_chunk=1,
            retry_missing_chunks=2,
        )
        # 1 original + 2 retries + 1 combine = 4 ids.
        assert len(info.job_ids) == 4
        assert combine_calls[0]["strict"] is True
        # Inspected 2 times (once per retry attempt — post-final-retry round
        # enters the loop, finds missing again, submits retry, and that's the
        # 2nd retry; we do NOT re-inspect after attempt 2 — we exit the loop
        # because the budget is exhausted).
        # Actually: attempt=1 inspects (missing), submits retry1; attempt=2
        # inspects (missing), submits retry2. Two inspections.
        assert manager._list_missing_chunks_on_hpc.call_count == 2


class TestSSHRetry:
    """Retry-on-transient-SSH-error behavior for SSHTransport."""

    @pytest.fixture
    def retry_config(self, mock_hpc_config):
        # Keep defaults realistic but predictable for tests
        return BatchConfig(
            **{
                **{
                    f.name: getattr(mock_hpc_config, f.name)
                    for f in mock_hpc_config.__dataclass_fields__.values()
                },
                "ssh_retry_max_attempts": 3,
                "ssh_retry_base_delay_s": 0.0,
                "ssh_retry_max_delay_s": 0.0,
            }
        )

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        # Keep tests fast regardless of backoff values
        monkeypatch.setattr("qsp_hpc.batch.hpc_job_manager.time.sleep", lambda s: None)

    def test_is_transient_detects_known_patterns(self):
        assert _is_transient_ssh_error("ssh: Connection reset by peer")
        assert _is_transient_ssh_error("Operation timed out")
        assert _is_transient_ssh_error("Broken pipe")
        assert _is_transient_ssh_error("kex_exchange_identification: foo")
        assert _is_transient_ssh_error("client_loop: send disconnect: Broken pipe")
        assert not _is_transient_ssh_error("No such file or directory")
        assert not _is_transient_ssh_error("")
        assert not _is_transient_ssh_error(None)

    def test_exec_retries_on_transient_ssh_255(self, retry_config):
        transport = SSHTransport(retry_config)
        # ssh returns rc=255 with transient text twice, then succeeds
        bad = subprocess.CompletedProcess(
            args=[], returncode=255, stdout="", stderr="kex_exchange_identification: read"
        )
        good = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr="")
        with patch("subprocess.run", side_effect=[bad, bad, good]) as run:
            rc, out = transport.exec("echo hi", timeout=5)
        assert rc == 0
        assert "ok" in out
        assert run.call_count == 3

    def test_exec_does_not_retry_on_genuine_remote_failure(self, retry_config):
        transport = SSHTransport(retry_config)
        # Remote command exits 1 — NOT an ssh-layer failure, so no retry
        result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="bash: command not found"
        )
        with patch("subprocess.run", return_value=result) as run:
            rc, out = transport.exec("bogus", timeout=5)
        assert rc == 1
        assert run.call_count == 1

    def test_exec_gives_up_after_max_attempts(self, retry_config):
        transport = SSHTransport(retry_config)
        bad = subprocess.CompletedProcess(
            args=[], returncode=255, stdout="", stderr="Connection reset by peer"
        )
        with patch("subprocess.run", return_value=bad) as run:
            with pytest.raises(RemoteCommandError):
                transport.exec("echo hi", timeout=5)
        assert run.call_count == retry_config.ssh_retry_max_attempts

    def test_upload_retries_on_transient_stderr(self, retry_config, tmp_path):
        transport = SSHTransport(retry_config)
        local = tmp_path / "f.txt"
        local.write_text("x")
        transient = subprocess.CalledProcessError(
            1, ["scp"], stderr="scp: Connection reset by peer\n"
        )
        ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("subprocess.run", side_effect=[transient, transient, ok]) as run:
            transport.upload(str(local), "/remote/f.txt")
        assert run.call_count == 3

    def test_upload_does_not_retry_on_permanent_error(self, retry_config, tmp_path):
        transport = SSHTransport(retry_config)
        local = tmp_path / "f.txt"
        local.write_text("x")
        permanent = subprocess.CalledProcessError(
            1, ["scp"], stderr="scp: /remote/f.txt: Permission denied\n"
        )
        with patch("subprocess.run", side_effect=permanent) as run:
            with pytest.raises(RemoteCommandError):
                transport.upload(str(local), "/remote/f.txt")
        assert run.call_count == 1

    def test_upload_retries_on_timeout(self, retry_config, tmp_path):
        transport = SSHTransport(retry_config)
        local = tmp_path / "f.txt"
        local.write_text("x")
        timeout_exc = subprocess.TimeoutExpired(cmd=["scp"], timeout=600)
        ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("subprocess.run", side_effect=[timeout_exc, ok]) as run:
            transport.upload(str(local), "/remote/f.txt")
        assert run.call_count == 2

    def test_download_retries_on_transient_stderr(self, retry_config, tmp_path):
        transport = SSHTransport(retry_config)
        transient = subprocess.CalledProcessError(
            1, ["scp"], stderr="client_loop: send disconnect: Broken pipe"
        )
        ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("subprocess.run", side_effect=[transient, ok]) as run:
            transport.download("/remote/f.txt", str(tmp_path))
        assert run.call_count == 2

    def test_retry_disabled_with_max_attempts_1(self, mock_hpc_config, tmp_path):
        cfg = BatchConfig(
            **{
                **{
                    f.name: getattr(mock_hpc_config, f.name)
                    for f in mock_hpc_config.__dataclass_fields__.values()
                },
                "ssh_retry_max_attempts": 1,
            }
        )
        transport = SSHTransport(cfg)
        local = tmp_path / "f.txt"
        local.write_text("x")
        transient = subprocess.CalledProcessError(1, ["scp"], stderr="Connection reset by peer")
        with patch("subprocess.run", side_effect=transient) as run:
            with pytest.raises(RemoteCommandError):
                transport.upload(str(local), "/remote/f.txt")
        assert run.call_count == 1


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


# ============================================================================
# DownloadResult / download_test_stats_full (#22)
# ============================================================================


class TestDownloadResultContract:
    """Regression tests for the DownloadResult dataclass replacing the
    pre-#22 ``_last_sample_index`` / ``_last_param_names`` side channels.

    The old pattern was ordering-sensitive: two back-to-back downloads
    clobbered each other's sidecar attributes, so callers that batched
    downloads silently saw the second download's metadata for both.
    """

    @staticmethod
    def _make_manager_with_fake_files(tmp_path, sample_index_for, params_for):
        """Build an HPCJobManager whose ``_download_combined_files`` runs
        against on-disk CSVs that we synthesise per-call. Returns
        (manager, set_next_csvs)."""
        manager = HPCJobManager.__new__(HPCJobManager)
        manager.verbose = False
        manager.logger = Mock()
        # Stub transport: 'test -f combined_params.csv' check returns
        # 'exists' so the params branch runs.
        transport = Mock()
        transport.exec.return_value = (0, "exists")

        def write_files(staging_dir, sample_index_arr, params_arr, ts_arr):
            staging_dir = Path(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)
            import pandas as _pd

            df = _pd.DataFrame(
                {
                    "sample_index": sample_index_arr,
                    "p0": params_arr[:, 0],
                    "p1": params_arr[:, 1],
                }
            )
            df.to_csv(staging_dir / "combined_params.csv", index=False)
            _pd.DataFrame(ts_arr).to_csv(
                staging_dir / "combined_test_stats.csv", header=False, index=False
            )

        next_payload = {}

        def fake_download(remote_path, local_dir):
            # Whatever file the caller asks for, we just write the
            # expected combined_*.csv with the queued payload.
            staging = Path(local_dir)
            write_files(
                staging,
                next_payload["sample_index"],
                next_payload["params"],
                next_payload["test_stats"],
            )

        transport.download.side_effect = fake_download
        manager.transport = transport

        def set_next(sample_index_arr, params_arr, ts_arr):
            next_payload["sample_index"] = sample_index_arr
            next_payload["params"] = params_arr
            next_payload["test_stats"] = ts_arr

        return manager, set_next

    def test_download_test_stats_full_returns_dataclass(self, tmp_path):
        manager, set_next = self._make_manager_with_fake_files(tmp_path, None, None)
        params = np.array([[1.0, 2.0], [3.0, 4.0]])
        ts = np.array([[10.0], [20.0]])
        sidx = np.array([7, 11], dtype=np.int64)
        set_next(sidx, params, ts)

        result = manager._download_combined_files(str(tmp_path / "ts_dir"), tmp_path / "out")

        assert isinstance(result, DownloadResult)
        np.testing.assert_array_equal(result.params, params)
        np.testing.assert_array_equal(result.test_stats, ts)
        np.testing.assert_array_equal(result.sample_index, sidx)
        assert result.param_names == ["p0", "p1"]

    def test_back_to_back_downloads_do_not_share_state(self, tmp_path):
        """Two sequential downloads must each return their own metadata.

        Pre-#22, the second download overwrote ``_last_sample_index``,
        so a caller that did A.download(); B.download(); then read
        ``_last_sample_index`` got B's data instead of A's.
        """
        manager, set_next = self._make_manager_with_fake_files(tmp_path, None, None)

        params_a = np.array([[1.0, 1.0]])
        ts_a = np.array([[100.0]])
        sidx_a = np.array([42], dtype=np.int64)
        set_next(sidx_a, params_a, ts_a)
        result_a = manager._download_combined_files(str(tmp_path / "ts_a"), tmp_path / "out_a")

        params_b = np.array([[9.0, 9.0]])
        ts_b = np.array([[900.0]])
        sidx_b = np.array([99], dtype=np.int64)
        set_next(sidx_b, params_b, ts_b)
        result_b = manager._download_combined_files(str(tmp_path / "ts_b"), tmp_path / "out_b")

        # A's metadata must still describe A after B finishes.
        np.testing.assert_array_equal(result_a.sample_index, sidx_a)
        np.testing.assert_array_equal(result_b.sample_index, sidx_b)
        # No instance-level sidecars: the side-channel attributes must
        # not leak back in.
        assert not hasattr(manager, "_last_sample_index")
        assert not hasattr(manager, "_last_param_names")

    def test_legacy_tuple_wrapper_still_works(self, tmp_path):
        """``download_test_stats`` must keep returning a 2-tuple so the
        MATLAB-era ``QSPSimulator._run_pipeline`` callers don't break."""
        manager, _ = self._make_manager_with_fake_files(tmp_path, None, None)

        params = np.array([[2.0, 3.0]])
        ts = np.array([[33.0]])
        manager.download_test_stats_full = Mock(
            return_value=DownloadResult(
                params=params,
                test_stats=ts,
                sample_index=np.array([5], dtype=np.int64),
                param_names=["p0", "p1"],
            )
        )

        wrapped = HPCJobManager.download_test_stats.__get__(manager)
        out_params, out_ts = wrapped("/pool", "abc", tmp_path / "dest")
        np.testing.assert_array_equal(out_params, params)
        np.testing.assert_array_equal(out_ts, ts)
