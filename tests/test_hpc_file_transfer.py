import subprocess
from types import SimpleNamespace

import pytest

from qsp_hpc.batch.hpc_file_transfer import HPCFileTransfer
from qsp_hpc.batch.hpc_job_manager import BatchConfig


class DummyTransport(SimpleNamespace):
    """Minimal transport stub for tests."""

    def exec(self, *_args, **_kwargs):
        return 0, ""

    def upload(self, *_args, **_kwargs):
        return None


@pytest.fixture
def base_config(tmp_path) -> BatchConfig:
    return BatchConfig(
        ssh_host="host",
        ssh_user="user",
        simulation_pool_path="/pool",
        hpc_venv_path="/venv",
        remote_project_path="/remote/base",
    )


def test_sync_codebase_raises_on_rsync_failure(monkeypatch, base_config, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_run(_cmd, capture_output, text):
        return subprocess.CompletedProcess(args=_cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    transfer = HPCFileTransfer(base_config, DummyTransport())

    with pytest.raises(RuntimeError, match="rsync failed .*boom"):
        transfer.sync_codebase()


def test_setup_remote_directories_requires_remote_root(base_config):
    bad_config = base_config.__class__(
        ssh_host=base_config.ssh_host,
        ssh_user=base_config.ssh_user,
        simulation_pool_path=base_config.simulation_pool_path,
        hpc_venv_path=base_config.hpc_venv_path,
        remote_project_path="",
    )

    transfer = HPCFileTransfer(bad_config, DummyTransport())

    with pytest.raises(ValueError, match="remote_project_path must be set"):
        transfer.setup_remote_directories()


def test_sync_codebase_handles_ssh_config_alias_with_empty_user(monkeypatch, tmp_path):
    """Test that empty ssh_user works with SSH config aliases (no @ symbol)."""
    monkeypatch.chdir(tmp_path)

    # Config with empty user (SSH config alias)
    config = BatchConfig(
        ssh_host="hpc",
        ssh_user="",  # Empty user - should use SSH config
        ssh_key="",
        simulation_pool_path="/pool",
        hpc_venv_path="/venv",
        remote_project_path="/remote/base",
    )

    captured_cmd = []

    def fake_run(cmd, capture_output, text):
        captured_cmd.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    transfer = HPCFileTransfer(config, DummyTransport())
    transfer.sync_codebase()

    # Verify remote target doesn't have @ symbol
    assert len(captured_cmd) == 1
    rsync_cmd = captured_cmd[0]
    remote_target = rsync_cmd[-1]

    # Should be "hpc:/remote/base" not "@hpc:/remote/base"
    assert remote_target == "hpc:/remote/base"
    assert "@" not in remote_target


def test_sync_codebase_handles_explicit_user(monkeypatch, tmp_path):
    """Test that explicit ssh_user includes user@host format."""
    monkeypatch.chdir(tmp_path)

    # Config with explicit user
    config = BatchConfig(
        ssh_host="cluster.edu",
        ssh_user="jeliaso2",
        ssh_key="",
        simulation_pool_path="/pool",
        hpc_venv_path="/venv",
        remote_project_path="/home/jeliaso2/projects",
    )

    captured_cmd = []

    def fake_run(cmd, capture_output, text):
        captured_cmd.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    transfer = HPCFileTransfer(config, DummyTransport())
    transfer.sync_codebase()

    # Verify remote target has user@host format
    assert len(captured_cmd) == 1
    rsync_cmd = captured_cmd[0]
    remote_target = rsync_cmd[-1]

    # Should be "jeliaso2@cluster.edu:/home/jeliaso2/projects"
    assert remote_target == "jeliaso2@cluster.edu:/home/jeliaso2/projects"
    assert "@" in remote_target


def test_sync_codebase_quotes_ssh_key_path(monkeypatch, tmp_path):
    """Test that SSH key path is properly quoted to handle spaces."""
    monkeypatch.chdir(tmp_path)

    # Config with SSH key containing spaces
    config = BatchConfig(
        ssh_host="hpc",
        ssh_user="",
        ssh_key="/path with spaces/id_rsa",
        simulation_pool_path="/pool",
        hpc_venv_path="/venv",
        remote_project_path="/remote/base",
    )

    captured_cmd = []

    def fake_run(cmd, capture_output, text):
        captured_cmd.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    transfer = HPCFileTransfer(config, DummyTransport())
    transfer.sync_codebase()

    # Verify SSH key is quoted in -e argument
    assert len(captured_cmd) == 1
    rsync_cmd = captured_cmd[0]

    # Find the -e flag
    e_flag_index = rsync_cmd.index("-e")
    ssh_command = rsync_cmd[e_flag_index + 1]

    # Should be: 'ssh -i "/path with spaces/id_rsa"'
    assert '"/path with spaces/id_rsa"' in ssh_command
    assert ssh_command.startswith('ssh -i "')


class TestUploadJobConfig:
    """Tests for job config upload including sim_config and dosing."""

    @pytest.fixture
    def transfer(self, base_config, tmp_path):
        """Create HPCFileTransfer with capture transport."""
        transport = DummyTransport()
        transport.uploaded_files = {}

        def capture_upload(local_path, remote_path):
            import json

            with open(local_path, "r") as f:
                transport.uploaded_files[remote_path] = json.load(f)

        transport.upload = capture_upload
        return HPCFileTransfer(base_config, transport)

    def test_upload_job_config_includes_sim_config(self, transfer, tmp_path):
        """Test that sim_config is included in job config JSON.

        Regression test for: HPC simulations running 30 days instead of 90 days
        because sim_config wasn't being passed from scenario YAML to job config.
        """
        sim_config = {
            "start_time": 0,
            "stop_time": 90,
            "time_units": "day",
            "solver": "sundials",
            "abs_tolerance": 1e-9,
            "rel_tolerance": 1e-6,
        }

        transfer.upload_job_config(
            test_stats_csv="test.csv",
            model_script="test_model",
            num_simulations=100,
            seed=42,
            jobs_per_chunk=10,
            sim_config=sim_config,
        )

        # Verify job_config.json was uploaded
        uploaded = transfer.transport.uploaded_files
        assert len(uploaded) == 1
        remote_path, config = list(uploaded.items())[0]
        assert "job_config.json" in remote_path

        # Verify sim_config is in job config
        assert "sim_config" in config
        assert config["sim_config"]["stop_time"] == 90
        assert config["sim_config"]["solver"] == "sundials"

    def test_upload_job_config_includes_dosing(self, transfer, tmp_path):
        """Test that dosing config is included in job config JSON.

        Regression test: Treatment scenarios require dosing config to be
        passed to MATLAB batch_worker for correct drug administration.
        """
        dosing = {
            "drugs": ["GVAX", "anti_PD1"],
            "schedule": {
                "GVAX": [{"time": 0, "dose": 1e8}],
                "anti_PD1": [{"time": 14, "dose": 3}],
            },
        }

        transfer.upload_job_config(
            test_stats_csv="test.csv",
            model_script="test_model",
            num_simulations=100,
            seed=42,
            jobs_per_chunk=10,
            dosing=dosing,
        )

        # Verify dosing is in job config
        uploaded = transfer.transport.uploaded_files
        config = list(uploaded.values())[0]

        assert "dosing" in config
        assert config["dosing"]["drugs"] == ["GVAX", "anti_PD1"]
        assert "schedule" in config["dosing"]

    def test_upload_job_config_includes_both_sim_config_and_dosing(self, transfer, tmp_path):
        """Test that both sim_config and dosing are included when provided."""
        sim_config = {"start_time": 0, "stop_time": 90, "time_units": "day"}
        dosing = {"drugs": ["drug1"], "schedule": {}}

        transfer.upload_job_config(
            test_stats_csv="test.csv",
            model_script="test_model",
            num_simulations=100,
            seed=42,
            jobs_per_chunk=10,
            sim_config=sim_config,
            dosing=dosing,
        )

        uploaded = transfer.transport.uploaded_files
        config = list(uploaded.values())[0]

        assert "sim_config" in config
        assert "dosing" in config
        assert config["sim_config"]["stop_time"] == 90
        assert config["dosing"]["drugs"] == ["drug1"]

    def test_upload_job_config_without_optional_configs(self, transfer, tmp_path):
        """Test that job config works without sim_config or dosing (backward compat)."""
        transfer.upload_job_config(
            test_stats_csv="test.csv",
            model_script="test_model",
            num_simulations=100,
            seed=42,
            jobs_per_chunk=10,
            # No sim_config or dosing provided
        )

        uploaded = transfer.transport.uploaded_files
        config = list(uploaded.values())[0]

        # Should NOT have sim_config or dosing keys
        assert "sim_config" not in config
        assert "dosing" not in config

        # But should still have required fields
        assert config["n_simulations"] == 100
        assert config["model_script"] == "test_model"
