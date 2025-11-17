import subprocess
from pathlib import Path
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
        transfer.setup_remote_directories("proj")
