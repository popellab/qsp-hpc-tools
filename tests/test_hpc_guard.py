"""Tests for the QSP_HPC_NO_SUBMIT kill switch (qsp_hpc.utils.hpc_guard)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from qsp_hpc.batch.hpc_file_transfer import HPCFileTransfer
from qsp_hpc.batch.hpc_job_manager import BatchConfig
from qsp_hpc.utils.hpc_guard import (
    NO_SUBMIT_ENV,
    HPCSubmitBlockedError,
    ensure_remote_writes_allowed,
    no_submit_enabled,
)


class DummyTransport(SimpleNamespace):
    def exec(self, *_args, **_kwargs):
        return 0, ""

    def upload(self, *_args, **_kwargs):
        return None


@pytest.fixture
def base_config() -> BatchConfig:
    return BatchConfig(
        ssh_host="host",
        ssh_user="user",
        simulation_pool_path="/pool",
        hpc_venv_path="/venv",
        remote_project_path="/remote/base",
    )


# --- unit: env parsing -------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " on "])
def test_no_submit_enabled_truthy(monkeypatch, value):
    monkeypatch.setenv(NO_SUBMIT_ENV, value)
    assert no_submit_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  "])
def test_no_submit_enabled_falsey(monkeypatch, value):
    monkeypatch.setenv(NO_SUBMIT_ENV, value)
    assert no_submit_enabled() is False


def test_no_submit_disabled_when_unset(monkeypatch):
    monkeypatch.delenv(NO_SUBMIT_ENV, raising=False)
    assert no_submit_enabled() is False


# --- unit: ensure_remote_writes_allowed -------------------------------------


def test_ensure_raises_when_set(monkeypatch):
    monkeypatch.setenv(NO_SUBMIT_ENV, "1")
    with pytest.raises(HPCSubmitBlockedError, match=r"submit_job.*QSP_HPC_NO_SUBMIT"):
        ensure_remote_writes_allowed("submit_job")


def test_ensure_passes_when_unset(monkeypatch):
    monkeypatch.delenv(NO_SUBMIT_ENV, raising=False)
    # Should not raise.
    ensure_remote_writes_allowed("submit_job")


# --- integration: sync_codebase blocks BEFORE touching rsync ----------------


def test_sync_codebase_blocked_before_rsync(monkeypatch, base_config, tmp_path):
    """The guard must fire before any subprocess.run, so no rsync --delete runs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(NO_SUBMIT_ENV, "1")

    called = {"run": False}

    def fake_run(*_a, **_k):  # pragma: no cover - must never be reached
        called["run"] = True
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    transfer = HPCFileTransfer(base_config, DummyTransport())
    with pytest.raises(HPCSubmitBlockedError):
        transfer.sync_codebase()

    assert called["run"] is False, "rsync must not run when the kill switch is set"


def test_sync_codebase_skip_sync_short_circuits_even_when_blocked(
    monkeypatch, base_config, tmp_path
):
    """skip_sync=True returns before the guard — explicit opt-out, no error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(NO_SUBMIT_ENV, "1")

    transfer = HPCFileTransfer(base_config, DummyTransport())
    # Should not raise: skip_sync is an explicit no-op.
    transfer.sync_codebase(skip_sync=True)
