"""Tests for qsp_hpc.simulation.hpc_session.HPCSession (Layer 2.5 scaffold).

The session is the local-eval rollout's setup-once orchestration layer
(see ``notes/architecture/local_observable_eval_plan.md`` in pdac-build).
At the scaffold milestone, ``run_scenario`` raises NotImplementedError;
tests cover ``ensure_remote`` idempotency, the D7 priors-hash guardrail
on ``sample_theta_pool``, and the run_scenario sentinel.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from qsp_hpc.simulation.hpc_session import HPCSession

PRIORS_CSV = """\
name,distribution,dist_param1,dist_param2
A,lognormal,0.0,0.5
B,lognormal,0.5,0.3
"""


@pytest.fixture
def session_inputs(tmp_path: Path):
    """Tmp binary + priors CSV + a mock job_manager."""
    binary = tmp_path / "qsp_sim"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    priors = tmp_path / "priors.csv"
    priors.write_text(PRIORS_CSV)
    job_manager = MagicMock()
    return binary, priors, job_manager


def _make_session(binary, priors, job_manager, **kwargs):
    return HPCSession(
        binary_path=binary,
        priors_csv=priors,
        job_manager=job_manager,
        seed=42,
        theta_pool_size=8,
        theta_pool_cache_dir=binary.parent / "theta_cache",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_session_rejects_missing_binary(tmp_path):
    priors = tmp_path / "priors.csv"
    priors.write_text(PRIORS_CSV)
    with pytest.raises(FileNotFoundError, match="qsp_sim binary"):
        HPCSession(
            binary_path=tmp_path / "missing",
            priors_csv=priors,
            job_manager=MagicMock(),
        )


def test_session_rejects_missing_priors(tmp_path):
    binary = tmp_path / "qsp_sim"
    binary.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="Priors CSV"):
        HPCSession(
            binary_path=binary,
            priors_csv=tmp_path / "missing.csv",
            job_manager=MagicMock(),
        )


# ---------------------------------------------------------------------------
# ensure_remote: idempotent + records priors hash
# ---------------------------------------------------------------------------


def test_ensure_remote_calls_setup_once(session_inputs):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)

    s.ensure_remote()
    s.ensure_remote()  # second call is a no-op
    s.ensure_remote()

    assert jm.ensure_hpc_venv.call_count == 1
    assert jm.ensure_cpp_binary.call_count == 1


def test_ensure_remote_snapshots_priors_hash(session_inputs):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    assert s.priors_csv_hash is None

    s.ensure_remote()
    assert s.priors_csv_hash is not None
    assert len(s.priors_csv_hash) == 64  # full sha256 hex


def test_ensure_remote_forwards_skip_flags(session_inputs):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm, remote_binary_path="/hpc/qsp_sim")

    s.ensure_remote(skip_git_pull=True, skip_build=True)

    jm.ensure_cpp_binary.assert_called_once_with(
        skip_git_pull=True,
        skip_build=True,
        binary_path="/hpc/qsp_sim",
    )


# ---------------------------------------------------------------------------
# sample_theta_pool: requires ensure_remote, guards against priors drift
# ---------------------------------------------------------------------------


def test_sample_theta_pool_requires_ensure_remote(session_inputs):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    with pytest.raises(RuntimeError, match="ensure_remote"):
        s.sample_theta_pool(4)


def test_sample_theta_pool_returns_deterministic_theta(session_inputs):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()

    theta1 = s.sample_theta_pool(4)
    theta2 = s.sample_theta_pool(4)  # cached, identical

    assert theta1.shape == (4, 2)
    np.testing.assert_array_equal(theta1, theta2)


def test_sample_theta_pool_detects_priors_drift(session_inputs):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()

    # Mid-run prior edit — D7 guardrail must reject.
    priors.write_text(PRIORS_CSV + "C,lognormal,1.0,0.2\n")

    with pytest.raises(RuntimeError, match="priors_csv .* has changed"):
        s.sample_theta_pool(4)


# ---------------------------------------------------------------------------
# run_scenario: scaffold sentinel
# ---------------------------------------------------------------------------


def test_run_scenario_not_implemented(session_inputs, tmp_path):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()

    with pytest.raises(NotImplementedError, match="Layer 4 contract"):
        s.run_scenario(
            scenario_yaml=tmp_path / "scen.yaml",
            n_simulations=10,
        )
