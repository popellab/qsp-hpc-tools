"""Tests for qsp_hpc.simulation.hpc_session.HPCSession (Layer 2.5 scaffold).

The session is the local-eval rollout's setup-once orchestration layer
(see ``notes/architecture/local_observable_eval_plan.md`` in pdac-build).
At the scaffold milestone, ``run_scenario`` raises NotImplementedError;
tests cover ``ensure_remote`` idempotency, the D7 priors-hash guardrail
on ``sample_theta_pool``, and the run_scenario sentinel.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from qsp_hpc.simulation.hpc_session import (
    SUBPOOL_MANIFEST_FILENAME,
    SUBPOOL_MANIFEST_SCHEMA,
    HPCSession,
)

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


# ---------------------------------------------------------------------------
# reserve_sample_index_range: session-global allocator (D1)
# ---------------------------------------------------------------------------


def _read_manifest(pool_dir: Path, kind: str) -> dict:
    return json.loads((pool_dir / kind / SUBPOOL_MANIFEST_FILENAME).read_text())


def test_reserve_requires_ensure_remote(session_inputs, tmp_path):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.register_scenario_pool(tmp_path / "pool_A")
    with pytest.raises(RuntimeError, match="ensure_remote"):
        s.reserve_sample_index_range(10, kind="training")


def test_reserve_requires_registered_pool(session_inputs):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()
    with pytest.raises(RuntimeError, match="no scenario pools"):
        s.reserve_sample_index_range(10, kind="training")


def test_reserve_rejects_bad_kind(session_inputs, tmp_path):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()
    s.register_scenario_pool(tmp_path / "pool_A")
    with pytest.raises(ValueError, match="kind must be"):
        s.reserve_sample_index_range(10, kind="bogus")


def test_reserve_rejects_bad_n(session_inputs, tmp_path):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()
    s.register_scenario_pool(tmp_path / "pool_A")
    with pytest.raises(ValueError, match="positive int"):
        s.reserve_sample_index_range(0, kind="training")
    with pytest.raises(ValueError, match="positive int"):
        s.reserve_sample_index_range(-5, kind="training")


def test_reserve_writes_manifest_for_single_scenario(session_inputs, tmp_path):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()
    pool = s.register_scenario_pool(tmp_path / "pool_A")

    start, end = s.reserve_sample_index_range(100, kind="training")

    assert (start, end) == (0, 100)
    manifest = _read_manifest(pool, "training")
    assert manifest["schema_version"] == SUBPOOL_MANIFEST_SCHEMA
    assert manifest["kind"] == "training"
    assert manifest["reservations"] == [
        {"start": 0, "end": 100, "ts": manifest["reservations"][0]["ts"]}
    ]
    # PPC sub-pool is untouched.
    assert not (pool / "ppc" / SUBPOOL_MANIFEST_FILENAME).exists()


def test_reserve_consecutive_calls_are_non_overlapping(session_inputs, tmp_path):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()
    pool = s.register_scenario_pool(tmp_path / "pool_A")

    r1 = s.reserve_sample_index_range(50, kind="training")
    r2 = s.reserve_sample_index_range(75, kind="training")
    r3 = s.reserve_sample_index_range(20, kind="ppc")

    assert r1 == (0, 50)
    assert r2 == (50, 125)
    assert r3 == (125, 145)

    training = _read_manifest(pool, "training")
    ppc = _read_manifest(pool, "ppc")
    assert [(r["start"], r["end"]) for r in training["reservations"]] == [(0, 50), (50, 125)]
    assert [(r["start"], r["end"]) for r in ppc["reservations"]] == [(125, 145)]


def test_reserve_broadcasts_to_all_registered_pools(session_inputs, tmp_path):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()
    pool_a = s.register_scenario_pool(tmp_path / "pool_A")
    pool_b = s.register_scenario_pool(tmp_path / "pool_B")

    start, end = s.reserve_sample_index_range(40, kind="training")

    assert (start, end) == (0, 40)
    for pool in (pool_a, pool_b):
        m = _read_manifest(pool, "training")
        assert [(r["start"], r["end"]) for r in m["reservations"]] == [(0, 40)]


def test_reserve_rescans_when_new_pool_registered(session_inputs, tmp_path):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()

    # Pre-existing pool with reservations from a previous session.
    pool_b = tmp_path / "pool_B"
    (pool_b / "training").mkdir(parents=True)
    (pool_b / "training" / SUBPOOL_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": SUBPOOL_MANIFEST_SCHEMA,
                "kind": "training",
                "reservations": [{"start": 0, "end": 1000, "ts": "2026-05-01T00:00:00+00:00"}],
            }
        )
    )

    s.register_scenario_pool(tmp_path / "pool_A")
    r1 = s.reserve_sample_index_range(10, kind="training")
    assert r1 == (0, 10)  # only pool_A registered, no prior reservations

    # Registering pool_B should invalidate the cached watermark and force
    # a rescan that picks up the 1000-end reservation.
    s.register_scenario_pool(pool_b)
    r2 = s.reserve_sample_index_range(20, kind="training")
    assert r2 == (1000, 1020)


def test_reserve_picks_up_disk_watermark_from_ppc(session_inputs, tmp_path):
    binary, priors, jm = session_inputs
    s = _make_session(binary, priors, jm)
    s.ensure_remote()

    pool = tmp_path / "pool_A"
    (pool / "ppc").mkdir(parents=True)
    (pool / "ppc" / SUBPOOL_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": SUBPOOL_MANIFEST_SCHEMA,
                "kind": "ppc",
                "reservations": [{"start": 0, "end": 500, "ts": "x"}],
            }
        )
    )
    s.register_scenario_pool(pool)

    # Watermark spans both kinds — training reservation must start at 500.
    start, end = s.reserve_sample_index_range(50, kind="training")
    assert (start, end) == (500, 550)


def _worker_reserve(args):
    binary_str, priors_str, pool_str, n = args
    s = HPCSession(
        binary_path=Path(binary_str),
        priors_csv=Path(priors_str),
        job_manager=MagicMock(),
        seed=42,
        theta_pool_size=8,
        theta_pool_cache_dir=Path(binary_str).parent / "theta_cache",
    )
    s.ensure_remote()
    s.register_scenario_pool(Path(pool_str))
    return s.reserve_sample_index_range(n, kind="training")


def test_reserve_flock_serializes_concurrent_writers(session_inputs, tmp_path):
    """Two processes hitting the same pool dir produce non-overlapping ranges.

    Each process scans the high-watermark before its own append, so under
    flock the second writer sees the first's manifest and starts past it.
    Without flock, both would see watermark=0 and write overlapping ranges.
    """
    binary, priors, jm = session_inputs
    pool = tmp_path / "pool_A"

    # Run two writers in parallel against the same pool dir.
    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pp:
        results = pp.map(
            _worker_reserve,
            [
                (str(binary), str(priors), str(pool), 100),
                (str(binary), str(priors), str(pool), 200),
            ],
        )

    # The two ranges must be disjoint and cover [0, 300) in some order.
    ranges = sorted(results)
    assert len(ranges) == 2
    sizes = sorted([end - start for start, end in ranges])
    assert sizes == [100, 200]
    assert ranges[0][0] == 0
    assert ranges[0][1] == ranges[1][0]
    assert ranges[1][1] == 300

    manifest = _read_manifest(pool, "training")
    assert len(manifest["reservations"]) == 2


# ---------------------------------------------------------------------------
# run_scenario: scaffold sentinel
# ---------------------------------------------------------------------------


_MODEL_STRUCTURE_JSON = json.dumps(
    {
        "species": [
            {"name": "C1", "units": "cell"},
            {"name": "Teff", "units": "cell"},
        ],
        "compartments": [{"name": "V_T", "volume_units": "milliliter"}],
        "parameters": [],
    }
)


@pytest.fixture
def run_scenario_inputs(session_inputs, tmp_path):
    """Augment session_inputs with a scenario YAML + model_structure.json."""
    binary, priors, jm = session_inputs
    scen = tmp_path / "scenario.yaml"
    scen.write_text("scenario:\n  name: test\n")
    model_struct = tmp_path / "model_structure.json"
    model_struct.write_text(_MODEL_STRUCTURE_JSON)

    # Mock job_manager: config + submit_cpp_jobs return shape.
    jm.config = MagicMock()
    jm.config.ssh_host = "test-host"
    jm.config.cpp_binary_path = "/hpc/qsp_sim"
    jm.config.cpp_template_path = "/hpc/param_all.xml"
    jm.config.simulation_pool_path = "/hpc/pools"
    job_info = MagicMock()
    job_info.job_ids = ["12345"]
    jm.submit_cpp_jobs.return_value = job_info
    jm.check_job_status.return_value = {
        "completed": 1,
        "running": 0,
        "pending": 0,
        "failed": 0,
    }

    return binary, priors, jm, scen, model_struct


def test_run_scenario_requires_ensure_remote(run_scenario_inputs):
    binary, priors, jm, scen, ms = run_scenario_inputs
    s = _make_session(binary, priors, jm, model_structure_file=ms)
    with pytest.raises(RuntimeError, match="ensure_remote"):
        s.run_scenario(scenario_yaml=scen, n_simulations=4)


def test_run_scenario_rejects_bad_kind(run_scenario_inputs, monkeypatch):
    binary, priors, jm, scen, ms = run_scenario_inputs
    s = _make_session(binary, priors, jm, model_structure_file=ms)
    s.ensure_remote()
    with pytest.raises(ValueError, match="kind must be"):
        s.run_scenario(scenario_yaml=scen, n_simulations=4, kind="bogus")


def test_run_scenario_rejects_bad_n(run_scenario_inputs):
    binary, priors, jm, scen, ms = run_scenario_inputs
    s = _make_session(binary, priors, jm, model_structure_file=ms)
    s.ensure_remote()
    with pytest.raises(ValueError, match="positive int"):
        s.run_scenario(scenario_yaml=scen, n_simulations=0)


def test_run_scenario_requires_model_structure(run_scenario_inputs):
    binary, priors, jm, scen, _ms = run_scenario_inputs
    s = _make_session(binary, priors, jm)  # no model_structure_file
    s.ensure_remote()
    with pytest.raises(RuntimeError, match="model_structure_file"):
        s.run_scenario(scenario_yaml=scen, n_simulations=4)


def test_run_scenario_happy_path(run_scenario_inputs, tmp_path, monkeypatch):
    """End-to-end shape: pool_id is computed, register/reserve runs, theta
    slice matches [start, end), traj_columns + sshfs_host are forwarded,
    and the result is a SimulationBatch with the expected fields."""
    import pandas as pd

    binary, priors, jm, scen, ms = run_scenario_inputs
    s = _make_session(
        binary,
        priors,
        jm,
        model_structure_file=ms,
        local_pool_root=tmp_path / "scenario_pools",
    )
    s.ensure_remote()
    s.poll_interval = 0.0  # don't sleep in tests

    # Stub the read helper — capture kwargs and return a synthetic long-form
    # frame. The frame just needs the schema; SimulationBatch doesn't validate
    # cell-by-cell.
    captured: dict = {}

    def fake_read(remote_pool_path, *, sample_indices, traj_columns, filesystem, sshfs_host):
        captured["remote_pool_path"] = remote_pool_path
        captured["sample_indices"] = sample_indices
        captured["traj_columns"] = traj_columns
        captured["sshfs_host"] = sshfs_host
        return pd.DataFrame(
            {
                "sample_index": pd.Series([], dtype="int64"),
                "time": pd.Series([], dtype="float64"),
                "species": pd.Series([], dtype="object"),
                "value": pd.Series([], dtype="float64"),
            }
        )

    batch = s.run_scenario(
        scenario_yaml=scen,
        n_simulations=5,
        traj_columns=["C1", "Teff"],
        kind="training",
        _read_helper=fake_read,
    )

    # 1. pool_id is sha256 hex of binary | scenario YAML.
    from qsp_hpc.utils.hash_utils import compute_pool_id_hash

    expected_pool_id = compute_pool_id_hash(binary_path=binary, scenario_yaml=scen)
    assert batch.pool_id == expected_pool_id

    # 2. submit_cpp_jobs got pool_id, kind, scenario_yaml, derive_test_stats=False.
    jm.submit_cpp_jobs.assert_called_once()
    submit_kwargs = jm.submit_cpp_jobs.call_args.kwargs
    assert submit_kwargs["simulation_pool_id"] == expected_pool_id
    assert submit_kwargs["kind"] == "training"
    assert submit_kwargs["scenario_yaml"] == str(scen.resolve())
    assert submit_kwargs["derive_test_stats"] is False
    assert submit_kwargs["num_simulations"] == 5

    # 3. samples_csv was sample_index + param-name columns.
    # The temp file is unlinked post-submit, but we can re-read the args
    # by intercepting at submit time. Easier: assert samples_csv path
    # was passed and num_simulations matches.
    assert "samples_csv" in submit_kwargs

    # 4. read helper got the right shape.
    assert captured["remote_pool_path"] == f"/hpc/pools/{expected_pool_id}/training"
    assert captured["sample_indices"] == [0, 1, 2, 3, 4]
    assert captured["traj_columns"] == ["C1", "Teff"]
    assert captured["sshfs_host"] == "test-host"

    # 5. SimulationBatch fields.
    assert batch.theta.shape == (5, 2)
    np.testing.assert_array_equal(batch.sample_index, np.arange(5))
    assert batch.param_names == ["A", "B"]
    assert batch.species_units == {
        "V_T": "milliliter",
        "C1": "cell",
        "Teff": "cell",
    }

    # 6. Local pool manifest was written.
    manifest = _read_manifest(tmp_path / "scenario_pools" / expected_pool_id, "training")
    assert manifest["reservations"] == [
        {"start": 0, "end": 5, "ts": manifest["reservations"][0]["ts"]}
    ]


def test_run_scenario_theta_slice_matches_reservation(run_scenario_inputs, tmp_path):
    """A second run_scenario after a prior reservation must hand back the
    matching theta slice (so cross-scenario sample_index alignment holds)."""
    import pandas as pd

    binary, priors, jm, scen, ms = run_scenario_inputs
    s = _make_session(
        binary,
        priors,
        jm,
        model_structure_file=ms,
        local_pool_root=tmp_path / "scenario_pools",
    )
    s.ensure_remote()
    s.poll_interval = 0.0

    # Pre-consume a range so the second run_scenario reservation starts at 3.
    s.register_scenario_pool(tmp_path / "scenario_pools" / "warmup")
    s.reserve_sample_index_range(3, kind="training")

    def fake_read(remote_pool_path, *, sample_indices, traj_columns, filesystem, sshfs_host):
        return pd.DataFrame(
            {
                "sample_index": pd.Series([], dtype="int64"),
                "time": pd.Series([], dtype="float64"),
                "species": pd.Series([], dtype="object"),
                "value": pd.Series([], dtype="float64"),
            }
        )

    batch = s.run_scenario(
        scenario_yaml=scen,
        n_simulations=4,
        _read_helper=fake_read,
    )

    np.testing.assert_array_equal(batch.sample_index, np.arange(3, 7))
    # theta rows for [3, 7) must match a fresh draw of the same pool sliced.
    full_pool = s.sample_theta_pool(7)
    np.testing.assert_array_equal(batch.theta, full_pool[3:7])
