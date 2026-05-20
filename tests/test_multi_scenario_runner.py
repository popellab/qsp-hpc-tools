"""Unit tests for :class:`MultiScenarioRunner`.

The runner is a thin orchestrator over :class:`CppSimulator` — most
behavior under test is "did we wire the right kwargs through, did we
fail-fast on misalignment, does the joint-NaN mask intersect right."
End-to-end ``run_all`` is exercised in the integration smoke
(``workflows/sbi_runner`` driving a real SLURM array).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from qsp_hpc.simulation.multi_scenario_runner import (
    MultiScenarioRunner,
    ScenarioResult,
)


def _fake_sim(
    *,
    priors_csv="priors.csv",
    seed=2025,
    submodel_priors_yaml=None,
    binary_path="qsp_sim",
    pool_id="poolXYZ",
    job_manager=None,
    healthy_state_yaml="healthy.yaml",
):
    """A MagicMock-backed CppSimulator surrogate carrying just the attrs
    the runner reads off it."""
    sim = MagicMock()
    sim.priors_csv = priors_csv
    sim.seed = seed
    sim.submodel_priors_yaml = submodel_priors_yaml
    sim.binary_path = binary_path
    sim.simulation_pool_id = pool_id
    sim.job_manager = job_manager
    sim.healthy_state_yaml = healthy_state_yaml
    sim.last_sample_index = None
    # Default to a local-cache miss so tests exercise the full HPC path
    # unless they explicitly opt into the all-local fast path.
    sim.local_cache_satisfies.return_value = False
    return sim


def _fake_jm(simulation_pool_path="/scratch/pools"):
    jm = MagicMock()
    jm.config.simulation_pool_path = simulation_pool_path
    return jm


class TestConstructorValidation:
    def test_empty_simulators_rejected(self):
        with pytest.raises(ValueError, match="simulators cannot be empty"):
            MultiScenarioRunner({})

    def test_no_job_manager_anywhere_rejected(self):
        sim = _fake_sim(job_manager=None)
        with pytest.raises(ValueError, match="needs a job_manager"):
            MultiScenarioRunner({"a": sim})

    def test_explicit_job_manager_accepted(self):
        sim = _fake_sim(job_manager=None)
        jm = _fake_jm()
        r = MultiScenarioRunner({"a": sim}, job_manager=jm)
        assert r.job_manager is jm

    def test_priors_mismatch_rejected(self):
        jm = _fake_jm()
        a = _fake_sim(priors_csv="priors_a.csv", job_manager=jm)
        b = _fake_sim(priors_csv="priors_b.csv", job_manager=jm)
        with pytest.raises(ValueError, match="priors_csv"):
            MultiScenarioRunner({"a": a, "b": b})

    def test_seed_mismatch_rejected(self):
        jm = _fake_jm()
        a = _fake_sim(seed=2025, job_manager=jm)
        b = _fake_sim(seed=2026, job_manager=jm)
        with pytest.raises(ValueError, match="seed"):
            MultiScenarioRunner({"a": a, "b": b})

    def test_binary_mismatch_rejected(self):
        jm = _fake_jm()
        a = _fake_sim(binary_path="bin_a", job_manager=jm)
        b = _fake_sim(binary_path="bin_b", job_manager=jm)
        with pytest.raises(ValueError, match="binary_path"):
            MultiScenarioRunner({"a": a, "b": b})

    def test_aligned_simulators_accepted(self):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        b = _fake_sim(job_manager=jm, pool_id="pool_b")
        r = MultiScenarioRunner({"a": a, "b": b})
        assert list(r.simulators) == ["a", "b"]


class TestUploadSharedSamplesCsv:
    def test_idempotent_within_instance(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm)
        # _write_params_csv → temp file with deterministic content
        csv = tmp_path / "samples.csv"
        csv.write_text("sample_index,k1\n0,1.0\n1,2.0\n")
        a._write_params_csv.return_value = csv
        jm.upload_shared_samples_csv.return_value = "/remote/samples_shared_xxxx.csv"

        r = MultiScenarioRunner({"a": a})
        p1 = r.upload_shared_samples_csv(2)
        p2 = r.upload_shared_samples_csv(2)
        assert p1 == p2
        # Underlying upload called only once.
        assert jm.upload_shared_samples_csv.call_count == 1

    def test_filename_is_content_hash(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm)
        csv = tmp_path / "samples.csv"
        csv.write_text("sample_index,k1\n0,1.0\n")
        a._write_params_csv.return_value = csv
        jm.upload_shared_samples_csv.return_value = "/remote/whatever.csv"

        r = MultiScenarioRunner({"a": a})
        r.upload_shared_samples_csv(1)
        # Inspect the filename argument to upload_shared_samples_csv.
        _, remote_filename = jm.upload_shared_samples_csv.call_args.args
        assert remote_filename.startswith("samples_shared_")
        assert remote_filename.endswith(".csv")
        # 12-char hash → "samples_shared_" + 12 hex + ".csv" = 31 chars
        assert len(remote_filename) == len("samples_shared_") + 12 + len(".csv")


class TestRunAll:
    def test_prepare_session_runs_once_then_all_scenarios_skip_setup(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        b = _fake_sim(job_manager=jm, pool_id="pool_b")
        csv = tmp_path / "samples.csv"
        csv.write_text("sample_index,k\n0,1\n")
        a._write_params_csv.return_value = csv
        jm.upload_shared_samples_csv.return_value = "/remote/shared.csv"

        a.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        b.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        r = MultiScenarioRunner({"a": a, "b": b})
        results = r.run_all(1)

        # Session setup runs exactly once before any scenario.
        assert jm.ensure_hpc_venv.call_count == 1
        assert jm.ensure_cpp_binary.call_count == 1

        # Both scenarios get skip_setup=True (setup already done above).
        a_kw = a.run_hpc.call_args.kwargs
        b_kw = b.run_hpc.call_args.kwargs
        assert a_kw["skip_setup"] is True
        assert b_kw["skip_setup"] is True
        assert a_kw["samples_csv_remote"] == "/remote/shared.csv"
        assert b_kw["samples_csv_remote"] == "/remote/shared.csv"

        assert set(results) == {"a", "b"}
        for name, res in results.items():
            assert isinstance(res, ScenarioResult)
            assert res.pool_id == f"pool_{name}"

    def test_shared_healthy_state_uploaded_once_and_threaded(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a", healthy_state_yaml="hs.yaml")
        b = _fake_sim(job_manager=jm, pool_id="pool_b", healthy_state_yaml="hs.yaml")
        csv = tmp_path / "samples.csv"
        csv.write_text("sample_index,k\n0,1\n")
        a._write_params_csv.return_value = csv
        jm.upload_shared_samples_csv.return_value = "/remote/shared.csv"
        jm.upload_shared_healthy_state.return_value = "/remote/healthy.yaml"
        a.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        b.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        r = MultiScenarioRunner({"a": a, "b": b})
        r.run_all(1)

        # Healthy state uploaded once, shared remote threaded into both run_hpc calls.
        assert jm.upload_shared_healthy_state.call_count == 1
        assert a.run_hpc.call_args.kwargs["healthy_state_yaml_remote"] == "/remote/healthy.yaml"
        assert b.run_hpc.call_args.kwargs["healthy_state_yaml_remote"] == "/remote/healthy.yaml"

    def test_mixed_healthy_state_falls_back_to_per_pool_upload(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a", healthy_state_yaml="hs_a.yaml")
        b = _fake_sim(job_manager=jm, pool_id="pool_b", healthy_state_yaml="hs_b.yaml")
        csv = tmp_path / "samples.csv"
        csv.write_text("sample_index,k\n0,1\n")
        a._write_params_csv.return_value = csv
        jm.upload_shared_samples_csv.return_value = "/remote/shared.csv"
        a.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        b.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        r = MultiScenarioRunner({"a": a, "b": b})
        r.run_all(1)

        # Shared upload skipped because YAMLs disagree.
        assert jm.upload_shared_healthy_state.call_count == 0
        assert a.run_hpc.call_args.kwargs["healthy_state_yaml_remote"] is None
        assert b.run_hpc.call_args.kwargs["healthy_state_yaml_remote"] is None

    def test_prepare_session_idempotent(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        csv = tmp_path / "samples.csv"
        csv.write_text("sample_index,k\n0,1\n")
        a._write_params_csv.return_value = csv
        jm.upload_shared_samples_csv.return_value = "/remote/shared.csv"
        a.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        r = MultiScenarioRunner({"a": a})
        r.prepare_session()
        r.prepare_session()
        r.run_all(1)
        # ensure_* should still be called exactly once total.
        assert jm.ensure_hpc_venv.call_count == 1
        assert jm.ensure_cpp_binary.call_count == 1

    def test_sample_index_pulled_from_simulator(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        csv = tmp_path / "samples.csv"
        csv.write_text("sample_index,k\n5,1\n7,2\n")
        a._write_params_csv.return_value = csv
        jm.upload_shared_samples_csv.return_value = "/remote/shared.csv"
        a.run_hpc.return_value = (np.zeros((2, 1)), np.zeros((2, 1)))
        a.last_sample_index = np.array([5, 7], dtype=np.int64)

        r = MultiScenarioRunner({"a": a})
        results = r.run_all(2)
        np.testing.assert_array_equal(results["a"].sample_index, [5, 7])

    def test_sample_index_falls_back_to_arange_when_missing(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        csv = tmp_path / "samples.csv"
        csv.write_text("sample_index,k\n0,1\n")
        a._write_params_csv.return_value = csv
        jm.upload_shared_samples_csv.return_value = "/remote/shared.csv"
        a.run_hpc.return_value = (np.zeros((3, 1)), np.zeros((3, 1)))
        a.last_sample_index = None

        r = MultiScenarioRunner({"a": a})
        results = r.run_all(3)
        np.testing.assert_array_equal(results["a"].sample_index, [0, 1, 2])


class TestAllLocalFastPath:
    """run_all skips HPC session prep + uploads when every scenario is
    locally cached."""

    def test_all_local_skips_prepare_session_and_uploads(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        b = _fake_sim(job_manager=jm, pool_id="pool_b")
        a.local_cache_satisfies.return_value = True
        b.local_cache_satisfies.return_value = True
        a.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        b.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        r = MultiScenarioRunner({"a": a, "b": b})
        results = r.run_all(1)

        # No HPC session prep, no shared uploads.
        assert jm.ensure_hpc_venv.call_count == 0
        assert jm.ensure_cpp_binary.call_count == 0
        assert jm.upload_shared_samples_csv.call_count == 0
        assert jm.upload_shared_healthy_state.call_count == 0
        # run_hpc still called per scenario (Tier 1 returns locally) with
        # all remotes None.
        for sim in (a, b):
            kw = sim.run_hpc.call_args.kwargs
            assert kw["samples_csv_remote"] is None
            assert kw["healthy_state_yaml_remote"] is None
            assert kw["scenario_yaml_remote"] is None
        assert set(results) == {"a", "b"}

    def test_one_scenario_uncached_forces_full_hpc_preamble(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        b = _fake_sim(job_manager=jm, pool_id="pool_b")
        a.local_cache_satisfies.return_value = True
        b.local_cache_satisfies.return_value = False  # one miss → full prep
        csv = tmp_path / "samples.csv"
        csv.write_text("sample_index,k\n0,1\n")
        a._write_params_csv.return_value = csv
        jm.upload_shared_samples_csv.return_value = "/remote/shared.csv"
        a.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        b.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        r = MultiScenarioRunner({"a": a, "b": b})
        r.run_all(1)

        # A single uncached scenario forces the session prep + shared upload.
        assert jm.ensure_hpc_venv.call_count == 1
        assert jm.upload_shared_samples_csv.call_count == 1
        assert a.run_hpc.call_args.kwargs["samples_csv_remote"] == "/remote/shared.csv"


class TestJointNanMask:
    def _result(self, x, sample_index):
        return ScenarioResult(
            theta=np.zeros((len(sample_index), 1)),
            x=np.asarray(x, dtype=float),
            sample_index=np.asarray(sample_index, dtype=np.int64),
            pool_id="p",
            pool_path="/p",
        )

    def test_all_finite_passes(self):
        a = self._result([[1.0], [2.0], [3.0]], [10, 11, 12])
        b = self._result([[100.0], [200.0], [300.0]], [10, 11, 12])
        masks = MultiScenarioRunner.joint_nan_mask({"a": a, "b": b})
        assert masks["a"].all() and masks["b"].all()

    def test_nan_in_one_scenario_drops_shared_row(self):
        a = self._result([[1.0], [np.nan], [3.0]], [10, 11, 12])
        b = self._result([[100.0], [200.0], [300.0]], [10, 11, 12])
        masks = MultiScenarioRunner.joint_nan_mask({"a": a, "b": b})
        np.testing.assert_array_equal(masks["a"], [True, False, True])
        np.testing.assert_array_equal(masks["b"], [True, False, True])

    def test_intersection_only_keeps_shared_indices(self):
        # a has [10, 11, 12]; b has [11, 12, 13] → shared = {11, 12}.
        a = self._result([[1.0], [2.0], [3.0]], [10, 11, 12])
        b = self._result([[200.0], [300.0], [400.0]], [11, 12, 13])
        masks = MultiScenarioRunner.joint_nan_mask({"a": a, "b": b})
        # Only sample_indices 11 and 12 are shared and finite.
        np.testing.assert_array_equal(masks["a"], [False, True, True])
        np.testing.assert_array_equal(masks["b"], [True, True, False])

    def test_inf_drops_row(self):
        a = self._result([[1.0], [np.inf]], [10, 11])
        b = self._result([[100.0], [200.0]], [10, 11])
        masks = MultiScenarioRunner.joint_nan_mask({"a": a, "b": b})
        np.testing.assert_array_equal(masks["a"], [True, False])
        np.testing.assert_array_equal(masks["b"], [True, False])
