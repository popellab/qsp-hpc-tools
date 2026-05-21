"""Unit tests for :class:`MultiScenarioRunner`.

The runner is a thin orchestrator over :class:`CppSimulator` — most
behavior under test is "did we wire the right kwargs through, did we
fail-fast on misalignment, did we plan + submit one fused array for the
uncached scenarios (#90 Phase 2), does the joint-NaN mask intersect
right." End-to-end ``run_all`` against a live SLURM array is exercised
in the integration smoke (``workflows/sbi_runner``).
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
    existing_depth=0,
):
    """A MagicMock-backed CppSimulator surrogate carrying just the attrs
    the runner reads off it.

    ``existing_depth`` is what :meth:`CppSimulator.hpc_existing_depth`
    returns — 0 (the default) means fully uncached, so the scenario
    joins the fused array; ``>= n`` means already satisfied and excluded.
    """
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
    # Phase 2 fused planning probe.
    sim.hpc_existing_depth.return_value = existing_depth
    sim._compute_test_stats_hash.return_value = f"tshash_{pool_id}"
    # Theta is regenerated locally from the deterministic pool during the
    # fused teardown — return an array row-aligned with the indices.
    sim._generate_parameters.side_effect = lambda idx: np.zeros((len(idx), 1))
    return sim


def _fake_jm(simulation_pool_path="/scratch/pools"):
    jm = MagicMock()
    jm.config.simulation_pool_path = simulation_pool_path

    # Fused teardown: one combine+download for every non-cached scenario,
    # returns {name: (sample_index, test_stats)}.
    def _fused_dl(scen_specs, local_dest):
        return {s["name"]: (np.array([0], dtype=np.int64), np.zeros((1, 1))) for s in scen_specs}

    jm.download_test_stats_fused.side_effect = _fused_dl
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


def _wire_samples(jm, sim, tmp_path):
    """Common run_all plumbing: a deterministic shared samples CSV."""
    csv = tmp_path / "samples.csv"
    csv.write_text("sample_index,k\n0,1\n")
    sim._write_params_csv.return_value = csv
    jm.upload_shared_samples_csv.return_value = "/remote/shared.csv"


class TestRunAll:
    def test_prepare_session_runs_once_then_fused_submit_skips_setup(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        b = _fake_sim(job_manager=jm, pool_id="pool_b")
        _wire_samples(jm, a, tmp_path)

        r = MultiScenarioRunner({"a": a, "b": b})
        results = r.run_all(1)

        # Session setup runs exactly once before the fused submit.
        assert jm.ensure_hpc_venv.call_count == 1
        assert jm.ensure_cpp_binary.call_count == 1
        # The fused array is submitted with skip_setup=True (setup already ran).
        assert jm.submit_cpp_fused_jobs.call_args.kwargs["skip_setup"] is True

        assert set(results) == {"a", "b"}
        for name, res in results.items():
            assert isinstance(res, ScenarioResult)
            assert res.pool_id == f"pool_{name}"

    def test_shared_healthy_state_uploaded_once_and_threaded(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a", healthy_state_yaml="hs.yaml")
        b = _fake_sim(job_manager=jm, pool_id="pool_b", healthy_state_yaml="hs.yaml")
        _wire_samples(jm, a, tmp_path)
        jm.upload_shared_healthy_state.return_value = "/remote/healthy.yaml"

        MultiScenarioRunner({"a": a, "b": b}).run_all(1)

        # Healthy state uploaded once; threaded into the fused submit.
        assert jm.upload_shared_healthy_state.call_count == 1
        fused_kw = jm.submit_cpp_fused_jobs.call_args.kwargs
        assert fused_kw["healthy_state_yaml_remote"] == "/remote/healthy.yaml"

    def test_mixed_healthy_state_falls_back_to_per_pool_upload(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a", healthy_state_yaml="hs_a.yaml")
        b = _fake_sim(job_manager=jm, pool_id="pool_b", healthy_state_yaml="hs_b.yaml")
        _wire_samples(jm, a, tmp_path)

        MultiScenarioRunner({"a": a, "b": b}).run_all(1)

        # Shared upload skipped because YAMLs disagree.
        assert jm.upload_shared_healthy_state.call_count == 0
        assert jm.submit_cpp_fused_jobs.call_args.kwargs["healthy_state_yaml_remote"] is None

    def test_prepare_session_idempotent(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        _wire_samples(jm, a, tmp_path)
        a.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        r = MultiScenarioRunner({"a": a})
        r.prepare_session()
        r.prepare_session()
        r.run_all(1)
        # ensure_* should still be called exactly once total.
        assert jm.ensure_hpc_venv.call_count == 1
        assert jm.ensure_cpp_binary.call_count == 1

    def test_sample_index_from_fused_download(self, tmp_path):
        """A fused scenario's sample_index comes from the download's
        sidecar; theta is then regenerated locally for exactly those
        indices."""
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        _wire_samples(jm, a, tmp_path)
        jm.download_test_stats_fused.side_effect = lambda specs, dest: {
            specs[0]["name"]: (np.array([5, 7], dtype=np.int64), np.zeros((2, 1)))
        }

        results = MultiScenarioRunner({"a": a}).run_all(2)
        np.testing.assert_array_equal(results["a"].sample_index, [5, 7])
        a._generate_parameters.assert_called_once()
        np.testing.assert_array_equal(a._generate_parameters.call_args.args[0], [5, 7])

    def test_local_cache_sample_index_falls_back_to_arange(self, tmp_path):
        """A locally-cached scenario whose run_hpc leaves last_sample_index
        unset falls back to a positional arange."""
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        a.local_cache_satisfies.return_value = True  # local read path
        _wire_samples(jm, a, tmp_path)
        a.run_hpc.return_value = (np.zeros((3, 1)), np.zeros((3, 1)))
        a.last_sample_index = None

        results = MultiScenarioRunner({"a": a}).run_all(3)
        np.testing.assert_array_equal(results["a"].sample_index, [0, 1, 2])


class TestAllLocalFastPath:
    """run_all skips HPC session prep + uploads + the fused submit when
    every scenario is locally cached."""

    def test_all_local_skips_prepare_session_and_fused_submit(self, tmp_path):
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        b = _fake_sim(job_manager=jm, pool_id="pool_b")
        a.local_cache_satisfies.return_value = True
        b.local_cache_satisfies.return_value = True
        a.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        b.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        r = MultiScenarioRunner({"a": a, "b": b})
        results = r.run_all(1)

        # No HPC session prep, no shared uploads, no fused array.
        assert jm.ensure_hpc_venv.call_count == 0
        assert jm.ensure_cpp_binary.call_count == 0
        assert jm.upload_shared_samples_csv.call_count == 0
        assert jm.upload_shared_healthy_state.call_count == 0
        assert jm.submit_cpp_fused_jobs.call_count == 0
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
        a.local_cache_satisfies.return_value = True  # locally cached
        b.local_cache_satisfies.return_value = False  # one miss → full prep
        _wire_samples(jm, a, tmp_path)
        a.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        b.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        r = MultiScenarioRunner({"a": a, "b": b})
        r.run_all(1)

        # A single uncached scenario forces the session prep + shared upload.
        assert jm.ensure_hpc_venv.call_count == 1
        assert jm.upload_shared_samples_csv.call_count == 1
        assert a.run_hpc.call_args.kwargs["samples_csv_remote"] == "/remote/shared.csv"


class TestFusedSubmission:
    """#90 Phase 2: run_all collapses N per-scenario arrays into ONE
    fused array spanning the union of the scenarios' deficits."""

    def test_cold_run_submits_one_fused_array_for_all_scenarios(self, tmp_path):
        jm = _fake_jm()
        sims = {
            n: _fake_sim(job_manager=jm, pool_id=f"pool_{n}", existing_depth=0)
            for n in ("a", "b", "c")
        }
        _wire_samples(jm, next(iter(sims.values())), tmp_path)
        for s in sims.values():
            s.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        MultiScenarioRunner(sims).run_all(20000)

        # Exactly one fused submit — not one array per scenario.
        jm.submit_cpp_fused_jobs.assert_called_once()
        kw = jm.submit_cpp_fused_jobs.call_args.kwargs
        # Cold: every scenario starts at 0, fused range is the full [0, n).
        assert kw["num_simulations"] == 20000
        assert kw["samples_start_offset"] == 0
        assert {s["name"] for s in kw["scenarios"]} == {"a", "b", "c"}
        for entry in kw["scenarios"]:
            assert entry["samples_start_offset"] == 0
        assert kw["skip_setup"] is True

    def test_per_scenario_deficits_span_union(self, tmp_path):
        """Scenarios at different pool depths → the fused array spans
        [min_depth, n); each scenario carries its own start offset."""
        jm = _fake_jm()
        baseline = _fake_sim(job_manager=jm, pool_id="pool_base", existing_depth=20000)
        gvax = _fake_sim(job_manager=jm, pool_id="pool_gvax", existing_depth=5000)
        gvax_nivo = _fake_sim(job_manager=jm, pool_id="pool_gn", existing_depth=0)
        _wire_samples(jm, baseline, tmp_path)
        for s in (baseline, gvax, gvax_nivo):
            s.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        sims = {"baseline": baseline, "gvax": gvax, "gvax_nivo": gvax_nivo}
        MultiScenarioRunner(sims).run_all(20000)

        kw = jm.submit_cpp_fused_jobs.call_args.kwargs
        # baseline (20000/20000) is already satisfied — excluded.
        assert {s["name"] for s in kw["scenarios"]} == {"gvax", "gvax_nivo"}
        # Fused range = union of deficits = [min(5000, 0), 20000) = [0, 20000).
        assert kw["samples_start_offset"] == 0
        assert kw["num_simulations"] == 20000
        offsets = {s["name"]: s["samples_start_offset"] for s in kw["scenarios"]}
        assert offsets == {"gvax": 5000, "gvax_nivo": 0}

    def test_fused_offset_is_min_when_all_partial(self, tmp_path):
        """When every fused scenario is partway done, the fused array
        starts at the shallowest depth, not 0."""
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a", existing_depth=8000)
        b = _fake_sim(job_manager=jm, pool_id="pool_b", existing_depth=12000)
        _wire_samples(jm, a, tmp_path)
        for s in (a, b):
            s.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        MultiScenarioRunner({"a": a, "b": b}).run_all(20000)

        kw = jm.submit_cpp_fused_jobs.call_args.kwargs
        assert kw["samples_start_offset"] == 8000
        assert kw["num_simulations"] == 12000  # n - min_offset
        offsets = {s["name"]: s["samples_start_offset"] for s in kw["scenarios"]}
        assert offsets == {"a": 8000, "b": 12000}

    def test_all_scenarios_satisfied_skips_fused_submit(self, tmp_path):
        """Every scenario already has n test stats on HPC (but not local,
        so _all_local is False) → no fused array; results come back via
        one fused combine+download."""
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a", existing_depth=100)
        b = _fake_sim(job_manager=jm, pool_id="pool_b", existing_depth=100)
        _wire_samples(jm, a, tmp_path)

        MultiScenarioRunner({"a": a, "b": b}).run_all(100)

        assert jm.submit_cpp_fused_jobs.call_count == 0
        # Still one fused combine+download covering both scenarios.
        jm.download_test_stats_fused.assert_called_once()
        names = {s["name"] for s in jm.download_test_stats_fused.call_args.args[0]}
        assert names == {"a", "b"}

    def test_fused_array_waited_on_before_download(self, tmp_path):
        """The fused array is waited on (via the first simulator's
        _wait_for_jobs) before the per-scenario download loop."""
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a", existing_depth=0)
        b = _fake_sim(job_manager=jm, pool_id="pool_b", existing_depth=0)
        _wire_samples(jm, a, tmp_path)
        for s in (a, b):
            s.run_hpc.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        jm.submit_cpp_fused_jobs.return_value = MagicMock(job_ids=["9001"])

        MultiScenarioRunner({"a": a, "b": b}).run_all(10)

        # First simulator's _wait_for_jobs got the fused array's job ids.
        a._wait_for_jobs.assert_called_once_with(["9001"])

    def test_fused_submit_has_no_evolve_pack_kwargs(self, tmp_path):
        """The evolve-pack emit/consume plumbing is retired — the fused
        submit must not carry evolve_pack_key / evolve_pack_mode (#90)."""
        jm = _fake_jm()
        sims = {n: _fake_sim(job_manager=jm, pool_id=f"pool_{n}") for n in ("a", "b")}
        _wire_samples(jm, next(iter(sims.values())), tmp_path)

        MultiScenarioRunner(sims).run_all(1)

        kwargs = jm.submit_cpp_fused_jobs.call_args.kwargs
        assert "evolve_pack_key" not in kwargs
        assert "evolve_pack_mode" not in kwargs

    def test_use_evolve_packs_kwarg_removed(self):
        """The use_evolve_packs constructor kwarg no longer exists (#90)."""
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        with pytest.raises(TypeError, match="use_evolve_packs"):
            MultiScenarioRunner({"a": a}, use_evolve_packs=False)


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
