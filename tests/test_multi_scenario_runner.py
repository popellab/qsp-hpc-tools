"""Unit tests for :class:`MultiScenarioRunner`.

The runner is a thin orchestrator over :class:`CppSimulator` — most
behavior under test is "did we wire the right kwargs through, did we
fail-fast on misalignment, did we plan + submit one fused array for the
uncached scenarios (#90 Phase 2), does the joint-NaN mask intersect
right." End-to-end ``run_all`` against a live SLURM array is exercised
in the integration smoke (``workflows/sbi_runner``).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qsp_hpc.cpp.batch_runner import BatchResult, CppBatchRunner, FusedScenarioSpec
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

    def test_no_job_manager_construction_allowed(self):
        """job_manager is optional at construction — the local PPC path
        (simulate_with_parameters_all) needs no transport."""
        sim = _fake_sim(job_manager=None)
        r = MultiScenarioRunner({"a": sim})
        assert r.job_manager is None

    def test_run_all_without_job_manager_rejected(self):
        """run_all is the HPC path — it raises when no job_manager is
        reachable, pointing the caller at the local PPC method."""
        sim = _fake_sim(job_manager=None)
        r = MultiScenarioRunner({"a": sim})
        with pytest.raises(ValueError, match="needs a job_manager"):
            r.run_all(1)

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

    def test_fused_teardown_persists_local_parquet(self, tmp_path):
        """The fused teardown must write the local Tier-1 parquet the
        single-scenario path writes via ``_download_and_persist``. The #90
        fused refactor dropped this, so ``local_cache_satisfies`` never hit
        after a fused run (every re-run paid an SSH round-trip) and direct
        parquet readers — e.g. the restriction-classifier retrain — found
        nothing on disk."""
        jm = _fake_jm()
        a = _fake_sim(job_manager=jm, pool_id="pool_a")
        a.param_names = ["k"]
        _wire_samples(jm, a, tmp_path)
        jm.download_test_stats_fused.side_effect = lambda specs, dest: {
            specs[0]["name"]: (np.array([0, 1], dtype=np.int64), np.zeros((2, 1)))
        }

        MultiScenarioRunner({"a": a}).run_all(2)

        # Persisted exactly once, keyed by the same hash scen_specs used,
        # with the regenerated theta + downloaded x + sample_index sidecar.
        a._persist_local_test_stats.assert_called_once()
        a._local_test_stats_path.assert_called_once_with("tshash_pool_a")
        call = a._persist_local_test_stats.call_args
        assert call.args[0] is a._local_test_stats_path.return_value
        np.testing.assert_array_equal(call.kwargs["sample_index"], [0, 1])
        assert call.kwargs["param_names"] == ["k"]

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


# ---------------------------------------------------------------------------
# Phase 3: fused local posterior-predictive (simulate_with_parameters_all)
# ---------------------------------------------------------------------------

_PPC_TEMPLATE = b"""<Param>
  <QSP>
    <A>1.0</A>
    <B>2.0</B>
  </QSP>
</Param>
"""

_PPC_PRIORS = """\
name,distribution,dist_param1,dist_param2
A,lognormal,0.0,0.5
B,lognormal,0.5,0.3
"""


def _make_fused_binary(tmp_path: Path) -> tuple[Path, Path]:
    """Fake qsp_sim with fused support.

    ``--dump-state`` writes an opaque evolve-state blob and appends one
    line to a counter file — so a test can assert how many evolves ran.
    A scenario run writes a 2-species v3 trajectory (spA / spB). Returns
    ``(binary, counter_file)``; the counter path is baked into the script
    so it works regardless of subprocess env handling.
    """
    counter = tmp_path / "evolve_calls.log"
    script = tmp_path / "fake_fused_qsp_sim.sh"
    script.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -e
        DUMP_STATE=""
        while [ $# -gt 0 ]; do
          case "$1" in
            --binary-out) BIN_OUT="$2"; shift 2 ;;
            --species-out) SP_OUT="$2"; shift 2 ;;
            --compartments-out) COMP_OUT="$2"; shift 2 ;;
            --rules-out) RULES_OUT="$2"; shift 2 ;;
            --t-end-days) TEND="$2"; shift 2 ;;
            --min-cadence-hours) DT="$2"; shift 2 ;;
            --dump-state) DUMP_STATE="$2"; shift 2 ;;
            *) shift ;;
          esac
        done
        if [ -n "$DUMP_STATE" ]; then
          printf 'FAKE_EVOLVE_STATE' > "$DUMP_STATE"
          echo x >> "{counter}"
          exit 0
        fi
        python3 - <<PY
        import struct
        header = struct.pack('<IIQQQQdddQQ', 0x51535042, 3, 2, 2, 0, 0, float("$DT"), float("$TEND"), 0.0, 0, 0)
        body = struct.pack('<6d', 0.0, 10.0, 20.0, 0.1, 30.0, 40.0)
        open("$BIN_OUT", 'wb').write(header + body)
        open("$SP_OUT", 'w').write("spA\\nspB\\n")
        open("$COMP_OUT", 'w').write('')
        open("$RULES_OUT", 'w').write('')
        PY
    """))
    script.chmod(0o755)
    return script, counter


def _make_cal_targets(tmp_path: Path) -> Path:
    """One calibration target reading spA — minimum the loader accepts."""
    import yaml as _yaml

    cal_dir = tmp_path / "calibration_targets"
    cal_dir.mkdir()
    target = {
        "calibration_target_id": "spA_t0",
        "observable": {
            "code": (
                "def compute_observable(time, species_dict, constants):\n"
                "    return species_dict['spA']\n"
            ),
            "units": "cell",
            "species": ["spA"],
            "constants": [],
        },
        "empirical_data": {
            "median": [10.0],
            "ci95": [[5.0, 20.0]],
            "units": "cell",
            "sample_size": 10,
            "index_values": None,
        },
    }
    (cal_dir / "spA_t0.yaml").write_text(_yaml.dump(target))
    return cal_dir


@pytest.fixture
def ppc_env(tmp_path: Path) -> dict:
    """Shared inputs for fused-PPC tests: a fused-capable fake binary,
    template, priors, calibration targets, healthy state, cache dir."""
    binary, counter = _make_fused_binary(tmp_path)
    template = tmp_path / "template.xml"
    template.write_bytes(_PPC_TEMPLATE)
    priors = tmp_path / "priors.csv"
    priors.write_text(_PPC_PRIORS)
    healthy = tmp_path / "healthy.yaml"
    healthy.write_text("# healthy state\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    return {
        "binary": binary,
        "evolve_counter": counter,
        "template": template,
        "priors": priors,
        "cal_dir": _make_cal_targets(tmp_path),
        "healthy": healthy,
        "cache": cache,
    }


def _ppc_sim(
    env: dict,
    scenario: str,
    *,
    t_end_days: float = 0.2,
    with_healthy: bool = True,
    evolve_trajectory_dir: Path | None = None,
):
    """Build a real CppSimulator for one fused-PPC scenario."""
    from qsp_hpc.simulation.cpp_simulator import CppSimulator

    return CppSimulator(
        priors_csv=env["priors"],
        binary_path=env["binary"],
        template_xml=env["template"],
        cache_dir=env["cache"],
        calibration_targets=env["cal_dir"],
        scenario=scenario,
        healthy_state_yaml=env["healthy"] if with_healthy else None,
        t_end_days=t_end_days,
        min_cadence_hours=0.1,
        evolve_trajectory_dir=evolve_trajectory_dir,
    )


def _fake_run_fused(**kwargs):
    """Stand-in for CppBatchRunner.run_fused: writes one synthetic species
    parquet per FusedScenarioSpec. spA is scaled by scenario position so
    each scenario derives a distinct ts:spA_t0; the evolve is never run."""
    theta = kwargs["theta_matrix"]
    param_names = list(kwargs["param_names"])
    scenarios = kwargs["scenarios"]
    sample_indices = np.asarray(kwargs["sample_indices"], dtype=np.int64)
    n = theta.shape[0]
    out: list[BatchResult] = []
    for scen_idx, spec in enumerate(scenarios):
        cols = {
            "sample_index": pa.array(sample_indices),
            "simulation_id": pa.array(np.arange(n, dtype=np.int64)),
            "status": pa.array(np.zeros(n, dtype=np.int64)),
            "time": pa.array([[0.0, 0.1]] * n, type=pa.list_(pa.float64())),
        }
        for j, name in enumerate(param_names):
            cols[f"param:{name}"] = pa.array(theta[:, j].astype(np.float64))
        # spA row i = (i + 1) * (scenario position + 1); spB constant.
        spa = [
            [float(i + 1) * (scen_idx + 1), float(i + 1) * (scen_idx + 1) + 0.1] for i in range(n)
        ]
        cols["spA"] = pa.array(spa, type=pa.list_(pa.float64()))
        cols["spB"] = pa.array([[2.0, 2.1]] * n, type=pa.list_(pa.float64()))
        Path(spec.output_path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), str(spec.output_path))
        out.append(
            BatchResult(
                parquet_path=Path(spec.output_path),
                n_sims=n,
                n_failed=0,
                species_names=["spA", "spB"],
                n_times=2,
            )
        )
    return out


class TestSimulateWithParametersAll:
    """MultiScenarioRunner.simulate_with_parameters_all — fused local PPC.

    These patch CppBatchRunner.run_fused (the binary fan-out is covered
    by test_batch_runner / test_cpp_batch_worker); the assertions are on
    the orchestration: one fused call, per-scenario cache probe, derive,
    and the {name: (theta_out, table)} contract.
    """

    def test_one_fused_call_for_all_uncached_scenarios(self, ppc_env):
        sims = {n: _ppc_sim(ppc_env, n) for n in ("baseline", "clinical", "gvax")}
        r = MultiScenarioRunner(sims)
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])
        with patch.object(CppBatchRunner, "run_fused", side_effect=_fake_run_fused) as mock_fused:
            results = r.simulate_with_parameters_all(theta)
        # Exactly one fused batch, carrying all three scenarios.
        assert mock_fused.call_count == 1
        specs = mock_fused.call_args.kwargs["scenarios"]
        assert [s.name for s in specs] == ["baseline", "clinical", "gvax"]
        assert all(isinstance(s, FusedScenarioSpec) for s in specs)
        assert all(s.start_index == 0 for s in specs)
        # Contract: {name: (theta_out, table)} in insertion order.
        assert list(results) == ["baseline", "clinical", "gvax"]
        for theta_out, table in results.values():
            np.testing.assert_allclose(theta_out, theta)
            assert "ts:spA_t0" in table.column_names
            assert table.num_rows == 2

    def test_per_scenario_tables_are_distinct(self, ppc_env):
        sims = {n: _ppc_sim(ppc_env, n) for n in ("s0", "s1", "s2")}
        r = MultiScenarioRunner(sims)
        theta = np.array([[0.5, 1.0], [1.5, 2.0], [3.0, 4.0]])
        with patch.object(CppBatchRunner, "run_fused", side_effect=_fake_run_fused):
            results = r.simulate_with_parameters_all(theta)
        # _fake_run_fused scales spA by scenario position → ts:spA_t0 is
        # (row+1) * (position+1).
        np.testing.assert_allclose(results["s0"][1].column("ts:spA_t0").to_numpy(), [1, 2, 3])
        np.testing.assert_allclose(results["s1"][1].column("ts:spA_t0").to_numpy(), [2, 4, 6])
        np.testing.assert_allclose(results["s2"][1].column("ts:spA_t0").to_numpy(), [3, 6, 9])

    def test_second_call_hits_cache_no_fused_run(self, ppc_env):
        sims = {n: _ppc_sim(ppc_env, n) for n in ("a", "b")}
        r = MultiScenarioRunner(sims)
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])
        with patch.object(CppBatchRunner, "run_fused", side_effect=_fake_run_fused) as mock_fused:
            first = r.simulate_with_parameters_all(theta)
            assert mock_fused.call_count == 1
            # Identical theta → every scenario satisfied from its
            # suffix-pool cache; no second fused batch.
            second = r.simulate_with_parameters_all(theta)
            assert mock_fused.call_count == 1
        for name in sims:
            np.testing.assert_allclose(
                first[name][1].column("ts:spA_t0").to_numpy(),
                second[name][1].column("ts:spA_t0").to_numpy(),
            )

    def test_already_cached_scenario_excluded_from_fused_set(self, ppc_env):
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])
        # Prime scenario "a" alone.
        a_only = MultiScenarioRunner({"a": _ppc_sim(ppc_env, "a")})
        with patch.object(CppBatchRunner, "run_fused", side_effect=_fake_run_fused):
            a_only.simulate_with_parameters_all(theta)
        # Now run "a" + "b": "a" is cached, only "b" should be fused.
        r = MultiScenarioRunner({"a": _ppc_sim(ppc_env, "a"), "b": _ppc_sim(ppc_env, "b")})
        with patch.object(CppBatchRunner, "run_fused", side_effect=_fake_run_fused) as mock_fused:
            results = r.simulate_with_parameters_all(theta)
        assert mock_fused.call_count == 1
        specs = mock_fused.call_args.kwargs["scenarios"]
        assert [s.name for s in specs] == ["b"]
        assert set(results) == {"a", "b"}

    def test_all_cached_runs_no_fused_batch(self, ppc_env):
        theta = np.array([[0.5, 1.0]])
        sims = {n: _ppc_sim(ppc_env, n) for n in ("a", "b")}
        r = MultiScenarioRunner(sims)
        with patch.object(CppBatchRunner, "run_fused", side_effect=_fake_run_fused):
            r.simulate_with_parameters_all(theta)
        # Fresh runner, same theta → all cached, run_fused never called.
        r2 = MultiScenarioRunner({n: _ppc_sim(ppc_env, n) for n in ("a", "b")})
        with patch.object(CppBatchRunner, "run_fused", side_effect=_fake_run_fused) as mock_fused:
            results = r2.simulate_with_parameters_all(theta)
        assert mock_fused.call_count == 0
        assert set(results) == {"a", "b"}

    def test_result_primes_single_scenario_cache(self, ppc_env):
        """A fused PPC result is byte-identical-cache-compatible with the
        single-scenario CppSimulator.simulate_with_parameters."""
        sims = {n: _ppc_sim(ppc_env, n) for n in ("a", "b")}
        r = MultiScenarioRunner(sims)
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])
        with patch.object(CppBatchRunner, "run_fused", side_effect=_fake_run_fused):
            fused = r.simulate_with_parameters_all(theta)
        # The single-scenario path on the same theta now hits the cache —
        # its own runner is never invoked.
        sim_a = sims["a"]
        with patch.object(sim_a._runner, "run") as mock_run:
            theta_out, table = sim_a.simulate_with_parameters(theta)
            mock_run.assert_not_called()
        np.testing.assert_allclose(
            table.column("ts:spA_t0").to_numpy(),
            fused["a"][1].column("ts:spA_t0").to_numpy(),
        )

    def test_requires_healthy_state(self, ppc_env):
        sims = {n: _ppc_sim(ppc_env, n, with_healthy=False) for n in ("a", "b")}
        r = MultiScenarioRunner(sims)
        with pytest.raises(ValueError, match="requires healthy_state_yaml"):
            r.simulate_with_parameters_all(np.array([[0.5, 1.0]]))

    def test_rejects_t_end_mismatch(self, ppc_env):
        sims = {
            "a": _ppc_sim(ppc_env, "a", t_end_days=0.2),
            "b": _ppc_sim(ppc_env, "b", t_end_days=0.9),
        }
        r = MultiScenarioRunner(sims)
        with pytest.raises(ValueError, match="t_end_days"):
            r.simulate_with_parameters_all(np.array([[0.5, 1.0]]))

    def test_rejects_evolve_trajectory_dir(self, ppc_env, tmp_path):
        sims = {
            "a": _ppc_sim(ppc_env, "a"),
            "b": _ppc_sim(ppc_env, "b", evolve_trajectory_dir=tmp_path / "traj"),
        }
        r = MultiScenarioRunner(sims)
        with pytest.raises(NotImplementedError, match="evolve_trajectory_dir"):
            r.simulate_with_parameters_all(np.array([[0.5, 1.0]]))


# ---------------------------------------------------------------------------
# Phase 2: fused HPC posterior-predictive
# (simulate_with_parameters_all backend='hpc')
# ---------------------------------------------------------------------------


def _ppc_hpc_jm(results_by_name: dict, *, pool_path: str = "/scratch/pools"):
    """A MagicMock HPCJobManager for the fused-HPC PPC path.

    ``results_by_name`` maps scenario name → ``(sample_index, test_stats)``,
    the pair :meth:`download_test_stats_fused` returns for that scenario.
    Session prep + deferred uploads are no-op MagicMock calls; the submit
    returns a stub JobInfo and the download replays ``results_by_name``.
    """
    jm = MagicMock()
    jm.config.simulation_pool_path = pool_path
    jm.config.cpp_binary_path = "/remote/qsp_sim"
    jm.config.cpp_template_path = "/remote/template.xml"
    jm.upload_shared_samples_csv.return_value = "remote/ppc_samples.csv"
    jm.upload_shared_healthy_state.return_value = "remote/healthy.yaml"
    jm.upload_shared_model_structure.return_value = "remote/model_structure.json"
    jm.upload_shared_test_stats_csv.return_value = "remote/test_stats.csv"
    jm.submit_cpp_fused_jobs.return_value = MagicMock(job_ids=["job1"])

    def _dl(scen_specs, local_dest):
        return {s["name"]: results_by_name[s["name"]] for s in scen_specs}

    jm.download_test_stats_fused.side_effect = _dl
    return jm


def _model_structure_file(tmp_path: Path) -> Path:
    p = tmp_path / "model_structure.json"
    p.write_text("{}\n")
    return p


def _ppc_hpc_sim(env, scenario, jm, model_structure, *, t_end_days: float = 0.2):
    """Real CppSimulator wired for the fused-HPC PPC path: job_manager +
    model_structure + remote binary/template paths."""
    from qsp_hpc.simulation.cpp_simulator import CppSimulator

    return CppSimulator(
        priors_csv=env["priors"],
        binary_path=env["binary"],
        template_xml=env["template"],
        cache_dir=env["cache"],
        calibration_targets=env["cal_dir"],
        scenario=scenario,
        healthy_state_yaml=env["healthy"],
        model_structure_file=model_structure,
        job_manager=jm,
        remote_binary_path="/remote/qsp_sim",
        remote_template_xml="/remote/template.xml",
        t_end_days=t_end_days,
        min_cadence_hours=0.1,
    )


class TestSimulateWithParametersAllHpc:
    """MultiScenarioRunner.simulate_with_parameters_all backend='hpc'.

    The cluster fan-out (submit_cpp_fused_jobs / download_test_stats_fused)
    is mocked; the assertions are on the orchestration: one fused submit
    over the suffix-pool ids, the download reshaped into the same
    (theta_out, table) contract as the local path, suffix-pool caching, and
    the fail-fast guards.
    """

    def test_requires_job_manager(self, ppc_env, tmp_path):
        ms = _model_structure_file(tmp_path)
        # job_manager=None on the sims → MSR has none → hpc backend rejects.
        sims = {n: _ppc_hpc_sim(ppc_env, n, None, ms) for n in ("a", "b")}
        r = MultiScenarioRunner(sims)
        with pytest.raises(RuntimeError, match="requires a\n.*job_manager|job_manager"):
            r.simulate_with_parameters_all(np.array([[0.5, 1.0]]), backend="hpc")

    def test_rejects_prediction_targets(self, ppc_env, tmp_path):
        ms = _model_structure_file(tmp_path)
        jm = _ppc_hpc_jm({})
        sims = {n: _ppc_hpc_sim(ppc_env, n, jm, ms) for n in ("a", "b")}
        r = MultiScenarioRunner(sims)
        with pytest.raises(NotImplementedError, match="prediction_targets"):
            r.simulate_with_parameters_all(
                np.array([[0.5, 1.0]]), backend="hpc", prediction_targets="/tmp/pred"
            )

    def test_rejects_unknown_backend(self, ppc_env, tmp_path):
        ms = _model_structure_file(tmp_path)
        jm = _ppc_hpc_jm({})
        sims = {n: _ppc_hpc_sim(ppc_env, n, jm, ms) for n in ("a", "b")}
        r = MultiScenarioRunner(sims)
        with pytest.raises(ValueError, match="backend must be"):
            r.simulate_with_parameters_all(np.array([[0.5, 1.0]]), backend="cloud")

    def test_one_fused_submit_reshapes_and_caches(self, ppc_env, tmp_path):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        ms = _model_structure_file(tmp_path)
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])
        n = theta.shape[0]
        # One ts column (spA_t0); distinct values per scenario.
        results = {
            "a": (np.arange(n, dtype=np.int64), np.array([[10.0], [11.0]])),
            "b": (np.arange(n, dtype=np.int64), np.array([[20.0], [21.0]])),
        }
        jm = _ppc_hpc_jm(results)
        sims = {nm: _ppc_hpc_sim(ppc_env, nm, jm, ms) for nm in ("a", "b")}
        r = MultiScenarioRunner(sims)
        with patch.object(CppSimulator, "_wait_for_jobs"):
            out = r.simulate_with_parameters_all(theta, backend="hpc")

        # Exactly one fused submit, carrying both scenarios at start_offset 0,
        # keyed by their theta-hashed suffix-pool dirs (not the base pool id).
        assert jm.submit_cpp_fused_jobs.call_count == 1
        scen_arg = jm.submit_cpp_fused_jobs.call_args.kwargs["scenarios"]
        assert [s["name"] for s in scen_arg] == ["a", "b"]
        assert all(s["samples_start_offset"] == 0 for s in scen_arg)
        assert all("posterior_predictive" in s["simulation_pool_id"] for s in scen_arg)
        assert jm.submit_cpp_fused_jobs.call_args.kwargs["num_simulations"] == n

        # Contract: {name: (theta_out, table)}, reshaped like the local path.
        assert list(out) == ["a", "b"]
        for nm in ("a", "b"):
            theta_out, table = out[nm]
            np.testing.assert_allclose(theta_out, theta)
            assert "ts:spA_t0" in table.column_names
            assert table.num_rows == n
        np.testing.assert_allclose(out["a"][1].column("ts:spA_t0").to_numpy(), [10.0, 11.0])
        np.testing.assert_allclose(out["b"][1].column("ts:spA_t0").to_numpy(), [20.0, 21.0])

    def test_reshape_reorders_shuffled_sample_index(self, ppc_env, tmp_path):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        ms = _model_structure_file(tmp_path)
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])
        # Download returns rows out of caller order: sample 1 first, then 0.
        results = {"a": (np.array([1, 0], dtype=np.int64), np.array([[99.0], [10.0]]))}
        jm = _ppc_hpc_jm(results)
        sims = {"a": _ppc_hpc_sim(ppc_env, "a", jm, ms)}
        r = MultiScenarioRunner(sims)
        with patch.object(CppSimulator, "_wait_for_jobs"):
            out = r.simulate_with_parameters_all(theta, backend="hpc")
        # Reordered to caller order: sample 0 → 10.0, sample 1 → 99.0.
        np.testing.assert_allclose(out["a"][1].column("ts:spA_t0").to_numpy(), [10.0, 99.0])

    def test_second_call_hits_cache_no_submit(self, ppc_env, tmp_path):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        ms = _model_structure_file(tmp_path)
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])
        n = theta.shape[0]
        results = {
            nm: (np.arange(n, dtype=np.int64), np.array([[1.0], [2.0]])) for nm in ("a", "b")
        }
        jm = _ppc_hpc_jm(results)
        sims = {nm: _ppc_hpc_sim(ppc_env, nm, jm, ms) for nm in ("a", "b")}
        r = MultiScenarioRunner(sims)
        with patch.object(CppSimulator, "_wait_for_jobs"):
            r.simulate_with_parameters_all(theta, backend="hpc")
            assert jm.submit_cpp_fused_jobs.call_count == 1
            # Identical theta → suffix-pool cache hit, no second submit.
            r.simulate_with_parameters_all(theta, backend="hpc")
            assert jm.submit_cpp_fused_jobs.call_count == 1

    def test_hpc_cache_distinct_from_local(self, ppc_env, tmp_path):
        """The 'hpc' backend tag keys a different suffix pool, so a prior
        local fused run does not satisfy an hpc call (and vice versa)."""
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        ms = _model_structure_file(tmp_path)
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])
        n = theta.shape[0]
        results = {"a": (np.arange(n, dtype=np.int64), np.array([[1.0], [2.0]]))}
        jm = _ppc_hpc_jm(results)
        # Prime the LOCAL cache first.
        local_sim = _ppc_hpc_sim(ppc_env, "a", jm, ms)
        r_local = MultiScenarioRunner({"a": local_sim})
        with patch.object(CppBatchRunner, "run_fused", side_effect=_fake_run_fused):
            r_local.simulate_with_parameters_all(theta, backend="local")
        # HPC call on the same theta must still submit (distinct pool key).
        r_hpc = MultiScenarioRunner({"a": _ppc_hpc_sim(ppc_env, "a", jm, ms)})
        with patch.object(CppSimulator, "_wait_for_jobs"):
            r_hpc.simulate_with_parameters_all(theta, backend="hpc")
        assert jm.submit_cpp_fused_jobs.call_count == 1


class TestSimulateWithParametersAllEvolveOnce:
    """The headline #90 Phase 3 acceptance criterion, end to end with a
    real fused batch + real binary: PPC evolves each theta ONCE, not once
    per scenario."""

    def test_evolve_runs_once_per_theta_not_per_scenario(self, ppc_env):
        sims = {n: _ppc_sim(ppc_env, n) for n in ("baseline", "clinical", "gvax")}
        r = MultiScenarioRunner(sims)
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])  # 2 thetas, 3 scenarios

        results = r.simulate_with_parameters_all(theta)

        # 2 thetas × 3 scenarios: a per-scenario loop would evolve 6
        # times; the fused batch evolves exactly 2 (once per theta).
        n_evolves = len(ppc_env["evolve_counter"].read_text().splitlines())
        assert n_evolves == 2
        assert set(results) == {"baseline", "clinical", "gvax"}
        for theta_out, table in results.values():
            np.testing.assert_allclose(theta_out, theta)
            assert table.num_rows == 2
            assert "ts:spA_t0" in table.column_names
