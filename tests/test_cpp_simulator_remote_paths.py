"""Regression tests for CppSimulator local-vs-HPC path separation.

``CppSimulator`` keeps a laptop-resident ``binary_path`` / ``template_xml``
for config-hash computation and local CppBatchRunner execution. Those paths
do NOT exist on HPC. The ``run_hpc()`` code path must pass the HPC-side
paths — from ``remote_binary_path`` / ``remote_template_xml`` ctor args, or
falling back to ``credentials.cpp.{binary,template}_path`` — to
``HPCJobManager.submit_cpp_jobs``; otherwise the embedded sbatch and the
binary-existence check fail on the cluster with a confusing "file not found"
pointing at a laptop path.

This test file locks that contract down so the next refactor doesn't
silently re-leak ``self.binary_path`` into ``submit_cpp_jobs``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Reuse fixtures from test_cpp_simulator.py via conftest import pattern.
# The fake binary + template + priors fixtures there are general-purpose.
from tests.test_cpp_simulator import (  # noqa: F401  (used as fixtures)
    binary_path,
    cache_dir,
    priors_csv,
    template_path,
)

PRIORS_CSV_CONTENT = """\
name,distribution,dist_param1,dist_param2
A,lognormal,0.0,0.5
B,lognormal,0.5,0.3
"""


@pytest.fixture
def test_stats_csv(tmp_path: Path) -> Path:
    """Minimal test-stats CSV. run_hpc() requires this to be set but we
    mock out the submit step, so contents don't need to be valid."""
    p = tmp_path / "test_stats.csv"
    p.write_text("name,model_output_code\n")
    return p


@pytest.fixture
def model_structure_file(tmp_path: Path) -> Path:
    """Minimal model structure file. Similar story — required by run_hpc()'s
    precondition check, never actually read in these tests."""
    p = tmp_path / "model_structure.json"
    p.write_text("{}")
    return p


def _make_job_manager(
    *,
    cpp_binary_path: str = "/hpc/SPQSP_PDAC/PDAC/qsp/sim/build/qsp_sim",
    cpp_template_path: str = "/hpc/SPQSP_PDAC/PDAC/sim/resource/param_all.xml",
    simulation_pool_path: str = "/hpc/pools",
) -> MagicMock:
    """Build a mock HPCJobManager that answers just enough of run_hpc()'s
    Tier-2/3/4 probe calls to route to Tier 4 (fresh-submit), then captures
    the submit_cpp_jobs kwargs."""
    jm = MagicMock()
    jm.config = SimpleNamespace(
        cpp_binary_path=cpp_binary_path,
        cpp_template_path=cpp_template_path,
        simulation_pool_path=simulation_pool_path,
    )
    # Tier-2 miss:
    jm.check_hpc_test_stats.return_value = False
    # Tier-3 miss (no pool at all):
    jm.result_collector.check_pool_directory_exists.return_value = False
    jm.result_collector.count_pool_simulations.return_value = 0
    # Tier-4: submit returns a stubbed JobInfo so run_hpc proceeds.
    jm.submit_cpp_jobs.return_value = SimpleNamespace(job_ids=["stub_array", "stub_derive"])
    return jm


class TestTopUpTier35:
    """run_hpc() must submit only the delta when pool has 0 < n_hpc < n."""

    def _build_simulator(
        self,
        *,
        priors_csv: Path,
        binary_path: Path,
        template_path: Path,
        cache_dir: Path,
        test_stats_csv: Path,
        model_structure_file: Path,
        job_manager: MagicMock,
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        return CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            job_manager=job_manager,
            test_stats_csv=test_stats_csv,
            model_structure_file=model_structure_file,
        )

    def _shortcircuit(self, sim, monkeypatch, n_request: int = 1000):
        """Stop run_hpc after submit_cpp_jobs returns so the call captures
        the kwargs without doing SCP/SLURM work."""
        import numpy as np

        monkeypatch.setattr(sim, "_wait_for_jobs", lambda _ids: None)
        monkeypatch.setattr(
            sim,
            "_download_and_persist",
            lambda *a, **k: (np.zeros((1, 2)), np.zeros((1, 1))),
        )
        monkeypatch.setattr(sim, "_sample_first_n", lambda p, t, n: (p, t))
        return sim.run_hpc(n_request)

    def _jm_with_partial_pool(self, n_hpc: int) -> MagicMock:
        jm = MagicMock()
        jm.config = SimpleNamespace(
            cpp_binary_path="/hpc/qsp_sim",
            cpp_template_path="/hpc/param_all.xml",
            simulation_pool_path="/hpc/pools",
        )
        # Tier-2 miss (no pre-derived test stats):
        jm.check_hpc_test_stats.return_value = False
        # Tier-3 miss (pool exists but n_hpc < n_requested):
        jm.result_collector.check_pool_directory_exists.return_value = True
        jm.result_collector.count_pool_simulations.return_value = n_hpc
        jm.submit_cpp_jobs.return_value = SimpleNamespace(
            job_ids=["array", "combine", "derive"]
        )
        return jm

    def test_partial_pool_submits_only_the_delta(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        test_stats_csv,
        model_structure_file,
        monkeypatch,
    ):
        """Pool has 720/1000 → submit 280, not a fresh 1000.

        Regression lock for the bug where Tier 4 resubmitted full N even
        when the pool already had most of the sims — wasting up to ~70%
        of the compute on any re-run.
        """
        jm = self._jm_with_partial_pool(n_hpc=720)
        sim = self._build_simulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_path=template_path,
            cache_dir=cache_dir,
            test_stats_csv=test_stats_csv,
            model_structure_file=model_structure_file,
            job_manager=jm,
        )
        self._shortcircuit(sim, monkeypatch)

        kwargs = jm.submit_cpp_jobs.call_args.kwargs
        assert kwargs["num_simulations"] == 280, (
            f"Expected delta submission of 280, got {kwargs['num_simulations']}"
        )

    def test_partial_pool_uses_offset_theta_indices(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        test_stats_csv,
        model_structure_file,
        monkeypatch,
    ):
        """The params CSV uploaded for the top-up must contain thetas at
        indices [720, 1000) — not [0, 280). The deterministic theta pool
        is seed-keyed, so index offset is what preserves identity across
        the existing pool batches + this new batch."""
        import pandas as pd

        jm = self._jm_with_partial_pool(n_hpc=720)
        sim = self._build_simulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_path=template_path,
            cache_dir=cache_dir,
            test_stats_csv=test_stats_csv,
            model_structure_file=model_structure_file,
            job_manager=jm,
        )
        self._shortcircuit(sim, monkeypatch)

        # Reconstruct what _write_params_csv would have produced for
        # (start_index=720, n_sims=280) vs (start=0, n=280) — they must
        # differ, otherwise the top-up is re-running the first slice.
        offset = sim._write_params_csv(280, start_index=720)
        zero_start = sim._write_params_csv(280, start_index=0)
        try:
            df_offset = pd.read_csv(offset)
            df_zero = pd.read_csv(zero_start)
            assert not df_offset.equals(df_zero), (
                "Offset and zero-start slices should differ — theta pool "
                "sampling is not being offset correctly."
            )
        finally:
            offset.unlink(missing_ok=True)
            zero_start.unlink(missing_ok=True)

    def test_empty_pool_submits_full_n(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        test_stats_csv,
        model_structure_file,
        monkeypatch,
    ):
        """Pool empty (no dir) → submit all 1000 starting at index 0."""
        jm = self._jm_with_partial_pool(n_hpc=0)
        jm.result_collector.check_pool_directory_exists.return_value = False
        sim = self._build_simulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_path=template_path,
            cache_dir=cache_dir,
            test_stats_csv=test_stats_csv,
            model_structure_file=model_structure_file,
            job_manager=jm,
        )
        self._shortcircuit(sim, monkeypatch)

        kwargs = jm.submit_cpp_jobs.call_args.kwargs
        assert kwargs["num_simulations"] == 1000


class TestRemotePathResolution:
    """run_hpc() must use HPC paths, not laptop paths, in submit_cpp_jobs."""

    def _build_simulator(
        self,
        *,
        priors_csv: Path,
        binary_path: Path,
        template_path: Path,
        cache_dir: Path,
        test_stats_csv: Path,
        model_structure_file: Path,
        job_manager: MagicMock,
        remote_binary_path: str | None = None,
        remote_template_xml: str | None = None,
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        return CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            job_manager=job_manager,
            test_stats_csv=test_stats_csv,
            model_structure_file=model_structure_file,
            remote_binary_path=remote_binary_path,
            remote_template_xml=remote_template_xml,
        )

    def _invoke_run_hpc(self, sim, monkeypatch):
        """Short-circuit run_hpc() after submit_cpp_jobs is called.

        We don't want to exercise _wait_for_jobs or _download_and_persist —
        those do real SCP work. Patch them to no-ops so the call chain
        terminates right after submit_cpp_jobs captures its kwargs.
        """
        import numpy as np

        monkeypatch.setattr(sim, "_wait_for_jobs", lambda _ids: None)
        monkeypatch.setattr(
            sim,
            "_download_and_persist",
            lambda *a, **k: (np.zeros((1, 2)), np.zeros((1, 1))),
        )
        monkeypatch.setattr(sim, "_sample_first_n", lambda p, t, n: (p, t))
        return sim.run_hpc(1)

    def test_remote_paths_from_ctor_win_over_credentials(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        test_stats_csv,
        model_structure_file,
        monkeypatch,
    ):
        """Explicit ctor args override credentials.yaml values."""
        jm = _make_job_manager(
            cpp_binary_path="/hpc/from-credentials/qsp_sim",
            cpp_template_path="/hpc/from-credentials/param_all.xml",
        )
        sim = self._build_simulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_path=template_path,
            cache_dir=cache_dir,
            test_stats_csv=test_stats_csv,
            model_structure_file=model_structure_file,
            job_manager=jm,
            remote_binary_path="/hpc/from-ctor/qsp_sim",
            remote_template_xml="/hpc/from-ctor/param_all.xml",
        )
        self._invoke_run_hpc(sim, monkeypatch)

        assert jm.submit_cpp_jobs.called, "submit_cpp_jobs was never invoked"
        kwargs = jm.submit_cpp_jobs.call_args.kwargs
        assert kwargs["binary_path"] == "/hpc/from-ctor/qsp_sim"
        assert kwargs["template_path"] == "/hpc/from-ctor/param_all.xml"
        # And critically NOT the laptop paths:
        assert kwargs["binary_path"] != str(sim.binary_path)
        assert kwargs["template_path"] != str(sim.template_xml)

    def test_remote_paths_fallback_to_credentials(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        test_stats_csv,
        model_structure_file,
        monkeypatch,
    ):
        """With no ctor args, run_hpc picks up credentials.cpp.*_path."""
        jm = _make_job_manager(
            cpp_binary_path="/hpc/from-credentials/qsp_sim",
            cpp_template_path="/hpc/from-credentials/param_all.xml",
        )
        sim = self._build_simulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_path=template_path,
            cache_dir=cache_dir,
            test_stats_csv=test_stats_csv,
            model_structure_file=model_structure_file,
            job_manager=jm,
        )
        self._invoke_run_hpc(sim, monkeypatch)

        kwargs = jm.submit_cpp_jobs.call_args.kwargs
        assert kwargs["binary_path"] == "/hpc/from-credentials/qsp_sim"
        assert kwargs["template_path"] == "/hpc/from-credentials/param_all.xml"

    def test_local_path_never_leaks_to_submit(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        test_stats_csv,
        model_structure_file,
        monkeypatch,
    ):
        """Regression lock for the 2026-04-16 smoke-run bug: the laptop
        binary_path must never be what submit_cpp_jobs sees."""
        jm = _make_job_manager()
        sim = self._build_simulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_path=template_path,
            cache_dir=cache_dir,
            test_stats_csv=test_stats_csv,
            model_structure_file=model_structure_file,
            job_manager=jm,
        )
        self._invoke_run_hpc(sim, monkeypatch)

        kwargs = jm.submit_cpp_jobs.call_args.kwargs
        # The local fixture binary lives under tmp_path; that must not
        # show up in what we hand to the cluster.
        assert str(binary_path) not in kwargs["binary_path"]
        assert str(template_path) not in kwargs["template_path"]

    def test_missing_remote_binary_raises(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        test_stats_csv,
        model_structure_file,
    ):
        """If neither ctor arg nor credentials supply a remote binary path,
        run_hpc must fail loudly — the alternative is silently reusing the
        laptop path on HPC."""
        jm = _make_job_manager(cpp_binary_path="")  # credentials unset
        sim = self._build_simulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_path=template_path,
            cache_dir=cache_dir,
            test_stats_csv=test_stats_csv,
            model_structure_file=model_structure_file,
            job_manager=jm,
        )
        with pytest.raises(RuntimeError, match="HPC binary path unset"):
            sim.run_hpc(1)

    def test_missing_remote_template_raises(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        test_stats_csv,
        model_structure_file,
    ):
        """Same guard for the template XML."""
        jm = _make_job_manager(cpp_template_path="")  # credentials unset
        sim = self._build_simulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_path=template_path,
            cache_dir=cache_dir,
            test_stats_csv=test_stats_csv,
            model_structure_file=model_structure_file,
            job_manager=jm,
        )
        with pytest.raises(RuntimeError, match="HPC template path unset"):
            sim.run_hpc(1)
