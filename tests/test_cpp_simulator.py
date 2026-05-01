"""Tests for qsp_hpc.simulation.cpp_simulator.CppSimulator."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qsp_hpc.cpp.batch_runner import BatchResult

# ---------------------------------------------------------------------------
# Fixtures: fake binary, template, priors CSV
# ---------------------------------------------------------------------------

MINI_TEMPLATE = b"""<Param>
  <QSP>
    <A>1.0</A>
    <B>2.0</B>
  </QSP>
</Param>
"""

PRIORS_CSV = """\
name,distribution,dist_param1,dist_param2
A,lognormal,0.0,0.5
B,lognormal,0.5,0.3
"""


def _make_fake_binary(tmp_path: Path) -> Path:
    """Shell script that mimics qsp_sim (2 time-points, 2 species)."""
    script = tmp_path / "fake_qsp_sim.sh"
    script.write_text(
        textwrap.dedent(
            """\
        #!/usr/bin/env bash
        set -e
        while [ $# -gt 0 ]; do
          case "$1" in
            --binary-out) BIN_OUT="$2"; shift 2 ;;
            --species-out) SP_OUT="$2"; shift 2 ;;
            --compartments-out) COMP_OUT="$2"; shift 2 ;;
            --rules-out) RULES_OUT="$2"; shift 2 ;;
            --param) PARAM="$2"; shift 2 ;;
            --t-end-days) TEND="$2"; shift 2 ;;
            --dt-days) DT="$2"; shift 2 ;;
            *) shift ;;
          esac
        done
        python3 - <<PY
import struct
header = struct.pack('<IIQQQQdd', 0x51535042, 2, 2, 2, 0, 0, float("$DT"), float("$TEND"))
body = struct.pack('<4d', 10.0, 20.0, 30.0, 40.0)
open("$BIN_OUT", 'wb').write(header + body)
open("$SP_OUT", 'w').write("spA\\nspB\\n")
open("$COMP_OUT", 'w').write('')
open("$RULES_OUT", 'w').write('')
PY
    """
        )
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    p = tmp_path / "template.xml"
    p.write_bytes(MINI_TEMPLATE)
    return p


@pytest.fixture
def binary_path(tmp_path: Path) -> Path:
    return _make_fake_binary(tmp_path)


@pytest.fixture
def priors_csv(tmp_path: Path) -> Path:
    p = tmp_path / "priors.csv"
    p.write_text(PRIORS_CSV)
    return p


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cache"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Helper to build a synthetic Parquet matching CppBatchRunner output
# ---------------------------------------------------------------------------


def _write_synthetic_parquet(
    path: Path,
    n_sims: int = 5,
    param_names: list[str] | None = None,
    species: list[str] | None = None,
    n_times: int = 2,
    dt: float = 0.1,
    seed: int = 0,
) -> None:
    param_names = param_names or ["A", "B"]
    species = species or ["spA", "spB"]
    rng = np.random.default_rng(seed)
    cols: dict[str, pa.Array] = {
        "simulation_id": pa.array(np.arange(n_sims, dtype=np.int64)),
        "status": pa.array(np.zeros(n_sims, dtype=np.int64)),
        "time": pa.array(
            [(np.arange(n_times) * dt).tolist()] * n_sims,
            type=pa.list_(pa.float64()),
        ),
    }
    for name in param_names:
        cols[f"param:{name}"] = pa.array(rng.uniform(0, 1, n_sims))
    for sp in species:
        cols[sp] = pa.array(
            [rng.uniform(0, 100, n_times).tolist() for _ in range(n_sims)],
            type=pa.list_(pa.float64()),
        )
    pq.write_table(pa.table(cols), str(path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCppSimulatorInit:
    def test_basic_init(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        assert sim.param_names == ["A", "B"]
        assert sim.pool_dir.exists()
        assert sim.config_hash  # non-empty string

    def test_missing_priors_raises(self, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        with pytest.raises(FileNotFoundError, match="Priors CSV"):
            CppSimulator(
                priors_csv="/nonexistent.csv",
                binary_path=binary_path,
                template_xml=template_path,
                cache_dir=cache_dir,
            )

    def test_config_hash_changes_with_binary(self, priors_csv, template_path, cache_dir, tmp_path):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        (tmp_path / "v1").mkdir(exist_ok=True)
        bin1 = _make_fake_binary(tmp_path / "v1")

        bin2_dir = tmp_path / "v2"
        bin2_dir.mkdir()
        bin2 = bin2_dir / "fake_qsp_sim.sh"
        bin2.write_text(bin1.read_text() + "\n# changed")
        bin2.chmod(0o755)

        s1 = CppSimulator(
            priors_csv=priors_csv,
            binary_path=bin1,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        s2 = CppSimulator(
            priors_csv=priors_csv,
            binary_path=bin2,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        assert s1.config_hash != s2.config_hash

    def test_config_hash_changes_with_template(self, priors_csv, binary_path, cache_dir, tmp_path):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        t1 = tmp_path / "t1.xml"
        t1.write_bytes(MINI_TEMPLATE)
        t2 = tmp_path / "t2.xml"
        t2.write_bytes(MINI_TEMPLATE.replace(b"1.0", b"9.0"))

        s1 = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=t1,
            cache_dir=cache_dir,
        )
        s2 = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=t2,
            cache_dir=cache_dir,
        )
        assert s1.config_hash != s2.config_hash

    def test_pool_dir_includes_scenario(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="my_scenario",
        )
        assert sim.pool_dir.name.endswith("_my_scenario")

    def test_config_hash_changes_with_scenario_yaml(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        drug = tmp_path / "drug.yaml"
        drug.write_text("drugs: {}\n")
        s1_yaml = tmp_path / "s1.yaml"
        s1_yaml.write_text("dosing: {nivolumab_dose: 3.0}\n")
        s2_yaml = tmp_path / "s2.yaml"
        s2_yaml.write_text("dosing: {nivolumab_dose: 5.0}\n")

        common = dict(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            drug_metadata_yaml=drug,
        )
        s1 = CppSimulator(**common, scenario_yaml=s1_yaml)
        s2 = CppSimulator(**common, scenario_yaml=s2_yaml)
        assert s1.config_hash != s2.config_hash

    def test_config_hash_changes_with_healthy_yaml(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        h1 = tmp_path / "h1.yaml"
        h1.write_text("densities: {C1: 1e6}\n")
        h2 = tmp_path / "h2.yaml"
        h2.write_text("densities: {C1: 2e6}\n")

        common = dict(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        s1 = CppSimulator(**common, healthy_state_yaml=h1)
        s2 = CppSimulator(**common, healthy_state_yaml=h2)
        assert s1.config_hash != s2.config_hash

    def test_config_hash_changes_with_restriction_classifier(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        """Restricted and unrestricted pools (same other config) must have
        distinct config hashes so they get distinct on-disk pool dirs."""
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        common = dict(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        clf_a = tmp_path / "clf_a"
        clf_a.mkdir()
        (clf_a / "classifier.pkl").write_bytes(b"clf-A-bytes")
        (clf_a / "metadata.json").write_text('{"v": 1}')
        clf_b = tmp_path / "clf_b"
        clf_b.mkdir()
        (clf_b / "classifier.pkl").write_bytes(b"clf-B-bytes")
        (clf_b / "metadata.json").write_text('{"v": 2}')

        s_none = CppSimulator(**common)
        s_a = CppSimulator(**common, restriction_classifier_dir=clf_a)
        s_b = CppSimulator(**common, restriction_classifier_dir=clf_b)
        s_a_tau9 = CppSimulator(
            **common, restriction_classifier_dir=clf_a, restriction_threshold=0.9
        )

        # All four hashes distinct: unrestricted vs restricted, different
        # classifiers, different thresholds.
        hashes = {s_none.config_hash, s_a.config_hash, s_b.config_hash, s_a_tau9.config_hash}
        assert len(hashes) == 4


class TestPoolCaching:
    def test_scan_empty_pool(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        assert sim.get_available_simulations() == 0

    def test_scan_detects_parquet(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
        )
        _write_synthetic_parquet(
            sim.pool_dir / "batch_20260415_120000_ctrl_10sims_seed42.parquet",
            n_sims=10,
        )
        assert sim.get_available_simulations() == 10

    def test_scan_ignores_wrong_scenario(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
        )
        _write_synthetic_parquet(
            sim.pool_dir / "batch_20260415_120000_treatment_10sims_seed42.parquet",
            n_sims=10,
        )
        assert sim.get_available_simulations() == 0

    def test_load_from_pool_filters_failed(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
        )
        # Write a parquet with one failed row
        n = 5
        cols: dict[str, pa.Array] = {
            "simulation_id": pa.array(np.arange(n, dtype=np.int64)),
            "status": pa.array([0, 0, 1, 0, 0], type=pa.int64()),
            "time": pa.array([[0.0, 0.1]] * n, type=pa.list_(pa.float64())),
            "param:A": pa.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "param:B": pa.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            "spA": pa.array(
                [[1.0, 2.0], [3.0, 4.0], [float("nan"), float("nan")], [7.0, 8.0], [9.0, 10.0]],
                type=pa.list_(pa.float64()),
            ),
        }
        pq.write_table(
            pa.table(cols),
            str(sim.pool_dir / "batch_20260415_120000_ctrl_5sims_seed42.parquet"),
        )
        theta, table = sim._load_from_pool(10)
        assert table.num_rows == 4  # one failed row filtered
        assert theta.shape[0] == 4

    def test_load_samples_when_pool_exceeds_request(
        self, priors_csv, binary_path, template_path, cache_dir
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
        )
        _write_synthetic_parquet(
            sim.pool_dir / "batch_20260415_120000_ctrl_20sims_seed42.parquet",
            n_sims=20,
        )
        theta, table = sim._load_from_pool(5)
        assert theta.shape[0] == 5
        assert table.num_rows == 5

    def test_scan_reads_count_from_parquet_metadata_not_filename(
        self, priors_csv, binary_path, template_path, cache_dir
    ):
        """Regression for #21: filename-based row counts overreported when
        array-task chunks dropped. _scan_pool now reads num_rows from the
        parquet footer, so a legacy file claiming '1000sims' but holding
        only 620 rows reports 620 — the truth.
        """
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
        )
        # Filename lies about the count; metadata holds the truth.
        _write_synthetic_parquet(
            sim.pool_dir / "batch_20260415_120000_ctrl_1000sims_seed42.parquet",
            n_sims=620,
        )
        assert sim.get_available_simulations() == 620

    def test_scan_accepts_new_filename_without_nsims_token(
        self, priors_csv, binary_path, template_path, cache_dir
    ):
        """New combine-worker filenames omit the _{N}sims_ segment (#21).
        _scan_pool must still match them and pull the count from metadata."""
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
        )
        _write_synthetic_parquet(
            sim.pool_dir / "batch_20260415_120000_ctrl_seed42.parquet",
            n_sims=15,
        )
        assert sim.get_available_simulations() == 15


class TestCall:
    def test_call_uses_cache_when_available(
        self, priors_csv, binary_path, template_path, cache_dir
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
        )
        _write_synthetic_parquet(
            sim.pool_dir / "batch_20260415_120000_ctrl_20sims_seed42.parquet",
            n_sims=20,
        )
        with patch.object(sim, "_runner") as mock_runner:
            theta, table = sim(10)
            mock_runner.run.assert_not_called()
        assert theta.shape[0] == 10

    def test_call_runs_batch_when_no_cache(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
            t_end_days=0.2,
            dt_days=0.1,
        )

        def fake_run(**kwargs):
            _write_synthetic_parquet(
                kwargs["output_path"],
                n_sims=kwargs["theta_matrix"].shape[0],
                param_names=list(kwargs["param_names"]),
            )
            return BatchResult(
                parquet_path=kwargs["output_path"],
                n_sims=kwargs["theta_matrix"].shape[0],
                n_failed=0,
                species_names=["spA", "spB"],
                n_times=2,
            )

        with patch.object(sim._runner, "run", side_effect=fake_run):
            theta, table = sim(5)

        assert theta.shape == (5, 2)
        assert table.num_rows == 5
        assert sim.get_available_simulations() == 5

    def test_call_tops_up_partial_cache(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
            t_end_days=0.2,
            dt_days=0.1,
        )
        # Pre-seed pool with 3 sims
        _write_synthetic_parquet(
            sim.pool_dir / "batch_20260415_120000_ctrl_3sims_seed42.parquet",
            n_sims=3,
        )

        run_calls = []

        def fake_run(**kwargs):
            run_calls.append(kwargs["theta_matrix"].shape[0])
            _write_synthetic_parquet(
                kwargs["output_path"],
                n_sims=kwargs["theta_matrix"].shape[0],
                param_names=list(kwargs["param_names"]),
            )
            return BatchResult(
                parquet_path=kwargs["output_path"],
                n_sims=kwargs["theta_matrix"].shape[0],
                n_failed=0,
                species_names=["spA", "spB"],
                n_times=2,
            )

        with patch.object(sim._runner, "run", side_effect=fake_run):
            theta, table = sim(10)

        assert run_calls == [7]  # 10 - 3 = 7 new sims
        assert theta.shape[0] == 10

    def test_call_tuple_batch_size(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
        )
        _write_synthetic_parquet(
            sim.pool_dir / "batch_20260415_120000_ctrl_20sims_seed42.parquet",
            n_sims=20,
        )
        with patch.object(sim, "_runner"):
            theta, table = sim((5,))
        assert theta.shape[0] == 5

    def test_second_call_uses_cache(self, priors_csv, binary_path, template_path, cache_dir):
        """After running once, second call with same size hits cache."""
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            scenario="ctrl",
            t_end_days=0.2,
            dt_days=0.1,
        )

        def fake_run(**kwargs):
            _write_synthetic_parquet(
                kwargs["output_path"],
                n_sims=kwargs["theta_matrix"].shape[0],
                param_names=list(kwargs["param_names"]),
            )
            return BatchResult(
                parquet_path=kwargs["output_path"],
                n_sims=kwargs["theta_matrix"].shape[0],
                n_failed=0,
                species_names=["spA", "spB"],
                n_times=2,
            )

        with patch.object(sim._runner, "run", side_effect=fake_run) as mock_run:
            sim(5)
            assert mock_run.call_count == 1
            sim(5)
            assert mock_run.call_count == 1  # no additional run


# ---------------------------------------------------------------------------
# M9: HPC tier (run_hpc) — 3-tier cache walk against an HPC pool
# ---------------------------------------------------------------------------


class TestCppSimulatorRunHpc:
    """run_hpc() mirrors QSPSimulator's 3-tier flow against C++ pools."""

    @staticmethod
    def _stub_test_stats_csv(tmp_path: Path) -> Path:
        p = tmp_path / "test_stats.csv"
        p.write_text("name,model_output_code\n")
        return p

    @staticmethod
    def _make_sim(priors_csv, binary_path, template_path, cache_dir, tmp_path):
        from unittest.mock import MagicMock

        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        ts_csv = TestCppSimulatorRunHpc._stub_test_stats_csv(tmp_path)
        ms_file = tmp_path / "model_structure.json"
        ms_file.write_text("{}\n")
        job_manager = MagicMock()
        job_manager.config.simulation_pool_path = "/scratch/sims"
        # Default to "plenty derived" so Tier 2 hits download by default.
        # Tests covering the partial top-up path override this explicitly.
        job_manager.count_hpc_test_stats.return_value = 10**9
        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            job_manager=job_manager,
            test_stats_csv=ts_csv,
            model_structure_file=ms_file,
            poll_interval=0.001,
        )
        # Don't actually sleep / poll on tests
        sim._wait_for_jobs = MagicMock()
        return sim, job_manager

    def test_run_hpc_requires_job_manager(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        with pytest.raises(RuntimeError, match="job_manager"):
            sim.run_hpc(5)

    def test_run_hpc_requires_model_structure_file(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        from unittest.mock import MagicMock

        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        ts_csv = self._stub_test_stats_csv(tmp_path)
        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            job_manager=MagicMock(),
            test_stats_csv=ts_csv,
        )
        with pytest.raises(RuntimeError, match="model_structure_file"):
            sim.run_hpc(5)

    def test_run_hpc_local_cache_hit_skips_hpc(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        sim, jm = self._make_sim(priors_csv, binary_path, template_path, cache_dir, tmp_path)
        ts_hash = sim._compute_test_stats_hash()

        # Pre-populate the local cache with 5 (params, test_stats) rows
        rng = np.random.default_rng(0)
        params = rng.uniform(size=(5, 2))
        test_stats = rng.uniform(size=(5, 3))
        sim._persist_local_test_stats(sim._local_test_stats_path(ts_hash), params, test_stats)

        out_params, out_ts = sim.run_hpc(3)
        assert out_params.shape == (3, 2)
        assert out_ts.shape == (3, 3)
        # Job manager should not have been touched at all on a cache hit
        jm.check_hpc_test_stats.assert_not_called()
        jm.submit_cpp_jobs.assert_not_called()

    def test_run_hpc_tier2_downloads_from_hpc(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        sim, jm = self._make_sim(priors_csv, binary_path, template_path, cache_dir, tmp_path)
        ts_hash = sim._compute_test_stats_hash()

        # No local cache — HPC has pre-derived stats
        from qsp_hpc.batch.hpc_job_manager import DownloadResult

        rng = np.random.default_rng(1)
        hpc_params = rng.uniform(size=(4, 2))
        hpc_ts = rng.uniform(size=(4, 3))
        jm.check_hpc_test_stats.return_value = True
        jm.download_test_stats_full.return_value = DownloadResult(
            params=hpc_params,
            test_stats=hpc_ts,
            sample_index=np.arange(4, dtype=np.int64),
            param_names=list(sim.param_names),
        )

        out_params, out_ts = sim.run_hpc(4)
        assert out_params.shape == (4, 2)
        assert out_ts.shape == (4, 3)

        jm.check_hpc_test_stats.assert_called_once()
        jm.download_test_stats_full.assert_called_once()
        # Pool path passed to check_hpc_test_stats must be
        # {simulation_pool_path}/{simulation_pool_id} so HPC and local agree
        (pool_path, _hash), kw = jm.check_hpc_test_stats.call_args
        assert pool_path.endswith(f"/{sim.simulation_pool_id}")
        assert kw["expected_n_sims"] == 4

        # Local cache must be populated for next-call hit
        assert sim._local_test_stats_path(ts_hash).exists()

    def test_run_hpc_tier2_partial_falls_through_to_topup(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        """Regression: Tier 2 returned TRUE for partial hits (e.g. 40
        derived stats on HPC when 1000 were requested), then
        ``_download_and_persist`` pulled the 40, ``_sample_first_n``
        returned all 40, and ``run_hpc`` silently returned an undersized
        array. Caught by the N=1000 SBI smoke — user requested 1000,
        got 40. Fix: when count < n, skip the pre-topup download and
        fall through to Tier 3 / 3.5 so the pool gets topped up and a
        single post-topup download covers all rows (issue #63)."""
        from qsp_hpc.batch.hpc_job_manager import DownloadResult, JobInfo

        sim, jm = self._make_sim(priors_csv, binary_path, template_path, cache_dir, tmp_path)

        # HPC has 40 test stats but we ask for 1000 → partial hit.
        rng = np.random.default_rng(10)
        jm.check_hpc_test_stats.return_value = True
        jm.count_hpc_test_stats.return_value = 40

        # Tier 3 sees the pool has 40 sims and falls into Tier 3.5 (top-up).
        jm.result_collector.check_pool_directory_exists.return_value = True
        jm.result_collector.count_pool_simulations.return_value = 40

        # Tier 3.5 submits a delta C++ array + chained derivation; then
        # the final download pulls 1000 rows (old 40 + new 960).
        jm.submit_cpp_jobs.return_value = JobInfo(
            job_ids=["array", "combine", "derive"],
            state_file="",
            n_jobs=1,
            n_simulations=960,
            submission_time="now",
        )
        full_params = rng.uniform(size=(1000, 2))
        full_ts = rng.uniform(size=(1000, 3))
        # Issue #63: partial Tier 2 should NOT pre-download — only the
        # post-topup download fires, returning the full 1000 rows.
        jm.download_test_stats_full.return_value = DownloadResult(
            params=full_params,
            test_stats=full_ts,
            sample_index=np.arange(1000, dtype=np.int64),
            param_names=list(sim.param_names),
        )

        out_params, out_ts = sim.run_hpc(1000)
        assert out_params.shape == (1000, 2)
        assert out_ts.shape == (1000, 3)

        # Tier 3.5 was entered: submit_cpp_jobs called for the delta.
        jm.submit_cpp_jobs.assert_called_once()
        # start_index should be 40 (already have those from pool).
        _, kwargs = jm.submit_cpp_jobs.call_args
        assert kwargs["num_simulations"] == 960
        # Issue #63: only one download — the post-topup one.
        assert jm.download_test_stats_full.call_count == 1

    def test_run_hpc_tier3_derives_when_full_sims_present(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        sim, jm = self._make_sim(priors_csv, binary_path, template_path, cache_dir, tmp_path)

        from qsp_hpc.batch.hpc_job_manager import DownloadResult

        rng = np.random.default_rng(2)
        params = rng.uniform(size=(3, 2))
        ts = rng.uniform(size=(3, 3))
        jm.check_hpc_test_stats.return_value = False
        jm.result_collector.check_pool_directory_exists.return_value = True
        jm.result_collector.count_pool_simulations.return_value = 3
        jm.submit_derivation_job.return_value = "9999"
        jm.download_test_stats_full.return_value = DownloadResult(
            params=params,
            test_stats=ts,
            sample_index=np.arange(3, dtype=np.int64),
            param_names=list(sim.param_names),
        )

        out_params, out_ts = sim.run_hpc(3)
        assert out_params.shape == (3, 2)
        assert out_ts.shape == (3, 3)

        # Derivation submitted, no chained C++ array
        jm.submit_derivation_job.assert_called_once()
        jm.submit_cpp_jobs.assert_not_called()
        sim._wait_for_jobs.assert_called_once_with(["9999"])

    def test_run_hpc_tier4_submits_array_with_chained_derivation(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        from qsp_hpc.batch.hpc_job_manager import DownloadResult, JobInfo

        sim, jm = self._make_sim(priors_csv, binary_path, template_path, cache_dir, tmp_path)

        rng = np.random.default_rng(3)
        params = rng.uniform(size=(2, 2))
        ts = rng.uniform(size=(2, 3))
        jm.check_hpc_test_stats.return_value = False
        jm.result_collector.check_pool_directory_exists.return_value = False
        jm.result_collector.count_pool_simulations.return_value = 0
        jm.submit_cpp_jobs.return_value = JobInfo(
            job_ids=["aaa", "bbb"],
            state_file="state.pkl",
            n_jobs=1,
            n_simulations=2,
            submission_time="now",
        )
        jm.download_test_stats_full.return_value = DownloadResult(
            params=params,
            test_stats=ts,
            sample_index=np.arange(2, dtype=np.int64),
            param_names=list(sim.param_names),
        )

        out_params, out_ts = sim.run_hpc(2)
        assert out_params.shape == (2, 2)
        assert out_ts.shape == (2, 3)

        # submit_cpp_jobs called with derive_test_stats=True and matching hash
        jm.submit_cpp_jobs.assert_called_once()
        kwargs = jm.submit_cpp_jobs.call_args.kwargs
        assert kwargs["derive_test_stats"] is True
        assert kwargs["test_stats_csv"] == str(sim.test_stats_csv)
        assert kwargs["test_stats_hash"] == sim._compute_test_stats_hash()
        assert kwargs["simulation_pool_id"] == sim.simulation_pool_id
        # Wait covered both job ids
        sim._wait_for_jobs.assert_called_once_with(["aaa", "bbb"])

    def test_simulation_pool_id_uses_binary_aware_hash(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        """Local pool dir name and HPC pool id must agree (binary-aware)."""
        sim, _ = self._make_sim(priors_csv, binary_path, template_path, cache_dir, tmp_path)
        assert sim.simulation_pool_id == sim.pool_dir.name
        # Format check: {model_version}_{hash[:8]}_{scenario}
        parts = sim.simulation_pool_id.rsplit("_", 1)
        assert parts[1] == "default"  # default scenario


# ---------------------------------------------------------------------------
# M9: calibration_targets — public-facing API used by pdac-build
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_calibration_targets_dir(tmp_path: Path) -> Path:
    """One YAML calibration target — minimum the loader accepts."""
    import yaml as _yaml

    cal_dir = tmp_path / "calibration_targets"
    cal_dir.mkdir()
    target = {
        "calibration_target_id": "spA_t0",
        "observable": {
            "code": (
                "def compute_observable(time, species_dict, constants, ureg):\n"
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


class TestCppSimulatorCalibrationTargets:
    """CppSimulator accepts calibration_targets (YAML dir) the same way
    QSPSimulator does — serializes to a temp CSV used everywhere
    downstream (hashing, HPC upload).
    """

    def test_init_with_calibration_targets(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        sample_calibration_targets_dir,
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
        )
        assert sim.test_stats_csv is not None
        assert sim.test_stats_csv.exists()
        # _calibration_targets_dir is normalized to List[Path] so the
        # multi-dir form (literature + mechanistic) is supported uniformly.
        assert sim._calibration_targets_dir == [sample_calibration_targets_dir.resolve()]

    def test_temp_csv_has_expected_columns(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        sample_calibration_targets_dir,
    ):
        import pandas as pd

        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
        )
        df = pd.read_csv(sim.test_stats_csv)
        assert "test_statistic_id" in df.columns
        assert "model_output_code" in df.columns
        assert df.iloc[0]["test_statistic_id"] == "spA_t0"

    def test_both_csv_and_yaml_raises(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        tmp_path,
        sample_calibration_targets_dir,
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        ts = tmp_path / "ts.csv"
        ts.write_text("name,model_output_code\n")
        with pytest.raises(ValueError, match="test_stats_csv OR calibration_targets"):
            CppSimulator(
                priors_csv=priors_csv,
                binary_path=binary_path,
                template_xml=template_path,
                cache_dir=cache_dir,
                test_stats_csv=ts,
                calibration_targets=sample_calibration_targets_dir,
            )

    def test_run_hpc_uses_serialized_csv_for_hash(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        tmp_path,
        sample_calibration_targets_dir,
    ):
        """When calibration_targets is provided, _compute_test_stats_hash
        hashes the serialized temp CSV — so HPC and local agree."""
        from unittest.mock import MagicMock

        from qsp_hpc.simulation.cpp_simulator import CppSimulator
        from qsp_hpc.utils.hash_utils import compute_test_stats_hash

        job_manager = MagicMock()
        job_manager.config.simulation_pool_path = "/scratch/sims"
        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
            job_manager=job_manager,
        )
        expected = compute_test_stats_hash(sim.test_stats_csv)
        assert sim._compute_test_stats_hash() == expected


# ---------------------------------------------------------------------------
# #31: CppSimulator.validate() pre-flight checks
# ---------------------------------------------------------------------------


class TestCppSimulatorValidate:
    """validate() catches local mistakes before any HPC round-trip.

    Motivating bug: 50 array tasks each crashed with ParamNotFoundError
    because pdac_priors.csv carried 40 orphan rows not in param_all.xml.
    Without set -e (#27) the workflow saw 50/50 done, failed=0; the bug
    only surfaced ~30min later when derivation found zero parquets.
    """

    def test_passes_when_priors_subset_of_template(
        self, priors_csv, binary_path, template_path, cache_dir
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        # Fixture priors lists A, B; template fixture exposes A, B.
        sim.validate()  # no raise

    def test_raises_when_priors_have_orphan_name(
        self, binary_path, template_path, cache_dir, tmp_path
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        bad_priors = tmp_path / "bad_priors.csv"
        bad_priors.write_text(
            "name,distribution,dist_param1,dist_param2\n"
            "A,lognormal,0.0,0.5\n"
            "ORPHAN_NOT_IN_XML,lognormal,0.0,0.5\n"
        )
        sim = CppSimulator(
            priors_csv=bad_priors,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        with pytest.raises(ValueError, match="not in XML template"):
            sim.validate()

    def test_error_message_names_offending_columns(
        self, binary_path, template_path, cache_dir, tmp_path
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        bad_priors = tmp_path / "bad_priors.csv"
        bad_priors.write_text(
            "name,distribution,dist_param1,dist_param2\n"
            "A,lognormal,0.0,0.5\n"
            "GHOST_PARAM_1,lognormal,0.0,0.5\n"
            "GHOST_PARAM_2,lognormal,0.0,0.5\n"
        )
        sim = CppSimulator(
            priors_csv=bad_priors,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        with pytest.raises(ValueError) as exc_info:
            sim.validate()
        msg = str(exc_info.value)
        assert "GHOST_PARAM_1" in msg
        assert "GHOST_PARAM_2" in msg
        assert "bad_priors.csv" in msg

    def test_run_hpc_invokes_validate_before_submit(
        self, binary_path, template_path, cache_dir, tmp_path
    ):
        """run_hpc must short-circuit on validation failure — no SSH /
        sbatch calls should fire when the priors are obviously broken."""
        from unittest.mock import MagicMock

        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        bad_priors = tmp_path / "bad_priors.csv"
        bad_priors.write_text(
            "name,distribution,dist_param1,dist_param2\n" "ORPHAN,lognormal,0.0,0.5\n"
        )
        ts_csv = tmp_path / "test_stats.csv"
        ts_csv.write_text("name,model_output_code\n")
        ms_file = tmp_path / "model_structure.json"
        ms_file.write_text("{}\n")

        job_manager = MagicMock()
        job_manager.config.simulation_pool_path = "/scratch/sims"
        job_manager.config.cpp_binary_path = "/hpc/bin/qsp_sim"
        job_manager.config.cpp_template_path = "/hpc/template.xml"

        sim = CppSimulator(
            priors_csv=bad_priors,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            job_manager=job_manager,
            test_stats_csv=ts_csv,
            model_structure_file=ms_file,
        )

        with pytest.raises(ValueError, match="not in XML template"):
            sim.run_hpc(10)

        # No HPC traffic of any kind.
        job_manager.check_hpc_test_stats.assert_not_called()
        job_manager.submit_cpp_jobs.assert_not_called()
        job_manager.submit_derivation_job.assert_not_called()


# ---------------------------------------------------------------------------
# Phase-D: simulate_with_parameters — posterior-predictive + prediction targets
# ---------------------------------------------------------------------------


@pytest.fixture
def prediction_targets_dir(tmp_path: Path) -> Path:
    """Prediction-target YAMLs whose observable picks off spA at t=0.

    Keeps the species surface minimal so the fake qsp_sim binary
    (produces 'spA'/'spB') can feed the derive worker without shims.
    """
    import yaml as _yaml

    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    target = {
        "prediction_target_id": "pred_spa_at_t0",
        "description": "Prediction of spA species at t=0.",
        "observable": {
            "code": (
                "def compute_observable(time, species_dict, constants, ureg):\n"
                "    return species_dict['spA']\n"
            ),
            "units": "cell",
            "species": ["spA"],
            "constants": [],
        },
        "scenario": "ctrl",
        "index_values": None,
        "index_unit": None,
        "index_type": None,
        "rationale": "smoke test.",
        "tags": [],
    }
    (pred_dir / "pred_spa_at_t0.yaml").write_text(_yaml.dump(target))
    return pred_dir


def _fake_run_factory(species=("spA", "spB")):
    """Build a fake ``_runner.run`` that writes a synthetic parquet matching
    the requested theta/param layout. Mirrors the helper the other
    TestCall tests use but preserves the exact sample_indices handed in
    (posterior-predictive needs row-level alignment)."""

    def fake_run(**kwargs):
        theta_matrix = kwargs["theta_matrix"]
        param_names = list(kwargs["param_names"])
        output_path = kwargs["output_path"]
        sample_indices = kwargs.get("sample_indices")

        n_sims = theta_matrix.shape[0]
        n_times = 2
        dt = 0.1
        time_rows = [(np.arange(n_times) * dt).tolist()] * n_sims
        cols = {
            "sample_index": pa.array(
                np.asarray(sample_indices, dtype=np.int64)
                if sample_indices is not None
                else np.arange(n_sims, dtype=np.int64)
            ),
            "simulation_id": pa.array(np.arange(n_sims, dtype=np.int64)),
            "status": pa.array(np.zeros(n_sims, dtype=np.int64)),
            "time": pa.array(time_rows, type=pa.list_(pa.float64())),
        }
        for j, name in enumerate(param_names):
            cols[f"param:{name}"] = pa.array(theta_matrix[:, j])
        # Deterministic species: spA = row index, spB = 2 * row index
        # so test stats can identify which row they came from.
        for sp_idx, sp in enumerate(species):
            trajectories = []
            for i in range(n_sims):
                base = float(i + 1) * (1.0 if sp_idx == 0 else 2.0)
                trajectories.append([base, base + 0.1])
            cols[sp] = pa.array(trajectories, type=pa.list_(pa.float64()))
        pq.write_table(pa.table(cols), str(output_path))
        return BatchResult(
            parquet_path=output_path,
            n_sims=n_sims,
            n_failed=0,
            species_names=list(species),
            n_times=n_times,
        )

    return fake_run


class TestSimulateWithParameters:
    """CppSimulator.simulate_with_parameters: posterior-predictive at
    user-supplied thetas, with optional prediction targets mixed into the
    output columns alongside calibration targets."""

    def test_requires_targets_csv_or_yaml(self, priors_csv, binary_path, template_path, cache_dir):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
        )
        with pytest.raises(RuntimeError, match="test_stats_csv or calibration_targets"):
            sim.simulate_with_parameters(np.zeros((2, 2)))

    def test_rejects_wrong_shape(
        self, priors_csv, binary_path, template_path, cache_dir, sample_calibration_targets_dir
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
        )
        with pytest.raises(ValueError, match="2-D"):
            sim.simulate_with_parameters(np.zeros(5))
        with pytest.raises(ValueError, match="priors CSV has"):
            sim.simulate_with_parameters(np.zeros((3, 7)))

    def test_returns_table_with_named_ts_columns(
        self, priors_csv, binary_path, template_path, cache_dir, sample_calibration_targets_dir
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
            scenario="ctrl",
            t_end_days=0.2,
            dt_days=0.1,
        )
        theta = np.array([[0.5, 1.0], [1.5, 2.0], [3.0, 4.0]])
        with patch.object(sim._runner, "run", side_effect=_fake_run_factory()):
            theta_out, table = sim.simulate_with_parameters(theta)

        assert theta_out.shape == theta.shape
        np.testing.assert_allclose(theta_out, theta)
        assert "sample_index" in table.column_names
        assert "status" in table.column_names
        assert "param:A" in table.column_names
        assert "param:B" in table.column_names
        # spA_t0 reads species_dict['spA'] → equals spA at t=0 = row+1.
        ts_col = table.column("ts:spA_t0").to_numpy()
        np.testing.assert_allclose(ts_col, [1.0, 2.0, 3.0])

    def test_prediction_columns_appear(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        sample_calibration_targets_dir,
        prediction_targets_dir,
    ):
        """Handoff acceptance criterion: 'prediction columns appear in the
        output test_stats DataFrame' when prediction_targets is supplied."""
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
            scenario="ctrl",
            t_end_days=0.2,
            dt_days=0.1,
        )
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])
        with patch.object(sim._runner, "run", side_effect=_fake_run_factory()):
            _, table = sim.simulate_with_parameters(
                theta, prediction_targets=prediction_targets_dir
            )
        assert "ts:spA_t0" in table.column_names
        assert "ts:pred_spa_at_t0" in table.column_names
        # Same observable shape (spA at t=0) → values agree within tolerance.
        np.testing.assert_allclose(
            table.column("ts:pred_spa_at_t0").to_numpy(),
            table.column("ts:spA_t0").to_numpy(),
        )

    def test_second_call_hits_cache_without_rerunning_sim(
        self, priors_csv, binary_path, template_path, cache_dir, sample_calibration_targets_dir
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
            scenario="ctrl",
            t_end_days=0.2,
            dt_days=0.1,
        )
        theta = np.array([[0.5, 1.0], [1.5, 2.0]])
        with patch.object(sim._runner, "run", side_effect=_fake_run_factory()) as mock_run:
            sim.simulate_with_parameters(theta)
            assert mock_run.call_count == 1
            # Identical theta + targets → cached. No second run.
            sim.simulate_with_parameters(theta)
            assert mock_run.call_count == 1

    def test_different_theta_produces_different_cache_dir(
        self, priors_csv, binary_path, template_path, cache_dir, sample_calibration_targets_dir
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
            scenario="ctrl",
            t_end_days=0.2,
            dt_days=0.1,
        )
        theta_a = np.array([[0.5, 1.0], [1.5, 2.0]])
        theta_b = np.array([[0.5, 1.0], [9.0, 9.0]])  # different row 1
        with patch.object(sim._runner, "run", side_effect=_fake_run_factory()) as mock_run:
            sim.simulate_with_parameters(theta_a)
            sim.simulate_with_parameters(theta_b)
            assert mock_run.call_count == 2
        # Two suffix-pool dirs: the sibling pools carry the hashes.
        siblings = [
            p
            for p in sim.pool_dir.parent.iterdir()
            if p.is_dir()
            and p.name.startswith(sim.pool_dir.name)
            and "_posterior_predictive_" in p.name
        ]
        assert len(siblings) == 2

    def test_prediction_targets_edit_invalidates_cache(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        sample_calibration_targets_dir,
        prediction_targets_dir,
    ):
        """Editing a prediction YAML must force a re-derivation — otherwise
        the cache returns endpoint values from the old observable code."""
        import yaml as _yaml

        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
            scenario="ctrl",
            t_end_days=0.2,
            dt_days=0.1,
        )
        theta = np.array([[0.5, 1.0]])
        with patch.object(sim._runner, "run", side_effect=_fake_run_factory()) as mock_run:
            sim.simulate_with_parameters(theta, prediction_targets=prediction_targets_dir)
            assert mock_run.call_count == 1

            # Edit the prediction YAML.
            yaml_file = prediction_targets_dir / "pred_spa_at_t0.yaml"
            data = _yaml.safe_load(yaml_file.read_text())
            data["rationale"] = "rewritten for v2"
            yaml_file.write_text(_yaml.dump(data))

            sim.simulate_with_parameters(theta, prediction_targets=prediction_targets_dir)
            assert mock_run.call_count == 2

    def test_id_collision_raises(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        sample_calibration_targets_dir,
        tmp_path,
    ):
        """Prediction id that collides with a calibration id would silently
        overwrite one function in the registry — must raise early."""
        import yaml as _yaml

        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        pred_dir = tmp_path / "colliding"
        pred_dir.mkdir()
        (pred_dir / "spA_t0.yaml").write_text(
            _yaml.dump(
                {
                    "prediction_target_id": "spA_t0",  # same as calibration id
                    "description": "collision",
                    "observable": {
                        "code": (
                            "def compute_observable(time, species_dict, constants, ureg):\n"
                            "    return species_dict['spA']\n"
                        ),
                        "units": "cell",
                        "species": ["spA"],
                        "constants": [],
                    },
                    "scenario": "ctrl",
                    "rationale": "collision test.",
                    "tags": [],
                }
            )
        )
        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
            scenario="ctrl",
        )
        with patch.object(sim._runner, "run", side_effect=_fake_run_factory()):
            with pytest.raises(ValueError, match="collide"):
                sim.simulate_with_parameters(np.array([[0.1, 0.2]]), prediction_targets=pred_dir)


class TestSimulateWithParametersHPC:
    """simulate_with_parameters(backend='hpc'): submit user theta as a
    dedicated HPC pool, wait, download, reshape to the local output schema."""

    @staticmethod
    def _make_sim(
        priors_csv, binary_path, template_path, cache_dir, sample_calibration_targets_dir, tmp_path
    ):
        from unittest.mock import MagicMock

        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        ms_file = tmp_path / "model_structure.json"
        ms_file.write_text("{}\n")
        job_manager = MagicMock()
        job_manager.config.simulation_pool_path = "/scratch/sims"
        job_manager.config.cpp_binary_path = "/scratch/qsp_sim"
        job_manager.config.cpp_template_path = "/scratch/param_all.xml"
        job_manager.count_hpc_test_stats.return_value = 10**9
        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
            job_manager=job_manager,
            model_structure_file=ms_file,
            scenario="ctrl",
            poll_interval=0.001,
        )
        sim._wait_for_jobs = MagicMock()
        return sim, job_manager

    def test_hpc_backend_requires_job_manager(
        self, priors_csv, binary_path, template_path, cache_dir, sample_calibration_targets_dir
    ):
        from qsp_hpc.simulation.cpp_simulator import CppSimulator

        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            calibration_targets=sample_calibration_targets_dir,
            scenario="ctrl",
        )
        with pytest.raises(RuntimeError, match="job_manager"):
            sim.simulate_with_parameters(np.array([[0.1, 0.2]]), backend="hpc")

    def test_hpc_backend_rejects_prediction_targets(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        sample_calibration_targets_dir,
        prediction_targets_dir,
        tmp_path,
    ):
        sim, _jm = self._make_sim(
            priors_csv,
            binary_path,
            template_path,
            cache_dir,
            sample_calibration_targets_dir,
            tmp_path,
        )
        with pytest.raises(NotImplementedError, match="prediction_targets"):
            sim.simulate_with_parameters(
                np.array([[0.1, 0.2]]),
                backend="hpc",
                prediction_targets=prediction_targets_dir,
            )

    def test_hpc_backend_fresh_submit_and_download(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        sample_calibration_targets_dir,
        tmp_path,
    ):
        """Full HPC path: no pre-derived stats → submit_cpp_jobs + download →
        reshape to (sample_index, status, param:*, ts:<id>) schema."""
        from qsp_hpc.batch.hpc_job_manager import DownloadResult, JobInfo

        sim, jm = self._make_sim(
            priors_csv,
            binary_path,
            template_path,
            cache_dir,
            sample_calibration_targets_dir,
            tmp_path,
        )
        jm.check_hpc_test_stats.return_value = False
        jm.submit_cpp_jobs.return_value = JobInfo(
            job_ids=["array", "derive"],
            state_file="",
            n_jobs=1,
            n_simulations=3,
            submission_time="now",
        )
        ts_values = np.array([[1.0], [2.0], [3.0]])  # one cal target: spA_t0
        jm.download_test_stats_full.return_value = DownloadResult(
            params=np.array([[0.5, 1.0], [1.5, 2.0], [3.0, 4.0]]),
            test_stats=ts_values,
            sample_index=np.arange(3, dtype=np.int64),
            param_names=list(sim.param_names),
        )

        theta = np.array([[0.5, 1.0], [1.5, 2.0], [3.0, 4.0]])
        theta_out, table = sim.simulate_with_parameters(theta, backend="hpc")

        # Schema matches local path
        assert theta_out.shape == theta.shape
        np.testing.assert_allclose(theta_out, theta)
        assert "sample_index" in table.column_names
        assert "status" in table.column_names
        assert "param:A" in table.column_names
        assert "param:B" in table.column_names
        assert "ts:spA_t0" in table.column_names
        np.testing.assert_allclose(table.column("ts:spA_t0").to_numpy(), ts_values[:, 0])

        # submit_cpp_jobs was called with derive_test_stats=True and the
        # suffix-pool dir name as simulation_pool_id.
        jm.submit_cpp_jobs.assert_called_once()
        _, kw = jm.submit_cpp_jobs.call_args
        assert kw["derive_test_stats"] is True
        assert kw["num_simulations"] == 3
        assert "_posterior_predictive_" in kw["simulation_pool_id"]

    def test_hpc_backend_prederived_skips_submit(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        sample_calibration_targets_dir,
        tmp_path,
    ):
        """If HPC already has derived test stats at the pool, skip
        submit_cpp_jobs and go straight to download."""
        from qsp_hpc.batch.hpc_job_manager import DownloadResult

        sim, jm = self._make_sim(
            priors_csv,
            binary_path,
            template_path,
            cache_dir,
            sample_calibration_targets_dir,
            tmp_path,
        )
        jm.check_hpc_test_stats.return_value = True
        jm.download_test_stats_full.return_value = DownloadResult(
            params=np.array([[0.1, 0.2]]),
            test_stats=np.array([[7.0]]),
            sample_index=np.arange(1, dtype=np.int64),
            param_names=list(sim.param_names),
        )
        theta = np.array([[0.1, 0.2]])
        _, table = sim.simulate_with_parameters(theta, backend="hpc")

        jm.submit_cpp_jobs.assert_not_called()
        jm.download_test_stats_full.assert_called_once()
        assert table.column("ts:spA_t0").to_numpy().tolist() == [7.0]

    def test_hpc_backend_cache_hit_avoids_hpc(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        sample_calibration_targets_dir,
        tmp_path,
    ):
        """Second call with the same theta should load from the local suffix
        pool and not touch the job manager at all."""
        from qsp_hpc.batch.hpc_job_manager import DownloadResult

        sim, jm = self._make_sim(
            priors_csv,
            binary_path,
            template_path,
            cache_dir,
            sample_calibration_targets_dir,
            tmp_path,
        )
        jm.check_hpc_test_stats.return_value = True
        jm.download_test_stats_full.return_value = DownloadResult(
            params=np.array([[0.1, 0.2]]),
            test_stats=np.array([[42.0]]),
            sample_index=np.arange(1, dtype=np.int64),
            param_names=list(sim.param_names),
        )
        theta = np.array([[0.1, 0.2]])
        sim.simulate_with_parameters(theta, backend="hpc")
        assert jm.download_test_stats_full.call_count == 1

        # Second call: identical theta hashes to the same suffix-pool dir,
        # local test_stats.parquet is read back, no HPC interaction.
        jm.check_hpc_test_stats.reset_mock()
        sim.simulate_with_parameters(theta, backend="hpc")
        assert jm.download_test_stats_full.call_count == 1
        jm.check_hpc_test_stats.assert_not_called()

    def test_hpc_backend_reindexes_by_sample_index(
        self,
        priors_csv,
        binary_path,
        template_path,
        cache_dir,
        sample_calibration_targets_dir,
        tmp_path,
    ):
        """HPC may return rows in arbitrary order; the reshape must reindex
        test stats back to caller's sample order so row j corresponds to
        theta[j]."""
        from qsp_hpc.batch.hpc_job_manager import DownloadResult

        sim, jm = self._make_sim(
            priors_csv,
            binary_path,
            template_path,
            cache_dir,
            sample_calibration_targets_dir,
            tmp_path,
        )
        jm.check_hpc_test_stats.return_value = True
        # HPC returns rows out of order: sample_index = [2, 0, 1]
        jm.download_test_stats_full.return_value = DownloadResult(
            params=np.array([[3.0, 4.0], [0.5, 1.0], [1.5, 2.0]]),
            test_stats=np.array([[30.0], [10.0], [20.0]]),
            sample_index=np.array([2, 0, 1], dtype=np.int64),
            param_names=list(sim.param_names),
        )
        theta = np.array([[0.5, 1.0], [1.5, 2.0], [3.0, 4.0]])
        _, table = sim.simulate_with_parameters(theta, backend="hpc")

        # After reindexing, row j's ts aligns with theta[j]
        assert table.column("ts:spA_t0").to_numpy().tolist() == [10.0, 20.0, 30.0]
