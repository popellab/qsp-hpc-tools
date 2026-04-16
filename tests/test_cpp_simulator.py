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
        job_manager = MagicMock()
        job_manager.config.simulation_pool_path = "/scratch/sims"
        sim = CppSimulator(
            priors_csv=priors_csv,
            binary_path=binary_path,
            template_xml=template_path,
            cache_dir=cache_dir,
            job_manager=job_manager,
            test_stats_csv=ts_csv,
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
        rng = np.random.default_rng(1)
        hpc_params = rng.uniform(size=(4, 2))
        hpc_ts = rng.uniform(size=(4, 3))
        jm.check_hpc_test_stats.return_value = True
        jm.download_test_stats.return_value = (hpc_params, hpc_ts)

        out_params, out_ts = sim.run_hpc(4)
        assert out_params.shape == (4, 2)
        assert out_ts.shape == (4, 3)

        jm.check_hpc_test_stats.assert_called_once()
        jm.download_test_stats.assert_called_once()
        # Pool path passed to check_hpc_test_stats must be
        # {simulation_pool_path}/{simulation_pool_id} so HPC and local agree
        ((pool_path, _hash), kw) = jm.check_hpc_test_stats.call_args
        assert pool_path.endswith(f"/{sim.simulation_pool_id}")
        assert kw["expected_n_sims"] == 4

        # Local cache must be populated for next-call hit
        assert sim._local_test_stats_path(ts_hash).exists()

    def test_run_hpc_tier3_derives_when_full_sims_present(
        self, priors_csv, binary_path, template_path, cache_dir, tmp_path
    ):
        sim, jm = self._make_sim(priors_csv, binary_path, template_path, cache_dir, tmp_path)

        rng = np.random.default_rng(2)
        params = rng.uniform(size=(3, 2))
        ts = rng.uniform(size=(3, 3))
        jm.check_hpc_test_stats.return_value = False
        jm.result_collector.check_pool_directory_exists.return_value = True
        jm.result_collector.count_pool_simulations.return_value = 3
        jm.submit_derivation_job.return_value = "9999"
        jm.download_test_stats.return_value = (params, ts)

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
        from qsp_hpc.batch.hpc_job_manager import JobInfo

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
        jm.download_test_stats.return_value = (params, ts)

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
        assert sim._calibration_targets_dir == sample_calibration_targets_dir.resolve()

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
