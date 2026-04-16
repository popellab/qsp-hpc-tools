"""Tests for C++ HPC integration (M6).

Covers:
- cpp_batch_worker.run_chunk() — the SLURM task entry point
- SLURMJobSubmitter._generate_cpp_slurm_script() — script content
- HPCJobManager.submit_cpp_jobs() — orchestration with mocked SSH
- BatchConfig cpp field parsing
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pyarrow.parquet as pq
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MINI_TEMPLATE = b"""<Param>
  <QSP>
    <A>1.0</A>
    <B>2.0</B>
  </QSP>
</Param>
"""


def _make_fake_binary(tmp_path: Path) -> Path:
    """Shell script mimicking qsp_sim (2 timepoints, 2 species)."""
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
            --param) shift 2 ;;
            --t-end-days) TEND="$2"; shift 2 ;;
            --dt-days) DT="$2"; shift 2 ;;
            *) shift ;;
          esac
        done
        python3 - <<PY
        import struct
        header = struct.pack('<IIQQdd', 0x51535042, 1, 2, 2, float("$DT"), float("$TEND"))
        body = struct.pack('<4d', 10.0, 20.0, 30.0, 40.0)
        open("$BIN_OUT", 'wb').write(header + body)
        open("$SP_OUT", 'w').write("spA\\nspB\\n")
        PY
    """
        )
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    return _make_fake_binary(tmp_path)


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    p = tmp_path / "template.xml"
    p.write_bytes(MINI_TEMPLATE)
    return p


@pytest.fixture
def params_csv(tmp_path: Path) -> Path:
    p = tmp_path / "params.csv"
    p.write_text("A,B\n1.0,2.0\n1.1,2.1\n1.2,2.2\n1.3,2.3\n1.4,2.4\n")
    return p


# ---------------------------------------------------------------------------
# cpp_batch_worker tests
# ---------------------------------------------------------------------------


class TestCppBatchWorker:
    def test_run_chunk_processes_correct_slice(
        self, tmp_path, fake_binary, template_path, params_csv
    ):
        from qsp_hpc.batch.cpp_batch_worker import run_chunk

        pool_dir = tmp_path / "pool"
        config = {
            "binary_path": str(fake_binary),
            "template_path": str(template_path),
            "subtree": "QSP",
            "param_csv": str(params_csv),
            "n_simulations": 5,
            "seed": 42,
            "jobs_per_chunk": 2,
            "t_end_days": 0.2,
            "dt_days": 0.1,
            "simulation_pool_id": "test_pool",
            "simulation_pool_path": str(pool_dir),
            "scenario": "ctrl",
        }

        # Task 0 processes sims [0, 2)
        run_chunk(config, array_idx=0)
        parquets = list((pool_dir / "test_pool").glob("*.parquet"))
        assert len(parquets) == 1
        table = pq.read_table(str(parquets[0]))
        assert table.num_rows == 2
        assert "ctrl" in parquets[0].name
        np.testing.assert_allclose(table.column("param:A").to_numpy(), [1.0, 1.1])

    def test_run_chunk_last_task_partial(self, tmp_path, fake_binary, template_path, params_csv):
        from qsp_hpc.batch.cpp_batch_worker import run_chunk

        pool_dir = tmp_path / "pool"
        config = {
            "binary_path": str(fake_binary),
            "template_path": str(template_path),
            "subtree": "QSP",
            "param_csv": str(params_csv),
            "n_simulations": 5,
            "seed": 42,
            "jobs_per_chunk": 2,
            "t_end_days": 0.2,
            "dt_days": 0.1,
            "simulation_pool_id": "test_pool",
            "simulation_pool_path": str(pool_dir),
            "scenario": "ctrl",
        }

        # Task 2 processes sims [4, 5) — only 1 sim
        run_chunk(config, array_idx=2)
        parquets = list((pool_dir / "test_pool").glob("*.parquet"))
        assert len(parquets) == 1
        table = pq.read_table(str(parquets[0]))
        assert table.num_rows == 1
        np.testing.assert_allclose(table.column("param:A").to_numpy(), [1.4])

    def test_run_chunk_past_end_is_noop(self, tmp_path, fake_binary, template_path, params_csv):
        from qsp_hpc.batch.cpp_batch_worker import run_chunk

        pool_dir = tmp_path / "pool"
        config = {
            "binary_path": str(fake_binary),
            "template_path": str(template_path),
            "subtree": "QSP",
            "param_csv": str(params_csv),
            "n_simulations": 5,
            "seed": 42,
            "jobs_per_chunk": 2,
            "t_end_days": 0.2,
            "dt_days": 0.1,
            "simulation_pool_id": "test_pool",
            "simulation_pool_path": str(pool_dir),
            "scenario": "ctrl",
        }

        # Task 10 is past the end
        run_chunk(config, array_idx=10)
        pool_path = pool_dir / "test_pool"
        if pool_path.exists():
            assert list(pool_path.glob("*.parquet")) == []


# ---------------------------------------------------------------------------
# SLURM script generation tests
# ---------------------------------------------------------------------------


class TestCppSlurmScript:
    def _make_submitter(self):
        from qsp_hpc.batch.hpc_job_manager import BatchConfig
        from qsp_hpc.batch.slurm_job_submitter import SLURMJobSubmitter

        config = BatchConfig(
            ssh_host="test.edu",
            ssh_user="testuser",
            simulation_pool_path="/scratch/sims",
            hpc_venv_path="/home/testuser/.venv/qsp",
            remote_project_path="/home/testuser/project",
            partition="shared",
            time_limit="02:00:00",
        )
        transport = MagicMock()
        return SLURMJobSubmitter(config, transport)

    def test_script_activates_venv_not_matlab(self):
        sub = self._make_submitter()
        script = sub._generate_cpp_slurm_script(n_jobs=5, cpus_per_task=4, memory="8G")

        assert "source" in script and ".venv/qsp/bin/activate" in script
        assert "module load" not in script
        assert "matlab" not in script.lower()
        assert "cpp_batch_worker" in script
        assert "cpp_job_config.json" in script

    def test_script_sbatch_directives(self):
        sub = self._make_submitter()
        script = sub._generate_cpp_slurm_script(n_jobs=10, cpus_per_task=8, memory="16G")

        assert "--array=0-9" in script
        assert "--cpus-per-task=8" in script
        assert "--mem=16G" in script
        assert "--partition=shared" in script
        assert "--time=02:00:00" in script
        assert "qsp_cpp_batch" in script


# ---------------------------------------------------------------------------
# BatchConfig cpp field parsing
# ---------------------------------------------------------------------------


class TestBatchConfigCpp:
    def test_cpp_fields_parsed_from_yaml(self):
        from qsp_hpc.batch.hpc_job_manager import HPCJobManager

        cfg = {
            "ssh": {"host": "test.edu", "user": "u"},
            "paths": {
                "simulation_pool_path": "/scratch/sims",
                "hpc_venv_path": "/home/u/.venv",
            },
            "cpp": {
                "binary_path": "/home/u/bin/qsp_sim",
                "template_path": "/home/u/SPQSP/param_all.xml",
                "subtree": "QSP",
            },
        }
        config = HPCJobManager._parse_config_dict(cfg)
        assert config.cpp_binary_path == "/home/u/bin/qsp_sim"
        assert config.cpp_template_path == "/home/u/SPQSP/param_all.xml"
        assert config.cpp_subtree == "QSP"

    def test_cpp_fields_default_empty(self):
        from qsp_hpc.batch.hpc_job_manager import HPCJobManager

        cfg = {
            "ssh": {"host": "test.edu"},
            "paths": {
                "simulation_pool_path": "/scratch/sims",
                "hpc_venv_path": "/home/u/.venv",
            },
        }
        config = HPCJobManager._parse_config_dict(cfg)
        assert config.cpp_binary_path == ""
        assert config.cpp_template_path == ""
        assert config.cpp_subtree == "QSP"


# ---------------------------------------------------------------------------
# HPCJobManager.submit_cpp_jobs — with mocked transport
# ---------------------------------------------------------------------------


class TestSubmitCppJobs:
    def _make_manager(self, cpp_binary="/usr/bin/qsp_sim", cpp_template="/tmp/p.xml"):
        from qsp_hpc.batch.hpc_job_manager import BatchConfig, HPCJobManager

        config = BatchConfig(
            ssh_host="test.edu",
            ssh_user="testuser",
            simulation_pool_path="/scratch/sims",
            hpc_venv_path="/home/testuser/.venv/qsp",
            remote_project_path="/home/testuser/project",
            partition="shared",
            time_limit="01:00:00",
            cpp_binary_path=cpp_binary,
            cpp_template_path=cpp_template,
            # Explicit repo_path so submit_cpp_jobs' ensure_cpp_binary call
            # doesn't try to derive from the stub `/usr/bin/qsp_sim`.
            cpp_repo_path="/home/testuser/SPQSP_PDAC",
        )
        transport = MagicMock()

        # Command-aware mock: `test -x ... && echo OK` must actually produce
        # "OK" so ensure_cpp_binary's existence check passes; all other
        # commands (git pull, cmake/make, submit) return the SLURM-style
        # stub. Tests that care about a specific exec call install their
        # own side_effect.
        def exec_side_effect(cmd, *args, **kwargs):
            if "echo OK" in cmd:
                return (0, "OK")
            return (0, "Submitted batch job 12345")

        transport.exec.side_effect = exec_side_effect
        transport.upload.return_value = None

        manager = HPCJobManager(config=config, transport=transport)
        return manager, transport

    def test_submit_cpp_jobs_returns_job_info(self, tmp_path):
        manager, transport = self._make_manager()

        csv = tmp_path / "params.csv"
        csv.write_text("A,B\n1.0,2.0\n1.1,2.1\n")

        info = manager.submit_cpp_jobs(
            samples_csv=str(csv),
            num_simulations=2,
            simulation_pool_id="v1_abc_ctrl",
            skip_sync=True,
        )

        assert info.job_ids == ["12345"]
        assert info.n_simulations == 2
        assert info.n_jobs >= 1

    def test_submit_cpp_jobs_uploads_config_json(self, tmp_path):
        manager, transport = self._make_manager()

        csv = tmp_path / "params.csv"
        csv.write_text("A,B\n1.0,2.0\n")

        # Capture the config JSON content during upload (temp file is
        # deleted before we return, so read it inside the mock).
        captured_configs: list[dict] = []

        def capturing_upload(local, remote):
            if "cpp_job_config.json" in remote:
                with open(local) as f:
                    captured_configs.append(json.load(f))

        transport.upload.side_effect = capturing_upload

        manager.submit_cpp_jobs(
            samples_csv=str(csv),
            num_simulations=1,
            simulation_pool_id="pool",
            skip_sync=True,
            t_end_days=90.0,
            dt_days=0.5,
            scenario="treatment",
        )

        assert len(captured_configs) == 1
        uploaded_config = captured_configs[0]
        assert uploaded_config["t_end_days"] == 90.0
        assert uploaded_config["dt_days"] == 0.5
        assert uploaded_config["scenario"] == "treatment"
        assert uploaded_config["binary_path"] == "/usr/bin/qsp_sim"

    def test_submit_cpp_jobs_raises_without_binary(self, tmp_path):
        manager, _ = self._make_manager(cpp_binary="", cpp_template="/tmp/p.xml")

        csv = tmp_path / "params.csv"
        csv.write_text("A\n1.0\n")

        with pytest.raises(ValueError, match="binary_path"):
            manager.submit_cpp_jobs(
                samples_csv=str(csv),
                num_simulations=1,
                simulation_pool_id="pool",
                skip_sync=True,
            )

    def test_submit_cpp_jobs_raises_without_template(self, tmp_path):
        manager, _ = self._make_manager(cpp_binary="/usr/bin/qsp_sim", cpp_template="")

        csv = tmp_path / "params.csv"
        csv.write_text("A\n1.0\n")

        with pytest.raises(ValueError, match="template_path"):
            manager.submit_cpp_jobs(
                samples_csv=str(csv),
                num_simulations=1,
                simulation_pool_id="pool",
                skip_sync=True,
            )

    def test_submit_cpp_jobs_uploads_scenario_yamls(self, tmp_path):
        """scenario_yaml / drug_metadata_yaml / healthy_state_yaml are
        uploaded, and their remote paths land in cpp_job_config.json."""
        import json as _json

        manager, transport = self._make_manager()

        csv = tmp_path / "params.csv"
        csv.write_text("A\n1.0\n")

        scen = tmp_path / "scen.yaml"
        scen.write_text("dosing: {}\n")
        drug = tmp_path / "drug.yaml"
        drug.write_text("drugs: {}\n")
        healthy = tmp_path / "healthy.yaml"
        healthy.write_text("densities: {}\n")

        captured: dict = {}

        def capture(local, remote):
            if "cpp_job_config.json" in remote:
                with open(local) as f:
                    captured["config"] = _json.load(f)

        transport.upload.side_effect = capture

        manager.submit_cpp_jobs(
            samples_csv=str(csv),
            num_simulations=1,
            simulation_pool_id="pool",
            skip_sync=True,
            scenario_yaml=str(scen),
            drug_metadata_yaml=str(drug),
            healthy_state_yaml=str(healthy),
        )

        # All three YAMLs should have been uploaded to batch_jobs/input/.
        upload_dests = [call.args[1] for call in transport.upload.call_args_list]
        assert any(d.endswith("/batch_jobs/input/scenario.yaml") for d in upload_dests)
        assert any(d.endswith("/batch_jobs/input/drug_metadata.yaml") for d in upload_dests)
        assert any(d.endswith("/batch_jobs/input/healthy_state.yaml") for d in upload_dests)

        cfg = captured["config"]
        assert cfg["scenario_yaml"].endswith("/batch_jobs/input/scenario.yaml")
        assert cfg["drug_metadata_yaml"].endswith("/batch_jobs/input/drug_metadata.yaml")
        assert cfg["healthy_state_yaml"].endswith("/batch_jobs/input/healthy_state.yaml")

    def test_submit_cpp_jobs_scenario_without_drug_meta_raises(self, tmp_path):
        manager, _ = self._make_manager()

        csv = tmp_path / "params.csv"
        csv.write_text("A\n1.0\n")
        scen = tmp_path / "scen.yaml"
        scen.write_text("dosing: {}\n")

        with pytest.raises(ValueError, match="scenario_yaml requires drug_metadata_yaml"):
            manager.submit_cpp_jobs(
                samples_csv=str(csv),
                num_simulations=1,
                simulation_pool_id="pool",
                skip_sync=True,
                scenario_yaml=str(scen),
            )

    def test_submit_cpp_jobs_override_paths(self, tmp_path):
        manager, transport = self._make_manager(cpp_binary="", cpp_template="")

        csv = tmp_path / "params.csv"
        csv.write_text("A\n1.0\n")

        info = manager.submit_cpp_jobs(
            samples_csv=str(csv),
            num_simulations=1,
            simulation_pool_id="pool",
            skip_sync=True,
            binary_path="/override/qsp_sim",
            template_path="/override/param.xml",
        )
        assert info.job_ids == ["12345"]


# ---------------------------------------------------------------------------
# HPCJobManager.ensure_cpp_binary
# ---------------------------------------------------------------------------


class TestEnsureCppBinary:
    def _make_manager(self, **overrides):
        from qsp_hpc.batch.hpc_job_manager import BatchConfig, HPCJobManager

        config_kwargs = dict(
            ssh_host="test.edu",
            ssh_user="testuser",
            simulation_pool_path="/scratch/sims",
            hpc_venv_path="/home/testuser/.venv/qsp",
            remote_project_path="/home/testuser/project",
            cpp_binary_path="/home/testuser/SPQSP_PDAC/PDAC/qsp/sim/build/qsp_sim",
            cpp_repo_path="/home/testuser/SPQSP_PDAC",
            cpp_branch="cpp-sweep-binary-io",
            cpp_build_modules="GCC/13.2.0",
        )
        config_kwargs.update(overrides)
        config = BatchConfig(**config_kwargs)

        transport = MagicMock()
        transport.upload.return_value = None
        return HPCJobManager(config=config, transport=transport), transport

    @staticmethod
    def _exec_side_effect(binary_ok=True, dirty=False, build_rc=0):
        """Classify commands by content; return appropriate stub results."""

        def side_effect(cmd, *args, **kwargs):
            if "git status --porcelain" in cmd:
                if dirty:
                    return (1, "HPC checkout has uncommitted changes:\n M qsp_sim.cpp")
                return (0, "")
            if "cmake --build" in cmd:
                return (build_rc, "" if build_rc == 0 else "make failed")
            if "echo OK" in cmd:
                return (0, "OK") if binary_ok else (1, "missing")
            return (0, "")

        return side_effect

    def test_happy_path_runs_all_three_steps(self):
        manager, transport = self._make_manager()
        transport.exec.side_effect = self._exec_side_effect()

        manager.ensure_cpp_binary()

        commands = [call.args[0] for call in transport.exec.call_args_list]
        assert any("git fetch origin" in c for c in commands), "should fetch"
        assert any("cmake --build" in c for c in commands), "should build"
        assert any("test -x" in c and "echo OK" in c for c in commands), "should verify binary"

    def test_skip_git_pull_skips_fetch(self):
        manager, transport = self._make_manager()
        transport.exec.side_effect = self._exec_side_effect()

        manager.ensure_cpp_binary(skip_git_pull=True)

        commands = [call.args[0] for call in transport.exec.call_args_list]
        assert not any("git fetch" in c for c in commands), "should not pull"
        assert any("cmake --build" in c for c in commands), "should still build"

    def test_skip_build_only_verifies(self):
        manager, transport = self._make_manager()
        transport.exec.side_effect = self._exec_side_effect()

        manager.ensure_cpp_binary(skip_git_pull=True, skip_build=True)

        commands = [call.args[0] for call in transport.exec.call_args_list]
        assert not any("cmake --build" in c for c in commands), "should not build"
        assert any("echo OK" in c for c in commands), "should still verify"

    def test_dirty_worktree_raises(self):
        manager, transport = self._make_manager()
        transport.exec.side_effect = self._exec_side_effect(dirty=True)

        with pytest.raises(RuntimeError, match="git pull failed"):
            manager.ensure_cpp_binary()

    def test_build_failure_raises(self):
        manager, transport = self._make_manager()
        transport.exec.side_effect = self._exec_side_effect(build_rc=2)

        with pytest.raises(RuntimeError, match="qsp_sim build failed"):
            manager.ensure_cpp_binary()

    def test_missing_binary_raises(self):
        manager, transport = self._make_manager()
        transport.exec.side_effect = self._exec_side_effect(binary_ok=False)

        with pytest.raises(RuntimeError, match="not found or not executable"):
            manager.ensure_cpp_binary(skip_git_pull=True, skip_build=True)

    def test_short_binary_path_needs_explicit_repo_path(self):
        manager, transport = self._make_manager(
            cpp_binary_path="/usr/bin/qsp_sim", cpp_repo_path=""
        )

        with pytest.raises(ValueError, match="Cannot derive cpp.repo_path"):
            manager.ensure_cpp_binary()

    def test_branch_override_reaches_git_commands(self):
        manager, transport = self._make_manager(cpp_branch="main")
        transport.exec.side_effect = self._exec_side_effect()

        manager.ensure_cpp_binary(branch="my-feature-branch")

        commands = [call.args[0] for call in transport.exec.call_args_list]
        pull_cmd = next(c for c in commands if "git pull" in c)
        assert "my-feature-branch" in pull_cmd
        assert "main" not in pull_cmd
