"""Tests for qsp_hpc.cpp.batch_runner.CppBatchRunner."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from qsp_hpc.cpp.batch_runner import (
    STATUS_FAILED,
    STATUS_OK,
    CppBatchRunner,
    FusedScenarioSpec,
    batch_filename,
)
from qsp_hpc.cpp.param_xml import ParamNotFoundError

# Template + fake qsp_sim built the same way as test_cpp_runner.py, but
# inlined here to keep tests independent and readable.

MINI_TEMPLATE = b"""<Param>
  <QSP>
    <A>1.0</A>
    <B>2.0</B>
    <C>3.0</C>
  </QSP>
</Param>
"""


def _make_fake_binary(tmp_path: Path, behavior: str) -> Path:
    """Writes a shell script that mimics qsp_sim. See test_cpp_runner.py for
    the other behaviors; here we need at least `ok` and `flaky` (fails on
    specific sim_ids)."""
    script = tmp_path / f"fake_qsp_sim_{behavior}.sh"
    script.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -e
        BEHAVIOR={behavior}
        while [ $# -gt 0 ]; do
          case "$1" in
            --binary-out) BIN_OUT="$2"; shift 2 ;;
            --species-out) SP_OUT="$2"; shift 2 ;;
            --compartments-out) COMP_OUT="$2"; shift 2 ;;
            --rules-out) RULES_OUT="$2"; shift 2 ;;
            --param) PARAM="$2"; shift 2 ;;
            --t-end-days) TEND="$2"; shift 2 ;;
            --min-cadence-hours) DT="$2"; shift 2 ;;
            *) shift ;;
          esac
        done

        case "$BEHAVIOR" in
          ok)
            python3 - <<PY
        import struct
        header = struct.pack('<IIQQQQdddQQ', 0x51535042, 3, 2, 3, 0, 0, float("$DT"), float("$TEND"), 0.0, 0, 0)
        body = struct.pack('<8d', 0.0, 1.0, 2.0, 3.0, 0.1, 4.0, 5.0, 6.0)
        open("$BIN_OUT", 'wb').write(header + body)
        open("$SP_OUT", 'w').write("spA\\nspB\\nspC\\n")
        open("$COMP_OUT", 'w').write('')
        open("$RULES_OUT", 'w').write('')
        PY
            ;;
          fail_on_bad_A)
            # Fail if the rendered XML contains <A>999.0</A>, else succeed.
            if grep -q '<A>999' "$PARAM"; then
              echo "intentional failure for poison param" >&2
              exit 7
            fi
            python3 - <<PY
        import struct
        header = struct.pack('<IIQQQQdddQQ', 0x51535042, 3, 2, 3, 0, 0, float("$DT"), float("$TEND"), 0.0, 0, 0)
        body = struct.pack('<8d', 0.0, 1.0, 2.0, 3.0, 0.1, 4.0, 5.0, 6.0)
        open("$BIN_OUT", 'wb').write(header + body)
        open("$SP_OUT", 'w').write("spA\\nspB\\nspC\\n")
        open("$COMP_OUT", 'w').write('')
        open("$RULES_OUT", 'w').write('')
        PY
            ;;
        esac
    """))
    script.chmod(0o755)
    return script


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    p = tmp_path / "template.xml"
    p.write_bytes(MINI_TEMPLATE)
    return p


@pytest.fixture
def ok_binary(tmp_path: Path) -> Path:
    return _make_fake_binary(tmp_path, "ok")


def test_batch_runner_happy_path(tmp_path: Path, template_path: Path, ok_binary: Path):
    runner = CppBatchRunner(ok_binary, template_path)
    theta = np.array([[1.1, 2.1], [1.2, 2.2], [1.3, 2.3]])
    out = tmp_path / "batch.parquet"
    result = runner.run(
        theta_matrix=theta,
        param_names=["A", "B"],
        t_end_days=0.2,
        min_cadence_hours=0.1,
        output_path=out,
        max_workers=2,
    )
    assert result.n_sims == 3
    assert result.n_failed == 0
    assert result.parquet_path == out
    assert result.species_names == ["spA", "spB", "spC"]
    assert result.n_times == 2

    table = pq.read_table(out)
    cols = table.column_names
    # sample_index leads (global theta-pool index — stamped on every row
    # for cross-scenario alignment), then local simulation_id, status, time.
    assert cols[:4] == ["sample_index", "simulation_id", "status", "time"]
    assert "param:A" in cols and "param:B" in cols
    # Species columns are list<double>.
    for sp in ("spA", "spB", "spC"):
        assert sp in cols
        assert str(table.schema.field(sp).type) == "list<element: double>"
    # All statuses OK.
    np.testing.assert_array_equal(table.column("status").to_numpy(), [0, 0, 0])
    # Simulation ids are zero-indexed and ordered.
    np.testing.assert_array_equal(table.column("simulation_id").to_numpy(), [0, 1, 2])
    # Param column values round-trip exactly.
    np.testing.assert_allclose(table.column("param:A").to_numpy(), [1.1, 1.2, 1.3])
    np.testing.assert_allclose(table.column("param:B").to_numpy(), [2.1, 2.2, 2.3])
    # Time axis has n_times entries for each row.
    for row_time in table.column("time").to_pylist():
        assert len(row_time) == 2
        assert row_time == [0.0, 0.1]
    # Species rows match the fake binary's (2,3) payload: row 0 = [1,2,3],
    # row 1 = [4,5,6]. So spA = [1, 4], spB = [2, 5], spC = [3, 6].
    assert table.column("spA").to_pylist()[0] == [1.0, 4.0]
    assert table.column("spB").to_pylist()[0] == [2.0, 5.0]
    assert table.column("spC").to_pylist()[0] == [3.0, 6.0]


def test_batch_runner_failed_sim_marked_with_nan(tmp_path: Path, template_path: Path):
    flaky = _make_fake_binary(tmp_path, "fail_on_bad_A")
    runner = CppBatchRunner(flaky, template_path)
    # sim_id 1 has A=999 → workers will exit 7 and raise QspSimError.
    theta = np.array([[1.0, 2.0], [999.0, 2.0], [1.5, 2.5]])
    out = tmp_path / "batch.parquet"
    result = runner.run(
        theta_matrix=theta,
        param_names=["A", "B"],
        t_end_days=0.2,
        min_cadence_hours=0.1,
        output_path=out,
        max_workers=2,
    )
    assert result.n_sims == 3
    assert result.n_failed == 1

    table = pq.read_table(out)
    statuses = table.column("status").to_numpy().tolist()
    assert statuses == [STATUS_OK, STATUS_FAILED, STATUS_OK]

    # Failed row's species column is NaN-filled; successful rows have
    # numeric payload.
    spa_rows = table.column("spA").to_pylist()
    assert not np.isnan(spa_rows[0][0])
    assert all(np.isnan(v) for v in spa_rows[1])
    assert not np.isnan(spa_rows[2][0])

    # Param columns are preserved for the failed row — downstream code
    # that correlates params with outcomes can still filter status==1.
    np.testing.assert_allclose(table.column("param:A").to_numpy(), [1.0, 999.0, 1.5])


def test_batch_runner_rejects_unknown_param_before_forking(
    tmp_path: Path, template_path: Path, ok_binary: Path
):
    runner = CppBatchRunner(ok_binary, template_path)
    theta = np.array([[1.0, 2.0]])
    with pytest.raises(ParamNotFoundError):
        runner.run(
            theta_matrix=theta,
            param_names=["A", "NOT_IN_TEMPLATE"],
            t_end_days=0.1,
            min_cadence_hours=0.05,
            output_path=tmp_path / "x.parquet",
        )


def test_batch_runner_shape_mismatch(tmp_path: Path, template_path: Path, ok_binary: Path):
    runner = CppBatchRunner(ok_binary, template_path)
    theta = np.zeros((2, 3))  # 3 cols but we'll pass only 2 names
    with pytest.raises(ValueError, match="columns but"):
        runner.run(
            theta_matrix=theta,
            param_names=["A", "B"],
            t_end_days=0.1,
            min_cadence_hours=0.05,
            output_path=tmp_path / "x.parquet",
        )


def test_batch_filename_format():
    from datetime import datetime

    ts = datetime(2026, 4, 15, 16, 30, 45)
    assert batch_filename(0, 100, "baseline", 42, timestamp=ts) == (
        "batch_000_20260415_163045_100sims_seed42.parquet"
    )


def test_batch_runner_writes_only_sampled_param_columns(
    tmp_path: Path, template_path: Path, ok_binary: Path
):
    """Thin parquets (#23): only sampled params appear as ``param:*``
    columns. Non-sampled template defaults live in pool_manifest.json
    (written by the caller at pool creation time) and are resolved by
    derive_test_stats_worker at cal-target eval time, not broadcast
    into every row."""
    runner = CppBatchRunner(ok_binary, template_path)
    # Vary only A; B is in the template but not sampled.
    theta = np.array([[1.1], [1.2], [1.3]])
    out = tmp_path / "batch.parquet"
    runner.run(
        theta_matrix=theta,
        param_names=["A"],
        t_end_days=0.2,
        min_cadence_hours=0.1,
        output_path=out,
        max_workers=2,
    )
    table = pq.read_table(out)
    # Sampled column present, varying per-sim.
    np.testing.assert_allclose(table.column("param:A").to_numpy(), [1.1, 1.2, 1.3])
    # Unsampled B is NOT broadcast — template default lives only in
    # pool_manifest.json, accessed by cal-target functions via the
    # manifest fallback in derive_test_stats_worker.
    assert "param:B" not in table.column_names


def test_write_pool_manifest_round_trip(tmp_path: Path):
    """write_pool_manifest is idempotent (concurrent array tasks safe);
    load_pool_manifest round-trips the content."""
    from qsp_hpc.cpp.batch_runner import (
        POOL_MANIFEST_FILENAME,
        POOL_MANIFEST_SCHEMA,
        load_pool_manifest,
        write_pool_manifest,
    )

    defaults = {"A": 1.5, "B": 2.0, "C": 3.14}
    sampled = ["A"]
    path = write_pool_manifest(tmp_path / "pool", defaults, sampled)
    assert path.name == POOL_MANIFEST_FILENAME
    loaded = load_pool_manifest(tmp_path / "pool")
    assert loaded["schema_version"] == POOL_MANIFEST_SCHEMA
    assert loaded["template_defaults"] == defaults
    assert loaded["sampled_params"] == sampled

    # Idempotent: second call doesn't overwrite. Concurrent array tasks
    # all racing to write the manifest produce one consistent file.
    write_pool_manifest(tmp_path / "pool", {"A": 999.0}, [])
    loaded2 = load_pool_manifest(tmp_path / "pool")
    assert loaded2["template_defaults"] == defaults


def _manifest_race_writer(args: tuple) -> int:
    """Module-level so ProcessPoolExecutor/spawn can pickle it.

    Each process builds an intentionally different payload — the race is
    visible if writers trample each other, because "last writer wins" on
    the target path would leave differing content across runs. Here all
    200 keys are the same shape so content IS identical; we're only
    asserting that no writer raises FileNotFoundError on its tmp→final
    rename."""
    from qsp_hpc.cpp.batch_runner import write_pool_manifest

    pool_dir, seed = args
    defaults = {f"k_{i}": float(seed + i) for i in range(200)}
    sampled = [f"k_{i}" for i in range(10)]
    try:
        write_pool_manifest(pool_dir, defaults, sampled)
        return 0
    except Exception as e:
        print(f"writer {seed} failed: {type(e).__name__}: {e}")
        return 1


def test_write_pool_manifest_concurrent_writers(tmp_path: Path):
    """Regression: SLURM array tasks racing to write the manifest must
    all succeed. Before the fix they used a single shared
    'pool_manifest.json.tmp' — task 0 renamed its tmp to the final path,
    leaving task 1 with a FileNotFoundError on its own tmp_path.replace().
    Caught in the SBI HPC smoke with a 2-task array."""
    import multiprocessing

    from qsp_hpc.cpp.batch_runner import load_pool_manifest

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()

    # 8 concurrent processes racing — matches a real SLURM array's
    # start-at-once cadence better than threads (separate PIDs, separate
    # os.replace calls). Spawn start method for portability (matches
    # Python 3.14+ default on Linux).
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        rcs = pool.map(_manifest_race_writer, [(pool_dir, seed) for seed in range(8)])

    assert all(rc == 0 for rc in rcs), f"some writers failed: {rcs}"

    # Exactly one manifest; no tmp residue.
    assert (pool_dir / "pool_manifest.json").exists()
    residue = sorted(pool_dir.glob("pool_manifest.json.tmp*"))
    assert residue == [], f"tmp files not cleaned up: {residue}"

    loaded = load_pool_manifest(pool_dir)
    assert "template_defaults" in loaded
    assert len(loaded["template_defaults"]) == 200


def test_load_pool_manifest_missing_returns_none(tmp_path: Path):
    """Pre-#23 pools have no manifest; readers treat None as 'no
    fallback needed' and look only at parquet columns (which carry
    the full param set in the wide layout)."""
    from qsp_hpc.cpp.batch_runner import load_pool_manifest

    assert load_pool_manifest(tmp_path / "nonexistent") is None
    (tmp_path / "empty").mkdir()
    assert load_pool_manifest(tmp_path / "empty") is None


def test_batch_runner_all_fail_raises(tmp_path: Path, template_path: Path):
    flaky = _make_fake_binary(tmp_path, "fail_on_bad_A")
    runner = CppBatchRunner(flaky, template_path)
    # Every sim has A=999 → every sim fails.
    theta = np.full((3, 2), [999.0, 2.0])
    from qsp_hpc.cpp.runner import QspSimError

    with pytest.raises(QspSimError, match="All 3 sims failed"):
        runner.run(
            theta_matrix=theta,
            param_names=["A", "B"],
            t_end_days=0.1,
            min_cadence_hours=0.05,
            output_path=tmp_path / "x.parquet",
        )


# --- Integration with real binary (opt-in, same gating as test_cpp_runner) -


def _real_binary_path() -> Path | None:
    env = os.environ.get("QSP_SIM_BINARY")
    if env and Path(env).exists():
        return Path(env)
    return None


def _real_template_path() -> Path | None:
    env = os.environ.get("QSP_SIM_TEMPLATE")
    if env and Path(env).exists():
        return Path(env)
    return None


@pytest.mark.skipif(
    _real_binary_path() is None or _real_template_path() is None,
    reason="qsp_sim binary or param_all_test.xml not found",
)
def test_real_batch_end_to_end(tmp_path: Path):
    """Run 3 real sims with varying k_C1_growth; check schema + signal."""
    runner = CppBatchRunner(_real_binary_path(), _real_template_path())
    theta = np.array([[0.001], [0.005], [0.01]])
    out = tmp_path / "real_batch.parquet"
    result = runner.run(
        theta_matrix=theta,
        param_names=["k_C1_growth"],
        t_end_days=5.0,
        min_cadence_hours=1.0,
        output_path=out,
        max_workers=2,
    )
    assert result.n_sims == 3
    assert result.n_failed == 0
    assert len(result.species_names) == 164
    # Under v3 CV_ONE_STEP cadence floor, n_times depends on solver
    # stepping. 1 h floor over 5 d gives ~120 dumps maximum; allow a
    # generous lower bound that captures any reasonable cadence.
    assert result.n_times >= 6

    table = pq.read_table(out)
    assert "V_T.C1" in table.column_names
    # Real signal: varying k_C1_growth by 10× should change V_T.C1 final
    # values meaningfully across rows.
    final_c1 = [row[-1] for row in table.column("V_T.C1").to_pylist()]
    assert final_c1[0] != final_c1[2], "V_T.C1 should differ across k_C1_growth values"


# --- Persistent evolve cache (#90) -----------------------------------------


def _real_healthy_yaml() -> Path | None:
    env = os.environ.get("QSP_SIM_HEALTHY_YAML")
    if env and Path(env).exists():
        return Path(env)
    return None


def test_evolve_cache_root_disabled_without_healthy_state(
    tmp_path: Path, template_path: Path, ok_binary: Path, caplog
):
    """evolve_cache_root is inert without a healthy_state_yaml — there is
    no evolve phase to cache. The runner WARNs and disables it rather than
    silently producing a cache that never engages (#34)."""
    import logging

    with caplog.at_level(logging.WARNING):
        runner = CppBatchRunner(
            ok_binary, template_path, evolve_cache_root=tmp_path / "evolve_cache"
        )
    assert runner.evolve_cache_root is None
    assert "no healthy_state_yaml" in caplog.text


def test_evolve_cache_root_kept_with_healthy_state(
    tmp_path: Path, template_path: Path, ok_binary: Path
):
    """With a healthy_state_yaml, evolve_cache_root is retained (resolved)."""
    healthy = tmp_path / "healthy.yaml"
    healthy.write_text("# dummy healthy state for the ctor probe\n")
    runner = CppBatchRunner(
        ok_binary,
        template_path,
        healthy_state_yaml=healthy,
        evolve_cache_root=tmp_path / "evolve_cache",
    )
    assert runner.evolve_cache_root == (tmp_path / "evolve_cache").resolve()


# --- Fused multi-scenario batch (#90 Phase 2) ------------------------------


def _make_fake_fused_binary(tmp_path: Path, behavior: str = "fused_ok") -> Path:
    """Fake qsp_sim for the fused path.

    ``--dump-state`` writes an opaque evolve blob and exits 0 (behavior
    ``fused_dump_fail`` exits nonzero instead). A scenario run writes a
    v3 trajectory whose ``spA`` row-0 value is read from the
    ``--scenario`` YAML's content — so a test can tell which scenario's
    trajectory landed in which parquet. A scenario YAML containing
    ``BAD`` makes that scenario's sim exit nonzero (the evolve still
    succeeds — only that one scenario fails).
    """
    script = tmp_path / f"fake_fused_{behavior}.sh"
    script.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -e
        BEHAVIOR={behavior}
        DUMP_STATE=""
        SCENARIO=""
        while [ $# -gt 0 ]; do
          case "$1" in
            --binary-out) BIN_OUT="$2"; shift 2 ;;
            --species-out) SP_OUT="$2"; shift 2 ;;
            --compartments-out) COMP_OUT="$2"; shift 2 ;;
            --rules-out) RULES_OUT="$2"; shift 2 ;;
            --param) PARAM="$2"; shift 2 ;;
            --t-end-days) TEND="$2"; shift 2 ;;
            --min-cadence-hours) DT="$2"; shift 2 ;;
            --dump-state) DUMP_STATE="$2"; shift 2 ;;
            --scenario) SCENARIO="$2"; shift 2 ;;
            *) shift ;;
          esac
        done

        if [ -n "$DUMP_STATE" ]; then
          if [ "$BEHAVIOR" = "fused_dump_fail" ]; then
            echo "evolve rejected" >&2
            exit 9
          fi
          printf 'FAKE_EVOLVE_STATE' > "$DUMP_STATE"
          exit 0
        fi

        SCEN_VAL=0.0
        if [ -n "$SCENARIO" ]; then
          RAW=$(cat "$SCENARIO")
          if [ "$RAW" = "BAD" ]; then
            echo "scenario sim failed" >&2
            exit 7
          fi
          SCEN_VAL="$RAW"
        fi

        python3 - <<PY
        import struct
        header = struct.pack('<IIQQQQdddQQ', 0x51535042, 3, 2, 3, 0, 0, float("$DT"), float("$TEND"), 0.0, 0, 0)
        body = struct.pack('<8d', 0.0, float("$SCEN_VAL"), 2.0, 3.0, 0.1, 4.0, 5.0, 6.0)
        open("$BIN_OUT", 'wb').write(header + body)
        open("$SP_OUT", 'w').write("spA\\nspB\\nspC\\n")
        open("$COMP_OUT", 'w').write('')
        open("$RULES_OUT", 'w').write('')
        PY
    """))
    script.chmod(0o755)
    return script


@pytest.fixture
def healthy_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "healthy.yaml"
    p.write_text("# dummy healthy state\n")
    return p


def _scenario_files(tmp_path: Path, name: str, marker: str) -> tuple[Path, Path]:
    """A (scenario_yaml, drug_metadata_yaml) pair; the scenario YAML's
    body is ``marker`` (the fake binary reads it as spA's row-0 value)."""
    scen = tmp_path / f"scenario_{name}.yaml"
    scen.write_text(marker)
    drug = tmp_path / f"drug_{name}.yaml"
    drug.write_text("# dummy drug metadata\n")
    return scen, drug


def test_run_fused_rejects_scenario_bound_runner(tmp_path, template_path, healthy_yaml):
    """A fused runner must be scenario-agnostic — per-scenario YAMLs ride
    on FusedScenarioSpec, not the runner."""
    scen, drug = _scenario_files(tmp_path, "x", "1")
    runner = CppBatchRunner(
        _make_fake_fused_binary(tmp_path),
        template_path,
        healthy_state_yaml=healthy_yaml,
        scenario_yaml=scen,
        drug_metadata_yaml=drug,
    )
    with pytest.raises(ValueError, match="scenario-agnostic"):
        runner.run_fused(
            theta_matrix=np.array([[1.0, 2.0]]),
            param_names=["A", "B"],
            t_end_days=0.2,
            min_cadence_hours=0.1,
            scenarios=[FusedScenarioSpec(name="x", output_path=tmp_path / "x.parquet")],
            sample_indices=np.array([0]),
        )


def test_run_fused_requires_healthy_state(tmp_path, template_path):
    """No healthy state → no burn-in → fusion has nothing to amortize."""
    runner = CppBatchRunner(_make_fake_fused_binary(tmp_path), template_path)
    with pytest.raises(ValueError, match="healthy_state_yaml"):
        runner.run_fused(
            theta_matrix=np.array([[1.0, 2.0]]),
            param_names=["A", "B"],
            t_end_days=0.2,
            min_cadence_hours=0.1,
            scenarios=[FusedScenarioSpec(name="x", output_path=tmp_path / "x.parquet")],
            sample_indices=np.array([0]),
        )


def test_run_fused_happy_path(tmp_path, template_path, healthy_yaml):
    """Three thetas × three scenarios — one evolve per theta, one parquet
    per scenario, each carrying its own scenario's trajectory."""
    runner = CppBatchRunner(
        _make_fake_fused_binary(tmp_path),
        template_path,
        healthy_state_yaml=healthy_yaml,
    )
    sa, da = _scenario_files(tmp_path, "A", "10.0")
    sb, db = _scenario_files(tmp_path, "B", "20.0")
    scenarios = [
        FusedScenarioSpec(
            name="drugA",
            output_path=tmp_path / "drugA.parquet",
            scenario_yaml=sa,
            drug_metadata_yaml=da,
        ),
        FusedScenarioSpec(
            name="drugB",
            output_path=tmp_path / "drugB.parquet",
            scenario_yaml=sb,
            drug_metadata_yaml=db,
        ),
        # An undosed scenario — no scenario YAML at all.
        FusedScenarioSpec(name="baseline", output_path=tmp_path / "baseline.parquet"),
    ]
    theta = np.array([[1.1, 2.1], [1.2, 2.2], [1.3, 2.3]])
    results = runner.run_fused(
        theta_matrix=theta,
        param_names=["A", "B"],
        t_end_days=0.2,
        min_cadence_hours=0.1,
        scenarios=scenarios,
        sample_indices=np.array([0, 1, 2]),
        max_workers=2,
    )
    assert len(results) == 3
    assert all(r is not None and r.n_sims == 3 and r.n_failed == 0 for r in results)

    # Each scenario's parquet carries that scenario's marker in spA row 0.
    for spec, marker in zip(scenarios, (10.0, 20.0, 0.0)):
        table = pq.read_table(spec.output_path)
        assert table.num_rows == 3
        np.testing.assert_array_equal(table.column("sample_index").to_numpy(), [0, 1, 2])
        for row in table.column("spA").to_pylist():
            assert row == [marker, 4.0]


def test_run_fused_per_scenario_deficit(tmp_path, template_path, healthy_yaml):
    """A scenario with a non-zero start_index only sims its deficit tail
    — its parquet carries just the thetas at sample_index >= start."""
    runner = CppBatchRunner(
        _make_fake_fused_binary(tmp_path),
        template_path,
        healthy_state_yaml=healthy_yaml,
    )
    sa, da = _scenario_files(tmp_path, "A", "1.0")
    sb, db = _scenario_files(tmp_path, "B", "2.0")
    scenarios = [
        FusedScenarioSpec(
            name="full",
            output_path=tmp_path / "full.parquet",
            scenario_yaml=sa,
            drug_metadata_yaml=da,
            start_index=0,
        ),
        FusedScenarioSpec(
            name="topup",
            output_path=tmp_path / "topup.parquet",
            scenario_yaml=sb,
            drug_metadata_yaml=db,
            start_index=2,
        ),
    ]
    results = runner.run_fused(
        theta_matrix=np.array([[1.1, 2.1], [1.2, 2.2], [1.3, 2.3]]),
        param_names=["A", "B"],
        t_end_days=0.2,
        min_cadence_hours=0.1,
        scenarios=scenarios,
        sample_indices=np.array([0, 1, 2]),
        max_workers=2,
    )
    full_res, topup_res = results
    assert full_res.n_sims == 3
    # topup only ran sample_index >= 2, i.e. just the last theta.
    assert topup_res.n_sims == 1
    topup_table = pq.read_table(tmp_path / "topup.parquet")
    assert topup_table.num_rows == 1
    np.testing.assert_array_equal(topup_table.column("sample_index").to_numpy(), [2])


def test_run_fused_scenario_excluded_when_no_thetas(tmp_path, template_path, healthy_yaml):
    """A scenario whose start_index excludes every theta in the chunk
    gets a None result (no parquet written)."""
    runner = CppBatchRunner(
        _make_fake_fused_binary(tmp_path),
        template_path,
        healthy_state_yaml=healthy_yaml,
    )
    sa, da = _scenario_files(tmp_path, "A", "1.0")
    sb, db = _scenario_files(tmp_path, "B", "2.0")
    scenarios = [
        FusedScenarioSpec(
            name="present",
            output_path=tmp_path / "present.parquet",
            scenario_yaml=sa,
            drug_metadata_yaml=da,
            start_index=0,
        ),
        FusedScenarioSpec(
            name="absent",
            output_path=tmp_path / "absent.parquet",
            scenario_yaml=sb,
            drug_metadata_yaml=db,
            start_index=99,
        ),
    ]
    results = runner.run_fused(
        theta_matrix=np.array([[1.1, 2.1], [1.2, 2.2]]),
        param_names=["A", "B"],
        t_end_days=0.2,
        min_cadence_hours=0.1,
        scenarios=scenarios,
        sample_indices=np.array([0, 1]),
        max_workers=2,
    )
    assert results[0] is not None and results[0].n_sims == 2
    assert results[1] is None
    assert not (tmp_path / "absent.parquet").exists()


def test_run_fused_one_scenario_fails_others_survive(tmp_path, template_path, healthy_yaml):
    """A scenario whose sim fails is marked failed for that scenario only
    — the shared evolve and the other scenarios are unaffected."""
    runner = CppBatchRunner(
        _make_fake_fused_binary(tmp_path),
        template_path,
        healthy_state_yaml=healthy_yaml,
    )
    sa, da = _scenario_files(tmp_path, "ok", "5.0")
    sbad, dbad = _scenario_files(tmp_path, "bad", "BAD")
    scenarios = [
        FusedScenarioSpec(
            name="ok",
            output_path=tmp_path / "ok.parquet",
            scenario_yaml=sa,
            drug_metadata_yaml=da,
        ),
        FusedScenarioSpec(
            name="bad",
            output_path=tmp_path / "bad.parquet",
            scenario_yaml=sbad,
            drug_metadata_yaml=dbad,
        ),
    ]
    ok_res, bad_res = runner.run_fused(
        theta_matrix=np.array([[1.1, 2.1], [1.2, 2.2]]),
        param_names=["A", "B"],
        t_end_days=0.2,
        min_cadence_hours=0.1,
        scenarios=scenarios,
        sample_indices=np.array([0, 1]),
        max_workers=2,
    )
    assert ok_res.n_failed == 0
    assert bad_res.n_failed == 2  # every sim of the bad scenario failed
    bad_table = pq.read_table(tmp_path / "bad.parquet")
    assert bad_table.column("status").to_pylist() == [STATUS_FAILED, STATUS_FAILED]


def test_run_fused_failed_evolve_fails_all_scenarios(tmp_path, template_path, healthy_yaml):
    """When --dump-state itself fails, every scenario for that theta is
    unrunnable — with no evolve state at all the batch raises."""
    runner = CppBatchRunner(
        _make_fake_fused_binary(tmp_path, behavior="fused_dump_fail"),
        template_path,
        healthy_state_yaml=healthy_yaml,
    )
    sa, da = _scenario_files(tmp_path, "A", "1.0")
    from qsp_hpc.cpp.runner import QspSimError

    with pytest.raises(QspSimError, match="All fused sims failed"):
        runner.run_fused(
            theta_matrix=np.array([[1.1, 2.1]]),
            param_names=["A", "B"],
            t_end_days=0.2,
            min_cadence_hours=0.1,
            scenarios=[
                FusedScenarioSpec(
                    name="A",
                    output_path=tmp_path / "A.parquet",
                    scenario_yaml=sa,
                    drug_metadata_yaml=da,
                )
            ],
            sample_indices=np.array([0]),
            max_workers=1,
        )


@pytest.mark.skipif(
    _real_binary_path() is None or _real_template_path() is None or _real_healthy_yaml() is None,
    reason="qsp_sim binary / template / healthy_state.yaml not found; set "
    "QSP_SIM_BINARY, QSP_SIM_TEMPLATE, QSP_SIM_HEALTHY_YAML",
)
def test_real_batch_writes_evolve_cache_shard(tmp_path: Path):
    """A cold batch with evolve_cache_root evolves every theta and writes
    one QSEP shard holding all the post-evolve states, keyed by theta."""
    from qsp_hpc.cpp.evolve_cache import EvolveCache

    cache_root = tmp_path / "evolve_cache"
    runner = CppBatchRunner(
        _real_binary_path(),
        _real_template_path(),
        healthy_state_yaml=_real_healthy_yaml(),
        evolve_cache_root=cache_root,
    )
    # 3 distinct thetas, all with k_C1_growth fast enough that
    # evolve_to_diagnosis reaches the 3.2 cm target (slow-growth values
    # get REJECTED before the burn-in finishes).
    theta = np.array([[0.4], [0.6], [0.8]])
    result = runner.run(
        theta_matrix=theta,
        param_names=["k_C1_growth"],
        t_end_days=1.0,
        min_cadence_hours=4.0,
        output_path=tmp_path / "batch.parquet",
        max_workers=2,
    )
    assert result.n_failed == 0
    assert result.evolve_shard_path is not None and result.evolve_shard_path.exists()

    # The shard lives in the (binary, healthy_state) namespace and holds
    # one evolve state per distinct theta.
    cache = EvolveCache.for_run(
        cache_root,
        binary_path=_real_binary_path(),
        healthy_state_yaml=_real_healthy_yaml(),
    )
    assert len(cache) == 3, "one evolve state per distinct theta"
    assert result.evolve_shard_path.parent == cache.namespace_dir


@pytest.mark.skipif(
    _real_binary_path() is None or _real_template_path() is None or _real_healthy_yaml() is None,
    reason="qsp_sim binary / template / healthy_state.yaml not found; set "
    "QSP_SIM_BINARY, QSP_SIM_TEMPLATE, QSP_SIM_HEALTHY_YAML",
)
def test_real_batch_reuses_evolve_cache(tmp_path: Path, caplog):
    """A second batch over the same thetas hits the cache, writes no new
    shard, and produces byte-identical scenario trajectories (same θ,
    same evolve state, same scenario)."""
    import logging

    binary, template, healthy = (
        _real_binary_path(),
        _real_template_path(),
        _real_healthy_yaml(),
    )
    cache_root = tmp_path / "evolve_cache"
    theta = np.array([[0.4], [0.6], [0.8]])
    kw = dict(
        param_names=["k_C1_growth"],
        t_end_days=1.0,
        min_cadence_hours=4.0,
        max_workers=2,
    )
    runner = CppBatchRunner(
        binary, template, healthy_state_yaml=healthy, evolve_cache_root=cache_root
    )

    # 1. Cold run — evolves every θ, writes a shard.
    cold = runner.run(theta_matrix=theta, output_path=tmp_path / "cold.parquet", **kw)
    assert cold.n_failed == 0 and cold.evolve_shard_path is not None

    # 2. Warm run — every θ hits the cache; no new shard.
    with caplog.at_level(logging.INFO):
        warm = runner.run(theta_matrix=theta, output_path=tmp_path / "warm.parquet", **kw)
    assert warm.n_failed == 0
    assert warm.evolve_shard_path is None, "a fully-cached run writes no shard"
    assert "all 3 theta(s) hit the cache" in caplog.text

    # 3. Correctness: the warm run reused the exact cached evolve states,
    # so the scenario trajectories must match the cold run bit-for-bit.
    cold_t = pq.read_table(tmp_path / "cold.parquet")
    warm_t = pq.read_table(tmp_path / "warm.parquet")
    for sp in ("V_T.C1",):
        for cr, wr in zip(cold_t.column(sp).to_pylist(), warm_t.column(sp).to_pylist()):
            np.testing.assert_array_equal(cr, wr)
