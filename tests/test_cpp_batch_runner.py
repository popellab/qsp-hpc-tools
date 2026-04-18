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
    script.write_text(
        textwrap.dedent(
            f"""\
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
            --dt-days) DT="$2"; shift 2 ;;
            *) shift ;;
          esac
        done

        case "$BEHAVIOR" in
          ok)
            python3 - <<PY
        import struct
        header = struct.pack('<IIQQQQdd', 0x51535042, 2, 2, 3, 0, 0, float("$DT"), float("$TEND"))
        body = struct.pack('<6d', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
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
        header = struct.pack('<IIQQQQdd', 0x51535042, 2, 2, 3, 0, 0, float("$DT"), float("$TEND"))
        body = struct.pack('<6d', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        open("$BIN_OUT", 'wb').write(header + body)
        open("$SP_OUT", 'w').write("spA\\nspB\\nspC\\n")
        open("$COMP_OUT", 'w').write('')
        open("$RULES_OUT", 'w').write('')
        PY
            ;;
        esac
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
        dt_days=0.1,
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
        dt_days=0.1,
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
            dt_days=0.05,
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
            dt_days=0.05,
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
        dt_days=0.1,
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
            dt_days=0.05,
            output_path=tmp_path / "x.parquet",
        )


# --- Integration with real binary (opt-in, same gating as test_cpp_runner) -


def _real_binary_path() -> Path | None:
    env = os.environ.get("QSP_SIM_BINARY")
    if env and Path(env).exists():
        return Path(env)
    here = Path(__file__).resolve().parent.parent
    for sibling in ("SPQSP_PDAC", "SPQSP_PDAC-cpp-sweep"):
        candidate = here.parent / sibling / "PDAC" / "qsp" / "sim" / "build" / "qsp_sim"
        if candidate.exists():
            return candidate
    return None


def _real_template_path() -> Path | None:
    env = os.environ.get("SPQSP_PDAC_ROOT")
    if env:
        c = Path(env) / "PDAC" / "sim" / "resource" / "param_all_test.xml"
        return c if c.exists() else None
    here = Path(__file__).resolve().parent.parent
    for sibling in ("SPQSP_PDAC", "SPQSP_PDAC-cpp-sweep"):
        c = here.parent / sibling / "PDAC" / "sim" / "resource" / "param_all_test.xml"
        if c.exists():
            return c
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
        dt_days=1.0,
        output_path=out,
        max_workers=2,
    )
    assert result.n_sims == 3
    assert result.n_failed == 0
    assert len(result.species_names) == 164
    assert result.n_times == 6  # 0..5 days at dt=1

    table = pq.read_table(out)
    assert "V_T.C1" in table.column_names
    # Real signal: varying k_C1_growth by 10× should change V_T.C1 final
    # values meaningfully across rows.
    final_c1 = [row[-1] for row in table.column("V_T.C1").to_pylist()]
    assert final_c1[0] != final_c1[2], "V_T.C1 should differ across k_C1_growth values"
