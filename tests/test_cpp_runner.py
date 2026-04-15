"""Tests for qsp_hpc.cpp.runner.

Unit tests use a synthetic binary file and a fake qsp_sim (shell script)
so the suite runs without SPQSP_PDAC compiled. The integration test at
the bottom runs the real qsp_sim if available.
"""

from __future__ import annotations

import os
import struct
import textwrap
from pathlib import Path

import numpy as np
import pytest

from qsp_hpc.cpp.param_xml import ParamNotFoundError
from qsp_hpc.cpp.runner import (
    BinaryFormatError,
    CppRunner,
    QspSimError,
    SimResult,
    read_binary_trajectory,
)

MINI_TEMPLATE = b"""<Param>
  <QSP>
    <A>1.0</A>
    <B>2.0</B>
  </QSP>
</Param>
"""


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    p = tmp_path / "template.xml"
    p.write_bytes(MINI_TEMPLATE)
    return p


# --- Binary-format parser tests -------------------------------------------


def _pack_binary(
    traj: np.ndarray, dt: float, t_end: float, magic: int = 0x51535042, version: int = 1
) -> bytes:
    n_t, n_sp = traj.shape
    header = struct.pack("<IIQQdd", magic, version, n_t, n_sp, dt, t_end)
    return header + traj.astype("<f8").tobytes()


def test_read_binary_trajectory_roundtrip(tmp_path: Path):
    traj = np.arange(24, dtype="f8").reshape(3, 8)  # 3 times × 8 species
    p = tmp_path / "ok.bin"
    p.write_bytes(_pack_binary(traj, dt=0.1, t_end=0.2))
    out, dt, t_end = read_binary_trajectory(p)
    np.testing.assert_array_equal(out, traj)
    assert dt == 0.1
    assert t_end == 0.2


def test_read_binary_bad_magic(tmp_path: Path):
    p = tmp_path / "bad.bin"
    p.write_bytes(_pack_binary(np.zeros((1, 1)), 1.0, 1.0, magic=0xDEADBEEF))
    with pytest.raises(BinaryFormatError, match="magic"):
        read_binary_trajectory(p)


def test_read_binary_unsupported_version(tmp_path: Path):
    p = tmp_path / "v2.bin"
    p.write_bytes(_pack_binary(np.zeros((1, 1)), 1.0, 1.0, version=2))
    with pytest.raises(BinaryFormatError, match="version"):
        read_binary_trajectory(p)


def test_read_binary_truncated_header(tmp_path: Path):
    p = tmp_path / "short.bin"
    p.write_bytes(b"\x00" * 10)
    with pytest.raises(BinaryFormatError, match="truncated"):
        read_binary_trajectory(p)


def test_read_binary_size_mismatch(tmp_path: Path):
    # Header claims 3×2 doubles, but the payload is only 1 double.
    header = struct.pack("<IIQQdd", 0x51535042, 1, 3, 2, 0.1, 0.3)
    p = tmp_path / "short_body.bin"
    p.write_bytes(header + b"\x00" * 8)
    with pytest.raises(BinaryFormatError, match="size mismatch"):
        read_binary_trajectory(p)


# --- Fake qsp_sim shell script for runner behavior tests -----------------


def _make_fake_binary(tmp_path: Path, behavior: str) -> Path:
    """Make an executable that mimics qsp_sim. `behavior` selects one of
    several canned responses — see cases in the script below."""
    script = tmp_path / f"fake_qsp_sim_{behavior}.sh"
    script.write_text(
        textwrap.dedent(
            f"""\
        #!/usr/bin/env bash
        set -e
        BEHAVIOR={behavior}
        # Parse flags we care about.
        while [ $# -gt 0 ]; do
          case "$1" in
            --binary-out) BIN_OUT="$2"; shift 2 ;;
            --species-out) SP_OUT="$2"; shift 2 ;;
            --param) PARAM="$2"; shift 2 ;;
            --t-end-days) TEND="$2"; shift 2 ;;
            --dt-days) DT="$2"; shift 2 ;;
            *) shift ;;
          esac
        done

        case "$BEHAVIOR" in
          ok)
            # Write a tiny 2-time-point × 3-species binary using python.
            python3 - <<PY
        import struct
        header = struct.pack('<IIQQdd', 0x51535042, 1, 2, 3, float("$DT"), float("$TEND"))
        body = struct.pack('<6d', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            # Stored as row-major (2 rows × 3 cols).
        open("$BIN_OUT", 'wb').write(header + body)
        open("$SP_OUT", 'w').write("\\n".join(['spA', 'spB', 'spC']) + '\\n')
        PY
            ;;
          crash)
            echo "intentional crash" >&2
            exit 42
            ;;
          hang)
            # Busy-wait long enough to blow the test timeout.
            sleep 60
            ;;
          bad_binary)
            echo 0xdeadbeef > "$BIN_OUT"  # not a valid binary
            echo -e "spA\\nspB\\nspC" > "$SP_OUT"
            ;;
          species_mismatch)
            python3 - <<PY
        import struct
        # 2×3 trajectory but species list has 2 entries.
        header = struct.pack('<IIQQdd', 0x51535042, 1, 2, 3, float("$DT"), float("$TEND"))
        body = struct.pack('<6d', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        open("$BIN_OUT", 'wb').write(header + body)
        open("$SP_OUT", 'w').write("spA\\nspB\\n")
        PY
            ;;
        esac
    """
        )
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def fake_ok(tmp_path: Path) -> Path:
    return _make_fake_binary(tmp_path, "ok")


def test_runner_happy_path(tmp_path: Path, template_path: Path, fake_ok: Path):
    runner = CppRunner(fake_ok, template_path)
    result = runner.run_one(
        params={"A": 5.0},
        t_end_days=0.2,
        dt_days=0.1,
        workdir=tmp_path / "work",
    )
    assert isinstance(result, SimResult)
    assert result.trajectory.shape == (2, 3)
    assert result.species_names == ["spA", "spB", "spC"]
    assert result.dt_days == 0.1
    assert result.t_end_days == 0.2
    # time_days reconstructed from dt × i
    np.testing.assert_array_equal(result.time_days, np.array([0.0, 0.1]))
    # Species names are cached on the runner for subsequent calls.
    assert runner.species_names == ["spA", "spB", "spC"]


def test_runner_missing_binary_raises(tmp_path: Path, template_path: Path):
    with pytest.raises(FileNotFoundError):
        CppRunner(tmp_path / "nope", template_path)


def test_runner_unknown_param_raises(tmp_path: Path, template_path: Path, fake_ok: Path):
    runner = CppRunner(fake_ok, template_path)
    with pytest.raises(ParamNotFoundError):
        runner.run_one(
            params={"unknown_param": 1.0},
            t_end_days=1.0,
            dt_days=1.0,
            workdir=tmp_path / "work",
        )


def test_runner_nonzero_exit_preserves_xml(tmp_path: Path, template_path: Path):
    crash = _make_fake_binary(tmp_path, "crash")
    runner = CppRunner(crash, template_path)
    work = tmp_path / "work"
    with pytest.raises(QspSimError, match="exited 42"):
        runner.run_one(params={}, t_end_days=1.0, dt_days=1.0, workdir=work)
    stashed = list((work / "failed").glob("*.nonzero-exit.xml"))
    assert len(stashed) == 1, "failed XML should be preserved under workdir/failed/"


def test_runner_timeout_preserves_xml(tmp_path: Path, template_path: Path):
    hang = _make_fake_binary(tmp_path, "hang")
    runner = CppRunner(hang, template_path, default_timeout_s=0.5)
    work = tmp_path / "work"
    with pytest.raises(QspSimError, match="timed out"):
        runner.run_one(params={}, t_end_days=1.0, dt_days=1.0, workdir=work)
    stashed = list((work / "failed").glob("*.timeout.xml"))
    assert len(stashed) == 1


def test_runner_bad_binary_preserves_xml(tmp_path: Path, template_path: Path):
    bad = _make_fake_binary(tmp_path, "bad_binary")
    runner = CppRunner(bad, template_path)
    work = tmp_path / "work"
    with pytest.raises(QspSimError, match="unreadable binary"):
        runner.run_one(params={}, t_end_days=1.0, dt_days=1.0, workdir=work)
    stashed = list((work / "failed").glob("*.bad-binary.xml"))
    assert len(stashed) == 1


def test_runner_species_mismatch_raises(tmp_path: Path, template_path: Path):
    mismatch = _make_fake_binary(tmp_path, "species_mismatch")
    runner = CppRunner(mismatch, template_path)
    with pytest.raises(QspSimError, match="species-out line count"):
        runner.run_one(params={}, t_end_days=1.0, dt_days=1.0, workdir=tmp_path / "work")


def test_runner_cleans_up_files_on_success(tmp_path: Path, template_path: Path, fake_ok: Path):
    runner = CppRunner(fake_ok, template_path)
    work = tmp_path / "work"
    runner.run_one(params={}, t_end_days=0.2, dt_days=0.1, workdir=work, keep_files=False)
    # workdir exists but should contain no .xml/.bin/.species.txt from the sim
    remnants = [p.name for p in work.iterdir() if p.is_file()]
    assert not remnants, f"unexpected leftovers: {remnants}"


def test_runner_keep_files_true_leaves_artifacts(
    tmp_path: Path, template_path: Path, fake_ok: Path
):
    runner = CppRunner(fake_ok, template_path)
    work = tmp_path / "work"
    runner.run_one(params={}, t_end_days=0.2, dt_days=0.1, workdir=work, keep_files=True)
    remnants = sorted(p.suffix for p in work.iterdir() if p.is_file())
    assert ".xml" in remnants and ".bin" in remnants and ".txt" in remnants


# --- Integration with real qsp_sim (opt-in) -------------------------------


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
    reason="qsp_sim binary or param_all_test.xml not found; skipping integration test",
)
def test_real_qsp_sim_end_to_end(tmp_path: Path):
    """Run the real qsp_sim on the real template, spot-check output shape."""
    runner = CppRunner(_real_binary_path(), _real_template_path())
    result = runner.run_one(
        params={"k_C1_growth": 0.01},
        t_end_days=5.0,
        dt_days=1.0,
        workdir=tmp_path,
    )
    assert result.trajectory.shape == (6, 164)  # 5 days + t=0, 164 species
    assert len(result.species_names) == 164
    assert "V_T.C1" in result.species_names
    # First row is initial conditions; should be finite and non-NaN.
    assert np.all(np.isfinite(result.trajectory[0]))
