"""Tests for qsp_hpc.cpp.evolve_cache.

Pure-Python tests use hand-crafted QSTH blobs to exercise the header
parser + cache invalidation logic. Integration tests run the real
qsp_sim binary when available, proving that a blob produced by
``--dump-state`` can be consumed by ``--initial-state`` to get
byte-identical trajectories.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from qsp_hpc.cpp.evolve_cache import (
    _QSTH_HASH_LEN,
    _QSTH_HEADER_SIZE,
    _QSTH_MAGIC,
    _QSTH_VERSION,
    CppEvolveCache,
    QsthHeader,
    QsthHeaderError,
    theta_hash_for_xml,
    wire_hash,
)

# --- Pure-Python: hash helpers --------------------------------------------


def test_theta_hash_for_xml_stable():
    assert theta_hash_for_xml(b"foo") == theta_hash_for_xml(b"foo")
    assert theta_hash_for_xml(b"foo") != theta_hash_for_xml(b"bar")
    # SHA-256 hex
    assert len(theta_hash_for_xml(b"foo")) == 64


def test_wire_hash_truncates():
    full = "a" * 64
    assert wire_hash(full) == "a" * _QSTH_HASH_LEN
    assert len(wire_hash(full)) == _QSTH_HASH_LEN


def test_wire_hash_rejects_short():
    with pytest.raises(ValueError, match="too short"):
        wire_hash("a" * 10)


# --- QSTH header parser ----------------------------------------------------


def _pack_qsth(
    n_species: int = 164,
    t_diag: float = 857.0,
    vt_diameter: float = 3.2,
    params_hash: str = "",
    magic: int = _QSTH_MAGIC,
    version: int = _QSTH_VERSION,
    trailing: bytes = b"",
) -> bytes:
    """Build a synthetic QSTH blob header. ``trailing`` is appended after
    the fixed header (the real C++ payload would go here)."""
    hash_bytes = params_hash.encode("ascii")[:_QSTH_HASH_LEN].ljust(_QSTH_HASH_LEN, b"\x00")
    fixed = struct.pack(
        f"<IIQdd{_QSTH_HASH_LEN}s",
        magic,
        version,
        n_species,
        t_diag,
        vt_diameter,
        hash_bytes,
    )
    return fixed + b"\x00" * (_QSTH_HEADER_SIZE - len(fixed)) + trailing


def test_qsth_header_parse_ok(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(_pack_qsth(params_hash="abc123"))
    h = QsthHeader.parse(path)
    assert h.version == _QSTH_VERSION
    assert h.n_species_var == 164
    assert h.t_diagnosis_days == 857.0
    assert h.vt_diameter_cm == 3.2
    assert h.params_hash == "abc123"


def test_qsth_header_bad_magic(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(_pack_qsth(magic=0xDEADBEEF))
    with pytest.raises(QsthHeaderError, match="bad magic"):
        QsthHeader.parse(path)


def test_qsth_header_bad_version(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(_pack_qsth(version=99))
    with pytest.raises(QsthHeaderError, match="version 99"):
        QsthHeader.parse(path)


def test_qsth_header_truncated(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(_pack_qsth()[: _QSTH_HEADER_SIZE - 10])
    with pytest.raises(QsthHeaderError, match="truncated"):
        QsthHeader.parse(path)


# --- Integration with the real qsp_sim binary -----------------------------
#
# Shared helpers with test_cpp_runner.py. Looks for a prebuilt qsp_sim in
# sibling SPQSP_PDAC checkouts or via QSP_SIM_BINARY. Tests skip cleanly
# if nothing is available (keeps CI green when the C++ side isn't built).


def _real_binary_path() -> Path | None:
    env = os.environ.get("QSP_SIM_BINARY")
    if env and Path(env).exists():
        return Path(env)
    here = Path(__file__).resolve().parent.parent
    # Prefer the M13 worktree — tests need the --dump-state flag which
    # hasn't landed on the other branches yet.
    for sibling in ("SPQSP_PDAC-m13", "SPQSP_PDAC", "SPQSP_PDAC-cpp-sweep"):
        c = here.parent / sibling / "PDAC" / "qsp" / "sim" / "build" / "qsp_sim"
        if c.exists():
            return c
    return None


def _real_template_path() -> Path | None:
    here = Path(__file__).resolve().parent.parent
    # Prefer the M13 worktree — tests need the --dump-state flag which
    # hasn't landed on the other branches yet.
    for sibling in ("SPQSP_PDAC-m13", "SPQSP_PDAC", "SPQSP_PDAC-cpp-sweep"):
        c = here.parent / sibling / "PDAC" / "sim" / "resource" / "param_all.xml"
        if c.exists():
            return c
    return None


def _real_healthy_yaml() -> Path | None:
    here = Path(__file__).resolve().parent.parent
    # Prefer the M13 worktree — tests need the --dump-state flag which
    # hasn't landed on the other branches yet.
    for sibling in ("SPQSP_PDAC-m13", "SPQSP_PDAC", "SPQSP_PDAC-cpp-sweep"):
        c = here.parent / sibling / "PDAC" / "sim" / "resource" / "healthy_state.yaml"
        if c.exists():
            return c
    return None


_SKIP_INTEGRATION = (
    _real_binary_path() is None or _real_template_path() is None or _real_healthy_yaml() is None
)
_SKIP_REASON = (
    "qsp_sim binary / param template / healthy_state.yaml not found; "
    "build SPQSP_PDAC first or set QSP_SIM_BINARY"
)


@pytest.fixture
def real_runner():
    from qsp_hpc.cpp.runner import CppRunner

    return CppRunner(
        binary_path=_real_binary_path(),
        template_path=_real_template_path(),
        healthy_state_yaml=_real_healthy_yaml(),
        default_timeout_s=60.0,
    )


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_get_or_build_miss_then_hit(real_runner, tmp_path: Path):
    """First get_or_build is a miss (builds the blob); second is a hit.

    Each call materializes a fresh on-disk copy from LMDB, so the
    returned paths refer to files (not LMDB keys) — both must exist and
    parse cleanly, but path identity isn't part of the contract.
    """
    cache = CppEvolveCache(
        cache_root=tmp_path / "evolve_cache",
        renderer=real_runner._renderer,
        runner=real_runner,
    )
    assert cache.stats == {"hits": 0, "misses": 0}

    path1, th1 = cache.get_or_build({}, workdir=tmp_path)
    assert path1.exists()
    assert cache.stats == {"hits": 0, "misses": 1}

    path2, th2 = cache.get_or_build({}, workdir=tmp_path)
    assert path2.exists()
    assert th2 == th1
    assert cache.stats == {"hits": 1, "misses": 1}

    # Both materializations should parse to the same params_hash.
    header = QsthHeader.parse(path2)
    assert header.params_hash == wire_hash(th1)
    assert header.n_species_var > 0
    assert header.t_diagnosis_days > 0


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_different_thetas_produce_distinct_blobs(real_runner, tmp_path: Path):
    cache = CppEvolveCache(
        cache_root=tmp_path / "evolve_cache",
        renderer=real_runner._renderer,
        runner=real_runner,
    )
    path_a, hash_a = cache.get_or_build({}, workdir=tmp_path)
    # Pick a non-zero default so multiplying by 1.01 changes the XML bytes.
    # (Some params default to 0.0; 0 * 1.01 == 0 — same bytes, same hash.)
    nonzero = {name: val for name, val in real_runner.template_defaults.items() if val != 0.0}
    assert nonzero, "template has no non-zero defaults; test needs rework"
    some_param = next(iter(nonzero))
    bumped = {some_param: nonzero[some_param] * 1.01}
    path_b, hash_b = cache.get_or_build(bumped, workdir=tmp_path)
    assert hash_a != hash_b
    assert path_a != path_b
    assert path_a.exists() and path_b.exists()


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_corrupt_blob_is_rebuilt(real_runner, tmp_path: Path):
    """A bad-magic blob in LMDB triggers a rebuild on next get_or_build."""
    cache = CppEvolveCache(
        cache_root=tmp_path / "evolve_cache",
        renderer=real_runner._renderer,
        runner=real_runner,
    )
    path, th = cache.get_or_build({}, workdir=tmp_path)
    # Reach into the LMDB env and clobber the stored magic bytes.
    key = th[:16].encode("ascii")
    with cache._env.begin(write=True) as txn:
        blob = bytearray(txn.get(key))
        assert blob, "blob should have been stored on miss"
        blob[:4] = b"\x00\x00\x00\x00"
        txn.put(key, bytes(blob), overwrite=True)

    # Next call rebuilds instead of using the corrupt blob.
    misses_before = cache.stats["misses"]
    path2, th2 = cache.get_or_build({}, workdir=tmp_path)
    assert th2 == th
    assert cache.stats["misses"] == misses_before + 1
    # Header should parse again now that it's been rebuilt.
    QsthHeader.parse(path2)
    with cache._env.begin() as txn:
        rebuilt = txn.get(key)
    assert rebuilt[:4] == b"HQTS"  # ASCII bytes of _QSTH_MAGIC (LE)


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_run_one_with_cached_state_matches_fresh_evolve(real_runner, tmp_path: Path):
    """Byte-identical trajectories: --initial-state vs full --evolve-to-diagnosis."""
    cache = CppEvolveCache(
        cache_root=tmp_path / "evolve_cache",
        renderer=real_runner._renderer,
        runner=real_runner,
    )
    path, th = cache.get_or_build({}, workdir=tmp_path)

    cached = real_runner.run_one(
        params={},
        t_end_days=1.0,
        dt_days=0.1,
        workdir=tmp_path,
        evolve_state_path=path,
        params_hash=th,
    )
    fresh = real_runner.run_one(
        params={},
        t_end_days=1.0,
        dt_days=0.1,
        workdir=tmp_path,
    )
    np.testing.assert_array_equal(cached.trajectory, fresh.trajectory)


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_run_one_requires_params_hash_with_initial_state(real_runner, tmp_path: Path):
    """evolve_state_path without params_hash is a usage error."""
    # Make a blob so the path points at something valid.
    cache = CppEvolveCache(
        cache_root=tmp_path / "evolve_cache",
        renderer=real_runner._renderer,
        runner=real_runner,
    )
    path, _ = cache.get_or_build({}, workdir=tmp_path)
    with pytest.raises(ValueError, match="params_hash is required"):
        real_runner.run_one(
            params={},
            t_end_days=1.0,
            dt_days=0.1,
            workdir=tmp_path,
            evolve_state_path=path,  # no params_hash
        )


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_wrong_params_hash_is_rejected(real_runner, tmp_path: Path):
    """qsp_sim --initial-state refuses a mismatched --params-hash."""
    from qsp_hpc.cpp.runner import QspSimError

    cache = CppEvolveCache(
        cache_root=tmp_path / "evolve_cache",
        renderer=real_runner._renderer,
        runner=real_runner,
    )
    path, _ = cache.get_or_build({}, workdir=tmp_path)
    with pytest.raises(QspSimError, match="params_hash mismatch"):
        real_runner.run_one(
            params={},
            t_end_days=1.0,
            dt_days=0.1,
            workdir=tmp_path,
            evolve_state_path=path,
            params_hash="f" * 32,  # definitely not the right hash
        )


# --- Batch-runner integration ---------------------------------------------


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_batch_runner_shares_cache_across_scenarios(tmp_path: Path):
    """Two batches with the same theta under different scenarios should
    produce exactly one evolve-cache blob (the second batch hits)."""
    from qsp_hpc.cpp.batch_runner import CppBatchRunner

    cache_root = tmp_path / "evolve_cache"
    br = CppBatchRunner(
        binary_path=_real_binary_path(),
        template_path=_real_template_path(),
        healthy_state_yaml=_real_healthy_yaml(),
        default_timeout_s=60.0,
        evolve_cache_root=cache_root,
    )

    # One theta (template defaults), two sims per batch.
    theta = np.empty((2, 0))

    r1 = br.run(
        theta_matrix=theta,
        param_names=[],
        t_end_days=1.0,
        dt_days=0.1,
        output_path=tmp_path / "batch_baseline.parquet",
        scenario="baseline",
        max_workers=1,
    )
    r2 = br.run(
        theta_matrix=theta,
        param_names=[],
        t_end_days=1.0,
        dt_days=0.1,
        output_path=tmp_path / "batch_treatment.parquet",
        scenario="treatment",
        max_workers=1,
    )
    assert r1.n_failed == 0 and r2.n_failed == 0

    # Exactly one LMDB env should exist (one model/healthy_state pair),
    # holding exactly one entry (the single shared theta).
    import lmdb

    env_dirs = [p.parent for p in cache_root.rglob("data.mdb")]
    assert len(env_dirs) == 1, f"expected one LMDB env dir, got {env_dirs}"
    env = lmdb.open(str(env_dirs[0]), readonly=True, subdir=True, lock=False)
    try:
        with env.begin() as txn:
            n_entries = txn.stat()["entries"]
    finally:
        env.close()
    assert n_entries == 1, f"expected one shared evolve blob, got {n_entries}"

    # Per-theta legacy .state.bin files should no longer be created in
    # the cache root (they may exist transiently in worker workdirs as
    # materialized copies, but those live elsewhere).
    legacy = list(cache_root.rglob("*.state.bin"))
    assert legacy == [], f"unexpected legacy blob files in cache: {legacy}"


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_batch_runner_warns_when_cache_requested_without_healthy_state(tmp_path: Path, caplog):
    """Regression guard for #34: passing evolve_cache_root without
    healthy_state_yaml used to DEBUG-log and silently disable. The fix
    upgrades to WARNING so the mismatch surfaces even when the caller
    runs at default log level (e.g. HPC workers where the smoke test
    found 0 blobs written with no warning in .err)."""
    import logging

    from qsp_hpc.cpp.batch_runner import CppBatchRunner

    cache_root = tmp_path / "evolve_cache"
    with caplog.at_level(logging.WARNING, logger="qsp_hpc.cpp.batch_runner"):
        br = CppBatchRunner(
            binary_path=_real_binary_path(),
            template_path=_real_template_path(),
            # Intentionally omit healthy_state_yaml.
            default_timeout_s=60.0,
            evolve_cache_root=cache_root,
        )
    assert br.evolve_cache_root is None
    assert any(
        "evolve_cache_root" in rec.message and "ignored" in rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
    ), f"expected a WARNING about evolve_cache_root being ignored; got {caplog.records}"
