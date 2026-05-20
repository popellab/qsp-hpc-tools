"""Tests for :mod:`qsp_hpc.cpp.evolve_pack` — the per-task QSEP pack format.

Pure-Python: synthetic QSTH blobs exercise the writer/reader without a
qsp_sim binary. The blob payload is opaque to the pack, so the bytes only
need a valid QSTH header (mirrors ``_pack_qsth`` in test_cpp_evolve_cache).
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from qsp_hpc.cpp.evolve_cache import (
    _QSTH_HASH_LEN,
    _QSTH_HEADER_SIZE,
    _QSTH_MAGIC,
    _QSTH_VERSION,
    wire_hash,
)
from qsp_hpc.cpp.evolve_pack import (
    EvolveStatePackError,
    EvolveStatePackReader,
    EvolveStatePackWriter,
    write_evolve_pack,
)

# --- synthetic QSTH blob helpers -------------------------------------------


def _pack_qsth(
    *,
    params_hash: str,
    n_species: int = 164,
    t_diag: float = 1450.0,
    vt_diameter: float = 3.2,
    magic: int = _QSTH_MAGIC,
    version: int = _QSTH_VERSION,
    trailing: bytes = b"\x00" * 96,
) -> bytes:
    """A synthetic QSTH blob: 128-byte header + opaque payload."""
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


def _theta_hash(seed: str) -> str:
    """A deterministic 64-char sha256 hex theta_hash for tests."""
    return hashlib.sha256(seed.encode()).hexdigest()


def _blob_for(theta_hash: str, **kw) -> bytes:
    """A QSTH blob whose header params_hash matches wire_hash(theta_hash)."""
    return _pack_qsth(params_hash=wire_hash(theta_hash), **kw)


def _entries(n: int) -> list[tuple[str, bytes]]:
    out = []
    for i in range(n):
        th = _theta_hash(f"theta-{i}")
        # vary payload length so offset bookkeeping is actually exercised
        out.append((th, _blob_for(th, trailing=bytes([i % 251]) * (96 + 8 * i))))
    return out


# --- round trip ------------------------------------------------------------


def test_round_trip_single(tmp_path: Path):
    th, blob = _entries(1)[0]
    pack = write_evolve_pack(tmp_path / "t.qsep", [(th, blob)])
    r = EvolveStatePackReader(pack)
    assert len(r) == 1
    assert th in r
    assert r.get(th) == blob


def test_round_trip_many_varied_sizes(tmp_path: Path):
    entries = _entries(12)
    write_evolve_pack(tmp_path / "t.qsep", entries)
    r = EvolveStatePackReader(tmp_path / "t.qsep")
    assert len(r) == 12
    for th, blob in entries:
        assert r.get(th) == blob, f"blob mismatch for {th[:16]}"
    assert set(r.theta_hashes) == {th for th, _ in entries}


def test_empty_pack_round_trips(tmp_path: Path):
    pack = EvolveStatePackWriter().write(tmp_path / "empty.qsep")
    r = EvolveStatePackReader(pack)
    assert len(r) == 0
    assert r.theta_hashes == []
    assert r.get(_theta_hash("absent")) is None


def test_writer_len_and_contains(tmp_path: Path):
    w = EvolveStatePackWriter()
    entries = _entries(3)
    for th, blob in entries:
        w.add(th, blob)
    assert len(w) == 3
    assert entries[0][0] in w
    assert _theta_hash("nope") not in w


def test_duplicate_theta_hash_last_wins(tmp_path: Path):
    th = _theta_hash("dup")
    b1 = _blob_for(th, trailing=b"\x01" * 96)
    b2 = _blob_for(th, trailing=b"\x02" * 96)
    w = EvolveStatePackWriter()
    w.add(th, b1)
    w.add(th, b2)
    assert len(w) == 1
    r = EvolveStatePackReader(w.write(tmp_path / "d.qsep"))
    assert r.get(th) == b2


# --- get / materialize / missing -------------------------------------------


def test_get_missing_returns_none(tmp_path: Path):
    entries = _entries(2)
    write_evolve_pack(tmp_path / "t.qsep", entries)
    r = EvolveStatePackReader(tmp_path / "t.qsep")
    assert r.get(_theta_hash("not-in-pack")) is None


def test_materialize_writes_blob_to_disk(tmp_path: Path):
    th, blob = _entries(1)[0]
    write_evolve_pack(tmp_path / "t.qsep", [(th, blob)])
    r = EvolveStatePackReader(tmp_path / "t.qsep")
    out = r.materialize(th, tmp_path / "wd")
    assert out.exists()
    assert out.read_bytes() == blob
    assert out.parent == tmp_path / "wd"


def test_materialize_missing_raises_keyerror(tmp_path: Path):
    write_evolve_pack(tmp_path / "t.qsep", _entries(1))
    r = EvolveStatePackReader(tmp_path / "t.qsep")
    with pytest.raises(KeyError, match="not in pack"):
        r.materialize(_theta_hash("absent"), tmp_path / "wd")


# --- multi-pack layering ---------------------------------------------------


def test_from_dir_layers_all_packs(tmp_path: Path):
    a = _entries(3)
    b = _entries(3)  # disjoint seeds? no — _entries(n) reuses seeds 0..n-1
    # make b disjoint
    b = [(_theta_hash(f"b-{i}"), _blob_for(_theta_hash(f"b-{i}"))) for i in range(4)]
    write_evolve_pack(tmp_path / "chunk_000.qsep", a)
    write_evolve_pack(tmp_path / "chunk_001.qsep", b)
    r = EvolveStatePackReader.from_dir(tmp_path)
    assert len(r) == 7
    assert len(r.packs) == 2
    for th, blob in a + b:
        assert r.get(th) == blob


def test_add_pack_later_wins_on_duplicate(tmp_path: Path):
    th = _theta_hash("shared")
    early = _blob_for(th, trailing=b"\xaa" * 96)
    late = _blob_for(th, trailing=b"\xbb" * 96)
    write_evolve_pack(tmp_path / "p0.qsep", [(th, early)])
    write_evolve_pack(tmp_path / "p1.qsep", [(th, late)])
    r = EvolveStatePackReader(tmp_path / "p0.qsep")
    r.add_pack(tmp_path / "p1.qsep")
    assert r.get(th) == late


def test_from_dir_empty_dir(tmp_path: Path):
    r = EvolveStatePackReader.from_dir(tmp_path)
    assert len(r) == 0
    assert r.packs == []


# --- atomic write ----------------------------------------------------------


def test_write_is_atomic_no_tmp_left(tmp_path: Path):
    write_evolve_pack(tmp_path / "t.qsep", _entries(3))
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"atomic write left temp files: {leftovers}"


def test_write_creates_parent_dirs(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "t.qsep"
    write_evolve_pack(nested, _entries(2))
    assert nested.exists()


# --- add() validation ------------------------------------------------------


def test_add_rejects_non_qsth_blob(tmp_path: Path):
    w = EvolveStatePackWriter()
    with pytest.raises(EvolveStatePackError, match="not a valid QSTH blob"):
        w.add(_theta_hash("x"), b"not a qsth blob at all")


def test_add_rejects_params_hash_mismatch(tmp_path: Path):
    th = _theta_hash("real")
    wrong = _blob_for(_theta_hash("other"))  # header hash belongs to a different theta
    w = EvolveStatePackWriter()
    with pytest.raises(EvolveStatePackError, match="filed under the wrong key"):
        w.add(th, wrong)


def test_add_rejects_bad_theta_hash_length(tmp_path: Path):
    w = EvolveStatePackWriter()
    with pytest.raises(EvolveStatePackError, match="64 hex chars"):
        w.add("deadbeef", _blob_for(_theta_hash("x")))


def test_add_rejects_non_hex_theta_hash(tmp_path: Path):
    w = EvolveStatePackWriter()
    with pytest.raises(EvolveStatePackError, match="not hex"):
        w.add("z" * 64, _blob_for(_theta_hash("x")))


def test_add_validate_false_accepts_arbitrary_bytes(tmp_path: Path):
    th = _theta_hash("x")
    w = EvolveStatePackWriter()
    w.add(th, b"opaque bytes", validate=False)
    r = EvolveStatePackReader(w.write(tmp_path / "t.qsep"))
    assert r.get(th) == b"opaque bytes"


# --- malformed pack files --------------------------------------------------


def test_reader_rejects_truncated_file(tmp_path: Path):
    p = tmp_path / "short.qsep"
    p.write_bytes(b"\x00" * 8)  # < 32-byte footer
    with pytest.raises(EvolveStatePackError, match="footer"):
        EvolveStatePackReader(p)


def test_reader_rejects_bad_magic(tmp_path: Path):
    p = tmp_path / "t.qsep"
    write_evolve_pack(p, _entries(2))
    data = bytearray(p.read_bytes())
    data[-32:-28] = b"\xde\xad\xbe\xef"  # clobber footer magic
    p.write_bytes(data)
    with pytest.raises(EvolveStatePackError, match="bad magic"):
        EvolveStatePackReader(p)


def test_reader_rejects_bad_version(tmp_path: Path):
    p = tmp_path / "t.qsep"
    write_evolve_pack(p, _entries(2))
    data = bytearray(p.read_bytes())
    data[-28:-24] = struct.pack("<I", 999)  # clobber footer version
    p.write_bytes(data)
    with pytest.raises(EvolveStatePackError, match="version 999 unsupported"):
        EvolveStatePackReader(p)
