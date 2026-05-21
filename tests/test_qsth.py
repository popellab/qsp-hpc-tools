"""Tests for :mod:`qsp_hpc.cpp.qsth` — the QSTH blob header + theta hashing.

Pure-Python: hand-crafted QSTH blobs exercise the header parser; the hash
helpers are deterministic and binary-free.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from qsp_hpc.cpp.qsth import (
    _QSTH_HASH_LEN,
    _QSTH_HEADER_SIZE,
    _QSTH_MAGIC,
    _QSTH_VERSION,
    QsthHeader,
    QsthHeaderError,
    sha256_of_file,
    theta_hash_for_xml,
    wire_hash,
)

# --- synthetic QSTH blob ---------------------------------------------------


def _pack_qsth(
    n_species: int = 164,
    t_diag: float = 857.0,
    vt_diameter: float = 3.2,
    params_hash: str = "",
    magic: int = _QSTH_MAGIC,
    version: int = _QSTH_VERSION,
    trailing: bytes = b"",
) -> bytes:
    """Build a synthetic QSTH blob. ``trailing`` stands in for the opaque
    CVODEBase payload after the fixed 128-byte header."""
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


# --- hash helpers ----------------------------------------------------------


def test_theta_hash_for_xml_stable():
    assert theta_hash_for_xml(b"foo") == theta_hash_for_xml(b"foo")
    assert theta_hash_for_xml(b"foo") != theta_hash_for_xml(b"bar")
    assert len(theta_hash_for_xml(b"foo")) == 64  # SHA-256 hex


def test_wire_hash_truncates():
    full = "a" * 64
    assert wire_hash(full) == "a" * _QSTH_HASH_LEN
    assert len(wire_hash(full)) == _QSTH_HASH_LEN


def test_wire_hash_rejects_short():
    with pytest.raises(ValueError, match="too short"):
        wire_hash("a" * 10)


def test_sha256_of_file(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"the quick brown fox")
    full = sha256_of_file(p)
    assert full == hashlib.sha256(b"the quick brown fox").hexdigest()
    assert len(full) == 64
    assert sha256_of_file(p, truncate=8) == full[:8]


# --- QSTH header parser ----------------------------------------------------


def test_qsth_header_parse_ok(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(_pack_qsth(params_hash="abc123"))
    h = QsthHeader.parse(path)
    assert h.version == _QSTH_VERSION
    assert h.n_species_var == 164
    assert h.t_diagnosis_days == 857.0
    assert h.vt_diameter_cm == 3.2
    assert h.params_hash == "abc123"


def test_qsth_header_parse_bytes_with_payload():
    """A blob with a trailing payload still parses — only the header is read."""
    blob = _pack_qsth(params_hash="deadbeef", trailing=b"\x07" * 4096)
    h = QsthHeader.parse_bytes(blob, source="<test>")
    assert h.params_hash == "deadbeef"


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


def test_qsth_header_missing_file(tmp_path: Path):
    with pytest.raises(QsthHeaderError, match="cannot read"):
        QsthHeader.parse(tmp_path / "does-not-exist.bin")
