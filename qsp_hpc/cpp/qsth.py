"""QSTH evolve-state blob: header format + theta hashing.

A QSTH blob is one serialized post-``evolve_to_diagnosis`` ODE state, as
written by ``qsp_sim --dump-state`` and consumed by ``qsp_sim
--initial-state``. The blob is a fixed 128-byte little-endian header
followed by an opaque CVODEBase full-state payload (state vectors +
delay-event queue + trigger flags). Python only ever reads the header;
the payload round-trips verbatim.

This module is the shared, dependency-free bottom layer for the two
things that frame QSTH blobs:

- :mod:`qsp_hpc.cpp.evolve_pack` — the QSEP pack format (a set of blobs
  with a ``theta_hash`` index).
- :mod:`qsp_hpc.cpp.evolve_cache` — the persistent, namespaced cache
  built on top of QSEP shards.

Keeping the header parser + hash helpers here (rather than in either
consumer) breaks an import cycle: ``evolve_cache`` imports ``evolve_pack``
to reuse the pack format, and both import this module.

QSTH blob header
----------------
Fixed 128-byte little-endian header (mirrors ``write_qsth_header`` in
``qsp_sim.cpp``), followed by the opaque CVODEBase full-state payload::

    uint32   magic          = 0x53545148  ('QSTH' little-endian)
    uint32   version        = 1
    uint64   n_species_var
    float64  t_diagnosis_days
    float64  vt_diameter_cm
    char[32] params_hash_hex (null-padded ASCII)
    char[64] reserved       (zero-filled)
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

_QSTH_MAGIC = 0x53545148  # 'QSTH' little-endian
_QSTH_VERSION = 1
_QSTH_HEADER_SIZE = 128
_QSTH_HASH_LEN = 32  # Max length of the params_hash field in the header.
_QSTH_FIELD_STRUCT = struct.Struct(f"<IIQdd{_QSTH_HASH_LEN}s")
assert _QSTH_FIELD_STRUCT.size <= _QSTH_HEADER_SIZE


# The cache key (SHA-256 hex) is 64 chars; the QSTH params_hash field holds
# at most 32. Everything that crosses the Python<->qsp_sim boundary (stored
# in the blob, passed to --params-hash, compared against a header value)
# uses this truncated form. 32 hex = 128 bits — plenty of collision space
# even at 10^9 thetas. The full 64-char hash stays the cache / pack index
# key and the diagnostic-logging form.
def wire_hash(full_hash: str) -> str:
    """Truncate a full SHA-256 hex to the QSTH header's 32-char field."""
    if len(full_hash) < _QSTH_HASH_LEN:
        raise ValueError(f"hash too short: need >={_QSTH_HASH_LEN} chars, got {len(full_hash)}")
    return full_hash[:_QSTH_HASH_LEN]


class QsthHeaderError(RuntimeError):
    """QSTH blob header is missing, truncated, or malformed."""


@dataclass
class QsthHeader:
    """Parsed QSTH blob header. The payload after it is opaque to Python."""

    version: int
    n_species_var: int
    t_diagnosis_days: float
    vt_diameter_cm: float
    params_hash: str  # hex string, as written by qsp_sim's --params-hash

    @classmethod
    def parse_bytes(cls, raw: bytes, *, source: str = "<bytes>") -> "QsthHeader":
        """Parse a QSTH header from the first 128 bytes of ``raw``."""
        if len(raw) < _QSTH_HEADER_SIZE:
            raise QsthHeaderError(
                f"QSTH blob {source} truncated: {len(raw)} bytes "
                f"< {_QSTH_HEADER_SIZE}-byte header"
            )
        magic, version, n_sp, t_diag, vt_diam, hash_bytes = _QSTH_FIELD_STRUCT.unpack_from(raw, 0)
        if magic != _QSTH_MAGIC:
            raise QsthHeaderError(
                f"{source} bad magic: 0x{magic:08x} != 0x{_QSTH_MAGIC:08x} "
                f"(not a QSTH evolve-state blob?)"
            )
        if version != _QSTH_VERSION:
            raise QsthHeaderError(
                f"{source} QSTH version {version} unsupported "
                f"(this code handles {_QSTH_VERSION})"
            )
        # Header stores hash null-padded; rebuild to a plain hex string.
        hash_str = hash_bytes.rstrip(b"\x00").decode("ascii", errors="strict")
        return cls(
            version=version,
            n_species_var=int(n_sp),
            t_diagnosis_days=float(t_diag),
            vt_diameter_cm=float(vt_diam),
            params_hash=hash_str,
        )

    @classmethod
    def parse(cls, path: Path) -> "QsthHeader":
        """Parse a QSTH header from a file."""
        try:
            raw = path.read_bytes()
        except OSError as e:
            raise QsthHeaderError(f"cannot read QSTH blob {path}: {e}") from e
        return cls.parse_bytes(raw, source=str(path))


# ---- Hashing helpers ----------------------------------------------------


def theta_hash_for_xml(xml_bytes: bytes) -> str:
    """SHA-256 of the rendered param-XML bytes (hex, 64 chars).

    Same input bytes ``qsp_sim`` parses, so a match implies
    parameter-level identity. Truncated via :func:`wire_hash` for the
    QSTH header's ``params_hash`` field.
    """
    return hashlib.sha256(xml_bytes).hexdigest()


def sha256_of_file(path: Path, *, truncate: int | None = None) -> str:
    """SHA-256 hex of a file's bytes, streamed in 64 KiB chunks.

    ``truncate`` returns only the leading ``truncate`` hex chars — used
    when a short, human-readable digest is enough (e.g. namespace
    segmentation).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:truncate] if truncate else digest
