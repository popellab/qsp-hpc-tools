"""QSEP pack: a flat, NFS-safe set of QSTH blobs with a theta_hash index.

Why this exists
---------------
``evolve_to_diagnosis`` is ~84-100% of every QSP sim's wall-clock and is
identical across scenarios for a given theta. Reusing it means never
sharing a *writable* store across SLURM tasks: a shared mutable store
(the LMDB env qsp-hpc-tools#86 replaced) needs coherent ``mmap`` + POSIX
``fcntl`` locks, which NFS does not provide, and deadlocks under fan-out.

The QSEP pack is the NFS-safe primitive: an append-only file holding a
set of QSTH blobs framed by a ``theta_hash -> blob`` index. A writer
emits one pack atomically (the same pattern the sim pool uses for
``batch_NNN/chunk_NNN.parquet``); readers ``mmap`` static files. The
persistent, namespaced :class:`~qsp_hpc.cpp.evolve_cache.EvolveCache`
layers a multi-shard ``theta_hash`` index over a directory of these
packs.

This module is just the pack format — a writer, a reader, and an
index-only scan (:func:`read_pack_index`). It has no SLURM / binary
dependency, just bytes, so it is unit-testable in isolation.

File format ("QSEP")
--------------------
::

    [ blob_0 ][ blob_1 ] ... [ blob_{K-1} ]   concatenated raw QSTH blobs
    [ index  ]                                K fixed-size index records
    [ footer ]                                fixed 32-byte trailer

    index record  (struct '<64sQQ', 80 bytes):
        char[64]  theta_hash    full sha256 hex of the rendered param XML
        uint64    blob_offset   bytes from file start
        uint64    blob_length

    footer  (struct '<IIQQQ', 32 bytes, at EOF-32):
        uint32    magic         = 0x51534550  ('QSEP', little-endian)
        uint32    version       = 1
        uint64    n_entries
        uint64    index_offset
        uint64    index_length

The QSTH blobs are self-describing (128-byte header + opaque payload —
see :mod:`qsp_hpc.cpp.qsth`); the pack just frames a set of them with an
O(1) ``theta_hash -> blob`` index. Footer-trailed so the writer streams
blobs first and finalises the index in one pass.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from qsp_hpc.cpp.qsth import QsthHeader, QsthHeaderError, wire_hash

logger = logging.getLogger(__name__)

_PACK_MAGIC = 0x51534550  # 'QSEP' little-endian
_PACK_VERSION = 1

_THETA_HASH_LEN = 64  # full sha256 hex — the pack index key

_INDEX_REC = struct.Struct(f"<{_THETA_HASH_LEN}sQQ")  # theta_hash, offset, length
_FOOTER = struct.Struct("<IIQQQ")  # magic, version, n_entries, index_offset, index_length

assert _INDEX_REC.size == 80
assert _FOOTER.size == 32


class EvolveStatePackError(RuntimeError):
    """Evolve-state pack is malformed, truncated, or has a bad footer."""


def _validate_theta_hash(theta_hash: str) -> str:
    """Return ``theta_hash`` if it is a 64-char sha256 hex, else raise."""
    if len(theta_hash) != _THETA_HASH_LEN:
        raise EvolveStatePackError(
            f"theta_hash must be {_THETA_HASH_LEN} hex chars, got {len(theta_hash)}"
        )
    try:
        int(theta_hash, 16)
    except ValueError as exc:
        raise EvolveStatePackError(f"theta_hash not hex: {theta_hash!r}") from exc
    return theta_hash


# ---- writer (task-side emission) ---------------------------------------


class EvolveStatePackWriter:
    """Accumulate ``(theta_hash, QSTH blob)`` pairs and write one pack.

    One instance per SLURM array task: the task evolves its chunk of
    thetas, :meth:`add`\\ s each blob, then :meth:`write`\\ s a single
    pack file. A per-task chunk is small (tens of thetas × ~1-2 KB), so
    blobs are held in memory until ``write`` — no streaming needed.

    Adding the same ``theta_hash`` twice keeps the last blob (a chunk
    has unique thetas, so this is purely defensive).
    """

    def __init__(self) -> None:
        # Insertion-ordered; dict dedups on theta_hash (last wins).
        self._entries: Dict[str, bytes] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, theta_hash: str) -> bool:
        return theta_hash in self._entries

    def add(self, theta_hash: str, blob: bytes, *, validate: bool = True) -> None:
        """Add one evolve-state blob under ``theta_hash``.

        When ``validate`` (default), the blob is parsed as a QSTH header
        and its stored ``params_hash`` is cross-checked against
        ``wire_hash(theta_hash)`` — catching a blob filed under the
        wrong key before it reaches a scenario sim.
        """
        _validate_theta_hash(theta_hash)
        if not isinstance(blob, (bytes, bytearray)):
            raise EvolveStatePackError(f"blob must be bytes, got {type(blob).__name__}")
        blob = bytes(blob)
        if validate:
            try:
                header = QsthHeader.parse_bytes(blob, source=f"<add:{theta_hash[:16]}>")
            except QsthHeaderError as exc:
                raise EvolveStatePackError(
                    f"blob for {theta_hash[:16]} is not a valid QSTH blob: {exc}"
                ) from exc
            expected = wire_hash(theta_hash)
            if header.params_hash != expected:
                raise EvolveStatePackError(
                    f"blob params_hash {header.params_hash} != wire_hash(theta_hash) "
                    f"{expected} — blob filed under the wrong key"
                )
        self._entries[theta_hash] = blob

    def write(self, path: str | Path) -> Path:
        """Serialize the pack to ``path`` (atomically) and return it.

        Writes ``<path>.tmp`` then ``os.replace``\\ s it onto ``path``
        so a concurrent reader never sees a half-written pack — and a
        crashed task leaves a ``.tmp`` rather than a corrupt pack.
        """
        path = Path(path)
        blobs = bytearray()
        index = bytearray()
        for theta_hash, blob in self._entries.items():
            index += _INDEX_REC.pack(theta_hash.encode("ascii"), len(blobs), len(blob))
            blobs += blob
        index_offset = len(blobs)
        footer = _FOOTER.pack(
            _PACK_MAGIC, _PACK_VERSION, len(self._entries), index_offset, len(index)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(bytes(blobs) + bytes(index) + footer)
        tmp.replace(path)
        logger.debug(
            "evolve pack written: %s (%d entries, %d bytes)",
            path,
            len(self._entries),
            index_offset + len(index) + _FOOTER.size,
        )
        return path


def write_evolve_pack(
    path: str | Path,
    entries: Iterable[Tuple[str, bytes]],
    *,
    validate: bool = True,
) -> Path:
    """One-shot convenience: build a pack from ``(theta_hash, blob)`` pairs."""
    writer = EvolveStatePackWriter()
    for theta_hash, blob in entries:
        writer.add(theta_hash, blob, validate=validate)
    return writer.write(path)


# ---- reader (scenario-side consumption) --------------------------------


class EvolveStatePackReader:
    """Read-only view over one or more evolve-state pack files.

    Scenario arrays open packs ``readonly`` and pull a theta's evolve
    state via :meth:`get` / :meth:`materialize`. Multiple per-task packs
    can be layered into one reader (:meth:`add_pack`) so a scenario sees
    a whole evolve phase's output without a merge step — later packs win
    on a duplicate ``theta_hash``.

    Packs are small (per-task: tens of KB; a full evolve phase: tens of
    MB), so each pack's bytes are read fully into memory. Switch to
    ``mmap`` here if aggregated packs ever outgrow that.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        # theta_hash -> blob bytes
        self._blobs: Dict[str, bytes] = {}
        self._packs: List[Path] = []
        if path is not None:
            self.add_pack(path)

    @classmethod
    def from_dir(cls, directory: str | Path, pattern: str = "*.qsep") -> "EvolveStatePackReader":
        """Build a reader over every pack matching ``pattern`` in ``directory``.

        The evolve phase deposits one pack per task; this layers them all
        into a single lookup with no explicit aggregation step.
        """
        directory = Path(directory)
        reader = cls()
        for pack in sorted(directory.glob(pattern)):
            reader.add_pack(pack)
        return reader

    def add_pack(self, path: str | Path) -> int:
        """Layer one pack file into the reader; return its entry count."""
        path = Path(path)
        data = path.read_bytes()
        if len(data) < _FOOTER.size:
            raise EvolveStatePackError(f"{path}: {len(data)} bytes < {_FOOTER.size}-byte footer")
        magic, version, n_entries, index_offset, index_length = _FOOTER.unpack(
            data[-_FOOTER.size :]
        )
        if magic != _PACK_MAGIC:
            raise EvolveStatePackError(
                f"{path}: bad magic 0x{magic:08x} != 0x{_PACK_MAGIC:08x} "
                f"(not a QSEP evolve-state pack?)"
            )
        if version != _PACK_VERSION:
            raise EvolveStatePackError(
                f"{path}: QSEP version {version} unsupported (this code handles "
                f"{_PACK_VERSION})"
            )
        index_end = index_offset + index_length
        if index_end > len(data) - _FOOTER.size:
            raise EvolveStatePackError(
                f"{path}: index [{index_offset}:{index_end}] overruns file " f"({len(data)} bytes)"
            )
        if index_length != n_entries * _INDEX_REC.size:
            raise EvolveStatePackError(
                f"{path}: index_length {index_length} != n_entries {n_entries} "
                f"× {_INDEX_REC.size}"
            )
        for i in range(n_entries):
            rec = data[
                index_offset + i * _INDEX_REC.size : index_offset + (i + 1) * _INDEX_REC.size
            ]
            hash_bytes, blob_offset, blob_length = _INDEX_REC.unpack(rec)
            theta_hash = hash_bytes.decode("ascii", errors="strict")
            blob_end = blob_offset + blob_length
            if blob_end > index_offset:
                raise EvolveStatePackError(
                    f"{path}: blob for {theta_hash[:16]} [{blob_offset}:{blob_end}] "
                    f"overruns the blob section (ends at {index_offset})"
                )
            self._blobs[theta_hash] = data[blob_offset:blob_end]
        self._packs.append(path)
        return n_entries

    def __len__(self) -> int:
        return len(self._blobs)

    def __contains__(self, theta_hash: str) -> bool:
        return theta_hash in self._blobs

    @property
    def theta_hashes(self) -> List[str]:
        return list(self._blobs.keys())

    @property
    def packs(self) -> List[Path]:
        """Pack files layered into this reader, in load order."""
        return list(self._packs)

    def get(self, theta_hash: str) -> Optional[bytes]:
        """Return the QSTH blob for ``theta_hash``, or None if absent."""
        return self._blobs.get(theta_hash)

    def materialize(self, theta_hash: str, workdir: str | Path) -> Path:
        """Write the blob for ``theta_hash`` to a file under ``workdir``.

        ``qsp_sim --initial-state`` consumes a path, so a scenario sim
        needs the blob on disk. Raises :class:`KeyError` when the theta
        is not in any layered pack.
        """
        blob = self._blobs.get(theta_hash)
        if blob is None:
            raise KeyError(f"theta_hash {theta_hash[:16]} not in pack(s) {self._packs}")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / f"{theta_hash[:16]}.evolve_state.bin"
        out.write_bytes(blob)
        return out

    def items(self) -> Iterator[Tuple[str, bytes]]:
        """Iterate ``(theta_hash, blob)`` over every layered entry."""
        return iter(self._blobs.items())


# ---- index-only scan (cache-side) --------------------------------------


def read_pack_index(path: str | Path) -> List[Tuple[str, int, int]]:
    """Read a QSEP pack's footer + index *without* loading any blob bytes.

    Returns ``[(theta_hash, blob_offset, blob_length), ...]`` in index
    order. This is the cheap scan
    :class:`~qsp_hpc.cpp.evolve_cache.EvolveCache` runs over many shards
    to build a ``theta_hash -> location`` map: only the fixed 32-byte
    footer and the ``n_entries x 80``-byte index are read, never the
    (multi-KB) blob payloads.

    Raises :class:`EvolveStatePackError` on a missing / truncated footer,
    bad magic, unsupported version, or an index that overruns the file.
    """
    path = Path(path)
    size = path.stat().st_size
    if size < _FOOTER.size:
        raise EvolveStatePackError(f"{path}: {size} bytes < {_FOOTER.size}-byte footer")
    with open(path, "rb") as f:
        f.seek(size - _FOOTER.size)
        magic, version, n_entries, index_offset, index_length = _FOOTER.unpack(f.read(_FOOTER.size))
        if magic != _PACK_MAGIC:
            raise EvolveStatePackError(
                f"{path}: bad magic 0x{magic:08x} != 0x{_PACK_MAGIC:08x} "
                f"(not a QSEP evolve-state pack?)"
            )
        if version != _PACK_VERSION:
            raise EvolveStatePackError(
                f"{path}: QSEP version {version} unsupported (this code handles {_PACK_VERSION})"
            )
        if index_length != n_entries * _INDEX_REC.size:
            raise EvolveStatePackError(
                f"{path}: index_length {index_length} != n_entries {n_entries} "
                f"x {_INDEX_REC.size}"
            )
        index_end = index_offset + index_length
        if index_end > size - _FOOTER.size:
            raise EvolveStatePackError(
                f"{path}: index [{index_offset}:{index_end}] overruns file ({size} bytes)"
            )
        f.seek(index_offset)
        index_data = f.read(index_length)
    out: List[Tuple[str, int, int]] = []
    for i in range(n_entries):
        rec = index_data[i * _INDEX_REC.size : (i + 1) * _INDEX_REC.size]
        hash_bytes, blob_offset, blob_length = _INDEX_REC.unpack(rec)
        theta_hash = hash_bytes.decode("ascii", errors="strict")
        if blob_offset + blob_length > index_offset:
            raise EvolveStatePackError(
                f"{path}: blob for {theta_hash[:16]} "
                f"[{blob_offset}:{blob_offset + blob_length}] overruns the blob "
                f"section (ends at {index_offset})"
            )
        out.append((theta_hash, blob_offset, blob_length))
    return out
