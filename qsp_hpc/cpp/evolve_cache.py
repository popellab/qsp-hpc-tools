"""Cache for ``evolve_to_diagnosis`` ODE states keyed on theta.

For multi-scenario sweeps (e.g. baseline + treatment arms sharing the
same parameter set), the ~857-day healthy-state integration is ~95% of
per-sim work and is identical across scenarios for a given theta. This
module reuses it: run it once per theta, serialize the post-evolve ODE
state, and let subsequent scenario runs load the state via qsp_sim's
``--initial-state`` flag.

Cache layout
------------
::

    <cache_root>/
      <healthy_state_hash[:8]>_<binary_hash[:8]>/
        data.mdb     # LMDB env; key=theta_hash[:16] -> QSTH blob bytes
        lock.mdb     # LMDB writer/reader lock

One LMDB env per (healthy_state, qsp_sim) pair. ~50k blobs occupy two
files instead of 50k inodes — load-bearing on group-quota HPC
filesystems. LMDB's mmap + COW commit semantics give lock-free readers
and a single writer transaction at a time, so the per-theta fcntl dance
is gone.

Segmentation by ``<healthy_state_hash[:8]>_<binary_hash[:8]>`` makes
invalidation automatic: edit ``healthy_state.yaml`` or rebuild
``qsp_sim`` and new theta requests land in a fresh sub-directory.
Old sub-directories are safe to prune but aren't removed automatically.

The blob header stores the full ``theta_hash`` (SHA-256 of the rendered
param-XML bytes — the same bytes ``qsp_sim`` parses). On load, ``qsp_sim
--params-hash`` verifies the stored hash matches the current theta and
refuses the load on mismatch.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import lmdb

from qsp_hpc.cpp.param_xml import ParamXMLRenderer
from qsp_hpc.cpp.runner import CppRunner

logger = logging.getLogger(__name__)


# ---- QSTH blob header (mirrors write_qsth_header in qsp_sim.cpp) --------
#
# Fixed 128-byte little-endian header, followed by the CVODEBase full-state
# payload (state vectors + delay-event queue + trigger flags). Python reads
# only the header — the payload is opaque and consumed by qsp_sim.
#
#   uint32   magic          = 0x53545148  ('QSTH' little-endian)
#   uint32   version        = 1
#   uint64   n_species_var
#   float64  t_diagnosis_days
#   float64  vt_diameter_cm
#   char[32] params_hash_hex (null-padded ASCII)
#   char[64] reserved       (zero-filled)

_QSTH_MAGIC = 0x53545148
_QSTH_VERSION = 1
_QSTH_HEADER_SIZE = 128
_QSTH_HASH_LEN = 32  # Max length of the params_hash field in the header.
_QSTH_FIELD_STRUCT = struct.Struct(f"<IIQdd{_QSTH_HASH_LEN}s")
assert _QSTH_FIELD_STRUCT.size <= _QSTH_HEADER_SIZE


# The cache key (SHA-256 hex) is 64 chars; the QSTH params_hash field holds
# at most 32. Everything that crosses the Python↔qsp_sim boundary (stored
# in the blob, passed to --params-hash, compared against a header value)
# uses this truncated form. 32 hex = 128 bits — plenty of collision space
# even at 10^9 thetas. Full 64-char hash is kept as the filename stem
# (16-char truncation) and for diagnostic logging.
def wire_hash(full_hash: str) -> str:
    """Truncate a full SHA-256 hex to the QSTH header's 32-char field."""
    if len(full_hash) < _QSTH_HASH_LEN:
        raise ValueError(f"hash too short: need >={_QSTH_HASH_LEN} chars, got {len(full_hash)}")
    return full_hash[:_QSTH_HASH_LEN]


class QsthHeaderError(RuntimeError):
    """QSTH blob header is missing, truncated, or malformed."""


@dataclass
class QsthHeader:
    version: int
    n_species_var: int
    t_diagnosis_days: float
    vt_diameter_cm: float
    params_hash: str  # hex string, as written by qsp_sim's --params-hash

    @classmethod
    def parse_bytes(cls, raw: bytes, *, source: str = "<bytes>") -> "QsthHeader":
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
        try:
            raw = path.read_bytes()
        except OSError as e:
            raise QsthHeaderError(f"cannot read QSTH blob {path}: {e}") from e
        return cls.parse_bytes(raw, source=str(path))


# ---- Hashing helpers ----------------------------------------------------


def theta_hash_for_xml(xml_bytes: bytes) -> str:
    """SHA-256 of the rendered param-XML bytes (hex, 64 chars).

    Same input bytes qsp_sim parses, so a match implies parameter-level
    identity. Stored in the QSTH header under ``params_hash``.
    """
    return hashlib.sha256(xml_bytes).hexdigest()


def _sha256_of_file(path: Path, *, truncate: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:truncate] if truncate else digest


# ---- Cache --------------------------------------------------------------


# LMDB map_size: virtual address-space cap for the env's mmap. Disk grows
# lazily, so this is a ceiling rather than an allocation. Sized for the
# realistic worst case (millions of QSTH blobs at a few KB each) with
# generous headroom; mostly-cheap on 64-bit hosts.
_LMDB_MAP_SIZE = 16 * 1024 * 1024 * 1024  # 16 GiB


class CppEvolveCache:
    """Per-theta post-evolve ODE state, shared across scenarios.

    Backed by one LMDB env per (healthy_state, qsp_sim_binary) pair, keyed
    by ``theta_hash[:16]`` with QSTH blob bytes as the value. Across
    processes, LMDB serializes write transactions internally — the prior
    per-theta fcntl lockfile dance is unnecessary. Readers are lock-free
    via the mmap'd snapshot.

    The class is not thread-safe within a process (the underlying
    ``CppRunner`` isn't), but is safe across processes hitting the same
    on-disk env.
    """

    def __init__(
        self,
        cache_root: str | Path,
        renderer: ParamXMLRenderer,
        runner: CppRunner,
    ):
        """
        Args:
            cache_root: Base directory for cached blobs. One LMDB env
                sub-directory per (healthy_state_yaml, qsp_sim_binary)
                pair lives inside it. Usually somewhere persistent like
                ``cache/cpp_simulations/evolve_cache/`` locally or the
                HPC scratch pool path.
            renderer: Must match the renderer used for the actual sims —
                the cache key is SHA-256 of the bytes this renderer
                produces, so a template change invalidates all entries
                implicitly (the bytes change → the hash changes).
            runner: Must have ``healthy_state_yaml`` set. Invoked with
                ``--dump-state`` to build missing blobs.
        """
        if runner.healthy_state_yaml is None:
            raise ValueError(
                "CppEvolveCache requires runner.healthy_state_yaml "
                "(there's nothing to cache without evolve-to-diagnosis)"
            )
        self.cache_root = Path(cache_root).resolve()
        self.renderer = renderer
        self.runner = runner

        # Segment by healthy-state YAML + qsp_sim binary so edits to
        # either auto-invalidate without touching existing blobs.
        hs_hash = _sha256_of_file(runner.healthy_state_yaml, truncate=8)
        bin_hash = _sha256_of_file(runner.binary_path, truncate=8)
        self._subdir_name = f"{hs_hash}_{bin_hash}"
        self._cache_dir = self.cache_root / self._subdir_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._env = lmdb.open(
            str(self._cache_dir),
            map_size=_LMDB_MAP_SIZE,
            subdir=True,
            max_readers=512,
            readahead=False,  # large random-access blobs; readahead just thrashes
            meminit=False,
        )

        self._stats_hits = 0
        self._stats_misses = 0

    @property
    def cache_dir(self) -> Path:
        """LMDB env directory holding blobs for the current
        (healthy_state, qsp_sim) pair."""
        return self._cache_dir

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._stats_hits, "misses": self._stats_misses}

    def close(self) -> None:
        env = getattr(self, "_env", None)
        if env is not None:
            env.close()
            self._env = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def theta_hash(self, params: Mapping[str, float]) -> tuple[str, bytes]:
        """Render params and return ``(theta_hash_hex, xml_bytes)``."""
        xml = self.renderer.render(params)
        return theta_hash_for_xml(xml), xml

    @staticmethod
    def _key_for_hash(theta_hash: str) -> bytes:
        if len(theta_hash) < 16:
            raise ValueError("theta_hash must be at least 16 hex chars")
        return theta_hash[:16].encode("ascii")

    def _materialize(self, blob: bytes, theta_hash: str, workdir: Path) -> Path:
        """Write blob bytes to a per-theta tmp file under workdir and
        return the path. qsp_sim's ``--initial-state`` consumes a path,
        so each get/build call needs an on-disk copy. The caller may
        unlink the path once qsp_sim has finished with it."""
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / f"{theta_hash[:16]}.evolve_state.bin"
        out.write_bytes(blob)
        return out

    def get_or_build(
        self,
        params: Mapping[str, float],
        workdir: str | Path,
        *,
        timeout_s: float | None = None,
    ) -> tuple[Path, str]:
        """Return ``(state_blob_path, theta_hash)``, building the blob if
        it's missing.

        On hit: blob bytes are written to a tmp file under ``workdir`` and
        that path is returned. On miss: qsp_sim's ``--dump-state`` writes
        the blob to a tmp file under ``workdir``, the bytes are validated
        and committed to LMDB, and the same tmp path is returned. Across
        processes LMDB serializes the put — racing builders both write,
        the loser overwrites with bit-identical bytes.
        """
        th, _xml_bytes = self.theta_hash(params)
        workdir = Path(workdir)
        key = self._key_for_hash(th)
        expected_wire = wire_hash(th)

        # Read path: lock-free txn against the mmap snapshot.
        with self._env.begin(write=False, buffers=False) as txn:
            blob = txn.get(key)
        if blob is not None:
            try:
                header = QsthHeader.parse_bytes(blob, source=f"<lmdb:{key.decode()}>")
            except QsthHeaderError:
                logger.warning(
                    "evolve cache: blob for %s has bad header, will rebuild", key.decode()
                )
                blob = None
            else:
                # Truncated-hash mismatch is the LMDB-side authoritative
                # check; qsp_sim's --params-hash check on load is the
                # secondary line of defense.
                if header.params_hash != expected_wire:
                    logger.warning(
                        "evolve cache: blob for %s has params_hash mismatch "
                        "(stored=%s expected=%s), will rebuild",
                        key.decode(),
                        header.params_hash,
                        expected_wire,
                    )
                    blob = None

        if blob is not None:
            self._stats_hits += 1
            logger.debug("evolve cache HIT: %s", key.decode())
            path = self._materialize(bytes(blob), th, workdir)
            return path, th

        logger.info("evolve cache MISS — building (theta_hash=%s)", th[:16])
        workdir.mkdir(parents=True, exist_ok=True)
        tmp_out = workdir / f"{th[:16]}.evolve_state.bin"
        # qsp_sim stores wire_hash(theta_hash) in the blob header; use the
        # same truncated form when calling --params-hash so load-side
        # comparisons succeed against the full theta_hash truncated the
        # same way.
        self.runner.dump_evolve_state(
            params=params,
            params_hash=expected_wire,
            state_out=tmp_out,
            workdir=workdir,
            timeout_s=timeout_s,
        )
        new_blob = tmp_out.read_bytes()
        # Sanity-check the bytes qsp_sim just produced before committing.
        header = QsthHeader.parse_bytes(new_blob, source=str(tmp_out))
        if header.params_hash != expected_wire:
            raise QsthHeaderError(
                f"qsp_sim --dump-state wrote params_hash={header.params_hash} "
                f"but expected {expected_wire}"
            )
        with self._env.begin(write=True) as txn:
            txn.put(key, new_blob, overwrite=True)
        self._stats_misses += 1
        return tmp_out, th
