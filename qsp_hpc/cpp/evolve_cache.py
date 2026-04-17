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
        <theta_hash[:16]>.state.bin   # QSTH blob (C++ qsp_sim writes this)
        <theta_hash[:16]>.lock        # fcntl-locked build coordinator

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

import fcntl
import hashlib
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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
    def parse(cls, path: Path) -> "QsthHeader":
        try:
            raw = path.read_bytes()
        except OSError as e:
            raise QsthHeaderError(f"cannot read QSTH blob {path}: {e}") from e
        if len(raw) < _QSTH_HEADER_SIZE:
            raise QsthHeaderError(
                f"QSTH blob {path} truncated: {len(raw)} bytes "
                f"< {_QSTH_HEADER_SIZE}-byte header"
            )
        magic, version, n_sp, t_diag, vt_diam, hash_bytes = _QSTH_FIELD_STRUCT.unpack_from(raw, 0)
        if magic != _QSTH_MAGIC:
            raise QsthHeaderError(
                f"{path} bad magic: 0x{magic:08x} != 0x{_QSTH_MAGIC:08x} "
                f"(not a QSTH evolve-state blob?)"
            )
        if version != _QSTH_VERSION:
            raise QsthHeaderError(
                f"{path} QSTH version {version} unsupported " f"(this code handles {_QSTH_VERSION})"
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


class CppEvolveCache:
    """Per-theta post-evolve ODE state, shared across scenarios.

    Not thread-safe within a process (the underlying ``CppRunner`` isn't).
    Across processes, concurrent ``get_or_build`` calls for the same theta
    are serialized by an advisory file lock so only one process pays the
    evolve cost; others block on the lock and read the resulting blob.
    """

    def __init__(
        self,
        cache_root: str | Path,
        renderer: ParamXMLRenderer,
        runner: CppRunner,
    ):
        """
        Args:
            cache_root: Base directory for cached blobs. One sub-directory
                per (healthy_state_yaml, qsp_sim_binary) pair will be
                created inside it. Usually somewhere persistent like
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

        self._stats_hits = 0
        self._stats_misses = 0

    @property
    def cache_dir(self) -> Path:
        """Sub-directory holding blobs for the current
        (healthy_state, qsp_sim) pair."""
        return self._cache_dir

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._stats_hits, "misses": self._stats_misses}

    def state_path_for_hash(self, theta_hash: str) -> Path:
        """Blob path for a theta hash — the hash is truncated to 16 chars
        in the filename (16 hex = 64 bits of collision space, which is
        plenty at theta-count < 10^9)."""
        if len(theta_hash) < 16:
            raise ValueError("theta_hash must be at least 16 hex chars")
        return self._cache_dir / f"{theta_hash[:16]}.state.bin"

    def theta_hash(self, params: Mapping[str, float]) -> tuple[str, bytes]:
        """Render params and return ``(theta_hash_hex, xml_bytes)``."""
        xml = self.renderer.render(params)
        return theta_hash_for_xml(xml), xml

    def get_or_build(
        self,
        params: Mapping[str, float],
        workdir: str | Path,
        *,
        timeout_s: float | None = None,
    ) -> tuple[Path, str]:
        """Return ``(state_blob_path, theta_hash)``, building the blob if
        it's missing or invalid.

        The first caller for a given theta runs the evolve phase and
        writes the blob under an exclusive file lock. Later callers for
        the same theta block on the lock, then find a valid blob and
        return without running qsp_sim.
        """
        th, _xml_bytes = self.theta_hash(params)
        path = self.state_path_for_hash(th)

        if self._is_valid(path, th):
            self._stats_hits += 1
            logger.debug("evolve cache HIT: %s", path.name)
            return path, th

        # Miss — build under lock. Re-check inside the lock so a racing
        # process that built the blob while we waited is honored.
        lock_path = path.with_name(path.stem + ".lock")
        with open(lock_path, "w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            if self._is_valid(path, th):
                self._stats_hits += 1
                logger.debug("evolve cache HIT after waiting on lock: %s", path.name)
                return path, th
            logger.info(
                "evolve cache MISS — building %s (theta_hash=%s)",
                path.name,
                th[:16],
            )
            self._build(params, th, path, Path(workdir), timeout_s=timeout_s)
            self._stats_misses += 1
        return path, th

    # -- internals --------------------------------------------------------

    def _is_valid(self, path: Path, expected_hash: str) -> bool:
        """Exists + parseable header + params_hash matches."""
        if not path.exists():
            return False
        try:
            header = QsthHeader.parse(path)
        except QsthHeaderError:
            logger.warning("evolve cache: blob %s has bad header, will rebuild", path)
            return False
        # QSTH header stores wire_hash(theta_hash) (32 chars). qsp_sim's
        # --params-hash comparison on load is the authoritative check, so
        # a stale blob that slips through here still fails safely at
        # simulation time.
        return header.params_hash == wire_hash(expected_hash)

    def _build(
        self,
        params: Mapping[str, float],
        theta_hash: str,
        path: Path,
        workdir: Path,
        *,
        timeout_s: float | None,
    ) -> None:
        """Invoke qsp_sim --dump-state to produce the blob at `path`."""
        # qsp_sim stores wire_hash(theta_hash) in the blob header; use the
        # same truncated form when calling --params-hash so load-side
        # comparisons succeed against the full theta_hash truncated the
        # same way.
        self.runner.dump_evolve_state(
            params=params,
            params_hash=wire_hash(theta_hash),
            state_out=path,
            workdir=workdir,
            timeout_s=timeout_s,
        )
