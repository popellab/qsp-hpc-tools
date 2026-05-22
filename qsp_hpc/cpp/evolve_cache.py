"""Persistent, theta-keyed, NFS-safe cache of post-evolve ODE states.

``evolve_to_diagnosis`` (the ~857-day healthy-state burn-in) is
~84-100% of every QSP sim's wall-clock and is identical across scenarios
for a given theta — it depends only on ``theta + healthy_state +
qsp_sim binary``, never on the scenario. This module caches it so:

- a second scenario over the same theta pool skips the evolve entirely,
- a re-run of the same sweep skips it,
- within one task the evolve is computed once and reused.

Design
------
The cache is a directory of **append-only QSEP shards**. Each writer
(one SLURM array task, or one local batch) flushes its own
``shard_<uuid>.qsep`` — shards are never mutated, so there is no shared
writable store and no cross-task locking. That is the NFS-safe property:
the LMDB env this module used to be (qsp-hpc-tools#86) needed coherent
``mmap`` + POSIX ``fcntl`` locks, which NFS does not provide, and
deadlocked under SLURM fan-out. Do not regress to a shared mutable store.

A reader builds a ``theta_hash -> (shard, offset, length)`` index by
scanning shard footers (cheap — :func:`~qsp_hpc.cpp.evolve_pack.read_pack_index`
reads only the footer + index, never the blob bytes) and serves
:meth:`EvolveCache.get` with one targeted read. :meth:`EvolveCache.compact`
folds the scan into ``manifest.json`` so later readers skip already-seen
shards.

Write-through, first-writer-wins: any task that evolves a theta on a
cache miss writes the state into its shard. There is no designated
emitter and no emit/consume ordering. A theta evolved twice by two
overlapping batches lands in two shards; the reader dedups by
``theta_hash``. The evolve is deterministic in the namespace inputs, so
duplicate blobs are byte-identical and which one a reader picks is
immaterial.

Layout::

    <root>/
      <namespace>/
        shard_<uuid>.qsep      append-only QSEP shards
        manifest.json          theta_hash -> [shard, offset, length]  (optional, rebuildable)

Namespace
---------
``namespace`` folds in everything ``evolve_to_diagnosis`` consumes
*other than theta* — the qsp_sim binary bytes and the healthy-state YAML.
A binary rebuild or a healthy-state edit lands new requests in a fresh
namespace, so stale states are ignored rather than silently reused.
``theta`` is the per-entry key (``theta_hash`` = SHA-256 of the rendered
param-XML); namespace + theta_hash together are the full cache key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from qsp_hpc.cpp.evolve_pack import (
    EvolveStatePackError,
    EvolveStatePackWriter,
    read_pack_index,
)
from qsp_hpc.cpp.qsth import (
    QsthHeader,
    QsthHeaderError,
    sha256_of_file,
    theta_hash_for_xml,
    wire_hash,
)

# Re-exported for back-compat: callers and tests that imported the QSTH
# header / hash helpers from this module before they moved to
# ``qsp_hpc.cpp.qsth``. New code should import them from ``qsth`` directly.
__all__ = [
    "EvolveCache",
    "EvolveCacheWriter",
    "compute_namespace",
    "QsthHeader",
    "QsthHeaderError",
    "theta_hash_for_xml",
    "wire_hash",
    "sha256_of_file",
]

logger = logging.getLogger(__name__)

_SHARD_GLOB = "shard_*.qsep"
_MANIFEST_NAME = "manifest.json"
_MANIFEST_SCHEMA = "evolve-cache-manifest-v1"
_NAMESPACE_LEN = 16  # hex chars of the namespace digest kept as the subdir name
# Default trigger for :meth:`EvolveCache.maybe_compact` — fold shards into
# the manifest once this many have accumulated outside it, so a reader's
# per-task footer scan (see :meth:`load`) stays bounded without compacting
# on every write. One SLURM array task writes one shard, so this is also
# roughly "compact every N tasks' worth of shards".
_COMPACT_MIN_UNCOMPACTED_SHARDS = 64


def compute_namespace(
    binary_path: str | Path,
    healthy_state_yaml: str | Path,
    *,
    extra: bytes | None = None,
) -> str:
    """Return the cache namespace for an evolve configuration.

    Folds the qsp_sim binary bytes and the healthy-state YAML bytes into
    one short hex digest. Any change to either (a rebuild, a healthy-state
    edit) yields a fresh namespace, so stale evolve states are ignored
    rather than silently reused.

    ``extra`` is an optional escape hatch for additional evolve-config
    bytes (e.g. a diagnosis-target override not carried by the YAML);
    unused today but keeps the key extensible without a layout migration.
    """
    h = hashlib.sha256()
    h.update(b"evolve-cache-ns-v1\0")
    h.update(sha256_of_file(Path(binary_path)).encode("ascii"))
    h.update(b"\0")
    h.update(sha256_of_file(Path(healthy_state_yaml)).encode("ascii"))
    if extra is not None:
        h.update(b"\0")
        h.update(extra)
    return h.hexdigest()[:_NAMESPACE_LEN]


@dataclass(frozen=True)
class _Loc:
    """Where one theta's blob lives: shard filename + byte range."""

    shard: str
    offset: int
    length: int


class EvolveCacheWriter:
    """Accumulates ``(theta_hash, blob)`` pairs and flushes one cache shard.

    One writer per SLURM array task / local batch. The task evolves the
    thetas it missed in the cache, :meth:`add`\\ s each QSTH blob, and
    :meth:`flush` writes a single append-only ``shard_<uuid>.qsep`` into
    the namespace directory. Shards are never mutated and each carries a
    unique name, so concurrent writers never collide.
    """

    def __init__(self, namespace_dir: str | Path) -> None:
        self.namespace_dir = Path(namespace_dir)
        self._pack = EvolveStatePackWriter()
        self._flushed: Path | None = None

    def add(self, theta_hash: str, blob: bytes, *, validate: bool = True) -> None:
        """Stage one evolve-state blob under ``theta_hash`` (see
        :meth:`EvolveStatePackWriter.add`)."""
        self._pack.add(theta_hash, blob, validate=validate)

    def __len__(self) -> int:
        return len(self._pack)

    def __contains__(self, theta_hash: str) -> bool:
        return theta_hash in self._pack

    @property
    def flushed_path(self) -> Path | None:
        """The shard written by :meth:`flush`, or None if not yet flushed."""
        return self._flushed

    def flush(self) -> Path | None:
        """Write the staged blobs as one shard; return its path.

        Returns None when nothing was staged (no misses → no shard).
        The underlying :meth:`EvolveStatePackWriter.write` is atomic
        (temp file + ``os.replace``), so a concurrent reader never sees a
        partial shard.
        """
        if len(self._pack) == 0:
            return None
        self.namespace_dir.mkdir(parents=True, exist_ok=True)
        shard = self.namespace_dir / f"shard_{uuid.uuid4().hex}.qsep"
        self._flushed = self._pack.write(shard)
        logger.info(
            "evolve cache: wrote shard %s (%d theta state(s))",
            shard.name,
            len(self._pack),
        )
        return self._flushed

    def __enter__(self) -> "EvolveCacheWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Flush only on a clean exit — a raising task should not publish a
        # shard mid-failure. Callers that want the partial evolve work
        # preserved regardless call flush() explicitly.
        if exc_type is None:
            self.flush()
        return False


class EvolveCache:
    """Persistent, theta-keyed view over one namespace of QSEP shards.

    Construct via :meth:`for_run` (which derives the namespace from the
    binary + healthy-state) or directly with an explicit namespace.
    :meth:`load` builds the ``theta_hash -> location`` index;
    :meth:`get` / :meth:`materialize` serve cached blobs; :meth:`writer`
    opens a shard writer for write-through; :meth:`compact` refreshes the
    manifest.

    Read access is lock-free (static files). The class is not thread-safe
    within a process, but is safe across processes / SLURM tasks hitting
    the same on-disk namespace.
    """

    def __init__(self, root: str | Path, namespace: str) -> None:
        self.root = Path(root).resolve()
        self.namespace = namespace
        self.namespace_dir = self.root / namespace
        self._index: dict[str, _Loc] = {}
        self._loaded = False
        self._stats_hits = 0
        self._stats_misses = 0

    @classmethod
    def for_run(
        cls,
        root: str | Path,
        *,
        binary_path: str | Path,
        healthy_state_yaml: str | Path,
        extra: bytes | None = None,
    ) -> "EvolveCache":
        """Open the cache namespace for a (binary, healthy_state) pair."""
        ns = compute_namespace(binary_path, healthy_state_yaml, extra=extra)
        return cls(root, ns)

    # ---- read side ------------------------------------------------------

    def load(self) -> "EvolveCache":
        """Build the ``theta_hash -> location`` index. Idempotent.

        Reads ``manifest.json`` if present, then scans every shard the
        manifest does not already cover. The manifest bounds the scan to
        recently-added shards; with no manifest, every shard is scanned
        (footer + index only — never the blob bytes).
        """
        if self._loaded:
            return self
        t0 = time.time()
        self._index = {}
        covered_shards: set[str] = set()
        n_from_manifest = 0
        manifest_path = self.namespace_dir / _MANIFEST_NAME
        if manifest_path.exists():
            entries, covered_shards = self._read_manifest(manifest_path)
            self._index.update(entries)
            n_from_manifest = len(entries)

        n_shards_scanned = 0
        if self.namespace_dir.is_dir():
            for shard in sorted(self.namespace_dir.glob(_SHARD_GLOB)):
                if shard.name in covered_shards:
                    continue
                try:
                    records = read_pack_index(shard)
                except (EvolveStatePackError, OSError) as e:
                    # A half-written shard (crashed task) or a foreign
                    # file — skip it, never let it abort the load.
                    logger.warning("evolve cache: skipping unreadable shard %s: %s", shard.name, e)
                    continue
                for theta_hash, offset, length in records:
                    # First-writer-wins on a duplicate theta. The blob is
                    # deterministic in the namespace inputs, so the choice
                    # is immaterial — setdefault just keeps it stable.
                    self._index.setdefault(theta_hash, _Loc(shard.name, offset, length))
                n_shards_scanned += 1

        self._loaded = True
        logger.info(
            "evolve cache loaded: namespace=%s — %d theta state(s) "
            "(%d from manifest, %d shard(s) scanned) in %.2fs",
            self.namespace,
            len(self._index),
            n_from_manifest,
            n_shards_scanned,
            time.time() - t0,
        )
        return self

    @staticmethod
    def _read_manifest(path: Path) -> tuple[dict[str, _Loc], set[str]]:
        """Parse ``manifest.json`` → (entries, covered shard names).

        A missing / corrupt / wrong-schema manifest is treated as absent
        (returns empties) — the caller then falls back to a full scan.
        """
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("evolve cache: ignoring unreadable manifest %s: %s", path, e)
            return {}, set()
        if not isinstance(data, dict) or data.get("schema") != _MANIFEST_SCHEMA:
            logger.warning(
                "evolve cache: ignoring manifest %s — unexpected schema %r",
                path,
                data.get("schema") if isinstance(data, dict) else type(data).__name__,
            )
            return {}, set()
        entries: dict[str, _Loc] = {}
        for theta_hash, loc in data.get("entries", {}).items():
            try:
                shard, offset, length = loc
                entries[theta_hash] = _Loc(str(shard), int(offset), int(length))
            except (TypeError, ValueError):
                logger.warning("evolve cache: skipping malformed manifest entry for %s", theta_hash)
        covered = {str(s) for s in data.get("shards", [])}
        return entries, covered

    def get(self, theta_hash: str) -> bytes | None:
        """Return the QSTH blob for ``theta_hash``, or None on a miss.

        A blob whose header is malformed, or whose stored ``params_hash``
        disagrees with ``theta_hash``, is treated as a miss (logged) —
        the caller re-evolves rather than feeding qsp_sim a wrong state.
        """
        if not self._loaded:
            self.load()
        loc = self._index.get(theta_hash)
        if loc is None:
            self._stats_misses += 1
            return None
        shard_path = self.namespace_dir / loc.shard
        try:
            with open(shard_path, "rb") as f:
                f.seek(loc.offset)
                blob = f.read(loc.length)
        except OSError as e:
            logger.warning(
                "evolve cache: cannot read blob for %s from %s: %s",
                theta_hash[:16],
                loc.shard,
                e,
            )
            self._stats_misses += 1
            return None
        if len(blob) != loc.length:
            logger.warning(
                "evolve cache: short read for %s (%d/%d bytes from %s)",
                theta_hash[:16],
                len(blob),
                loc.length,
                loc.shard,
            )
            self._stats_misses += 1
            return None
        try:
            header = QsthHeader.parse_bytes(blob, source=f"<{loc.shard}:{theta_hash[:16]}>")
        except QsthHeaderError as e:
            logger.warning("evolve cache: bad blob header for %s: %s", theta_hash[:16], e)
            self._stats_misses += 1
            return None
        if header.params_hash != wire_hash(theta_hash):
            logger.warning(
                "evolve cache: params_hash mismatch for %s (stored=%s) — treating as miss",
                theta_hash[:16],
                header.params_hash,
            )
            self._stats_misses += 1
            return None
        self._stats_hits += 1
        return blob

    def materialize(self, theta_hash: str, workdir: str | Path) -> Path | None:
        """Write a cached blob to a file under ``workdir`` for ``qsp_sim``.

        ``qsp_sim --initial-state`` consumes a path, so a hit must land
        on disk. Returns the path, or None on a cache miss.
        """
        blob = self.get(theta_hash)
        if blob is None:
            return None
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / f"{theta_hash[:16]}.evolve_state.bin"
        out.write_bytes(blob)
        return out

    def __contains__(self, theta_hash: str) -> bool:
        if not self._loaded:
            self.load()
        return theta_hash in self._index

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._index)

    @property
    def stats(self) -> dict[str, int]:
        """``{"hits": N, "misses": M}`` over this instance's lifetime."""
        return {"hits": self._stats_hits, "misses": self._stats_misses}

    # ---- write side -----------------------------------------------------

    def writer(self) -> EvolveCacheWriter:
        """Open a shard writer for write-through into this namespace."""
        return EvolveCacheWriter(self.namespace_dir)

    # ---- maintenance ----------------------------------------------------

    def compact(self) -> Path | None:
        """Rebuild ``manifest.json`` from a full shard scan. Atomic.

        Folds every shard's ``theta_hash -> location`` records into one
        manifest so subsequent :meth:`load` calls skip the per-shard
        footer scan. Written via temp file + atomic rename, so concurrent
        compactions are safe — last writer wins and the content converges.

        Returns the manifest path, or None when the namespace has no
        shards yet. This rewrites the *manifest*; it does not physically
        merge shard files (shard count is bounded by writers-per-run and
        pruned per namespace, and the manifest alone bounds reader scan
        cost).
        """
        if not self.namespace_dir.is_dir():
            return None
        shards = sorted(self.namespace_dir.glob(_SHARD_GLOB))
        if not shards:
            return None
        index: dict[str, list] = {}
        for shard in shards:
            try:
                records = read_pack_index(shard)
            except (EvolveStatePackError, OSError) as e:
                logger.warning(
                    "evolve cache compact: skipping unreadable shard %s: %s", shard.name, e
                )
                continue
            for theta_hash, offset, length in records:
                index.setdefault(theta_hash, [shard.name, offset, length])
        payload = {
            "schema": _MANIFEST_SCHEMA,
            "shards": [s.name for s in shards],
            "entries": index,
        }
        manifest_path = self.namespace_dir / _MANIFEST_NAME
        # Unique temp name so concurrent compactions don't fight over one
        # ``.tmp`` (a shared name lets writer A rename its tmp away before
        # writer B renames its own — FileNotFoundError on B).
        tmp = manifest_path.with_name(f"{_MANIFEST_NAME}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(manifest_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        logger.info(
            "evolve cache compacted: %s (%d theta state(s), %d shard(s))",
            manifest_path,
            len(index),
            len(shards),
        )
        return manifest_path

    def maybe_compact(self, min_uncompacted: int = _COMPACT_MIN_UNCOMPACTED_SHARDS) -> Path | None:
        """Compact only if enough shards have accumulated outside the manifest.

        A reader's per-task scan cost (:meth:`load`) is one footer read
        per shard the manifest does not already cover. This folds those
        shards in once the uncovered count reaches ``min_uncompacted`` —
        so scan cost stays bounded without paying a full compaction on
        every write.

        Cheap enough for the write path: one directory glob plus a small
        manifest read. Returns the manifest path when a compaction ran,
        else None (too few uncovered shards, or none at all). Compaction
        itself is atomic and idempotent (see :meth:`compact`), so the
        several tasks that finish a SLURM array near-together may each
        call this: the first to write the manifest covers the bulk, and
        the rest then see too few uncovered shards and skip — the work is
        self-limiting, not an O(tasks²) storm.
        """
        if not self.namespace_dir.is_dir():
            return None
        shards = list(self.namespace_dir.glob(_SHARD_GLOB))
        if not shards:
            return None
        covered: set[str] = set()
        manifest_path = self.namespace_dir / _MANIFEST_NAME
        if manifest_path.exists():
            _entries, covered = self._read_manifest(manifest_path)
        n_uncovered = sum(1 for s in shards if s.name not in covered)
        if n_uncovered < min_uncompacted:
            return None
        logger.info(
            "evolve cache: %d shard(s) outside the manifest (>= %d) — " "compacting namespace %s",
            n_uncovered,
            min_uncompacted,
            self.namespace,
        )
        return self.compact()
