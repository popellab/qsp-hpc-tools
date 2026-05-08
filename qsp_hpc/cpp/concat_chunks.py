"""Concat per-chunk trajectory parquets into one file per pool sub-pool.

The C++ batch worker writes one ``chunk_NNN.trajectory.parquet`` per
SLURM array task under ``{pool_dir}/{kind}/batch_<ts>_<scenario>_<seed>/``.
For 5k+ sims at the default ``jobs_per_chunk`` that's 250+ small files,
which is slow to read over sshfs because every file open pays an SSH
round-trip and the new ``batch_*/`` walk adds an ``ls`` per batch.

Running this concat once on HPC (where the parquets are local) and
then reading the combined file over sshfs cuts wall-time from ~50 s to
a few seconds for typical sweeps.

Invoked via :meth:`qsp_hpc.batch.hpc_job_manager.HPCJobManager.concat_trajectory_chunks`,
which ``transport.exec``\\s ``python -m qsp_hpc.cpp.concat_chunks
--pool-dir {pool_dir} --kind training``.

Output: ``{pool_dir}/{kind}/combined.trajectory.parquet`` written from
``ParquetWriter`` row-group-by-row-group so memory stays bounded even
on large chunk counts. Idempotent: if the combined file's mtime is
newer than every input chunk, the script no-ops. Safe to call from
multiple sessions concurrently because the write goes via a temp file
+ atomic rename under a per-pool flock.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_TRAJECTORY_SUFFIX = ".trajectory.parquet"
_COMBINED_NAME = "combined.trajectory.parquet"


def _list_chunk_paths(pool_dir: Path, kind: str) -> List[Path]:
    """Return ``chunk_*.trajectory.parquet`` paths under
    ``{pool_dir}/{kind}/batch_*/`` (one level deep) plus any flat-level
    chunks at ``{pool_dir}/{kind}/`` for back-compat with the legacy
    layout.
    """
    sub_dir = pool_dir / kind
    if not sub_dir.is_dir():
        return []
    paths: List[Path] = []
    for entry in sub_dir.iterdir():
        if (
            entry.is_file()
            and entry.name.endswith(_TRAJECTORY_SUFFIX)
            and entry.name != _COMBINED_NAME
        ):
            paths.append(entry)
        elif entry.is_dir() and entry.name.startswith("batch_"):
            for sub in entry.iterdir():
                if sub.is_file() and sub.name.endswith(_TRAJECTORY_SUFFIX):
                    paths.append(sub)
    paths.sort()
    return paths


def _is_up_to_date(combined: Path, chunks: Iterable[Path]) -> bool:
    if not combined.exists():
        return False
    combined_mtime = combined.stat().st_mtime
    return all(c.stat().st_mtime <= combined_mtime for c in chunks)


def concat_trajectory_chunks(pool_dir: Path, kind: str) -> Path:
    """Combine every ``chunk_*.trajectory.parquet`` under
    ``{pool_dir}/{kind}/`` into a single ``combined.trajectory.parquet``.

    Returns the path to the combined file. Idempotent on no-op when
    every chunk is older than the combined file. Acquires a per-pool
    flock to serialize concurrent invocations.
    """
    sub_dir = pool_dir / kind
    combined = sub_dir / _COMBINED_NAME

    # Tolerate a missing sub-pool dir: a scancel'd or never-started SLURM
    # array writes nothing to ``{pool}/{kind}/``, and concat is called
    # unconditionally by the orchestrator. Treat that as "no chunks to
    # concat, no-op" rather than an error so the caller flow stays
    # graceful.
    if not sub_dir.is_dir():
        logger.info("concat_trajectory_chunks: %s does not exist — skipping", sub_dir)
        return combined

    sub_dir.mkdir(parents=True, exist_ok=True)
    lock_path = sub_dir / ".combined.lock"
    if not lock_path.exists():
        lock_path.touch()

    with open(lock_path, "r+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        chunks = _list_chunk_paths(pool_dir, kind)
        if not chunks:
            logger.info("concat_trajectory_chunks: no chunks under %s — skipping", sub_dir)
            return combined

        if _is_up_to_date(combined, chunks):
            logger.info(
                "concat_trajectory_chunks: %s is up-to-date (%d chunks); no rewrite",
                combined,
                len(chunks),
            )
            return combined

        # Stream row-group-by-row-group through ParquetWriter so memory
        # stays bounded even at 5k+ sims × 21+ days.
        tmp = combined.with_name(combined.name + ".tmp")
        writer: pq.ParquetWriter | None = None
        n_rows_total = 0
        n_chunks_kept = 0
        try:
            for chunk_path in chunks:
                pf = pq.ParquetFile(chunk_path)
                if pf.metadata.num_rows == 0:
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(tmp, pf.schema_arrow, compression="zstd")
                for rg in range(pf.num_row_groups):
                    table = pf.read_row_group(rg)
                    writer.write_table(table)
                    n_rows_total += table.num_rows
                n_chunks_kept += 1
            if writer is not None:
                writer.close()
                os.replace(tmp, combined)
                logger.info(
                    "concat_trajectory_chunks: wrote %s (%d chunks → %d rows)",
                    combined,
                    n_chunks_kept,
                    n_rows_total,
                )
            else:
                # Every chunk was empty.
                tmp.unlink(missing_ok=True)
                # Write a typed empty parquet so the reader path is happy.
                empty_schema = pa.schema(
                    [
                        ("sample_index", pa.int64()),
                        ("time", pa.float64()),
                        ("species", pa.string()),
                        ("value", pa.float64()),
                    ]
                )
                empty = pa.Table.from_pylist([], schema=empty_schema)
                pq.write_table(empty, combined, compression="zstd")
                logger.info(
                    "concat_trajectory_chunks: every chunk empty; wrote typed-empty %s",
                    combined,
                )
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", required=True, type=Path)
    parser.add_argument("--kind", default="training", choices=["training", "ppc"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    out = concat_trajectory_chunks(args.pool_dir, args.kind)
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    sys.exit(main())
