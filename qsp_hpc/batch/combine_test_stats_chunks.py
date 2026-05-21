#!/usr/bin/env python3
"""
Combine Test Statistics Chunks

This script runs on HPC to combine chunk files from test statistics derivation
into single combined files.

Combines, per test-stats directory:
- chunk_XXX_test_stats.csv  -> combined_test_stats.csv     (no header)
- chunk_XXX_params.csv      -> combined_params.csv          (header)
- chunk_XXX_params.csv[col0]-> combined_sample_index.csv    (no header, 1 col)

``combined_sample_index.csv`` is a sidecar carrying just the ``sample_index``
column, row-aligned with ``combined_test_stats.csv``. It lets the fused
multi-scenario download skip the (large, redundant) ``combined_params.csv``
transfer entirely — theta is regenerated locally from the deterministic
theta pool, so only the sample_index -> row mapping has to come back.

Usage:
    # single directory
    python -m qsp_hpc.batch.combine_test_stats_chunks <test_stats_dir>

    # fused: combine many directories in one process (one ssh round-trip)
    python -m qsp_hpc.batch.combine_test_stats_chunks --fused <dir1> <dir2> ...
"""

import sys
from pathlib import Path


def _find_chunks(test_stats_dir: Path, pattern: str) -> list[str]:
    """Find chunk files at any depth under ``test_stats_dir``.

    Inline-derive writes shards to ``test_stats/<hash>/<batch_subdir>/chunk_*.csv``;
    the legacy cold-path derive worker wrote them flat at
    ``test_stats/<hash>/chunk_*.csv``. ``rglob`` matches both so pools
    with mixed-history layouts combine cleanly. Sorted lexicographically
    so chunk order is deterministic across runs.
    """
    return sorted(str(p) for p in test_stats_dir.rglob(pattern))


def combine_test_stats(test_stats_dir: Path) -> int:
    """
    Combine test statistics chunk files (no headers).

    Args:
        test_stats_dir: Directory containing chunk files

    Returns:
        Number of chunks combined
    """
    chunk_files = _find_chunks(test_stats_dir, "chunk_*_test_stats.csv")

    if not chunk_files:
        print("WARNING: No test stats chunk files found")
        return 0

    output_file = test_stats_dir / "combined_test_stats.csv"

    # Simple concatenation since no headers
    with open(output_file, "w") as outf:
        for chunk_file in chunk_files:
            with open(chunk_file, "r") as inf:
                outf.write(inf.read())

    print(f"Combined {len(chunk_files)} test stats chunks → {output_file}")
    return len(chunk_files)


def combine_params(test_stats_dir: Path) -> int:
    """
    Combine parameter chunk files (has CSV headers).

    Args:
        test_stats_dir: Directory containing chunk files

    Returns:
        Number of chunks combined
    """
    chunk_files = _find_chunks(test_stats_dir, "chunk_*_params.csv")

    if not chunk_files:
        print("INFO: No params chunk files found (may be older format)")
        return 0

    output_file = test_stats_dir / "combined_params.csv"

    # Keep header from first file only
    with open(output_file, "w") as outf:
        for i, chunk_file in enumerate(chunk_files):
            with open(chunk_file, "r") as inf:
                if i == 0:
                    # First file: write everything including header
                    outf.write(inf.read())
                else:
                    # Other files: skip header line
                    lines = inf.readlines()
                    if len(lines) > 1:  # Has header + data
                        outf.writelines(lines[1:])

    print(f"Combined {len(chunk_files)} params chunks → {output_file}")
    return len(chunk_files)


def combine_sample_index(test_stats_dir: Path) -> int:
    """
    Extract a ``combined_sample_index.csv`` sidecar row-aligned with
    ``combined_test_stats.csv``.

    For every ``chunk_NNN_test_stats.csv`` (the files that drive
    :func:`combine_test_stats`, in the same sorted order) the sibling
    ``chunk_NNN_params.csv`` is opened and its ``sample_index`` column
    (the first column) is appended. The result is a headerless,
    single-column CSV whose row ``i`` is the sample_index of row ``i`` of
    ``combined_test_stats.csv`` — so a fused download never has to ship
    the full ``combined_params.csv`` just to recover the row mapping.

    Args:
        test_stats_dir: Directory containing chunk files

    Returns:
        Number of sample_index rows written
    """
    test_stats_chunks = _find_chunks(test_stats_dir, "chunk_*_test_stats.csv")

    if not test_stats_chunks:
        print("WARNING: No test stats chunk files found — no sample_index sidecar")
        return 0

    output_file = test_stats_dir / "combined_sample_index.csv"
    n_rows = 0
    with open(output_file, "w") as outf:
        for ts_chunk in test_stats_chunks:
            # Sibling params chunk: chunk_NNN_test_stats.csv -> chunk_NNN_params.csv
            params_chunk = ts_chunk.replace("_test_stats.csv", "_params.csv")
            if not Path(params_chunk).exists():
                raise FileNotFoundError(
                    f"params chunk missing for {ts_chunk} — cannot build "
                    f"sample_index sidecar (expected {params_chunk})"
                )
            with open(params_chunk, "r") as inf:
                lines = inf.read().splitlines()
            if len(lines) < 2:
                continue  # header only / empty
            header = lines[0].split(",")
            if header[0] != "sample_index":
                raise ValueError(
                    f"{params_chunk}: first column is {header[0]!r}, expected "
                    f"'sample_index' — chunk params layout changed"
                )
            for row in lines[1:]:
                outf.write(row.split(",", 1)[0] + "\n")
                n_rows += 1

    print(f"Combined {n_rows} sample_index rows → {output_file}")
    return n_rows


def combine_dir(test_stats_dir: Path) -> int:
    """Combine all chunk file types for one directory.

    Returns the test-stats chunk count (0 means nothing was combined —
    the caller treats that as a failed/empty derive).
    """
    if not test_stats_dir.exists():
        print(f"ERROR: Directory not found: {test_stats_dir}")
        return 0

    print(f"Combining chunks in: {test_stats_dir}")
    n_test_stats = combine_test_stats(test_stats_dir)
    combine_params(test_stats_dir)
    combine_sample_index(test_stats_dir)
    return n_test_stats


def main():
    """Main entry point for combining chunk files.

    Single-directory mode::

        combine_test_stats_chunks.py <test_stats_dir>

    Fused mode — combine every directory in one process (one ssh
    round-trip for an N-scenario joint run)::

        combine_test_stats_chunks.py --fused <dir1> <dir2> ...
    """
    args = sys.argv[1:]

    if args and args[0] == "--fused":
        dirs = [Path(d) for d in args[1:]]
        if not dirs:
            print("Usage: combine_test_stats_chunks.py --fused <dir1> <dir2> ...")
            sys.exit(1)
        failed = []
        for d in dirs:
            if combine_dir(d) == 0:
                failed.append(str(d))
        if failed:
            print(f"ERROR: no chunks combined for: {', '.join(failed)}")
            sys.exit(1)
        print(f"Done! ({len(dirs)} directories)")
        return

    if len(args) != 1:
        print("Usage: combine_test_stats_chunks.py <test_stats_dir>")
        print("   or: combine_test_stats_chunks.py --fused <dir1> <dir2> ...")
        sys.exit(1)

    if combine_dir(Path(args[0])) == 0:
        print("ERROR: No chunks combined")
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
