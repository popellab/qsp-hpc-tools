#!/usr/bin/env python3
"""
Combine Test Statistics Chunks

This script runs on HPC to combine chunk files from test statistics derivation
into single combined files.

Combines:
- chunk_XXX_test_stats.csv files → combined_test_stats.csv (no headers)
- chunk_XXX_params.csv files → combined_params.csv (has headers)

Usage:
    python combine_test_stats_chunks.py <test_stats_dir>

Args:
    test_stats_dir: Directory containing chunk files
"""

import glob
import sys
from pathlib import Path


def combine_test_stats(test_stats_dir: Path) -> int:
    """
    Combine test statistics chunk files (no headers).

    Args:
        test_stats_dir: Directory containing chunk files

    Returns:
        Number of chunks combined
    """
    chunk_files = sorted(glob.glob(str(test_stats_dir / "chunk_*_test_stats.csv")))

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
    chunk_files = sorted(glob.glob(str(test_stats_dir / "chunk_*_params.csv")))

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


def main():
    """Main entry point for combining chunk files."""
    if len(sys.argv) != 2:
        print("Usage: combine_test_stats_chunks.py <test_stats_dir>")
        sys.exit(1)

    test_stats_dir = Path(sys.argv[1])

    if not test_stats_dir.exists():
        print(f"ERROR: Directory not found: {test_stats_dir}")
        sys.exit(1)

    print(f"Combining chunks in: {test_stats_dir}")

    # Combine both file types
    n_test_stats = combine_test_stats(test_stats_dir)
    combine_params(test_stats_dir)

    if n_test_stats == 0:
        print("ERROR: No chunks combined")
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
