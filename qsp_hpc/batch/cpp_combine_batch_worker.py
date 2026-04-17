"""Post-array combine worker for C++ simulation batches.

Each SLURM array task (``cpp_batch_worker``) writes its chunk of
simulations to a per-submission staging directory::

    {pool_dir}/.staging/{SLURM_ARRAY_JOB_ID}/chunk_{NNN}.parquet

This worker runs once per submission (chained via ``--dependency=afterok``)
and consolidates all chunks into a single pool-level batch parquet
matching the MATLAB ``QSPSimulator`` naming convention::

    {pool_dir}/batch_{YYYYMMDD_HHMMSS}_{scenario}_{N}sims_seed{S}.parquet

The staging directory is removed after a successful combine.  Test-stats
derivation runs ``afterok`` on this combine job, so it sees exactly one
new batch file per submission — no task-id shards.

Usage (invoked by the generated SLURM script)::

    python -m qsp_hpc.batch.cpp_combine_batch_worker combine_config.json

Config JSON schema::

    {
        "pool_base": "/abs/path/to/pools",
        "pool_id": "v1_<hash>_<scenario>",
        "staging_dir": "/abs/path/to/pools/v1_.../.staging/12345678",
        "scenario": "baseline",
        "n_simulations": 1000,
        "seed": 2025,
        "expected_chunks": 50
    }
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from qsp_hpc.utils.logging_config import setup_logger

logger = setup_logger(__name__, verbose=True)


def combine_chunks(config: dict) -> Path:
    """Concatenate chunk parquets into a single batch file.

    Returns the path of the consolidated batch parquet.
    """
    pool_base = Path(config["pool_base"])
    pool_id = config["pool_id"]
    staging_dir = Path(config["staging_dir"])
    scenario = config["scenario"]
    n_sims = int(config["n_simulations"])
    seed = int(config["seed"])
    expected_chunks = int(config.get("expected_chunks", 0))

    pool_dir = pool_base / pool_id

    if not staging_dir.is_dir():
        raise FileNotFoundError(f"Staging directory not found: {staging_dir}")

    chunk_files = sorted(staging_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        raise RuntimeError(f"No chunk parquets in staging dir: {staging_dir}")

    if expected_chunks and len(chunk_files) != expected_chunks:
        logger.warning(
            "Found %d chunks but expected %d — proceeding with what's present",
            len(chunk_files),
            expected_chunks,
        )

    logger.info("Combining %d chunks from %s", len(chunk_files), staging_dir)

    tables = []
    total_rows = 0
    for cf in chunk_files:
        t = pq.read_table(str(cf))
        tables.append(t)
        total_rows += t.num_rows

    combined = pa.concat_tables(tables, promote_options="default")
    logger.info("  combined rows: %d (sum of chunk rows: %d)", combined.num_rows, total_rows)

    # Re-number simulation_id so the combined batch has a contiguous range.
    # Individual chunks reuse local ids 0..chunk_size-1; downstream tools
    # that group by simulation_id need them unique within a batch file.
    if "simulation_id" in combined.column_names:
        new_ids = pa.array(range(combined.num_rows), type=pa.int64())
        idx = combined.column_names.index("simulation_id")
        combined = combined.set_column(idx, "simulation_id", new_ids)

    # Filename encodes the ACTUAL number of rows written (combined.num_rows),
    # not n_sims from the config. When some array tasks fail to flush their
    # parquet (e.g. wall-time kills on oversubscribed nodes), the combined
    # batch has fewer rows than requested. Trusting the config count here
    # would cause count_pool_simulations (which parses `{N}sims` from the
    # filename) to overreport, and the Tier 2 check_hpc_test_stats would
    # then keep nuking the derived test_stats and re-deriving them
    # indefinitely because derived_count < filename_count.
    #
    # Microsecond-resolution timestamp prevents filename collisions when
    # two combines for the same pool complete within the same wall-second
    # (e.g. quick top-up after the initial batch finishes).
    n_rows_written = combined.num_rows
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"batch_{ts}_{scenario}_{n_rows_written}sims_seed{seed}.parquet"
    output_path = pool_dir / filename
    if n_rows_written != n_sims:
        logger.warning(
            "Actual row count %d differs from requested %d "
            "(%d chunks lost) — naming batch by actual count",
            n_rows_written,
            n_sims,
            n_sims - n_rows_written,
        )

    # Atomic write: temp file in same dir, then rename. Prevents a reader
    # from seeing a partial parquet if the process is killed mid-write.
    pool_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    pq.write_table(combined, str(tmp_path))
    tmp_path.replace(output_path)
    logger.info("Wrote consolidated batch: %s", output_path)

    # Cleanup staging. Best-effort — log and continue if it fails.
    try:
        shutil.rmtree(staging_dir)
        logger.info("Removed staging dir: %s", staging_dir)
    except OSError as exc:
        logger.warning("Failed to remove staging dir %s: %s", staging_dir, exc)

    return output_path


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m qsp_hpc.batch.cpp_combine_batch_worker <config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)

    logger.info("C++ Combine Batch Worker")
    logger.info("  pool_base:    %s", config["pool_base"])
    logger.info("  pool_id:      %s", config["pool_id"])
    logger.info("  staging_dir:  %s", config["staging_dir"])
    logger.info("  scenario:     %s", config["scenario"])
    logger.info("  n_simulations:%d", config["n_simulations"])

    combine_chunks(config)


if __name__ == "__main__":
    main()
