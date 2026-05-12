#!/usr/bin/env python3
"""
Test Statistics Derivation Worker for HPC

This script runs on HPC compute nodes (via SLURM) to derive test statistics
from full simulation data stored in Parquet format. It reads simulation outputs,
applies Python test statistic functions, and saves the results.

Usage:
    python derive_test_stats_worker.py <config_json>

The config JSON should contain:
    - simulation_pool_dir: Path to full simulation pool on HPC
    - test_stats_csv: Path to test statistics CSV
    - model_structure_file: Path to model_structure.json (species metadata with units)
    - output_dir: Path to output directory for derived test stats
    - test_stats_hash: Hash of test statistics configuration
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qsp_hpc.batch.test_stats_compute import (  # noqa: E402
    build_test_stat_registry,
    compute_test_statistics_batch,
)
from qsp_hpc.cpp.batch_runner import load_pool_manifest  # noqa: E402
from qsp_hpc.utils.logging_config import setup_logger  # noqa: E402
from qsp_hpc.utils.model_structure_units import load_units_from_model_structure  # noqa: E402

# Re-exported from qsp_hpc.batch.test_stats_compute so existing import
# paths (including mock.patch targets in tests) keep working. The
# definitions live in test_stats_compute so the inline-derive path in
# the C++ batch worker can share the same code without pulling in the
# CLI/file-I/O scaffolding below.
__all__ = ["build_test_stat_registry", "compute_test_statistics_batch"]

# Setup logger
logger = setup_logger(__name__, verbose=True)


def process_single_batch(
    batch_idx: int,
    parquet_source: Path,
    test_stats_df: pd.DataFrame,
    test_stat_registry: dict,
    species_units: dict,
    test_stats_output_dir: Path,
    template_defaults: dict[str, float] | None = None,
    aux_by_sample_index: dict[int, dict[str, float]] | None = None,
    auxiliary_units: dict[str, str] | None = None,
) -> int:
    """
    Process a single batch and save results.

    ``parquet_source`` is either:
      - a single ``batch_*.parquet`` file (legacy pre-#43 layout), or
      - a directory ``batch_*/`` containing ``chunk_NNN.parquet`` shards
        (#43 option A: combine step removed; array tasks write chunks
        directly into the batch dir).

    Args:
        batch_idx: Index of this batch (for output file naming)
        parquet_source: Path to the Parquet file OR batch subdir to process
        test_stats_df: DataFrame with test statistics configuration
        test_stat_registry: Dict mapping test_statistic_id -> compiled function
        species_units: Dict mapping species names to unit strings
        test_stats_output_dir: Directory to save output files

    Returns:
        Number of simulations processed
    """
    logger.info(f"Processing batch {batch_idx}: {parquet_source.name}")

    if parquet_source.is_dir():
        parquet_files = sorted(parquet_source.glob("chunk_*.parquet"))
        if not parquet_files:
            logger.warning(f"  No chunk_*.parquet found in {parquet_source} — skipping")
            return 0
        logger.info(f"  {len(parquet_files)} chunk file(s)")
    else:
        parquet_files = [parquet_source]

    # Stream parquets row-group-by-row-group rather than loading the full
    # file(s) with pd.read_parquet. For wide scenarios (15k-sim
    # clinical_progression: 11 GB on disk, 300 row groups, list-typed
    # time-series columns) a single pd.read_parquet blows past any
    # reasonable SLURM --mem limit — the list cells become Python lists
    # in pandas (~80 B / element) instead of packed numpy arrays
    # (~8 B / element), so the working set can reach 100+ GB. Streaming
    # keeps peak memory at one row group (~50 sims for our chunk layout).
    params_output_file = test_stats_output_dir / f"chunk_{batch_idx:03d}_params.csv"
    test_stats_output_file = test_stats_output_dir / f"chunk_{batch_idx:03d}_test_stats.csv"

    params_f = None
    params_header_written = False
    total_sims = 0
    param_cols: list[str] = []
    clean_names: list[str] = []
    has_sample_index = False

    try:
        with open(test_stats_output_file, "w") as ts_f:
            for pf_idx, parquet_file in enumerate(parquet_files):
                pf = pq.ParquetFile(str(parquet_file))
                n_row_groups = pf.num_row_groups
                schema_names = pf.schema_arrow.names
                n_sims_this_file = pf.metadata.num_rows
                logger.info(
                    f"  [{pf_idx + 1}/{len(parquet_files)}] {parquet_file.name}: "
                    f"{n_sims_this_file} sims across {n_row_groups} row group(s)"
                )

                if pf_idx == 0:
                    param_prefix = "param:"
                    param_cols = [col for col in schema_names if col.startswith(param_prefix)]
                    clean_names = [col[len(param_prefix) :] for col in param_cols]
                    has_sample_index = "sample_index" in schema_names
                    if param_cols:
                        logger.debug(
                            f"  Found {len(param_cols)} parameter columns: "
                            f"{clean_names[:5]}{'...' if len(clean_names) > 5 else ''}"
                        )
                        params_f = open(params_output_file, "w")

                for rg_idx in range(n_row_groups):
                    sim_df = pf.read_row_group(rg_idx).to_pandas()
                    # compute_test_statistics_batch uses positional indexing
                    # on its preallocated (n_sims, n_test_stats) matrix via
                    # sim_df.iterrows(); force a fresh 0..N-1 index in case
                    # pyarrow hands back a non-RangeIndex on some builds.
                    sim_df = sim_df.reset_index(drop=True)

                    if params_f is not None:
                        rg_params_df = sim_df[param_cols].copy()
                        rg_params_df.columns = clean_names
                        if has_sample_index:
                            rg_params_df.insert(
                                0,
                                "sample_index",
                                sim_df["sample_index"].astype("int64").values,
                            )
                        rg_params_df.to_csv(
                            params_f,
                            index=False,
                            header=not params_header_written,
                            float_format="%.12e",
                        )
                        params_header_written = True

                    test_stats_matrix_rg = compute_test_statistics_batch(
                        sim_df,
                        test_stats_df,
                        test_stat_registry,
                        species_units,
                        template_defaults=template_defaults,
                        aux_by_sample_index=aux_by_sample_index,
                        auxiliary_units=auxiliary_units,
                    )
                    np.savetxt(ts_f, test_stats_matrix_rg, delimiter=",", fmt="%.12e")

                total_sims += n_sims_this_file
    finally:
        if params_f is not None:
            params_f.close()

    if param_cols:
        logger.debug(f"  ✓ Parameters saved: {params_output_file.name}")
    logger.info(f"  ✓ Saved: {test_stats_output_file.name} ({total_sims} sims)")

    return total_sims


def main():
    """Main entry point for derivation worker."""
    if len(sys.argv) != 2:
        logger.error("Usage: python derive_test_stats_worker.py <config_json>")
        sys.exit(1)

    config_file = sys.argv[1]

    logger.info("🔬 Test Statistics Derivation Worker")
    logger.info(f"Node: {os.getenv('SLURMD_NODENAME', 'localhost')}")
    logger.info(f"Job ID: {os.getenv('SLURM_JOB_ID', 'local')}")

    # Load configuration
    with open(config_file, "r") as f:
        config = json.load(f)

    simulation_pool_dir = Path(config["simulation_pool_dir"])
    test_stats_csv = Path(config["test_stats_csv"])
    output_dir = Path(config["output_dir"])
    test_stats_hash = config["test_stats_hash"]
    model_structure_file_str = config.get("model_structure_file")
    model_structure_file = Path(model_structure_file_str) if model_structure_file_str else None
    max_batches = config.get("max_batches")  # None means process all batches

    logger.info(f"Simulation pool: {simulation_pool_dir}")
    logger.info(f"Test stats CSV: {test_stats_csv}")
    logger.info(f"Model structure: {model_structure_file}")
    logger.info(f"Output dir: {output_dir}")
    if max_batches is not None:
        logger.info(f"Max batches to process: {max_batches}")

    # Create output directory for this test stats hash
    test_stats_output_dir = output_dir / "test_stats" / test_stats_hash
    test_stats_output_dir.mkdir(parents=True, exist_ok=True)

    # Load units from model structure (species + compartments + parameters).
    # Covers all entities a calibration target may reference via species_dict.
    if model_structure_file and model_structure_file.exists():
        logger.info("Loading units from model structure...")
        species_units = load_units_from_model_structure(model_structure_file)
        logger.info(f"Loaded units for {len(species_units)} names")
    else:
        logger.info("No model structure file provided - using dimensionless for all names")
        species_units = {}

    # Load test statistics configuration
    logger.info("Loading test statistics configuration...")
    test_stats_df = pd.read_csv(test_stats_csv)
    logger.info(f"Found {len(test_stats_df)} test statistics")

    # Build test statistic function registry from CSV
    logger.info("Building test statistic function registry from CSV...")
    test_stat_registry = build_test_stat_registry(test_stats_df)

    # #23: load the pool's template_defaults manifest if present. Thin
    # parquets drop broadcast columns for non-sampled params; the
    # manifest is what cal-target functions fall back to when they ask
    # for one. Pre-#23 pools have no manifest — the parquet columns
    # still cover everything, so None is a safe default.
    pool_manifest = load_pool_manifest(simulation_pool_dir)
    template_defaults: dict[str, float] | None = None
    if pool_manifest is not None:
        template_defaults = {
            str(k): float(v) for k, v in pool_manifest.get("template_defaults", {}).items()
        }
        logger.info(
            "Loaded pool manifest (schema=%s): %d template defaults available",
            pool_manifest.get("schema_version", "unknown"),
            len(template_defaults),
        )
    else:
        logger.info("No pool_manifest.json found — assuming wide parquets (pre-#23 layout)")

    # Auxiliary parameter samples: optional sidecar CSV indexed by
    # sample_index, plus a {name: units} map. Both come from the
    # inference orchestrator (sbi_runner) — derive-time concerns only,
    # never seen by qsp_sim. When absent, aux-bearing wrapper code will
    # KeyError on species_dict[aux_name] for those targets, leaving
    # them NaN; non-aux targets are unaffected.
    aux_samples_csv_str = config.get("aux_samples_csv")
    auxiliary_units = config.get("auxiliary_units") or {}
    aux_by_sample_index: dict[int, dict[str, float]] = {}
    if aux_samples_csv_str:
        aux_samples_csv = Path(aux_samples_csv_str)
        if not aux_samples_csv.exists():
            logger.warning(
                "aux_samples_csv configured but not found at %s — aux-bearing "
                "test stats will fail",
                aux_samples_csv,
            )
        else:
            aux_df = pd.read_csv(aux_samples_csv)
            if "sample_index" not in aux_df.columns:
                logger.error(
                    "aux_samples_csv %s missing 'sample_index' column",
                    aux_samples_csv,
                )
                sys.exit(1)
            aux_names_csv = [c for c in aux_df.columns if c != "sample_index"]
            for row in aux_df.itertuples(index=False):
                sid = int(getattr(row, "sample_index"))
                aux_by_sample_index[sid] = {
                    name: float(getattr(row, name)) for name in aux_names_csv
                }
            logger.info(
                "Loaded aux samples for %d sims, %d aux name(s): %s",
                len(aux_by_sample_index),
                len(aux_names_csv),
                aux_names_csv,
            )
            missing_units = [n for n in aux_names_csv if n not in auxiliary_units]
            if missing_units:
                logger.warning(
                    "aux names without units in auxiliary_units (will use " "dimensionless): %s",
                    missing_units,
                )

    # Find all batches in the pool. Supports two layouts:
    #   - Legacy (pre-#43): flat batch_*.parquet files (one per submission,
    #     produced by the retired cpp_combine_batch_worker).
    #   - Current (#43 option A): batch_*/ subdirs containing
    #     chunk_NNN.parquet shards written directly by array tasks.
    # Both are enumerated and merged so derive works across mixed pools
    # (e.g. an old pool topped up with a fresh submission).
    batch_sources: list[Path] = []
    for entry in sorted(simulation_pool_dir.iterdir()):
        if not entry.name.startswith("batch_"):
            continue
        if entry.is_dir():
            batch_sources.append(entry)
        elif entry.is_file() and entry.suffix == ".parquet":
            batch_sources.append(entry)

    if not batch_sources:
        logger.error(f"No simulation batches found in {simulation_pool_dir}")
        sys.exit(1)

    # Limit to max_batches if specified
    total_available = len(batch_sources)
    if max_batches is not None and max_batches < total_available:
        batch_sources = batch_sources[:max_batches]
        logger.info(
            f"Processing {len(batch_sources)} of {total_available} batches (limited by max_batches)"
        )
    else:
        logger.info(f"Found {len(batch_sources)} simulation batches to process")

    # Process batches in a single task (no array job needed)
    total_sims = 0
    for batch_idx, parquet_source in enumerate(batch_sources):
        n_sims = process_single_batch(
            batch_idx=batch_idx,
            parquet_source=parquet_source,
            test_stats_df=test_stats_df,
            test_stat_registry=test_stat_registry,
            species_units=species_units,
            test_stats_output_dir=test_stats_output_dir,
            template_defaults=template_defaults,
            aux_by_sample_index=aux_by_sample_index,
            auxiliary_units=auxiliary_units,
        )
        total_sims += n_sims

    logger.info(
        f"✓ Derivation complete! Processed {total_sims} simulations from {len(batch_sources)} batches"
    )


if __name__ == "__main__":
    main()
