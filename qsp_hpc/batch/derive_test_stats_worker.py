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

from qsp_hpc.utils.unit_registry import ureg

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qsp_hpc.cpp.batch_runner import load_pool_manifest  # noqa: E402
from qsp_hpc.utils.logging_config import setup_logger  # noqa: E402
from qsp_hpc.utils.model_structure_units import load_units_from_model_structure  # noqa: E402


# Test statistic functions are now stored directly in the CSV
# This eliminates the need for separate test_stat_functions.py files
def build_test_stat_registry(test_stats_df: pd.DataFrame) -> dict:
    """
    Build test statistic function registry from CSV model_output_code column.

    Each row in the CSV should have:
    - test_statistic_id: Unique identifier
    - model_output_code: Python function code as string

    The function code should define a function named 'compute_test_statistic' with signature:
        def compute_test_statistic(time: np.ndarray, species_dict: dict, ureg) -> float

    Where:
        - time: numpy array of time points
        - species_dict: maps species names (e.g., 'V_T.CD8') to numpy arrays
        - ureg: Pint UnitRegistry for unit-aware calculations

    Args:
        test_stats_df: DataFrame with test statistics configuration

    Returns:
        Dictionary mapping test_statistic_id -> compiled function
    """
    registry = {}

    # Check for model_output_code column
    if "model_output_code" not in test_stats_df.columns:
        raise ValueError(
            "Test statistics CSV missing required 'model_output_code' column. "
            "This column should contain Python function code to compute test statistics."
        )

    function_col = "model_output_code"

    for _, row in test_stats_df.iterrows():
        test_stat_id = row["test_statistic_id"]

        # Check if function is provided
        if pd.isna(row[function_col]):
            raise ValueError(
                f"Test statistic '{test_stat_id}' has empty {function_col}. "
                "All test statistics must define a Python function."
            )

        function_code = row[function_col]

        try:
            # Create isolated namespace for function
            import pint

            namespace = {"np": np, "numpy": np, "pint": pint, "ureg": ureg}

            # Compile and execute the function code in an isolated namespace.
            # Security note: function_code comes from user-authored calibration target
            # definitions (YAML/CSV) and is trusted project input, not external user input.
            exec(function_code, namespace)

            # Extract the 'compute_test_statistic' function
            if "compute_test_statistic" not in namespace:
                raise ValueError(
                    f"Test statistic '{test_stat_id}': {function_col} must define "
                    "a function named 'compute_test_statistic'"
                )

            registry[test_stat_id] = namespace["compute_test_statistic"]
            logger.debug(f"Loaded function for '{test_stat_id}'")

        except Exception as e:
            logger.error(f"Failed to compile function for '{test_stat_id}': {e}")
            logger.error(f"Function code:\n{function_code}")
            raise

    logger.info(f"Built registry with {len(registry)} test statistic functions")
    return registry


# Setup logger
logger = setup_logger(__name__, verbose=True)


def compute_test_statistics_batch(
    sim_df: pd.DataFrame,
    test_stats_df: pd.DataFrame,
    test_stat_registry: dict,
    species_units: dict,
    template_defaults: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Compute test statistics for a batch of simulations.

    Args:
        sim_df: DataFrame with full simulation data (from Parquet)
                Columns: simulation_id, status, time, species_1, species_2, ...
        test_stats_df: DataFrame with test statistics configuration
                       Columns: test_statistic_id, required_species, model_output_code
        test_stat_registry: Dict mapping test_statistic_id -> compiled function
                           Functions have signature: compute_test_statistic(time, species_dict, ureg)
        species_units: Dict mapping species names to unit strings (e.g., {'V_T.CD8': 'cell'})
        template_defaults: Optional ``{name: default}`` map loaded from the
            pool's ``pool_manifest.json``. When a calibration-target
            function's ``required_species`` lists a parameter that
            isn't in ``sim_df`` (thin-parquet pools post-#23), we fall
            back to ``template_defaults[name]`` as a scalar. ``None``
            preserves the pre-#23 behavior: every parameter has to be
            a parquet column or raise.

    Returns:
        test_stats_matrix: Array of shape (n_sims, n_test_stats)
    """
    template_defaults = template_defaults or {}
    n_sims = len(sim_df)
    n_test_stats = len(test_stats_df)

    test_stats_matrix = np.full((n_sims, n_test_stats), np.nan, dtype=float)

    logger.info(f"Computing {n_test_stats} test statistics for {n_sims} simulations...")

    # ── Plan phase (once per batch) ──────────────────────────────────────────
    #
    # The previous implementation nested `for test_stat: for sim: for species:`
    # and called ``ureg.parse_expression(unit_str)`` and re-wrapped every
    # species as a fresh Pint Quantity inside the innermost loop. For the
    # 110k-sim PDAC baseline scenario that put ~60-75% of derive wall-clock
    # on Pint symbolic algebra that reduces to compile-time constants
    # (unit strings, template defaults, registry lookups never change between
    # iterations). This plan phase hoists all of it.

    def _parse_unit(species_name: str):
        unit_info = species_units.get(species_name, "dimensionless")
        unit_str = unit_info["units"] if isinstance(unit_info, dict) else unit_info
        return ureg.parse_expression(unit_str)

    time_unit = ureg.day

    # Per-test-stat metadata: (col_j, tsid, func, required_species, missing_required)
    # ``func is None`` → registry miss (logged once); ``missing_required`` is
    # the subset of required species that can't be resolved from sim_df or
    # template_defaults for this batch. Both cases leave the output column
    # as NaN without entering the hot path.
    tests_meta: list[tuple[int, str, object, list[str], list[str]]] = []
    all_required: set[str] = set()
    for j, row in test_stats_df.iterrows():
        tsid = row["test_statistic_id"]
        required = [s.strip() for s in row["required_species"].split(",")]
        func = test_stat_registry.get(tsid)
        if func is None:
            logger.warning(
                f"Test statistic '{tsid}' not found in registry. "
                "Skipping (function may have failed to compile)."
            )
        tests_meta.append((j, tsid, func, required, []))
        all_required.update(required)

    # Per-species resolution plan. Strategies:
    #   ('series', unit)        — time-series column (or scalar compartment)
    #   ('param', col, unit)    — param:<name> column (always scalar per sim)
    #   ('template', qty)       — pre-wrapped template_defaults value (reused
    #                             across every sim in the batch, zero
    #                             per-sim Pint work)
    #   ('missing',)            — unresolvable; every test stat that requires
    #                             this species fails fast with NaN
    species_plan: dict[str, tuple] = {}
    sim_cols = set(sim_df.columns)
    for s in all_required:
        if s in sim_cols:
            species_plan[s] = ("series", _parse_unit(s))
        elif f"param:{s}" in sim_cols:
            species_plan[s] = ("param", f"param:{s}", _parse_unit(s))
        elif s in template_defaults:
            species_plan[s] = ("template", float(template_defaults[s]) * _parse_unit(s))
        else:
            species_plan[s] = ("missing",)

    # Back-fill each test stat's list of unresolvable species so the inner
    # loop can short-circuit without catching ValueError.
    for meta_idx, (j, tsid, func, required, _) in enumerate(tests_meta):
        missing = [s for s in required if species_plan[s][0] == "missing"]
        tests_meta[meta_idx] = (j, tsid, func, required, missing)

    # Pre-materialize column arrays we index per sim. `.to_numpy()` on
    # list-typed columns yields an object array; indexing into it is
    # ~10x cheaper than `sim_df.iloc[i][col]` or `sim_row[col]`.
    series_cols_np = {
        s: sim_df[s].to_numpy() for s, plan in species_plan.items() if plan[0] == "series"
    }
    param_cols_np = {
        s: sim_df[plan[1]].to_numpy() for s, plan in species_plan.items() if plan[0] == "param"
    }
    time_col_np = sim_df["time"].to_numpy() if "time" in sim_cols else None
    status_np = sim_df["status"].to_numpy() if "status" in sim_cols else np.zeros(n_sims, dtype=int)

    # ── Execute phase (sim outer, test-stat inner) ──────────────────────────
    #
    # Per sim we build ``time`` and ``species_dict`` once and hand them to
    # every test stat. Pint algebra still runs per (test_stat, sim) — the
    # test stat bodies may call .to(...) or parse new unit exprs internally —
    # but the wrapping work done by the derive worker itself is now O(n_sims)
    # instead of O(n_sims × n_test_stats × n_required).

    for i in range(n_sims):
        if status_np[i] != 0:
            # status==0 = success; anything else = qsp_sim/MATLAB failure,
            # leave the whole row NaN.
            continue

        # Build time Quantity once per sim.
        try:
            if time_col_np is not None:
                time_q = np.asarray(time_col_np[i]) * time_unit
            else:
                time_q = None  # pragma: no cover — guarded by upstream
        except Exception as e:
            logger.warning(f"Error extracting time for simulation {i}: {e}")
            continue

        # Build species_dict once per sim — the union over all test stats'
        # required species. Handing extra entries to a test stat is free
        # (functions index by key), so we avoid rebuilding per test stat.
        species_dict: dict[str, object] = {}
        for s, plan in species_plan.items():
            kind = plan[0]
            if kind == "series":
                val = series_cols_np[s][i]
                # Compartment columns (e.g. V_T) and non-time-series species
                # arrive as Python/numpy scalars; time-series species arrive
                # as list-of-floats. Distinguish to avoid wrapping a scalar
                # as a 0-d array, which some test stats don't expect.
                if isinstance(val, (int, float, np.integer, np.floating)):
                    species_dict[s] = float(val) * plan[1]
                else:
                    species_dict[s] = np.asarray(val) * plan[1]
            elif kind == "param":
                species_dict[s] = float(param_cols_np[s][i]) * plan[2]
            elif kind == "template":
                # Pre-wrapped quantity — the same Pint object is handed to
                # every sim. Safe because test stats don't mutate inputs
                # (arithmetic returns new Quantities).
                species_dict[s] = plan[1]
            # 'missing' → not populated; any test stat requiring this
            # species was already marked NaN in tests_meta.

        # Dispatch each test stat against the shared time_q + species_dict.
        for j, tsid, func, required, missing in tests_meta:
            if func is None:
                continue
            if missing:
                # Fire once per (test_stat, sim) to preserve the original
                # logging volume — helps spot systematic missing-species
                # bugs without swallowing the signal.
                logger.warning(
                    f"Error computing {tsid} for simulation {i}: "
                    f"Species {missing} not found in simulation data or "
                    "pool manifest template_defaults"
                )
                continue
            try:
                result = func(time_q, species_dict, ureg)
                if hasattr(result, "magnitude"):
                    test_stats_matrix[i, j] = float(result.magnitude)
                else:
                    test_stats_matrix[i, j] = float(result)
            except Exception as e:
                logger.warning(f"Error computing {tsid} for simulation {i}: {e}")
                # test_stats_matrix[i, j] already NaN from np.full

    n_computed: int = int(np.sum(~np.isnan(test_stats_matrix)))
    n_total = test_stats_matrix.size
    logger.info(
        f"Computed {n_computed}/{n_total} test statistic values ({100*n_computed/n_total:.1f}%)"
    )

    return test_stats_matrix  # type: ignore[no-any-return]


def process_single_batch(
    batch_idx: int,
    parquet_source: Path,
    test_stats_df: pd.DataFrame,
    test_stat_registry: dict,
    species_units: dict,
    test_stats_output_dir: Path,
    template_defaults: dict[str, float] | None = None,
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
        )
        total_sims += n_sims

    logger.info(
        f"✓ Derivation complete! Processed {total_sims} simulations from {len(batch_sources)} batches"
    )


if __name__ == "__main__":
    main()
