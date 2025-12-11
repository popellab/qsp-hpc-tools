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
    - species_units_file: Path to species_units.json (maps species names to unit strings)
    - output_dir: Path to output directory for derived test stats
    - test_stats_hash: Hash of test statistics configuration
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qsp_hpc.utils.unit_registry import ureg

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qsp_hpc.utils.logging_config import setup_logger  # noqa: E402


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
            "This column should contain Python function code to compute test statistics. "
            "See docs/TEST_STATISTICS_CSV_FORMAT.md for format specification."
        )

    function_col = "model_output_code"

    for _, row in test_stats_df.iterrows():
        test_stat_id = row["test_statistic_id"]

        # Check if function is provided
        if pd.isna(row[function_col]):
            raise ValueError(
                f"Test statistic '{test_stat_id}' has empty {function_col}. "
                "All test statistics must define a Python function. "
                "See docs/TEST_STATISTICS_CSV_FORMAT.md for examples."
            )

        function_code = row[function_col]

        try:
            # Create isolated namespace for function
            import pint

            namespace = {"np": np, "numpy": np, "pint": pint, "ureg": ureg}

            # Compile and execute the function code
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

    Returns:
        test_stats_matrix: Array of shape (n_sims, n_test_stats)
    """
    n_sims = len(sim_df)
    n_test_stats = len(test_stats_df)

    test_stats_matrix = np.full((n_sims, n_test_stats), np.nan, dtype=float)

    logger.info(f"Computing {n_test_stats} test statistics for {n_sims} simulations...")

    for j, row in test_stats_df.iterrows():
        test_stat_id = row["test_statistic_id"]
        required_species_str = row["required_species"]

        # Parse required species (comma-separated)
        # Species names use dots (e.g., V_T.CD8) in both CSV and parquet columns
        required_species = [s.strip() for s in required_species_str.split(",")]

        # Get test statistic function from registry
        if test_stat_id not in test_stat_registry:
            logger.warning(
                f"Test statistic '{test_stat_id}' not found in registry. "
                "Skipping (function may have failed to compile)."
            )
            continue

        test_stat_func = test_stat_registry[test_stat_id]

        # Apply function to each simulation
        for i, sim_row in sim_df.iterrows():
            if sim_row["status"] != 1:
                # Failed simulation - skip
                continue

            try:
                # Extract time with units (days)
                time = np.array(sim_row["time"]) * ureg.day

                # Build species_dict with Pint Quantities
                # Also supports scalar parameters (not just time-series species)
                species_dict = {}
                for species_name in required_species:
                    if species_name not in sim_df.columns:
                        raise ValueError(f"Species '{species_name}' not found in simulation data")
                    val = sim_row[species_name]

                    # Get unit string from species_units, default to dimensionless
                    unit_str = species_units.get(species_name, "dimensionless")
                    unit = ureg.parse_expression(unit_str)

                    # Check if it's a scalar (parameter) or array (species time-series)
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        # Scalar parameter - wrap with units
                        species_dict[species_name] = float(val) * unit
                    else:
                        # Time-series species - convert to array with units
                        species_dict[species_name] = np.array(val) * unit

                # Compute test statistic (signature: time, species_dict, ureg)
                result = test_stat_func(time, species_dict, ureg)

                # Extract magnitude from Pint Quantity result
                if hasattr(result, "magnitude"):
                    test_stat_value = float(result.magnitude)
                else:
                    test_stat_value = float(result)
                test_stats_matrix[i, j] = test_stat_value

            except Exception as e:
                logger.warning(f"Error computing {test_stat_id} for simulation {i}: {e}")
                test_stats_matrix[i, j] = np.nan

    n_computed: int = int(np.sum(~np.isnan(test_stats_matrix)))
    n_total = test_stats_matrix.size
    logger.info(
        f"Computed {n_computed}/{n_total} test statistic values ({100*n_computed/n_total:.1f}%)"
    )

    return test_stats_matrix  # type: ignore[no-any-return]


def process_single_batch(
    batch_idx: int,
    parquet_file: Path,
    test_stats_df: pd.DataFrame,
    test_stat_registry: dict,
    species_units: dict,
    test_stats_output_dir: Path,
) -> int:
    """
    Process a single batch file and save results.

    Args:
        batch_idx: Index of this batch (for output file naming)
        parquet_file: Path to the Parquet file to process
        test_stats_df: DataFrame with test statistics configuration
        test_stat_registry: Dict mapping test_statistic_id -> compiled function
        species_units: Dict mapping species names to unit strings
        test_stats_output_dir: Directory to save output files

    Returns:
        Number of simulations processed
    """
    logger.info(f"Processing batch {batch_idx}: {parquet_file.name}")

    # Load simulation batch
    sim_df = pd.read_parquet(parquet_file)
    n_sims = len(sim_df)
    logger.info(f"  Loaded {n_sims} simulations")

    # Extract parameter columns
    # Parameters are scalar columns (not lists) that are not metadata columns
    metadata_cols = {"simulation_id", "status", "time"}
    param_cols = []
    for col in sim_df.columns:
        if col not in metadata_cols:
            # Check if column contains scalar values (not lists)
            sample_val = sim_df[col].iloc[0]
            if not isinstance(sample_val, (list, np.ndarray)):
                param_cols.append(col)

    if param_cols:
        logger.debug(
            f"  Found {len(param_cols)} parameter columns: "
            f"{param_cols[:5]}{'...' if len(param_cols) > 5 else ''}"
        )

        # Save parameters to chunk_XXX_params.csv
        params_output_file = test_stats_output_dir / f"chunk_{batch_idx:03d}_params.csv"

        # Extract parameter values (n_sims x n_params)
        params_df = sim_df[param_cols]
        params_df.to_csv(params_output_file, index=False, float_format="%.12e")

        logger.debug(f"  ✓ Parameters saved: {params_output_file.name}")

    # Compute test statistics
    test_stats_matrix = compute_test_statistics_batch(
        sim_df, test_stats_df, test_stat_registry, species_units
    )

    # Save results
    output_file = test_stats_output_dir / f"chunk_{batch_idx:03d}_test_stats.csv"

    # Save as CSV (n_sims x n_test_stats)
    np.savetxt(output_file, test_stats_matrix, delimiter=",", fmt="%.12e")

    logger.info(f"  ✓ Saved: {output_file.name}")

    return n_sims


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
    species_units_file_str = config.get("species_units_file")
    species_units_file = Path(species_units_file_str) if species_units_file_str else None

    logger.info(f"Simulation pool: {simulation_pool_dir}")
    logger.info(f"Test stats CSV: {test_stats_csv}")
    logger.info(f"Species units: {species_units_file}")
    logger.info(f"Output dir: {output_dir}")

    # Create output directory for this test stats hash
    test_stats_output_dir = output_dir / "test_stats" / test_stats_hash
    test_stats_output_dir.mkdir(parents=True, exist_ok=True)

    # Load species units (optional - use empty dict if not provided)
    if species_units_file and species_units_file.exists():
        logger.info("Loading species units...")
        with open(species_units_file, "r") as f:
            species_units = json.load(f)
        logger.info(f"Loaded units for {len(species_units)} species")
    else:
        logger.info("No species units file provided - using dimensionless for all species")
        species_units = {}

    # Load test statistics configuration
    logger.info("Loading test statistics configuration...")
    test_stats_df = pd.read_csv(test_stats_csv)
    logger.info(f"Found {len(test_stats_df)} test statistics")

    # Build test statistic function registry from CSV
    logger.info("Building test statistic function registry from CSV...")
    test_stat_registry = build_test_stat_registry(test_stats_df)

    # Find all Parquet files in simulation pool
    parquet_files = sorted(simulation_pool_dir.glob("batch_*.parquet"))
    if not parquet_files:
        logger.error(f"No simulation batches found in {simulation_pool_dir}")
        sys.exit(1)

    logger.info(f"Found {len(parquet_files)} simulation batches to process")

    # Process ALL batches in a single task (no array job needed)
    total_sims = 0
    for batch_idx, parquet_file in enumerate(parquet_files):
        n_sims = process_single_batch(
            batch_idx=batch_idx,
            parquet_file=parquet_file,
            test_stats_df=test_stats_df,
            test_stat_registry=test_stat_registry,
            species_units=species_units,
            test_stats_output_dir=test_stats_output_dir,
        )
        total_sims += n_sims

    logger.info(
        f"✓ Derivation complete! Processed {total_sims} simulations from {len(parquet_files)} batches"
    )


if __name__ == "__main__":
    main()
