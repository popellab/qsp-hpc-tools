#!/usr/bin/env python3
"""
Trajectory Grid Derivation Worker for HPC

Extracts a dense species × timepoint grid from full simulation data stored
in Parquet format. Unlike the test statistics worker (which computes scalar
summaries), this worker extracts raw species values at specified timepoints,
enabling downstream MI analysis for optimal experimental design (OBED).

Usage:
    python derive_trajectory_grid_worker.py <config_json>

The config JSON should contain:
    - simulation_pool_dir: Path to full simulation pool on HPC
    - species_list: List of species names to extract (or "all" for all species)
    - time_grid: List of timepoints to extract (days), or "daily" for integer days
    - tumor_volume_species: Species name for tumor volume (for response classification)
    - output_dir: Path to output directory
    - scenario_name: Name of the scenario (for output labeling)

Output:
    - trajectory_grid.parquet: (n_sims, n_species * n_timepoints) matrix
      Column names: "{species}__t{timepoint}" (e.g., "V_T.CD8__t14")
    - trajectory_meta.json: Species list, time grid, and column mapping
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qsp_hpc.utils.logging_config import setup_logger

logger = setup_logger(__name__, verbose=True)


def extract_trajectory_grid_batch(
    sim_df: pd.DataFrame,
    species_list: list[str],
    time_grid: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Extract species values at specified timepoints for a batch of simulations.

    For each simulation, interpolates species time series to the requested
    time grid using linear interpolation.

    Args:
        sim_df: DataFrame with full simulation data (from Parquet).
                Columns: simulation_id, status, time, species_1, species_2, ...
        species_list: List of species column names to extract
        time_grid: Array of timepoints (in days) at which to evaluate

    Returns:
        grid_matrix: (n_sims, n_species * n_timepoints) array
        column_names: List of column names "{species}__t{timepoint}"
    """
    n_sims = len(sim_df)
    n_species = len(species_list)
    n_times = len(time_grid)
    n_cols = n_species * n_times

    grid_matrix = np.full((n_sims, n_cols), np.nan, dtype=np.float64)

    # Build column names
    column_names = []
    for species in species_list:
        for t in time_grid:
            column_names.append(f"{species}__t{t:.1f}")

    for i, (_, sim_row) in enumerate(sim_df.iterrows()):
        if sim_row["status"] != 1:
            continue

        try:
            sim_time = np.array(sim_row["time"], dtype=np.float64)

            col_idx = 0
            for species in species_list:
                if species not in sim_df.columns:
                    col_idx += n_times
                    continue

                values = np.array(sim_row[species], dtype=np.float64)

                # Interpolate to time grid
                interp_values = np.interp(
                    time_grid,
                    sim_time,
                    values,
                    left=np.nan,
                    right=np.nan,
                )

                grid_matrix[i, col_idx : col_idx + n_times] = interp_values
                col_idx += n_times

        except Exception as e:
            logger.warning(f"Error extracting trajectories for simulation {i}: {e}")

    n_valid = int(np.sum(~np.all(np.isnan(grid_matrix), axis=1)))
    logger.info(
        f"Extracted trajectory grid: {n_valid}/{n_sims} valid simulations, "
        f"{n_species} species × {n_times} timepoints = {n_cols} columns"
    )

    return grid_matrix, column_names


def discover_species(sim_df: pd.DataFrame) -> list[str]:
    """Discover all species columns in simulation data.

    Species columns are those that are not metadata (simulation_id, status,
    time) and not parameters (param:*).

    Args:
        sim_df: DataFrame with simulation data

    Returns:
        Sorted list of species column names
    """
    meta_cols = {"simulation_id", "status", "time"}
    species = [
        col for col in sim_df.columns if col not in meta_cols and not col.startswith("param:")
    ]
    return sorted(species)


def process_pool(
    pool_dir: Path,
    species_list: list[str] | str,
    time_grid: np.ndarray,
    output_dir: Path,
    scenario_name: str = "",
) -> Path:
    """Process all batches in a simulation pool.

    Args:
        pool_dir: Path to simulation pool directory containing Parquet files
        species_list: List of species to extract, or "all"
        time_grid: Array of timepoints (days)
        output_dir: Where to save output files
        scenario_name: For labeling output

    Returns:
        Path to output trajectory_grid.parquet
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all batch Parquet files
    parquet_files = sorted(pool_dir.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No batch_*.parquet files found in {pool_dir}")

    logger.info(f"Found {len(parquet_files)} batch files in {pool_dir}")

    # Auto-discover species if needed
    if species_list == "all":
        first_batch = pd.read_parquet(parquet_files[0])
        species_list = discover_species(first_batch)
        logger.info(f"Auto-discovered {len(species_list)} species")

    # Process each batch
    all_grids = []
    all_params = []
    column_names = None

    for batch_file in parquet_files:
        logger.info(f"Processing {batch_file.name}...")
        sim_df = pd.read_parquet(batch_file)

        grid, col_names = extract_trajectory_grid_batch(sim_df, species_list, time_grid)
        all_grids.append(grid)

        if column_names is None:
            column_names = col_names

        # Extract parameters if present
        param_cols = [c for c in sim_df.columns if c.startswith("param:")]
        if param_cols:
            params = sim_df[param_cols].values
            all_params.append(params)

    # Combine all batches
    grid_matrix = np.vstack(all_grids)
    logger.info(f"Combined trajectory grid: {grid_matrix.shape}")

    # Save as Parquet
    grid_df = pd.DataFrame(grid_matrix, columns=column_names)

    # Add parameters if available
    if all_params:
        params_matrix = np.vstack(all_params)
        param_names = [c.replace("param:", "") for c in param_cols]
        for j, pname in enumerate(param_names):
            grid_df[f"param:{pname}"] = params_matrix[:, j]

    output_path = output_dir / "trajectory_grid.parquet"
    grid_df.to_parquet(output_path, index=False)
    logger.info(f"Saved trajectory grid: {output_path}")

    # Save metadata
    meta = {
        "scenario_name": scenario_name,
        "species_list": list(species_list),
        "time_grid": time_grid.tolist(),
        "n_simulations": grid_matrix.shape[0],
        "n_columns": grid_matrix.shape[1],
        "column_names": column_names,
    }
    meta_path = output_dir / "trajectory_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")

    return output_path


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config_json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    pool_dir = Path(config["simulation_pool_dir"])
    output_dir = Path(config["output_dir"])
    species_list = config.get("species_list", "all")
    scenario_name = config.get("scenario_name", "")

    # Build time grid
    time_grid_config = config.get("time_grid", "daily")
    if time_grid_config == "daily":
        stop_time = config.get("stop_time", 21)
        time_grid = np.arange(0, stop_time + 1, 1.0)
    elif isinstance(time_grid_config, list):
        time_grid = np.array(time_grid_config, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported time_grid config: {time_grid_config}")

    process_pool(pool_dir, species_list, time_grid, output_dir, scenario_name)


if __name__ == "__main__":
    main()
