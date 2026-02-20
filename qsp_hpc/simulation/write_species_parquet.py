#!/usr/bin/env python3
"""
Python helper to write species data to Parquet format from MATLAB.

This script is called by MATLAB batch_worker.m to save full simulation
outputs to Parquet format on HPC.
"""

import json
import sys

import numpy as np
import pandas as pd


def write_species_parquet(json_file: str, output_file: str) -> None:
    """
    Write species data to Parquet format.

    Args:
        json_file: Path to JSON file with species data structure
        output_file: Path to output Parquet file

    The JSON file should contain:
    {
        "n_sims": int,
        "n_species": int,
        "species_names": [str, ...],
        "param_names": [str, ...],  # Parameter names (optional)
        "param_values": [[float, ...], ...],  # n_sims x n_params (optional)
        "time_arrays": [[float, ...], ...],  # n_sims x n_timepoints
        "species_arrays": [[[float, ...], ...], ...],  # n_sims x n_species x n_timepoints
        "status": [int, ...]  # n_sims
    }
    """
    # Load JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    n_sims = data["n_sims"]
    species_names = data["species_names"]
    time_arrays = data["time_arrays"]
    species_arrays = data["species_arrays"]
    status = data["status"]

    # Ensure arrays are lists (MATLAB jsonencode may create scalars for n_sims=1)
    if not isinstance(status, list):
        status = [status]
    if not isinstance(time_arrays, list):
        time_arrays = [time_arrays]
    if not isinstance(species_arrays, list):
        species_arrays = [species_arrays]

    # Extract parameter data (optional)
    param_names = data.get("param_names", [])
    param_values = data.get("param_values", [])

    # Convert param_values to 2D array if needed (MATLAB jsonencode flattens 1xN matrices)
    if param_values and param_names:
        param_values = np.array(param_values)
        # If 1D (single simulation), reshape to 2D
        if param_values.ndim == 1:
            param_values = param_values.reshape(1, -1)
        # If still not matching dimensions, reshape
        if param_values.shape[0] != n_sims:
            # Assume row-major order (MATLAB default)
            param_values = param_values.reshape(n_sims, len(param_names))

    # Build Parquet table
    # Schema: simulation_id | param:name_1 | param:name_2 | ... | status | time | species_1 | ...
    # Parameter columns are prefixed with "param:" to avoid name collisions with species
    # columns (e.g., "k_C1_growth" exists as both a parameter and a SimBiology species).
    # Without the prefix, species list columns overwrite parameter scalar columns.
    param_prefix = "param:"

    records = []
    for i in range(n_sims):
        record = {"simulation_id": i, "status": status[i]}

        # Add parameters (if provided) - prefix with "param:" to avoid species collisions
        if param_names and len(param_values) > 0:
            for j, param_name in enumerate(param_names):
                record[f"{param_prefix}{param_name}"] = float(param_values[i][j])

        # Add time array
        record["time"] = time_arrays[i] if time_arrays[i] else []

        # Add each species (keep dot notation to match SimBiology convention)
        for j, species_name in enumerate(species_names):
            species_data = (
                species_arrays[i][j]
                if (i < len(species_arrays) and j < len(species_arrays[i]))
                else []
            )
            record[species_name] = species_data if species_data else []

        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Write to Parquet with compression
    df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)

    print(f"   ✓ Saved {n_sims} simulations to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python write_species_parquet.py <json_file> <output_parquet>")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        write_species_parquet(json_file, output_file)
    except Exception as e:
        print(f"Error writing Parquet: {e}", file=sys.stderr)
        sys.exit(1)
