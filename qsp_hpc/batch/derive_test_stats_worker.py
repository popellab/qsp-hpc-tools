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
    - output_dir: Path to output directory for derived test stats
    - test_stats_hash: Hash of test statistics configuration
"""

import sys
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qsp_hpc.data.test_stat_functions import get_test_stat_function


def compute_test_statistics_batch(
    sim_df: pd.DataFrame,
    test_stats_df: pd.DataFrame
) -> np.ndarray:
    """
    Compute test statistics for a batch of simulations.

    Args:
        sim_df: DataFrame with full simulation data (from Parquet)
                Columns: simulation_id, status, time, species_1, species_2, ...
        test_stats_df: DataFrame with test statistics configuration
                       Columns: test_statistic_id, required_species, ...

    Returns:
        test_stats_matrix: Array of shape (n_sims, n_test_stats)
    """
    n_sims = len(sim_df)
    n_test_stats = len(test_stats_df)

    test_stats_matrix = np.full((n_sims, n_test_stats), np.nan, dtype=float)

    print(f"   Computing {n_test_stats} test statistics for {n_sims} simulations...")

    for j, row in test_stats_df.iterrows():
        test_stat_id = row['test_statistic_id']
        required_species_str = row['required_species']

        # Parse required species (comma-separated, dots replaced with underscores)
        # Parquet columns have full names like V_T_C1 (compartment.species with dots -> underscores)
        required_species = [s.strip().replace('.', '_') for s in required_species_str.split(',')]

        # Get test statistic function
        try:
            test_stat_func = get_test_stat_function(test_stat_id)
        except KeyError as e:
            print(f"     ⚠️  Warning: {e}")
            continue

        # Apply function to each simulation
        for i, sim_row in sim_df.iterrows():
            if sim_row['status'] != 1:
                # Failed simulation - skip
                continue

            try:
                # Extract time
                time = np.array(sim_row['time'])

                # Extract required species
                species_args = [time]
                for species_name in required_species:
                    if species_name not in sim_df.columns:
                        raise ValueError(f"Species '{species_name}' not found in simulation data")
                    species_data = np.array(sim_row[species_name])
                    species_args.append(species_data)

                # Compute test statistic
                test_stat_value = test_stat_func(*species_args)
                test_stats_matrix[i, j] = test_stat_value

            except Exception as e:
                print(f"     ⚠️  Error computing {test_stat_id} for simulation {i}: {e}")
                test_stats_matrix[i, j] = np.nan

    n_computed = np.sum(~np.isnan(test_stats_matrix))
    n_total = test_stats_matrix.size
    print(f"   Computed {n_computed}/{n_total} test statistic values ({100*n_computed/n_total:.1f}%)")

    return test_stats_matrix


def main():
    """Main entry point for derivation worker."""
    if len(sys.argv) != 2:
        print("Usage: python derive_test_stats_worker.py <config_json>")
        sys.exit(1)

    config_file = sys.argv[1]

    print("🔬 Test Statistics Derivation Worker")
    print(f"   Node: {os.getenv('SLURMD_NODENAME', 'localhost')}")
    print(f"   Job ID: {os.getenv('SLURM_JOB_ID', 'local')}")
    print(f"   Array Task ID: {os.getenv('SLURM_ARRAY_TASK_ID', '0')}")

    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    simulation_pool_dir = Path(config['simulation_pool_dir'])
    test_stats_csv = Path(config['test_stats_csv'])
    output_dir = Path(config['output_dir'])
    test_stats_hash = config['test_stats_hash']

    print(f"   Simulation pool: {simulation_pool_dir}")
    print(f"   Test stats CSV: {test_stats_csv}")
    print(f"   Output dir: {output_dir}")

    # Create output directory for this test stats hash
    test_stats_output_dir = output_dir / 'test_stats' / test_stats_hash
    test_stats_output_dir.mkdir(parents=True, exist_ok=True)

    # Load test statistics configuration
    print("   Loading test statistics configuration...")
    test_stats_df = pd.read_csv(test_stats_csv)
    print(f"   Found {len(test_stats_df)} test statistics")

    # Find all Parquet files in simulation pool
    parquet_files = sorted(simulation_pool_dir.glob('batch_*.parquet'))
    if not parquet_files:
        print(f"   ⚠️  No simulation batches found in {simulation_pool_dir}")
        sys.exit(1)

    print(f"   Found {len(parquet_files)} simulation batches")

    # For array jobs, process only assigned batch
    array_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
    if array_task_id >= len(parquet_files):
        print(f"   ⚠️  Array task {array_task_id} exceeds number of batches {len(parquet_files)}")
        sys.exit(1)

    parquet_file = parquet_files[array_task_id]
    print(f"   Processing batch {array_task_id}: {parquet_file.name}")

    # Load simulation batch
    print("   Loading simulation data...")
    sim_df = pd.read_parquet(parquet_file)
    n_sims = len(sim_df)
    print(f"   Loaded {n_sims} simulations")
    print(f"   DataFrame columns ({len(sim_df.columns)}): {list(sim_df.columns)[:10]}...")  # Show first 10 columns

    # Extract parameter columns
    # Parameters are scalar columns (not lists) that are not metadata columns
    metadata_cols = {'simulation_id', 'status', 'time'}
    param_cols = []
    for col in sim_df.columns:
        if col not in metadata_cols:
            # Check if column contains scalar values (not lists)
            sample_val = sim_df[col].iloc[0]
            if not isinstance(sample_val, (list, np.ndarray)):
                param_cols.append(col)

    if param_cols:
        print(f"   Found {len(param_cols)} parameter columns: {param_cols[:5]}{'...' if len(param_cols) > 5 else ''}")

        # Save parameters to chunk_XXX_params.csv
        params_output_file = test_stats_output_dir / f"chunk_{array_task_id:03d}_params.csv"
        print(f"   Saving parameters to {params_output_file}...")

        # Extract parameter values (n_sims x n_params)
        params_df = sim_df[param_cols]
        params_df.to_csv(params_output_file, index=False, float_format='%.12e')

        print(f"   ✓ Parameters saved: {params_output_file}")
    else:
        print("   No parameter columns found in Parquet file (may be older format)")

    # Compute test statistics
    test_stats_matrix = compute_test_statistics_batch(sim_df, test_stats_df)

    # Save results
    output_file = test_stats_output_dir / f"chunk_{array_task_id:03d}_test_stats.csv"
    print(f"   Saving results to {output_file}...")

    # Save as CSV (n_sims x n_test_stats)
    np.savetxt(output_file, test_stats_matrix, delimiter=',', fmt='%.12e')

    print(f"   ✓ Test statistics saved: {output_file}")
    print(f"   Derivation complete!")


if __name__ == '__main__':
    main()
