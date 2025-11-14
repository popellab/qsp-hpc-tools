#!/usr/bin/env python3
"""
Update simulation pool manifest on HPC.

This script scans a simulation pool directory for Parquet files and creates/updates
the manifest.json file with metadata about available simulations.

The manifest enables fast checking for simulation availability without parsing filenames.

Usage:
    python update_manifest.py <pool_directory>
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def parse_parquet_filename(filename: str) -> Dict:
    """
    Parse Parquet filename to extract metadata.

    Format: batch_XXX_YYYYMMDD_HHMMSS_NNNsims_seedSSS.parquet

    Args:
        filename: Parquet filename

    Returns:
        Dictionary with metadata
    """
    # Extract components using regex
    pattern = r'batch_(\d+)_(\d{8})_(\d{6})_(\d+)sims_seed(\d+)\.parquet'
    match = re.match(pattern, filename)

    if not match:
        return None

    batch_id = int(match.group(1))
    date = match.group(2)
    time = match.group(3)
    n_sims = int(match.group(4))
    seed = int(match.group(5))

    return {
        'batch_id': batch_id,
        'filename': filename,
        'n_simulations': n_sims,
        'seed': seed,
        'timestamp': f"{date}_{time}"
    }


def update_manifest(pool_dir: Path) -> None:
    """
    Update manifest.json in simulation pool directory.

    Args:
        pool_dir: Path to simulation pool directory
    """
    if not pool_dir.exists():
        print(f"Error: Pool directory not found: {pool_dir}")
        sys.exit(1)

    # Find all Parquet files
    parquet_files = sorted(pool_dir.glob('batch_*.parquet'))

    if not parquet_files:
        print(f"Warning: No Parquet files found in {pool_dir}")
        return

    # Parse filenames
    batches = []
    total_simulations = 0

    for pf in parquet_files:
        metadata = parse_parquet_filename(pf.name)
        if metadata:
            batches.append(metadata)
            total_simulations += metadata['n_simulations']
        else:
            print(f"Warning: Could not parse filename: {pf.name}")

    # Create manifest
    manifest = {
        'total_simulations': total_simulations,
        'n_batches': len(batches),
        'batches': batches,
        'last_updated': datetime.now().isoformat(),
        'pool_path': str(pool_dir)
    }

    # Write manifest
    manifest_file = pool_dir / 'manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Manifest updated: {manifest_file}")
    print(f"  Total simulations: {total_simulations}")
    print(f"  Batches: {len(batches)}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python update_manifest.py <pool_directory>")
        sys.exit(1)

    pool_dir = Path(sys.argv[1])
    update_manifest(pool_dir)


if __name__ == '__main__':
    main()
