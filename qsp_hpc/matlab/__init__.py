"""
MATLAB worker scripts for HPC job execution.

This package contains MATLAB functions that run on HPC nodes:
- batch_worker.m: Main SLURM array job worker
- extract_all_species_arrays.m: Extract simulation timecourse data
- save_species_to_parquet.m: Save data in Parquet format for Python
- load_parameter_samples_csv.m: Load parameter samples from CSV
"""

import os
from pathlib import Path

def get_matlab_path() -> Path:
    """Get path to MATLAB scripts directory."""
    return Path(__file__).parent
