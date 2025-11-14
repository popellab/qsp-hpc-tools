# HPC Python Environment Setup

This document explains the Python environment setup for HPC-based QSP simulation caching.

## Overview

The HPC caching system requires Python packages (numpy, pandas, pyarrow, scipy) for:
- Saving full simulations to Parquet format
- Deriving test statistics from full simulations

To avoid package conflicts and ensure reproducibility, these packages are installed in a dedicated virtual environment on HPC using **uv** (a fast Python package manager that can also manage Python versions).

## Prerequisites

**uv must be installed on HPC.** If not already installed:

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc

# Verify installation
uv --version
```

uv will automatically download and use Python 3.11 for the virtual environment, solving issues with old system Python versions.

## Automatic Setup

**The virtual environment is set up automatically** when you first run simulations on HPC. The system will:

1. Check if `~/qspio_venv` exists on HPC
2. If not found, automatically create it and install required packages
3. Use the venv for all subsequent Python operations

You'll see logging like:
```
🐍 Checking HPC Python environment...
→ Setting up HPC Python environment (first time only)...
✓ HPC Python environment configured
```

## Manual Setup (Optional)

If you prefer to set up the environment manually, or if automatic setup fails:

```bash
# SSH to HPC
ssh your-hpc-cluster

# Clone/navigate to project
cd ~/qspio-pdac

# Run setup script
bash metadata/hpc-setup/setup_hpc_venv.sh
```

This creates `~/qspio_venv` and installs packages from `requirements_hpc.txt`.

## Requirements

The following packages are installed in the HPC venv:

- **numpy** (≥1.24.0) - Array operations
- **pandas** (≥2.0.0) - DataFrame operations
- **pyarrow** (≥12.0.0) - Parquet I/O
- **scipy** (≥1.10.0) - Scientific computing

See `requirements_hpc.txt` for exact versions.

## Virtual Environment Location

```
~/qspio_venv/               # Virtual environment root
├── bin/
│   └── python              # Python interpreter used by MATLAB and SLURM jobs
├── lib/
│   └── python3.x/
│       └── site-packages/  # Installed packages
└── ...
```

## How It's Used

### 1. MATLAB Parquet Saving
`save_species_to_parquet.m` automatically detects and uses venv Python:
```matlab
venv_python = fullfile(getenv('HOME'), 'qspio_venv', 'bin', 'python');
if exist(venv_python, 'file')
    % Use venv Python
else
    % Fall back to system Python (local mode)
end
```

### 2. SLURM Derivation Jobs
The derivation SLURM script activates the venv:
```bash
source "$HOME/qspio_venv/bin/activate"
python core/batch/derive_test_stats_worker.py ...
```

## Troubleshooting

### Package Import Errors
If you see `ModuleNotFoundError` on HPC:

1. Check if venv exists:
   ```bash
   ls -la ~/qspio_venv
   ```

2. Re-run setup:
   ```bash
   cd ~/qspio-pdac
   bash metadata/hpc-setup/setup_hpc_venv.sh
   ```

3. Verify packages:
   ```bash
   source ~/qspio_venv/bin/activate
   python -c "import numpy, pandas, pyarrow, scipy; print('OK')"
   ```

### Setup Script Fails

If `setup_hpc_venv.sh` fails:

1. Check Python 3 is available:
   ```bash
   python3 --version
   ```

2. Create venv manually:
   ```bash
   python3 -m venv ~/qspio_venv
   source ~/qspio_venv/bin/activate
   pip install numpy pandas pyarrow scipy
   ```

### Permission Issues

If you don't have write access to `$HOME`:
1. Edit `setup_hpc_venv.sh` to use a different location
2. Update `save_species_to_parquet.m` and `_generate_derivation_slurm_script()` accordingly

## Updating Packages

To update packages in the venv:

```bash
source ~/qspio_venv/bin/activate
pip install --upgrade numpy pandas pyarrow scipy
```

Or re-run the setup script:
```bash
bash metadata/hpc-setup/setup_hpc_venv.sh  # Will update existing venv
```
