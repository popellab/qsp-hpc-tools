# Architecture

## Overview

QSP HPC Tools manages large-scale MATLAB-based QSP simulation campaigns on SLURM clusters, with intelligent caching and a Python-callable interface.

## Core Components

### 1. SimulationPoolManager

Manages local simulation caching with scenario support.

**File naming:** `batch_{timestamp}_{scenario}_{n_sims}sims_seed{seed}.mat`

**Pool directories:** `{model_version}_{config_hash[:8]}/`

**Location:** `qsp_hpc/simulation/simulation_pool.py`

### 2. QSPSimulator

Main callable interface. Implements 3-tier caching strategy and automatic HPC job submission.

**Location:** `qsp_hpc/simulation/qsp_simulator.py`

### 3. HPCJobManager

SSH/SLURM integration for job management. Handles codebase syncing, job submission, monitoring, and result collection.

**Location:** `qsp_hpc/batch/hpc_job_manager.py`

### 4. MATLAB Workers

- `batch_worker.m`: Main SLURM array job worker
- `extract_all_species_arrays.m`: Extracts simulation timecourse data
- `save_species_to_parquet.m`: MATLAB→Python bridge for Parquet writing

**Location:** `qsp_hpc/matlab/`

## Caching Strategy

### Three-Tier System

1. **Local Pool** (SimulationPoolManager)
   - Test statistics cached locally in `.mat` files
   - Fast access, scenario-specific

2. **HPC Test Statistics**
   - Derived statistics stored on HPC for quick downloads
   - Shared across scenarios when parameter sets match

3. **HPC Full Simulations**
   - Complete timecourse data for deriving new test statistics
   - Shared pool across all uses (training, prior PPCs)

4. **On-Demand Execution**
   - Run new simulations only when needed
   - Automatic job submission and monitoring

### Design Principles

1. **Content-based Hashing**: Configuration hashes ensure cache invalidation when semantic changes occur
2. **Manifest-free Design**: All metadata encoded in filenames for simplicity
3. **Multi-scenario Support**: Same parameter sets evaluated under different therapy protocols
4. **Simulation Pooling vs Test Statistics Separation**:
   - Full simulations are SHARED (expensive to compute)
   - Test statistics are SEPARATE (cheap to derive, purpose-specific)

## File Organization

```
cache/simulations/
└── {model_version}_{config_hash[:8]}/
    ├── batch_{timestamp}_{scenario}_1000sims_seed42.mat
    ├── batch_{timestamp}_{scenario}_500sims_seed43.mat
    └── ...
```

## Multi-Scenario Support

Each scenario (e.g., different therapy protocols) maintains independent caches while sharing the same parameter space. This enables joint parameter inference across scenarios.

**Example scenarios:**
- `control`: No treatment
- `treatment_a`: Drug A monotherapy
- `treatment_b`: Drug B monotherapy
- `combination`: Drug A + B combination therapy

## Hash Computation

### Included in Hash
- Units, compartment, model_context, tags, canonical_scale

### Excluded from Hash
- name, description, value, created_at, created_by

**Rationale:** Semantic changes trigger re-extraction; cosmetic changes don't.

## Workflow Example

```python
from qsp_hpc import QSPSimulator

# 1. Create simulator (reads config, sets up caching)
sim = QSPSimulator(
    priors_csv='priors.csv',
    submodel_priors_yaml='submodel_priors.yaml',  # optional: fitted marginals + copula
    calibration_targets='calibration_targets/control/',
    model_structure_file='model_structure.json',
    model_script='my_model',
    model_version='v1',
    scenario='control',
)

# 2. Request simulations (automatic caching)
params, obs = sim(n_simulations=1000)
# Checks: local cache → HPC test stats → HPC full sims → run new

# 3. Results cached for future use
```
