# QSP HPC Tools

A Python package for running quantitative systems pharmacology (QSP) simulations on HPC clusters with intelligent caching and pooling.

## Features

- **Simulation Pool Management**: Efficient local caching of simulation results with scenario support
- **HPC Integration**: Submit and monitor MATLAB QSP simulations on SLURM clusters via SSH
- **Multi-Scenario Support**: Run the same parameters under different therapy protocols independently
- **Intelligent Caching**: Three-tier caching strategy (local pool → HPC test statistics → HPC full simulations)
- **Flexible Configuration**: Content-based hashing for automatic cache invalidation

## Installation

```bash
pip install qsp-hpc-tools
```

For development:
```bash
git clone https://github.com/jeliason/qsp-hpc-tools.git
cd qsp-hpc-tools
pip install -e ".[dev]"
```

## Quick Start

### Basic Simulation

```python
from qsp_hpc import QSPSimulator

# Create simulator for a specific scenario
simulator = QSPSimulator(
    test_stats_csv='path/to/test_stats.csv',
    priors_csv='path/to/priors.csv',
    model_script='my_qsp_model',
    model_version='v1',
    scenario='control',
    cache_dir='cache/simulations'
)

# Run simulations (automatically cached)
params, observables = simulator(n_simulations=1000)
```

### Multi-Scenario Workflow

```python
from qsp_hpc import SimulationPoolManager

# Initialize pool manager
pool = SimulationPoolManager(
    cache_dir='cache/simulations',
    model_version='baseline_v1',
    model_description='QSP model with 8 params, 12 observables',
    priors_csv='priors.csv',
    test_stats_csv='test_stats.csv',
    model_script='my_model'
)

# Check available scenarios
scenarios = pool.list_scenarios()
print(f"Available scenarios: {scenarios}")

# Load simulations for multiple scenarios
data = pool.load_multi_scenario(
    scenarios=['control', 'treatment_a', 'treatment_b'],
    n_requested=500
)

for scenario, (params, obs) in data.items():
    print(f"{scenario}: {params.shape}, {obs.shape}")
```

### HPC Job Management

```python
from qsp_hpc.batch import HPCJobManager

# Initialize job manager (reads from batch_credentials.yaml)
job_manager = HPCJobManager()

# Submit batch jobs
job_info = job_manager.submit_jobs(
    samples_csv='parameters.csv',
    test_stats_csv='test_stats.csv',
    model_script='my_model',
    num_simulations=1000,
    project_name='my_project'
)

# Monitor and collect results
results = job_manager.collect_results(job_info.state_file)
```

## Architecture

### Simulation Pooling

Simulations are cached locally using a manifest-free design where all metadata is encoded in filenames:

```
cache/simulations/
└── {model_version}_{config_hash[:8]}/
    ├── batch_{timestamp}_{scenario}_1000sims_seed42.mat
    ├── batch_{timestamp}_{scenario}_500sims_seed43.mat
    └── ...
```

### Caching Strategy

1. **Local Pool** (SimulationPoolManager): Test statistics cached locally in .mat files
2. **HPC Test Statistics**: Derived statistics stored on HPC for quick downloads
3. **HPC Full Simulations**: Complete timecourse data for deriving new test statistics
4. **On-Demand Execution**: Run new simulations only when needed

### Multi-Scenario Support

Each scenario (e.g., different therapy protocols) maintains independent caches while sharing the same parameter space. This enables joint parameter inference across scenarios.

## Configuration

Create a `batch_credentials.yaml` in your project root:

```yaml
ssh:
  host: "hpc-cluster"
  user: "username"
  key: "~/.ssh/id_rsa"

slurm:
  partition: "normal"
  time: "04:00:00"
  mem_per_cpu: "4G"

paths:
  remote_project_path: "/home/username/projects"
  simulation_pool_path: "/scratch/username/simulations"
```

## Requirements

- Python ≥3.9
- SSH access to SLURM HPC cluster
- MATLAB installed on HPC cluster
- QSP model implemented in MATLAB

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black qsp_hpc/
ruff check qsp_hpc/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{qsp_hpc_tools,
  author = {Eliason, Joel},
  title = {QSP HPC Tools},
  year = {2025},
  url = {https://github.com/jeliason/qsp-hpc-tools}
}
```

## Related Packages

- **qsp-sbi**: SBI/inference tools for QSP models (coming soon)
- **qspio-pdac**: PDAC-specific QSP model using these tools
