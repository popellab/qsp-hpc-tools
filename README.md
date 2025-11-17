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

# Run interactive setup wizard
qsp-hpc setup
```

After installation, run `qsp-hpc setup` to configure your HPC connection details. This creates `~/.config/qsp-hpc/credentials.yaml` with SSH, SLURM, and MATLAB settings.

### Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management:

```bash
# Clone the repository
git clone https://github.com/jeliason/qsp-hpc-tools.git
cd qsp-hpc-tools

# Create virtual environment and install dependencies with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

Or with pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick Start

### 1. Configure HPC Connection

Run the interactive setup wizard:

```bash
qsp-hpc setup
```

This will prompt you for:
- SSH connection details (host, username, key)
- SLURM configuration (partition, time limits)
- HPC paths (base directory, venv, simulation pool)
- MATLAB module name

The wizard tests your connection and saves configuration to `~/.config/qsp-hpc/credentials.yaml`.

### 2. Verify Connection

```bash
# Test HPC connection
qsp-hpc test

# View configuration
qsp-hpc info

# View job logs (after submitting jobs)
qsp-hpc logs --job-id 12345
```

### 3. Use in Your Code

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

# Initialize job manager (reads from ~/.config/qsp-hpc/credentials.yaml)
job_manager = HPCJobManager()

# Submit batch jobs
job_info = job_manager.submit_jobs(
    samples_csv='parameters.csv',
    test_stats_csv='test_stats.csv',
    model_script='my_model',
    num_simulations=1000,
    project_name='my_project'  # Creates remote dir: base_dir/my_project
)

# Monitor and collect results
results = job_manager.collect_results(job_info.state_file)
```

## CLI Reference

### qsp-hpc setup
Interactive setup wizard for HPC credentials. Tests SSH connection and SLURM access.

```bash
qsp-hpc setup
```

### qsp-hpc test
Test HPC connection, verify SLURM and MATLAB availability.

```bash
qsp-hpc test
qsp-hpc test --timeout 15  # Custom timeout
```

### qsp-hpc info
Display current configuration.

```bash
qsp-hpc info
qsp-hpc info --show-secrets  # Show SSH key path
```

### qsp-hpc logs
View HPC job logs.

```bash
qsp-hpc logs --job-id 12345           # View specific job
qsp-hpc logs --job-id 12345 --task-id 3  # View array task 3
qsp-hpc logs --job-id 12345 --lines 100  # Show 100 lines
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

### Using SSH Config (Recommended)

The easiest way to configure HPC access is with an SSH config file:

**1. Create `~/.ssh/config`:**
```ssh
Host hpc
    HostName hpc-cluster.edu
    User your_username
    IdentityFile ~/.ssh/id_rsa
    ServerAliveInterval 60
```

**2. Test it works:**
```bash
ssh hpc echo "OK"
```

**3. Run qsp-hpc setup:**
```bash
qsp-hpc setup
# Wizard will:
# - Detect your SSH config hosts
# - Test connection
# - Check/create remote directories
# - Optionally set up Python venv on HPC
# - Test SLURM and MATLAB
```

The setup wizard handles everything - SSH config detection, directory creation, and Python environment setup.

### Manual Configuration

Configuration is stored in `~/.config/qsp-hpc/credentials.yaml` and shared across all projects.

Run `qsp-hpc setup` to create the configuration interactively, or create it manually:

```yaml
ssh:
  host: "hpc"              # Can use SSH config alias
  user: ""                 # Empty = uses SSH config
  key: ""                  # Empty = uses SSH config

  # OR specify directly:
  # host: "hpc-cluster.edu"
  # user: "username"
  # key: "~/.ssh/id_rsa"

cluster:
  matlab_module: "matlab/R2024a"

paths:
  remote_base_dir: "/home/username/qsp-projects"    # Base for all projects
  hpc_venv_path: "/home/username/.venv/hpc-qsp"
  simulation_pool_path: "/scratch/username/simulations"

slurm:
  partition: "normal"
  time_limit: "04:00:00"
  mem_per_cpu: "4G"
```

Project-specific settings (like project name) are passed as method parameters:

```python
manager.submit_jobs(
    project_name='pdac_2025',  # Creates /home/username/qsp-projects/pdac_2025
    ...
)
```

## Requirements

- Python ≥3.9
- SSH access to SLURM HPC cluster
- MATLAB installed on HPC cluster
- QSP model implemented in MATLAB

## Development

### Running Tests

```bash
# Run all tests (skips HPC integration tests)
pytest

# Run with coverage
pytest --cov=qsp_hpc --cov-report=term-missing

# Run specific test file
pytest tests/test_batch_utils.py

# Run HPC integration tests (requires credentials)
pytest -m hpc -v

# Explicitly skip HPC tests
pytest -m "not hpc" -v
```

**HPC Integration Tests**: Tests marked with `@pytest.mark.hpc` connect to a real HPC cluster to validate SSH/SLURM integration. These are skipped by default. See `tests/README_TESTING.md` for setup instructions.

### Code Quality

```bash
# Format code with black
black qsp_hpc/ tests/

# Lint with ruff
ruff check qsp_hpc/ tests/

# Type checking (if mypy is added)
# mypy qsp_hpc/
```

### Project Structure

```
qsp-hpc-tools/
├── qsp_hpc/                    # Main package
│   ├── simulation/             # Simulation pool and QSP simulator
│   ├── batch/                  # HPC job management
│   └── utils/                  # Hash utilities
├── matlab/                     # MATLAB HPC workers
├── scripts/                    # Setup and utility scripts
│   ├── setup_credentials.py   # Interactive credential setup
│   └── hpc/                   # HPC environment setup
├── tests/                      # Test suite
├── docs/                       # Documentation
└── pyproject.toml             # Package configuration
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/my-feature`
2. Write tests first (TDD approach)
3. Implement feature
4. Ensure tests pass: `pytest`
5. Format code: `black qsp_hpc/ tests/`
6. Submit pull request

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
