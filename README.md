# QSP HPC Tools

[![CI](https://github.com/jeliason/qsp-hpc-tools/workflows/CI/badge.svg)](https://github.com/jeliason/qsp-hpc-tools/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for running quantitative systems pharmacology (QSP) simulations on HPC clusters with intelligent caching and pooling.

## Features

- **Two backends**: original MATLAB / SimBiology path and a faster C++ `qsp_sim`
  path (~25-87× speedup; closes the column-set gap with SimBiology so calibration
  targets work unchanged). See [docs/CPP_SIMULATION_PLAN.md](docs/CPP_SIMULATION_PLAN.md).
- **Simulation Pool Management**: Efficient local caching of simulation results with scenario support
- **HPC Integration**: Submit and monitor QSP simulations on SLURM clusters via SSH
- **Multi-Scenario Support**: Run the same parameters under different therapy protocols independently
- **Intelligent Caching**: Three-tier caching strategy (local pool → HPC test statistics → HPC full simulations)
- **Flexible Configuration**: Content-based hashing for automatic cache invalidation

## Quick Start

### Installation

```bash
pip install qsp-hpc-tools

# Run interactive setup wizard
qsp-hpc setup
```

### Basic Usage

**MATLAB backend** (`QSPSimulator`):

```python
from qsp_hpc import QSPSimulator

simulator = QSPSimulator(
    priors_csv='priors.csv',
    submodel_priors_yaml='submodel_priors.yaml',  # optional: narrows priors for calibrated params
    calibration_targets='calibration_targets/control/',
    model_structure_file='model_structure.json',
    model_script='my_qsp_model',
    model_version='v1',
    scenario='control',
)
params, observables = simulator(1000)
```

**C++ backend** (`CppSimulator`) — drop-in for the simulation step, runs
the same 3-tier cache walk via `run_hpc()` when given an `HPCJobManager`:

```python
from qsp_hpc.batch.hpc_job_manager import HPCJobManager
from qsp_hpc.simulation.cpp_simulator import CppSimulator

simulator = CppSimulator(
    priors_csv='priors.csv',
    binary_path='/path/to/SPQSP_PDAC/PDAC/qsp/sim/build/qsp_sim',
    template_xml='/path/to/SPQSP_PDAC/PDAC/sim/resource/param_all.xml',
    scenario_yaml='scenarios/baseline_no_treatment.yaml',
    drug_metadata_yaml='/path/to/SPQSP_PDAC/PDAC/sim/resource/drug_metadata.yaml',
    healthy_state_yaml='/path/to/SPQSP_PDAC/PDAC/sim/resource/healthy_state.yaml',
    calibration_targets='calibration_targets/baseline_no_treatment/',
    model_structure_file='model_structure.json',
    model_version='v1',
    scenario='baseline_no_treatment',
    job_manager=HPCJobManager(),
)
params, test_stats = simulator.run_hpc(1000)
```

### CLI Commands

```bash
qsp-hpc setup     # Interactive configuration wizard
qsp-hpc test      # Test HPC connection
qsp-hpc info      # Display current configuration
qsp-hpc logs      # View HPC job logs
```

## Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** - Full walkthrough from install to first run
- **[MATLAB Model Requirements](docs/MATLAB_MODEL_REQUIREMENTS.md)** - Model interface contract
- **[Configuration Guide](docs/CONFIGURATION.md)** - HPC setup and configuration
- **[CLI Reference](docs/CLI.md)** - Command-line interface documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design and caching strategy
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and testing

## Requirements

- Python ≥3.11
- SSH access to SLURM HPC cluster
- MATLAB installed on HPC cluster
- QSP model implemented in MATLAB

## Development

```bash
# Clone and install
git clone https://github.com/jeliason/qsp-hpc-tools.git
cd qsp-hpc-tools

# Install with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

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

- **qsp-sbi**: SBI/inference tools for QSP models
- **qspio-pdac**: PDAC-specific QSP model using these tools
