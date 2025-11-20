# QSP HPC Tools

[![CI](https://github.com/jeliason/qsp-hpc-tools/workflows/CI/badge.svg)](https://github.com/jeliason/qsp-hpc-tools/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for running quantitative systems pharmacology (QSP) simulations on HPC clusters with intelligent caching and pooling.

## Features

- **Simulation Pool Management**: Efficient local caching of simulation results with scenario support
- **HPC Integration**: Submit and monitor MATLAB QSP simulations on SLURM clusters via SSH
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

```python
from qsp_hpc import QSPSimulator

# Create simulator for a specific scenario
simulator = QSPSimulator(
    test_stats_csv='path/to/test_stats.csv',
    priors_csv='path/to/priors.csv',
    model_script='my_qsp_model',
    model_version='v1',
    scenario='control'
)

# Run simulations (automatically cached)
params, observables = simulator(n_simulations=1000)
```

### CLI Commands

```bash
qsp-hpc setup     # Interactive configuration wizard
qsp-hpc test      # Test HPC connection
qsp-hpc info      # Display current configuration
qsp-hpc logs      # View HPC job logs
```

## Documentation

- **[Configuration Guide](docs/CONFIGURATION.md)** - HPC setup and configuration
- **[CLI Reference](docs/CLI.md)** - Command-line interface documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design and caching strategy
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and testing

## Requirements

- Python ≥3.9
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
