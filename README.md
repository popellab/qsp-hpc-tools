# QSP HPC Tools

[![CI](https://github.com/popellab/qsp-hpc-tools/workflows/CI/badge.svg)](https://github.com/popellab/qsp-hpc-tools/actions)
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
- **Posterior-predictive sims**: `simulate_with_parameters(theta, backend="local"|"hpc")`
  on both backends — runs the simulator at user-supplied thetas (typically
  posterior draws from `qsp-inference`) with theta-hashed pool caching. See
  [docs/SIMULATE_WITH_PARAMETERS.md](docs/SIMULATE_WITH_PARAMETERS.md).
- **Burn-in trajectory dumps + LMDB-packed `evolve_cache`**: dump per-sim
  pre-diagnosis trajectories with `--evolve-trajectory-out` / `evolve_trajectory_dir`
  and reuse post-evolve ODE state across scenarios (~N× speedup on the dominant
  term). See [docs/EVOLVE_TRAJECTORIES.md](docs/EVOLVE_TRAJECTORIES.md).
- **Classifier-restricted theta pool**: `get_theta_pool(...,
  restriction_classifier_dir=...)` rejection-samples the prior against a
  `qsp_inference.inference.RestrictionClassifier` so simulator jobs aren't
  wasted on draws that always fail. Supports `lognormal` / `normal` /
  `uniform` / `beta` priors in the CSV.

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
    binary_path='/path/to/pdac-build/cpp/sim/build/qsp_sim',
    template_xml='/path/to/pdac-build/cpp/sim/resource/param_all.xml',
    scenario_yaml='scenarios/baseline_no_treatment.yaml',
    drug_metadata_yaml='/path/to/pdac-build/cpp/sim/resource/drug_metadata.yaml',
    healthy_state_yaml='/path/to/pdac-build/cpp/sim/resource/healthy_state.yaml',
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
- **[Configuration Guide](docs/CONFIGURATION.md)** - HPC setup, MATLAB and C++ backend fields
- **[CLI Reference](docs/CLI.md)** - Command-line interface documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design and caching strategy
- **[`simulate_with_parameters`](docs/SIMULATE_WITH_PARAMETERS.md)** - Posterior-predictive sims at user-supplied thetas, local vs HPC backend, cache key
- **[Evolve trajectories](docs/EVOLVE_TRAJECTORIES.md)** - Burn-in trajectory dumps, the LMDB-packed `evolve_cache`, and the long-form assemblers
- **[C++ Simulation Plan](docs/CPP_SIMULATION_PLAN.md)** - Design notes and milestone history for the C++ backend
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and testing

## Requirements

- Python ≥3.11
- SSH access to SLURM HPC cluster
- One of:
  - **C++ backend** (preferred): a built `qsp_sim` binary on the HPC (see
    [docs/CPP_SIMULATION_PLAN.md](docs/CPP_SIMULATION_PLAN.md))
  - **MATLAB backend**: MATLAB + SimBiology on the HPC, with a QSP model
    implemented in MATLAB

## Development

```bash
# Clone and install
git clone https://github.com/popellab/qsp-hpc-tools.git
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
  url = {https://github.com/popellab/qsp-hpc-tools}
}
```

## Related Packages

- **[qsp-inference](https://github.com/popellab/qsp-inference)**: Bayesian
  inference tools (submodel MCMC, NPE / SBI diagnostics, parameter audit) that
  consume `simulate_with_parameters` and `evolve_trajectory` outputs.
- **[qsp-codegen](https://github.com/popellab/qsp-codegen)**: SBML → C++ code
  generator pip-installed by `ensure_cpp_binary` to drive the in-repo
  `cpp/sim` CMake build.
