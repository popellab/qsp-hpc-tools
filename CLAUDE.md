# Claude Code Context

This document provides context for AI assistants (particularly Claude Code) working on this repository.

## Project Overview

**qsp-hpc-tools** is a Python package for running Quantitative Systems Pharmacology (QSP) simulations on HPC clusters with intelligent caching and pooling. It bridges Python-based simulation-based inference (SBI) workflows with MATLAB-based QSP models running on SLURM clusters.

## Key Design Principles

1. **Content-based Hashing**: Configuration hashes ensure cache invalidation when semantic changes occur
2. **Manifest-free Design**: All metadata encoded in filenames for simplicity
3. **Multi-scenario Support**: Same parameter sets evaluated under different therapy protocols
4. **Three-tier Caching**: Local pool → HPC test stats → HPC full simulations → On-demand execution

## Architecture

### Core Components

1. **SimulationPoolManager** (`qsp_hpc/simulation/simulation_pool.py`)
   - Manages local simulation caching with scenario support
   - Uses filename-based metadata: `batch_{timestamp}_{scenario}_{n_sims}sims_seed{seed}.mat`
   - Pool directories: `{model_version}_{config_hash[:8]}/`

2. **QSPSimulator** (`qsp_hpc/simulation/qsp_simulator.py`)
   - Main interface for SBI workflows
   - Implements 3-tier caching strategy
   - Handles automatic HPC job submission when cache misses occur

3. **HPCJobManager** (`qsp_hpc/batch/hpc_job_manager.py`)
   - SSH/SLURM integration for job management
   - Hierarchical config loading (global + project-specific)
   - Handles codebase syncing, job submission, monitoring, result collection

4. **MATLAB Workers** (`matlab/`)
   - `batch_worker.m`: Main SLURM array job worker
   - `extract_all_species_arrays.m`: Extracts simulation timecourse data
   - `save_species_to_parquet.m`: MATLAB→Python bridge for Parquet writing

### File Organization (Recently Updated)

```
qsp-hpc-tools/
├── qsp_hpc/                    # Main Python package
│   ├── simulation/             # Pool manager, QSP simulator
│   ├── batch/                  # HPC job management
│   │   └── logs_show.py       # [MOVED HERE] Log viewing utility
│   └── utils/                  # Hash utilities
├── matlab/                     # MATLAB HPC workers
├── scripts/                    # [NEW] Setup and utility scripts
│   ├── setup_credentials.py   # [MOVED HERE] Credential setup
│   └── hpc/                   # [RENAMED from hpc-setup/]
│       ├── setup_hpc_venv.sh
│       └── README.md
├── tests/                      # [NEW] Test suite
│   ├── test_batch_utils.py
│   ├── test_hash_utils.py
│   └── test_simulation_pool.py
└── pyproject.toml
```

## Recent Changes (2025-01-14)

### CLI Addition ✨ NEW
- **Created `qsp_hpc/cli.py`**: Full-featured CLI with Click framework
- **Commands**:
  - `qsp-hpc setup` - Interactive setup wizard
    - Detects SSH config hosts automatically
    - Tests connection in real-time
    - Checks/creates remote directories
    - Sets up Python venv on HPC (optional)
    - Tests SLURM and MATLAB
  - `qsp-hpc test` - Validate HPC connection and SLURM access
  - `qsp-hpc info` - Display current configuration
  - `qsp-hpc logs` - View HPC job logs
- **Entry point**: Installed via `pip install -e .` → `qsp-hpc` command available

### Configuration Refactor
- **Global config**: `~/.config/qsp-hpc/credentials.yaml` (one config for all projects)
- **Removed**: Project-specific `batch_credentials.yaml` pattern
- **Simplified structure**: No nested `projects/{project_name}/` directories
- **Updated template**: `credentials.yaml.template` matches new structure

### File Reorganization
- Moved `logs_show.py` from root → `qsp_hpc/batch/` (functionality now in CLI)
- Moved `setup_credentials.py` from root → `scripts/` (functionality now in CLI)
- Renamed `hpc-setup/` → `scripts/hpc/`

### Testing Infrastructure
- Created comprehensive test suite:
  - `tests/test_batch_utils.py`: Pure function tests (batch splitting) - 15 tests
  - `tests/test_hash_utils.py`: Critical caching logic tests - 31 tests
  - `tests/test_simulation_pool.py`: Pool manager tests - 23 tests
  - `tests/test_hpc_job_manager.py`: HPC integration tests - 23 tests (ready to run)
- Added `pytest.ini` configuration with `@pytest.mark.hpc` marker
- Created shared fixtures in `tests/conftest.py`
- **Total: 92 tests** (69 passing unit tests, 23 HPC integration tests ready)

### Development Environment
- Set up uv-based development workflow
- Updated README with development instructions
- Added Click dependency for CLI

### Pool Structure Consistency ✨ NEW (2025-01-19)
- **Scenario included in config hash**: Local and HPC pools now use consistent directory structure
  - Local: `{model_version}_{config_hash[:8]}_{scenario}/`
  - HPC: `{model_version}_{priors_hash[:8]}_{scenario}/`
  - Config hash now includes: priors CSV, test stats CSV, model script, model version, **scenario**
  - Each scenario gets its own pool directory for clarity and consistency
  - Eliminates confusion between local and HPC pool structures

### Logging Improvements ✨ NEW (2025-01-19)
- **Enhanced `qsp_hpc/utils/logging_config.py`**: Added structured logging utilities
  - `separator()` - Visual section separators
  - `format_config()` - Format key-value pairs with indentation
  - `create_child_logger()` - Hierarchical logger names (e.g., `QSPSimulator.baseline_no_treatment`)
  - `log_section()` - Context manager for sections with separators
  - `log_operation()` - Context manager for operations with timing
  - `log_summary_section()` - Summary boxes with metrics
  - Updated default format to always show timestamps and logger names

- **QSPSimulator logging** (`qsp_hpc/simulation/qsp_simulator.py`)
  - Hierarchical loggers by scenario (e.g., `QSPSimulator.baseline_no_treatment`)
  - Initialization logs show: test stats path, priors path, model version, config hash, pool directory, seed
  - Simulation requests show: requested count, local availability, HPC checks
  - 3-tier cache checking with explicit status messages:
    - ✓ marks for cache hits
    - Clear progression through: local pool → HPC test stats → HPC full sims → new generation
  - Timing information for downloads and derivations
  - Summary of simulations returned

- **SimulationPoolManager logging** (`qsp_hpc/simulation/simulation_pool.py`)
  - Hierarchical loggers by model version
  - Pool creation/reuse messages
  - Batch addition details: filename, simulation count, total pool stats
  - Loading messages show scenario and sampling behavior

- **HPCJobManager logging** (`qsp_hpc/batch/hpc_job_manager.py`)
  - Job submission shows: project, simulation count, array tasks, sims per task, model, seed
  - Structured progression through: sync → upload → submit
  - ✓ marks for successful operations
  - Timing for codebase sync
  - Derivation job details: pool path, test stats hash

**Example Log Output**:
```
2025-01-19 10:30:15 - QSPSimulator.baseline_no_treatment - INFO - Initializing QSP simulator for scenario: baseline_no_treatment
2025-01-19 10:30:15 - QSPSimulator.baseline_no_treatment - INFO -   Test Stats Csv: scenarios/test_statistics/baseline_no_treatment.csv
2025-01-19 10:30:15 - QSPSimulator.baseline_no_treatment - INFO -   Priors Csv: priors.csv
2025-01-19 10:30:15 - QSPSimulator.baseline_no_treatment - INFO -   Model Version: test_v1
2025-01-19 10:30:15 - QSPSimulator.baseline_no_treatment - INFO -   Config Hash: a3f7b2c8...
2025-01-19 10:30:15 - QSPSimulator.baseline_no_treatment - INFO - Simulation request: 500 simulations (seed=42)
2025-01-19 10:30:15 - QSPSimulator.baseline_no_treatment - INFO - No local simulations - checking HPC
2025-01-19 10:30:15 - SimulationPool.test_v1 - INFO - Creating new pool: cache/sbi_simulations/test_v1_a3f7b2c8
```

## Testing Strategy

### Test Categories

1. **Unit Tests** (fast, no I/O)
   - `test_batch_utils.py`: Batch splitting calculations
   - `test_hash_utils.py`: Hash computation and normalization

2. **Integration Tests** (require temp files)
   - `test_simulation_pool.py`: Pool manager with file I/O

3. **Not Yet Implemented**
   - HPC job manager tests (requires SSH mocking)
   - End-to-end workflow tests
   - MATLAB worker tests

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_hash_utils.py

# With coverage
pytest --cov=qsp_hpc --cov-report=term-missing
```

## Important Patterns

### Hash Computation
- **Include in hash**: Units, compartment, model_context, tags, canonical_scale
- **Exclude from hash**: name, description, value, created_at, created_by
- Rationale: Semantic changes trigger re-extraction; cosmetic changes don't

### Model Context Normalization
- Sort derived_from_context alphabetically
- Extract only names from dict entries (ignore descriptions)
- Sort other_parameters and other_species
- Sort reactions by (reaction, rule, reaction_rate) tuple

### Filename Conventions
- Batch files: `batch_{timestamp}_{scenario}_{n_sims}sims_seed{seed}.mat`
- Pool directories: `{model_version}_{config_hash[:8]}/`
- Parsed with regex: `r'batch_(\d{8}_\d{6})_([^_]+)_(\d+)sims_seed(\d+)\.mat'`

## Configuration

### Global Config (One File for All Projects)
**Location**: `~/.config/qsp-hpc/credentials.yaml`

Created via:
```bash
qsp-hpc setup  # Interactive wizard
```

### Required Config Fields
```yaml
ssh:
  host: "hpc-cluster.edu"
  user: "username"
  key: "~/.ssh/id_rsa"

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

## Common Tasks

### First Time Setup
```bash
pip install qsp-hpc-tools
qsp-hpc setup        # Interactive wizard (auto-creates dirs, sets up Python venv)
qsp-hpc test         # Verify connection
qsp-hpc info         # View config
```

The setup wizard detects SSH config, creates directories, and can set up the HPC Python venv automatically.

### Using in Projects
```python
from qsp_hpc import HPCJobManager

manager = HPCJobManager()  # Reads ~/.config/qsp-hpc/credentials.yaml
manager.submit_jobs(...)
```

### Adding New Test Statistics
1. Update test_stats.csv in project
2. Hash will change automatically
3. New pool directory created
4. Old simulations invalidated (semantic change)

### Adding New Scenarios
1. Define scenario in project config
2. Submit jobs with `scenario='new_scenario'`
3. Results cached independently
4. Can load multi-scenario: `pool.load_multi_scenario(['old', 'new'])`

### Debugging HPC Jobs
```python
from qsp_hpc.batch.logs_show import show_logs

# Show logs for task 3
show_logs(array_task_id=3, project='my_project')
```

## Known Constraints

1. **MATLAB Dependency**: QSP models must be in MATLAB
2. **SLURM Only**: Currently only supports SLURM clusters
3. **SSH Access Required**: No direct SLURM API integration yet
4. **Parquet Format**: Uses Parquet for efficient storage (requires Python on HPC)

## Future Enhancements

- [ ] Add HPC job manager tests with SSH mocking
- [ ] Add mypy type checking
- [ ] Create CLI commands for common operations
- [ ] Support for non-SLURM schedulers (PBS, LSF)
- [ ] Direct SLURM API integration (no SSH)
- [ ] Web dashboard for monitoring jobs

## Useful References

- **SLURM Integration Guide**: `docs/SLURM_Integration_Guide.md`
- **HPC Setup**: `scripts/hpc/README.md`
- **Package Config**: `pyproject.toml`
- **Test Fixtures**: `tests/conftest.py`

## When Making Changes

1. **Always write tests first** (TDD approach)
2. **Run full test suite** before committing: `pytest`
3. **Format code**: `black qsp_hpc/ tests/`
4. **Check linting**: `ruff check qsp_hpc/ tests/`
5. **Update this file** if architecture changes

## Contact

Maintainer: Joel Eliason
Issues: https://github.com/jeliason/qsp-hpc-tools/issues
