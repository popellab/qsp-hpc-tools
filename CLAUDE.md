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
   - Status convention: `0`=success, `1`=failure (matches C++ backend; was
     formerly tri-state `{1, 0, -1}` but standardized 2026-04-16 — see
     git log for the cross-cutting change).

5. **C++ backend** (`qsp_hpc/cpp/` + `qsp_hpc/simulation/cpp_simulator.py`)
   - `CppSimulator`: drop-in for the simulation step; `run_hpc(n)` mirrors
     `QSPSimulator.__call__`'s 3-tier walk against C++ pools. Constructor
     takes `binary_path`, `template_xml`, optional `scenario_yaml` /
     `drug_metadata_yaml` / `healthy_state_yaml`, and either
     `calibration_targets` (YAML dir) or `test_stats_csv`.
   - `CppRunner` / `CppBatchRunner`: invoke the `qsp_sim` binary, parse
     the v2 raw-binary trajectory format (56-byte header + species +
     compartments + assignment-rules columns), write MATLAB-compatible
     Parquet. Every model parameter is broadcast as a `param:*` column
     (sampled values from theta_matrix; non-sampled defaults from the
     template) so calibration-target functions can read any parameter
     via `species_dict[name]`.
   - `HPCJobManager.submit_cpp_jobs(derive_test_stats=True, ...)`: chains
     a derivation job with `--dependency=afterok:<array_id>` so test
     stats land on HPC without raw-trajectory download.
   - Evolve cache (`qsp_hpc/cpp/evolve_cache.py` + `evolve_pack.py` +
     `qsth.py`, issue #90 Phase 1): `evolve_to_diagnosis` (the ~857-day
     burn-in, ~84%+ of per-sim cost) is cached per theta in a
     persistent, NFS-safe store — `{simulation_pool_path}/evolve_cache/
     {namespace}/shard_*.qsep`. `namespace = hash(binary, healthy_state)`;
     entry key is the rendered-param-XML hash. Write-through,
     first-writer-wins (append-only QSEP shards, never a shared mutable
     store — LMDB deadlocked on NFS, #86). Reuse spans scenarios and
     runs. `EvolveCache.compact()` folds shard scans into `manifest.json`.
   - Scenario-fused multi-scenario task (issue #90 Phase 2): an
     N-scenario joint run submits **one** SLURM array, not N. Each task,
     per theta, resolves the evolve once and runs every scenario from it
     via `--initial-state` — paying the ~90s fixed per-task overhead 1×.
     `CppBatchRunner.run_fused(scenarios=[FusedScenarioSpec, ...])`,
     `cpp_batch_worker.run_fused_chunk` (dispatched by a `scenarios` key
     in the job config), `HPCJobManager.submit_cpp_fused_jobs`,
     `MultiScenarioRunner.run_all` (`_plan_fused` fuses only scenarios
     still needing sims, each at its own deficit `start_index`). Fusion
     amortizes the evolve across scenarios in-run; the Phase 1 cache
     amortizes it across runs — they compose. Single-scenario path
     (`submit_cpp_jobs` / `CppSimulator.run_hpc`) untouched.
   - Fused local posterior-predictive (issue #90 Phase 3):
     `MultiScenarioRunner.simulate_with_parameters_all(theta)` — the
     local, posterior-predictive twin of `run_all`. One fused
     `CppBatchRunner.run_fused` batch evaluates a user-supplied theta
     matrix under every scenario on-host, evolving each theta once, not
     once per scenario. `CppSimulator`'s PPC front half is extracted into
     `_resolve_ppc_context` / `_ppc_cache_hit` / `_finalize_ppc` (→
     `PpcContext`), shared with single-scenario `simulate_with_parameters`
     so a fused run primes the cache a later single-scenario call reads.
     `MultiScenarioRunner.job_manager` is now optional at construction
     (only `run_all` needs it).
   - Evolve-cache hygiene (issue #90 Phase 4): `EvolveCache.maybe_compact()`
     rebuilds `manifest.json` only once enough shards accumulate outside it
     (default 64), called by `CppBatchRunner.run` / `run_fused` after the
     write-through — bounds the per-task footer scan, self-limiting, no
     compaction storm. `HPCJobManager.prune_simulation_pools` gained a
     second age-based pass over `evolve_cache/{namespace}/` subdirs (a
     binary rebuild orphans old namespaces); shared find+age+rm logic
     factored into `_prune_stale_subdirs`, `keep_evolve_namespaces` spares
     the live namespace.
   - Full plan + history in `docs/CPP_SIMULATION_PLAN.md`.

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

## Backend Status

The C++ backend (`qsp_sim`) is the primary path for new work. As of
2026-04, milestones M1–M7 and M10 are done: binary build, param-XML
renderer, raw-binary trajectory parser, Parquet parity with MATLAB
output, on-cluster test-stat derivation via chained SLURM dependency
(M9), dosing + `evolve_to_diagnosis` in `qsp_sim` (M10). `CppSimulator.run_hpc()`
walks the 3-tier cache against C++ pools. MATLAB retirement (M8) is
the queued follow-up. KLU/analytical-Jacobian work is infrastructure-
landed but deferred (OFF by default) due to 1/x rate-law singularities
at boundary states.

See `docs/CPP_SIMULATION_PLAN.md` for milestone history and performance
numbers.

**Codegen vs build split**: SBML → C++ source generation lives in the
standalone `qsp-codegen` package (`~/Projects/qsp-codegen`). The
`qsp_sim` binary itself is built in a consumer repo (SPQSP_PDAC,
pdac-build, etc.) that owns CMake + third-party deps. `cpp.repo_path`
in credentials points at that consumer repo, not at qsp-codegen.

## Logging

`qsp_hpc.utils.logging_config` provides hierarchical loggers per scenario
(e.g. `QSPSimulator.baseline_no_treatment`, `SimulationPool.v1`) plus
section/operation context managers. Log messages trace the 3-tier cache
progression and include timing for downloads and derivations.

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
  matlab_module: "matlab/R2024a"          # only needed for MATLAB backend

paths:
  remote_base_dir: "/home/username/qsp-projects"    # Base for all projects
  hpc_venv_path: "/home/username/.venv/hpc-qsp"
  simulation_pool_path: "/scratch/username/simulations"

slurm:
  partition: "normal"
  time_limit: "04:00:00"
  mem_per_cpu: "4G"

cpp:                                       # C++ backend
  repo_path: "/home/username/SPQSP_PDAC"   # required when using CppSimulator
  binary_path: "/home/username/SPQSP_PDAC/PDAC/qsp/sim/build/qsp_sim"
  template_path: "/home/username/SPQSP_PDAC/PDAC/sim/resource/param_all.xml"
  branch: "cpp-sweep-binary-io"
  subtree: "QSP"
  runtime_modules: ""                      # optional: module load lines for runtime
  build_modules: ""                        # optional: module load lines for build
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

1. **Backend**: MATLAB/SimBiology (legacy) or compiled C++ `qsp_sim` (preferred).
   The C++ path does not need MATLAB on HPC.
2. **SLURM Only**: Currently only supports SLURM clusters
3. **SSH Access Required**: No direct SLURM API integration yet
4. **Parquet Format**: Uses Parquet for efficient storage (requires Python on HPC)

## Useful References

- **C++ Backend Plan**: `docs/CPP_SIMULATION_PLAN.md`
- **Architecture**: `docs/ARCHITECTURE.md`
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
