# Configuration Guide

## Overview

QSP HPC Tools uses a single configuration file shared across all projects: `~/.config/qsp-hpc/credentials.yaml`

## Quick Setup with SSH Config (Recommended)

The easiest way to configure HPC access is with an SSH config file.

### 1. Create `~/.ssh/config`

```ssh
Host hpc
    HostName hpc-cluster.edu
    User your_username
    IdentityFile ~/.ssh/id_rsa
    ServerAliveInterval 60
```

### 2. Test SSH connection

```bash
ssh hpc echo "OK"
```

### 3. Run setup wizard

```bash
qsp-hpc setup
```

The wizard will:
- Detect your SSH config hosts
- Test connection
- Check/create remote directories
- Optionally set up Python venv on HPC
- Test SLURM and MATLAB

## Manual Configuration

Create `~/.config/qsp-hpc/credentials.yaml` manually:

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
  matlab_module: "matlab/R2024a"          # only needed for MATLAB backend

paths:
  remote_base_dir: "/home/username/qsp-projects"    # Base for all projects
  hpc_venv_path: "/home/username/.venv/hpc-qsp"
  simulation_pool_path: "/scratch/username/simulations"

slurm:
  partition: "normal"
  time_limit: "04:00:00"
  mem_per_cpu: "4G"

# Required when using the C++ backend (CppSimulator / submit_cpp_jobs)
cpp:
  repo_path: "/home/username/pdac-build"
  binary_path: "/home/username/pdac-build/cpp/sim/build/qsp_sim"
  template_path: "/home/username/pdac-build/cpp/sim/resource/param_all.xml"
  branch: "main"
  subtree: "QSP"
  sim_subdir: "cpp/sim"                   # relative to repo_path; default shown
  runtime_modules: ""                     # optional: module load lines at runtime
  build_modules: ""                       # optional: module load lines at build time
  codegen_source: "git+ssh://git@github.com/popellab/qsp-codegen.git"
```

### C++ backend fields

- `cpp.repo_path` — required. Root of the consumer repo on the HPC (e.g.
  `pdac-build`, or a sibling `SPQSP_PDAC` checkout). Previously derived
  from `binary_path` by path-chopping; now must be explicit so layout
  changes don't silently break.
- `cpp.binary_path` — authoritative path to the built `qsp_sim` binary.
- `cpp.template_path` — path to `param_all.xml` used as the parameter
  template.
- `cpp.branch` — branch to track when `ensure_cpp_binary` auto-rebuilds
  because the binary is missing or stale.
- `cpp.sim_subdir` — path *relative to* `cpp.repo_path` where the
  `qsp_sim` `CMakeLists.txt` lives. Defaults to `cpp/sim` (matches
  pdac-build's in-repo layout). Set to whatever subdirectory holds the
  CMake project for non-default consumer-repo layouts.
- `cpp.subtree` — XML subtree label used when reading parameters out of
  `param_all.xml`. Defaults to `QSP`.
- `cpp.runtime_modules` — space-separated `module load` arguments run
  before `qsp_sim` invocations. Use this for the cluster's gcc / OpenMP
  runtime modules.
- `cpp.build_modules` — modules for the build phase (cmake, git, gcc).
  Falls back to `runtime_modules` when empty.
- `cpp.codegen_source` — pip-installable URL of the
  [`qsp-codegen`](https://github.com/popellab/qsp-codegen) package. The
  in-repo `cpp/sim` `CMakeLists.txt` invokes
  `python3 -m qsp_codegen.cmake` to locate `qsp_sim_core`, so
  `ensure_cpp_binary` pip-installs `qsp-codegen` into `hpc_venv_path`
  before running cmake and points cmake at that python via
  `-DPython3_EXECUTABLE`. Override for a fork or a feature branch. Set
  to an empty string only for unusual builds that vendor `qsp_sim_core`
  or supply `Python3_EXECUTABLE` another way.

### Calibration-target sources

`CppSimulator(calibration_targets=...)` accepts a single directory of
`SubmodelTarget` / `CalibrationTarget` YAMLs **or** a list of directories
that get merged in order. This lets a project keep cross-cutting targets
(e.g. clinical endpoints shared across scenarios) in a separate directory
from per-scenario calibration targets without having to symlink them
together.

### Pool-id hashing

`compute_pool_id_hash` folds the C++ binary content, scenario YAMLs, and
restriction-classifier kwargs into the pool ID, so editing any of those
invalidates the cache automatically. The `model_version` field has been
retired — anyone hand-computing pool IDs in scripts needs to switch to
the helper.

## Project-Specific Settings

Project-specific settings are passed as method parameters, not stored in the config file:

```python
from qsp_hpc.batch import HPCJobManager

manager = HPCJobManager()  # Reads ~/.config/qsp-hpc/credentials.yaml

manager.submit_jobs(
    ...
)
```

## Verification

After configuration, verify everything works:

```bash
qsp-hpc test   # Test connection and SLURM
qsp-hpc info   # View configuration
```
