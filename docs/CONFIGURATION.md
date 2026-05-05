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
  repo_path: "/home/username/SPQSP_PDAC"
  binary_path: "/home/username/SPQSP_PDAC/PDAC/qsp/sim/build/qsp_sim"
  template_path: "/home/username/SPQSP_PDAC/PDAC/sim/resource/param_all.xml"
  branch: "cpp-sweep-binary-io"
  subtree: "QSP"
  runtime_modules: ""                     # optional: module load lines at runtime
  build_modules: ""                       # optional: module load lines for build
```

### C++ backend fields

- `cpp.repo_path` — required. Root of the sibling `SPQSP_PDAC` checkout on
  the HPC. Previously derived from `binary_path` by path-chopping; now
  must be explicit so layout changes don't silently break.
- `cpp.binary_path` — authoritative path to the built `qsp_sim` binary.
- `cpp.template_path` — path to `param_all.xml` used as the parameter
  template.
- `cpp.branch` — SPQSP_PDAC branch to build from when the binary is
  missing or stale.

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
