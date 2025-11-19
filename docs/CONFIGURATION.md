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

## Project-Specific Settings

Project-specific settings are passed as method parameters, not stored in the config file:

```python
from qsp_hpc.batch import HPCJobManager

manager = HPCJobManager()  # Reads ~/.config/qsp-hpc/credentials.yaml

manager.submit_jobs(
    project_name='pdac_2025',  # Creates /home/username/qsp-projects/pdac_2025
    ...
)
```

## Verification

After configuration, verify everything works:

```bash
qsp-hpc test   # Test connection and SLURM
qsp-hpc info   # View configuration
```
