# CLI Reference

The `qsp-hpc` command-line interface provides tools for configuring, testing, and monitoring HPC simulations.

## Commands

### `qsp-hpc setup`

Interactive setup wizard for HPC credentials. Tests SSH connection and SLURM access.

```bash
qsp-hpc setup
```

**What it does:**
- Detects SSH config hosts automatically
- Tests connection in real-time
- Checks/creates remote directories
- Sets up Python venv on HPC (optional)
- Tests SLURM and MATLAB availability
- Saves configuration to `~/.config/qsp-hpc/credentials.yaml`

**The wizard does not configure the C++ backend.** When using
`CppSimulator` or `submit_cpp_jobs`, edit
`~/.config/qsp-hpc/credentials.yaml` after `qsp-hpc setup` to add the
`cpp:` block — see [Configuration Guide](CONFIGURATION.md#c-backend-fields).

### `qsp-hpc test`

Test HPC connection, verify SLURM and MATLAB availability.

```bash
qsp-hpc test
qsp-hpc test --timeout 15  # Custom timeout
```

### `qsp-hpc info`

Display current configuration.

```bash
qsp-hpc info
qsp-hpc info --show-secrets  # Show SSH key path
```

### `qsp-hpc logs`

View HPC job logs.

```bash
qsp-hpc logs --job-id 12345           # View specific job
qsp-hpc logs --job-id 12345 --task-id 3  # View array task 3
qsp-hpc logs --job-id 12345 --lines 100  # Show 100 lines
```

## Configuration File

Configuration is stored in `~/.config/qsp-hpc/credentials.yaml`. See [Configuration Guide](CONFIGURATION.md) for details.
