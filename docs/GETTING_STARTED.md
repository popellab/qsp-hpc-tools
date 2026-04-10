# Getting Started

This guide walks through the full path from installation to running your first QSP simulation campaign.

## Prerequisites

- Python 3.11+
- SSH access to a SLURM HPC cluster
- MATLAB installed on the cluster (with SimBiology)
- A MATLAB QSP model (see [MATLAB Model Requirements](MATLAB_MODEL_REQUIREMENTS.md))

## 1. Install

```bash
pip install qsp-hpc-tools
```

Or for development:

```bash
git clone https://github.com/jeliason/qsp-hpc-tools.git
cd qsp-hpc-tools
pip install -e ".[dev]"
```

## 2. Configure HPC connection

Run the interactive setup wizard:

```bash
qsp-hpc setup
```

This creates `~/.config/qsp-hpc/credentials.yaml` with your SSH, SLURM, and path settings. The wizard will:

- Detect your SSH config aliases
- Test the SSH connection in real time
- Check SLURM availability
- Optionally create remote directories and set up a Python virtual environment on the cluster

Verify the connection:

```bash
qsp-hpc test
```

## 3. Prepare your project

A typical project directory looks like this:

```
my_project/
├── priors.csv                       # Parameter prior distributions
├── calibration_targets/             # Observable definitions (YAML)
│   ├── control/
│   │   ├── tumor_volume_day60.yaml
│   │   └── immune_infiltrate.yaml
│   └── treatment/
│       ├── tumor_volume_day60.yaml
│       └── immune_infiltrate.yaml
├── scenarios/                       # Simulation configuration per scenario
│   ├── control.yaml
│   └── treatment.yaml
├── my_model.m                       # MATLAB model setup script
└── startup.m                        # MATLAB path setup
```

See `examples/` in this repository for sample data files.

### priors.csv

Defines the parameter prior distributions. Each row is one parameter:

```csv
name,distribution,dist_param1,dist_param2,units,description
k_tumor_growth,lognormal,-2.3,0.5,1/day,Tumor growth rate
k_tumor_death,lognormal,-3.0,0.4,1/day,Tumor natural death rate
k_immune_kill,lognormal,-4.0,0.8,1/day,Immune-mediated kill rate
```

- `name`: must match a parameter or species name in your MATLAB model
- `distribution`: currently supports `lognormal`
- `dist_param1`, `dist_param2`: distribution parameters (for lognormal: mean and std of the log)

### Calibration targets

Observables are defined as a directory of YAML files, one per calibration target. Each file specifies the observable function, required species, empirical data, and units. This format is compatible with MAPLE for LLM-assisted literature extraction.

```yaml
# calibration_targets/control/tumor_volume_day60.yaml
calibration_target_id: tumor_volume_day60

observable:
  species:
    - V_T.Tumor
  units: cell
  code: |
    def compute_observable(time, species_dict, constants, ureg):
        tumor = species_dict['V_T.Tumor']
        idx = np.argmin(np.abs(time.magnitude - 60))
        return float(tumor[idx]) * ureg.cell

empirical_data:
  median: [1.0e+06]
  ci95: [[5.0e+05, 2.0e+06]]
  sample_size: 50
```

See `examples/data/calibration_targets/` for more examples.

### Scenario YAMLs

Each scenario defines simulation settings and (optionally) a dosing protocol. Place these in `scenarios/` under your project root.

```yaml
# scenarios/control.yaml — no treatment
sim_config:
  start_time: 0
  stop_time: 60
  time_units: day
  solver: sundials
  abs_tolerance: 1.0e-9
  rel_tolerance: 1.0e-6
```

```yaml
# scenarios/treatment.yaml — with dosing
sim_config:
  start_time: 0
  stop_time: 60
  time_units: day
  solver: sundials
  abs_tolerance: 1.0e-9
  rel_tolerance: 1.0e-6

dosing:
  drugs: [GVAX, pembrolizumab]
  GVAX_dose: 1.0e8
  GVAX_start_time: 0
  GVAX_interval: 7
  GVAX_repeat_count: 4
  pembrolizumab_dose: 200
  pembrolizumab_start_time: 21
  pembrolizumab_interval: 21
  pembrolizumab_repeat_count: 3
```

If no scenario YAML is found, defaults are used (stop_time=30 days, sundials solver, no dosing).

### MATLAB model

Your model script must create a SimBiology `model` variable when evaluated. See [MATLAB Model Requirements](MATLAB_MODEL_REQUIREMENTS.md) for the full contract and a minimal example.

## 4. Run simulations

```python
from qsp_hpc import QSPSimulator

# Initialize for the control scenario
sim_control = QSPSimulator(
    priors_csv="priors.csv",
    calibration_targets="calibration_targets/control/",
    model_script="my_model",
    model_version="v1",
    scenario="control",
    project_root=".",
    seed=42,
)

# Request 1000 simulations
# Returns (parameters, observables) as numpy arrays
theta, x = sim_control(1000)

print(f"Parameters: {theta.shape}")   # (1000, n_params)
print(f"Observables: {x.shape}")      # (1000, n_observables)
```

The first call submits SLURM jobs and waits for results. Subsequent calls with the same configuration return cached results instantly.

### Multi-scenario runs

To evaluate the same virtual patients under different therapies, use `cache_sampling_seed`:

```python
shared_seed = 42

sim_control = QSPSimulator(
    priors_csv="priors.csv",
    calibration_targets="calibration_targets/control/",
    model_script="my_model",
    model_version="v1",
    scenario="control",
    project_root=".",
    seed=shared_seed,
    cache_sampling_seed=shared_seed,
)

sim_treatment = QSPSimulator(
    priors_csv="priors.csv",
    calibration_targets="calibration_targets/treatment/",
    model_script="my_model",
    model_version="v1",
    scenario="treatment",
    project_root=".",
    seed=shared_seed,
    cache_sampling_seed=shared_seed,
)

theta_ctrl, x_ctrl = sim_control(1000)
theta_treat, x_treat = sim_treatment(1000)

# theta_ctrl and theta_treat are identical parameter draws
```

### Local-only mode (no HPC)

For testing or small runs with a local MATLAB installation:

```python
sim = QSPSimulator(
    priors_csv="priors.csv",
    calibration_targets="calibration_targets/control/",
    model_script="my_model",
    model_version="v1",
    scenario="control",
    project_root=".",
    local_only=True,
    matlab_path="/usr/local/MATLAB/R2024a/bin/matlab",
)

theta, x = sim(50)  # Small batch, runs locally
```

## 5. Use the results

The returned arrays are standard numpy — use them however you need:

```python
import numpy as np

# Filter virtual patients by clinical criteria
valid = x[:, 0] < 1e7  # e.g., tumor volume below threshold
theta_valid, x_valid = theta[valid], x[valid]

# Feed into downstream analysis (Bayesian inference, sensitivity analysis, etc.)
# Example with SBI:
#   from sbi.inference import SNPE
#   inference = SNPE()
#   inference.append_simulations(torch.tensor(theta), torch.tensor(x))
#   posterior = inference.train()
```

## Next steps

- [Architecture](ARCHITECTURE.md) — how the three-tier cache works
- [Configuration Guide](CONFIGURATION.md) — advanced HPC settings
- [CLI Reference](CLI.md) — all command-line options
- [MATLAB Model Requirements](MATLAB_MODEL_REQUIREMENTS.md) — model interface details
- `examples/feature_walkthrough.ipynb` — interactive walkthrough of all features
