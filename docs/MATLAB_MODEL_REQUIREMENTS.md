# MATLAB Model Requirements

`qsp-hpc-tools` wraps existing MATLAB QSP models. This document describes what your MATLAB code must provide for the package to use it.

## Model script contract

When you pass `model_script="my_model"` to `QSPSimulator`, the HPC batch worker runs:

```matlab
eval('my_model');
```

After this line executes, a variable named **`model`** must exist in the workspace. This should be a SimBiology model object (created via `sbiomodel` or loaded from a `.sbproj` file).

### Minimal example

```matlab
% my_model.m — Minimal QSP model setup script
%
% This script must create a variable called `model` in the workspace.
% The batch_worker will then:
%   1. Clone the model for each virtual patient
%   2. Set parameter values from the sampled priors
%   3. Configure solver settings from the scenario YAML
%   4. Apply dosing (if defined in the scenario)
%   5. Run the simulation and extract species timecourses

model = sbiomodel('TumorImmune');

% Compartments
tumor = addcompartment(model, 'V_T', 1.0);   % Tumor compartment

% Species
addspecies(tumor, 'Tumor',   1e4);   % Tumor cell count
addspecies(tumor, 'CD8',     100);   % Cytotoxic T cells

% Parameters (names must match priors.csv)
addparameter(model, 'k_tumor_growth', 0.1);
addparameter(model, 'k_tumor_death',  0.05);
addparameter(model, 'k_immune_kill',  0.01);

% Reactions / rules
addreaction(model, 'null -> V_T.Tumor', ...
    'ReactionRate', 'k_tumor_growth * V_T.Tumor');
addreaction(model, 'V_T.Tumor -> null', ...
    'ReactionRate', 'k_tumor_death * V_T.Tumor + k_immune_kill * V_T.CD8 * V_T.Tumor');

% A startup.m file in your project root should add this script to the path.
```

### What the batch worker does with `model`

1. **Clones** the model for each virtual patient (`copyobj(model)`)
2. **Sets parameters**: for each column in the parameter CSV, it finds the matching model parameter or compartment initial condition and sets its value
3. **Configures the solver** using `sim_config` from the scenario YAML (solver type, tolerances, stop time)
4. **Applies dosing** if the scenario YAML has a `dosing` section (calls `schedule_dosing`)
5. **Runs** `sbiosimulate` and collects species timecourses

## Parameter naming

Parameter names in `priors.csv` must match either:

- A **model parameter** name (e.g., `k_tumor_growth`)
- A **compartment.species** name for setting initial conditions (e.g., `V_T.Tumor`)

The batch worker iterates over the CSV columns and calls the appropriate SimBiology setter for each.

## Project structure on HPC

The batch worker expects a `startup.m` file in the working directory that adds your model to the MATLAB path. A typical HPC project layout:

```
my_project/              # remote_base_dir on HPC
├── startup.m            # Adds model code to MATLAB path
├── my_model.m           # Model setup script (creates `model`)
├── batch_jobs/          # Created automatically by qsp-hpc-tools
│   ├── input/
│   │   ├── job_config.json
│   │   └── params.csv
│   └── output/
│       ├── chunk_001_results.mat
│       └── chunk_001_status.csv
└── ...
```

`batch_jobs/` is managed entirely by the package — you should not need to create or modify it manually.

## Dosing

If your scenario YAML includes a `dosing` section, the batch worker builds dose schedules and passes them to `schedule_dosing`. Your model must define the dosed species/parameters that the dosing targets. See `examples/scenarios/treatment.yaml` for the YAML format.

## Solver settings

Solver configuration comes from the scenario YAML's `sim_config` section. If not provided, defaults are:

| Field | Default |
|-------|---------|
| `solver` | `sundials` |
| `start_time` | `0` |
| `stop_time` | `30` |
| `time_units` | `day` |
| `abs_tolerance` | `1e-9` |
| `rel_tolerance` | `1e-6` |

## Outputs

The batch worker saves:

- **`chunk_XXX_results.mat`**: simulation results and metadata
- **`chunk_XXX_status.csv`**: per-patient status (1 = success, 0 = failed initial conditions, -1 = simulation error)
- **Parquet files** (if `save_full_simulations` is enabled): full species timecourses for the simulation pool
