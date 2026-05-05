# Evolve trajectories: burn-in dumps, the LMDB cache, and assemblers

Most calibration scenarios start with an `evolve_to_diagnosis` burn-in: integrate
the model from a healthy steady state up to the diagnosis time, then start the
post-diagnosis scenario. Two pieces of infrastructure surround that burn-in:

- **`evolve_cache`** caches the post-evolve ODE state per `(model, config, theta)`
  so multiple downstream scenarios reuse the same burn-in instead of redoing it.
- **Burn-in trajectory dumps** optionally write the full pre-diagnosis trajectory
  to disk for posterior-predictive plotting.

This guide covers both, plus the long-form assemblers that pivot the resulting
binaries into pandas DataFrames suitable for plotting and downstream metric
evaluation.

## Burn-in trajectory dumps

`qsp_sim --evolve-trajectory-out <path>` writes the burn-in trajectory to a v2
binary file (same format as `--binary-out`, parseable by
`qsp_hpc.cpp.runner.read_binary_trajectory`). One file per sim.

From Python, route this through the runner / simulator instead of invoking
the binary directly:

```python
# Per-call, on a CppBatchRunner-backed simulator:
sim.simulate_with_parameters(
    theta,
    evolve_trajectory_dir="runs/foo/evolve",
    evolve_trajectory_dt_days=0.5,   # optional override
)

# Lower level — on CppRunner.run_one():
runner.run_one(
    ...,
    evolve_trajectory_out=Path("evolve.bin"),
    evolve_trajectory_dt_days=0.5,
)
```

The output directory ends up with `sim_<sample_index>.bin` files
(local backend only — HPC raises `NotImplementedError` until per-sim
trajectory binaries get an HPC-side collection path).

`evolve_trajectory_dt_days=None` lets the C++ side use the evolve spec's
`step_days`. Override when you want a denser sample for plotting or a
sparser one to keep the binaries small.

### Cache bypass

When `evolve_trajectory_dir` is set on `simulate_with_parameters`, the
suffix-pool **cache is bypassed**: the cached test-statistic parquet has no
trajectory data, so a hit can't surface the per-sim binaries the caller asked
for. Force-fresh sims fire on every call. This is the right behaviour but is
worth knowing — toggling `evolve_trajectory_dir` on for one run and off for
the next will re-run the sims even if test stats would otherwise hit cache.

## The `evolve_cache`

The `evolve_to_diagnosis` phase is ~95% of total wall time for many scenarios,
and identical evolve-affecting knobs across scenarios mean the post-evolve ODE
state is reusable. `qsp_hpc.cpp.evolve_cache` exposes a content-addressed
store of post-evolve states keyed on `(theta, evolve-affecting config)` — the
scenario YAML is intentionally **not** part of the key. N scenarios collapse
from `~N × T_evolve` to `~T_evolve + N × T_post`.

### Storage layout

Cache entries are packed into one LMDB environment per
`(model, config)` subdirectory. Previous versions used one file per entry,
which scaled poorly to millions of small reads on parallel filesystems.

The engagement / disable state is logged loudly at runner startup:

```
evolve_cache: ENABLED at /scratch/.../evolve_cache/<model_hash>/
evolve_cache: DISABLED (no evolve-to-diagnosis in scenario YAML)
```

If you don't see one of these at the head of a run, the cache code path
didn't get exercised. See `docs/CPP_SIMULATION_PLAN.md` (M13 milestone) for
the design of the cache-key derivation.

### When the cache is bypassed

The cache is intentionally bypassed when the caller sets
`evolve_trajectory_dir` or otherwise asks for burn-in trajectories: there is
no point caching a final state when the caller wants the full timecourse. Any
trajectory binary the caller asks for therefore reflects a fresh integration.

## Long-form assemblers

Once a directory of `sim_*.bin` files exists, two functions in
`qsp_hpc.cpp.evolve_trajectory` pivot them into long-form DataFrames keyed on
`(sample_index, time, column, value)`:

### `assemble_evolve_trajectory_long`

Reads the per-sim binary files written during burn-in. Time axis is
**time-to-diagnosis** (`t_to_diagnosis_days = t_model - t_diagnosis_days`):
0 = diagnosis sample, earlier rows are negative. Posterior-predictive draws
with variable evolve durations align cleanly on the right edge.

```python
import pandas as pd
from qsp_hpc.cpp.evolve_trajectory import assemble_evolve_trajectory_long

df = assemble_evolve_trajectory_long(
    traj_dir="runs/foo/evolve",
    species_names=[...],
    compartment_names=[...],
    rule_names=[...],
    columns=None,                    # subset; None keeps all
    sample_indices=None,             # subset; None reads everything
)
# df.columns == ["sample_index", "t_to_diagnosis_days", "column", "value"]
```

Column names come from the same `--species-out` / `--compartments-out` /
`--rules-out` files the post-scenario writer produces — the binary header
records counts but not names, so this metadata must be supplied.

### `assemble_post_scenario_trajectory_long`

Companion that reads the **post-diagnosis** trajectory out of a
`CppBatchRunner` species parquet. Time axis is `time_days`, non-negative,
starting at 0 = scenario start (which is diagnosis for evolve-to-diagnosis
scenarios). Same long-form shape so the two can be concatenated for
end-to-end plots.

```python
from qsp_hpc.cpp.evolve_trajectory import assemble_post_scenario_trajectory_long

df_post = assemble_post_scenario_trajectory_long(
    species_parquet="runs/foo/scenarios/baseline/species_0.parquet",
    columns=["Tumor_Volume", "CD8_count"],
    sample_indices=df["sample_index"].unique(),
    drop_failed=True,
)
# df_post.columns == ["sample_index", "time_days", "column", "value"]
```

Failed sims (`status != 0`) are silently dropped by default — the C++ batch
runner writes a row even for failures with NaN-padded list cells, so leaving
them in produces all-NaN trajectory bands.

### Pairing with `qsp-inference`

`qsp_inference.inference.evaluate_calibration_target_over_trajectory`
consumes the long-form output of these assemblers to evaluate calibration-
target observables across posterior-predictive draws. This is the path used
by Stage 2 trajectory plotting in `qsp-inference`'s
[Stage 2 SBI guide](https://github.com/popellab/qsp-inference/blob/main/docs/stage2-sbi-guide.md).

## See also

- [`docs/SIMULATE_WITH_PARAMETERS.md`](SIMULATE_WITH_PARAMETERS.md) —
  posterior-predictive entry point that drives the trajectory dumps.
- [`docs/CPP_SIMULATION_PLAN.md`](CPP_SIMULATION_PLAN.md) — full design notes
  for the C++ backend, including the M13 evolve-cache milestone.
