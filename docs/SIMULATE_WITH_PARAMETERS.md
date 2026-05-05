# `simulate_with_parameters`: posterior-predictive sims at explicit thetas

`QSPSimulator.simulate_with_parameters` and
`CppSimulator.simulate_with_parameters` run the simulator at a caller-supplied
parameter matrix instead of drawing from the prior theta pool. They are the
entry point for posterior-predictive checks, OBED retraining sweeps, and any
other workflow where the thetas come from outside the prior — typically from
an `sbi`-trained NPE posterior or another inference pipeline.

This guide covers the C++ path (`CppSimulator`); `QSPSimulator` exposes the
same shape minus the local/HPC backend split (always local) and the prediction-
target loader.

## API

```python
table_or_arr, theta_table = sim.simulate_with_parameters(
    theta,                              # (n, n_params) ndarray
    backend="local",                    # or "hpc"
    prediction_targets=None,            # local-only: extra YAMLs concatenated with cal targets
    pool_suffix="posterior_predictive", # cache-isolation label
    evolve_trajectory_dir=None,         # local-only: dump per-sim burn-in trajectories
    evolve_trajectory_dt_days=None,     # sampling interval for evolve dumps
)
```

`theta` columns must align with `sim.param_names` (the order from the priors
CSV used to build the simulator). Returns a tuple:

- `theta_table` — same shape as input. Failed rows stay NaN-filled rather than
  being dropped, so the caller decides how to filter.
- `table` — pyarrow Table with `sample_index` (int64), `status` (int64;
  0 = ok, nonzero = failed), one `param:<name>` column per parameter, and
  one `ts:<target_id>` column per derived test statistic.

`QSPSimulator.simulate_with_parameters` returns just the test-statistic
ndarray (no pyarrow Table) for backwards compatibility.

## Cache key

The suffix-pool directory name is derived from a SHA-256 of:

```
theta.tobytes() | calibration_targets_hash | prediction_targets_hash | "backend=<local|hpc>"
```

Identical inputs hit the cache with no row-by-row matching. Edits to a
calibration-target YAML invalidate automatically. Local and HPC results for
the same theta are kept in separate caches so a local smoke test can be run
against an HPC reference output.

`pool_suffix` is the human-readable label combined with the hash. Only change
it when two logically-distinct posterior-predictive runs need to stay
cache-isolated even when the thetas happen to collide (rare).

## Local vs HPC backend

`backend="local"` runs the C++ binary on this host via `CppBatchRunner`.
Suitable for ≤ a few thousand sims on a laptop or workstation. The local
path is the only one that supports `prediction_targets` and
`evolve_trajectory_dir` today.

`backend="hpc"` submits the theta matrix as a dedicated HPC pool named after
the suffix-pool, with chained test-statistic derivation on the cluster. The
result is downloaded and reshaped to the same pyarrow Table layout as the
local backend. Requires `job_manager` set at construction (see the
[Configuration Guide](CONFIGURATION.md)).

HPC mode currently raises `NotImplementedError` for:

- `prediction_targets` — the merged calibration + prediction CSV isn't yet
  shipped to the cluster. Run locally or split the call.
- `evolve_trajectory_dir` — per-sim trajectory binaries don't have an
  HPC-side collection path yet.

## Prediction targets

`prediction_targets` is an optional directory of `PredictionTarget` YAMLs
(`prediction_target_id` schema). When given, the prediction rows are
concatenated with the calibration rows before derivation, so the returned
table has extra `ts:<prediction_target_id>` columns alongside the cal-target
columns. Useful when the same posterior should evaluate both calibration
recovery (does the model fit the data we trained on?) and clinical
predictions (what does the model predict for an unseen scenario?) without
re-running sims.

## Burn-in trajectory dumps

`evolve_trajectory_dir` triggers a per-sim binary trajectory dump in addition
to the test-statistic computation. Each accepted sim writes a
`sim_<sample_index>.bin` file in the supplied directory; the format matches
`qsp_sim --evolve-trajectory-out` and is parseable by
`qsp_hpc.cpp.runner.read_binary_trajectory`. See
[`docs/EVOLVE_TRAJECTORIES.md`](EVOLVE_TRAJECTORIES.md) for the assembler.

When this is set, the suffix-pool cache is **bypassed**: the cached
test-statistic parquet has no trajectory data, so a hit can't surface the
binaries the caller is asking for. Force-fresh sims fire each call.

## Theta restriction

`simulate_with_parameters` evaluates whatever thetas the caller provides — it
does not filter against a `RestrictionClassifier`. Restriction is applied
upstream when *generating* the prior pool (`get_theta_pool(...,
restriction_classifier_dir=...)`). For posterior-predictive workflows the
posterior already concentrates on viable regions, so restriction is rarely
needed; if you want to filter anyway, score with
`RestrictionClassifier.accept_named` before passing the theta matrix in.

## Stage 2 wiring

This is the API that `qsp-inference`'s
[Stage 2 SBI guide](https://github.com/popellab/qsp-inference/blob/main/docs/stage2-sbi-guide.md)
calls into for posterior-predictive simulations. The
`examples/stage2_pipeline.py` script there imports `qsp_hpc.simulation.QSPSimulator`
or `qsp_hpc.simulation.CppSimulator` and uses the local backend by default;
swap to `backend="hpc"` for production runs.

## See also

- [`docs/EVOLVE_TRAJECTORIES.md`](EVOLVE_TRAJECTORIES.md) — burn-in trajectory
  dumping, the LMDB-backed `evolve_cache`, and the long-form assemblers.
- [`docs/CONFIGURATION.md`](CONFIGURATION.md) — `cpp:` block fields the local
  and HPC backends rely on.
- [`docs/CPP_SIMULATION_PLAN.md`](CPP_SIMULATION_PLAN.md) — design notes and
  milestone history for the C++ backend.
