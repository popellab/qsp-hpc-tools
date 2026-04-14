# SimFunction Deployment Refactor

Design notes for compiling QSP simulations as standalone executables that run
against MATLAB Runtime instead of launching per-task MATLAB sessions. Written
after a two-day POC against the synthetic and real PDAC models.

## Why

Two independent pressures point the same direction:

1. **Per-task floor.** Each SLURM array task today pays (measured on `batch_worker.m`,
   `[timing-summary]` lines):

   | stage | cost | fixable by export/accelerate? |
   |---|---|---|
   | MATLAB startup | ~0.3s | — |
   | `parpool` open | ~36s | yes (skip parpool) |
   | `eval(model_script)` | ~6s | yes (bake into artifact) |
   | `sbioaccelerate` | ~42s | yes (move offline) |
   | per-sim loop (`copyobj`+`addvariant`+`sbiosimulate`) | ~10s/patient | yes (replace with SimFunction call) |

   The fixed-per-task cost is ~84s. Five patients per task means ~65% of wall
   is infrastructure.

2. **License concurrency.** Each `matlab -batch` consumes a MATLAB seat.
   Scaling SLURM-level parallelism hits the license cap before it hits the
   compute cap. Compiled exes against MATLAB Runtime consume zero seats, so
   fan-out becomes node-bound instead of license-bound.

## POC measurements

Build-once / load-many architecture, single offline MATLAB session produces an
accelerated SimFunction + mcc'd exe:

### Synthetic model (5-species exponential decay)
- Build total: 50s (accelerate 26s, mcc 13s)
- Compiled exe, 100 sims in one invocation: **32s wall**, of which ~25s is MCR
  initialization, 4s is all MATLAB-side work (load + 100 sims).
- Per-sim warm cost: **2.2 ms**.

### Real PDAC model (164 species, 401 params, 88 rules)
- Build total: 94s (model build 7s, accelerate 33s, mcc 13s)
- Compiled exe, 50 sims × 180 days: **33s wall**, 10s MATLAB-side.
- Per-sim warm cost: **152 ms** (vs ~10s in the current loop — **~66× faster**).

### Sequential-launch test (4× back-to-back same node)
- Wall per invocation: 26.4, 28.4, 26.2, 27.0s — **MCR init is not
  cold-cache-bound.** Every process launch pays the ~25s Runtime init.
- **Implication:** must batch many sims per exe invocation. One-patient-per-exe
  would be catastrophic (1000 patients × 25s = 7hr of init).

### Headline comparison (50 patients)

| path | wall | license seats | scaling ceiling |
|---|---|---|---|
| current (parpool + sbioaccelerate + sim loop) | ~584s | 1 | license cap |
| deployed exe + SimFunction | **~33s** | **0** | node cap |

~**18× faster, license-free, fan-out unlimited.**

## Architecture

```
offline one-shot build job (1 MATLAB seat, 1 Compiler seat, ~2-10 min)
├── construct PDAC model (existing immune_oncology_model_PDAC script)
├── createSimFunction(sf_evolve) for healthy→diagnosis
├── createSimFunction(sf_treat)  for diagnosis→treatment
├── accelerate(sf_evolve); accelerate(sf_treat)
├── save sf_*.mat
└── mcc -m pdac_main.m -a sf_*.mat -a <DependentFiles>

runtime SLURM array (0 license seats, ~28s floor per task)
└── ./pdac_main params.csv target_diameter out.mat
    ├── load sf_evolve, sf_treat
    └── for each patient:
        ├── res = sf_evolve(theta, 7300)
        ├── find diameter crossing in res.Data(:, V_T_idx)
        ├── reject if not reached or < 120 days
        └── sf_treat([theta, state_at_crossing], 180)
```

### Why two SimFunctions instead of one

`evolve_to_diagnosis.m` currently mutates the model object at runtime — applies
healthy ICs, simulates 20 years, writes trajectory endpoint back into
`model.Species.InitialAmount`, deactivates `initialAssignment` rules,
re-configures, simulates treatment. That pattern doesn't port:
`model.Species(i).InitialAmount = ...`, `sbiosimulate(model, variant)`, etc. are
non-deployable.

Two SimFunctions reframe the mutation as data flow: evolve outputs the full
species vector at diagnosis; treat takes rate params + species ICs as inputs.
No model mutation at runtime. The one-time `rule.Active = false` loop moves to
build time.

## Port checklist (evolve_to_diagnosis)

Deployment-surface probe results on the PDAC model
(`scripts/poc/probe_deployment_surface.m` in pdac-build):

| concern | resolution |
|---|---|
| **22 initialAssignment rules** | deactivate at build time inside `sf_treat`'s model — so the species ICs we pass as inputs stick |
| **66 repeatedAssignment rules** | leave active — these ARE the model's dynamics |
| **0 rate rules** | no handoff needed |
| **63 time-varying parameters** | all via repeatedAssignment → recomputed from species on step 1 of `sf_treat`. No cross-boundary plumbing needed. Lines 244-249 of evolve_to_diagnosis.m are effectively dead code. |
| **1 time-varying compartment (V_T)** | same as above — repeatedAssignment recomputes from species. Lines 252-257 also effectively dead. |
| **Compartment in SimFunction input/observable lists** | both work (confirmed) |
| **`set_healthy_populations`** | reads `D_cell` from model; D_cell is NOT in priors → deterministic → bake at build time |
| **`initial_tumour_diameter` (reject threshold)** | per-patient scalar → CLI arg to the exe, not a SimFunction input |

Net effect: the refactored handoff is **one species vector** (164 values) +
the per-patient `target_diameter` scalar. No parameters, no compartments
cross the boundary.

## Open items for real port

1. **MATLAB Runtime deployment path.** The POC used
   `LD_LIBRARY_PATH=$MATLAB_ROOT/runtime/glnxa64:...` pointing at the full
   MATLAB R2024a install. Confirm this is the recommended pattern at JHU
   (vs. installing a standalone Runtime bundle) before baking it into `run_poc.sbatch`-style templates.

2. **`immune_oncology_model_PDAC.m` is a script, not a function.** The POC
   works because MATLAB runs scripts inside function workspaces. For the real
   refactor, lifting it to a function `[model] = build_pdac_model()` is cleaner
   and removes the `startup.m` auto-run warning seen in the POC stderr.

3. **Observable list for `sf_treat`** is currently the union of all test-statistics'
   required species. That list is derivable from `test_stats.csv` at build
   time — but the current qsp-hpc-tools architecture decouples simulation
   from test-stats derivation (Python does derivation from Parquet). For the
   refactor, we need to either (a) compute a superset of required species at
   build time from the schema, or (b) observe every species and filter at
   derivation. (b) is simpler; cost is a bigger Parquet file.

4. **Dosing.** POC had none. `dose_schedule` in `batch_worker.m` is a struct
   array built from `schedule_dosing()`. SimFunction accepts `ScheduleDose` /
   `RepeatDose` objects (deployable per the docs). Straightforward but needs
   an adapter from the current YAML scenario format to the deployable dose
   objects.

5. **Build-artifact identity.** Every model change invalidates the `.mat`+exe.
   Need a content-hash-keyed artifact cache analogous to the existing config
   hash (`qsp_hpc.utils.hash_utils`), so builds don't rerun on every submit.

## Sequencing

1. (✅ done) POC both synthetic + real PDAC, confirm numbers and deployment surface.
2. Lift `immune_oncology_model_PDAC` from script → function (item 2 above).
3. Port `evolve_to_diagnosis` and `set_healthy_populations` via the two-SimFunction pattern; verify numerically against the current path on a handful of patients.
4. Integrate with `qsp-hpc-tools`: new batch-worker that `./pdac_main`s instead of `matlab -batch`, artifact cache, Runtime-path env setup.
5. Deprecate parpool / sbioaccelerate codepaths in `batch_worker.m`.

No unknown blockers after the POC.
