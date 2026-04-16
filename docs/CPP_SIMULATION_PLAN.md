# C++ Simulation Path — Implementation Plan (Option 1: subprocess + raw binary)

Replace the MATLAB-based simulation backend in `qsp-hpc-tools` with a Python
orchestration layer that calls the C++/CVODE driver living in
`SPQSP_PDAC/PDAC/sim/tests/ode_compile/dump_trajectories.cpp`. Keep both
paths coexisting until the C++ path is verified end-to-end; retire MATLAB in a
follow-up PR.

Numerical validity of the C++ core is already established (ran one param
set through `dump_trajectories` and compared against SimBiology — agrees).

## Design decisions (locked)

- **Invocation**: subprocess per sim. Binary built once in `SPQSP_PDAC`,
  shipped to HPC via the existing codebase-sync step.
- **I/O**: raw float64 binary per sim (tiny header + contiguous array), not CSV.
  CSV path stays in `dump_trajectories` for validation/debugging.
- **Params in**: sampled θ rendered into a per-sim XML by templating over a
  reference XML (from `SPQSP_PDAC/PDAC/sim/resource/param_all.xml`). Only the
  parameters named in the priors CSV are substituted; everything else inherits
  the template default.
- **Output aggregation**: Python driver concatenates per-sim raw-binary files
  into one Parquet per batch — same on-disk shape as today's MATLAB output so
  `SimulationPoolManager` / `QSPSimulator` caching works unchanged.
- **Python shim**: new module `qsp_hpc/simulation/cpp_simulator.py`
  paralleling `qsp_simulator.py`. Both coexist. Retire the MATLAB one once the
  C++ path is trusted.
- **Dosing / scenarios**: deferred — first milestone is single-scenario,
  no-dose. Event-machinery question is being investigated separately.
- **Performance optimizations out of scope for MVP**: pybind11, KLU sparse
  solver, analytical Jacobian, in-process thread pool. All live behind the
  same Python API so they're additive later.

## Milestones

### M1 — C++ raw-binary output flag (in SPQSP_PDAC)

*Est: 0.5 day.*

1. Add `--binary-out <path>` flag to `dump_trajectories.cpp`. When set, skip
   CSV and write a small header + `double[n_timepoints * n_species]`:
   ```
   magic:   uint32    (e.g. 0x51535042  "QSPB")
   version: uint32    (1)
   n_times: uint64
   n_sp:    uint64
   dt:      float64   (days)
   t_end:   float64   (days)
   data:    float64[n_times * n_sp]   (row-major, row = timepoint)
   ```
2. Optional: add `--species-out <path>` to dump the species-name list once per
   batch (from `ODE_system::getHeader()`) so Python doesn't have to reparse it
   per sim.
3. Keep CSV as default; binary is opt-in.
4. Sanity-check: `dump_trajectories --binary-out x.bin` on one param set;
   Python `np.fromfile` reads it; float-exact vs CSV path.

### M2 — Param-XML template + renderer (Python side)

*Est: 1-2 days.*

1. Copy `SPQSP_PDAC/PDAC/sim/resource/param_all.xml` into this repo under
   `qsp_hpc/cpp/resources/param_template.xml` (or reference it via config).
2. New module `qsp_hpc/cpp/param_xml.py`:
   - `load_template(path) -> ElementTree`
   - `render_params(template, theta: dict[str, float]) -> bytes` — walk the
     tree, substitute leaf text for every key in `theta`. Unknown keys raise.
   - Keep the XML structure byte-identical to the template outside the
     substituted leaves (`QSPParam::_readParameters` is path-driven, so leaf
     order doesn't matter but element paths do).
3. Build a parameter-name map from the priors CSV column names to XML paths.
   The path list is `QSPParam::_xml_paths[]` in
   `SPQSP_PDAC/PDAC/qsp/ode/QSPParam.cpp`. Emit the mapping once to a JSON
   file (`qsp_hpc/cpp/resources/param_paths.json`) rather than reparsing on
   every call.
4. Unit handling: `QSPParam` does unit conversion at load time
   (SimBiology custom units → SI). The priors CSV is already in the model's
   source units, same as the MATLAB path consumes. **Verify this** on one
   param — render XML, run sim, check trajectory matches the SimBiology run
   to machine tolerance.
5. Tests: round-trip template→render→parse matches input θ within float
   tolerance; unknown parameter names raise; missing parameter names inherit
   template defaults.

### M3 — Local single-sim runner

*Est: 1 day.*

1. `qsp_hpc/cpp/runner.py`:
   - `CppRunner(binary_path, template_xml)` — holds paths, validates binary
     exists and is executable.
   - `run_one(theta: dict, t_end: float, dt: float, workdir: Path) -> ndarray`
     — render XML, `subprocess.run` with a timeout, parse raw binary,
     return trajectory `ndarray[n_times, n_species]`. Returns species names
     too (cached from the `--species-out` dump).
   - On nonzero exit: include stderr in the raised exception; keep the
     offending XML on disk under `workdir/failed/` for debugging.
2. Tests: run a single known-good θ, assert shape and a couple of species
   final-values vs recorded reference.

### M4 — Batch runner (local parallelism)

*Est: 1 day.*

1. `CppBatchRunner.run(theta_matrix, param_names, seed, scenario, ...)`:
   - `ProcessPoolExecutor` over `run_one`, configurable `max_workers`.
   - Writes each trajectory to the pool's binary file; on batch completion,
     aggregates to **one Parquet** with the same column schema
     (`save_species_to_parquet.m` output) as the MATLAB pipeline produces.
   - Reuses `SimulationPoolManager` unchanged — the Parquet lands at
     `batch_{timestamp}_{scenario}_{n_sims}sims_seed{seed}.parquet` in the
     expected pool directory.
2. Parquet column layout: match today's MATLAB output exactly so downstream
   `QSPSimulator` / test-statistics derivation doesn't care which backend
   produced the file. Verify column names + ordering against an existing
   MATLAB Parquet.

### M5 — CppSimulator (top-level Python API)

*Est: 1 day.*

1. `qsp_hpc/simulation/cpp_simulator.py`:
   - `CppSimulator(priors_csv, calibration_targets, model_version, scenario,
     binary_path, template_xml)` — mirrors `QSPSimulator.__init__` surface.
   - `__call__(n_sims, seed)` — samples priors, delegates to
     `CppBatchRunner`, reuses `SimulationPoolManager` for caching.
   - Config-hash inputs include `binary_path.sha256()` + template path, so
     rebuilding the C++ core invalidates cache automatically. Keeps the
     invariant that a hash hit guarantees identical numerical output.
2. Duplicate only what's necessary — don't refactor `QSPSimulator` yet.
   Any shared logic (priors sampling, scenario handling) extracts later
   once both paths are stable.

### M6 — HPC integration

*Est: 2-3 days.*

1. Build the C++ binary on Rockfish. Add build steps to
   `scripts/hpc/setup_hpc_venv.sh` (or a sibling script) that clone/update
   `SPQSP_PDAC`, invoke the existing CMake target, drop the binary at a
   known path (e.g. `$HPC_VENV_PATH/../bin/dump_trajectories`).
2. New SLURM worker script replacing `matlab/batch_worker.m`:
   `qsp_hpc/batch/cpp_batch_worker.py`. Takes `(param_csv_chunk,
   scenario, seed, output_pool)` args; does the same work `CppBatchRunner`
   does locally, but as one SLURM array task per chunk of sims.
3. `SlurmJobSubmitter` / `HPCJobManager`: add a `backend='cpp'` switch that
   selects the Python worker and module-loads nothing (no MATLAB).
4. Codebase sync: the binary itself is large and architecture-specific — do
   *not* `rsync` it from laptop to HPC. Build once on HPC, treat the built
   binary as HPC infrastructure like the Python venv.

### M7 — Verification

*Est: 2-3 days.*

1. Pick a ~100-sim sweep with fixed seed. Run through:
   - MATLAB path (current main)
   - CppSimulator local
   - CppSimulator HPC
2. Load all three Parquets. Assert per-species trajectories agree to
   `rtol=1e-6`, `atol=1e-9` — if CVODE tolerances in the C++ driver match the
   SimBiology ones this should hold easily. If they don't agree, the failure
   is almost certainly a param-name mapping bug (M2, step 3) or a unit bug
   (M2, step 4), not a solver bug.
3. Add this as a marked test (`@pytest.mark.validation`) so it can be run on
   demand but isn't a default CI dep.

### M8 — MATLAB retirement (follow-up PR)

Only after M7 passes and at least one real SBI workflow has used the C++
path end-to-end.

- Delete `qsp_hpc/matlab/`.
- Remove MATLAB code paths from `batch_runner.py`, `qsp_simulator.py`,
  `hpc_job_manager.py`, `setup_hpc_venv.sh`.
- Update `CLAUDE.md`, `README.md`, examples. Drop the `matlab_module`
  config key.
- Keep `test_ode_vs_matlab.py` in `SPQSP_PDAC` — that's the validation
  anchor for the C++ numerics and should stay.

## Candidate next steps (post-M7, post-M10)

The C++ path is numerically validated, runs end-to-end on Rockfish at
100k scale, and now covers scenarios with `evolve_to_diagnosis` + dosing
(M10). Everything below is optional — ordered roughly by the leverage
each gives before we need to think about the MATLAB retirement in M8.
Pick what's useful next; they're not a strict sequence.

### M9 — On-cluster test-stat derivation for the C++ path

*Est: 1–2 days.*

Once sweep sizes exceed ~10k, downloading raw trajectories is the slow
part (100k ≈ 21 GB Parquet, 1M ≈ 210 GB). The MATLAB path already has
`qsp_hpc/batch/derive_test_stats_worker.py` running as a single SLURM
task that chews through a pool directory of Parquets and emits
`chunk_XXX_test_stats.csv` files. That worker is backend-agnostic
(reads Parquet + species unit metadata), so it should run against
C++-produced pools unchanged.

What this unlocks:

- `QSPSimulator` / SBI workflows can point at `simulation_pool_path` on
  HPC and download per-sim test statistics (small) instead of raw
  trajectories (huge).
- Existing 3-tier caching (local pool → HPC test stats → HPC full sims
  → new simulations) works end-to-end for the C++ backend.

Steps:

1. Verify `derive_test_stats_worker` reads the C++ Parquet schema
   without modification. It mostly depends on `status`, `time`,
   species-name columns, and (optionally) the `param:*` columns.
2. Update `HPCJobManager.submit_cpp_jobs()` to optionally chain a
   derivation job after the sim array (via `SLURMJobSubmitter.submit_derivation_job`
   with a `--dependency=afterok:<array_id>` flag).
3. Wire C++ pool paths into `QSPSimulator`'s `check_hpc_test_stats` /
   `check_hpc_full_sims` lookups. Config hash compatibility: local
   `SimulationPoolManager` uses a subset of what
   `CppSimulator._compute_config_hash` includes. For caching to work
   across backends the hash definition needs to agree.

### M10 — Dosing / init-function support in qsp_sim: DONE (2026-04-16)

M7 had bypassed `baseline_no_treatment.yaml` because its
`initialization_function: 'evolve_to_diagnosis'` wasn't implemented in
C++ yet, and all treatment scenarios use dosing. M10 closes that gap
end-to-end.

**Shipped (SPQSP_PDAC `cpp-sweep-binary-io`)**:

1. Cherry-picked the four PDAC_qsp_sync commits onto cpp-sweep-binary-io:
   SBML `<listOfEvents>` codegen (`1e85ffc8`), YAML-driven dosing +
   `drug_metadata.yaml` (`f3cb3e7a`), `get_compartment_volume()` codegen
   (`8a06aa27`), `evolve_to_diagnosis` port (`06dc4f75`). The dumper +
   dosing code was re-targeted from the old
   `PDAC/sim/tests/ode_compile/dump_trajectories.cpp` to the
   post-promotion `PDAC/qsp/sim/qsp_sim.cpp` layout.
2. `setupSamplingRun(tEndOfSim, t0)` gained a `t0` parameter
   (`df5d3cfd`). Without this, the fast sampling path re-inited CVODE
   at `t=0` after evolve_to_diagnosis had already advanced ~857 days,
   and the first `simOdeSample` blasted through in garbage steps.
3. Segmented sampling (`8f2cacdf`): `qsp_sim::main` now partitions
   `[t_offset, t_stop]` on dose times, running `simOdeSample` inside
   each segment and `setupSamplingRun` only at boundaries. With no
   doses it degenerates to the previous fast path; with doses it's
   ~300× faster on fine output grids than the step-everywhere
   predecessor. Bonus: doses fire at exact times (not next-tick).

**Shipped (qsp-hpc-tools `cpp-simulation-backend`)**:

1. `CppSimulator` exposes `scenario_yaml` / `drug_metadata_yaml` /
   `healthy_state_yaml` kwargs; config hash folds their file contents
   so scenario edits invalidate the pool (`1b63c99`).
2. `validate_cpp_vs_matlab.py --scenario <yaml>` drives both paths
   through a pdac-build scenario, inferring stop_time + dosing from
   the YAML (`c1428e6`).
3. `HPCJobManager.submit_cpp_jobs` auto-pulls + rebuilds `qsp_sim` on
   HPC via `ensure_cpp_binary()`; `skip_git_pull` / `skip_build`
   escape hatches for iteration (`2f3990e`, `7d1082c`).
4. Scenario YAMLs thread through `submit_cpp_jobs` → uploaded to
   `batch_jobs/input/` → read by `cpp_batch_worker` → passed into
   `CppBatchRunner` (`eddaf79`).

**Validation**:

- `test_ode_vs_matlab.py --run-matlab`: 17/17 pass in 62s, including
  `test_scenario_trajectories_match` (dosing parity),
  `test_evolved_initial_conditions_match` (evolve parity), and
  `test_event_trajectories_match` (events parity).
- M7-harness local sweep (4 sims × 5 varied params each), 25–27×
  C++ speedup on the dosing scenarios:

  | scenario | median rel_diff | Pearson r |
  |---|---|---|
  | `baseline_no_treatment` | 0.13% | 0.9999 |
  | `gvax_neoadjuvant_zheng2022` | 0.13% | 0.9999 |
  | `gvax_nivo_neoadjuvant_zheng2022` | 0.19% | 0.9999 |

  Outliers are near-zero PD1/PDL1 complex species pre-dose
  (MATLAB exactly 0, C++ ~1e-16). Absolute diffs ≤ 2e-14.

### M11 — Push past the 48-task concurrency ceiling

*Est: 0.5–1 day.*

The 100k Rockfish run achieved 48× parallelism wall-side, close to the
`shared` partition's apparent concurrency cap. For 1M-scale sweeps we
hit a ~20 min wall floor from this alone. Options:

1. **Dedicated allocation** (`apopel1` account on `defq`/`parallel`):
   higher concurrent-task ceiling. Needs a credentials change
   (`slurm.partition`) and a re-benchmark.
2. **Multi-node jobs per task**: not applicable — qsp_sim is
   single-process and CppBatchRunner already saturates `cpus_per_task`
   cores within a task.
3. **`max_cpus_per_account` auto-sizing**: `batch_utils.auto_size_max_tasks`
   already handles the one-wave optimization for the MATLAB path;
   submit_cpp_jobs should use the same logic so users don't have to
   hand-tune `jobs_per_chunk` for every partition.

### M12 — Compression + streaming for very large sweeps

*Est: 1 day.*

At 1M-scale (200 GB+ raw), a few efficiency moves matter:

- **Compression**: replace Parquet `snappy` with `zstd` — typical 2–3×
  smaller at negligible read cost. One-line change in
  `CppBatchRunner._write_batch_parquet`.
- **Column pruning**: emit only the species listed in `required_species`
  across the test-stats CSV. The other ~100 species columns go unused
  once test stats are derived. A 3–5× size reduction.
- **Streaming derivation**: today the derivation worker loads one
  Parquet into memory at a time. For 500+ sims/Parquet × 164 species ×
  361 timepoints × 8 bytes ≈ 240 MB per Parquet, which is fine — but
  if we bump sims/chunk higher to amortize the 5–8s startup, streaming
  the Parquet rather than loading it all becomes worthwhile.

## Completed milestones and benchmarks (2026-04-15 through 2026-04-16)

### M1-M7 + M10: DONE

| | what | commit (SPQSP_PDAC) | commit (qsp-hpc-tools) |
|---|---|---|---|
| M1 | qsp_sim binary + raw-binary I/O | f250a0cd | — |
| M2 | ParamXMLRenderer (dict → XML) | — | 57c1a79 |
| M3 | CppRunner (single sim) | — | f2e4141 |
| M4 | CppBatchRunner + MATLAB-schema Parquet | — | b22cb3e |
| M5 | CppSimulator (top-level Python API) | — | (pending) |
| M6 | HPC integration (SLURM + C++ worker) | — | (pending) |
| M7 | Numerical validation vs MATLAB | — | (pending) |
| M10 | Dosing / evolve_to_diagnosis + HPC wiring | 1e85ffc8…8f2cacdf | 1b63c99…7d1082c |

### M7 results (20 sims × 30 days, no dosing, basic no-evolve-to-diagnosis)

- **125/164 meaningful-magnitude species compared** (39 at floating-point noise floor, both paths ≈ 0)
- **Median Pearson r = 1.000000** across meaningful species
- **Median max_rel_diff = 1.76e-6** (CVODE tolerance level)
- **p95 max_rel_diff = 2.23e-4** (tight)
- **Worst non-noise disagreement: V_T.P0/P1 at ~2.6e-3 relative, ~3.7e-9 absolute** (within `abs_tol=1e-9` integrator drift over 60 half-day steps)
- **Speedup: 87× on the 20-sim workload** (MATLAB 3.72 s/sim → C++ 0.04 s/sim)

**Validation gotcha resolved**: MATLAB's `batch_worker.m` hardcodes
`time = 0:0.5:stop_time`, so the C++ side must also use `dt=0.5` for
pointwise alignment. The script enforces this and fails loudly if the
time grids disagree.

**Known asymmetry (not a bug)**: MATLAB emits 253 columns (164 species
+ 78 assignment-rule outputs + 11 compartments); C++ emits just the 164
species. The rule-param gap is an existing codegen gap (noted in M1-M4
findings). All species that appear in both agree.

**Scenario caveat**: `baseline_no_treatment.yaml` uses
`evolve_to_diagnosis` as init — not implemented in C++ yet — so the
validation bypasses the scenario YAML and passes `sim_config` directly
with no `initialization_function`. Validation of scenarios that depend
on `evolve_to_diagnosis` requires M8 or later work on the events/init
branch.

### HPC scaling results (2026-04-16, Rockfish `shared` partition)

First end-to-end C++ sweeps on Rockfish via `HPCJobManager.submit_cpp_jobs()`.
All 110,100 sims across the table succeeded (100%).

| N sims | tasks | jpc | wall | sum task compute | parallelism | ms/sim wall |
|---|---|---|---|---|---|---|
| 100 | 4 | 25 | ~15s | ~3s | 4× | 150 |
| 1k | 40 | 25 | 6s | 45s | 7.5× | 6 |
| 10k | 100 | 100 | 30s | 532s | 17× | 3 |
| **100k** | **200** | **500** | **130s (2m10s)** | **6237s (1h44m)** | **48×** | **1.3** |

**Rockfish vs MATLAB** (extrapolated from the M7 3.72 s/sim local baseline):

- MATLAB laptop serial: 100k ≈ **4.3 days**
- MATLAB HPC (40-task parfor array, typical): 100k ≈ **30–60 min**
- **C++ HPC (this work)**: 100k in **2m 10s** → ~20–40× faster than MATLAB HPC

**Chunk-shape benchmark at N=1000** (varying `jobs_per_chunk`):

| jpc | tasks | wall | parallelism |
|---|---|---|---|
| 10 | 100 | 13s | 4.8× |
| **25** | **40** | **6s** | **7.5×** |
| 50 | 20 | 9s | 5.9× |
| 100 | 10 | 7s | 5.7× |
| 250 | 4 | 12s | 2.6× |

Sweet spot for small N is ~25-100 sims/task. Extreme chunk shapes are
slower: too-small tasks pay setup overhead (module load + venv activate +
Python import ≈ 5–8s); too-large tasks don't parallelize across SLURM.
For 100k we chose jpc=500 (200 tasks) — large enough to avoid SLURM array
limits, small enough to fan out across ~50 concurrent workers.

**Bottlenecks identified**:

1. **Per-task startup ~5–8s** (module load + venv activate + Python import).
   For tasks with fewer than ~25 sims, overhead dominates.
2. **Submit-side overhead ~30s** per `submit_cpp_jobs()` call, mostly from
   `ensure_hpc_venv()` re-upgrading qsp-hpc-tools from GitHub. Amortizes
   across multiple same-session submissions.
3. **Rockfish `shared` partition concurrency cap ~48 tasks**. Above that,
   tasks queue into additional waves. Breaking past this wall (for
   1M-scale sweeps) needs a bigger allocation or a different partition.
4. **Storage scales linearly**: 100k × 180 days × 164 species (snappy
   Parquet) ≈ 21 GB. A 1M-sim sweep would be ~210 GB — at that scale,
   deriving test statistics on the cluster and dropping the raw
   trajectories makes sense (see M9 below).

**Filename-collision fix** (caught in the first smoke sweep): all 4
tasks in the initial 100-sim run started in the same second and 2
tasks' Parquets were overwritten — identical timestamps + scenario +
chunk_size + seed in the filename. Fixed by including
`SLURM_ARRAY_TASK_ID` in the Parquet name
(`batch_{ts}_task{NNN}_{scenario}_{N}sims_seed{S}.parquet`).
`CppSimulator._batch_pattern` accepts both old and new formats.

### HPC scaling results — M10 scenarios (2026-04-16, Rockfish `shared`)

End-to-end scenario sweeps via `scripts/bench_cpp_scenarios.py` on
cpp-sweep-binary-io after the M10 port landed. N=1000, jpc=25 (40
tasks), 4 cpus/task, same seed + 5 varied params across all four
configs. All 4000 sims across the table succeeded (100%).

| config        | tasks | wall | sum task compute | mean/task | parallelism | ms/sim wall |
|---------------|-------|------|------------------|-----------|-------------|-------------|
| no-scenario   | 40    | 7.0s | 27.4s            | 0.68s     | 3.9×        | 7.0         |
| baseline      | 40    | 24.0s | 414.6s          | 10.37s    | 17.3×       | 24.0        |
| gvax          | 40    | 18.0s | 414.3s          | 10.36s    | 23.0×       | 18.0        |
| gvax-nivo     | 40    | 27.0s | 419.5s          | 10.49s    | 15.5×       | 27.0        |

**Key observations**:

1. **Evolve dominates.** All three scenarios sit at ~10.4s mean per
   task (sum_sim ~414–420s) regardless of dosing or post-diagnosis
   duration. The 857-day `evolve_to_diagnosis` is ~95% of per-sim
   work; the 1d/21d/30d post-diagnosis integration and the 1-2
   bolus events are noise on top.
2. **~15× scenario overhead** vs the raw-ODE reference (0.68s →
   10.37s mean/task). Matches the 857-vs-30-day ODE-work ratio.
3. **Per-sim wall at 40 tasks**: 7 ms (no-scenario) vs 18–27 ms
   (scenarios). Extrapolates to ~5–7 min for 1M sims at full
   48-task `shared`-partition fan-out — the M11 ceiling, not a
   scenario-cost artifact.
4. **The no-scenario row is parallelism-noise-limited** (only 3.9×).
   Each task finishes in 0.68s, which is comparable to SLURM's
   per-task dispatch granularity — the array barely staggers.
   Scenario rows hit 15–23×, closer to the 40-task theoretical
   ceiling.

**What this validates, end-to-end**:

- SPQSP_PDAC cpp-sweep-binary-io cherry-picks + `setupSamplingRun(t0)`
  fix + segmented sampling → qsp_sim with `--scenario` +
  `--evolve-to-diagnosis` works on HPC.
- `HPCJobManager.ensure_cpp_binary()` correctly auto-`git pull`s +
  incremental-builds the remote binary from the pushed branch.
- `cpp.build_modules` in credentials.yaml drives the HPC build
  environment (`module purge && module load ...`) exactly like
  `scripts/hpc/build_qsp_sim.sh`.
- Scenario YAMLs upload to `batch_jobs/input/` and thread through to
  the worker's `CppBatchRunner` via `cpp_job_config.json`.

**Gotchas encountered**:

1. **HPC venv reinstalls from git.** `ensure_hpc_venv` re-pip-installs
   qsp-hpc-tools from the configured git source on every submit, so
   unpushed commits make it look like `submit_cpp_jobs`' new kwargs
   are silently ignored. Symptom: workers ran at ~0.3s/task with
   `--evolve-to-diagnosis` config but no actual evolve work. Always
   push qsp-hpc-tools before a scaling run.
2. **`cmake ..` re-runs FetchContent.** The first draft of
   `ensure_cpp_binary` ran `cmake ..` unconditionally, burning ~60–80s
   per submit as SUNDIALS/yaml-cpp git refs re-verified. Fix: only
   run `cmake ..` when `build/CMakeCache.txt` is absent;
   `cmake --build` auto-re-runs configure on CMakeLists.txt changes
   anyway (commit `7d1082c`). Brought per-submit cmake from ~80s to
   ~13s.
3. **Parents-index off-by-one.** The derivation of `cpp_repo_path`
   from `cpp_binary_path` used `parents[3]` (→ `.../PDAC`) instead of
   `parents[4]` (→ `.../SPQSP_PDAC`). Surfaced on the first real
   submit. The unit test used an explicit `cpp_repo_path`, so it
   didn't catch the derivation bug.
4. **`parquets_ok=False` is flaky.** `bench_cpp_scenarios` counts
   `ls {pool_dir}/batch_*.parquet | wc -l` 5s after the job drains;
   file-system sync can lag, so a fully-successful run occasionally
   reports a short count. Cosmetic — `sacct` confirmed all 40 tasks
   `COMPLETED` and `sum_sim` values match across all three scenario
   runs.

### Partition switch: `shared` → `parallel` (2026-04-16)

After the 1k-sim scenario bench, the next question was how to scale toward
1M sims in a 5–10 min wall window. The `shared` partition caps per-job
allocation to 1 node (`MaxNodes=1`), and the 32-CPU `shared` QOS practically
limits fan-out to ~160 CPUs in-flight (40 tasks × 4 cpus). The `parallel`
partition is the big one: 712 nodes × 48 CPUs = 34,176 CPUs total, ~15,000
idle at any moment, no GrpTRES cap on the `apopel1` account, with
`ExclusiveUser=YES` (whole nodes). For evolve-dominated scenarios that's
a ~6× increase in in-flight CPUs.

`bench_cpp_scenarios.py` gained `--partition` and `--time-limit` args
(and the existing `--cpus-per-task` / `--memory` already handled the
whole-node ask).

**10k × gvax-nivo, `parallel`, whole-node tasks:**

| metric | value |
|---|---|
| tasks | 20 (× 500 sims each) |
| cpus/task | 48 (full node via `ExclusiveUser=YES`) |
| memory/task | 180G |
| wall (first-start → last-end) | **34s** |
| per-task wall | 25–29s (tight spread) |
| sum task compute | ~526s |
| parallelism | 15.5× (single wave, all 20 tasks ran concurrently) |
| **ms/sim wall** | **3.4** |
| evolve-rejected sims | ~16% (biologically implausible param draws; returned as `STATUS_FAILED` NaN rows in Parquet) |

**Comparison**: 1k gvax-nivo on `shared` was 27 ms/sim wall; 10k on
`parallel` is 3.4 ms/sim — **~8× better cluster throughput**.

The ~8× factor decomposes cleanly:
- **~6× from more CPUs in flight** (160 → 960)
- **~1.3× from amortizing fixed per-task overhead** (5–8s of module
  load + venv activate + Python import) over 500 sims/task instead of
  25 sims/task

A small countervailing loss (~30% per-CPU throughput at 48-worker
ProcessPool vs 4-worker — NUMA + shared-bus contention on a full
node) is easily swamped by the 12× cpus/task.

**Invocation** for future runs:

```bash
uv run python scripts/bench_cpp_scenarios.py \
  --n-sims 10000 --jobs-per-chunk 500 \
  --cpus-per-task 48 --memory 180G \
  --partition parallel --time-limit 00:30:00 \
  --configs gvax-nivo
```

**Gotchas**:

1. **`192G` memory request fails on `parallel`** even though `MaxMemPerCPU=4000` × 48 = 192000 MB. SLURM interprets `G` as `1024*1024*1024` bytes (196608 MB > 192000 cap). Use `180G` — works cleanly. The SLURM error is misleading: "CPU count per node can not be satisfied" when the actual failure is memory.
2. **Evolve-to-diagnosis rejections are not bugs.** `qsp_sim` exits 2 when the evolve rejects (max_days exceeded, or diagnosis too fast). CppBatchRunner sees this as `QspSimError`, records `STATUS_FAILED`, writes a NaN row. ~16% of 10k lognormal param draws hit one of the rejection criteria — this is the model correctly flagging implausible draws.
3. **Pool cleanup timeout.** `ssh hpc "rm -rf {pool_dir}"` can stall past 60s on 10k+ sim pools (~15 GB). Bench scripts now use `timeout=600`.

**100k × gvax-nivo, `parallel`, 40 whole-node tasks (2026-04-16):**

| metric | value |
|---|---|
| tasks | 40 (× 2500 sims each) |
| cpus/task | 48 (full node) |
| nodes used | 40 (~6% of `parallel`'s 712 total) |
| total cpus in flight | **1920** |
| sweep wall | **238s (3m 58s)** |
| median per-task wall | ~120s |
| **straggler**: task 33 | 238s (**1.98× median**) — set the sweep wall |
| sweep wall *without* the straggler | ~132s → ~1.3 ms/sim |
| reported ms/sim wall | **2.4** |
| parallelism | 20.2× |

**Straggler tax is real at scale.** Task 33 finished at 238s vs a
~130s median; 39 of 40 nodes met the predicted ~1.3 ms/sim wall, but
the 40th stretched the whole sweep to ~2.4 ms/sim. This is a classic
long-tail problem on large clusters — hardware variance, noisy
neighbors on shared network/storage, or an unlucky parameter draw
pushing CVODE into more integration steps. Budget ~1.5–2× the
mean-task wall for sweep planning.

**Scaling curve across runs:**

| N sims | partition | tasks | cpus in flight | wall | ms/sim wall |
|---|---|---|---|---|---|
| 1k | `shared` | 40 | 160 | 27s | 27 |
| 10k | `parallel` | 20 | 960 | 34s | 3.4 |
| 100k | `parallel` | 40 | 1920 | 238s | 2.4 |

**Extrapolation to 1M**: at ~400 nodes × 48 = 19,200 CPUs in flight,
per-task compute for jpc=2500 is ~8.6s, but straggler variance at
400-node concurrency is at least as bad as at 40-node (maybe worse —
more chances to hit a slow node). Realistic wall **3–6 min** for 1M
gvax-nivo, with straggler mitigation (watchdog + requeue, or smaller
chunks for finer granularity) needed to hit the low end.

**Cost (Service Units ≈ CPU-hours).** SLURM array jobs allocate
per-task; each task releases its node back to the pool when it
completes. So billing is `sum(task_elapsed) × cpus_per_task`, not
`wall × nodes_allocated`. Stragglers cost their own extra elapsed
time, not the wait of their idle siblings.

| run | sum elapsed × cpus | CPU-hours | per-sim |
|---|---|---|---|
| 1k `shared` | 415s × 4 | 0.46 | 0.46 µSU/sim |
| 10k `parallel` | 526s × 48 | 7.0 | 0.70 µSU/sim |
| 100k `parallel` | 4991s × 48 | 66.5 | 0.67 µSU/sim |

Per-sim cost on `parallel` is ~constant ≈ 0.67–0.70 µSU/sim — same
solver work per sim. The partition switch doesn't cost more per sim,
it pays for parallelism in wall-clock. Straggler "tax" on the 100k run:
task 33 ran 238s vs 120s median, costing ~1.57 extra CPU-hours (2.4%
of total). Stragglers are a wall-clock problem, not a billing problem.

**1M projection**: ~670 CPU-hours ≈ 670 SUs. On typical JHU ARCH
institutional allocations (~1M SUs/yr per lab), that's ~0.07% of
annual budget. Cloud equivalent ≈ $10–25 on AWS spot; purchased
Rockfish SU ≈ $10.

### Performance milestones (SPQSP_PDAC branch `cpp-sweep-binary-io`)

1. **simOdeSample** (commit 62571257): lets CVODE keep internal history
   across output points instead of CVodeReInit on every step. 2-8× faster
   depending on output cadence (dt=0.1 day: 832 → 104 ms/sim; dt=1 day:
   190 → 84 ms/sim serial).

2. **Jacobian sparsity profile** (commit feea0132): FD-computed Jacobian is
   1.3-1.9% dense, stable across the full 180-day sim (max fan-in 16,
   median 2). Strongly favours KLU sparse solver.

3. **Analytical Jacobian codegen** (commit a37d24f5): qsp_codegen.py emits
   a sympy-derived J with 1434 nnz (5.33% structural density). CSE with
   `optimizations=None` compresses 1.7 MB → ~50 KB.

4. **KLU infrastructure** (commit 06426261): CVODEBase virtual hooks +
   CMake KLU detection + conditional sparse-solver path. USE_KLU defaults
   to OFF because of boundary-singularity issues (see below).

### KLU/analytical Jacobian — deferred

The analytical Jacobian has genuine `1/x` singularities at boundary
states where species = 0 (e.g. `d/d(collagen)` in rate laws that use
collagen in Hill-function denominators). At `collagen = 0`, J entries
evaluate to `Inf` → KLU numeric factorization fails.

Attempted remedies:
- NaN/Inf clamp to 0: gets past setup but CVODE exhausts mxstep budget
  (clamped entries are wrong, Newton iteration diverges on subsequent steps)
- IC floor to 1e-30: CVODE's non-negativity constraint drives species back
  to 0, re-triggering the singularity
- IC floor + constraint removal: segfault mid-integration inside KLU

Future paths (2-5 hrs each, not blocking M5-M8):
- Per-entry FD fallback: detect NaN columns in jac(), compute those via FD
- Regularize rate laws in codegen: add ε to pow/div denominators
- SPGMR iterative solver with matrix-free Jac-vector product: avoids
  explicit J entirely, tolerates singularities

### Benchmark numbers (8-core M-series laptop, Release -O3, dt=1 day)

| metric | value |
|---|---|
| per-sim serial | 84 ms |
| 100-sim parallel (8 workers) | 1.78s (18 ms/sim wall) |
| 10k-sim extrapolated (8 cores) | ~3 min |
| Rockfish 48-core projection | ~30s |
| vs MATLAB SimFunction POC (10k) | ~4.8× faster |

### Parquet schema parity: CONFIRMED

C++ produces 164 species columns; all 164 overlap 100% with MATLAB v4
Parquet from data-apopel1. MATLAB's extra 90 columns are assignment-rule
outputs (H_*, *_total) not emitted by the C++ codegen. Metadata types
(simulation_id, status, time) match.

## Open questions

- **Dosing / event machinery**: separate SPQSP_PDAC branch in flight.
  Current qsp_sim stepping (simOdeSample) is event-free; a hybrid
  stepper is needed for dosing scenarios. The events branch will own
  the stepping-strategy design.
- **Scenario → param diffs**: for non-dosing scenarios the param-XML
  path Just Works. For dosing scenarios, TBD pending events branch.
- **KLU activation**: see deferred section above. Not blocking.

## Total effort estimate

M1–M7 done. Core path (build → validate → scale to 100k on HPC) is
production-viable for no-dosing sweeps.

**Remaining required work:**

- **M8** (MATLAB retirement) — follow-up PR after the C++ path has been
  used in at least one real SBI workflow end-to-end.

**Candidate next steps** (not required, ordered by leverage):

- **M9** (on-cluster test-stat derivation) — needed before 1M-scale
  sweeps are practical. ~1–2 days.
- **M10** (dosing + evolve_to_diagnosis in qsp_sim) — blocks treatment
  scenarios. Depends on SPQSP_PDAC events branch.
- **M11** (push past 48-task concurrency cap) — ~0.5–1 day, mostly a
  credentials / partition change.
- **M12** (compression + column pruning for 1M-scale) — ~1 day, small
  Parquet changes.
