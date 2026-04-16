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

## Completed milestones and benchmarks (2026-04-15)

### M1-M4: DONE

| | what | commit (SPQSP_PDAC) | commit (qsp-hpc-tools) |
|---|---|---|---|
| M1 | qsp_sim binary + raw-binary I/O | f250a0cd | — |
| M2 | ParamXMLRenderer (dict → XML) | — | 57c1a79 |
| M3 | CppRunner (single sim) | — | f2e4141 |
| M4 | CppBatchRunner + MATLAB-schema Parquet | — | b22cb3e |

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

M1-M4 done. Remaining: M5 (~1 day), M6 (~2-3 days), M7 (~2 days),
M8 (follow-up PR after M7 is validated in production).
