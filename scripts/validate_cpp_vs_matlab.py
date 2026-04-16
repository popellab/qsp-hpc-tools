"""M7 validation: run the same sweep through MATLAB and C++, compare trajectories.

Runs the same fixed-seed parameter sweep through:
    1. MATLAB path   — `run_batch_worker` → `batch_worker.m` → sbiosimulate
    2. C++ path      — `CppBatchRunner`   → qsp_sim binary

Produces:
    - Per-species max |diff|, RMSE, Pearson r
    - Wall-clock timing for each path
    - Pickle with the comparison for deeper inspection

Scenario design: `baseline_no_treatment.yaml` uses `evolve_to_diagnosis` as an
initialization function (not implemented in C++ yet). So we bypass the scenario
YAML and pass a custom `sim_config` directly — no init function, no dosing.
The model's native ICs are used (same path as `export_trajectories.m` used for
the original single-sim validation).

Usage (defaults = 20 sims, 5 params sampled, t_end=30 days)::

    python scripts/validate_cpp_vs_matlab.py

Outputs land under ``cache/validation/``.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_PRIORS_CSV = Path("/Users/joeleliason/Projects/pdac-build/parameters/pdac_priors.csv")
DEFAULT_MATLAB_PROJECT_ROOT = Path("/Users/joeleliason/Projects/pdac-build")
DEFAULT_MATLAB_MODEL_SCRIPT = "immune_oncology_model_PDAC"
DEFAULT_CPP_BINARY = Path(
    "/Users/joeleliason/Projects/SPQSP_PDAC-cpp-sweep/PDAC/qsp/sim/build/qsp_sim"
)
DEFAULT_CPP_TEMPLATE = Path(
    "/Users/joeleliason/Projects/SPQSP_PDAC-cpp-sweep/PDAC/sim/resource/param_all.xml"
)

# Parameters to actually vary — all 5 are core rate constants that exist
# in both the SimBiology model and param_all.xml.
DEFAULT_SAMPLED_PARAMS = [
    "k_C1_growth",
    "k_C1_death",
    "k_Treg_pro",
    "k_Treg_death",
    "k_cell_clear",
]


def sample_params(
    priors_csv: Path,
    param_names: list[str],
    n_sims: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Draw ``n_sims`` lognormal samples for each of ``param_names``."""
    priors = pd.read_csv(priors_csv)
    priors = priors.set_index("name")
    missing = set(param_names) - set(priors.index)
    if missing:
        raise ValueError(f"Param(s) not in priors CSV: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    rows = []
    for name in param_names:
        row = priors.loc[name]
        if row["distribution"] != "lognormal":
            raise ValueError(f"{name}: expected lognormal, got {row['distribution']}")
        samples = rng.lognormal(
            mean=float(row["dist_param1"]),
            sigma=float(row["dist_param2"]),
            size=n_sims,
        )
        rows.append(samples)
    return np.column_stack(rows), list(param_names)


def run_matlab(
    theta: np.ndarray,
    param_names: list[str],
    project_root: Path,
    model_script: str,
    t_end_days: float,
    dt_days: float,
    seed: int,
    out_dir: Path,
) -> tuple[Path, float]:
    """Run the MATLAB path end-to-end, return (parquet, wall_seconds)."""
    from qsp_hpc.simulation.batch_runner import run_batch_worker

    # sim_config WITHOUT initialization_function so we skip evolve_to_diagnosis.
    # NB: batch_worker.m hardcodes dt=0.5 — it ignores any time_vector we pass.
    # Match this on the C++ side by defaulting dt_days to 0.5 in the caller.
    if dt_days != 0.5:
        raise ValueError(
            f"MATLAB's batch_worker.m hardcodes dt=0.5; got dt_days={dt_days}. "
            "Pass --dt-days 0.5 or adjust batch_worker.m to honor a custom grid."
        )
    sim_config = {
        "start_time": 0,
        "stop_time": t_end_days,
        "time_units": "day",
        "solver": "sundials",
        "abs_tolerance": 1.0e-9,
        "rel_tolerance": 1.0e-6,
    }

    t0 = time.time()
    parquet = run_batch_worker(
        params=theta,
        param_names=param_names,
        model_script=model_script,
        project_root=project_root,
        seed=seed,
        sim_config=sim_config,
        simulation_pool_path=out_dir,
        simulation_pool_id="matlab_run",
        verbose=True,
    )
    elapsed = time.time() - t0
    return parquet, elapsed


def run_cpp(
    theta: np.ndarray,
    param_names: list[str],
    binary_path: Path,
    template_path: Path,
    t_end_days: float,
    dt_days: float,
    seed: int,
    out_dir: Path,
    max_workers: int | None = None,
) -> tuple[Path, float]:
    """Run the C++ path end-to-end, return (parquet, wall_seconds)."""
    from qsp_hpc.cpp.batch_runner import CppBatchRunner

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "cpp_run.parquet"

    runner = CppBatchRunner(
        binary_path=binary_path,
        template_path=template_path,
        subtree="QSP",
    )
    t0 = time.time()
    runner.run(
        theta_matrix=theta,
        param_names=param_names,
        t_end_days=t_end_days,
        dt_days=dt_days,
        output_path=output_path,
        scenario="cpp_validation",
        seed=seed,
        max_workers=max_workers,
    )
    elapsed = time.time() - t0
    return output_path, elapsed


def compare_parquets(
    matlab_parquet: Path,
    cpp_parquet: Path,
) -> pd.DataFrame:
    """Return a per-species agreement table."""
    ml = pq.read_table(str(matlab_parquet))
    cp = pq.read_table(str(cpp_parquet))

    ml_species = {
        c
        for c in ml.column_names
        if not c.startswith(("param:", "simulation_id", "status", "time", "sample_index"))
    }
    cp_species = {
        c
        for c in cp.column_names
        if not c.startswith(("param:", "simulation_id", "status", "time", "sample_index"))
    }
    common = sorted(ml_species & cp_species)
    matlab_only = sorted(ml_species - cp_species)
    cpp_only = sorted(cp_species - ml_species)

    print(
        f"Species: {len(common)} common, "
        f"{len(matlab_only)} MATLAB-only, {len(cpp_only)} C++-only"
    )
    if matlab_only[:3]:
        print(f"  MATLAB-only (first 3): {matlab_only[:3]}")
    if cpp_only[:3]:
        print(f"  C++-only (first 3): {cpp_only[:3]}")

    # Filter to successful rows only.
    ml_df = ml.to_pandas()
    cp_df = cp.to_pandas()

    # Sanity: time grids must agree before we compare trajectories by index.
    ml_t = np.asarray(ml_df.iloc[0]["time"], dtype=float)
    cp_t = np.asarray(cp_df.iloc[0]["time"], dtype=float)
    if len(ml_t) != len(cp_t) or not np.allclose(ml_t, cp_t, atol=1e-9):
        raise RuntimeError(
            f"Time axes differ — comparison would be meaningless.\n"
            f"  MATLAB t (len={len(ml_t)}): {ml_t[:5]} ... {ml_t[-3:]}\n"
            f"  C++ t    (len={len(cp_t)}): {cp_t[:5]} ... {cp_t[-3:]}\n"
            f"Fix: pass matching dt/t_end to both paths."
        )

    # MATLAB's status convention is 1=success, -1/0=fail.
    # CppBatchRunner's is 0=success, 1=fail.
    ml_ok = ml_df["status"] == 1
    cp_ok = cp_df["status"] == 0
    both_ok = ml_ok.values & cp_ok.values
    n_both_ok = int(both_ok.sum())
    print(f"Sims with both paths succeeding: {n_both_ok}/{len(ml_df)}")
    if n_both_ok == 0:
        raise RuntimeError("No sims succeeded in both paths — can't compare")

    rows = []
    for sp in common:
        # Each row is a list of floats (one per timepoint).
        ml_arr = np.stack([np.asarray(v, dtype=float) for v in ml_df.loc[both_ok, sp]])
        cp_arr = np.stack([np.asarray(v, dtype=float) for v in cp_df.loc[both_ok, sp]])

        # Truncate to shortest time axis (MATLAB often has finer output).
        n_t = min(ml_arr.shape[1], cp_arr.shape[1])
        ml_sub = ml_arr[:, :n_t]
        cp_sub = cp_arr[:, :n_t]

        # Skip if either is all-NaN.
        if np.all(np.isnan(ml_sub)) or np.all(np.isnan(cp_sub)):
            continue

        diff = np.abs(ml_sub - cp_sub)
        scale = np.maximum(np.abs(ml_sub), 1e-30)
        rel_diff = diff / scale

        # Nan-safe stats
        with np.errstate(invalid="ignore"):
            max_abs = float(np.nanmax(diff))
            rmse = float(np.sqrt(np.nanmean(diff**2)))
            max_rel = float(np.nanmax(rel_diff))
            ml_flat = ml_sub.flatten()
            cp_flat = cp_sub.flatten()
            mask = np.isfinite(ml_flat) & np.isfinite(cp_flat)
            if mask.sum() > 1 and np.std(ml_flat[mask]) > 0 and np.std(cp_flat[mask]) > 0:
                corr = float(np.corrcoef(ml_flat[mask], cp_flat[mask])[0, 1])
            else:
                corr = float("nan")

        rows.append(
            {
                "species": sp,
                "max_abs_diff": max_abs,
                "rmse": rmse,
                "max_rel_diff": max_rel,
                "pearson_r": corr,
                "matlab_final_mean": float(np.nanmean(ml_sub[:, -1])),
                "cpp_final_mean": float(np.nanmean(cp_sub[:, -1])),
            }
        )

    return pd.DataFrame(rows).sort_values("max_rel_diff", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-end-days", type=float, default=30.0)
    ap.add_argument(
        "--dt-days",
        type=float,
        default=0.5,
        help="Output timestep. Must be 0.5 to match batch_worker.m's hardcoded grid.",
    )
    ap.add_argument("--priors-csv", type=Path, default=DEFAULT_PRIORS_CSV)
    ap.add_argument("--matlab-project-root", type=Path, default=DEFAULT_MATLAB_PROJECT_ROOT)
    ap.add_argument("--matlab-model-script", default=DEFAULT_MATLAB_MODEL_SCRIPT)
    ap.add_argument("--cpp-binary", type=Path, default=DEFAULT_CPP_BINARY)
    ap.add_argument("--cpp-template", type=Path, default=DEFAULT_CPP_TEMPLATE)
    ap.add_argument(
        "--params",
        nargs="+",
        default=DEFAULT_SAMPLED_PARAMS,
        help="Parameter names to vary (must exist in both priors CSV and XML template)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/validation/cpp_vs_matlab"),
    )
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument(
        "--skip-matlab",
        action="store_true",
        help="Reuse an existing MATLAB Parquet instead of re-running",
    )
    args = ap.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(
        f"M7 validation: {args.n_sims} sims, {len(args.params)} varied params, "
        f"{args.t_end_days} days"
    )
    print(f"  MATLAB:   {args.matlab_model_script} in {args.matlab_project_root}")
    print(f"  C++:      {args.cpp_binary.name}")
    print(f"  Template: {args.cpp_template.name}")
    print(f"  Params:   {args.params}")
    print(f"  Output:   {out_dir}")
    print("=" * 70)

    # 1. Sample parameters.
    theta, param_names = sample_params(
        priors_csv=args.priors_csv,
        param_names=args.params,
        n_sims=args.n_sims,
        seed=args.seed,
    )
    print(f"Sampled theta: shape={theta.shape}")
    print(f"  min:  {theta.min(axis=0)}")
    print(f"  max:  {theta.max(axis=0)}")

    # Persist the theta matrix so reruns produce identical inputs.
    pd.DataFrame(theta, columns=param_names).to_csv(out_dir / "theta.csv", index=False)

    # 2. MATLAB path.
    matlab_parquet_cached = out_dir / "matlab_run" / "batch_0_*.parquet"
    if args.skip_matlab:
        hits = list((out_dir / "matlab_run").glob("batch_*.parquet"))
        if not hits:
            raise FileNotFoundError(f"No cached MATLAB parquet at {matlab_parquet_cached}")
        matlab_parquet = hits[0]
        matlab_time = float("nan")
        print(f"\n[MATLAB] Using cached: {matlab_parquet.name}")
    else:
        print("\n[MATLAB] Running...")
        matlab_parquet, matlab_time = run_matlab(
            theta=theta,
            param_names=param_names,
            project_root=args.matlab_project_root,
            model_script=args.matlab_model_script,
            t_end_days=args.t_end_days,
            dt_days=args.dt_days,
            seed=args.seed,
            out_dir=out_dir,
        )
        print(f"[MATLAB] Done in {matlab_time:.1f}s → {matlab_parquet.name}")

    # 3. C++ path.
    print("\n[C++] Running...")
    cpp_parquet, cpp_time = run_cpp(
        theta=theta,
        param_names=param_names,
        binary_path=args.cpp_binary,
        template_path=args.cpp_template,
        t_end_days=args.t_end_days,
        dt_days=args.dt_days,
        seed=args.seed,
        out_dir=out_dir,
        max_workers=args.max_workers,
    )
    print(f"[C++] Done in {cpp_time:.1f}s → {cpp_parquet.name}")

    # 4. Compare.
    print("\n[Compare] Per-species agreement:")
    report = compare_parquets(matlab_parquet, cpp_parquet)
    report.to_csv(out_dir / "species_report.csv", index=False)

    # Flag species whose magnitudes are below a floating-point noise floor so
    # their "huge relative disagreement" doesn't drown out real findings.
    # 1e-9 matches the CVODE abs_tolerance — anything below that is ODE-solver
    # noise, not a real numerical disagreement.
    noise_floor = 1e-9
    meaningful = report[
        (report["matlab_final_mean"].abs() > noise_floor)
        | (report["cpp_final_mean"].abs() > noise_floor)
    ]
    noise = report[~report.index.isin(meaningful.index)]

    print("\nTop 10 worst-agreement species among meaningful-magnitude (|final| > 1e-10):")
    print(meaningful.head(10).to_string(index=False))

    print("\nOverall (meaningful-magnitude species only):")
    print(f"  meaningful species:    {len(meaningful)} / {len(report)}")
    print(f"  noise-floor species:   {len(noise)} (MATLAB & C++ both ≈ 0)")
    print(f"  median max_rel_diff:   {meaningful['max_rel_diff'].median():.3e}")
    print(f"  p95 max_rel_diff:      {meaningful['max_rel_diff'].quantile(0.95):.3e}")
    print(f"  worst max_rel_diff:    {meaningful['max_rel_diff'].max():.3e}")
    print(f"  median Pearson r:      {meaningful['pearson_r'].median():.6f}")
    print(f"  min Pearson r:         {meaningful['pearson_r'].min():.6f}")

    print("\nTiming:")
    if not np.isnan(matlab_time):
        print(f"  MATLAB:    {matlab_time:.1f}s ({matlab_time/args.n_sims:.2f}s/sim)")
    print(f"  C++:       {cpp_time:.1f}s ({cpp_time/args.n_sims:.2f}s/sim)")
    if not np.isnan(matlab_time) and cpp_time > 0:
        print(f"  speedup:   {matlab_time/cpp_time:.1f}×")

    # Pickle full results.
    with open(out_dir / "validation_result.pkl", "wb") as f:
        pickle.dump(
            {
                "args": vars(args),
                "theta": theta,
                "param_names": param_names,
                "matlab_parquet": str(matlab_parquet),
                "cpp_parquet": str(cpp_parquet),
                "matlab_time_s": matlab_time,
                "cpp_time_s": cpp_time,
                "report": report,
            },
            f,
        )
    print(f"\nFull result pickled: {out_dir / 'validation_result.pkl'}")


if __name__ == "__main__":
    main()
