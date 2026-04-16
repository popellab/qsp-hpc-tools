"""Submit a 100-sim C++ sweep to HPC and record timing.

Quick smoke test of the C++ backend end-to-end on Rockfish:
  1. Samples 100 thetas from pdac_priors.csv (5 varied params)
  2. Uploads params.csv + cpp_job_config.json to HPC
  3. Submits a SLURM array job running qsp_sim via cpp_batch_worker
  4. Polls squeue + reports timing when done

Usage::

    python scripts/submit_cpp_smoke_sweep.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PRIORS_CSV = Path("/Users/joeleliason/Projects/pdac-build/parameters/pdac_priors.csv")

# Same 5 params as M7 local validation.
DEFAULT_SAMPLED_PARAMS = [
    "k_C1_growth",
    "k_C1_death",
    "k_Treg_pro",
    "k_Treg_death",
    "k_cell_clear",
]


def sample_params(
    priors_csv: Path, param_names: list[str], n_sims: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    """Draw lognormal samples for ``param_names`` from the priors CSV."""
    priors = pd.read_csv(priors_csv).set_index("name")
    missing = set(param_names) - set(priors.index)
    if missing:
        raise ValueError(f"Param(s) not in priors CSV: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    cols = []
    for name in param_names:
        row = priors.loc[name]
        cols.append(
            rng.lognormal(
                mean=float(row["dist_param1"]),
                sigma=float(row["dist_param2"]),
                size=n_sims,
            )
        )
    return np.column_stack(cols), list(param_names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-end-days", type=float, default=180.0)
    ap.add_argument("--dt-days", type=float, default=0.5)
    ap.add_argument(
        "--scenario",
        default="cpp_hpc_smoke",
        help="Scenario label (used in Parquet filenames + pool dir suffix).",
    )
    ap.add_argument(
        "--scenario-yaml",
        type=Path,
        default=None,
        help=(
            "pdac-build scenario YAML (e.g. scenarios/baseline_no_treatment.yaml). "
            "When set, qsp_sim runs with full dosing/init from the YAML; "
            "requires --drug-metadata-yaml."
        ),
    )
    ap.add_argument(
        "--drug-metadata-yaml",
        type=Path,
        default=None,
        help="SPQSP_PDAC drug_metadata.yaml (required with --scenario-yaml).",
    )
    ap.add_argument(
        "--healthy-state-yaml",
        type=Path,
        default=None,
        help=(
            "SPQSP_PDAC healthy_state.yaml — required for scenarios with "
            "initialization_function: evolve_to_diagnosis."
        ),
    )
    ap.add_argument("--pool-id", default="cpp_smoke_v1")
    ap.add_argument("--jobs-per-chunk", type=int, default=25)
    ap.add_argument("--cpus-per-task", type=int, default=4)
    ap.add_argument("--memory", default="4G")
    ap.add_argument("--time-limit", default="00:30:00")
    ap.add_argument("--priors-csv", type=Path, default=DEFAULT_PRIORS_CSV)
    ap.add_argument(
        "--params",
        nargs="+",
        default=DEFAULT_SAMPLED_PARAMS,
        help="Parameter names to vary",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/hpc_smoke"),
    )
    ap.add_argument(
        "--derive-test-stats",
        action="store_true",
        help=(
            "Chain a test-stats derivation job after the C++ array using "
            "--dependency=afterok:<array_id>. Requires --calibration-targets "
            "or --test-stats-csv."
        ),
    )
    ap.add_argument(
        "--calibration-targets",
        type=Path,
        default=None,
        help=(
            "Directory of calibration target YAMLs (public-facing API used "
            "by pdac-build). Mutually exclusive with --test-stats-csv."
        ),
    )
    ap.add_argument(
        "--test-stats-csv",
        type=Path,
        default=None,
        help=(
            "Path to a test_stats.csv (legacy/internal form). Mutually "
            "exclusive with --calibration-targets."
        ),
    )
    ap.add_argument(
        "--model-structure-file",
        type=Path,
        default=None,
        help="Optional model_structure.json with species unit metadata.",
    )
    args = ap.parse_args()

    if args.calibration_targets and args.test_stats_csv:
        ap.error("--calibration-targets and --test-stats-csv are mutually exclusive")
    if args.derive_test_stats and not (args.calibration_targets or args.test_stats_csv):
        ap.error("--derive-test-stats requires --calibration-targets or --test-stats-csv")
    if args.derive_test_stats and not args.model_structure_file:
        ap.error(
            "--derive-test-stats requires --model-structure-file (path to "
            "model_structure.json) — without it the derivation worker tags "
            "every species as dimensionless and most cal-target unit "
            "conversions silently NaN out."
        )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"C++ HPC smoke sweep: {args.n_sims} sims, {args.t_end_days} days")
    print(f"  jobs_per_chunk:  {args.jobs_per_chunk}")
    print(f"  array tasks:     {(args.n_sims + args.jobs_per_chunk - 1) // args.jobs_per_chunk}")
    print(f"  cpus/task:       {args.cpus_per_task}")
    print(f"  memory:          {args.memory}")
    print(f"  time limit:      {args.time_limit}")
    print("=" * 70)

    # Sample and write params CSV.
    theta, names = sample_params(
        priors_csv=args.priors_csv,
        param_names=args.params,
        n_sims=args.n_sims,
        seed=args.seed,
    )
    params_csv = out_dir / "params.csv"
    pd.DataFrame(theta, columns=names).to_csv(params_csv, index=False)
    print(f"\nWrote params: {params_csv}")
    print(f"  shape: {theta.shape}, lognormal samples with seed={args.seed}")

    # Submit via HPCJobManager.
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

    manager = HPCJobManager(verbose=True)

    # Override time_limit for the C++ job (shorter than MATLAB default).
    manager.config.time_limit = args.time_limit

    test_stats_hash: str | None = None
    test_stats_csv_for_submit: Path | None = args.test_stats_csv
    if args.derive_test_stats:
        from qsp_hpc.utils.hash_utils import compute_test_stats_hash

        if args.calibration_targets is not None:
            # Mirror QSPSimulator/CppSimulator: serialize calibration target
            # YAMLs to a temp CSV and let submit_cpp_jobs upload that.
            from qsp_hpc.calibration import load_calibration_targets

            cal_df = load_calibration_targets(args.calibration_targets)
            tmp_csv = out_dir / "calibration_targets.csv"
            cal_df.to_csv(tmp_csv, index=False)
            test_stats_csv_for_submit = tmp_csv
            print(
                f"\nSerialized {args.calibration_targets} → {tmp_csv} " f"({len(cal_df)} target(s))"
            )

        test_stats_hash = compute_test_stats_hash(test_stats_csv_for_submit)
        print(f"Deriving test stats on cluster: hash={test_stats_hash[:8]}...")

    t_submit = time.time()
    info = manager.submit_cpp_jobs(
        samples_csv=str(params_csv),
        num_simulations=args.n_sims,
        simulation_pool_id=args.pool_id,
        t_end_days=args.t_end_days,
        dt_days=args.dt_days,
        scenario=args.scenario,
        seed=args.seed,
        jobs_per_chunk=args.jobs_per_chunk,
        max_workers=args.cpus_per_task,
        cpp_cpus_per_task=args.cpus_per_task,
        cpp_memory=args.memory,
        scenario_yaml=str(args.scenario_yaml) if args.scenario_yaml else None,
        drug_metadata_yaml=(str(args.drug_metadata_yaml) if args.drug_metadata_yaml else None),
        healthy_state_yaml=(str(args.healthy_state_yaml) if args.healthy_state_yaml else None),
        derive_test_stats=args.derive_test_stats,
        test_stats_csv=(str(test_stats_csv_for_submit) if test_stats_csv_for_submit else None),
        test_stats_hash=test_stats_hash,
        model_structure_file=(
            str(args.model_structure_file) if args.model_structure_file else None
        ),
    )
    submit_time = time.time() - t_submit

    print()
    print("=" * 70)
    print(f"Submitted: job_id={info.job_ids}, tasks={info.n_jobs}")
    print(f"State file: {info.state_file}")
    print(f"Submit time: {submit_time:.1f}s")
    print("=" * 70)
    print()
    print("Watch progress with:")
    watched = ",".join(info.job_ids)
    print(f"  ssh hpc 'squeue -u $USER -j {watched}'")
    print(f"  ssh hpc 'tail -f ~/qsp-projects/batch_jobs/logs/qsp_cpp_{info.job_ids[0]}_*.out'")
    if args.derive_test_stats and len(info.job_ids) > 1:
        derive_id = info.job_ids[1]
        print(f"  ssh hpc 'tail -f ~/qsp-projects/batch_jobs/logs/qsp_derive_{derive_id}.out'")


if __name__ == "__main__":
    main()
