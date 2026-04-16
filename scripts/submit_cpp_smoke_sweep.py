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
    ap.add_argument("--scenario", default="cpp_hpc_smoke")
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
    args = ap.parse_args()

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
    print(f"  ssh hpc 'squeue -u $USER -j {info.job_ids[0]}'")
    print(f"  ssh hpc 'tail -f ~/qsp-projects/batch_jobs/logs/qsp_cpp_{info.job_ids[0]}_*.out'")


if __name__ == "__main__":
    main()
