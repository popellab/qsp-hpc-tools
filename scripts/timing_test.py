#!/usr/bin/env python3
"""
HPC timing probe for SimBiology startup stages.

Submits a small HPC batch with `accelerate=True`. Each array task prints a
`[timing-summary]` line covering startup, model build (eval), simulation_config,
sbioaccelerate, and parfor_total — use this to decide whether a SimBiology
`export`/`accelerate` deployment refactor is worth the API churn.

Run from a project directory that has the usual pdac-build layout
(priors CSV, calibration_targets/, scenarios/, model script on HPC sync path).
Defaults match pdac-build; override with flags for other projects.

    # from ~/Projects/pdac-build
    ../qsp-hpc-tools/scripts/timing_test.py --n-sims 40 --sims-per-task 10

After SLURM completes, grep the per-task stdout for timing lines:

    grep "\\[timing" logs/*.out           # if collected locally
    qsp-hpc logs                          # browse latest job's task logs

Each array task emits a single summary line, e.g.:

    [timing-summary] startup=1.23s model_build=0.45s sim_config=0.02s \\
      sbioaccelerate=18.7s parfor_total=12.4s n_patients=10

The floor we're trying to amortize is (startup + model_build + sbioaccelerate),
paid once per array task today.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--priors-csv", default="parameters/pdac_priors.csv")
    ap.add_argument("--calibration-targets", default="calibration_targets/baseline_no_treatment")
    ap.add_argument("--model-structure-file", default="model_structure.json")
    ap.add_argument(
        "--model-script",
        default="immune_oncology_model_PDAC",
        help="MATLAB model script name (must resolve on HPC after rsync)",
    )
    ap.add_argument("--scenario", default="baseline_no_treatment")
    ap.add_argument(
        "--model-version",
        default="timing_probe",
        help="Version tag; default forces a fresh pool → guarantees HPC submit",
    )
    ap.add_argument("--n-sims", type=int, default=20)
    ap.add_argument(
        "--sims-per-task",
        type=int,
        default=5,
        help="Controls array-task count (≈ n-sims / sims-per-task)",
    )
    ap.add_argument(
        "--no-accelerate",
        action="store_true",
        help="Disable sbioaccelerate for a baseline (no MEX compile) comparison",
    )
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    project_root = Path.cwd().resolve()

    # Sanity-check required inputs exist in CWD before spinning up a submission.
    missing = [
        p for p in (args.priors_csv, args.calibration_targets) if not (project_root / p).exists()
    ]
    if missing:
        print(f"error: missing required paths (cwd={project_root}):", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print("Pass --priors-csv / --calibration-targets to override.", file=sys.stderr)
        return 2

    n_tasks = max(1, -(-args.n_sims // args.sims_per_task))  # ceil div

    print("=" * 72)
    print("SimBiology HPC startup timing probe")
    print("=" * 72)
    print(f"  project_root     : {project_root}")
    print(f"  model_script     : {args.model_script}")
    print(f"  scenario         : {args.scenario}")
    print(f"  priors_csv       : {args.priors_csv}")
    print(f"  calibration      : {args.calibration_targets}")
    print(f"  model_version    : {args.model_version}   (fresh pool → forces HPC submit)")
    print(f"  n_sims           : {args.n_sims}")
    print(f"  sims_per_task    : {args.sims_per_task}  →  ~{n_tasks} array tasks")
    print(f"  accelerate_model : {not args.no_accelerate}")
    print(f"  seed             : {args.seed}")
    print("=" * 72)

    from qsp_hpc import QSPSimulator

    sim = QSPSimulator(
        priors_csv=args.priors_csv,
        calibration_targets=args.calibration_targets,
        model_structure_file=args.model_structure_file,
        model_script=args.model_script,
        model_version=args.model_version,
        model_description="SimBiology startup timing probe",
        scenario=args.scenario,
        project_root=project_root,
        seed=args.seed,
        max_tasks=n_tasks,
        accelerate=not args.no_accelerate,
        verbose=True,
    )

    theta, x = sim(args.n_sims)

    print()
    print(f"Done. theta={theta.shape}  x={x.shape}")
    print()
    print("Next — pull timing lines from collected HPC logs:")
    print("    qsp-hpc logs")
    print("    # or, once logs are local:")
    print("    grep '\\[timing' logs/*.out | sort")
    return 0


if __name__ == "__main__":
    sys.exit(main())
