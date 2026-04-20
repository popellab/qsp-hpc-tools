"""Benchmark C++ HPC sweeps across chunk shapes at fixed total N.

Submits a series of sweeps with identical parameters except
``jobs_per_chunk``, waits for each to complete, and reports:

  - parallel wall time (first-task start → last-task end)
  - total compute (sum of per-task qsp_sim work from logs)
  - achieved parallelism (total compute / wall)
  - per-sim throughput (wall / N)
  - submit-side overhead

Deletes each run's pool immediately after measuring to keep disk use
minimal.  Skips the qsp-hpc-tools venv upgrade on subsequent runs.

Usage::

    python scripts/bench_cpp_chunking.py
    python scripts/bench_cpp_chunking.py --n-sims 2000 --chunks 10 25 50 100
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PRIORS_CSV = Path("/Users/joeleliason/Projects/pdac-build/parameters/pdac_priors.csv")
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
    priors = pd.read_csv(priors_csv).set_index("name")
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


def ssh(cmd: str, timeout: int = 60) -> str:
    """Run a shell command on hpc via ssh, return stdout (raises on error)."""
    r = subprocess.run(
        ["ssh", "hpc", cmd], capture_output=True, text=True, timeout=timeout, check=False
    )
    return (r.stdout or "") + (r.stderr or "")


def wait_for_job(job_id: str, n_tasks: int, poll_s: int = 10) -> None:
    """Poll squeue until the job array is fully drained."""
    while True:
        out = ssh(f'squeue -h -j {job_id} -o "%T" | sort -u | tr "\\n" "," | sed "s/,$//"')
        state = out.strip()
        if not state:
            break
        time.sleep(poll_s)


def collect_timing(job_id: str, log_dir: str, pool_dir: str) -> dict:
    """Pull per-task wall (sacct) and sim work (from worker logs)."""
    # Wall times from sacct.
    sacct = ssh(
        f"sacct -j {job_id} --format=JobID,State,Start,End,Elapsed -n -P | "
        "grep -v '\\.' | grep -v '^$'"
    )
    tasks = []
    for line in sacct.strip().split("\n"):
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jobid, state, start, end, elapsed = parts
        tasks.append(
            {"jobid": jobid, "state": state, "start": start, "end": end, "elapsed": elapsed}
        )
    if not tasks:
        return {}
    starts = [t["start"] for t in tasks if t["state"] == "COMPLETED"]
    ends = [t["end"] for t in tasks if t["state"] == "COMPLETED"]
    from datetime import datetime

    def parse(s):
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

    wall = (max(parse(e) for e in ends) - min(parse(s) for s in starts)).total_seconds()

    # Per-task sim work from log regex.
    log_grep = ssh(
        f"grep -h 'Task .* complete' {log_dir}/qsp_cpp_{job_id}_*.out 2>/dev/null || true"
    )
    sim_times = []
    for line in log_grep.splitlines():
        m = re.search(r"succeeded in ([\d.]+)s", line)
        if m:
            sim_times.append(float(m.group(1)))

    # Parquet count.
    pq_count = ssh(f"ls {pool_dir}/batch_*.parquet 2>/dev/null | wc -l").strip()

    return {
        "n_tasks": len(tasks),
        "wall_s": wall,
        "sum_sim_s": sum(sim_times) if sim_times else 0.0,
        "mean_sim_s": (sum(sim_times) / len(sim_times)) if sim_times else 0.0,
        "n_task_logs": len(sim_times),
        "n_parquets": int(pq_count) if pq_count.isdigit() else 0,
    }


def run_one_bench(
    n_sims: int,
    jobs_per_chunk: int,
    pool_id: str,
    scenario: str,
    cpus_per_task: int,
    memory: str,
    seed: int,
    priors_csv: Path,
    params: list[str],
) -> dict:
    """Submit one sweep, wait, collect timing, delete the pool."""
    # Fresh theta (but deterministic — same seed across runs).
    theta, names = sample_params(priors_csv, params, n_sims, seed)
    tmp_csv = Path("/tmp") / f"bench_{pool_id}_params.csv"
    pd.DataFrame(theta, columns=names).to_csv(tmp_csv, index=False)

    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

    manager = HPCJobManager(verbose=False)
    manager.config.time_limit = "00:15:00"

    t_submit = time.time()
    info = manager.submit_cpp_jobs(
        samples_csv=str(tmp_csv),
        num_simulations=n_sims,
        simulation_pool_id=pool_id,
        t_end_days=180.0,
        dt_days=0.5,
        scenario=scenario,
        seed=seed,
        jobs_per_chunk=jobs_per_chunk,
        max_workers=cpus_per_task,
        cpp_cpus_per_task=cpus_per_task,
        cpp_memory=memory,
    )
    submit_s = time.time() - t_submit
    job_id = info.job_ids[0]

    print(f"    job_id={job_id}, tasks={info.n_jobs}, submit_s={submit_s:.1f}s. Waiting...")
    wait_for_job(job_id, info.n_jobs)

    # Allow sacct to catch up.
    time.sleep(5)

    pool_dir = f"{manager.config.simulation_pool_path}/{pool_id}"
    log_dir = f"{manager.config.remote_project_path}/batch_jobs/logs"
    timing = collect_timing(job_id, log_dir, pool_dir)
    timing["submit_s"] = submit_s
    timing["jobs_per_chunk"] = jobs_per_chunk

    # Delete pool to save disk. Generous timeout: at 10k+ sims the pool is
    # multi-GB and ssh can hang well past 60s closing the connection even
    # though the remote rm itself is fast.
    ssh(f"rm -rf {pool_dir}", timeout=600)
    tmp_csv.unlink(missing_ok=True)

    return timing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 250],
        help="jobs_per_chunk values to benchmark",
    )
    ap.add_argument("--cpus-per-task", type=int, default=4)
    ap.add_argument("--memory", default="4G")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 80)
    print(f"C++ HPC chunking benchmark: N={args.n_sims}, cpus/task={args.cpus_per_task}")
    print(f"jobs_per_chunk: {args.chunks}")
    print("=" * 80)

    results = []
    for jpc in args.chunks:
        n_tasks = (args.n_sims + jpc - 1) // jpc
        print(f"\n>>> Benchmarking jobs_per_chunk={jpc} → {n_tasks} tasks")
        timing = run_one_bench(
            n_sims=args.n_sims,
            jobs_per_chunk=jpc,
            pool_id=f"bench_jpc{jpc}",
            scenario=f"bench_jpc{jpc}",
            cpus_per_task=args.cpus_per_task,
            memory=args.memory,
            seed=args.seed,
            priors_csv=DEFAULT_PRIORS_CSV,
            params=DEFAULT_SAMPLED_PARAMS,
        )
        print(
            f"    wall={timing['wall_s']:.1f}s  "
            f"sum_sim={timing['sum_sim_s']:.1f}s  "
            f"mean_sim/task={timing['mean_sim_s']:.2f}s  "
            f"parquets={timing['n_parquets']}"
        )
        results.append(timing)

    # Summary table.
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    cols = [
        ("jpc", 6),
        ("tasks", 6),
        ("wall_s", 8),
        ("sum_sim_s", 11),
        ("mean_sim/task_s", 17),
        ("speedup_from_parallelism", 26),
        ("ms/sim_wall", 13),
        ("parquets_ok", 13),
    ]
    print("".join(f"{c:<{w}}" for c, w in cols))
    print("-" * sum(w for _, w in cols))
    for r in results:
        parallelism = r["sum_sim_s"] / r["wall_s"] if r["wall_s"] > 0 else 0.0
        ms_per_sim = r["wall_s"] / args.n_sims * 1000
        ok = r["n_parquets"] == r["n_tasks"]
        print(
            f"{r['jobs_per_chunk']:<6}"
            f"{r['n_tasks']:<6}"
            f"{r['wall_s']:<8.1f}"
            f"{r['sum_sim_s']:<11.1f}"
            f"{r['mean_sim_s']:<17.2f}"
            f"{parallelism:<26.1f}"
            f"{ms_per_sim:<13.1f}"
            f"{str(ok):<13}"
        )
    print()


if __name__ == "__main__":
    main()
