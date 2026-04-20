"""Benchmark C++ HPC sweeps across M10 scenarios at fixed N.

For each of several scenarios, submits a 1k-sim sweep (same theta, same jpc),
waits for the job array to drain, and reports:

  - wall (first task start → last task end)
  - sum of per-task qsp_sim compute time (from worker logs)
  - parallelism = sum_compute / wall
  - ms/sim wall

Scenarios exercise the newly-wired M10 code paths (evolve_to_diagnosis,
schedule_dosing, segmented sampling). A no-scenario row is included as
a baseline so the per-scenario overhead is visible.

Usage::

    python scripts/bench_cpp_scenarios.py
    python scripts/bench_cpp_scenarios.py --configs no-scenario baseline
    python scripts/bench_cpp_scenarios.py --n-sims 2000 --jobs-per-chunk 50

Reuses the polling / timing helpers from ``bench_cpp_chunking``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Reuse polling + timing helpers from the sibling chunking bench. scripts/
# isn't a package, so add it to sys.path before importing.
sys.path.insert(0, str(Path(__file__).parent))
from bench_cpp_chunking import (  # noqa: E402
    DEFAULT_SAMPLED_PARAMS,
    collect_timing,
    sample_params,
    ssh,
    wait_for_job,
)

DEFAULT_PRIORS_CSV = Path("/Users/joeleliason/Projects/pdac-build/parameters/pdac_priors.csv")

# pdac-build scenario YAMLs (local paths; uploaded to HPC per-submit).
PDAC_BUILD_SCENARIOS = Path("/Users/joeleliason/Projects/pdac-build/scenarios")
# Drug-metadata + healthy-state live with the SPQSP_PDAC C++ side.
SPQSP_PDAC_RESOURCE = Path("/Users/joeleliason/Projects/SPQSP_PDAC-cpp-sweep/PDAC/sim/resource")
DRUG_META_YAML = SPQSP_PDAC_RESOURCE / "drug_metadata.yaml"
HEALTHY_STATE_YAML = SPQSP_PDAC_RESOURCE / "healthy_state.yaml"


@dataclass
class ScenarioConfig:
    """One row of the scenario ladder."""

    name: str  # short handle used in logs + filenames
    scenario_yaml: Optional[Path]  # None = no --scenario, no --evolve-to-diagnosis
    t_end_days: float  # passed to qsp_sim; scenario YAML stop_time overrides it
    needs_evolve: bool  # controls whether healthy_state.yaml is wired in
    description: str


CONFIGS: list[ScenarioConfig] = [
    ScenarioConfig(
        name="no-scenario",
        scenario_yaml=None,
        t_end_days=30.0,
        needs_evolve=False,
        description="raw ODE, native ICs, no dosing (reference)",
    ),
    ScenarioConfig(
        name="baseline",
        scenario_yaml=PDAC_BUILD_SCENARIOS / "baseline_no_treatment.yaml",
        t_end_days=1.0,  # overridden by scenario stop_time
        needs_evolve=True,
        description="evolve_to_diagnosis, no doses",
    ),
    ScenarioConfig(
        name="gvax",
        scenario_yaml=PDAC_BUILD_SCENARIOS / "gvax_neoadjuvant_zheng2022.yaml",
        t_end_days=21.0,
        needs_evolve=True,
        description="evolve + 1 GVAX bolus at day 7",
    ),
    ScenarioConfig(
        name="gvax-nivo",
        scenario_yaml=PDAC_BUILD_SCENARIOS / "gvax_nivo_neoadjuvant_zheng2022.yaml",
        t_end_days=30.0,
        needs_evolve=True,
        description="evolve + GVAX + nivolumab at day 7",
    ),
]


def run_one_scenario(
    cfg: ScenarioConfig,
    n_sims: int,
    jobs_per_chunk: int,
    cpus_per_task: int,
    memory: str,
    seed: int,
    priors_csv: Path,
    params: list[str],
    partition: Optional[str] = None,
    time_limit: str = "00:20:00",
) -> dict:
    """Submit one scenario's sweep, wait, collect timing, delete the pool."""
    # Fresh theta with a deterministic seed so all scenarios see the same
    # parameter draw — comparing wall times across configs at the same seed
    # keeps solver work identical modulo the scenario-specific init/dosing.
    theta, names = sample_params(priors_csv, params, n_sims, seed)
    tmp_csv = Path("/tmp") / f"bench_scen_{cfg.name}_params.csv"
    pd.DataFrame(theta, columns=names).to_csv(tmp_csv, index=False)

    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

    manager = HPCJobManager(verbose=False)
    manager.config.time_limit = time_limit
    # Override the default partition from credentials.yaml when the caller
    # wants to retarget (e.g. `parallel` for whole-node fan-out). Leave
    # unchanged when partition=None so the credentials value is respected.
    if partition is not None:
        manager.config.partition = partition

    # Only pass --evolve-to-diagnosis when the scenario actually asks for it;
    # a pure-ODE run shouldn't silently pay the 857-day evolve cost.
    healthy = HEALTHY_STATE_YAML if cfg.needs_evolve else None
    drug_meta = DRUG_META_YAML if cfg.scenario_yaml is not None else None

    t_submit = time.time()
    info = manager.submit_cpp_jobs(
        samples_csv=str(tmp_csv),
        num_simulations=n_sims,
        simulation_pool_id=f"bench_scen_{cfg.name}",
        t_end_days=cfg.t_end_days,
        dt_days=0.5,
        scenario=f"bench_scen_{cfg.name}",
        seed=seed,
        jobs_per_chunk=jobs_per_chunk,
        max_workers=cpus_per_task,
        cpp_cpus_per_task=cpus_per_task,
        cpp_memory=memory,
        scenario_yaml=str(cfg.scenario_yaml) if cfg.scenario_yaml else None,
        drug_metadata_yaml=str(drug_meta) if drug_meta else None,
        healthy_state_yaml=str(healthy) if healthy else None,
    )
    submit_s = time.time() - t_submit
    job_id = info.job_ids[0]

    print(f"    job_id={job_id}, tasks={info.n_jobs}, submit_s={submit_s:.1f}s. Waiting...")
    wait_for_job(job_id, info.n_jobs)
    time.sleep(5)  # let sacct catch up

    pool_dir = f"{manager.config.simulation_pool_path}/bench_scen_{cfg.name}"
    log_dir = f"{manager.config.remote_project_path}/batch_jobs/logs"
    timing = collect_timing(job_id, log_dir, pool_dir)
    timing["submit_s"] = submit_s
    timing["config_name"] = cfg.name
    timing["jobs_per_chunk"] = jobs_per_chunk

    # Pool cleanup; iterating over scenarios back-to-back would otherwise
    # accumulate GB of trajectories. Use a generous timeout: at 10k+ sims
    # the pool directory is ~15 GB and can take well over a minute for
    # the remote ssh call to return even though the rm itself is fast.
    ssh(f"rm -rf {pool_dir}", timeout=600)
    tmp_csv.unlink(missing_ok=True)

    return timing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--jobs-per-chunk", type=int, default=25)
    ap.add_argument("--cpus-per-task", type=int, default=4)
    ap.add_argument("--memory", default="4G")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--partition",
        default=None,
        help="SLURM partition override (default: credentials.yaml). "
        "Use 'parallel' for whole-node fan-out on Rockfish.",
    )
    ap.add_argument(
        "--time-limit",
        default="00:20:00",
        help="SLURM time limit per array task.",
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[c.name for c in CONFIGS],
        help="Subset of config names to run (default: all)",
    )
    args = ap.parse_args()

    configs_by_name = {c.name: c for c in CONFIGS}
    unknown = set(args.configs) - set(configs_by_name)
    if unknown:
        ap.error(f"Unknown config(s): {sorted(unknown)}. Pick from {list(configs_by_name)}.")
    selected = [configs_by_name[n] for n in args.configs]

    print("=" * 90)
    print(
        f"C++ HPC scenario benchmark: N={args.n_sims}, jpc={args.jobs_per_chunk}, "
        f"cpus/task={args.cpus_per_task}"
    )
    for cfg in selected:
        print(f"  - {cfg.name:12} {cfg.description}")
    print("=" * 90)

    results = []
    for cfg in selected:
        print(f"\n>>> Benchmarking {cfg.name}: {cfg.description}")
        timing = run_one_scenario(
            cfg=cfg,
            n_sims=args.n_sims,
            jobs_per_chunk=args.jobs_per_chunk,
            cpus_per_task=args.cpus_per_task,
            memory=args.memory,
            seed=args.seed,
            priors_csv=DEFAULT_PRIORS_CSV,
            params=DEFAULT_SAMPLED_PARAMS,
            partition=args.partition,
            time_limit=args.time_limit,
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
    print("=" * 100)
    print("SCENARIO BENCHMARK SUMMARY")
    print("=" * 100)
    cols = [
        ("config", 14),
        ("tasks", 6),
        ("wall_s", 8),
        ("sum_sim_s", 11),
        ("mean_sim/task_s", 17),
        ("parallelism", 13),
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
            f"{r['config_name']:<14}"
            f"{r['n_tasks']:<6}"
            f"{r['wall_s']:<8.1f}"
            f"{r['sum_sim_s']:<11.1f}"
            f"{r['mean_sim_s']:<17.2f}"
            f"{parallelism:<13.1f}"
            f"{ms_per_sim:<13.1f}"
            f"{str(ok):<13}"
        )
    print()


if __name__ == "__main__":
    main()
