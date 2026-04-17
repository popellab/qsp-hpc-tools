"""M13 evolve-cache end-to-end smoke test on HPC.

Submits two tiny C++ batches to the cluster under the SAME theta but
DIFFERENT scenario pool IDs, then verifies:

  1. Exactly one QSTH blob landed in ``{simulation_pool_path}/evolve_cache``
     — so the second batch reused the first batch's evolve state instead
     of re-running the ~857-day healthy-state integration.
  2. The second batch's wall time is substantially shorter than the first
     (first pays the evolve cost; second is dominated by post-diagnosis
     integration only).

Neither assertion makes strict claims about absolute speeds — Rockfish
queueing noise is too high for that. We report relative ratios and
flag if the cache clearly didn't engage.

Usage::

    python scripts/smoke_test_evolve_cache_hpc.py \\
        --healthy-state-yaml /path/to/SPQSP_PDAC/PDAC/sim/resource/healthy_state.yaml \\
        --scenario-yaml-1 /path/to/qspio-pdac/scenarios/baseline_no_treatment.yaml \\
        --drug-metadata-yaml /path/to/SPQSP_PDAC/PDAC/sim/resource/drug_metadata.yaml

The script is idempotent across runs *per test-id*: every invocation
uses a unique ``--test-id`` (default: current timestamp) for the pool-
dir prefix so retries don't collide with prior runs. The shared
evolve-cache sub-directory is intentionally *not* included in the
test-id — that's the whole point (cache is shared across runs).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def build_tiny_params_csv(priors_csv: Path, out_csv: Path, n_sims: int) -> list[str]:
    """Emit a params.csv with n_sims IDENTICAL rows (template defaults).

    Identical rows is the whole point of the test: every sim should hash
    to the same theta and share one cached evolve blob.
    """
    priors = pd.read_csv(priors_csv)
    # Use the dist_param1 as a "default" — close enough to a real theta
    # while keeping all rows identical. Any stable deterministic choice
    # works for this test since we only care that every row is the same.
    row = priors.set_index("name")["dist_param1"].to_dict()
    names = list(row.keys())
    arr = np.tile(np.array([row[n] for n in names], dtype=float), (n_sims, 1))
    pd.DataFrame(arr, columns=names).to_csv(out_csv, index=False)
    return names


def submit_and_wait(
    manager,
    params_csv: Path,
    pool_id: str,
    scenario_label: str,
    scenario_yaml: Path,
    drug_metadata_yaml: Path,
    healthy_state_yaml: Path,
    n_sims: int,
    t_end_days: float,
    dt_days: float,
    jobs_per_chunk: int,
    cpus_per_task: int,
    memory: str,
    time_limit: str,
    poll_interval: float,
    max_wait_s: float,
) -> tuple[float, str]:
    """Submit one batch, poll until every array task completes, return
    (elapsed_s, job_id)."""
    manager.config.time_limit = time_limit
    t0 = time.time()
    info = manager.submit_cpp_jobs(
        samples_csv=str(params_csv),
        num_simulations=n_sims,
        simulation_pool_id=pool_id,
        t_end_days=t_end_days,
        dt_days=dt_days,
        scenario=scenario_label,
        seed=1,
        jobs_per_chunk=jobs_per_chunk,
        max_workers=cpus_per_task,
        cpp_cpus_per_task=cpus_per_task,
        cpp_memory=memory,
        scenario_yaml=str(scenario_yaml),
        drug_metadata_yaml=str(drug_metadata_yaml),
        healthy_state_yaml=str(healthy_state_yaml),
        # Explicit: turn on the cache (this is the default but be loud).
        evolve_cache=True,
    )
    job_id = info.job_ids[0]
    print(
        f"  submitted {job_id} ({info.n_jobs} tasks) — polling every "
        f"{poll_interval}s (max {max_wait_s}s)..."
    )

    # Brief delay so the job shows up in squeue.
    time.sleep(5.0)
    start_poll = time.time()
    while True:
        try:
            status = manager.check_job_status(job_id)
        except Exception as e:
            print(f"  status check failed: {e} (retrying)")
            time.sleep(poll_interval)
            continue
        active = status.get("running", 0) + status.get("pending", 0)
        elapsed = time.time() - start_poll
        print(
            f"  [{int(elapsed // 60)}m{int(elapsed % 60):02d}s] "
            f"completed={status.get('completed', 0)} "
            f"running={status.get('running', 0)} "
            f"pending={status.get('pending', 0)} "
            f"failed={status.get('failed', 0)}"
        )
        if active == 0 and sum(status.values()) > 0:
            break
        if elapsed > max_wait_s:
            raise TimeoutError(f"Batch {job_id} did not finish within {max_wait_s}s")
        time.sleep(poll_interval)
    return time.time() - t0, job_id


def check_evolve_cache(manager, cache_root: str) -> tuple[int, list[str]]:
    """Return (blob_count, ls_output_lines) from the HPC cache dir."""
    # The subdir name = "<healthy_state_hash[:8]>_<binary_hash[:8]>";
    # there may be exactly one such subdir for this run. Use ** so we
    # don't need to predict the segmentation name.
    find_cmd = (
        f"bash -lc "
        f'\'if [ -d "{cache_root}" ]; then '
        f'find "{cache_root}" -name "*.state.bin" -type f; '
        f"fi'"
    )
    status, output = manager.transport.exec(find_cmd, timeout=30)
    if status != 0:
        raise RuntimeError(f"remote find in {cache_root} failed (status {status}): {output}")
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return len(lines), lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--priors-csv",
        type=Path,
        required=True,
        help="Path to pdac_priors.csv (columns: name, dist_param1, ...)",
    )
    ap.add_argument(
        "--healthy-state-yaml", type=Path, required=True, help="SPQSP_PDAC/.../healthy_state.yaml"
    )
    ap.add_argument(
        "--scenario-yaml-1",
        type=Path,
        required=True,
        help="First scenario YAML (e.g. baseline_no_treatment.yaml)",
    )
    ap.add_argument(
        "--scenario-yaml-2",
        type=Path,
        default=None,
        help="Optional second scenario YAML. Defaults to "
        "--scenario-yaml-1 — same theta, same scenario, "
        "different pool ID still proves the cache works.",
    )
    ap.add_argument(
        "--drug-metadata-yaml", type=Path, required=True, help="SPQSP_PDAC/.../drug_metadata.yaml"
    )
    ap.add_argument(
        "--n-sims",
        type=int,
        default=4,
        help="Sims per batch (default 4). All identical → " "they all share one cache blob.",
    )
    ap.add_argument(
        "--t-end-days",
        type=float,
        default=1.0,
        help="Post-diagnosis integration horizon (keep short "
        "so cache-speedup dominates the signal).",
    )
    ap.add_argument("--dt-days", type=float, default=0.1)
    ap.add_argument("--jobs-per-chunk", type=int, default=4)
    ap.add_argument("--cpus-per-task", type=int, default=2)
    ap.add_argument("--memory", default="4G")
    ap.add_argument("--time-limit", default="00:10:00")
    ap.add_argument("--poll-interval", type=float, default=10.0)
    ap.add_argument(
        "--max-wait-s",
        type=float,
        default=900.0,
        help="Bail if a single batch takes longer than this.",
    )
    ap.add_argument(
        "--test-id",
        default=time.strftime("%Y%m%d_%H%M%S"),
        help="Unique suffix for this run's pool dirs (default: "
        "current timestamp). Unique per invocation so "
        "repeated runs don't collide.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/m13_hpc_smoke"),
        help="Local workdir for params.csv / state files.",
    )
    ap.add_argument(
        "--clear-cache-first",
        action="store_true",
        help="Remove the remote evolve_cache directory before "
        "running. Recommended for reliable timing signal; "
        "otherwise a previous run's blob may make the "
        "first batch look fast.",
    )
    args = ap.parse_args()

    scenario_2 = args.scenario_yaml_2 or args.scenario_yaml_1

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("M13 evolve-cache HPC smoke test")
    print()
    print("NOTE: the HPC qsp_sim binary must be built from a branch that")
    print("includes the M13 --dump-state / --initial-state flags. Before")
    print("running this test:")
    print("  - push SPQSP_PDAC's m13-evolve-cache branch to origin")
    print("  - checkout that branch in the HPC SPQSP_PDAC clone")
    print("  - HPCJobManager.ensure_cpp_binary() will git-pull + rebuild")
    print()
    print(f"  test_id:             {args.test_id}")
    print(f"  n_sims per batch:    {args.n_sims}")
    print(f"  t_end_days:          {args.t_end_days}")
    print(f"  scenario 1 YAML:     {args.scenario_yaml_1}")
    print(f"  scenario 2 YAML:     {scenario_2}")
    print(f"  healthy state YAML:  {args.healthy_state_yaml}")
    print("=" * 72)

    # 1. Identical-row params CSV.
    params_csv = out_dir / f"params_{args.test_id}.csv"
    names = build_tiny_params_csv(args.priors_csv, params_csv, args.n_sims)
    print(f"\nWrote {params_csv} — {args.n_sims} identical rows × " f"{len(names)} params")

    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

    manager = HPCJobManager(verbose=True)
    cache_root = f"{manager.config.simulation_pool_path}/evolve_cache"

    if args.clear_cache_first:
        print(f"\nClearing remote cache: {cache_root}")
        status, output = manager.transport.exec(
            f"bash -lc 'rm -rf \"{cache_root}\" && echo cleared'",
            timeout=60,
        )
        print(f"  status={status} output={output.strip()}")

    # 2. Batch 1 — cache miss for this theta. Pays the evolve.
    pool_id_1 = f"m13_smoke_{args.test_id}_a"
    print(f"\n--- Batch 1 — pool={pool_id_1} (expect CACHE MISS) ---")
    t1, job1 = submit_and_wait(
        manager=manager,
        params_csv=params_csv,
        pool_id=pool_id_1,
        scenario_label=f"smoke_{args.test_id}_a",
        scenario_yaml=args.scenario_yaml_1,
        drug_metadata_yaml=args.drug_metadata_yaml,
        healthy_state_yaml=args.healthy_state_yaml,
        n_sims=args.n_sims,
        t_end_days=args.t_end_days,
        dt_days=args.dt_days,
        jobs_per_chunk=args.jobs_per_chunk,
        cpus_per_task=args.cpus_per_task,
        memory=args.memory,
        time_limit=args.time_limit,
        poll_interval=args.poll_interval,
        max_wait_s=args.max_wait_s,
    )
    n_blobs_after_1, blob_paths_1 = check_evolve_cache(manager, cache_root)
    print(f"\n  batch 1 elapsed (wall): {t1:.1f}s  job={job1}")
    print(f"  cache blobs after batch 1: {n_blobs_after_1}")
    for p in blob_paths_1:
        print(f"    {p}")

    # 3. Batch 2 — same theta, different pool_id. Should HIT the cache.
    pool_id_2 = f"m13_smoke_{args.test_id}_b"
    print(f"\n--- Batch 2 — pool={pool_id_2} (expect CACHE HIT) ---")
    t2, job2 = submit_and_wait(
        manager=manager,
        params_csv=params_csv,
        pool_id=pool_id_2,
        scenario_label=f"smoke_{args.test_id}_b",
        scenario_yaml=scenario_2,
        drug_metadata_yaml=args.drug_metadata_yaml,
        healthy_state_yaml=args.healthy_state_yaml,
        n_sims=args.n_sims,
        t_end_days=args.t_end_days,
        dt_days=args.dt_days,
        jobs_per_chunk=args.jobs_per_chunk,
        cpus_per_task=args.cpus_per_task,
        memory=args.memory,
        time_limit=args.time_limit,
        poll_interval=args.poll_interval,
        max_wait_s=args.max_wait_s,
    )
    n_blobs_after_2, blob_paths_2 = check_evolve_cache(manager, cache_root)
    print(f"\n  batch 2 elapsed (wall): {t2:.1f}s  job={job2}")
    print(f"  cache blobs after batch 2: {n_blobs_after_2}")

    # 4. Assertions.
    print("\n" + "=" * 72)
    print("Results")
    print("=" * 72)
    print(f"  batch 1 wall:  {t1:.1f}s  (cache miss — paid evolve)")
    print(f"  batch 2 wall:  {t2:.1f}s  (cache hit  — skipped evolve)")
    ratio = t2 / t1 if t1 > 0 else float("nan")
    print(f"  ratio (b2/b1): {ratio:.2f}  (lower = better)")
    print(f"  cache blobs:   {n_blobs_after_2}  (expected 1)")

    failures: list[str] = []
    if n_blobs_after_1 != 1:
        failures.append(f"expected exactly 1 blob after batch 1, got {n_blobs_after_1}")
    if n_blobs_after_2 != 1:
        failures.append(
            f"expected exactly 1 blob after batch 2 (shared with batch 1), "
            f"got {n_blobs_after_2}"
        )
    # Relaxed check on timing. Queueing noise can dominate at this size,
    # so we only flag extreme regressions (ratio > 0.9 means batch 2 was
    # nearly as slow as batch 1, suggesting cache didn't engage).
    if ratio > 0.9:
        failures.append(
            f"batch 2 wall ({t2:.1f}s) is not meaningfully shorter than "
            f"batch 1 ({t1:.1f}s) — ratio {ratio:.2f}. Likely the cache "
            f"didn't engage for batch 2. Check that healthy_state_yaml "
            f"was passed to both and evolve_cache defaulted True."
        )

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: evolve-cache engaged on HPC; batch 2 reused batch 1's blob.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
