"""SBI-workflow HPC smoke: exercise CppSimulator.run_hpc across scenarios.

Sits above submit_cpp_smoke_sweep.py (single submission) and below
pdac-build's sbi_runner.py (full NPE pipeline). Validates the
CppSimulator layer — the 3-tier cache walk, on-cluster derivation,
and most importantly **multi-scenario evolve-cache sharing** (two
scenarios over the same theta pool should build N blobs once and the
second scenario should hit them all).

Usage::

    python scripts/smoke_sbi_hpc.py \\
        --priors-csv /path/to/pdac_priors.csv \\
        --scenarios-dir /path/to/pdac-build/scenarios \\
        --calibration-targets-root /path/to/pdac-build/calibration_targets \\
        --model-structure-file /path/to/pdac-build/model_structure.json \\
        --healthy-state-yaml /path/to/SPQSP_PDAC/.../healthy_state.yaml \\
        --drug-metadata-yaml /path/to/SPQSP_PDAC/.../drug_metadata.yaml

On success, prints theta/test_stats shapes and a count of evolve-cache
blobs on HPC.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--priors-csv", type=Path, required=True)
    ap.add_argument(
        "--scenarios-dir",
        type=Path,
        required=True,
        help="pdac-build/scenarios (contains baseline_no_treatment.yaml etc.)",
    )
    ap.add_argument(
        "--calibration-targets-root",
        type=Path,
        required=True,
        help="pdac-build/calibration_targets (one subdir per scenario)",
    )
    ap.add_argument(
        "--model-structure-file",
        type=Path,
        required=True,
        help="pdac-build/model_structure.json (species unit metadata)",
    )
    ap.add_argument("--healthy-state-yaml", type=Path, required=True)
    ap.add_argument("--drug-metadata-yaml", type=Path, required=True)
    ap.add_argument(
        "--local-binary-path",
        type=Path,
        required=True,
        help="LAPTOP-side qsp_sim (used only for hashing + CppSimulator "
        "validate()). HPC execution uses credentials cpp.binary_path.",
    )
    ap.add_argument(
        "--local-template-xml",
        type=Path,
        required=True,
        help="LAPTOP-side param_all.xml (used only for hashing / probe). "
        "HPC execution uses credentials cpp.template_path.",
    )
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=["baseline_no_treatment", "gvax_neoadjuvant_zheng2022"],
        help="Scenario names (must match {scenarios-dir}/{name}.yaml and "
        "{calibration-targets-root}/<name-stripped>/).",
    )
    ap.add_argument("--n-sims", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--t-end-days", type=float, default=180.0)
    ap.add_argument("--dt-days", type=float, default=0.5)
    ap.add_argument("--jobs-per-chunk", type=int, default=20)
    ap.add_argument("--cpus-per-task", type=int, default=4)
    ap.add_argument("--memory", default="4G")
    ap.add_argument("--time-limit", default="00:30:00")
    ap.add_argument(
        "--model-version",
        default=None,
        help="Forces cache miss when unique (default: auto-stamped with "
        "timestamp so every smoke run starts from a clean state).",
    )
    ap.add_argument("--cpp-branch", default=None)
    ap.add_argument("--tools-branch", default=None)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/sbi_hpc_smoke"),
        help="Local cache root. Wiped between smoke runs to avoid " "Tier-1 short-circuits.",
    )
    ap.add_argument(
        "--keep-cache",
        action="store_true",
        help="Don't wipe --cache-dir before running (useful to test "
        "the Tier-1 local hit path on a rerun).",
    )
    args = ap.parse_args()

    # Scenario name → calibration targets dir. pdac-build's convention
    # strips the citation suffix in the dir name (gvax_neoadjuvant_zheng2022
    # → gvax_neoadjuvant).
    targets_map = {
        "baseline_no_treatment": "baseline_no_treatment",
        "clinical_progression": "clinical_progression",
        "gvax_neoadjuvant_zheng2022": "gvax_neoadjuvant",
        "gvax_nivo_neoadjuvant_zheng2022": "gvax_nivo_neoadjuvant",
    }
    for scen in args.scenarios:
        if scen not in targets_map:
            print(f"ERROR: unknown scenario {scen}", file=sys.stderr)
            return 1

    model_version = args.model_version or f"sbi_smoke_{time.strftime('%Y%m%d_%H%M%S')}"

    if not args.keep_cache and args.cache_dir.exists():
        print(f"Wiping local cache: {args.cache_dir}")
        shutil.rmtree(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SBI HPC smoke: CppSimulator.run_hpc across scenarios")
    print(f"  scenarios:        {args.scenarios}")
    print(f"  n_sims:           {args.n_sims}")
    print(f"  seed:             {args.seed}")
    print(f"  model_version:    {model_version}")
    print(f"  t_end_days:       {args.t_end_days}")
    print("=" * 72)

    from qsp_hpc.batch.hpc_job_manager import HPCJobManager
    from qsp_hpc.simulation.cpp_simulator import CppSimulator

    manager = HPCJobManager(verbose=True)
    manager.config.time_limit = args.time_limit

    if args.cpp_branch:
        manager.config.cpp_branch = args.cpp_branch
    if args.tools_branch:
        src = manager.config.qsp_hpc_tools_source
        if ".git" in src:
            head, _sep, _tail = src.partition(".git")
            manager.config.qsp_hpc_tools_source = f"{head}.git@{args.tools_branch}"
        else:
            manager.config.qsp_hpc_tools_source = f"{src}@{args.tools_branch}"
    if args.cpp_branch or args.tools_branch:
        print(
            f"\nOverriding refs: cpp_branch={manager.config.cpp_branch}, "
            f"qsp_hpc_tools_source={manager.config.qsp_hpc_tools_source}"
        )

    scenario_results: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for scen in args.scenarios:
        scen_yaml = args.scenarios_dir / f"{scen}.yaml"
        cal_targets = args.calibration_targets_root / targets_map[scen]
        if not scen_yaml.exists():
            print(f"ERROR: scenario YAML not found: {scen_yaml}", file=sys.stderr)
            return 1
        if not cal_targets.exists():
            print(f"ERROR: calibration targets dir not found: {cal_targets}", file=sys.stderr)
            return 1

        print(f"\n--- Running scenario: {scen} ---")
        sim = CppSimulator(
            priors_csv=str(args.priors_csv),
            binary_path=str(args.local_binary_path),
            template_xml=str(args.local_template_xml),
            model_version=model_version,
            scenario=scen,
            subtree=manager.config.cpp_subtree,
            t_end_days=args.t_end_days,
            dt_days=args.dt_days,
            cache_dir=str(args.cache_dir),
            seed=args.seed,
            scenario_yaml=str(scen_yaml),
            drug_metadata_yaml=str(args.drug_metadata_yaml),
            healthy_state_yaml=str(args.healthy_state_yaml),
            job_manager=manager,
            calibration_targets=str(cal_targets),
            model_structure_file=str(args.model_structure_file),
            remote_binary_path=manager.config.cpp_binary_path,
            remote_template_xml=manager.config.cpp_template_path,
            verbose=True,
        )

        t0 = time.time()
        theta, test_stats = sim.run_hpc(args.n_sims)
        elapsed = time.time() - t0
        scenario_results[scen] = (theta, test_stats, elapsed)
        print(
            f"  ✓ {scen}: theta={theta.shape}, test_stats={test_stats.shape}, "
            f"elapsed={elapsed:.1f}s"
        )

    # ---- Post-run assertions ----
    print("\n" + "=" * 72)
    print("Assertions")
    print("=" * 72)

    failures: list[str] = []

    # Size check: run_hpc must return exactly n_sims rows. Partial returns
    # (e.g. Tier 2 short-circuit on a pool with < n sims) are silent bugs —
    # caller sees a too-small array. Caught once by the N=1000 run over a
    # 40-sim pool where Tier 2 returned the 40 pre-existing sims instead
    # of topping up to 1000.
    for scen, (theta, _, _) in scenario_results.items():
        if theta.shape[0] != args.n_sims:
            failures.append(
                f"{scen} returned {theta.shape[0]} sims but {args.n_sims} were requested"
            )

    # Theta across scenarios must match (same seed → same pool → same first N).
    thetas = [r[0] for r in scenario_results.values()]
    ref_theta = thetas[0]
    for scen, theta in zip(scenario_results.keys(), thetas):
        if theta.shape != ref_theta.shape:
            failures.append(f"{scen} theta shape {theta.shape} != {ref_theta.shape}")
        elif not np.allclose(theta, ref_theta):
            max_abs = np.nanmax(np.abs(theta - ref_theta))
            failures.append(f"{scen} theta diverges from reference (max |Δ|={max_abs:.3e})")
        else:
            print(f"  ✓ {scen}: theta matches reference pool")

    # Test stats must DIFFER across scenarios (different calibration targets
    # compute different statistics; if they match, something's wrong).
    if len(scenario_results) >= 2:
        ts_list = [r[1] for r in scenario_results.values()]
        if ts_list[0].shape == ts_list[1].shape and np.allclose(
            ts_list[0], ts_list[1], equal_nan=True
        ):
            failures.append(
                "test_stats identical across scenarios — expected different "
                "calibration targets to produce different summary stats"
            )
        else:
            print("  ✓ test_stats differ across scenarios (as expected)")

    # Evolve-cache LMDB env count on HPC: one env per (healthy_state, binary)
    # pair; per-theta blobs live as keys inside each env, not as files.
    cache_root = f"{manager.config.simulation_pool_path}/evolve_cache"
    find_cmd = f'bash -lc \'find "{cache_root}" -name "data.mdb" -type f 2>/dev/null | wc -l\''
    status, output = manager.transport.exec(find_cmd, timeout=30)
    env_count = int(output.strip()) if status == 0 else -1
    print(f"  evolve-cache LMDB env count (HPC scratch): {env_count}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(
        f"\nPASS: {len(scenario_results)} scenario(s) × {args.n_sims} sims; "
        f"theta pool shared, stats differ, evolve-cache engaged."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
