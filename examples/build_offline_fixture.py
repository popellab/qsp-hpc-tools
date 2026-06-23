#!/usr/bin/env python3
"""Generate the offline example pool for ``feature_walkthrough.ipynb``.

The walkthrough demonstrates ``CppSimulator`` with **no HPC cluster, no MATLAB,
and no compiled ``qsp_sim`` binary**. It works because ``CppSimulator`` checks a
local pool before running anything: if the pool already holds enough
simulations, ``sim(n)`` returns them and the binary is never invoked.

This script bakes that pool. For each scenario it constructs the *real*
``CppSimulator`` (pointed at the committed stub binary + template under
``fixtures/``), so the pool directory name is the exact content hash the
notebook will recompute. It then writes one ``batch_*.parquet`` of synthetic —
but schema-faithful — results into that directory. The values are illustrative
fixture data, not a real simulation; the point is to exercise the genuine
cache-read path offline.

Re-run after changing the priors, the stub, or the template:

    python examples/build_offline_fixture.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qsp_hpc.simulation.cpp_simulator import CppSimulator

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
CACHE_DIR = HERE / "cache"
PRIORS = HERE / "data" / "priors.csv"

N_SIMS = 200
SCENARIOS = ["control", "treatment"]


def _make_simulator(scenario: str) -> CppSimulator:
    """Construct the same CppSimulator the notebook builds for ``scenario``.

    scenario_yaml / healthy_state are intentionally omitted: the offline demo
    distinguishes scenarios only by the pool-dir suffix (independent caches for
    the same parameter draws), and leaving the evolve inputs unset keeps the
    config hash a pure function of the committed priors + stub + template.
    """
    return CppSimulator(
        priors_csv=PRIORS,
        binary_path=FIXTURES / "qsp_sim_stub",
        template_xml=FIXTURES / "template_param.xml",
        calibration_targets=HERE / "data" / "calibration_targets" / scenario,
        model_version="v1",
        scenario=scenario,
        cache_dir=CACHE_DIR,
        seed=2025,
        evolve_cache=False,
    )


def _synthetic_outputs(theta: np.ndarray, param_names: list[str], scenario: str) -> dict:
    """Deterministic, illustrative per-patient outputs as a function of theta.

    Treatment lowers tumour burden and delays progression relative to control,
    so the two scenarios differ in outputs while sharing identical parameter
    draws — the multi-scenario alignment the walkthrough highlights.
    """
    idx = {name: i for i, name in enumerate(param_names)}
    growth = theta[:, idx["k_tumor_growth"]]
    death = theta[:, idx["k_tumor_death"]]
    immune = theta[:, idx["k_immune_kill"]]
    drug = theta[:, idx["k_drug_effect"]]
    capacity = theta[:, idx["V_tumor_carrying_capacity"]]

    # Treatment adds drug-driven killing on top of immune killing, so tumour
    # burden falls and progression is delayed relative to the untreated arm.
    kill = immune + (0.45 * drug if scenario == "treatment" else np.zeros_like(drug))
    net_growth = growth - death - kill

    tumor_volume_day60 = capacity * np.exp(-3.0 * kill) * np.exp(1.5 * growth)
    immune_infiltrate_peak = 1e3 * (immune + 0.5 * kill) / (death + 1e-3)
    time_to_progression = np.clip(20.0 - 80.0 * net_growth, 5.0, 120.0)

    return {
        "tumor_volume_day60": tumor_volume_day60.astype("float64"),
        "immune_infiltrate_peak": immune_infiltrate_peak.astype("float64"),
        "time_to_progression": time_to_progression.astype("float64"),
    }


def build_pool(scenario: str) -> Path:
    sim = _make_simulator(scenario)
    indices = np.arange(N_SIMS, dtype=np.int64)
    theta = sim._generate_parameters(indices)  # deterministic theta pool draw

    columns: dict[str, object] = {
        "sample_index": indices,
        "simulation_id": np.arange(N_SIMS, dtype=np.int64),
        "status": np.zeros(N_SIMS, dtype=np.int64),  # 0 = success
        "time": np.full(N_SIMS, 60.0, dtype="float64"),
    }
    for i, name in enumerate(sim.param_names):
        columns[f"param:{name}"] = theta[:, i].astype("float64")
    columns.update(_synthetic_outputs(theta, sim.param_names, scenario))

    table = pa.table(columns)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = sim.pool_dir / f"batch_{ts}_{scenario}_seed{sim.seed}.parquet"
    pq.write_table(table, out)
    print(f"  {scenario:>9}: wrote {N_SIMS} sims -> {out.relative_to(HERE)}")
    return sim.pool_dir


N_POSTERIOR = 50


def build_ppc_pool(sim: CppSimulator, theta_posterior: np.ndarray) -> Path:
    """Bake the posterior-predictive suffix-pool cache for ``theta_posterior``.

    Uses the real ``_resolve_ppc_context`` so the cache path and key hash are
    exactly what ``simulate_with_parameters`` will recompute. Writes a
    schema-faithful ``test_stats.parquet`` (``param:<name>`` + ``ts:<target_id>``
    columns) so the call resolves to a cache hit instead of running the binary.
    """
    ctx = sim._resolve_ppc_context(
        theta_posterior,
        backend="local",
        prediction_targets=None,
        pool_suffix="posterior_predictive",
        aux_by_sample_index=None,
        auxiliary_units=None,
    )
    sorted_names = sorted(sim.param_names)
    n = theta_posterior.shape[0]
    cols: dict[str, object] = {
        "sample_index": np.arange(n, dtype=np.int64),
        "status": np.zeros(n, dtype=np.int64),
    }
    # theta_posterior columns are in sorted-name order (as returned by sim(n));
    # map each back to its param name for the param:<name> columns.
    for k, name in enumerate(sorted_names):
        cols[f"param:{name}"] = theta_posterior[:, k].astype("float64")
    # One ts:<target_id> column per calibration target, illustrative values
    # derived from theta so posterior-predictive outputs look plausible.
    outputs = _synthetic_outputs(theta_posterior, sorted_names, sim.scenario)
    target_ids = list(ctx.test_stats_df["test_statistic_id"])
    rng = np.random.default_rng(0)
    for tid in target_ids:
        base = outputs.get(tid)
        if base is None:
            base = 1.0 + rng.random(n)
        cols[f"ts:{tid}"] = np.asarray(base, dtype="float64")

    table = pa.table(cols)
    pq.write_table(table, str(ctx.cache_path))
    print(f"  ppc({sim.scenario}): wrote {n} sims -> {ctx.cache_path.relative_to(HERE)}")
    return ctx.cache_path


def main() -> None:
    print(f"Building offline example pool under {CACHE_DIR.relative_to(HERE)}/ ...")
    for scenario in SCENARIOS:
        build_pool(scenario)

    # Posterior-predictive cache for the control arm, keyed on the exact theta
    # slice the notebook passes (first N_POSTERIOR rows of sim(N_SIMS)).
    control = _make_simulator("control")
    theta_full, _ = control(N_SIMS)
    theta_posterior = np.ascontiguousarray(theta_full[:N_POSTERIOR])
    build_ppc_pool(control, theta_posterior)

    # Verify a fresh simulator gets a cache hit and aligned theta.
    print("Verifying cache hits ...")
    thetas = {}
    for scenario in SCENARIOS:
        sim = _make_simulator(scenario)
        assert sim.get_available_simulations() >= N_SIMS, scenario
        theta, table = sim(N_SIMS)
        thetas[scenario] = theta
        print(f"  {scenario:>9}: cache hit, theta={theta.shape}, cols={len(table.column_names)}")
    assert np.array_equal(
        thetas["control"], thetas["treatment"]
    ), "theta must align across scenarios"

    sim = _make_simulator("control")
    theta_full, _ = sim(N_SIMS)
    x_post = sim.simulate_with_parameters(np.ascontiguousarray(theta_full[:N_POSTERIOR]))
    print(f"  ppc(control): cache hit, returned {x_post[1].num_rows} rows")
    print("OK: cache hits resolve, theta aligns across scenarios, PPC cache hits.")


if __name__ == "__main__":
    main()
