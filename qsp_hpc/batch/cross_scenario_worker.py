#!/usr/bin/env python3
"""
Cross-Scenario Test Statistics Composer.

Computes cross-scenario observables from already-aligned per-scenario
cross-input matrices. Runs *after* each scenario's per-arm cross inputs have
been derived (``cross_scenario_derive.compute_cross_inputs_batch``) and
gathered; consumes the resulting ``scenario_meta`` dict plus a
``cross_scenario_df`` produced by
``qsp_hpc.calibration.cross_scenario_loader.load_cross_scenario_targets``.

Each cross-scenario target's inputs are self-contained per-arm scalars (one
per (target, role), keyed ``f"{target}::{role}"``). The composer is a pure,
pintless reduction over those raw scalars — ``compute(inputs) -> float``,
where ``inputs`` is a role-keyed dict of raw floats. One column per
cross-scenario target is appended to the output matrix.

Public API:
    build_cross_scenario_registry(cross_scenario_df) -> dict
    compute_cross_scenario_statistics(scenario_meta, cross_scenario_df,
        registry=None) -> np.ndarray
"""

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from qsp_hpc.batch.cross_scenario_derive import cross_input_column_key
from qsp_hpc.utils.logging_config import setup_logger

logger = setup_logger(__name__, verbose=True)


def build_cross_scenario_registry(cross_scenario_df: pd.DataFrame) -> dict:
    """
    Compile each cross-scenario target's reduction code into a callable.

    Each row in the DataFrame must define:
      - cross_scenario_target_id: unique identifier
      - composer_code: Python source defining ``compute(inputs) -> float``

    The compiled function lives in an isolated namespace with numpy
    pre-imported so common idioms (np.where, np.percentile) work without
    per-target imports. The runtime is pintless: ``inputs[role]`` is a raw
    float, the reduction returns a raw float, and any unit arithmetic is done
    numerically inline. Failures during compilation raise immediately.
    """
    if "composer_code" not in cross_scenario_df.columns:
        raise ValueError("Cross-scenario DataFrame missing required 'composer_code' column.")

    registry: dict = {}
    for _, row in cross_scenario_df.iterrows():
        target_id = row["cross_scenario_target_id"]
        code = row["composer_code"]
        if pd.isna(code):
            raise ValueError(f"Cross-scenario target '{target_id}' has empty composer_code.")

        # Trusted user-authored YAML content — same security posture as
        # per-scenario test-stat code in test_stats_compute.
        namespace: dict = {"np": np, "numpy": np}
        try:
            exec(code, namespace)  # noqa: S102
        except Exception as e:
            logger.error(f"Failed to compile composer for '{target_id}': {e}")
            logger.error(f"Composer code:\n{code}")
            raise

        if "compute" not in namespace:
            raise ValueError(
                f"Cross-scenario target '{target_id}': composer_code must "
                "define a function named 'compute'."
            )
        registry[target_id] = namespace["compute"]

    logger.info(f"Built cross-scenario registry with {len(registry)} composers")
    return registry


def compute_cross_scenario_statistics(
    scenario_meta: Mapping[str, Mapping[str, Any]],
    cross_scenario_df: pd.DataFrame,
    registry: dict | None = None,
) -> np.ndarray:
    """
    Compute cross-scenario test statistics from aligned per-arm cross inputs.

    Args:
        scenario_meta: Dict keyed by scenario name. Each value referenced by a
            cross target must provide:
              - 'cross_inputs': dict mapping cross-input column key
                (``f"{target}::{role}"``) -> np.ndarray of shape (n_sims,),
                row-aligned across scenarios by sample_index (caller's
                responsibility — the SBI runner intersects scenarios to a
                shared sample set before this runs).
            Scenarios not referenced by any cross target need not provide it.
        cross_scenario_df: DataFrame from ``load_cross_scenario_targets``.
        registry: Optional precompiled composer registry (from
            ``build_cross_scenario_registry``). Built on demand if None.

    Returns:
        np.ndarray of shape (n_sims, n_cross_targets). Rows where any input is
        non-finite (NaN-propagated from a per-arm derive miss) or the reduction
        raises are NaN for that target.
    """
    if registry is None:
        registry = build_cross_scenario_registry(cross_scenario_df)

    n_cross = len(cross_scenario_df)

    # Cache each scenario's cross-input dict (key -> 1d array).
    scen_cross_inputs: dict[str, Mapping[str, np.ndarray]] = {}
    for scen_name, meta in scenario_meta.items():
        ci = meta.get("cross_inputs")
        if ci is not None:
            scen_cross_inputs[scen_name] = ci

    # Determine n_sims from the first available cross-input array.
    n_sims = None
    for ci in scen_cross_inputs.values():
        for arr in ci.values():
            n_sims = int(len(arr))
            break
        if n_sims is not None:
            break
    if n_sims is None:
        raise ValueError(
            "Could not determine n_sims from scenario_meta — at least one "
            "scenario must provide a non-empty 'cross_inputs' dict."
        )

    out = np.full((n_sims, n_cross), np.nan, dtype=float)

    # Plan phase — per-target resolution metadata so we don't reparse
    # inputs_json or look up column keys per (target × sim).
    plans: list[dict] = []
    for j, row in cross_scenario_df.iterrows():
        target_id = row["cross_scenario_target_id"]
        try:
            inputs_spec = json.loads(row["inputs_json"])
        except Exception as e:
            raise ValueError(f"Cross-scenario target '{target_id}' has invalid inputs_json: {e}")

        plan = {"j": j, "target_id": target_id, "func": registry[target_id], "inputs": []}
        for inp in inputs_spec:
            scen_name = inp["scenario"]
            if scen_name not in scenario_meta:
                raise ValueError(
                    f"Cross-scenario target '{target_id}' references "
                    f"scenario '{scen_name}' which is not in scenario_meta. "
                    f"Available: {sorted(scenario_meta.keys())}"
                )
            ci = scen_cross_inputs.get(scen_name)
            if ci is None:
                raise ValueError(
                    f"Cross-scenario target '{target_id}' input role='{inp['role']}' "
                    f"references scenario '{scen_name}', but that scenario has no "
                    "'cross_inputs' in scenario_meta (its per-arm inputs were not "
                    "derived)."
                )
            col_key = cross_input_column_key(target_id, inp["role"])
            if col_key not in ci:
                raise ValueError(
                    f"Cross-scenario target '{target_id}' input role='{inp['role']}' "
                    f"expects cross-input column '{col_key}' in scenario "
                    f"'{scen_name}', but it is missing. "
                    f"Available: {sorted(ci.keys())[:8]}{'...' if len(ci) > 8 else ''}"
                )
            plan["inputs"].append(
                {"role": inp["role"], "scenario": scen_name, "column": ci[col_key]}
            )
        plans.append(plan)

    for i in range(n_sims):
        for plan in plans:
            inputs_dict: dict[str, float] = {}
            failed = False
            for entry in plan["inputs"]:
                val = entry["column"][i]
                if not np.isfinite(val):
                    failed = True
                    break
                inputs_dict[entry["role"]] = float(val)

            if failed:
                continue

            try:
                result = plan["func"](inputs_dict)
                out[i, plan["j"]] = float(result)
            except Exception as e:
                # Match the per-scenario worker's logging volume so
                # systematic composer bugs are visible without burying
                # the rest of the run.
                logger.warning(
                    f"Error computing cross-scenario '{plan['target_id']}' "
                    f"for simulation {i}: {e}"
                )

    n_computed = int(np.sum(~np.isnan(out)))
    n_total = out.size
    logger.info(
        f"Computed {n_computed}/{n_total} cross-scenario values "
        f"({100 * n_computed / max(n_total, 1):.1f}%)"
    )
    return out
