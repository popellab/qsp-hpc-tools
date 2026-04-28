#!/usr/bin/env python3
"""
Cross-Scenario Test Statistics Composer.

Computes cross-scenario observables from already-aligned per-scenario
simulation outputs. Designed to run *after* per-scenario
``compute_test_statistics_batch`` has produced each scenario's
test-statistics matrix; consumes the resulting ``scenario_meta`` dict
plus a ``cross_scenario_df`` produced by
``qsp_hpc.calibration.cross_scenario_loader.load_cross_scenario_targets``.

Two input kinds per cross-scenario target are supported:

- ``input_kind='test_statistic'``: the composer receives a precomputed
  scalar Pint Quantity (per-scenario units) for that input.
- ``input_kind='timeseries'``: the composer receives a dict mapping
  ``'time'`` and each entry of ``required_species`` to a Pint Quantity,
  matching the species_dict shape passed to per-scenario observables in
  ``compute_test_statistics_batch``.

The composer output is a scalar Pint Quantity; one column per
cross-scenario target is appended to the output matrix.

Public API:
    build_cross_scenario_registry(cross_scenario_df) -> dict
    compute_cross_scenario_statistics(scenario_meta, cross_scenario_df,
        species_units, registry=None, template_defaults=None) -> np.ndarray
"""

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pint

from qsp_hpc.utils.logging_config import setup_logger
from qsp_hpc.utils.unit_registry import ureg

logger = setup_logger(__name__, verbose=True)


def build_cross_scenario_registry(cross_scenario_df: pd.DataFrame) -> dict:
    """
    Compile each cross-scenario target's composer code into a callable.

    Each row in the DataFrame must define:
      - cross_scenario_target_id: unique identifier
      - composer_code: Python source defining ``compute(inputs, ureg) -> Pint Quantity``

    The compiled function lives in an isolated namespace with numpy + pint
    pre-imported so common idioms (np.where, np.percentile, ureg.dimensionless)
    work without per-target imports. Failures during compilation raise
    immediately — there is no silent registry miss path here, unlike the
    per-scenario ``build_test_stat_registry`` which tolerates broken rows.
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
        # per-scenario test-stat code in derive_test_stats_worker.
        namespace: dict = {"np": np, "numpy": np, "pint": pint, "ureg": ureg}
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


def _parse_unit(species_name: str, species_units: dict):
    """Mirror of derive_test_stats_worker._parse_unit so timeseries input
    species get the same unit treatment as per-scenario observables."""
    unit_info = species_units.get(species_name, "dimensionless")
    unit_str = unit_info["units"] if isinstance(unit_info, dict) else unit_info
    return ureg.parse_expression(unit_str)


def _build_timeseries_species_dict(
    sim_row: pd.Series,
    required_species: list[str],
    species_units: dict,
    template_defaults: dict[str, float],
    time_unit,
) -> dict[str, Any] | None:
    """Build the per-scenario species_dict the composer consumes for one
    timeseries input. Returns None if any required species is unresolvable
    (which fails the cross-scenario target NaN for that row).

    Resolution strategy mirrors compute_test_statistics_batch's species_plan:
    series column → param column → template default → missing.
    """
    out: dict[str, Any] = {}

    time_val = sim_row.get("time")
    if time_val is None:
        return None
    out["time"] = np.asarray(time_val) * time_unit

    for s in required_species:
        if s in sim_row.index:
            val = sim_row[s]
            unit = _parse_unit(s, species_units)
            if isinstance(val, (int, float, np.integer, np.floating)):
                out[s] = float(val) * unit
            else:
                out[s] = np.asarray(val) * unit
        elif f"param:{s}" in sim_row.index:
            unit = _parse_unit(s, species_units)
            out[s] = float(sim_row[f"param:{s}"]) * unit
        elif s in template_defaults:
            unit = _parse_unit(s, species_units)
            out[s] = float(template_defaults[s]) * unit
        else:
            return None
    return out


def compute_cross_scenario_statistics(
    scenario_meta: Mapping[str, Mapping[str, Any]],
    cross_scenario_df: pd.DataFrame,
    species_units: dict,
    registry: dict | None = None,
    template_defaults: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Compute cross-scenario test statistics from aligned per-scenario outputs.

    Args:
        scenario_meta: Dict keyed by scenario name. Each value must
            provide:
              - 'sim_df': pd.DataFrame (for timeseries inputs); rows are
                aligned by sample_index across scenarios (caller's
                responsibility — sbi_runner.py:688-720 handles this for
                production runs)
              - 'x_raw': np.ndarray of shape (n_sims, n_per_scen_test_stats)
                (for test_statistic inputs)
              - 'observable_names': list[str] giving the column order of
                'x_raw' (lookup target by test_statistic_id)
            'sim_df' may be omitted for scenarios that only contribute
            test_statistic inputs; 'x_raw' may be omitted for scenarios
            that only contribute timeseries inputs.
        cross_scenario_df: DataFrame from
            ``load_cross_scenario_targets``.
        species_units: Dict mapping species names to unit strings.
        registry: Optional precompiled composer registry (from
            ``build_cross_scenario_registry``). Built on demand if None.
        template_defaults: Optional ``{name: default}`` map for
            timeseries-input species not present as parquet columns
            (parallel to compute_test_statistics_batch's argument).

    Returns:
        np.ndarray of shape (n_sims, n_cross_targets). Rows where any
        input fails to resolve (missing species, missing test_statistic_id,
        composer exception) are NaN for that target.
    """
    template_defaults = template_defaults or {}
    if registry is None:
        registry = build_cross_scenario_registry(cross_scenario_df)

    n_cross = len(cross_scenario_df)

    # Determine n_sims from the first scenario that has data. Both x_raw
    # and sim_df share the same row count by the time this runs.
    n_sims = None
    for meta in scenario_meta.values():
        if "x_raw" in meta and meta["x_raw"] is not None:
            n_sims = int(meta["x_raw"].shape[0])
            break
        if "sim_df" in meta and meta["sim_df"] is not None:
            n_sims = int(len(meta["sim_df"]))
            break
    if n_sims is None:
        raise ValueError(
            "Could not determine n_sims from scenario_meta — at least one "
            "scenario must provide 'x_raw' or 'sim_df'."
        )

    out = np.full((n_sims, n_cross), np.nan, dtype=float)

    # Plan phase — per-target resolution metadata so we don't reparse
    # inputs_json or look up scenarios per (target × sim).
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
            entry = {"role": inp["role"], "scenario": scen_name, "input_kind": inp["input_kind"]}
            if inp["input_kind"] == "test_statistic":
                tsid = inp["test_statistic_id"]
                obs_names = scenario_meta[scen_name].get("observable_names") or []
                if tsid not in obs_names:
                    raise ValueError(
                        f"Cross-scenario target '{target_id}' input role='{inp['role']}' "
                        f"references test_statistic_id='{tsid}' in scenario "
                        f"'{scen_name}', but that test stat is not produced by "
                        f"the scenario. Available: {obs_names[:8]}{'...' if len(obs_names) > 8 else ''}"
                    )
                entry["col_idx"] = obs_names.index(tsid)
                # Pre-parse the unit once per target so the per-row hot
                # path doesn't reparse 'cell/mm**2' style strings every sim.
                entry["unit"] = _parse_unit(tsid, species_units)
            else:
                entry["required_species"] = list(inp["required_species"])
            plan["inputs"].append(entry)
        plans.append(plan)

    time_unit = ureg.day

    # Cache references to per-scenario sim_dfs (timeseries inputs) and
    # x_raw matrices (scalar inputs) so the per-row dispatch is just
    # indexing.
    scen_sim_dfs: dict[str, pd.DataFrame] = {}
    scen_x_raw: dict[str, np.ndarray] = {}
    for scen_name, meta in scenario_meta.items():
        if meta.get("sim_df") is not None:
            scen_sim_dfs[scen_name] = meta["sim_df"]
        if meta.get("x_raw") is not None:
            scen_x_raw[scen_name] = meta["x_raw"]

    for i in range(n_sims):
        for plan in plans:
            inputs_dict: dict[str, Any] = {}
            failed = False
            for entry in plan["inputs"]:
                scen_name = entry["scenario"]
                if entry["input_kind"] == "test_statistic":
                    x_raw = scen_x_raw.get(scen_name)
                    if x_raw is None:
                        failed = True
                        break
                    val = x_raw[i, entry["col_idx"]]
                    if not np.isfinite(val):
                        failed = True
                        break
                    inputs_dict[entry["role"]] = float(val) * entry["unit"]
                else:
                    sim_df = scen_sim_dfs.get(scen_name)
                    if sim_df is None:
                        failed = True
                        break
                    species_dict = _build_timeseries_species_dict(
                        sim_df.iloc[i],
                        entry["required_species"],
                        species_units,
                        template_defaults,
                        time_unit,
                    )
                    if species_dict is None:
                        failed = True
                        break
                    inputs_dict[entry["role"]] = species_dict

            if failed:
                continue

            try:
                result = plan["func"](inputs_dict, ureg)
                if hasattr(result, "magnitude"):
                    out[i, plan["j"]] = float(result.magnitude)
                else:
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
