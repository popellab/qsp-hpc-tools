#!/usr/bin/env python3
"""
Derive-time computation of cross-scenario per-arm inputs.

A cross-scenario target's per-arm input is just a scalar-from-trajectory
observable (``compute_test_statistic(time, species_dict) -> float``). This
module reshapes the per-arm inputs of every cross target that references a
given scenario into a ``test_stats_df``-style frame, so the existing
``build_test_stat_registry`` / ``compute_test_statistics_batch`` machinery
computes them in the same row-group pass as the regular test stats — while the
trajectory is still in hand, before ``discard_trajectories`` drops it.

The cross-arm reduction (the C/B ratio) is NOT done here. The derive worker is
scenario-local: each arm is computed by a separate worker invocation, so no
single worker sees two arms at one theta. The reduction runs downstream in
``cross_scenario_worker.compute_cross_scenario_statistics`` once every
scenario's matrix is gathered and aligned on ``sample_index``.

Column key convention: ``f"{cross_scenario_target_id}::{role}"`` — unique per
(target, role) and ties each per-arm column to the input that defined it.

Public API:
    cross_input_column_key(target_id, role) -> str
    build_cross_input_test_stats_df(cross_scenario_df, scenario_name) -> DataFrame | None
    compute_cross_inputs_batch(sim_df, cross_scenario_df, scenario_name,
        species_units, ...) -> (np.ndarray | None, list[str])
"""

import json

import numpy as np
import pandas as pd

from qsp_hpc.batch.test_stats_compute import (
    build_test_stat_registry,
    compute_test_statistics_batch,
)

CROSS_INPUT_KEY_SEP = "::"


def cross_input_column_key(target_id: str, role: str) -> str:
    """Stable, globally-unique column key for one per-arm cross input.

    Roles are unique within a target (schema-enforced) and the target_id
    prefix disambiguates the same role name reused across targets, so the
    key never collides within a scenario.
    """
    return f"{target_id}{CROSS_INPUT_KEY_SEP}{role}"


def build_cross_input_test_stats_df(
    cross_scenario_df: pd.DataFrame, scenario_name: str
) -> pd.DataFrame | None:
    """Reshape every per-arm input that references ``scenario_name`` into a
    ``test_stats_df``-style frame.

    Columns match what ``build_test_stat_registry`` /
    ``compute_test_statistics_batch`` expect:
      - ``test_statistic_id``: the cross-input column key
        (``{target}::{role}``)
      - ``required_species``: comma-joined species list
      - ``model_output_code``: the input's ``observable_code``

    Returns ``None`` when no cross target references this scenario (so callers
    can skip the extra derive pass cheaply).
    """
    rows: list[dict] = []
    for _, row in cross_scenario_df.iterrows():
        target_id = row["cross_scenario_target_id"]
        try:
            inputs = json.loads(row["inputs_json"])
        except Exception as e:  # pragma: no cover — loader guarantees valid JSON
            raise ValueError(f"Cross-scenario target '{target_id}' has invalid inputs_json: {e}")
        for inp in inputs:
            if inp["scenario"] != scenario_name:
                continue
            rows.append(
                {
                    "test_statistic_id": cross_input_column_key(target_id, inp["role"]),
                    "required_species": ",".join(inp["required_species"]),
                    "model_output_code": inp["observable_code"],
                }
            )

    if not rows:
        return None
    return pd.DataFrame(
        rows, columns=["test_statistic_id", "required_species", "model_output_code"]
    )


def compute_cross_inputs_batch(
    sim_df: pd.DataFrame,
    cross_scenario_df: pd.DataFrame,
    scenario_name: str,
    species_units: dict,
    template_defaults: dict[str, float] | None = None,
    aux_by_sample_index: dict[int, dict[str, float]] | None = None,
    auxiliary_units: dict[str, str] | None = None,
) -> tuple[np.ndarray | None, list[str]]:
    """Compute the per-arm cross-input scalars for one scenario over a batch.

    Reuses the regular per-scenario test-stat machinery verbatim — a cross
    input is exactly a scalar-from-trajectory observable. Returns
    ``(matrix, column_keys)`` where ``matrix`` is ``(n_sims, n_cross_inputs)``
    row-aligned with ``sim_df`` (and therefore with this scenario's regular
    test-stats matrix), or ``(None, [])`` when no cross target references this
    scenario.
    """
    ts_df = build_cross_input_test_stats_df(cross_scenario_df, scenario_name)
    if ts_df is None:
        return None, []

    registry = build_test_stat_registry(ts_df)
    matrix = compute_test_statistics_batch(
        sim_df,
        ts_df,
        registry,
        species_units,
        template_defaults=template_defaults,
        aux_by_sample_index=aux_by_sample_index,
        auxiliary_units=auxiliary_units,
    )
    return matrix, list(ts_df["test_statistic_id"])
