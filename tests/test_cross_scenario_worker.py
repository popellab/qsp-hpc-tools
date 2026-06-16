"""Tests for cross_scenario_worker (pintless composer dispatch)."""

import json

import numpy as np
import pandas as pd
import pytest

from qsp_hpc.batch.cross_scenario_derive import cross_input_column_key
from qsp_hpc.batch.cross_scenario_worker import (
    build_cross_scenario_registry,
    compute_cross_scenario_statistics,
)

# A per-arm observable used in the inputs_json payloads. Never executed by the
# composer (it runs at derive time); only its presence matters here.
_OBS_CODE = (
    "def compute_test_statistic(time, species_dict):\n"
    "    import numpy as np\n"
    "    return float(np.interp(21.0, np.asarray(time, dtype=float),\n"
    "                           np.asarray(species_dict['V_T.C1'], dtype=float)))\n"
)


def _ratio_target_row(target_id="fold", scen_a="A", scen_b="B"):
    """Composer = inputs['a'] / inputs['b'] over two per-arm scalars."""
    return {
        "cross_scenario_target_id": target_id,
        "composer_code": "def compute(inputs):\n    return inputs['a'] / inputs['b']\n",
        "units": "dimensionless",
        "inputs_json": json.dumps(
            [
                {
                    "role": "a",
                    "scenario": scen_a,
                    "observable_code": _OBS_CODE,
                    "required_species": ["V_T.C1"],
                },
                {
                    "role": "b",
                    "scenario": scen_b,
                    "observable_code": _OBS_CODE,
                    "required_species": ["V_T.C1"],
                },
            ]
        ),
        "median": 1.0,
        "ci95_lower": 0.5,
        "ci95_upper": 2.0,
        "sample_size": 1,
    }


def _cross_inputs(target_id, role_to_array):
    """Build a scenario's cross_inputs dict keyed by {target}::{role}."""
    return {
        cross_input_column_key(target_id, role): np.asarray(arr, dtype=float)
        for role, arr in role_to_array.items()
    }


# ============================================================================
# Registry compilation
# ============================================================================


def test_build_registry_compiles_all_rows():
    df = pd.DataFrame([_ratio_target_row("t1"), _ratio_target_row("t2")])
    reg = build_cross_scenario_registry(df)
    assert set(reg.keys()) == {"t1", "t2"}
    assert all(callable(f) for f in reg.values())


def test_build_registry_rejects_empty_code():
    df = pd.DataFrame([{**_ratio_target_row("bad"), "composer_code": None}])
    with pytest.raises(ValueError, match="empty composer_code"):
        build_cross_scenario_registry(df)


def test_build_registry_rejects_missing_compute():
    df = pd.DataFrame([{**_ratio_target_row("bad"), "composer_code": "x = 1"}])
    with pytest.raises(ValueError, match="must define a function named 'compute'"):
        build_cross_scenario_registry(df)


# ============================================================================
# Compose
# ============================================================================


def test_compose_basic():
    """Two scenarios, composer = a / b over per-arm cross inputs."""
    df = pd.DataFrame([_ratio_target_row("fold")])
    scenario_meta = {
        "A": {"cross_inputs": _cross_inputs("fold", {"a": [1.0, 2.0, 4.0, 8.0]})},
        "B": {"cross_inputs": _cross_inputs("fold", {"b": [1.0, 1.0, 2.0, 4.0]})},
    }
    out = compute_cross_scenario_statistics(scenario_meta, df)
    assert out.shape == (4, 1)
    np.testing.assert_array_almost_equal(out[:, 0], [1.0, 2.0, 2.0, 2.0])


def test_compose_propagates_nan():
    """Non-finite in either per-arm input → cross-scenario NaN for that row."""
    df = pd.DataFrame([_ratio_target_row("fold")])
    scenario_meta = {
        "A": {"cross_inputs": _cross_inputs("fold", {"a": [1.0, np.nan, 4.0]})},
        "B": {"cross_inputs": _cross_inputs("fold", {"b": [1.0, 1.0, np.nan]})},
    }
    out = compute_cross_scenario_statistics(scenario_meta, df)
    assert out[0, 0] == 1.0
    assert np.isnan(out[1, 0])
    assert np.isnan(out[2, 0])


def test_compose_reduction_error_is_nan():
    """A reduction that raises (e.g. divide-by-zero handled as inf, or a real
    exception) leaves that row NaN without killing the batch."""
    row = _ratio_target_row("boom")
    row["composer_code"] = "def compute(inputs):\n    raise RuntimeError('nope')\n"
    df = pd.DataFrame([row])
    scenario_meta = {
        "A": {"cross_inputs": _cross_inputs("boom", {"a": [1.0, 2.0]})},
        "B": {"cross_inputs": _cross_inputs("boom", {"b": [1.0, 1.0]})},
    }
    out = compute_cross_scenario_statistics(scenario_meta, df)
    assert np.all(np.isnan(out[:, 0]))


def test_two_targets_independent_columns():
    df = pd.DataFrame([_ratio_target_row("t1"), _ratio_target_row("t2")])
    scenario_meta = {
        "A": {
            "cross_inputs": {
                **_cross_inputs("t1", {"a": [4.0, 6.0]}),
                **_cross_inputs("t2", {"a": [10.0, 20.0]}),
            }
        },
        "B": {
            "cross_inputs": {
                **_cross_inputs("t1", {"b": [2.0, 3.0]}),
                **_cross_inputs("t2", {"b": [5.0, 5.0]}),
            }
        },
    }
    out = compute_cross_scenario_statistics(scenario_meta, df)
    assert out.shape == (2, 2)
    np.testing.assert_array_almost_equal(out[:, 0], [2.0, 2.0])  # t1
    np.testing.assert_array_almost_equal(out[:, 1], [2.0, 4.0])  # t2


def test_unknown_scenario_raises():
    df = pd.DataFrame([_ratio_target_row("fold", scen_a="MISSING")])
    scenario_meta = {
        "B": {"cross_inputs": _cross_inputs("fold", {"b": [1.0]})},
    }
    with pytest.raises(ValueError, match="not in scenario_meta"):
        compute_cross_scenario_statistics(scenario_meta, df)


def test_scenario_without_cross_inputs_raises():
    df = pd.DataFrame([_ratio_target_row("fold")])
    scenario_meta = {
        "A": {"cross_inputs": _cross_inputs("fold", {"a": [1.0]})},
        "B": {},  # referenced but has no cross_inputs
    }
    with pytest.raises(ValueError, match="has no 'cross_inputs'"):
        compute_cross_scenario_statistics(scenario_meta, df)


def test_missing_column_key_raises():
    """Scenario present with cross_inputs, but the expected {target}::{role}
    column was not derived → loud failure."""
    df = pd.DataFrame([_ratio_target_row("fold")])
    scenario_meta = {
        "A": {"cross_inputs": _cross_inputs("fold", {"a": [1.0]})},
        # B has a cross_inputs dict but keyed for the wrong role.
        "B": {"cross_inputs": _cross_inputs("fold", {"wrong_role": [1.0]})},
    }
    with pytest.raises(ValueError, match="is missing"):
        compute_cross_scenario_statistics(scenario_meta, df)
