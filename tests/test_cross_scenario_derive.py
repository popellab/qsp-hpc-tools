"""Tests for cross_scenario_derive (per-arm cross-input derivation)."""

import json

import numpy as np
import pandas as pd

from qsp_hpc.batch.cross_scenario_derive import (
    build_cross_input_test_stats_df,
    compute_cross_inputs_batch,
    cross_input_column_key,
)

# Within-TLA CD8 number at day 21 = (CD8_TLA + CD8_TLA_act) interpolated to t=21.
_INTLA_CODE = (
    "def compute_test_statistic(time, species_dict):\n"
    "    import numpy as np\n"
    "    t = np.asarray(time, dtype=float)\n"
    "    n = (np.asarray(species_dict['V_T.CD8_TLA'], dtype=float)\n"
    "         + np.asarray(species_dict['V_T.CD8_TLA_act'], dtype=float))\n"
    "    return float(np.interp(21.0, t, n))\n"
)


def _cross_df(target_id="invar"):
    """A cross target referencing two scenarios (nivo / urelumab) with the same
    within-TLA-number per-arm observable."""
    inputs = [
        {
            "role": "nivo",
            "scenario": "gvax_nivo",
            "observable_code": _INTLA_CODE,
            "required_species": ["V_T.CD8_TLA", "V_T.CD8_TLA_act"],
        },
        {
            "role": "urelumab",
            "scenario": "gvax_nivo_urelumab",
            "observable_code": _INTLA_CODE,
            "required_species": ["V_T.CD8_TLA", "V_T.CD8_TLA_act"],
        },
    ]
    return pd.DataFrame(
        [
            {
                "cross_scenario_target_id": target_id,
                "composer_code": "def compute(inputs):\n    return inputs['urelumab'] / inputs['nivo']\n",
                "units": "dimensionless",
                "inputs_json": json.dumps(inputs),
                "median": 1.0,
                "ci95_lower": 0.5,
                "ci95_upper": 2.0,
                "sample_size": 1,
            }
        ]
    )


def _sim_df(resting_per_row, active_per_row, time_series=(0.0, 10.0, 21.0)):
    """One row per sim; species columns hold per-row lists over the time axis."""
    rows = []
    for r, a in zip(resting_per_row, active_per_row):
        rows.append(
            {
                "status": 0,
                "time": list(time_series),
                "V_T.CD8_TLA": [r, r, r],
                "V_T.CD8_TLA_act": [a, a, a],
            }
        )
    return pd.DataFrame(rows)


def test_column_key():
    assert cross_input_column_key("invar", "nivo") == "invar::nivo"


def test_build_test_stats_df_filters_by_scenario():
    df = _cross_df("invar")
    nivo = build_cross_input_test_stats_df(df, "gvax_nivo")
    assert list(nivo["test_statistic_id"]) == ["invar::nivo"]
    assert nivo.iloc[0]["required_species"] == "V_T.CD8_TLA,V_T.CD8_TLA_act"
    assert "compute_test_statistic" in nivo.iloc[0]["model_output_code"]

    ure = build_cross_input_test_stats_df(df, "gvax_nivo_urelumab")
    assert list(ure["test_statistic_id"]) == ["invar::urelumab"]


def test_build_test_stats_df_none_when_unreferenced():
    df = _cross_df("invar")
    assert build_cross_input_test_stats_df(df, "some_other_scenario") is None


def test_compute_cross_inputs_batch():
    df = _cross_df("invar")
    species_units = {"V_T.CD8_TLA": "cell", "V_T.CD8_TLA_act": "cell"}
    sim = _sim_df(resting_per_row=[10.0, 20.0], active_per_row=[5.0, 0.0])
    matrix, keys = compute_cross_inputs_batch(sim, df, "gvax_nivo", species_units)
    assert keys == ["invar::nivo"]
    assert matrix.shape == (2, 1)
    # within-TLA number = resting + active at d21
    np.testing.assert_array_almost_equal(matrix[:, 0], [15.0, 20.0])


def test_compute_cross_inputs_batch_none_when_unreferenced():
    df = _cross_df("invar")
    sim = _sim_df([1.0], [1.0])
    matrix, keys = compute_cross_inputs_batch(sim, df, "unrelated", {})
    assert matrix is None
    assert keys == []


def test_derive_then_compose_end_to_end():
    """Full Option-B chain: derive per-arm cross inputs on each scenario's
    sim_df, assemble the cross_inputs channel, compose the cross-arm ratio."""
    import numpy as np

    from qsp_hpc.batch.cross_scenario_worker import compute_cross_scenario_statistics

    df = _cross_df("invar")
    units = {"V_T.CD8_TLA": "cell", "V_T.CD8_TLA_act": "cell"}

    # nivo arm: within-TLA number at d21 = resting + active.
    nivo_sim = _sim_df(resting_per_row=[10.0, 20.0, 30.0], active_per_row=[0.0, 0.0, 0.0])
    # urelumab arm: same number, activation shifted (number invariant).
    ure_sim = _sim_df(resting_per_row=[2.0, 4.0, 6.0], active_per_row=[8.0, 16.0, 24.0])

    nivo_mat, nivo_keys = compute_cross_inputs_batch(nivo_sim, df, "gvax_nivo", units)
    ure_mat, ure_keys = compute_cross_inputs_batch(ure_sim, df, "gvax_nivo_urelumab", units)

    scenario_meta = {
        "gvax_nivo": {"cross_inputs": {k: nivo_mat[:, i] for i, k in enumerate(nivo_keys)}},
        "gvax_nivo_urelumab": {"cross_inputs": {k: ure_mat[:, i] for i, k in enumerate(ure_keys)}},
    }
    out = compute_cross_scenario_statistics(scenario_meta, df)
    # nivo = [10,20,30], urelumab = [10,20,30] → ratio = 1.0 everywhere.
    np.testing.assert_array_almost_equal(out[:, 0], [1.0, 1.0, 1.0])
