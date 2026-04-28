"""Tests for cross_scenario_loader (YAML → DataFrame)."""

import json

import pytest
import yaml

from qsp_hpc.calibration.cross_scenario_loader import (
    hash_cross_scenario_targets,
    load_cross_scenario_targets,
)


def _scalar_yaml(target_id="fold_a_vs_b"):
    return {
        "cross_scenario_target_id": target_id,
        "observable": {
            "code": "def compute(inputs, ureg):\n    return inputs['a'] / inputs['b']\n",
            "units": "dimensionless",
            "inputs": [
                {
                    "role": "a",
                    "scenario": "scen_a",
                    "input_kind": "test_statistic",
                    "test_statistic_id": "stat_x",
                },
                {
                    "role": "b",
                    "scenario": "scen_b",
                    "input_kind": "test_statistic",
                    "test_statistic_id": "stat_x",
                },
            ],
        },
        "empirical_data": {
            "median": [1.0],
            "ci95": [[0.5, 2.0]],
            "units": "dimensionless",
            "sample_size": 1,
            "sample_size_rationale": "mechanistic prior",
            "inputs": [],
            "assumptions": [],
            "distribution_code": "def derive_distribution(inputs, ureg): return {}\n",
        },
        "study_interpretation": "test",
        "key_assumptions": ["mechanistic"],
        "epistemic_basis": "mechanistic",
    }


def _timeseries_yaml(target_id="ts_a_vs_b"):
    cfg = _scalar_yaml(target_id)
    cfg["observable"]["inputs"] = [
        {
            "role": "a",
            "scenario": "scen_a",
            "input_kind": "timeseries",
            "required_species": ["V_T.C1"],
        },
        {
            "role": "b",
            "scenario": "scen_b",
            "input_kind": "timeseries",
            "required_species": ["V_T.C1", "V_T"],
        },
    ]
    return cfg


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(payload))
    return p


def test_load_scalar_target(tmp_path):
    _write(tmp_path, "t1.yaml", _scalar_yaml("fold1"))
    df = load_cross_scenario_targets(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["cross_scenario_target_id"] == "fold1"
    assert row["units"] == "dimensionless"
    assert row["median"] == 1.0
    assert row["ci95_lower"] == 0.5
    assert row["ci95_upper"] == 2.0

    inputs = json.loads(row["inputs_json"])
    assert len(inputs) == 2
    assert inputs[0] == {
        "role": "a",
        "scenario": "scen_a",
        "input_kind": "test_statistic",
        "test_statistic_id": "stat_x",
    }


def test_load_timeseries_target(tmp_path):
    _write(tmp_path, "ts.yaml", _timeseries_yaml("ts1"))
    df = load_cross_scenario_targets(tmp_path)
    inputs = json.loads(df.iloc[0]["inputs_json"])
    assert inputs[0]["input_kind"] == "timeseries"
    assert inputs[0]["required_species"] == ["V_T.C1"]
    assert inputs[1]["required_species"] == ["V_T.C1", "V_T"]
    assert "test_statistic_id" not in inputs[0]


def test_load_multiple_targets_sorted(tmp_path):
    _write(tmp_path, "b.yaml", _scalar_yaml("b"))
    _write(tmp_path, "a.yaml", _scalar_yaml("a"))
    df = load_cross_scenario_targets(tmp_path)
    # files are sorted by name → 'a.yaml' before 'b.yaml'
    assert list(df["cross_scenario_target_id"]) == ["a", "b"]


def test_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No YAML files"):
        load_cross_scenario_targets(tmp_path)


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cross_scenario_targets(tmp_path / "missing")


def test_single_input_rejected(tmp_path):
    cfg = _scalar_yaml()
    cfg["observable"]["inputs"] = cfg["observable"]["inputs"][:1]
    _write(tmp_path, "t.yaml", cfg)
    with pytest.raises(ValueError, match="fewer than 2 inputs"):
        load_cross_scenario_targets(tmp_path)


def test_test_statistic_input_missing_id_rejected(tmp_path):
    cfg = _scalar_yaml()
    del cfg["observable"]["inputs"][0]["test_statistic_id"]
    _write(tmp_path, "t.yaml", cfg)
    with pytest.raises(ValueError, match="no test_statistic_id"):
        load_cross_scenario_targets(tmp_path)


def test_timeseries_input_missing_species_rejected(tmp_path):
    cfg = _timeseries_yaml()
    cfg["observable"]["inputs"][0]["required_species"] = []
    _write(tmp_path, "t.yaml", cfg)
    with pytest.raises(ValueError, match="no required_species"):
        load_cross_scenario_targets(tmp_path)


def test_invalid_kind_rejected(tmp_path):
    cfg = _scalar_yaml()
    cfg["observable"]["inputs"][0]["input_kind"] = "garbage"
    _write(tmp_path, "t.yaml", cfg)
    with pytest.raises(ValueError, match="invalid input_kind"):
        load_cross_scenario_targets(tmp_path)


def test_hash_changes_on_content_edit(tmp_path):
    _write(tmp_path, "t.yaml", _scalar_yaml("h1"))
    h1 = hash_cross_scenario_targets(tmp_path)

    cfg = _scalar_yaml("h1")
    cfg["empirical_data"]["median"] = [2.0]
    _write(tmp_path, "t.yaml", cfg)
    h2 = hash_cross_scenario_targets(tmp_path)
    assert h1 != h2


def test_hash_stable_on_unchanged(tmp_path):
    _write(tmp_path, "t.yaml", _scalar_yaml("h1"))
    h1 = hash_cross_scenario_targets(tmp_path)
    h2 = hash_cross_scenario_targets(tmp_path)
    assert h1 == h2
