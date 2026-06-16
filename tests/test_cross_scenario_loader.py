"""Tests for cross_scenario_loader (YAML → DataFrame)."""

import json

import pytest
import yaml

from qsp_hpc.calibration.cross_scenario_loader import (
    hash_cross_scenario_targets,
    load_cross_scenario_targets,
)

_OBS_CODE = (
    "def compute_test_statistic(time, species_dict):\n"
    "    import numpy as np\n"
    "    return float(np.interp(21.0, np.asarray(time, dtype=float),\n"
    "                           np.asarray(species_dict['V_T.C1'], dtype=float)))\n"
)


def _input(role, scenario, species):
    return {
        "role": role,
        "scenario": scenario,
        "observable_code": _OBS_CODE,
        "required_species": species,
    }


def _cross_yaml(target_id="fold_a_vs_b"):
    return {
        "cross_scenario_target_id": target_id,
        "observable": {
            "code": "def compute(inputs):\n    return inputs['a'] / inputs['b']\n",
            "units": "dimensionless",
            "inputs": [
                _input("a", "scen_a", ["V_T.C1"]),
                _input("b", "scen_b", ["V_T.C1", "V_T"]),
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


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(payload))
    return p


def test_load_target(tmp_path):
    _write(tmp_path, "t1.yaml", _cross_yaml("fold1"))
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
    assert inputs[0]["role"] == "a"
    assert inputs[0]["scenario"] == "scen_a"
    assert inputs[0]["required_species"] == ["V_T.C1"]
    assert "units" not in inputs[0]  # pintless runtime: no per-input units
    assert "compute_test_statistic" in inputs[0]["observable_code"]
    assert inputs[1]["required_species"] == ["V_T.C1", "V_T"]


def test_load_multiple_targets_sorted(tmp_path):
    _write(tmp_path, "b.yaml", _cross_yaml("b"))
    _write(tmp_path, "a.yaml", _cross_yaml("a"))
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
    cfg = _cross_yaml()
    cfg["observable"]["inputs"] = cfg["observable"]["inputs"][:1]
    _write(tmp_path, "t.yaml", cfg)
    with pytest.raises(ValueError, match="fewer than 2 inputs"):
        load_cross_scenario_targets(tmp_path)


def test_input_missing_observable_code_rejected(tmp_path):
    cfg = _cross_yaml()
    del cfg["observable"]["inputs"][0]["observable_code"]
    _write(tmp_path, "t.yaml", cfg)
    with pytest.raises(ValueError, match="no observable_code"):
        load_cross_scenario_targets(tmp_path)


def test_input_missing_species_rejected(tmp_path):
    cfg = _cross_yaml()
    cfg["observable"]["inputs"][0]["required_species"] = []
    _write(tmp_path, "t.yaml", cfg)
    with pytest.raises(ValueError, match="no required_species"):
        load_cross_scenario_targets(tmp_path)


def test_legacy_input_kind_rejected(tmp_path):
    cfg = _cross_yaml()
    cfg["observable"]["inputs"][0]["input_kind"] = "test_statistic"
    cfg["observable"]["inputs"][0]["test_statistic_id"] = "stat_x"
    _write(tmp_path, "t.yaml", cfg)
    with pytest.raises(ValueError, match="retired"):
        load_cross_scenario_targets(tmp_path)


def test_hash_changes_on_content_edit(tmp_path):
    _write(tmp_path, "t.yaml", _cross_yaml("h1"))
    h1 = hash_cross_scenario_targets(tmp_path)

    cfg = _cross_yaml("h1")
    cfg["empirical_data"]["median"] = [2.0]
    _write(tmp_path, "t.yaml", cfg)
    h2 = hash_cross_scenario_targets(tmp_path)
    assert h1 != h2


def test_hash_stable_on_unchanged(tmp_path):
    _write(tmp_path, "t.yaml", _cross_yaml("h1"))
    h1 = hash_cross_scenario_targets(tmp_path)
    h2 = hash_cross_scenario_targets(tmp_path)
    assert h1 == h2
