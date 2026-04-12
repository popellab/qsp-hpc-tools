"""Tests for model_structure.json unit loader."""

import json

import pytest

from qsp_hpc.utils.model_structure_units import load_units_from_model_structure


@pytest.fixture
def structure_file(tmp_path):
    data = {
        "species": [
            {"name": "V_T.C1", "units": "cell"},
            {"name": "V_T.collagen", "units": "milligram"},
            {"name": "dimless_species"},
        ],
        "compartments": [
            {"name": "V_T", "volume_units": "milliliter"},
            {"name": "V_LN"},
        ],
        "parameters": [
            {"name": "rho_collagen", "units": "gram/milliliter"},
            {"name": "phi_collagen", "units": "dimensionless"},
            {"name": "no_units_param"},
        ],
    }
    p = tmp_path / "model_structure.json"
    p.write_text(json.dumps(data))
    return p


def test_species_units(structure_file):
    u = load_units_from_model_structure(structure_file)
    assert u["V_T.C1"] == "cell"
    assert u["V_T.collagen"] == "milligram"


def test_compartment_volume_units(structure_file):
    u = load_units_from_model_structure(structure_file)
    assert u["V_T"] == "milliliter"


def test_parameter_units(structure_file):
    u = load_units_from_model_structure(structure_file)
    assert u["rho_collagen"] == "gram/milliliter"
    assert u["phi_collagen"] == "dimensionless"


def test_missing_units_default_to_dimensionless(structure_file):
    u = load_units_from_model_structure(structure_file)
    assert u["dimless_species"] == "dimensionless"
    assert u["V_LN"] == "dimensionless"
    assert u["no_units_param"] == "dimensionless"


def test_name_collision_species_wins_over_param(tmp_path):
    # If a species and parameter share a name, species units should take precedence
    # (species is the canonical simulated state; parameters with same name are rare
    # and typically represent derived/initial-value duplicates).
    data = {
        "species": [{"name": "x", "units": "cell"}],
        "compartments": [],
        "parameters": [{"name": "x", "units": "milligram"}],
    }
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    assert load_units_from_model_structure(p)["x"] == "cell"


def test_name_collision_param_wins_over_compartment(tmp_path):
    data = {
        "species": [],
        "compartments": [{"name": "x", "volume_units": "milliliter"}],
        "parameters": [{"name": "x", "units": "gram"}],
    }
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    assert load_units_from_model_structure(p)["x"] == "gram"


def test_accepts_path_or_string(structure_file):
    as_path = load_units_from_model_structure(structure_file)
    as_str = load_units_from_model_structure(str(structure_file))
    assert as_path == as_str


def test_empty_sections(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"species": [], "compartments": [], "parameters": []}))
    assert load_units_from_model_structure(p) == {}


def test_missing_sections(tmp_path):
    # Only species provided — compartments/parameters keys absent
    p = tmp_path / "partial.json"
    p.write_text(json.dumps({"species": [{"name": "a", "units": "cell"}]}))
    assert load_units_from_model_structure(p) == {"a": "cell"}
