"""Tests for hash utility functions."""

import pytest

from qsp_hpc.utils.hash_utils import (
    _safe_sort_key,
    compute_definition_hash,
    compute_pool_id_hash,
    generate_filename,
    normalize_model_context,
)


class TestComputeDefinitionHash:
    """Tests for compute_definition_hash function."""

    def test_parameter_hash_basic(self):
        """Test basic parameter hash computation."""
        definition = {
            "name": "k_absorption",
            "units": "1/hour",
            "canonical_scale": "log",
            "value": 1.5,
        }
        hash_val = compute_definition_hash(definition, "parameter")
        assert isinstance(hash_val, str)
        assert len(hash_val) == 8
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_species_hash_basic(self):
        """Test basic species hash computation."""
        definition = {
            "name": "drug_plasma",
            "compartment": "plasma",
            "units": "ng/mL",
            "value": 0.0,
        }
        hash_val = compute_definition_hash(definition, "species")
        assert isinstance(hash_val, str)
        assert len(hash_val) == 8

    def test_hash_stability(self):
        """Test that same definition produces same hash."""
        definition = {
            "name": "k_clearance",
            "units": "L/hour",
            "canonical_scale": "linear",
            "tags": ["pharmacokinetic", "clearance"],
        }
        hash1 = compute_definition_hash(definition, "parameter")
        hash2 = compute_definition_hash(definition, "parameter")
        assert hash1 == hash2

    def test_name_change_no_hash_change(self):
        """Test that renaming doesn't change hash (name excluded)."""
        def1 = {"name": "old_name", "units": "mg/L", "canonical_scale": "log"}
        def2 = {"name": "new_name", "units": "mg/L", "canonical_scale": "log"}
        hash1 = compute_definition_hash(def1, "parameter")
        hash2 = compute_definition_hash(def2, "parameter")
        assert hash1 == hash2

    def test_description_change_no_hash_change(self):
        """Test that description changes don't affect hash."""
        def1 = {"name": "k_absorption", "description": "Old description", "units": "1/hour"}
        def2 = {
            "name": "k_absorption",
            "description": "New improved description",
            "units": "1/hour",
        }
        hash1 = compute_definition_hash(def1, "parameter")
        hash2 = compute_definition_hash(def2, "parameter")
        assert hash1 == hash2

    def test_value_change_no_hash_change(self):
        """Test that value changes don't affect hash."""
        def1 = {"name": "k_absorption", "units": "1/hour", "value": 1.0}
        def2 = {"name": "k_absorption", "units": "1/hour", "value": 10.0}
        hash1 = compute_definition_hash(def1, "parameter")
        hash2 = compute_definition_hash(def2, "parameter")
        assert hash1 == hash2

    def test_units_change_changes_hash(self):
        """Test that units change triggers hash change."""
        def1 = {"name": "k_absorption", "units": "1/hour"}
        def2 = {"name": "k_absorption", "units": "1/minute"}
        hash1 = compute_definition_hash(def1, "parameter")
        hash2 = compute_definition_hash(def2, "parameter")
        assert hash1 != hash2

    def test_scale_change_changes_hash(self):
        """Test that canonical scale change triggers hash change."""
        def1 = {"name": "k_absorption", "canonical_scale": "linear"}
        def2 = {"name": "k_absorption", "canonical_scale": "log"}
        hash1 = compute_definition_hash(def1, "parameter")
        hash2 = compute_definition_hash(def2, "parameter")
        assert hash1 != hash2

    def test_compartment_change_changes_hash(self):
        """Test that compartment change triggers hash change for species."""
        def1 = {"name": "drug", "compartment": "plasma"}
        def2 = {"name": "drug", "compartment": "tissue"}
        hash1 = compute_definition_hash(def1, "species")
        hash2 = compute_definition_hash(def2, "species")
        assert hash1 != hash2

    def test_tags_order_stability(self):
        """Test that tag order doesn't affect hash."""
        def1 = {"name": "k_absorption", "tags": ["pk", "absorption", "oral"]}
        def2 = {"name": "k_absorption", "tags": ["oral", "pk", "absorption"]}
        hash1 = compute_definition_hash(def1, "parameter")
        hash2 = compute_definition_hash(def2, "parameter")
        assert hash1 == hash2

    def test_tags_content_changes_hash(self):
        """Test that changing tag content triggers hash change."""
        def1 = {"name": "k_absorption", "tags": ["pk", "absorption"]}
        def2 = {"name": "k_absorption", "tags": ["pk", "absorption", "oral"]}
        hash1 = compute_definition_hash(def1, "parameter")
        hash2 = compute_definition_hash(def2, "parameter")
        assert hash1 != hash2

    def test_empty_definition(self):
        """Test hash computation with minimal definition."""
        definition = {"name": "minimal"}
        hash_val = compute_definition_hash(definition, "parameter")
        assert isinstance(hash_val, str)
        assert len(hash_val) == 8


class TestNormalizeModelContext:
    """Tests for normalize_model_context function."""

    def test_basic_normalization(self):
        """Test basic model context normalization."""
        context = {
            "derived_from_context": ["param1", "param2"],
            "reactions_and_rules": [{"reaction": "A -> B", "reaction_rate": "k1 * A"}],
        }
        normalized = normalize_model_context(context)
        assert "derived_from_context" in normalized
        assert "reactions_and_rules" in normalized

    def test_derived_from_context_sorting(self):
        """Test that derived_from_context is sorted."""
        context = {"derived_from_context": ["zebra", "apple", "banana"], "reactions_and_rules": []}
        normalized = normalize_model_context(context)
        assert normalized["derived_from_context"] == ["apple", "banana", "zebra"]

    def test_derived_from_context_with_dicts(self):
        """Test derived_from_context normalization with dict entries."""
        context = {
            "derived_from_context": [
                {"name": "param1", "description": "First param"},
                {"name": "param2", "description": "Second param"},
            ],
            "reactions_and_rules": [],
        }
        normalized = normalize_model_context(context)
        assert normalized["derived_from_context"] == ["param1", "param2"]

    def test_reactions_sorting(self):
        """Test that reactions are sorted consistently."""
        context = {
            "reactions_and_rules": [
                {"reaction": "C -> D", "reaction_rate": "k3 * C"},
                {"reaction": "A -> B", "reaction_rate": "k1 * A"},
                {"reaction": "B -> C", "reaction_rate": "k2 * B"},
            ]
        }
        normalized = normalize_model_context(context)
        reactions = normalized["reactions_and_rules"]

        # Should be sorted by reaction
        assert reactions[0]["reaction"] == "A -> B"
        assert reactions[1]["reaction"] == "B -> C"
        assert reactions[2]["reaction"] == "C -> D"

    def test_other_parameters_normalization(self):
        """Test that other_parameters are normalized to names only."""
        context = {
            "reactions_and_rules": [
                {
                    "reaction": "A -> B",
                    "other_parameters": [
                        {"name": "k2", "description": "Rate constant"},
                        {"name": "k1", "description": "Another rate"},
                    ],
                }
            ]
        }
        normalized = normalize_model_context(context)
        params = normalized["reactions_and_rules"][0]["other_parameters"]
        assert params == ["k1", "k2"]  # Sorted

    def test_other_species_normalization(self):
        """Test that other_species are normalized to names only."""
        context = {
            "reactions_and_rules": [
                {
                    "reaction": "A -> B",
                    "other_species": [
                        {"name": "C", "description": "Species C"},
                        {"name": "A", "description": "Species A"},
                    ],
                }
            ]
        }
        normalized = normalize_model_context(context)
        species = normalized["reactions_and_rules"][0]["other_species"]
        assert species == ["A", "C"]  # Sorted

    def test_normalization_stability(self):
        """Test that normalization is stable (same input -> same output)."""
        context = {
            "derived_from_context": ["param2", "param1"],
            "reactions_and_rules": [
                {"reaction": "B -> C", "reaction_rate": "k * B"},
                {"reaction": "A -> B", "reaction_rate": "k * A"},
            ],
        }
        normalized1 = normalize_model_context(context)
        normalized2 = normalize_model_context(context)

        # Convert to JSON to compare
        import json

        json1 = json.dumps(normalized1, sort_keys=True)
        json2 = json.dumps(normalized2, sort_keys=True)
        assert json1 == json2

    def test_invalid_model_context(self):
        """Test that invalid model context raises error."""
        with pytest.raises(ValueError, match="model_context must be a dict"):
            normalize_model_context("invalid")

    def test_empty_model_context(self):
        """Test normalization with empty model context."""
        context = {}
        normalized = normalize_model_context(context)
        assert normalized == {}

    def test_rule_type_preservation(self):
        """Test that rule_type is preserved in normalization."""
        context = {"reactions_and_rules": [{"rule": "x = y + z", "rule_type": "assignment"}]}
        normalized = normalize_model_context(context)
        assert normalized["reactions_and_rules"][0]["rule_type"] == "assignment"


class TestSafeSortKey:
    """Tests for _safe_sort_key function."""

    def test_basic_sort_key(self):
        """Test basic sort key generation."""
        entry = {"reaction": "A -> B", "rule": "", "reaction_rate": "k * A"}
        key = _safe_sort_key(entry)
        assert isinstance(key, tuple)
        assert len(key) == 3

    def test_missing_fields(self):
        """Test sort key with missing fields."""
        entry = {}
        key = _safe_sort_key(entry)
        assert key == ("", "", "")

    def test_none_values(self):
        """Test sort key with None values."""
        entry = {"reaction": None, "rule": None, "reaction_rate": None}
        key = _safe_sort_key(entry)
        assert key == ("", "", "")

    def test_sort_ordering(self):
        """Test that sort keys produce correct ordering."""
        entries = [
            {"reaction": "C -> D", "rule": "", "reaction_rate": "k3"},
            {"reaction": "A -> B", "rule": "", "reaction_rate": "k1"},
            {"reaction": "B -> C", "rule": "", "reaction_rate": "k2"},
        ]
        sorted_entries = sorted(entries, key=_safe_sort_key)
        assert sorted_entries[0]["reaction"] == "A -> B"
        assert sorted_entries[1]["reaction"] == "B -> C"
        assert sorted_entries[2]["reaction"] == "C -> D"


class TestGenerateFilename:
    """Tests for generate_filename function."""

    def test_basic_filename(self):
        """Test basic filename generation."""
        filename = generate_filename("k_absorption", "abc12345")
        assert filename == "k_absorption_abc12345.yaml"

    def test_custom_extension(self):
        """Test filename with custom extension."""
        filename = generate_filename("drug_plasma", "xyz789ab", "json")
        assert filename == "drug_plasma_xyz789ab.json"

    def test_filename_components(self):
        """Test that filename contains all components."""
        name = "test_param"
        hash_val = "12345678"
        filename = generate_filename(name, hash_val)
        assert name in filename
        assert hash_val in filename
        assert filename.endswith(".yaml")


class TestComputePoolIdHash:
    """Tests for the simplified pool-id hash (Layer 3 of the
    local-observable-eval plan).

    Inputs collapsed to ``sha256(binary_bytes | scenario_yaml_content)`` —
    everything else is either upstream sampling machinery (priors / seed,
    enforced at the runner) or a read-time decision (test stats, traj
    columns, cadence). See ``compute_pool_id_hash`` docstring.
    """

    @staticmethod
    def _write_inputs(tmp_path, *, binary_bytes=b"\x7fELF binary v1", scenario="a: 1\n"):
        binary = tmp_path / "qsp_sim"
        binary.write_bytes(binary_bytes)
        scenario_yaml = tmp_path / "scenario.yaml"
        scenario_yaml.write_text(scenario)
        return binary, scenario_yaml

    def test_hash_is_deterministic(self, tmp_path):
        binary, scenario_yaml = self._write_inputs(tmp_path)
        h1 = compute_pool_id_hash(binary_path=binary, scenario_yaml=scenario_yaml)
        h2 = compute_pool_id_hash(binary_path=binary, scenario_yaml=scenario_yaml)
        assert h1 == h2

    def test_hash_is_full_sha256_hex(self, tmp_path):
        binary, scenario_yaml = self._write_inputs(tmp_path)
        h = compute_pool_id_hash(binary_path=binary, scenario_yaml=scenario_yaml)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_binary_bytes_change_hash(self, tmp_path):
        """Rebuilding qsp_sim with new model semantics must invalidate
        the pool (#56). Two binaries with different bytes produce
        different hashes."""
        bin_a, scen = self._write_inputs(tmp_path, binary_bytes=b"\x7fELF binary A")
        bin_b = tmp_path / "qsp_sim_b"
        bin_b.write_bytes(b"\x7fELF binary B")
        h_a = compute_pool_id_hash(binary_path=bin_a, scenario_yaml=scen)
        h_b = compute_pool_id_hash(binary_path=bin_b, scenario_yaml=scen)
        assert h_a != h_b

    def test_scenario_content_changes_hash(self, tmp_path):
        """Edits to scenario YAML (drug regimen, IC, stop_time) must
        invalidate the pool — they change ``f(theta)``."""
        binary, scen_a = self._write_inputs(tmp_path, scenario="stop_time: 21\n")
        scen_b = tmp_path / "scenario_b.yaml"
        scen_b.write_text("stop_time: 90\n")
        h_a = compute_pool_id_hash(binary_path=binary, scenario_yaml=scen_a)
        h_b = compute_pool_id_hash(binary_path=binary, scenario_yaml=scen_b)
        assert h_a != h_b

    def test_keyword_only_signature(self, tmp_path):
        """Args are keyword-only — positional calls must fail. Guards
        against silent breakage if someone reorders the parameters."""
        binary, scenario_yaml = self._write_inputs(tmp_path)
        with pytest.raises(TypeError):
            compute_pool_id_hash(binary, scenario_yaml)  # type: ignore[misc]

    def test_str_and_path_inputs_equivalent(self, tmp_path):
        """str-paths and Path objects produce the same hash."""
        binary, scenario_yaml = self._write_inputs(tmp_path)
        h_path = compute_pool_id_hash(binary_path=binary, scenario_yaml=scenario_yaml)
        h_str = compute_pool_id_hash(binary_path=str(binary), scenario_yaml=str(scenario_yaml))
        assert h_path == h_str


class TestHashIntegration:
    """Integration tests for hash functionality."""

    def test_full_parameter_workflow(self):
        """Test complete parameter hash workflow."""
        definition = {
            "name": "k_clearance",
            "units": "L/hour",
            "canonical_scale": "log",
            "value": 5.0,
            "description": "Clearance rate constant",
            "tags": ["pk", "clearance"],
            "model_context": {
                "derived_from_context": ["CL", "V"],
                "reactions_and_rules": [
                    {"reaction": "Drug_central -> ", "reaction_rate": "k_clearance * Drug_central"}
                ],
            },
        }

        # Compute hash
        hash_val = compute_definition_hash(definition, "parameter")

        # Generate filename
        filename = generate_filename(definition["name"], hash_val)

        assert len(hash_val) == 8
        assert filename.startswith("k_clearance_")
        assert filename.endswith(".yaml")

    def test_semantic_vs_cosmetic_changes(self):
        """Test that semantic changes affect hash but cosmetic don't."""
        base_def = {
            "name": "k_absorption",
            "units": "1/hour",
            "canonical_scale": "log",
            "value": 1.5,
            "description": "Absorption rate",
        }

        base_hash = compute_definition_hash(base_def, "parameter")

        # Cosmetic changes (should NOT change hash)
        cosmetic_changes = [
            {**base_def, "name": "k_abs"},  # Rename
            {**base_def, "description": "New description"},  # Edit description
            {**base_def, "value": 10.0},  # Change value
            {**base_def, "created_at": "2025-01-01"},  # Add metadata
        ]

        for changed_def in cosmetic_changes:
            changed_hash = compute_definition_hash(changed_def, "parameter")
            assert changed_hash == base_hash, f"Cosmetic change affected hash: {changed_def}"

        # Semantic changes (SHOULD change hash)
        semantic_changes = [
            {**base_def, "units": "1/minute"},  # Change units
            {**base_def, "canonical_scale": "linear"},  # Change scale
            {**base_def, "tags": ["new_tag"]},  # Change tags
        ]

        for changed_def in semantic_changes:
            changed_hash = compute_definition_hash(changed_def, "parameter")
            assert changed_hash != base_hash, f"Semantic change didn't affect hash: {changed_def}"
