"""Tests for YAML calibration target loader."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from qsp_hpc.calibration.yaml_loader import (
    _generate_wrapper_code,
    hash_calibration_targets,
    load_calibration_targets,
)
from qsp_hpc.utils.unit_registry import ureg

# ============================================================================
# YAML fixture data
# ============================================================================

BASELINE_YAML = {
    "calibration_target_id": "m1_m2_ratio",
    "observable": {
        "code": (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    M1 = species_dict['V_T.Mac_M1']\n"
            "    M2 = species_dict['V_T.Mac_M2']\n"
            "    ratio = (M1 / M2).to('dimensionless')\n"
            "    return ratio\n"
        ),
        "units": "dimensionless",
        "species": ["V_T.Mac_M1", "V_T.Mac_M2"],
        "constants": [],
    },
    "empirical_data": {
        "median": [0.518],
        "ci95": [[0.071, 3.722]],
        "units": "dimensionless",
        "sample_size": 30,
        "index_values": None,
    },
}

CONSTANTS_YAML = {
    "calibration_target_id": "cd8_density_baseline",
    "observable": {
        "code": (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    cd8_eff = species_dict['V_T.CD8']\n"
            "    cd8_exh = species_dict['V_T.CD8_exh']\n"
            "    c_cells = species_dict['V_T.C1']\n"
            "    area_per_cell = constants['pdac_cancer_cell_cross_section']\n"
            "    cellularity = constants['pdac_cellularity_fraction']\n"
            "    tumor_area = c_cells * area_per_cell / cellularity\n"
            "    cd8_total = cd8_eff + cd8_exh\n"
            "    density = cd8_total / tumor_area\n"
            "    return density.to('cell/mm**2')\n"
        ),
        "units": "cell / millimeter**2",
        "species": ["V_T.CD8", "V_T.CD8_exh", "V_T.C1"],
        "constants": [
            {
                "name": "pdac_cancer_cell_cross_section",
                "value": 0.000227,
                "units": "millimeter**2 / cell",
            },
            {
                "name": "pdac_cellularity_fraction",
                "value": 0.25,
                "units": "dimensionless",
            },
        ],
    },
    "empirical_data": {
        "median": [138.78],
        "ci95": [[20.31, 965.18]],
        "units": "cell / millimeter**2",
        "sample_size": 444,
        "index_values": None,
    },
}

TREATMENT_YAML = {
    "calibration_target_id": "cd8_fold_increase_gvax_nivo_d21",
    "observable": {
        "code": (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    import numpy as np\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    baseline = cd8[0]\n"
            "    eps = 1e-12 * baseline.units\n"
            "    effective_baseline = baseline + eps\n"
            "    fold_change = cd8 / effective_baseline\n"
            "    return fold_change.to('dimensionless')\n"
        ),
        "units": "dimensionless",
        "species": ["V_T.CD8"],
        "constants": [],
    },
    "empirical_data": {
        "median": [3.028],
        "ci95": [[1.4, 6.55]],
        "units": "dimensionless",
        "sample_size": 10,
        "index_values": [21.0],
        "index_unit": "day",
        "index_type": "time",
    },
}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def single_baseline_dir(temp_dir):
    """Directory with a single baseline YAML target."""
    yaml_dir = temp_dir / "targets"
    yaml_dir.mkdir()
    with open(yaml_dir / "m1_m2_ratio.yaml", "w") as f:
        yaml.dump(BASELINE_YAML, f)
    return yaml_dir


@pytest.fixture
def multi_yaml_dir(temp_dir):
    """Directory with three YAML targets (baseline, constants, treatment)."""
    yaml_dir = temp_dir / "targets"
    yaml_dir.mkdir()
    for name, data in [
        ("cd8_density_baseline.yaml", CONSTANTS_YAML),
        ("cd8_fold_increase.yaml", TREATMENT_YAML),
        ("m1_m2_ratio.yaml", BASELINE_YAML),
    ]:
        with open(yaml_dir / name, "w") as f:
            yaml.dump(data, f)
    return yaml_dir


@pytest.fixture
def empty_yaml_dir(temp_dir):
    """Directory with no YAML files."""
    yaml_dir = temp_dir / "empty"
    yaml_dir.mkdir()
    return yaml_dir


@pytest.fixture
def mixed_files_dir(temp_dir):
    """Directory with YAML files and non-YAML files."""
    yaml_dir = temp_dir / "mixed"
    yaml_dir.mkdir()
    with open(yaml_dir / "target.yaml", "w") as f:
        yaml.dump(BASELINE_YAML, f)
    (yaml_dir / "readme.txt").write_text("not a yaml file")
    (yaml_dir / "notes.md").write_text("# notes")
    return yaml_dir


# ============================================================================
# Tests: load_calibration_targets
# ============================================================================


class TestLoadCalibrationTargets:
    def test_load_single_baseline_yaml(self, single_baseline_dir):
        """Load one YAML, check DataFrame columns and values."""
        df = load_calibration_targets(single_baseline_dir)

        assert len(df) == 1
        assert list(df.columns) == [
            "test_statistic_id",
            "required_species",
            "model_output_code",
            "median",
            "ci95_lower",
            "ci95_upper",
            "units",
            "sample_size",
        ]

        row = df.iloc[0]
        assert row["test_statistic_id"] == "m1_m2_ratio"
        assert row["required_species"] == "V_T.Mac_M1,V_T.Mac_M2"
        assert row["median"] == pytest.approx(0.518)
        assert row["ci95_lower"] == pytest.approx(0.071)
        assert row["ci95_upper"] == pytest.approx(3.722)
        assert row["units"] == "dimensionless"
        assert row["sample_size"] == 30

    def test_load_multiple_yamls(self, multi_yaml_dir):
        """Load directory with 3 YAMLs, check row count and IDs."""
        df = load_calibration_targets(multi_yaml_dir)

        assert len(df) == 3
        ids = set(df["test_statistic_id"])
        assert ids == {"m1_m2_ratio", "cd8_density_baseline", "cd8_fold_increase_gvax_nivo_d21"}

    def test_constants_target_fields(self, multi_yaml_dir):
        """Check that constants target has correct species and values."""
        df = load_calibration_targets(multi_yaml_dir)
        cd8_row = df[df["test_statistic_id"] == "cd8_density_baseline"].iloc[0]

        assert "V_T.CD8" in cd8_row["required_species"]
        assert "V_T.CD8_exh" in cd8_row["required_species"]
        assert "V_T.C1" in cd8_row["required_species"]
        assert cd8_row["median"] == pytest.approx(138.78)
        assert cd8_row["sample_size"] == 444

    def test_treatment_target_fields(self, multi_yaml_dir):
        """Check that treatment target has correct values."""
        df = load_calibration_targets(multi_yaml_dir)
        tx_row = df[df["test_statistic_id"] == "cd8_fold_increase_gvax_nivo_d21"].iloc[0]

        assert tx_row["median"] == pytest.approx(3.028)
        assert tx_row["ci95_lower"] == pytest.approx(1.4)
        assert tx_row["ci95_upper"] == pytest.approx(6.55)
        assert tx_row["sample_size"] == 10

    def test_empty_directory_raises(self, empty_yaml_dir):
        """No YAML files raises ValueError."""
        with pytest.raises(ValueError, match="No YAML files"):
            load_calibration_targets(empty_yaml_dir)

    def test_nonexistent_directory_raises(self, temp_dir):
        """Non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_calibration_targets(temp_dir / "nonexistent")

    def test_non_yaml_files_ignored(self, mixed_files_dir):
        """Only .yaml files are loaded; .txt and .md are skipped."""
        df = load_calibration_targets(mixed_files_dir)
        assert len(df) == 1
        assert df.iloc[0]["test_statistic_id"] == "m1_m2_ratio"

    def test_deterministic_ordering(self, multi_yaml_dir):
        """Rows are ordered by sorted filename."""
        df = load_calibration_targets(multi_yaml_dir)
        ids = df["test_statistic_id"].tolist()
        # Files sorted: cd8_density_baseline.yaml, cd8_fold_increase.yaml, m1_m2_ratio.yaml
        assert ids == [
            "cd8_density_baseline",
            "cd8_fold_increase_gvax_nivo_d21",
            "m1_m2_ratio",
        ]


# ============================================================================
# Tests: wrapper code generation (using real Pint)
# ============================================================================


class TestGenerateWrapperCode:
    def test_wrapper_baseline_ratio(self, single_baseline_dir):
        """Wrapper for M1/M2 ratio runs with real Pint quantities at t=0."""
        df = load_calibration_targets(single_baseline_dir)
        code = df.iloc[0]["model_output_code"]

        ns = {}
        exec(code, ns)

        time = np.array([0.0]) * ureg.day
        species_dict = {
            "V_T.Mac_M1": np.array([500.0]) * ureg.cell,
            "V_T.Mac_M2": np.array([1000.0]) * ureg.cell,
        }

        result = ns["compute_test_statistic"](time, species_dict, ureg)
        # 500/1000 = 0.5, scalar at t=0
        assert float(result.magnitude) == pytest.approx(0.5)

    def test_wrapper_with_constants(self, multi_yaml_dir):
        """Wrapper for cd8_density inlines constants and computes correctly."""
        df = load_calibration_targets(multi_yaml_dir)
        cd8_row = df[df["test_statistic_id"] == "cd8_density_baseline"].iloc[0]
        code = cd8_row["model_output_code"]

        ns = {}
        exec(code, ns)

        # Single-timepoint baseline
        time = np.array([0.0]) * ureg.day
        species_dict = {
            "V_T.CD8": np.array([1000.0]) * ureg.cell,
            "V_T.CD8_exh": np.array([200.0]) * ureg.cell,
            "V_T.C1": np.array([1e6]) * ureg.cell,
        }

        result = ns["compute_test_statistic"](time, species_dict, ureg)
        # tumor_area = 1e6 * 0.000227 / 0.25 = 908 mm^2
        # cd8_total = 1200 cells
        # density = 1200 / 908 ≈ 1.3216 cell/mm^2
        expected = 1200.0 / (1e6 * 0.000227 / 0.25)
        assert float(result.magnitude) == pytest.approx(expected, rel=1e-4)

    def test_wrapper_treatment_interp_at_d21(self, multi_yaml_dir):
        """Wrapper for fold-change target interpolates at day 21."""
        df = load_calibration_targets(multi_yaml_dir)
        tx_row = df[df["test_statistic_id"] == "cd8_fold_increase_gvax_nivo_d21"].iloc[0]
        code = tx_row["model_output_code"]

        ns = {}
        exec(code, ns)

        # Time series: days 0, 7, 14, 21
        time = np.array([0.0, 7.0, 14.0, 21.0]) * ureg.day
        # CD8 cells grow from 100 to 300 over 21 days
        species_dict = {
            "V_T.CD8": np.array([100.0, 150.0, 200.0, 300.0]) * ureg.cell,
        }

        result = ns["compute_test_statistic"](time, species_dict, ureg)
        # fold_change at d21 = 300/100 = 3.0
        assert float(result.magnitude) == pytest.approx(3.0, rel=1e-6)

    def test_wrapper_treatment_interp_between_points(self):
        """Wrapper interpolates correctly between time points."""
        code = _generate_wrapper_code(
            observable_code=(
                "def compute_observable(time, species_dict, constants, ureg):\n"
                "    return species_dict['V_T.CD8']\n"
            ),
            constants=[],
            index_values=[15.0],
        )

        ns = {}
        exec(code, ns)

        time = np.array([0.0, 7.0, 14.0, 21.0]) * ureg.day
        species_dict = {
            "V_T.CD8": np.array([1.0, 2.0, 3.0, 4.0]) * ureg.cell,
        }

        result = ns["compute_test_statistic"](time, species_dict, ureg)
        # np.interp(15.0, [0,7,14,21], [1,2,3,4]) = 3 + 1*(1/7) ≈ 3.1429
        expected = np.interp(15.0, [0.0, 7.0, 14.0, 21.0], [1.0, 2.0, 3.0, 4.0])
        assert float(result.magnitude) == pytest.approx(expected, rel=1e-6)

    def test_wrapper_inlines_constants(self):
        """Verify constants appear in generated code."""
        code = _generate_wrapper_code(
            observable_code=(
                "def compute_observable(time, species_dict, constants, ureg):\n"
                "    return species_dict['V_T.C1']\n"
            ),
            constants=[
                {
                    "name": "pdac_cancer_cell_cross_section",
                    "value": 0.000227,
                    "units": "mm**2/cell",
                },
                {"name": "pdac_cellularity_fraction", "value": 0.25, "units": "dimensionless"},
            ],
            index_values=None,
        )

        assert "pdac_cancer_cell_cross_section" in code
        assert "0.000227" in code
        assert "pdac_cellularity_fraction" in code
        assert "0.25" in code
        assert "ureg.parse_expression" in code

    def test_wrapper_extracts_at_target_time(self):
        """Treatment target with index_values evaluates at the correct time."""
        code = _generate_wrapper_code(
            observable_code=(
                "def compute_observable(time, species_dict, constants, ureg):\n"
                "    return species_dict['V_T.CD8']\n"
            ),
            constants=[],
            index_values=[21.0],
        )

        assert "_target_t = 21.0" in code

    def test_wrapper_extracts_at_baseline(self):
        """Baseline target evaluates at t=0."""
        code = _generate_wrapper_code(
            observable_code=(
                "def compute_observable(time, species_dict, constants, ureg):\n"
                "    return species_dict['V_T.CD8']\n"
            ),
            constants=[],
            index_values=None,
        )

        assert "_target_t = 0.0" in code

    def test_wrapper_no_constants_still_passes_empty_dict(self):
        """Wrapper with no constants still passes _constants={} to observable."""
        code = _generate_wrapper_code(
            observable_code=(
                "def compute_observable(time, species_dict, constants, ureg):\n" "    return 42\n"
            ),
            constants=[],
            index_values=None,
        )

        assert "_constants = {}" in code
        assert "compute_observable(time, species_dict, _constants, ureg)" in code

    def test_wrapper_handles_scalar_result(self):
        """Wrapper that returns a plain scalar (no Pint) works correctly."""
        code = _generate_wrapper_code(
            observable_code=(
                "def compute_observable(time, species_dict, constants, ureg):\n" "    return 42.0\n"
            ),
            constants=[],
            index_values=None,
        )

        ns = {}
        exec(code, ns)

        time = np.array([0.0]) * ureg.day
        result = ns["compute_test_statistic"](time, {}, ureg)
        assert result == 42.0


# ============================================================================
# Tests: hash_calibration_targets
# ============================================================================


class TestHashCalibrationTargets:
    def test_hash_deterministic(self, multi_yaml_dir):
        """Same directory produces same hash."""
        h1 = hash_calibration_targets(multi_yaml_dir)
        h2 = hash_calibration_targets(multi_yaml_dir)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_hash_changes_on_file_change(self, single_baseline_dir):
        """Modifying a YAML changes the hash."""
        h1 = hash_calibration_targets(single_baseline_dir)

        # Modify the YAML
        yaml_file = list(single_baseline_dir.glob("*.yaml"))[0]
        data = yaml.safe_load(yaml_file.read_text())
        data["empirical_data"]["median"] = [0.999]
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        h2 = hash_calibration_targets(single_baseline_dir)
        assert h1 != h2

    def test_hash_nonexistent_raises(self, temp_dir):
        """Non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            hash_calibration_targets(temp_dir / "nonexistent")

    def test_hash_empty_dir(self, empty_yaml_dir):
        """Empty directory produces a valid (but deterministic) hash."""
        h = hash_calibration_targets(empty_yaml_dir)
        assert len(h) == 64


# ============================================================================
# Tests: round-trip CSV compatibility
# ============================================================================


class TestRoundTrip:
    def test_round_trip_csv(self, multi_yaml_dir, temp_dir):
        """YAML -> DataFrame -> CSV -> read_csv -> build_test_stat_registry -> compile."""
        from qsp_hpc.batch.derive_test_stats_worker import build_test_stat_registry

        df = load_calibration_targets(multi_yaml_dir)

        # Write to CSV
        csv_path = temp_dir / "test_stats.csv"
        df.to_csv(csv_path, index=False)

        # Read back
        df_read = pd.read_csv(csv_path)

        # Verify columns survived round-trip
        assert "test_statistic_id" in df_read.columns
        assert "model_output_code" in df_read.columns
        assert "required_species" in df_read.columns
        assert "median" in df_read.columns
        assert len(df_read) == 3

        # Build test stat registry (this compiles the wrapper code)
        registry = build_test_stat_registry(df_read)
        assert len(registry) == 3
        assert "m1_m2_ratio" in registry
        assert "cd8_density_baseline" in registry
        assert "cd8_fold_increase_gvax_nivo_d21" in registry

        # Each entry should be a callable function
        for fn in registry.values():
            assert callable(fn)
