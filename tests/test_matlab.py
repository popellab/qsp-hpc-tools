"""
Pytest integration for MATLAB tests.

This module runs MATLAB unit tests as part of the pytest test suite.
Tests are marked with @pytest.mark.matlab and can be skipped if MATLAB
is not available.

Run with:
    pytest tests/test_matlab.py -v
    pytest tests/test_matlab.py -v -m matlab  # Only MATLAB tests

Skip MATLAB tests:
    pytest -m "not matlab"
"""

import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree

import pytest

# Path to MATLAB tests
MATLAB_TEST_DIR = Path(__file__).parent.parent / "qsp_hpc" / "matlab" / "tests"


def matlab_available() -> bool:
    """Check if MATLAB is available on the system."""
    return shutil.which("matlab") is not None


def run_matlab_tests(junit_file: Path) -> subprocess.CompletedProcess:
    """
    Run MATLAB tests and generate JUnit XML output.

    Args:
        junit_file: Path to write JUnit XML results

    Returns:
        CompletedProcess with return code and output
    """
    matlab_cmd = [
        "matlab",
        "-batch",
        f"cd('{MATLAB_TEST_DIR}'); "
        f"results = run_all_tests('JUnitFile', '{junit_file}'); "
        f"exit(results);",
    ]

    return subprocess.run(
        matlab_cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
    )


def parse_junit_xml(junit_file: Path) -> list[dict]:
    """
    Parse JUnit XML file and extract test results.

    Args:
        junit_file: Path to JUnit XML file

    Returns:
        List of test result dicts with keys: name, classname, time, passed, message
    """
    tree = ElementTree.parse(junit_file)
    root = tree.getroot()

    results = []
    for testsuite in root.findall(".//testsuite"):
        for testcase in testsuite.findall("testcase"):
            result = {
                "name": testcase.get("name"),
                "classname": testcase.get("classname"),
                "time": float(testcase.get("time", 0)),
                "passed": True,
                "message": None,
            }

            # Check for failures
            failure = testcase.find("failure")
            if failure is not None:
                result["passed"] = False
                result["message"] = failure.text or failure.get("message", "Test failed")

            # Check for errors
            error = testcase.find("error")
            if error is not None:
                result["passed"] = False
                result["message"] = error.text or error.get("message", "Test error")

            # Check for skipped
            skipped = testcase.find("skipped")
            if skipped is not None:
                result["passed"] = None  # Neither pass nor fail
                result["message"] = skipped.text or skipped.get("message", "Test skipped")

            results.append(result)

    return results


@pytest.fixture(scope="module")
def matlab_test_results(tmp_path_factory):
    """
    Run MATLAB tests once per module and cache results.

    This fixture runs all MATLAB tests in a single MATLAB session
    (for efficiency) and returns the parsed results.
    """
    if not matlab_available():
        pytest.skip("MATLAB not available")

    # Create temp file for JUnit output
    tmp_dir = tmp_path_factory.mktemp("matlab_results")
    junit_file = tmp_dir / "matlab_results.xml"

    # Run MATLAB tests
    result = run_matlab_tests(junit_file)

    if not junit_file.exists():
        pytest.fail(
            f"MATLAB tests did not produce JUnit output.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    # Parse and return results
    return parse_junit_xml(junit_file)


@pytest.mark.matlab
class TestMatlabLoadParameterSamples:
    """Tests for load_parameter_samples_csv.m"""

    def test_loads_multiple_parameters(self, matlab_test_results):
        """Test loading CSV with multiple parameters."""
        result = next(
            (r for r in matlab_test_results if "test_loads_multiple_parameters" in r["name"]),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_parameter_names_correct(self, matlab_test_results):
        """Test that parameter names are correctly extracted."""
        result = next(
            (r for r in matlab_test_results if "test_parameter_names_correct" in r["name"]),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_parameter_values_correct(self, matlab_test_results):
        """Test that parameter values are correctly loaded."""
        result = next(
            (r for r in matlab_test_results if "test_parameter_values_correct" in r["name"]),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_single_parameter(self, matlab_test_results):
        """Test loading CSV with single parameter."""
        result = next(
            (r for r in matlab_test_results if "test_single_parameter" in r["name"]),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]


@pytest.mark.matlab
class TestMatlabExtractSpeciesArrays:
    """Tests for extract_all_species_arrays.m"""

    def test_extracts_species_from_single_simulation(self, matlab_test_results):
        """Test extraction from a single successful simulation."""
        result = next(
            (
                r
                for r in matlab_test_results
                if "test_extracts_species_from_single_simulation" in r["name"]
            ),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_species_names_include_compartment_prefix(self, matlab_test_results):
        """Test that species names are fully qualified."""
        result = next(
            (
                r
                for r in matlab_test_results
                if "test_species_names_include_compartment_prefix" in r["name"]
            ),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_handles_failed_simulation(self, matlab_test_results):
        """Test handling of failed simulations."""
        result = next(
            (r for r in matlab_test_results if "test_handles_failed_simulation" in r["name"]),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_extracts_multiple_simulations(self, matlab_test_results):
        """Test extraction from multiple simulations."""
        result = next(
            (r for r in matlab_test_results if "test_extracts_multiple_simulations" in r["name"]),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_constant_compartment_uses_capacity(self, matlab_test_results):
        """Test that constant compartments use model Capacity directly."""
        result = next(
            (
                r
                for r in matlab_test_results
                if "test_constant_compartment_uses_capacity" in r["name"]
            ),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_nonconstant_compartment_extracted_from_simdata(self, matlab_test_results):
        """Test that non-constant compartments are extracted from simdata."""
        result = next(
            (
                r
                for r in matlab_test_results
                if "test_nonconstant_compartment_extracted_from_simdata" in r["name"]
            ),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]


@pytest.mark.matlab
class TestMatlabSaveSpeciesToParquet:
    """Tests for save_species_to_parquet.m"""

    def test_creates_parquet_file(self, matlab_test_results):
        """Test that Parquet file is created successfully."""
        result = next(
            (r for r in matlab_test_results if "test_creates_parquet_file" in r["name"]),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_multiple_simulations(self, matlab_test_results):
        """Test with multiple simulations."""
        result = next(
            (r for r in matlab_test_results if "test_multiple_simulations" in r["name"]),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_with_parameters(self, matlab_test_results):
        """Test that parameters are included in output."""
        result = next(
            (r for r in matlab_test_results if "test_with_parameters" in r["name"]),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]

    def test_with_real_simbiology_simulation(self, matlab_test_results):
        """Integration test with actual SimBiology simulation."""
        result = next(
            (r for r in matlab_test_results if "test_with_real_simbiology_simulation" in r["name"]),
            None,
        )
        if result is None:
            pytest.skip("Test not found in MATLAB results")
        if result["passed"] is None:
            pytest.skip(result["message"])
        assert result["passed"], result["message"]
