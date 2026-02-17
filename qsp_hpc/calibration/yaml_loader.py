"""
YAML Calibration Target Loader

Loads calibration target YAML files (from qsp-llm-workflows) and converts them
into the DataFrame format expected by the qsp-hpc-tools pipeline. Each YAML
contains a 4-arg `compute_observable` function, optional constants, and empirical
data with CI95 bounds. The loader generates 3-arg `compute_test_statistic` wrapper
functions that inline constants and extract scalar values at target times.

Public API:
    load_calibration_targets(yaml_dir) -> pd.DataFrame
    hash_calibration_targets(yaml_dir) -> str
"""

import hashlib
import textwrap
from pathlib import Path
from typing import List

import pandas as pd
import yaml


def load_calibration_targets(yaml_dir: Path) -> pd.DataFrame:
    """
    Load calibration target YAMLs from a directory into a test-statistics DataFrame.

    Reads all ``*.yaml`` files, extracts fields, generates 3-arg wrapper code,
    and returns a DataFrame compatible with ``build_test_stat_registry()``.

    Args:
        yaml_dir: Directory containing calibration target YAML files.

    Returns:
        DataFrame with columns:
            test_statistic_id, required_species, model_output_code,
            median, ci95_lower, ci95_upper, units, sample_size

    Raises:
        ValueError: If directory contains no YAML files.
        FileNotFoundError: If yaml_dir does not exist.
    """
    yaml_dir = Path(yaml_dir)
    if not yaml_dir.exists():
        raise FileNotFoundError(f"Calibration targets directory not found: {yaml_dir}")

    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No YAML files found in {yaml_dir}")

    rows: List[dict] = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        target_id = data["calibration_target_id"]
        observable = data["observable"]
        empirical = data["empirical_data"]

        # Extract species list
        species_list = observable.get("species", [])
        required_species = ",".join(species_list) if species_list else ""

        # Extract empirical data (lists with single elements for point-in-time targets)
        median_vals = empirical.get("median", [])
        ci95_vals = empirical.get("ci95", [])

        median_val = median_vals[0] if median_vals else float("nan")

        if ci95_vals and ci95_vals[0]:
            ci95_lower = ci95_vals[0][0]
            ci95_upper = ci95_vals[0][1]
        else:
            ci95_lower = float("nan")
            ci95_upper = float("nan")

        units = observable.get("units", "")
        sample_size = empirical.get("sample_size", float("nan"))

        # Extract index_values for time-indexed targets
        index_values = empirical.get("index_values")

        # Generate wrapper code
        wrapper_code = _generate_wrapper_code(
            observable_code=observable["code"],
            constants=observable.get("constants") or [],
            index_values=index_values,
        )

        rows.append(
            {
                "test_statistic_id": target_id,
                "required_species": required_species,
                "model_output_code": wrapper_code,
                "median": median_val,
                "ci95_lower": ci95_lower,
                "ci95_upper": ci95_upper,
                "units": units,
                "sample_size": sample_size,
            }
        )

    return pd.DataFrame(rows)


def hash_calibration_targets(yaml_dir: Path) -> str:
    """
    Compute a deterministic SHA256 hash of calibration target YAML files.

    Hash is based on sorted filenames and their contents, ensuring consistent
    cache invalidation when targets change.

    Args:
        yaml_dir: Directory containing calibration target YAML files.

    Returns:
        SHA256 hex digest string.

    Raises:
        FileNotFoundError: If yaml_dir does not exist.
    """
    yaml_dir = Path(yaml_dir)
    if not yaml_dir.exists():
        raise FileNotFoundError(f"Calibration targets directory not found: {yaml_dir}")

    hasher = hashlib.sha256()
    yaml_files = sorted(yaml_dir.glob("*.yaml"))

    for yaml_file in yaml_files:
        # Include filename in hash for ordering sensitivity
        hasher.update(yaml_file.name.encode("utf-8"))
        hasher.update(yaml_file.read_bytes())

    return hasher.hexdigest()


def _generate_wrapper_code(
    observable_code: str,
    constants: list,
    index_values: list | None,
) -> str:
    """
    Generate a 3-arg ``compute_test_statistic`` wrapper around a 4-arg observable.

    The wrapper:
    1. Builds a ``_constants`` dict with Pint units from the YAML constants list.
    2. Embeds the original ``compute_observable`` function verbatim.
    3. Calls ``compute_observable(time, species_dict, _constants, ureg)``.
    4. Extracts a scalar value at the target time (from ``index_values`` or t=0).
    5. Returns a scalar Pint Quantity.

    Args:
        observable_code: The ``compute_observable`` function source from the YAML.
        constants: List of constant dicts with ``name``, ``value``, ``units`` keys.
        index_values: List of target time points (e.g., ``[21.0]``), or None/empty
                      for baseline (t=0) targets.

    Returns:
        String containing complete ``compute_test_statistic`` function source.
    """
    lines = ["import numpy as np", "", "def compute_test_statistic(time, species_dict, ureg):"]

    # Build constants dict
    if constants:
        lines.append("    _constants = {}")
        for const in constants:
            name = const["name"]
            value = const["value"]
            units_str = const["units"]
            lines.append(
                f"    _constants[{name!r}] = {value!r} * ureg.parse_expression({units_str!r})"
            )
        lines.append("")
    else:
        lines.append("    _constants = {}")
        lines.append("")

    # Embed the original compute_observable function, indented under the wrapper
    indented_code = textwrap.indent(textwrap.dedent(observable_code), "    ")
    lines.append(indented_code.rstrip())
    lines.append("")

    # Call the observable
    lines.append("    _result = compute_observable(time, species_dict, _constants, ureg)")
    lines.append("")

    # Determine target time
    if index_values and len(index_values) > 0:
        target_t = float(index_values[0])
    else:
        target_t = 0.0

    # Extract scalar at target time
    lines.append(f"    _target_t = {target_t!r}")
    lines.append("    if hasattr(_result, 'magnitude'):")
    lines.append("        _mag = _result.magnitude")
    lines.append("        if hasattr(_mag, '__len__') and len(_mag) > 1:")
    lines.append(
        "            return np.interp(_target_t, time.magnitude, _mag) * _result.units"
    )
    lines.append(
        "        return _mag.item() * _result.units if hasattr(_mag, '__len__') else _result"
    )
    lines.append("    return _result")

    return "\n".join(lines)
