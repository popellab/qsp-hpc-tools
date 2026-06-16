"""
Cross-Scenario Calibration Target Loader.

Loads CrossScenarioCalibrationTarget YAMLs (defined in maple) and produces
a DataFrame parallel in spirit to ``load_calibration_targets``, but with
inputs serialized as JSON (a list of role-keyed dicts). Each input is
self-contained: it carries its own per-arm observable (``observable_code`` +
``required_species``) computed at derive time on the worker, rather than
referencing a per-scenario calibration target.

The runtime composer worker
(``qsp_hpc.batch.cross_scenario_worker.compute_cross_scenario_statistics``)
consumes this DataFrame plus the aligned ``scenario_meta`` produced by
the SBI runner and emits one column per cross-scenario target.

Public API:
    load_cross_scenario_targets(yaml_dir) -> pd.DataFrame
    hash_cross_scenario_targets(yaml_dir) -> str
"""

import hashlib
import json
from pathlib import Path
from typing import List

import pandas as pd
import yaml

# Column order is the canonical layout. New columns must be appended (not
# inserted) to preserve hash-pinned cache stability across qsp-hpc-tools
# upgrades.
_CROSS_SCENARIO_COLUMNS = [
    "cross_scenario_target_id",
    "composer_code",
    "units",
    "inputs_json",  # JSON-encoded list[dict]: [{role, scenario, observable_code, required_species}, ...]
    "median",
    "ci95_lower",
    "ci95_upper",
    "sample_size",
]


def _resolve_yaml_dir(yaml_dir: Path | str) -> Path:
    yaml_dir = Path(yaml_dir)
    if not yaml_dir.exists():
        raise FileNotFoundError(f"Cross-scenario targets directory not found: {yaml_dir}")
    return yaml_dir


def _normalize_input(raw: dict) -> dict:
    """Normalize one self-contained per-arm input into the serialized JSON
    payload: ``{role, scenario, observable_code, required_species}``.

    Full validation happens at YAML authoring time via maple's
    CrossScenarioInput Pydantic model. We re-check the minimum invariants
    here so a hand-edited YAML doesn't sail past with a silently broken
    derive/compose at runtime. Legacy ``input_kind`` / ``test_statistic_id``
    fields are rejected loudly so stale YAMLs don't load as no-ops.
    """
    role = raw.get("role")
    scenario = raw.get("scenario")
    if not role or not scenario:
        raise ValueError(
            f"Cross-scenario input is missing required 'role'/'scenario' fields: {raw}"
        )

    if "input_kind" in raw or "test_statistic_id" in raw:
        raise ValueError(
            f"Cross-scenario input role='{role}' uses the retired "
            "'input_kind'/'test_statistic_id' schema. Each input must now "
            "carry its own 'observable_code' + 'required_species' (see "
            "maple CrossScenarioInput)."
        )

    observable_code = raw.get("observable_code")
    if not observable_code:
        raise ValueError(
            f"Cross-scenario input role='{role}' has no observable_code. Each "
            "input must define compute_test_statistic(time, species_dict)."
        )

    species = raw.get("required_species") or []
    if not species:
        raise ValueError(f"Cross-scenario input role='{role}' has no required_species.")

    return {
        "role": role,
        "scenario": scenario,
        "observable_code": observable_code,
        "required_species": list(species),
    }


def load_cross_scenario_targets(yaml_dir: Path | str) -> pd.DataFrame:
    """
    Load CrossScenarioCalibrationTarget YAMLs into a DataFrame.

    Args:
        yaml_dir: Directory containing cross-scenario calibration target
            YAML files.

    Returns:
        DataFrame with columns:
            cross_scenario_target_id, composer_code, units, inputs_json,
            median, ci95_lower, ci95_upper, sample_size

        ``inputs_json`` is a JSON-encoded list of dicts; each dict has
        ``role``, ``scenario``, ``observable_code``, and ``required_species``.

    Raises:
        FileNotFoundError: If yaml_dir does not exist.
        ValueError: If no YAMLs are found, or any YAML has invalid input
            shape (missing role/scenario/observable_code/required_species,
            or a retired input_kind/test_statistic_id field).
    """
    yaml_dir = _resolve_yaml_dir(yaml_dir)
    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No YAML files found in {yaml_dir}")

    rows: List[dict] = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        target_id = data["cross_scenario_target_id"]
        observable = data["observable"]
        empirical = data["empirical_data"]

        composer_code = observable["code"]
        units = observable["units"]

        normalized_inputs = [_normalize_input(inp) for inp in observable.get("inputs", [])]
        if len(normalized_inputs) < 2:
            raise ValueError(
                f"Cross-scenario target '{target_id}' has fewer than 2 inputs; "
                "a cross-scenario observable must compose at least two scenarios."
            )

        # Empirical data is expected to be a single-element distribution
        # (cross-scenario targets are scalars in practice). Vector-valued
        # cross-scenario targets aren't supported in this round — they
        # would need an index_values plumbing pass through the composer.
        median_vals = empirical.get("median", [])
        ci95_vals = empirical.get("ci95", [])

        median_val = median_vals[0] if median_vals else float("nan")
        if ci95_vals and ci95_vals[0]:
            ci95_lower = ci95_vals[0][0]
            ci95_upper = ci95_vals[0][1]
        else:
            ci95_lower = float("nan")
            ci95_upper = float("nan")

        sample_size = empirical.get("sample_size", float("nan"))
        if isinstance(sample_size, list):
            # Vector sample_size doesn't apply to cross-scenario scalar
            # targets; reject loudly rather than pick the first element.
            raise ValueError(
                f"Cross-scenario target '{target_id}' has vector "
                "sample_size; only scalar sample_size is supported."
            )

        rows.append(
            {
                "cross_scenario_target_id": target_id,
                "composer_code": composer_code,
                "units": units,
                "inputs_json": json.dumps(normalized_inputs),
                "median": median_val,
                "ci95_lower": ci95_lower,
                "ci95_upper": ci95_upper,
                "sample_size": sample_size,
            }
        )

    return pd.DataFrame(rows, columns=_CROSS_SCENARIO_COLUMNS)


def hash_cross_scenario_targets(yaml_dir: Path | str) -> str:
    """
    Deterministic SHA256 of cross-scenario target YAMLs.

    Parallel to :func:`hash_calibration_targets`. Feeds into pool /
    derive cache keys alongside the per-scenario calibration-target hashes
    so edits to cross-scenario YAMLs invalidate downstream caches even
    though their schema lives outside the per-scenario test-stats CSV.
    """
    yaml_dir = _resolve_yaml_dir(yaml_dir)
    yaml_files = sorted(yaml_dir.glob("*.yaml"))

    hasher = hashlib.sha256()
    for yaml_file in yaml_files:
        hasher.update(yaml_file.name.encode("utf-8"))
        hasher.update(yaml_file.read_bytes())
    return hasher.hexdigest()
