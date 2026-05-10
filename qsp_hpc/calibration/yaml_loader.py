"""
YAML Calibration / Prediction Target Loader

Loads calibration target YAML files (from MAPLE) and converts them
into the DataFrame format expected by the qsp-hpc-tools pipeline. Each YAML
contains a 3-arg ``compute_observable(time, species_dict, constants)`` function,
optional constants, and empirical data with CI95 bounds. The loader generates
2-arg ``compute_test_statistic(time, species_dict)`` wrapper functions that
inline constants and extract scalar values at target times.

All values flowing through the wrapper are raw floats / numpy arrays in
canonical model_structure.json units; observable code is responsible for any
inline numerical conversions (mirroring SubmodelTarget forward_model
convention).

Prediction targets (``prediction_target_id`` / no empirical block) share the
observable-code protocol but carry no measurement. They are loaded through a
parallel API that emits the same DataFrame schema — empirical columns are NaN
and the ``is_prediction_only`` flag is ``True`` — so calibration and prediction
rows can be concatenated and handed to ``build_test_stat_registry`` /
``compute_test_statistics_batch`` unchanged.

Public API:
    load_calibration_targets(yaml_dir) -> pd.DataFrame
    hash_calibration_targets(yaml_dir) -> str
    load_prediction_targets(yaml_dir) -> pd.DataFrame
    hash_prediction_targets(yaml_dir) -> str
"""

import hashlib
import textwrap
from pathlib import Path
from typing import List

import pandas as pd
import yaml

# Column order used by both loaders (calibration-only columns are the
# canonical layout; the prediction loader appends ``is_prediction_only``
# for callers that need to distinguish rows after concat).
#
# NOTE: ``load_calibration_targets`` must preserve the pre-2026-04
# column set byte-for-byte when serialized to CSV — the HPC test-stats
# pool is keyed by ``compute_test_stats_hash(csv)`` and any schema change
# invalidates every existing derived-stats cache. The ``is_prediction_only``
# marker is therefore only emitted by ``load_prediction_targets`` and
# back-filled on calibration rows at concat time (see CppSimulator
# ``_load_test_stats_df``).
_CALIBRATION_COLUMNS = [
    "test_statistic_id",
    "required_species",
    "model_output_code",
    "median",
    "ci95_lower",
    "ci95_upper",
    "units",
    "sample_size",
]
_PREDICTION_COLUMNS = [*_CALIBRATION_COLUMNS, "is_prediction_only"]


def _resolve_yaml_dirs(yaml_dir: Path | str | List) -> List[Path]:
    """Normalize the ``yaml_dir`` argument into a list of existing directories.

    Accepts a single Path/str or a list of Path/str entries. Each entry must
    exist; raises ``FileNotFoundError`` on the first that doesn't, naming the
    bad path so callers can fix the misconfiguration.
    """
    if isinstance(yaml_dir, (str, Path)):
        dirs = [Path(yaml_dir)]
    else:
        dirs = [Path(d) for d in yaml_dir]
        if not dirs:
            raise ValueError("yaml_dir must be a Path or a non-empty list of Paths")

    for d in dirs:
        if not d.exists():
            raise FileNotFoundError(f"Calibration targets directory not found: {d}")
    return dirs


def _gather_yaml_files(dirs: List[Path]) -> List[Path]:
    """Return ``*.yaml`` files across all dirs, sorted by basename for
    deterministic ordering across union sets that mix scenario + mechanistic
    directories.

    Detects basename collisions across directories — two same-named YAMLs
    produce ambiguous test_statistic_ids downstream — and raises before
    silently overriding either.
    """
    seen: dict[str, Path] = {}
    for d in dirs:
        for f in sorted(d.glob("*.yaml")):
            if f.name in seen:
                raise ValueError(
                    f"Duplicate YAML basename '{f.name}' across calibration "
                    f"target directories: {seen[f.name]} and {f}. "
                    "Rename one to disambiguate."
                )
            seen[f.name] = f
    return [seen[name] for name in sorted(seen)]


def load_calibration_targets(yaml_dir: Path | str | List) -> pd.DataFrame:
    """
    Load calibration target YAMLs from one or more directories into a
    test-statistics DataFrame.

    Reads all ``*.yaml`` files, extracts fields, generates 3-arg wrapper code,
    and returns a DataFrame compatible with ``build_test_stat_registry()``.

    Args:
        yaml_dir: Directory (Path or str) containing calibration target YAML
            files, OR a list of such directories. When a list is passed, the
            union of YAML files is loaded; basename collisions across dirs
            raise ``ValueError`` to prevent ambiguous test_statistic_ids.

            The list form is intended for splitting literature targets and
            mechanistic-prior targets into parallel directory trees per
            scenario — e.g.
            ``["calibration_targets/clinical_progression",
               "calibration_targets/mechanistic/clinical_progression"]``.

    Returns:
        DataFrame with columns:
            test_statistic_id, required_species, model_output_code,
            median, ci95_lower, ci95_upper, units, sample_size

        The ``is_prediction_only`` marker used to separate prediction rows
        from calibration rows is intentionally *not* emitted here — see
        the note on ``_CALIBRATION_COLUMNS`` above for why CSV stability
        matters — and is back-filled by callers (CppSimulator) at concat.

    Raises:
        ValueError: If no YAML files are found across the provided directories,
            or if two directories contain a YAML with the same basename.
        FileNotFoundError: If any provided directory does not exist.
    """
    dirs = _resolve_yaml_dirs(yaml_dir)
    yaml_files = _gather_yaml_files(dirs)
    if not yaml_files:
        joined = ", ".join(str(d) for d in dirs)
        raise ValueError(f"No YAML files found in {joined}")

    rows: List[dict] = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        target_id = data["calibration_target_id"]
        observable = data["observable"]
        empirical = data["empirical_data"]

        # Extract species list
        species_list = list(observable.get("species", []) or [])
        # Auxiliary parameter names ride alongside species in
        # required_species so the derive worker resolves them via the
        # same `param:<name>` / `species_dict[name]` rail used for QSP
        # parameters and species. The wrapper code (below) then moves
        # them into _constants under the YAML-declared units.
        aux_params = observable.get("auxiliary_parameters") or []
        aux_names = [a["name"] for a in aux_params]
        all_required = species_list + aux_names
        required_species = ",".join(all_required) if all_required else ""

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
            auxiliary_parameters=aux_params,
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

    return pd.DataFrame(rows, columns=_CALIBRATION_COLUMNS)


def hash_calibration_targets(yaml_dir: Path | str | List) -> str:
    """
    Compute a deterministic SHA256 hash of calibration target YAML files.

    Hash is based on sorted filenames and their contents, ensuring consistent
    cache invalidation when targets change. When ``yaml_dir`` is a list,
    files are unioned across all dirs and sorted by basename — same as
    :func:`load_calibration_targets`, so the hash matches the loaded set.

    Args:
        yaml_dir: Directory or list of directories containing calibration
            target YAML files (see :func:`load_calibration_targets`).

    Returns:
        SHA256 hex digest string.

    Raises:
        FileNotFoundError: If any provided directory does not exist.
        ValueError: If two directories contain a YAML with the same basename.
    """
    dirs = _resolve_yaml_dirs(yaml_dir)
    yaml_files = _gather_yaml_files(dirs)

    hasher = hashlib.sha256()
    for yaml_file in yaml_files:
        # Include filename in hash for ordering sensitivity
        hasher.update(yaml_file.name.encode("utf-8"))
        hasher.update(yaml_file.read_bytes())

    return hasher.hexdigest()


def load_prediction_targets(yaml_dir: Path) -> pd.DataFrame:
    """Load PredictionTarget YAMLs into the same test-statistics DataFrame shape.

    Each YAML must define the PredictionTarget schema (see
    ``pdac-build/scripts/prediction_target.py``): ``prediction_target_id``,
    ``observable.code`` / ``observable.units`` / ``observable.species`` /
    ``observable.constants``, optional ``index_values`` / ``index_unit`` /
    ``index_type``. No empirical data and no provenance — those columns come
    back NaN so calibration and prediction rows can be concatenated into a
    single registry.

    Args:
        yaml_dir: Directory containing prediction target YAML files.

    Returns:
        DataFrame with the same columns as ``load_calibration_targets``:
            test_statistic_id, required_species, model_output_code,
            median=NaN, ci95_lower=NaN, ci95_upper=NaN, units, sample_size=NaN,
            is_prediction_only=True.

    Raises:
        ValueError: If directory contains no YAML files.
        FileNotFoundError: If yaml_dir does not exist.
    """
    yaml_dir = Path(yaml_dir)
    if not yaml_dir.exists():
        raise FileNotFoundError(f"Prediction targets directory not found: {yaml_dir}")

    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No YAML files found in {yaml_dir}")

    rows: List[dict] = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        # PredictionTarget IDs live under `prediction_target_id` — the
        # test_statistic_id column is the common name used by the derive
        # worker, so we rename here rather than forking the schema.
        target_id = data["prediction_target_id"]
        observable = data["observable"]

        species_list = list(observable.get("species", []) or [])
        aux_params = observable.get("auxiliary_parameters") or []
        aux_names = [a["name"] for a in aux_params]
        all_required = species_list + aux_names
        required_species = ",".join(all_required) if all_required else ""
        units = observable.get("units", "")

        # Prediction targets keep their index_values at the top level
        # (PredictionTarget schema), not under empirical_data. Fall back
        # to None so the wrapper code evaluates at t=0 when absent.
        index_values = data.get("index_values")

        wrapper_code = _generate_wrapper_code(
            observable_code=observable["code"],
            constants=observable.get("constants") or [],
            index_values=index_values,
            auxiliary_parameters=aux_params,
        )

        rows.append(
            {
                "test_statistic_id": target_id,
                "required_species": required_species,
                "model_output_code": wrapper_code,
                "median": float("nan"),
                "ci95_lower": float("nan"),
                "ci95_upper": float("nan"),
                "units": units,
                "sample_size": float("nan"),
                "is_prediction_only": True,
            }
        )

    return pd.DataFrame(rows, columns=_PREDICTION_COLUMNS)


def hash_prediction_targets(yaml_dir: Path) -> str:
    """Deterministic SHA256 of prediction-target YAMLs (parallel to
    :func:`hash_calibration_targets`).

    The digest feeds into theta-scoped pool / cache keys alongside the
    calibration-target hash so edits to prediction YAMLs invalidate the
    suffix pool — otherwise a stale cache would hand back endpoint columns
    computed against the old observable code.
    """
    yaml_dir = Path(yaml_dir)
    if not yaml_dir.exists():
        raise FileNotFoundError(f"Prediction targets directory not found: {yaml_dir}")

    hasher = hashlib.sha256()
    yaml_files = sorted(yaml_dir.glob("*.yaml"))

    for yaml_file in yaml_files:
        hasher.update(yaml_file.name.encode("utf-8"))
        hasher.update(yaml_file.read_bytes())

    return hasher.hexdigest()


def _generate_wrapper_code(
    observable_code: str,
    constants: list,
    index_values: list | None,
    auxiliary_parameters: list | None = None,
) -> str:
    """
    Generate a 2-arg ``compute_test_statistic`` wrapper around a 3-arg observable.

    The wrapper:
    1. Builds a ``_constants`` dict of raw float values in YAML-declared units.
    2. Resolves any declared auxiliary parameters from ``species_dict`` (the
       derive worker joins per-sim aux draws into ``species_dict`` from the
       aux samples sidecar; see ``derive_test_stats_worker``) into ``_constants``.
    3. Embeds the original ``compute_observable`` function verbatim.
    4. Calls ``compute_observable(time, species_dict, _constants)``.
    5. Extracts a scalar at the target time (from ``index_values`` or t=0).

    All values are raw floats in their canonical / declared units; observable
    code is responsible for any inline numerical conversions (matching the
    submodel forward-model precedent).

    Args:
        observable_code: The ``compute_observable`` function source from the YAML.
        constants: List of constant dicts with ``name``, ``value``, ``units`` keys.
            ``units`` is documentation only — the value is emitted as-is.
        index_values: List of target time points (e.g., ``[21.0]``), or None/empty
                      for baseline (t=0) targets.
        auxiliary_parameters: Optional list of aux parameter dicts from
            ``observable.auxiliary_parameters``. When non-empty, the wrapper
            relocates ``species_dict[aux_name]`` (a raw float in the units
            declared by ``auxiliary_units`` on the derive side) into
            ``_constants[aux_name]``. Caller is responsible for appending
            aux names to ``required_species``.

    Returns:
        String containing complete ``compute_test_statistic`` function source.
    """
    lines = ["import numpy as np", "", "def compute_test_statistic(time, species_dict):"]

    if constants:
        lines.append("    _constants = {}")
        for const in constants:
            name = const["name"]
            value = const["value"]
            units_str = const["units"]
            # units kept in a comment for traceability; observable.code is
            # authored knowing the declared unit per constant.
            lines.append(f"    _constants[{name!r}] = {value!r}  # {units_str}")
        lines.append("")
    else:
        lines.append("    _constants = {}")
        lines.append("")

    if auxiliary_parameters:
        for aux in auxiliary_parameters:
            aux_name = aux["name"]
            aux_units = aux["units"]
            lines.append(
                f"    _constants[{aux_name!r}] = species_dict[{aux_name!r}]  # {aux_units}"
            )
        lines.append("")

    # Embed the original compute_observable function, indented under the wrapper
    indented_code = textwrap.indent(textwrap.dedent(observable_code), "    ")
    lines.append(indented_code.rstrip())
    lines.append("")

    lines.append("    _result = compute_observable(time, species_dict, _constants)")
    lines.append("")

    if index_values and len(index_values) > 0:
        target_t = float(index_values[0])
    else:
        target_t = 0.0

    # Extract scalar at target time. Result is a raw float or numpy array.
    lines.append(f"    _target_t = {target_t!r}")
    lines.append("    _arr = np.asarray(_result, dtype=float)")
    lines.append("    if _arr.ndim > 0 and _arr.size > 1:")
    lines.append("        return float(np.interp(_target_t, np.asarray(time, dtype=float), _arr))")
    lines.append("    return float(_arr.item()) if _arr.size == 1 else float(_arr)")

    return "\n".join(lines)
