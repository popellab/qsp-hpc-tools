"""Reader / assembler for burn-in trajectory dumps produced by
``qsp_sim --evolve-trajectory-out`` (one binary v2 file per sim).

The C++ side writes one ``sim_<sample_index>.bin`` per simulation in a
caller-supplied directory (``CppBatchRunner(evolve_trajectory_dir=...)``
or ``CppSimulator(evolve_trajectory_dir=...)``). Each file is a standard
v2 binary-trajectory blob (same magic / layout as ``--binary-out``) so
:func:`qsp_hpc.cpp.runner.read_binary_trajectory` parses it without
modification.

This module assembles a directory of those files into a single
long-form pandas DataFrame keyed by ``(sample_index, t_to_diagnosis_days,
column)`` with one ``value`` per row. Time axis is **time-to-diagnosis**:
``t_to_diagnosis_days = t_model - t_diagnosis_days``, so 0 is the diagnosis
sample and earlier rows are negative. This makes posterior-predictive
visualizations across draws with variable evolve durations align cleanly
on the right edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from qsp_hpc.cpp.runner import (
    BinaryFormatError,
    TrajectoryHeader,
    read_binary_trajectory,
)

_SIM_FILENAME_RE = re.compile(r"sim_(\d+)\.bin$")


@dataclass
class EvolveTrajectoryFile:
    """One per-sim trajectory file resolved from an assembler directory."""

    path: Path
    sample_index: int


def _iter_trajectory_files(traj_dir: Path) -> Iterable[EvolveTrajectoryFile]:
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"evolve trajectory dir not found: {traj_dir}")
    for p in sorted(traj_dir.glob("sim_*.bin")):
        m = _SIM_FILENAME_RE.search(p.name)
        if not m:
            continue
        yield EvolveTrajectoryFile(path=p, sample_index=int(m.group(1)))


def assemble_evolve_trajectory_long(
    traj_dir: str | Path,
    species_names: list[str],
    compartment_names: list[str],
    rule_names: list[str],
    columns: Optional[list[str]] = None,
    sample_indices: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Walk per-sim trajectory binaries → long-form DataFrame.

    Args:
        traj_dir: Directory of ``sim_<sample_index>.bin`` files written by
            ``qsp_sim --evolve-trajectory-out``.
        species_names / compartment_names / rule_names: Column-order
            metadata. The binary body is laid out as
            ``[species..., compartments..., rules...]`` and each file's
            header records counts but not names — these come from the
            same ``--species-out`` / ``--compartments-out`` /
            ``--rules-out`` files the post-scenario writer produces.
            Length sums must match each binary's header column count.
        columns: Optional subset of column names to retain. ``None``
            keeps all (species + compartments + rules). Useful when
            only a few cal-target inputs are needed downstream — keeps
            the parquet small.
        sample_indices: Optional iterable of sample-index ints. When
            given, only files matching these indices are read (others
            are silently skipped); rows for missing indices are not
            invented. ``None`` reads everything in ``traj_dir`` matching
            the ``sim_*.bin`` pattern.

    Returns:
        DataFrame with columns
        ``["sample_index", "t_to_diagnosis_days", "column", "value"]``.
        ``t_to_diagnosis_days = t_model - t_diagnosis_days``: 0 is the
        diagnosis sample, earlier rows are negative.

    Raises:
        FileNotFoundError if ``traj_dir`` is missing.
        BinaryFormatError if any file fails the v2 header check or has a
            column-count mismatch with the supplied names.
    """
    traj_dir = Path(traj_dir)
    requested_idx: Optional[set[int]] = (
        set(int(i) for i in sample_indices) if sample_indices is not None else None
    )
    all_names = list(species_names) + list(compartment_names) + list(rule_names)
    if columns is not None:
        col_lookup = {name: i for i, name in enumerate(all_names)}
        missing = [c for c in columns if c not in col_lookup]
        if missing:
            raise KeyError(f"requested column(s) not in species/compartment/rule names: {missing}")
        keep_idx = np.array([col_lookup[c] for c in columns], dtype=np.int64)
        keep_names = list(columns)
    else:
        keep_idx = np.arange(len(all_names), dtype=np.int64)
        keep_names = all_names

    rows: list[pd.DataFrame] = []
    for ftraj in _iter_trajectory_files(traj_dir):
        if requested_idx is not None and ftraj.sample_index not in requested_idx:
            continue
        arr, header = read_binary_trajectory(ftraj.path)
        _validate_header_columns(
            header, len(species_names), len(compartment_names), len(rule_names), ftraj.path
        )
        # Time axis: t_end_days is the diagnosis time; rows are at
        # t_model = i * dt_days (i = 0..n_t-1). Convert to time-to-diagnosis
        # = t_model - t_diagnosis (negative leading up to 0).
        t_model = np.arange(header.n_times, dtype=np.float64) * header.dt_days
        t_to_diag = t_model - header.t_end_days
        sub = arr[:, keep_idx]
        # Long-form: one row per (timepoint, column).
        n_t = header.n_times
        n_c = sub.shape[1]
        long = pd.DataFrame(
            {
                "sample_index": np.full(n_t * n_c, ftraj.sample_index, dtype=np.int64),
                "t_to_diagnosis_days": np.repeat(t_to_diag, n_c),
                "column": np.tile(np.asarray(keep_names, dtype=object), n_t),
                "value": sub.reshape(-1),
            }
        )
        rows.append(long)

    if not rows:
        return pd.DataFrame(columns=["sample_index", "t_to_diagnosis_days", "column", "value"])
    return pd.concat(rows, ignore_index=True)


def _validate_header_columns(
    header: TrajectoryHeader,
    n_species: int,
    n_comp: int,
    n_rules: int,
    path: Path,
) -> None:
    if header.n_species != n_species:
        raise BinaryFormatError(
            f"{path.name}: header n_species={header.n_species} but caller "
            f"supplied {n_species} species names"
        )
    if header.n_compartments != n_comp:
        raise BinaryFormatError(
            f"{path.name}: header n_compartments={header.n_compartments} but "
            f"caller supplied {n_comp} compartment names"
        )
    if header.n_rules != n_rules:
        raise BinaryFormatError(
            f"{path.name}: header n_rules={header.n_rules} but caller "
            f"supplied {n_rules} rule names"
        )


# --- post-scenario trajectory assembler ----------------------------------


_SCENARIO_META_COLS = frozenset({"sample_index", "simulation_id", "status", "time"})


def assemble_post_scenario_trajectory_long(
    species_parquet: str | Path,
    *,
    columns: Optional[list[str]] = None,
    sample_indices: Optional[Iterable[int]] = None,
    drop_failed: bool = True,
) -> pd.DataFrame:
    """Pivot a CppBatchRunner species parquet → long-form trajectory frame.

    Companion to :func:`assemble_evolve_trajectory_long`: that one reads
    per-sim binary files dumped during burn-in; this one reads the
    list-typed columns of a CppBatchRunner Parquet and unpacks them
    along the per-sim ``time`` axis. Both produce long-form
    ``[sample_index, time_days, column, value]`` (note the column name
    is ``time_days`` here, not ``t_to_diagnosis_days`` — post-scenario
    time is non-negative and starts at 0 = scenario start, which is
    diagnosis for evolve-to-diagnosis scenarios).

    Args:
        species_parquet: Path to ``species_*.parquet`` written by
            :class:`qsp_hpc.cpp.batch_runner.CppBatchRunner`. Each row
            represents one simulation; the ``time`` column carries the
            shared time axis (per-sim — usually identical across rows
            since dt/t_end are pool-level), and species/compartment/rule
            columns are list-typed with one entry per timepoint.
        columns: Optional subset of column names to retain (species,
            compartment, or assignment-rule names). ``None`` keeps all
            non-meta columns. Useful when only a few cal-target inputs
            are needed downstream — keeps the output frame small.
        sample_indices: Optional iterable of sample-index ints. When
            given, only rows matching these indices are included.
        drop_failed: When True (default), rows with ``status != 0`` are
            silently dropped. The C++ batch runner writes a row even for
            failed sims (with NaN-padded list cells), so leaving them in
            would produce all-NaN trajectory bands.

    Returns:
        DataFrame with columns ``[sample_index, time_days, column, value]``,
        sorted by ``(sample_index, time_days)`` ascending.

    Raises:
        FileNotFoundError if ``species_parquet`` is missing.
        ValueError if ``columns`` references a name not in the parquet,
            or if a list-typed cell length doesn't match the row's
            ``time`` array.
    """
    species_parquet = Path(species_parquet)
    if not species_parquet.exists():
        raise FileNotFoundError(f"species parquet not found: {species_parquet}")

    df = pd.read_parquet(species_parquet)
    if "time" not in df.columns:
        raise ValueError(
            f"{species_parquet.name}: missing 'time' column (not a "
            "CppBatchRunner species parquet?)"
        )
    if "sample_index" not in df.columns:
        raise ValueError(f"{species_parquet.name}: missing 'sample_index' column")

    if drop_failed and "status" in df.columns:
        df = df[df["status"] == 0]
    if sample_indices is not None:
        wanted = set(int(i) for i in sample_indices)
        df = df[df["sample_index"].isin(wanted)]
    if df.empty:
        return pd.DataFrame(columns=["sample_index", "time_days", "column", "value"])

    data_cols = [
        c for c in df.columns if c not in _SCENARIO_META_COLS and not c.startswith("param:")
    ]
    if columns is not None:
        missing = [c for c in columns if c not in data_cols]
        if missing:
            raise ValueError(f"requested columns not in species parquet: {missing}")
        keep_names = list(columns)
    else:
        keep_names = data_cols

    rows: list[pd.DataFrame] = []
    for _, sim_row in df.iterrows():
        time_arr = np.asarray(sim_row["time"], dtype=np.float64)
        n_t = len(time_arr)
        if n_t == 0:
            continue
        sample_idx = int(sim_row["sample_index"])
        # Stack the requested columns per sim, length-validate against the
        # sim's own time axis.
        per_col_values: list[np.ndarray] = []
        for col in keep_names:
            cell = sim_row[col]
            arr = np.asarray(cell, dtype=np.float64)
            if arr.shape == ():
                # scalar — broadcast across time (parameter-like columns
                # would land here, though we exclude param:* upstream).
                arr = np.full(n_t, float(arr))
            elif arr.shape[0] != n_t:
                raise ValueError(
                    f"{species_parquet.name}: column {col!r} has "
                    f"{arr.shape[0]} entries but time has {n_t} for "
                    f"sample_index={sample_idx}"
                )
            per_col_values.append(arr)
        wide = np.stack(per_col_values, axis=1)  # (n_t, n_cols)
        n_c = wide.shape[1]
        rows.append(
            pd.DataFrame(
                {
                    "sample_index": np.full(n_t * n_c, sample_idx, dtype=np.int64),
                    "time_days": np.repeat(time_arr, n_c),
                    "column": np.tile(np.asarray(keep_names, dtype=object), n_t),
                    "value": wide.reshape(-1),
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["sample_index", "time_days", "column", "value"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["sample_index", "time_days"]).reset_index(drop=True)
