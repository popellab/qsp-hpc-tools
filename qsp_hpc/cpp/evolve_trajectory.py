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
