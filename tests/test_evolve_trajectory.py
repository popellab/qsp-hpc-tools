"""Tests for qsp_hpc.cpp.evolve_trajectory.

Builds synthetic per-sim binary v2 trajectory files (the same format
qsp_sim --evolve-trajectory-out emits) and exercises the assembler.
No qsp_sim dependency.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsp_hpc.cpp.evolve_trajectory import assemble_evolve_trajectory_long
from qsp_hpc.cpp.runner import BinaryFormatError


def _pack_binary(
    traj: np.ndarray,
    dt: float,
    t_end: float,
    n_sp: int,
    n_comp: int = 0,
    n_rules: int = 0,
    magic: int = 0x51535042,
    version: int = 2,
) -> bytes:
    n_t, n_cols = traj.shape
    assert n_cols == n_sp + n_comp + n_rules
    header = struct.pack("<IIQQQQdd", magic, version, n_t, n_sp, n_comp, n_rules, dt, t_end)
    return header + traj.astype("<f8").tobytes()


def _write_fake_sim(
    traj_dir: Path,
    sample_index: int,
    arr: np.ndarray,
    *,
    dt: float,
    t_end: float,
    n_sp: int,
    n_comp: int,
    n_rules: int,
) -> Path:
    p = traj_dir / f"sim_{sample_index:09d}.bin"
    p.write_bytes(_pack_binary(arr, dt=dt, t_end=t_end, n_sp=n_sp, n_comp=n_comp, n_rules=n_rules))
    return p


# Shared schema across the assembler tests.
SPECIES = ["CD8", "Treg", "M1"]
COMPS = ["V_T", "V_LN"]
RULES = ["density_rule"]
N_COLS = len(SPECIES) + len(COMPS) + len(RULES)  # 6


def test_assemble_happy_path(tmp_path: Path) -> None:
    """Three sims with different evolve durations align on t=0 (diagnosis)."""
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    # sim 0: 3 samples at dt=10, evolve ended at t=20 → t_to_diag in {-20, -10, 0}
    arr0 = np.arange(3 * N_COLS, dtype="f8").reshape(3, N_COLS)
    _write_fake_sim(traj_dir, 0, arr0, dt=10.0, t_end=20.0, n_sp=3, n_comp=2, n_rules=1)
    # sim 7: 4 samples at dt=5, evolve ended at t=15 → {-15, -10, -5, 0}
    arr7 = (np.arange(4 * N_COLS, dtype="f8") + 100).reshape(4, N_COLS)
    _write_fake_sim(traj_dir, 7, arr7, dt=5.0, t_end=15.0, n_sp=3, n_comp=2, n_rules=1)
    # sim 42: 2 samples at dt=14, evolve ended at t=14 → {-14, 0}
    arr42 = (np.arange(2 * N_COLS, dtype="f8") + 200).reshape(2, N_COLS)
    _write_fake_sim(traj_dir, 42, arr42, dt=14.0, t_end=14.0, n_sp=3, n_comp=2, n_rules=1)

    df = assemble_evolve_trajectory_long(
        traj_dir,
        species_names=SPECIES,
        compartment_names=COMPS,
        rule_names=RULES,
    )
    assert list(df.columns) == [
        "sample_index",
        "t_to_diagnosis_days",
        "column",
        "value",
    ]
    # Total rows = sum(n_t * n_cols) = (3 + 4 + 2) * 6 = 54
    assert len(df) == 54

    # Each sim's last sample lands at t_to_diag = 0 (diagnosis).
    for idx in (0, 7, 42):
        sim_df = df[df["sample_index"] == idx]
        max_t = sim_df["t_to_diagnosis_days"].max()
        assert max_t == 0.0, f"sim {idx}: max t_to_diag should be 0 (diagnosis)"
    # And the earliest sample is negative (healthy IC, far before diagnosis).
    assert df[df["sample_index"] == 0]["t_to_diagnosis_days"].min() == -20.0
    assert df[df["sample_index"] == 7]["t_to_diagnosis_days"].min() == -15.0

    # Column ordering is species → compartments → rules at each timepoint.
    sim0_t0 = df[(df["sample_index"] == 0) & (df["t_to_diagnosis_days"] == -20.0)].sort_index()
    assert list(sim0_t0["column"]) == SPECIES + COMPS + RULES
    np.testing.assert_array_equal(sim0_t0["value"].to_numpy(), arr0[0])


def test_assemble_columns_subset(tmp_path: Path) -> None:
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    arr = np.arange(2 * N_COLS, dtype="f8").reshape(2, N_COLS)
    _write_fake_sim(traj_dir, 0, arr, dt=10.0, t_end=10.0, n_sp=3, n_comp=2, n_rules=1)
    df = assemble_evolve_trajectory_long(
        traj_dir,
        species_names=SPECIES,
        compartment_names=COMPS,
        rule_names=RULES,
        columns=["CD8", "V_T"],
    )
    assert set(df["column"]) == {"CD8", "V_T"}
    # 2 timepoints × 2 columns = 4 rows
    assert len(df) == 4


def test_assemble_columns_unknown_raises(tmp_path: Path) -> None:
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    arr = np.zeros((1, N_COLS), dtype="f8")
    _write_fake_sim(traj_dir, 0, arr, dt=1.0, t_end=1.0, n_sp=3, n_comp=2, n_rules=1)
    with pytest.raises(KeyError, match="not in species/compartment/rule"):
        assemble_evolve_trajectory_long(
            traj_dir,
            species_names=SPECIES,
            compartment_names=COMPS,
            rule_names=RULES,
            columns=["NotAColumn"],
        )


def test_assemble_sample_indices_filter(tmp_path: Path) -> None:
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    for idx in (0, 1, 2):
        arr = np.full((1, N_COLS), float(idx), dtype="f8")
        _write_fake_sim(traj_dir, idx, arr, dt=1.0, t_end=1.0, n_sp=3, n_comp=2, n_rules=1)
    df = assemble_evolve_trajectory_long(
        traj_dir,
        species_names=SPECIES,
        compartment_names=COMPS,
        rule_names=RULES,
        sample_indices=[0, 2],
    )
    assert set(df["sample_index"]) == {0, 2}


def test_assemble_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        assemble_evolve_trajectory_long(
            tmp_path / "does_not_exist",
            species_names=SPECIES,
            compartment_names=COMPS,
            rule_names=RULES,
        )


def test_assemble_empty_dir_returns_empty_df(tmp_path: Path) -> None:
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    df = assemble_evolve_trajectory_long(
        traj_dir,
        species_names=SPECIES,
        compartment_names=COMPS,
        rule_names=RULES,
    )
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == [
        "sample_index",
        "t_to_diagnosis_days",
        "column",
        "value",
    ]


def test_assemble_skips_unrelated_files(tmp_path: Path) -> None:
    """Non-`sim_*.bin` files in the dir are silently ignored."""
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    arr = np.zeros((1, N_COLS), dtype="f8")
    _write_fake_sim(traj_dir, 0, arr, dt=1.0, t_end=1.0, n_sp=3, n_comp=2, n_rules=1)
    (traj_dir / "README.txt").write_text("ignore me")
    (traj_dir / "checkpoint.pkl").write_bytes(b"junk")
    df = assemble_evolve_trajectory_long(
        traj_dir,
        species_names=SPECIES,
        compartment_names=COMPS,
        rule_names=RULES,
    )
    assert set(df["sample_index"]) == {0}


def test_assemble_column_count_mismatch_raises(tmp_path: Path) -> None:
    """Caller-supplied names must match the binary header column counts."""
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    # Binary has 3 species, but caller supplies only 2 names — should reject.
    arr = np.zeros((1, N_COLS), dtype="f8")
    _write_fake_sim(traj_dir, 0, arr, dt=1.0, t_end=1.0, n_sp=3, n_comp=2, n_rules=1)
    with pytest.raises(BinaryFormatError, match="n_species"):
        assemble_evolve_trajectory_long(
            traj_dir,
            species_names=["CD8", "Treg"],  # short by one
            compartment_names=COMPS,
            rule_names=RULES,
        )
