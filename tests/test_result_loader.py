"""Tests for QSPResultLoader."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsp_hpc.simulation.result_loader import (
    HASH_PREFIX_LENGTH,
    QSPResultLoader,
)

PRIORS_CSV = "name,median,units,distribution\nk1,0.1,1/day,lognormal\nk2,1.0,1/day,lognormal\n"
TEST_STATS_CSV = "test_statistic_id,median,ci95_lower,ci95_upper\nobs1,1.0,0.5,2.0\n"


def _write_pool_layout(
    tmp_path: Path,
    *,
    model_version: str,
    scenario: str,
    priors_csv_text: str,
    test_stats_csv_text: str,
    params_rows: int,
    stats_rows: int,
    n_cols_params: int = 3,
    n_cols_stats: int = 2,
) -> tuple[Path, Path, Path]:
    """Create a pool dir + combined files with the correct hashed paths."""
    model_script = "dummy_model_script"

    # Files the loader reads
    priors_file = tmp_path / "priors.csv"
    priors_file.write_text(priors_csv_text)
    stats_csv_file = tmp_path / "test_stats.csv"
    stats_csv_file.write_text(test_stats_csv_text)

    # Compute hashes the same way the loader does
    priors_h = hashlib.sha256()
    priors_h.update(priors_csv_text.encode("utf-8"))
    priors_h.update(model_script.encode("utf-8"))
    priors_h.update(model_version.encode("utf-8"))
    priors_digest = priors_h.hexdigest()
    stats_digest = hashlib.sha256(test_stats_csv_text.encode("utf-8")).hexdigest()

    pool_root = tmp_path / "pool_root"
    pool_id = f"{model_version}_{priors_digest[:HASH_PREFIX_LENGTH]}_{scenario}"
    ts_dir = pool_root / pool_id / "test_stats" / stats_digest
    ts_dir.mkdir(parents=True)

    # combined_params.csv: header + rows
    params = pd.DataFrame(
        np.arange(params_rows * n_cols_params, dtype=float).reshape(params_rows, n_cols_params),
        columns=[f"p{i}" for i in range(n_cols_params)],
    )
    params.to_csv(ts_dir / "combined_params.csv", index=False)

    # combined_test_stats.csv: NO header
    stats = pd.DataFrame(
        np.arange(stats_rows * n_cols_stats, dtype=float).reshape(stats_rows, n_cols_stats)
    )
    stats.to_csv(ts_dir / "combined_test_stats.csv", index=False, header=False)

    return pool_root, priors_file, stats_csv_file


def test_loader_reads_combined_files(tmp_path):
    model_version = "mv1"
    scenario = "scen_a"
    pool_root, priors_file, stats_csv = _write_pool_layout(
        tmp_path,
        model_version=model_version,
        scenario=scenario,
        priors_csv_text=PRIORS_CSV,
        test_stats_csv_text=TEST_STATS_CSV,
        params_rows=50,
        stats_rows=50,
    )

    loader = QSPResultLoader(
        pool_root=pool_root,
        priors_csv=priors_file,
        test_stats_csv=stats_csv,
        model_version=model_version,
        model_script="dummy_model_script",
        scenario=scenario,
    )

    params, obs, sample_index = loader.load(n_simulations=20)
    assert params.shape == (20, 3)
    assert obs.shape == (20, 2)
    assert sample_index.shape == (20,)
    # Sanity: first row of params should be [0, 1, 2]
    np.testing.assert_array_equal(params[0], [0.0, 1.0, 2.0])
    # Legacy fixtures lack sample_index column → loader falls back to range.
    np.testing.assert_array_equal(sample_index, np.arange(20))


def test_loader_raises_when_combined_missing(tmp_path):
    pool_root = tmp_path / "pool_root"
    priors_file = tmp_path / "priors.csv"
    priors_file.write_text(PRIORS_CSV)
    stats_csv = tmp_path / "test_stats.csv"
    stats_csv.write_text(TEST_STATS_CSV)

    loader = QSPResultLoader(
        pool_root=pool_root,
        priors_csv=priors_file,
        test_stats_csv=stats_csv,
        model_version="mv1",
        model_script="dummy",
        scenario="missing_scenario",
    )

    with pytest.raises(FileNotFoundError):
        loader.load(n_simulations=10)


def test_loader_raises_when_short(tmp_path):
    model_version = "mv1"
    scenario = "scen_short"
    pool_root, priors_file, stats_csv = _write_pool_layout(
        tmp_path,
        model_version=model_version,
        scenario=scenario,
        priors_csv_text=PRIORS_CSV,
        test_stats_csv_text=TEST_STATS_CSV,
        params_rows=5,
        stats_rows=5,
    )

    loader = QSPResultLoader(
        pool_root=pool_root,
        priors_csv=priors_file,
        test_stats_csv=stats_csv,
        model_version=model_version,
        model_script="dummy_model_script",
        scenario=scenario,
    )

    with pytest.raises(ValueError, match="need 20"):
        loader.load(n_simulations=20)


def test_loader_rejects_both_test_stats_and_calibration(tmp_path):
    priors_file = tmp_path / "priors.csv"
    priors_file.write_text(PRIORS_CSV)
    stats_csv = tmp_path / "test_stats.csv"
    stats_csv.write_text(TEST_STATS_CSV)

    with pytest.raises(ValueError, match="test_stats_csv OR calibration_targets"):
        QSPResultLoader(
            pool_root=tmp_path / "pool",
            priors_csv=priors_file,
            test_stats_csv=stats_csv,
            calibration_targets=tmp_path / "cal_targets",
            model_version="mv1",
            model_script="dummy",
            scenario="s",
        )


def test_loader_rejects_neither_source(tmp_path):
    priors_file = tmp_path / "priors.csv"
    priors_file.write_text(PRIORS_CSV)

    with pytest.raises(ValueError, match="Must provide"):
        QSPResultLoader(
            pool_root=tmp_path / "pool",
            priors_csv=priors_file,
            model_version="mv1",
            model_script="dummy",
            scenario="s",
        )
