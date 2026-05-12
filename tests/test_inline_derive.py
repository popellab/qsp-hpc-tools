"""Tests for the inline-derive path in the C++ batch worker.

Covers:
- :func:`qsp_hpc.batch.test_stats_compute.derive_chunk_to_csv`: end-to-end
  derive on a single chunk parquet → ``chunk_NNN_{test_stats,params}.csv``
  shards under a per-batch subdir.
- :func:`qsp_hpc.batch.cpp_batch_worker._run_inline_derive`: gated on
  ``test_stats_csv`` config presence; calls the helper with the right
  paths; no-ops when test_stats fields are absent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qsp_hpc.batch.cpp_batch_worker import _run_inline_derive
from qsp_hpc.batch.test_stats_compute import (
    build_test_stat_registry,
    derive_chunk_to_csv,
)


def _write_chunk_parquet(path: Path, n_sims: int = 3) -> None:
    """Write a minimal chunk parquet with the shape inline derive expects.

    Columns: status, time (list<float>), V_T.C1 (list<float>),
    param:k (scalar), sample_index (int).
    """
    time = [0.0, 1.0, 2.0]
    rows = {
        "status": [0] * n_sims,
        "time": [time for _ in range(n_sims)],
        "V_T.C1": [[10.0, 20.0, 30.0] for _ in range(n_sims)],
        "param:k": [0.5 for _ in range(n_sims)],
        "sample_index": list(range(n_sims)),
    }
    table = pa.Table.from_pydict(rows)
    pq.write_table(table, str(path))


def _test_stats_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "test_statistic_id": ["mean_c1"],
            "required_species": ["V_T.C1"],
            "model_output_code": [
                "def compute_test_statistic(time, species_dict):\n"
                "    return float(np.mean(species_dict['V_T.C1']))"
            ],
        }
    )


class TestDeriveChunkToCsv:
    def test_writes_shards_with_correct_shape(self, tmp_path: Path):
        chunk = tmp_path / "chunk_007.parquet"
        _write_chunk_parquet(chunk, n_sims=4)
        out_dir = tmp_path / "out" / "batch_x"

        df = _test_stats_df()
        registry = build_test_stat_registry(df)

        n = derive_chunk_to_csv(
            chunk_parquet=chunk,
            output_dir=out_dir,
            chunk_idx=7,
            test_stats_df=df,
            test_stat_registry=registry,
            species_units={},
        )

        assert n == 4
        ts_csv = out_dir / "chunk_007_test_stats.csv"
        params_csv = out_dir / "chunk_007_params.csv"
        assert ts_csv.exists()
        assert params_csv.exists()

        # All sims have V_T.C1 = [10,20,30] → mean = 20.0; matrix is (4,1).
        ts_matrix = np.loadtxt(ts_csv, delimiter=",", ndmin=2)
        assert ts_matrix.shape == (4, 1)
        np.testing.assert_allclose(ts_matrix[:, 0], 20.0)

        # Params CSV carries sample_index + param columns (no 'param:' prefix).
        params_df = pd.read_csv(params_csv)
        assert list(params_df.columns) == ["sample_index", "k"]
        assert len(params_df) == 4

    def test_no_params_csv_when_no_param_columns(self, tmp_path: Path):
        # Strip param:* columns — derive_chunk_to_csv should skip the
        # params file rather than writing an empty one.
        chunk = tmp_path / "chunk_000.parquet"
        time = [0.0, 1.0]
        table = pa.Table.from_pydict(
            {
                "status": [0, 0],
                "time": [time, time],
                "V_T.C1": [[1.0, 2.0], [3.0, 4.0]],
            }
        )
        pq.write_table(table, str(chunk))
        out_dir = tmp_path / "out"

        df = _test_stats_df()
        registry = build_test_stat_registry(df)
        derive_chunk_to_csv(
            chunk_parquet=chunk,
            output_dir=out_dir,
            chunk_idx=0,
            test_stats_df=df,
            test_stat_registry=registry,
            species_units={},
        )
        assert (out_dir / "chunk_000_test_stats.csv").exists()
        assert not (out_dir / "chunk_000_params.csv").exists()


class TestRunInlineDerive:
    def _config(self, tmp_path: Path) -> dict:
        ts_csv = tmp_path / "test_stats.csv"
        _test_stats_df().to_csv(ts_csv, index=False)
        return {
            "test_stats_csv": str(ts_csv),
            "test_stats_hash": "abcdef01" * 8,
        }

    def test_writes_to_per_batch_subdir(self, tmp_path: Path):
        config = self._config(tmp_path)
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        batch_subdir = "batch_20260512_001122_default_seed7"
        batch_dir = pool_dir / batch_subdir
        batch_dir.mkdir()
        chunk = batch_dir / "chunk_003.parquet"
        _write_chunk_parquet(chunk, n_sims=2)

        _run_inline_derive(
            config=config,
            chunk_parquet=chunk,
            pool_dir=pool_dir,
            batch_subdir=batch_subdir,
            array_idx=3,
        )

        expected = (
            pool_dir
            / "test_stats"
            / config["test_stats_hash"]
            / batch_subdir
            / "chunk_003_test_stats.csv"
        )
        assert expected.exists()

    def test_noop_when_test_stats_missing(self, tmp_path: Path):
        # No test_stats_csv in config → helper returns without writing.
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        chunk = pool_dir / "chunk_000.parquet"
        _write_chunk_parquet(chunk)

        with patch("qsp_hpc.batch.cpp_batch_worker.derive_chunk_to_csv") as m:
            _run_inline_derive(
                config={},
                chunk_parquet=chunk,
                pool_dir=pool_dir,
                batch_subdir="batch_x",
                array_idx=0,
            )
        m.assert_not_called()
        assert not (pool_dir / "test_stats").exists()
