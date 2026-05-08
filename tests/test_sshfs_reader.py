"""Tests for ``qsp_hpc.cpp.sshfs_reader``.

Tests inject a :class:`fsspec.implementations.local.LocalFileSystem`
through the ``filesystem=`` kwarg so they don't need a live SSH server;
the helper's behavior is identical between the two backends from
pyarrow's perspective (both implement the fsspec ``ls`` / ``open`` API
that ``ParquetDataset`` calls into).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsp_hpc.cpp.batch_runner import _write_batch_parquet
from qsp_hpc.cpp.sshfs_reader import sshfs_read_long_form_chunks


@pytest.fixture
def local_fs():
    pytest.importorskip("fsspec")
    from fsspec.implementations.local import LocalFileSystem

    return LocalFileSystem()


def _write_chunk(
    pool_dir: Path,
    chunk_index: int,
    sample_indices: np.ndarray,
    species: list[str] = ("spA", "spB", "spC"),
    n_times: int = 3,
) -> tuple[Path, Path]:
    """Emit one (trajectory, params) pair shaped like batch_runner's writer."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    n_sims = len(sample_indices)
    species = list(species)
    theta = np.arange(n_sims * 2, dtype=np.float64).reshape(n_sims, 2)
    trajectories = []
    time_arrays = []
    for i in range(n_sims):
        sidx = int(sample_indices[i])
        traj = np.zeros((n_times, len(species)), dtype=np.float64)
        for sp_idx in range(len(species)):
            traj[:, sp_idx] = sidx * 100 + sp_idx * 10 + np.arange(n_times)
        trajectories.append(traj)
        time_arrays.append(np.arange(n_times, dtype=np.float64) * 0.1)
    return _write_batch_parquet(
        output_path=pool_dir / f"chunk_{chunk_index:04d}.parquet",
        theta_matrix=theta,
        param_names=["p0", "p1"],
        statuses=[0] * n_sims,
        trajectories=trajectories,
        time_arrays=time_arrays,
        species_names=species,
        compartment_names=[],
        rule_names=[],
        t_end_days=float(n_times) * 0.1,
        min_cadence_hours=2.4,
        sample_indices=sample_indices,
    )


class TestSshfsReadLongFormChunks:
    def test_reads_all_chunks_concatenated(self, tmp_path, local_fs):
        pool = tmp_path / "pool"
        _write_chunk(pool, 0, np.array([0, 1], dtype=np.int64))
        _write_chunk(pool, 1, np.array([2, 3], dtype=np.int64))

        df = sshfs_read_long_form_chunks(str(pool), filesystem=local_fs)

        assert list(df.columns) == ["sample_index", "t_to_diagnosis_days", "column", "value"]
        # 4 sims × 3 species × 3 times = 36 rows.
        assert len(df) == 36
        assert sorted(df["sample_index"].unique()) == [0, 1, 2, 3]
        assert sorted(df["column"].unique()) == ["spA", "spB", "spC"]

    def test_sample_index_filter_pushdown(self, tmp_path, local_fs):
        pool = tmp_path / "pool"
        _write_chunk(pool, 0, np.array([10, 11, 12], dtype=np.int64))
        _write_chunk(pool, 1, np.array([20, 21], dtype=np.int64))

        df = sshfs_read_long_form_chunks(
            str(pool),
            sample_indices=[11, 21],
            filesystem=local_fs,
        )
        assert sorted(df["sample_index"].unique()) == [11, 21]
        # 2 sims × 3 species × 3 times = 18 rows.
        assert len(df) == 18

    def test_traj_columns_filter(self, tmp_path, local_fs):
        pool = tmp_path / "pool"
        _write_chunk(pool, 0, np.array([0, 1], dtype=np.int64))

        df = sshfs_read_long_form_chunks(
            str(pool),
            traj_columns=["spA", "spC"],
            filesystem=local_fs,
        )
        assert sorted(df["column"].unique()) == ["spA", "spC"]
        # 2 sims × 2 species × 3 times = 12 rows.
        assert len(df) == 12
        # Decoded to plain string, not dictionary-typed (pandas may
        # surface this as ``object`` or ``StringDtype`` depending on
        # version — both indicate dict decoding succeeded).
        assert not isinstance(df["column"].dtype, pd.CategoricalDtype)

    def test_combined_filters(self, tmp_path, local_fs):
        pool = tmp_path / "pool"
        _write_chunk(pool, 0, np.array([0, 1, 2], dtype=np.int64))
        _write_chunk(pool, 1, np.array([3, 4], dtype=np.int64))

        df = sshfs_read_long_form_chunks(
            str(pool),
            sample_indices=[1, 4],
            traj_columns=["spB"],
            filesystem=local_fs,
        )
        assert sorted(df["sample_index"].unique()) == [1, 4]
        assert df["column"].unique().tolist() == ["spB"]
        # 2 sims × 1 species × 3 times = 6 rows.
        assert len(df) == 6

    def test_empty_pool_returns_typed_empty_frame(self, tmp_path, local_fs):
        pool = tmp_path / "pool"
        pool.mkdir()

        df = sshfs_read_long_form_chunks(str(pool), filesystem=local_fs)
        assert len(df) == 0
        assert list(df.columns) == ["sample_index", "t_to_diagnosis_days", "column", "value"]
        assert df["sample_index"].dtype == np.int64

    def test_empty_sample_indices_short_circuits(self, tmp_path, local_fs):
        pool = tmp_path / "pool"
        _write_chunk(pool, 0, np.array([0, 1], dtype=np.int64))

        df = sshfs_read_long_form_chunks(
            str(pool),
            sample_indices=[],
            filesystem=local_fs,
        )
        assert len(df) == 0

    def test_empty_traj_columns_short_circuits(self, tmp_path, local_fs):
        pool = tmp_path / "pool"
        _write_chunk(pool, 0, np.array([0, 1], dtype=np.int64))

        df = sshfs_read_long_form_chunks(
            str(pool),
            traj_columns=[],
            filesystem=local_fs,
        )
        assert len(df) == 0

    def test_missing_pool_raises_filenotfound(self, tmp_path, local_fs):
        missing = tmp_path / "nope"
        with pytest.raises(FileNotFoundError):
            sshfs_read_long_form_chunks(str(missing), filesystem=local_fs)

    def test_requires_filesystem_or_host(self, tmp_path):
        with pytest.raises(ValueError, match="filesystem= or sshfs_host="):
            sshfs_read_long_form_chunks(str(tmp_path))

    def test_ignores_non_trajectory_files(self, tmp_path, local_fs):
        pool = tmp_path / "pool"
        _write_chunk(pool, 0, np.array([0, 1], dtype=np.int64))
        # Stray params parquet (sidecar) and unrelated file — must be skipped.
        (pool / "stray.parquet").write_bytes(b"not parquet")
        # The params sidecar already exists from _write_chunk; confirm
        # the helper doesn't pick it up (and would have failed-parse if
        # it had).
        df = sshfs_read_long_form_chunks(str(pool), filesystem=local_fs)
        # 2 sims × 3 species × 3 times = 18 rows from the trajectory
        # parquet only; the stray + sidecar files contribute nothing.
        assert len(df) == 18

    def test_preserves_per_sim_time_vector(self, tmp_path, local_fs):
        pool = tmp_path / "pool"
        sample_indices = np.array([0, 1], dtype=np.int64)
        _write_chunk(pool, 0, sample_indices, n_times=4)

        df = sshfs_read_long_form_chunks(str(pool), filesystem=local_fs)
        for sidx in sample_indices:
            sub = df[(df["sample_index"] == sidx) & (df["column"] == "spA")]
            np.testing.assert_allclose(
                sorted(sub["t_to_diagnosis_days"].tolist()),
                [0.0, 0.1, 0.2, 0.3],
            )
