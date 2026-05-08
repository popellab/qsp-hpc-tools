"""Tests for qsp_hpc.simulation.simulation_batch.SimulationBatch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsp_hpc.simulation.simulation_batch import SimulationBatch


def _make_batch(n_sims: int = 4, sample_index_offset: int = 0) -> SimulationBatch:
    theta = np.arange(n_sims * 2, dtype=np.float64).reshape(n_sims, 2)
    sample_index = np.arange(n_sims, dtype=np.int64) + sample_index_offset
    rows = []
    for sidx in sample_index:
        for sp in ("spA", "spB"):
            for t in (0.0, 0.1):
                rows.append(dict(sample_index=int(sidx), time=t, species=sp, value=float(sidx) + t))
    traj_df = pd.DataFrame(rows)
    return SimulationBatch(
        theta=theta,
        sample_index=sample_index,
        traj_df=traj_df,
        species_units={"spA": "cell", "spB": "nM"},
        param_names=["A", "B"],
        pool_id="abcd1234",
    )


class TestSimulationBatchValidation:
    def test_basic_construction(self):
        b = _make_batch()
        assert b.n_sims == 4
        assert b.theta.shape == (4, 2)
        assert b.pool_id == "abcd1234"

    def test_rejects_1d_theta(self):
        with pytest.raises(ValueError, match="2-D"):
            SimulationBatch(
                theta=np.zeros(4),
                sample_index=np.arange(4, dtype=np.int64),
                traj_df=pd.DataFrame(),
                species_units={},
                param_names=["A"],
                pool_id="x",
            )

    def test_rejects_sample_index_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            SimulationBatch(
                theta=np.zeros((4, 2)),
                sample_index=np.arange(3, dtype=np.int64),
                traj_df=pd.DataFrame(),
                species_units={},
                param_names=["A", "B"],
                pool_id="x",
            )

    def test_rejects_param_names_mismatch(self):
        with pytest.raises(ValueError, match="param_names"):
            SimulationBatch(
                theta=np.zeros((4, 2)),
                sample_index=np.arange(4, dtype=np.int64),
                traj_df=pd.DataFrame(),
                species_units={},
                param_names=["A"],
                pool_id="x",
            )


class TestSliceToIndices:
    def test_slice_to_subset(self):
        b = _make_batch(n_sims=4)
        sliced = b.slice_to_indices([0, 2])
        assert sliced.n_sims == 2
        np.testing.assert_array_equal(sliced.sample_index, [0, 2])
        # theta rows match the corresponding original positions
        np.testing.assert_array_equal(sliced.theta[0], b.theta[0])
        np.testing.assert_array_equal(sliced.theta[1], b.theta[2])
        # traj_df only contains rows for sample_index in [0, 2]
        assert set(sliced.traj_df["sample_index"].unique()) == {0, 2}

    def test_slice_preserves_request_order(self):
        b = _make_batch(n_sims=4)
        sliced = b.slice_to_indices([3, 1])
        np.testing.assert_array_equal(sliced.sample_index, [3, 1])
        np.testing.assert_array_equal(sliced.theta[0], b.theta[3])
        np.testing.assert_array_equal(sliced.theta[1], b.theta[1])

    def test_slice_to_offset_indices(self):
        # Real cross-scenario alignment: this batch starts at sample_index=10.
        b = _make_batch(n_sims=4, sample_index_offset=10)
        sliced = b.slice_to_indices([10, 12])
        np.testing.assert_array_equal(sliced.sample_index, [10, 12])

    def test_missing_index_raises(self):
        b = _make_batch(n_sims=4)
        with pytest.raises(KeyError, match="not in batch"):
            b.slice_to_indices([0, 99])

    def test_metadata_and_pool_id_preserved(self):
        b = _make_batch()
        b.metadata["foo"] = "bar"
        sliced = b.slice_to_indices([1])
        assert sliced.pool_id == b.pool_id
        assert sliced.metadata == {"foo": "bar"}
