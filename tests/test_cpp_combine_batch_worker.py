"""Tests for qsp_hpc.batch.cpp_combine_batch_worker.

Covers the post-array combine step that consolidates per-task chunk
parquets into one pool-level batch file matching the MATLAB layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _write_chunk(path: Path, n_sims: int, param_names: list[str], species: list[str]) -> None:
    """Write a synthetic chunk parquet matching what cpp_batch_worker
    produces — param:* columns plus species list columns."""
    rng = np.random.default_rng(int(path.stem.split("_")[-1]))  # deterministic per chunk idx
    cols: dict[str, pa.Array] = {
        "simulation_id": pa.array(np.arange(n_sims, dtype=np.int64)),
        "status": pa.array(np.zeros(n_sims, dtype=np.int64)),
        "time": pa.array(
            [[0.0, 1.0]] * n_sims,
            type=pa.list_(pa.float64()),
        ),
    }
    for name in param_names:
        cols[f"param:{name}"] = pa.array(rng.uniform(0, 1, n_sims))
    for sp in species:
        cols[sp] = pa.array(
            [rng.uniform(0, 100, 2).tolist() for _ in range(n_sims)],
            type=pa.list_(pa.float64()),
        )
    pq.write_table(pa.table(cols), str(path))


@pytest.fixture
def staging_pool(tmp_path: Path) -> tuple[Path, Path]:
    """Build a pool dir with a staging subdir containing 3 chunks of
    4 sims each. Returns (pool_base, staging_dir)."""
    pool_base = tmp_path / "pools"
    pool_id = "v1_abc12345_scen"
    array_job_id = "7654321"
    staging = pool_base / pool_id / ".staging" / array_job_id
    staging.mkdir(parents=True)

    params = ["k1", "k2"]
    species = ["Tumor", "Immune"]
    for idx in range(3):
        _write_chunk(staging / f"chunk_{idx:03d}.parquet", n_sims=4, param_names=params, species=species)

    return pool_base, staging


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestCombineChunks:
    def test_produces_single_batch_matching_matlab_layout(self, staging_pool):
        from qsp_hpc.batch.cpp_combine_batch_worker import combine_chunks

        pool_base, staging = staging_pool
        pool_id = staging.parent.parent.name

        out = combine_chunks(
            {
                "pool_base": str(pool_base),
                "pool_id": pool_id,
                "staging_dir": str(staging),
                "scenario": "scen",
                "n_simulations": 12,
                "seed": 2025,
                "expected_chunks": 3,
            }
        )
        assert out.exists()
        # MATLAB-style filename: no _taskNNN, no chunk_NNN — one batch per submission.
        assert out.name.startswith("batch_")
        assert "_task" not in out.name
        assert out.name.endswith("_scen_12sims_seed2025.parquet")

    def test_consolidates_all_chunk_rows(self, staging_pool):
        from qsp_hpc.batch.cpp_combine_batch_worker import combine_chunks

        pool_base, staging = staging_pool
        pool_id = staging.parent.parent.name

        out = combine_chunks(
            {
                "pool_base": str(pool_base),
                "pool_id": pool_id,
                "staging_dir": str(staging),
                "scenario": "scen",
                "n_simulations": 12,
                "seed": 2025,
                "expected_chunks": 3,
            }
        )
        table = pq.read_table(str(out))
        # 3 chunks × 4 sims = 12 rows total.
        assert table.num_rows == 12
        # Preserves the full column set.
        assert set(table.column_names) >= {
            "simulation_id",
            "status",
            "time",
            "param:k1",
            "param:k2",
            "Tumor",
            "Immune",
        }

    def test_renumbers_simulation_id_to_be_unique(self, staging_pool):
        """Each chunk reuses simulation_id 0..3 locally; the combined file
        should renumber them 0..11 so downstream groupby-simulation_id works."""
        from qsp_hpc.batch.cpp_combine_batch_worker import combine_chunks

        pool_base, staging = staging_pool
        pool_id = staging.parent.parent.name

        out = combine_chunks(
            {
                "pool_base": str(pool_base),
                "pool_id": pool_id,
                "staging_dir": str(staging),
                "scenario": "scen",
                "n_simulations": 12,
                "seed": 2025,
                "expected_chunks": 3,
            }
        )
        table = pq.read_table(str(out))
        sim_ids = table.column("simulation_id").to_pylist()
        assert sim_ids == list(range(12))

    def test_removes_staging_dir_on_success(self, staging_pool):
        from qsp_hpc.batch.cpp_combine_batch_worker import combine_chunks

        pool_base, staging = staging_pool
        pool_id = staging.parent.parent.name
        combine_chunks(
            {
                "pool_base": str(pool_base),
                "pool_id": pool_id,
                "staging_dir": str(staging),
                "scenario": "scen",
                "n_simulations": 12,
                "seed": 2025,
                "expected_chunks": 3,
            }
        )
        assert not staging.exists(), "staging dir should be removed after combine"

    def test_missing_staging_dir_raises(self, tmp_path):
        from qsp_hpc.batch.cpp_combine_batch_worker import combine_chunks

        with pytest.raises(FileNotFoundError, match="Staging"):
            combine_chunks(
                {
                    "pool_base": str(tmp_path / "pools"),
                    "pool_id": "v1_x_s",
                    "staging_dir": str(tmp_path / "nowhere"),
                    "scenario": "s",
                    "n_simulations": 1,
                    "seed": 0,
                    "expected_chunks": 1,
                }
            )

    def test_empty_staging_dir_raises(self, tmp_path):
        from qsp_hpc.batch.cpp_combine_batch_worker import combine_chunks

        staging = tmp_path / "pools" / "v1_x_s" / ".staging" / "123"
        staging.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="No chunk parquets"):
            combine_chunks(
                {
                    "pool_base": str(tmp_path / "pools"),
                    "pool_id": "v1_x_s",
                    "staging_dir": str(staging),
                    "scenario": "s",
                    "n_simulations": 1,
                    "seed": 0,
                    "expected_chunks": 1,
                }
            )

    def test_fewer_chunks_than_expected_still_succeeds(self, staging_pool):
        """If a pathological task fails to write its chunk (e.g. wall-time
        kill after log-flush but before parquet flush), combine should still
        produce a consolidated batch from what's available. The filename
        encodes the ACTUAL row count — not the requested n_simulations —
        so downstream filename-based sim-counters (count_pool_simulations)
        don't overreport and trigger spurious re-derivations."""
        from qsp_hpc.batch.cpp_combine_batch_worker import combine_chunks

        pool_base, staging = staging_pool
        pool_id = staging.parent.parent.name
        # 3 chunks present, claim 5 expected.
        out = combine_chunks(
            {
                "pool_base": str(pool_base),
                "pool_id": pool_id,
                "staging_dir": str(staging),
                "scenario": "scen",
                "n_simulations": 20,  # requested — may not be achievable
                "seed": 2025,
                "expected_chunks": 5,
            }
        )
        assert out.exists()
        # Filename reflects actual delivered count (12 = 3 chunks × 4 sims),
        # not the 20 originally requested.
        assert out.name.endswith("_scen_12sims_seed2025.parquet")
        assert "_scen_20sims_" not in out.name
        table = pq.read_table(str(out))
        assert table.num_rows == 12

    def test_multiple_combines_in_same_pool_do_not_collide(self, tmp_path):
        """Two successive submissions into the same pool (eg a top-up) each
        land their own batch_*.parquet without overwriting the prior one."""
        from qsp_hpc.batch.cpp_combine_batch_worker import combine_chunks

        pool_base = tmp_path / "pools"
        pool_id = "v1_abc12345_scen"
        pool_dir = pool_base / pool_id

        for run_i, array_id in enumerate(["111", "222"]):
            staging = pool_dir / ".staging" / array_id
            staging.mkdir(parents=True)
            _write_chunk(
                staging / "chunk_000.parquet",
                n_sims=4,
                param_names=["k1"],
                species=["Tumor"],
            )
            combine_chunks(
                {
                    "pool_base": str(pool_base),
                    "pool_id": pool_id,
                    "staging_dir": str(staging),
                    "scenario": "scen",
                    "n_simulations": 4,
                    "seed": 2025,
                    "expected_chunks": 1,
                }
            )

        batches = sorted(pool_dir.glob("batch_*.parquet"))
        # Either two distinct files OR one combined — we require two so
        # the pool keeps the append-only semantics MATLAB relies on.
        assert len(batches) == 2, (
            f"Expected 2 batch files, got {len(batches)}: {[b.name for b in batches]}"
        )
