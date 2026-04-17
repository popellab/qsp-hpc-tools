"""Tests for selective batch derivation optimization."""

import json
import math
from unittest.mock import MagicMock, Mock

from qsp_hpc.batch.hpc_job_manager import HPCJobManager


class TestCalculateBatchesNeeded:
    """Test _calculate_batches_needed core logic."""

    def test_calculates_batches_needed_correctly(self):
        """Test batch calculation with realistic numbers."""
        manager = self._create_manager()

        # 47 batches, 400 sims total, need 200
        manager.transport.exec = Mock(return_value=(0, "47"))
        manager.result_collector.count_pool_simulations = Mock(return_value=400)

        batches = manager._calculate_batches_needed("/pool/path", num_simulations=200)

        # avg = 400/47 ≈ 8.5 sims/batch → need ceil(200/8.5) = 24 batches
        expected = math.ceil(200 / (400 / 47))
        assert batches == expected
        assert batches < 47  # Should derive less than total

    def test_returns_all_batches_when_num_simulations_none(self):
        """Test that num_simulations=None means derive all."""
        manager = self._create_manager()

        manager.transport.exec = Mock(return_value=(0, "10"))
        manager.result_collector.count_pool_simulations = Mock(return_value=1000)

        batches = manager._calculate_batches_needed("/pool/path", num_simulations=None)

        assert batches == 10  # All batches

    def test_caps_at_total_batches_available(self):
        """Test can't derive more batches than exist."""
        manager = self._create_manager()

        # 10 batches, 100 sims, need 500
        manager.transport.exec = Mock(return_value=(0, "10"))
        manager.result_collector.count_pool_simulations = Mock(return_value=100)

        batches = manager._calculate_batches_needed("/pool/path", num_simulations=500)

        assert batches == 10  # Capped at available

    @staticmethod
    def _create_manager():
        """Create mock HPCJobManager."""
        manager = HPCJobManager.__new__(HPCJobManager)
        manager.transport = MagicMock()
        manager.result_collector = MagicMock()
        manager.logger = MagicMock()
        return manager


class TestCheckHPCTestStatsValidation:
    """Test >= matching instead of exact matching."""

    def test_accepts_more_derived_than_needed(self):
        """Test that having 400 derived when needing 200 is OK."""
        manager = self._create_manager()

        # Have 400 derived, need 200
        manager.transport.exec = Mock(
            side_effect=[
                (0, "TEST_STATS_CHUNKS:10\nPARAMS_CHUNKS:10"),
                (0, "400"),  # More than needed
            ]
        )

        result = manager.check_hpc_test_stats("/pool", "abc123", expected_n_sims=200)

        assert result is True  # Should accept >= match

    def test_preserves_partial_derivations_for_topup(self):
        """When HPC has fewer derived rows than needed, keep them and let
        the caller's top-up path submit the delta.

        n_derived cannot exceed the pool's actual row count, so if it's
        short the pool itself is short — deleting and re-deriving produces
        the same count and wastes compute. Regression lock for the commit
        that stopped the old delete-and-re-derive behaviour; see
        qsp_hpc/batch/hpc_job_manager.py:check_hpc_test_stats.
        """
        manager = self._create_manager()

        # Have 100 derived, need 200 — only two execs (chunk check + count);
        # no rm -rf cleanup should be issued.
        manager.transport.exec = Mock(
            side_effect=[
                (0, "TEST_STATS_CHUNKS:2\nPARAMS_CHUNKS:2"),
                (0, "100"),
            ]
        )

        result = manager.check_hpc_test_stats("/pool", "abc123", expected_n_sims=200)

        assert result is True  # Accept; caller tops up the missing 100.

        for call in manager.transport.exec.call_args_list:
            cmd = call.args[0] if call.args else call.kwargs.get("cmd", "")
            assert "rm -rf" not in cmd, f"unexpected destructive cmd: {cmd}"

    @staticmethod
    def _create_manager():
        """Create mock HPCJobManager."""
        manager = HPCJobManager.__new__(HPCJobManager)
        manager.transport = MagicMock()
        manager.logger = MagicMock()
        return manager


class TestMaxBatchesInDerivationConfig:
    """Regression tests for max_batches being passed to derivation config.

    Bug fix: Previously, _calculate_batches_needed computed the correct number
    of batches, but this value was never passed to the derivation worker config.
    This caused the worker to process ALL batches instead of just the needed ones,
    leading to params/test_stats count mismatches.

    Regression test for selective derivation batch count bug.
    """

    def test_max_batches_always_none_for_full_derivation(self, tmp_path):
        """Test that max_batches is always None to derive all batches.

        We always derive ALL batches to handle incremental pool growth correctly.
        Trying to derive only "first N batches" breaks when new batches are added
        because we'd re-derive old batches instead of processing new ones.
        """
        manager = self._create_manager()

        # Mock transport to capture uploaded config
        uploaded_configs = []

        def capture_upload(local_path, remote_path):
            if local_path.endswith(".json"):
                with open(local_path, "r") as f:
                    uploaded_configs.append(json.load(f))

        manager.transport.upload = Mock(side_effect=capture_upload)

        # Mock exec to handle different commands appropriately
        def mock_exec(cmd, timeout=None):
            if "echo $HOME" in cmd:
                return (0, "/home/user")
            elif "batch_*.parquet" in cmd and "wc -l" in cmd:
                return (0, "79")
            elif "mkdir" in cmd:
                return (0, "")
            return (0, "")

        manager.transport.exec = Mock(side_effect=mock_exec)
        manager.result_collector.count_pool_simulations = Mock(return_value=5000)
        manager.slurm_submitter.submit_derivation_job = Mock(return_value="12345")
        manager.file_transfer.ensure_hpc_venv = Mock()

        # Create a dummy test_stats CSV
        test_stats_csv = tmp_path / "test_stats.csv"
        test_stats_csv.write_text("test_statistic_id,required_species\nstat1,V_T.C1\n")

        # Call submit_derivation_job with num_simulations=4500
        # Even though we only need 4500, we should derive ALL batches
        manager.submit_derivation_job(
            pool_path="/pool/path",
            test_stats_csv=str(test_stats_csv),
            test_stats_hash="abc123",
            num_simulations=4500,
        )

        # Verify max_batches is None (derive all batches)
        assert len(uploaded_configs) == 1
        config = uploaded_configs[0]
        assert "max_batches" in config
        assert config["max_batches"] is None, "max_batches should be None to derive all batches"

    def test_max_batches_none_when_num_simulations_none(self, tmp_path):
        """Test that max_batches is None when num_simulations is not specified.

        When num_simulations is None, we want to derive ALL batches, so
        max_batches should be None (not a number).
        """
        manager = self._create_manager()

        # Mock transport to capture uploaded config
        uploaded_configs = []

        def capture_upload(local_path, remote_path):
            if local_path.endswith(".json"):
                with open(local_path, "r") as f:
                    uploaded_configs.append(json.load(f))

        manager.transport.upload = Mock(side_effect=capture_upload)

        # Mock exec to handle different commands appropriately
        def mock_exec(cmd, timeout=None):
            if "echo $HOME" in cmd:
                return (0, "/home/user")
            elif "batch_*.parquet" in cmd and "wc -l" in cmd:
                return (0, "10")
            elif "mkdir" in cmd:
                return (0, "")
            return (0, "")

        manager.transport.exec = Mock(side_effect=mock_exec)
        manager.result_collector.count_pool_simulations = Mock(return_value=1000)
        manager.slurm_submitter.submit_derivation_job = Mock(return_value="12345")
        manager.file_transfer.ensure_hpc_venv = Mock()

        # Create a dummy test_stats CSV
        test_stats_csv = tmp_path / "test_stats.csv"
        test_stats_csv.write_text("test_statistic_id,required_species\nstat1,V_T.C1\n")

        # Call submit_derivation_job WITHOUT num_simulations
        manager.submit_derivation_job(
            pool_path="/pool/path",
            test_stats_csv=str(test_stats_csv),
            test_stats_hash="abc123",
            num_simulations=None,  # Derive all
        )

        # Verify max_batches is None
        assert len(uploaded_configs) == 1
        config = uploaded_configs[0]
        assert "max_batches" in config
        assert config["max_batches"] is None

    @staticmethod
    def _create_manager():
        """Create mock HPCJobManager with all required attributes."""
        manager = HPCJobManager.__new__(HPCJobManager)
        manager.transport = MagicMock()
        manager.logger = MagicMock()
        manager.result_collector = MagicMock()
        manager.slurm_submitter = MagicMock()
        manager.file_transfer = MagicMock()
        manager.config = MagicMock()
        manager.config.remote_project_path = "/home/user/project"
        manager.config.hpc_venv_path = "/home/user/.venv"
        manager.verbose = False
        return manager
