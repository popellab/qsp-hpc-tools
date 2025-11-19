"""Tests for selective batch derivation optimization."""

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

    def test_rejects_less_derived_than_needed(self):
        """Test that having 100 derived when needing 200 triggers re-derivation."""
        manager = self._create_manager()

        # Have 100 derived, need 200
        manager.transport.exec = Mock(
            side_effect=[
                (0, "TEST_STATS_CHUNKS:2\nPARAMS_CHUNKS:2"),
                (0, "100"),  # Not enough
                (0, ""),  # rm -rf cleanup
            ]
        )

        result = manager.check_hpc_test_stats("/pool", "abc123", expected_n_sims=200)

        assert result is False  # Should reject and trigger re-derivation

    @staticmethod
    def _create_manager():
        """Create mock HPCJobManager."""
        manager = HPCJobManager.__new__(HPCJobManager)
        manager.transport = MagicMock()
        manager.logger = MagicMock()
        return manager
