"""Tests for batch processing utilities."""

import pytest
from qsp_hpc.batch.batch_utils import calculate_batch_split, calculate_num_tasks


class TestCalculateBatchSplit:
    """Tests for calculate_batch_split function."""

    def test_even_split(self):
        """Test when simulations divide evenly across tasks."""
        jobs_per_chunk, n_tasks = calculate_batch_split(100, 10)
        assert jobs_per_chunk == 10
        assert n_tasks == 10

    def test_uneven_split(self):
        """Test when simulations don't divide evenly."""
        jobs_per_chunk, n_tasks = calculate_batch_split(25, 10)
        assert jobs_per_chunk == 3
        assert n_tasks == 9  # ceil(25 / 3) = 9

    def test_fewer_sims_than_tasks(self):
        """Test when we have fewer simulations than max tasks."""
        jobs_per_chunk, n_tasks = calculate_batch_split(5, 10)
        assert jobs_per_chunk == 1
        assert n_tasks == 5

    def test_single_simulation(self):
        """Test edge case with single simulation."""
        jobs_per_chunk, n_tasks = calculate_batch_split(1, 10)
        assert jobs_per_chunk == 1
        assert n_tasks == 1

    def test_single_task(self):
        """Test edge case with single task."""
        jobs_per_chunk, n_tasks = calculate_batch_split(100, 1)
        assert jobs_per_chunk == 100
        assert n_tasks == 1

    def test_large_numbers(self):
        """Test with large number of simulations."""
        jobs_per_chunk, n_tasks = calculate_batch_split(10000, 100)
        assert jobs_per_chunk == 100
        assert n_tasks == 100

    def test_result_consistency(self):
        """Test that results are internally consistent."""
        for num_sims in [1, 5, 25, 100, 105, 1000]:
            for max_tasks in [1, 5, 10, 50, 100]:
                jobs_per_chunk, n_tasks = calculate_batch_split(num_sims, max_tasks)

                # Verify at least 1 job per chunk
                assert jobs_per_chunk >= 1

                # Verify n_tasks doesn't exceed max_tasks
                assert n_tasks <= max_tasks

                # Verify all simulations can be covered
                assert jobs_per_chunk * n_tasks >= num_sims

                # Verify we're not allocating too many extra slots
                # (at most jobs_per_chunk - 1 unused slots)
                assert jobs_per_chunk * n_tasks < num_sims + jobs_per_chunk


class TestCalculateNumTasks:
    """Tests for calculate_num_tasks function."""

    def test_even_division(self):
        """Test when simulations divide evenly by chunk size."""
        n_tasks = calculate_num_tasks(100, 10)
        assert n_tasks == 10

    def test_uneven_division(self):
        """Test when simulations don't divide evenly."""
        n_tasks = calculate_num_tasks(105, 10)
        assert n_tasks == 11

    def test_single_simulation(self):
        """Test edge case with single simulation."""
        n_tasks = calculate_num_tasks(1, 10)
        assert n_tasks == 1

    def test_chunk_larger_than_total(self):
        """Test when chunk size is larger than total simulations."""
        n_tasks = calculate_num_tasks(5, 10)
        assert n_tasks == 1

    def test_single_job_per_chunk(self):
        """Test when each task handles one simulation."""
        n_tasks = calculate_num_tasks(50, 1)
        assert n_tasks == 50

    def test_result_coverage(self):
        """Test that result covers all simulations."""
        for num_sims in [1, 5, 25, 99, 100, 101, 1000]:
            for jobs_per_chunk in [1, 5, 10, 50, 100]:
                n_tasks = calculate_num_tasks(num_sims, jobs_per_chunk)

                # Verify all simulations are covered
                assert n_tasks * jobs_per_chunk >= num_sims

                # Verify we're not over-allocating
                # (at most jobs_per_chunk - 1 unused slots)
                assert n_tasks * jobs_per_chunk < num_sims + jobs_per_chunk


class TestBatchSplitIntegration:
    """Integration tests combining both functions."""

    def test_roundtrip_consistency(self):
        """Test that split and num_tasks calculations are consistent."""
        num_simulations = 137
        max_tasks = 15

        # Calculate split
        jobs_per_chunk, n_tasks = calculate_batch_split(num_simulations, max_tasks)

        # Verify using calculate_num_tasks
        recalculated_tasks = calculate_num_tasks(num_simulations, jobs_per_chunk)
        assert n_tasks == recalculated_tasks

    def test_various_scenarios(self):
        """Test various realistic batch scenarios."""
        scenarios = [
            (1000, 100),    # Large batch, many tasks
            (50, 10),       # Medium batch
            (5, 100),       # Few simulations, many tasks available
            (1, 1),         # Minimum case
            (10000, 500),   # Very large batch
        ]

        for num_sims, max_tasks in scenarios:
            jobs_per_chunk, n_tasks = calculate_batch_split(num_sims, max_tasks)

            # Verify consistency
            assert calculate_num_tasks(num_sims, jobs_per_chunk) == n_tasks

            # Verify coverage
            assert jobs_per_chunk * n_tasks >= num_sims
            assert jobs_per_chunk * (n_tasks - 1) < num_sims
