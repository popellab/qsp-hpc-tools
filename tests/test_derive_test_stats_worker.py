"""Tests for derive_test_stats_worker.py (CSV-based function loading)."""

import numpy as np
import pandas as pd
import pytest

from qsp_hpc.batch.derive_test_stats_worker import (
    build_test_stat_registry,
    compute_test_statistics_batch,
)


class TestBuildTestStatRegistry:
    """Test building test statistic registry from CSV python_function column."""

    def test_simple_function(self):
        """Test loading a simple test statistic function from CSV."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["simple_mean"],
                "required_species": ["V_T.C1"],
                "python_function": [
                    """def compute(time, V_T_C1):
    return np.mean(V_T_C1)"""
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        assert "simple_mean" in registry
        assert callable(registry["simple_mean"])

        # Test the function works
        time = np.array([0, 1, 2, 3])
        species = np.array([10, 20, 30, 40])
        result = registry["simple_mean"](time, species)
        assert result == 25.0

    def test_multiple_species(self):
        """Test function with multiple species arguments."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["ratio_test"],
                "required_species": ["V_T.CD8, V_T.Treg"],
                "python_function": [
                    """def compute(time, V_T_CD8, V_T_Treg):
    cd8_0 = np.interp(0.0, time, V_T_CD8)
    treg_0 = np.interp(0.0, time, V_T_Treg)
    return cd8_0 / max(treg_0, 1e-12)"""
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        time = np.array([0.0, 1.0, 2.0])
        cd8 = np.array([100.0, 110.0, 120.0])
        treg = np.array([50.0, 55.0, 60.0])

        result = registry["ratio_test"](time, cd8, treg)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_growth_rate_calculation(self):
        """Test realistic growth rate function."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["log_growth_rate"],
                "required_species": ["V_T.C1"],
                "python_function": [
                    """def compute(time, V_T_C1):
    t0, t1 = 0, 60
    if len(time) == 0 or len(V_T_C1) == 0:
        return np.nan

    t_eval = np.arange(t0, t1 + 1, 1.0)
    c1_interp = np.interp(t_eval, time, V_T_C1)
    c1_interp = np.maximum(c1_interp, np.finfo(float).eps)

    y = np.log(c1_interp)
    slope = np.polyfit(t_eval, y, 1)[0]
    return slope"""
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        # Exponential growth: C(t) = c0 * exp(r*t) with r=0.01
        time = np.linspace(0, 60, 61)
        c0 = 1e6
        r = 0.01
        tumor_cells = c0 * np.exp(r * time)

        result = registry["log_growth_rate"](time, tumor_cells)
        assert result == pytest.approx(r, rel=1e-2)

    def test_missing_python_function(self):
        """Test that rows without python_function are skipped with warning."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["legacy_stat"],
                "required_species": ["V_T.C1"],
                # Missing python_function column
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        # Should return empty registry and log warning
        assert len(registry) == 0

    def test_nan_python_function(self):
        """Test that NaN python_function is skipped."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["missing_func"],
                "required_species": ["V_T.C1"],
                "python_function": [np.nan],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        assert len(registry) == 0

    def test_invalid_function_code(self):
        """Test that invalid Python code raises error."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["bad_syntax"],
                "required_species": ["V_T.C1"],
                "python_function": ["def compute(time, V_T_C1)\n    return ???"],  # Syntax error
            }
        )

        with pytest.raises(SyntaxError):
            build_test_stat_registry(test_stats_df)

    def test_missing_compute_function(self):
        """Test that code without 'compute' function raises error."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["wrong_name"],
                "required_species": ["V_T.C1"],
                "python_function": [
                    """def my_function(time, V_T_C1):
    return np.mean(V_T_C1)"""
                ],  # Wrong function name
            }
        )

        with pytest.raises(ValueError, match="must define a function named 'compute'"):
            build_test_stat_registry(test_stats_df)

    def test_multiple_functions(self):
        """Test loading multiple functions into registry."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["stat1", "stat2", "stat3"],
                "required_species": ["V_T.C1", "V_T.C1", "V_T.C1"],
                "python_function": [
                    "def compute(time, V_T_C1):\n    return np.mean(V_T_C1)",
                    "def compute(time, V_T_C1):\n    return np.max(V_T_C1)",
                    "def compute(time, V_T_C1):\n    return np.min(V_T_C1)",
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        assert len(registry) == 3
        assert "stat1" in registry
        assert "stat2" in registry
        assert "stat3" in registry

        time = np.array([0, 1, 2])
        species = np.array([10, 20, 30])

        assert registry["stat1"](time, species) == 20.0
        assert registry["stat2"](time, species) == 30.0
        assert registry["stat3"](time, species) == 10.0


class TestComputeTestStatisticsBatch:
    """Test computing test statistics from simulation DataFrame."""

    def test_basic_computation(self):
        """Test basic test statistic computation from simulation data."""
        # Create test statistics definition
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["mean_tumor"],
                "required_species": ["V_T_C1"],
                "python_function": ["def compute(time, V_T_C1):\n    return np.mean(V_T_C1)"],
            }
        )

        # Build registry
        registry = build_test_stat_registry(test_stats_df)

        # Create simulation data (2 sims, 1 test stat)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0, 1],
                "status": [1, 1],  # Both successful
                "time": [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                "V_T_C1": [[100.0, 200.0, 300.0], [50.0, 100.0, 150.0]],  # Mean = 200  # Mean = 100
            }
        )

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry)

        assert result.shape == (2, 1)
        assert result[0, 0] == pytest.approx(200.0)
        assert result[1, 0] == pytest.approx(100.0)

    def test_failed_simulation(self):
        """Test that failed simulations get NaN values."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["mean_tumor"],
                "required_species": ["V_T_C1"],
                "python_function": ["def compute(time, V_T_C1):\n    return np.mean(V_T_C1)"],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        # Simulation 1 failed (status != 1)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0, 1],
                "status": [1, 0],  # Second simulation failed
                "time": [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                "V_T_C1": [[100.0, 200.0, 300.0], [50.0, 100.0, 150.0]],
            }
        )

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry)

        assert result.shape == (2, 1)
        assert result[0, 0] == pytest.approx(200.0)
        assert np.isnan(result[1, 0])  # Failed simulation

    def test_multiple_test_stats(self):
        """Test computing multiple test statistics per simulation."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["mean_tumor", "max_tumor", "min_tumor"],
                "required_species": ["V_T_C1", "V_T_C1", "V_T_C1"],
                "python_function": [
                    "def compute(time, V_T_C1):\n    return np.mean(V_T_C1)",
                    "def compute(time, V_T_C1):\n    return np.max(V_T_C1)",
                    "def compute(time, V_T_C1):\n    return np.min(V_T_C1)",
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        sim_df = pd.DataFrame(
            {
                "simulation_id": [0],
                "status": [1],
                "time": [[0.0, 1.0, 2.0]],
                "V_T_C1": [[100.0, 200.0, 300.0]],
            }
        )

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry)

        assert result.shape == (1, 3)
        assert result[0, 0] == pytest.approx(200.0)  # mean
        assert result[0, 1] == pytest.approx(300.0)  # max
        assert result[0, 2] == pytest.approx(100.0)  # min

    def test_multiple_species(self):
        """Test test statistic using multiple species."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["cd8_treg_ratio"],
                "required_species": ["V_T_CD8, V_T_Treg"],
                "python_function": [
                    """def compute(time, V_T_CD8, V_T_Treg):
    cd8_0 = np.interp(0.0, time, V_T_CD8)
    treg_0 = np.interp(0.0, time, V_T_Treg)
    return cd8_0 / max(treg_0, 1e-12)"""
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        sim_df = pd.DataFrame(
            {
                "simulation_id": [0],
                "status": [1],
                "time": [[0.0, 1.0, 2.0]],
                "V_T_CD8": [[100.0, 110.0, 120.0]],
                "V_T_Treg": [[50.0, 55.0, 60.0]],
            }
        )

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry)

        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(2.0, rel=1e-6)

    def test_missing_test_stat_in_registry(self):
        """Test that missing test statistics are skipped with warning."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["missing_stat"],
                "required_species": ["V_T_C1"],
                "python_function": [np.nan],  # Won't be in registry
            }
        )

        # Empty registry (function was NaN)
        registry = {}

        sim_df = pd.DataFrame(
            {"simulation_id": [0], "status": [1], "time": [[0.0, 1.0]], "V_T_C1": [[100.0, 200.0]]}
        )

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry)

        # Should return NaN for missing test stat
        assert result.shape == (1, 1)
        assert np.isnan(result[0, 0])

    def test_function_raises_exception(self):
        """Test that exceptions in test stat functions result in NaN."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["error_stat"],
                "required_species": ["V_T_C1"],
                "python_function": [
                    """def compute(time, V_T_C1):
    raise ValueError("Test error")"""
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        sim_df = pd.DataFrame(
            {"simulation_id": [0], "status": [1], "time": [[0.0, 1.0]], "V_T_C1": [[100.0, 200.0]]}
        )

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry)

        # Exception should result in NaN
        assert result.shape == (1, 1)
        assert np.isnan(result[0, 0])
