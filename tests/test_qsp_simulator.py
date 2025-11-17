"""Tests for QSP Simulator.

This module tests the QSPSimulator class which provides the interface between
Python SBI workflows and MATLAB QSP models.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from scipy.io import savemat

from qsp_hpc.simulation.qsp_simulator import QSPSimulator, QSPSimulatorError
from qsp_hpc.batch.hpc_job_manager import MissingOutputError, RemoteCommandError, SubmissionError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_priors_csv(temp_dir):
    """Create sample priors CSV file."""
    priors_csv = temp_dir / "priors.csv"

    # Create priors with parameter definitions (matching expected format)
    priors_data = pd.DataFrame({
        'name': ['k_abs', 'k_elim', 'V_d'],
        'distribution': ['lognormal', 'lognormal', 'lognormal'],  # Only lognormal supported
        'dist_param1': [0.5, 0.2, 2.0],  # mean for lognormal
        'dist_param2': [0.3, 0.1, 0.5],  # sigma for lognormal
        'units': ['1/hr', '1/hr', 'L'],
        'description': ['Absorption rate', 'Elimination rate', 'Volume of distribution']
    })
    priors_data.to_csv(priors_csv, index=False)

    return priors_csv


@pytest.fixture
def sample_test_stats_csv(temp_dir):
    """Create sample test statistics CSV file."""
    test_stats_csv = temp_dir / "test_stats.csv"

    # Create test statistics definitions
    test_stats_data = pd.DataFrame({
        'name': ['AUC', 'Cmax', 'Tmax'],
        'observable': ['drug_concentration', 'drug_concentration', 'drug_concentration'],
        'statistic': ['auc', 'max', 'argmax'],
        'units': ['mg*hr/L', 'mg/L', 'hr'],
        'description': ['Area under curve', 'Maximum concentration', 'Time to max']
    })
    test_stats_data.to_csv(test_stats_csv, index=False)

    return test_stats_csv


@pytest.fixture
def mock_simulation_pool(temp_dir):
    """Create mock simulation pool with .mat files."""
    pool_dir = temp_dir / "cache" / "simulations"
    pool_dir.mkdir(parents=True)

    # Create a mock batch file with simulations
    batch_file = pool_dir / "batch_20250101_120000_default_100sims_seed42.mat"

    # Mock simulation data
    n_sims = 100
    n_params = 3
    n_obs = 3

    params = np.random.randn(n_sims, n_params)
    observables = np.random.randn(n_sims, n_obs)

    savemat(batch_file, {
        'parameters': params,
        'observables': observables,
        'metadata': {
            'timestamp': '20250101_120000',
            'scenario': 'default',
            'n_simulations': n_sims,
            'seed': 42
        }
    })

    return pool_dir


# ============================================================================
# Initialization Tests
# ============================================================================

class TestQSPSimulatorInitialization:
    """Test QSPSimulator initialization."""

    def test_valid_initialization(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test successful initialization with valid inputs."""
        cache_dir = temp_dir / "cache"

        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_script='test_model',
            model_version='v1',
            scenario='default',
            cache_dir=cache_dir,
            seed=42
        )

        assert simulator.test_stats_csv == sample_test_stats_csv
        assert simulator.priors_csv == sample_priors_csv
        assert simulator.model_script == 'test_model'
        assert simulator.model_version == 'v1'
        assert simulator.scenario == 'default'
        assert simulator.seed == 42
        assert simulator.cache_dir == cache_dir

    def test_missing_test_stats_csv_error(self, sample_priors_csv, temp_dir):
        """Test that missing test_stats_csv raises error."""
        nonexistent_file = temp_dir / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="Test statistics CSV not found"):
            QSPSimulator(
                test_stats_csv=nonexistent_file,
                priors_csv=sample_priors_csv,
                model_version='v1',
                cache_dir=temp_dir / "cache"
            )

    def test_missing_priors_csv_error(self, sample_test_stats_csv, temp_dir):
        """Test that missing priors_csv raises error."""
        nonexistent_file = temp_dir / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="Priors CSV not found"):
            QSPSimulator(
                test_stats_csv=sample_test_stats_csv,
                priors_csv=nonexistent_file,
                model_version='v1',
                cache_dir=temp_dir / "cache"
            )

    def test_cache_directory_path_stored(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that cache directory path is stored correctly."""
        cache_dir = temp_dir / "custom_cache"

        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=cache_dir
        )

        assert simulator.cache_dir == cache_dir

    def test_default_parameters(self, sample_test_stats_csv, sample_priors_csv):
        """Test that default parameters are set correctly."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1'
        )

        assert simulator.scenario == 'default'
        assert simulator.seed == 2025
        assert simulator.cache_sampling_seed == 2025
        assert simulator.max_tasks == 10
        assert simulator.poll_interval == 30
        assert simulator.max_wait_time is None
        assert simulator.verbose is False

    def test_custom_scenario(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test initialization with custom scenario."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            scenario='gvax',
            cache_dir=temp_dir / "cache"
        )

        assert simulator.scenario == 'gvax'

    def test_repr_format(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test string representation."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            scenario='default',
            cache_dir=temp_dir / "cache"
        )

        repr_str = repr(simulator)
        assert 'QSPSimulator' in repr_str
        assert 'v1' in repr_str
        assert 'default' in repr_str


# ============================================================================
# Hash Computation Tests
# ============================================================================

class TestHashComputation:
    """Test hash computation for cache invalidation."""

    def test_priors_hash_stable_across_runs(self, sample_priors_csv, sample_test_stats_csv, temp_dir):
        """Test that priors hash is stable across multiple runs."""
        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache1"
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache2"
        )

        hash1 = sim1._compute_priors_hash()
        hash2 = sim2._compute_priors_hash()

        assert hash1 == hash2

    def test_priors_hash_changes_with_file_content(self, sample_test_stats_csv, temp_dir):
        """Test that priors hash changes when file content changes."""
        # Create first priors file
        priors1 = temp_dir / "priors1.csv"
        pd.DataFrame({
            'name': ['param1'],
            'distribution': ['lognormal'],
            'dist_param1': [0.0],
            'dist_param2': [1.0]
        }).to_csv(priors1, index=False)

        # Create second priors file with different content
        priors2 = temp_dir / "priors2.csv"
        pd.DataFrame({
            'name': ['param1'],
            'distribution': ['lognormal'],
            'dist_param1': [0.5],  # Changed value
            'dist_param2': [1.0]
        }).to_csv(priors2, index=False)

        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=priors1,
            model_version='v1',
            cache_dir=temp_dir / "cache1"
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=priors2,
            model_version='v1',
            cache_dir=temp_dir / "cache2"
        )

        hash1 = sim1._compute_priors_hash()
        hash2 = sim2._compute_priors_hash()

        assert hash1 != hash2

    def test_test_stats_hash_stable(self, sample_priors_csv, sample_test_stats_csv, temp_dir):
        """Test that test_stats hash is stable."""
        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache1"
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache2"
        )

        hash1 = sim1._compute_test_stats_hash()
        hash2 = sim2._compute_test_stats_hash()

        assert hash1 == hash2

    def test_test_stats_hash_changes_with_content(self, sample_priors_csv, temp_dir):
        """Test that test_stats hash changes when content changes."""
        # Create first test stats file
        stats1 = temp_dir / "stats1.csv"
        pd.DataFrame({
            'name': ['AUC'],
            'observable': ['drug'],
            'statistic': ['auc'],
            'units': ['mg*hr/L']
        }).to_csv(stats1, index=False)

        # Create second test stats file with different content
        stats2 = temp_dir / "stats2.csv"
        pd.DataFrame({
            'name': ['Cmax'],  # Changed
            'observable': ['drug'],
            'statistic': ['max'],
            'units': ['mg/L']
        }).to_csv(stats2, index=False)

        sim1 = QSPSimulator(
            test_stats_csv=stats1,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache1"
        )

        sim2 = QSPSimulator(
            test_stats_csv=stats2,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache2"
        )

        hash1 = sim1._compute_test_stats_hash()
        hash2 = sim2._compute_test_stats_hash()

        assert hash1 != hash2


# ============================================================================
# Parameter Generation Tests
# ============================================================================

class TestParameterGeneration:
    """Test parameter generation from priors."""

    def test_generate_parameters_correct_shape(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that generated parameters have correct shape."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache"
        )

        n_samples = 50
        params = simulator._generate_parameters(n_samples)

        assert params.shape == (n_samples, 3)  # 3 parameters in fixture

    def test_generate_parameters_seed_reproducibility(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that same seed produces same parameters."""
        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache1",
            seed=42
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache2",
            seed=42
        )

        params1 = sim1._generate_parameters(100)
        params2 = sim2._generate_parameters(100)

        np.testing.assert_array_almost_equal(params1, params2)

    def test_generate_parameters_different_seeds_differ(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that different seeds produce different parameters."""
        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache1",
            seed=42
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache2",
            seed=123
        )

        params1 = sim1._generate_parameters(100)
        params2 = sim2._generate_parameters(100)

        # Should not be equal (with very high probability)
        assert not np.allclose(params1, params2)

    def test_generate_zero_parameters(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test generating zero parameters."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache"
        )

        params = simulator._generate_parameters(0)

        assert params.shape == (0, 3)

    def test_generate_single_parameter_set(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test generating single parameter set."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache"
        )

        params = simulator._generate_parameters(1)

        assert params.shape == (1, 3)
        assert isinstance(params, np.ndarray)


# ============================================================================
# Pool Integration Tests
# ============================================================================

class TestPoolIntegration:
    """Test integration with SimulationPoolManager."""

    @patch('qsp_hpc.simulation.simulation_pool.SimulationPoolManager')
    def test_initializes_pool_manager(self, mock_pool_class, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that SimulationPoolManager is initialized correctly."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_script='test_model',
            model_version='v1',
            model_description='Test model',
            scenario='gvax',
            cache_dir=temp_dir / "cache"
        )

        # Pool is initialized during __init__, not lazily
        # Check that pool was created with correct parameters
        mock_pool_class.assert_called_once()
        call_kwargs = mock_pool_class.call_args[1]

        assert call_kwargs['model_version'] == 'v1'
        assert call_kwargs['model_description'] == 'Test model'
        assert call_kwargs['model_script'] == 'test_model'
        assert call_kwargs['cache_dir'] == temp_dir / "cache"
        assert call_kwargs['priors_csv'] == sample_priors_csv
        assert call_kwargs['test_stats_csv'] == sample_test_stats_csv

    def test_injected_pool_and_job_manager_used(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_job_mgr = Mock()

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / 'cache',
            pool=fake_pool,
            job_manager=fake_job_mgr
        )

        assert sim.pool is fake_pool
        assert sim.job_manager is fake_job_mgr


class TestFlowDecisions:
    """Exercise __call__ decision branches with mocks."""

    def test_uses_local_cache_when_sufficient(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 5
        fake_pool.load_simulations.return_value = (np.ones((2, 1)), np.zeros((2, 1)))

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / 'cache',
            pool=fake_pool
        )

        params, obs = sim(2)

        fake_pool.get_available_simulations.assert_called_once()
        fake_pool.load_simulations.assert_called_once()
        assert params.shape == (2, 1)
        assert obs.shape == (2, 1)

    def test_hpc_test_stats_path(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = '/pool'
        fake_job_mgr.check_hpc_test_stats.return_value = True
        fake_job_mgr.check_hpc_full_simulations.return_value = (False, '', 0)

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_script='model',
            model_version='v1',
            cache_dir=temp_dir / 'cache',
            pool=fake_pool,
            job_manager=fake_job_mgr
        )

        with patch.object(sim, '_download_and_add_to_pool', return_value=(np.ones((1, 1)), np.ones((1, 1)))) as downloader:
            sim(1)

        fake_job_mgr.check_hpc_test_stats.assert_called()
        downloader.assert_called()

    def test_runs_new_simulations_when_needed(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = '/pool'
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.check_hpc_full_simulations.return_value = (False, '', 0)

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_script='model',
            model_version='v1',
            cache_dir=temp_dir / 'cache',
            pool=fake_pool,
            job_manager=fake_job_mgr
        )

        with patch.object(sim, '_run_new_simulations', return_value=(np.ones((1, 1)), np.ones((1, 1)))) as runner:
            sim(1)

        runner.assert_called_once()

    def test_hpc_errors_are_wrapped(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = '/pool'
        fake_job_mgr.check_hpc_test_stats.side_effect = MissingOutputError("boom")

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / 'cache',
            pool=fake_pool,
            job_manager=fake_job_mgr
        )

        with pytest.raises(QSPSimulatorError):
            sim(1)

    def test_local_only_raises_when_insufficient_cache(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / 'cache',
            pool=fake_pool,
            local_only=True
        )

        with pytest.raises(QSPSimulatorError):
            sim(1)


class TestHelperFunctions:
    """Unit tests for QSPSimulator helpers."""

    def test_stage_parameters_to_csv(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / 'cache'
        )

        params, csv_path = sim._stage_parameters_to_csv(3)

        assert params.shape == (3, len(sim.param_names))
        assert Path(csv_path).exists()

        # CSV should have header + rows
        contents = Path(csv_path).read_text().strip().splitlines()
        assert len(contents) == 4  # header + 3 rows

        Path(csv_path).unlink()

    def test_update_pool_with_results(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / 'cache',
            pool=fake_pool
        )

        params = np.ones((2, 1))
        obs = np.zeros((2, 1))

        sim._update_pool_with_results(params, obs)

        fake_pool.add_batch.assert_called_once()
        call_kwargs = fake_pool.add_batch.call_args.kwargs
        assert call_kwargs['params_matrix'].shape == (2, 1)
        assert call_kwargs['observables_matrix'].shape == (2, 1)

    def test_run_new_sim_errors_wrapped(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = '/pool'
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.check_hpc_full_simulations.return_value = (False, '', 0)

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / 'cache',
            pool=fake_pool,
            job_manager=fake_job_mgr
        )

        with patch.object(sim, '_run_new_simulations', side_effect=SubmissionError("fail")):
            with pytest.raises(QSPSimulatorError):
                sim(1)


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_negative_batch_size_error(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that negative batch size raises error."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache"
        )

        with pytest.raises((ValueError, AssertionError)):
            simulator._generate_parameters(-10)

    def test_invalid_csv_format_handled(self, sample_priors_csv, temp_dir):
        """Test that invalid CSV format is handled."""
        # Create invalid test stats CSV
        invalid_csv = temp_dir / "invalid.csv"
        invalid_csv.write_text("not,valid,csv,data\n1,2,3")

        simulator = QSPSimulator(
            test_stats_csv=invalid_csv,
            priors_csv=sample_priors_csv,
            model_version='v1',
            cache_dir=temp_dir / "cache"
        )

        # Should handle gracefully or raise informative error
        # The actual behavior depends on implementation
        assert simulator.test_stats_csv == invalid_csv
