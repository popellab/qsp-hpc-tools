"""Tests for QSP Simulator.

This module tests the QSPSimulator class which provides the interface between
Python SBI workflows and MATLAB QSP models.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from qsp_hpc.batch.hpc_job_manager import MissingOutputError, RemoteCommandError, SubmissionError
from qsp_hpc.simulation.qsp_simulator import QSPSimulator, QSPSimulatorError

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
    priors_data = pd.DataFrame(
        {
            "name": ["k_abs", "k_elim", "V_d"],
            "distribution": ["lognormal", "lognormal", "lognormal"],  # Only lognormal supported
            "dist_param1": [0.5, 0.2, 2.0],  # mean for lognormal
            "dist_param2": [0.3, 0.1, 0.5],  # sigma for lognormal
            "units": ["1/hr", "1/hr", "L"],
            "description": ["Absorption rate", "Elimination rate", "Volume of distribution"],
        }
    )
    priors_data.to_csv(priors_csv, index=False)

    return priors_csv


@pytest.fixture
def sample_test_stats_csv(temp_dir):
    """Create sample test statistics CSV file."""
    test_stats_csv = temp_dir / "test_stats.csv"

    # Create test statistics definitions
    test_stats_data = pd.DataFrame(
        {
            "name": ["AUC", "Cmax", "Tmax"],
            "observable": ["drug_concentration", "drug_concentration", "drug_concentration"],
            "statistic": ["auc", "max", "argmax"],
            "units": ["mg*hr/L", "mg/L", "hr"],
            "description": ["Area under curve", "Maximum concentration", "Time to max"],
        }
    )
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

    savemat(
        batch_file,
        {
            "parameters": params,
            "observables": observables,
            "metadata": {"timestamp": "20250101_120000", "scenario": "default", "n_simulations": n_sims, "seed": 42},
        },
    )

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
            project_name="test_project",
            model_script="test_model",
            model_version="v1",
            scenario="default",
            cache_dir=cache_dir,
            seed=42,
            local_only=True,
        )

        assert simulator.test_stats_csv == sample_test_stats_csv
        assert simulator.priors_csv == sample_priors_csv
        assert simulator.model_script == "test_model"
        assert simulator.model_version == "v1"
        assert simulator.scenario == "default"
        assert simulator.seed == 42
        assert simulator.cache_dir == cache_dir

    def test_missing_test_stats_csv_error(self, sample_priors_csv, temp_dir):
        """Test that missing test_stats_csv raises error."""
        nonexistent_file = temp_dir / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="Test statistics CSV not found"):
            QSPSimulator(
                test_stats_csv=nonexistent_file,
                priors_csv=sample_priors_csv,
                project_name="test_project",
                model_version="v1",
                cache_dir=temp_dir / "cache",
            )

    def test_missing_priors_csv_error(self, sample_test_stats_csv, temp_dir):
        """Test that missing priors_csv raises error."""
        nonexistent_file = temp_dir / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="Priors CSV not found"):
            QSPSimulator(
                test_stats_csv=sample_test_stats_csv,
                priors_csv=nonexistent_file,
                project_name="test_project",
                model_version="v1",
                cache_dir=temp_dir / "cache",
            )

    def test_cache_directory_path_stored(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that cache directory path is stored correctly."""
        cache_dir = temp_dir / "custom_cache"

        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            local_only=True,
            cache_dir=cache_dir,
        )

        assert simulator.cache_dir == cache_dir

    def test_default_parameters(self, sample_test_stats_csv, sample_priors_csv):
        """Test that default parameters are set correctly."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            local_only=True,
        )

        assert simulator.scenario == "default"
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
            project_name="test_project",
            model_version="v1",
            scenario="gvax",
            cache_dir=temp_dir / "cache",
            local_only=True,
        )

        assert simulator.scenario == "gvax"

    def test_repr_format(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test string representation."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            scenario="default",
            cache_dir=temp_dir / "cache",
            local_only=True,
        )

        repr_str = repr(simulator)
        assert "QSPSimulator" in repr_str
        assert "v1" in repr_str
        assert "default" in repr_str


class TestParameterGeneration:
    """Tests for parameter sampling behavior."""

    def test_generate_parameters_varies_across_calls(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            seed=7,
        )

        first = simulator._generate_parameters(5)
        second = simulator._generate_parameters(5)

        assert not np.allclose(first, second)


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
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache2",
        )

        hash1 = sim1._compute_priors_hash()
        hash2 = sim2._compute_priors_hash()

        assert hash1 == hash2

    def test_priors_hash_changes_with_file_content(self, sample_test_stats_csv, temp_dir):
        """Test that priors hash changes when file content changes."""
        # Create first priors file
        priors1 = temp_dir / "priors1.csv"
        pd.DataFrame(
            {"name": ["param1"], "distribution": ["lognormal"], "dist_param1": [0.0], "dist_param2": [1.0]}
        ).to_csv(priors1, index=False)

        # Create second priors file with different content
        priors2 = temp_dir / "priors2.csv"
        pd.DataFrame(
            {
                "name": ["param1"],
                "distribution": ["lognormal"],
                "dist_param1": [0.5],  # Changed value
                "dist_param2": [1.0],
            }
        ).to_csv(priors2, index=False)

        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=priors1,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=priors2,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache2",
        )

        hash1 = sim1._compute_priors_hash()
        hash2 = sim2._compute_priors_hash()

        assert hash1 != hash2

    def test_test_stats_hash_stable(self, sample_priors_csv, sample_test_stats_csv, temp_dir):
        """Test that test_stats hash is stable."""
        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache2",
        )

        hash1 = sim1._compute_test_stats_hash()
        hash2 = sim2._compute_test_stats_hash()

        assert hash1 == hash2

    def test_test_stats_hash_changes_with_content(self, sample_priors_csv, temp_dir):
        """Test that test_stats hash changes when content changes."""
        # Create first test stats file
        stats1 = temp_dir / "stats1.csv"
        pd.DataFrame({"name": ["AUC"], "observable": ["drug"], "statistic": ["auc"], "units": ["mg*hr/L"]}).to_csv(
            stats1, index=False
        )

        # Create second test stats file with different content
        stats2 = temp_dir / "stats2.csv"
        pd.DataFrame(
            {"name": ["Cmax"], "observable": ["drug"], "statistic": ["max"], "units": ["mg/L"]}  # Changed
        ).to_csv(stats2, index=False)

        sim1 = QSPSimulator(
            test_stats_csv=stats1,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
        )

        sim2 = QSPSimulator(
            test_stats_csv=stats2,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache2",
        )

        hash1 = sim1._compute_test_stats_hash()
        hash2 = sim2._compute_test_stats_hash()

        assert hash1 != hash2


# ============================================================================
# Parameter Generation Tests
# ============================================================================


class TestParameterGenerationBasics:
    """Test basic parameter generation from priors."""

    def test_generate_parameters_correct_shape(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that generated parameters have correct shape."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
        )

        n_samples = 50
        params = simulator._generate_parameters(n_samples)

        assert params.shape == (n_samples, 3)  # 3 parameters in fixture

    def test_generate_parameters_seed_reproducibility(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that same seed produces same parameters."""
        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
            seed=42,
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache2",
            seed=42,
        )

        params1 = sim1._generate_parameters(100)
        params2 = sim2._generate_parameters(100)

        np.testing.assert_array_almost_equal(params1, params2)

    def test_generate_parameters_different_seeds_differ(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that different seeds produce different parameters."""
        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
            seed=42,
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache2",
            seed=123,
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
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
        )

        params = simulator._generate_parameters(0)

        assert params.shape == (0, 3)

    def test_generate_single_parameter_set(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test generating single parameter set."""
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
        )

        params = simulator._generate_parameters(1)

        assert params.shape == (1, 3)
        assert isinstance(params, np.ndarray)


# ============================================================================
# Pool Integration Tests
# ============================================================================


class TestPoolIntegration:
    """Test integration with SimulationPoolManager."""

    def test_initializes_pool_manager(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that SimulationPoolManager is initialized correctly."""
        # Create simulator and check that pool is initialized
        simulator = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_script="test_model",
            model_version="v1",
            model_description="Test model",
            scenario="gvax",
            cache_dir=temp_dir / "cache",
            local_only=True,  # Prevent lazy-loading of HPCJobManager
        )

        # Pool is initialized during __init__, not lazily
        # Check that pool was created and has expected attributes
        assert simulator.pool is not None
        assert simulator.pool.model_version == "v1"
        assert simulator.pool.model_description == "Test model"
        assert simulator.pool.model_script == "test_model"

    def test_injected_pool_and_job_manager_used(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_job_mgr = Mock()

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
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
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
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
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = True
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_script="model",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        with patch.object(
            sim, "_download_and_add_to_pool", return_value=(np.ones((1, 1)), np.ones((1, 1)))
        ) as downloader:
            sim(1)

        fake_job_mgr.check_hpc_test_stats.assert_called()
        downloader.assert_called()

    def test_runs_new_simulations_when_needed(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_script="model",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # Mock pool to return simulations after _run_new_simulations adds them
        fake_pool.load_simulations.return_value = (np.ones((1, 1)), np.ones((1, 1)))
        with patch.object(sim, "_run_new_simulations") as runner:
            sim(1)

        runner.assert_called_once_with(1)

    def test_hpc_errors_are_wrapped(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.side_effect = MissingOutputError("boom")

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        with pytest.raises(QSPSimulatorError):
            sim(1)

    def test_local_only_raises_when_insufficient_cache(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            local_only=True,
        )

        with pytest.raises(QSPSimulatorError):
            sim(1)


class TestHelperFunctions:
    """Unit tests for QSPSimulator helpers."""

    def test_stage_parameters_to_csv(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
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
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
        )

        params = np.ones((2, 1))
        obs = np.zeros((2, 1))

        sim._update_pool_with_results(params, obs)

        fake_pool.add_batch.assert_called_once()
        call_kwargs = fake_pool.add_batch.call_args.kwargs
        assert call_kwargs["params_matrix"].shape == (2, 1)
        assert call_kwargs["observables_matrix"].shape == (2, 1)

    def test_run_new_sim_errors_wrapped(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        with patch.object(sim, "_run_new_simulations", side_effect=SubmissionError("fail")):
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
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
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
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
        )

        # Should handle gracefully or raise informative error
        # The actual behavior depends on implementation
        assert simulator.test_stats_csv == invalid_csv


# ============================================================================
# Error Path Coverage Tests (Week 2)
# ============================================================================


class TestNetworkAndSSHFailures:
    """Test handling of SSH and network failures."""

    def test_ssh_connection_timeout(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test handling of SSH connection timeout."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.side_effect = TimeoutError("SSH connection timeout")

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # TimeoutError is raised directly, not wrapped
        with pytest.raises(TimeoutError, match="SSH connection timeout"):
            sim(1)

    def test_remote_command_error(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test handling of remote command execution errors."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        # RemoteCommandError takes message, command, returncode as positional args
        fake_job_mgr.check_hpc_test_stats.side_effect = RemoteCommandError("Command failed: test", "test", 1)

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # RemoteCommandError is wrapped in QSPSimulatorError
        with pytest.raises(QSPSimulatorError, match="Failed checking HPC test stats"):
            sim(1)

    def test_network_interruption_during_download(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test handling of network interruption during file download."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = True
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        with patch.object(sim, "_download_and_add_to_pool", side_effect=ConnectionError("Network interrupted")):
            # ConnectionError is wrapped in QSPSimulatorError
            with pytest.raises(QSPSimulatorError, match="Failed downloading test stats from HPC"):
                sim(1)


class TestJobSubmissionFailures:
    """Test handling of HPC job submission failures."""

    def test_slurm_submission_error(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test handling of SLURM job submission error."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        with patch.object(sim, "_run_new_simulations", side_effect=SubmissionError("SLURM queue full")):
            with pytest.raises(QSPSimulatorError, match="SLURM queue full"):
                sim(1)

    def test_insufficient_resources_error(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test handling when HPC has insufficient resources."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        with patch.object(sim, "_run_new_simulations", side_effect=SubmissionError("Insufficient memory")):
            with pytest.raises(QSPSimulatorError):
                sim(1)


class TestJobMonitoringAndTimeouts:
    """Test job monitoring and timeout scenarios."""

    def test_job_monitoring_timeout(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test handling of job monitoring timeout."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
            max_wait_time=1,  # Very short timeout
        )

        # TimeoutError is raised directly, not wrapped
        with patch.object(sim, "_run_new_simulations", side_effect=TimeoutError("Job monitoring timeout")):
            with pytest.raises(TimeoutError, match="timeout"):
                sim(1)

    def test_stuck_job_detection(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test detection of stuck jobs that never complete."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0
        fake_job_mgr.wait_for_jobs.return_value = False  # Jobs never complete

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # RuntimeError is raised directly, not wrapped
        with patch.object(sim, "_stage_parameters_to_csv", return_value=(np.ones((1, 1)), "/tmp/params.csv")):
            with patch.object(sim, "_run_new_simulations", side_effect=RuntimeError("Jobs stuck")):
                with pytest.raises(RuntimeError):
                    sim(1)


class TestResultDownloadFailures:
    """Test handling of result download failures."""

    def test_missing_output_files(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test handling when expected output files are missing."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = True

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        with patch.object(sim, "_download_and_add_to_pool", side_effect=MissingOutputError("Results not found")):
            with pytest.raises(QSPSimulatorError, match="Failed downloading test stats from HPC"):
                sim(1)

    def test_corrupt_result_data(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test handling of corrupt result data."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = True

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        with patch.object(sim, "_download_and_add_to_pool", side_effect=ValueError("Corrupt CSV data")):
            with pytest.raises(QSPSimulatorError, match="Failed downloading test stats from HPC"):
                sim(1)

    def test_partial_download(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test handling of partial file download."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = True

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        with patch.object(sim, "_download_and_add_to_pool", side_effect=IOError("Incomplete download")):
            with pytest.raises(QSPSimulatorError, match="Failed downloading test stats from HPC"):
                sim(1)


class TestConfigurationErrors:
    """Test handling of configuration errors."""

    def test_missing_hpc_credentials(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test handling when HPC credentials are missing."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        # Create simulator (lazy loading means no error yet)
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
        )

        # Error should occur when we try to access job_manager (lazy loading)
        with patch("qsp_hpc.batch.hpc_job_manager.HPCJobManager", side_effect=FileNotFoundError("No credentials")):
            with pytest.raises(FileNotFoundError, match="No credentials"):
                _ = sim.job_manager  # Trigger lazy loading

    def test_invalid_model_script_path(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that invalid model script is handled."""
        # This should work at initialization - validation happens at runtime
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_script="nonexistent_model_script",
            model_version="v1",
            cache_dir=temp_dir / "cache",
        )

        assert sim.model_script == "nonexistent_model_script"


class TestCacheInvalidationScenarios:
    """Test cache invalidation scenarios."""

    def test_cache_invalidated_on_priors_change(self, sample_test_stats_csv, temp_dir):
        """Test that cache is invalidated when priors change."""
        # Create first priors
        priors1 = temp_dir / "priors1.csv"
        pd.DataFrame(
            {"name": ["param1"], "distribution": ["lognormal"], "dist_param1": [0.0], "dist_param2": [1.0]}
        ).to_csv(priors1, index=False)

        fake_pool1 = Mock()
        fake_pool1.get_available_simulations.return_value = 100

        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=priors1,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
            pool=fake_pool1,
        )
        hash1 = sim1._compute_priors_hash()

        # Create second priors with different values
        priors2 = temp_dir / "priors2.csv"
        pd.DataFrame(
            {
                "name": ["param1"],
                "distribution": ["lognormal"],
                "dist_param1": [0.5],  # Different value
                "dist_param2": [1.0],
            }
        ).to_csv(priors2, index=False)

        fake_pool2 = Mock()
        fake_pool2.get_available_simulations.return_value = 100

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=priors2,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache2",
            pool=fake_pool2,
        )
        hash2 = sim2._compute_priors_hash()

        # Hashes should be different, forcing cache invalidation
        assert hash1 != hash2

    def test_cache_invalidated_on_test_stats_change(self, sample_priors_csv, temp_dir):
        """Test that cache is invalidated when test statistics change."""
        # Create first test stats
        stats1 = temp_dir / "stats1.csv"
        pd.DataFrame({"name": ["AUC"], "observable": ["drug"], "statistic": ["auc"], "units": ["mg*hr/L"]}).to_csv(
            stats1, index=False
        )

        sim1 = QSPSimulator(
            test_stats_csv=stats1,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
        )
        hash1 = sim1._compute_test_stats_hash()

        # Create second test stats with different observable
        stats2 = temp_dir / "stats2.csv"
        pd.DataFrame(
            {
                "name": ["AUC"],
                "observable": ["metabolite"],  # Different observable
                "statistic": ["auc"],
                "units": ["mg*hr/L"],
            }
        ).to_csv(stats2, index=False)

        sim2 = QSPSimulator(
            test_stats_csv=stats2,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache2",
        )
        hash2 = sim2._compute_test_stats_hash()

        # Hashes should be different
        assert hash1 != hash2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_simulations_requested(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test requesting zero simulations."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 100
        fake_pool.load_simulations.return_value = (np.zeros((0, 1)), np.zeros((0, 1)))

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
        )

        params, obs = sim(0)
        assert params.shape[0] == 0
        assert obs.shape[0] == 0

    def test_very_large_simulation_request(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test requesting very large number of simulations."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
            max_tasks=100,  # Limit max tasks
        )

        # Should handle splitting into batches
        large_n = 10000
        params_matrix = np.random.rand(large_n, 3)

        fake_pool.load_simulations.return_value = (params_matrix, params_matrix)
        with patch.object(sim, "_run_new_simulations"):
            params, obs = sim(large_n)
            assert params.shape[0] == large_n

    def test_scenario_switching(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that scenario parameter is used correctly."""
        fake_pool = Mock()

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            scenario="gvax",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
        )

        assert sim.scenario == "gvax"

        # Create another simulator with different scenario
        fake_pool2 = Mock()
        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            scenario="anti_pd1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool2,
        )

        assert sim2.scenario == "anti_pd1"
        assert sim.scenario != sim2.scenario

    def test_verbose_mode_enabled(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that verbose mode is properly set."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            verbose=True,
        )

        assert sim.verbose is True

    def test_custom_poll_interval(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test custom poll interval setting."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            poll_interval=60,
        )

        assert sim.poll_interval == 60


# ============================================================================
# Integration Tests - Caching Strategy (Phase 1)
# ============================================================================


class TestThreeTierCachingStrategy:
    """Test the 3-tier caching strategy with realistic scenarios."""

    def test_tier1_local_cache_hit(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that local cache is used when sufficient simulations available."""
        # Setup real pool with actual data
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        pool = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        # Add real simulations to pool
        params = np.random.randn(100, 3)
        obs = np.random.randn(100, 3)
        pool.add_batch(params, obs, seed=42, scenario="default")

        # Create simulator with this pool
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=pool,
            local_only=True,  # Force local-only to test tier 1
        )

        # Should use local cache
        result_params, result_obs = sim(50)

        assert result_params.shape == (50, 3)
        assert result_obs.shape == (50, 3)
        # Verify data came from pool (should be subset of original)
        assert pool.get_available_simulations(scenario="default") == 100

    def test_tier1_insufficient_raises_in_local_only_mode(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that local-only mode raises error when cache insufficient."""
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        pool = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        # Only add 10 simulations
        params = np.random.randn(10, 3)
        obs = np.random.randn(10, 3)
        pool.add_batch(params, obs, seed=42, scenario="default")

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=pool,
            local_only=True,
        )

        # Should raise when requesting more than available
        with pytest.raises(QSPSimulatorError, match="Local-only mode"):
            sim(50)

    def test_tier2_hpc_test_stats_download(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test downloading test statistics from HPC when available."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0  # Empty local cache

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/hpc/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = True  # Test stats available

        # Mock successful download
        downloaded_params = np.random.randn(50, 3)
        downloaded_obs = np.random.randn(50, 3)

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        with patch.object(
            sim, "_download_and_add_to_pool", return_value=(downloaded_params, downloaded_obs)
        ) as mock_download:
            params, obs = sim(50)

            # Verify download was called
            mock_download.assert_called_once()
            assert params.shape == (50, 3)
            assert obs.shape == (50, 3)

    def test_tier3_full_simulations_derivation(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test deriving test statistics from full HPC simulations."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/hpc/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False  # No test stats
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = True  # Full sims available
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 50

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # Mock derivation and download
        result_params = np.random.randn(50, 3)
        result_obs = np.random.randn(50, 3)

        with patch.object(sim, "_derive_test_statistics") as mock_derive:
            with patch.object(
                sim, "_download_and_add_to_pool", return_value=(result_params, result_obs)
            ) as mock_download:
                params, obs = sim(50)

                # Verify derivation was called
                mock_derive.assert_called_once()
                mock_download.assert_called_once()
                assert params.shape == (50, 3)

    def test_tier4_run_new_simulations(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test running new simulations when nothing is cached."""
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/hpc/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False  # Nothing available
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # Mock running new simulations
        new_params = np.random.randn(50, 3)
        new_obs = np.random.randn(50, 3)

        fake_pool.load_simulations.return_value = (new_params, new_obs)
        with patch.object(sim, "_run_new_simulations") as mock_run:
            params, obs = sim(50)

            # Verify new simulations were run
            mock_run.assert_called_once_with(50)
            assert params.shape == (50, 3)


# ============================================================================
# Regression Tests for Pool Path Construction (Scenario Support)
# ============================================================================


class TestPoolPathConsistency:
    """Regression tests to prevent pool path naming bugs."""

    def test_hpc_pool_path_includes_scenario(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """
        Regression test: Ensure HPC pool paths include scenario suffix.

        Bug context: Previously, code was constructing paths as {version}_{hash}
        when checking but {version}_{hash}_{scenario} when saving, causing
        simulations to never be found on HPC.
        """
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/hpc/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="baseline_pdac",
            scenario="gvax",  # Custom scenario
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # Mock _run_new_simulations to capture behavior
        fake_pool.load_simulations.return_value = (np.ones((1, 1)), np.ones((1, 1)))
        with patch.object(sim, "_run_new_simulations"):
            sim(1)

        # Verify that result_collector was called with scenario-aware path
        calls = fake_job_mgr.result_collector.check_pool_directory_exists.call_args_list
        assert len(calls) > 0, "Should have checked for pool directory"

        # Extract the path that was checked
        checked_path = calls[0][0][0]  # First call, first positional arg

        # Path MUST include all three components in correct order: {version}_{hash}_{scenario}
        assert "baseline_pdac" in checked_path, "Path should include model version"
        assert "gvax" in checked_path, "Path should include scenario"

        # Verify correct order: scenario should come AFTER hash, not before
        # Path format: /hpc/pool/baseline_pdac_{hash}_gvax
        path_parts = checked_path.split("/")[-1].split("_")
        assert len(path_parts) >= 3, f"Path should have at least 3 parts: {checked_path}"
        assert path_parts[0] == "baseline", "First part should be 'baseline'"
        assert path_parts[1] == "pdac", "Second part should be 'pdac'"
        assert path_parts[-1] == "gvax", f"Last part should be scenario 'gvax', got: {path_parts[-1]}"

    def test_save_and_check_paths_match(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """
        Regression test: Verify path used for saving matches path used for checking.

        This test ensures that when we save full simulations to HPC, we use the
        same path pattern that we later use to check for those simulations.
        """
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        # Track paths used during operations
        checked_paths = []

        def track_check(path):
            checked_paths.append(path)
            return False  # Simulate not found

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/hpc/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.side_effect = track_check
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            scenario="control",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # Compute the hash that would be used
        priors_hash = sim._compute_priors_hash()

        # Expected path pattern: {simulation_pool_path}/{version}_{hash[:8]}_{scenario}
        expected_suffix = f"v1_{priors_hash[:8]}_control"

        # Trigger simulation which will check for existing pool
        fake_pool.load_simulations.return_value = (np.ones((1, 1)), np.ones((1, 1)))
        with patch.object(sim, "_run_new_simulations"):
            sim(1)

        # Verify the checked path matches expected pattern
        assert len(checked_paths) > 0, "Should have checked at least one path"

        # All checked paths should end with the expected suffix
        for path in checked_paths:
            assert path.endswith(expected_suffix), f"Path '{path}' should end with '{expected_suffix}'"
            assert "control_" not in path or path.endswith(
                "_control"
            ), f"Scenario should be at END of path, not middle: {path}"

    def test_multiple_scenarios_use_different_paths(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """
        Test that different scenarios create different pool paths.

        This ensures scenarios are truly isolated on HPC storage.
        """
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/hpc/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        # Create simulators with different scenarios
        sim_gvax = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            scenario="gvax",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        sim_control = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            scenario="control",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # Mock simulations
        fake_pool.load_simulations.return_value = (np.ones((1, 1)), np.ones((1, 1)))
        with patch.object(sim_gvax, "_run_new_simulations"):
            sim_gvax(1)

        gvax_calls = fake_job_mgr.result_collector.check_pool_directory_exists.call_args_list

        fake_job_mgr.result_collector.check_pool_directory_exists.reset_mock()

        with patch.object(sim_control, "_run_new_simulations"):
            sim_control(1)

        control_calls = fake_job_mgr.result_collector.check_pool_directory_exists.call_args_list

        # Extract paths
        gvax_path = gvax_calls[0][0][0]
        control_path = control_calls[0][0][0]

        # Paths should be different
        assert (
            gvax_path != control_path
        ), f"Different scenarios should use different paths:\n  gvax: {gvax_path}\n  control: {control_path}"

        # Both should contain scenario name
        assert "gvax" in gvax_path, f"GVAX path should contain 'gvax': {gvax_path}"
        assert "control" in control_path, f"Control path should contain 'control': {control_path}"

        # Hash portion should be the same (same priors/model)
        gvax_parts = gvax_path.split("/")[-1].split("_")
        control_parts = control_path.split("/")[-1].split("_")

        # Extract hash (should be second-to-last component before scenario)
        gvax_hash = gvax_parts[-2]
        control_hash = control_parts[-2]

        assert gvax_hash == control_hash, f"Hash should be same for same model/priors: {gvax_hash} vs {control_hash}"


# ============================================================================
# Regression Tests for Returning Full Requested Simulations
# ============================================================================


class TestFullSimulationReturn:
    """Regression tests to ensure all requested simulations are returned."""

    def test_returns_all_simulations_when_combining_cache_and_new(
        self, sample_test_stats_csv, sample_priors_csv, temp_dir
    ):
        """
        Regression test: Ensure full requested amount is returned when combining cached + new.

        Bug context: Previously, when 6 were cached locally and 10 requested, code would run
        4 new simulations on HPC but return only those 4 instead of all 10.

        Flow: Local has 0 -> Check HPC (has 6) -> Run 4 new -> Load all 10 from local pool
        """
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 0  # No local cache initially

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/hpc/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        # HPC has 6 full simulations available
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = True
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 6

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # Mock the results: after deriving from 6 HPC sims and running 4 new, should have all 10
        all_10_params = np.random.randn(10, 3)
        all_10_obs = np.random.randn(10, 3)
        fake_pool.load_simulations.return_value = (all_10_params, all_10_obs)

        # Mock both derivation and running new simulations
        with patch.object(sim, "_derive_test_statistics"):
            with patch.object(sim, "_run_new_simulations") as mock_run:
                params, obs = sim(10)

        # Verify _run_new_simulations was called with the deficit (10 - 6 = 4)
        mock_run.assert_called_once_with(4)

        # Verify load_simulations was called with FULL amount (10)
        fake_pool.load_simulations.assert_called_once()
        call_kwargs = fake_pool.load_simulations.call_args[1]
        assert call_kwargs["n_requested"] == 10, "Should request all 10 from pool, not just the 4 new ones"

        # Verify we got back all 10, not just 4
        assert params.shape == (10, 3), f"Should return all 10 simulations, got {params.shape[0]}"
        assert obs.shape == (10, 3), f"Should return all 10 simulations, got {obs.shape[0]}"

    def test_runs_correct_deficit_amount(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """
        Regression test: Verify correct number of new simulations are run.

        Tests various HPC cache states to ensure deficit calculation is correct.
        """
        test_cases = [
            (0, 10, 10),  # HPC empty: run all 10
            (6, 10, 4),  # HPC has 6: run 4 more
            (8, 10, 2),  # HPC has 8: run 2 more
            (10, 10, 0),  # HPC has all: run 0 (shouldn't reach _run_new_simulations)
        ]

        for n_hpc_cached, n_requested, expected_new in test_cases:
            fake_pool = Mock()
            fake_pool.get_available_simulations.return_value = 0  # No local cache

            fake_job_mgr = Mock()
            fake_job_mgr.config.simulation_pool_path = "/hpc/pool"
            fake_job_mgr.check_hpc_test_stats.return_value = False
            # Mock HPC with n_hpc_cached simulations
            if n_hpc_cached > 0:
                fake_job_mgr.result_collector.check_pool_directory_exists.return_value = True
                fake_job_mgr.result_collector.count_pool_simulations.return_value = n_hpc_cached
            else:
                fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
                fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

            sim = QSPSimulator(
                test_stats_csv=sample_test_stats_csv,
                priors_csv=sample_priors_csv,
                project_name="test_project",
                model_version="v1",
                cache_dir=temp_dir / "cache",
                pool=fake_pool,
                job_manager=fake_job_mgr,
            )

            # Mock return values
            result_params = np.random.randn(n_requested, 3)
            result_obs = np.random.randn(n_requested, 3)
            fake_pool.load_simulations.return_value = (result_params, result_obs)

            if n_hpc_cached >= n_requested:
                # HPC has enough - should derive and download
                with patch.object(sim, "_derive_test_statistics"):
                    with patch.object(sim, "_download_and_add_to_pool", return_value=(result_params, result_obs)):
                        params, obs = sim(n_requested)
                assert params.shape[0] == n_requested
            else:
                # Should derive from HPC cache and run new simulations for the deficit
                with patch.object(sim, "_derive_test_statistics"):
                    with patch.object(sim, "_run_new_simulations") as mock_run:
                        params, obs = sim(n_requested)

                        if expected_new > 0:
                            mock_run.assert_called_once_with(
                                expected_new
                            ), f"HPC cache={n_hpc_cached}, Need={n_requested}, should run {expected_new} new"

                        # Always verify full amount returned
                        assert (
                            params.shape[0] == n_requested
                        ), f"Should return {n_requested} total, got {params.shape[0]}"

    def test_load_simulations_called_with_correct_scenario(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """
        Regression test: Ensure load_simulations is called with correct scenario.

        When loading combined results, must use the correct scenario to get the right pool.
        """
        fake_pool = Mock()
        fake_pool.get_available_simulations.return_value = 5  # Have 5 cached

        fake_job_mgr = Mock()
        fake_job_mgr.config.simulation_pool_path = "/hpc/pool"
        fake_job_mgr.check_hpc_test_stats.return_value = False
        fake_job_mgr.result_collector.check_pool_directory_exists.return_value = False
        fake_job_mgr.result_collector.count_pool_simulations.return_value = 0

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            scenario="gvax",  # Custom scenario
            cache_dir=temp_dir / "cache",
            pool=fake_pool,
            job_manager=fake_job_mgr,
        )

        # Mock return
        fake_pool.load_simulations.return_value = (np.random.randn(10, 3), np.random.randn(10, 3))

        with patch.object(sim, "_run_new_simulations"):
            sim(10)

        # Verify load_simulations was called with correct scenario
        call_kwargs = fake_pool.load_simulations.call_args[1]
        assert call_kwargs["scenario"] == "gvax", "Should load from correct scenario pool"
        assert call_kwargs["n_requested"] == 10, "Should request full amount"


# ============================================================================
# ============================================================================
# Regression Tests for Parameter Generation Integration
# ============================================================================


class TestParameterGenerationIntegration:
    """Test parameter generation with various pool states."""

    def test_parameter_generation_respects_pool_state(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that new parameters are generated when pool is empty."""
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        pool = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=pool,
            seed=42,
        )

        # Generate parameters when pool is empty
        params1 = sim._generate_parameters(20)
        assert params1.shape == (20, 3)

        # Generate again - should be different (RNG state advanced)
        params2 = sim._generate_parameters(20)
        assert not np.allclose(params1, params2)

    def test_parameter_sampling_from_pool(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that parameters are sampled from pool when available."""
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        pool = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        # Add known parameters to pool
        known_params = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        obs = np.random.randn(3, 3)
        pool.add_batch(known_params, obs, seed=42, scenario="default")

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=pool,
            local_only=True,
        )

        # Load from pool
        params, _ = sim(2)

        # Should be subset of known params
        assert params.shape == (2, 3)
        for row in params:
            # Each row should match one of the known params
            matches = np.any([np.allclose(row, known_row) for known_row in known_params])
            assert matches

    def test_large_batch_parameter_generation(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test parameter generation for large batches."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            seed=42,
        )

        # Generate large batch
        params = sim._generate_parameters(10000)

        assert params.shape == (10000, 3)
        # Verify lognormal properties (positive values)
        assert np.all(params > 0)


class TestScenarioWorkflows:
    """Test scenario-specific simulation workflows."""

    def test_scenario_pool_isolation(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that different scenarios use separate pool storage."""
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        # Create pool for 'gvax' scenario
        pool = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        # Add simulations for 'gvax' scenario
        params_gvax = np.random.randn(50, 3)
        obs_gvax = np.random.randn(50, 3)
        pool.add_batch(params_gvax, obs_gvax, seed=42, scenario="gvax")

        # Add simulations for 'control' scenario
        params_control = np.random.randn(30, 3)
        obs_control = np.random.randn(30, 3)
        pool.add_batch(params_control, obs_control, seed=42, scenario="control")

        # Check that scenarios are isolated
        assert pool.get_available_simulations(scenario="gvax") == 50
        assert pool.get_available_simulations(scenario="control") == 30
        assert pool.get_available_simulations(scenario="anti_pd1") == 0

    def test_multi_scenario_simulation(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test running simulations for multiple scenarios."""
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        pool = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        # Create simulators for different scenarios
        sim_gvax = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            scenario="gvax",
            cache_dir=temp_dir / "cache",
            pool=pool,
        )

        sim_control = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            scenario="control",
            cache_dir=temp_dir / "cache",
            pool=pool,
        )

        # Verify scenarios are different
        assert sim_gvax.scenario == "gvax"
        assert sim_control.scenario == "control"


class TestHashComputationIntegration:
    """Test hash computation in realistic scenarios."""

    def test_hash_changes_invalidate_cache(self, sample_test_stats_csv, temp_dir):
        """Test that changing priors invalidates HPC cache lookups."""
        # Create two different priors files
        priors1 = temp_dir / "priors_v1.csv"
        pd.DataFrame(
            {
                "name": ["param1", "param2"],
                "distribution": ["lognormal", "lognormal"],
                "dist_param1": [0.0, 0.0],
                "dist_param2": [1.0, 1.0],
            }
        ).to_csv(priors1, index=False)

        priors2 = temp_dir / "priors_v2.csv"
        pd.DataFrame(
            {
                "name": ["param1", "param2"],
                "distribution": ["lognormal", "lognormal"],
                "dist_param1": [0.5, 0.5],  # Different values
                "dist_param2": [1.0, 1.0],
            }
        ).to_csv(priors2, index=False)

        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=priors1,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=priors2,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache2",
        )

        # Hashes should differ
        hash1 = sim1._compute_priors_hash()
        hash2 = sim2._compute_priors_hash()
        assert hash1 != hash2

        # Pool IDs should also differ
        from qsp_hpc.constants import HASH_PREFIX_LENGTH

        pool_id1 = f"v1_{hash1[:HASH_PREFIX_LENGTH]}"
        pool_id2 = f"v1_{hash2[:HASH_PREFIX_LENGTH]}"
        assert pool_id1 != pool_id2

    def test_hash_includes_model_script(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that model_script is included in priors hash."""
        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_script="model_v1",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_script="model_v2",  # Different model script
            model_version="v1",
            cache_dir=temp_dir / "cache2",
        )

        # Hashes should differ
        assert sim1._compute_priors_hash() != sim2._compute_priors_hash()


class TestEndToEndIntegration:
    """End-to-end integration tests with real file I/O."""

    def test_complete_local_workflow(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test complete workflow from empty cache to loaded results."""
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        cache_dir = temp_dir / "e2e_cache"

        # Create simulator with fresh cache
        pool = SimulationPoolManager(
            cache_dir=cache_dir,
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=cache_dir,
            pool=pool,
            seed=42,
        )

        # Verify pool is empty
        assert pool.get_available_simulations(scenario="default") == 0

        # Add simulations manually (simulating HPC results)
        params = sim._generate_parameters(100)
        obs = np.random.randn(100, 3)
        pool.add_batch(params, obs, seed=42, scenario="default")

        # Now request simulations - should use cache
        result_params, result_obs = sim(50)

        assert result_params.shape == (50, 3)
        assert result_obs.shape == (50, 3)

        # Verify pool still has all simulations
        assert pool.get_available_simulations(scenario="default") == 100

    def test_cache_persistence_across_instances(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that cache persists across simulator instances."""
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        cache_dir = temp_dir / "persistent_cache"

        # First instance: add to cache
        pool1 = SimulationPoolManager(
            cache_dir=cache_dir,
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        params = np.random.randn(100, 3)
        obs = np.random.randn(100, 3)
        pool1.add_batch(params, obs, seed=42, scenario="default")

        # Second instance: should see cached data
        pool2 = SimulationPoolManager(
            cache_dir=cache_dir,
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        # Should find the cached simulations
        assert pool2.get_available_simulations(scenario="default") == 100


# ============================================================================
# Additional Coverage Tests - Methods Without Heavy Mocking
# ============================================================================


class TestHashAndConfigMethods:
    """Test hash computation and configuration methods with real execution."""

    def test_compute_priors_hash_with_different_model_versions(
        self, sample_test_stats_csv, sample_priors_csv, temp_dir
    ):
        """Test that model_version affects priors hash."""
        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_script="script",
            model_version="v1",
            cache_dir=temp_dir / "cache1",
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_script="script",
            model_version="v2",  # Different version
            cache_dir=temp_dir / "cache2",
        )

        # Different model versions should produce different hashes
        assert sim1._compute_priors_hash() != sim2._compute_priors_hash()

    def test_compute_test_stats_hash_consistency(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that test stats hash is consistent across multiple calls."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
        )

        hash1 = sim._compute_test_stats_hash()
        hash2 = sim._compute_test_stats_hash()
        hash3 = sim._compute_test_stats_hash()

        # Should be identical
        assert hash1 == hash2 == hash3
        # Should be valid hex string
        assert len(hash1) == 64  # SHA256 produces 64 hex characters


class TestCallableInterface:
    """Test the __call__ method integration paths."""

    def test_call_with_tuple_batch_size(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that __call__ handles tuple batch sizes (BayesFlow format)."""
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        pool = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        # Add simulations
        params = np.random.randn(100, 3)
        obs = np.random.randn(100, 3)
        pool.add_batch(params, obs, seed=42, scenario="default")

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=pool,
            local_only=True,
        )

        # Call with tuple (BayesFlow format: (10,) means 10 simulations)
        result_params, result_obs = sim((10,))

        assert result_params.shape == (10, 3)
        assert result_obs.shape == (10, 3)

    def test_call_with_multidimensional_tuple(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that __call__ handles multi-dimensional tuple batch sizes."""
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        pool = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        # Add simulations
        params = np.random.randn(100, 3)
        obs = np.random.randn(100, 3)
        pool.add_batch(params, obs, seed=42, scenario="default")

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=pool,
            local_only=True,
        )

        # Call with multi-dimensional tuple: (5, 2) = 10 simulations
        result_params, result_obs = sim((5, 2))

        assert result_params.shape == (10, 3)
        assert result_obs.shape == (10, 3)


class TestHelperMethodsRealExecution:
    """Test helper methods with real execution."""

    def test_stage_parameters_creates_valid_csv(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that _stage_parameters_to_csv creates a valid CSV file."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            seed=42,
        )

        # Stage 50 parameters
        params, csv_path = sim._stage_parameters_to_csv(50)

        # Verify parameters shape
        assert params.shape == (50, 3)

        # Verify CSV exists and is valid
        import pandas as pd

        df = pd.read_csv(csv_path)
        assert df.shape == (50, 3)
        assert list(df.columns) == sim.param_names

        # Cleanup
        Path(csv_path).unlink()

    def test_update_pool_adds_to_cache(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that _update_pool_with_results actually adds to pool."""
        from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

        pool = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v1",
            model_description="Test model",
            model_script="test_script",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
        )

        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            pool=pool,
        )

        # Initially empty
        assert pool.get_available_simulations(scenario="default") == 0

        # Add results
        params = np.random.randn(25, 3)
        obs = np.random.randn(25, 3)
        sim._update_pool_with_results(params, obs)

        # Should now have 25 simulations
        assert pool.get_available_simulations(scenario="default") == 25


class TestLoggingAndVerbosity:
    """Test logging configuration and verbosity."""

    def test_verbose_mode_creates_logger(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that verbose mode properly configures logger."""
        sim_verbose = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            verbose=True,
        )

        # Logger should be configured
        assert sim_verbose.logger is not None
        assert sim_verbose.verbose is True

    def test_logging_methods_execute_without_error(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that logging helper methods execute without error."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            verbose=True,
        )

        # These should not raise errors
        sim._info("Test info message")
        sim._debug("Test debug message")
        sim._warning("Test warning message")
        sim._error("Test error message")


class TestParameterNames:
    """Test parameter name loading and handling."""

    def test_param_names_loaded_from_priors_csv(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that parameter names are correctly loaded from priors CSV."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
        )

        # From fixture: ['k_abs', 'k_elim', 'V_d']
        assert len(sim.param_names) == 3
        assert "k_abs" in sim.param_names
        assert "k_elim" in sim.param_names
        assert "V_d" in sim.param_names

    def test_param_names_order_matches_csv(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that parameter names maintain CSV order."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
        )

        # Order should match CSV
        assert sim.param_names[0] == "k_abs"
        assert sim.param_names[1] == "k_elim"
        assert sim.param_names[2] == "V_d"


class TestRNGState:
    """Test random number generator state management."""

    def test_cache_sampling_seed_independence(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that cache_sampling_seed is independent of param generation seed."""
        sim1 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            seed=42,
            cache_sampling_seed=999,
        )

        sim2 = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            seed=42,
            cache_sampling_seed=999,
        )

        # Both should have same cache_sampling_seed
        assert sim1.cache_sampling_seed == sim2.cache_sampling_seed == 999

        # But different from main seed
        assert sim1.cache_sampling_seed != sim1.seed

    def test_param_rng_state_advances(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that parameter RNG state advances with each call."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            seed=42,
        )

        # Generate parameters three times
        params1 = sim._generate_parameters(10)
        params2 = sim._generate_parameters(10)
        params3 = sim._generate_parameters(10)

        # All should be different (RNG state advancing)
        assert not np.allclose(params1, params2)
        assert not np.allclose(params2, params3)
        assert not np.allclose(params1, params3)


class TestProjectConfiguration:
    """Test project-specific configuration."""

    def test_project_name_stored(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that project_name is stored correctly."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="my_custom_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
        )

        assert sim.project_name == "my_custom_project"

    def test_model_description_stored(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that model_description is stored."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            model_description="My test model description",
            cache_dir=temp_dir / "cache",
        )

        assert sim.model_description == "My test model description"

    def test_max_tasks_configuration(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """Test that max_tasks is configurable."""
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="v1",
            cache_dir=temp_dir / "cache",
            max_tasks=50,
        )

        assert sim.max_tasks == 50


class TestPriorPPCSimulationReuse:
    """
    Test that prior PPCs correctly reuse existing simulations from shared pool.

    This validates the fix for the bug where prior PPCs would run ALL n_samples
    new simulations instead of only running the deficit and reusing existing ones.
    """

    def test_prior_ppc_reuses_training_simulations(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """
        Test that prior PPCs reuse existing training simulations.

        Scenario:
        1. Training generates 18 simulations (saved to shared HPC pool)
        2. Prior PPC needs 200 simulations
        3. Simulator should find 18 existing, run only 182 new
        4. Return all 200 samples

        This is a regression test for the bug where prior PPCs called
        simulate_with_parameters() which didn't reuse existing simulations.
        """
        # Mock job manager to avoid needing config file
        fake_job_manager = MagicMock()
        fake_job_manager.check_hpc_test_stats = MagicMock(return_value=False)
        fake_job_manager.result_collector.check_pool_directory_exists = MagicMock(return_value=True)
        fake_job_manager.result_collector.count_pool_simulations = MagicMock(return_value=18)

        # Create simulator with mocked job manager
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="baseline_pdac",
            scenario="default",
            cache_dir=temp_dir / "cache",
            job_manager=fake_job_manager,  # Inject mock to avoid config requirement
        )

        # Mock the pool to simulate having 18 existing simulations
        fake_pool = MagicMock()
        sim.pool = fake_pool

        # First call: pool has 0 locally, HPC has 18 full sims
        fake_pool.get_available_simulations.return_value = 0

        # Mock running new simulations (182 needed)
        sim._run_new_simulations = MagicMock()

        # Mock pool loading after running new sims - returns all 200
        fake_pool.load_simulations.return_value = (
            np.random.rand(200, 8),  # 200 params
            np.random.rand(200, 12),  # 200 observables
        )

        # Call simulator (like prior PPCs do)
        theta, obs = sim(200)

        # Verify it ran only 182 new simulations (not 200)
        sim._run_new_simulations.assert_called_once_with(182)

        # Verify it loaded 200 from pool after adding new ones
        fake_pool.load_simulations.assert_called_once()
        assert fake_pool.load_simulations.call_args[1]["n_requested"] == 200
        assert fake_pool.load_simulations.call_args[1]["scenario"] == "default"

        # Verify we got 200 samples back
        assert theta.shape == (200, 8)
        assert obs.shape == (200, 12)

    def test_prior_ppc_runs_all_when_pool_empty(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """
        Test that prior PPCs run all simulations when pool is empty.

        Scenario:
        1. No existing simulations
        2. Prior PPC needs 200 simulations
        3. Simulator should run all 200
        """
        # Mock job manager to avoid needing config file
        fake_job_manager = MagicMock()
        fake_job_manager.check_hpc_test_stats = MagicMock(return_value=False)
        fake_job_manager.result_collector.check_pool_directory_exists = MagicMock(return_value=False)
        fake_job_manager.result_collector.count_pool_simulations = MagicMock(return_value=0)

        # Create simulator with mocked job manager
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="baseline_pdac",
            scenario="default",
            cache_dir=temp_dir / "cache",
            job_manager=fake_job_manager,  # Inject mock to avoid config requirement
        )

        # Mock the pool - empty
        fake_pool = MagicMock()
        sim.pool = fake_pool
        fake_pool.get_available_simulations.return_value = 0

        # Mock running new simulations
        sim._run_new_simulations = MagicMock()

        # Mock pool loading
        fake_pool.load_simulations.return_value = (np.random.rand(200, 8), np.random.rand(200, 12))

        # Call simulator
        theta, obs = sim(200)

        # Verify it ran all 200 simulations
        sim._run_new_simulations.assert_called_once_with(200)

        # Verify we got 200 samples back
        assert theta.shape == (200, 8)
        assert obs.shape == (200, 12)

    def test_prior_ppc_uses_all_when_pool_sufficient(self, sample_test_stats_csv, sample_priors_csv, temp_dir):
        """
        Test that prior PPCs use existing simulations when enough are available.

        Scenario:
        1. Pool has 500 existing simulations
        2. Prior PPC needs 200 simulations
        3. Simulator should use existing (no new runs needed)
        """
        # Mock job manager to avoid needing config file
        fake_job_manager = MagicMock()
        fake_job_manager.check_hpc_test_stats = MagicMock(return_value=False)
        fake_job_manager.result_collector.check_pool_directory_exists = MagicMock(return_value=True)
        fake_job_manager.result_collector.count_pool_simulations = MagicMock(return_value=500)

        # Create simulator with mocked job manager
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            project_name="test_project",
            model_version="baseline_pdac",
            scenario="default",
            cache_dir=temp_dir / "cache",
            job_manager=fake_job_manager,  # Inject mock to avoid config requirement
        )

        # Mock the pool - has plenty
        fake_pool = MagicMock()
        sim.pool = fake_pool
        fake_pool.get_available_simulations.return_value = 0  # Not in local cache

        # Mock derivation and download
        sim._derive_test_statistics = MagicMock()
        sim._download_and_add_to_pool = MagicMock(return_value=(np.random.rand(200, 8), np.random.rand(200, 12)))

        # Mock running new simulations (should NOT be called)
        sim._run_new_simulations = MagicMock()

        # Call simulator
        theta, obs = sim(200)

        # Verify it did NOT run new simulations
        sim._run_new_simulations.assert_not_called()

        # Verify it derived test stats and downloaded
        sim._derive_test_statistics.assert_called_once()
        sim._download_and_add_to_pool.assert_called_once()

        # Verify we got 200 samples back
        assert theta.shape == (200, 8)
        assert obs.shape == (200, 12)

    def test_posterior_ppc_does_not_reuse_prior_simulations(self, temp_dir, sample_priors_csv, sample_test_stats_csv):
        """Test that posterior PPCs with specific parameters do NOT reuse prior simulations."""
        # Create mock job manager that simulates having prior simulations available
        fake_job_manager = MagicMock()
        fake_job_manager.config.simulation_pool_path = "/fake/pool"

        # Simulate that shared pool has 200 prior simulations available
        fake_job_manager.result_collector.check_pool_directory_exists = MagicMock(return_value=True)
        fake_job_manager.result_collector.count_pool_simulations = MagicMock(return_value=200)
        fake_job_manager.check_hpc_test_stats = MagicMock(return_value=False)

        # Create simulator
        sim = QSPSimulator(
            test_stats_csv=sample_test_stats_csv,
            priors_csv=sample_priors_csv,
            model_script="test_model",
            model_version="v1",
            scenario="default",
            project_name="test_project",
            cache_dir=temp_dir,
            verbose=True,
            job_manager=fake_job_manager,
        )

        # Mock the actual simulation execution
        mock_observables = np.random.randn(200, 12)
        sim._run_matlab_simulation = MagicMock(return_value=mock_observables)

        # Create specific posterior parameter values
        posterior_params = np.random.randn(200, 8)

        # Call simulate_with_parameters (used by posterior PPCs)
        test_stats = sim.simulate_with_parameters(posterior_params, pool_suffix="posterior_ppc")

        # Verify we ran MATLAB simulation instead of reusing from shared pool
        sim._run_matlab_simulation.assert_called_once()

        # Verify we did NOT check/download from HPC pool
        fake_job_manager.check_hpc_test_stats.assert_not_called()

        # Verify we got results back
        assert test_stats.shape[0] == 200
        assert np.array_equal(test_stats, mock_observables)
