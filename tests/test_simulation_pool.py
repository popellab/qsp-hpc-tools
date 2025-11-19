"""Tests for simulation pool manager."""

import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat, savemat

from qsp_hpc.simulation.simulation_pool import SimulationPoolManager


@pytest.fixture
def sample_priors_csv(temp_dir):
    """Create sample priors CSV file."""
    priors = pd.DataFrame(
        {
            "parameter": ["param1", "param2", "param3"],
            "distribution": ["uniform", "lognormal", "normal"],
            "min": [0.0, 0.1, -1.0],
            "max": [1.0, 10.0, 1.0],
        }
    )
    csv_path = temp_dir / "priors.csv"
    priors.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_test_stats_csv(temp_dir):
    """Create sample test statistics CSV file."""
    test_stats = pd.DataFrame(
        {
            "observable": ["obs1", "obs2", "obs3"],
            "statistic": ["mean", "max", "auc"],
            "compartment": ["plasma", "tumor", "plasma"],
        }
    )
    csv_path = temp_dir / "test_stats.csv"
    test_stats.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def pool_manager(temp_dir, sample_priors_csv, sample_test_stats_csv):
    """Create a simulation pool manager instance."""
    return SimulationPoolManager(
        cache_dir=temp_dir / "cache",
        model_version="test_v1",
        model_description="Test model",
        priors_csv=sample_priors_csv,
        test_stats_csv=sample_test_stats_csv,
        model_script="test_model",
    )


class TestSimulationPoolManagerInit:
    """Tests for SimulationPoolManager initialization."""

    def test_basic_initialization(self, pool_manager):
        """Test basic initialization of pool manager."""
        assert pool_manager.model_version == "test_v1"
        assert pool_manager.model_description == "Test model"
        assert pool_manager.model_script == "test_model"
        assert pool_manager.config_hash is not None
        assert len(pool_manager.config_hash) == 64  # SHA256 full hex

    def test_pool_directory_created(self, pool_manager):
        """Test that pool directory is created."""
        assert pool_manager.pool_dir.exists()
        assert pool_manager.pool_dir.is_dir()

    def test_pool_directory_naming(self, pool_manager):
        """Test that pool directory follows naming convention."""
        pool_name = pool_manager.pool_dir.name
        assert pool_name.startswith("test_v1_")
        assert len(pool_name) == len("test_v1_") + 8  # version + 8-char hash

    def test_missing_priors_csv_error(self, temp_dir, sample_test_stats_csv):
        """Test that missing priors CSV raises error."""
        with pytest.raises(FileNotFoundError, match="Priors CSV not found"):
            SimulationPoolManager(
                cache_dir=temp_dir / "cache",
                model_version="test_v1",
                model_description="Test model",
                priors_csv=temp_dir / "nonexistent.csv",
                test_stats_csv=sample_test_stats_csv,
                model_script="test_model",
            )

    def test_missing_test_stats_csv_error(self, temp_dir, sample_priors_csv):
        """Test that missing test stats CSV raises error."""
        with pytest.raises(FileNotFoundError, match="Test statistics CSV not found"):
            SimulationPoolManager(
                cache_dir=temp_dir / "cache",
                model_version="test_v1",
                model_description="Test model",
                priors_csv=sample_priors_csv,
                test_stats_csv=temp_dir / "nonexistent.csv",
                model_script="test_model",
            )

    def test_config_hash_stability(self, temp_dir, sample_priors_csv, sample_test_stats_csv):
        """Test that config hash is stable for same configuration."""
        pool1 = SimulationPoolManager(
            cache_dir=temp_dir / "cache1",
            model_version="test_v1",
            model_description="Test model",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
            model_script="test_model",
        )
        pool2 = SimulationPoolManager(
            cache_dir=temp_dir / "cache2",
            model_version="test_v1",
            model_description="Test model",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
            model_script="test_model",
        )
        assert pool1.config_hash == pool2.config_hash

    def test_config_hash_changes_with_model_version(
        self, temp_dir, sample_priors_csv, sample_test_stats_csv
    ):
        """Test that config hash changes when model version changes."""
        pool1 = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v1",
            model_description="Test model",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
            model_script="test_model",
        )
        pool2 = SimulationPoolManager(
            cache_dir=temp_dir / "cache",
            model_version="v2",
            model_description="Test model",
            priors_csv=sample_priors_csv,
            test_stats_csv=sample_test_stats_csv,
            model_script="test_model",
        )
        assert pool1.config_hash != pool2.config_hash


class TestBatchScanning:
    """Tests for batch file scanning functionality."""

    def test_scan_empty_pool(self, pool_manager):
        """Test scanning an empty pool."""
        batches = pool_manager._scan_batches()
        assert batches == []

    def test_scan_single_batch(self, pool_manager):
        """Test scanning pool with single batch file."""
        # Create a batch file with correct naming
        batch_file = pool_manager.pool_dir / "batch_20250114_120530_gvax_1000sims_seed42.mat"
        savemat(batch_file, {"params": np.random.rand(1000, 3), "obs": np.random.rand(1000, 3)})

        batches = pool_manager._scan_batches()
        assert len(batches) == 1
        assert batches[0]["filename"] == batch_file.name
        assert batches[0]["scenario"] == "gvax"
        assert batches[0]["n_sims"] == 1000
        assert batches[0]["seed"] == 42

    def test_scan_multiple_batches(self, pool_manager):
        """Test scanning pool with multiple batch files."""
        # Create multiple batch files
        batch_files = [
            "batch_20250114_120530_gvax_1000sims_seed42.mat",
            "batch_20250114_130530_gvax_500sims_seed43.mat",
            "batch_20250114_140530_control_750sims_seed44.mat",
        ]

        for batch_name in batch_files:
            batch_file = pool_manager.pool_dir / batch_name
            savemat(batch_file, {"params": np.random.rand(100, 3), "obs": np.random.rand(100, 3)})

        batches = pool_manager._scan_batches()
        assert len(batches) == 3

    def test_scan_with_scenario_filter(self, pool_manager):
        """Test scanning with scenario filter."""
        # Create batch files for different scenarios
        savemat(
            pool_manager.pool_dir / "batch_20250114_120530_gvax_1000sims_seed42.mat",
            {"params": np.random.rand(100, 3), "obs": np.random.rand(100, 3)},
        )
        savemat(
            pool_manager.pool_dir / "batch_20250114_130530_control_500sims_seed43.mat",
            {"params": np.random.rand(100, 3), "obs": np.random.rand(100, 3)},
        )

        # Filter for gvax scenario
        batches = pool_manager._scan_batches(scenario="gvax")
        assert len(batches) == 1
        assert batches[0]["scenario"] == "gvax"

    def test_scan_ignores_invalid_filenames(self, pool_manager):
        """Test that scanning ignores files with invalid naming."""
        # Create files with invalid naming
        invalid_files = ["invalid_name.mat", "batch_wrong_format.mat", "not_a_batch.txt"]

        for filename in invalid_files:
            (pool_manager.pool_dir / filename).touch()

        batches = pool_manager._scan_batches()
        assert batches == []


class TestScenarioManagement:
    """Tests for scenario listing and management."""

    def test_list_scenarios_empty_pool(self, pool_manager):
        """Test listing scenarios in empty pool."""
        scenarios = pool_manager.list_scenarios()
        assert scenarios == []

    def test_list_scenarios_single_scenario(self, pool_manager):
        """Test listing scenarios with single scenario."""
        savemat(
            pool_manager.pool_dir / "batch_20250114_120530_gvax_1000sims_seed42.mat",
            {"params": np.random.rand(100, 3), "obs": np.random.rand(100, 3)},
        )

        scenarios = pool_manager.list_scenarios()
        assert scenarios == ["gvax"]

    def test_list_scenarios_multiple_scenarios(self, pool_manager):
        """Test listing scenarios with multiple scenarios."""
        scenarios_to_create = ["gvax", "control", "anti_pd1"]

        for i, scenario in enumerate(scenarios_to_create):
            batch_file = (
                pool_manager.pool_dir / f"batch_20250114_12{i:02d}30_{scenario}_1000sims_seed42.mat"
            )
            savemat(batch_file, {"params": np.random.rand(100, 3), "obs": np.random.rand(100, 3)})

        scenarios = pool_manager.list_scenarios()
        assert scenarios == sorted(scenarios_to_create)

    def test_list_scenarios_deduplicates(self, pool_manager):
        """Test that list_scenarios deduplicates scenarios."""
        # Create multiple batches for same scenario
        for i in range(3):
            batch_file = (
                pool_manager.pool_dir / f"batch_20250114_12{i:02d}30_gvax_1000sims_seed{42+i}.mat"
            )
            savemat(batch_file, {"params": np.random.rand(100, 3), "obs": np.random.rand(100, 3)})

        scenarios = pool_manager.list_scenarios()
        assert scenarios == ["gvax"]


class TestAvailableSimulations:
    """Tests for counting available simulations."""

    def test_available_simulations_empty_pool(self, pool_manager):
        """Test counting simulations in empty pool."""
        assert pool_manager.get_available_simulations() == 0

    def test_available_simulations_single_batch(self, pool_manager):
        """Test counting simulations with single batch."""
        savemat(
            pool_manager.pool_dir / "batch_20250114_120530_gvax_1000sims_seed42.mat",
            {"params": np.random.rand(1000, 3), "obs": np.random.rand(1000, 3)},
        )

        assert pool_manager.get_available_simulations(scenario="gvax") == 1000

    def test_available_simulations_multiple_batches(self, pool_manager):
        """Test counting simulations across multiple batches."""
        savemat(
            pool_manager.pool_dir / "batch_20250114_120530_gvax_1000sims_seed42.mat",
            {"params": np.random.rand(1000, 3), "obs": np.random.rand(1000, 3)},
        )
        savemat(
            pool_manager.pool_dir / "batch_20250114_130530_gvax_500sims_seed43.mat",
            {"params": np.random.rand(500, 3), "obs": np.random.rand(500, 3)},
        )

        assert pool_manager.get_available_simulations(scenario="gvax") == 1500

    def test_available_simulations_scenario_filter(self, pool_manager):
        """Test counting simulations with scenario filter."""
        savemat(
            pool_manager.pool_dir / "batch_20250114_120530_gvax_1000sims_seed42.mat",
            {"params": np.random.rand(1000, 3), "obs": np.random.rand(1000, 3)},
        )
        savemat(
            pool_manager.pool_dir / "batch_20250114_130530_control_750sims_seed43.mat",
            {"params": np.random.rand(750, 3), "obs": np.random.rand(750, 3)},
        )

        assert pool_manager.get_available_simulations(scenario="gvax") == 1000
        assert pool_manager.get_available_simulations(scenario="control") == 750


class TestBatchPatternParsing:
    """Tests for batch filename pattern parsing."""

    def test_pattern_matches_valid_filename(self, pool_manager):
        """Test that pattern matches valid batch filenames."""
        valid_names = [
            "batch_20250114_120530_gvax_1000sims_seed42.mat",
            "batch_20250101_000000_control_1sims_seed0.mat",
            "batch_20251231_235959_anti_pd1_999999sims_seed123.mat",
        ]

        for name in valid_names:
            match = pool_manager.batch_pattern.match(name)
            assert match is not None, f"Pattern should match: {name}"

    def test_pattern_rejects_invalid_filename(self, pool_manager):
        """Test that pattern rejects invalid batch filenames."""
        invalid_names = [
            "batch_invalid.mat",
            "wrong_format.mat",
            "batch_20250114_gvax_1000sims.mat",  # Missing seed
            "batch_20250114_120530_1000sims_seed42.mat",  # Missing scenario
        ]

        for name in invalid_names:
            match = pool_manager.batch_pattern.match(name)
            assert match is None, f"Pattern should not match: {name}"

    def test_pattern_extracts_correct_fields(self, pool_manager):
        """Test that pattern extracts correct fields."""
        filename = "batch_20250114_120530_gvax_1000sims_seed42.mat"
        match = pool_manager.batch_pattern.match(filename)

        assert match is not None
        timestamp, scenario, n_sims, seed = match.groups()

        assert timestamp == "20250114_120530"
        assert scenario == "gvax"
        assert n_sims == "1000"
        assert seed == "42"


class TestLoadAndAdd:
    """Tests for adding and loading batches from the pool."""

    def test_add_batch_creates_file(self, pool_manager):
        params = np.array([[1.0, 2.0], [3.0, 4.0]])
        observables = np.array([[10.0], [20.0]])

        filename = pool_manager.add_batch(params, observables, seed=7, scenario="gvax")
        batch_file = pool_manager.pool_dir / filename

        assert batch_file.exists()
        data = loadmat(batch_file)
        assert data["params_matrix"].shape == (2, 2)
        assert data["observables_matrix"].shape == (2, 1)

    def test_load_simulations_subset_sampling(self, pool_manager):
        params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        observables = np.array([[10], [20], [30], [40]])
        pool_manager.add_batch(params, observables, seed=1, scenario="gvax")

        rng = np.random.default_rng(0)
        loaded_params, loaded_obs = pool_manager.load_simulations(
            n_requested=2, scenario="gvax", random_state=rng
        )

        assert loaded_params.shape == (2, 3)
        assert loaded_obs.shape == (2, 1)

        expected_indices = np.random.default_rng(0).choice(4, size=2, replace=False)
        np.testing.assert_array_equal(loaded_obs.flatten(), observables.flatten()[expected_indices])

    def test_load_simulations_requires_scenario(self, pool_manager):
        with pytest.raises(ValueError):
            pool_manager.load_simulations(n_requested=1, scenario=None)

    def test_load_multi_scenario(self, pool_manager):
        params = np.array([[1.0], [2.0], [3.0]])
        obs = np.array([[10.0], [20.0], [30.0]])
        pool_manager.add_batch(params, obs, seed=1, scenario="gvax")
        pool_manager.add_batch(params + 10, obs + 10, seed=2, scenario="control")

        rng = np.random.default_rng(1)
        result = pool_manager.load_multi_scenario(
            ["gvax", "control"], n_requested=2, random_state=rng
        )

        assert set(result.keys()) == {"gvax", "control"}
        assert result["gvax"][0].shape[0] == 2
        assert result["control"][0].shape[0] == 2

    def test_over_request_returns_available(self, pool_manager):
        params = np.array([[1.0, 2.0], [3.0, 4.0]])
        obs = np.array([[10.0], [20.0]])
        pool_manager.add_batch(params, obs, seed=1, scenario="gvax")

        loaded_params, loaded_obs = pool_manager.load_simulations(
            n_requested=5, scenario="gvax", random_state=np.random.default_rng(0)
        )

        assert loaded_params.shape == (2, 2)
        assert loaded_obs.shape == (2, 1)

    def test_corrupt_batch_raises(self, pool_manager):
        corrupt_file = pool_manager.pool_dir / "batch_20250114_120530_gvax_2sims_seed1.mat"
        corrupt_file.write_text("not a mat file")

        with pytest.raises(ValueError):
            pool_manager.load_simulations(n_requested=1, scenario="gvax")

    def test_missing_required_keys_raise(self, pool_manager):
        # Write a .mat without required keys
        bad_file = pool_manager.pool_dir / "batch_20250114_120531_gvax_2sims_seed1.mat"
        savemat(bad_file, {"foo": np.array([1, 2])})

        with pytest.raises(ValueError, match="required keys"):
            pool_manager.load_simulations(n_requested=1, scenario="gvax")

    def test_load_multi_scenario_requires_all(self, pool_manager):
        params = np.array([[1.0], [2.0]])
        obs = np.array([[10.0], [20.0]])
        pool_manager.add_batch(params, obs, seed=1, scenario="gvax")

        with pytest.raises(ValueError):
            pool_manager.load_multi_scenario(["gvax", "missing"], n_requested=1)

    def test_add_batch_metadata_and_shape_normalization(self, pool_manager):
        params = np.array([1.0, 2.0])  # 1D arrays should be reshaped
        obs = np.array([10.0, 20.0])

        filename = pool_manager.add_batch(params, obs, seed=3, scenario="gvax")
        data = loadmat(pool_manager.pool_dir / filename)

        assert data["params_matrix"].shape == (1, 2)
        assert data["observables_matrix"].shape == (1, 2)
        metadata = data["metadata"]
        # metadata is stored as a numpy object; ensure keys exist
        assert "scenario" in metadata.dtype.names
        assert "seed" in metadata.dtype.names


class TestListPools:
    """Tests for listing pools with noisy cache directories."""

    def test_list_pools_ignores_noise(self, pool_manager):
        # Add a valid batch with underscored scenario
        params = np.array([[1.0, 2.0]])
        obs = np.array([[10.0]])
        pool_manager.add_batch(params, obs, seed=5, scenario="gvax_pd1")

        # Add noise: unrelated files/dirs
        (pool_manager.cache_dir / "random.txt").write_text("noise")
        (pool_manager.cache_dir / "not_a_pool").mkdir()

        pools = SimulationPoolManager.list_pools(pool_manager.cache_dir)

        assert len(pools) == 1
        info = pools[0]
        assert "gvax_pd1" in info["scenarios"]
        assert info["n_batches"] == 1
        assert info["total_simulations"] == 1
