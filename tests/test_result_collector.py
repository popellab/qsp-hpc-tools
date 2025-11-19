#!/usr/bin/env python3
"""
Tests for ResultCollector.

Tests result collection, parsing, and downloading from HPC including
error handling for missing files, corrupt data, and network failures.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from qsp_hpc.batch.hpc_job_manager import BatchConfig
from qsp_hpc.batch.result_collector import MissingOutputError, ResultCollector


@pytest.fixture
def mock_config():
    """Create mock BatchConfig."""
    return BatchConfig(
        ssh_host="hpc.example.edu",
        ssh_user="testuser",
        ssh_key="~/.ssh/id_rsa",
        remote_project_path="/home/testuser/qsp-projects",
        hpc_venv_path="/home/testuser/.venv/hpc-qsp",
        simulation_pool_path="/scratch/testuser/simulations",
    )


@pytest.fixture
def mock_transport():
    """Create mock SSH transport."""
    transport = Mock()
    transport.exec = Mock()
    transport.download = Mock()
    return transport


@pytest.fixture
def result_collector(mock_config, mock_transport):
    """Create ResultCollector instance with mocks."""
    return ResultCollector(config=mock_config, transport=mock_transport, verbose=False)


class TestCheckPoolDirectoryExists:
    """Tests for check_pool_directory_exists method."""

    def test_directory_exists(self, result_collector, mock_transport):
        """Test checking for existing directory."""
        mock_transport.exec.return_value = (0, "exists\n")

        exists = result_collector.check_pool_directory_exists(
            "/scratch/testuser/simulations/test_pool"
        )

        assert exists is True
        mock_transport.exec.assert_called_once()

    def test_directory_not_exists(self, result_collector, mock_transport):
        """Test checking for non-existent directory."""
        mock_transport.exec.return_value = (0, "not_found\n")

        exists = result_collector.check_pool_directory_exists(
            "/scratch/testuser/simulations/missing_pool"
        )

        assert exists is False

    def test_ssh_command_failure(self, result_collector, mock_transport):
        """Test handling of SSH command failure."""
        mock_transport.exec.return_value = (1, "error\n")

        exists = result_collector.check_pool_directory_exists(
            "/scratch/testuser/simulations/test_pool"
        )

        # Should still work - not_found is not in output
        assert exists is True  # Because 'not_found' not in 'error'


class TestCountPoolSimulations:
    """Tests for count_pool_simulations method."""

    def test_count_from_manifest(self, result_collector, mock_transport):
        """Test counting simulations from manifest.json."""
        manifest = {"total_simulations": 1000, "n_batches": 10}
        output = f"MANIFEST_FOUND\n{json.dumps(manifest)}"
        mock_transport.exec.return_value = (0, output)

        count = result_collector.count_pool_simulations("/scratch/testuser/simulations/test_pool")

        assert count == 1000

    def test_count_from_filenames(self, result_collector, mock_transport):
        """Test counting simulations from filenames when no manifest."""
        output = """COUNTING_FILES
N_FILES:5
N_SIMS:500"""
        mock_transport.exec.return_value = (0, output)

        count = result_collector.count_pool_simulations("/scratch/testuser/simulations/test_pool")

        assert count == 500

    def test_count_no_files(self, result_collector, mock_transport):
        """Test counting when no simulation files exist."""
        output = """COUNTING_FILES
N_FILES:0
N_SIMS:0"""
        mock_transport.exec.return_value = (0, output)

        count = result_collector.count_pool_simulations("/scratch/testuser/simulations/test_pool")

        assert count == 0

    def test_count_command_failure(self, result_collector, mock_transport):
        """Test handling of count command failure."""
        mock_transport.exec.return_value = (1, "error")

        count = result_collector.count_pool_simulations("/scratch/testuser/simulations/test_pool")

        assert count == 0

    def test_count_corrupt_manifest(self, result_collector, mock_transport):
        """Test handling of corrupt manifest JSON."""
        output = "MANIFEST_FOUND\n{invalid json}"
        mock_transport.exec.return_value = (0, output)

        count = result_collector.count_pool_simulations("/scratch/testuser/simulations/test_pool")

        # Should gracefully handle JSON parse error
        assert count == 0

    def test_count_manifest_missing_field(self, result_collector, mock_transport):
        """Test handling of manifest without total_simulations field."""
        manifest = {"n_batches": 10}  # Missing total_simulations
        output = f"MANIFEST_FOUND\n{json.dumps(manifest)}"
        mock_transport.exec.return_value = (0, output)

        count = result_collector.count_pool_simulations("/scratch/testuser/simulations/test_pool")

        assert count == 0

    def test_count_malformed_filename_output(self, result_collector, mock_transport):
        """Test handling of malformed filename counting output."""
        output = """COUNTING_FILES
N_FILES:5
N_SIMS:not_a_number"""
        mock_transport.exec.return_value = (0, output)

        count = result_collector.count_pool_simulations("/scratch/testuser/simulations/test_pool")

        # Should gracefully handle parse error
        assert count == 0

    def test_count_empty_output(self, result_collector, mock_transport):
        """Test handling of empty command output."""
        mock_transport.exec.return_value = (0, "")

        count = result_collector.count_pool_simulations("/scratch/testuser/simulations/test_pool")

        assert count == 0


class TestCheckHPCFullSimulations:
    """Tests for check_hpc_full_simulations method."""

    def test_has_enough_simulations(self, result_collector, mock_transport):
        """Test when HPC pool has enough simulations."""
        # Directory exists
        mock_transport.exec.side_effect = [
            (0, "exists\n"),  # check directory
            (0, "COUNTING_FILES\nN_FILES:10\nN_SIMS:1000"),  # count sims
        ]

        has_enough, pool_path, n_available = result_collector.check_hpc_full_simulations(
            model_version="baseline_pdac", priors_hash="abc123defg", n_requested=500
        )

        assert has_enough is True
        assert pool_path == "/scratch/testuser/simulations/baseline_pdac_abc123de"
        assert n_available == 1000

    def test_insufficient_simulations(self, result_collector, mock_transport):
        """Test when HPC pool has insufficient simulations."""
        mock_transport.exec.side_effect = [
            (0, "exists\n"),  # check directory
            (0, "COUNTING_FILES\nN_FILES:3\nN_SIMS:300"),  # count sims
        ]

        has_enough, pool_path, n_available = result_collector.check_hpc_full_simulations(
            model_version="baseline_pdac", priors_hash="abc123defg", n_requested=500
        )

        assert has_enough is False
        assert n_available == 300

    def test_pool_not_exists(self, result_collector, mock_transport):
        """Test when pool directory doesn't exist on HPC."""
        mock_transport.exec.return_value = (0, "not_found\n")

        has_enough, pool_path, n_available = result_collector.check_hpc_full_simulations(
            model_version="baseline_pdac", priors_hash="abc123defg", n_requested=500
        )

        assert has_enough is False
        assert n_available == 0

    def test_hash_truncation(self, result_collector, mock_transport):
        """Test that priors_hash is correctly truncated to 8 chars in pool name."""
        mock_transport.exec.return_value = (0, "not_found\n")

        has_enough, pool_path, n_available = result_collector.check_hpc_full_simulations(
            model_version="baseline_pdac",
            priors_hash="abcdefghijklmnop",
            n_requested=100,  # 16 chars
        )

        # Pool path should use only first 8 chars of hash
        assert pool_path == "/scratch/testuser/simulations/baseline_pdac_abcdefgh"


class TestCheckHPCTestStats:
    """Tests for check_hpc_test_stats method."""

    def test_test_stats_exist(self, result_collector, mock_transport):
        """Test when test statistics files exist."""
        mock_transport.exec.return_value = (0, "exists\n")

        exists = result_collector.check_hpc_test_stats(
            pool_path="/scratch/testuser/simulations/test_pool", test_stats_hash="xyz789abc"
        )

        assert exists is True

    def test_test_stats_missing(self, result_collector, mock_transport):
        """Test when test statistics files are missing."""
        mock_transport.exec.return_value = (1, "")

        exists = result_collector.check_hpc_test_stats(
            pool_path="/scratch/testuser/simulations/test_pool", test_stats_hash="xyz789abc"
        )

        assert exists is False

    def test_test_stats_with_count_validation(self, result_collector, mock_transport):
        """Test test stats existence with expected simulation count validation."""
        mock_transport.exec.side_effect = [
            (0, "exists\n"),  # Both files exist
            (0, "1001\n"),  # 1000 sims + 1 header = 1001 lines
        ]

        exists = result_collector.check_hpc_test_stats(
            pool_path="/scratch/testuser/simulations/test_pool",
            test_stats_hash="xyz789abc",
            expected_n_sims=1000,
        )

        assert exists is True

    def test_test_stats_count_mismatch(self, result_collector, mock_transport):
        """Test when test stats file has fewer rows than expected."""
        mock_transport.exec.side_effect = [
            (0, "exists\n"),  # Both files exist
            (0, "501\n"),  # 500 sims + 1 header = 501 lines
        ]

        exists = result_collector.check_hpc_test_stats(
            pool_path="/scratch/testuser/simulations/test_pool",
            test_stats_hash="xyz789abc",
            expected_n_sims=1000,  # Expect 1000 but only 500 exist
        )

        assert exists is False

    def test_test_stats_count_validation_failed_command(self, result_collector, mock_transport):
        """Test when count validation command fails."""
        mock_transport.exec.side_effect = [
            (0, "exists\n"),  # Both files exist
            (1, "error"),  # wc command fails
        ]

        exists = result_collector.check_hpc_test_stats(
            pool_path="/scratch/testuser/simulations/test_pool",
            test_stats_hash="xyz789abc",
            expected_n_sims=1000,
        )

        # Should return True - files exist, validation command failed but doesn't block
        # The code doesn't treat a failed validation command as False, only missing files
        assert exists is True


class TestDownloadTestStats:
    """Tests for download_test_stats method."""

    def test_download_test_stats_success(self, result_collector, mock_transport, tmp_path):
        """Test successful download and loading of test statistics."""
        # Create mock CSV files
        params_csv = tmp_path / "test_stats_xyz789ab_params.csv"
        stats_csv = tmp_path / "test_stats_xyz789ab.csv"

        params_csv.write_text("param1,param2\n0.5,1.0\n0.7,1.2\n")
        stats_csv.write_text("stat1,stat2\n10.0,20.0\n15.0,25.0\n")

        # Mock download to copy from tmp_path
        def mock_download(remote_file, local_dir):
            filename = Path(remote_file).name
            src = tmp_path / filename
            dst = Path(local_dir) / filename
            dst.write_text(src.read_text())

        mock_transport.download.side_effect = mock_download

        # Download test stats
        local_cache = tmp_path / "cache"
        params, test_stats = result_collector.download_test_stats(
            pool_path="/scratch/testuser/simulations/test_pool",
            test_stats_hash="xyz789abc",
            local_cache_dir=local_cache,
        )

        # Verify arrays
        assert params.shape == (2, 2)
        assert test_stats.shape == (2, 2)
        assert params[0, 0] == pytest.approx(0.5)
        assert test_stats[0, 0] == pytest.approx(10.0)

        # Verify files were renamed
        assert (local_cache / "params.csv").exists()
        assert (local_cache / "test_stats.csv").exists()

    def test_download_creates_cache_directory(self, result_collector, mock_transport, tmp_path):
        """Test that download creates cache directory if it doesn't exist."""
        params_csv = tmp_path / "test_stats_xyz789ab_params.csv"
        stats_csv = tmp_path / "test_stats_xyz789ab.csv"

        params_csv.write_text("param1\n0.5\n")
        stats_csv.write_text("stat1\n10.0\n")

        def mock_download(remote_file, local_dir):
            filename = Path(remote_file).name
            src = tmp_path / filename
            dst = Path(local_dir) / filename
            dst.write_text(src.read_text())

        mock_transport.download.side_effect = mock_download

        # Use non-existent cache directory
        local_cache = tmp_path / "new_cache" / "subdir"
        params, test_stats = result_collector.download_test_stats(
            pool_path="/scratch/testuser/simulations/test_pool",
            test_stats_hash="xyz789abc",
            local_cache_dir=local_cache,
        )

        # Verify directory was created
        assert local_cache.exists()
        assert local_cache.is_dir()


class TestResultCollectorVerboseMode:
    """Tests for verbose logging in ResultCollector."""

    def test_verbose_logging_enabled(self, mock_config, mock_transport):
        """Test that verbose mode enables debug logging."""
        collector = ResultCollector(config=mock_config, transport=mock_transport, verbose=True)

        assert collector.verbose is True

    def test_verbose_logging_disabled(self, mock_config, mock_transport):
        """Test that verbose mode can be disabled."""
        collector = ResultCollector(config=mock_config, transport=mock_transport, verbose=False)

        assert collector.verbose is False


class TestMissingOutputError:
    """Tests for MissingOutputError exception."""

    def test_exception_creation(self):
        """Test creating MissingOutputError exception."""
        error = MissingOutputError("Test error message")
        assert isinstance(error, RuntimeError)
        assert str(error) == "Test error message"

    def test_exception_raising(self):
        """Test raising MissingOutputError."""
        with pytest.raises(MissingOutputError) as exc_info:
            raise MissingOutputError("Missing required files")

        assert "Missing required files" in str(exc_info.value)
