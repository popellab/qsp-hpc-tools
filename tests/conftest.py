"""Pytest configuration and shared fixtures."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def pytest_addoption(parser):
    """Register CLI options for optional test groups."""
    parser.addoption(
        "--run-hpc",
        action="store_true",
        default=False,
        help="Run tests marked with 'hpc' that require a real cluster",
    )


def pytest_configure(config):
    """Expose custom markers even if pytest.ini is not read."""
    config.addinivalue_line("markers", "hpc: tests that require a real HPC cluster")


def pytest_runtest_setup(item):
    """Skip HPC tests unless explicitly enabled."""
    if "hpc" in {mark.name for mark in item.iter_markers()}:
        if not item.config.getoption("--run-hpc"):
            pytest.skip("HPC tests skipped (use --run-hpc to enable)")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_priors():
    """Sample prior parameters for testing."""
    return pd.DataFrame(
        {"param1": [1.0, 2.0, 3.0], "param2": [0.5, 1.5, 2.5], "param3": [10.0, 20.0, 30.0]}
    )


@pytest.fixture
def sample_test_stats():
    """Sample test statistics for testing."""
    return pd.DataFrame(
        {"stat1": [0.1, 0.2, 0.3], "stat2": [1.0, 2.0, 3.0], "stat3": [100.0, 200.0, 300.0]}
    )


@pytest.fixture
def sample_params():
    """Sample parameter sets for simulation."""
    return pd.DataFrame(
        {
            "param1": np.random.uniform(0, 1, 10),
            "param2": np.random.uniform(0, 10, 10),
            "param3": np.random.uniform(100, 1000, 10),
        }
    )


@pytest.fixture
def sample_simulation_batch(temp_dir):
    """Create sample simulation batch files for testing."""
    batch_dir = temp_dir / "simulations"
    batch_dir.mkdir()

    # Create some sample parquet files with typical naming pattern
    for i in range(5):
        df = pd.DataFrame(
            {
                "param1": [np.random.uniform(0, 1)],
                "param2": [np.random.uniform(0, 10)],
                "result": [np.random.uniform(0, 100)],
            }
        )
        filename = f"batch_{i:04d}_scenario_default_task_0.parquet"
        df.to_parquet(batch_dir / filename)

    return batch_dir


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "ssh_host": "test-hpc.example.com",
        "ssh_user": "testuser",
        "remote_base_dir": "/home/testuser/qsp-hpc",
        "slurm": {
            "partition": "test-partition",
            "nodes": 1,
            "cpus_per_task": 4,
            "memory": "16GB",
            "time": "01:00:00",
        },
    }


@pytest.fixture
def sample_model_context():
    """Sample model context for hash testing."""
    return {
        "model_name": "test_model",
        "solver": "ode45",
        "tspan": [0, 100],
        "parameters": {"param1": 1.0, "param2": 2.0},
    }


@pytest.fixture(autouse=True)
def ssh_rate_limit(request):
    """Add delay between HPC tests to avoid SSH rate limiting."""
    import time

    # Only apply to HPC tests
    if "hpc" in [mark.name for mark in request.node.iter_markers()]:
        yield
        # Small delay after each HPC test to avoid rate limiting
        time.sleep(2)
    else:
        yield
