"""Pytest configuration and shared fixtures."""

import shutil
import tempfile
from pathlib import Path

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
