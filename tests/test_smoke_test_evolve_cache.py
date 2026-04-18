"""Regression tests for scripts/smoke_test_evolve_cache_hpc.py.

The smoke test lives in scripts/ and normally isn't exercised in CI,
but a silent drift in `build_tiny_params_csv` (e.g. reading the wrong
priors column) crashes every qsp_sim invocation on HPC while looking
superficially fine locally. These tests guard the few pure-Python
invariants that catch the drift without needing a live HPC.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SMOKE_TEST_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "smoke_test_evolve_cache_hpc.py"
)


def _import_smoke_module():
    """Import scripts/smoke_test_evolve_cache_hpc.py without running main."""
    spec = importlib.util.spec_from_file_location("smoke_evolve_cache", _SMOKE_TEST_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke_evolve_cache"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_tiny_params_csv_uses_median_column(tmp_path: Path):
    """`build_tiny_params_csv` must pull from ``median``, not ``dist_param1``.

    Regression guard for #34 diagnostic rerun: an earlier version of
    this function read ``dist_param1`` (the lognormal μ — mean-of-log,
    often negative). Negative μ applied as a rate-constant value crashed
    CVode at t≈0 and aborted every qsp_sim invocation. The fix: use
    ``median``, which is the parameter value in its natural units.
    """
    smoke = _import_smoke_module()

    # Realistic shape: three lognormal-style rows where dist_param1 is
    # NEGATIVE but median is POSITIVE. If the function reads the wrong
    # column we'll catch it: all output values would be negative.
    priors = pd.DataFrame(
        {
            "name": ["k_a", "k_b", "k_c"],
            "median": [0.05, 0.1, 2.0],
            "distribution": ["lognormal", "lognormal", "lognormal"],
            "dist_param1": [-2.9957, -2.3026, 0.6931],  # log(median)
            "dist_param2": [1.2, 1.2, 0.5],
        }
    )
    priors_csv = tmp_path / "priors.csv"
    priors.to_csv(priors_csv, index=False)

    out_csv = tmp_path / "params.csv"
    names = smoke.build_tiny_params_csv(priors_csv, out_csv, n_sims=4)

    assert names == ["k_a", "k_b", "k_c"]
    out = pd.read_csv(out_csv)
    assert list(out.columns) == ["k_a", "k_b", "k_c"]
    assert len(out) == 4

    # Every row identical (the whole point of the test — shared theta).
    np.testing.assert_array_equal(out.values, np.tile(out.iloc[0].values, (4, 1)))

    # Values match the MEDIAN column, not dist_param1. This is the
    # critical invariant — swapping columns would have every value
    # negative here (dist_param1 is log-space).
    np.testing.assert_allclose(out.iloc[0].values, [0.05, 0.1, 2.0])
    assert (out.values > 0).all(), "smoke-test theta must be strictly positive"
