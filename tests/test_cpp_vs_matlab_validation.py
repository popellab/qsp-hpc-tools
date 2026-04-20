"""M7 numerical validation: C++ qsp_sim vs MATLAB sbiosimulate.

Wraps ``scripts/validate_cpp_vs_matlab.py`` as a pytest marker-gated test.
Requires:
  - MATLAB on PATH
  - qsp_sim binary built at the expected SPQSP_PDAC-cpp-sweep location
  - pdac-build project root with immune_oncology_model_PDAC + priors

Enable with::

    pytest -m validation --run-validation tests/test_cpp_vs_matlab_validation.py

Skipped by default so CI doesn't drag in MATLAB / C++ dependencies.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

PRIORS_CSV = Path("/Users/joeleliason/Projects/pdac-build/parameters/pdac_priors.csv")
MATLAB_PROJECT_ROOT = Path("/Users/joeleliason/Projects/pdac-build")
CPP_BINARY = Path("/Users/joeleliason/Projects/SPQSP_PDAC-cpp-sweep/PDAC/qsp/sim/build/qsp_sim")
CPP_TEMPLATE = Path(
    "/Users/joeleliason/Projects/SPQSP_PDAC-cpp-sweep/PDAC/sim/resource/param_all.xml"
)


@pytest.mark.validation
def test_cpp_matches_matlab_20sims_30days(tmp_path):
    """Run 20 sims × 30 days on both paths and assert trajectories agree.

    Thresholds chosen from the 2026-04-15 baseline run:
      - median Pearson r across meaningful-magnitude species: 1.0
      - median max_rel_diff: ~2e-6
      - p95 max_rel_diff: ~2e-4
    We assert somewhat looser bounds so CVODE jitter across machines
    doesn't flake the test, but still catch regressions that would
    produce order-of-magnitude disagreement.
    """
    for p in (PRIORS_CSV, MATLAB_PROJECT_ROOT, CPP_BINARY, CPP_TEMPLATE):
        if not p.exists():
            pytest.skip(f"required path missing: {p}")
    if shutil.which("matlab") is None:
        pytest.skip("matlab not on PATH")

    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    try:
        from validate_cpp_vs_matlab import (
            DEFAULT_SAMPLED_PARAMS,
            compare_parquets,
            run_cpp,
            run_matlab,
            sample_params,
        )
    finally:
        sys.path.pop(0)

    theta, names = sample_params(
        priors_csv=PRIORS_CSV,
        param_names=DEFAULT_SAMPLED_PARAMS,
        n_sims=20,
        seed=42,
    )

    matlab_parquet, matlab_time = run_matlab(
        theta=theta,
        param_names=names,
        project_root=MATLAB_PROJECT_ROOT,
        model_script="immune_oncology_model_PDAC",
        t_end_days=30.0,
        dt_days=0.5,
        seed=42,
        out_dir=tmp_path,
    )
    cpp_parquet, cpp_time = run_cpp(
        theta=theta,
        param_names=names,
        binary_path=CPP_BINARY,
        template_path=CPP_TEMPLATE,
        t_end_days=30.0,
        dt_days=0.5,
        seed=42,
        out_dir=tmp_path,
    )

    report = compare_parquets(matlab_parquet, cpp_parquet)

    # Filter to meaningful-magnitude species (matches the script's convention).
    noise_floor = 1e-9
    meaningful = report[
        (report["matlab_final_mean"].abs() > noise_floor)
        | (report["cpp_final_mean"].abs() > noise_floor)
    ]

    assert (
        len(meaningful) >= 100
    ), f"expected ≥100 meaningful-magnitude species, got {len(meaningful)}"
    assert (
        meaningful["pearson_r"].median() > 0.9999
    ), f"median Pearson r dropped: {meaningful['pearson_r'].median()}"
    assert (
        meaningful["max_rel_diff"].median() < 1e-4
    ), f"median max_rel_diff drifted: {meaningful['max_rel_diff'].median()}"
    assert (
        meaningful["max_rel_diff"].quantile(0.95) < 1e-2
    ), f"p95 max_rel_diff drifted: {meaningful['max_rel_diff'].quantile(0.95)}"

    # C++ should be meaningfully faster than MATLAB (not a strict threshold —
    # just catches catastrophic C++ regressions that would make the whole
    # exercise pointless).
    assert (
        cpp_time * 5 < matlab_time
    ), f"C++ unexpectedly slow: matlab={matlab_time:.1f}s vs cpp={cpp_time:.1f}s"
