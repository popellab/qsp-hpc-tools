"""Tests for derive_test_stats_worker.py (CSV-based function loading)."""

import numpy as np
import pandas as pd
import pytest

from qsp_hpc.batch.derive_test_stats_worker import (
    build_test_stat_registry,
    compute_test_statistics_batch,
    process_single_batch,
)


class TestBuildTestStatRegistry:
    """Test building test statistic registry from CSV model_output_code column."""

    def test_simple_function(self):
        """Test loading a simple test statistic function from CSV."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["simple_mean"],
                "required_species": ["V_T.C1"],
                "model_output_code": [
                    """def compute_test_statistic(time, species_dict, ureg):
    return np.mean(species_dict['V_T.C1'].magnitude)"""
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        assert "simple_mean" in registry
        assert callable(registry["simple_mean"])

        # Test the function works - now uses species_dict signature
        from qsp_hpc.utils.unit_registry import ureg

        time = np.array([0, 1, 2, 3]) * ureg.day
        species_dict = {"V_T.C1": np.array([10, 20, 30, 40]) * ureg.cell}
        result = registry["simple_mean"](time, species_dict, ureg)
        assert result == 25.0

    def test_multiple_species(self):
        """Test function with multiple species arguments."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["ratio_test"],
                "required_species": ["V_T.CD8, V_T.Treg"],
                "model_output_code": [
                    """def compute_test_statistic(time, species_dict, ureg):
    cd8_0 = np.interp(0.0, time.magnitude, species_dict['V_T.CD8'].magnitude)
    treg_0 = np.interp(0.0, time.magnitude, species_dict['V_T.Treg'].magnitude)
    return cd8_0 / max(treg_0, 1e-12)"""
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        from qsp_hpc.utils.unit_registry import ureg

        time = np.array([0.0, 1.0, 2.0]) * ureg.day
        species_dict = {
            "V_T.CD8": np.array([100.0, 110.0, 120.0]) * ureg.cell,
            "V_T.Treg": np.array([50.0, 55.0, 60.0]) * ureg.cell,
        }

        result = registry["ratio_test"](time, species_dict, ureg)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_growth_rate_calculation(self):
        """Test realistic growth rate function."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["log_growth_rate"],
                "required_species": ["V_T.C1"],
                "model_output_code": [
                    """def compute_test_statistic(time, species_dict, ureg):
    t0, t1 = 0, 60
    time_vals = time.magnitude
    V_T_C1 = species_dict['V_T.C1'].magnitude
    if len(time_vals) == 0 or len(V_T_C1) == 0:
        return np.nan

    t_eval = np.arange(t0, t1 + 1, 1.0)
    c1_interp = np.interp(t_eval, time_vals, V_T_C1)
    c1_interp = np.maximum(c1_interp, np.finfo(float).eps)

    y = np.log(c1_interp)
    slope = np.polyfit(t_eval, y, 1)[0]
    return slope"""
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        from qsp_hpc.utils.unit_registry import ureg

        # Exponential growth: C(t) = c0 * exp(r*t) with r=0.01
        time = np.linspace(0, 60, 61) * ureg.day
        c0 = 1e6
        r = 0.01
        tumor_cells = c0 * np.exp(r * np.linspace(0, 60, 61)) * ureg.cell
        species_dict = {"V_T.C1": tumor_cells}

        result = registry["log_growth_rate"](time, species_dict, ureg)
        assert result == pytest.approx(r, rel=1e-2)

    def test_missing_model_output_code_column(self):
        """Test that missing model_output_code column raises error."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["legacy_stat"],
                "required_species": ["V_T.C1"],
                # Missing model_output_code column
            }
        )

        with pytest.raises(ValueError, match="missing required 'model_output_code' column"):
            build_test_stat_registry(test_stats_df)

    def test_nan_model_output_code(self):
        """Test that NaN model_output_code raises error."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["missing_func"],
                "required_species": ["V_T.C1"],
                "model_output_code": [np.nan],
            }
        )

        with pytest.raises(ValueError, match="has empty model_output_code"):
            build_test_stat_registry(test_stats_df)

    def test_invalid_function_code(self):
        """Test that invalid Python code raises error."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["bad_syntax"],
                "required_species": ["V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg)\n    return ???"
                ],  # Syntax error
            }
        )

        with pytest.raises(SyntaxError):
            build_test_stat_registry(test_stats_df)

    def test_missing_compute_function(self):
        """Test that code without 'compute_test_statistic' function raises error."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["wrong_name"],
                "required_species": ["V_T.C1"],
                "model_output_code": [
                    """def my_function(time, species_dict, ureg):
    return np.mean(species_dict['V_T.C1'].magnitude)"""
                ],  # Wrong function name
            }
        )

        with pytest.raises(
            ValueError, match="must define a function named 'compute_test_statistic'"
        ):
            build_test_stat_registry(test_stats_df)

    def test_multiple_functions(self):
        """Test loading multiple functions into registry."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["stat1", "stat2", "stat3"],
                "required_species": ["V_T.C1", "V_T.C1", "V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.mean(species_dict['V_T.C1'].magnitude)",
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.max(species_dict['V_T.C1'].magnitude)",
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.min(species_dict['V_T.C1'].magnitude)",
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        assert len(registry) == 3
        assert "stat1" in registry
        assert "stat2" in registry
        assert "stat3" in registry

        from qsp_hpc.utils.unit_registry import ureg

        time = np.array([0, 1, 2]) * ureg.day
        species_dict = {"V_T.C1": np.array([10, 20, 30]) * ureg.cell}

        assert registry["stat1"](time, species_dict, ureg) == 20.0
        assert registry["stat2"](time, species_dict, ureg) == 30.0
        assert registry["stat3"](time, species_dict, ureg) == 10.0


class TestTemplateDefaultsFallback:
    """#23: when a calibration-target function asks for a parameter that
    isn't a parquet column (thin-parquet pools), the derive worker
    resolves it via the pool manifest's template_defaults."""

    def test_manifest_default_resolves_missing_param(self):
        # cal-target references rho_collagen, which isn't in the parquet
        # (not sampled) — should come from template_defaults.
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["k_times_rho"],
                "required_species": ["V_T.C1, rho_collagen"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return species_dict['rho_collagen'].magnitude "
                    "* np.mean(species_dict['V_T.C1'].magnitude)"
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0],
                "status": [0],
                "time": [[0.0, 1.0, 2.0]],
                "V_T.C1": [[10.0, 20.0, 30.0]],  # mean = 20
                # NO param:rho_collagen column — the thin-parquet case.
            }
        )
        species_units = {"V_T.C1": "cell", "rho_collagen": "dimensionless"}
        template_defaults = {"rho_collagen": 0.5}

        result = compute_test_statistics_batch(
            sim_df,
            test_stats_df,
            registry,
            species_units,
            template_defaults=template_defaults,
        )
        # 0.5 (manifest default) × 20 (mean V_T.C1) = 10
        assert result[0, 0] == pytest.approx(10.0)

    def test_parquet_column_wins_over_manifest(self):
        """When a param:* column IS in the parquet (sampled), it takes
        precedence over the manifest default. Test with an unambiguous
        value difference."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["return_k"],
                "required_species": ["k"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return species_dict['k'].magnitude"
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0],
                "status": [0],
                "time": [[0.0]],
                "param:k": [7.5],  # sampled value
            }
        )
        species_units = {"k": "dimensionless"}
        template_defaults = {"k": 0.1}  # manifest default should be IGNORED

        result = compute_test_statistics_batch(
            sim_df,
            test_stats_df,
            registry,
            species_units,
            template_defaults=template_defaults,
        )
        assert result[0, 0] == pytest.approx(7.5)

    def test_missing_param_without_manifest_raises(self):
        """Pre-#23 behavior: without a manifest, a missing param raises
        ValueError (via the .warning + nan-row path in the inner loop).
        Cal-target functions that depend on a now-absent column blow up
        loudly instead of silently returning garbage."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["needs_z"],
                "required_species": ["z"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return species_dict['z'].magnitude"
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0],
                "status": [0],
                "time": [[0.0]],
            }
        )
        species_units = {}
        # No template_defaults supplied → should NaN (error logged + caught).
        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)
        assert np.isnan(result[0, 0])


class TestComputeTestStatisticsBatch:
    """Test computing test statistics from simulation DataFrame."""

    def test_basic_computation(self):
        """Test basic test statistic computation from simulation data."""
        # Create test statistics definition (using dot notation for species names)
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["mean_tumor"],
                "required_species": ["V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.mean(species_dict['V_T.C1'].magnitude)"
                ],
            }
        )

        # Build registry
        registry = build_test_stat_registry(test_stats_df)

        # Create simulation data (2 sims, 1 test stat) - species names use dots
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0, 1],
                "status": [0, 0],  # Both successful (status==0 = success)
                "time": [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                "V_T.C1": [[100.0, 200.0, 300.0], [50.0, 100.0, 150.0]],  # Mean = 200  # Mean = 100
            }
        )

        # Species units mapping
        species_units = {"V_T.C1": "cell"}

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)

        assert result.shape == (2, 1)
        assert result[0, 0] == pytest.approx(200.0)
        assert result[1, 0] == pytest.approx(100.0)

    def test_failed_simulation(self):
        """Test that failed simulations get NaN values."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["mean_tumor"],
                "required_species": ["V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.mean(species_dict['V_T.C1'].magnitude)"
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        # Simulation 1 failed (status==1 = failure)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0, 1],
                "status": [0, 1],  # First successful, second failed
                "time": [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                "V_T.C1": [[100.0, 200.0, 300.0], [50.0, 100.0, 150.0]],
            }
        )

        species_units = {"V_T.C1": "cell"}

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)

        assert result.shape == (2, 1)
        assert result[0, 0] == pytest.approx(200.0)
        assert np.isnan(result[1, 0])  # Failed simulation

    def test_multiple_test_stats(self):
        """Test computing multiple test statistics per simulation."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["mean_tumor", "max_tumor", "min_tumor"],
                "required_species": ["V_T.C1", "V_T.C1", "V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.mean(species_dict['V_T.C1'].magnitude)",
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.max(species_dict['V_T.C1'].magnitude)",
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.min(species_dict['V_T.C1'].magnitude)",
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        sim_df = pd.DataFrame(
            {
                "simulation_id": [0],
                "status": [0],
                "time": [[0.0, 1.0, 2.0]],
                "V_T.C1": [[100.0, 200.0, 300.0]],
            }
        )

        species_units = {"V_T.C1": "cell"}

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)

        assert result.shape == (1, 3)
        assert result[0, 0] == pytest.approx(200.0)  # mean
        assert result[0, 1] == pytest.approx(300.0)  # max
        assert result[0, 2] == pytest.approx(100.0)  # min

    def test_multiple_species(self):
        """Test test statistic using multiple species."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["cd8_treg_ratio"],
                "required_species": ["V_T.CD8, V_T.Treg"],
                "model_output_code": [
                    """def compute_test_statistic(time, species_dict, ureg):
    cd8_0 = np.interp(0.0, time.magnitude, species_dict['V_T.CD8'].magnitude)
    treg_0 = np.interp(0.0, time.magnitude, species_dict['V_T.Treg'].magnitude)
    return cd8_0 / max(treg_0, 1e-12)"""
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        sim_df = pd.DataFrame(
            {
                "simulation_id": [0],
                "status": [0],
                "time": [[0.0, 1.0, 2.0]],
                "V_T.CD8": [[100.0, 110.0, 120.0]],
                "V_T.Treg": [[50.0, 55.0, 60.0]],
            }
        )

        species_units = {"V_T.CD8": "cell", "V_T.Treg": "cell"}

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)

        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(2.0, rel=1e-6)

    def test_missing_test_stat_in_registry(self):
        """Test that test stat not in registry is skipped with NaN."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["stat_not_in_registry"],
                "required_species": ["V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.mean(species_dict['V_T.C1'].magnitude)"
                ],
            }
        )

        # Manually create empty registry (simulating a compilation failure)
        registry = {}

        sim_df = pd.DataFrame(
            {"simulation_id": [0], "status": [0], "time": [[0.0, 1.0]], "V_T.C1": [[100.0, 200.0]]}
        )

        species_units = {"V_T.C1": "cell"}

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)

        # Should return NaN for missing test stat in registry
        assert result.shape == (1, 1)
        assert np.isnan(result[0, 0])

    def test_function_raises_exception(self):
        """Test that exceptions in test stat functions result in NaN."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["error_stat"],
                "required_species": ["V_T.C1"],
                "model_output_code": [
                    """def compute_test_statistic(time, species_dict, ureg):
    raise ValueError("Test error")"""
                ],
            }
        )

        registry = build_test_stat_registry(test_stats_df)

        sim_df = pd.DataFrame(
            {"simulation_id": [0], "status": [0], "time": [[0.0, 1.0]], "V_T.C1": [[100.0, 200.0]]}
        )

        species_units = {"V_T.C1": "cell"}

        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)

        # Exception should result in NaN
        assert result.shape == (1, 1)
        assert np.isnan(result[0, 0])


class TestSharedSpeciesDictAcrossTestStats:
    """Regression tests for the 2026-04-20 loop-order refactor (swap sim/
    test-stat nesting, hoist unit parsing). These exercise invariants that
    the old per-(test_stat, sim) species_dict rebuild satisfied trivially
    but which the new shared species_dict must preserve:

      1. A missing species in one test stat's required list must not
         corrupt or silently NaN a sibling test stat that doesn't need it.
      2. Test stats sharing species must receive the same Pint Quantity
         values across both dispatches (no stale-dict aliasing).
      3. Pre-wrapped template_defaults shared across sims must not mutate
         between sims (test stats treat Quantities as immutable in
         practice, but the refactor explicitly reuses the same Python
         object to skip rewrap work).
    """

    def test_missing_species_in_one_stat_does_not_break_others(self):
        """Batch with test stat A (needs missing species) + test stat B
        (needs present species). A→NaN, B→correct value. Old code
        naturally handled this via independent rebuilds; new code has to
        get it right with a shared plan."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["needs_missing", "needs_present"],
                "required_species": ["V_T.missing", "V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return species_dict['V_T.missing'].magnitude",
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return np.mean(species_dict['V_T.C1'].magnitude)",
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0, 1],
                "status": [0, 0],
                "time": [[0.0, 1.0, 2.0]] * 2,
                "V_T.C1": [[100.0, 200.0, 300.0], [10.0, 20.0, 30.0]],
                # No V_T.missing column, no param:V_T.missing, no template.
            }
        )
        species_units = {"V_T.C1": "cell"}
        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)
        assert result.shape == (2, 2)
        # needs_missing → NaN for every sim
        assert np.all(np.isnan(result[:, 0]))
        # needs_present → correct mean for every sim (not contaminated by
        # the sibling's missing-species failure)
        assert result[0, 1] == pytest.approx(200.0)
        assert result[1, 1] == pytest.approx(20.0)

    def test_shared_species_across_stats_sees_same_values(self):
        """Two test stats requiring the same species must see the same
        per-sim array. A plan-level aliasing bug could, e.g., reuse the
        first sim's quantity across all subsequent sims."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["mean_c1", "max_c1"],
                "required_species": ["V_T.C1", "V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return np.mean(species_dict['V_T.C1'].magnitude)",
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return np.max(species_dict['V_T.C1'].magnitude)",
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        # Three sims with deliberately distinct trajectories so an
        # aliasing bug would surface as repeated values.
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0, 1, 2],
                "status": [0, 0, 0],
                "time": [[0.0, 1.0, 2.0]] * 3,
                "V_T.C1": [
                    [1.0, 2.0, 3.0],  # mean=2, max=3
                    [10.0, 20.0, 30.0],  # mean=20, max=30
                    [100.0, 200.0, 300.0],  # mean=200, max=300
                ],
            }
        )
        species_units = {"V_T.C1": "cell"}
        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)
        np.testing.assert_allclose(result[:, 0], [2.0, 20.0, 200.0])
        np.testing.assert_allclose(result[:, 1], [3.0, 30.0, 300.0])

    def test_template_default_shared_across_sims_stays_stable(self):
        """The plan wraps template_defaults once and reuses the Pint
        Quantity across every sim in the batch. Verify the scalar value
        each test stat receives is the original template value (not a
        previous-sim's overwritten reference)."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["return_rho"],
                "required_species": ["rho_collagen"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return species_dict['rho_collagen'].magnitude"
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0, 1, 2, 3],
                "status": [0, 0, 0, 0],
                "time": [[0.0, 1.0]] * 4,
                # Deliberately NOT including rho_collagen or param:rho_collagen
            }
        )
        species_units = {"rho_collagen": "dimensionless"}
        template_defaults = {"rho_collagen": 1.27}
        result = compute_test_statistics_batch(
            sim_df,
            test_stats_df,
            registry,
            species_units,
            template_defaults=template_defaults,
        )
        np.testing.assert_allclose(result[:, 0], [1.27, 1.27, 1.27, 1.27])


class TestNonSpeciesResolution:
    """Test that compartments and scalar parameters resolve correctly.

    Regression tests for the bug where calibration targets referencing
    compartment volumes (e.g., V_T) or scalar parameters (e.g., rho_collagen)
    produced NaN because species_units only contained species, forcing
    `.to('milliliter')` / `.to('gram/milliliter')` to fail on dimensionless defaults.
    """

    def test_compartment_volume_resolves_with_units(self):
        """A compartment column (V_T) in sim_df should pick up its unit from species_units."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["cellularity"],
                "required_species": ["V_T.C1,V_T"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    cancer_cells = np.mean(species_dict['V_T.C1'].magnitude)\n"
                    "    V_T_mL = species_dict['V_T'].to('milliliter').magnitude\n"
                    "    return cancer_cells * 1e-9 / V_T_mL"
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0],
                "status": [0],
                "time": [[0.0, 1.0]],
                "V_T.C1": [[1e8, 1e8]],
                "V_T": [0.5],  # scalar compartment volume
            }
        )
        species_units = {"V_T.C1": "cell", "V_T": "milliliter"}
        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)
        assert not np.isnan(result[0, 0])
        assert result[0, 0] == pytest.approx(1e8 * 1e-9 / 0.5)

    def test_scalar_parameter_resolves_with_units(self):
        """A `param:`-prefixed scalar parameter should pick up its unit from species_units."""
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["rho_identity"],
                "required_species": ["rho_collagen"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return species_dict['rho_collagen'].to('gram/milliliter').magnitude"
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0],
                "status": [0],
                "time": [[0.0]],
                "param:rho_collagen": [0.025],
            }
        )
        species_units = {"rho_collagen": "gram/milliliter"}
        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)
        assert result[0, 0] == pytest.approx(0.025)

    def test_wrong_unit_yields_nan(self):
        """Sanity check: if species_units is missing the right unit (old buggy behavior),
        `.to('milliliter')` on a dimensionless default should fail and produce NaN.
        """
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["cellularity"],
                "required_species": ["V_T"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return species_dict['V_T'].to('milliliter').magnitude"
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        sim_df = pd.DataFrame({"simulation_id": [0], "status": [0], "time": [[0.0]], "V_T": [0.5]})
        # Deliberately omit V_T from species_units → defaults to dimensionless
        species_units = {}
        result = compute_test_statistics_batch(sim_df, test_stats_df, registry, species_units)
        assert np.isnan(result[0, 0])


class TestProcessSingleBatch:
    """Test process_single_batch function."""

    @pytest.fixture
    def test_stats_df(self):
        """Create test statistics definition."""
        return pd.DataFrame(
            {
                "test_statistic_id": ["mean_tumor", "max_tumor"],
                "required_species": ["V_T.C1", "V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.mean(species_dict['V_T.C1'].magnitude)",
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.max(species_dict['V_T.C1'].magnitude)",
                ],
            }
        )

    @pytest.fixture
    def test_stat_registry(self, test_stats_df):
        """Build test statistic registry."""
        return build_test_stat_registry(test_stats_df)

    @pytest.fixture
    def species_units(self):
        """Create species units mapping."""
        return {"V_T.C1": "cell"}

    def test_process_single_batch_returns_sim_count(
        self, test_stats_df, test_stat_registry, species_units, tmp_path
    ):
        """Test that process_single_batch returns number of simulations."""
        # Create parquet file with 3 simulations (using dot notation for species)
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0, 1, 2],
                "status": [0, 0, 0],
                "time": [[0.0, 1.0, 2.0]] * 3,
                "V_T.C1": [[100.0, 200.0, 300.0], [50.0, 100.0, 150.0], [10.0, 20.0, 30.0]],
            }
        )

        parquet_file = tmp_path / "batch_test.parquet"
        sim_df.to_parquet(parquet_file)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        n_sims = process_single_batch(
            batch_idx=0,
            parquet_source=parquet_file,
            test_stats_df=test_stats_df,
            test_stat_registry=test_stat_registry,
            species_units=species_units,
            test_stats_output_dir=output_dir,
        )

        assert n_sims == 3

    def test_process_single_batch_creates_output_files(
        self, test_stats_df, test_stat_registry, species_units, tmp_path
    ):
        """Test that process_single_batch creates correct output files."""
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0, 1],
                "status": [0, 0],
                "time": [[0.0, 1.0, 2.0]] * 2,
                "V_T.C1": [[100.0, 200.0, 300.0], [50.0, 100.0, 150.0]],
            }
        )

        parquet_file = tmp_path / "batch_test.parquet"
        sim_df.to_parquet(parquet_file)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        process_single_batch(
            batch_idx=5,  # Test non-zero batch index
            parquet_source=parquet_file,
            test_stats_df=test_stats_df,
            test_stat_registry=test_stat_registry,
            species_units=species_units,
            test_stats_output_dir=output_dir,
        )

        # Check output file exists with correct name
        test_stats_file = output_dir / "chunk_005_test_stats.csv"
        assert test_stats_file.exists()

        # Load and verify content
        result = np.loadtxt(test_stats_file, delimiter=",")
        assert result.shape == (2, 2)  # 2 sims, 2 test stats
        # First test stat is mean: [200, 100]
        assert result[0, 0] == pytest.approx(200.0)
        assert result[1, 0] == pytest.approx(100.0)
        # Second test stat is max: [300, 150]
        assert result[0, 1] == pytest.approx(300.0)
        assert result[1, 1] == pytest.approx(150.0)


class TestMaxBatchesLimit:
    """Regression tests for max_batches limiting derivation scope.

    Bug fix: Previously, the derivation worker always processed ALL batches in
    the pool directory. When the pool contained pre-existing batches plus newly
    generated ones, this caused a mismatch between params (new only) and
    test_stats (all batches).

    The fix adds max_batches to the config, which limits how many batches
    the worker processes.
    """

    def test_worker_respects_max_batches_limit(self, tmp_path):
        """Test that worker only processes max_batches files when specified.

        Regression test: With 5 batches in pool but max_batches=3, worker
        should only process the first 3 batches.
        """
        # Create test statistics
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["mean_val"],
                "required_species": ["value"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.mean(species_dict['value'].magnitude)"
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        species_units = {"value": "dimensionless"}

        # Create 5 batch files
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()

        for i in range(5):
            sim_df = pd.DataFrame(
                {
                    "simulation_id": [i * 10 + j for j in range(10)],
                    "status": [0] * 10,
                    "time": [[0.0, 1.0]] * 10,
                    "value": [[float(i * 100 + j)] * 2 for j in range(10)],
                }
            )
            sim_df.to_parquet(pool_dir / f"batch_{i:03d}.parquet")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Simulate what the worker does with max_batches=3
        parquet_files = sorted(pool_dir.glob("batch_*.parquet"))
        assert len(parquet_files) == 5  # Sanity check

        max_batches = 3
        parquet_files = parquet_files[:max_batches]  # Limit to first 3

        total_sims = 0
        for batch_idx, parquet_file in enumerate(parquet_files):
            n_sims = process_single_batch(
                batch_idx=batch_idx,
                parquet_source=parquet_file,
                test_stats_df=test_stats_df,
                test_stat_registry=registry,
                species_units=species_units,
                test_stats_output_dir=output_dir,
            )
            total_sims += n_sims

        # Should only process 3 batches x 10 sims = 30 sims (not 50)
        assert total_sims == 30

        # Only 3 output files should exist
        assert (output_dir / "chunk_000_test_stats.csv").exists()
        assert (output_dir / "chunk_001_test_stats.csv").exists()
        assert (output_dir / "chunk_002_test_stats.csv").exists()
        assert not (output_dir / "chunk_003_test_stats.csv").exists()
        assert not (output_dir / "chunk_004_test_stats.csv").exists()

    def test_worker_processes_all_when_max_batches_none(self, tmp_path):
        """Test that worker processes all batches when max_batches is None."""
        # Create test statistics
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["mean_val"],
                "required_species": ["value"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.mean(species_dict['value'].magnitude)"
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        species_units = {"value": "dimensionless"}

        # Create 4 batch files
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()

        for i in range(4):
            sim_df = pd.DataFrame(
                {
                    "simulation_id": [i * 5 + j for j in range(5)],
                    "status": [0] * 5,
                    "time": [[0.0, 1.0]] * 5,
                    "value": [[float(i * 100 + j)] * 2 for j in range(5)],
                }
            )
            sim_df.to_parquet(pool_dir / f"batch_{i:03d}.parquet")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Simulate what the worker does with max_batches=None (process all)
        parquet_files = sorted(pool_dir.glob("batch_*.parquet"))
        max_batches = None

        if max_batches is not None:
            parquet_files = parquet_files[:max_batches]

        total_sims = 0
        for batch_idx, parquet_file in enumerate(parquet_files):
            n_sims = process_single_batch(
                batch_idx=batch_idx,
                parquet_source=parquet_file,
                test_stats_df=test_stats_df,
                test_stat_registry=registry,
                species_units=species_units,
                test_stats_output_dir=output_dir,
            )
            total_sims += n_sims

        # Should process all 4 batches x 5 sims = 20 sims
        assert total_sims == 20

        # All 4 output files should exist
        for i in range(4):
            assert (output_dir / f"chunk_{i:03d}_test_stats.csv").exists()


class TestSingleTaskDerivation:
    """Regression tests for single-task derivation (not array job).

    These tests verify that derivation processes ALL batches in a single
    task, rather than using SLURM array jobs with one task per batch.
    """

    def test_worker_processes_all_batches_not_array_task(self, tmp_path):
        """Verify worker processes all batches, not just SLURM_ARRAY_TASK_ID.

        Regression test: Previously, the worker used SLURM_ARRAY_TASK_ID to
        select a single batch to process. Now it should process ALL batches.
        """
        # Create test statistics
        test_stats_df = pd.DataFrame(
            {
                "test_statistic_id": ["mean_val"],
                "required_species": ["value"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n    return np.mean(species_dict['value'].magnitude)"
                ],
            }
        )
        registry = build_test_stat_registry(test_stats_df)
        species_units = {"value": "dimensionless"}

        # Create multiple batch files
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()

        for i in range(3):
            sim_df = pd.DataFrame(
                {
                    "simulation_id": [i * 10 + j for j in range(5)],
                    "status": [0] * 5,
                    "time": [[0.0, 1.0]] * 5,
                    "value": [[float(i * 100 + j)] * 2 for j in range(5)],
                }
            )
            sim_df.to_parquet(pool_dir / f"batch_{i:03d}.parquet")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Process all batches (simulating what the worker does)
        parquet_files = sorted(pool_dir.glob("batch_*.parquet"))
        assert len(parquet_files) == 3  # Sanity check

        total_sims = 0
        for batch_idx, parquet_file in enumerate(parquet_files):
            n_sims = process_single_batch(
                batch_idx=batch_idx,
                parquet_source=parquet_file,
                test_stats_df=test_stats_df,
                test_stat_registry=registry,
                species_units=species_units,
                test_stats_output_dir=output_dir,
            )
            total_sims += n_sims

        # Verify ALL batches were processed
        assert total_sims == 15  # 3 batches x 5 sims each

        # Verify output files for all batches
        assert (output_dir / "chunk_000_test_stats.csv").exists()
        assert (output_dir / "chunk_001_test_stats.csv").exists()
        assert (output_dir / "chunk_002_test_stats.csv").exists()


class TestProcessSingleBatchRowGroupStreaming:
    """Regression for the 2026-04-18 OOM on the 15k clinical_progression
    batch: derive used to do ``pd.read_parquet`` on the whole file, which
    spiked to ~100 GB on wide list-typed parquets even with --mem=32G.
    Now we stream row-groups. The on-disk output for a multi-row-group
    parquet must exactly match what the single-read path produced, so
    downstream ``combine_test_stats_chunks.py`` stays oblivious."""

    @pytest.fixture
    def test_stats_df(self):
        return pd.DataFrame(
            {
                "test_statistic_id": ["mean_tumor", "max_tumor"],
                "required_species": ["V_T.C1", "V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return np.mean(species_dict['V_T.C1'].magnitude)",
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return np.max(species_dict['V_T.C1'].magnitude)",
                ],
            }
        )

    @pytest.fixture
    def test_stat_registry(self, test_stats_df):
        return build_test_stat_registry(test_stats_df)

    @pytest.fixture
    def species_units(self):
        return {"V_T.C1": "cell"}

    def _write_multi_row_group_parquet(self, path, n_sims, rows_per_group):
        """Write a parquet with multiple row groups so we can verify
        the streaming path actually iterates over them."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        sim_df = pd.DataFrame(
            {
                "simulation_id": list(range(n_sims)),
                "sample_index": list(range(n_sims)),
                "status": [0] * n_sims,
                "time": [[0.0, 1.0, 2.0]] * n_sims,
                "V_T.C1": [
                    [float(i) * 10.0, float(i) * 20.0, float(i) * 30.0] for i in range(n_sims)
                ],
                "param:k_growth": [0.1 + 0.01 * i for i in range(n_sims)],
            }
        )
        table = pa.Table.from_pandas(sim_df)
        # chunksize controls row group size — critical for the test.
        pq.write_table(table, str(path), row_group_size=rows_per_group)
        return sim_df

    def test_multi_row_group_parquet_matches_single_read(
        self, test_stats_df, test_stat_registry, species_units, tmp_path
    ):
        """12 sims in 3 row groups of 4 must produce the same on-disk
        CSV as a hypothetical single-read, and the header must appear
        exactly once in the params CSV."""
        parquet_file = tmp_path / "batch.parquet"
        self._write_multi_row_group_parquet(parquet_file, n_sims=12, rows_per_group=4)

        # Sanity-check the fixture produced >1 row group — if it didn't,
        # this test wouldn't exercise the streaming path.
        import pyarrow.parquet as pq

        assert pq.ParquetFile(str(parquet_file)).num_row_groups == 3

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        n_sims = process_single_batch(
            batch_idx=0,
            parquet_source=parquet_file,
            test_stats_df=test_stats_df,
            test_stat_registry=test_stat_registry,
            species_units=species_units,
            test_stats_output_dir=output_dir,
        )

        assert n_sims == 12

        # test_stats CSV: 12 data rows, no header, 2 columns.
        ts_lines = (output_dir / "chunk_000_test_stats.csv").read_text().splitlines()
        assert len(ts_lines) == 12
        # First sim: time series [0, 0, 0] → mean=0, max=0.
        first_row = [float(x) for x in ts_lines[0].split(",")]
        assert first_row == [0.0, 0.0]
        # Last sim (i=11): [110, 220, 330] → mean=220, max=330.
        last_row = [float(x) for x in ts_lines[-1].split(",")]
        assert last_row == pytest.approx([220.0, 330.0])

        # Params CSV: exactly ONE header row, 12 data rows, sample_index
        # preserved across row-group boundaries.
        params_df = pd.read_csv(output_dir / "chunk_000_params.csv")
        assert len(params_df) == 12
        assert list(params_df["sample_index"]) == list(range(12))
        # Header must appear exactly once — regression against a bug
        # where each row group wrote its own header.
        params_text = (output_dir / "chunk_000_params.csv").read_text()
        assert params_text.count("sample_index,k_growth") == 1

    def test_single_row_group_parquet_still_works(
        self, test_stats_df, test_stat_registry, species_units, tmp_path
    ):
        """The streaming path must handle a 1-row-group parquet (what
        pd.to_parquet produces by default in the rest of the test suite)
        exactly like the old single-read path."""
        parquet_file = tmp_path / "batch.parquet"
        sim_df = pd.DataFrame(
            {
                "simulation_id": [0, 1, 2],
                "status": [0, 0, 0],
                "time": [[0.0, 1.0, 2.0]] * 3,
                "V_T.C1": [[100.0, 200.0, 300.0], [50.0, 100.0, 150.0], [10.0, 20.0, 30.0]],
            }
        )
        sim_df.to_parquet(parquet_file)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        n_sims = process_single_batch(
            batch_idx=0,
            parquet_source=parquet_file,
            test_stats_df=test_stats_df,
            test_stat_registry=test_stat_registry,
            species_units=species_units,
            test_stats_output_dir=output_dir,
        )
        assert n_sims == 3
        ts_lines = (output_dir / "chunk_000_test_stats.csv").read_text().splitlines()
        assert len(ts_lines) == 3


class TestProcessSingleBatchSubdirLayout:
    """#43 option A: array tasks now drop chunks directly into a
    per-submission ``batch_*/`` subdir, no combine step. process_single_batch
    must treat the subdir as one batch — walking ``chunk_*.parquet`` within
    — and produce the same CSVs as a single consolidated parquet would."""

    @pytest.fixture
    def test_stats_df(self):
        return pd.DataFrame(
            {
                "test_statistic_id": ["mean_tumor"],
                "required_species": ["V_T.C1"],
                "model_output_code": [
                    "def compute_test_statistic(time, species_dict, ureg):\n"
                    "    return np.mean(species_dict['V_T.C1'].magnitude)"
                ],
            }
        )

    @pytest.fixture
    def test_stat_registry(self, test_stats_df):
        return build_test_stat_registry(test_stats_df)

    @pytest.fixture
    def species_units(self):
        return {"V_T.C1": "cell"}

    def test_batch_subdir_aggregates_chunks(
        self, test_stats_df, test_stat_registry, species_units, tmp_path
    ):
        """A batch_*/ subdir with 3 chunk_*.parquet files (4 sims each) must
        derive as one 12-row output — chunks iterated in filename order."""
        batch_dir = tmp_path / "batch_20260418_000000_ctrl_seed42"
        batch_dir.mkdir()

        for task_idx in range(3):
            sim_df = pd.DataFrame(
                {
                    "simulation_id": list(range(4)),  # local per-chunk ids
                    "sample_index": list(range(task_idx * 4, (task_idx + 1) * 4)),
                    "status": [0] * 4,
                    "time": [[0.0, 1.0]] * 4,
                    "V_T.C1": [
                        [float(task_idx * 100 + i) * 10.0, float(task_idx * 100 + i) * 20.0]
                        for i in range(4)
                    ],
                    "param:k_growth": [0.1 + 0.01 * (task_idx * 4 + i) for i in range(4)],
                }
            )
            sim_df.to_parquet(batch_dir / f"chunk_{task_idx:03d}.parquet")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        n_sims = process_single_batch(
            batch_idx=0,
            parquet_source=batch_dir,
            test_stats_df=test_stats_df,
            test_stat_registry=test_stat_registry,
            species_units=species_units,
            test_stats_output_dir=output_dir,
        )

        assert n_sims == 12

        # 12 rows of test stats across 3 chunk files, concatenated in filename order.
        ts_lines = (output_dir / "chunk_000_test_stats.csv").read_text().splitlines()
        assert len(ts_lines) == 12

        # sample_index preserved, exactly one header, covers chunk boundaries.
        params_df = pd.read_csv(output_dir / "chunk_000_params.csv")
        assert len(params_df) == 12
        assert list(params_df["sample_index"]) == list(range(12))
        params_text = (output_dir / "chunk_000_params.csv").read_text()
        assert params_text.count("sample_index,k_growth") == 1

    def test_empty_batch_subdir_returns_zero(
        self, test_stats_df, test_stat_registry, species_units, tmp_path
    ):
        """A batch_*/ subdir with no chunks yet (task failed before flush)
        must be skipped, not crash. Derive's top-level loop will still
        record total_sims correctly."""
        batch_dir = tmp_path / "batch_20260418_000000_ctrl_seed42"
        batch_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        n_sims = process_single_batch(
            batch_idx=0,
            parquet_source=batch_dir,
            test_stats_df=test_stats_df,
            test_stat_registry=test_stat_registry,
            species_units=species_units,
            test_stats_output_dir=output_dir,
        )
        assert n_sims == 0
        assert not (output_dir / "chunk_000_test_stats.csv").exists()
