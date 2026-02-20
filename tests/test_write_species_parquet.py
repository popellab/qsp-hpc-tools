"""Tests for write_species_parquet.py - MATLAB→Parquet conversion."""

import json

import pandas as pd

from qsp_hpc.simulation.write_species_parquet import write_species_parquet


class TestWriteSpeciesParquet:
    """Tests for the write_species_parquet function."""

    def test_basic_conversion(self, temp_dir):
        """Test basic JSON to Parquet conversion."""
        json_data = {
            "n_sims": 2,
            "n_species": 2,
            "species_names": ["Cancer", "Immune"],
            "time_arrays": [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            "species_arrays": [
                [[100.0, 90.0, 80.0], [10.0, 15.0, 20.0]],
                [[100.0, 85.0, 70.0], [10.0, 18.0, 25.0]],
            ],
            "status": [1, 1],
        }

        json_file = temp_dir / "test.json"
        output_file = temp_dir / "output.parquet"

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        write_species_parquet(str(json_file), str(output_file))

        # Verify output
        df = pd.read_parquet(output_file)
        assert len(df) == 2
        assert list(df["simulation_id"]) == [0, 1]
        assert list(df.loc[0, "Cancer"]) == [100.0, 90.0, 80.0]
        assert list(df.loc[0, "Immune"]) == [10.0, 15.0, 20.0]

    def test_single_simulation_matlab_edge_case(self, temp_dir):
        """Test MATLAB edge case where n_sims=1 creates scalars instead of lists."""
        json_data = {
            "n_sims": 1,
            "n_species": 2,
            "species_names": ["Cancer", "Immune"],
            "time_arrays": [[0.0, 1.0, 2.0]],  # Properly wrapped
            "species_arrays": [[[100.0, 90.0, 80.0], [10.0, 15.0, 20.0]]],  # Properly wrapped
            "status": 1,  # Scalar - this is the edge case
        }

        json_file = temp_dir / "single.json"
        output_file = temp_dir / "single.parquet"

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        write_species_parquet(str(json_file), str(output_file))

        df = pd.read_parquet(output_file)
        assert len(df) == 1
        assert df.loc[0, "status"] == 1
        assert list(df.loc[0, "time"]) == [0.0, 1.0, 2.0]

    def test_with_parameters(self, temp_dir):
        """Test parameter columns are added correctly."""
        json_data = {
            "n_sims": 2,
            "n_species": 1,
            "species_names": ["Cancer"],
            "param_names": ["k_growth", "k_death"],
            "param_values": [[0.1, 0.05], [0.2, 0.08]],
            "time_arrays": [[0.0, 1.0], [0.0, 1.0]],
            "species_arrays": [[[100.0, 110.0]], [[100.0, 120.0]]],
            "status": [1, 1],
        }

        json_file = temp_dir / "params.json"
        output_file = temp_dir / "params.parquet"

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        write_species_parquet(str(json_file), str(output_file))

        df = pd.read_parquet(output_file)
        assert "param:k_growth" in df.columns
        assert "param:k_death" in df.columns
        assert df.loc[0, "param:k_growth"] == 0.1
        assert df.loc[1, "param:k_death"] == 0.08

    def test_parameter_reshaping_1d_to_2d(self, temp_dir):
        """Test parameter value reshaping for single simulation."""
        json_data = {
            "n_sims": 1,
            "n_species": 1,
            "species_names": ["Cancer"],
            "param_names": ["k_growth", "k_death"],
            "param_values": [0.1, 0.05],  # 1D array
            "time_arrays": [[0.0, 1.0]],
            "species_arrays": [[[100.0, 110.0]]],
            "status": [1],
        }

        json_file = temp_dir / "reshape.json"
        output_file = temp_dir / "reshape.parquet"

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        write_species_parquet(str(json_file), str(output_file))

        df = pd.read_parquet(output_file)
        assert df.loc[0, "param:k_growth"] == 0.1
        assert df.loc[0, "param:k_death"] == 0.05

    def test_species_names_with_dots(self, temp_dir):
        """Test that dots in species names are preserved (SimBiology convention)."""
        json_data = {
            "n_sims": 1,
            "n_species": 2,
            "species_names": ["Cancer.Tumor", "Immune.TCell"],
            "time_arrays": [[0.0, 1.0]],
            "species_arrays": [[[100.0, 110.0], [10.0, 15.0]]],
            "status": [1],
        }

        json_file = temp_dir / "dots.json"
        output_file = temp_dir / "dots.parquet"

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        write_species_parquet(str(json_file), str(output_file))

        df = pd.read_parquet(output_file)
        # Species names now preserve dots (SimBiology convention)
        assert "Cancer.Tumor" in df.columns
        assert "Immune.TCell" in df.columns

    def test_empty_arrays_for_failed_sims(self, temp_dir):
        """Test handling of empty arrays for failed simulations."""
        json_data = {
            "n_sims": 2,
            "n_species": 1,
            "species_names": ["Cancer"],
            "time_arrays": [[], [0.0, 1.0]],
            "species_arrays": [[[]], [[100.0, 110.0]]],
            "status": [0, 1],  # First sim failed
        }

        json_file = temp_dir / "empty.json"
        output_file = temp_dir / "empty.parquet"

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        write_species_parquet(str(json_file), str(output_file))

        df = pd.read_parquet(output_file)
        assert len(df.loc[0, "time"]) == 0
        assert len(df.loc[0, "Cancer"]) == 0
        assert df.loc[0, "status"] == 0
