"""Tests for combine_test_stats_chunks module."""

from qsp_hpc.batch.combine_test_stats_chunks import combine_params, combine_test_stats


class TestCombineTestStats:
    """Tests for combining test statistics chunks (no headers)."""

    def test_combine_basic(self, temp_dir):
        """Test basic combining of test stats files without headers."""
        # Create chunk files with numeric data (no headers)
        chunk_files = []
        for i in range(3):
            chunk_file = temp_dir / f"chunk_{i:03d}_test_stats.csv"
            # Write 2 rows of data per chunk
            chunk_file.write_text(f"{i}.1,{i}.2,{i}.3\n{i}.4,{i}.5,{i}.6\n")
            chunk_files.append(chunk_file)

        # Combine chunks
        n_combined = combine_test_stats(temp_dir)

        # Verify
        assert n_combined == 3
        output_file = temp_dir / "combined_test_stats.csv"
        assert output_file.exists()

        # Check content - should be all 6 rows concatenated
        content = output_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 6  # 2 rows per chunk * 3 chunks
        assert lines[0] == "0.1,0.2,0.3"
        assert lines[1] == "0.4,0.5,0.6"
        assert lines[2] == "1.1,1.2,1.3"

    def test_combine_empty_directory(self, temp_dir):
        """Test combining when no chunk files exist."""
        n_combined = combine_test_stats(temp_dir)
        assert n_combined == 0

        # Output file should not be created
        output_file = temp_dir / "combined_test_stats.csv"
        assert not output_file.exists()

    def test_combine_single_chunk(self, temp_dir):
        """Test combining with only one chunk file."""
        chunk_file = temp_dir / "chunk_000_test_stats.csv"
        chunk_file.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n")

        n_combined = combine_test_stats(temp_dir)

        assert n_combined == 1
        output_file = temp_dir / "combined_test_stats.csv"
        assert output_file.exists()
        assert output_file.read_text() == "1.0,2.0,3.0\n4.0,5.0,6.0\n"


class TestCombineParams:
    """Tests for combining parameter chunks (with CSV headers)."""

    def test_combine_with_headers(self, temp_dir):
        """Test combining params files while preserving only first header."""
        # Create chunk files with CSV headers
        for i in range(3):
            chunk_file = temp_dir / f"chunk_{i:03d}_params.csv"
            # Each file has header + 2 data rows
            content = f"param1,param2,param3\n{i}.1,{i}.2,{i}.3\n{i}.4,{i}.5,{i}.6\n"
            chunk_file.write_text(content)

        # Combine chunks
        n_combined = combine_params(temp_dir)

        # Verify
        assert n_combined == 3
        output_file = temp_dir / "combined_params.csv"
        assert output_file.exists()

        # Check content - should have 1 header + 6 data rows
        content = output_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 7  # 1 header + 2 rows per chunk * 3 chunks

        # First line should be header
        assert lines[0] == "param1,param2,param3"

        # Data rows should not have header repeated
        assert lines[1] == "0.1,0.2,0.3"
        assert lines[2] == "0.4,0.5,0.6"
        assert lines[3] == "1.1,1.2,1.3"
        assert lines[4] == "1.4,1.5,1.6"

        # Verify no duplicate headers in data
        for line in lines[1:]:
            assert "param1" not in line

    def test_combine_empty_directory(self, temp_dir):
        """Test combining when no param chunk files exist."""
        n_combined = combine_params(temp_dir)
        assert n_combined == 0

        # Output file should not be created
        output_file = temp_dir / "combined_params.csv"
        assert not output_file.exists()

    def test_combine_single_chunk(self, temp_dir):
        """Test combining with only one params chunk file."""
        chunk_file = temp_dir / "chunk_000_params.csv"
        content = "param1,param2\n1.0,2.0\n3.0,4.0\n"
        chunk_file.write_text(content)

        n_combined = combine_params(temp_dir)

        assert n_combined == 1
        output_file = temp_dir / "combined_params.csv"
        assert output_file.exists()
        assert output_file.read_text() == content

    def test_combine_header_only_chunks(self, temp_dir):
        """Test combining chunks that have only headers (edge case)."""
        for i in range(2):
            chunk_file = temp_dir / f"chunk_{i:03d}_params.csv"
            chunk_file.write_text("param1,param2\n")

        n_combined = combine_params(temp_dir)

        assert n_combined == 2
        output_file = temp_dir / "combined_params.csv"
        assert output_file.exists()

        # Should have just the header from first file
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1
        assert lines[0] == "param1,param2"
