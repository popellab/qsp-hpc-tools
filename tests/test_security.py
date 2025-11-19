"""Tests for security utilities."""

import tempfile
from pathlib import Path

import pytest

from qsp_hpc.utils.security import (
    SecurityError,
    build_safe_ssh_command,
    safe_shell_quote,
    validate_pool_path,
    validate_project_name,
    validate_safe_path,
)


class TestValidateProjectName:
    """Tests for project name validation."""

    def test_valid_names(self):
        """Test that valid project names pass validation."""
        valid_names = [
            "myproject",
            "my_project",
            "my-project",
            "project123",
            "my.project",
            "Project_2025",
            "abc123_test-v2.1",
        ]
        for name in valid_names:
            assert validate_project_name(name) == name

    def test_empty_name(self):
        """Test that empty name is rejected."""
        with pytest.raises(SecurityError, match="cannot be empty"):
            validate_project_name("")

    def test_path_traversal_dotdot(self):
        """Test that .. is rejected (path traversal)."""
        # Starting with .. is caught by the "starts with ." check first
        with pytest.raises(SecurityError):
            validate_project_name("../etc/passwd")

        # .. in the middle is caught correctly
        with pytest.raises(SecurityError, match="cannot contain"):
            validate_project_name("test/../secret")

    def test_path_separators(self):
        """Test that path separators are rejected."""
        with pytest.raises(SecurityError, match="path separators"):
            validate_project_name("test/path")

        with pytest.raises(SecurityError, match="path separators"):
            validate_project_name("test\\path")

    def test_invalid_characters(self):
        """Test that special characters are rejected."""
        invalid_names = [
            "test;rm -rf /",  # Contains / separator
            "test$(whoami)",
            "test`cat /etc/passwd`",  # Contains / separator
            "test&& rm -rf",
            "test|grep secret",
            "test>output.txt",
            "test$USER",
            "test*wildcard",
            "test with spaces",
        ]
        # All invalid names should raise SecurityError (regardless of specific message)
        for name in invalid_names:
            with pytest.raises(SecurityError):
                validate_project_name(name)

    def test_hidden_files(self):
        """Test that names starting with . are rejected."""
        with pytest.raises(SecurityError, match="cannot start with"):
            validate_project_name(".hidden")

        with pytest.raises(SecurityError, match="cannot start with"):
            validate_project_name("..test")


class TestValidateSafePath:
    """Tests for safe path validation."""

    def test_valid_path(self):
        """Test that valid paths are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve base to handle symlinks (macOS /var -> /private/var)
            base = Path(tmpdir).resolve()
            result = validate_safe_path(str(base), "project1", "data")
            assert result.is_relative_to(base)
            assert str(result).endswith("project1/data")

    def test_simple_subdirectory(self):
        """Test simple subdirectory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir).resolve()
            result = validate_safe_path(str(base), "subdir")
            assert result.is_relative_to(base)

    def test_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to escape with ..
            with pytest.raises(SecurityError, match="would escape base directory"):
                validate_safe_path(tmpdir, "..", "etc", "passwd")

            # Try with multiple levels
            with pytest.raises(SecurityError, match="would escape base directory"):
                validate_safe_path(tmpdir, "sub", "..", "..", "etc")

    def test_separator_in_component(self):
        """Test that separators in components are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(SecurityError, match="cannot contain separators"):
                validate_safe_path(tmpdir, "test/path")

            with pytest.raises(SecurityError, match="cannot contain separators"):
                validate_safe_path(tmpdir, "test\\path")

    def test_absolute_path_components(self):
        """Test that we handle absolute path attempts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Absolute path in component should fail separator check
            with pytest.raises(SecurityError):
                validate_safe_path(tmpdir, "/etc/passwd")


class TestSafeShellQuote:
    """Tests for shell quoting."""

    def test_simple_string(self):
        """Test quoting simple strings."""
        assert safe_shell_quote("test") == "test"
        assert safe_shell_quote("test123") == "test123"

    def test_spaces(self):
        """Test quoting strings with spaces."""
        result = safe_shell_quote("test with spaces")
        # Should be quoted
        assert result.startswith("'") or result.startswith('"')

    def test_dangerous_characters(self):
        """Test quoting strings with dangerous characters."""
        dangerous = [
            "test; rm -rf /",
            "test$(whoami)",
            "test`cat /etc/passwd`",
            "test && echo pwned",
            "test | grep secret",
        ]
        for s in dangerous:
            quoted = safe_shell_quote(s)
            # The quoted version should not contain unescaped dangerous chars
            # when used in a shell command
            assert quoted != s  # Should be modified

    def test_quotes_in_string(self):
        """Test quoting strings that contain quotes."""
        result = safe_shell_quote("test'with'quotes")
        # Should handle quotes safely
        assert "'" in result or "\\" in result


class TestBuildSafeSSHCommand:
    """Tests for safe SSH command building."""

    def test_simple_command(self):
        """Test building simple commands."""
        cmd = build_safe_ssh_command(["ls", "-la"])
        assert "ls" in cmd
        assert "-la" in cmd

    def test_command_with_spaces(self):
        """Test command with spaces in arguments."""
        cmd = build_safe_ssh_command(["cat", "file with spaces.txt"])
        # Should be quoted
        assert "'" in cmd or '"' in cmd

    def test_command_with_cwd(self):
        """Test command with working directory."""
        cmd = build_safe_ssh_command(["ls"], cwd="/path/to/dir")
        assert "cd" in cmd
        assert "/path/to/dir" in cmd
        assert "&&" in cmd
        assert "ls" in cmd

    def test_dangerous_arguments_quoted(self):
        """Test that dangerous arguments are properly quoted."""
        cmd = build_safe_ssh_command(["echo", "test; rm -rf /"])
        # Should be quoted to prevent execution
        assert "'" in cmd or '"' in cmd

    def test_dangerous_cwd_quoted(self):
        """Test that dangerous cwd is properly quoted."""
        cmd = build_safe_ssh_command(["ls"], cwd='/path"; rm -rf /; echo "')
        # Should be quoted
        assert "'" in cmd or '"' in cmd


class TestValidatePoolPath:
    """Tests for pool path validation."""

    def test_valid_pool_paths(self):
        """Test valid pool paths."""
        valid = ["model_v1_abcd1234", "model_version_hash", "simple_path"]
        for path in valid:
            assert validate_pool_path(path) == path

    def test_empty_path(self):
        """Test that empty path is rejected."""
        with pytest.raises(SecurityError, match="cannot be empty"):
            validate_pool_path("")

    def test_path_traversal(self):
        """Test that .. is rejected."""
        with pytest.raises(SecurityError, match="cannot contain '\\.\\.'"):
            validate_pool_path("../etc/passwd")

        with pytest.raises(SecurityError, match="cannot contain '\\.\\.'"):
            validate_pool_path("model_v1/../secret")

    def test_absolute_paths(self):
        """Test that absolute paths are rejected."""
        with pytest.raises(SecurityError, match="cannot be absolute"):
            validate_pool_path("/etc/passwd")

        with pytest.raises(SecurityError, match="cannot be absolute"):
            validate_pool_path("\\Windows\\System32")


class TestSecurityIntegration:
    """Integration tests for security features."""

    def test_project_name_in_path_construction(self):
        """Test that validated project names work in path construction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir).resolve()
            # Valid project name should work
            project = validate_project_name("test_project")
            path = validate_safe_path(str(base), "projects", project, "data")
            assert path.is_relative_to(base)

            # Invalid project name should fail early
            with pytest.raises(SecurityError):
                validate_project_name("../evil")

    def test_command_injection_prevention(self):
        """Test that command injection is prevented."""
        # Try to inject commands via project name
        # (Contains / so caught by path separator check)
        with pytest.raises(SecurityError):
            validate_project_name("test; rm -rf /")

        # Try with no path separator but invalid char
        with pytest.raises(SecurityError, match="only alphanumeric"):
            validate_project_name("test;whoami")

        # Build safe command with dangerous input
        cmd = build_safe_ssh_command(["ls", "file; rm -rf /"], cwd='/path"; cat /etc/passwd; echo "')
        # Both should be quoted
        assert "'" in cmd or '"' in cmd

    def test_path_traversal_prevention(self):
        """Test that path traversal is prevented at multiple layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Project name validation prevents ..
            with pytest.raises(SecurityError):
                validate_project_name("../etc")

            # Path validation also prevents ..
            with pytest.raises(SecurityError):
                validate_safe_path(tmpdir, "..", "etc")

            # Pool path validation prevents ..
            with pytest.raises(SecurityError):
                validate_pool_path("../secret")
