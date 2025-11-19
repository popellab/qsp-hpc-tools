"""Security utilities for path validation and command sanitization."""

import re
import shlex
from pathlib import Path
from typing import List, Optional


class SecurityError(Exception):
    """Raised when a security validation fails."""

    pass


def validate_project_name(name: str) -> str:
    """
    Validate project name to prevent path traversal and command injection.

    Args:
        name: The project name to validate

    Returns:
        The validated project name (unchanged if valid)

    Raises:
        SecurityError: If the project name contains invalid characters

    Examples:
        >>> validate_project_name("my_project")
        'my_project'
        >>> validate_project_name("../etc/passwd")  # doctest: +SKIP
        Traceback (most recent call last):
        SecurityError: Invalid project name...
    """
    if not name:
        raise SecurityError("Project name cannot be empty")

    # Prevent hidden files (check before .. to get correct error message)
    if name.startswith("."):
        raise SecurityError(f"Invalid project name '{name}': cannot start with '.'")

    # Check for path traversal attempts
    if ".." in name:
        raise SecurityError(f"Invalid project name '{name}': cannot contain '..'")

    # Check for path separators
    if "/" in name or "\\" in name:
        raise SecurityError(f"Invalid project name '{name}': cannot contain path separators (/ \\)")

    # Only allow safe characters: alphanumeric, underscore, hyphen, period
    if not re.match(r"^[a-zA-Z0-9_.-]+$", name):
        raise SecurityError(
            f"Invalid project name '{name}': "
            f"only alphanumeric characters, underscore, hyphen, and period allowed"
        )

    return name


def validate_safe_path(base_dir: str, *path_components: str) -> Path:
    """
    Construct a path and validate it doesn't escape the base directory.

    Args:
        base_dir: The base directory that the path must stay within
        *path_components: Path components to join (each will be validated)

    Returns:
        Resolved Path object guaranteed to be within base_dir

    Raises:
        SecurityError: If the resolved path would escape base_dir

    Examples:
        >>> base = "/home/user/projects"
        >>> validate_safe_path(base, "myproject", "data")  # doctest: +SKIP
        Path('/home/user/projects/myproject/data')

        >>> validate_safe_path(base, "..", "etc", "passwd")  # doctest: +SKIP
        Traceback (most recent call last):
        SecurityError: Path would escape base directory...
    """
    # Resolve base first, handling symlinks
    base = Path(base_dir).resolve()

    # Build target path
    target = base
    for component in path_components:
        # Validate each component doesn't contain separators
        if "/" in component or "\\" in component:
            raise SecurityError(f"Path component '{component}' cannot contain separators")
        target = target / component

    # Resolve and check it's within base
    resolved = target.resolve()
    try:
        # This works even with symlinks because both are resolved
        resolved.relative_to(base)
    except ValueError:
        raise SecurityError(
            f"Path would escape base directory: "
            f"base={base}, target={target}, resolved={resolved}"
        )

    return resolved


def safe_shell_quote(s: str) -> str:
    """
    Safely quote a string for use in shell commands.

    Uses shlex.quote() to ensure the string is properly escaped
    for shell interpolation.

    Args:
        s: String to quote

    Returns:
        Shell-safe quoted string

    Examples:
        >>> safe_shell_quote("normal_path")
        'normal_path'
        >>> safe_shell_quote("path with spaces")
        "'path with spaces'"
        >>> safe_shell_quote("dangerous'; rm -rf /")
        "'dangerous'\"'\"'; rm -rf /'"
    """
    return shlex.quote(s)


def build_safe_ssh_command(command_parts: List[str], cwd: Optional[str] = None) -> str:
    """
    Build a safe shell command from parts, with optional working directory.

    This prevents command injection by properly quoting all arguments.

    Args:
        command_parts: List of command parts to join
        cwd: Optional working directory (will be quoted)

    Returns:
        Safe shell command string

    Examples:
        >>> build_safe_ssh_command(['ls', '-la'])
        'ls -la'
        >>> build_safe_ssh_command(['cat', 'file with spaces.txt'])
        "cat 'file with spaces.txt'"
        >>> build_safe_ssh_command(['ls'], cwd='/path/to/dir')
        "cd '/path/to/dir' && ls"
    """
    # Quote each part
    safe_parts = [shlex.quote(part) for part in command_parts]

    command = " ".join(safe_parts)

    if cwd:
        safe_cwd = shlex.quote(cwd)
        command = f"cd {safe_cwd} && {command}"

    return command


def validate_pool_path(pool_path: str) -> str:
    """
    Validate simulation pool path format.

    Pool paths should be in format: model_version_hash/

    Args:
        pool_path: The pool path to validate

    Returns:
        The validated pool path

    Raises:
        SecurityError: If the pool path format is invalid
    """
    if not pool_path:
        raise SecurityError("Pool path cannot be empty")

    # Check for path traversal
    if ".." in pool_path:
        raise SecurityError(f"Invalid pool path '{pool_path}': cannot contain '..'")

    # Pool paths should be relatively simple (no absolute paths)
    if pool_path.startswith("/") or pool_path.startswith("\\"):
        raise SecurityError(f"Invalid pool path '{pool_path}': cannot be absolute path")

    return pool_path
