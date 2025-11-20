#!/usr/bin/env python3
"""
Centralized logging configuration for QSP HPC Tools.

Provides consistent logging across all modules with configurable verbosity.
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


def setup_logger(name: str, level: Optional[int] = None, verbose: bool = False) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO if not verbose, DEBUG if verbose)
        verbose: If True, set to DEBUG level and add more detailed formatting

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__, verbose=True)
        >>> logger.info("Starting simulation...")
        >>> logger.debug("Parameter count: 10")
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if logger.handlers:
        return logger

    # Set level
    if level is None:
        level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter - always use full format with timestamp and logger name
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with standard configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing batch...")
    """
    return logging.getLogger(name)


def set_verbosity(verbose: bool = False):
    """
    Set global verbosity level for all qsp_hpc loggers.

    Args:
        verbose: If True, set all loggers to DEBUG level

    Example:
        >>> set_verbosity(verbose=True)  # Enable debug logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Update all existing qsp_hpc loggers
    for name in logging.Logger.manager.loggerDict:
        if name.startswith("qsp_hpc"):
            logger = logging.getLogger(name)
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)


# ============================================================================
# Enhanced Logging Utilities
# ============================================================================


def separator(width: int = 80) -> str:
    """
    Create a visual separator line for log sections.

    Args:
        width: Width of the separator line

    Returns:
        Separator string

    Example:
        >>> logger.info(separator())
    """
    return "=" * width


def format_config(config: Dict[str, Any], indent: str = "  ") -> List[str]:
    """
    Format a configuration dictionary as indented key-value pairs.

    Args:
        config: Configuration dictionary
        indent: Indentation string for each line

    Returns:
        List of formatted strings, one per key-value pair

    Example:
        >>> config = {"test_stats": "path/to/file.csv", "scenario": "baseline"}
        >>> for line in format_config(config):
        ...     logger.info(line)
    """
    lines = []
    for key, value in config.items():
        # Format key in a readable way (replace underscores with spaces, title case)
        display_key = key.replace("_", " ").title()
        lines.append(f"{indent}{display_key}: {value}")
    return lines


def create_child_logger(parent_logger: logging.Logger, context: str) -> logging.Logger:
    """
    Create a child logger with additional context in the name.

    This creates hierarchical logger names like:
    - QSPSimulator -> QSPSimulator.baseline_no_treatment
    - SimulationPool -> SimulationPool.test_v1_fresh

    Args:
        parent_logger: Parent logger instance
        context: Context string to append (e.g., scenario name, pool name)

    Returns:
        Child logger with hierarchical name

    Example:
        >>> base_logger = setup_logger("QSPSimulator")
        >>> scenario_logger = create_child_logger(base_logger, "baseline_no_treatment")
        >>> scenario_logger.info("Starting simulation")
        # Logs as: "QSPSimulator.baseline_no_treatment - INFO - Starting simulation"
    """
    child_name = f"{parent_logger.name}.{context}"
    return logging.getLogger(child_name)


@contextmanager
def log_section(logger: logging.Logger, title: str, separator_width: int = 80):
    """
    Context manager for logging a section with separators.

    Args:
        logger: Logger instance
        title: Section title
        separator_width: Width of separator lines

    Example:
        >>> with log_section(logger, "Multi-Scenario SBI Workflow: test_v1"):
        ...     logger.info("Scenarios: ['baseline', 'treatment']")
        ...     logger.info("Total simulations: 1000")
    """
    logger.info(separator(separator_width))
    logger.info(title)
    logger.info(separator(separator_width))
    try:
        yield
    finally:
        pass  # Could add footer separator if desired


@contextmanager
def log_operation(logger: logging.Logger, operation_name: str, log_start: bool = True):
    """
    Context manager for logging an operation with timing.

    Args:
        logger: Logger instance
        operation_name: Name of the operation being performed
        log_start: Whether to log the start message

    Yields:
        Dictionary that can be updated with additional metrics

    Example:
        >>> with log_operation(logger, "MATLAB simulation") as op:
        ...     run_matlab_simulation()
        # Logs: "✓ MATLAB simulation complete (123.4s)"
    """
    start_time = time.time()
    metrics = {}

    if log_start:
        logger.info(f"Starting {operation_name}...")

    try:
        yield metrics
        elapsed = time.time() - start_time
        logger.info(f"✓ {operation_name} complete ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {operation_name} failed after {elapsed:.1f}s: {e}")
        raise


def log_summary_section(
    logger: logging.Logger, title: str, metrics: Dict[str, Any], separator_width: int = 80
):
    """
    Log a summary section with metrics.

    Args:
        logger: Logger instance
        title: Summary section title
        metrics: Dictionary of metrics to display
        separator_width: Width of separator lines

    Example:
        >>> metrics = {
        ...     "total_requested": 1500,
        ...     "reused_from_pools": 0,
        ...     "newly_generated": 1500,
        ...     "reuse_percentage": "0.0%"
        ... }
        >>> log_summary_section(logger, "Simulation Summary", metrics)
    """
    logger.info(separator(separator_width))
    logger.info(title)
    logger.info(separator(separator_width))
    for line in format_config(metrics):
        logger.info(line)
