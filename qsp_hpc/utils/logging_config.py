#!/usr/bin/env python3
"""
Centralized logging configuration for QSP HPC Tools.

Provides consistent logging across all modules with configurable verbosity.
"""

import logging
import sys
from typing import Optional


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

    # Create formatter
    if verbose:
        # Detailed format for debugging
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        # Simpler format for normal use
        formatter = logging.Formatter("%(levelname)s: %(message)s")

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
