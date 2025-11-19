#!/usr/bin/env python3
"""
Tests for logging configuration utilities.

Tests logger setup, verbosity control, and formatting.
"""

import logging

from qsp_hpc.utils.logging_config import get_logger, set_verbosity, setup_logger


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_basic(self):
        """Test basic logger setup with default settings."""
        logger = setup_logger("test.basic")

        assert logger.name == "test.basic"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.propagate is False

        # Clean up
        logger.handlers.clear()

    def test_setup_logger_verbose(self):
        """Test logger setup with verbose=True."""
        logger = setup_logger("test.verbose", verbose=True)

        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1

        # Check verbose formatter includes timestamp and logger name
        handler = logger.handlers[0]
        formatter = handler.formatter
        assert formatter is not None

        # Verbose format should include asctime, name, levelname, message
        # We can't directly check the format string, but we can check the output
        record = logging.LogRecord(
            name="test.verbose",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "test.verbose" in formatted
        assert "DEBUG" in formatted
        assert "Test message" in formatted

        # Clean up
        logger.handlers.clear()

    def test_setup_logger_non_verbose_format(self):
        """Test that non-verbose logger has simple format."""
        logger = setup_logger("test.simple", verbose=False)

        handler = logger.handlers[0]
        formatter = handler.formatter

        # Create a test record
        record = logging.LogRecord(
            name="test.simple",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)

        # Simple format should just be "LEVEL: message"
        assert formatted == "INFO: Test message"

        # Clean up
        logger.handlers.clear()

    def test_setup_logger_custom_level(self):
        """Test logger setup with custom level."""
        logger = setup_logger("test.custom", level=logging.WARNING)

        assert logger.level == logging.WARNING

        # Clean up
        logger.handlers.clear()

    def test_setup_logger_already_configured(self):
        """Test that already configured logger is returned as-is."""
        # First setup
        logger1 = setup_logger("test.configured")
        handler_count_1 = len(logger1.handlers)

        # Second setup should not add more handlers
        logger2 = setup_logger("test.configured")
        handler_count_2 = len(logger2.handlers)

        assert logger1 is logger2
        assert handler_count_1 == handler_count_2

        # Clean up
        logger1.handlers.clear()

    def test_setup_logger_prevents_propagation(self):
        """Test that logger does not propagate to root logger."""
        logger = setup_logger("test.propagation")

        assert logger.propagate is False

        # Clean up
        logger.handlers.clear()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_basic(self):
        """Test getting a logger."""
        logger = get_logger("test.get")

        assert logger.name == "test.get"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_same_instance(self):
        """Test that get_logger returns same instance for same name."""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")

        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Test that different names return different loggers."""
        logger1 = get_logger("test.logger1")
        logger2 = get_logger("test.logger2")

        assert logger1 is not logger2
        assert logger1.name == "test.logger1"
        assert logger2.name == "test.logger2"


class TestSetVerbosity:
    """Tests for set_verbosity function."""

    def test_set_verbosity_debug(self):
        """Test setting verbosity to DEBUG."""
        # Create some qsp_hpc loggers
        logger1 = setup_logger("qsp_hpc.test1", verbose=False)
        logger2 = setup_logger("qsp_hpc.test2", verbose=False)

        # Both should be at INFO level
        assert logger1.level == logging.INFO
        assert logger2.level == logging.INFO

        # Set verbosity
        set_verbosity(verbose=True)

        # Both should now be at DEBUG level
        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.DEBUG

        # Handlers should also be at DEBUG level
        for handler in logger1.handlers:
            assert handler.level == logging.DEBUG
        for handler in logger2.handlers:
            assert handler.level == logging.DEBUG

        # Clean up
        logger1.handlers.clear()
        logger2.handlers.clear()

    def test_set_verbosity_info(self):
        """Test setting verbosity to INFO."""
        # Create some qsp_hpc loggers with DEBUG level
        logger1 = setup_logger("qsp_hpc.test3", verbose=True)
        logger2 = setup_logger("qsp_hpc.test4", verbose=True)

        # Both should be at DEBUG level
        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.DEBUG

        # Set verbosity to False
        set_verbosity(verbose=False)

        # Both should now be at INFO level
        assert logger1.level == logging.INFO
        assert logger2.level == logging.INFO

        # Handlers should also be at INFO level
        for handler in logger1.handlers:
            assert handler.level == logging.INFO
        for handler in logger2.handlers:
            assert handler.level == logging.INFO

        # Clean up
        logger1.handlers.clear()
        logger2.handlers.clear()

    def test_set_verbosity_only_affects_qsp_hpc_loggers(self):
        """Test that set_verbosity only affects qsp_hpc.* loggers."""
        # Create a qsp_hpc logger and a non-qsp_hpc logger
        qsp_logger = setup_logger("qsp_hpc.test5", verbose=False)
        other_logger = setup_logger("other.test", verbose=False)

        # Set verbosity
        set_verbosity(verbose=True)

        # Only qsp_hpc logger should be affected
        assert qsp_logger.level == logging.DEBUG
        assert other_logger.level == logging.INFO  # Unchanged

        # Clean up
        qsp_logger.handlers.clear()
        other_logger.handlers.clear()

    def test_set_verbosity_no_handlers(self):
        """Test set_verbosity works with loggers that have no handlers."""
        # Get a logger without setting it up (no handlers)
        logger = get_logger("qsp_hpc.test6")
        logger.setLevel(logging.INFO)

        # This should not raise an error
        set_verbosity(verbose=True)

        # Logger level should be updated
        assert logger.level == logging.DEBUG
