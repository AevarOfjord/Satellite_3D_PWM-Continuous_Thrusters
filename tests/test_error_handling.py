"""
Tests for error handling utilities.

Tests decorators, context managers, and error handling patterns.
"""

import logging
import pytest

from src.satellite_control.core.error_handling import (
    ErrorHandler,
    error_context,
    handle_recoverable_error,
    log_and_continue,
    safe_execute,
    with_error_context,
)
from src.satellite_control.core.exceptions import (
    OperationError,
    OptimizationError,
    SolverTimeoutError,
    is_recoverable,
    is_safety_critical,
)


class TestWithErrorContext:
    """Tests for @with_error_context decorator."""

    def test_successful_execution(self, caplog):
        """Decorator should not interfere with successful execution."""
        @with_error_context("test operation", reraise=True)
        def successful_function():
            return 42

        result = successful_function()
        assert result == 42
        assert len(caplog.records) == 0

    def test_exception_logging(self, caplog):
        """Decorator should log exceptions."""
        @with_error_context("test operation", reraise=True)
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(OperationError) as exc_info:
            failing_function()

        assert "test operation" in str(exc_info.value)
        assert len(caplog.records) > 0
        assert "test operation" in caplog.records[0].message

    def test_no_reraising(self, caplog):
        """Decorator can suppress exceptions."""
        @with_error_context("test operation", reraise=False)
        def failing_function():
            raise ValueError("test error")

        result = failing_function()
        assert result is None
        assert len(caplog.records) > 0

    def test_captures_args(self, caplog):
        """Decorator can capture function arguments."""
        @with_error_context("test operation", reraise=True, capture_args=True)
        def failing_function(x, y):
            raise ValueError(f"error with {x} and {y}")

        with pytest.raises(OperationError):
            failing_function(1, 2)

        # Check that args were logged
        assert any("args" in record.message.lower() for record in caplog.records)


class TestErrorContext:
    """Tests for error_context context manager."""

    def test_successful_execution(self):
        """Context manager should not interfere with successful execution."""
        with error_context("test operation"):
            result = 42
        assert result == 42

    def test_exception_logging(self, caplog):
        """Context manager should log exceptions."""
        with pytest.raises(OperationError):
            with error_context("test operation", reraise=True):
                raise ValueError("test error")

        assert len(caplog.records) > 0
        assert "test operation" in caplog.records[0].message

    def test_suppress_exceptions(self, caplog):
        """Context manager can suppress specific exceptions."""
        with error_context("test operation", reraise=False, suppress_exceptions=ValueError):
            raise ValueError("test error")

        assert len(caplog.records) > 0

    def test_no_reraising(self, caplog):
        """Context manager can suppress all exceptions."""
        with error_context("test operation", reraise=False):
            raise ValueError("test error")

        assert len(caplog.records) > 0


class TestHandleRecoverableError:
    """Tests for handle_recoverable_error function."""

    def test_successful_execution(self):
        """Function should return result on success."""
        def successful_function():
            return 42

        result = handle_recoverable_error(successful_function)
        assert result == 42

    def test_recoverable_error_retry(self, caplog):
        """Function should retry on recoverable errors."""
        attempts = [0]

        def failing_function():
            attempts[0] += 1
            if attempts[0] < 3:
                raise SolverTimeoutError("temporary failure")
            return 42

        result = handle_recoverable_error(failing_function, max_retries=3)
        assert result == 42
        assert attempts[0] == 3

    def test_unrecoverable_error_fails(self):
        """Function should fail immediately on unrecoverable errors."""
        def failing_function():
            raise ValueError("unrecoverable error")

        with pytest.raises(OperationError):
            handle_recoverable_error(failing_function)

    def test_max_retries_exceeded(self, caplog):
        """Function should fail after max retries."""
        def always_failing():
            raise SolverTimeoutError("always fails")

        with pytest.raises(OperationError):
            handle_recoverable_error(always_failing, max_retries=2)

        # Should have attempted 3 times (initial + 2 retries)
        assert len(caplog.records) >= 3


class TestSafeExecute:
    """Tests for safe_execute function."""

    def test_successful_execution(self):
        """Function should return result on success."""
        def successful_function():
            return 42

        result = safe_execute(successful_function)
        assert result == 42

    def test_exception_returns_default(self, caplog):
        """Function should return default on exception."""
        def failing_function():
            raise ValueError("error")

        result = safe_execute(failing_function, default_return=0)
        assert result == 0
        assert len(caplog.records) > 0

    def test_exception_returns_none(self, caplog):
        """Function should return None by default on exception."""
        def failing_function():
            raise ValueError("error")

        result = safe_execute(failing_function)
        assert result is None


class TestLogAndContinue:
    """Tests for @log_and_continue decorator."""

    def test_successful_execution(self):
        """Decorator should not interfere with successful execution."""
        @log_and_continue("test operation")
        def successful_function():
            return 42

        result = successful_function()
        assert result == 42

    def test_exception_logged_and_returns_none(self, caplog):
        """Decorator should log exception and return None."""
        @log_and_continue("test operation")
        def failing_function():
            raise ValueError("error")

        result = failing_function()
        assert result is None
        assert len(caplog.records) > 0


class TestErrorHandler:
    """Tests for ErrorHandler class."""

    def test_record_error(self, caplog):
        """ErrorHandler should record errors."""
        handler = ErrorHandler("TestContext")
        handler.record_error("test operation", ValueError("error"))

        assert handler.has_errors()
        assert len(handler.get_errors()) == 1
        assert len(caplog.records) > 0

    def test_clear_errors(self):
        """ErrorHandler should clear recorded errors."""
        handler = ErrorHandler("TestContext")
        handler.record_error("test operation", ValueError("error"))
        handler.clear_errors()

        assert not handler.has_errors()
        assert len(handler.get_errors()) == 0

    def test_context_manager(self, caplog):
        """ErrorHandler context manager should work."""
        handler = ErrorHandler("TestContext")

        with handler.handle("test operation", reraise=False):
            raise ValueError("error")

        assert handler.has_errors()
        assert len(caplog.records) > 0

    def test_context_manager_reraises(self):
        """ErrorHandler context manager can re-raise."""
        handler = ErrorHandler("TestContext")

        with pytest.raises(OperationError):
            with handler.handle("test operation", reraise=True):
                raise ValueError("error")


class TestExceptionUtilities:
    """Tests for exception utility functions."""

    def test_is_recoverable(self):
        """Test is_recoverable function."""
        assert is_recoverable(SolverTimeoutError("timeout"))
        assert not is_recoverable(ValueError("error"))

    def test_is_safety_critical(self):
        """Test is_safety_critical function."""
        from src.satellite_control.core.exceptions import SafetyException

        assert is_safety_critical(SafetyException("safety violation"))
        assert not is_safety_critical(ValueError("error"))
