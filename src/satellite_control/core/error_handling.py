"""
Error Handling Utilities for Satellite Control System

Provides consistent error handling patterns across the codebase:
- Decorators for automatic error context
- Context managers for error handling
- Retry logic for recoverable errors
- Error logging and reporting

Usage:
    from src.satellite_control.core.error_handling import with_error_context
    
    @with_error_context("MPC solve")
    def solve_mpc(self, state, target):
        return self.controller.solve(state, target)
"""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Tuple, Type, TypeVar, Union

from src.satellite_control.core.exceptions import (
    OperationError,
    SatelliteControlException,
    is_recoverable,
    is_safety_critical,
)

logger = logging.getLogger(__name__)

# Type variable for function return type
F = TypeVar("F", bound=Callable[..., Any])


def with_error_context(
    operation: str,
    reraise: bool = True,
    log_level: int = logging.ERROR,
    capture_args: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to add error context to function calls.
    
    Automatically catches exceptions, logs them with context, and optionally
    re-raises them wrapped in SatelliteControlException.
    
    Args:
        operation: Description of the operation (e.g., "MPC solve", "Physics step")
        reraise: If True, re-raise exception (wrapped if needed). If False, log and return None.
        log_level: Logging level for errors (default: ERROR)
        capture_args: If True, log function arguments in error messages
        
    Returns:
        Decorated function
        
    Example:
        @with_error_context("MPC solve")
        def solve_mpc(self, state, target):
            return self.controller.solve(state, target)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except SatelliteControlException:
                # Re-raise our custom exceptions as-is
                raise
            except KeyboardInterrupt:
                # Don't wrap keyboard interrupts
                raise
            except Exception as e:
                # Log with context
                error_msg = f"{operation} failed: {e}"
                if capture_args:
                    error_msg += f" (args={args}, kwargs={kwargs})"
                logger.log(log_level, error_msg, exc_info=True)
                
                # Check if safety-critical
                if is_safety_critical(e):
                    logger.critical(f"SAFETY CRITICAL: {error_msg}")
                
                if reraise:
                    # Wrap in OperationError if not already a SatelliteControlException
                    if not isinstance(e, SatelliteControlException):
                        raise OperationError(operation, e) from e
                    raise
                return None
        return wrapper  # type: ignore
    return decorator


@contextmanager
def error_context(
    operation: str,
    reraise: bool = True,
    log_level: int = logging.ERROR,
    suppress_exceptions: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None,
):
    """
    Context manager for error handling with context.
    
    Usage:
        with error_context("Loading configuration"):
            config = load_config()
    
    Args:
        operation: Description of the operation
        reraise: If True, re-raise exception. If False, log and suppress.
        log_level: Logging level for errors
        suppress_exceptions: Exception type(s) to suppress even if reraise=True
    """
    try:
        yield
    except SatelliteControlException:
        raise
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # Check if this exception type should be suppressed
        if suppress_exceptions and isinstance(e, suppress_exceptions):
            error_msg = f"{operation} failed (suppressed): {e}"
            logger.log(log_level, error_msg, exc_info=True)
            return
        
        error_msg = f"{operation} failed: {e}"
        logger.log(log_level, error_msg, exc_info=True)
        
        if is_safety_critical(e):
            logger.critical(f"SAFETY CRITICAL: {error_msg}")
        
        if reraise:
            if not isinstance(e, SatelliteControlException):
                raise OperationError(operation, e) from e
            raise


def handle_recoverable_error(
    func: Callable[..., Any],
    max_retries: int = 3,
    operation: str = "Operation",
    default_return: Any = None,
) -> Any:
    """
    Execute function with retry logic for recoverable errors.
    
    Retries up to max_retries times if error is recoverable.
    Non-recoverable errors are raised immediately.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts (total attempts = max_retries + 1)
        operation: Description of operation for logging
        default_return: Value to return if all retries fail (None = raise exception)
        
    Returns:
        Function result or default_return if all retries fail
        
    Raises:
        Exception: If error is not recoverable or all retries exhausted
    """
    last_exception: Optional[Exception] = None
    
    # Total attempts = initial + max_retries
    total_attempts = max_retries + 1
    
    for attempt in range(total_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            # Don't retry non-recoverable errors
            if not is_recoverable(e):
                logger.error(f"{operation} failed (non-recoverable): {e}")
                raise OperationError(operation, e) from e
            
            # Don't retry safety-critical errors
            if is_safety_critical(e):
                logger.critical(f"{operation} failed (safety-critical): {e}")
                raise OperationError(operation, e) from e
            
            # Log retry attempt
            if attempt < total_attempts - 1:
                logger.warning(
                    f"{operation} failed (attempt {attempt + 1}/{total_attempts}): {e}. Retrying..."
                )
            else:
                logger.error(f"{operation} failed after {total_attempts} attempts: {e}")
    
    # All retries exhausted
    if default_return is not None:
        logger.warning(f"{operation} failed, returning default value")
        return default_return
    
    if last_exception:
        raise OperationError(operation, last_exception) from last_exception
    raise OperationError(operation, RuntimeError(f"{operation} failed after {total_attempts} attempts"))


def safe_execute(
    func: Callable[..., Any],
    operation: str = "Operation",
    default_return: Any = None,
    log_errors: bool = True,
) -> Any:
    """
    Safely execute a function, catching all exceptions.
    
    Useful for non-critical operations that shouldn't crash the system.
    
    Args:
        func: Function to execute
        operation: Description of operation for logging
        default_return: Value to return if function fails
        log_errors: If True, log errors. If False, suppress.
        
    Returns:
        Function result or default_return if exception occurs
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"{operation} failed: {e}", exc_info=True)
        return default_return


class ErrorHandler:
    """
    Error handler class for more complex error handling scenarios.
    
    Provides methods for handling errors with different strategies.
    """
    
    def __init__(self, context: str, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize error handler.
        
        Args:
            context: Description of the context/operation
            logger_instance: Logger to use (default: module logger)
        """
        self.context = context
        self.logger = logger_instance or logger
        self._errors: List[Tuple[str, Exception]] = []
    
    def record_error(self, operation: str, error: Exception) -> None:
        """
        Record an error for later inspection.
        
        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
        """
        self._errors.append((operation, error))
        error_msg = f"{self.context}: {operation} failed: {error}"
        self.logger.error(error_msg, exc_info=True)
    
    def clear_errors(self) -> None:
        """Clear all recorded errors."""
        self._errors.clear()
    
    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return len(self._errors) > 0
    
    def get_errors(self) -> List[Tuple[str, Exception]]:
        """Get list of recorded errors."""
        return self._errors.copy()
    
    @contextmanager
    def handle(self, operation: str, reraise: bool = True):
        """
        Context manager for error handling.
        
        Args:
            operation: Description of the operation
            reraise: If True, re-raise exceptions. If False, record and suppress.
        """
        try:
            yield
        except SatelliteControlException:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.record_error(operation, e)
            
            if is_safety_critical(e):
                self.logger.critical(f"SAFETY CRITICAL: {self.context}: {operation} failed: {e}")
            
            if reraise:
                if not isinstance(e, SatelliteControlException):
                    raise OperationError(operation, e) from e
                raise
    
    def execute(
        self,
        func: Callable[..., Any],
        reraise: bool = True,
        default_return: Any = None,
    ) -> Any:
        """
        Execute function with error handling.
        
        Args:
            func: Function to execute
            reraise: If True, re-raise exceptions. If False, return default_return.
            default_return: Value to return if reraise=False and exception occurs
            
        Returns:
            Function result or default_return
        """
        try:
            return func()
        except SatelliteControlException:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            error_msg = f"{self.context} failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            
            if is_safety_critical(e):
                self.logger.critical(f"SAFETY CRITICAL: {error_msg}")
            
            if reraise:
                if not isinstance(e, SatelliteControlException):
                    raise SatelliteControlException(error_msg) from e
                raise
            
            return default_return
    
    def execute_with_retry(
        self,
        func: Callable[..., Any],
        max_retries: int = 3,
        default_return: Any = None,
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            default_return: Value to return if all retries fail
            
        Returns:
            Function result or default_return
        """
        return handle_recoverable_error(
            func=func,
            max_retries=max_retries,
            operation=self.context,
            default_return=default_return,
        )


def log_and_continue(operation: str = "Operation"):
    """
    Decorator that logs errors but allows execution to continue.
    
    Useful for non-critical operations in loops or batch processing.
    
    Args:
        operation: Description of the operation
        
    Returns:
        Decorated function that returns None on error
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{operation} failed (continuing): {e}")
                return None
        return wrapper  # type: ignore
    return decorator
