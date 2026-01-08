"""
Core simulation and control modules.

This package contains the core simulation engine, physics interface,
and control algorithms.
"""

from .exceptions import (
    ConfigurationError,
    DataException,
    HardwareException,
    InvalidMissionError,
    InvalidStateError,
    MissionException,
    MissionTimeoutError,
    MPCException,
    OptimizationError,
    ParameterValidationError,
    SafetyException,
    SafetyViolationError,
    SatelliteControlException,
    SimulationDivergenceError,
    SimulationException,
    SolverTimeoutError,
    WorkspaceBoundaryError,
    format_exception_message,
    is_recoverable,
    is_safety_critical,
)

# Error handling utilities
from .error_handling import (
    ErrorHandler,
    error_context,
    handle_recoverable_error,
    log_and_continue,
    safe_execute,
    with_error_context,
)

__all__ = [
    # Exceptions
    "SatelliteControlException",
    "ConfigurationError",
    "ParameterValidationError",
    "MPCException",
    "OptimizationError",
    "SolverTimeoutError",
    "HardwareException",
    "SafetyException",
    "SafetyViolationError",
    "WorkspaceBoundaryError",
    "MissionException",
    "InvalidMissionError",
    "MissionTimeoutError",
    "SimulationException",
    "SimulationDivergenceError",
    "InvalidStateError",
    "DataException",
    # Exception utilities
    "format_exception_message",
    "is_safety_critical",
    "is_recoverable",
    # Error handling utilities
    "with_error_context",
    "error_context",
    "ErrorHandler",
    "handle_recoverable_error",
    "safe_execute",
    "log_and_continue",
]
