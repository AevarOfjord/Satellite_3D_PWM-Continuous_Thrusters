"""
Custom Exception Hierarchy for Satellite Thruster Control System

Defines structured exception classes for error handling across the
entire project.
Provides clear, informative error messages with context-specific details.

Exception categories:
- Configuration errors: Invalid parameters and settings
- MPC/Optimization errors: Solver failures, timeouts, infeasibility
- Hardware/Communication errors: Device connection and communication failures
- Mission/State errors: Mission execution and state validation issues
- Safety errors: Collision detection and safety boundary violations
"""

from typing import Any


class SatelliteControlException(Exception):
    """Base exception for all satellite control system errors."""

    pass


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigurationError(SatelliteControlException):
    """Raised when configuration is invalid or inconsistent."""

    pass


class ParameterValidationError(ConfigurationError):
    """Raised when a parameter fails validation."""

    def __init__(self, parameter_name: str, value: Any, reason: str) -> None:
        message = f"Invalid parameter '{parameter_name}' = {value}: {reason}"
        super().__init__(message)
        self.parameter_name = parameter_name
        self.value = value
        self.reason = reason


# ============================================================================
# MPC/Optimization Errors
# ============================================================================


class MPCException(SatelliteControlException):
    """Base exception for MPC controller errors."""

    pass


class OptimizationError(MPCException):
    """Raised when optimization solver fails."""

    def __init__(self, status: str, message: str = ""):
        full_message = f"Optimization failed with status: {status}"
        if message:
            full_message += f" - {message}"
        super().__init__(full_message)
        self.status = status


class SolverTimeoutError(MPCException):
    """Raised when MPC solver exceeds time limit."""

    def __init__(self, time_limit: float, actual_time: float):
        message = f"Solver timeout: {actual_time:.3f}s exceeded limit of " f"{time_limit:.3f}s"
        super().__init__(message)
        self.time_limit = time_limit
        self.actual_time = actual_time


class InfeasibleProblemError(MPCException):
    """Raised when optimization problem is infeasible."""

    def __init__(self, reason: str = ""):
        message = "Optimization problem is infeasible"
        if reason:
            message += f": {reason}"
        super().__init__(message)


# ============================================================================
# Hardware/Communication Errors
# ============================================================================


class HardwareException(SatelliteControlException):
    """Base exception for hardware interface errors."""

    pass


class HardwareConnectionError(HardwareException):
    """Raised when hardware connection fails."""

    def __init__(self, device: str, reason: str = ""):
        message = f"Failed to connect to {device}"
        if reason:
            message += f": {reason}"
        super().__init__(message)
        self.device = device


class MotionCaptureError(HardwareException):
    """Raised when motion capture system errors occur."""

    def __init__(self, reason: str):
        message = f"Motion capture error: {reason}"
        super().__init__(message)


# ============================================================================
# Safety Errors
# ============================================================================


class SafetyException(SatelliteControlException):
    """Base exception for safety violations."""

    pass


class SafetyViolationError(SafetyException):
    """Raised when a safety constraint is violated."""

    def __init__(self, violation_type: str, current_value: Any, limit: Any) -> None:
        message = f"Safety violation: {violation_type} = {current_value} " f"exceeds limit {limit}"
        super().__init__(message)
        self.violation_type = violation_type
        self.current_value = current_value
        self.limit = limit


class WorkspaceBoundaryError(SafetyException):
    """Raised when satellite exceeds workspace boundaries."""

    def __init__(self, position: tuple, bounds: float):
        message = f"Workspace boundary exceeded: position {position} " f"outside ±{bounds}m"
        super().__init__(message)
        self.position = position
        self.bounds = bounds


class VelocityLimitError(SafetyException):
    """Raised when velocity exceeds safety limits."""

    def __init__(self, velocity: float, limit: float, velocity_type: str = "linear"):
        message = (
            f"{velocity_type.capitalize()} velocity {velocity:.3f} " f"exceeds limit {limit:.3f}"
        )
        super().__init__(message)
        self.velocity = velocity
        self.limit = limit
        self.velocity_type = velocity_type


# ============================================================================
# Mission Errors
# ============================================================================


class MissionException(SatelliteControlException):
    """Base exception for mission-related errors."""

    pass


class InvalidMissionError(MissionException):
    """Raised when mission configuration is invalid."""

    def __init__(self, reason: str):
        message = f"Invalid mission configuration: {reason}"
        super().__init__(message)


class MissionTimeoutError(MissionException):
    """Raised when mission exceeds time limit."""

    def __init__(self, elapsed_time: float, time_limit: float):
        message = f"Mission timeout: {elapsed_time:.1f}s exceeded limit " f"{time_limit:.1f}s"
        super().__init__(message)
        self.elapsed_time = elapsed_time
        self.time_limit = time_limit


# ============================================================================
# Simulation Errors
# ============================================================================


class SimulationException(SatelliteControlException):
    """Base exception for simulation errors."""

    pass


class SimulationDivergenceError(SimulationException):
    """Raised when simulation diverges (numerical instability)."""

    def __init__(self, step: int, state: Any) -> None:
        message = f"Simulation diverged at step {step}: state = {state}"
        super().__init__(message)
        self.step = step
        self.state = state


class InvalidStateError(SimulationException):
    """Raised when state vector is invalid."""

    def __init__(self, state: Any, reason: str) -> None:
        message = f"Invalid state: {reason}"
        super().__init__(message)
        self.state = state
        self.reason = reason


# ============================================================================
# Data/File Errors
# ============================================================================


class DataException(SatelliteControlException):
    """Base exception for data handling errors."""

    pass


class DataFileError(DataException):
    """Raised when data file operations fail."""

    def __init__(self, file_path: str, operation: str, reason: str = ""):
        message = f"Data file {operation} failed for {file_path}"
        if reason:
            message += f": {reason}"
        super().__init__(message)
        self.file_path = file_path
        self.operation = operation


class InvalidDataError(DataException):
    """Raised when data is malformed or invalid."""

    def __init__(self, data_type: str, reason: str):
        message = f"Invalid {data_type} data: {reason}"
        super().__init__(message)
        self.data_type = data_type
        self.reason = reason


# ============================================================================
# Utility Functions
# ============================================================================


def format_exception_message(exc: Exception) -> str:
    """
    Format exception message with context information.

    Args:
        exc: Exception to format

    Returns:
        Formatted error message string
    """
    exc_type = type(exc).__name__
    exc_message = str(exc)
    return f"[{exc_type}] {exc_message}"


def is_safety_critical(exc: Exception) -> bool:
    """
    Determine if an exception is safety-critical.

    Safety-critical exceptions should trigger immediate shutdown.

    Args:
        exc: Exception to check

    Returns:
        True if exception is safety-critical
    """
    return isinstance(exc, SafetyException)


def is_recoverable(exc: Exception) -> bool:
    """
    Determine if an exception is potentially recoverable.

    Args:
        exc: Exception to check

    Returns:
        True if exception might be recoverable
    """
    recoverable_types = (
        MotionCaptureError,
        SolverTimeoutError,
    )
    return isinstance(exc, recoverable_types)
