"""
Logging Configuration for Satellite Control System

Provides standardized logging setup with consistent formatting across the project.
Supports both console and file output with configurable levels.

Key features:
- Centralized logging configuration
- Console and file output options
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Timestamped log messages with module names
- Simple and detailed format options
- Structured logging support (JSON format)
- Per-module log level configuration
- Context managers for temporary log level changes

Usage:
    from logging_config import setup_logging, get_logger

    # Set up logging
    logger = setup_logging(__name__)

    # Or use convenience function
    logger = get_logger(__name__)

    # Use logger instead of print
    logger.info("System initialized")
    logger.warning("High solve time detected")
    logger.error("MPC solver failed")
    
    # Structured logging
    logger.info("MPC solve", extra={"solve_time": 0.002, "status": "success"})
"""

import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Formats log records as JSON for easy parsing and analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra") and record.extra:
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add stack info if present
        if record.stack_info:
            log_data["stack"] = self.formatStack(record.stack_info)

        return json.dumps(log_data, default=str)


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    simple_format: bool = False,
    structured: bool = False,
) -> logging.Logger:
    """
    Set up standardized logging.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        log_file: Log file path (None for no file logging)
        console: Enable console output
        simple_format: Use simplified format (timestamp only, no level/name)
        structured: Use JSON structured format (for file logging)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Format - choose simple, detailed, or structured
    if structured:
        formatter = StructuredFormatter()
        console_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    elif simple_format:
        formatter = logging.Formatter(fmt="%(asctime)s, %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler (always use readable format)
    if console:
        # Force UTF-8 encoding for Windows compatibility
        import io

        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        console_handler = logging.StreamHandler(utf8_stdout)
        console_handler.setLevel(level)
        if structured:
            console_handler.setFormatter(console_formatter)
        else:
            console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (can use structured format)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get or create a logger with default configuration.
    
    Convenience function that uses module-level defaults.
    
    Args:
        name: Logger name (typically __name__)
        level: Optional log level override
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


@contextmanager
def temporary_log_level(logger_name: str, level: int):
    """
    Context manager for temporarily changing log level.
    
    Usage:
        with temporary_log_level("src.satellite_control.core.mpc_controller", logging.DEBUG):
            # Debug logging enabled here
            mpc_controller.solve(state, target)
        # Original log level restored
    
    Args:
        logger_name: Name of logger to modify
        level: Temporary log level
    """
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(original_level)


def configure_module_log_levels(module_levels: Dict[str, int]) -> None:
    """
    Configure log levels for specific modules.
    
    Args:
        module_levels: Dictionary mapping module names to log levels
        
    Example:
        configure_module_log_levels({
            "src.satellite_control.core.mpc_controller": logging.DEBUG,
            "src.satellite_control.core.simulation": logging.INFO,
        })
    """
    for module_name, level in module_levels.items():
        logger = logging.getLogger(module_name)
        logger.setLevel(level)


# Default module log levels (can be overridden)
DEFAULT_MODULE_LEVELS: Dict[str, int] = {
    "src.satellite_control.core.mpc_controller": logging.INFO,
    "src.satellite_control.core.simulation": logging.INFO,
    "src.satellite_control.core.mujoco_satellite": logging.WARNING,
    "src.satellite_control.control.mpc_controller": logging.INFO,
}

# Apply default module levels on import
configure_module_log_levels(DEFAULT_MODULE_LEVELS)


# Example usage
if __name__ == "__main__":
    # Standard logging
    logger = setup_logging(__name__, log_file="Data/satellite.log")
    logger.info("System initialized")
    logger.warning("High CPU usage detected")
    logger.error("Connection failed")

    # Structured logging
    structured_logger = setup_logging(
        __name__, log_file="Data/satellite_structured.log", structured=True
    )
    structured_logger.info(
        "MPC solve completed",
        extra={"solve_time": 0.002, "status": "success", "iterations": 15},
    )

    # Temporary log level
    with temporary_log_level(__name__, logging.DEBUG):
        logger.debug("This debug message will be shown")
    logger.debug("This debug message will be hidden")
