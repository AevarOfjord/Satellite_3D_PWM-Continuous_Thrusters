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

Usage:
    from logging_config import setup_logging

    # Set up logging
    logger = setup_logging(__name__)

    # Use logger instead of print
    logger.info("System initialized")
    logger.warning("High solve time detected")
    logger.error("MPC solver failed")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    simple_format: bool = False,
) -> logging.Logger:
    """
    Set up standardized logging.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        log_file: Log file path (None for no file logging)
        console: Enable console output
        simple_format: Use simplified format (timestamp only, no level/name)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Format - choose simple or detailed
    if simple_format:
        formatter = logging.Formatter(fmt="%(asctime)s, %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    if console:
        # Force UTF-8 encoding for Windows compatibility
        import io

        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        console_handler = logging.StreamHandler(utf8_stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Example usage
if __name__ == "__main__":
    logger = setup_logging(__name__, log_file="Data/satellite.log")

    logger.info("System initialized")
    logger.warning("High CPU usage detected")
    logger.error("Connection failed")
