# Logging Configuration Guide

This guide explains how to configure and use logging in the Satellite Control System.

## Quick Start

```python
from src.satellite_control.utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("System initialized")
```

## Logging Levels

The system uses standard Python logging levels:

- **DEBUG**: Detailed diagnostic information (development only)
- **INFO**: General informational messages (default)
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical errors requiring immediate attention

## Basic Usage

### Standard Logging

```python
from src.satellite_control.utils.logging_config import setup_logging

logger = setup_logging(__name__)
logger.info("Starting simulation")
logger.warning("High solve time detected")
logger.error("MPC solver failed")
```

### Structured Logging

For machine-readable logs (JSON format):

```python
logger = setup_logging(__name__, structured=True, log_file="simulation.jsonl")

logger.info(
    "MPC solve completed",
    extra={
        "solve_time": 0.002,
        "status": "success",
        "iterations": 15,
        "objective_value": 0.123
    }
)
```

Output (JSON):
```json
{
  "timestamp": "2026-01-07 10:30:45",
  "level": "INFO",
  "logger": "src.satellite_control.control.mpc_controller",
  "message": "MPC solve completed",
  "solve_time": 0.002,
  "status": "success",
  "iterations": 15,
  "objective_value": 0.123
}
```

## Per-Module Log Levels

Configure different log levels for different modules:

```python
from src.satellite_control.utils.logging_config import configure_module_log_levels
import logging

configure_module_log_levels({
    "src.satellite_control.core.mpc_controller": logging.DEBUG,
    "src.satellite_control.core.simulation": logging.INFO,
    "src.satellite_control.core.mujoco_satellite": logging.WARNING,
})
```

## Temporary Log Level Changes

Use context manager for temporary debug logging:

```python
from src.satellite_control.utils.logging_config import temporary_log_level
import logging

# Temporarily enable debug logging
with temporary_log_level("src.satellite_control.control.mpc_controller", logging.DEBUG):
    result = mpc_controller.solve(state, target)
    # Debug messages will be shown here
# Original log level restored
```

## Log File Configuration

### Console and File Logging

```python
logger = setup_logging(
    __name__,
    log_file="Data/simulation.log",
    console=True  # Also log to console
)
```

### File Only (Headless Mode)

```python
logger = setup_logging(
    __name__,
    log_file="Data/simulation.log",
    console=False  # Only log to file
)
```

### Simple Format (CSV-like)

```python
logger = setup_logging(
    __name__,
    log_file="Data/simulation.csv",
    simple_format=True  # Timestamp, message only
)
```

## Best Practices

### 1. Use Appropriate Log Levels

```python
# Good
logger.debug(f"State vector: {state}")  # Only in development
logger.info("MPC solve completed")  # Normal operation
logger.warning(f"Solve time {time:.3f}s exceeds threshold")  # Potential issue
logger.error(f"MPC solve failed: {error}")  # Error occurred
logger.critical("Safety violation detected!")  # Critical failure
```

### 2. Include Context

```python
# Good - includes context
logger.warning(
    f"MPC solve time ({solve_time:.3f}s) exceeds limit ({limit:.3f}s)",
    extra={"solve_time": solve_time, "limit": limit, "step": step_number}
)

# Less useful - no context
logger.warning("MPC solve too slow")
```

### 3. Use Structured Logging for Analysis

```python
# Structured logging makes analysis easier
logger.info(
    "Control update",
    extra={
        "step": step_number,
        "solve_time": solve_time,
        "position_error": pos_error,
        "angle_error": ang_error,
        "active_thrusters": active_count
    }
)
```

### 4. Don't Log Sensitive Data

```python
# Bad
logger.info(f"API key: {api_key}")

# Good
logger.info("API authentication successful")
```

## Integration with Error Handling

The error handling system automatically logs errors:

```python
from src.satellite_control.core.error_handling import with_error_context

@with_error_context("MPC solve")
def solve_mpc(self, state, target):
    # Errors are automatically logged with context
    return self.controller.solve(state, target)
```

## Performance Considerations

- **File logging**: Use buffered writes for high-frequency logging
- **Structured logging**: Slightly slower but enables better analysis
- **Debug logging**: Disable in production to avoid performance impact

## Troubleshooting

### Logs Not Appearing

1. Check log level:
   ```python
   logger.setLevel(logging.DEBUG)
   ```

2. Verify handlers:
   ```python
   print(logger.handlers)  # Should show handlers
   ```

3. Check file permissions:
   ```python
   # Ensure log directory is writable
   Path("Data").mkdir(exist_ok=True)
   ```

### Too Much Logging

Use per-module log levels to reduce noise:

```python
configure_module_log_levels({
    "noisy_module": logging.WARNING,  # Only warnings and errors
})
```

### Log File Too Large

- Use log rotation (implement with `RotatingFileHandler`)
- Disable debug logging in production
- Use structured logging with filtering

## Examples

### Simulation Logging

```python
from src.satellite_control.utils.logging_config import setup_logging

logger = setup_logging(
    __name__,
    log_file="Data/Simulation/run.log",
    level=logging.INFO
)

logger.info("Starting simulation")
logger.info(f"Configuration: {config}")
logger.warning("High solve time detected")
logger.error("Simulation failed", exc_info=True)
```

### Debug Session

```python
from src.satellite_control.utils.logging_config import temporary_log_level
import logging

# Enable debug for specific module
with temporary_log_level("src.satellite_control.control.mpc_controller", logging.DEBUG):
    # Debug messages will appear
    result = mpc_controller.solve(state, target)
```

### Structured Logging for Analysis

```python
logger = setup_logging(__name__, structured=True, log_file="analysis.jsonl")

# Log with structured data
logger.info(
    "Step completed",
    extra={
        "step": step,
        "time": time,
        "state": state.tolist(),
        "control": control.tolist(),
        "error": error
    }
)
```
