"""
Timing Configuration for Satellite Control System

Timing parameters for simulation, control loops, and mission phase durations.
Controls timesteps, update intervals, and stabilization requirements.

Timing categories:
- Simulation Timestep: Integration step for physics simulation
- Control Interval: MPC update frequency
- Mission Timeouts: Maximum duration for different mission types
- Stabilization Times: Hold durations at waypoints and targets
- Phase Transitions: Timing for mission phase changes

Key parameters:
- simulation_dt: Physics integration timestep (typically 0.01s)
- control_dt: MPC control update interval (typically 0.1s)
- target_hold_time: Intermediate waypoint hold duration
- Final stabilization times per mission type

Features:
- Mode-specific stabilization requirements
- Configurable timeout limits
- Independent control and simulation rates
- Integration with mission_state_manager timing
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimingConfig:
    """
    Timing-related constants for simulation and control loops.

    Attributes:
        simulation_dt: Simulation timestep in seconds
        control_dt: MPC control interval in seconds
        max_simulation_time: Maximum simulation duration in seconds
        target_hold_time: Time to hold at intermediate waypoints in seconds
        use_final_stabilization_in_simulation: Enable final stabilization
        waypoint_final_stabilization_time: Final hold time for waypoints
        shape_final_stabilization_time: Final hold time for shapes
        shape_positioning_stabilization_time: Positioning phase hold time
    """

    simulation_dt: float
    control_dt: float
    max_simulation_time: float
    target_hold_time: float
    use_final_stabilization_in_simulation: bool
    waypoint_final_stabilization_time: float
    shape_final_stabilization_time: float
    shape_positioning_stabilization_time: float


# DEFAULT TIMING PARAMETERS
# ============================================================================

# Simulation and control intervals
SIMULATION_DT = 0.005  # seconds
CONTROL_DT = 0.050  # 50 ms
MAX_SIMULATION_TIME = 500.0  # seconds

# Stabilization timers
TARGET_HOLD_TIME = 5.0  # seconds
USE_FINAL_STABILIZATION_IN_SIMULATION = False

# Mission-specific stabilization times
WAYPOINT_FINAL_STABILIZATION_TIME = 10.0  # seconds
SHAPE_FINAL_STABILIZATION_TIME = 15.0  # seconds - final STABILIZING phase
SHAPE_POSITIONING_STABILIZATION_TIME = 5.0  # seconds - PATH_STABILIZATION
DEFAULT_TARGET_SPEED = 0.1  # m/s - default speed for shape following missions


def get_timing_params() -> TimingConfig:
    """
    Get default timing configuration.

    Returns:
        TimingConfig with default timing parameters
    """
    return TimingConfig(
        simulation_dt=SIMULATION_DT,
        control_dt=CONTROL_DT,
        max_simulation_time=MAX_SIMULATION_TIME,
        target_hold_time=TARGET_HOLD_TIME,
        use_final_stabilization_in_simulation=(USE_FINAL_STABILIZATION_IN_SIMULATION),
        waypoint_final_stabilization_time=WAYPOINT_FINAL_STABILIZATION_TIME,
        shape_final_stabilization_time=SHAPE_FINAL_STABILIZATION_TIME,
        shape_positioning_stabilization_time=(SHAPE_POSITIONING_STABILIZATION_TIME),
    )


def validate_timing_params(config: TimingConfig) -> bool:
    """
    Validate timing parameters for consistency.

    Args:
        config: TimingConfig to validate

    Returns:
        True if valid, False otherwise
    """
    issues = []

    # Control interval must be larger than simulation timestep
    if config.control_dt <= config.simulation_dt:
        issues.append(
            f"Control interval ({config.control_dt}s) must be > "
            f"simulation timestep ({config.simulation_dt}s)"
        )

    # Ensure positive values
    if config.simulation_dt <= 0:
        issues.append(f"Simulation timestep must be positive: {config.simulation_dt}")

    if config.control_dt <= 0:
        issues.append(f"Control interval must be positive: {config.control_dt}")

    if config.max_simulation_time <= 0:
        issues.append(f"Max simulation time must be positive: " f"{config.max_simulation_time}")

    # Report validation results
    if issues:
        logger.warning("Timing parameter validation failed:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False

    return True


def print_stabilization_times(config: TimingConfig) -> None:
    """
    Print current stabilization time configuration.

    Args:
        config: TimingConfig to print
    """
    simulation_mode = "ENABLED" if config.use_final_stabilization_in_simulation else "DISABLED"
    logger.info("STABILIZATION TIME CONFIGURATION:")
    logger.info(f"  Final stabilization in simulation: {simulation_mode}")
    logger.info(
        f"  Waypoint Navigation: intermediate={config.target_hold_time:.1f}s, "
        f"final={config.waypoint_final_stabilization_time:.1f}s"
    )
    logger.info(
        f"  Shape Following: "
        f"positioning={config.shape_positioning_stabilization_time:.1f}s, "
        f"final={config.shape_final_stabilization_time:.1f}s"
    )
