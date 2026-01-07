"""
Simulation Constants for Satellite Control System

Named constants to replace magic numbers throughout the codebase.
Provides clear documentation and single source of truth for threshold values.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class SimulationConstants:
    """
    Named constants for simulation parameters.

    All values are documented with their purpose and units.
    Using frozen dataclass ensures immutability.
    """

    # Target change detection thresholds
    # Used to detect when mission manager advances to a new waypoint
    TARGET_POSITION_CHANGE_THRESHOLD: float = 1e-4  # meters
    TARGET_ANGLE_CHANGE_THRESHOLD: float = 1e-4  # radians

    # MPC timing margins
    # Warning threshold: MPC must complete before next control update
    MPC_TIME_MARGIN_SECONDS: float = 0.02  # 20ms buffer before control deadline

    # Thruster activation threshold
    # Values below this are considered "off" for display purposes
    THRUSTER_ACTIVATION_THRESHOLD: float = 0.01  # 1% duty cycle

    # Physics stability thresholds
    # Used to detect simulation divergence
    MAX_REASONABLE_POSITION: float = 10.0  # meters from origin
    MAX_REASONABLE_VELOCITY: float = 5.0  # m/s
    MAX_REASONABLE_ANGULAR_VELOCITY: float = 10.0  # rad/s

    # Waypoint advancement threshold
    # Minimum progress required to consider waypoint advancement
    WAYPOINT_PROGRESS_THRESHOLD: float = 0.01  # meters

    # DXF path tracking
    DXF_PATH_COMPLETION_THRESHOLD: float = 0.02  # meters from path end
    DXF_SPEED_TOLERANCE: float = 0.01  # m/s speed matching tolerance

    # State validation
    STATE_VECTOR_LENGTH: int = 13  # [pos(3), quat(4), vel(3), w(3)]
    NUM_THRUSTERS: int = 12

    # Numerical stability
    EPSILON: float = 1e-10  # Small value to avoid division by zero
    ANGLE_WRAP_TOLERANCE: float = 1e-6  # Tolerance for angle wrap checks


# Global singleton instance
SIMULATION_CONSTANTS: Final[SimulationConstants] = SimulationConstants()


# Convenience aliases for common constants
TARGET_CHANGE_THRESHOLD: Final[float] = SIMULATION_CONSTANTS.TARGET_POSITION_CHANGE_THRESHOLD
MPC_TIME_MARGIN: Final[float] = SIMULATION_CONSTANTS.MPC_TIME_MARGIN_SECONDS
THRUSTER_THRESHOLD: Final[float] = SIMULATION_CONSTANTS.THRUSTER_ACTIVATION_THRESHOLD
STATE_DIM: Final[int] = SIMULATION_CONSTANTS.STATE_VECTOR_LENGTH
NUM_THRUSTERS: Final[int] = SIMULATION_CONSTANTS.NUM_THRUSTERS
EPSILON: Final[float] = SIMULATION_CONSTANTS.EPSILON
