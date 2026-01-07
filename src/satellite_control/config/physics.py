"""
Physical Parameters for Satellite Control System

    Complete physical model parameters for satellite dynamics and thruster
    configuration.
    Includes mass properties, thruster geometry, and realistic physics
    effects.

Configuration sections:
- Mass Properties: Total mass, moment of inertia, center of mass offset
- Thruster Configuration: Eight-thruster layout with positions and directions
- Thruster Forces: Individual force calibration per thruster
- Realistic Physics: Damping, friction, sensor noise
- Air Bearing System: Three-point support configuration

Thruster layout:
- Eight thrusters arranged around satellite body
- Individual position and direction vectors
- Configurable force magnitude per thruster
- Support for force calibration and testing

Key features:
- Individual thruster force calibration
- Realistic damping and friction modeling
- Sensor noise simulation for testing
- Center of mass calculation from air bearing
- Integration with testing_environment physics
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhysicsConfig:
    """
    Physical properties and optional realism toggles.

    Attributes:
        total_mass: Total satellite mass in kg
        moment_of_inertia: Rotational inertia in kg·m²
        satellite_size: Characteristic dimension in meters
        com_offset: Center of mass offset [x, y] in meters
        thruster_positions: Dict[int, Tuple[float, float, float]]
        # Dict mapping thruster ID (1-12) to (x, y, z) position in meters
        thruster_directions: Dict mapping thruster ID to unit direction vector
        thruster_forces: Dict mapping thruster ID to force magnitude in Newtons
        use_realistic_physics: Enable realistic physics modeling
        linear_damping_coeff: Linear drag coefficient in N/(m/s)
        rotational_damping_coeff: Rotational drag coefficient in N*m/(rad/s)
        position_noise_std: Position measurement noise std dev in meters
        velocity_noise_std: Velocity estimation noise std dev in m/s
        angle_noise_std: Orientation noise std dev in radians
        angular_velocity_noise_std: Angular velocity noise std dev in rad/s
        thruster_valve_delay: Solenoid valve opening delay in seconds
        thruster_rampup_time: Time for thrust to reach full force in seconds
        thruster_force_noise_std: Fractional thrust force variation (std dev)
        enable_random_disturbances: Enable random environmental disturbances
        disturbance_force_std: Random disturbance force std dev in Newtons
        disturbance_torque_std: Random disturbance torque std dev in N*m
    """

    # Core physical properties
    total_mass: float
    moment_of_inertia: float
    satellite_size: float
    com_offset: np.ndarray

    # Thruster configuration
    thruster_positions: Dict[int, Tuple[float, float]]
    thruster_directions: Dict[int, np.ndarray]
    thruster_forces: Dict[int, float]

    # Realistic physics modeling
    use_realistic_physics: bool = True
    linear_damping_coeff: float = 1.8
    rotational_damping_coeff: float = 0.3

    # Sensor noise
    position_noise_std: float = 0.000
    velocity_noise_std: float = 0.000
    angle_noise_std: float = 0.0
    angular_velocity_noise_std: float = 0.0

    # Actuator dynamics
    thruster_valve_delay: float = 0.04
    thruster_rampup_time: float = 0.01
    thruster_force_noise_std: float = 0.00

    # Environmental disturbances
    enable_random_disturbances: bool = True
    disturbance_force_std: float = 0.4
    disturbance_torque_std: float = 0.1


# DEFAULT PHYSICAL PARAMETERS
# ============================================================================

# Mass properties
TOTAL_MASS = 10.0  # kg
SATELLITE_SIZE = 0.29  # m

MOMENT_OF_INERTIA = (1 / 6) * TOTAL_MASS * SATELLITE_SIZE**2

# Thruster configuration
THRUSTER_POSITIONS = {
    1: (0.145, 0.06, 0.0),  # Right-top
    2: (0.145, -0.06, 0.0),  # Right-bottom
    3: (0.06, -0.145, 0.0),  # Bottom-right
    4: (-0.06, -0.145, 0.0),  # Bottom-left
    5: (-0.145, -0.06, 0.0),  # Left-bottom
    6: (-0.145, 0.06, 0.0),  # Left-top
    7: (-0.06, 0.145, 0.0),  # Top-left
    8: (0.06, 0.145, 0.0),  # Top-right
    # New Z-axis thrusters (3D)
    # Top face (+Z) pushing DOWN (-Z). Placed on X-axis for Pitch control.
    9: (0.145, 0.0, 0.145),  # Top-Front (Front in X)
    10: (-0.145, 0.0, 0.145),  # Top-Back (Back in X)
    # Bottom face (-Z) pushing UP (+Z). Placed on Y-axis for Roll control.
    11: (0.0, 0.145, -0.145),  # Bottom-Right (Right in Y)
    12: (0.0, -0.145, -0.145),  # Bottom-Left (Left in Y)
}

THRUSTER_DIRECTIONS = {
    1: np.array([-1, 0, 0]),  # Left (-X)
    2: np.array([-1, 0, 0]),  # Left (-X)
    3: np.array([0, 1, 0]),  # Up (+Y)
    4: np.array([0, 1, 0]),  # Up (+Y)
    5: np.array([1, 0, 0]),  # Right (+X)
    6: np.array([1, 0, 0]),  # Right (+X)
    7: np.array([0, -1, 0]),  # Down (-Y)
    8: np.array([0, -1, 0]),  # Down (-Y)
    # Z-axis thrusters (3D)
    9: np.array([0, 0, -1]),  # Down (-Z)
    10: np.array([0, 0, -1]),  # Down (-Z)
    11: np.array([0, 0, 1]),  # Up (+Z)
    12: np.array([0, 0, 1]),  # Up (+Z)
}

THRUSTER_FORCES = {
    1: 0.441450,
    2: 0.430659,
    3: 0.427716,
    4: 0.438017,
    5: 0.468918,
    6: 0.446846,
    7: 0.466956,
    8: 0.484124,
    # Default 0.45N for new thrusters until calibrated
    9: 0.45,
    10: 0.45,
    11: 0.45,
    12: 0.45,
}

GRAVITY_M_S2 = 9.81  # m/s²


def calculate_com_offset() -> np.ndarray:
    """
    Calculate center of mass offset.
    Hardcoded to (0,0) as per configuration request.
    """
    # Force CoM to be at geometric center
    return np.zeros(3)


COM_OFFSET = calculate_com_offset()


def get_physics_params() -> PhysicsConfig:
    """
    Get default physics configuration.

    Returns:
        PhysicsConfig with default physical parameters
    """
    return PhysicsConfig(
        total_mass=TOTAL_MASS,
        moment_of_inertia=MOMENT_OF_INERTIA,
        satellite_size=SATELLITE_SIZE,
        com_offset=COM_OFFSET.copy(),
        thruster_positions=THRUSTER_POSITIONS.copy(),
        thruster_directions={k: v.copy() for k, v in THRUSTER_DIRECTIONS.items()},
        thruster_forces=THRUSTER_FORCES.copy(),
        use_realistic_physics=False,
        linear_damping_coeff=0.0,
        rotational_damping_coeff=0.0,
        position_noise_std=0.0,
        velocity_noise_std=0.0,
        angle_noise_std=np.deg2rad(0.0),
        angular_velocity_noise_std=np.deg2rad(0.0),
        thruster_valve_delay=0.0,
        thruster_rampup_time=0.0,
        thruster_force_noise_std=0.0,
        enable_random_disturbances=False,
        disturbance_force_std=0.0,
        disturbance_torque_std=0.0,
    )


def set_thruster_force(thruster_id: int, force: float) -> None:
    """
    Set individual thruster force for calibration.

    Args:
        thruster_id: Thruster ID (1-8)
        force: Force magnitude in Newtons

    Raises:
        ValueError: If thruster_id invalid or force non-positive
    """
    if thruster_id not in range(1, 13):
        raise ValueError(f"Thruster ID must be 1-12, got {thruster_id}")
    if force <= 0:
        raise ValueError(f"Force must be positive, got {force}")

    THRUSTER_FORCES[thruster_id] = force
    logger.info(f"Thruster {thruster_id} force set to {force:.3f} N")


def set_all_thruster_forces(force: float) -> None:
    """
    Set all thruster forces to the same value.

    Args:
        force: Force magnitude in Newtons for all thrusters

    Raises:
        ValueError: If force is non-positive
    """
    if force <= 0:
        raise ValueError(f"Force must be positive, got {force}")

    for thruster_id in range(1, 13):
        THRUSTER_FORCES[thruster_id] = force
    logger.info(f"All thruster forces set to {force:.3f} N")


def get_thruster_force(thruster_id: int) -> float:
    """
    Get individual thruster force.

    Args:
        thruster_id: Thruster ID (1-8)

    Returns:
        Force magnitude in Newtons

    Raises:
        ValueError: If thruster_id is invalid
    """
    if thruster_id not in range(1, 13):
        raise ValueError(f"Thruster ID must be 1-12, got {thruster_id}")
    return THRUSTER_FORCES[thruster_id]


def print_thruster_forces() -> None:
    """Print current thruster force configuration."""
    logger.info("CURRENT THRUSTER FORCE CONFIGURATION:")
    for thruster_id in range(1, 13):
        force = THRUSTER_FORCES[thruster_id]
        logger.info(f"  Thruster {thruster_id}: {force:.3f} N")


def validate_physics_params(config: PhysicsConfig) -> bool:
    """
    Validate physical parameters for consistency.

    Args:
        config: PhysicsConfig to validate

    Returns:
        True if valid, False otherwise
    """
    issues = []

    # Mass validation
    if config.total_mass <= 0:
        issues.append(f"Invalid mass: {config.total_mass}")

    # Inertia validation
    if config.moment_of_inertia <= 0:
        issues.append(f"Invalid moment of inertia: {config.moment_of_inertia}")

    # Thruster configuration validation
    if len(config.thruster_positions) != 12:
        issues.append(f"Expected 12 thrusters, got {len(config.thruster_positions)}")

    if len(config.thruster_directions) != 12:
        issues.append(f"Expected 12 thruster directions, got " f"{len(config.thruster_directions)}")

    if len(config.thruster_forces) != 12:
        issues.append(f"Expected 12 thruster forces, got {len(config.thruster_forces)}")

    # Validate thruster force values are positive
    for thruster_id, force in config.thruster_forces.items():
        if force <= 0:
            issues.append(f"Thruster {thruster_id} force must be positive, got {force}")

    # Report validation results
    if issues:
        logger.warning("Physics parameter validation failed:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False

    return True
