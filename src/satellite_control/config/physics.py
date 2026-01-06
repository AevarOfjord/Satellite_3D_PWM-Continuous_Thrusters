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
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhysicsConfig:
    """
    Physical properties and optional realism toggles.

    Supports both 2D (planar) and 3D (6-DOF) configurations.

    Attributes:
        total_mass: Total satellite mass in kg
        moment_of_inertia: Rotational inertia in kg·m² (scalar for 2D)
        inertia_tensor: 3x3 inertia tensor for 3D mode
        satellite_size: Characteristic dimension in meters
        com_offset: Center of mass offset [x, y] or [x, y, z] in meters
        thruster_positions: Dict mapping thruster ID to position tuple
        thruster_directions: Dict mapping thruster ID to unit direction vector
        thruster_forces: Dict mapping thruster ID to force magnitude in Newtons
        use_3d_mode: If True, use 3D (6-DOF) configuration
        use_realistic_physics: Enable realistic physics modeling
        linear_damping_coeff: Linear drag coefficient in N/(m/s)
        rotational_damping_coeff: Rotational drag coefficient in N*m/(rad/s)
    """

    # Core physical properties
    total_mass: float
    moment_of_inertia: float  # Scalar for 2D, Iz for 3D
    satellite_size: float
    com_offset: np.ndarray

    # Thruster configuration (2D or 3D depending on mode)
    thruster_positions: Dict[int, Tuple]
    thruster_directions: Dict[int, np.ndarray]
    thruster_forces: Dict[int, float]

    # 3D-specific fields
    use_3d_mode: bool = False
    inertia_tensor: Optional[np.ndarray] = None  # 3x3 matrix for 3D

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

    def __post_init__(self):
        """Initialize default inertia tensor if not provided."""
        if self.inertia_tensor is None:
            self.inertia_tensor = np.diag(
                [self.moment_of_inertia, self.moment_of_inertia, self.moment_of_inertia]
            )


# DEFAULT PHYSICAL PARAMETERS
# ============================================================================

# Mass properties
TOTAL_MASS = 10.0  # kg
SATELLITE_SIZE = 0.29  # m

MOMENT_OF_INERTIA = (1 / 6) * TOTAL_MASS * SATELLITE_SIZE**2

# Thruster configuration
THRUSTER_POSITIONS = {
    1: (0.145, 0.06),  # Right-top
    2: (0.145, -0.06),  # Right-bottom
    3: (0.06, -0.145),  # Bottom-right
    4: (-0.06, -0.145),  # Bottom-left
    5: (-0.145, -0.06),  # Left-bottom
    6: (-0.145, 0.06),  # Left-top
    7: (-0.06, 0.145),  # Top-left
    8: (0.06, 0.145),  # Top-right
}

THRUSTER_DIRECTIONS = {
    1: np.array([-1, 0]),  # Left
    2: np.array([-1, 0]),  # Left
    3: np.array([0, 1]),  # Up
    4: np.array([0, 1]),  # Up
    5: np.array([1, 0]),  # Right
    6: np.array([1, 0]),  # Right
    7: np.array([0, -1]),  # Down
    8: np.array([0, -1]),  # Down
}

THRUSTER_FORCES = {
    1: 0.441450,  # N - Measured thruster forces
    2: 0.430659,
    3: 0.427716,
    4: 0.438017,
    5: 0.468918,
    6: 0.446846,
    7: 0.466956,
    8: 0.484124,
}

GRAVITY_M_S2 = 9.81  # m/s²

# ============================================================================
# 3D MODE CONFIGURATION
# ============================================================================

USE_3D_MODE = True  # Toggle between 2D (planar) and 3D (6-DOF) control

# 3D Inertia tensor (kg·m²) - assuming uniform cube distribution
# For a cube: I = (1/6) * m * s²  for each axis
_I_scalar = (1 / 6) * TOTAL_MASS * SATELLITE_SIZE**2
MOMENT_OF_INERTIA_TENSOR = np.diag([_I_scalar, _I_scalar, _I_scalar])

# 3D Center of Mass offset (x, y, z)
COM_OFFSET_3D = np.zeros(3)

# 3D Thruster Configuration: 12 thrusters for full 6-DOF control
# Layout: 2 thrusters per direction (+X, -X, +Y, -Y, +Z, -Z)
# Placed at corners to provide both force and torque authority
_s = SATELLITE_SIZE / 2  # Half-size for positioning

THRUSTER_POSITIONS_3D = {
    # +X face thrusters (push -X direction)
    1: (_s, _s * 0.4, 0.0),  # +X face, upper
    2: (_s, -_s * 0.4, 0.0),  # +X face, lower
    # -X face thrusters (push +X direction)
    3: (-_s, _s * 0.4, 0.0),  # -X face, upper
    4: (-_s, -_s * 0.4, 0.0),  # -X face, lower
    # +Y face thrusters (push -Y direction)
    5: (_s * 0.4, _s, 0.0),  # +Y face, right
    6: (-_s * 0.4, _s, 0.0),  # +Y face, left
    # -Y face thrusters (push +Y direction)
    7: (_s * 0.4, -_s, 0.0),  # -Y face, right
    8: (-_s * 0.4, -_s, 0.0),  # -Y face, left
    # +Z face thrusters (push -Z direction)
    9: (_s * 0.4, 0.0, _s),  # +Z face, right
    10: (-_s * 0.4, 0.0, _s),  # +Z face, left
    # -Z face thrusters (push +Z direction)
    11: (_s * 0.4, 0.0, -_s),  # -Z face, right
    12: (-_s * 0.4, 0.0, -_s),  # -Z face, left
}

THRUSTER_DIRECTIONS_3D = {
    # +X face → push -X
    1: np.array([-1.0, 0.0, 0.0]),
    2: np.array([-1.0, 0.0, 0.0]),
    # -X face → push +X
    3: np.array([1.0, 0.0, 0.0]),
    4: np.array([1.0, 0.0, 0.0]),
    # +Y face → push -Y
    5: np.array([0.0, -1.0, 0.0]),
    6: np.array([0.0, -1.0, 0.0]),
    # -Y face → push +Y
    7: np.array([0.0, 1.0, 0.0]),
    8: np.array([0.0, 1.0, 0.0]),
    # +Z face → push -Z
    9: np.array([0.0, 0.0, -1.0]),
    10: np.array([0.0, 0.0, -1.0]),
    # -Z face → push +Z
    11: np.array([0.0, 0.0, 1.0]),
    12: np.array([0.0, 0.0, 1.0]),
}

# Default 3D thruster forces (uniform 0.45N)
THRUSTER_FORCES_3D = {i: 0.45 for i in range(1, 13)}

# Number of thrusters based on mode
NUM_THRUSTERS = 12 if USE_3D_MODE else 8


def calculate_com_offset() -> np.ndarray:
    """
    Calculate center of mass offset.
    Hardcoded to (0,0) as per configuration request.
    """
    # Force CoM to be at geometric center
    return np.zeros(2)


COM_OFFSET = calculate_com_offset()


def get_physics_params() -> PhysicsConfig:
    """
    Get default physics configuration.

    Returns 2D or 3D configuration based on USE_3D_MODE flag.

    Returns:
        PhysicsConfig with default physical parameters
    """
    if USE_3D_MODE:
        return PhysicsConfig(
            total_mass=TOTAL_MASS,
            moment_of_inertia=MOMENT_OF_INERTIA,
            satellite_size=SATELLITE_SIZE,
            com_offset=COM_OFFSET_3D.copy(),
            thruster_positions=THRUSTER_POSITIONS_3D.copy(),
            thruster_directions={k: v.copy() for k, v in THRUSTER_DIRECTIONS_3D.items()},
            thruster_forces=THRUSTER_FORCES_3D.copy(),
            use_3d_mode=True,
            inertia_tensor=MOMENT_OF_INERTIA_TENSOR.copy(),
            use_realistic_physics=False,
            linear_damping_coeff=0.0,
            rotational_damping_coeff=0.0,
            position_noise_std=0.0,
            velocity_noise_std=0.0,
            angle_noise_std=0.0,
            angular_velocity_noise_std=0.0,
            thruster_valve_delay=0.0,
            thruster_rampup_time=0.0,
            thruster_force_noise_std=0.0,
            enable_random_disturbances=False,
            disturbance_force_std=0.0,
            disturbance_torque_std=0.0,
        )
    else:
        return PhysicsConfig(
            total_mass=TOTAL_MASS,
            moment_of_inertia=MOMENT_OF_INERTIA,
            satellite_size=SATELLITE_SIZE,
            com_offset=COM_OFFSET.copy(),
            thruster_positions=THRUSTER_POSITIONS.copy(),
            thruster_directions={k: v.copy() for k, v in THRUSTER_DIRECTIONS.items()},
            thruster_forces=THRUSTER_FORCES.copy(),
            use_3d_mode=False,
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
    if thruster_id not in range(1, 9):
        raise ValueError(f"Thruster ID must be 1-8, got {thruster_id}")
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

    for thruster_id in range(1, 9):
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
    if thruster_id not in range(1, 9):
        raise ValueError(f"Thruster ID must be 1-8, got {thruster_id}")
    return THRUSTER_FORCES[thruster_id]


def print_thruster_forces() -> None:
    """Print current thruster force configuration."""
    logger.info("CURRENT THRUSTER FORCE CONFIGURATION:")
    for thruster_id in range(1, 9):
        force = THRUSTER_FORCES[thruster_id]
        logger.info(f"  Thruster {thruster_id}: {force:.3f} N")


def validate_physics_params(config: PhysicsConfig) -> bool:
    """
    Validate physical parameters for consistency.

    Supports both 2D (8 thrusters) and 3D (12 thrusters) configurations.

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

    # Thruster configuration validation (8 for 2D, 12 for 3D)
    expected_thrusters = 12 if config.use_3d_mode else 8

    if len(config.thruster_positions) != expected_thrusters:
        issues.append(
            f"Expected {expected_thrusters} thrusters, got {len(config.thruster_positions)}"
        )

    if len(config.thruster_directions) != expected_thrusters:
        issues.append(
            f"Expected {expected_thrusters} thruster directions, "
            f"got {len(config.thruster_directions)}"
        )

    if len(config.thruster_forces) != expected_thrusters:
        issues.append(
            f"Expected {expected_thrusters} thruster forces, got {len(config.thruster_forces)}"
        )

    # Validate thruster force values are positive
    for thruster_id, force in config.thruster_forces.items():
        if force <= 0:
            issues.append(f"Thruster {thruster_id} force must be positive, got {force}")

    # 3D-specific validation
    if config.use_3d_mode:
        if config.inertia_tensor is None:
            issues.append("Inertia tensor required for 3D mode")
        elif config.inertia_tensor.shape != (3, 3):
            issues.append(f"Inertia tensor must be 3x3, got {config.inertia_tensor.shape}")

        if config.com_offset.shape != (3,):
            issues.append(f"COM offset must be 3D for 3D mode, got shape {config.com_offset.shape}")

    # Report validation results
    if issues:
        logger.warning("Physics parameter validation failed:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False

    return True
