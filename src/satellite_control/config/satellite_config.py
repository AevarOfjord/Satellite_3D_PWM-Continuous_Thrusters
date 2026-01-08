"""
Central Configuration for Satellite Control System

Provides a unified interface to the modular configuration system.

Import from the config package:
    from config import PhysicsConfig, TimingConfig, MPCConfig, etc.

Or use the SatelliteConfig wrapper:
    from config import SatelliteConfig
    SatelliteConfig.TOTAL_MASS

Individual Thruster Force Calibration:
You can manually adjust the thrust force for each thruster for calibration
and testing purposes:

Example usage:
    from config import SatelliteConfig

    # Set individual thruster force
    SatelliteConfig.set_thruster_force(1, 0.280)  # Thruster 1 to 0.280 N
    SatelliteConfig.set_thruster_force(3, 0.310)  # Thruster 3 to 0.310 N

    # Set all thrusters to same force
    SatelliteConfig.set_all_thruster_forces(0.290)  # All to 0.290 N

    # Check current forces
    SatelliteConfig.print_thruster_forces()

The individual thruster forces are automatically used by all controllers
(MPC, Testing Environment, Dashboard) for consistent behavior.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import from modular config system
from . import (
    constants,
    mission_state,
    mpc_params,
    obstacles,
    physics,
    timing,
)
from .models import AppConfig, MPCParams, SatellitePhysicalParams, SimulationParams


# Initialize global configuration instance
def _create_default_config() -> AppConfig:
    """Create default configuration from legacy modules."""

    # Physics
    phys = SatellitePhysicalParams(
        total_mass=physics.TOTAL_MASS,
        moment_of_inertia=physics.MOMENT_OF_INERTIA,
        satellite_size=physics.SATELLITE_SIZE,
        com_offset=tuple(physics.COM_OFFSET),
        thruster_positions=physics.THRUSTER_POSITIONS,
        thruster_directions={k: tuple(v) for k, v in physics.THRUSTER_DIRECTIONS.items()},
        thruster_forces=physics.THRUSTER_FORCES,
        use_realistic_physics=False,
        damping_linear=0.0,
        damping_angular=0.0,
    )

    # MPC
    mpc = MPCParams(
        prediction_horizon=mpc_params.MPC_PREDICTION_HORIZON,
        control_horizon=mpc_params.MPC_CONTROL_HORIZON,
        dt=timing.CONTROL_DT,
        solver_time_limit=mpc_params.MPC_SOLVER_TIME_LIMIT,
        solver_type=mpc_params.MPC_SOLVER_TYPE,
        q_position=mpc_params.Q_POSITION,
        q_velocity=mpc_params.Q_VELOCITY,
        q_angle=mpc_params.Q_ANGLE,
        q_angular_velocity=mpc_params.Q_ANGULAR_VELOCITY,
        r_thrust=mpc_params.R_THRUST,
        max_velocity=mpc_params.MAX_VELOCITY,
        max_angular_velocity=mpc_params.MAX_ANGULAR_VELOCITY,
        position_bounds=mpc_params.POSITION_BOUNDS,
        damping_zone=mpc_params.DAMPING_ZONE,
        velocity_threshold=mpc_params.VELOCITY_THRESHOLD,
        max_velocity_weight=mpc_params.MAX_VELOCITY_WEIGHT,
        thruster_type=mpc_params.THRUSTER_TYPE,
    )

    # Simulation
    sim = SimulationParams(
        dt=0.005,
        max_duration=timing.MAX_SIMULATION_TIME,
        headless=constants.Constants.HEADLESS_MODE,
        window_width=constants.Constants.WINDOW_WIDTH,
        window_height=constants.Constants.WINDOW_HEIGHT,
    )

    return AppConfig(physics=phys, mpc=mpc, simulation=sim, input_file_path=None)


class StructuredConfig:
    """
    Structured configuration class for testing and advanced usage.

    This class provides a structured way to manage configuration with
    the ability to clone and modify configurations.
    """

    def __init__(self, config_dict=None):
        """Initialize structured config from dictionary."""
        self.config_dict = config_dict or {}

    def clone(self):
        """Create a deep copy of this configuration."""
        import copy

        return StructuredConfig(copy.deepcopy(self.config_dict))

    def get(self, key, default=None):
        """Get a configuration value."""
        return self.config_dict.get(key, default)

    def set(self, key, value):
        """Set a configuration value."""
        self.config_dict[key] = value


@contextmanager
def use_structured_config(config):
    """
    Context manager for temporarily using a structured configuration.

    Args:
        config: StructuredConfig instance to use within context

    Yields:
        The provided config
    """
    # This is a simple implementation that just yields the config
    # In a more complex system, this might swap out global configuration
    try:
        yield config
    finally:
        pass


class SatelliteConfig:
    """
    Backward-compatible facade for satellite configuration.

    This class maintains the same interface as the original SatelliteConfig
    while delegating to the new modular configuration system.
    """

    # Global Pydantic Configuration
    _config: AppConfig = _create_default_config()

    @classmethod
    def get_app_config(cls) -> AppConfig:
        """Get the Pydantic configuration object."""
        return cls._config

    # ========================================================================
    # TIMING PARAMETERS
    # ========================================================================

    SIMULATION_DT = 0.005  # 5ms for finer PWM resolution
    CONTROL_DT = timing.CONTROL_DT
    MAX_SIMULATION_TIME = timing.MAX_SIMULATION_TIME
    TARGET_HOLD_TIME = timing.TARGET_HOLD_TIME
    USE_FINAL_STABILIZATION_IN_SIMULATION = timing.USE_FINAL_STABILIZATION_IN_SIMULATION
    WAYPOINT_FINAL_STABILIZATION_TIME = timing.WAYPOINT_FINAL_STABILIZATION_TIME
    SHAPE_FINAL_STABILIZATION_TIME = timing.SHAPE_FINAL_STABILIZATION_TIME
    SHAPE_POSITIONING_STABILIZATION_TIME = timing.SHAPE_POSITIONING_STABILIZATION_TIME

    # ========================================================================
    # MPC PARAMETERS
    # ========================================================================

    MPC_PREDICTION_HORIZON = mpc_params.MPC_PREDICTION_HORIZON
    MPC_CONTROL_HORIZON = mpc_params.MPC_CONTROL_HORIZON
    MPC_SOLVER_TIME_LIMIT = mpc_params.MPC_SOLVER_TIME_LIMIT
    MPC_SOLVER_TYPE = mpc_params.MPC_SOLVER_TYPE

    VERBOSE_MPC = mpc_params.VERBOSE_MPC

    Q_POSITION = mpc_params.Q_POSITION
    Q_VELOCITY = mpc_params.Q_VELOCITY
    Q_ANGLE = mpc_params.Q_ANGLE
    Q_ANGULAR_VELOCITY = mpc_params.Q_ANGULAR_VELOCITY
    R_THRUST = mpc_params.R_THRUST

    MAX_VELOCITY = mpc_params.MAX_VELOCITY
    MAX_ANGULAR_VELOCITY = mpc_params.MAX_ANGULAR_VELOCITY
    POSITION_BOUNDS = mpc_params.POSITION_BOUNDS
    ANGLE_BOUNDS = mpc_params.ANGLE_BOUNDS
    DAMPING_ZONE = mpc_params.DAMPING_ZONE
    VELOCITY_THRESHOLD = mpc_params.VELOCITY_THRESHOLD
    MAX_VELOCITY_WEIGHT = mpc_params.MAX_VELOCITY_WEIGHT

    POSITION_TOLERANCE = mpc_params.POSITION_TOLERANCE
    ANGLE_TOLERANCE = mpc_params.ANGLE_TOLERANCE
    VELOCITY_TOLERANCE = mpc_params.VELOCITY_TOLERANCE
    ANGULAR_VELOCITY_TOLERANCE = mpc_params.ANGULAR_VELOCITY_TOLERANCE

    THRUSTER_TYPE = mpc_params.THRUSTER_TYPE

    # ========================================================================
    # PHYSICAL PARAMETERS
    # ========================================================================

    TOTAL_MASS = physics.TOTAL_MASS
    SATELLITE_SIZE = physics.SATELLITE_SIZE
    MOMENT_OF_INERTIA = physics.MOMENT_OF_INERTIA
    COM_OFFSET = physics.COM_OFFSET
    GRAVITY_M_S2 = physics.GRAVITY_M_S2

    THRUSTER_POSITIONS = physics.THRUSTER_POSITIONS
    THRUSTER_DIRECTIONS = physics.THRUSTER_DIRECTIONS
    THRUSTER_FORCES = physics.THRUSTER_FORCES

    # Realistic physics parameters
    USE_REALISTIC_PHYSICS = False
    LINEAR_DAMPING_COEFF = 0.0
    ROTATIONAL_DAMPING_COEFF = 0.0
    DAMPING_LINEAR = LINEAR_DAMPING_COEFF
    DAMPING_ANGULAR = ROTATIONAL_DAMPING_COEFF
    POSITION_NOISE_STD = 0.0
    VELOCITY_NOISE_STD = 0.0
    ANGLE_NOISE_STD = 0.0
    ANGULAR_VELOCITY_NOISE_STD = 0.0
    THRUSTER_VALVE_DELAY = 0.0
    THRUSTER_RAMPUP_TIME = 0.0
    THRUSTER_FORCE_NOISE_STD = 0.0
    ENABLE_RANDOM_DISTURBANCES = False
    DISTURBANCE_FORCE_STD = 0.0
    DISTURBANCE_TORQUE_STD = 0.0
    STATE_ESTIMATION_DELAY = 0.000
    CONTROL_COMPUTATION_DELAY = 0.000

    # ========================================================================
    # CONSTANTS
    # ========================================================================

    WINDOW_WIDTH = constants.Constants.WINDOW_WIDTH
    WINDOW_HEIGHT = constants.Constants.WINDOW_HEIGHT
    HEADLESS_MODE = constants.Constants.HEADLESS_MODE
    DATA_DIR = constants.Constants.DATA_DIR
    SUBPLOT_CONFIG = constants.Constants.SUBPLOT_CONFIG
    OVERLAY_HEIGHT = constants.Constants.OVERLAY_HEIGHT
    ARROW_X_OFFSET = constants.Constants.ARROW_X_OFFSET
    ARROW_Y_OFFSET = constants.Constants.ARROW_Y_OFFSET
    ARROW_WIDTH = constants.Constants.ARROW_WIDTH
    SLEEP_TARGET_DT = constants.Constants.SLEEP_TARGET_DT
    HEADLESS_MODE = constants.Constants.HEADLESS_MODE

    DEG_PER_CIRCLE = constants.Constants.DEG_PER_CIRCLE
    RAD_TO_DEG = constants.Constants.RAD_TO_DEG
    DEG_TO_RAD = constants.Constants.DEG_TO_RAD

    DATA_DIR = constants.Constants.DATA_DIR
    LINEARIZED_DATA_DIR = constants.Constants.LINEARIZED_DATA_DIR
    THRUSTER_DATA_DIR = constants.Constants.THRUSTER_DATA_DIR
    CSV_TIMESTAMP_FORMAT = constants.Constants.CSV_TIMESTAMP_FORMAT

    FFMPEG_PATH = constants.Constants.FFMPEG_PATH
    FFMPEG_PATH_WINDOWS = constants.Constants.FFMPEG_PATH_WINDOWS
    FFMPEG_PATH_MACOS = constants.Constants.FFMPEG_PATH_MACOS
    FFMPEG_PATH_LINUX = constants.Constants.FFMPEG_PATH_LINUX

    DEFAULT_START_POS = constants.Constants.DEFAULT_START_POS
    DEFAULT_TARGET_POS = constants.Constants.DEFAULT_TARGET_POS
    DEFAULT_START_ANGLE = constants.Constants.DEFAULT_START_ANGLE
    DEFAULT_TARGET_ANGLE = constants.Constants.DEFAULT_TARGET_ANGLE

    # ========================================================================
    # MISSION STATE (Mutable runtime state)
    # ========================================================================

    # Global mission state instance
    _mission_state = mission_state.create_mission_state()

    # ========================================================================
    # MISSION 1: WAYPOINT NAVIGATION (supports single or multiple waypoints)
    # ========================================================================
    ENABLE_WAYPOINT_MODE = False
    WAYPOINT_TARGETS: List[Tuple[float, float]] = []
    WAYPOINT_ANGLES: List[Tuple[float, float, float]] = []
    CURRENT_TARGET_INDEX = 0
    MULTI_POINT_PHASE: Optional[str] = None
    TARGET_STABILIZATION_START_TIME = None
    WAYPOINT_PHASE = None

    # ========================================================================
    # MISSION 2: SHAPE FOLLOWING (circles, rectangles, triangles, hexagons,
    # DXF)
    # ========================================================================
    # Note: Shape following uses DXF_SHAPE_MODE_ACTIVE and related fields below
    # DXF shape mode
    DXF_SHAPE_MODE_ACTIVE = False
    DXF_SHAPE_CENTER = None
    DXF_SHAPE_PATH: List[Tuple[float, float]] = []
    DXF_BASE_SHAPE: List[Tuple[float, float]] = []
    DXF_MISSION_START_TIME = None
    DXF_SHAPE_PHASE = "POSITIONING"
    DXF_PATH_LENGTH = 0.0
    DXF_CLOSEST_POINT_INDEX = 0
    DXF_CURRENT_TARGET_POSITION = None
    DXF_POSITIONING_START_TIME = None  # Track time in POSITIONING phase
    DXF_PATH_STABILIZATION_START_TIME = None  # Track time in PATH_STABILIZATION phase
    DXF_TRACKING_START_TIME = None
    DXF_TARGET_START_DISTANCE = 0.0
    DXF_STABILIZATION_START_TIME = None
    DXF_FINAL_POSITION = None
    DXF_SHAPE_ROTATION = 0.0
    DXF_HAS_RETURN = False
    DXF_RETURN_POSITION = None
    DXF_RETURN_ANGLE = None
    DXF_RETURN_START_TIME = None
    DXF_TARGET_SPEED = 0.1
    DXF_ESTIMATED_DURATION = 60.0
    DXF_OFFSET_DISTANCE = 0.5

    # ========================================================================
    # OBSTACLE AVOIDANCE
    # ========================================================================

    _obstacle_manager = obstacles.create_obstacle_manager()

    OBSTACLES_ENABLED = False
    OBSTACLES: List[Tuple[float, float, float]] = []
    DEFAULT_OBSTACLE_RADIUS = obstacles.DEFAULT_OBSTACLE_RADIUS
    OBSTACLE_SAFETY_MARGIN = obstacles.OBSTACLE_SAFETY_MARGIN
    MIN_OBSTACLE_DISTANCE = obstacles.MIN_OBSTACLE_DISTANCE
    OBSTACLE_PATH_RESOLUTION = obstacles.OBSTACLE_PATH_RESOLUTION
    OBSTACLE_WAYPOINT_STABILIZATION_TIME = obstacles.OBSTACLE_WAYPOINT_STABILIZATION_TIME
    OBSTACLE_FLYTHROUGH_TOLERANCE = obstacles.OBSTACLE_FLYTHROUGH_TOLERANCE
    OBSTACLE_AVOIDANCE_SAFETY_MARGIN = obstacles.OBSTACLE_AVOIDANCE_SAFETY_MARGIN

    # ========================================================================
    # PARAMETER VALIDATION
    # ========================================================================

    @classmethod
    def validate_parameters(cls) -> bool:
        """
        Validate that all parameters are consistent and safe.

        Returns:
            bool: True if parameters are valid, False otherwise
        """
        issues = []

        # Timing validation
        timing_config = timing.get_timing_params()
        if not timing.validate_timing_params(timing_config):
            issues.append("Timing parameters validation failed")

        # MPC validation
        mpc_config = mpc_params.get_mpc_params()
        if not mpc_params.validate_mpc_params(mpc_config, cls.CONTROL_DT):
            issues.append("MPC parameters validation failed")

        # Physics validation
        physics_config = physics.get_physics_params()
        if not physics.validate_physics_params(physics_config):
            issues.append("Physics parameters validation failed")

        # Report validation results
        if issues:
            print("Parameter validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("Parameter validation passed")
            return True

    # ========================================================================
    # PARAMETER FACTORY METHODS
    # ========================================================================

    @classmethod
    def get_satellite_params(cls) -> dict:
        """
        Get satellite physical parameters as dictionary.

        Returns:
            dict: Satellite parameters for controller initialization
        """
        return {
            "mass": cls.TOTAL_MASS,
            "inertia": cls.MOMENT_OF_INERTIA,
            "size": cls.SATELLITE_SIZE,  # Add 'size' alias for tests
            "satellite_size": cls.SATELLITE_SIZE,
            "com_offset": cls.COM_OFFSET.copy(),
            "thruster_positions": cls.THRUSTER_POSITIONS.copy(),
            "thruster_directions": cls.THRUSTER_DIRECTIONS.copy(),
            "thruster_forces": cls.THRUSTER_FORCES.copy(),
            "damping_linear": cls.DAMPING_LINEAR,
            "damping_angular": cls.DAMPING_ANGULAR,
        }

    @classmethod
    def get_mpc_params(cls) -> dict:
        """
        Get MPC controller parameters as dictionary.

        Returns:
            dict: MPC parameters for controller initialization
        """
        return {
            "prediction_horizon": cls.MPC_PREDICTION_HORIZON,
            "control_horizon": cls.MPC_CONTROL_HORIZON,
            "dt": cls.CONTROL_DT,
            "solver_time_limit": cls.MPC_SOLVER_TIME_LIMIT,
            "solver_type": cls.MPC_SOLVER_TYPE,
            "Q_pos": cls.Q_POSITION,
            "Q_vel": cls.Q_VELOCITY,
            "Q_ang": cls.Q_ANGLE,
            "Q_angvel": cls.Q_ANGULAR_VELOCITY,
            "R_thrust": cls.R_THRUST,
            "q_position": cls.Q_POSITION,
            "q_velocity": cls.Q_VELOCITY,
            "q_angle": cls.Q_ANGLE,
            "q_angular_velocity": cls.Q_ANGULAR_VELOCITY,
            "r_thrust": cls.R_THRUST,
            "max_velocity": cls.MAX_VELOCITY,
            "max_angular_velocity": cls.MAX_ANGULAR_VELOCITY,
            "position_bounds": cls.POSITION_BOUNDS,
            "damping_zone": cls.DAMPING_ZONE,
            "velocity_threshold": cls.VELOCITY_THRESHOLD,
            "max_velocity_weight": cls.MAX_VELOCITY_WEIGHT,
            "thruster_type": cls.THRUSTER_TYPE,
        }

    @classmethod
    def get_simulation_params(cls) -> dict:
        """Get simulation-specific parameters."""
        return constants.Constants.get_simulation_params()

    # ========================================================================
    # THRUSTER FORCE MANAGEMENT
    # ========================================================================

    @classmethod
    def set_thruster_force(cls, thruster_id: int, force: float):
        """Set individual thruster force for calibration."""
        physics.set_thruster_force(thruster_id, force)
        cls.THRUSTER_FORCES = physics.THRUSTER_FORCES

    @classmethod
    def set_all_thruster_forces(cls, force: float):
        """Set all thruster forces to the same value."""
        physics.set_all_thruster_forces(force)
        cls.THRUSTER_FORCES = physics.THRUSTER_FORCES

    @classmethod
    def get_thruster_force(cls, thruster_id: int) -> float:
        """Get individual thruster force."""
        return physics.get_thruster_force(thruster_id)

    @classmethod
    def print_thruster_forces(cls):
        """Print current thruster force configuration."""
        physics.print_thruster_forces()

    # ========================================================================
    # MISSION CONFIGURATION METHODS
    # ========================================================================

    @classmethod
    def set_waypoint_mode(cls, enable: bool):
        """Enable or disable waypoint navigation mode (single or multiple)."""
        cls.ENABLE_WAYPOINT_MODE = enable
        if enable:
            print("Waypoint navigation mode enabled")
        else:
            print("Waypoint navigation mode disabled")

    @classmethod
    def set_multi_point_mode(cls, enable: bool):
        """Backward-compatible alias for enabling waypoint mode."""
        cls.set_waypoint_mode(enable)

    @staticmethod
    def _format_euler_deg(angle: Tuple[float, float, float]) -> str:
        roll, pitch, yaw = np.degrees(angle)
        return f"roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°"

    @classmethod
    def set_waypoint_targets(cls, target_points: list, target_angles: list):
        """Set waypoint target points and orientations."""
        if len(target_points) != len(target_angles):
            raise ValueError("Number of target points and angles must match")

        cls.WAYPOINT_TARGETS = target_points.copy()
        cls.WAYPOINT_ANGLES = target_angles.copy()
        cls.CURRENT_TARGET_INDEX = 0
        cls.TARGET_STABILIZATION_START_TIME = None

        num_targets = len(target_points)
        target_word = "target" if num_targets == 1 else "targets"
        print(f"Waypoint mission configured: {num_targets} {target_word}")
        for i, (pos, angle) in enumerate(zip(target_points, target_angles)):
            print(
                f"  Waypoint {i + 1}: ({pos[0]:.2f}, {pos[1]:.2f}) m, "
                f"{cls._format_euler_deg(angle)}"
            )

    @classmethod
    def get_current_waypoint_target(cls) -> tuple:
        """Get current waypoint target position and angle."""
        if not cls.WAYPOINT_TARGETS:
            return None, None

        if cls.CURRENT_TARGET_INDEX >= len(cls.WAYPOINT_TARGETS):
            return None, None

        target_pos = cls.WAYPOINT_TARGETS[cls.CURRENT_TARGET_INDEX]
        target_angle = cls.WAYPOINT_ANGLES[cls.CURRENT_TARGET_INDEX]

        return target_pos, target_angle

    @classmethod
    def advance_to_next_target(cls) -> bool:
        """Advance to next waypoint target in sequence."""
        if not cls.WAYPOINT_TARGETS:
            return False

        cls.CURRENT_TARGET_INDEX += 1
        cls.TARGET_STABILIZATION_START_TIME = None

        if cls.CURRENT_TARGET_INDEX >= len(cls.WAYPOINT_TARGETS):
            print("ALL WAYPOINTS COMPLETED! Mission successful.")
            return False
        else:
            target_pos, target_angle = cls.get_current_waypoint_target()
            print(
                f"ADVANCING TO WAYPOINT {cls.CURRENT_TARGET_INDEX + 1}: "
                f"({target_pos[0]:.2f}, {target_pos[1]:.2f}) m, "
                f"{cls._format_euler_deg(target_angle)}"
            )
            return True

    @classmethod
    def is_last_target(cls):
        """Check if current target is the last waypoint."""
        return cls.CURRENT_TARGET_INDEX >= len(cls.WAYPOINT_TARGETS) - 1

    @classmethod
    def set_final_stabilization_times(
        cls,
        waypoint: Optional[float] = None,
        shape: Optional[float] = None,
        use_in_simulation: Optional[bool] = None,
    ):
        """Set final stabilization times for mission types.

        Args:
            waypoint: Stabilization time for waypoint missions
            shape: Stabilization time for shape following missions
            use_in_simulation: Enable/disable stabilization in simulation mode
        """
        if waypoint is not None:
            cls.WAYPOINT_FINAL_STABILIZATION_TIME = waypoint
            print(f"Waypoint final stabilization time set to {waypoint:.1f} s")

        if shape is not None:
            cls.SHAPE_FINAL_STABILIZATION_TIME = shape
            print("Shape following final stabilization time set to " f"{shape:.1f} s")

        if use_in_simulation is not None:
            cls.USE_FINAL_STABILIZATION_IN_SIMULATION = use_in_simulation
            if use_in_simulation:
                print("Final stabilization times ENABLED in simulation mode")
            else:
                print(
                    "Final stabilization times DISABLED in simulation mode "
                    "(immediate termination)"
                )

    @classmethod
    def print_stabilization_times(cls):
        """Print current stabilization time configuration."""
        timing_config = timing.get_timing_params()
        timing.print_stabilization_times(timing_config)

    # ========================================================================
    # OBSTACLE AVOIDANCE METHODS
    # ========================================================================

    @classmethod
    def set_obstacles(cls, obstacles_list: list):
        """Set obstacles for all navigation modes."""
        cls._obstacle_manager.set_obstacles(obstacles_list)
        cls.OBSTACLES = cls._obstacle_manager.get_obstacles()
        cls.OBSTACLES_ENABLED = cls._obstacle_manager.enabled

    @classmethod
    def add_obstacle(cls, x: float, y: float, radius: Optional[float] = None):
        """Add a single obstacle."""
        cls._obstacle_manager.add_obstacle(x, y, radius)
        cls.OBSTACLES = cls._obstacle_manager.get_obstacles()
        cls.OBSTACLES_ENABLED = cls._obstacle_manager.enabled

    @classmethod
    def clear_obstacles(cls):
        """Clear all obstacles and disable obstacle avoidance."""
        cls._obstacle_manager.clear_obstacles()
        cls.OBSTACLES = []
        cls.OBSTACLES_ENABLED = False

    @classmethod
    def get_obstacles(cls) -> list:
        """Get current obstacle configuration."""
        return cls._obstacle_manager.get_obstacles()

    @classmethod
    def is_path_clear(
        cls,
        start_pos: tuple,
        end_pos: tuple,
        safety_margin: Optional[float] = None,
    ) -> bool:
        """Check if straight path between two points is clear of obstacles."""
        return cls._obstacle_manager.is_path_clear(start_pos, end_pos, safety_margin)

    @staticmethod
    def _point_to_line_distance(
        point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
    ) -> float:
        """Calculate the shortest distance from a point to a line segment."""
        return obstacles.ObstacleManager._point_to_line_distance(
            obstacles.ObstacleManager(), point, line_start, line_end
        )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    @classmethod
    def print_configuration(cls):
        """Print comprehensive configuration for debugging and verification."""
        print("=" * 80)
        print("SATELLITE CONTROL SYSTEM - COMPLETE CONFIGURATION")
        print("=" * 80)

        # Timing Parameters
        print("\nTIMING PARAMETERS:")
        print(f"   Simulation DT:          {cls.SIMULATION_DT:.3f} s")
        print(f"   Control DT:             {cls.CONTROL_DT:.3f} s")
        print(f"   Max simulation time:    {cls.MAX_SIMULATION_TIME:.1f} s")

        # Physical Parameters
        print("\nPHYSICAL PARAMETERS:")
        print(f"   Total mass:             {cls.TOTAL_MASS:.3f} kg")
        print(f"   Moment of inertia:      {cls.MOMENT_OF_INERTIA:.3f} kg·m²")
        print(f"   Satellite size:         {cls.SATELLITE_SIZE:.3f} m")
        print(
            f"   COM offset:             ({cls.COM_OFFSET[0]:.6f}, "
            f"{cls.COM_OFFSET[1]:.6f}, {cls.COM_OFFSET[2]:.6f}) m"
        )

        # Thruster Configuration
        print("\nTHRUSTER CONFIGURATION:")
        thrust_values = list(cls.THRUSTER_FORCES.values())
        thrust_range = f"{min(thrust_values):.3f}-{max(thrust_values):.3f}"
        print(f"   Force range:            {thrust_range} N")
        for thruster_id, pos in cls.THRUSTER_POSITIONS.items():
            direction = cls.THRUSTER_DIRECTIONS[thruster_id]
            force = cls.THRUSTER_FORCES[thruster_id]
            print(
                f"   T{thruster_id}: ({pos[0]:+.3f}, {pos[1]:+.3f}) m, "
                f"dir: ({direction[0]:+.0f}, {direction[1]:+.0f}), "
                f"force: {force:.3f} N"
            )

        # MPC Controller Parameters
        print("\nMPC CONTROLLER PARAMETERS:")
        print(f"   Prediction horizon:     {cls.MPC_PREDICTION_HORIZON} steps")
        print(f"   Control horizon:        {cls.MPC_CONTROL_HORIZON} steps")
        print(f"   Solver time limit:      {cls.MPC_SOLVER_TIME_LIMIT:.3f} s")
        print(f"   Solver type:            {cls.MPC_SOLVER_TYPE}")

        # Cost Function Weights
        print("\nCOST FUNCTION WEIGHTS:")
        print(f"   Position weight (Q):    {cls.Q_POSITION:.1f}")
        print(f"   Velocity weight (Q):    {cls.Q_VELOCITY:.1f}")
        print(f"   Angle weight (Q):       {cls.Q_ANGLE:.1f}")
        print(f"   Angular vel weight (Q): {cls.Q_ANGULAR_VELOCITY:.1f}")
        print(f"   Thrust penalty (R):     {cls.R_THRUST:.3f}")

        # System Constraints
        print("\nSYSTEM CONSTRAINTS:")
        print(f"   Max velocity:           {cls.MAX_VELOCITY:.2f} m/s")
        print(f"   Max angular velocity:   {cls.MAX_ANGULAR_VELOCITY:.2f} rad/s")
        print(f"   Position bounds:        ±{cls.POSITION_BOUNDS:.1f} m")

        print("=" * 80)

    @staticmethod
    def _calculate_com_offset():
        """Internal method to calculate COM offset."""
        return physics.calculate_com_offset()

    @classmethod
    def reset_mission_state(cls) -> None:
        """
        Reset all mutable mission state to defaults.

        Call this between simulations or tests to ensure clean state.
        This addresses the mutable class attribute anti-pattern.
        """
        # Waypoint navigation state
        cls.ENABLE_WAYPOINT_MODE = False
        cls.WAYPOINT_TARGETS = []
        cls.WAYPOINT_ANGLES = []
        cls.CURRENT_TARGET_INDEX = 0
        cls.TARGET_STABILIZATION_START_TIME = None
        cls.WAYPOINT_PHASE = None

        # DXF shape following state
        cls.DXF_SHAPE_MODE_ACTIVE = False
        cls.DXF_SHAPE_CENTER = None
        cls.DXF_SHAPE_PATH = []
        cls.DXF_BASE_SHAPE = []
        cls.DXF_MISSION_START_TIME = None
        cls.DXF_SHAPE_PHASE = "POSITIONING"
        cls.DXF_PATH_LENGTH = 0.0
        cls.DXF_CLOSEST_POINT_INDEX = 0
        cls.DXF_CURRENT_TARGET_POSITION = None
        cls.DXF_POSITIONING_START_TIME = None
        cls.DXF_PATH_STABILIZATION_START_TIME = None
        cls.DXF_TRACKING_START_TIME = None
        cls.DXF_TARGET_START_DISTANCE = 0.0
        cls.DXF_STABILIZATION_START_TIME = None
        cls.DXF_FINAL_POSITION = None
        cls.DXF_SHAPE_ROTATION = 0.0
        cls.DXF_HAS_RETURN = False
        cls.DXF_RETURN_POSITION = None
        cls.DXF_RETURN_ANGLE = None
        cls.DXF_RETURN_START_TIME = None
        cls.DXF_TARGET_SPEED = 0.1
        cls.DXF_ESTIMATED_DURATION = 60.0
        cls.DXF_OFFSET_DISTANCE = 0.5

        # Obstacle state
        cls._obstacle_manager = obstacles.create_obstacle_manager()
        cls.OBSTACLES_ENABLED = False
        cls.OBSTACLES = []

        # Reset mission state manager
        cls._mission_state = mission_state.create_mission_state()

    @classmethod
    def reset(cls) -> None:
        """
        Full configuration reset including mission state.

        Convenience method that calls reset_mission_state().
        """
        cls.reset_mission_state()


# MODULE INITIALIZATION
def initialize_config():
    """
    Initialize and validate configuration.
    Call this at module import to ensure parameters are valid.
    """
    if not SatelliteConfig.validate_parameters():
        raise ValueError("Configuration validation failed! Check parameters in config.py")


def build_structured_config(overrides=None):
    """
    Build a structured configuration with optional overrides for testing.

    Args:
        overrides: Dictionary of configuration overrides organized by section

    Returns:
        StructuredConfig object with configuration sections
    """
    config: Dict[str, Dict[str, Any]] = {
        "timing": {},
        "mpc": {},
        "physics": {},
        "mission": {},
    }

    if overrides:
        for section, values in overrides.items():
            if section in config:
                config[section].update(values)

    return StructuredConfig(config)


# Run validation on import
if __name__ != "__main__":
    initialize_config()


if __name__ == "__main__":
    print("SATELLITE CONFIGURATION ANALYZER")
    print("=" * 50)
    SatelliteConfig.print_configuration()
