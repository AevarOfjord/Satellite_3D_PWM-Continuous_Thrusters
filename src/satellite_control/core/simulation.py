"""
Linearized MPC Simulation for Satellite Thruster Control

Physics-based simulation environment for testing MPC control
algorithms.
Implements realistic satellite dynamics with thruster actuation
and disturbances.

Simulation features:
- Linearized dynamics with A, B matrices around equilibrium
- Eight-thruster configuration with individual force calibration
- Collision avoidance with circular obstacles
- Mission execution (waypoint, shape following)
- Sensor noise and disturbance simulation
- Real-time visualization with matplotlib

Physics modeling:
- 2D planar motion (x, y, θ) with velocities (vx, vy, ω)
- Thruster force and torque calculations
- Moment of inertia and mass properties
- Integration with configurable time steps

Data collection:
- Complete state history logging
- Control input recording
- MPC solve time statistics
- Mission performance metrics
- CSV export for analysis

Configuration:
- Modular config package for all parameters
- Structured config system for clean access
- Consistent with real hardware configuration
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.satellite_control.config import (
    SatelliteConfig,
    StructuredConfig,
    build_structured_config,
    use_structured_config,
)

from src.satellite_control.config.constants import Constants
from src.satellite_control.config.satellite_config import (
    initialize_config,
)
from src.satellite_control.control.mpc_controller import MPCController
from src.satellite_control.core.mujoco_satellite import SatelliteThrusterTester
from src.satellite_control.core.simulation_io import SimulationIO
from src.satellite_control.core.thruster_manager import ThrusterManager
from src.satellite_control.mission.mission_report_generator import (
    create_mission_report_generator,
)
from src.satellite_control.mission.mission_state_manager import (
    MissionStateManager,
)
from src.satellite_control.utils.data_logger import create_data_logger
from src.satellite_control.utils.logging_config import setup_logging

from src.satellite_control.utils.navigation_utils import (
    angle_difference,
    normalize_angle,
    point_to_line_distance,
)
from src.satellite_control.utils.orientation_utils import (
    euler_xyz_to_quat_wxyz,
    quat_angle_error,
)
from src.satellite_control.utils.simulation_state_validator import (
    create_state_validator_from_config,
)
from src.satellite_control.visualization.simulation_visualization import (
    create_simulation_visualizer,
)

initialize_config()

# Set up logger with simple format for clean output
logger = setup_logging(
    __name__, log_file=f"{Constants.DATA_DIR}/simulation.log", simple_format=True
)


# Use centralized FFMPEG path from Constants (handles all platforms)
plt.rcParams["animation.ffmpeg_path"] = Constants.FFMPEG_PATH

try:
    from src.satellite_control.mission.mission_manager import MissionManager
    from src.satellite_control.visualization.unified_visualizer import (
        UnifiedVisualizationGenerator,
    )
except ImportError:
    logger.warning(
        "WARNING: Could not import visualization or mission components. "
        "Limited functionality available."
    )
    UnifiedVisualizationGenerator = None  # type: ignore
    MissionManager = None  # type: ignore


class SatelliteMPCLinearizedSimulation:
    """
    Simulation environment for linearized MPC satellite control.

    Combines physics from TestingEnvironment with linearized MPC controller
    for satellite navigation using linearized dynamics.
    """

    def __init__(
        self,
        start_pos: Optional[Tuple[float, float]] = None,
        target_pos: Optional[Tuple[float, ...]] = None,
        start_angle: Optional[Tuple[float, float, float]] = None,
        target_angle: Optional[Tuple[float, float, float]] = None,
        start_vx: float = 0.0,
        start_vy: float = 0.0,
        start_vz: float = 0.0,
        start_omega: float = 0.0,
        config: Optional[StructuredConfig] = None,
        config_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        use_mujoco_viewer: bool = True,
    ):
        """
        Initialize linearized MPC simulation.

        Args:
            start_pos: Starting position coords (uses Config default if None)
            target_pos: Target position coords (uses Config default if None)
            start_angle: Starting orientation in radians (roll, pitch, yaw)
            target_angle: Target orientation in radians (roll, pitch, yaw)
            start_vx: Initial X velocity in m/s (default: 0.0)
            start_vy: Initial Y velocity in m/s (default: 0.0)
            start_vz: Initial Z velocity in m/s (default: 0.0)
            start_omega: Initial angular velocity (Z-spin) in rad/s (default: 0.0)
            config: Optional structured config snapshot to run against
            config_overrides: Nested override dict for build_structured_config
            use_mujoco_viewer: If True, use MuJoCo viewer (default: True)
        """
        self.use_mujoco_viewer = use_mujoco_viewer
        self.structured_config = (
            config.clone() if config else build_structured_config(config_overrides)
        )
        with use_structured_config(self.structured_config.clone()):
            self._initialize_from_active_config(
                start_pos,
                target_pos,
                start_angle,
                target_angle,
                start_vx,
                start_vy,
                start_vz,
                start_omega,
            )

    def _initialize_from_active_config(
        self,
        start_pos: Optional[Tuple[float, ...]],
        target_pos: Optional[Tuple[float, ...]],
        start_angle: Optional[Tuple[float, float, float]],
        target_angle: Optional[Tuple[float, float, float]],
        start_vx: float = 0.0,
        start_vy: float = 0.0,
        start_vz: float = 0.0,
        start_omega: float = 0.0,
    ) -> None:
        if start_pos is None:
            start_pos = SatelliteConfig.DEFAULT_START_POS
        if target_pos is None:
            target_pos = SatelliteConfig.DEFAULT_TARGET_POS
        if start_angle is None:
            start_angle = SatelliteConfig.DEFAULT_START_ANGLE
        if target_angle is None:
            target_angle = SatelliteConfig.DEFAULT_TARGET_ANGLE

        self.satellite = SatelliteThrusterTester(use_mujoco_viewer=self.use_mujoco_viewer)
        self.satellite.external_simulation_mode = True

        # Set initial state (including velocities)
        # Ensure start_pos is 3D
        sp = np.array(start_pos, dtype=np.float64)
        if sp.shape == (2,):
            sp = np.pad(sp, (0, 1), "constant")
        self.satellite.position = sp

        self.satellite.velocity = np.array([start_vx, start_vy, start_vz], dtype=np.float64)
        self.satellite.angle = start_angle
        # Type ignore: Property setter accepts float, getter returns ndarray
        self.satellite.angular_velocity = start_omega  # type: ignore

        # Store initial starting position and angle for reset functionality
        self.initial_start_pos = sp.copy()
        self.initial_start_angle = start_angle

        # Point-to-point mode (3D State: [p(3), q(4), v(3), w(3)])
        # Target State
        self.target_state = np.zeros(13)

        # Robust 3D target assignment
        tp = np.array(target_pos, dtype=np.float64)
        if tp.shape == (2,):
            tp = np.pad(tp, (0, 1), "constant")
        self.target_state[0:3] = tp

        # Target Orientation (3D Euler -> Quaternion)
        target_quat = euler_xyz_to_quat_wxyz(target_angle)
        self.target_state[3:7] = target_quat
        # Velocities = 0

        logger.info(
            f"INFO: POINT-TO-POINT MODE: "
            f"Target ({target_pos[0]:.2f}, {target_pos[1]:.2f}, 0.00)"
        )

        # Simulation state
        self.is_running = False
        self.simulation_time = 0.0
        self.max_simulation_time = SatelliteConfig.MAX_SIMULATION_TIME
        self.control_update_interval = SatelliteConfig.CONTROL_DT
        self.last_control_update = 0.0
        self.next_control_simulation_time = 0.0  # Track next scheduled control update

        # ===== HARDWARE COMMAND DELAY SIMULATION =====
        # Simulates the delay between sending a command and
        # thrusters actually firing
        # Uses Config parameters for realistic physics when enabled
        if SatelliteConfig.USE_REALISTIC_PHYSICS:
            self.VALVE_DELAY = SatelliteConfig.THRUSTER_VALVE_DELAY  # 50ms valve open/close delay
            self.THRUST_RAMPUP_TIME = (
                SatelliteConfig.THRUSTER_RAMPUP_TIME
            )  # 15ms ramp-up after valve opens
        else:
            self.VALVE_DELAY = 0.0  # Instant response for idealized physics
            self.THRUST_RAMPUP_TIME = 0.0

        # Thruster management (valve delays, ramp-up, PWM) - delegated
        self.num_thrusters = len(SatelliteConfig.THRUSTER_POSITIONS)
        self.thruster_manager = ThrusterManager(
            num_thrusters=self.num_thrusters,
            valve_delay=self.VALVE_DELAY,
            thrust_rampup_time=self.THRUST_RAMPUP_TIME,
            use_realistic_physics=SatelliteConfig.USE_REALISTIC_PHYSICS,
            thruster_type=SatelliteConfig.THRUSTER_TYPE,
        )

        # Convenience aliases for backward compatibility (read-only access)
        # These properties delegate to thruster_manager
        # ==============================================

        # Target maintenance tracking
        self.target_reached_time: Optional[float] = None
        self.approach_phase_start_time = 0.0
        self.target_maintenance_time = 0.0
        self.times_lost_target = 0
        self.maintenance_position_errors: List[float] = []
        self.maintenance_angle_errors: List[float] = []

        # Data logging
        self.state_history: List[np.ndarray] = []
        self.command_history: List[List[int]] = []  # For visual replay
        self.control_history: List[np.ndarray] = []
        self.target_history: List[np.ndarray] = []
        self.mpc_solve_times: List[float] = []
        self.mpc_info_history: List[dict] = []

        self.data_save_path: Optional[Path] = None

        # Previous command for rate limiting
        self.previous_command: Optional[np.ndarray] = None

        # Current control
        self.current_thrusters = np.zeros(self.num_thrusters, dtype=np.float64)
        self.previous_thrusters = np.zeros(self.num_thrusters, dtype=np.float64)

        self.position_tolerance = SatelliteConfig.POSITION_TOLERANCE
        self.angle_tolerance = SatelliteConfig.ANGLE_TOLERANCE
        self.velocity_tolerance = SatelliteConfig.VELOCITY_TOLERANCE
        self.angular_velocity_tolerance = SatelliteConfig.ANGULAR_VELOCITY_TOLERANCE

        # Initialize MPC Controller directly
        logger.info("Initializing MPC Controller (Mode: PWM/OSQP)...")
        app_config = SatelliteConfig.get_app_config()
        self.mpc_controller = MPCController(
            satellite_params=app_config.physics, mpc_params=app_config.mpc
        )
        # Initialize MissionStateManager for centralized mission logic
        self.mission_manager = MissionStateManager(
            position_tolerance=self.position_tolerance,
            angle_tolerance=self.angle_tolerance,
            normalize_angle_func=self.normalize_angle,
            angle_difference_func=self.angle_difference,
            point_to_line_distance_func=self.point_to_line_distance,
        )

        # Initialize state validator for centralized state validation
        self.state_validator = create_state_validator_from_config(
            {
                "position_tolerance": self.position_tolerance,
                "angle_tolerance": self.angle_tolerance,
                "velocity_tolerance": self.velocity_tolerance,
                "angular_velocity_tolerance": self.angular_velocity_tolerance,
            }
        )

        # Initialize data logger with FastLogger
        # Initialize data logger

        self.data_logger = create_data_logger(
            mode="simulation",
            filename="control_data.csv",
        )
        self.physics_logger = create_data_logger(
            mode="physics",
            filename="physics_data.csv",
        )

        self.report_generator = create_mission_report_generator(SatelliteConfig)
        self.data_save_path = None

        # Initialize IO helper for data export operations
        self._io = SimulationIO(self)


        # Initialize Simulation Context for logging
        from src.satellite_control.core.simulation_context import SimulationContext

        self.context = SimulationContext()
        self.context.dt = self.satellite.dt
        self.context.control_dt = self.control_update_interval

        logger.info("Linearized MPC Simulation initialized:")
        logger.info("INFO: Formulation: A*x[k] + B*u[k] (Linearized Dynamics)")
        def _format_euler_deg(euler: Tuple[float, float, float]) -> str:
            roll, pitch, yaw = np.degrees(euler)
            return f"roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°"

        s_ang_str = _format_euler_deg(start_angle)
        t_ang_str = _format_euler_deg(target_angle)

        logger.info(f"INFO: Start: {start_pos} m, {s_ang_str}")
        logger.info(f"INFO: Target: {target_pos} m, {t_ang_str}")
        logger.info(f"INFO: Control update rate: " f"{1 / self.control_update_interval:.1f} Hz")
        logger.info(f"INFO: Prediction horizon: {app_config.mpc.prediction_horizon}")
        logger.info(f"INFO: Control horizon: {app_config.mpc.control_horizon}")

        if SatelliteConfig.USE_REALISTIC_PHYSICS:
            logger.info("WARNING: REALISTIC PHYSICS ENABLED:")
            logger.info(f"WARNING: - Valve delay: " f"{self.VALVE_DELAY * 1000:.0f} ms")
            logger.info(f"WARNING: - Ramp-up time: " f"{self.THRUST_RAMPUP_TIME * 1000:.0f} ms")
            logger.info(
                f"WARNING: - Linear damping: " f"{SatelliteConfig.LINEAR_DAMPING_COEFF:.3f} N/(m/s)"
            )
            logger.info(
                f"WARNING: - Rotational damping: "
                f"{SatelliteConfig.ROTATIONAL_DAMPING_COEFF:.4f} N*m/(rad/s)"
            )
            logger.info(
                f"WARNING: - Position noise: "
                f"{SatelliteConfig.POSITION_NOISE_STD * 1000:.2f} mm std"
            )
            angle_noise_deg = np.degrees(SatelliteConfig.ANGLE_NOISE_STD)
            logger.info(f"WARNING: - Angle noise: {angle_noise_deg:.2f}° std")
        else:
            logger.info("INFO: Idealized physics (no delays, noise, or damping)")

        # Apply obstacle avoidance based on mode
        if SatelliteConfig.OBSTACLES_ENABLED and SatelliteConfig.get_obstacles():
            logger.info("Obstacle avoidance enabled.")

        # Initialize visualization manager
        self.visualizer = create_simulation_visualizer(self)

    def get_current_state(self) -> np.ndarray:
        """Get current satellite state [pos(3), quat(4), vel(3), ang_vel(3)]."""
        s = self.satellite
        # [x, y, z]
        pos = s.position
        # [w, x, y, z]
        quat = s.quaternion
        # [vx, vy, vz]
        vel = s.velocity
        # [wx, wy, wz]
        ang_vel = s.angular_velocity

        return np.concatenate([pos, quat, vel, ang_vel])

    # Backward-compatible properties delegating to ThrusterManager
    @property
    def thruster_actual_output(self) -> np.ndarray:
        """Get actual thruster output levels [0, 1] for each thruster."""
        return self.thruster_manager.thruster_actual_output

    @property
    def thruster_last_command(self) -> np.ndarray:
        """Get last commanded thruster pattern."""
        return self.thruster_manager.thruster_last_command

    def get_noisy_state(self, true_state: np.ndarray) -> np.ndarray:
        """
        Add realistic sensor noise to state measurements.
        Models OptiTrack measurement uncertainty and velocity estimation
        errors.

        Delegates to SimulationStateValidator for noise application.

        Args:
            true_state: True state [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]

        Returns:
            Noisy state with measurement errors added
        """
        return self.state_validator.apply_sensor_noise(true_state)

    def create_data_directories(self) -> Path:
        """
        Create the directory structure for saving data.
        Returns the path to the timestamped subdirectory.
        """
        return self._io.create_data_directories()

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range (navigation_utils)."""
        return normalize_angle(angle)

    def angle_difference(self, target_angle: float, currentAngle: float) -> float:
        """
        Calculate shortest angular difference between angles.
        Delegated to navigation_utils.
        Prevents the 360°/0° transition issue by taking shortest path.
        Returns: angle difference in [-pi, pi], positive = CCW rotation
        """
        return angle_difference(target_angle, currentAngle)

    def point_to_line_distance(
        self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
    ) -> float:
        """Calculate distance from point to line segment (navigation_utils)."""
        return point_to_line_distance(point, line_start, line_end)

    # OBSTACLE AVOIDANCE METHODS

    def log_simulation_step(
        self,
        mpc_start_time: float,
        command_sent_time: float,
        thruster_action: np.ndarray,
        mpc_info: Optional[dict],
    ):
        """
        Log detailed simulation step data for CSV export with timing analysis.
        Delegates to SimulationLogger.
        """
        current_state = self.get_current_state()

        # --- Visual Replay Logging (Legacy InMemory) ---
        self.state_history.append(current_state.copy())

        # Determine active thrusters
        # Note: thruster_action is passed in, use that instead of
        # self.current_thrusters to be safe
        active_thrusters = [i + 1 for i, val in enumerate(thruster_action) if val > 0.01]
        self.command_history.append(active_thrusters)
        # ---------------------------------------------------------------------

        # Determine mission phase
        if getattr(SatelliteConfig, "DXF_SHAPE_MODE_ACTIVE", False) and getattr(
            SatelliteConfig, "DXF_SHAPE_PHASE", ""
        ):
            mission_phase = SatelliteConfig.DXF_SHAPE_PHASE
        else:
            mission_phase = "STABILIZING" if self.target_reached_time is not None else "APPROACHING"

        # Delegate to SimulationLogger
        if not hasattr(self, "logger_helper"):
            from src.satellite_control.core.simulation_logger import (
                SimulationLogger,
            )

            self.logger_helper = SimulationLogger(self.data_logger)

        previous_thruster_action: Optional[np.ndarray] = (
            self.previous_command if hasattr(self, "previous_command") else None
        )

        # Update Context
        self.context.update_state(self.simulation_time, current_state, self.target_state)
        self.context.step_number = self.data_logger.current_step
        self.context.mission_phase = mission_phase
        self.context.previous_thruster_command = previous_thruster_action

        # Ensure mpc_info has required keys, providing defaults if missing
        mpc_info_safe = mpc_info if mpc_info is not None else {}
        # mpc_computation_time = mpc_info_safe.get("solve_time", 0.0)

        self.logger_helper.log_step(
            self.context,
            mpc_start_time,
            command_sent_time,
            thruster_action,
            mpc_info_safe,
        )

        self.previous_command = thruster_action.copy()

    def log_physics_step(self):
        """Log high-frequency physics data (every 5ms)."""
        if not self.data_save_path:
            return

        current_state = self.get_current_state()

        # Get target state (handle if not set)
        target_state = self.target_state if self.target_state is not None else np.zeros(13)

        # Delegate to SimulationLogger
        if not hasattr(self, "physics_logger_helper"):
            from src.satellite_control.core.simulation_logger import (
                SimulationLogger,
            )

            self.physics_logger_helper = SimulationLogger(self.physics_logger)

        self.physics_logger_helper.log_physics_step(
            simulation_time=self.simulation_time,
            current_state=current_state,
            target_state=target_state,
            thruster_actual_output=self.thruster_actual_output,
            thruster_last_command=self.thruster_last_command,
            normalize_angle_func=self.normalize_angle,
        )

    def save_csv_data(self) -> None:
        """Save all logged data to CSV file (delegated to SimulationIO)."""
        self._io.save_csv_data()

    def save_mission_summary(self) -> None:
        """Generate and save mission summary report (delegated to SimulationIO)."""
        self._io.save_mission_summary()

    def save_animation_mp4(self, fig: Any, ani: Any) -> Optional[str]:
        """
        Save the animation as MP4 file (delegated to SimulationIO).

        Args:
            fig: Matplotlib figure object
            ani: Matplotlib animation object

        Returns:
            Path to saved MP4 file or None if save failed
        """
        return self._io.save_animation_mp4(fig, ani)

    def set_thruster_pattern(self, thruster_pattern: np.ndarray) -> None:
        """
        Send thruster command (delegated to ThrusterManager).

        Command is sent at current simulation_time, but valve opening/closing
        takes VALVE_DELAY to complete.

        Args:
            thruster_pattern: Array [0,1] for thruster commands (duty cycle)
        """
        self.thruster_manager.set_thruster_pattern(thruster_pattern, self.simulation_time)
        # Keep simulation-level current_thrusters in sync
        self.current_thrusters = self.thruster_manager.current_thrusters

    def process_command_queue(self) -> None:
        """
        Update actual thruster output based on valve delays and ramp-up.

        Delegated to ThrusterManager which handles all valve physics.
        Called every simulation timestep to update actual thruster forces.
        """
        self.thruster_manager.process_command_queue(
            simulation_time=self.simulation_time,
            control_update_interval=self.control_update_interval,
            last_control_update=self.last_control_update,
            sim_dt=self.satellite.dt,
            satellite=self.satellite,
        )

    def update_target_state_for_mode(self, current_state: np.ndarray) -> None:
        """
        Update the target state based on the current control mode.

        Delegates to MissionStateManager for centralized mission logic.
        Replaces ~330 lines of complex nested code with clean delegation.

        Args:
            current_state: Current state vector [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        # Track target index to detect waypoint advances
        prev_target_index = getattr(SatelliteConfig, "CURRENT_TARGET_INDEX", 0)

        # Delegate to MissionStateManager for centralized mission logic
        target_state = self.mission_manager.update_target_state(
            current_position=self.satellite.position,
            current_quat=self.satellite.quaternion,
            current_time=self.simulation_time,
            current_state=current_state,
        )

        # Detect if MissionStateManager advanced to a new waypoint
        new_target_index = getattr(SatelliteConfig, "CURRENT_TARGET_INDEX", 0)
        if new_target_index != prev_target_index:
            # Check if mission is complete to avoid false reset
            is_mission_complete = getattr(SatelliteConfig, "MULTI_POINT_PHASE", "") == "COMPLETE"

            # Only reset if NOT complete (advancing to valid next target)
            if not is_mission_complete:
                # Reset target reached status for proper phase display
                self.target_reached_time = None
                self.target_maintenance_time = 0.0
                self.approach_phase_start_time = self.simulation_time

        # If mission returns None, it means mission is complete
        if target_state is None:
            # Check if it's a completion signal (mission finished)
            if self.mission_manager.dxf_completed:
                logger.info("DXF PROFILE COMPLETED! Profile successfully traversed.")
                self.is_running = False
                self.print_performance_summary()
                return
        else:
            # Check for significant target change in Waypoint Mode (Robustness fallback)
            # Triggers if index tracking failed to catch the transition
            if self.target_state is not None and getattr(
                SatelliteConfig, "ENABLE_WAYPOINT_MODE", False
            ):
                # 3D Position Change
                pos_change = np.linalg.norm(target_state[:3] - self.target_state[:3])

                # 3D Angle Change (Quaternion)
                q1 = target_state[3:7]
                q2 = self.target_state[3:7]
                dot = np.abs(np.dot(q1, q2))
                dot = min(1.0, max(-1.0, dot))
                ang_change = 2.0 * np.arccos(dot)

                if pos_change > 1e-4 or ang_change > 1e-4:
                    if self.target_reached_time is not None:
                        logger.info("Target changed significantly - resetting reached timer")
                        self.target_reached_time = None
                        self.target_maintenance_time = 0.0
                        self.approach_phase_start_time = self.simulation_time

            # Update our target state from the mission manager
            self.target_state = target_state

        return

    def update_mpc_control(self) -> None:
        """Update control action using linearized MPC with strict timing."""
        # Force MPC to send commands at fixed intervals
        if self.simulation_time >= self.next_control_simulation_time:

            # Delegate to MPCRunner
            if not hasattr(self, "mpc_runner"):
                from src.satellite_control.core.mpc_runner import MPCRunner

                # Initialize MPC Runner wrapper
                self.mpc_runner = MPCRunner(
                    mpc_controller=self.mpc_controller,
                    config=self.structured_config,
                    state_validator=self.state_validator,
                )

            current_state = self.get_current_state()
            mpc_start_time = self.simulation_time

            # Generate Trajectory for Smart MPC
            # Default to N=10, dt=0.05 if no controller params found
            horizon = getattr(self.mpc_controller, "N", 10)
            # mpc_params removed (unused)
            dt = getattr(self.mpc_controller, "dt", 0.05)

            target_trajectory = self.mission_manager.get_trajectory(
                current_time=self.simulation_time,
                dt=dt,
                horizon=horizon,
                current_state=current_state,
                external_target_state=self.target_state,
            )

            # Compute action
            (
                thruster_action,
                mpc_info,
                mpc_computation_time,
                command_sent_time,
            ) = self.mpc_runner.compute_control_action(
                true_state=current_state,
                target_state=self.target_state,
                previous_thrusters=self.previous_thrusters,
                target_trajectory=target_trajectory,
            )

            # Update simulation state
            self.last_control_update = self.simulation_time
            self.next_control_simulation_time += self.control_update_interval

            # Update history / command queue
            self.previous_thrusters = thruster_action.copy()
            self.control_history.append(thruster_action.copy())

            self.set_thruster_pattern(thruster_action)

            # Log Data - Use simulation time for consistency with
            # mpc_start_time
            command_sent_sim_time = self.simulation_time

            self.log_simulation_step(
                mpc_start_time,
                command_sent_sim_time,
                thruster_action,
                mpc_info,
            )

            # Verify timing constraint
            if mpc_computation_time > (self.control_update_interval - 0.02):
                logger.warning(
                    f"WARNING: MPC computation time "
                    f"({mpc_computation_time:.3f}s) exceeds real-time!"
                )

            # Print status with timing information
            pos_error = np.linalg.norm(current_state[:3] - self.target_state[:3])

            # Quaternion error: 2 * arccos(|<q1, q2>|)
            ang_error = quat_angle_error(self.target_state[3:7], current_state[3:7])

            # Determine status message
            status_msg = f"Traveling to Target (t={self.simulation_time:.1f}s)"
            stabilization_time = None

            if self.target_reached_time is not None:
                stabilization_time = self.simulation_time - self.target_reached_time
                status_msg = f"Stabilizing on Target (t = {stabilization_time:.1f}s)"

            elif getattr(SatelliteConfig, "DXF_SHAPE_MODE_ACTIVE", False):
                phase = getattr(SatelliteConfig, "DXF_SHAPE_PHASE", "UNKNOWN")
                # Map internal phase names to user-friendly display names
                phase_display_names = {
                    "POSITIONING": "Traveling to Path",
                    "PATH_STABILIZATION": "Stabilizing on Path",
                    "TRACKING": "Traveling on Path",
                    "STABILIZING": "Stabilizing on Path",
                    "RETURNING": "Traveling to Target",
                }
                display_phase = phase_display_names.get(phase, phase)
                # For RETURNING phase, check if we're stabilizing at the end
                if phase == "RETURNING" and self.target_reached_time is not None:
                    display_phase = "Stabilizing on Target"
                status_msg = f"{display_phase} (t = {self.simulation_time:.1f}s)"
            else:
                status_msg = f"Traveling to Target (t = {self.simulation_time:.1f}s)"

            # Prepare display variables and update command history
            if thruster_action.ndim > 1:
                display_thrusters = thruster_action[0, :]
            else:
                display_thrusters = thruster_action

            active_thruster_ids = [int(x) for x in np.where(display_thrusters > 0.01)[0] + 1]
            self.command_history.append(active_thruster_ids)

            # Helper for clean state formatting
            def fmt_state(s):
                x_mm = s[0] * 1000
                y_mm = s[1] * 1000
                z_mm = s[2] * 1000
                # Display Roll/Pitch/Yaw
                # Simpler: just Z-angle if we assume flat?
                # But it's 3D.
                # Let's show Quat or convert to Euler?
                # Using simple Z-axis rotation approximation for logging conciseness?
                # No, show Euler. mju_quat2Mat logic...
                # Simple conversion to "Yaw" approx
                q = s[3:7]
                yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                yaw_deg = np.degrees(yaw)
                return f"[x:{x_mm:.0f}, y:{y_mm:.0f}, z:{z_mm:.0f}]mm Yaw:{yaw_deg:.1f}°"

            safe_target = self.target_state if self.target_state is not None else np.zeros(13)

            ang_err_deg = np.degrees(ang_error)
            solve_ms = mpc_info.get("solve_time", 0) * 1000
            next_upd = self.next_control_simulation_time
            # Show duty cycle for each active thruster (matching active_thruster_ids)
            thr_out = [round(float(display_thrusters[i - 1]), 2) for i in active_thruster_ids]
            logger.info(
                f"t = {self.simulation_time:.1f}s: {status_msg}\n"
                f"     Pos Err = {pos_error:.3f}m, "
                f"Ang Err = {ang_err_deg:.1f}°\n"
                f"     Current = {fmt_state(current_state)}\n"
                f"     Target = {fmt_state(safe_target)}\n"
                f"     Solve = {solve_ms:.1f}ms, Next = {next_upd:.3f}s\n"
                f"     Thrusters = {active_thruster_ids}\n"
                f"     Output = {thr_out}\n"
            )

            # Log terminal message to CSV
            terminal_entry = {
                "Time": self.simulation_time,
                "Status": status_msg,
                "Stabilization_Time": (
                    stabilization_time if stabilization_time is not None else ""
                ),
                "Position_Error_m": pos_error,
                "Angle_Error_deg": np.degrees(ang_error),
                "Active_Thrusters": str(active_thruster_ids),
                "Solve_Time_s": mpc_computation_time,
                "Next_Update_s": self.next_control_simulation_time,
            }
            self.data_logger.log_terminal_message(terminal_entry)

    def check_target_reached(self) -> bool:
        """
        Check if satellite has reached the target within tolerances.

        Delegates to SimulationStateValidator for tolerance checking.
        """
        current_state = self.get_current_state()
        return self.state_validator.check_target_reached(current_state, self.target_state)

    def update_simulation(self, frame: int) -> List[Any]:
        """
        Update simulation step (called by matplotlib animation).

        Args:
            frame: Current frame number

        Returns:
            List of artists for matplotlib animation
        """
        if not self.is_running:
            return []

        # Update target state based on mission mode (unified with Real.py)
        current_state = self.get_current_state()
        self.update_target_state_for_mode(current_state)

        self.update_mpc_control()

        # Process command queue to apply delayed commands (sets
        # active_thrusters)
        self.process_command_queue()

        # Advance MuJoCo physics: keep time bases aligned for valve timing
        dt = self.satellite.dt
        self.satellite.simulation_time = self.simulation_time
        self.satellite.update_physics(dt)
        self.simulation_time = self.satellite.simulation_time

        # Log High-Frequency Physics Data
        self.log_physics_step()

        if not (
            hasattr(SatelliteConfig, "DXF_SHAPE_MODE_ACTIVE")
            and SatelliteConfig.DXF_SHAPE_MODE_ACTIVE
        ):
            target_currently_reached = self.check_target_reached()

            if target_currently_reached:
                if self.target_reached_time is None:
                    # First time reaching target
                    self.target_reached_time = self.simulation_time
                    print(
                        f"\nTARGET REACHED! Time: {self.simulation_time:.1f}s"
                        " - MPC will maintain position"
                    )
                else:
                    # Update maintenance tracking
                    self.target_maintenance_time = self.simulation_time - self.target_reached_time
                    current_state = self.get_current_state()
                    pos_error = np.linalg.norm(current_state[:3] - self.target_state[:3])
                    ang_error = quat_angle_error(
                        self.target_state[3:7], current_state[3:7]
                    )
                    self.maintenance_position_errors.append(float(pos_error))
                    self.maintenance_angle_errors.append(float(ang_error))

                if (
                    hasattr(SatelliteConfig, "ENABLE_WAYPOINT_MODE")
                    and SatelliteConfig.ENABLE_WAYPOINT_MODE
                ):
                    stabilization_time = self.target_maintenance_time

                    is_final_target = (
                        SatelliteConfig.CURRENT_TARGET_INDEX
                        >= len(SatelliteConfig.WAYPOINT_TARGETS) - 1
                    )
                    if is_final_target:
                        required_hold_time = SatelliteConfig.WAYPOINT_FINAL_STABILIZATION_TIME
                    else:
                        required_hold_time = getattr(SatelliteConfig, "TARGET_HOLD_TIME", 3.0)

                    if stabilization_time >= required_hold_time:
                        # Advance to next target
                        next_available = SatelliteConfig.advance_to_next_target()
                        if next_available:
                            # Update target state to next target with obstacle
                            # avoidance
                            (
                                target_pos,
                                target_angle,
                            ) = SatelliteConfig.get_current_waypoint_target()
                            if target_pos is not None:
                                roll_deg, pitch_deg, yaw_deg = np.degrees(target_angle)
                                logger.info(
                                    f"MOVING TO NEXT TARGET: "
                                    f"({target_pos[0]:.2f}, "
                                    f"{target_pos[1]:.2f}) m, "
                                    f"roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°, yaw={yaw_deg:.1f}°"
                                )
                                self.target_state = np.zeros(13, dtype=float)
                                self.target_state[0:3] = target_pos
                                self.target_state[3:7] = euler_xyz_to_quat_wxyz(target_angle)
                                self.target_reached_time = None
                                self.approach_phase_start_time = self.simulation_time
                                self.target_maintenance_time = 0.0
                        else:
                            # All targets completed - end simulation
                            logger.info("WAYPOINT MISSION COMPLETED! " "All targets visited.")
                            use_stab = SatelliteConfig.USE_FINAL_STABILIZATION_IN_SIMULATION
                            if not use_stab:
                                self.is_running = False
                                self.print_performance_summary()
                                return []
                            SatelliteConfig.MULTI_POINT_PHASE = "COMPLETE"
            else:
                if self.target_reached_time is not None:
                    self.times_lost_target += 1
                    t = self.simulation_time
                    print(f"WARNING: Target lost at t={t:.1f}s" " - MPC working to regain control")

        if (
            not SatelliteConfig.USE_FINAL_STABILIZATION_IN_SIMULATION
            and self.target_reached_time is not None
            and not (
                hasattr(SatelliteConfig, "ENABLE_WAYPOINT_MODE")
                and SatelliteConfig.ENABLE_WAYPOINT_MODE
            )
            and not (
                hasattr(SatelliteConfig, "DXF_SHAPE_MODE_ACTIVE")
                and SatelliteConfig.DXF_SHAPE_MODE_ACTIVE
            )
        ):
            # Mission 1: Waypoint Navigation (single waypoint) with immediate
            # termination after target reached
            current_maintenance_time = self.simulation_time - self.target_reached_time
            if current_maintenance_time >= SatelliteConfig.WAYPOINT_FINAL_STABILIZATION_TIME:
                stab_time = SatelliteConfig.WAYPOINT_FINAL_STABILIZATION_TIME
                print(
                    f"\n WAYPOINT MISSION COMPLETE! "
                    f"Stable at target for {stab_time:.1f} seconds."
                )
                self.is_running = False
                self.print_performance_summary()
                return []

        if (
            SatelliteConfig.USE_FINAL_STABILIZATION_IN_SIMULATION
            and self.target_reached_time is not None
        ):
            current_maintenance_time = self.simulation_time - self.target_reached_time

            # Waypoint navigation: check completion for single or multiple
            # waypoints
            if not (
                hasattr(SatelliteConfig, "DXF_SHAPE_MODE_ACTIVE")
                and SatelliteConfig.DXF_SHAPE_MODE_ACTIVE
            ):
                # Single waypoint (no ENABLE_WAYPOINT_MODE set)
                if not (
                    hasattr(SatelliteConfig, "ENABLE_WAYPOINT_MODE")
                    and SatelliteConfig.ENABLE_WAYPOINT_MODE
                ):
                    if (
                        current_maintenance_time
                        >= SatelliteConfig.WAYPOINT_FINAL_STABILIZATION_TIME
                    ):
                        final_stab = SatelliteConfig.WAYPOINT_FINAL_STABILIZATION_TIME
                        stab_t = final_stab
                        print(
                            f"\n WAYPOINT MISSION COMPLETE! "
                            f"Stable at target for {stab_t:.1f} seconds."
                        )
                        self.is_running = False
                        self.print_performance_summary()
                        return []
                # Multiple waypoints (ENABLE_WAYPOINT_MODE = True)
                elif getattr(SatelliteConfig, "MULTI_POINT_PHASE", None) == "COMPLETE":
                    if (
                        current_maintenance_time
                        >= SatelliteConfig.WAYPOINT_FINAL_STABILIZATION_TIME
                    ):
                        final_stab = SatelliteConfig.WAYPOINT_FINAL_STABILIZATION_TIME
                        stab_t = final_stab
                        print(
                            f"\n WAYPOINT MISSION COMPLETE! "
                            f"All targets stable for {stab_t:.1f} seconds."
                        )
                        self.is_running = False
                        self.print_performance_summary()
                        return []

        # Only stop simulation when max time is reached
        if self.simulation_time >= self.max_simulation_time:
            print(f"\nSIMULATION COMPLETE at {self.simulation_time:.1f}s")
            self.is_running = False
            self.print_performance_summary()

        # Redraw
        self.draw_simulation()
        self.update_mpc_info_panel()  # Use custom MPC info panel instead

        return []

    def draw_simulation(self) -> None:
        """Draw the simulation with satellite, target, and trajectory."""
        self.visualizer.sync_from_controller()
        self.visualizer.draw_simulation()

    def _draw_obstacles(self) -> None:
        """Draw configured obstacles on the visualization (delegated)."""
        self.visualizer._draw_obstacles()

    def _draw_obstacle_avoidance_waypoints(self) -> None:
        """Draw obstacle avoidance waypoints for point-to-point modes."""
        self.visualizer._draw_obstacle_avoidance_waypoints()

    def _draw_satellite_elements(self) -> None:
        """Draw satellite elements manually to avoid conflicts (delegated)."""
        self.visualizer._draw_satellite_elements()

    def update_mpc_info_panel(self) -> None:
        """Update the information panel to match visualization format."""
        self.visualizer.sync_from_controller()
        self.visualizer.update_mpc_info_panel()

    def print_performance_summary(self) -> None:
        """Print performance summary at the end of simulation (delegated)."""
        self.visualizer.sync_from_controller()
        self.visualizer.print_performance_summary()

    def reset_simulation(self) -> None:
        """Reset simulation to initial state (delegated)."""
        self.visualizer.sync_from_controller()
        self.visualizer.reset_simulation()

    def auto_generate_visualizations(self) -> None:
        """Generate all visualizations after simulation completion."""
        self.visualizer.sync_from_controller()
        self.visualizer.auto_generate_visualizations()

    def _run_simulation_with_globals(self, show_animation: bool = True) -> None:
        """
        Run linearized MPC simulation.

        Args:
            show_animation: Whether to display animation during simulation
        """
        print("\nStarting Linearized MPC Simulation...")
        print("Press 'q' to quit early, Space to pause/resume")
        self.is_running = True

        # Clear any previous data from the logger
        self.data_logger.clear_logs()
        self.physics_logger.clear_logs()

        self.data_save_path = self.create_data_directories()
        if self.data_save_path:
            self.data_logger.set_save_path(self.data_save_path)
            self.physics_logger.set_save_path(self.data_save_path)
            logger.info("Created data directory: %s", self.data_save_path)

        # Simulation Context
        from src.satellite_control.core.simulation_context import (
            SimulationContext,
        )

        if not hasattr(self, "context"):
            self.context = SimulationContext()
            self.context.dt = self.satellite.dt
            self.context.control_dt = self.control_update_interval

        # Initialize MPC Controller (Linearized Model)
        try:
            # When using MuJoCo viewer, skip matplotlib animation (MuJoCo
            # viewer updates itself)
            if show_animation and not self.use_mujoco_viewer:
                # Matplotlib animation mode (legacy)
                fig = self.satellite.fig
                ani = FuncAnimation(
                    fig,
                    self.update_simulation,
                    interval=int(self.satellite.dt * 1000),
                    blit=False,
                    repeat=False,
                    cache_frame_data=False,
                )
                plt.show()  # Show the animation window live

                # After animation is complete, save files
                if self.data_save_path is not None:
                    print("\nSaving simulation data...")
                    self.save_csv_data()
                    self.visualizer.sync_from_controller()
                    self.save_mission_summary()
                    self.save_animation_mp4(fig, ani)
                    print(f" Data saved to: {self.data_save_path}")

                    print("\n Auto-generating performance plots...")
                    self.auto_generate_visualizations()
                    if hasattr(self.visualizer, "save_mujoco_video"):
                        self.visualizer.save_mujoco_video(self.data_save_path)
                    print(" All visualizations complete!")
            else:
                # Run with MuJoCo viewer or without animation

                # Performance Optimization: Batch physics steps
                # Calculate how many physics steps fit in one control update
                steps_per_batch = int(self.control_update_interval / self.satellite.dt)
                if steps_per_batch < 1:
                    steps_per_batch = 1

                batch_mode = steps_per_batch > 1
                logger.info(
                    f"Running optimized simulation loop. "
                    f"Batch: {steps_per_batch} (dt={self.satellite.dt:.4f}s)"
                )

                fast_batch_steps = steps_per_batch - 1

                while self.is_running:
                    step_only = False
                    if (
                        self.use_mujoco_viewer
                        and hasattr(self.satellite, "is_viewer_paused")
                        and self.satellite.is_viewer_paused()
                    ):
                        if hasattr(self.satellite, "consume_viewer_step") and self.satellite.consume_viewer_step():
                            step_only = True
                        else:
                            if hasattr(self.satellite, "sync_viewer"):
                                self.satellite.sync_viewer()
                            time.sleep(0.01)
                            continue

                    # Optimized Batch: Run physics steps without control logic
                    # overhead
                    if batch_mode and not step_only:
                        for _ in range(fast_batch_steps):
                            # Inline logic for speed
                            self.process_command_queue()
                            self.satellite.update_physics(self.satellite.dt)
                            self.simulation_time = self.satellite.simulation_time
                            self.log_physics_step()

                    # Full Update (run MPC check, Mission Check, Logging, 1
                    # Physics Step)
                    self.update_simulation(None)  # type: ignore[arg-type]

                    if not self.is_running:
                        break

                if self.data_save_path is not None:
                    print("\nSaving simulation data...")
                    self.save_csv_data()
                    self.visualizer.sync_from_controller()
                    self.save_mission_summary()
                    print(f" CSV data saved to: {self.data_save_path}")

                    # Auto-generate all visualizations
                    print("\n Auto-generating visualizations...")
                    self.auto_generate_visualizations()
                    # Also generate MuJoCo 3D render if possible
                    if hasattr(self.visualizer, "save_mujoco_video"):
                        self.visualizer.save_mujoco_video(self.data_save_path)
                    print(" All visualizations complete!")

        except KeyboardInterrupt:
            print("\n\nSimulation cancelled by user")
            self.is_running = False

            # Save data when interrupted
            if self.data_save_path is not None and self.data_logger.get_log_count() > 0:
                print("\nSaving simulation data...")
                self.save_csv_data()
                self.visualizer.sync_from_controller()
                self.save_mission_summary()
                print(f" Data saved to: {self.data_save_path}")

                # Try to generate visualizations if we have enough data
                if self.data_logger.get_log_count() > 10:
                    try:
                        print("\n Auto-generating visualizations...")
                        self.auto_generate_visualizations()
                        if hasattr(self.visualizer, "save_mujoco_video"):
                            self.visualizer.save_mujoco_video(self.data_save_path)
                        print(" All visualizations complete!")
                    except Exception as e:
                        logger.warning(f"WARNING: Could not generate visualizations: {e}")

        finally:
            # Cleanup
            pass
        return self.data_save_path  # type: ignore[return-value]

    def run_simulation(self, show_animation: bool = True) -> None:
        """
        Run simulation with a per-instance structured config sandbox.

        Args:
            show_animation: Whether to display animation during simulation
        """
        with use_structured_config(self.structured_config.clone()):
            return self._run_simulation_with_globals(show_animation=show_animation)

    def close(self) -> None:
        """
        Clean up simulation resources.
        """
        # Close matplotlib figures if any
        plt.close("all")

        # Close visualizer if it supports it
        if hasattr(self, "visualizer") and hasattr(self.visualizer, "close"):
            self.visualizer.close()

        # Close MuJoCo viewer if accessible
        if hasattr(self, "satellite") and hasattr(self.satellite, "close"):
            self.satellite.close()

        logger.info("Simulation closed.")
