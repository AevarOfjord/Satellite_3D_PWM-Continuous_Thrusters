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

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# V4.0.0: SatelliteConfig removed - use SimulationConfig only
from src.satellite_control.config import (
    SimulationConfig,
    StructuredConfig,
    build_structured_config,
    use_structured_config,
)

# V4.0.0: SatelliteConfig removed - use SimulationConfig only
from src.satellite_control.config.constants import Constants
from src.satellite_control.config.satellite_config import (
    initialize_config,
)
from src.satellite_control.control.mpc_controller import MPCController
from src.satellite_control.core.mujoco_satellite import SatelliteThrusterTester
from src.satellite_control.core.simulation_initialization import SimulationInitializer
from src.satellite_control.core.simulation_io import SimulationIO
from src.satellite_control.core.simulation_loop import SimulationLoop
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

    This class now acts as a public API orchestrator, delegating to:
    - SimulationInitializer: Handles all initialization logic
    - SimulationLoop: Handles main loop execution
    - Various managers: MPC, Mission, Thruster, etc.

    The public API remains unchanged for backward compatibility.
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
        simulation_config: Optional[SimulationConfig] = None,
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
            config: Optional structured config snapshot to run against (deprecated, use simulation_config)
            config_overrides: Nested override dict for build_structured_config (deprecated, use simulation_config)
            simulation_config: Optional SimulationConfig (preferred, eliminates global state)
            use_mujoco_viewer: If True, use MuJoCo viewer (default: True)
        """
        self.use_mujoco_viewer = use_mujoco_viewer
        
        # Use SimulationConfig if provided (new preferred way)
        if simulation_config is not None:
            self.simulation_config = simulation_config
            # Convert to structured_config for backward compatibility
            self.structured_config = build_structured_config(
                {
                    "mpc": simulation_config.get_mpc_params(),
                    "physics": simulation_config.get_physics_params(),
                    "simulation": simulation_config.get_simulation_params(),
                }
            )
        else:
            # Backward compatibility: use old structured_config approach
            self.structured_config = (
                config.clone() if config else build_structured_config(config_overrides)
            )
            # V4.0.0: Create SimulationConfig from structured_config for internal use
            # No SatelliteConfig fallback - use defaults if needed
            from src.satellite_control.config.simulation_config import SimulationConfig
            app_config = SimulationConfig.create_default().app_config
            # For v2.0.0, prefer a fresh MissionState; legacy CLIs now
            # populate mission_state directly via SimulationConfig.
            from src.satellite_control.config.mission_state import create_mission_state

            mission_state = create_mission_state()

            self.simulation_config = SimulationConfig(
                app_config=app_config,
                mission_state=mission_state,
            )
        
        # Validate configuration if overrides were provided
        if config_overrides:
            from src.satellite_control.config.validator import ConfigValidator
            
            validator = ConfigValidator(self.simulation_config.app_config)
            issues = validator.validate()
            if issues:
                logger.warning("Configuration validation issues detected:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                # Don't raise - allow simulation to proceed with warnings
                # This is less strict than startup validation
        
        # Use SimulationInitializer to handle all initialization
        with use_structured_config(self.structured_config.clone()):
            initializer = SimulationInitializer(
                simulation=self,
                simulation_config=self.simulation_config,
                use_mujoco_viewer=self.use_mujoco_viewer,
            )
            initializer.initialize(
                start_pos,
                target_pos,
                start_angle,
                target_angle,
                start_vx,
                start_vy,
                start_vz,
                start_omega,
            )

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
        # V4.0.0: Get mission phase from mission_manager if available
        if (
            self.mission_manager
            and self.mission_manager.mission_state
            and self.mission_manager.mission_state.dxf_shape_mode_active
        ):
            mission_phase = self.mission_manager.mission_state.dxf_shape_phase or "STABILIZING"
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
        # Track target index to detect waypoint advances (V4.0.0: use mission_manager)
        prev_target_index = (
            self.mission_manager.mission_state.current_target_index
            if self.mission_manager and self.mission_manager.mission_state
            else 0
        )

        # Delegate to MissionStateManager for centralized mission logic
        target_state = self.mission_manager.update_target_state(
            current_position=self.satellite.position,
            current_quat=self.satellite.quaternion,
            current_time=self.simulation_time,
            current_state=current_state,
        )

        # Detect if MissionStateManager advanced to a new waypoint (V4.0.0: use mission_manager)
        new_target_index = (
            self.mission_manager.mission_state.current_target_index
            if self.mission_manager and self.mission_manager.mission_state
            else 0
        )
        if new_target_index != prev_target_index:
            # Check if mission is complete to avoid false reset (V4.0.0: use mission_manager)
            is_mission_complete = (
                self.mission_manager.mission_state.multi_point_phase == "COMPLETE"
                if self.mission_manager and self.mission_manager.mission_state
                else False
            )

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
            # Triggers if index tracking failed to catch the transition (V4.0.0: use mission_manager)
            enable_waypoint = (
                self.mission_manager.mission_state.enable_waypoint_mode
                if self.mission_manager and self.mission_manager.mission_state
                else False
            )
            if self.target_state is not None and enable_waypoint:
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

            # Record performance metrics
            solve_time = mpc_info.get("solve_time", mpc_computation_time) if mpc_info else mpc_computation_time
            timeout = mpc_info.get("timeout", False) if mpc_info else False
            self.performance_monitor.record_mpc_solve(solve_time, timeout=timeout)
            
            # Record control loop time (from MPC start to command sent)
            control_loop_time = command_sent_time - mpc_start_time
            timing_violation = mpc_computation_time > (self.control_update_interval - 0.02)
            self.performance_monitor.record_control_loop(control_loop_time, timing_violation=timing_violation)

            # Verify timing constraint
            if timing_violation:
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

            elif (
                self.mission_manager
                and self.mission_manager.mission_state
                and self.mission_manager.mission_state.dxf_shape_mode_active
            ):
                phase = self.mission_manager.mission_state.dxf_shape_phase or "UNKNOWN"
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
        """Print performance summary at the end of simulation."""
        # Export performance metrics
        if self.data_save_path:
            metrics_path = self.data_save_path / "performance_metrics.json"
            try:
                self.performance_monitor.export_metrics(metrics_path)
                logger.info(f"Performance metrics exported to {metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to export performance metrics: {e}")
        
        # Print performance summary
        self.performance_monitor.print_summary()
        
        # Check thresholds and warn
        warnings = self.performance_monitor.check_thresholds()
        if warnings:
            logger.warning("Performance threshold violations detected:")
            for warning in warnings:
                logger.warning(f"  ⚠️  {warning}")
        
        # Delegate to visualizer if available
        if self.visualizer:
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


    def run_simulation(self, show_animation: bool = True) -> None:
        """
        Run simulation with a per-instance structured config sandbox.

        Args:
            show_animation: Whether to display animation during simulation
        """
        # Use SimulationLoop to handle all loop logic
        loop = SimulationLoop(self)
        with use_structured_config(self.structured_config.clone()):
            return loop.run(show_animation=show_animation, structured_config=self.structured_config)

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
