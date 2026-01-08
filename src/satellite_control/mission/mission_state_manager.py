"""
Mission State Manager for Satellite Control System

Centralized mission logic for the simulation.
Provides unified mission state transitions and target calculations for all
mission types.

Mission types supported:
1. Waypoint Navigation: Navigate to single/multiple waypoints with rotation
2. Shape Following: Follow geometric paths (circles, rectangles, etc.)

Key features:
- Unified state machine for mission progression
- Position and orientation tolerance checking
- Target calculation and waypoint management
- DXF shape import and path generation
- Mission completion detection
- Spline-based smooth obstacle avoidance
"""

from typing import Callable, List, Optional, Tuple

try:
    from src.satellite_control.utils.spline_path import (
        ObstacleAvoidanceSpline,
        create_obstacle_avoidance_spline,
    )

    SPLINE_AVAILABLE = True
except ImportError:
    SPLINE_AVAILABLE = False

import logging

import numpy as np

# V4.0.0: SatelliteConfig removed - use SimulationConfig/AppConfig/MissionState only
from src.satellite_control.config.mission_state import MissionState
from src.satellite_control.config.models import AppConfig
from src.satellite_control.utils.orientation_utils import (
    euler_xyz_to_quat_wxyz,
    quat_angle_error,
)

logger = logging.getLogger(__name__)


class MissionStateManager:
    """Manages mission state transitions and target calculations.

    This class provides a unified implementation of mission logic for both
    simulation and real hardware control systems.
    """

    def __init__(
        self,
        mission_state: MissionState,  # V4.0.0: Required, no fallback
        app_config: "AppConfig",  # V4.0.0: Required, no fallback
        position_tolerance: float = 0.05,
        angle_tolerance: float = 0.05,
        normalize_angle_func: Optional[Callable[[float], float]] = None,
        angle_difference_func: Optional[Callable[[float, float], float]] = None,
        point_to_line_distance_func: Optional[
            Callable[[np.ndarray, np.ndarray, np.ndarray], float]
        ] = None,
    ):
        """
        Initialize mission state manager.

        Args:
            position_tolerance: Position error tolerance in meters
            angle_tolerance: Angle error tolerance in radians
            normalize_angle_func: Function to normalize angles to [-pi, pi]
            angle_difference_func: Function to calculate angle difference
            point_to_line_distance_func: Point-to-line distance function
            mission_state: MissionState (required in V4.0.0)
            app_config: AppConfig for timing parameters (required in V4.0.0)
        """
        if mission_state is None:
            raise ValueError("mission_state is required (V4.0.0: no SatelliteConfig fallback)")
        if app_config is None:
            raise ValueError("app_config is required (V4.0.0: no SatelliteConfig fallback)")
        
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance

        # Store helper functions
        self.normalize_angle = normalize_angle_func or self._default_normalize_angle
        self.angle_difference = angle_difference_func or self._default_angle_difference
        self.point_to_line_distance = (
            point_to_line_distance_func or self._default_point_to_line_distance
        )
        
        # V4.0.0: Required, no fallback to SatelliteConfig
        self.mission_state = mission_state
        self.app_config = app_config

        # Mission state tracking
        self.current_nav_waypoint_idx: int = 0
        self.nav_target_reached_time: Optional[float] = None

        self.dxf_completed: bool = False
        self.multi_point_target_reached_time: Optional[float] = None

        self.shape_stabilization_start_time: Optional[float] = None
        self.return_stabilization_start_time: Optional[float] = None
        self.final_waypoint_stabilization_start_time: Optional[float] = None
        # Obstacle Avoidance State
        self.obstacle_waypoints: List[Tuple[float, float]] = []
        self.current_obstacle_idx: int = 0
        self.last_target_index: int = -1
        self.obstacle_waypoint_reached_time: Optional[float] = None

        # Spline-based obstacle avoidance (new)
        self.avoidance_spline: Optional["ObstacleAvoidanceSpline"] = None
        self.spline_arc_progress: float = 0.0
        self.spline_cruise_speed: float = 0.12  # m/s along spline
        self._active_obstacle: Optional[Tuple[float, float, float]] = None
    
    def _update_dxf_phase(self, phase: str) -> None:
        """Update DXF shape phase in mission_state (V4.0.0: no SatelliteConfig fallback)."""
        self.mission_state.dxf_shape_phase = phase
    
    def _get_dxf_phase(self) -> str:
        """Get DXF shape phase from mission_state (V4.0.0: required)."""
        return self.mission_state.dxf_shape_phase or "POSITIONING"
    
    def _get_obstacles_enabled(self) -> bool:
        """Get obstacles enabled status from mission_state (V4.0.0: required)."""
        return self.mission_state.obstacles_enabled
    
    def _get_obstacles(self) -> List[Tuple[float, float, float]]:
        """Get obstacles list from mission_state (V4.0.0: required)."""
        return self.mission_state.obstacles
    
    def _get_current_waypoint_target(self) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """
        Get current waypoint target from mission_state (V4.0.0: required).
        
        Returns:
            Tuple of (target_pos, target_angle) or (None, None)
        """
        targets = self.mission_state.waypoint_targets or self.mission_state.multi_point_targets
        angles = self.mission_state.waypoint_angles or self.mission_state.multi_point_angles
        
        if not targets or self.mission_state.current_target_index >= len(targets):
            return None, None
        
        target_pos = targets[self.mission_state.current_target_index]
        # Handle angle format (could be single float or tuple)
        target_angle_raw = angles[self.mission_state.current_target_index]
        if isinstance(target_angle_raw, (int, float)):
            target_angle = (0.0, 0.0, float(target_angle_raw))
        else:
            target_angle = target_angle_raw
        
        return target_pos, target_angle
    
    def _advance_to_next_target(self) -> bool:
        """
        Advance to next waypoint target in mission_state (V4.0.0: required).
        
        Returns:
            True if advanced, False if all targets completed
        """
        targets = self.mission_state.waypoint_targets or self.mission_state.multi_point_targets
        if not targets:
            return False
        
        self.mission_state.current_target_index += 1
        self.mission_state.target_stabilization_start_time = None
        
        if self.mission_state.current_target_index >= len(targets):
            return False
        return True
    
    def _get_current_target_index(self) -> int:
        """Get current target index from mission_state (V4.0.0: required)."""
        return self.mission_state.current_target_index
    
    def _get_waypoint_targets(self) -> List[Tuple[float, float, float]]:
        """Get waypoint targets from mission_state (V4.0.0: required)."""
        return self.mission_state.waypoint_targets or self.mission_state.multi_point_targets
    
    def _get_dxf_shape_path(self) -> List[Tuple[float, float, float]]:
        """Get DXF shape path from mission_state (V4.0.0: required)."""
        return self.mission_state.dxf_shape_path
    
    def _get_dxf_target_speed(self) -> float:
        """Get DXF target speed from mission_state (V4.0.0: required)."""
        return self.mission_state.dxf_target_speed
    
    def _get_dxf_closest_point_index(self) -> int:
        """Get DXF closest point index from mission_state (V4.0.0: required)."""
        return self.mission_state.dxf_closest_point_index
    
    def _get_dxf_path_length(self) -> float:
        """Get DXF path length from mission_state (V4.0.0: required)."""
        return self.mission_state.dxf_path_length
    
    def _get_dxf_tracking_start_time(self) -> Optional[float]:
        """Get DXF tracking start time from mission_state (V4.0.0: required)."""
        return self.mission_state.dxf_tracking_start_time
    
    def _get_dxf_positioning_start_time(self) -> Optional[float]:
        """Get DXF positioning start time from mission_state (V4.0.0: required)."""
        return getattr(self.mission_state, "dxf_positioning_start_time", None)
    
    def _set_dxf_positioning_start_time(self, value: float) -> None:
        """Set DXF positioning start time in mission_state (V4.0.0: no SatelliteConfig)."""
        setattr(self.mission_state, "dxf_positioning_start_time", value)
    
    def _get_dxf_current_target_position(self) -> Optional[Tuple[float, float]]:
        """Get DXF current target position from mission_state (V4.0.0: required)."""
        return getattr(self.mission_state, "dxf_current_target_position", None)
    
    def _set_dxf_current_target_position(self, value: Optional[Tuple[float, float]]) -> None:
        """Set DXF current target position in mission_state (V4.0.0: no SatelliteConfig)."""
        setattr(self.mission_state, "dxf_current_target_position", value)
    
    def _get_dxf_final_position(self) -> Optional[Tuple[float, float]]:
        """Get DXF final position from mission_state (V4.0.0: required)."""
        return self.mission_state.dxf_final_position
    
    def _set_dxf_final_position(self, value: Optional[Tuple[float, float]]) -> None:
        """Set DXF final position in mission_state (V4.0.0: no SatelliteConfig)."""
        self.mission_state.dxf_final_position = value
    
    def _get_dxf_stabilization_start_time(self) -> Optional[float]:
        """Get DXF stabilization start time from mission_state (V4.0.0: required)."""
        return self.mission_state.dxf_stabilization_start_time
    
    def _set_dxf_stabilization_start_time(self, value: Optional[float]) -> None:
        """Set DXF stabilization start time in mission_state (V4.0.0: no SatelliteConfig)."""
        self.mission_state.dxf_stabilization_start_time = value
    
    def _get_dxf_target_start_distance(self) -> float:
        """Get DXF target start distance from mission_state (V4.0.0: required)."""
        return getattr(self.mission_state, "dxf_target_start_distance", 0.0)
    
    def _set_dxf_target_start_distance(self, value: float) -> None:
        """Set DXF target start distance in mission_state (V4.0.0: no SatelliteConfig)."""
        setattr(self.mission_state, "dxf_target_start_distance", value)
    
    def _get_dxf_has_return(self) -> bool:
        """Get DXF has return flag from mission_state (V4.0.0: required)."""
        return self.mission_state.dxf_has_return
    
    def _get_waypoint_final_stabilization_time(self) -> float:
        """Get waypoint final stabilization time from app_config (V4.0.0: required)."""
        return self.app_config.simulation.waypoint_final_stabilization_time
    
    def _get_target_hold_time(self) -> float:
        """Get target hold time from app_config (V4.0.0: required)."""
        return self.app_config.simulation.target_hold_time
    
    def _get_shape_positioning_stabilization_time(self) -> float:
        """Get shape positioning stabilization time from app_config (V4.0.0: required)."""
        return self.app_config.simulation.shape_positioning_stabilization_time
    
    def _get_shape_final_stabilization_time(self) -> float:
        """Get shape final stabilization time from app_config (V4.0.0: required)."""
        return self.app_config.simulation.shape_final_stabilization_time
    
    def _get_control_dt(self) -> float:
        """Get control dt from app_config (V4.0.0: required)."""
        return self.app_config.simulation.control_dt
    
    def _set_multi_point_phase(self, phase: str) -> None:
        """Set multi-point phase in mission_state (V4.0.0: no SatelliteConfig)."""
        self.mission_state.multi_point_phase = phase
    
    def _get_waypoint_targets_old(self) -> List[Tuple[float, float, float]]:
        """Get waypoint targets from mission_state (V4.0.0: required, legacy method name)."""
        return self.mission_state.waypoint_targets or self.mission_state.multi_point_targets

    def _create_3d_state(
        self,
        x: float,
        y: float,
        orientation: Tuple[float, float, float],
        vx: float = 0.0,
        vy: float = 0.0,
        omega: float = 0.0,
        z: float = 0.0,
        vz: float = 0.0,
    ) -> np.ndarray:
        """Helper to create 13-element 3D state from position and Euler angles."""
        state = np.zeros(13)
        state[0] = x
        state[1] = y
        state[2] = z

        quat = euler_xyz_to_quat_wxyz(orientation)
        state[3:7] = quat

        state[7] = vx
        state[8] = vy
        state[9] = vz

        state[12] = omega
        return state

    @staticmethod
    def _yaw_to_euler(yaw: float) -> Tuple[float, float, float]:
        return (0.0, 0.0, float(yaw))

    @staticmethod
    def _format_euler_deg(angle: Tuple[float, float, float]) -> str:
        roll, pitch, yaw = np.degrees(angle)
        return f"roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°"

    def get_trajectory(
        self,
        current_time: float,
        dt: float,
        horizon: int,
        current_state: np.ndarray,
        external_target_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate a reference trajectory for the MPC prediction horizon.

        Args:
            current_time: Current simulation time
            dt: Control timestep seconds
            horizon: Number of steps to predict (N)
            current_state: Current satellite state [pos(3), quat(4), vel(3), w(3)]
            external_target_state: Optional manual target 13-element array

        Returns:
            trajectory: Numpy array of shape (horizon+1, 13)
        """
        # Initialize trajectory array
        trajectory = np.zeros((horizon + 1, 13))

        current_quat = current_state[3:7]

        # Determine active mode (V4.0.0: mission_state required)
        is_dxf = self.mission_state.dxf_shape_mode_active

        # 3D Position for internal logic
        curr_pos_3d = current_state[:3]

        if is_dxf:
            # --- SHAPE FOLLOWING PREDICTION ---
            path = self.mission_state.dxf_shape_path
            phase = self.mission_state.dxf_shape_phase
            start_time = self.mission_state.dxf_tracking_start_time
            speed = self.mission_state.dxf_target_speed
            path_len = max(self.mission_state.dxf_path_length, 1e-9)
            closest_point_idx = self.mission_state.dxf_closest_point_index

            if phase == "TRACKING" and start_time is not None:

                from src.satellite_control.mission.mission_manager import (
                    get_path_tangent_orientation,
                    get_position_on_path,
                )

                for k in range(horizon + 1):
                    future_time = current_time + k * dt
                    tracking_time = future_time - start_time
                    distance = speed * tracking_time

                    if distance >= path_len:
                        current_path_position, _ = get_position_on_path(
                            path,
                            path_len,
                            closest_point_idx,
                        )
                        target_orientation = get_path_tangent_orientation(
                            path,
                            path_len,
                            closest_point_idx,
                        )
                        trajectory[k] = self._create_3d_state(
                            current_path_position[0],
                            current_path_position[1],
                            self._yaw_to_euler(target_orientation),
                        )
                    else:
                        wrapped_s = distance % path_len
                        pos, _ = get_position_on_path(
                            path,
                            wrapped_s,
                            closest_point_idx,
                        )
                        orient = get_path_tangent_orientation(
                            path,
                            wrapped_s,
                            closest_point_idx,
                        )
                        vx = speed * np.cos(orient)
                        vy = speed * np.sin(orient)
                        trajectory[k] = self._create_3d_state(
                            pos[0], pos[1], self._yaw_to_euler(orient), vx, vy
                        )
            else:
                target = self.update_target_state(
                    curr_pos_3d, current_quat, current_time, current_state
                )
                if target is not None:
                    trajectory[:] = target
                elif external_target_state is not None:
                    trajectory[:] = external_target_state
                else:
                    trajectory[:] = current_state
                    trajectory[:, 7:] = 0  # Zero velocities

        else:
            # --- STANDARD WAYPOINT ---
            target = self.update_target_state(curr_pos_3d, current_quat, current_time, current_state)

            if target is not None:
                trajectory[:] = target

                # Obstacle Avoidance Spline Prediction
                if self.avoidance_spline is not None:
                    progress = self.spline_arc_progress
                    speed = self.spline_cruise_speed

                    for k in range(horizon + 1):
                        future_progress = progress + (k * dt * speed)

                        if self.avoidance_spline.is_complete(future_progress):
                            trajectory[k] = target
                        else:
                            s_pos = self.avoidance_spline.evaluate(future_progress)
                            s_tan = self.avoidance_spline.tangent(future_progress)
                            vx = s_tan[0] * speed
                            vy = s_tan[1] * speed
                            trajectory[k] = self._create_3d_state(
                                s_pos[0], s_pos[1], (0.0, 0.0, 0.0), vx, vy
                            )

            elif external_target_state is not None:
                trajectory[:] = external_target_state
            else:
                trajectory[:] = current_state
                trajectory[:, 7:] = 0

        return trajectory

    def _has_clear_path_to_target(
        self, current_pos: np.ndarray, target_pos: Tuple[float, float]
    ) -> bool:
        """Check if direct path to target is clear of the active obstacle."""
        if self._active_obstacle is None:
            return True

        obs_x, obs_y, obs_radius = self._active_obstacle
        # V4.0.0: Use hardcoded safety margin (could be moved to AppConfig in future)
        safety = 0.5

        # Calculate distance from obstacle center to path line
        start = current_pos[:2]
        end = np.array(target_pos)
        obstacle = np.array([obs_x, obs_y])

        path_vec = end - start
        path_length = np.linalg.norm(path_vec)
        if path_length < 0.01:
            return True

        path_dir = path_vec / path_length
        to_obstacle = obstacle - start
        projection = np.dot(to_obstacle, path_dir)

        # Obstacle behind us or past target
        if projection < 0 or projection > path_length:
            return True

        # Perpendicular distance to path
        closest = start + projection * path_dir
        dist_to_path = np.linalg.norm(obstacle - closest)

        return bool(dist_to_path > (obs_radius + safety))

    @staticmethod
    def _default_normalize_angle(angle: float) -> float:
        """Default angle normalization to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def _default_angle_difference(angle1: float, angle2: float) -> float:
        """Default angle difference calculation."""
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    @staticmethod
    def _default_point_to_line_distance(
        point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
    ) -> float:
        """Default point-to-line distance calculation."""
        line_vec = line_end - line_start
        line_length_sq = np.dot(line_vec, line_vec)

        if line_length_sq == 0:
            return float(np.linalg.norm(point - line_start))

        point_vec = point - line_start
        t = np.dot(point_vec, line_vec) / line_length_sq
        t = max(0, min(1, t))

        closest_point = line_start + t * line_vec
        return float(np.linalg.norm(point - closest_point))

    def update_target_state(
        self,
        current_position: np.ndarray,
        current_quat: np.ndarray,
        current_time: float,
        current_state: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Update target state based on active mission mode.

        Args:
            current_position: Current satellite position [x, y, z]
            current_quat: Current satellite orientation quaternion [w, x, y, z]
            current_time: Current simulation/control time in seconds
            current_state: Full state vector [pos(3), quat(4), vel(3), w(3)]

        Returns:
            Target state vector [pos(3), quat(4), vel(3), w(3)] or None
        """
        # Waypoint mode (V4.0.0: mission_state required)
        enable_waypoint = (
            self.mission_state.enable_waypoint_mode
            or self.mission_state.enable_multi_point_mode
        )
        
        if enable_waypoint:
            return self._handle_multi_point_mode(current_position, current_quat, current_time)

        # DXF shape mode (V4.0.0: mission_state required)
        is_dxf = self.mission_state.dxf_shape_mode_active
        
        if is_dxf:
            return self._handle_dxf_shape_mode(current_position, current_quat, current_time)

        # Point-to-point mode (no-op, handled by caller)
        return None

    def _handle_multi_point_mode(
        self,
        current_position: np.ndarray,
        current_quat: np.ndarray,
        current_time: float,
    ) -> Optional[np.ndarray]:
        """Handle waypoint sequential navigation mode."""
        final_target_pos, final_target_angle = self._get_current_waypoint_target()
        if final_target_pos is None:
            return None

        # --- Obstacle Avoidance Logic ---
        current_target_index = self._get_current_target_index()

        # Check if target changed or we need to clear old path
        if current_target_index != self.last_target_index:
            self.obstacle_waypoints = []
            self.current_obstacle_idx = 0
            self.last_target_index = current_target_index
            self.obstacle_waypoint_reached_time = None

        target_pos = final_target_pos
        target_orientation = final_target_angle
        target_vx, target_vy = 0.0, 0.0
        using_spline = False

        # SPLINE-BASED OBSTACLE AVOIDANCE with moving reference
        obstacles_enabled = self._get_obstacles_enabled()
        obstacles = self._get_obstacles()
        if obstacles_enabled and obstacles and SPLINE_AVAILABLE:
            # Generate spline if needed (first time through obstacle zone)
            if self.avoidance_spline is None:
                # Try to create spline for first blocking obstacle
                for obs_x, obs_y, obs_radius in obstacles:
                    spline = create_obstacle_avoidance_spline(
                        start_pos=current_position,
                        target_pos=np.array(final_target_pos),
                        obstacle_center=np.array([obs_x, obs_y]),
                        obstacle_radius=obs_radius,
                        safety_margin=0.5,  # V4.0.0: Hardcoded (could be in AppConfig)
                    )
                    if spline is not None:
                        self.avoidance_spline = spline
                        self.spline_arc_progress = 0.0
                        self._active_obstacle = (obs_x, obs_y, obs_radius)
                        break

            # If we have an active spline, track along it
            if self.avoidance_spline is not None:
                # Check if we now have a clear path to final target
                clear_path = self._has_clear_path_to_target(current_position, final_target_pos)

                if clear_path:
                    # Exit spline early - we have clear line of sight
                    self.avoidance_spline = None
                    self.spline_arc_progress = 0.0
                    self._active_obstacle = None
                    using_spline = False
                else:
                    using_spline = True

                    # Advance progress based on control timestep and cruise
                    # speed
                    dt = self._get_control_dt()
                    self.spline_arc_progress += self.spline_cruise_speed * dt

                    # Check if spline is complete
                    if self.avoidance_spline.is_complete(self.spline_arc_progress):
                        # Spline complete - transition to final target
                        self.avoidance_spline = None
                        self.spline_arc_progress = 0.0
                        self._active_obstacle = None
                        using_spline = False
                    else:
                        # Get moving reference point on spline
                        target_pos = self.avoidance_spline.evaluate(self.spline_arc_progress)

                        # Get tangent for velocity direction
                        tangent = self.avoidance_spline.tangent(self.spline_arc_progress)
                        target_vx = tangent[0] * self.spline_cruise_speed
                        target_vy = tangent[1] * self.spline_cruise_speed

                        # Keep neutral angle during spline traversal
                        target_orientation = (0.0, 0.0, 0.0)

        target_state = self._create_3d_state(
            target_pos[0],
            target_pos[1],
            target_orientation,
            target_vx,
            target_vy,
            0.0,
            target_pos[2],
        )

        pos_error = np.linalg.norm(current_position - np.array(target_pos))
        ang_error = quat_angle_error(target_state[3:7], current_quat)

        # Only count as "REACHED" if NOT on spline (targeting final
        # destination)
        is_final_approach = not using_spline

        if (
            is_final_approach
            and pos_error < self.position_tolerance
            and ang_error < self.angle_tolerance
        ):
            if self.multi_point_target_reached_time is None:
                self.multi_point_target_reached_time = current_time
                logger.info(
                    f" TARGET {self._get_current_target_index() + 1} " "REACHED! Stabilizing..."
                )
            else:
                targets = self._get_waypoint_targets()
                is_final_target = (
                    self._get_current_target_index() >= len(targets) - 1
                )

                if is_final_target:
                    required_hold_time = self._get_waypoint_final_stabilization_time()
                else:
                    required_hold_time = self._get_target_hold_time()

                maintenance_time = current_time - self.multi_point_target_reached_time
                if maintenance_time >= required_hold_time:
                    # Advance to next target
                    next_available = self._advance_to_next_target()
                    if next_available:
                        (
                            new_target_pos,
                            new_target_angle,
                        ) = self._get_current_waypoint_target()
                        idx = self._get_current_target_index() + 1
                        px, py = new_target_pos[0], new_target_pos[1]
                        p_z = new_target_pos[2]
                        logger.info(
                            f" MOVING TO NEXT TARGET {idx}: "
                            f"({px:.2f}, {py:.2f}, {p_z:.2f}) m, "
                            f"{self._format_euler_deg(new_target_angle)}"
                        )
                        self.multi_point_target_reached_time = None
                        # Reset obstacle path for next target (will happen
                        # automatically at top of loop)
                    else:
                        # All targets completed
                        self._set_multi_point_phase("COMPLETE")
                        logger.info(" ALL WAYPOINTS REACHED! Final stabilization phase.")
                        return None  # Signal mission complete
        else:
            self.multi_point_target_reached_time = None

        return target_state

    def _handle_dxf_shape_mode(
        self,
        current_position: np.ndarray,
        current_quat: np.ndarray,
        current_time: float,
    ) -> Optional[np.ndarray]:
        """Handle DXF shape following mode."""
        # Import Mission functions if available
        try:
            from src.satellite_control.mission.mission_manager import (  # noqa: F401
                get_path_tangent_orientation,
                get_position_on_path,
            )
        except ImportError:
            logger.info(" Mission module not available for DXF shape mode")
            return None

        # Initialize mission start time (V4.0.0: mission_state required)
        path = self.mission_state.dxf_shape_path
        mission_start_time = self.mission_state.dxf_mission_start_time
        
        if mission_start_time is None:
            self.mission_state.dxf_mission_start_time = current_time
            min_dist = float("inf")
            closest_idx = 0
            closest_point = path[0]

            for i, point in enumerate(path):
                dist = float(np.linalg.norm(current_position - np.array(point)))
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    closest_point = point

            self.mission_state.dxf_closest_point_index = closest_idx
            self.mission_state.dxf_current_target_position = closest_point

            # Calculate total path length
            total_length: float = 0.0
            for i in range(len(path)):
                idx = (closest_idx + i) % len(path)
                next_idx = (closest_idx + i + 1) % len(path)
                total_length += float(
                    np.linalg.norm(np.array(path[next_idx]) - np.array(path[idx]))
                )

            self.mission_state.dxf_path_length = total_length

            logger.info(f" PROFILE FOLLOWING MISSION STARTED at t={current_time:.2f}s")
            cx, cy = closest_point[0], closest_point[1]
            logger.info(f"   Phase 1: Moving to closest point on path " f"({cx:.3f}, {cy:.3f})")
            logger.info(f" Profile path length: {total_length:.3f} m")

        # Get path and phase from mission_state (V4.0.0: required)
        dxf_path = self.mission_state.dxf_shape_path
        phase = self._get_dxf_phase()

        # Phase 1: POSITIONING
        if phase == "POSITIONING":
            return self._dxf_positioning_phase(
                current_position, current_quat, current_time, dxf_path
            )

        # Phase 2: TRACKING
        elif phase == "TRACKING":
            return self._dxf_tracking_phase(current_position, current_time, dxf_path)

        # Phase 3: PATH_STABILIZATION (at path waypoints)
        elif phase == "PATH_STABILIZATION":
            return self._dxf_path_stabilization_phase(
                current_position, current_quat, current_time, dxf_path
            )

        # Phase 4: STABILIZING
        elif phase == "STABILIZING":
            return self._dxf_stabilizing_phase(current_time, dxf_path)

        # Phase 5: RETURNING
        elif phase == "RETURNING":
            return self._dxf_returning_phase(current_position, current_quat, current_time)

        return None

    def _dxf_positioning_phase(
        self,
        current_position: np.ndarray,
        current_quat: np.ndarray,
        current_time: float,
        path: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """Handle DXF positioning phase."""
        from src.satellite_control.mission.mission_manager import (
            get_path_tangent_orientation,
        )

        # Set phase start time on first entry
        if self._get_dxf_positioning_start_time() is None:
            self._set_dxf_positioning_start_time(current_time)

        target_pos = self._get_dxf_current_target_position()
        if target_pos is None:
            return None
        target_orientation = get_path_tangent_orientation(
            path, 0.0, self._get_dxf_closest_point_index()
        )

        target_state = self._create_3d_state(
            target_pos[0],
            target_pos[1],
            self._yaw_to_euler(target_orientation),
            0.0,
            0.0,
            0.0,
        )

        pos_error = np.linalg.norm(current_position - np.array(target_pos))
        ang_error = quat_angle_error(target_state[3:7], current_quat)

        if pos_error < self.position_tolerance and ang_error < self.angle_tolerance:
            if self.shape_stabilization_start_time is None:
                self.shape_stabilization_start_time = current_time
                self._update_dxf_phase("PATH_STABILIZATION")
                self.mission_state.dxf_stabilization_start_time = current_time
                stab_time = self._get_shape_positioning_stabilization_time()
                logger.info(
                    f" Reached starting position, stabilizing for "
                    f"{stab_time:.1f}s..."
                )
            else:
                stabilization_time = current_time - self.shape_stabilization_start_time
                stab_time = self._get_shape_positioning_stabilization_time()
                if stabilization_time >= stab_time:
                    self._update_dxf_phase("TRACKING")
                    self.mission_state.dxf_tracking_start_time = current_time
                    self.mission_state.dxf_target_start_distance = 0.0
                    self._set_dxf_target_start_distance(0.0)
                    logger.info(" Satellite stable! Starting profile tracking...")
                    speed = self._get_dxf_target_speed()
                    logger.info(f"   Target speed: {speed:.2f} m/s")
        else:
            self.shape_stabilization_start_time = None

        return target_state

    def _dxf_tracking_phase(
        self,
        current_position: np.ndarray,
        current_time: float,
        path: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """Handle DXF tracking phase."""
        from src.satellite_control.mission.mission_manager import (
            get_path_tangent_orientation,
            get_position_on_path,
        )

        tracking_start = self._get_dxf_tracking_start_time()
        if tracking_start is None:
            return None
        tracking_time = current_time - tracking_start
        distance_traveled = self._get_dxf_target_speed() * tracking_time
        path_len = max(self._get_dxf_path_length(), 1e-9)

        if distance_traveled >= path_len:
            # Path complete
            current_path_position, _ = get_position_on_path(
                path, path_len, self._get_dxf_closest_point_index()
            )
            has_return = self._get_dxf_has_return()
            if has_return:
                # Start path stabilization phase at final waypoint before
                # returning
                if self.final_waypoint_stabilization_start_time is None:
                    self.final_waypoint_stabilization_start_time = current_time
                    self._update_dxf_phase("PATH_STABILIZATION")
                    self.mission_state.dxf_final_position = current_path_position
                    self._set_dxf_final_position(current_path_position)
                    stab_time = self._get_shape_positioning_stabilization_time()
                    logger.info(
                        f" Stabilizing at final waypoint for {stab_time:.1f} "
                        "seconds before return..."
                    )
                # Return None to let next update handle PATH_STABILIZATION
                # phase
                return None
            else:
                self._update_dxf_phase("STABILIZING")
                if self.mission_state is not None:
                    self.mission_state.dxf_final_position = current_path_position
                # Backward compatibility
                self._set_dxf_stabilization_start_time(current_time)
                self._set_dxf_final_position(current_path_position)
                logger.info(" Profile traversal completed! Stabilizing at final position...")
                return None
        else:
            # Continue tracking
            wrapped_s = distance_traveled % path_len
            current_path_position, _ = get_position_on_path(
                path, wrapped_s, self._get_dxf_closest_point_index()
            )
            self._set_dxf_current_target_position(current_path_position)

            target_orientation = get_path_tangent_orientation(
                path, wrapped_s, self._get_dxf_closest_point_index()
            )

            return self._create_3d_state(
                current_path_position[0],
                current_path_position[1],
                self._yaw_to_euler(target_orientation),
                0.0,
                0.0,
                0.0,
            )

    def _dxf_path_stabilization_phase(
        self,
        current_position: np.ndarray,
        current_quat: np.ndarray,
        current_time: float,
        path: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """Handle DXF path stabilization phase - stabilizing at path waypoints (start or end)."""
        from src.satellite_control.mission.mission_manager import (
            get_path_tangent_orientation,
        )

        # Determine if we're stabilizing at start or end based on which timer is active
        # and whether DXF_FINAL_POSITION has been set (V4.0.0: use mission_state)
        is_end_stabilization = self.mission_state.dxf_final_position is not None

        if is_end_stabilization:
            # Stabilizing at END of path (before returning)
            target_pos = self.mission_state.dxf_final_position
            path_s = self.mission_state.dxf_path_length
        else:
            # Stabilizing at START of path (before tracking begins)
            target_pos = getattr(self.mission_state, "dxf_current_target_position", None)
            path_s = 0.0

        if target_pos is None:
            return None

        target_orientation = get_path_tangent_orientation(
            path, path_s, self.mission_state.dxf_closest_point_index
        )

        target_state = self._create_3d_state(
            target_pos[0],
            target_pos[1],
            self._yaw_to_euler(target_orientation),
            0.0,
            0.0,
            0.0,
        )

        pos_error = float(np.linalg.norm(current_position - np.array(target_pos)))
        ang_error = quat_angle_error(target_state[3:7], current_quat)

        if pos_error < self.position_tolerance and ang_error < self.angle_tolerance:
            if is_end_stabilization:
                # END stabilization logic
                if self.final_waypoint_stabilization_start_time is None:
                    self.final_waypoint_stabilization_start_time = current_time
                    logger.info(
                        " Satellite reached final waypoint. Stabilizing for "
                        f"{self.app_config.simulation.shape_positioning_stabilization_time:.1f}s "
                        "before return..."
                    )
                else:
                    stabilization_time = current_time - self.final_waypoint_stabilization_start_time
                    stab_time = self._get_shape_positioning_stabilization_time()
                    if stabilization_time >= stab_time:
                        # V4.0.0: Update mission_state only (no SatelliteConfig)
                        self.mission_state.dxf_shape_phase = "RETURNING"
                        return_pos = getattr(self.mission_state, "dxf_return_position", None)
                        logger.info(" Path stabilization complete!")
                        if return_pos is not None:
                            rx, ry = return_pos[0], return_pos[1]
                            logger.info(f" Starting return to position ({rx:.2f}, {ry:.2f}) m")
                        self.final_waypoint_stabilization_start_time = None
                        return None
            else:
                # START stabilization logic
                if self.shape_stabilization_start_time is None:
                    self.shape_stabilization_start_time = current_time
                    stab_time = self._get_shape_positioning_stabilization_time()
                    logger.info(
                        f" Satellite reached path start. Stabilizing for "
                        f"{stab_time:.1f} seconds before tracking..."
                    )
                else:
                    stabilization_time = current_time - self.shape_stabilization_start_time
                stab_time = self._get_shape_positioning_stabilization_time()
                if stabilization_time >= stab_time:
                    self._update_dxf_phase("TRACKING")
                    if self.mission_state is not None:
                        self.mission_state.dxf_tracking_start_time = current_time
                        self.mission_state.dxf_target_start_distance = 0.0
                        self._set_dxf_target_start_distance(0.0)
                    logger.info(" Path stabilization complete! Starting profile tracking...")
                    logger.info(f"   Target speed: {self._get_dxf_target_speed():.2f} m/s")
                    self.shape_stabilization_start_time = None
        else:
            # Reset stabilization timer if satellite drifts away
            if is_end_stabilization:
                self.final_waypoint_stabilization_start_time = None
            else:
                self.shape_stabilization_start_time = None

        return target_state

    def _dxf_stabilizing_phase(
        self, current_time: float, path: List[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """Handle DXF stabilizing phase."""
        from src.satellite_control.mission.mission_manager import (
            get_path_tangent_orientation,
        )

        # V4.0.0: Use mission_state only
        final_pos = self.mission_state.dxf_final_position
        if final_pos is None:
            return None

        # Determine target orientation: use return angle if at return position,
        # otherwise use path tangent
        has_return = self.mission_state.dxf_has_return
        return_pos = getattr(self.mission_state, "dxf_return_position", None)
        at_return_position = (
            has_return
            and return_pos is not None
            and final_pos is not None
            and np.allclose(final_pos, return_pos, atol=0.001)
        )

        if at_return_position:
            # Stabilizing at return position - use return angle
            target_orientation = getattr(
                self.mission_state, "dxf_return_angle", (0.0, 0.0, 0.0)
            ) or (0.0, 0.0, 0.0)
        else:
            # Stabilizing at end of path - use path tangent
            end_s = self.mission_state.dxf_path_length
            target_orientation = get_path_tangent_orientation(
                path, end_s, self.mission_state.dxf_closest_point_index
            )

        target_state = self._create_3d_state(
            final_pos[0],
            final_pos[1],
            self._yaw_to_euler(target_orientation)
            if not at_return_position
            else target_orientation,
            0.0,
            0.0,
            0.0,
        )

        if self._get_dxf_stabilization_start_time() is None:
            self._set_dxf_stabilization_start_time(current_time)

        stab_start = self._get_dxf_stabilization_start_time()
        stabilization_time = current_time - stab_start if stab_start else 0.0
        stab_time = self._get_shape_final_stabilization_time()
        if stabilization_time >= stab_time:
            logger.info(" PROFILE FOLLOWING MISSION COMPLETED!")
            logger.info("   Profile successfully traversed and stabilized")
            self.dxf_completed = True
            return None  # Signal mission complete

        return target_state

    def _dxf_returning_phase(
        self,
        current_position: np.ndarray,
        current_quat: np.ndarray,
        current_time: float,
    ) -> Optional[np.ndarray]:
        """Handle DXF returning phase (V4.0.0: use mission_state only)."""
        return_pos = getattr(self.mission_state, "dxf_return_position", (0.0, 0.0))
        return_angle = getattr(
            self.mission_state, "dxf_return_angle", (0.0, 0.0, 0.0)
        ) or (0.0, 0.0, 0.0)

        target_state = self._create_3d_state(
            return_pos[0], return_pos[1], return_angle, 0.0, 0.0, 0.0
        )

        pos_error = float(np.linalg.norm(current_position - np.array(return_pos)))
        ang_error = quat_angle_error(target_state[3:7], current_quat)

        # Only check position error for transition, not angle error during
        # movement
        if pos_error < self.position_tolerance:
            # Now enforce angle error once at position
            if ang_error < self.angle_tolerance:
                # Reached return position - transition to STABILIZING phase
                if self.return_stabilization_start_time is None:
                    self.return_stabilization_start_time = current_time
                    self.mission_state.dxf_shape_phase = "STABILIZING"
                    self.mission_state.dxf_stabilization_start_time = current_time
                    self.mission_state.dxf_final_position = return_pos
                    logger.info(
                        " Reached return position! " "Transitioning to final stabilization..."
                    )
                    return None  # Let next update handle STABILIZING phase
            else:
                # Still at position but not at correct angle, keep trying
                self.return_stabilization_start_time = None
        else:
            self.return_stabilization_start_time = None

        return target_state

    def reset(self) -> None:
        """Reset all mission state for a new mission."""
        self.current_nav_waypoint_idx = 0
        self.nav_target_reached_time = None

        self.multi_point_target_reached_time = None

        self.shape_stabilization_start_time = None
        self.return_stabilization_start_time = None
        self.final_waypoint_stabilization_start_time = None

        self.obstacle_waypoints = []
        self.current_obstacle_idx = 0
        self.last_target_index = -1
        self.obstacle_waypoint_reached_time = None
