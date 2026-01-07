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

from src.satellite_control.config import SatelliteConfig
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
            point_to_line_distance_func: Point-to-line distance function
        """
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance

        # Store helper functions
        self.normalize_angle = normalize_angle_func or self._default_normalize_angle
        self.angle_difference = angle_difference_func or self._default_angle_difference
        self.point_to_line_distance = (
            point_to_line_distance_func or self._default_point_to_line_distance
        )

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

        # Determine active mode
        is_dxf = (
            hasattr(SatelliteConfig, "DXF_SHAPE_MODE_ACTIVE")
            and SatelliteConfig.DXF_SHAPE_MODE_ACTIVE
        )

        # 3D Position for internal logic
        curr_pos_3d = current_state[:3]

        if is_dxf:
            # --- SHAPE FOLLOWING PREDICTION ---
            path: List[Tuple[float, float, float]] = SatelliteConfig.DXF_SHAPE_PATH
            phase = getattr(SatelliteConfig, "DXF_SHAPE_PHASE", "POSITIONING")
            start_time = getattr(SatelliteConfig, "DXF_TRACKING_START_TIME", None)

            if phase == "TRACKING" and start_time is not None:
                speed = SatelliteConfig.DXF_TARGET_SPEED
                path_len = max(getattr(SatelliteConfig, "DXF_PATH_LENGTH", 0.0), 1e-9)

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
                            SatelliteConfig.DXF_CLOSEST_POINT_INDEX,
                        )
                        target_orientation = get_path_tangent_orientation(
                            path,
                            path_len,
                            SatelliteConfig.DXF_CLOSEST_POINT_INDEX,
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
                            SatelliteConfig.DXF_CLOSEST_POINT_INDEX,
                        )
                        orient = get_path_tangent_orientation(
                            path,
                            wrapped_s,
                            SatelliteConfig.DXF_CLOSEST_POINT_INDEX,
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
        safety = SatelliteConfig.OBSTACLE_AVOIDANCE_SAFETY_MARGIN

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
        # Waypoint mode
        if (
            hasattr(SatelliteConfig, "ENABLE_WAYPOINT_MODE")
            and SatelliteConfig.ENABLE_WAYPOINT_MODE
        ):
            return self._handle_multi_point_mode(current_position, current_quat, current_time)

        # DXF shape mode
        elif (
            hasattr(SatelliteConfig, "DXF_SHAPE_MODE_ACTIVE")
            and SatelliteConfig.DXF_SHAPE_MODE_ACTIVE
        ):
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
        final_target_pos, final_target_angle = SatelliteConfig.get_current_waypoint_target()
        if final_target_pos is None:
            return None

        # --- Obstacle Avoidance Logic ---
        current_target_index = SatelliteConfig.CURRENT_TARGET_INDEX

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
        if (
            SatelliteConfig.OBSTACLES_ENABLED
            and SatelliteConfig.get_obstacles()
            and SPLINE_AVAILABLE
        ):
            # Generate spline if needed (first time through obstacle zone)
            if self.avoidance_spline is None:
                obstacles = SatelliteConfig.get_obstacles()
                # Try to create spline for first blocking obstacle
                for obs_x, obs_y, obs_radius in obstacles:
                    spline = create_obstacle_avoidance_spline(
                        start_pos=current_position,
                        target_pos=np.array(final_target_pos),
                        obstacle_center=np.array([obs_x, obs_y]),
                        obstacle_radius=obs_radius,
                        safety_margin=SatelliteConfig.OBSTACLE_AVOIDANCE_SAFETY_MARGIN,
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
                    dt = SatelliteConfig.CONTROL_DT
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
                    f" TARGET {SatelliteConfig.CURRENT_TARGET_INDEX + 1} " "REACHED! Stabilizing..."
                )
            else:
                is_final_target = (
                    SatelliteConfig.CURRENT_TARGET_INDEX
                    >= len(SatelliteConfig.WAYPOINT_TARGETS) - 1
                )

                if is_final_target:
                    required_hold_time = SatelliteConfig.WAYPOINT_FINAL_STABILIZATION_TIME
                else:
                    required_hold_time = getattr(SatelliteConfig, "TARGET_HOLD_TIME", 3.0)

                maintenance_time = current_time - self.multi_point_target_reached_time
                if maintenance_time >= required_hold_time:
                    # Advance to next target
                    next_available = SatelliteConfig.advance_to_next_target()
                    if next_available:
                        (
                            new_target_pos,
                            new_target_angle,
                        ) = SatelliteConfig.get_current_waypoint_target()
                        idx = SatelliteConfig.CURRENT_TARGET_INDEX + 1
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
                        SatelliteConfig.MULTI_POINT_PHASE = "COMPLETE"
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

        # Initialize mission start time
        if SatelliteConfig.DXF_MISSION_START_TIME is None:
            SatelliteConfig.DXF_MISSION_START_TIME = current_time  # type: ignore

            # Find closest point on path
            path: List[Tuple[float, float, float]] = SatelliteConfig.DXF_SHAPE_PATH
            min_dist = float("inf")
            closest_idx = 0
            closest_point = path[0]

            for i, point in enumerate(path):
                dist = float(np.linalg.norm(current_position - np.array(point)))
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    closest_point = point

            SatelliteConfig.DXF_CLOSEST_POINT_INDEX = closest_idx
            SatelliteConfig.DXF_CURRENT_TARGET_POSITION = closest_point  # type: ignore

            # Calculate total path length
            total_length: float = 0.0
            for i in range(len(path)):
                idx = (closest_idx + i) % len(path)
                next_idx = (closest_idx + i + 1) % len(path)
                total_length += float(
                    np.linalg.norm(np.array(path[next_idx]) - np.array(path[idx]))
                )

            SatelliteConfig.DXF_PATH_LENGTH = total_length

            logger.info(f" PROFILE FOLLOWING MISSION STARTED at t={current_time:.2f}s")
            cx, cy = closest_point[0], closest_point[1]
            logger.info(f"   Phase 1: Moving to closest point on path " f"({cx:.3f}, {cy:.3f})")
            logger.info(f" Profile path length: {total_length:.3f} m")

        dxf_path: List[Tuple[float, float]] = SatelliteConfig.DXF_SHAPE_PATH
        phase = getattr(SatelliteConfig, "DXF_SHAPE_PHASE", "POSITIONING")

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
        if SatelliteConfig.DXF_POSITIONING_START_TIME is None:
            SatelliteConfig.DXF_POSITIONING_START_TIME = current_time  # type: ignore

        target_pos = getattr(SatelliteConfig, "DXF_CURRENT_TARGET_POSITION", None)
        if target_pos is None:
            return None
        target_orientation = get_path_tangent_orientation(
            path, 0.0, SatelliteConfig.DXF_CLOSEST_POINT_INDEX
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
                SatelliteConfig.DXF_SHAPE_PHASE = "PATH_STABILIZATION"
                SatelliteConfig.DXF_PATH_STABILIZATION_START_TIME = current_time  # type: ignore
                logger.info(
                    f" Reached starting position, stabilizing for "
                    f"{SatelliteConfig.SHAPE_POSITIONING_STABILIZATION_TIME:.1f}s..."
                )
            else:
                stabilization_time = current_time - self.shape_stabilization_start_time
                if stabilization_time >= SatelliteConfig.SHAPE_POSITIONING_STABILIZATION_TIME:
                    SatelliteConfig.DXF_SHAPE_PHASE = "TRACKING"
                    SatelliteConfig.DXF_TRACKING_START_TIME = current_time  # type: ignore
                    SatelliteConfig.DXF_TARGET_START_DISTANCE = 0.0
                    logger.info(" Satellite stable! Starting profile tracking...")
                    logger.info(f"   Target speed: {SatelliteConfig.DXF_TARGET_SPEED:.2f} m/s")
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

        tracking_time = current_time - SatelliteConfig.DXF_TRACKING_START_TIME  # type: ignore
        distance_traveled = SatelliteConfig.DXF_TARGET_SPEED * tracking_time
        path_len = max(getattr(SatelliteConfig, "DXF_PATH_LENGTH", 0.0), 1e-9)

        if distance_traveled >= path_len:
            # Path complete
            current_path_position, _ = get_position_on_path(
                path, path_len, SatelliteConfig.DXF_CLOSEST_POINT_INDEX
            )
            has_return = getattr(SatelliteConfig, "DXF_HAS_RETURN", False)
            if has_return:
                # Start path stabilization phase at final waypoint before
                # returning
                if self.final_waypoint_stabilization_start_time is None:
                    self.final_waypoint_stabilization_start_time = current_time
                    SatelliteConfig.DXF_SHAPE_PHASE = "PATH_STABILIZATION"
                    # Reset for second PATH_STABILIZATION
                    setattr(
                        SatelliteConfig,
                        "DXF_PATH_STABILIZATION_START_TIME",
                        current_time,
                    )
                    setattr(SatelliteConfig, "DXF_FINAL_POSITION", current_path_position)
                    stab_time = SatelliteConfig.SHAPE_POSITIONING_STABILIZATION_TIME
                    logger.info(
                        f" Stabilizing at final waypoint for {stab_time:.1f} "
                        "seconds before return..."
                    )
                # Return None to let next update handle PATH_STABILIZATION
                # phase
                return None
            else:
                setattr(SatelliteConfig, "DXF_STABILIZATION_START_TIME", current_time)
                SatelliteConfig.DXF_SHAPE_PHASE = "STABILIZING"
                setattr(SatelliteConfig, "DXF_FINAL_POSITION", current_path_position)
                logger.info(" Profile traversal completed! Stabilizing at final position...")
                return None
        else:
            # Continue tracking
            wrapped_s = distance_traveled % path_len
            current_path_position, _ = get_position_on_path(
                path, wrapped_s, SatelliteConfig.DXF_CLOSEST_POINT_INDEX
            )
            SatelliteConfig.DXF_CURRENT_TARGET_POSITION = current_path_position  # type: ignore

            target_orientation = get_path_tangent_orientation(
                path, wrapped_s, SatelliteConfig.DXF_CLOSEST_POINT_INDEX
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
        # and whether DXF_FINAL_POSITION has been set
        is_end_stabilization = (
            hasattr(SatelliteConfig, "DXF_FINAL_POSITION")
            and SatelliteConfig.DXF_FINAL_POSITION is not None
        )

        if is_end_stabilization:
            # Stabilizing at END of path (before returning)
            target_pos = SatelliteConfig.DXF_FINAL_POSITION
            path_s = getattr(SatelliteConfig, "DXF_PATH_LENGTH", 0.0)
        else:
            # Stabilizing at START of path (before tracking begins)
            target_pos = SatelliteConfig.DXF_CURRENT_TARGET_POSITION
            path_s = 0.0

        target_orientation = get_path_tangent_orientation(
            path, path_s, SatelliteConfig.DXF_CLOSEST_POINT_INDEX
        )
        target_pos = getattr(SatelliteConfig, "DXF_CURRENT_TARGET_POSITION", None)
        if target_pos is None:
            return None

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
                        f"{SatelliteConfig.SHAPE_POSITIONING_STABILIZATION_TIME:.1f}s "
                        "before return..."
                    )
                else:
                    stabilization_time = current_time - self.final_waypoint_stabilization_start_time
                    if stabilization_time >= SatelliteConfig.SHAPE_POSITIONING_STABILIZATION_TIME:
                        SatelliteConfig.DXF_SHAPE_PHASE = "RETURNING"
                        setattr(SatelliteConfig, "DXF_RETURN_START_TIME", current_time)
                        return_pos = getattr(SatelliteConfig, "DXF_RETURN_POSITION", None)
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
                    stab_time = SatelliteConfig.SHAPE_POSITIONING_STABILIZATION_TIME
                    logger.info(
                        f" Satellite reached path start. Stabilizing for "
                        f"{stab_time:.1f} seconds before tracking..."
                    )
                else:
                    stabilization_time = current_time - self.shape_stabilization_start_time
                    if stabilization_time >= SatelliteConfig.SHAPE_POSITIONING_STABILIZATION_TIME:
                        SatelliteConfig.DXF_SHAPE_PHASE = "TRACKING"
                        setattr(SatelliteConfig, "DXF_TRACKING_START_TIME", current_time)
                        SatelliteConfig.DXF_TARGET_START_DISTANCE = 0.0
                        logger.info(" Path stabilization complete! Starting profile tracking...")
                        logger.info(f"   Target speed: {SatelliteConfig.DXF_TARGET_SPEED:.2f} m/s")
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

        final_pos = getattr(SatelliteConfig, "DXF_FINAL_POSITION", None)
        if final_pos is None:
            return None

        # Determine target orientation: use return angle if at return position,
        # otherwise use path tangent
        has_return = getattr(SatelliteConfig, "DXF_HAS_RETURN", False)
        return_pos = getattr(SatelliteConfig, "DXF_RETURN_POSITION", None)
        at_return_position = (
            has_return
            and return_pos is not None
            and final_pos is not None
            and np.allclose(final_pos, return_pos, atol=0.001)
        )

        if at_return_position:
            # Stabilizing at return position - use return angle
            target_orientation = getattr(
                SatelliteConfig, "DXF_RETURN_ANGLE", (0.0, 0.0, 0.0)
            ) or (0.0, 0.0, 0.0)
        else:
            # Stabilizing at end of path - use path tangent
            end_s = getattr(SatelliteConfig, "DXF_PATH_LENGTH", 0.0)
            target_orientation = get_path_tangent_orientation(
                path, end_s, SatelliteConfig.DXF_CLOSEST_POINT_INDEX
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

        if SatelliteConfig.DXF_STABILIZATION_START_TIME is None:
            setattr(SatelliteConfig, "DXF_STABILIZATION_START_TIME", current_time)

        stab_start = SatelliteConfig.DXF_STABILIZATION_START_TIME
        stabilization_time = current_time - stab_start if stab_start else 0.0
        if stabilization_time >= SatelliteConfig.SHAPE_FINAL_STABILIZATION_TIME:
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
        """Handle DXF returning phase."""
        return_pos = getattr(SatelliteConfig, "DXF_RETURN_POSITION", (0.0, 0.0))
        return_angle = getattr(
            SatelliteConfig, "DXF_RETURN_ANGLE", (0.0, 0.0, 0.0)
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
                    SatelliteConfig.DXF_SHAPE_PHASE = "STABILIZING"
                    setattr(SatelliteConfig, "DXF_STABILIZATION_START_TIME", current_time)
                    setattr(SatelliteConfig, "DXF_FINAL_POSITION", return_pos)
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
