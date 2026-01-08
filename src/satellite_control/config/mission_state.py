"""
Mission State Management for Satellite Control System

Runtime mission state tracking for waypoint navigation and shape following.
Maintains mutable state variables for mission execution and phase transitions.

Mission types supported:
1. Waypoint Navigation: Single or multiple sequential waypoints
2. Shape Following: Geometric paths (circle, rectangle, triangle, hexagon, DXF)

State tracking:
- Current target index and position
- Waypoint phase (approaching, stabilizing, holding)
- Stabilization timers and counters
- Shape path progress and segment tracking
- DXF shape configuration and parameters

Key features:
- Clean separation of mutable state from immutable config
- Type-safe dataclass with default values
- Integration with mission_state_manager
- Thread-safe for concurrent access
- Reset functionality for mission restart
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MissionState:
    """
    Mission state tracking for runtime execution.

    Attributes:
        # Waypoint Navigation
        enable_waypoint_mode: Enable waypoint navigation
        waypoint_targets: List of (x, y) target positions
        waypoint_angles: List of target angles in radians
        current_target_index: Current target index
        target_stabilization_start_time: Stabilization timer
        waypoint_phase: Current phase

        # Shape Following
        dxf_shape_mode_active: Enable shape following mode
        dxf_shape_center: Center point (x, y) of shape
        dxf_shape_path: List of (x, y) points defining shape path
        dxf_base_shape: Base shape before transformations
        dxf_target_speed: Target speed along path in m/s
        dxf_estimated_duration: Estimated completion time
        dxf_shape_phase: Current phase ("POSITIONING", "TRACKING", etc.)
        dxf_path_length: Total path length in meters
        dxf_closest_point_index: Index of closest point on path
        dxf_current_target_position: Current target position (x, y)
        dxf_tracking_start_time: Tracking start time
        dxf_target_start_distance: Distance at tracking start
        dxf_mission_start_time: Mission start time
        dxf_stabilization_start_time: Stabilization start time
        dxf_final_position: Final position (x, y)
        dxf_shape_rotation: Shape rotation angle in radians
        dxf_offset_distance: Offset distance for shape positioning
        dxf_has_return: Whether to return to start position
        dxf_return_position: Return position (x, y)
        dxf_return_angle: Return angle in radians
        dxf_return_start_time: Return phase start time
    """

    # Waypoint Navigation
    enable_waypoint_mode: bool = False
    waypoint_targets: List[Tuple[float, float, float]] = field(default_factory=list)
    waypoint_angles: List[float] = field(default_factory=list)
    current_target_index: int = 0
    target_stabilization_start_time: Optional[float] = None
    waypoint_phase: Optional[str] = None

    enable_multi_point_mode: bool = False
    multi_point_targets: List[Tuple[float, float, float]] = field(default_factory=list)
    multi_point_angles: List[float] = field(default_factory=list)
    multi_point_phase: Optional[str] = None

    # Shape Following - Path tracking
    dxf_shape_mode_active: bool = False
    dxf_shape_center: Optional[Tuple[float, float, float]] = None
    dxf_shape_path: List[Tuple[float, float, float]] = field(default_factory=list)
    dxf_base_shape: List[Tuple[float, float, float]] = field(default_factory=list)
    dxf_target_speed: float = 0.1
    dxf_estimated_duration: float = 60.0
    dxf_shape_phase: str = "POSITIONING"
    dxf_path_length: float = 0.0
    dxf_closest_point_index: int = 0
    dxf_current_target_position: Optional[Tuple[float, float, float]] = None
    dxf_tracking_start_time: Optional[float] = None
    dxf_target_start_distance: float = 0.0
    dxf_mission_start_time: Optional[float] = None
    dxf_stabilization_start_time: Optional[float] = None
    dxf_final_position: Optional[Tuple[float, float, float]] = None
    dxf_shape_rotation: float = 0.0
    dxf_offset_distance: float = 0.5
    dxf_has_return: bool = False
    dxf_return_position: Optional[Tuple[float, float, float]] = None
    dxf_return_angle: Optional[float] = None
    dxf_return_start_time: Optional[float] = None

    def reset(self) -> None:
        """Reset all mission state to defaults."""
        self.__init__()  # type: ignore[misc]

    def get_current_mission_type(self) -> str:
        """
        Get the currently active mission type.

        Returns:
            WAYPOINT_NAVIGATION, WAYPOINT_NAVIGATION_MULTI,
            SHAPE_FOLLOWING, or NONE
        """
        if self.dxf_shape_mode_active:
            return "SHAPE_FOLLOWING"
        elif self.enable_waypoint_mode or self.enable_multi_point_mode:
            num_targets = len(
                self.waypoint_targets if self.waypoint_targets else self.multi_point_targets
            )
            return "WAYPOINT_NAVIGATION_MULTI" if num_targets > 1 else "WAYPOINT_NAVIGATION"
        else:
            return "NONE"


def create_mission_state() -> MissionState:
    """
    Create a new mission state with default values.

    Returns:
        MissionState initialized to defaults
    """
    return MissionState()


def sync_mission_state_from_satellite_config(
    mission_state: Optional[MissionState] = None,
) -> MissionState:
    """
    Sync mission state from SatelliteConfig (for backward compatibility).
    
    This function reads the current state from SatelliteConfig and updates
    the MissionState object. Used when CLI components mutate SatelliteConfig
    and we need to sync that state to MissionState for the simulation.
    
    Args:
        mission_state: Optional MissionState to update (creates new if None)
        
    Returns:
        Updated MissionState object
    """
    from src.satellite_control.config import SatelliteConfig
    
    if mission_state is None:
        mission_state = create_mission_state()
    
    # Sync waypoint mode
    if hasattr(SatelliteConfig, "ENABLE_WAYPOINT_MODE"):
        mission_state.enable_waypoint_mode = SatelliteConfig.ENABLE_WAYPOINT_MODE
    if hasattr(SatelliteConfig, "WAYPOINT_TARGETS"):
        # Convert 2D tuples to 3D if needed
        waypoint_targets = []
        for target in SatelliteConfig.WAYPOINT_TARGETS:
            if len(target) == 2:
                waypoint_targets.append((target[0], target[1], 0.0))
            else:
                waypoint_targets.append(target)
        mission_state.waypoint_targets = waypoint_targets
    if hasattr(SatelliteConfig, "WAYPOINT_ANGLES"):
        # Convert tuple angles to single float if needed
        waypoint_angles = []
        for angle in SatelliteConfig.WAYPOINT_ANGLES:
            if isinstance(angle, (tuple, list)) and len(angle) == 3:
                # Use yaw (z) component
                waypoint_angles.append(float(angle[2]))
            else:
                waypoint_angles.append(float(angle))
        mission_state.waypoint_angles = waypoint_angles
    if hasattr(SatelliteConfig, "CURRENT_TARGET_INDEX"):
        mission_state.current_target_index = SatelliteConfig.CURRENT_TARGET_INDEX
    if hasattr(SatelliteConfig, "TARGET_STABILIZATION_START_TIME"):
        mission_state.target_stabilization_start_time = SatelliteConfig.TARGET_STABILIZATION_START_TIME
    
    # Sync DXF shape mode
    if hasattr(SatelliteConfig, "DXF_SHAPE_MODE_ACTIVE"):
        mission_state.dxf_shape_mode_active = SatelliteConfig.DXF_SHAPE_MODE_ACTIVE
    if hasattr(SatelliteConfig, "DXF_SHAPE_CENTER"):
        center = SatelliteConfig.DXF_SHAPE_CENTER
        if center is not None:
            if len(center) == 2:
                mission_state.dxf_shape_center = (center[0], center[1], 0.0)
            else:
                mission_state.dxf_shape_center = center
    if hasattr(SatelliteConfig, "DXF_SHAPE_PATH"):
        # Convert 2D paths to 3D if needed
        path = SatelliteConfig.DXF_SHAPE_PATH
        if path:
            path_3d = []
            for point in path:
                if len(point) == 2:
                    path_3d.append((point[0], point[1], 0.0))
                else:
                    path_3d.append(point)
            mission_state.dxf_shape_path = path_3d
    if hasattr(SatelliteConfig, "DXF_BASE_SHAPE"):
        base_shape = SatelliteConfig.DXF_BASE_SHAPE
        if base_shape:
            base_shape_3d = []
            for point in base_shape:
                if len(point) == 2:
                    base_shape_3d.append((point[0], point[1], 0.0))
                else:
                    base_shape_3d.append(point)
            mission_state.dxf_base_shape = base_shape_3d
    if hasattr(SatelliteConfig, "DXF_TARGET_SPEED"):
        mission_state.dxf_target_speed = SatelliteConfig.DXF_TARGET_SPEED
    if hasattr(SatelliteConfig, "DXF_ESTIMATED_DURATION"):
        mission_state.dxf_estimated_duration = SatelliteConfig.DXF_ESTIMATED_DURATION
    if hasattr(SatelliteConfig, "DXF_SHAPE_PHASE"):
        mission_state.dxf_shape_phase = SatelliteConfig.DXF_SHAPE_PHASE
    if hasattr(SatelliteConfig, "DXF_PATH_LENGTH"):
        mission_state.dxf_path_length = SatelliteConfig.DXF_PATH_LENGTH
    if hasattr(SatelliteConfig, "DXF_CLOSEST_POINT_INDEX"):
        mission_state.dxf_closest_point_index = SatelliteConfig.DXF_CLOSEST_POINT_INDEX
    if hasattr(SatelliteConfig, "DXF_HAS_RETURN"):
        mission_state.dxf_has_return = SatelliteConfig.DXF_HAS_RETURN
    if hasattr(SatelliteConfig, "DXF_RETURN_POSITION"):
        return_pos = getattr(SatelliteConfig, "DXF_RETURN_POSITION", None)
        if return_pos is not None:
            if len(return_pos) == 2:
                mission_state.dxf_return_position = (return_pos[0], return_pos[1], 0.0)
            else:
                mission_state.dxf_return_position = return_pos
    if hasattr(SatelliteConfig, "DXF_RETURN_ANGLE"):
        return_angle = getattr(SatelliteConfig, "DXF_RETURN_ANGLE", None)
        if return_angle is not None:
            if isinstance(return_angle, (tuple, list)) and len(return_angle) == 3:
                mission_state.dxf_return_angle = float(return_angle[2])  # Use yaw
            else:
                mission_state.dxf_return_angle = float(return_angle)
    
    return mission_state


def print_mission_state(state: MissionState) -> None:
    """Print current mission state."""
    print("=" * 80)
    print("MISSION STATE")
    print("=" * 80)

    mission_type = state.get_current_mission_type()
    print(f"\nMission: {mission_type}")

    if state.enable_waypoint_mode or state.enable_multi_point_mode:
        targets = state.waypoint_targets if state.waypoint_targets else state.multi_point_targets
        print("\nWaypoint Navigation:")
        print(f"  Targets: {len(targets)}")
        print(f"  Current: {state.current_target_index}")
        print(f"  Phase: {state.waypoint_phase or state.multi_point_phase}")

    if state.dxf_shape_mode_active:
        print("\nShape Following:")
        print(f"  Center: {state.dxf_shape_center}")
        print(f"  Points: {len(state.dxf_shape_path)}")
        print(f"  Phase: {state.dxf_shape_phase}")
        print(f"  Length: {state.dxf_path_length:.2f} m")

    print("=" * 80)
