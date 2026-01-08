"""
Waypoint Navigation Mission Plugin

V4.0.0: Phase 2 - Mission Plugin System
"""

from typing import List, Tuple, Optional
import numpy as np

from src.satellite_control.mission.plugin import MissionPlugin
from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.config.mission_state import MissionState, create_mission_state
from src.satellite_control.mission.mission_cli import MissionCLI
from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz


class WaypointMissionPlugin(MissionPlugin):
    """
    Waypoint Navigation Mission Plugin.
    
    Allows the satellite to navigate to one or more waypoints in sequence.
    """
    
    def __init__(self):
        """Initialize waypoint mission plugin."""
        self.cli = MissionCLI()
    
    def get_name(self) -> str:
        """Get plugin name."""
        return "waypoint"
    
    def get_display_name(self) -> str:
        """Get display name."""
        return "Waypoint Navigation"
    
    def get_description(self) -> str:
        """Get description."""
        return (
            "Navigate to one or more waypoints in sequence. "
            "Each waypoint specifies a target position and orientation. "
            "The satellite will stabilize at each waypoint before proceeding to the next."
        )
    
    def get_version(self) -> str:
        """Get version."""
        return "1.0.0"
    
    def get_author(self) -> Optional[str]:
        """Get author."""
        return "Satellite Control Team"
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters."""
        return [
            "enable_waypoint_mode",
            "enable_multi_point_mode",
            "waypoint_targets",
            "waypoint_angles",
            "current_target_index",
        ]
    
    def configure(self, config: SimulationConfig) -> MissionState:
        """
        Configure waypoint mission parameters.
        
        This delegates to the existing MissionCLI to maintain compatibility
        with the current interactive configuration flow.
        
        Args:
            config: Base simulation configuration
            
        Returns:
            Configured MissionState for waypoint mission
        """
        # Use existing CLI to configure waypoint mission
        result = self.cli.run_multi_point_mode(return_simulation_config=True)
        
        if result and "simulation_config" in result:
            return result["simulation_config"].mission_state
        
        # Fallback: create empty mission state
        return create_mission_state()
    
    def get_target_state(
        self,
        current_state: np.ndarray,
        time: float,
        mission_state: MissionState,
    ) -> np.ndarray:
        """
        Get target state for waypoint mission.
        
        This delegates to MissionStateManager for the actual calculation,
        but provides the plugin interface.
        
        Args:
            current_state: Current satellite state [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            time: Current simulation time
            mission_state: Current mission state
            
        Returns:
            Target state vector [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        # Check if waypoint mode is enabled
        if not (mission_state.enable_waypoint_mode or mission_state.enable_multi_point_mode):
            return current_state  # No waypoint mission active
        
        # Get current waypoint
        if not mission_state.waypoint_targets or len(mission_state.waypoint_targets) == 0:
            return current_state  # No waypoints configured
        
        current_idx = mission_state.current_target_index
        if current_idx >= len(mission_state.waypoint_targets):
            # All waypoints reached
            return current_state
        
        # Get target position and angle
        target_pos = mission_state.waypoint_targets[current_idx]
        target_angles = mission_state.waypoint_angles if mission_state.waypoint_angles else []
        
        # Convert to 3D position if needed
        if len(target_pos) == 2:
            target_pos_3d = (target_pos[0], target_pos[1], 0.0)
        else:
            target_pos_3d = target_pos
        
        # Get target orientation
        if target_angles and current_idx < len(target_angles):
            target_angle = target_angles[current_idx]
            # Convert to quaternion
            if isinstance(target_angle, (int, float)):
                # Single yaw angle
                target_quat = euler_xyz_to_quat_wxyz((0.0, 0.0, float(target_angle)))
            elif isinstance(target_angle, (tuple, list)) and len(target_angle) == 3:
                # Full Euler angles
                target_quat = euler_xyz_to_quat_wxyz(target_angle)
            else:
                # Default: no rotation
                target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Default: no rotation
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Build target state: [pos(3), quat(4), vel(3), w(3)]
        target_state = np.zeros(13)
        target_state[0:3] = np.array(target_pos_3d)
        target_state[3:7] = target_quat
        target_state[7:10] = 0.0  # Target velocity (zero)
        target_state[10:13] = 0.0  # Target angular velocity (zero)
        
        return target_state
    
    def is_complete(
        self,
        current_state: np.ndarray,
        time: float,
        mission_state: MissionState,
    ) -> bool:
        """
        Check if waypoint mission is complete.
        
        Args:
            current_state: Current satellite state
            time: Current simulation time
            mission_state: Current mission state
            
        Returns:
            True if all waypoints have been reached, False otherwise
        """
        if not (mission_state.enable_waypoint_mode or mission_state.enable_multi_point_mode):
            return False
        
        if not mission_state.waypoint_targets:
            return False
        
        # Check if we've reached the last waypoint
        current_idx = mission_state.current_target_index
        if current_idx >= len(mission_state.waypoint_targets):
            return True
        
        # Check if multi_point_phase indicates completion
        if hasattr(mission_state, "multi_point_phase"):
            return mission_state.multi_point_phase == "COMPLETE"
        
        return False
