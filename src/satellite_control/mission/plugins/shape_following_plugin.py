"""
Shape Following Mission Plugin

V4.0.0: Phase 2 - Mission Plugin System
"""

from typing import List, Tuple, Optional
import numpy as np

from src.satellite_control.mission.plugin import MissionPlugin
from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.config.mission_state import MissionState, create_mission_state
from src.satellite_control.mission.mission_manager import MissionManager
from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz


class ShapeFollowingMissionPlugin(MissionPlugin):
    """
    Shape Following Mission Plugin.
    
    Allows the satellite to follow a moving target along a shape path
    (circle, rectangle, triangle, hexagon, or custom DXF file).
    """
    
    def __init__(self):
        """Initialize shape following mission plugin."""
        self.manager = MissionManager()
    
    def get_name(self) -> str:
        """Get plugin name."""
        return "shape_following"
    
    def get_display_name(self) -> str:
        """Get display name."""
        return "Shape Following"
    
    def get_description(self) -> str:
        """Get description."""
        return (
            "Follow a moving target along a shape path. "
            "Supports demo shapes (circle, rectangle, triangle, hexagon) "
            "or custom shapes loaded from DXF CAD files. "
            "The satellite will track the moving target as it travels along the path."
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
            "dxf_shape_mode_active",
            "dxf_shape_path",
            "dxf_shape_phase",
            "dxf_target_speed",
            "dxf_path_length",
            "dxf_closest_point_index",
        ]
    
    def configure(self, config: SimulationConfig) -> MissionState:
        """
        Configure shape following mission parameters.
        
        This delegates to the existing MissionManager to maintain compatibility
        with the current interactive configuration flow.
        
        Args:
            config: Base simulation configuration
            
        Returns:
            Configured MissionState for shape following mission
        """
        # Use existing manager to configure shape following mission
        result = self.manager.run_dxf_shape_mode(return_simulation_config=True)
        
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
        Get target state for shape following mission.
        
        This delegates to MissionStateManager for the actual calculation,
        but provides the plugin interface.
        
        Args:
            current_state: Current satellite state [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            time: Current simulation time
            mission_state: Current mission state
            
        Returns:
            Target state vector [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        # Check if shape following mode is enabled
        if not mission_state.dxf_shape_mode_active:
            return current_state  # No shape following mission active
        
        # Get shape path
        if not mission_state.dxf_shape_path or len(mission_state.dxf_shape_path) == 0:
            return current_state  # No shape path configured
        
        # The actual target calculation is complex and handled by MissionStateManager
        # This is a simplified version that returns the current state
        # In practice, this would delegate to MissionStateManager.update_target_state()
        # For now, return current state as placeholder
        return current_state
    
    def is_complete(
        self,
        current_state: np.ndarray,
        time: float,
        mission_state: MissionState,
    ) -> bool:
        """
        Check if shape following mission is complete.
        
        Args:
            current_state: Current satellite state
            time: Current simulation time
            mission_state: Current mission state
            
        Returns:
            True if mission is complete, False otherwise
        """
        if not mission_state.dxf_shape_mode_active:
            return False
        
        # Check if shape phase indicates completion
        if hasattr(mission_state, "dxf_shape_phase"):
            return mission_state.dxf_shape_phase == "COMPLETE"
        
        return False
