"""
Example Custom Mission Plugin

V4.0.0: Phase 2 - Mission Plugin System

This is an example custom mission plugin that demonstrates how to create
a new mission type. This plugin implements a simple "hover" mission that
keeps the satellite at a fixed position and orientation.

To use this plugin:
1. Copy this file to your custom plugins directory
2. Modify the mission logic as needed
3. Register it via config file or install-mission command
"""

from typing import List, Optional, Tuple
import numpy as np

from src.satellite_control.mission.plugin import MissionPlugin
from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.config.mission_state import MissionState, create_mission_state
from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz


class ExampleHoverMissionPlugin(MissionPlugin):
    """
    Example Hover Mission Plugin.
    
    A simple mission that keeps the satellite at a fixed position and orientation.
    This demonstrates how to create a custom mission plugin.
    """
    
    def __init__(self):
        """Initialize hover mission plugin."""
        self.target_position: Optional[Tuple[float, float, float]] = None
        self.target_orientation: Optional[Tuple[float, float, float]] = None
    
    def get_name(self) -> str:
        """Get plugin name."""
        return "example_hover"
    
    def get_display_name(self) -> str:
        """Get display name."""
        return "Example: Hover Mission"
    
    def get_description(self) -> str:
        """Get description."""
        return (
            "Example custom mission that keeps the satellite at a fixed position "
            "and orientation. This demonstrates how to create custom mission plugins."
        )
    
    def get_version(self) -> str:
        """Get version."""
        return "1.0.0"
    
    def get_author(self) -> Optional[str]:
        """Get author."""
        return "Example Plugin Developer"
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters."""
        return []  # This plugin doesn't require specific MissionState parameters
    
    def configure(self, config: SimulationConfig) -> MissionState:
        """
        Configure hover mission parameters.
        
        This is a simple example that prompts for a target position.
        In a real plugin, you might use a GUI, config file, or other method.
        
        Args:
            config: Base simulation configuration
            
        Returns:
            Configured MissionState for hover mission
        """
        # Create mission state
        mission_state = create_mission_state()
        
        # Simple interactive configuration
        print("\n=== Example Hover Mission Configuration ===")
        print("This is an example custom mission plugin.")
        
        try:
            x_str = input("Target X position (meters, default 0.0): ").strip()
            y_str = input("Target Y position (meters, default 0.0): ").strip()
            z_str = input("Target Z position (meters, default 0.0): ").strip()
            
            x = float(x_str) if x_str else 0.0
            y = float(y_str) if y_str else 0.0
            z = float(z_str) if z_str else 0.0
            
            self.target_position = (x, y, z)
            self.target_orientation = (0.0, 0.0, 0.0)  # Default: no rotation
            
            print(f"\nHover mission configured:")
            print(f"  Target position: ({x:.2f}, {y:.2f}, {z:.2f}) m")
            print(f"  Target orientation: (0.0°, 0.0°, 0.0°)")
            
        except (ValueError, KeyboardInterrupt) as e:
            print(f"\nConfiguration cancelled or invalid: {e}")
            # Use defaults
            self.target_position = (0.0, 0.0, 0.0)
            self.target_orientation = (0.0, 0.0, 0.0)
        
        return mission_state
    
    def get_target_state(
        self,
        current_state: np.ndarray,
        time: float,
        mission_state: MissionState,
    ) -> np.ndarray:
        """
        Get target state for hover mission.
        
        Always returns the configured hover position and orientation.
        
        Args:
            current_state: Current satellite state [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            time: Current simulation time
            mission_state: Current mission state
            
        Returns:
            Target state vector [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        if self.target_position is None:
            # Use current position if not configured
            return current_state
        
        # Convert orientation to quaternion
        target_quat = euler_xyz_to_quat_wxyz(self.target_orientation)
        
        # Build target state: [pos(3), quat(4), vel(3), w(3)]
        target_state = np.zeros(13)
        target_state[0:3] = np.array(self.target_position)
        target_state[3:7] = target_quat
        target_state[7:10] = 0.0  # Target velocity (zero for hover)
        target_state[10:13] = 0.0  # Target angular velocity (zero for hover)
        
        return target_state
    
    def is_complete(
        self,
        current_state: np.ndarray,
        time: float,
        mission_state: MissionState,
    ) -> bool:
        """
        Check if hover mission is complete.
        
        For this example, the mission never completes (infinite hover).
        In a real plugin, you might check for time limits, position accuracy, etc.
        
        Args:
            current_state: Current satellite state
            time: Current simulation time
            mission_state: Current mission state
            
        Returns:
            False (mission never completes in this example)
        """
        # This example mission never completes
        # In a real plugin, you might check:
        # - Time limit: return time > max_time
        # - Position accuracy: return position_error < tolerance
        # - User command: return mission_state.should_stop
        return False
