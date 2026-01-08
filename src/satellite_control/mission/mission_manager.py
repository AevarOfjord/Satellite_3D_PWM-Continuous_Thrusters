#!/usr/bin/env python3
"""
Unified Mission Module for MPC Satellite Control

Provides unified mission system for both simulation and real testing systems.
Defines common mission types and their configurations.

Mission Types:
1. Waypoint Navigation: Navigate to single or multiple waypoints in sequence
2. Shape Following: Follow moving target along custom shape paths
   - Demo shapes: Circle, Rectangle, Triangle, Hexagon
   - Custom shapes: Load from DXF CAD files

Features:
- Unified interface for both simulation and real hardware
- Automatic mode detection and configuration
- Interactive user input for mission parameters
- Obstacle configuration support
- Mission validation and confirmation
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# V3.0.0: No longer import SatelliteConfig or timing module
from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.mission.mission_cli import MissionCLI
from src.satellite_control.mission.mission_logic import MissionLogic

# Add path for optional DXF_Viewer module (3 levels up from
# src/satellite_control/mission)
sys.path.insert(0, str(Path(__file__).parents[3] / "DXF"))


class MissionManager:
    """
    Unified mission manager.
    Refactored to delegate to MissionLogic and MissionCLI.
    """

    def __init__(self):
        """Initialize mission manager."""
        self.mode = "simulation"
        self.logic = MissionLogic()
        self.cli = MissionCLI(self.logic)
        self.system_title = "Satellite Control Simulation"

    def show_mission_menu(self) -> str:
        """Show main mission selection menu."""
        return self.cli.show_mission_menu()

    def run_selected_mission(self, mode_choice: str) -> Optional[Dict[str, Any]]:
        """Run the selected mission type.
        
        V4.0.0: Phase 2 - Now uses plugin system to run missions.
        Falls back to legacy hardcoded missions for backward compatibility.
        """
        from src.satellite_control.mission.plugin import get_registry
        from src.satellite_control.config.simulation_config import SimulationConfig
        
        registry = get_registry()
        plugin = registry.get_plugin(mode_choice)
        
        if plugin:
            # Use plugin system
            config = SimulationConfig.create_default()
            mission_state = plugin.configure(config)
            config.mission_state = mission_state
            
            return {
                "simulation_config": config,
                "mission_type": mode_choice,
                "plugin_name": plugin.get_display_name(),
            }
        
        # Fallback to legacy hardcoded missions for backward compatibility
        if mode_choice == "waypoint":
            # Delegate entirely to CLI which has full implementation
            # V3.0.0: Always request simulation_config
            return self.cli.run_multi_point_mode(return_simulation_config=True)
        elif mode_choice == "shape_following":
            if self.cli.confirm_mission("Shape Following"):
                # V3.0.0: Always request simulation_config
                return self.run_dxf_shape_mode(return_simulation_config=True)
        return None

    def run_dxf_shape_mode(self, return_simulation_config: bool = False) -> Dict[str, Any]:
        """Mode 2: Shape Following (Refactored).
        
        Args:
            return_simulation_config: If True, includes SimulationConfig in returned dict.
            
        Returns:
            Dictionary with mission configuration. If return_simulation_config=True, includes 'simulation_config' key.
        """
        # Create SimulationConfig for v2.0.0 pattern
        simulation_config = SimulationConfig.create_default()
        mission_state = simulation_config.mission_state
        
        print("\n SHAPE FOLLOWING MISSION")
        print("Satellite will follow a moving target along a shape path.")
        print("  Phase 2: Track moving target along the shape path")
        # V3.0.0: Use app_config instead of SatelliteConfig
        stab_time = simulation_config.app_config.simulation.shape_final_stabilization_time
        print(f"  Phase 3: Stabilize at completion point ({stab_time:.1f}s)")

        print("\nSimulation starting configuration:")
        # V3.0.0: Use defaults from mission_state or hardcoded defaults
        default_start_pos = (0.0, 0.0, 0.0)
        default_start_angle = (0.0, 0.0, 0.0)
        start_pos = self.cli.get_user_position("starting", default_start_pos)
        start_angle = self.cli.get_user_orientation("starting", default_start_angle)
        start_vx, start_vy, start_vz, start_omega = self.cli.get_user_velocities()

        # Print summary
        roll_deg, pitch_deg, yaw_deg = np.degrees(start_angle)
        omega_deg = np.degrees(start_omega)
        print(
            f"Starting: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) m, "
            f"roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°, yaw={yaw_deg:.1f}°, "
            f"({start_vx:.2f}, {start_vy:.2f}, {start_vz:.2f}) m/s, "
            f"{omega_deg:.1f}°/s"
        )

        print("\nShape configuration:")
        shape_points = None
        try:
            import ezdxf  # noqa: F401

            dxf_available = True
        except ImportError:
            dxf_available = False
            print("  ezdxf library not installed. Using demo shapes only.")

        if dxf_available:
            print("\nShape source:")
            print("1. Load from DXF file")
            print("2. Circle")
            print("3. Rectangle")
            print("4. Triangle")
            print("5. Hexagon")
            choice = input("Select option (1-5): ").strip()
            if choice == "1":
                dxf_path = input("Enter DXF file path: ").strip()
                try:
                    shape_points = self.logic.load_dxf_shape(dxf_path)
                    n_pts = len(shape_points)
                    print(f" Loaded shape with {n_pts} points from DXF")
                except Exception as e:
                    print(f" Failed to load DXF: {e}")
                    print("Falling back to demo circle")
                    shape_points = self.logic.generate_demo_shape("circle")
            elif choice == "2":
                shape_points = self.logic.generate_demo_shape("circle")
            elif choice == "3":
                shape_points = self.logic.generate_demo_shape("rectangle")
            elif choice == "4":
                shape_points = self.logic.generate_demo_shape("triangle")
            elif choice == "5":
                shape_points = self.logic.generate_demo_shape("hexagon")
            else:
                print("Invalid choice. Using demo circle.")
                shape_points = self.logic.generate_demo_shape("circle")
        else:
            print("\nDemo shapes available:")
            print("1. Circle")
            print("2. Rectangle")
            print("3. Triangle")
            print("4. Hexagon")
            choice = input("Select demo shape (1-4): ").strip()
            if choice == "1":
                shape_points = self.logic.generate_demo_shape("circle")
            elif choice == "2":
                shape_points = self.logic.generate_demo_shape("rectangle")
            elif choice == "3":
                shape_points = self.logic.generate_demo_shape("triangle")
            elif choice == "4":
                shape_points = self.logic.generate_demo_shape("hexagon")
            else:
                print("Invalid choice. Using circle.")
                shape_points = self.logic.generate_demo_shape("circle")

        try:
            shape_center = self.cli.get_user_position("shape center", (0.0, 0.0))
            shape_rotation_input = input("Shape rotation angle (degrees, default 0): ").strip()
            shape_rotation_deg = float(shape_rotation_input) if shape_rotation_input else 0.0
            shape_rotation_rad = np.radians(shape_rotation_deg)
        except ValueError:
            print("Invalid input. Using default shape parameters.")
            shape_center = (0.0, 0.0)
            shape_rotation_deg = 0.0
            shape_rotation_rad = 0.0

        print(f"Shape center: ({shape_center[0]:.2f}, {shape_center[1]:.2f}) m")
        print(f"Shape rotation: {shape_rotation_deg:.1f}°")

        try:
            offset_input = input("Offset distance from shape (meters, default 0.5): ").strip()
            offset_distance = float(offset_input) if offset_input else 0.5
            if offset_distance < 0.1:
                print("Minimum offset 0.1m. Using 0.1m.")
                offset_distance = 0.1
            elif offset_distance > 2.0:
                print("Maximum offset 2.0m. Using 2.0m.")
                offset_distance = 2.0
        except ValueError:
            print("Invalid input. Using default 0.5m offset.")
            offset_distance = 0.5

        print(f"Offset distance: {offset_distance:.2f} m")

        transformed_shape = self.logic.transform_shape(
            shape_points, shape_center, shape_rotation_rad
        )
        # Use robust upscale_shape instead of naive make_offset_path
        upscaled_path = self.logic.upscale_shape(transformed_shape, offset_distance)
        print(f" Created upscaled path with {len(upscaled_path)} points")

        print("\nMoving target configuration:")
        try:
            path_length = self.logic.calculate_path_length(upscaled_path)
            print(f"Path length: {path_length:.2f} m")
            # V3.0.0: Use app_config instead of timing module
            default_spd = simulation_config.app_config.simulation.default_target_speed
            target_speed_input = input(
                f"Target speed (meters/second, default {default_spd}): "
            ).strip()
            target_speed_mps = (
                float(target_speed_input) if target_speed_input else default_spd
            )
            if target_speed_mps <= 0:
                spd = default_spd
                print(f"Speed must be positive. Using {spd} m/s.")
                target_speed_mps = default_spd
            elif target_speed_mps > 0.5:
                print("Maximum speed 0.5 m/s. Using 0.5 m/s.")
                target_speed_mps = 0.5
        except ValueError:
            print("Invalid input. Using default speed.")
            target_speed_mps = simulation_config.app_config.simulation.default_target_speed

        print(f"Target speed: {target_speed_mps:.2f} m/s")

        # Get return position and orientation
        print("\nReturn position configuration:")
        use_return = (
            input("Return to specific position after profile? (y/n, default n): ").strip().lower()
        )
        if use_return == "y" or use_return == "yes":
            return_pos = self.cli.get_user_position("return", start_pos)
            return_angle = self.cli.get_user_orientation("return", start_angle)
            if return_pos is not None and return_angle is not None:
                rp = return_pos
                ra_roll, ra_pitch, ra_yaw = np.degrees(return_angle)
                print(
                    f"Return position: ({rp[0]:.2f}, {rp[1]:.2f}) m, "
                    f"roll={ra_roll:.1f}°, pitch={ra_pitch:.1f}°, yaw={ra_yaw:.1f}°"
                )
            has_return = True
        else:
            return_pos = None
            return_angle = None
            has_return = False
            print("No return position specified. Mission will end after profile.")

        path_length = self.logic.calculate_path_length(upscaled_path)
        # Estimate duration: 3s initial positioning + profile time + 10s final
        # stabilization
        estimated_duration = 3.0 + (path_length / target_speed_mps) + 10.0
        print(f"Estimated mission duration: {estimated_duration:.1f} seconds")

        print(f"\n{'=' * 50}")
        print("  SHAPE FOLLOWING MISSION SUMMARY")
        print(f"{'=' * 50}")
        sa_roll, sa_pitch, sa_yaw = np.degrees(start_angle)
        print(
            f"Starting: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) m, "
            f"roll={sa_roll:.1f}°, pitch={sa_pitch:.1f}°, yaw={sa_yaw:.1f}°"
        )
        print(f"Shape center: ({shape_center[0]:.2f}, {shape_center[1]:.2f}) m")
        print(f"Shape rotation: {shape_rotation_deg:.1f}°")
        print(f"Offset distance: {offset_distance:.2f} m")
        print(f"Path points: {len(upscaled_path)}")
        print(f"Path length: {path_length:.2f} m")
        print(f"Target speed: {target_speed_mps:.2f} m/s")

        if has_return and return_pos is not None and return_angle is not None:
            rp = return_pos
            ra_roll, ra_pitch, ra_yaw = np.degrees(return_angle)
            print(
                f"Return position: ({rp[0]:.2f}, {rp[1]:.2f}) m, "
                f"roll={ra_roll:.1f}°, pitch={ra_pitch:.1f}°, yaw={ra_yaw:.1f}°"
            )

        print("Mission phases:")
        print("  Phase 1: Position at closest point and stabilize (3s)")
        print("  Phase 2: Track moving target along shape path")
        if has_return:
            print("  Phase 3: Return to specified position and stabilize (10s)")
        else:
            print("  Phase 3: Stabilize at final shape position (10s)")
        print(f"Mission duration: ~{estimated_duration:.1f}s estimated")

        # Configure obstacles (V3.0.0: always uses mission_state)
        obstacles = self.cli.configure_obstacles(mission_state=mission_state)

        # Update MissionState (v2.0.0)
        mission_state.dxf_shape_mode_active = True
        mission_state.dxf_shape_center = shape_center
        mission_state.dxf_base_shape = transformed_shape
        mission_state.dxf_shape_path = upscaled_path
        mission_state.dxf_target_speed = target_speed_mps
        mission_state.dxf_estimated_duration = estimated_duration
        mission_state.dxf_mission_start_time = None
        mission_state.dxf_shape_phase = "POSITIONING"
        mission_state.dxf_path_length = path_length
        mission_state.dxf_has_return = has_return
        if return_pos:
            mission_state.dxf_return_position = return_pos
        if return_angle:
            mission_state.dxf_return_angle = return_angle

        config = {
            "mission_type": "shape_following",
            "shape_center": shape_center,
            "shape_rotation_rad": shape_rotation_rad,
            "shape_rotation_deg": shape_rotation_deg,
            "offset_distance": offset_distance,
            "target_speed_mps": target_speed_mps,
            "estimated_duration": estimated_duration,
            "shape_points": transformed_shape,
            "upscaled_path": upscaled_path,
            "path_length": path_length,
            "has_return": has_return,
            "return_pos": return_pos,
            "return_angle": return_angle,
        }

        config.update(
            {
                "start_pos": start_pos,
                "start_angle": start_angle,
                "start_vx": start_vx,
                "start_vy": start_vy,
                "start_vz": start_vz,
                "start_omega": start_omega,
            }
        )
        
        # V3.0.0: All state is managed via MissionState, no SatelliteConfig mutations
        if return_simulation_config:
            config["simulation_config"] = simulation_config

        return config

    def load_dxf_shape(self, file_path: str) -> List[Tuple[float, float]]:
        """
        Load shape points from DXF using the same pipeline as DXF_Viewer.

        Args:
            file_path: Path to DXF file

        Returns:
            List of (x, y) points in meters
        """
        import ezdxf

        try:
            from dxf_viewer import (
                extract_boundary_polygon,
                sanitize_boundary,
                units_code_to_name_and_scale,
            )
        except Exception as e:
            raise ImportError(f"dxf_viewer utilities unavailable: {e}")

        # Read DXF and determine units
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        insunits = int(doc.header.get("$INSUNITS", 0))
        units_name, to_m = units_code_to_name_and_scale(insunits)

        # Extract and sanitize boundary in native units, then scale to meters
        boundary = extract_boundary_polygon(msp)
        boundary = sanitize_boundary(boundary, to_m)
        boundary_m = [(float(x) * to_m, float(y) * to_m) for (x, y) in boundary] if boundary else []

        if not boundary_m:
            raise ValueError("No usable DXF boundary could be constructed.")

        print(f" DXF Loaded (viewer parity): {len(boundary_m)} points")
        print(f"   Units: {units_name} (INSUNITS={insunits}), " f"scaled → meters (x{to_m})")
        return boundary_m


# Helper functions for external use
def get_path_tangent_orientation(
    path: List[Tuple[float, float]], distance: float, start_idx: int = 0
) -> float:
    """
    Calculate tangent orientation at a given distance along path.

    Args:
        path: List of (x, y) points
        distance: Distance along path in meters
        start_idx: Index to start searching from (optimization)

    Returns:
        Orientation angle in radians
    """
    if not path:
        return 0.0

    current_dist = 0.0
    for i in range(len(path) - 1):
        idx = (start_idx + i) % (len(path) - 1)
        p1 = np.array(path[idx])
        p2 = np.array(path[idx + 1])
        segment_len = np.linalg.norm(p2 - p1)

        if current_dist + segment_len >= distance:
            # Found segment
            direction = p2 - p1
            angle = np.arctan2(direction[1], direction[0])
            return float(angle)

        current_dist += float(segment_len)

    # If past end, return angle of last segment
    p1 = np.array(path[-2])
    p2 = np.array(path[-1])
    direction = p2 - p1
    return float(np.arctan2(direction[1], direction[0]))


def get_position_on_path(
    path: List[Tuple[float, float]], distance: float, start_idx: int = 0
) -> Tuple[Tuple[float, float], int]:
    """
    Calculate position at a given distance along path.

    Args:
        path: List of (x, y) points
        distance: Distance along path in meters
        start_idx: Index to start searching from

    Returns:
        Tuple of ((x, y), new_index)
    """
    if not path:
        return (0.0, 0.0), 0

    current_dist = 0.0
    # Search forward from start_idx
    total_segments = len(path) - 1

    for i in range(total_segments):
        idx = (start_idx + i) % total_segments
        p1 = np.array(path[idx])
        p2 = np.array(path[idx + 1])
        segment_len = np.linalg.norm(p2 - p1)

        if current_dist + segment_len >= distance:
            t = (distance - current_dist) / max(float(segment_len), 1e-6)
            pos = p1 + t * (p2 - p1)
            return (float(pos[0]), float(pos[1])), idx

        current_dist += float(segment_len)

    return path[-1], total_segments - 1


if __name__ == "__main__":
    # Test interactive menu
    manager = MissionManager()
    mode = manager.show_mission_menu()
    config = manager.run_selected_mission(mode)
    if config:
        print("\nMission Configuration Generated:")
        for k, v in config.items():
            if k == "shape_points" or k == "upscaled_path":
                print(f"  {k}: <list of {len(v)} points>")
            else:
                print(f"  {k}: {v}")
