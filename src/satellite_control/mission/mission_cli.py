"""
Mission CLI Module

Handles user interaction, menus, and input gathering.
Uses MissionLogic for validation and calculations.
"""

import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.satellite_control.config import SatelliteConfig
from src.satellite_control.mission.mission_logic import MissionLogic


class MissionCLI:
    """
    Handles all user interaction code (input/print).
    """

    def __init__(self, logic: Optional[MissionLogic] = None):
        self.logic = logic or MissionLogic()
        self.system_title = "Satellite Control Simulation"

    def show_mission_menu(self) -> str:
        """Show main mission selection menu."""
        print(f"\n{'=' * 50}")
        print(f"  {self.system_title.upper()}")
        print(f"{'=' * 50}")
        print("Select Mission Mode:")
        print("1. Waypoint Navigation (Point-to-Point)")
        print("2. Shape Following (Circle, Square, DXF profile)")
        print("q. Quit")

        while True:
            choice = input("\nEnter choice (1-2 or q): ").strip().lower()

            if choice == "1":
                return "waypoint"
            elif choice == "2":
                return "shape_following"
            elif choice == "q":
                print("Exiting.")
                sys.exit(0)
            else:
                print("Invalid choice. Please try again.")

    def get_user_position(
        self,
        position_type: str,
        default_pos: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """Get position input from user."""
        while True:
            try:
                x_input = input(f"{position_type.title()} X position (meters): ").strip()
                if x_input == "" and default_pos is not None:
                    return default_pos
                x = float(x_input)

                y_input = input(f"{position_type.title()} Y position (meters): ").strip()
                if y_input == "" and default_pos is not None:
                    return default_pos
                y = float(y_input)

                z_input = input(f"{position_type.title()} Z position (meters): ").strip()
                if z_input == "" and default_pos is not None:
                    return default_pos
                # Default Z to 0.0 if not provided and no default passed,
                # but logic above returns default_pos if input empty.
                # If default_pos is None, we need input.
                z = float(z_input) if z_input else 0.0

                return (x, y, z)
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                if default_pos is not None:
                    if self._confirm_use_default(default_pos):
                        return default_pos
            except KeyboardInterrupt:
                print("\nCancelled by user.")
                raise

    def get_user_orientation(
        self,
        orientation_type: str,
        default_angle: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """Get orientation input from user as 3D Euler angles (degrees)."""
        default_deg = tuple(np.degrees(default_angle)) if default_angle is not None else None

        def _read_component(label: str, default_rad: Optional[float]) -> float:
            if default_rad is not None:
                prompt = (
                    f"{orientation_type.title()} {label} (degrees, "
                    f"default {np.degrees(default_rad):.1f}): "
                )
            else:
                prompt = f"{orientation_type.title()} {label} (degrees): "
            value = input(prompt).strip()
            if value == "":
                if default_rad is None:
                    raise ValueError("Missing value")
                return float(default_rad)
            return float(np.radians(float(value)))

        while True:
            try:
                roll = _read_component("roll", default_angle[0] if default_angle else None)
                pitch = _read_component("pitch", default_angle[1] if default_angle else None)
                yaw = _read_component("yaw", default_angle[2] if default_angle else None)
                return (roll, pitch, yaw)
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                if default_angle is not None:
                    default_label = (
                        f"roll={default_deg[0]:.1f}°, "
                        f"pitch={default_deg[1]:.1f}°, "
                        f"yaw={default_deg[2]:.1f}°"
                    )
                    if self._confirm_use_default(default_label):
                        return default_angle
            except KeyboardInterrupt:
                print("\nCancelled by user.")
                raise

    def get_user_velocities(
        self,
        default_vx: float = 0.0,
        default_vy: float = 0.0,
        default_vz: float = 0.0,
        default_omega: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """Get initial velocity values from user input.

        Args:
            default_vx: Default X velocity in m/s
            default_vy: Default Y velocity in m/s
            default_vz: Default Z velocity in m/s
            default_omega: Default angular velocity (Z-axis) in rad/s

        Returns:
            Tuple of (vx, vy, vz, omega) velocities
        """
        while True:
            try:
                vx_input = input(f"X velocity (m/s, default: {default_vx:.3f}): ").strip()
                vy_input = input(f"Y velocity (m/s, default: {default_vy:.3f}): ").strip()
                vz_input = input(f"Z velocity (m/s, default: {default_vz:.3f}): ").strip()
                omega_input = input(
                    f"Angular velocity Z (rad/s, default: {default_omega:.3f}): "
                ).strip()

                vx = float(vx_input) if vx_input else default_vx
                vy = float(vy_input) if vy_input else default_vy
                vz = float(vz_input) if vz_input else default_vz
                omega = float(omega_input) if omega_input else default_omega

                return (vx, vy, vz, omega)

            except ValueError:
                print("Invalid velocity input. Please enter numeric values.")
            except KeyboardInterrupt:
                print(f"\n{self.system_title} cancelled by user.")
                raise

    def _confirm_use_default(self, value: Any) -> bool:
        """Helper to confirm default value usage."""
        use_default = input(f"Use default ({value})? (y/n): ").strip().lower()
        return use_default == "y"

    def confirm_mission(self, mission_type: str) -> bool:
        """Confirm mission start."""
        confirm = input(f"\nProceed with {mission_type} simulation? (y/n): ").strip().lower()
        if confirm != "y":
            print("Mission cancelled.")
            return False
        return True

    def configure_obstacles(self) -> None:
        """Configure obstacles with preset menu or custom input."""
        SatelliteConfig.clear_obstacles()

        print("\n=== Obstacle Configuration ===")
        print("1. No obstacles")
        print("2. Single central obstacle (0, 0)")
        print("3. Obstacle corridor (gap between two obstacles)")
        print("4. Scattered obstacles (4 random positions)")
        print("5. Custom (manual entry)")

        choice = input("Select option (1-5, default 1): ").strip()

        if choice == "2":
            # Single central obstacle
            SatelliteConfig.add_obstacle(0.0, 0.0, 0.3)
            print("  Added: Central obstacle at (0.0, 0.0), r=0.30")

        elif choice == "3":
            # Corridor - two obstacles with gap in middle
            SatelliteConfig.add_obstacle(0.0, 0.4, 0.25)
            SatelliteConfig.add_obstacle(0.0, -0.4, 0.25)
            print("  Added: Corridor with gap at Y=0")
            print("    Obstacle 1: (0.0, 0.4), r=0.25")
            print("    Obstacle 2: (0.0, -0.4), r=0.25")

        elif choice == "4":
            # Scattered obstacles
            positions = [
                (0.5, 0.5, 0.2),
                (-0.5, 0.5, 0.2),
                (0.5, -0.5, 0.2),
                (-0.5, -0.5, 0.2),
            ]
            for x, y, r in positions:
                SatelliteConfig.add_obstacle(x, y, r)
            print("  Added: 4 scattered obstacles at corners")
            for x, y, r in positions:
                print(f"    ({x:.1f}, {y:.1f}), r={r:.2f}")

        elif choice == "5":
            # Custom - manual entry
            self._configure_obstacles_manual()

        else:
            # No obstacles (choice == "1" or invalid)
            print("  No obstacles configured.")

        if SatelliteConfig.get_obstacles():
            SatelliteConfig.OBSTACLES_ENABLED = True
            num_obs = len(SatelliteConfig.get_obstacles())
            print(f"\nObstacles enabled: {num_obs} obstacle(s) configured.")
            # Offer edit/modify options
            self._obstacle_edit_menu()
        else:
            SatelliteConfig.OBSTACLES_ENABLED = False

    def _obstacle_edit_menu(self) -> None:
        """Interactive menu to edit/remove obstacles after config."""
        while True:
            obstacles = SatelliteConfig.get_obstacles()
            if not obstacles:
                print("\nNo obstacles configured.")
                break

            print(f"\nCurrent obstacles ({len(obstacles)}):")
            for i, (x, y, r) in enumerate(obstacles, 1):
                print(f"  {i}. ({x:.2f}, {y:.2f}) r={r:.2f}")

            print("\nOptions: [A]dd, [E]dit #, [R]emove #, [C]lear all, [D]one")
            choice = input("Choice: ").strip().lower()

            if choice == "d" or choice == "":
                break
            elif choice == "a":
                self._add_single_obstacle()
            elif choice == "c":
                SatelliteConfig.clear_obstacles()
                SatelliteConfig.OBSTACLES_ENABLED = False
                print("  All obstacles cleared.")
                break
            elif choice.startswith("e") and len(choice) > 1:
                try:
                    idx = int(choice[1:].strip()) - 1
                    if 0 <= idx < len(obstacles):
                        self._edit_obstacle(idx, obstacles)
                    else:
                        print(f"  Invalid index. Use 1-{len(obstacles)}")
                except ValueError:
                    print("  Invalid format. Use 'E1', 'E2', etc.")
            elif choice.startswith("r") and len(choice) > 1:
                try:
                    idx = int(choice[1:].strip()) - 1
                    if 0 <= idx < len(obstacles):
                        removed = obstacles.pop(idx)
                        SatelliteConfig.set_obstacles(obstacles)
                        rx, ry = removed[0], removed[1]
                        print(f"  Removed obstacle at ({rx:.2f}, {ry:.2f})")
                    else:
                        print(f"  Invalid index. Use 1-{len(obstacles)}")
                except ValueError:
                    print("  Invalid format. Use 'R1', 'R2', etc.")
            else:
                print("  Unknown command. Use A/E#/R#/C/D")

    def _add_single_obstacle(self) -> None:
        """Add a single obstacle interactively."""
        try:
            obs_x = float(input("  X position (meters): "))
            obs_y = float(input("  Y position (meters): "))
            obs_r_input = input("  Radius (meters, default 0.3): ").strip()
            obs_r = float(obs_r_input) if obs_r_input else 0.3
            SatelliteConfig.add_obstacle(obs_x, obs_y, obs_r)
        except ValueError:
            print("  Invalid input, obstacle not added.")

    def _edit_obstacle(self, idx: int, obstacles: List) -> None:
        """Edit an existing obstacle."""
        old = obstacles[idx]
        print(f"  Editing obstacle {idx+1}: " f"({old[0]:.2f}, {old[1]:.2f}) r={old[2]:.2f}")
        try:
            x_input = input(f"  New X (enter for {old[0]:.2f}): ").strip()
            y_input = input(f"  New Y (enter for {old[1]:.2f}): ").strip()
            r_input = input(f"  New radius (enter for {old[2]:.2f}): ").strip()

            new_x = float(x_input) if x_input else old[0]
            new_y = float(y_input) if y_input else old[1]
            new_r = float(r_input) if r_input else old[2]

            obstacles[idx] = (new_x, new_y, new_r)
            SatelliteConfig.set_obstacles(obstacles)
            print(f"  Updated: ({new_x:.2f}, {new_y:.2f}) r={new_r:.2f}")
        except ValueError:
            print("  Invalid input, obstacle unchanged.")

    def _configure_obstacles_manual(self) -> None:
        """Manual obstacle entry (legacy method)."""
        add_obs = input("Add obstacle? (y/n): ").strip().lower()
        while add_obs == "y":
            try:
                obs_x = float(input("  Obstacle X position (meters): "))
                obs_y = float(input("  Obstacle Y position (meters): "))
                obs_r_input = input("  Obstacle radius (meters, default 0.5): ").strip()
                obs_r = float(obs_r_input) if obs_r_input else 0.5
                SatelliteConfig.add_obstacle(obs_x, obs_y, obs_r)
                ox, oy, or_ = obs_x, obs_y, obs_r
                print(f"  Obstacle added: ({ox:.2f}, {oy:.2f}), r={or_:.2f}")
            except ValueError:
                print("  Invalid input, skipping obstacle.")
            except KeyboardInterrupt:
                print(f"\n{self.system_title} cancelled by user.")
                raise

            add_obs = input("Add another obstacle? (y/n): ").strip().lower()

    def select_mission_preset(self) -> Optional[Dict[str, Any]]:
        """Select a mission preset for quick start.

        Returns:
            Mission config dict if preset selected, None for custom mission.
        """
        print("\n=== Mission Setup ===")
        print("1. Custom mission (manual entry)")
        print("2. Demo: Simple (1,1) → (0,0)")
        print("3. Demo: Diagonal with obstacle")
        print("4. Demo: Multi-waypoint square")
        print("5. Demo: Corridor navigation")

        choice = input("Select option (1-5, default 1): ").strip()

        if choice == "2":
            # Simple demo: corner to origin
            print("\n  Preset: Simple navigation (1,1) → (0,0)")
            SatelliteConfig.clear_obstacles()
            SatelliteConfig.OBSTACLES_ENABLED = False

            if not self.confirm_mission("simple demo"):
                return {}

            # Configure waypoints
            self._configure_preset_waypoints(
                start_pos=(1.0, 1.0, 0.0),
                start_angle=(0.0, 0.0, np.radians(90)),
                targets=[((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))],
            )

            return {
                "mission_type": "waypoint_navigation",
                "mode": "multi_point",
                "start_pos": (1.0, 1.0, 0.0),
                "start_angle": (0.0, 0.0, np.radians(90)),
                "start_vx": 0.0,
                "start_vy": 0.0,
                "start_vz": 0.0,
                "start_omega": 0.0,
            }

        elif choice == "3":
            # Diagonal with central obstacle
            print("\n  Preset: Diagonal with central obstacle")
            print("    (1,1) → (-1,-1) avoiding obstacle at (0,0)")
            SatelliteConfig.clear_obstacles()
            SatelliteConfig.add_obstacle(0.0, 0.0, 0.3)
            SatelliteConfig.OBSTACLES_ENABLED = True

            if not self.confirm_mission("obstacle avoidance demo"):
                return {}

            # Configure waypoints
            self._configure_preset_waypoints(
                start_pos=(1.0, 1.0, 0.0),
                start_angle=(0.0, 0.0, np.radians(45)),
                targets=[((-1.0, -1.0, 0.0), (0.0, 0.0, np.radians(-135)))],
            )

            return {
                "mission_type": "waypoint_navigation",
                "mode": "multi_point",
                "start_pos": (1.0, 1.0, 0.0),
                "start_angle": (0.0, 0.0, np.radians(45)),
                "start_vx": 0.0,
                "start_vy": 0.0,
                "start_vz": 0.0,
                "start_omega": 0.0,
            }

        elif choice == "4":
            # Multi-waypoint square pattern
            print("\n  Preset: Multi-waypoint square pattern")
            print("    (0,0) → (1,0) → (1,1) → (0,1) → (0,0)")
            SatelliteConfig.clear_obstacles()
            SatelliteConfig.OBSTACLES_ENABLED = False

            if not self.confirm_mission("multi-waypoint demo"):
                return {}

            targets = [
                ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                ((1.0, 1.0, 0.0), (0.0, 0.0, np.radians(90))),
                ((0.0, 1.0, 0.0), (0.0, 0.0, np.radians(180))),
                ((0.0, 0.0, 0.0), (0.0, 0.0, np.radians(270))),
            ]
            self._configure_preset_waypoints(
                start_pos=(0.0, 0.0, 0.0),
                start_angle=(0.0, 0.0, 0.0),
                targets=targets,
            )

            return {
                "mission_type": "waypoint_navigation",
                "mode": "multi_point",
                "start_pos": (0.0, 0.0, 0.0),
                "start_angle": (0.0, 0.0, 0.0),
                "start_vx": 0.0,
                "start_vy": 0.0,
                "start_vz": 0.0,
                "start_omega": 0.0,
            }

        elif choice == "5":
            # Corridor navigation
            print("\n  Preset: Corridor navigation")
            print("    (1,0) → (-1,0) through gap between obstacles")
            SatelliteConfig.clear_obstacles()
            SatelliteConfig.add_obstacle(0.0, 0.5, 0.3)
            SatelliteConfig.add_obstacle(0.0, -0.5, 0.3)
            SatelliteConfig.OBSTACLES_ENABLED = True

            if not self.confirm_mission("corridor navigation demo"):
                return {}

            self._configure_preset_waypoints(
                start_pos=(1.0, 0.0, 0.0),
                start_angle=(0.0, 0.0, np.radians(180)),
                targets=[((-1.0, 0.0, 0.0), (0.0, 0.0, np.radians(180)))],
            )

            return {
                "mission_type": "waypoint_navigation",
                "mode": "multi_point",
                "start_pos": (1.0, 0.0, 0.0),
                "start_angle": (0.0, 0.0, np.radians(180)),
                "start_vx": 0.0,
                "start_vy": 0.0,
                "start_vz": 0.0,
                "start_omega": 0.0,
            }

        # Default: return None for custom mission flow
        return None

    def _configure_preset_waypoints(
        self,
        start_pos: Tuple[float, float, float],
        start_angle: Tuple[float, float, float],
        targets: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    ) -> None:
        """Configure SatelliteConfig for preset waypoint missions."""
        target_positions = [t[0] for t in targets]
        target_angles = [t[1] for t in targets]

        SatelliteConfig.DEFAULT_START_POS = start_pos
        SatelliteConfig.DEFAULT_START_ANGLE = start_angle
        SatelliteConfig.DEFAULT_TARGET_POS = target_positions[0]
        SatelliteConfig.DEFAULT_TARGET_ANGLE = target_angles[0]

        SatelliteConfig.set_multi_point_mode(True)
        SatelliteConfig.ENABLE_WAYPOINT_MODE = True
        SatelliteConfig.WAYPOINT_TARGETS = target_positions
        SatelliteConfig.WAYPOINT_ANGLES = target_angles
        SatelliteConfig.CURRENT_TARGET_INDEX = 0
        SatelliteConfig.TARGET_STABILIZATION_START_TIME = None

    def run_multi_point_mode(self) -> Dict[str, Any]:
        """Run the multi-point waypoint mission workflow."""
        # 1. Try to select a preset first
        preset_config = self.select_mission_preset()
        if preset_config:
            # Presets handle their own obstacles, just return.
            return preset_config

        # 2. If no preset selected (returns None), do custom configuration
        print("\n=== Custom Waypoint Mission ===")
        print("Define start position and sequence of target waypoints.")

        start_pos = self.get_user_position("starting")
        start_angle = self.get_user_orientation("starting", (0.0, 0.0, 0.0))
        start_vx, start_vy, start_vz, start_omega = self.get_user_velocities()

        # Get targets
        targets: List[Tuple[float, float, float]] = []
        angles: List[Tuple[float, float, float]] = []

        print("\nDefine Target Waypoints (enter empty X to finish):")
        counter = 1
        while True:
            print(f"-- Waypoint {counter} --")
            try:
                x_input = input(f"Target {counter} X (meters): ").strip()
                if not x_input:
                    if not targets:
                        print("At least one waypoint is required.")
                        continue
                    break

                x = float(x_input)
                y_input = input(f"Target {counter} Y (meters): ").strip()
                y = float(y_input) if y_input else 0.0
                z_input = input(f"Target {counter} Z (meters): ").strip()
                z = float(z_input) if z_input else 0.0

                angle = self.get_user_orientation(f"Target {counter}", (0.0, 0.0, 0.0))

                targets.append((x, y, z))
                angles.append(angle)
                counter += 1
            except ValueError:
                print("Invalid input. Numeric values required.")

        self.configure_obstacles()

        # Update Config
        SatelliteConfig.DEFAULT_START_POS = start_pos
        SatelliteConfig.DEFAULT_START_ANGLE = start_angle
        SatelliteConfig.DEFAULT_TARGET_POS = targets[0]
        SatelliteConfig.DEFAULT_TARGET_ANGLE = angles[0]

        SatelliteConfig.set_multi_point_mode(True)
        SatelliteConfig.ENABLE_WAYPOINT_MODE = True
        SatelliteConfig.WAYPOINT_TARGETS = targets
        SatelliteConfig.WAYPOINT_ANGLES = angles
        SatelliteConfig.CURRENT_TARGET_INDEX = 0
        SatelliteConfig.TARGET_STABILIZATION_START_TIME = None

        return {
            "mission_type": "waypoint_navigation",
            "mode": "multi_point",
            "start_pos": start_pos,
            "start_angle": start_angle,
            "start_vx": start_vx,
            "start_vy": start_vy,
            "start_vz": start_vz,
            "start_omega": start_omega,
        }
