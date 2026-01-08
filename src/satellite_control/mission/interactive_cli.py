"""
Interactive Mission CLI Module

Enhanced user interface using questionary for styled menus
and rich for formatted output.
"""

import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.mission.mission_logic import MissionLogic

# Custom style for questionary
MISSION_STYLE = Style(
    [
        ("qmark", "fg:gray"),  # Subtle gray instead of bold cyan
        ("question", "fg:white"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
        ("separator", "fg:gray"),
        ("instruction", "fg:gray italic"),
    ]
)

# Use a subtle arrow instead of "?" for prompts
QMARK = "›"

console = Console()


class InteractiveMissionCLI:
    """
    Interactive mission CLI using questionary for styled menus.

    Provides a modern terminal UI experience with:
    - Arrow key navigation
    - Visual mission presets
    - Real-time input validation
    - Parameter preview before confirmation
    """

    def __init__(self, logic: Optional[MissionLogic] = None):
        self.logic = logic or MissionLogic()
        self.system_title = "Satellite Control Simulation"

    def show_welcome_banner(self) -> None:
        """Display welcome banner with system info."""
        banner = Text()
        banner.append("◈  ", style="bold")
        banner.append("SATELLITE CONTROL SYSTEM", style="bold cyan")
        banner.append("  ◈", style="bold")

        console.print()
        console.print(
            Panel(
                banner,
                subtitle="MPC-Based Precision Control",
                style="cyan",
                padding=(0, 2),
            )
        )

    def show_mission_menu(self) -> str:
        """Show main mission selection menu with styling."""
        self.show_welcome_banner()

        choices = [
            questionary.Choice(
                title="›  Point-to-Point Navigation",
                value="waypoint",
            ),
            questionary.Choice(
                title="◇  Shape Following (DXF/Demo)",
                value="shape_following",
            ),
            questionary.Separator(),
            questionary.Choice(
                title="×  Exit",
                value="exit",
            ),
        ]

        result = questionary.select(
            "Select Mission Type:",
            choices=choices,
            style=MISSION_STYLE,
            instruction="(Use arrow keys)",
            qmark=QMARK,
        ).ask()

        if result == "exit" or result is None:
            console.print("[yellow]Exiting...[/yellow]")
            sys.exit(0)

        return str(result)

    def select_mission_preset(
        self, return_simulation_config: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Select a mission preset with visual preview.

        Args:
            return_simulation_config: If True, also create and return a SimulationConfig
                with MissionState updated from the preset (v2.0.0 path).
        """
        console.print()
        console.print(Panel("Mission Presets", style="blue"))

        choices = [
            questionary.Choice(
                title="○  Custom Mission (manual configuration)",
                value="custom",
            ),
            questionary.Separator("─── Quick Start Demos ───"),
            questionary.Choice(
                title="●  Simple: (1,1,1) → (0,0,0)",
                value="simple",
            ),
            questionary.Choice(
                title="◆  Obstacle Avoidance: diagonal 3D path",
                value="obstacle",
            ),
            questionary.Choice(
                title="◇  Multi-Waypoint: 3D square ramp",
                value="square",
            ),
            questionary.Choice(
                title="▫  Corridor: navigate through gap + Z",
                value="corridor",
            ),
        ]

        result = questionary.select(
            "Select mission preset:",
            choices=choices,
            style=MISSION_STYLE,
            qmark=QMARK,
        ).ask()

        if result is None:
            return None

        if result == "custom":
            return None  # Signal to use custom flow

        # V3.0.0: Always create SimulationConfig and MissionState
        preset = self._get_preset_config(result)
        if not preset:
            return {}

        # Create SimulationConfig and update its MissionState using the same preset
        simulation_config = SimulationConfig.create_default()
        mission_state = simulation_config.mission_state
        self._apply_preset(preset, mission_state=mission_state)

        preset["simulation_config"] = simulation_config
        return preset

    def _get_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """Get configuration for a preset mission."""
        presets = {
            "simple": {
                "name": "Simple Navigation",
                "start_pos": (1.0, 1.0, 1.0),
                "start_angle": (0.0, 0.0, np.radians(90)),
                "targets": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))],
                "obstacles": [],
            },
            "obstacle": {
                "name": "Obstacle Avoidance",
                "start_pos": (1.0, 1.0, 1.0),
                "start_angle": (0.0, 0.0, np.radians(45)),
                "targets": [((-1.0, -1.0, 0.0), (0.0, 0.0, np.radians(-135)))],
                "obstacles": [(0.0, 0.0, 0.3)],
            },
            "square": {
                "name": "Square Pattern",
                "start_pos": (0.0, 0.0, 0.5),
                "start_angle": (0.0, 0.0, 0.0),
                "targets": [
                    ((1.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
                    ((1.0, 1.0, 0.5), (0.0, 0.0, np.radians(90))),
                    ((0.0, 1.0, 0.0), (0.0, 0.0, np.radians(180))),
                    ((0.0, 0.0, 0.5), (0.0, 0.0, np.radians(270))),
                ],
                "obstacles": [],
            },
            "corridor": {
                "name": "Corridor Navigation",
                "start_pos": (1.0, 0.0, 1.0),
                "start_angle": (0.0, 0.0, np.radians(180)),
                "targets": [((-1.0, 0.0, 0.5), (0.0, 0.0, np.radians(180)))],
                "obstacles": [(0.0, 0.5, 0.3), (0.0, -0.5, 0.3)],
            },
        }

        preset = presets.get(preset_name, presets["simple"])

        # Show preview
        self._show_mission_preview(preset)

        if not self._confirm_mission(preset["name"]):
            return {}

        # Return full preset dict (includes obstacles, targets, etc.)
        # This allows _apply_preset to work correctly
        preset["mission_type"] = "waypoint_navigation"
        preset["mode"] = "multi_point"
        preset["preset_name"] = preset_name
        return preset

    @staticmethod
    def _format_euler_deg(euler: Tuple[float, float, float]) -> str:
        roll, pitch, yaw = np.degrees(euler)
        return f"roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°"

    def _show_mission_preview(self, preset: Dict[str, Any]) -> None:
        """Show a visual preview of the mission configuration."""
        table = Table(title=f"◇ {preset['name']}", style="cyan")
        table.add_column("Parameter", style="bold")
        table.add_column("Value", style="green")

        # Start position
        sp = preset["start_pos"]
        sa = self._format_euler_deg(preset["start_angle"])
        if len(sp) == 3:
            table.add_row("Start Position", f"({sp[0]:.1f}, {sp[1]:.1f}, {sp[2]:.1f}) m")
        else:
            table.add_row("Start Position", f"({sp[0]:.1f}, {sp[1]:.1f}) m")
        table.add_row("Start Angle", sa)

        # Waypoints
        for i, (pos, angle) in enumerate(preset["targets"], 1):
            a_deg = self._format_euler_deg(angle)
            if len(pos) == 3:
                table.add_row(
                    f"Waypoint {i}",
                    f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) m @ {a_deg}",
                )
            else:
                table.add_row(f"Waypoint {i}", f"({pos[0]:.1f}, {pos[1]:.1f}) m @ {a_deg}")

        # Obstacles
        if preset["obstacles"]:
            for i, (x, y, r) in enumerate(preset["obstacles"], 1):
                table.add_row(f"Obstacle {i}", f"({x:.1f}, {y:.1f}) r={r:.2f}m")
        else:
            table.add_row("Obstacles", "None")

        console.print()
        console.print(table)

    def _confirm_mission(self, mission_name: str) -> bool:
        """Confirm mission start with styled prompt."""
        return (
            questionary.confirm(
                f"Proceed with {mission_name}?",
                default=True,
                style=MISSION_STYLE,
                qmark=QMARK,
            ).ask()
            or False
        )

    def _apply_preset(self, preset: Dict[str, Any], mission_state) -> None:
        """Apply preset configuration.
        
        Args:
            preset: Preset configuration dictionary
            mission_state: MissionState to update (required in V3.0.0).
        """
        obstacles = preset.get("obstacles", [])
        
        # V3.0.0: Always require mission_state (no legacy fallback)
        if mission_state is None:
            raise ValueError("mission_state is required (V3.0.0: no SatelliteConfig fallback)")
        
        # Update MissionState
        mission_state.obstacles = obstacles
        mission_state.obstacles_enabled = len(obstacles) > 0
        
        targets = preset.get("targets", [])
        # Extract yaw from angles if they're tuples
        waypoint_angles = [t[1][2] if isinstance(t[1], tuple) and len(t[1]) == 3 else t[1] for t in targets]
        
        mission_state.enable_waypoint_mode = True
        mission_state.enable_multi_point_mode = True
        mission_state.waypoint_targets = [t[0] for t in targets]
        mission_state.waypoint_angles = waypoint_angles
        mission_state.current_target_index = 0
        mission_state.target_stabilization_start_time = None

    def get_position_interactive(
        self,
        prompt: str,
        default: Tuple[float, ...] = (0.0, 0.0, 0.0),
        dim: int = 3,
    ) -> Tuple[float, ...]:
        """Get position with interactive validation."""
        console.print(f"\n[bold]{prompt}[/bold]")

        def_x = default[0] if len(default) > 0 else 0.0
        def_y = default[1] if len(default) > 1 else 0.0
        def_z = default[2] if len(default) > 2 else 0.0

        x = questionary.text(
            f"X position (meters) [{def_x:.2f}]:",
            default=str(def_x),
            validate=lambda x: self._validate_float(x),
            style=MISSION_STYLE,
            qmark=QMARK,
        ).ask()

        y = questionary.text(
            f"Y position (meters) [{def_y:.2f}]:",
            default=str(def_y),
            validate=lambda x: self._validate_float(x),
            style=MISSION_STYLE,
            qmark=QMARK,
        ).ask()

        if dim == 3:
            z = questionary.text(
                f"Z position (meters) [{def_z:.2f}]:",
                default=str(def_z),
                validate=lambda x: self._validate_float(x),
                style=MISSION_STYLE,
                qmark=QMARK,
            ).ask()
            return (float(x or def_x), float(y or def_y), float(z or def_z))

        return (float(x or def_x), float(y or def_y))

    def get_angle_interactive(
        self,
        prompt: str,
        default_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Tuple[float, float, float]:
        """Get 3D Euler angles with interactive validation."""
        roll_default, pitch_default, yaw_default = default_deg
        roll = questionary.text(
            f"{prompt} roll (degrees) [{roll_default:.1f}]:",
            default=str(roll_default),
            validate=lambda x: self._validate_float(x),
            style=MISSION_STYLE,
            qmark=QMARK,
        ).ask()
        pitch = questionary.text(
            f"{prompt} pitch (degrees) [{pitch_default:.1f}]:",
            default=str(pitch_default),
            validate=lambda x: self._validate_float(x),
            style=MISSION_STYLE,
            qmark=QMARK,
        ).ask()
        yaw = questionary.text(
            f"{prompt} yaw (degrees) [{yaw_default:.1f}]:",
            default=str(yaw_default),
            validate=lambda x: self._validate_float(x),
            style=MISSION_STYLE,
            qmark=QMARK,
        ).ask()

        return (
            float(np.radians(float(roll or roll_default))),
            float(np.radians(float(pitch or pitch_default))),
            float(np.radians(float(yaw or yaw_default))),
        )

    def configure_obstacles_interactive(self, mission_state) -> Tuple[List[Tuple[float, float, float]], bool]:
        """Configure obstacles with interactive menu.
        
        Args:
            mission_state: MissionState to update (required in V3.0.0).
            
        Returns:
            Tuple of (obstacles list, obstacles_enabled bool).
        """
        # V3.0.0: Always require mission_state (no legacy fallback)
        if mission_state is None:
            raise ValueError("mission_state is required (V3.0.0: no SatelliteConfig fallback)")
        
        console.print()
        console.print(Panel("Obstacle Configuration", style="yellow"))

        choices = [
            questionary.Choice(title="○  No obstacles", value="none"),
            questionary.Choice(title="●  Central obstacle", value="central"),
            questionary.Choice(title="◇  Corridor (two obstacles)", value="corridor"),
            questionary.Choice(title="◆  Scattered (four corners)", value="scattered"),
            questionary.Choice(title="▫  Custom (manual entry)", value="custom"),
        ]

        result = questionary.select(
            "Select obstacle configuration:",
            choices=choices,
            style=MISSION_STYLE,
            qmark=QMARK,
        ).ask()

        obstacles: List[Tuple[float, float, float]] = []

        if result == "central":
            obstacles.append((0.0, 0.0, 0.3))
            console.print("[green]+ Added central obstacle at (0, 0)[/green]")

        elif result == "corridor":
            obstacles.extend([(0.0, 0.4, 0.25), (0.0, -0.4, 0.25)])
            console.print("[green]+ Added corridor obstacles[/green]")

        elif result == "scattered":
            scattered = [(0.5, 0.5, 0.2), (-0.5, 0.5, 0.2), (0.5, -0.5, 0.2), (-0.5, -0.5, 0.2)]
            obstacles.extend(scattered)
            console.print("[green]+ Added 4 corner obstacles[/green]")

        elif result == "custom":
            obstacles = self._add_custom_obstacles(mission_state)

        obstacles_enabled = len(obstacles) > 0
        
        # Update MissionState
        mission_state.obstacles = obstacles
        mission_state.obstacles_enabled = obstacles_enabled
        
        return obstacles, obstacles_enabled

    def _add_custom_obstacles(self, mission_state) -> List[Tuple[float, float, float]]:
        """Add custom obstacles interactively.
        
        Args:
            mission_state: MissionState to update (required in V3.0.0).
        
        Returns:
            List of obstacles as (x, y, radius) tuples.
        """
        obstacles: List[Tuple[float, float, float]] = []
        while True:
            add_more = questionary.confirm(
                "Add an obstacle?",
                default=False,
                style=MISSION_STYLE,
            ).ask()

            if not add_more:
                break

            pos = self.get_position_interactive("Obstacle position", dim=2)
            radius = questionary.text(
                "Radius (meters) [0.3]:",
                default="0.3",
                validate=lambda x: self._validate_positive_float(x),
                style=MISSION_STYLE,
            ).ask()

            r = float(radius or 0.3)
            obstacles.append((pos[0], pos[1], r))
            console.print(f"[green]+ Added obstacle at {pos}[/green]")
        
        # V3.0.0: Update mission_state directly
        mission_state.obstacles = obstacles
        mission_state.obstacles_enabled = len(obstacles) > 0
        
        return obstacles

    def run_custom_waypoint_mission(self) -> Dict[str, Any]:
        """Run custom waypoint mission configuration."""
        # Create SimulationConfig for v2.0.0 pattern
        simulation_config = SimulationConfig.create_default()
        mission_state = simulation_config.mission_state
        
        console.print()
        console.print(Panel("Custom Waypoint Mission", style="green"))

        # Get start position
        start_pos = self.get_position_interactive(
            "Starting Position",
            default=(1.0, 1.0, 0.0),
        )
        start_angle = self.get_angle_interactive("Starting Angle", (0.0, 0.0, 0.0))

        # Get waypoints
        waypoints: List[Tuple[Tuple[float, float], Tuple[float, float, float]]] = []

        console.print("\n[bold]Define waypoints[/bold] (at least 1 required)")

        while True:
            wp_num = len(waypoints) + 1
            console.print(f"\n[cyan]Waypoint {wp_num}[/cyan]")

            pos = self.get_position_interactive(f"Target {wp_num}", (0.0, 0.0, 0.0))
            angle = self.get_angle_interactive(
                f"Target {wp_num} angle", (0.0, 0.0, 0.0)
            )

            waypoints.append((pos, angle))

            if len(waypoints) >= 1:
                add_more = questionary.confirm(
                    "Add another waypoint?",
                    default=False,
                    style=MISSION_STYLE,
                ).ask()
                if not add_more:
                    break

        # Configure obstacles
        self.configure_obstacles_interactive(mission_state)

        # Show summary
        self._show_custom_mission_summary(start_pos, start_angle, waypoints, mission_state)

        if not self._confirm_mission("Custom Waypoint Mission"):
            return {}

        # Extract yaw from angles for waypoint_angles
        waypoint_angles = [wp[1][2] if isinstance(wp[1], tuple) and len(wp[1]) == 3 else wp[1] for wp in waypoints]
        
        # Update MissionState (V3.0.0)
        mission_state.enable_waypoint_mode = True
        mission_state.enable_multi_point_mode = True
        mission_state.waypoint_targets = [wp[0] for wp in waypoints]
        mission_state.waypoint_angles = waypoint_angles
        mission_state.current_target_index = 0
        mission_state.target_stabilization_start_time = None

        return {
            "mission_type": "waypoint_navigation",
            "mode": "multi_point",
            "start_pos": start_pos,
            "start_angle": start_angle,
            "simulation_config": simulation_config,
        }

    def _show_custom_mission_summary(
        self,
        start_pos: Tuple[float, float],
        start_angle: Tuple[float, float, float],
        waypoints: List[Tuple[Tuple[float, float], Tuple[float, float, float]]],
        mission_state,
    ) -> None:
        """Show summary of custom mission configuration."""
        table = Table(title="◇ Mission Summary", style="green")
        table.add_column("Parameter", style="bold")
        table.add_column("Value", style="cyan")

        sp = start_pos
        sa = self._format_euler_deg(start_angle)
        if len(sp) == 3:
            table.add_row("Start", f"({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f}) @ {sa}")
        else:
            table.add_row("Start", f"({sp[0]:.2f}, {sp[1]:.2f}) @ {sa}")

        for i, (pos, angle) in enumerate(waypoints, 1):
            a_deg = self._format_euler_deg(angle)
            if len(pos) == 3:
                table.add_row(
                    f"Waypoint {i}", f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) @ {a_deg}"
                )
            else:
                table.add_row(f"Waypoint {i}", f"({pos[0]:.2f}, {pos[1]:.2f}) @ {a_deg}")

        # V3.0.0: Read from mission_state instead of SatelliteConfig
        obstacles = mission_state.obstacles if mission_state else []
        if obstacles:
            table.add_row("Obstacles", f"{len(obstacles)} configured")
        else:
            table.add_row("Obstacles", "None")

        console.print()
        console.print(table)

    @staticmethod
    def _validate_float(value: str) -> bool:
        """Validate float input."""
        if not value:
            return True  # Allow empty for defaults
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _validate_positive_float(value: str) -> bool:
        """Validate positive float input."""
        if not value:
            return True
        try:
            return float(value) > 0
        except ValueError:
            return False

    # =========================================================================
    # Shape Following Mission
    # =========================================================================

    def run_shape_following_mission(self) -> Dict[str, Any]:
        """Run interactive shape following mission configuration."""
        from src.satellite_control.config import timing

        console.print()
        console.print(Panel("◇ Shape Following Mission", style="blue"))
        console.print("[dim]Track a moving target along a shape path[/dim]\n")

        # Get start position
        start_pos = self.get_position_interactive("Starting Position", default=(1.0, 1.0, 0.0))
        start_angle = self.get_angle_interactive("Starting Angle", (0.0, 0.0, 90.0))

        # Select shape type
        console.print()
        shape_type = self._select_shape_type()
        if shape_type is None:
            return {}

        # Get shape parameters
        shape_center = self.get_position_interactive("Shape Center", default=(0.0, 0.0, 0.0))

        rotation_deg = float(
            questionary.text(
                "Shape rotation (degrees) [0]:",
                default="0",
                validate=lambda x: self._validate_float(x),
                style=MISSION_STYLE,
                qmark=QMARK,
            ).ask()
            or "0"
        )

        offset = float(
            questionary.text(
                "Offset distance (meters) [0.5]:",
                default="0.5",
                validate=lambda x: self._validate_positive_float(x),
                style=MISSION_STYLE,
                qmark=QMARK,
            ).ask()
            or "0.5"
        )
        offset = max(0.1, min(2.0, offset))  # Clamp to valid range

        # Get target speed
        default_speed = timing.DEFAULT_TARGET_SPEED
        target_speed = float(
            questionary.text(
                f"Target speed (m/s) [{default_speed}]:",
                default=str(default_speed),
                validate=lambda x: self._validate_positive_float(x),
                style=MISSION_STYLE,
                qmark=QMARK,
            ).ask()
            or str(default_speed)
        )
        target_speed = max(0.01, min(0.5, target_speed))

        # Return position option
        has_return = questionary.confirm(
            "Return to start position after shape?",
            default=False,
            style=MISSION_STYLE,
            qmark=QMARK,
        ).ask()

        return_pos = start_pos if has_return else None
        return_angle = start_angle if has_return else None

        # Create SimulationConfig for v2.0.0 pattern
        simulation_config = SimulationConfig.create_default()
        mission_state = simulation_config.mission_state

        # Configure obstacles
        self.configure_obstacles_interactive(mission_state)

        # Generate shape and show preview
        if shape_type == "custom_dxf" and hasattr(self, "_custom_dxf_points"):
            shape_points = self._custom_dxf_points
        else:
            shape_points = self.logic.generate_demo_shape(shape_type)

        rotation_rad = np.radians(rotation_deg)
        transformed = self.logic.transform_shape(shape_points, shape_center, rotation_rad)
        upscaled_path = self.logic.upscale_shape(transformed, offset)
        path_length = self.logic.calculate_path_length(upscaled_path)

        # Improved duration estimation
        TRANSIT_SPEED = 0.15  # m/s average speed during point-to-point transit
        STABILIZATION_TIME = 5.0  # seconds to stabilize at each waypoint

        # 1. Time from start to first path point
        first_path_point = upscaled_path[0]
        dist_to_start = np.sqrt(
            (start_pos[0] - first_path_point[0]) ** 2 + (start_pos[1] - first_path_point[1]) ** 2
        )
        transit_to_path = dist_to_start / TRANSIT_SPEED

        # 2. Time to traverse path at target speed
        path_traverse_time = path_length / target_speed

        # 3. Final stabilization
        final_stabilization = 10.0  # seconds

        # 4. Return transit (if enabled)
        return_transit = 0.0
        if has_return and return_pos is not None:
            last_path_point = upscaled_path[-1]
            dist_return = np.sqrt(
                (return_pos[0] - last_path_point[0]) ** 2
                + (return_pos[1] - last_path_point[1]) ** 2
            )
            return_transit = dist_return / TRANSIT_SPEED + STABILIZATION_TIME

        estimated_duration = (
            transit_to_path
            + STABILIZATION_TIME
            + path_traverse_time
            + final_stabilization
            + return_transit
        )

        # Show summary
        self._show_shape_mission_summary(
            start_pos=start_pos,
            start_angle=start_angle,
            shape_type=shape_type,
            shape_center=shape_center,
            rotation_deg=rotation_deg,
            offset=offset,
            target_speed=target_speed,
            path_length=path_length,
            path_points=len(upscaled_path),
            estimated_duration=estimated_duration,
            has_return=has_return,
        )

        if not self._confirm_mission("Shape Following Mission"):
            return {}

        # Apply configuration to MissionState (V3.0.0)
        self._apply_shape_config(
            start_pos=start_pos,
            start_angle=start_angle,
            shape_center=shape_center,
            rotation_rad=rotation_rad,
            transformed_shape=transformed,
            upscaled_path=upscaled_path,
            target_speed=target_speed,
            path_length=path_length,
            estimated_duration=estimated_duration,
            has_return=has_return,
            return_pos=return_pos,
            return_angle=return_angle,
            mission_state=mission_state,
        )

        return {
            "mission_type": "shape_following",
            "start_pos": start_pos,
            "start_angle": start_angle,
            "shape_type": shape_type,
            "simulation_config": simulation_config,
        }

    def _select_shape_type(self) -> Optional[str]:
        """Select shape type with visual menu."""
        # Check if ezdxf is available
        try:
            import ezdxf  # noqa: F401

            dxf_available = True
        except ImportError:
            dxf_available = False

        choices = [
            questionary.Choice(title="○  Circle", value="circle"),
            questionary.Choice(title="□  Rectangle", value="rectangle"),
            questionary.Choice(title="△  Triangle", value="triangle"),
            questionary.Choice(title="⬡  Hexagon", value="hexagon"),
        ]

        if dxf_available:
            choices.insert(0, questionary.Separator("─── Demo Shapes ───"))
            choices.insert(
                0,
                questionary.Choice(
                    title="▢  Load from DXF file",
                    value="dxf",
                ),
            )

        result = questionary.select(
            "Select shape type:",
            choices=choices,
            style=MISSION_STYLE,
            qmark=QMARK,
        ).ask()

        if result == "dxf":
            loaded = self._load_dxf_shape()
            return str(loaded) if loaded else "circle"

        return str(result) if result else None

    def _load_dxf_shape(self) -> Optional[str]:
        """Load custom shape from DXF file picker."""
        from pathlib import Path

        # Find DXF files in the DXF folder
        dxf_folder = Path("DXF/DXF_Files")
        if not dxf_folder.exists():
            # Try alternate locations
            dxf_folder = Path(__file__).parents[3] / "DXF" / "DXF_Files"

        dxf_files = []
        if dxf_folder.exists():
            dxf_files = sorted([f for f in dxf_folder.glob("*.dxf")])

        if not dxf_files:
            console.print("[yellow]No DXF files found in DXF/DXF_Files/[/yellow]")
            console.print("[yellow]Using Circle instead.[/yellow]")
            return "circle"

        # Build choices from available DXF files
        choices = [
            questionary.Choice(
                title=f"▫  {f.name}",
                value=str(f),
            )
            for f in dxf_files
        ]
        choices.append(questionary.Separator())
        choices.append(
            questionary.Choice(
                title="▢  Enter custom path...",
                value="_custom_path",
            )
        )

        console.print()
        console.print(f"[dim]Found {len(dxf_files)} DXF files in {dxf_folder}[/dim]")

        result = questionary.select(
            "Select DXF file:",
            choices=choices,
            style=MISSION_STYLE,
            qmark=QMARK,
        ).ask()

        if result is None:
            return "circle"

        # Handle custom path option
        if result == "_custom_path":
            dxf_path = questionary.path(
                "Enter DXF file path:",
                style=MISSION_STYLE,
            ).ask()
            if not dxf_path:
                console.print("[yellow]No file selected, using Circle.[/yellow]")
                return "circle"
            result = dxf_path

        try:
            # Load DXF using MissionLogic
            shape_points = self.logic.load_dxf_shape(result)
            console.print(f"[green]+ Loaded {len(shape_points)} points from DXF[/green]")
            # Store the loaded points for later use
            self._custom_dxf_points = shape_points
            return "custom_dxf"
        except Exception as e:
            console.print(f"[red]Failed to load DXF: {e}[/red]")
            console.print("[yellow]Falling back to Circle.[/yellow]")
            return "circle"

    def _show_shape_mission_summary(
        self,
        start_pos: Tuple[float, float],
        start_angle: Tuple[float, float, float],
        shape_type: str,
        shape_center: Tuple[float, float],
        rotation_deg: float,
        offset: float,
        target_speed: float,
        path_length: float,
        path_points: int,
        estimated_duration: float,
        has_return: bool,
    ) -> None:
        """Show shape following mission summary."""
        table = Table(title="◇ Shape Following Summary", style="blue")
        table.add_column("Parameter", style="bold")
        table.add_column("Value", style="cyan")

        sa = self._format_euler_deg(start_angle)
        table.add_row("Start", f"({start_pos[0]:.1f}, {start_pos[1]:.1f}) @ {sa}")
        table.add_row("Shape", shape_type.title())
        table.add_row("Center", f"({shape_center[0]:.1f}, {shape_center[1]:.1f})")
        table.add_row("Rotation", f"{rotation_deg:.0f}°")
        table.add_row("Offset", f"{offset:.2f} m")
        table.add_row("Speed", f"{target_speed:.2f} m/s")
        table.add_row("Path Length", f"{path_length:.1f} m ({path_points} points)")
        table.add_row("Est. Duration", f"~{estimated_duration:.0f}s")
        table.add_row("Return", "Yes" if has_return else "No")

        console.print()
        console.print(table)
        console.print()

    def _apply_shape_config(
        self,
        start_pos: Tuple[float, float],
        start_angle: Tuple[float, float, float],
        shape_center: Tuple[float, float],
        rotation_rad: float,
        transformed_shape: List[Tuple[float, float]],
        upscaled_path: List[Tuple[float, float]],
        target_speed: float,
        path_length: float,
        estimated_duration: float,
        has_return: bool,
        return_pos: Optional[Tuple[float, float]],
        return_angle: Optional[Tuple[float, float, float]],
        mission_state,
    ) -> None:
        """Apply shape following configuration to MissionState.
        
        Args:
            start_pos: Starting position (x, y)
            start_angle: Starting angle (roll, pitch, yaw)
            shape_center: Shape center position (x, y)
            rotation_rad: Shape rotation in radians
            transformed_shape: Transformed shape points
            upscaled_path: Upscaled path points
            target_speed: Target speed in m/s
            path_length: Path length in meters
            estimated_duration: Estimated duration in seconds
            has_return: Whether to return to start
            return_pos: Return position (x, y) if has_return
            return_angle: Return angle if has_return
            mission_state: MissionState to update (required in V3.0.0).
        """
        # V3.0.0: Always require mission_state (no legacy fallback)
        if mission_state is None:
            raise ValueError("mission_state is required (V3.0.0: no SatelliteConfig fallback)")
        
        # Convert 2D to 3D for mission_state (add z=0.0)
        start_pos_3d = (start_pos[0], start_pos[1], 0.0) if len(start_pos) == 2 else start_pos
        shape_center_3d = (shape_center[0], shape_center[1], 0.0) if len(shape_center) == 2 else shape_center
        upscaled_path_3d = [(p[0], p[1], 0.0) if len(p) == 2 else p for p in upscaled_path]
        transformed_shape_3d = [(p[0], p[1], 0.0) if len(p) == 2 else p for p in transformed_shape]
        return_pos_3d = (return_pos[0], return_pos[1], 0.0) if return_pos and len(return_pos) == 2 else return_pos

        # Update MissionState
        mission_state.dxf_shape_mode_active = True
        mission_state.dxf_shape_center = shape_center_3d
        mission_state.dxf_base_shape = transformed_shape_3d
        mission_state.dxf_shape_path = upscaled_path_3d
        mission_state.dxf_target_speed = target_speed
        mission_state.dxf_estimated_duration = estimated_duration
        mission_state.dxf_mission_start_time = None
        mission_state.dxf_shape_phase = "POSITIONING"
        mission_state.dxf_path_length = path_length
        mission_state.dxf_has_return = has_return
        mission_state.dxf_return_position = return_pos_3d
        mission_state.dxf_return_angle = return_angle[2] if return_angle and len(return_angle) == 3 else return_angle

        # Clear transient state
        mission_state.dxf_tracking_start_time = None
        mission_state.dxf_target_start_distance = 0.0
        mission_state.dxf_stabilization_start_time = None
        mission_state.dxf_final_position = None
        mission_state.dxf_return_start_time = None
