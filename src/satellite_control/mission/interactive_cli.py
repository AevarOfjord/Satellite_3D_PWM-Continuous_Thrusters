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

from src.satellite_control.config import SatelliteConfig
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

    def select_mission_preset(self) -> Optional[Dict[str, Any]]:
        """Select a mission preset with visual preview."""
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

        # Return preset configuration
        return self._get_preset_config(result)

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

        # Configure SatelliteConfig
        self._apply_preset(preset)

        return {
            "mission_type": "waypoint_navigation",
            "mode": "multi_point",
            "start_pos": preset["start_pos"],
            "start_angle": preset["start_angle"],
            "preset_name": preset_name,
        }

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

    def _apply_preset(self, preset: Dict[str, Any]) -> None:
        """Apply preset configuration to SatelliteConfig."""
        # Clear and set obstacles
        SatelliteConfig.clear_obstacles()
        for x, y, r in preset["obstacles"]:
            SatelliteConfig.add_obstacle(x, y, r)
        SatelliteConfig.OBSTACLES_ENABLED = len(preset["obstacles"]) > 0

        # Set positions
        SatelliteConfig.DEFAULT_START_POS = preset["start_pos"]
        SatelliteConfig.DEFAULT_START_ANGLE = preset["start_angle"]

        targets = preset["targets"]
        SatelliteConfig.DEFAULT_TARGET_POS = targets[0][0]
        SatelliteConfig.DEFAULT_TARGET_ANGLE = targets[0][1]

        # Configure waypoint mode
        SatelliteConfig.set_multi_point_mode(True)
        SatelliteConfig.ENABLE_WAYPOINT_MODE = True
        SatelliteConfig.WAYPOINT_TARGETS = [t[0] for t in targets]
        SatelliteConfig.WAYPOINT_ANGLES = [t[1] for t in targets]
        SatelliteConfig.CURRENT_TARGET_INDEX = 0
        SatelliteConfig.TARGET_STABILIZATION_START_TIME = None

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

    def configure_obstacles_interactive(self) -> None:
        """Configure obstacles with interactive menu."""
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

        SatelliteConfig.clear_obstacles()

        if result == "central":
            SatelliteConfig.add_obstacle(0.0, 0.0, 0.3)
            console.print("[green]+ Added central obstacle at (0, 0)[/green]")

        elif result == "corridor":
            SatelliteConfig.add_obstacle(0.0, 0.4, 0.25)
            SatelliteConfig.add_obstacle(0.0, -0.4, 0.25)
            console.print("[green]+ Added corridor obstacles[/green]")

        elif result == "scattered":
            for x, y in [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]:
                SatelliteConfig.add_obstacle(x, y, 0.2)
            console.print("[green]+ Added 4 corner obstacles[/green]")

        elif result == "custom":
            self._add_custom_obstacles()

        SatelliteConfig.OBSTACLES_ENABLED = len(SatelliteConfig.get_obstacles()) > 0

    def _add_custom_obstacles(self) -> None:
        """Add custom obstacles interactively."""
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

            SatelliteConfig.add_obstacle(pos[0], pos[1], float(radius or 0.3))
            console.print(f"[green]+ Added obstacle at {pos}[/green]")

    def run_custom_waypoint_mission(self) -> Dict[str, Any]:
        """Run custom waypoint mission configuration."""
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
        self.configure_obstacles_interactive()

        # Show summary
        self._show_custom_mission_summary(start_pos, start_angle, waypoints)

        if not self._confirm_mission("Custom Waypoint Mission"):
            return {}

        # Apply configuration
        SatelliteConfig.DEFAULT_START_POS = start_pos
        SatelliteConfig.DEFAULT_START_ANGLE = start_angle
        SatelliteConfig.DEFAULT_TARGET_POS = waypoints[0][0]
        SatelliteConfig.DEFAULT_TARGET_ANGLE = waypoints[0][1]

        SatelliteConfig.set_multi_point_mode(True)
        SatelliteConfig.ENABLE_WAYPOINT_MODE = True
        SatelliteConfig.WAYPOINT_TARGETS = [wp[0] for wp in waypoints]
        SatelliteConfig.WAYPOINT_ANGLES = [wp[1] for wp in waypoints]
        SatelliteConfig.CURRENT_TARGET_INDEX = 0

        return {
            "mission_type": "waypoint_navigation",
            "mode": "multi_point",
            "start_pos": start_pos,
            "start_angle": start_angle,
        }

    def _show_custom_mission_summary(
        self,
        start_pos: Tuple[float, float],
        start_angle: Tuple[float, float, float],
        waypoints: List[Tuple[Tuple[float, float], Tuple[float, float, float]]],
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

        obstacles = SatelliteConfig.get_obstacles()
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

        # Configure obstacles
        self.configure_obstacles_interactive()

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

        # Apply configuration to SatelliteConfig
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
        )

        return {
            "mission_type": "shape_following",
            "start_pos": start_pos,
            "start_angle": start_angle,
            "shape_type": shape_type,
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
    ) -> None:
        """Apply shape following configuration to SatelliteConfig."""
        SatelliteConfig.DEFAULT_START_POS = start_pos
        SatelliteConfig.DEFAULT_START_ANGLE = start_angle

        # Shape following specific config - use setattr for dynamic attributes
        SatelliteConfig.DXF_SHAPE_MODE_ACTIVE = True
        setattr(SatelliteConfig, "DXF_SHAPE_CENTER", shape_center)
        SatelliteConfig.DXF_BASE_SHAPE = transformed_shape
        SatelliteConfig.DXF_SHAPE_PATH = upscaled_path
        SatelliteConfig.DXF_TARGET_SPEED = target_speed
        SatelliteConfig.DXF_ESTIMATED_DURATION = estimated_duration
        SatelliteConfig.DXF_MISSION_START_TIME = None
        SatelliteConfig.DXF_SHAPE_PHASE = "POSITIONING"
        SatelliteConfig.DXF_PATH_LENGTH = path_length
        SatelliteConfig.DXF_HAS_RETURN = has_return
        setattr(SatelliteConfig, "DXF_RETURN_POSITION", return_pos)
        setattr(SatelliteConfig, "DXF_RETURN_ANGLE", return_angle)

        # Clear transient state
        for attr in [
            "DXF_TRACKING_START_TIME",
            "DXF_TARGET_START_DISTANCE",
            "DXF_STABILIZATION_START_TIME",
            "DXF_FINAL_POSITION",
            "DXF_RETURN_START_TIME",
        ]:
            if hasattr(SatelliteConfig, attr):
                delattr(SatelliteConfig, attr)
