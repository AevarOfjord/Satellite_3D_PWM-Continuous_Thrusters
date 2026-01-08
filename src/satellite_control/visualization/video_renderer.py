"""
Video Renderer for Visualization

Provides video rendering functionality for animation generation.
Extracted from unified_visualizer.py to improve modularity.

This module handles all video/animation generation (MP4 output).
"""

import multiprocessing
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# V4.0.0: SatelliteConfig removed - use AppConfig/MissionState only
from src.satellite_control.config.mission_state import MissionState
from src.satellite_control.config.models import AppConfig
from src.satellite_control.visualization.unified_visualizer import PlotStyle

# Worker process cache for parallel frame rendering
_worker_gen_cache: Optional[Any] = None


class VideoRenderer:
    """
    Generates MP4 animations from simulation data.

    This class handles all video/animation generation, separating rendering
    logic from data management and plotting.
    """

    def __init__(
        self,
        data_accessor: Any,
        dt: float,
        fps: float,
        output_dir: Path,
        system_title: str = "Satellite Control System",
        speedup_factor: float = 1.0,
        satellite_size: float = 0.2,
        satellite_color: str = "blue",
        target_color: str = "red",
        trajectory_color: str = "green",
        thrusters: Optional[Dict[int, List[float]]] = None,
        dxf_base_shape: Optional[List] = None,
        dxf_offset_path: Optional[List] = None,
        dxf_center: Optional[List] = None,
        overlay_dxf: bool = False,
        frame_title_template: str = "Frame {frame}",
        app_config: Optional[AppConfig] = None,
        mission_state: Optional[MissionState] = None,
    ):
        """
        Initialize video renderer.

        Args:
            data_accessor: Object with data access methods (_col, _row, _get_len)
            dt: Simulation timestep in seconds
            fps: Frame rate for video
            output_dir: Directory to save video
            system_title: Title for video
            speedup_factor: Animation speedup factor
            satellite_size: Size of satellite visualization
            satellite_color: Color of satellite
            target_color: Color of target
            trajectory_color: Color of trajectory
            thrusters: Dictionary of thruster positions
            dxf_base_shape: DXF base shape for overlay
            dxf_offset_path: DXF offset path for overlay
            dxf_center: DXF center point
            overlay_dxf: Whether to overlay DXF shape
            frame_title_template: Template for frame titles
            app_config: Optional AppConfig for accessing configuration (v4.0.0: no fallback)
            mission_state: Optional MissionState for accessing mission-specific data (v4.0.0: no fallback)
        """
        self.data_accessor = data_accessor
        self.dt = dt
        self.fps = fps
        self.output_dir = output_dir
        self.system_title = system_title
        self.speedup_factor = speedup_factor
        self.satellite_size = satellite_size
        self.satellite_color = satellite_color
        self.target_color = target_color
        self.trajectory_color = trajectory_color
        self.thrusters = thrusters or {}
        self.dxf_base_shape = dxf_base_shape
        self.dxf_offset_path = dxf_offset_path
        self.dxf_center = dxf_center
        self.overlay_dxf = overlay_dxf
        self.frame_title_template = frame_title_template

        # Store config references (v3.0.0)
        self.app_config = app_config
        self.mission_state = mission_state

        self.fig: Optional[Figure] = None
        self.ax_main: Optional[Axes] = None
        self.ax_info: Optional[Axes] = None

    def _col(self, name: str) -> np.ndarray:
        """Get column data from data accessor."""
        return self.data_accessor._col(name)

    def _row(self, idx: int) -> Dict[str, Any]:
        """Get row data from data accessor."""
        return self.data_accessor._row(idx)

    def _get_len(self) -> int:
        """Get data length from data accessor."""
        return self.data_accessor._get_len()

    def parse_command_vector(self, command_str: Any) -> np.ndarray:
        """Parse command vector string to numpy array."""
        try:
            if command_str is None or command_str == "":
                return np.zeros(12)
            command_str = str(command_str)
            command_str = command_str.strip("[]")
            values = [float(x.strip()) for x in command_str.split(",")]
            return np.array(values)
        except Exception:
            return np.zeros(12)

    def get_active_thrusters(self, command_vector: np.ndarray) -> list:
        """Get list of active thruster IDs from command vector."""
        active = []
        for i, val in enumerate(command_vector):
            if abs(val) > 1e-6:  # Threshold for "active"
                active.append(i + 1)
        return active

    def setup_plot(self) -> None:
        """Set up the 3D plot for animation."""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_main = self.fig.add_subplot(111, projection="3d")
        self.ax_info = self.fig.add_axes([0.02, 0.02, 0.25, 0.35])

    def draw_satellite(
        self, x: float, y: float, z: float, yaw: float, active_thrusters: list
    ) -> None:
        """Draw satellite at given position and orientation (3D)."""
        assert self.ax_main is not None, "ax_main must be initialized"

        # Rotation matrix (2D Yaw only for now, visualized in 3D)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

        s = self.satellite_size / 2
        # Define 3D cube corners relative to center
        corners = np.array(
            [
                [-s, -s, -s],
                [s, -s, -s],
                [s, s, -s],
                [-s, s, -s],  # Bottom
                [-s, -s, s],
                [s, -s, s],
                [s, s, s],
                [-s, s, s],  # Top
            ]
        )

        # Rotate and translate
        rotated_corners = corners @ rotation_matrix.T + np.array([x, y, z])

        # Define faces for Poly3DCollection
        faces = [
            [
                rotated_corners[0],
                rotated_corners[1],
                rotated_corners[2],
                rotated_corners[3],
            ],  # Bottom
            [rotated_corners[4], rotated_corners[5], rotated_corners[6], rotated_corners[7]],  # Top
            [
                rotated_corners[0],
                rotated_corners[1],
                rotated_corners[5],
                rotated_corners[4],
            ],  # Front
            [
                rotated_corners[2],
                rotated_corners[3],
                rotated_corners[7],
                rotated_corners[6],
            ],  # Back
            [
                rotated_corners[1],
                rotated_corners[2],
                rotated_corners[6],
                rotated_corners[5],
            ],  # Right
            [
                rotated_corners[0],
                rotated_corners[3],
                rotated_corners[7],
                rotated_corners[4],
            ],  # Left
        ]

        # Draw Satellite Body
        poly = Poly3DCollection(faces, alpha=0.5, edgecolor="black", linewidths=0.5)
        poly.set_facecolor(self.satellite_color)
        self.ax_main.add_collection3d(poly)

        # Draw thrusters
        for thruster_id, pos in self.thrusters.items():
            tx, ty, tz = pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0

            # Rotate thruster position
            thruster_pos = np.array([tx, ty, tz]) @ rotation_matrix.T
            thruster_x = x + thruster_pos[0]
            thruster_y = y + thruster_pos[1]
            thruster_z = z + thruster_pos[2]

            # Color and size based on activity
            if thruster_id in active_thrusters:
                color = "red"
                size = 80
                marker = "o"
                alpha = 1.0
            else:
                color = "gray"
                size = 20
                marker = "o"
                alpha = 0.3

            self.ax_main.scatter(
                thruster_x,
                thruster_y,
                thruster_z,
                c=color,
                s=size,
                marker=marker,
                alpha=alpha,
                edgecolors="black",
                linewidth=0.5,
                depthshade=True,
            )

        # Draw orientation arrow (Forward X-axis)
        arrow_length = self.satellite_size * 1.5
        arrow_end_x = x + arrow_length * cos_yaw
        arrow_end_y = y + arrow_length * sin_yaw
        arrow_end_z = z

        self.ax_main.plot(
            [x, arrow_end_x], [y, arrow_end_y], [z, arrow_end_z], color="green", linewidth=2
        )

    def draw_target(
        self, target_x: float, target_y: float, target_z: float, target_yaw: float
    ) -> None:
        """Draw target position and orientation (3D)."""
        assert self.ax_main is not None, "ax_main must be initialized"

        self.ax_main.scatter(
            target_x,
            target_y,
            target_z,
            c=self.target_color,
            s=200,
            marker="x",
            linewidth=4,
            label="Target",
        )

        # Target Sphere (wireframe visual)
        theta = np.linspace(0, 2 * np.pi, 20)
        cx = target_x + 0.1 * np.cos(theta)
        cy = target_y + 0.1 * np.sin(theta)
        cz = np.full_like(cx, target_z)
        self.ax_main.plot(cx, cy, cz, color=self.target_color, alpha=0.5, linestyle="--")

        # Target orientation arrow
        arrow_length = self.satellite_size * 0.6
        arrow_end_x = target_x + arrow_length * np.cos(target_yaw)
        arrow_end_y = target_y + arrow_length * np.sin(target_yaw)
        arrow_end_z = target_z
        self.ax_main.plot(
            [target_x, arrow_end_x],
            [target_y, arrow_end_y],
            [target_z, arrow_end_z],
            color=self.target_color,
            alpha=0.8,
            linewidth=2,
        )

    def draw_trajectory(self, trajectory_x: list, trajectory_y: list, trajectory_z: list) -> None:
        """Draw satellite trajectory (3D)."""
        assert self.ax_main is not None, "ax_main must be initialized"

        # Ensure lengths match
        min_len = min(len(trajectory_x), len(trajectory_y), len(trajectory_z))
        if min_len > 1:
            self.ax_main.plot(
                trajectory_x[:min_len],
                trajectory_y[:min_len],
                trajectory_z[:min_len],
                color=self.trajectory_color,
                linewidth=2,
                alpha=0.8,
                linestyle="-",
                label="Trajectory",
            )

    def draw_obstacles(self, mission_state: Optional[MissionState] = None) -> None:
        """Draw obstacles if they are configured (V4.0.0: mission_state required, no fallback).
        
        Args:
            mission_state: Optional MissionState to get obstacles from (v4.0.0).
                          If None, uses self.mission_state if available. No fallback.
        """
        assert self.ax_main is not None, "ax_main must be initialized"

        # V4.0.0: Get obstacles from mission_state (required, no fallback)
        obstacles_enabled = False
        obstacles = []
        state_to_use = mission_state or self.mission_state
        if state_to_use:
            obstacles_enabled = state_to_use.obstacles_enabled
            obstacles = list(state_to_use.obstacles) if state_to_use.obstacles else []
        # V4.0.0: No fallback - if no mission_state, obstacles are empty
        
        if obstacles_enabled and obstacles:
            for i, (obs_x, obs_y, obs_radius) in enumerate(obstacles, 1):
                # Draw obstacle as red filled circle
                obstacle_circle = Circle(
                    (obs_x, obs_y),
                    obs_radius,
                    fill=True,
                    color="red",
                    alpha=0.6,
                    edgecolor="darkred",
                    linewidth=2,
                    zorder=15,
                    label="Obstacle" if i == 1 else "",
                )
                self.ax_main.add_patch(obstacle_circle)

                # Add obstacle label
                self.ax_main.text(
                    obs_x,
                    obs_y,
                    f"O{i}",
                    fontsize=8,
                    color="white",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    zorder=16,
                )

    def update_info_panel(self, step: int, current_data: Any) -> None:
        """Update information panel with current data using professional styling."""
        assert self.ax_info is not None, "ax_info must be initialized"

        self.ax_info.clear()
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis("off")

        # Professional Title
        self.ax_info.text(
            0.05,
            0.95,
            "System Telemetry",
            fontsize=14,
            weight="bold",
            color=PlotStyle.COLOR_PRIMARY,
            fontfamily=getattr(PlotStyle, "FONT_FAMILY", "serif"),
        )

        time = step * float(self.dt)

        # Determine Mission Phase and Metrics from Control Data
        mission_phase = ""
        mpc_solve_time = 0.0
        accumulated_usage_s = 0.0
        if hasattr(self.data_accessor, "control_data") and self.data_accessor.control_data is not None:
            import pandas as pd

            # Find closest control timestamp
            if (
                "Control_Time" in self.data_accessor.control_data.columns
                and "Mission_Phase" in self.data_accessor.control_data.columns
            ):
                # Ensure sorted
                idx = self.data_accessor.control_data["Control_Time"].searchsorted(time, side="right") - 1
                idx = max(0, min(idx, len(self.data_accessor.control_data) - 1))
                row = self.data_accessor.control_data.iloc[idx]

                # Phase
                phase_val = row["Mission_Phase"]
                if phase_val is not None and str(phase_val).lower() != "nan":
                    mission_phase = str(phase_val)

                # Solver Time
                if "MPC_Solve_Time" in row and pd.notna(row["MPC_Solve_Time"]):
                    mpc_solve_time = float(row["MPC_Solve_Time"]) * 1000.0
                elif "MPC_Computation_Time" in row:
                    mpc_solve_time = float(row["MPC_Computation_Time"]) * 1000.0  # to ms

                # Accumulated Thruster Usage
                if "Accumulated_Usage_S" in row:
                    accumulated_usage_s = float(row["Accumulated_Usage_S"])

        # Display Logic: Refined Content
        header_lines = [f"Time: {time:.3f} s"]
        if mission_phase:
            header_lines.append(f"Phase: {mission_phase}")

        state_lines = [
            "STATE",
            f"  X:   {current_data['Current_X']:>7.3f} m",
            f"  Y:   {current_data['Current_Y']:>7.3f} m",
            f"  Yaw: {np.degrees(current_data['Current_Yaw']):>6.1f}°",
        ]

        lin_speed = current_data.get("Linear_Speed", 0.0)
        ang_speed = abs(np.degrees(current_data.get("Current_Angular_Vel", 0.0)))
        dynamics_lines = [
            "DYNAMICS",
            f"  Speed: {lin_speed:.3f} m/s",
            f"  Ang V: {ang_speed:.1f} °/s",
        ]

        err_x = abs(current_data["Error_X"])
        err_y = abs(current_data["Error_Y"])
        err_yaw = abs(np.degrees(current_data["Error_Yaw"]))

        error_lines = [
            "ERRORS",
            f"  Err X:   {err_x:.3f} m",
            f"  Err Y:   {err_y:.3f} m",
            f"  Err Yaw: {err_yaw:.1f}°",
        ]

        system_lines = [
            "SYSTEM",
            f"  Total Thruster Usage: {accumulated_usage_s:.1f} s",
            f"  Solver: {mpc_solve_time:.1f} ms",
        ]

        groups = [
            (header_lines, PlotStyle.COLOR_PRIMARY, None),
            (state_lines, PlotStyle.COLOR_SIGNAL_POS, 0),
            (dynamics_lines, PlotStyle.COLOR_SIGNAL_POS, 0),
            (error_lines, PlotStyle.COLOR_TARGET, 0),
            (system_lines, PlotStyle.COLOR_PRIMARY, 0),
        ]
        y_pos = 0.85
        line_height = 0.035
        group_spacing = 0.02

        font_family = getattr(PlotStyle, "FONT_FAMILY", "serif")

        for lines, color, bold_idx in groups:
            for i, line in enumerate(lines):
                weight = "bold" if (bold_idx is not None and i == bold_idx) else "normal"
                size = 11 if weight == "bold" else 10

                self.ax_info.text(
                    0.05,
                    y_pos,
                    line,
                    fontsize=size,
                    weight=weight,
                    color=color,
                    verticalalignment="top",
                    fontfamily=font_family,
                )
                y_pos -= line_height
            y_pos -= group_spacing

    def animate_frame(self, frame: int) -> List[Any]:
        """Animation update function for each frame (3D)."""
        assert self.ax_main is not None, "ax_main must be initialized"

        # Clear main plot
        self.ax_main.clear()
        self.ax_main.set_xlim(-3, 3)
        self.ax_main.set_ylim(-3, 3)
        self.ax_main.set_zlim(-3, 3)
        self.ax_main.set_xlabel("X (m)")
        self.ax_main.set_ylabel("Y (m)")
        self.ax_main.set_zlabel("Z (m)")
        self.ax_main.set_title(self.frame_title_template.format(frame=frame))

        # Consistent viewing angle
        self.ax_main.view_init(elev=30.0, azim=45)

        # Get current data
        step = min(int(frame * self.speedup_factor), self._get_len() - 1)
        current_data = self._row(step)

        # Parse command vector and get active thrusters
        command_vector = self.parse_command_vector(current_data["Command_Vector"])
        active_thrusters = self.get_active_thrusters(command_vector)

        # Type-safe extraction
        target_x = float(current_data.get("Target_X", 0.0) or 0.0)
        target_y = float(current_data.get("Target_Y", 0.0) or 0.0)
        target_z = float(current_data.get("Target_Z", 0.0) or 0.0)
        target_yaw = float(current_data.get("Target_Yaw", 0.0) or 0.0)

        self.draw_target(target_x, target_y, target_z, target_yaw)

        # Draw trajectory
        traj_x = self._col("Current_X")[: step + 1].tolist()
        traj_y = self._col("Current_Y")[: step + 1].tolist()
        if "Current_Z" in current_data:
            traj_z = self._col("Current_Z")[: step + 1].tolist()
        else:
            traj_z = [0.0] * len(traj_x)
        self.draw_trajectory(traj_x, traj_y, traj_z)

        # Draw satellite
        curr_x = float(current_data.get("Current_X", 0.0) or 0.0)
        curr_y = float(current_data.get("Current_Y", 0.0) or 0.0)
        curr_z = float(current_data.get("Current_Z", 0.0) or 0.0)
        curr_yaw = float(current_data.get("Current_Yaw", 0.0) or 0.0)

        self.draw_satellite(
            curr_x,
            curr_y,
            curr_z,
            curr_yaw,
            active_thrusters,
        )

        # Draw obstacles
        self.draw_obstacles()

        # Add legend
        self.ax_main.legend(loc="upper right", fontsize=9)

        # Update info panel
        self.update_info_panel(step, current_data)

        return []

    def generate_animation(self, output_filename: Optional[str] = None) -> None:
        """Generate and save the MP4 animation using parallel rendering + imageio."""
        if output_filename is None:
            output_filename = f"animation.mp4"

        print(f"\n{'=' * 60}")
        print(f"GENERATING {self.system_title.upper()} ANIMATION (PARALLEL)")
        print(f"{'=' * 60}")

        # Calculate number of frames
        total_frames = self._get_len() // int(self.speedup_factor)
        if total_frames == 0:
            total_frames = 1

        print("Animation parameters:")
        print(f"  - Total data points: {self._get_len()}")
        print(f"  - Animation frames: {total_frames}")
        print(f"  - Frame rate: {self.fps} FPS")
        print(f"  - Speedup factor: {self.speedup_factor}x")
        print(f"  - Animation duration: {total_frames / self.fps:.1f} seconds")

        # Prepare configuration for workers
        if (
            hasattr(self.data_accessor, "_data_backend")
            and self.data_accessor._data_backend == "pandas"
        ):
            if hasattr(self.data_accessor, "data") and self.data_accessor.data is not None:
                data_dict = {col: self.data_accessor.data[col].values for col in self.data_accessor.data.columns}
        elif hasattr(self.data_accessor, "_col_data") and self.data_accessor._col_data is not None:
            data_dict = self.data_accessor._col_data
        else:
            raise ValueError("No data available for animation")

        # Config dictionary to reconstruct state in workers
        config = {
            "dt": self.dt,
            "fps": self.fps,
            "speedup_factor": self.speedup_factor,
            "system_title": self.system_title,
            "frame_title_template": self.frame_title_template,
            "satellite_size": self.satellite_size,
            "satellite_color": self.satellite_color,
            "target_color": self.target_color,
            "trajectory_color": self.trajectory_color,
            "thrusters": self.thrusters,
            "dxf_base_shape": self.dxf_base_shape,
            "dxf_offset_path": self.dxf_offset_path,
            "dxf_center": self.dxf_center,
            "dxf_mode_active": (
                self.mission_state.dxf_shape_mode_active
                if self.mission_state and hasattr(self.mission_state, "dxf_shape_mode_active")
                else False  # V4.0.0: No fallback
            ),
            "overlay_dxf": self.overlay_dxf,
            "obstacles_enabled": (
                self.mission_state.obstacles_enabled
                if self.mission_state
                else False  # V4.0.0: No fallback
            ),
            "obstacles_list": (
                list(self.mission_state.obstacles) if self.mission_state and self.mission_state.obstacles
                else []  # V4.0.0: No fallback
            ),
            "data_directory": str(getattr(self.data_accessor, "data_directory", Path("."))),
        }

        # Pass sibling control data if available
        if hasattr(self.data_accessor, "control_data") and self.data_accessor.control_data is not None:
            config["control_data_dict"] = self.data_accessor.control_data.to_dict(orient="list")
        else:
            config["control_data_dict"] = None

        # Use absolute path to ensure file is saved correctly
        output_path = Path(self.output_dir).resolve() / output_filename

        # Use temp directory in output folder
        frames_dir = Path(self.output_dir).resolve() / f"_frames_{uuid.uuid4().hex[:8]}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(
                f"\nRendering {total_frames} frames using "
                f"{multiprocessing.cpu_count()} cores..."
            )

            frames = list(range(total_frames))

            # Parallel frame rendering
            temp_dir = str(frames_dir)
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor() as executor:
                results_iterator = executor.map(
                    VideoRenderer._render_frame_task,
                    [(f, temp_dir, data_dict, config) for f in frames],
                )

                try:
                    from tqdm import tqdm

                    futures = list(
                        tqdm(
                            results_iterator,
                            total=total_frames,
                            unit="frames",
                            desc="Rendering Frames",
                            ncols=80,
                        )
                    )
                except ImportError:
                    futures = []
                    print("Rendering frames (tqdm not installed)...")
                    for i, res in enumerate(results_iterator):
                        futures.append(res)
                        if i % 10 == 0 or i == total_frames - 1:
                            sys.stdout.write(f"\rProcessed {i+1}/{total_frames} frames")
                            sys.stdout.flush()
                    print()

            # Assemble video with imageio
            print("Assembling video with imageio...")

            fps_for_video = max(1.0, self.fps)

            # Ensure imageio finds the system ffmpeg
            import shutil

            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

            try:
                with imageio.get_writer(
                    str(output_path),
                    fps=fps_for_video,
                    macro_block_size=1,
                    quality=8,
                ) as writer:
                    for frame_idx in range(total_frames):
                        frame_path = os.path.join(temp_dir, f"frame_{frame_idx:05d}.png")
                        if os.path.exists(frame_path):
                            frame = imageio.imread(frame_path)
                            writer.append_data(frame)  # type: ignore[attr-defined]

                print("\n Animation saved successfully!")
                print(f" File location: {output_path}")
            except Exception as e:
                print(f"Error assembling video: {e}")
                raise
        finally:
            # Clean up frames directory
            import shutil

            if frames_dir.exists():
                shutil.rmtree(frames_dir)

    @staticmethod
    def _render_frame_task(args):
        """Worker function to render a single frame (runs in separate process)."""
        frame_idx, output_dir, data_dict, config = args

        try:
            global _worker_gen_cache
            if _worker_gen_cache is None:
                # Create a simple data accessor for worker process
                class WorkerDataAccessor:
                    def __init__(self, data_dict):
                        self._col_data = data_dict
                        self._data_backend = "csv"
                        self.control_data = None

                    def _col(self, name: str) -> np.ndarray:
                        return self._col_data.get(name, np.array([]))

                    def _row(self, idx: int) -> Dict[str, Any]:
                        return {
                            k: (self._col_data[k][idx] if k in self._col_data else None)
                            for k in self._col_data.keys()
                        }

                    def _get_len(self) -> int:
                        if self._col_data:
                            first_key = next(iter(self._col_data))
                            return len(self._col_data[first_key])
                        return 0

                # Create data accessor
                data_accessor = WorkerDataAccessor(data_dict)

                # Restore control data if provided
                if config.get("control_data_dict"):
                    try:
                        import pandas as pd

                        data_accessor.control_data = pd.DataFrame(config["control_data_dict"])
                    except ImportError:
                        pass

                # Create renderer
                renderer = VideoRenderer(
                    data_accessor=data_accessor,
                    dt=config["dt"],
                    fps=config["fps"],
                    output_dir=Path(config["data_directory"]),
                    system_title=config["system_title"],
                    speedup_factor=config["speedup_factor"],
                    satellite_size=config["satellite_size"],
                    satellite_color=config["satellite_color"],
                    target_color=config["target_color"],
                    trajectory_color=config["trajectory_color"],
                    thrusters=config["thrusters"],
                    dxf_base_shape=config["dxf_base_shape"],
                    dxf_offset_path=config["dxf_offset_path"],
                    dxf_center=config["dxf_center"],
                    overlay_dxf=config["overlay_dxf"],
                    frame_title_template=config["frame_title_template"],
                )

                # V4.0.0: Config state is passed via config dict only (no global state mutation)
                # Worker process uses config dict values directly

                renderer.setup_plot()
                _worker_gen_cache = renderer
            else:
                renderer = _worker_gen_cache

            if renderer is not None:
                renderer.animate_frame(frame_idx)
                filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
                if renderer.fig is not None:
                    renderer.fig.savefig(filename, dpi=100)
        except Exception:
            import traceback

            traceback.print_exc()
            return False

        return True
