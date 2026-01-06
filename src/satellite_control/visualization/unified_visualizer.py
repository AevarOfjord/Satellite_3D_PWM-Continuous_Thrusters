#!/usr/bin/env python3
"""
Unified Visualization Module for MPC Satellite Control

Unified visualization system for simulation systems.
Automatically generates MP4 animations and performance plots from CSV data.

Features:
- Unified interface for both simulation and real test data
- Automatic data detection and loading
- High-quality MP4 animation generation
- Comprehensive performance analysis plots
- Configurable titles and labels based on data source
- Real-time animation with proper timing
"""

import csv
import os
import platform
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from src.satellite_control.config import SatelliteConfig

matplotlib.use("Agg")  # Use non-interactive backend


# Configure ffmpeg path with Config-first logic, then safe fallbacks
try:
    if getattr(SatelliteConfig, "FFMPEG_PATH", None):
        plt.rcParams["animation.ffmpeg_path"] = SatelliteConfig.FFMPEG_PATH
    else:
        # If ffmpeg already in PATH, do nothing
        if shutil.which("ffmpeg") is None:
            if platform.system() == "Darwin":
                brew_ffmpeg = "/opt/homebrew/bin/ffmpeg"
                if os.path.exists(brew_ffmpeg):
                    plt.rcParams["animation.ffmpeg_path"] = brew_ffmpeg
except Exception:
    pass


warnings.filterwarnings("ignore")

# Worker process cache for parallel frame rendering
_worker_gen_cache: Optional[Any] = None

# Import cycler and Circle from correct modules

# Use Seaborn for prettier plot styling (falls back to matplotlib if not installed)
try:
    import seaborn as sns

    # Professional/Academic style: White background, ticks, serif font
    sns.set_theme(style="ticks", font="serif", font_scale=1.1)
    sns.set_context("paper", rc={"lines.linewidth": 1.5})
    _SEABORN_AVAILABLE = True
except ImportError:
    _SEABORN_AVAILABLE = False
    # Fallback to matplotlib color scheme
    plt.rcParams["axes.prop_cycle"] = cycler(
        color=[
            "#000000",  # Black
            "#CC0000",  # Dark Red
            "#1f77b4",  # Blue
            "#2ca02c",  # Green
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
        ]
    )

# Consistent plot styling (applied on top of seaborn if available)
plt.rcParams.update(
    {
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "font.family": "serif",  # Serif for academic look
        "lines.linewidth": 2,
        "grid.alpha": 0.2,
    }
)


class PlotStyle:
    """Centralized plot styling constants for consistent visualizations."""

    # Figure sizes
    FIGSIZE_SINGLE = (10, 6)  # Slightly smaller for papers
    FIGSIZE_SUBPLOTS = (10, 8)
    FIGSIZE_WIDE = (12, 5)
    FIGSIZE_TALL = (8, 8)

    # Resolution (High for print)
    DPI = 300

    # Line properties
    LINEWIDTH = 1.5
    LINEWIDTH_THICK = 2.5
    MARKER_SIZE = 8

    # Grid
    GRID_ALPHA = 0.15

    # Font sizes
    AXIS_LABEL_SIZE = 14
    LEGEND_SIZE = 12
    TITLE_SIZE = 16
    ANNOTATION_SIZE = 12

    # Text box style (Minimalist)
    TEXTBOX_STYLE = dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="black",  # High contrast border
        linewidth=0.5,
        alpha=1.0,  # Opaque
    )

    # Colors (4-Color Scheme: Black, Blue, Red, Green)
    COLOR_PRIMARY = "#000000"  # Black (Axes, Text)

    # Semantic Roles
    COLOR_SIGNAL_POS = "#1f77b4"  # Blue (Position Signals)
    COLOR_SIGNAL_ANG = "#2ca02c"  # Green (Angular Signals)

    COLOR_TARGET = "#d62728"  # Red (Targets/Reference)
    COLOR_THRESHOLD = "#d62728"  # Red (Limits/Tolerances)

    COLOR_BARS = "#1f77b4"  # Blue (Thruster Bars)
    COLOR_SUCCESS = "#2ca02c"  # Green
    COLOR_ERROR = "#d62728"  # Red

    # Legacy aliases for compatibility if needed
    COLOR_SIGNAL = "#000000"  # Default black
    COLOR_SECONDARY = "#d62728"  # Red

    @staticmethod
    def apply_axis_style(ax, xlabel: str = "", ylabel: str = "", title: str = ""):
        """Apply consistent styling to an axis."""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=PlotStyle.AXIS_LABEL_SIZE)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=PlotStyle.AXIS_LABEL_SIZE)
        if title:
            ax.set_title(title, fontsize=PlotStyle.TITLE_SIZE)
        ax.grid(True, alpha=PlotStyle.GRID_ALPHA)
        ax.legend(fontsize=PlotStyle.LEGEND_SIZE)

    @staticmethod
    def save_figure(fig, path, close: bool = True):
        """Save figure with consistent settings."""
        plt.tight_layout()
        fig.savefig(path, dpi=PlotStyle.DPI, bbox_inches="tight")
        if close:
            plt.close(fig)


class UnifiedVisualizationGenerator:
    """Unified visualization generator for simulation data."""

    def __init__(
        self,
        data_directory: str,
        interactive: bool = False,
        load_data: bool = True,
        prefer_pandas: bool = True,
        overlay_dxf: bool = False,
    ):
        """Initialize the visualization generator.

        Args:
            data_directory: Base directory containing the data
            interactive: If True, allow user to select which data to visualize
            load_data: If True, automatically locate and load the newest CSV
            prefer_pandas: If True, try pandas; otherwise use csv module backend
            overlay_dxf: If True, overlay DXF shape on trajectory (if available)
        """
        self.data_directory = Path(data_directory)
        self.mode = "simulation"
        self.csv_path: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.data: Optional[Any] = None  # DataFrame when pandas, self when csv backend
        self.fig: Optional[Figure] = None
        self.ax_main: Optional[Axes] = None
        self.ax_info: Optional[Axes] = None
        # Data backend management
        self._data_backend: Optional[str] = None  # 'pandas' or 'csv'
        self._rows: Optional[List[Dict[str, Any]]] = None  # list of dicts when csv backend
        self._col_data: Optional[Dict[str, np.ndarray]] = (
            None  # dict of column -> array when csv backend
        )
        self.use_pandas = prefer_pandas
        self.overlay_dxf = overlay_dxf  # Option to overlay DXF shape

        # Set titles and labels
        self._setup_labels()

        self.fps: Optional[float] = None  # Will be calculated from simulation timestep
        self.speedup_factor = 1.0  # Real-time playback (no speedup)
        self.real_time = True  # Enable real-time animation
        self.dt: Optional[float] = None  # Simulation timestep (will be auto-detected)

        self.satellite_size = SatelliteConfig.SATELLITE_SIZE
        self.satellite_color = "blue"
        self.target_color = "red"
        self.trajectory_color = "cyan"

        self.dxf_base_shape: List[Any] = getattr(SatelliteConfig, "DXF_BASE_SHAPE", [])
        self.dxf_offset_path = getattr(SatelliteConfig, "DXF_SHAPE_PATH", [])
        self.dxf_center = getattr(SatelliteConfig, "DXF_SHAPE_CENTER", None)

        self.thrusters = {}
        for thruster_id, pos in SatelliteConfig.THRUSTER_POSITIONS.items():
            self.thrusters[thruster_id] = pos

        self.thruster_forces = SatelliteConfig.THRUSTER_FORCES.copy()

        if load_data:
            if interactive:
                self.select_data_interactively()
            else:
                self.find_newest_data()

    def _setup_labels(self) -> None:
        """Setup titles and labels for simulation."""
        self.system_name = "Simulation"
        self.system_title = "MPC Simulation"
        self.data_source = "Simulation Data"
        self.plot_prefix = "Simulation"
        self.animation_title = "MPC Satellite Simulation Visualization"
        self.trajectory_title = "MPC - Satellite Trajectory"
        self.frame_title_template = "MPC - Frame {}"

    def find_newest_data(self) -> None:
        """Find the newest data folder and CSV file."""
        print(f"Searching for {self.system_name} data in: {self.data_directory}")

        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")

        # First, check if the current directory itself contains CSV files
        csv_patterns = [
            "physics_data.csv",
            "control_data.csv",
            "simulation_data.csv",
            "*.csv",
        ]
        csv_files = []

        for pattern in csv_patterns:
            csv_files = list(self.data_directory.glob(pattern))
            if csv_files:
                # Current directory has CSV files - use it directly
                self.csv_path = csv_files[0]
                self.output_dir = self.data_directory
                print(f"Using CSV file: {self.csv_path.name} in current directory")
                self.load_csv_data()
                return

        # If no CSV in current directory, search subdirectories
        data_folders = [d for d in self.data_directory.iterdir() if d.is_dir()]

        if not data_folders:
            raise FileNotFoundError(f"No data folders found in: {self.data_directory}")

        data_folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        print(f"Found {len(data_folders)} data folders")

        # Try folders until we find one with CSV data
        newest_folder = None
        csv_files = []

        for folder in data_folders:
            print(f"Checking folder: {folder.name}")

            for pattern in csv_patterns:
                csv_files = list(folder.glob(pattern))
                if csv_files:
                    break

            if csv_files:
                newest_folder = folder
                print(f"Using folder: {newest_folder.name}")
                break
            else:
                print("  No CSV files found, trying next folder...")

        if not csv_files or newest_folder is None:
            raise FileNotFoundError("No CSV files found in any data folder")

        self.csv_path = csv_files[0]
        self.output_dir = newest_folder

        print(f"Using CSV file: {self.csv_path.name}")
        print(f"Output directory: {self.output_dir}")

        # Load and validate data
        self.load_csv_data()

    def select_data_interactively(self) -> None:
        """Allow user to interactively select which data to visualize."""
        print(f"\n{'=' * 60}")
        print(f"INTERACTIVE {self.system_name.upper()} DATA SELECTION")
        print(f"{'=' * 60}")

        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")

        # Find all subdirectories with CSV files
        data_folders = []
        for folder in self.data_directory.iterdir():
            if folder.is_dir():
                csv_files = list(folder.glob("*.csv"))
                if csv_files:
                    data_folders.append((folder, csv_files))

        if not data_folders:
            raise FileNotFoundError(f"No folders with CSV data found in: {self.data_directory}")

        data_folders.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

        print(f"Found {len(data_folders)} folders with data:")
        print()

        for i, (folder, csv_files) in enumerate(data_folders, 1):
            # Get folder timestamp
            timestamp = datetime.fromtimestamp(folder.stat().st_mtime)
            print(f"{i:2}. {folder.name}")
            print(f"    Modified: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    CSV files: {len(csv_files)}")
            print()

        # Get user selection
        while True:
            try:
                choice = input(f"Select folder (1-{len(data_folders)}) or 'q' to quit: ").strip()
                if choice.lower() == "q":
                    print("Visualization cancelled.")
                    sys.exit(0)
                folder_idx = int(choice) - 1
                if 0 <= folder_idx < len(data_folders):
                    selected_folder, csv_files = data_folders[folder_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(data_folders)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")

        if len(csv_files) > 1:
            print(f"\nMultiple CSV files found in {selected_folder.name}:")
            for i, csv_file in enumerate(csv_files, 1):
                print(f"{i}. {csv_file.name}")

            while True:
                try:
                    csv_choice = input(f"Select CSV file (1-{len(csv_files)}): ").strip()
                    csv_idx = int(csv_choice) - 1
                    if 0 <= csv_idx < len(csv_files):
                        selected_csv = csv_files[csv_idx]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(csv_files)}")
                except ValueError:
                    print("Please enter a valid number")
        else:
            selected_csv = csv_files[0]

        self.csv_path = selected_csv
        self.output_dir = selected_folder

        print(f"\n Selected: {selected_folder.name}/{selected_csv.name}")

        # Load and validate data
        self.load_csv_data()

    def load_csv_data(self) -> None:
        """Load and validate CSV data."""
        assert self.csv_path is not None, "CSV path must be set before loading data"

        try:
            if self.use_pandas:
                import pandas as pd

                print(f"Loading CSV data from: {self.csv_path}")
                self.data = pd.read_csv(self.csv_path)
                self._data_backend = "pandas"
                print(f"Loaded {len(self.data)} data points")

                # If loading physics_data.csv, try to load control_data.csv for
                # MPC metrics
                if self.csv_path.name == "physics_data.csv":
                    control_path = self.csv_path.with_name("control_data.csv")
                    if control_path.exists():
                        try:
                            self.control_data = pd.read_csv(control_path)
                            print(f"Loaded sibling control data: {len(self.control_data)} points")
                        except Exception as e:
                            print(f"Failed to load sibling control data: {e}")
                            self.control_data = None
                    else:
                        print("Sibling control_data.csv not found.")

                # Pre-calculate derived metrics for animation
                self._detect_timestep()  # Ensure dt is set
                if self.dt is not None:
                    # Linear Speed
                    if "Current_VX" in self.data.columns and "Current_VY" in self.data.columns:
                        self.data["Linear_Speed"] = np.sqrt(
                            self.data["Current_VX"] ** 2 + self.data["Current_VY"] ** 2
                        )
                    else:
                        self.data["Linear_Speed"] = 0.0

                    # Accumulated Thruster Usage (s) - Sum of all valve states
                    # Use physics_data Thruster_X_Val columns for accurate PWM
                    # tracking. This matches the thruster_usage.png calculation.
                    has_valve_data = "Thruster_1_Val" in self.data.columns
                    if has_valve_data:
                        # Sum valve states for all 8 thrusters at each timestep
                        valve_sum_per_step = np.zeros(len(self.data))
                        for i in range(8):
                            col_name = f"Thruster_{i+1}_Val"
                            if col_name in self.data.columns:
                                vals = self.data[col_name].values
                                try:
                                    vals = np.array([float(x) for x in vals])
                                except (ValueError, TypeError):
                                    vals = np.zeros(len(vals), dtype=float)
                                valve_sum_per_step += vals

                        # Cumulative sum of valve usage * physics dt
                        physics_dt = float(self.dt)
                        cumulative_usage = np.cumsum(valve_sum_per_step) * physics_dt
                        self.data["Accumulated_Usage_S"] = cumulative_usage

                        # Also store on control_data by time-matching
                        if (
                            self.control_data is not None
                            and "Control_Time" in self.control_data.columns
                        ):
                            # For each control row, find closest physics row
                            ctrl_times = self.control_data["Control_Time"].values
                            phys_times = self.data["Time"].values
                            ctrl_usage = []
                            for ct in ctrl_times:
                                idx = np.searchsorted(phys_times, ct, side="right") - 1
                                idx = max(0, min(idx, len(cumulative_usage) - 1))
                                ctrl_usage.append(cumulative_usage[idx])
                            self.control_data["Accumulated_Usage_S"] = ctrl_usage

                        print(
                            f"Calculated Max Usage from Control Data: "
                            f"{cumulative_usage[-1]:.2f} s"
                        )
                    else:
                        print("Could not calculate usage: " "Thruster_X_Val columns missing")

            else:
                print(f"Loading CSV data (csv backend) from: {self.csv_path}")
                self._load_csv_data_csvmodule()
                self._data_backend = "csv"
                print(f"Loaded {self._get_len()} data points")

            # Validate required columns
            required_cols = [
                "Current_X",
                "Current_Y",
                "Current_Yaw",
                "Target_X",
                "Target_Y",
                "Target_Yaw",
                "Command_Vector",
            ]
            if self._data_backend == "pandas" and self.data is not None:
                cols = self.data.columns
            elif self._col_data is not None:
                cols = self._col_data.keys()
            else:
                cols = []
            missing_cols = [col for col in required_cols if col not in cols]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            self._detect_timestep()

            # Determine dt again if not set (redundant but safe)
            if self.dt is None:
                self._detect_timestep()

            print("CSV data validation successful")
            if self.dt is not None:
                print(f"Time range: 0.0s to {self._get_len() * float(self.dt):.1f}s")
                print(f"Detected timestep (dt): {self.dt:.3f}s")
            if self.fps is not None:
                print(f"Calculated frame rate: {self.fps:.1f} FPS")

        except Exception as e:
            print(f"Error loading CSV data: {e}")
            raise

    def _detect_timestep(self) -> None:
        """Auto-detect timestep from data and calculate frame rate."""
        if self._data_backend == "pandas" and self.data is not None:
            cols = self.data.columns
        elif self._col_data is not None:
            cols = self._col_data.keys()
        else:
            cols = []
        if "CONTROL_DT" in cols:
            if self._data_backend == "pandas" and self.data is not None:
                val = self.data["CONTROL_DT"].iloc[0]
            else:
                val = self._col("CONTROL_DT")[0]
            self.dt = float(val)
            print(f" Detected timestep from CONTROL_DT: {self.dt}s")

        elif "Time" in cols and self._get_len() > 1:
            try:
                # Calculate average time difference
                time_diffs = []
                for i in range(1, min(10, self._get_len())):
                    curr = float(self._row(i)["Time"])
                    prev = float(self._row(i - 1)["Time"])
                    time_diffs.append(curr - prev)

                if time_diffs:
                    self.dt = float(np.mean(time_diffs))
                    print(f" Detected timestep from Time column: {self.dt:.4f}s")
                else:
                    self.dt = 0.005  # Fallback
            except Exception:
                self.dt = 0.005

        elif "Actual_Time_Interval" in cols and self._get_len() > 1:
            # Use average of actual time intervals
            intervals = self._col("Actual_Time_Interval")
            valid_intervals = intervals[intervals > 0]  # Filter out zero values
            if len(valid_intervals) > 0:
                self.dt = float(np.mean(valid_intervals))
                print(f" Detected timestep from Actual_Time_Interval: {self.dt:.3f}s")
            else:
                self.dt = 0.25  # Fallback
                print(f"Using fallback timestep: {self.dt}s")

        elif "Step" in cols and self._get_len() > 1:
            if "Control_Time" in cols:
                control_time_col = self._col("Control_Time")
                total_time = float(control_time_col[-1]) if len(control_time_col) > 0 else 0.0
                total_steps = self._get_len() - 1
                if total_steps > 0:
                    self.dt = total_time / total_steps
                    print(f" Calculated timestep from time: {self.dt:.3f}s")
                else:
                    self.dt = 0.25
                    print(f"Using fallback timestep: {self.dt}s")
            else:
                self.dt = 0.25
                print(f"Using default timestep assumption: {self.dt}s")

        elif "MPC_Start_Time" in cols and self._get_len() > 1:
            try:
                # Calculate average time difference between consecutive rows
                time_diffs = []
                for i in range(1, min(10, self._get_len())):  # Sample first 10 rows
                    curr_time_raw = self._row(i)["MPC_Start_Time"]
                    prev_time_raw = self._row(i - 1)["MPC_Start_Time"]
                    if curr_time_raw is not None and prev_time_raw is not None:
                        curr_time = float(curr_time_raw)
                        prev_time = float(prev_time_raw)
                        time_diff = curr_time - prev_time
                        if 0.01 <= time_diff <= 10.0:  # Reasonable timestep range
                            time_diffs.append(time_diff)

                if time_diffs:
                    self.dt = float(np.mean(time_diffs))
                else:
                    self.dt = 0.25  # Default fallback
            except Exception:
                self.dt = 0.25  # Default fallback
        else:
            # Default timestep
            self.dt = 0.25

        # This ensures each frame represents one timestep
        assert self.dt is not None and self.dt > 0, "dt must be positive"

        ideal_fps = 1.0 / self.dt
        TARGET_PLAYBACK_FPS = 60.0

        if ideal_fps > TARGET_PLAYBACK_FPS:
            # Calculate integer step size (frame skipping)
            step_size = max(1, int(round(ideal_fps / TARGET_PLAYBACK_FPS)))

            self.speedup_factor = float(step_size)

            # adjust video FPS to match the effective sampling rate
            # effective_fps = ideal_fps / step_size
            self.fps = ideal_fps / step_size

            print(
                f"Optimization: Simulation runs at {ideal_fps:.1f}Hz. "
                f"Rendering every {step_size}th frame. "
                f"Video FPS set to {self.fps:.2f} for Real-Time playback."
            )
        else:
            self.speedup_factor = 1.0
            self.fps = ideal_fps

    @staticmethod
    def parse_command_vector(command_str: Any) -> np.ndarray:
        """Parse command vector string to numpy array.

        Args:
            command_str: String representation of command vector (or any type)

        Returns:
            numpy array of thruster commands
        """
        try:
            # Handle None or empty
            if command_str is None or command_str == "":
                return np.zeros(8)

            # Convert to string if not already
            command_str = str(command_str)

            # Remove brackets and split
            command_str = command_str.strip("[]")
            values = [float(x.strip()) for x in command_str.split(",")]
            return np.array(values)
        except Exception:
            return np.zeros(8)

    def get_active_thrusters(self, command_vector: np.ndarray) -> list:
        """Get list of active thruster IDs from command vector.

        Args:
            command_vector: Array of thruster commands

        Returns:
            List of active thruster IDs (1-8)
        """
        return [i + 1 for i, cmd in enumerate(command_vector) if cmd > 0.5]

    def setup_plot(self) -> None:
        """Initialize matplotlib figure and axes."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.suptitle(self.animation_title, fontsize=16)

        self.ax_main = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        self.ax_main.set_xlim(-3, 3)
        self.ax_main.set_ylim(-3, 3)
        self.ax_main.set_aspect("equal")
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlabel("X Position (m)")
        self.ax_main.set_ylabel("Y Position (m)")
        self.ax_main.set_title(self.trajectory_title)

        self.ax_info = plt.subplot2grid((1, 3), (0, 2))
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis("off")
        self.ax_info.set_title("System Info", fontsize=12)

        plt.tight_layout()

    def draw_satellite(self, x: float, y: float, yaw: float, active_thrusters: list) -> None:
        """Draw satellite at given position and orientation.

        Args:
            x, y: Satellite position
            yaw: Satellite orientation (radians)
            active_thrusters: List of active thruster IDs
        """
        assert self.ax_main is not None, "ax_main must be initialized"

        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        body_corners = np.array(
            [
                [-self.satellite_size / 2, -self.satellite_size / 2],
                [self.satellite_size / 2, -self.satellite_size / 2],
                [self.satellite_size / 2, self.satellite_size / 2],
                [-self.satellite_size / 2, self.satellite_size / 2],
                [-self.satellite_size / 2, -self.satellite_size / 2],
            ]
        )

        # Rotate and translate body
        rotated_body = body_corners @ rotation_matrix.T + np.array([x, y])
        self.ax_main.plot(
            rotated_body[:, 0],
            rotated_body[:, 1],
            color=self.satellite_color,
            linewidth=3,
            label="Satellite",
        )

        # Fill the satellite body
        self.ax_main.fill(
            rotated_body[:, 0],
            rotated_body[:, 1],
            color=self.satellite_color,
            alpha=0.3,
        )

        # Draw thrusters
        for thruster_id, (tx, ty) in self.thrusters.items():
            # Rotate thruster position
            thruster_pos = np.array([tx, ty]) @ rotation_matrix.T
            thruster_x = x + thruster_pos[0]
            thruster_y = y + thruster_pos[1]

            # Color and size based on activity
            if thruster_id in active_thrusters:
                color = "red"
                size = 80
                marker = "o"
                alpha = 1.0
            else:
                color = "gray"
                size = 40
                marker = "s"
                alpha = 0.5

            self.ax_main.scatter(
                thruster_x,
                thruster_y,
                c=color,
                s=size,
                marker=marker,
                alpha=alpha,
                edgecolors="black",
                linewidth=1,
            )

        # Draw orientation arrow
        arrow_length = self.satellite_size * 0.8
        arrow_end_x = x + arrow_length * cos_yaw
        arrow_end_y = y + arrow_length * sin_yaw
        self.ax_main.arrow(
            x,
            y,
            arrow_end_x - x,
            arrow_end_y - y,
            head_width=0.08,
            head_length=0.08,
            fc="green",
            ec="green",
            linewidth=2,
            alpha=0.8,
        )

    def draw_target(self, target_x: float, target_y: float, target_yaw: float) -> None:
        """Draw target position and orientation.

        Args:
            target_x, target_y: Target position
            target_yaw: Target orientation (radians)
        """
        assert self.ax_main is not None, "ax_main must be initialized"

        self.ax_main.scatter(
            target_x,
            target_y,
            c=self.target_color,
            s=200,
            marker="x",
            linewidth=4,
            label="Target",
        )

        # Target circle
        circle = Circle(
            (target_x, target_y),
            0.05,
            color=self.target_color,
            fill=False,
            linewidth=2,
            alpha=0.7,
        )
        self.ax_main.add_patch(circle)

        # Target orientation arrow
        arrow_length = self.satellite_size * 0.6
        arrow_end_x = target_x + arrow_length * np.cos(target_yaw)
        arrow_end_y = target_y + arrow_length * np.sin(target_yaw)
        self.ax_main.arrow(
            target_x,
            target_y,
            arrow_end_x - target_x,
            arrow_end_y - target_y,
            head_width=0.06,
            head_length=0.06,
            fc=self.target_color,
            ec=self.target_color,
            alpha=0.8,
            linewidth=2,
        )

    def draw_trajectory(self, trajectory_x: list, trajectory_y: list) -> None:
        """Draw satellite trajectory.

        Args:
            trajectory_x, trajectory_y: Lists of trajectory points
        """
        assert self.ax_main is not None, "ax_main must be initialized"

        if len(trajectory_x) > 1:
            self.ax_main.plot(
                trajectory_x,
                trajectory_y,
                color=self.trajectory_color,
                linewidth=2,
                alpha=0.8,
                linestyle="-",
                label="Trajectory",
            )

    def draw_dxf_shape_overlays(self) -> None:
        """Draw Profile Following mission overlays: the base object shape and the offset path."""
        assert self.ax_main is not None, "ax_main must be initialized"

        try:
            if isinstance(self.dxf_base_shape, (list, tuple)) and len(self.dxf_base_shape) >= 3:
                bx = [p[0] for p in self.dxf_base_shape] + [self.dxf_base_shape[0][0]]
                by = [p[1] for p in self.dxf_base_shape] + [self.dxf_base_shape[0][1]]
                self.ax_main.plot(
                    bx,
                    by,
                    color="#9b59b6",
                    linewidth=2.5,
                    alpha=0.7,
                    label="Object Shape",
                )
            if isinstance(self.dxf_offset_path, (list, tuple)) and len(self.dxf_offset_path) >= 2:
                px = [p[0] for p in self.dxf_offset_path]
                py = [p[1] for p in self.dxf_offset_path]
                self.ax_main.plot(
                    px,
                    py,
                    color="#e67e22",
                    linewidth=2,
                    alpha=0.8,
                    linestyle="--",
                    label="Shape Path",
                )
            if isinstance(self.dxf_center, (list, tuple)) and len(self.dxf_center) == 2:
                self.ax_main.scatter(
                    self.dxf_center[0],
                    self.dxf_center[1],
                    c="#2ecc71",
                    s=60,
                    marker="+",
                    linewidth=2,
                    label="Shape Center",
                )
        except Exception:
            pass

    def draw_obstacles(self) -> None:
        """Draw obstacles if they are configured."""
        assert self.ax_main is not None, "ax_main must be initialized"

        if hasattr(SatelliteConfig, "OBSTACLES_ENABLED") and SatelliteConfig.OBSTACLES_ENABLED:
            obstacles = SatelliteConfig.get_obstacles()
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
        """Update information panel with current data using professional styling.

        Args:
            step: Current step
            current_data: Current row of data
        """
        assert self.ax_info is not None, "ax_info must be initialized"
        assert self.dt is not None, "dt must be set"

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
        if hasattr(self, "control_data") and self.control_data is not None:
            import pandas as pd

            # Find closest control timestamp
            if (
                "Control_Time" in self.control_data.columns
                and "Mission_Phase" in self.control_data.columns
            ):
                # Ensure sorted
                idx = self.control_data["Control_Time"].searchsorted(time, side="right") - 1
                idx = max(0, min(idx, len(self.control_data) - 1))
                row = self.control_data.iloc[idx]

                # Phase
                phase_val = row["Mission_Phase"]
                if phase_val is not None and str(phase_val).lower() != "nan":
                    mission_phase = str(phase_val)

                # Solver Time
                if "MPC_Solve_Time" in row and pd.notna(row["MPC_Solve_Time"]):
                    mpc_solve_time = float(row["MPC_Solve_Time"]) * 1000.0
                elif "MPC_Computation_Time" in row:
                    mpc_solve_time = float(row["MPC_Computation_Time"]) * 1000.0  # to ms

                # Accumulated Thruster Usage (calculated on control_data)
                if "Accumulated_Usage_S" in row:
                    accumulated_usage_s = float(row["Accumulated_Usage_S"])

        # Display Logic: Refined Content

        # Group 1: Time & Phase (High Level)
        header_lines = [f"Time: {time:.3f} s"]
        if mission_phase:
            header_lines.append(f"Phase: {mission_phase}")

        # Group 2: STATE (Blue)
        state_lines = [
            "STATE",
            f"  X:   {current_data['Current_X']:>7.3f} m",
            f"  Y:   {current_data['Current_Y']:>7.3f} m",
            f"  Yaw: {np.degrees(current_data['Current_Yaw']):>6.1f}°",
        ]

        # Group 3: DYNAMICS (Blue/Black) - Speed info
        lin_speed = current_data.get("Linear_Speed", 0.0)
        ang_speed = abs(np.degrees(current_data.get("Current_Angular_Vel", 0.0)))
        dynamics_lines = [
            "DYNAMICS",
            f"  Speed: {lin_speed:.3f} m/s",
            f"  Ang V: {ang_speed:.1f} °/s",
        ]

        # Group 4: ERRORS (Red)
        err_x = abs(current_data["Error_X"])
        err_y = abs(current_data["Error_Y"])
        err_yaw = abs(np.degrees(current_data["Error_Yaw"]))

        error_lines = [
            "ERRORS",
            f"  Err X:   {err_x:.3f} m",
            f"  Err Y:   {err_y:.3f} m",
            f"  Err Yaw: {err_yaw:.1f}°",
        ]

        # Group 5: SYSTEM (Black) - Usage & Perf
        system_lines = [
            "SYSTEM",
            f"  Total Thruster Usage: {accumulated_usage_s:.1f} s",
            f"  Solver: {mpc_solve_time:.1f} ms",
        ]

        # Define data groups with semantic colors
        # Format: (List of strings, Color, bold_header_index)
        groups = [
            (header_lines, PlotStyle.COLOR_PRIMARY, None),
            (state_lines, PlotStyle.COLOR_SIGNAL_POS, 0),  # Blue
            (
                dynamics_lines,
                PlotStyle.COLOR_SIGNAL_POS,
                0,
            ),  # Blue (Grouped by association)
            (error_lines, PlotStyle.COLOR_TARGET, 0),  # Red
            (system_lines, PlotStyle.COLOR_PRIMARY, 0),  # Black
        ]
        y_pos = 0.85
        line_height = 0.035
        group_spacing = 0.02

        font_family = getattr(PlotStyle, "FONT_FAMILY", "serif")

        for lines, color, bold_idx in groups:
            for i, line in enumerate(lines):
                weight = "bold" if (bold_idx is not None and i == bold_idx) else "normal"
                size = 11 if weight == "bold" else 10

                # Special handling for Phase to make it pop?
                # Keeping simple for consistency first.

                current_color = color
                # Hack: semantic coloring within the group for "Error" lines?
                # The prompt asked for "Green Block" for Angle & Errors.
                # If we want errors red, we'd need line-by-line control.
                # For now, sticking to the section color logic to avoid complex spaghetti.

                self.ax_info.text(
                    0.05,
                    y_pos,
                    line,
                    fontsize=size,
                    weight=weight,
                    color=current_color,
                    verticalalignment="top",
                    fontfamily=font_family,
                )
                y_pos -= line_height
            y_pos -= group_spacing

    def animate_frame(self, frame: int) -> List[Any]:
        """Animation update function for each frame.

        Args:
            frame: Frame number
        """
        assert self.ax_main is not None, "ax_main must be initialized"
        assert self.dt is not None, "dt must be set"

        # Clear main plot
        self.ax_main.clear()
        self.ax_main.set_xlim(-3, 3)
        self.ax_main.set_ylim(-3, 3)
        self.ax_main.set_aspect("equal")
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlabel("X Position (m)")
        self.ax_main.set_ylabel("Y Position (m)")
        self.ax_main.set_title(self.frame_title_template.format(frame))

        # Get current data
        step = min(int(frame * self.speedup_factor), self._get_len() - 1)
        current_data = self._row(step)

        # Parse command vector and get active thrusters
        command_vector = self.parse_command_vector(current_data["Command_Vector"])
        active_thrusters = self.get_active_thrusters(command_vector)

        # Type-safe extraction with float conversion
        target_x_val = current_data["Target_X"]
        target_x = float(target_x_val) if target_x_val is not None else 0.0
        target_y_val = current_data["Target_Y"]
        target_y = float(target_y_val) if target_y_val is not None else 0.0
        target_yaw_val = current_data["Target_Yaw"]
        target_yaw = float(target_yaw_val) if target_yaw_val is not None else 0.0

        self.draw_target(target_x, target_y, target_yaw)

        # Draw DXF shape if mode is active OR if overlay_dxf option is enabled
        if getattr(SatelliteConfig, "DXF_SHAPE_MODE_ACTIVE", False) or self.overlay_dxf:
            self.draw_dxf_shape_overlays()

        # Draw trajectory up to current point
        trajectory_x = self._col("Current_X")[: step + 1].tolist()
        trajectory_y = self._col("Current_Y")[: step + 1].tolist()
        self.draw_trajectory(trajectory_x, trajectory_y)

        # Draw satellite with type-safe extraction
        current_x_val = current_data["Current_X"]
        current_x = float(current_x_val) if current_x_val is not None else 0.0
        current_y_val = current_data["Current_Y"]
        current_y = float(current_y_val) if current_y_val is not None else 0.0
        current_yaw_val = current_data["Current_Yaw"]
        current_yaw = float(current_yaw_val) if current_yaw_val is not None else 0.0

        self.draw_satellite(
            current_x,
            current_y,
            current_yaw,
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
        """Generate and save the MP4 animation using parallel rendering + imageio.

        Uses ProcessPoolExecutor for parallel frame rendering (fast),
        then imageio to assemble frames into MP4 (simple, no ffmpeg subprocess).

        Args:
            output_filename: Name of output MP4 file (optional)
        """
        assert self.fps is not None, "FPS must be calculated before generating animation"
        assert self.dt is not None, "dt must be set before generating animation"
        assert self.output_dir is not None, "Output directory must be set"

        if output_filename is None:
            output_filename = f"{self.plot_prefix}_animation.mp4"

        print(f"\n{'=' * 60}")
        print(f"GENERATING {self.system_name.upper()} ANIMATION (PARALLEL)")
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

        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        import imageio

        # Prepare configuration for workers
        if self._data_backend == "pandas" and self.data is not None:
            data_dict = {col: self.data[col].values for col in self.data.columns}
        elif self._col_data is not None:
            data_dict = self._col_data
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
            "dxf_mode_active": getattr(SatelliteConfig, "DXF_SHAPE_MODE_ACTIVE", False),
            "overlay_dxf": self.overlay_dxf,
            "obstacles_enabled": getattr(SatelliteConfig, "OBSTACLES_ENABLED", False),
            "obstacles_list": (
                SatelliteConfig.get_obstacles()
                if getattr(SatelliteConfig, "OBSTACLES_ENABLED", False)
                else []
            ),
            "data_directory": str(self.data_directory),
        }

        # Pass sibling control data if available (for Mission Phase info)
        if hasattr(self, "control_data") and self.control_data is not None:
            # Convert DataFrame to dict for pickling safety/simplicity
            config["control_data_dict"] = self.control_data.to_dict(orient="list")
        else:
            config["control_data_dict"] = None

        # Use absolute path to ensure file is saved correctly
        output_path = Path(self.output_dir).resolve() / output_filename

        # Use temp directory in output folder (subprocesses need absolute paths)
        # Create a unique frames subdirectory that we'll clean up afterward
        import uuid

        frames_dir = Path(self.output_dir).resolve() / f"_frames_{uuid.uuid4().hex[:8]}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(
                f"\nRendering {total_frames} frames using "
                f"{multiprocessing.cpu_count()} cores..."
            )

            frames = list(range(total_frames))

            # Parallel frame rendering - use absolute path for frames directory
            temp_dir = str(frames_dir)
            with ProcessPoolExecutor() as executor:
                results_iterator = executor.map(
                    UnifiedVisualizationGenerator._render_frame_task,
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

            # Assemble video with imageio (simpler than ffmpeg subprocess)
            print("Assembling video with imageio...")

            # Use exact FPS for precise real-time playback
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
                    macro_block_size=1,  # Avoid resizing warnings
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
            # Check if cache is None (it is always in globals as it's defined at module level)
            if _worker_gen_cache is None:
                # Initialize generator in worker process
                gen = UnifiedVisualizationGenerator(config["data_directory"], load_data=False)
                gen._col_data = data_dict
                gen._data_backend = "csv"

                # Inject config
                gen.dt = config["dt"]
                gen.fps = config["fps"]
                gen.speedup_factor = config["speedup_factor"]
                gen.system_title = config["system_title"]
                gen.frame_title_template = config["frame_title_template"]

                # Restore control data if provided
                if config.get("control_data_dict"):
                    try:
                        import pandas as pd

                        gen.control_data = pd.DataFrame(config["control_data_dict"])
                    except ImportError:
                        pass

                gen.satellite_size = config["satellite_size"]
                gen.satellite_color = config["satellite_color"]
                gen.target_color = config["target_color"]
                gen.trajectory_color = config["trajectory_color"]
                gen.thrusters = config["thrusters"]
                gen.dxf_base_shape = config["dxf_base_shape"]
                gen.dxf_offset_path = config["dxf_offset_path"]
                gen.dxf_center = config["dxf_center"]

                SatelliteConfig.DXF_SHAPE_MODE_ACTIVE = config["dxf_mode_active"]
                SatelliteConfig.OBSTACLES_ENABLED = config["obstacles_enabled"]
                if config["obstacles_enabled"] and config.get("obstacles_list"):
                    SatelliteConfig.set_obstacles(config["obstacles_list"])
                gen.overlay_dxf = config["overlay_dxf"]

                gen.setup_plot()
                _worker_gen_cache = gen
            else:
                gen = _worker_gen_cache

            if gen is not None:
                gen.animate_frame(frame_idx)
                filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
                if gen.fig is not None:
                    gen.fig.savefig(filename, dpi=100)
        except Exception:
            import traceback

            traceback.print_exc()
            return False

        return True

    def _progress_callback(self, current_frame: int, total_frames: int) -> None:
        """Progress callback for animation saving with visual progress bar.

        Args:
            current_frame: Current frame being processed
            total_frames: Total number of frames
        """
        progress = (current_frame / total_frames) * 100
        bar_length = 40
        filled = int(bar_length * current_frame / total_frames)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(
            f"\rAnimating: |{bar}| {progress:.1f}% ({current_frame}/{total_frames} frames)",
            end="",
            flush=True,
        )

    def generate_performance_plots(self) -> None:
        """Generate performance analysis plots."""
        assert self.dt is not None, "dt must be set before generating plots"
        assert self.output_dir is not None, "output_dir must be set before generating plots"

        print("Generating performance analysis plots...")

        # Create Plots subfolder
        plots_dir = self.output_dir / "Plots"
        plots_dir.mkdir(exist_ok=True)
        print(f" Created Plots directory: {plots_dir}")

        # Generate specific performance plots
        self._generate_position_tracking_plot(plots_dir)
        self._generate_position_error_plot(plots_dir)
        self._generate_angular_tracking_plot(plots_dir)
        self._generate_angular_error_plot(plots_dir)
        self._generate_trajectory_plot(plots_dir)
        self._generate_thruster_usage_plot(plots_dir)
        self._generate_thruster_valve_activity_plot(plots_dir)
        self._generate_pwm_quantization_plot(plots_dir)
        self._generate_control_effort_plot(plots_dir)
        self._generate_velocity_magnitude_plot(plots_dir)
        self._generate_mpc_performance_plot(plots_dir)
        self._generate_timing_intervals_plot(plots_dir)

        print(f"Performance plots saved to: {plots_dir}")

    def _generate_position_tracking_plot(self, plot_dir: Path) -> None:
        """Generate position tracking over time plot."""
        assert self.dt is not None, "dt must be set"

        fig, axes = plt.subplots(2, 1, figsize=PlotStyle.FIGSIZE_SUBPLOTS)
        fig.suptitle(f"Position Tracking - {self.system_title}")

        time = np.arange(self._get_len()) * float(self.dt)

        # X position tracking
        axes[0].plot(
            time,
            self._col("Current_X"),
            color=PlotStyle.COLOR_SIGNAL_POS,
            linewidth=PlotStyle.LINEWIDTH,
            label="Current X",
        )
        axes[0].plot(
            time,
            self._col("Target_X"),
            color=PlotStyle.COLOR_TARGET,
            linestyle="--",
            linewidth=PlotStyle.LINEWIDTH,
            label="Target X",
        )
        axes[0].set_ylabel("X Position (m)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[0].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[0].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[0].set_title("X Position Tracking")

        # Y position tracking
        axes[1].plot(
            time,
            self._col("Current_Y"),
            color=PlotStyle.COLOR_SIGNAL_POS,
            linewidth=PlotStyle.LINEWIDTH,
            label="Current Y",
        )
        axes[1].plot(
            time,
            self._col("Target_Y"),
            color=PlotStyle.COLOR_TARGET,
            linestyle="--",
            linewidth=PlotStyle.LINEWIDTH,
            label="Target Y",
        )
        axes[1].set_xlabel("Time (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[1].set_ylabel("Y Position (m)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[1].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[1].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[1].set_title("Y Position Tracking")

        PlotStyle.save_figure(fig, plot_dir / "position_tracking.png")

    def _generate_position_error_plot(self, plot_dir: Path) -> None:
        """Generate position error plot."""
        assert self.dt is not None, "dt must be set"

        fig, ax = plt.subplots(1, 1, figsize=PlotStyle.FIGSIZE_SINGLE)

        time = np.arange(self._get_len()) * float(self.dt)
        pos_error = np.sqrt(self._col("Error_X") ** 2 + self._col("Error_Y") ** 2)

        target_threshold = getattr(SatelliteConfig, "POSITION_TOLERANCE", 0.05)

        ax.plot(
            time,
            pos_error,
            color=PlotStyle.COLOR_SIGNAL_POS,
            linewidth=PlotStyle.LINEWIDTH,
            label="Position Error",
        )
        ax.axhline(
            y=target_threshold,
            color=PlotStyle.COLOR_THRESHOLD,
            linestyle="--",
            linewidth=PlotStyle.LINEWIDTH,
            alpha=0.8,
            label=f"Target Threshold ({target_threshold}m)",
        )

        ax.set_xlabel("Time (seconds)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_ylabel("Position Error (meters)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_title(f"Position Error Over Time - {self.system_title}")
        ax.grid(True, alpha=PlotStyle.GRID_ALPHA)
        ax.legend(fontsize=PlotStyle.LEGEND_SIZE)

        # Add final error as text
        final_error = pos_error[-1] if len(pos_error) > 0 else 0.0
        ax.text(
            0.02,
            0.98,
            f"Final Error: {final_error:.3f}m",
            transform=ax.transAxes,
            fontsize=PlotStyle.ANNOTATION_SIZE,
            verticalalignment="top",
            bbox=PlotStyle.TEXTBOX_STYLE,
        )

        PlotStyle.save_figure(fig, plot_dir / "position_error.png")

    @staticmethod
    def _break_wrap_around(times, values, threshold=300):
        """Insert NaNs to break lines where values wrap around (e.g. 359 -> 1)."""
        values = np.array(values, dtype=float)
        times = np.array(times, dtype=float)  # Ensure float to hold NaNs

        # Calculate absolute differences
        diffs = np.abs(np.diff(values))

        # Find indices where the jump is too large (wrapping)
        wrap_indices = np.where(diffs > threshold)[0]

        if len(wrap_indices) == 0:
            return times, values

        # Insert NaNs at wrap points
        times_clean = np.insert(times, wrap_indices + 1, np.nan)
        values_clean = np.insert(values, wrap_indices + 1, np.nan)

        return times_clean, values_clean

    def _generate_angular_tracking_plot(self, plot_dir: Path) -> None:
        """Generate angular tracking plot (0-360° visualization with wrap handling)."""
        assert self.dt is not None, "dt must be set"

        fig, ax = plt.subplots(1, 1, figsize=PlotStyle.FIGSIZE_SINGLE)

        time = np.arange(self._get_len()) * float(self.dt)

        def to_0_360(rad_series):
            deg = np.degrees(np.array(rad_series))
            return np.mod(deg, 360.0)

        # Map to 0-360 range
        cur_deg = to_0_360(self._col("Current_Yaw"))
        tgt_deg = to_0_360(self._col("Target_Yaw"))

        # Clean wrap-around artifacts so 359->1 doesn't draw a vertical line
        t_cur, y_cur = self._break_wrap_around(time, cur_deg)
        t_tgt, y_tgt = self._break_wrap_around(time, tgt_deg)

        ax.plot(
            t_cur,
            y_cur,
            color=PlotStyle.COLOR_SIGNAL_ANG,
            linewidth=PlotStyle.LINEWIDTH,
            label="Current Yaw",
        )
        ax.plot(
            t_tgt,
            y_tgt,
            color=PlotStyle.COLOR_TARGET,
            linestyle="--",
            linewidth=PlotStyle.LINEWIDTH,
            label="Target Yaw",
        )

        ax.set_xlabel("Time (seconds)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_ylabel("Yaw Angle (degrees, 0–360°)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_title(f"Angular Tracking - {self.system_title}")

        # Set Y-axis to 0-360
        ax.set_ylim(0, 360)
        ax.set_yticks(np.arange(0, 361, 45))

        ax.grid(True, alpha=PlotStyle.GRID_ALPHA)
        ax.legend(fontsize=PlotStyle.LEGEND_SIZE)

        PlotStyle.save_figure(fig, plot_dir / "angular_tracking.png")

    def _generate_angular_error_plot(self, plot_dir: Path) -> None:
        """Generate angular error plot."""
        assert self.dt is not None, "dt must be set"

        fig, ax = plt.subplots(1, 1, figsize=PlotStyle.FIGSIZE_SINGLE)

        time = np.arange(self._get_len()) * float(self.dt)
        angle_error = np.abs(np.degrees(self._col("Error_Yaw")))

        # Get threshold in degrees (config stored in radians)
        target_threshold_rad = getattr(SatelliteConfig, "ANGLE_TOLERANCE", np.deg2rad(3))
        target_threshold_deg = np.degrees(target_threshold_rad)

        ax.plot(
            time,
            angle_error,
            color=PlotStyle.COLOR_SIGNAL_ANG,
            linewidth=PlotStyle.LINEWIDTH,
            label="Angular Error",
        )
        ax.axhline(
            y=target_threshold_deg,
            color=PlotStyle.COLOR_THRESHOLD,
            linestyle="--",
            linewidth=PlotStyle.LINEWIDTH,
            alpha=0.8,
            label=f"Target Threshold ({target_threshold_deg:.1f}°)",
        )

        ax.set_xlabel("Time (seconds)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_ylabel("Angular Error (degrees)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_title(f"Angular Error Over Time - {self.system_title}")
        ax.grid(True, alpha=PlotStyle.GRID_ALPHA)
        ax.legend(fontsize=PlotStyle.LEGEND_SIZE)

        # Add final error as text
        final_error = angle_error[-1] if len(angle_error) > 0 else 0.0
        ax.text(
            0.02,
            0.98,
            f"Final Error: {final_error:.1f}°",
            transform=ax.transAxes,
            fontsize=PlotStyle.ANNOTATION_SIZE,
            verticalalignment="top",
            bbox=PlotStyle.TEXTBOX_STYLE,
        )

        PlotStyle.save_figure(fig, plot_dir / "angular_error.png")

    def _generate_trajectory_plot(self, plot_dir: Path) -> None:
        """Generate trajectory plot."""
        fig, ax = plt.subplots(1, 1, figsize=PlotStyle.FIGSIZE_SUBPLOTS)

        x_pos = self._col("Current_X")
        y_pos = self._col("Current_Y")
        target_x_col = self._col("Target_X")
        target_y_col = self._col("Target_Y")
        target_x = target_x_col[0] if len(target_x_col) > 0 else 0.0
        target_y = target_y_col[0] if len(target_y_col) > 0 else 0.0

        # Plot trajectory
        ax.plot(
            x_pos,
            y_pos,
            color=PlotStyle.COLOR_SIGNAL_POS,
            linewidth=PlotStyle.LINEWIDTH_THICK,
            alpha=0.8,
            label="Satellite Path",
        )
        if len(x_pos) > 0:
            ax.plot(
                x_pos[0],
                y_pos[0],
                "o",
                color=PlotStyle.COLOR_SUCCESS,
                markersize=PlotStyle.MARKER_SIZE,
                label="Start Position",
            )
            ax.plot(
                x_pos[-1],
                y_pos[-1],
                "o",
                color=PlotStyle.COLOR_ERROR,
                markersize=PlotStyle.MARKER_SIZE,
                label="Final Position",
            )
        ax.plot(target_x, target_y, "r*", markersize=20, label="Target")

        try:
            if isinstance(self.dxf_base_shape, (list, tuple)) and len(self.dxf_base_shape) >= 3:
                bx = [p[0] for p in self.dxf_base_shape] + [self.dxf_base_shape[0][0]]
                by = [p[1] for p in self.dxf_base_shape] + [self.dxf_base_shape[0][1]]
                ax.plot(
                    bx,
                    by,
                    color="#9b59b6",
                    linewidth=PlotStyle.LINEWIDTH,
                    alpha=0.7,
                    label="Object Shape",
                )
            if isinstance(self.dxf_offset_path, (list, tuple)) and len(self.dxf_offset_path) >= 2:
                px = [p[0] for p in self.dxf_offset_path]
                py = [p[1] for p in self.dxf_offset_path]
                ax.plot(
                    px,
                    py,
                    color="#e67e22",
                    linewidth=PlotStyle.LINEWIDTH,
                    alpha=0.8,
                    linestyle="--",
                    label="Shape Path",
                )
            if isinstance(self.dxf_center, (list, tuple)) and len(self.dxf_center) == 2:
                ax.scatter(
                    self.dxf_center[0],
                    self.dxf_center[1],
                    c="#2ecc71",
                    s=60,
                    marker="+",
                    linewidth=PlotStyle.LINEWIDTH,
                    label="Shape Center",
                )
        except Exception:
            pass

        # Add target circle
        circle = Circle(
            (target_x, target_y),
            0.1,
            color=PlotStyle.COLOR_TARGET,
            fill=False,
            linewidth=PlotStyle.LINEWIDTH,
            linestyle="--",
            alpha=0.7,
            label="Target Zone (±0.1m)",
        )
        ax.add_patch(circle)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel("X Position (meters)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_ylabel("Y Position (meters)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_title(f"Satellite Trajectory - {self.system_title}")
        ax.grid(True, alpha=PlotStyle.GRID_ALPHA)
        ax.legend(fontsize=PlotStyle.LEGEND_SIZE)
        ax.set_aspect("equal")

        # Add distance info
        if len(x_pos) > 0:
            final_distance = np.sqrt((x_pos[-1] - target_x) ** 2 + (y_pos[-1] - target_y) ** 2)
        else:
            final_distance = 0.0
        ax.text(
            0.02,
            0.98,
            f"Final Distance to Target: {final_distance:.3f}m",
            transform=ax.transAxes,
            fontsize=PlotStyle.ANNOTATION_SIZE,
            verticalalignment="top",
            bbox=PlotStyle.TEXTBOX_STYLE,
        )

        PlotStyle.save_figure(fig, plot_dir / "trajectory.png")

    def _generate_thruster_usage_plot(self, plot_dir: Path) -> None:
        """Generate thruster usage plot using actual valve states.

        Uses Thruster_X_Val columns from physics_data.csv which captures the
        actual valve state (0.0-1.0) at every physics timestep (5ms). This
        properly accounts for PWM duty cycles where thrusters are only active
        for part of a control interval.
        """
        assert self.dt is not None, "dt must be set"

        fig, ax = plt.subplots(1, 1, figsize=PlotStyle.FIGSIZE_SUBPLOTS)

        thruster_ids = np.arange(1, 9)  # Thrusters 1-8
        total_activation_time = np.zeros(8)
        data_source = "commanded"

        # Try to use actual valve states (Thruster_X_Val) for accurate PWM tracking
        has_valve_data = False
        if self._data_backend == "pandas" and self.data is not None:
            if "Thruster_1_Val" in self.data.columns:
                has_valve_data = True
        elif self._col_data is not None:
            if "Thruster_1_Val" in self._col_data:
                has_valve_data = True

        if has_valve_data:
            # Use actual valve states - most accurate for PWM
            data_source = "actual valve"
            for i in range(8):
                col_name = f"Thruster_{i+1}_Val"
                vals = self._col(col_name)
                try:
                    vals = np.array([float(x) for x in vals])
                except (ValueError, TypeError):
                    vals = np.zeros(len(vals), dtype=float)
                # Sum valve states and multiply by physics timestep
                total_activation_time[i] = np.sum(vals) * float(self.dt)
        else:
            # Fallback: Use commanded duty cycles from Command_Vector
            command_data = []
            for idx in range(self._get_len()):
                row = self._row(idx)
                cmd_vec = self.parse_command_vector(row["Command_Vector"])
                command_data.append(cmd_vec)
            command_matrix = np.array(command_data)
            total_activation_time = np.sum(command_matrix, axis=0) * float(self.dt)

        # Create bar plot
        bars = ax.bar(
            thruster_ids,
            total_activation_time,
            color=PlotStyle.COLOR_BARS,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
        )

        # Add value labels on top of bars
        for _i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}s",
                ha="center",
                va="bottom",
                fontsize=PlotStyle.ANNOTATION_SIZE,
                fontfamily=(
                    PlotStyle.FONT_FAMILY if hasattr(PlotStyle, "FONT_FAMILY") else "serif"
                ),
            )

        ax.set_xlabel("Thruster ID", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_ylabel("Total Active Time (seconds)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_title(
            f"Thruster Usage Summary - {self.system_title}",
            fontsize=PlotStyle.TITLE_SIZE,
            fontweight="bold",
        )
        ax.grid(True, axis="y", alpha=PlotStyle.GRID_ALPHA)  # Vert grid less useful for bar charts
        ax.set_xticks(thruster_ids)

        total_thruster_seconds = float(np.sum(total_activation_time))
        ax.text(
            0.98,
            0.95,
            f"Total active time ({data_source}): {total_thruster_seconds:.2f}s",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=PlotStyle.ANNOTATION_SIZE,
            bbox=PlotStyle.TEXTBOX_STYLE,
        )

        PlotStyle.save_figure(fig, plot_dir / "thruster_usage.png")

    def _generate_thruster_valve_activity_plot(self, plot_dir: Path) -> None:
        """Generate detailed valve activity plot for each thruster (0.0 to 1.0)."""
        assert self.dt is not None, "dt must be set"

        # Check if we have valve data
        has_valve_data = False
        if self._data_backend == "pandas" and self.data is not None:
            if "Thruster_1_Val" in self.data.columns:
                has_valve_data = True
        elif self._col_data is not None:
            if "Thruster_1_Val" in self._col_data:
                has_valve_data = True

        if not has_valve_data:
            return

        fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
        axes = axes.flatten()

        time = np.arange(self._get_len()) * float(self.dt)

        for i in range(8):
            thruster_id = i + 1
            col_name_val = f"Thruster_{thruster_id}_Val"
            col_name_cmd = f"Thruster_{thruster_id}_Cmd"

            # Get Valve data
            vals = self._col(col_name_val)
            try:
                vals = np.array([float(x) for x in vals])
            except (ValueError, TypeError):
                vals = np.zeros_like(vals, dtype=float)

            # Get Command data (Duty Cycle)
            cmds = self._col(col_name_cmd)
            try:
                cmds = np.array([float(x) for x in cmds])
            except (ValueError, TypeError):
                # If command column missing or invalid, default to zeros
                cmds = np.zeros_like(vals, dtype=float)

            # 1. Update Combined Plot
            ax = axes[i]
            # Fill area for actual valve state
            ax.fill_between(time, vals, color="tab:blue", alpha=0.3, label="Valve")
            ax.plot(time, vals, color="tab:blue", linewidth=1)
            # Overlay Commanded Duty Cycle
            ax.plot(
                time,
                cmds,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label="Command",
            )

            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(f"T{thruster_id}")
            ax.set_title(f"Thruster {thruster_id}", fontsize=10, pad=2)

            if i >= 6:
                ax.set_xlabel("Time (s)")

            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

            # 2. Generate Separate Plot for this Thruster (Two subplots)
            fig_ind, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

            # Top plot: Binary valve state (0 or 1)
            ax_top.fill_between(
                time,
                vals,
                color="tab:blue",
                alpha=0.3,
            )
            ax_top.plot(time, vals, color="tab:blue", linewidth=1)
            ax_top.set_ylim(-0.1, 1.1)
            ax_top.set_yticks([0, 1])
            ax_top.set_yticklabels(["OFF", "ON"])
            ax_top.grid(True, alpha=0.3)
            ax_top.set_ylabel("Valve State")
            ax_top.set_title(f"Thruster {thruster_id} - {self.system_title}", fontsize=12)

            # Bottom plot: Commanded duty cycle (u value)
            ax_bot.plot(
                time,
                cmds,
                color="red",
                linewidth=1.5,
            )
            ax_bot.fill_between(time, cmds, color="red", alpha=0.2)
            ax_bot.set_ylim(-0.05, 1.05)
            ax_bot.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_bot.grid(True, alpha=0.3)
            ax_bot.set_ylabel("Cmd Duty Cycle")
            ax_bot.set_xlabel("Time (s)")

            plt.tight_layout()
            # Save individual plot
            plt.savefig(
                plot_dir / f"thruster_{thruster_id}_valve_activity.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_ind)

        fig.suptitle(
            f"Thruster Valve Activity (0.0 - 1.0) - {self.system_title}",
            fontsize=16,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(
            plot_dir / "thruster_valve_activity.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _generate_pwm_quantization_plot(self, plot_dir: Path) -> None:
        """Generate PWM duty cycle plot showing MPC u-values vs time.

        This plot proves PWM MPC outputs continuous duty cycles by showing:
        - The commanded duty cycle (u = 0-1) for each thruster over time
        - Uses control_data.csv which logs at each MPC update (60ms)
        """
        assert self.dt is not None, "dt must be set"

        # Use control_data which has the actual MPC outputs per control step
        control_df = None
        if hasattr(self, "control_data") and self.control_data is not None:
            control_df = self.control_data

        if control_df is None or "Command_Vector" not in control_df.columns:
            # Fallback: check if physics_data has the cmd columns
            return self._generate_pwm_duty_cycles_from_physics(plot_dir)

        # Parse Command_Vector column to get duty cycles per thruster
        time = control_df["Control_Time"].values

        # Parse the command vectors
        duty_cycles_per_thruster: Dict[int, List[float]] = {i: [] for i in range(8)}

        for cmd_str in control_df["Command_Vector"]:
            try:
                # Parse "[0.000, 1.000, ...]" format
                values = cmd_str.strip('[]"').split(",")
                for i, v in enumerate(values[:8]):
                    duty_cycles_per_thruster[i].append(float(v.strip()))
            except BaseException:
                for i in range(8):
                    duty_cycles_per_thruster[i].append(0.0)

        # Create figure with 8 subplots (one per thruster)
        fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True, sharey=True)
        axes = axes.flatten()
        fig.suptitle(
            f"PWM MPC Duty Cycle Output (u) vs Time - {self.system_title}",
            fontsize=14,
            fontweight="bold",
        )

        colors = plt.cm.tab10(np.linspace(0, 1, 8))  # type: ignore[attr-defined]

        # Track if we find any intermediate values
        all_intermediate_values = []

        for i, ax in enumerate(axes):
            thruster_id = i + 1
            duty_cycles = np.array(duty_cycles_per_thruster[i])

            # Plot duty cycle over time as step function
            ax.step(
                time,
                duty_cycles,
                where="post",
                color=colors[i],
                linewidth=1.5,
                alpha=0.9,
            )
            ax.fill_between(time, 0, duty_cycles, step="post", color=colors[i], alpha=0.3)

            # Add horizontal lines at key duty cycle levels
            ax.axhline(y=0.0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
            ax.axhline(y=1.0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

            # Find intermediate values (not 0 or 1)
            intermediate = [(t, u) for t, u in zip(time, duty_cycles) if 0.01 < u < 0.99]
            if intermediate:
                all_intermediate_values.extend([u for _, u in intermediate])
                # Mark intermediate points with dots
                for t_val, u_val in intermediate:
                    ax.plot(t_val, u_val, "ko", markersize=4)

            # Labels
            ax.set_ylabel(f"T{thruster_id}", fontsize=10, fontweight="bold")
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(True, alpha=0.3)

            # Stats annotation
            intermediate_count = len(intermediate)
            if intermediate_count > 0:
                ax.text(
                    0.98,
                    0.95,
                    f"Intermediate: {intermediate_count}",
                    transform=ax.transAxes,
                    fontsize=8,
                    ha="right",
                    va="top",
                    bbox=dict(
                        facecolor="lightgreen",
                        alpha=0.7,
                        boxstyle="round,pad=0.2",
                    ),
                )

        # X-axis label on bottom row
        axes[6].set_xlabel("Time (s)", fontsize=12)
        axes[7].set_xlabel("Time (s)", fontsize=12)

        # Add summary annotation
        if all_intermediate_values:
            unique_values = sorted(set([round(v, 3) for v in all_intermediate_values]))
            n_instances = len(all_intermediate_values)
            summary = f"Intermediate u-values found: {n_instances} instances\n"
            ellipsis = "..." if len(unique_values) > 10 else ""
            summary += f"Unique values: {unique_values[:10]}{ellipsis}\n"
            summary += "✓ PWM MPC outputs continuous duty cycles [0, 1]"
            fig.text(
                0.5,
                0.01,
                summary,
                ha="center",
                fontsize=10,
                bbox=dict(facecolor="lightgreen", alpha=0.8, boxstyle="round,pad=0.5"),
            )
        else:
            fig.text(
                0.5,
                0.01,
                "No intermediate u-values found (all 0 or 1)\n"
                "MPC chose full thrust for this scenario",
                ha="center",
                fontsize=10,
                bbox=dict(
                    facecolor="lightyellow",
                    alpha=0.8,
                    boxstyle="round,pad=0.5",
                ),
            )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(plot_dir / "pwm_duty_cycles.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_pwm_duty_cycles_from_physics(self, plot_dir: Path) -> None:
        """Fallback: Generate PWM plot from physics_data Thruster_X_Cmd columns."""
        # This shows binary valve states, not continuous duty cycles
        pass

    def _generate_control_effort_plot(self, plot_dir: Path) -> None:
        """Generate control effort plot."""
        assert self.dt is not None, "dt must be set"

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        time = np.arange(self._get_len()) * float(self.dt)

        # Parse command vectors
        command_data = []
        for idx in range(self._get_len()):
            row = self._row(idx)
            cmd_vec = self.parse_command_vector(row["Command_Vector"])
            command_data.append(cmd_vec)
        command_matrix = np.array(command_data)

        total_effort_per_step = np.sum(command_matrix, axis=1)
        ax.plot(time, total_effort_per_step, "c-", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Total Control Effort")
        ax.set_title(f"Control Effort Over Time - {self.system_title}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / "control_effort.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_velocity_magnitude_plot(self, plot_dir: Path) -> None:
        """Generate velocity magnitude over time plot (speed vs time)."""
        assert self.dt is not None, "dt must be set"

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        n = self._get_len()
        if n < 2:
            ax.text(
                0.5,
                0.5,
                "Not enough data to compute velocity",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            ax.set_title(f"Velocity Magnitude - {self.system_title}")
            plt.tight_layout()
            plt.savefig(
                plot_dir / "velocity_magnitude.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            return

        dt_float = float(self.dt)
        time = np.arange(n) * dt_float
        x = self._col("Current_X")
        y = self._col("Current_Y")

        vx = np.gradient(x, dt_float)
        vy = np.gradient(y, dt_float)
        speed = np.sqrt(vx**2 + vy**2)

        ax.plot(time, speed, color="teal", linewidth=2.5, label="Speed (|v|)")
        ax.axhline(
            y=float(np.mean(speed)),
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {np.mean(speed):.3f} m/s",
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title(f"Velocity Magnitude Over Time - {self.system_title}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(plot_dir / "velocity_magnitude.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_mpc_performance_plot(self, plot_dir: Path) -> None:
        """Generate MPC performance plot."""
        assert self.dt is not None, "dt must be set"

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Determine data source
        df = None
        cols = []

        # Check for sibling control data first (dual-log mode)
        if hasattr(self, "control_data") and self.control_data is not None:
            df = self.control_data
            cols = df.columns
            print("Using sibling control_data for MPC performance plot")
        elif self._data_backend == "pandas" and self.data is not None:
            df = self.data
            cols = df.columns
        elif self._col_data is not None:
            cols = list(self._col_data.keys())

        if "MPC_Computation_Time" in cols:
            # Determine time axis
            if df is not None and "Control_Time" in cols:
                time = df["Control_Time"].values
            elif df is not None and "CONTROL_DT" in cols:
                dt_val = df["CONTROL_DT"].iloc[0]
                time = np.arange(len(df)) * float(dt_val)
            else:
                # Fallback to main dt (might be wrong if mixed, but better than
                # nothing)
                time = np.arange(len(df) if df is not None else self._get_len()) * float(self.dt)

            # Ensure we have numeric data - convert and handle non-numeric
            # values
            if df is not None:
                raw_comp_times = df["MPC_Solve_Time"].values
            else:
                raw_comp_times = self._col("MPC_Solve_Time")

            comp_times = []
            for val in raw_comp_times:
                try:
                    comp_times.append(float(val) * 1000)  # Convert to ms
                except (ValueError, TypeError):
                    comp_times.append(0.0)  # Default for invalid values
            comp_times = np.array(comp_times)  # type: ignore[assignment]

            # Optional: solver time limit per step
            limit_ms = None
            if "MPC_Solver_Time_Limit" in cols:
                if df is not None:
                    raw_limits = df["MPC_Solver_Time_Limit"].values
                else:
                    raw_limits = self._col("MPC_Solver_Time_Limit")
                limits = []
                for val in raw_limits:
                    try:
                        limits.append(float(val) * 1000)
                    except (ValueError, TypeError):
                        limits.append(0.0)
                limit_ms = np.array(limits)

            ax.plot(
                time,
                comp_times,
                "purple",
                linewidth=2,
                label="Computation Time",
            )
            mean_ms = float(np.mean(comp_times))
            max_ms = float(np.max(comp_times))
            ax.axhline(
                y=mean_ms,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Mean: {mean_ms:.1f} ms",
            )
            ax.axhline(
                y=max_ms,
                color="g",
                linestyle=":",
                alpha=0.7,
                label=f"Max: {max_ms:.1f} ms",
            )
            if limit_ms is not None and np.any(limit_ms > 0):
                # If limit varies, plot as line; else constant hline
                if len(np.unique(limit_ms)) > 1:
                    ax.plot(
                        time,
                        limit_ms,
                        color="black",
                        linestyle=":",
                        alpha=0.6,
                        label="Time Limit",
                    )
                else:
                    limit_val = float(limit_ms[0] if isinstance(limit_ms, np.ndarray) else limit_ms)
                    if limit_val > 0:
                        ax.axhline(
                            y=limit_val,
                            color="black",
                            linestyle=":",
                            alpha=0.6,
                            label="Time Limit",
                        )
                # Highlight exceedances
                if "MPC_Time_Limit_Exceeded" in cols:
                    exceeded_vals = self._col("MPC_Time_Limit_Exceeded")
                    exceeded_idx = []
                    for i, val in enumerate(exceeded_vals):
                        try:
                            if bool(val) or (isinstance(val, str) and val.lower() == "true"):
                                exceeded_idx.append(i)
                        except Exception:
                            pass
                    exceeded_idx_arr = np.array(exceeded_idx)
                    if len(exceeded_idx_arr) > 0:
                        ax.scatter(
                            np.array(time)[exceeded_idx_arr],
                            comp_times[exceeded_idx_arr],
                            color="red",
                            s=25,
                            label="Exceeded",
                            zorder=5,
                        )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Computation Time (ms)")
            ax.set_title(f"MPC Computation Time - {self.system_title}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "MPC Computation Time\nData Not Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            ax.set_title(f"MPC Computation Time - {self.system_title}")

        plt.tight_layout()
        plt.savefig(plot_dir / "mpc_performance.png", dpi=300, bbox_inches="tight")
        plt.close()

    # --- CSV backend helpers ---
    def _load_csv_data_csvmodule(self) -> None:
        """Load CSV data using csv module backend."""
        assert self.csv_path is not None, "CSV path must be set"

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                rows.append(r)
        # Build column-wise data with type conversion
        cols = reader.fieldnames or []
        col_data: Dict[str, List[Any]] = {c: [] for c in cols}
        for r in rows:
            for c in cols:
                v = r.get(c, "")
                if v is None:
                    col_data[c].append("")
                    continue
                # Convert booleans and numbers where applicable
                lv = v.strip()
                if lv.lower() in ("true", "false"):
                    col_data[c].append(lv.lower() == "true")
                else:
                    try:
                        col_data[c].append(float(lv))
                    except Exception:
                        col_data[c].append(v)
        # Store
        self._rows = rows
        self._col_data = {k: np.array(v) for k, v in col_data.items()}
        self.data = self  # allow attribute access in unchanged code paths

    def _get_len(self) -> int:
        """Get length of data."""
        if self._data_backend == "pandas":
            return len(self.data) if self.data is not None else 0
        if self._rows is not None:
            return len(self._rows)
        if self._col_data is not None and len(self._col_data) > 0:
            # Get length of first column
            first_key = next(iter(self._col_data))
            return len(self._col_data[first_key])
        return 0

    def _col(self, name: str) -> np.ndarray:
        """Get column data."""
        if self._data_backend == "pandas" and self.data is not None:
            return (
                self.data[name].values
                if hasattr(self.data[name], "values")
                else np.array(self.data[name])
            )
        return (
            self._col_data.get(name, np.array([])) if self._col_data is not None else np.array([])
        )

    def _row(self, idx: int) -> Dict[str, Any]:
        """Get row data."""
        if self._data_backend == "pandas" and self.data is not None:
            row_data: Dict[str, Any] = dict(self.data.iloc[idx])
            return row_data
        # Build a dict using typed column arrays so consumers see floats/bools
        if self._col_data is not None:
            return {
                k: (self._col_data[k][idx] if k in self._col_data else None)
                for k in self._col_data.keys()
            }
        return {}

    def _generate_timing_intervals_plot(self, plot_dir: Path) -> None:
        """Generate timing intervals plot."""
        assert self.dt is not None, "dt must be set"

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Check for sibling control data first (dual-log mode)
        df = None
        cols = []
        if hasattr(self, "control_data") and self.control_data is not None:
            df = self.control_data
            cols = df.columns
        elif self._data_backend == "pandas" and self.data is not None:
            df = self.data
            cols = df.columns
        elif self._col_data is not None:
            cols = list(self._col_data.keys())

        if "Actual_Time_Interval" in cols:
            # Determine time axis
            if df is not None and "Control_Time" in cols:
                time = df["Control_Time"].values
            elif df is not None and "CONTROL_DT" in cols:
                dt_val = df["CONTROL_DT"].iloc[0]
                time = np.arange(len(df)) * float(dt_val)
            else:
                time = np.arange(len(df) if df is not None else self._get_len()) * float(self.dt)

            # Get intervals
            if df is not None:
                intervals = df["Actual_Time_Interval"].values
            else:
                intervals = self._col("Actual_Time_Interval")

            # Determine target dt
            target_dt = self.dt
            if df is not None and "CONTROL_DT" in cols:
                target_dt = float(df["CONTROL_DT"].iloc[0])

            ax.plot(
                time,
                intervals,
                "orange",
                linewidth=2,
                label="Actual Intervals",
            )
            ax.axhline(
                y=target_dt,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Target: {target_dt:.3f}s",
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Time Interval (s)")
            ax.set_title(f"Timing Intervals - {self.system_title}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Timing Interval\nData Not Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            ax.set_title(f"Timing Intervals - {self.system_title}")

        plt.tight_layout()
        plt.savefig(plot_dir / "timing_intervals.png", dpi=300, bbox_inches="tight")
        plt.close()


class LinearizedVisualizationGenerator(UnifiedVisualizationGenerator):
    """Legacy compatibility class for simulation visualization."""

    def __init__(self, data_directory: str = "Data/Simulation"):
        super().__init__(data_directory)


class RealLinearizedVisualizationGenerator(UnifiedVisualizationGenerator):
    """Simulation-only build stub for legacy real test visualization."""

    def __init__(self, data_directory: str = "Data/Real_Test", interactive: bool = False):
        raise RuntimeError("Simulation-only build: real-data visualization has been removed.")


def get_demo_shape(shape_type: str) -> List[tuple]:
    """Get predefined demo shape points (matches Mission.py implementation)."""
    if shape_type == "rectangle":
        # 0.4m x 0.3m rectangle centered at origin
        return [
            (-0.2, -0.15),
            (0.2, -0.15),
            (0.2, 0.15),
            (-0.2, 0.15),
            (-0.2, -0.15),  # Close the shape
        ]
    elif shape_type == "triangle":
        # Equilateral triangle with 0.4m sides
        return [(0.0, 0.2), (-0.173, -0.1), (0.173, -0.1), (0.0, 0.2)]
    elif shape_type == "hexagon":
        # Regular hexagon with 0.2m radius
        points = []
        for i in range(7):  # 7 points to close the shape
            angle = i * np.pi / 3
            x = 0.2 * np.cos(angle)
            y = 0.2 * np.sin(angle)
            points.append((x, y))
        return points
    else:
        # Default to rectangle
        return get_demo_shape("rectangle")


def transform_shape(points: List[tuple], center: tuple, rotation: float) -> List[tuple]:
    """Transform shape points to specified center and rotation."""
    transformed = []
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    for x, y in points:
        # Rotate
        x_rot = x * cos_r - y * sin_r
        y_rot = x * sin_r + y * cos_r

        # Translate
        x_final = x_rot + center[0]
        y_final = y_rot + center[1]

        transformed.append((x_final, y_final))

    return transformed


def make_offset_path(points: List[tuple], offset_distance: float) -> List[tuple]:
    """Create an outward offset path (matches Mission.py implementation)."""
    if len(points) < 3:
        return points

    # Try to use DXF_Viewer's make_offset_path if available
    try:
        from DXF.dxf_viewer import make_offset_path as viewer_make_offset_path

        return viewer_make_offset_path(
            points,
            float(offset_distance),
            1.0,
            join="round",
            resolution=24,
            mode="buffer",
        )
    except Exception:
        pass

    # Fallback: simple centroid-based offset
    pts = points[:]
    if np.linalg.norm(np.array(pts[0]) - np.array(pts[-1])) > 1e-8:
        pts = pts + [pts[0]]

    # Calculate shape centroid
    centroid_x = np.mean([p[0] for p in pts])
    centroid_y = np.mean([p[1] for p in pts])
    centroid = np.array([centroid_x, centroid_y])

    upscaled = []
    for i in range(len(pts) - 1):
        current = np.array(pts[i])
        next_point = np.array(pts[i + 1])

        edge_vec = next_point - current
        edge_normal = np.array([-edge_vec[1], edge_vec[0]])
        if np.linalg.norm(edge_normal) > 0:
            edge_normal = edge_normal / np.linalg.norm(edge_normal)

        mid_point = (current + next_point) / 2
        to_mid = mid_point - centroid
        if np.dot(edge_normal, to_mid) < 0:
            edge_normal = -edge_normal

        offset_current = current + edge_normal * offset_distance
        upscaled.append(tuple(offset_current))

    if np.linalg.norm(np.array(pts[0]) - np.array(pts[-1])) < 1e-6:
        upscaled.append(upscaled[0])

    return upscaled


def load_dxf_shape(dxf_path: str) -> List[tuple]:
    """Load shape points from DXF file using DXF_Viewer pipeline (matches Mission.py)."""
    import ezdxf

    try:
        from DXF.dxf_viewer import (
            extract_boundary_polygon,
            sanitize_boundary,
            units_code_to_name_and_scale,
        )
    except Exception as e:
        raise ImportError(f"dxf_viewer utilities unavailable: {e}")

    # Read DXF and determine units
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    insunits = int(doc.header.get("$INSUNITS", 0))
    units_name, to_m = units_code_to_name_and_scale(insunits)

    # Extract and sanitize boundary in native units, then scale to meters
    boundary = extract_boundary_polygon(msp)
    boundary = sanitize_boundary(boundary, to_m)
    boundary_m = [(float(x) * to_m, float(y) * to_m) for (x, y) in boundary] if boundary else []

    if not boundary_m:
        raise ValueError("No usable DXF boundary could be constructed.")

    print(f" DXF Loaded: {len(boundary_m)} points")
    print(f"   Units: {units_name} (INSUNITS={insunits}), scaled → meters (x{to_m})")
    return boundary_m


def configure_dxf_overlay_interactive() -> bool:
    """Interactive configuration of DXF shape overlay.

    Prompts user for shape selection, center, rotation, and offset,
    then sets the configuration in SatelliteConfig.

    Returns:
        True if overlay was configured, False if user cancelled
    """
    print("\n" + "=" * 60)
    print("   DXF SHAPE OVERLAY CONFIGURATION")
    print("=" * 60)

    # Ask if user wants overlay
    print("\nAdd DXF shape overlay to animation?")
    response = (
        input("Enter 'yes' or 'y' to configure overlay (or press Enter to skip): ").strip().lower()
    )
    if response not in ["yes", "y"]:
        print("Skipping DXF overlay.")
        return False

    # Shape selection
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
        print("2. Use demo rectangle")
        print("3. Use demo triangle")
        print("4. Use demo hexagon")
        choice = input("Select option (1-4): ").strip()

        if choice == "1":
            dxf_path = input("Enter DXF file path: ").strip()
            try:
                shape_points = load_dxf_shape(dxf_path)
                print(f" Loaded shape with {len(shape_points)} points from DXF")
            except Exception as e:
                print(f" Failed to load DXF: {e}")
                print("Falling back to demo rectangle")
                shape_points = get_demo_shape("rectangle")
        elif choice == "2":
            shape_points = get_demo_shape("rectangle")
        elif choice == "3":
            shape_points = get_demo_shape("triangle")
        elif choice == "4":
            shape_points = get_demo_shape("hexagon")
        else:
            print("Invalid choice. Using demo rectangle.")
            shape_points = get_demo_shape("rectangle")
    else:
        print("\nDemo shapes available:")
        print("1. Rectangle")
        print("2. Triangle")
        print("3. Hexagon")
        choice = input("Select demo shape (1-3): ").strip()

        if choice == "1":
            shape_points = get_demo_shape("rectangle")
        elif choice == "2":
            shape_points = get_demo_shape("triangle")
        elif choice == "3":
            shape_points = get_demo_shape("hexagon")
        else:
            print("Invalid choice. Using rectangle.")
            shape_points = get_demo_shape("rectangle")

    # Get shape center
    try:
        center_x = float(input("Shape center X position (meters): "))
        center_y = float(input("Shape center Y position (meters): "))
        shape_center = (center_x, center_y)
    except ValueError:
        print("Invalid input. Using default center (0.0, 0.0).")
        shape_center = (0.0, 0.0)

    # Get shape rotation
    try:
        shape_rotation_input = input("Shape rotation angle (degrees, default 0): ").strip()
        shape_rotation_deg = float(shape_rotation_input) if shape_rotation_input else 0.0
        shape_rotation_rad = np.radians(shape_rotation_deg)
    except ValueError:
        print("Invalid input. Using default rotation 0°.")
        shape_rotation_deg = 0.0
        shape_rotation_rad = 0.0

    print(f"Shape center: ({shape_center[0]:.2f}, {shape_center[1]:.2f}) m")
    print(f"Shape rotation: {shape_rotation_deg:.1f}°")

    # Get offset distance
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

    # Transform and compute offset path
    transformed_shape = transform_shape(shape_points, shape_center, shape_rotation_rad)
    offset_path = make_offset_path(transformed_shape, offset_distance)
    print(f" Created offset path with {len(offset_path)} points")

    # Store in SatelliteConfig
    SatelliteConfig.DXF_SHAPE_MODE_ACTIVE = True
    SatelliteConfig.DXF_BASE_SHAPE = transformed_shape
    SatelliteConfig.DXF_SHAPE_PATH = offset_path
    setattr(SatelliteConfig, "DXF_SHAPE_CENTER", shape_center)

    print("\n DXF overlay configured successfully!")
    return True


def select_data_file_interactive() -> tuple:
    """Interactive file browser to select a CSV data file."""
    print("\n" + "=" * 60)
    print("   VISUALIZATION DATA FILE SELECTOR")
    print("=" * 60)

    # Scan for available data directories
    data_root = Path("Data")
    if not data_root.exists():
        print(f" Error: Data directory not found at {data_root.absolute()}")
        return None, None

    # Find all CSV files in Data/Simulation and Data/Real_Test
    sim_csvs = (
        list((data_root / "Simulation").rglob("simulation_data.csv"))
        if (data_root / "Simulation").exists()
        else []
    )
    real_csvs = (
        list((data_root / "Real_Test").rglob("real_test_data.csv"))
        if (data_root / "Real_Test").exists()
        else []
    )

    all_csvs = []

    # Add simulation data
    for csv_path in sorted(sim_csvs, key=lambda p: p.stat().st_mtime, reverse=True):
        timestamp_dir = csv_path.parent.name
        all_csvs.append(("Simulation", timestamp_dir, csv_path))

    # Add real data
    for csv_path in sorted(real_csvs, key=lambda p: p.stat().st_mtime, reverse=True):
        timestamp_dir = csv_path.parent.name
        all_csvs.append(("Real", timestamp_dir, csv_path))

    if not all_csvs:
        print(" No data files found in Data/Simulation or Data/Real")
        return None, None

    # Display available files
    print(f"\nFound {len(all_csvs)} data file(s):\n")
    for idx, (mode, timestamp, csv_path) in enumerate(all_csvs, 1):
        file_size = csv_path.stat().st_size / 1024  # KB
        mod_time = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{idx:2d}. [{mode:10s}] {timestamp} ({file_size:.1f} KB) - {mod_time}")

    # Get user selection
    print("\n" + "-" * 60)
    while True:
        try:
            choice = input(f"Select file (1-{len(all_csvs)}) or 'q' to quit: ").strip()
            if choice.lower() == "q":
                print("Cancelled.")
                return None, None

            idx = int(choice) - 1
            if 0 <= idx < len(all_csvs):
                mode, timestamp, csv_path = all_csvs[idx]
                print(f"\n Selected: {csv_path}")
                return str(csv_path.parent), mode.lower()
            else:
                print(f"Invalid selection. Please enter 1-{len(all_csvs)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None, None


def main() -> int:
    """Main function for standalone usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate MPC visualization")
    parser.add_argument(
        "--mode",
        choices=["simulation", "real"],
        default=None,
        help="Visualization mode (simulation or real)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Data directory (optional, uses defaults based on mode)",
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive data selection")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate only plots, skip animation",
    )
    parser.add_argument(
        "--overlay-dxf",
        action="store_true",
        help="Overlay DXF shape on trajectory (if available in Config)",
    )

    args = parser.parse_args()

    # If no arguments provided, use interactive mode by default
    if args.data_dir is None and args.mode is None and not args.interactive:
        print("No arguments provided - using interactive file selector")
        args.interactive = True

    # Interactive file selection
    if args.interactive or (args.data_dir is None and args.mode is None):
        data_dir, mode = select_data_file_interactive()
        if data_dir is None:
            return 1
        args.data_dir = data_dir
        if args.mode is None:
            args.mode = mode

        # After file selection, ask about DXF overlay (unless already specified
        # via CLI)
        if not args.overlay_dxf:
            overlay_configured = configure_dxf_overlay_interactive()
            if overlay_configured:
                args.overlay_dxf = True

    # Set default data directory based on mode if still not set
    if args.data_dir is None:
        if args.mode == "simulation":
            args.data_dir = "Data/Simulation"
        else:
            args.data_dir = "Data/Real"

    # Default mode if not specified
    if args.mode is None:
        args.mode = "simulation"

    try:
        # Create visualizer
        viz = UnifiedVisualizationGenerator(
            data_directory=args.data_dir,
            interactive=False,
            overlay_dxf=args.overlay_dxf,
        )

        # Generate plots
        viz.generate_performance_plots()

        # Generate animation unless plots-only
        if not args.plots_only:
            viz.generate_animation()

    except Exception as e:
        print(f" Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
