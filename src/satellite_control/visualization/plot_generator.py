"""
Plot Generator for Visualization

Provides plotting functionality for performance analysis plots.
Extracted from unified_visualizer.py to improve modularity.

This module handles all static plot generation (not animation frames).
"""

from pathlib import Path
from typing import Any, Dict, List
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from typing import Optional

from src.satellite_control.config.models import AppConfig
from src.satellite_control.visualization.unified_visualizer import PlotStyle


class PlotGenerator:
    """
    Generates performance analysis plots from simulation data.
    
    This class handles all static plot generation, separating plotting
    logic from data management and animation generation.
    """
    
    def __init__(
        self,
        data_accessor: Any,
        dt: float,
        system_title: str = "Satellite Control System",
        app_config: Optional[AppConfig] = None,
    ):
        """
        Initialize plot generator.
        
        Args:
            data_accessor: Object with data access methods (_col, _row, _get_len)
            dt: Simulation timestep in seconds
            system_title: Title for plots
            app_config: Optional AppConfig for accessing configuration (v3.0.0)
        """
        self.data_accessor = data_accessor
        self.dt = dt
        self.system_title = system_title
        self.app_config = app_config
    
    def _col(self, name: str) -> np.ndarray:
        """Get column data from data accessor."""
        return self.data_accessor._col(name)
    
    def _row(self, idx: int) -> Dict[str, Any]:
        """Get row data from data accessor."""
        return self.data_accessor._row(idx)
    
    def _get_len(self) -> int:
        """Get data length from data accessor."""
        return self.data_accessor._get_len()
    
    def generate_all_plots(self, plot_dir: Path) -> None:
        """
        Generate all performance analysis plots.
        
        Args:
            plot_dir: Directory to save plots
        """
        print("Generating performance analysis plots...")
        plot_dir.mkdir(exist_ok=True)
        print(f" Created Plots directory: {plot_dir}")
        
        # Generate specific performance plots
        self.generate_position_tracking_plot(plot_dir)
        self.generate_position_error_plot(plot_dir)
        self.generate_angular_tracking_plot(plot_dir)
        self.generate_angular_error_plot(plot_dir)
        self.generate_trajectory_plot(plot_dir)
        self.generate_trajectory_3d_interactive_plot(plot_dir)
        self.generate_thruster_usage_plot(plot_dir)
        self.generate_thruster_valve_activity_plot(plot_dir)
        self.generate_pwm_quantization_plot(plot_dir)
        self.generate_control_effort_plot(plot_dir)
        self.generate_velocity_tracking_plot(plot_dir)
        self.generate_velocity_magnitude_plot(plot_dir)
        self.generate_mpc_performance_plot(plot_dir)
        self.generate_timing_intervals_plot(plot_dir)
        
        print(f"Performance plots saved to: {plot_dir}")
    
    # Plot generation methods will be moved here from UnifiedVisualizationGenerator
    # For now, these are placeholder methods that delegate to the original class
    # They will be fully implemented in subsequent steps
    
    def generate_position_tracking_plot(self, plot_dir: Path) -> None:
        """Generate position tracking over time plot."""
        fig, axes = plt.subplots(3, 1, figsize=PlotStyle.FIGSIZE_SUBPLOTS)
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
        
        # Z position tracking
        axes[2].plot(
            time,
            self._col("Current_Z"),
            color=PlotStyle.COLOR_SIGNAL_POS,
            linewidth=PlotStyle.LINEWIDTH,
            label="Current Z",
        )
        axes[2].plot(
            time,
            self._col("Target_Z"),
            color=PlotStyle.COLOR_TARGET,
            linestyle="--",
            linewidth=PlotStyle.LINEWIDTH,
            label="Target Z",
        )
        axes[2].set_xlabel("Time (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[2].set_ylabel("Z Position (m)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[2].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[2].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[2].set_title("Z Position Tracking")
        
        PlotStyle.save_figure(fig, plot_dir / "position_tracking.png")
    
    def generate_position_error_plot(self, plot_dir: Path) -> None:
        """Generate position error plot."""
        fig, axes = plt.subplots(3, 1, figsize=PlotStyle.FIGSIZE_SUBPLOTS)
        fig.suptitle(f"Position Error - {self.system_title}")
        
        time = np.arange(self._get_len()) * float(self.dt)
        
        # Calculate errors
        error_x = self._col("Current_X") - self._col("Target_X")
        error_y = self._col("Current_Y") - self._col("Target_Y")
        error_z = self._col("Current_Z") - self._col("Target_Z")
        
        # X error
        axes[0].plot(
            time,
            error_x,
            color=PlotStyle.COLOR_ERROR,
            linewidth=PlotStyle.LINEWIDTH,
            label="X Error",
        )
        axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        axes[0].set_ylabel("X Error (m)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[0].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[0].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[0].set_title("X Position Error")
        
        # Y error
        axes[1].plot(
            time,
            error_y,
            color=PlotStyle.COLOR_ERROR,
            linewidth=PlotStyle.LINEWIDTH,
            label="Y Error",
        )
        axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        axes[1].set_ylabel("Y Error (m)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[1].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[1].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[1].set_title("Y Position Error")
        
        # Z error
        axes[2].plot(
            time,
            error_z,
            color=PlotStyle.COLOR_ERROR,
            linewidth=PlotStyle.LINEWIDTH,
            label="Z Error",
        )
        axes[2].axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        axes[2].set_xlabel("Time (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[2].set_ylabel("Z Error (m)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[2].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[2].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[2].set_title("Z Position Error")
        
        PlotStyle.save_figure(fig, plot_dir / "position_error.png")
    
    def generate_angular_tracking_plot(self, plot_dir: Path) -> None:
        """Generate angular tracking plot."""
        fig, axes = plt.subplots(3, 1, figsize=PlotStyle.FIGSIZE_SUBPLOTS)
        fig.suptitle(f"Angular Tracking - {self.system_title}")
        
        time = np.arange(self._get_len()) * float(self.dt)
        
        # Roll tracking
        axes[0].plot(
            time,
            np.degrees(self._col("Current_Roll")),
            color=PlotStyle.COLOR_SIGNAL_ANG,
            linewidth=PlotStyle.LINEWIDTH,
            label="Current Roll",
        )
        axes[0].plot(
            time,
            np.degrees(self._col("Target_Roll")),
            color=PlotStyle.COLOR_TARGET,
            linestyle="--",
            linewidth=PlotStyle.LINEWIDTH,
            label="Target Roll",
        )
        axes[0].set_ylabel("Roll (deg)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[0].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[0].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[0].set_title("Roll Tracking")
        
        # Pitch tracking
        axes[1].plot(
            time,
            np.degrees(self._col("Current_Pitch")),
            color=PlotStyle.COLOR_SIGNAL_ANG,
            linewidth=PlotStyle.LINEWIDTH,
            label="Current Pitch",
        )
        axes[1].plot(
            time,
            np.degrees(self._col("Target_Pitch")),
            color=PlotStyle.COLOR_TARGET,
            linestyle="--",
            linewidth=PlotStyle.LINEWIDTH,
            label="Target Pitch",
        )
        axes[1].set_ylabel("Pitch (deg)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[1].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[1].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[1].set_title("Pitch Tracking")
        
        # Yaw tracking
        axes[2].plot(
            time,
            np.degrees(self._col("Current_Yaw")),
            color=PlotStyle.COLOR_SIGNAL_ANG,
            linewidth=PlotStyle.LINEWIDTH,
            label="Current Yaw",
        )
        axes[2].plot(
            time,
            np.degrees(self._col("Target_Yaw")),
            color=PlotStyle.COLOR_TARGET,
            linestyle="--",
            linewidth=PlotStyle.LINEWIDTH,
            label="Target Yaw",
        )
        axes[2].set_xlabel("Time (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[2].set_ylabel("Yaw (deg)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[2].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[2].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[2].set_title("Yaw Tracking")
        
        PlotStyle.save_figure(fig, plot_dir / "angular_tracking.png")
    
    def generate_angular_error_plot(self, plot_dir: Path) -> None:
        """Generate angular error plot."""
        fig, axes = plt.subplots(3, 1, figsize=PlotStyle.FIGSIZE_SUBPLOTS)
        fig.suptitle(f"Angular Error - {self.system_title}")
        
        time = np.arange(self._get_len()) * float(self.dt)
        
        # Calculate errors
        error_roll = self._col("Current_Roll") - self._col("Target_Roll")
        error_pitch = self._col("Current_Pitch") - self._col("Target_Pitch")
        error_yaw = self._col("Current_Yaw") - self._col("Target_Yaw")
        
        # Normalize angles to [-180, 180] degrees
        error_roll = np.degrees(np.arctan2(np.sin(error_roll), np.cos(error_roll)))
        error_pitch = np.degrees(np.arctan2(np.sin(error_pitch), np.cos(error_pitch)))
        error_yaw = np.degrees(np.arctan2(np.sin(error_yaw), np.cos(error_yaw)))
        
        # Roll error
        axes[0].plot(
            time,
            error_roll,
            color=PlotStyle.COLOR_ERROR,
            linewidth=PlotStyle.LINEWIDTH,
            label="Roll Error",
        )
        axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        axes[0].set_ylabel("Roll Error (deg)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[0].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[0].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[0].set_title("Roll Error")
        
        # Pitch error
        axes[1].plot(
            time,
            error_pitch,
            color=PlotStyle.COLOR_ERROR,
            linewidth=PlotStyle.LINEWIDTH,
            label="Pitch Error",
        )
        axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        axes[1].set_ylabel("Pitch Error (deg)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[1].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[1].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[1].set_title("Pitch Error")
        
        # Yaw error
        axes[2].plot(
            time,
            error_yaw,
            color=PlotStyle.COLOR_ERROR,
            linewidth=PlotStyle.LINEWIDTH,
            label="Yaw Error",
        )
        axes[2].axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        axes[2].set_xlabel("Time (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[2].set_ylabel("Yaw Error (deg)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        axes[2].grid(True, alpha=PlotStyle.GRID_ALPHA)
        axes[2].legend(fontsize=PlotStyle.LEGEND_SIZE)
        axes[2].set_title("Yaw Error")
        
        PlotStyle.save_figure(fig, plot_dir / "angular_error.png")
    
    def generate_trajectory_plot(self, plot_dir: Path) -> None:
        """Generate trajectory plot."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        x_pos = self._col("Current_X")
        y_pos = self._col("Current_Y")
        z_pos = self._col("Current_Z")
        target_x_col = self._col("Target_X")
        target_y_col = self._col("Target_Y")
        target_z_col = self._col("Target_Z")
        target_x = target_x_col[0] if len(target_x_col) > 0 else 0.0
        target_y = target_y_col[0] if len(target_y_col) > 0 else 0.0
        target_z = target_z_col[0] if len(target_z_col) > 0 else 0.0
        
        # Plot trajectory X-Y
        ax_xy = axes[0]
        ax_xy.plot(
            x_pos,
            y_pos,
            color=PlotStyle.COLOR_SIGNAL_POS,
            linewidth=PlotStyle.LINEWIDTH_THICK,
            alpha=0.8,
            label="Satellite Path",
        )
        if len(x_pos) > 0:
            ax_xy.plot(
                x_pos[0],
                y_pos[0],
                "o",
                color=PlotStyle.COLOR_SUCCESS,
                markersize=PlotStyle.MARKER_SIZE,
                label="Start Position",
            )
            ax_xy.plot(
                x_pos[-1],
                y_pos[-1],
                "o",
                color=PlotStyle.COLOR_ERROR,
                markersize=PlotStyle.MARKER_SIZE,
                label="Final Position",
            )
        ax_xy.plot(target_x, target_y, "r*", markersize=20, label="Target")
        
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
        ax_xy.add_patch(circle)
        
        ax_xy.set_xlim(-3, 3)
        ax_xy.set_ylim(-3, 3)
        ax_xy.set_xlabel("X Position (meters)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax_xy.set_ylabel("Y Position (meters)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax_xy.set_title("Trajectory (X-Y)")
        ax_xy.grid(True, alpha=PlotStyle.GRID_ALPHA)
        ax_xy.legend(fontsize=PlotStyle.LEGEND_SIZE)
        ax_xy.set_aspect("equal")
        
        # Plot trajectory X-Z
        ax_xz = axes[1]
        ax_xz.plot(
            x_pos,
            z_pos,
            color=PlotStyle.COLOR_SIGNAL_POS,
            linewidth=PlotStyle.LINEWIDTH_THICK,
            alpha=0.8,
            label="Satellite Path",
        )
        if len(x_pos) > 0 and len(z_pos) > 0:
            ax_xz.plot(
                x_pos[0],
                z_pos[0],
                "o",
                color=PlotStyle.COLOR_SUCCESS,
                markersize=PlotStyle.MARKER_SIZE,
                label="Start Position",
            )
            ax_xz.plot(
                x_pos[-1],
                z_pos[-1],
                "o",
                color=PlotStyle.COLOR_ERROR,
                markersize=PlotStyle.MARKER_SIZE,
                label="Final Position",
            )
        ax_xz.plot(target_x, target_z, "r*", markersize=20, label="Target")
        
        circle_xz = Circle(
            (target_x, target_z),
            0.1,
            color=PlotStyle.COLOR_TARGET,
            fill=False,
            linewidth=PlotStyle.LINEWIDTH,
            linestyle="--",
            alpha=0.7,
            label="Target Zone (±0.1m)",
        )
        ax_xz.add_patch(circle_xz)
        
        ax_xz.set_xlim(-3, 3)
        ax_xz.set_ylim(-3, 3)
        ax_xz.set_xlabel("X Position (meters)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax_xz.set_ylabel("Z Position (meters)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax_xz.set_title("Trajectory (X-Z)")
        ax_xz.grid(True, alpha=PlotStyle.GRID_ALPHA)
        ax_xz.legend(fontsize=PlotStyle.LEGEND_SIZE)
        ax_xz.set_aspect("equal")
        
        # Add distance info
        if len(x_pos) > 0 and len(y_pos) > 0 and len(z_pos) > 0:
            final_distance = np.sqrt(
                (x_pos[-1] - target_x) ** 2
                + (y_pos[-1] - target_y) ** 2
                + (z_pos[-1] - target_z) ** 2
            )
        else:
            final_distance = 0.0
        ax_xy.text(
            0.02,
            0.98,
            f"Final Distance to Target: {final_distance:.3f}m",
            transform=ax_xy.transAxes,
            fontsize=PlotStyle.ANNOTATION_SIZE,
            verticalalignment="top",
            bbox=PlotStyle.TEXTBOX_STYLE,
        )
        
        PlotStyle.save_figure(fig, plot_dir / "trajectory.png")
    
    def generate_trajectory_3d_interactive_plot(self, plot_dir: Path) -> None:
        """Generate interactive 3D trajectory plot (HTML)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            import sys
            print(
                "Plotly not installed; skipping interactive 3D trajectory plot.",
                file=sys.stderr,
            )
            return

        x_pos = self._col("Current_X")
        y_pos = self._col("Current_Y")
        z_pos = self._col("Current_Z")
        if len(x_pos) == 0 or len(y_pos) == 0 or len(z_pos) == 0:
            print("No trajectory data available for interactive 3D plot.")
            return

        target_x_col = self._col("Target_X")
        target_y_col = self._col("Target_Y")
        target_z_col = self._col("Target_Z")
        target_x = float(target_x_col[0]) if len(target_x_col) > 0 else 0.0
        target_y = float(target_y_col[0]) if len(target_y_col) > 0 else 0.0
        target_z = float(target_z_col[0]) if len(target_z_col) > 0 else 0.0

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=x_pos,
                y=y_pos,
                z=z_pos,
                mode="lines",
                line=dict(color=PlotStyle.COLOR_SIGNAL_POS, width=4),
                name="Trajectory",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[x_pos[0]],
                y=[y_pos[0]],
                z=[z_pos[0]],
                mode="markers",
                marker=dict(size=5, color=PlotStyle.COLOR_SUCCESS),
                name="Start",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[x_pos[-1]],
                y=[y_pos[-1]],
                z=[z_pos[-1]],
                mode="markers",
                marker=dict(size=5, color=PlotStyle.COLOR_ERROR),
                name="Final",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[target_x],
                y=[target_y],
                z=[target_z],
                mode="markers",
                marker=dict(size=6, color=PlotStyle.COLOR_TARGET, symbol="x"),
                name="Target",
            )
        )

        fig.update_layout(
            title=f"Interactive 3D Trajectory - {self.system_title}",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
            ),
            legend=dict(itemsizing="constant"),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        output_path = plot_dir / "trajectory_3d_interactive.html"
        fig.write_html(output_path, include_plotlyjs="cdn")
    
    def generate_thruster_usage_plot(self, plot_dir: Path) -> None:
        """Generate thruster usage plot using actual valve states."""
        fig, ax = plt.subplots(1, 1, figsize=PlotStyle.FIGSIZE_SUBPLOTS)

        thruster_count = self._get_thruster_count()
        thruster_ids = np.arange(1, thruster_count + 1)
        total_activation_time = np.zeros(thruster_count)
        data_source = "commanded"

        # Try to use actual valve states (Thruster_X_Val) for accurate PWM tracking
        has_valve_data = False
        if (
            hasattr(self.data_accessor, "_data_backend")
            and self.data_accessor._data_backend == "pandas"
        ):
            if (
                hasattr(self.data_accessor, "data")
                and self.data_accessor.data is not None
                and "Thruster_1_Val" in self.data_accessor.data.columns
            ):
                has_valve_data = True
        elif (
            hasattr(self.data_accessor, "_col_data")
            and self.data_accessor._col_data is not None
            and "Thruster_1_Val" in self.data_accessor._col_data
        ):
            has_valve_data = True

        if has_valve_data:
            # Use actual valve states - most accurate for PWM
            data_source = "actual valve"
            for i in range(thruster_count):
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
                cmd_vec = self._parse_command_vector(row["Command_Vector"])
                command_data.append(cmd_vec)
            command_matrix = np.array(command_data)
            if command_matrix.ndim == 2:
                if command_matrix.shape[1] >= thruster_count:
                    command_matrix = command_matrix[:, :thruster_count]
                else:
                    pad = np.zeros((command_matrix.shape[0], thruster_count))
                    pad[:, : command_matrix.shape[1]] = command_matrix
                    command_matrix = pad
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
        ax.grid(True, axis="y", alpha=PlotStyle.GRID_ALPHA)
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
    
    def generate_thruster_valve_activity_plot(self, plot_dir: Path) -> None:
        """Generate detailed valve activity plot for each thruster (0.0 to 1.0)."""
        # Check if we have valve data
        has_valve_data = False
        if (
            hasattr(self.data_accessor, "_data_backend")
            and self.data_accessor._data_backend == "pandas"
        ):
            if (
                hasattr(self.data_accessor, "data")
                and self.data_accessor.data is not None
                and "Thruster_1_Val" in self.data_accessor.data.columns
            ):
                has_valve_data = True
        elif (
            hasattr(self.data_accessor, "_col_data")
            and self.data_accessor._col_data is not None
            and "Thruster_1_Val" in self.data_accessor._col_data
        ):
            has_valve_data = True

        if not has_valve_data:
            return

        thruster_count = self._get_thruster_count()
        cols = 3 if thruster_count > 8 else 2
        rows = int(np.ceil(thruster_count / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
        axes = axes.flatten()

        time = np.arange(self._get_len()) * float(self.dt)

        for i in range(thruster_count):
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

            # Update Combined Plot
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

            if i // cols == rows - 1:
                ax.set_xlabel("Time (s)")

            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

            # Generate Separate Plot for this Thruster (Two subplots)
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
        for idx in range(thruster_count, len(axes)):
            axes[idx].axis("off")
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(
            plot_dir / "thruster_valve_activity.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    
    def generate_pwm_quantization_plot(self, plot_dir: Path) -> None:
        """Generate PWM duty cycle plot showing MPC u-values vs time."""
        # Use control_data which has the actual MPC outputs per control step
        control_df = None
        if (
            hasattr(self.data_accessor, "control_data")
            and self.data_accessor.control_data is not None
        ):
            control_df = self.data_accessor.control_data

        if control_df is None or "Command_Vector" not in control_df.columns:
            # Fallback: check if physics_data has the cmd columns
            return self.generate_pwm_duty_cycles_from_physics(plot_dir)

        # Parse Command_Vector column to get duty cycles per thruster
        time = control_df["Control_Time"].values

        # Parse the command vectors
        thruster_count = self._get_thruster_count()
        duty_cycles_per_thruster: Dict[int, List[float]] = {
            i: [] for i in range(thruster_count)
        }

        for cmd_str in control_df["Command_Vector"]:
            try:
                # Parse "[0.000, 1.000, ...]" format
                values = cmd_str.strip('[]"').split(",")
                for i in range(thruster_count):
                    if i < len(values):
                        duty_cycles_per_thruster[i].append(float(values[i].strip()))
                    else:
                        duty_cycles_per_thruster[i].append(0.0)
            except BaseException:
                for i in range(thruster_count):
                    duty_cycles_per_thruster[i].append(0.0)

        cols = 3 if thruster_count > 8 else 2
        rows = int(np.ceil(thruster_count / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(14, 3.5 * rows), sharex=True, sharey=True
        )
        axes = axes.flatten()
        fig.suptitle(
            f"PWM MPC Duty Cycle Output (u) vs Time - {self.system_title}",
            fontsize=14,
            fontweight="bold",
        )

        colors = plt.cm.tab20(np.linspace(0, 1, thruster_count))

        # Track if we find any intermediate values
        all_intermediate_values = []

        for i, ax in enumerate(axes):
            if i >= thruster_count:
                ax.axis("off")
                continue
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
        for i in range(thruster_count):
            if i // cols == rows - 1:
                axes[i].set_xlabel("Time (s)", fontsize=12)

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

    def generate_pwm_duty_cycles_from_physics(self, plot_dir: Path) -> None:
        """Fallback: Generate PWM plot from physics_data Thruster_X_Cmd columns."""
        # This shows binary valve states, not continuous duty cycles
        pass
    
    def _get_thruster_count(self) -> int:
        """Determine thruster count based on available data or config."""
        cols: list[str] = []
        if (
            hasattr(self.data_accessor, "_data_backend")
            and self.data_accessor._data_backend == "pandas"
        ):
            if hasattr(self.data_accessor, "data") and self.data_accessor.data is not None:
                cols = list(self.data_accessor.data.columns)
        elif (
            hasattr(self.data_accessor, "_col_data")
            and self.data_accessor._col_data is not None
        ):
            cols = list(self.data_accessor._col_data.keys())

        max_id = 0
        if cols:
            for i in range(1, 33):
                if f"Thruster_{i}_Val" in cols or f"Thruster_{i}_Cmd" in cols:
                    max_id = i

        if max_id > 0:
            return max_id

        if (
            hasattr(self.data_accessor, "control_data")
            and self.data_accessor.control_data is not None
        ):
            if (
                "Command_Vector" in self.data_accessor.control_data.columns
                and len(self.data_accessor.control_data) > 0
            ):
                sample_vec = self._parse_command_vector(
                    self.data_accessor.control_data["Command_Vector"].iloc[0]
                )
                if sample_vec.size > 0:
                    return int(sample_vec.size)

        # Fallback to default
        try:
            # Get thruster count from app_config if available, otherwise fallback
            if self.app_config and self.app_config.physics:
                return len(self.app_config.physics.thruster_positions)
            else:
                # V4.0.0: Use default config if app_config not available
                from src.satellite_control.config.simulation_config import SimulationConfig
                default_config = SimulationConfig.create_default()
                return len(default_config.app_config.physics.thruster_positions)
        except Exception:
            return 12

    def _parse_command_vector(self, command_str: Any) -> np.ndarray:
        """Parse command vector string to numpy array.
        
        Args:
            command_str: String representation of command vector (or any type)
        
        Returns:
            numpy array of thruster commands
        """
        try:
            # Handle None or empty
            if command_str is None or command_str == "":
                return np.zeros(12)
            
            # Convert to string if not already
            command_str = str(command_str)
            
            # Remove brackets and split
            command_str = command_str.strip("[]")
            values = [float(x.strip()) for x in command_str.split(",")]
            return np.array(values)
        except Exception:
            return np.zeros(12)
    
    def generate_control_effort_plot(self, plot_dir: Path) -> None:
        """Generate control effort plot."""
        fig, ax = plt.subplots(1, 1, figsize=PlotStyle.FIGSIZE_SINGLE)
        
        time = np.arange(self._get_len()) * float(self.dt)
        
        # Parse command vectors
        command_data = []
        for idx in range(self._get_len()):
            row = self._row(idx)
            cmd_vec = self._parse_command_vector(row["Command_Vector"])
            command_data.append(cmd_vec)
        command_matrix = np.array(command_data)
        
        total_effort_per_step = np.sum(command_matrix, axis=1)
        ax.plot(
            time,
            total_effort_per_step,
            color="c",
            linewidth=PlotStyle.LINEWIDTH,
            label="Total Control Effort",
        )
        ax.set_xlabel("Time (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_ylabel("Total Control Effort", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_title(f"Control Effort Over Time - {self.system_title}")
        ax.grid(True, alpha=PlotStyle.GRID_ALPHA)
        ax.legend(fontsize=PlotStyle.LEGEND_SIZE)
        
        PlotStyle.save_figure(fig, plot_dir / "control_effort.png")
    
    def generate_velocity_tracking_plot(self, plot_dir: Path) -> None:
        """Generate velocity tracking over time plot."""
        fig, axes = plt.subplots(3, 1, figsize=PlotStyle.FIGSIZE_SUBPLOTS)
        fig.suptitle(f"Velocity Tracking - {self.system_title}")
        
        time = np.arange(self._get_len()) * float(self.dt)
        
        def plot_velocity(ax, axis_label, current_col, target_col):
            current_vals = self._col(current_col)
            min_len = min(len(time), len(current_vals))
            if min_len == 0:
                return
            ax.plot(
                time[:min_len],
                current_vals[:min_len],
                color=PlotStyle.COLOR_SIGNAL_POS,
                linewidth=PlotStyle.LINEWIDTH,
                label=f"Current {axis_label}",
            )
            target_vals = self._col(target_col)
            if len(target_vals) > 0:
                tgt_len = min(len(time), len(target_vals))
                ax.plot(
                    time[:tgt_len],
                    target_vals[:tgt_len],
                    color=PlotStyle.COLOR_TARGET,
                    linestyle="--",
                    linewidth=PlotStyle.LINEWIDTH,
                    label=f"Target {axis_label}",
                )
            ax.set_ylabel(f"{axis_label} Velocity (m/s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
            ax.grid(True, alpha=PlotStyle.GRID_ALPHA)
            ax.legend(fontsize=PlotStyle.LEGEND_SIZE)
        
        plot_velocity(axes[0], "X", "Current_VX", "Target_VX")
        plot_velocity(axes[1], "Y", "Current_VY", "Target_VY")
        plot_velocity(axes[2], "Z", "Current_VZ", "Target_VZ")
        axes[2].set_xlabel("Time (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        
        PlotStyle.save_figure(fig, plot_dir / "velocity_tracking.png")
    
    def generate_velocity_magnitude_plot(self, plot_dir: Path) -> None:
        """Generate velocity magnitude over time plot (speed vs time)."""
        fig, ax = plt.subplots(1, 1, figsize=PlotStyle.FIGSIZE_SINGLE)
        
        n = self._get_len()
        if n < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for velocity magnitude plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            PlotStyle.save_figure(fig, plot_dir / "velocity_magnitude.png")
            return
        
        time = np.arange(n) * float(self.dt)
        
        # Calculate velocity magnitude
        vx = self._col("Current_VX")
        vy = self._col("Current_VY")
        vz = self._col("Current_VZ")
        
        min_len = min(len(time), len(vx), len(vy), len(vz))
        if min_len == 0:
            return
        
        velocity_magnitude = np.sqrt(
            vx[:min_len] ** 2 + vy[:min_len] ** 2 + vz[:min_len] ** 2
        )
        
        ax.plot(
            time[:min_len],
            velocity_magnitude,
            color=PlotStyle.COLOR_SIGNAL_POS,
            linewidth=PlotStyle.LINEWIDTH,
            label="Velocity Magnitude",
        )
        ax.set_xlabel("Time (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_ylabel("Speed (m/s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
        ax.set_title(f"Velocity Magnitude Over Time - {self.system_title}")
        ax.grid(True, alpha=PlotStyle.GRID_ALPHA)
        ax.legend(fontsize=PlotStyle.LEGEND_SIZE)
        
        PlotStyle.save_figure(fig, plot_dir / "velocity_magnitude.png")
    
    def generate_mpc_performance_plot(self, plot_dir: Path) -> None:
        """Generate MPC performance plot."""
        fig, ax = plt.subplots(1, 1, figsize=PlotStyle.FIGSIZE_SINGLE)
        
        # Determine data source
        df = None
        cols = []
        
        # Check for sibling control data first (dual-log mode)
        if (
            hasattr(self.data_accessor, "control_data")
            and self.data_accessor.control_data is not None
        ):
            df = self.data_accessor.control_data
            cols = df.columns
        elif (
            hasattr(self.data_accessor, "_data_backend")
            and self.data_accessor._data_backend == "pandas"
        ):
            if (
                hasattr(self.data_accessor, "data")
                and self.data_accessor.data is not None
            ):
                df = self.data_accessor.data
                cols = df.columns
        elif (
            hasattr(self.data_accessor, "_col_data")
            and self.data_accessor._col_data is not None
        ):
            cols = list(self.data_accessor._col_data.keys())
        
        if "MPC_Computation_Time" in cols or "MPC_Solve_Time" in cols:
            # Determine time axis
            if df is not None and "Control_Time" in cols:
                time = df["Control_Time"].values
            elif df is not None and "CONTROL_DT" in cols:
                dt_val = df["CONTROL_DT"].iloc[0]
                time = np.arange(len(df)) * float(dt_val)
            else:
                time = np.arange(len(df) if df is not None else self._get_len()) * float(self.dt)
            
            # Get computation times
            if df is not None:
                raw_comp_times = df.get(
                    "MPC_Solve_Time", df.get("MPC_Computation_Time", [])
                ).values
            else:
                raw_comp_times = (
                    self._col("MPC_Solve_Time")
                    if "MPC_Solve_Time" in cols
                    else self._col("MPC_Computation_Time")
                )
            
            comp_times = []
            for val in raw_comp_times:
                try:
                    comp_times.append(float(val) * 1000)  # Convert to ms
                except (ValueError, TypeError):
                    comp_times.append(0.0)
            comp_times = np.array(comp_times)
            
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
                color="purple",
                linewidth=PlotStyle.LINEWIDTH,
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
            ax.set_xlabel("Time (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
            ax.set_ylabel("Computation Time (ms)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
            ax.set_title(f"MPC Computation Time - {self.system_title}")
            ax.grid(True, alpha=PlotStyle.GRID_ALPHA)
            ax.legend(fontsize=PlotStyle.LEGEND_SIZE)
        else:
            ax.text(
                0.5,
                0.5,
                "MPC Computation Time\nData Not Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=PlotStyle.ANNOTATION_SIZE,
            )
            ax.set_title(f"MPC Computation Time - {self.system_title}")
        
        PlotStyle.save_figure(fig, plot_dir / "mpc_performance.png")
    
    def generate_timing_intervals_plot(self, plot_dir: Path) -> None:
        """Generate timing intervals plot."""
        fig, ax = plt.subplots(1, 1, figsize=PlotStyle.FIGSIZE_SINGLE)
        
        # Check for sibling control data first (dual-log mode)
        df = None
        cols = []
        if (
            hasattr(self.data_accessor, "control_data")
            and self.data_accessor.control_data is not None
        ):
            df = self.data_accessor.control_data
            cols = df.columns
        elif (
            hasattr(self.data_accessor, "_data_backend")
            and self.data_accessor._data_backend == "pandas"
        ):
            if (
                hasattr(self.data_accessor, "data")
                and self.data_accessor.data is not None
            ):
                df = self.data_accessor.data
                cols = df.columns
        elif (
            hasattr(self.data_accessor, "_col_data")
            and self.data_accessor._col_data is not None
        ):
            cols = list(self.data_accessor._col_data.keys())
        
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
                color="orange",
                linewidth=PlotStyle.LINEWIDTH,
                label="Actual Intervals",
            )
            ax.axhline(
                y=target_dt,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Target: {target_dt:.3f}s",
            )
            ax.set_xlabel("Time (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
            ax.set_ylabel("Time Interval (s)", fontsize=PlotStyle.AXIS_LABEL_SIZE)
            ax.set_title(f"Timing Intervals - {self.system_title}")
            ax.grid(True, alpha=PlotStyle.GRID_ALPHA)
            ax.legend(fontsize=PlotStyle.LEGEND_SIZE)
        else:
            ax.text(
                0.5,
                0.5,
                "Timing Interval\nData Not Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=PlotStyle.ANNOTATION_SIZE,
            )
            ax.set_title(f"Timing Intervals - {self.system_title}")
        
        PlotStyle.save_figure(fig, plot_dir / "timing_intervals.png")
