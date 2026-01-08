"""
Python-Native Visualization Module

Provides interactive matplotlib-based visualization without requiring a web server.
All visualization runs in Python with interactive plot windows.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
# Use interactive backend (works on all platforms)
try:
    matplotlib.use('TkAgg')  # Tkinter backend (cross-platform)
except ImportError:
    try:
        matplotlib.use('Qt5Agg')  # Qt backend (if available)
    except ImportError:
        matplotlib.use('MacOSX' if __import__('sys').platform == 'darwin' else 'TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)


def load_simulation_data(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load simulation data from CSV file."""
    csv_path = data_dir / "control_data.csv"
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None


def visualize_trajectory_3d(df: pd.DataFrame, interactive: bool = True, save_path: Optional[Path] = None):
    """Create interactive 3D trajectory plot."""
    if df.empty:
        logger.warning("No data to visualize")
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get position columns (handle both 2D and 3D)
    x_col = 'Current_X' if 'Current_X' in df.columns else 'X'
    y_col = 'Current_Y' if 'Current_Y' in df.columns else 'Y'
    
    if 'Current_Z' in df.columns:
        z_col = 'Current_Z'
        ax.plot(df[x_col], df[y_col], df[z_col], 'b-', linewidth=2, label='Trajectory')
        ax.scatter(df[x_col].iloc[0], df[y_col].iloc[0], df[z_col].iloc[0], 
                  color='green', s=100, marker='o', label='Start')
        ax.scatter(df[x_col].iloc[-1], df[y_col].iloc[-1], df[z_col].iloc[-1], 
                  color='red', s=100, marker='s', label='End')
    else:
        # 2D plot (Z=0)
        z = np.zeros(len(df))
        ax.plot(df[x_col], df[y_col], z, 'b-', linewidth=2, label='Trajectory')
        ax.scatter(df[x_col].iloc[0], df[y_col].iloc[0], 0, 
                  color='green', s=100, marker='o', label='Start')
        ax.scatter(df[x_col].iloc[-1], df[y_col].iloc[-1], 0, 
                  color='red', s=100, marker='s', label='End')
    
    # Plot target if available
    if 'Target_X' in df.columns:
        target_x = df['Target_X'].iloc[-1]
        target_y = df['Target_Y'].iloc[-1]
        target_z = df['Target_Z'].iloc[-1] if 'Target_Z' in df.columns else 0
        ax.scatter([target_x], [target_y], [target_z], 
                  color='orange', s=150, marker='*', label='Target')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Satellite Trajectory (3D)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    
    # Set equal aspect ratio
    max_range = np.array([
        df[x_col].max() - df[x_col].min(),
        df[y_col].max() - df[y_col].min(),
        df['Current_Z'].max() - df['Current_Z'].min() if 'Current_Z' in df.columns else 1
    ]).max() / 2.0
    mid_x = (df[x_col].max() + df[x_col].min()) * 0.5
    mid_y = (df[y_col].max() + df[y_col].min()) * 0.5
    mid_z = (df['Current_Z'].max() + df['Current_Z'].min()) * 0.5 if 'Current_Z' in df.columns else 0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    if 'Current_Z' in df.columns:
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if interactive:
        plt.show(block=False)
    else:
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()


def visualize_telemetry(df: pd.DataFrame, interactive: bool = True, save_path: Optional[Path] = None):
    """Create telemetry plots (position, velocity, orientation)."""
    if df.empty:
        logger.warning("No data to visualize")
        return
    
    if 'Control_Time' not in df.columns:
        logger.warning("No time column found")
        return
    
    time = df['Control_Time']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Satellite Telemetry', fontsize=16, fontweight='bold')
    
    # Position
    if 'Current_X' in df.columns:
        axes[0, 0].plot(time, df['Current_X'], label='X', linewidth=2)
        axes[0, 0].plot(time, df['Current_Y'], label='Y', linewidth=2)
        if 'Current_Z' in df.columns:
            axes[0, 0].plot(time, df['Current_Z'], label='Z', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].set_title('Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Velocity
    if 'Velocity_X' in df.columns:
        axes[0, 1].plot(time, df['Velocity_X'], label='Vx', linewidth=2)
        axes[0, 1].plot(time, df['Velocity_Y'], label='Vy', linewidth=2)
        if 'Velocity_Z' in df.columns:
            axes[0, 1].plot(time, df['Velocity_Z'], label='Vz', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Orientation (Quaternion)
    if 'Quaternion_W' in df.columns:
        axes[1, 0].plot(time, df['Quaternion_W'], label='qw', linewidth=2)
        axes[1, 0].plot(time, df['Quaternion_X'], label='qx', linewidth=2)
        axes[1, 0].plot(time, df['Quaternion_Y'], label='qy', linewidth=2)
        axes[1, 0].plot(time, df['Quaternion_Z'], label='qz', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Quaternion')
        axes[1, 0].set_title('Orientation (Quaternion)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Angular Velocity
    if 'Angular_Velocity_X' in df.columns:
        axes[1, 1].plot(time, df['Angular_Velocity_X'], label='ωx', linewidth=2)
        axes[1, 1].plot(time, df['Angular_Velocity_Y'], label='ωy', linewidth=2)
        axes[1, 1].plot(time, df['Angular_Velocity_Z'], label='ωz', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
        axes[1, 1].set_title('Angular Velocity')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Position Error
    if 'Position_Error' in df.columns:
        axes[2, 0].plot(time, df['Position_Error'], 'r-', linewidth=2)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Position Error (m)')
        axes[2, 0].set_title('Position Tracking Error')
        axes[2, 0].grid(True)
    
    # MPC Solve Time
    if 'MPC_Solve_Time' in df.columns:
        axes[2, 1].plot(time, df['MPC_Solve_Time'] * 1000, 'g-', linewidth=2)  # Convert to ms
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Solve Time (ms)')
        axes[2, 1].set_title('MPC Solver Performance')
        axes[2, 1].grid(True)
    
    plt.tight_layout()
    
    if interactive:
        plt.show(block=False)
    else:
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()


def visualize_performance(df: pd.DataFrame, interactive: bool = True, save_path: Optional[Path] = None):
    """Create performance analysis plots."""
    if df.empty:
        logger.warning("No data to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Analysis', fontsize=16, fontweight='bold')
    
    time = df['Control_Time'] if 'Control_Time' in df.columns else np.arange(len(df))
    
    # Position error over time
    if 'Position_Error' in df.columns:
        axes[0, 0].plot(time, df['Position_Error'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Error (m)')
        axes[0, 0].set_title('Position Tracking Error')
        axes[0, 0].grid(True)
        if len(df) > 0:
            axes[0, 0].axhline(y=df['Position_Error'].mean(), color='r', 
                              linestyle='--', label=f'Mean: {df["Position_Error"].mean():.4f}m')
            axes[0, 0].legend()
    
    # MPC solve time distribution
    if 'MPC_Solve_Time' in df.columns:
        solve_times_ms = df['MPC_Solve_Time'].dropna() * 1000  # Convert to ms
        axes[0, 1].hist(solve_times_ms, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Solve Time (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('MPC Solver Time Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        if len(solve_times_ms) > 0:
            axes[0, 1].axvline(x=solve_times_ms.mean(), color='r', 
                              linestyle='--', label=f'Mean: {solve_times_ms.mean():.2f}ms')
            axes[0, 1].legend()
    
    # Thruster usage (if available)
    thruster_cols = [col for col in df.columns if col.startswith('Thruster_')]
    if thruster_cols:
        thruster_usage = df[thruster_cols].abs().sum(axis=1)
        axes[1, 0].plot(time, thruster_usage, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Total Thrust (N)')
        axes[1, 0].set_title('Total Thruster Usage')
        axes[1, 0].grid(True)
    
    # Energy consumption (if available)
    if 'Energy_Consumption' in df.columns:
        axes[1, 1].plot(time, df['Energy_Consumption'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Energy (J)')
        axes[1, 1].set_title('Cumulative Energy Consumption')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if interactive:
        plt.show(block=False)
    else:
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()


def visualize_simulation_data(data_dir: Path, interactive: bool = True):
    """
    Main visualization function - creates all plots.
    
    Args:
        data_dir: Path to simulation data directory
        interactive: If True, opens interactive windows; if False, saves static images
    """
    from rich.console import Console
    
    console = Console()
    console.print(f"[bold blue]Loading simulation data from:[/bold blue] {data_dir}")
    
    df = load_simulation_data(data_dir)
    
    if df is None or df.empty:
        console.print("[red]Error: Could not load simulation data[/red]")
        return
    
    console.print(f"[green]✓ Loaded {len(df)} data points[/green]")
    console.print()
    
    if interactive:
        console.print("[bold]Opening interactive plots...[/bold]")
        console.print("[dim]Close plot windows when done viewing[/dim]")
        console.print()
    
    # Create all visualizations
    try:
        save_traj = None if interactive else data_dir / f"{data_dir.name}_trajectory_3d.png"
        save_tel = None if interactive else data_dir / f"{data_dir.name}_telemetry.png"
        save_perf = None if interactive else data_dir / f"{data_dir.name}_performance.png"
        
        visualize_trajectory_3d(df, interactive=interactive, save_path=save_traj)
        visualize_telemetry(df, interactive=interactive, save_path=save_tel)
        visualize_performance(df, interactive=interactive, save_path=save_perf)
        
        if interactive:
            console.print("[green]✓ All plots opened[/green]")
            console.print("[yellow]Press Enter to continue after closing plots...[/yellow]")
            input()  # Wait for user to close plots
        else:
            console.print("[green]✓ All plots saved[/green]")
    
    except Exception as e:
        console.print(f"[red]Error creating visualizations: {e}[/red]")
        logger.exception("Visualization error")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
        visualize_simulation_data(data_path, interactive=True)
    else:
        print("Usage: python -m src.satellite_control.visualize <data_directory>")
