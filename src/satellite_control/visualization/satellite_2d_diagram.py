"""
Satellite System Model Visualization

Provides 2D visualization tools for the satellite thruster control
platform.
Creates detailed diagrams of the physical system layout for analysis
and documentation.

Visualization components:
- Satellite body geometry with dimensions
- Eight thruster positions and force vectors
- Air bearing system configuration (3-point support)
- Center of mass calculation and display
- Moment of inertia visualization
- Color-coded force magnitude indicators

Use cases:
- System design verification
- Physical layout documentation
- Hardware configuration validation
- Educational demonstrations
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from typing import Optional

# V4.0.0: SatelliteConfig removed - use AppConfig only
from src.satellite_control.config.models import AppConfig
from src.satellite_control.config.simulation_config import SimulationConfig


def _get_physics_config(app_config: Optional[AppConfig] = None):
    """
    Get physics configuration from app_config (V4.0.0: uses defaults if None).
    
    Args:
        app_config: Optional AppConfig (v4.0.0: uses defaults if None)
    
    Returns:
        Dictionary with physics parameters
    """
    if app_config and app_config.physics:
        return {
            "total_mass": app_config.physics.total_mass,
            "moment_of_inertia": app_config.physics.moment_of_inertia,
            "satellite_size": app_config.physics.satellite_size,
            "thruster_positions": app_config.physics.thruster_positions,
            "thruster_directions": app_config.physics.thruster_directions,
            "thruster_forces": app_config.physics.thruster_forces,
        }
    else:
        # V4.0.0: Use default config if not provided
        default_config = SimulationConfig.create_default()
        physics = default_config.app_config.physics
        return {
            "total_mass": physics.total_mass,
            "moment_of_inertia": physics.moment_of_inertia,
            "satellite_size": physics.satellite_size,
            "thruster_positions": physics.thruster_positions,
            "thruster_directions": physics.thruster_directions,
            "thruster_forces": physics.thruster_forces,
        }


def _get_simulation_config(app_config: Optional[AppConfig] = None):
    """
    Get simulation configuration from app_config (V4.0.0: uses defaults if None).
    
    Args:
        app_config: Optional AppConfig (v4.0.0: uses defaults if None)
    
    Returns:
        Dictionary with simulation parameters
    """
    if app_config and app_config.simulation:
        return {
            "dt": app_config.simulation.dt,
            "control_dt": app_config.simulation.control_dt,
        }
    else:
        # V4.0.0: Use default config if not provided
        default_config = SimulationConfig.create_default()
        sim = default_config.app_config.simulation
        return {
            "dt": sim.dt,
            "control_dt": sim.control_dt,
        }


def _get_mpc_config(app_config: Optional[AppConfig] = None):
    """
    Get MPC configuration from app_config (V4.0.0: uses defaults if None).
    
    Args:
        app_config: Optional AppConfig (v4.0.0: uses defaults if None)
    
    Returns:
        Dictionary with MPC parameters
    """
    if app_config and app_config.mpc:
        return {
            "mpc_dt": app_config.mpc.dt,
            "mpc_time_limit": app_config.mpc.solver_time_limit,
        }
    else:
        # V4.0.0: Use default config if not provided
        default_config = SimulationConfig.create_default()
        mpc = default_config.app_config.mpc
        return {
            "mpc_dt": mpc.dt,
            "mpc_time_limit": mpc.solver_time_limit,
        }


def create_satellite_model(app_config: Optional[AppConfig] = None):
    """
    Generate a 2D visualization of the satellite test platform.

    Creates a model showing the satellite body, thruster positions,
    air bearing system, and center of mass calculations. The visualization
    uses scaling and color coding for components.

    Args:
        app_config: Optional AppConfig for accessing physics parameters (v3.0.0)

    Returns:
        tuple: matplotlib figure and axis objects for further customization
    """

    # Get physics configuration
    physics = _get_physics_config(app_config)
    satellite_size = physics["satellite_size"]
    thrusters = physics["thruster_positions"]
    thruster_forces = physics["thruster_forces"]

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    satellite_rect = patches.Rectangle(
        (-satellite_size / 2, -satellite_size / 2),
        satellite_size,
        satellite_size,
        linewidth=3,
        edgecolor="black",
        facecolor="lightgray",
        alpha=0.7,
    )
    ax.add_patch(satellite_rect)

    # Draw center point
    ax.plot(0, 0, "ko", markersize=8, label="Satellite Center")

    # Draw thrusters
    thruster_colors = plt.cm.tab10(np.linspace(0, 1, 8))  # type: ignore

    for i, (thruster_id, (x, y)) in enumerate(thrusters.items()):
        # Draw thruster position
        ax.plot(
            x,
            y,
            "o",
            color=thruster_colors[i],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Determine which face the thruster is on and calculate
        # perpendicular direction
        if abs(x) > abs(y):  # On left or right face
            if x > 0:  # Right face (thrusters 1, 2)
                thrust_direction = np.array([-1, 0])  # Push left
            else:  # Left face (thrusters 5, 6)
                thrust_direction = np.array([1, 0])  # Push right
        else:  # On top or bottom face
            if y > 0:  # Top face (thrusters 7, 8)
                thrust_direction = np.array([0, -1])  # Push down
            else:  # Bottom face (thrusters 3, 4)
                thrust_direction = np.array([0, 1])  # Push up

        thrust_scale = 0.05  # Scale factor for visualization

        # Draw thrust vector
        ax.arrow(
            x,
            y,
            thrust_direction[0] * thrust_scale,
            thrust_direction[1] * thrust_scale,
            head_width=0.01,
            head_length=0.01,
            fc=thruster_colors[i],
            ec=thruster_colors[i],
            linewidth=2,
        )

        # Label thruster
        offset = 0.02
        ax.annotate(
            f"T{thruster_id}",
            (x + offset, y + offset),
            fontsize=9,
            fontweight="bold",
        )

    # Add coordinate system arrows
    arrow_length = 0.08
    ax.arrow(
        0.2,
        -0.25,
        arrow_length,
        0,
        head_width=0.01,
        head_length=0.01,
        fc="black",
        ec="black",
        linewidth=2,
    )
    ax.arrow(
        0.2,
        -0.25,
        0,
        arrow_length,
        head_width=0.01,
        head_length=0.01,
        fc="black",
        ec="black",
        linewidth=2,
    )
    ax.text(
        0.2 + arrow_length / 2,
        -0.27,
        "X",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.18,
        -0.25 + arrow_length / 2,
        "Y",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    # Add rotation direction indicator
    circle = patches.Arc(
        (0, 0),
        0.1,
        0.1,
        angle=0,
        theta1=0,
        theta2=270,
        linewidth=2,
        color="purple",
    )
    ax.add_patch(circle)
    ax.arrow(
        0.05,
        0,
        0,
        0.02,
        head_width=0.01,
        head_length=0.01,
        fc="purple",
        ec="purple",
        linewidth=2,
    )
    ax.text(0.08, 0.08, "+θ (CCW)", fontsize=10, color="purple", fontweight="bold")

    # Set axis properties
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Satellite 2D Model\nCubic Satellite with Thrusters and Air Bearings",
        fontsize=14,
        fontweight="bold",
    )

    thrust_values = list(thruster_forces.values())
    thrust_min = min(thrust_values)
    thrust_max = max(thrust_values)
    thrust_avg = sum(thrust_values) / len(thrust_values)

    # Create legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Satellite Center",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:blue",
            markersize=10,
            label="Thrusters (8 total)",
        ),
        Line2D(
            [0],
            [0],
            color="tab:blue",
            linewidth=2,
            label=f"Thrust Vectors ({thrust_min:.3f}-{thrust_max:.3f} N)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    # Add technical specifications text box
    specs_text = f"""Technical Specifications:
• Satellite: Cubic shape, 2D motion
• Thrusters: 8 binary solenoids
• Thrust Force: {thrust_min:.3f}-{thrust_max:.3f} N (avg {thrust_avg:.3f} N)"""

    ax.text(
        -0.29,
        0.29,
        specs_text,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    return fig, ax


def print_thruster_info(app_config: Optional[AppConfig] = None):
    """
    Print detailed thruster information.
    
    Args:
        app_config: Optional AppConfig for accessing physics parameters (v3.0.0)
    """
    physics = _get_physics_config(app_config)
    thrusters = physics["thruster_positions"]
    thruster_forces = physics["thruster_forces"]
    thrust_values = list(thruster_forces.values())
    thrust_min = min(thrust_values)
    thrust_max = max(thrust_values)
    thrust_avg = sum(thrust_values) / len(thrust_values)
    total_thrust = sum(thrust_values)

    print("=" * 50)
    print("THRUSTER CONFIGURATION")
    print("=" * 50)
    print(f"Individual Thruster Forces: {thrust_min:.6f} - {thrust_max:.6f} N")
    print(f"Average Thruster Force: {thrust_avg:.6f} N")
    print(f"Total Available Thrust: {total_thrust:.6f} N")
    print("\nThruster Positions and Forces:")

    for thruster_id, (x, y) in thrusters.items():
        if abs(x) > abs(y):  # On left or right face
            if x > 0:  # Right face (thrusters 1, 2)
                thrust_dir_x, thrust_dir_y = -1, 0  # Push left
                face = "Right"
            else:  # Left face (thrusters 5, 6)
                thrust_dir_x, thrust_dir_y = 1, 0  # Push right
                face = "Left"
        else:  # On top or bottom face
            if y > 0:  # Top face (thrusters 7, 8)
                thrust_dir_x, thrust_dir_y = 0, -1  # Push down
                face = "Top"
            else:  # Bottom face (thrusters 3, 4)
                thrust_dir_x, thrust_dir_y = 0, 1  # Push up
                face = "Bottom"

        # Get individual thruster force
        thruster_force = thruster_forces[thruster_id]

        print(
            f"Thruster {thruster_id}: Position ({x:7.3f}, {y:7.3f}) m, "
            f"Face: {face:6s}, Force: {thruster_force:.3f} N, "
            f"Direction ({thrust_dir_x:4.0f}, {thrust_dir_y:4.0f})"
        )


def print_bearing_info():
    """Print detailed air bearing information"""
    print("\n" + "=" * 50)
    print("AIR BEARING CONFIGURATION")
    print("=" * 50)
    print("Air bearing usage disabled in this configuration.")


if __name__ == "__main__":
    # Create and display the model
    fig, ax = create_satellite_model()

    # Print technical information
    print_thruster_info()
    print_bearing_info()

    # Show the plot
    plt.show()
