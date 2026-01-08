"""
Interactive Configuration Editor

Provides Rich-based interactive forms for editing satellite and MPC configurations.
Everything runs in Python - no web server required.
"""

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
import questionary

from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.config.io import ConfigIO

logger = logging.getLogger(__name__)
console = Console()


def press_any_key_to_continue():
    """Wait for user to press Enter."""
    try:
        input("\n[dim]Press Enter to continue...[/dim]")
    except (EOFError, KeyboardInterrupt):
        pass


def load_config(path: Optional[Path] = None) -> SimulationConfig:
    """Load configuration from file or create default."""
    if path and path.exists():
        try:
            config = ConfigIO.load(path)
            console.print(f"[green]✓ Loaded configuration from {path}[/green]")
            return config
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {path}: {e}[/yellow]")
            console.print("[yellow]Using default configuration[/yellow]")
    
    return SimulationConfig.create_default()


def save_config(config: SimulationConfig, path: Path):
    """Save configuration to file."""
    try:
        ConfigIO.save(config, path)
        console.print(f"[green]✓ Configuration saved to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving configuration: {e}[/red]")
        raise


def edit_physics_params(config: SimulationConfig) -> SimulationConfig:
    """Edit satellite physical parameters interactively."""
    console.print()
    console.print(Panel("[bold]Satellite Physical Parameters[/bold]", border_style="yellow"))
    console.print()
    
    physics = config.app_config.physics
    
    # Create table showing current values
    table = Table(title="Current Physics Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Mass", f"{physics.total_mass:.3f} kg")
    table.add_row("Moment of Inertia", f"{physics.moment_of_inertia:.6f} kg·m²")
    table.add_row("Satellite Size", f"{physics.satellite_size:.3f} m")
    table.add_row("COM Offset", f"({physics.com_offset[0]:.3f}, {physics.com_offset[1]:.3f}, {physics.com_offset[2]:.3f}) m")
    
    console.print(table)
    console.print()
    
    # Ask if user wants to edit
    if not Confirm.ask("Edit physics parameters?", default=False):
        return config
    
    # Edit mass
    new_mass = Prompt.ask(
        "Total Mass (kg)",
        default=str(physics.total_mass),
        type=float
    )
    
    # Edit inertia
    new_inertia = Prompt.ask(
        "Moment of Inertia (kg·m²)",
        default=str(physics.moment_of_inertia),
        type=float
    )
    
    # Update config using create_with_overrides
    config = SimulationConfig.create_with_overrides(
        {
            "physics": {
                "total_mass": float(new_mass),
                "moment_of_inertia": float(new_inertia),
            }
        },
        base_config=config
    )
    console.print("[green]✓ Physics parameters updated[/green]")
    
    return config


def edit_mpc_params(config: SimulationConfig) -> SimulationConfig:
    """Edit MPC parameters interactively."""
    console.print()
    console.print(Panel("[bold]MPC Control Parameters[/bold]", border_style="magenta"))
    console.print()
    
    mpc = config.app_config.mpc
    
    # Create table showing current values
    table = Table(title="Current MPC Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Prediction Horizon", str(mpc.prediction_horizon))
    table.add_row("Control Horizon", str(mpc.control_horizon))
    table.add_row("Q Position Weight", f"{mpc.q_position:.1f}")
    table.add_row("Q Velocity Weight", f"{mpc.q_velocity:.1f}")
    table.add_row("Q Angle Weight", f"{mpc.q_angle:.1f}")
    table.add_row("R Thrust Penalty", f"{mpc.r_thrust:.1f}")
    table.add_row("Max Velocity", f"{mpc.max_velocity:.2f} m/s")
    table.add_row("Max Angular Velocity", f"{mpc.max_angular_velocity:.2f} rad/s")
    
    console.print(table)
    console.print()
    
    # Ask if user wants to edit
    if not Confirm.ask("Edit MPC parameters?", default=False):
        return config
    
    # Edit key parameters
    new_q_pos = Prompt.ask(
        "Q Position Weight",
        default=str(mpc.q_position),
        type=float
    )
    
    new_q_vel = Prompt.ask(
        "Q Velocity Weight",
        default=str(mpc.q_velocity),
        type=float
    )
    
    new_r_thrust = Prompt.ask(
        "R Thrust Penalty",
        default=str(mpc.r_thrust),
        type=float
    )
    
    # Update config using create_with_overrides
    config = SimulationConfig.create_with_overrides(
        {
            "mpc": {
                "q_position": float(new_q_pos),
                "q_velocity": float(new_q_vel),
                "r_thrust": float(new_r_thrust),
            }
        },
        base_config=config
    )
    console.print("[green]✓ MPC parameters updated[/green]")
    
    return config


def edit_config_interactive(config_path: Optional[Path] = None):
    """
    Interactive configuration editor.
    
    Args:
        config_path: Optional path to configuration file
    """
    console.print()
    console.print(Panel.fit(
        "[bold blue]Configuration Editor[/bold blue]\n"
        "[dim]Edit satellite and MPC parameters interactively[/dim]",
        border_style="blue"
    ))
    console.print()
    
    # Load configuration
    config = load_config(config_path)
    
    while True:
        # Show main menu
        choice = questionary.select(
            "What would you like to edit?",
            choices=[
                "Physics Parameters",
                "MPC Parameters",
                "View Current Configuration",
                "Save Configuration",
                "Load Configuration",
                "Back to Main Menu",
            ],
            style=questionary.Style([
                ('question', 'bold fg:#0066ff'),
                ('pointer', 'fg:#0066ff bold'),
                ('highlighted', 'bg:#0066ff fg:white bold'),
            ])
        ).ask()
        
        if not choice or choice == "Back to Main Menu":
            break
        
        elif choice == "Physics Parameters":
            config = edit_physics_params(config)
        
        elif choice == "MPC Parameters":
            config = edit_mpc_params(config)
        
        elif choice == "View Current Configuration":
            view_config(config)
        
        elif choice == "Save Configuration":
            save_path = Prompt.ask(
                "Save path (or press Enter for default)",
                default="my_config.yaml"
            )
            if save_path:
                save_config(config, Path(save_path))
        
        elif choice == "Load Configuration":
            load_path = Prompt.ask("Configuration file path")
            if load_path:
                config = load_config(Path(load_path))
        
        console.print()
    
    return config


def view_config(config: SimulationConfig):
    """Display current configuration in a formatted table."""
    console.print()
    console.print(Panel("[bold]Current Configuration[/bold]", border_style="cyan"))
    console.print()
    
    # Physics params
    physics = config.app_config.physics
    table_physics = Table(title="Physics Parameters", show_header=True, header_style="bold magenta")
    table_physics.add_column("Parameter", style="cyan")
    table_physics.add_column("Value", style="green")
    
    table_physics.add_row("Total Mass", f"{physics.total_mass:.3f} kg")
    table_physics.add_row("Moment of Inertia", f"{physics.moment_of_inertia:.6f} kg·m²")
    table_physics.add_row("Satellite Size", f"{physics.satellite_size:.3f} m")
    table_physics.add_row("Number of Thrusters", str(len(physics.thruster_positions)))
    
    console.print(table_physics)
    console.print()
    
    # MPC params
    mpc = config.app_config.mpc
    table_mpc = Table(title="MPC Parameters", show_header=True, header_style="bold magenta")
    table_mpc.add_column("Parameter", style="cyan")
    table_mpc.add_column("Value", style="green")
    
    table_mpc.add_row("Prediction Horizon", str(mpc.prediction_horizon))
    table_mpc.add_row("Control Horizon", str(mpc.control_horizon))
    table_mpc.add_row("Q Position", f"{mpc.q_position:.1f}")
    table_mpc.add_row("Q Velocity", f"{mpc.q_velocity:.1f}")
    table_mpc.add_row("Q Angle", f"{mpc.q_angle:.1f}")
    table_mpc.add_row("R Thrust", f"{mpc.r_thrust:.1f}")
    table_mpc.add_row("Max Velocity", f"{mpc.max_velocity:.2f} m/s")
    
    console.print(table_mpc)
    console.print()
    
    press_any_key_to_continue()


if __name__ == "__main__":
    edit_config_interactive()
