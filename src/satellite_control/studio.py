"""
Interactive Studio Mode for Satellite Control System

Provides a centralized, interactive menu system for all satellite control operations.
Everything runs in Python - no web server required.
"""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import questionary

# CLI run function will be called via subprocess or direct import
from src.satellite_control.visualize import visualize_simulation_data
from src.satellite_control.config_editor import edit_config_interactive
from src.satellite_control.mission.plugin import get_registry

console = Console()


def press_any_key_to_continue():
    """Wait for user to press Enter."""
    try:
        input("\n[dim]Press Enter to continue...[/dim]")
    except (EOFError, KeyboardInterrupt):
        pass


def show_main_menu() -> str:
    """Display main interactive menu and return user choice."""
    console.print()
    console.print(Panel.fit(
        "[bold blue]🛰️  Satellite Control Studio[/bold blue]\n"
        "[dim]Complete satellite control and mission simulation[/dim]",
        border_style="blue"
    ))
    console.print()
    
    choice = questionary.select(
        "What would you like to do?",
        choices=[
            questionary.Choice("Run Simulation", value="run"),
            questionary.Choice("Analyze Previous Run", value="analyze"),
            questionary.Choice("Configure Satellite", value="configure"),
            questionary.Choice("Tune MPC Parameters", value="tune"),
            questionary.Choice("Mission Planning", value="mission"),
            questionary.Choice("View Visualizations", value="visualize"),
            questionary.Choice("Export Data", value="export"),
            questionary.Choice("Quit", value="quit"),
        ],
        style=questionary.Style([
            ('question', 'bold fg:#0066ff'),
            ('pointer', 'fg:#0066ff bold'),
            ('highlighted', 'bg:#0066ff fg:white bold'),
        ])
    ).ask()
    
    return choice or "quit"


def run_simulation_menu():
    """Interactive simulation runner."""
    console.print()
    console.print(Panel("[bold]Run Simulation[/bold]", border_style="green"))
    console.print()
    
    # Ask for preset or custom
    use_preset = questionary.confirm("Use a preset configuration?", default=True).ask()
    
    if use_preset:
        from src.satellite_control.config.presets import list_presets, get_preset_description
        
        presets = list_presets()
        preset_choices = []
        for preset in presets:
            desc = get_preset_description(preset)
            preset_choices.append(
                questionary.Choice(f"{preset.capitalize()}: {desc}", value=preset)
            )
        
        selected_preset = questionary.select(
            "Select preset:",
            choices=preset_choices
        ).ask()
        
        if selected_preset:
            console.print(f"[green]Selected preset: {selected_preset}[/green]")
            console.print()
            console.print("[yellow]To run simulation with this preset, use:[/yellow]")
            console.print(f"  [bold]satellite-control run --preset {selected_preset}[/bold]")
            console.print()
    
    # For interactive simulation, users can use CLI
    console.print("[dim]Note: Run simulations using the CLI command:[/dim]")
    console.print("  [bold]satellite-control run[/bold]")
    console.print("  [bold]satellite-control run --preset <preset_name>[/bold]")
    console.print()
    press_any_key_to_continue()


def analyze_results_menu():
    """Analyze previous simulation results."""
    console.print()
    console.print(Panel("[bold]Analyze Previous Run[/bold]", border_style="cyan"))
    console.print()
    
    # Find simulation directories
    data_dir = Path("Data/Simulation")
    if not data_dir.exists():
        console.print("[red]No simulation data found. Run a simulation first.[/red]")
        press_any_key_to_continue()
        return
    
    sim_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, 
                     reverse=True)
    
    if not sim_dirs:
        console.print("[red]No simulation data found. Run a simulation first.[/red]")
        press_any_key_to_continue()
        return
    
    # Show recent simulations
    choices = []
    for sim_dir in sim_dirs[:10]:  # Show last 10
        timestamp = sim_dir.name
        choices.append(
            questionary.Choice(f"{timestamp}", value=str(sim_dir))
        )
    
    selected = questionary.select(
        "Select simulation to analyze:",
        choices=choices
    ).ask()
    
    if selected:
        console.print(f"[green]Analyzing: {selected}[/green]")
        console.print()
        
        try:
            visualize_simulation_data(Path(selected), interactive=True)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        
        press_any_key_to_continue()


def configure_satellite_menu():
    """Configure satellite parameters."""
    console.print()
    console.print(Panel("[bold]Configure Satellite[/bold]", border_style="yellow"))
    console.print()
    
    try:
        edit_config_interactive()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    
    press_any_key_to_continue()


def tune_mpc_menu():
    """Tune MPC parameters interactively."""
    console.print()
    console.print(Panel("[bold]Tune MPC Parameters[/bold]", border_style="magenta"))
    console.print()
    
    console.print("[yellow]MPC tuning interface coming soon![/yellow]")
    console.print("[dim]For now, edit configuration files directly or use configure menu.[/dim]")
    console.print()
    press_any_key_to_continue()


def mission_planning_menu():
    """Mission planning interface."""
    console.print()
    console.print(Panel("[bold]Mission Planning[/bold]", border_style="blue"))
    console.print()
    
    # List available missions
    registry = get_registry()
    plugins = registry.list_plugins()
    
    if not plugins:
        console.print("[yellow]No mission plugins found.[/yellow]")
        press_any_key_to_continue()
        return
    
    choices = []
    for plugin_name in plugins:
        plugin = registry.get_plugin(plugin_name)
        if plugin:
            choices.append(
                questionary.Choice(
                    f"{plugin.get_display_name()}: {plugin.get_description()}",
                    value=plugin_name
                )
            )
    
    selected = questionary.select(
        "Select mission type:",
        choices=choices
    ).ask()
    
    if selected:
        console.print(f"[green]Selected mission: {selected}[/green]")
        console.print("[dim]Mission planning interface coming soon![/dim]")
        console.print()
    
    press_any_key_to_continue()


def visualize_menu():
    """Visualization options."""
    console.print()
    console.print(Panel("[bold]View Visualizations[/bold]", border_style="cyan"))
    console.print()
    
    analyze_results_menu()  # Reuse analyze menu for now


def export_menu():
    """Export data options."""
    console.print()
    console.print(Panel("[bold]Export Data[/bold]", border_style="green"))
    console.print()
    
    console.print("[yellow]Export functionality coming soon![/yellow]")
    console.print("[dim]Use visualization commands to generate plots and videos.[/dim]")
    console.print()
    press_any_key_to_continue()


def run_studio():
    """Main studio mode loop."""
    console.print()
    console.print(Panel.fit(
        "[bold blue]Satellite Control Studio[/bold blue]\n"
        "[dim]Everything in Python - No web server required[/dim]",
        border_style="blue"
    ))
    
    while True:
        choice = show_main_menu()
        
        if choice == "quit":
            console.print("\n[yellow]Exiting studio mode. Goodbye![/yellow]\n")
            break
        elif choice == "run":
            run_simulation_menu()
        elif choice == "analyze":
            analyze_results_menu()
        elif choice == "configure":
            configure_satellite_menu()
        elif choice == "tune":
            tune_mpc_menu()
        elif choice == "mission":
            mission_planning_menu()
        elif choice == "visualize":
            visualize_menu()
        elif choice == "export":
            export_menu()
        else:
            console.print(f"[red]Unknown option: {choice}[/red]")
        
        console.print()


if __name__ == "__main__":
    run_studio()
