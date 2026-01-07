"""
Satellite Control CLI
=====================

Main entry point for the satellite control system.
Provides commands for running simulations, verification tests, and managing configuration.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

# Import internal modules (lazy import where possible to speed up help)
from src.satellite_control.config import SatelliteConfig
from src.satellite_control.core.simulation import SatelliteMPCLinearizedSimulation
from src.satellite_control.mission.mission_manager import MissionManager

app = typer.Typer(
    help="Satellite Control System - MPC Simulation and Verification CLI",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    auto: bool = typer.Option(
        False, "--auto", "-a", help="Run in auto mode with default parameters"
    ),
    duration: Optional[float] = typer.Option(
        None, "--duration", "-d", help="Override max simulation time in seconds"
    ),
    no_anim: bool = typer.Option(False, "--no-anim", help="Disable animation (headless mode)"),
    headless: bool = typer.Option(
        False, "--headless", help="Alias for --no-anim (deprecated)", hidden=True
    ),
    classic: bool = typer.Option(
        False, "--classic", help="Use classic text-based menu instead of interactive"
    ),
):
    """
    Run the Satellite MPC Simulation.
    """
    if headless:
        no_anim = True

    console.print(
        Panel.fit(
            "Satellite Control Simulation",
            style="bold blue",
            subtitle="MPC Control System",
        )
    )

    if auto:
        console.print("[yellow]Running in AUTO mode with default parameters...[/yellow]")
        SatelliteConfig.DEFAULT_START_POS = (1.0, 1.0, 0.0)
        SatelliteConfig.DEFAULT_TARGET_POS = (0.0, 0.0, 0.0)
        SatelliteConfig.DEFAULT_START_ANGLE = (0.0, 0.0, 0.0)
        SatelliteConfig.DEFAULT_TARGET_ANGLE = (0.0, 0.0, 0.0)
    elif classic:
        # Use classic text-based menu
        mission_manager = MissionManager()
        mode = mission_manager.show_mission_menu()
        if not mission_manager.run_selected_mission(mode):
            console.print("[red]Mission configuration cancelled.[/red]")
            raise typer.Exit()
    else:
        # Use new interactive menu
        try:
            from src.satellite_control.mission.interactive_cli import (
                InteractiveMissionCLI,
            )

            interactive_cli = InteractiveMissionCLI()
            mode = interactive_cli.show_mission_menu()

            if mode == "waypoint":
                preset = interactive_cli.select_mission_preset()
                if preset is None:
                    # Custom mission flow
                    config = interactive_cli.run_custom_waypoint_mission()
                    if not config:
                        console.print("[red]Mission cancelled.[/red]")
                        raise typer.Exit()
                elif not preset:
                    # User cancelled preset
                    console.print("[red]Mission cancelled.[/red]")
                    raise typer.Exit()
            elif mode == "shape_following":
                # Use new interactive shape following UI
                config = interactive_cli.run_shape_following_mission()
                if not config:
                    console.print("[red]Mission cancelled.[/red]")
                    raise typer.Exit()
        except ImportError:
            # Fallback if interactive module fails
            console.print("[yellow]Falling back to classic menu...[/yellow]")
            mission_manager = MissionManager()
            mode = mission_manager.show_mission_menu()
            if not mission_manager.run_selected_mission(mode):
                console.print("[red]Mission cancelled.[/red]")
                raise typer.Exit()

    # Apply Overrides
    console.print("\n[bold]Initializing Simulation...[/bold]")
    if duration:
        SatelliteConfig.MAX_SIMULATION_TIME = duration
        console.print(f"  Override: Duration = {duration}s")

    # Initialize Simulation
    try:
        sim = SatelliteMPCLinearizedSimulation()
        console.print("[green]Simulation initialized successfully.[/green]")
        console.print("Starting Simulation loop...")

        sim.run_simulation(show_animation=not no_anim)

    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation stopping (KeyboardInterrupt)...[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error running simulation:[/bold red] {e}")
        raise typer.Exit(code=1)
    finally:
        if "sim" in locals():
            sim.close()


@app.command()
def verify(
    full: bool = typer.Option(False, "--full", help="Run full test suite (slow)"),
):
    """
    Run verification tests (E2E and Unit).
    """
    import pytest

    console.print("[bold]Running Verification Tests...[/bold]")

    args = ["tests/e2e/test_simulation_runner.py", "-v"]
    if not full:
        # Skip slow tests if not full? Currently E2E are marked slow but we want to run them.
        # Let's just run E2E by default as they are high value.
        pass

    ret_code = pytest.main(args)
    if ret_code == 0:
        console.print("\n[bold green]Verification Passed![/bold green]")
    else:
        console.print("\n[bold red]Verification Failed![/bold red]")
        raise typer.Exit(code=ret_code)


@app.command()
def config(
    dump: bool = typer.Option(False, "--dump", help="Dump current effective config"),
):
    """
    Inspect or validate configuration.
    """
    # AppConfig imported only if needed for type checking but was unused

    try:
        app_config = SatelliteConfig.get_app_config()

        if dump:
            console.print_json(app_config.model_dump_json())
        else:
            console.print("[bold green]Configuration is valid.[/bold green]")
            # console.print(f"MPC Type: {app_config.mpc.mpc_type}")
            mode_str = "Realistic" if app_config.physics.use_realistic_physics else "Idealized"
            console.print(f"Physics Mode: {mode_str}")

    except Exception as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
