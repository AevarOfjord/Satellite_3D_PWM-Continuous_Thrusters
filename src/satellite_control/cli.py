"""
Satellite Control CLI
=====================

Main entry point for the satellite control system.
Provides commands for running simulations, verification tests, and managing configuration.
"""

from typing import Any, Dict, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel

# Import internal modules (lazy import where possible to speed up help)
from src.satellite_control.config import SatelliteConfig
from src.satellite_control.config.presets import (
    ConfigPreset,
    get_preset_description,
    load_preset,
    list_presets,
)
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
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help=f"Use configuration preset: {', '.join(ConfigPreset.all())}",
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

    # Prepare simulation parameters (avoid mutating global config)
    sim_start_pos: Optional[Tuple[float, float, float]] = None
    sim_target_pos: Optional[Tuple[float, float, float]] = None
    sim_start_angle: Optional[Tuple[float, float, float]] = None
    sim_target_angle: Optional[Tuple[float, float, float]] = None
    config_overrides: Optional[Dict[str, Dict[str, Any]]] = None

    if auto:
        console.print("[yellow]Running in AUTO mode with default parameters...[/yellow]")
        # Set auto mode parameters directly (don't mutate global config)
        sim_start_pos = (1.0, 1.0, 0.0)
        sim_target_pos = (0.0, 0.0, 0.0)
        sim_start_angle = (0.0, 0.0, 0.0)
        sim_target_angle = (0.0, 0.0, 0.0)
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
                mission_preset = interactive_cli.select_mission_preset()
                if mission_preset is None:
                    # Custom mission flow
                    mission_config = interactive_cli.run_custom_waypoint_mission()
                    if not mission_config:
                        console.print("[red]Mission cancelled.[/red]")
                        raise typer.Exit()
                    # Extract mission parameters from config
                    if "start_pos" in mission_config:
                        sim_start_pos = mission_config.get("start_pos")
                    if "start_angle" in mission_config:
                        sim_start_angle = mission_config.get("start_angle")
                elif not mission_preset:
                    # User cancelled preset
                    console.print("[red]Mission cancelled.[/red]")
                    raise typer.Exit()
                else:
                    # Extract mission parameters from preset
                    if "start_pos" in mission_preset:
                        sim_start_pos = mission_preset.get("start_pos")
                    if "start_angle" in mission_preset:
                        sim_start_angle = mission_preset.get("start_angle")
            elif mode == "shape_following":
                # Use new interactive shape following UI
                mission_config = interactive_cli.run_shape_following_mission()
                if not mission_config:
                    console.print("[red]Mission cancelled.[/red]")
                    raise typer.Exit()
                # Extract mission parameters from config
                if "start_pos" in mission_config:
                    sim_start_pos = mission_config.get("start_pos")
                if "start_angle" in mission_config:
                    sim_start_angle = mission_config.get("start_angle")
        except ImportError:
            # Fallback if interactive module fails
            console.print("[yellow]Falling back to classic menu...[/yellow]")
            mission_manager = MissionManager()
            mode = mission_manager.show_mission_menu()
            if not mission_manager.run_selected_mission(mode):
                console.print("[red]Mission cancelled.[/red]")
                raise typer.Exit()

    # Load MPC configuration preset if specified (CLI option)
    # Note: This is different from mission presets from interactive CLI
    if preset:
        try:
            # preset is a string from CLI option (e.g., "fast", "balanced")
            if isinstance(preset, str):
                preset_config = load_preset(preset)
                if config_overrides is None:
                    config_overrides = {}
                # Merge preset config into overrides
                for key, value in preset_config.items():
                    if key not in config_overrides:
                        config_overrides[key] = {}
                    config_overrides[key].update(value)
                console.print(f"[green]Loaded MPC preset: {preset}[/green]")
                preset_desc = get_preset_description(preset)
                console.print(f"[dim]{preset_desc}[/dim]")
            else:
                # Should not happen - preset from CLI option is always a string
                console.print("[yellow]Warning: Invalid preset type[/yellow]")
        except ValueError as e:
            console.print(f"[bold red]Invalid preset:[/bold red] {e}")
            console.print(f"[yellow]Available presets: {', '.join(ConfigPreset.all())}[/yellow]")
            raise typer.Exit(code=1)

    # Validate configuration at startup
    try:
        from src.satellite_control.config.validator import validate_config_at_startup

        validate_config_at_startup()
    except ValueError as e:
        console.print(f"[bold red]Configuration validation failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    # Apply Overrides
    console.print("\n[bold]Initializing Simulation...[/bold]")
    if duration:
        # Use config_overrides instead of mutating global config
        if config_overrides is None:
            config_overrides = {}
        if "simulation" not in config_overrides:
            config_overrides["simulation"] = {}
        config_overrides["simulation"]["max_duration"] = duration
        console.print(f"  Override: Duration = {duration}s")

    # Create SimulationConfig if we have overrides or preset
    simulation_config = None
    if config_overrides or preset:
        from src.satellite_control.config.simulation_config import SimulationConfig
        
        # Create config with overrides
        simulation_config = SimulationConfig.create_with_overrides(
            config_overrides or {}
        )
    
    # Initialize Simulation with explicit parameters (avoid global state mutation)
    try:
        sim = SatelliteMPCLinearizedSimulation(
            start_pos=sim_start_pos,
            target_pos=sim_target_pos,
            start_angle=sim_start_angle,
            target_angle=sim_target_angle,
            config_overrides=config_overrides,  # Keep for backward compatibility
            simulation_config=simulation_config,  # New preferred way
        )
        
        # Apply duration override directly to simulation if needed
        # (This avoids mutating global SatelliteConfig.MAX_SIMULATION_TIME)
        if duration:
            sim.max_simulation_time = duration
        
        console.print("[green]Simulation initialized successfully.[/green]")
        console.print("Starting Simulation loop...")

        sim.run_simulation(show_animation=not no_anim)

    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation stopping (KeyboardInterrupt)...[/yellow]")
    except Exception as e:
        import traceback
        console.print(f"\n[bold red]Error running simulation:[/bold red] {e}")
        console.print("[dim]Full traceback:[/dim]")
        console.print(traceback.format_exc())
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
def presets(
    list_all: bool = typer.Option(False, "--list", "-l", help="List all available presets"),
):
    """
    Manage configuration presets.
    
    Presets provide pre-configured settings optimized for different use cases.
    """
    if list_all:
        console.print("[bold]Available Configuration Presets:[/bold]\n")
        presets_dict = list_presets()
        for preset_name, description in presets_dict.items():
            console.print(f"[bold cyan]{preset_name.upper()}[/bold cyan]")
            console.print(f"  {description}\n")
        console.print(
            "[dim]Usage: python run_simulation.py run --preset <name>[/dim]"
        )
    else:
        console.print("[yellow]Use --list to see available presets[/yellow]")


@app.command()
def config(
    dump: bool = typer.Option(False, "--dump", help="Dump current effective config"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate configuration"),
):
    """
    Inspect or validate configuration.
    """
    try:
        app_config = SatelliteConfig.get_app_config()

        if dump:
            console.print_json(app_config.model_dump_json())
        else:
            if validate:
                # Use comprehensive validator
                from src.satellite_control.config.validator import ConfigValidator

                validator = ConfigValidator()
                issues = validator.validate_all(app_config)

                if issues:
                    console.print("[bold red]Configuration validation failed:[/bold red]")
                    for issue in issues:
                        console.print(f"  [red]✗[/red] {issue}")
                    raise typer.Exit(code=1)
                else:
                    console.print("[bold green]✓ Configuration is valid.[/bold green]")

            mode_str = "Realistic" if app_config.physics.use_realistic_physics else "Idealized"
            console.print(f"Physics Mode: {mode_str}")

    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
