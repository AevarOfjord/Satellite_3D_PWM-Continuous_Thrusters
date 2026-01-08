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
# V4.0.0: SatelliteConfig removed - use SimulationConfig only
from src.satellite_control.config.presets import (
    ConfigPreset,
    get_preset_description,
    load_preset,
    list_presets,
)
from src.satellite_control.core.simulation import SatelliteMPCLinearizedSimulation
from src.satellite_control.mission.mission_manager import MissionManager
from src.satellite_control.mission.plugin import get_registry, discover_plugins
from src.satellite_control.mission.plugins import *  # Auto-register built-in plugins

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
    # Mission configuration from CLI (v2.0.0: uses SimulationConfig)
    mission_simulation_config = None
    
    if classic:
        # Use classic text-based menu
        mission_manager = MissionManager()
        mode = mission_manager.show_mission_menu()
        mission_result = mission_manager.run_selected_mission(mode)
        if not mission_result:
            console.print("[red]Mission configuration cancelled.[/red]")
            raise typer.Exit()
        # Extract SimulationConfig if available (v2.0.0)
        if isinstance(mission_result, dict) and "simulation_config" in mission_result:
            mission_simulation_config = mission_result["simulation_config"]
        # Extract start position/angle for backward compatibility
        if isinstance(mission_result, dict):
            if "start_pos" in mission_result:
                sim_start_pos = mission_result.get("start_pos")
            if "start_angle" in mission_result:
                sim_start_angle = mission_result.get("start_angle")
    else:
        # Use new interactive menu
        try:
            from src.satellite_control.mission.interactive_cli import (
                InteractiveMissionCLI,
            )

            interactive_cli = InteractiveMissionCLI()
            mode = interactive_cli.show_mission_menu()

            if mode == "waypoint":
                mission_preset = interactive_cli.select_mission_preset(return_simulation_config=True)
                if mission_preset is None:
                    # Custom mission flow
                    mission_config = interactive_cli.run_custom_waypoint_mission()
                    if not mission_config:
                        console.print("[red]Mission cancelled.[/red]")
                        raise typer.Exit()
                    # Extract SimulationConfig if available (v2.0.0)
                    if "simulation_config" in mission_config:
                        mission_simulation_config = mission_config["simulation_config"]
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
                    # Extract SimulationConfig if available (v2.0.0)
                    if "simulation_config" in mission_preset:
                        mission_simulation_config = mission_preset["simulation_config"]
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
                # Extract SimulationConfig if available (v2.0.0)
                if "simulation_config" in mission_config:
                    mission_simulation_config = mission_config["simulation_config"]
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
            mission_result = mission_manager.run_selected_mission(mode)
            if not mission_result:
                console.print("[red]Mission cancelled.[/red]")
                raise typer.Exit()
            # Extract SimulationConfig if available
            if isinstance(mission_result, dict) and "simulation_config" in mission_result:
                mission_simulation_config = mission_result["simulation_config"]

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

    # Create SimulationConfig (v2.0.0 pattern)
    from src.satellite_control.config.simulation_config import SimulationConfig
    
    simulation_config = None
    if mission_simulation_config:
        # Use SimulationConfig from mission CLI (preferred, v2.0.0)
        simulation_config = mission_simulation_config
        # Apply overrides if provided
        if config_overrides:
            simulation_config = SimulationConfig.create_with_overrides(
                config_overrides, base_config=simulation_config
            )
    elif config_overrides or preset:
        # Create config with overrides (no mission config available)
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
        # V4.0.0: No global state mutation
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


# Interactive subcommand (explicit)
@app.command()
def interactive(
    duration: Optional[float] = typer.Option(
        None, "--duration", "-d", help="Override max simulation time in seconds"
    ),
    no_anim: bool = typer.Option(False, "--no-anim", help="Disable animation (headless mode)"),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help=f"Use configuration preset: {', '.join(ConfigPreset.all())}",
    ),
):
    """
    Run the interactive mission selector and start a simulation.
    """
    # Reuse run() logic; headless/classic/auto are off in interactive mode
    run(
        auto=False,
        duration=duration,
        no_anim=no_anim,
        headless=False,
        classic=False,
        preset=preset,
    )


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
def bench(
    marker: str = typer.Option(
        "benchmark", "--marker", "-k", help="Pytest -k expression (default: benchmark)"
    ),
    full: bool = typer.Option(False, "--full", help="Run all benchmarks (may be slow)"),
):
    """
    Run benchmark suite (pytest-benchmark).
    """
    import pytest

    console.print("[bold]Running benchmarks...[/bold]")
    args = ["tests", "--benchmark-only"]
    if not full:
        args.extend(["-k", marker])

    ret_code = pytest.main(args)
    if ret_code == 0:
        console.print("\n[bold green]Benchmarks completed.[/bold green]")
    else:
        console.print("\n[bold red]Benchmarks failed.[/bold red]")
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
def export_config(
    output: str = typer.Option(
        "config.yaml",
        "--output",
        "-o",
        help="Output file path (.yaml, .yml, or .json)",
    ),
    format: str = typer.Option(
        "auto",
        "--format",
        "-f",
        help="File format: auto, yaml, or json",
    ),
):
    """
    Export current configuration to file (V3.0.0).
    
    Saves the current SimulationConfig to a YAML or JSON file.
    The file can be loaded later with 'load-config' command.
    """
    from src.satellite_control.config.io import ConfigIO
    from src.satellite_control.config.simulation_config import SimulationConfig
    
    try:
        # Create default config (or could use current simulation config)
        config = SimulationConfig.create_default()
        
        # Save to file
        ConfigIO.save(config, output, format=format)
        
        console.print(f"[green]✓ Configuration exported to {output}[/green]")
        console.print(f"[dim]Version: {ConfigIO.CURRENT_CONFIG_VERSION}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Failed to export config:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def load_config(
    file_path: str = typer.Argument(..., help="Path to config file (.yaml, .yml, or .json)"),
    migrate: bool = typer.Option(
        True,
        "--migrate/--no-migrate",
        help="Automatically migrate older config versions",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate configuration after loading",
    ),
):
    """
    Load configuration from file (V3.0.0).
    
    Loads a SimulationConfig from a YAML or JSON file.
    Older config versions will be automatically migrated if --migrate is enabled.
    """
    from pathlib import Path
    from src.satellite_control.config.io import ConfigIO
    from src.satellite_control.config.simulation_config import SimulationConfig
    
    try:
        config_file = Path(file_path)
        
        if not config_file.exists():
            console.print(f"[bold red]Config file not found: {file_path}[/bold red]")
            raise typer.Exit(code=1)
        
        # Load config
        config = ConfigIO.load(config_file, migrate=migrate)
        
        console.print(f"[green]✓ Configuration loaded from {file_path}[/green]")
        
        # Validate if requested
        if validate:
            from src.satellite_control.config.validator import ConfigValidator
            
            validator = ConfigValidator()
            issues = validator.validate_all(config.app_config)
            
            if issues:
                console.print("[bold yellow]Configuration validation warnings:[/bold yellow]")
                for issue in issues:
                    console.print(f"  [yellow]⚠[/yellow] {issue}")
            else:
                console.print("[green]✓ Configuration is valid[/green]")
        
        # Show summary
        console.print("\n[bold]Configuration Summary:[/bold]")
        console.print(f"  MPC dt: {config.app_config.mpc.dt}s")
        console.print(f"  Simulation dt: {config.app_config.simulation.dt}s")
        console.print(f"  Max duration: {config.app_config.simulation.max_duration}s")
        console.print(f"  Control dt: {config.app_config.simulation.control_dt}s")
        
        console.print("\n[dim]To use this config, pass it to SimulationConfig.create_with_overrides()[/dim]")
        
    except FileNotFoundError:
        console.print(f"[bold red]Config file not found: {file_path}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Failed to load config:[/bold red] {e}")
        import traceback
        console.print("[dim]Traceback:[/dim]")
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command()
def config(
    dump: bool = typer.Option(False, "--dump", help="Dump current effective config"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate configuration"),
):
    """
    Inspect or validate configuration.
    """
    try:
        # V4.0.0: Create default config (no SatelliteConfig fallback)
        from src.satellite_control.config.simulation_config import SimulationConfig
        
        # Try to get from default config first (v3.0.0)
        try:
            default_config = SimulationConfig.create_default()
            app_config = default_config.app_config
        except Exception:
            # V4.0.0: Use default config (no SatelliteConfig fallback)
            from src.satellite_control.config.simulation_config import SimulationConfig
            default_config = SimulationConfig.create_default()
            app_config = default_config.app_config

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


@app.command("list-missions")
def list_missions(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed plugin information"),
):
    """
    List all available mission plugins.
    
    Shows built-in and discovered mission plugins with their descriptions.
    """
    # Discover plugins from search paths
    registry = get_registry()
    discovered_count = discover_plugins()
    
    console.print(Panel.fit("Available Mission Plugins", style="bold blue"))
    
    plugins = registry.list_plugins()
    if not plugins:
        console.print("[yellow]No mission plugins found.[/yellow]")
        return
    
    console.print(f"\nFound {len(plugins)} mission plugin(s):\n")
    
    for name in plugins:
        info = registry.get_plugin_info(name)
        if info is None:
            continue
        
        console.print(f"[bold cyan]{info['display_name']}[/bold cyan] ({name})")
        console.print(f"  Version: {info['version']}")
        if info['author']:
            console.print(f"  Author: {info['author']}")
        console.print(f"  Description: {info['description']}")
        
        if verbose:
            console.print(f"  File: {info['file_path']}")
            if info['required_parameters']:
                console.print(f"  Required parameters: {', '.join(info['required_parameters'])}")
        
        console.print()


@app.command("install-mission")
def install_mission(
    plugin_path: str = typer.Argument(..., help="Path to Python file containing mission plugin"),
):
    """
    Install a custom mission plugin from a file.
    
    The plugin file must contain a class that inherits from MissionPlugin.
    """
    from pathlib import Path
    
    file_path = Path(plugin_path).expanduser().resolve()
    
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    if not file_path.is_file():
        console.print(f"[red]Error: Not a file: {file_path}[/red]")
        raise typer.Exit(1)
    
    if not file_path.suffix == ".py":
        console.print(f"[red]Error: File must be a Python file (.py): {file_path}[/red]")
        raise typer.Exit(1)
    
    registry = get_registry()
    plugin = registry.load_plugin_from_file(file_path)
    
    if plugin is None:
        console.print(f"[red]Error: Failed to load plugin from {file_path}[/red]")
        console.print("[yellow]Make sure the file contains a class inheriting from MissionPlugin[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"[green]✓ Successfully installed plugin: {plugin.get_display_name()}[/green]")
    console.print(f"  Name: {plugin.get_name()}")
    console.print(f"  Version: {plugin.get_version()}")
    console.print(f"\nUse 'satellite-control list-missions' to see all available plugins.")


@app.command("studio")
def studio():
    """
    Launch interactive studio mode.
    
    Opens an interactive menu system for running simulations, analyzing results,
    configuring satellites, and viewing visualizations - all in Python, no web server.
    """
    from src.satellite_control.studio import run_studio
    
    try:
        run_studio()
    except KeyboardInterrupt:
        console.print("\n[yellow]Studio mode exited.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in studio mode: {e}[/red]")
        raise typer.Exit(1)


@app.command("visualize")
def visualize(
    data_path: str = typer.Argument(..., help="Path to simulation data directory"),
    interactive: bool = typer.Option(True, "--interactive/--static", help="Open interactive plots (default) or generate static images"),
):
    """
    Visualize simulation results using Python-native matplotlib.
    
    Opens interactive matplotlib windows for trajectory, telemetry, and performance analysis.
    No web server required - everything runs in Python.
    """
    from pathlib import Path
    from src.satellite_control.visualize import visualize_simulation_data
    
    data_dir = Path(data_path).expanduser().resolve()
    
    if not data_dir.exists():
        console.print(f"[red]Error: Data directory not found: {data_dir}[/red]")
        raise typer.Exit(1)
    
    try:
        visualize_simulation_data(data_dir, interactive=interactive)
    except Exception as e:
        console.print(f"[red]Error visualizing data: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
