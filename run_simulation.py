#!/usr/bin/env python3
"""
Satellite Control Simulation Entry Point
Delegates to the new CLI interface.
"""
import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.satellite_control.cli import app  # noqa: E402

if __name__ == "__main__":
    # If run without arguments, default to 'run' command to maintain backward compatibility check?
    # Typer requires a command. If the user just ran "python run_simulation.py",
    # we want it to behave like "python run_simulation.py run".
    # We can inject 'run' if no args provided, or just let Typer handle it (it will show help).
    # The original script started interactive mode immediately.
    # To preserve exact behavior:
    if len(sys.argv) == 1:
        sys.argv.append("run")

    app()
