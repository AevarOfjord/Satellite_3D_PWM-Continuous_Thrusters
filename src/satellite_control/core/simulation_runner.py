"""
Simulation Test Modes - Launchpad for various simulation scenarios.
"""

import argparse

from src.satellite_control.config import SatelliteConfig
from src.satellite_control.core.simulation import (
    SatelliteMPCLinearizedSimulation,
)
from src.satellite_control.mission.mission_manager import MissionManager


def main():
    """Main entry point for simulation execution."""
    parser = argparse.ArgumentParser(description="Satellite Simulation")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run in auto mode with default parameters (skip prompts)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override max simulation time in seconds",
    )
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="Disable animation",
    )

    args = parser.parse_args()

    # Use mission manager for configuration (Centralized)
    mission_manager = MissionManager()

    if args.auto:
        print("Running in AUTO mode with default parameters...")

        # Configure default mission (Point to Point default)
        SatelliteConfig.DEFAULT_START_POS = (1.0, 1.0, 0.0)
        SatelliteConfig.DEFAULT_TARGET_POS = (0.0, 0.0, 0.0)
        SatelliteConfig.DEFAULT_START_ANGLE = (0.0, 0.0, 0.0)
        SatelliteConfig.DEFAULT_TARGET_ANGLE = (0.0, 0.0, 0.0)
    else:
        # Interactive Mode
        print("\nSatellite MPC Simulation")
        print("========================\n")

        mode = mission_manager.show_mission_menu()
        if not mission_manager.run_selected_mission(mode):
            print("Mission configuration cancelled.")
            return

    # Create and run simulation
    print("\nInitializing Simulation...")
    if args.duration:
        SatelliteConfig.MAX_SIMULATION_TIME = args.duration

    sim = SatelliteMPCLinearizedSimulation()

    print("Starting Simulation...")
    try:
        sim.run_simulation(show_animation=not args.no_anim)
    except KeyboardInterrupt:
        print("\nSimulation stopping...")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
