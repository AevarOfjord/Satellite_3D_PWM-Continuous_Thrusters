#!/usr/bin/env python3
"""
Mission System Integration Test

Tests the mission executor with different mission templates.
"""

from pathlib import Path
import sys
import logging

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.satellite_control.mission import (
    Mission,
    MissionExecutor,
    create_flyby_mission,
    create_circumnavigation_mission,
    create_station_keeping_mission,
    create_inspection_mission,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_flyby_mission():
    """Test flyby mission execution."""
    print("=" * 60)
    print("FLYBY MISSION TEST")
    print("=" * 60)

    mission = create_flyby_mission(
        start_distance=6.0,
        pass_distance=4.0,
        approach_speed=0.15,  # Faster for testing
    )

    print(f"\nMission: {mission.name}")
    print(f"Type: {mission.mission_type.value}")
    print(f"Phases: {len(mission.phases)}")
    print(f"Total waypoints: {mission.total_waypoints}")

    for phase in mission.phases:
        print(f"\n  Phase: {phase.name}")
        for i, wp in enumerate(phase.waypoints):
            print(f"    {i+1}. {wp.name}: {wp.position}")

    # Create executor
    model_path = ROOT / "models" / "satellite_rw.xml"
    executor = MissionExecutor(model_path=str(model_path))

    # Execute with timeout
    print("\nExecuting mission (max 120s)...")
    result = executor.execute(mission, timeout=120.0)

    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration:.1f}s")
    print(f"  Waypoints: {result.waypoints_reached}/{result.total_waypoints}")
    print(f"  Final error: {result.final_error*100:.2f}cm")

    return result.success


def test_circumnavigation_mission():
    """Test circumnavigation mission."""
    print("\n" + "=" * 60)
    print("CIRCUMNAVIGATION MISSION TEST")
    print("=" * 60)

    mission = create_circumnavigation_mission(
        orbit_radius=4.0,
        num_points=4,  # 4 points for faster test
        hold_time=2.0,  # Short holds
    )

    print(f"\nMission: {mission.name}")
    print(f"Waypoints: {mission.total_waypoints}")

    model_path = ROOT / "models" / "satellite_rw.xml"
    executor = MissionExecutor(model_path=str(model_path))

    print("\nExecuting mission (max 180s)...")
    result = executor.execute(mission, timeout=180.0)

    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration:.1f}s")
    print(f"  Waypoints: {result.waypoints_reached}/{result.total_waypoints}")

    return result.success


def test_station_keeping_mission():
    """Test station-keeping mission."""
    print("\n" + "=" * 60)
    print("STATION-KEEPING MISSION TEST")
    print("=" * 60)

    import numpy as np

    mission = create_station_keeping_mission(
        position=np.array([5.0, 0.0, 0.0]),
        duration=30.0,  # 30 second hold
    )

    print(f"\nMission: {mission.name}")
    print(f"Hold position: {mission.phases[0].waypoints[0].position}")
    print(f"Duration: {mission.phases[0].waypoints[0].hold_time}s")

    model_path = ROOT / "models" / "satellite_rw.xml"
    executor = MissionExecutor(model_path=str(model_path))

    print("\nExecuting mission...")
    result = executor.execute(mission, timeout=60.0)

    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration:.1f}s")
    print(f"  Final error: {result.final_error*100:.2f}cm")

    return result.success


def test_mission_save_load():
    """Test mission serialization."""
    print("\n" + "=" * 60)
    print("MISSION SAVE/LOAD TEST")
    print("=" * 60)

    import tempfile

    # Create mission
    mission = create_flyby_mission()

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        mission.save(Path(f.name))
        print(f"Saved to: {f.name}")

        # Load back
        loaded = Mission.load(Path(f.name))

        print(f"Original: {mission.name}, {mission.total_waypoints} waypoints")
        print(f"Loaded: {loaded.name}, {loaded.total_waypoints} waypoints")

        # Verify
        success = mission.name == loaded.name and mission.total_waypoints == loaded.total_waypoints

        print(f"Match: {'✓ PASS' if success else '✗ FAIL'}")

        return success


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MISSION SYSTEM INTEGRATION TESTS")
    print("=" * 60 + "\n")

    # Run tests
    save_load_ok = test_mission_save_load()
    station_ok = test_station_keeping_mission()
    flyby_ok = test_flyby_mission()

    # Skip circumnavigation for now (takes longer)
    # circum_ok = test_circumnavigation_mission()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Save/Load: {'✓' if save_load_ok else '✗'}")
    print(f"Station-keeping: {'✓' if station_ok else '✗'}")
    print(f"Flyby: {'✓' if flyby_ok else '✗'}")

    all_pass = save_load_ok and station_ok and flyby_ok
    print(f"\nOVERALL: {'✓ PASS' if all_pass else '✗ FAIL'}")

    sys.exit(0 if all_pass else 1)
