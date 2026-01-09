"""
Mission Types and Data Structures

Defines the core mission types for orbital inspection:
- Flyby: Single pass by target
- Circumnavigation: Orbit around target
- Station-Keeping: Hold position relative to target
- Inspection: Multi-point detailed inspection
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np
import json
from pathlib import Path


class MissionType(Enum):
    """Types of inspection missions."""

    FLYBY = "flyby"
    CIRCUMNAVIGATION = "circumnavigation"
    STATION_KEEPING = "station_keeping"
    INSPECTION = "inspection"
    CUSTOM = "custom"


class MissionStatus(Enum):
    """Status of mission execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class Waypoint:
    """A single waypoint in the mission trajectory."""

    position: np.ndarray  # [x, y, z] meters
    hold_time: float = 0.0  # Time to hold at waypoint (seconds)
    approach_speed: float = 0.05  # m/s
    name: str = ""

    def to_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "hold_time": self.hold_time,
            "approach_speed": self.approach_speed,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Waypoint":
        return cls(
            position=np.array(data["position"]),
            hold_time=data.get("hold_time", 0.0),
            approach_speed=data.get("approach_speed", 0.05),
            name=data.get("name", ""),
        )


@dataclass
class MissionPhase:
    """A phase of the mission with multiple waypoints."""

    name: str
    waypoints: List[Waypoint] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MissionPhase":
        return cls(
            name=data["name"],
            waypoints=[Waypoint.from_dict(wp) for wp in data.get("waypoints", [])],
            description=data.get("description", ""),
        )


@dataclass
class Mission:
    """
    Complete mission definition.

    A mission consists of:
    - Metadata (name, type, description)
    - Start position
    - One or more phases with waypoints
    - Safety parameters
    """

    name: str
    mission_type: MissionType
    phases: List[MissionPhase] = field(default_factory=list)
    start_position: np.ndarray = field(default_factory=lambda: np.array([5.0, 0.0, 0.0]))
    description: str = ""

    # Safety parameters
    keep_out_radius: float = 2.0  # Minimum distance from target (m)
    max_speed: float = 0.1  # Maximum approach speed (m/s)

    # Execution state
    status: MissionStatus = MissionStatus.PENDING
    current_phase_idx: int = 0
    current_waypoint_idx: int = 0

    @property
    def total_waypoints(self) -> int:
        return sum(len(phase.waypoints) for phase in self.phases)

    @property
    def current_phase(self) -> Optional[MissionPhase]:
        if 0 <= self.current_phase_idx < len(self.phases):
            return self.phases[self.current_phase_idx]
        return None

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        phase = self.current_phase
        if phase and 0 <= self.current_waypoint_idx < len(phase.waypoints):
            return phase.waypoints[self.current_waypoint_idx]
        return None

    def get_all_waypoints(self) -> List[Waypoint]:
        """Get all waypoints from all phases in order."""
        waypoints = []
        for phase in self.phases:
            waypoints.extend(phase.waypoints)
        return waypoints

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mission_type": self.mission_type.value,
            "description": self.description,
            "start_position": self.start_position.tolist(),
            "keep_out_radius": self.keep_out_radius,
            "max_speed": self.max_speed,
            "phases": [phase.to_dict() for phase in self.phases],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Mission":
        return cls(
            name=data["name"],
            mission_type=MissionType(data["mission_type"]),
            description=data.get("description", ""),
            start_position=np.array(data.get("start_position", [5.0, 0.0, 0.0])),
            keep_out_radius=data.get("keep_out_radius", 2.0),
            max_speed=data.get("max_speed", 0.1),
            phases=[MissionPhase.from_dict(p) for p in data.get("phases", [])],
        )

    def save(self, filepath: Path):
        """Save mission to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "Mission":
        """Load mission from JSON file."""
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))


# ============================================
# Mission Templates
# ============================================


def create_flyby_mission(
    name: str = "Flyby Inspection",
    start_distance: float = 10.0,
    pass_distance: float = 3.0,
    approach_speed: float = 0.05,
) -> Mission:
    """
    Create a flyby mission template.

    Trajectory: Start → Approach → Flyby → Depart
    """
    mission = Mission(
        name=name,
        mission_type=MissionType.FLYBY,
        description="Single-pass flyby inspection of target",
        start_position=np.array([start_distance, 0.0, 0.0]),
    )

    mission.phases = [
        MissionPhase(
            name="Approach",
            description="Approach target from starting position",
            waypoints=[
                Waypoint(
                    position=np.array([pass_distance + 2, 0.0, 0.0]),
                    approach_speed=approach_speed,
                    name="Approach point",
                ),
            ],
        ),
        MissionPhase(
            name="Flyby",
            description="Pass by target at close range",
            waypoints=[
                Waypoint(
                    position=np.array([pass_distance, 2.0, 0.0]),
                    approach_speed=approach_speed * 0.5,
                    name="Flyby start",
                ),
                Waypoint(
                    position=np.array([pass_distance, -2.0, 0.0]),
                    approach_speed=approach_speed * 0.5,
                    name="Flyby end",
                ),
            ],
        ),
        MissionPhase(
            name="Departure",
            description="Depart to safe distance",
            waypoints=[
                Waypoint(
                    position=np.array([start_distance, 0.0, 0.0]),
                    approach_speed=approach_speed,
                    name="Safe distance",
                ),
            ],
        ),
    ]

    return mission


def create_circumnavigation_mission(
    name: str = "Circumnavigation",
    orbit_radius: float = 5.0,
    num_points: int = 8,
    hold_time: float = 5.0,
) -> Mission:
    """
    Create a circumnavigation mission template.

    Orbit around target in XY plane with inspection holds.
    """
    mission = Mission(
        name=name,
        mission_type=MissionType.CIRCUMNAVIGATION,
        description="Complete orbit around target with inspection points",
        start_position=np.array([orbit_radius, 0.0, 0.0]),
    )

    # Generate circular waypoints
    waypoints = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = orbit_radius * np.cos(angle)
        y = orbit_radius * np.sin(angle)
        waypoints.append(
            Waypoint(
                position=np.array([x, y, 0.0]),
                hold_time=hold_time,
                approach_speed=0.03,
                name=f"Point {i+1}",
            )
        )

    # Close the loop
    waypoints.append(
        Waypoint(
            position=np.array([orbit_radius, 0.0, 0.0]),
            hold_time=0,
            approach_speed=0.03,
            name="Return",
        )
    )

    mission.phases = [
        MissionPhase(
            name="Orbit",
            description="Circular inspection orbit",
            waypoints=waypoints,
        ),
    ]

    return mission


def create_station_keeping_mission(
    name: str = "Station Keeping",
    position: np.ndarray = None,
    duration: float = 300.0,
) -> Mission:
    """
    Create a station-keeping mission template.

    Hold position at specified offset from target.
    """
    if position is None:
        position = np.array([5.0, 0.0, 0.0])

    mission = Mission(
        name=name,
        mission_type=MissionType.STATION_KEEPING,
        description=f"Hold position at {position} for {duration}s",
        start_position=position.copy(),
    )

    mission.phases = [
        MissionPhase(
            name="Station Keeping",
            description="Maintain position",
            waypoints=[
                Waypoint(
                    position=position.copy(),
                    hold_time=duration,
                    approach_speed=0.02,
                    name="Hold position",
                ),
            ],
        ),
    ]

    return mission


def create_inspection_mission(
    name: str = "Detailed Inspection",
    inspection_points: List[np.ndarray] = None,
    hold_time: float = 30.0,
) -> Mission:
    """
    Create a multi-point inspection mission.

    Visit multiple inspection points with extended holds.
    """
    if inspection_points is None:
        # Default: 6 faces of target cube
        d = 3.0  # Distance from target
        inspection_points = [
            np.array([d, 0, 0]),  # +X
            np.array([0, d, 0]),  # +Y
            np.array([0, 0, d]),  # +Z
            np.array([-d, 0, 0]),  # -X
            np.array([0, -d, 0]),  # -Y
            np.array([0, 0, -d]),  # -Z
        ]

    mission = Mission(
        name=name,
        mission_type=MissionType.INSPECTION,
        description="Multi-point detailed inspection",
        start_position=np.array([5.0, 5.0, 0.0]),
    )

    waypoints = [
        Waypoint(
            position=pt.copy(),
            hold_time=hold_time,
            approach_speed=0.02,
            name=f"Inspection point {i+1}",
        )
        for i, pt in enumerate(inspection_points)
    ]

    mission.phases = [
        MissionPhase(
            name="Inspection",
            description="Visit all inspection points",
            waypoints=waypoints,
        ),
        MissionPhase(
            name="Return",
            description="Return to safe distance",
            waypoints=[
                Waypoint(
                    position=np.array([5.0, 5.0, 0.0]),
                    hold_time=0,
                    approach_speed=0.05,
                    name="Safe standoff",
                ),
            ],
        ),
    ]

    return mission


# Pre-built mission library
MISSION_LIBRARY = {
    "flyby": create_flyby_mission,
    "circumnavigation": create_circumnavigation_mission,
    "station_keeping": create_station_keeping_mission,
    "inspection": create_inspection_mission,
}
