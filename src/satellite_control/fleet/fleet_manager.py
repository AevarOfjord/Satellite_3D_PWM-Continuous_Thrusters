"""
Fleet Manager for Multi-Satellite Coordination

Manages N inspector satellites orbiting a target satellite.
Coordinates waypoints, timing, and collision avoidance.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class InspectorStatus(Enum):
    """Status of an inspector satellite."""

    IDLE = "idle"
    MANEUVERING = "maneuvering"
    STATION_KEEPING = "station_keeping"
    INSPECTING = "inspecting"
    AVOIDING = "avoiding"


@dataclass
class InspectorState:
    """State of a single inspector satellite."""

    id: int
    position: np.ndarray  # [x, y, z] relative to target
    velocity: np.ndarray  # [vx, vy, vz]
    quaternion: np.ndarray  # [qw, qx, qy, qz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    wheel_speeds: np.ndarray  # [ωrx, ωry, ωrz]
    status: InspectorStatus = InspectorStatus.IDLE

    @property
    def full_state(self) -> np.ndarray:
        """Get 16-element state vector."""
        return np.concatenate(
            [
                self.position,
                self.quaternion,
                self.velocity,
                self.angular_velocity,
                self.wheel_speeds,
            ]
        )

    def distance_to(self, other: "InspectorState") -> float:
        """Compute distance to another inspector."""
        return float(np.linalg.norm(self.position - other.position))

    def distance_to_origin(self) -> float:
        """Compute distance to target (origin)."""
        return float(np.linalg.norm(self.position))


@dataclass
class ObstacleConfig:
    """Configuration for obstacle avoidance."""

    target_keep_out_radius: float = 2.0  # meters from target center
    min_inspector_separation: float = 1.0  # meters between inspectors
    avoidance_penalty: float = 1e6  # MPC penalty weight
    soft_margin: float = 0.5  # meters extra margin for soft constraints


@dataclass
class FormationConfig:
    """Configuration for fleet formation."""

    num_inspectors: int = 3
    formation_radius: float = 5.0  # meters from target
    angular_separation: float = 120.0  # degrees between inspectors
    formation_plane: str = "xy"  # Plane for circular formation


@dataclass
class FleetManager:
    """
    Manages multiple inspector satellites.

    Coordinates:
    - Waypoint assignment
    - Formation maintenance
    - Collision avoidance
    - Timing synchronization
    """

    num_inspectors: int = 3
    obstacle_config: ObstacleConfig = field(default_factory=ObstacleConfig)
    formation_config: FormationConfig = field(default_factory=FormationConfig)

    # Internal state
    inspectors: Dict[int, InspectorState] = field(default_factory=dict)
    waypoints: Dict[int, List[np.ndarray]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize formation waypoints."""
        self.formation_config.num_inspectors = self.num_inspectors
        self._generate_formation_waypoints()

    def _generate_formation_waypoints(self):
        """Generate circular formation waypoints around target."""
        r = self.formation_config.formation_radius
        n = self.num_inspectors

        for i in range(n):
            angle = np.radians(i * self.formation_config.angular_separation)

            # Position on circle in XY plane
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = 0.0

            self.waypoints[i] = [np.array([x, y, z])]

    def get_formation_target(self, inspector_id: int) -> np.ndarray:
        """Get target position for inspector in formation."""
        if inspector_id not in self.waypoints or not self.waypoints[inspector_id]:
            return np.zeros(3)
        return self.waypoints[inspector_id][0]

    def update_inspector(
        self,
        inspector_id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
        wheel_speeds: np.ndarray,
    ):
        """Update state of an inspector satellite."""
        if inspector_id not in self.inspectors:
            self.inspectors[inspector_id] = InspectorState(
                id=inspector_id,
                position=position,
                velocity=velocity,
                quaternion=quaternion,
                angular_velocity=angular_velocity,
                wheel_speeds=wheel_speeds,
            )
        else:
            state = self.inspectors[inspector_id]
            state.position = position
            state.velocity = velocity
            state.quaternion = quaternion
            state.angular_velocity = angular_velocity
            state.wheel_speeds = wheel_speeds

    def get_obstacles(self, inspector_id: int) -> List[Tuple[np.ndarray, float]]:
        """
        Get list of obstacles for an inspector to avoid.

        Returns:
            List of (position, radius) tuples for each obstacle
        """
        obstacles = []

        # Target satellite keep-out zone
        obstacles.append((np.zeros(3), self.obstacle_config.target_keep_out_radius))

        # Other inspectors
        for other_id, other_state in self.inspectors.items():
            if other_id != inspector_id:
                obstacles.append(
                    (other_state.position, self.obstacle_config.min_inspector_separation)
                )

        return obstacles

    def check_collisions(self) -> Dict[str, bool]:
        """
        Check for collision violations.

        Returns:
            Dict with violation flags
        """
        violations = {
            "target_keep_out": False,
            "inspector_collision": False,
        }

        keep_out = self.obstacle_config.target_keep_out_radius
        min_sep = self.obstacle_config.min_inspector_separation

        inspector_ids = list(self.inspectors.keys())

        for i, id1 in enumerate(inspector_ids):
            state1 = self.inspectors[id1]

            # Check target keep-out
            if state1.distance_to_origin() < keep_out:
                violations["target_keep_out"] = True

            # Check inter-inspector separation
            for id2 in inspector_ids[i + 1 :]:
                state2 = self.inspectors[id2]
                if state1.distance_to(state2) < min_sep:
                    violations["inspector_collision"] = True

        return violations

    def get_min_separations(self) -> Dict[str, float]:
        """
        Get minimum separation distances.

        Returns:
            Dict with 'to_target' and 'inter_inspector' min distances
        """
        min_to_target = float("inf")
        min_inter = float("inf")

        inspector_ids = list(self.inspectors.keys())

        for i, id1 in enumerate(inspector_ids):
            state1 = self.inspectors[id1]

            # Distance to target
            dist = state1.distance_to_origin()
            min_to_target = min(min_to_target, dist)

            # Inter-inspector distance
            for id2 in inspector_ids[i + 1 :]:
                state2 = self.inspectors[id2]
                dist = state1.distance_to(state2)
                min_inter = min(min_inter, dist)

        return {
            "to_target": min_to_target if min_to_target < float("inf") else 0.0,
            "inter_inspector": min_inter if min_inter < float("inf") else 0.0,
        }

    def compute_obstacle_penalty(
        self,
        inspector_id: int,
        position: np.ndarray,
    ) -> float:
        """
        Compute obstacle avoidance penalty for MPC cost function.

        Args:
            inspector_id: ID of inspector
            position: Candidate position to evaluate

        Returns:
            Penalty value (0 if no violations)
        """
        penalty = 0.0
        obstacles = self.get_obstacles(inspector_id)
        margin = self.obstacle_config.soft_margin
        weight = self.obstacle_config.avoidance_penalty

        for obs_pos, obs_radius in obstacles:
            dist = np.linalg.norm(position - obs_pos)
            safe_dist = obs_radius + margin

            if dist < safe_dist:
                # Quadratic penalty inside margin
                violation = safe_dist - dist
                penalty += weight * violation**2

        return penalty


def create_fleet_manager(
    num_inspectors: int = 3,
    formation_radius: float = 5.0,
) -> FleetManager:
    """Create a fleet manager with default configuration."""
    return FleetManager(
        num_inspectors=num_inspectors,
        formation_config=FormationConfig(
            num_inspectors=num_inspectors,
            formation_radius=formation_radius,
        ),
    )
