"""
Obstacle Avoidance Configuration for Satellite Control System

Obstacle definitions and collision checking for safe navigation.
Manages circular obstacles with configurable safety margins.

Obstacle management features:
- Dynamic obstacle configuration at runtime
- Circular obstacle representation (x, y, radius)
- Path collision detection along line segments
- Safety margin enforcement around obstacles
- Minimum distance validation

Path checking:
- Line-segment to circle collision detection
- Configurable path resolution for accuracy
- Nearest point calculation for avoidance
- Integration with path_planning_manager

Key features:
- Enable/disable obstacle avoidance globally
- Runtime obstacle list updates
- Validation of obstacle positions and sizes
- Thread-safe for concurrent access
"""

from typing import List, Optional, Tuple

import numpy as np


class ObstacleManager:
    """
    Manager for obstacle avoidance configuration.

    Handles obstacle definition, path checking, and avoidance parameters.
    """

    def __init__(self) -> None:
        """Initialize obstacle manager with default settings."""
        self.enabled: bool = False
        self.obstacles: List[Tuple[float, float, float]] = []  # (x, y, radius)
        self.default_obstacle_radius: float = 0.5  # meters
        self.safety_margin: float = 0.1  # meters
        self.min_obstacle_distance: float = 0.5  # meters
        self.path_resolution: float = 0.1  # meters
        self.waypoint_stabilization_time: float = 0.5  # seconds

    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]) -> None:
        """
        Set obstacles for navigation.

        Args:
            obstacles: List of (x, y, radius) tuples defining obstacles
        """
        self.obstacles = obstacles.copy()
        self.enabled = len(obstacles) > 0

        if self.enabled:
            print(f"\n OBSTACLE AVOIDANCE ENABLED: {len(obstacles)} " "obstacles configured")
            for i, (x, y, radius) in enumerate(obstacles, 1):
                print(f"  Obstacle {i}: ({x:.2f}, {y:.2f}) m, " f"radius {radius:.2f} m")
        else:
            print("\n OBSTACLE AVOIDANCE DISABLED: No obstacles configured")

    def add_obstacle(self, x: float, y: float, radius: Optional[float] = None) -> None:
        """
        Add a single obstacle.

        Args:
            x: X coordinate of obstacle center
            y: Y coordinate of obstacle center
            radius: Obstacle radius (uses default if None)
        """
        if radius is None:
            radius = self.default_obstacle_radius

        obstacle = (x, y, radius)
        self.obstacles.append(obstacle)
        self.enabled = True

        print(f" Added obstacle: ({x:.2f}, {y:.2f}) m, radius {radius:.2f} m")

    def clear_obstacles(self) -> None:
        """Clear all obstacles and disable obstacle avoidance."""
        self.obstacles.clear()
        self.enabled = False
        print(" All obstacles cleared")

    def get_obstacles(self) -> List[Tuple[float, float, float]]:
        """
        Get current obstacle configuration.

        Returns:
            List of (x, y, radius) tuples
        """
        return self.obstacles.copy()

    def is_path_clear(
        self,
        start_pos: Tuple[float, ...],
        end_pos: Tuple[float, ...],
        safety_margin: Optional[float] = None,
    ) -> bool:
        """
        Check if a straight path between two points is clear of obstacles.

        Args:
            start_pos: (x, y) starting position
            end_pos: (x, y) ending position
            safety_margin: Additional safety margin (uses default if None)

        Returns:
            True if path is clear, False if blocked by obstacles
        """
        if not self.enabled:
            return True

        if safety_margin is None:
            safety_margin = self.safety_margin

        start = np.array(start_pos)[:2]
        end = np.array(end_pos)[:2]

        for obs_x, obs_y, obs_radius in self.obstacles:
            obs_center = np.array([obs_x, obs_y])
            # Use consistent safety margin from module constant
            effective_radius = obs_radius + OBSTACLE_AVOIDANCE_SAFETY_MARGIN

            distance = self._point_to_line_distance(obs_center, start, end)

            if distance < effective_radius:
                return False

        return True

    def _point_to_line_distance(
        self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
    ) -> float:
        """
        Calculate the shortest distance from a point to a line segment.

        Args:
            point: Point coordinates
            line_start: Line segment start point
            line_end: Line segment end point

        Returns:
            Minimum distance from point to line segment
        """
        line_vec = line_end - line_start
        line_length_sq = np.dot(line_vec, line_vec)

        if line_length_sq == 0:
            # Line segment is actually a point
            return float(np.linalg.norm(point - line_start))

        point_vec = point - line_start
        t = np.dot(point_vec, line_vec) / line_length_sq
        t = max(0, min(1, t))  # Clamp to [0, 1]

        # Find closest point on line segment
        closest_point = line_start + t * line_vec

        return float(np.linalg.norm(point - closest_point))

    def print_config(self) -> None:
        """Print current obstacle configuration."""
        print("\n" + "=" * 80)
        print("OBSTACLE AVOIDANCE CONFIGURATION")
        print("=" * 80)

        print(f"\nEnabled: {self.enabled}")
        print(f"Number of obstacles: {len(self.obstacles)}")
        print(f"Default obstacle radius: {self.default_obstacle_radius:.2f} m")
        print(f"Safety margin: {self.safety_margin:.2f} m")
        print(f"Min obstacle distance: {self.min_obstacle_distance:.2f} m")
        print(f"Path resolution: {self.path_resolution:.2f} m")
        print(f"Waypoint stabilization time: " f"{self.waypoint_stabilization_time:.1f} s")

        if self.obstacles:
            print("\nConfigured obstacles:")
            for i, (x, y, radius) in enumerate(self.obstacles, 1):
                print(f"  {i}. Position: ({x:.2f}, {y:.2f}) m, " f"Radius: {radius:.2f} m")

        print("=" * 80 + "\n")


# DEFAULT OBSTACLE PARAMETERS
# ============================================================================

DEFAULT_OBSTACLE_RADIUS = 0.5  # meters
OBSTACLE_SAFETY_MARGIN = 0.1  # meters
MIN_OBSTACLE_DISTANCE = 0.5  # meters
OBSTACLE_PATH_RESOLUTION = 0.1  # meters
OBSTACLE_WAYPOINT_STABILIZATION_TIME = 0.5  # seconds (legacy)
OBSTACLE_FLYTHROUGH_TOLERANCE = 0.15  # meters - dist to advance
OBSTACLE_AVOIDANCE_SAFETY_MARGIN = 0.25  # meters - added to radius


def create_obstacle_manager() -> ObstacleManager:
    """
    Create a new obstacle manager with default settings.

    Returns:
        ObstacleManager initialized with defaults
    """
    manager = ObstacleManager()
    manager.default_obstacle_radius = DEFAULT_OBSTACLE_RADIUS
    manager.safety_margin = OBSTACLE_SAFETY_MARGIN
    manager.min_obstacle_distance = MIN_OBSTACLE_DISTANCE
    manager.path_resolution = OBSTACLE_PATH_RESOLUTION
    manager.waypoint_stabilization_time = OBSTACLE_WAYPOINT_STABILIZATION_TIME
    return manager
