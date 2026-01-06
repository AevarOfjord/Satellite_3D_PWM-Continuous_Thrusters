"""
Spline Path Generator for Smooth Obstacle Avoidance

Generates smooth Bezier curves around obstacles with arc-length
parameterization for constant-speed traversal.
"""

from typing import Optional, Tuple

import numpy as np


class ObstacleAvoidanceSpline:
    """
    Cubic Bezier spline for smooth obstacle avoidance.

    Generates a smooth curve from start to end that passes through
    a control point (typically placed to avoid an obstacle).
    Uses arc-length parameterization for constant-speed sampling.
    """

    def __init__(
        self,
        start: Tuple[float, float],
        control: Tuple[float, float],
        end: Tuple[float, float],
        num_samples: int = 100,
    ):
        """
        Initialize spline with start, control point, and end.

        Args:
            start: Starting position (x, y)
            control: Control point to curve around (x, y)
            end: Ending position (x, y)
            num_samples: Number of samples for arc-length table
        """
        self.start = np.array(start)
        self.control = np.array(control)
        self.end = np.array(end)
        self.num_samples = num_samples

        # Build arc-length lookup table
        self._build_arc_length_table()

    def _quadratic_bezier(self, t: float) -> np.ndarray:
        """Evaluate quadratic Bezier curve at parameter t."""
        # B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        result: np.ndarray = (
            (1 - t) ** 2 * self.start + 2 * (1 - t) * t * self.control + t**2 * self.end
        )
        return result

    def _quadratic_bezier_derivative(self, t: float) -> np.ndarray:
        """Derivative of quadratic Bezier curve at parameter t."""
        # B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
        result: np.ndarray = 2 * (1 - t) * (self.control - self.start) + 2 * t * (
            self.end - self.control
        )
        return result

    def _build_arc_length_table(self) -> None:
        """Build lookup table mapping arc length to parameter t."""
        self._t_values = np.linspace(0, 1, self.num_samples)
        self._arc_lengths = np.zeros(self.num_samples)

        prev_point = self.start
        cumulative_length = 0.0

        for i, t in enumerate(self._t_values):
            point = self._quadratic_bezier(t)
            cumulative_length += float(np.linalg.norm(point - prev_point))
            self._arc_lengths[i] = cumulative_length
            prev_point = point

        self._total_length = cumulative_length

    @property
    def total_length(self) -> float:
        """Total arc length of the spline."""
        return self._total_length

    def _arc_length_to_t(self, s: float) -> float:
        """Convert arc length s to parameter t using interpolation."""
        if s <= 0:
            return 0.0
        if s >= self._total_length:
            return 1.0

        # Binary search / interpolation
        idx = np.searchsorted(self._arc_lengths, s)
        if idx == 0:
            return 0.0

        # Linear interpolation between samples
        s0 = self._arc_lengths[idx - 1]
        s1 = self._arc_lengths[idx]
        t0 = self._t_values[idx - 1]
        t1 = self._t_values[idx]

        if s1 - s0 < 1e-10:
            return float(t0)

        alpha = (s - s0) / (s1 - s0)
        return float(t0 + alpha * (t1 - t0))

    def evaluate(self, arc_length: float) -> Tuple[float, float]:
        """
        Evaluate position at given arc length.

        Args:
            arc_length: Distance along curve from start

        Returns:
            (x, y) position on curve
        """
        t = self._arc_length_to_t(arc_length)
        point = self._quadratic_bezier(t)
        return (float(point[0]), float(point[1]))

    def tangent(self, arc_length: float) -> Tuple[float, float]:
        """
        Get normalized tangent direction at given arc length.

        Args:
            arc_length: Distance along curve from start

        Returns:
            (dx, dy) normalized tangent vector
        """
        t = self._arc_length_to_t(arc_length)
        deriv = self._quadratic_bezier_derivative(t)
        norm = np.linalg.norm(deriv)
        if norm < 1e-10:
            # At endpoints, use direction to next point
            if t < 0.5:
                direction = self.control - self.start
            else:
                direction = self.end - self.control
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                return (1.0, 0.0)
            return (float(direction[0] / norm), float(direction[1] / norm))

        return (float(deriv[0] / norm), float(deriv[1] / norm))

    def is_complete(self, arc_length: float) -> bool:
        """Check if arc length has reached end of spline."""
        return arc_length >= self._total_length

    def get_progress_fraction(self, arc_length: float) -> float:
        """Get progress as fraction (0 to 1)."""
        if self._total_length < 1e-10:
            return 1.0
        return min(1.0, arc_length / self._total_length)


def create_obstacle_avoidance_spline(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    safety_margin: float = 0.1,
) -> Optional[ObstacleAvoidanceSpline]:
    """
    Create a spline that curves around an obstacle.

    Args:
        start_pos: Starting position [x, y]
        target_pos: Target position [x, y]
        obstacle_center: Obstacle center [x, y]
        obstacle_radius: Obstacle radius
        safety_margin: Extra clearance

    Returns:
        ObstacleAvoidanceSpline or None if no obstacle in path
    """
    start = np.array(start_pos[:2])
    target = np.array(target_pos[:2])
    obstacle = np.array(obstacle_center[:2])

    # Check if obstacle is actually in the path
    # Project obstacle onto line from start to target
    path_vec = target - start
    path_length = np.linalg.norm(path_vec)
    if path_length < 1e-6:
        return None

    path_dir = path_vec / path_length
    to_obstacle = obstacle - start
    projection = np.dot(to_obstacle, path_dir)

    # Obstacle not between start and target
    if projection < 0 or projection > path_length:
        return None

    # Distance from obstacle to path line
    closest_on_path = start + projection * path_dir
    distance_to_path = np.linalg.norm(obstacle - closest_on_path)

    effective_radius = obstacle_radius + safety_margin

    # Path doesn't intersect obstacle
    if distance_to_path > effective_radius:
        return None

    # Calculate control point to go around obstacle
    # Perpendicular direction
    perp = np.array([-path_dir[1], path_dir[0]])

    # Determine which side to go around (choose side closer to target)
    test_point1 = obstacle + perp * effective_radius
    test_point2 = obstacle - perp * effective_radius

    if np.linalg.norm(test_point1 - target) < np.linalg.norm(test_point2 - target):
        control_point = obstacle + perp * (effective_radius * 1.5)
    else:
        control_point = obstacle - perp * (effective_radius * 1.5)

    return ObstacleAvoidanceSpline(
        start=tuple(start), control=tuple(control_point), end=tuple(target)
    )
