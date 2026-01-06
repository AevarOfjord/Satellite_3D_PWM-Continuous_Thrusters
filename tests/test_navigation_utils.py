"""
Unit tests for navigation_utils.py module.

Tests navigation and geometry utility functions used across
the satellite control system.
"""

import numpy as np
import pytest
from typing import List, Tuple

from src.satellite_control.utils.navigation_utils import (
    normalize_angle,
    angle_difference,
    point_to_line_distance,
    calculate_safe_path_to_waypoint,
)


class TestNormalizeAngle:
    """Test angle normalization to [-pi, pi] range."""

    def test_zero_angle(self):
        """Test that zero angle remains zero."""
        assert abs(normalize_angle(0.0)) < 1e-10

    def test_positive_angles_in_range(self):
        """Test that positive angles already in range are unchanged."""
        assert abs(normalize_angle(np.pi / 4) - np.pi / 4) < 1e-10
        assert abs(normalize_angle(np.pi / 2) - np.pi / 2) < 1e-10
        assert abs(normalize_angle(3 * np.pi / 4) - 3 * np.pi / 4) < 1e-10

    def test_negative_angles_in_range(self):
        """Test that negative angles already in range are unchanged."""
        assert abs(normalize_angle(-np.pi / 4) + np.pi / 4) < 1e-10
        assert abs(normalize_angle(-np.pi / 2) + np.pi / 2) < 1e-10
        assert abs(normalize_angle(-3 * np.pi / 4) + 3 * np.pi / 4) < 1e-10

    def test_angle_exactly_pi(self):
        """Test normalization of angle exactly at pi."""
        result = normalize_angle(np.pi)
        assert abs(result - np.pi) < 1e-10

    def test_angle_exactly_negative_pi(self):
        """Test normalization of angle exactly at -pi."""
        result = normalize_angle(-np.pi)
        assert abs(result + np.pi) < 1e-10

    def test_angle_just_over_pi(self):
        """Test normalization of angle just over pi."""
        result = normalize_angle(np.pi + 0.1)
        expected = -np.pi + 0.1
        assert abs(result - expected) < 1e-10

    def test_angle_just_under_negative_pi(self):
        """Test normalization of angle just under -pi."""
        result = normalize_angle(-np.pi - 0.1)
        expected = np.pi - 0.1
        assert abs(result - expected) < 1e-10

    def test_angle_2pi(self):
        """Test normalization of 2*pi wraps to zero."""
        result = normalize_angle(2 * np.pi)
        assert abs(result) < 1e-10

    def test_angle_negative_2pi(self):
        """Test normalization of -2*pi wraps to zero."""
        result = normalize_angle(-2 * np.pi)
        assert abs(result) < 1e-10

    def test_angle_3pi(self):
        """Test normalization of 3*pi wraps to pi."""
        result = normalize_angle(3 * np.pi)
        assert abs(result - np.pi) < 1e-10 or abs(result + np.pi) < 1e-10

    def test_large_positive_angle(self):
        """Test normalization of very large positive angle."""
        result = normalize_angle(10 * np.pi)
        assert abs(result) < 1e-10

    def test_large_negative_angle(self):
        """Test normalization of very large negative angle."""
        result = normalize_angle(-10 * np.pi)
        assert abs(result) < 1e-10

    def test_fractional_rotation(self):
        """Test normalization of fractional rotations."""
        result = normalize_angle(2.5 * np.pi)
        expected = 0.5 * np.pi
        assert abs(result - expected) < 1e-10


class TestAngleDifference:
    """Test angle difference calculation."""

    def test_zero_difference(self):
        """Test that identical angles have zero difference."""
        assert abs(angle_difference(0.0, 0.0)) < 1e-10
        assert abs(angle_difference(np.pi / 2, np.pi / 2)) < 1e-10

    def test_simple_positive_difference(self):
        """Test simple positive angle difference."""
        result = angle_difference(np.pi / 2, 0.0)
        assert abs(result - np.pi / 2) < 1e-10

    def test_simple_negative_difference(self):
        """Test simple negative angle difference."""
        result = angle_difference(0.0, np.pi / 2)
        assert abs(result + np.pi / 2) < 1e-10

    def test_shortest_path_over_pi(self):
        """Test that difference takes shortest path over pi boundary."""
        # From 3pi/4 (135°) to -3pi/4 (-135°=225°), shortest path is +90° CCW
        result = angle_difference(-3 * np.pi / 4, 3 * np.pi / 4)
        # Shortest path should be positive (CCW)
        assert result > 0
        assert abs(result - np.pi / 2) < 1e-10

    def test_shortest_path_under_negative_pi(self):
        """Test that difference takes shortest path under -pi boundary."""
        # From -3pi/4 (-135°=225°) to 3pi/4 (135°), shortest path is -90° CW
        result = angle_difference(3 * np.pi / 4, -3 * np.pi / 4)
        # Shortest path should be negative (CW)
        assert result < 0
        assert abs(abs(result) - np.pi / 2) < 1e-10

    def test_pi_to_negative_pi(self):
        """Test difference from pi to -pi (same angle)."""
        result = angle_difference(np.pi, -np.pi)
        assert abs(result) < 1e-10

    def test_small_angle_near_zero(self):
        """Test small angle differences near zero."""
        result = angle_difference(0.1, -0.1)
        assert abs(result - 0.2) < 1e-10

    def test_quarter_turn_ccw(self):
        """Test quarter turn counter-clockwise."""
        result = angle_difference(np.pi / 2, 0.0)
        assert abs(result - np.pi / 2) < 1e-10

    def test_quarter_turn_cw(self):
        """Test quarter turn clockwise."""
        result = angle_difference(0.0, np.pi / 2)
        assert abs(result + np.pi / 2) < 1e-10

    def test_half_turn(self):
        """Test half turn (180 degrees)."""
        result = angle_difference(np.pi, 0.0)
        assert abs(abs(result) - np.pi) < 1e-10

    def test_unnormalized_inputs(self):
        """Test that unnormalized input angles are handled correctly."""
        # 2*pi and 0 should be the same angle
        result = angle_difference(2 * np.pi, 0.0)
        assert abs(result) < 1e-10

        # 3*pi and pi should be the same angle
        result = angle_difference(3 * np.pi, np.pi)
        assert abs(result) < 1e-10


class TestPointToLineDistance:
    """Test point-to-line-segment distance calculation."""

    def test_point_on_line(self):
        """Test distance when point is exactly on the line."""
        point = np.array([0.5, 0.5])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 1.0])

        distance = point_to_line_distance(point, line_start, line_end)
        assert abs(distance) < 1e-10

    def test_point_perpendicular_to_horizontal_line(self):
        """Test distance when point is perpendicular to horizontal line."""
        point = np.array([0.5, 1.0])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 0.0])

        distance = point_to_line_distance(point, line_start, line_end)
        assert abs(distance - 1.0) < 1e-10

    def test_point_perpendicular_to_vertical_line(self):
        """Test distance when point is perpendicular to vertical line."""
        point = np.array([1.0, 0.5])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([0.0, 1.0])

        distance = point_to_line_distance(point, line_start, line_end)
        assert abs(distance - 1.0) < 1e-10

    def test_point_at_line_start(self):
        """Test distance when point is at line start."""
        point = np.array([0.0, 0.0])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 0.0])

        distance = point_to_line_distance(point, line_start, line_end)
        assert abs(distance) < 1e-10

    def test_point_at_line_end(self):
        """Test distance when point is at line end."""
        point = np.array([1.0, 0.0])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 0.0])

        distance = point_to_line_distance(point, line_start, line_end)
        assert abs(distance) < 1e-10

    def test_point_beyond_line_end(self):
        """Test distance when point is beyond line segment end."""
        point = np.array([2.0, 0.0])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 0.0])

        distance = point_to_line_distance(point, line_start, line_end)
        # Should be distance to closest endpoint (1.0)
        assert abs(distance - 1.0) < 1e-10

    def test_point_before_line_start(self):
        """Test distance when point is before line segment start."""
        point = np.array([-1.0, 0.0])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 0.0])

        distance = point_to_line_distance(point, line_start, line_end)
        # Should be distance to closest endpoint (1.0)
        assert abs(distance - 1.0) < 1e-10

    def test_zero_length_line(self):
        """Test distance when line segment has zero length (single point)."""
        point = np.array([1.0, 1.0])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([0.0, 0.0])

        distance = point_to_line_distance(point, line_start, line_end)
        # Should be Euclidean distance to the point
        expected = np.sqrt(2)
        assert abs(distance - expected) < 1e-10

    def test_diagonal_line(self):
        """Test distance to diagonal line."""
        point = np.array([0.0, 1.0])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 0.0])

        distance = point_to_line_distance(point, line_start, line_end)
        assert abs(distance - 1.0) < 1e-10

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        point = np.array([-1.0, -1.0])
        line_start = np.array([-2.0, -2.0])
        line_end = np.array([0.0, 0.0])

        # Point is on the line
        distance = point_to_line_distance(point, line_start, line_end)
        assert abs(distance) < 1e-10


class TestCalculateSafePathToWaypoint:
    """Test safe path calculation with obstacle avoidance."""

    def test_direct_path_no_obstacles(self):
        """Test that direct path is used when no obstacles block the way."""
        start_pos = np.array([0.0, 0.0])
        target_pos = np.array([1.0, 1.0])
        obstacles: List[Tuple[float, float]] = []
        safety_radius = 0.2

        path = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, safety_radius
        )

        # Should be direct path: start -> target
        assert len(path) == 2
        assert path[0] == (0.0, 0.0)
        assert path[1] == (1.0, 1.0)

    def test_direct_path_with_distant_obstacles(self):
        """Test direct path when obstacles are far from path."""
        start_pos = np.array([0.0, 0.0])
        target_pos = np.array([1.0, 0.0])
        obstacles: List[Tuple[float, float]] = [(0.5, 5.0)]  # Obstacle far above path
        safety_radius = 0.2

        path = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, safety_radius
        )

        # Should still be direct path
        assert len(path) == 2

    def test_path_with_blocking_obstacle(self):
        """Test that intermediate waypoint is generated when obstacle blocks path."""
        start_pos = np.array([0.0, 0.0])
        target_pos = np.array([1.0, 0.0])
        obstacles: List[Tuple[float, float]] = [(0.5, 0.0)]  # Obstacle directly on path
        safety_radius = 0.2

        path = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, safety_radius
        )

        # Should have intermediate waypoint: start -> intermediate -> target
        assert len(path) == 3
        assert path[0] == (0.0, 0.0)
        assert path[2] == (1.0, 0.0)
        # Intermediate waypoint should be offset from obstacle
        intermediate = np.array(path[1])
        obstacle = np.array(obstacles[0])
        distance_to_obstacle = np.linalg.norm(intermediate - obstacle)
        assert distance_to_obstacle >= safety_radius

    def test_path_with_near_miss_obstacle(self):
        """Test when obstacle is just barely within safety radius."""
        start_pos = np.array([0.0, 0.0])
        target_pos = np.array([1.0, 0.0])
        obstacles: List[Tuple[float, float]] = [
            (0.5, 0.15)
        ]  # Just within safety radius of 0.2
        safety_radius = 0.2

        path = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, safety_radius
        )

        # Should generate intermediate waypoint
        assert len(path) == 3

    def test_path_with_multiple_obstacles(self):
        """Test path generation with multiple obstacles."""
        start_pos = np.array([0.0, 0.0])
        target_pos = np.array([1.0, 0.0])
        obstacles: List[Tuple[float, float]] = [
            (0.3, 0.0),  # First obstacle on path
            (0.7, 0.0),  # Second obstacle on path
        ]
        safety_radius = 0.2

        path = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, safety_radius
        )

        # Should generate at least one intermediate waypoint
        # (Current implementation creates waypoint for closest obstacle)
        assert len(path) >= 3

    def test_obstacle_at_target(self):
        """Test behavior when obstacle is at target location."""
        start_pos = np.array([0.0, 0.0])
        target_pos = np.array([1.0, 0.0])
        obstacles: List[Tuple[float, float]] = [(1.0, 0.0)]  # Obstacle at target
        safety_radius = 0.2

        path = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, safety_radius
        )

        # Should still generate a path (though may not be physically feasible)
        assert len(path) >= 2
        assert path[-1] == (1.0, 0.0)  # Final waypoint should still be target

    def test_obstacle_at_start(self):
        """Test behavior when obstacle is at start location."""
        start_pos = np.array([0.0, 0.0])
        target_pos = np.array([1.0, 0.0])
        obstacles: List[Tuple[float, float]] = [(0.0, 0.0)]  # Obstacle at start
        safety_radius = 0.2

        path = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, safety_radius
        )

        # Should generate intermediate waypoint
        assert len(path) >= 3

    def test_waypoint_coordinates_format(self):
        """Test that waypoints are returned as tuples."""
        start_pos = np.array([0.0, 0.0])
        target_pos = np.array([1.0, 1.0])
        obstacles: List[Tuple[float, float]] = []
        safety_radius = 0.2

        path = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, safety_radius
        )

        # All waypoints should be tuples
        for waypoint in path:
            assert isinstance(waypoint, tuple)
            assert len(waypoint) == 2

    def test_path_starts_at_start_pos(self):
        """Test that path always starts at start position."""
        start_pos = np.array([0.5, 0.5])
        target_pos = np.array([1.0, 1.0])
        obstacles: List[Tuple[float, float]] = []
        safety_radius = 0.2

        path = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, safety_radius
        )

        assert path[0] == (0.5, 0.5)

    def test_path_ends_at_target_pos(self):
        """Test that path always ends at target position."""
        start_pos = np.array([0.0, 0.0])
        target_pos = np.array([2.5, 3.7])
        obstacles: List[Tuple[float, float]] = []
        safety_radius = 0.2

        path = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, safety_radius
        )

        assert path[-1] == (2.5, 3.7)

    def test_safety_radius_scaling(self):
        """Test that larger safety radius generates waypoints further from obstacles."""
        start_pos = np.array([0.0, 0.0])
        target_pos = np.array([1.0, 0.0])
        obstacles: List[Tuple[float, float]] = [(0.5, 0.0)]

        # Try with different safety radii
        path_small = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, 0.1
        )
        path_large = calculate_safe_path_to_waypoint(
            start_pos, target_pos, obstacles, 0.5
        )

        # Both should avoid obstacle
        assert len(path_small) == 3
        assert len(path_large) == 3

        # Larger safety radius should push waypoint further from obstacle
        if len(path_small) > 2 and len(path_large) > 2:
            intermediate_small = np.array(path_small[1])
            intermediate_large = np.array(path_large[1])
            obstacle_pos = np.array(obstacles[0])

            dist_small = np.linalg.norm(intermediate_small - obstacle_pos)
            dist_large = np.linalg.norm(intermediate_large - obstacle_pos)

            assert dist_large > dist_small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
