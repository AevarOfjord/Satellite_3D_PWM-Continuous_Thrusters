"""
Navigation and Geometry Utilities for Satellite Control

Shared mathematical and geometric utility functions for satellite navigation.
Used by both real hardware and simulation controllers for consistent behavior.

Utility functions:
- Angle normalization and shortest-path difference calculation
- Point-to-line distance for path tracking
- Safe path generation with obstacle avoidance
- Linear and circular interpolation for trajectories
- Geometric calculations for waypoint navigation

Key features:
- Handles angle wrapping around ±π correctly
- Prevents 270° transition issues with shortest-path logic
- Circular obstacle avoidance for safe navigation
- Shared between simulation and hardware for consistency
"""

from typing import List, Tuple

import numpy as np


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi] range.

    Args:
        angle: Angle in radians (any range)

    Returns:
        Normalized angle in [-pi, pi] range
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def angle_difference(target_angle: float, current_angle: float) -> float:
    """
    Calculate the shortest angular difference between target and current angles.
    This prevents the 270° transition issue by always taking the shortest path.

    Args:
        target_angle: Target orientation in radians
        current_angle: Current orientation in radians

    Returns:
        Angle difference in [-pi, pi] range, positive = CCW rotation needed
    """
    # Normalize both angles first
    target = normalize_angle(target_angle)
    current = normalize_angle(current_angle)

    # Calculate raw difference
    diff = target - current

    # Ensure we take the shortest path around the circle
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi

    return diff


def point_to_line_distance(
    point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
) -> float:
    """
    Calculate the shortest distance from a point to a line segment.

    Args:
        point: Point position as [x, y] numpy array
        line_start: Line segment start position as [x, y] numpy array
        line_end: Line segment end position as [x, y] numpy array

    Returns:
        Shortest distance from point to line segment in meters
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)

    # Handle zero-length line segment
    if line_len < 1e-10:
        return float(np.linalg.norm(point_vec))

    line_unitvec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unitvec)

    # Clamp to line segment
    proj_length = max(0, min(line_len, proj_length))
    proj_point = line_start + line_unitvec * proj_length

    return float(np.linalg.norm(point - proj_point))


def calculate_safe_path_to_waypoint(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    all_obstacles: List[Tuple[float, float]],
    safety_radius: float,
) -> List[Tuple[float, float]]:
    """
    Calculate a safe path from start to target that avoids circular obstacles.

    Uses a simple single-waypoint strategy: if the direct path crosses an obstacle,
    generates an intermediate waypoint that goes around it.

    Args:
        start_pos: Starting position as [x, y] numpy array
        target_pos: Target position as [x, y] numpy array
        all_obstacles: List of (x, y) obstacle center positions
        safety_radius: Minimum distance to maintain from obstacle centers

    Returns:
        List of waypoints [(x, y), ...] from start to target
    """
    # Simple approach: If direct path crosses obstacles, generate intermediate
    # waypoint
    # Determine Z coordinates
    start_z = start_pos[2] if len(start_pos) > 2 else 0.0
    target_z = target_pos[2] if len(target_pos) > 2 else 0.0

    # Work in 2D for obstacle avoidance (Cylindrical obstacles)
    start_2d = start_pos[:2]
    target_2d = target_pos[:2]

    waypoints = [tuple(start_pos)]

    blocking_obstacles = []
    for obstacle_center_tuple in all_obstacles:
        obstacle_center = np.array(obstacle_center_tuple)
        distance_to_path = point_to_line_distance(obstacle_center, start_2d, target_2d)
        if distance_to_path < safety_radius:
            blocking_obstacles.append(obstacle_center)

    if blocking_obstacles:
        # For simplicity, create a waypoint that goes around the closest
        # blocking obstacle
        closest_obstacle = min(
            blocking_obstacles,
            key=lambda obs: float(np.linalg.norm(obs - start_2d)),
        )

        # Check if obstacle is at start position (edge case)
        obstacle_distance = np.linalg.norm(closest_obstacle - start_2d)

        # Linearly interpolate Z for the intermediate waypoint based on distance ratio
        # But for simplicity, just average Z or keep start Z seems safer for collision avoidance
        # Let's average Z
        avg_z = (start_z + target_z) / 2.0

        if obstacle_distance < 1e-10:
            # Obstacle is at start - move perpendicular to target direction
            direction_to_target = target_2d - start_2d
            target_norm = np.linalg.norm(direction_to_target)
            if target_norm > 1e-10:
                direction_to_target_norm = direction_to_target / target_norm
                # Move perpendicular to get away from obstacle
                perpendicular = np.array(
                    [-direction_to_target_norm[1], direction_to_target_norm[0]]
                )
                intermediate_2d = start_2d + perpendicular * safety_radius * 1.5
            else:
                # Both at start and target at start - just offset in x
                # direction
                intermediate_2d = start_2d + np.array([safety_radius * 1.5, 0])
        else:
            # Normal case: create intermediate waypoint by going around the
            # obstacle
            direction_to_target = target_2d - start_2d

            # Helper to avoid div by zero
            denom = np.linalg.norm(direction_to_target)
            if denom < 1e-10:
                direction_to_target_norm = np.array([1.0, 0.0])
            else:
                direction_to_target_norm = direction_to_target / denom

            direction_to_obstacle = closest_obstacle - start_2d
            direction_to_obstacle_norm = direction_to_obstacle / obstacle_distance

            # Calculate perpendicular direction
            perpendicular = np.array(
                [-direction_to_obstacle_norm[1], direction_to_obstacle_norm[0]]
            )

            # Create waypoint that goes around the obstacle
            # Use constant from obstacles module (not mutable, so safe to use directly)
            from src.satellite_control.config.obstacles import OBSTACLE_AVOIDANCE_SAFETY_MARGIN
            
            intermediate_2d = closest_obstacle + perpendicular * (
                safety_radius + OBSTACLE_AVOIDANCE_SAFETY_MARGIN
            )

        # Construct 3D intermediate waypoint
        intermediate_waypoint = np.array([intermediate_2d[0], intermediate_2d[1], avg_z])

        waypoints.append(tuple(intermediate_waypoint))
        wp_x, wp_y = intermediate_waypoint[0], intermediate_waypoint[1]
        print(
            f" Generated intermediate waypoint to avoid obstacle: "
            f"({wp_x:.3f}, {wp_y:.3f}, {avg_z:.3f})"
        )

    waypoints.append(tuple(target_pos))
    return waypoints
