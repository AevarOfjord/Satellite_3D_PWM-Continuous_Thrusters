"""
Unit tests for mission_state_manager.py module.

Tests the MissionStateManager class which centralizes mission logic
for all mission types across simulation and real hardware testing.
"""

import numpy as np
import pytest

from src.satellite_control.mission.mission_state_manager import MissionStateManager


class TestMissionStateManagerInitialization:
    """Test MissionStateManager initialization."""

    def test_default_initialization(self):
        """Test that MissionStateManager can be initialized with defaults."""
        manager = MissionStateManager()

        assert manager.position_tolerance == 0.05
        assert manager.angle_tolerance == 0.05
        assert manager.current_nav_waypoint_idx == 0
        assert manager.dxf_completed is False
        assert manager.multi_point_target_reached_time is None

    def test_custom_tolerances(self):
        """Test initialization with custom tolerances."""
        manager = MissionStateManager(position_tolerance=0.1, angle_tolerance=0.2)

        assert manager.position_tolerance == 0.1
        assert manager.angle_tolerance == 0.2

    def test_custom_functions(self):
        """Test initialization with custom helper functions."""

        def custom_normalize(angle):
            return angle % (2 * np.pi)

        def custom_angle_diff(angle1, angle2):
            return angle1 - angle2

        manager = MissionStateManager(
            normalize_angle_func=custom_normalize,
            angle_difference_func=custom_angle_diff,
        )

        assert manager.normalize_angle == custom_normalize
        assert manager.angle_difference == custom_angle_diff


class TestAngleFunctions:
    """Test angle normalization and difference functions."""

    def test_default_normalize_angle(self):
        """Test default angle normalization to [-pi, pi]."""
        manager = MissionStateManager()

        # Test angles already in range
        assert abs(manager.normalize_angle(0.0) - 0.0) < 1e-10
        assert abs(manager.normalize_angle(np.pi / 2) - np.pi / 2) < 1e-10
        assert abs(manager.normalize_angle(-np.pi / 2) + np.pi / 2) < 1e-10

        # Test angles outside range
        assert abs(manager.normalize_angle(2 * np.pi) - 0.0) < 1e-10
        assert abs(manager.normalize_angle(3 * np.pi) - np.pi) < 1e-10
        assert abs(manager.normalize_angle(-3 * np.pi) + np.pi) < 1e-10

    def test_default_angle_difference(self):
        """Test default angle difference calculation."""
        manager = MissionStateManager()

        # Test simple differences
        assert abs(manager.angle_difference(np.pi / 2, 0.0) - np.pi / 2) < 1e-10
        assert abs(manager.angle_difference(0.0, np.pi / 2) + np.pi / 2) < 1e-10

        # Test wraparound
        assert abs(manager.angle_difference(np.pi, -np.pi) - 0.0) < 1e-10
        assert abs(manager.angle_difference(0.1, -0.1) - 0.2) < 1e-10


class TestPointToLineDistance:
    """Test point-to-line distance calculation."""

    def test_point_on_line(self):
        """Test distance when point is on the line."""
        manager = MissionStateManager()

        point = np.array([0.5, 0.5])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 1.0])

        distance = manager.point_to_line_distance(point, line_start, line_end)
        assert abs(distance) < 1e-10

    def test_point_perpendicular_to_line(self):
        """Test distance when point is perpendicular to line."""
        manager = MissionStateManager()

        point = np.array([0.5, 1.0])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 0.0])

        distance = manager.point_to_line_distance(point, line_start, line_end)
        assert abs(distance - 1.0) < 1e-10

    def test_point_off_line_segment(self):
        """Test distance when point is beyond line segment."""
        manager = MissionStateManager()

        point = np.array([2.0, 0.0])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 0.0])

        distance = manager.point_to_line_distance(point, line_start, line_end)
        assert abs(distance - 1.0) < 1e-10  # Distance to closest endpoint


class TestMissionStateTracking:
    """Test mission state tracking variables."""

    def test_multi_point_state_initialization(self):
        """Test multi-point mission state is properly initialized."""
        manager = MissionStateManager()

        assert manager.multi_point_target_reached_time is None

    def test_dxf_shape_state_initialization(self):
        """Test DXF shape mission state is properly initialized."""
        manager = MissionStateManager()

        assert manager.shape_stabilization_start_time is None
        assert manager.return_stabilization_start_time is None


class TestTargetStateCalculation:
    """Test target state vector calculation."""

    def test_simple_target_state(self):
        """Test creating a simple target state vector."""
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        manager = MissionStateManager()

        target_pos = (1.0, 2.0, 0.0)
        target_angle = (0.0, 0.0, np.pi / 4)

        target_state = manager._create_3d_state(
            target_pos[0], target_pos[1], target_angle, z=target_pos[2]
        )

        assert target_state.shape == (13,)
        assert target_state[0] == 1.0
        assert target_state[1] == 2.0
        assert np.allclose(target_state[3:7], euler_xyz_to_quat_wxyz(target_angle))


class TestMultiPointMission:
    """Test multi-point mission logic."""

    def test_multi_point_timing(self):
        """Test multi-point target reached timing."""
        manager = MissionStateManager()

        assert manager.multi_point_target_reached_time is None

        manager.multi_point_target_reached_time = 5.0
        assert manager.multi_point_target_reached_time == 5.0


class TestDXFShapeMission:
    """Test DXF shape following mission logic."""

    def test_shape_stabilization_timing(self):
        """Test shape stabilization timing."""
        manager = MissionStateManager()

        assert manager.shape_stabilization_start_time is None
        assert manager.return_stabilization_start_time is None

        manager.shape_stabilization_start_time = 10.0
        assert manager.shape_stabilization_start_time == 10.0

        manager.return_stabilization_start_time = 20.0
        assert manager.return_stabilization_start_time == 20.0


class TestTolerances:
    """Test tolerance checking."""

    def test_position_tolerance_check(self):
        """Test position error tolerance checking."""
        manager = MissionStateManager(position_tolerance=0.1)

        current_pos = np.array([0.0, 0.0])
        target_pos = np.array([0.05, 0.05])

        pos_error = np.linalg.norm(target_pos - current_pos)
        assert pos_error < manager.position_tolerance

    def test_angle_tolerance_check(self):
        """Test angle error tolerance checking."""
        manager = MissionStateManager(angle_tolerance=0.1)

        current_angle = 0.0
        target_angle = 0.05

        angle_error = abs(manager.angle_difference(target_angle, current_angle))
        assert angle_error < manager.angle_tolerance


class TestHelperFunctionInjection:
    """Test injection of custom helper functions."""

    def test_custom_normalize_angle_injection(self):
        """Test injecting custom angle normalization function."""

        def custom_normalize(angle):
            # Custom normalization that does nothing
            return angle

        manager = MissionStateManager(normalize_angle_func=custom_normalize)

        result = manager.normalize_angle(3 * np.pi)
        assert result == 3 * np.pi  # Custom function doesn't normalize

    def test_custom_angle_difference_injection(self):
        """Test injecting custom angle difference function."""

        def custom_diff(angle1, angle2):
            # Custom difference that just subtracts
            return angle1 - angle2

        manager = MissionStateManager(angle_difference_func=custom_diff)

        result = manager.angle_difference(5.0, 2.0)
        assert result == 3.0

    def test_custom_point_to_line_distance_injection(self):
        """Test injecting custom point-to-line distance function."""

        def custom_distance(point, line_start, line_end):
            # Custom distance that always returns 1.0
            return 1.0

        manager = MissionStateManager(point_to_line_distance_func=custom_distance)

        point = np.array([0.0, 0.0])
        line_start = np.array([0.0, 0.0])
        line_end = np.array([1.0, 0.0])

        result = manager.point_to_line_distance(point, line_start, line_end)
        assert result == 1.0


class TestMissionStateReset:
    """Test resetting mission state between runs."""

    def test_reset_mission_state(self):
        """Test resetting mission state between runs."""
        manager = MissionStateManager()

        # Set some state
        manager.current_nav_waypoint_idx = 1
        manager.nav_target_reached_time = 5.0
        manager.multi_point_target_reached_time = 10.0

        # Reset
        manager.reset()

        assert manager.current_nav_waypoint_idx == 0
        assert manager.nav_target_reached_time is None
        assert manager.multi_point_target_reached_time is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
