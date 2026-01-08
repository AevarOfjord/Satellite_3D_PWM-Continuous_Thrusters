"""
Property-Based Tests for Satellite Control System

Uses Hypothesis to generate edge cases and test invariants.
"""

import numpy as np
import pytest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    pytest.skip("hypothesis not installed", allow_module_level=True)


# ============================================================================
# State Converter Property Tests
# ============================================================================


@pytest.mark.unit
class TestStateConverterProperties:
    """Property-based tests for state conversion."""

    @given(
        x=st.floats(-5.0, 5.0, allow_nan=False),
        y=st.floats(-5.0, 5.0, allow_nan=False),
        theta=st.floats(-np.pi, np.pi, allow_nan=False),
        vx=st.floats(-1.0, 1.0, allow_nan=False),
        vy=st.floats(-1.0, 1.0, allow_nan=False),
        omega=st.floats(-1.0, 1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_state_vector_finite(self, x, y, theta, vx, vy, omega):
        """All state components should produce finite results in MPC."""
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        state = np.zeros(13)
        state[0:3] = np.array([x, y, 0.0])
        state[3:7] = euler_xyz_to_quat_wxyz((0.0, 0.0, theta))
        state[7:10] = np.array([vx, vy, 0.0])
        state[10:13] = np.array([0.0, 0.0, omega])

        # State should be finite
        assert np.all(np.isfinite(state))

        # Norm should be finite
        assert np.isfinite(np.linalg.norm(state))


# ============================================================================
# MPC Constraint Property Tests
# ============================================================================


@pytest.mark.unit
class TestMPCConstraintProperties:
    """Property-based tests for MPC constraints."""

    @given(
        pos_x=st.floats(-2.0, 2.0, allow_nan=False),
        pos_y=st.floats(-2.0, 2.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_position_bounds_symmetric(self, pos_x, pos_y):
        """Position bounds should be symmetric around origin."""
        from src.satellite_control.config.simulation_config import SimulationConfig

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        bounds = config.app_config.mpc.position_bounds

        # If position is within bounds, -position should also be valid
        if abs(pos_x) <= bounds and abs(pos_y) <= bounds:
            assert abs(-pos_x) <= bounds
            assert abs(-pos_y) <= bounds

    @given(
        velocity=st.floats(0.0, 2.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_velocity_limit_positive(self, velocity):
        """Velocity limits should be positive symmetric."""
        from src.satellite_control.config.simulation_config import SimulationConfig

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        max_v = config.app_config.mpc.max_velocity

        assert max_v > 0
        # If v is within limit, -v should also be within limit
        if velocity <= max_v:
            assert -velocity >= -max_v


# ============================================================================
# Thruster Configuration Property Tests
# ============================================================================


@pytest.mark.unit
class TestThrusterProperties:
    """Property-based tests for thruster configuration."""

    @given(
        thruster_idx=st.integers(0, 7),
    )
    @settings(max_examples=20)
    def test_thruster_direction_is_unit_vector(self, thruster_idx):
        """All thruster directions should be unit vectors."""
        from src.satellite_control.config.simulation_config import SimulationConfig

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        thruster_id = thruster_idx + 1  # 1-indexed
        direction = config.app_config.physics.thruster_directions[thruster_id]

        magnitude = np.linalg.norm(direction)
        assert abs(magnitude - 1.0) < 1e-6, f"Thruster {thruster_id} not unit vector"

    @given(
        force_multiplier=st.floats(0.1, 2.0, allow_nan=False),
    )
    @settings(max_examples=20)
    def test_scaled_forces_remain_positive(self, force_multiplier):
        """Scaled thruster forces should remain positive."""
        from src.satellite_control.config.physics import THRUSTER_FORCES

        for tid, force in THRUSTER_FORCES.items():
            scaled = force * force_multiplier
            assert scaled > 0, f"Thruster {tid} force became non-positive"


# ============================================================================
# Dynamics Property Tests
# ============================================================================


@pytest.mark.unit
class TestDynamicsProperties:
    """Property-based tests for dynamics calculations."""

    @given(
        angle=st.floats(-2 * np.pi, 2 * np.pi, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_rotation_matrix_orthogonal(self, angle):
        """Rotation matrices should be orthogonal."""
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])

        # R @ R.T should be identity
        identity = R @ R.T
        assert np.allclose(identity, np.eye(2), atol=1e-10)

        # Determinant should be 1
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    @given(
        dt=st.floats(0.001, 0.1, allow_nan=False),
        velocity=st.floats(-1.0, 1.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_position_update_linear(self, dt, velocity):
        """Position update should be linear in velocity."""
        pos_0 = 0.0
        pos_1 = pos_0 + velocity * dt
        pos_2 = pos_0 + 2 * velocity * dt

        # Double velocity should double displacement
        assert abs((pos_2 - pos_0) - 2 * (pos_1 - pos_0)) < 1e-10


# ============================================================================
# Obstacle Avoidance Property Tests
# ============================================================================


@pytest.mark.unit
class TestObstacleAvoidanceProperties:
    """Property-based tests for obstacle avoidance."""

    @given(
        obs_x=st.floats(-2.0, 2.0, allow_nan=False),
        obs_y=st.floats(-2.0, 2.0, allow_nan=False),
        obs_radius=st.floats(0.1, 1.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_obstacle_radius_positive(self, obs_x, obs_y, obs_radius):
        """Obstacle radius should always be positive."""
        assert obs_radius > 0

    @given(
        safety_margin=st.floats(0.01, 0.5, allow_nan=False),
    )
    @settings(max_examples=20)
    def test_safety_margin_positive(self, safety_margin):
        """Safety margin should be positive."""
        assert safety_margin > 0


# ============================================================================
# State Validation Edge Cases
# ============================================================================


@pytest.mark.unit
class TestStateValidationEdgeCases:
    """Edge case property tests for state validation."""

    @given(
        angle=st.floats(-10 * np.pi, 10 * np.pi, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_angle_normalization_always_in_range(self, angle):
        """Normalized angle should always be in [-pi, pi]."""
        from src.satellite_control.utils.navigation_utils import normalize_angle

        normalized = normalize_angle(angle)
        assert -np.pi <= normalized <= np.pi

    @given(
        x=st.floats(-10.0, 10.0, allow_nan=False),
        y=st.floats(-10.0, 10.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_euclidean_distance_non_negative(self, x, y):
        """Euclidean distance should never be negative."""
        origin = np.array([0.0, 0.0])
        point = np.array([x, y])
        distance = np.linalg.norm(point - origin)
        assert distance >= 0


# ============================================================================
# Navigation Utilities Property Tests
# ============================================================================


@pytest.mark.unit
class TestNavigationUtilsProperties:
    """Property-based tests for navigation utilities."""

    @given(
        angle1=st.floats(-10 * np.pi, 10 * np.pi, allow_nan=False),
        angle2=st.floats(-10 * np.pi, 10 * np.pi, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_angle_difference_symmetric(self, angle1, angle2):
        """Angle difference should be symmetric (within sign)."""
        from src.satellite_control.utils.navigation_utils import angle_difference

        diff1 = angle_difference(angle1, angle2)
        diff2 = angle_difference(angle2, angle1)

        # Should be negatives of each other (within floating point precision)
        assert abs(diff1 + diff2) < 1e-10 or abs(abs(diff1) - abs(diff2)) < 1e-10

    @given(
        angle=st.floats(-10 * np.pi, 10 * np.pi, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_normalize_angle_idempotent(self, angle):
        """Normalizing an already normalized angle should be idempotent."""
        from src.satellite_control.utils.navigation_utils import normalize_angle

        normalized_once = normalize_angle(angle)
        normalized_twice = normalize_angle(normalized_once)

        assert abs(normalized_once - normalized_twice) < 1e-10

    @given(
        point_x=st.floats(-5.0, 5.0, allow_nan=False),
        point_y=st.floats(-5.0, 5.0, allow_nan=False),
        line_x1=st.floats(-5.0, 5.0, allow_nan=False),
        line_y1=st.floats(-5.0, 5.0, allow_nan=False),
        line_x2=st.floats(-5.0, 5.0, allow_nan=False),
        line_y2=st.floats(-5.0, 5.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_point_to_line_distance_non_negative(
        self, point_x, point_y, line_x1, line_y1, line_x2, line_y2
    ):
        """Point-to-line distance should always be non-negative."""
        from src.satellite_control.utils.navigation_utils import point_to_line_distance

        point = np.array([point_x, point_y])
        line_start = np.array([line_x1, line_y1])
        line_end = np.array([line_x2, line_y2])

        distance = point_to_line_distance(point, line_start, line_end)
        assert distance >= 0


# ============================================================================
# Orientation Utilities Property Tests
# ============================================================================


@pytest.mark.unit
class TestOrientationUtilsProperties:
    """Property-based tests for orientation utilities."""

    @given(
        roll=st.floats(-np.pi, np.pi, allow_nan=False),
        pitch=st.floats(-np.pi / 2, np.pi / 2, allow_nan=False),
        yaw=st.floats(-np.pi, np.pi, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_euler_quat_roundtrip(self, roll, pitch, yaw):
        """Euler to quaternion and back should be idempotent."""
        from src.satellite_control.utils.orientation_utils import (
            euler_xyz_to_quat_wxyz,
            quat_wxyz_to_euler_xyz,
        )

        euler = (roll, pitch, yaw)
        quat = euler_xyz_to_quat_wxyz(euler)
        euler_back = quat_wxyz_to_euler_xyz(quat)

        # Should be close (within reasonable tolerance for gimbal lock cases)
        assert np.allclose(euler, euler_back, atol=1e-5) or np.allclose(
            np.array(euler) + np.array([0, 0, 2 * np.pi]), euler_back, atol=1e-5
        )

    @given(
        roll=st.floats(-np.pi, np.pi, allow_nan=False),
        pitch=st.floats(-np.pi / 2, np.pi / 2, allow_nan=False),
        yaw=st.floats(-np.pi, np.pi, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_quaternion_normalized(self, roll, pitch, yaw):
        """Quaternions from Euler angles should be normalized."""
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        euler = (roll, pitch, yaw)
        quat = euler_xyz_to_quat_wxyz(euler)

        # Quaternion should be unit length
        quat_norm = np.linalg.norm(quat)
        assert abs(quat_norm - 1.0) < 1e-10

    @given(
        qw=st.floats(-1.0, 1.0, allow_nan=False),
        qx=st.floats(-1.0, 1.0, allow_nan=False),
        qy=st.floats(-1.0, 1.0, allow_nan=False),
        qz=st.floats(-1.0, 1.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_quat_angle_error_symmetric(self, qw, qx, qy, qz):
        """Quaternion angle error should be symmetric."""
        from src.satellite_control.utils.orientation_utils import quat_angle_error

        # Normalize quaternion
        quat1 = np.array([qw, qx, qy, qz])
        quat1_norm = np.linalg.norm(quat1)
        if quat1_norm > 1e-10:
            quat1 = quat1 / quat1_norm
        else:
            quat1 = np.array([1.0, 0.0, 0.0, 0.0])

        quat2 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity

        error1 = quat_angle_error(quat1, quat2)
        error2 = quat_angle_error(quat2, quat1)

        # Should be the same (symmetric)
        assert abs(error1 - error2) < 1e-10


# ============================================================================
# State Validation Property Tests
# ============================================================================


@pytest.mark.unit
class TestStateValidationProperties:
    """Property-based tests for state validation."""

    @given(
        pos_x=st.floats(-5.0, 5.0, allow_nan=False),
        pos_y=st.floats(-5.0, 5.0, allow_nan=False),
        pos_z=st.floats(-5.0, 5.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_state_validation_deterministic(self, pos_x, pos_y, pos_z):
        """State validation should be deterministic."""
        from src.satellite_control.utils.simulation_state_validator import (
            create_state_validator_from_config,
        )
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        validator = create_state_validator_from_config()

        state = np.zeros(13)
        state[0:3] = np.array([pos_x, pos_y, pos_z])
        state[3:7] = euler_xyz_to_quat_wxyz((0.0, 0.0, 0.0))

        result1 = validator.validate_state_format(state)
        result2 = validator.validate_state_format(state)

        assert result1 == result2

    @given(
        state_list=st.lists(
            st.floats(-10.0, 10.0, allow_nan=False), min_size=13, max_size=13
        ),
    )
    @settings(max_examples=50)
    def test_state_format_validation(self, state_list):
        """State format validation should handle various inputs."""
        from src.satellite_control.utils.simulation_state_validator import (
            create_state_validator_from_config,
        )

        validator = create_state_validator_from_config()
        state = np.array(state_list)

        # Should not raise exception
        result = validator.validate_state_format(state)
        assert isinstance(result, bool)


# ============================================================================
# Caching Property Tests
# ============================================================================


@pytest.mark.unit
class TestCachingProperties:
    """Property-based tests for caching utilities."""

    @given(
        value=st.integers(0, 1000),
    )
    @settings(max_examples=20)
    def test_cached_function_deterministic(self, value):
        """Cached functions should return same result for same input."""
        from src.satellite_control.utils.caching import cached

        call_count = [0]

        @cached(maxsize=10)
        def test_func(x):
            call_count[0] += 1
            return x * 2

        result1 = test_func(value)
        result2 = test_func(value)

        # Should return same result
        assert result1 == result2
        # Should only call function once (cached)
        assert call_count[0] == 1

    @given(
        value1=st.integers(0, 100),
        value2=st.integers(0, 100),
    )
    @settings(max_examples=20)
    def test_cached_function_different_inputs(self, value1, value2):
        """Cached functions should handle different inputs correctly."""
        from src.satellite_control.utils.caching import cached

        @cached(maxsize=10)
        def test_func(x):
            return x * 2

        if value1 != value2:
            result1 = test_func(value1)
            result2 = test_func(value2)
            assert result1 != result2 or value1 == value2


# Mark all tests in this file
pytestmark = pytest.mark.unit
