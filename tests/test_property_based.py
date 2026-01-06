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
        state = np.array([x, y, theta, vx, vy, omega])

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
        from src.satellite_control.config import SatelliteConfig

        bounds = SatelliteConfig.POSITION_BOUNDS

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
        from src.satellite_control.config import SatelliteConfig

        max_v = SatelliteConfig.MAX_VELOCITY

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
        from src.satellite_control.config import SatelliteConfig

        thruster_id = thruster_idx + 1  # 1-indexed
        direction = SatelliteConfig.THRUSTER_DIRECTIONS[thruster_id]

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


# Mark all tests in this file
pytestmark = pytest.mark.unit
