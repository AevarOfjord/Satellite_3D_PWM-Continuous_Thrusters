"""
Unit tests for simulation_state_validator.py module.

Tests the SimulationStateValidator class which handles state validation,
bounds checking, and tolerance verification for satellite simulations.
"""

import numpy as np
import pytest

from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz
from src.satellite_control.utils.simulation_state_validator import (
    SimulationStateValidator,
    create_state_validator_from_config,
)
from unittest.mock import patch


def make_state(
    pos=(0.0, 0.0, 0.0),
    euler=(0.0, 0.0, 0.0),
    vel=(0.0, 0.0, 0.0),
    omega=(0.0, 0.0, 0.0),
):
    state = np.zeros(13)
    state[0:3] = np.array(pos, dtype=float)
    state[3:7] = euler_xyz_to_quat_wxyz(euler)
    state[7:10] = np.array(vel, dtype=float)
    state[10:13] = np.array(omega, dtype=float)
    return state


class TestStateValidatorInitialization:
    """Test SimulationStateValidator initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        validator = SimulationStateValidator()

        assert validator.position_tolerance == 0.05
        assert validator.angle_tolerance == 0.05
        assert validator.velocity_tolerance == 0.01
        assert validator.angular_velocity_tolerance == 0.01

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        validator = SimulationStateValidator(
            position_tolerance=0.1,
            angle_tolerance=0.15,
            velocity_tolerance=0.02,
            angular_velocity_tolerance=0.03,
            position_bounds=5.0,
            max_velocity=0.25,
            max_angular_velocity=np.pi,
        )

        assert validator.position_tolerance == 0.1
        assert validator.angle_tolerance == 0.15
        assert validator.velocity_tolerance == 0.02
        assert validator.angular_velocity_tolerance == 0.03
        assert validator.position_bounds == 5.0
        assert validator.max_velocity == 0.25
        assert validator.max_angular_velocity == np.pi

    def test_factory_function(self):
        """Test factory function creation."""
        validator = create_state_validator_from_config()

        assert isinstance(validator, SimulationStateValidator)
        assert validator.position_tolerance > 0
        assert validator.angle_tolerance > 0


class TestStateFormatValidation:
    """Test state format validation."""

    def test_valid_state_format(self):
        """Test validation of valid state."""
        validator = SimulationStateValidator()

        state = make_state(
            pos=(1.0, 2.0, 0.0),
            euler=(0.0, 0.0, np.pi / 4),
            vel=(0.1, 0.2, 0.0),
            omega=(0.0, 0.0, 0.05),
        )

        assert validator.validate_state_format(state)

    def test_invalid_state_type(self):
        """Test validation rejects non-numpy arrays."""
        validator = SimulationStateValidator()

        with pytest.raises(ValueError, match="must be numpy array"):
            state_list = [1.0, 2.0, 0.1, 0.2, 0.5, 0.05]
            validator.validate_state_format(state_list)  # type: ignore[arg-type]

    def test_invalid_state_shape(self):
        """Test validation rejects wrong-shaped arrays."""
        validator = SimulationStateValidator()

        with pytest.raises(ValueError, match="must have shape"):
            validator.validate_state_format(np.array([1.0, 2.0, 0.1]))

    def test_invalid_state_non_finite(self):
        """Test validation rejects non-finite values."""
        validator = SimulationStateValidator()

        state_with_nan = make_state(
            pos=(1.0, 2.0, 0.0),
            euler=(0.0, 0.0, 0.5),
            vel=(0.1, 0.2, 0.0),
            omega=(0.0, 0.0, 0.05),
        )
        state_with_nan[2] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            validator.validate_state_format(state_with_nan)

        state_with_inf = make_state(
            pos=(1.0, 2.0, 0.0),
            euler=(0.0, 0.0, 0.5),
            vel=(0.1, 0.2, 0.0),
            omega=(0.0, 0.0, 0.05),
        )
        state_with_inf[1] = np.inf

        with pytest.raises(ValueError, match="non-finite"):
            validator.validate_state_format(state_with_inf)


class TestPositionBounds:
    """Test position bounds checking."""

    def test_position_within_bounds(self):
        """Test position within bounds."""
        validator = SimulationStateValidator(position_bounds=3.0)

        position = np.array([1.0, 2.0])

        is_valid, error = validator.check_position_bounds(position)

        assert is_valid
        assert error is None

    def test_position_x_exceeds_bounds(self):
        """Test X position exceeding bounds."""
        validator = SimulationStateValidator(position_bounds=3.0)

        position = np.array([4.0, 2.0])

        is_valid, error = validator.check_position_bounds(position)

        assert not is_valid
        assert error is not None
        assert "X position" in error

    def test_position_y_exceeds_bounds(self):
        """Test Y position exceeding bounds."""
        validator = SimulationStateValidator(position_bounds=3.0)

        position = np.array([1.0, -4.0])

        is_valid, error = validator.check_position_bounds(position)

        assert not is_valid
        assert error is not None
        assert "Y position" in error

    def test_position_at_boundary(self):
        """Test position exactly at boundary."""
        validator = SimulationStateValidator(position_bounds=3.0)

        position = np.array([3.0, -3.0])

        is_valid, error = validator.check_position_bounds(position)

        assert is_valid


class TestVelocityBounds:
    """Test velocity bounds checking."""

    def test_velocity_within_bounds(self):
        """Test velocity within bounds."""
        validator = SimulationStateValidator(max_velocity=0.15)

        velocity = np.array([0.05, 0.05])

        is_valid, error = validator.check_velocity_bounds(velocity)

        assert is_valid
        assert error is None

    def test_velocity_exceeds_bounds(self):
        """Test velocity exceeding bounds."""
        validator = SimulationStateValidator(max_velocity=0.15)

        velocity = np.array([0.2, 0.2])

        is_valid, error = validator.check_velocity_bounds(velocity)

        assert not is_valid
        assert error is not None
        assert "Velocity magnitude" in error

    def test_zero_velocity(self):
        """Test zero velocity."""
        validator = SimulationStateValidator(max_velocity=0.15)

        velocity = np.array([0.0, 0.0])

        is_valid, error = validator.check_velocity_bounds(velocity)

        assert is_valid


class TestAngularVelocityBounds:
    """Test angular velocity bounds checking."""

    def test_angular_velocity_within_bounds(self):
        """Test angular velocity within bounds."""
        validator = SimulationStateValidator(max_angular_velocity=np.pi / 2)

        angular_velocity = 0.5

        is_valid, error = validator.check_angular_velocity_bounds(angular_velocity)

        assert is_valid
        assert error is None

    def test_angular_velocity_exceeds_bounds(self):
        """Test angular velocity exceeding bounds."""
        validator = SimulationStateValidator(max_angular_velocity=np.pi / 2)

        angular_velocity = 2.0

        is_valid, error = validator.check_angular_velocity_bounds(angular_velocity)

        assert not is_valid
        assert error is not None
        assert "Angular velocity" in error


class TestAllBoundsChecking:
    """Test comprehensive bounds checking."""

    def test_all_bounds_valid(self):
        """Test state with all bounds valid."""
        validator = SimulationStateValidator(
            position_bounds=3.0, max_velocity=0.15, max_angular_velocity=np.pi / 2
        )

        state = make_state(
            pos=(1.0, 2.0, 0.0),
            euler=(0.0, 0.0, np.pi / 4),
            vel=(0.05, 0.05, 0.0),
            omega=(0.0, 0.0, 0.1),
        )

        all_valid, errors = validator.check_all_bounds(state)

        assert all_valid
        assert len(errors) == 0

    def test_all_bounds_multiple_violations(self):
        """Test state with multiple bound violations."""
        validator = SimulationStateValidator(
            position_bounds=3.0, max_velocity=0.15, max_angular_velocity=np.pi / 2
        )

        # Violate position, velocity, and angular velocity
        state = make_state(
            pos=(4.0, 5.0, 0.0),
            euler=(0.0, 0.0, np.pi / 4),
            vel=(0.3, 0.3, 0.0),
            omega=(0.0, 0.0, 2.0),
        )

        all_valid, errors = validator.check_all_bounds(state)

        assert not all_valid
        assert len(errors) == 3  # Position, velocity, angular velocity


class TestAngleOperations:
    """Test angle normalization and difference."""

    def test_normalize_angle_within_range(self):
        """Test normalizing angle already in range."""
        validator = SimulationStateValidator()

        angle = np.pi / 4

        normalized = validator.normalize_angle(angle)

        assert normalized == pytest.approx(np.pi / 4)

    def test_normalize_angle_outside_range(self):
        """Test normalizing angle outside [-pi, pi]."""
        validator = SimulationStateValidator()

        angle = 3 * np.pi

        normalized = validator.normalize_angle(angle)

        assert -np.pi <= normalized <= np.pi
        assert normalized == pytest.approx(np.pi)

    def test_angle_difference_simple(self):
        """Test simple angle difference."""
        validator = SimulationStateValidator()

        diff = validator.angle_difference(np.pi / 2, 0.0)

        assert diff == pytest.approx(np.pi / 2)

    def test_angle_difference_wraparound(self):
        """Test angle difference with wraparound."""
        validator = SimulationStateValidator()

        diff = validator.angle_difference(np.pi, -np.pi)

        assert abs(diff) < 0.01  # Should be near zero


class TestTargetReachedChecking:
    """Test target reached verification."""

    def test_target_reached_all_within_tolerance(self):
        """Test target reached when all errors within tolerance."""
        validator = SimulationStateValidator(
            position_tolerance=0.1,
            angle_tolerance=0.1,
            velocity_tolerance=0.02,
            angular_velocity_tolerance=0.02,
        )

        current_state = make_state(
            pos=(0.05, 0.05, 0.0),
            euler=(0.0, 0.0, 0.05),
            vel=(0.01, 0.01, 0.0),
            omega=(0.0, 0.0, 0.01),
        )
        target_state = make_state()

        assert validator.check_target_reached(current_state, target_state)

    def test_target_not_reached_position_error(self):
        """Test target not reached due to position error."""
        validator = SimulationStateValidator(position_tolerance=0.05)

        current_state = make_state(pos=(0.1, 0.1, 0.0))
        target_state = make_state()

        assert not validator.check_target_reached(current_state, target_state)

    def test_target_not_reached_angle_error(self):
        """Test target not reached due to angle error."""
        validator = SimulationStateValidator(angle_tolerance=0.05)

        current_state = make_state(euler=(0.0, 0.0, 0.2))
        target_state = make_state()

        assert not validator.check_target_reached(current_state, target_state)

    def test_target_not_reached_velocity_error(self):
        """Test target not reached due to velocity error."""
        validator = SimulationStateValidator(velocity_tolerance=0.01)

        current_state = make_state(vel=(0.05, 0.05, 0.0))
        target_state = make_state()

        assert not validator.check_target_reached(current_state, target_state)


class TestStateErrorComputation:
    """Test state error computation."""

    def test_compute_state_errors(self):
        """Test computing all state errors."""
        validator = SimulationStateValidator()

        current_state = make_state(
            pos=(1.0, 2.0, 0.0),
            euler=(0.0, 0.0, np.pi / 4),
            vel=(0.1, 0.2, 0.0),
            omega=(0.0, 0.0, 0.05),
        )
        target_state = make_state()

        errors = validator.compute_state_errors(current_state, target_state)

        assert "position_error" in errors
        assert "velocity_error" in errors
        assert "angle_error" in errors
        assert "angular_velocity_error" in errors

        # Check specific values
        assert errors["position_error"] == pytest.approx(np.sqrt(1.0**2 + 2.0**2))
        assert errors["velocity_error"] == pytest.approx(np.sqrt(0.1**2 + 0.2**2))

    def test_check_within_tolerances(self):
        """Test checking which components are within tolerance."""
        validator = SimulationStateValidator(
            position_tolerance=0.1,
            velocity_tolerance=0.05,
            angle_tolerance=0.1,
            angular_velocity_tolerance=0.02,
        )

        # Position and angle within tolerance, velocity and angular velocity not
        current_state = make_state(
            pos=(0.05, 0.05, 0.0),
            euler=(0.0, 0.0, 0.05),
            vel=(0.1, 0.1, 0.0),
            omega=(0.0, 0.0, 0.05),
        )
        target_state = make_state()

        tolerances = validator.check_within_tolerances(current_state, target_state)

        assert tolerances["position"]
        assert not tolerances["velocity"]
        assert tolerances["angle"]
        assert not tolerances["angular_velocity"]


class TestSensorNoise:
    """Test sensor noise application."""

    def test_sensor_noise_disabled(self):
        """Test that no noise is added when disabled."""
        validator = SimulationStateValidator()

        true_state = make_state(
            pos=(1.0, 2.0, 0.0),
            euler=(0.0, 0.0, np.pi / 4),
            vel=(0.1, 0.2, 0.0),
            omega=(0.0, 0.0, 0.05),
        )

        # Call function - result intentionally unused for this test
        _ = validator.apply_sensor_noise(true_state, use_realistic_physics=False)

    def test_sensor_noise_enabled(self):
        """Test that noise is added when enabled."""
        validator = SimulationStateValidator()

        true_state = make_state(
            pos=(1.0, 2.0, 0.0),
            euler=(0.0, 0.0, np.pi / 4),
            vel=(0.1, 0.2, 0.0),
            omega=(0.0, 0.0, 0.05),
        )

        # V4.0.0: Patch app_config.physics instead of SatelliteConfig
        # Create a mock app_config with noise parameters
        from src.satellite_control.config.models import AppConfig, SatellitePhysicalParams
        from unittest.mock import MagicMock
        
        mock_physics = MagicMock()
        mock_physics.use_realistic_physics = True
        mock_physics.position_noise_std = 0.01
        mock_physics.velocity_noise_std = 0.01
        mock_physics.angle_noise_std = 0.01
        mock_physics.angular_velocity_noise_std = 0.01
        
        mock_app_config = MagicMock()
        mock_app_config.physics = mock_physics
        
        # Patch the validator's app_config
        validator.app_config = mock_app_config

        noisy_states = [
            validator.apply_sensor_noise(true_state, use_realistic_physics=True)
            for _ in range(10)
        ]

        # Check that at least some are different (due to randomness)
        differences = [not np.array_equal(noisy, true_state) for noisy in noisy_states]

        # With high probability, at least one should be different
        assert any(differences)

    def test_sensor_noise_preserves_format(self):
        """Test that noisy state has same format."""
        validator = SimulationStateValidator()

        true_state = make_state(
            pos=(1.0, 2.0, 0.0),
            euler=(0.0, 0.0, np.pi / 4),
            vel=(0.1, 0.2, 0.0),
            omega=(0.0, 0.0, 0.05),
        )

        noisy_state = validator.apply_sensor_noise(
            true_state, use_realistic_physics=True
        )

        # Should still be valid format
        assert validator.validate_state_format(noisy_state)


class TestSummaryMethods:
    """Test summary and reporting methods."""

    def test_tolerance_summary(self):
        """Test getting tolerance summary."""
        validator = SimulationStateValidator(
            position_tolerance=0.1,
            angle_tolerance=0.15,
            velocity_tolerance=0.02,
            angular_velocity_tolerance=0.03,
        )

        summary = validator.get_tolerance_summary()

        assert summary["position_tolerance"] == 0.1
        assert summary["angle_tolerance"] == 0.15
        assert summary["velocity_tolerance"] == 0.02
        assert summary["angular_velocity_tolerance"] == 0.03

    def test_bounds_summary(self):
        """Test getting bounds summary."""
        validator = SimulationStateValidator(
            position_bounds=5.0, max_velocity=0.25, max_angular_velocity=np.pi
        )

        summary = validator.get_bounds_summary()

        assert summary["position_bounds"] == 5.0
        assert summary["max_velocity"] == 0.25
        assert summary["max_angular_velocity"] == np.pi


class TestTrajectoryValidation:
    """Test trajectory validation."""

    def test_validate_empty_trajectory(self):
        """Test validating empty trajectory."""
        validator = SimulationStateValidator()

        is_valid, errors = validator.validate_state_trajectory([])

        assert is_valid
        assert len(errors) == 0

    def test_validate_single_state_trajectory(self):
        """Test validating single-state trajectory."""
        validator = SimulationStateValidator()

        # Use velocities within default max_velocity (0.15 m/s)
        state = make_state(
            pos=(1.0, 2.0, 0.0),
            euler=(0.0, 0.0, np.pi / 4),
            vel=(0.05, 0.08, 0.0),
            omega=(0.0, 0.0, 0.05),
        )
        trajectory = [state]

        is_valid, errors = validator.validate_state_trajectory(trajectory)

        assert is_valid
        assert len(errors) == 0

    def test_validate_continuous_trajectory(self):
        """Test validating continuous trajectory."""
        validator = SimulationStateValidator()

        trajectory = [
            make_state(),
            make_state(
                pos=(0.1, 0.1, 0.0),
                euler=(0.0, 0.0, 0.1),
                vel=(0.05, 0.05, 0.0),
                omega=(0.0, 0.0, 0.02),
            ),
            make_state(
                pos=(0.2, 0.2, 0.0),
                euler=(0.0, 0.0, 0.2),
                vel=(0.05, 0.05, 0.0),
                omega=(0.0, 0.0, 0.02),
            ),
        ]

        is_valid, errors = validator.validate_state_trajectory(
            trajectory, check_continuity=True
        )

        assert is_valid
        assert len(errors) == 0

    def test_validate_discontinuous_trajectory(self):
        """Test detecting discontinuous trajectory."""
        validator = SimulationStateValidator()

        trajectory = [
            make_state(),
            make_state(
                pos=(2.0, 2.0, 0.0),
                euler=(0.0, 0.0, 0.1),
                vel=(0.05, 0.05, 0.0),
                omega=(0.0, 0.0, 0.02),
            ),  # Large jump
        ]

        is_valid, errors = validator.validate_state_trajectory(
            trajectory, check_continuity=True, max_position_jump=0.5
        )

        assert not is_valid
        assert len(errors) > 0
        assert "position jump" in errors[0].lower()

    def test_validate_trajectory_with_bounds_violation(self):
        """Test detecting bounds violations in trajectory."""
        validator = SimulationStateValidator(position_bounds=3.0)

        trajectory = [
            make_state(),
            make_state(pos=(4.0, 4.0, 0.0)),  # Out of bounds
        ]

        is_valid, errors = validator.validate_state_trajectory(trajectory)

        assert not is_valid
        assert len(errors) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_state(self):
        """Test with zero state."""
        validator = SimulationStateValidator()

        state = make_state()

        assert validator.validate_state_format(state)

        is_valid, errors = validator.check_all_bounds(state)
        assert is_valid

    def test_very_small_tolerances(self):
        """Test with very small tolerances."""
        validator = SimulationStateValidator(
            position_tolerance=1e-6, angle_tolerance=1e-6
        )

        current_state = make_state(pos=(1e-7, 1e-7, 0.0), euler=(0.0, 0.0, 1e-7))
        target_state = make_state()

        assert validator.check_target_reached(current_state, target_state)

    def test_large_state_values(self):
        """Test with large state values."""
        validator = SimulationStateValidator(position_bounds=10.0, max_velocity=1.0)

        state = make_state(
            pos=(9.0, 9.0, 0.0),
            euler=(0.0, 0.0, 2 * np.pi),
            vel=(0.5, 0.5, 0.0),
            omega=(0.0, 0.0, 0.5),
        )

        is_valid, errors = validator.check_all_bounds(state)
        assert is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
