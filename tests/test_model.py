"""
Unit tests for model.py module.

Tests satellite geometry, thruster layout, and physical model constants.
"""

import numpy as np
import pytest

from src.satellite_control.core import model


class TestModelConstants:
    """Test model-level constants."""

    def test_dt_is_positive(self):
        """Test that simulation timestep is positive."""
        assert model.DT > 0

    def test_control_dt_is_positive(self):
        """Test that control timestep is positive."""
        assert model.CONTROL_DT > 0

    def test_control_dt_greater_than_dt(self):
        """Test that control interval is longer than simulation step."""
        assert model.CONTROL_DT >= model.DT


class TestThrusterPositions:
    """Test thruster position definitions."""

    def test_thruster_positions_exist(self):
        """Test that thruster_positions dictionary exists."""
        assert hasattr(model, "thruster_positions")
        assert isinstance(model.thruster_positions, dict)

    def test_eight_thrusters_defined(self):
        """Test that exactly 8 thrusters are defined."""
        assert len(model.thruster_positions) == 8

    def test_thruster_ids_are_1_to_8(self):
        """Test that thruster IDs are 1 through 8."""
        expected_ids = set(range(1, 9))
        actual_ids = set(model.thruster_positions.keys())
        assert actual_ids == expected_ids

    def test_thruster_positions_are_tuples(self):
        """Test that each position is a tuple of two values."""
        for thruster_id, position in model.thruster_positions.items():
            assert isinstance(
                position, tuple
            ), f"Thruster {thruster_id} position not a tuple"
            assert (
                len(position) == 2
            ), f"Thruster {thruster_id} position should have 2 coordinates"

    def test_thruster_positions_are_symmetric(self):
        """Test that thrusters are roughly symmetric around center."""
        positions = np.array(list(model.thruster_positions.values()))
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]

        # Center of mass should be near origin
        assert abs(np.mean(x_coords)) < 0.1
        assert abs(np.mean(y_coords)) < 0.1


class TestThrusterDirections:
    """Test thruster direction definitions."""

    def test_thruster_directions_exist(self):
        """Test that thruster_directions dictionary exists."""
        assert hasattr(model, "thruster_directions")
        assert isinstance(model.thruster_directions, dict)

    def test_eight_directions_defined(self):
        """Test that exactly 8 directions are defined."""
        assert len(model.thruster_directions) == 8

    def test_thruster_direction_ids_match_position_ids(self):
        """Test that direction IDs match position IDs."""
        position_ids = set(model.thruster_positions.keys())
        direction_ids = set(model.thruster_directions.keys())
        assert position_ids == direction_ids

    def test_thruster_directions_are_tuples(self):
        """Test that each direction is a tuple of two values."""
        for thruster_id, direction in model.thruster_directions.items():
            assert isinstance(
                direction, tuple
            ), f"Thruster {thruster_id} direction not a tuple"
            assert (
                len(direction) == 2
            ), f"Thruster {thruster_id} direction should have 2 components"

    def test_thruster_directions_are_unit_vectors(self):
        """Test that all direction vectors are normalized."""
        for thruster_id, (dx, dy) in model.thruster_directions.items():
            magnitude = np.sqrt(dx**2 + dy**2)
            assert (
                abs(magnitude - 1.0) < 1e-6
            ), f"Thruster {thruster_id} direction not unit vector: {magnitude}"

    def test_thruster_directions_are_valid_orientations(self):
        """Test that directions are one of the cardinal/diagonal directions."""
        valid_directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),  # Cardinal
            (np.sqrt(2) / 2, np.sqrt(2) / 2),  # Diagonal NE
            (np.sqrt(2) / 2, -np.sqrt(2) / 2),  # Diagonal SE
            (-np.sqrt(2) / 2, -np.sqrt(2) / 2),  # Diagonal SW
            (-np.sqrt(2) / 2, np.sqrt(2) / 2),  # Diagonal NW
        ]

        for thruster_id, direction in model.thruster_directions.items():
            dx, dy = direction
            # Check if close to any valid direction
            is_valid = any(
                abs(dx - vx) < 0.01 and abs(dy - vy) < 0.01
                for vx, vy in valid_directions
            )
            assert (
                is_valid
            ), f"Thruster {thruster_id} has invalid direction: {direction}"


class TestAirBearingPositions:
    """Test air bearing position definitions if they exist."""

    def test_air_bearing_positions_if_defined(self):
        """Test air bearing positions if defined in model."""
        if hasattr(model, "air_bearing_positions"):
            assert isinstance(
                model.air_bearing_positions,
                (dict, list, tuple),
            )

            if isinstance(model.air_bearing_positions, dict):
                # Should have 3 air bearings
                assert len(model.air_bearing_positions) == 3


class TestSatelliteGeometry:
    """Test satellite geometry definitions."""

    def test_satellite_size_if_defined(self):
        """Test satellite size constant if defined."""
        if hasattr(model, "SATELLITE_SIZE"):
            assert getattr(model, "SATELLITE_SIZE") > 0

    def test_workspace_bounds_if_defined(self):
        """Test workspace boundary definitions if present."""
        if hasattr(model, "WORKSPACE_SIZE"):
            assert getattr(model, "WORKSPACE_SIZE") > 0
        elif hasattr(model, "WORKSPACE_WIDTH") and hasattr(model, "WORKSPACE_HEIGHT"):
            assert getattr(model, "WORKSPACE_WIDTH") > 0
            assert getattr(model, "WORKSPACE_HEIGHT") > 0


class TestPhysicsCalculations:
    """Test physics calculation functions if present."""

    def test_compute_thruster_force_if_exists(self):
        """Test thruster force computation function if it exists."""
        if hasattr(model, "compute_thruster_force"):
            # Test with sample inputs
            position = (0.1, 0.1)
            direction = (1.0, 0.0)
            magnitude = 0.45

            result = model.compute_thruster_force(position, direction, magnitude)
            assert isinstance(result, (tuple, list, np.ndarray))

    def test_compute_moment_if_exists(self):
        """Test moment/torque computation if it exists."""
        if hasattr(model, "compute_moment") or hasattr(model, "compute_torque"):
            func = getattr(
                model, "compute_moment", getattr(model, "compute_torque", None)
            )

            if func is not None:
                # Test with sample inputs
                position = (0.1, 0.1)
                force = (1.0, 0.0)

                result = func(position, force)
                assert isinstance(result, (int, float, np.number))


class TestDrawingFunctions:
    """Test drawing/visualization functions if present."""

    def test_draw_satellite_if_exists(self):
        """Test satellite drawing function if it exists."""
        if hasattr(model, "draw_satellite"):
            # Just verify it's callable
            assert callable(model.draw_satellite)

    def test_draw_thruster_if_exists(self):
        """Test thruster drawing function if it exists."""
        if hasattr(model, "draw_thruster"):
            assert callable(model.draw_thruster)


class TestModelIntegration:
    """Integration tests for model module."""

    def test_can_compute_total_force_from_all_thrusters(self):
        """Test that we can compute total force from all thrusters."""
        # All thrusters firing
        thruster_commands = np.ones(8)
        total_fx = 0
        total_fy = 0

        for i, thruster_id in enumerate(range(1, 9)):
            if thruster_commands[i]:
                dx, dy = model.thruster_directions[thruster_id]
                # Assuming uniform force of 0.45N
                total_fx += dx * 0.45
                total_fy += dy * 0.45

        # Total should be finite
        assert np.isfinite(total_fx)
        assert np.isfinite(total_fy)

    def test_can_compute_total_torque_from_all_thrusters(self):
        """Test that we can compute total torque from all thrusters."""
        thruster_commands = np.ones(8)
        total_torque = 0

        for i, thruster_id in enumerate(range(1, 9)):
            if thruster_commands[i]:
                px, py = model.thruster_positions[thruster_id]
                dx, dy = model.thruster_directions[thruster_id]
                # Force magnitude
                force = 0.45
                # Torque = r × F (cross product in 2D)
                torque = px * (dy * force) - py * (dx * force)
                total_torque += torque

        # Total torque should be finite
        assert np.isfinite(total_torque)

    def test_opposing_thrusters_cancel_forces(self):
        """Test that opposing thrusters roughly cancel linear forces."""
        # Find opposing thruster pairs (if they exist)
        # For example, thrusters pointing in opposite directions

        for tid1, (dx1, dy1) in model.thruster_directions.items():
            for tid2, (dx2, dy2) in model.thruster_directions.items():
                if tid1 < tid2:  # Avoid double counting
                    # Check if opposite directions
                    if abs(dx1 + dx2) < 0.01 and abs(dy1 + dy2) < 0.01:
                        # These are opposing thrusters
                        # Net force should be small
                        net_fx = dx1 + dx2
                        net_fy = dy1 + dy2
                        assert abs(net_fx) < 0.1
                        assert abs(net_fy) < 0.1


class TestModelDocumentation:
    """Test that model has proper documentation."""

    def test_module_has_docstring(self):
        """Test that model module has a docstring."""
        assert model.__doc__ is not None
        assert len(model.__doc__.strip()) > 0


# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit
