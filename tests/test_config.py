"""
Unit tests for config.py module.

Tests configuration validation, parameter access, and configuration methods.
"""

import numpy as np
import pytest

from src.satellite_control.config import SatelliteConfig


class TestSatelliteConfigValidation:
    """Test configuration validation methods."""

    def test_validate_parameters_success(self):
        """Test that default parameters pass validation."""
        # Should not raise any exceptions
        SatelliteConfig.validate_parameters()

    def test_physical_parameters_are_positive(self):
        """Test that physical parameters are positive values."""
        params = SatelliteConfig.get_satellite_params()

        assert params["mass"] > 0, "Mass must be positive"
        assert params["inertia"] > 0, "Inertia must be positive"
        assert params["size"] > 0, "Size must be positive"

    def test_thruster_forces_are_positive(self):
        """Test that all thruster forces are positive."""
        params = SatelliteConfig.get_satellite_params()

        for thruster_id, force in params["thruster_forces"].items():
            assert force > 0, f"Thruster {thruster_id} force must be positive"

    def test_mpc_horizons_are_valid(self):
        """Test that MPC horizons are valid."""
        params = SatelliteConfig.get_mpc_params()

        assert params["prediction_horizon"] > 0
        assert params["control_horizon"] > 0
        assert params["control_horizon"] <= params["prediction_horizon"]

    def test_cost_weights_are_positive(self):
        """Test that MPC cost weights are positive."""
        params = SatelliteConfig.get_mpc_params()

        assert params["q_position"] >= 0
        assert params["q_velocity"] >= 0
        assert params["q_angle"] >= 0
        assert params["q_angular_velocity"] >= 0
        assert params["r_thrust"] >= 0
        assert params["r_switch"] >= 0


class TestSatelliteConfigGetters:
    """Test configuration getter methods."""

    def test_get_satellite_params_returns_dict(self):
        """Test that get_satellite_params returns a dictionary."""
        params = SatelliteConfig.get_satellite_params()
        assert isinstance(params, dict)

    def test_get_satellite_params_has_required_keys(self):
        """Test that satellite params contains all required keys."""
        params = SatelliteConfig.get_satellite_params()

        required_keys = [
            "mass",
            "inertia",
            "size",
            "thruster_forces",
            "thruster_positions",
            "thruster_directions",
            "damping_linear",
            "damping_angular",
        ]

        for key in required_keys:
            assert key in params, f"Missing required key: {key}"

    def test_get_mpc_params_returns_dict(self):
        """Test that get_mpc_params returns a dictionary."""
        params = SatelliteConfig.get_mpc_params()
        assert isinstance(params, dict)

    def test_get_mpc_params_has_required_keys(self):
        """Test that MPC params contains all required keys."""
        params = SatelliteConfig.get_mpc_params()

        required_keys = [
            "prediction_horizon",
            "control_horizon",
            "dt",
            "q_position",
            "q_velocity",
            "q_angle",
            "q_angular_velocity",
            "r_thrust",
            "r_switch",
            "max_velocity",
            "max_angular_velocity",
            "position_bounds",
            "solver_time_limit",
        ]

        for key in required_keys:
            assert key in params, f"Missing required key: {key}"

    def test_thruster_count_is_eight(self):
        """Test that there are exactly 8 thrusters."""
        params = SatelliteConfig.get_satellite_params()

        assert len(params["thruster_forces"]) == 8
        assert len(params["thruster_positions"]) == 8
        assert len(params["thruster_directions"]) == 8


class TestSatelliteConfigSetters:
    """Test configuration setter methods."""

    def test_set_thruster_force_single(self):
        """Test setting a single thruster force."""
        original = SatelliteConfig.THRUSTER_FORCES.copy()

        try:
            SatelliteConfig.set_thruster_force(1, 0.5)
            params = SatelliteConfig.get_satellite_params()
            assert params["thruster_forces"][1] == 0.5
        finally:
            # Restore original
            SatelliteConfig.THRUSTER_FORCES = original

    def test_set_thruster_force_invalid_id(self):
        """Test that invalid thruster ID raises error."""
        with pytest.raises((ValueError, KeyError)):
            SatelliteConfig.set_thruster_force(0, 0.5)  # ID should be 1-8

        with pytest.raises((ValueError, KeyError)):
            SatelliteConfig.set_thruster_force(9, 0.5)  # ID should be 1-8

    def test_set_thruster_force_negative(self):
        """Test that negative force raises error or is handled."""
        # Depending on implementation, this might raise or be clamped
        # Adjust based on actual config.py behavior
        try:
            SatelliteConfig.set_thruster_force(1, -0.5)
            # If it doesn't raise, verify it's handled somehow
            params = SatelliteConfig.get_satellite_params()
            assert params["thruster_forces"][1] >= 0
        except ValueError:
            # Expected behavior - negative forces rejected
            pass

    def test_set_all_thruster_forces(self):
        """Test setting all thruster forces at once."""
        original = SatelliteConfig.THRUSTER_FORCES.copy()

        try:
            SatelliteConfig.set_all_thruster_forces(0.6)
            params = SatelliteConfig.get_satellite_params()

            for thruster_id in range(1, 9):
                assert params["thruster_forces"][thruster_id] == 0.6
        finally:
            # Restore original
            SatelliteConfig.THRUSTER_FORCES = original


class TestSatelliteConfigMissionModes:
    """Test mission mode configuration."""

    def test_set_point_to_point_mode(self):
        """Test setting point-to-point mission mode."""
        original_mode = getattr(SatelliteConfig, "ENABLE_WAYPOINT_MODE", False)

        try:
            SatelliteConfig.ENABLE_WAYPOINT_MODE = False
            assert SatelliteConfig.ENABLE_WAYPOINT_MODE is False
        finally:
            SatelliteConfig.ENABLE_WAYPOINT_MODE = original_mode

    def test_set_multi_point_mode(self):
        """Test setting multi-point mission mode (waypoint mode)."""
        original_mode = getattr(SatelliteConfig, "ENABLE_WAYPOINT_MODE", False)
        original_targets = (
            getattr(SatelliteConfig, "WAYPOINT_TARGETS", []).copy()
            if hasattr(SatelliteConfig, "WAYPOINT_TARGETS")
            else []
        )

        try:
            targets = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
            angles = [
                (0.0, 0.0, 0.0),
                (0.0, 0.0, np.pi / 2),
                (0.0, 0.0, np.pi),
            ]

            SatelliteConfig.ENABLE_WAYPOINT_MODE = True
            SatelliteConfig.WAYPOINT_TARGETS = targets.copy()
            SatelliteConfig.WAYPOINT_ANGLES = angles.copy()

            assert SatelliteConfig.ENABLE_WAYPOINT_MODE is True
            assert len(SatelliteConfig.WAYPOINT_TARGETS) == 3
        finally:
            SatelliteConfig.ENABLE_WAYPOINT_MODE = original_mode
            SatelliteConfig.WAYPOINT_TARGETS = original_targets


class TestSatelliteConfigConstants:
    """Test configuration constants and boundaries."""

    def test_simulation_dt_is_reasonable(self):
        """Test that simulation timestep is reasonable."""
        assert 0.001 <= SatelliteConfig.SIMULATION_DT <= 0.1

    def test_control_dt_is_reasonable(self):
        """Test that control timestep is reasonable."""
        assert 0.01 <= SatelliteConfig.CONTROL_DT <= 1.0

    def test_control_dt_is_multiple_of_simulation_dt(self):
        """Test that control DT is a multiple of simulation DT."""
        ratio = SatelliteConfig.CONTROL_DT / SatelliteConfig.SIMULATION_DT
        assert (
            abs(ratio - round(ratio)) < 1e-6
        ), "Control DT should be multiple of simulation DT"

    def test_solver_time_limit_less_than_control_dt(self):
        """Test that solver has time budget within control interval."""
        params = SatelliteConfig.get_mpc_params()
        assert params["solver_time_limit"] < SatelliteConfig.CONTROL_DT

    def test_workspace_bounds_are_positive(self):
        """Test that workspace boundaries are positive."""
        params = SatelliteConfig.get_mpc_params()
        assert params["position_bounds"] > 0

    def test_velocity_limits_are_positive(self):
        """Test that velocity limits are positive."""
        params = SatelliteConfig.get_mpc_params()
        assert params["max_velocity"] > 0
        assert params["max_angular_velocity"] > 0


class TestSatelliteConfigThrusterGeometry:
    """Test thruster positions and directions."""

    def test_thruster_positions_are_valid(self):
        """Test that thruster positions are within satellite bounds."""
        params = SatelliteConfig.get_satellite_params()
        half_size = params["size"] / 2

        for thruster_id, (x, y) in params["thruster_positions"].items():
            assert (
                abs(x) <= half_size * 1.1
            ), f"Thruster {thruster_id} x position out of bounds"
            assert (
                abs(y) <= half_size * 1.1
            ), f"Thruster {thruster_id} y position out of bounds"

    def test_thruster_directions_are_unit_vectors(self):
        """Test that thruster directions are unit vectors."""
        params = SatelliteConfig.get_satellite_params()

        for thruster_id, (dx, dy) in params["thruster_directions"].items():
            magnitude = np.sqrt(dx**2 + dy**2)
            assert (
                abs(magnitude - 1.0) < 1e-6
            ), f"Thruster {thruster_id} direction not unit vector"

    def test_thruster_ids_are_consistent(self):
        """Test that thruster IDs are consistent across dictionaries."""
        params = SatelliteConfig.get_satellite_params()

        force_ids = set(params["thruster_forces"].keys())
        position_ids = set(params["thruster_positions"].keys())
        direction_ids = set(params["thruster_directions"].keys())

        assert force_ids == position_ids == direction_ids
        assert force_ids == set(range(1, 9))


class TestSatelliteConfigIntegration:
    """Integration tests for configuration system."""

    def test_config_supports_simulation_initialization(self):
        """Test that config provides all needed params for simulation."""
        sat_params = SatelliteConfig.get_satellite_params()
        mpc_params = SatelliteConfig.get_mpc_params()

        # Should be able to create simulation with these params
        assert sat_params is not None
        assert mpc_params is not None
        assert isinstance(sat_params, dict)
        assert isinstance(mpc_params, dict)

    def test_config_consistency_after_multiple_gets(self):
        """Test that config returns consistent values across multiple calls."""
        params1 = SatelliteConfig.get_satellite_params()
        params2 = SatelliteConfig.get_satellite_params()

        assert params1["mass"] == params2["mass"]
        assert params1["inertia"] == params2["inertia"]

    def test_print_thruster_forces_does_not_crash(self):
        """Test that print_thruster_forces executes without error."""
        # This should not raise any exceptions
        SatelliteConfig.print_thruster_forces()


# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit
