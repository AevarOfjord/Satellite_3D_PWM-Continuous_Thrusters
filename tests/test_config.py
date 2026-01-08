"""
Unit tests for config.py module.

Tests configuration validation, parameter access, and configuration methods.

V4.0.0: Tests updated to use SimulationConfig where possible.
Some tests still use SatelliteConfig for deprecated API compatibility.
"""

import numpy as np
import pytest

# V4.0.0: Prefer SimulationConfig, but keep SatelliteConfig for deprecated API tests
from src.satellite_control.config import SatelliteConfig  # DEPRECATED
from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.config.validator import ConfigValidator


class TestSatelliteConfigValidation:
    """Test configuration validation methods (V4.0.0: uses SimulationConfig)."""

    def test_validate_parameters_success(self):
        """Test that default parameters pass validation."""
        # V4.0.0: Use SimulationConfig and ConfigValidator instead of SatelliteConfig
        config = SimulationConfig.create_default()
        # Should not raise any exceptions
        ConfigValidator.validate_and_raise(config.app_config)

    def test_physical_parameters_are_positive(self):
        """Test that physical parameters are positive values."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        physics = config.app_config.physics

        assert physics.total_mass > 0, "Mass must be positive"
        assert physics.moment_of_inertia > 0, "Inertia must be positive"
        assert physics.satellite_size > 0, "Size must be positive"

    def test_thruster_forces_are_positive(self):
        """Test that all thruster forces are positive."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        physics = config.app_config.physics

        for thruster_id, force in physics.thruster_forces.items():
            assert force > 0, f"Thruster {thruster_id} force must be positive"

    def test_mpc_horizons_are_valid(self):
        """Test that MPC horizons are valid."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        mpc = config.app_config.mpc

        assert mpc.prediction_horizon > 0
        assert mpc.control_horizon > 0
        assert mpc.control_horizon <= mpc.prediction_horizon

    def test_cost_weights_are_positive(self):
        """Test that MPC cost weights are positive."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        mpc = config.app_config.mpc

        assert mpc.q_position >= 0
        assert mpc.q_velocity >= 0
        assert mpc.q_angle >= 0
        assert mpc.q_angular_velocity >= 0
        assert mpc.r_thrust >= 0
        # Note: r_switch is not in MPCParams (was in old config)


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
    """Test configuration setter methods (V4.0.0: tests deprecated API for compatibility)."""

    def test_set_thruster_force_single(self):
        """Test setting a single thruster force (deprecated API)."""
        # V4.0.0: This tests deprecated SatelliteConfig API
        original = SatelliteConfig.THRUSTER_FORCES.copy()

        try:
            SatelliteConfig.set_thruster_force(1, 0.5)
            params = SatelliteConfig.get_satellite_params()
            assert params["thruster_forces"][1] == 0.5
        finally:
            # Restore original
            SatelliteConfig.THRUSTER_FORCES = original

    def test_set_thruster_force_invalid_id(self):
        """Test that invalid thruster ID raises error (deprecated API)."""
        # V4.0.0: This tests deprecated SatelliteConfig API
        with pytest.raises((ValueError, KeyError)):
            SatelliteConfig.set_thruster_force(0, 0.5)  # ID should be 1-12 (3D)

        with pytest.raises((ValueError, KeyError)):
            SatelliteConfig.set_thruster_force(13, 0.5)  # ID should be 1-12 (3D)

    def test_set_thruster_force_negative(self):
        """Test that negative force raises error or is handled (deprecated API)."""
        # V4.0.0: This tests deprecated SatelliteConfig API
        try:
            SatelliteConfig.set_thruster_force(1, -0.5)
            # If it doesn't raise, verify it's handled somehow
            params = SatelliteConfig.get_satellite_params()
            assert params["thruster_forces"][1] >= 0
        except ValueError:
            # Expected behavior - negative forces rejected
            pass

    def test_set_all_thruster_forces(self):
        """Test setting all thruster forces at once (deprecated API)."""
        # V4.0.0: This tests deprecated SatelliteConfig API
        original = SatelliteConfig.THRUSTER_FORCES.copy()

        try:
            SatelliteConfig.set_all_thruster_forces(0.6)
            params = SatelliteConfig.get_satellite_params()

            # V4.0.0: 3D system has 12 thrusters
            for thruster_id in range(1, 13):
                assert params["thruster_forces"][thruster_id] == 0.6
        finally:
            # Restore original
            SatelliteConfig.THRUSTER_FORCES = original


class TestSatelliteConfigMissionModes:
    """Test mission mode configuration (V4.0.0: tests deprecated API for compatibility)."""

    def test_set_point_to_point_mode(self):
        """Test setting point-to-point mission mode (deprecated API)."""
        # V4.0.0: This tests deprecated SatelliteConfig API
        # Note: In V4.0.0, use MissionState.enable_waypoint_mode instead
        original_mode = getattr(SatelliteConfig, "ENABLE_WAYPOINT_MODE", False)

        try:
            SatelliteConfig.ENABLE_WAYPOINT_MODE = False
            assert SatelliteConfig.ENABLE_WAYPOINT_MODE is False
        finally:
            SatelliteConfig.ENABLE_WAYPOINT_MODE = original_mode

    def test_set_multi_point_mode(self):
        """Test setting multi-point mission mode (waypoint mode) (deprecated API)."""
        # V4.0.0: This tests deprecated SatelliteConfig API
        # Note: In V4.0.0, use MissionState.waypoint_targets instead
        original_mode = getattr(SatelliteConfig, "ENABLE_WAYPOINT_MODE", False)
        original_targets = (
            getattr(SatelliteConfig, "WAYPOINT_TARGETS", []).copy()
            if hasattr(SatelliteConfig, "WAYPOINT_TARGETS")
            else []
        )

        try:
            targets = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]  # 3D
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
    """Test configuration constants and boundaries (V4.0.0: uses SimulationConfig)."""

    def test_simulation_dt_is_reasonable(self):
        """Test that simulation timestep is reasonable."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        dt = config.app_config.simulation.dt
        assert 0.001 <= dt <= 0.1

    def test_control_dt_is_reasonable(self):
        """Test that control timestep is reasonable."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        control_dt = config.app_config.simulation.control_dt
        assert 0.01 <= control_dt <= 1.0

    def test_control_dt_is_multiple_of_simulation_dt(self):
        """Test that control DT is a multiple of simulation DT."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        dt = config.app_config.simulation.dt
        control_dt = config.app_config.simulation.control_dt
        ratio = control_dt / dt
        assert (
            abs(ratio - round(ratio)) < 1e-6
        ), "Control DT should be multiple of simulation DT"

    def test_solver_time_limit_less_than_control_dt(self):
        """Test that solver has time budget within control interval."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        solver_time_limit = config.app_config.mpc.solver_time_limit
        control_dt = config.app_config.simulation.control_dt
        assert solver_time_limit < control_dt

    def test_workspace_bounds_are_positive(self):
        """Test that workspace boundaries are positive."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        position_bounds = config.app_config.mpc.position_bounds
        assert position_bounds > 0

    def test_velocity_limits_are_positive(self):
        """Test that velocity limits are positive."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        mpc = config.app_config.mpc
        assert mpc.max_velocity > 0
        assert mpc.max_angular_velocity > 0


class TestSatelliteConfigThrusterGeometry:
    """Test thruster positions and directions (V4.0.0: uses SimulationConfig)."""

    def test_thruster_positions_are_valid(self):
        """Test that thruster positions are within satellite bounds."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        physics = config.app_config.physics
        half_size = physics.satellite_size / 2

        for thruster_id, pos in physics.thruster_positions.items():
            x, y = pos[0], pos[1]
            assert (
                abs(x) <= half_size * 1.1
            ), f"Thruster {thruster_id} x position out of bounds"
            assert (
                abs(y) <= half_size * 1.1
            ), f"Thruster {thruster_id} y position out of bounds"

    def test_thruster_directions_are_unit_vectors(self):
        """Test that thruster directions are unit vectors."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        physics = config.app_config.physics

        for thruster_id, direction in physics.thruster_directions.items():
            dx, dy = direction[0], direction[1]
            magnitude = np.sqrt(dx**2 + dy**2)
            assert (
                abs(magnitude - 1.0) < 1e-6
            ), f"Thruster {thruster_id} direction not unit vector"

    def test_thruster_ids_are_consistent(self):
        """Test that thruster IDs are consistent across dictionaries."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        physics = config.app_config.physics

        force_ids = set(physics.thruster_forces.keys())
        position_ids = set(physics.thruster_positions.keys())
        direction_ids = set(physics.thruster_directions.keys())

        assert force_ids == position_ids == direction_ids
        # Note: V4.0.0 uses 12 thrusters (3D), not 8 (2D)
        assert len(force_ids) == 12


class TestSatelliteConfigIntegration:
    """Integration tests for configuration system (V4.0.0: uses SimulationConfig)."""

    def test_config_supports_simulation_initialization(self):
        """Test that config provides all needed params for simulation."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()

        # Should be able to create simulation with these params
        assert config.app_config.physics is not None
        assert config.app_config.mpc is not None
        assert config.app_config.simulation is not None

    def test_config_consistency_after_multiple_gets(self):
        """Test that config returns consistent values across multiple calls."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        config1 = SimulationConfig.create_default()
        config2 = SimulationConfig.create_default()

        assert config1.app_config.physics.total_mass == config2.app_config.physics.total_mass
        assert config1.app_config.physics.moment_of_inertia == config2.app_config.physics.moment_of_inertia

    def test_print_thruster_forces_does_not_crash(self):
        """Test that print_thruster_forces executes without error (deprecated API)."""
        # V4.0.0: This tests deprecated SatelliteConfig API
        # Should not raise any exceptions
        SatelliteConfig.print_thruster_forces()


class TestConfigValidator:
    """Test comprehensive configuration validator."""

    def test_validator_validates_default_config(self):
        """Test that default configuration passes validation."""
        from src.satellite_control.config import SatelliteConfig

        app_config = SatelliteConfig.get_app_config()
        validator = ConfigValidator()
        issues = validator.validate_all(app_config)

        # Default config should be valid
        assert len(issues) == 0, f"Default config has validation issues: {issues}"

    def test_validator_detects_invalid_mass(self):
        """Test that validator detects invalid mass."""
        from src.satellite_control.config.models import AppConfig, MPCParams, SatellitePhysicalParams, SimulationParams

        # Create config with invalid mass
        invalid_config = AppConfig(
            physics=SatellitePhysicalParams(
                total_mass=-1.0,  # Invalid: negative mass
                moment_of_inertia=0.312,
                satellite_size=0.29,
                com_offset=(0.0, 0.0, 0.0),
                thruster_positions={1: (0.1, 0.1, 0.0)},
                thruster_directions={1: (1.0, 0.0, 0.0)},
                thruster_forces={1: 0.45},
            ),
            mpc=MPCParams(
                prediction_horizon=15,
                control_horizon=12,
                dt=0.06,
                solver_time_limit=0.05,
                solver_type="OSQP",
                q_position=1000.0,
                q_velocity=1750.0,
                q_angle=1000.0,
                q_angular_velocity=1500.0,
                r_thrust=1.0,
                max_velocity=0.15,
                max_angular_velocity=1.57,
                position_bounds=3.0,
                damping_zone=0.1,
                velocity_threshold=0.03,
                max_velocity_weight=100.0,
                thruster_type="PWM",
            ),
            simulation=SimulationParams(
                dt=0.005,
                max_duration=500.0,
                headless=True,
                window_width=700,
                window_height=600,
            ),
        )

        validator = ConfigValidator()
        issues = validator.validate_all(invalid_config)

        # Should detect invalid mass
        assert len(issues) > 0, "Validator should detect invalid mass"
        assert any("mass" in issue.lower() for issue in issues), "Should report mass issue"

    def test_validator_detects_invalid_mpc_horizon(self):
        """Test that validator detects invalid MPC horizon."""
        from src.satellite_control.config import SatelliteConfig
        from src.satellite_control.config.models import AppConfig

        app_config = SatelliteConfig.get_app_config()

        # Create invalid config with control horizon > prediction horizon
        invalid_config = AppConfig(
            physics=app_config.physics,
            mpc=app_config.mpc.model_copy(update={"control_horizon": 20, "prediction_horizon": 15}),
            simulation=app_config.simulation,
        )

        validator = ConfigValidator()
        issues = validator.validate_all(invalid_config)

        # Should detect invalid horizon relationship
        assert len(issues) > 0, "Validator should detect invalid horizon"
        assert any("horizon" in issue.lower() for issue in issues), "Should report horizon issue"


# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit
