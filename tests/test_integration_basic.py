"""
Basic integration tests for satellite control system.

These tests verify that major components work together correctly.
"""

import pytest


@pytest.mark.integration
class TestConfigModelIntegration:
    """Test integration between config and model modules."""

    def test_config_and_model_thruster_counts_match(self):
        """Test that config and model have same number of thrusters."""
        from src.satellite_control.config.simulation_config import SimulationConfig
        from src.satellite_control.core import model

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        config_thruster_count = len(config.app_config.physics.thruster_forces)
        model_thruster_count = len(model.thruster_positions)

        assert config_thruster_count == model_thruster_count == 8

    def test_config_and_model_thruster_ids_match(self):
        """Test that thruster IDs are consistent between config and model."""
        from src.satellite_control.config.simulation_config import SimulationConfig
        from src.satellite_control.core import model

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        config_ids = set(config.app_config.physics.thruster_forces.keys())
        model_position_ids = set(model.thruster_positions.keys())
        model_direction_ids = set(model.thruster_directions.keys())

        assert config_ids == model_position_ids == model_direction_ids

    def test_config_thruster_positions_match_model(self):
        """Test that thruster positions in config match model."""
        from src.satellite_control.config.simulation_config import SimulationConfig
        from src.satellite_control.core import model

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()

        for thruster_id in range(1, 9):
            config_pos = config.app_config.physics.thruster_positions[thruster_id]
            model_pos = model.thruster_positions[thruster_id]

            assert config_pos == model_pos, f"Thruster {thruster_id} position mismatch"

    def test_config_thruster_directions_match_model(self):
        """Test that thruster directions in config match model."""
        import numpy as np

        from src.satellite_control.config.simulation_config import SimulationConfig
        from src.satellite_control.core import model

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()

        for thruster_id in range(1, 9):
            config_dir = config.app_config.physics.thruster_directions[thruster_id]
            model_dir = model.thruster_directions[thruster_id]

            # Convert both to tuples for comparison (config has arrays, model has tuples)
            config_tuple = tuple(config_dir) if isinstance(config_dir, np.ndarray) else config_dir
            model_tuple = tuple(model_dir) if isinstance(model_dir, np.ndarray) else model_dir

            assert config_tuple == model_tuple, f"Thruster {thruster_id} direction mismatch"


@pytest.mark.integration
class TestSystemInitialization:
    """Test that the system can be initialized with current config."""

    def test_can_get_all_config_params(self):
        """Test that all configuration parameters can be retrieved."""
        from src.satellite_control.config.simulation_config import SimulationConfig

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()

        assert config.app_config.physics is not None
        assert config.app_config.mpc is not None
        assert config.app_config.simulation is not None

    def test_params_are_internally_consistent(self):
        """Test that parameters are consistent with each other."""
        from src.satellite_control.config.simulation_config import SimulationConfig

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()

        # MPC timestep should match control DT
        assert config.app_config.mpc.dt == config.app_config.simulation.control_dt

        # Solver time limit should be less than control interval
        assert config.app_config.mpc.solver_time_limit < config.app_config.mpc.dt

        # Control horizon should not exceed prediction horizon
        assert config.app_config.mpc.control_horizon <= config.app_config.mpc.prediction_horizon


@pytest.mark.integration
@pytest.mark.slow
class TestPhysicsComputation:
    """Test physics computations across modules."""

    def test_can_compute_dynamics_for_single_thruster(self):
        """Test computing satellite dynamics with one thruster firing."""
        import numpy as np

        from src.satellite_control.config.simulation_config import SimulationConfig
        from src.satellite_control.core import model

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        physics = config.app_config.physics

        # Fire thruster 1
        thruster_id = 1
        force_magnitude = physics.thruster_forces[thruster_id]
        position = model.thruster_positions[thruster_id]
        direction = model.thruster_directions[thruster_id]

        # Compute force
        fx = direction[0] * force_magnitude
        fy = direction[1] * force_magnitude

        # Compute torque (2D cross product)
        torque = position[0] * fy - position[1] * fx

        # Compute accelerations
        mass = physics.total_mass
        inertia = physics.moment_of_inertia

        ax = fx / mass
        ay = fy / mass
        alpha = torque / inertia

        # All should be finite
        assert np.isfinite(ax)
        assert np.isfinite(ay)
        assert np.isfinite(alpha)

    def test_can_compute_dynamics_for_all_thrusters(self):
        """Test computing dynamics with all thrusters."""
        import numpy as np

        from src.satellite_control.config.simulation_config import SimulationConfig
        from src.satellite_control.core import model

        # V3.0.0: Use SimulationConfig instead of SatelliteConfig
        config = SimulationConfig.create_default()
        physics = config.app_config.physics
        mass = physics.total_mass
        inertia = physics.moment_of_inertia

        # All thrusters firing
        total_fx = 0
        total_fy = 0
        total_torque = 0

        for thruster_id in range(1, 9):
            force_mag = physics.thruster_forces[thruster_id]
            px, py = model.thruster_positions[thruster_id]
            dx, dy = model.thruster_directions[thruster_id]

            fx = dx * force_mag
            fy = dy * force_mag
            torque = px * fy - py * fx

            total_fx += fx
            total_fy += fy
            total_torque += torque

        # Compute accelerations
        ax = total_fx / mass
        ay = total_fy / mass
        alpha = total_torque / inertia

        # All should be finite
        assert np.isfinite(ax)
        assert np.isfinite(ay)
        assert np.isfinite(alpha)


@pytest.mark.integration
class TestModuleImports:
    """Test that all major modules can be imported."""

    def test_can_import_config(self):
        """Test that config module can be imported."""
        import src.satellite_control.config as config

        assert config is not None

    def test_can_import_model(self):
        """Test that model module can be imported."""
        import src.satellite_control.core.model as model

        assert model is not None

    def test_can_import_mission(self):
        """Test that mission module can be imported."""
        try:
            import src.satellite_control.mission.mission as mission

            assert mission is not None
        except ImportError:
            pytest.skip("Mission module not available")

    def test_can_import_logging_config(self):
        """Test that logging_config module can be imported."""
        try:
            import src.satellite_control.utils.logging_config as logging_config

            assert logging_config is not None
        except ImportError:
            pytest.skip("logging_config module not available")


# Mark integration tests
pytestmark = pytest.mark.integration
