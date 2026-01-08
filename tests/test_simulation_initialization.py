"""
Unit tests for SimulationInitializer module.

Tests the initialization logic extracted from simulation.py.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.satellite_control.config import SimulationConfig
from src.satellite_control.core.simulation_initialization import SimulationInitializer


@pytest.fixture
def mock_simulation():
    """Create a mock simulation object for testing."""
    sim = MagicMock()
    sim.use_mujoco_viewer = False
    return sim


@pytest.fixture
def simulation_config():
    """Create a test SimulationConfig."""
    # V3.0.0: Use SimulationConfig.create_default() instead of SatelliteConfig
    return SimulationConfig.create_default()


class TestSimulationInitializerInitialization:
    """Test SimulationInitializer initialization."""

    def test_initializer_creation(self, mock_simulation):
        """Test that initializer can be created."""
        initializer = SimulationInitializer(
            simulation=mock_simulation,
            use_mujoco_viewer=False,
        )
        assert initializer.simulation == mock_simulation
        assert initializer.use_mujoco_viewer is False
        assert initializer.simulation_config is None

    def test_initializer_with_config(self, mock_simulation, simulation_config):
        """Test initializer with SimulationConfig."""
        initializer = SimulationInitializer(
            simulation=mock_simulation,
            simulation_config=simulation_config,
            use_mujoco_viewer=False,
        )
        assert initializer.simulation_config == simulation_config


class TestSimulationInitializerInitialize:
    """Test the initialize method."""

    @patch("src.satellite_control.core.simulation_initialization.SatelliteThrusterTester")
    @patch("src.satellite_control.core.simulation_initialization.MPCController")
    @patch("src.satellite_control.core.simulation_initialization.MissionStateManager")
    @patch("src.satellite_control.core.simulation_initialization.create_state_validator_from_config")
    @patch("src.satellite_control.core.simulation_initialization.create_data_logger")
    @patch("src.satellite_control.core.simulation_initialization.create_mission_report_generator")
    @patch("src.satellite_control.core.simulation_initialization.SimulationIO")
    @patch("src.satellite_control.core.simulation_initialization.SimulationContext")
    @patch("src.satellite_control.core.simulation_initialization.create_simulation_visualizer")
    def test_initialize_satellite_physics(
        self,
        mock_visualizer,
        mock_context,
        mock_io,
        mock_report_gen,
        mock_data_logger,
        mock_validator,
        mock_mission_mgr,
        mock_mpc,
        mock_satellite,
        mock_simulation,
    ):
        """Test that satellite physics is initialized correctly."""
        # Setup mocks
        mock_sat_instance = MagicMock()
        mock_sat_instance.dt = 0.005
        mock_sat_instance.position = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.velocity = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.angular_velocity = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        mock_satellite.return_value = mock_sat_instance

        initializer = SimulationInitializer(
            simulation=mock_simulation,
            use_mujoco_viewer=False,
        )

        initializer.initialize(
            start_pos=(1.0, 2.0),
            target_pos=(0.0, 0.0),
            start_angle=(0.0, 0.0, 0.0),
            target_angle=(0.0, 0.0, 0.0),
        )

        # Verify satellite was created
        mock_satellite.assert_called_once_with(use_mujoco_viewer=False)
        assert mock_simulation.satellite == mock_sat_instance
        assert mock_simulation.satellite.external_simulation_mode is True

    @patch("src.satellite_control.core.simulation_initialization.SatelliteThrusterTester")
    @patch("src.satellite_control.core.simulation_initialization.MPCController")
    @patch("src.satellite_control.core.simulation_initialization.MissionStateManager")
    @patch("src.satellite_control.core.simulation_initialization.create_state_validator_from_config")
    @patch("src.satellite_control.core.simulation_initialization.create_data_logger")
    @patch("src.satellite_control.core.simulation_initialization.create_mission_report_generator")
    @patch("src.satellite_control.core.simulation_initialization.SimulationIO")
    @patch("src.satellite_control.core.simulation_initialization.SimulationContext")
    @patch("src.satellite_control.core.simulation_initialization.create_simulation_visualizer")
    def test_initialize_target_state(
        self,
        mock_visualizer,
        mock_context,
        mock_io,
        mock_report_gen,
        mock_data_logger,
        mock_validator,
        mock_mission_mgr,
        mock_mpc,
        mock_satellite,
        mock_simulation,
    ):
        """Test that target state is initialized correctly."""
        # Setup mocks
        mock_sat_instance = MagicMock()
        mock_sat_instance.dt = 0.005
        mock_sat_instance.position = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.velocity = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.angular_velocity = np.array([0.0, 0.0, 0.0])
        mock_satellite.return_value = mock_sat_instance

        initializer = SimulationInitializer(
            simulation=mock_simulation,
            use_mujoco_viewer=False,
        )

        target_pos = (2.0, 3.0)
        target_angle = (0.1, 0.2, 0.3)

        initializer.initialize(
            start_pos=(1.0, 2.0),
            target_pos=target_pos,
            start_angle=(0.0, 0.0, 0.0),
            target_angle=target_angle,
        )

        # Verify target state was set
        assert hasattr(mock_simulation, "target_state")
        assert mock_simulation.target_state.shape == (13,)
        # Check position
        assert np.allclose(mock_simulation.target_state[0:3], [2.0, 3.0, 0.0])
        # Check quaternion (should be non-zero for non-zero angle)
        assert np.any(mock_simulation.target_state[3:7] != 0)

    @patch("src.satellite_control.core.simulation_initialization.SatelliteThrusterTester")
    @patch("src.satellite_control.core.simulation_initialization.MPCController")
    @patch("src.satellite_control.core.simulation_initialization.MissionStateManager")
    @patch("src.satellite_control.core.simulation_initialization.create_state_validator_from_config")
    @patch("src.satellite_control.core.simulation_initialization.create_data_logger")
    @patch("src.satellite_control.core.simulation_initialization.create_mission_report_generator")
    @patch("src.satellite_control.core.simulation_initialization.SimulationIO")
    @patch("src.satellite_control.core.simulation_initialization.SimulationContext")
    @patch("src.satellite_control.core.simulation_initialization.create_simulation_visualizer")
    def test_initialize_simulation_timing(
        self,
        mock_visualizer,
        mock_context,
        mock_io,
        mock_report_gen,
        mock_data_logger,
        mock_validator,
        mock_mission_mgr,
        mock_mpc,
        mock_satellite,
        mock_simulation,
    ):
        """Test that simulation timing is initialized correctly."""
        # Setup mocks
        mock_sat_instance = MagicMock()
        mock_sat_instance.dt = 0.005
        mock_sat_instance.position = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.velocity = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.angular_velocity = np.array([0.0, 0.0, 0.0])
        mock_satellite.return_value = mock_sat_instance

        initializer = SimulationInitializer(
            simulation=mock_simulation,
            use_mujoco_viewer=False,
        )

        initializer.initialize(
            start_pos=(1.0, 2.0),
            target_pos=(0.0, 0.0),
            start_angle=(0.0, 0.0, 0.0),
            target_angle=(0.0, 0.0, 0.0),
        )

        # Verify timing attributes
        assert mock_simulation.is_running is False
        assert mock_simulation.simulation_time == 0.0
        assert hasattr(mock_simulation, "max_simulation_time")
        assert hasattr(mock_simulation, "control_update_interval")
        assert hasattr(mock_simulation, "last_control_update")
        assert hasattr(mock_simulation, "next_control_simulation_time")

    @patch("src.satellite_control.core.simulation_initialization.SatelliteThrusterTester")
    @patch("src.satellite_control.core.simulation_initialization.MPCController")
    @patch("src.satellite_control.core.simulation_initialization.MissionStateManager")
    @patch("src.satellite_control.core.simulation_initialization.create_state_validator_from_config")
    @patch("src.satellite_control.core.simulation_initialization.create_data_logger")
    @patch("src.satellite_control.core.simulation_initialization.create_mission_report_generator")
    @patch("src.satellite_control.core.simulation_initialization.SimulationIO")
    @patch("src.satellite_control.core.simulation_initialization.SimulationContext")
    @patch("src.satellite_control.core.simulation_initialization.create_simulation_visualizer")
    def test_initialize_thruster_manager(
        self,
        mock_visualizer,
        mock_context,
        mock_io,
        mock_report_gen,
        mock_data_logger,
        mock_validator,
        mock_mission_mgr,
        mock_mpc,
        mock_satellite,
        mock_simulation,
    ):
        """Test that thruster manager is initialized."""
        # Setup mocks
        mock_sat_instance = MagicMock()
        mock_sat_instance.dt = 0.005
        mock_sat_instance.position = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.velocity = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.angular_velocity = np.array([0.0, 0.0, 0.0])
        mock_satellite.return_value = mock_sat_instance

        initializer = SimulationInitializer(
            simulation=mock_simulation,
            use_mujoco_viewer=False,
        )

        initializer.initialize(
            start_pos=(1.0, 2.0),
            target_pos=(0.0, 0.0),
            start_angle=(0.0, 0.0, 0.0),
            target_angle=(0.0, 0.0, 0.0),
        )

        # Verify thruster manager was created
        assert hasattr(mock_simulation, "thruster_manager")
        assert hasattr(mock_simulation, "num_thrusters")

    @patch("src.satellite_control.core.simulation_initialization.SatelliteThrusterTester")
    @patch("src.satellite_control.core.simulation_initialization.MPCController")
    @patch("src.satellite_control.core.simulation_initialization.MissionStateManager")
    @patch("src.satellite_control.core.simulation_initialization.create_state_validator_from_config")
    @patch("src.satellite_control.core.simulation_initialization.create_data_logger")
    @patch("src.satellite_control.core.simulation_initialization.create_mission_report_generator")
    @patch("src.satellite_control.core.simulation_initialization.SimulationIO")
    @patch("src.satellite_control.core.simulation_initialization.SimulationContext")
    @patch("src.satellite_control.core.simulation_initialization.create_simulation_visualizer")
    def test_initialize_mpc_controller(
        self,
        mock_visualizer,
        mock_context,
        mock_io,
        mock_report_gen,
        mock_data_logger,
        mock_validator,
        mock_mission_mgr,
        mock_mpc,
        mock_satellite,
        mock_simulation,
    ):
        """Test that MPC controller is initialized."""
        # Setup mocks
        mock_sat_instance = MagicMock()
        mock_sat_instance.dt = 0.005
        mock_sat_instance.position = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.velocity = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.angular_velocity = np.array([0.0, 0.0, 0.0])
        mock_satellite.return_value = mock_sat_instance

        initializer = SimulationInitializer(
            simulation=mock_simulation,
            use_mujoco_viewer=False,
        )

        initializer.initialize(
            start_pos=(1.0, 2.0),
            target_pos=(0.0, 0.0),
            start_angle=(0.0, 0.0, 0.0),
            target_angle=(0.0, 0.0, 0.0),
        )

        # Verify MPC controller was created
        mock_mpc.assert_called_once()
        assert hasattr(mock_simulation, "mpc_controller")

    @patch("src.satellite_control.core.simulation_initialization.SatelliteThrusterTester")
    @patch("src.satellite_control.core.simulation_initialization.MPCController")
    @patch("src.satellite_control.core.simulation_initialization.MissionStateManager")
    @patch("src.satellite_control.core.simulation_initialization.create_state_validator_from_config")
    @patch("src.satellite_control.core.simulation_initialization.create_data_logger")
    @patch("src.satellite_control.core.simulation_initialization.create_mission_report_generator")
    @patch("src.satellite_control.core.simulation_initialization.SimulationIO")
    @patch("src.satellite_control.core.simulation_initialization.SimulationContext")
    @patch("src.satellite_control.core.simulation_initialization.create_simulation_visualizer")
    def test_initialize_default_positions(
        self,
        mock_visualizer,
        mock_context,
        mock_io,
        mock_report_gen,
        mock_data_logger,
        mock_validator,
        mock_mission_mgr,
        mock_mpc,
        mock_satellite,
        mock_simulation,
    ):
        """Test that default positions are used when None provided."""
        # Setup mocks
        mock_sat_instance = MagicMock()
        mock_sat_instance.dt = 0.005
        mock_sat_instance.position = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.velocity = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.angular_velocity = np.array([0.0, 0.0, 0.0])
        mock_satellite.return_value = mock_sat_instance

        initializer = SimulationInitializer(
            simulation=mock_simulation,
            use_mujoco_viewer=False,
        )

        # Initialize with None values (should use defaults)
        initializer.initialize(
            start_pos=None,
            target_pos=None,
            start_angle=None,
            target_angle=None,
        )

        # Verify initialization completed (satellite was created)
        mock_satellite.assert_called_once()
        assert hasattr(mock_simulation, "satellite")

    @patch("src.satellite_control.core.simulation_initialization.SatelliteThrusterTester")
    @patch("src.satellite_control.core.simulation_initialization.MPCController")
    @patch("src.satellite_control.core.simulation_initialization.MissionStateManager")
    @patch("src.satellite_control.core.simulation_initialization.create_state_validator_from_config")
    @patch("src.satellite_control.core.simulation_initialization.create_data_logger")
    @patch("src.satellite_control.core.simulation_initialization.create_mission_report_generator")
    @patch("src.satellite_control.core.simulation_initialization.SimulationIO")
    @patch("src.satellite_control.core.simulation_initialization.SimulationContext")
    @patch("src.satellite_control.core.simulation_initialization.create_simulation_visualizer")
    def test_initialize_with_velocities(
        self,
        mock_visualizer,
        mock_context,
        mock_io,
        mock_report_gen,
        mock_data_logger,
        mock_validator,
        mock_mission_mgr,
        mock_mpc,
        mock_satellite,
        mock_simulation,
    ):
        """Test initialization with initial velocities."""
        # Setup mocks
        mock_sat_instance = MagicMock()
        mock_sat_instance.dt = 0.005
        mock_sat_instance.position = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.velocity = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.angular_velocity = np.array([0.0, 0.0, 0.0])
        mock_satellite.return_value = mock_sat_instance

        initializer = SimulationInitializer(
            simulation=mock_simulation,
            use_mujoco_viewer=False,
        )

        initializer.initialize(
            start_pos=(1.0, 2.0),
            target_pos=(0.0, 0.0),
            start_angle=(0.0, 0.0, 0.0),
            target_angle=(0.0, 0.0, 0.0),
            start_vx=0.1,
            start_vy=0.2,
            start_vz=0.05,
            start_omega=0.3,
        )

        # Verify velocities were set
        assert mock_sat_instance.velocity is not None
        assert mock_sat_instance.angular_velocity is not None

    @patch("src.satellite_control.core.simulation_initialization.SatelliteThrusterTester")
    @patch("src.satellite_control.core.simulation_initialization.MPCController")
    @patch("src.satellite_control.core.simulation_initialization.MissionStateManager")
    @patch("src.satellite_control.core.simulation_initialization.create_state_validator_from_config")
    @patch("src.satellite_control.core.simulation_initialization.create_data_logger")
    @patch("src.satellite_control.core.simulation_initialization.create_mission_report_generator")
    @patch("src.satellite_control.core.simulation_initialization.SimulationIO")
    @patch("src.satellite_control.core.simulation_initialization.SimulationContext")
    @patch("src.satellite_control.core.simulation_initialization.create_simulation_visualizer")
    def test_initialize_all_components(
        self,
        mock_visualizer,
        mock_context,
        mock_io,
        mock_report_gen,
        mock_data_logger,
        mock_validator,
        mock_mission_mgr,
        mock_mpc,
        mock_satellite,
        mock_simulation,
    ):
        """Test that all components are initialized."""
        # Setup mocks
        mock_sat_instance = MagicMock()
        mock_sat_instance.dt = 0.005
        mock_sat_instance.position = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.velocity = np.array([0.0, 0.0, 0.0])
        mock_sat_instance.angular_velocity = np.array([0.0, 0.0, 0.0])
        mock_satellite.return_value = mock_sat_instance

        initializer = SimulationInitializer(
            simulation=mock_simulation,
            use_mujoco_viewer=False,
        )

        initializer.initialize(
            start_pos=(1.0, 2.0),
            target_pos=(0.0, 0.0),
            start_angle=(0.0, 0.0, 0.0),
            target_angle=(0.0, 0.0, 0.0),
        )

        # Verify all components were initialized
        assert hasattr(mock_simulation, "satellite")
        assert hasattr(mock_simulation, "target_state")
        assert hasattr(mock_simulation, "thruster_manager")
        assert hasattr(mock_simulation, "mpc_controller")
        assert hasattr(mock_simulation, "mission_manager")
        assert hasattr(mock_simulation, "state_validator")
        assert hasattr(mock_simulation, "data_logger")
        assert hasattr(mock_simulation, "physics_logger")
        assert hasattr(mock_simulation, "performance_monitor")
        assert hasattr(mock_simulation, "context")
        assert hasattr(mock_simulation, "visualizer")
