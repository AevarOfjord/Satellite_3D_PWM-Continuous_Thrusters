"""
Unit tests for SimulationLogger module.

Tests the logging functionality for simulation steps.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.satellite_control.core.simulation_context import SimulationContext
from src.satellite_control.core.simulation_logger import SimulationLogger
from src.satellite_control.utils.data_logger import DataLogger


@pytest.fixture
def mock_data_logger():
    """Create a mock DataLogger for testing."""
    logger = MagicMock(spec=DataLogger)
    logger.log_entry = MagicMock()
    return logger


@pytest.fixture
def simulation_logger(mock_data_logger):
    """Create a SimulationLogger instance for testing."""
    return SimulationLogger(mock_data_logger)


@pytest.fixture
def sample_context():
    """Create a sample SimulationContext for testing."""
    from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

    context = SimulationContext()
    context.simulation_time = 1.0
    context.control_dt = 0.1
    context.step_number = 10
    context.mission_phase = "APPROACHING"
    context.waypoint_number = 1
    context.last_control_update_time = 0.9
    context.active_thrusters = [1, 3, 5]
    context.previous_thruster_command = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    context.computation_time_last_step = 0.005

    # Set current state
    context.current_state = np.zeros(13)
    context.current_state[0:3] = [1.0, 2.0, 0.5]
    context.current_state[3:7] = euler_xyz_to_quat_wxyz((0.1, 0.2, 0.3))
    context.current_state[7:10] = [0.1, 0.2, 0.05]
    context.current_state[10:13] = [0.0, 0.0, 0.1]

    # Set target state
    context.target_state = np.zeros(13)
    context.target_state[0:3] = [0.0, 0.0, 0.0]
    context.target_state[3:7] = euler_xyz_to_quat_wxyz((0.0, 0.0, 0.0))
    context.target_state[7:10] = [0.0, 0.0, 0.0]
    context.target_state[10:13] = [0.0, 0.0, 0.0]

    return context


class TestSimulationLoggerInitialization:
    """Test SimulationLogger initialization."""

    def test_logger_creation(self, mock_data_logger):
        """Test that SimulationLogger can be created."""
        logger = SimulationLogger(mock_data_logger)
        assert logger.data_logger == mock_data_logger


class TestSimulationLoggerLogStep:
    """Test the log_step method."""

    def test_log_step_calls_data_logger(
        self, simulation_logger, sample_context, mock_data_logger
    ):
        """Test that log_step calls data_logger.log_entry."""
        thruster_action = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        mpc_info = {"solve_time": 0.01, "status": "optimal"}

        simulation_logger.log_step(
            context=sample_context,
            mpc_start_time=0.99,
            command_sent_time=1.0,
            thruster_action=thruster_action,
            mpc_info=mpc_info,
        )

        # Verify log_entry was called
        mock_data_logger.log_entry.assert_called_once()
        call_args = mock_data_logger.log_entry.call_args[0][0]

        # Verify log entry contains expected fields
        assert "Step" in call_args
        assert "Control_Time" in call_args
        assert "Current_X" in call_args
        assert "Current_Y" in call_args
        assert "Target_X" in call_args
        assert "Target_Y" in call_args

    def test_log_step_with_none_mpc_info(
        self, simulation_logger, sample_context, mock_data_logger
    ):
        """Test that log_step handles None mpc_info."""
        thruster_action = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        simulation_logger.log_step(
            context=sample_context,
            mpc_start_time=0.99,
            command_sent_time=1.0,
            thruster_action=thruster_action,
            mpc_info=None,
        )

        # Should not raise exception
        mock_data_logger.log_entry.assert_called_once()

    def test_log_step_extracts_state_correctly(
        self, simulation_logger, sample_context, mock_data_logger
    ):
        """Test that log_step extracts state components correctly."""
        thruster_action = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        simulation_logger.log_step(
            context=sample_context,
            mpc_start_time=0.99,
            command_sent_time=1.0,
            thruster_action=thruster_action,
            mpc_info={},
        )

        call_args = mock_data_logger.log_entry.call_args[0][0]

        # Verify position extraction
        assert call_args["Current_X"] == pytest.approx(1.0)
        assert call_args["Current_Y"] == pytest.approx(2.0)
        assert call_args["Current_Z"] == pytest.approx(0.5)

        # Verify target position
        assert call_args["Target_X"] == pytest.approx(0.0)
        assert call_args["Target_Y"] == pytest.approx(0.0)
        assert call_args["Target_Z"] == pytest.approx(0.0)

    def test_log_step_calculates_errors(
        self, simulation_logger, sample_context, mock_data_logger
    ):
        """Test that log_step calculates errors correctly."""
        thruster_action = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        simulation_logger.log_step(
            context=sample_context,
            mpc_start_time=0.99,
            command_sent_time=1.0,
            thruster_action=thruster_action,
            mpc_info={},
        )

        call_args = mock_data_logger.log_entry.call_args[0][0]

        # Verify error calculation
        assert "Error_X" in call_args
        assert "Error_Y" in call_args
        assert "Error_Z" in call_args
        assert call_args["Error_X"] == pytest.approx(1.0)  # 1.0 - 0.0
        assert call_args["Error_Y"] == pytest.approx(2.0)  # 2.0 - 0.0


class TestSimulationLoggerLogPhysicsStep:
    """Test the log_physics_step method."""

    def test_log_physics_step(self, simulation_logger, mock_data_logger):
        """Test that log_physics_step logs physics data."""
        current_state = np.zeros(13)
        current_state[0:3] = [1.0, 2.0, 0.5]
        target_state = np.zeros(13)
        thruster_actual = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
        thruster_command = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        simulation_logger.log_physics_step(
            simulation_time=1.0,
            current_state=current_state,
            target_state=target_state,
            thruster_actual_output=thruster_actual,
            thruster_last_command=thruster_command,
        )

        # Verify log_entry was called
        mock_data_logger.log_entry.assert_called_once()
        call_args = mock_data_logger.log_entry.call_args[0][0]

        # Verify physics log contains expected fields
        assert "Time" in call_args
        assert "Current_X" in call_args
        assert "Current_Y" in call_args
        assert "Command_Vector" in call_args

    def test_log_physics_step_with_normalize_angle(
        self, simulation_logger, mock_data_logger
    ):
        """Test that log_physics_step uses normalize_angle function if provided."""
        current_state = np.zeros(13)
        target_state = np.zeros(13)

        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        simulation_logger.log_physics_step(
            simulation_time=1.0,
            current_state=current_state,
            target_state=target_state,
            thruster_actual_output=np.zeros(8),
            thruster_last_command=np.zeros(8),
            normalize_angle_func=normalize_angle,
        )

        # Should not raise exception
        mock_data_logger.log_entry.assert_called_once()
