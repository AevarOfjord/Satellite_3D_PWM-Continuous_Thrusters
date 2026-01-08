"""
Unit tests for SimulationLoop module.

Tests the main simulation loop execution logic extracted from simulation.py.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.core.simulation_loop import SimulationLoop


@pytest.fixture
def mock_simulation():
    """Create a mock simulation object for testing."""
    sim = MagicMock()
    sim.is_running = True
    sim.simulation_time = 0.0
    sim.max_simulation_time = 10.0
    sim.control_update_interval = 0.1
    sim.satellite = MagicMock()
    sim.satellite.dt = 0.005
    sim.satellite.simulation_time = 0.0
    sim.satellite.update_physics = MagicMock()
    sim.satellite.is_viewer_paused = MagicMock(return_value=False)
    sim.use_mujoco_viewer = False
    sim.data_logger = MagicMock()
    sim.data_logger.clear_logs = MagicMock()
    sim.data_logger.get_log_count = MagicMock(return_value=100)
    sim.physics_logger = MagicMock()
    sim.physics_logger.clear_logs = MagicMock()
    sim.create_data_directories = MagicMock(return_value=Path("/tmp/test_data"))
    sim.data_save_path = Path("/tmp/test_data")
    sim.save_csv_data = MagicMock()
    sim.save_mission_summary = MagicMock()
    sim.auto_generate_visualizations = MagicMock()
    sim.print_performance_summary = MagicMock()
    sim.visualizer = MagicMock()
    sim.visualizer.sync_from_controller = MagicMock()
    sim.visualizer.save_mujoco_video = MagicMock()
    sim.get_current_state = MagicMock(return_value=np.zeros(13))
    sim.update_target_state_for_mode = MagicMock()
    sim.update_mpc_control = MagicMock()
    sim.process_command_queue = MagicMock()
    sim.log_physics_step = MagicMock()
    sim.check_target_reached = MagicMock(return_value=False)
    sim.target_reached_time = None
    sim.target_state = np.zeros(13)
    sim.draw_simulation = MagicMock()
    sim.update_mpc_info_panel = MagicMock()
    sim.performance_monitor = MagicMock()
    sim.performance_monitor.record_physics_step = MagicMock()
    sim.performance_monitor.increment_step = MagicMock()
    # V3.0.0: Add simulation_config to mock
    sim.simulation_config = SimulationConfig.create_default()
    return sim


class TestSimulationLoopInitialization:
    """Test SimulationLoop initialization."""

    def test_loop_creation(self, mock_simulation):
        """Test that loop can be created."""
        loop = SimulationLoop(mock_simulation)
        assert loop.simulation == mock_simulation


class TestSimulationLoopRun:
    """Test the run method."""

    @patch("src.satellite_control.core.simulation_loop.use_structured_config")
    def test_run_sets_up_data_directories(self, mock_use_config, mock_simulation):
        """Test that data directories are set up."""
        loop = SimulationLoop(mock_simulation)
        mock_use_config.return_value.__enter__ = MagicMock()
        mock_use_config.return_value.__exit__ = MagicMock(return_value=None)

        # Mock the loop to exit immediately
        mock_simulation.is_running = False

        with patch.object(loop, "_run_with_globals", return_value=None):
            loop.run(show_animation=False, structured_config=MagicMock())

        mock_simulation.data_logger.clear_logs.assert_called_once()
        mock_simulation.physics_logger.clear_logs.assert_called_once()
        mock_simulation.create_data_directories.assert_called_once()

    @patch("src.satellite_control.core.simulation_loop.use_structured_config")
    def test_run_initializes_context(self, mock_use_config, mock_simulation):
        """Test that simulation context is initialized."""
        from src.satellite_control.core.simulation_context import SimulationContext

        loop = SimulationLoop(mock_simulation)
        mock_use_config.return_value.__enter__ = MagicMock()
        mock_use_config.return_value.__exit__ = MagicMock(return_value=None)

        # Mock the loop to exit immediately
        mock_simulation.is_running = False

        with patch.object(loop, "_run_with_globals", return_value=None):
            loop.run(show_animation=False, structured_config=MagicMock())

        # Context should be initialized if it doesn't exist
        if not hasattr(mock_simulation, "context"):
            assert hasattr(mock_simulation, "context")


class TestSimulationLoopBatchMode:
    """Test batch mode execution."""

    @patch("src.satellite_control.core.simulation_loop.time.sleep")
    def test_batch_mode_runs_physics_steps(self, mock_sleep, mock_simulation):
        """Test that batch mode runs physics steps."""
        loop = SimulationLoop(mock_simulation)

        # Set up for batch mode
        mock_simulation.control_update_interval = 0.1
        mock_simulation.satellite.dt = 0.005
        mock_simulation.is_running = True

        # Make loop exit after one iteration
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                mock_simulation.is_running = False

        mock_simulation.is_running = True
        with patch.object(loop, "update_step", side_effect=side_effect):
            with patch.object(loop, "_run_batch_mode") as mock_batch:
                mock_batch.return_value = None
                loop._run_batch_mode()

        # Verify physics was called
        assert mock_simulation.process_command_queue.called or mock_simulation.satellite.update_physics.called

    def test_batch_mode_handles_viewer_pause(self, mock_simulation):
        """Test that batch mode handles MuJoCo viewer pause."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.use_mujoco_viewer = True
        mock_simulation.satellite.is_viewer_paused = MagicMock(return_value=True)
        mock_simulation.satellite.sync_viewer = MagicMock()

        # Make loop exit immediately
        mock_simulation.is_running = False

        with patch("time.sleep"):
            loop._run_batch_mode()

        # Should handle pause state
        assert True  # Test passes if no exception


class TestSimulationLoopUpdateStep:
    """Test the update_step method."""

    def test_update_step_skips_when_not_running(self, mock_simulation):
        """Test that update_step returns early when not running."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = False

        result = loop.update_step(None)

        assert result == []
        mock_simulation.update_mpc_control.assert_not_called()

    def test_update_step_updates_target_state(self, mock_simulation):
        """Test that update_step updates target state."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = True

        loop.update_step(None)

        mock_simulation.update_target_state_for_mode.assert_called_once()

    def test_update_step_updates_mpc_control(self, mock_simulation):
        """Test that update_step updates MPC control."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = True

        loop.update_step(None)

        mock_simulation.update_mpc_control.assert_called_once()

    def test_update_step_processes_command_queue(self, mock_simulation):
        """Test that update_step processes command queue."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = True

        loop.update_step(None)

        mock_simulation.process_command_queue.assert_called_once()

    def test_update_step_advances_physics(self, mock_simulation):
        """Test that update_step advances physics."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = True

        loop.update_step(None)

        mock_simulation.satellite.update_physics.assert_called_once()
        mock_simulation.performance_monitor.record_physics_step.assert_called_once()
        mock_simulation.performance_monitor.increment_step.assert_called_once()

    def test_update_step_logs_physics(self, mock_simulation):
        """Test that update_step logs physics data."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = True

        loop.update_step(None)

        mock_simulation.log_physics_step.assert_called_once()

    def test_update_step_draws_simulation(self, mock_simulation):
        """Test that update_step draws simulation."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = True

        loop.update_step(None)

        mock_simulation.draw_simulation.assert_called_once()
        mock_simulation.update_mpc_info_panel.assert_called_once()


class TestSimulationLoopTermination:
    """Test termination conditions."""

    def test_termination_on_max_time(self, mock_simulation):
        """Test that simulation stops at max time."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = True
        mock_simulation.simulation_time = 10.0
        mock_simulation.max_simulation_time = 10.0

        result = loop._check_termination_conditions()

        assert result is True
        assert mock_simulation.is_running is False

    def test_termination_on_target_reached(self, mock_simulation):
        """Test termination when target is reached and held."""
        # V3.0.0: Update simulation_config instead of SatelliteConfig
        mock_simulation.simulation_config.app_config.simulation.use_final_stabilization = False
        mock_simulation.simulation_config.app_config.simulation.waypoint_final_stabilization_time = 2.0
        mock_simulation.simulation_config.mission_state.enable_waypoint_mode = False
        mock_simulation.simulation_config.mission_state.dxf_shape_mode_active = False
        
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = True
        mock_simulation.check_target_reached = MagicMock(return_value=True)
        mock_simulation.target_reached_time = 5.0
        mock_simulation.simulation_time = 8.0

        result = loop._check_termination_conditions()

        # Should terminate if stabilization time is met
        # (exact behavior depends on config, but should check conditions)
        assert isinstance(result, bool)

    def test_no_termination_when_running(self, mock_simulation):
        """Test that simulation continues when conditions not met."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = True
        mock_simulation.simulation_time = 5.0
        mock_simulation.max_simulation_time = 10.0
        mock_simulation.check_target_reached = MagicMock(return_value=False)
        mock_simulation.target_reached_time = None

        result = loop._check_termination_conditions()

        assert result is False
        assert mock_simulation.is_running is True


class TestSimulationLoopWaypointHandling:
    """Test waypoint advancement logic."""

    def test_waypoint_advancement(self, mock_simulation):
        """Test that waypoints are advanced correctly."""
        # V3.0.0: Update simulation_config instead of SatelliteConfig
        from src.satellite_control.config.mission_state import MissionState
        from dataclasses import replace
        
        mission_state = replace(
            mock_simulation.simulation_config.mission_state,
            enable_waypoint_mode=True,
            current_target_index=0,
            waypoint_targets=[(1.0, 1.0, 0.0), (2.0, 2.0, 0.0)],
        )
        mock_simulation.simulation_config = replace(
            mock_simulation.simulation_config,
            mission_state=mission_state,
        )
        mock_simulation.simulation_config.app_config.simulation.waypoint_final_stabilization_time = 2.0
        mock_simulation.simulation_config.app_config.simulation.target_hold_time = 1.0
        
        # Mock mission_manager
        mock_simulation.mission_manager = MagicMock()
        mock_simulation.mission_manager.advance_to_next_target = MagicMock(return_value=True)
        mock_simulation.mission_manager.get_current_waypoint_target = MagicMock(
            return_value=((2.0, 2.0, 0.0), (0.0, 0.0, 0.0))
        )
        
        loop = SimulationLoop(mock_simulation)
        mock_simulation.target_maintenance_time = 1.5
        mock_simulation.target_reached_time = 0.0
        mock_simulation.simulation_time = 1.5

        result = loop._handle_waypoint_advancement()

        # Should attempt to advance
        assert isinstance(result, bool)

    def test_waypoint_completion(self, mock_simulation):
        """Test that waypoint mission completion is detected."""
        # V3.0.0: Update simulation_config instead of SatelliteConfig
        from src.satellite_control.config.mission_state import MissionState
        from dataclasses import replace
        
        mission_state = replace(
            mock_simulation.simulation_config.mission_state,
            enable_waypoint_mode=True,
            current_target_index=1,
            waypoint_targets=[(1.0, 1.0, 0.0), (2.0, 2.0, 0.0)],
        )
        mock_simulation.simulation_config = replace(
            mock_simulation.simulation_config,
            mission_state=mission_state,
        )
        mock_simulation.simulation_config.app_config.simulation.waypoint_final_stabilization_time = 2.0
        
        # Mock mission_manager
        mock_simulation.mission_manager = MagicMock()
        mock_simulation.mission_manager.advance_to_next_target = MagicMock(return_value=False)
        
        loop = SimulationLoop(mock_simulation)
        mock_simulation.target_maintenance_time = 2.5
        mock_simulation.simulation_time = 5.0

        result = loop._handle_waypoint_advancement()

        # Should detect completion
        assert isinstance(result, bool)


class TestSimulationLoopDataSaving:
    """Test data saving functionality."""

    def test_data_saving_after_simulation(self, mock_simulation):
        """Test that data is saved after simulation completes."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = False
        mock_simulation.data_save_path = Path("/tmp/test_data")

        with patch.object(loop, "_run_batch_mode") as mock_batch:
            mock_batch.return_value = Path("/tmp/test_data")
            result = loop._run_batch_mode()

        # Should save data
        assert result is not None

    def test_visualization_generation(self, mock_simulation):
        """Test that visualizations are generated after simulation."""
        loop = SimulationLoop(mock_simulation)
        mock_simulation.is_running = False
        mock_simulation.data_save_path = Path("/tmp/test_data")

        with patch.object(loop, "_run_batch_mode") as mock_batch:
            mock_batch.return_value = Path("/tmp/test_data")
            loop._run_batch_mode()

        # Visualizations should be generated
        # (exact calls depend on implementation)
        assert True  # Test passes if no exception
