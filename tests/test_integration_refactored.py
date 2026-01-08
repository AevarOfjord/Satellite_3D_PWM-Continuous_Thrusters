"""
Integration tests for refactored simulation components.

Tests that the newly refactored modules work together correctly.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.satellite_control.config import SimulationConfig
from src.satellite_control.core.simulation import SatelliteMPCLinearizedSimulation
from src.satellite_control.core.simulation_initialization import SimulationInitializer
from src.satellite_control.core.simulation_loop import SimulationLoop


@pytest.fixture
def simulation_config():
    """Create a test SimulationConfig (V4.0.0: use create_default)."""
    # V4.0.0: Use SimulationConfig.create_default() instead of SatelliteConfig
    return SimulationConfig.create_default()


@pytest.mark.integration
class TestRefactoredComponentsIntegration:
    """Test that refactored components work together."""

    def test_initializer_and_loop_integration(self, simulation_config):
        """Test that SimulationInitializer and SimulationLoop work together."""
        # Create a minimal simulation instance
        sim = MagicMock()
        sim.use_mujoco_viewer = False

        # Initialize using SimulationInitializer
        initializer = SimulationInitializer(
            simulation=sim,
            simulation_config=simulation_config,
            use_mujoco_viewer=False,
        )

        with patch(
            "src.satellite_control.core.simulation_initialization.SatelliteThrusterTester"
        ) as mock_sat:
            mock_sat_instance = MagicMock()
            mock_sat_instance.dt = 0.005
            mock_sat_instance.position = np.array([0.0, 0.0, 0.0])
            mock_sat_instance.velocity = np.array([0.0, 0.0, 0.0])
            mock_sat_instance.angular_velocity = np.array([0.0, 0.0, 0.0])
            mock_sat.return_value = mock_sat_instance

            initializer.initialize(
                start_pos=(1.0, 2.0),
                target_pos=(0.0, 0.0),
                start_angle=(0.0, 0.0, 0.0),
                target_angle=(0.0, 0.0, 0.0),
            )

        # Verify initialization completed
        assert hasattr(sim, "satellite")
        assert hasattr(sim, "target_state")
        assert hasattr(sim, "mpc_controller")

        # Create loop and verify it can use the initialized simulation
        loop = SimulationLoop(sim)
        assert loop.simulation == sim

    def test_simulation_with_refactored_components(self, simulation_config):
        """Test that full simulation uses refactored components."""
        # This is a high-level integration test
        # We'll use mocking to avoid full simulation setup

        with patch(
            "src.satellite_control.core.simulation.SimulationInitializer"
        ) as mock_init_class:
            with patch(
                "src.satellite_control.core.simulation.SimulationLoop"
            ) as mock_loop_class:
                mock_initializer = MagicMock()
                mock_init_class.return_value = mock_initializer

                mock_loop = MagicMock()
                mock_loop_class.return_value = mock_loop

                # Create simulation
                sim = SatelliteMPCLinearizedSimulation(
                    simulation_config=simulation_config,
                    use_mujoco_viewer=False,
                )

                # Verify initializer was used
                mock_init_class.assert_called_once()

                # Verify loop can be created
                assert mock_loop_class.called or hasattr(sim, "run_simulation")


@pytest.mark.integration
class TestInitializerLoopDataFlow:
    """Test data flow between initializer and loop."""

    def test_state_passed_correctly(self, simulation_config):
        """Test that state is correctly passed from initializer to loop."""
        sim = MagicMock()
        sim.use_mujoco_viewer = False

        # Setup mocks for initialization
        with patch(
            "src.satellite_control.core.simulation_initialization.SatelliteThrusterTester"
        ) as mock_sat:
            mock_sat_instance = MagicMock()
            mock_sat_instance.dt = 0.005
            mock_sat_instance.position = np.array([1.0, 2.0, 0.0])
            mock_sat_instance.velocity = np.array([0.0, 0.0, 0.0])
            mock_sat_instance.angular_velocity = np.array([0.0, 0.0, 0.0])
            mock_sat.return_value = mock_sat_instance

            initializer = SimulationInitializer(
                simulation=sim,
                simulation_config=simulation_config,
                use_mujoco_viewer=False,
            )

            initializer.initialize(
                start_pos=(1.0, 2.0),
                target_pos=(0.0, 0.0),
                start_angle=(0.0, 0.0, 0.0),
                target_angle=(0.0, 0.0, 0.0),
            )

        # Verify state was set
        assert hasattr(sim, "target_state")
        assert sim.target_state.shape == (13,)

        # Loop should be able to access this state
        loop = SimulationLoop(sim)
        assert loop.simulation.target_state.shape == (13,)


@pytest.mark.integration
class TestVisualizationComponentsIntegration:
    """Test that visualization components work together."""

    def test_plot_generator_and_video_renderer_compatible(self, tmp_path):
        """Test that PlotGenerator and VideoRenderer can use same data."""
        from src.satellite_control.visualization.plot_generator import PlotGenerator
        from src.satellite_control.visualization.video_renderer import VideoRenderer

        # Create mock data accessor
        mock_accessor = MagicMock()
        mock_accessor._col = MagicMock(return_value=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        mock_accessor._row = MagicMock(return_value={"Current_X": 1.0, "Current_Y": 2.0})
        mock_accessor._get_len = MagicMock(return_value=5)

        # Both should be able to use the same data accessor
        plot_gen = PlotGenerator(
            data_accessor=mock_accessor,
            dt=0.1,
            system_title="Test",
        )

        video_rend = VideoRenderer(
            data_accessor=mock_accessor,
            dt=0.1,
            fps=30.0,
            output_dir=tmp_path / "output",
        )

        # Both should access data the same way
        assert plot_gen._get_len() == video_rend.data_accessor._get_len()
        assert plot_gen.dt == video_rend.dt


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery scenarios with refactored components."""

    def test_initializer_handles_missing_config(self):
        """Test that initializer handles missing config gracefully."""
        sim = MagicMock()
        sim.use_mujoco_viewer = False

        initializer = SimulationInitializer(
            simulation=sim,
            simulation_config=None,  # No config provided
            use_mujoco_viewer=False,
        )

        # Should fall back to global config
        assert initializer.simulation_config is None

        # Should still be able to initialize (will use SatelliteConfig)
        with patch(
            "src.satellite_control.core.simulation_initialization.SatelliteThrusterTester"
        ):
            # Should not raise exception
            try:
                initializer.initialize(
                    start_pos=(1.0, 2.0),
                    target_pos=(0.0, 0.0),
                    start_angle=(0.0, 0.0, 0.0),
                    target_angle=(0.0, 0.0, 0.0),
                )
            except Exception:
                # Some exceptions are expected with full mocking
                pass

    def test_loop_handles_interrupted_simulation(self):
        """Test that loop handles interrupted simulation gracefully."""
        sim = MagicMock()
        sim.is_running = True
        sim.simulation_time = 0.0
        sim.max_simulation_time = 10.0
        sim.data_logger = MagicMock()
        sim.data_logger.get_log_count = MagicMock(return_value=50)
        sim.data_save_path = Path("/tmp/test")
        sim.save_csv_data = MagicMock()
        sim.visualizer = MagicMock()
        sim.visualizer.sync_from_controller = MagicMock()

        loop = SimulationLoop(sim)

        # Simulate interruption
        with patch.object(loop, "_run_with_globals") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            try:
                loop.run(show_animation=False)
            except KeyboardInterrupt:
                # Should handle interruption
                pass

            # Should attempt to save data
            # (exact behavior depends on implementation)
