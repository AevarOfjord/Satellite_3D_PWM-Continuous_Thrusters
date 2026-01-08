"""
Unit tests for PlotGenerator module.

Tests the plotting functionality extracted from unified_visualizer.py.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.satellite_control.visualization.plot_generator import PlotGenerator


@pytest.fixture
def mock_data_accessor():
    """Create a mock data accessor for testing."""
    accessor = MagicMock()
    accessor._col = MagicMock(return_value=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    accessor._row = MagicMock(return_value={"Current_X": 1.0, "Current_Y": 2.0})
    accessor._get_len = MagicMock(return_value=5)
    return accessor


@pytest.fixture
def plot_generator(mock_data_accessor):
    """Create a PlotGenerator instance for testing."""
    return PlotGenerator(
        data_accessor=mock_data_accessor,
        dt=0.1,
        system_title="Test System",
    )


class TestPlotGeneratorInitialization:
    """Test PlotGenerator initialization."""

    def test_plot_generator_creation(self, mock_data_accessor):
        """Test that PlotGenerator can be created."""
        generator = PlotGenerator(
            data_accessor=mock_data_accessor,
            dt=0.1,
            system_title="Test",
        )
        assert generator.data_accessor == mock_data_accessor
        assert generator.dt == 0.1
        assert generator.system_title == "Test"

    def test_data_accessor_methods(self, plot_generator, mock_data_accessor):
        """Test that data accessor methods work correctly."""
        result = plot_generator._col("Current_X")
        mock_data_accessor._col.assert_called_once_with("Current_X")
        assert isinstance(result, np.ndarray)

        result = plot_generator._row(0)
        mock_data_accessor._row.assert_called_once_with(0)
        assert isinstance(result, dict)

        result = plot_generator._get_len()
        mock_data_accessor._get_len.assert_called_once()
        assert result == 5


class TestPlotGeneratorMethods:
    """Test PlotGenerator plotting methods."""

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_position_tracking_plot(
        self, mock_close, mock_savefig, mock_subplots, plot_generator, tmp_path
    ):
        """Test position tracking plot generation."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        plot_dir = tmp_path / "plots"
        plot_generator.generate_position_tracking_plot(plot_dir)

        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_position_error_plot(
        self, mock_close, mock_savefig, mock_subplots, plot_generator, tmp_path
    ):
        """Test position error plot generation."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        plot_dir = tmp_path / "plots"
        plot_generator.generate_position_error_plot(plot_dir)

        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_trajectory_plot(
        self, mock_close, mock_savefig, mock_subplots, plot_generator, tmp_path
    ):
        """Test trajectory plot generation."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plot_dir = tmp_path / "plots"
        plot_generator.generate_trajectory_plot(plot_dir)

        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_all_plots(
        self, mock_close, mock_savefig, mock_subplots, plot_generator, tmp_path
    ):
        """Test that generate_all_plots calls all plot methods."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        plot_dir = tmp_path / "plots"

        with patch.object(
            plot_generator, "generate_position_tracking_plot"
        ) as mock_pos, patch.object(
            plot_generator, "generate_position_error_plot"
        ) as mock_pos_err, patch.object(
            plot_generator, "generate_angular_tracking_plot"
        ) as mock_ang, patch.object(
            plot_generator, "generate_angular_error_plot"
        ) as mock_ang_err, patch.object(
            plot_generator, "generate_trajectory_plot"
        ) as mock_traj, patch.object(
            plot_generator, "generate_trajectory_3d_interactive_plot"
        ) as mock_traj_3d, patch.object(
            plot_generator, "generate_thruster_usage_plot"
        ) as mock_thruster, patch.object(
            plot_generator, "generate_thruster_valve_activity_plot"
        ) as mock_valve, patch.object(
            plot_generator, "generate_pwm_quantization_plot"
        ) as mock_pwm, patch.object(
            plot_generator, "generate_control_effort_plot"
        ) as mock_effort, patch.object(
            plot_generator, "generate_velocity_tracking_plot"
        ) as mock_vel, patch.object(
            plot_generator, "generate_velocity_magnitude_plot"
        ) as mock_vel_mag, patch.object(
            plot_generator, "generate_mpc_performance_plot"
        ) as mock_mpc, patch.object(
            plot_generator, "generate_timing_intervals_plot"
        ) as mock_timing:
            plot_generator.generate_all_plots(plot_dir)

        # Verify all plot methods were called
        mock_pos.assert_called_once_with(plot_dir)
        mock_pos_err.assert_called_once_with(plot_dir)
        mock_ang.assert_called_once_with(plot_dir)
        mock_ang_err.assert_called_once_with(plot_dir)
        mock_traj.assert_called_once_with(plot_dir)
        mock_traj_3d.assert_called_once_with(plot_dir)
        mock_thruster.assert_called_once_with(plot_dir)
        mock_valve.assert_called_once_with(plot_dir)
        mock_pwm.assert_called_once_with(plot_dir)
        mock_effort.assert_called_once_with(plot_dir)
        mock_vel.assert_called_once_with(plot_dir)
        mock_vel_mag.assert_called_once_with(plot_dir)
        mock_mpc.assert_called_once_with(plot_dir)
        mock_timing.assert_called_once_with(plot_dir)

        # Verify directory was created
        assert plot_dir.exists()


class TestPlotGeneratorDataHandling:
    """Test PlotGenerator data handling."""

    def test_handles_empty_data(self, mock_data_accessor):
        """Test that PlotGenerator handles empty data gracefully."""
        mock_data_accessor._get_len.return_value = 0
        mock_data_accessor._col.return_value = np.array([])

        generator = PlotGenerator(
            data_accessor=mock_data_accessor,
            dt=0.1,
        )

        # Should not raise exception
        assert generator._get_len() == 0

    def test_handles_missing_columns(self, mock_data_accessor):
        """Test that PlotGenerator handles missing columns gracefully."""
        def side_effect(name):
            if name == "Current_X":
                return np.array([1.0, 2.0, 3.0])
            else:
                return np.array([0.0, 0.0, 0.0])

        mock_data_accessor._col.side_effect = side_effect

        generator = PlotGenerator(
            data_accessor=mock_data_accessor,
            dt=0.1,
        )

        # Should handle missing columns
        result = generator._col("Current_X")
        assert len(result) == 3
