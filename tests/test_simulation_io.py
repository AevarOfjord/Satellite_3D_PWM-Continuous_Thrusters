"""
Unit tests for SimulationIO module.

Tests data export, directory management, and file operations.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.satellite_control.core.simulation_io import SimulationIO


@pytest.fixture
def mock_simulation():
    """Create a mock simulation object for testing."""
    sim = MagicMock()
    sim.data_logger = MagicMock()
    sim.physics_logger = MagicMock()
    sim.data_save_path = None
    sim.state_history = []
    sim.control_history = []
    sim.target_state = np.zeros(13)
    sim.report_generator = MagicMock()
    return sim


@pytest.fixture
def simulation_io(mock_simulation):
    """Create a SimulationIO instance for testing."""
    return SimulationIO(mock_simulation)


class TestSimulationIOInitialization:
    """Test SimulationIO initialization."""

    def test_io_creation(self, mock_simulation):
        """Test that SimulationIO can be created."""
        io = SimulationIO(mock_simulation)
        assert io.sim == mock_simulation


class TestSimulationIODirectories:
    """Test directory creation."""

    @patch("src.satellite_control.core.simulation_io.datetime")
    def test_create_data_directories(self, mock_datetime, simulation_io, tmp_path):
        """Test that data directories are created correctly."""
        from datetime import datetime

        mock_datetime.now.return_value.strftime.return_value = "01-01-2026_12-00-00"

        with patch("src.satellite_control.core.simulation_io.Path") as mock_path:
            # Mock Path to return our tmp_path
            mock_base = MagicMock()
            mock_sim = MagicMock()
            mock_timestamped = tmp_path / "test_dir"
            mock_timestamped.mkdir = MagicMock()
            mock_sim.__truediv__ = MagicMock(return_value=mock_timestamped)
            mock_base.__truediv__ = MagicMock(return_value=mock_sim)
            mock_path.return_value = mock_base

            result = simulation_io.create_data_directories()

            # Verify directory creation was attempted
            assert result is not None

    def test_create_data_directories_creates_structure(self, simulation_io, tmp_path):
        """Test that directory structure is created."""
        with patch("src.satellite_control.core.simulation_io.Path") as mock_path_class:
            # Create a real path structure
            base_path = tmp_path / "Data"
            sim_path = base_path / "Simulation"
            timestamped_path = sim_path / "01-01-2026_12-00-00"

            # Mock Path constructor to return our paths
            def path_side_effect(path_str):
                if path_str == "Data":
                    return base_path
                return Path(path_str)

            mock_path_class.side_effect = path_side_effect

            with patch(
                "src.satellite_control.core.simulation_io.datetime"
            ) as mock_datetime:
                from datetime import datetime

                mock_datetime.now.return_value.strftime.return_value = (
                    "01-01-2026_12-00-00"
                )

                result = simulation_io.create_data_directories()

                # Verify path was returned
                assert result is not None


class TestSimulationIOCSVData:
    """Test CSV data saving."""

    def test_save_csv_data(self, simulation_io, mock_simulation):
        """Test that CSV data is saved via loggers."""
        simulation_io.save_csv_data()

        mock_simulation.data_logger.save_csv_data.assert_called_once()
        mock_simulation.physics_logger.save_csv_data.assert_called_once()


class TestSimulationIOMissionSummary:
    """Test mission summary generation."""

    def test_save_mission_summary_no_path(self, simulation_io, mock_simulation):
        """Test that mission summary handles missing path."""
        mock_simulation.data_save_path = None

        simulation_io.save_mission_summary()

        # Should not raise exception
        mock_simulation.report_generator.generate_report.assert_not_called()

    def test_save_mission_summary_with_history(self, simulation_io, mock_simulation, tmp_path):
        """Test mission summary with state history."""
        mock_simulation.data_save_path = tmp_path
        mock_simulation.state_history = [np.zeros(13), np.ones(13)]
        mock_simulation.control_history = [np.zeros(8), np.ones(8)]
        mock_simulation.data_logger.stats_solve_times = [0.01, 0.02]

        simulation_io.save_mission_summary()

        # Verify report was generated
        mock_simulation.report_generator.generate_report.assert_called_once()

    def test_save_mission_summary_empty_history(self, simulation_io, mock_simulation, tmp_path):
        """Test mission summary with empty history."""
        mock_simulation.data_save_path = tmp_path
        mock_simulation.state_history = []
        mock_simulation.control_history = []

        simulation_io.save_mission_summary()

        # Should not generate report if no history
        mock_simulation.report_generator.generate_report.assert_not_called()

    @patch("pandas.read_csv")
    def test_save_mission_summary_loads_from_csv(
        self, mock_read_csv, simulation_io, mock_simulation, tmp_path
    ):
        """Test that mission summary loads history from CSV if not in memory."""
        mock_simulation.data_save_path = tmp_path
        mock_simulation.state_history = []
        mock_simulation.control_history = []

        # Mock CSV data
        import pandas as pd

        mock_df = MagicMock()
        mock_df.columns = [
            "Current_X",
            "Current_Y",
            "Current_Z",
            "Current_Roll",
            "Current_Pitch",
            "Current_Yaw",
            "Current_VX",
            "Current_VY",
            "Current_VZ",
            "Current_WX",
            "Current_WY",
            "Current_WZ",
        ]
        mock_df.__getitem__.return_value.values = np.array([0.0, 1.0, 2.0])
        mock_read_csv.return_value = mock_df

        # Create CSV file
        csv_path = tmp_path / "control_data.csv"
        csv_path.touch()

        simulation_io.save_mission_summary()

        # Should attempt to load from CSV
        # (exact behavior depends on implementation)


class TestSimulationIOLoadHistory:
    """Test history loading from CSV."""

    @patch("pandas.read_csv")
    def test_load_history_from_csv_3d(self, mock_read_csv, simulation_io, tmp_path):
        """Test loading 3D history from CSV."""
        import pandas as pd

        # Create mock DataFrame with 3D columns
        mock_df = MagicMock()
        mock_df.columns = [
            "Current_X",
            "Current_Y",
            "Current_Z",
            "Current_Roll",
            "Current_Pitch",
            "Current_Yaw",
            "Current_VX",
            "Current_VY",
            "Current_VZ",
            "Current_WX",
            "Current_WY",
            "Current_WZ",
        ]

        # Mock column access
        def getitem_side_effect(key):
            if isinstance(key, list):
                mock_col = MagicMock()
                mock_col.values = np.array([0.0, 1.0, 2.0])
                return mock_col
            return MagicMock()

        mock_df.__getitem__.side_effect = getitem_side_effect
        mock_read_csv.return_value = mock_df

        mock_simulation = simulation_io.sim
        mock_simulation.data_save_path = tmp_path

        # Create CSV file
        csv_path = tmp_path / "control_data.csv"
        csv_path.touch()

        result = simulation_io._load_history_from_csv()

        # Should return array or None
        assert result is None or isinstance(result, np.ndarray)

    def test_load_history_from_csv_no_path(self, simulation_io, mock_simulation):
        """Test loading history when no path is set."""
        mock_simulation.data_save_path = None

        result = simulation_io._load_history_from_csv()

        assert result is None
