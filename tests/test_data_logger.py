"""
Unit tests for data_logger.py module.

Tests the DataLogger class which provides centralized logging
for simulation runs.
"""

import csv
import tempfile
from pathlib import Path

import pytest

from src.satellite_control.utils.data_logger import DataLogger, create_data_logger


class TestDataLoggerInitialization:
    """Test DataLogger initialization."""

    def test_default_initialization_simulation(self):
        """Test DataLogger initialization with simulation mode."""
        logger = DataLogger(mode="simulation")

        assert logger.mode == "simulation"
        assert logger.detailed_log_data == []
        assert logger.data_save_path is None
        assert logger.current_step == 0

    def test_initialization_case_insensitive(self):
        """Test that mode is case-insensitive."""
        logger1 = DataLogger(mode="SIMULATION")

        assert logger1.mode == "simulation"

    def test_initialization_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Mode must be 'simulation' or 'physics'"):
            DataLogger(mode="invalid")


class TestFactoryFunction:
    """Test the create_data_logger factory function."""

    def test_create_simulation_logger(self):
        """Test creating simulation logger via factory."""
        logger = create_data_logger(mode="simulation")

        assert isinstance(logger, DataLogger)
        assert logger.mode == "simulation"


class TestSavePathManagement:
    """Test save path setting and management."""

    def test_set_save_path(self):
        """Test setting the save path."""
        logger = DataLogger()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_data"
            logger.set_save_path(save_path)

            assert logger.data_save_path == save_path
            assert save_path.exists()  # Should create directory if it doesn't exist

    def test_set_save_path_creates_directory(self):
        """Test that set_save_path creates non-existent directories."""
        logger = DataLogger()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "path" / "test_data"
            logger.set_save_path(save_path)

            assert save_path.exists()
            assert save_path.is_dir()


class TestLogEntry:
    """Test logging individual entries."""

    def test_log_single_entry(self):
        """Test logging a single entry."""
        logger = DataLogger()

        entry = {"Step": 1, "Current_X": 0.5, "Current_Y": 0.3, "MPC_Solve_Time": 0.025}

        logger.log_entry(entry)

        assert logger.get_log_count() == 1
        assert logger.current_step == 1
        assert logger.detailed_log_data[0] == entry

    def test_log_multiple_entries(self):
        """Test logging multiple entries."""
        logger = DataLogger()

        entries = [
            {"Step": 1, "Current_X": 0.0},
            {"Step": 2, "Current_X": 0.1},
            {"Step": 3, "Current_X": 0.2},
        ]

        for entry in entries:
            logger.log_entry(entry)

        assert logger.get_log_count() == 3
        assert logger.current_step == 3

    def test_log_entry_increments_step(self):
        """Test that logging increments the current step counter."""
        logger = DataLogger()

        for i in range(10):
            logger.log_entry({"Step": i})

        assert logger.current_step == 10


class TestGetLogCount:
    """Test getting the count of logged entries."""

    def test_get_log_count_empty(self):
        """Test log count when no entries logged."""
        logger = DataLogger()
        assert logger.get_log_count() == 0

    def test_get_log_count_with_entries(self):
        """Test log count after logging entries."""
        logger = DataLogger()

        for i in range(5):
            logger.log_entry({"Step": i})

        assert logger.get_log_count() == 5


class TestClearLogs:
    """Test clearing logged data."""

    def test_clear_logs(self):
        """Test that clear_logs removes all entries."""
        logger = DataLogger()

        # Add some entries
        for i in range(5):
            logger.log_entry({"Step": i})

        assert logger.get_log_count() == 5

        # Clear logs
        logger.clear_logs()

        assert logger.get_log_count() == 0
        assert logger.current_step == 0
        assert logger.detailed_log_data == []


class TestCSVExport:
    """Test CSV file export functionality."""

    def test_save_csv_data_simulation_mode(self):
        """Test saving CSV data in simulation mode."""
        logger = DataLogger(mode="simulation")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            logger.set_save_path(save_path)

            # Add test data
            logger.log_entry(
                {
                    "Step": 1,
                    "MPC_Start_Time": 0.0,
                    "Control_Time": 0.02,
                    "Current_X": 0.5,
                    "Current_Y": 0.3,
                }
            )

            # Save CSV
            success = logger.save_csv_data()

            assert success is True

            # Verify file exists
            csv_file = save_path / "control_data.csv"
            assert csv_file.exists()

            # Verify CSV content
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0]["Step"] == "1"

    def test_save_csv_data_no_path(self):
        """Test that save fails if no save path is set."""
        logger = DataLogger()
        logger.log_entry({"Step": 1})

        success = logger.save_csv_data()
        assert success is False

    def test_save_csv_data_no_entries(self):
        """Test that save fails if no data is logged."""
        logger = DataLogger()

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.set_save_path(Path(tmpdir))
            success = logger.save_csv_data()
            assert success is False

    def test_save_csv_with_missing_fields(self):
        """Test CSV save handles missing fields gracefully."""
        logger = DataLogger(mode="simulation")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            logger.set_save_path(save_path)

            # Entry with only some fields
            logger.log_entry(
                {
                    "Step": 1,
                    "Current_X": 0.5,
                    # Missing many other fields
                }
            )

            success = logger.save_csv_data()
            assert success is True

            # Verify file was created
            csv_file = save_path / "control_data.csv"
            assert csv_file.exists()


class TestSummaryStatistics:
    """Test summary statistics calculation."""

    def test_get_summary_stats_empty(self):
        """Test summary stats when no data is logged."""
        logger = DataLogger()
        stats = logger.get_summary_stats()

        assert stats == {}

    def test_get_summary_stats_basic(self):
        """Test basic summary statistics."""
        logger = DataLogger(mode="simulation")

        # Add some entries
        for i in range(5):
            logger.log_entry(
                {
                    "Step": i,
                    "MPC_Solve_Time": 0.020 + i * 0.005,
                }
            )

        stats = logger.get_summary_stats()

        assert stats["total_steps"] == 5
        assert stats["mode"] == "simulation"
        assert "avg_solve_time" in stats
        assert "max_solve_time" in stats
        assert "min_solve_time" in stats

    def test_get_summary_stats_solve_times(self):
        """Test MPC solve time statistics calculation."""
        logger = DataLogger()

        solve_times = [0.010, 0.020, 0.030, 0.015, 0.025]
        for i, solve_time in enumerate(solve_times):
            logger.log_entry(
                {
                    "Step": i,
                    "MPC_Solve_Time": solve_time,
                }
            )

        stats = logger.get_summary_stats()

        assert stats["min_solve_time"] == 0.010
        assert stats["max_solve_time"] == 0.030
        assert abs(stats["avg_solve_time"] - 0.020) < 1e-9

    def test_get_summary_stats_timing_violations(self):
        """Test timing violation counting."""
        logger = DataLogger()

        # Add entries with violations
        logger.log_entry({"Step": 1, "Timing_Violation": "YES"})
        logger.log_entry({"Step": 2, "Timing_Violation": "NO"})
        logger.log_entry({"Step": 3, "Timing_Violation": "YES"})

        stats = logger.get_summary_stats()

        assert stats["timing_violations"] == 2

    def test_get_summary_stats_time_limit_exceeded(self):
        """Test time limit exceeded counting."""
        logger = DataLogger()

        logger.log_entry({"Step": 1, "MPC_Time_Limit_Exceeded": "YES"})
        logger.log_entry({"Step": 2, "MPC_Time_Limit_Exceeded": "NO"})
        logger.log_entry({"Step": 3, "MPC_Time_Limit_Exceeded": "YES"})
        logger.log_entry({"Step": 4, "MPC_Time_Limit_Exceeded": "YES"})

        stats = logger.get_summary_stats()

        assert stats["time_limit_exceeded"] == 3

    def test_get_summary_stats_handles_invalid_solve_times(self):
        """Test that invalid solve times are handled gracefully."""
        logger = DataLogger()

        logger.log_entry({"Step": 1, "MPC_Solve_Time": 0.020})
        logger.log_entry({"Step": 2, "MPC_Solve_Time": "invalid"})
        logger.log_entry({"Step": 3, "MPC_Solve_Time": ""})
        logger.log_entry({"Step": 4, "MPC_Solve_Time": 0.030})

        stats = logger.get_summary_stats()

        # Should only count valid solve times
        assert stats["total_steps"] == 4
        assert "avg_solve_time" in stats
        # Average should be (0.020 + 0.030) / 2 = 0.025
        assert abs(stats["avg_solve_time"] - 0.025) < 1e-9


class TestPrintSummary:
    """Test print summary functionality."""

    def test_print_summary_no_data(self, capsys):
        """Test print summary with no logged data."""
        logger = DataLogger()
        logger.print_summary()

        captured = capsys.readouterr()
        assert "No data logged" in captured.out

    def test_print_summary_with_data(self, capsys):
        """Test print summary with logged data."""
        logger = DataLogger(mode="simulation")

        for i in range(5):
            logger.log_entry(
                {
                    "Step": i,
                    "MPC_Solve_Time": 0.020,
                }
            )

        logger.print_summary()

        captured = capsys.readouterr()
        assert "DATA LOGGER SUMMARY" in captured.out
        assert "SIMULATION MODE" in captured.out
        assert "Total steps logged: 5" in captured.out


class TestHeaders:
    """Test CSV header generation."""

    def test_simulation_headers(self):
        """Test that simulation mode has correct headers."""
        logger = DataLogger(mode="simulation")
        headers = logger._get_simulation_headers()

        assert "Step" in headers
        assert "Control_Time" in headers
        assert "Current_X" in headers
        assert "MPC_Solve_Time" in headers


class TestIntegration:
    """Integration tests for complete logging workflow."""

    def test_complete_logging_workflow(self):
        """Test a complete logging workflow from start to finish."""
        logger = create_data_logger(mode="simulation")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set save path
            save_path = Path(tmpdir)
            logger.set_save_path(save_path)

            # Log multiple entries
            for i in range(10):
                logger.log_entry(
                    {
                        "Step": i,
                        "MPC_Start_Time": i * 0.02,
                        "Current_X": i * 0.1,
                        "Current_Y": i * 0.05,
                        "MPC_Solve_Time": 0.015 + i * 0.001,
                        "Timing_Violation": "YES" if i % 3 == 0 else "NO",
                    }
                )

            # Verify log count
            assert logger.get_log_count() == 10

            # Get summary stats
            stats = logger.get_summary_stats()
            assert stats["total_steps"] == 10
            assert "avg_solve_time" in stats

            # Save to CSV
            success = logger.save_csv_data()
            assert success is True

            # Verify file exists and contains correct data
            csv_file = save_path / "control_data.csv"
            assert csv_file.exists()

            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 10

            # Clear logs
            logger.clear_logs()
            assert logger.get_log_count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
