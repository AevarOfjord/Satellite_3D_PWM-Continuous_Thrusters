"""
Data Logger for Satellite Control System

Centralized data logging and CSV export for simulation runs.
Handles step-by-step data collection and exports to standardized CSV format.

Key features:
- Simulation mode logging
- Detailed log data with full state history per timestep
- Terminal message logging for debugging and analysis
- Automatic CSV export with standardized headers
- Configurable save paths with automatic directory creation
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


class DataLogger:
    """
    Centralized data logging and CSV export.

    Handles logging of simulation data and exports to CSV format
    for analysis. Used by SatelliteMPCLinearizedSimulation.
    """

    def __init__(
        self,
        mode: str = "simulation",
        buffer_size: int = 1,
        filename: str = "control_data.csv",
    ):
        """
        Initialize data logger.

        Args:
            mode: "simulation" or "physics"
            buffer_size: Number of entries to keep in memory before flushing to disk
            filename: Name of the CSV file to write
        """
        self.mode = mode.lower()
        if self.mode not in ["simulation", "physics"]:
            raise ValueError("Mode must be 'simulation' or 'physics'")
        self.filename = filename
        self.buffer_size = buffer_size

        # Internal state
        self.detailed_log_data: List[Dict[str, Any]] = []
        self.terminal_log_data: List[Dict[str, Any]] = []
        self.data_save_path: Optional[Path] = None
        self.current_step = 0
        self._headers_written = False

        # Incremental stats tracking
        self.stats_solve_times: List[float] = []
        self.stats_timing_violations = 0
        self.stats_time_limit_exceeded = 0
        self.total_steps_recorded = 0

    def set_save_path(self, path: Path) -> None:
        """
        Set the directory path where data will be saved.

        Args:
            path: Directory path for saving data files
        """
        self.data_save_path = path
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        # Reset headers flag when path changes
        self._headers_written = False

    def log_entry(self, entry: Dict[str, Any]) -> None:
        """
        Add a log entry to the detailed log data.
        Flushes to disk if buffer size is exceeded.

        Args:
            entry: Dictionary containing log data for one timestep
        """
        self.detailed_log_data.append(entry)
        self.current_step += 1
        self.total_steps_recorded += 1

        # Update incremental stats
        if "MPC_Solve_Time" in entry and entry["MPC_Solve_Time"] != "":
            try:
                self.stats_solve_times.append(float(entry["MPC_Solve_Time"]))
            except (ValueError, TypeError):
                pass

        if entry.get("Timing_Violation") == "YES":
            self.stats_timing_violations += 1

        if entry.get("MPC_Time_Limit_Exceeded") == "YES":
            self.stats_time_limit_exceeded += 1

        if len(self.detailed_log_data) >= self.buffer_size:
            self.flush()

    # ...
    def flush(self) -> bool:
        """
        Flush current buffer to disk.
        """
        if not self.data_save_path or not self.detailed_log_data:
            return False

        csv_file_path = self.data_save_path / self.filename  # Use custom filename
        headers = self._get_simulation_headers()
        if self.mode == "physics":
            headers = self._get_physics_headers()
        else:
            headers = self._get_simulation_headers()

        mode = "a" if self._headers_written else "w"

        try:
            with open(csv_file_path, mode, newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not self._headers_written:
                    writer.writerow(headers)
                    self._headers_written = True

                for log_entry in self.detailed_log_data:
                    row = [
                        self._format_value(header, log_entry.get(header, "")) for header in headers
                    ]
                    writer.writerow(row)

            # Clear buffer after successful write
            self.detailed_log_data = []
            return True
        except Exception as e:
            print(f" Error flushing CSV data: {e}")
            return False

    def log_terminal_message(self, message_data: Dict[str, Any]) -> None:
        """
        Add a terminal output message to the terminal log.

        Args:
            message_data: Dictionary containing terminal message data
                Expected keys: time, status, stabilization_time (optional),
                pos_error, ang_error, thrusters, solve_time, next_update (optional)
        """
        self.terminal_log_data.append(message_data)

    def save_csv_data(self) -> bool:
        """
        Final export of any remaining data to CSV file.
        Also saves terminal log.

        Returns:
            True if save successful, False otherwise
        """
        if not self.data_save_path:
            return False

        # Flush any remaining data
        if self.detailed_log_data:
            success = self.flush()
            wrote_any_files = success
        else:
            success = True
            # Or True? If we assume "save completed" (nothing to save) is
            # success.
            wrote_any_files = False
            # If we want success = "data on disk", return True.
            # wrote_any_files implies we created a file.
            # But wrote_any_files implies we created a file.
            # If buffer was empty and no file existed, we didn't write.
            # But headers are written on first flush.
            # If headers were written previously, file exists.
            wrote_any_files = self._headers_written

        # Save terminal log (always overwrite/new file for terminal log as it's
        # small)
        if self.terminal_log_data:
            terminal_log_path = self.data_save_path / f"{self.mode}_terminal_log.csv"
            try:
                with open(terminal_log_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self._get_terminal_log_headers())

                    for log_entry in self.terminal_log_data:
                        row = [
                            self._format_terminal_value(header, log_entry.get(header, ""))
                            for header in self._get_terminal_log_headers()
                        ]
                        writer.writerow(row)

                print(f" Terminal log saved to: {terminal_log_path}")
                wrote_any_files = True
            except Exception as e:
                print(f" Error saving terminal log: {e}")
                success = False

        return success and wrote_any_files

    def _get_physics_headers(self) -> List[str]:
        """Get CSV headers for physics mode (high-frequency valve tracking)."""
        return [
            "Time",
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
            "Target_X",
            "Target_Y",
            "Target_Z",
            "Target_Roll",
            "Target_Pitch",
            "Target_Yaw",
            "Error_X",
            "Error_Y",
            "Error_Z",
            "Error_Roll",
            "Error_Pitch",
            "Error_Yaw",
            "Command_Vector",
            "Thruster_1_Cmd",
            "Thruster_1_Val",
            "Thruster_2_Cmd",
            "Thruster_2_Val",
            "Thruster_3_Cmd",
            "Thruster_3_Val",
            "Thruster_4_Cmd",
            "Thruster_4_Val",
            "Thruster_5_Cmd",
            "Thruster_5_Val",
            "Thruster_6_Cmd",
            "Thruster_6_Val",
            "Thruster_7_Cmd",
            "Thruster_7_Val",
            "Thruster_8_Cmd",
            "Thruster_8_Val",
            "Thruster_9_Cmd",
            "Thruster_9_Val",
            "Thruster_10_Cmd",
            "Thruster_10_Val",
            "Thruster_11_Cmd",
            "Thruster_11_Val",
            "Thruster_12_Cmd",
            "Thruster_12_Val",
        ]

    def _get_simulation_headers(self) -> List[str]:
        """Get CSV headers for simulation mode (historical format)."""
        return [
            "Step",
            "MPC_Start_Time",
            "Control_Time",
            "Actual_Time_Interval",
            "CONTROL_DT",
            "Mission_Phase",
            "Waypoint_Number",
            "Telemetry_X_mm",
            "Telemetry_Y_mm",
            "Telemetry_Z_mm",
            "Telemetry_Roll_deg",
            "Telemetry_Pitch_deg",
            "Telemetry_Yaw_deg",
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
            "Target_X",
            "Target_Y",
            "Target_Z",
            "Target_Roll",
            "Target_Pitch",
            "Target_Yaw",
            "Target_VX",
            "Target_VY",
            "Target_VZ",
            "Target_WX",
            "Target_WY",
            "Target_WZ",
            "Error_X",
            "Error_Y",
            "Error_Z",
            "Error_Roll",
            "Error_Pitch",
            "Error_Yaw",
            "Error_VX",
            "Error_VY",
            "Error_VZ",
            "Error_WX",
            "Error_WY",
            "Error_WZ",
            "MPC_Computation_Time",
            "MPC_Status",
            "MPC_Solver",
            "MPC_Solver_Time_Limit",
            "MPC_Solve_Time",
            "MPC_Time_Limit_Exceeded",
            "MPC_Fallback_Used",
            "MPC_Objective",
            "MPC_Iterations",
            "MPC_Optimality_Gap",
            "Command_Vector",
            "Command_Hex",
            "Command_Sent_Time",
            "Total_Active_Thrusters",
            "Thruster_Switches",
            "Total_MPC_Loop_Time",
            "Timing_Violation",
        ]

    def _get_terminal_log_headers(self) -> List[str]:
        """Get CSV headers for terminal log."""
        return [
            "Time",
            "Status",
            "Stabilization_Time",
            "Position_Error_m",
            "Angle_Error_deg",
            "Active_Thrusters",
            "Solve_Time_s",
            "Next_Update_s",
        ]

    def _format_value(self, header: str, value: Any) -> str:
        """
        Format numeric values with appropriate precision based on column type.

        Args:
            header: Column name
            value: Value to format

        Returns:
            Formatted string value
        """
        # Handle empty/None values
        if value is None or value == "":
            return ""

        # Boolean values
        if isinstance(value, bool):
            return str(value)

        # String values (includes Command_Vector, Command_Hex, Status)
        if isinstance(value, str):
            return value

        # Numeric formatting based on column type
        try:
            num_value = float(value)

            # Integer columns
            if header in [
                "Step",
                "Waypoint_Number",
                "Total_Active_Thrusters",
                "Thruster_Switches",
                "MPC_Iterations",
            ]:
                return str(int(num_value))

            # Time values - 4 decimals (0.1ms precision)
            elif header in [
                "MPC_Start_Time",
                "Control_Time",
                "Actual_Time_Interval",
                "Command_Sent_Time",
                "MPC_Computation_Time",
                "MPC_Solve_Time",
                "Total_MPC_Loop_Time",
            ]:
                return f"{num_value:.4f}"

            # Configuration values - 3 decimals
            elif header in ["CONTROL_DT", "MPC_Solver_Time_Limit"]:
                return f"{num_value:.3f}"

            # Telemetry positions (mm) - 2 decimals (0.01mm precision)
            elif header in ["Telemetry_X_mm", "Telemetry_Y_mm", "Telemetry_Z_mm"]:
                return f"{num_value:.2f}"

            # Telemetry angle (degrees) - 2 decimals (0.01 degree precision)
            elif header in ["Telemetry_Roll_deg", "Telemetry_Pitch_deg", "Telemetry_Yaw_deg"]:
                return f"{num_value:.2f}"

            # Position values (meters) - 5 decimals (0.01mm precision)
            elif header in [
                "Current_X",
                "Current_Y",
                "Current_Z",
                "Target_X",
                "Target_Y",
                "Target_Z",
                "Error_X",
                "Error_Y",
                "Error_Z",
            ]:
                return f"{num_value:.5f}"

            # Angle values (radians) - 5 decimals (0.01 degree precision)
            elif header in [
                "Current_Roll",
                "Current_Pitch",
                "Current_Yaw",
                "Target_Roll",
                "Target_Pitch",
                "Target_Yaw",
                "Error_Roll",
                "Error_Pitch",
                "Error_Yaw",
            ]:
                return f"{num_value:.5f}"

            # Velocity values - 5 decimals (0.01mm/s precision)
            elif header in [
                "Current_VX",
                "Current_VY",
                "Current_VZ",
                "Current_WX",
                "Current_WY",
                "Current_WZ",
                "Target_VX",
                "Target_VY",
                "Target_VZ",
                "Target_WX",
                "Target_WY",
                "Target_WZ",
                "Error_VX",
                "Error_VY",
                "Error_VZ",
                "Error_WX",
                "Error_WY",
                "Error_WZ",
            ]:
                return f"{num_value:.5f}"

            # Objective value and optimality gap - 3 decimals
            elif header in ["MPC_Objective", "MPC_Optimality_Gap"]:
                return f"{num_value:.3f}"

            # Default: 6 decimals for unknown numeric columns
            else:
                return f"{num_value:.6f}"

        except (ValueError, TypeError):
            # If conversion fails, return as-is
            return str(value)

    def _format_terminal_value(self, header: str, value: Any) -> str:
        """
        Format terminal log values with appropriate precision.

        Args:
            header: Column name
            value: Value to format

        Returns:
            Formatted string value
        """
        if value is None or value == "":
            return ""

        if isinstance(value, str):
            return value

        try:
            num_value = float(value)

            # Time values - 4 decimals (0.1ms precision)
            if header in [
                "Time",
                "Stabilization_Time",
                "Solve_Time_s",
                "Next_Update_s",
            ]:
                return f"{num_value:.4f}"

            # Position error - 5 decimals (0.01mm precision)
            elif header == "Position_Error_m":
                return f"{num_value:.5f}"

            # Angle error - 2 decimals (0.01 degree precision)
            elif header == "Angle_Error_deg":
                return f"{num_value:.2f}"

            else:
                return f"{num_value:.4f}"

        except (ValueError, TypeError):
            return str(value)

    def get_log_count(self) -> int:
        """Get the total number of logged entries recorded in this session."""
        return self.total_steps_recorded

    def clear_logs(self) -> None:
        """Clear all logged data."""
        self.detailed_log_data = []
        self.terminal_log_data = []
        self.current_step = 0
        self.stats_solve_times = []
        self.stats_timing_violations = 0
        self.stats_time_limit_exceeded = 0
        self.total_steps_recorded = 0
        self._headers_written = False

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics from logged data.

        Returns:
            Dictionary containing summary statistics
        """
        if self.total_steps_recorded == 0:
            return {}

        stats = {
            "total_steps": self.total_steps_recorded,
            "mode": self.mode,
        }

        if self.stats_solve_times:
            import numpy as np

            stats["avg_solve_time"] = float(np.mean(self.stats_solve_times))
            stats["max_solve_time"] = float(np.max(self.stats_solve_times))
            stats["min_solve_time"] = float(np.min(self.stats_solve_times))
            stats["std_solve_time"] = float(np.std(self.stats_solve_times))

        stats["timing_violations"] = self.stats_timing_violations
        stats["time_limit_exceeded"] = self.stats_time_limit_exceeded

        return stats

    def print_summary(self) -> None:
        """Print summary of logged data."""
        stats = self.get_summary_stats()

        if not stats:
            print("No data logged")
            return

        print("\n" + "=" * 60)
        print(f"DATA LOGGER SUMMARY ({self.mode.upper()} MODE)")
        print("=" * 60)
        print(f"Total steps logged: {stats['total_steps']}")

        if "avg_solve_time" in stats:
            print("\nMPC Solve Time Statistics:")
            print(f"  Average: {stats['avg_solve_time'] * 1000:.2f} ms")
            print(f"  Min:     {stats['min_solve_time'] * 1000:.2f} ms")
            print(f"  Max:     {stats['max_solve_time'] * 1000:.2f} ms")
            print(f"  Std Dev: {stats['std_solve_time'] * 1000:.2f} ms")

        print("\nTiming Performance:")
        print(f"  Timing violations:     {stats['timing_violations']}")
        print(f"  Time limits exceeded:  {stats['time_limit_exceeded']}")

        print("=" * 60 + "\n")


def create_data_logger(mode: str = "simulation", filename: str = "control_data.csv") -> DataLogger:
    """
    Factory function to create a data logger.

    Args:
        mode: "simulation" or "physics"
        filename: Output CSV filename

    Returns:
        Configured DataLogger instance
    """
    return DataLogger(mode=mode, filename=filename)
