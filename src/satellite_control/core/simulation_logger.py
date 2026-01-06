"""
Simulation Logger Module

Handles the formatting and logging of simulation data.
De-clutters the main simulation loop by encapsulating the data mapping logic.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

# from src.satellite_control.config import SatelliteConfig  # Unused import removed
from src.satellite_control.utils.data_logger import DataLogger

logger = logging.getLogger(__name__)


class SimulationLogger:
    """
    Handles logging of simulation steps to DataLogger.
    """

    def __init__(self, data_logger: DataLogger):
        self.data_logger = data_logger

    def log_step(
        self,
        # Typed as SimulationContext, but using Any to avoid circular
        # imports at runtime if needed
        context: Any,
        mpc_start_time: float,
        command_sent_time: float,
        thruster_action: np.ndarray,
        # Removed mpc_computation_time from signature as per edit,
        # it will be calculated or passed via mpc_info
        mpc_info: Optional[Dict[str, Any]],
    ) -> None:
        """
        Log detailed simulation step data using SimulationContext.
        """
        # The following lines are moved/interpreted from the user's edit
        # to be syntactically correct and functional within the method body.
        # Assuming mpc_computation_time is now derived or passed in mpc_info.
        # If mpc_info is None, initialize it as an empty dict to avoid errors.
        if mpc_info is None:
            mpc_info = {}

        # Unwrap context
        current_state = context.current_state
        target_state = context.target_state
        simulation_time = context.simulation_time
        control_update_interval = context.control_dt
        step_number = context.step_number
        mission_phase = context.mission_phase
        waypoint_number = context.waypoint_number
        previous_thruster_action = context.previous_thruster_command

        # Calculate errors
        error_x = current_state[0] - target_state[0]
        error_y = current_state[1] - target_state[1]

        # Angle difference
        error_yaw = target_state[4] - current_state[4]
        error_yaw = (error_yaw + np.pi) % (2 * np.pi) - np.pi

        # Command strings
        command_vector_binary = (thruster_action > 0.5).astype(int)
        command_hex = "0x" + "".join([str(x) for x in command_vector_binary])

        command_vector_str = "[" + ", ".join([f"{x:.3f}" for x in thruster_action]) + "]"

        # Timing
        total_mpc_loop_time = command_sent_time - mpc_start_time
        actual_time_interval = control_update_interval  # Simplified

        # MPC Info
        mpc_status_name = mpc_info.get("status_name") if mpc_info else None
        mpc_solver_type = mpc_info.get("solver_type") if mpc_info else None
        mpc_time_limit = mpc_info.get("solver_time_limit") if mpc_info else None
        mpc_time_exceeded = mpc_info.get("time_limit_exceeded") if mpc_info else None
        mpc_fallback_used = mpc_info.get("solver_fallback") if mpc_info else None
        mpc_objective = mpc_info.get("objective_value") if mpc_info else None
        mpc_solve_time = mpc_info.get("solve_time") if mpc_info else None
        mpc_iterations = mpc_info.get("iterations") if mpc_info else None
        mpc_optimality_gap = mpc_info.get("optimality_gap") if mpc_info else None
        # Assuming mpc_computation_time is now derived from mpc_solve_time or other means
        mpc_computation_time = mpc_info.get(
            "mpc_computation_time", 0.0
        )  # Default to 0 if not found

        # Velocity errors
        error_vx = target_state[2] - current_state[2]
        error_vy = target_state[3] - current_state[3]
        error_angular_vel = target_state[5] - current_state[5]

        # Active Thrusters
        total_active_thrusters = int(np.sum(thruster_action > 0.01))

        # Switches
        thruster_switches = 0
        if previous_thruster_action is not None:
            # The user's edit here seems to be a copy-paste error from a control loop.
            # It's syntactically incorrect and doesn't fit the logging context.
            # Reverting to the original logic for calculating thruster switches.
            thruster_switches = int(
                np.sum(np.abs(thruster_action - previous_thruster_action) > 0.01)
            )

        log_entry = {
            "Step": step_number,
            "MPC_Start_Time": mpc_start_time,
            "Control_Time": simulation_time,
            "Actual_Time_Interval": actual_time_interval,
            "CONTROL_DT": control_update_interval,
            "Mission_Phase": mission_phase,
            "Waypoint_Number": waypoint_number,
            "Telemetry_X_mm": current_state[0] * 1000,
            "Telemetry_Z_mm": current_state[1] * 1000,
            "Telemetry_Yaw_deg": np.degrees(current_state[4]),
            "Current_X": current_state[0],
            "Current_Y": current_state[1],
            "Current_Yaw": current_state[4],
            "Current_VX": current_state[2],
            "Current_VY": current_state[3],
            "Current_Angular_Vel": current_state[5],
            "Target_X": target_state[0],
            "Target_Y": target_state[1],
            "Target_Yaw": target_state[4],
            "Target_VX": target_state[2],
            "Target_VY": target_state[3],
            "Target_Angular_Vel": target_state[5],
            "Error_X": error_x,
            "Error_Y": error_y,
            "Error_Yaw": error_yaw,
            "Error_VX": error_vx,
            "Error_VY": error_vy,
            "Error_Angular_Vel": error_angular_vel,
            "MPC_Computation_Time": mpc_computation_time,
            "MPC_Status": mpc_status_name,
            "MPC_Solver": mpc_solver_type,
            "MPC_Solver_Time_Limit": mpc_time_limit,
            "MPC_Solve_Time": mpc_solve_time,
            "MPC_Time_Limit_Exceeded": mpc_time_exceeded,
            "MPC_Fallback_Used": mpc_fallback_used,
            "MPC_Objective": mpc_objective,
            "MPC_Iterations": mpc_iterations,
            "MPC_Optimality_Gap": mpc_optimality_gap,
            "Command_Vector": command_vector_str,
            "Command_Hex": command_hex,
            "Command_Sent_Time": command_sent_time,
            "Total_Active_Thrusters": total_active_thrusters,
            "Thruster_Switches": thruster_switches,
            "Total_MPC_Loop_Time": total_mpc_loop_time,
            "Timing_Violation": ("YES" if total_mpc_loop_time > control_update_interval else "NO"),
        }

        self.data_logger.log_entry(log_entry)

    def log_physics_step(
        self,
        simulation_time: float,
        current_state: np.ndarray,
        target_state: np.ndarray,
        thruster_actual_output: np.ndarray,
        thruster_last_command: np.ndarray,
        normalize_angle_func: Optional[Any] = None,
    ) -> None:
        """
        Log high-frequency physics data.
        """
        # Calculate errors
        error_x = target_state[0] - current_state[0]
        error_y = target_state[1] - current_state[1]

        raw_yaw_error = target_state[4] - current_state[4]
        if normalize_angle_func:
            error_yaw = normalize_angle_func(raw_yaw_error)
        else:
            # Fallback simple normalization
            error_yaw = (raw_yaw_error + np.pi) % (2 * np.pi) - np.pi

        # Format Command Vector string
        cmd_vec_str = "[" + ", ".join([f"{val:.3f}" for val in thruster_actual_output]) + "]"

        entry = {
            "Time": f"{simulation_time:.4f}",
            "Current_X": f"{current_state[0]:.5f}",
            "Current_Y": f"{current_state[1]:.5f}",
            "Current_Yaw": f"{current_state[4]:.5f}",
            "Current_VX": f"{current_state[2]:.5f}",
            "Current_VY": f"{current_state[3]:.5f}",
            "Current_Angular_Vel": f"{current_state[5]:.5f}",
            "Target_X": f"{target_state[0]:.5f}",
            "Target_Y": f"{target_state[1]:.5f}",
            "Target_Yaw": f"{target_state[4]:.5f}",
            "Error_X": f"{error_x:.5f}",
            "Error_Y": f"{error_y:.5f}",
            "Error_Yaw": f"{error_yaw:.5f}",
            "Command_Vector": cmd_vec_str,
        }

        # Log Thruster States
        for i in range(8):
            entry[f"Thruster_{i+1}_Cmd"] = f"{thruster_last_command[i]:.3f}"
            entry[f"Thruster_{i+1}_Val"] = f"{thruster_actual_output[i]:.3f}"

        self.data_logger.log_entry(entry)
