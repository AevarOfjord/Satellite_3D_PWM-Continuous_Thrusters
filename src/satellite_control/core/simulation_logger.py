"""
Simulation Logger Module

Handles the formatting and logging of simulation data.
De-clutters the main simulation loop by encapsulating the data mapping logic.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

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
        # Typed as SimulationContext, but using Any
        context: Any,
        mpc_start_time: float,
        command_sent_time: float,
        thruster_action: np.ndarray,
        mpc_info: Optional[Dict[str, Any]],
    ) -> None:
        """
        Log detailed simulation step data using SimulationContext.
        """
        if mpc_info is None:
            mpc_info = {}

        # Unwrap context
        current_state = context.current_state  # 13 elements
        target_state = context.target_state  # 13 elements
        simulation_time = context.simulation_time
        control_update_interval = context.control_dt
        step_number = context.step_number
        mission_phase = context.mission_phase
        waypoint_number = context.waypoint_number
        previous_thruster_action = context.previous_thruster_command

        # Extract 3D State
        # Pos
        curr_x, curr_y, curr_z = current_state[0:3]
        targ_x, targ_y, targ_z = target_state[0:3]

        # Quat -> Yaw
        q = current_state[3:7]
        # Quat -> Euler (scipy)
        from scipy.spatial.transform import Rotation

        q_scipy = [q[1], q[2], q[3], q[0]]  # xyzw for scipy, wxyz input
        curr_euler = Rotation.from_quat(q_scipy).as_euler("xyz", degrees=False)
        curr_roll, curr_pitch, curr_yaw = curr_euler

        q_t = target_state[3:7]
        q_t_scipy = [q_t[1], q_t[2], q_t[3], q_t[0]]
        targ_euler = Rotation.from_quat(q_t_scipy).as_euler("xyz", degrees=False)
        targ_roll, targ_pitch, targ_yaw = targ_euler

        # Vel
        curr_vx, curr_vy, curr_vz = current_state[7:10]
        targ_vx, targ_vy, targ_vz = target_state[7:10]

        # Ang Vel (X, Y, Z)
        curr_wx, curr_wy, curr_wz = current_state[10:13]
        targ_wx, targ_wy, targ_wz = target_state[10:13]

        # Calculate errors
        error_x = curr_x - targ_x
        error_y = curr_y - targ_y
        error_z = curr_z - targ_z

        # Angle difference (wrap to [-pi, pi])
        def wrap_angle(angle: float) -> float:
            return (angle + np.pi) % (2 * np.pi) - np.pi

        error_roll = wrap_angle(targ_roll - curr_roll)
        error_pitch = wrap_angle(targ_pitch - curr_pitch)
        error_yaw = wrap_angle(targ_yaw - curr_yaw)

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
        mpc_computation_time = mpc_info.get("mpc_computation_time", 0.0)

        # Velocity errors
        error_vx = targ_vx - curr_vx
        error_vy = targ_vy - curr_vy
        error_vz = targ_vz - curr_vz
        error_wx = targ_wx - curr_wx
        error_wy = targ_wy - curr_wy
        error_wz = targ_wz - curr_wz

        # Active Thrusters
        total_active_thrusters = int(np.sum(thruster_action > 0.01))

        # Switches
        thruster_switches = 0
        if previous_thruster_action is not None:
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
            "Telemetry_X_mm": curr_x * 1000,
            "Telemetry_Y_mm": curr_y * 1000,
            "Telemetry_Z_mm": curr_z * 1000,
            "Telemetry_Roll_deg": np.degrees(curr_roll),
            "Telemetry_Pitch_deg": np.degrees(curr_pitch),
            "Telemetry_Yaw_deg": np.degrees(curr_yaw),
            "Current_X": curr_x,
            "Current_Y": curr_y,
            "Current_Z": curr_z,
            "Current_Yaw": curr_yaw,
            "Current_Roll": curr_roll,
            "Current_Pitch": curr_pitch,
            "Current_VX": curr_vx,
            "Current_VY": curr_vy,
            "Current_VZ": curr_vz,
            "Current_WX": curr_wx,
            "Current_WY": curr_wy,
            "Current_WZ": curr_wz,
            "Target_X": targ_x,
            "Target_Y": targ_y,
            "Target_Z": targ_z,
            "Target_Yaw": targ_yaw,
            "Target_Roll": targ_roll,
            "Target_Pitch": targ_pitch,
            "Target_VX": targ_vx,
            "Target_VY": targ_vy,
            "Target_VZ": targ_vz,
            "Target_WX": targ_wx,
            "Target_WY": targ_wy,
            "Target_WZ": targ_wz,
            "Error_X": error_x,
            "Error_Y": error_y,
            "Error_Z": error_z,
            "Error_Yaw": error_yaw,
            "Error_Roll": error_roll,
            "Error_Pitch": error_pitch,
            "Error_VX": error_vx,
            "Error_VY": error_vy,
            "Error_VZ": error_vz,
            "Error_WX": error_wx,
            "Error_WY": error_wy,
            "Error_WZ": error_wz,
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
        # Extract 3D components
        curr_x, curr_y, curr_z = current_state[0:3]

        from src.satellite_control.utils.orientation_utils import quat_wxyz_to_euler_xyz

        q = current_state[3:7]
        curr_roll, curr_pitch, curr_yaw = quat_wxyz_to_euler_xyz(q)

        curr_vx, curr_vy, curr_vz = current_state[7:10]
        curr_wx, curr_wy, curr_wz = current_state[10:13]

        targ_x, targ_y, targ_z = target_state[0:3]

        q_t = target_state[3:7]
        targ_roll, targ_pitch, targ_yaw = quat_wxyz_to_euler_xyz(q_t)

        # Calculate errors
        error_x = targ_x - curr_x
        error_y = targ_y - curr_y
        error_z = targ_z - curr_z

        def wrap_angle(angle: float) -> float:
            if normalize_angle_func:
                return normalize_angle_func(angle)
            return (angle + np.pi) % (2 * np.pi) - np.pi

        error_roll = wrap_angle(targ_roll - curr_roll)
        error_pitch = wrap_angle(targ_pitch - curr_pitch)
        error_yaw = wrap_angle(targ_yaw - curr_yaw)

        # Format Command Vector string
        cmd_vec_str = "[" + ", ".join([f"{val:.3f}" for val in thruster_actual_output]) + "]"

        entry = {
            "Time": f"{simulation_time:.4f}",
            "Current_X": f"{curr_x:.5f}",
            "Current_Y": f"{curr_y:.5f}",
            "Current_Z": f"{curr_z:.5f}",
            "Current_Roll": f"{curr_roll:.5f}",
            "Current_Pitch": f"{curr_pitch:.5f}",
            "Current_Yaw": f"{curr_yaw:.5f}",
            "Current_VX": f"{curr_vx:.5f}",
            "Current_VY": f"{curr_vy:.5f}",
            "Current_VZ": f"{curr_vz:.5f}",
            "Current_WX": f"{curr_wx:.5f}",
            "Current_WY": f"{curr_wy:.5f}",
            "Current_WZ": f"{curr_wz:.5f}",
            "Target_X": f"{targ_x:.5f}",
            "Target_Y": f"{targ_y:.5f}",
            "Target_Z": f"{targ_z:.5f}",
            "Target_Roll": f"{targ_roll:.5f}",
            "Target_Pitch": f"{targ_pitch:.5f}",
            "Target_Yaw": f"{targ_yaw:.5f}",
            "Error_X": f"{error_x:.5f}",
            "Error_Y": f"{error_y:.5f}",
            "Error_Z": f"{error_z:.5f}",
            "Error_Roll": f"{error_roll:.5f}",
            "Error_Pitch": f"{error_pitch:.5f}",
            "Error_Yaw": f"{error_yaw:.5f}",
            "Command_Vector": cmd_vec_str,
        }

        # Log Thruster States (12 thrusters)
        num_thrusters = len(thruster_actual_output)
        for i in range(num_thrusters):
            entry[f"Thruster_{i+1}_Cmd"] = f"{thruster_last_command[i]:.3f}"
            entry[f"Thruster_{i+1}_Val"] = f"{thruster_actual_output[i]:.3f}"

        self.data_logger.log_entry(entry)
