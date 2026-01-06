"""
3D State Format Converter Utility

Provides conversion between simulation and MPC state formats for 3D control.

State Formats:
- Simulation/MuJoCo: [x, y, z, qw, qx, qy, qz, vx, vy, vz, ωx, ωy, ωz] (13 elements)
- MPC (linearized):  [x, y, z, qe_x, qe_y, qe_z, vx, vy, vz, ωx, ωy, ωz] (12 elements)

The MPC uses a linearized quaternion error representation for the solver.
"""

from typing import List, Union

import numpy as np


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """
    Compute quaternion error as 3-element vector.

    Returns the vector part of the error quaternion (scaled by 2),
    which is proportional to the axis-angle error for small rotations.
    """
    q_current = normalize_quaternion(q_current)
    q_target = normalize_quaternion(q_target)

    # Quaternion conjugate of target
    q_target_conj = np.array([q_target[0], -q_target[1], -q_target[2], -q_target[3]])

    # Error quaternion: q_err = q_target^-1 * q_current
    w1, x1, y1, z1 = q_target_conj
    w2, x2, y2, z2 = q_current

    q_err = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )

    # Ensure shortest path (positive scalar part)
    if q_err[0] < 0:
        q_err = -q_err

    # Return vector part scaled by 2 (axis-angle approximation)
    return 2.0 * q_err[1:4]


class StateConverter3D:
    """
    Convert between 3D state vector formats used in simulation and MPC.

    Simulation uses full quaternion (13 elements).
    MPC uses linearized quaternion error (12 elements).
    """

    # Index mappings for simulation state (13 elements)
    SIM_X, SIM_Y, SIM_Z = 0, 1, 2
    SIM_QW, SIM_QX, SIM_QY, SIM_QZ = 3, 4, 5, 6
    SIM_VX, SIM_VY, SIM_VZ = 7, 8, 9
    SIM_WX, SIM_WY, SIM_WZ = 10, 11, 12

    # Index mappings for MPC state (12 elements)
    MPC_X, MPC_Y, MPC_Z = 0, 1, 2
    MPC_QEX, MPC_QEY, MPC_QEZ = 3, 4, 5
    MPC_VX, MPC_VY, MPC_VZ = 6, 7, 8
    MPC_WX, MPC_WY, MPC_WZ = 9, 10, 11

    @staticmethod
    def sim_to_mpc(state: np.ndarray, target_quat: np.ndarray = None) -> np.ndarray:
        """
        Convert simulation state to MPC state format.

        Args:
            state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, ωx, ωy, ωz] (13 elements)
            target_quat: Target quaternion [qw, qx, qy, qz] for error computation
                        (defaults to identity)

        Returns:
            [x, y, z, qe_x, qe_y, qe_z, vx, vy, vz, ωx, ωy, ωz] (12 elements)
        """
        if len(state) != 13:
            raise ValueError(f"Expected 13-element state, got {len(state)}")

        if target_quat is None:
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        current_quat = state[3:7]
        quat_err = quaternion_error(current_quat, target_quat)

        return np.array(
            [
                state[0],  # x
                state[1],  # y
                state[2],  # z
                quat_err[0],  # qe_x
                quat_err[1],  # qe_y
                quat_err[2],  # qe_z
                state[7],  # vx
                state[8],  # vy
                state[9],  # vz
                state[10],  # ωx
                state[11],  # ωy
                state[12],  # ωz
            ],
            dtype=np.float64,
        )

    @staticmethod
    def get_state_from_mujoco(data) -> np.ndarray:
        """
        Extract 13-element state from MuJoCo data.

        Args:
            data: MuJoCo data object

        Returns:
            [x, y, z, qw, qx, qy, qz, vx, vy, vz, ωx, ωy, ωz]
        """
        # For free joint: qpos = [x, y, z, qw, qx, qy, qz]
        # qvel = [vx, vy, vz, ωx, ωy, ωz]
        return np.concatenate(
            [
                data.qpos[:7],  # position + quaternion
                data.qvel[:6],  # velocity + angular velocity
            ]
        )

    @staticmethod
    def set_state_to_mujoco(data, state: np.ndarray) -> None:
        """
        Set 13-element state to MuJoCo data.

        Args:
            data: MuJoCo data object
            state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, ωx, ωy, ωz]
        """
        if len(state) != 13:
            raise ValueError(f"Expected 13-element state, got {len(state)}")

        # position + quaternion
        data.qpos[:7] = state[:7]
        # Normalize quaternion
        data.qpos[3:7] = normalize_quaternion(data.qpos[3:7])
        # velocity + angular velocity
        data.qvel[:6] = state[7:13]


# Legacy compatibility: keep 2D converter available
from src.satellite_control.utils.state_converter import StateConverter as StateConverter2D
