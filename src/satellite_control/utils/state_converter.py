"""
State Format Converter Utility

Provides centralized conversion between simulation and MPC state formats.

State Formats:
- Simulation: [x, y, vx, vy, theta, omega]
- MPC:        [x, y, theta, vx, vy, omega]
"""

from typing import List, Union

import numpy as np


class StateConverter:
    """
    Convert between state vector formats used in simulation and MPC.

    The simulation uses [x, y, vx, vy, theta, omega] format for consistency
    with physics (position, velocity pairs).

    The MPC uses [x, y, theta, vx, vy, omega] format which groups all
    positions together and all velocities together.
    """

    # Index mappings
    SIM_X, SIM_Y, SIM_VX, SIM_VY, SIM_THETA, SIM_OMEGA = 0, 1, 2, 3, 4, 5
    MPC_X, MPC_Y, MPC_THETA, MPC_VX, MPC_VY, MPC_OMEGA = 0, 1, 2, 3, 4, 5

    @staticmethod
    def sim_to_mpc(state: np.ndarray) -> np.ndarray:
        """
        Convert simulation state to MPC state format.

        Args:
            state: [x, y, vx, vy, theta, omega] (2D) OR
                   [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (3D)

        Returns:
            Converted state vector
        """
        # 3D State (Identity)
        if len(state) == 13:
            return state.copy()

        if len(state) != 6:
            raise ValueError(f"Expected 6 or 13-element state, got {len(state)}")

        return np.array(
            [
                state[0],  # x
                state[1],  # y
                state[4],  # theta (was index 4, now index 2)
                state[2],  # vx (was index 2, now index 3)
                state[3],  # vy (was index 3, now index 4)
                state[5],  # omega
            ],
            dtype=np.float64,
        )

    @staticmethod
    def mpc_to_sim(state: np.ndarray) -> np.ndarray:
        """
        Convert MPC state to simulation state format.

        Args:
            state: [x, y, theta, vx, vy, omega] (2D) OR
                   [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (3D)

        Returns:
            Converted state vector
        """
        # 3D State (Identity)
        if len(state) == 13:
            return state.copy()

        if len(state) != 6:
            raise ValueError(f"Expected 6 or 13-element state, got {len(state)}")

        return np.array(
            [
                state[0],  # x
                state[1],  # y
                state[3],  # vx (was index 3, now index 2)
                state[4],  # vy (was index 4, now index 3)
                state[2],  # theta (was index 2, now index 4)
                state[5],  # omega
            ],
            dtype=np.float64,
        )

    @staticmethod
    def sim_to_mpc_trajectory(
        trajectory: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """
        Convert a trajectory of simulation states to MPC format.

        Args:
            trajectory: Array of shape (N, 6) or list of 6-element arrays

        Returns:
            Array of shape (N, 6) in MPC format
        """
        trajectory = np.array(trajectory)
        if trajectory.ndim == 1:
            return StateConverter.sim_to_mpc(trajectory)

        return np.array([StateConverter.sim_to_mpc(state) for state in trajectory])

    @staticmethod
    def mpc_to_sim_trajectory(
        trajectory: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """
        Convert a trajectory of MPC states to simulation format.

        Args:
            trajectory: Array of shape (N, 6) or list of 6-element arrays

        Returns:
            Array of shape (N, 6) in simulation format
        """
        trajectory = np.array(trajectory)
        if trajectory.ndim == 1:
            return StateConverter.mpc_to_sim(trajectory)

        return np.array([StateConverter.mpc_to_sim(state) for state in trajectory])
