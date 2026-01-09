"""
Orbital Dynamics Module

Implements Hill-Clohessy-Wiltshire (CW) equations for relative motion
around a target satellite in circular orbit.

CW Equations (relative motion in Hill/LVLH frame):
    ẍ = 2nẏ + 3n²x + ax    (Radial - away from Earth)
    ÿ = -2nẋ + ay          (Along-track - velocity direction)
    z̈ = -n²z + az          (Cross-track - out of plane)

where n = √(μ/a³) is the orbital mean motion and a is thrust acceleration.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.satellite_control.config.orbital_config import OrbitalConfig


@dataclass
class CWDynamics:
    """
    Hill-Clohessy-Wiltshire relative motion dynamics.

    Computes accelerations due to gravity gradient effects
    for a chaser satellite relative to a target in circular orbit.
    """

    orbital_config: OrbitalConfig

    def __post_init__(self):
        """Cache frequently used values."""
        self.n = self.orbital_config.mean_motion
        self.n_sq = self.n**2

    def compute_acceleration(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute CW gravitational acceleration.

        Args:
            position: Relative position [x, y, z] in Hill frame [m]
            velocity: Relative velocity [vx, vy, vz] in Hill frame [m/s]

        Returns:
            Gravitational acceleration [ax, ay, az] [m/s²]
        """
        x, y, z = position[0], position[1], position[2]
        vx, vy, vz = velocity[0], velocity[1], velocity[2]

        # CW equations
        ax = 3 * self.n_sq * x + 2 * self.n * vy  # Radial
        ay = -2 * self.n * vx  # Along-track
        az = -self.n_sq * z  # Cross-track

        return np.array([ax, ay, az])

    def get_state_matrices(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get discrete-time state transition matrices for CW dynamics.

        State: [x, y, z, vx, vy, vz]

        Args:
            dt: Time step [s]

        Returns:
            (A, B) discrete-time matrices where x_{k+1} = A @ x_k + B @ u_k
        """
        n = self.n

        # Continuous-time A matrix (6x6)
        # ẋ = Ax + Bu
        # State: [x, y, z, vx, vy, vz]
        A_cont = np.array(
            [
                [0, 0, 0, 1, 0, 0],  # ẋ = vx
                [0, 0, 0, 0, 1, 0],  # ẏ = vy
                [0, 0, 0, 0, 0, 1],  # ż = vz
                [3 * n**2, 0, 0, 0, 2 * n, 0],  # v̇x = 3n²x + 2nvy
                [0, 0, 0, -2 * n, 0, 0],  # v̇y = -2nvx
                [0, 0, -(n**2), 0, 0, 0],  # v̇z = -n²z
            ]
        )

        # B matrix (6x3) - input is acceleration [ax, ay, az]
        B_cont = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

        # Discretize using forward Euler (simple but sufficient for small dt)
        A_disc = np.eye(6) + A_cont * dt
        B_disc = B_cont * dt

        return A_disc, B_disc

    def get_mpc_dynamics_matrices(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 16-state dynamics matrices including CW terms for MPC.

        Integrates CW accelerations into the full satellite state:
        [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, ωrx, ωry, ωrz]

        Args:
            dt: Time step [s]

        Returns:
            (A_cw, B_cw) matrices to add to existing MPC dynamics
        """
        n = self.n

        # Additional A terms for velocity due to CW (affects positions 7-9)
        # v̇x += 3n²x + 2nvy
        # v̇y += -2nvx
        # v̇z += -n²z

        A_cw = np.zeros((16, 16))

        # Position → velocity coupling (CW gravity gradient)
        A_cw[7, 0] = 3 * n**2 * dt  # dvx/dx
        A_cw[7, 8] = 2 * n * dt  # dvx/dvy (Coriolis)
        A_cw[8, 7] = -2 * n * dt  # dvy/dvx (Coriolis)
        A_cw[9, 2] = -(n**2) * dt  # dvz/dz

        return A_cw, np.zeros((16, 9))  # B unchanged


def compute_cw_acceleration(
    position: np.ndarray,
    velocity: np.ndarray,
    mean_motion: float,
) -> np.ndarray:
    """
    Compute CW gravitational acceleration (standalone function).

    Args:
        position: Relative position [x, y, z] [m]
        velocity: Relative velocity [vx, vy, vz] [m/s]
        mean_motion: Orbital mean motion n [rad/s]

    Returns:
        Acceleration [ax, ay, az] [m/s²]
    """
    n = mean_motion
    n_sq = n**2

    x, z = position[0], position[2]
    vx, vy = velocity[0], velocity[1]

    ax = 3 * n_sq * x + 2 * n * vy
    ay = -2 * n * vx
    az = -n_sq * z

    return np.array([ax, ay, az])


def compute_cw_force(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: float,
    orbital_config: Optional[OrbitalConfig] = None,
) -> np.ndarray:
    """
    Compute CW gravitational force for MuJoCo xfrc_applied.

    Args:
        position: Relative position [x, y, z] [m]
        velocity: Relative velocity [vx, vy, vz] [m/s]
        mass: Satellite mass [kg]
        orbital_config: Orbital configuration (default: LEO 400km)

    Returns:
        Force [Fx, Fy, Fz] [N]
    """
    if orbital_config is None:
        orbital_config = OrbitalConfig()

    accel = compute_cw_acceleration(position, velocity, orbital_config.mean_motion)

    return mass * accel
