"""
Reaction Wheel Configuration

Physical parameters and constraints for cubesat reaction wheel actuators.
Typical values based on commercial cubesat reaction wheels (e.g., Blue Canyon XACT).
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ReactionWheelParams:
    """
    Physical parameters for a single reaction wheel.

    Attributes:
        inertia: Wheel moment of inertia [kg·m²]
        max_torque: Maximum motor torque [N·m]
        max_speed: Maximum wheel speed [rad/s]
        friction: Viscous friction coefficient [N·m·s/rad]
        axis: Rotation axis in body frame (unit vector)
    """

    inertia: float = 0.001  # kg·m² (typical cubesat RW: 0.0005-0.002)
    max_torque: float = 0.1  # N·m (increased for faster attitude control)
    max_speed: float = 628.0  # rad/s (~6000 RPM)
    friction: float = 0.0001  # N·m·s/rad
    axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    @property
    def max_momentum(self) -> float:
        """Maximum angular momentum storage [N·m·s]."""
        return self.inertia * self.max_speed

    @property
    def axis_array(self) -> np.ndarray:
        """Return axis as numpy array."""
        return np.array(self.axis, dtype=np.float64)


@dataclass
class ReactionWheelArrayConfig:
    """
    Configuration for 3-axis reaction wheel assembly.

    Standard orthogonal configuration with one wheel per axis.
    """

    wheel_x: ReactionWheelParams
    wheel_y: ReactionWheelParams
    wheel_z: ReactionWheelParams

    # Saturation warning thresholds (fraction of max speed)
    saturation_warning_threshold: float = 0.8
    saturation_critical_threshold: float = 0.95

    @classmethod
    def create_default(cls) -> "ReactionWheelArrayConfig":
        """Create default 3-axis reaction wheel configuration."""
        return cls(
            wheel_x=ReactionWheelParams(axis=(1.0, 0.0, 0.0)),
            wheel_y=ReactionWheelParams(axis=(0.0, 1.0, 0.0)),
            wheel_z=ReactionWheelParams(axis=(0.0, 0.0, 1.0)),
        )

    def get_wheel(self, axis: int) -> ReactionWheelParams:
        """
        Get wheel parameters by axis index.

        Args:
            axis: 0 for X, 1 for Y, 2 for Z

        Returns:
            ReactionWheelParams for the specified axis
        """
        if axis == 0:
            return self.wheel_x
        elif axis == 1:
            return self.wheel_y
        elif axis == 2:
            return self.wheel_z
        else:
            raise ValueError(f"Invalid axis index: {axis}. Must be 0, 1, or 2.")

    @property
    def inertia_matrix(self) -> np.ndarray:
        """
        Diagonal inertia matrix for all wheels [3x3].

        Returns:
            Diagonal matrix with wheel inertias
        """
        return np.diag(
            [
                self.wheel_x.inertia,
                self.wheel_y.inertia,
                self.wheel_z.inertia,
            ]
        )

    @property
    def max_torque_vector(self) -> np.ndarray:
        """Maximum torque vector [3]."""
        return np.array(
            [
                self.wheel_x.max_torque,
                self.wheel_y.max_torque,
                self.wheel_z.max_torque,
            ]
        )

    @property
    def max_speed_vector(self) -> np.ndarray:
        """Maximum speed vector [3]."""
        return np.array(
            [
                self.wheel_x.max_speed,
                self.wheel_y.max_speed,
                self.wheel_z.max_speed,
            ]
        )

    def check_saturation(self, wheel_speeds: np.ndarray) -> dict:
        """
        Check wheel saturation status.

        Args:
            wheel_speeds: Current wheel speeds [ωx, ωy, ωz] in rad/s

        Returns:
            Dict with saturation info for each wheel
        """
        max_speeds = self.max_speed_vector
        ratios = np.abs(wheel_speeds) / max_speeds

        return {
            "ratios": ratios,
            "warning": np.any(ratios > self.saturation_warning_threshold),
            "critical": np.any(ratios > self.saturation_critical_threshold),
            "saturated_axes": np.where(ratios > self.saturation_critical_threshold)[0].tolist(),
        }


# Default configuration
def get_reaction_wheel_config() -> ReactionWheelArrayConfig:
    """Get default reaction wheel configuration."""
    return ReactionWheelArrayConfig.create_default()


def calculate_wheel_dynamics(
    wheel_speeds: np.ndarray,
    commanded_torques: np.ndarray,
    config: ReactionWheelArrayConfig,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate reaction wheel dynamics for one timestep.

    Args:
        wheel_speeds: Current wheel speeds [ωx, ωy, ωz] rad/s
        commanded_torques: Commanded motor torques [τx, τy, τz] N·m
        config: Reaction wheel configuration
        dt: Timestep in seconds

    Returns:
        Tuple of (new_wheel_speeds, reaction_torques_on_satellite)
    """
    # Clamp torques to limits
    max_torques = config.max_torque_vector
    clamped_torques = np.clip(commanded_torques, -max_torques, max_torques)

    # Wheel dynamics: I_w * ω̇_w = τ_motor - friction * ω_w
    inertias = np.array(
        [
            config.wheel_x.inertia,
            config.wheel_y.inertia,
            config.wheel_z.inertia,
        ]
    )
    frictions = np.array(
        [
            config.wheel_x.friction,
            config.wheel_y.friction,
            config.wheel_z.friction,
        ]
    )

    # Wheel acceleration
    wheel_accel = (clamped_torques - frictions * wheel_speeds) / inertias

    # Integrate wheel speeds
    new_speeds = wheel_speeds + wheel_accel * dt

    # Clamp to max speeds
    max_speeds = config.max_speed_vector
    new_speeds = np.clip(new_speeds, -max_speeds, max_speeds)

    # Reaction torque on satellite (Newton's 3rd law: equal and opposite)
    reaction_torques = -clamped_torques

    return new_speeds, reaction_torques
