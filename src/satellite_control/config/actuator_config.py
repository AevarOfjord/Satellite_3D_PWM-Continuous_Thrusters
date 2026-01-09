"""
Unified Actuator Configuration

Supports multiple actuator modes:
- reaction_wheels: 3 reaction wheels + 6 thrusters (new default)
- legacy_thrusters: 12 thrusters only (backward compatibility)

This module provides a unified interface for actuator configuration
regardless of the underlying actuation mode.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from .reaction_wheel_config import ReactionWheelArrayConfig, get_reaction_wheel_config


class ActuatorMode(Enum):
    """Actuator configuration modes."""

    REACTION_WHEELS = "reaction_wheels"  # RW + thrusters (new)
    LEGACY_THRUSTERS = "legacy_thrusters"  # 12 thrusters (old)


@dataclass
class ThrusterSetConfig:
    """
    Configuration for simplified thruster set (6 thrusters, 2 per axis).

    Used in reaction wheel mode for translation control.
    """

    # Thruster positions (face center of cube)
    positions: Dict[str, Tuple[float, float, float]] = field(
        default_factory=lambda: {
            "px": (0.145, 0.0, 0.0),  # +X face
            "mx": (-0.145, 0.0, 0.0),  # -X face
            "py": (0.0, 0.145, 0.0),  # +Y face
            "my": (0.0, -0.145, 0.0),  # -Y face
            "pz": (0.0, 0.0, 0.145),  # +Z face
            "mz": (0.0, 0.0, -0.145),  # -Z face
        }
    )

    # Thruster directions (thrust vector)
    directions: Dict[str, Tuple[float, float, float]] = field(
        default_factory=lambda: {
            "px": (-1.0, 0.0, 0.0),  # Push -X
            "mx": (1.0, 0.0, 0.0),  # Push +X
            "py": (0.0, -1.0, 0.0),  # Push -Y
            "my": (0.0, 1.0, 0.0),  # Push +Y
            "pz": (0.0, 0.0, -1.0),  # Push -Z
            "mz": (0.0, 0.0, 1.0),  # Push +Z
        }
    )

    # Force magnitude per thruster [N]
    force: float = 0.45

    # Thruster dynamics
    valve_delay: float = 0.04  # seconds
    rampup_time: float = 0.01  # seconds

    def get_force_vectors(self) -> Dict[str, np.ndarray]:
        """Get force vectors (direction * magnitude) for each thruster."""
        return {
            name: self.force * np.array(direction) for name, direction in self.directions.items()
        }


@dataclass
class ActuatorConfig:
    """
    Unified actuator configuration supporting multiple modes.

    Attributes:
        mode: ActuatorMode (reaction_wheels or legacy_thrusters)
        reaction_wheels: Reaction wheel config (if mode is reaction_wheels)
        thrusters: Thruster set config (if mode is reaction_wheels)
        legacy_thruster_positions: 12-thruster positions (if mode is legacy)
        legacy_thruster_directions: 12-thruster directions (if mode is legacy)
        legacy_thruster_forces: 12-thruster forces (if mode is legacy)
    """

    mode: ActuatorMode = ActuatorMode.REACTION_WHEELS

    # Reaction wheel mode config
    reaction_wheels: Optional[ReactionWheelArrayConfig] = None
    thrusters: Optional[ThrusterSetConfig] = None

    # Legacy mode config (12 thrusters)
    legacy_thruster_positions: Optional[Dict[int, Tuple[float, float, float]]] = None
    legacy_thruster_directions: Optional[Dict[int, Tuple[float, float, float]]] = None
    legacy_thruster_forces: Optional[Dict[int, float]] = None

    def __post_init__(self):
        """Initialize mode-specific configurations."""
        if self.mode == ActuatorMode.REACTION_WHEELS:
            if self.reaction_wheels is None:
                self.reaction_wheels = get_reaction_wheel_config()
            if self.thrusters is None:
                self.thrusters = ThrusterSetConfig()
        elif self.mode == ActuatorMode.LEGACY_THRUSTERS:
            if self.legacy_thruster_positions is None:
                self._init_legacy_thrusters()

    def _init_legacy_thrusters(self):
        """Initialize legacy 12-thruster configuration."""
        self.legacy_thruster_positions = {
            1: (0.145, 0.06, 0.0),
            2: (0.145, -0.06, 0.0),
            3: (0.06, -0.145, 0.0),
            4: (-0.06, -0.145, 0.0),
            5: (-0.145, -0.06, 0.0),
            6: (-0.145, 0.06, 0.0),
            7: (-0.06, 0.145, 0.0),
            8: (0.06, 0.145, 0.0),
            9: (0.145, 0.0, 0.145),
            10: (-0.145, 0.0, 0.145),
            11: (0.0, 0.145, -0.145),
            12: (0.0, -0.145, -0.145),
        }
        self.legacy_thruster_directions = {
            1: (-1.0, 0.0, 0.0),
            2: (-1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (0.0, 1.0, 0.0),
            5: (1.0, 0.0, 0.0),
            6: (1.0, 0.0, 0.0),
            7: (0.0, -1.0, 0.0),
            8: (0.0, -1.0, 0.0),
            9: (0.0, 0.0, -1.0),
            10: (0.0, 0.0, -1.0),
            11: (0.0, 0.0, 1.0),
            12: (0.0, 0.0, 1.0),
        }
        self.legacy_thruster_forces = {i: 0.45 for i in range(1, 13)}

    @property
    def control_dimension(self) -> int:
        """Get control vector dimension based on mode."""
        if self.mode == ActuatorMode.REACTION_WHEELS:
            return 9  # 3 RW torques + 6 thruster forces
        else:
            return 12  # 12 thruster duty cycles

    @property
    def state_dimension(self) -> int:
        """Get state vector dimension based on mode."""
        if self.mode == ActuatorMode.REACTION_WHEELS:
            return 16  # 13 base + 3 wheel speeds
        else:
            return 13  # Standard 6-DOF state

    @property
    def model_path(self) -> str:
        """Get MuJoCo model path for this actuator mode."""
        if self.mode == ActuatorMode.REACTION_WHEELS:
            return "models/satellite_rw.xml"
        else:
            return "models/satellite_3d.xml"

    def get_control_labels(self) -> list:
        """Get human-readable labels for control vector elements."""
        if self.mode == ActuatorMode.REACTION_WHEELS:
            return [
                "τ_rw_x",
                "τ_rw_y",
                "τ_rw_z",  # Reaction wheel torques
                "F_px",
                "F_mx",
                "F_py",
                "F_my",
                "F_pz",
                "F_mz",  # Thruster forces
            ]
        else:
            return [f"u_{i}" for i in range(1, 13)]

    def get_state_labels(self) -> list:
        """Get human-readable labels for state vector elements."""
        base_labels = [
            "x",
            "y",
            "z",  # Position
            "qw",
            "qx",
            "qy",
            "qz",  # Quaternion
            "vx",
            "vy",
            "vz",  # Linear velocity
            "ωx",
            "ωy",
            "ωz",  # Angular velocity
        ]
        if self.mode == ActuatorMode.REACTION_WHEELS:
            base_labels.extend(["ω_rw_x", "ω_rw_y", "ω_rw_z"])
        return base_labels


def create_actuator_config(
    mode: Literal["reaction_wheels", "legacy_thrusters"] = "reaction_wheels",
) -> ActuatorConfig:
    """
    Create actuator configuration.

    Args:
        mode: "reaction_wheels" (new) or "legacy_thrusters" (old)

    Returns:
        ActuatorConfig instance
    """
    actuator_mode = ActuatorMode(mode)
    return ActuatorConfig(mode=actuator_mode)


# Default configuration
DEFAULT_ACTUATOR_MODE: ActuatorMode = ActuatorMode.REACTION_WHEELS
