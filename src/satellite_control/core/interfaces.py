"""
Protocol Interfaces for Satellite Control System

Defines abstract interfaces for dependency injection and type safety.
Uses Python's Protocol (structural subtyping) for flexibility.

Key protocols:
- MotionController: MPC controller interface
- PhysicsSimulator: Physics engine interface
- DataLogger: Logging interface
- MissionManager: Mission logic interface
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class MotionController(Protocol):
    """
    Protocol for motion control systems (MPC controllers).

    Any class implementing this protocol can be used as the control system
    for satellite navigation.
    """

    @property
    def N(self) -> int:
        """Prediction horizon length."""
        ...

    @property
    def dt(self) -> float:
        """Control timestep in seconds."""
        ...

    @property
    def max_velocity(self) -> float:
        """Maximum allowed linear velocity."""
        ...

    @property
    def max_angular_velocity(self) -> float:
        """Maximum allowed angular velocity."""
        ...

    def get_control_action(
        self,
        x_current: NDArray[np.floating],
        x_target: NDArray[np.floating],
        previous_thrusters: Optional[NDArray[np.floating]] = None,
        x_target_trajectory: Optional[NDArray[np.floating]] = None,
    ) -> Tuple[NDArray[np.floating], Dict]:
        """
        Compute optimal control action.

        Args:
            x_current: Current state [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            x_target: Target state [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            previous_thrusters: Previous control action for switching cost
            x_target_trajectory: Optional trajectory of targets (N+1, 13)

        Returns:
            Tuple of (control_action, info_dict)
        """
        ...

    def update_velocity_limits(
        self,
        max_velocity: float,
        max_angular_velocity: float,
    ) -> None:
        """
        Update velocity constraint limits (e.g., for docking mode).

        Args:
            max_velocity: New maximum linear velocity
            max_angular_velocity: New maximum angular velocity
        """
        ...


@runtime_checkable
class PhysicsSimulator(Protocol):
    """
    Protocol for physics simulation backends.

    Supports both MuJoCo-based simulation and any compatible physics engine.
    """

    @property
    def position(self) -> NDArray[np.floating]:
        """Current position [x, y] in meters."""
        ...

    @position.setter
    def position(self, value: NDArray[np.floating]) -> None: ...

    @property
    def velocity(self) -> NDArray[np.floating]:
        """Current velocity [vx, vy] in m/s."""
        ...

    @velocity.setter
    def velocity(self, value: NDArray[np.floating]) -> None: ...

    @property
    def angle(self) -> NDArray[np.floating]:
        """Current orientation (implementation-defined)."""
        ...

    @angle.setter
    def angle(self, value: Union[NDArray[np.floating], Tuple[float, float, float]]) -> None: ...

    @property
    def angular_velocity(self) -> NDArray[np.floating]:
        """Current angular velocity (omega vector [x, y, z]) or scalar."""
        ...

    @angular_velocity.setter
    def angular_velocity(
        self, value: Union[float, NDArray[np.floating], Tuple[float, ...]]
    ) -> None: ...

    @property
    def dt(self) -> float:
        """Physics timestep in seconds."""
        ...

    def update_physics(self, dt: Optional[float] = None) -> None:
        """
        Advance physics simulation by one timestep.

        Args:
            dt: Optional timestep override
        """
        ...

    def set_thruster_level(self, thruster_id: int, level: float) -> None:
        """
        Set thrust level for a specific thruster.

        Args:
            thruster_id: Thruster index (1-12)
            level: Thrust level (0.0 to 1.0)
        """
        ...

    def calculate_forces_and_torques(
        self,
    ) -> Tuple[NDArray[np.floating], float]:
        """
        Calculate net force and torque from active thrusters.

        Returns:
            Tuple of (net_force [fx, fy, fz], net_torque [tx, ty, tz])
        """
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...


@runtime_checkable
class DataLogger(Protocol):
    """
    Protocol for data logging systems.

    Provides interface for various logging backends (CSV, memory, database).
    """

    @property
    def current_step(self) -> int:
        """Current step/row number."""
        ...

    def log(self, data: Dict) -> None:
        """
        Log a data record.

        Args:
            data: Dictionary of field names to values
        """
        ...

    def log_terminal_message(self, entry: Dict) -> None:
        """
        Log a terminal/status message.

        Args:
            entry: Dictionary with message data
        """
        ...

    def get_log_count(self) -> int:
        """Get total number of logged records."""
        ...

    def flush(self) -> None:
        """Flush buffered data to storage."""
        ...

    def close(self) -> None:
        """Close logger and release resources."""
        ...


@runtime_checkable
class MissionLogic(Protocol):
    """
    Protocol for mission state management.

    Handles target updates, phase transitions, and trajectory generation.
    """

    @property
    def dxf_completed(self) -> bool:
        """Whether DXF/shape-following mission is complete."""
        ...

    def update_target_state(
        self,
        current_position: NDArray[np.floating],
        current_quat: NDArray[np.floating],
        current_time: float,
        current_state: NDArray[np.floating],
    ) -> Optional[NDArray[np.floating]]:
        """
        Update target state based on current position and mission logic.

        Args:
            current_position: Current [x, y, z] position
            current_quat: Current orientation quaternion [w, x, y, z]
            current_time: Current simulation time
            current_state: Full state vector [pos(3), quat(4), vel(3), w(3)]

        Returns:
            New target state or None if mission complete
        """
        ...

    def get_trajectory(
        self,
        current_time: float,
        dt: float,
        horizon: int,
        current_state: NDArray[np.floating],
        external_target_state: Optional[NDArray[np.floating]] = None,
    ) -> Optional[NDArray[np.floating]]:
        """
        Generate target trajectory for MPC.

        Args:
            current_time: Current simulation time
            dt: Trajectory timestep
            horizon: Number of trajectory points
            current_state: Current state vector
            external_target_state: Optional override target

        Returns:
            Trajectory array (horizon+1, 13) or None
        """
        ...


@runtime_checkable
class StateValidator(Protocol):
    """
    Protocol for state validation and noise application.
    """

    def check_target_reached(
        self,
        current_state: NDArray[np.floating],
        target_state: NDArray[np.floating],
    ) -> bool:
        """
        Check if satellite has reached target within tolerances.

        Args:
            current_state: Current state vector
            target_state: Target state vector

        Returns:
            True if target reached
        """
        ...

    def apply_sensor_noise(
        self,
        true_state: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Apply realistic sensor noise to state measurements.

        Args:
            true_state: True state vector

        Returns:
            Noisy state vector
        """
        ...

    def validate_state(
        self,
        state: NDArray[np.floating],
    ) -> Tuple[bool, List[str]]:
        """
        Validate state vector for physical plausibility.

        Args:
            state: State vector to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        ...
