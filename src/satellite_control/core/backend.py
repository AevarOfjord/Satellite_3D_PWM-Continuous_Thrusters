"""
Simulation Backend Interface (V3.0.0)

Defines the abstract base class for physics simulation backends.
This enables pluggable physics engines (MuJoCo, Dummy, etc.).

Usage:
    from src.satellite_control.core.backend import SimulationBackend
    from src.satellite_control.core.mujoco_backend import MujocoBackend
    
    # Use backend
    backend: SimulationBackend = MujocoBackend(...)
    backend.update_physics(dt=0.005)
    state = backend.get_state()
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class SimulationBackend(ABC):
    """
    Abstract base class for physics simulation backends.
    
    All physics backends must implement this interface to be compatible
    with the simulation system. This enables pluggable physics engines.
    
    State format: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13 elements)
    """
    
    @property
    @abstractmethod
    def dt(self) -> float:
        """
        Physics timestep in seconds.
        
        Returns:
            Timestep in seconds
        """
        pass
    
    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        """
        Current position [x, y, z] in meters.
        
        Returns:
            3D position vector
        """
        pass
    
    @position.setter
    @abstractmethod
    def position(self, value: np.ndarray) -> None:
        """
        Set position [x, y, z] in meters.
        
        Args:
            value: 3D position vector
        """
        pass
    
    @property
    @abstractmethod
    def quaternion(self) -> np.ndarray:
        """
        Current orientation as quaternion [qw, qx, qy, qz].
        
        Returns:
            Quaternion vector (normalized)
        """
        pass
    
    @quaternion.setter
    @abstractmethod
    def quaternion(self, value: np.ndarray) -> None:
        """
        Set orientation as quaternion [qw, qx, qy, qz].
        
        Args:
            value: Quaternion vector (will be normalized)
        """
        pass
    
    @property
    @abstractmethod
    def velocity(self) -> np.ndarray:
        """
        Current linear velocity [vx, vy, vz] in m/s.
        
        Returns:
            3D velocity vector
        """
        pass
    
    @velocity.setter
    @abstractmethod
    def velocity(self, value: np.ndarray) -> None:
        """
        Set linear velocity [vx, vy, vz] in m/s.
        
        Args:
            value: 3D velocity vector
        """
        pass
    
    @property
    @abstractmethod
    def angular_velocity(self) -> np.ndarray:
        """
        Current angular velocity [wx, wy, wz] in rad/s.
        
        Returns:
            3D angular velocity vector
        """
        pass
    
    @angular_velocity.setter
    @abstractmethod
    def angular_velocity(self, value: np.ndarray) -> None:
        """
        Set angular velocity [wx, wy, wz] in rad/s.
        
        Args:
            value: 3D angular velocity vector
        """
        pass
    
    @property
    @abstractmethod
    def simulation_time(self) -> float:
        """
        Current simulation time in seconds.
        
        Returns:
            Simulation time
        """
        pass
    
    @simulation_time.setter
    @abstractmethod
    def simulation_time(self, value: float) -> None:
        """
        Set simulation time in seconds.
        
        Args:
            value: Simulation time
        """
        pass
    
    @abstractmethod
    def update_physics(self, dt: Optional[float] = None) -> None:
        """
        Advance physics simulation by one timestep.
        
        Args:
            dt: Optional timestep override (uses self.dt if None)
        """
        pass
    
    @abstractmethod
    def set_thruster_level(self, thruster_id: int, level: float) -> None:
        """
        Set thrust level for a specific thruster.
        
        Args:
            thruster_id: Thruster index (1-12)
            level: Thrust level (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Get current state vector.
        
        Returns:
            State vector [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        pass
    
    @abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        """
        Set state vector.
        
        Args:
            state: State vector [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        pass
    
    def reset(self) -> None:
        """
        Reset simulation to initial state.
        
        Default implementation sets state to zeros.
        Override for custom reset behavior.
        """
        zero_state = np.zeros(13)
        zero_state[3] = 1.0  # Quaternion w = 1 (no rotation)
        self.set_state(zero_state)
        self.simulation_time = 0.0
    
    def close(self) -> None:
        """
        Cleanup resources.
        
        Called when simulation is finished.
        Default implementation does nothing (override if needed).
        """
        pass
    
    def calculate_forces_and_torques(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate net forces and torques from active thrusters.
        
        Returns:
            Tuple of (force_vector [3], torque_vector [3])
        """
        # Default implementation returns zeros
        # Override in concrete backends
        return np.zeros(3), np.zeros(3)
