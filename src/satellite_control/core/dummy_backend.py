"""
Dummy Simulation Backend (V3.0.0)

A no-physics backend for fast testing and development.
State updates are instantaneous (no physics integration).

Usage:
    from src.satellite_control.core.dummy_backend import DummyBackend
    
    backend = DummyBackend(dt=0.005)
    backend.set_state(initial_state)
    backend.update_physics()  # No actual physics, just advances time
"""

from typing import Optional

import numpy as np

from .backend import SimulationBackend


class DummyBackend(SimulationBackend):
    """
    Dummy physics backend with no actual physics simulation.
    
    Useful for:
    - Fast testing without physics overhead
    - Controller validation
    - Unit testing
    - Development/debugging
    
    State updates are instantaneous - no integration is performed.
    """
    
    def __init__(self, dt: float = 0.005):
        """
        Initialize dummy backend.
        
        Args:
            dt: Timestep in seconds (for compatibility)
        """
        self._dt = dt
        self._position = np.zeros(3)
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self._velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._simulation_time = 0.0
        self._thruster_levels = np.zeros(12)
    
    @property
    def dt(self) -> float:
        """Physics timestep in seconds."""
        return self._dt
    
    @property
    def position(self) -> np.ndarray:
        """Current position [x, y, z]."""
        return self._position.copy()
    
    @position.setter
    def position(self, value: np.ndarray) -> None:
        """Set position [x, y, z]."""
        self._position = np.array(value, dtype=float)
    
    @property
    def quaternion(self) -> np.ndarray:
        """Current quaternion [qw, qx, qy, qz]."""
        return self._quaternion.copy()
    
    @quaternion.setter
    def quaternion(self, value: np.ndarray) -> None:
        """Set quaternion [qw, qx, qy, qz] (normalized)."""
        q = np.array(value, dtype=float)
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            self._quaternion = q / norm
        else:
            self._quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    
    @property
    def velocity(self) -> np.ndarray:
        """Current velocity [vx, vy, vz]."""
        return self._velocity.copy()
    
    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        """Set velocity [vx, vy, vz]."""
        self._velocity = np.array(value, dtype=float)
    
    @property
    def angular_velocity(self) -> np.ndarray:
        """Current angular velocity [wx, wy, wz]."""
        return self._angular_velocity.copy()
    
    @angular_velocity.setter
    def angular_velocity(self, value: np.ndarray) -> None:
        """Set angular velocity [wx, wy, wz]."""
        self._angular_velocity = np.array(value, dtype=float)
    
    @property
    def simulation_time(self) -> float:
        """Current simulation time."""
        return self._simulation_time
    
    @simulation_time.setter
    def simulation_time(self, value: float) -> None:
        """Set simulation time."""
        self._simulation_time = float(value)
    
    def update_physics(self, dt: Optional[float] = None) -> None:
        """
        Advance simulation (no physics, just time).
        
        Args:
            dt: Timestep (uses self.dt if None)
        """
        step_dt = dt if dt is not None else self._dt
        self._simulation_time += step_dt
        # No physics integration - state remains unchanged
    
    def set_thruster_level(self, thruster_id: int, level: float) -> None:
        """
        Set thruster level (stored but not applied).
        
        Args:
            thruster_id: Thruster index (1-12, stored as 0-11)
            level: Thrust level (0.0 to 1.0)
        """
        idx = thruster_id - 1  # Convert 1-based to 0-based
        if 0 <= idx < 12:
            self._thruster_levels[idx] = max(0.0, min(1.0, level))
    
    def get_state(self) -> np.ndarray:
        """Get current state vector [13]."""
        return np.concatenate([
            self._position,
            self._quaternion,
            self._velocity,
            self._angular_velocity,
        ])
    
    def set_state(self, state: np.ndarray) -> None:
        """
        Set state vector [13].
        
        Args:
            state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        if len(state) != 13:
            raise ValueError(f"State must be 13 elements, got {len(state)}")
        
        self._position = state[0:3].copy()
        self._quaternion = state[3:7].copy()
        # Normalize quaternion
        q_norm = np.linalg.norm(self._quaternion)
        if q_norm > 1e-10:
            self._quaternion /= q_norm
        else:
            self._quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._velocity = state[7:10].copy()
        self._angular_velocity = state[10:13].copy()
    
    def reset(self) -> None:
        """Reset to zero state."""
        self._position = np.zeros(3)
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._simulation_time = 0.0
        self._thruster_levels = np.zeros(12)
