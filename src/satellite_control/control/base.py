"""
Controller Base Interface (V3.0.0)

Defines the abstract base class for all controllers in the satellite control system.
This enables pluggable controller implementations (MPC, PID, LQR, etc.).

Usage:
    from src.satellite_control.control.base import Controller
    from src.satellite_control.control.mpc_controller import MPCController
    
    # MPCController implements Controller
    controller: Controller = MPCController(...)
    
    # Use controller
    action, info = controller.get_control_action(current_state, target_state)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class Controller(ABC):
    """
    Abstract base class for satellite controllers.
    
    All controllers must implement this interface to be compatible
    with the simulation system. This enables pluggable controller
    architectures.
    
    State format: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13 elements)
    Control format: [u1, ..., u12] (12 thrusters)
    """
    
    @property
    @abstractmethod
    def dt(self) -> float:
        """
        Control update interval in seconds.
        
        Returns:
            Control timestep in seconds
        """
        pass
    
    @property
    @abstractmethod
    def prediction_horizon(self) -> Optional[int]:
        """
        Prediction horizon for predictive controllers (None for non-predictive).
        
        Returns:
            Number of steps in prediction horizon, or None if not applicable
        """
        pass
    
    @abstractmethod
    def get_control_action(
        self,
        x_current: np.ndarray,
        x_target: np.ndarray,
        previous_thrusters: Optional[np.ndarray] = None,
        x_target_trajectory: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute optimal control action.
        
        Args:
            x_current: Current state vector [13] (position, quaternion, velocity, angular velocity)
            x_target: Target state vector [13]
            previous_thrusters: Previous thruster commands [12] (optional, for smoothness)
            x_target_trajectory: Target trajectory [N, 13] (optional, for predictive controllers)
        
        Returns:
            Tuple of:
            - thruster_action: Control action [12] (thruster forces)
            - info: Dictionary with solver metadata (status, iterations, cost, etc.)
        
        Raises:
            OptimizationError: If optimization fails
            SolverTimeoutError: If solver exceeds time limit
        """
        pass
    
    def reset(self) -> None:
        """
        Reset controller internal state.
        
        Called when starting a new mission or resetting the simulation.
        Default implementation does nothing (override if needed).
        """
        pass
    
    def get_solver_stats(self) -> Dict[str, Any]:
        """
        Get solver performance statistics.
        
        Returns:
            Dictionary with statistics like:
            - solve_times: List of solve times
            - average_solve_time: Average solve time
            - max_solve_time: Maximum solve time
            - solve_count: Number of solves
        """
        return {}
    
    def validate_state(self, state: np.ndarray) -> bool:
        """
        Validate state vector format.
        
        Args:
            state: State vector to validate
        
        Returns:
            True if state is valid, False otherwise
        """
        if state is None:
            return False
        if not isinstance(state, np.ndarray):
            return False
        if state.shape != (13,):
            return False
        # Check quaternion normalization (approximately)
        quat = state[3:7]
        quat_norm = np.linalg.norm(quat)
        if abs(quat_norm - 1.0) > 0.1:
            return False
        return True
    
    def validate_target(self, target: np.ndarray) -> bool:
        """
        Validate target state vector format.
        
        Args:
            target: Target state vector to validate
        
        Returns:
            True if target is valid, False otherwise
        """
        return self.validate_state(target)
