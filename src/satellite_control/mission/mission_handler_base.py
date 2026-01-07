"""
Mission Handler Base Class

Abstract base class for mission mode handlers.
Provides interface for waypoint navigation and shape following handlers.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np


class MissionHandler(ABC):
    """
    Abstract base class for mission mode handlers.

    Each mission mode (waypoint, DXF shape) implements this interface
    to provide consistent target calculation and completion checking.
    """

    def __init__(
        self,
        position_tolerance: float = 0.05,
        angle_tolerance: float = 0.05,
        normalize_angle_func: Optional[Callable[[float], float]] = None,
        angle_difference_func: Optional[Callable[[float, float], float]] = None,
    ):
        """
        Initialize mission handler.

        Args:
            position_tolerance: Position error tolerance in meters
            angle_tolerance: Angle error tolerance in radians
            normalize_angle_func: Function to normalize angles to [-pi, pi]
            angle_difference_func: Function to calculate angle difference
        """
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        self.normalize_angle = normalize_angle_func or self._default_normalize_angle
        self.angle_difference = angle_difference_func or self._default_angle_difference

    @abstractmethod
    def update_target(
        self,
        current_position: np.ndarray,
        current_quat: np.ndarray,
        current_time: float,
    ) -> Optional[np.ndarray]:
        """
        Calculate target state based on current position and time.

        Args:
            current_position: Current satellite position [x, y, z]
            current_quat: Current satellite orientation quaternion [w, x, y, z]
            current_time: Current simulation time in seconds

        Returns:
            Target state vector [pos(3), quat(4), vel(3), w(3)] or None
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """
        Check if the mission is complete.

        Returns:
            True if mission is complete, False otherwise
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset handler state for a new mission."""
        pass

    @property
    @abstractmethod
    def phase(self) -> str:
        """Get current mission phase name."""
        pass

    @staticmethod
    def _default_normalize_angle(angle: float) -> float:
        """Default angle normalization to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def _default_angle_difference(angle1: float, angle2: float) -> float:
        """Default angle difference calculation."""
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
