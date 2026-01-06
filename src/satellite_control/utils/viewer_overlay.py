"""
MuJoCo Viewer Overlay Module

Provides real-time telemetry overlay for the MuJoCo simulation viewer.
Renders text overlays showing mission status, position error, and thruster activity.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class OverlayData:
    """Data structure for overlay rendering."""

    # Mission info
    phase: str = "Initializing"
    simulation_time: float = 0.0

    # Position tracking
    current_pos: Tuple[float, float] = (0.0, 0.0)
    target_pos: Tuple[float, float] = (0.0, 0.0)
    position_error: float = 0.0

    # Angle tracking
    current_angle: float = 0.0
    target_angle: float = 0.0
    angle_error: float = 0.0

    # MPC info
    mpc_solve_time_ms: float = 0.0
    mpc_status: str = "OK"

    # Thruster activity (8 thrusters, 0.0-1.0 each)
    thruster_levels: Tuple[float, ...] = (0.0,) * 8


class ViewerOverlay:
    """
    Manages overlay rendering for MuJoCo viewer.

    Provides formatted text strings for telemetry display.
    """

    def __init__(self):
        self.data = OverlayData()
        self._warning_threshold_ms = 30.0  # MPC warning threshold

    def update(
        self,
        phase: str,
        sim_time: float,
        current_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        current_angle: float,
        target_angle: float,
        mpc_solve_time: float,
        mpc_status: str,
        thruster_levels: Tuple[float, ...],
    ) -> None:
        """Update overlay data from simulation state."""
        self.data.phase = phase
        self.data.simulation_time = sim_time
        self.data.current_pos = current_pos
        self.data.target_pos = target_pos
        self.data.current_angle = current_angle
        self.data.target_angle = target_angle
        self.data.mpc_solve_time_ms = mpc_solve_time * 1000
        self.data.mpc_status = mpc_status
        self.data.thruster_levels = thruster_levels

        # Calculate errors
        self.data.position_error = np.sqrt(
            (current_pos[0] - target_pos[0]) ** 2 + (current_pos[1] - target_pos[1]) ** 2
        )
        self.data.angle_error = abs(self._normalize_angle(current_angle - target_angle))

    def get_status_text(self) -> str:
        """Get formatted status text for top-left overlay."""
        lines = [
            f"Phase: {self.data.phase}",
            f"Time: {self.data.simulation_time:.1f}s",
            "",
            f"Pos Error: {self.data.position_error*1000:.1f}mm",
            f"Ang Error: {np.degrees(self.data.angle_error):.1f}°",
        ]
        return "\n".join(lines)

    def get_mpc_text(self) -> str:
        """Get formatted MPC text for top-right overlay."""
        # Add warning indicator if solve time is high
        warning = ""
        if self.data.mpc_solve_time_ms > self._warning_threshold_ms:
            warning = " ⚠️"

        lines = [
            f"MPC: {self.data.mpc_status}{warning}",
            f"Solve: {self.data.mpc_solve_time_ms:.1f}ms",
        ]
        return "\n".join(lines)

    def get_thruster_bar(self) -> str:
        """Get ASCII thruster activity bar."""
        chars = []
        for level in self.data.thruster_levels:
            if level > 0.8:
                chars.append("█")  # Full
            elif level > 0.5:
                chars.append("▓")  # High
            elif level > 0.2:
                chars.append("▒")  # Medium
            elif level > 0.01:
                chars.append("░")  # Low
            else:
                chars.append("·")  # Off

        return f"T: [{' '.join(chars)}]"

    def get_position_text(self) -> str:
        """Get formatted position text."""
        cx, cy = self.data.current_pos
        tx, ty = self.data.target_pos

        lines = [
            f"Pos: ({cx*1000:.0f}, {cy*1000:.0f})mm",
            f"Tgt: ({tx*1000:.0f}, {ty*1000:.0f})mm",
        ]
        return "\n".join(lines)

    def get_full_overlay_text(self) -> str:
        """Get complete overlay text for single text block."""
        lines = [
            "━" * 28,
            f" 🛰️  {self.data.phase}",
            "━" * 28,
            f" Time: {self.data.simulation_time:.1f}s",
            f" Pos Err: {self.data.position_error*1000:.1f}mm",
            f" Ang Err: {np.degrees(self.data.angle_error):.1f}°",
            "",
            f" MPC: {self.data.mpc_solve_time_ms:.1f}ms ({self.data.mpc_status})",
            f" {self.get_thruster_bar()}",
            "━" * 28,
        ]
        return "\n".join(lines)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def create_overlay() -> ViewerOverlay:
    """Factory function to create overlay instance."""
    return ViewerOverlay()
