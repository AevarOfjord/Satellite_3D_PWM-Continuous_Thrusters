"""
Simulation Context Module

Defines the SimulationContext data class which holds the instantaneous state
of the simulation. This facilitates passing state between components without
long argument lists.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SimulationContext:
    """
    Holds the state of the simulation at a given timestep.
    """

    # Time
    simulation_time: float = 0.0
    dt: float = 0.005  # Physics timestep
    control_dt: float = 0.1  # Control update interval
    step_number: int = 0
    # State [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    current_state: np.ndarray = field(default_factory=lambda: np.zeros(13))
    target_state: np.ndarray = field(default_factory=lambda: np.zeros(13))
    # Mission
    mission_phase: str = "IDLE"
    waypoint_number: int = 0
    # Control
    last_control_update_time: float = 0.0
    active_thrusters: List[int] = field(default_factory=list)
    previous_thruster_command: Optional[np.ndarray] = None
    # Performance / Internals
    computation_time_last_step: float = 0.0

    def update_state(self, time: float, state: np.ndarray, target: np.ndarray):
        """Update dynamic state variables."""
        self.simulation_time = time
        self.current_state = state
        self.target_state = target
