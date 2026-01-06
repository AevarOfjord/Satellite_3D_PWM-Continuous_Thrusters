"""
Thruster Configuration Dataclass

Type-safe representation of thruster physical parameters.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ThrusterConfig:
    """
    Configuration for a single thruster.

    Attributes:
        position: (x, y) position relative to satellite center [m]
        direction: (dx, dy) unit vector for thrust direction
        force: Maximum thrust force [N]
    """

    position: Tuple[float, float]
    direction: Tuple[float, float]
    force: float

    @property
    def position_array(self):
        """Return position as numpy array."""
        import numpy as np

        return np.array(self.position, dtype=np.float64)

    @property
    def direction_array(self):
        """Return direction as numpy array."""
        import numpy as np

        return np.array(self.direction, dtype=np.float64)

    def compute_torque(self, com_offset: Tuple[float, float] = (0.0, 0.0)):
        """
        Compute torque about center of mass.

        Args:
            com_offset: (x, y) offset of center of mass from geometric center

        Returns:
            Torque value (positive = CCW)
        """
        import numpy as np

        rel_pos = np.array(self.position) - np.array(com_offset)
        force_vec = self.force * np.array(self.direction)
        # 2D cross product: r x F
        return rel_pos[0] * force_vec[1] - rel_pos[1] * force_vec[0]


@dataclass
class ThrusterArrayConfig:
    """
    Configuration for all 8 thrusters.

    Provides type-safe access to thruster parameters and converts
    from legacy dictionary format.
    """

    thrusters: Dict[int, ThrusterConfig]

    @classmethod
    def from_dicts(
        cls,
        positions: Dict[int, Tuple[float, float]],
        directions: Dict[int, Tuple[float, float]],
        forces: Dict[int, float],
    ) -> "ThrusterArrayConfig":
        """
        Create from legacy dictionary format.

        Args:
            positions: {thruster_id: (x, y)} dictionary
            directions: {thruster_id: (dx, dy)} dictionary
            forces: {thruster_id: force} dictionary

        Returns:
            ThrusterArrayConfig instance
        """
        thrusters = {}
        for thruster_id in positions.keys():
            thrusters[thruster_id] = ThrusterConfig(
                position=positions[thruster_id],
                direction=directions[thruster_id],
                force=forces[thruster_id],
            )
        return cls(thrusters=thrusters)

    def __getitem__(self, thruster_id: int) -> ThrusterConfig:
        """Get thruster by ID (1-8)."""
        return self.thrusters[thruster_id]

    def __iter__(self):
        """Iterate over thruster IDs in order."""
        return iter(sorted(self.thrusters.keys()))

    def __len__(self):
        """Return number of thrusters."""
        return len(self.thrusters)
