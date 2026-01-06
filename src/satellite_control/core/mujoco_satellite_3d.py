"""
3D MuJoCo Satellite Physics Simulation

6-DOF satellite simulation with 12 thrusters for full attitude and position control.
Uses quaternion representation for orientation.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import mujoco
import numpy as np
from mujoco import viewer as mujoco_viewer

from src.satellite_control.config import SatelliteConfig
from src.satellite_control.config.physics import USE_3D_MODE


class MuJoCoSatelliteSimulator3D:
    """
    3D MuJoCo-based satellite physics simulator.

    Provides full 6-DOF control with 12 thrusters and quaternion-based orientation.
    """

    def __init__(
        self,
        model_path: str = "models/satellite_3d.xml",
        use_mujoco_viewer: bool = True,
    ):
        """Initialize 3D MuJoCo simulation."""
        if not USE_3D_MODE:
            raise RuntimeError("MuJoCoSatelliteSimulator3D requires USE_3D_MODE=True in physics.py")

        model_full_path = Path(model_path)
        if not model_full_path.exists():
            raise FileNotFoundError(f"MuJoCo 3D model not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_full_path))
        self.data = mujoco.MjData(self.model)

        # Visualization mode
        self.use_mujoco_viewer = use_mujoco_viewer
        self.viewer = None

        # Configuration from SatelliteConfig
        self.satellite_size = SatelliteConfig.SATELLITE_SIZE
        self.total_mass = SatelliteConfig.TOTAL_MASS
        self.inertia_tensor = SatelliteConfig.MOMENT_OF_INERTIA_TENSOR
        self.com_offset = np.array(SatelliteConfig.COM_OFFSET)

        # Thruster configuration (12 thrusters for 3D)
        self.num_thrusters = 12
        self.thruster_positions = SatelliteConfig.THRUSTER_POSITIONS
        self.thruster_directions = SatelliteConfig.THRUSTER_DIRECTIONS
        self.thruster_forces = SatelliteConfig.THRUSTER_FORCES.copy()

        # Active thrusters tracking
        self.active_thrusters: Set[int] = set()
        self.thruster_levels: Dict[int, float] = {}

        # Simulation timing
        self.dt = SatelliteConfig.SIMULATION_DT
        self.model.opt.timestep = self.dt
        self.simulation_time = 0.0

        # Trajectory tracking
        self.trajectory: List[np.ndarray] = []
        self.max_trajectory_points = 500

        # Get body ID
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "satellite")

        # Initialize state
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Setup visualization
        if self.use_mujoco_viewer:
            self._setup_viewer()

        print("3D MuJoCo satellite simulator initialized")
        print(f"  Model: {model_path}")
        print(f"  Timestep: {self.model.opt.timestep:.4f}s")
        print(f"  Mass: {self.total_mass:.2f} kg")
        print(f"  Thrusters: {self.num_thrusters}")

    def _setup_viewer(self):
        """Setup MuJoCo viewer for 3D visualization."""
        try:
            os.environ.setdefault("MUJOCO_GL", "glfw")
            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)

            if self.viewer is not None:
                # Isometric view for 3D visualization
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
                self.viewer.cam.distance = 5.0
                self.viewer.cam.elevation = -30
                self.viewer.cam.azimuth = 45

            print("  MuJoCo 3D viewer launched")

        except Exception as e:
            print(f"  Warning: Could not launch viewer: {e}")
            self.use_mujoco_viewer = False
            self.viewer = None

    @property
    def position(self) -> np.ndarray:
        """Get current position [x, y, z]."""
        return self.data.qpos[0:3].copy()

    @position.setter
    def position(self, value: np.ndarray):
        """Set position [x, y, z]."""
        self.data.qpos[0:3] = value
        mujoco.mj_forward(self.model, self.data)

    @property
    def quaternion(self) -> np.ndarray:
        """Get current orientation quaternion [w, x, y, z]."""
        return self.data.qpos[3:7].copy()

    @quaternion.setter
    def quaternion(self, value: np.ndarray):
        """Set orientation quaternion [w, x, y, z]."""
        # Normalize
        value = value / np.linalg.norm(value)
        self.data.qpos[3:7] = value
        mujoco.mj_forward(self.model, self.data)

    @property
    def velocity(self) -> np.ndarray:
        """Get current linear velocity [vx, vy, vz]."""
        return self.data.qvel[0:3].copy()

    @velocity.setter
    def velocity(self, value: np.ndarray):
        """Set linear velocity [vx, vy, vz]."""
        self.data.qvel[0:3] = value
        mujoco.mj_forward(self.model, self.data)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Get current angular velocity [ωx, ωy, ωz]."""
        return self.data.qvel[3:6].copy()

    @angular_velocity.setter
    def angular_velocity(self, value: np.ndarray):
        """Set angular velocity [ωx, ωy, ωz]."""
        self.data.qvel[3:6] = value
        mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> np.ndarray:
        """
        Get full 13-element state vector.

        Returns:
            [x, y, z, qw, qx, qy, qz, vx, vy, vz, ωx, ωy, ωz]
        """
        return np.concatenate(
            [
                self.data.qpos[:7],  # position + quaternion
                self.data.qvel[:6],  # velocity + angular velocity
            ]
        )

    def set_state(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ):
        """Set satellite state directly."""
        self.data.qpos[0:3] = position
        self.data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
        self.data.qvel[0:3] = velocity
        self.data.qvel[3:6] = angular_velocity
        mujoco.mj_forward(self.model, self.data)

    def set_thruster_level(self, thruster_id: int, level: float):
        """Set thrust level (0.0 to 1.0) for a thruster."""
        self.thruster_levels[thruster_id] = max(0.0, min(1.0, level))

        if level > 0.01:
            self.active_thrusters.add(thruster_id)
        else:
            self.active_thrusters.discard(thruster_id)

    def set_all_thrusters(self, levels: np.ndarray):
        """Set all 12 thruster levels at once."""
        for i, level in enumerate(levels):
            self.set_thruster_level(i + 1, level)

    def _quat_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def update_physics(self, dt: Optional[float] = None):
        """Update satellite physics for one timestep."""
        if dt is None:
            dt = self.dt

        # Zero out applied forces
        self.data.xfrc_applied[self.body_id, :] = 0

        # Get rotation matrix from body to world
        R = self._quat_to_rotation_matrix(self.quaternion)

        # Apply thruster forces
        for thruster_id in range(1, self.num_thrusters + 1):
            level = self.thruster_levels.get(thruster_id, 0.0)
            if level < 0.01:
                continue

            # Get force in body frame
            direction = np.array(self.thruster_directions[thruster_id])
            force_magnitude = self.thruster_forces[thruster_id] * level
            force_body = force_magnitude * direction

            # Transform to world frame
            force_world = R @ force_body

            # Apply force
            self.data.xfrc_applied[self.body_id, 0:3] += force_world

            # Calculate and apply torque
            pos_body = np.array(self.thruster_positions[thruster_id]) - self.com_offset
            torque_body = np.cross(pos_body, force_body)
            torque_world = R @ torque_body
            self.data.xfrc_applied[self.body_id, 3:6] += torque_world

        # Step MuJoCo simulation
        num_steps = max(1, int(dt / self.model.opt.timestep))
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)

        self.simulation_time += dt

        # Update trajectory
        self.trajectory.append(self.position.copy())
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)

        # Sync viewer
        if self.use_mujoco_viewer and self.viewer is not None:
            self.viewer.sync()

    def reset_state(self):
        """Reset satellite to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.active_thrusters.clear()
        self.thruster_levels.clear()
        self.simulation_time = 0.0
        self.trajectory.clear()
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        """Close the MuJoCo viewer."""
        if self.viewer is not None:
            try:
                self.viewer.close()
                print("MuJoCo 3D viewer closed")
            except Exception:
                pass
            self.viewer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
