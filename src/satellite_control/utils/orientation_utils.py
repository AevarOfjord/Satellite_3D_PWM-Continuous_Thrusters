"""
Orientation Utilities

Helpers for converting between 3D Euler angles and quaternions, and
computing quaternion angle errors.
"""

from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


def euler_xyz_to_quat_wxyz(euler_xyz: Iterable[float]) -> np.ndarray:
    """
    Convert 3D Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z].
    """
    euler = np.array(list(euler_xyz), dtype=float)
    if euler.shape != (3,):
        raise ValueError(f"Expected 3-element Euler angles, got shape {euler.shape}")

    quat_xyzw = Rotation.from_euler("xyz", euler, degrees=False).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def quat_wxyz_to_euler_xyz(quat_wxyz: Iterable[float]) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3D Euler angles (roll, pitch, yaw).
    """
    quat = np.array(list(quat_wxyz), dtype=float)
    if quat.shape != (4,):
        raise ValueError(f"Expected 4-element quaternion, got shape {quat.shape}")

    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


def quat_angle_error(q_target: Iterable[float], q_current: Iterable[float]) -> float:
    """
    Compute the shortest-angle error between two quaternions (radians).
    """
    q1 = np.array(list(q_target), dtype=float)
    q2 = np.array(list(q_current), dtype=float)
    if q1.shape != (4,) or q2.shape != (4,):
        raise ValueError("Quaternion inputs must be shape (4,)")

    dot = float(np.dot(q1, q2))
    dot = float(np.clip(np.abs(dot), -1.0, 1.0))
    return 2.0 * np.arccos(dot)
