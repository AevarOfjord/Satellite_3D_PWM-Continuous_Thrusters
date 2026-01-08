import time

import numpy as np

from src.satellite_control.config.satellite_config import SatelliteConfig
from src.satellite_control.control import MPCController
from scipy.spatial.transform import Rotation


def test_mpc_perf():
    # Setup - use 3D tuples
    start_pos = (1.0, 1.0, 0.0)
    start_euler = (0.0, 0.0, 0.0)
    target_pos = (0.0, 0.0, 0.0)
    target_euler = (0.0, 0.0, 0.0)

    # Init Controller
    satellite_params = SatelliteConfig.get_app_config().physics
    mpc_params = SatelliteConfig.get_app_config().mpc
    controller = MPCController(satellite_params, mpc_params)

    # Current State (13 elements): [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    # Convert Euler to Quat (w, x, y, z)
    q_start = Rotation.from_euler("xyz", start_euler, degrees=False).as_quat()
    # Scipy returns [x, y, z, w], we want [w, x, y, z]
    q_start_mj = [q_start[3], q_start[0], q_start[1], q_start[2]]

    q_target = Rotation.from_euler("xyz", target_euler, degrees=False).as_quat()
    q_target_mj = [q_target[3], q_target[0], q_target[1], q_target[2]]

    x_current = np.array(
        [
            start_pos[0],
            start_pos[1],
            start_pos[2],
            q_start_mj[0],
            q_start_mj[1],
            q_start_mj[2],
            q_start_mj[3],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    x_target = np.array(
        [
            target_pos[0],
            target_pos[1],
            target_pos[2],
            q_target_mj[0],
            q_target_mj[1],
            q_target_mj[2],
            q_target_mj[3],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # Warm up
    print("Warming up...")
    controller.get_control_action(x_current, x_target)

    # Create Constant Trajectory (simulating point-to-point behavior in simulation.py)
    horizon = controller.N
    # Shape: (N+1, 13)
    trajectory = np.tile(x_target, (horizon + 1, 1))

    print(f"Testing with Trajectory (Shape: {trajectory.shape})...")

    # Loop
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        # Pass trajectory
        controller.get_control_action(x_current, x_target, x_target_trajectory=trajectory)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"Step {i}: {dt*1000:.3f} ms")

    avg = np.mean(times)
    print(f"Average Solve Time: {avg*1000:.3f} ms")


if __name__ == "__main__":
    test_mpc_perf()
