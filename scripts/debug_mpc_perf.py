import time

import numpy as np

from src.satellite_control.config.satellite_config import SatelliteConfig
from src.satellite_control.control.pwm_mpc import PWMMPC


def test_mpc_perf():
    # Setup
    start_pos = (1.0, 1.0)
    start_angle = 0.0
    target_pos = (0.0, 0.0)
    target_angle = 0.0

    # Init Controller
    satellite_params = SatelliteConfig.get_app_config().physics
    mpc_params = SatelliteConfig.get_app_config().mpc
    controller = PWMMPC(satellite_params, mpc_params)

    # Current State
    x_current = np.array([start_pos[0], start_pos[1], 0.0, 0.0, start_angle, 0.0])
    x_target = np.array([target_pos[0], target_pos[1], 0.0, 0.0, target_angle, 0.0])

    # Warm up
    print("Warming up...")
    controller.get_control_action(x_current, x_target)

    # Create Constant Trajectory (simulating point-to-point behavior in simulation.py)
    horizon = controller.N
    # Shape: (N+1, 6)
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
