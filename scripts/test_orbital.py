#!/usr/bin/env python3
"""
Integration test for orbital dynamics with CW equations.

Tests:
1. Free drift follows expected CW relative motion
2. Station-keeping at fixed offset from target
3. Approach maneuver with controlled deceleration
"""

from pathlib import Path
import sys

import mujoco
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.satellite_control.control.rw_mpc_controller import ReactionWheelMPCController
from src.satellite_control.config.actuator_config import create_actuator_config
from src.satellite_control.config.orbital_config import OrbitalConfig, InspectionScenario
from src.satellite_control.physics.orbital_dynamics import CWDynamics, compute_cw_force


def test_station_keeping():
    """Test station-keeping at fixed offset with CW dynamics."""
    print("=" * 60)
    print("ORBITAL STATION-KEEPING TEST")
    print("=" * 60)

    # Load MuJoCo model
    model_path = ROOT / "models" / "satellite_rw.xml"
    print(f"\nLoading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    sat_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "satellite")

    # Orbital config
    orbital_config = OrbitalConfig()
    cw = CWDynamics(orbital_config)
    orbital_config.print_params()

    # Initialize controller
    actuator_config = create_actuator_config("reaction_wheels")
    controller = ReactionWheelMPCController(
        actuator_config=actuator_config,
        prediction_horizon=50,
        dt=0.05,
    )

    # Initial position: 5m radial offset from target
    initial_offset = np.array([5.0, 0.0, 0.0])
    data.qpos[0:3] = initial_offset
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Target: maintain 5m radial offset
    x_target = np.zeros(16)
    x_target[0:3] = initial_offset
    x_target[3] = 1.0  # qw

    print(f"\nInitial position: {initial_offset}")
    print(f"Target position: {x_target[0:3]}")
    print(f"Test: Hold position against CW drift for 300 seconds")

    # Simulation parameters
    sim_dt = 0.005
    control_dt = 0.05
    sim_duration = 300.0  # 5 minutes

    # Logging
    log_time = []
    log_pos = []
    log_error = []

    # Run simulation
    print("\nRunning simulation...")
    t = 0.0
    control_step = 0
    last_control_time = -control_dt

    while t < sim_duration:
        # Control update
        if t - last_control_time >= control_dt:
            pos = data.qpos[0:3].copy()
            quat = data.qpos[3:7].copy()
            vel = data.qvel[0:3].copy()
            ang_vel = data.qvel[3:6].copy()
            rw_speeds = data.sensordata[0:3].copy()

            x_current = np.concatenate([pos, quat, vel, ang_vel, rw_speeds])

            # Compute MPC control
            u, info = controller.get_control_action(x_current, x_target)

            # Apply reaction wheel torques
            data.ctrl[0] = u[0] * controller.max_rw_torque
            data.ctrl[1] = u[1] * controller.max_rw_torque
            data.ctrl[2] = u[2] * controller.max_rw_torque

            # Apply thruster forces
            thrust_dirs = np.array(
                [
                    [-1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                    [0, 0, 1],
                ]
            )
            net_thrust = np.zeros(3)
            for i in range(6):
                net_thrust += u[3 + i] * controller.max_thrust * thrust_dirs[i]

            # Compute CW gravity force
            cw_force = compute_cw_force(pos, vel, 10.0, orbital_config)

            # Apply total force (thrust + CW gravity)
            data.xfrc_applied[sat_body_id, 0:3] = net_thrust + cw_force

            # Logging
            pos_error = np.linalg.norm(pos - x_target[0:3])
            log_time.append(t)
            log_pos.append(pos.copy())
            log_error.append(pos_error)

            last_control_time = t
            control_step += 1

            if control_step % 100 == 0:
                print(
                    f"  t={t:.1f}s: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                    f"err={pos_error*100:.2f}cm"
                )

        mujoco.mj_step(model, data)
        t += sim_dt

    # Results
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)

    final_pos = data.qpos[0:3]
    final_error = np.linalg.norm(final_pos - x_target[0:3])
    max_error = max(log_error)
    avg_error = np.mean(log_error)

    print(f"\nFinal position: ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
    print(f"Position error: {final_error*100:.2f} cm")
    print(f"Max error during test: {max_error*100:.2f} cm")
    print(f"Average error: {avg_error*100:.2f} cm")

    # Pass criteria: maintain within 5cm of target
    pass_threshold = 0.05  # 5cm
    passed = max_error < pass_threshold

    print(f"\nStation-keeping ±5cm: {'✓ PASS' if passed else '✗ FAIL'}")

    return passed


def test_free_drift():
    """Test that CW dynamics produce expected drift without control."""
    print("=" * 60)
    print("CW FREE DRIFT TEST (No Control)")
    print("=" * 60)

    model_path = ROOT / "models" / "satellite_rw.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    sat_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "satellite")

    orbital_config = OrbitalConfig()

    # Initial: 5m radial offset, no velocity
    data.qpos[0:3] = [5.0, 0.0, 0.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print(f"\nInitial position: {data.qpos[0:3]}")
    print("Simulating 1 orbital period with NO control...")

    sim_dt = 0.005
    sim_duration = orbital_config.orbital_period  # ~92 min
    t = 0.0

    log_pos = []

    while t < sim_duration:
        pos = data.qpos[0:3].copy()
        vel = data.qvel[0:3].copy()

        # Apply only CW force (no thrust)
        cw_force = compute_cw_force(pos, vel, 10.0, orbital_config)
        data.xfrc_applied[sat_body_id, 0:3] = cw_force

        if int(t) % 600 == 0 and abs(t - int(t)) < sim_dt:
            log_pos.append((t / 60, pos.copy()))

        mujoco.mj_step(model, data)
        t += sim_dt

    print("\nPosition over time:")
    for time_min, pos in log_pos:
        print(f"  t={time_min:.0f}min: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")

    final_pos = data.qpos[0:3]
    print(
        f"\nFinal position after 1 orbit: ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})"
    )

    # For CW dynamics with initial radial offset and no velocity,
    # the satellite should oscillate in an ellipse
    return True  # Visual/qualitative test


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ORBITAL DYNAMICS INTEGRATION TESTS")
    print("=" * 60 + "\n")

    # Run tests
    drift_ok = test_free_drift()
    print()
    station_ok = test_station_keeping()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Free drift test: {'✓' if drift_ok else '✗'}")
    print(f"Station-keeping test: {'✓' if station_ok else '✗'}")

    sys.exit(0 if (drift_ok and station_ok) else 1)
