#!/usr/bin/env python3
"""
Integration test for reaction wheel satellite control system.

Tests the full control loop:
1. MuJoCo physics with reaction wheel model
2. Reaction wheel MPC controller
3. Closed-loop attitude and translation control
"""

import time
from pathlib import Path

import mujoco
import numpy as np

# Add project root to path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.satellite_control.control.rw_mpc_controller import ReactionWheelMPCController
from src.satellite_control.config.actuator_config import create_actuator_config


def test_reaction_wheel_control():
    """Test closed-loop reaction wheel control."""
    print("=" * 60)
    print("REACTION WHEEL SATELLITE CONTROL TEST")
    print("=" * 60)

    # Load MuJoCo model
    model_path = ROOT / "models" / "satellite_rw.xml"
    print(f"\nLoading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Get body and joint IDs
    sat_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "satellite")
    rw_x_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rw_x_joint")
    rw_y_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rw_y_joint")
    rw_z_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rw_z_joint")

    print(f"Satellite body ID: {sat_body_id}")
    print(f"Reaction wheel joint IDs: X={rw_x_joint_id}, Y={rw_y_joint_id}, Z={rw_z_joint_id}")

    # Initialize controller
    actuator_config = create_actuator_config("reaction_wheels")
    controller = ReactionWheelMPCController(
        actuator_config=actuator_config,
        prediction_horizon=20,
        dt=0.05,  # 50ms control loop
    )

    print(f"\nController initialized:")
    print(f"  State dimension: {controller.nx}")
    print(f"  Control dimension: {controller.nu}")
    print(f"  Max RW torque: {controller.max_rw_torque} N·m")
    print(f"  Max thrust: {controller.max_thrust} N")

    # Set initial state
    data.qpos[0:3] = [0.0, 0.0, 0.0]  # Position at origin
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
    data.qvel[:] = 0.0  # Zero velocities
    mujoco.mj_forward(model, data)

    # Target state: move to x=0.5, rotate 45° about Z
    target_quat = np.array([np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)])  # 45° about Z
    x_target = np.zeros(16)
    x_target[0] = 0.5  # Target X position
    x_target[3:7] = target_quat  # Target orientation

    print(f"\nInitial position: {data.qpos[0:3]}")
    print(f"Target position: {x_target[0:3]}")
    print(f"Target quaternion: {x_target[3:7]}")

    # Simulation parameters
    sim_dt = 0.005  # 5ms physics timestep
    control_dt = 0.05  # 50ms control update
    sim_duration = 15.0  # 15 seconds for convergence

    # Logging
    log_time = []
    log_pos_x = []
    log_pos_y = []
    log_rw_speed = []
    log_solve_time = []

    # Run simulation
    print("\nRunning simulation...")
    t = 0.0
    control_step = 0
    last_control_time = -control_dt  # Force first control update

    while t < sim_duration:
        # Control update at 20Hz
        if t - last_control_time >= control_dt:
            # Build current state (16 elements)
            pos = data.qpos[0:3].copy()
            quat = data.qpos[3:7].copy()
            vel = data.qvel[0:3].copy()
            ang_vel = data.qvel[3:6].copy()

            # Get reaction wheel speeds from sensor data
            # Sensors: rw_x_speed, rw_y_speed, rw_z_speed are first 3 sensors
            rw_speeds = data.sensordata[0:3].copy()

            x_current = np.concatenate([pos, quat, vel, ang_vel, rw_speeds])

            # Compute control
            u, info = controller.get_control_action(x_current, x_target)

            # Apply reaction wheel torques via actuators
            # Actuators 0, 1, 2 are rw_x_torque, rw_y_torque, rw_z_torque
            data.ctrl[0] = u[0] * controller.max_rw_torque  # Denormalize
            data.ctrl[1] = u[1] * controller.max_rw_torque
            data.ctrl[2] = u[2] * controller.max_rw_torque

            # Apply thruster forces via xfrc_applied
            # Thrusters indexed 3-8 in control: [px, mx, py, my, pz, mz]
            thrust_dirs = np.array(
                [
                    [-1, 0, 0],  # px
                    [1, 0, 0],  # mx
                    [0, -1, 0],  # py
                    [0, 1, 0],  # my
                    [0, 0, -1],  # pz
                    [0, 0, 1],  # mz
                ]
            )

            net_force = np.zeros(3)
            for i in range(6):
                force_mag = u[3 + i] * controller.max_thrust
                net_force += force_mag * thrust_dirs[i]

            data.xfrc_applied[sat_body_id, 0:3] = net_force

            # Logging
            log_time.append(t)
            log_pos_x.append(pos[0])
            log_pos_y.append(pos[1])
            log_rw_speed.append(rw_speeds.copy())
            log_solve_time.append(info["solve_time"] * 1000)  # ms

            last_control_time = t
            control_step += 1

            # Progress update
            if control_step % 20 == 0:
                pos_err = np.linalg.norm(pos - x_target[0:3])
                # Calculate yaw angle
                yaw = 2 * np.arctan2(quat[3], quat[0])
                yaw_deg = np.degrees(yaw)
                print(
                    f"  t={t:.2f}s: pos=({pos[0]:.3f}, {pos[1]:.3f}), "
                    f"err={pos_err:.4f}m, yaw={yaw_deg:.2f}°, "
                    f"ωz={ang_vel[2]:.4f}, τ_z={u[2]:.3f}"
                )

        # Physics step
        mujoco.mj_step(model, data)
        t += sim_dt

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)

    # Final state
    final_pos = data.qpos[0:3]
    final_quat = data.qpos[3:7]
    pos_error = np.linalg.norm(final_pos - x_target[0:3])

    # Quaternion error (simple angular difference)
    quat_dot = np.abs(np.dot(final_quat, x_target[3:7]))
    ang_error_rad = 2 * np.arccos(min(quat_dot, 1.0))
    ang_error_deg = np.degrees(ang_error_rad)

    print(f"\nFinal position: ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
    print(f"Target position: ({x_target[0]:.4f}, {x_target[1]:.4f}, {x_target[2]:.4f})")
    print(f"Position error: {pos_error:.4f} m")
    print(f"Angular error: {ang_error_deg:.2f}°")
    print(f"Average solve time: {np.mean(log_solve_time):.2f} ms")
    print(f"Max solve time: {np.max(log_solve_time):.2f} ms")

    # Check targets
    pos_ok = pos_error < 0.01  # 1cm
    ang_ok = ang_error_deg < 2.0  # 2°

    print(f"\nTargets:")
    print(f"  Position ±1cm: {'✓ PASS' if pos_ok else '✗ FAIL'}")
    print(f"  Angle ±2°: {'✓ PASS' if ang_ok else '✗ FAIL'}")

    return pos_ok and ang_ok


if __name__ == "__main__":
    success = test_reaction_wheel_control()
    sys.exit(0 if success else 1)
