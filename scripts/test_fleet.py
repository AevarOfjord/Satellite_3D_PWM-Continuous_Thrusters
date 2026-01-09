#!/usr/bin/env python3
"""
Multi-satellite fleet coordination test.

Tests:
1. Formation keeping - 3 inspectors maintain 120° spacing
2. Collision avoidance - no inter-inspector collisions
3. Keep-out zone - all inspectors stay >2m from target
"""

from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.satellite_control.control.rw_mpc_controller import ReactionWheelMPCController
from src.satellite_control.config.actuator_config import create_actuator_config
from src.satellite_control.fleet.fleet_manager import FleetManager, create_fleet_manager
from src.satellite_control.config.orbital_config import OrbitalConfig
from src.satellite_control.physics.orbital_dynamics import compute_cw_force


def test_fleet_formation():
    """Test fleet formation keeping with 3 inspectors."""
    print("=" * 60)
    print("FLEET FORMATION KEEPING TEST")
    print("=" * 60)

    # Load fleet model
    model_path = ROOT / "models" / "satellite_fleet.xml"
    print(f"\nLoading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Get body IDs
    body_ids = {
        1: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "inspector_1"),
        2: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "inspector_2"),
        3: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "inspector_3"),
    }
    print(f"Inspector body IDs: {body_ids}")

    # Create fleet manager
    fleet = create_fleet_manager(num_inspectors=3, formation_radius=5.0)
    print(f"\nFormation targets:")
    for i in range(3):
        target = fleet.get_formation_target(i)
        print(f"  Inspector {i}: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")

    # Create controllers (one per inspector)
    actuator_config = create_actuator_config("reaction_wheels")
    controllers = {
        i: ReactionWheelMPCController(
            actuator_config=actuator_config,
            prediction_horizon=30,
            dt=0.05,
        )
        for i in range(3)
    }

    # Orbital config for CW forces
    orbital_config = OrbitalConfig()

    # Simulation parameters
    sim_dt = 0.005
    control_dt = 0.05
    sim_duration = 60.0  # 1 minute

    # Logging
    log_separations = []
    log_to_target = []
    log_errors = {0: [], 1: [], 2: []}

    print(f"\nSimulating {sim_duration}s of formation keeping...")

    t = 0.0
    control_step = 0
    last_control_time = -control_dt

    while t < sim_duration:
        if t - last_control_time >= control_dt:
            # Process each inspector
            for insp_id in range(3):
                # Sensor layout per inspector (16 values each):
                # RW speeds: 0-2, pos: 3-5, quat: 6-9, vel: 10-12, angvel: 13-15
                base = insp_id * 16

                rw_speeds = data.sensordata[base : base + 3].copy()
                pos = data.sensordata[base + 3 : base + 6].copy()
                quat = data.sensordata[base + 6 : base + 10].copy()
                vel = data.sensordata[base + 10 : base + 13].copy()
                angvel = data.sensordata[base + 13 : base + 16].copy()

                # Update fleet manager
                fleet.update_inspector(insp_id, pos, vel, quat, angvel, rw_speeds)

                # Get target state
                target_pos = fleet.get_formation_target(insp_id)
                x_target = np.zeros(16)
                x_target[0:3] = target_pos
                x_target[3] = 1.0  # Identity quaternion

                # Build current state
                x_current = np.concatenate([pos, quat, vel, angvel, rw_speeds])

                # Compute control
                u, info = controllers[insp_id].get_control_action(x_current, x_target)

                # Apply RW torques
                ctrl_offset = insp_id * 3
                data.ctrl[ctrl_offset : ctrl_offset + 3] = (
                    u[:3] * controllers[insp_id].max_rw_torque
                )

                # Apply thruster forces + CW gravity
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
                for j in range(6):
                    net_thrust += u[3 + j] * controllers[insp_id].max_thrust * thrust_dirs[j]

                cw_force = compute_cw_force(pos, vel, 10.0, orbital_config)
                data.xfrc_applied[body_ids[insp_id + 1], 0:3] = net_thrust + cw_force

                # Log error
                error = np.linalg.norm(pos - target_pos)
                log_errors[insp_id].append(error)

            # Check separations
            separations = fleet.get_min_separations()
            log_separations.append(separations["inter_inspector"])
            log_to_target.append(separations["to_target"])

            last_control_time = t
            control_step += 1

            if control_step % 50 == 0:
                avg_err = np.mean([log_errors[i][-1] for i in range(3)])
                print(
                    f"  t={t:.1f}s: avg_err={avg_err*100:.1f}cm, "
                    f"min_sep={separations['inter_inspector']:.2f}m, "
                    f"min_to_target={separations['to_target']:.2f}m"
                )

        mujoco.mj_step(model, data)
        t += sim_dt

    # Results
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)

    # Calculate metrics
    final_errors = [log_errors[i][-1] for i in range(3)]
    avg_final_error = np.mean(final_errors)
    max_final_error = max(final_errors)

    # Filter out initial zeros (before all inspectors registered)
    valid_separations = [s for s in log_separations if s > 0]
    valid_to_target = [s for s in log_to_target if s > 0]
    min_separation = min(valid_separations) if valid_separations else 0
    min_to_target = min(valid_to_target) if valid_to_target else 0

    print(f"\nFormation Errors:")
    for i in range(3):
        print(f"  Inspector {i}: {final_errors[i]*100:.2f} cm")
    print(f"  Average: {avg_final_error*100:.2f} cm")

    print(f"\nSafety Margins:")
    print(f"  Min inter-inspector separation: {min_separation:.2f} m (target: >1m)")
    print(f"  Min distance to target: {min_to_target:.2f} m (target: >2m)")

    # Pass criteria
    formation_ok = max_final_error < 0.10  # 10cm
    separation_ok = min_separation > 1.0  # 1m
    keepout_ok = min_to_target > 2.0  # 2m

    print(f"\nResults:")
    print(f"  Formation ±10cm: {'✓ PASS' if formation_ok else '✗ FAIL'}")
    print(f"  Inter-inspector >1m: {'✓ PASS' if separation_ok else '✗ FAIL'}")
    print(f"  Target keep-out >2m: {'✓ PASS' if keepout_ok else '✗ FAIL'}")

    return formation_ok and separation_ok and keepout_ok


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTI-SATELLITE FLEET COORDINATION TESTS")
    print("=" * 60 + "\n")

    success = test_fleet_formation()

    print("\n" + "=" * 60)
    print(f"OVERALL: {'✓ PASS' if success else '✗ FAIL'}")
    print("=" * 60)

    sys.exit(0 if success else 1)
