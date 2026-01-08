import numpy as np

from src.satellite_control.core.simulation import SatelliteMPCLinearizedSimulation


def test_headless_physics():
    print("Testing Headless Physics...")

    # Configure for movement -- 3D
    start_pos = (0.0, 0.0, 0.0)
    target_pos = (1.0, 1.0, 0.0)  # Aim for (1,1,0)

    # Initialize simulation in headless mode
    sim = SatelliteMPCLinearizedSimulation(
        start_pos=start_pos,
        target_pos=target_pos,
        start_angle=(0.0, 0.0, 0.0),
        target_angle=(0.0, 0.0, 0.0),
        use_mujoco_viewer=False,
    )

    print(f"Initial State: {sim.get_current_state()}")

    # Run for a few steps
    # We need to actuate thrusters to move.
    # The MPC should compute controls to move towards target.

    # Run for a short duration
    sim.max_simulation_time = 0.5  # Run for 0.5 seconds

    try:
        sim.run_simulation(show_animation=False)
    except SystemExit:
        pass
    except Exception as e:
        print(f"Simulation error: {e}")

    final_state = sim.get_current_state()
    print(f"\nFinal State: {final_state}")

    dist = np.linalg.norm(final_state[0:3] - np.array(start_pos))
    print(f"Distance moved: {dist:.6f} m")

    if dist < 0.001:
        print("FAIL: Satellite did not translate significantly.")
    else:
        print("PASS: Satellite translated.")

    sim.close()


if __name__ == "__main__":
    test_headless_physics()
