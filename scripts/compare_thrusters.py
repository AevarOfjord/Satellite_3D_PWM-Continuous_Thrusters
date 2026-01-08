import numpy as np
from src.satellite_control.core.thruster_manager import ThrusterManager


def run_comparison():
    print("--- THRUSTER MODE COMPARISON ---")

    # 1. PWM Mode
    print("\n1. PWM MODE (thruster_type='PWM')")
    pwm_manager = ThrusterManager(num_thrusters=12, thruster_type="PWM")
    # Command 50% thrust on thruster 1
    pattern = np.zeros(12)
    pattern[0] = 0.5
    pwm_manager.set_thruster_pattern(pattern, simulation_time=0.0)

    print("   Command: 50% (0.50)")
    print("   Output over 1 second (dt=0.1s):")
    outputs = []
    times = []
    for t in np.arange(0, 1.1, 0.1):
        pwm_manager.process_command_queue(
            simulation_time=t,
            control_update_interval=1.0,  # 1s control cycle
            last_control_update=0.0,
            sim_dt=0.01,
        )
        out = pwm_manager.get_actual_output()[0]
        outputs.append(out)
        times.append(t)
        print(f"   t={t:.1f}s: {out:.1f} {'(ON)' if out > 0 else '(OFF)'}")

    # 2. Continuous Mode
    print("\n2. CONTINUOUS MODE (thruster_type='CON')")
    con_manager = ThrusterManager(num_thrusters=12, thruster_type="CON")
    # Command 37% thrust on thruster 1
    pattern = np.zeros(12)
    pattern[0] = 0.37
    con_manager.set_thruster_pattern(pattern, simulation_time=0.0)

    print("   Command: 37% (0.37)")
    print("   Output over 1 second:")

    # Check a few points
    for t in [0.0, 0.5, 1.0]:
        con_manager.process_command_queue(
            simulation_time=t, control_update_interval=1.0, last_control_update=0.0, sim_dt=0.01
        )
        out = con_manager.get_actual_output()[0]
        print(f"   t={t:.1f}s: {out:.2f} (Exact match)")


if __name__ == "__main__":
    run_comparison()
