import unittest
import numpy as np
from src.satellite_control.core.thruster_manager import ThrusterManager


class TestThrusterToggle(unittest.TestCase):
    def test_pwm_mode(self):
        """Test that PWM mode produces binary outputs."""
        manager = ThrusterManager(
            num_thrusters=2,
            valve_delay=0.0,
            thrust_rampup_time=0.0,
            use_realistic_physics=False,
            thruster_type="PWM",
        )

        # Test 50% duty cycle
        manager.set_thruster_pattern(np.array([0.5, 0.0]), simulation_time=0.0)

        # At t=0 (start of interval), it should be ON (binary 1)
        manager.process_command_queue(
            simulation_time=0.0, control_update_interval=1.0, last_control_update=0.0, sim_dt=0.1
        )
        output_start = manager.get_actual_output()
        self.assertEqual(output_start[0], 1.0, "PWM should start ON")

        # At t=0.6 (> 0.5), it should be OFF (binary 0)
        manager.process_command_queue(
            simulation_time=0.6, control_update_interval=1.0, last_control_update=0.0, sim_dt=0.1
        )
        output_end = manager.get_actual_output()
        self.assertEqual(output_end[0], 0.0, "PWM should turn OFF after duty cycle")

    def test_continuous_mode(self):
        """Test that CON mode produces continuous outputs."""
        manager = ThrusterManager(
            num_thrusters=2,
            valve_delay=0.0,
            thrust_rampup_time=0.0,
            use_realistic_physics=False,
            thruster_type="CON",
        )

        # Test 37% thrust
        target_val = 0.37
        manager.set_thruster_pattern(np.array([target_val, 0.0]), simulation_time=0.0)

        # At any time, it should output exactly the commanded value
        manager.process_command_queue(
            simulation_time=0.0, control_update_interval=1.0, last_control_update=0.0, sim_dt=0.1
        )
        output = manager.get_actual_output()
        self.assertAlmostEqual(
            output[0], target_val, places=5, msg="CON should output exact command"
        )

        # Even later in the interval
        manager.process_command_queue(
            simulation_time=0.6, control_update_interval=1.0, last_control_update=0.0, sim_dt=0.1
        )
        output = manager.get_actual_output()
        self.assertAlmostEqual(output[0], target_val, places=5, msg="CON should hold exact command")


if __name__ == "__main__":
    unittest.main()
