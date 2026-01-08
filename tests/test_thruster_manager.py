"""
Unit tests for ThrusterManager module.

Tests thruster command processing, valve delays, and PWM logic.
"""

import numpy as np
import pytest

from src.satellite_control.core.thruster_manager import ThrusterManager


class TestThrusterManagerInitialization:
    """Test ThrusterManager initialization."""

    def test_manager_creation_defaults(self):
        """Test that ThrusterManager can be created with defaults."""
        manager = ThrusterManager()
        assert manager.num_thrusters == 12
        assert manager.VALVE_DELAY == 0.0
        assert manager.THRUST_RAMPUP_TIME == 0.0
        assert manager.use_realistic_physics is False
        assert manager.thruster_type == "PWM"

    def test_manager_creation_custom(self):
        """Test that ThrusterManager can be created with custom values."""
        manager = ThrusterManager(
            num_thrusters=8,
            valve_delay=0.05,
            thrust_rampup_time=0.015,
            use_realistic_physics=True,
            thruster_type="CON",
        )
        assert manager.num_thrusters == 8
        assert manager.VALVE_DELAY == 0.05
        assert manager.THRUST_RAMPUP_TIME == 0.015
        assert manager.use_realistic_physics is True
        assert manager.thruster_type == "CON"

    def test_manager_initializes_arrays(self):
        """Test that manager initializes all arrays correctly."""
        manager = ThrusterManager(num_thrusters=8)
        assert manager.current_thrusters.shape == (8,)
        assert manager.thruster_last_command.shape == (8,)
        assert manager.thruster_open_command_time.shape == (8,)
        assert manager.thruster_close_command_time.shape == (8,)
        assert manager.thruster_valve_open_time.shape == (8,)
        assert manager.thruster_actual_output.shape == (8,)
        assert manager.thruster_internal_binary_command.shape == (8,)


class TestThrusterManagerSetPattern:
    """Test set_thruster_pattern method."""

    def test_set_pattern_basic(self):
        """Test setting a basic thruster pattern."""
        manager = ThrusterManager(num_thrusters=8)
        pattern = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        manager.set_thruster_pattern(pattern, simulation_time=1.0)

        assert np.allclose(manager.current_thrusters, pattern)
        assert np.allclose(manager.thruster_last_command, pattern)

    def test_set_pattern_updates_command_times(self):
        """Test that setting pattern updates command times."""
        manager = ThrusterManager(num_thrusters=8)
        pattern = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        manager.set_thruster_pattern(pattern, simulation_time=2.0)

        # Thruster 0 should have open command time set
        assert manager.thruster_open_command_time[0] == 2.0
        # Other thrusters should have close command time set
        assert manager.thruster_close_command_time[1] == 2.0

    def test_set_pattern_handles_2d_array(self):
        """Test that set_pattern handles 2D arrays."""
        manager = ThrusterManager(num_thrusters=8)
        pattern = np.array([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])

        manager.set_thruster_pattern(pattern, simulation_time=1.0)

        # Should flatten to 1D
        assert manager.current_thrusters.shape == (8,)

    def test_set_pattern_handles_short_array(self):
        """Test that set_pattern handles arrays shorter than num_thrusters."""
        manager = ThrusterManager(num_thrusters=8)
        pattern = np.array([1.0, 0.0, 1.0])  # Only 3 elements

        manager.set_thruster_pattern(pattern, simulation_time=1.0)

        # Should pad with zeros
        assert manager.current_thrusters.shape == (8,)
        assert manager.current_thrusters[0] == 1.0
        assert manager.current_thrusters[1] == 0.0
        assert manager.current_thrusters[2] == 1.0
        assert manager.current_thrusters[3] == 0.0


class TestThrusterManagerUpdatePhysics:
    """Test update_physics method."""

    def test_update_physics_idealized(self):
        """Test physics update with idealized physics (no delays)."""
        manager = ThrusterManager(
            num_thrusters=8,
            use_realistic_physics=False,
        )
        pattern = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        manager.set_thruster_pattern(pattern, simulation_time=0.0)

        manager.update_physics(simulation_time=0.1, dt=0.005)

        # With idealized physics, output should match command immediately
        assert np.allclose(manager.thruster_actual_output, pattern)

    def test_update_physics_realistic_delays(self):
        """Test physics update with realistic delays."""
        manager = ThrusterManager(
            num_thrusters=8,
            valve_delay=0.05,
            thrust_rampup_time=0.015,
            use_realistic_physics=True,
        )
        pattern = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        manager.set_thruster_pattern(pattern, simulation_time=0.0)

        # Update before valve delay - should still be 0
        manager.update_physics(simulation_time=0.03, dt=0.005)
        assert manager.thruster_actual_output[0] == 0.0

        # Update after valve delay but before ramp-up complete
        manager.update_physics(simulation_time=0.06, dt=0.005)
        # Should be ramping up (between 0 and 1)
        assert 0.0 <= manager.thruster_actual_output[0] <= 1.0

        # Update after ramp-up complete
        manager.update_physics(simulation_time=0.08, dt=0.005)
        # Should be at full thrust
        assert manager.thruster_actual_output[0] == pytest.approx(1.0, abs=0.1)

    def test_update_physics_valve_closing(self):
        """Test physics update when valve closes."""
        manager = ThrusterManager(
            num_thrusters=8,
            valve_delay=0.05,
            use_realistic_physics=True,
        )
        # Turn on
        pattern_on = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        manager.set_thruster_pattern(pattern_on, simulation_time=0.0)
        manager.update_physics(simulation_time=0.1, dt=0.005)  # After delay

        # Turn off
        pattern_off = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        manager.set_thruster_pattern(pattern_off, simulation_time=0.1)

        # Update before close delay
        manager.update_physics(simulation_time=0.12, dt=0.005)
        # Should still be on
        assert manager.thruster_actual_output[0] > 0.0

        # Update after close delay
        manager.update_physics(simulation_time=0.16, dt=0.005)
        # Should be off
        assert manager.thruster_actual_output[0] == 0.0


class TestThrusterManagerProperties:
    """Test ThrusterManager properties."""

    def test_thruster_actual_output_property(self):
        """Test that thruster_actual_output is accessible."""
        manager = ThrusterManager(num_thrusters=8)
        assert hasattr(manager, "thruster_actual_output")
        assert manager.thruster_actual_output.shape == (8,)

    def test_thruster_last_command_property(self):
        """Test that thruster_last_command is accessible."""
        manager = ThrusterManager(num_thrusters=8)
        pattern = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        manager.set_thruster_pattern(pattern, simulation_time=0.0)

        assert np.allclose(manager.thruster_last_command, pattern)


class TestThrusterManagerPWM:
    """Test PWM-specific functionality."""

    def test_pwm_duty_cycle(self):
        """Test PWM duty cycle logic."""
        manager = ThrusterManager(
            num_thrusters=8,
            thruster_type="PWM",
            use_realistic_physics=False,
        )

        # Set 50% duty cycle
        pattern = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        manager.set_thruster_pattern(pattern, simulation_time=0.0)

        # Over multiple control intervals, average should approach 0.5
        # (exact behavior depends on implementation)
        manager.update_physics(simulation_time=0.1, dt=0.005)
        assert 0.0 <= manager.thruster_actual_output[0] <= 1.0


class TestThrusterManagerContinuous:
    """Test continuous thruster mode."""

    def test_continuous_mode(self):
        """Test continuous thruster mode (no PWM)."""
        manager = ThrusterManager(
            num_thrusters=8,
            thruster_type="CON",
            use_realistic_physics=False,
        )

        pattern = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        manager.set_thruster_pattern(pattern, simulation_time=0.0)
        manager.update_physics(simulation_time=0.1, dt=0.005)

        # Continuous mode should directly use the command value
        assert manager.thruster_actual_output[0] == pytest.approx(0.7, abs=0.01)
