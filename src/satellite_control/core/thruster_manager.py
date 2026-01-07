"""
Thruster Manager Module

Handles thruster valve delays, ramp-up physics, and PWM duty cycle logic.
Extracted from simulation.py for cleaner architecture.

Thruster Valve Behavior:
- ON command sent at t → valve opens at t + VALVE_DELAY
- OFF command sent at t → valve closes at t + VALVE_DELAY
- Once valve opens, thrust ramps linearly over THRUST_RAMPUP_TIME
- PWM commands use duty cycle logic within control intervals
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    # Avoid circular import
    from src.satellite_control.core.mujoco_satellite import SatelliteThrusterTester


class ThrusterManager:
    """
    Manages thruster command processing, valve delays, and thrust output.

    Simulates realistic solenoid valve behavior including:
    - Command-to-valve-action delay
    - Thrust ramp-up after valve opening
    - PWM duty cycle within control intervals
    """

    def __init__(
        self,
        num_thrusters: int = 12,
        valve_delay: float = 0.0,
        thrust_rampup_time: float = 0.0,
        use_realistic_physics: bool = False,
        thruster_type: str = "PWM",
    ):
        """
        Initialize thruster manager.

        Args:
            num_thrusters: Number of thrusters (default 12)
            valve_delay: Time from command to valve action [s]
            thrust_rampup_time: Time for thrust to reach full value [s]
            use_realistic_physics: Enable valve delays and ramp-up
            thruster_type: 'PWM' (Binary) or 'CON' (Continuous)
        """
        self.num_thrusters = num_thrusters
        self.VALVE_DELAY = valve_delay
        self.THRUST_RAMPUP_TIME = thrust_rampup_time
        self.use_realistic_physics = use_realistic_physics
        self.thruster_type = thruster_type

        # Current commanded pattern (duty cycle 0-1)
        self.current_thrusters = np.zeros(num_thrusters, dtype=np.float64)

        # Last command value for each thruster
        self.thruster_last_command = np.zeros(num_thrusters, dtype=np.float64)

        # Timing for valve open commands
        self.thruster_open_command_time = np.full(num_thrusters, -1000.0, dtype=np.float64)

        # Timing for valve close commands
        self.thruster_close_command_time = np.full(num_thrusters, -1000.0, dtype=np.float64)

        # Actual valve open times (after delay)
        self.thruster_valve_open_time = np.full(num_thrusters, -1000.0, dtype=np.float64)

        # Actual output level [0, 1] for each thruster
        self.thruster_actual_output = np.zeros(num_thrusters, dtype=np.float64)

        # Internal binary state (for PWM pulse tracking)
        self.thruster_internal_binary_command = np.zeros(num_thrusters, dtype=np.float64)

    def set_thruster_pattern(
        self,
        thruster_pattern: np.ndarray,
        simulation_time: float,
    ) -> None:
        """
        Send thruster command (immediate, but valve response is delayed).

        Command is sent at current simulation_time, but valve opening/closing
        takes VALVE_DELAY to complete.

        Args:
            thruster_pattern: Array [0,1] for thruster commands (duty cycle)
            simulation_time: Current simulation time
        """
        thruster_pattern = np.asarray(thruster_pattern, dtype=np.float64)
        if thruster_pattern.ndim == 2:
            thruster_pattern = thruster_pattern[:, 0]
        thruster_pattern = thruster_pattern.ravel()

        if thruster_pattern.size < self.num_thrusters:
            padded = np.zeros(self.num_thrusters, dtype=np.float64)
            padded[: thruster_pattern.size] = thruster_pattern
            thruster_pattern = padded
        elif thruster_pattern.size > self.num_thrusters:
            thruster_pattern = thruster_pattern[: self.num_thrusters]

        self.current_thrusters = thruster_pattern.copy()

        # Record command time for each thruster that changed state
        for i in range(self.num_thrusters):
            new_command = thruster_pattern[i]
            old_command = self.thruster_last_command[i]

            if new_command != old_command:
                # Command state changed
                if new_command > 0.01:  # ON command (PWM/Throttle > 1%)
                    self.thruster_open_command_time[i] = simulation_time
                else:  # OFF command
                    self.thruster_close_command_time[i] = simulation_time

                self.thruster_last_command[i] = new_command

    def process_command_queue(
        self,
        simulation_time: float,
        control_update_interval: float,
        last_control_update: float,
        sim_dt: float,
        satellite: Optional["SatelliteThrusterTester"] = None,
    ) -> None:
        """
        Update actual thruster output based on valve delays and ramp-up.

        Called every simulation timestep to update actual thruster forces.

        Args:
            simulation_time: Current simulation time
            control_update_interval: Duration of control interval
            last_control_update: Time of last control update
            sim_dt: Physics simulation timestep
            satellite: Satellite object to update (optional)
        """
        if self.thruster_type == "PWM":
            # --- PWM / BINARY MODE ---
            for i in range(self.num_thrusters):
                current_command = self.thruster_last_command[i]

                # Determine instantaneous command based on PWM Duty Cycle
                thrust_cmd_binary = 0.0

                # Calculate position in control interval for PWM
                if control_update_interval > 0:
                    time_since_update = simulation_time - last_control_update
                    if time_since_update < 0:
                        time_since_update = 0

                    # Duty Cycle Logic
                    duty_cycle = max(0.0, min(1.0, current_command))

                    # Calculate raw pulse duration
                    raw_pulse_duration = duty_cycle * control_update_interval

                    # Quantize pulse duration to physics timestep
                    # Ensure at least 1 step for any non-zero duty cycle
                    steps = round(raw_pulse_duration / sim_dt)
                    if duty_cycle > 0.01 and steps == 0:
                        steps = 1  # Minimum 1 physics step if any thrust commanded
                    pulse_duration = steps * sim_dt

                    # Epsilon for float comparison
                    if time_since_update < pulse_duration - 1e-9:
                        thrust_cmd_binary = 1
                    else:
                        thrust_cmd_binary = 0
                else:
                    # Fallback if no interval defined
                    thrust_cmd_binary = 1 if current_command > 0.5 else 0

                # Detect PWM Switch Event
                if thrust_cmd_binary != self.thruster_internal_binary_command[i]:
                    self.thruster_internal_binary_command[i] = thrust_cmd_binary
                    if thrust_cmd_binary == 1:
                        self.thruster_open_command_time[i] = simulation_time
                    else:
                        self.thruster_close_command_time[i] = simulation_time

                # Get timing values
                open_cmd_time = self.thruster_open_command_time[i]
                close_cmd_time = self.thruster_close_command_time[i]

                if thrust_cmd_binary == 1:  # Currently commanded ON
                    # Check if valve has had time to open
                    valve_open_time = open_cmd_time + self.VALVE_DELAY

                    if simulation_time >= valve_open_time:
                        # Valve is open, apply ramp-up
                        time_since_valve_open = simulation_time - valve_open_time

                        # Record new opening
                        if self.thruster_valve_open_time[i] < valve_open_time - 0.001:
                            self.thruster_valve_open_time[i] = valve_open_time

                        if self.use_realistic_physics and self.THRUST_RAMPUP_TIME > 0:
                            ramp_progress = time_since_valve_open / self.THRUST_RAMPUP_TIME
                            self.thruster_actual_output[i] = min(1.0, ramp_progress)
                        else:
                            # Binary full thrust when PWM pulse is ON
                            self.thruster_actual_output[i] = 1.0
                    else:
                        # Valve hasn't opened yet
                        self.thruster_actual_output[i] = 0.0

                else:  # Currently commanded OFF
                    # Check if valve has had time to close
                    valve_close_time = close_cmd_time + self.VALVE_DELAY

                    if simulation_time >= valve_close_time:
                        # Valve is closed
                        self.thruster_actual_output[i] = 0.0
                    else:
                        # Valve hasn't closed yet
                        valve_open_time = open_cmd_time + self.VALVE_DELAY
                        if simulation_time < valve_open_time:
                            # Never opened
                            self.thruster_actual_output[i] = 0.0
                        else:
                            # Was open, now closing
                            self.thruster_actual_output[i] = 0.0

        elif self.thruster_type == "CON":
            # --- CONTINUOUS MODE ---
            # Bypass all PWM/Valve logic and just output the commanded value
            # We still allow for a simple delay/ramp if realistic physics is on,
            # but usually continuous means "proportional valve" or electric prop.

            for i in range(self.num_thrusters):
                target = self.thruster_last_command[i]

                # If we want instant response:
                if not self.use_realistic_physics:
                    self.thruster_actual_output[i] = target
                else:
                    # Simple first-order lag or rate limit could go here if needed
                    # For now, let's assume continuous valves are fast enough or
                    # modelled by the "ramp up" but without the binary switch.
                    # Simplified: just pass it through for now as requested.
                    self.thruster_actual_output[i] = target

        # Update satellite if provided
        if satellite is not None:
            self._update_satellite_thrusters(satellite, simulation_time)

    def _update_satellite_thrusters(
        self,
        satellite: "SatelliteThrusterTester",
        simulation_time: float,
    ) -> None:
        """
        Update satellite thruster states based on actual output.

        Args:
            satellite: Satellite object to update
            simulation_time: Current simulation time
        """
        previous_active = satellite.active_thrusters.copy()
        satellite.active_thrusters.clear()

        for i, output in enumerate(self.thruster_actual_output):
            thruster_id = i + 1  # Thrusters are 1-indexed

            # Apply level to satellite physics
            if hasattr(satellite, "set_thruster_level"):
                satellite.set_thruster_level(thruster_id, output)

            if output > 0.01:  # Threshold for activation
                satellite.active_thrusters.add(thruster_id)

                # Set activation time for NEW activations
                if thruster_id not in previous_active:
                    satellite.thruster_activation_time[thruster_id] = simulation_time
                    if thruster_id in satellite.thruster_deactivation_time:
                        del satellite.thruster_deactivation_time[thruster_id]
            else:
                # Thruster commanded OFF
                if thruster_id in previous_active:
                    satellite.thruster_deactivation_time[thruster_id] = simulation_time

                if thruster_id in satellite.thruster_activation_time:
                    del satellite.thruster_activation_time[thruster_id]

        # Clean up old deactivation times
        if self.use_realistic_physics:
            total_shutoff_time = self.VALVE_DELAY + self.THRUST_RAMPUP_TIME
            thrusters_to_cleanup = [
                tid
                for tid, deact_time in satellite.thruster_deactivation_time.items()
                if simulation_time - deact_time >= total_shutoff_time
            ]
            for thruster_id in thrusters_to_cleanup:
                del satellite.thruster_deactivation_time[thruster_id]

    def get_actual_output(self) -> np.ndarray:
        """Get current actual thruster output levels."""
        return self.thruster_actual_output.copy()

    def get_commanded_pattern(self) -> np.ndarray:
        """Get current commanded thruster pattern."""
        return self.current_thrusters.copy()

    def reset(self) -> None:
        """Reset all thruster states to initial values."""
        self.current_thrusters.fill(0.0)
        self.thruster_last_command.fill(0.0)
        self.thruster_open_command_time.fill(-1000.0)
        self.thruster_close_command_time.fill(-1000.0)
        self.thruster_valve_open_time.fill(-1000.0)
        self.thruster_actual_output.fill(0.0)
        self.thruster_internal_binary_command.fill(0.0)
