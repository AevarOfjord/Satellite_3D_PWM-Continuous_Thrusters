"""
Unit tests for SimulationContext module.

Tests the SimulationContext dataclass that holds simulation state.
"""

import numpy as np
import pytest

from src.satellite_control.core.simulation_context import SimulationContext


class TestSimulationContextInitialization:
    """Test SimulationContext initialization."""

    def test_context_creation_defaults(self):
        """Test that SimulationContext can be created with defaults."""
        context = SimulationContext()
        assert context.simulation_time == 0.0
        assert context.dt == 0.005
        assert context.control_dt == 0.1
        assert context.step_number == 0
        assert context.mission_phase == "IDLE"
        assert context.waypoint_number == 0
        assert context.last_control_update_time == 0.0
        assert context.computation_time_last_step == 0.0

    def test_context_state_arrays(self):
        """Test that state arrays are initialized correctly."""
        context = SimulationContext()
        assert context.current_state.shape == (13,)
        assert context.target_state.shape == (13,)
        assert np.all(context.current_state == 0)
        assert np.all(context.target_state == 0)

    def test_context_lists(self):
        """Test that list fields are initialized correctly."""
        context = SimulationContext()
        assert isinstance(context.active_thrusters, list)
        assert len(context.active_thrusters) == 0
        assert context.previous_thruster_command is None

    def test_context_custom_values(self):
        """Test that SimulationContext can be created with custom values."""
        current_state = np.ones(13)
        target_state = np.ones(13) * 2
        context = SimulationContext(
            simulation_time=1.5,
            dt=0.01,
            control_dt=0.2,
            step_number=10,
            current_state=current_state,
            target_state=target_state,
            mission_phase="APPROACHING",
            waypoint_number=2,
            last_control_update_time=1.4,
            active_thrusters=[1, 3, 5],
            previous_thruster_command=np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            computation_time_last_step=0.005,
        )

        assert context.simulation_time == 1.5
        assert context.dt == 0.01
        assert context.control_dt == 0.2
        assert context.step_number == 10
        assert np.all(context.current_state == current_state)
        assert np.all(context.target_state == target_state)
        assert context.mission_phase == "APPROACHING"
        assert context.waypoint_number == 2
        assert context.last_control_update_time == 1.4
        assert context.active_thrusters == [1, 3, 5]
        assert np.all(context.previous_thruster_command == np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
        assert context.computation_time_last_step == 0.005


class TestSimulationContextUpdateState:
    """Test the update_state method."""

    def test_update_state(self):
        """Test that update_state updates state correctly."""
        context = SimulationContext()
        new_time = 1.0
        new_state = np.ones(13)
        new_target = np.ones(13) * 2

        context.update_state(new_time, new_state, new_target)

        assert context.simulation_time == new_time
        assert np.all(context.current_state == new_state)
        assert np.all(context.target_state == new_target)

    def test_update_state_preserves_other_fields(self):
        """Test that update_state doesn't modify other fields."""
        context = SimulationContext(
            step_number=5,
            mission_phase="TRACKING",
            waypoint_number=1,
        )

        context.update_state(1.0, np.ones(13), np.ones(13))

        assert context.step_number == 5
        assert context.mission_phase == "TRACKING"
        assert context.waypoint_number == 1


class TestSimulationContextStateShape:
    """Test state array shape validation."""

    def test_state_must_be_13_elements(self):
        """Test that state arrays must be 13 elements."""
        context = SimulationContext()

        # Valid 13-element state
        valid_state = np.zeros(13)
        context.current_state = valid_state
        assert context.current_state.shape == (13,)

        # Should accept any 13-element array
        context.current_state = np.ones(13) * 5
        assert context.current_state.shape == (13,)
        assert np.all(context.current_state == 5)


class TestSimulationContextImmutability:
    """Test that context fields can be modified (not immutable)."""

    def test_context_fields_mutable(self):
        """Test that context fields can be modified."""
        context = SimulationContext()

        # Modify various fields
        context.simulation_time = 10.0
        context.step_number = 100
        context.mission_phase = "COMPLETE"
        context.waypoint_number = 5
        context.active_thrusters.append(1)
        context.active_thrusters.append(2)

        assert context.simulation_time == 10.0
        assert context.step_number == 100
        assert context.mission_phase == "COMPLETE"
        assert context.waypoint_number == 5
        assert context.active_thrusters == [1, 2]
