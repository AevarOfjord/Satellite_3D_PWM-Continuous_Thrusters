"""
Unit tests for unified MPCController.

Tests the MPCController class which provides Model Predictive Control
for satellite thruster systems using linearized dynamics and OSQP.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
from typing import Any, Dict
from src.satellite_control.control.mpc_controller import MPCController


# creating dummy classes to mimic config objects if needed,
# but we can pass dicts if the controller supports it, or Mock objects.
# The controller expects Pydantic models or dicts.


class TestMPCControllerInitialization:
    """Test MPC controller initialization."""

    def test_default_initialization(self):
        """Test that MPC controller can be initialized with default parameters."""
        satellite_params: Dict[str, Any] = {
            "mass": 10.0,
            "inertia": 0.1,
            "com_offset": (0.0, 0.0, 0.0),
            "thruster_positions": {i: (0.1, 0.0, 0.0) for i in range(1, 9)},
            "thruster_directions": {i: (1.0, 0.0, 0.0) for i in range(1, 9)},
            "thruster_forces": {i: 1.0 for i in range(1, 9)},
            "damping_linear": 0.0,
            "damping_angular": 0.0,
        }

        mpc_params = {
            "prediction_horizon": 10,
            "control_horizon": 10,
            "dt": 0.1,
            "solver_time_limit": 0.01,
            "position_bounds": 5.0,
            "max_velocity": 0.5,
            "max_angular_velocity": 1.0,
            # Weights
            "q_position": 10.0,
            "q_velocity": 1.0,
            "q_angle": 10.0,
            "q_angular_velocity": 1.0,
            "r_thrust": 0.1,
            "r_switch": 0.0,
            # Adaptive
            "damping_zone": 0.1,
            "velocity_threshold": 0.01,
            "max_velocity_weight": 100.0,
        }

        mpc = MPCController(satellite_params, mpc_params)

        # Check that key attributes are initialized
        assert mpc.total_mass == 10.0
        assert mpc.moment_of_inertia == 0.1
        assert mpc.N == 10  # Prediction horizon
        assert mpc.dt == 0.1
        assert mpc.nx == 13  # State dimension
        assert mpc.nu == 12  # Control dimension

        # Check OSQP initialization
        assert mpc.prob is not None

    def test_thruster_precomputation(self):
        """Test that thruster forces and torques are precomputed."""
        # Setup similar to above
        satellite_params: Dict[str, Any] = {
            "mass": 10.0,
            "inertia": 0.1,
            "com_offset": (0.0, 0.0, 0.0),
            "thruster_positions": {1: (0.1, 0.0, 0.0)},
            "thruster_directions": {1: (0.0, 1.0, 0.0)},  # firing Y+ at X+ -> Torque +
            "thruster_forces": {1: 1.0},
            "damping_linear": 0.0,
            "damping_angular": 0.0,
        }
        # fill remaining thrusters with dummies to pass strict validation if needed
        for i in range(2, 9):
            satellite_params["thruster_positions"][i] = (0.0, 0.0, 0.0)
            satellite_params["thruster_directions"][i] = (1.0, 0.0, 0.0)
            satellite_params["thruster_forces"][i] = 0.0

        mpc_params = {
            "prediction_horizon": 5,
            "control_horizon": 5,
            "dt": 0.1,
            "solver_time_limit": 0.01,
            "position_bounds": 5.0,
            "max_velocity": 0.5,
            "max_angular_velocity": 1.0,
            # Weights needed for Q_diag
            "q_position": 10.0,
            "q_velocity": 1.0,
            "q_angle": 10.0,
            "q_angular_velocity": 1.0,
            "r_thrust": 0.1,
            "r_switch": 0.0,
        }

        mpc = MPCController(satellite_params, mpc_params)

        # Check body frame forces
        # T1 at (0.1, 0) pointing (0, 1) -> force (0, 1)
        # Torque = r x F = 0.1*1 - 0*0 = 0.1
        assert mpc.body_frame_forces[0, 1] == pytest.approx(1.0)
        assert mpc.body_frame_torques[0, 2] == pytest.approx(0.1)


class TestLinearization:
    """Test linearization of satellite dynamics."""

    def test_linearize_dynamics_identity(self):
        """Test A matrix structure."""
        mpc = MPCController()  # Use defaults if allowed or Minimal setup

        x = np.zeros(13)
        x[3] = 1.0
        A, B = mpc.linearize_dynamics(x)

        # A should be I + dt conversion
        assert A[0, 0] == 1.0
        assert A[0, 7] == mpc.dt
        assert A.shape == (13, 13)
        assert B.shape == (13, 12)


class TestControlAction:
    """Test control action computation."""

    def test_get_control_action_osqp_mock(self):
        """Test that get_control_action calls OSQP solver."""
        mpc = MPCController()

        # Mock OSQP methods
        mpc.prob = MagicMock()

        class MockResult:
            x = np.zeros(mpc.nx * (mpc.N + 1) + mpc.nu * mpc.N)  # Simplified
            info = MagicMock()
            info.run_time = 0.001
            info.status = "solved"

        mock_res = MockResult()
        # Mock the extraction logic by mocking the solution vector to have 0.5 for first thruster
        total_vars = mpc.nx * (mpc.N + 1) + mpc.nu * mpc.N
        mock_res.x = np.zeros(total_vars)

        # Set first thruster to 0.5
        u_start_idx = mpc.nx * (mpc.N + 1)
        mock_res.x[u_start_idx] = 0.5

        mpc.prob.solve.return_value = mock_res

        x_curr = np.zeros(13)
        x_target = np.zeros(13)
        x_curr[3] = 1.0
        x_target[3] = 1.0

        u, info = mpc.get_control_action(x_curr, x_target)

        assert mpc.prob.solve.called
        assert mpc.prob.update.called  # Should update vectors
        assert u[0] == 0.5
        assert info["solve_time"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
