"""
Unit tests for StateConverter module.

Tests state format conversion between simulation and MPC formats.

Note: The system uses 3D state format with quaternions:
      [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13 elements)
      
      2D format [x, y, vx, vy, theta, omega] (6 elements) is legacy support only.
"""

import numpy as np
import pytest

from src.satellite_control.utils.state_converter import StateConverter


class TestStateConverterSimToMPC:
    """Test sim_to_mpc conversion."""

    def test_sim_to_mpc_3d_identity(self):
        """
        Test that 3D state conversion is identity (primary format).
        
        The system uses 3D states with quaternions:
        [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        sim_state_3d = np.zeros(13)
        sim_state_3d[0:3] = [1.0, 2.0, 3.0]  # position
        sim_state_3d[3:7] = euler_xyz_to_quat_wxyz((0.1, 0.2, 0.3))  # quaternion from euler
        sim_state_3d[7:10] = [0.1, 0.2, 0.3]  # velocity
        sim_state_3d[10:13] = [0.0, 0.0, 0.1]  # angular velocity

        mpc_state = StateConverter.sim_to_mpc(sim_state_3d)

        # 3D states should be identical (no conversion needed)
        assert np.allclose(mpc_state, sim_state_3d)
        assert mpc_state.shape == (13,)

    def test_sim_to_mpc_3d_preserves_quaternion(self):
        """Test that 3D conversion preserves quaternion correctly."""
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        # Create state with non-identity quaternion
        sim_state_3d = np.zeros(13)
        sim_state_3d[0:3] = [1.0, 2.0, 0.0]
        quat = euler_xyz_to_quat_wxyz((np.pi / 4, 0.0, 0.0))  # 45 degree roll
        sim_state_3d[3:7] = quat
        sim_state_3d[7:10] = [0.0, 0.0, 0.0]
        sim_state_3d[10:13] = [0.0, 0.0, 0.0]

        mpc_state = StateConverter.sim_to_mpc(sim_state_3d)

        # Quaternion should be preserved exactly
        assert np.allclose(mpc_state[3:7], quat)
        assert np.allclose(mpc_state[0:3], [1.0, 2.0, 0.0])

    def test_sim_to_mpc_2d_legacy(self):
        """
        Test 2D state conversion (legacy support only).
        
        Note: This is for backward compatibility. The system primarily uses 3D.
        """
        sim_state = np.array([1.0, 2.0, 0.1, 0.2, 0.5, 0.3])  # [x, y, vx, vy, theta, omega]

        mpc_state = StateConverter.sim_to_mpc(sim_state)

        # MPC format: [x, y, theta, vx, vy, omega]
        assert mpc_state[0] == 1.0  # x
        assert mpc_state[1] == 2.0  # y
        assert mpc_state[2] == 0.5  # theta
        assert mpc_state[3] == 0.1  # vx
        assert mpc_state[4] == 0.2  # vy
        assert mpc_state[5] == 0.3  # omega

    def test_sim_to_mpc_invalid_length(self):
        """Test that invalid state length raises error."""
        invalid_state = np.array([1.0, 2.0, 3.0])  # Only 3 elements

        with pytest.raises(ValueError):
            StateConverter.sim_to_mpc(invalid_state)


class TestStateConverterMPCToSim:
    """Test mpc_to_sim conversion."""

    def test_mpc_to_sim_3d_identity(self):
        """
        Test that 3D state conversion is identity (primary format).
        
        The system uses 3D states with quaternions:
        [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        mpc_state_3d = np.zeros(13)
        mpc_state_3d[0:3] = [1.0, 2.0, 3.0]  # position
        mpc_state_3d[3:7] = euler_xyz_to_quat_wxyz((0.1, 0.2, 0.3))  # quaternion
        mpc_state_3d[7:10] = [0.1, 0.2, 0.3]  # velocity
        mpc_state_3d[10:13] = [0.0, 0.0, 0.1]  # angular velocity

        sim_state = StateConverter.mpc_to_sim(mpc_state_3d)

        # 3D states should be identical (no conversion needed)
        assert np.allclose(sim_state, mpc_state_3d)
        assert sim_state.shape == (13,)

    def test_mpc_to_sim_2d_legacy(self):
        """
        Test 2D state conversion (legacy support only).
        
        Note: This is for backward compatibility. The system primarily uses 3D.
        """
        mpc_state = np.array([1.0, 2.0, 0.5, 0.1, 0.2, 0.3])  # [x, y, theta, vx, vy, omega]

        sim_state = StateConverter.mpc_to_sim(mpc_state)

        # Sim format: [x, y, vx, vy, theta, omega]
        assert sim_state[0] == 1.0  # x
        assert sim_state[1] == 2.0  # y
        assert sim_state[2] == 0.1  # vx
        assert sim_state[3] == 0.2  # vy
        assert sim_state[4] == 0.5  # theta
        assert sim_state[5] == 0.3  # omega

    def test_mpc_to_sim_invalid_length(self):
        """Test that invalid state length raises error."""
        invalid_state = np.array([1.0, 2.0, 3.0])  # Only 3 elements

        with pytest.raises(ValueError):
            StateConverter.mpc_to_sim(invalid_state)


class TestStateConverterRoundTrip:
    """Test round-trip conversion."""

    def test_round_trip_3d_identity(self):
        """
        Test that 3D state round-trip is identity (primary format).
        
        Since 3D states are identity operations, round-trip should be exact.
        """
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        # Create realistic 3D state
        sim_state_3d = np.zeros(13)
        sim_state_3d[0:3] = [1.5, -2.3, 0.5]  # position
        sim_state_3d[3:7] = euler_xyz_to_quat_wxyz((0.1, -0.2, 1.2))  # quaternion
        sim_state_3d[7:10] = [0.15, -0.25, 0.05]  # velocity
        sim_state_3d[10:13] = [0.0, 0.0, -0.8]  # angular velocity

        mpc_state = StateConverter.sim_to_mpc(sim_state_3d)
        sim_state_back = StateConverter.mpc_to_sim(mpc_state)

        # Should be exact identity (no conversion for 3D)
        assert np.allclose(sim_state_3d, sim_state_back, atol=1e-10)

    def test_round_trip_3d_preserves_quaternion(self):
        """Test that round-trip preserves quaternion normalization."""
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        # Create state with normalized quaternion
        sim_state_3d = np.zeros(13)
        sim_state_3d[0:3] = [1.0, 2.0, 0.0]
        quat = euler_xyz_to_quat_wxyz((np.pi / 4, np.pi / 6, np.pi / 3))
        sim_state_3d[3:7] = quat
        sim_state_3d[7:10] = [0.0, 0.0, 0.0]
        sim_state_3d[10:13] = [0.0, 0.0, 0.0]

        mpc_state = StateConverter.sim_to_mpc(sim_state_3d)
        sim_state_back = StateConverter.mpc_to_sim(mpc_state)

        # Quaternion should be preserved exactly
        assert np.allclose(sim_state_back[3:7], quat, atol=1e-10)
        # Quaternion should remain normalized
        assert abs(np.linalg.norm(sim_state_back[3:7]) - 1.0) < 1e-10

    def test_round_trip_2d_legacy(self):
        """
        Test that 2D state round-trip returns original (legacy support).
        
        Note: This is for backward compatibility. The system primarily uses 3D.
        """
        sim_state = np.array([1.0, 2.0, 0.1, 0.2, 0.5, 0.3])

        mpc_state = StateConverter.sim_to_mpc(sim_state)
        sim_state_back = StateConverter.mpc_to_sim(mpc_state)

        assert np.allclose(sim_state, sim_state_back)


class TestStateConverterTrajectory:
    """Test trajectory conversion methods."""

    def test_sim_to_mpc_trajectory_3d(self):
        """Test trajectory conversion for 3D states (primary format)."""
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        # Create trajectory of 3D states
        trajectory = []
        for i in range(5):
            state = np.zeros(13)
            state[0:3] = [i * 0.1, i * 0.2, 0.0]
            state[3:7] = euler_xyz_to_quat_wxyz((0.0, 0.0, i * 0.1))
            trajectory.append(state)

        mpc_trajectory = StateConverter.sim_to_mpc_trajectory(trajectory)

        # Should be identity for 3D
        assert mpc_trajectory.shape == (5, 13)
        for i, state in enumerate(trajectory):
            assert np.allclose(mpc_trajectory[i], state)

    def test_mpc_to_sim_trajectory_3d(self):
        """Test trajectory conversion from MPC to sim for 3D states."""
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        # Create trajectory of 3D states
        trajectory = []
        for i in range(5):
            state = np.zeros(13)
            state[0:3] = [i * 0.1, i * 0.2, 0.0]
            state[3:7] = euler_xyz_to_quat_wxyz((0.0, 0.0, i * 0.1))
            trajectory.append(state)

        sim_trajectory = StateConverter.mpc_to_sim_trajectory(trajectory)

        # Should be identity for 3D
        assert sim_trajectory.shape == (5, 13)
        for i, state in enumerate(trajectory):
            assert np.allclose(sim_trajectory[i], state)


class TestStateConverterConstants:
    """Test StateConverter index constants (for legacy 2D support)."""

    def test_sim_indices_2d(self):
        """
        Test that SIM indices are correct for 2D format (legacy).
        
        Note: These are only used for 2D state conversion.
        """
        assert StateConverter.SIM_X == 0
        assert StateConverter.SIM_Y == 1
        assert StateConverter.SIM_VX == 2
        assert StateConverter.SIM_VY == 3
        assert StateConverter.SIM_THETA == 4
        assert StateConverter.SIM_OMEGA == 5

    def test_mpc_indices_2d(self):
        """
        Test that MPC indices are correct for 2D format (legacy).
        
        Note: These are only used for 2D state conversion.
        """
        assert StateConverter.MPC_X == 0
        assert StateConverter.MPC_Y == 1
        assert StateConverter.MPC_THETA == 2
        assert StateConverter.MPC_VX == 3
        assert StateConverter.MPC_VY == 4
        assert StateConverter.MPC_OMEGA == 5
