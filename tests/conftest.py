"""
Pytest configuration and shared fixtures.

This file provides common fixtures and configuration for all tests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Prefer SimulationConfig for tests (v3.0.0)
from src.satellite_control.config.simulation_config import SimulationConfig  # noqa: E402

# ============================================================================
# Configuration Reset Fixture
# ============================================================================


@pytest.fixture(autouse=True)
def fresh_config():
    """
    Provide a fresh SimulationConfig (preferred in v3.0.0).

    This replaces resetting SatelliteConfig and encourages tests to
    depend on explicit configs.
    """
    sim_config = SimulationConfig.create_default()
    yield sim_config


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def satellite_params():
    """Provide standard satellite parameters for testing."""
    return {
        "mass": 23.09,
        "inertia": 0.312,
        "size": 0.29,
        "thruster_forces": [0.45] * 8,
        "thruster_positions": {
            1: (0.145, 0.145),
            2: (0.145, -0.145),
            3: (-0.145, -0.145),
            4: (-0.145, 0.145),
            5: (0.145, 0.0),
            6: (0.0, -0.145),
            7: (-0.145, 0.0),
            8: (0.0, 0.145),
        },
        "thruster_directions": {
            1: (-1, 0),
            2: (0, 1),
            3: (1, 0),
            4: (0, -1),
            5: (0, -1),
            6: (1, 0),
            7: (0, 1),
            8: (-1, 0),
        },
        "damping_linear": 0.60,
        "damping_angular": 0.035,
    }


@pytest.fixture
def mpc_params():
    """Provide standard MPC parameters for testing."""
    return {
        "prediction_horizon": 15,
        "control_horizon": 12,
        "dt": 0.06,
        "q_position": 1000.0,
        "q_velocity": 1750.0,
        "q_angle": 1000.0,
        "q_angular_velocity": 1500.0,
        "r_thrust": 1.0,
        "r_switch": 0.0,
        "max_velocity": 0.15,
        "max_angular_velocity": np.pi / 2,
        "position_bounds": 3.0,
        "solver_time_limit": 0.05,
    }


# ============================================================================
# State Fixtures
# ============================================================================


@pytest.fixture
def zero_state():
    """Provide a zero state vector."""
    from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

    state = np.zeros(13)
    state[3:7] = euler_xyz_to_quat_wxyz((0.0, 0.0, 0.0))
    return state


@pytest.fixture
def sample_state():
    """Provide a sample non-zero state."""
    from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

    state = np.zeros(13)
    state[0:3] = np.array([1.0, 0.5, 0.2])
    state[3:7] = euler_xyz_to_quat_wxyz((0.1, -0.2, np.pi / 4))
    state[7:10] = np.array([0.1, 0.05, -0.02])
    state[10:13] = np.array([0.0, 0.0, 0.1])
    return state


@pytest.fixture
def target_state():
    """Provide a typical target state."""
    from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

    state = np.zeros(13)
    state[3:7] = euler_xyz_to_quat_wxyz((0.0, 0.0, 0.0))
    return state


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_motion_capture():
    """Provide mock motion capture data."""
    data = MagicMock()
    data.position = (0.0, 0.0)
    data.orientation = (0.0, 0.0, 0.0)
    data.timestamp = 0.0
    return data


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def thruster_command_valid():
    """Provide a valid thruster command."""
    return np.array([1, 0, 1, 0, 1, 0, 1, 0])


@pytest.fixture
def thruster_command_all_on():
    """Provide an all-thrusters-on command."""
    return np.array([1, 1, 1, 1, 1, 1, 1, 1])


@pytest.fixture
def thruster_command_all_off():
    """Provide an all-thrusters-off command."""
    return np.array([0, 0, 0, 0, 0, 0, 0, 0])


# ============================================================================
# File/Path Fixtures
# ============================================================================


@pytest.fixture
def temp_data_dir(tmp_path):
    """Provide a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_csv_data(temp_data_dir):
    """Create a sample CSV file for testing visualization."""
    import pandas as pd

    data = {
        "Step": [0, 1, 2],
        "Control_Time": [0.0, 0.06, 0.12],
        "Current_X": [0.0, 0.1, 0.2],
        "Current_Y": [0.0, 0.05, 0.1],
        "Current_Yaw": [0.0, 0.1, 0.2],
        "Target_X": [1.0, 1.0, 1.0],
        "Target_Y": [1.0, 1.0, 1.0],
        "Target_Yaw": [0.0, 0.0, 0.0],
        "Command_Vector": ["[0,0,0,0,0,0,0,0]"] * 3,
    }

    df = pd.DataFrame(data)
    csv_path = temp_data_dir / "test_data.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "hardware: mark test as requiring hardware")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to tests in test_unit_* files
        if "test_unit_" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Add 'integration' marker to tests in test_integration_* files
        if "test_integration_" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add 'hardware' marker to hardware tests
        if "hardware" in item.nodeid.lower():
            item.add_marker(pytest.mark.hardware)


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")
