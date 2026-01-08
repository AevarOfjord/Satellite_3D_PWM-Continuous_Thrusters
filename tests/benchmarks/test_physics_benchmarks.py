"""
Performance benchmarks for physics simulation components.

Tests the performance of physics step computations, state updates,
and related operations.
"""

import numpy as np
import pytest

# Check if pytest-benchmark is available
try:
    import pytest_benchmark  # noqa: F401

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
class TestPhysicsBenchmarks:
    """Benchmark tests for physics simulation performance."""

    @pytest.fixture
    def physics_simulator(self):
        """Create a physics simulator instance."""
        from src.satellite_control.core.mujoco_satellite import MuJoCoSatellite

        return MuJoCoSatellite()

    def test_physics_step_time(self, benchmark, physics_simulator):
        """Benchmark physics step computation time.
        
        Target: < 1ms per step for real-time simulation at 200Hz.
        """
        dt = 0.005  # 5ms timestep

        def step():
            physics_simulator.step(dt)
            return physics_simulator.get_state()

        result = benchmark(step)
        assert result is not None

    def test_state_getter_performance(self, benchmark, physics_simulator):
        """Benchmark state retrieval performance."""
        # Initialize with a state
        state = np.zeros(13)
        state[3] = 1.0  # Valid quaternion
        physics_simulator.set_state(state)

        result = benchmark(physics_simulator.get_state)
        assert result is not None
        assert len(result) == 13

    def test_state_setter_performance(self, benchmark, physics_simulator):
        """Benchmark state setting performance."""
        state = np.zeros(13)
        state[3] = 1.0  # Valid quaternion

        result = benchmark(physics_simulator.set_state, state)
        assert result is None  # set_state returns None

    def test_control_application_performance(self, benchmark, physics_simulator):
        """Benchmark control input application."""
        control = np.ones(12) * 0.5  # 50% thrust on all thrusters

        def apply_control():
            physics_simulator.apply_control(control)
            physics_simulator.step(0.005)

        benchmark(apply_control)


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
class TestOrientationBenchmarks:
    """Benchmark tests for orientation conversion utilities."""

    def test_euler_to_quaternion(self, benchmark):
        """Benchmark Euler to quaternion conversion."""
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        euler = (0.1, 0.2, 0.3)

        result = benchmark(euler_xyz_to_quat_wxyz, euler)
        assert result is not None
        assert len(result) == 4

    def test_quaternion_to_euler(self, benchmark):
        """Benchmark quaternion to Euler conversion."""
        from src.satellite_control.utils.orientation_utils import quat_wxyz_to_euler_xyz

        quat = np.array([0.98, 0.1, 0.2, 0.0])  # Normalized quaternion

        result = benchmark(quat_wxyz_to_euler_xyz, quat)
        assert result is not None
        assert len(result) == 3

    def test_quaternion_normalization(self, benchmark):
        """Benchmark quaternion normalization."""
        from src.satellite_control.utils.orientation_utils import normalize_quaternion

        quat = np.array([1.1, 0.2, 0.3, 0.1])  # Unnormalized

        result = benchmark(normalize_quaternion, quat)
        assert result is not None
        assert len(result) == 4
        assert np.isclose(np.linalg.norm(result), 1.0)


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
class TestNavigationBenchmarks:
    """Benchmark tests for navigation utilities."""

    def test_distance_computation(self, benchmark):
        """Benchmark distance computation."""
        from src.satellite_control.utils.navigation_utils import compute_distance

        pos1 = np.array([1.0, 2.0, 3.0])
        pos2 = np.array([4.0, 5.0, 6.0])

        result = benchmark(compute_distance, pos1, pos2)
        assert result > 0

    def test_angle_error_computation(self, benchmark):
        """Benchmark angle error computation."""
        from src.satellite_control.utils.navigation_utils import compute_angle_error

        quat1 = np.array([1.0, 0.0, 0.0, 0.0])
        quat2 = np.array([0.98, 0.0, 0.2, 0.0])

        result = benchmark(compute_angle_error, quat1, quat2)
        assert result >= 0
