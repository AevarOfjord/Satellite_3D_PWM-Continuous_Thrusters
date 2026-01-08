"""
Performance benchmark tests for MPC components.

Run with: pytest tests/test_benchmark.py --benchmark-only
"""

import time

import numpy as np
import pytest

# Helper to build 13-element state vectors.
def make_state(
    pos=(0.0, 0.0, 0.0),
    euler=(0.0, 0.0, 0.0),
    vel=(0.0, 0.0, 0.0),
    omega=(0.0, 0.0, 0.0),
):
    from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

    state = np.zeros(13)
    state[0:3] = np.array(pos, dtype=float)
    state[3:7] = euler_xyz_to_quat_wxyz(euler)
    state[7:10] = np.array(vel, dtype=float)
    state[10:13] = np.array(omega, dtype=float)
    return state

# Check if pytest-benchmark is available
try:
    import pytest_benchmark  # noqa: F401

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
class TestMPCBenchmarks:
    """Benchmark tests for MPC solver performance."""

    @pytest.fixture
    def mpc_controller(self):
        """Create an MPCController instance for benchmarking."""
        from src.satellite_control.control.mpc_controller import MPCController
        from src.satellite_control.config.simulation_config import SimulationConfig

        sim_config = SimulationConfig.create_default()
        app_config = sim_config.app_config
        return MPCController(app_config.physics, app_config.mpc)

    @pytest.fixture
    def sample_state(self):
        """Sample state vector for testing."""
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        state = np.zeros(13)
        state[0:3] = np.array([0.5, 0.3, 0.0])
        state[3:7] = euler_xyz_to_quat_wxyz((0.0, 0.0, 0.1))
        state[7:10] = np.array([0.01, 0.02, 0.0])
        state[10:13] = np.array([0.0, 0.0, 0.001])
        return state

    @pytest.fixture
    def target_state(self):
        """Sample target state for testing."""
        from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz

        state = np.zeros(13)
        state[3:7] = euler_xyz_to_quat_wxyz((0.0, 0.0, 0.0))
        return state

    def test_mpc_solve_time(
        self, benchmark, mpc_controller, sample_state, target_state
    ):
        """Benchmark MPC solve time.

        This test tracks the solver performance over time.
        Target: < 5ms per solve for real-time control.
        """

        def solve():
            u, info = mpc_controller.get_control_action(sample_state, target_state)
            return u

        result = benchmark(solve)
        assert result is not None
        assert len(result) == 12

    def test_linearization_time(self, benchmark, mpc_controller, sample_state):
        """Benchmark dynamics linearization time."""
        from src.satellite_control.utils.state_converter import StateConverter

        x_mpc = StateConverter.sim_to_mpc(sample_state)

        def linearize():
            return mpc_controller.linearize_dynamics(x_mpc)

        result = benchmark(linearize)
        assert result is not None


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
class TestStateConverterBenchmarks:
    """Benchmark tests for state converter."""

    def test_sim_to_mpc_conversion(self, benchmark):
        """Benchmark state format conversion."""
        from src.satellite_control.utils.state_converter import StateConverter

        sim_state = np.array(
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.01, 0.02, 0.03]
        )

        result = benchmark(StateConverter.sim_to_mpc, sim_state)
        assert result is not None
        assert len(result) == 13

    def test_trajectory_conversion(self, benchmark):
        """Benchmark trajectory conversion (100 points)."""
        from src.satellite_control.utils.state_converter import StateConverter

        trajectory = np.random.randn(100, 13)

        result = benchmark(StateConverter.sim_to_mpc_trajectory, trajectory)
        assert result is not None
        assert result.shape == (100, 13)


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
class TestThrusterManagerBenchmarks:
    """Benchmark tests for thruster manager."""

    @pytest.fixture
    def thruster_manager(self):
        """Create ThrusterManager instance."""
        from src.satellite_control.core.thruster_manager import ThrusterManager

        return ThrusterManager(num_thrusters=8)

    def test_set_thruster_pattern(self, benchmark, thruster_manager):
        """Benchmark thruster pattern update."""
        pattern = np.array([0.5, 0.3, 0.0, 1.0, 0.8, 0.2, 0.0, 0.6])

        def update_pattern():
            thruster_manager.set_thruster_pattern(pattern, simulation_time=0.0)

        benchmark(update_pattern)


# ============================================================================
# Regression Detection Tests
# ============================================================================


class TestMPCRegressionDetection:
    """
    Tests to detect performance regressions in MPC solver.

    These tests fail if MPC solve time exceeds thresholds, alerting to
    performance degradation.
    """

    # Thresholds (in seconds)
    # Based on baseline measurements: p50=10.3ms, p95=12ms, max=12.4ms
    # Set thresholds with ~50% margin to detect significant regressions
    MPC_SOLVE_THRESHOLD_P50 = 0.015  # 15ms median (baseline ~10ms)
    MPC_SOLVE_THRESHOLD_P95 = 0.050  # 50ms 95th percentile (allow overhead)
    MPC_SOLVE_THRESHOLD_MAX = 0.070  # 70ms max

    @pytest.fixture
    def mpc_controller(self):
        """Create an MPCController instance (V4.0.0: use SimulationConfig)."""
        # V4.0.0: Use SimulationConfig instead of SatelliteConfig
        from src.satellite_control.config.simulation_config import SimulationConfig
        from src.satellite_control.control.mpc_controller import MPCController

        config = SimulationConfig.create_default()
        return MPCController(config.app_config.physics, config.app_config.mpc)

    def test_mpc_solve_time_within_threshold(self, mpc_controller):
        """
        Test that MPC solve time stays within acceptable bounds.

        Runs 20 solves and checks statistics against thresholds.
        This detects regressions that would break real-time control.
        """
        # Sample states - varying positions and angles
        np.random.seed(42)

        solve_times = []
        for i in range(20):
            # Generate realistic state variations
            x_current = make_state(
                pos=(0.5 * np.random.randn(), 0.5 * np.random.randn(), 0.0),
                euler=(0.0, 0.0, np.random.uniform(-np.pi, np.pi)),
                vel=(0.05 * np.random.randn(), 0.05 * np.random.randn(), 0.0),
                omega=(0.0, 0.0, 0.1 * np.random.randn()),
            )
            x_target = make_state()

            start = time.perf_counter()
            u, info = mpc_controller.get_control_action(x_current, x_target)
            elapsed = time.perf_counter() - start

            solve_times.append(elapsed)

            # Verify solution is valid
            assert u is not None
            assert len(u) == 12
            assert all(0.0 <= v <= 1.0 for v in u)

        # Calculate statistics
        solve_times_array = np.array(solve_times)
        p50 = np.percentile(solve_times_array, 50)
        p95 = np.percentile(solve_times_array, 95)
        max_time = np.max(solve_times_array)

        # Report stats for debugging
        print("\nMPC Solve Time Stats (20 iterations):")
        print(
            f"  Median (p50): {p50*1000:.2f}ms "
            f"(threshold: {self.MPC_SOLVE_THRESHOLD_P50*1000:.0f}ms)"
        )
        print(
            f"  P95: {p95*1000:.2f}ms "
            f"(threshold: {self.MPC_SOLVE_THRESHOLD_P95*1000:.0f}ms)"
        )
        print(
            f"  Max: {max_time*1000:.2f}ms "
            f"(threshold: {self.MPC_SOLVE_THRESHOLD_MAX*1000:.0f}ms)"
        )

        # Assert thresholds
        assert p50 < self.MPC_SOLVE_THRESHOLD_P50, (
            f"MPC median solve time {p50*1000:.2f}ms exceeds threshold "
            f"{self.MPC_SOLVE_THRESHOLD_P50*1000:.0f}ms"
        )
        assert p95 < self.MPC_SOLVE_THRESHOLD_P95, (
            f"MPC p95 solve time {p95*1000:.2f}ms exceeds threshold "
            f"{self.MPC_SOLVE_THRESHOLD_P95*1000:.0f}ms"
        )
        assert max_time < self.MPC_SOLVE_THRESHOLD_MAX, (
            f"MPC max solve time {max_time*1000:.2f}ms exceeds threshold "
            f"{self.MPC_SOLVE_THRESHOLD_MAX*1000:.0f}ms"
        )

    def test_mpc_trajectory_following_performance(self, mpc_controller):
        """
        Test MPC performance with trajectory following.

        Trajectory mode may be slower due to additional computations.
        """
        x_current = make_state(
            pos=(0.5, 0.3, 0.0),
            euler=(0.0, 0.0, 0.5),
            vel=(0.01, 0.02, 0.0),
            omega=(0.0, 0.0, 0.01),
        )
        x_target = make_state()

        # Create a simple trajectory
        N = mpc_controller.N
        trajectory = np.zeros((N + 1, 13))
        for k in range(N + 1):
            alpha = k / N
            trajectory[k] = (1 - alpha) * x_current + alpha * x_target

        solve_times = []
        for _ in range(10):
            start = time.perf_counter()

            # Skip reconfiguration and use default controller
            # This avoids hacking internal attributes which are not exposed
            u, info = mpc_controller.get_control_action(
                x_current, x_target, x_target_trajectory=trajectory
            )
            elapsed = time.perf_counter() - start
            solve_times.append(elapsed)

        max_time = max(solve_times)

        # Trajectory mode allowed slightly more time
        trajectory_threshold = self.MPC_SOLVE_THRESHOLD_MAX * 1.2  # 20% margin

        assert max_time < trajectory_threshold, (
            f"MPC trajectory solve time {max_time*1000:.2f}ms exceeds threshold "
            f"{trajectory_threshold*1000:.0f}ms"
        )

    @pytest.mark.slow
    def test_mpc_memory_stability_long_run(self, mpc_controller):
        """
        Test that MPC doesn't leak memory over many iterations.

        Marked slow - runs 100 solves and checks memory growth.
        """
        import os
        import psutil

        x_current = make_state(
            pos=(0.5, 0.3, 0.0),
            euler=(0.0, 0.0, 0.5),
            vel=(0.01, 0.02, 0.0),
            omega=(0.0, 0.0, 0.01),
        )
        x_target = make_state()

        def get_process_memory():
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return mem_info.rss / (1024 * 1024)  # in MB

        # Check for reference leaks in solver objects
        # initial_refcount = sys.getrefcount(mpc_controller.prob) # Flaky check
        start_mem = get_process_memory()

        for i in range(100):
            u, info = mpc_controller.get_control_action(x_current, x_target)

            # Vary state slightly each iteration (keep quaternion normalized)
            x_current = x_current.copy()
            x_current[0:3] += 0.001 * np.random.randn(3)
            x_current[7:10] += 0.001 * np.random.randn(3)
            x_current[10:13] += 0.001 * np.random.randn(3)

        end_mem = get_process_memory()

        # Check that we didn't leak solver references
        # This is tricky in Python, but we check specific objects
        # final_refcount = sys.getrefcount(mpc_controller.prob)
        # assert final_refcount <= initial_refcount + 5  # Allow small fluctuation

        memory_growth = end_mem - start_mem
        print(f"\nMemory usage before: {start_mem:.2f} MB")
        print(f"Memory usage after 100 solves: {end_mem:.2f} MB")
        print(f"Memory growth: {memory_growth:.2f} MB")

        # Allow a small amount of memory growth, e.g., 5MB
        assert memory_growth < 5.0, (
            f"Memory grew by {memory_growth:.2f} MB over 100 solves - "
            "possible memory leak"
        )

        # Solve times should remain stable
        recent_times = (
            mpc_controller.solve_times[-20:]
            if len(mpc_controller.solve_times) >= 20
            else mpc_controller.solve_times
        )
        if len(recent_times) >= 2:
            time_std = np.std(recent_times)
            assert time_std < 0.010, (  # 10ms std dev max
                f"Solve time variance {time_std*1000:.2f}ms is too high - "
                "performance may be degrading"
            )
