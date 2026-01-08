"""
Performance Monitoring for Satellite Control System

Collects and analyzes performance metrics during simulation runs.
Provides structured metrics export and performance regression detection.

Usage:
    monitor = PerformanceMonitor()
    
    # During simulation
    monitor.record_mpc_solve(0.002)  # 2ms
    monitor.record_physics_step(0.0001)  # 0.1ms
    
    # At end of simulation
    metrics = monitor.get_metrics()
    metrics.export_to_json("metrics.json")
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a simulation run.
    
    Collects timing data for all major components and provides
    statistical analysis and export capabilities.
    """

    # MPC Performance
    mpc_solve_times: List[float] = field(default_factory=list)
    mpc_solve_count: int = 0
    mpc_timeout_count: int = 0

    # Physics Performance
    physics_step_times: List[float] = field(default_factory=list)
    physics_step_count: int = 0

    # Control Loop Performance
    control_loop_times: List[float] = field(default_factory=list)
    control_loop_count: int = 0
    timing_violations: int = 0

    # Simulation Overall
    total_simulation_time: float = 0.0
    total_steps: int = 0
    simulation_start_time: float = 0.0
    simulation_end_time: float = 0.0

    # Memory (if available)
    peak_memory_mb: Optional[float] = None

    def record_mpc_solve(self, solve_time: float, timeout: bool = False) -> None:
        """Record an MPC solve time."""
        self.mpc_solve_times.append(solve_time)
        self.mpc_solve_count += 1
        if timeout:
            self.mpc_timeout_count += 1

    def record_physics_step(self, step_time: float) -> None:
        """Record a physics step time."""
        self.physics_step_times.append(step_time)
        self.physics_step_count += 1

    def record_control_loop(self, loop_time: float, timing_violation: bool = False) -> None:
        """Record a control loop time."""
        self.control_loop_times.append(loop_time)
        self.control_loop_count += 1
        if timing_violation:
            self.timing_violations += 1

    def start_simulation(self) -> None:
        """Mark simulation start."""
        self.simulation_start_time = time.perf_counter()

    def end_simulation(self) -> None:
        """Mark simulation end and calculate total time."""
        self.simulation_end_time = time.perf_counter()
        self.total_simulation_time = self.simulation_end_time - self.simulation_start_time

    @property
    def mpc_p50_ms(self) -> float:
        """MPC solve time 50th percentile in milliseconds."""
        if not self.mpc_solve_times:
            return 0.0
        return np.percentile(self.mpc_solve_times, 50) * 1000

    @property
    def mpc_p95_ms(self) -> float:
        """MPC solve time 95th percentile in milliseconds."""
        if not self.mpc_solve_times:
            return 0.0
        return np.percentile(self.mpc_solve_times, 95) * 1000

    @property
    def mpc_p99_ms(self) -> float:
        """MPC solve time 99th percentile in milliseconds."""
        if not self.mpc_solve_times:
            return 0.0
        return np.percentile(self.mpc_solve_times, 99) * 1000

    @property
    def mpc_mean_ms(self) -> float:
        """MPC solve time mean in milliseconds."""
        if not self.mpc_solve_times:
            return 0.0
        return np.mean(self.mpc_solve_times) * 1000

    @property
    def mpc_max_ms(self) -> float:
        """MPC solve time maximum in milliseconds."""
        if not self.mpc_solve_times:
            return 0.0
        return np.max(self.mpc_solve_times) * 1000

    @property
    def physics_avg_ms(self) -> float:
        """Average physics step time in milliseconds."""
        if not self.physics_step_times:
            return 0.0
        return np.mean(self.physics_step_times) * 1000

    @property
    def physics_max_ms(self) -> float:
        """Maximum physics step time in milliseconds."""
        if not self.physics_step_times:
            return 0.0
        return np.max(self.physics_step_times) * 1000

    @property
    def control_loop_avg_ms(self) -> float:
        """Average control loop time in milliseconds."""
        if not self.control_loop_times:
            return 0.0
        return np.mean(self.control_loop_times) * 1000

    @property
    def simulation_fps(self) -> float:
        """Simulation steps per second."""
        if self.total_simulation_time == 0:
            return 0.0
        return self.total_steps / self.total_simulation_time

    @property
    def mpc_timeout_rate(self) -> float:
        """MPC timeout rate (0.0 to 1.0)."""
        if self.mpc_solve_count == 0:
            return 0.0
        return self.mpc_timeout_count / self.mpc_solve_count

    @property
    def timing_violation_rate(self) -> float:
        """Control loop timing violation rate (0.0 to 1.0)."""
        if self.control_loop_count == 0:
            return 0.0
        return self.timing_violations / self.control_loop_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON export."""
        return {
            "mpc": {
                "solve_count": self.mpc_solve_count,
                "timeout_count": self.mpc_timeout_count,
                "timeout_rate": self.mpc_timeout_rate,
                "mean_ms": self.mpc_mean_ms,
                "p50_ms": self.mpc_p50_ms,
                "p95_ms": self.mpc_p95_ms,
                "p99_ms": self.mpc_p99_ms,
                "max_ms": self.mpc_max_ms,
            },
            "physics": {
                "step_count": self.physics_step_count,
                "avg_ms": self.physics_avg_ms,
                "max_ms": self.physics_max_ms,
            },
            "control_loop": {
                "loop_count": self.control_loop_count,
                "timing_violations": self.timing_violations,
                "violation_rate": self.timing_violation_rate,
                "avg_ms": self.control_loop_avg_ms,
            },
            "simulation": {
                "total_time_s": self.total_simulation_time,
                "total_steps": self.total_steps,
                "fps": self.simulation_fps,
                "peak_memory_mb": self.peak_memory_mb,
            },
        }

    def export_to_json(self, file_path: Path) -> None:
        """Export metrics to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_summary_string(self) -> str:
        """Get human-readable performance summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("PERFORMANCE SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Simulation Time: {self.total_simulation_time:.2f}s")
        lines.append(f"Total Steps: {self.total_steps}")
        lines.append(f"Simulation FPS: {self.simulation_fps:.1f}")
        lines.append("")
        lines.append("MPC Performance:")
        lines.append(f"  Solves: {self.mpc_solve_count}")
        lines.append(f"  Mean: {self.mpc_mean_ms:.2f}ms")
        lines.append(f"  P50: {self.mpc_p50_ms:.2f}ms")
        lines.append(f"  P95: {self.mpc_p95_ms:.2f}ms")
        lines.append(f"  P99: {self.mpc_p99_ms:.2f}ms")
        lines.append(f"  Max: {self.mpc_max_ms:.2f}ms")
        if self.mpc_timeout_count > 0:
            lines.append(f"  ⚠️  Timeouts: {self.mpc_timeout_count} ({self.mpc_timeout_rate*100:.1f}%)")
        lines.append("")
        lines.append("Physics Performance:")
        lines.append(f"  Steps: {self.physics_step_count}")
        lines.append(f"  Avg: {self.physics_avg_ms:.3f}ms")
        lines.append(f"  Max: {self.physics_max_ms:.3f}ms")
        lines.append("")
        lines.append("Control Loop:")
        lines.append(f"  Loops: {self.control_loop_count}")
        lines.append(f"  Avg: {self.control_loop_avg_ms:.2f}ms")
        if self.timing_violations > 0:
            lines.append(
                f"  ⚠️  Timing Violations: {self.timing_violations} "
                f"({self.timing_violation_rate*100:.1f}%)"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def check_performance_thresholds(
        self,
        mpc_p95_threshold_ms: float = 5.0,
        mpc_timeout_rate_threshold: float = 0.01,
        timing_violation_rate_threshold: float = 0.01,
    ) -> List[str]:
        """
        Check performance against thresholds.
        
        Returns list of warning messages for threshold violations.
        """
        warnings = []

        if self.mpc_p95_ms > mpc_p95_threshold_ms:
            warnings.append(
                f"MPC P95 solve time ({self.mpc_p95_ms:.2f}ms) exceeds threshold "
                f"({mpc_p95_threshold_ms:.2f}ms)"
            )

        if self.mpc_timeout_rate > mpc_timeout_rate_threshold:
            warnings.append(
                f"MPC timeout rate ({self.mpc_timeout_rate*100:.1f}%) exceeds threshold "
                f"({mpc_timeout_rate_threshold*100:.1f}%)"
            )

        if self.timing_violation_rate > timing_violation_rate_threshold:
            warnings.append(
                f"Timing violation rate ({self.timing_violation_rate*100:.1f}%) exceeds threshold "
                f"({timing_violation_rate_threshold*100:.1f}%)"
            )

        return warnings


class PerformanceMonitor:
    """
    Performance monitor for simulation runs.
    
    Collects metrics during simulation and provides analysis.
    """

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.metrics = PerformanceMetrics()
        self.metrics.start_simulation()

    def record_mpc_solve(self, solve_time: float, timeout: bool = False) -> None:
        """Record MPC solve time."""
        self.metrics.record_mpc_solve(solve_time, timeout)

    def record_physics_step(self, step_time: float) -> None:
        """Record physics step time."""
        self.metrics.record_physics_step(step_time)

    def record_control_loop(self, loop_time: float, timing_violation: bool = False) -> None:
        """Record control loop time."""
        self.metrics.record_control_loop(loop_time, timing_violation)

    def increment_step(self) -> None:
        """Increment total step counter."""
        self.metrics.total_steps += 1

    def finish(self) -> PerformanceMetrics:
        """Finish monitoring and return metrics."""
        self.metrics.end_simulation()
        return self.metrics

    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics (may be incomplete if simulation still running)."""
        return self.metrics

    def export_metrics(self, file_path: Path) -> None:
        """Export metrics to JSON file."""
        self.metrics.end_simulation()
        self.metrics.export_to_json(file_path)

    def print_summary(self) -> None:
        """Print performance summary to console."""
        self.metrics.end_simulation()
        print(self.metrics.get_summary_string())

    def check_thresholds(self, **kwargs) -> List[str]:
        """Check performance thresholds and return warnings."""
        return self.metrics.check_performance_thresholds(**kwargs)
