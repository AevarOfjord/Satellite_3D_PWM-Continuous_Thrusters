"""
Tests for performance monitoring utilities.

Tests PerformanceMonitor and PerformanceMetrics classes.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.satellite_control.core.performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_initialization(self):
        """PerformanceMetrics should initialize with default values."""
        metrics = PerformanceMetrics()
        assert metrics.total_simulation_time == 0.0
        assert metrics.total_steps == 0
        assert len(metrics.mpc_solve_times) == 0

    def test_calculate_metrics(self):
        """calculate_metrics should compute statistics."""
        metrics = PerformanceMetrics()
        metrics.total_simulation_time = 10.0
        metrics.total_steps = 100
        metrics.mpc_solve_times = [0.001, 0.002, 0.003, 0.004, 0.005]
        metrics.physics_step_times = [0.0001, 0.0002, 0.0003]
        metrics.control_loop_times = [0.0015, 0.0025, 0.0035]

        metrics.calculate_metrics()

        assert metrics.simulation_fps == 10.0  # 100 steps / 10s
        assert metrics.mpc_mean_solve_time > 0
        assert metrics.mpc_p50_solve_time > 0
        assert metrics.mpc_p95_solve_time > 0
        assert metrics.mpc_p99_solve_time > 0
        assert metrics.mpc_max_solve_time == 0.005
        assert metrics.mpc_min_solve_time == 0.001

    def test_to_dict(self):
        """to_dict should return serializable dictionary."""
        metrics = PerformanceMetrics()
        metrics.total_simulation_time = 10.0
        metrics.total_steps = 100
        metrics.mpc_solve_times = [0.001, 0.002, 0.003]

        metrics.calculate_metrics()
        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert "total_simulation_time_s" in result
        assert "total_steps" in result
        assert "mpc_mean_solve_time_ms" in result
        # Should be JSON serializable
        json.dumps(result)

    def test_print_summary(self, caplog):
        """print_summary should log performance summary."""
        metrics = PerformanceMetrics()
        metrics.total_simulation_time = 10.0
        metrics.total_steps = 100
        metrics.mpc_solve_times = [0.001, 0.002, 0.003]

        metrics.calculate_metrics()
        metrics.print_summary()

        assert len(caplog.records) > 0
        assert any("PERFORMANCE SUMMARY" in record.message for record in caplog.records)

    def test_export_to_json(self, tmp_path):
        """export_to_json should write JSON file."""
        metrics = PerformanceMetrics()
        metrics.total_simulation_time = 10.0
        metrics.total_steps = 100
        metrics.mpc_solve_times = [0.001, 0.002, 0.003]

        metrics.calculate_metrics()

        file_path = tmp_path / "metrics.json"
        metrics.export_to_json(file_path)

        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)
        assert "total_simulation_time_s" in data


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_initialization(self):
        """PerformanceMonitor should initialize."""
        monitor = PerformanceMonitor()
        assert monitor.metrics is not None

    def test_record_mpc_solve(self):
        """record_mpc_solve should record solve time."""
        monitor = PerformanceMonitor()
        monitor.record_mpc_solve(0.001, timeout=False, failed=False)

        assert len(monitor.metrics.mpc_solve_times) == 1
        assert monitor.metrics.mpc_solve_times[0] == 0.001
        assert monitor.metrics.mpc_solve_timeouts == 0
        assert monitor.metrics.mpc_solve_failures == 0

    def test_record_mpc_solve_timeout(self):
        """record_mpc_solve should record timeouts."""
        monitor = PerformanceMonitor()
        monitor.record_mpc_solve(0.1, timeout=True, failed=False)

        assert monitor.metrics.mpc_solve_timeouts == 1
        assert monitor.metrics.mpc_solve_failures == 0

    def test_record_mpc_solve_failure(self):
        """record_mpc_solve should record failures."""
        monitor = PerformanceMonitor()
        monitor.record_mpc_solve(0.1, timeout=False, failed=True)

        assert monitor.metrics.mpc_solve_timeouts == 0
        assert monitor.metrics.mpc_solve_failures == 1

    def test_record_physics_step(self):
        """record_physics_step should record step time."""
        monitor = PerformanceMonitor()
        monitor.record_physics_step(0.0001)

        assert len(monitor.metrics.physics_step_times) == 1
        assert monitor.metrics.physics_step_times[0] == 0.0001

    def test_record_control_loop(self):
        """record_control_loop should record loop time."""
        monitor = PerformanceMonitor()
        monitor.record_control_loop(0.002, timing_violation=False)

        assert len(monitor.metrics.control_loop_times) == 1
        assert monitor.metrics.control_loop_times[0] == 0.002
        assert monitor.metrics.control_timing_violations == 0

    def test_record_control_loop_violation(self):
        """record_control_loop should record timing violations."""
        monitor = PerformanceMonitor()
        monitor.record_control_loop(0.1, timing_violation=True)

        assert monitor.metrics.control_timing_violations == 1

    def test_get_summary(self):
        """get_summary should return metrics dictionary."""
        monitor = PerformanceMonitor()
        monitor.record_mpc_solve(0.001)
        monitor.record_physics_step(0.0001)
        monitor.record_control_loop(0.002)

        summary = monitor.get_summary()
        assert isinstance(summary, dict)
        assert "mpc_solves" in summary

    def test_check_thresholds(self):
        """check_thresholds should detect violations."""
        monitor = PerformanceMonitor()
        monitor.record_mpc_solve(0.1)  # Exceeds default threshold

        violations = monitor.check_thresholds(
            mpc_solve_time_threshold=0.05,
            physics_step_time_threshold=0.001,
            control_loop_time_threshold=0.01,
        )

        assert len(violations) > 0
        assert any("MPC solve time" in v for v in violations)

    def test_reset(self):
        """reset should clear all metrics."""
        monitor = PerformanceMonitor()
        monitor.record_mpc_solve(0.001)
        monitor.record_physics_step(0.0001)
        monitor.reset()

        assert len(monitor.metrics.mpc_solve_times) == 0
        assert len(monitor.metrics.physics_step_times) == 0
        assert len(monitor.metrics.control_loop_times) == 0
