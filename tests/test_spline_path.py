"""
Unit tests for SplinePath module.

Tests Bezier spline generation for obstacle avoidance.
"""

import numpy as np
import pytest

from src.satellite_control.utils.spline_path import ObstacleAvoidanceSpline


class TestObstacleAvoidanceSplineInitialization:
    """Test ObstacleAvoidanceSpline initialization."""

    def test_spline_creation(self):
        """Test that spline can be created."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        assert np.allclose(spline.start, [0.0, 0.0])
        assert np.allclose(spline.control, [1.0, 1.0])
        assert np.allclose(spline.end, [2.0, 0.0])
        assert spline.num_samples > 0

    def test_spline_builds_arc_length_table(self):
        """Test that spline builds arc length table on init."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        assert hasattr(spline, "_t_values")
        assert hasattr(spline, "_arc_lengths")
        assert len(spline._t_values) == spline.num_samples
        assert len(spline._arc_lengths) == spline.num_samples


class TestObstacleAvoidanceSplineBezier:
    """Test Bezier curve evaluation."""

    def test_quadratic_bezier_at_start(self):
        """Test Bezier curve at t=0 (start point)."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        point = spline._quadratic_bezier(0.0)

        assert np.allclose(point, spline.start)

    def test_quadratic_bezier_at_end(self):
        """Test Bezier curve at t=1 (end point)."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        point = spline._quadratic_bezier(1.0)

        assert np.allclose(point, spline.end)

    def test_quadratic_bezier_at_midpoint(self):
        """Test Bezier curve at t=0.5."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        point = spline._quadratic_bezier(0.5)

        # Should be somewhere between start and end
        assert point[0] >= 0.0
        assert point[0] <= 2.0

    def test_quadratic_bezier_derivative(self):
        """Test Bezier curve derivative."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        derivative = spline._quadratic_bezier_derivative(0.0)

        # Derivative should be a vector
        assert len(derivative) == 2
        assert np.isfinite(derivative).all()


class TestObstacleAvoidanceSplineArcLength:
    """Test arc length functionality."""

    def test_total_length_property(self):
        """Test that total_length property works."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        assert spline.total_length > 0
        assert np.isfinite(spline.total_length)

    def test_arc_length_increases(self):
        """Test that arc length increases monotonically."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        # Arc lengths should be non-decreasing
        for i in range(1, len(spline._arc_lengths)):
            assert spline._arc_lengths[i] >= spline._arc_lengths[i - 1]

    def test_sample_at_arc_length(self):
        """Test sampling at specific arc length."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        # Sample at start (arc_length = 0)
        point = spline.sample_at_arc_length(0.0)
        assert np.allclose(point, spline.start, atol=0.1)

        # Sample at end
        point = spline.sample_at_arc_length(spline.total_length)
        assert np.allclose(point, spline.end, atol=0.1)

    def test_sample_at_arc_length_midpoint(self):
        """Test sampling at midpoint arc length."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        midpoint_length = spline.total_length / 2
        point = spline.sample_at_arc_length(midpoint_length)

        # Should be a valid point
        assert len(point) == 2
        assert np.isfinite(point).all()


class TestObstacleAvoidanceSplineSampling:
    """Test spline sampling methods."""

    def test_sample_uniform(self):
        """Test uniform sampling along spline."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
            num_samples=10,
        )

        points = spline.sample_uniform(num_points=5)

        assert len(points) == 5
        assert all(len(p) == 2 for p in points)
        # First point should be start
        assert np.allclose(points[0], spline.start, atol=0.1)
        # Last point should be end
        assert np.allclose(points[-1], spline.end, atol=0.1)

    def test_sample_uniform_single_point(self):
        """Test uniform sampling with single point."""
        spline = ObstacleAvoidanceSpline(
            start=(0.0, 0.0),
            control=(1.0, 1.0),
            end=(2.0, 0.0),
        )

        points = spline.sample_uniform(num_points=1)

        assert len(points) == 1
