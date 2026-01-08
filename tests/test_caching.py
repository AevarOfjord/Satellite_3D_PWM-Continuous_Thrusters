"""
Tests for caching utilities.

Tests caching decorators and utilities.
"""

import pytest

from src.satellite_control.config.models import AppConfig, MPCParams, SatellitePhysicalParams, SimulationParams
from src.satellite_control.utils.caching import (
    cache_by_config,
    cache_key_from_config,
    cached,
    cache_with_stats,
)


class TestCached:
    """Tests for @cached decorator."""

    def test_basic_caching(self):
        """@cached should cache function results."""
        call_count = [0]

        @cached(maxsize=128)
        def expensive_function(x):
            call_count[0] += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count[0] == 1  # Should only be called once

    def test_different_arguments(self):
        """@cached should cache different arguments separately."""
        call_count = [0]

        @cached(maxsize=128)
        def expensive_function(x):
            call_count[0] += 1
            return x * 2

        expensive_function(5)
        expensive_function(10)

        assert call_count[0] == 2  # Different arguments, both called

    def test_cache_info(self):
        """@cached should provide cache_info."""
        @cached(maxsize=128)
        def expensive_function(x):
            return x * 2

        expensive_function(5)
        info = expensive_function.cache_info()

        assert info.hits == 0
        assert info.misses == 1

        expensive_function(5)
        info = expensive_function.cache_info()
        assert info.hits == 1
        assert info.misses == 1

    def test_cache_clear(self):
        """@cached should support cache_clear."""
        call_count = [0]

        @cached(maxsize=128)
        def expensive_function(x):
            call_count[0] += 1
            return x * 2

        expensive_function(5)
        expensive_function.cache_clear()
        expensive_function(5)

        assert call_count[0] == 2  # Cache cleared, called again


class TestCacheByConfig:
    """Tests for @cache_by_config decorator."""

    def test_config_based_caching(self):
        """@cache_by_config should cache based on config hash."""
        call_count = [0]

        @cache_by_config(maxsize=10)
        def build_matrices(config, horizon):
            call_count[0] += 1
            return {"config": config, "horizon": horizon}

        # Create test config
        physics = SatellitePhysicalParams(
            total_mass=10.0,
            moment_of_inertia=1.0,
            satellite_size=0.5,
            thruster_positions={i: (0.0, 0.0, 0.0) for i in range(1, 13)},
            thruster_directions={i: (1.0, 0.0, 0.0) for i in range(1, 13)},
            thruster_forces={i: 1.0 for i in range(1, 13)},
        )
        mpc = MPCParams(
            prediction_horizon=50,
            control_horizon=50,
            dt=0.06,
            solver_time_limit=0.05,
            solver_type="OSQP",
            q_position=1000.0,
            q_velocity=10000.0,
            q_angle=1000.0,
            q_angular_velocity=1500.0,
            r_thrust=1.0,
            max_velocity=0.5,
            max_angular_velocity=1.57,
            position_bounds=3.0,
            damping_zone=0.25,
            velocity_threshold=0.03,
            max_velocity_weight=1000.0,
            thruster_type="PWM",
        )
        sim = SimulationParams(dt=0.005, max_duration=60.0)
        config = AppConfig(physics=physics, mpc=mpc, simulation=sim)

        result1 = build_matrices(config, 50)
        result2 = build_matrices(config, 50)

        assert call_count[0] == 1  # Should only be called once
        assert result1 == result2

    def test_cache_info(self):
        """@cache_by_config should provide cache_info."""
        @cache_by_config(maxsize=10)
        def build_matrices(config, horizon):
            return {"config": config, "horizon": horizon}

        # Create minimal config for testing
        physics = SatellitePhysicalParams(
            total_mass=10.0,
            moment_of_inertia=1.0,
            satellite_size=0.5,
            thruster_positions={i: (0.0, 0.0, 0.0) for i in range(1, 13)},
            thruster_directions={i: (1.0, 0.0, 0.0) for i in range(1, 13)},
            thruster_forces={i: 1.0 for i in range(1, 13)},
        )
        mpc = MPCParams(
            prediction_horizon=50,
            control_horizon=50,
            dt=0.06,
            solver_time_limit=0.05,
            solver_type="OSQP",
            q_position=1000.0,
            q_velocity=10000.0,
            q_angle=1000.0,
            q_angular_velocity=1500.0,
            r_thrust=1.0,
            max_velocity=0.5,
            max_angular_velocity=1.57,
            position_bounds=3.0,
            damping_zone=0.25,
            velocity_threshold=0.03,
            max_velocity_weight=1000.0,
            thruster_type="PWM",
        )
        sim = SimulationParams(dt=0.005, max_duration=60.0)
        config = AppConfig(physics=physics, mpc=mpc, simulation=sim)

        build_matrices(config, 50)
        info = build_matrices.cache_info()

        assert "size" in info
        assert "maxsize" in info


class TestCacheKeyFromConfig:
    """Tests for cache_key_from_config function."""

    def test_pydantic_model(self):
        """cache_key_from_config should work with Pydantic models."""
        physics = SatellitePhysicalParams(
            total_mass=10.0,
            moment_of_inertia=1.0,
            satellite_size=0.5,
            thruster_positions={i: (0.0, 0.0, 0.0) for i in range(1, 13)},
            thruster_directions={i: (1.0, 0.0, 0.0) for i in range(1, 13)},
            thruster_forces={i: 1.0 for i in range(1, 13)},
        )

        key1 = cache_key_from_config(physics)
        key2 = cache_key_from_config(physics)

        assert key1 == key2  # Same config should produce same key
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length

    def test_dictionary(self):
        """cache_key_from_config should work with dictionaries."""
        config_dict = {"mass": 10.0, "inertia": 1.0}

        key1 = cache_key_from_config(config_dict)
        key2 = cache_key_from_config(config_dict)

        assert key1 == key2
        assert isinstance(key1, str)

    def test_different_configs(self):
        """cache_key_from_config should produce different keys for different configs."""
        physics1 = SatellitePhysicalParams(
            total_mass=10.0,
            moment_of_inertia=1.0,
            satellite_size=0.5,
            thruster_positions={i: (0.0, 0.0, 0.0) for i in range(1, 13)},
            thruster_directions={i: (1.0, 0.0, 0.0) for i in range(1, 13)},
            thruster_forces={i: 1.0 for i in range(1, 13)},
        )
        physics2 = SatellitePhysicalParams(
            total_mass=20.0,  # Different mass
            moment_of_inertia=1.0,
            satellite_size=0.5,
            thruster_positions={i: (0.0, 0.0, 0.0) for i in range(1, 13)},
            thruster_directions={i: (1.0, 0.0, 0.0) for i in range(1, 13)},
            thruster_forces={i: 1.0 for i in range(1, 13)},
        )

        key1 = cache_key_from_config(physics1)
        key2 = cache_key_from_config(physics2)

        assert key1 != key2  # Different configs should produce different keys


class TestCacheWithStats:
    """Tests for @cache_with_stats decorator."""

    def test_stats_tracking(self):
        """@cache_with_stats should track cache statistics."""
        @cache_with_stats(maxsize=128)
        def expensive_function(x):
            return x * 2

        expensive_function(5)
        expensive_function(5)  # Cache hit

        stats = expensive_function.cache_stats
        assert stats.hits > 0
        assert stats.misses > 0
        assert stats.hit_rate > 0
