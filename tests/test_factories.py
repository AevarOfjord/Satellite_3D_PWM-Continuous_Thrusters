"""
Test Factories for Satellite Control System

Provides factory methods for creating pre-configured test objects,
reducing the need for extensive mocking in integration tests.
"""

from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np


class TestSimulationFactory:
    """
    Factory for creating test simulation instances with minimal mocking.

    Provides pre-configured simulations for common test scenarios.

    Usage:
        factory = TestSimulationFactory()

        # Create a simple headless simulation
        sim = factory.create_headless_simulation()

        # Create simulation with custom config
        sim = factory.create_simulation(
            start_pos=(0.5, 0.5),
            target_pos=(0.0, 0.0),
        )
    """

    # Default configuration for fast testing
    DEFAULT_TEST_CONFIG = {
        "mpc": {
            "prediction_horizon": 10,
            "control_horizon": 8,
            "solver_time_limit": 0.05,
        },
        "simulation": {
            "max_simulation_time": 10.0,
            "control_dt": 0.25,
        },
        "physics": {
            "USE_REALISTIC_PHYSICS": False,
        },
    }

    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize factory with optional default config overrides.

        Args:
            config_overrides: Default overrides to apply to all simulations
        """
        self.default_overrides = config_overrides or {}

    def create_simulation(
        self,
        start_pos: Tuple[float, float] = (0.5, 0.5),
        target_pos: Tuple[float, float] = (0.0, 0.0),
        start_angle: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_angle: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        config_overrides: Optional[Dict[str, Any]] = None,
        use_mujoco_viewer: bool = False,
    ):
        """
        Create a simulation instance for testing.

        Args:
            start_pos: Starting position (x, y) in meters
            target_pos: Target position (x, y) in meters
            start_angle: Starting orientation (roll, pitch, yaw) in radians
            target_angle: Target orientation (roll, pitch, yaw) in radians
            config_overrides: Additional config overrides
            use_mujoco_viewer: Whether to use MuJoCo viewer (default: False)

        Returns:
            Configured SatelliteMPCLinearizedSimulation instance
        """
        from src.satellite_control.core.simulation import (
            SatelliteMPCLinearizedSimulation,
        )

        # Merge configs: defaults < factory defaults < call-specific
        merged_config = self._merge_configs(
            self.DEFAULT_TEST_CONFIG,
            self.default_overrides,
            config_overrides or {},
        )

        return SatelliteMPCLinearizedSimulation(
            start_pos=start_pos,
            target_pos=target_pos,
            start_angle=start_angle,
            target_angle=target_angle,
            config_overrides=merged_config,
            use_mujoco_viewer=use_mujoco_viewer,
        )

    def create_headless_simulation(
        self,
        start_pos: Tuple[float, float] = (0.5, 0.5),
        target_pos: Tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        """
        Create a headless simulation for fast testing.

        Args:
            start_pos: Starting position
            target_pos: Target position
            **kwargs: Additional arguments passed to create_simulation

        Returns:
            Headless simulation instance
        """
        return self.create_simulation(
            start_pos=start_pos,
            target_pos=target_pos,
            use_mujoco_viewer=False,
            **kwargs,
        )

    def create_mock_mpc_controller(
        self,
        default_action: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> MagicMock:
        """
        Create a mock MPC controller for testing.

        Args:
            default_action: Default control action (zeros if not specified)
            info: Default info dict

        Returns:
            MagicMock configured as MPC controller
        """
        mock_mpc = MagicMock()

        action = default_action if default_action is not None else np.zeros(8)
        default_info = info or {
            "status": 2,
            "status_name": "OPTIMAL",
            "solve_time": 0.01,
        }

        mock_mpc.get_control_action.return_value = (action, default_info)
        mock_mpc.N = 10
        mock_mpc.dt = 0.05
        mock_mpc.max_velocity = 0.5
        mock_mpc.max_angular_velocity = 1.0
        mock_mpc.position_bounds = 2.0

        return mock_mpc

    def create_mock_satellite(self) -> MagicMock:
        """
        Create a mock satellite for testing.

        Returns:
            MagicMock configured as satellite physics simulator
        """
        mock_sat = MagicMock()
        mock_sat.position = np.array([0.5, 0.5, 0.0])
        mock_sat.velocity = np.array([0.0, 0.0, 0.0])
        mock_sat.angle = (0.0, 0.0, 0.0)
        mock_sat.angular_velocity = 0.0
        mock_sat.dt = 0.005
        mock_sat.external_simulation_mode = True

        return mock_sat

    @staticmethod
    def _merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge multiple config dictionaries.

        """
        result: Dict[str, Any] = {}
        for config in configs:
            for section, values in config.items():
                if section not in result:
                    result[section] = {}
                if isinstance(values, dict):
                    result[section].update(values)
                else:
                    result[section] = values
        return result


class MockMPCContext:
    """
    Context manager for mocking MPC in simulations.

    Usage:
        with MockMPCContext(sim) as mock_mpc:
            mock_mpc.return_action(np.array([1, 0, 1, 0, 0, 0, 0, 0]))
            sim.update_mpc_control()
    """

    def __init__(self, simulation):
        self.simulation = simulation
        self.mock_mpc = None
        self._patcher = None

    def __enter__(self):
        self._patcher = patch.object(
            self.simulation.mpc_controller,
            "get_control_action",
        )
        self.mock_mpc = self._patcher.__enter__()

        # Set default return value
        self.mock_mpc.return_value = (
            np.zeros(8),
            {"status": 2, "status_name": "OPTIMAL", "solve_time": 0.01},
        )

        return self

    def __exit__(self, *args):
        if self._patcher:
            self._patcher.__exit__(*args)

    def return_action(
        self,
        action: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ):
        """Set the action that MPC will return."""
        default_info = info or {
            "status": 2,
            "status_name": "OPTIMAL",
            "solve_time": 0.01,
        }
        assert self.mock_mpc is not None
        self.mock_mpc.return_value = (action, default_info)

    def return_failure(self, status: int = -1, status_name: str = "INFEASIBLE"):
        """Configure MPC to return a failure."""
        assert self.mock_mpc is not None
        self.mock_mpc.return_value = (
            None,
            {"status": status, "status_name": status_name, "solve_time": 0.01},
        )


# Convenience function for quick test setup
def create_test_simulation(**kwargs):
    """
    Quick factory function for creating test simulations.

    Args:
        **kwargs: Arguments passed to TestSimulationFactory.create_simulation

    Returns:
        Configured simulation instance
    """
    factory = TestSimulationFactory()
    return factory.create_headless_simulation(**kwargs)
