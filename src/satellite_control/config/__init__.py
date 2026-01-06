"""
Configuration Package for Satellite Control System

Structured, modular configuration system organized by functional concern.
Provides centralized access to all system parameters with type safety.

Configuration modules:
- physics: Physical parameters (mass, inertia, thruster configuration)
- timing: Control loop timing and stabilization parameters
- mpc_params: MPC controller configuration and solver settings

- mission_state: Mutable runtime mission state management
- constants: UI, network, and data management constants
- obstacles: Obstacle avoidance configuration

Usage:
    from config import PhysicsConfig, MPCConfig, SatelliteConfig

    # Access structured config
    mass = PhysicsConfig.TOTAL_MASS

    # Or use unified wrapper
    mass = SatelliteConfig.TOTAL_MASS
"""

from .constants import Constants
from .mission_state import MissionState
from .mpc_params import MPCConfig, get_mpc_params
from .obstacles import ObstacleManager
from .physics import PhysicsConfig, get_physics_params
from .satellite_config import (
    SatelliteConfig,
    StructuredConfig,
    build_structured_config,
    use_structured_config,
)
from .timing import TimingConfig, get_timing_params

__all__ = [
    "PhysicsConfig",
    "TimingConfig",
    "MPCConfig",
    "MissionState",
    "Constants",
    "ObstacleManager",
    "get_physics_params",
    "get_timing_params",
    "get_mpc_params",
    "SatelliteConfig",
    "build_structured_config",  # For testing
    "StructuredConfig",  # Structured configuration
    "use_structured_config",  # Configuration context manager
]
