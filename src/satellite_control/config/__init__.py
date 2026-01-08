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

Usage (V4.0.0+):
    from config import SimulationConfig
    
    # Create and use immutable configuration
    config = SimulationConfig.create_default()
    mass = config.app_config.physics.total_mass

Legacy (deprecated):
    from config import SatelliteConfig  # DEPRECATED in V4.0.0
    # Use SimulationConfig instead
"""

from .constants import Constants
from .mission_state import MissionState
from .mpc_params import MPCConfig, get_mpc_params
from .obstacles import ObstacleManager
from .physics import PhysicsConfig, get_physics_params
from .presets import (
    ConfigPreset,
    get_preset_description,
    list_presets,
    load_preset,
)
# V4.0.0: SatelliteConfig is deprecated - use SimulationConfig instead
# Kept for internal compatibility (SatelliteConfigAdapter) only
from .satellite_config import (
    SatelliteConfig,  # DEPRECATED: Use SimulationConfig instead
    SatelliteConfigAdapter,  # Internal compatibility adapter
    StructuredConfig,
    build_structured_config,
    use_structured_config,
)
from .simulation_config import SimulationConfig
from .timing import TimingConfig, get_timing_params
from .validator import ConfigValidator, validate_config_at_startup

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
    "SatelliteConfig",  # DEPRECATED in V4.0.0: Use SimulationConfig instead
    "SatelliteConfigAdapter",  # Internal compatibility adapter (not for public use)
    "build_structured_config",  # For testing
    "StructuredConfig",  # Structured configuration
    "use_structured_config",  # Configuration context manager
    "ConfigValidator",  # Configuration validator
    "validate_config_at_startup",  # Startup validation function
    "ConfigPreset",  # Configuration preset names
    "get_preset_description",  # Get preset description
    "list_presets",  # List all presets
    "load_preset",  # Load preset configuration
    "SimulationConfig",  # Immutable simulation configuration container (V4.0.0+)
]
