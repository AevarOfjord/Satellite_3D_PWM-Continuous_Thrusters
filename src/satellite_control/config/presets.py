"""
Configuration Presets for Satellite Control System

Provides pre-configured settings optimized for different use cases:
- FAST: Aggressive control, faster movement, less stable
- BALANCED: Default balanced configuration
- STABLE: Conservative control, slower but more stable
- PRECISION: High precision, slower movement, very stable

Usage:
    from src.satellite_control.config.presets import ConfigPreset, load_preset
    
    # Load a preset
    config = load_preset(ConfigPreset.FAST)
    
    # Use in simulation
    sim = SatelliteMPCLinearizedSimulation(config_overrides=config)
"""

from typing import Dict, Any

from .models import AppConfig, MPCParams, SatellitePhysicalParams, SimulationParams
from . import physics, mpc_params, timing, constants


class ConfigPreset:
    """Configuration preset names."""
    
    FAST = "fast"
    BALANCED = "balanced"
    STABLE = "stable"
    PRECISION = "precision"
    
    @classmethod
    def all(cls) -> list[str]:
        """Get all available preset names."""
        return [cls.FAST, cls.BALANCED, cls.STABLE, cls.PRECISION]


def _get_base_physics() -> SatellitePhysicalParams:
    """Get base physics parameters (shared across presets)."""
    return SatellitePhysicalParams(
        total_mass=physics.TOTAL_MASS,
        moment_of_inertia=physics.MOMENT_OF_INERTIA,
        satellite_size=physics.SATELLITE_SIZE,
        com_offset=tuple(physics.COM_OFFSET),
        thruster_positions=physics.THRUSTER_POSITIONS,
        thruster_directions={k: tuple(v) for k, v in physics.THRUSTER_DIRECTIONS.items()},
        thruster_forces=physics.THRUSTER_FORCES,
        use_realistic_physics=False,
        damping_linear=0.0,
        damping_angular=0.0,
    )


def _get_base_simulation() -> SimulationParams:
    """Get base simulation parameters (shared across presets)."""
    return SimulationParams(
        dt=0.005,
        max_duration=timing.MAX_SIMULATION_TIME,
        headless=constants.Constants.HEADLESS_MODE,
        window_width=constants.Constants.WINDOW_WIDTH,
        window_height=constants.Constants.WINDOW_HEIGHT,
    )


def _create_fast_mpc() -> MPCParams:
    """
    Create FAST preset MPC parameters.
    
    Characteristics:
    - Higher position/angle weights for aggressive tracking
    - Lower velocity weights to allow faster movement
    - Lower thrust penalty to allow more aggressive control
    - Higher max velocities
    """
    return MPCParams(
        prediction_horizon=40,  # Slightly shorter for speed
        control_horizon=40,
        dt=timing.CONTROL_DT,
        solver_time_limit=mpc_params.MPC_SOLVER_TIME_LIMIT,
        solver_type=mpc_params.MPC_SOLVER_TYPE,
        q_position=2000.0,  # Higher position weight
        q_velocity=5000.0,  # Lower velocity weight (allows faster movement)
        q_angle=2000.0,  # Higher angle weight
        q_angular_velocity=1000.0,  # Lower angular velocity weight
        r_thrust=0.5,  # Lower thrust penalty (more aggressive)
        max_velocity=0.8,  # Higher max velocity
        max_angular_velocity=2.0,  # Higher max angular velocity
        position_bounds=mpc_params.POSITION_BOUNDS,
        damping_zone=0.15,  # Smaller damping zone
        velocity_threshold=0.05,
        max_velocity_weight=500.0,  # Lower velocity weight near target
        thruster_type=mpc_params.THRUSTER_TYPE,
    )


def _create_balanced_mpc() -> MPCParams:
    """
    Create BALANCED preset MPC parameters (default).
    
    Characteristics:
    - Balanced weights for good performance
    - Moderate velocities
    - Standard settings
    """
    return MPCParams(
        prediction_horizon=mpc_params.MPC_PREDICTION_HORIZON,
        control_horizon=mpc_params.MPC_CONTROL_HORIZON,
        dt=timing.CONTROL_DT,
        solver_time_limit=mpc_params.MPC_SOLVER_TIME_LIMIT,
        solver_type=mpc_params.MPC_SOLVER_TYPE,
        q_position=mpc_params.Q_POSITION,
        q_velocity=mpc_params.Q_VELOCITY,
        q_angle=mpc_params.Q_ANGLE,
        q_angular_velocity=mpc_params.Q_ANGULAR_VELOCITY,
        r_thrust=mpc_params.R_THRUST,
        max_velocity=mpc_params.MAX_VELOCITY,
        max_angular_velocity=mpc_params.MAX_ANGULAR_VELOCITY,
        position_bounds=mpc_params.POSITION_BOUNDS,
        damping_zone=mpc_params.DAMPING_ZONE,
        velocity_threshold=mpc_params.VELOCITY_THRESHOLD,
        max_velocity_weight=mpc_params.MAX_VELOCITY_WEIGHT,
        thruster_type=mpc_params.THRUSTER_TYPE,
    )


def _create_stable_mpc() -> MPCParams:
    """
    Create STABLE preset MPC parameters.
    
    Characteristics:
    - Higher velocity weights for smoother movement
    - Higher thrust penalty to reduce aggressive control
    - Lower max velocities for stability
    - Larger damping zone for smoother approach
    """
    return MPCParams(
        prediction_horizon=50,  # Longer horizon for stability
        control_horizon=50,
        dt=timing.CONTROL_DT,
        solver_time_limit=mpc_params.MPC_SOLVER_TIME_LIMIT,
        solver_type=mpc_params.MPC_SOLVER_TYPE,
        q_position=1000.0,  # Standard position weight
        q_velocity=15000.0,  # Higher velocity weight (smoother)
        q_angle=1000.0,  # Standard angle weight
        q_angular_velocity=2000.0,  # Higher angular velocity weight
        r_thrust=2.0,  # Higher thrust penalty (less aggressive)
        max_velocity=0.3,  # Lower max velocity
        max_angular_velocity=1.0,  # Lower max angular velocity
        position_bounds=mpc_params.POSITION_BOUNDS,
        damping_zone=0.4,  # Larger damping zone
        velocity_threshold=0.02,
        max_velocity_weight=2000.0,  # Higher velocity weight near target
        thruster_type=mpc_params.THRUSTER_TYPE,
    )


def _create_precision_mpc() -> MPCParams:
    """
    Create PRECISION preset MPC parameters.
    
    Characteristics:
    - Very high position/angle weights for precision
    - Very high velocity weights for smooth movement
    - Very low max velocities for precise control
    - Large damping zone for smooth approach
    """
    return MPCParams(
        prediction_horizon=60,  # Longer horizon for precision
        control_horizon=60,
        dt=timing.CONTROL_DT,
        solver_time_limit=mpc_params.MPC_SOLVER_TIME_LIMIT,
        solver_type=mpc_params.MPC_SOLVER_TYPE,
        q_position=5000.0,  # Very high position weight
        q_velocity=20000.0,  # Very high velocity weight
        q_angle=5000.0,  # Very high angle weight
        q_angular_velocity=3000.0,  # Very high angular velocity weight
        r_thrust=3.0,  # High thrust penalty (conservative)
        max_velocity=0.2,  # Very low max velocity
        max_angular_velocity=0.5,  # Very low max angular velocity
        position_bounds=mpc_params.POSITION_BOUNDS,
        damping_zone=0.5,  # Very large damping zone
        velocity_threshold=0.01,
        max_velocity_weight=5000.0,  # Very high velocity weight near target
        thruster_type=mpc_params.THRUSTER_TYPE,
    )


def load_preset(preset_name: str) -> Dict[str, Any]:
    """
    Load a configuration preset.
    
    Args:
        preset_name: Name of preset (fast, balanced, stable, precision)
        
    Returns:
        Dictionary of configuration overrides compatible with AppConfig
        
    Raises:
        ValueError: If preset name is invalid
        
    Example:
        config = load_preset(ConfigPreset.FAST)
        sim = SatelliteMPCLinearizedSimulation(config_overrides=config)
    """
    preset_name = preset_name.lower()
    
    if preset_name == ConfigPreset.FAST:
        mpc = _create_fast_mpc()
    elif preset_name == ConfigPreset.BALANCED:
        mpc = _create_balanced_mpc()
    elif preset_name == ConfigPreset.STABLE:
        mpc = _create_stable_mpc()
    elif preset_name == ConfigPreset.PRECISION:
        mpc = _create_precision_mpc()
    else:
        available = ", ".join(ConfigPreset.all())
        raise ValueError(
            f"Invalid preset name '{preset_name}'. "
            f"Available presets: {available}"
        )
    
    # Convert MPC params to dict for config_overrides
    return {
        "mpc": {
            "prediction_horizon": mpc.prediction_horizon,
            "control_horizon": mpc.control_horizon,
            "q_position": mpc.q_position,
            "q_velocity": mpc.q_velocity,
            "q_angle": mpc.q_angle,
            "q_angular_velocity": mpc.q_angular_velocity,
            "r_thrust": mpc.r_thrust,
            "max_velocity": mpc.max_velocity,
            "max_angular_velocity": mpc.max_angular_velocity,
            "damping_zone": mpc.damping_zone,
            "velocity_threshold": mpc.velocity_threshold,
            "max_velocity_weight": mpc.max_velocity_weight,
        }
    }


def get_preset_description(preset_name: str) -> str:
    """
    Get a description of a preset.
    
    Args:
        preset_name: Name of preset
        
    Returns:
        Human-readable description
    """
    descriptions = {
        ConfigPreset.FAST: (
            "Fast preset: Aggressive control for rapid movement. "
            "Higher position/angle weights, lower velocity weights, "
            "higher max velocities. Less stable but faster."
        ),
        ConfigPreset.BALANCED: (
            "Balanced preset: Default configuration with good balance "
            "between speed and stability. Recommended for most use cases."
        ),
        ConfigPreset.STABLE: (
            "Stable preset: Conservative control for smooth, stable movement. "
            "Higher velocity weights, lower max velocities, larger damping zone. "
            "Slower but more stable."
        ),
        ConfigPreset.PRECISION: (
            "Precision preset: High precision control for precise positioning. "
            "Very high weights, very low max velocities, large damping zone. "
            "Slowest but most precise and stable."
        ),
    }
    
    preset_name = preset_name.lower()
    if preset_name not in descriptions:
        available = ", ".join(ConfigPreset.all())
        raise ValueError(
            f"Invalid preset name '{preset_name}'. "
            f"Available presets: {available}"
        )
    
    return descriptions[preset_name]


def list_presets() -> Dict[str, str]:
    """
    List all available presets with descriptions.
    
    Returns:
        Dictionary mapping preset names to descriptions
    """
    return {
        preset: get_preset_description(preset)
        for preset in ConfigPreset.all()
    }
