"""
Immutable Simulation Configuration Container

Provides a dependency-injection friendly configuration container that eliminates
the need for mutable global state. This is the foundation for better testability
and thread-safety.

Usage:
    from src.satellite_control.config.simulation_config import SimulationConfig
    
    # Create default config
    config = SimulationConfig.create_default()
    
    # Use in simulation
    sim = SatelliteMPCLinearizedSimulation(config=config)
    
    # Create with overrides
    config = SimulationConfig.create_with_overrides({
        "mpc": {"prediction_horizon": 60}
    })
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .models import AppConfig
from .mission_state import MissionState, create_mission_state
from .satellite_config import _create_default_config


@dataclass(frozen=True)
class SimulationConfig:
    """
    Immutable configuration container for simulations.
    
    This class holds all configuration needed for a simulation run, eliminating
    the need for mutable global state. It's designed for dependency injection.
    
    Attributes:
        app_config: Application configuration (MPC, physics, simulation params)
        mission_state: Runtime mission state (waypoints, targets, etc.)
    """
    
    app_config: AppConfig
    mission_state: MissionState
    
    @classmethod
    def create_default(cls) -> "SimulationConfig":
        """
        Create a default simulation configuration.
        
        Returns:
            SimulationConfig with default settings
        """
        return cls(
            app_config=_create_default_config(),
            mission_state=create_mission_state(),
        )
    
    @classmethod
    def create_with_overrides(
        cls,
        overrides: Dict[str, Dict[str, Any]],
        base_config: Optional["SimulationConfig"] = None,
    ) -> "SimulationConfig":
        """
        Create configuration with overrides applied.
        
        Args:
            overrides: Dictionary of configuration overrides
            base_config: Base configuration (defaults to create_default() if None)
            
        Returns:
            SimulationConfig with overrides applied
        """
        if base_config is None:
            base_config = cls.create_default()
        
        # Create new AppConfig with overrides
        app_config_dict = base_config.app_config.model_dump()
        
        # Apply overrides
        for section, section_overrides in overrides.items():
            if section in app_config_dict:
                app_config_dict[section].update(section_overrides)
        
        # Create new AppConfig from updated dict
        new_app_config = AppConfig(**app_config_dict)
        
        # Return new immutable config
        return cls(
            app_config=new_app_config,
            mission_state=base_config.mission_state,  # Mission state not overridden here
        )
    
    def get_mpc_params(self) -> Dict[str, Any]:
        """Get MPC parameters as dictionary (for backward compatibility)."""
        return {
            "prediction_horizon": self.app_config.mpc.prediction_horizon,
            "control_horizon": self.app_config.mpc.control_horizon,
            "dt": self.app_config.mpc.dt,
            "solver_time_limit": self.app_config.mpc.solver_time_limit,
            "solver_type": self.app_config.mpc.solver_type,
            "q_position": self.app_config.mpc.q_position,
            "q_velocity": self.app_config.mpc.q_velocity,
            "q_angle": self.app_config.mpc.q_angle,
            "q_angular_velocity": self.app_config.mpc.q_angular_velocity,
            "r_thrust": self.app_config.mpc.r_thrust,
            "max_velocity": self.app_config.mpc.max_velocity,
            "max_angular_velocity": self.app_config.mpc.max_angular_velocity,
            "position_bounds": self.app_config.mpc.position_bounds,
            "damping_zone": self.app_config.mpc.damping_zone,
            "velocity_threshold": self.app_config.mpc.velocity_threshold,
            "max_velocity_weight": self.app_config.mpc.max_velocity_weight,
            "thruster_type": self.app_config.mpc.thruster_type,
        }
    
    def get_physics_params(self) -> Dict[str, Any]:
        """Get physics parameters as dictionary (for backward compatibility)."""
        return {
            "total_mass": self.app_config.physics.total_mass,
            "moment_of_inertia": self.app_config.physics.moment_of_inertia,
            "satellite_size": self.app_config.physics.satellite_size,
            "com_offset": self.app_config.physics.com_offset,
            "thruster_positions": self.app_config.physics.thruster_positions,
            "thruster_directions": self.app_config.physics.thruster_directions,
            "thruster_forces": self.app_config.physics.thruster_forces,
            "use_realistic_physics": self.app_config.physics.use_realistic_physics,
            "damping_linear": self.app_config.physics.damping_linear,
            "damping_angular": self.app_config.physics.damping_angular,
        }
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """Get simulation parameters as dictionary (for backward compatibility)."""
        return {
            "dt": self.app_config.simulation.dt,
            "max_duration": self.app_config.simulation.max_duration,
            "headless": self.app_config.simulation.headless,
            "window_width": self.app_config.simulation.window_width,
            "window_height": self.app_config.simulation.window_height,
            "use_final_stabilization": self.app_config.simulation.use_final_stabilization,
            "control_dt": self.app_config.simulation.control_dt,
            "target_hold_time": self.app_config.simulation.target_hold_time,
            "waypoint_final_stabilization_time": self.app_config.simulation.waypoint_final_stabilization_time,
            "shape_final_stabilization_time": self.app_config.simulation.shape_final_stabilization_time,
            "shape_positioning_stabilization_time": self.app_config.simulation.shape_positioning_stabilization_time,
            "default_target_speed": self.app_config.simulation.default_target_speed,
        }
