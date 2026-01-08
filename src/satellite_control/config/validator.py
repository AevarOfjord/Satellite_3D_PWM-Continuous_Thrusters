"""
Comprehensive Configuration Validator

Validates entire configuration system for consistency, safety, and correctness.
Provides detailed error messages for debugging configuration issues.

Usage:
    from src.satellite_control.config.validator import ConfigValidator
    
    validator = ConfigValidator()
    issues = validator.validate_all(config)
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        raise ValueError("Configuration validation failed")
"""

from typing import List, Optional

from src.satellite_control.config.models import AppConfig


class ConfigValidator:
    """
    Comprehensive configuration validator.
    
    Validates all aspects of configuration including:
    - Individual parameter validity
    - Cross-parameter consistency
    - Physical plausibility
    - Safety constraints
    """

    @staticmethod
    def validate_all(config: AppConfig) -> List[str]:
        """
        Validate entire configuration and return list of issues.
        
        Args:
            config: AppConfig to validate
            
        Returns:
            List of validation issue messages (empty if valid)
        """
        issues: List[str] = []

        # Validate individual sections
        issues.extend(ConfigValidator._validate_physics(config.physics))
        issues.extend(ConfigValidator._validate_mpc(config.mpc))
        issues.extend(ConfigValidator._validate_simulation(config.simulation))

        # Validate cross-parameter consistency
        issues.extend(ConfigValidator._validate_cross_parameter(config))

        # Validate safety constraints
        issues.extend(ConfigValidator._validate_safety(config))

        return issues

    @staticmethod
    def _validate_physics(physics) -> List[str]:
        """Validate physics parameters."""
        issues: List[str] = []

        # Mass validation
        if physics.total_mass <= 0:
            issues.append(f"Mass must be positive: {physics.total_mass} kg")
        if physics.total_mass > 100:
            issues.append(
                f"Mass seems unusually high for a small satellite: {physics.total_mass} kg"
            )

        # Inertia validation
        if physics.moment_of_inertia <= 0:
            issues.append(
                f"Moment of inertia must be positive: {physics.moment_of_inertia} kg⋅m²"
            )

        # Thruster validation
        if len(physics.thruster_positions) != len(physics.thruster_directions):
            issues.append(
                f"Thruster positions ({len(physics.thruster_positions)}) and "
                f"directions ({len(physics.thruster_directions)}) count mismatch"
            )

        if len(physics.thruster_positions) != len(physics.thruster_forces):
            issues.append(
                f"Thruster positions ({len(physics.thruster_positions)}) and "
                f"forces ({len(physics.thruster_forces)}) count mismatch"
            )

        # Validate thruster forces are positive
        for thruster_id, force in physics.thruster_forces.items():
            if force <= 0:
                issues.append(f"Thruster {thruster_id} force must be positive: {force} N")
            if force > 10:
                issues.append(
                    f"Thruster {thruster_id} force seems unusually high: {force} N"
                )

        # Damping validation
        if physics.use_realistic_physics:
            if physics.damping_linear < 0:
                issues.append(f"Linear damping must be non-negative: {physics.damping_linear}")
            if physics.damping_angular < 0:
                issues.append(
                    f"Angular damping must be non-negative: {physics.damping_angular}"
                )

        return issues

    @staticmethod
    def _validate_mpc(mpc) -> List[str]:
        """Validate MPC parameters."""
        issues: List[str] = []

        # Horizon validation
        if mpc.prediction_horizon <= 0:
            issues.append(f"Prediction horizon must be positive: {mpc.prediction_horizon}")
        if mpc.control_horizon <= 0:
            issues.append(f"Control horizon must be positive: {mpc.control_horizon}")
        if mpc.control_horizon > mpc.prediction_horizon:
            issues.append(
                f"Control horizon ({mpc.control_horizon}) > "
                f"prediction horizon ({mpc.prediction_horizon})"
            )

        # Timing validation
        if mpc.dt <= 0:
            issues.append(f"MPC dt must be positive: {mpc.dt} s")
        if mpc.solver_time_limit <= 0:
            issues.append(f"Solver time limit must be positive: {mpc.solver_time_limit} s")
        if mpc.solver_time_limit >= mpc.dt:
            issues.append(
                f"Solver time limit ({mpc.solver_time_limit}s) >= "
                f"control dt ({mpc.dt}s) - solver may not complete in time"
            )

        # Constraint validation
        if mpc.max_velocity <= 0:
            issues.append(f"Max velocity must be positive: {mpc.max_velocity} m/s")
        if mpc.max_angular_velocity <= 0:
            issues.append(
                f"Max angular velocity must be positive: {mpc.max_angular_velocity} rad/s"
            )
        if mpc.position_bounds <= 0:
            issues.append(f"Position bounds must be positive: {mpc.position_bounds} m")

        # Cost weight validation
        if mpc.q_position < 0:
            issues.append(f"Position weight must be non-negative: {mpc.q_position}")
        if mpc.q_velocity < 0:
            issues.append(f"Velocity weight must be non-negative: {mpc.q_velocity}")
        if mpc.q_angle < 0:
            issues.append(f"Angle weight must be non-negative: {mpc.q_angle}")
        if mpc.q_angular_velocity < 0:
            issues.append(
                f"Angular velocity weight must be non-negative: {mpc.q_angular_velocity}"
            )
        if mpc.r_thrust < 0:
            issues.append(f"Thrust penalty must be non-negative: {mpc.r_thrust}")

        # Thruster type validation
        if mpc.thruster_type not in ["PWM", "CON"]:
            issues.append(
                f"Invalid thruster type: {mpc.thruster_type}. Must be 'PWM' or 'CON'"
            )

        return issues

    @staticmethod
    def _validate_simulation(simulation) -> List[str]:
        """Validate simulation parameters."""
        issues: List[str] = []

        # Timing validation
        if simulation.dt <= 0:
            issues.append(f"Simulation dt must be positive: {simulation.dt} s")
        if simulation.max_duration <= 0:
            issues.append(
                f"Max simulation duration must be positive: {simulation.max_duration} s"
            )

        # Window size validation (if not headless)
        if not simulation.headless:
            if simulation.window_width <= 0:
                issues.append(f"Window width must be positive: {simulation.window_width} px")
            if simulation.window_height <= 0:
                issues.append(
                    f"Window height must be positive: {simulation.window_height} px"
                )

        return issues

    @staticmethod
    def _validate_cross_parameter(config: AppConfig) -> List[str]:
        """Validate cross-parameter consistency."""
        issues: List[str] = []

        # MPC dt should be a multiple of simulation dt
        ratio = config.mpc.dt / config.simulation.dt
        if abs(ratio - round(ratio)) > 0.001:
            issues.append(
                f"MPC dt ({config.mpc.dt}s) should be a multiple of "
                f"simulation dt ({config.simulation.dt}s) for consistent integration"
            )

        # Solver time limit should be much less than control dt
        if config.mpc.solver_time_limit > config.mpc.dt * 0.8:
            issues.append(
                f"Solver time limit ({config.mpc.solver_time_limit}s) is too close to "
                f"control dt ({config.mpc.dt}s). Recommend < 80% of control dt"
            )

        # Max velocity should be reasonable for workspace
        if config.mpc.max_velocity > config.mpc.position_bounds:
            issues.append(
                f"Max velocity ({config.mpc.max_velocity} m/s) exceeds position bounds "
                f"({config.mpc.position_bounds} m) - may cause boundary violations"
            )

        # Check if prediction horizon is reasonable
        prediction_time = config.mpc.prediction_horizon * config.mpc.dt
        if prediction_time > config.simulation.max_duration:
            issues.append(
                f"Prediction horizon ({prediction_time:.1f}s) exceeds "
                f"max simulation duration ({config.simulation.max_duration:.1f}s)"
            )

        return issues

    @staticmethod
    def _validate_safety(config: AppConfig) -> List[str]:
        """Validate safety constraints."""
        issues: List[str] = []

        # Velocity limits should be reasonable
        if config.mpc.max_velocity > 5.0:
            issues.append(
                f"Max velocity ({config.mpc.max_velocity} m/s) seems very high "
                f"for a small satellite - verify this is intentional"
            )

        # Angular velocity limits
        if config.mpc.max_angular_velocity > 10.0:
            issues.append(
                f"Max angular velocity ({config.mpc.max_angular_velocity} rad/s) "
                f"seems very high - verify this is intentional"
            )

        # Solver timeout should allow for real-time operation
        if config.mpc.solver_time_limit > 0.1:
            issues.append(
                f"Solver time limit ({config.mpc.solver_time_limit}s) is very high. "
                f"May cause control loop timing violations"
            )

        return issues

    @staticmethod
    def validate_and_raise(config: AppConfig) -> None:
        """
        Validate configuration and raise exception if invalid.
        
        Args:
            config: AppConfig to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        issues = ConfigValidator.validate_all(config)
        if issues:
            error_msg = "Configuration validation failed:\n"
            error_msg += "\n".join(f"  - {issue}" for issue in issues)
            raise ValueError(error_msg)


def validate_config_at_startup(app_config: Optional[AppConfig] = None) -> None:
    """
    Validate configuration at application startup.
    
    Should be called early in application initialization.
    Raises ValueError if configuration is invalid.
    
    Args:
        app_config: Optional AppConfig to validate (v4.0.0). If None, uses defaults.
    """
    if app_config is None:
        # V4.0.0: Use default config if not provided (no SatelliteConfig fallback)
        from src.satellite_control.config.simulation_config import SimulationConfig
        default_config = SimulationConfig.create_default()
        app_config = default_config.app_config
    
    ConfigValidator.validate_and_raise(app_config)
