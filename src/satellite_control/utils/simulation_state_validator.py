"""
Simulation State Validation Module

Centralized state validation, bounds checking, and tolerance verification for simulations.
Provides consistent validation logic across the simulation system.

Validation capabilities:
- State vector format and dimension checking
- Target reached verification with configurable tolerances
- Position, angle, and velocity tolerance checking
- State bounds validation (workspace limits, velocity limits)
- Physical constraint validation (NaN/Inf detection)

Utility functions:
- Angle normalization to [-π, π] range
- Shortest-path angle difference calculation
- Sensor noise simulation for realistic testing
- State component extraction and validation

Key features:
- Configurable tolerance levels for different mission phases
- Separate position and orientation tolerance checking
- Velocity stabilization verification
- Physical plausibility checks
- Integration with navigation_utils for consistency
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

# V4.0.0: SatelliteConfig removed - use AppConfig only
from src.satellite_control.config.models import AppConfig
from src.satellite_control.config import mpc_params
from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.utils.navigation_utils import (
    angle_difference,
    normalize_angle,
)


class SimulationStateValidator:
    """
    Validates and checks satellite state vectors for simulation.

    This class centralizes all state validation logic for the simulation system.
    """

    def __init__(
        self,
        position_tolerance: float = 0.05,
        angle_tolerance: float = 0.05,
        velocity_tolerance: float = 0.01,
        angular_velocity_tolerance: float = 0.01,
        position_bounds: Optional[float] = None,
        max_velocity: Optional[float] = None,
        max_angular_velocity: Optional[float] = None,
        app_config: Optional[AppConfig] = None,
    ):
        """
        Initialize state validator with tolerances and bounds.

        Args:
            position_tolerance: Position error tolerance in meters
            angle_tolerance: Angle error tolerance in radians
            velocity_tolerance: Velocity error tolerance in m/s
            angular_velocity_tolerance: Angular velocity tolerance in rad/s
            position_bounds: Maximum position value (default: from app_config or Config)
            max_velocity: Maximum linear velocity (default: from app_config or Config)
            max_angular_velocity: Maximum angular velocity (default: from app_config or Config)
            app_config: Optional AppConfig for accessing MPC parameters (v3.0.0)
        """
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.angular_velocity_tolerance = angular_velocity_tolerance
        self.app_config = app_config

        # V4.0.0: Load bounds from app_config (required, no fallback)
        if position_bounds is None or max_velocity is None or max_angular_velocity is None:
            if app_config and app_config.mpc:
                mpc_params_dict = {
                    "position_bounds": app_config.mpc.position_bounds,
                    "max_velocity": app_config.mpc.max_velocity,
                    "max_angular_velocity": app_config.mpc.max_angular_velocity,
                }
            else:
                # V4.0.0: Use default config if not provided
                default_config = SimulationConfig.create_default()
                mpc_params_dict = {
                    "position_bounds": default_config.app_config.mpc.position_bounds,
                    "max_velocity": default_config.app_config.mpc.max_velocity,
                    "max_angular_velocity": default_config.app_config.mpc.max_angular_velocity,
                }
            
            self.position_bounds = (
                position_bounds if position_bounds is not None else mpc_params_dict.get("position_bounds", 3.0)
            )
            self.max_velocity = (
                max_velocity if max_velocity is not None else mpc_params_dict.get("max_velocity", 0.15)
            )
            self.max_angular_velocity = (
                max_angular_velocity
                if max_angular_velocity is not None
                else mpc_params_dict.get("max_angular_velocity", np.pi / 2)
            )
        else:
            self.position_bounds = position_bounds
            self.max_velocity = max_velocity
            self.max_angular_velocity = max_angular_velocity

    def validate_state_format(self, state: np.ndarray) -> bool:
        """
        Validate that state vector has correct format.

        Expected format: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]

        Args:
            state: State vector to validate

        Returns:
            True if state format is valid

        Raises:
            ValueError: If state format is invalid
        """
        if not isinstance(state, np.ndarray):
            raise ValueError(f"State must be numpy array, got {type(state)}")

        if state.shape != (13,):
            raise ValueError(f"State must have shape (13,), got {state.shape}")

        if not np.all(np.isfinite(state)):
            raise ValueError(f"State contains non-finite values: {state}")

        return True

    def check_position_bounds(self, position: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Check if position is within bounds.

        Args:
            position: Position vector [x, y]

        Returns:
            Tuple of (is_valid, error_message)
        """
        if abs(position[0]) > self.position_bounds:
            return (
                False,
                f"X position {position[0]:.3f} exceeds bounds ±{self.position_bounds}",
            )

        if abs(position[1]) > self.position_bounds:
            return (
                False,
                f"Y position {position[1]:.3f} exceeds bounds ±{self.position_bounds}",
            )

        if len(position) > 2 and abs(position[2]) > self.position_bounds:
            return (
                False,
                f"Z position {position[2]:.3f} exceeds bounds ±{self.position_bounds}",
            )

        return True, None

    def check_velocity_bounds(self, velocity: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Check if velocity is within bounds.

        Args:
            velocity: Velocity vector [vx, vy]

        Returns:
            Tuple of (is_valid, error_message)
        """
        vel_magnitude = np.linalg.norm(velocity)

        if vel_magnitude > self.max_velocity:
            return (
                False,
                f"Velocity magnitude {vel_magnitude:.3f} exceeds max {self.max_velocity}",
            )

        return True, None

    def check_angular_velocity_bounds(self, angular_velocity: float) -> Tuple[bool, Optional[str]]:
        """
        Check if angular velocity is within bounds.

        Args:
            angular_velocity: Angular velocity in rad/s

        Returns:
            Tuple of (is_valid, error_message)
        """
        if abs(angular_velocity) > self.max_angular_velocity:
            return (
                False,
                f"Angular velocity {angular_velocity:.3f} exceeds max {self.max_angular_velocity}",
            )

        return True, None

    def check_all_bounds(self, state: np.ndarray) -> Tuple[bool, list]:
        """
        Check all state bounds.

        Args:
            state: State vector [pos(3), quat(4), vel(3), w(3)]

        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []

        # Check position
        pos_valid, pos_error = self.check_position_bounds(state[:3])
        if not pos_valid:
            errors.append(pos_error)

        # Check velocity
        vel_valid, vel_error = self.check_velocity_bounds(state[7:10])
        if not vel_valid:
            errors.append(vel_error)

        # Check angular velocity (norm of vector)
        ang_vel_mag = np.linalg.norm(state[10:13])
        angvel_valid, angvel_error = self.check_angular_velocity_bounds(ang_vel_mag)
        if not angvel_valid:
            errors.append(angvel_error)

        return len(errors) == 0, errors

    def normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-pi, pi] range.

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle in [-pi, pi]
        """
        return normalize_angle(angle)

    def angle_difference(self, target_angle: float, current_angle: float) -> float:
        """
        Calculate shortest angular difference.

        Args:
            target_angle: Target angle in radians
            current_angle: Current angle in radians

        Returns:
            Shortest angle difference in [-pi, pi]
        """
        return angle_difference(target_angle, current_angle)

    def check_target_reached(self, current_state: np.ndarray, target_state: np.ndarray) -> bool:
        """
        Check if satellite has reached target within tolerances.

        Args:
            current_state: Current state [pos(3), quat(4), vel(3), w(3)]
            target_state: Target state [pos(3), quat(4), vel(3), w(3)]

        Returns:
            True if target reached within all tolerances
        """
        # Position error (3D)
        pos_error = np.linalg.norm(current_state[:3] - target_state[:3])
        if pos_error >= self.position_tolerance:
            return False

        # Velocity error (3D)
        vel_error = np.linalg.norm(current_state[7:10] - target_state[7:10])
        if vel_error >= self.velocity_tolerance:
            return False

        # Angle error (Quaternion dot prod)
        q1 = current_state[3:7]
        q2 = target_state[3:7]
        dot = np.abs(np.dot(q1, q2))
        dot = min(1.0, max(-1.0, dot))
        ang_error = 2.0 * np.arccos(dot)

        if ang_error >= self.angle_tolerance:
            return False

        # Angular velocity error (3D)
        angvel_error = np.linalg.norm(current_state[10:13] - target_state[10:13])
        if angvel_error >= self.angular_velocity_tolerance:
            return False

        return True

    def compute_state_errors(
        self, current_state: np.ndarray, target_state: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all state errors.

        Args:
            current_state: Current state [pos(3), quat(4), vel(3), w(3)]
            target_state: Target state [pos(3), quat(4), vel(3), w(3)]

        Returns:
            Dictionary with error values
        """
        pos_error = np.linalg.norm(current_state[:3] - target_state[:3])
        vel_error = np.linalg.norm(current_state[7:10] - target_state[7:10])

        q1 = current_state[3:7]
        q2 = target_state[3:7]
        # dot product ensures shortest path
        dot = np.abs(np.dot(q1, q2))
        dot = min(1.0, max(-1.0, dot))
        ang_error = 2.0 * np.arccos(dot)

        angvel_error = np.linalg.norm(current_state[10:13] - target_state[10:13])

        return {
            "position_error": float(pos_error),
            "velocity_error": float(vel_error),
            "angle_error": float(ang_error),
            "angular_velocity_error": float(angvel_error),
            "position_error_x": float(current_state[0] - target_state[0]),
            "position_error_y": float(current_state[1] - target_state[1]),
            "position_error_z": float(current_state[2] - target_state[2]),
        }

    def check_within_tolerances(
        self, current_state: np.ndarray, target_state: np.ndarray
    ) -> Dict[str, bool]:
        """
        Check which state components are within tolerance.

        Args:
            current_state: Current state [pos(3), quat(4), vel(3), w(3)]
            target_state: Target state [pos(3), quat(4), vel(3), w(3)]

        Returns:
            Dictionary indicating which components are within tolerance
        """
        errors = self.compute_state_errors(current_state, target_state)

        return {
            "position": errors["position_error"] < self.position_tolerance,
            "velocity": errors["velocity_error"] < self.velocity_tolerance,
            "angle": errors["angle_error"] < self.angle_tolerance,
            "angular_velocity": errors["angular_velocity_error"] < self.angular_velocity_tolerance,
        }

    def apply_sensor_noise(
        self,
        true_state: np.ndarray,
        use_realistic_physics: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Add realistic sensor noise to state measurements.

        Models OptiTrack measurement uncertainty and velocity estimation errors.

        Args:
            true_state: True state [pos(3), quat(4), vel(3), w(3)]
            use_realistic_physics: Override for USE_REALISTIC_PHYSICS config

        Returns:
            Noisy state with measurement errors added
        """
        if use_realistic_physics is None:
            # V4.0.0: Get from app_config (required, no fallback)
            if self.app_config and self.app_config.physics:
                use_realistic_physics = self.app_config.physics.use_realistic_physics
            else:
                # V4.0.0: Use default config if not provided
                default_config = SimulationConfig.create_default()
                use_realistic_physics = default_config.app_config.physics.use_realistic_physics

        if not use_realistic_physics:
            return true_state

        noisy_state = true_state.copy()

        # Get noise parameters from app_config if available, otherwise fallback
        if self.app_config and self.app_config.physics:
            # Note: Noise parameters not yet in AppConfig, so use defaults
            # TODO: Add noise parameters to AppConfig in future refactor
            position_noise_std = 0.0
            velocity_noise_std = 0.0
            angle_noise_std = 0.0
            angular_velocity_noise_std = 0.0
        else:
            # V4.0.0: Use default config if not provided (no noise by default)
            default_config = SimulationConfig.create_default()
            physics = default_config.app_config.physics
            position_noise_std = getattr(physics, "position_noise_std", 0.0)
            velocity_noise_std = getattr(physics, "velocity_noise_std", 0.0)
            angle_noise_std = getattr(physics, "angle_noise_std", 0.0)
            angular_velocity_noise_std = getattr(physics, "angular_velocity_noise_std", 0.0)

        # Position noise (OptiTrack position uncertainty ~0.1-1mm)
        noisy_state[0] += np.random.normal(0, position_noise_std)
        noisy_state[1] += np.random.normal(0, position_noise_std)
        noisy_state[2] += np.random.normal(0, position_noise_std)

        # Velocity noise (from numerical differentiation + filtering)
        # [7,8,9]
        noisy_state[7] += np.random.normal(0, velocity_noise_std)
        noisy_state[8] += np.random.normal(0, velocity_noise_std)
        noisy_state[9] += np.random.normal(0, velocity_noise_std)

        # Angle noise (Rotation about Z for now to match noise config)
        # Apply small rotation perturbation
        angle_noise = np.random.normal(0, angle_noise_std)
        half = angle_noise / 2.0
        dq = np.array([np.cos(half), 0, 0, np.sin(half)])  # Z-rotation noise
        # q_new = q_old * dq (approx) or dq * q_old
        # Simplified: Just perturbs Z-component of orientation if predominantly planar?
        # Better: full quaternion multiplication.
        # But to be safe and simple, let's just add noise to Euler Z (Yaw) logic if we can.
        # Since we don't have quat logic here easily, assume simple scalar update isn't enough.
        # Skip complex noise or implement simple quat mult to avoid imports.
        # Implementation of simple q_mult(dq, q):
        # q = [w, x, y, z]
        qw, qx, qy, qz = noisy_state[3], noisy_state[4], noisy_state[5], noisy_state[6]
        dw, dx, dy, dz = dq[0], dq[1], dq[2], dq[3]
        # Hamilton product
        nw = dw * qw - dx * qx - dy * qy - dz * qz
        nx = dw * qx + dx * qw + dy * qz - dz * qy
        ny = dw * qy - dx * qz + dy * qw + dz * qx
        nz = dw * qz + dx * qy - dy * qx + dz * qw
        # Normalize
        norm = np.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        noisy_state[3:7] = np.array([nw, nx, ny, nz]) / norm

        # Angular velocity noise (10,11,12)
        noisy_state[10] += np.random.normal(0, angular_velocity_noise_std)
        noisy_state[11] += np.random.normal(0, angular_velocity_noise_std)
        noisy_state[12] += np.random.normal(0, angular_velocity_noise_std)

        return noisy_state

    def get_tolerance_summary(self) -> Dict[str, float]:
        """
        Get summary of all tolerances.

        Returns:
            Dictionary with tolerance values
        """
        return {
            "position_tolerance": self.position_tolerance,
            "angle_tolerance": self.angle_tolerance,
            "velocity_tolerance": self.velocity_tolerance,
            "angular_velocity_tolerance": self.angular_velocity_tolerance,
        }

    def get_bounds_summary(self) -> Dict[str, float]:
        """
        Get summary of all bounds.

        Returns:
            Dictionary with bound values
        """
        return {
            "position_bounds": self.position_bounds,
            "max_velocity": self.max_velocity,
            "max_angular_velocity": self.max_angular_velocity,
        }

    def validate_state_trajectory(
        self,
        state_history: list,
        check_continuity: bool = True,
        max_position_jump: float = 0.5,
        max_velocity_jump: float = 0.2,
    ) -> Tuple[bool, list]:
        """
        Validate a sequence of states for continuity and bounds.

        Args:
            state_history: List of state vectors
            check_continuity: Whether to check for discontinuous jumps
            max_position_jump: Maximum allowed position jump between steps
            max_velocity_jump: Maximum allowed velocity jump between steps

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if len(state_history) == 0:
            return True, []

        # Check first state format
        try:
            self.validate_state_format(state_history[0])
        except ValueError as e:
            errors.append(f"Invalid state format at step 0: {e}")
            return False, errors

        # Check each state
        for i, state in enumerate(state_history):
            # Format validation
            try:
                self.validate_state_format(state)
            except ValueError as e:
                errors.append(f"Invalid state format at step {i}: {e}")
                continue

            # Bounds validation
            all_valid, bounds_errors = self.check_all_bounds(state)
            if not all_valid:
                for error in bounds_errors:
                    errors.append(f"Step {i}: {error}")

            # Continuity check
            if check_continuity and i > 0:
                prev_state = state_history[i - 1]

                # Position jump (3D)
                pos_jump = np.linalg.norm(state[:3] - prev_state[:3])
                if pos_jump > max_position_jump:
                    errors.append(f"Step {i}: Large position jump {pos_jump:.3f}m")

                # Velocity jump
                vel_jump = np.linalg.norm(state[7:10] - prev_state[7:10])
                if vel_jump > max_velocity_jump:
                    errors.append(f"Step {i}: Large velocity jump {vel_jump:.3f}m/s")

        return len(errors) == 0, errors


def create_state_validator_from_config(
    config: Optional[Dict[str, Any]] = None,
    app_config: Optional[AppConfig] = None,
) -> SimulationStateValidator:
    """
    Factory function to create state validator from Config.

    Args:
        config: Optional configuration dictionary with overrides
        app_config: Optional AppConfig for accessing MPC parameters (v3.0.0)

    Returns:
        Configured SimulationStateValidator instance
    """
    if config is None:
        config = {}

    # Get tolerances from config dict, app_config, or fallback to constants
    if app_config and app_config.mpc:
        # Use constants from mpc_params module (v3.0.0)
        position_tolerance = config.get("position_tolerance", mpc_params.POSITION_TOLERANCE)
        angle_tolerance = config.get("angle_tolerance", mpc_params.ANGLE_TOLERANCE)
        # Note: velocity_tolerance and angular_velocity_tolerance not in MPCParams, use constants
        velocity_tolerance = config.get("velocity_tolerance", mpc_params.VELOCITY_TOLERANCE)
        angular_velocity_tolerance = config.get(
            "angular_velocity_tolerance", mpc_params.ANGULAR_VELOCITY_TOLERANCE
        )
    else:
        # V4.0.0: Use mpc_params constants if not in config
        position_tolerance = config.get("position_tolerance", mpc_params.POSITION_TOLERANCE)
        angle_tolerance = config.get("angle_tolerance", mpc_params.ANGLE_TOLERANCE)
        velocity_tolerance = config.get("velocity_tolerance", mpc_params.VELOCITY_TOLERANCE)
        angular_velocity_tolerance = config.get(
            "angular_velocity_tolerance",
            mpc_params.ANGULAR_VELOCITY_TOLERANCE,
        )

    # V4.0.0: Get bounds from app_config (required, no fallback)
    if app_config and app_config.mpc:
        mpc_params_dict = {
            "position_bounds": app_config.mpc.position_bounds,
            "max_velocity": app_config.mpc.max_velocity,
            "max_angular_velocity": app_config.mpc.max_angular_velocity,
        }
    else:
        # V4.0.0: Use default config if not provided
        default_config = SimulationConfig.create_default()
        mpc_params_dict = {
            "position_bounds": default_config.app_config.mpc.position_bounds,
            "max_velocity": default_config.app_config.mpc.max_velocity,
            "max_angular_velocity": default_config.app_config.mpc.max_angular_velocity,
        }
    
    position_bounds = config.get("position_bounds", mpc_params_dict.get("position_bounds"))
    max_velocity = config.get("max_velocity", mpc_params_dict.get("max_velocity"))
    max_angular_velocity = config.get(
        "max_angular_velocity", mpc_params_dict.get("max_angular_velocity")
    )

    return SimulationStateValidator(
        position_tolerance=position_tolerance,
        angle_tolerance=angle_tolerance,
        velocity_tolerance=velocity_tolerance,
        angular_velocity_tolerance=angular_velocity_tolerance,
        position_bounds=position_bounds,
        max_velocity=max_velocity,
        max_angular_velocity=max_angular_velocity,
        app_config=app_config,
    )


if __name__ == "__main__":
    # Quick test
    print("SimulationStateValidator Module")
    print("=" * 70)

    validator = create_state_validator_from_config()

    print("\nValidator Configuration:")
    print(f"  Tolerances: {validator.get_tolerance_summary()}")
    print(f"  Bounds: {validator.get_bounds_summary()}")

    # Test state validation
    test_state = np.zeros(13)
    test_state[0:3] = np.array([0.5, 0.5, 0.0])
    test_state[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    test_state[7:10] = np.array([0.1, 0.1, 0.0])
    test_state[10:13] = np.array([0.0, 0.0, 0.05])
    target_state = np.zeros(13)
    target_state[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

    print("\nTest State Validation:")
    print(f"  State: {test_state}")
    print(f"  Format valid: {validator.validate_state_format(test_state)}")

    errors = validator.compute_state_errors(test_state, target_state)
    print(f"\n  Errors: {errors}")

    tolerances = validator.check_within_tolerances(test_state, target_state)
    print(f"  Within tolerances: {tolerances}")

    reached = validator.check_target_reached(test_state, target_state)
    print(f"  Target reached: {reached}")

    print("\n" + "=" * 70)
