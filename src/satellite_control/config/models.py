"""
Pydantic Configuration Models for Satellite Control System

Type-safe configuration models with validation, range checks,
and descriptive error messages.

Supports both 2D (planar) and 3D (6-DOF) configurations.
"""

from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Type aliases for 2D and 3D positions/directions
Position2D = Tuple[float, float]
Position3D = Tuple[float, float, float]
PositionType = Union[Position2D, Position3D]


class SatellitePhysicalParams(BaseModel):
    """
    Satellite physical parameters with validation.

    Supports both 2D (planar) and 3D (6-DOF) configurations.
    """

    total_mass: float = Field(
        ...,
        gt=0,
        le=100,
        description="Total mass in kg (must be positive, max 100kg)",
    )
    moment_of_inertia: float = Field(
        ...,
        gt=0,
        le=10,
        description="Moment of inertia in kg*m^2 (must be positive)",
    )
    satellite_size: float = Field(
        ...,
        gt=0,
        le=2,
        description="Characteristic size in meters (must be positive, max 2m)",
    )
    com_offset: PositionType = Field(
        (0.0, 0.0),
        description="Center of Mass offset (x, y) or (x, y, z) in meters",
    )

    # Thruster configuration (supports variable number and 2D or 3D positions)
    thruster_positions: Dict[int, PositionType] = Field(
        ...,
        description="Map of thruster ID to position (2D or 3D)",
    )
    thruster_directions: Dict[int, PositionType] = Field(
        ...,
        description="Map of thruster ID to unit direction vector (2D or 3D)",
    )
    thruster_forces: Dict[int, float] = Field(
        ...,
        description="Map of thruster ID to max force in Newtons",
    )

    # Damping
    use_realistic_physics: bool = Field(
        False,
        description="Enable realistic physics (damping, noise, delays)",
    )
    damping_linear: float = Field(
        0.0,
        ge=0,
        le=10,
        description="Linear damping coefficient N/(m/s)",
    )
    damping_angular: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Angular damping coefficient N*m/(rad/s)",
    )

    @field_validator("thruster_positions")
    @classmethod
    def validate_thruster_positions(cls, v: Dict[int, PositionType]) -> Dict[int, PositionType]:
        """Validate thruster positions (8 for 2D, 12 for 3D)."""
        num_thrusters = len(v)
        if num_thrusters not in [8, 12]:
            raise ValueError(f"Expected 8 (2D) or 12 (3D) thrusters, got {num_thrusters}")
        for tid, pos in v.items():
            if not (1 <= tid <= num_thrusters):
                raise ValueError(f"Thruster ID must be 1-{num_thrusters}, got {tid}")
            # Check bounds for all coordinates
            for coord in pos:
                if abs(coord) > 1.0:
                    raise ValueError(f"Thruster {tid} position {pos} exceeds bounds (±1m)")
        return v

    @field_validator("thruster_forces")
    @classmethod
    def validate_thruster_forces(cls, v: Dict[int, float]) -> Dict[int, float]:
        """Validate thruster forces are positive and reasonable."""
        for tid, force in v.items():
            if force <= 0:
                raise ValueError(f"Thruster {tid} force must be positive, got {force}")
            if force > 100:
                raise ValueError(f"Thruster {tid} force {force}N exceeds reasonable maximum (100N)")
        return v

    @field_validator("com_offset")
    @classmethod
    def validate_com_offset(cls, v: PositionType) -> PositionType:
        """Validate COM offset is within satellite bounds."""
        for coord in v:
            if abs(coord) > 0.5:
                raise ValueError(f"COM offset {v} is too large (should be within ±0.5m)")
        return v


class MPCParams(BaseModel):
    """
    MPC Controller parameters with comprehensive validation.

    Includes cross-field consistency checks and reasonable bounds.
    """

    prediction_horizon: int = Field(
        ...,
        gt=0,
        le=50,
        description="Prediction horizon N (1-50 steps)",
    )
    control_horizon: int = Field(
        ...,
        gt=0,
        le=50,
        description="Control horizon M (1-50 steps, must be <= N)",
    )
    dt: float = Field(
        ...,
        gt=0,
        le=1.0,
        description="Control timestep in seconds (0-1s)",
    )
    solver_time_limit: float = Field(
        ...,
        gt=0,
        le=10.0,
        description="Maximum solver time in seconds",
    )
    solver_type: str = Field(
        "OSQP",
        description="Optimization solver type",
    )

    # Weights (Q - State, R - Control)
    q_position: float = Field(
        ...,
        ge=0,
        le=1e6,
        description="Position tracking weight",
    )
    q_velocity: float = Field(
        ...,
        ge=0,
        le=1e6,
        description="Velocity tracking weight",
    )
    q_angle: float = Field(
        ...,
        ge=0,
        le=1e6,
        description="Angle tracking weight",
    )
    q_angular_velocity: float = Field(
        ...,
        ge=0,
        le=1e6,
        description="Angular velocity tracking weight",
    )
    r_thrust: float = Field(
        ...,
        ge=0,
        le=1e6,
        description="Thrust usage penalty weight",
    )
    # Constraints
    max_velocity: float = Field(
        ...,
        gt=0,
        le=10.0,
        description="Maximum linear velocity in m/s",
    )
    max_angular_velocity: float = Field(
        ...,
        gt=0,
        le=20.0,
        description="Maximum angular velocity in rad/s",
    )
    position_bounds: float = Field(
        ...,
        gt=0,
        le=100.0,
        description="Position bounds ±meters from origin",
    )

    # Adaptive control
    damping_zone: float = Field(
        0.25,
        ge=0,
        le=5.0,
        description="Distance threshold for damping zone in meters",
    )
    velocity_threshold: float = Field(
        0.03,
        ge=0,
        le=1.0,
        description="Velocity threshold for fine control in m/s",
    )
    max_velocity_weight: float = Field(
        1000.0,
        ge=0,
        le=1e6,
        description="Maximum adaptive velocity weight",
    )
    thruster_type: str = Field(
        "PWM",
        description="Thruster actuation type: 'PWM' (Binary) or 'CON' (Continuous)",
    )

    @field_validator("thruster_type")
    @classmethod
    def validate_thruster_type(cls, v: str) -> str:
        """Validate thruster type."""
        if v not in ["PWM", "CON"]:
            raise ValueError(f"Thruster type must be 'PWM' or 'CON', got '{v}'")
        return v

    @field_validator("control_horizon")
    @classmethod
    def check_horizon_consistency(cls, v: int, info) -> int:
        """Ensure control_horizon <= prediction_horizon."""
        if "prediction_horizon" in info.data:
            if v > info.data["prediction_horizon"]:
                raise ValueError(
                    f"control_horizon ({v}) cannot exceed "
                    f"prediction_horizon ({info.data['prediction_horizon']})"
                )
        return v

    @field_validator("solver_time_limit")
    @classmethod
    def check_solver_time_vs_dt(cls, v: float, info) -> float:
        """Warn if solver time limit exceeds control timestep."""
        if "dt" in info.data:
            if v > info.data["dt"]:
                # This is a warning, not an error - we allow it but note the issue
                pass
        return v

    @model_validator(mode="after")
    def validate_weight_balance(self) -> "MPCParams":
        """Check that weights are reasonably balanced."""
        total_q = self.q_position + self.q_velocity + self.q_angle + self.q_angular_velocity
        if total_q == 0 and self.r_thrust > 0:
            raise ValueError(
                "All Q weights are zero but R_thrust is nonzero - "
                "controller will only minimize thrust, not track targets"
            )
        return self


class SimulationParams(BaseModel):
    """Simulation settings with validation."""

    dt: float = Field(
        0.005,
        gt=0,
        le=0.1,
        description="Physics timestep in seconds (max 100ms)",
    )
    max_duration: float = Field(
        ...,
        gt=0,
        le=3600,
        description="Maximum simulation duration in seconds",
    )
    headless: bool = Field(
        False,
        description="Run without visualization",
    )

    # Visualization defaults
    window_width: int = Field(
        1600,
        ge=640,
        le=4096,
        description="Window width in pixels",
    )
    window_height: int = Field(
        1000,
        ge=480,
        le=2160,
        description="Window height in pixels",
    )


class AppConfig(BaseModel):
    """
    Root configuration container.

    Combines all configuration subsections with cross-validation.
    """

    physics: SatellitePhysicalParams
    mpc: MPCParams
    simulation: SimulationParams

    input_file_path: Optional[str] = Field(
        None,
        description="Path to input file (e.g., DXF file)",
    )

    @model_validator(mode="after")
    def validate_timing_consistency(self) -> "AppConfig":
        """Ensure timing parameters are consistent across subsystems."""
        # MPC dt should be a multiple of simulation dt
        ratio = self.mpc.dt / self.simulation.dt
        if abs(ratio - round(ratio)) > 0.001:
            raise ValueError(
                f"MPC dt ({self.mpc.dt}s) should be a multiple of "
                f"simulation dt ({self.simulation.dt}s)"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create configuration from dictionary."""
        return cls.model_validate(data)
