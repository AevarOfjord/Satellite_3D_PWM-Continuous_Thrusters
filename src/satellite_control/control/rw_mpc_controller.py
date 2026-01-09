"""
Reaction Wheel MPC Controller

MPC controller for satellite with reaction wheels (attitude) + thrusters (translation).

State Vector (16 elements):
    [x, y, z, qw, qx, qy, qz, vx, vy, vz, ωx, ωy, ωz, ω_rw_x, ω_rw_y, ω_rw_z]

Control Vector (9 elements):
    [τ_rw_x, τ_rw_y, τ_rw_z, F_px, F_mx, F_py, F_my, F_pz, F_mz]

Dynamics:
    - Position: ṗ = v
    - Velocity: v̇ = (1/m) * Σ F_thrusters
    - Orientation: q̇ = 0.5 * q ⊗ ω
    - Angular velocity: I * ω̇ = -ω × (I*ω) - τ_rw  (reaction torque from wheels)
    - Wheel speeds: ω̇_rw = τ_rw / I_rw
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import osqp
import scipy.sparse as sp

from src.satellite_control.config.actuator_config import ActuatorConfig, ActuatorMode
from src.satellite_control.config.reaction_wheel_config import ReactionWheelArrayConfig
from src.satellite_control.core.error_handling import with_error_context
from src.satellite_control.core.exceptions import OptimizationError

from .base import Controller

logger = logging.getLogger(__name__)


class ReactionWheelMPCController(Controller):
    """
    MPC Controller for satellite with reaction wheels + thrusters.

    Optimizes:
        - Reaction wheel torques for attitude control
        - Thruster forces for translation control

    Constraints:
        - Wheel torque limits (±max_torque)
        - Wheel speed limits (saturation avoidance)
        - Thruster force limits (0 to max_force)
    """

    def __init__(
        self,
        actuator_config: Optional[ActuatorConfig] = None,
        prediction_horizon: int = 50,
        dt: float = 0.05,
        satellite_mass: float = 10.0,
        satellite_inertia: float = 0.14,
        solver_time_limit: float = 0.04,
    ):
        """
        Initialize reaction wheel MPC controller.

        Args:
            actuator_config: Actuator configuration (default: reaction wheels)
            prediction_horizon: Number of prediction steps N
            dt: Control timestep in seconds
            satellite_mass: Satellite mass in kg
            satellite_inertia: Satellite moment of inertia in kg·m²
            solver_time_limit: Maximum solver time in seconds
        """
        # Actuator configuration
        if actuator_config is None:
            from src.satellite_control.config.actuator_config import create_actuator_config

            actuator_config = create_actuator_config("reaction_wheels")

        if actuator_config.mode != ActuatorMode.REACTION_WHEELS:
            raise ValueError("ReactionWheelMPCController requires reaction_wheels mode")

        self.actuator_config = actuator_config
        self.rw_config = actuator_config.reaction_wheels
        self.thruster_config = actuator_config.thrusters

        # Satellite parameters
        self.mass = satellite_mass
        self.inertia = satellite_inertia  # Scalar (assume spherical for now)

        # MPC parameters
        self.N = prediction_horizon
        self._dt = dt
        self.solver_time_limit = solver_time_limit

        # State and control dimensions
        self.nx = 16  # [pos(3), quat(4), vel(3), angvel(3), wheel_speed(3)]
        self.nu = 9  # [rw_torque(3), thruster_force(6)]

        # Cost weights
        self.Q_pos = 1000.0  # Position tracking
        self.Q_vel = 100.0  # Velocity tracking
        self.Q_ang = 2000.0  # Orientation tracking (quaternion) - increased for attitude control
        self.Q_angvel = 200.0  # Angular velocity tracking - increased for damping
        self.Q_wheel = 0.0  # No wheel speed penalty (allow momentum buildup)
        self.R_torque = 0.01  # Very low torque penalty for aggressive control
        self.R_thrust = 0.1  # Thruster usage penalty

        # Build Q diagonal
        self.Q_diag = np.concatenate(
            [
                np.full(3, self.Q_pos),  # Position
                np.full(4, self.Q_ang),  # Quaternion
                np.full(3, self.Q_vel),  # Velocity
                np.full(3, self.Q_angvel),  # Angular velocity
                np.full(3, self.Q_wheel),  # Wheel speeds
            ]
        )

        # Build R diagonal
        self.R_diag = np.concatenate(
            [
                np.full(3, self.R_torque),  # RW torques
                np.full(6, self.R_thrust),  # Thruster forces
            ]
        )

        # Actuator limits
        self.max_rw_torque = self.rw_config.wheel_x.max_torque
        self.max_rw_speed = self.rw_config.wheel_x.max_speed
        self.max_thrust = self.thruster_config.force

        # Performance tracking
        self.solve_times: list[float] = []

        # Problem dimensions
        self.n_vars = (self.N + 1) * self.nx + self.N * self.nu

        # Initialize OSQP solver
        self.prob = osqp.OSQP()
        self._init_solver_structures()

        # Previous quaternion for dynamics update
        self.prev_quat = np.array([-999.0] * 4)

        logger.info("ReactionWheelMPCController initialized")

    @property
    def dt(self) -> float:
        """Control update interval in seconds."""
        return self._dt

    @property
    def prediction_horizon(self) -> Optional[int]:
        """Prediction horizon N."""
        return self.N

    def _get_thrust_directions(self) -> np.ndarray:
        """Get thrust direction vectors for each thruster (6x3)."""
        directions = np.array(
            [
                [-1, 0, 0],  # px: pushes -X
                [1, 0, 0],  # mx: pushes +X
                [0, -1, 0],  # py: pushes -Y
                [0, 1, 0],  # my: pushes +Y
                [0, 0, -1],  # pz: pushes -Z
                [0, 0, 1],  # mz: pushes +Z
            ],
            dtype=np.float64,
        )
        return directions

    def linearize_dynamics(self, x_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics around current state.

        Args:
            x_current: Current state (16 elements)

        Returns:
            (A, B) matrices for discrete-time dynamics x_{k+1} = A*x_k + B*u_k
        """
        # Extract current quaternion
        q = x_current[3:7]
        w, qx, qy, qz = q[0], q[1], q[2], q[3]

        # State transition matrix A (16x16)
        A = np.eye(16)

        # Position dynamics: ṗ = v
        A[0, 7] = self.dt  # dx/dvx
        A[1, 8] = self.dt  # dy/dvy
        A[2, 9] = self.dt  # dz/dvz

        # Quaternion dynamics: q̇ = 0.5 * G(q) * ω
        # G(q) = [-qx, -qy, -qz; qw, -qz, qy; qz, qw, -qx; -qy, qx, qw]
        G = 0.5 * np.array(
            [
                [-qx, -qy, -qz],
                [w, -qz, qy],
                [qz, w, -qx],
                [-qy, qx, w],
            ]
        )
        A[3:7, 10:13] = G * self.dt

        # Control input matrix B (16x9)
        B = np.zeros((16, 9))

        # Velocity from thrusters: v̇ = (1/m) * Σ F
        # Control indices 3-8 are thruster forces
        thrust_dirs = self._get_thrust_directions()  # 6x3
        for i in range(6):
            B[7:10, 3 + i] = thrust_dirs[i] * self.max_thrust / self.mass * self.dt

        # Angular velocity from reaction wheel torque
        # ω̇ = -τ_rw / I_sat (reaction torque is opposite to motor torque)
        for i in range(3):
            B[10 + i, i] = -self.max_rw_torque / self.inertia * self.dt

        # Wheel speed dynamics: ω̇_rw = τ_rw / I_rw
        rw_inertia = self.rw_config.wheel_x.inertia
        for i in range(3):
            B[13 + i, i] = self.max_rw_torque / rw_inertia * self.dt

        return A, B

    def _init_solver_structures(self):
        """Initialize OSQP solver with constraint matrices."""
        # Cost matrix P (quadratic cost)
        Q_diag_full = np.concatenate([self.Q_diag] * self.N + [self.Q_diag * 10])  # Terminal weight
        R_diag_full = np.concatenate([self.R_diag] * self.N)
        P_diag = np.concatenate([Q_diag_full, R_diag_full])
        self.P = sp.diags(P_diag, format="csc")

        # Linear cost q (will be updated each solve)
        self.q = np.zeros(self.n_vars)

        # Build constraint matrices
        # Dynamics constraints: x_{k+1} = A*x_k + B*u_k
        # Initial state constraint: x_0 = x_current
        # Control bounds: u_min <= u <= u_max
        # State bounds (optional): x_min <= x <= x_max

        # Get template dynamics
        x_template = np.zeros(16)
        x_template[3] = 1.0  # Valid quaternion
        A_dyn, B_dyn = self.linearize_dynamics(x_template)

        # Count constraints
        n_dynamics = self.N * self.nx
        n_init = self.nx
        n_state_bounds = (self.N + 1) * self.nx
        n_control_bounds = self.N * self.nu
        self.n_constraints = n_dynamics + n_init + n_state_bounds + n_control_bounds

        # Build constraint matrix A_con (sparse)
        triples = []
        row = 0

        # Dynamics constraints: -A*x_k + x_{k+1} - B*u_k = 0
        for k in range(self.N):
            x_k_start = k * self.nx
            x_kp1_start = (k + 1) * self.nx
            u_k_start = (self.N + 1) * self.nx + k * self.nu

            # -A term
            for i in range(self.nx):
                for j in range(self.nx):
                    if A_dyn[i, j] != 0:
                        triples.append((row + i, x_k_start + j, -A_dyn[i, j]))

            # +I term (x_{k+1})
            for i in range(self.nx):
                triples.append((row + i, x_kp1_start + i, 1.0))

            # -B term
            for i in range(self.nx):
                for j in range(self.nu):
                    if B_dyn[i, j] != 0:
                        triples.append((row + i, u_k_start + j, -B_dyn[i, j]))

            row += self.nx

        # Initial state constraint: x_0 = x_current
        for i in range(self.nx):
            triples.append((row + i, i, 1.0))
        row += self.nx

        # State bounds: x_min <= x <= x_max
        for k in range(self.N + 1):
            x_k_start = k * self.nx
            for i in range(self.nx):
                triples.append((row + i, x_k_start + i, 1.0))
            row += self.nx

        # Control bounds: u_min <= u <= u_max
        for k in range(self.N):
            u_k_start = (self.N + 1) * self.nx + k * self.nu
            for i in range(self.nu):
                triples.append((row + i, u_k_start + i, 1.0))
            row += self.nu

        # Build sparse matrix
        rows, cols, vals = zip(*triples)
        self.A_con = sp.csc_matrix((vals, (rows, cols)), shape=(self.n_constraints, self.n_vars))
        self.A_con.sort_indices()

        # Initialize bounds
        self.l = np.zeros(self.n_constraints)
        self.u = np.zeros(self.n_constraints)

        # Dynamics constraints: equality (0 = 0)
        # (already initialized to 0)

        # Initial state: will be set each solve
        init_start = n_dynamics
        # (will be set in get_control_action)

        # State bounds (loose for now)
        state_start = n_dynamics + n_init
        self.l[state_start : state_start + n_state_bounds] = -1e10
        self.u[state_start : state_start + n_state_bounds] = 1e10

        # Wheel speed limits
        for k in range(self.N + 1):
            wheel_start = state_start + k * self.nx + 13  # Wheel speeds are indices 13-15
            self.l[wheel_start : wheel_start + 3] = -self.max_rw_speed * 0.95
            self.u[wheel_start : wheel_start + 3] = self.max_rw_speed * 0.95

        # Control bounds
        ctrl_start = state_start + n_state_bounds
        for k in range(self.N):
            # RW torques: [-1, 1] normalized
            self.l[ctrl_start + k * self.nu : ctrl_start + k * self.nu + 3] = -1.0
            self.u[ctrl_start + k * self.nu : ctrl_start + k * self.nu + 3] = 1.0
            # Thruster forces: [0, 1] normalized
            self.l[ctrl_start + k * self.nu + 3 : ctrl_start + k * self.nu + 9] = 0.0
            self.u[ctrl_start + k * self.nu + 3 : ctrl_start + k * self.nu + 9] = 1.0

        # Setup OSQP
        self.prob.setup(
            self.P,
            self.q,
            self.A_con,
            self.l,
            self.u,
            verbose=False,
            time_limit=self.solver_time_limit,
        )

    def get_solver_stats(self) -> Dict[str, Any]:
        """Get solver performance statistics."""
        if not self.solve_times:
            return {"solve_count": 0, "average_solve_time": 0.0, "max_solve_time": 0.0}
        return {
            "solve_times": self.solve_times.copy(),
            "solve_count": len(self.solve_times),
            "average_solve_time": sum(self.solve_times) / len(self.solve_times),
            "max_solve_time": max(self.solve_times),
        }

    @with_error_context("RW MPC solve", reraise=True)
    def get_control_action(
        self,
        x_current: np.ndarray,
        x_target: np.ndarray,
        previous_thrusters: Optional[np.ndarray] = None,
        x_target_trajectory: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute optimal control action.

        Args:
            x_current: Current state (16 elements)
            x_target: Target state (16 elements)
            previous_thrusters: Previous control (unused for compatibility)
            x_target_trajectory: Optional trajectory to track

        Returns:
            Tuple of (control_action, info_dict)
            control_action: [τx, τy, τz, Fpx, Fmx, Fpy, Fmy, Fpz, Fmz] normalized
        """
        start_time = time.time()

        # Ensure state dimensions
        if len(x_current) < 16:
            # Pad with zeros for wheel speeds
            x_current = np.concatenate([x_current, np.zeros(16 - len(x_current))])
        if len(x_target) < 16:
            x_target = np.concatenate([x_target, np.zeros(16 - len(x_target))])

        # Update dynamics if quaternion changed significantly
        quat = x_current[3:7]
        if np.linalg.norm(quat - self.prev_quat) > 0.05:
            A_dyn, B_dyn = self.linearize_dynamics(x_current)
            # Would need to update A_con here - for now, use initial linearization
            self.prev_quat = quat.copy()

        # Update linear cost q = -Q * x_ref
        self.q.fill(0.0)
        ref = x_target
        Q_term = self.Q_diag * 10  # Terminal weight

        for k in range(self.N):
            idx = k * self.nx
            self.q[idx : idx + self.nx] = -self.Q_diag * ref

        # Terminal state
        idx = self.N * self.nx
        self.q[idx : idx + self.nx] = -Q_term * ref

        # Update initial state constraint
        n_dynamics = self.N * self.nx
        init_start = n_dynamics
        self.l[init_start : init_start + self.nx] = x_current
        self.u[init_start : init_start + self.nx] = x_current

        # Update solver
        self.prob.update(q=self.q, l=self.l, u=self.u)

        # Solve
        try:
            res = self.prob.solve()
        except Exception as e:
            logger.error(f"OSQP solver exception: {e}")
            raise OptimizationError("solver_exception", str(e))

        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)

        # Check result
        if res.info.status not in ["solved", "solved_inaccurate"]:
            logger.warning(f"RW MPC solver status: {res.info.status}")
            # Return zero control
            return np.zeros(self.nu), {"status": -1, "solve_time": solve_time}

        # Extract first control action
        u_start = (self.N + 1) * self.nx
        u_opt = res.x[u_start : u_start + self.nu]

        # Clip to bounds
        u_opt[:3] = np.clip(u_opt[:3], -1.0, 1.0)  # RW torques
        u_opt[3:] = np.clip(u_opt[3:], 0.0, 1.0)  # Thruster forces

        return u_opt, {"status": 1, "solve_time": solve_time}

    def denormalize_control(self, u_normalized: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert normalized control to physical units.

        Args:
            u_normalized: Normalized control [9 elements]

        Returns:
            Dict with 'rw_torques' [N·m] and 'thruster_forces' [N]
        """
        return {
            "rw_torques": u_normalized[:3] * self.max_rw_torque,
            "thruster_forces": u_normalized[3:] * self.max_thrust,
        }
