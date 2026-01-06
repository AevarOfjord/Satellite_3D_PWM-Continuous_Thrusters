"""
3D MPC Controller using OSQP.

Extends satellite control to full 6-DOF using quaternion attitude representation.

State vector (13 elements):
    [x, y, z, qw, qx, qy, qz, vx, vy, vz, ωx, ωy, ωz]

Control vector (12 elements):
    Duty cycle [0, 1] for each of 12 thrusters
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import osqp
import scipy.sparse as sp

from src.satellite_control.config import SatelliteConfig
from src.satellite_control.config.models import MPCParams, SatellitePhysicalParams
from src.satellite_control.config.physics import USE_3D_MODE
from src.satellite_control.utils.profiler import mpc_profiler

logger = logging.getLogger(__name__)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """
    Compute quaternion error for MPC cost.

    Returns a 3-element error vector (axis-angle like).
    """
    # Quaternion conjugate of target
    q_target_conj = np.array([q_target[0], -q_target[1], -q_target[2], -q_target[3]])

    # Error quaternion: q_err = q_target^-1 * q_current
    w1, x1, y1, z1 = q_target_conj
    w2, x2, y2, z2 = q_current

    q_err = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )

    # Ensure shortest path
    if q_err[0] < 0:
        q_err = -q_err

    # Return vector part (proportional to axis * angle for small angles)
    return 2.0 * q_err[1:4]


class MPCController3D:
    """
    3D Satellite Model Predictive Controller using OSQP.

    Uses quaternion representation for attitude, linearized dynamics,
    and 12 thrusters for full 6-DOF control.
    """

    def __init__(
        self,
        satellite_params: Optional[Union[Dict[str, Any], SatellitePhysicalParams]] = None,
        mpc_params: Optional[Union[Dict[str, Any], MPCParams]] = None,
    ):
        """Initialize 3D MPC controller."""
        if not USE_3D_MODE:
            raise RuntimeError("MPCController3D requires USE_3D_MODE=True in physics.py")

        # Load defaults from global Config if needed
        if satellite_params is None:
            satellite_params = SatelliteConfig.get_app_config().physics
        if mpc_params is None:
            mpc_params = SatelliteConfig.get_app_config().mpc

        # Helper to extract parameters
        def get_param(obj, attr, key, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            if isinstance(obj, dict):
                if attr in obj:
                    return obj[attr]
                if key in obj:
                    return obj[key]
            return obj.get(key, default) if isinstance(obj, dict) else default

        # Satellite physical parameters
        self.total_mass = get_param(satellite_params, "total_mass", "mass")
        self.inertia_tensor = get_param(satellite_params, "inertia_tensor", "inertia_tensor")
        if self.inertia_tensor is None:
            inertia = get_param(satellite_params, "moment_of_inertia", "inertia")
            self.inertia_tensor = np.diag([inertia, inertia, inertia])

        self.thruster_positions = get_param(
            satellite_params, "thruster_positions", "thruster_positions"
        )
        self.thruster_directions = get_param(
            satellite_params, "thruster_directions", "thruster_directions"
        )
        self.thruster_forces = get_param(satellite_params, "thruster_forces", "thruster_forces")
        self.com_offset = get_param(satellite_params, "com_offset", "com_offset", np.zeros(3))
        self.com_offset = np.array(self.com_offset)

        # MPC parameters
        self.N = get_param(mpc_params, "prediction_horizon", "prediction_horizon")
        self.M = get_param(mpc_params, "control_horizon", "control_horizon")
        self.dt = get_param(mpc_params, "dt", "dt")
        self.solver_time_limit = get_param(mpc_params, "solver_time_limit", "solver_time_limit")

        # Cost weights
        self.Q_pos = get_param(mpc_params, "q_position", "Q_pos")
        self.Q_vel = get_param(mpc_params, "q_velocity", "Q_vel")
        self.Q_ang = get_param(mpc_params, "q_angle", "Q_ang")
        self.Q_angvel = get_param(mpc_params, "q_angular_velocity", "Q_angvel")
        self.R_thrust = get_param(mpc_params, "r_thrust", "R_thrust")

        # Constraints
        self.max_velocity = get_param(mpc_params, "max_velocity", "max_velocity")
        self.max_angular_velocity = get_param(
            mpc_params, "max_angular_velocity", "max_angular_velocity"
        )
        self.position_bounds = get_param(mpc_params, "position_bounds", "position_bounds")

        # State dimension for linearized 3D: [pos(3), quat_err(3), vel(3), omega(3)] = 12
        # Note: We linearize quaternion to 3D error for MPC numerics
        self.nx = 12
        # Control dimension: 12 thrusters
        self.nu = 12

        # Performance tracking
        self.solve_times: list[float] = []

        # Precompute thruster forces and torques
        self._precompute_thruster_forces()

        if SatelliteConfig.VERBOSE_MPC:
            print("OSQP 3D MPC Controller Initializing...")
            print(f"  State dim: {self.nx}, Control dim: {self.nu}")

        # Problem dimensions
        self.n_vars = (self.N + 1) * self.nx + self.N * self.nu

        # Constraint counts
        n_dyn = self.N * self.nx
        n_init = self.nx
        n_bounds_x = (self.N + 1) * self.nx
        n_bounds_u = self.N * self.nu
        self.n_constraints = n_dyn + n_init + n_bounds_x + n_bounds_u

        # OSQP Solver
        self.prob = osqp.OSQP()

        # Initialize solver matrices
        self._init_solver_structures()

        # State tracking
        self.prev_quat = np.array([1.0, 0.0, 0.0, 0.0])

        if SatelliteConfig.VERBOSE_MPC:
            print("OSQP 3D MPC Ready.")

    def _precompute_thruster_forces(self) -> None:
        """Precompute thruster forces and torques in body frame."""
        self.body_frame_forces = np.zeros((self.nu, 3), dtype=np.float64)
        self.thruster_torques = np.zeros((self.nu, 3), dtype=np.float64)

        for i in range(self.nu):
            thruster_id = i + 1
            pos = np.array(self.thruster_positions[thruster_id])
            rel_pos = pos - self.com_offset
            direction = np.array(self.thruster_directions[thruster_id])
            force = self.thruster_forces[thruster_id] * direction

            self.body_frame_forces[i] = force
            # Torque = r × F
            self.thruster_torques[i] = np.cross(rel_pos, force)

    def linearize_dynamics(self, q_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize 3D dynamics around current state.

        State: [x, y, z, qe_x, qe_y, qe_z, vx, vy, vz, ωx, ωy, ωz]
        where qe is the quaternion error (3-element)
        """
        # Rotation matrix from body to world
        R = quat_to_rotation_matrix(q_current)

        # Inverse inertia tensor
        I_inv = np.linalg.inv(self.inertia_tensor)

        # State transition A (12x12)
        A = np.eye(self.nx)
        # Position integrates velocity
        A[0:3, 6:9] = np.eye(3) * self.dt
        # Quaternion error integrates angular velocity (linearized)
        A[3:6, 9:12] = np.eye(3) * self.dt

        # Control input B (12x12)
        B = np.zeros((self.nx, self.nu))

        for i in range(self.nu):
            # Transform force to world frame
            F_world = R @ self.body_frame_forces[i]
            # Acceleration = F/m * dt
            B[6:9, i] = F_world / self.total_mass * self.dt

            # Angular acceleration = I^-1 * τ * dt (in body frame)
            tau_body = self.thruster_torques[i]
            B[9:12, i] = I_inv @ tau_body * self.dt

        return A, B

    def _init_solver_structures(self):
        """Build initial sparse structure of P and A matrices."""
        # Cost matrix Q diagonal
        Q_diag = np.array(
            [
                self.Q_pos,
                self.Q_pos,
                self.Q_pos,  # position
                self.Q_ang,
                self.Q_ang,
                self.Q_ang,  # attitude (quat error)
                self.Q_vel,
                self.Q_vel,
                self.Q_vel,  # velocity
                self.Q_angvel,
                self.Q_angvel,
                self.Q_angvel,  # angular velocity
            ]
        )
        R_diag = np.full(self.nu, self.R_thrust)

        # Build P matrix
        diags = []
        for _ in range(self.N):
            diags.append(Q_diag)
        diags.append(Q_diag * 10.0)  # Terminal cost
        for _ in range(self.N):
            diags.append(R_diag)

        self.P = sp.diags(np.concatenate(diags), format="csc")

        # Linear cost q (updated each solve)
        self.q = np.zeros(self.n_vars)

        # Build A matrix with placeholder dynamics
        dummy_quat = np.array([1.0, 0.0, 0.0, 0.0])
        A_template, _ = self.linearize_dynamics(dummy_quat)

        triples = []
        row_idx = 0

        # Dynamics constraints
        for k in range(self.N):
            x_k_idx = k * self.nx
            x_kp1_idx = (k + 1) * self.nx
            u_k_idx = (self.N + 1) * self.nx + k * self.nu

            # -A term
            for r in range(self.nx):
                for c in range(self.nx):
                    if A_template[r, c] != 0:
                        triples.append((row_idx + r, x_k_idx + c, -A_template[r, c]))

            # I term
            for r in range(self.nx):
                triples.append((row_idx + r, x_kp1_idx + r, 1.0))

            # -B term (placeholder 1.0)
            for r in range(self.nx):
                for c in range(self.nu):
                    triples.append((row_idx + r, u_k_idx + c, 1.0))

            row_idx += self.nx

        # Initial state constraint
        for r in range(self.nx):
            triples.append((row_idx + r, r, 1.0))
        row_idx += self.nx

        # State bounds
        for k in range(self.N + 1):
            x_k_idx = k * self.nx
            for r in range(self.nx):
                triples.append((row_idx + r, x_k_idx + r, 1.0))
            row_idx += self.nx

        # Control bounds
        for k in range(self.N):
            u_k_idx = (self.N + 1) * self.nx + k * self.nu
            for r in range(self.nu):
                triples.append((row_idx + r, u_k_idx + r, 1.0))
            row_idx += self.nu

        # Build sparse matrix
        rows = [t[0] for t in triples]
        cols = [t[1] for t in triples]
        vals = [t[2] for t in triples]

        self.A = sp.csc_matrix((vals, (rows, cols)), shape=(self.n_constraints, self.n_vars))
        self.A.sort_indices()

        # Map B indices for fast updates
        self.B_idx_map: Dict[Tuple[int, int], list] = {}
        for r in range(self.nx):
            for c in range(self.nu):
                self.B_idx_map[(r, c)] = []

        u_start_idx = (self.N + 1) * self.nx
        for k in range(self.N):
            current_u_idx = u_start_idx + k * self.nu
            current_row_base = k * self.nx

            for c in range(self.nu):
                col = current_u_idx + c
                start_ptr = self.A.indptr[col]
                end_ptr = self.A.indptr[col + 1]
                col_rows = self.A.indices[start_ptr:end_ptr]

                for r in range(self.nx):
                    target_row = current_row_base + r
                    match = np.where(col_rows == target_row)[0]
                    if len(match) > 0:
                        data_idx = start_ptr + match[0]
                        self.B_idx_map[(r, c)].append(data_idx)

        # Initialize l and u vectors
        self.l = np.zeros(self.n_constraints)
        self.u = np.zeros(self.n_constraints)

        # Control bounds [0, 1]
        ctrl_bound_start = self.n_constraints - self.N * self.nu
        self.l[ctrl_bound_start:] = 0.0
        self.u[ctrl_bound_start:] = 1.0

        # State bounds
        state_bound_start = self.N * self.nx + self.nx
        state_bound_end = state_bound_start + (self.N + 1) * self.nx

        x_min = np.full(self.nx, -np.inf)
        x_max = np.full(self.nx, np.inf)
        # Velocity bounds
        x_min[6:9] = -self.max_velocity
        x_max[6:9] = self.max_velocity
        # Angular velocity bounds
        x_min[9:12] = -self.max_angular_velocity
        x_max[9:12] = self.max_angular_velocity

        if self.position_bounds:
            x_min[0:3] = -self.position_bounds * 2.0
            x_max[0:3] = self.position_bounds * 2.0

        self.l[state_bound_start:state_bound_end] = np.tile(x_min, self.N + 1)
        self.u[state_bound_start:state_bound_end] = np.tile(x_max, self.N + 1)

        # Setup OSQP
        self.prob.setup(
            self.P,
            self.q,
            self.A,
            self.l,
            self.u,
            warm_start=True,
            verbose=False,
            time_limit=self.solver_time_limit,
            eps_abs=1e-4,
            eps_rel=1e-4,
            polish=False,
            check_termination=10,
        )

    def _update_A_data(self, B_dyn: np.ndarray):
        """Update A constraints with new B matrix."""
        for r in range(self.nx):
            for c in range(self.nu):
                val = -B_dyn[r, c]
                for idx in self.B_idx_map[(r, c)]:
                    self.A.data[idx] = val

    def get_control_action(
        self,
        state_full: np.ndarray,
        target_full: np.ndarray,
        previous_thrusters: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute optimal 3D control action.

        Args:
            state_full: [x,y,z, qw,qx,qy,qz, vx,vy,vz, ωx,ωy,ωz] = 13 elements
            target_full: [x,y,z, qw,qx,qy,qz, vx,vy,vz, ωx,ωy,ωz] = 13 elements

        Returns:
            Control vector (12 elements), info dict
        """
        start_time = time.time()

        # Extract quaternion
        q_current = state_full[3:7]
        q_current = q_current / np.linalg.norm(q_current)  # Normalize

        q_target = target_full[3:7]
        q_target = q_target / np.linalg.norm(q_target)

        # Convert to linearized state (12 elements)
        x_curr = np.zeros(self.nx)
        x_curr[0:3] = state_full[0:3]  # position
        x_curr[3:6] = quat_error(q_current, q_target)  # attitude error
        x_curr[6:9] = state_full[7:10]  # linear velocity
        x_curr[9:12] = state_full[10:13]  # angular velocity

        x_targ = np.zeros(self.nx)
        x_targ[0:3] = target_full[0:3]
        x_targ[3:6] = np.zeros(3)  # Target quat error is zero
        x_targ[6:9] = target_full[7:10]
        x_targ[9:12] = target_full[10:13]

        # Update dynamics
        _, B_dyn = self.linearize_dynamics(q_current)
        self._update_A_data(B_dyn)
        self.prob.update(Ax=self.A.data)

        # Update linear cost
        self.q.fill(0.0)
        Q_diag = np.array(
            [
                self.Q_pos,
                self.Q_pos,
                self.Q_pos,
                self.Q_ang,
                self.Q_ang,
                self.Q_ang,
                self.Q_vel,
                self.Q_vel,
                self.Q_vel,
                self.Q_angvel,
                self.Q_angvel,
                self.Q_angvel,
            ]
        )

        for k in range(self.N + 1):
            idx_start = k * self.nx
            weight = 10.0 if k == self.N else 1.0
            self.q[idx_start : idx_start + self.nx] = -Q_diag * weight * x_targ

        # Update initial state constraint
        init_start = self.N * self.nx
        self.l[init_start : init_start + self.nx] = x_curr
        self.u[init_start : init_start + self.nx] = x_curr

        # Push updates
        self.prob.update(q=self.q, l=self.l, u=self.u)

        # Solve
        with mpc_profiler.measure("osqp.solve_3d"):
            res = self.prob.solve()

        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)

        # Extract result
        if res.info.status not in ["solved", "solved_inaccurate"]:
            logger.warning(f"OSQP 3D Failed: {res.info.status}")
            return np.zeros(self.nu), {
                "status": -1,
                "status_name": res.info.status,
                "solve_time": solve_time,
                "solver_fallback": True,
            }

        u_start_idx = (self.N + 1) * self.nx
        u_opt = res.x[u_start_idx : u_start_idx + self.nu]
        u_opt = np.clip(u_opt, 0.0, 1.0)

        return u_opt, {
            "status": 1,
            "status_name": "OPTIMAL",
            "solve_time": solve_time,
            "solver_fallback": False,
            "obj_val": res.info.obj_val,
        }
