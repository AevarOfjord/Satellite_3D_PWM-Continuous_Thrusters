"""
Unified MPC Controller using OSQP.

Combines logic from previous BaseMPC and PWMMPC into a single, optimized class.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import osqp
import scipy.sparse as sp

from src.satellite_control.config import mpc_params
from src.satellite_control.config.models import MPCParams, SatellitePhysicalParams
from src.satellite_control.core.error_handling import with_error_context
from src.satellite_control.core.exceptions import OptimizationError, SolverTimeoutError
from src.satellite_control.utils.caching import cached

from .base import Controller

logger = logging.getLogger(__name__)


class MPCController(Controller):
    """
    Satellite Model Predictive Controller using OSQP (3D).

    Unified implementation for 6-DOF control.
    State: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13 elements).
    Control: [u1, ..., u12] (12 thrusters).
    """

    def __init__(
        self,
        satellite_params: Optional[Union[Dict[str, Any], SatellitePhysicalParams]] = None,
        mpc_params: Optional[Union[Dict[str, Any], MPCParams]] = None,
    ):
        """
        Initialize MPC controller with OSQP and pre-allocate matrices.
        """
        # Load defaults from SimulationConfig if not provided (V3.0.0)
        if satellite_params is None or mpc_params is None:
            from src.satellite_control.config.simulation_config import SimulationConfig

            default_config = SimulationConfig.create_default()
            if satellite_params is None:
                satellite_params = default_config.app_config.physics
            if mpc_params is None:
                mpc_params = default_config.app_config.mpc

        # Normalization Helper
        def get_param(obj, attr, key, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            if isinstance(obj, dict):
                # Try both keys
                if attr in obj:
                    return obj[attr]
                if key in obj:
                    return obj[key]
            return obj.get(key, default) if isinstance(obj, dict) else default

        # Satellite physical parameters
        self.total_mass = get_param(satellite_params, "total_mass", "mass")
        self.moment_of_inertia = get_param(satellite_params, "moment_of_inertia", "inertia")
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
        self._dt = get_param(mpc_params, "dt", "dt")

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

        # State dimension: [p(3), q(4), v(3), w(3)]
        self.nx = 13
        # Control dimension: 12 thrusters
        self.nu = 12

        # Performance tracking
        self.solve_times: list[float] = []

        # Precompute thruster forces (body frame)
        self._precompute_thruster_forces()

        # Precompute Q_diag (used frequently in get_control_action)
        self.Q_diag = np.concatenate(
            [
                np.full(3, self.Q_pos),
                np.full(4, self.Q_ang),
                np.full(3, self.Q_vel),
                np.full(3, self.Q_angvel),
            ]
        )

        # Use verbose flag from mpc_params
        self.verbose = get_param(mpc_params, "verbose_mpc", "VERBOSE_MPC", False)
        
        if self.verbose:
            print("OSQP MPC Controller Initializing (3D)...")

        # Problem dimensions
        self.n_vars = (self.N + 1) * self.nx + self.N * self.nu

        # Constraints counts
        n_dyn = self.N * self.nx
        n_init = self.nx
        n_bounds_x = (self.N + 1) * self.nx
        n_bounds_u = self.N * self.nu
        self.n_constraints = n_dyn + n_init + n_bounds_x + n_bounds_u

        # OSQP Solver instance
        self.prob = osqp.OSQP()

        # Initialize Persistent Matrices
        self._init_solver_structures()

        # State tracking for updates
        self.prev_quat = np.array([-999.0] * 4)  # Forces update

        if self.verbose:
            print("OSQP MPC Ready.")
    
    @property
    def dt(self) -> float:
        """Control update interval in seconds (Controller interface)."""
        return self._dt
    
    @property
    def prediction_horizon(self) -> Optional[int]:
        """Prediction horizon (Controller interface)."""
        return self.N

    def _precompute_thruster_forces(self) -> None:
        """Precompute thruster forces and torques in body frame (3D)."""
        # Forces: 12 x 3
        # Torques: 12 x 3
        self.body_frame_forces = np.zeros((12, 3), dtype=np.float64)
        self.body_frame_torques = np.zeros((12, 3), dtype=np.float64)

        for i in range(12):
            thruster_id = i + 1
            if thruster_id not in self.thruster_forces:
                continue

            pos = np.array(self.thruster_positions[thruster_id])
            rel_pos = pos - self.com_offset
            direction = np.array(self.thruster_directions[thruster_id])
            force = self.thruster_forces[thruster_id] * direction

            self.body_frame_forces[i] = force
            self.body_frame_torques[i] = np.cross(rel_pos, force)

    @cached(maxsize=128)
    def _compute_rotation_matrix(self, quat_tuple: tuple) -> np.ndarray:
        """
        Compute rotation matrix from quaternion (cached).
        
        Args:
            quat_tuple: Quaternion as tuple (for hashing)
            
        Returns:
            3x3 rotation matrix
        """
        import mujoco

        qv = np.array(quat_tuple)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, qv)
        return R.reshape(3, 3)

    def linearize_dynamics(self, x_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics around current state (quaternion).
        
        Uses caching for expensive rotation matrix computation.
        """
        # x = [px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        # Indices:
        # P: 0-2 (3)
        # Q: 3-6 (4)
        # V: 7-9 (3)
        # W: 10-12 (3)

        qv = x_current[3:7]  # Current quaternion

        # Round quaternion to reduce cache size (binning for similar orientations)
        quat_rounded = tuple(np.round(qv, decimals=3))
        
        # Use cached rotation matrix computation
        R = self._compute_rotation_matrix(quat_rounded)

        # State transition A (13 x 13)
        A = np.eye(13)

        # dPos/dVel = I * dt
        A[0, 7] = self.dt
        A[1, 8] = self.dt
        A[2, 9] = self.dt

        # dQuat/dOmega = 0.5 * G(q) * dt
        w, x, y, z = qv[0], qv[1], qv[2], qv[3]
        G = 0.5 * np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]]) * self.dt
        A[3:7, 10:13] = G

        # Control input B (13 x 12)
        B = np.zeros((13, 12))

        for i in range(12):
            # Velocity update (World frame) -> Force rotated
            F_body = self.body_frame_forces[i]
            F_world = R @ F_body
            B[7:10, i] = F_world / self.total_mass * self.dt

            # Angular velocity update (Body frame) -> Torque in body
            T_body = self.body_frame_torques[i]
            B[10:13, i] = T_body / self.moment_of_inertia * self.dt

        return A, B

    def _init_solver_structures(self):
        """Build initial solver structures."""
        # Cost Matrix P
        # Weights: [pos(3), quat(4), vel(3), angvel(3)]
        # Quat weight? If Target is [1,0,0,0], penalize x,y,z components heavily?
        # Q_ang applies to all 4 or just vector part?
        # Let's apply uniformly for now.

        Q_diag = np.concatenate(
            [
                np.full(3, self.Q_pos),
                np.full(4, self.Q_ang),
                np.full(3, self.Q_vel),
                np.full(3, self.Q_angvel),
            ]
        )  # 13 elements

        R_diag = np.full(self.nu, self.R_thrust)

        diags = []
        for _ in range(self.N):
            diags.append(Q_diag)
        diags.append(Q_diag * 10.0)  # Terminal
        for _ in range(self.N):
            diags.append(R_diag)

        self.P = sp.diags(np.concatenate(diags), format="csc")

        # Linear Cost q
        self.q = np.zeros(self.n_vars)

        # Equality Constraints A (Dynamics)
        # B mapping: A matrix data depends on B, which depends on 'q' (orientation).
        self.B_idx_map: Dict[Tuple[int, int], List[int]] = {}
        for r in range(13):  # 13 rows for dynamics
            for c in range(12):  # 12 cols for control
                self.B_idx_map[(r, c)] = []

        row_idx = 0
        dummy_state = np.zeros(13)
        dummy_state[3] = 1.0  # Valid quat
        A_template, B_template = self.linearize_dynamics(dummy_state)

        triples = []

        # Dyn: -A x_k + x_{k+1} - B u_k = 0
        for k in range(self.N):
            x_k_idx = k * self.nx
            x_kp1_idx = (k + 1) * self.nx
            u_k_idx = (self.N + 1) * self.nx + k * self.nu

            # -A
            for r in range(self.nx):
                for c in range(self.nx):
                    if A_template[r, c] != 0:
                        triples.append((row_idx + r, x_k_idx + c, -A_template[r, c]))

            # +I
            for r in range(self.nx):
                triples.append((row_idx + r, x_kp1_idx + r, 1.0))

            # -B
            for r in range(self.nx):
                for c in range(self.nu):
                    # We store -1.0 placeholder, update later
                    triples.append((row_idx + r, u_k_idx + c, -B_template[r, c]))

            row_idx += self.nx

        # Init
        for r in range(self.nx):
            triples.append((row_idx + r, r, 1.0))
        row_idx += self.nx

        # Bounds X
        for k in range(self.N + 1):
            x_k_idx = k * self.nx
            for r in range(self.nx):
                triples.append((row_idx + r, x_k_idx + r, 1.0))
            row_idx += self.nx

        # Bounds U
        for k in range(self.N):
            u_k_idx = (self.N + 1) * self.nx + k * self.nu
            for r in range(self.nu):
                triples.append((row_idx + r, u_k_idx + r, 1.0))
            row_idx += self.nu

        # Build A
        rows = [t[0] for t in triples]
        cols = [t[1] for t in triples]
        vals = [t[2] for t in triples]

        self.A = sp.csc_matrix((vals, (rows, cols)), shape=(self.n_constraints, self.n_vars))
        self.A.sort_indices()

        # Map B indices
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
                        self.B_idx_map[(r, c)].append(start_ptr + match[0])

        # Bounds l/u
        self.l = np.zeros(self.n_constraints)
        self.u = np.zeros(self.n_constraints)

        # Control bounds [0, 1]
        ctrl_idx = self.n_constraints - self.N * self.nu
        self.l[ctrl_idx:] = 0.0
        self.u[ctrl_idx:] = 1.0  # Max normalized thrust

        # State Bounds
        # default to inf
        state_idx_start = self.N * self.nx + self.nx
        state_idx_end = state_idx_start + (self.N + 1) * self.nx

        self.l[state_idx_start:state_idx_end] = -1e20
        self.u[state_idx_start:state_idx_end] = 1e20

        # Setup OSQP
        self.prob.setup(
            self.P, self.q, self.A, self.l, self.u, verbose=False, time_limit=self.solver_time_limit
        )

    def _update_A_data(self, B_dyn: np.ndarray):
        """Update B-matrix part of A."""
        for r in range(self.nx):
            for c in range(self.nu):
                val = -B_dyn[r, c]
                indices = self.B_idx_map[(r, c)]
                for idx in indices:
                    self.A.data[idx] = val

    def get_solver_stats(self) -> Dict[str, Any]:
        """Get solver performance statistics (Controller interface)."""
        if not self.solve_times:
            return {
                "solve_count": 0,
                "average_solve_time": 0.0,
                "max_solve_time": 0.0,
            }
        return {
            "solve_times": self.solve_times.copy(),
            "solve_count": len(self.solve_times),
            "average_solve_time": sum(self.solve_times) / len(self.solve_times),
            "max_solve_time": max(self.solve_times),
        }
    
    @with_error_context("MPC solve", reraise=True)
    def get_control_action(
        self,
        x_current: np.ndarray,
        x_target: np.ndarray,
        previous_thrusters: Optional[np.ndarray] = None,
        x_target_trajectory: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal control action."""
        start_time = time.time()

        # Ensure x_current is 13-dim
        if len(x_current) < 13:
            # Expand? Assuming caller (simulation.py) already updated.
            pass

        # Prepare targets (identity mapping now)
        x_curr_mpc = x_current  # StateConverter.sim_to_mpc(x_current) is identity
        x_targ_mpc = x_target

        # Dynamics update if Quat changed significantly
        quat = x_current[3:7]
        if np.linalg.norm(quat - self.prev_quat) > 0.05:
            _, B_dyn = self.linearize_dynamics(x_current)
            self._update_A_data(B_dyn)
            self.prob.update(Ax=self.A.data)
            self.prev_quat = quat.copy()

        # Update Cost Vector q
        # J = 0.5 x' P x + q' x
        # min ||x - x_ref||^2_Q = x'Qx - 2 x_ref' Q x
        # So q = -Q * x_ref

        self.q.fill(0.0)

        # Use precomputed Q_diag
        Q_diag = self.Q_diag

        # Terminal weight
        Q_term = Q_diag * 10.0

        if x_target_trajectory is not None:
            # Track trajectory
            for k in range(self.N):
                idx = k * self.nx
                ref = x_target_trajectory[min(k, len(x_target_trajectory) - 1)]
                self.q[idx : idx + self.nx] = -Q_diag * ref
            # Terminal
            idx = self.N * self.nx
            ref = x_target_trajectory[-1]
            self.q[idx : idx + self.nx] = -Q_term * ref
        else:
            # Fixed setpoint
            ref = x_targ_mpc
            for k in range(self.N):
                idx = k * self.nx
                self.q[idx : idx + self.nx] = -Q_diag * ref
            idx = self.N * self.nx
            self.q[idx : idx + self.nx] = -Q_term * ref

        # Update init constraint
        init_start = self.N * self.nx
        self.l[init_start : init_start + self.nx] = x_curr_mpc
        self.u[init_start : init_start + self.nx] = x_curr_mpc

        self.prob.update(q=self.q, l=self.l, u=self.u)

        # Solve
        try:
            res = self.prob.solve()
        except Exception as e:
            logger.error(f"OSQP solver raised exception: {e}")
            raise OptimizationError("solver_exception", f"OSQP solver failed: {e}")

        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)

        # Check for timeout
        if solve_time > self.solver_time_limit:
            logger.warning(
                f"MPC solve time ({solve_time:.3f}s) exceeded limit ({self.solver_time_limit:.3f}s)"
            )
            # Don't raise - use fallback instead
            return self._get_fallback_control(x_curr_mpc, x_targ_mpc), {
                "status": -1,
                "solve_time": solve_time,
                "timeout": True,
            }

        if res.info.status not in ["solved", "solved_inaccurate"]:
            status_msg = res.info.status if hasattr(res.info, "status") else "unknown"
            logger.warning(f"MPC solver status: {status_msg}")
            raise OptimizationError(status_msg, f"Solver returned status: {status_msg}")

        u_idx = (self.N + 1) * self.nx
        u_opt = res.x[u_idx : u_idx + self.nu]
        u_opt = np.clip(u_opt, 0.0, 1.0)

        return u_opt, {"status": 1, "solve_time": solve_time}

    def _get_fallback_control(self, x_current: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        """3D PD Controller fallback."""
        u = np.zeros(12)
        # Placeholder: Zero thrust
        return u
