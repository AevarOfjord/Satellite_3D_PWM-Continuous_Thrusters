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

from src.satellite_control.config import SatelliteConfig
from src.satellite_control.config.models import MPCParams, SatellitePhysicalParams
from src.satellite_control.utils.profiler import mpc_profiler
from src.satellite_control.utils.state_converter import StateConverter

logger = logging.getLogger(__name__)


class MPCController:
    """
    Satellite Model Predictive Controller using OSQP.

    Unified implementation replacing the BaseMPC/PWMMPC hierarchy.
    """

    def __init__(
        self,
        satellite_params: Optional[Union[Dict[str, Any], SatellitePhysicalParams]] = None,
        mpc_params: Optional[Union[Dict[str, Any], MPCParams]] = None,
    ):
        """
        Initialize MPC controller with OSQP and pre-allocate matrices.
        """
        # Load defaults from global Config if needed
        if satellite_params is None:
            satellite_params = SatelliteConfig.get_app_config().physics
        if mpc_params is None:
            mpc_params = SatelliteConfig.get_app_config().mpc

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
        self.com_offset = get_param(satellite_params, "com_offset", "com_offset", np.zeros(2))

        # Ensure com_offset is array (Model gives tuple)
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

        # Adaptive control parameters
        self.damping_zone = get_param(mpc_params, "damping_zone", "damping_zone")
        self.velocity_threshold = get_param(mpc_params, "velocity_threshold", "velocity_threshold")
        self.max_velocity_weight = get_param(
            mpc_params, "max_velocity_weight", "max_velocity_weight"
        )

        # State dimension: [x, y, theta, vx, vy, omega]
        self.nx = 6
        # Control dimension: 8 thrusters
        self.nu = 8

        # Performance tracking
        self.solve_times: list[float] = []

        # Linearization cache
        self._linearization_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Precompute thruster forces
        self._precompute_thruster_forces()

        if SatelliteConfig.VERBOSE_MPC:
            print("OSQP MPC Controller Initializing...")

        # Problem dimensions
        # Variables z = [x_0, ... x_N, u_0, ... u_{N-1}]
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
        self.prev_theta = -999.0  # Force update first time

        if SatelliteConfig.VERBOSE_MPC:
            print("OSQP MPC Ready.")

    def _precompute_thruster_forces(self) -> None:
        """Precompute thruster forces in body frame."""
        self.body_frame_forces = np.zeros((8, 2), dtype=np.float64)
        self.thruster_torques = np.zeros(8, dtype=np.float64)

        for i in range(8):
            thruster_id = i + 1
            pos = np.array(self.thruster_positions[thruster_id])
            rel_pos = pos - self.com_offset
            direction = np.array(self.thruster_directions[thruster_id])
            force = self.thruster_forces[thruster_id] * direction

            self.body_frame_forces[i] = force
            self.thruster_torques[i] = rel_pos[0] * force[1] - rel_pos[1] * force[0]

    def linearize_dynamics(self, x_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize dynamics around current state."""
        theta = x_current[2]
        from src.satellite_control.config import mpc_params

        cache_resolution = mpc_params.MPC_CACHE_RESOLUTION
        key = int(theta / cache_resolution)

        if key in self._linearization_cache:
            return self._linearization_cache[key]

        # State transition A
        A = np.eye(6)
        A[0, 3] = self.dt
        A[1, 4] = self.dt
        A[2, 5] = self.dt

        # Control input B
        B = np.zeros((6, 8))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])

        for i in range(8):
            F_world = R @ self.body_frame_forces[i]
            B[3, i] = F_world[0] / self.total_mass * self.dt
            B[4, i] = F_world[1] / self.total_mass * self.dt
            B[5, i] = self.thruster_torques[i] / self.moment_of_inertia * self.dt

        if len(self._linearization_cache) < 1000:
            self._linearization_cache[key] = (A, B)

        return A, B

    def _init_solver_structures(self):
        """
        Builds the initial sparse structure of P and A matrices.
        Creates mapping indices for fast updates.
        """
        if SatelliteConfig.VERBOSE_MPC:
            print(f"DEBUG: MPC Params: dt={self.dt}, R_thrust={self.R_thrust}")

        # --- 1. Constant P Matrix (Cost) ---
        Q_diag = np.array(
            [self.Q_pos, self.Q_pos, self.Q_ang, self.Q_vel, self.Q_vel, self.Q_angvel]
        )
        R_diag = np.full(self.nu, self.R_thrust)

        diags = []
        for _ in range(self.N):
            diags.append(Q_diag)
        diags.append(Q_diag * 10.0)  # Terminal cost
        for _ in range(self.N):
            diags.append(R_diag)

        self.P = sp.diags(np.concatenate(diags), format="csc")

        # --- 2. Initial q Vector (Linear Cost) ---
        self.q = np.zeros(self.n_vars)

        # --- 3. A Matrix Structure & Mapping ---
        # We need to map where the B-matrix elements land in the final CSC data array
        self.B_idx_map: Dict[Tuple[int, int], List[int]] = {}
        for r in range(self.nx):
            for c in range(self.nu):
                self.B_idx_map[(r, c)] = []

        row_idx = 0

        # a) Dynamics: -A x_k + I x_{k+1} - B u_k = 0
        # CRITICAL: Use ACTUAL kinematics A-matrix template
        dummy_state = np.zeros(self.nx)
        A_template, _ = self.linearize_dynamics(dummy_state)
        dummy_B_val = 1.0

        triples = []

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

            # -B term
            for r in range(self.nx):
                for c in range(self.nu):
                    triples.append((row_idx + r, u_k_idx + c, dummy_B_val))

            row_idx += self.nx

        # b) Init State
        for r in range(self.nx):
            triples.append((row_idx + r, r, 1.0))
        row_idx += self.nx

        # c) State Bounds
        for k in range(self.N + 1):
            x_k_idx = k * self.nx
            for r in range(self.nx):
                triples.append((row_idx + r, x_k_idx + r, 1.0))
            row_idx += self.nx

        # d) Control Bounds
        for k in range(self.N):
            u_k_idx = (self.N + 1) * self.nx + k * self.nu
            for r in range(self.nu):
                triples.append((row_idx + r, u_k_idx + r, 1.0))
            row_idx += self.nu

        # Build Matrix
        rows = [t[0] for t in triples]
        cols = [t[1] for t in triples]
        vals = [t[2] for t in triples]

        self.A = sp.csc_matrix((vals, (rows, cols)), shape=(self.n_constraints, self.n_vars))
        self.A.sort_indices()

        # --- Map B Indices Robustly ---
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

        # --- Initialize vectors l and u ---
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
        x_min[3:5] = -self.max_velocity
        x_max[3:5] = self.max_velocity
        x_min[5] = -self.max_angular_velocity
        x_max[5] = self.max_angular_velocity

        if hasattr(self, "position_bounds") and self.position_bounds:
            x_min[0:2] = -self.position_bounds * 2.0
            x_max[0:2] = self.position_bounds * 2.0

        self.l[state_bound_start:state_bound_end] = np.tile(x_min, self.N + 1)
        self.u[state_bound_start:state_bound_end] = np.tile(x_max, self.N + 1)

        # Setup Solver
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
        """Fast in-place update of A constraints with new B matrix."""
        for r in range(self.nx):
            for c in range(self.nu):
                val = -B_dyn[r, c]
                indices = self.B_idx_map[(r, c)]
                for idx in indices:
                    self.A.data[idx] = val

    def get_control_action(
        self,
        x_current: np.ndarray,
        x_target: np.ndarray,
        previous_thrusters: Optional[np.ndarray] = None,
        x_target_trajectory: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal control action."""
        start_time = time.time()

        # 1. Prepare Targets
        x_curr_mpc, targets_mpc, x_targ_mpc = self.prepare_targets(
            x_current, x_target, x_target_trajectory
        )

        # 2. Update Dynamics (A matrix)
        theta = x_curr_mpc[2]
        if abs(theta - self.prev_theta) > 1e-4:
            _, B_dyn = self.linearize_dynamics(x_curr_mpc)
            self._update_A_data(B_dyn)
            self.prev_theta = theta
            self.prob.update(Ax=self.A.data)

        # 3. Update Linear Cost (q vector)
        self.q.fill(0.0)
        Q_diag = np.array(
            [self.Q_pos, self.Q_pos, self.Q_ang, self.Q_vel, self.Q_vel, self.Q_angvel]
        )

        for k in range(self.N + 1):
            idx_start = k * self.nx
            x_ref = targets_mpc[min(k, len(targets_mpc) - 1)]
            weight = 10.0 if k == self.N else 1.0
            self.q[idx_start : idx_start + self.nx] = -Q_diag * weight * x_ref

        # 4. Update Initial State Constraint
        init_start = self.N * self.nx
        self.l[init_start : init_start + self.nx] = x_curr_mpc
        self.u[init_start : init_start + self.nx] = x_curr_mpc

        # 5. Push Updates
        self.prob.update(q=self.q, l=self.l, u=self.u)

        # 6. Solve
        with mpc_profiler.measure("osqp.solve"):
            res = self.prob.solve()

        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)

        # 7. Extract Result
        if res.info.status != "solved" and res.info.status != "solved_inaccurate":
            logger.warning(f"OSQP Failed: {res.info.status}")
            return self._get_fallback_control(x_curr_mpc, x_targ_mpc), {
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

    def prepare_targets(
        self,
        x_current: np.ndarray,
        x_target: np.ndarray,
        x_target_trajectory: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, list, np.ndarray]:
        """Prepare inputs and wrap angles."""
        x_curr_mpc = StateConverter.sim_to_mpc(x_current)

        targets_mpc = []
        if x_target_trajectory is not None:
            for i in range(len(x_target_trajectory)):
                state = x_target_trajectory[i]
                targets_mpc.append(StateConverter.sim_to_mpc(state))
            x_targ_mpc = targets_mpc[0] if targets_mpc else np.zeros(6)
        else:
            x_targ_mpc = StateConverter.sim_to_mpc(x_target)
            targets_mpc = [x_targ_mpc] * (self.N + 1)

        # Angle wrap
        current_theta = x_curr_mpc[2]
        for k in range(len(targets_mpc)):
            targ = targets_mpc[k]
            diff = targ[2] - current_theta
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            targ[2] = current_theta + diff
            current_theta = targ[2]

        return x_curr_mpc, targets_mpc, x_targ_mpc

    def _get_fallback_control(self, x_current: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        """P-controller fallback."""
        u = np.zeros(8, dtype=np.float64)
        pos_err = x_target[:2] - x_current[:2]
        vel_err = x_target[3:5] - x_current[3:5]
        ang_err = x_target[2] - x_current[2]
        ang_err = np.arctan2(np.sin(ang_err), np.cos(ang_err))

        pos_th = 0.1
        vel_th = 0.05
        ang_th = 0.1

        if ang_err > ang_th:
            u[0] = u[3] = 1.0
        elif ang_err < -ang_th:
            u[1] = u[2] = 1.0

        if abs(ang_err) < np.pi / 4:
            theta = x_current[2]
            c, s = np.cos(theta), np.sin(theta)
            R_sw = np.array([[c, s], [-s, c]])
            pos_err_body = R_sw @ pos_err
            vel_err_body = R_sw @ vel_err

            if pos_err_body[0] > pos_th or vel_err_body[0] > vel_th:
                u[4] = u[5] = 1.0
            elif pos_err_body[0] < -pos_th or vel_err_body[0] < -vel_th:
                u[0] = u[1] = 1.0

            if pos_err_body[1] > pos_th or vel_err_body[1] > vel_th:
                u[2] = u[3] = 1.0
            elif pos_err_body[1] < -pos_th or vel_err_body[1] < -vel_th:
                u[6] = u[7] = 1.0
        return u
