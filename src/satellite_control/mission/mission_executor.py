"""
Mission Executor

Executes missions using the MPC controller and MuJoCo simulation.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import mujoco
import numpy as np

from .mission_types import Mission, MissionStatus, Waypoint

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of mission execution."""

    success: bool
    duration: float  # Total time in seconds
    waypoints_reached: int
    total_waypoints: int
    final_position: np.ndarray
    final_error: float  # Distance from final target
    trajectory: List[np.ndarray] = field(default_factory=list)
    message: str = ""


class MissionExecutor:
    """
    Executes missions in MuJoCo simulation.

    Handles:
    - Loading appropriate model
    - Running control loop
    - Tracking progress through waypoints
    - Safety checks (keep-out zones)
    """

    def __init__(
        self,
        model_path: str = None,
        control_dt: float = 0.05,
        sim_dt: float = 0.005,
    ):
        """
        Initialize mission executor.

        Args:
            model_path: Path to MuJoCo XML model
            control_dt: Control loop timestep (seconds)
            sim_dt: Physics timestep (seconds)
        """
        self.model_path = model_path
        self.control_dt = control_dt
        self.sim_dt = sim_dt

        # State
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.controller = None
        self.current_mission: Optional[Mission] = None

        # Callbacks
        self.on_waypoint_reached: Optional[Callable[[int, Waypoint], None]] = None
        self.on_progress: Optional[Callable[[float, np.ndarray], None]] = None

    def load_model(self, model_path: str = None):
        """Load MuJoCo model."""
        if model_path:
            self.model_path = model_path

        if not self.model_path:
            from pathlib import Path

            self.model_path = str(
                Path(__file__).parent.parent.parent.parent.parent / "models" / "satellite_rw.xml"
            )

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        logger.info(f"Loaded model: {self.model_path}")

    def load_controller(self):
        """Load the RW MPC controller."""
        from src.satellite_control.control.rw_mpc_controller import ReactionWheelMPCController
        from src.satellite_control.config.actuator_config import create_actuator_config

        actuator_config = create_actuator_config("reaction_wheels")
        self.controller = ReactionWheelMPCController(
            actuator_config=actuator_config,
            prediction_horizon=30,
            dt=self.control_dt,
        )
        logger.info("Controller loaded")

    def execute(
        self,
        mission: Mission,
        real_time: bool = False,
        timeout: float = 600.0,  # 10 minute max
    ) -> ExecutionResult:
        """
        Execute a mission.

        Args:
            mission: Mission to execute
            real_time: If True, pace simulation to real time
            timeout: Maximum execution time in seconds

        Returns:
            ExecutionResult with success status and metrics
        """
        # Initialize
        if self.model is None:
            self.load_model()
        if self.controller is None:
            self.load_controller()

        self.current_mission = mission
        mission.status = MissionStatus.RUNNING
        mission.current_phase_idx = 0
        mission.current_waypoint_idx = 0

        # Get all waypoints
        all_waypoints = mission.get_all_waypoints()
        if not all_waypoints:
            return ExecutionResult(
                success=False,
                duration=0,
                waypoints_reached=0,
                total_waypoints=0,
                final_position=mission.start_position,
                final_error=0,
                message="No waypoints in mission",
            )

        # Set initial position
        sat_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "satellite")
        self.data.qpos[0:3] = mission.start_position
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Execution state
        waypoint_idx = 0
        current_target = all_waypoints[0]
        waypoints_reached = 0
        hold_start_time = None
        trajectory = [mission.start_position.copy()]

        # Timing
        t = 0.0
        start_wall = time.time()
        last_control_time = -self.control_dt

        logger.info(f"Starting mission: {mission.name}")
        logger.info(f"Total waypoints: {len(all_waypoints)}")

        while t < timeout:
            # Control update
            if t - last_control_time >= self.control_dt:
                # Get current state
                pos = self.data.qpos[0:3].copy()
                quat = self.data.qpos[3:7].copy()
                vel = self.data.qvel[0:3].copy()
                ang_vel = self.data.qvel[3:6].copy()
                rw_speeds = self.data.sensordata[0:3].copy()

                x_current = np.concatenate([pos, quat, vel, ang_vel, rw_speeds])

                # Build target state
                x_target = np.zeros(16)
                x_target[0:3] = current_target.position
                x_target[3] = 1.0  # Identity quaternion

                # Compute control
                u, info = self.controller.get_control_action(x_current, x_target)

                # Apply control
                self.data.ctrl[0] = u[0] * self.controller.max_rw_torque
                self.data.ctrl[1] = u[1] * self.controller.max_rw_torque
                self.data.ctrl[2] = u[2] * self.controller.max_rw_torque

                # Apply thruster forces
                thrust_dirs = np.array(
                    [
                        [-1, 0, 0],
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 1, 0],
                        [0, 0, -1],
                        [0, 0, 1],
                    ]
                )
                net_force = np.zeros(3)
                for i in range(6):
                    net_force += u[3 + i] * self.controller.max_thrust * thrust_dirs[i]

                # Add CW orbital forces
                from src.satellite_control.config.orbital_config import OrbitalConfig
                from src.satellite_control.physics.orbital_dynamics import compute_cw_force

                orbital_config = OrbitalConfig()
                cw_force = compute_cw_force(pos, vel, 10.0, orbital_config)

                self.data.xfrc_applied[sat_body_id, 0:3] = net_force + cw_force

                # Check waypoint reached
                error = np.linalg.norm(pos - current_target.position)

                # Consider waypoint reached if within 5cm
                if error < 0.05:
                    if hold_start_time is None:
                        hold_start_time = t
                        logger.info(f"Reached waypoint {waypoint_idx + 1}: {current_target.name}")

                    # Check hold time
                    if t - hold_start_time >= current_target.hold_time:
                        waypoints_reached += 1

                        if self.on_waypoint_reached:
                            self.on_waypoint_reached(waypoint_idx, current_target)

                        # Move to next waypoint
                        waypoint_idx += 1
                        if waypoint_idx >= len(all_waypoints):
                            # Mission complete
                            mission.status = MissionStatus.COMPLETED
                            break

                        current_target = all_waypoints[waypoint_idx]
                        hold_start_time = None
                        logger.info(f"Moving to waypoint {waypoint_idx + 1}")
                else:
                    hold_start_time = None

                # Safety check: keep-out zone
                dist_to_target = np.linalg.norm(pos)
                if dist_to_target < mission.keep_out_radius * 0.9:
                    logger.warning(f"Keep-out zone violation: {dist_to_target:.2f}m")

                # Progress callback
                if self.on_progress:
                    self.on_progress(t, pos)

                # Log trajectory
                trajectory.append(pos.copy())

                last_control_time = t

            # Physics step
            mujoco.mj_step(self.model, self.data)
            t += self.sim_dt

            # Real-time pacing
            if real_time:
                elapsed_wall = time.time() - start_wall
                if elapsed_wall < t:
                    time.sleep(t - elapsed_wall)

        # Final metrics
        final_pos = self.data.qpos[0:3].copy()
        final_error = np.linalg.norm(final_pos - all_waypoints[-1].position)

        result = ExecutionResult(
            success=(mission.status == MissionStatus.COMPLETED),
            duration=t,
            waypoints_reached=waypoints_reached,
            total_waypoints=len(all_waypoints),
            final_position=final_pos,
            final_error=final_error,
            trajectory=trajectory,
            message=f"Mission {mission.status.value}",
        )

        logger.info(
            f"Mission complete: {result.waypoints_reached}/{result.total_waypoints} waypoints"
        )

        return result

    def abort(self):
        """Abort current mission."""
        if self.current_mission:
            self.current_mission.status = MissionStatus.ABORTED
            logger.info("Mission aborted")
