"""
MuJoCo-based Satellite Physics Simulation

Drop-in replacement for SatelliteThrusterTester using MuJoCo physics engine.
Provides same interface for compatibility with simulation.py infrastructure.

Features:
- Accurate physics simulation with RK4 integration
- Realistic thruster dynamics with valve delays and ramp-up
- Damping and disturbance modeling
- Planar (2D) motion constraint
- Same interface as testing_environment.SatelliteThrusterTester
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import matplotlib.pyplot as plt
import mujoco
from typing import Any, List, Optional, Set, Tuple, Union
from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz
import numpy as np
from mujoco import viewer as mujoco_viewer

# V4.0.0: SatelliteConfig removed - use AppConfig only
from src.satellite_control.config.models import AppConfig, SatellitePhysicalParams, SimulationParams

from .backend import SimulationBackend


class MuJoCoSatelliteSimulator(SimulationBackend):
    """
    MuJoCo-based satellite physics simulator.

    Drop-in replacement for SatelliteThrusterTester with MuJoCo backend.
    Provides same interface for compatibility with simulation.py.
    """

    def __init__(
        self,
        model_path: str = "models/satellite_3d.xml",
        use_mujoco_viewer: bool = True,
        app_config: Optional[AppConfig] = None,
    ):
        """Initialize MuJoCo simulation (V4.0.0: app_config required).

        Args:
            model_path: Path to MuJoCo XML model file
            use_mujoco_viewer: If True, use MuJoCo's native viewer for
                visualization. If False, use matplotlib
                (for headless/testing)
            app_config: AppConfig (required in V4.0.0, no fallback)
        """
        # V4.0.0: app_config is required
        if app_config is None:
            from src.satellite_control.config.simulation_config import SimulationConfig
            app_config = SimulationConfig.create_default().app_config
        
        # Load MuJoCo model
        model_full_path = Path(model_path)
        if not model_full_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_full_path))
        self.data = mujoco.MjData(self.model)

        # Visualization mode
        self.use_mujoco_viewer = use_mujoco_viewer
        self.viewer = None
        self.viewer_paused = False
        self.viewer_step_request = False

        # V4.0.0: app_config is required (no fallback)
        self._app_config = app_config

        # Store physics and simulation params for easy access
        self._physics_params = self._app_config.physics
        self._simulation_params = self._app_config.simulation

        # Configuration from AppConfig (V4.0.0: required, no fallback)
        self.satellite_size = self._physics_params.satellite_size
        self.total_mass = self._physics_params.total_mass
        self.moment_of_inertia = self._physics_params.moment_of_inertia
        self.com_offset = np.array(self._physics_params.com_offset)

        # Thruster configuration
        self.thrusters = self._physics_params.thruster_positions
        self.thruster_forces = self._physics_params.thruster_forces.copy()
        self.thruster_directions = self._physics_params.thruster_directions

        # Realistic physics flags
        self.use_realistic_physics = self._physics_params.use_realistic_physics
        self.linear_damping_coeff = self._physics_params.damping_linear
        self.rotational_damping_coeff = self._physics_params.damping_angular

        # Active thrusters tracking (same as SatelliteThrusterTester)
        self.active_thrusters: Set[int] = set()
        self.thruster_activation_time: Dict[int, float] = {}
        self.thruster_deactivation_time: Dict[int, float] = {}
        self.thruster_levels: Dict[int, float] = {}  # Track thrust level [0.0, 1.0]

        # Simulation timing
        self._dt = self._simulation_params.dt
        self.model.opt.timestep = self._dt  # Override XML timestep
        self._simulation_time = 0.0
        self.last_time = time.time()
        # Always use external step control
        self.external_simulation_mode = True

        # Trajectory tracking
        self.trajectory: List[np.ndarray] = []
        self.max_trajectory_points = 200

        # Get actuator indices
        # (8 actuators: thruster1_x, thruster2_x, ..., thruster8_y)
        self.actuator_ids = {}
        thruster_axis_map = {
            1: "x",
            2: "x",  # Thrusters 1-2 on X axis
            3: "y",
            4: "y",  # Thrusters 3-4 on Y axis
            5: "x",
            6: "x",  # Thrusters 5-6 on X axis
            7: "y",
            8: "y",  # Thrusters 7-8 on Y axis
        }

        for i in range(1, 9):
            axis = thruster_axis_map[i]
            actuator_name = f"thruster{i}_{axis}"
            try:
                self.actuator_ids[i] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
            except Exception as e:
                print(f"Warning: Could not find actuator {actuator_name}: {e}")
                self.actuator_ids[i] = i - 1  # Fallback to sequential indexing

        # Get sensor indices
        try:
            self.sensor_x_pos = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "x_pos")
            self.sensor_y_pos = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "y_pos")
            self.sensor_theta = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, "z_rot_pos"
            )
            self.sensor_x_vel = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "x_vel")
            self.sensor_y_vel = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "y_vel")
            self.sensor_omega = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, "z_rot_vel"
            )
        except Exception as e:
            print(f"Warning: Could not get sensor IDs: {e}")
            # Fallback to direct qpos/qvel access
            self.sensor_x_pos = 0
            self.sensor_y_pos = 1
            self.sensor_theta = 2
            self.sensor_x_vel = 0
            self.sensor_y_vel = 1
            self.sensor_omega = 2

        # Thruster colors (for visualization compatibility)
        # Thruster colors (for visualization compatibility)
        self.thruster_colors = {
            1: "#FF6B6B",  # Red
            2: "#4ECDC4",  # Teal
            3: "#45B7D1",  # Blue
            4: "#96CEB4",  # Green
            5: "#FFEAA7",  # Yellow
            6: "#DDA0DD",  # Plum
            7: "#98D8C8",  # Mint
            8: "#F7DC6F",  # Light Yellow
            9: "#00FFFF",  # Cyan
            10: "#00FFFF",  # Cyan
            11: "#FF00FF",  # Magenta
            12: "#FF00FF",  # Magenta
        }

        # Initialize state
        mujoco.mj_resetData(self.model, self.data)

        # --------------------------------------------------------------------
        # FORCE CONFIG PARAMS INTO MUJOCO MODEL
        # This fixes the "Model Mismatch" where Config says CoM=(0,0) but
        # the XML file has CoM=(0.005, 0.005), causing the Controller to fight.
        # --------------------------------------------------------------------
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "satellite")

            # 1. Update Mass
            self.model.body_mass[body_id] = self.total_mass

            # 2. Update Inertia (Diagonal approximation for 3D,
            # Z-axis matters for 2D)
            # MuJoCo uses full inertia tensor, but we can update logical
            # diagonal. For a 2D planar sat, Ix and Iy don't matter much,
            # Iz is key. We assume a cube/cylinder distribution:
            # Ix=Iy=Inertia/2, Iz=Inertia (Just setting the full vector)
            self.model.body_inertia[body_id, :] = [
                self.moment_of_inertia / 2,  # Ix
                self.moment_of_inertia / 2,  # Iy
                self.moment_of_inertia,  # Iz (The one that counts)
            ]

            # 3. Update Center of Mass (ipos)
            # This is the "Relative position of center of mass"
            self.model.body_ipos[body_id, 0] = self.com_offset[0]
            self.model.body_ipos[body_id, 1] = self.com_offset[1]
            # self.model.body_ipos[body_id, 2] unchanged (Z)

            print("  Applied Config override to MuJoCo Model:")
            print(f"    - Mass: {self.model.body_mass[body_id]}")
            print(f"    - Inertia: {self.model.body_inertia[body_id]}")
            print(f"    - CoM: {self.model.body_ipos[body_id]}")

        except Exception as e:
            print(f"  Warning: Failed to override MuJoCo physics " f"parameters: {e}")
            print("  Simulation will use XML default parameters " "(Risk of Model Mismatch!)")

        mujoco.mj_forward(self.model, self.data)

        # Setup visualization
        if self.use_mujoco_viewer:
            self.setup_mujoco_viewer()
        else:
            # Setup matplotlib axes for visualization compatibility
            # (fallback mode)
            self.setup_plot()

        print("MuJoCo satellite simulator initialized")
        print(f"  Model: {model_path}")
        viz_mode = "MuJoCo native viewer" if use_mujoco_viewer else "matplotlib (legacy)"
        print(f"  Visualization: {viz_mode}")
        print(f"  Timestep: {self.model.opt.timestep:.4f}s")
        print(f"  Mass: {self.total_mass:.2f} kg")
        print(f"  Inertia: {self.moment_of_inertia:.3f} kg*m^2")
        print(f"  COM offset: ({self.com_offset[0]:.6f}, " f"{self.com_offset[1]:.6f}) m")

    def set_thruster_level(self, thruster_id: int, level: float):
        """Set the thrust level (0.0 to 1.0) for a specific thruster."""
        self.thruster_levels[thruster_id] = max(0.0, min(1.0, level))

        # Maintain compatibility with set/active_thrusters for binary checks
        if level > 0.01:
            if thruster_id not in self.active_thrusters:
                self.activate_thruster(thruster_id)
        else:
            if thruster_id in self.active_thrusters:
                self.deactivate_thruster(thruster_id)

    def setup_plot(self):
        """
        Setup matplotlib axes for visualization compatibility.

        Creates fig, ax_main, and ax_info to match
        SatelliteThrusterTester interface.
        These are used by simulation_visualization.py.
        """
        # Create figure with 3 subplots (controls, main, info)
        # Only main and info are used by simulation_visualization.py
        self.fig, (ax_controls, self.ax_main, self.ax_info) = plt.subplots(
            1, 3, figsize=(20, 8), gridspec_kw={"width_ratios": [2, 3, 2]}
        )

        # Setup main plot
        self.ax_main.set_xlim(-3.0, 3.0)
        self.ax_main.set_ylim(-3.0, 3.0)
        self.ax_main.set_aspect("equal")
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
        self.ax_main.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")

        # Setup info panel
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis("off")

        # Controls panel (not used but needed for compatibility)
        ax_controls.set_xlim(0, 1)
        ax_controls.set_ylim(0, 1)
        ax_controls.axis("off")

        # Set window title
        try:
            if hasattr(self.fig.canvas, "manager") and self.fig.canvas.manager is not None:
                self.fig.canvas.manager.set_window_title("MuJoCo Satellite Simulation")
        except AttributeError:
            pass  # Window title not supported on this backend

    @property
    def dt(self) -> float:
        """Physics timestep in seconds (SimulationBackend interface)."""
        return self._dt

    @property
    def simulation_time(self) -> float:
        """Current simulation time in seconds (SimulationBackend interface)."""
        return self._simulation_time

    @simulation_time.setter
    def simulation_time(self, value: float) -> None:
        """Set simulation time (SimulationBackend interface)."""
        self._simulation_time = value

    def setup_mujoco_viewer(self):
        """
        Setup MuJoCo's native passive viewer for real-time visualization.

        This provides much better performance than matplotlib and includes:
        - Real-time 3D rendering
        - Interactive camera controls
        - Native physics visualization
        """
        try:
            # Ensure an OpenGL backend is selected (helps on macOS)
            os.environ.setdefault("MUJOCO_GL", "glfw")

            def _handle_keypress(key: int) -> None:
                # Space or P: toggle pause. '.' or 'S': single-step.
                if key in (ord(" "), ord("p"), ord("P")):
                    self.viewer_paused = not self.viewer_paused
                elif key in (ord("."), ord("s"), ord("S")):
                    self.viewer_step_request = True
                    self.viewer_paused = True

            # Launch passive viewer (non-blocking)
            self.viewer = mujoco_viewer.launch_passive(
                self.model, self.data, key_callback=_handle_keypress
            )

            # Configure camera for top-down view of planar motion
            if self.viewer is not None:
                # Fixed corner view covering entire 3x3 workspace
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.0]  # Center
                self.viewer.cam.distance = 6.0  # User requested 6m
                self.viewer.cam.elevation = -90  # Top-down view for best planar visibility
                self.viewer.cam.azimuth = 0  # Align with XY axis

            print("  MuJoCo viewer launched successfully")
            print("  Viewer controls: Space/P = pause/resume, S/. = single-step")

        except Exception as e:
            print(
                f"  Warning: Could not launch MuJoCo viewer: {e}\n"
                "  Tip: Install an OpenGL backend "
                "(e.g., `pip install glfw` on macOS) "
                "and run with MUJOCO_GL=glfw. "
                "Falling back to matplotlib."
            )
            self.use_mujoco_viewer = False
            self.viewer = None
            self.setup_plot()

    def is_viewer_paused(self) -> bool:
        """Return True if the MuJoCo viewer has paused the sim."""
        return self.viewer is not None and self.viewer_paused

    def consume_viewer_step(self) -> bool:
        """Consume a pending single-step request."""
        if self.viewer_step_request:
            self.viewer_step_request = False
            return True
        return False

    def sync_viewer(self) -> None:
        """Sync the MuJoCo viewer if active."""
        if self.use_mujoco_viewer and self.viewer is not None:
            self.viewer.sync()

    @property
    def position(self) -> np.ndarray:
        """Get current position [x, y, z]."""
        try:
            # Try getting from sensor if available (not defined in xml yet), else qpos
            # In free joint, qpos[0:3] is position
            x = self.data.qpos[0]
            y = self.data.qpos[1]
            z = self.data.qpos[2]
            return np.array([x, y, z])
        except Exception:
            return np.zeros(3)

    @position.setter
    def position(self, value: np.ndarray):
        """Set position [x, y, z]."""
        if len(value) < 3:
            pos = np.zeros(3)
            pos[: len(value)] = value
            value = pos
        self.data.qpos[0] = value[0]
        self.data.qpos[1] = value[1]
        self.data.qpos[2] = value[2]
        mujoco.mj_forward(self.model, self.data)

    @property
    def velocity(self) -> np.ndarray:
        """Get current linear velocity [vx, vy, vz]."""
        try:
            # Free joint: qvel[0:3] is linear velocity
            vx = self.data.qvel[0]
            vy = self.data.qvel[1]
            vz = self.data.qvel[2]
            return np.array([vx, vy, vz])
        except Exception:
            return np.zeros(3)

    @velocity.setter
    def velocity(self, value: np.ndarray):
        """Set velocity [vx, vy, vz]."""
        if len(value) < 3:
            vel = np.zeros(3)
            vel[: len(value)] = value
            value = vel
        self.data.qvel[0] = value[0]
        self.data.qvel[1] = value[1]
        self.data.qvel[2] = value[2]
        mujoco.mj_forward(self.model, self.data)

    @property
    def quaternion(self) -> np.ndarray:
        """Get current orientation quaternion [w, x, y, z]."""
        # Free joint: qpos[3:7] is quaternion (w, x, y, z)
        return self.data.qpos[3:7].copy()

    @property
    def angle(self) -> float:
        """Deprecated: Get Z-rotation angle. Kept for legacy compatibility."""
        # Convert quat to Z-angle approx
        # For planar this works. For 3D, it's just Yaw.
        # w = q[0], z = q[3]
        q = self.quaternion
        # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    @angle.setter
    def angle(self, value: Union[Tuple[float, float, float], np.ndarray]):
        """
        Set orientation.
        Expects (Roll, Pitch, Yaw) in radians.
        """
        if not (isinstance(value, (tuple, list, np.ndarray)) and len(value) == 3):
            raise ValueError("Orientation must be a 3-element Euler tuple (roll, pitch, yaw).")

        quat = euler_xyz_to_quat_wxyz(value)
        self.data.qpos[3:7] = quat

        mujoco.mj_forward(self.model, self.data)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Get current angular velocity [wx, wy, wz] in rad/s."""
        # Free joint: qvel[3:6] is angular velocity
        return self.data.qvel[3:6].copy()

    @angular_velocity.setter
    def angular_velocity(self, value: Any):
        """Set angular velocity. If scalar, assumes Z-axis rate."""
        if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
            self.data.qvel[3] = 0.0
            self.data.qvel[4] = 0.0
            self.data.qvel[5] = float(value)
        else:
            # Assume 3-vector
            v = np.array(value, dtype=float)
            if v.shape == (3,):
                self.data.qvel[3] = v[0]
                self.data.qvel[4] = v[1]
                self.data.qvel[5] = v[2]
        mujoco.mj_forward(self.model, self.data)

    def activate_thruster(self, thruster_id: int):
        """Activate a thruster (same interface as SatelliteThrusterTester)."""
        if thruster_id not in self.active_thrusters:
            self.active_thrusters.add(thruster_id)
            self.thruster_activation_time[thruster_id] = self.simulation_time
            # Clear deactivation time if re-activating
            if thruster_id in self.thruster_deactivation_time:
                del self.thruster_deactivation_time[thruster_id]

    def deactivate_thruster(self, thruster_id: int):
        """Deactivate a thruster (SatelliteThrusterTester interface)."""
        if thruster_id in self.active_thrusters:
            self.active_thrusters.remove(thruster_id)
            self.thruster_deactivation_time[thruster_id] = self.simulation_time

    def calculate_forces_and_torques(self):
        """
        Calculate net force and torque from active thrusters in 3D.

        Returns:
            tuple: (net_force, net_torque) where both are 3D numpy arrays
        """
        net_force = np.zeros(3)
        net_torque = np.zeros(3)

        # Current orientation quaternion
        quat = self.quaternion

        # Calculate net force and torque
        for thruster_id in range(1, 13):
            force_magnitude = self.get_thrust_force(thruster_id)

            if force_magnitude > 0:
                # Get thruster direction in body frame
                dir_body = np.array(self.model.site_rgba[0])  # Dummy init

                # Fetch from config (we can't easily rely on self.thrusters if it's outdated)
                # But self.thrusters was initialized from Config.
                # Assuming self.thrusters / self.com_offset matching config.

                # We need to rely on the config data stored in 'self'
                # But 'self.thrusters' in __init__ is just 'THRUSTER_POSITIONS'
                # We need DIRECTIONS too.
                # Let's import config to be safe
                pass

        # Since this method is tricky to partially replace without full context of helper data,
        # I'll rely on update_physics to do the heavy lifting and this method checks applied forces?
        # No, update_physics calls this? No, update_physics computes it internally.
        # simulation.py calls this for visualization.

        # Re-implementing logic (V4.0.0: uses self.thrusters and self.thruster_directions from app_config)

        for thruster_id in range(1, 13):
            force_magnitude = self.get_thrust_force(thruster_id)
            if force_magnitude > 0:
                # Local pos and dir
                pos_body = (
                    np.array(self.thrusters[thruster_id]) - self.com_offset
                )
                dir_body = np.array(self.thruster_directions[thruster_id])

                # Rotate to world frame
                dir_world = np.zeros(3)
                mujoco.mju_rotVecQuat(dir_world, dir_body, quat)

                # Force in world
                force_world = force_magnitude * dir_world
                net_force += force_world

                # Torque = r x F (in body or world? rigid body dynamics usually world or body depending on frame)
                # MuJoCo applies torques in body frame usually? No, xfrc_applied is global?
                # Actually xfrc_applied is in global frame (Cartesian force/torque).

                # Position relative to CoM in world
                pos_world_rel = np.zeros(3)
                mujoco.mju_rotVecQuat(pos_world_rel, pos_body, quat)

                torque = np.cross(pos_world_rel, force_world)
                net_torque += torque

        return net_force, net_torque

    def get_thrust_force(self, thruster_id: int) -> float:
        """
        Get current thrust force.

        Thrusters are BINARY: either full nominal force (8N) or OFF (0N).
        PWM controls timing (how long thruster is ON), not force magnitude.
        """
        nominal_force = self.thruster_forces.get(thruster_id, 0.0)

        # Get binary active state from thruster_levels (1.0 = ON, 0.0 = OFF)
        level = self.thruster_levels.get(thruster_id, 0.0)
        is_active = level > 0.01 or thruster_id in self.active_thrusters

        if not self.use_realistic_physics:
            # Binary thrust: either full force or zero
            return nominal_force if is_active else 0.0

        # Handle deactivation (valve closing + ramp-down)
        if thruster_id in self.thruster_deactivation_time:
            deactivation_time = self.thruster_deactivation_time[thruster_id]
            time_since_deactivation = self.simulation_time - deactivation_time

            # Phase 1: Valve closing delay - maintains full thrust
            # V4.0.0: Use hardcoded defaults (TODO: move to AppConfig.physics in future)
            valve_delay = 0.05  # 50ms valve delay
            rampup_time = 0.015  # 15ms ramp-up time
            if time_since_deactivation < valve_delay:
                force = nominal_force  # Assumes valve was fully open
            # Phase 2: Ramp-down
            elif time_since_deactivation < (valve_delay + rampup_time):
                rampdown_progress = (time_since_deactivation - valve_delay) / rampup_time
                force = nominal_force * (1.0 - rampdown_progress)
            else:
                # Fully off - remove from deactivation tracking
                del self.thruster_deactivation_time[thruster_id]
                force = 0.0

            # Scale by level if we are just throttling down, but usually
            # deactivation means going to 0
            # For simplicity in this complex transition, we treat deactivation
            # as going from MAX to 0.

            # Add noise
            # V4.0.0: Use hardcoded default (TODO: move to AppConfig.physics in future)
            noise_std = 0.0  # No noise by default
            noise_factor = 1.0 + np.random.normal(0, noise_std)
            return force * noise_factor

        # Handle activation (valve opening + ramp-up)
        # Check either active set OR positive level
        if thruster_id not in self.active_thrusters and level <= 0.001:
            return 0.0

        activation_time = self.thruster_activation_time.get(thruster_id, self.simulation_time)
        time_since_activation = self.simulation_time - activation_time

        # Phase 1: Valve opening delay - no thrust
        # V4.0.0: Use hardcoded defaults (TODO: move to AppConfig.physics in future)
        valve_delay = 0.05  # 50ms valve delay
        rampup_time = 0.015  # 15ms ramp-up time
        if time_since_activation < valve_delay:
            return 0.0

        # Phase 2: Ramp-up
        rampup_end = valve_delay + rampup_time
        if time_since_activation < rampup_end:
            rampup_progress = (time_since_activation - valve_delay) / rampup_time
            force = nominal_force * rampup_progress
        else:
            # Phase 3: Full thrust
            force = nominal_force

        # Add noise
        # V4.0.0: Use hardcoded default (TODO: move to AppConfig.physics in future)
        noise_std = 0.0  # No noise by default
        noise_factor = 1.0 + np.random.normal(0, noise_std)
        return force * noise_factor

    def update_physics(self, dt: Optional[float] = None):
        """
        Update satellite physics for one timestep.

        This is the main interface used by simulation.py.
        Matches SatelliteThrusterTester.

        Args:
            dt: Time step (uses self.dt if None)
        """
        if dt is None:
            dt = self.dt

        # Get body id for satellite
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "satellite")
        except Exception:
            body_id = 1  # Fallback

        # Zero out applied forces
        self.data.xfrc_applied[body_id, :] = 0

        # Apply thruster forces directly via xfrc_applied
        # We need to transform forces from body frame to world frame
        quat = self.quaternion

        # Calculate and apply forces/torques
        # Note: We duplicate logic from calculate_forces_and_torques here or call it?
        # Calling it implies overhead? Let's inline for clarity or just call it.
        # Calling it is safer for consistency.

        net_f, net_t = self.calculate_forces_and_torques()

        self.data.xfrc_applied[body_id, 0] = net_f[0]
        self.data.xfrc_applied[body_id, 1] = net_f[1]
        self.data.xfrc_applied[body_id, 2] = net_f[2]
        self.data.xfrc_applied[body_id, 3] = net_t[0]
        self.data.xfrc_applied[body_id, 4] = net_t[1]
        self.data.xfrc_applied[body_id, 5] = net_t[2]

        # Add damping and disturbances
        if self.use_realistic_physics:
            # Linear damping
            drag_force = -self.linear_damping_coeff * self.velocity
            self.data.xfrc_applied[body_id, 0] += drag_force[0]
            self.data.xfrc_applied[body_id, 1] += drag_force[1]

            # Rotational damping
            drag_torque = -self.rotational_damping_coeff * self.angular_velocity
            self.data.xfrc_applied[body_id, 5] += drag_torque

            # Random disturbances
            # V4.0.0: Use hardcoded defaults (TODO: move to AppConfig.physics in future)
            enable_disturbances = False  # Disabled by default
            if enable_disturbances:
                disturbance_force_std = 0.0  # No force disturbance by default
                disturbance_torque_std = 0.0  # No torque disturbance by default
                disturbance_force = np.random.normal(0, disturbance_force_std, 2)
                self.data.xfrc_applied[body_id, 0] += disturbance_force[0]
                self.data.xfrc_applied[body_id, 1] += disturbance_force[1]

                disturbance_torque = np.random.normal(0, disturbance_torque_std)
                self.data.xfrc_applied[body_id, 5] += disturbance_torque

        # Step MuJoCo simulation
        num_steps = max(1, int(dt / self.model.opt.timestep))
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)

        self._simulation_time += dt

        # Update visual elements (thruster glow)
        self._update_visuals()

        # Update trajectory
        self.trajectory.append(self.position.copy())
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)

        # Sync MuJoCo viewer if active
        if self.use_mujoco_viewer and self.viewer is not None:
            self.viewer.sync()

    def _update_visuals(self):
        """Update visual elements based on simulation state."""
        # 1. Update thruster glow
        # material IDs for active/inactive
        try:
            # Get material IDs (not used currently, but could be for
            # future material-based visual updates)
            _ = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "mat_thruster_active")
            _ = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_MATERIAL,
                "mat_thruster_inactive",
            )

            # Map thruster IDs to their visual sites
            # Per xml: thruster1, thruster2, ...
            # Map thruster IDs to their visual sites
            # Per xml: thruster1, thruster2, ...
            for i in range(1, 13):
                site_name = f"thruster{i}"
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)

                if site_id != -1:
                    is_active = i in self.active_thrusters
                    level = self.thruster_levels.get(i, 0.0)

                    if is_active or level > 0.01:
                        # Scale alpha/intensity by thrust level
                        # Minimum alpha 0.4 for visibility when active, max 1.0
                        alpha = 0.4 + 0.6 * level
                        self.model.site_rgba[site_id] = [
                            1.0,
                            0.2,
                            0.2,
                            alpha,
                        ]  # Light Red glow
                    else:
                        # Revert to original colors
                        # This is a bit hacky, hardcoding the colors
                        # from XML or storing them
                        # For now, just dim them significantly
                        original_colors = {i: [0.0, 0.45, 1.0, 0.35] for i in range(1, 13)}
                        self.model.site_rgba[site_id] = original_colors.get(
                            i, [0.0, 0.45, 1.0, 0.35]
                        )

        except Exception:
            # Prevent crashing if visual update fails
            pass

    def get_state(self) -> np.ndarray:
        """
        Get the full current state of the satellite (SimulationBackend interface).
        
        Returns:
            State vector [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13 elements)
        """
        pos = self.position
        quat = self.quaternion
        vel = self.velocity
        ang_vel = self.angular_velocity
        return np.concatenate([pos, quat, vel, ang_vel])

    def set_state(
        self,
        state: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        quaternion: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
        angle: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Set satellite state (SimulationBackend interface with legacy compatibility).
        
        Supports two calling conventions:
        1. set_state(state) - where state is [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13 elements)
           This matches the SimulationBackend interface.
        2. set_state(position=..., velocity=..., quaternion=..., ...) - legacy interface
        
        Args:
            state: Full state vector [13 elements] (SimulationBackend interface)
            position: Position [x, y, z] (legacy interface)
            velocity: Velocity [vx, vy, vz] (legacy interface)
            quaternion: Quaternion [qw, qx, qy, qz] (legacy interface)
            angular_velocity: Angular velocity [wx, wy, wz] (legacy interface)
            angle: Euler angles [roll, pitch, yaw] (legacy interface)
        """
        # If state vector provided as positional argument (SimulationBackend interface)
        # Check if state is provided and no keyword arguments are used
        if state is not None and position is None and velocity is None and quaternion is None and angular_velocity is None and angle is None:
            if len(state) != 13:
                raise ValueError(f"State vector must be 13 elements, got {len(state)}")
            position = state[0:3]
            quaternion = state[3:7]
            velocity = state[7:10]
            angular_velocity = state[10:13]
        
        # Set position (3D)
        if position is not None:
            if len(position) < 3:
                # Pad
                pos3 = np.zeros(3)
                pos3[: len(position)] = position
                position = pos3
            self.data.qpos[0:3] = position

        # Set quaternion (Orientation)
        if quaternion is not None:
            self.data.qpos[3:7] = quaternion
        elif angle is not None:
            self.data.qpos[3:7] = euler_xyz_to_quat_wxyz(angle)
        elif state is not None and position is not None:
            # If state was provided, quaternion was already extracted above
            pass
        # If neither quaternion nor angle provided, leave existing quaternion

        # Set velocity
        if velocity is not None:
            if len(velocity) < 3:
                vel3 = np.zeros(3)
                vel3[: len(velocity)] = velocity
                velocity = vel3
            self.data.qvel[0:3] = velocity

        # Ang Vel
        if angular_velocity is not None:
            self.data.qvel[3:6] = angular_velocity

        # Update MuJoCo internal state
        mujoco.mj_forward(self.model, self.data)

    def reset_state(self):
        """Reset satellite to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.active_thrusters.clear()
        self.thruster_activation_time.clear()
        self.thruster_deactivation_time.clear()
        self._simulation_time = 0.0
        self.trajectory.clear()
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        """Close the MuJoCo viewer if active."""
        if self.viewer is not None:
            try:
                self.viewer.close()
                print("MuJoCo viewer closed")
            except Exception as e:
                print(f"Warning: Error closing viewer: {e}")
            self.viewer = None

    def __del__(self):
        """Cleanup when object is deleted."""
        try:
            self.close()
        except Exception:
            pass


# Alias for easier drop-in replacement
SatelliteThrusterTester = MuJoCoSatelliteSimulator
