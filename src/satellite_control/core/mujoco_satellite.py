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
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import viewer as mujoco_viewer

from src.satellite_control.config import SatelliteConfig


class MuJoCoSatelliteSimulator:
    """
    MuJoCo-based satellite physics simulator.

    Drop-in replacement for SatelliteThrusterTester with MuJoCo backend.
    Provides same interface for compatibility with simulation.py.
    """

    def __init__(
        self,
        model_path: str = "models/satellite_planar.xml",
        use_mujoco_viewer: bool = True,
    ):
        """Initialize MuJoCo simulation.

        Args:
            model_path: Path to MuJoCo XML model file
            use_mujoco_viewer: If True, use MuJoCo's native viewer for
                visualization. If False, use matplotlib
                (for headless/testing)
        """
        # Load MuJoCo model
        model_full_path = Path(model_path)
        if not model_full_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_full_path))
        self.data = mujoco.MjData(self.model)

        # Visualization mode
        self.use_mujoco_viewer = use_mujoco_viewer
        self.viewer = None

        # Configuration from SatelliteConfig
        self.satellite_size = SatelliteConfig.SATELLITE_SIZE
        self.total_mass = SatelliteConfig.TOTAL_MASS
        self.moment_of_inertia = SatelliteConfig.MOMENT_OF_INERTIA
        self.com_offset = np.array(SatelliteConfig.COM_OFFSET)

        # Thruster configuration
        self.thrusters = SatelliteConfig.THRUSTER_POSITIONS
        self.thruster_forces = SatelliteConfig.THRUSTER_FORCES.copy()

        # Active thrusters tracking (same as SatelliteThrusterTester)
        self.active_thrusters: Set[int] = set()
        self.thruster_activation_time: Dict[int, float] = {}
        self.thruster_deactivation_time: Dict[int, float] = {}
        self.thruster_levels: Dict[int, float] = {}  # Track thrust level [0.0, 1.0]

        # Simulation timing
        self.dt = SatelliteConfig.SIMULATION_DT
        self.model.opt.timestep = self.dt  # Override XML timestep
        self.simulation_time = 0.0
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
        self.thruster_colors = {
            1: "#FF6B6B",  # Red
            2: "#4ECDC4",  # Teal
            3: "#45B7D1",  # Blue
            4: "#96CEB4",  # Green
            5: "#FFEAA7",  # Yellow
            6: "#DDA0DD",  # Plum
            7: "#98D8C8",  # Mint
            8: "#F7DC6F",  # Light Yellow
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

            # Launch passive viewer (non-blocking)
            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)

            # Configure camera for top-down view of planar motion
            if self.viewer is not None:
                # Fixed corner view covering entire 3x3 workspace
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.0]  # Center
                self.viewer.cam.distance = 6.0  # User requested 6m
                self.viewer.cam.elevation = -90  # Top-down view for best planar visibility
                self.viewer.cam.azimuth = 0  # Align with XY axis

            print("  MuJoCo viewer launched successfully")

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

    @property
    def position(self) -> np.ndarray:
        """Get current position [x, y]."""
        try:
            x = self.data.sensordata[self.sensor_x_pos]
            y = self.data.sensordata[self.sensor_y_pos]
        except Exception:
            # Fallback to qpos
            x = self.data.qpos[0]
            y = self.data.qpos[1]
        return np.array([x, y])

    @position.setter
    def position(self, value: np.ndarray):
        """Set position [x, y]."""
        self.data.qpos[0] = value[0]
        self.data.qpos[1] = value[1]
        mujoco.mj_forward(self.model, self.data)

    @property
    def velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy]."""
        try:
            vx = self.data.sensordata[self.sensor_x_vel]
            vy = self.data.sensordata[self.sensor_y_vel]
        except Exception:
            # Fallback to qvel
            vx = self.data.qvel[0]
            vy = self.data.qvel[1]
        return np.array([vx, vy])

    @velocity.setter
    def velocity(self, value: np.ndarray):
        """Set velocity [vx, vy]."""
        self.data.qvel[0] = value[0]
        self.data.qvel[1] = value[1]
        mujoco.mj_forward(self.model, self.data)

    @property
    def angle(self) -> float:
        """Get current angle (theta) in radians."""
        try:
            return float(self.data.sensordata[self.sensor_theta])
        except Exception:
            # Fallback to qpos
            return float(self.data.qpos[2])

    @angle.setter
    def angle(self, value: float):
        """Set angle (theta) in radians."""
        self.data.qpos[2] = value
        mujoco.mj_forward(self.model, self.data)

    @property
    def angular_velocity(self) -> float:
        """Get current angular velocity (omega) in rad/s."""
        try:
            return float(self.data.sensordata[self.sensor_omega])
        except Exception:
            # Fallback to qvel
            return float(self.data.qvel[2])

    @angular_velocity.setter
    def angular_velocity(self, value: float):
        """Set angular velocity (omega) in rad/s."""
        self.data.qvel[2] = value
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
        Calculate net force and torque from active thrusters.

        Required for compatibility with simulation.py visualization.
        Returns the forces/torques that would be applied this timestep.

        Returns:
            tuple: (net_force, net_torque) where net_force is [fx, fy]
                   and net_torque is scalar
        """
        net_force = np.array([0.0, 0.0])
        net_torque = 0.0

        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)

        # Thruster positions and directions (from config)
        thruster_pos_body = {
            1: np.array([0.145, 0.06]),
            2: np.array([0.145, -0.06]),
            3: np.array([0.06, -0.145]),
            4: np.array([-0.06, -0.145]),
            5: np.array([-0.145, -0.06]),
            6: np.array([-0.145, 0.06]),
            7: np.array([-0.06, 0.145]),
            8: np.array([0.06, 0.145]),
        }

        thruster_dir_body = {
            1: np.array([-1, 0]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, 1]),
            5: np.array([1, 0]),
            6: np.array([1, 0]),
            7: np.array([0, -1]),
            8: np.array([0, -1]),
        }

        # Calculate net force and torque
        for thruster_id in range(1, 9):
            force_magnitude = self.get_thrust_force(thruster_id)

            if force_magnitude > 0:
                # Get thruster direction in body frame
                dir_body = thruster_dir_body[thruster_id]

                # Transform to world frame
                dir_world = np.array(
                    [
                        cos_theta * dir_body[0] - sin_theta * dir_body[1],
                        sin_theta * dir_body[0] + cos_theta * dir_body[1],
                    ]
                )

                # Add force
                force_world = force_magnitude * dir_world
                net_force += force_world

                # Calculate torque about COM
                pos_body = thruster_pos_body[thruster_id] - self.com_offset
                pos_world = np.array(
                    [
                        cos_theta * pos_body[0] - sin_theta * pos_body[1],
                        sin_theta * pos_body[0] + cos_theta * pos_body[1],
                    ]
                )

                # Torque = r x F (in 2D)
                torque = pos_world[0] * force_world[1] - pos_world[1] * force_world[0]
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

        if not SatelliteConfig.USE_REALISTIC_PHYSICS:
            # Binary thrust: either full force or zero
            return nominal_force if is_active else 0.0

        # Handle deactivation (valve closing + ramp-down)
        if thruster_id in self.thruster_deactivation_time:
            deactivation_time = self.thruster_deactivation_time[thruster_id]
            time_since_deactivation = self.simulation_time - deactivation_time

            # Phase 1: Valve closing delay - maintains full thrust
            if time_since_deactivation < SatelliteConfig.THRUSTER_VALVE_DELAY:
                force = nominal_force  # Assumes valve was fully open
            # Phase 2: Ramp-down
            elif time_since_deactivation < (
                SatelliteConfig.THRUSTER_VALVE_DELAY + SatelliteConfig.THRUSTER_RAMPUP_TIME
            ):
                rampdown_progress = (
                    time_since_deactivation - SatelliteConfig.THRUSTER_VALVE_DELAY
                ) / SatelliteConfig.THRUSTER_RAMPUP_TIME
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
            noise_factor = 1.0 + np.random.normal(0, SatelliteConfig.THRUSTER_FORCE_NOISE_STD)
            return force * noise_factor

        # Handle activation (valve opening + ramp-up)
        # Check either active set OR positive level
        if thruster_id not in self.active_thrusters and level <= 0.001:
            return 0.0

        activation_time = self.thruster_activation_time.get(thruster_id, self.simulation_time)
        time_since_activation = self.simulation_time - activation_time

        # Phase 1: Valve opening delay - no thrust
        if time_since_activation < SatelliteConfig.THRUSTER_VALVE_DELAY:
            return 0.0

        # Phase 2: Ramp-up
        rampup_end = SatelliteConfig.THRUSTER_VALVE_DELAY + SatelliteConfig.THRUSTER_RAMPUP_TIME
        if time_since_activation < rampup_end:
            rampup_progress = (
                time_since_activation - SatelliteConfig.THRUSTER_VALVE_DELAY
            ) / SatelliteConfig.THRUSTER_RAMPUP_TIME
            force = nominal_force * rampup_progress
        else:
            # Phase 3: Full thrust
            force = nominal_force

        # Add noise
        noise_factor = 1.0 + np.random.normal(0, SatelliteConfig.THRUSTER_FORCE_NOISE_STD)
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
        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)

        # Thruster positions (from config)
        # Thruster directions (from config)
        thruster_pos_body = {
            1: np.array([0.145, 0.06]),
            2: np.array([0.145, -0.06]),
            3: np.array([0.06, -0.145]),
            4: np.array([-0.06, -0.145]),
            5: np.array([-0.145, -0.06]),
            6: np.array([-0.145, 0.06]),
            7: np.array([-0.06, 0.145]),
            8: np.array([0.06, 0.145]),
        }

        thruster_dir_body = {
            1: np.array([-1, 0]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, 1]),
            5: np.array([1, 0]),
            6: np.array([1, 0]),
            7: np.array([0, -1]),
            8: np.array([0, -1]),
        }

        # Calculate net force and torque
        for thruster_id in range(1, 9):
            force_magnitude = self.get_thrust_force(thruster_id)

            if force_magnitude > 0:
                # Get thruster direction in body frame
                dir_body = thruster_dir_body[thruster_id]

                # Transform to world frame
                dir_world = np.array(
                    [
                        cos_theta * dir_body[0] - sin_theta * dir_body[1],
                        sin_theta * dir_body[0] + cos_theta * dir_body[1],
                    ]
                )

                # Apply force in world frame
                force_world = force_magnitude * dir_world
                self.data.xfrc_applied[body_id, 0] += force_world[0]
                self.data.xfrc_applied[body_id, 1] += force_world[1]

                # Calculate torque about COM
                pos_body = thruster_pos_body[thruster_id] - self.com_offset
                # Transform position to world frame
                pos_world = np.array(
                    [
                        cos_theta * pos_body[0] - sin_theta * pos_body[1],
                        sin_theta * pos_body[0] + cos_theta * pos_body[1],
                    ]
                )

                # Torque = r x F (in 2D, this is scalar)
                torque = pos_world[0] * force_world[1] - pos_world[1] * force_world[0]
                self.data.xfrc_applied[body_id, 5] += torque

        # Add damping and disturbances
        if SatelliteConfig.USE_REALISTIC_PHYSICS:
            # Linear damping
            drag_force = -SatelliteConfig.LINEAR_DAMPING_COEFF * self.velocity
            self.data.xfrc_applied[body_id, 0] += drag_force[0]
            self.data.xfrc_applied[body_id, 1] += drag_force[1]

            # Rotational damping
            drag_torque = -SatelliteConfig.ROTATIONAL_DAMPING_COEFF * self.angular_velocity
            self.data.xfrc_applied[body_id, 5] += drag_torque

            # Random disturbances
            if SatelliteConfig.ENABLE_RANDOM_DISTURBANCES:
                disturbance_force = np.random.normal(0, SatelliteConfig.DISTURBANCE_FORCE_STD, 2)
                self.data.xfrc_applied[body_id, 0] += disturbance_force[0]
                self.data.xfrc_applied[body_id, 1] += disturbance_force[1]

                disturbance_torque = np.random.normal(0, SatelliteConfig.DISTURBANCE_TORQUE_STD)
                self.data.xfrc_applied[body_id, 5] += disturbance_torque

        # Step MuJoCo simulation
        num_steps = max(1, int(dt / self.model.opt.timestep))
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)

        self.simulation_time += dt

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
            for i in range(1, 9):
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
                        original_colors = {
                            1: [1, 0, 0, 0.3],
                            2: [1, 0.2, 0, 0.3],
                            3: [0, 1, 0, 0.3],
                            4: [0.2, 1, 0, 0.3],
                            5: [0, 0, 1, 0.3],
                            6: [0, 0.2, 1, 0.3],
                            7: [1, 1, 0, 0.3],
                            8: [1, 0.8, 0, 0.3],
                        }
                        self.model.site_rgba[site_id] = original_colors.get(i, [0.5, 0.5, 0.5, 0.3])

        except Exception:
            # Prevent crashing if visual update fails
            pass

    def set_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        angle: float,
        angular_velocity: float,
    ):
        """Set satellite state directly (for initialization)."""
        # Set position
        self.data.qpos[0] = position[0]  # x
        self.data.qpos[1] = position[1]  # y
        self.data.qpos[2] = angle  # theta

        # Set velocity
        self.data.qvel[0] = velocity[0]  # vx
        self.data.qvel[1] = velocity[1]  # vy
        self.data.qvel[2] = angular_velocity  # omega

        # Update MuJoCo internal state
        mujoco.mj_forward(self.model, self.data)

    def reset_state(self):
        """Reset satellite to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.active_thrusters.clear()
        self.thruster_activation_time.clear()
        self.thruster_deactivation_time.clear()
        self.simulation_time = 0.0
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
