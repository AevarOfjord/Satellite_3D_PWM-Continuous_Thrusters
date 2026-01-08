"""
Simulation Loop Module

Handles the main simulation loop execution.
Extracted from simulation.py to improve modularity.

This module handles:
- Main loop setup and initialization
- Animation mode (matplotlib) vs batch mode (MuJoCo/headless)
- Step-by-step simulation execution
- Loop termination conditions
- Data saving and cleanup
"""

import logging
import time
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# V3.0.0: No longer import SatelliteConfig
from src.satellite_control.config import use_structured_config

logger = logging.getLogger(__name__)


class SimulationLoop:
    """
    Handles the main simulation loop execution.
    
    This class encapsulates all the loop logic that was previously
    in SatelliteMPCLinearizedSimulation._run_simulation_with_globals
    and update_simulation.
    """

    def __init__(self, simulation: Any):
        """
        Initialize the simulation loop.
        
        Args:
            simulation: The SatelliteMPCLinearizedSimulation instance
        """
        self.simulation = simulation
    
    def _get_mission_state(self):
        """Get mission_state from simulation_config (V3.0.0: required)."""
        if not hasattr(self.simulation, 'simulation_config') or not self.simulation.simulation_config:
            raise ValueError("simulation_config is required (V3.0.0: no SatelliteConfig fallback)")
        return self.simulation.simulation_config.mission_state
    
    def _get_app_config(self):
        """Get app_config from simulation_config (V3.0.0: required)."""
        if not hasattr(self.simulation, 'simulation_config') or not self.simulation.simulation_config:
            raise ValueError("simulation_config is required (V3.0.0: no SatelliteConfig fallback)")
        return self.simulation.simulation_config.app_config
    
    def _is_waypoint_mode(self) -> bool:
        """Check if waypoint mode is enabled."""
        mission_state = self._get_mission_state()
        return mission_state.enable_waypoint_mode or mission_state.enable_multi_point_mode
    
    def _is_dxf_mode(self) -> bool:
        """Check if DXF shape mode is enabled."""
        mission_state = self._get_mission_state()
        return mission_state.dxf_shape_mode_active
    
    def _get_current_target_index(self) -> int:
        """Get current target index."""
        mission_state = self._get_mission_state()
        return mission_state.current_target_index
    
    def _get_waypoint_targets(self):
        """Get waypoint targets list."""
        mission_state = self._get_mission_state()
        return mission_state.waypoint_targets or mission_state.multi_point_targets

    def run(
        self,
        show_animation: bool = True,
        structured_config: Any = None,
    ) -> Optional[Path]:
        """
        Run the simulation loop.
        
        Args:
            show_animation: Whether to display animation during simulation
            structured_config: Structured config to use (for context manager)
        
        Returns:
            Path to data save directory, or None
        """
        if structured_config is not None:
            with use_structured_config(structured_config.clone()):
                return self._run_with_globals(show_animation=show_animation)
        else:
            return self._run_with_globals(show_animation=show_animation)

    def _run_with_globals(self, show_animation: bool = True) -> Optional[Path]:
        """
        Run linearized MPC simulation.
        
        Args:
            show_animation: Whether to display animation during simulation
        """
        print("\nStarting Linearized MPC Simulation...")
        print("Press 'q' to quit early, Space to pause/resume")
        self.simulation.is_running = True

        # Clear any previous data from the logger
        self.simulation.data_logger.clear_logs()
        self.simulation.physics_logger.clear_logs()

        self.simulation.data_save_path = self.simulation.create_data_directories()
        if self.simulation.data_save_path:
            self.simulation.data_logger.set_save_path(self.simulation.data_save_path)
            self.simulation.physics_logger.set_save_path(self.simulation.data_save_path)
            logger.info("Created data directory: %s", self.simulation.data_save_path)

        # Simulation Context
        from src.satellite_control.core.simulation_context import (
            SimulationContext,
        )

        if not hasattr(self.simulation, "context"):
            self.simulation.context = SimulationContext()
            self.simulation.context.dt = self.simulation.satellite.dt
            self.simulation.context.control_dt = self.simulation.control_update_interval

        # Initialize MPC Controller (Linearized Model)
        try:
            # When using MuJoCo viewer, skip matplotlib animation (MuJoCo
            # viewer updates itself)
            if show_animation and not self.simulation.use_mujoco_viewer:
                # Matplotlib animation mode (legacy)
                return self._run_matplotlib_animation()
            else:
                # Run with MuJoCo viewer or without animation
                return self._run_batch_mode()

        except KeyboardInterrupt:
            print("\n\nSimulation cancelled by user")
            self.simulation.is_running = False

            # Save data when interrupted
            if (
                self.simulation.data_save_path is not None
                and self.simulation.data_logger.get_log_count() > 0
            ):
                print("\nSaving simulation data...")
                self.simulation.save_csv_data()
                self.simulation.visualizer.sync_from_controller()
                self.simulation.save_mission_summary()
                print(f" Data saved to: {self.simulation.data_save_path}")

                # Try to generate visualizations if we have enough data
                if self.simulation.data_logger.get_log_count() > 10:
                    try:
                        print("\n Auto-generating visualizations...")
                        self.simulation.auto_generate_visualizations()
                        if hasattr(self.simulation.visualizer, "save_mujoco_video"):
                            self.simulation.visualizer.save_mujoco_video(
                                self.simulation.data_save_path
                            )
                        print(" All visualizations complete!")
                    except Exception as e:
                        logger.warning(
                            f"WARNING: Could not generate visualizations: {e}"
                        )

        finally:
            # Cleanup
            pass
        return self.simulation.data_save_path

    def _run_matplotlib_animation(self) -> Optional[Path]:
        """Run simulation with matplotlib animation."""
        fig = self.simulation.satellite.fig
        ani = FuncAnimation(
            fig,
            self.update_step,
            interval=int(self.simulation.satellite.dt * 1000),
            blit=False,
            repeat=False,
            cache_frame_data=False,
        )
        plt.show()  # Show the animation window live

        # After animation is complete, save files
        if self.simulation.data_save_path is not None:
            print("\nSaving simulation data...")
            self.simulation.save_csv_data()
            self.simulation.visualizer.sync_from_controller()
            self.simulation.save_mission_summary()
            self.simulation.save_animation_mp4(fig, ani)
            print(f" Data saved to: {self.simulation.data_save_path}")

            print("\n Auto-generating performance plots...")
            self.simulation.auto_generate_visualizations()
            if hasattr(self.simulation.visualizer, "save_mujoco_video"):
                self.simulation.visualizer.save_mujoco_video(
                    self.simulation.data_save_path
                )
            print(" All visualizations complete!")
        return self.simulation.data_save_path

    def _run_batch_mode(self) -> Optional[Path]:
        """Run simulation in batch mode (MuJoCo or headless)."""
        # Performance Optimization: Batch physics steps
        # Calculate how many physics steps fit in one control update
        steps_per_batch = int(
            self.simulation.control_update_interval / self.simulation.satellite.dt
        )
        if steps_per_batch < 1:
            steps_per_batch = 1

        batch_mode = steps_per_batch > 1
        logger.info(
            f"Running optimized simulation loop. "
            f"Batch: {steps_per_batch} (dt={self.simulation.satellite.dt:.4f}s)"
        )

        fast_batch_steps = steps_per_batch - 1

        while self.simulation.is_running:
            step_only = False
            if (
                self.simulation.use_mujoco_viewer
                and hasattr(self.simulation.satellite, "is_viewer_paused")
                and self.simulation.satellite.is_viewer_paused()
            ):
                if (
                    hasattr(self.simulation.satellite, "consume_viewer_step")
                    and self.simulation.satellite.consume_viewer_step()
                ):
                    step_only = True
                else:
                    if hasattr(self.simulation.satellite, "sync_viewer"):
                        self.simulation.satellite.sync_viewer()
                    time.sleep(0.01)
                    continue

            # Optimized Batch: Run physics steps without control logic
            # overhead
            if batch_mode and not step_only:
                for _ in range(fast_batch_steps):
                    # Inline logic for speed
                    self.simulation.process_command_queue()
                    self.simulation.satellite.update_physics(self.simulation.satellite.dt)
                    self.simulation.simulation_time = (
                        self.simulation.satellite.simulation_time
                    )
                    self.simulation.log_physics_step()

            # Full Update (run MPC check, Mission Check, Logging, 1
            # Physics Step)
            self.update_step(None)  # type: ignore[arg-type]

            if not self.simulation.is_running:
                break

        if self.simulation.data_save_path is not None:
            print("\nSaving simulation data...")
            self.simulation.save_csv_data()
            self.simulation.visualizer.sync_from_controller()
            self.simulation.save_mission_summary()
            print(f" CSV data saved to: {self.simulation.data_save_path}")

            # Auto-generate all visualizations
            print("\n Auto-generating visualizations...")
            self.simulation.auto_generate_visualizations()
            # Also generate MuJoCo 3D render if possible
            if hasattr(self.simulation.visualizer, "save_mujoco_video"):
                self.simulation.visualizer.save_mujoco_video(
                    self.simulation.data_save_path
                )
            print(" All visualizations complete!")
        return self.simulation.data_save_path

    def update_step(self, frame: Optional[int]) -> List[Any]:
        """
        Update simulation step (called by matplotlib animation or batch loop).
        
        Args:
            frame: Current frame number (None for batch mode)
        
        Returns:
            List of artists for matplotlib animation
        """
        if not self.simulation.is_running:
            return []

        # Update target state based on mission mode (unified with Real.py)
        current_state = self.simulation.get_current_state()
        self.simulation.update_target_state_for_mode(current_state)

        self.simulation.update_mpc_control()

        # Process command queue to apply delayed commands (sets
        # active_thrusters)
        self.simulation.process_command_queue()

        # Advance MuJoCo physics: keep time bases aligned for valve timing
        dt = self.simulation.satellite.dt
        self.simulation.satellite.simulation_time = self.simulation.simulation_time

        # Record physics step time
        physics_start = time.perf_counter()
        self.simulation.satellite.update_physics(dt)
        physics_time = time.perf_counter() - physics_start
        self.simulation.performance_monitor.record_physics_step(physics_time)
        self.simulation.performance_monitor.increment_step()
        self.simulation.simulation_time = self.simulation.satellite.simulation_time

        # Log High-Frequency Physics Data
        self.simulation.log_physics_step()

        # Check termination conditions
        if self._check_termination_conditions():
            return []

        # Redraw
        self.simulation.draw_simulation()
        self.simulation.update_mpc_info_panel()  # Use custom MPC info panel instead

        return []

    def _check_termination_conditions(self) -> bool:
        """
        Check if simulation should terminate.
        
        Returns:
            True if simulation should stop, False otherwise
        """
        from src.satellite_control.utils.orientation_utils import quat_angle_error

        if not self._is_dxf_mode():
            target_currently_reached = self.simulation.check_target_reached()

            if target_currently_reached:
                if self.simulation.target_reached_time is None:
                    # First time reaching target
                    self.simulation.target_reached_time = self.simulation.simulation_time
                    print(
                        f"\nTARGET REACHED! Time: {self.simulation.simulation_time:.1f}s"
                        " - MPC will maintain position"
                    )
                else:
                    # Update maintenance tracking
                    self.simulation.target_maintenance_time = (
                        self.simulation.simulation_time - self.simulation.target_reached_time
                    )
                    current_state = self.simulation.get_current_state()
                    pos_error = np.linalg.norm(
                        current_state[:3] - self.simulation.target_state[:3]
                    )
                    ang_error = quat_angle_error(
                        self.simulation.target_state[3:7], current_state[3:7]
                    )
                    self.simulation.maintenance_position_errors.append(float(pos_error))
                    self.simulation.maintenance_angle_errors.append(float(ang_error))

                if self._is_waypoint_mode():
                    if self._handle_waypoint_advancement():
                        return True
            else:
                if self.simulation.target_reached_time is not None:
                    self.simulation.times_lost_target += 1
                    t = self.simulation.simulation_time
                    print(
                        f"WARNING: Target lost at t={t:.1f}s"
                        " - MPC working to regain control"
                    )

        app_config = self._get_app_config()
        use_final_stab = app_config.simulation.use_final_stabilization
        
        if (
            not use_final_stab
            and self.simulation.target_reached_time is not None
            and not self._is_waypoint_mode()
            and not self._is_dxf_mode()
        ):
            # Mission 1: Waypoint Navigation (single waypoint) with immediate
            # termination after target reached
            current_maintenance_time = (
                self.simulation.simulation_time - self.simulation.target_reached_time
            )
            stab_time = app_config.simulation.waypoint_final_stabilization_time
            if current_maintenance_time >= stab_time:
                print(
                    f"\n WAYPOINT MISSION COMPLETE! "
                    f"Stable at target for {stab_time:.1f} seconds."
                )
                self.simulation.is_running = False
                self.simulation.print_performance_summary()
                return True

        if use_final_stab and self.simulation.target_reached_time is not None:
            if self._check_final_stabilization():
                return True

        # Only stop simulation when max time is reached
        if self.simulation.simulation_time >= self.simulation.max_simulation_time:
            print(f"\nSIMULATION COMPLETE at {self.simulation.simulation_time:.1f}s")
            self.simulation.is_running = False
            self.simulation.print_performance_summary()
            return True

        return False

    def _handle_waypoint_advancement(self) -> bool:
        """
        Handle waypoint advancement logic.
        
        Returns:
            True if simulation should stop, False otherwise
        """
        from src.satellite_control.utils.orientation_utils import (
            euler_xyz_to_quat_wxyz,
        )

        stabilization_time = self.simulation.target_maintenance_time

        mission_state = self._get_mission_state()
        app_config = self._get_app_config()
        targets = self._get_waypoint_targets()
        current_idx = self._get_current_target_index()
        
        is_final_target = current_idx >= len(targets) - 1
        # V3.0.0: Use timing from SimulationParams
        if is_final_target:
            required_hold_time = app_config.simulation.waypoint_final_stabilization_time
        else:
            required_hold_time = app_config.simulation.target_hold_time

        if stabilization_time >= required_hold_time:
            # Advance to next target (V3.0.0: always use mission_manager)
            if not self.simulation.mission_manager:
                raise ValueError("mission_manager is required (V3.0.0: no SatelliteConfig fallback)")
            
            next_available = self.simulation.mission_manager._advance_to_next_target()
            
            if next_available:
                # Update target state to next target with obstacle avoidance
                target_pos, target_angle = self.simulation.mission_manager._get_current_waypoint_target()
                if target_pos is not None:
                    roll_deg, pitch_deg, yaw_deg = np.degrees(target_angle)
                    logger.info(
                        f"MOVING TO NEXT TARGET: "
                        f"({target_pos[0]:.2f}, "
                        f"{target_pos[1]:.2f}) m, "
                        f"roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°, yaw={yaw_deg:.1f}°"
                    )
                    self.simulation.target_state = np.zeros(13, dtype=float)
                    self.simulation.target_state[0:3] = target_pos
                    self.simulation.target_state[3:7] = euler_xyz_to_quat_wxyz(
                        target_angle
                    )
                    self.simulation.target_reached_time = None
                    self.simulation.approach_phase_start_time = (
                        self.simulation.simulation_time
                    )
                    self.simulation.target_maintenance_time = 0.0
            else:
                # All targets completed - end simulation
                logger.info("WAYPOINT MISSION COMPLETED! " "All targets visited.")
                app_config = self._get_app_config()
                use_stab = app_config.simulation.use_final_stabilization
                if not use_stab:
                    self.simulation.is_running = False
                    self.simulation.print_performance_summary()
                    return True
                # Update phase in mission_state (V3.0.0: always available)
                mission_state = self._get_mission_state()
                mission_state.multi_point_phase = "COMPLETE"
        return False

    def _check_final_stabilization(self) -> bool:
        """
        Check final stabilization conditions.
        
        Returns:
            True if simulation should stop, False otherwise
        """
        current_maintenance_time = (
            self.simulation.simulation_time - self.simulation.target_reached_time
        )

        # Waypoint navigation: check completion for single or multiple waypoints
        mission_state = self._get_mission_state()
        app_config = self._get_app_config()
        # V3.0.0: Use timing from SimulationParams
        stab_time = app_config.simulation.waypoint_final_stabilization_time
        
        if not self._is_dxf_mode():
            # Single waypoint (no ENABLE_WAYPOINT_MODE set)
            if not self._is_waypoint_mode():
                if current_maintenance_time >= stab_time:
                    print(
                        f"\n WAYPOINT MISSION COMPLETE! "
                        f"Stable at target for {stab_time:.1f} seconds."
                    )
                    self.simulation.is_running = False
                    self.simulation.print_performance_summary()
                    return True
            # Multiple waypoints (ENABLE_WAYPOINT_MODE = True)
            elif mission_state.multi_point_phase == "COMPLETE":
                if current_maintenance_time >= stab_time:
                    print(
                        f"\n WAYPOINT MISSION COMPLETE! "
                        f"All targets stable for {stab_time:.1f} seconds."
                    )
                    self.simulation.is_running = False
                    self.simulation.print_performance_summary()
                    return True
        return False
