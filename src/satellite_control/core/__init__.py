"""
Core Module

Core simulation and control components.

Public API:
- ThrusterManager: Valve delays, ramp-up, PWM logic
- MPCRunner: MPC control loop execution
- SatelliteMPCLinearizedSimulation: Main simulation class
- model: Backward compatibility module for physical parameters
"""

from src.satellite_control.core import model
from src.satellite_control.core.mpc_runner import MPCRunner
from src.satellite_control.core.thruster_manager import ThrusterManager

__all__ = [
    "ThrusterManager",
    "MPCRunner",
    "model",
]
