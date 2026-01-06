"""
Backward Compatibility Module for Satellite Model

This module re-exports physical parameters from config.physics for backward
compatibility with test files that import from core.model.

The original model.py was refactored and its contents moved to the modular
config system.
"""

from src.satellite_control.config import physics, timing

# Re-export timing parameters
DT = timing.SIMULATION_DT
CONTROL_DT = timing.CONTROL_DT

# Re-export physical parameters
SATELLITE_SIZE = physics.SATELLITE_SIZE
TOTAL_MASS = physics.TOTAL_MASS
MOMENT_OF_INERTIA = physics.MOMENT_OF_INERTIA

# Re-export thruster configuration
# Note: Convert numpy arrays to tuples for backward compatibility
thruster_positions = physics.THRUSTER_POSITIONS

# Convert direction arrays to tuples for tests expecting tuples
thruster_directions = {k: tuple(v) for k, v in physics.THRUSTER_DIRECTIONS.items()}

thruster_forces = physics.THRUSTER_FORCES
