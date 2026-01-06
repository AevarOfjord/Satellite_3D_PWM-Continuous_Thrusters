# Physics Engine: MuJoCo

## Overview

This project uses **MuJoCo (Multi-Joint dynamics with Contact)** as its high-fidelity physics engine. MuJoCo provides industry-standard accuracy for rigid body dynamics, enabling this simulation to closely match real-world physical behavior including contacts, friction, and constrained motion.

---

## Architecture & Data Flow

The system uses a clean separation of concerns where the MPC controller operates in a discrete time domain (control ticks), while MuJoCo integrates the continuous physics in smaller substeps.

### Data Flow Diagram

```text
┌─────────────────────────────────────────────────────────────┐
│                         MPC Controller                       │
│  Solves optimization → outputs PWM commands [0.0, 0.5, 1.0]  │
└─────────────────────┬───────────────────────────────────────┘
                      │ PWM thruster commands (0.0 - 1.0)
                      ↓
┌─────────────────────────────────────────────────────────────┐
│              MuJoCo Wrapper                                 │
│                                                             │
│  Job: Transform MPC commands → Force vectors                │
│  ────────────────────────────────────────────────────────   │
│  1. Get thruster config (positions, force magnitudes)       │
│  2. Apply valve delays & ramp-up (realistic dynamics)       │
│  3. Transform body frame → world frame                      │
│  4. Calculate torques about COM                             │
│  5. Pass to MuJoCo via xfrc_applied                         │
└─────────────────────┬───────────────────────────────────────┘
                      │ Force vectors [Fx, Fy, τz]
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                    MuJoCo Physics Engine                     │
│                                                             │
│  DOES ALL THE PHYSICS:                                      │
│  ─────────────────────                                      │
│  1. Integrate forces → accelerations (F=ma)                 │
│  2. Integrate accelerations → velocities (RK4/Euler)        │
│  3. Integrate velocities → positions                        │
│  4. Enforce constraints (2D plane, contacts)                │
│  5. Add damping and friction                                │
└─────────────────────┬───────────────────────────────────────┘
                      │ New state [x, y, θ, vx, vy, ω]
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                         MPC Controller                       │
│  Uses new state to solve next optimization                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Physics Features

### 1. Integration Method

The physics engine uses advanced numerical integration to ensure stability and energy conservation:

- **Integrator**: Configurable (Euler, RK4, or Implicit Fast).
- **Time Step**: Runs at **200 Hz** (5ms step) to capture high-frequency dynamics.
- **Sub-stepping**: The physics loop runs faster than the control loop (16.67 Hz) to simulate continuous dynamics between control updates.

### 2. Force Application

Thruster forces and torques are applied directly to the rigid body using `mj_data.xfrc_applied`.

```python
# Applying forces in MuJoCo
self.data.xfrc_applied[body_id, :] = [fx, fy, fz, tx, ty, tz]
```

- **Constraint Handling**: MuJoCo automatically solves for constraint forces (e.g., keeping the satellite on the air-bearing table plane 3DOF).
- **Friction**: Uses a physical friction model for any contacts.

### 3. Numerical Accuracy

MuJoCo is chosen for its superior numerical stability compared to simple custom integrators.

- **Collision**: Accurate computation of contact forces and impulses, essential for multi-agent or obstacle-rich environments.
- **Stability**: Robust constraint solver prevents non-physical penetration or drift.

### 4. 3D Capability

While the current mission profile focuses on planar (3-DOF) movement to simulate an air-bearing table, the underlying MuJoCo engine is fully 3D. This allows:

- **Complex Geometries**: Support for arbitrary convex meshes for collision detection.

---

## Simulation vs. Real-World Parity

The simulation is designed to be a "Digital Twin" of a physical satellite testbed.

| Feature               | Simulation Implementation                                        |
| --------------------- | ---------------------------------------------------------------- |
| **Mass**              | 10.0 kg (configured in `config/physics.py`)                      |
| **Inertia**           | 0.140 kg·m² (1/6 × m × L²)                                       |
| **Thruster Geometry** | 8 thrusters at precision mounting coordinates                    |
| **Actuators**         | Optional valve delays and force ramp-up (disabled by default)    |
| **Sensors**           | Optional Gaussian noise injection (disabled by default)          |
| **Damping**           | Configurable linear and rotational damping (disabled by default) |

By default, the simulation runs in **idealized mode** (no delays, noise, or damping) for clean MPC testing. Realistic physics can be enabled via `config/physics.py` for robustness validation.

---

## Configuration

The physics simulation is configured via XML (MJCF) and Python config files.

- **Model Files**: `models/satellite.xml` and `models/satellite_planar.xml` (define the body, geometry, and mass properties)
- **Physics Parameters**: `src/satellite_control/config/physics.py` (mass, inertia, thruster config, damping coefficients)
- **MPC Parameters**: `src/satellite_control/config/mpc_params.py` (cost weights, horizons, constraints)
- **Timing**: `src/satellite_control/config/timing.py` (control dt, physics dt)

---

## Why MuJoCo?

1.  **Speed**: Highly optimized C-based engine, allowing faster-than-real-time training and testing.
2.  **Contact Dynamics**: Best-in-class contact solver (convex optimization).
3.  **Industry Standard**: Widely adopted in robotics and RL (DeepMind, OpenAI).
