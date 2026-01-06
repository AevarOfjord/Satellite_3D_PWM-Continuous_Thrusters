# Frequently Asked Questions (FAQ)

Common questions about the Satellite Control System.

---

## General Questions

### What is this project?

A high-fidelity satellite control simulation using Model Predictive Control (MPC) with MuJoCo physics. It demonstrates real-time optimal control of a planar satellite with 8 thrusters.

**Key Features:**

- OSQP-based MPC solver (~1-2ms solve times)
- MuJoCo physics engine (200 Hz integration)
- Waypoint navigation and shape following missions
- Real-time visualization and comprehensive telemetry

### Is this a real satellite or simulation?

**Simulation**, but designed to match real hardware. The system uses the same control algorithms and telemetry format as physical satellite testbeds.

### What hardware do I need?

**Software only!** Any modern computer works:

- CPU: Any 2015+ processor
- RAM: 4GB minimum, 8GB recommended
- Storage: 500MB for dependencies
- OS: macOS, Linux, or Windows

---

## Installation & Setup

### What Python version do I need?

**Python 3.9, 3.10, 3.11, or 3.12**

Check yours:

```bash
python --version
```

### Do I need to install MuJoCo separately?

**No!** MuJoCo is included in the pip package. Just run:

```bash
pip install -r requirements.txt
```

### Why does installation take so long?

The system installs several large packages:

- MuJoCo physics engine (~100MB)
- NumPy, SciPy (scientific computing)
- Matplotlib (visualization)
- OSQP (optimization solver)

**Typical time:** 5-10 minutes on good internet connection.

### Can I skip FFmpeg?

**Yes**, but you won't get animation videos. CSV data and plots still work.

To run without animations:

```bash
python run_simulation.py --no-anim
```

---

## Running Simulations

### How do I run my first simulation?

```bash
python run_simulation.py
```

Select "Waypoint Navigation" → "Simple Translation" preset.

See [QUICKSTART.md](QUICKSTART.md) for full walkthrough.

### What's the difference between mission types?

| Mission Type            | Description                      | Use Case                    |
| ----------------------- | -------------------------------- | --------------------------- |
| **Waypoint Navigation** | Move to fixed points in sequence | Point-to-point maneuvering  |
| **Shape Following**     | Track moving target along path   | Inspection patterns, orbits |

### How long do simulations take?

**Depends on mission:**

- Simple waypoint: 10-30 seconds
- Multi-waypoint: 30-60 seconds
- Shape following: 30-90 seconds

**Real-time factor:** Usually 2-5× faster than real-time (headless mode).

### Where are my results saved?

```
Data/
└── YYYY-MM-DD_HH-MM-SS/
    ├── physics_data.csv
    ├── control_data.csv
    └── Simulation_animation.mp4
```

Latest run is the most recent timestamp.

---

## Performance & Tuning

### What does "MPC solve time" mean?

Time for the optimizer to compute thruster commands. Target: **<5ms**

Check yours:

```bash
grep "mpc_solve_time" Data/<timestamp>/control_data.csv
```

If >10ms consistently, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#mpc-solver-too-slow).

### How accurate is the control?

**Typical performance:**

- Position error: <5cm at target
- Angle error: <3° at target
- Settling time: 10-30s for 1m moves

### My satellite oscillates around the target. How do I fix it?

Increase velocity damping in `src/satellite_control/config/mpc_params.py`:

```python
Q_VELOCITY = 15000.0  # Increase from 10000
Q_ANGULAR_VELOCITY = 2000.0  # Increase from 1500
```

### My satellite moves too slowly. How do I speed it up?

```python
# In mpc_params.py
Q_POSITION = 1500.0    # Increase from 1000 (more aggressive)
R_THRUST = 0.5         # Decrease from 1.0 (allow more thrust)
MAX_VELOCITY = 0.6     # Increase from 0.5 m/s
```

### What's the control loop frequency?

- **Control (MPC):** 16.67 Hz (60ms period)
- **Physics:** 200 Hz (5ms timestep)

Physics runs 12× faster to capture inter-sample dynamics.

---

## Understanding Output

### What's in the CSV files?

**physics_data.csv** - State history (200 Hz):

- Position, velocity, orientation
- Targets and errors
- Mission phase

**control_data.csv** - Control history (16.67 Hz):

- Thruster PWM duty cycles (0-1)
- MPC solve time and status
- Cost function values

See [VISUALIZATION.md - CSV Format Appendix](VISUALIZATION.md#appendix-csv-data-format-reference) for complete column reference.

### How do I plot the trajectory?

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/<timestamp>/physics_data.csv')
plt.plot(df['x'], df['y'])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True)
plt.show()
```

Or use the built-in visualizer:

```bash
python -m src.satellite_control.visualization.unified_visualizer
```

### What does "OPTIMAL" vs "SUBOPTIMAL" mean?

**MPC solver status:**

- `OPTIMAL`: Best solution found (good!)
- `SUBOPTIMAL`: Acceptable solution within tolerances
- `TIME_LIMIT`: Solver hit time limit (still usable)
- `FAILED`: Solver failed (control falls back to proportional)

Occasional SUBOPTIMAL is fine. Frequent FAILED needs parameter tuning.

---

## Technical Questions

### What solver does MPC use?

**OSQP** (Operator Splitting Quadratic Program) - convex QP solver

**Why OSQP?**

- Extremely fast (~1ms for this problem size)
- Warm-starting support
- Reliable convergence
- Open source

### Is this MPC continuous or discrete-time?

**Discrete-time** with 60ms control intervals. The continuous dynamics are discretized using Euler integration for the MPC prediction model.

### How many decision variables in the optimization?

**Per MPC solve:**

- States: 6 × 50 steps = 300 variables
- Controls: 8 × 50 steps = 400 variables
- **Total: ~700 variables**

Still fast due to sparse problem structure.

### What's the prediction horizon?

**Default: 50 steps = 3.0 seconds**

Configurable in `mpc_params.py`:

```python
MPC_PREDICTION_HORIZON = 50
```

### Does it model disturbances?

**Optional.** By default runs idealized physics. Enable realistic effects in `config/physics.py`:

```python
use_realistic_physics = True
linear_damping_coeff = 1.8  # Air drag
enable_random_disturbances = True
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

You're not in the project root directory.

```bash
cd /path/to/Satellite_3D_MuJoCo
python run_simulation.py
```

### Animation file is 0 bytes or won't play

FFmpeg issue. Check installation:

```bash
ffmpeg -version
```

Regenerate:

```bash
python -m src.satellite_control.visualization.unified_visualizer
```

### "Simulation unstable" warnings

Physics timestep may be too large or forces too high. Usually resolves itself; check final results in CSV.

If persistent, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#simulation-unstable-or-oscillating).

### Tests fail with timeout

Run only fast tests:

```bash
pytest -m "not slow" tests/
```

Or increase timeout:

```bash
pytest --timeout=30 tests/
```

---

## Development

### Can I add new mission types?

**Yes!** See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md#adding-new-features) for step-by-step instructions.

### How do I change thruster configuration?

Edit `src/satellite_control/config/thruster_config.py`:

- Positions (body frame)
- Directions (unit vectors)
- Force magnitudes (Newtons)

### Can I use a different controller?

Yes, implement the `MotionController` protocol from `core/interfaces.py`. Current MPC can be swapped out while keeping the simulation infrastructure.

### Where's the mathematical formulation?

See [MATHEMATICS.md](MATHEMATICS.md) for complete MPC formulation, dynamics equations, and linearization details.

---

## Comparison to Real Hardware

### How does this compare to a real satellite testbed?

**Similarities:**

- Same control algorithms (MPC with OSQP)
- Same telemetry format (CSV structure)
- Same timing constraints (60ms control loop)
- Realistic thruster geometry

**Differences:**

- No sensor noise (unless enabled)
- No valve delays (unless enabled)
- Perfect state knowledge
- No communication latency

**Purpose:** Algorithm development and testing before hardware deployment.

---

## Getting Help

**Can't find your question?**

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)
3. Search existing GitHub issues
4. Open a new issue with:
   - Python version
   - Error message
   - Steps to reproduce

---

**Still stuck?** Check the documentation:

- [QUICKSTART.md](QUICKSTART.md) - Installation walkthrough
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [VISUALIZATION.md](VISUALIZATION.md) - Understanding output
