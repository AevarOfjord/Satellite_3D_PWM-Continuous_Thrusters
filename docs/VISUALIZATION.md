# Simulation Visualization Guide

## Overview

After each simulation run, the system automatically generates comprehensive visualizations to help you analyze satellite performance, control behavior, and mission success. All outputs are saved to timestamped directories in the `Data/` folder.

---

## Quick Start

### Running a Simulation

```bash
python run_simulation.py
```

After the simulation completes, find your results:

```bash
ls Data/  # Lists timestamped directories
cd Data/2026-01-06_10-30-45  # Navigate to latest run
```

### What Gets Generated

Every simulation automatically creates:

1. **CSV Data Files** - Raw telemetry data
2. **Trajectory Animation (MP4)** - Animated playback of the mission
3. **Performance Plots (PNG)** - Static analysis charts

---

## Output Directory Structure

After running a simulation, you'll find:

```
Data/
└── 2026-01-06_10-30-45/           # Timestamp: YYYY-MM-DD_HH-MM-SS
    ├── physics_data.csv           # Position, velocity, acceleration
    ├── control_data.csv          # Thruster commands, MPC performance
    ├── Simulation_animation.mp4  # Animated mission playback
    └── (plots generated on demand)
```

---

## 1. CSV Data Files

### physics_data.csv

**Contains:** Complete state history of the satellite

**Columns:**

- `time` - Simulation time (seconds)
- `x`, `y` - Position in world frame (meters)
- `vx`, `vy` - Velocity in world frame (m/s)
- `theta` - Orientation angle (radians)
- `omega` - Angular velocity (rad/s)
- `target_x`, `target_y` - Current target position
- `target_theta` - Target orientation
- `mission_phase` - Current mission state (APPROACHING, STABILIZING, etc.)

**Example Row:**

```csv
time,x,y,vx,vy,theta,omega,target_x,target_y,target_theta,mission_phase
12.34,0.452,-0.231,0.082,-0.034,0.125,-0.012,1.0,0.0,0.0,APPROACHING
```

**Use Case:** Import into your own analysis tools, MATLAB, Python pandas, etc.

```python
import pandas as pd
df = pd.read_csv('Data/2026-01-06_10-30-45/physics_data.csv')
print(df.describe())
```

---

### control_data.csv

**Contains:** Control inputs and MPC solver performance

**Columns:**

- `time` - Simulation time (seconds)
- `thruster_1` through `thruster_8` - Duty cycle [0-1] for each thruster
- `mpc_solve_time` - Optimization solver time (seconds)
- `mpc_cost` - Objective function value
- `mpc_status` - Solver status (OPTIMAL, SUBOPTIMAL, FAILED)
- `num_active_thrusters` - Count of thrusters currently firing

**Example Row:**

```csv
time,thruster_1,thruster_2,...,thruster_8,mpc_solve_time,mpc_cost,mpc_status,num_active_thrusters
12.34,0.0,0.0,0.0,0.0,0.641,0.0,0.823,0.0,0.00142,245.3,OPTIMAL,2
```

**Use Case:** Analyze control effort, MPC performance, thruster usage patterns

```python
import pandas as pd
df = pd.read_csv('Data/2026-01-06_10-30-45/control_data.csv')

# Average solve time
print(f"Avg MPC solve time: {df['mpc_solve_time'].mean()*1000:.2f} ms")

# Total control effort
total_effort = df[[f'thruster_{i}' for i in range(1,9)]].sum().sum()
print(f"Total thrust effort: {total_effort:.2f}")
```

---

## 2. Animated Mission Playback (MP4)

### Simulation_animation.mp4

**What it shows:**

- Full mission trajectory in X-Y plane
- Satellite position and orientation over time
- Active thrusters highlighted when firing
- Target/waypoint positions
- Real-time telemetry overlay:
  - Current position and velocity
  - Distance to target
  - Mission phase
  - Active thruster list

**Playback Controls:**

- Standard video player controls
- Scrub timeline to specific moments
- Pause to inspect details
- Speed up/slow down in most players

**Technical Details:**

- **Resolution:** 1920×1080 (Full HD)
- **Frame Rate:** 30 FPS
- **Duration:** Matches real simulation time (e.g., 30s mission = 30s video)
- **Codec:** H.264 (widely compatible)

**Viewing:**

```bash
# macOS
open Data/2026-01-06_10-30-45/Simulation_animation.mp4

# Linux
xdg-open Data/2026-01-06_10-30-45/Simulation_animation.mp4

# Windows
start Data/2026-01-06_10-30-45/Simulation_animation.mp4
```

---

## 3. Performance Analysis Plots

Generate detailed static plots on demand:

```bash
python -m src.satellite_control.visualization.unified_visualizer
```

This creates multiple PNG plots in the same `Data/<timestamp>/` directory.

---

### 3.1 Trajectory Plot

**Filename:** `trajectory_plot.png`

**Shows:**

- Complete X-Y path of satellite (blue line)
- Starting position (green marker)
- Waypoints/targets (red markers)
- Trajectory highlights where thrusters fired
- Workspace boundaries (±3m)

**What to Look For:**

- ✓ Smooth path to target
- ✓ Minimal oscillations
- ✓ Respects workspace bounds
- ✗ Overshooting targets
- ✗ Erratic path segments

**Example Interpretation:**

```
┌─────────────────────────────────────┐
│                                     │
│     Start (0,0) ●────────→ ● Target│
│                  \                  │
│                   \ (slight curve)  │
│                    \                │
│                     ● Waypoint 1    │
└─────────────────────────────────────┘

Good: Smooth path with gentle curve
```

---

### 3.2 Position vs Time

**Filename:** `position_vs_time.png`

**Shows:**

- X position over time (top subplot)
- Y position over time (bottom subplot)
- Target positions (dashed reference lines)
- Position error shaded region

**What to Look For:**

- ✓ Converges to target positions
- ✓ Minimal overshoot
- ✓ Smooth approach (no sharp jumps)
- ✗ Oscillations around target
- ✗ Never reaching target

**Key Metrics:**

- **Settling time:** How long to reach and stay at target
- **Overshoot:** Maximum distance past target
- **Steady-state error:** Final position error at mission end

---

### 3.3 Velocity Profile

**Filename:** `velocity_profile.png`

**Shows:**

- Linear velocity magnitude over time (top)
- Angular velocity over time (bottom)
- Velocity limits shown as reference lines

**What to Look For:**

- ✓ Velocity approaches zero near target
- ✓ Stays within limits (0.5 m/s linear, π/2 rad/s angular)
- ✓ Smooth acceleration/deceleration
- ✗ Sustained high velocity near target
- ✗ Velocity limit violations

**Performance Indicators:**

- **Aggressive control:** High velocities, fast settling
- **Conservative control:** Low velocities, slow but stable
- **Good damping:** Velocity smoothly decreases to zero

---

### 3.4 Thruster Activity Heatmap

**Filename:** `thruster_activity.png`

**Shows:**

- 8 rows (one per thruster)
- Time on X-axis
- Color intensity = duty cycle (0% = dark, 100% = bright)

**What to Look For:**

- ✓ Balanced thruster usage (no single thruster overused)
- ✓ Clear firing patterns matching mission phases
- ✓ Symmetric pairs firing for rotation
- ✗ Continuous high duty cycles (inefficient)
- ✗ Chattering (rapid on/off switching)

**Interpretation:**

```
Thruster 1: ████░░░░░░░░░░  (fired early, then idle)
Thruster 2: ░░░░████████░░  (fired mid-mission)
Thruster 3: ░░░░░░░░████░░  (fired for final correction)
...
```

---

### 3.5 MPC Performance

**Filename:** `mpc_performance.png`

**Shows:**

- Solve time per MPC iteration (milliseconds)
- Cost function value over time
- Solver status (success/failure markers)

**What to Look For:**

- ✓ Solve times consistently < 10 ms
- ✓ Cost function decreasing over time
- ✓ All iterations marked as "OPTIMAL"
- ✗ Solve times approaching control period (60 ms)
- ✗ Frequent solver failures

**Performance Targets:**

- **Excellent:** < 2 ms average solve time
- **Good:** 2-5 ms average
- **Acceptable:** 5-10 ms average
- **Poor:** > 10 ms (may indicate problem too large or poorly conditioned)

---

### 3.6 Error Analysis

**Filename:** `error_analysis.png`

**Shows:**

- Position error magnitude over time (meters)
- Angle error over time (degrees)
- Error convergence rate

**What to Look For:**

- ✓ Error decreasing monotonically
- ✓ Final error < tolerance (typically 0.05 m, 3°)
- ✓ Exponential decay shape
- ✗ Error increasing over time
- ✗ Oscillating error (overshooting)

**Success Criteria:**

- **Position error:** < 0.05 m at mission end
- **Angle error:** < 3° at mission end
- **Convergence time:** Depends on mission (typically 10-30s)

---

## Customizing Visualizations

### Changing Plot Appearance

Edit `src/satellite_control/visualization/unified_visualizer.py`:

```python
# Change plot style
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')  # or 'ggplot', 'dark_background'

# Adjust figure sizes
self.figsize = (12, 8)  # Width x Height in inches

# Customize colors
self.trajectory_color = '#2E86AB'  # Blue
self.target_color = '#A23B72'      # Purple
self.thruster_colormap = 'viridis' # or 'plasma', 'hot', 'coolwarm'
```

### Changing Animation Settings

```python
# In unified_visualizer.py

# Frame rate (higher = smoother but larger file)
self.fps = 30  # 60 for very smooth, 20 for smaller files

# Resolution
self.dpi = 150  # 100 for faster render, 200 for high quality

# Video codec
self.codec = 'h264'  # or 'mpeg4', 'libx264'

# Playback speed (1.0 = real-time, 2.0 = 2x speed)
self.playback_speed = 1.0
```

### Generating Only Specific Plots

```python
from src.satellite_control.visualization.unified_visualizer import UnifiedVisualizationGenerator

viz = UnifiedVisualizationGenerator(data_directory="Data")

# Generate only trajectory plot
viz.generate_trajectory_plot()

# Generate only animation
viz.generate_animation("output.mp4")

# Generate all plots except animation
viz.generate_plots()  # Skips MP4 generation
```

---

## Comparing Multiple Runs

To compare different missions or parameter configurations:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load two different runs
run1 = pd.read_csv('Data/2026-01-06_10-00-00/physics_data.csv')
run2 = pd.read_csv('Data/2026-01-06_10-30-00/physics_data.csv')

# Compare position error
plt.figure(figsize=(10, 6))
plt.plot(run1['time'], run1['error'], label='Run 1: N=50')
plt.plot(run2['time'], run2['error'], label='Run 2: N=100')
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.legend()
plt.grid(True)
plt.savefig('comparison.png')
```

---

## Exporting for Presentations

### High-Quality Still Images

```python
viz = UnifiedVisualizationGenerator(data_directory="Data")

# Export trajectory as high-res PNG
viz.generate_trajectory_plot()
# Saved to Data/<timestamp>/trajectory_plot.png

# Convert to PDF for LaTeX/presentations
import subprocess
subprocess.run([
    'convert',
    'trajectory_plot.png',
    'trajectory_plot.pdf'
])
```

### Embedding Animation in Slides

1. **PowerPoint:** Insert → Video → From File
2. **Google Slides:** Insert → Video → Upload
3. **LaTeX Beamer:** Use `\movie` command or convert to GIF

**Convert MP4 to GIF (smaller for web):**

```bash
ffmpeg -i Simulation_animation.mp4 -vf "fps=15,scale=800:-1" output.gif
```

---

## Troubleshooting

### No Plots Generated

**Check:**

1. Simulation completed successfully
2. CSV files exist in `Data/<timestamp>/`
3. Run the visualizer manually:
   ```bash
   python -m src.satellite_control.visualization.unified_visualizer
   ```

### Animation File Corrupted

**Symptoms:** Can't open MP4, 0 byte file

**Solutions:**

1. Check FFmpeg installation: `ffmpeg -version`
2. Re-run visualizer with verbose output
3. Check disk space

### Poor Video Quality

**Improve quality:**

```python
# In unified_visualizer.py
self.fps = 60           # Higher frame rate
self.dpi = 200          # Higher resolution
self.bitrate = 8000     # Higher bitrate (kbps)
```

### Plots Look Cluttered

**Simplify:**

```python
# Reduce data points plotted
df_downsampled = df[::10]  # Plot every 10th point

# Increase figure size
plt.figure(figsize=(14, 10))

# Use cleaner style
plt.style.use('seaborn-v0_8-white')
```

---

## Best Practices

### For Analysis

1. **Always check solve times first** - MPC failures indicate parameter issues
2. **Compare error plot to velocity** - Slow convergence may need higher Q_vel
3. **Review thruster activity** - Excessive chattering wastes fuel
4. **Watch the animation** - Catches issues not obvious in plots

### For Presentations

1. **Use trajectory + animation** - Best visual demo
2. **Include error analysis** - Shows quantitative performance
3. **Highlight key metrics** - Settling time, final error, solve time
4. **Compare to baselines** - Show improvement over previous attempts

### For Debugging

1. **Check mission phase transitions** in CSV data
2. **Look for MPC solver failures** in control_data
3. **Identify oscillation patterns** in position plot
4. **Verify thruster symmetry** in heatmap

---

## Summary

After each simulation, you get:

| Output                       | Purpose            | Key Insights                       |
| ---------------------------- | ------------------ | ---------------------------------- |
| **physics_data.csv**         | Raw state data     | Import to custom analysis tools    |
| **control_data.csv**         | Control & MPC data | Solver performance, thruster usage |
| **Simulation_animation.mp4** | Visual playback    | See actual mission execution       |
| **trajectory_plot.png**      | Path analysis      | Verify smooth navigation           |
| **velocity_profile.png**     | Speed analysis     | Check damping and limits           |
| **thruster_activity.png**    | Control effort     | Identify inefficiencies            |
| **mpc_performance.png**      | Solver metrics     | Detect computational issues        |
| **error_analysis.png**       | Accuracy metrics   | Quantify mission success           |

**Next Steps:**

- Run a simulation: `python run_simulation.py`
- Find your data: `ls Data/`
- Review the animation and plots
- Iterate on MPC parameters if needed
- See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for tuning tips

---

## Appendix: CSV Data Format Reference

Both simulation and real hardware produce identical CSV format for easy comparison.

### Output Files

Each mission run generates two CSV files:

1. **physics_data.csv** (or `simulation_data.csv` in legacy format)

   - Full control loop data with all state variables, MPC info, and commands
   - 45+ columns per row
   - **Identical format for both simulation and real hardware**

2. **control_data.csv** (or `simulation_terminal_log.csv` in legacy format)
   - Control commands and MPC performance metrics
   - Logged at control frequency (16.67 Hz)

### physics_data.csv Column Definitions

#### Timing & Control

| Column        | Unit    | Description                                           |
| ------------- | ------- | ----------------------------------------------------- |
| time          | seconds | Simulation time                                       |
| step          | -       | Control iteration number                              |
| mission_phase | -       | Current phase (APPROACHING/STABILIZING/TRACKING etc.) |

#### Current State

| Column | Unit    | Description               |
| ------ | ------- | ------------------------- |
| x      | meters  | X position in world frame |
| y      | meters  | Y position in world frame |
| theta  | radians | Yaw angle in world frame  |
| vx     | m/s     | Velocity in X direction   |
| vy     | m/s     | Velocity in Y direction   |
| omega  | rad/s   | Angular velocity          |

#### Target State

| Column       | Unit    | Description             |
| ------------ | ------- | ----------------------- |
| target_x     | meters  | Target X position       |
| target_y     | meters  | Target Y position       |
| target_theta | radians | Target yaw angle        |
| target_vx    | m/s     | Target velocity in X    |
| target_vy    | m/s     | Target velocity in Y    |
| target_omega | rad/s   | Target angular velocity |

#### Tracking Error

| Column      | Unit    | Description                           |
| ----------- | ------- | ------------------------------------- |
| error_x     | meters  | Position error in X                   |
| error_y     | meters  | Position error in Y                   |
| error_theta | radians | Yaw error (shortest angular distance) |
| error_vx    | m/s     | Velocity error in X                   |
| error_vy    | m/s     | Velocity error in Y                   |
| error_omega | rad/s   | Angular velocity error                |

### control_data.csv Column Definitions

#### Thruster Commands (8 columns)

| Column                   | Unit | Description                            |
| ------------------------ | ---- | -------------------------------------- |
| thruster_1 to thruster_8 | -    | PWM duty cycle [0-1] for each thruster |

#### MPC Performance

| Column               | Unit    | Description                                     |
| -------------------- | ------- | ----------------------------------------------- |
| mpc_solve_time       | seconds | Time spent in MPC solver                        |
| mpc_status           | -       | Solver status (OPTIMAL, SUBOPTIMAL, TIME_LIMIT) |
| mpc_cost             | -       | Objective function value                        |
| num_active_thrusters | -       | Count of thrusters currently firing             |

### Unit Consistency

**SI Units (Internal):**

- Position: meters
- Angle: radians
- Velocity: m/s
- Angular velocity: rad/s
- Time: seconds

**Telemetry Units (if using hardware):**

- Position: millimeters (matches OptiTrack output)
- Angle: degrees (matches OptiTrack output)

### Mission Phases

**Waypoint Navigation:**

1. **APPROACHING** - Moving toward waypoint target
2. **STABILIZING** - Holding at waypoint (configurable duration)

**Shape Following (without return):**

1. **POSITIONING** - Moving to closest point on path
2. **PATH_STABILIZATION** - Stabilizing before tracking begins
3. **TRACKING** - Following moving target along path
4. **STABILIZING** - Holding at final position

**Shape Following (with return):**

1. **POSITIONING** - Moving to closest point on path
2. **PATH_STABILIZATION** - Stabilizing at start waypoint
3. **TRACKING** - Following moving target
4. **PATH_STABILIZATION** - Stabilizing at final waypoint
5. **RETURNING** - Moving to return position
6. **STABILIZING** - Holding at return position

### Data Analysis Examples

**Load and analyze CSV data:**

```python
import pandas as pd
import numpy as np

# Load files
physics = pd.read_csv('Data/<timestamp>/physics_data.csv')
control = pd.read_csv('Data/<timestamp>/control_data.csv')

# Calculate position error over time
pos_error = np.sqrt(physics['error_x']**2 + physics['error_y']**2)
print(f"Mean error: {pos_error.mean():.4f} m")
print(f"Final error: {pos_error.iloc[-1]:.4f} m")

# MPC performance summary
print(f"Avg solve time: {control['mpc_solve_time'].mean()*1000:.2f} ms")
print(f"Max solve time: {control['mpc_solve_time'].max()*1000:.2f} ms")

# Solver success rate
optimal_count = (control['mpc_status'] == 'OPTIMAL').sum()
total_count = len(control)
print(f"Optimal solutions: {optimal_count}/{total_count} ({100*optimal_count/total_count:.1f}%)")
```

**Plot trajectory:**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.plot(physics['x'], physics['y'], 'b-', label='Actual path')
plt.plot(physics['target_x'], physics['target_y'], 'r--', label='Target path')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('trajectory.png')
```

### Comparison Notes

When comparing simulation to real hardware:

1. **Telemetry**: Simulation is noiseless, real has measurement noise
2. **MPC timing**: Faster in simulation (no hardware communication overhead)
3. **Command execution**: Instantaneous in simulation, valve delays in real hardware
4. **State estimation**: Perfect in simulation, filtered/estimated in real hardware
