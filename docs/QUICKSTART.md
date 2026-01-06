# Quick Start Guide

Get your first satellite control simulation running in 5 minutes.

---

## Prerequisites

- **Python 3.9+** (check: `python --version`)
- **20 minutes** for full installation
- **macOS, Linux, or Windows**

---

## Installation

### 1. Clone and Navigate

```bash
git clone https://github.com/AevarOfjord/Satellite_2D_MuJoCo.git
cd Satellite_3D_MuJoCo
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** This installs MuJoCo physics engine, OSQP solver, and visualization tools (~150MB).

### 4. Install FFmpeg (for animations)

**macOS:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

---

## Run Your First Simulation

### Launch Interactive Menu

```bash
python run_simulation.py
```

### Try the "Simple Translation" Preset

1. Select: **Waypoint Navigation**
2. Choose preset: **Simple Translation**
3. Press Enter to start

**What you'll see:**

- Real-time terminal dashboard with telemetry
- Mission completes in ~15 seconds
- Results saved to `Data/` folder

### View Results

```bash
# Find your simulation output
ls -lt Data/ | head -2

# Navigate to latest run
cd Data/<timestamp>

# View animation
open Simulation_animation.mp4  # macOS
# or: xdg-open Simulation_animation.mp4  # Linux
```

---

## Understanding the Output

Every simulation creates:

| File                       | Purpose                                    |
| -------------------------- | ------------------------------------------ |
| `physics_data.csv`         | Position, velocity, acceleration over time |
| `control_data.csv`         | Thruster commands and MPC performance      |
| `Simulation_animation.mp4` | Animated mission playback                  |

**Example analysis:**

```python
import pandas as pd
df = pd.read_csv('Data/<timestamp>/physics_data.csv')
print(f"Final position error: {df['error'].iloc[-1]:.4f} m")
```

---

## Next Steps

### Try Different Missions

**Waypoint Navigation:**

- Multi-waypoint tour
- Diagonal movement
- Rotation test

**Shape Following:**

- Circle
- Rectangle
- Star pattern
- Custom DXF shapes

### Customize Parameters

Edit MPC weights in `src/satellite_control/config/mpc_params.py`:

```python
Q_POSITION = 1000.0     # Higher = faster approach
Q_VELOCITY = 10000.0    # Higher = smoother motion
```

### Run Tests

```bash
# Quick verification
python run_simulation.py verify

# Full test suite
pytest tests/
```

---

## Troubleshooting

### "ModuleNotFoundError"

```bash
# Ensure you're in project root and venv is activated
pwd  # Should show .../Satellite_3D_MuJoCo
which python  # Should show .../venv/bin/python
```

### "ffmpeg not found"

```bash
# Verify installation
ffmpeg -version

# Run without animation if needed
python run_simulation.py --no-anim
```

### Slow Performance

```bash
# Run headless for max speed
python run_simulation.py --no-anim --auto
```

**More help:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## What Just Happened?

1. **MPC Controller** computed optimal thruster commands every 60ms
2. **MuJoCo Physics** simulated satellite dynamics at 200Hz
3. **8 Thrusters** fired based on continuous PWM duty cycles
4. **Data Logger** recorded full mission telemetry
5. **Visualizer** created animation and plots

**Key Achievement:** Your satellite navigated to the target with <5cm accuracy using Model Predictive Control!

---

## Learn More

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design
- **[MATHEMATICS.md](MATHEMATICS.md)** - MPC formulation
- **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** - Contributing
- **[VISUALIZATION.md](VISUALIZATION.md)** - Understanding plots

---

**Ready for more?** Try the shape following missions or customize the control parameters!
