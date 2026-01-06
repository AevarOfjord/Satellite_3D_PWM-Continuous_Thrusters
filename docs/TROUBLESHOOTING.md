# Troubleshooting Guide

Common issues and solutions for the Satellite Control System.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Control Performance Issues](#control-performance-issues)
- [Visualization Issues](#visualization-issues)
- [Testing Issues](#testing-issues)
- [Getting Help](#getting-help)

---

## Installation Issues

### Python Version Incompatibility

**Problem**: Package installation fails or incompatibility warnings

**Solution**:

```bash
# Check your Python version
python --version

# Recommended: Python 3.9, 3.10, 3.11, or 3.12
# Download from https://www.python.org/downloads/

# Using pyenv for version management
pyenv install 3.11.0
pyenv local 3.11.0

# Using conda
conda create -n satellite python=3.11
conda activate satellite
```

### Missing Dependencies

**Problem**: `ModuleNotFoundError` for mujoco, osqp, numpy, scipy, etc.

**Solution**:

```bash
# Reinstall all requirements
pip install --upgrade -r requirements.txt

# If specific package fails, install individually
pip install mujoco osqp numpy scipy matplotlib rich questionary
```

### MuJoCo Installation Issues

**Problem**: "Cannot find MuJoCo library" or import errors

**Solution**:

```bash
# MuJoCo is included in pip package (no manual install needed)
pip install mujoco

# Verify installation
python -c "import mujoco; print(mujoco.__version__)"
```

### FFmpeg Not Found

**Problem**: "ffmpeg not found" when generating animation videos

**macOS**:

```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian)**:

```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows**:

1. Download from: https://ffmpeg.org/download.html
2. Extract and add `bin` folder to system PATH

**Verify installation**:

```bash
ffmpeg -version
```

### Virtual Environment Issues

**Problem**: Dependencies installed but imports still fail

**Solution**:

```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify you're using venv Python
which python  # Should show venv/bin/python
```

---

## Runtime Errors

### MPC Solver Too Slow

**Problem**: "MPC solve time exceeded acceptable limits" or simulation lags

**Current defaults**: N=50, solve time ~1-2ms (target), <5ms (acceptable)

**Solutions**:

```python
# Option 1: Reduce prediction horizon
# src/satellite_control/config/mpc_params.py
MPC_PREDICTION_HORIZON = 30  # Reduce from 50
MPC_CONTROL_HORIZON = 30     # Keep equal to prediction

# Option 2: Relax solver tolerances
# In mpc_controller.py, adjust OSQP settings:
eps_abs = 1e-3  # Looser (was 1e-4)
eps_rel = 1e-3  # Looser (was 1e-4)

# Option 3: Increase time limit (last resort)
MPC_SOLVER_TIME_LIMIT = 0.08  # Increase from 0.05s
```

**Check solve times**:

```bash
# After simulation, check control_data.csv
python -c "
import pandas as pd
df = pd.read_csv('Data/<timestamp>/control_data.csv')
print(f'Avg: {df[\"mpc_solve_time\"].mean()*1000:.2f}ms')
print(f'Max: {df[\"mpc_solve_time\"].max()*1000:.2f}ms')
"
```

### Simulation Unstable or Oscillating

**Problem**: Satellite spinning, overshooting, or oscillating around target

**Diagnosis**:

```bash
# Check velocity cost weight
grep Q_VELOCITY src/satellite_control/config/mpc_params.py
```

**Solutions**:

```python
# Increase damping (velocity cost weights)
Q_VELOCITY = 15000.0          # Increase from 10000
Q_ANGULAR_VELOCITY = 2000.0   # Increase from 1500

# Reduce control aggressiveness
R_THRUST = 1.5  # Increase from 1.0 (penalize control more)

# Lower position weight for smoother approach
Q_POSITION = 500.0  # Decrease from 1000
```

### Cannot Reach Target

**Problem**: Satellite never converges to target, times out

**Possible causes**:

- Target outside workspace (±3.0m)
- Insufficient thrust relative to distance
- Velocity limits too restrictive

**Solutions**:

```python
# Verify target within bounds
assert abs(target_x) < 3.0 and abs(target_y) < 3.0

# Check velocity limit
# src/satellite_control/config/mpc_params.py
MAX_VELOCITY = 0.5  # m/s (current default)

# Relax convergence tolerances
POSITION_TOLERANCE = 0.08  # meters (was 0.05)
ANGLE_TOLERANCE = np.deg2rad(5)  # degrees (was 3)
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src.satellite_control'`

**Solutions**:

```bash
# Ensure you're in project root
cd /Users/aevar/Desktop/Satellite_3D_MuJoCo
pwd  # Should show project root

# Run from project root
python run_simulation.py

# If using pytest
pytest tests/

# If needed, add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Thruster Commands Invalid

**Problem**: "Thruster command out of range" errors

**Cause**: MPC outputs should be [0, 1] PWM duty cycles

**Solution**:

```python
# Verify MPC output is clipped
u_opt = np.clip(u_opt, 0.0, 1.0)

# Check control_data.csv for invalid values
import pandas as pd
df = pd.read_csv('Data/<timestamp>/control_data.csv')
for i in range(1, 9):
    col = f'thruster_{i}'
    print(f"{col}: min={df[col].min()}, max={df[col].max()}")
```

---

## Control Performance Issues

### Poor Tracking Accuracy

**Problem**: Large position/angle errors during mission

**Diagnosis**:

```bash
# Check final errors in physics_data.csv
tail Data/<timestamp>/physics_data.csv
```

**Solutions**:

```python
# Increase state tracking weights
Q_POSITION = 2000.0   # Increase from 1000
Q_ANGLE = 2000.0      # Increase from 1000

# Increase prediction horizon (more look-ahead)
MPC_PREDICTION_HORIZON = 60  # Increase from 50

# Ensure velocity damping is sufficient
Q_VELOCITY = 12000.0  # Was 10000
```

### Excessive Control Effort

**Problem**: Too many thruster firings, wasting fuel

**Diagnosis**:

```bash
# Check thruster activity in visualization
python -m src.satellite_control.visualization.unified_visualizer
# Look at thruster_activity.png heatmap
```

**Solutions**:

```python
# Increase control cost
R_THRUST = 2.0  # Increase from 1.0

# Relax position tolerance (don't try to be so precise)
POSITION_TOLERANCE = 0.08  # Increase from 0.05

# Add switching penalty (reduce chattering)
R_SWITCH = 0.1  # Currently 0.0 (disabled)
```

### Slow Convergence

**Problem**: Takes too long to reach target

**Solutions**:

```python
# Increase aggressiveness
Q_POSITION = 1500.0    # Increase from 1000
R_THRUST = 0.5         # Decrease from 1.0 (allow more thrust)

# Reduce velocity penalty (allow faster motion)
Q_VELOCITY = 8000.0    # Decrease from 10000

# Increase max velocity
MAX_VELOCITY = 0.6     # Increase from 0.5 m/s
```

---

## Visualization Issues

### Animation Not Generated

**Problem**: No `Simulation_animation.mp4` file after simulation

**Solutions**:

```bash
# 1. Check FFmpeg installation
ffmpeg -version

# 2. Check for errors in terminal output
# Look for "Error generating animation" messages

# 3. Manually generate from existing data
python -m src.satellite_control.visualization.unified_visualizer

# 4. Check disk space
df -h  # Ensure sufficient free space
```

### Plots Look Wrong

**Problem**: Garbled plots, missing data, or incorrect scaling

**Solutions**:

```python
# Regenerate with fresh data
python -m src.satellite_control.visualization.unified_visualizer

# Check CSV data integrity
import pandas as pd
physics = pd.read_csv('Data/<timestamp>/physics_data.csv')
control = pd.read_csv('Data/<timestamp>/control_data.csv')

print(physics.info())  # Check for NaN or inf values
print(control.info())
```

### Terminal Dashboard Not Displaying

**Problem**: No colored terminal output during simulation

**Solutions**:

```bash
# Check terminal compatibility
echo $TERM  # Should be xterm-256color or similar

# Update Rich library
pip install --upgrade rich

# Use basic mode if terminal incompatible
export TERM=dumb
python run_simulation.py

# Or run headless
python run_simulation.py --no-anim
```

### Video Quality Poor

**Problem**: Blurry or low-quality MP4 animation

**Solution**:

```python
# Edit unified_visualizer.py
# Increase DPI and bitrate
self.dpi = 200      # Increase from 150
self.bitrate = 8000  # Increase from 5000
self.fps = 60       # Increase from 30 for smoother
```

---

## Testing Issues

### Tests Fail to Run

**Problem**: `pytest` command not found or tests don't execute

**Solution**:

```bash
# Install pytest
pip install pytest pytest-cov

# Run from project root
cd /Users/aevar/Desktop/Satellite_3D_MuJoCo
pytest tests/

# If module import errors, check __init__.py files
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/e2e/__init__.py
```

### Tests Timeout

**Problem**: Tests hang or take forever

**Solution**:

```bash
# Set timeout for slow tests
pytest --timeout=10

# Run only fast tests
pytest -m "not slow"

# Skip integration tests
pytest tests/unit/
```

### Flaky Tests

**Problem**: Tests pass sometimes, fail other times

**Solutions**:

```python
# Add retry decorator for unreliable tests
import pytest

@pytest.mark.flaky(reruns=3)
def test_sometimes_fails():
    # Test that might fail randomly
    pass

# Control randomness with seeds
import numpy as np
np.random.seed(42)
```

---

## Data Analysis Issues

### CSV File Not Found

**Problem**: "File not found" when loading data

**Solution**:

```bash
# List available data directories
ls -lt Data/ | head

# Use most recent
DATA_DIR=$(ls -t Data/ | head -1)
echo "Latest data: Data/$DATA_DIR"

# Check files exist
ls -lh Data/$DATA_DIR/
```

### CSV Contains NaN Values

**Problem**: NaN or inf values in data files

**Diagnosis**:

```python
import pandas as pd
df = pd.read_csv('Data/<timestamp>/physics_data.csv')

# Check for NaN
print(df.isna().sum())

# Check for inf
import numpy as np
print(np.isinf(df.select_dtypes(include=[np.number])).sum())
```

**Solutions**:

- May indicate MPC solver failures
- Check `mpc_status` column in control_data.csv
- Review MPC parameters if frequent failures

### Insufficient Data Points

**Problem**: Too few data points for analysis

**Cause**: Simulation ended too early

**Solution**:

```bash
# Run longer simulation
python run_simulation.py --duration 60.0

# Check mission complete criteria
# May need to adjust tolerances in config/mpc_params.py
```

---

## Getting Help

### Documentation Resources

1. **README.md** - Quick start and overview
2. **ARCHITECTURE.md** - System design and components
3. **DEVELOPMENT_GUIDE.md** - Development workflows
4. **MATHEMATICS.md** - MPC formulation and equations
5. **TESTING.md** - Running and writing tests
6. **VISUALIZATION.md** - Understanding output visualizations

### Debug Checklist

1. **Enable verbose logging**:

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check config validity**:

   ```bash
   python run_simulation.py config
   ```

3. **Run verification tests**:

   ```bash
   python run_simulation.py verify
   ```

4. **Check MPC solve times**:

   ```bash
   grep "MPC Solve" -r Data/<timestamp>/
   ```

5. **Review simulation logs**:
   ```bash
   # Check terminal output or saved logs
   cat simulation.log
   ```

### Common Parameter Values

For reference, current defaults:

```python
# Physics (config/physics.py)
TOTAL_MASS = 10.0 kg
MOMENT_OF_INERTIA = 0.140 kg·m²

# MPC (config/mpc_params.py)
MPC_PREDICTION_HORIZON = 50        # 3.0s ahead
MPC_CONTROL_HORIZON = 50
Q_POSITION = 1000.0
Q_VELOCITY = 10000.0
Q_ANGLE = 1000.0
Q_ANGULAR_VELOCITY = 1500.0
R_THRUST = 1.0
MAX_VELOCITY = 0.5 m/s

# Timing (config/timing.py)
CONTROL_DT = 0.06s                # 16.67 Hz
SIMULATION_DT = 0.005s            # 200 Hz
```

### Reporting Issues

If you can't resolve the issue:

1. **Search existing issues** on GitHub
2. **Gather information**:
   - Python version (`python --version`)
   - OS and version
   - Error message and full traceback
   - Configuration values (if modified)
   - Data files (if relevant)
3. **Create minimal reproducible example**
4. **Open a new issue** with details

---

## Quick Fixes

### Simulation won't start

```bash
# Check you're using the right command
python run_simulation.py  # NOT python simulation.py

# Verify dependencies
pip install -r requirements.txt
```

### Everything is slow

```bash
# Run headless for max speed
python run_simulation.py --auto --no-anim

# Reduce horizon
# Edit mpc_params.py: MPC_PREDICTION_HORIZON = 30
```

### Can't see visualization

```bash
# Check FFmpeg installed
ffmpeg -version

# Generate manually after run
python -m src.satellite_control.visualization.unified_visualizer
```

### Tests won't run

```bash
# Install pytest
pip install pytest

# Run from project root
cd /Users/aevar/Desktop/Satellite_3D_MuJoCo
pytest tests/
```

---

**For additional support, see [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) and [README.md](README.md)**
