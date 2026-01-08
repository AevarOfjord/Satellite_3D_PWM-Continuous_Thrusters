# Dashboard Usage Guide

**V4.0.0: Phase 3 - Web-Based Dashboard**

## Quick Start

The dashboard can be launched in several ways:

### Method 1: Using Python Module (Recommended)

```bash
python3 -m src.satellite_control.cli dashboard
```

This will launch the dashboard at `http://localhost:8501`

### Method 2: Using run_simulation.py

```bash
python3 run_simulation.py dashboard
```

### Method 3: Direct Streamlit Command

```bash
streamlit run src/satellite_control/visualization/dashboard.py
```

### Method 4: With Custom Port/Host

```bash
# Custom port
python3 -m src.satellite_control.cli dashboard --port 8502

# Custom host (for remote access)
python3 -m src.satellite_control.cli dashboard --host 0.0.0.0 --port 8501
```

## Dashboard Features

Once launched, the dashboard provides:

1. **📊 Overview Tab**
   - Key simulation metrics
   - Data preview table
   - Quick statistics

2. **🎯 3D Trajectory Tab**
   - Interactive 3D plot of satellite path
   - Start/end markers
   - Target position overlay

3. **📈 Telemetry Tab**
   - Position, velocity, and orientation plots
   - Time-series data visualization

4. **⚙️ Configuration Tab**
   - Current configuration display
   - (Editor coming soon)

5. **🎯 Mission Progress Tab**
   - Mission phase tracking
   - Waypoint progress
   - Phase distribution charts

6. **📊 Performance Metrics Tab**
   - MPC solver performance
   - Control error analysis
   - Thruster activity metrics

## Requirements

- Streamlit must be installed: `pip install streamlit`
- Plotly must be installed: `pip install plotly`
- Simulation data in `Data/Simulation/` directory

## Troubleshooting

### "Command not found: satellite-control"

The `satellite-control` command is not installed as a system command. Use one of the methods above instead.

### "No simulation data found"

Make sure you have run at least one simulation. Data should be in:
```
Data/Simulation/DD-MM-YYYY_HH-MM-SS/control_data.csv
```

### Dashboard won't start

1. Check Streamlit is installed: `pip install streamlit`
2. Check you're in the project root directory
3. Try running directly: `streamlit run src/satellite_control/visualization/dashboard.py`

## Accessing the Dashboard

Once launched, open your web browser and navigate to:
- Local: `http://localhost:8501`
- Remote: `http://<your-ip>:8501` (if using `--host 0.0.0.0`)
