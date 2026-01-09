# Orbital Inspector Satellite Control System

[![Python 3.9-3.12](https://img.shields.io/badge/python-3.9--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Optimization: OSQP](https://img.shields.io/badge/Optimization-OSQP-green.svg)](https://osqp.org/)
[![Physics: MuJoCo](https://img.shields.io/badge/Physics-MuJoCo-orange.svg)](https://mujoco.org/)

**A 6-DOF orbital inspection satellite simulation with Model Predictive Control, reaction wheels, and multi-satellite coordination.**

---

## 🚀 Features

| Feature                    | Description                                            |
| -------------------------- | ------------------------------------------------------ |
| **Reaction Wheel Control** | 3-axis attitude control with 0.18° precision           |
| **Orbital Dynamics**       | Hill-Clohessy-Wiltshire gravity gradient model         |
| **Multi-Satellite Fleet**  | 3 inspectors with collision avoidance                  |
| **Mission Designer UI**    | Three.js web interface for trajectory planning         |
| **Mission System**         | Pre-built flyby, circumnavigation, inspection missions |

## 🛠️ Technology Stack

| Component   | Technology                         |
| ----------- | ---------------------------------- |
| **Solver**  | OSQP (<5ms solve times)            |
| **Physics** | MuJoCo 3.x                         |
| **Control** | 16-state MPC with 9 control inputs |
| **UI**      | Three.js 3D visualization          |

## 📁 Project Structure

```
├── models/
│   ├── satellite_rw.xml        # Single inspector + target
│   └── satellite_fleet.xml     # Multi-inspector fleet
├── src/satellite_control/
│   ├── control/                # MPC controllers
│   ├── config/                 # Orbital & actuator configs
│   ├── fleet/                  # Multi-satellite coordination
│   ├── mission/                # Mission types & executor
│   └── physics/                # Orbital dynamics (CW equations)
├── missions/                   # Sample mission JSON files
├── ui/                         # Web-based mission designer
└── scripts/                    # Test scripts
```

## ⚡ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python3 scripts/run_all_tests.py

# Launch mission designer UI
cd ui && python3 -m http.server 8080
# Open http://localhost:8080
```

## 🧪 Tests

| Test               | Command                              |
| ------------------ | ------------------------------------ |
| All tests          | `python3 scripts/run_all_tests.py`   |
| Reaction wheels    | `python3 scripts/test_rw_control.py` |
| Orbital dynamics   | `python3 scripts/test_orbital.py`    |
| Fleet coordination | `python3 scripts/test_fleet.py`      |
| Mission system     | `python3 scripts/test_mission.py`    |

## 📊 Performance

| Metric                | Value  |
| --------------------- | ------ |
| Position control      | ±0.5mm |
| Attitude control      | ±0.18° |
| MPC solve time        | <1ms   |
| Station-keeping error | 0.00cm |
| Formation separation  | 8.66m  |

## 📋 Mission Types

```python
from src.satellite_control.mission import (
    create_flyby_mission,
    create_circumnavigation_mission,
    create_station_keeping_mission,
    create_inspection_mission,
)

# Create and execute
mission = create_flyby_mission()
executor = MissionExecutor()
result = executor.execute(mission)
```

## 📄 License

MIT License - See [LICENSE](LICENSE)
