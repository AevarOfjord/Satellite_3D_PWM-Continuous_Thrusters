# Real-Time Satellite MPC Control with MuJoCo

[![Python 3.9-3.12](https://img.shields.io/badge/python-3.9--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Optimization: OSQP](https://img.shields.io/badge/Optimization-OSQP-green.svg)](https://osqp.org/)
[![Physics: MuJoCo](https://img.shields.io/badge/Physics-MuJoCo-orange.svg)](https://mujoco.org/)

**A high-fidelity 2D planar control system demonstrating real-time Model Predictive Control (MPC) of a satellite with 8 thrusters on an air-bearing platform.**

This project showcases advanced control engineering capabilities, including convex optimization, state-space modeling, and physics-based simulation architecture.

---

## 🚀 Engineering Highlights

This is a personal portfolio project demonstrating expertise in:

- **Real-Time Optimization**: Implementation of a custom MPC controller using **OSQP** (Operator Splitting Quadratic Program), achieving **<5ms solve times** suitable for 20Hz+ control loops.
- **Physics Simulation**: Integration with **MuJoCo** for industry-standard rigid body dynamics, modeling friction, collisions, and thruster actuation.
- **Control Architecture**: Robust state machine handling waypoint navigation, trajectory generation, and obstacle avoidance.
- **Modern Python Stack**: Type-hinted codebase, automated testing, and modular design patterns.

## 🛠️ Technology Stack

| Component    | Technology | Description                                                    |
| :----------- | :--------- | :------------------------------------------------------------- |
| **Solver**   | **OSQP**   | Fast, robust convex QP solver replacing legacy MIP approaches. |
| **Physics**  | **MuJoCo** | High-performance physics engine for realistic dynamics.        |
| **Language** | **Python** | Core logic, with `numpy` vectorization for performance.        |
| **Testing**  | **Pytest** | Comprehensive unit and integration test suite.                 |

## 🏗️ Architecture

The system operates a closed-loop control cycle:

```mermaid
graph LR
    A[Mission Planner] -->|Target State| B[MPC Controller]
    B -->|Solve QP| C(Optimal Control u*)
    C -->|PWM Command| D[MuJoCo Physics]
    D -->|State Feedback| B
    style B fill:#d4f1f4,stroke:#0077b6,stroke-width:2px
    style D fill:#ffe8d6,stroke:#ff9f1c,stroke-width:2px
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## ⚡ Quick Start

### Prerequisites

- Python 3.9 - 3.12 (Python 3.13 not yet supported by dependencies)

### Installation

```bash
# Clone and install dependencies
git clone https://github.com/AevarOfjord/mpc-air-bearing-satellite.git
cd mpc-air-bearing-satellite
pip install -r requirements.txt
```

### Running the Simulation

```bash
python3 run_simulation.py
```

Follow the interactive prompts to select missions (e.g., Waypoint Navigation, Shape Following).

### Visualization

The simulation automatically generates MP4 animations and telemetry plots in the `Data/` directory.

---

## 📂 Project Structure

- `src/satellite_control`: Core source code.
  - `control/`: MPC implementation (`mpc_controller.py`).
  - `core/`: Simulation loop and physics interface.
  - `config/`: Physics and control parameters.
- `tests/`: Comprehensive test suite.
- `docs/`: Technical documentation.

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

### Getting Started

- **[QUICKSTART.md](docs/QUICKSTART.md)** - 5-minute guide to your first simulation
- **[FAQ.md](docs/FAQ.md)** - Common questions and troubleshooting

### Core Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and component overview
- **[SIMULATION.md](docs/SIMULATION.md)** - Mission types, loop architecture, and timing
- **[MATHEMATICS.md](docs/MATHEMATICS.md)** - MPC formulation and control theory

### Development & Testing

- **[DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md)** - Contributing and development workflows
- **[TESTING.md](docs/TESTING.md)** - Running tests and simulation validation

### Reference

- **[PHYSICS_ENGINE.md](docs/PHYSICS_ENGINE.md)** - MuJoCo integration details
- **[VISUALIZATION.md](docs/VISUALIZATION.md)** - Understanding output and CSV format
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Project version history

## 👨‍💻 About

**Aevar A. Oefjoerd**  
_Robot Control & Automation Engineer_

This project serves as a reference implementation for my skills in **GNC (Guidance, Navigation, and Control)**, **Optimization**, and **Software Engineering**.

[GitHub Profile](https://github.com/AevarOfjord)

---

_Note: This repository is a professional portfolio project demonstrating advanced control engineering capabilities._
