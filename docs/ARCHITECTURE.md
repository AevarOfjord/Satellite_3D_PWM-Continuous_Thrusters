# Satellite Thruster Control System - Architecture Documentation

This document provides a complete file listing and description of the system architecture. For project overview and features, see [README](../README.md).

> **Note:** This project focuses on high-fidelity simulation using MuJoCo with OSQP-based Model Predictive Control.

---

## Directory Structure

```
Satellite_3D_MuJoCo/
│
├── run_simulation.py                 # Main Entry Point (delegates to CLI)
│
├── src/
│   └── satellite_control/            # Main Package
│       │
│       ├── cli.py                    # Command-Line Interface (Typer)
│       │
│       ├── core/                     # Core Simulation & Physics
│       │   ├── simulation.py         # Main Simulation Engine (Orchestrator)
│       │   ├── simulation_initialization.py # Simulation Setup & Initialization
│       │   ├── simulation_loop.py    # Main Simulation Loop Execution
│       │   ├── simulation_runner.py  # Simulation Execution Wrapper
│       │   ├── simulation_context.py # Simulation Context Object
│       │   ├── simulation_logger.py  # Real-Time Terminal Dashboard
│       │   ├── simulation_io.py      # File I/O Operations
│       │   ├── simulation_constants.py # Core Constants
│       │   ├── performance_monitor.py # Performance Monitoring & Metrics
│       │   ├── error_handling.py     # Error Handling Utilities
│       │   ├── mujoco_satellite.py   # MuJoCo Physics Wrapper
│       │   ├── model.py              # Satellite Physics Model
│       │   ├── thruster_manager.py   # Thruster Valve & PWM Logic
│       │   ├── mpc_runner.py         # MPC Execution Loop
│       │   ├── interfaces.py         # Protocol Definitions
│       │   └── exceptions.py         # Custom Exceptions
│       │
│       ├── control/                  # Control Algorithms
│       │   └── mpc_controller.py     # MPC Optimizer (OSQP)
│       │
│       ├── mission/                  # Mission Logic & UI
│       │   ├── mission_manager.py    # Mission Orchestration
│       │   ├── mission_state_manager.py # State Machine & Phase Logic
│       │   ├── mission_report_generator.py # Post-Mission Analysis
│       │   ├── mission_logic.py      # Mission Type Definitions
│       │   ├── mission_handler_base.py # Base Handler Class
│       │   ├── interactive_cli.py    # Modern Interactive Menu (Questionary)
│       │   └── mission_cli.py        # Classic Text-Based Menu
│       │
│       ├── visualization/            # Rendering & Output
│       │   ├── unified_visualizer.py # Post-Process Visualization Orchestrator
│       │   ├── plot_generator.py     # Performance Plot Generation
│       │   ├── video_renderer.py     # MP4 Animation Generation
│       │   ├── simulation_visualization.py # Live MuJoCo Viewer Logic
│       │   ├── satellite_2d_diagram.py # 2D Satellite Diagram Helpers
│       │   └── shape_utils.py        # Shape Transformation Utilities
│       │
│       ├── utils/                    # Utilities
│       │   ├── data_logger.py        # Buffered CSV Logger
│       │   ├── logging_config.py     # Logging Setup
│       │   ├── navigation_utils.py   # Geometric Helpers
│       │   ├── orientation_utils.py # Quaternion & Euler Conversions
│       │   ├── simulation_state_validator.py # State Validation
│       │   ├── profiler.py           # Performance Profiling
│       │   ├── spline_path.py        # Spline Path Generation
│       │   ├── state_converter.py    # State Conversion Utilities
│       │   ├── caching.py            # LRU Cache Utilities
│       │   └── viewer_overlay.py     # MuJoCo Viewer Overlays
│       │
│       ├── config/                   # Configuration
│       │   ├── satellite_config.py   # Main Config Interface (Legacy)
│       │   ├── simulation_config.py  # Immutable Config Container (Preferred)
│       │   ├── mpc_params.py         # MPC Tuning Parameters
│       │   ├── physics.py            # Physical Constants
│       │   ├── timing.py             # Timing Parameters
│       │   ├── mission_state.py      # Runtime Mission State
│       │   ├── models.py             # Pydantic Config Models
│       │   ├── constants.py          # System-Wide Constants
│       │   ├── obstacles.py          # Obstacle Definitions
│       │   ├── thruster_config.py    # Thruster Configuration
│       │   ├── presets.py            # Configuration Presets
│       │   └── validator.py          # Configuration Validation
│       │
│       └── testing/                  # Test Utilities
│           └── ... (test helpers)
│
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md               # This file
│   ├── MATHEMATICS.md                # Mathematical Formulation
│   ├── DEVELOPMENT_GUIDE.md          # Developer Guide
│   ├── TESTING.md                    # Testing Documentation
│   └── ... (other docs)
│
├── Data/                             # Output Directory
│   └── Simulation/                   # Logs, Videos, Reports
│       └── [timestamp]/              # Per-run output
│           ├── simulation_data.csv
│           ├── Simulation_animation.mp4
│           └── mission_summary.txt
│
├── models/                           # MuJoCo Model Files
│   └── satellite_model.xml
│
├── DXF/                              # Custom Shape Files
│
└── tests/                            # Test Suite
    ├── e2e/                          # End-to-End Tests
    ├── unit/                         # Unit Tests
    └── integration/                  # Integration Tests
```

---

## Key Architectural Improvements

1. **Modular Package Structure**: All source code is encapsulated within `src/satellite_control`, separated into semantic subpackages (`core`, `control`, `mission`, `visualization`, `utils`, `config`).

2. **Performance-Optimized MPC**: The MPC controller uses a **persistent OSQP solver** with matrix warm-starting, achieving solve times of **~1-2ms** per timestep.

3. **Memory-Efficient Logging**: Buffered CSV writing prevents RAM exhaustion during long simulations, with configurable flush intervals.

4. **Decoupled Visualization**: Post-process visualization reads from CSV files, ensuring the physics loop is never blocked by rendering overhead.

5. **Rich Terminal UI**: Live dashboard using `rich` library with real-time telemetry, thruster status, and mission phase information.

6. **Interactive CLI**: Modern menu system using `questionary` for arrow-key navigation and visual mission configuration.

---

## Component Descriptions

### Entry Point

| File                | Purpose                                                                              |
| ------------------- | ------------------------------------------------------------------------------------ |
| `run_simulation.py` | Lightweight entry point that delegates to the CLI module for backward compatibility. |
| `cli.py`            | Main CLI interface using Typer. Provides `run`, `verify`, and `config` commands.     |

### Core (`src/satellite_control/core`)

The simulation engine and physics integration.

| File                      | Purpose                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------ |
| `simulation.py`           | **The orchestrator.** Main simulation class that coordinates initialization, loop execution, and cleanup. |
| `simulation_initialization.py` | Handles all simulation setup: physics initialization, target state, timing, thruster manager, MPC controller, mission manager, etc. |
| `simulation_loop.py`      | Encapsulates main simulation loop execution: step updates, termination conditions, waypoint advancement, data saving. |
| `simulation_runner.py`    | High-level simulation execution wrapper with error handling and resource cleanup.                      |
| `simulation_context.py`   | Encapsulates simulation dependencies and runtime state in a single context object.                     |
| `simulation_logger.py`    | Real-time terminal dashboard displaying telemetry, thruster activity, and mission status using `rich`. |
| `simulation_io.py`        | Handles file I/O operations for saving simulation data and reports.                                    |
| `simulation_constants.py` | Core simulation constants and configuration values.                                                    |
| `performance_monitor.py`  | Performance monitoring with timing statistics, threshold checking, and metrics export.                 |
| `error_handling.py`       | Error handling utilities and context managers.                                                          |
| `mujoco_satellite.py`     | Low-level wrapper around MuJoCo physics engine. Handles force application and state retrieval.         |
| `model.py`                | Defines satellite physical properties (mass, inertia, thruster geometry) for controller and simulator. |
| `thruster_manager.py`     | Manages realistic thruster behavior including valve delays, thrust ramp-up, and PWM duty cycle logic.  |
| `mpc_runner.py`           | MPC execution loop that coordinates controller calls and command application.                          |
| `interfaces.py`           | Protocol definitions for clean component interfaces (runtime type checking).                           |
| `exceptions.py`           | Custom exception hierarchy for simulation error handling.                                              |

### Control (`src/satellite_control/control`)

| File                | Purpose                                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `mpc_controller.py` | **The Brain.** Linearized Model Predictive Controller using OSQP. Solves a quadratic program at every timestep (~2ms). |

### Mission (`src/satellite_control/mission`)

High-level mission orchestration and user interface.

| File                          | Purpose                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| `mission_manager.py`          | Coordinates mission setup, execution, and completion across different mission types.       |
| `mission_state_manager.py`    | State machine managing mission phases (e.g., "approaching waypoint", "stabilizing").       |
| `mission_report_generator.py` | Post-mission analysis generating detailed performance reports with accuracy and stability. |
| `mission_logic.py`            | Defines mission type logic (Waypoint Navigation, Shape Following, etc.).                   |
| `mission_handler_base.py`     | Base class for mission-specific handlers.                                                  |
| `interactive_cli.py`          | Modern interactive menu using `questionary` for styled, arrow-key navigation.              |
| `mission_cli.py`              | Classic text-based menu interface (fallback).                                              |

### Visualization (`src/satellite_control/visualization`)

| File                          | Purpose                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `unified_visualizer.py`       | Post-process visualization orchestrator. Coordinates plot generation, video rendering, and report creation. |
| `plot_generator.py`           | Generates performance plots: trajectory, position/angle tracking, control effort, thruster usage, etc. |
| `video_renderer.py`           | Generates MP4 animations from simulation data with 3D visualization and info panels.                  |
| `simulation_visualization.py` | Live visualization coordinator during simulation runtime. Manages MuJoCo viewer updates.               |
| `satellite_2d_diagram.py`     | Utility functions for drawing 2D satellite diagrams and trajectory plots.                              |
| `shape_utils.py`              | Shape transformation utilities: demo shapes, DXF loading, path offsetting, rotation/translation.      |

### Configuration (`src/satellite_control/config`)

Centralized configuration using dataclasses and Pydantic models.

| File                  | Purpose                                                                            |
| --------------------- | ---------------------------------------------------------------------------------- |
| `satellite_config.py` | **Main config interface (legacy).** Central hub for accessing all subsystem configurations. Deprecated in favor of `SimulationConfig`. |
| `simulation_config.py` | **Immutable config container (preferred).** Dependency injection pattern with `AppConfig` and `MissionState`. Eliminates global mutable state. |
| `mpc_params.py`       | MPC tuning parameters (prediction horizon, control weights, constraints).          |
| `physics.py`          | Physical constants (mass, inertia, thruster force, damping coefficients).          |
| `timing.py`           | Timing parameters (timestep, control update rate, max simulation time).            |
| `mission_state.py`    | Runtime mission state (current waypoint, trajectory, mission phase).               |
| `models.py`           | Pydantic models for typed configuration with validation (`AppConfig`, `MPCParams`, `SatellitePhysicalParams`, etc.). |
| `constants.py`        | System-wide constants (coordinate systems, conversion factors).                    |
| `obstacles.py`        | Obstacle definitions and presets (positions, radii, safety margins).               |
| `thruster_config.py`  | Thruster physical configuration (positions, orientations, valve characteristics).  |
| `presets.py`          | Configuration presets system with interactive selection and validation.          |
| `validator.py`        | Configuration validation utilities with comprehensive parameter checking.           |

### Utilities (`src/satellite_control/utils`)

| File                            | Purpose                                                                             |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| `data_logger.py`                | High-performance buffered CSV logger with configurable flush intervals.             |
| `logging_config.py`             | Python logging configuration for file and console output (structured JSON support). |
| `navigation_utils.py`           | Geometric utility functions (distance, angle wrapping, coordinate transformations). |
| `orientation_utils.py`         | Quaternion and Euler angle conversion utilities for 3D orientation handling.        |
| `simulation_state_validator.py` | Validates simulation state for numerical stability and physical plausibility.       |
| `profiler.py`                   | Performance profiling utilities with histogram generation and timing statistics.    |
| `spline_path.py`                | Cubic spline path generation for smooth trajectory following.                       |
| `state_converter.py`            | State vector conversion utilities between different representations (sim↔MPC formats). |
| `caching.py`                     | LRU cache utilities with config-based caching for expensive computations.           |
| `viewer_overlay.py`             | Text overlay rendering for MuJoCo viewer (telemetry display).                       |

---

## Data Flow Architecture

```mermaid
graph TD
    User[User/CLI] -->|Configure| Mission[Mission Manager]
    Mission -->|Initialize| Sim[Simulation Loop]

    subgraph "Core Simulation Loop (50+ Hz)"
        Sim -->|Get State| Physics[MuJoCo Physics]
        Physics -->|State x,v,θ,ω| MPC[MPC Controller]
        MPC -->|Linearize & Solve QP| MPC
        MPC -->|Thruster Commands| ThrusterMgr[Thruster Manager]
        ThrusterMgr -->|Valve Delays & Ramp-up| ThrusterMgr
        ThrusterMgr -->|Actual Forces| Physics
        Physics -->|Apply Forces| Physics
    end

    Sim -->|Update Phase| MissionState[Mission State Manager]
    MissionState -->|New Target| Sim

    Physics -->|Log State| Logger[Buffered Data Logger]
    Logger -->|Periodic Flush| Disk[(CSV Files)]

    subgraph "Post-Process Visualization"
        Disk -->|Read CSV| Viz[Unified Visualizer]
        Viz -->|Render Frames| Video[MP4 Animation]
        Viz -->|Plot Data| Charts[Performance Plots]
        Viz -->|Analyze| Report[Mission Report]
    end

    subgraph "Live Display (Optional)"
        Physics -.->|Real-time View| Viewer[MuJoCo Viewer]
        Sim -.->|Terminal UI| Dashboard[Rich Dashboard]
    end
```

---

## Key Workflows

### Running a Simulation

1. **Launch**: User runs `python run_simulation.py` (or `python -m src.satellite_control.cli run`)
2. **Configuration**:
   - Interactive menu (questionary) or classic text menu
   - User selects mission type and parameters
   - Configuration applied to `SatelliteConfig`
3. **Initialization**:
   - `Simulation` object created
   - MuJoCo physics engine initialized with satellite model
   - `MPCController` sets up persistent OSQP solver environment
   - `ThrusterManager` initializes valve delay queues
4. **Simulation Loop** (via `SimulationLoop`, runs at physics timestep, typically 50-200 Hz):
   - **Step 1**: Retrieve current state from MuJoCo (`position`, `velocity`, `angle`, `angular_velocity`)
   - **Step 2**: `MissionStateManager` updates current target based on mission phase
   - **Step 3**: `MPCController` linearizes dynamics around current state and solves optimal control problem (~1-2ms)
   - **Step 4**: Thruster commands sent to `ThrusterManager`
   - **Step 5**: `ThrusterManager` processes valve delays and thrust ramp-up
   - **Step 6**: Actual thruster forces applied to physics engine
   - **Step 7**: State and controls buffered in `DataLogger`
   - **Step 8**: Live dashboard updated with current telemetry
   - **Step 9**: Check termination conditions (target reached, time limit, etc.)
5. **Completion**:
   - `DataLogger` flushes remaining data to `simulation_data.csv`
   - `SimulationIO` saves mission summary and finalizes data export
   - `PerformanceMonitor` exports metrics and checks thresholds
   - `MissionReportGenerator` analyzes trajectory and creates `mission_summary.txt`
   - `UnifiedVisualizer` orchestrates post-process visualization:
     - `PlotGenerator` creates performance plots
     - `VideoRenderer` generates `Simulation_animation.mp4` and `Simulation_3D_Render.mp4`
   - Resources cleaned up (MuJoCo model, file handles, etc.)

### Testing & Verification

The project includes comprehensive test coverage:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Full simulation runs with validation
- **Property-Based Tests**: Hypothesis-driven testing for edge cases

Run tests via:

```bash
python run_simulation.py verify              # Quick E2E tests
python run_simulation.py verify --full       # Full test suite
pytest tests/                                # Direct pytest invocation
```

---

## Performance Characteristics

| Component            | Performance Metric                  |
| -------------------- | ----------------------------------- |
| MPC Solve Time       | ~1-2ms per timestep (OSQP)          |
| Physics Timestep     | 0.01-0.02s (configurable)           |
| Control Update Rate  | 50-100 Hz                           |
| CSV Write Latency    | Buffered, ~1s flush interval        |
| Memory Usage         | <100MB for 10-minute simulation     |
| Animation Generation | ~30 seconds for 5-minute simulation |

---

## Design Principles

1. **Separation of Concerns**: Physics, control, visualization, and mission logic are cleanly separated
2. **Performance First**: Critical loops optimized, with profiling infrastructure built-in
3. **Type Safety**: Comprehensive type hints with mypy validation
4. **Testability**: Protocol-based interfaces enable easy mocking and testing
5. **Configurability**: All parameters accessible via `SatelliteConfig` hierarchy
6. **Professional UX**: Rich terminal UI and interactive menus for excellent developer experience

---

## Extension Points

To extend the system:

- **New Mission Types**: Implement in `mission_logic.py` and add to `mission_manager.py`
- **Alternative Controllers**: Implement control protocol in `interfaces.py`
- **Custom Visualizations**: 
  - Add new plot types in `PlotGenerator`
  - Extend `VideoRenderer` for custom animation styles
  - Add shape utilities in `shape_utils.py`
- **Different Physics**: Implement physics protocol for alternative backends
- **Advanced Profiling**: Add instrumentation points in `profiler.py`
- **New Configuration Sources**: Extend `SimulationConfig` or add new preset types in `presets.py`

## Recent Architectural Improvements

### Large File Refactoring (Completed ✅)

**`simulation.py` Refactoring:**
- **Before**: 1360+ lines, single monolithic class
- **After**: 769 lines (43% reduction), orchestrator pattern
- **New Modules**:
  - `simulation_initialization.py`: All initialization logic (~400 lines)
  - `simulation_loop.py`: Main loop execution (~300 lines)
  - `simulation_context.py`: State container (dataclass)
  - `simulation_logger.py`: Logging utilities
  - `simulation_io.py`: File I/O operations

**`unified_visualizer.py` Refactoring:**
- **Before**: 3203 lines, single monolithic class
- **After**: Reduced size, modular components
- **New Modules**:
  - `plot_generator.py`: Performance plot generation
  - `video_renderer.py`: MP4 animation generation
  - `shape_utils.py`: Shape transformation utilities

### Configuration System Refactoring (Completed ✅)

- **New**: `SimulationConfig` - Immutable configuration container
- **New**: `AppConfig` - Pydantic-based typed configuration
- **Migration**: Gradual migration from mutable `SatelliteConfig` to dependency injection pattern
- **Benefits**: Thread-safe, testable, no global state pollution

### Test Coverage Improvements (In Progress)

- **New Test Files**: 13+ new test modules covering refactored components
- **Test Types**: Unit tests, integration tests, property-based tests (Hypothesis)
- **Coverage**: Targeting 80%+ overall coverage

---

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [OSQP Documentation](https://osqp.org/)
- [Mathematical Formulation](MATHEMATICS.md)
- [Development Guide](DEVELOPMENT_GUIDE.md)
