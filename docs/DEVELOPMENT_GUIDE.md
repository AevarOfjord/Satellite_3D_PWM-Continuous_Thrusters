# Development Guide

This guide provides technical details for developers and engineers interested in understanding the codebase architecture and modification process.

**Note**: This project is a demonstration of advanced control systems engineering. While it is primarily a portfolio piece, forks and independent experimentation are encouraged.

---

## Table of Contents

- [Quick Start for Development](#quick-start-for-development)
- [Project Structure](#project-structure)
- [Code Style Guidelines](#code-style-guidelines)
- [Working with Data and CSV Logging](#working-with-data-and-csv-logging)
- [Testing Your Changes](#testing-your-changes)
- [Adding New Features](#adding-new-features)
- [Modifying Parameters](#modifying-parameters)
- [Debugging](#debugging)
- [Git Workflow](#git-workflow)
- [Contributing](#contributing)

---

## Quick Start for Development

### Setup

```bash
# Clone the repository (or your fork)
git clone https://github.com/AevarOfjord/Satellite_2D_MuJoCo.git
cd Satellite_3D_MuJoCo

# Create virtual environment (highly recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Verify Everything Works

```bash
# Run the simulation with default settings
python run_simulation.py

# Run tests
pytest tests/

# Run verification suite
python run_simulation.py verify

# Check code quality
flake8 src/
mypy src/
```

---

## Project Structure

For a complete project structure and design overview, see [ARCHITECTURE.md](ARCHITECTURE.md).

### Key Design Principles

1. **Centralized Configuration**: All parameters in `src/satellite_control/config/` package

   - Never hardcode parameters in other files
   - Import configuration using `SatelliteConfig`

2. **Modular Architecture**: Each module has a single responsibility

   - `mpc_controller.py`: Optimization only
   - `mujoco_satellite.py`: Physics simulation only
   - `simulation.py`: Control loop orchestration

3. **Type Hints**: All functions use comprehensive type hints

   - Enables static analysis and IDE support
   - Catch errors before runtime with `mypy`

4. **Protocol-Based Interfaces**: Clean component boundaries

   - Defined in `core/interfaces.py`
   - Enables mocking and testing

5. **Docstrings**: Google-style docstrings for all classes and functions

   ```python
   def my_function(x: float, y: float) -> float:
       """Brief description.

       Longer description if needed.

       Args:
           x: Description of x
           y: Description of y

       Returns:
           Description of return value

       Raises:
           ValueError: If x is negative
       """
       return x + y
   ```

---

## Code Style Guidelines

### Automated Formatting

We use `black` for code formatting, `isort` for import sorting, and `flake8` for linting.

```bash
# Auto-format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/

# Type check
mypy src/
```

### Style Rules

- **Line length**: Maximum 100 characters (configured in `.flake8`)
- **Naming**:

  - Classes: `PascalCase` (e.g., `SatelliteConfig`, `MPCController`)
  - Functions: `snake_case` (e.g., `calculate_thrust`, `update_state`)
  - Constants: `UPPER_CASE` (e.g., `CONTROL_DT`, `MAX_FORCE`)
  - Private: Leading underscore (e.g., `_helper_function`, `_internal_state`)

- **Imports**: Group in standard order (enforced by `isort`):

  1. Standard library (`import os`, `import sys`)
  2. Third-party (`import numpy`, `import mujoco`)
  3. Local (`from src.satellite_control.config import SatelliteConfig`)

- **Functions**: Keep < 50 lines when possible

  - Break into smaller functions
  - Each function does one thing

- **Comments**: Explain _why_, not _what_

  ```python
  # Good
  # MPC requires linearization around current state for convex approximation
  A_lin = self._linearize_dynamics(x_current)

  # Avoid
  # Linearize dynamics
  A_lin = self._linearize_dynamics(x_current)
  ```

---

## Working with Data and CSV Logging

### CSV Data Format

The simulation produces several CSV files:

**Physics Data**: `physics_data.csv`

- Raw simulation state at every physics timestep (high frequency, ~200 Hz)
- Columns: Time, X_Pos, Y_Pos, Z_Pos, X_Vel, Y_Vel, Z_Vel, Quat_W, Quat_X, Quat_Y, Quat_Z, etc.

**Control Data**: `control_data.csv`

- Controller decisions and commands at control frequency (lower frequency, ~20-50 Hz)
- Columns: Time, Control_Step, Target_X, Target_Y, Target_Yaw, Thruster_Commands, MPC_Solve_Time, etc.

**Terminal Log**: `simulation_terminal_log.csv`

- Human-readable status messages with mission phase information
- 8 columns: Timestamp, Sim_Time, Phase, Pos_Error, Ang_Error, Active_Thrusters, etc.

For complete column definitions, see [VISUALIZATION.md - CSV Format Appendix](VISUALIZATION.md#appendix-csv-data-format-reference).

### Using the DataLogger

The `DataLogger` class in `utils/data_logger.py` handles buffered CSV writing:

```python
from src.satellite_control.utils.data_logger import DataLogger

# Create logger
logger = DataLogger(
    output_dir="Data/Simulation/test_run",
    buffer_size=1000  # Flush every 1000 entries
)

# Log physics data
logger.log_physics_state({
    "Time": current_time,
    "X_Pos": x_position,
    "Y_Pos": y_position,
    "Yaw": yaw_angle,
    # ... other columns
})

# Log control data
logger.log_control_step({
    "Control_Step": step_count,
    "MPC_Solve_Time": solve_time,
    "Thruster_Commands": command_vector,
    # ... other columns
})

# Flush and save when done
logger.close()
```

**Important**: The logger uses buffered writing for performance. Data is periodically flushed to disk, not immediately.

### Modifying CSV Format

To change the CSV format:

1. **Update DataLogger**: Edit `_get_physics_headers()` or `_get_control_headers()` in `utils/data_logger.py`
2. **Update visualization**: Modify `visualization/unified_visualizer.py` to read new columns
3. **Update documentation**: Reflect changes in [VISUALIZATION.md](VISUALIZATION.md) CSV format appendix
4. **Update tests**: Verify new format in `tests/unit/test_data_logger.py`

---

## Testing Your Changes

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_mpc_controller.py

# Run specific test function
pytest tests/unit/test_mpc_controller.py::test_mpc_solve_time

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage

# Run E2E tests only
pytest tests/e2e/ -v

# Run fast tests (skip slow E2E)
pytest -m "not slow"
```

### Using the CLI verify Command

```bash
# Run quick verification tests
python run_simulation.py verify

# Run full test suite (slower)
python run_simulation.py verify --full
```

### Writing Tests

Create tests following existing patterns:

```python
# tests/unit/test_myfeature.py
import pytest
import numpy as np
from src.satellite_control.config import SatelliteConfig
from src.satellite_control.mymodule import my_function


class TestMyFeature:
    """Test suite for my new feature."""

    def test_basic_functionality(self):
        """Test basic case."""
        result = my_function(1.0, 2.0)
        assert result == pytest.approx(3.0)

    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            my_function(-1.0, 2.0)

    @pytest.fixture
    def sample_config(self):
        """Provide sample configuration for tests."""
        return SatelliteConfig.get_app_config()

    def test_with_fixture(self, sample_config):
        """Test using fixture."""
        assert sample_config.physics.mass > 0
```

### Testing Workflow

```bash
# 1. Make changes to code
# (Edit files in src/satellite_control/)

# 2. Format and check
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/

# 3. Run tests to catch regressions
pytest

# 4. Test in simulation
python run_simulation.py
# Select mission and verify behavior
```

---

## Adding New Features

### Example: Adding a New Mission Type

**Goal**: Add "Figure-8 Navigation" mission

#### Step 1: Define Mission Logic

Add to `src/satellite_control/mission/mission_logic.py`:

```python
def configure_figure8_mission(
    center: Tuple[float, float],
    size: float,
    speed: float
) -> Dict[str, Any]:
    """Configure figure-8 navigation mission.

    Args:
        center: Center point of figure-8
        size: Size of each lobe (meters)
        speed: Target velocity along path (m/s)

    Returns:
        Mission configuration dictionary
    """
    # Generate figure-8 path points
    t = np.linspace(0, 2*np.pi, 100)
    x = center[0] + size * np.sin(t)
    y = center[1] + size * np.sin(t) * np.cos(t)

    mission = {
        'type': 'figure8',
        'path_points': list(zip(x, y)),
        'target_speed': speed,
        'center': center,
    }
    return mission
```

#### Step 2: Add Mission Handler

Create handler in `src/satellite_control/mission/mission_state_manager.py`:

```python
def _update_figure8_mission(self, current_state: Dict) -> Tuple[float, float, float]:
    """Update target for figure-8 mission.

    Args:
        current_state: Current satellite state

    Returns:
        Target (x, y, yaw) tuple
    """
    # Find closest point on path
    # Update target based on progress
    # Return next target position and orientation
    pass
```

#### Step 3: Add CLI Menu Option

Update `src/satellite_control/mission/interactive_cli.py`:

```python
def show_mission_menu(self) -> str:
    """Show main mission menu."""
    choice = questionary.select(
        "Select mission type:",
        choices=[
            "Waypoint Navigation",
            "Shape Following",
            "Figure-8 Navigation",  # New option
            "Exit"
        ],
        style=CUSTOM_STYLE,
        qmark=QMARK
    ).ask()

    if choice == "Figure-8 Navigation":
        return "figure8"
    # ... handle other choices
```

#### Step 4: Add Configuration

If needed, add parameters to `src/satellite_control/config/mission_state.py`:

```python
# Figure-8 mission defaults
FIGURE8_DEFAULT_SIZE = 2.0  # meters
FIGURE8_DEFAULT_SPEED = 0.5  # m/s
```

#### Step 5: Add Visualization Support

Update `src/satellite_control/visualization/unified_visualizer.py` to draw figure-8 path overlay.

#### Step 6: Write Tests

Add tests in `tests/unit/test_mission_logic.py`:

```python
def test_figure8_mission_generation():
    """Test figure-8 path generation."""
    mission = configure_figure8_mission(
        center=(0, 0),
        size=2.0,
        speed=0.5
    )

    assert mission['type'] == 'figure8'
    assert len(mission['path_points']) > 0
    assert mission['target_speed'] == 0.5
```

#### Step 7: Document

Update [README.md](../README.md) with new mission type description.

#### Step 8: Test End-to-End

```bash
# Run tests
pytest tests/unit/test_mission_logic.py

# Test in simulation
python run_simulation.py
# Select: Figure-8 Navigation
# Verify trajectory follows expected path
```

---

## Modifying Parameters

### Where Parameters Live

All parameters are in `src/satellite_control/config/`:

```
config/
├── satellite_config.py   # Main config interface
├── physics.py            # Mass, inertia, thruster specs
├── timing.py             # Control rate, timesteps
├── mpc_params.py         # MPC horizons, weights, constraints
├── mission_state.py      # Mission parameters
├── constants.py          # System constants
├── obstacles.py          # Obstacle definitions
├── thruster_config.py    # Thruster configuration
└── models.py             # Pydantic validation models
```

### Guidelines

1. **Never hardcode values** in implementation files

   ```python
   # Good
   from src.satellite_control.config import SatelliteConfig
   config = SatelliteConfig.get_app_config()
   mass = config.physics.mass

   # Bad
   mass = 12.5  # Hardcoded!
   ```

2. **Group related parameters**

   ```python
   # Good - related parameters together in mpc_params.py
   PREDICTION_HORIZON = 15
   CONTROL_HORIZON = 10
   SOLVER_TIME_LIMIT = 0.04  # seconds
   ```

3. **Use descriptive names with units**

   ```python
   # Good
   STABILIZATION_VELOCITY_THRESHOLD = 0.03  # m/s
   VALVE_DELAY_TIME = 0.015  # seconds

   # Unclear
   V_THRESH = 0.03
   DELAY = 0.015
   ```

4. **Use Pydantic models for validation**

   ```python
   # In config/models.py
   from pydantic import BaseModel, Field

   class PhysicsConfig(BaseModel):
       mass: float = Field(gt=0, description="Satellite mass in kg")
       inertia: float = Field(gt=0, description="Moment of inertia in kg⋅m²")

       @field_validator('mass')
       def validate_mass(cls, v):
           if v > 100:
               warnings.warn("Mass seems unusually high")
           return v
   ```

### Adding New Parameters

1. Add to appropriate config file:

   ```python
   # In config/physics.py
   NEW_DAMPING_COEFFICIENT = 0.05  # N⋅s/m
   ```

2. If using Pydantic models, add to the model:

   ```python
   # In config/models.py
   class PhysicsConfig(BaseModel):
       # ...existing fields...
       damping_coefficient: float = 0.05
   ```

3. Access in your code:

   ```python
   from src.satellite_control.config import SatelliteConfig

   config = SatelliteConfig.get_app_config()
   damping = config.physics.damping_coefficient
   ```

### Configuration Validation

```bash
# Validate configuration
python run_simulation.py config

# Dump full configuration as JSON
python run_simulation.py config --dump
```

---

## Debugging

### Using Logging

The project uses Python's `logging` module configured in `utils/logging_config.py`:

```python
import logging
from src.satellite_control.utils.logging_config import setup_logging

# Setup logging (done automatically at startup)
setup_logging()

# Get logger for your module
logger = logging.getLogger(__name__)

# Log at different levels
logger.debug(f"Current state: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")
logger.info("Starting MPC optimization")
logger.warning(f"Solve time {solve_time:.3f}s exceeds limit {limit:.3f}s")
logger.error(f"MPC solve failed: {error_msg}")
```

### Interactive Debugging

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use built-in breakpoint() (Python 3.7+)
breakpoint()

# Debugger commands:
# n: next line
# s: step into function
# c: continue execution
# p variable: print variable value
# l: list surrounding code
# h: help
# q: quit debugger
```

### Testing with Debug Output

```bash
# Run tests with print output visible
pytest tests/unit/test_mpc_controller.py -v -s

# -v: verbose test names
# -s: show print statements and logs
```

### Performance Profiling

The project includes built-in profiling via `utils/profiler.py`:

```python
from src.satellite_control.utils.profiler import PerformanceProfiler

# In simulation code
profiler = PerformanceProfiler()

with profiler.measure("mpc_solve"):
    solution = mpc.solve(state, target)

with profiler.measure("physics_step"):
    physics.step(dt)

# Print statistics
profiler.print_summary()
```

### Live Simulation Debugging

The rich terminal dashboard shows real-time telemetry. To add custom debug info:

```python
# In simulation_logger.py, add custom panel
from rich.panel import Panel

debug_text = f"Custom Debug Info:\n{my_variable}"
console.print(Panel(debug_text, title="Debug"))
```

---

## Git Workflow

### Working with Your Fork

Since this is primarily a portfolio demonstration project, if you want to make modifications:

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Make changes** on feature branches
4. **Commit and push** to your fork
5. **Report bugs** back as issues if you find problems

### Making Changes

```bash
# Create feature branch
git checkout -b feature/my-enhancement

# Make changes and test
# (edit files, run tests, etc.)

# Stage changes
git add src/satellite_control/myfile.py tests/test_myfile.py

# Commit with descriptive message
git commit -m "feat: add new feature X

Detailed explanation of changes:
- Added Y functionality
- Modified Z for better performance
- Tests included"

# Push to your fork
git push -u origin feature/my-enhancement
```

### Commit Message Guidelines

Follow conventional commits format:

```
<type>: <brief description>

<optional detailed explanation>

<optional footer>
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring without behavior change
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `chore`: Maintenance tasks
- `style`: Code style changes (formatting, etc.)

Examples:

```
feat: add figure-8 mission type

Implements a new mission type that follows a figure-8 trajectory.
Includes interactive CLI integration and visualization support.

fix: correct thruster force calculation in realistic physics mode

The force was previously not accounting for valve delay properly.
This fix ensures thrust ramp-up is applied after valve opening.

refactor: extract thruster management into separate module

Moves thruster valve delay and PWM logic from simulation.py
into new thruster_manager.py for better separation of concerns.
```

### Before Committing

```bash
# Format code
black src/ tests/
isort src/ tests/

# Run linters
flake8 src/ tests/
mypy src/

# Run tests
pytest

# If all pass, commit
git add .
git commit -m "your message"
```

### Pre-commit Hooks (Optional)

Install pre-commit hooks to automate checks:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Now checks run automatically before each commit
```

The project includes a `.pre-commit-config.yaml` that runs:

- `black` (formatting)
- `isort` (import sorting)
- `flake8` (linting)
- `mypy` (type checking)

---

## Contributing

### For Your Own Fork

When working on your own fork, maintain these quality standards:

### Code Quality Checklist

Before committing:

- [ ] Code follows PEP 8 style (checked by `flake8`)
- [ ] All code formatted with `black` and `isort`
- [ ] All functions have docstrings (Google style)
- [ ] Type hints on all function signatures
- [ ] Tests written for new functionality
- [ ] All tests pass (`pytest`)
- [ ] No new warnings from `flake8` or `mypy`
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventional commits

### Adding Tests

For every new feature or bug fix:

1. **Write test first** (TDD approach)
2. **Implement feature/fix** until test passes
3. **Add edge case tests** for robustness
4. **Verify coverage** with `pytest --cov`

Example test structure:

```python
# tests/unit/test_new_feature.py
import pytest
from src.satellite_control.mymodule import MyClass


class TestMyClass:
    """Test suite for MyClass."""

    @pytest.fixture
    def instance(self):
        """Provide MyClass instance for tests."""
        return MyClass(param=1.0)

    def test_basic_operation(self, instance):
        """Test basic functionality."""
        result = instance.process(2.0)
        assert result == pytest.approx(3.0)

    def test_edge_case_zero(self, instance):
        """Test with zero input."""
        result = instance.process(0.0)
        assert result == pytest.approx(1.0)

    def test_invalid_input(self, instance):
        """Test with invalid input."""
        with pytest.raises(ValueError, match="must be positive"):
            instance.process(-1.0)
```

### Documentation

When making significant changes:

- **README.md**: Update for user-facing changes
- **ARCHITECTURE.md**: Update for design/structure changes
- **Docstrings**: Keep inline with code changes
- **This guide**: Update for development process changes
- **VISUALIZATION.md**: Update CSV format appendix if data format changes

### Reporting Bugs

If you discover bugs in the original project:

1. Create an issue on GitHub with:

   - Clear problem description
   - Steps to reproduce
   - Environment info (OS, Python version, MuJoCo version)
   - Error messages or log snippets
   - Suggested fix if you have one

2. The maintainer will evaluate and fix verified bugs when available

3. You'll be credited in the issue and commit messages

---

## Common Development Tasks

### Running Simulation with Custom Parameters

```bash
# Run with specific duration
python run_simulation.py --duration 60

# Run in auto mode (skip menu)
python run_simulation.py --auto

# Run without animation (faster)
python run_simulation.py --no-anim

# Use classic text menu instead of interactive
python run_simulation.py --classic
```

### Generating Visualizations Only

If you have existing simulation data:

```python
from src.satellite_control.visualization.unified_visualizer import UnifiedVisualizationGenerator

# Generate visualizations from existing data
viz = UnifiedVisualizationGenerator(
    data_directory="Data/Simulation",
    interactive=True  # Let user select which run
)
```

### Comparing Different Configurations

```bash
# Run baseline
python run_simulation.py
# Results saved to Data/Simulation/<timestamp-1>/

# Modify parameters in config/
# Edit config/mpc_params.py or config/physics.py

# Run with new parameters
python run_simulation.py
# Results saved to Data/Simulation/<timestamp-2>/

# Compare results by examining mission_summary.txt in each folder
```

### Profiling Performance

```python
# In your code
from src.satellite_control.utils.profiler import PerformanceProfiler

profiler = PerformanceProfiler()

# Wrap code to profile
with profiler.measure("section_name"):
    # Code to profile
    pass

# View results
profiler.print_summary()
profiler.plot_histogram("section_name")
```

---

## Troubleshooting Development

### Import Errors

```python
# Problem: "ModuleNotFoundError: No module named 'src.satellite_control'"

# Solution 1: Always run from project root
cd /path/to/Satellite_3D_MuJoCo
python run_simulation.py

# Solution 2: Install package in editable mode
pip install -e .

# Solution 3: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Satellite_3D_MuJoCo"
```

### Test Discovery Issues

```bash
# Ensure proper test structure
# tests/__init__.py should exist
touch tests/__init__.py

# Run with explicit test directory
pytest tests/

# Debug test discovery
pytest --collect-only -v
```

### Type Checking Issues

```bash
# Install type stubs for dependencies
pip install types-numpy types-setuptools

# Check specific module
mypy src/satellite_control/control/mpc_controller.py

# Ignore specific line if needed
result = external_function()  # type: ignore

# Add type: ignore comment with reason
result = legacy_code()  # type: ignore  # TODO: add types to legacy module
```

### MuJoCo Installation Issues

```bash
# Ensure MuJoCo is properly installed
pip install mujoco

# Verify installation
python -c "import mujoco; print(mujoco.__version__)"

# If issues persist, check MuJoCo documentation:
# https://mujoco.readthedocs.io/
```

### OSQP Solver Issues

```bash
# Reinstall OSQP if needed
pip uninstall osqp
pip install osqp

# Verify installation
python -c "import osqp; print(osqp.__version__)"
```

---

## Resources

### Documentation

- **MuJoCo**: https://mujoco.readthedocs.io/
- **OSQP**: https://osqp.org/docs/
- **NumPy**: https://numpy.org/doc/
- **Rich**: https://rich.readthedocs.io/ (for terminal UI)
- **Questionary**: https://questionary.readthedocs.io/ (for interactive menus)

### Python Best Practices

- **PEP 8**: https://www.python.org/dev/peps/pep-0008/
- **Google Python Style Guide**: https://google.github.io/styleguide/pyguide.html
- **Type Hints**: https://docs.python.org/3/library/typing.html
- **Pytest Documentation**: https://docs.pytest.org/

### Control Theory

- **MPC Introduction**: https://www.mpc.berkeley.edu/mpc-course-material
- **Linearization**: Understanding system linearization for MPC
- **Quadratic Programming**: OSQP is a QP solver

---

## Related Documentation

This guide complements other important documents:

- **[README.md](../README.md)** - Project overview and quick start
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and component relationships
- **[QUICKSTART.md](QUICKSTART.md)** - Getting started guide for new users
- **[VISUALIZATION.md](VISUALIZATION.md)** - Output analysis and CSV format reference
- **[MATHEMATICS.md](MATHEMATICS.md)** - Mathematical formulation of the MPC problem
- **[TESTING.md](TESTING.md)** - Comprehensive testing and simulation validation guide
- **[SIMULATION.md](SIMULATION.md)** - Mission types and simulation loop architecture

---

## Getting Help

If you're stuck:

1. **Check existing documentation** listed above
2. **Review related code** for examples of similar functionality
3. **Run tests** to understand expected behavior
4. **Create an issue** on GitHub with:
   - What you're trying to do
   - What you've tried
   - Error messages or unexpected behavior
   - Your environment (OS, Python version, etc.)

Happy developing! 🚀
