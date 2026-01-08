# Building API Documentation

This guide explains how to build and view the API documentation for the Satellite Control System.

## Prerequisites

Install documentation dependencies:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

Or install from `pyproject.toml`:

```bash
pip install -e ".[docs]"
```

## Building Documentation

### Quick Build

```bash
cd docs
make html
```

The documentation will be built in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser.

### Clean Build

```bash
cd docs
make clean
make html
```

### View Documentation

After building, open:
```
docs/_build/html/index.html
```

## Documentation Structure

- **User Guide**: Architecture, development guide, testing guide
- **API Reference**: Auto-generated from source code docstrings
  - `api/control.rst`: MPC controllers
  - `api/core.rst`: Core simulation components
  - `api/utils.rst`: Utility functions
  - `api/testing.rst`: Testing utilities
- **Additional Docs**: Simulation, visualization, physics, mathematics

## Writing Documentation

### Docstring Format

Use Google-style docstrings (supported by Napoleon extension):

```python
def solve_mpc(self, state: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Solve MPC optimization problem.
    
    Args:
        state: Current state vector [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        target: Target state vector (same format)
        
    Returns:
        Optimal control action [u1, u2, ..., u12]
        
    Raises:
        OptimizationError: If solver fails
    """
    ...
```

### Module Documentation

Add module-level docstrings:

```python
"""
MPC Controller Module

Provides Model Predictive Control for satellite thruster control.
Uses OSQP solver for real-time optimization.
"""
```

## CI Integration

Documentation is automatically built in CI (`.github/workflows/ci.yml`). The build is non-blocking but will warn if it fails.

## Troubleshooting

### Import Errors

If you see import errors when building:
- Check that all dependencies are installed
- Verify `sys.path` in `docs/conf.py` includes the source directory
- Some external dependencies (mujoco, osqp) are mocked - this is normal

### Missing Documentation

If a module doesn't appear in the API docs:
- Check that it's listed in the appropriate `.rst` file in `docs/api/`
- Verify the module path is correct
- Ensure the module has docstrings

### Build Errors

If the build fails:
- Run `make clean` first
- Check for syntax errors in `.rst` files
- Verify Sphinx version compatibility
