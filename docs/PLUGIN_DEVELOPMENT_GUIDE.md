# Mission Plugin Development Guide

**V4.0.0: Phase 2 - Mission Plugin System**

This guide explains how to create custom mission plugins for the Satellite Control System.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Plugin Interface](#plugin-interface)
4. [Creating a Plugin](#creating-a-plugin)
5. [Plugin Registration](#plugin-registration)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

## Overview

Mission plugins allow you to create custom mission types that integrate seamlessly with the satellite control system. Plugins can:

- Define custom mission logic
- Configure mission parameters interactively or from files
- Calculate target states for the MPC controller
- Determine mission completion conditions

## Quick Start

1. **Copy the example plugin:**
   ```bash
   cp src/satellite_control/mission/plugins/example_custom_plugin.py my_mission.py
   ```

2. **Modify the plugin class:**
   - Change the class name
   - Update `get_name()`, `get_display_name()`, and `get_description()`
   - Implement your mission logic in `configure()`, `get_target_state()`, and `is_complete()`

3. **Register your plugin:**
   ```bash
   satellite-control install-mission my_mission.py
   ```

4. **Use your plugin:**
   ```bash
   satellite-control run --classic
   # Select your mission from the menu
   ```

## Plugin Interface

All plugins must inherit from `MissionPlugin` and implement these methods:

### Required Methods

#### `get_name() -> str`
Returns a unique identifier for the plugin (e.g., `"my_custom_mission"`).

#### `get_display_name() -> str`
Returns a human-readable name (e.g., `"My Custom Mission"`).

#### `get_description() -> str`
Returns a description of what the mission does.

#### `configure(config: SimulationConfig) -> MissionState`
Configures the mission parameters. This method should:
- Interact with the user (if needed) to gather parameters
- Create and return a `MissionState` object
- Set any mission-specific state in the `MissionState`

#### `get_target_state(current_state, time, mission_state) -> np.ndarray`
Calculates the target state for the current time. Returns a 13-element numpy array:
```
[x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
```

#### `is_complete(current_state, time, mission_state) -> bool`
Determines if the mission is complete. Returns `True` when the mission should end.

### Optional Methods

#### `get_version() -> str`
Returns the plugin version (default: `"1.0.0"`).

#### `get_author() -> Optional[str]`
Returns the plugin author (default: `None`).

#### `get_required_parameters() -> List[str]`
Returns a list of required `MissionState` parameter names (default: `[]`).

#### `validate_config(config: SimulationConfig) -> bool`
Validates that the config is compatible with this mission (default: always returns `True`).

## Creating a Plugin

### Step 1: Create the Plugin File

Create a new Python file (e.g., `my_mission.py`) with your plugin class:

```python
from typing import List, Optional, Tuple
import numpy as np

from src.satellite_control.mission.plugin import MissionPlugin
from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.config.mission_state import MissionState, create_mission_state
from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz


class MyCustomMissionPlugin(MissionPlugin):
    """My custom mission plugin."""
    
    def get_name(self) -> str:
        return "my_custom_mission"
    
    def get_display_name(self) -> str:
        return "My Custom Mission"
    
    def get_description(self) -> str:
        return "Description of what this mission does"
    
    def configure(self, config: SimulationConfig) -> MissionState:
        # Configure mission parameters
        mission_state = create_mission_state()
        # ... set mission_state attributes ...
        return mission_state
    
    def get_target_state(
        self,
        current_state: np.ndarray,
        time: float,
        mission_state: MissionState,
    ) -> np.ndarray:
        # Calculate target state
        target_state = np.zeros(13)
        # ... set target_state values ...
        return target_state
    
    def is_complete(
        self,
        current_state: np.ndarray,
        time: float,
        mission_state: MissionState,
    ) -> bool:
        # Check if mission is complete
        return False
```

### Step 2: Implement Mission Logic

#### Configuration (`configure`)

The `configure` method should gather mission parameters. You can:

- Prompt the user for input
- Load from a config file
- Use default values
- Read from `SimulationConfig`

Example:
```python
def configure(self, config: SimulationConfig) -> MissionState:
    mission_state = create_mission_state()
    
    # Get user input
    x = float(input("Target X: ") or "0.0")
    y = float(input("Target Y: ") or "0.0")
    
    # Store in mission state
    mission_state.enable_waypoint_mode = True
    mission_state.waypoint_targets = [(x, y, 0.0)]
    
    return mission_state
```

#### Target State Calculation (`get_target_state`)

Calculate the desired state for the MPC controller. The state vector format is:

```python
[x, y, z,           # Position (meters)
 qw, qx, qy, qz,    # Orientation quaternion
 vx, vy, vz,        # Linear velocity (m/s)
 wx, wy, wz]        # Angular velocity (rad/s)
```

Example:
```python
def get_target_state(self, current_state, time, mission_state):
    target_state = np.zeros(13)
    target_state[0:3] = [1.0, 2.0, 0.0]  # Target position
    target_state[3:7] = [1.0, 0.0, 0.0, 0.0]  # No rotation
    target_state[7:10] = [0.0, 0.0, 0.0]  # Zero velocity
    target_state[10:13] = [0.0, 0.0, 0.0]  # Zero angular velocity
    return target_state
```

#### Mission Completion (`is_complete`)

Determine when the mission should end:

```python
def is_complete(self, current_state, time, mission_state):
    # Example: Complete after 10 seconds
    if time > 10.0:
        return True
    
    # Example: Complete when position error is small
    pos_error = np.linalg.norm(current_state[0:3] - target_pos)
    if pos_error < 0.05:
        return True
    
    return False
```

## Plugin Registration

### Method 1: Install Command

```bash
satellite-control install-mission my_mission.py
```

### Method 2: Config File

Create a YAML or JSON config file:

**plugin_config.yaml:**
```yaml
plugins:
  - path: /path/to/my_mission.py
    name: my_custom_mission  # optional name override
```

Then load it:
```python
from src.satellite_control.mission.plugin import load_plugins_from_config
load_plugins_from_config("plugin_config.yaml")
```

### Method 3: Auto-Discovery

Place your plugin in one of these directories:
- `src/satellite_control/mission/plugins/` (built-in)
- `~/.satellite_control/plugins/` (user plugins)

Plugins in these directories are automatically discovered.

## Best Practices

1. **Error Handling**: Always validate user input and handle errors gracefully
2. **State Management**: Use `MissionState` to store mission-specific data
3. **Documentation**: Provide clear descriptions and docstrings
4. **Testing**: Test your plugin with various scenarios
5. **Versioning**: Use semantic versioning for plugin versions
6. **Validation**: Implement `validate_config()` to check compatibility

## Examples

See the example plugin:
- `src/satellite_control/mission/plugins/example_custom_plugin.py`

This demonstrates:
- Basic plugin structure
- Interactive configuration
- Target state calculation
- Mission completion logic

## Troubleshooting

### Plugin Not Appearing in Menu

1. Check that the plugin is registered:
   ```bash
   satellite-control list-missions
   ```

2. Verify the plugin file is valid Python
3. Check for validation errors in logs

### Plugin Validation Errors

Common issues:
- Missing required methods
- Invalid plugin name (must be alphanumeric with `_` or `-`)
- Methods returning wrong types

### Import Errors

Make sure all imports are available:
```python
from src.satellite_control.mission.plugin import MissionPlugin
from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.config.mission_state import MissionState
```

## Advanced Topics

### Using MissionState

The `MissionState` object can store custom attributes:

```python
mission_state.custom_target = (1.0, 2.0, 0.0)
mission_state.custom_parameter = 42
```

### Accessing Configuration

Use `SimulationConfig` to access system parameters:

```python
def configure(self, config: SimulationConfig):
    dt = config.app_config.simulation.control_dt
    mass = config.app_config.physics.total_mass
    # ...
```

### Complex Mission Logic

For complex missions, consider:
- Using helper classes
- Storing intermediate state in the plugin instance
- Implementing state machines
- Using external libraries (with proper error handling)

## Support

For questions or issues:
1. Check the example plugin
2. Review existing built-in plugins
3. Check the codebase documentation
4. Review plugin validation errors
