# Configuration Migration Guide

## Overview

This guide helps you migrate from the deprecated mutable global state pattern (`SatelliteConfig` class attributes) to the new dependency injection pattern using `SimulationConfig` and `MissionState`.

## Why Migrate?

The old pattern has several problems:
- **Test Pollution**: Tests can interfere with each other
- **Thread Safety**: Not thread-safe, cannot run parallel simulations
- **Hard to Track**: Direct attribute access makes dependencies unclear
- **Mutable Global State**: Anti-pattern that makes code harder to reason about

The new pattern provides:
- **Thread Safety**: Each simulation has its own immutable config
- **Testability**: Easy to test with different configurations
- **Explicit Dependencies**: Configuration passed explicitly via dependency injection
- **Type Safety**: Better IDE support and type checking

## Migration Timeline

- **v1.0.0**: Deprecation warnings added
- **v2.0.0**: Migration in progress
- **v3.0.0**: Migration continued
- **v4.0.0**: ✅ **COMPLETE** - `SatelliteConfig` removed from production code

## Quick Reference

### Mission State Attributes

| Old (Deprecated) | New (Recommended) |
|-----------------|-------------------|
| `SatelliteConfig.ENABLE_WAYPOINT_MODE` | `mission_state.enable_waypoint_mode` |
| `SatelliteConfig.WAYPOINT_TARGETS` | `mission_state.waypoint_targets` |
| `SatelliteConfig.WAYPOINT_ANGLES` | `mission_state.waypoint_angles` |
| `SatelliteConfig.CURRENT_TARGET_INDEX` | `mission_state.current_target_index` |
| `SatelliteConfig.DXF_SHAPE_MODE_ACTIVE` | `mission_state.dxf_shape_mode_active` |
| `SatelliteConfig.DXF_SHAPE_CENTER` | `mission_state.dxf_shape_center` |
| `SatelliteConfig.DXF_SHAPE_PATH` | `mission_state.dxf_shape_path` |
| `SatelliteConfig.OBSTACLES_ENABLED` | `mission_state.obstacles_enabled` |
| `SatelliteConfig.OBSTACLES` | `mission_state.obstacles` |

### Methods

| Old (Deprecated) | New (Recommended) |
|-----------------|-------------------|
| `SatelliteConfig.set_waypoint_mode(True)` | `mission_state.enable_waypoint_mode = True` |
| `SatelliteConfig.set_waypoint_targets(...)` | `mission_state.waypoint_targets = [...]` |
| `SatelliteConfig.set_obstacles(...)` | `mission_state.obstacles = [...]` or `ObstacleManager.set_obstacles(...)` |

## Step-by-Step Migration

### Step 1: Create SimulationConfig

**Old Way:**
```python
# Direct attribute access (deprecated)
SatelliteConfig.ENABLE_WAYPOINT_MODE = True
SatelliteConfig.WAYPOINT_TARGETS = [(1.0, 1.0), (2.0, 2.0)]

sim = SatelliteMPCLinearizedSimulation()
```

**New Way:**
```python
from src.satellite_control.config import SimulationConfig, MissionState

# Create mission state
mission_state = MissionState()
mission_state.enable_waypoint_mode = True
mission_state.waypoint_targets = [(1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]  # Note: 3D tuples

# Create simulation config
config = SimulationConfig.create_with_overrides({
    "mission": {
        "enable_waypoint_mode": True,
        "waypoint_targets": [(1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]
    }
})

# Or update mission state directly
config.mission_state.enable_waypoint_mode = True
config.mission_state.waypoint_targets = [(1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]

sim = SatelliteMPCLinearizedSimulation(config=config)
```

### Step 2: Update Waypoint Configuration

**Old Way:**
```python
SatelliteConfig.set_waypoint_mode(True)
SatelliteConfig.set_waypoint_targets(
    target_points=[(1.0, 1.0), (2.0, 2.0)],
    target_angles=[(0, 0, 0), (0, 0, 0)]
)
```

**New Way:**
```python
from src.satellite_control.config import SimulationConfig

config = SimulationConfig.create_default()
config.mission_state.enable_waypoint_mode = True
config.mission_state.waypoint_targets = [(1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]
config.mission_state.waypoint_angles = [0.0, 0.0]  # Single yaw values
```

### Step 3: Update Obstacle Configuration

**Old Way:**
```python
SatelliteConfig.set_obstacles([(1.0, 1.0, 0.2), (2.0, 2.0, 0.3)])
```

**New Way:**
```python
from src.satellite_control.config import SimulationConfig

config = SimulationConfig.create_default()
config.mission_state.obstacles = [(1.0, 1.0, 0.2), (2.0, 2.0, 0.3)]
config.mission_state.obstacles_enabled = True

# Or use ObstacleManager
from src.satellite_control.config.obstacles import ObstacleManager
obstacle_manager = ObstacleManager()
obstacle_manager.set_obstacles([(1.0, 1.0, 0.2), (2.0, 2.0, 0.3)])
```

### Step 4: Update DXF Shape Configuration

**Old Way:**
```python
SatelliteConfig.DXF_SHAPE_MODE_ACTIVE = True
SatelliteConfig.DXF_SHAPE_CENTER = (0.0, 0.0)
SatelliteConfig.DXF_SHAPE_PATH = [(0, 0), (1, 1), (2, 2)]
```

**New Way:**
```python
from src.satellite_control.config import SimulationConfig

config = SimulationConfig.create_default()
config.mission_state.dxf_shape_mode_active = True
config.mission_state.dxf_shape_center = (0.0, 0.0)
config.mission_state.dxf_shape_path = [(0, 0), (1, 1), (2, 2)]
```

## Common Patterns

### Pattern 1: CLI Integration

**Old Way:**
```python
# In CLI, directly mutating SatelliteConfig
SatelliteConfig.ENABLE_WAYPOINT_MODE = True
SatelliteConfig.WAYPOINT_TARGETS = waypoints

# In simulation
sim = SatelliteMPCLinearizedSimulation()  # Reads from SatelliteConfig
```

**New Way:**
```python
# In CLI, create config
from src.satellite_control.config import SimulationConfig

config = SimulationConfig.create_default()
config.mission_state.enable_waypoint_mode = True
config.mission_state.waypoint_targets = waypoints

# In simulation
sim = SatelliteMPCLinearizedSimulation(config=config)
```

**Note:** The CLI still mutates `SatelliteConfig` for backward compatibility, but the simulation automatically syncs this to `MissionState`. For new code, prefer creating `SimulationConfig` directly.

### Pattern 2: Testing

**Old Way:**
```python
def test_waypoint_mission():
    SatelliteConfig.ENABLE_WAYPOINT_MODE = True
    SatelliteConfig.WAYPOINT_TARGETS = [(1.0, 1.0)]
    # ... test code ...
    SatelliteConfig.reset_mission_state()  # Cleanup
```

**New Way:**
```python
def test_waypoint_mission():
    from src.satellite_control.config import SimulationConfig
    
    config = SimulationConfig.create_default()
    config.mission_state.enable_waypoint_mode = True
    config.mission_state.waypoint_targets = [(1.0, 1.0, 0.0)]
    
    sim = SatelliteMPCLinearizedSimulation(config=config)
    # ... test code ...
    # No cleanup needed - each test gets its own config
```

### Pattern 3: Multiple Simulations

**Old Way:**
```python
# Problem: Second simulation affects first
SatelliteConfig.ENABLE_WAYPOINT_MODE = True
sim1 = SatelliteMPCLinearizedSimulation()

SatelliteConfig.ENABLE_WAYPOINT_MODE = False
sim2 = SatelliteMPCLinearizedSimulation()  # Affects sim1!
```

**New Way:**
```python
# Each simulation has its own config
config1 = SimulationConfig.create_default()
config1.mission_state.enable_waypoint_mode = True
sim1 = SatelliteMPCLinearizedSimulation(config=config1)

config2 = SimulationConfig.create_default()
config2.mission_state.enable_waypoint_mode = False
sim2 = SatelliteMPCLinearizedSimulation(config=config2)  # Independent!
```

## Backward Compatibility

During the transition period, the old pattern still works but shows deprecation warnings:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress warnings

# Old code still works
SatelliteConfig.ENABLE_WAYPOINT_MODE = True
sim = SatelliteMPCLinearizedSimulation()  # Automatically syncs to MissionState
```

However, we recommend migrating to the new pattern for:
- Better testability
- Thread safety
- Clearer dependencies
- Future-proofing

## FAQ

### Q: Do I need to migrate immediately?

**A:** No. The old pattern will continue to work until v2.0.0. However, migrating now gives you:
- Better test isolation
- Thread safety
- Clearer code

### Q: What if I'm using the CLI?

**A:** The CLI still mutates `SatelliteConfig` for backward compatibility. The simulation automatically syncs this to `MissionState`. For programmatic use, prefer creating `SimulationConfig` directly.

### Q: Can I mix old and new patterns?

**A:** Yes, during the transition period. The simulation automatically syncs `SatelliteConfig` mutations to `MissionState`. However, for new code, prefer the new pattern.

### Q: What about immutable config values (like MPC_PREDICTION_HORIZON)?

**A:** Immutable config values (from `AppConfig`) are not deprecated. Only mutable mission state attributes are deprecated. You can still use:
```python
SatelliteConfig.MPC_PREDICTION_HORIZON  # OK - not deprecated
SatelliteConfig.TOTAL_MASS  # OK - not deprecated
```

### Q: How do I access config in my custom code?

**A:** If you're writing code that needs config, prefer dependency injection:

```python
# Good: Accept config as parameter
def my_function(simulation_config: SimulationConfig):
    mpc_horizon = simulation_config.app_config.mpc.prediction_horizon
    waypoint_mode = simulation_config.mission_state.enable_waypoint_mode

# Avoid: Direct SatelliteConfig access
def my_function():
    mpc_horizon = SatelliteConfig.MPC_PREDICTION_HORIZON  # OK (immutable)
    waypoint_mode = SatelliteConfig.ENABLE_WAYPOINT_MODE  # Deprecated!
```

## Additional Resources

- `docs/CONFIG_REFACTOR_PLAN.md` - Detailed refactoring plan
- `src/satellite_control/config/simulation_config.py` - SimulationConfig implementation
- `src/satellite_control/config/mission_state.py` - MissionState implementation

## Getting Help

If you encounter issues during migration:
1. Check this guide first
2. Review the deprecation warnings (they include migration hints)
3. Look at existing migrated code for examples
4. Open an issue if you find a bug
