# SatelliteConfig Migration Tracker (V3.0.0)

**Last Updated:** 2026-01-08  
**Target:** Complete removal of `SatelliteConfig` in V3.0.0

## Current Status

- **Total Uses:** 253 across 20 source files
- **Migration Progress:** ~30% (timing params, core simulation partially migrated)
- **Remaining:** ~177 uses across 20 files

## File-by-File Migration Status

### ✅ Partially Migrated (High Priority)

#### `src/satellite_control/core/simulation_loop.py`
- **Uses:** 0 (import removed, only comments remain)
- **Status:** ✅ MIGRATED - All functions require `simulation_config`, no `SatelliteConfig` fallbacks
- **Remaining:**
  - None (import removed)
- **Action:** Complete

#### `src/satellite_control/core/simulation_initialization.py`
- **Uses:** 1 (report generator compatibility only)
- **Status:** ✅ MIGRATED - All functions require `simulation_config`, no fallbacks
- **Remaining:**
  - `SatelliteConfig` used only for `create_mission_report_generator()` compatibility (TODO: migrate report generator)
- **Action:** Migrate `mission_report_generator.py` to accept `SimulationConfig`

#### `src/satellite_control/mission/mission_state_manager.py`
- **Uses:** ~9 (fallback only, docstrings)
- **Status:** ✅ MIGRATED - Uses `app_config` and `mission_state` when provided, falls back to `SatelliteConfig` for backward compatibility. Helper methods added for all DXF and timing parameters.
- **Remaining:**
  - All `SatelliteConfig.*` attribute reads (fallbacks)
  - `SatelliteConfig.DXF_*` attributes
  - `SatelliteConfig.OBSTACLES_*` attributes
- **Action:** High priority - Core mission logic

### ⏳ Not Started (High Priority)

#### `src/satellite_control/core/mujoco_satellite.py`
- **Uses:** 19 (fallback only)
- **Status:** ✅ MIGRATED - Accepts `AppConfig`/`SimulationParams`/`SatellitePhysicalParams`, uses them when provided, falls back to `SatelliteConfig` for backward compatibility
- **Remaining:**
  - `SatelliteConfig.SIMULATION_DT`
  - `SatelliteConfig.SATELLITE_SIZE`
  - `SatelliteConfig.TOTAL_MASS`
  - `SatelliteConfig.MOMENT_OF_INERTIA`
  - `SatelliteConfig.COM_OFFSET`
  - `SatelliteConfig.THRUSTER_POSITIONS`
  - `SatelliteConfig.THRUSTER_FORCES`
  - `SatelliteConfig.USE_REALISTIC_PHYSICS`
  - `SatelliteConfig.LINEAR_DAMPING_COEFF`
  - `SatelliteConfig.ROTATIONAL_DAMPING_COEFF`
  - `SatelliteConfig.ENABLE_RANDOM_DISTURBANCES`
  - `SatelliteConfig.DISTURBANCE_FORCE_STD`
  - `SatelliteConfig.DISTURBANCE_TORQUE_STD`
- **Action:** Pass `AppConfig` to constructor, use `app_config.physics.*`

#### `src/satellite_control/control/mpc_controller.py`
- **Uses:** 2
- **Status:** ⚠️ PARTIAL - Uses `AppConfig` but has fallback
- **Remaining:**
  - `SatelliteConfig.get_app_config()` (fallback only)
- **Action:** Remove fallback

### ⏳ Not Started (Medium Priority - Mission Management)

#### `src/satellite_control/mission/mission_cli.py`
- **Uses:** ~11 (legacy methods only, dead code)
- **Status:** ✅ MIGRATED - Main flow uses `MissionState` only, no `SatelliteConfig` mutations
- **Remaining:**
  - Legacy methods `_obstacle_edit_menu`, `_add_single_obstacle`, `_edit_obstacle` (dead code, never called)
- **Action:** Consider removing legacy methods in future cleanup

#### `src/satellite_control/mission/interactive_cli.py`
- **Uses:** 0 (import removed, only comments remain)
- **Status:** ✅ MIGRATED - All functions use `MissionState` only, no `SatelliteConfig` mutations
- **Remaining:**
  - None (import removed)
- **Action:** Complete

#### `src/satellite_control/mission/mission_manager.py`
- **Uses:** 0 (import removed)
- **Status:** ✅ MIGRATED - All functions use `SimulationConfig` and `MissionState`, no `SatelliteConfig` mutations
- **Remaining:**
  - None (import removed)
- **Action:** Complete

### ⏳ Not Started (Lower Priority - Visualization)

#### `src/satellite_control/visualization/unified_visualizer.py`
- **Uses:** ~9 (fallback-only, backward compatibility)
- **Status:** ✅ MIGRATED - Accepts `app_config` and `mission_state` parameters, uses them when provided, falls back to `SatelliteConfig` for backward compatibility
- **Remaining:**
  - `SatelliteConfig.SATELLITE_SIZE` (fallback in `__init__`)
  - `SatelliteConfig.THRUSTER_POSITIONS` (fallback in `__init__`)
  - `SatelliteConfig.THRUSTER_FORCES` (fallback in `__init__`)
  - `SatelliteConfig.DXF_*` attributes (fallback in `__init__`)
  - `SatelliteConfig.OBSTACLES_ENABLED` and `get_obstacles()` (fallback in `draw_obstacles()`)
  - `SatelliteConfig.THRUSTER_POSITIONS` (fallback in `_get_thruster_count()`)
  - `SatelliteConfig.DXF_*` mutations (backward compatibility in `configure_dxf_overlay_interactive()`)
- **Action:** Complete - All main paths use `app_config`/`mission_state`, fallbacks are for backward compatibility only

#### `src/satellite_control/visualization/video_renderer.py`
- **Uses:** ~4 (fallback-only, backward compatibility)
- **Status:** ✅ MIGRATED - Accepts `app_config` and `mission_state` parameters, uses them when provided, falls back to `SatelliteConfig` for backward compatibility
- **Remaining:**
  - `SatelliteConfig.OBSTACLES_ENABLED` and `get_obstacles()` (fallback in `draw_obstacles()`)
  - `SatelliteConfig.DXF_SHAPE_MODE_ACTIVE` (fallback in `generate_animation()`)
  - `SatelliteConfig.*` mutations (backward compatibility in `_render_frame_task()` worker process)
- **Action:** Complete - All main paths use `app_config`/`mission_state`, fallbacks are for backward compatibility only

#### `src/satellite_control/visualization/simulation_visualization.py`
- **Uses:** ~2 (fallback-only, backward compatibility)
- **Status:** ✅ MIGRATED - Uses `app_config` and `mission_state` from controller's `simulation_config`, falls back to `SatelliteConfig` for backward compatibility
- **Remaining:**
  - `SatelliteConfig.get_app_config()` (fallback in `__init__` if controller doesn't have `simulation_config`)
  - `SatelliteConfig.OBSTACLES_ENABLED` and `get_obstacles()` (fallback in `_draw_obstacles()`)
- **Action:** Complete - All main paths use `app_config`/`mission_state`, fallbacks are for backward compatibility only

#### `src/satellite_control/visualization/plot_generator.py`
- **Uses:** ~1 (fallback-only, backward compatibility)
- **Status:** ✅ MIGRATED - Accepts `app_config` parameter, uses it when provided, falls back to `SatelliteConfig` for backward compatibility
- **Remaining:**
  - `SatelliteConfig.THRUSTER_POSITIONS` (fallback in `_get_thruster_count()`)
- **Action:** Complete - All main paths use `app_config`, fallback is for backward compatibility only

### ⏳ Not Started (Utilities)

#### `src/satellite_control/utils/simulation_state_validator.py`
- **Uses:** ~10 (fallback-only, backward compatibility)
- **Status:** ✅ MIGRATED - Accepts `app_config` parameter, uses it when provided, falls back to `SatelliteConfig` for backward compatibility
- **Remaining:**
  - `SatelliteConfig.get_app_config()` (fallback in `__init__` and `create_state_validator_from_config`)
  - `SatelliteConfig.get_mpc_params()` (fallback in `__init__` and `create_state_validator_from_config`)
  - `SatelliteConfig.*` reads for noise params (fallback in `apply_sensor_noise`)
  - `SatelliteConfig.USE_REALISTIC_PHYSICS` (fallback in `apply_sensor_noise`)
  - `SatelliteConfig.POSITION_TOLERANCE`, `ANGLE_TOLERANCE`, `VELOCITY_TOLERANCE`, `ANGULAR_VELOCITY_TOLERANCE` (fallback in `create_state_validator_from_config`)
- **Action:** Complete - All main paths use `app_config`, fallbacks are for backward compatibility only

#### `src/satellite_control/core/mpc_runner.py`
- **Uses:** 0 (import removed, type hint updated)
- **Status:** ✅ MIGRATED - Type hint changed from `SatelliteConfig` to `Optional[AppConfig]`, import removed
- **Remaining:**
  - None (config parameter not currently used, stored for future use)
- **Action:** Complete - No `SatelliteConfig` dependencies

#### `src/satellite_control/core/simulation_runner.py`
- **Uses:** 0 (mutations removed)
- **Status:** ✅ MIGRATED - Removed all `SatelliteConfig` mutations, now creates and passes `SimulationConfig` to simulation
- **Remaining:**
  - None (all mutations removed, uses `SimulationConfig.create_default()` and mission manager's `SimulationConfig`)
- **Action:** Complete - No `SatelliteConfig` dependencies

#### `src/satellite_control/visualization/satellite_2d_diagram.py`
- **Uses:** ~16 (fallback-only, backward compatibility)
- **Status:** ✅ MIGRATED - Functions accept `app_config` parameter, use it when provided, fall back to `SatelliteConfig` for backward compatibility
- **Remaining:**
  - Module-level constants removed, replaced with helper functions that accept `app_config`
  - `SatelliteConfig.*` accesses (all in fallback helper functions)
- **Action:** Complete - All main paths use `app_config`, fallbacks are for backward compatibility only

### ⏳ Not Started (Config & CLI)

#### `src/satellite_control/cli.py`
- **Uses:** ~1 (fallback-only, backward compatibility)
- **Status:** ✅ MIGRATED - Uses `SimulationConfig.create_default()` when available, falls back to `SatelliteConfig.get_app_config()` for backward compatibility
- **Remaining:**
  - `SatelliteConfig.get_app_config()` (fallback in `config` command)
- **Action:** Complete - Main paths use `SimulationConfig`, fallback is for backward compatibility only

#### `src/satellite_control/config/validator.py`
- **Uses:** ~1 (fallback-only, backward compatibility)
- **Status:** ✅ MIGRATED - `validate_config_at_startup()` accepts optional `app_config` parameter, falls back to `SatelliteConfig.get_app_config()` for backward compatibility
- **Remaining:**
  - `SatelliteConfig.get_app_config()` (fallback in `validate_config_at_startup()`)
- **Action:** Complete - Function accepts `app_config` parameter, fallback is for backward compatibility only

## Migration Strategy

### Phase 1: Core Simulation (In Progress)
1. ✅ `simulation_loop.py` - Timing params migrated
2. ✅ `simulation_initialization.py` - Uses `SimulationConfig`
3. ⏳ `mujoco_satellite.py` - **NEXT**
4. ⏳ `mpc_controller.py` - Remove fallback

### Phase 2: Mission Management
1. ⏳ `mission_state_manager.py` - Remove all fallbacks
2. ⏳ `mission_cli.py` - Remove `SatelliteConfig` mutations
3. ⏳ `interactive_cli.py` - Remove `SatelliteConfig` mutations
4. ⏳ `mission_manager.py` - Use `SimulationParams`

### Phase 3: Visualization
1. ⏳ `unified_visualizer.py` - Remove fallbacks
2. ⏳ `video_renderer.py` - Remove fallbacks
3. ⏳ `simulation_visualization.py` - Remove fallback
4. ⏳ `plot_generator.py` - Pass config
5. ⏳ `satellite_2d_diagram.py` - Pass physics config

### Phase 4: Utilities
1. ⏳ `simulation_state_validator.py` - Pass `app_config`
2. ⏳ `mpc_runner.py` - Use `SimulationConfig`
3. ⏳ `simulation_runner.py` - Use `SimulationConfig`

### Phase 5: Final Cleanup
1. ⏳ Remove `SatelliteConfig` class entirely
2. ⏳ Update all tests
3. ⏳ Update documentation

## Migration Patterns

### Pattern 1: Replace Attribute Access

**Before:**
```python
dt = SatelliteConfig.SIMULATION_DT
mass = SatelliteConfig.TOTAL_MASS
```

**After:**
```python
# In function/class that has access to config
dt = app_config.simulation.dt
mass = app_config.physics.total_mass

# Or pass SimulationConfig
config = SimulationConfig.create_default()
dt = config.app_config.simulation.dt
```

### Pattern 2: Replace Method Calls

**Before:**
```python
SatelliteConfig.set_waypoint_mode(True)
SatelliteConfig.add_obstacle(1.0, 2.0, 0.5)
```

**After:**
```python
config = SimulationConfig.create_default()
config.mission_state.enable_waypoint_mode = True
config.mission_state.obstacles.append((1.0, 2.0, 0.5))
```

### Pattern 3: Update Function Signatures

**Before:**
```python
def my_function():
    if SatelliteConfig.ENABLE_WAYPOINT_MODE:
        ...
```

**After:**
```python
def my_function(config: Optional[SimulationConfig] = None):
    config = config or SimulationConfig.create_default()
    if config.mission_state.enable_waypoint_mode:
        ...
```

### Pattern 4: Update Class Initialization

**Before:**
```python
class MyClass:
    def __init__(self):
        self.targets = SatelliteConfig.WAYPOINT_TARGETS
```

**After:**
```python
class MyClass:
    def __init__(self, config: Optional[SimulationConfig] = None):
        config = config or SimulationConfig.create_default()
        self.targets = config.mission_state.waypoint_targets
        self.config = config  # Store for later use
```

## Testing Strategy

1. **Unit Tests:** Update to use `SimulationConfig.create_default()`
2. **Integration Tests:** Pass `SimulationConfig` explicitly
3. **No Global State:** Each test gets its own config
4. **Parallel Safe:** Tests can run in parallel without interference

## Success Criteria

- [ ] Zero `SatelliteConfig.` attribute accesses in source code
- [ ] Zero `SatelliteConfig.` method calls in source code
- [ ] All tests pass with `SimulationConfig` only
- [ ] `SatelliteConfig` class can be safely removed
- [ ] Documentation updated

## Notes

- Many current uses are **fallbacks** for backward compatibility
- Once all code is migrated, fallbacks can be removed
- `SatelliteConfig` class will be removed entirely in V3.0.0
- Migration should be done incrementally to avoid breaking changes
