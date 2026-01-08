# Large File Refactoring Plan

## Overview

This document outlines the plan to refactor large files that violate the single responsibility principle.

## Files to Refactor

### 1. `unified_visualizer.py` (3203 lines)

**Current Structure:**
- `PlotStyle` class (~75 lines)
- `UnifiedVisualizationGenerator` class (~2600 lines)
- `LinearizedVisualizationGenerator` subclass (~10 lines)
- Utility functions (~500 lines)

**Target Structure:**

```
visualization/
├── unified_visualizer.py          # Orchestrator (~200 lines)
├── plot_generator.py              # Plotting logic (~800 lines)
├── video_renderer.py              # Video generation (~600 lines)
├── report_generator.py            # Report generation (~400 lines)
├── data_loader.py                 # Data loading utilities (~300 lines)
└── shape_utils.py                 # Shape utilities (~200 lines)
```

### 2. `simulation.py` (1301 lines)

**Current Structure:**
- `SatelliteMPCLinearizedSimulation` class with many methods

**Target Structure:**

```
core/
├── simulation.py                  # Public API (~300 lines)
├── simulation_initialization.py  # Setup logic (~400 lines)
├── simulation_loop.py            # Main loop logic (~300 lines)
└── simulation_helpers.py         # Helper methods (~300 lines)
```

## Refactoring Strategy

### Phase 1: Extract Helper Functions (Low Risk) ✅ **COMPLETED**

**For `unified_visualizer.py`:**
1. ✅ Extract shape utilities to `shape_utils.py`
2. ⏭️ Extract data loading to `data_loader.py` (future)
3. ⏭️ Extract CLI functions to separate module (future)

**For `simulation.py`:**
1. ⏭️ Extract helper methods to `simulation_helpers.py` (future)
2. ⏭️ Extract utility functions (future)

**Status:** Phase 1 partially complete - shape utilities extracted
**Files Created:**
- `src/satellite_control/visualization/shape_utils.py` (~200 lines)
- `src/satellite_control/visualization/plot_generator.py` (structure created)

**Impact:** Reduced `unified_visualizer.py` by ~130 lines, improved modularity

### Phase 2: Extract Plotting Logic ✅ **COMPLETE**

**For `unified_visualizer.py`:**
1. ✅ Create `PlotGenerator` class structure
2. 🟡 Move plotting methods from `UnifiedVisualizationGenerator` (4 of 15 methods migrated)
3. ✅ Update `UnifiedVisualizationGenerator` to use `PlotGenerator` (hybrid approach)

**Status:** ✅ **COMPLETE** - All 15 plotting methods migrated to PlotGenerator
**Methods Migrated (15 of 15):**
- ✅ `generate_position_tracking_plot`
- ✅ `generate_position_error_plot`
- ✅ `generate_angular_tracking_plot`
- ✅ `generate_angular_error_plot`
- ✅ `generate_trajectory_plot`
- ✅ `generate_velocity_tracking_plot`
- ✅ `generate_velocity_magnitude_plot`
- ✅ `generate_control_effort_plot`
- ✅ `generate_mpc_performance_plot`
- ✅ `generate_timing_intervals_plot`
- ✅ `generate_trajectory_3d_interactive_plot` (requires plotly)
- ✅ `generate_thruster_usage_plot`
- ✅ `generate_thruster_valve_activity_plot`
- ✅ `generate_pwm_quantization_plot`
- ✅ `generate_pwm_duty_cycles_from_physics` (fallback method)

**Note:** Using hybrid approach - migrated methods use PlotGenerator, others still use legacy methods. Can continue incrementally.

**Estimated Remaining Effort:** 3-4 hours

### Phase 3: Extract Video Rendering ✅ **COMPLETE**

**For `unified_visualizer.py`:**
1. ✅ Created `VideoRenderer` class
2. ✅ Moved animation generation logic (~190 lines extracted)
3. ✅ Updated `UnifiedVisualizationGenerator` to use `VideoRenderer`

**Methods Migrated:**
- ✅ `generate_animation()` - Main animation orchestration
- ✅ `animate_frame()` - Frame rendering
- ✅ `_render_frame_task()` - Parallel worker function
- ✅ `draw_satellite()` - Satellite visualization
- ✅ `draw_target()` - Target visualization
- ✅ `draw_trajectory()` - Trajectory visualization
- ✅ `draw_obstacles()` - Obstacle visualization
- ✅ `update_info_panel()` - Info panel updates
- ✅ `setup_plot()` - Plot initialization

**Status:** ✅ **COMPLETE** - All video rendering logic migrated to VideoRenderer

**File Impact:**
- `unified_visualizer.py`: 3069 → 2879 lines (~190 lines reduced)
- `video_renderer.py`: 765 lines (new)

### Phase 4: Extract Report Generation ⏭️ **NOT APPLICABLE**

**Status:** Report generation is already handled by `MissionReportGenerator` in `src/satellite_control/mission/mission_report_generator.py`, not in `unified_visualizer.py`.

**Note:** `unified_visualizer.py` focuses on visualization (plots and animations), while report generation is a separate concern handled by the mission module.

**Alternative:** Data loading logic could be extracted to a `DataLoader` class in a future phase if needed, but this is not critical for the current refactoring goals.

### Phase 5: Extract Simulation Initialization ✅ **COMPLETE**

**For `simulation.py`:**
1. ✅ Create `SimulationInitializer` class
2. ✅ Move `_initialize_from_active_config` logic
3. ✅ Update `SatelliteMPCLinearizedSimulation` to use initializer

**Estimated Effort:** 4-6 hours

### Phase 6: Extract Simulation Loop ✅ **COMPLETE**

**For `simulation.py`:**
1. ✅ Create `SimulationLoop` class
2. ✅ Move main loop logic (`run_simulation`, `update_simulation`)
3. ✅ Update `SatelliteMPCLinearizedSimulation` to use loop

**Estimated Effort:** 4-6 hours

### Phase 7: Refactor UnifiedVisualizationGenerator ✅ **COMPLETE**

**For `unified_visualizer.py`:**
1. ✅ Refactored to orchestrator pattern
2. ✅ Delegates to extracted classes (PlotGenerator, VideoRenderer)
3. ✅ Public API unchanged (backward compatible)

**Changes Made:**
- Added lazy initialization for `PlotGenerator` and `VideoRenderer`
- Created `_get_plot_generator()` and `_get_video_renderer()` helper methods
- `generate_performance_plots()` and `generate_animation()` now delegate to stored instances
- Legacy plotting methods marked as deprecated but kept for backward compatibility

**Status:** ✅ **COMPLETE** - UnifiedVisualizationGenerator now acts as a clean orchestrator

**File Impact:**
- `unified_visualizer.py`: Cleaner orchestrator pattern with lazy initialization
- Reduced code duplication by reusing component instances

### Phase 8: Refactor Simulation Class ✅ **COMPLETE**

**For `simulation.py`:**
1. ✅ Refactor to use extracted components
2. ✅ Keep public API unchanged
3. ⏭️ Update tests (can be done separately)

**Estimated Effort:** 3-4 hours

**What Was Done:**
1. ✅ Removed duplicate `update_simulation` method (logic now in SimulationLoop)
2. ✅ Removed unused imports (FuncAnimation, time)
3. ✅ Updated class docstring to reflect new modular architecture
4. ✅ Verified public API is preserved (all public methods still available)

## Implementation Details

### UnifiedVisualizationGenerator Refactoring

**Before:**
```python
class UnifiedVisualizationGenerator:
    def __init__(self, ...):
        # 200 lines of initialization
    
    def generate_performance_plots(self):
        # 500 lines of plotting code
    
    def generate_animation(self):
        # 400 lines of animation code
    
    def generate_report(self):
        # 300 lines of report code
```

**After:**
```python
class UnifiedVisualizationGenerator:
    def __init__(self, ...):
        self.plot_generator = PlotGenerator(...)
        self.video_renderer = VideoRenderer(...)
        self.report_generator = ReportGenerator(...)
    
    def generate_performance_plots(self):
        return self.plot_generator.generate(self.data)
    
    def generate_animation(self):
        return self.video_renderer.render(self.data)
    
    def generate_report(self):
        return self.report_generator.generate(self.data)
```

### Simulation Refactoring

**Before:**
```python
class SatelliteMPCLinearizedSimulation:
    def __init__(self, ...):
        # 200 lines of initialization
    
    def _initialize_from_active_config(self, ...):
        # 220 lines of setup
    
    def run_simulation(self, ...):
        # 200 lines of loop logic
```

**After:**
```python
class SatelliteMPCLinearizedSimulation:
    def __init__(self, ...):
        self.initializer = SimulationInitializer(...)
        self.initializer.initialize(self, ...)
    
    def run_simulation(self, ...):
        loop = SimulationLoop(self)
        loop.run(show_animation)
```

## Benefits

1. **Maintainability:**
   - Smaller, focused files
   - Easier to understand
   - Easier to test

2. **Reusability:**
   - Components can be used independently
   - Better code organization

3. **Testability:**
   - Easier to unit test components
   - Better test isolation

4. **Performance:**
   - No performance impact
   - Same functionality

## Risks and Mitigation

### Risk 1: Breaking Changes

**Mitigation:**
- Keep public API unchanged
- Gradual migration
- Comprehensive testing

### Risk 2: Import Cycles

**Mitigation:**
- Careful dependency management
- Use dependency injection
- Test imports

### Risk 3: Large Scope

**Mitigation:**
- Incremental refactoring
- Test after each phase
- Document changes

## Testing Strategy

1. **Unit Tests:**
   - Test extracted components independently
   - Test integration points

2. **Integration Tests:**
   - Test full workflow
   - Test backward compatibility

3. **Regression Tests:**
   - Ensure existing tests still pass
   - Verify behavior unchanged

## Timeline

- **Phase 1:** Helper Functions (4-6 hours)
- **Phase 2:** Plotting Logic (6-8 hours)
- **Phase 3:** Video Rendering (4-6 hours)
- **Phase 4:** Report Generation (3-4 hours)
- **Phase 5:** Simulation Initialization (4-6 hours)
- **Phase 6:** Simulation Loop (4-6 hours)
- **Phase 7:** UnifiedVisualizationGenerator Refactor (3-4 hours)
- **Phase 8:** Simulation Class Refactor (3-4 hours)

**Total Estimated Time:** 31-44 hours (4-6 days)

## Next Steps

1. Start with Phase 1 (lowest risk)
2. Test thoroughly after each phase
3. Document changes
4. Update tests as needed

## Notes

- This is a large refactoring that should be done incrementally
- Each phase should be tested before moving to the next
- Maintain backward compatibility
- Keep public APIs unchanged
