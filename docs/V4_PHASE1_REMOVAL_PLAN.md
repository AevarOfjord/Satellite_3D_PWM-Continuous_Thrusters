# V4.0.0 Phase 1: SatelliteConfig Removal Plan

**Goal:** Complete removal of `SatelliteConfig` and all backward compatibility code.

## ✅ COMPLETE: Phase 1.1-1.4

- **Status:** ✅ **COMPLETE** (V4.0.0)
- **Total Code References Removed:** 246+
- **Files Migrated:** 14 production + 5 test files
- **Remaining:** 9 uses (class definition itself - needed for adapter), deprecated API tests
- **Strategy:** ✅ Successfully removed all fallbacks systematically

## Removal Order (Priority)

### Phase 1.1: Core Mission Logic (Highest Priority)
1. `mission_state_manager.py` (117 uses) - Remove all fallbacks, require `app_config` and `mission_state`
2. `mission_cli.py` (10 uses) - Remove legacy dead code methods
3. `mission_manager.py` (3 uses) - Final cleanup

### Phase 1.2: Core Physics & Simulation
4. `mujoco_satellite.py` (37 uses) - Remove all fallbacks, require `app_config`
5. `simulation_initialization.py` (5 uses) - Remove report generator compatibility
6. `simulation_loop.py` (4 uses) - Final cleanup

### Phase 1.3: Visualization
7. `unified_visualizer.py` (21 uses) - Remove all fallbacks
8. `video_renderer.py` (15 uses) - Remove all fallbacks
9. `simulation_visualization.py` (5 uses) - Remove all fallbacks
10. `plot_generator.py` (2 uses) - Final cleanup
11. `satellite_2d_diagram.py` (14 uses) - Remove all fallbacks

### Phase 1.4: Utilities & Config
12. `simulation_state_validator.py` (18 uses) - Remove all fallbacks
13. `validator.py` (4 uses) - Remove fallback
14. `simulation_runner.py` (2 uses) - Final cleanup
15. `cli.py` (4 uses) - Remove fallback

### Phase 1.5: Final Cleanup
16. Remove `SatelliteConfig` class from `satellite_config.py`
17. Remove `SatelliteConfig` from `config/__init__.py`
18. Update all tests
19. Remove deprecation warnings
20. Update documentation

## Strategy

For each file:
1. Identify all `SatelliteConfig` usages
2. Replace with `SimulationConfig`/`AppConfig`/`MissionState` equivalents
3. Remove fallback code paths
4. Update function signatures to require new config (no optional fallbacks)
5. Update tests
6. Verify no regressions

## Breaking Changes

- All functions that currently accept optional `app_config`/`mission_state` will require them
- No backward compatibility with V2.x code
- Minimum version requirement: V3.0.0

## Testing Strategy

1. Run full test suite after each file migration
2. Verify simulation runs successfully
3. Check that all mission types work
4. Verify visualization still works
