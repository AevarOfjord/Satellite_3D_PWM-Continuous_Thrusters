# Legacy Code Removal Plan

## Why We Keep Legacy Code (Current Strategy)

### 1. **Backward Compatibility During Transition**
- **Current Status**: 59 uses of deprecated `SatelliteConfig` attributes across 11 source files
- **Test Usage**: 13 uses in test files
- **Reason**: Allows existing code to continue working while developers migrate

### 2. **Gradual Migration Strategy**
- **Timeline**: 
  - v1.0.0: Deprecation warnings (current)
  - v2.0.0: Complete removal (planned)
- **Benefit**: Reduces risk of breaking changes, allows incremental migration

### 3. **CLI Still Uses Legacy Pattern**
- The CLI (`mission_cli.py`, `interactive_cli.py`) still mutates `SatelliteConfig`
- Automatic sync ensures compatibility, but migration is needed for full removal

### 4. **Automatic Sync Mechanism**
- Simulation automatically syncs `SatelliteConfig` → `MissionState` at initialization
- This allows old code to work without immediate migration

## Current Usage Analysis

### Source Files Using Legacy Code (11 files, 59 uses)
1. `mission_cli.py` - 11 uses (CLI interface)
2. `interactive_cli.py` - 9 uses (Interactive CLI)
3. `mission_state_manager.py` - 14 uses (Mission state management)
4. `simulation_loop.py` - 7 uses (Simulation loop)
5. `unified_visualizer.py` - 3 uses (Visualization)
6. `video_renderer.py` - 3 uses (Video rendering)
7. `mission_manager.py` - 3 uses (Mission management)
8. `mission_state.py` - 6 uses (State sync)
9. `simulation.py` - 1 use (Backward compat)
10. `simulation_initialization.py` - 1 use (Backward compat)
11. `satellite_config.py` - 1 use (Self-reference)

### Test Files Using Legacy Code (2 files, 13 uses)
1. `test_config.py` - 9 uses
2. `test_simulation_loop.py` - 4 uses

## Options: Keep vs. Remove

### Option A: Keep Legacy Code (Current Strategy) ✅

**Pros:**
- ✅ No breaking changes
- ✅ Gradual migration
- ✅ Existing code continues to work
- ✅ Lower risk

**Cons:**
- ❌ Code duplication
- ❌ Maintenance burden
- ❌ Confusion about which pattern to use
- ❌ Technical debt

**Timeline:** Remove in v2.0.0 (future release)

### Option B: Remove Legacy Code Now (Accelerated Migration) 🚀

**Pros:**
- ✅ Clean codebase immediately
- ✅ No confusion about patterns
- ✅ Forces best practices
- ✅ Reduces maintenance

**Cons:**
- ❌ Breaking changes (requires migration)
- ❌ Need to update all 11 source files + 2 test files
- ❌ CLI needs refactoring
- ❌ Higher risk of bugs

**Timeline:** Complete migration now, remove in next release

## Recommended Approach: Accelerated Migration

If you want to remove legacy code now, here's the plan:

### Phase 1: Migrate CLI (High Priority)
- Update `mission_cli.py` to create `SimulationConfig` directly
- Update `interactive_cli.py` to use `SimulationConfig`
- Remove `SatelliteConfig` mutations

### Phase 2: Migrate Core Components
- Update `mission_state_manager.py` to use `MissionState` from config
- Update `simulation_loop.py` to use config from simulation
- Update visualization components

### Phase 3: Update Tests
- Migrate `test_config.py` to use `SimulationConfig`
- Migrate `test_simulation_loop.py` to use `SimulationConfig`
- Remove legacy test fixtures

### Phase 4: Remove Legacy Code
- Remove deprecated attributes from `SatelliteConfig`
- Remove backward compatibility code
- Remove sync mechanisms
- Update documentation

## Migration Checklist

### CLI Migration
- [ ] `mission_cli.py` - Create `SimulationConfig` instead of mutating `SatelliteConfig`
- [ ] `interactive_cli.py` - Use `SimulationConfig` for all configuration
- [ ] Remove `SatelliteConfig` mutations from CLI

### Core Migration
- [ ] `mission_state_manager.py` - Use `mission_state` from `SimulationConfig`
- [ ] `simulation_loop.py` - Get config from simulation object
- [ ] `unified_visualizer.py` - Use config from simulation
- [ ] `video_renderer.py` - Use config from simulation
- [ ] `mission_manager.py` - Use config from simulation

### Test Migration
- [ ] `test_config.py` - Use `SimulationConfig` in all tests
- [ ] `test_simulation_loop.py` - Use `SimulationConfig` in all tests
- [ ] Remove legacy test fixtures

### Cleanup
- [ ] Remove deprecated attributes from `SatelliteConfig`
- [ ] Remove `sync_mission_state_from_satellite_config()` function
- [ ] Remove backward compatibility code
- [ ] Update all documentation

## Decision Matrix

| Factor | Keep Legacy | Remove Now |
|--------|------------|------------|
| **Risk** | Low | Medium |
| **Effort** | Low (maintain) | High (migrate) |
| **Code Quality** | Medium | High |
| **Breaking Changes** | None | Yes |
| **Timeline** | v2.0.0 | Next release |
| **Maintenance** | Ongoing | One-time |

## Recommendation

**For Production Codebase:** Keep legacy code until v2.0.0
- Lower risk
- Allows gradual migration
- Better for users

**For Clean Codebase:** Remove legacy code now
- Forces best practices
- Cleaner architecture
- Better long-term maintainability

**Hybrid Approach:** Migrate incrementally, remove in v2.0.0
- Migrate high-priority files (CLI, core) now
- Keep backward compat for low-priority files
- Remove all in v2.0.0

## Next Steps

If you want to accelerate migration:
1. I can help migrate the CLI files first (highest impact)
2. Then migrate core components
3. Update tests
4. Remove legacy code

If you prefer gradual migration:
1. Keep current strategy
2. Remove in v2.0.0
3. Continue adding deprecation warnings

**What would you like to do?**
