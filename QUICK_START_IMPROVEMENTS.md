# Quick Start: Implementing Improvements

This guide helps you implement the improvements from `IMPROVEMENT_PLAN.md` in order of priority.

## 🚀 Immediate Actions (5 minutes)

### 1. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install

# Test it works
pre-commit run --all-files
```

**What this does:** Automatically checks code quality before every commit.

---

## 🔧 Quick Wins (30 minutes each)

### 2. Fix Hardcoded Config in CLI

**File:** `src/satellite_control/cli.py`

**Change:**
```python
# BEFORE (lines 59-62):
if auto:
    console.print("[yellow]Running in AUTO mode with default parameters...[/yellow]")
    SatelliteConfig.DEFAULT_START_POS = (1.0, 1.0, 0.0)
    SatelliteConfig.DEFAULT_TARGET_POS = (0.0, 0.0, 0.0)
    # ...

# AFTER:
if auto:
    console.print("[yellow]Running in AUTO mode with default parameters...[/yellow]")
    config_overrides = {
        'default_start_pos': (1.0, 1.0, 0.0),
        'default_target_pos': (0.0, 0.0, 0.0),
        'default_start_angle': (0.0, 0.0, 0.0),
        'default_target_angle': (0.0, 0.0, 0.0),
    }
else:
    config_overrides = {}

# Then pass to simulation:
sim = SatelliteMPCLinearizedSimulation(config_overrides=config_overrides)
```

---

### 3. Add Coverage Threshold

**File:** `.github/workflows/ci.yml` (already done!)

The coverage job now enforces 70% minimum. To increase:
```yaml
--cov-fail-under=80  # Change 70 to 80
```

---

## 📋 Testing Your Changes

After making changes, run:

```bash
# 1. Run pre-commit checks
pre-commit run --all-files

# 2. Run tests
pytest tests/ -v

# 3. Check coverage
pytest tests/ --cov=src/satellite_control --cov-report=term-missing

# 4. Run type checking
mypy src/satellite_control/ --ignore-missing-imports
```

---

## 🎯 Next Steps

1. **Read** `IMPROVEMENT_PLAN.md` for detailed implementation guides
2. **Prioritize** based on your needs:
   - Need better tests? → Focus on #5 (Test Coverage)
   - Having config issues? → Focus on #1 (Configuration Refactor)
   - Want better performance? → Focus on #6 (Performance Monitoring)
3. **Implement incrementally** - don't try to do everything at once

---

## 💡 Tips

- **Use feature branches** for each improvement
- **Write tests first** (TDD) for new features
- **Update documentation** as you go
- **Run CI locally** before pushing:
  ```bash
  act -j test  # If you have act installed
  ```

---

## 🆘 Need Help?

- Check `docs/DEVELOPMENT_GUIDE.md` for development patterns
- Review `docs/ARCHITECTURE.md` for system design
- See `IMPROVEMENT_PLAN.md` for detailed implementation steps
