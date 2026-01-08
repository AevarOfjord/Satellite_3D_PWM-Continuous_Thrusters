# Performance Benchmarking Guide

This guide explains how to run and interpret performance benchmarks for the Satellite Control System.

## Quick Start

```bash
# Install benchmark dependencies
pip install -e ".[dev]"

# Run all benchmarks
pytest tests/benchmarks/ --benchmark-only

# Run with comparison (requires previous benchmark data)
pytest tests/benchmarks/ --benchmark-compare

# Generate JSON report
pytest tests/benchmarks/ --benchmark-json=benchmarks.json
```

## Benchmark Categories

### 1. MPC Controller Benchmarks

Tests MPC solver performance:

```bash
pytest tests/test_benchmark.py::TestMPCBenchmarks --benchmark-only
```

**Key Metrics:**
- MPC solve time (target: < 5ms)
- Linearization time
- Trajectory following performance

### 2. Physics Simulation Benchmarks

Tests physics step performance:

```bash
pytest tests/benchmarks/test_physics_benchmarks.py --benchmark-only
```

**Key Metrics:**
- Physics step time (target: < 1ms)
- State getter/setter performance
- Control application performance

### 3. Regression Detection Tests

Tests that detect performance regressions:

```bash
pytest tests/test_benchmark.py::TestMPCRegressionDetection -v
```

**Thresholds:**
- Median solve time: < 15ms
- P95 solve time: < 50ms
- Max solve time: < 70ms

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
pytest tests/benchmarks/ --benchmark-only

# Run specific benchmark
pytest tests/test_benchmark.py::TestMPCBenchmarks::test_mpc_solve_time --benchmark-only

# Run with verbose output
pytest tests/benchmarks/ --benchmark-only -v
```

### Benchmark Comparison

Compare current benchmarks against previous runs:

```bash
# First run (creates baseline)
pytest tests/benchmarks/ --benchmark-only --benchmark-json=baseline.json

# Later run (compares against baseline)
pytest tests/benchmarks/ --benchmark-compare=baseline.json
```

### Generating Reports

```bash
# JSON report
pytest tests/benchmarks/ --benchmark-json=benchmarks.json

# HTML report (requires pytest-html)
pytest tests/benchmarks/ --benchmark-only --html=benchmark_report.html
```

## Interpreting Results

### Benchmark Output

```
test_mpc_solve_time (test_benchmark.py::TestMPCBenchmarks) ... 
  Mean: 4.23ms
  Median: 4.10ms
  Std Dev: 0.15ms
  Min: 3.95ms
  Max: 4.50ms
```

### Performance Thresholds

| Component | Metric | Target | Warning |
|-----------|--------|--------|---------|
| MPC Solve | Median | < 5ms | > 10ms |
| MPC Solve | P95 | < 15ms | > 30ms |
| Physics Step | Mean | < 1ms | > 2ms |
| Linearization | Mean | < 0.5ms | > 1ms |

### Regression Detection

Regression tests automatically fail if:
- Median solve time exceeds threshold
- P95 solve time exceeds threshold
- Max solve time exceeds threshold
- Memory growth > 5MB over 100 iterations

## CI Integration

Benchmarks run automatically in CI but don't fail the build:

```yaml
# .github/workflows/ci.yml
- name: Run benchmarks
  run: pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmarks.json
  continue-on-error: true
```

## Best Practices

### 1. Run Benchmarks Regularly

```bash
# Before committing
pytest tests/benchmarks/ --benchmark-only

# Before releases
pytest tests/benchmarks/ --benchmark-compare=baseline.json
```

### 2. Track Performance Over Time

```bash
# Save baseline
pytest tests/benchmarks/ --benchmark-json=baseline_$(date +%Y%m%d).json

# Compare against baseline
pytest tests/benchmarks/ --benchmark-compare=baseline_20260107.json
```

### 3. Isolate Performance Issues

```bash
# Run specific benchmark
pytest tests/test_benchmark.py::TestMPCBenchmarks::test_mpc_solve_time --benchmark-only

# Run with profiling
pytest tests/test_benchmark.py::TestMPCBenchmarks::test_mpc_solve_time --benchmark-only --profile
```

### 4. Monitor Memory Usage

```bash
# Run memory stability test
pytest tests/test_benchmark.py::TestMPCRegressionDetection::test_mpc_memory_stability_long_run -v
```

## Troubleshooting

### Benchmarks Too Slow

- Reduce number of iterations: `--benchmark-min-rounds=5`
- Skip slow tests: `-m "not slow"`
- Run specific benchmarks only

### Inconsistent Results

- Ensure system is idle
- Close other applications
- Run multiple times and average
- Check for thermal throttling

### Import Errors

```bash
# Install benchmark dependencies
pip install pytest-benchmark

# Or install all dev dependencies
pip install -e ".[dev]"
```

## Advanced Usage

### Custom Benchmark Configuration

Create `pytest.ini`:

```ini
[pytest]
benchmark_min_rounds = 10
benchmark_max_time = 10.0
benchmark_warmup_iterations = 2
```

### Benchmark Hooks

```python
def pytest_benchmark_update_machine_info(config, machine_info):
    """Add custom machine info to benchmark results."""
    machine_info['cpu_count'] = os.cpu_count()
    machine_info['python_version'] = sys.version
```

### Performance Profiling

```bash
# Profile with cProfile
pytest tests/benchmarks/ --benchmark-only --profile

# Profile with line_profiler
pytest tests/benchmarks/ --benchmark-only --profile --profile-svg
```

## Example Workflow

```bash
# 1. Run benchmarks before changes
pytest tests/benchmarks/ --benchmark-json=before.json

# 2. Make code changes

# 3. Run benchmarks after changes
pytest tests/benchmarks/ --benchmark-json=after.json

# 4. Compare results
pytest tests/benchmarks/ --benchmark-compare=before.json --benchmark-compare=after.json

# 5. Check for regressions
pytest tests/test_benchmark.py::TestMPCRegressionDetection -v
```

## Resources

- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [Performance Monitoring Guide](docs/DEVELOPMENT_GUIDE.md#performance-profiling)
- [CI Benchmark Reports](.github/workflows/ci.yml)
