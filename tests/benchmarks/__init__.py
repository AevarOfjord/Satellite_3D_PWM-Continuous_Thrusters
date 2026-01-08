"""
Performance benchmark suite for Satellite Control System.

This package contains comprehensive performance benchmarks for:
- MPC controller solve times
- Physics simulation steps
- State conversions
- Thruster management
- Memory usage
- Performance regression detection

Run benchmarks:
    pytest tests/benchmarks/ --benchmark-only

Compare benchmarks:
    pytest tests/benchmarks/ --benchmark-compare

Generate benchmark report:
    pytest tests/benchmarks/ --benchmark-json=benchmarks.json
"""
