# Tests

This directory is for unit tests and integration tests for the Satellite Thruster Control System.

## Running Tests

Once tests are added, you can run them with:

```bash
python3 -m pytest tests/
```

## Test Structure

- `test_mpc_controller.py` - Tests for MPC controller
- `test_config.py` - Tests for configuration validation
- `test_model.py` - Tests for physical model
- `test_mission_state_manager.py` - Tests for mission logic
- `test_integration_basic.py` - Basic integration tests
- `test_integration_missions.py` - Mission integration tests
- Additional test files as needed

## Testing Tools

For manual testing and verification:
- Use `simulation.py` for testing different mission scenarios
- Use `comparison.py` for comparing simulation vs real test data
