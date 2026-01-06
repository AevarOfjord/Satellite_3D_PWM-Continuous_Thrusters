# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New documentation reorganization (January 2026)

## [2.0.0] - 2026-01-06

### Added

- **QUICKSTART.md** - 5-minute getting started guide for new users
- **FAQ.md** - Comprehensive FAQ covering installation, usage, performance, and troubleshooting
- **TESTING.md** - Unified testing guide combining simulation testing and pytest testing
- **SIMULATION.md** - Consolidated guide covering mission types, simulation loop, and configuration

### Changed

- **VISUALIZATION.md** - Enhanced with CSV format reference as appendix
- Documentation structure reorganized from 18 to 11 markdown files
- Improved cross-references and navigation across documentation

### Removed

- **MUJOCO_LOG.TXT** - Log file removed from documentation directory
- **CODE_OF_CONDUCT.md** - Removed as not applicable for portfolio project
- **SECURITY.md** - Removed as not applicable for simulation project
- **SIMULATION_TESTING_GUIDE.md** - Consolidated into TESTING.md
- **TESTING_GUIDE.md** - Consolidated into TESTING.md
- **MISSION_ARCHITECTURE.md** - Consolidated into SIMULATION.md
- **SIMULATION_LOOP.md** - Consolidated into SIMULATION.md
- **CSV_FORMAT.md** - Moved to VISUALIZATION.md appendix

### Fixed

- Added .gitignore rules to prevent log files in docs/
- Updated cross-references to point to consolidated documentation

## [1.0.0] - 2025-12-10

### Added

- Initial satellite control system implementation
- MuJoCo physics integration (200 Hz)
- OSQP-based Model Predictive Control (16.67 Hz)
- Waypoint navigation mission type
- Shape following mission type
- Real-time terminal dashboard
- Comprehensive data logging (CSV format)
- Automated visualization generation
- Complete test suite (pytest)

### Features

- 8-thruster PWM control with MPC optimization
- Prediction horizon: 50 steps (3.0 seconds)
- Position accuracy: <5cm
- MPC solve times: 1-2ms typical
- Support for custom DXF shape import

---

## Version History Summary

| Version | Date       | Description                                     |
| ------- | ---------- | ----------------------------------------------- |
| 2.0.0   | 2026-01-06 | Documentation reorganization and enhancement    |
| 1.0.0   | 2025-12-10 | Initial release with MPC and MuJoCo integration |

---

## Contributing

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for contribution guidelines.

## Questions?

Check [FAQ.md](FAQ.md) or [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.
