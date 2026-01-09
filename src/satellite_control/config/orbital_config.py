"""
Orbital Configuration Module

Defines orbital parameters for LEO inspection mission scenarios.
Uses Hill-Clohessy-Wiltshire (CW) relative motion model.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Physical constants
MU_EARTH = 3.986004418e14  # Earth gravitational parameter [m³/s²]
EARTH_RADIUS = 6.371e6  # Earth mean radius [m]


@dataclass(frozen=True)
class OrbitalConfig:
    """
    Orbital parameters for the target satellite's reference orbit.

    The inspector satellite's position is defined relative to this
    target satellite using Hill's frame (LVLH - Local Vertical Local Horizontal):
    - X: Radial (away from Earth)
    - Y: Along-track (velocity direction)
    - Z: Cross-track (normal to orbital plane)

    Attributes:
        altitude: Orbital altitude above Earth surface [m]
        mu: Gravitational parameter [m³/s²]
        earth_radius: Earth radius [m]
        inclination: Orbital inclination [rad] (not used in CW, for reference)
    """

    altitude: float = 400_000  # 400 km LEO (ISS altitude)
    mu: float = MU_EARTH
    earth_radius: float = EARTH_RADIUS
    inclination: float = np.deg2rad(51.6)  # ISS inclination

    @property
    def orbital_radius(self) -> float:
        """Semi-major axis (circular orbit radius) [m]."""
        return self.earth_radius + self.altitude

    @property
    def mean_motion(self) -> float:
        """Orbital mean motion n = √(μ/a³) [rad/s]."""
        return np.sqrt(self.mu / self.orbital_radius**3)

    @property
    def orbital_period(self) -> float:
        """Orbital period T = 2π/n [s]."""
        return 2 * np.pi / self.mean_motion

    @property
    def orbital_velocity(self) -> float:
        """Circular orbital velocity [m/s]."""
        return np.sqrt(self.mu / self.orbital_radius)

    def print_params(self):
        """Print orbital parameters summary."""
        print("Orbital Parameters:")
        print(f"  Altitude: {self.altitude/1000:.1f} km")
        print(f"  Orbital radius: {self.orbital_radius/1000:.1f} km")
        print(f"  Mean motion: {self.mean_motion:.6f} rad/s")
        print(f"  Orbital period: {self.orbital_period/60:.1f} min")
        print(f"  Orbital velocity: {self.orbital_velocity/1000:.2f} km/s")


@dataclass
class InspectionScenario:
    """
    Defines an inspection mission scenario.

    Attributes:
        orbital_config: Orbital parameters of target
        initial_offset: Initial position offset from target [m] in Hill frame
        target_offset: Target position for inspection [m] in Hill frame
        approach_speed: Maximum approach speed [m/s]
        keep_out_radius: Minimum distance from target [m]
    """

    orbital_config: OrbitalConfig
    initial_offset: np.ndarray  # [x, y, z] in Hill frame [m]
    target_offset: np.ndarray  # [x, y, z] in Hill frame [m]
    approach_speed: float = 0.05  # Max 5 cm/s approach
    keep_out_radius: float = 1.0  # 1m minimum distance

    @classmethod
    def create_default(cls) -> "InspectionScenario":
        """Create default inspection scenario: approach from 10m to 3m."""
        return cls(
            orbital_config=OrbitalConfig(),
            initial_offset=np.array([10.0, 0.0, 0.0]),  # 10m radial
            target_offset=np.array([3.0, 0.0, 0.0]),  # 3m final offset
        )

    @classmethod
    def create_v_bar_approach(cls) -> "InspectionScenario":
        """V-bar approach: along velocity vector."""
        return cls(
            orbital_config=OrbitalConfig(),
            initial_offset=np.array([0.0, -20.0, 0.0]),  # 20m behind
            target_offset=np.array([0.0, -3.0, 0.0]),  # 3m behind
        )

    @classmethod
    def create_r_bar_approach(cls) -> "InspectionScenario":
        """R-bar approach: along radial (nadir) vector."""
        return cls(
            orbital_config=OrbitalConfig(),
            initial_offset=np.array([-10.0, 0.0, 0.0]),  # 10m below
            target_offset=np.array([-3.0, 0.0, 0.0]),  # 3m below
        )


def get_default_orbital_config() -> OrbitalConfig:
    """Get default LEO orbital configuration."""
    return OrbitalConfig()


# Pre-computed for common altitudes
LEO_400KM = OrbitalConfig(altitude=400_000)
LEO_550KM = OrbitalConfig(altitude=550_000)  # Starlink
GEO = OrbitalConfig(altitude=35_786_000)  # Geostationary
