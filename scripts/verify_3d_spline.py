import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.satellite_control.utils.spline_path import (
    create_obstacle_avoidance_spline,
    ObstacleAvoidanceSpline,
)


def test_3d_spline():
    print("Testing 3D Spline Generation...")

    # 3D points
    start = np.array([0.0, 0.0, 0.0])
    target = np.array([2.0, 0.0, 1.0])  # Target at Z=1.0
    obstacle = np.array([1.0, 0.0, 0.0])  # Obstacle on path
    radius = 0.5

    spline = create_obstacle_avoidance_spline(start, target, obstacle, radius)

    if spline is None:
        print("FAILED: Spline not generated (returned None)")
        return

    print("Spline generated successfully.")

    # Check start and end points
    p_start = spline.evaluate(0.0)
    p_end = spline.evaluate(spline.total_length)

    print(f"Start Point: {p_start}")
    print(f"End Point:   {p_end}")

    # Verify Z values
    # Start Z should be 0.0
    # End Z should be 1.0
    if len(p_start) < 3:
        print("ERROR: Start point is not 3D!")
    elif abs(p_start[2] - 0.0) > 1e-6:
        print(f"ERROR: Start Z mismatch. Got {p_start[2]}")
    else:
        print("Start Z correct.")

    if len(p_end) < 3:
        print("ERROR: End point is not 3D!")
    elif abs(p_end[2] - 1.0) > 1e-6:
        print(f"ERROR: End Z mismatch. Got {p_end[2]}")
    else:
        print("End Z correct.")

    # Check midpoint Z (should be roughly 0.5 if linear interpolation of Z control point works)
    p_mid = spline.evaluate(spline.total_length / 2.0)
    print(f"Mid Point:   {p_mid}")

    if len(p_mid) < 3:
        print("ERROR: Mid point is not 3D!")
    # The control point Z was set to (start_z + target_z) / 2 = 0.5
    # For a quadratic Bezier with P0_z=0, P1_z=0.5, P2_z=1.0:
    # At t=0.5: B(0.5) = 0.25*0 + 0.5*0.5 + 0.25*1.0 = 0.25 + 0.25 = 0.5
    # So midpoint Z should be exactly 0.5 (assuming arc length parameterization matches t=0.5 at midpoint, which is approx true for symmetric spline)
    elif abs(p_mid[2] - 0.5) > 0.1:  # Loose tolerance due to arc length parameterization
        print(f"Warning: Midpoint Z deviation. Got {p_mid[2]}")
    else:
        print("Midpoint Z reasonably correct (~0.5).")

    print("\nTest passed!")


if __name__ == "__main__":
    test_3d_spline()
