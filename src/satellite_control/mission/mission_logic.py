"""
Mission Logic Module

Handles the business logic for mission planning, coordinate validation,
and shape generation. Pure Python, no user interaction.
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MissionLogic:
    """
    Pure logic handler for satellite missions.
    Decoupled from user input / CLI.
    """

    def __init__(self):
        # Default tolerances could be injected or loaded from config
        pass

    def calculate_path_length(self, points: List[Tuple[float, float]]) -> float:
        """Calculate total length of path."""
        length = 0.0
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])
            length += float(np.linalg.norm(p2 - p1))
        return float(length)

    def validate_coordinates(self, input_str: str) -> Optional[Tuple[float, float]]:
        """
        Parse and validate coordinate string "x, y".
        Returns tuple (x, y) if valid, None otherwise.
        """
        try:
            parts = input_str.replace("(", "").replace(")", "").split(",")
            if len(parts) == 2:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                return (x, y)
        except (ValueError, IndexError):
            pass
        return None

    def validate_angle(self, input_str: str) -> Optional[float]:
        """
        Parse and validate angle string (degrees).
        Returns angle in radians if valid, None otherwise.
        """
        try:
            val = float(input_str.strip())
            return math.radians(val)
        except ValueError:
            return None

    def generate_demo_shape(self, shape_type: str) -> List[Tuple[float, float]]:
        """
        Get predefined demo shape points.

        Args:
            shape_type: "circle", "rectangle", "triangle", "hexagon"
        """
        points = []
        if shape_type == "circle":
            # 1m radius circle
            for i in range(37):  # 36 segments + closure
                angle = math.radians(i * 10)
                points.append((math.cos(angle), math.sin(angle)))

        elif shape_type == "rectangle":
            # 2x1 rectangle
            points = [
                (-1.0, -0.5),
                (1.0, -0.5),
                (1.0, 0.5),
                (-1.0, 0.5),
                (-1.0, -0.5),
            ]

        elif shape_type == "triangle":
            # Equilateral triangle
            points = [
                (0.0, 1.0),
                (math.cos(math.radians(210)), math.sin(math.radians(210))),
                (math.cos(math.radians(330)), math.sin(math.radians(330))),
                (0.0, 1.0),
            ]

        elif shape_type == "hexagon":
            for i in range(7):
                angle = math.radians(i * 60)
                points.append((math.cos(angle), math.sin(angle)))

        return points

    def transform_shape(
        self,
        points: List[Tuple[float, float]],
        center: Tuple[float, float],
        rotation: float,
    ) -> List[Tuple[float, float]]:
        """
        Transform shape points to specified center and rotation.
        """
        if not points:
            return []

        # Center centering (assume shape centered at 0,0 or calculate
        # centroid?) The original treated points as relative to 0,0.
        # unless they were DXF loaded. For generated shapes, they are around
        # 0,0.

        # Original implementation calculates centroid to re-center?
        # Let's trust the input points are defining the shape geometry.

        # However, to typically "move" a shape to a target, we often want
        # the centroid at target. But if we just rotate and translate,
        # we are assuming input points are local.
        # But if we just rotate and translate, we are assuming input points are
        # local.

        cos_theta = math.cos(rotation)
        sin_theta = math.sin(rotation)

        transformed = []
        for x, y in points:
            # Rotate
            rot_x = x * cos_theta - y * sin_theta
            rot_y = x * sin_theta + y * cos_theta
            # Translate
            final_x = rot_x + center[0]
            final_y = rot_y + center[1]
            transformed.append((final_x, final_y))

        return transformed

    def upscale_shape(
        self, points: List[Tuple[float, float]], offset_distance: float
    ) -> List[Tuple[float, float]]:
        """Create upscaled path at fixed offset from shape using Shapely."""
        if len(points) < 3:
            return points

        try:
            import importlib
            import importlib.util

            # Check for shapely
            if importlib.util.find_spec("shapely") is None:
                # Fallback to centroid scaling if shapely missing (better than
                # the edge shift bug)
                points_arr = np.array(points)
                centroid = np.mean(points_arr, axis=0)
                # Crude approximation: Scale relative to centroid
                # Requires estimating 'radius'
                radii = np.linalg.norm(points_arr - centroid, axis=1)
                avg_radius = np.mean(radii)
                if avg_radius < 1e-6:
                    avg_radius = 1.0
                scale_factor = (avg_radius + offset_distance) / avg_radius

                scaled = []
                for p in points:
                    vec = np.array(p) - centroid
                    new_p = centroid + vec * scale_factor
                    scaled.append(tuple(new_p))
                return scaled

            # Use Shapely for correct offsetting
            from shapely.geometry import Polygon

            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)

            # Buffer
            # join_style=1 (round) is standard, resolution controls smoothness
            # of corners
            buffered = poly.buffer(offset_distance, join_style=1, resolution=16)

            if buffered.is_empty:
                return points

            # Extract exterior coords
            exterior = list(buffered.exterior.coords)
            return [(float(x), float(y)) for x, y in exterior]

        except Exception as e:
            logger.warning(f" Shapely offset failed in upscale_shape: {e}")
            return points

    def load_dxf_shape(self, file_path: str) -> List[Tuple[float, float]]:
        """
        Load shape points from DXF using the same pipeline as DXF_Viewer.

        Args:
            file_path: Path to DXF file

        Returns:
            List of (x, y) points in meters
        """
        try:
            import ezdxf
        except ImportError:
            raise ImportError("ezdxf library not installed")

        try:
            # Ensure sys.path is set up by caller or
            # try relative import if structured
            # For now, rely on module availability in environment
            from dxf_viewer import (
                extract_boundary_polygon,
                sanitize_boundary,
                units_code_to_name_and_scale,
            )
        except ImportError as e:
            raise ImportError(f"dxf_viewer utilities unavailable: {e}")

        # Read DXF and determine units
        try:
            doc = ezdxf.readfile(file_path)
            msp = doc.modelspace()
            insunits = int(doc.header.get("$INSUNITS", 0))
            units_name, to_m = units_code_to_name_and_scale(insunits)

            # Extract and sanitize boundary in native units, then scale to
            # meters
            boundary = extract_boundary_polygon(msp)
            boundary = sanitize_boundary(boundary, to_m)
            boundary_m = (
                [(float(x) * to_m, float(y) * to_m) for (x, y) in boundary] if boundary else []
            )

            if not boundary_m:
                raise ValueError("No usable DXF boundary could be constructed.")

            return boundary_m

        except Exception as e:
            raise ValueError(f"DXF processing failed: {e}")
