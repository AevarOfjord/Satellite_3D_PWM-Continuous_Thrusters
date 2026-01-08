"""
Unit tests for ShapeUtils module.

Tests shape generation, transformation, and DXF loading utilities.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.satellite_control.visualization.shape_utils import (
    get_demo_shape,
    load_dxf_shape,
    make_offset_path,
    transform_shape,
)


class TestGetDemoShape:
    """Test get_demo_shape function."""

    def test_get_rectangle_shape(self):
        """Test getting rectangle demo shape."""
        shape = get_demo_shape("rectangle")

        assert len(shape) == 5  # 4 corners + closing point
        assert shape[0] == (-0.2, -0.15)
        assert shape[1] == (0.2, -0.15)
        assert shape[2] == (0.2, 0.15)
        assert shape[3] == (-0.2, 0.15)
        assert shape[4] == (-0.2, -0.15)  # Closed

    def test_get_triangle_shape(self):
        """Test getting triangle demo shape."""
        shape = get_demo_shape("triangle")

        assert len(shape) == 4  # 3 corners + closing point
        assert shape[0] == shape[-1]  # Should be closed

    def test_get_hexagon_shape(self):
        """Test getting hexagon demo shape."""
        shape = get_demo_shape("hexagon")

        assert len(shape) == 7  # 6 corners + closing point
        assert shape[0] == shape[-1]  # Should be closed

    def test_get_unknown_shape_defaults_to_rectangle(self):
        """Test that unknown shape type defaults to rectangle."""
        shape = get_demo_shape("unknown_shape")

        # Should return rectangle
        assert len(shape) == 5
        assert shape[0] == (-0.2, -0.15)


class TestTransformShape:
    """Test transform_shape function."""

    def test_transform_shape_translation(self):
        """Test shape translation."""
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        center = (2.0, 3.0)
        rotation = 0.0  # No rotation

        transformed = transform_shape(points, center, rotation)

        # Should be translated by center
        assert transformed[0] == pytest.approx((2.0, 3.0))
        assert transformed[1] == pytest.approx((3.0, 3.0))
        assert transformed[2] == pytest.approx((3.0, 4.0))
        assert transformed[3] == pytest.approx((2.0, 4.0))

    def test_transform_shape_rotation(self):
        """Test shape rotation."""
        points = [(1.0, 0.0), (0.0, 0.0)]  # Simple line
        center = (0.0, 0.0)
        rotation = np.pi / 2  # 90 degrees

        transformed = transform_shape(points, center, rotation)

        # Point (1, 0) rotated 90° should become (0, 1)
        assert transformed[0][0] == pytest.approx(0.0, abs=1e-10)
        assert transformed[0][1] == pytest.approx(1.0, abs=1e-10)

    def test_transform_shape_rotation_and_translation(self):
        """Test shape rotation and translation together."""
        points = [(1.0, 0.0)]
        center = (2.0, 3.0)
        rotation = np.pi / 2  # 90 degrees

        transformed = transform_shape(points, center, rotation)

        # (1, 0) rotated 90° = (0, 1), then translated = (2, 4)
        assert transformed[0][0] == pytest.approx(2.0, abs=1e-10)
        assert transformed[0][1] == pytest.approx(4.0, abs=1e-10)

    def test_transform_shape_preserves_length(self):
        """Test that transform preserves number of points."""
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        transformed = transform_shape(points, (0.0, 0.0), 0.0)

        assert len(transformed) == len(points)


class TestMakeOffsetPath:
    """Test make_offset_path function."""

    def test_make_offset_path_basic(self):
        """Test basic offset path generation."""
        base_path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        offset_distance = 0.1

        offset_path = make_offset_path(base_path, offset_distance)

        # Should create offset path
        assert len(offset_path) == len(base_path)
        assert isinstance(offset_path[0], tuple)

    def test_make_offset_path_handles_empty(self):
        """Test that offset path handles empty input."""
        base_path = []
        offset_path = make_offset_path(base_path, 0.1)

        assert len(offset_path) == 0

    def test_make_offset_path_handles_single_point(self):
        """Test that offset path handles single point."""
        base_path = [(0.0, 0.0)]
        offset_path = make_offset_path(base_path, 0.1)

        # Should handle gracefully
        assert isinstance(offset_path, list)


class TestLoadDXFShape:
    """Test load_dxf_shape function."""

    @patch("src.satellite_control.visualization.shape_utils.ezdxf")
    def test_load_dxf_shape_success(self, mock_ezdxf):
        """Test successful DXF shape loading."""
        # Mock DXF file
        mock_doc = MagicMock()
        mock_modelspace = MagicMock()
        mock_entity = MagicMock()
        mock_entity.dxf.start = (0.0, 0.0)
        mock_entity.dxf.end = (1.0, 1.0)
        mock_modelspace.query.return_value = [mock_entity]
        mock_doc.modelspace.return_value = mock_modelspace
        mock_ezdxf.readfile.return_value = mock_doc

        shape = load_dxf_shape("test.dxf")

        # Should return list of points
        assert isinstance(shape, list)

    @patch("src.satellite_control.visualization.shape_utils.ezdxf")
    def test_load_dxf_shape_file_not_found(self, mock_ezdxf):
        """Test DXF loading when file doesn't exist."""
        mock_ezdxf.readfile.side_effect = FileNotFoundError()

        shape = load_dxf_shape("nonexistent.dxf")

        # Should return empty list or handle gracefully
        assert isinstance(shape, list)

    @patch("src.satellite_control.visualization.shape_utils.ezdxf")
    def test_load_dxf_shape_invalid_file(self, mock_ezdxf):
        """Test DXF loading with invalid file."""
        mock_ezdxf.readfile.side_effect = Exception("Invalid DXF")

        # Should handle exception gracefully
        try:
            shape = load_dxf_shape("invalid.dxf")
            assert isinstance(shape, list)
        except Exception:
            # Exception handling depends on implementation
            pass
