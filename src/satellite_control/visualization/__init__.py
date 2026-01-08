"""
Visualization Module

Provides visualization tools for satellite simulation data.

Public API:
- UnifiedVisualizationGenerator: Main visualization class
- PlotStyle: Styling constants for consistent plots
- LinearizedVisualizationGenerator: Legacy compatibility alias
"""

from src.satellite_control.visualization.simulation_visualization import (
    create_simulation_visualizer,
)
from src.satellite_control.visualization.unified_visualizer import (
    LinearizedVisualizationGenerator,
    PlotStyle,
    UnifiedVisualizationGenerator,
)
from src.satellite_control.visualization.shape_utils import (
    get_demo_shape,
    load_dxf_shape,
    make_offset_path,
    transform_shape,
)

__all__ = [
    "UnifiedVisualizationGenerator",
    "LinearizedVisualizationGenerator",
    "PlotStyle",
    "create_simulation_visualizer",
]
