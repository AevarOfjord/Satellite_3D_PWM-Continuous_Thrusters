"""
Mission Plugins Package

V4.0.0: Phase 2 - Mission Plugin System

Auto-registers built-in mission plugins.
"""

from src.satellite_control.mission.plugin import register_plugin
from src.satellite_control.mission.plugins.waypoint_plugin import WaypointMissionPlugin
from src.satellite_control.mission.plugins.shape_following_plugin import ShapeFollowingMissionPlugin

# Auto-register built-in plugins
_waypoint_plugin = WaypointMissionPlugin()
register_plugin(_waypoint_plugin)

_shape_following_plugin = ShapeFollowingMissionPlugin()
register_plugin(_shape_following_plugin)

__all__ = ["WaypointMissionPlugin", "ShapeFollowingMissionPlugin"]
