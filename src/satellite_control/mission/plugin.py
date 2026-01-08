"""
Mission Plugin System for Satellite Control

Provides a plugin architecture for extensible mission types.
Allows users to create custom mission types and load them dynamically.

V4.0.0: Phase 2 - Mission Plugin System
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np
from pathlib import Path
import importlib.util
import logging

from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.config.mission_state import MissionState

logger = logging.getLogger(__name__)


class MissionPlugin(ABC):
    """
    Abstract base class for mission plugins.
    
    All mission types must implement this interface to be compatible
    with the plugin system.
    
    Example:
        class MyCustomMission(MissionPlugin):
            def get_name(self) -> str:
                return "my_custom_mission"
            
            def configure(self, config: SimulationConfig) -> MissionState:
                # Configure mission parameters
                return mission_state
            
            def get_target_state(self, current_state: np.ndarray, time: float) -> np.ndarray:
                # Calculate target state
                return target_state
            
            def is_complete(self, current_state: np.ndarray, time: float) -> bool:
                # Check if mission is complete
                return False
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the unique name of this mission plugin.
        
        Returns:
            Unique identifier for this mission type (e.g., "waypoint", "shape_following")
        """
        pass
    
    @abstractmethod
    def get_display_name(self) -> str:
        """
        Get the human-readable display name.
        
        Returns:
            Display name (e.g., "Waypoint Navigation", "Shape Following")
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of this mission type.
        
        Returns:
            Description string
        """
        pass
    
    @abstractmethod
    def configure(self, config: SimulationConfig) -> MissionState:
        """
        Configure mission parameters from a SimulationConfig.
        
        This method should interact with the user (if needed) to gather
        mission-specific parameters and return a configured MissionState.
        
        Args:
            config: Base simulation configuration
            
        Returns:
            Configured MissionState for this mission type
        """
        pass
    
    @abstractmethod
    def get_target_state(
        self,
        current_state: np.ndarray,
        time: float,
        mission_state: MissionState,
    ) -> np.ndarray:
        """
        Get the target state for the current time.
        
        Args:
            current_state: Current satellite state [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            time: Current simulation time
            mission_state: Current mission state
            
        Returns:
            Target state vector [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        pass
    
    @abstractmethod
    def is_complete(
        self,
        current_state: np.ndarray,
        time: float,
        mission_state: MissionState,
    ) -> bool:
        """
        Check if the mission is complete.
        
        Args:
            current_state: Current satellite state
            time: Current simulation time
            mission_state: Current mission state
            
        Returns:
            True if mission is complete, False otherwise
        """
        pass
    
    def get_version(self) -> str:
        """
        Get the version of this plugin.
        
        Returns:
            Version string (default: "1.0.0")
        """
        return "1.0.0"
    
    def get_author(self) -> Optional[str]:
        """
        Get the author of this plugin.
        
        Returns:
            Author name or None
        """
        return None
    
    def validate_config(self, config: SimulationConfig) -> bool:
        """
        Validate that the given config is compatible with this mission type.
        
        Args:
            config: Simulation configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True
    
    def get_required_parameters(self) -> List[str]:
        """
        Get a list of required MissionState parameters for this mission type.
        
        Returns:
            List of parameter names (e.g., ["waypoint_targets", "enable_waypoint_mode"])
        """
        return []


class MissionPluginRegistry:
    """
    Registry for mission plugins.
    
    Handles plugin discovery, loading, and management.
    """
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, MissionPlugin] = {}
        self._plugin_paths: Dict[str, Path] = {}
        self._search_paths: List[Path] = []
        
        # Add default search paths
        self.add_search_path(Path(__file__).parent / "plugins")
        self.add_search_path(Path.home() / ".satellite_control" / "plugins")
    
    def add_search_path(self, path: Path) -> None:
        """
        Add a directory to search for plugins.
        
        Args:
            path: Directory path to search
        """
        if path.exists() and path.is_dir():
            self._search_paths.append(path)
            logger.debug(f"Added plugin search path: {path}")
        else:
            logger.warning(f"Plugin search path does not exist: {path}")
    
    def register_plugin(self, plugin: MissionPlugin, name: Optional[str] = None) -> None:
        """
        Register a plugin instance.
        
        Args:
            plugin: Plugin instance to register
            name: Optional name override (defaults to plugin.get_name())
        """
        plugin_name = name or plugin.get_name()
        if plugin_name in self._plugins:
            logger.warning(f"Plugin '{plugin_name}' already registered, overwriting")
        
        self._plugins[plugin_name] = plugin
        logger.info(f"Registered plugin: {plugin_name} ({plugin.get_display_name()})")
    
    def load_plugin_from_file(self, file_path: Path) -> Optional[MissionPlugin]:
        """
        Load a plugin from a Python file.
        
        Args:
            file_path: Path to Python file containing plugin class
            
        Returns:
            Plugin instance or None if loading failed
        """
        try:
            spec = importlib.util.spec_from_file_location(
                f"mission_plugin_{file_path.stem}", file_path
            )
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create spec for {file_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find MissionPlugin subclass in module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, MissionPlugin)
                    and attr is not MissionPlugin
                ):
                    plugin = attr()
                    self.register_plugin(plugin)
                    self._plugin_paths[plugin.get_name()] = file_path
                    return plugin
            
            logger.warning(f"No MissionPlugin subclass found in {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {file_path}: {e}")
            return None
    
    def discover_plugins(self) -> int:
        """
        Discover and load plugins from all search paths.
        
        Returns:
            Number of plugins discovered
        """
        count = 0
        for search_path in self._search_paths:
            if not search_path.exists():
                continue
            
            # Look for Python files
            for file_path in search_path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue  # Skip private modules
                
                plugin = self.load_plugin_from_file(file_path)
                if plugin is not None:
                    count += 1
        
        logger.info(f"Discovered {count} plugins")
        return count
    
    def get_plugin(self, name: str) -> Optional[MissionPlugin]:
        """
        Get a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """
        Get a list of all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())
    
    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Dictionary with plugin information or None
        """
        plugin = self.get_plugin(name)
        if plugin is None:
            return None
        
        return {
            "name": plugin.get_name(),
            "display_name": plugin.get_display_name(),
            "description": plugin.get_description(),
            "version": plugin.get_version(),
            "author": plugin.get_author(),
            "required_parameters": plugin.get_required_parameters(),
            "file_path": str(self._plugin_paths.get(name, "built-in")),
        }


# Global registry instance
_registry = MissionPluginRegistry()


def get_registry() -> MissionPluginRegistry:
    """Get the global plugin registry."""
    return _registry


def register_plugin(plugin: MissionPlugin, name: Optional[str] = None) -> None:
    """Register a plugin in the global registry."""
    _registry.register_plugin(plugin, name)


def discover_plugins() -> int:
    """Discover plugins in the global registry."""
    return _registry.discover_plugins()


def get_plugin(name: str) -> Optional[MissionPlugin]:
    """Get a plugin from the global registry."""
    return _registry.get_plugin(name)
