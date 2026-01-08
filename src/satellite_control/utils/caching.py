"""
Caching Utilities for Satellite Control System

Provides caching mechanisms for expensive computations.
Supports config-based caching and LRU caching.

Usage:
    from src.satellite_control.utils.caching import cache_by_config, cached
    
    @cached(maxsize=128)
    def expensive_computation(x, y):
        return complex_calculation(x, y)
    
    @cache_by_config
    def build_matrices(config):
        return build_expensive_matrices(config)
"""

import hashlib
import json
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def cache_key_from_config(config: Any) -> str:
    """
    Generate cache key from configuration object.
    
    Works with Pydantic models, dictionaries, or any object with
    a serializable representation.
    
    Args:
        config: Configuration object (Pydantic model, dict, etc.)
        
    Returns:
        MD5 hash string of configuration
    """
    if hasattr(config, "model_dump_json"):
        # Pydantic model
        config_str = config.model_dump_json()
    elif hasattr(config, "model_dump"):
        # Pydantic model (older API)
        config_str = json.dumps(config.model_dump(), sort_keys=True)
    elif isinstance(config, dict):
        # Dictionary
        config_str = json.dumps(config, sort_keys=True)
    else:
        # Try to convert to dict
        try:
            config_dict = dict(config) if hasattr(config, "__dict__") else {}
            config_str = json.dumps(config_dict, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback to string representation
            config_str = str(config)
    
    return hashlib.md5(config_str.encode()).hexdigest()


def cache_by_config(func: Optional[F] = None, maxsize: int = 10) -> F:
    """
    Decorator to cache function results based on configuration hash.
    
    The first argument is assumed to be a configuration object.
    Results are cached based on the config's hash.
    
    Args:
        func: Function to cache (if None, returns decorator)
        maxsize: Maximum cache size
        
    Returns:
        Decorated function with caching
        
    Example:
        @cache_by_config(maxsize=5)
        def build_mpc_matrices(config: AppConfig, horizon: int):
            # Expensive matrix building
            return P, q, A, l, u
    """
    cache: Dict[str, Any] = {}

    def decorator(f: F) -> F:
        @wraps(f)
        def wrapper(config: Any, *args: Any, **kwargs: Any) -> Any:
            # Generate cache key from config
            config_key = cache_key_from_config(config)
            
            # Create full cache key including args and kwargs
            args_key = str(args) + str(sorted(kwargs.items()))
            full_key = f"{config_key}:{hashlib.md5(args_key.encode()).hexdigest()}"
            
            # Check cache
            if full_key in cache:
                return cache[full_key]
            
            # Compute result
            result = f(config, *args, **kwargs)
            
            # Store in cache (with size limit)
            if len(cache) >= maxsize:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[full_key] = result
            return result

        # Add cache management methods
        wrapper.cache_clear = cache.clear  # type: ignore
        wrapper.cache_info = lambda: {  # type: ignore
            "size": len(cache),
            "maxsize": maxsize,
            "keys": list(cache.keys())[:5],  # Show first 5 keys
        }

        return wrapper  # type: ignore

    if func is None:
        return decorator  # type: ignore
    return decorator(func)  # type: ignore


def cached(maxsize: int = 128) -> Callable[[F], F]:
    """
    Simple LRU cache decorator with configurable size.
    
    Wrapper around functools.lru_cache with better defaults.
    
    Args:
        maxsize: Maximum cache size (None for unlimited)
        
    Returns:
        Decorated function with LRU caching
        
    Example:
        @cached(maxsize=256)
        def expensive_computation(x: float, y: float) -> float:
            return complex_calculation(x, y)
    """
    return lru_cache(maxsize=maxsize)


def cache_clear_all() -> None:
    """
    Clear all function caches.
    
    Useful for testing or when configuration changes significantly.
    """
    # This would need to track all cached functions
    # For now, individual caches can be cleared via .cache_clear()
    pass


class CacheStats:
    """Statistics for a cache."""

    def __init__(self, name: str):
        """Initialize cache statistics."""
        self.name = name
        self.hits = 0
        self.misses = 0
        self.size = 0
        self.maxsize = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        """String representation of cache stats."""
        return (
            f"{self.name}: hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.2%}, size={self.size}/{self.maxsize}"
        )


def cache_with_stats(maxsize: int = 128) -> Callable[[F], F]:
    """
    LRU cache decorator with statistics tracking.
    
    Args:
        maxsize: Maximum cache size
        
    Returns:
        Decorated function with caching and stats
    """
    stats = CacheStats("cache")

    def decorator(func: F) -> F:
        cached_func = lru_cache(maxsize=maxsize)(func)
        stats.maxsize = maxsize

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_info = cached_func.cache_info()
            stats.size = cache_info.currsize
            
            if cache_info.hits > stats.hits:
                stats.hits = cache_info.hits
            if cache_info.misses > stats.misses:
                stats.misses = cache_info.misses
            
            return cached_func(*args, **kwargs)

        wrapper.cache_stats = stats  # type: ignore
        wrapper.cache_clear = cached_func.cache_clear  # type: ignore
        wrapper.cache_info = cached_func.cache_info  # type: ignore

        return wrapper  # type: ignore

    return decorator
