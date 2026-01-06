"""
Profiling Utilities for Satellite Control System

Tools for measuring and optimizing performance of critical code paths.
Includes histogram support and configurable warning thresholds.
"""

import cProfile
import functools
import io
import logging
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Default histogram bucket boundaries in milliseconds
DEFAULT_BUCKETS_MS: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


@dataclass
class TimingStats:
    """Statistics for a profiled function with histogram support."""

    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    # Histogram buckets (in seconds)
    bucket_boundaries: Tuple[float, ...] = field(
        default_factory=lambda: tuple(b / 1000 for b in DEFAULT_BUCKETS_MS)
    )
    bucket_counts: List[int] = field(default_factory=list)

    # Warning threshold (in seconds) - None means no warning
    warning_threshold: Optional[float] = None
    warning_count: int = 0

    def __post_init__(self):
        """Initialize histogram buckets."""
        if not self.bucket_counts:
            # +1 for the overflow bucket (>max boundary)
            self.bucket_counts = [0] * (len(self.bucket_boundaries) + 1)

    @property
    def avg_time(self) -> float:
        """Average time per call."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_time_ms(self) -> float:
        """Average time in milliseconds."""
        return self.avg_time * 1000

    @property
    def p50_time(self) -> float:
        """Median (50th percentile) time."""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = len(sorted_times) // 2
        return sorted_times[idx]

    @property
    def p95_time(self) -> float:
        """95th percentile time."""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_time(self) -> float:
        """99th percentile time."""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def record(self, elapsed: float) -> bool:
        """
        Record a timing measurement.

        Returns:
            True if warning threshold was exceeded
        """
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)

        # Update histogram
        bucket_idx = len(self.bucket_boundaries)  # Default to overflow
        for i, boundary in enumerate(self.bucket_boundaries):
            if elapsed <= boundary:
                bucket_idx = i
                break
        self.bucket_counts[bucket_idx] += 1

        # Check warning threshold
        exceeded_threshold = False
        if self.warning_threshold is not None and elapsed > self.warning_threshold:
            self.warning_count += 1
            exceeded_threshold = True

        return exceeded_threshold

    def get_histogram_string(self) -> str:
        """Get a string representation of the histogram."""
        if self.call_count == 0:
            return "No data"

        lines = []
        prev_boundary = 0.0
        for i, count in enumerate(self.bucket_counts):
            pct = (count / self.call_count) * 100 if self.call_count > 0 else 0
            bar = "█" * int(pct / 2)  # Scale to fit

            if i < len(self.bucket_boundaries):
                boundary_ms = self.bucket_boundaries[i] * 1000
                lines.append(
                    f"  {prev_boundary:.1f}-{boundary_ms:.1f}ms: " f"{bar} ({count}, {pct:.1f}%)"
                )
                prev_boundary = boundary_ms
            else:
                lines.append(f"  >{prev_boundary:.1f}ms: {bar} ({count}, {pct:.1f}%)")

        return "\n".join(lines)

    def __str__(self) -> str:
        if self.call_count == 0:
            return f"{self.name}: no calls"

        base = (
            f"{self.name}: {self.call_count} calls, "
            f"avg={self.avg_time_ms:.2f}ms, "
            f"p50={self.p50_time*1000:.2f}ms, "
            f"p95={self.p95_time*1000:.2f}ms, "
            f"min={self.min_time*1000:.2f}ms, "
            f"max={self.max_time*1000:.2f}ms"
        )

        if self.warning_count > 0:
            base += f" [⚠️ {self.warning_count} warnings]"

        return base


class PerformanceProfiler:
    """
    Lightweight profiler for tracking function execution times.

    Usage:
        profiler = PerformanceProfiler()

        @profiler.profile
        def my_function():
            ...

        # Or as context manager:
        with profiler.measure("operation_name"):
            ...

        profiler.print_stats()
    """

    def __init__(
        self,
        warning_threshold: Optional[float] = None,
        bucket_boundaries_ms: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """
        Initialize profiler.

        Args:
            warning_threshold: Default warning threshold in seconds
            bucket_boundaries_ms: Histogram bucket boundaries in milliseconds
        """
        self.stats: Dict[str, TimingStats] = {}
        self.enabled = True
        self.default_warning_threshold = warning_threshold
        self.bucket_boundaries = (
            tuple(b / 1000 for b in bucket_boundaries_ms)
            if bucket_boundaries_ms
            else tuple(b / 1000 for b in DEFAULT_BUCKETS_MS)
        )

    def profile(
        self,
        func: Optional[Callable] = None,
        *,
        warning_threshold: Optional[float] = None,
    ) -> Callable:
        """
        Decorator to profile a function.

        Args:
            func: Function to profile
            warning_threshold: Optional warning threshold in seconds
        """

        def decorator(f: Callable) -> Callable:
            name = f.__qualname__

            # Initialize stats with thresholds
            if name not in self.stats:
                self.stats[name] = TimingStats(
                    name=name,
                    bucket_boundaries=self.bucket_boundaries,
                    warning_threshold=warning_threshold or self.default_warning_threshold,
                )

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return f(*args, **kwargs)

                start = time.perf_counter()
                try:
                    return f(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    exceeded = self._record(name, elapsed)
                    warn_thresh = self.stats[name].warning_threshold
                    if exceeded and warn_thresh is not None:
                        threshold_ms = warn_thresh * 1000
                        logger.warning(
                            f"⚠️ {name} exceeded threshold: "
                            f"{elapsed*1000:.2f}ms > {threshold_ms:.2f}ms"
                        )

            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    @contextmanager
    def measure(self, name: str, warning_threshold: Optional[float] = None):
        """Context manager for profiling a code block."""
        if not self.enabled:
            yield
            return

        # Initialize stats if needed
        if name not in self.stats:
            self.stats[name] = TimingStats(
                name=name,
                bucket_boundaries=self.bucket_boundaries,
                warning_threshold=warning_threshold or self.default_warning_threshold,
            )

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            exceeded = self._record(name, elapsed)
            warn_thresh = self.stats[name].warning_threshold
            if exceeded and warn_thresh is not None:
                threshold_ms = warn_thresh * 1000
                logger.warning(
                    f"⚠️ {name} exceeded threshold: " f"{elapsed*1000:.2f}ms > {threshold_ms:.2f}ms"
                )

    def _record(self, name: str, elapsed: float) -> bool:
        """Record a timing measurement. Returns True if threshold exceeded."""
        if name not in self.stats:
            self.stats[name] = TimingStats(
                name=name,
                bucket_boundaries=self.bucket_boundaries,
                warning_threshold=self.default_warning_threshold,
            )
        return self.stats[name].record(elapsed)

    def set_warning_threshold(self, name: str, threshold: float) -> None:
        """Set warning threshold for a specific metric."""
        if name in self.stats:
            self.stats[name].warning_threshold = threshold
        else:
            self.stats[name] = TimingStats(
                name=name,
                bucket_boundaries=self.bucket_boundaries,
                warning_threshold=threshold,
            )

    def check_threshold(self, name: str) -> Tuple[bool, int]:
        """
        Check if a metric has exceeded its threshold.

        Returns:
            Tuple of (has_warnings, warning_count)
        """
        if name not in self.stats:
            return False, 0
        stats = self.stats[name]
        return stats.warning_count > 0, stats.warning_count

    def print_stats(self, top_n: Optional[int] = None, show_histograms: bool = False) -> None:
        """Print profiling statistics sorted by total time."""
        sorted_stats = sorted(self.stats.values(), key=lambda s: s.total_time, reverse=True)

        if top_n:
            sorted_stats = sorted_stats[:top_n]

        logger.info("=" * 60)
        logger.info("PERFORMANCE PROFILE")
        logger.info("=" * 60)
        for stat in sorted_stats:
            logger.info(str(stat))
            if show_histograms:
                logger.info(stat.get_histogram_string())
        logger.info("=" * 60)

    def reset(self) -> None:
        """Reset all statistics."""
        self.stats.clear()

    def get_stats(self, name: str) -> Optional[TimingStats]:
        """Get stats for a specific function."""
        return self.stats.get(name)


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile a function with cProfile.

    Prints top 10 cumulative time entries after each call.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()

            # Print stats
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats("cumtime")
            stats.print_stats(10)
            logger.info(f"\n{stream.getvalue()}")

        return result

    return wrapper


# Global profiler instance for MPC optimization
# Set warning threshold to 30ms (assuming 50ms control interval with 20ms margin)
mpc_profiler = PerformanceProfiler(warning_threshold=0.030)
