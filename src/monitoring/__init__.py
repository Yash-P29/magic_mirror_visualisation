"""
Monitoring package for Magic Mirror.

Provides FPS and latency measurement utilities that are independent of any
specific pipeline stage and can be reused as the project grows.
"""

from .metrics import FPSCounter, LatencyTracker

__all__ = ["FPSCounter", "LatencyTracker"]
