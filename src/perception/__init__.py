"""
perception/__init__.py — Fast Perception package for the Magic Mirror pipeline.

Public API re-exported here so downstream code can write:
    from src.perception import HumanPerceptionPipeline, PerceptionResult
"""

from .human import HumanPerceptionPipeline
from .models import (
    HandData,
    HandLandmarkIndex,
    Landmark,
    PerceptionResult,
    PoseData,
    PoseLandmarkIndex,
)

__all__ = [
    "HumanPerceptionPipeline",
    "PerceptionResult",
    "PoseData",
    "HandData",
    "Landmark",
    "PoseLandmarkIndex",
    "HandLandmarkIndex",
]
