"""
Camera package for Magic Mirror.

Provides the Camera abstraction and CapturedFrame dataclass used by all
downstream pipeline stages.
"""

from .capture import AsyncCameraReader, Camera, CapturedFrame

__all__ = ["AsyncCameraReader", "Camera", "CapturedFrame"]
