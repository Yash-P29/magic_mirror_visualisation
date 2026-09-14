"""
perception/objects.py — Fast Object Detection using YOLOv8.

Wraps ultralytics YOLO to detect standard objects in the classroom/office.
Prioritizes lowest latency.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Lazy import ultralytics to avoid slowing down startup if not used
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.perception.models import BoundingBox, DetectedObject, ObjectDetectionResult
from src.camera.capture import CapturedFrame

logger = logging.getLogger(__name__)

# Standard COCO classes we care about in a lecture environment
# 0: person, 39: bottle, 41: cup, 43: fork, 44: knife, 45: spoon, 56: chair
# 63: laptop, 64: mouse, 65: remote, 66: keyboard, 67: cell phone
# 73: book
_TARGET_CLASSES = {0, 39, 41, 56, 63, 64, 65, 66, 67, 73}

class ObjectDetector:
    """Real-time object detector using YOLO.
    
    Initializes the model once.
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.25):
        self._model_name = model_name
        self._conf_threshold = conf_threshold
        self._model = None
        self._is_ready = False

    def open(self) -> None:
        """Load the YOLO model."""
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Run `pip install ultralytics`.")
        
        logger.info("Loading YOLO model: %s", self._model_name)
        # Load model. It will download automatically if not present.
        self._model = YOLO(self._model_name)
        
        # Warmup
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy_img, verbose=False)
        self._is_ready = True
        logger.info("YOLO model ready.")

    def close(self) -> None:
        """Release resources."""
        self._model = None
        self._is_ready = False
        logger.info("YOLO model closed.")

    def is_ready(self) -> bool:
        return self._is_ready

    def process(self, frame: CapturedFrame) -> ObjectDetectionResult:
        perception_start = time.monotonic()
        
        if not self._is_ready or self._model is None:
            logger.warning("ObjectDetector.process() called before open().")
            return ObjectDetectionResult(
                frame_id=frame.frame_id,
                capture_timestamp=frame.capture_timestamp,
                perception_start=perception_start,
                perception_end=time.monotonic(),
                objects=(),
            )
        
        detected_objects: List[DetectedObject] = []
        
        try:
            # Run YOLO inference
            # We pass the BGR frame directly (YOLO handles BGR)
            results = self._model(
                frame.image, 
                verbose=False,
                conf=self._conf_threshold,
                classes=list(_TARGET_CLASSES)
            )
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                # Image dimensions for normalization
                h, w = frame.image.shape[:2]
                
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    # xyxy coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Normalize bounding box
                    nx1 = float(x1) / w
                    ny1 = float(y1) / h
                    nx2 = float(x2) / w
                    ny2 = float(y2) / h
                    
                    cls_name = result.names[cls_id]
                    
                    detected_objects.append(DetectedObject(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=BoundingBox(
                            x_min=nx1,
                            y_min=ny1,
                            x_max=nx2,
                            y_max=ny2
                        )
                    ))
                    
        except Exception as exc:
            logger.warning("ObjectDetector error on frame %d: %s", frame.frame_id, exc)

        perception_end = time.monotonic()
        
        return ObjectDetectionResult(
            frame_id=frame.frame_id,
            capture_timestamp=frame.capture_timestamp,
            perception_start=perception_start,
            perception_end=perception_end,
            objects=tuple(detected_objects)
        )
        
    def __enter__(self) -> "ObjectDetector":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
