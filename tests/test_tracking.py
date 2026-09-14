import time
from typing import Tuple

import pytest

from src.perception.models import BoundingBox, DetectedObject, ObjectDetectionResult, TrackedObject
from src.perception.tracking import ObjectTracker


def test_tracker_creation_and_matching():
    tracker = ObjectTracker(max_disappeared_seconds=0.5, iou_threshold=0.3)

    # Frame 1: Detection
    obj1 = DetectedObject(class_id=1, class_name="laptop", confidence=0.9, bbox=BoundingBox(0.1, 0.1, 0.3, 0.3))
    res1 = ObjectDetectionResult(frame_id=1, capture_timestamp=1.0, perception_start=1.0, perception_end=1.01, objects=(obj1,))
    
    tracks = tracker.update(res1, 1.0)
    assert len(tracks) == 1
    assert tracks[0].class_name == "laptop"
    assert tracks[0].track_id == 1

    # Frame 2: Same object moved slightly
    obj2 = DetectedObject(class_id=1, class_name="laptop", confidence=0.9, bbox=BoundingBox(0.12, 0.12, 0.32, 0.32))
    res2 = ObjectDetectionResult(frame_id=2, capture_timestamp=1.1, perception_start=1.1, perception_end=1.11, objects=(obj2,))
    
    tracks = tracker.update(res2, 1.1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1  # Should match same ID
    
    # Velocity should be non-zero now
    assert tracks[0].velocity[0] > 0
    assert tracks[0].velocity[1] > 0

def test_tracker_staleness():
    tracker = ObjectTracker(max_disappeared_seconds=0.5, iou_threshold=0.3)

    # Frame 1: Detection
    obj1 = DetectedObject(class_id=1, class_name="laptop", confidence=0.9, bbox=BoundingBox(0.1, 0.1, 0.3, 0.3))
    res1 = ObjectDetectionResult(frame_id=1, capture_timestamp=1.0, perception_start=1.0, perception_end=1.01, objects=(obj1,))
    
    tracker.update(res1, 1.0)
    
    # Frame 2: No detection, within staleness limit
    tracks = tracker.update(None, 1.2)
    assert len(tracks) == 1
    
    # Frame 3: No detection, exceeds staleness limit
    tracks = tracker.update(None, 1.6)
    assert len(tracks) == 0
