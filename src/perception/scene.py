"""
perception/scene.py — High-level scene understanding and state construction.

Combines Human Perception (MediaPipe), Object Detection (YOLO),
Tracking, and Temporal History to produce a unified SceneState.
Evaluates heuristics for basic gestures and hand-object relationships.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

from src.camera.capture import CapturedFrame
from src.perception.human import HumanPerceptionPipeline
from src.perception.models import (
    ActivityState,
    Gesture,
    HandData,
    HandNearObject,
    HandOverlappingObject,
    HandPointingAtObject,
    PerceptionResult,
    PoseData,
    Relationship,
    SceneState,
    TrackedObject,
)
from src.perception.objects import ObjectDetector
from src.perception.temporal import SceneHistory
from src.perception.tracking import ObjectTracker

logger = logging.getLogger(__name__)


def _compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _is_pointing(hand: HandData) -> bool:
    """Simple heuristic for a pointing gesture."""
    wrist = hand.wrist
    index_tip = hand.index_tip
    index_mcp = hand.index_mcp
    middle_tip = hand.middle_tip
    ring_tip = hand.ring_tip
    pinky_tip = hand.pinky_tip

    if not all([wrist, index_tip, index_mcp, middle_tip, ring_tip, pinky_tip]):
        return False

    # Check if index finger is extended
    index_ext = _compute_distance(wrist.x, wrist.y, index_tip.x, index_tip.y)
    index_base = _compute_distance(wrist.x, wrist.y, index_mcp.x, index_mcp.y)
    
    # Check if other fingers are curled (closer to wrist than index tip)
    mid_dist = _compute_distance(wrist.x, wrist.y, middle_tip.x, middle_tip.y)
    ring_dist = _compute_distance(wrist.x, wrist.y, ring_tip.x, ring_tip.y)
    pinky_dist = _compute_distance(wrist.x, wrist.y, pinky_tip.x, pinky_tip.y)

    if index_ext > index_base * 1.5:
        if max(mid_dist, ring_dist, pinky_dist) < index_ext * 0.7:
            return True

    return False


def _is_raised_hand(hand: HandData, pose: PoseData) -> bool:
    """Simple heuristic for a raised hand (hand above shoulder/head)."""
    wrist = hand.wrist
    if not wrist:
        return False
        
    shoulder = pose.left_shoulder if hand.is_left else pose.right_shoulder
    nose = pose.nose
    
    if not shoulder or not nose:
        return False
        
    # In normalized coords, y=0 is top. Hand above shoulder means wrist.y < shoulder.y
    return wrist.y < shoulder.y and wrist.y < nose.y


class SceneBuilder:
    """Orchestrates perception layers to build a unified SceneState."""

    def __init__(
        self,
        yolo_interval_ms: float = 100.0,
        tracking_iou_threshold: float = 0.3,
        history_frames: int = 90
    ):
        self._human_pipeline = HumanPerceptionPipeline()
        self._detector = ObjectDetector()
        self._tracker = ObjectTracker(iou_threshold=tracking_iou_threshold)
        self._history = SceneHistory(max_history_frames=history_frames)
        
        self._yolo_interval_s = yolo_interval_ms / 1000.0
        self._last_yolo_time = 0.0

    def open(self) -> None:
        self._human_pipeline.open()
        self._detector.open()

    def close(self) -> None:
        self._detector.close()
        self._human_pipeline.close()

    def process(self, frame: CapturedFrame) -> Tuple[SceneState, float, float, float]:
        """Process a frame through all perception layers.
        
        Returns:
            Tuple of (SceneState, human_perception_ms, yolo_ms, tracking_ms)
        """
        # 1. Human Perception (MediaPipe)
        human_res = self._human_pipeline.process(frame)

        # 2. Object Detection (YOLO) - Throttled
        now = time.monotonic()
        yolo_res = None
        yolo_ms = 0.0
        if now - self._last_yolo_time >= self._yolo_interval_s:
            yolo_res = self._detector.process(frame)
            yolo_ms = yolo_res.perception_ms
            self._last_yolo_time = now

        # 3. Tracking
        track_start = time.monotonic()
        tracks = self._tracker.update(yolo_res, frame.capture_timestamp)
        tracking_ms = (time.monotonic() - track_start) * 1000.0

        # 4. Activity and Gestures
        activity = self._infer_activity(human_res)

        # 5. Hand-Object Relationships
        relations = self._infer_relationships(human_res, tracks, activity)

        # 6. Events and Temporal State
        events = self._history.detect_events(tracks, activity, frame.capture_timestamp)

        # 7. Construct SceneState
        state = SceneState(
            frame_id=frame.frame_id,
            timestamp=frame.capture_timestamp,
            human_perception=human_res,
            tracked_objects=tracks,
            relationships=tuple(relations),
            recent_events=tuple(events),
            activity=activity,
        )
        self._history.append(state)

        return state, human_res.perception_ms, yolo_ms, tracking_ms

    def _infer_activity(self, human_res: PerceptionResult) -> ActivityState:
        left_gesture = Gesture.NONE
        right_gesture = Gesture.NONE
        
        for hand in human_res.hands:
            gesture = Gesture.NONE
            if _is_pointing(hand):
                gesture = Gesture.POINTING
            elif human_res.has_pose and _is_raised_hand(hand, human_res.pose):
                gesture = Gesture.RAISED_HAND
                
            if hand.is_left:
                left_gesture = gesture
            else:
                right_gesture = gesture

        return ActivityState(
            left_hand_gesture=left_gesture,
            right_hand_gesture=right_gesture,
            motion="none"  # Placeholder for broader motion tracking
        )

    def _infer_relationships(
        self, 
        human_res: PerceptionResult, 
        tracks: Tuple[TrackedObject, ...],
        activity: ActivityState
    ) -> List[Relationship]:
        relations: List[Relationship] = []

        for hand in human_res.hands:
            wrist = hand.wrist
            index_tip = hand.index_tip
            if not wrist or not index_tip:
                continue

            hand_gesture = activity.left_hand_gesture if hand.is_left else activity.right_hand_gesture

            for obj in tracks:
                box = obj.bbox
                
                # Check overlap (is wrist or index tip inside the bounding box?)
                is_overlapping = False
                if (box.x_min <= wrist.x <= box.x_max and box.y_min <= wrist.y <= box.y_max):
                    is_overlapping = True
                if (box.x_min <= index_tip.x <= box.x_max and box.y_min <= index_tip.y <= box.y_max):
                    is_overlapping = True
                    
                if is_overlapping:
                    relations.append(HandOverlappingObject(hand_is_left=hand.is_left, object_id=obj.track_id))
                    continue

                # Check pointing
                if hand_gesture == Gesture.POINTING:
                    # Simple raycast check: vector from wrist to index tip, check if it intersects box
                    # For a simple heuristic, just check distance from index tip to object center
                    dist_to_center = _compute_distance(index_tip.x, index_tip.y, obj.center[0], obj.center[1])
                    # And ensure it's generally pointing in that direction
                    vec_finger = (index_tip.x - wrist.x, index_tip.y - wrist.y)
                    vec_obj = (obj.center[0] - wrist.x, obj.center[1] - wrist.y)
                    
                    dot_product = vec_finger[0] * vec_obj[0] + vec_finger[1] * vec_obj[1]
                    if dot_product > 0 and dist_to_center < 0.3:  # Pointing roughly towards and not too far
                        relations.append(HandPointingAtObject(hand_is_left=hand.is_left, object_id=obj.track_id))
                        continue

                # Check near
                dist_to_center = _compute_distance(wrist.x, wrist.y, obj.center[0], obj.center[1])
                if dist_to_center < max(box.width, box.height) * 1.5:
                    relations.append(HandNearObject(hand_is_left=hand.is_left, object_id=obj.track_id))

        return relations
