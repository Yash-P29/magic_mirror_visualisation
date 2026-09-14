"""
perception/temporal.py — Temporal scene state management.

Maintains a short rolling history of scene states to detect events
like objects appearing/disappearing or gestures starting/stopping.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

from src.perception.models import (
    ActivityState,
    Gesture,
    HandMoved,
    ObjectAppeared,
    ObjectDisappeared,
    PointingStarted,
    PointingStopped,
    SceneEvent,
    SceneState,
    TrackedObject,
)


class SceneHistory:
    """Maintains a rolling buffer of SceneState."""

    def __init__(self, max_history_frames: int = 90):
        # 90 frames is ~3 seconds at 30 FPS.
        self._history: Deque[SceneState] = deque(maxlen=max_history_frames)

    def append(self, state: SceneState) -> None:
        self._history.append(state)

    @property
    def current_state(self) -> Optional[SceneState]:
        if not self._history:
            return None
        return self._history[-1]

    @property
    def previous_state(self) -> Optional[SceneState]:
        if len(self._history) < 2:
            return None
        return self._history[-2]

    def detect_events(self, new_objects: Tuple[TrackedObject, ...], new_activity: ActivityState, timestamp: float) -> List[SceneEvent]:
        """Detect events by comparing new state elements against the previous state."""
        events: List[SceneEvent] = []
        prev = self.current_state

        if prev is None:
            # Everything is new
            for obj in new_objects:
                events.append(ObjectAppeared(timestamp=timestamp, object_id=obj.track_id, class_name=obj.class_name))
            return events

        prev_obj_ids = {obj.track_id: obj for obj in prev.tracked_objects}
        new_obj_ids = {obj.track_id: obj for obj in new_objects}

        # Object Appeared
        for obj_id, obj in new_obj_ids.items():
            if obj_id not in prev_obj_ids:
                events.append(ObjectAppeared(timestamp=timestamp, object_id=obj_id, class_name=obj.class_name))

        # Object Disappeared
        for obj_id, obj in prev_obj_ids.items():
            if obj_id not in new_obj_ids:
                events.append(ObjectDisappeared(timestamp=timestamp, object_id=obj_id, class_name=obj.class_name))

        # Gestures Started / Stopped
        if prev.activity:
            # Left Hand Pointing
            if new_activity.left_hand_gesture == Gesture.POINTING and prev.activity.left_hand_gesture != Gesture.POINTING:
                events.append(PointingStarted(timestamp=timestamp, hand_is_left=True))
            elif new_activity.left_hand_gesture != Gesture.POINTING and prev.activity.left_hand_gesture == Gesture.POINTING:
                events.append(PointingStopped(timestamp=timestamp, hand_is_left=True))
                
            # Right Hand Pointing
            if new_activity.right_hand_gesture == Gesture.POINTING and prev.activity.right_hand_gesture != Gesture.POINTING:
                events.append(PointingStarted(timestamp=timestamp, hand_is_left=False))
            elif new_activity.right_hand_gesture != Gesture.POINTING and prev.activity.right_hand_gesture == Gesture.POINTING:
                events.append(PointingStopped(timestamp=timestamp, hand_is_left=False))

        return events
