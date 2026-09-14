"""
perception/tracking.py — Lightweight object tracking.

Provides IoU (Intersection over Union) based matching to track objects
across frames without relying on heavy appearance features or complex filters.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.perception.models import BoundingBox, DetectedObject, ObjectDetectionResult, TrackedObject


def _compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    x_left = max(box1.x_min, box2.x_min)
    y_top = max(box1.y_min, box2.y_min)
    x_right = min(box1.x_max, box2.x_max)
    y_bottom = min(box1.y_max, box2.y_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
    area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
    
    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


class ObjectTracker:
    """A lightweight IoU-based object tracker."""
    
    def __init__(
        self,
        max_disappeared_seconds: float = 0.5,
        iou_threshold: float = 0.3
    ):
        self._max_disappeared_seconds = max_disappeared_seconds
        self._iou_threshold = iou_threshold
        
        self._next_id = 1
        self._tracks: Dict[int, TrackedObject] = {}
        
    def update(
        self, 
        detections: Optional[ObjectDetectionResult], 
        current_timestamp: float
    ) -> Tuple[TrackedObject, ...]:
        """Update tracks with new detections, or extrapolate if none provided.
        
        Args:
            detections: The latest YOLO detections. If None, the tracker
                will just extrapolate current tracks using velocity.
            current_timestamp: The capture timestamp of the current frame.
        """
        if detections is not None and detections.objects:
            self._match_and_update(detections.objects, current_timestamp)
        else:
            self._extrapolate(current_timestamp)
            
        self._remove_stale_tracks(current_timestamp)
        return tuple(self._tracks.values())
        
    def _match_and_update(self, detected_objects: Tuple[DetectedObject, ...], timestamp: float) -> None:
        if not self._tracks:
            # Register all as new
            for obj in detected_objects:
                self._register_new_track(obj, timestamp)
            return

        track_ids = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].bbox for tid in track_ids]
        
        # Build IoU matrix
        iou_matrix = np.zeros((len(detected_objects), len(track_ids)))
        for i, det in enumerate(detected_objects):
            for j, tbox in enumerate(track_boxes):
                if det.class_name == self._tracks[track_ids[j]].class_name:
                    iou_matrix[i, j] = _compute_iou(det.bbox, tbox)
                
        # Greedy matching
        matched_det_indices = set()
        matched_track_indices = set()
        
        # Sort indices by highest IoU first
        # We flatten the matrix, sort it, and get the pairs
        if iou_matrix.size > 0:
            flat_indices = np.argsort(iou_matrix, axis=None)[::-1]
            for idx in flat_indices:
                det_idx = int(idx // len(track_ids))
                trk_idx = int(idx % len(track_ids))
                
                iou = iou_matrix[det_idx, trk_idx]
                if iou < self._iou_threshold:
                    break
                    
                if det_idx not in matched_det_indices and trk_idx not in matched_track_indices:
                    matched_det_indices.add(det_idx)
                    matched_track_indices.add(trk_idx)
                    
                    # Update matched track
                    self._update_track(track_ids[trk_idx], detected_objects[det_idx], timestamp)

        # Register unmatched detections
        for i, det in enumerate(detected_objects):
            if i not in matched_det_indices:
                self._register_new_track(det, timestamp)
                
    def _register_new_track(self, obj: DetectedObject, timestamp: float) -> None:
        track_id = self._next_id
        self._next_id += 1
        
        self._tracks[track_id] = TrackedObject(
            track_id=track_id,
            class_name=obj.class_name,
            bbox=obj.bbox,
            center=(obj.bbox.center_x, obj.bbox.center_y),
            velocity=(0.0, 0.0),
            last_seen=timestamp,
            confidence=obj.confidence
        )
        
    def _update_track(self, track_id: int, obj: DetectedObject, timestamp: float) -> None:
        old_track = self._tracks[track_id]
        
        dt = timestamp - old_track.last_seen
        new_center_x = obj.bbox.center_x
        new_center_y = obj.bbox.center_y
        
        vx = old_track.velocity[0]
        vy = old_track.velocity[1]
        
        if dt > 0:
            vx = (new_center_x - old_track.center[0]) / dt
            vy = (new_center_y - old_track.center[1]) / dt
            
            # Simple exponential moving average for velocity
            alpha = 0.5
            vx = alpha * vx + (1 - alpha) * old_track.velocity[0]
            vy = alpha * vy + (1 - alpha) * old_track.velocity[1]
            
        self._tracks[track_id] = TrackedObject(
            track_id=track_id,
            class_name=obj.class_name,
            bbox=obj.bbox,
            center=(new_center_x, new_center_y),
            velocity=(vx, vy),
            last_seen=timestamp,
            confidence=obj.confidence
        )

    def _extrapolate(self, timestamp: float) -> None:
        # Move boxes slightly based on velocity
        for track_id, track in list(self._tracks.items()):
            dt = timestamp - track.last_seen
            # Only extrapolate a little bit to avoid boxes flying away
            extrapolate_dt = min(dt, 0.1) 
            
            dx = track.velocity[0] * extrapolate_dt
            dy = track.velocity[1] * extrapolate_dt
            
            # We don't update last_seen because it wasn't actually seen
            new_bbox = BoundingBox(
                x_min=track.bbox.x_min + dx,
                y_min=track.bbox.y_min + dy,
                x_max=track.bbox.x_max + dx,
                y_max=track.bbox.y_max + dy,
            )
            
            new_center = (new_bbox.center_x, new_bbox.center_y)
            
            self._tracks[track_id] = TrackedObject(
                track_id=track.track_id,
                class_name=track.class_name,
                bbox=new_bbox,
                center=new_center,
                velocity=track.velocity,
                last_seen=track.last_seen,
                confidence=track.confidence
            )
            
    def _remove_stale_tracks(self, timestamp: float) -> None:
        stale_ids = [
            tid for tid, track in self._tracks.items()
            if (timestamp - track.last_seen) > self._max_disappeared_seconds
        ]
        for tid in stale_ids:
            del self._tracks[tid]
