"""
tests/test_perception.py — Unit tests for the Fast Perception layer.

These tests do NOT require a physical webcam or a real MediaPipe model.
MediaPipe is mocked where necessary.

Test coverage:
  - PerceptionResult construction and properties
  - PoseData construction and landmark accessors
  - HandData construction and landmark accessors
  - Landmark dataclass
  - Empty / partial / missing detections
  - HumanPerceptionPipeline result conversion from mocked MediaPipe output
  - Perception timing recorded in PerceptionResult
  - Extended LatencyTracker.record_perception()
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.perception.models import (
    HandData,
    HandLandmarkIndex,
    Landmark,
    PerceptionResult,
    PoseData,
    PoseLandmarkIndex,
)
from src.monitoring.metrics import LatencyStats, LatencyTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmark(x=0.5, y=0.5, z=0.0, vis=1.0) -> Landmark:
    return Landmark(x=x, y=y, z=z, visibility=vis)


def _make_landmarks(n: int, x_offset: float = 0.0) -> List[Landmark]:
    return [_make_landmark(x=i * 0.03 + x_offset) for i in range(n)]


def _make_pose_data(n_landmarks: int = 33) -> PoseData:
    lms = _make_landmarks(n_landmarks)
    return PoseData(landmarks=lms, world_landmarks=[])


def _make_hand_data(handedness: str = "Left", n_landmarks: int = 21) -> HandData:
    lms = _make_landmarks(n_landmarks)
    return HandData(handedness=handedness, landmarks=lms, score=0.9)


def _make_perception_result(
    frame_id: int = 0,
    pose: Optional[PoseData] = None,
    hands: Optional[List[HandData]] = None,
    perception_start: float = 1.0,
    perception_end: float = 1.025,
) -> PerceptionResult:
    return PerceptionResult(
        frame_id=frame_id,
        capture_timestamp=0.9,
        perception_start=perception_start,
        perception_end=perception_end,
        pose=pose,
        hands=hands or [],
    )


# ---------------------------------------------------------------------------
# Landmark tests
# ---------------------------------------------------------------------------

class TestLandmark:
    def test_basic_construction(self):
        lm = Landmark(x=0.1, y=0.2, z=0.3, visibility=0.9)
        assert lm.x == 0.1
        assert lm.y == 0.2
        assert lm.z == 0.3
        assert lm.visibility == 0.9

    def test_visibility_optional(self):
        lm = Landmark(x=0.0, y=0.0, z=0.0)
        assert lm.visibility is None

    def test_frozen(self):
        """Landmarks are immutable."""
        lm = Landmark(x=0.5, y=0.5, z=0.0)
        with pytest.raises((AttributeError, TypeError)):
            lm.x = 0.9  # type: ignore


# ---------------------------------------------------------------------------
# PoseData tests
# ---------------------------------------------------------------------------

class TestPoseData:
    def test_construction_with_33_landmarks(self):
        pose = _make_pose_data(33)
        assert len(pose.landmarks) == 33
        assert len(pose.world_landmarks) == 0

    def test_get_in_range(self):
        pose = _make_pose_data(33)
        lm = pose.get(PoseLandmarkIndex.NOSE)
        assert lm is not None
        assert isinstance(lm, Landmark)

    def test_get_out_of_range_returns_none(self):
        pose = _make_pose_data(33)
        assert pose.get(100) is None
        assert pose.get(-1) is None

    def test_convenience_accessors(self):
        pose = _make_pose_data(33)
        assert pose.nose is not None
        assert pose.left_shoulder is not None
        assert pose.right_shoulder is not None
        assert pose.left_wrist is not None
        assert pose.right_wrist is not None
        assert pose.left_hip is not None
        assert pose.right_hip is not None
        assert pose.left_knee is not None
        assert pose.right_knee is not None
        assert pose.left_ankle is not None
        assert pose.right_ankle is not None

    def test_empty_landmarks_returns_none_on_get(self):
        pose = PoseData(landmarks=[], world_landmarks=[])
        assert pose.get(0) is None
        assert pose.nose is None

    def test_world_landmarks_accessible(self):
        wlms = _make_landmarks(33)
        pose = PoseData(landmarks=_make_landmarks(33), world_landmarks=wlms)
        assert len(pose.world_landmarks) == 33

    def test_frozen(self):
        pose = _make_pose_data(33)
        with pytest.raises((AttributeError, TypeError)):
            pose.landmarks = []  # type: ignore


# ---------------------------------------------------------------------------
# HandData tests
# ---------------------------------------------------------------------------

class TestHandData:
    def test_construction(self):
        hand = _make_hand_data("Right", 21)
        assert hand.handedness == "Right"
        assert len(hand.landmarks) == 21
        assert hand.score == 0.9

    def test_left_hand(self):
        hand = _make_hand_data("Left")
        assert hand.handedness == "Left"

    def test_get_in_range(self):
        hand = _make_hand_data()
        lm = hand.get(HandLandmarkIndex.WRIST)
        assert lm is not None

    def test_get_out_of_range(self):
        hand = _make_hand_data()
        assert hand.get(100) is None

    def test_convenience_accessors(self):
        hand = _make_hand_data()
        assert hand.wrist is not None
        assert hand.index_tip is not None
        assert hand.thumb_tip is not None

    def test_frozen(self):
        hand = _make_hand_data()
        with pytest.raises((AttributeError, TypeError)):
            hand.handedness = "Unknown"  # type: ignore


# ---------------------------------------------------------------------------
# PerceptionResult tests
# ---------------------------------------------------------------------------

class TestPerceptionResult:
    def test_empty_result(self):
        result = _make_perception_result()
        assert not result.has_pose
        assert result.num_hands == 0
        assert result.pose is None
        assert result.hands == []

    def test_with_pose(self):
        pose = _make_pose_data()
        result = _make_perception_result(pose=pose)
        assert result.has_pose
        assert result.pose is not None

    def test_with_two_hands(self):
        hands = [_make_hand_data("Left"), _make_hand_data("Right")]
        result = _make_perception_result(hands=hands)
        assert result.num_hands == 2

    def test_perception_ms_calculated(self):
        result = _make_perception_result(
            perception_start=1.000,
            perception_end=1.025,
        )
        assert abs(result.perception_ms - 25.0) < 1e-6

    def test_perception_ms_zero_duration(self):
        result = _make_perception_result(
            perception_start=1.000,
            perception_end=1.000,
        )
        assert result.perception_ms == 0.0

    def test_frame_id_stored(self):
        result = _make_perception_result(frame_id=999)
        assert result.frame_id == 999

    def test_capture_timestamp_accessible(self):
        result = _make_perception_result()
        assert isinstance(result.capture_timestamp, float)

    def test_missing_pose_and_hands(self):
        result = _make_perception_result(pose=None, hands=[])
        assert not result.has_pose
        assert result.num_hands == 0

    def test_partial_detection_one_hand(self):
        result = _make_perception_result(
            pose=None,
            hands=[_make_hand_data("Left")],
        )
        assert not result.has_pose
        assert result.num_hands == 1

    def test_frozen(self):
        result = _make_perception_result()
        with pytest.raises((AttributeError, TypeError)):
            result.frame_id = 42  # type: ignore


# ---------------------------------------------------------------------------
# HumanPerceptionPipeline mock-based tests
# ---------------------------------------------------------------------------

class TestHumanPerceptionPipelineConversion:
    """
    Tests that the pipeline correctly converts MediaPipe result objects
    into our own data models.  MediaPipe itself is mocked; these tests
    do NOT require a model file or a webcam.
    """

    def _make_mp_normalized_landmark(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        """Build a mock that looks like a MediaPipe NormalizedLandmark."""
        lm = MagicMock()
        lm.x = x
        lm.y = y
        lm.z = z
        lm.visibility = visibility
        return lm

    def _make_mp_pose_result(self, n=33):
        """Build a mock MediaPipe HolisticLandmarkerResult with pose only."""
        result = MagicMock()
        lms = [self._make_mp_normalized_landmark(x=i * 0.01) for i in range(n)]
        result.pose_landmarks = lms
        result.pose_world_landmarks = lms
        result.left_hand_landmarks = []
        result.right_hand_landmarks = []
        return result

    def _make_mp_hands_result(self, left=True, right=True):
        """Build a mock result with hands but no pose."""
        result = MagicMock()
        result.pose_landmarks = []
        result.pose_world_landmarks = []
        lms_21 = [self._make_mp_normalized_landmark(x=i * 0.01) for i in range(21)]
        result.left_hand_landmarks = lms_21 if left else []
        result.right_hand_landmarks = lms_21 if right else []
        return result

    def _make_mp_empty_result(self):
        result = MagicMock()
        result.pose_landmarks = []
        result.pose_world_landmarks = []
        result.left_hand_landmarks = []
        result.right_hand_landmarks = []
        return result

    def _call_extract_pose(self, mp_result):
        from src.perception.human import HumanPerceptionPipeline
        return HumanPerceptionPipeline._extract_pose(mp_result)

    def _call_extract_hands(self, mp_result):
        from src.perception.human import HumanPerceptionPipeline
        return HumanPerceptionPipeline._extract_hands(mp_result)

    def test_pose_extracted_correctly(self):
        mp_result = self._make_mp_pose_result(n=33)
        pose = self._call_extract_pose(mp_result)
        assert pose is not None
        assert len(pose.landmarks) == 33
        assert isinstance(pose.landmarks[0], Landmark)

    def test_pose_x_values_transferred(self):
        mp_result = self._make_mp_pose_result(n=33)
        pose = self._call_extract_pose(mp_result)
        for i, lm in enumerate(pose.landmarks):
            assert abs(lm.x - i * 0.01) < 1e-6

    def test_empty_pose_returns_none(self):
        mp_result = self._make_mp_empty_result()
        pose = self._call_extract_pose(mp_result)
        assert pose is None

    def test_both_hands_extracted(self):
        mp_result = self._make_mp_hands_result(left=True, right=True)
        hands = self._call_extract_hands(mp_result)
        assert len(hands) == 2
        handedness = {h.handedness for h in hands}
        assert "Left" in handedness
        assert "Right" in handedness

    def test_left_hand_only(self):
        mp_result = self._make_mp_hands_result(left=True, right=False)
        hands = self._call_extract_hands(mp_result)
        assert len(hands) == 1
        assert hands[0].handedness == "Left"

    def test_right_hand_only(self):
        mp_result = self._make_mp_hands_result(left=False, right=True)
        hands = self._call_extract_hands(mp_result)
        assert len(hands) == 1
        assert hands[0].handedness == "Right"

    def test_no_hands_returns_empty_list(self):
        mp_result = self._make_mp_empty_result()
        hands = self._call_extract_hands(mp_result)
        assert hands == []

    def test_hand_has_21_landmarks(self):
        mp_result = self._make_mp_hands_result(left=True, right=False)
        hands = self._call_extract_hands(mp_result)
        assert len(hands[0].landmarks) == 21

    def test_landmark_type_is_our_class(self):
        mp_result = self._make_mp_hands_result(left=True, right=False)
        hands = self._call_extract_hands(mp_result)
        for lm in hands[0].landmarks:
            assert isinstance(lm, Landmark)


# ---------------------------------------------------------------------------
# Extended LatencyTracker — perception tracking
# ---------------------------------------------------------------------------

class TestLatencyTrackerPerception:
    def test_record_perception_updates_current(self):
        tracker = LatencyTracker()
        tracker.record_perception(perception_ms=20.0)
        assert abs(tracker._stats.perception_ms - 20.0) < 1e-6

    def test_record_perception_rolling_average(self):
        tracker = LatencyTracker(rolling_window=3)
        tracker.record_perception(perception_ms=10.0)
        tracker.record_perception(perception_ms=20.0)
        tracker.record_perception(perception_ms=30.0)
        assert abs(tracker._stats.avg_perception_ms - 20.0) < 1e-6

    def test_record_perception_min_max(self):
        tracker = LatencyTracker()
        tracker.record_perception(perception_ms=15.0)
        tracker.record_perception(perception_ms=5.0)
        tracker.record_perception(perception_ms=25.0)
        assert abs(tracker._stats.min_perception_ms - 5.0) < 1e-6
        assert abs(tracker._stats.max_perception_ms - 25.0) < 1e-6

    def test_perception_independent_from_app(self):
        """Perception and app timing buffers must not interfere."""
        tracker = LatencyTracker()
        tracker.record(capture_timestamp=0.0, process_start=0.001, process_end=0.003)
        tracker.record_perception(perception_ms=18.0)
        # App stats unchanged by perception call
        assert abs(tracker._stats.processing_ms - 2.0) < 1e-6
        assert abs(tracker._stats.perception_ms - 18.0) < 1e-6

    def test_reset_clears_perception_buffer(self):
        tracker = LatencyTracker()
        tracker.record_perception(perception_ms=50.0)
        tracker.reset()
        tracker.record_perception(perception_ms=10.0)
        assert abs(tracker._stats.avg_perception_ms - 10.0) < 1e-6

    def test_perception_evicts_old_values(self):
        tracker = LatencyTracker(rolling_window=2)
        tracker.record_perception(perception_ms=100.0)
        tracker.record_perception(perception_ms=10.0)
        tracker.record_perception(perception_ms=20.0)
        # Window holds last 2: [10, 20] → avg=15, 100 evicted
        assert abs(tracker._stats.avg_perception_ms - 15.0) < 1e-6


# ---------------------------------------------------------------------------
# Pipeline open() without model file
# ---------------------------------------------------------------------------

class TestHumanPerceptionPipelineErrors:
    def test_open_raises_file_not_found_for_missing_model(self, tmp_path):
        from src.perception.human import HumanPerceptionPipeline
        pipeline = HumanPerceptionPipeline(model_path=tmp_path / "nonexistent.task")
        with pytest.raises(FileNotFoundError):
            pipeline.open()

    def test_process_before_open_returns_empty_result(self, tmp_path):
        import numpy as np
        from src.camera.capture import CapturedFrame
        from src.perception.human import HumanPerceptionPipeline

        pipeline = HumanPerceptionPipeline(model_path=tmp_path / "nonexistent.task")
        # Don't call open()
        fake_frame = CapturedFrame(
            frame_id=0,
            capture_timestamp=time.monotonic(),
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
        )
        result = pipeline.process(fake_frame)
        assert result.pose is None
        assert result.hands == []

    def test_is_ready_false_before_open(self, tmp_path):
        from src.perception.human import HumanPerceptionPipeline
        pipeline = HumanPerceptionPipeline(model_path=tmp_path / "x.task")
        assert not pipeline.is_ready()
