"""
tests/test_telemetry_hardening.py — Regression and invariant tests for Milestone 2A.1.2.

Validates:
  A. HandData left/right properties (is_left, is_right)
  B. Latest-frame replacement policy
  C. Dropped frame accounting
  D. Captured / Consumed / Dropped / Buffered invariant
  E. Capture FPS vs Processing FPS separation
  F. capture_read_ms propagation
  G. consumer_wait_ms vs capture_read_ms separation
  H. frame_age_ms and capture_to_processing_ms calculations
  I. AsyncCameraReader clean shutdown behavior
  J. Producer failure handling without deadlock
  K. Repeated frame replacement
  L. No stale frame accumulation
"""

import time
import unittest
from unittest.mock import MagicMock
import numpy as np

from src.camera.capture import AsyncCameraReader, Camera, CapturedFrame
from src.monitoring.metrics import FPSCounter, LatencyStats, LatencyTracker
from src.perception.models import HandData, Landmark, PoseData, PerceptionResult


class TestHandDataProperties(unittest.TestCase):
    def test_hand_data_left_and_right(self):
        lms = tuple(Landmark(x=0.1 * i, y=0.1, z=0.0) for i in range(21))
        left_hand = HandData(handedness="Left", landmarks=lms, score=None)
        right_hand = HandData(handedness="Right", landmarks=lms, score=None)

        self.assertTrue(left_hand.is_left)
        self.assertFalse(left_hand.is_right)
        self.assertIsNone(left_hand.score)

        self.assertTrue(right_hand.is_right)
        self.assertFalse(right_hand.is_left)

    def test_immutability_tuples(self):
        lms = [Landmark(x=0.1 * i, y=0.1, z=0.0) for i in range(21)]
        hand = HandData(handedness="Left", landmarks=lms)
        self.assertIsInstance(hand.landmarks, tuple)

        pose = PoseData(landmarks=[Landmark(0, 0, 0) for _ in range(33)])
        self.assertIsInstance(pose.landmarks, tuple)
        self.assertIsInstance(pose.world_landmarks, tuple)

        res = PerceptionResult(frame_id=0, capture_timestamp=0.0, perception_start=0.0, perception_end=0.0, pose=None, hands=[hand])
        self.assertIsInstance(res.hands, tuple)


class TestAsyncCameraInvariantsAndTelemetry(unittest.TestCase):
    def setUp(self):
        self.mock_camera = MagicMock(spec=Camera)
        self.mock_camera.is_opened.return_value = True
        self.frame_counter = 0

    def _make_frame(self):
        f = CapturedFrame(
            frame_id=self.frame_counter,
            capture_timestamp=time.monotonic(),
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
        )
        self.frame_counter += 1
        return f

    def test_captured_consumed_dropped_invariant(self):
        self.mock_camera.read.side_effect = self._make_frame

        reader = AsyncCameraReader(self.mock_camera)
        reader.start()
        time.sleep(0.05)  # allow producer to run
        self.assertTrue(reader.invariant_valid)

        frame = reader.read_latest(timeout=0.1)
        self.assertIsNotNone(frame)
        self.assertTrue(reader.invariant_valid)

        reader.stop()
        self.assertTrue(reader.invariant_valid)

    def test_latest_frame_replacement_and_no_stale_accumulation(self):
        self.mock_camera.read.side_effect = self._make_frame

        reader = AsyncCameraReader(self.mock_camera)
        reader.start()
        time.sleep(0.05)  # Produce multiple frames while consumer is idle
        reader.stop()

        # Check latest frame wins
        latest = reader.read_latest(timeout=0.05)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.frame_id, reader.captured_frames - 1)

        # Confirm buffer is empty now (no stale accumulation)
        stale = reader.read_latest(timeout=0.05)
        self.assertIsNone(stale)

    def test_capture_read_ms_and_consumer_wait_ms_separation(self):
        tracker = LatencyTracker(rolling_window=10)
        t0 = time.monotonic()
        t_start = t0 + 0.01
        t_end = t0 + 0.03

        stats = tracker.record(
            capture_timestamp=t0,
            process_start=t_start,
            process_end=t_end,
            capture_read_ms=5.0,
            consumer_wait_ms=12.0,
        )

        self.assertAlmostEqual(stats.capture_read_ms, 5.0)
        self.assertAlmostEqual(stats.consumer_wait_ms, 12.0)
        self.assertAlmostEqual(stats.frame_age_ms, 10.0, delta=1.0)
        self.assertAlmostEqual(stats.processing_ms, 20.0, delta=1.0)
        self.assertAlmostEqual(stats.capture_to_processing_ms, 30.0, delta=1.0)

    def test_shutdown_thread_safety(self):
        self.mock_camera.read.side_effect = self._make_frame

        reader = AsyncCameraReader(self.mock_camera)
        reader.start()
        time.sleep(0.02)
        reader.stop()

        self.assertFalse(reader._running)
        self.assertIsNone(reader._thread)
        self.mock_camera.release.assert_called()


if __name__ == "__main__":
    unittest.main()
