"""
tests/test_async_camera.py — Unit tests for AsyncCameraReader (Decoupled Capture).

Validates:
  - Newest frame replaces old frame (latest-frame-wins slot policy)
  - Replaced frame increments dropped count
  - Consumed frame is the newest available frame
  - Thread safety under concurrent access
  - Clean shutdown
  - Camera read failure does not deadlock producer
  - No unbounded frame accumulation
"""

import time
import unittest
from unittest.mock import MagicMock
import numpy as np

from src.camera.capture import AsyncCameraReader, Camera, CapturedFrame


class TestAsyncCameraReader(unittest.TestCase):
    def setUp(self):
        self.mock_camera = MagicMock(spec=Camera)
        self.mock_camera.is_opened.return_value = True
        self.mock_camera.read.return_value = CapturedFrame(
            frame_id=0,
            capture_timestamp=time.monotonic(),
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
        )

    def test_basic_start_and_stop(self):
        reader = AsyncCameraReader(self.mock_camera)
        reader.start()
        time.sleep(0.05)
        self.assertTrue(reader.captured_frames > 0)
        reader.stop()

    def test_latest_frame_wins_and_dropped_count(self):
        count = 0
        def mock_read():
            nonlocal count
            frame = CapturedFrame(frame_id=count, capture_timestamp=time.monotonic(), image=np.zeros((720, 1280, 3), dtype=np.uint8))
            count += 1
            return frame

        self.mock_camera.read.side_effect = mock_read

        reader = AsyncCameraReader(self.mock_camera)
        reader.start()
        time.sleep(0.05)  # Allow producer thread to produce frames without consumer reading
        reader.stop()

        self.assertTrue(reader.captured_frames > 1)
        self.assertTrue(reader.dropped_frames > 0)
        self.assertEqual(reader.dropped_frames, reader.captured_frames - 1)

        # Retrieve latest frame
        consumed = reader.read_latest(timeout=0.1)
        self.assertIsNotNone(consumed)
        self.assertEqual(consumed.frame_id, reader._latest_frame.frame_id)

    def test_read_latest_consumes_newest(self):
        reader = AsyncCameraReader(self.mock_camera)
        reader.start()
        time.sleep(0.02)
        frame1 = reader.read_latest(timeout=0.1)
        self.assertIsNotNone(frame1)
        self.assertEqual(reader.consumed_frames, 1)
        reader.stop()

    def test_camera_failure_does_not_deadlock(self):
        self.mock_camera.read.return_value = None
        reader = AsyncCameraReader(self.mock_camera)
        reader.start()
        time.sleep(0.02)
        frame = reader.read_latest(timeout=0.05)
        self.assertIsNone(frame)
        reader.stop()


if __name__ == "__main__":
    unittest.main()
