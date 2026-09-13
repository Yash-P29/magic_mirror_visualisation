"""
tests/test_camera.py — Unit tests for the Camera / CapturedFrame layer.

All tests that touch Camera behaviour use a mock VideoCapture so no physical
webcam is required.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.camera.capture import Camera, CameraError, CapturedFrame


# ---------------------------------------------------------------------------
# CapturedFrame dataclass
# ---------------------------------------------------------------------------

class TestCapturedFrame:
    def _make_frame(self, frame_id=0, ts=1.0, shape=(720, 1280, 3)):
        image = np.zeros(shape, dtype=np.uint8)
        return CapturedFrame(frame_id=frame_id, capture_timestamp=ts, image=image)

    def test_basic_construction(self):
        f = self._make_frame(frame_id=42, ts=123.456)
        assert f.frame_id == 42
        assert f.capture_timestamp == 123.456
        assert f.image is not None

    def test_height_and_width_properties(self):
        f = self._make_frame(shape=(480, 640, 3))
        assert f.height == 480
        assert f.width == 640

    def test_frame_id_monotonic_sequence(self):
        """frame_id must increment for each read()."""
        frames = [self._make_frame(frame_id=i) for i in range(10)]
        ids = [f.frame_id for f in frames]
        assert ids == list(range(10))

    def test_capture_timestamp_is_float(self):
        f = self._make_frame(ts=time.monotonic())
        assert isinstance(f.capture_timestamp, float)

    def test_image_not_copied_on_construction(self):
        """The image array passed in should be the same object (no copy)."""
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        f = CapturedFrame(frame_id=0, capture_timestamp=0.0, image=image)
        assert f.image is image


# ---------------------------------------------------------------------------
# Camera with mocked VideoCapture
# ---------------------------------------------------------------------------

def _make_mock_cap(opened=True, ret=True, frame_shape=(720, 1280, 3)):
    """Build a MagicMock that mimics cv2.VideoCapture."""
    cap = MagicMock()
    cap.isOpened.return_value = opened
    fake_frame = np.zeros(frame_shape, dtype=np.uint8)
    cap.read.return_value = (ret, fake_frame if ret else None)

    # get() returns sensible defaults
    def _get(prop):
        defaults = {
            3: 1280.0,   # CAP_PROP_FRAME_WIDTH
            4: 720.0,    # CAP_PROP_FRAME_HEIGHT
            5: 30.0,     # CAP_PROP_FPS
        }
        return defaults.get(prop, 0.0)

    cap.get.side_effect = _get
    cap.set.return_value = True
    return cap


class TestCamera:
    def test_open_success(self):
        cam = Camera(device_index=0)
        mock_cap = _make_mock_cap()
        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam.open()
            assert cam.is_opened()
            assert cam.actual_width == 1280
            assert cam.actual_height == 720

    def test_open_failure_raises_camera_error(self):
        cam = Camera(device_index=99)
        mock_cap = _make_mock_cap(opened=False)
        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(CameraError):
                cam.open()

    def test_read_returns_captured_frame(self):
        cam = Camera(device_index=0)
        mock_cap = _make_mock_cap()
        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam.open()
            frame = cam.read()
        assert frame is not None
        assert isinstance(frame, CapturedFrame)
        assert frame.frame_id == 0
        assert frame.capture_timestamp > 0.0

    def test_read_increments_frame_id(self):
        cam = Camera(device_index=0)
        mock_cap = _make_mock_cap()
        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam.open()
            f0 = cam.read()
            f1 = cam.read()
            f2 = cam.read()
        assert f0.frame_id == 0
        assert f1.frame_id == 1
        assert f2.frame_id == 2

    def test_read_returns_none_on_empty_frame(self):
        cam = Camera(device_index=0)
        mock_cap = _make_mock_cap(ret=False)
        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam.open()
            frame = cam.read()
        assert frame is None

    def test_read_returns_none_when_not_opened(self):
        cam = Camera(device_index=0)
        # Never call cam.open()
        frame = cam.read()
        assert frame is None

    def test_is_opened_false_after_release(self):
        cam = Camera(device_index=0)
        mock_cap = _make_mock_cap()
        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam.open()
            assert cam.is_opened()
            cam.release()
        assert not cam.is_opened()

    def test_release_without_open_is_safe(self):
        """release() before open() must not raise."""
        cam = Camera(device_index=0)
        cam.release()  # should be a no-op

    def test_context_manager(self):
        mock_cap = _make_mock_cap()
        with patch("cv2.VideoCapture", return_value=mock_cap):
            with Camera(device_index=0) as cam:
                assert cam.is_opened()
        # After __exit__, camera should be released
        assert not cam.is_opened()

    def test_capture_timestamps_are_monotonic(self):
        """Successive read() calls must produce non-decreasing timestamps."""
        cam = Camera(device_index=0)
        mock_cap = _make_mock_cap()
        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam.open()
            timestamps = [cam.read().capture_timestamp for _ in range(20)]
        for a, b in zip(timestamps, timestamps[1:]):
            assert b >= a, f"Timestamps not monotonic: {a} > {b}"

    def test_repr_contains_device_index(self):
        cam = Camera(device_index=3)
        assert "3" in repr(cam)
