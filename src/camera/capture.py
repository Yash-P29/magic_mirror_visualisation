"""
camera/capture.py — Camera abstraction for the Magic Mirror pipeline.

Responsibilities:
  - Open and configure a webcam device.
  - Read frames and stamp each with a monotonic timestamp and a frame ID.
  - Expose a simple, stable interface so future pipeline stages can consume
    frames without knowing anything about the underlying capture backend.

Latency note:
  The capture_timestamp recorded in CapturedFrame reflects the moment our
  application calls cv2.VideoCapture.read() and receives a decoded frame
  buffer from OpenCV.  It does NOT capture:
    - Webcam sensor / ISP latency
    - USB transfer time
    - Kernel / V4L2 / DirectShow driver buffering
    - OpenCV's own internal frame queue (partially mitigated by setting
      CAP_PROP_BUFFERSIZE = 1 where supported)
  The delta between capture_timestamp and when your code finishes processing
  is therefore a lower bound on true photon-to-screen latency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.monitoring.metrics import FPSCounter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Desired camera configuration (applied as hints; actual values are read back)
# ---------------------------------------------------------------------------
_DESIRED_WIDTH: int = 1280
_DESIRED_HEIGHT: int = 720
_DESIRED_FPS: float = 30.0

# Buffer size of 1 minimises the number of pre-decoded frames OpenCV queues
# internally.  A smaller queue means the frame we receive on read() is more
# recent.  Not all backends honour this setting; we attempt it and move on.
_DESIRED_BUFFER_SIZE: int = 1


@dataclass
class CapturedFrame:
    """A single captured video frame with associated metadata.

    Attributes:
        frame_id: Monotonically increasing integer, starting at 0.
        capture_timestamp: time.monotonic() value recorded immediately after
            cv2.VideoCapture.read() returns.  Use this for latency deltas.
            Do NOT use it as a wall-clock time.
        image: Raw BGR frame as a NumPy ndarray (H × W × 3, dtype=uint8).
            The array is the buffer returned directly by OpenCV; it is NOT
            copied.  If downstream code needs to retain the frame after the
            next read() call, it must copy the array itself.
    """

    frame_id: int
    capture_timestamp: float  # time.monotonic()
    image: np.ndarray = field(repr=False)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


class CameraError(Exception):
    """Raised when the camera cannot be opened or encounters a fatal error."""


class Camera:
    """Thin abstraction over cv2.VideoCapture.

    Usage::

        cam = Camera(device_index=0)
        cam.open()
        try:
            while cam.is_opened():
                frame = cam.read()
                if frame is None:
                    break
                # … process frame …
        finally:
            cam.release()

    The class does not start a background thread — frames are pulled
    synchronously by the caller.  This keeps latency measurement simple and
    precise at this stage of the project.
    """

    def __init__(
        self,
        device_index: int = 0,
        backend: int = cv2.CAP_ANY,
    ) -> None:
        self._device_index = device_index
        self._backend = backend
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0

        # Actual configuration read back from the camera after opening.
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0
        self.actual_backend: int = backend
        self.actual_fourcc: str = "N/A"
        self.actual_buffersize: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera device and attempt to apply the desired configuration.

        Reads back and stores the actual width, height, and FPS that the
        driver accepted.  Logs a warning if the values differ from the
        requested configuration.

        Raises:
            CameraError: If the device cannot be opened.
        """
        logger.info("Opening camera device %d (backend=%d) …", self._device_index, self._backend)
        cap = cv2.VideoCapture(self._device_index, self._backend)

        if not cap.isOpened():
            raise CameraError(
                f"Cannot open camera device {self._device_index}. "
                "Check that a webcam is connected and not in use by another application."
            )

        # --- attempt low-latency buffer configuration -------------------
        # CAP_PROP_BUFFERSIZE is only honoured by certain backends
        # (e.g. V4L2 on Linux, MSMF on Windows).  Failure is non-fatal.
        if not cap.set(cv2.CAP_PROP_BUFFERSIZE, _DESIRED_BUFFER_SIZE):
            logger.debug(
                "CAP_PROP_BUFFERSIZE not supported by this backend — "
                "driver-level frame buffering is not controllable."
            )

        # --- request resolution and frame rate --------------------------
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, _DESIRED_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _DESIRED_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, _DESIRED_FPS)

        # --- read back actual configuration -----------------------------
        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = cap.get(cv2.CAP_PROP_FPS)
        self.actual_backend = self._backend

        try:
            fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
            if fourcc_val > 0:
                chars = [chr((fourcc_val >> (8 * i)) & 0xFF) for i in range(4)]
                self.actual_fourcc = "".join(chars)
            else:
                self.actual_fourcc = "N/A"
        except Exception:
            self.actual_fourcc = "N/A"

        try:
            buf_val = int(cap.get(cv2.CAP_PROP_BUFFERSIZE))
            self.actual_buffersize = buf_val if buf_val >= 0 else None
        except Exception:
            self.actual_buffersize = None

        self._cap = cap
        self._frame_id = 0

        # Report what we actually got.
        if self.actual_width != _DESIRED_WIDTH or self.actual_height != _DESIRED_HEIGHT:
            logger.warning(
                "Requested %dx%d but camera reported %dx%d.",
                _DESIRED_WIDTH,
                _DESIRED_HEIGHT,
                self.actual_width,
                self.actual_height,
            )
        if self.actual_fps and abs(self.actual_fps - _DESIRED_FPS) > 1.0:
            logger.warning(
                "Requested %.0f FPS but camera reported %.1f FPS.",
                _DESIRED_FPS,
                self.actual_fps,
            )

        logger.info(
            "Camera opened: %dx%d @ %.1f FPS (backend=%d, FOURCC=%s, buffer_size=%s).",
            self.actual_width,
            self.actual_height,
            self.actual_fps,
            self.actual_backend,
            self.actual_fourcc,
            str(self.actual_buffersize),
        )

    def read(self) -> Optional[CapturedFrame]:
        """Capture the next frame from the camera.

        Records a monotonic timestamp immediately after the cv2.read() call
        returns so the timestamp is as close as possible to the moment the
        frame data arrived from the driver.

        Returns:
            A CapturedFrame, or None if the camera returned an empty frame
            (which may indicate a temporary glitch or disconnection).
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, image = self._cap.read()
        # Timestamp recorded immediately after read() so the delta to
        # "processing complete" is as tight as possible.
        capture_ts = time.monotonic()

        if not ret or image is None or image.size == 0:
            logger.warning("Frame %d: empty or invalid frame received.", self._frame_id)
            return None

        frame = CapturedFrame(
            frame_id=self._frame_id,
            capture_timestamp=capture_ts,
            image=image,  # no copy — caller must copy if retention is needed
        )
        self._frame_id += 1
        return frame

    def release(self) -> None:
        """Release the camera device and associated resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera device %d released.", self._device_index)

    def is_opened(self) -> bool:
        """Return True if the camera device is currently open."""
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def __repr__(self) -> str:
        status = "open" if self.is_opened() else "closed"
        return (
            f"Camera(device={self._device_index}, "
            f"status={status}, "
            f"resolution={self.actual_width}x{self.actual_height}, "
            f"fps={self.actual_fps:.1f}, "
            f"fourcc={self.actual_fourcc})"
        )


# ---------------------------------------------------------------------------
# AsyncCameraReader (Threaded Latest-Frame-Wins Producer)
# ---------------------------------------------------------------------------

class AsyncCameraReader:
    """Threaded camera reader decoupling hardware capture from downstream processing.

    Continuously pulls frames from the camera in a dedicated background thread.
    Uses a thread-safe 1-element latest-frame slot (capacity 1).
    When a new frame arrives, if the unread slot is occupied, the older frame
    is discarded and counted as dropped.

    Metrics tracked:
        - captured_frames: total frames captured by camera thread
        - consumed_frames: total frames retrieved by processing thread
        - dropped_frames: total frames replaced/overwritten before retrieval
        - capture_fps: rolling FPS of the hardware capture loop
    """

    def __init__(self, camera: Camera) -> None:
        import threading

        self._camera = camera
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None

        self._running: bool = False
        self._latest_frame: Optional[CapturedFrame] = None
        self._has_new_frame: bool = False

        # Counters & Telemetry
        self._captured_frames: int = 0
        self._consumed_frames: int = 0
        self._dropped_frames: int = 0
        self._last_read_duration_ms: float = 0.0
        self._capture_fps_counter = FPSCounter(window_seconds=2.0)

    @property
    def camera(self) -> Camera:
        return self._camera

    @property
    def captured_frames(self) -> int:
        with self._lock:
            return self._captured_frames

    @property
    def consumed_frames(self) -> int:
        with self._lock:
            return self._consumed_frames

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_frames

    @property
    def currently_buffered(self) -> int:
        with self._lock:
            return 1 if self._has_new_frame else 0

    @property
    def invariant_valid(self) -> bool:
        with self._lock:
            return self._captured_frames == (
                self._consumed_frames + self._dropped_frames + (1 if self._has_new_frame else 0)
            )

    @property
    def drop_ratio(self) -> float:
        with self._lock:
            if self._captured_frames == 0:
                return 0.0
            return self._dropped_frames / self._captured_frames

    @property
    def capture_fps(self) -> float:
        with self._lock:
            return self._capture_fps_counter.fps

    @property
    def last_read_duration_ms(self) -> float:
        with self._lock:
            return self._last_read_duration_ms

    def start(self) -> None:
        """Start the background capture thread."""
        import threading

        with self._lock:
            if self._running:
                return
            if not self._camera.is_opened():
                self._camera.open()
            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="AsyncCameraThread",
                daemon=True,
            )
            self._thread.start()
            logger.info("AsyncCameraReader background thread started.")

    def _capture_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break

            t0 = time.monotonic()
            frame = self._camera.read()
            t1 = time.monotonic()
            read_ms = (t1 - t0) * 1000.0

            with self._lock:
                if not self._running:
                    break

                self._last_read_duration_ms = read_ms

                if read_ms > 100:
                    logger.warning("Camera read took %.1f ms", read_ms)

                if frame is not None:
                    self._captured_frames += 1
                    self._capture_fps_counter.tick(frame.capture_timestamp)

                    if self._has_new_frame:
                        self._dropped_frames += 1

                    self._latest_frame = frame
                    self._has_new_frame = True
                    self._condition.notify_all()
                else:
                    # Camera read failed or returned empty; yield slightly to prevent CPU spin
                    time.sleep(0.005)

    def read_latest(self, timeout: float = 0.5) -> Optional[CapturedFrame]:
        """Retrieve the newest available frame, blocking up to timeout if none.

        Returns:
            The newest CapturedFrame, or None if timeout expired or stopped.
        """
        with self._lock:
            if not self._has_new_frame and self._running:
                self._condition.wait(timeout=timeout)

            if not self._has_new_frame or self._latest_frame is None:
                return None

            frame = self._latest_frame
            self._has_new_frame = False
            self._consumed_frames += 1
            return frame

    def stop(self) -> None:
        """Stop the background capture thread and release camera resources cleanly."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._condition.notify_all()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        self._camera.release()
        logger.info("AsyncCameraReader stopped.")

    def __enter__(self) -> "AsyncCameraReader":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

