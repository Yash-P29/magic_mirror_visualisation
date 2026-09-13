"""
monitoring/metrics.py — Performance measurement for the Magic Mirror pipeline.

Three independent, reusable utilities:

  FPSCounter     — rolling-window frames-per-second measurement.
  LatencyTracker — per-frame application and perception latency statistics
                   (min / max / rolling average).

Neither class has any dependency on OpenCV, the camera layer, or MediaPipe,
so they can be unit-tested without a physical camera attached.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


# ---------------------------------------------------------------------------
# FPS Counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """Rolling-window FPS counter.

    Maintains a deque of frame arrival timestamps and computes FPS as:

        FPS = (number of timestamps in window - 1) / (newest_ts - oldest_ts)

    This gives a stable measurement that is immune to single-frame spikes
    and does not require division by a possibly-zero elapsed time.

    Args:
        window_seconds: Width of the rolling time window in seconds.
            Timestamps older than this are discarded on each tick().
            Default is 2.0 s, which balances responsiveness and stability.
    """

    def __init__(self, window_seconds: float = 2.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window: float = window_seconds
        self._timestamps: Deque[float] = deque()
        self._total_frames: int = 0

    def tick(self, timestamp: Optional[float] = None) -> None:
        """Record one frame arrival.

        Args:
            timestamp: A time.monotonic() value.  If None, the current
                monotonic time is used.  Pass an explicit value in tests.
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        self._timestamps.append(ts)
        self._total_frames += 1

        # Discard timestamps older than the window.
        cutoff = ts - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    @property
    def fps(self) -> float:
        """Current rolling FPS estimate.  Returns 0.0 if fewer than 2 frames
        have been recorded in the current window."""
        n = len(self._timestamps)
        if n < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (n - 1) / elapsed

    @property
    def total_frames(self) -> int:
        """Total number of frames ticked since construction or reset."""
        return self._total_frames

    def reset(self) -> None:
        """Clear all recorded timestamps and reset the frame counter."""
        self._timestamps.clear()
        self._total_frames = 0


# ---------------------------------------------------------------------------
# Latency Tracker
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Snapshot of latency statistics at a point in time.

    Application metrics measure everything our code does per frame (overlay
    drawing, metric updates, etc.).  Perception metrics measure the time
    spent running the MediaPipe model(s) — tracked separately so they are
    never conflated.
    """

    # --- Application timing ---
    processing_ms: float = 0.0              # App work per frame (ms)
    capture_to_processing_ms: float = 0.0  # capture_timestamp → app end (ms)
    frame_age_ms: float = 0.0               # capture_timestamp → app start (ms)
    capture_read_ms: float = 0.0           # cap.read() duration (ms)
    frame_wait_ms: float = 0.0              # wait duration for new frame (ms)
    avg_processing_ms: float = 0.0
    avg_capture_to_processing_ms: float = 0.0
    avg_frame_age_ms: float = 0.0
    min_processing_ms: float = float("inf")
    max_processing_ms: float = 0.0
    min_capture_to_processing_ms: float = float("inf")
    max_capture_to_processing_ms: float = 0.0

    # --- Perception timing (MediaPipe model inference) ---
    perception_ms: float = 0.0              # MediaPipe inference per frame (ms)
    avg_perception_ms: float = 0.0
    min_perception_ms: float = float("inf")
    max_perception_ms: float = 0.0


class LatencyTracker:
    """Tracks per-frame latency measurements and maintains rolling statistics."""

    def __init__(self, rolling_window: int = 60) -> None:
        if rolling_window < 1:
            raise ValueError("rolling_window must be at least 1")
        self._window = rolling_window

        # Store raw values (ms) for rolling average computation.
        self._proc_buf: Deque[float] = deque(maxlen=rolling_window)
        self._cap2proc_buf: Deque[float] = deque(maxlen=rolling_window)
        self._age_buf: Deque[float] = deque(maxlen=rolling_window)
        self._perc_buf: Deque[float] = deque(maxlen=rolling_window)

        self._stats = LatencyStats()

    def record(
        self,
        *,
        capture_timestamp: float,
        process_start: float,
        process_end: float,
        capture_read_ms: float = 0.0,
        frame_wait_ms: float = 0.0,
    ) -> LatencyStats:
        """Record timing for one frame and return an updated LatencyStats."""
        proc_ms = (process_end - process_start) * 1_000.0
        cap2proc_ms = (process_end - capture_timestamp) * 1_000.0
        age_ms = (process_start - capture_timestamp) * 1_000.0

        self._proc_buf.append(proc_ms)
        self._cap2proc_buf.append(cap2proc_ms)
        self._age_buf.append(age_ms)

        s = self._stats
        s.processing_ms = proc_ms
        s.capture_to_processing_ms = cap2proc_ms
        s.frame_age_ms = age_ms
        s.capture_read_ms = capture_read_ms
        s.frame_wait_ms = frame_wait_ms

        # Rolling averages over the buffer.
        s.avg_processing_ms = sum(self._proc_buf) / len(self._proc_buf)
        s.avg_capture_to_processing_ms = sum(self._cap2proc_buf) / len(self._cap2proc_buf)
        s.avg_frame_age_ms = sum(self._age_buf) / len(self._age_buf)

        # All-time min/max.
        s.min_processing_ms = min(s.min_processing_ms, proc_ms)
        s.max_processing_ms = max(s.max_processing_ms, proc_ms)
        s.min_capture_to_processing_ms = min(s.min_capture_to_processing_ms, cap2proc_ms)
        s.max_capture_to_processing_ms = max(s.max_capture_to_processing_ms, cap2proc_ms)

        return LatencyStats(
            processing_ms=s.processing_ms,
            capture_to_processing_ms=s.capture_to_processing_ms,
            frame_age_ms=s.frame_age_ms,
            capture_read_ms=s.capture_read_ms,
            frame_wait_ms=s.frame_wait_ms,
            avg_processing_ms=s.avg_processing_ms,
            avg_capture_to_processing_ms=s.avg_capture_to_processing_ms,
            avg_frame_age_ms=s.avg_frame_age_ms,
            min_processing_ms=s.min_processing_ms,
            max_processing_ms=s.max_processing_ms,
            min_capture_to_processing_ms=s.min_capture_to_processing_ms,
            max_capture_to_processing_ms=s.max_capture_to_processing_ms,
            perception_ms=s.perception_ms,
            avg_perception_ms=s.avg_perception_ms,
            min_perception_ms=s.min_perception_ms,
            max_perception_ms=s.max_perception_ms,
        )

    def record_perception(self, *, perception_ms: float) -> None:
        """Record the MediaPipe inference time for one frame."""
        self._perc_buf.append(perception_ms)

        s = self._stats
        s.perception_ms = perception_ms
        s.avg_perception_ms = sum(self._perc_buf) / len(self._perc_buf)
        s.min_perception_ms = min(s.min_perception_ms, perception_ms)
        s.max_perception_ms = max(s.max_perception_ms, perception_ms)

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._proc_buf.clear()
        self._cap2proc_buf.clear()
        self._age_buf.clear()
        self._perc_buf.clear()
        self._stats = LatencyStats()
