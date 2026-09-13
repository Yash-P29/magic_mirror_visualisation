"""
tests/test_metrics.py — Unit tests for FPSCounter and LatencyTracker.

These tests do NOT require a physical camera.  All timing values are injected
explicitly so the tests are deterministic and fast.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.monitoring.metrics import FPSCounter, LatencyTracker


# ---------------------------------------------------------------------------
# FPSCounter tests
# ---------------------------------------------------------------------------

class TestFPSCounter:
    def test_initial_fps_is_zero(self):
        counter = FPSCounter()
        assert counter.fps == 0.0

    def test_single_tick_fps_is_zero(self):
        """Need at least 2 timestamps to compute FPS."""
        counter = FPSCounter()
        counter.tick(timestamp=0.0)
        assert counter.fps == 0.0

    def test_fps_two_frames_at_30hz(self):
        """Two frames 1/30 s apart should give FPS ≈ 30."""
        counter = FPSCounter(window_seconds=2.0)
        counter.tick(timestamp=0.0)
        counter.tick(timestamp=1 / 30)
        assert abs(counter.fps - 30.0) < 0.1

    def test_fps_uniform_30hz_many_frames(self):
        """60 uniformly-spaced frames at 30 FPS should give FPS ≈ 30."""
        counter = FPSCounter(window_seconds=10.0)
        interval = 1 / 30
        for i in range(60):
            counter.tick(timestamp=i * interval)
        assert abs(counter.fps - 30.0) < 0.5

    def test_fps_window_discards_old_timestamps(self):
        """Timestamps outside the window should not affect the FPS calculation."""
        counter = FPSCounter(window_seconds=1.0)
        # Lay down some old frames at t=0..0.5
        for i in range(10):
            counter.tick(timestamp=i * 0.05)
        # Now add new frames at 20 FPS (at t=10 s and beyond)
        for i in range(5):
            counter.tick(timestamp=10.0 + i * 0.05)
        # Only the recent window should matter; FPS should be ≈ 20
        assert abs(counter.fps - 20.0) < 1.0

    def test_total_frames_increments(self):
        counter = FPSCounter()
        for i in range(7):
            counter.tick(timestamp=float(i))
        assert counter.total_frames == 7

    def test_reset_clears_state(self):
        counter = FPSCounter()
        counter.tick(timestamp=0.0)
        counter.tick(timestamp=1.0)
        counter.reset()
        assert counter.fps == 0.0
        assert counter.total_frames == 0

    def test_negative_window_raises(self):
        with pytest.raises(ValueError):
            FPSCounter(window_seconds=-1.0)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError):
            FPSCounter(window_seconds=0.0)

    def test_tick_uses_current_time_when_no_timestamp(self):
        """Verify tick() with no argument doesn't crash and produces sane FPS."""
        counter = FPSCounter(window_seconds=2.0)
        t0 = time.monotonic()
        for _ in range(5):
            counter.tick()
        elapsed = time.monotonic() - t0
        # FPS should be very high since frames were tight; just check it's > 0
        assert counter.fps > 0.0 or elapsed == 0.0


# ---------------------------------------------------------------------------
# LatencyTracker tests
# ---------------------------------------------------------------------------

class TestLatencyTracker:
    def _record(self, tracker, *, capture_ts, start, end):
        return tracker.record(
            capture_timestamp=capture_ts,
            process_start=start,
            process_end=end,
        )

    def test_processing_ms_calculation(self):
        tracker = LatencyTracker()
        stats = self._record(tracker, capture_ts=0.0, start=0.010, end=0.014)
        assert abs(stats.processing_ms - 4.0) < 1e-6

    def test_capture_to_processing_ms_calculation(self):
        tracker = LatencyTracker()
        stats = self._record(tracker, capture_ts=0.000, start=0.005, end=0.012)
        # capture→processing = (0.012 - 0.000) * 1000 = 12 ms
        assert abs(stats.capture_to_processing_ms - 12.0) < 1e-6

    def test_min_max_tracking(self):
        tracker = LatencyTracker()
        self._record(tracker, capture_ts=0.0, start=0.0, end=0.010)  # 10 ms proc
        self._record(tracker, capture_ts=0.0, start=0.0, end=0.002)  # 2 ms proc
        self._record(tracker, capture_ts=0.0, start=0.0, end=0.020)  # 20 ms proc
        stats = self._record(tracker, capture_ts=0.0, start=0.0, end=0.005)  # 5 ms

        assert abs(stats.min_processing_ms - 2.0) < 1e-6
        assert abs(stats.max_processing_ms - 20.0) < 1e-6

    def test_rolling_average(self):
        tracker = LatencyTracker(rolling_window=3)
        # Record processing durations: 10 ms, 20 ms, 30 ms
        for end in (0.010, 0.020, 0.030):
            stats = self._record(tracker, capture_ts=0.0, start=0.0, end=end)
        # Average of [10, 20, 30] = 20 ms
        assert abs(stats.avg_processing_ms - 20.0) < 1e-6

    def test_rolling_window_evicts_old_values(self):
        """With window=2, the oldest value should be evicted."""
        tracker = LatencyTracker(rolling_window=2)
        self._record(tracker, capture_ts=0.0, start=0.0, end=0.100)  # 100 ms
        self._record(tracker, capture_ts=0.0, start=0.0, end=0.010)  # 10 ms
        stats = self._record(tracker, capture_ts=0.0, start=0.0, end=0.020)  # 20 ms
        # Window holds last 2: [10, 20] → avg = 15 ms  (100 ms evicted)
        assert abs(stats.avg_processing_ms - 15.0) < 1e-6

    def test_reset_clears_stats(self):
        tracker = LatencyTracker()
        self._record(tracker, capture_ts=0.0, start=0.0, end=0.010)
        tracker.reset()
        # After reset, a new record should start fresh
        stats = self._record(tracker, capture_ts=0.0, start=0.0, end=0.005)
        assert abs(stats.avg_processing_ms - 5.0) < 1e-6

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            LatencyTracker(rolling_window=0)

    def test_processing_ms_always_non_negative(self):
        """Processing duration should always be ≥ 0."""
        tracker = LatencyTracker()
        stats = self._record(tracker, capture_ts=1.0, start=2.0, end=2.003)
        assert stats.processing_ms >= 0.0

    def test_capture_to_processing_ms_always_non_negative(self):
        """capture_to_processing should be ≥ 0 when timestamps are monotonic."""
        tracker = LatencyTracker()
        stats = self._record(tracker, capture_ts=1.0, start=1.001, end=1.005)
        assert stats.capture_to_processing_ms >= 0.0


# ---------------------------------------------------------------------------
# Monotonicity test (no camera needed)
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_monotonic_clock_is_monotonic(self):
        """Sanity check that time.monotonic() never goes backward."""
        samples = [time.monotonic() for _ in range(100)]
        for a, b in zip(samples, samples[1:]):
            assert b >= a, f"Monotonicity violated: {a} > {b}"
