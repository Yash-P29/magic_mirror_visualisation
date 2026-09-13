"""
scripts/camera_matrix_diagnostic.py — Diagnostic script for Milestone 2A.1.1.

Tests 6 standard configurations:
  A: DSHOW @ 1280x720
  B: MSMF  @ 1280x720
  C: ANY   @ 1280x720
  D: DSHOW @ 640x480
  E: MSMF  @ 640x480
  F: ANY   @ 640x480
Plus extra configurations:
  G: DSHOW @ 960x540
  H: Default Native Configuration (no explicit width/height/FPS/backend)

For each configuration:
  1. Runs a raw capture test (no perception) for 10 seconds.
  2. Runs an async perception capture test for 8 seconds.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, List

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2

from src.camera import AsyncCameraReader, Camera
from src.monitoring import FPSCounter, LatencyTracker
from src.perception import HumanPerceptionPipeline

CONFIGS = [
    ("A", "DSHOW", cv2.CAP_DSHOW, 1280, 720),
    ("B", "MSMF",  cv2.CAP_MSMF,  1280, 720),
    ("C", "ANY",   cv2.CAP_ANY,   1280, 720),
    ("D", "DSHOW", cv2.CAP_DSHOW, 640,  480),
    ("E", "MSMF",  cv2.CAP_MSMF,  640,  480),
    ("F", "ANY",   cv2.CAP_ANY,   640,  480),
    ("G", "DSHOW", cv2.CAP_DSHOW, 960,  540),
]


def run_raw_diagnostic(
    backend_flag: int,
    req_w: int,
    req_h: int,
    duration: float = 10.0,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(0, backend_flag)
    if not cap.isOpened():
        return {"opened": False, "error": "Camera failed to open"}

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    cap.set(cv2.CAP_PROP_FPS, 30.0)

    act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    act_fps = cap.get(cv2.CAP_PROP_FPS)

    # Warmup 5 frames
    for _ in range(5):
        cap.read()

    timestamps: List[float] = []
    read_durations: List[float] = []
    successes = 0
    failures = 0

    t_start = time.monotonic()
    while time.monotonic() - t_start < duration:
        t0 = time.monotonic()
        ret, frame = cap.read()
        t1 = time.monotonic()

        read_durations.append((t1 - t0) * 1000.0)
        if ret and frame is not None and frame.size > 0:
            successes += 1
            timestamps.append(t1)
        else:
            failures += 1

    cap.release()

    elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    measured_fps = (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    intervals = [
        (timestamps[i] - timestamps[i - 1]) * 1000.0
        for i in range(1, len(timestamps))
    ]

    return {
        "opened": True,
        "actual_width": act_w,
        "actual_height": act_h,
        "driver_fps": act_fps,
        "successful_reads": successes,
        "failed_reads": failures,
        "measured_fps": measured_fps,
        "intervals": {
            "avg": statistics.mean(intervals) if intervals else 0.0,
            "median": statistics.median(intervals) if intervals else 0.0,
            "min": min(intervals) if intervals else 0.0,
            "max": max(intervals) if intervals else 0.0,
        },
        "read_durations": {
            "avg": statistics.mean(read_durations) if read_durations else 0.0,
            "median": statistics.median(read_durations) if read_durations else 0.0,
            "min": min(read_durations) if read_durations else 0.0,
            "max": max(read_durations) if read_durations else 0.0,
        },
    }


def run_default_native_diagnostic(duration: float = 10.0) -> Dict[str, Any]:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"opened": False}

    act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    act_fps = cap.get(cv2.CAP_PROP_FPS)

    for _ in range(5):
        cap.read()

    timestamps: List[float] = []
    read_durations: List[float] = []
    successes = 0
    failures = 0

    t_start = time.monotonic()
    while time.monotonic() - t_start < duration:
        t0 = time.monotonic()
        ret, frame = cap.read()
        t1 = time.monotonic()

        read_durations.append((t1 - t0) * 1000.0)
        if ret and frame is not None and frame.size > 0:
            successes += 1
            timestamps.append(t1)
        else:
            failures += 1

    cap.release()

    elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    measured_fps = (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0
    intervals = [
        (timestamps[i] - timestamps[i - 1]) * 1000.0
        for i in range(1, len(timestamps))
    ]

    return {
        "opened": True,
        "actual_width": act_w,
        "actual_height": act_h,
        "driver_fps": act_fps,
        "successful_reads": successes,
        "failed_reads": failures,
        "measured_fps": measured_fps,
        "intervals": {
            "avg": statistics.mean(intervals) if intervals else 0.0,
            "median": statistics.median(intervals) if intervals else 0.0,
            "min": min(intervals) if intervals else 0.0,
            "max": max(intervals) if intervals else 0.0,
        },
        "read_durations": {
            "avg": statistics.mean(read_durations) if read_durations else 0.0,
            "median": statistics.median(read_durations) if read_durations else 0.0,
            "min": min(read_durations) if read_durations else 0.0,
            "max": max(read_durations) if read_durations else 0.0,
        },
    }


def run_async_diagnostic(
    backend_flag: int,
    req_w: int,
    req_h: int,
    duration: float = 8.0,
) -> Dict[str, Any]:
    cam = Camera(device_index=0, backend=backend_flag)
    try:
        cam.open()
    except Exception as e:
        return {"opened": False, "error": str(e)}

    cam._cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cam._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    cam._cap.set(cv2.CAP_PROP_FPS, 30.0)

    reader = AsyncCameraReader(cam)
    reader.start()

    pipeline = HumanPerceptionPipeline()
    try:
        pipeline.open()
    except Exception:
        pipeline = None

    time.sleep(0.5)

    fps_counter = FPSCounter(window_seconds=2.0)
    lat_tracker = LatencyTracker(rolling_window=60)

    t_start = time.monotonic()
    while time.monotonic() - t_start < duration:
        frame = reader.read_latest(timeout=0.1)
        if frame is None:
            continue

        p_start = time.monotonic()
        fps_counter.tick(frame.capture_timestamp)
        if pipeline:
            p_res = pipeline.process(frame)
            p_ms = p_res.perception_ms
        else:
            p_ms = 0.0
        p_end = time.monotonic()

        lat_tracker.record_perception(perception_ms=p_ms)
        lat_tracker.record(
            capture_timestamp=frame.capture_timestamp,
            process_start=p_start,
            process_end=p_end,
        )

    if pipeline:
        pipeline.close()
    reader.stop()

    s = lat_tracker._stats
    return {
        "captured": reader.captured_frames,
        "consumed": reader.consumed_frames,
        "dropped": reader.dropped_frames,
        "drop_ratio": reader.drop_ratio,
        "app_fps": fps_counter.fps,
        "avg_frame_age_ms": s.avg_frame_age_ms,
        "avg_cap2proc_ms": s.avg_capture_to_processing_ms,
        "avg_perception_ms": s.avg_perception_ms,
    }


def main():
    print("=== STARTING MILESTONE 2A.1.1 CAMERA DIAGNOSTIC MATRIX ===")
    results = {}

    for tag, b_name, b_flag, w, h in CONFIGS:
        print(f"\n--- Testing Config {tag}: {b_name} @ {w}x{h} ---")
        raw_res = run_raw_diagnostic(b_flag, w, h, duration=10.0)
        print(f"Raw Test -> Opened: {raw_res.get('opened')}, Res: {raw_res.get('actual_width')}x{raw_res.get('actual_height')}, Driver FPS: {raw_res.get('driver_fps')}, Measured FPS: {raw_res.get('measured_fps'):.2f}, Avg Read ms: {raw_res.get('read_durations', {}).get('avg'):.1f}, Median Interval ms: {raw_res.get('intervals', {}).get('median'):.1f}")

        async_res = {}
        if raw_res.get("opened"):
            async_res = run_async_diagnostic(b_flag, w, h, duration=8.0)
            print(f"Async Test -> Captured: {async_res.get('captured')}, Consumed: {async_res.get('consumed')}, Dropped: {async_res.get('dropped')}, App FPS: {async_res.get('app_fps'):.2f}, Frame Age ms: {async_res.get('avg_frame_age_ms'):.1f}")

        results[tag] = {
            "backend": b_name,
            "req_res": f"{w}x{h}",
            "raw": raw_res,
            "async": async_res,
        }

    print("\n--- Testing Native Default Configuration (Config H) ---")
    native_res = run_default_native_diagnostic(duration=10.0)
    print(f"Native Default -> Res: {native_res.get('actual_width')}x{native_res.get('actual_height')}, Driver FPS: {native_res.get('driver_fps')}, Measured FPS: {native_res.get('measured_fps'):.2f}, Avg Read ms: {native_res.get('read_durations', {}).get('avg'):.1f}, Median Interval ms: {native_res.get('intervals', {}).get('median'):.1f}")
    results["H"] = {
        "backend": "NATIVE_DEFAULT",
        "req_res": "DEFAULT",
        "raw": native_res,
    }

    with open("camera_diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDiagnostic matrix complete. Saved to camera_diagnostic_results.json.")


if __name__ == "__main__":
    main()
