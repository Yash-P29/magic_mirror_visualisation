"""
main.py — Magic Mirror Milestone 2A.1: Live Camera + Decoupled Fast Perception.

Entry point for the camera + perception pipeline.  Responsibilities:
  1. Open webcam with specified backend (DirectShow or MSMF).
  2. Initialise AsyncCameraReader (background capture thread, latest-frame slot).
  3. Pull newest frame from AsyncCameraReader.
  4. Run Fast Perception on each frame (synchronous VIDEO mode).
  5. Visualise pose skeleton and hand landmarks on live frame.
  6. Measure and display separate app, perception, capture-to-processing, frame age, and dropped frame metrics.
  7. Handle quit signals and camera/perception errors cleanly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

# ---------------------------------------------------------------------------
# Path fix so this script runs from the repo root without package install.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.camera.capture import AsyncCameraReader, Camera, CameraError
from src.monitoring.metrics import FPSCounter, LatencyStats, LatencyTracker
from src.perception.models import (
    HandNearObject,
    HandOverlappingObject,
    HandPointingAtObject,
    PerceptionResult,
    SceneState,
)
from src.perception.scene import SceneBuilder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("magic_mirror.main")

# ---------------------------------------------------------------------------
# Overlay / display configuration
# ---------------------------------------------------------------------------
_WINDOW_NAME = "Magic Mirror — Fast Perception (Decoupled)"

_FONT            = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE      = 0.55
_FONT_THICKNESS  = 1
_LINE_TYPE       = cv2.LINE_AA

# Colours (BGR)
_COLOR_GOOD       = (100, 230, 100)   # green — FPS healthy
_COLOR_WARN       = (60,  180, 240)   # amber — FPS low
_COLOR_TEXT       = (220, 220, 220)   # near-white
_COLOR_SHADOW     = (10,  10,  10)    # shadow
_COLOR_POSE       = (255, 200,  50)   # yellow — pose connections
_COLOR_POSE_LM    = (50,  200, 255)   # cyan   — pose landmark dots
_COLOR_HAND_L     = (150, 255, 150)   # light-green  — left hand
_COLOR_HAND_R     = (150, 150, 255)   # light-blue   — right hand
_COLOR_PERC       = (200, 160, 255)   # lavender — perception timing line

_PANEL_X         = 10
_PANEL_Y_START   = 24
_LINE_HEIGHT     = 20

# ---------------------------------------------------------------------------
# MediaPipe drawing connections
# ---------------------------------------------------------------------------
_POSE_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (27, 31), (28, 32),
]

_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),
    (0, 5),  (5, 6),  (6, 7),  (7, 8),
    (0, 9),  (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9),  (9, 13), (13, 17),
]

def _put_shadowed_text(
    frame,
    text: str,
    x: int,
    y: int,
    color: tuple,
    scale: float = _FONT_SCALE,
    thickness: int = _FONT_THICKNESS,
) -> None:
    cv2.putText(frame, text, (x + 1, y + 1), _FONT, scale, _COLOR_SHADOW, thickness + 1, _LINE_TYPE)
    cv2.putText(frame, text, (x, y),          _FONT, scale, color,         thickness,     _LINE_TYPE)


def _draw_overlay(
    frame,
    cap_fps: float,
    proc_fps: float,
    frame_id: int,
    stats: LatencyStats,
    state: Optional[SceneState],
    perception_enabled: bool,
    captured: int = 0,
    dropped: int = 0,
    drop_ratio: float = 0.0,
) -> None:
    num_lines = 16 if perception_enabled else 9
    panel_h = _LINE_HEIGHT * num_lines + 10
    panel_w = 320
    frame[:panel_h, :panel_w] = (frame[:panel_h, :panel_w] * 0.40).astype(frame.dtype)

    y = _PANEL_Y_START

    fps_color = _COLOR_GOOD if cap_fps >= 25 else _COLOR_WARN
    _put_shadowed_text(frame, f"Cap FPS:    {cap_fps:5.1f} | Proc: {proc_fps:4.1f}", _PANEL_X, y, fps_color)
    y += _LINE_HEIGHT

    _put_shadowed_text(frame, f"Frame:   {frame_id:7d}", _PANEL_X, y, _COLOR_TEXT)
    y += _LINE_HEIGHT

    _put_shadowed_text(frame, f"Cap Read:    {stats.capture_read_ms:5.1f} ms", _PANEL_X, y, _COLOR_TEXT)
    y += _LINE_HEIGHT

    _put_shadowed_text(frame, f"Consumer Wait:{stats.consumer_wait_ms:5.1f} ms", _PANEL_X, y, _COLOR_TEXT)
    y += _LINE_HEIGHT

    _put_shadowed_text(frame, f"Frame Age:   {stats.frame_age_ms:5.1f} ms (avg {stats.avg_frame_age_ms:5.1f})", _PANEL_X, y, _COLOR_TEXT)
    y += _LINE_HEIGHT

    _put_shadowed_text(frame, f"App Work:    {stats.processing_ms:5.1f} ms", _PANEL_X, y, _COLOR_TEXT)
    y += _LINE_HEIGHT

    if perception_enabled:
        perc_color = _COLOR_PERC if stats.total_perception_ms < 50 else _COLOR_WARN
        _put_shadowed_text(frame, f"MediaPipe:   {stats.perception_ms:5.1f} ms", _PANEL_X, y, perc_color)
        y += _LINE_HEIGHT
        _put_shadowed_text(frame, f"YOLO:        {stats.yolo_ms:5.1f} ms", _PANEL_X, y, perc_color)
        y += _LINE_HEIGHT
        _put_shadowed_text(frame, f"Tracking:    {stats.tracking_ms:5.1f} ms", _PANEL_X, y, perc_color)
        y += _LINE_HEIGHT
        _put_shadowed_text(frame, f"Total Perc:  {stats.total_perception_ms:5.1f} ms", _PANEL_X, y, perc_color)
        y += _LINE_HEIGHT

    _put_shadowed_text(frame, f"Cap\u2192Total:   {stats.capture_to_processing_ms:5.1f} ms", _PANEL_X, y, _COLOR_TEXT)
    y += _LINE_HEIGHT

    _put_shadowed_text(frame, f"Drops:   {dropped}/{captured} ({drop_ratio*100:4.1f}%)", _PANEL_X, y, _COLOR_TEXT)
    y += _LINE_HEIGHT

    pose_str = "no"
    hands_str = "none"
    objects_str = "0"
    activity_str = "none"
    if state is not None:
        if state.human_perception:
            pose_str = "YES" if state.human_perception.has_pose else "no"
            num_hands = state.human_perception.num_hands
            hands_str = str(num_hands) if num_hands > 0 else "none"
        objects_str = str(len(state.tracked_objects))
        
        if state.activity:
            lg = state.activity.left_hand_gesture.value
            rg = state.activity.right_hand_gesture.value
            activity_str = f"L:{lg[:4]} R:{rg[:4]}"
        
    _put_shadowed_text(frame, f"Pose:{pose_str:>3}  Hands:{hands_str:>4}  Objs:{objects_str:>3}", _PANEL_X, y, _COLOR_TEXT)
    y += _LINE_HEIGHT
    _put_shadowed_text(frame, f"Gesture: {activity_str}", _PANEL_X, y, _COLOR_TEXT)


def _draw_perception(frame, state: SceneState) -> None:
    h, w = frame.shape[0], frame.shape[1]
    
    # 1. Draw human pose and hands
    if state.human_perception:
        result = state.human_perception
        if result.pose:
            for idx1, idx2 in _POSE_CONNECTIONS:
                lm1 = result.pose.get(idx1)
                lm2 = result.pose.get(idx2)
                if lm1 and lm2 and (lm1.visibility is None or lm1.visibility > 0.5) and (lm2.visibility is None or lm2.visibility > 0.5):
                    pt1 = (int(lm1.x * w), int(lm1.y * h))
                    pt2 = (int(lm2.x * w), int(lm2.y * h))
                    cv2.line(frame, pt1, pt2, _COLOR_POSE, 2, _LINE_TYPE)
            for lm in result.pose.landmarks:
                if lm.visibility is None or lm.visibility > 0.5:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame, pt, 3, _COLOR_POSE_LM, -1, _LINE_TYPE)

        for hand in result.hands:
            color = _COLOR_HAND_L if hand.is_left else _COLOR_HAND_R
            for idx1, idx2 in _HAND_CONNECTIONS:
                lm1 = hand.get(idx1)
                lm2 = hand.get(idx2)
                if lm1 and lm2:
                    pt1 = (int(lm1.x * w), int(lm1.y * h))
                    pt2 = (int(lm2.x * w), int(lm2.y * h))
                    cv2.line(frame, pt1, pt2, color, 1, _LINE_TYPE)
            for lm in hand.landmarks:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, pt, 2, color, -1, _LINE_TYPE)

    # 2. Draw objects
    for obj in state.tracked_objects:
        x1 = int(obj.bbox.x_min * w)
        y1 = int(obj.bbox.y_min * h)
        x2 = int(obj.bbox.x_max * w)
        y2 = int(obj.bbox.y_max * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 50), 2)
        cv2.putText(frame, f"{obj.class_name} {obj.track_id}", (x1, max(y1 - 5, 10)), _FONT, 0.5, (255, 255, 255), 1, _LINE_TYPE)
        
    # 3. Draw relationships
    for rel in state.relationships:
        # Find the hand position
        hand_pt = None
        if state.human_perception:
            for hand in state.human_perception.hands:
                if hand.is_left == rel.hand_is_left and hand.index_tip:
                    hand_pt = (int(hand.index_tip.x * w), int(hand.index_tip.y * h))
                    break
                    
        # Find the object position
        obj_pt = None
        obj_bbox = None
        for obj in state.tracked_objects:
            if obj.track_id == rel.object_id:
                obj_pt = (int(obj.center[0] * w), int(obj.center[1] * h))
                obj_bbox = obj.bbox
                break
                
        if hand_pt and obj_pt and obj_bbox:
            color = _COLOR_HAND_L if rel.hand_is_left else _COLOR_HAND_R
            
            if isinstance(rel, HandPointingAtObject):
                cv2.line(frame, hand_pt, obj_pt, color, 2, _LINE_TYPE)
                cv2.circle(frame, obj_pt, 5, color, -1)
            elif isinstance(rel, (HandNearObject, HandOverlappingObject)):
                # Highlight the object box
                x1 = int(obj_bbox.x_min * w)
                y1 = int(obj_bbox.y_min * h)
                x2 = int(obj_bbox.x_max * w)
                y2 = int(obj_bbox.y_max * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)


def _print_startup_info(cam: Camera, backend_name: str, perception_enabled: bool, async_mode: bool) -> None:
    logger.info("============================================================")
    logger.info("  Magic Mirror — Milestone 2A.1.2: Hardened Capture & Telemetry")
    logger.info("============================================================")
    logger.info("Camera resolution (reported) : %dx%d", cam.actual_width, cam.actual_height)
    logger.info("Camera FPS (reported driver) : %.1f", cam.actual_fps)
    logger.info("Capture Backend (flag/fourcc): %s (flag=%d, FOURCC=%s, buffer_size=%s)",
                backend_name, cam.actual_backend, cam.actual_fourcc, str(cam.actual_buffersize))
    logger.info("Decoupled Async              : %s", "YES (Threaded Latest-Frame)" if async_mode else "NO (Sync Direct)")
    logger.info("Perception enabled           : %s", "YES" if perception_enabled else "NO")
    logger.info("Press 'q' or 'ESC' to quit.")
    logger.info("------------------------------------------------------------")


def _print_summary(fps_counter: FPSCounter, lat_tracker: LatencyTracker, reader: Optional[AsyncCameraReader] = None) -> None:
    s = lat_tracker._stats
    logger.info("============================================================")
    logger.info("  Session Summary")
    logger.info("============================================================")
    logger.info("Total frames processed : %d", fps_counter.total_frames)
    logger.info("Processing rolling FPS : %.2f FPS", fps_counter.fps)
    if reader is not None:
        logger.info("Capture rolling FPS    : %.2f FPS", reader.capture_fps)
        logger.info("Captured frames        : %d", reader.captured_frames)
        logger.info("Consumed frames        : %d", reader.consumed_frames)
        logger.info("Dropped frames         : %d", reader.dropped_frames)
        logger.info("Drop ratio             : %.1f%%", reader.drop_ratio * 100.0)
        logger.info("Invariant valid        : %s", "YES" if reader.invariant_valid else "NO")
    logger.info("Avg capture read ms    : %.2f ms", s.avg_capture_read_ms)
    logger.info("Avg consumer wait ms   : %.2f ms", s.avg_consumer_wait_ms)
    logger.info("Avg app work duration  : %.2f ms (min %.2f, max %.2f)", s.avg_processing_ms, s.min_processing_ms, s.max_processing_ms)
    logger.info("Avg perception duration: %.2f ms (min %.2f, max %.2f)", s.avg_perception_ms, s.min_perception_ms, s.max_perception_ms)
    logger.info("Avg frame age at start : %.2f ms", s.avg_frame_age_ms)
    logger.info("Avg capture→proc total : %.2f ms (min %.2f, max %.2f)", s.avg_capture_to_processing_ms, s.min_capture_to_processing_ms, s.max_capture_to_processing_ms)
    logger.info("============================================================")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Magic Mirror Milestone 2A.1 — Decoupled Fast Perception",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0, metavar="N", help="Camera device index.")
    parser.add_argument("--backend", choices=["dshow", "msmf", "any"], default="dshow", help="OpenCV capture backend.")
    parser.add_argument("--sync", action="store_true", default=False, help="Run in legacy synchronous mode (no async producer thread).")
    parser.add_argument("--no-flip", action="store_true", default=False, help="Disable horizontal mirror-flip.")
    parser.add_argument("--no-perception", action="store_true", default=False, help="Disable MediaPipe.")
    parser.add_argument("--model", type=Path, default=None, help="Path to holistic_landmarker.task.")
    parser.add_argument("--timeout", type=float, default=0.0, help="Stop after N seconds (0=run forever).")
    return parser.parse_args()


def run(
    device_index: int = 0,
    backend_str: str = "dshow",
    async_mode: bool = True,
    mirror_flip: bool = True,
    perception_enabled: bool = True,
    model_path: Optional[Path] = None,
    timeout: float = 0.0,
) -> int:
    backend_map = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }
    backend_flag = backend_map.get(backend_str.lower(), cv2.CAP_DSHOW)

    cam = Camera(device_index=device_index, backend=backend_flag)
    try:
        cam.open()
    except CameraError as exc:
        logger.error("Camera error: %s", exc)
        return 1

    reader: Optional[AsyncCameraReader] = None
    if async_mode:
        reader = AsyncCameraReader(cam)
        reader.start()

    pipeline = None
    if perception_enabled:
        pipeline = SceneBuilder(yolo_interval_ms=100.0)
        try:
            pipeline.open()
        except Exception as exc:
            logger.error("Perception init error: %s", exc)
            pipeline = None
            perception_enabled = False

    _print_startup_info(cam, backend_str.upper(), perception_enabled, async_mode)

    fps_counter = FPSCounter(window_seconds=2.0)
    lat_tracker = LatencyTracker(rolling_window=60)
    last_stats = LatencyStats()
    last_result: Optional[PerceptionResult] = None

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    start_time = time.monotonic()

    try:
        while True:
            if timeout > 0.0 and (time.monotonic() - start_time) >= timeout:
                logger.info("Timeout of %.1fs reached. Exiting.", timeout)
                break
                
            t_wait_start = time.monotonic()
            if async_mode and reader is not None:
                frame_data = reader.read_latest(timeout=0.2)
            else:
                frame_data = cam.read()
            t_wait_end = time.monotonic()
            wait_duration_ms = (t_wait_end - t_wait_start) * 1000.0

            if frame_data is None:
                continue

            process_start = time.monotonic()

            fps_counter.tick(frame_data.capture_timestamp)

            if mirror_flip:
                cv2.flip(frame_data.image, 1, dst=frame_data.image)

            if pipeline is not None:
                try:
                    last_result, h_ms, y_ms, t_ms = pipeline.process(frame_data)
                    lat_tracker.record_perception(
                        perception_ms=h_ms, 
                        yolo_ms=y_ms, 
                        tracking_ms=t_ms
                    )
                except Exception as exc:
                    logger.warning("Perception error on frame %d: %s", frame_data.frame_id, exc)
                    last_result = None

            if last_result is not None:
                _draw_perception(frame_data.image, last_result)

            cap_cnt = reader.captured_frames if reader else fps_counter.total_frames
            drop_cnt = reader.dropped_frames if reader else 0
            drop_rat = reader.drop_ratio if reader else 0.0
            cap_fps_val = reader.capture_fps if reader else fps_counter.fps
            proc_fps_val = fps_counter.fps

            capture_read_duration = reader.last_read_duration_ms if reader else wait_duration_ms
            consumer_wait_duration = wait_duration_ms if reader else 0.0

            # Record stats BEFORE drawing overlay so HUD shows current frame metrics
            process_end = time.monotonic()
            last_stats = lat_tracker.record(
                capture_timestamp=frame_data.capture_timestamp,
                process_start=process_start,
                process_end=process_end,
                capture_read_ms=capture_read_duration,
                consumer_wait_ms=consumer_wait_duration,
            )

            _draw_overlay(
                frame_data.image,
                cap_fps=cap_fps_val,
                proc_fps=proc_fps_val,
                frame_id=frame_data.frame_id,
                stats=last_stats,
                state=last_result,
                perception_enabled=perception_enabled,
                captured=cap_cnt,
                dropped=drop_cnt,
                drop_ratio=drop_rat,
            )

            cv2.imshow(_WINDOW_NAME, frame_data.image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except cv2.error as exc:
        logger.error("OpenCV error: %s", exc)
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()
        if reader is not None:
            reader.stop()
        else:
            cam.release()
        cv2.destroyAllWindows()

    _print_summary(fps_counter, lat_tracker, reader)
    return 0


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(
        device_index=args.device,
        backend_str=args.backend,
        async_mode=not args.sync,
        mirror_flip=not args.no_flip,
        perception_enabled=not args.no_perception,
        model_path=args.model,
        timeout=args.timeout,
    ))
