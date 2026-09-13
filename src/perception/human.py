"""
perception/human.py — Fast Perception pipeline for the Magic Mirror project.

This module is the ONLY place in the entire codebase that imports MediaPipe.
All other modules interact exclusively with our own data models defined in
perception/models.py.

Architecture
------------
Camera → CapturedFrame → HumanPerceptionPipeline.process() → PerceptionResult

The pipeline wraps MediaPipe HolisticLandmarker (Tasks API, v1.0+).
HolisticLandmarker detects pose AND hands in a single model pass, which is
more efficient than running PoseLandmarker + HandLandmarker separately.

Model
-----
Uses holistic_landmarker.task (downloaded to models/ at setup time).
The model path is resolved relative to this file's location so the pipeline
works regardless of the working directory.

Threading / RunningMode
-----------------------
RunningMode.VIDEO is used for the synchronous streaming loop.
  - IMAGE mode: no temporal tracking between frames (high jitter).
  - VIDEO mode: enables temporal smoothing; requires monotonically increasing
    timestamps; synchronous call (blocks until result ready).
  - LIVE_STREAM mode: async callback; more complex; not needed here yet.

VIDEO mode is the right choice for the current synchronous capture loop.

RGB conversion cost
-------------------
OpenCV captures in BGR; MediaPipe requires RGB.
cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) is required per frame.
This is unavoidable given the two libraries' differing colour conventions.
The cost is typically < 1 ms for 720p and is included in perception_ms.

Failure handling
----------------
If MediaPipe raises during a frame, process() returns an empty PerceptionResult
(pose=None, hands=[]) rather than propagating the exception, so the camera
loop continues uninterrupted.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ---- MediaPipe Tasks API ---------------------------------------------------
# All MediaPipe imports are confined to this file.
import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark import (
    Landmark as _MPLandmark,
    NormalizedLandmark as _MPNormalizedLandmark,
)
from mediapipe.tasks import python as _mp_tasks_python
from mediapipe.tasks.python import vision as _mp_vision

# ---- Our own data models (no MediaPipe dependency) -------------------------
from src.perception.models import (
    HandData,
    Landmark,
    PerceptionResult,
    PoseData,
)
from src.camera.capture import CapturedFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
# The model file is stored in models/ at the repository root.
# Path is resolved relative to this source file so it works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "holistic_landmarker.task"


# ---------------------------------------------------------------------------
# Converter helpers  (MediaPipe → our types)
# ---------------------------------------------------------------------------

def _convert_normalized_landmark(lm: _MPNormalizedLandmark) -> Landmark:
    """Convert a MediaPipe NormalizedLandmark to our Landmark dataclass."""
    return Landmark(
        x=lm.x,
        y=lm.y,
        z=lm.z,
        visibility=lm.visibility,  # None for hands, float for pose
    )


def _convert_landmark_list(
    lms: List[_MPNormalizedLandmark],
) -> List[Landmark]:
    return [_convert_normalized_landmark(lm) for lm in lms]


# ---------------------------------------------------------------------------
# HumanPerceptionPipeline
# ---------------------------------------------------------------------------

class HumanPerceptionPipeline:
    """Real-time human pose and hand landmark detector.

    Wraps MediaPipe HolisticLandmarker using the Tasks API (MediaPipe ≥ 1.0).

    Usage::

        pipeline = HumanPerceptionPipeline()
        pipeline.open()
        try:
            while ...:
                frame = cam.read()
                result = pipeline.process(frame)
                # use result.pose, result.hands
        finally:
            pipeline.close()

    Or as a context manager::

        with HumanPerceptionPipeline() as pipeline:
            result = pipeline.process(frame)

    Thread safety: not thread-safe.  Call from a single thread.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_pose_detection_confidence: float = 0.5,
        min_pose_landmarks_confidence: float = 0.5,
        min_hand_landmarks_confidence: float = 0.5,
        min_face_detection_confidence: float = 0.5,
        min_face_landmarks_confidence: float = 0.5,
    ) -> None:
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._min_pose_det = min_pose_detection_confidence
        self._min_pose_lm = min_pose_landmarks_confidence
        self._min_hand_lm = min_hand_landmarks_confidence
        self._min_face_det = min_face_detection_confidence
        self._min_face_lm = min_face_landmarks_confidence

        self._landmarker: Optional[_mp_vision.HolisticLandmarker] = None
        # Frame counter used to produce monotonically increasing timestamps
        # for VIDEO mode (required by MediaPipe).
        self._frame_count: int = 0

    def open(self) -> None:
        """Initialise the MediaPipe HolisticLandmarker.

        Raises:
            FileNotFoundError: If the model file is not found.
            RuntimeError: If MediaPipe fails to initialise.
        """
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {self._model_path}.\n"
                "Run the model download step:\n"
                "  python -c \"import urllib.request, pathlib; "
                "pathlib.Path('models').mkdir(exist_ok=True); "
                "urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/"
                "holistic_landmarker/holistic_landmarker/float16/latest/"
                "holistic_landmarker.task', 'models/holistic_landmarker.task')\""
            )

        logger.info("Loading HolisticLandmarker model from %s …", self._model_path)

        base_options = _mp_tasks_python.BaseOptions(
            model_asset_path=str(self._model_path)
        )
        options = _mp_vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=_mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self._min_pose_det,
            min_pose_landmarks_confidence=self._min_pose_lm,
            min_hand_landmarks_confidence=self._min_hand_lm,
            min_face_detection_confidence=self._min_face_det,
            min_face_landmarks_confidence=self._min_face_lm,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self._landmarker = _mp_vision.HolisticLandmarker.create_from_options(options)
        self._frame_count = 0
        logger.info("HolisticLandmarker ready.")

    def close(self) -> None:
        """Release the MediaPipe landmarker and associated resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("HolisticLandmarker closed.")

    def is_ready(self) -> bool:
        """Return True if the pipeline has been opened and is ready."""
        return self._landmarker is not None

    def process(self, frame: CapturedFrame) -> PerceptionResult:
        """Run pose and hand detection on a single captured frame.

        Records precise monotonic timestamps around the MediaPipe call so
        perception_ms in the result is accurate.

        Args:
            frame: A CapturedFrame from Camera.read().

        Returns:
            PerceptionResult with pose / hands populated (or None/[]) if not
            detected or if an error occurred.  Never raises.
        """
        perception_start = time.monotonic()

        pose_data: Optional[PoseData] = None
        hands: List[HandData] = []

        if self._landmarker is None:
            logger.warning("process() called before open() — returning empty result.")
            return PerceptionResult(
                frame_id=frame.frame_id,
                capture_timestamp=frame.capture_timestamp,
                perception_start=perception_start,
                perception_end=time.monotonic(),
                pose=None,
                hands=[],
            )

        try:
            # ------------------------------------------------------------------
            # BGR → RGB  (OpenCV is BGR; MediaPipe requires RGB)
            # This conversion is unavoidable.  Cost: typically < 1 ms @ 720p.
            # ------------------------------------------------------------------
            rgb_frame = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)

            # Wrap as MediaPipe Image (zero-copy when possible)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # ------------------------------------------------------------------
            # MediaPipe VIDEO mode requires a monotonically increasing timestamp
            # in milliseconds.  We use the frame's capture_timestamp (monotonic
            # seconds) converted to integer milliseconds.
            # ------------------------------------------------------------------
            timestamp_ms = int(frame.capture_timestamp * 1_000)

            # ------------------------------------------------------------------
            # Run the holistic landmarker (synchronous in VIDEO mode)
            # ------------------------------------------------------------------
            mp_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            # ------------------------------------------------------------------
            # Convert MediaPipe result → our data models
            # ------------------------------------------------------------------
            pose_data = self._extract_pose(mp_result)
            hands = self._extract_hands(mp_result)

        except Exception as exc:  # noqa: BLE001
            # Never crash the capture loop.  Log and return an empty result.
            logger.warning("Perception error on frame %d: %s", frame.frame_id, exc)

        perception_end = time.monotonic()

        return PerceptionResult(
            frame_id=frame.frame_id,
            capture_timestamp=frame.capture_timestamp,
            perception_start=perception_start,
            perception_end=perception_end,
            pose=pose_data,
            hands=hands,
        )

    # ------------------------------------------------------------------
    # Private conversion methods
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pose(mp_result) -> Optional[PoseData]:
        """Convert MediaPipe HolisticLandmarkerResult pose data → PoseData."""
        if not mp_result.pose_landmarks:
            return None

        landmarks = _convert_landmark_list(mp_result.pose_landmarks)

        world_landmarks: List[Landmark] = []
        if mp_result.pose_world_landmarks:
            # World landmarks use Landmark (not NormalizedLandmark) but have
            # the same x/y/z/visibility fields — our converter works for both.
            world_landmarks = [
                Landmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility,
                )
                for lm in mp_result.pose_world_landmarks
            ]

        return PoseData(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
        )

    @staticmethod
    def _extract_hands(mp_result) -> List[HandData]:
        """Convert MediaPipe HolisticLandmarkerResult hand data → [HandData]."""
        hands: List[HandData] = []

        # HolisticLandmarkerResult has separate left/right hand fields.
        hand_pairs = [
            ("Left",  mp_result.left_hand_landmarks),
            ("Right", mp_result.right_hand_landmarks),
        ]
        for handedness, lm_list in hand_pairs:
            if not lm_list:
                continue
            landmarks = _convert_landmark_list(lm_list)
            hands.append(HandData(
                handedness=handedness,
                landmarks=landmarks,
                score=1.0,  # HolisticLandmarker doesn't expose handedness scores
            ))

        return hands

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "HumanPerceptionPipeline":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "ready" if self.is_ready() else "closed"
        return f"HumanPerceptionPipeline(status={status}, model={self._model_path.name})"
