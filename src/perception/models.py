"""
perception/models.py — Pure Python data structures for the Fast Perception layer.

These dataclasses are the public API consumed by all downstream pipeline stages.
No module outside of src/perception/human.py should import anything from
mediapipe directly.  The rest of the system works exclusively with these types.

Design goals:
  - Immutable snapshots (frozen=True where practical).
  - No MediaPipe types exposed.
  - Lightweight: no heavy computation or I/O here.
  - Future-proof: easy to add fields without breaking downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Pose landmark indices (MediaPipe Pose — 33 landmarks)
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# ---------------------------------------------------------------------------

class PoseLandmarkIndex:
    """Named constants for MediaPipe Pose landmark indices.

    Only the landmarks most relevant to a teaching/lecturing context are
    named here.  All 33 are captured; these names are for convenient access.
    """
    NOSE            = 0
    LEFT_EYE_INNER  = 1
    LEFT_EYE        = 2
    LEFT_EYE_OUTER  = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE       = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR        = 7
    RIGHT_EAR       = 8
    MOUTH_LEFT      = 9
    MOUTH_RIGHT     = 10
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13
    RIGHT_ELBOW     = 14
    LEFT_WRIST      = 15
    RIGHT_WRIST     = 16
    LEFT_PINKY      = 17
    RIGHT_PINKY     = 18
    LEFT_INDEX      = 19
    RIGHT_INDEX     = 20
    LEFT_THUMB      = 21
    RIGHT_THUMB     = 22
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
    LEFT_KNEE       = 25
    RIGHT_KNEE      = 26
    LEFT_ANKLE      = 27
    RIGHT_ANKLE     = 28
    LEFT_HEEL       = 29
    RIGHT_HEEL      = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# ---------------------------------------------------------------------------
# Hand landmark indices (MediaPipe Hands — 21 landmarks per hand)
# ---------------------------------------------------------------------------

class HandLandmarkIndex:
    """Named constants for MediaPipe Hands landmark indices."""
    WRIST             = 0
    THUMB_CMC         = 1
    THUMB_MCP         = 2
    THUMB_IP          = 3
    THUMB_TIP         = 4
    INDEX_FINGER_MCP  = 5
    INDEX_FINGER_PIP  = 6
    INDEX_FINGER_DIP  = 7
    INDEX_FINGER_TIP  = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP   = 13
    RING_FINGER_PIP   = 14
    RING_FINGER_DIP   = 15
    RING_FINGER_TIP   = 16
    PINKY_MCP         = 17
    PINKY_PIP         = 18
    PINKY_DIP         = 19
    PINKY_TIP         = 20


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Landmark:
    """A single 3-D landmark point.

    Coordinates follow MediaPipe conventions:
      x, y: Normalized to [0, 1] relative to the image frame.
             x=0 is left edge, x=1 is right edge.
             y=0 is top edge, y=1 is bottom edge.
      z:     Depth relative to the root landmark.
             For pose: relative to the hip midpoint (smaller z = closer to camera).
             For hands: relative to the wrist.
             Scale is roughly the same as x/y in normalized space.
      visibility: Confidence that the landmark is visible in the frame.
             Range [0, 1].  Only present for pose landmarks; None for hands.

    Downstream code should treat x, y, z as normalized floats.
    To convert to pixel coordinates:
        px = int(x * frame_width)
        py = int(y * frame_height)
    """
    x: float
    y: float
    z: float
    visibility: Optional[float] = None  # None for hand landmarks


@dataclass(frozen=True)
class PoseData:
    """Pose estimation result for one frame.

    Attributes:
        landmarks:       33 body landmarks in normalized image coordinates.
        world_landmarks: 33 body landmarks in real-world metric coordinates
                         (meters, origin at hip midpoint).  May be empty if
                         the backend did not provide world landmarks.
    """
    landmarks: List[Landmark]        # len == 33, normalized image coords
    world_landmarks: List[Landmark]  # len == 33, metric coords; may be []

    def get(self, index: int) -> Optional[Landmark]:
        """Return the landmark at *index*, or None if out of range."""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None

    # Convenience accessors for the landmarks most relevant to lecture analysis.
    @property
    def nose(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.NOSE)

    @property
    def left_shoulder(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.LEFT_SHOULDER)

    @property
    def right_shoulder(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.RIGHT_SHOULDER)

    @property
    def left_elbow(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.LEFT_ELBOW)

    @property
    def right_elbow(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.RIGHT_ELBOW)

    @property
    def left_wrist(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.LEFT_WRIST)

    @property
    def right_wrist(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.RIGHT_WRIST)

    @property
    def left_hip(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.LEFT_HIP)

    @property
    def right_hip(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.RIGHT_HIP)

    @property
    def left_knee(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.LEFT_KNEE)

    @property
    def right_knee(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.RIGHT_KNEE)

    @property
    def left_ankle(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.LEFT_ANKLE)

    @property
    def right_ankle(self) -> Optional[Landmark]:
        return self.get(PoseLandmarkIndex.RIGHT_ANKLE)


@dataclass(frozen=True)
class HandData:
    """Hand landmark result for one detected hand.

    Attributes:
        handedness: "Left" or "Right" as reported by MediaPipe.
                    Note: MediaPipe reports from the person's perspective
                    (mirrored relative to the camera image by default).
        landmarks:  21 hand landmarks in normalized image coordinates.
                    Index mapping follows HandLandmarkIndex constants.
        score:      Handedness classification confidence [0, 1].
    """
    handedness: str       # "Left" or "Right"
    landmarks: List[Landmark]   # len == 21
    score: float = 1.0    # handedness confidence

    def get(self, index: int) -> Optional[Landmark]:
        """Return the landmark at *index*, or None if out of range."""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None

    @property
    def wrist(self) -> Optional[Landmark]:
        return self.get(HandLandmarkIndex.WRIST)

    @property
    def index_tip(self) -> Optional[Landmark]:
        return self.get(HandLandmarkIndex.INDEX_FINGER_TIP)

    @property
    def thumb_tip(self) -> Optional[Landmark]:
        return self.get(HandLandmarkIndex.THUMB_TIP)


@dataclass(frozen=True)
class PerceptionResult:
    """The complete output of one Fast Perception pass on a single frame.

    This is the object that all downstream pipeline stages (Temporal
    Processing, Semantic Understanding, etc.) consume.  It contains no
    MediaPipe types.

    Attributes:
        frame_id:          Matches the CapturedFrame.frame_id this was derived from.
        capture_timestamp: Monotonic timestamp from the source CapturedFrame.
        perception_start:  time.monotonic() recorded before running MediaPipe.
        perception_end:    time.monotonic() recorded after MediaPipe returned.
        pose:              Body pose landmarks, or None if no person was detected.
        hands:             List of detected hands (0, 1, or 2 entries).
    """
    frame_id: int
    capture_timestamp: float       # from CapturedFrame — for latency chaining
    perception_start: float        # time.monotonic()
    perception_end: float          # time.monotonic()
    pose: Optional[PoseData]
    hands: List[HandData] = field(default_factory=list)

    @property
    def perception_ms(self) -> float:
        """Time spent running the MediaPipe models for this frame (ms)."""
        return (self.perception_end - self.perception_start) * 1_000.0

    @property
    def has_pose(self) -> bool:
        return self.pose is not None

    @property
    def num_hands(self) -> int:
        return len(self.hands)
