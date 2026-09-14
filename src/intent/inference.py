"""
intent/inference.py — Deterministic semantic scene understanding and intent inference.

This module derives high-level meaning from the structured perception data in
SceneState. It operates at a lower update rate (~5 Hz) than the perception layer
(~20–30 Hz), consuming compact SceneState objects — NOT raw video frames.

All logic here is:
  - Deterministic heuristics (NOT a VLM or neural model)
  - Explicitly documented so failures are debuggable
  - Designed for future extension with speech/VLM context
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

from src.intent.models import (
    FocusTarget,
    IntentCategory,
    SemanticActivity,
    SemanticContext,
    SemanticSceneState,
    VisualizationOpportunity,
)
from src.perception.models import (
    ActivityState,
    Gesture,
    HandNearObject,
    HandOverlappingObject,
    HandPointingAtObject,
    ObjectAppeared,
    ObjectDisappeared,
    PointingStarted,
    PointingStopped,
    SceneEvent,
    SceneState,
    TrackedObject,
)
from src.perception.temporal import SceneHistory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focus Target Scoring
# ---------------------------------------------------------------------------

def _score_focus_candidates(
    state: SceneState,
) -> Dict[int, float]:
    """Return a score ≥ 0.0 for each tracked object.

    Higher score = stronger evidence it is the current focus.

    Evidence types and their contributions:
      - HandPointingAtObject:   +0.60   (strong directional evidence)
      - HandOverlappingObject:  +0.50   (contact, very strong)
      - HandNearObject:         +0.20   (proximity, weak)
      - YOLO confidence:        +0.15 * obj.confidence  (detection strength)
      - Object is large/central: +0.10  (bonus for central/large objects)
    """
    scores: Dict[int, float] = {obj.track_id: 0.0 for obj in state.tracked_objects}

    # Base score from YOLO confidence
    for obj in state.tracked_objects:
        scores[obj.track_id] += 0.15 * obj.confidence

    # Bonus for objects near center of frame (normalized coords → 0.5, 0.5)
    for obj in state.tracked_objects:
        cx, cy = obj.center
        dist_to_center = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
        centrality_bonus = max(0.0, 0.10 * (1.0 - dist_to_center * 2.0))
        scores[obj.track_id] += centrality_bonus

    # Relationship bonuses
    for rel in state.relationships:
        track_id = rel.object_id
        if track_id not in scores:
            continue
        if isinstance(rel, HandPointingAtObject):
            scores[track_id] += 0.60
        elif isinstance(rel, HandOverlappingObject):
            scores[track_id] += 0.50
        elif isinstance(rel, HandNearObject):
            scores[track_id] += 0.20

    return scores


def _select_focus_target(state: SceneState) -> Optional[FocusTarget]:
    """Select the best focus candidate. Returns None if no objects are visible."""
    if not state.tracked_objects:
        return None

    scores = _score_focus_candidates(state)
    if not scores:
        return None

    best_id = max(scores, key=lambda k: scores[k])
    best_score = scores[best_id]

    # Must have at least some non-trivial evidence
    if best_score < 0.20:
        return None

    # Find the corresponding TrackedObject for its label
    best_obj = next((o for o in state.tracked_objects if o.track_id == best_id), None)
    if best_obj is None:
        return None

    # Determine the dominant relation label
    relation = "detected"
    for rel in state.relationships:
        if rel.object_id == best_id:
            if isinstance(rel, HandPointingAtObject):
                relation = "pointing_toward"
                break
            elif isinstance(rel, HandOverlappingObject):
                relation = "overlapping"
                break
            elif isinstance(rel, HandNearObject):
                relation = "near"

    # Normalise score to a [0, 1] confidence — cap at 1.0
    confidence = min(1.0, best_score)

    return FocusTarget(
        track_id=best_id,
        label=best_obj.class_name,
        relation=relation,
        confidence=round(confidence, 3),
    )


# ---------------------------------------------------------------------------
# Activity Inference
# ---------------------------------------------------------------------------

def _infer_semantic_activity(
    state: SceneState,
    focus_target: Optional[FocusTarget],
    recent_event_names: List[str],
) -> SemanticActivity:
    """Deterministic activity classification.

    Priority (first matching rule wins):
      1. No person visible → UNKNOWN
      2. Hand pointing + pointing relationship to object → POINTING
      3. Hand overlapping or near moving object → INTERACTING_WITH_OBJECT
      4. Any non-NONE gesture + objects present → PRESENTING
      5. Any non-NONE gesture → GESTURING
      6. Person visible but no active gestures → STANDING
      7. Default → IDLE
    """
    if state.human_perception is None or not state.human_perception.has_pose:
        return SemanticActivity.UNKNOWN

    activity = state.activity
    has_left_pointing = activity and activity.left_hand_gesture == Gesture.POINTING
    has_right_pointing = activity and activity.right_hand_gesture == Gesture.POINTING
    has_pointing = has_left_pointing or has_right_pointing

    has_any_gesture = activity and (
        activity.left_hand_gesture not in (Gesture.NONE, Gesture.UNKNOWN)
        or activity.right_hand_gesture not in (Gesture.NONE, Gesture.UNKNOWN)
    )

    has_objects = len(state.tracked_objects) > 0

    # Rule 1: Pointing at an object
    has_pointing_relation = any(isinstance(r, HandPointingAtObject) for r in state.relationships)
    if has_pointing and has_pointing_relation and focus_target is not None:
        return SemanticActivity.POINTING

    # Rule 2: Hand interacting with object (overlap or near a moving object)
    for rel in state.relationships:
        if isinstance(rel, HandOverlappingObject):
            return SemanticActivity.INTERACTING_WITH_OBJECT
        if isinstance(rel, HandNearObject):
            obj = next((o for o in state.tracked_objects if o.track_id == rel.object_id), None)
            if obj and (abs(obj.velocity[0]) > 0.01 or abs(obj.velocity[1]) > 0.01):
                return SemanticActivity.INTERACTING_WITH_OBJECT

    # Rule 3: Gesturing while objects are visible → presenting
    if has_any_gesture and has_objects:
        return SemanticActivity.PRESENTING

    # Rule 4: Gesturing without objects
    if has_any_gesture:
        return SemanticActivity.GESTURING

    # Rule 5: Person visible but hands still
    if state.human_perception.has_pose:
        return SemanticActivity.STANDING

    return SemanticActivity.IDLE


# ---------------------------------------------------------------------------
# Intent Inference
# ---------------------------------------------------------------------------

def _infer_intent(
    activity: SemanticActivity,
    focus_target: Optional[FocusTarget],
    recent_event_names: List[str],
    history: SceneHistory,
) -> Tuple[IntentCategory, float, List[str]]:
    """Deterministic intent classification.

    Returns (IntentCategory, confidence, reasons).

    Intent evidence accumulation:
      Each rule adds to a running confidence score.
      The highest-scoring intent category wins.
    """
    # Score each candidate intent
    candidate_scores: Dict[IntentCategory, Tuple[float, List[str]]] = {}

    def _add(intent: IntentCategory, score_delta: float, reason: str) -> None:
        existing_score, existing_reasons = candidate_scores.get(intent, (0.0, []))
        candidate_scores[intent] = (existing_score + score_delta, existing_reasons + [reason])

    # --- Pointing to object ---
    if activity == SemanticActivity.POINTING and focus_target is not None:
        _add(IntentCategory.POINTING_TO_OBJECT, 0.60, "pointing_gesture")
        if focus_target.relation == "pointing_toward":
            _add(IntentCategory.POINTING_TO_OBJECT, 0.20, "pointing_toward_object_confirmed")
        if "pointing_started" in recent_event_names:
            _add(IntentCategory.POINTING_TO_OBJECT, 0.10, "recent_pointing_started_event")

    # --- Demonstrating ---
    if activity == SemanticActivity.INTERACTING_WITH_OBJECT:
        _add(IntentCategory.DEMONSTRATING, 0.55, "interaction_with_object")
        if focus_target is not None:
            _add(IntentCategory.DEMONSTRATING, 0.20, "persistent_focus_target")
        if "interaction_started" in recent_event_names:
            _add(IntentCategory.DEMONSTRATING, 0.10, "interaction_started_event")

    # --- Presenting/Explaining ---
    if activity == SemanticActivity.PRESENTING:
        _add(IntentCategory.EXPLAINING, 0.45, "presenting_activity")
    if activity == SemanticActivity.GESTURING:
        _add(IntentCategory.EXPLAINING, 0.30, "active_gestures")

    # --- Visualization candidate (requires stronger evidence) ---
    if activity in (SemanticActivity.POINTING, SemanticActivity.INTERACTING_WITH_OBJECT):
        if focus_target is not None and focus_target.confidence > 0.60:
            _add(IntentCategory.VISUALIZATION_CANDIDATE, 0.50, "high_confidence_focus_target")
        if "pointing_started" in recent_event_names or "interaction_started" in recent_event_names:
            _add(IntentCategory.VISUALIZATION_CANDIDATE, 0.25, "active_interaction_event")
        if activity == SemanticActivity.INTERACTING_WITH_OBJECT:
            _add(IntentCategory.VISUALIZATION_CANDIDATE, 0.20, "demonstrating_object")

    # Fallback
    if not candidate_scores:
        return IntentCategory.NONE, 0.0, []

    # Pick winner
    best_intent = max(candidate_scores, key=lambda k: candidate_scores[k][0])
    best_score, reasons = candidate_scores[best_intent]
    confidence = min(1.0, best_score)

    return best_intent, round(confidence, 3), reasons


# ---------------------------------------------------------------------------
# Event name helpers
# ---------------------------------------------------------------------------

_EVENT_NAMES: Dict[type, str] = {
    ObjectAppeared:      "object_appeared",
    ObjectDisappeared:   "object_disappeared",
    PointingStarted:     "pointing_started",
    PointingStopped:     "pointing_stopped",
}


def _event_name(event: SceneEvent) -> str:
    return _EVENT_NAMES.get(type(event), type(event).__name__.lower())


# ---------------------------------------------------------------------------
# SemanticEngine — main public class
# ---------------------------------------------------------------------------

class SemanticEngine:
    """Consumes SceneState snapshots and produces SemanticContext.

    This class is designed to be called from the main processing loop at a
    reduced rate (~5 Hz) or when significant events occur. It does NOT
    process raw frames and contains NO MediaPipe or YOLO code.

    Usage::

        engine = SemanticEngine()
        context = engine.update(scene_state, scene_history)
    """

    def __init__(self, update_interval_s: float = 0.20):
        """
        Args:
            update_interval_s: Minimum seconds between full semantic updates.
                               Default is 0.20s (5 Hz).
        """
        self._update_interval_s = update_interval_s
        self._last_update_time: float = 0.0
        self._last_context: Optional[SemanticContext] = None

    @property
    def last_context(self) -> Optional[SemanticContext]:
        return self._last_context

    def should_update(self, state: SceneState) -> bool:
        """Return True if a semantic update should be triggered.

        Triggers:
          - Rate-limited: at least _update_interval_s has elapsed.
          - Event-driven: new SceneEvents have been generated.
        """
        elapsed = time.monotonic() - self._last_update_time
        if elapsed >= self._update_interval_s:
            return True
        if state.recent_events:
            return True
        return False

    def update(self, state: SceneState, history: SceneHistory) -> SemanticContext:
        """Compute a new SemanticContext from the current SceneState.

        This method should only be called when should_update() returns True.
        """
        self._last_update_time = time.monotonic()

        # 1. Collect recent events (last N frames in history)
        recent_scene_events: List[SceneEvent] = list(state.recent_events)
        # Also grab events from a few frames back
        h = history._history
        for past_state in list(h)[-10:]:  # Look back ~10 frames
            for evt in past_state.recent_events:
                if evt not in recent_scene_events:
                    recent_scene_events.append(evt)
        recent_event_names = [_event_name(e) for e in recent_scene_events]

        # 2. Focus Target
        focus_target = _select_focus_target(state)

        # 3. Activity
        activity = _infer_semantic_activity(state, focus_target, recent_event_names)

        # 4. Intent
        intent, intent_confidence, reasons = _infer_intent(
            activity, focus_target, recent_event_names, history
        )

        # 5. Assemble SemanticSceneState
        person_count = 1 if (state.human_perception and state.human_perception.has_pose) else 0

        visible_objects = [
            {
                "track_id": obj.track_id,
                "label": obj.class_name,
                "confidence": round(obj.confidence, 3),
            }
            for obj in state.tracked_objects
        ]

        active_gestures = []
        if state.activity:
            lg = state.activity.left_hand_gesture
            rg = state.activity.right_hand_gesture
            if lg not in (Gesture.NONE, Gesture.UNKNOWN):
                active_gestures.append({"hand": "left", "gesture": lg.value})
            if rg not in (Gesture.NONE, Gesture.UNKNOWN):
                active_gestures.append({"hand": "right", "gesture": rg.value})

        # Motion summary: describe the dominant relationship type
        motion_summary = "static"
        if any(isinstance(r, HandPointingAtObject) for r in state.relationships):
            motion_summary = "hand_pointing_at_object"
        elif any(isinstance(r, HandOverlappingObject) for r in state.relationships):
            motion_summary = "hand_interacting_with_object"
        elif any(isinstance(r, HandNearObject) for r in state.relationships):
            motion_summary = "hand_near_object"

        # Scene confidence: average of focus target and intent confidences
        scene_confidence = round(
            (focus_target.confidence if focus_target else 0.3) * 0.5
            + intent_confidence * 0.5,
            3,
        )

        # Deduplicate recent event names for the state's report
        seen = set()
        deduped_events = []
        for name in recent_event_names:
            if name not in seen:
                seen.add(name)
                deduped_events.append(name)

        semantic_scene = SemanticSceneState(
            timestamp=state.timestamp,
            activity=activity,
            person_count=person_count,
            visible_objects=visible_objects,
            active_gestures=active_gestures,
            focus_target=focus_target,
            recent_events=deduped_events[:8],  # Cap for readability
            motion_summary=motion_summary,
            confidence=scene_confidence,
        )

        # 6. Visualization Opportunity
        should_visualize = (
            intent in (IntentCategory.VISUALIZATION_CANDIDATE, IntentCategory.POINTING_TO_OBJECT)
            and intent_confidence >= 0.60
            and focus_target is not None
        )

        opportunity = VisualizationOpportunity(
            should_visualize=should_visualize,
            intent=intent,
            concept=None,  # Requires speech context — not yet available
            target_track_id=focus_target.track_id if focus_target else None,
            target_label=focus_target.label if focus_target else None,
            anchor_type="object" if focus_target else None,
            anchor_track_id=focus_target.track_id if focus_target else None,
            confidence=intent_confidence,
            reasons=reasons,
        )

        context = SemanticContext(
            current_scene=semantic_scene,
            opportunity=opportunity,
        )
        self._last_context = context
        return context
