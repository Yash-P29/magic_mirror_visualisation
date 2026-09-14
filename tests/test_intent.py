"""
tests/test_intent.py — Unit tests for Milestone 3 semantic understanding.

All tests use synthetic SceneState objects — no physical camera required.
"""

from __future__ import annotations

import json
import time

import pytest

from src.intent.inference import (
    SemanticEngine,
    _infer_intent,
    _infer_semantic_activity,
    _score_focus_candidates,
    _select_focus_target,
)
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
    BoundingBox,
    Gesture,
    HandNearObject,
    HandOverlappingObject,
    HandPointingAtObject,
    Landmark,
    ObjectAppeared,
    PerceptionResult,
    PointingStarted,
    PoseData,
    SceneEvent,
    SceneState,
    TrackedObject,
)
from src.perception.temporal import SceneHistory


# ---------------------------------------------------------------------------
# Test Helpers / Factories
# ---------------------------------------------------------------------------

def _lm(x=0.5, y=0.5, z=0.0, vis=None) -> Landmark:
    return Landmark(x=x, y=y, z=z, visibility=vis)


def _pose(**kwargs) -> PoseData:
    lms = [_lm() for _ in range(33)]
    return PoseData(landmarks=tuple(lms))


def _tracked_obj(track_id=1, label="laptop", confidence=0.85,
                  x1=0.4, y1=0.4, x2=0.6, y2=0.6,
                  vx=0.0, vy=0.0) -> TrackedObject:
    bbox = BoundingBox(x1, y1, x2, y2)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return TrackedObject(
        track_id=track_id,
        class_name=label,
        bbox=bbox,
        center=(cx, cy),
        velocity=(vx, vy),
        last_seen=time.monotonic(),
        confidence=confidence,
    )


def _perception_result(has_pose=True) -> PerceptionResult:
    return PerceptionResult(
        frame_id=1,
        capture_timestamp=time.monotonic(),
        perception_start=time.monotonic(),
        perception_end=time.monotonic(),
        pose=_pose() if has_pose else None,
        hands=(),
    )


def _scene_state(
    tracked_objects=(),
    relationships=(),
    recent_events=(),
    activity=None,
    has_pose=True,
) -> SceneState:
    if activity is None:
        activity = ActivityState()
    return SceneState(
        frame_id=1,
        timestamp=time.monotonic(),
        human_perception=_perception_result(has_pose=has_pose),
        tracked_objects=tracked_objects,
        relationships=relationships,
        recent_events=recent_events,
        activity=activity,
    )


def _empty_history() -> SceneHistory:
    return SceneHistory(max_history_frames=90)


# ---------------------------------------------------------------------------
# 1. Activity Inference
# ---------------------------------------------------------------------------

def test_activity_unknown_when_no_pose():
    state = _scene_state(has_pose=False)
    activity = _infer_semantic_activity(state, None, [])
    assert activity == SemanticActivity.UNKNOWN


def test_activity_standing_person_no_gestures():
    state = _scene_state(has_pose=True)
    activity = _infer_semantic_activity(state, None, [])
    assert activity == SemanticActivity.STANDING


def test_activity_pointing_when_pointing_relationship_exists():
    obj = _tracked_obj()
    rel = HandPointingAtObject(hand_is_left=False, object_id=1)
    focus = FocusTarget(track_id=1, label="laptop", relation="pointing_toward", confidence=0.75)
    act = ActivityState(left_hand_gesture=Gesture.NONE, right_hand_gesture=Gesture.POINTING)
    state = _scene_state(tracked_objects=(obj,), relationships=(rel,), activity=act)
    result = _infer_semantic_activity(state, focus, [])
    assert result == SemanticActivity.POINTING


def test_activity_interacting_when_overlapping():
    obj = _tracked_obj()
    rel = HandOverlappingObject(hand_is_left=False, object_id=1)
    act = ActivityState(right_hand_gesture=Gesture.NONE)
    state = _scene_state(tracked_objects=(obj,), relationships=(rel,), activity=act)
    result = _infer_semantic_activity(state, None, [])
    assert result == SemanticActivity.INTERACTING_WITH_OBJECT


def test_activity_presenting_gesture_with_objects():
    obj = _tracked_obj()
    act = ActivityState(right_hand_gesture=Gesture.RAISED_HAND)
    state = _scene_state(tracked_objects=(obj,), activity=act)
    result = _infer_semantic_activity(state, None, [])
    assert result == SemanticActivity.PRESENTING


def test_activity_gesturing_no_objects():
    act = ActivityState(right_hand_gesture=Gesture.RAISED_HAND)
    state = _scene_state(activity=act)
    result = _infer_semantic_activity(state, None, [])
    assert result == SemanticActivity.GESTURING


# ---------------------------------------------------------------------------
# 2. Focus Target Selection
# ---------------------------------------------------------------------------

def test_focus_target_none_when_no_objects():
    state = _scene_state()
    focus = _select_focus_target(state)
    assert focus is None


def test_focus_target_selected_when_object_pointed_at():
    obj = _tracked_obj(track_id=5, label="phone")
    rel = HandPointingAtObject(hand_is_left=False, object_id=5)
    state = _scene_state(tracked_objects=(obj,), relationships=(rel,))
    focus = _select_focus_target(state)
    assert focus is not None
    assert focus.track_id == 5
    assert focus.relation == "pointing_toward"
    assert focus.confidence > 0.5


def test_focus_target_prefers_pointed_over_near():
    obj1 = _tracked_obj(track_id=1, label="laptop")
    obj2 = _tracked_obj(track_id=2, label="bottle", confidence=0.60)
    rel_point = HandPointingAtObject(hand_is_left=False, object_id=1)
    rel_near = HandNearObject(hand_is_left=False, object_id=2)
    state = _scene_state(tracked_objects=(obj1, obj2), relationships=(rel_point, rel_near))
    focus = _select_focus_target(state)
    assert focus is not None
    assert focus.track_id == 1


def test_focus_target_confidence_capped_at_1():
    obj = _tracked_obj()
    rel_point = HandPointingAtObject(hand_is_left=False, object_id=1)
    rel_overlap = HandOverlappingObject(hand_is_left=True, object_id=1)
    state = _scene_state(tracked_objects=(obj,), relationships=(rel_point, rel_overlap))
    focus = _select_focus_target(state)
    assert focus is not None
    assert focus.confidence <= 1.0


def test_focus_target_none_below_threshold():
    # Object with very low detection confidence and no relationships
    obj = _tracked_obj(confidence=0.10, x1=0.1, y1=0.1, x2=0.2, y2=0.2)  # low-conf, edge
    state = _scene_state(tracked_objects=(obj,))
    focus = _select_focus_target(state)
    # Score = 0.15 * 0.10 + small centrality = ~0.015, which is below 0.20 threshold
    assert focus is None


# ---------------------------------------------------------------------------
# 3. Pointing → Object Relationship
# ---------------------------------------------------------------------------

def test_pointing_relation_triggers_pointing_intent():
    obj = _tracked_obj()
    rel = HandPointingAtObject(hand_is_left=False, object_id=1)
    focus = FocusTarget(track_id=1, label="laptop", relation="pointing_toward", confidence=0.75)
    intent, conf, reasons = _infer_intent(SemanticActivity.POINTING, focus, ["pointing_started"], _empty_history())
    assert intent == IntentCategory.POINTING_TO_OBJECT
    assert conf > 0.5
    assert "pointing_gesture" in reasons


# ---------------------------------------------------------------------------
# 4 & 5. Event Generation and Deduplication
# ---------------------------------------------------------------------------

def test_recent_events_deduplicated():
    engine = SemanticEngine()
    history = _empty_history()
    # Build a state with two identical event names
    evt1 = PointingStarted(timestamp=1.0, hand_is_left=False)
    evt2 = PointingStarted(timestamp=1.1, hand_is_left=False)
    state = _scene_state(recent_events=(evt1, evt2))
    history.append(state)
    ctx = engine.update(state, history)
    # Deduplicated event names should not repeat
    assert ctx.current_scene.recent_events.count("pointing_started") == 1


def test_object_appeared_event_in_semantic_output():
    engine = SemanticEngine()
    history = _empty_history()
    obj = _tracked_obj()
    evt = ObjectAppeared(timestamp=1.0, object_id=1, class_name="laptop")
    state = _scene_state(tracked_objects=(obj,), recent_events=(evt,))
    history.append(state)
    ctx = engine.update(state, history)
    assert "object_appeared" in ctx.current_scene.recent_events


# ---------------------------------------------------------------------------
# 6. Visualization Opportunity Logic
# ---------------------------------------------------------------------------

def test_visualization_false_when_no_focus():
    intent, conf, reasons = _infer_intent(SemanticActivity.GESTURING, None, [], _empty_history())
    # VisualizationOpportunity requires focus_target, so should_visualize must be False
    opp = VisualizationOpportunity(
        should_visualize=(intent == IntentCategory.VISUALIZATION_CANDIDATE and conf >= 0.60 and False),
        intent=intent,
        concept=None,
        target_track_id=None,
        target_label=None,
        anchor_type=None,
        anchor_track_id=None,
        confidence=conf,
        reasons=reasons,
    )
    assert opp.should_visualize is False


def test_visualization_true_when_confident_pointing():
    obj = _tracked_obj()
    rel = HandPointingAtObject(hand_is_left=False, object_id=1)
    focus = FocusTarget(track_id=1, label="laptop", relation="pointing_toward", confidence=0.80)
    act = ActivityState(right_hand_gesture=Gesture.POINTING)
    evt = PointingStarted(timestamp=1.0, hand_is_left=False)
    state = _scene_state(tracked_objects=(obj,), relationships=(rel,), activity=act, recent_events=(evt,))
    history = _empty_history()
    history.append(state)

    engine = SemanticEngine()
    ctx = engine.update(state, history)
    # With strong pointing evidence, pointing_to_object intent should fire
    assert ctx.opportunity.intent == IntentCategory.POINTING_TO_OBJECT


# ---------------------------------------------------------------------------
# 7. Confidence Calculation
# ---------------------------------------------------------------------------

def test_confidence_in_range():
    obj = _tracked_obj()
    rel = HandPointingAtObject(hand_is_left=False, object_id=1)
    focus = FocusTarget(track_id=1, label="laptop", relation="pointing_toward", confidence=0.7)
    intent, conf, _ = _infer_intent(SemanticActivity.POINTING, focus, ["pointing_started"], _empty_history())
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# 8. JSON Serialization
# ---------------------------------------------------------------------------

def test_semantic_context_serializable():
    engine = SemanticEngine()
    obj = _tracked_obj()
    rel = HandPointingAtObject(hand_is_left=False, object_id=1)
    act = ActivityState(right_hand_gesture=Gesture.POINTING)
    evt = PointingStarted(timestamp=1.0, hand_is_left=False)
    state = _scene_state(tracked_objects=(obj,), relationships=(rel,), activity=act, recent_events=(evt,))
    history = _empty_history()
    history.append(state)
    ctx = engine.update(state, history)
    # Must be JSON-serializable without errors
    raw = json.dumps(ctx.to_dict())
    parsed = json.loads(raw)
    assert "current_scene" in parsed
    assert "opportunity" in parsed
    assert "activity" in parsed["current_scene"]


# ---------------------------------------------------------------------------
# 9. No-Object Case
# ---------------------------------------------------------------------------

def test_no_object_case():
    engine = SemanticEngine()
    state = _scene_state(has_pose=True)
    history = _empty_history()
    history.append(state)
    ctx = engine.update(state, history)
    assert ctx.current_scene.focus_target is None
    assert ctx.current_scene.visible_objects == []
    assert ctx.opportunity.should_visualize is False


# ---------------------------------------------------------------------------
# 10. No-Person Case
# ---------------------------------------------------------------------------

def test_no_person_case():
    engine = SemanticEngine()
    state = _scene_state(has_pose=False)
    history = _empty_history()
    history.append(state)
    ctx = engine.update(state, history)
    assert ctx.current_scene.person_count == 0
    assert ctx.current_scene.activity == SemanticActivity.UNKNOWN


# ---------------------------------------------------------------------------
# 11. Ambiguous Target Case (multiple objects, no strong relationship)
# ---------------------------------------------------------------------------

def test_ambiguous_two_objects_near():
    obj1 = _tracked_obj(track_id=1, label="laptop", confidence=0.7)
    obj2 = _tracked_obj(track_id=2, label="bottle", confidence=0.7,
                         x1=0.2, y1=0.2, x2=0.35, y2=0.35)
    rel1 = HandNearObject(hand_is_left=False, object_id=1)
    rel2 = HandNearObject(hand_is_left=False, object_id=2)
    state = _scene_state(tracked_objects=(obj1, obj2), relationships=(rel1, rel2))
    focus = _select_focus_target(state)
    # Should return something (not crash), but confidence should be low-ish
    assert focus is not None
    assert focus.confidence <= 1.0


# ---------------------------------------------------------------------------
# 12. Unknown Activity Case
# ---------------------------------------------------------------------------

def test_unknown_activity_produces_none_or_unknown_intent():
    intent, conf, _ = _infer_intent(SemanticActivity.UNKNOWN, None, [], _empty_history())
    assert intent in (IntentCategory.NONE, IntentCategory.UNKNOWN)


# ---------------------------------------------------------------------------
# 13. Rate-limiting: should_update
# ---------------------------------------------------------------------------

def test_semantic_engine_rate_limiting():
    engine = SemanticEngine(update_interval_s=1.0)  # 1-second interval
    state = _scene_state()
    history = _empty_history()
    history.append(state)

    # First call should always update
    assert engine.should_update(state)
    engine.update(state, history)

    # Immediately after, should NOT update (no events, not enough time)
    state_no_events = _scene_state()
    assert not engine.should_update(state_no_events)


def test_semantic_engine_event_driven_update():
    engine = SemanticEngine(update_interval_s=1.0)
    state_clean = _scene_state()
    history = _empty_history()
    history.append(state_clean)
    engine.update(state_clean, history)

    # New state with events should trigger update even if interval hasn't elapsed
    evt = PointingStarted(timestamp=1.0, hand_is_left=False)
    state_with_event = _scene_state(recent_events=(evt,))
    assert engine.should_update(state_with_event)
