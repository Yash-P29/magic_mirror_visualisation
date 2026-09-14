"""
models.py — Data structures for Semantic Scene Understanding and Intent Inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SemanticActivity(str, Enum):
    """Deterministic activities inferred from low-level perception."""
    IDLE = "idle"
    STANDING = "standing"
    MOVING = "moving"
    GESTURING = "gesturing"
    POINTING = "pointing"
    INTERACTING_WITH_OBJECT = "interacting_with_object"
    PRESENTING = "presenting"
    UNKNOWN = "unknown"


class IntentCategory(str, Enum):
    """High-level intent inferred from activity and focus."""
    NONE = "none"
    EXPLAINING = "explaining"
    POINTING_TO_OBJECT = "pointing_to_object"
    DEMONSTRATING = "demonstrating"
    COMPARING = "comparing"
    EMPHASIZING = "emphasizing"
    VISUALIZATION_CANDIDATE = "visualization_candidate"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FocusTarget:
    """Represents an object that the user is currently focused on."""
    track_id: int
    label: str
    relation: str
    confidence: float


@dataclass(frozen=True)
class SemanticSceneState:
    """Compact semantic representation of a single moment in time."""
    timestamp: float
    activity: SemanticActivity
    person_count: int
    visible_objects: List[Dict[str, Any]]
    active_gestures: List[Dict[str, str]]
    focus_target: Optional[FocusTarget]
    recent_events: List[str]
    motion_summary: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "activity": self.activity.value,
            "person_count": self.person_count,
            "visible_objects": self.visible_objects,
            "active_gestures": self.active_gestures,
            "focus_target": {
                "track_id": self.focus_target.track_id,
                "label": self.focus_target.label,
                "relation": self.focus_target.relation,
                "confidence": self.focus_target.confidence,
            } if self.focus_target else None,
            "recent_events": self.recent_events,
            "motion_summary": self.motion_summary,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class VisualizationOpportunity:
    """Structured output defining whether visualization is recommended."""
    should_visualize: bool
    intent: IntentCategory
    concept: Optional[str]
    target_track_id: Optional[int]
    target_label: Optional[str]
    anchor_type: Optional[str]
    anchor_track_id: Optional[int]
    confidence: float
    reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_visualize": self.should_visualize,
            "intent": self.intent.value,
            "concept": self.concept,
            "target": {
                "track_id": self.target_track_id,
                "label": self.target_label
            } if self.target_track_id is not None else None,
            "anchor": {
                "type": self.anchor_type,
                "track_id": self.anchor_track_id
            } if self.anchor_track_id is not None else None,
            "confidence": self.confidence,
            "reason": self.reasons,
        }


@dataclass(frozen=True)
class SemanticContext:
    """A container packaging recent state and events for future VLM consumption."""
    current_scene: SemanticSceneState
    opportunity: VisualizationOpportunity
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_scene": self.current_scene.to_dict(),
            "opportunity": self.opportunity.to_dict(),
        }
