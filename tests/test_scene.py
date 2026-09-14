import pytest
from unittest.mock import Mock

from src.perception.models import (
    HandData, Landmark, PerceptionResult, PoseData, Gesture, ActivityState, 
    HandPointingAtObject, HandOverlappingObject, BoundingBox, TrackedObject
)
from src.perception.scene import _is_pointing, _is_raised_hand, SceneBuilder

def test_is_pointing_heuristic():
    # Setup hand landmarks for pointing
    # Wrist
    lm_wrist = Landmark(x=0.5, y=0.5, z=0)
    # Index extended
    lm_index_mcp = Landmark(x=0.5, y=0.4, z=0)
    lm_index_tip = Landmark(x=0.5, y=0.2, z=0)
    # Others curled
    lm_mid_tip = Landmark(x=0.5, y=0.45, z=0)
    lm_ring_tip = Landmark(x=0.5, y=0.45, z=0)
    lm_pinky_tip = Landmark(x=0.5, y=0.45, z=0)
    
    landmarks = [None] * 21
    landmarks[0] = lm_wrist
    landmarks[5] = lm_index_mcp
    landmarks[8] = lm_index_tip
    landmarks[12] = lm_mid_tip
    landmarks[16] = lm_ring_tip
    landmarks[20] = lm_pinky_tip
    
    # Fill remaining with dummies to avoid None errors if accessed
    for i in range(21):
        if landmarks[i] is None:
            landmarks[i] = Landmark(x=0.5, y=0.5, z=0)
            
    hand = HandData(handedness="Right", landmarks=tuple(landmarks))
    
    assert _is_pointing(hand) == True
    
def test_is_raised_hand_heuristic():
    lm_wrist = Landmark(x=0.5, y=0.1, z=0) # high up (y=0 is top)
    landmarks = [Landmark(x=0.5, y=0.5, z=0)] * 21
    landmarks[0] = lm_wrist
    hand = HandData(handedness="Right", landmarks=tuple(landmarks))
    
    pose_lms = [Landmark(x=0.5, y=0.5, z=0)] * 33
    pose_lms[0] = Landmark(x=0.5, y=0.2, z=0) # nose
    pose_lms[12] = Landmark(x=0.5, y=0.3, z=0) # right shoulder
    
    pose = PoseData(landmarks=tuple(pose_lms))
    
    assert _is_raised_hand(hand, pose) == True
