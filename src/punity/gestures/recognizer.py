from __future__ import annotations

from dataclasses import dataclass

from punity.gestures.features import compute_features
from punity.models import GestureFrame, GestureLabel, HandObservation


@dataclass(slots=True)
class GestureConfig:
    pinch_on: float
    pinch_off: float
    min_confidence: float


class GestureRecognizer:
    def __init__(self, config: GestureConfig) -> None:
        self._config = config
        self._pinching = False

    def recognize(self, observation: HandObservation | None, t_ms: int) -> GestureFrame:
        del t_ms

        if observation is None:
            self._pinching = False
            return GestureFrame(
                label=GestureLabel.NONE,
                cursor_point_norm=None,
                pinch_strength=0.0,
                confidence=0.0,
                pinch_distance_norm=999.0,
                swipe=None,
            )

        features = compute_features(observation)
        pinch_dist = features.pinch_distance_norm

        if self._pinching:
            if pinch_dist >= self._config.pinch_off:
                self._pinching = False
        elif pinch_dist <= self._config.pinch_on:
            self._pinching = True

        # Priority: crossed toggle > fist safety > scroll pose > pinky-drag > pinch-click > pointer.
        label = GestureLabel.NONE
        if features.is_fingers_crossed:
            label = GestureLabel.FINGERS_CROSSED
        elif features.is_fist:
            label = GestureLabel.FIST
        elif features.is_scroll_up_pose:
            label = GestureLabel.SCROLL_UP
        elif features.is_scroll_down_pose:
            label = GestureLabel.SCROLL_DOWN
        elif features.is_pinky_drag:
            label = GestureLabel.PINKY_DRAG
        elif self._pinching:
            label = GestureLabel.PINCHING
        elif features.is_pointer:
            label = GestureLabel.POINTER

        conf = min(observation.detection_confidence, 1.0)
        return GestureFrame(
            label=label,
            cursor_point_norm=features.cursor_point_norm,
            pinch_strength=features.pinch_strength,
            confidence=conf,
            pinch_distance_norm=features.pinch_distance_norm,
            swipe=None,
        )