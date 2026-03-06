from __future__ import annotations

from dataclasses import dataclass

from punity.gestures.features import compute_features
from punity.models import GestureFrame, GestureLabel, HandObservation


@dataclass(slots=True)
class GestureConfig:
    pinch_on: float
    pinch_off: float
    min_confidence: float
    swipe_enabled: bool
    swipe_velocity_threshold: float
    swipe_cooldown_ms: int


class GestureRecognizer:
    def __init__(self, config: GestureConfig) -> None:
        self._config = config
        self._pinching = False
        self._last_center: tuple[float, float] | None = None
        self._last_t_ms: int | None = None
        self._last_swipe_ms = 0

    def recognize(self, observation: HandObservation | None, t_ms: int) -> GestureFrame:
        if observation is None:
            self._pinching = False
            self._last_center = None
            self._last_t_ms = None
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

        label = GestureLabel.NONE
        if self._pinching:
            label = GestureLabel.PINCHING
        elif features.is_fist:
            label = GestureLabel.FIST
        elif features.is_open_palm:
            label = GestureLabel.OPEN_PALM

        swipe = None
        if self._config.swipe_enabled:
            swipe = self._detect_swipe(features.hand_center_norm, t_ms, label)
            if swipe == "LEFT":
                label = GestureLabel.SWIPE_LEFT
            elif swipe == "RIGHT":
                label = GestureLabel.SWIPE_RIGHT

        conf = min(observation.detection_confidence, 1.0)
        return GestureFrame(
            label=label,
            cursor_point_norm=features.cursor_point_norm,
            pinch_strength=features.pinch_strength,
            confidence=conf,
            pinch_distance_norm=features.pinch_distance_norm,
            swipe=swipe,
        )

    def _detect_swipe(
        self,
        center: tuple[float, float],
        t_ms: int,
        label: GestureLabel,
    ) -> str | None:
        if label not in (GestureLabel.OPEN_PALM, GestureLabel.NONE):
            self._last_center = center
            self._last_t_ms = t_ms
            return None

        if self._last_center is None or self._last_t_ms is None:
            self._last_center = center
            self._last_t_ms = t_ms
            return None

        dt_ms = max(1, t_ms - self._last_t_ms)
        dx = center[0] - self._last_center[0]
        dy = center[1] - self._last_center[1]
        vx = dx / (dt_ms / 1000.0)
        vy = dy / (dt_ms / 1000.0)

        self._last_center = center
        self._last_t_ms = t_ms

        if t_ms - self._last_swipe_ms < self._config.swipe_cooldown_ms:
            return None

        if abs(vx) >= self._config.swipe_velocity_threshold and abs(vy) < abs(vx) * 0.55:
            self._last_swipe_ms = t_ms
            return "RIGHT" if vx > 0 else "LEFT"

        return None
