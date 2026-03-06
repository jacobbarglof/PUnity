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

        self._last_swipe_point: tuple[float, float] | None = None
        self._swipe_anchor: tuple[float, float] | None = None
        self._last_t_ms: int | None = None
        self._last_swipe_ms = 0
        self._smoothed_vx = 0.0

    def recognize(self, observation: HandObservation | None, t_ms: int) -> GestureFrame:
        if observation is None:
            self._pinching = False
            self._last_swipe_point = None
            self._swipe_anchor = None
            self._last_t_ms = None
            self._smoothed_vx = 0.0
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

        can_swipe = features.is_open_palm and not self._pinching and not features.is_fist

        swipe = None
        if self._config.swipe_enabled:
            swipe = self._detect_swipe(features.cursor_point_norm, t_ms, can_swipe)

        # Prioritize FIST over PINCHING so a closed hand is always an immediate stop signal.
        label = GestureLabel.NONE
        if features.is_fist:
            label = GestureLabel.FIST
        elif self._pinching:
            label = GestureLabel.PINCHING
        elif features.is_open_palm:
            label = GestureLabel.OPEN_PALM

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
        point: tuple[float, float],
        t_ms: int,
        can_swipe: bool,
    ) -> str | None:
        if self._last_swipe_point is None or self._last_t_ms is None:
            self._last_swipe_point = point
            self._swipe_anchor = point
            self._last_t_ms = t_ms
            self._smoothed_vx = 0.0
            return None

        dt_ms = max(1, t_ms - self._last_t_ms)
        dt_s = dt_ms / 1000.0
        dx = point[0] - self._last_swipe_point[0]
        dy = point[1] - self._last_swipe_point[1]
        vx = dx / dt_s
        vy = dy / dt_s

        alpha = min(1.0, dt_ms / 45.0)
        self._smoothed_vx = (1.0 - alpha) * self._smoothed_vx + alpha * vx

        self._last_swipe_point = point
        self._last_t_ms = t_ms

        if not can_swipe:
            self._swipe_anchor = point
            self._smoothed_vx = 0.0
            return None

        if self._swipe_anchor is None:
            self._swipe_anchor = point

        disp_x = point[0] - self._swipe_anchor[0]

        if t_ms - self._last_swipe_ms < self._config.swipe_cooldown_ms:
            return None

        if abs(disp_x) < 0.10:
            return None

        if abs(self._smoothed_vx) < self._config.swipe_velocity_threshold:
            return None

        if abs(vy) > abs(self._smoothed_vx) * 0.8:
            return None

        self._last_swipe_ms = t_ms
        self._swipe_anchor = point
        return "RIGHT" if self._smoothed_vx > 0 else "LEFT"
