from __future__ import annotations

from dataclasses import dataclass

from punity.models import HandObservation


@dataclass(slots=True)
class HandStateStatus:
    present: bool
    confidence: float
    just_lost: bool
    idle: bool


class HandStateTracker:
    def __init__(self, lost_timeout_ms: int = 500, idle_timeout_ms: int = 4000) -> None:
        self.lost_timeout_ms = lost_timeout_ms
        self.idle_timeout_ms = idle_timeout_ms
        self._last_seen_ms: int | None = None
        self._last_move_ms: int | None = None
        self._last_point: tuple[float, float] | None = None
        self._present = False

    def update(
        self,
        observation: HandObservation | None,
        t_ms: int,
        cursor_point_norm: tuple[float, float] | None,
    ) -> HandStateStatus:
        previous_present = self._present

        if observation is not None:
            self._last_seen_ms = t_ms
            self._present = True
            confidence = observation.detection_confidence
        else:
            if self._last_seen_ms is None:
                self._present = False
                confidence = 0.0
            else:
                dt = t_ms - self._last_seen_ms
                if dt > self.lost_timeout_ms:
                    self._present = False
                    confidence = 0.0
                else:
                    self._present = True
                    confidence = max(0.0, 1.0 - dt / self.lost_timeout_ms)

        if cursor_point_norm is not None:
            if self._last_point is None:
                self._last_move_ms = t_ms
            else:
                dx = abs(cursor_point_norm[0] - self._last_point[0])
                dy = abs(cursor_point_norm[1] - self._last_point[1])
                if dx + dy > 0.003:
                    self._last_move_ms = t_ms
            self._last_point = cursor_point_norm

        idle = False
        if self._last_move_ms is not None and t_ms - self._last_move_ms >= self.idle_timeout_ms:
            idle = True

        just_lost = previous_present and not self._present
        return HandStateStatus(
            present=self._present,
            confidence=confidence,
            just_lost=just_lost,
            idle=idle,
        )
