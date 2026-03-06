from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GestureLabel(str, Enum):
    NONE = "NONE"
    OPEN_PALM = "OPEN_PALM"
    PINCHING = "PINCHING"
    FIST = "FIST"
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"


class AppState(str, Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    POINTING = "POINTING"
    DRAGGING = "DRAGGING"
    COMMAND = "COMMAND"


class EventType(str, Enum):
    MOVE_CURSOR = "MOVE_CURSOR"
    MOUSE_DOWN_LEFT = "MOUSE_DOWN_LEFT"
    MOUSE_UP_LEFT = "MOUSE_UP_LEFT"
    HOTKEY = "HOTKEY"
    TOGGLE_ACTIVE = "TOGGLE_ACTIVE"


@dataclass(slots=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class HandObservation:
    landmarks: list[Landmark]
    handedness: str
    detection_confidence: float
    tracked_at_ms: int


@dataclass(slots=True)
class GestureFrame:
    label: GestureLabel
    cursor_point_norm: tuple[float, float] | None
    pinch_strength: float
    confidence: float
    pinch_distance_norm: float
    swipe: str | None = None


@dataclass(slots=True)
class ControlEvent:
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18
WRIST = 0