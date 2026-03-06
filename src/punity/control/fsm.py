from __future__ import annotations

from dataclasses import dataclass, field

from punity.models import AppState, ControlEvent, EventType, GestureFrame, GestureLabel
from punity.tracking.hand_state import HandStateStatus


@dataclass(slots=True)
class ControlConfig:
    min_confidence: float
    require_open_palm: bool
    hotkey_cooldown_ms: int
    pinch_click_cooldown_ms: int = 200
    scroll_repeat_ms: int = 80


@dataclass
class ControlFSM:
    config: ControlConfig
    state: AppState = AppState.IDLE
    active: bool = True
    _mouse_down: bool = False
    _last_label: GestureLabel = GestureLabel.NONE
    _last_hotkey_ms: dict[str, int] = field(default_factory=dict)
    _last_pinch_click_ms: int = -10000
    _last_scroll_emit_ms: int = -10000

    def set_active(self, active: bool) -> None:
        self.active = active
        if not active:
            self.state = AppState.IDLE

    def step(
        self,
        gesture: GestureFrame,
        hand_status: HandStateStatus,
        t_ms: int,
        mappings: dict[str, dict[str, object]],
    ) -> list[ControlEvent]:
        events: list[ControlEvent] = []

        if not self.active:
            self._release_if_needed(events)
            self._last_label = gesture.label
            return events

        if not hand_status.present or hand_status.confidence < self.config.min_confidence:
            self._release_if_needed(events)
            self.state = AppState.ARMED
            self._last_label = gesture.label
            return events

        if gesture.label == GestureLabel.FIST:
            self._release_if_needed(events)
            self.state = AppState.IDLE
            self._last_label = gesture.label
            return events

        if hand_status.idle:
            self._release_if_needed(events)
            self.state = AppState.ARMED
            self._last_label = gesture.label
            return events

        if self.config.require_open_palm:
            can_point = gesture.label in (GestureLabel.POINTER, GestureLabel.PINCHING, GestureLabel.PINKY_DRAG)
        else:
            can_point = gesture.label != GestureLabel.FIST

        if can_point and gesture.cursor_point_norm is not None:
            events.append(
                ControlEvent(
                    event_type=EventType.MOVE_CURSOR,
                    payload={"point_norm": gesture.cursor_point_norm},
                )
            )
            self.state = AppState.POINTING

        # Pinky gesture drives click-and-hold drag.
        if gesture.label == GestureLabel.PINKY_DRAG:
            if not self._mouse_down:
                self._mouse_down = True
                events.append(ControlEvent(event_type=EventType.MOUSE_DOWN_LEFT))
            self.state = AppState.DRAGGING
        elif self._mouse_down:
            self._mouse_down = False
            events.append(ControlEvent(event_type=EventType.MOUSE_UP_LEFT))
            if can_point:
                self.state = AppState.POINTING
            else:
                self.state = AppState.ARMED

        # Pinch is a debounced click (not hold).
        pinch_edge = gesture.label == GestureLabel.PINCHING and self._last_label != GestureLabel.PINCHING
        if pinch_edge and (t_ms - self._last_pinch_click_ms) >= self.config.pinch_click_cooldown_ms:
            events.append(ControlEvent(event_type=EventType.MOUSE_DOWN_LEFT))
            events.append(ControlEvent(event_type=EventType.MOUSE_UP_LEFT))
            self._last_pinch_click_ms = t_ms

        # Scroll gesture repeats while pose is held.
        if gesture.label in (GestureLabel.SCROLL_UP, GestureLabel.SCROLL_DOWN):
            if t_ms - self._last_scroll_emit_ms >= self.config.scroll_repeat_ms:
                dy = 2 if gesture.label == GestureLabel.SCROLL_UP else -2
                events.append(ControlEvent(event_type=EventType.SCROLL, payload={"dx": 0, "dy": dy}))
                self._last_scroll_emit_ms = t_ms
            self.state = AppState.COMMAND

        self._emit_mapped_action(events, gesture.label, t_ms, mappings)

        self._last_label = gesture.label
        return events

    def _release_if_needed(self, events: list[ControlEvent]) -> None:
        if self._mouse_down:
            self._mouse_down = False
            events.append(ControlEvent(event_type=EventType.MOUSE_UP_LEFT))

    def _emit_mapped_action(
        self,
        events: list[ControlEvent],
        label: GestureLabel,
        t_ms: int,
        mappings: dict[str, dict[str, object]],
    ) -> None:
        action = mappings.get(label.value)
        if action is None:
            return

        # Fire mapped gesture action on label edge only.
        if self._last_label == label:
            return

        cooldown_ms = int(action.get("cooldown_ms", self.config.hotkey_cooldown_ms))
        key = label.value
        last = self._last_hotkey_ms.get(key, 0)
        if t_ms - last < cooldown_ms:
            return

        event_name = str(action.get("event", "")).upper()
        if event_name != EventType.HOTKEY.value:
            return

        keys = action.get("keys")
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            return

        self._last_hotkey_ms[key] = t_ms
        events.append(
            ControlEvent(
                event_type=EventType.HOTKEY,
                payload={"keys": keys},
            )
        )