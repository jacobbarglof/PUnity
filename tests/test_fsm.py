from punity.control.fsm import ControlConfig, ControlFSM
from punity.models import GestureFrame, GestureLabel
from punity.tracking.hand_state import HandStateStatus


def _gesture(label: GestureLabel, point=(0.5, 0.5)) -> GestureFrame:
    return GestureFrame(
        label=label,
        cursor_point_norm=point,
        pinch_strength=0.8 if label == GestureLabel.PINCHING else 0.0,
        confidence=0.9,
        pinch_distance_norm=0.5,
        swipe=None,
    )


def _status() -> HandStateStatus:
    return HandStateStatus(present=True, confidence=0.9, just_lost=False, idle=False)


def test_drag_press_and_release() -> None:
    fsm = ControlFSM(
        config=ControlConfig(min_confidence=0.6, require_open_palm=True, hotkey_cooldown_ms=800)
    )

    events_1 = fsm.step(_gesture(GestureLabel.PINCHING), _status(), 1000, {})
    assert any(e.event_type.value == "MOUSE_DOWN_LEFT" for e in events_1)

    events_2 = fsm.step(_gesture(GestureLabel.OPEN_PALM), _status(), 1020, {})
    assert any(e.event_type.value == "MOUSE_UP_LEFT" for e in events_2)


def test_hotkey_mapping_edge_only() -> None:
    fsm = ControlFSM(
        config=ControlConfig(min_confidence=0.6, require_open_palm=True, hotkey_cooldown_ms=800)
    )
    mappings = {
        "SWIPE_LEFT": {
            "event": "HOTKEY",
            "keys": ["cmd", "ctrl", "left"],
            "cooldown_ms": 1000,
        }
    }

    first = fsm.step(_gesture(GestureLabel.SWIPE_LEFT), _status(), 2000, mappings)
    second = fsm.step(_gesture(GestureLabel.SWIPE_LEFT), _status(), 2010, mappings)

    assert sum(1 for e in first if e.event_type.value == "HOTKEY") == 1
    assert sum(1 for e in second if e.event_type.value == "HOTKEY") == 0
