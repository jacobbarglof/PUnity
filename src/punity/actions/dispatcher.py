from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass

from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.mouse import Button, Controller as MouseController

from punity.models import ControlEvent, EventType


_KEY_ALIASES = {
    "alt": Key.alt,
    "alt_l": Key.alt_l,
    "alt_r": Key.alt_r,
    "ctrl": Key.ctrl,
    "ctrl_l": Key.ctrl_l,
    "ctrl_r": Key.ctrl_r,
    "shift": Key.shift,
    "shift_l": Key.shift_l,
    "shift_r": Key.shift_r,
    "cmd": Key.cmd,
    "cmd_l": Key.cmd_l,
    "cmd_r": Key.cmd_r,
    "tab": Key.tab,
    "left": Key.left,
    "right": Key.right,
    "up": Key.up,
    "down": Key.down,
    "space": Key.space,
    "enter": Key.enter,
    "esc": Key.esc,
    "play_pause": Key.media_play_pause,
    "next": Key.media_next,
    "prev": Key.media_previous,
    "volume_up": Key.media_volume_up,
    "volume_down": Key.media_volume_down,
    "mute": Key.media_volume_mute,
}


@dataclass(slots=True)
class CursorConfig:
    sensitivity: float
    accel: float
    deadzone_px: float
    edge_padding_px: float


class ActionDispatcher:
    def __init__(self, cursor_config: CursorConfig) -> None:
        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        self._cursor_config = cursor_config
        self._screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        self._screen_h = ctypes.windll.user32.GetSystemMetrics(1)

    def execute(self, event: ControlEvent) -> None:
        if event.event_type == EventType.MOVE_CURSOR:
            point_norm = event.payload.get("point_norm")
            if not point_norm:
                return
            self._move_cursor(point_norm)
            return

        if event.event_type == EventType.MOUSE_DOWN_LEFT:
            self._mouse.press(Button.left)
            return

        if event.event_type == EventType.MOUSE_UP_LEFT:
            self._mouse.release(Button.left)
            return

        if event.event_type == EventType.SCROLL:
            dx = int(event.payload.get("dx", 0))
            dy = int(event.payload.get("dy", 0))
            if dx != 0 or dy != 0:
                self._mouse.scroll(dx, dy)
            return

        if event.event_type == EventType.HOTKEY:
            keys = event.payload.get("keys", [])
            if isinstance(keys, list):
                self._tap_hotkey(keys)

    def _move_cursor(self, point_norm: tuple[float, float]) -> None:
        nx = max(0.0, min(1.0, float(point_norm[0])))
        ny = max(0.0, min(1.0, float(point_norm[1])))

        # Overscan padding makes screen edges easier to reach without extreme hand motion.
        edge_pad = max(0.0, float(self._cursor_config.edge_padding_px))
        pad_x = min(edge_pad, (self._screen_w - 1) / 2.0)
        pad_y = min(edge_pad, (self._screen_h - 1) / 2.0)

        target_x = int((-pad_x) + nx * ((self._screen_w - 1) + 2.0 * pad_x))
        target_y = int((-pad_y) + ny * ((self._screen_h - 1) + 2.0 * pad_y))

        curr_x, curr_y = self._mouse.position
        dx = target_x - curr_x
        dy = target_y - curr_y
        dist = math.hypot(dx, dy)
        if dist < self._cursor_config.deadzone_px:
            return

        accel_gain = 1.0 + self._cursor_config.accel * min(1.0, dist / 400.0)
        sx = int(curr_x + dx * self._cursor_config.sensitivity * accel_gain)
        sy = int(curr_y + dy * self._cursor_config.sensitivity * accel_gain)
        sx = max(0, min(self._screen_w - 1, sx))
        sy = max(0, min(self._screen_h - 1, sy))
        self._mouse.position = (sx, sy)

    def _tap_hotkey(self, keys: list[str]) -> None:
        mapped = [self._map_key(k) for k in keys]
        mapped = [k for k in mapped if k is not None]
        if not mapped:
            return

        modifiers = mapped[:-1]
        trigger = mapped[-1]
        for key in modifiers:
            self._keyboard.press(key)
        self._keyboard.press(trigger)
        self._keyboard.release(trigger)
        for key in reversed(modifiers):
            self._keyboard.release(key)

    def _map_key(self, token: str):
        token_l = token.lower().strip()
        if len(token_l) == 1:
            return token_l
        return _KEY_ALIASES.get(token_l)