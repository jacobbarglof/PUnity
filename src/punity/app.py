from __future__ import annotations

import argparse
import sys
import time

import cv2
from pynput import keyboard

from punity.actions.dispatcher import ActionDispatcher, CursorConfig
from punity.capture.camera import CameraCapture, CameraConfig
from punity.config.profile import AppProfile, load_profile
from punity.control.fsm import ControlConfig, ControlFSM
from punity.gestures.recognizer import GestureConfig, GestureRecognizer
from punity.models import GestureLabel
from punity.tracking.filters import EMAFilter2D, OneEuroFilter2D
from punity.tracking.hand_state import HandStateTracker
from punity.ui.overlay import draw_overlay


class GestureMouseApp:
    def __init__(self, profile: AppProfile) -> None:
        from punity.perception.hands import HandsDetector

        self._profile = profile
        self._camera = CameraCapture(
            CameraConfig(
                device_index=profile.camera.device_index,
                width=profile.camera.width,
                height=profile.camera.height,
                fps=profile.camera.fps,
            )
        )
        self._detector = HandsDetector(
            min_detection_confidence=profile.thresholds.min_detection_confidence,
            min_tracking_confidence=profile.thresholds.min_tracking_confidence,
            max_num_hands=1,
        )
        self._recognizer = GestureRecognizer(
            GestureConfig(
                pinch_on=profile.thresholds.pinch_on,
                pinch_off=profile.thresholds.pinch_off,
                min_confidence=profile.thresholds.min_confidence,
            )
        )
        self._hand_state = HandStateTracker(
            lost_timeout_ms=450,
            idle_timeout_ms=profile.safety.idle_timeout_ms,
        )
        self._fsm = ControlFSM(
            config=ControlConfig(
                min_confidence=profile.thresholds.min_confidence,
                require_open_palm=profile.safety.require_open_palm,
                hotkey_cooldown_ms=900,
            )
        )
        self._dispatcher = ActionDispatcher(
            CursorConfig(
                sensitivity=profile.cursor.sensitivity,
                accel=profile.cursor.accel,
                deadzone_px=profile.cursor.deadzone_px,
                edge_padding_px=profile.cursor.edge_padding_px,
            )
        )

        if profile.cursor.filter == "ema":
            self._cursor_filter = EMAFilter2D(alpha=profile.cursor.smoothing)
        else:
            self._cursor_filter = OneEuroFilter2D(
                freq=max(5.0, float(profile.camera.fps)),
                min_cutoff=max(0.5, profile.cursor.smoothing * 4.0),
                beta=0.02,
                d_cutoff=1.2,
            )

        self._active = True
        self._crossed_latch = False
        self._cross_hold_start_ms: int | None = None
        self._cross_last_toggle_ms = -10000
        self._fps = 0.0
        self._last_frame_t = time.monotonic()
        self._kill_switch_token = profile.safety.kill_switch_key.lower().strip()
        self._listener = keyboard.Listener(on_press=self._on_key_press)

    def _toggle_active(self) -> None:
        self._active = not self._active
        self._fsm.set_active(self._active)

    def _on_key_press(self, key) -> None:
        if _key_matches(key, self._kill_switch_token):
            self._toggle_active()

    def run(self) -> None:
        self._listener.start()
        try:
            while True:
                frame_bgr, t_ms = self._camera.read_frame()
                if self._profile.overlay.mirror_preview:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                detect_bgr = frame_bgr
                frame_h, frame_w = frame_bgr.shape[:2]
                if frame_w > 640:
                    detect_h = max(1, int(frame_h * (640.0 / frame_w)))
                    detect_bgr = cv2.resize(frame_bgr, (640, detect_h), interpolation=cv2.INTER_LINEAR)

                frame_rgb = self._detector.bgr_to_rgb(detect_bgr)
                observation = self._detector.detect(frame_rgb, t_ms)

                gesture = self._recognizer.recognize(observation, t_ms)
                if gesture.label == GestureLabel.FINGERS_CROSSED:
                    if self._cross_hold_start_ms is None:
                        self._cross_hold_start_ms = t_ms

                    hold_elapsed = t_ms - self._cross_hold_start_ms
                    cooldown_elapsed = t_ms - self._cross_last_toggle_ms
                    if hold_elapsed >= 400 and cooldown_elapsed >= 1000 and not self._crossed_latch:
                        self._toggle_active()
                        self._crossed_latch = True
                        self._cross_last_toggle_ms = t_ms
                else:
                    self._cross_hold_start_ms = None
                    self._crossed_latch = False

                if gesture.cursor_point_norm is not None:
                    gesture.cursor_point_norm = self._cursor_filter.update(gesture.cursor_point_norm)
                else:
                    self._cursor_filter.reset()

                hand_status = self._hand_state.update(observation, t_ms, gesture.cursor_point_norm)

                self._fsm.set_active(self._active)
                events = self._fsm.step(gesture, hand_status, t_ms, self._profile.mappings)
                for event in events:
                    self._dispatcher.execute(event)

                self._update_fps()

                if self._profile.overlay.enabled:
                    draw_overlay(
                        frame_bgr=frame_bgr,
                        state=self._fsm.state,
                        gesture=gesture,
                        fps=self._fps,
                        active=self._active,
                        idle=hand_status.idle,
                    )
                    cv2.imshow("PUnity", frame_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        finally:
            self._listener.stop()
            self._camera.release()
            self._detector.close()
            cv2.destroyAllWindows()

    def _update_fps(self) -> None:
        now = time.monotonic()
        dt = now - self._last_frame_t
        self._last_frame_t = now
        if dt <= 0.0:
            return
        inst = 1.0 / dt
        if self._fps <= 0.0:
            self._fps = inst
        else:
            self._fps = self._fps * 0.9 + inst * 0.1


def _key_matches(key, token: str) -> bool:
    if len(token) == 1:
        return getattr(key, "char", "") == token

    key_name = token
    if not key_name.startswith("Key."):
        key_name = f"Key.{key_name}"

    return str(key) == key_name


def _ensure_supported_python() -> None:
    if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
        ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            f"Python {ver} detected. PUnity requires Python 3.12.x for MediaPipe Hands support."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PUnity gesture mouse controller")
    parser.add_argument(
        "--profile",
        default="profiles/default.json",
        help="Path to configuration JSON profile",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_supported_python()
    args = parse_args()
    profile = load_profile(args.profile)
    app = GestureMouseApp(profile)
    app.run()


if __name__ == "__main__":
    main()
