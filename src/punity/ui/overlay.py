from __future__ import annotations

import cv2

from punity.models import AppState, GestureFrame


def draw_overlay(
    frame_bgr,
    state: AppState,
    gesture: GestureFrame,
    fps: float,
    active: bool,
    idle: bool,
) -> None:
    h, w = frame_bgr.shape[:2]

    status = "ACTIVE" if active else "PAUSED"
    color = (20, 220, 40) if active else (20, 20, 220)
    cv2.rectangle(frame_bgr, (8, 8), (w - 8, 125), (0, 0, 0), -1)
    cv2.rectangle(frame_bgr, (8, 8), (w - 8, 125), color, 2)

    lines = [
        f"Status: {status}",
        f"State: {state.value}",
        f"Gesture: {gesture.label.value}",
        f"Conf: {gesture.confidence:.2f}   Pinch: {gesture.pinch_strength:.2f}   FPS: {fps:.1f}",
    ]

    if idle:
        lines.append("Idle timeout active")

    y = 34
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 24
