from __future__ import annotations

import cv2

from punity.models import AppState, GestureFrame, HandObservation

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


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


def draw_hand_skeleton(
    frame_bgr,
    observation: HandObservation,
    line_color: tuple[int, int, int] = (255, 210, 0),
    point_color: tuple[int, int, int] = (0, 255, 255),
) -> None:
    h, w = frame_bgr.shape[:2]
    points: list[tuple[int, int]] = []
    for lm in observation.landmarks:
        x = int(max(0.0, min(1.0, lm.x)) * (w - 1))
        y = int(max(0.0, min(1.0, lm.y)) * (h - 1))
        points.append((x, y))

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, points[a], points[b], line_color, 2, cv2.LINE_AA)

    for x, y in points:
        cv2.circle(frame_bgr, (x, y), 3, point_color, -1, cv2.LINE_AA)
