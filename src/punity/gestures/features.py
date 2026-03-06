from __future__ import annotations

import math
from dataclasses import dataclass

from punity.models import (
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    HandObservation,
)


@dataclass(slots=True)
class GestureFeatures:
    pinch_distance_norm: float
    pinch_strength: float
    finger_extended: dict[str, bool]
    is_open_palm: bool
    is_fist: bool
    palm_scale: float
    cursor_point_norm: tuple[float, float]
    hand_center_norm: tuple[float, float]


def _distance_2d(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _finger_extensions(obs: HandObservation) -> dict[str, bool]:
    lm = obs.landmarks
    index_extended = lm[INDEX_TIP].y < lm[INDEX_PIP].y
    middle_extended = lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y
    ring_extended = lm[RING_TIP].y < lm[RING_PIP].y
    pinky_extended = lm[PINKY_TIP].y < lm[PINKY_PIP].y

    thumb_tip = lm[THUMB_TIP]
    wrist = lm[WRIST]
    thumb_extended = _distance_2d(thumb_tip.x, thumb_tip.y, wrist.x, wrist.y) > 0.18

    return {
        "thumb": thumb_extended,
        "index": index_extended,
        "middle": middle_extended,
        "ring": ring_extended,
        "pinky": pinky_extended,
    }


def compute_features(obs: HandObservation) -> GestureFeatures:
    lm = obs.landmarks

    wrist = lm[WRIST]
    mid_mcp = lm[9]
    palm_scale = _distance_2d(wrist.x, wrist.y, mid_mcp.x, mid_mcp.y)
    palm_scale = max(palm_scale, 1e-4)

    thumb = lm[THUMB_TIP]
    index = lm[INDEX_TIP]
    pinch_distance = _distance_2d(thumb.x, thumb.y, index.x, index.y)
    pinch_distance_norm = pinch_distance / palm_scale
    pinch_strength = 1.0 - _clamp((pinch_distance_norm - 0.20) / 0.5, 0.0, 1.0)

    extended = _finger_extensions(obs)
    extended_count = sum(1 for v in extended.values() if v)

    is_open_palm = extended_count >= 4 and extended["index"] and extended["middle"]
    is_fist = extended_count <= 1

    cursor_point = (_clamp(index.x, 0.0, 1.0), _clamp(index.y, 0.0, 1.0))
    hand_center = (_clamp(wrist.x, 0.0, 1.0), _clamp(wrist.y, 0.0, 1.0))

    return GestureFeatures(
        pinch_distance_norm=pinch_distance_norm,
        pinch_strength=pinch_strength,
        finger_extended=extended,
        is_open_palm=is_open_palm,
        is_fist=is_fist,
        palm_scale=palm_scale,
        cursor_point_norm=cursor_point,
        hand_center_norm=hand_center,
    )
