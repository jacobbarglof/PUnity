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
    is_pointer: bool
    is_pinky_drag: bool
    is_scroll_up_pose: bool
    is_scroll_down_pose: bool
    is_open_palm: bool
    is_fist: bool
    is_fingers_crossed: bool
    palm_scale: float
    cursor_point_norm: tuple[float, float]
    hand_center_norm: tuple[float, float]


def _distance_2d(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _finger_is_extended(obs: HandObservation, tip_idx: int, pip_idx: int) -> bool:
    lm = obs.landmarks
    wrist = lm[WRIST]
    tip = lm[tip_idx]
    pip = lm[pip_idx]

    tip_to_wrist = _distance_2d(tip.x, tip.y, wrist.x, wrist.y)
    pip_to_wrist = _distance_2d(pip.x, pip.y, wrist.x, wrist.y)
    return tip_to_wrist > pip_to_wrist * 1.06


def compute_features(obs: HandObservation) -> GestureFeatures:
    lm = obs.landmarks

    wrist = lm[WRIST]
    mid_mcp = lm[9]
    palm_scale = _distance_2d(wrist.x, wrist.y, mid_mcp.x, mid_mcp.y)
    palm_scale = max(palm_scale, 1e-4)

    thumb = lm[THUMB_TIP]
    index = lm[INDEX_TIP]
    middle = lm[MIDDLE_TIP]
    pinch_distance = _distance_2d(thumb.x, thumb.y, index.x, index.y)
    pinch_distance_norm = pinch_distance / palm_scale
    pinch_strength = 1.0 - _clamp((pinch_distance_norm - 0.20) / 0.5, 0.0, 1.0)

    # Landmark IDs: thumb_mcp=2, index_mcp=5, middle_mcp=9.
    thumb_mcp = lm[2]
    index_mcp = lm[5]
    middle_mcp = lm[9]
    thumb_tip_to_wrist = _distance_2d(thumb.x, thumb.y, wrist.x, wrist.y)
    thumb_mcp_to_wrist = _distance_2d(thumb_mcp.x, thumb_mcp.y, wrist.x, wrist.y)
    thumb_tip_to_index_mcp = _distance_2d(thumb.x, thumb.y, index_mcp.x, index_mcp.y)
    thumb_extended = (
        thumb_tip_to_wrist > thumb_mcp_to_wrist * 1.15
        and thumb_tip_to_index_mcp > palm_scale * 0.55
    )

    index_extended = _finger_is_extended(obs, INDEX_TIP, INDEX_PIP)
    middle_extended = _finger_is_extended(obs, MIDDLE_TIP, MIDDLE_PIP)
    ring_extended = _finger_is_extended(obs, RING_TIP, RING_PIP)
    pinky_extended = _finger_is_extended(obs, PINKY_TIP, PINKY_PIP)

    # Fingers-crossed: index+middle extended, tips close, and their tip ordering swaps
    # compared to MCP ordering across the hand.
    index_middle_tip_dist_norm = _distance_2d(index.x, index.y, middle.x, middle.y) / palm_scale
    mcp_order = index_mcp.x - middle_mcp.x
    tip_order = index.x - middle.x
    swapped_order = (mcp_order * tip_order) < 0.0
    is_fingers_crossed = (
        index_extended
        and middle_extended
        and index_middle_tip_dist_norm < 0.42
        and swapped_order
    )

    extended = {
        "thumb": thumb_extended,
        "index": index_extended,
        "middle": middle_extended,
        "ring": ring_extended,
        "pinky": pinky_extended,
    }
    extended_count = sum(1 for v in extended.values() if v)

    # Pointer gesture: index extended, remaining three fingers curled.
    is_pointer = (
        index_extended
        and not middle_extended
        and not ring_extended
        and not pinky_extended
        and pinch_distance_norm > 0.38
        and not is_fingers_crossed
    )

    # Pinky drag gesture: index + pinky extended, middle + ring curled.
    is_pinky_drag = (
        index_extended
        and pinky_extended
        and not middle_extended
        and not ring_extended
        and pinch_distance_norm > 0.40
        and not is_fingers_crossed
    )

    # Scroll pose base: index + middle outstretched, ring + pinky curled.
    scroll_base_pose = (
        index_extended
        and middle_extended
        and not ring_extended
        and not pinky_extended
        and pinch_distance_norm > 0.42
        and not is_fingers_crossed
    )

    # Direction from finger orientation in image space:
    # up => fingertip y above PIP y; down => fingertip y below PIP y.
    avg_tip_y = (index.y + middle.y) * 0.5
    avg_pip_y = (lm[INDEX_PIP].y + lm[MIDDLE_PIP].y) * 0.5
    orient_delta = avg_tip_y - avg_pip_y
    is_scroll_up_pose = scroll_base_pose and orient_delta < -0.02
    is_scroll_down_pose = scroll_base_pose and orient_delta > 0.02

    # Retained for compatibility/debugging.
    is_open_palm = (
        extended_count >= 4
        and index_extended
        and middle_extended
        and pinch_distance_norm > 0.55
        and not is_fingers_crossed
    )

    # Fist is four fingers curled plus thumb folded toward palm.
    curled_four = not index_extended and not middle_extended and not ring_extended and not pinky_extended
    thumb_folded = not thumb_extended or thumb_tip_to_index_mcp <= palm_scale * 0.48
    is_fist = curled_four and thumb_folded

    cursor_point = (_clamp(index.x, 0.0, 1.0), _clamp(index.y, 0.0, 1.0))
    hand_center = (_clamp(mid_mcp.x, 0.0, 1.0), _clamp(mid_mcp.y, 0.0, 1.0))

    return GestureFeatures(
        pinch_distance_norm=pinch_distance_norm,
        pinch_strength=pinch_strength,
        finger_extended=extended,
        is_pointer=is_pointer,
        is_pinky_drag=is_pinky_drag,
        is_scroll_up_pose=is_scroll_up_pose,
        is_scroll_down_pose=is_scroll_down_pose,
        is_open_palm=is_open_palm,
        is_fist=is_fist,
        is_fingers_crossed=is_fingers_crossed,
        palm_scale=palm_scale,
        cursor_point_norm=cursor_point,
        hand_center_norm=hand_center,
    )