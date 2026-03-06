from __future__ import annotations

import cv2
import mediapipe as mp

from punity.models import HandObservation, Landmark


def _resolve_hands_module():
    # Preferred API path on most MediaPipe builds.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return mp.solutions.hands

    # Compatibility path used by some package builds.
    try:
        from mediapipe.python import solutions as mp_solutions

        if hasattr(mp_solutions, "hands"):
            return mp_solutions.hands
    except Exception:
        pass

    raise RuntimeError(
        "MediaPipe Hands API is unavailable in this environment. "
        "Use Python 3.11 or 3.12 with mediapipe<=0.10.21, recreate your venv, and reinstall dependencies."
    )


class HandsDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        max_num_hands: int = 1,
    ) -> None:
        self._mp_hands = _resolve_hands_module()
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_rgb, t_ms: int) -> HandObservation | None:
        result = self._hands.process(frame_rgb)
        if not result.multi_hand_landmarks:
            return None

        landmarks_proto = result.multi_hand_landmarks[0]
        handedness = "UNKNOWN"
        confidence = 1.0
        if result.multi_handedness:
            hand_info = result.multi_handedness[0].classification[0]
            handedness = hand_info.label
            confidence = float(hand_info.score)

        landmarks = [
            Landmark(x=float(lm.x), y=float(lm.y), z=float(lm.z))
            for lm in landmarks_proto.landmark
        ]
        return HandObservation(
            landmarks=landmarks,
            handedness=handedness,
            detection_confidence=confidence,
            tracked_at_ms=t_ms,
        )

    @staticmethod
    def bgr_to_rgb(frame_bgr):
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._hands.close()

