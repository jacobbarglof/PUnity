from punity.gestures.recognizer import GestureConfig, GestureRecognizer
from punity.models import GestureLabel, HandObservation, Landmark


def _make_obs(index_tip=(0.55, 0.45), thumb_tip=(0.45, 0.45)) -> HandObservation:
    points = [Landmark(0.5, 0.7, 0.0) for _ in range(21)]

    points[0] = Landmark(0.5, 0.8, 0.0)  # wrist
    points[9] = Landmark(0.5, 0.55, 0.0)  # middle mcp

    points[4] = Landmark(thumb_tip[0], thumb_tip[1], 0.0)
    points[8] = Landmark(index_tip[0], index_tip[1], 0.0)
    points[6] = Landmark(0.55, 0.6, 0.0)

    points[12] = Landmark(0.5, 0.42, 0.0)
    points[10] = Landmark(0.5, 0.6, 0.0)

    points[16] = Landmark(0.44, 0.43, 0.0)
    points[14] = Landmark(0.44, 0.61, 0.0)

    points[20] = Landmark(0.62, 0.45, 0.0)
    points[18] = Landmark(0.62, 0.63, 0.0)

    return HandObservation(
        landmarks=points,
        handedness="Right",
        detection_confidence=0.9,
        tracked_at_ms=0,
    )


def test_pinch_hysteresis() -> None:
    rec = GestureRecognizer(
        GestureConfig(
            pinch_on=0.52,
            pinch_off=0.67,
            min_confidence=0.6,
            swipe_enabled=False,
            swipe_velocity_threshold=1.7,
            swipe_cooldown_ms=800,
        )
    )

    # Close pinch engages.
    g1 = rec.recognize(_make_obs(index_tip=(0.51, 0.46), thumb_tip=(0.49, 0.46)), 100)
    assert g1.label == GestureLabel.PINCHING

    # Slight release stays pinching (hysteresis).
    g2 = rec.recognize(_make_obs(index_tip=(0.56, 0.46), thumb_tip=(0.49, 0.46)), 120)
    assert g2.label == GestureLabel.PINCHING

    # Wide release exits pinch.
    g3 = rec.recognize(_make_obs(index_tip=(0.64, 0.46), thumb_tip=(0.49, 0.46)), 140)
    assert g3.label != GestureLabel.PINCHING
