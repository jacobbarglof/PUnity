from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ThresholdConfig:
    pinch_on: float = 0.44
    pinch_off: float = 0.60
    min_confidence: float = 0.65
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6


@dataclass(slots=True)
class CursorProfile:
    sensitivity: float = 0.55
    accel: float = 0.35
    smoothing: float = 0.35
    filter: str = "one_euro"
    deadzone_px: float = 3.0
    edge_padding_px: float = 80.0


@dataclass(slots=True)
class SafetyProfile:
    kill_switch_key: str = "f8"
    idle_timeout_ms: int = 4000
    require_open_palm: bool = True


@dataclass(slots=True)
class SwipeProfile:
    enabled: bool = False
    velocity_threshold: float = 1.15
    cooldown_ms: int = 850


@dataclass(slots=True)
class CameraProfile:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(slots=True)
class OverlayProfile:
    enabled: bool = True
    mirror_preview: bool = True


@dataclass(slots=True)
class AppProfile:
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    cursor: CursorProfile = field(default_factory=CursorProfile)
    safety: SafetyProfile = field(default_factory=SafetyProfile)
    swipe: SwipeProfile = field(default_factory=SwipeProfile)
    camera: CameraProfile = field(default_factory=CameraProfile)
    overlay: OverlayProfile = field(default_factory=OverlayProfile)
    mappings: dict[str, dict[str, object]] = field(default_factory=dict)


_DEFAULT = AppProfile(mappings={})


def _merge(dst: dict, src: dict) -> dict:
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_profile(path: str | Path) -> AppProfile:
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    raw = json.loads(text)

    default_json = asdict(_DEFAULT)
    merged = _merge(default_json, raw)

    return AppProfile(
        thresholds=ThresholdConfig(**merged["thresholds"]),
        cursor=CursorProfile(**merged["cursor"]),
        safety=SafetyProfile(**merged["safety"]),
        swipe=SwipeProfile(**merged["swipe"]),
        camera=CameraProfile(**merged["camera"]),
        overlay=OverlayProfile(**merged["overlay"]),
        mappings=merged.get("mappings", {}),
    )
