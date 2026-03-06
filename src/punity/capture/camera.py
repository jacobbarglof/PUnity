from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2


@dataclass(slots=True)
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


class CameraCapture:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._open()

    def _open(self) -> None:
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self._config.device_index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        # Reduce capture queue depth to limit stale-frame latency on webcams.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Prefer MJPG where available; many UVC webcams deliver it with lower decode overhead.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    def read_frame(self) -> tuple[Any, int]:
        if self._cap is None or not self._cap.isOpened():
            self._open()

        assert self._cap is not None

        ok, frame = self._cap.read()

        if not ok:
            self._open()
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError("Unable to read webcam frame")

        t_ms = int(time.monotonic() * 1000)
        return frame, t_ms

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
