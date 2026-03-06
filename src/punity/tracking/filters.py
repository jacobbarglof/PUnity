from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EMAFilter2D:
    alpha: float
    _value: tuple[float, float] | None = None

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        if self._value is None:
            self._value = point
            return point

        px, py = point
        vx, vy = self._value
        out = (
            self.alpha * px + (1.0 - self.alpha) * vx,
            self.alpha * py + (1.0 - self.alpha) * vy,
        )
        self._value = out
        return out

    def reset(self) -> None:
        self._value = None


class _LowPass:
    def __init__(self) -> None:
        self._y: float | None = None

    def apply(self, x: float, alpha: float) -> float:
        if self._y is None:
            self._y = x
            return x
        self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y


class OneEuroFilter2D:
    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = max(freq, 1e-3)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._dx = _LowPass()
        self._dy = _LowPass()
        self._x = _LowPass()
        self._y = _LowPass()
        self._prev: tuple[float, float] | None = None

    @staticmethod
    def _alpha(freq: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        if self._prev is None:
            self._prev = point
            return point

        px, py = self._prev
        dx = (x - px) * self.freq
        dy = (y - py) * self.freq

        ad = self._alpha(self.freq, self.d_cutoff)
        edx = self._dx.apply(dx, ad)
        edy = self._dy.apply(dy, ad)

        cutoff_x = self.min_cutoff + self.beta * abs(edx)
        cutoff_y = self.min_cutoff + self.beta * abs(edy)

        ax = self._alpha(self.freq, cutoff_x)
        ay = self._alpha(self.freq, cutoff_y)

        fx = self._x.apply(x, ax)
        fy = self._y.apply(y, ay)

        self._prev = (fx, fy)
        return fx, fy

    def reset(self) -> None:
        self._prev = None
        self._dx = _LowPass()
        self._dy = _LowPass()
        self._x = _LowPass()
        self._y = _LowPass()
