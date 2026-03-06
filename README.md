# PUnity

PUnity is a Windows 11 desktop app that tracks one hand from a webcam and turns a small, reliable gesture set into mouse/keyboard actions.

## What this build includes

- Smooth cursor control from index fingertip (`OPEN_PALM` gated by default)
- Pinch-to-click with pinch-hold drag (`PINCHING`)
- Fist pause behavior (`FIST` stops movement and releases drag)
- Optional swipe gestures with cooldown-based hotkeys (`SWIPE_LEFT` / `SWIPE_RIGHT`)
- Safety controls:
  - Instant kill switch (default `F8`)
  - Idle timeout
  - Confidence gating
- JSON profile configuration for thresholds, smoothing, mappings, and safety
- OpenCV overlay HUD for state, gesture, confidence, and FPS

## Python version requirement

Use Python **3.11 or 3.12** and pin MediaPipe to **<=0.10.21** because newer releases removed `mp.solutions` used by this app.

If you currently created your venv with Python 3.14, recreate it with Python 3.12:

```powershell
# From C:\Programs\PUnity
Deactivate 2>$null
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -e .
```
## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Run

```powershell
punity --profile profiles/default.json
```

- Press `F8` to toggle active/pause instantly.
- Press `Q` or `Esc` in the preview window to exit.

## Gesture Set

- `OPEN_PALM`: dead-man switch for cursor movement (enabled by `require_open_palm`)
- `PINCHING`: mouse down + drag
- Pinch release: mouse up
- `FIST`: immediate pause behavior (movement blocked, drag released)
- `SWIPE_LEFT` / `SWIPE_RIGHT` (optional): hotkeys via mapping

## Config (`profiles/default.json`)

Top-level sections:

- `thresholds`: pinch hysteresis + detector confidence
- `cursor`: sensitivity, acceleration, smoothing filter, deadzone
- `safety`: kill switch key, idle timeout, palm requirement
- `swipe`: enable/disable and velocity/cooldown tuning
- `camera`: webcam index and requested resolution/FPS
- `overlay`: debug HUD options
- `mappings`: gesture labels -> actions (currently `HOTKEY`)

Example mapping:

```json
"SWIPE_LEFT": {
  "event": "HOTKEY",
  "keys": ["cmd", "ctrl", "left"],
  "cooldown_ms": 1200
}
```

## Project structure

```text
src/punity/
  capture/camera.py
  perception/hands.py
  tracking/filters.py
  tracking/hand_state.py
  gestures/features.py
  gestures/recognizer.py
  control/fsm.py
  actions/dispatcher.py
  config/profile.py
  ui/overlay.py
  app.py
```

## Notes

- This project injects inputs at user level through `pynput`.
- For stricter injection behavior, add a Windows `SendInput` backend later under `actions/win_sendinput.py` and switch dispatcher wiring.


