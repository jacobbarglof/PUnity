# Unity

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
- PUnity control GUI (`punity-gui`) with embedded engine runtime:
  - webcam preview + HUD inside the app window
  - gesture toolbar (what/how/action)
  - dedicated HUD panel (outside camera feed)
  - profile manager + live settings editor
  - view dropdowns (preview mode and layout mode)
  - start/stop engine controls + runtime log

## Python version requirement

Use Python **3.12.x** and pin MediaPipe to **<=0.10.21** because newer releases removed `mp.solutions` used by this app.

## Install

Recommended (always builds with Python 3.12 and recreates `.venv`):

```powershell
# From C:\Programs\PUnity
.\scripts\bootstrap.ps1
```

Manual equivalent:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -e .
```

## Run

Unified PUnity GUI (recommended):

```powershell
punity-gui
```

Optional engine-only mode (OpenCV window):

```powershell
punity --profile profiles/default.json
```

- Press `F8` to toggle active/pause instantly.

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
  ui/punity_gui.py
  app.py
```

## Notes

- This project injects inputs at user level through `pynput`.
- For stricter injection behavior, add a Windows `SendInput` backend later under `actions/win_sendinput.py` and switch dispatcher wiring.
