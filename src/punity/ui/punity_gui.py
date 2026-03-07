from __future__ import annotations

import ctypes
import json
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from pynput import keyboard
from tkinter import messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

from punity.actions.dispatcher import ActionDispatcher, CursorConfig
from punity.capture.camera import CameraCapture, CameraConfig
from punity.config.profile import (
    AppProfile,
    CameraProfile,
    CursorProfile,
    OverlayProfile,
    SafetyProfile,
    SwipeProfile,
    ThresholdConfig,
)
from punity.control.fsm import ControlConfig, ControlFSM
from punity.gestures.recognizer import GestureConfig, GestureRecognizer
from punity.models import GestureLabel
from punity.tracking.filters import EMAFilter2D, OneEuroFilter2D
from punity.tracking.hand_state import HandStateTracker
from punity.ui.overlay import draw_hand_skeleton


def _detect_project_root() -> Path:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for base in candidates:
        if (base / "pyproject.toml").exists() and (base / "profiles").exists():
            return base
    return Path.cwd()


PROJECT_ROOT = _detect_project_root()
PROFILES_DIR = PROJECT_ROOT / "profiles"
PREVIEW_MODES: tuple[str, ...] = ("Camera + Skeleton", "Camera", "Skeleton")
LAYOUT_MODES: tuple[str, ...] = ("Split View", "Focus Camera", "Focus Settings")


@dataclass(frozen=True)
class SettingField:
    section: str
    key: str
    label: str
    kind: str


SETTINGS_FIELDS: tuple[SettingField, ...] = (
    SettingField("thresholds", "pinch_on", "Pinch Engage", "float"),
    SettingField("thresholds", "pinch_off", "Pinch Release", "float"),
    SettingField("thresholds", "min_confidence", "Min Confidence", "float"),
    SettingField("thresholds", "min_detection_confidence", "Min Detection Confidence", "float"),
    SettingField("thresholds", "min_tracking_confidence", "Min Tracking Confidence", "float"),
    SettingField("cursor", "sensitivity", "Cursor Sensitivity", "float"),
    SettingField("cursor", "accel", "Cursor Accel", "float"),
    SettingField("cursor", "smoothing", "Cursor Smoothing", "float"),
    SettingField("cursor", "deadzone_px", "Deadzone Pixels", "float"),
    SettingField("cursor", "edge_padding_px", "Edge Padding (px)", "float"),
    SettingField("camera", "width", "Camera Width", "int"),
    SettingField("camera", "height", "Camera Height", "int"),
    SettingField("camera", "fps", "Camera FPS", "int"),
    SettingField("safety", "kill_switch_key", "Kill Switch Key", "str"),
    SettingField("safety", "idle_timeout_ms", "Idle Timeout (ms)", "int"),
    SettingField("safety", "require_open_palm", "Require Pointer", "bool"),
)


GESTURE_GUIDE: tuple[tuple[str, str, str], ...] = (
    ("POINTER", "Raise index finger and curl remaining fingers", "Enable pointing + move cursor"),
    ("PINCHING", "Touch thumb tip to index tip", "Single click (200ms cooldown)"),
    ("SCROLL", "Hold index+middle up for up scroll, point them down for down scroll", "Continuous scroll while held"),
    ("PINKY_DRAG", "Raise index + pinky fingers", "Mouse down and drag while held"),
    ("FIST", "Curl all fingers in", "Immediate pause and drag release"),
    ("FINGERS_CROSSED", "Cross index and middle fingers", "Toggle pause/resume controls"),
)


HUD_FIELDS: tuple[tuple[str, str], ...] = (
    ("status", "Status"),
    ("state", "FSM State"),
    ("gesture", "Gesture"),
    ("confidence", "Confidence"),
    ("pinch", "Pinch Strength"),
    ("fps", "Engine FPS"),
    ("hand", "Hand Present"),
    ("idle", "Idle"),
    ("kill", "Kill Switch"),
)


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _default_profile_dict() -> dict[str, Any]:
    profile = AppProfile(
        thresholds=ThresholdConfig(),
        cursor=CursorProfile(),
        safety=SafetyProfile(),
        swipe=SwipeProfile(),
        camera=CameraProfile(),
        overlay=OverlayProfile(),
        mappings={},
    )
    return asdict(profile)


def _profile_from_dict(raw: dict[str, Any]) -> AppProfile:
    merged = _deep_merge(_default_profile_dict(), raw)
    return AppProfile(
        thresholds=ThresholdConfig(**merged["thresholds"]),
        cursor=CursorProfile(**merged["cursor"]),
        safety=SafetyProfile(**merged["safety"]),
        swipe=SwipeProfile(**merged["swipe"]),
        camera=CameraProfile(**merged["camera"]),
        overlay=OverlayProfile(**merged["overlay"]),
        mappings=merged.get("mappings", {}),
    )


def _ensure_supported_python() -> None:
    if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
        ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            f"Python {ver} detected. PUnity GUI requires Python 3.12.x for MediaPipe Hands support."
        )


def _apply_dark_title_bar(window: tk.Tk) -> None:
    if sys.platform != "win32":
        return

    try:
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        enabled = ctypes.c_int(1)
        for attr in (20, 19):
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd,
                ctypes.c_int(attr),
                ctypes.byref(enabled),
                ctypes.sizeof(enabled),
            )
            if result == 0:
                break

        caption_color = ctypes.c_uint(0x000000)
        text_color = ctypes.c_uint(0xFFFFFF)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            ctypes.c_int(35),
            ctypes.byref(caption_color),
            ctypes.sizeof(caption_color),
        )
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            ctypes.c_int(36),
            ctypes.byref(text_color),
            ctypes.sizeof(text_color),
        )
    except Exception:
        pass


class PUnityRuntimeEngine:
    def __init__(
        self,
        initial_profile: dict[str, Any],
        log_queue: queue.Queue[str],
        frame_queue: queue.Queue[Any],
        hud_queue: queue.Queue[dict[str, Any]],
        preview_mode: str = PREVIEW_MODES[0],
    ) -> None:
        self._log_queue = log_queue
        self._frame_queue = frame_queue
        self._hud_queue = hud_queue

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._config_lock = threading.Lock()
        self._profile_dict = json.loads(json.dumps(initial_profile))
        self._profile_version = 1

        self._active = True
        self._active_lock = threading.Lock()
        self._kill_switch_token = "f8"
        self._listener: keyboard.Listener | None = None

        self._preview_mode_lock = threading.Lock()
        self._preview_mode = preview_mode if preview_mode in PREVIEW_MODES else PREVIEW_MODES[0]

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        self._thread = None

    def update_profile(self, profile_data: dict[str, Any]) -> None:
        with self._config_lock:
            self._profile_dict = json.loads(json.dumps(profile_data))
            self._profile_version += 1
            safety = self._profile_dict.get("safety", {})
            if isinstance(safety, dict):
                self._kill_switch_token = str(safety.get("kill_switch_key", "f8")).strip().lower()

    def set_preview_mode(self, mode: str) -> None:
        if mode not in PREVIEW_MODES:
            return
        with self._preview_mode_lock:
            self._preview_mode = mode

    def _get_preview_mode(self) -> str:
        with self._preview_mode_lock:
            return self._preview_mode

    def _read_profile_snapshot(self) -> tuple[dict[str, Any], int]:
        with self._config_lock:
            return json.loads(json.dumps(self._profile_dict)), self._profile_version

    def _toggle_active(self, source: str = "Kill switch") -> None:
        with self._active_lock:
            self._active = not self._active
            state = "ACTIVE" if self._active else "PAUSED"
            self._log_queue.put(f"{source} toggled -> {state}")

    def _is_active(self) -> bool:
        with self._active_lock:
            return self._active

    def _on_key_press(self, key) -> None:
        token = self._kill_switch_token
        if len(token) == 1:
            if getattr(key, "char", "") == token:
                self._toggle_active("Kill switch")
            return

        key_name = token if token.startswith("Key.") else f"Key.{token}"
        if str(key) == key_name:
            self._toggle_active("Kill switch")

    def _push_frame(self, frame_bgr) -> None:
        try:
            self._frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame_bgr)
            except queue.Full:
                pass

    def _push_hud(self, payload: dict[str, Any]) -> None:
        try:
            self._hud_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._hud_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._hud_queue.put_nowait(payload)
            except queue.Full:
                pass
    def _run(self) -> None:
        camera: CameraCapture | None = None
        detector = None
        recognizer: GestureRecognizer | None = None
        hand_state: HandStateTracker | None = None
        fsm: ControlFSM | None = None
        dispatcher: ActionDispatcher | None = None
        cursor_filter = None
        profile: AppProfile | None = None
        profile_version = 0
        fps = 0.0
        last_t = time.monotonic()
        target_loop_hz = 30.0
        crossed_latch = False
        cross_hold_start_ms: int | None = None
        cross_last_toggle_ms = -10000

        from punity.perception.hands import HandsDetector

        def rebuild_pipeline(raw_profile: dict[str, Any]) -> AppProfile:
            nonlocal camera, detector, recognizer, hand_state, fsm, dispatcher, cursor_filter, target_loop_hz

            parsed = _profile_from_dict(raw_profile)

            if camera is not None:
                camera.release()
            if detector is not None:
                detector.close()

            target_loop_hz = float(max(5, min(90, parsed.camera.fps)))

            camera = CameraCapture(
                CameraConfig(
                    device_index=parsed.camera.device_index,
                    width=parsed.camera.width,
                    height=parsed.camera.height,
                    fps=int(target_loop_hz),
                )
            )
            detector = HandsDetector(
                min_detection_confidence=parsed.thresholds.min_detection_confidence,
                min_tracking_confidence=parsed.thresholds.min_tracking_confidence,
                max_num_hands=1,
            )
            recognizer = GestureRecognizer(
                GestureConfig(
                    pinch_on=parsed.thresholds.pinch_on,
                    pinch_off=parsed.thresholds.pinch_off,
                    min_confidence=parsed.thresholds.min_confidence,
                )
            )
            hand_state = HandStateTracker(
                lost_timeout_ms=450,
                idle_timeout_ms=parsed.safety.idle_timeout_ms,
            )
            fsm = ControlFSM(
                config=ControlConfig(
                    min_confidence=parsed.thresholds.min_confidence,
                    require_open_palm=parsed.safety.require_open_palm,
                    hotkey_cooldown_ms=900,
                )
            )
            dispatcher = ActionDispatcher(
                CursorConfig(
                    sensitivity=parsed.cursor.sensitivity,
                    accel=parsed.cursor.accel,
                    deadzone_px=parsed.cursor.deadzone_px,
                    edge_padding_px=parsed.cursor.edge_padding_px,
                )
            )

            if parsed.cursor.filter == "ema":
                cursor_filter = EMAFilter2D(alpha=parsed.cursor.smoothing)
            else:
                cursor_filter = OneEuroFilter2D(
                    freq=max(5.0, target_loop_hz),
                    min_cutoff=max(0.5, parsed.cursor.smoothing * 4.0),
                    beta=0.02,
                    d_cutoff=1.2,
                )

            self._kill_switch_token = parsed.safety.kill_switch_key.lower().strip()
            self._log_queue.put("Pipeline updated from settings")
            return parsed

        try:
            self._listener = keyboard.Listener(on_press=self._on_key_press)
            self._listener.start()

            raw_profile, profile_version = self._read_profile_snapshot()
            profile = rebuild_pipeline(raw_profile)

            while not self._stop_event.is_set():

                raw_profile, next_ver = self._read_profile_snapshot()
                if next_ver != profile_version:
                    profile = rebuild_pipeline(raw_profile)
                    profile_version = next_ver

                assert camera is not None
                assert detector is not None
                assert recognizer is not None
                assert hand_state is not None
                assert fsm is not None
                assert dispatcher is not None
                assert cursor_filter is not None
                assert profile is not None

                frame_bgr, t_ms = camera.read_frame()
                if profile.overlay.mirror_preview:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                detect_bgr = frame_bgr
                frame_h, frame_w = frame_bgr.shape[:2]
                if frame_w > 640:
                    detect_h = max(1, int(frame_h * (640.0 / frame_w)))
                    detect_bgr = cv2.resize(frame_bgr, (640, detect_h), interpolation=cv2.INTER_LINEAR)

                frame_rgb = detector.bgr_to_rgb(detect_bgr)
                observation = detector.detect(frame_rgb, t_ms)

                gesture = recognizer.recognize(observation, t_ms)
                if gesture.label == GestureLabel.FINGERS_CROSSED:
                    if cross_hold_start_ms is None:
                        cross_hold_start_ms = t_ms

                    hold_elapsed = t_ms - cross_hold_start_ms
                    cooldown_elapsed = t_ms - cross_last_toggle_ms
                    if hold_elapsed >= 400 and cooldown_elapsed >= 1000 and not crossed_latch:
                        self._toggle_active("Fingers crossed")
                        crossed_latch = True
                        cross_last_toggle_ms = t_ms
                else:
                    cross_hold_start_ms = None
                    crossed_latch = False

                if gesture.cursor_point_norm is not None:
                    gesture.cursor_point_norm = cursor_filter.update(gesture.cursor_point_norm)
                else:
                    cursor_filter.reset()

                hand_status = hand_state.update(observation, t_ms, gesture.cursor_point_norm)

                fsm.set_active(self._is_active())
                events = fsm.step(gesture, hand_status, t_ms, profile.mappings)
                for event in events:
                    dispatcher.execute(event)

                now = time.monotonic()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    instant = 1.0 / dt
                    if fps <= 0.0:
                        fps = instant
                    else:
                        fps = fps * 0.88 + instant * 0.12

                preview_mode = self._get_preview_mode()
                if preview_mode == "Skeleton":
                    preview_frame = frame_bgr.copy()
                    preview_frame[:] = 0
                else:
                    preview_frame = frame_bgr.copy()

                if observation is not None and preview_mode in ("Camera + Skeleton", "Skeleton"):
                    draw_hand_skeleton(preview_frame, observation)

                self._push_frame(preview_frame)
                self._push_hud(
                    {
                        "status": "ACTIVE" if self._is_active() else "PAUSED",
                        "state": fsm.state.value,
                        "gesture": gesture.label.value,
                        "confidence": float(gesture.confidence),
                        "pinch": float(gesture.pinch_strength),
                        "fps": float(fps),
                        "hand": bool(hand_status.present),
                        "idle": bool(hand_status.idle),
                        "kill": self._kill_switch_token.upper(),
                    }
                )



        except Exception as exc:
            self._log_queue.put(f"Engine error: {exc}")
        finally:
            if self._listener is not None:
                self._listener.stop()
                self._listener = None
            if camera is not None:
                camera.release()
            if detector is not None:
                detector.close()

            self._push_hud(
                {
                    "status": "IDLE",
                    "state": "IDLE",
                    "gesture": "NONE",
                    "confidence": 0.0,
                    "pinch": 0.0,
                    "fps": 0.0,
                    "hand": False,
                    "idle": False,
                    "kill": self._kill_switch_token.upper(),
                }
            )
            self._log_queue.put("Engine stopped")


class PUnityControlApp(tk.Tk):
    BG = "#020912"
    PANEL = "#07111d"
    PANEL_ALT = "#0a1624"
    ACCENT = "#00E5FF"
    ACCENT_SOFT = "#59EEFF"
    TEXT = "#C8F8FF"
    WARN = "#FF9D3D"
    GOOD = "#38FFAE"

    def __init__(self) -> None:
        super().__init__()
        self.title("PUnity Control Console")
        self.geometry("1920x1080")
        self.minsize(1440, 810)
        self.configure(bg=self.BG)

        self._profile_data: dict[str, Any] = {}
        self._field_vars: dict[tuple[str, str], tk.Variable] = {}
        self._gesture_action_labels: dict[str, tk.Label] = {}
        self._hud_value_labels: dict[str, tk.Label] = {}
        self._preview_photo: ImageTk.PhotoImage | None = None

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._frame_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._hud_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._engine: PUnityRuntimeEngine | None = None

        self._body: tk.Frame | None = None
        self._left_col: tk.Frame | None = None
        self._right_col: tk.Frame | None = None
        self._mousewheel_canvas: tk.Canvas | None = None

        self.profile_name_var = tk.StringVar(value="default.json")
        self.runtime_status_var = tk.StringVar(value="IDLE")
        self.preview_mode_var = tk.StringVar(value=PREVIEW_MODES[0])
        self.layout_mode_var = tk.StringVar(value=LAYOUT_MODES[0])

        self.hud_vars: dict[str, tk.StringVar] = {
            key: tk.StringVar(value="--") for key, _label in HUD_FIELDS
        }

        self._configure_ttk()
        self._build_layout()
        self._set_runtime_status("IDLE", good=False)
        self._reset_hud_panel()

        self._refresh_profile_list()
        self._load_profile_into_editor(self.profile_name_var.get())

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(10, self._poll_runtime)
        self.after(50, lambda: _apply_dark_title_bar(self))
    def _configure_ttk(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "PUnity.TCombobox",
            fieldbackground=self.PANEL_ALT,
            background=self.PANEL_ALT,
            foreground=self.TEXT,
            arrowcolor=self.ACCENT,
            bordercolor=self.ACCENT,
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT,
        )
        style.map(
            "PUnity.TCombobox",
            fieldbackground=[("readonly", self.PANEL_ALT)],
            foreground=[("readonly", self.TEXT)],
            background=[("readonly", self.PANEL_ALT)],
            selectbackground=[("readonly", self.PANEL_ALT)],
            selectforeground=[("readonly", self.TEXT)],
        )
        # Avoid default white dropdown listbox overlay.
        self.option_add("*TCombobox*Listbox.background", self.PANEL_ALT)
        self.option_add("*TCombobox*Listbox.foreground", self.TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", "#123049")
        self.option_add("*TCombobox*Listbox.selectForeground", self.TEXT)

    def _build_layout(self) -> None:
        header = tk.Frame(self, bg=self.BG)
        header.pack(fill="x", padx=16, pady=(14, 10))

        title = tk.Label(
            header,
            text="PUNITY // CONTROL CONSOLE",
            bg=self.BG,
            fg=self.ACCENT,
            font=("Consolas", 18, "bold"),
        )
        title.pack(side="left")

        status_frame = tk.Frame(header, bg=self.BG)
        status_frame.pack(side="right")

        self.status_chip = tk.Label(
            status_frame,
            textvariable=self.runtime_status_var,
            bg="#11351f",
            fg=self.GOOD,
            font=("Consolas", 11, "bold"),
            padx=12,
            pady=4,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.GOOD,
        )
        self.status_chip.pack(side="left", padx=(0, 10))

        tk.Label(status_frame, text="Preview", bg=self.BG, fg=self.TEXT, font=("Consolas", 9)).pack(
            side="left", padx=(0, 6)
        )
        preview_combo = ttk.Combobox(
            status_frame,
            textvariable=self.preview_mode_var,
            values=PREVIEW_MODES,
            state="readonly",
            style="PUnity.TCombobox",
            width=20,
        )
        preview_combo.pack(side="left", padx=(0, 10))
        preview_combo.bind("<<ComboboxSelected>>", self._on_preview_mode_changed)

        tk.Label(status_frame, text="Layout", bg=self.BG, fg=self.TEXT, font=("Consolas", 9)).pack(
            side="left", padx=(0, 6)
        )
        layout_combo = ttk.Combobox(
            status_frame,
            textvariable=self.layout_mode_var,
            values=LAYOUT_MODES,
            state="readonly",
            style="PUnity.TCombobox",
            width=18,
        )
        layout_combo.pack(side="left", padx=(0, 10))
        layout_combo.bind("<<ComboboxSelected>>", self._on_layout_mode_changed)

        self.start_btn = self._make_button(status_frame, "Start Engine", self._on_start_engine)
        self.start_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = self._make_button(status_frame, "Stop Engine", self._on_stop_engine, warn=True)
        self.stop_btn.pack(side="left")

        gestures_panel = self._make_panel(self, "GESTURES")
        gestures_panel.pack(fill="x", padx=16, pady=(0, 10))
        self._build_gesture_toolbar(gestures_panel.content)

        self._body = tk.Frame(self, bg=self.BG)
        self._body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._body.grid_columnconfigure(0, weight=4)
        self._body.grid_columnconfigure(1, weight=3)
        self._body.grid_rowconfigure(0, weight=1)

        self._left_col = tk.Frame(self._body, bg=self.BG)
        self._left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._left_col.grid_rowconfigure(0, weight=10)
        self._left_col.grid_rowconfigure(1, weight=1)
        self._left_col.grid_rowconfigure(2, weight=0)
        self._left_col.grid_columnconfigure(0, weight=1)

        preview_panel = self._make_panel(self._left_col, "LIVE CAMERA")
        preview_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        preview_panel.content.grid_rowconfigure(0, weight=1)
        preview_panel.content.grid_columnconfigure(0, weight=1)

        self.preview_label = tk.Label(
            preview_panel.content,
            text="Engine stopped",
            bg="#04070d",
            fg=self.ACCENT,
            font=("Consolas", 12, "bold"),
            anchor="center",
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.ACCENT,
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        hud_panel = self._make_panel(self._left_col, "HUD")
        hud_panel.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        self._build_hud_panel(hud_panel.content)

        logs_panel = self._make_panel(self._left_col, "RUNTIME LOG")
        logs_panel.grid(row=2, column=0, sticky="nsew")
        self.logs_text = ScrolledText(
            logs_panel.content,
            bg="#04070d",
            fg="#9CE8B6",
            insertbackground="#9CE8B6",
            font=("Consolas", 10),
            bd=0,
            relief="flat",
            wrap="word",
            height=6,
        )
        self.logs_text.pack(fill="both", expand=True)
        self.logs_text.configure(state="disabled")

        self._right_col = tk.Frame(self._body, bg=self.BG)
        self._right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._right_col.grid_rowconfigure(1, weight=1)
        self._right_col.grid_columnconfigure(0, weight=1)

        profile_panel = self._make_panel(self._right_col, "PROFILE MANAGER")
        profile_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_profile_manager(profile_panel.content)

        settings_panel = self._make_panel(self._right_col, "SETTINGS")
        settings_panel.grid(row=1, column=0, sticky="nsew")
        self._build_settings_editor(settings_panel.content)

        self._apply_layout_mode()

    def _make_button(self, parent: tk.Misc, text: str, command, warn: bool = False) -> tk.Button:
        fg = self.WARN if warn else self.ACCENT
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=self.PANEL_ALT,
            fg=fg,
            activebackground="#123049",
            activeforeground="#E6FDFF",
            font=("Consolas", 10, "bold"),
            relief="flat",
            padx=10,
            pady=6,
            highlightthickness=1,
            highlightbackground=fg,
            bd=0,
        )

    def _make_panel(self, parent: tk.Misc, title: str) -> tk.Frame:
        panel = tk.Frame(
            parent,
            bg=self.PANEL,
            highlightthickness=1,
            highlightbackground=self.ACCENT,
            bd=0,
        )
        heading = tk.Label(
            panel,
            text=title,
            bg=self.PANEL,
            fg=self.ACCENT,
            font=("Consolas", 11, "bold"),
            anchor="w",
        )
        heading.pack(fill="x", padx=10, pady=(8, 6))

        inner = tk.Frame(panel, bg=self.PANEL)
        inner.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        panel.content = inner
        return panel

    def _build_gesture_toolbar(self, parent: tk.Frame) -> None:
        for col, (name, how_to, action) in enumerate(GESTURE_GUIDE):
            card = tk.Frame(
                parent,
                bg=self.PANEL_ALT,
                highlightthickness=1,
                highlightbackground=self.ACCENT,
                bd=0,
                padx=10,
                pady=8,
            )
            card.grid(row=0, column=col, sticky="nsew", padx=(0, 8 if col < len(GESTURE_GUIDE) - 1 else 0))
            parent.grid_columnconfigure(col, weight=1)

            tk.Label(
                card,
                text=name,
                bg=self.PANEL_ALT,
                fg=self.ACCENT,
                font=("Consolas", 11, "bold"),
                anchor="w",
            ).pack(fill="x")

            tk.Label(
                card,
                text=f"How: {how_to}",
                bg=self.PANEL_ALT,
                fg=self.TEXT,
                font=("Consolas", 9),
                justify="left",
                anchor="w",
                wraplength=260,
            ).pack(fill="x", pady=(4, 4))

            action_label = tk.Label(
                card,
                text=f"Action: {action}",
                bg=self.PANEL_ALT,
                fg="#8FFFA6",
                font=("Consolas", 9, "bold"),
                justify="left",
                anchor="w",
                wraplength=260,
            )
            action_label.pack(fill="x")
            self._gesture_action_labels[name] = action_label

    def _build_hud_panel(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=self.PANEL)
        row.pack(fill="x", expand=True)

        for col, (key, label) in enumerate(HUD_FIELDS):
            card = tk.Frame(
                row,
                bg=self.PANEL_ALT,
                highlightthickness=1,
                highlightbackground=self.ACCENT,
                bd=0,
                padx=6,
                pady=4,
            )
            card.grid(
                row=0,
                column=col,
                sticky="nsew",
                padx=(0, 6 if col < len(HUD_FIELDS) - 1 else 0),
            )
            row.grid_columnconfigure(col, weight=1, uniform="hud")

            tk.Label(
                card,
                text=label,
                bg=self.PANEL_ALT,
                fg=self.TEXT,
                font=("Consolas", 8),
                anchor="w",
            ).pack(anchor="w")

            value_width = 18 if key == "gesture" else 10
            value_label = tk.Label(
                card,
                textvariable=self.hud_vars[key],
                bg=self.PANEL_ALT,
                fg=self.ACCENT_SOFT,
                font=("Consolas", 9, "bold"),
                anchor="w",
                width=value_width,
            )
            value_label.pack(anchor="w", pady=(2, 0))
            self._hud_value_labels[key] = value_label

    def _build_profile_manager(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=self.PANEL)
        row.pack(fill="x")

        tk.Label(
            row,
            text="Profile",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Consolas", 10),
        ).pack(side="left", padx=(0, 8))

        self.profile_combo = ttk.Combobox(
            row,
            textvariable=self.profile_name_var,
            state="readonly",
            style="PUnity.TCombobox",
            width=28,
        )
        self.profile_combo.pack(side="left", fill="x", expand=True)
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_selected)

        actions = tk.Frame(parent, bg=self.PANEL)
        actions.pack(fill="x", pady=(10, 0))

        self._make_button(actions, "Reload", self._on_reload_profile).pack(side="left", padx=(0, 8))
        self._make_button(actions, "Apply Live", self._on_apply_live).pack(side="left", padx=(0, 8))
        self._make_button(actions, "Save", self._on_save_profile).pack(side="left", padx=(0, 8))
        self._make_button(actions, "Save As", self._on_save_as_profile).pack(side="left")

    def _build_settings_editor(self, parent: tk.Frame) -> None:
        canvas = tk.Canvas(
            parent,
            bg=self.PANEL,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        scrollbar = tk.Scrollbar(
            parent,
            orient="vertical",
            command=canvas.yview,
            bg=self.PANEL_ALT,
            troughcolor=self.PANEL,
            activebackground="#123049",
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        grid = tk.Frame(canvas, bg=self.PANEL)
        grid_window = canvas.create_window((0, 0), window=grid, anchor="nw")

        def _sync_scrollregion(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_canvas_width(event) -> None:
            canvas.itemconfigure(grid_window, width=event.width)

        grid.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _sync_canvas_width)
        canvas.bind("<Enter>", lambda _e: self._bind_mousewheel(canvas))
        canvas.bind("<Leave>", lambda _e: self._unbind_mousewheel())

        for row_idx, field in enumerate(SETTINGS_FIELDS):
            tk.Label(
                grid,
                text=field.label,
                bg=self.PANEL,
                fg=self.TEXT,
                font=("Consolas", 10),
                anchor="w",
            ).grid(row=row_idx, column=0, sticky="w", padx=(0, 10), pady=3)

            if field.kind == "bool":
                var = tk.BooleanVar(value=False)
                widget = tk.Checkbutton(
                    grid,
                    variable=var,
                    onvalue=True,
                    offvalue=False,
                    command=self._auto_apply_live,
                    bg=self.PANEL,
                    activebackground=self.PANEL,
                    fg=self.ACCENT,
                    selectcolor=self.PANEL_ALT,
                    highlightthickness=1,
                    highlightbackground=self.ACCENT,
                    width=4,
                )
            else:
                var = tk.StringVar(value="")
                widget = tk.Entry(
                    grid,
                    textvariable=var,
                    bg=self.PANEL_ALT,
                    fg=self.ACCENT_SOFT,
                    insertbackground=self.ACCENT,
                    relief="flat",
                    highlightthickness=1,
                    highlightbackground=self.ACCENT,
                    font=("Consolas", 10),
                )
                widget.bind("<FocusOut>", lambda _e: self._auto_apply_live())
                widget.bind("<Return>", lambda _e: self._auto_apply_live())

            widget.grid(row=row_idx, column=1, sticky="ew", pady=3)
            self._field_vars[(field.section, field.key)] = var

        grid.grid_columnconfigure(1, weight=1)
        _sync_scrollregion()

    def _bind_mousewheel(self, canvas: tk.Canvas) -> None:
        self._mousewheel_canvas = canvas
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self) -> None:
        self._mousewheel_canvas = None
        self.unbind_all("<MouseWheel>")
        self.unbind_all("<Button-4>")
        self.unbind_all("<Button-5>")

    def _on_mousewheel(self, event) -> None:
        if self._mousewheel_canvas is None:
            return
        if hasattr(event, "delta") and event.delta:
            self._mousewheel_canvas.yview_scroll(int(-event.delta / 120), "units")
            return
        if getattr(event, "num", None) == 4:
            self._mousewheel_canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self._mousewheel_canvas.yview_scroll(1, "units")

    def _refresh_profile_list(self) -> None:
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        names = sorted(path.name for path in PROFILES_DIR.glob("*.json"))
        if not names:
            fallback = PROFILES_DIR / "default.json"
            fallback.write_text(json.dumps(_default_profile_dict(), indent=2) + "\n", encoding="utf-8")
            names = [fallback.name]

        self.profile_combo["values"] = names
        current = self.profile_name_var.get()
        if current not in names:
            self.profile_name_var.set(names[0])

    def _on_profile_selected(self, _event=None) -> None:
        self._load_profile_into_editor(self.profile_name_var.get())

    def _load_profile_into_editor(self, profile_name: str) -> None:
        path = PROFILES_DIR / profile_name
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            parsed = _profile_from_dict(raw)
            data = asdict(parsed)
        except Exception as exc:
            messagebox.showerror("Profile Load Error", f"Failed to load profile:\n{exc}")
            return

        self._profile_data = data
        self._apply_profile_to_fields()
        self._refresh_gesture_actions()
        self._sync_hud_static_from_profile()
        self._append_log(f"Loaded profile {profile_name}")

        if self._engine is not None and self._engine.is_running:
            self._engine.update_profile(self._profile_data)
            self._append_log("Live profile applied")

    def _on_reload_profile(self) -> None:
        self._load_profile_into_editor(self.profile_name_var.get())

    def _apply_profile_to_fields(self) -> None:
        for field in SETTINGS_FIELDS:
            section = self._profile_data.get(field.section, {})
            value = section.get(field.key) if isinstance(section, dict) else None
            var = self._field_vars[(field.section, field.key)]

            if field.kind == "bool":
                var.set(bool(value))
            elif value is None:
                var.set("")
            else:
                var.set(str(value))

    def _collect_profile_from_fields(self) -> dict[str, Any]:
        data = json.loads(json.dumps(self._profile_data)) if self._profile_data else _default_profile_dict()

        for field in SETTINGS_FIELDS:
            section = data.setdefault(field.section, {})
            if not isinstance(section, dict):
                section = {}
                data[field.section] = section

            var = self._field_vars[(field.section, field.key)]
            section[field.key] = self._coerce_field_value(field, var.get())

        return asdict(_profile_from_dict(data))

    def _coerce_field_value(self, field: SettingField, raw: object) -> Any:
        if field.kind == "bool":
            return bool(raw)

        text = str(raw).strip()
        if field.kind == "int":
            return int(text)
        if field.kind == "float":
            return float(text)
        return text

    def _on_save_profile(self) -> None:
        if self._save_profile(show_popup=True):
            self._append_log("Profile saved")

    def _save_profile(self, show_popup: bool) -> bool:
        try:
            data = self._collect_profile_from_fields()
        except Exception as exc:
            messagebox.showerror("Save Error", f"Could not save profile:\n{exc}")
            return False

        profile_name = self.profile_name_var.get().strip()
        if not profile_name:
            messagebox.showerror("Save Error", "Profile name is empty")
            return False

        if not profile_name.endswith(".json"):
            profile_name = f"{profile_name}.json"
            self.profile_name_var.set(profile_name)

        path = PROFILES_DIR / profile_name
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self._profile_data = data
        self._refresh_profile_list()
        self._refresh_gesture_actions()
        self._sync_hud_static_from_profile()

        if self._engine is not None and self._engine.is_running:
            self._engine.update_profile(self._profile_data)
            self._append_log("Live settings applied")

        if show_popup:
            messagebox.showinfo("Saved", f"Profile saved to\n{path}")
        return True

    def _on_save_as_profile(self) -> None:
        name = simpledialog.askstring("Save As", "New profile name (without .json):", parent=self)
        if not name:
            return

        clean = name.strip().replace("\\", "_").replace("/", "_")
        if not clean:
            return

        self.profile_name_var.set(f"{clean}.json")
        if self._save_profile(show_popup=True):
            self._append_log(f"Created profile {clean}.json")

    def _on_apply_live(self) -> None:
        try:
            data = self._collect_profile_from_fields()
        except Exception as exc:
            messagebox.showerror("Apply Error", f"Could not apply settings:\n{exc}")
            return

        self._profile_data = data
        self._refresh_gesture_actions()
        self._sync_hud_static_from_profile()

        if self._engine is not None and self._engine.is_running:
            self._engine.update_profile(self._profile_data)
            self._append_log("Live settings applied")

    def _auto_apply_live(self) -> None:
        if self._engine is None or not self._engine.is_running:
            return

        try:
            data = self._collect_profile_from_fields()
        except Exception:
            return

        self._profile_data = data
        self._refresh_gesture_actions()
        self._sync_hud_static_from_profile()
        self._engine.update_profile(self._profile_data)
    def _refresh_gesture_actions(self) -> None:
        for gesture_name, _how, default_action in GESTURE_GUIDE:
            label = self._gesture_action_labels.get(gesture_name)
            if label is None:
                continue
            label.configure(text=f"Action: {default_action}")
    def _format_mapping_action(self, gesture: str, mappings: dict[str, Any] | None = None) -> str:
        if mappings is None:
            raw = self._profile_data.get("mappings", {})
            mappings = raw if isinstance(raw, dict) else {}

        action = mappings.get(gesture) if isinstance(mappings, dict) else None
        if not isinstance(action, dict):
            return "No mapping"

        event_name = str(action.get("event", "UNKNOWN"))
        keys = action.get("keys")
        if isinstance(keys, list) and keys:
            combo = "+".join(str(k).upper() for k in keys)
            return f"{event_name}: {combo}"
        return event_name
    def _on_start_engine(self) -> None:
        if self._engine is not None and self._engine.is_running:
            return

        try:
            self._profile_data = self._collect_profile_from_fields()
        except Exception as exc:
            messagebox.showerror("Start Error", f"Cannot start engine:\n{exc}")
            return

        self._engine = PUnityRuntimeEngine(
            self._profile_data,
            self._log_queue,
            self._frame_queue,
            self._hud_queue,
            preview_mode=self.preview_mode_var.get(),
        )
        self._engine.start()
        self._set_runtime_status("RUNNING", good=True)
        self._append_log("Engine started")

    def _on_stop_engine(self) -> None:
        if self._engine is None:
            return

        self._engine.stop()
        self._engine = None
        self._set_runtime_status("IDLE", good=False)
        self._append_log("Engine stop requested")
        self.preview_label.configure(image="", text="Engine stopped")
        self._reset_hud_panel()

    def _on_preview_mode_changed(self, _event=None) -> None:
        if self._engine is not None and self._engine.is_running:
            self._engine.set_preview_mode(self.preview_mode_var.get())

    def _on_layout_mode_changed(self, _event=None) -> None:
        self._apply_layout_mode()

    def _apply_layout_mode(self) -> None:
        if self._body is None or self._left_col is None or self._right_col is None:
            return

        mode = self.layout_mode_var.get()
        if mode == "Focus Camera":
            self._right_col.grid_remove()
            self._left_col.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=(0, 0))
            self._body.grid_columnconfigure(0, weight=1)
            self._body.grid_columnconfigure(1, weight=0)
            return

        if mode == "Focus Settings":
            self._left_col.grid_remove()
            self._right_col.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=(0, 0))
            self._body.grid_columnconfigure(0, weight=1)
            self._body.grid_columnconfigure(1, weight=0)
            return

        self._left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._body.grid_columnconfigure(0, weight=4)
        self._body.grid_columnconfigure(1, weight=3)

    def _poll_runtime(self) -> None:
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)

        latest_frame = None
        while True:
            try:
                latest_frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break

        latest_hud = None
        while True:
            try:
                latest_hud = self._hud_queue.get_nowait()
            except queue.Empty:
                break

        if latest_frame is not None:
            self._render_preview(latest_frame)

        if latest_hud is not None:
            self._update_hud_panel(latest_hud)

        if self._engine is not None and not self._engine.is_running:
            self._engine = None
            self._set_runtime_status("IDLE", good=False)

        self.after(10, self._poll_runtime)

    def _render_preview(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        target_w = max(320, self.preview_label.winfo_width())
        target_h = max(240, self.preview_label.winfo_height())
        if target_w > 0 and target_h > 0:
            scale = min(target_w / img.width, target_h / img.height)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.Resampling.BILINEAR)

        photo = ImageTk.PhotoImage(img)
        self._preview_photo = photo
        self.preview_label.configure(image=photo, text="")

    def _append_log(self, line: str) -> None:
        self.logs_text.configure(state="normal")
        self.logs_text.insert(tk.END, line + "\n")

        line_count = int(float(self.logs_text.index("end-1c").split(".")[0]))
        if line_count > 220:
            self.logs_text.delete("1.0", f"{line_count - 220}.0")

        self.logs_text.see(tk.END)
        self.logs_text.configure(state="disabled")

    def _set_runtime_status(self, text: str, good: bool) -> None:
        self.runtime_status_var.set(text)
        if good:
            self.status_chip.configure(bg="#11351f", fg=self.GOOD, highlightbackground=self.GOOD)
        else:
            self.status_chip.configure(bg="#3a2512", fg=self.WARN, highlightbackground=self.WARN)

    def _reset_hud_panel(self) -> None:
        self.hud_vars["status"].set("IDLE")
        self.hud_vars["state"].set("IDLE")
        self.hud_vars["gesture"].set("NONE")
        self.hud_vars["confidence"].set("0.00")
        self.hud_vars["pinch"].set("0.00")
        self.hud_vars["fps"].set("0.0")
        self.hud_vars["hand"].set("NO")
        self.hud_vars["idle"].set("NO")
        self.hud_vars["kill"].set("F8")

        status_label = self._hud_value_labels.get("status")
        if status_label is not None:
            status_label.configure(fg=self.ACCENT_SOFT)

    def _sync_hud_static_from_profile(self) -> None:
        safety = self._profile_data.get("safety", {})
        if isinstance(safety, dict):
            token = str(safety.get("kill_switch_key", "f8")).strip().upper()
            self.hud_vars["kill"].set(token)

    def _update_hud_panel(self, hud: dict[str, Any]) -> None:
        status_text = str(hud.get("status", "--"))
        self.hud_vars["status"].set(status_text)

        status_label = self._hud_value_labels.get("status")
        if status_label is not None:
            status_fg = self.WARN if status_text.upper() == "PAUSED" else self.ACCENT_SOFT
            status_label.configure(fg=status_fg)
        self.hud_vars["state"].set(str(hud.get("state", "--")))
        self.hud_vars["gesture"].set(str(hud.get("gesture", "--")))

        try:
            self.hud_vars["confidence"].set(f"{float(hud.get('confidence', 0.0)):.2f}")
        except Exception:
            self.hud_vars["confidence"].set("0.00")

        try:
            self.hud_vars["pinch"].set(f"{float(hud.get('pinch', 0.0)):.2f}")
        except Exception:
            self.hud_vars["pinch"].set("0.00")

        try:
            self.hud_vars["fps"].set(f"{float(hud.get('fps', 0.0)):.1f}")
        except Exception:
            self.hud_vars["fps"].set("0.0")

        self.hud_vars["hand"].set("YES" if bool(hud.get("hand", False)) else "NO")
        self.hud_vars["idle"].set("YES" if bool(hud.get("idle", False)) else "NO")

        kill = str(hud.get("kill", self.hud_vars["kill"].get()))
        self.hud_vars["kill"].set(kill)

    def _on_close(self) -> None:
        if self._engine is not None:
            self._engine.stop()
            self._engine = None
        self.destroy()


def main() -> None:
    _ensure_supported_python()
    app = PUnityControlApp()
    app.mainloop()


if __name__ == "__main__":
    main()
