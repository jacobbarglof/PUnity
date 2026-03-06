from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText


def _detect_project_root() -> Path:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for base in candidates:
        if (base / "pyproject.toml").exists() and (base / "profiles").exists():
            return base
    return Path.cwd()


PROJECT_ROOT = _detect_project_root()
PROFILES_DIR = PROJECT_ROOT / "profiles"


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
    SettingField("cursor", "sensitivity", "Cursor Sensitivity", "float"),
    SettingField("cursor", "accel", "Cursor Accel", "float"),
    SettingField("cursor", "smoothing", "Cursor Smoothing", "float"),
    SettingField("cursor", "deadzone_px", "Deadzone Pixels", "float"),
    SettingField("camera", "width", "Camera Width", "int"),
    SettingField("camera", "height", "Camera Height", "int"),
    SettingField("camera", "fps", "Camera FPS", "int"),
    SettingField("safety", "kill_switch_key", "Kill Switch Key", "str"),
    SettingField("safety", "idle_timeout_ms", "Idle Timeout (ms)", "int"),
    SettingField("safety", "require_open_palm", "Require Open Palm", "bool"),
    SettingField("swipe", "enabled", "Enable Swipe", "bool"),
    SettingField("swipe", "velocity_threshold", "Swipe Velocity", "float"),
    SettingField("swipe", "cooldown_ms", "Swipe Cooldown (ms)", "int"),
)


GESTURE_GUIDE: tuple[tuple[str, str, str], ...] = (
    ("OPEN_PALM", "Spread fingers and face palm to camera", "Enable pointing + move cursor"),
    ("PINCHING", "Touch thumb tip to index tip", "Mouse down and drag while held"),
    ("FIST", "Curl all fingers in", "Immediate pause and drag release"),
    ("SWIPE_LEFT", "Quick horizontal sweep to left", "Mapped command action"),
    ("SWIPE_RIGHT", "Quick horizontal sweep to right", "Mapped command action"),
)


class TronControlApp(tk.Tk):
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
        self.title("PUnity Control Grid")
        self.geometry("1420x900")
        self.minsize(1200, 760)
        self.configure(bg=self.BG)

        self._engine_proc: subprocess.Popen[str] | None = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._profile_data: dict[str, object] = {}
        self._field_vars: dict[tuple[str, str], tk.Variable] = {}
        self._gesture_action_labels: dict[str, tk.Label] = {}

        self.profile_name_var = tk.StringVar(value="default.json")
        self.runtime_status_var = tk.StringVar(value="IDLE")

        self._configure_ttk()
        self._build_layout()
        self._set_runtime_status("IDLE", good=False)
        self._refresh_profile_list()
        self._load_profile_into_editor(self.profile_name_var.get())

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._poll_runtime)

    def _configure_ttk(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Tron.TCombobox",
            fieldbackground=self.PANEL_ALT,
            background=self.PANEL_ALT,
            foreground=self.TEXT,
            bordercolor=self.ACCENT,
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT,
        )

    def _build_layout(self) -> None:
        header = tk.Frame(self, bg=self.BG)
        header.pack(fill="x", padx=16, pady=(14, 10))

        title = tk.Label(
            header,
            text="PUNITY // TRON COMMAND CONSOLE",
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

        self.start_btn = self._make_button(status_frame, "Start Engine", self._on_start_engine)
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = self._make_button(status_frame, "Stop Engine", self._on_stop_engine, warn=True)
        self.stop_btn.pack(side="left")

        gestures_panel = self._make_panel(self, "GESTURE TOOLBAR")
        gestures_panel.pack(fill="x", padx=16, pady=(0, 10))
        self._build_gesture_toolbar(gestures_panel.content)

        body = tk.Frame(self, bg=self.BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left_col = tk.Frame(body, bg=self.BG)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_col.grid_rowconfigure(0, weight=2)
        left_col.grid_rowconfigure(1, weight=3)
        left_col.grid_columnconfigure(0, weight=1)

        commands_panel = self._make_panel(left_col, "COMMANDS")
        commands_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        self.commands_text = tk.Text(
            commands_panel.content,
            bg=self.PANEL_ALT,
            fg=self.ACCENT_SOFT,
            insertbackground=self.ACCENT,
            font=("Consolas", 11),
            bd=0,
            relief="flat",
            wrap="word",
        )
        self.commands_text.pack(fill="both", expand=True)
        self.commands_text.configure(state="disabled")

        logs_panel = self._make_panel(left_col, "RUNTIME LOG")
        logs_panel.grid(row=1, column=0, sticky="nsew")

        self.logs_text = ScrolledText(
            logs_panel.content,
            bg="#04070d",
            fg="#9CE8B6",
            insertbackground="#9CE8B6",
            font=("Consolas", 10),
            bd=0,
            relief="flat",
            wrap="word",
            height=14,
        )
        self.logs_text.pack(fill="both", expand=True)
        self.logs_text.configure(state="disabled")

        right_col = tk.Frame(body, bg=self.BG)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right_col.grid_rowconfigure(1, weight=1)
        right_col.grid_columnconfigure(0, weight=1)

        profile_panel = self._make_panel(right_col, "PROFILE MANAGER")
        profile_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_profile_manager(profile_panel.content)

        settings_panel = self._make_panel(right_col, "SETTINGS")
        settings_panel.grid(row=1, column=0, sticky="nsew")
        self._build_settings_editor(settings_panel.content)

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

            name_label = tk.Label(
                card,
                text=name,
                bg=self.PANEL_ALT,
                fg=self.ACCENT,
                font=("Consolas", 11, "bold"),
                anchor="w",
            )
            name_label.pack(fill="x")

            how_label = tk.Label(
                card,
                text=f"How: {how_to}",
                bg=self.PANEL_ALT,
                fg=self.TEXT,
                font=("Consolas", 9),
                justify="left",
                anchor="w",
                wraplength=240,
            )
            how_label.pack(fill="x", pady=(4, 4))

            action_label = tk.Label(
                card,
                text=f"Action: {action}",
                bg=self.PANEL_ALT,
                fg="#8FFFA6",
                font=("Consolas", 9, "bold"),
                justify="left",
                anchor="w",
                wraplength=240,
            )
            action_label.pack(fill="x")

            self._gesture_action_labels[name] = action_label

    def _build_profile_manager(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=self.PANEL)
        row.pack(fill="x")

        profile_label = tk.Label(
            row,
            text="Profile",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Consolas", 10),
        )
        profile_label.pack(side="left", padx=(0, 8))

        self.profile_combo = ttk.Combobox(
            row,
            textvariable=self.profile_name_var,
            state="readonly",
            style="Tron.TCombobox",
            width=28,
        )
        self.profile_combo.pack(side="left", fill="x", expand=True)
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_selected)

        actions = tk.Frame(parent, bg=self.PANEL)
        actions.pack(fill="x", pady=(10, 0))

        self._make_button(actions, "Reload", self._on_reload_profile).pack(side="left", padx=(0, 8))
        self._make_button(actions, "Save", self._on_save_profile).pack(side="left", padx=(0, 8))
        self._make_button(actions, "Save As", self._on_save_as_profile).pack(side="left")

    def _build_settings_editor(self, parent: tk.Frame) -> None:
        grid = tk.Frame(parent, bg=self.PANEL)
        grid.pack(fill="x")

        for row_idx, field in enumerate(SETTINGS_FIELDS):
            label = tk.Label(
                grid,
                text=field.label,
                bg=self.PANEL,
                fg=self.TEXT,
                font=("Consolas", 10),
                anchor="w",
            )
            label.grid(row=row_idx, column=0, sticky="w", padx=(0, 10), pady=3)

            if field.kind == "bool":
                var = tk.BooleanVar(value=False)
                widget = tk.Checkbutton(
                    grid,
                    variable=var,
                    onvalue=True,
                    offvalue=False,
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

            widget.grid(row=row_idx, column=1, sticky="ew", pady=3)
            self._field_vars[(field.section, field.key)] = var

        grid.grid_columnconfigure(1, weight=1)

        mapping_label = tk.Label(
            parent,
            text="Mappings JSON",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Consolas", 10),
            anchor="w",
        )
        mapping_label.pack(fill="x", pady=(10, 4))

        self.mapping_text = ScrolledText(
            parent,
            bg=self.PANEL_ALT,
            fg="#CFF4FF",
            insertbackground=self.ACCENT,
            font=("Consolas", 10),
            bd=0,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.ACCENT,
            height=9,
            wrap="none",
        )
        self.mapping_text.pack(fill="both", expand=True)

    def _refresh_profile_list(self) -> None:
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        names = sorted(path.name for path in PROFILES_DIR.glob("*.json"))

        if not names:
            fallback = PROFILES_DIR / "default.json"
            fallback.write_text("{}\n", encoding="utf-8")
            names = [fallback.name]

        self.profile_combo["values"] = names
        current = self.profile_name_var.get()
        if current not in names:
            self.profile_name_var.set(names[0])

    def _on_profile_selected(self, _event=None) -> None:
        self._load_profile_into_editor(self.profile_name_var.get())

    def _on_reload_profile(self) -> None:
        self._load_profile_into_editor(self.profile_name_var.get())
        self._append_log("Profile reloaded")

    def _load_profile_into_editor(self, profile_name: str) -> None:
        path = PROFILES_DIR / profile_name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Profile Load Error", f"Failed to load profile:\n{exc}")
            return

        self._profile_data = data
        self._apply_profile_to_fields()
        self._refresh_mappings_editor()
        self._refresh_commands_panel()
        self._refresh_gesture_actions()

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

    def _refresh_mappings_editor(self) -> None:
        mappings = self._profile_data.get("mappings", {})
        if not isinstance(mappings, dict):
            mappings = {}

        self.mapping_text.delete("1.0", tk.END)
        self.mapping_text.insert("1.0", json.dumps(mappings, indent=2))

    def _collect_profile_from_fields(self) -> dict[str, object]:
        data = json.loads(json.dumps(self._profile_data)) if self._profile_data else {}

        for field in SETTINGS_FIELDS:
            section = data.setdefault(field.section, {})
            if not isinstance(section, dict):
                section = {}
                data[field.section] = section

            var = self._field_vars[(field.section, field.key)]
            section[field.key] = self._coerce_field_value(field, var.get())

        mappings_raw = self.mapping_text.get("1.0", tk.END).strip()
        mappings = json.loads(mappings_raw) if mappings_raw else {}
        if not isinstance(mappings, dict):
            raise ValueError("Mappings JSON must be an object")
        data["mappings"] = mappings

        return data

    def _coerce_field_value(self, field: SettingField, raw: object) -> object:
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
        self._refresh_commands_panel()
        self._refresh_gesture_actions()

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

    def _refresh_gesture_actions(self) -> None:
        for gesture_name, _how, default_action in GESTURE_GUIDE:
            label = self._gesture_action_labels.get(gesture_name)
            if label is None:
                continue

            if gesture_name.startswith("SWIPE"):
                action_text = self._format_mapping_action(gesture_name)
            else:
                action_text = default_action
            label.configure(text=f"Action: {action_text}")

    def _refresh_commands_panel(self) -> None:
        safety = self._profile_data.get("safety", {})
        mappings = self._profile_data.get("mappings", {})

        kill = "F8"
        if isinstance(safety, dict):
            kill = str(safety.get("kill_switch_key", "f8")).upper()

        lines = [
            "SYSTEM COMMANDS",
            f"  Kill Switch: {kill} (toggle active/pause)",
            "  Stop App: Q or ESC in overlay window",
            "",
            "GESTURE COMMANDS",
            "  OPEN_PALM  -> Enable pointer movement",
            "  PINCHING   -> Mouse down / drag while held",
            "  RELEASE    -> Mouse up",
            "  FIST       -> Pause movement and release drag",
            f"  SWIPE_LEFT -> {self._format_mapping_action('SWIPE_LEFT', mappings)}",
            f"  SWIPE_RIGHT-> {self._format_mapping_action('SWIPE_RIGHT', mappings)}",
            "",
            "LAUNCH",
            f"  CLI: punity --profile {self.profile_name_var.get()}",
            "  GUI: punity-gui",
        ]

        self.commands_text.configure(state="normal")
        self.commands_text.delete("1.0", tk.END)
        self.commands_text.insert("1.0", "\n".join(lines))
        self.commands_text.configure(state="disabled")

    def _format_mapping_action(
        self,
        gesture: str,
        mappings: dict[str, object] | None = None,
    ) -> str:
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
        if self._engine_proc is not None and self._engine_proc.poll() is None:
            return

        if not self._save_profile(show_popup=False):
            return

        profile_path = PROFILES_DIR / self.profile_name_var.get()
        cmd = [sys.executable, "-m", "punity", "--profile", str(profile_path)]

        try:
            self._engine_proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            messagebox.showerror("Start Error", f"Could not start engine:\n{exc}")
            return

        threading.Thread(target=self._read_engine_output, daemon=True).start()
        self._set_runtime_status("RUNNING", good=True)
        self._append_log(f"Engine started with profile {profile_path.name}")

    def _on_stop_engine(self) -> None:
        if self._engine_proc is None:
            return

        if self._engine_proc.poll() is None:
            self._engine_proc.terminate()
            self._append_log("Stop requested")
        else:
            self._append_log("Engine already stopped")

    def _read_engine_output(self) -> None:
        proc = self._engine_proc
        if proc is None or proc.stdout is None:
            return

        for line in proc.stdout:
            self._log_queue.put(line.rstrip())

    def _poll_runtime(self) -> None:
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)

        if self._engine_proc is not None and self._engine_proc.poll() is not None:
            code = self._engine_proc.returncode
            self._append_log(f"Engine exited with code {code}")
            self._engine_proc = None
            self._set_runtime_status("IDLE", good=False)

        self.after(120, self._poll_runtime)

    def _append_log(self, line: str) -> None:
        self.logs_text.configure(state="normal")
        self.logs_text.insert(tk.END, line + "\n")
        self.logs_text.see(tk.END)
        self.logs_text.configure(state="disabled")

    def _set_runtime_status(self, text: str, good: bool) -> None:
        self.runtime_status_var.set(text)
        if good:
            self.status_chip.configure(bg="#11351f", fg=self.GOOD, highlightbackground=self.GOOD)
        else:
            self.status_chip.configure(bg="#3a2512", fg=self.WARN, highlightbackground=self.WARN)

    def _on_close(self) -> None:
        if self._engine_proc is not None and self._engine_proc.poll() is None:
            try:
                self._engine_proc.terminate()
            except Exception:
                pass
        self.destroy()


def main() -> None:
    app = TronControlApp()
    app.mainloop()


if __name__ == "__main__":
    main()




