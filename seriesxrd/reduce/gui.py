"""Tabbed Tkinter GUI for the batch reduction stage.

This is stage 2 of the pipeline: calibrate (calib/gui.py) -> accept a PONI ->
reduce (this stage) -> analysis (analysis/gui.py). Accepting a calibration
fills in the Calibration tab below automatically; a finished reduction here
hands its output HDF5 to the analysis stage the same way.

Workflow left-to-right across tabs:
    1 Calibration — the accepted calibration (auto-filled; import a prior run optionally)
    2 Dataset     — pick the frame folder, preview the file list
    3 Settings    — integration parameters
    4 Run         — launch the crash-isolated worker, watch progress + log
    5 Review      — inspect the reduced HDF5 before handing it to analysis
    6 Gallery     — per-frame cake/1D thumbnails; click to exclude bad frames

Same supervision model as calib/gui.py: pyFAI runs in reduce/worker.py as a
subprocess; this process never imports pyFAI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import queue
import subprocess
import sys
import threading

from ..core.config import TOOL_NAME, read_json, write_json, ensure_dir, now_iso, now_timestamp, output_base
from ..core.handoff import load_handoff
from ..core.naming import next_available_path
from ..core.processes import terminate_process_tree, worker_popen
from ..guikit.theme import BG, BG2, FG, ACCENT, ACCENT2, WARN, MUTED, ENTRY_BG
from ..guikit.tkstyle import apply_dark_theme
from ..guikit.tooltip import ToolTip as _ToolTip
from ..guikit.mpl_embed import embed_figure, make_canvas_responsive
from .processing import scan_dataset, DEFAULT_PATTERNS


def _tk_imports():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    return tk, ttk, filedialog, messagebox


HELP = {
    "handoff_file":   "The accepted calibration (PONI + mask) this reduction uses. Auto-fills when you accept a calibration on the Calibration stage; use Import to pull one from an earlier run.",
    "dataset_dir":    "Folder containing the sample dataset frames to integrate.",
    "file_patterns":  "Semicolon-separated glob patterns, e.g. *.tif;*.edf",
    "npt_1d":         ("Number of bins in the 1D intensity pattern. Leave blank for auto: about 1 bin "
                       "per pixel of radial extent from the detector geometry (pyFAI's rule of thumb). "
                       "Too few bins under-samples sharp peaks, so patterns look stepped and peak "
                       "fitting degrades."),
    "h5_data_path":   ("HDF5/NeXus stack containers only: the dataset path holding the frame "
                       "stack (e.g. entry/data/data). Blank = auto-detect (NeXus convention "
                       "first, else the largest 3D image dataset). Plain image files ignore "
                       "this."),
    "method":         "pyFAI 1D integration method. csr is fast after the first frame.",
    "robust_1d":      ("Also computes a spot-suppressed pattern: the mean of a narrow azimuthal "
                       "quantile band around the median. Rejects diamond spots like a median does, "
                       "but without the median's integer-count quantization."),
    "robust_quant_halfwidth": ("Half-width of the azimuthal quantile band the robust pattern averages "
                               "over. Default 0.05 = the 45-55% band. Set to 0 for a pure median: on "
                               "integer photon counts that quantizes, so low-intensity patterns render "
                               "as staircases and clean/baseline inherit the steps."),
    "sigmaclip_1d":   "Also computes an azimuthal sigma-clipped (trimmed-mean) pattern. Rejects diamond spots like the median does, but keeps real peaks on sparse/textured rings that a median would drop. This is the recommended fit source for Step 2 peak fitting.",
    "save_cakes":     "Also saves 2D cakes. Increases the output file size; needed for azimuthal analysis.",
    "cake_every":     "Save a cake for every Nth frame only, to bound file size.",
    "num_workers":    "Parallel worker processes. 0 = automatic (CPU count - 1).",
    "make_thumbnails": "Renders a small cake+1D preview per frame during reduction, for the Gallery tab. Turn off for very large datasets to save time and disk space.",
    "reduced_h5_file": "A reduced_*.h5 produced by a reduction run. Inspect it here before handing it to the Analysis stage.",
}


class ReductionApp:
    def __init__(self, config_path: "str | Path", parent=None):
        tk, ttk, filedialog, messagebox = _tk_imports()
        self.tk, self.ttk, self.filedialog, self.messagebox = tk, ttk, filedialog, messagebox
        self.config_path = Path(config_path).expanduser().resolve()
        self.config: Dict[str, Any] = read_json(self.config_path)
        self.config.setdefault("session_config_path", str(self.config_path))
        if parent is None:
            self._owns_root = True
            self.root = tk.Tk()
            self.root.title(f"{TOOL_NAME} Reduction")
            self.root.geometry("1180x780")
            self.root.minsize(960, 640)
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        else:
            self._owns_root = False
            self.root = parent.winfo_toplevel()
        self._embed_parent = parent  # None when standalone, ttk.Frame when embedded
        self.vars: Dict[str, Any] = {}
        self._run_proc: "subprocess.Popen | None" = None
        # Thread-safe logging: worker threads push lines onto this queue; a poller
        # on the Tk thread drains them into the text widget.
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        # Watch-mode control events (live_file/total/done) from the watcher's
        # stdout-reader thread, drained by the same main-thread poller.
        self._watch_queue: "queue.Queue[tuple]" = queue.Queue()
        # History buffer so no lines are lost before the console window is opened.
        self._log_history: "list[str]" = []
        # Handoff and dataset state for status bar
        self._handoff_state: str = "none"
        self._frame_count: int = 0
        self._worker_status: str = "idle"
        # Listeners notified with the reduced .h5 path when a reduction finishes,
        # so the analysis stage can auto-fill its input/output without a manual
        # "Use latest reduced output" click.
        self._reduced_listeners: "list" = []
        self._build_gui()
        self._drain_log_queue()
        self.log("GUI initialized")
        self.save_config(silent=True)
        self._update_status_bar()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_gui(self):
        tk, ttk = self.tk, self.ttk
        # Only theme globally when standalone; embedded, the host already did.
        if self._owns_root:
            apply_dark_theme(self.root, ttk)
        _container = self._embed_parent if self._embed_parent is not None else self.root
        outer = ttk.Frame(_container, padding=6)
        outer.pack(fill="both", expand=True)

        # Top bar with title and "Open Console Logs" button (mirrors calib/gui.py)
        topbar = ttk.Frame(outer)
        topbar.pack(fill="x", pady=(0, 6))
        ttk.Label(topbar, text="Reduction", font=("TkDefaultFont", 14, "bold")).pack(side="left")
        ttk.Button(topbar, text="View log", command=self.open_console_logs).pack(side="right", padx=4)

        # Status bar at the bottom (packed before notebook so it carves space first)
        self._status_bar_frame = ttk.Frame(outer, relief="sunken")
        self._status_bar_frame.pack(side="bottom", fill="x", pady=(2, 0))

        self.nb = ttk.Notebook(outer)
        self.nb.pack(fill="both", expand=True)
        self.tabs: Dict[str, Any] = {}
        for name, builder in [
            ("1 Calibration", self._tab_calibration),
            ("2 Dataset",  self._tab_dataset),
            ("3 Settings", self._tab_settings),
            ("4 Run",      self._tab_run),
            ("5 Review",   self._tab_review),
            ("6 Gallery",  self._tab_gallery),
        ]:
            frame = ttk.Frame(self.nb, padding=10)
            builder(frame)
            self.nb.add(frame, text=name)
            self.tabs[name] = frame

        # Populate the status bar after all widgets exist
        self._build_status_bar()

    def _build_status_bar(self):
        """Populate the persistent structured status bar."""
        ttk = self.ttk
        bar = self._status_bar_frame
        # Session name (left)
        self.status_session = ttk.Label(bar, text="", foreground=MUTED, anchor="w")
        self.status_session.pack(side="left", padx=(6, 12))
        # Handoff state
        self.status_handoff = ttk.Label(bar, text="calibration: none", foreground=MUTED, anchor="w")
        self.status_handoff.pack(side="left", padx=(0, 12))
        # Dataset frame count
        self.status_frames = ttk.Label(bar, text="frames: —", foreground=MUTED, anchor="w")
        self.status_frames.pack(side="left", padx=(0, 12))
        # Worker status (right-aligned)
        self.status_worker = ttk.Label(bar, text="idle", foreground=MUTED, anchor="e")
        self.status_worker.pack(side="right", padx=6)

    def _update_status_bar(self):
        """Refresh all status bar labels from current app state."""
        try:
            session = self.config.get("session_name", "")
            if hasattr(self, "status_session"):
                self.status_session.configure(
                    text=f"session: {session}" if session else "session: (unnamed)")
            if hasattr(self, "status_handoff"):
                self.status_handoff.configure(text=f"calibration: {self._handoff_state}")
            if hasattr(self, "status_frames"):
                fc = self._frame_count
                self.status_frames.configure(
                    text=f"frames: {fc}" if fc > 0 else "frames: —")
            if hasattr(self, "status_worker"):
                self.status_worker.configure(text=self._worker_status)
        except Exception:
            pass

    # -- shared small widgets ------------------------------------------------

    def field(self, parent, key, label, browse=None, row=None, width=80):
        """Entry field with a tooltip help label instead of a permanent label row."""
        tk, ttk = self.tk, self.ttk
        var = tk.StringVar(value=str(self.config.get(key, "")))
        self.vars[key] = var
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=4, pady=3)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="we", padx=4, pady=3)
        if not hasattr(self, "entry_widgets"):
            self.entry_widgets: Dict[str, Any] = {}
        self.entry_widgets[key] = entry
        if browse:
            ttk.Button(parent, text="Browse",
                       command=lambda: self.browse_into(key, browse)).grid(row=row, column=2, padx=4)
        txt = HELP.get(key, "")
        if txt:
            _ToolTip(lbl, txt)
            _ToolTip(entry, txt)
        parent.columnconfigure(1, weight=1)

    def checkbox(self, parent, key, label, row=None):
        """Checkbox with a tooltip help label instead of a permanent label row."""
        tk, ttk = self.tk, self.ttk
        var = tk.BooleanVar(value=bool(self.config.get(key, False)))
        self.vars[key] = var
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=3)
        txt = HELP.get(key, "")
        if txt:
            _ToolTip(cb, txt)

    def browse_into(self, key, mode):
        if mode == "dir":
            value = self.filedialog.askdirectory(title=f"Select {key}")
        else:
            value = self.filedialog.askopenfilename(title=f"Select {key}")
        if value:
            self.vars[key].set(value)
            self.save_config(silent=True)

    def pull_vars(self):
        for key, var in self.vars.items():
            self.config[key] = var.get()

    def save_config(self, silent=False):
        self.pull_vars()
        self.config["updated_at"] = now_iso()
        write_json(self.config_path, self.config)
        if not silent:
            self.log(f"Config saved: {self.config_path}")

    # ------------------------------------------------------------------
    # Thread-safe logging (mirrors calib/gui.py pattern)
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "INFO"):
        line = f"[{now_iso()}] [{level}] {message}"
        print(line, flush=True)
        # Touching Tk widgets off the main thread crashes on Linux/macOS, so
        # worker threads queue the line for the main-thread poller instead.
        if threading.current_thread() is not threading.main_thread():
            self._log_queue.put(line)
            return
        self._insert_log_line(line)

    def _insert_log_line(self, line: str):
        # Always append to history buffer (cap at 5000 lines).
        self._log_history.append(line)
        if len(self._log_history) > 5000:
            self._log_history = self._log_history[-5000:]
        # Write to the console window's Text widget if it exists and is still alive.
        if hasattr(self, "log_text"):
            try:
                self.log_text.configure(state="normal")
                self.log_text.insert("end", line + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            except self.tk.TclError:
                pass
        # Keep existing in-tab run log widget updated too.
        if hasattr(self, "run_log_text"):
            try:
                self.run_log_text.configure(state="normal")
                self.run_log_text.insert("end", line + "\n")
                self.run_log_text.see("end")
                self.run_log_text.configure(state="disabled")
            except self.tk.TclError:
                pass
        # Update simple status_var for backward compat if it still exists.
        if hasattr(self, "status_var"):
            try:
                self.status_var.set(message if len(message) <= 120 else message[:117] + "...")
            except Exception:
                pass

    def _drain_log_queue(self):
        """Recurring main-thread poller that flushes queued log lines into the widget."""
        if getattr(self, "_closing", False):
            return  # stop rescheduling once the pane is shutting down
        try:
            while True:
                line = self._log_queue.get_nowait()
                self._insert_log_line(line)
        except queue.Empty:
            pass
        # Watch-mode events pushed by the watcher's reader thread (Tk is not
        # thread-safe, so the thread must never touch widgets or root.after).
        try:
            while True:
                evt = self._watch_queue.get_nowait()
                try:
                    kind, payload = evt
                    if kind == "live_file":
                        self._watch_live_file(payload)
                    elif kind == "total":
                        self._watch_status.configure(
                            text=f"watching — {payload} frame(s) so far",
                            foreground=MUTED)
                    elif kind == "done":
                        self._watch_done(*payload)
                except Exception as e:
                    self.log(f"watch-event handler failed: {e!r}", "WARN")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    # ------------------------------------------------------------------
    # Console log window (mirrors calib/gui.py open_console_logs)
    # ------------------------------------------------------------------

    def open_console_logs(self):
        """Open (or raise) the separate console-log Toplevel window."""
        tk, ttk = self.tk, self.ttk
        # If the window already exists and is still alive, bring it to front.
        try:
            if getattr(self, "_log_window", None) and self._log_window.winfo_exists():
                self._log_window.deiconify()
                self._log_window.lift()
                self._log_window.focus_set()
                return
        except self.tk.TclError:
            pass
        # Create the Toplevel.
        self._log_window = tk.Toplevel(self.root)
        self._log_window.title("SeriesXRD — Reduction log")
        self._log_window.geometry("900x420")
        self._log_window.configure(bg=BG)
        # Build the Text + scrollbar.
        self.log_text = tk.Text(
            self._log_window, wrap="word", state="disabled",
            font=("TkFixedFont", 10), bg=BG2, fg=FG,
            insertbackground=FG, selectbackground=ACCENT,
        )
        scroll = ttk.Scrollbar(self._log_window, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)
        # Populate from history so no lines are lost.
        self.log_text.configure(state="normal")
        if self._log_history:
            self.log_text.insert("end", "\n".join(self._log_history) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        # Closing hides (withdraws) instead of destroying, so the live widget
        # keeps receiving log lines and reopening is instant.
        self._log_window.protocol("WM_DELETE_WINDOW", self._hide_console_logs)

    def _hide_console_logs(self):
        """Hide (withdraw) the console log window without destroying it."""
        if getattr(self, "_log_window", None):
            try:
                self._log_window.withdraw()
            except self.tk.TclError:
                pass

    # ------------------------------------------------------------------
    # Tab 1 — Handoff
    # ------------------------------------------------------------------

    def _tab_calibration(self, frame):
        ttk = self.ttk
        self.field(frame, "handoff_file", "Active calibration", browse=None, row=0)
        ttk.Label(frame, text="The accepted geometry and mask appear here automatically. "
                  "Use a previous calibration only when processing data collected with the same setup.",
                  foreground=MUTED, wraplength=640, justify="left").grid(
                  row=1, column=0, columnspan=3, sticky="w", padx=4)
        btns = ttk.Frame(frame)
        btns.grid(row=2, column=0, columnspan=3, sticky="w", padx=2, pady=8)
        ttk.Button(btns, text="Use latest calibration",
                   command=self._use_latest_calibration).pack(side="left", padx=2)
        ttk.Button(btns, text="Import a previous run…",
                   command=self._import_calibration_file).pack(side="left", padx=2)
        self.handoff_text = self.tk.Text(frame, height=14, bg=BG2, fg=FG, insertbackground=FG,
                                         relief="flat", state="disabled", font=("TkFixedFont", 10))
        self.handoff_text.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(3, weight=1)
        if self.config.get("handoff_file"):
            self.root.after(200, self.verify_handoff)

    def _import_calibration_file(self):
        """Import an accepted calibration from a previous run (its handoff JSON)."""
        p = self.filedialog.askopenfilename(
            title="Import a previously calibrated run",
            filetypes=[("Calibration record", "*.json"), ("All files", "*.*")])
        if p:
            self.set_handoff(p)

    def add_reduced_listener(self, fn) -> None:
        """Register a callback invoked with the reduced .h5 path when a reduction
        completes (used to auto-wire the analysis stage's input)."""
        if callable(fn):
            self._reduced_listeners.append(fn)

    def set_handoff(self, handoff_path) -> None:
        """Populate the handoff from an external source (e.g. the calibration
        pane just exported one) and verify it. Quiet — no dialogs — since this
        may fire while the user is on another tab."""
        p = str(handoff_path or "")
        if not p:
            return
        self.config["handoff_file"] = p
        if "handoff_file" in self.vars:
            self.vars["handoff_file"].set(p)
        self.save_config(silent=True)
        self.log(f"Calibration handoff received: {p}")
        try:
            self.nb.select(self.tabs["1 Calibration"])
        except Exception:
            pass
        self.verify_handoff()

    def _use_latest_calibration(self) -> None:
        """Find the newest accepted-calibration handoff in this workspace."""
        from ..core.handoff import find_latest_handoff
        from ..core.config import default_workspace_paths
        self.pull_vars()
        roots = []
        ws = self.config.get("workspace_root", "")
        if ws:
            roots.append(default_workspace_paths(ws)["accepted_output_root"])
        for k in ("accepted_output_root", "processed_root"):
            v = self.config.get(k, "")
            if v:
                roots.append(v)
        latest = None
        for r in roots:
            latest = find_latest_handoff(r)
            if latest:
                break
        if not latest:
            self.messagebox.showinfo(
                "No calibration found",
                "No accepted-calibration handoff was found in this workspace.\n\n"
                "Run and accept a calibration first, or use Browse to select a handoff JSON "
                "from an earlier run.")
            return
        self.set_handoff(str(latest))

    def verify_handoff(self):
        self.pull_vars()
        handoff = load_handoff(self.config.get("handoff_file", ""))
        self.handoff_text.configure(state="normal")
        self.handoff_text.delete("1.0", "end")
        self.handoff_text.insert("end", "\n".join(handoff.summary_lines()) + "\n")
        self.handoff_text.configure(state="disabled")
        if handoff.ok:
            gen = handoff.accepted_generation or (handoff.path.name if handoff.path else "?")
            self.log(f"Handoff OK: {gen}")
            self._handoff_state = f"{gen} OK"
        else:
            self.log("Handoff has problems — see Handoff tab", "ERROR")
            self._handoff_state = "invalid"
        self._update_status_bar()
        return handoff

    # ------------------------------------------------------------------
    # Tab 2 — Dataset
    # ------------------------------------------------------------------

    def _tab_dataset(self, frame):
        ttk = self.ttk
        self.field(frame, "dataset_dir", "Data folder", browse="dir", row=0)
        self.field(frame, "file_patterns", "File patterns", row=2, width=40)
        self.field(frame, "h5_data_path", "HDF5 data path (blank=auto)", row=3, width=40)
        self.checkbox(frame, "recursive", "Search subfolders recursively", row=4)
        ttk.Button(frame, text="Scan dataset", command=self.scan_dataset_clicked).grid(row=6, column=0, sticky="w", padx=4, pady=8)
        self.dataset_text = self.tk.Text(frame, height=14, bg=BG2, fg=FG, insertbackground=FG,
                                         relief="flat", state="disabled", font=("TkFixedFont", 10))
        self.dataset_text.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(7, weight=1)

    def scan_dataset_clicked(self):
        self.pull_vars()
        files = scan_dataset(self.config.get("dataset_dir", ""),
                             self.config.get("file_patterns", DEFAULT_PATTERNS),
                             bool(self.vars["recursive"].get()))
        total_bytes = sum(f.stat().st_size for f in files[:5000])
        size_note = (f"~{total_bytes/1e6:.0f} MB"
                     + (" (first 5000 files)" if len(files) > 5000 else ""))
        # HDF5 stack containers count as one file but many frames — expand so
        # the preview shows what the reduction will actually process.
        from ..core.io import expand_frame_sources, frame_display_name
        sources, n_stacks = expand_frame_sources(
            files, str(self.config.get("h5_data_path", "") or ""))
        self._frame_count = len(sources)
        lines = [f"{len(sources)} frames found"
                 + (f" ({n_stacks} HDF5 stack container(s) expanded)"
                    if n_stacks else "")]
        files = sources
        if files:
            lines.append(size_note)
            lines.append("")
            _nm = frame_display_name
            lines += [f"  {_nm(f)}" for f in files[:15]]
            if len(files) > 30:
                lines.append(f"  ... {len(files) - 30} more ...")
            lines += [f"  {_nm(f)}" for f in files[-15:]] if len(files) > 15 else []
        self.dataset_text.configure(state="normal")
        self.dataset_text.delete("1.0", "end")
        self.dataset_text.insert("end", "\n".join(lines) + "\n")
        self.dataset_text.configure(state="disabled")
        self.log(f"Dataset scan: {len(files)} frames")
        self._update_status_bar()
        self.save_config(silent=True)

    # ------------------------------------------------------------------
    # Tab 3 — Settings
    # ------------------------------------------------------------------

    def _tab_settings(self, frame):
        tk, ttk = self.tk, self.ttk
        self.field(frame, "npt_1d", "1D bins (blank=auto)", row=0, width=14)
        ttk.Label(frame, text="Integration unit").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        unit_var = tk.StringVar(value=str(self.config.get("unit", "q_A^-1")))
        self.vars["unit"] = unit_var
        ttk.Combobox(frame, textvariable=unit_var, values=["q_A^-1", "q_nm^-1", "2th_deg", "2th_rad"],
                     width=14, state="readonly").grid(row=2, column=1, sticky="w", padx=4)
        self.field(frame, "method", "pyFAI 1D method", row=3, width=14)
        self.field(frame, "polarization_factor", "Polarization factor (optional)", row=5, width=14)
        self.checkbox(frame, "robust_1d", "Robust 1D pattern (azimuthal quantile band — suppresses diamond spots)", row=6)
        self.field(frame, "robust_quant_halfwidth", "Robust quantile half-width", row=7, width=14)
        self.checkbox(frame, "sigmaclip_1d", "Sigma-clip 1D pattern (azimuthal trimmed mean — keeps textured-ring peaks)", row=8)
        self.checkbox(frame, "save_cakes", "Save 2D cakes", row=9)
        self.field(frame, "npt_radial", "Cake radial bins", row=11, width=14)
        self.field(frame, "npt_azimuthal", "Cake azimuth bins", row=12, width=14)
        self.field(frame, "cake_every", "Cake every Nth frame", row=13, width=14)
        self.config.setdefault("make_thumbnails", True)
        self.checkbox(frame, "make_thumbnails", "Render per-frame gallery thumbnails", row=14)
        self.field(frame, "num_workers", "Parallel workers (0 = auto)", row=15, width=14)

        # Disable the cake-only fields when "Save 2D cakes" is unchecked.
        def _toggle_cake_fields(*_):
            on = bool(self.vars["save_cakes"].get())
            for k in ("npt_radial", "npt_azimuthal", "cake_every"):
                w = self.entry_widgets.get(k)
                if w is not None:
                    w.configure(state=("normal" if on else "disabled"))
        self.vars["save_cakes"].trace_add("write", _toggle_cake_fields)
        _toggle_cake_fields()

    # ------------------------------------------------------------------
    # Tab 4 — Run
    # ------------------------------------------------------------------

    def _tab_run(self, frame):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(frame)
        top.pack(fill="x")
        self.run_btn = ttk.Button(top, text="Run reduction", command=self.run_reduction)
        self.run_btn.pack(side="left", padx=4, pady=4)
        self.cancel_btn = ttk.Button(top, text="Cancel", command=self.cancel_reduction, state="disabled")
        self.cancel_btn.pack(side="left", padx=4, pady=4)

        # Live watch mode: append frames as they land during a beamtime.
        watch_row = ttk.Frame(frame)
        watch_row.pack(fill="x")
        self.watch_btn = ttk.Button(watch_row, text="Start live processing",
                                    command=self.start_watch)
        self.watch_btn.pack(side="left", padx=4, pady=4)
        _ToolTip(self.watch_btn, (
            "Process new frames as they arrive and refresh the selected analysis "
            "steps after each batch. Run a normal reduction afterward to create "
            "the final archival dataset."))
        self.watch_stop_btn = ttk.Button(watch_row, text="Stop live processing",
                                         command=self.stop_watch, state="disabled")
        self.watch_stop_btn.pack(side="left", padx=4, pady=4)
        ttk.Label(watch_row, text="Analysis:", foreground=MUTED).pack(
            side="left", padx=(12, 2))
        self.vars["watch_steps"] = tk.StringVar(
            value=str(self.config.get("watch_steps", "12")))
        _wsteps = ttk.Combobox(watch_row, textvariable=self.vars["watch_steps"],
                               values=["off", "12", "123"], state="readonly",
                               width=5)
        _wsteps.pack(side="left", padx=2)
        _ToolTip(_wsteps, "Analysis steps re-run as frames arrive: 12 = "
                          "background + peaks (default), 123 adds phase ID "
                          "(uses the workspace's configured candidates), "
                          "off = only reduce.")
        ttk.Label(watch_row, text="Poll (s):", foreground=MUTED).pack(
            side="left", padx=(10, 2))
        self.vars["watch_poll"] = tk.StringVar(
            value=str(self.config.get("watch_poll", "5")))
        ttk.Entry(watch_row, textvariable=self.vars["watch_poll"],
                  width=5).pack(side="left", padx=2)
        self._watch_status = ttk.Label(watch_row, text="", foreground=MUTED)
        self._watch_status.pack(side="left", padx=12)
        self._watch_proc = None

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=4, pady=6)
        self.progress_label = ttk.Label(frame, text="Idle", foreground=MUTED)
        self.progress_label.pack(anchor="w", padx=6)
        # In-tab run log (kept alongside the console-log window)
        self.run_log_text = tk.Text(frame, bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                                    state="disabled", font=("TkFixedFont", 9))
        self.run_log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def run_reduction(self):
        if self._run_proc is not None:
            self.messagebox.showinfo("Busy", "A reduction is already running.")
            return
        if getattr(self, "_watch_proc", None) is not None:
            self.messagebox.showinfo(
                "Busy", "Live watch is running — stop it before a batch "
                        "reduction (both would write to the same session).")
            return
        self.save_config(silent=True)
        handoff = self.verify_handoff()
        if not handoff.ok:
            self.messagebox.showerror("Invalid handoff", "\n".join(handoff.problems))
            return
        backend_dir = self.config.get("backend_dir", str(Path(__file__).resolve().parents[1]))
        python_exe = Path(self.config.get("python_exe", sys.executable))
        logs_root = self.config.get("logs_root", "") or str(output_base(self.config) / "logs")
        ensure_dir(Path(logs_root))
        out_json = str(next_available_path(Path(logs_root) / f"reduce_{now_timestamp()}.json"))
        worker_script = str(Path(backend_dir) / "reduce" / "worker.py")
        if not Path(worker_script).is_file():
            self.messagebox.showerror(
                "Worker not found",
                f"Reduction worker script not found:\n{worker_script}\n\n"
                "Check 'backend_dir' in the session config.")
            return
        cmd = [str(python_exe), worker_script, "--config", str(self.config_path), "--output-json", out_json]
        self.log("Worker command: " + " ".join(cmd))
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(value=0)
        self.progress_label.configure(text="Starting worker ...")
        self._cancel_requested = False
        self._worker_status = "running"
        self._update_status_bar()

        def _worker_thread():
            try:
                proc = worker_popen(cmd, cwd=backend_dir, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
                self._run_proc = proc
                assert proc.stdout is not None
                try:
                    for line in proc.stdout:
                        line = line.rstrip()
                        if line.startswith("[PROGRESS]"):
                            try:
                                _, done, total = line.split()
                                self.root.after(0, self._update_progress, int(done), int(total))
                            except Exception:
                                pass
                        else:
                            self.log(line)
                except (ValueError, OSError):
                    pass
                rc = int(proc.wait())
                self.root.after(0, self._run_done, rc, out_json)
            except Exception as e:
                self.root.after(0, self._run_error, repr(e))
        threading.Thread(target=_worker_thread, daemon=True).start()

    def _update_progress(self, done: int, total: int):
        self.progress.configure(maximum=total, value=done)
        self.progress_label.configure(text=f"{done} / {total} frames")

    def _run_done(self, returncode: int, out_json: str):
        self._run_proc = None
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        if returncode != 0:
            if getattr(self, "_cancel_requested", False):
                self._cancel_requested = False
                self._worker_status = "cancelled"
                self._update_status_bar()
                self.progress_label.configure(
                    text="Cancelled — partial output was discarded; run reduction again to write a fresh HDF5.",
                    foreground=MUTED)
                self.log("Reduction cancelled by user.", "WARN")
                return
            self._worker_status = "failed"
            self._update_status_bar()
            self.progress_label.configure(text=f"Failed (return code {returncode})", foreground=WARN)
            self.messagebox.showerror("Reduction failed", f"Worker return code {returncode}\nSee the Run tab log.")
            return
        manifest = read_json(out_json)
        n = manifest.get("n_frames", "?")
        nf = manifest.get("n_failed", 0)
        h5 = manifest.get("h5_file", "")
        self._worker_status = "done"
        self._frame_count = int(n) if isinstance(n, int) else self._frame_count
        self._update_status_bar()
        self.progress_label.configure(text=f"Done: {n} frames ({nf} failed) -> {h5}", foreground=ACCENT2)
        self.log(f"Reduction complete: {h5}")
        if h5:
            self.config["reduced_h5_file"] = h5
            if "reduced_h5_file" in self.vars:
                self.vars["reduced_h5_file"].set(h5)
            self.log("Reduced HDF5 ready — see the Review tab to check it before analysis.")
            try:
                self.inspect_h5_clicked()
            except Exception as e:
                self.log(f"Auto-inspect failed: {e!r}", "WARN")
            try:
                self.load_gallery()
            except Exception as e:
                self.log(f"Auto gallery load failed: {e!r}", "WARN")
            for fn in self._reduced_listeners:
                try:
                    fn(h5)
                except Exception as e:
                    self.log(f"Reduced-output listener failed: {e!r}", "WARN")
        if nf:
            self.messagebox.showwarning("Reduction finished with failures",
                                        f"{nf} of {n} frames failed — see manifest:\n{manifest.get('manifest_file', '')}")

    def _run_error(self, err: str):
        self._run_proc = None
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self._worker_status = "failed"
        self._update_status_bar()
        self.progress_label.configure(text="Launch error", foreground=WARN)
        self.messagebox.showerror("Worker launch error", err)

    def cancel_reduction(self):
        proc = self._run_proc
        if proc is not None and proc.poll() is None:
            self._cancel_requested = True
            self.cancel_btn.configure(state="disabled")
            self.progress_label.configure(text="Cancelling ...", foreground=MUTED)
            terminate_process_tree(proc)
            self.log("Cancel requested — stopped worker process tree", "WARN")

    # ------------------------------------------------------------------
    # Live watch mode (seriesxrd-watch as a supervised subprocess)
    # ------------------------------------------------------------------

    def start_watch(self):
        if self._watch_proc is not None:
            self.messagebox.showinfo("Busy", "Already watching.")
            return
        if self._run_proc is not None:
            self._watch_status.configure(
                text="A batch reduction is running — wait or cancel it first.",
                foreground=WARN)
            return
        self.save_config(silent=True)
        handoff = load_handoff(self.config.get("handoff_file", ""))
        if not handoff.ok:
            msg = "No valid calibration handoff: " + "; ".join(handoff.problems)
            self._watch_status.configure(text=msg, foreground=WARN)
            self.log(msg, "WARN")
            return
        backend_dir = self.config.get(
            "backend_dir", str(Path(__file__).resolve().parents[1]))
        python_exe = Path(self.config.get("python_exe", sys.executable))
        watch_script = Path(backend_dir) / "reduce" / "watch.py"
        if not watch_script.is_file():
            self._watch_status.configure(
                text=f"watch.py not found under {backend_dir}", foreground=WARN)
            return
        steps = str(self.vars["watch_steps"].get() or "12")
        steps = "" if steps == "off" else steps
        poll = str(self.vars["watch_poll"].get() or "5").strip() or "5"
        cmd = [str(python_exe), str(watch_script),
               "--config", str(self.config_path),
               "--steps", steps, "--poll", poll]
        self.log("Watch command: " + " ".join(cmd))
        self.watch_btn.configure(state="disabled")
        self.watch_stop_btn.configure(state="normal")
        self.run_btn.configure(state="disabled")
        self._watch_status.configure(text="watching — waiting for frames …",
                                     foreground=MUTED)
        self._worker_status = "watching"
        self._update_status_bar()

        def _watch_thread():
            try:
                proc = worker_popen(cmd, cwd=backend_dir,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
                self._watch_proc = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip()
                    self.log(line)                      # queue-based, thread-safe
                    if "[WATCH] live file: " in line:
                        path = line.split("live file: ", 1)[1]
                        path = path.split(" (npt_1d=", 1)[0].strip()
                        self._watch_queue.put(("live_file", path))
                    elif "frame(s) -> " in line and line.endswith("total"):
                        try:
                            total = int(line.rsplit("-> ", 1)[1].split()[0])
                            self._watch_queue.put(("total", total))
                        except (ValueError, IndexError):
                            pass
                rc = int(proc.wait())
                self._watch_queue.put(("done", (rc, "")))
            except Exception as e:
                self._watch_queue.put(("done", (-1, repr(e))))
        threading.Thread(target=_watch_thread, daemon=True).start()

    def _watch_live_file(self, path: str):
        """The watcher announced its live output — hand it to Review/analysis
        listeners right away so downstream views can follow the growing file."""
        self.config["reduced_h5_file"] = path
        if "reduced_h5_file" in self.vars:
            self.vars["reduced_h5_file"].set(path)
        self.save_config(silent=True)
        for fn in self._reduced_listeners:
            try:
                fn(path)
            except Exception as e:
                self.log(f"Reduced-output listener failed: {e!r}", "WARN")

    def _watch_done(self, returncode: int, err: str = ""):
        self._watch_proc = None
        self.watch_btn.configure(state="normal")
        self.watch_stop_btn.configure(state="disabled")
        self.run_btn.configure(state="normal")
        self._worker_status = "idle"
        self._update_status_bar()
        if returncode not in (0, None):
            self._watch_status.configure(
                text=f"watch ended (rc={returncode})" + (f": {err}" if err else "")
                     + " — see the log", foreground=WARN)
        else:
            self._watch_status.configure(text="watch stopped", foreground=MUTED)
            self.log("Watch stopped. The live file is a working view — run a "
                     "full reduction for the archival file.")

    def stop_watch(self):
        proc = self._watch_proc
        if proc is not None and proc.poll() is None:
            self._watch_status.configure(text="stopping — finishing the "
                                               "current batch …",
                                          foreground=MUTED)
            proc.terminate()   # SIGTERM → the watcher flushes a final analysis

    # ------------------------------------------------------------------
    # Tab 5 — Review (read-only HDF5 checkpoint before analysis)
    # ------------------------------------------------------------------

    def _tab_review(self, frame):
        tk, ttk = self.tk, self.ttk
        self.field(frame, "reduced_h5_file", "Reduced data file", browse="file", row=0)
        ctrl = ttk.Frame(frame)
        ctrl.grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=8)
        ttk.Button(ctrl, text="Inspect data", command=self.inspect_h5_clicked).pack(
            side="left")
        # Default to separated (one frame per x-axis, scrollable); overlay is the
        # toggleable option for comparing patterns on a shared axis.
        self._review_overlay = tk.BooleanVar(value=bool(self.config.get("review_overlay", False)))
        ttk.Checkbutton(ctrl, text="Overlay patterns", variable=self._review_overlay,
                        command=self._on_review_overlay_toggle).pack(side="left", padx=(12, 0))
        _diag_btn = ttk.Button(ctrl, text="Diagnose waviness",
                               command=lambda: self._run_straighten_job("diagnose"))
        _diag_btn.pack(side="left", padx=(12, 0))
        _ToolTip(_diag_btn, (
            "Fits r(φ) to the strongest rings of every saved cake and reports the "
            "waviness amplitude, the 1D doublet splitting it causes (~2·A1), and "
            "the implied transverse sample offset in mm. Needs cakes in the file "
            "(Settings: save 2D cakes). Wavy rings mean the sample sat off the "
            "calibrant position — the proper fix is re-refining the geometry and "
            "re-reducing."))
        _str_btn = ttk.Button(ctrl, text="Write straightened 1D",
                              command=lambda: self._run_straighten_job("straighten"))
        _str_btn.pack(side="left", padx=(6, 0))
        _ToolTip(_str_btn, (
            "Rescue channel for wavy data you can't re-collect or re-reduce: "
            "aligns each cake's rings before averaging and writes the corrected "
            "mean + spot-suppressed median to /patterns/intensity_straightened "
            "(+_robust) plus /frames/waviness_A1. Frames without a saved cake "
            "stay NaN. Then in Analysis → Background set 'Background source = "
            "straightened' to fit de-waved rings. Lower radial resolution than a "
            "proper re-reduction — prefer fixing the geometry."))
        _texture_btn = ttk.Button(
            ctrl, text="Analyze texture", command=self._run_texture_job)
        _texture_btn.pack(side="left", padx=(6, 0))
        _ToolTip(
            _texture_btn,
            "Measure azimuthal intensity variation on the strongest saved rings. "
            "Requires 2D cakes and stores the results with the reduced data.",
        )
        self._straighten_status = ttk.Label(ctrl, text="", foreground=MUTED)
        self._straighten_status.pack(side="left", padx=12)
        paned = ttk.PanedWindow(frame, orient="horizontal")
        paned.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(3, weight=1)
        frame.columnconfigure(0, weight=1)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=1)
        self.review_text = tk.Text(left, bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                                   state="disabled", font=("TkFixedFont", 9), width=58)
        self.review_text.pack(fill="both", expand=True)
        self.review_plot_frame = right
        ttk.Label(right, text="Inspect a reduced .h5 to plot sample patterns.",
                  foreground=MUTED).pack(anchor="center", expand=True)
        # Only auto-inspect at startup if the file still exists (it may have
        # been deleted since last session — don't pop an error dialog on launch).
        _h5 = self.config.get("reduced_h5_file", "")
        if _h5 and Path(_h5).is_file():
            self.root.after(250, self.inspect_h5_clicked)

    def inspect_h5_clicked(self):
        self.pull_vars()
        path = str(self.config.get("reduced_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.messagebox.showerror("Review", "Select a reduced .h5 file first.")
            return
        from .review import inspect_reduction, structure_report
        self.log(f"Inspecting reduced HDF5: {path}")
        try:
            review = inspect_reduction(path)
        except Exception as e:
            self.messagebox.showerror("Review failed", repr(e))
            return
        self.review_text.configure(state="normal")
        self.review_text.delete("1.0", "end")
        self.review_text.insert("end", structure_report(review) + "\n")
        self.review_text.configure(state="disabled")
        self._render_review_plots(review)
        self.save_config(silent=True)

    def _on_review_overlay_toggle(self):
        """Re-render the loaded review when the overlay toggle flips."""
        self.config["review_overlay"] = bool(self._review_overlay.get())
        review = getattr(self, "_last_review", None)
        if review is not None:
            self._render_review_plots(review)

    def _run_texture_job(self):
        """Measure texture on saved cakes without blocking the Tk event loop."""
        if getattr(self, "_straighten_busy", False):
            self.messagebox.showinfo("Busy", "Another reduced-data tool is already running.")
            return
        self.pull_vars()
        path = str(self.config.get("reduced_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.messagebox.showerror("Texture analysis", "Select a reduced data file first.")
            return
        from tkinter import simpledialog
        n_rings = simpledialog.askinteger(
            "Texture analysis", "Strongest rings per frame:",
            initialvalue=3, minvalue=1, maxvalue=20, parent=self.root)
        if n_rings is None:
            return
        self._straighten_busy = True
        self._straighten_status.configure(text="Analyzing texture …", foreground=MUTED)
        box: "Dict[str, Any]" = {}

        def _work():
            try:
                from .texture import run_texture
                box["result"] = run_texture(path, n_rings=n_rings)
            except Exception as exc:
                box["error"] = str(exc)

        thread = threading.Thread(target=_work, daemon=True)
        thread.start()

        def _poll():
            if thread.is_alive():
                self.root.after(200, _poll)
                return
            self._straighten_busy = False
            if box.get("error"):
                self._straighten_status.configure(text=box["error"], foreground=WARN)
                self.log(f"Texture analysis failed: {box['error']}", "ERROR")
                return
            result = box.get("result") or {}
            status = f"texture: {result.get('n_cakes', 0)} cakes, {n_rings} rings each"
            self._straighten_status.configure(text=status, foreground=MUTED)
            self.log(f"Texture analysis complete: {status}")
            self.inspect_h5_clicked()

        self.root.after(200, _poll)

    def _run_straighten_job(self, kind: str):
        """Run the cake-waviness diagnosis or the straightened-1D rescue in a
        worker thread; results land in the log + review text. ``kind`` is
        "diagnose" or "straighten". Widgets are only touched from the Tk
        thread (a main-thread poll watches the worker)."""
        if getattr(self, "_straighten_busy", False):
            self.messagebox.showinfo("Busy", "A waviness job is already running.")
            return
        self.pull_vars()
        path = str(self.config.get("reduced_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.messagebox.showerror("Waviness", "Select a reduced .h5 file first.")
            return
        self._straighten_busy = True
        label = "Diagnosing waviness" if kind == "diagnose" else "Straightening"
        self._straighten_status.configure(text=f"{label} …")
        self.log(f"{label}: {path}")
        box: "Dict[str, Any]" = {}

        def _work():
            try:
                from .straighten import diagnose_reduced, straighten_reduced
                box["result"] = (diagnose_reduced(path) if kind == "diagnose"
                                 else straighten_reduced(path))
            except Exception as e:
                box["error"] = repr(e)

        t = threading.Thread(target=_work, daemon=True)
        t.start()

        def _poll():
            if t.is_alive():
                self.root.after(200, _poll)
                return
            self._straighten_busy = False
            err = box.get("error")
            res = box.get("result") or {}
            if err is None and kind == "diagnose" and not res.get("ok", True):
                err = res.get("error", "diagnosis failed")
            if err:
                self._straighten_status.configure(text=err, foreground=WARN)
                self.log(f"Waviness job failed: {err}", "ERROR")
                return
            lines = [f"=== {label} ==="]
            if kind == "diagnose":
                summ = res.get("summary") or {}
                unit = res.get("unit", "")
                if not summ:
                    lines.append(f"{res.get('n_cakes', 0)} cake(s) examined; "
                                 "no ring fit succeeded.")
                    status = "diagnosis: no usable rings"
                else:
                    lines.append(f"{res.get('n_cakes', 0)} cake(s) examined.")
                    lines.append(f"median waviness A1 = {summ['A1_median']:.4g} {unit} "
                                 f"(A2 = {summ['A2_median']:.4g})")
                    lines.append(f"implied 1D doublet splitting ≈ "
                                 f"{summ['doublet_splitting']:.4g} {unit}")
                    if "offset_mm" in summ:
                        lines.append(
                            f"implied transverse sample offset ≈ "
                            f"{summ['offset_mm']:.3f} mm at "
                            f"{summ['distance_mm']:.1f} mm distance")
                        lines.append("Fix: re-refine the geometry on a sample ring "
                                     "and re-reduce; 'Write straightened 1D' is the "
                                     "rescue for data you can't re-reduce.")
                    status = (f"A1 = {summ['A1_median']:.4g} {unit}; "
                              f"doublet ≈ {summ['doublet_splitting']:.4g}")
            else:
                lines.append(f"{res.get('n_straightened', 0)}/{res.get('n_frames', 0)} "
                             "frames written to /patterns/intensity_straightened")
                a1 = res.get("A1_median")
                if a1 is not None:
                    lines.append(f"median waviness A1 = {a1:.4g}")
                status = (f"straightened {res.get('n_straightened', 0)}"
                          f"/{res.get('n_frames', 0)} frames")
            self._straighten_status.configure(text=status, foreground=MUTED)
            for ln in lines:
                self.log(ln)
            try:
                self.review_text.configure(state="normal")
                self.review_text.insert("end", "\n" + "\n".join(lines) + "\n")
                self.review_text.see("end")
                self.review_text.configure(state="disabled")
            except self.tk.TclError:
                pass

        self.root.after(200, _poll)

    @staticmethod
    def _style_review_ax(ax):
        ax.set_facecolor(BG2)
        ax.tick_params(colors=FG, which="both", labelsize=8)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        for s in ax.spines.values():
            s.set_edgecolor(FG)

    def _draw_review_cake(self, ax, review):
        import numpy as np
        cake = np.asarray(review["cake"], dtype=float)
        cr = review.get("cake_radial")
        caz = review.get("cake_azimuthal")
        extent = None
        if cr is not None and caz is not None:
            extent = [float(np.min(cr)), float(np.max(cr)),
                      float(np.min(caz)), float(np.max(caz))]
        cc = cake.copy()
        cc[cc <= 0] = np.nan
        vmin = np.nanpercentile(cc, 5) if np.any(np.isfinite(cc)) else None
        vmax = np.nanpercentile(cc, 99) if np.any(np.isfinite(cc)) else None
        ax.imshow(cc, aspect="auto", origin="lower", cmap="magma",
                  extent=extent, vmin=vmin, vmax=vmax)
        fi = review.get("cake_frame_index")
        ax.set_title(f"Cake (frame {fi})" if fi is not None else "Cake")
        ax.set_xlabel(review.get("unit") or "radial")
        ax.set_ylabel("azimuth (deg)")
        self._style_review_ax(ax)

    def _render_review_plots(self, review: dict):
        # Remember the review so the overlay toggle can re-render without a
        # re-inspect.
        self._last_review = review
        # Close the previous figure before discarding its canvas, else each
        # re-inspect leaks a matplotlib Figure.
        prev = getattr(self, "_review_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._review_fig = None
        for w in self.review_plot_frame.winfo_children():
            w.destroy()
        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
        except Exception as e:
            self.ttk.Label(self.review_plot_frame, text=f"matplotlib unavailable: {e}",
                           foreground=WARN).pack(anchor="center", expand=True)
            return
        import numpy as np

        patterns = review.get("patterns", [])
        cake_present = bool(review.get("cake_present") and review.get("cake") is not None)
        radial = review.get("radial")
        x = np.asarray(radial) if radial is not None else None
        unit = review.get("unit") or "radial bin"
        overlay = bool(self._review_overlay.get()) if hasattr(self, "_review_overlay") else False

        if overlay or not patterns:
            # Overlaid view: all sample patterns on a shared axis (good for
            # comparison). Fully resizable + zoomable via the nav toolbar.
            nrows = 1 + (1 if cake_present else 0)
            fig = Figure(figsize=(5.5, 5.6), dpi=100, layout="constrained")
            self._review_fig = fig
            fig.patch.set_facecolor(BG)
            ax1 = fig.add_subplot(nrows, 1, 1)
            for pr in patterns:
                y = np.asarray(pr["intensity"], dtype=float)
                if x is not None and x.shape == y.shape:
                    ax1.plot(x, y, lw=0.8, alpha=0.7)
                else:
                    ax1.plot(y, lw=0.8, alpha=0.7)
            ax1.set_title(f"{len(patterns)} sample patterns (overlaid)")
            ax1.set_xlabel(unit)
            ax1.set_ylabel("intensity")
            self._style_review_ax(ax1)
            if cake_present:
                self._draw_review_cake(fig.add_subplot(nrows, 1, 2), review)
            self._embed_review_figure(fig, scroll=False)
            return

        # Separated view (default): one sample pattern per subplot, each with its
        # own x-axis, stacked in a vertically-scrollable strip so individual
        # features are easy to read. Cake (if any) goes at the bottom.
        n = len(patterns)
        rows = n + (1 if cake_present else 0)
        per_in = 1.7
        fig = Figure(figsize=(5.5, max(2.2, per_in * rows)), dpi=100, layout="constrained")
        self._review_fig = fig
        fig.patch.set_facecolor(BG)
        axes = fig.subplots(rows, 1, squeeze=False)[:, 0]
        for k, pr in enumerate(patterns):
            ax = axes[k]
            y = np.asarray(pr["intensity"], dtype=float)
            if x is not None and x.shape == y.shape:
                ax.plot(x, y, lw=0.9, color=ACCENT2)
            else:
                ax.plot(y, lw=0.9, color=ACCENT2)
            ax.set_title(pr.get("name") or f"frame {pr.get('index')}", fontsize=9)
            ax.set_ylabel("intensity", fontsize=8)
            ax.set_xlabel(unit, fontsize=8)
            self._style_review_ax(ax)
        if cake_present:
            self._draw_review_cake(axes[n], review)
        self._embed_review_figure(fig, scroll=True)

    def _embed_review_figure(self, fig, scroll: bool):
        """Embed the review figure in the right pane.

        ``scroll=False`` fills the pane and tracks resizes (overlay view).
        ``scroll=True`` keeps the figure's tall pixel height and wraps it in a
        vertically-scrollable canvas (separated view); the width still tracks
        the pane so each subplot stays readable.
        """
        tk, ttk = self.tk, self.ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        parent = self.review_plot_frame
        if not scroll:
            canvas = embed_figure(
                parent,
                fig,
                self.root,
                toolbar_factory=self._add_review_toolbar,
            )
            self._review_canvas = canvas
            return

        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)
        sc = tk.Canvas(container, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        sc.pack(side="left", fill="both", expand=True)
        canvas = FigureCanvasTkAgg(fig, master=sc)
        widget = canvas.get_tk_widget()
        fig_h_px = int(fig.get_figheight() * fig.get_dpi())
        widget.configure(width=10, height=fig_h_px)
        win = sc.create_window((0, 0), window=widget, anchor="nw")

        def _on_sc_config(event):
            # Width tracks the viewport; height stays tall so the strip scrolls.
            sc.itemconfigure(win, width=event.width)
            sc.configure(scrollregion=(0, 0, event.width, fig_h_px))

        sc.bind("<Configure>", _on_sc_config)

        def _wheel(event):
            up = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
            sc.yview_scroll(-1 if up else 1, "units")

        sc.bind("<Enter>", lambda e: (sc.bind_all("<MouseWheel>", _wheel),
                                      sc.bind_all("<Button-4>", _wheel),
                                      sc.bind_all("<Button-5>", _wheel)))
        sc.bind("<Leave>", lambda e: (sc.unbind_all("<MouseWheel>"),
                                      sc.unbind_all("<Button-4>"),
                                      sc.unbind_all("<Button-5>")))
        make_canvas_responsive(canvas, self.root, fixed_height_px=fig_h_px)
        self._review_canvas = canvas

    def _add_review_toolbar(self, canvas, parent):
        """Dark-styled matplotlib nav toolbar (zoom/pan/home/save) under the
        overlay canvas. Degrades silently if unavailable."""
        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
            tb.update()
            # matplotlib's toolbar glyphs are dark, so a near-black button fill
            # hides them. Give buttons a light fill; keep the frame + coordinate
            # label on the dark palette.
            try:
                tb.configure(background=BG)
                for child in tb.winfo_children():
                    cls = child.winfo_class()
                    try:
                        if cls in ("Button", "Checkbutton", "Radiobutton"):
                            child.configure(background=FG, activebackground=ACCENT,
                                            highlightbackground=BG, relief="flat")
                        elif cls == "Label":
                            child.configure(background=BG, foreground=FG)
                        else:
                            child.configure(background=BG)
                    except Exception:
                        pass
            except Exception:
                pass
            tb.pack(side="bottom", fill="x")
            return tb
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Tab 6 — Gallery (per-frame cake/1D matrix with click-to-exclude)
    # ------------------------------------------------------------------

    _CELL_W = 150   # px per gallery cell (thumbnail is 240x200 scaled to fit)
    _CELL_H = 168

    def _tab_gallery(self, frame):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(frame)
        top.pack(fill="x")
        ttk.Button(top, text="Load gallery", command=self.load_gallery).pack(side="left", padx=4, pady=4)
        self.gallery_status = ttk.Label(top, text="No gallery loaded.", foreground=MUTED)
        self.gallery_status.pack(side="left", padx=12)
        ttk.Label(top, text="Click a frame to exclude/include it (saved to the .h5).",
                  foreground=MUTED).pack(side="right", padx=8)

        # Scrollable canvas holding the cell grid.
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)
        self.gallery_canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.gallery_canvas.yview)
        self.gallery_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.gallery_canvas.pack(side="left", fill="both", expand=True)
        self.gallery_inner = ttk.Frame(self.gallery_canvas)
        self._gallery_window = self.gallery_canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")
        self.gallery_inner.bind("<Configure>",
                                lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all")))
        self.gallery_canvas.bind("<Configure>", self._on_gallery_configure)
        # Mouse wheel + lazy thumbnail loading on scroll.
        self.gallery_canvas.bind("<Enter>", lambda e: self._bind_gallery_wheel(True))
        self.gallery_canvas.bind("<Leave>", lambda e: self._bind_gallery_wheel(False))

        # State
        self._gallery_frames = []        # list of frame dicts from gallery_frames()
        self._gallery_cells = {}         # index -> {frame, label, img_label, loaded}
        self._gallery_photos = {}        # index -> PhotoImage (keep refs alive)
        self._gallery_cols = 1
        self._gallery_lazy_after = None

    def _bind_gallery_wheel(self, on: bool):
        c = self.gallery_canvas
        if on:
            c.bind_all("<MouseWheel>", self._gallery_wheel)
            c.bind_all("<Button-4>", self._gallery_wheel)
            c.bind_all("<Button-5>", self._gallery_wheel)
        else:
            c.unbind_all("<MouseWheel>")
            c.unbind_all("<Button-4>")
            c.unbind_all("<Button-5>")

    def _gallery_wheel(self, event):
        import sys as _sys
        if getattr(event, "num", None) == 4:
            self.gallery_canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.gallery_canvas.yview_scroll(1, "units")
        elif _sys.platform == "darwin":
            self.gallery_canvas.yview_scroll(int(-1 * event.delta), "units")
        else:
            self.gallery_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._schedule_lazy_load()

    def load_gallery(self):
        self.pull_vars()
        path = str(self.config.get("reduced_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.messagebox.showerror("Gallery", "Select a reduced .h5 on the Review tab first.")
            return
        from .review import gallery_frames
        g = gallery_frames(path)
        if not g["ok_to_read"]:
            self.messagebox.showerror("Gallery", g.get("error") or "Could not read gallery.")
            return
        self._gallery_h5 = path
        self._gallery_frames = g["frames"]
        self._has_thumbs = any(f["thumb"] for f in self._gallery_frames)
        self.log(f"Gallery: {g['n_frames']} frames" + ("" if self._has_thumbs else " (no thumbnails — run with thumbnails on for previews)"))
        self._build_gallery_cells()
        self._update_gallery_status()

    def _build_gallery_cells(self):
        ttk, tk = self.ttk, self.tk
        for w in self.gallery_inner.winfo_children():
            w.destroy()
        self._gallery_cells.clear()
        self._gallery_photos.clear()
        cols = max(1, self._gallery_cols)
        for n, fr in enumerate(self._gallery_frames):
            idx = fr["index"]
            cell = ttk.Frame(self.gallery_inner, relief="solid", borderwidth=1, width=self._CELL_W, height=self._CELL_H)
            cell.grid(row=n // cols, column=n % cols, padx=3, pady=3, sticky="nsew")
            cell.grid_propagate(False)
            name = Path(fr["filename"]).name
            shown = name if len(name) <= 20 else name[:9] + "…" + name[-9:]
            lbl = ttk.Label(cell, text=shown, font=("TkDefaultFont", 7))
            lbl.pack(side="top", fill="x")
            img_label = ttk.Label(cell, text="…", anchor="center")
            img_label.pack(side="top", fill="both", expand=True)
            for w in (cell, lbl, img_label):
                w.bind("<Button-1>", lambda e, _i=idx: self._toggle_exclude(_i))
            self._gallery_cells[idx] = {"frame": cell, "label": lbl, "img": img_label,
                                        "loaded": False, "name": shown}
            self._apply_cell_style(idx)
        self.gallery_inner.update_idletasks()
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
        self._schedule_lazy_load()

    def _apply_cell_style(self, idx: int):
        """Color a cell by its excluded/failed state."""
        cell = self._gallery_cells.get(idx)
        fr = next((f for f in self._gallery_frames if f["index"] == idx), None)
        if not cell or fr is None:
            return
        if fr["excluded"]:
            cell["label"].configure(foreground=WARN)
            cell["frame"].configure(relief="solid", borderwidth=2)
        elif not fr["ok"]:
            cell["label"].configure(foreground="#fab387")  # failed frame
        else:
            cell["label"].configure(foreground=FG)
            cell["frame"].configure(relief="solid", borderwidth=1)

    def _on_gallery_configure(self, event):
        self.gallery_canvas.itemconfigure(self._gallery_window, width=event.width)
        new_cols = max(1, event.width // (self._CELL_W + 6))
        if new_cols != self._gallery_cols:
            self._gallery_cols = new_cols
            if self._gallery_frames:
                self._build_gallery_cells()
        else:
            self._schedule_lazy_load()

    def _schedule_lazy_load(self):
        if self._gallery_lazy_after is not None:
            try:
                self.root.after_cancel(self._gallery_lazy_after)
            except Exception:
                pass
        self._gallery_lazy_after = self.root.after(120, self._load_visible_thumbs)

    def _load_visible_thumbs(self):
        self._gallery_lazy_after = None
        if not self._gallery_cells or not getattr(self, "_has_thumbs", False):
            return
        try:
            from PIL import Image, ImageTk  # type: ignore
        except Exception:
            return
        c = self.gallery_canvas
        top = c.canvasy(0)
        bot = top + c.winfo_height()
        for idx, cell in self._gallery_cells.items():
            if cell["loaded"]:
                continue
            w = cell["frame"]
            try:
                y = w.winfo_y()
            except Exception:
                continue
            if y + self._CELL_H < top - 200 or y > bot + 200:
                continue  # not near the viewport
            fr = next((f for f in self._gallery_frames if f["index"] == idx), None)
            if fr is None or not fr["thumb"] or not Path(fr["thumb"]).is_file():
                cell["img"].configure(text="(no preview)")
                cell["loaded"] = True
                continue
            try:
                im = Image.open(fr["thumb"])
                im.thumbnail((self._CELL_W - 8, self._CELL_H - 26))
                photo = ImageTk.PhotoImage(im)
                self._gallery_photos[idx] = photo
                cell["img"].configure(image=photo, text="")
                cell["loaded"] = True
            except Exception:
                cell["img"].configure(text="(load error)")
                cell["loaded"] = True

    def _toggle_exclude(self, idx: int):
        fr = next((f for f in self._gallery_frames if f["index"] == idx), None)
        if fr is None or not getattr(self, "_gallery_h5", None):
            return
        new_state = not fr["excluded"]
        from .review import set_excluded
        try:
            set_excluded(self._gallery_h5, [idx], new_state)
            fr["excluded"] = new_state
            self._apply_cell_style(idx)
            self._update_gallery_status()
        except Exception as e:
            self.messagebox.showerror("Exclude failed", repr(e))

    def _update_gallery_status(self):
        n = len(self._gallery_frames)
        nx = sum(1 for f in self._gallery_frames if f["excluded"])
        nf = sum(1 for f in self._gallery_frames if not f["ok"])
        self.gallery_status.configure(
            text=f"{n} frames   {nx} excluded   {nf} failed"
            + ("" if getattr(self, "_has_thumbs", False) else "   (no thumbnails)"))

    # ------------------------------------------------------------------

    def register_menus(self, menubar, file_menu, tools_menu=None, help_menu=None):
        """Add reduction-stage commands to shared menus (unified host or standalone)."""
        file_menu.add_command(label="Open reduction config...", command=self._open_config_dialog)

    def _open_config_dialog(self):
        try:
            path = self.filedialog.askopenfilename(
                title="Open reduction config",
                filetypes=[("JSON config", "*.json"), ("All files", "*.*")])
            if not path:
                return
            self.messagebox.showinfo(
                "Switching config requires relaunch",
                "To use the selected reduction config, relaunch with:\n\n"
                f'python -m seriesxrd.reduce.run_gui --config "{path}"\n\nSelected:\n{path}')
        except Exception as e:
            self.log(f"Open config dialog error: {e}", "WARN")

    def confirm_shutdown(self) -> bool:
        """Return whether the pane may close without changing pane state."""
        if getattr(self, "_straighten_busy", False):
            self.messagebox.showinfo(
                "Processing in progress",
                "A reduced-data tool is still running. Wait for it to finish before closing.",
            )
            return False
        if self._run_proc is not None and self._run_proc.poll() is None:
            if not self.messagebox.askyesno(
                    "Reduction running", "Reduction is still running. Stop it and close?"):
                return False
        wp = getattr(self, "_watch_proc", None)
        if wp is not None and wp.poll() is None:
            if not self.messagebox.askyesno(
                    "Live processing", "Live processing is still running. Stop it and close?"):
                return False
        return True

    def shutdown(self, confirm: bool = True) -> bool:
        """Save and tear down. Returns False if the user cancelled."""
        if confirm and not self.confirm_shutdown():
            return False
        if self._run_proc is not None and self._run_proc.poll() is None:
            terminate_process_tree(self._run_proc)
        wp = getattr(self, "_watch_proc", None)
        if wp is not None and wp.poll() is None:
            wp.terminate()   # SIGTERM → the watcher flushes and exits
        self._closing = True  # stop the log-drain poller from rescheduling
        self.save_config(silent=True)
        return True

    def on_close(self):
        if not self.shutdown(confirm=True):
            return
        if self._owns_root:
            self.root.destroy()


def make_reduce_pane(parent_frame, config_path: "str | Path") -> "ReductionApp":
    """Construct ReductionApp embedded in a parent frame (for the unified app)."""
    return ReductionApp(config_path, parent=parent_frame)


def run_app(config_path: "str | Path") -> int:
    from ..guikit.dpi import enable_hi_dpi
    enable_hi_dpi()
    app = ReductionApp(config_path)
    assert app._owns_root, "run_app is the standalone entry point and must own the root"
    app.root.mainloop()
    return 0
