"""Tabbed Tkinter GUI for the analysis stage (Step 1 background + Step 2 peaks).

Workflow left-to-right across tabs:
    1 Input      — pick the reduced HDF5, inspect its structure, see the analysis output
    2 Background — SNIP baseline + diamond-spot residual parameters
    3 Peaks      — pseudo-Voigt fitting parameters
    4 Run        — launch the crash-isolated worker, watch progress + log
    5 Review     — single-frame QC: traces + fitted peaks + contamination curve
    6 Heatmap    — scatter of peak positions across all frames (Step-3 precursor)
    7 Phases     — reference-phase library: bundled + user phases, import CIFs, toggle candidates
    8 Identify   — Step 3a: deterministic EOS phase matching, per-frame pressure + confidence

Same supervision model as reduce/gui.py: heavy computation runs in
analysis/worker.py as a subprocess; this process never imports numpy/h5py
directly except lazily inside plotting methods.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import queue
import subprocess
import sys
import threading

from ..core.config import (
    TOOL_NAME, read_json, write_json, ensure_dir,
    now_iso, now_timestamp, output_base,
)
from ..core.naming import next_available_path
from ..guikit.theme import (
    BG, BG2, FG, ACCENT, ACCENT2, WARN, MUTED, ENTRY_BG,
    CLR_RAW, CLR_MSKD, CLR_SMTH, CLR_DIFF,
)
from ..guikit.tkstyle import apply_dark_theme
from ..guikit.tooltip import ToolTip as _ToolTip


def _tk_imports():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    return tk, ttk, filedialog, messagebox


HELP: Dict[str, str] = {
    # Input / output
    "reduced_h5_file": (
        "A reduced_*.h5 produced by the reduction stage. Must contain "
        "intensity_robust (azimuthal-median pattern) for Step 1."
    ),
    "analysis_h5_file": (
        "Output analysis HDF5. Leave blank to auto-place beside the reduced "
        "file as <stem>_analysis.h5."
    ),
    # Run scope
    "run_step1": "Run Step 1: SNIP baseline estimation + diamond-spot residual extraction.",
    "run_step2": "Run Step 2: pseudo-Voigt peak fitting on the clean (baseline-subtracted) patterns.",
    "run_step3": (
        "Run Step 3a: match the fitted peaks against the enabled candidate phases by fitting "
        "each phase's Birch–Murnaghan EOS — gives a per-frame pressure and match confidence "
        "per phase."
    ),
    # Step 3a
    "p_min": "Pressure search range (GPa) for the EOS fit.",
    "p_max": "Pressure search range (GPa) for the EOS fit.",
    "rel_tol": (
        "Peak-match tolerance as a fraction of d-spacing (e.g. 0.01 = 1%). "
        "Looser = more tolerant matching, fuzzier pressure."
    ),
    "identify_wavelength": (
        "X-ray wavelength (Å). Needed only for a 2θ axis; leave blank to auto-read it "
        "from the reduced file's PONI. Not needed for q-axis data."
    ),
    # Step 1
    "max_half_window": (
        "Widest feature (in bins) treated as background; ~1.5-2x the broadest "
        "Bragg peak half-width. Wider values = more background. Too wide erodes "
        "real broad peaks irreversibly — conservative is safer."
    ),
    "n_passes": (
        "Number of SNIP iteration passes. 1 is almost always sufficient; "
        "more passes make the baseline slightly smoother at a small cost."
    ),
    "use_lls": (
        "Apply Log-Log-Sqrt (LLS) transform before SNIP to compress dynamic "
        "range. Strongly recommended for XRD — suppresses baseline overshoot "
        "under sharp intense peaks."
    ),
    "contamination_threshold": (
        "Flag frames whose diamond contamination score exceeds this value. "
        "Leave blank to skip flagging. The score is the sum of positive "
        "spot_residual bins per frame."
    ),
    # Step 2
    "min_snr": (
        "Minimum signal-to-noise ratio (peak height / MAD noise floor) for a "
        "peak to be accepted. Lower = more peaks, more noise hits. Default 5."
    ),
    "window_factor": (
        "Fit-window half-width as a multiple of the initial FWHM estimate. "
        "Wider windows capture more baseline; narrow windows are faster. Default 3."
    ),
    "max_chi2": (
        "Maximum reduced chi-square for a fitted peak to be accepted as 'good'. "
        "Higher = more permissive. Default 25 is already generous; tighten for "
        "cleaner peak maps."
    ),
    "propagate_seeds": (
        "Seed peak detection for frame k+1 from good centers in frame k. "
        "Recommended — keeps reflections continuous as the lattice compresses "
        "under pressure."
    ),
}


class AnalysisApp:
    def __init__(self, config_path: "str | Path", parent=None):
        tk, ttk, filedialog, messagebox = _tk_imports()
        self.tk, self.ttk, self.filedialog, self.messagebox = (
            tk, ttk, filedialog, messagebox
        )
        self.config_path = Path(config_path).expanduser().resolve()
        self.config: Dict[str, Any] = read_json(self.config_path)
        self.config.setdefault("session_config_path", str(self.config_path))
        if parent is None:
            self._owns_root = True
            self.root = tk.Tk()
            self.root.title(f"{TOOL_NAME} Analysis")
            self.root.geometry("1180x780")
            self.root.minsize(960, 640)
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        else:
            self._owns_root = False
            self.root = parent.winfo_toplevel()
        self._embed_parent = parent  # None when standalone, ttk.Frame when embedded

        self.vars: Dict[str, Any] = {}
        self._run_proc: "subprocess.Popen | None" = None
        # Thread-safe logging: worker threads push lines here; a main-thread
        # poller drains them into the Text widget.
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        # Thread-safe run events: the worker thread pushes ("progress"|"done"|
        # "error", ...) tuples here and the main-thread poller dispatches them.
        # Tkinter is not thread-safe, so the worker must NEVER touch widgets (or
        # even root.after) directly — that can deadlock the event loop.
        self._event_queue: "queue.Queue[tuple]" = queue.Queue()
        # History buffer so lines aren't lost before the console window opens.
        self._log_history: "list[str]" = []

        # State for status bar
        self._frame_count: int = 0
        self._worker_status: str = "idle"

        # Review tab state
        self._review_nframes: int = 0
        self._review_contamination = None  # numpy array or None
        self._review_after: "int | None" = None  # debounce scheduler id

        self._build_gui()
        self._drain_log_queue()
        self.log("GUI initialized")
        self.save_config(silent=True)
        self._update_status_bar()

        # Auto-inspect on startup if the reduced file already exists.
        _h5 = self.config.get("reduced_h5_file", "")
        if _h5 and Path(_h5).is_file():
            self.root.after(300, self.inspect_input_clicked)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_gui(self):
        tk, ttk = self.tk, self.ttk
        if self._owns_root:
            apply_dark_theme(self.root, ttk)
        _container = self._embed_parent if self._embed_parent is not None else self.root
        outer = ttk.Frame(_container, padding=6)
        outer.pack(fill="both", expand=True)

        topbar = ttk.Frame(outer)
        topbar.pack(fill="x", pady=(0, 6))
        ttk.Label(
            topbar, text=f"{TOOL_NAME} Analysis",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(side="left")
        ttk.Button(
            topbar, text="Open Console Logs", command=self.open_console_logs,
        ).pack(side="right", padx=4)

        # Status bar carved out before the notebook so it's always visible.
        self._status_bar_frame = ttk.Frame(outer, relief="sunken")
        self._status_bar_frame.pack(side="bottom", fill="x", pady=(2, 0))

        self.nb = ttk.Notebook(outer)
        self.nb.pack(fill="both", expand=True)
        self.tabs: Dict[str, Any] = {}
        for name, builder in [
            ("1 Input",      self._tab_input),
            ("2 Background", self._tab_background),
            ("3 Peaks",      self._tab_peaks),
            ("4 Run",        self._tab_run),
            ("5 Review",     self._tab_review),
            ("6 Heatmap",    self._tab_heatmap),
            ("7 Phases",     self._tab_phases),
            ("8 Identify",   self._tab_identify),
            ("9 Pattern map", self._tab_patternmap),
        ]:
            frame = ttk.Frame(self.nb, padding=10)
            builder(frame)
            self.nb.add(frame, text=name)
            self.tabs[name] = frame

        self._build_status_bar()

    def _build_status_bar(self):
        ttk = self.ttk
        bar = self._status_bar_frame
        self.status_session = ttk.Label(bar, text="", foreground=MUTED, anchor="w")
        self.status_session.pack(side="left", padx=(6, 12))
        self.status_input = ttk.Label(bar, text="input: none", foreground=MUTED, anchor="w")
        self.status_input.pack(side="left", padx=(0, 12))
        self.status_frames = ttk.Label(bar, text="frames: —", foreground=MUTED, anchor="w")
        self.status_frames.pack(side="left", padx=(0, 12))
        self.status_worker = ttk.Label(bar, text="idle", foreground=MUTED, anchor="e")
        self.status_worker.pack(side="right", padx=6)

    def _update_status_bar(self):
        try:
            session = self.config.get("session_name", "")
            if hasattr(self, "status_session"):
                self.status_session.configure(
                    text=f"session: {session}" if session else "session: (unnamed)")
            if hasattr(self, "status_input"):
                h5 = self.config.get("reduced_h5_file", "")
                bname = Path(h5).name if h5 else "none"
                self.status_input.configure(text=f"input: {bname}")
            if hasattr(self, "status_frames"):
                fc = self._frame_count
                self.status_frames.configure(
                    text=f"frames: {fc}" if fc > 0 else "frames: —")
            if hasattr(self, "status_worker"):
                self.status_worker.configure(text=self._worker_status)
        except Exception:
            pass

    # -- shared small widgets -----------------------------------------------

    def field(self, parent, key, label, browse=None, row=None, width=80):
        """Entry field bound to a config key, with optional Browse button."""
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
            ttk.Button(
                parent, text="Browse",
                command=lambda: self.browse_into(key, browse),
            ).grid(row=row, column=2, padx=4)
        txt = HELP.get(key, "")
        if txt:
            _ToolTip(lbl, txt)
            _ToolTip(entry, txt)
        parent.columnconfigure(1, weight=1)

    def checkbox(self, parent, key, label, row=None):
        """Checkbox bound to a boolean config key."""
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
    # Thread-safe logging
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "INFO"):
        line = f"[{now_iso()}] [{level}] {message}"
        print(line, flush=True)
        if threading.current_thread() is not threading.main_thread():
            self._log_queue.put(line)
            return
        self._insert_log_line(line)

    def _insert_log_line(self, line: str):
        self._log_history.append(line)
        if len(self._log_history) > 5000:
            self._log_history = self._log_history[-5000:]
        if hasattr(self, "log_text"):
            try:
                self.log_text.configure(state="normal")
                self.log_text.insert("end", line + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            except self.tk.TclError:
                pass
        if hasattr(self, "run_log_text"):
            try:
                self.run_log_text.configure(state="normal")
                self.run_log_text.insert("end", line + "\n")
                self.run_log_text.see("end")
                self.run_log_text.configure(state="disabled")
            except self.tk.TclError:
                pass

    def _drain_log_queue(self):
        """Recurring main-thread poller: flush queued lines into the widget."""
        if getattr(self, "_closing", False):
            return
        try:
            while True:
                line = self._log_queue.get_nowait()
                self._insert_log_line(line)
        except queue.Empty:
            pass
        # Dispatch run events on the main thread.
        try:
            while True:
                evt = self._event_queue.get_nowait()
                self._dispatch_run_event(evt)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    def _dispatch_run_event(self, evt: tuple):
        """Handle a worker run event on the main thread (see _worker_thread)."""
        kind = evt[0]
        try:
            if kind == "progress":
                self._update_progress(evt[1], evt[2], evt[3])
            elif kind == "done":
                self._run_done(evt[1], evt[2])
            elif kind == "error":
                self._run_error(evt[1])
        except Exception as e:  # never let a dispatch error wedge the poller
            self.log(f"run-event handler failed ({kind}): {e!r}", "WARN")

    # ------------------------------------------------------------------
    # Console log window
    # ------------------------------------------------------------------

    def open_console_logs(self):
        tk, ttk = self.tk, self.ttk
        try:
            if getattr(self, "_log_window", None) and self._log_window.winfo_exists():
                self._log_window.deiconify()
                self._log_window.lift()
                self._log_window.focus_set()
                return
        except self.tk.TclError:
            pass
        self._log_window = tk.Toplevel(self.root)
        self._log_window.title("Bulk-XRD Analysis Console Logs")
        self._log_window.geometry("900x420")
        self._log_window.configure(bg=BG)
        self.log_text = tk.Text(
            self._log_window, wrap="word", state="disabled",
            font=("TkFixedFont", 10), bg=BG2, fg=FG,
            insertbackground=FG, selectbackground=ACCENT,
        )
        scroll = ttk.Scrollbar(
            self._log_window, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_text.configure(state="normal")
        if self._log_history:
            self.log_text.insert("end", "\n".join(self._log_history) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self._log_window.protocol("WM_DELETE_WINDOW", self._hide_console_logs)

    def _hide_console_logs(self):
        if getattr(self, "_log_window", None):
            try:
                self._log_window.withdraw()
            except self.tk.TclError:
                pass

    # ------------------------------------------------------------------
    # Tab 1 — Input
    # ------------------------------------------------------------------

    def _tab_input(self, frame):
        tk, ttk = self.tk, self.ttk
        self.field(frame, "reduced_h5_file", "Reduced HDF5", browse="file", row=0)
        self.field(frame, "analysis_h5_file", "Analysis HDF5 (output)", browse="file", row=1)
        btns = ttk.Frame(frame)
        btns.grid(row=2, column=0, columnspan=3, sticky="w", padx=2, pady=6)
        ttk.Button(
            btns, text="Inspect input", command=self.inspect_input_clicked,
        ).pack(side="left", padx=2)

        # Warn if robust pattern is missing — analysis requires it.
        self._robust_warn_label = ttk.Label(btns, text="", foreground=WARN)
        self._robust_warn_label.pack(side="left", padx=12)

        self.input_text = tk.Text(
            frame, height=20, bg=BG2, fg=FG, insertbackground=FG,
            relief="flat", state="disabled", font=("TkFixedFont", 10),
        )
        self.input_text.grid(
            row=3, column=0, columnspan=3, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(3, weight=1)
        frame.columnconfigure(1, weight=1)

    def inspect_input_clicked(self):
        """Inspect the reduced HDF5 (and analysis HDF5 if present) and show reports."""
        self.pull_vars()
        reduced = str(self.config.get("reduced_h5_file", "") or "").strip()
        if not reduced or not Path(reduced).is_file():
            self.messagebox.showerror(
                "Input", "Select a reduced .h5 file first (1 Input tab).")
            return
        from ..reduce.review import inspect_reduction, structure_report as reduce_report
        self.log(f"Inspecting reduced HDF5: {reduced}")
        try:
            review_r = inspect_reduction(reduced)
        except Exception as e:
            self.messagebox.showerror("Inspect failed", repr(e))
            return

        lines = ["=== Reduced HDF5 ===", reduce_report(review_r)]

        # Warn about missing robust pattern.
        robust_ok = bool(review_r.get("robust_present"))
        if hasattr(self, "_robust_warn_label"):
            if not robust_ok:
                self._robust_warn_label.configure(
                    text="WARNING: robust pattern missing — re-run reduction with "
                         "robust_1d=True before analysis.")
            else:
                self._robust_warn_label.configure(text="")

        # Update frame count for status bar.
        nf = review_r.get("n_frames", 0)
        if nf:
            self._frame_count = int(nf)

        # If an analysis HDF5 already exists, append its summary too.
        analysis = str(self.config.get("analysis_h5_file", "") or "").strip()
        if analysis and Path(analysis).is_file():
            from .review import inspect_analysis, structure_report as analysis_report
            self.log(f"Inspecting analysis HDF5: {analysis}")
            try:
                review_a = inspect_analysis(analysis)
                lines += ["", "=== Analysis HDF5 ===", analysis_report(review_a)]
                if review_a.get("n_frames"):
                    self._frame_count = int(review_a["n_frames"])
            except Exception as e:
                lines += ["", f"[Could not inspect analysis HDF5: {e!r}]"]

        combined = "\n".join(lines)
        self.input_text.configure(state="normal")
        self.input_text.delete("1.0", "end")
        self.input_text.insert("end", combined + "\n")
        self.input_text.configure(state="disabled")
        self._update_status_bar()
        self.save_config(silent=True)

    def set_reduced(self, path: "str | Path") -> None:
        """Called by the host app to wire in the reduced file after reduction.

        Sets reduced_h5_file in config + its StringVar, saves, switches to the
        Input tab, and refreshes the input summary.
        """
        p = str(path or "").strip()
        if not p:
            return
        self.config["reduced_h5_file"] = p
        if "reduced_h5_file" in self.vars:
            self.vars["reduced_h5_file"].set(p)
        self.save_config(silent=True)
        self.log(f"Reduced HDF5 received: {p}")
        try:
            self.nb.select(self.tabs["1 Input"])
        except Exception:
            pass
        if Path(p).is_file():
            self.root.after(100, self.inspect_input_clicked)

    # ------------------------------------------------------------------
    # Tab 2 — Background
    # ------------------------------------------------------------------

    def _tab_background(self, frame):
        ttk = self.ttk
        self.checkbox(frame, "run_step1", "Run Step 1 — background separation", row=0)
        self.field(frame, "max_half_window", "Max half-window (bins)", row=2, width=14)
        self.field(frame, "n_passes", "SNIP passes", row=3, width=14)
        self.checkbox(frame, "use_lls", "Use LLS transform (Log-Log-Sqrt compression)", row=4)
        self.field(frame, "contamination_threshold",
                   "Contamination threshold (blank = off)", row=6, width=14)
        ttk.Label(
            frame,
            text=(
                "Step 1 separates background from powder signal.\n\n"
                "spot_residual = azimuthal mean − azimuthal median\n"
                "  Diamond single-crystal spots appear in the mean but not in the\n"
                "  median (< 50 % of azimuthal bins), so the difference isolates them.\n\n"
                "baseline = SNIP(robust)  — Statistics-sensitive Non-linear Iterative\n"
                "  Peak-clipping on the robust (median) pattern, optionally with LLS\n"
                "  transform for dynamic-range compression under intense peaks.\n\n"
                "clean = robust − baseline\n"
                "  This is what goes to peak fitting in Step 2."
            ),
            foreground=MUTED, justify="left", wraplength=640,
        ).grid(row=8, column=0, columnspan=3, sticky="w", padx=6, pady=(12, 4))

    # ------------------------------------------------------------------
    # Tab 3 — Peaks
    # ------------------------------------------------------------------

    def _tab_peaks(self, frame):
        ttk = self.ttk
        self.checkbox(frame, "run_step2", "Run Step 2 — pseudo-Voigt peak fitting", row=0)
        self.field(frame, "min_snr", "Min SNR", row=2, width=14)
        self.field(frame, "window_factor", "Window factor (× FWHM)", row=3, width=14)
        self.field(frame, "max_chi2", "Max reduced χ²", row=4, width=14)
        self.checkbox(frame, "propagate_seeds",
                      "Propagate peak seeds frame-to-frame", row=6)
        ttk.Label(
            frame,
            text=(
                "Step 2 fits pseudo-Voigt profiles to each background-subtracted\n"
                "clean pattern.\n\n"
                "Profile: A·(η·Lorentzian + (1−η)·Gaussian), both normalised to\n"
                "peak height A. η is the Lorentzian fraction fitted freely.\n\n"
                "Detection: scipy find_peaks + MAD noise floor (1.4826·median|x−median|).\n"
                "Peaks below min_snr are rejected.\n\n"
                "Seed propagation: good peak centers from frame k seed frame k+1,\n"
                "keeping reflections continuous as the lattice compresses under pressure.\n\n"
                "Rejection flags: LOW_AMP=1, BAD_CHI2=2, CENTER_DRIFT=4,\n"
                "WIDTH_BOUND=8, NO_CONVERGE=16."
            ),
            foreground=MUTED, justify="left", wraplength=640,
        ).grid(row=8, column=0, columnspan=3, sticky="w", padx=6, pady=(12, 4))

    # ------------------------------------------------------------------
    # Tab 4 — Run
    # ------------------------------------------------------------------

    def _tab_run(self, frame):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(frame)
        top.pack(fill="x")
        self.run_btn = ttk.Button(top, text="Run analysis", command=self.run_analysis)
        self.run_btn.pack(side="left", padx=4, pady=4)
        self.cancel_btn = ttk.Button(
            top, text="Cancel", command=self.cancel_analysis, state="disabled")
        self.cancel_btn.pack(side="left", padx=4, pady=4)
        ttk.Label(top, text="Workers:", foreground=MUTED).pack(side="left", padx=(16, 2))
        _w_var = tk.StringVar(value=str(self.config.get("num_workers", "0")))
        self.vars["num_workers"] = _w_var
        _w_entry = ttk.Entry(top, textvariable=_w_var, width=5)
        _w_entry.pack(side="left", padx=2)
        _ToolTip(_w_entry, "Parallel worker processes for all steps. "
                           "0 = auto (CPU count − 1), 1 = serial.")
        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=4, pady=6)
        self.progress_label = ttk.Label(frame, text="Idle", foreground=MUTED)
        self.progress_label.pack(anchor="w", padx=6)
        self.run_log_text = tk.Text(
            frame, bg=BG2, fg=FG, insertbackground=FG, relief="flat",
            state="disabled", font=("TkFixedFont", 9),
        )
        self.run_log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def run_analysis(self):
        if self._run_proc is not None:
            self.messagebox.showinfo("Busy", "An analysis is already running.")
            return
        self.save_config(silent=True)
        self.pull_vars()

        run_step1 = bool(self.config.get("run_step1", True))
        run_step2 = bool(self.config.get("run_step2", True))
        run_step3 = bool(self.config.get("run_step3", False))
        if not run_step1 and not run_step2 and not run_step3:
            self.messagebox.showerror(
                "Nothing to run",
                "Enable at least one of 'Run Step 1', 'Run Step 2', or 'Run Step 3a' "
                "on the Background / Peaks / Identify tabs.")
            return

        if run_step1:
            reduced = str(self.config.get("reduced_h5_file", "") or "").strip()
            if not reduced or not Path(reduced).is_file():
                self.messagebox.showerror(
                    "Input missing",
                    "Step 1 requires a reduced HDF5.\n"
                    "Set it on the '1 Input' tab first.")
                return

        backend_dir = self.config.get(
            "backend_dir", str(Path(__file__).resolve().parents[1]))
        python_exe = Path(self.config.get("python_exe", sys.executable))
        logs_root = (
            self.config.get("logs_root", "")
            or str(output_base(self.config) / "logs")
        )
        ensure_dir(Path(logs_root))
        out_json = str(
            next_available_path(Path(logs_root) / f"analysis_{now_timestamp()}.json"))
        worker_script = str(Path(backend_dir) / "analysis" / "worker.py")
        if not Path(worker_script).is_file():
            self.messagebox.showerror(
                "Worker not found",
                f"Analysis worker script not found:\n{worker_script}\n\n"
                "Check 'backend_dir' in the session config.")
            return

        cmd = [
            str(python_exe), worker_script,
            "--config", str(self.config_path),
            "--output-json", out_json,
        ]
        self.log("Worker command: " + " ".join(cmd))
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(value=0)
        self.progress_label.configure(text="Starting worker ...")
        self._worker_status = "running"
        self._update_status_bar()

        def _worker_thread():
            try:
                proc = subprocess.Popen(
                    cmd, cwd=backend_dir,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                self._run_proc = proc
                assert proc.stdout is not None
                # Watchdog: once the process exits, close stdout so the reader
                # below can't hang on a pipe still held open by forked pool
                # workers / the multiprocessing resource tracker. Without this
                # the loop never sees EOF, _run_done never fires, the run stays
                # stuck at "running", and closing the app always warns.
                def _watch(p=proc):
                    p.wait()
                    try:
                        if p.stdout is not None:
                            p.stdout.close()
                    except Exception:
                        pass
                threading.Thread(target=_watch, daemon=True).start()
                try:
                    for line in proc.stdout:
                        line = line.rstrip()
                        parts = line.split()
                        if len(parts) == 3 and parts[0] in ("[ANALYSIS]", "[PEAKS]", "[IDENTIFY]"):
                            try:
                                done = int(parts[1])
                                total = int(parts[2])
                                _phase_labels = {
                                    "[ANALYSIS]": "Background",
                                    "[PEAKS]": "Peaks",
                                    "[IDENTIFY]": "Identify",
                                }
                                phase = _phase_labels.get(parts[0], parts[0])
                                self._event_queue.put(
                                    ("progress", phase, done, total))
                                continue
                            except ValueError:
                                pass
                        self.log(line)
                except (ValueError, OSError):
                    pass  # stdout closed by the watchdog once the process exited
                rc = int(proc.wait())
                self._event_queue.put(("done", rc, out_json))
            except Exception as e:
                self._event_queue.put(("error", repr(e)))

        threading.Thread(target=_worker_thread, daemon=True).start()

    def _update_progress(self, phase: str, done: int, total: int):
        self.progress.configure(maximum=max(total, 1), value=done)
        self.progress_label.configure(text=f"{phase}: {done} / {total} frames")

    def _run_done(self, returncode: int, out_json: str):
        self._run_proc = None
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        if returncode != 0:
            self._worker_status = "failed"
            self._update_status_bar()
            self.progress_label.configure(
                text=f"Failed (return code {returncode})", foreground=WARN)
            self.messagebox.showerror(
                "Analysis failed",
                f"Worker return code {returncode}\nSee the Run tab log.")
            return
        try:
            manifest = read_json(out_json)
        except Exception as e:
            self.log(f"Could not read manifest: {e!r}", "WARN")
            manifest = {}
        h5 = manifest.get("analysis_h5_file", "")
        steps = manifest.get("steps", [])
        self._worker_status = "done"
        self._update_status_bar()
        self.progress_label.configure(
            text=f"Done: {', '.join(steps)} -> {h5}", foreground=ACCENT2)
        self.log(f"Analysis complete: {h5}")
        if h5:
            self.config["analysis_h5_file"] = h5
            if "analysis_h5_file" in self.vars:
                self.vars["analysis_h5_file"].set(h5)
            self.save_config(silent=True)
            # Log peak summary if Step 2 ran.
            s2 = manifest.get("step2", {})
            if s2:
                n_peaks = s2.get("n_peaks", "?")
                n_good = s2.get("n_good", "?")
                self.log(f"Peak fitting: {n_good} good / {n_peaks} total peaks")
            # Log Step 3a summary if it ran.
            s3 = manifest.get("step3", {})
            if s3:
                for name, d in s3.get("summary", {}).items():
                    try:
                        pm = d.get("pressure_median")
                        p_txt = f"{pm:.1f} GPa" if pm is not None and pm == pm else "n/a"
                        self.log(
                            f"  {name}: seen in {d['n_frames_seen']} frame(s) "
                            f"(conf>{d.get('seen_conf', 0.5):.2f}); "
                            f"best recall {d.get('max_recall', 0.0):.2f}, "
                            f"best precision {d.get('max_precision', 0.0):.2f}, "
                            f"up to {d.get('max_matched', 0)} line(s) matched, "
                            f"median P={p_txt}"
                        )
                    except Exception:
                        pass
        # Refresh the views off the new results. Some loads are heavy (the
        # pattern map runs pymatgen reflection-track simulation), so stagger
        # them via after() rather than running all in one synchronous blast —
        # that kept the event loop from pumping and showed "Not responding"
        # right after a phase match. Each step yields to the loop between runs.
        loaders = [
            ("inspect", self.inspect_input_clicked),
            ("review", self.load_review),
            ("heatmap", self.load_heatmap),
            ("identify", self.load_identify),
            ("pattern map", self.load_pattern_map),
        ]

        def _run_loader(i=0):
            if i >= len(loaders):
                return
            name, fn = loaders[i]
            try:
                fn()
            except Exception as e:
                import traceback
                self.log(f"Auto {name} load failed: {e!r}", "WARN")
                self.log(traceback.format_exc(), "WARN")
            self.root.after(20, lambda: _run_loader(i + 1))

        self.root.after(20, _run_loader)

    def _run_error(self, err: str):
        self._run_proc = None
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self._worker_status = "failed"
        self._update_status_bar()
        self.progress_label.configure(text="Launch error", foreground=WARN)
        self.messagebox.showerror("Worker launch error", err)

    def cancel_analysis(self):
        proc = self._run_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            self.log("Cancel requested — terminating worker", "WARN")

    # ------------------------------------------------------------------
    # Tab 5 — Review (single-frame QC)
    # ------------------------------------------------------------------

    def _tab_review(self, frame):
        tk, ttk = self.tk, self.ttk

        # Controls row
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(0, 4))

        ttk.Button(ctrl, text="Load review", command=self.load_review).pack(
            side="left", padx=4)

        ttk.Label(ctrl, text="Frame:", foreground=MUTED).pack(side="left", padx=(12, 2))
        self._review_idx_var = tk.IntVar(value=0)
        self._review_scale = ttk.Scale(
            ctrl, from_=0, to=0, orient="horizontal", length=200,
            variable=self._review_idx_var,
            command=self._on_review_slider,
        )
        self._review_scale.pack(side="left", padx=2)
        self._review_spinbox = ttk.Spinbox(
            ctrl, from_=0, to=0, width=6,
            textvariable=self._review_idx_var,
            command=self._on_review_spinbox,
        )
        self._review_spinbox.pack(side="left", padx=2)

        # Trace overlays
        self._show_mean = tk.BooleanVar(value=True)
        self._show_robust = tk.BooleanVar(value=True)
        self._show_baseline = tk.BooleanVar(value=True)
        self._show_clean = tk.BooleanVar(value=True)
        self._show_spot = tk.BooleanVar(value=False)
        self._show_peaks = tk.BooleanVar(value=True)
        for var, label in [
            (self._show_mean, "mean"),
            (self._show_robust, "robust"),
            (self._show_baseline, "baseline"),
            (self._show_clean, "clean"),
            (self._show_spot, "spot_residual"),
            (self._show_peaks, "fitted peaks"),
        ]:
            ttk.Checkbutton(ctrl, text=label, variable=var,
                            command=self._schedule_review_render).pack(
                side="left", padx=2)

        # Matplotlib area
        self.review_plot_frame = ttk.Frame(frame)
        self.review_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.review_plot_frame,
            text="Load the analysis HDF5 to plot per-frame traces.",
            foreground=MUTED,
        ).pack(anchor="center", expand=True)

    def load_review(self):
        """Load metadata from the analysis HDF5 and render the current frame."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            # Silently skip (called on auto-refresh); show error only if user triggered.
            return
        from .review import inspect_analysis
        try:
            info = inspect_analysis(path)
        except Exception as e:
            self.log(f"Review load failed: {e!r}", "WARN")
            return
        if not info.get("ok_to_read"):
            self.log("Analysis HDF5 not readable for review.", "WARN")
            return
        nf = int(info.get("n_frames", 0))
        self._review_nframes = nf
        contam = info.get("contamination")
        self._review_contamination = contam
        if nf > 0:
            self._frame_count = nf
            self._review_scale.configure(to=max(nf - 1, 0))
            self._review_spinbox.configure(to=max(nf - 1, 0))
        self._update_status_bar()
        self._render_review(int(self._review_idx_var.get()))

    def _on_review_slider(self, value):
        """Called on every slider tick — schedule a debounced render."""
        try:
            idx = int(float(value))
        except (ValueError, TypeError):
            return
        # Keep spinbox in sync immediately.
        try:
            self._review_idx_var.set(idx)
        except Exception:
            pass
        self._schedule_review_render()

    def _on_review_spinbox(self):
        self._schedule_review_render()

    def _schedule_review_render(self):
        """Debounce rapid slider drags: cancel any pending render and re-schedule."""
        if self._review_after is not None:
            try:
                self.root.after_cancel(self._review_after)
            except Exception:
                pass
        self._review_after = self.root.after(
            120, self._fire_review_render)

    def _fire_review_render(self):
        self._review_after = None
        try:
            idx = int(self._review_idx_var.get())
        except (ValueError, TypeError):
            idx = 0
        self._render_review(idx)

    def _render_review(self, frame_index: int):
        """Render the two-axis review figure for one frame."""
        # Close previous figure to avoid leaks.
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

        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.ttk.Label(
                self.review_plot_frame,
                text="No analysis HDF5 loaded — run analysis or set path on Input tab.",
                foreground=MUTED,
            ).pack(anchor="center", expand=True)
            return

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            self.ttk.Label(
                self.review_plot_frame,
                text=f"matplotlib unavailable: {e}",
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            return

        import numpy as np

        from .review import frame_data
        try:
            fd = frame_data(path, frame_index)
        except Exception as e:
            self.ttk.Label(
                self.review_plot_frame,
                text=f"frame_data error: {e}",
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            return

        if not fd.get("ok"):
            self.ttk.Label(
                self.review_plot_frame,
                text=f"frame_data: {fd.get('error', 'unknown error')}",
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            return

        radial = fd.get("radial")
        unit = fd.get("unit") or "radial bin"
        x = np.asarray(radial) if radial is not None else None

        # constrained layout recomputes margins on every resize (one-shot
        # tight_layout leaves labels clipped/overlapping when the pane resizes).
        fig = Figure(figsize=(7, 6), dpi=100, layout="constrained")
        self._review_fig = fig
        fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0])   # pattern traces get the bulk of the height
        ax2 = fig.add_subplot(gs[1])   # contamination strip below

        def _plot(ax, arr, label, color, lw=0.9, alpha=0.85):
            if arr is None:
                return
            y = np.asarray(arr, dtype=float)
            if x is not None and x.shape == y.shape:
                ax.plot(x, y, lw=lw, alpha=alpha, label=label, color=color)
            else:
                ax.plot(y, lw=lw, alpha=alpha, label=label, color=color)

        if self._show_mean.get():
            _plot(ax1, fd.get("mean"), "mean", CLR_RAW)
        if self._show_robust.get():
            _plot(ax1, fd.get("robust"), "robust", CLR_MSKD)
        if self._show_baseline.get():
            _plot(ax1, fd.get("baseline"), "baseline", CLR_SMTH)
        if self._show_clean.get():
            _plot(ax1, fd.get("clean"), "clean", ACCENT2)
        if self._show_spot.get():
            _plot(ax1, fd.get("spot_residual"), "spot_residual", CLR_DIFF)

        # Overlay fitted peaks if requested.
        peaks = fd.get("peaks", [])
        if self._show_peaks.get() and peaks:
            good_centers = [
                p["center"] for p in peaks if p.get("flag", 0) == 0
            ]
            bad_centers = [
                p["center"] for p in peaks if p.get("flag", 0) != 0
            ]
            # Retrieve clean for amplitude reference.
            clean_arr = fd.get("clean")
            y_ref = np.asarray(clean_arr, dtype=float) if clean_arr is not None else None
            for c in good_centers:
                ax1.axvline(c, color=ACCENT2, lw=0.7, alpha=0.6)
            for c in bad_centers:
                ax1.axvline(c, color=WARN, lw=0.7, alpha=0.5)

        fname = Path(fd.get("filename", "")).name or f"frame {frame_index}"
        ax1.set_title(f"{fname}  [frame {frame_index}]", color=FG)
        ax1.set_xlabel(unit)
        ax1.set_ylabel("intensity")
        if any([
            self._show_mean.get(), self._show_robust.get(),
            self._show_baseline.get(), self._show_clean.get(),
            self._show_spot.get(),
        ]):
            ax1.legend(fontsize=7, framealpha=0.4)
        self._style_ax(ax1)

        # Bottom axis: contamination across series.
        contam = self._review_contamination
        if contam is not None and len(contam):
            c_arr = np.asarray(contam, dtype=float)
            ax2.plot(c_arr, lw=0.8, color=CLR_DIFF, alpha=0.85)
            ax2.axvline(frame_index, color=ACCENT, lw=1.2, alpha=0.8)
            ax2.set_xlabel("frame")
            ax2.set_ylabel("contamination")
            ax2.set_title("Contamination vs frame", color=FG)
        else:
            ax2.set_title("Contamination (not available)", color=FG)
            ax2.set_xlabel("frame")
        self._style_ax(ax2)

        self._review_canvas = self._embed_figure(self.review_plot_frame, fig)

    def _style_ax(self, ax):
        ax.set_facecolor(BG2)
        ax.tick_params(colors=FG, which="both")
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        for s in ax.spines.values():
            s.set_edgecolor(FG)

    def _style_colorbar(self, cb):
        """Recolour a colorbar to the dark palette — its label, ticks, and
        outline default to black and vanish against the figure background."""
        try:
            cb.ax.tick_params(colors=FG, which="both")
            cb.ax.yaxis.label.set_color(FG)
            cb.ax.xaxis.label.set_color(FG)
            if cb.outline is not None:
                cb.outline.set_edgecolor(FG)
        except Exception:
            pass

    def _embed_figure(self, parent, fig, toolbar=True):
        """Embed a matplotlib figure so it tracks the pane size instead of forcing it.

        A ttk.Notebook sizes itself to its largest tab, so a fixed-size canvas
        (figsize×dpi ≈ 700–800 px) would pin the whole window to at least that
        size — the plot then loads larger than the GUI and can't shrink. Giving
        the canvas widget a tiny *requested* size removes that floor; fill+expand
        grows it to fill the pane, and matplotlib's own <Configure> handler
        redraws the figure at the allocated size (constrained layout re-flows the
        margins on each resize).

        A navigation toolbar (home / pan / box-zoom / save) is packed beneath the
        canvas so dense patterns can be zoomed into without resizing the window.
        Returns the canvas.
        """
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=parent)
        widget = canvas.get_tk_widget()
        widget.configure(width=10, height=10)   # don't let the canvas set the min size
        # Reserve the toolbar strip at the bottom first (best-effort: it degrades
        # to None if the backend is unavailable and never blocks the plot), then
        # let the canvas fill the remainder.
        if toolbar:
            self._add_nav_toolbar(canvas, parent)
        widget.pack(side="top", fill="both", expand=True)

        # Keep the figure sized to its allocated canvas so it can never render
        # larger than the pane (matplotlib's own resize wasn't constraining it
        # in this embedding).
        def _apply_size(w, h, canvas=canvas, fig=fig):
            if w < 20 or h < 20:
                return False
            dpi = fig.get_dpi() or 100
            fig.set_size_inches(w / dpi, h / dpi, forward=False)
            canvas.draw_idle()
            return True

        widget.bind("<Configure>", lambda e: _apply_size(e.width, e.height), add="+")

        # Initial fit: <Configure> only fires on later resizes, so without this
        # the first render keeps the figure's large default size and overflows
        # the pane until the user resizes the window. Poll until the widget has
        # its real allocated size, then size the figure to it.
        def _initial_fit(tries=0, widget=widget):
            if _apply_size(widget.winfo_width(), widget.winfo_height()):
                return
            if tries < 40:
                self.root.after(25, lambda: _initial_fit(tries + 1))
        self.root.after(0, _initial_fit)

        canvas.draw()
        return canvas

    def _add_nav_toolbar(self, canvas, parent):
        """Add a dark-styled matplotlib navigation toolbar below an embedded
        canvas (pan / box-zoom / home / save). Degrades silently if the toolbar
        backend is unavailable."""
        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
            tb.update()
            # matplotlib's toolbar glyphs are dark, so painting the buttons the
            # window background (near-black) hides them. Give the buttons a light
            # fill so the icons read clearly; keep the frame + coordinate label
            # on the dark palette.
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
    # Tab 6 — Heatmap
    # ------------------------------------------------------------------

    def _tab_heatmap(self, frame):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(0, 4))
        ttk.Button(top, text="Load heatmap", command=self.load_heatmap).pack(
            side="left", padx=4)
        self._heatmap_good_only = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top, text="Good peaks only", variable=self._heatmap_good_only,
            command=self.load_heatmap,
        ).pack(side="left", padx=8)
        ttk.Label(top, text="Color by:", foreground=MUTED).pack(side="left", padx=(12, 2))
        self._heatmap_color_by = tk.StringVar(value="area")
        ttk.Combobox(
            top, textvariable=self._heatmap_color_by,
            values=["area", "amplitude", "fwhm"],
            width=10, state="readonly",
        ).pack(side="left", padx=2)
        ttk.Button(top, text="Refresh", command=self.load_heatmap).pack(
            side="left", padx=4)

        self.heatmap_status = ttk.Label(top, text="", foreground=MUTED)
        self.heatmap_status.pack(side="left", padx=12)

        self.heatmap_plot_frame = ttk.Frame(frame)
        self.heatmap_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.heatmap_plot_frame,
            text="Load the analysis HDF5 to display the peak heatmap.",
            foreground=MUTED,
        ).pack(anchor="center", expand=True)

    def load_heatmap(self):
        """Render the peak-map scatter plot from the analysis HDF5."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            return  # silently skip auto-calls

        prev = getattr(self, "_heatmap_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._heatmap_fig = None

        for w in self.heatmap_plot_frame.winfo_children():
            w.destroy()

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.colors as mcolors
        except Exception as e:
            self.ttk.Label(
                self.heatmap_plot_frame,
                text=f"matplotlib unavailable: {e}",
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            return

        import numpy as np

        from .review import peak_map
        good_only = bool(self._heatmap_good_only.get())
        try:
            pm = peak_map(path, good_only=good_only)
        except Exception as e:
            self.ttk.Label(
                self.heatmap_plot_frame,
                text=f"peak_map error: {e}",
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            return

        if not pm.get("ok"):
            err = pm.get("error", "unknown error")
            self.ttk.Label(
                self.heatmap_plot_frame,
                text=f"Peak map: {err}",
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            if hasattr(self, "heatmap_status"):
                self.heatmap_status.configure(text=err)
            return

        frame_arr = np.asarray(pm["frame"], dtype=float)
        center_arr = np.asarray(pm["center"], dtype=float)
        color_by = str(self._heatmap_color_by.get())
        c_arr = np.asarray(pm.get(color_by, pm["area"]), dtype=float)

        n_pts = int(frame_arr.size)
        if hasattr(self, "heatmap_status"):
            self.heatmap_status.configure(
                text=f"{n_pts} peaks plotted" + (" (good only)" if good_only else ""))

        unit = pm.get("unit") or "radial"

        fig = Figure(figsize=(7, 5), dpi=100, layout="constrained")
        self._heatmap_fig = fig
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(1, 1, 1)
        self._style_ax(ax)

        if n_pts == 0:
            ax.set_title("No peaks to display", color=FG)
        else:
            # Log-safe normalisation for area / amplitude; linear for fwhm.
            if color_by in ("area", "amplitude"):
                pos = c_arr[c_arr > 0]
                if pos.size > 0:
                    vmin = float(pos.min())
                    vmax = float(c_arr.max())
                    norm = mcolors.LogNorm(vmin=max(vmin, 1e-9), vmax=max(vmax, vmin + 1e-9))
                else:
                    norm = None
            else:
                norm = None

            # Larger markers with a light edge so even dark-coloured (low-value)
            # points read against the dark axes background.
            sc = ax.scatter(
                frame_arr, center_arr, c=c_arr,
                cmap="viridis", s=28, alpha=0.9, norm=norm,
                edgecolors=FG, linewidths=0.4,
            )
            try:
                cb = fig.colorbar(sc, ax=ax, label=color_by)
                self._style_colorbar(cb)
            except Exception:
                pass
            ax.set_xlabel("frame index", color=FG)
            ax.set_ylabel(f"peak center ({unit})", color=FG)
            ax.set_title(f"Peak map — {n_pts} peaks", color=FG)

        self._heatmap_canvas = self._embed_figure(self.heatmap_plot_frame, fig)

    # ------------------------------------------------------------------
    # Tab 7 — Phases (reference-phase library)
    # ------------------------------------------------------------------

    def _phases_workspace(self) -> Path:
        """Return the workspace dir for the user phase library."""
        ws = self.config.get("workspace_root")
        if ws:
            return Path(ws)
        return self.config_path.parent

    def _tab_phases(self, frame):
        tk, ttk = self.tk, self.ttk

        # Controls row
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(0, 4))
        ttk.Button(ctrl, text="Import CIF…", command=self.import_cif_clicked).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="Add phase…",
                   command=lambda: self._phase_dialog(None)).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="Edit…", command=self.edit_phase_clicked).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="Remove", command=self.remove_phase_clicked).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="Refresh", command=self.load_phases_table).pack(
            side="left", padx=4)
        self._phases_status = ttk.Label(ctrl, text="", foreground=MUTED)
        self._phases_status.pack(side="right", padx=8)

        # pymatgen availability hint
        self._phases_pymatgen_label = ttk.Label(frame, text="", foreground=MUTED,
                                                wraplength=800, justify="left")
        self._phases_pymatgen_label.pack(fill="x", padx=4, pady=(0, 2))

        # Treeview with scrollbar
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        cols = ("enabled", "name", "category", "spacegroup", "K0", "K0p", "source")
        self.phases_tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                        selectmode="browse")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self.phases_tree.yview)
        self.phases_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.phases_tree.pack(side="left", fill="both", expand=True)

        self.phases_tree.heading("enabled",     text="✓")
        self.phases_tree.heading("name",        text="Name")
        self.phases_tree.heading("category",    text="Category")
        self.phases_tree.heading("spacegroup",  text="Space group")
        self.phases_tree.heading("K0",          text="K0 (GPa)")
        self.phases_tree.heading("K0p",         text="K0'")
        self.phases_tree.heading("source",      text="Source/Origin")

        self.phases_tree.column("enabled",    width=36,  minwidth=32,  anchor="center", stretch=False)
        self.phases_tree.column("name",       width=140, minwidth=80)
        self.phases_tree.column("category",   width=80,  minwidth=60)
        self.phases_tree.column("spacegroup", width=100, minwidth=60)
        self.phases_tree.column("K0",         width=80,  minwidth=50,  anchor="center")
        self.phases_tree.column("K0p",        width=60,  minwidth=40,  anchor="center")
        self.phases_tree.column("source",     width=260, minwidth=100)

        self.phases_tree.tag_configure("user", foreground=ACCENT)

        self.phases_tree.bind("<Button-1>", self._phases_tree_click)
        self.phases_tree.bind("<Double-1>", self.edit_phase_clicked)

        self._phases_by_name: "Dict[str, Any]" = {}

        self.load_phases_table()

    def load_phases_table(self):
        from .phases import list_phases, pymatgen_available
        ws = self._phases_workspace()
        phases = list_phases(ws)
        self._phases_by_name = {p.name: p for p in phases}

        enabled_set = set(str(n) for n in self.config.get("candidate_phases", []))

        # Clear and repopulate
        self.phases_tree.delete(*self.phases_tree.get_children())
        for p in phases:
            eos = p.eos or {}
            k0_val = eos.get("K0")
            k0p_val = eos.get("K0p")
            k0_str = f"{k0_val:g}" if k0_val is not None else "—"
            k0p_str = f"{k0p_val:g}" if k0p_val is not None else "—"
            if p.builtin:
                origin = p.source or "bundled"
            else:
                origin = "(user)" if not p.source else f"(user) {p.source}"
            enabled_mark = "✓" if p.name in enabled_set else ""
            tags = () if p.builtin else ("user",)
            self.phases_tree.insert(
                "", "end", iid=p.name,
                values=(enabled_mark, p.name, p.category,
                        p.space_group or "—", k0_str, k0p_str, origin),
                tags=tags,
            )

        n_total = len(phases)
        n_user = sum(1 for p in phases if not p.builtin)
        n_builtin = n_total - n_user
        n_enabled = len([n for n in enabled_set if n in self._phases_by_name])
        if hasattr(self, "_phases_status"):
            self._phases_status.configure(
                text=(f"{n_total} phases ({n_user} user, {n_builtin} bundled)"
                      f"  ·  {n_enabled} enabled"))

        if hasattr(self, "_phases_pymatgen_label"):
            if pymatgen_available():
                self._phases_pymatgen_label.configure(
                    text="pymatgen available — CIF auto-parsing and pattern simulation enabled.",
                    foreground=MUTED)
            else:
                self._phases_pymatgen_label.configure(
                    text=("pymatgen not installed — CIF auto-parsing & pattern simulation "
                          "disabled (pip install pymatgen). You can still add phases manually."),
                    foreground=WARN)

    def _phases_tree_click(self, event):
        col = self.phases_tree.identify_column(event.x)
        if col == "#1":
            row = self.phases_tree.identify_row(event.y)
            if row:
                self._toggle_phase_enabled(row)
        # Otherwise let normal selection happen (no return / no break needed)

    def _toggle_phase_enabled(self, name: str):
        enabled = list(self.config.get("candidate_phases", []))
        enabled_strs = [str(n) for n in enabled]
        if name in enabled_strs:
            enabled_strs.remove(name)
        else:
            enabled_strs.append(name)
        self.config["candidate_phases"] = sorted(set(enabled_strs))
        self.save_config(silent=True)
        # Update just this row's enabled cell
        mark = "✓" if name in self.config["candidate_phases"] else ""
        try:
            self.phases_tree.set(name, "enabled", mark)
        except Exception:
            pass
        # Refresh status count
        enabled_set = set(self.config["candidate_phases"])
        n_enabled = len([n for n in enabled_set if n in self._phases_by_name])
        n_total = len(self._phases_by_name)
        n_user = sum(1 for p in self._phases_by_name.values() if not p.builtin)
        n_builtin = n_total - n_user
        if hasattr(self, "_phases_status"):
            self._phases_status.configure(
                text=(f"{n_total} phases ({n_user} user, {n_builtin} bundled)"
                      f"  ·  {n_enabled} enabled"))
        self.log(f"Phase '{name}' {'enabled' if mark else 'disabled'} as candidate.")

    def _phase_dialog(self, existing):
        """Add (existing=None) or Edit (existing=Phase) a user phase."""
        from .phases import Phase, upsert_user_phase, CATEGORIES
        tk, ttk = self.tk, self.ttk

        title = "Edit phase" if existing else "Add phase"
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.configure(bg=BG)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(True, True)

        def _f(s):
            """Parse float leniently; return None on blank/invalid."""
            try:
                v = str(s).strip()
                return float(v) if v else None
            except (ValueError, TypeError):
                return None

        pad = {"padx": 6, "pady": 3}

        content = ttk.Frame(dlg, padding=10)
        content.pack(fill="both", expand=True)

        row = 0
        # Name
        ttk.Label(content, text="Name").grid(row=row, column=0, sticky="w", **pad)
        v_name = tk.StringVar(value=existing.name if existing else "")
        ttk.Entry(content, textvariable=v_name, width=36).grid(
            row=row, column=1, columnspan=3, sticky="we", **pad)
        row += 1

        # Formula
        ttk.Label(content, text="Formula").grid(row=row, column=0, sticky="w", **pad)
        v_formula = tk.StringVar(value=existing.formula if existing else "")
        ttk.Entry(content, textvariable=v_formula, width=36).grid(
            row=row, column=1, columnspan=3, sticky="we", **pad)
        row += 1

        # Category
        ttk.Label(content, text="Category").grid(row=row, column=0, sticky="w", **pad)
        v_category = tk.StringVar(
            value=(existing.category if existing else "marker"))
        ttk.Combobox(content, textvariable=v_category,
                     values=list(CATEGORIES), state="readonly", width=16).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1

        # Space group
        ttk.Label(content, text="Space group").grid(row=row, column=0, sticky="w", **pad)
        v_sg = tk.StringVar(value=existing.space_group if existing else "")
        ttk.Entry(content, textvariable=v_sg, width=20).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1

        # Lattice — six small entries in one row
        ttk.Label(content, text="Lattice (Å, °)").grid(
            row=row, column=0, sticky="w", **pad)
        lat_frame = ttk.Frame(content)
        lat_frame.grid(row=row, column=1, columnspan=3, sticky="w", **pad)
        lat_keys = ("a", "b", "c", "alpha", "beta", "gamma")
        lat_defaults = {"alpha": "90", "beta": "90", "gamma": "90"}
        lat_vars: "Dict[str, tk.StringVar]" = {}
        for i, k in enumerate(lat_keys):
            ex_val = ""
            if existing and existing.lattice:
                v_raw = existing.lattice.get(k)
                ex_val = f"{v_raw:g}" if v_raw is not None else ""
            if not ex_val:
                ex_val = lat_defaults.get(k, "")
            ttk.Label(lat_frame, text=k).grid(row=0, column=i * 2, sticky="e",
                                               padx=(6 if i else 0, 1))
            sv = tk.StringVar(value=ex_val)
            ttk.Entry(lat_frame, textvariable=sv, width=8).grid(
                row=0, column=i * 2 + 1, padx=(0, 4))
            lat_vars[k] = sv
        row += 1

        # EOS
        ttk.Label(content, text="EOS  V0 (Å³)").grid(
            row=row, column=0, sticky="w", **pad)
        eos_frame = ttk.Frame(content)
        eos_frame.grid(row=row, column=1, columnspan=3, sticky="w", **pad)
        ex_eos = (existing.eos or {}) if existing else {}
        eos_keys = ("V0", "K0", "K0p")
        eos_labels = ("V0 (Å³)", "K0 (GPa)", "K0'")
        eos_vars: "Dict[str, tk.StringVar]" = {}
        for i, (k, lbl) in enumerate(zip(eos_keys, eos_labels)):
            v_raw = ex_eos.get(k)
            ex_val = f"{v_raw:g}" if v_raw is not None else ""
            ttk.Label(eos_frame, text=lbl).grid(row=0, column=i * 2,
                                                 sticky="e", padx=(6 if i else 0, 1))
            sv = tk.StringVar(value=ex_val)
            ttk.Entry(eos_frame, textvariable=sv, width=10).grid(
                row=0, column=i * 2 + 1, padx=(0, 4))
            eos_vars[k] = sv
        row += 1

        # Source
        ttk.Label(content, text="Source").grid(row=row, column=0, sticky="w", **pad)
        v_source = tk.StringVar(value=existing.source if existing else "")
        ttk.Entry(content, textvariable=v_source, width=50).grid(
            row=row, column=1, columnspan=3, sticky="we", **pad)
        row += 1

        # Notes
        ttk.Label(content, text="Notes").grid(row=row, column=0, sticky="nw", **pad)
        notes_text = tk.Text(content, width=50, height=3, bg=BG2, fg=FG,
                             insertbackground=FG, relief="flat", wrap="word")
        notes_text.grid(row=row, column=1, columnspan=3, sticky="we", **pad)
        if existing and existing.notes:
            notes_text.insert("1.0", existing.notes)
        row += 1

        content.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(content)
        btn_frame.grid(row=row, column=0, columnspan=4, sticky="e", pady=(8, 0))

        def _save():
            name = v_name.get().strip()
            if not name:
                self.messagebox.showerror("Validation", "Name is required.",
                                          parent=dlg)
                return

            lattice = {}
            for k in lat_keys:
                fv = _f(lat_vars[k].get())
                if fv is not None:
                    lattice[k] = fv

            eos: "Dict[str, Any]" = {"type": "BM3"}
            for k in eos_keys:
                fv = _f(eos_vars[k].get())
                if fv is not None:
                    eos[k] = fv

            notes = notes_text.get("1.0", "end-1c")

            phase = Phase(
                name=name,
                formula=v_formula.get().strip(),
                category=v_category.get(),
                space_group=v_sg.get().strip(),
                lattice=lattice,
                atoms=(existing.atoms if existing else []),
                eos=eos,
                axial_eos=(existing.axial_eos if existing else {}),
                amorphous=(existing.amorphous if existing else False),
                cif_path=(existing.cif_path if existing else ""),
                source=v_source.get().strip(),
                notes=notes,
            )
            try:
                upsert_user_phase(self._phases_workspace(), phase)
            except Exception as e:
                self.messagebox.showerror("Save failed", repr(e), parent=dlg)
                return
            dlg.destroy()
            self.load_phases_table()
            action = "updated" if existing else "added"
            self.log(f"Phase '{name}' {action}.")

        ttk.Button(btn_frame, text="Save", command=_save).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel",
                   command=dlg.destroy).pack(side="left", padx=4)

    def edit_phase_clicked(self, event=None):
        sel = self.phases_tree.selection()
        if not sel:
            self.messagebox.showinfo("Edit phase", "Select a phase to edit.")
            return
        name = sel[0]
        phase = self._phases_by_name.get(name)
        if phase is None:
            self.messagebox.showinfo("Edit phase", f"Phase '{name}' not found.")
            return
        self._phase_dialog(phase)

    def remove_phase_clicked(self):
        from .phases import remove_user_phase
        sel = self.phases_tree.selection()
        if not sel:
            self.messagebox.showinfo("Remove phase", "Select a phase to remove.")
            return
        name = sel[0]
        if not self.messagebox.askyesno(
                "Remove phase",
                f"Remove user phase '{name}'? This cannot be undone."):
            return
        ws = self._phases_workspace()
        removed = remove_user_phase(ws, name)
        if not removed:
            self.messagebox.showinfo(
                "Remove phase",
                f"'{name}' is a bundled phase and cannot be deleted.\n"
                "Use Edit to create a user override.")
            return
        # Drop from candidate_phases if present
        enabled = list(self.config.get("candidate_phases", []))
        if name in enabled:
            enabled.remove(name)
            self.config["candidate_phases"] = sorted(set(enabled))
            self.save_config(silent=True)
        self.log(f"Phase '{name}' removed.")
        self.load_phases_table()

    def import_cif_clicked(self):
        from .phases import import_cif, pymatgen_available
        path = self.filedialog.askopenfilename(
            title="Import CIF",
            filetypes=[("CIF", "*.cif"), ("All files", "*.*")],
        )
        if not path:
            return
        ws = self._phases_workspace()
        try:
            phase = import_cif(ws, path)
        except Exception as e:
            self.messagebox.showerror("Import CIF failed", repr(e))
            return
        self.log(f"CIF imported: {path} -> phase '{phase.name}'")
        self.load_phases_table()
        if not pymatgen_available():
            self.messagebox.showinfo(
                "CIF imported",
                "The CIF was stored but could not be auto-parsed (pymatgen is not "
                "installed). Install pymatgen for automatic lattice/structure parsing, "
                "or fill in the lattice and EOS fields manually below.")
        # Always open the edit dialog so the user can fill in / verify the EOS
        self._phase_dialog(self._phases_by_name.get(phase.name, phase))

    # ------------------------------------------------------------------
    # Tab 8 — Identify (Step 3a: deterministic EOS phase matching)
    # ------------------------------------------------------------------

    def _tab_identify(self, frame):
        tk, ttk = self.tk, self.ttk

        # -- params area --------------------------------------------------
        ttk.Label(
            frame, text="Phase identification (EOS matching)",
            font=("TkDefaultFont", 12, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 2))

        self.checkbox(frame, "run_step3",
                      "Enable phase identification in the next run", row=1)
        self.field(frame, "p_min", "Pressure min (GPa)", row=2, width=12)
        self.field(frame, "p_max", "Pressure max (GPa)", row=3, width=12)
        self.field(frame, "rel_tol", "Match tolerance (Δd/d)", row=4, width=12)
        self.field(frame, "identify_wavelength",
                   "Wavelength (Å, blank=auto)", row=5, width=12)

        ttk.Label(
            frame,
            text=(
                "What this does: for each candidate phase, fit its Birch–Murnaghan "
                "equation of state to find the pressure whose simulated peak "
                "positions best match the fitted peaks in each frame — giving a "
                "per-frame pressure estimate and a match confidence per phase. "
                "(This is “Step 3a” of the analysis pipeline; requires pymatgen for "
                "d-spacing simulation.)\n\n"
                "How to run it:\n"
                "  1. Phases tab — enable the candidate phases known to be in the cell.\n"
                "  2. Here — tick “Enable phase identification” and set the pressure "
                "range / tolerance.\n"
                "  3. Run tab — click Run (it executes every enabled step).\n"
                "  4. Back here — click “Load identification” to plot pressure vs frame.\n"
                "Already have a results file? Just click “Load identification” below."
            ),
            foreground=MUTED, justify="left", wraplength=640,
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=6, pady=(12, 4))

        # -- controls row -------------------------------------------------
        ctrl = ttk.Frame(frame)
        ctrl.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 2))

        ttk.Button(ctrl, text="Load identification",
                   command=self.load_identify).pack(side="left", padx=4)

        ttk.Label(ctrl, text="Min confidence:", foreground=MUTED).pack(
            side="left", padx=(12, 2))
        self._identify_conf_var = tk.StringVar(value="0.5")
        _conf_entry = ttk.Entry(ctrl, textvariable=self._identify_conf_var, width=6)
        _conf_entry.pack(side="left", padx=2)
        _conf_entry.bind("<Return>", lambda e: self.load_identify())

        ttk.Button(ctrl, text="Redraw",
                   command=self.load_identify).pack(side="left", padx=4)

        self._identify_status = ttk.Label(ctrl, text="", foreground=MUTED)
        self._identify_status.pack(side="left", padx=12)

        # -- plot area ----------------------------------------------------
        self.identify_plot_frame = ttk.Frame(frame)
        self.identify_plot_frame.grid(
            row=8, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(8, weight=1)
        frame.columnconfigure(0, weight=1)

        ttk.Label(
            self.identify_plot_frame,
            text="Enable phase identification and Run (see steps above), or click "
                 "“Load identification” to plot pressure vs frame from a results file.",
            foreground=MUTED,
        ).pack(anchor="center", expand=True)

    def load_identify(self):
        """Render the Step-3a pressure-vs-frame plot from the analysis HDF5."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            return  # silently skip auto-calls

        # prev-figure-close leak guard
        prev = getattr(self, "_identify_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._identify_fig = None

        for w in self.identify_plot_frame.winfo_children():
            w.destroy()

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            self.ttk.Label(
                self.identify_plot_frame,
                text=f"matplotlib unavailable: {e}",
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            return

        import numpy as np

        from .review import identify_tracks
        tr = identify_tracks(path)
        if not tr["ok"]:
            self.ttk.Label(
                self.identify_plot_frame,
                text=tr["error"],
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            if hasattr(self, "_identify_status"):
                self._identify_status.configure(text=tr["error"])
            return

        # Parse confidence threshold
        conf_min = 0.5
        try:
            conf_min = float(self._identify_conf_var.get())
            conf_min = max(0.0, min(1.0, conf_min))
        except (ValueError, AttributeError):
            pass

        fig = Figure(figsize=(7, 6), dpi=100, layout="constrained")
        self._identify_fig = fig
        fig.patch.set_facecolor(BG)
        ax_pres = fig.add_subplot(2, 1, 1)
        ax_conf = fig.add_subplot(2, 1, 2)

        for rec in tr["phases"]:
            name = rec["name"]
            pressure = np.asarray(rec["pressure"], dtype=float)
            conf_arr = (
                np.asarray(rec["confidence"], dtype=float)
                if rec["confidence"] is not None
                else np.zeros(pressure.size, dtype=float)
            )
            x = np.arange(pressure.size)
            mask = conf_arr >= conf_min

            label = name if rec["has_eos"] else f"{name} (no EOS)"

            # Plot pressure where mask is satisfied; capture the line color.
            if mask.any():
                (ln,) = ax_pres.plot(
                    x[mask], pressure[mask],
                    marker=".", markersize=3, linewidth=0.7,
                    label=label,
                )
                color = ln.get_color()
            else:
                # No points meet threshold — still need a color for confidence axis.
                (ln,) = ax_pres.plot([], [], marker=".", markersize=3,
                                     linewidth=0.7, label=label)
                color = ln.get_color()

            # Always show confidence trace in the same color.
            ax_conf.plot(x, conf_arr, linewidth=0.7, color=color)

        ax_pres.set_ylabel("pressure (GPa)")
        ax_pres.set_title("Best-fit pressure per phase", color=FG)
        handles, labels = ax_pres.get_legend_handles_labels()
        if handles:
            ax_pres.legend(fontsize=7, framealpha=0.4)
        self._style_ax(ax_pres)

        ax_conf.axhline(conf_min, color=MUTED, linewidth=0.8, linestyle="--")
        ax_conf.set_xlabel("frame index")
        ax_conf.set_ylabel("confidence")
        ax_conf.set_ylim(0, 1.02)
        self._style_ax(ax_conf)

        self._identify_canvas = self._embed_figure(self.identify_plot_frame, fig)

        if hasattr(self, "_identify_status"):
            self._identify_status.configure(
                text=f"{len(tr['phases'])} phase(s), {tr['n_frames']} frames")

    # ------------------------------------------------------------------
    # Helpers shared by Tab 9
    # ------------------------------------------------------------------

    def _enabled_phase_objects(self):
        """Return Phase objects for names in config candidate_phases, resolved from the library."""
        from .phases import load_library
        ws = self._phases_workspace()
        try:
            library = load_library(ws)
        except Exception:
            return []
        names = self.config.get("candidate_phases", [])
        return [library[n] for n in names if n in library]

    # ------------------------------------------------------------------
    # Tab 9 — Pattern map (Hrubiak/XDI-style waterfall + tracks + layers)
    # ------------------------------------------------------------------

    def _tab_patternmap(self, frame):
        tk, ttk = self.tk, self.ttk

        # Controls row 1
        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=(0, 2))

        ttk.Button(row1, text="Load pattern map",
                   command=self.load_pattern_map).pack(side="left", padx=4)

        ttk.Label(row1, text="Source:", foreground=MUTED).pack(side="left", padx=(12, 2))
        self._pm_source = ttk.Combobox(
            row1,
            values=["clean", "robust", "mean", "baseline", "spot_residual"],
            state="readonly", width=12,
        )
        self._pm_source.set("clean")
        self._pm_source.pack(side="left", padx=2)
        self._pm_source.bind("<<ComboboxSelected>>",
                             lambda e: self.load_pattern_map())

        self._pm_tracks = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row1, text="Overlay reflection tracks",
            variable=self._pm_tracks, command=self.load_pattern_map,
        ).pack(side="left", padx=8)

        self._pm_layers = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Show phase layers",
            variable=self._pm_layers, command=self.load_pattern_map,
        ).pack(side="left", padx=4)

        self._pm_status = ttk.Label(row1, text="", foreground=MUTED)
        self._pm_status.pack(side="right", padx=8)

        # Controls row 2
        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=(0, 4))

        ttk.Label(row2, text="Export →", foreground=MUTED).pack(side="left", padx=(4, 6))
        ttk.Button(row2, text="Export ML dataset…",
                   command=self.export_ml_clicked).pack(side="left", padx=2)
        ttk.Button(row2, text="Export simulated set…",
                   command=self.export_sim_clicked).pack(side="left", padx=2)

        self._pm_pymatgen = ttk.Label(row2, text="", foreground=MUTED, wraplength=600,
                                      justify="left")
        self._pm_pymatgen.pack(side="left", padx=12)

        # Plot area
        self.patternmap_plot_frame = ttk.Frame(frame)
        self.patternmap_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.patternmap_plot_frame,
            text="Run the pipeline or Load pattern map to view the pattern waterfall.",
            foreground=MUTED,
        ).pack(anchor="center", expand=True)

    def load_pattern_map(self):
        """Render the pattern waterfall (and optional tracks/layers) from the analysis HDF5."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            return  # silently skip auto-calls

        # prev-figure-close leak guard
        prev = getattr(self, "_patternmap_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._patternmap_fig = None

        for w in self.patternmap_plot_frame.winfo_children():
            w.destroy()

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            self.ttk.Label(
                self.patternmap_plot_frame,
                text=f"matplotlib unavailable: {e}",
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            return

        import numpy as np

        from .heatmap import pattern_image, reflection_tracks, phase_layers
        from .phases import pymatgen_available

        # Update pymatgen hint label
        if hasattr(self, "_pm_pymatgen"):
            if pymatgen_available():
                self._pm_pymatgen.configure(text="", foreground=MUTED)
            else:
                self._pm_pymatgen.configure(
                    text=(
                        "pymatgen not installed — reflection tracks, phase layers, and the "
                        "simulated set are disabled (waterfall + ML export of measured data "
                        "still work)"
                    ),
                    foreground=WARN,
                )

        img = pattern_image(path, source=self._pm_source.get(), x_axis="frame")
        if not img["ok"]:
            self.ttk.Label(
                self.patternmap_plot_frame,
                text=img["error"],
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            if hasattr(self, "_pm_status"):
                self._pm_status.configure(text=img["error"])
            return

        show_layers = bool(self._pm_layers.get()) and pymatgen_available()

        if show_layers:
            fig = Figure(figsize=(8, 6), dpi=100, layout="constrained")
            ax = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
        else:
            fig = Figure(figsize=(8, 5), dpi=100, layout="constrained")
            ax = fig.add_subplot(1, 1, 1)
            ax2 = None

        fig.patch.set_facecolor(BG)
        self._patternmap_fig = fig

        # Waterfall
        Z = img["Z"]
        radial = img["radial"]
        n = img["n_frames"]

        pos = Z[np.isfinite(Z) & (Z > 0)]
        if pos.size:
            vmin = float(np.percentile(pos, 5))
            vmax = float(np.percentile(pos, 99))
        else:
            vmin = None
            vmax = None

        ax.imshow(
            Z, aspect="auto", origin="lower", cmap="magma",
            extent=[0, max(n - 1, 1), float(radial.min()), float(radial.max())],
            vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel("frame index")
        ax.set_ylabel(img["unit"] or "radial")
        ax.set_title(f"Pattern waterfall — {img['source']}", color=FG)
        self._style_ax(ax)

        # Reflection-track overlays
        if self._pm_tracks.get() and pymatgen_available():
            any_phase_plotted = False
            for phase_obj in self._enabled_phase_objects():
                tr = reflection_tracks(path, phase_obj)
                if not tr["ok"]:
                    continue
                phase_color = None
                first_track = True
                for track in tr["tracks"]:
                    centers = track["centers"]
                    if not np.any(np.isfinite(centers)):
                        continue
                    x_coords = np.arange(n, dtype=float)
                    if first_track:
                        (ln,) = ax.plot(
                            x_coords, centers, lw=0.6, alpha=0.7,
                            label=phase_obj.name,
                        )
                        phase_color = ln.get_color()
                        first_track = False
                        any_phase_plotted = True
                    else:
                        ax.plot(x_coords, centers, lw=0.6, alpha=0.7,
                                color=phase_color, label="_nolegend_")
            if any_phase_plotted:
                ax.legend(fontsize=7, framealpha=0.4)

        # Phase layers on the bottom axis
        if show_layers and ax2 is not None:
            pl = phase_layers(path, self._enabled_phase_objects())
            if pl["ok"]:
                for layer in pl["layers"]:
                    ax2.plot(
                        np.arange(layer["intensity"].size),
                        layer["intensity"],
                        lw=0.8, label=layer["name"],
                    )
                ax2.set_xlabel("frame index")
                ax2.set_ylabel("layer intensity (norm.)")
                handles2, _ = ax2.get_legend_handles_labels()
                if handles2:
                    ax2.legend(fontsize=7, framealpha=0.4)
            else:
                ax2.set_title(pl["error"], color=WARN)
            self._style_ax(ax2)

        self._patternmap_canvas = self._embed_figure(self.patternmap_plot_frame, fig)

        if hasattr(self, "_pm_status"):
            self._pm_status.configure(
                text=f"{img['n_frames']} frames × {radial.size} bins")

    def export_ml_clicked(self):
        """Export the analysis frames as an ML-ready .npz dataset."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.messagebox.showerror(
                "Export ML dataset",
                "No analysis HDF5 found. Run the pipeline or set the path on the Input tab.")
            return

        out = self.filedialog.asksaveasfilename(
            title="Export ML dataset",
            defaultextension=".npz",
            filetypes=[("NumPy npz", "*.npz")],
        )
        if not out:
            return

        from . import mldata
        try:
            man = mldata.export_ml_dataset(
                path, out,
                channels=("clean", "spot_residual"),
                normalize=True,
            )
        except Exception as e:
            self.messagebox.showerror("Export ML dataset failed", repr(e))
            return

        self.log(
            f"ML dataset exported: {man['n_frames']} frames × {man['n_channels']} channels "
            f"→ {out}  labels: {'yes' if man['has_labels'] else 'no'}"
        )
        self.messagebox.showinfo(
            "Export complete",
            f"{man['n_frames']} frames × {man['n_channels']} channels → {out}\n"
            f"Labels: {'yes' if man['has_labels'] else 'no (run Step 3a first)'}",
        )

    def export_sim_clicked(self):
        """Export a pressure-augmented simulated training set as .npz."""
        from .phases import pymatgen_available
        if not pymatgen_available():
            self.messagebox.showinfo(
                "Export simulated set",
                "pymatgen is required to simulate XRD patterns.\n"
                "Install it with:  pip install pymatgen")
            return

        phases = self._enabled_phase_objects()
        if not phases:
            self.messagebox.showinfo(
                "Export simulated set",
                "Enable candidate phases on the Phases tab first.")
            return

        out = self.filedialog.asksaveasfilename(
            title="Export simulated training set",
            defaultextension=".npz",
            filetypes=[("NumPy npz", "*.npz")],
        )
        if not out:
            return

        import numpy as np
        try:
            pmin = float(self.config.get("p_min", 0) or 0)
        except (ValueError, TypeError):
            pmin = 0.0
        try:
            pmax = float(self.config.get("p_max", 100) or 100)
        except (ValueError, TypeError):
            pmax = 100.0
        pressures = np.linspace(pmin, pmax, 21)

        from . import mldata
        try:
            man = mldata.export_simulated_dataset(out, phases, pressures=pressures)
        except Exception as e:
            self.messagebox.showerror("Export simulated set failed", repr(e))
            return

        self.log(
            f"Simulated dataset exported: {man['n_samples']} patterns "
            f"({len(man['phases'])} phases) → {out}"
        )
        self.messagebox.showinfo(
            "Export complete",
            f"{man['n_samples']} simulated patterns ({len(man['phases'])} phases) → {out}",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, confirm: bool = True) -> bool:
        """Save and tear down. Returns False if the user cancelled."""
        if self._run_proc is not None and self._run_proc.poll() is None:
            if confirm and not self.messagebox.askyesno(
                    "Analysis running",
                    "An analysis is still running. Terminate it and exit?"):
                return False
            self._run_proc.terminate()
        self._closing = True  # stop the log-drain poller from rescheduling
        self.save_config(silent=True)
        return True

    def on_close(self):
        if not self.shutdown(confirm=True):
            return
        if self._owns_root:
            self.root.destroy()


# ---------------------------------------------------------------------------
# Public factory / entry-point functions
# ---------------------------------------------------------------------------

def make_analysis_pane(
    parent_frame, config_path: "str | Path"
) -> "AnalysisApp":
    """Construct AnalysisApp embedded in a parent frame (for the unified app)."""
    return AnalysisApp(config_path, parent=parent_frame)


def run_app(config_path: "str | Path") -> int:
    """Standalone entry point."""
    from ..guikit.dpi import enable_hi_dpi
    enable_hi_dpi()
    app = AnalysisApp(config_path)
    assert app._owns_root, "run_app is the standalone entry point and must own the root"
    app.root.mainloop()
    return 0
