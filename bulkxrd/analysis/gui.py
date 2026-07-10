"""Tabbed Tkinter GUI for the analysis stage (Steps 1-3).

Workflow left-to-right across tabs — configure (1-6), run (7), inspect (8-11):
    1 Input       — pick the reduced HDF5, inspect its structure, see the analysis output
    2 Background  — SNIP baseline + spot-residual parameters
    3 Peaks       — pseudo-Voigt fitting parameters
    4 Phases      — reference-phase library: bundled + user phases, import CIFs, toggle candidates
    5 Frame meta  — per-frame conditions (P, T): parse filenames or import a CSV (Step-3 prior)
    6 Identify    — Step 3a/3b settings: EOS matching, metadata prior, ML candidate ranking
    7 Run         — launch the crash-isolated worker, watch progress + log
    8 Review      — single-frame QC: traces + fitted peaks + contamination curve
    9 Peak map    — scatter of fitted peak positions across the series
    10 Pattern map — waterfall / reflection tracks / per-phase intensity (needs Step 3a)
    11 Unknowns    — stacked Step-3c unknown-cluster diagram + exports
    12 Grid map    — per-frame scalars refolded onto a 2D scan grid (mapping runs)

Series plots (9-11) can use frame index, pressure, temperature, or elapsed
time as the independent variable.

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
from ..core.processes import terminate_process_tree, worker_popen
from ..guikit.theme import (
    BG, BG2, FG, ACCENT, ACCENT2, WARN, MUTED, ENTRY_BG,
    CLR_RAW, CLR_MSKD, CLR_SMTH, CLR_DIFF, CLR_REF,
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
        "A reduced_*.h5 from the reduction stage. Step 1 needs its "
        "intensity_robust channel; if it is missing, re-run reduction with "
        "the robust pattern enabled."
    ),
    "analysis_h5_file": (
        "Output analysis HDF5. Blank = <stem>_analysis.h5 beside the reduced file."
    ),
    # Run scope
    "run_step1": "Run Step 1: SNIP baseline + spot-residual separation.",
    "run_step2": "Run Step 2: pseudo-Voigt peak fitting on the selected Peak source.",
    "run_step3": (
        "Run Step 3a: match fitted peaks against the candidate phases via each "
        "phase's EOS. Gives per-frame phase confidence and, for pressure series, "
        "a per-frame pressure estimate."
    ),
    # Step 3a
    "identify_all_phases": (
        "Score every phase in the reference library (bundled + yours) against "
        "each frame instead of only the Phases-tab selection. Slower and more "
        "prone to spurious matches; use it when you don't know what is in the "
        "sample. This searches your library, not all of ICSD/MP."
    ),
    "p_min": "Lower bound (GPa) of the EOS pressure search. 0 for ambient work.",
    "p_max": (
        "Upper bound (GPa) of the EOS pressure search. Phases with an EOS "
        "validity ceiling (p_max in their entry) are capped there regardless."
    ),
    "rel_tol": (
        "Peak-match tolerance as a fraction of d-spacing (0.01 = 1%). Raise it "
        "if real lines just miss their match; too loose lets wrong phases match."
    ),
    "seen_conf": (
        "Confidence (0-1) above which a phase counts as present in a frame. "
        "Present phases are subtracted in the residual step. Default 0.5."
    ),
    "identify_wavelength": (
        "X-ray wavelength (Å). Only needed for 2θ data; blank = read from the "
        "reduced file's PONI. q-axis data never needs it."
    ),
    "intensity_k": (
        "Weight (0-1) of the intensity-agreement factor in the confidence. "
        "0 = match on positions only. Keep it low (default 0.3): texture and "
        "spotty rings distort measured intensities."
    ),
    "use_frame_temperature": (
        "Apply each frame's temperature (Frame metadata tab) to the predicted "
        "d-spacings of phases that define thermal expansion. Uncheck to treat "
        "all frames as ambient temperature."
    ),
    "unknown_tracking_axis": (
        "Axis used to link residual peaks into unknown tracks. same follows the "
        "Peaks tab's seed order; frame preserves collection order; "
        "pressure/temperature/time sort frames by that metadata and allow smooth "
        "peak drift along the chosen axis."
    ),
    "unknown_group_by": (
        "Keep independent series separate while tracking unknowns. same follows "
        "the Peaks tab's seed grouping. Use scan for datasets named with "
        "scan001/scan034-style tokens; use folder when each scan lives in its "
        "own directory."
    ),
    "unknown_axis_predictor": (
        "For pressure/temperature/time tracking, extrapolate the next peak center "
        "from the track's recent slope. Keep on for pressure-shifting unknowns."
    ),
    "unknown_link_tol_fwhm": (
        "Linking tolerance in fitted-peak widths. Raise if a real unknown line "
        "splits into short tracks; lower if nearby unrelated peaks are merging."
    ),
    "unknown_max_gap": (
        "How many missing ordered samples a track may skip before it is closed. "
        "With pressure tracking, samples are pressure-sorted frames."
    ),
    "unknown_max_axis_gap": (
        "Optional maximum physical-axis jump between linked observations: GPa for "
        "pressure, K for temperature, seconds for time. Blank = no physical cap."
    ),
    "unknown_min_frames": "Minimum distinct observations required to keep an unknown track.",
    "unknown_jaccard": (
        "Co-occurrence threshold for merging tracks into one unknown cluster. "
        "Higher = stricter clustering."
    ),
    # Step 1
    "max_half_window": (
        "Widest feature (bins) SNIP treats as background. Set to 1.5-2x the "
        "half-width of your broadest real peak. Too wide flattens broad peaks "
        "into the baseline. Default 40."
    ),
    "n_passes": "SNIP passes. 1 is enough in practice. Default 1.",
    "use_lls": (
        "Compress dynamic range (log-log-sqrt) before SNIP. Keep it on: without "
        "it the baseline overshoots under intense sharp peaks."
    ),
    "contamination_threshold": (
        "Flag frames whose spot-contamination score (sum of positive "
        "spot_residual) exceeds this value. Blank = no flagging."
    ),
    "robust_source": (
        "Spot-suppressed channel Step 1 builds on.\n"
        "robust = azimuthal median (default).\n"
        "straightened = cake de-waved median (patterns/intensity_straightened_"
        "robust). Run the reduce stage's Review → 'Write straightened 1D' first "
        "(needs saved cakes). Use it when the sample sat off the calibrant "
        "position and rings arrive as double-horned peaks; cake-less frames fall "
        "back to the ordinary median automatically."
    ),
    # Step 2
    "peak_source": (
        "Signal the peaks are fit on. auto (default) = the reduce-side sigmaclip "
        "channel if present, else hybrid. clean = azimuthal median minus "
        "baseline: cleanest, but drops intensity on spotty/textured/incomplete "
        "rings. hybrid = clean plus the broad part of (mean − median), rejecting "
        "narrow single-crystal spikes. sigmaclip = the reduce-side trimmed mean "
        "(best; enable it in reduction). mean keeps everything including "
        "diamond spots — diagnostic only. spots = fit (mean − median) itself: "
        "the SINGLE-CRYSTAL SAMPLE channel — a crystal's sparse reflections are "
        "rejected by every median-based channel exactly like diamond spots, and "
        "this is where they end up. If peaks you can see in the pattern are "
        "missing from the fit, try hybrid or mean; if the sample is a crystal, "
        "run a spots pass."
    ),
    "sensitivity": (
        "Preset for the detection knobs below (any left blank). conservative = "
        "fewer, cleaner peaks; sensitive = catches weak shoulders but more noise "
        "hits. Explicit values below override the preset. Default normal."
    ),
    "auto_range": (
        "When Fit min/max are blank, skip the beamstop ramp and the dead "
        "detector tail automatically (trims at most the outer ~15%). Uncheck to "
        "fit the full axis."
    ),
    "hybrid_spike_bins": (
        "Hybrid source only: mean-excess narrower than this many bins is "
        "removed as a single-crystal spike; broader excess is kept as texture. "
        "Default 5."
    ),
    "min_snr": (
        "Peak height threshold in noise-floor units. Lower = more peaks, more "
        "noise hits. Blank = preset (normal 5)."
    ),
    "min_prominence_snr": (
        "Peak prominence threshold in noise-floor units; controls whether a "
        "shoulder on a stronger peak counts as its own peak. Blank = preset "
        "(normal 2)."
    ),
    "window_factor": (
        "Fit-window half-width as a multiple of the estimated FWHM. Default 3."
    ),
    "max_chi2": (
        "Reduced χ² above which a fit is flagged bad. Default 25. Tighten for "
        "cleaner peak maps; loosen if good peaks are being rejected."
    ),
    "fit_min": (
        "Lower fit bound (q or 2θ). Set just above the beamstop onset — the "
        "low-angle ramp inflates the noise floor and hides weak peaks. "
        "Blank = auto range."
    ),
    "fit_max": (
        "Upper fit bound (q or 2θ). Set below the noisy detector tail. "
        "Blank = auto range."
    ),
    "edge_bins": (
        "Drop peaks within this many bins of either end of the pattern (edge "
        "artefacts). Blank = preset (normal 5)."
    ),
    "min_fwhm_bins": (
        "Reject peaks narrower than this many bins — a real peak spans several; "
        "1-bin spikes are noise. If real peaks trip this, the pattern is "
        "under-sampled: re-reduce with more bins (see the run log's npt "
        "recommendation). Blank = preset (normal 2)."
    ),
    "detrend_bins": (
        "Detection-only local-baseline window (bins): removes broad background "
        "SNIP left behind so weak peaks clear the noise threshold. Fitting "
        "still uses the un-detrended signal. Size it a few peak widths. "
        "0 = off. Default 81."
    ),
    "propagate_seeds": (
        "Seed each frame's detection with the previous frame's good peak "
        "centers, so a reflection keeps its identity as it drifts through the "
        "series (compression, heating). Keep on for series data."
    ),
    "seed_tracking_axis": (
        "Order used for peak-seed propagation (Step 2). frame uses collection "
        "order; pressure/temperature/time sort frames by metadata so seeds move "
        "along the physical scan. Step 3c's Unknown tracking can mirror this "
        "(its 'same')."
    ),
    "seed_group_by": (
        "Keep seed propagation inside independent series (Step 2). Use scan for "
        "scan001/scan034-style names, or folder when each scan lives in its own "
        "directory. Step 3c's Unknown grouping can mirror this (its 'same')."
    ),
    "seed_axis_predictor": (
        "For pressure/temperature/time seed order, shift seed centers by their "
        "recent drift before fitting the next frame. Keep on for pressure scans."
    ),
    "seed_max_axis_gap": (
        "Optional physical-axis jump that resets seed memory: GPa for pressure, "
        "K for temperature, seconds for time. Blank = no cap."
    ),
    # Step 3a metadata-prior knobs
    "use_pressure_prior": (
        "Confine each phase's pressure fit to the frame's metadata pressure "
        "± window instead of the full p_min-p_max search. This is the main "
        "accuracy control for pressure series: without it, a wrong phase can "
        "slide along pressure until a few lines coincide. Needs frame "
        "pressures (Frame metadata tab)."
    ),
    "pressure_window": (
        "Half-width (GPa) of the prior window when a frame has no per-frame "
        "uncertainty. 0.5-2 GPa is typical. Default 2."
    ),
    "pressure_sigma_k": (
        "When a frame has a pressure uncertainty (CSV import), the window is "
        "k·σ instead of the fixed value. Default 2."
    ),
    "marker_prior": (
        "No metadata pressures? Fit the marker-category phases first and reuse "
        "the best marker's per-frame pressure as the prior for everything else."
    ),
    "min_matched": (
        "Reflections a phase must match (one-to-one) to count as present. "
        "Guards against 1-2 line coincidences. Default 3."
    ),
    "allow_sparse": (
        "Let phases below Min matched still be subtracted in the residual "
        "(e.g. sparse pressure markers). Off by default."
    ),
    # Series plots / grid map
    "map_value": (
        "Per-frame scalar shown on the grid: integrated or max intensity of "
        "the fit source (optionally within the ROI below), contamination "
        "score, peak count, P, T, or one phase's matched-reflection intensity."
    ),
    "map_layout": (
        "How frames are placed on the grid. 'scan lines' uses the collection "
        "order plus the controls to the right (frames per line, direction, "
        "serpentine). 'coordinates' places each frame by its stage position "
        "(/frames/pos_x, pos_y — import them on the Frame meta tab via CSV "
        "or the frame headers); no other input needed."
    ),
    "map_line_len": (
        "Frames per scan line — how many frames the stage collected before "
        "turning (horizontal) or how many rows tall a column is (vertical). "
        "Not needed with the 'coordinates' layout."
    ),
    "map_order": (
        "horizontal = scan lines are rows of the map; vertical = scan lines "
        "are columns."
    ),
    "map_serpentine": (
        "Checked = boustrophedon (stage reverses direction every line). "
        "Unchecked = unidirectional raster (every line scans the same way)."
    ),
    "map_roi_min": "ROI lower bound on the radial axis for intensity values. Blank = full axis.",
    "map_roi_max": "ROI upper bound on the radial axis for intensity values. Blank = full axis.",
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
            # Workflow order: configure everything (input → background → peaks
            # → phases → frame metadata → identify), then run, then inspect the
            # results (review → peak map → pattern map → grid map). Result tabs
            # that need Step-3a output sit AFTER the tabs that configure it.
            ("1 Input",      self._tab_input),
            ("2 Background", self._tab_background),
            ("3 Peaks",      self._tab_peaks),
            ("4 Phases",     self._tab_phases),
            ("5 Frame meta", self._tab_frame_metadata),
            ("6 Identify",   self._tab_identify),
            ("7 Run",        self._tab_run),
            ("8 Review",     self._tab_review),
            ("9 Peak map",   self._tab_heatmap),
            ("10 Pattern map", self._tab_patternmap),
            ("11 Unknowns",   self._tab_unknowns),
            ("12 Grid map",  self._tab_gridmap),
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

    def field(self, parent, key, label, browse=None, row=None, width=80, col=0):
        """Entry field bound to a config key, with optional Browse button.

        ``col`` places the pair in a second (third, ...) column group so dense
        tabs can lay parameters out side by side instead of one tall stack."""
        tk, ttk = self.tk, self.ttk
        var = tk.StringVar(value=str(self.config.get(key, "")))
        self.vars[key] = var
        base = int(col) * 3
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=base, sticky="w", padx=4, pady=3)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=base + 1, sticky="we", padx=4, pady=3)
        if not hasattr(self, "entry_widgets"):
            self.entry_widgets: Dict[str, Any] = {}
        self.entry_widgets[key] = entry
        if browse:
            ttk.Button(
                parent, text="Browse",
                command=lambda: self.browse_into(key, browse),
            ).grid(row=row, column=base + 2, padx=4)
        txt = HELP.get(key, "")
        if txt:
            _ToolTip(lbl, txt)
            _ToolTip(entry, txt)
        parent.columnconfigure(base + 1, weight=1)

    def autowrap(self, label, pad=28):
        """Keep a long explanatory label's wraplength tracking its parent's
        width, so text reflows instead of running off the tab on narrow
        windows (fixed wraplengths clipped on small screens)."""
        def _fit(event, lbl=label):
            try:
                w = max(240, int(event.width) - pad)
                lbl.configure(wraplength=w)
            except Exception:
                pass
        label.master.bind("<Configure>", _fit, add="+")

    def checkbox(self, parent, key, label, row=None, col=0):
        """Checkbox bound to a boolean config key (``col`` = column group)."""
        tk, ttk = self.tk, self.ttk
        var = tk.BooleanVar(value=bool(self.config.get(key, False)))
        self.vars[key] = var
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.grid(row=row, column=int(col) * 3, columnspan=2, sticky="w", padx=4, pady=3)
        txt = HELP.get(key, "")
        if txt:
            _ToolTip(cb, txt)

    def combo(self, parent, key, label, values, row=None, width=16, default=""):
        """Read-only combobox bound to a config key."""
        tk, ttk = self.tk, self.ttk
        cur = str(self.config.get(key, default) or (default or (values[0] if values else "")))
        var = tk.StringVar(value=cur)
        self.vars[key] = var
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=4, pady=3)
        cb = ttk.Combobox(parent, textvariable=var, values=list(values),
                          state="readonly", width=width)
        cb.grid(row=row, column=1, sticky="w", padx=4, pady=3)
        txt = HELP.get(key, "")
        if txt:
            _ToolTip(lbl, txt)
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

    @staticmethod
    def _derive_analysis_output(reduced: str) -> str:
        """Default analysis output path beside the reduced input
        (``<stem>_analysis.h5``) — mirrors background.run_background_separation."""
        r = str(reduced or "").strip()
        if not r:
            return ""
        p = Path(r)
        return str(p.with_name(p.stem + "_analysis.h5"))

    def _autofill_analysis_output(self, *_):
        """Keep the output Analysis HDF5 in step with the reduced input.

        Fills it from the input's default when the field is blank or still holds
        the value we last auto-derived (so a path the user typed is never
        clobbered)."""
        if "reduced_h5_file" not in self.vars or "analysis_h5_file" not in self.vars:
            return
        reduced = str(self.vars["reduced_h5_file"].get() or "").strip()
        derived = self._derive_analysis_output(reduced)
        if not derived:
            return
        current = str(self.vars["analysis_h5_file"].get() or "").strip()
        if current == derived:
            self._auto_out_value = derived   # already matches → adopt as managed
            return
        if current and current != getattr(self, "_auto_out_value", ""):
            return  # user-customized — leave it alone
        self.vars["analysis_h5_file"].set(derived)
        self.config["analysis_h5_file"] = derived
        self._auto_out_value = derived

    def _tab_input(self, frame):
        tk, ttk = self.tk, self.ttk
        self.field(frame, "reduced_h5_file", "Reduced HDF5", browse="file", row=0, width=64)
        self.field(frame, "analysis_h5_file", "Analysis HDF5 (output)", browse="file",
                   row=1, width=64)
        # Auto-derive the output path whenever the reduced input changes (typed,
        # browsed, or handed off from the reduction stage).
        self.vars["reduced_h5_file"].trace_add("write", self._autofill_analysis_output)
        self._autofill_analysis_output()
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
        self.combo(frame, "robust_source", "Background source",
                   ["robust", "straightened"], row=7, width=14, default="robust")
        _bg_help = ttk.Label(
            frame,
            text=(
                "Step 1 splits each pattern into background, sample signal, and\n"
                "single-crystal contamination:\n\n"
                "  spot_residual = mean − median   (spots hit the mean, not the median)\n"
                "  baseline = SNIP(robust)         (iterative peak-clipping estimate)\n"
                "  clean = robust − baseline       (what the later steps build on)\n\n"
                "If broad peaks lose height after Step 1, the SNIP window is too\n"
                "wide — reduce Max half-window. The stored channels let every later\n"
                "step rebuild its own fit source, so nothing is lost."
            ),
            foreground=MUTED, justify="left", wraplength=640,
        )
        _bg_help.grid(row=8, column=0, columnspan=3, sticky="w", padx=6, pady=(12, 4))
        self.autowrap(_bg_help)

    # ------------------------------------------------------------------
    # Tab 3 — Peaks
    # ------------------------------------------------------------------

    def _tab_peaks(self, frame):
        tk, ttk = self.tk, self.ttk
        self.checkbox(frame, "run_step2", "Run Step 2 — pseudo-Voigt peak fitting", row=0)
        # Primary controls: fit source + sensitivity preset + auto range.
        self.combo(frame, "peak_source", "Peak source",
                   ["auto", "hybrid", "sigmaclip", "clean", "mean", "spots"],
                   row=1, default="auto")
        self.combo(frame, "sensitivity", "Sensitivity",
                   ["conservative", "normal", "sensitive"], row=2, default="normal")
        self.checkbox(frame, "auto_range", "Auto valid q/2θ range (blank Fit min/max)", row=3)
        ttk.Label(frame, text="Advanced (blank = follow the Sensitivity preset):",
                  foreground=MUTED).grid(row=4, column=0, columnspan=6, sticky="w",
                                         padx=4, pady=(10, 0))
        # Two column-groups so the tab fits on ~700px-tall screens.
        self.field(frame, "min_snr", "Min SNR (height)", row=5, width=12)
        self.field(frame, "min_prominence_snr", "Min prominence SNR", row=6, width=12)
        self.field(frame, "window_factor", "Window factor (× FWHM)", row=7, width=12)
        self.field(frame, "max_chi2", "Max reduced χ²", row=8, width=12)
        self.field(frame, "fit_min", "Fit 2θ/q min (blank=auto)", row=9, width=12)
        self.field(frame, "fit_max", "Fit 2θ/q max (blank=auto)", row=5, width=12, col=1)
        self.field(frame, "edge_bins", "Edge guard (bins)", row=6, width=12, col=1)
        self.field(frame, "min_fwhm_bins", "Min FWHM (bins)", row=7, width=12, col=1)
        self.field(frame, "hybrid_spike_bins", "Hybrid spike width (bins)", row=8, width=12, col=1)
        self.field(frame, "detrend_bins", "Detrend window (bins, 0=off)", row=9, width=12, col=1)
        self.checkbox(frame, "propagate_seeds",
                      "Propagate peak seeds frame-to-frame", row=10)
        seedrow = ttk.Frame(frame)
        seedrow.grid(row=11, column=0, columnspan=6, sticky="w", padx=4, pady=3)
        ttk.Label(seedrow, text="Seed order", foreground=MUTED).pack(side="left", padx=(0, 4))
        self.vars["seed_tracking_axis"] = tk.StringVar(
            value=str(self.config.get("seed_tracking_axis", "frame") or "frame"))
        _seed_axis = ttk.Combobox(
            seedrow, textvariable=self.vars["seed_tracking_axis"],
            values=["frame", "pressure", "temperature", "time"],
            state="readonly", width=11)
        _seed_axis.pack(side="left")
        _ToolTip(_seed_axis, HELP["seed_tracking_axis"])
        ttk.Label(seedrow, text="group by", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["seed_group_by"] = tk.StringVar(
            value=str(self.config.get("seed_group_by", "none") or "none"))
        _seed_group = ttk.Combobox(
            seedrow, textvariable=self.vars["seed_group_by"],
            values=["none", "scan", "folder"], state="readonly", width=8)
        _seed_group.pack(side="left")
        _ToolTip(_seed_group, HELP["seed_group_by"])
        self.vars["seed_axis_predictor"] = tk.BooleanVar(
            value=bool(self.config.get("seed_axis_predictor", True)))
        _seed_pred = ttk.Checkbutton(
            seedrow, text="predict drift", variable=self.vars["seed_axis_predictor"])
        _seed_pred.pack(side="left", padx=(10, 2))
        _ToolTip(_seed_pred, HELP["seed_axis_predictor"])
        ttk.Label(seedrow, text="axis gap", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["seed_max_axis_gap"] = tk.StringVar(
            value=str(self.config.get("seed_max_axis_gap", "")))
        _seed_gap = ttk.Entry(seedrow, textvariable=self.vars["seed_max_axis_gap"], width=7)
        _seed_gap.pack(side="left")
        _ToolTip(_seed_gap, HELP["seed_max_axis_gap"])
        _pk_help = ttk.Label(
            frame,
            text=(
                "Step 2 fits pseudo-Voigt profiles A·(η·Lorentzian + (1−η)·Gaussian) "
                "to the selected source. Start with Peak source, Sensitivity, and "
                "Auto range; leave the advanced fields blank unless a specific "
                "problem points at one (each tooltip says which). Common cases: "
                "visible peaks missing from the fit → source 'hybrid' or 'mean'; "
                "weak shoulders not detected → Sensitivity 'sensitive'; noise fitted "
                "as peaks → 'conservative'; stepped/blocky patterns → too few bins, "
                "re-reduce (see run log). Rejection flags: LOW_AMP=1, BAD_CHI2=2, "
                "CENTER_DRIFT=4, WIDTH_BOUND=8, NO_CONVERGE=16."
            ),
            foreground=MUTED, justify="left", wraplength=640,
        )
        _pk_help.grid(row=12, column=0, columnspan=6, sticky="w", padx=6, pady=(12, 4))
        self.autowrap(_pk_help)

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
        # Fresh run: clear any leftover cancelled/failed/done styling and state.
        self._cancel_requested = False
        self.progress_label.configure(text="Starting worker ...", foreground=MUTED)
        self._worker_status = "running"
        self._update_status_bar()

        def _worker_thread():
            try:
                proc = worker_popen(
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
            # A user-requested cancel also exits nonzero — don't call it a
            # failure or pop an error dialog for a deliberate act. Say what
            # state things are in and what Run will do next.
            if getattr(self, "_cancel_requested", False):
                self._cancel_requested = False
                self._worker_status = "cancelled"
                self._update_status_bar()
                self.progress_label.configure(
                    text="Cancelled — completed steps were saved to the analysis "
                         "file; interrupted steps left no partial output (atomic "
                         "writes). Run analysis re-runs every enabled step.",
                    foreground=MUTED)
                self.log("Run cancelled by user.", "WARN")
                return
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
            ("peak map", self.load_heatmap),
            ("identify", self.load_identify),
            ("pattern map", self.load_pattern_map),
            ("unknowns", self.load_unknowns),
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
            self._cancel_requested = True
            self.cancel_btn.configure(state="disabled")
            self.progress_label.configure(text="Cancelling ...", foreground=MUTED)
            terminate_process_tree(proc)
            self.log("Cancel requested — stopped worker process tree", "WARN")

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
        ttk.Button(ctrl, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_review_fig", None), "Review")
                   ).pack(side="left", padx=4)
        _rev_exp = ttk.Button(ctrl, text="Export frame…",
                              command=self.review_export_frame_clicked)
        _rev_exp.pack(side="left", padx=4)
        _ToolTip(_rev_exp, (
            "Export the frame shown here: its pattern as a two-column .xy "
            "(channel of your choice) and optionally its fitted peaks as "
            "peaks.csv. Select several frames on the Frame meta tab to "
            "export a batch."))

        # Trace toggles live on their own row: one long row of controls used
        # to run wider than the window and clip on the right.
        togglerow = ttk.Frame(frame)
        togglerow.pack(fill="x", pady=(0, 4))

        ttk.Label(ctrl, text="Frame:", foreground=MUTED).pack(side="left", padx=(12, 2))
        self._review_idx_var = tk.IntVar(value=0)
        # NOTE: the Scale is deliberately NOT linked to _review_idx_var. When the
        # Scale and the Spinbox shared the variable, the Scale's callback echoed a
        # var.set() back into the Spinbox mid-arrow-press, and the press applied
        # its increment twice (one click advanced two frames). The slider now
        # drives the var only through _on_review_slider (change-guarded), and the
        # spinbox/render paths sync the slider explicitly.
        self._review_scale = ttk.Scale(
            ctrl, from_=0, to=0, orient="horizontal", length=200,
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
        self._show_residual = tk.BooleanVar(value=False)
        self._show_peaks = tk.BooleanVar(value=True)
        self._show_residual_peaks = tk.BooleanVar(value=False)
        self._show_unknowns = tk.BooleanVar(value=False)
        self._show_cake = tk.BooleanVar(value=False)
        for var, label in [
            (self._show_mean, "mean"),
            (self._show_robust, "robust"),
            (self._show_baseline, "baseline"),
            (self._show_clean, "clean"),
            (self._show_spot, "spot_residual"),
            (self._show_residual, "residual"),
            (self._show_peaks, "fitted peaks"),
            (self._show_residual_peaks, "residual peaks"),
            (self._show_unknowns, "unknown peaks"),
            (self._show_cake, "cake (2D)"),
        ]:
            cb = ttk.Checkbutton(togglerow, text=label, variable=var,
                                 command=self._schedule_review_render)
            cb.pack(side="left", padx=2)
            if label == "residual":
                _ToolTip(cb, "Step-3a removal result: /residual/clean.")
            elif label == "residual peaks":
                _ToolTip(cb, "Peaks re-fitted on /residual/clean.")
            elif label == "unknown peaks":
                _ToolTip(cb, "Step-3c unknown-track observations for this frame.")
        self._review_source_reduced = ""

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
        self._review_source_reduced = str(info.get("source_reduced", "") or "")
        contam = info.get("contamination")
        self._review_contamination = contam
        if nf > 0:
            self._frame_count = nf
            self._review_scale.configure(to=max(nf - 1, 0))
            self._review_spinbox.configure(to=max(nf - 1, 0))
        self._update_status_bar()
        self._render_review(int(self._review_idx_var.get()))

    def _on_review_slider(self, value):
        """Called on every slider tick — snap to an int frame and debounce.

        Change-guarded: writing the var only when the frame actually changes is
        what keeps _sync_review_scale() below loop-free (scale.set fires this
        callback once, sees no change, stops)."""
        try:
            idx = int(round(float(value)))
        except (ValueError, TypeError):
            return
        try:
            if int(self._review_idx_var.get() or 0) == idx:
                return
            self._review_idx_var.set(idx)
        except Exception:
            return
        self._schedule_review_render()

    def _sync_review_scale(self):
        """Move the slider to the spinbox's frame (guarded, see slider callback)."""
        try:
            self._review_scale.set(int(self._review_idx_var.get() or 0))
        except Exception:
            pass

    def _on_review_spinbox(self):
        self._sync_review_scale()
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
        self._sync_review_scale()   # typed entry / programmatic changes move the slider too
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
        show_cake = bool(self._show_cake.get())
        fig = Figure(figsize=(7, 6), dpi=100, layout="constrained")
        self._review_fig = fig
        fig.patch.set_facecolor(BG)
        if show_cake:
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1])
            ax1 = fig.add_subplot(gs[0])    # pattern traces
            ax_cake = fig.add_subplot(gs[1])  # 2D cake (azimuth × radial)
            ax2 = fig.add_subplot(gs[2])    # contamination strip
        else:
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax1 = fig.add_subplot(gs[0])   # pattern traces get the bulk of the height
            ax2 = fig.add_subplot(gs[1])   # contamination strip below
            ax_cake = None

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
        if self._show_residual.get():
            _plot(ax1, fd.get("residual"), "residual", CLR_REF, lw=1.0, alpha=0.9)

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

        residual_peaks = fd.get("residual_peaks", [])
        if self._show_residual_peaks.get() and residual_peaks:
            for k, p in enumerate(residual_peaks):
                ax1.axvline(
                    p["center"], color=CLR_REF, lw=0.9, alpha=0.75,
                    linestyle="--",
                    label="residual peaks" if k == 0 else None,
                )

        unknown_obs = fd.get("unknown_obs", [])
        if self._show_unknowns.get() and unknown_obs:
            for k, p in enumerate(unknown_obs):
                ax1.axvline(
                    p["center"], color=WARN, lw=1.0, alpha=0.85,
                    linestyle=":",
                    label="unknown peaks" if k == 0 else None,
                )

        fname = Path(fd.get("filename", "")).name or f"frame {frame_index}"
        ax1.set_title(f"{fname}  [frame {frame_index}]", color=FG)
        ax1.set_xlabel(unit)
        ax1.set_ylabel("intensity")
        if ax1.get_legend_handles_labels()[1]:
            ax1.legend(fontsize=7, framealpha=0.4)
        self._style_ax(ax1)

        # Optional middle axis: the 2D cake (azimuth × radial) for this frame,
        # read from the source reduced file (cakes don't live in the analysis file).
        if ax_cake is not None:
            from .review import cake_for_frame
            ck = cake_for_frame(self._review_source_reduced, frame_index)
            if ck.get("ok") and ck.get("cake") is not None:
                cake = np.asarray(ck["cake"], dtype=float)
                cr, caz = ck.get("radial"), ck.get("azimuthal")
                extent = None
                if cr is not None and caz is not None and cr.size and caz.size:
                    extent = [float(np.min(cr)), float(np.max(cr)),
                              float(np.min(caz)), float(np.max(caz))]
                cc = cake.copy()
                cc[~np.isfinite(cc) | (cc <= 0)] = np.nan
                finite = cc[np.isfinite(cc)]
                vmin = float(np.percentile(finite, 5)) if finite.size else None
                vmax = float(np.percentile(finite, 99)) if finite.size else None
                ax_cake.imshow(cc, aspect="auto", origin="lower", cmap="magma",
                               extent=extent, vmin=vmin, vmax=vmax)
                ax_cake.set_xlabel(ck.get("unit") or unit)
                ax_cake.set_ylabel("azimuth (deg)")
                ax_cake.set_title("Cake (2D)", color=FG)
            else:
                ax_cake.set_title(f"Cake: {ck.get('error', 'unavailable')}", color=WARN)
                ax_cake.set_xticks([]); ax_cake.set_yticks([])
            self._style_ax(ax_cake)

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

    def _toggle_identify_help(self):
        """Show/hide the Identify instructions so the plot can use the space."""
        if self._identify_help_var.get():
            self._identify_help.grid()
        else:
            self._identify_help.grid_remove()

    def _open_plot_window(self, fig, title="Plot"):
        """Open ``fig`` in a separate, large, resizable window with its own
        toolbar. The figure is shared with the inline canvas (these plots are
        static, rendered once per load), so the pop-out is just a bigger view;
        reloading the tab refreshes both."""
        if fig is None:
            self.messagebox.showinfo("Open in window", "Load a plot first.")
            return
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            win = self.tk.Toplevel(self.root)
            win.title(f"Bulk-XRD — {title}")
            try:
                win.configure(bg=BG)
                win.geometry("1100x800")
            except Exception:
                pass
            canvas = FigureCanvasTkAgg(fig, master=win)
            self._add_nav_toolbar(canvas, win)
            w = canvas.get_tk_widget()
            w.configure(width=10, height=10)
            w.pack(side="top", fill="both", expand=True)
            canvas.draw()
        except Exception as e:
            self.messagebox.showerror("Open in window", f"Could not open: {e!r}")

    def _attach_hover(self, canvas, status_label):
        """Live cursor read-out into ``status_label``, restoring its text on
        leave. Call AFTER the label's base text is set. The x value is shown
        as-is (frame index, pressure, ... — whatever the plot's x-axis is)."""
        if canvas is None or status_label is None:
            return
        base = {"text": status_label.cget("text")}
        def _move(event):
            if event.inaxes is not None and event.xdata is not None:
                if event.ydata is not None:
                    status_label.configure(
                        text=f"{base['text']}   |   x={event.xdata:.6g}, y={event.ydata:.4g}")
                else:
                    status_label.configure(text=f"{base['text']}   |   x={event.xdata:.6g}")
        def _leave(event):
            status_label.configure(text=base["text"])
        try:
            canvas.mpl_connect("motion_notify_event", _move)
            canvas.mpl_connect("axes_leave_event", _leave)
        except Exception:
            pass

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
        # its real allocated size, then size the figure to it. The poll must
        # tolerate the widget being destroyed mid-flight (a reload replaces the
        # canvas while an earlier poll is still scheduled).
        def _initial_fit(tries=0, widget=widget):
            try:
                if not widget.winfo_exists():
                    return
                if _apply_size(widget.winfo_width(), widget.winfo_height()):
                    return
            except self.tk.TclError:
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
    # Tab 9 — Peak map (fitted peak positions across the series)
    # ------------------------------------------------------------------

    def _tab_heatmap(self, frame):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(0, 4))
        ttk.Button(top, text="Load peak map", command=self.load_heatmap).pack(
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
        ttk.Label(top, text="X axis:", foreground=MUTED).pack(side="left", padx=(12, 2))
        self._heatmap_xaxis = ttk.Combobox(
            top, values=["frame", "pressure", "temperature", "time"],
            state="readonly", width=11)
        self._heatmap_xaxis.set("frame")
        self._heatmap_xaxis.pack(side="left", padx=2)
        self._heatmap_xaxis.bind("<<ComboboxSelected>>",
                                 lambda e: self.load_heatmap())
        ttk.Button(top, text="Refresh", command=self.load_heatmap).pack(
            side="left", padx=4)
        ttk.Button(top, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_heatmap_fig", None), "Peak map")
                   ).pack(side="left", padx=4)

        self.heatmap_status = ttk.Label(top, text="", foreground=MUTED)
        self.heatmap_status.pack(side="left", padx=12)

        self.heatmap_plot_frame = ttk.Frame(frame)
        self.heatmap_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.heatmap_plot_frame,
            text="Load the analysis HDF5 to display the peak map.",
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

        # Map the frame index onto the chosen independent variable.
        x_kind = getattr(self._heatmap_xaxis, "get", lambda: "frame")() or "frame"
        x_arr, x_label = frame_arr, "frame index"
        if x_kind != "frame":
            from .heatmap import series_axis
            sx = series_axis(path, x_kind)
            if not sx["ok"]:
                self.ttk.Label(self.heatmap_plot_frame, text=sx["error"],
                               foreground=WARN).pack(anchor="center", expand=True)
                if hasattr(self, "heatmap_status"):
                    self.heatmap_status.configure(text=sx["error"])
                return
            xv = np.asarray(sx["x"], dtype=float)
            idx = frame_arr.astype(int)
            ok = (idx >= 0) & (idx < xv.size)
            x_arr = np.full(frame_arr.size, np.nan)
            x_arr[ok] = xv[idx[ok]]
            x_label = sx["label"]
            keep = np.isfinite(x_arr)
            x_arr, center_arr, c_arr = x_arr[keep], center_arr[keep], c_arr[keep]

        n_pts = int(center_arr.size)
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
                x_arr, center_arr, c=c_arr,
                cmap="viridis", s=28, alpha=0.9, norm=norm,
                edgecolors=FG, linewidths=0.4,
            )
            try:
                cb = fig.colorbar(sc, ax=ax, label=color_by)
                self._style_colorbar(cb)
            except Exception:
                pass
            ax.set_xlabel(x_label, color=FG)
            ax.set_ylabel(f"peak center ({unit})", color=FG)
            ax.set_title(f"Peak map — {n_pts} peaks", color=FG)

        self._heatmap_canvas = self._embed_figure(self.heatmap_plot_frame, fig)
        self._attach_hover(self._heatmap_canvas, self.heatmap_status)

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
        self._refresh_gridmap_values()

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
        self._refresh_gridmap_values()

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
        from .phases import EOS_TYPES, _eos_norm_type
        ttk.Label(content, text="EOS").grid(row=row, column=0, sticky="w", **pad)
        eos_frame = ttk.Frame(content)
        eos_frame.grid(row=row, column=1, columnspan=3, sticky="w", **pad)
        ex_eos = (existing.eos or {}) if existing else {}
        ttk.Label(eos_frame, text="type").grid(row=0, column=0, sticky="e", padx=(0, 1))
        v_eos_type = tk.StringVar(value=_eos_norm_type(ex_eos) if ex_eos else "BM3")
        ttk.Combobox(eos_frame, textvariable=v_eos_type, values=list(EOS_TYPES),
                     state="readonly", width=10).grid(row=0, column=1, padx=(0, 4))
        # K0'' only used by BM4; left blank otherwise.
        eos_keys = ("V0", "K0", "K0p", "K0pp")
        eos_labels = ("V0 (Å³)", "K0 (GPa)", "K0'", "K0'' (1/GPa, BM4)")
        eos_vars: "Dict[str, tk.StringVar]" = {}
        for i, (k, lbl) in enumerate(zip(eos_keys, eos_labels)):
            v_raw = ex_eos.get(k)
            ex_val = f"{v_raw:g}" if v_raw is not None else ""
            ttk.Label(eos_frame, text=lbl).grid(row=1, column=i * 2,
                                                sticky="e", padx=(6 if i else 0, 1))
            sv = tk.StringVar(value=ex_val)
            ttk.Entry(eos_frame, textvariable=sv, width=10).grid(
                row=1, column=i * 2 + 1, padx=(0, 4))
            eos_vars[k] = sv
        ttk.Label(eos_frame,
                  text="V0 optional (cancels in scaling); only K0 is required. "
                       "Forms: BM2/BM3/BM4, Vinet, Murnaghan.",
                  foreground=MUTED).grid(row=2, column=0, columnspan=8, sticky="w", pady=(2, 0))
        row += 1

        # Axial (anisotropic) EOS — optional, per-axis K0/K0' for non-cubic phases.
        ttk.Label(content, text="Axial EOS").grid(row=row, column=0, sticky="nw", **pad)
        ax_frame = ttk.Frame(content)
        ax_frame.grid(row=row, column=1, columnspan=3, sticky="w", **pad)
        ex_ax = (existing.axial_eos or {}) if existing else {}
        ttk.Label(ax_frame, text="axis").grid(row=0, column=0, padx=(0, 4))
        ttk.Label(ax_frame, text="K0 (GPa)").grid(row=0, column=1, padx=2)
        ttk.Label(ax_frame, text="K0'").grid(row=0, column=2, padx=2)
        axial_vars: "Dict[str, tuple]" = {}
        for i, axis in enumerate(("a", "b", "c")):
            e = ex_ax.get(axis) if isinstance(ex_ax.get(axis), dict) else {}
            k0 = e.get("K0"); kp = e.get("K0p")
            ttk.Label(ax_frame, text=axis).grid(row=i + 1, column=0, sticky="e", padx=(0, 4))
            v_k0 = tk.StringVar(value=f"{k0:g}" if k0 is not None else "")
            v_kp = tk.StringVar(value=f"{kp:g}" if kp is not None else "")
            ttk.Entry(ax_frame, textvariable=v_k0, width=10).grid(row=i + 1, column=1, padx=2)
            ttk.Entry(ax_frame, textvariable=v_kp, width=8).grid(row=i + 1, column=2, padx=2)
            axial_vars[axis] = (v_k0, v_kp)
        ttk.Label(ax_frame,
                  text="Optional. Fill per-axis K0 (on the cubed axis length, "
                       "PASCal/EosFit convention) for anisotropic compression; "
                       "blank axes fall back to the volume EOS. b inherits a if equal.",
                  foreground=MUTED, wraplength=420, justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))
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

            eos: "Dict[str, Any]" = {"type": v_eos_type.get() or "BM3"}
            for k in eos_keys:
                fv = _f(eos_vars[k].get())
                if fv is not None:
                    eos[k] = fv

            # Per-axis EOS (only axes with a K0 entered); same form as the main EOS.
            axial_eos: "Dict[str, Any]" = {}
            for axis, (v_k0, v_kp) in axial_vars.items():
                k0 = _f(v_k0.get())
                if k0 is not None:
                    ae = {"type": v_eos_type.get() or "BM3", "K0": k0}
                    kp = _f(v_kp.get())
                    if kp is not None:
                        ae["K0p"] = kp
                    axial_eos[axis] = ae

            notes = notes_text.get("1.0", "end-1c")

            phase = Phase(
                name=name,
                formula=v_formula.get().strip(),
                category=v_category.get(),
                space_group=v_sg.get().strip(),
                lattice=lattice,
                atoms=(existing.atoms if existing else []),
                eos=eos,
                axial_eos=axial_eos,
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
    # Tab 8 — Frame metadata (pressure prior)
    # ------------------------------------------------------------------

    def _tab_frame_metadata(self, frame):
        ttk = self.ttk

        ttk.Label(
            frame, text="Frame metadata — series conditions (P, T)",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w", padx=6, pady=(4, 0))
        _fm_sub = ttk.Label(
            frame,
            text=(
                "Per-frame pressure and temperature feed the Step-3 prior and "
                "the series plots. Populate them here (filenames, CSV, or by hand)."
            ),
            foreground=MUTED, justify="left", wraplength=760,
        )
        _fm_sub.pack(anchor="w", padx=6, pady=(0, 6))
        self.autowrap(_fm_sub)

        # Controls row
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(0, 4))
        ttk.Button(ctrl, text="Extract from filenames",
                   command=self.extract_pressures_clicked).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Import CSV…",
                   command=self.import_pressure_csv_clicked).pack(side="left", padx=4)
        _hdr_btn = ttk.Button(ctrl, text="Read X/Y from headers…",
                              command=self.import_positions_clicked)
        _hdr_btn.pack(side="left", padx=4)
        _ToolTip(_hdr_btn, (
            "Mapping scans: read per-frame stage positions from the raw frame "
            "files' headers (EDF/CBF motor entries) into /frames/pos_x, pos_y — "
            "the Grid map's 'coordinates' layout then places frames "
            "automatically."))
        ttk.Button(ctrl, text="Preview pressure vs frame",
                   command=self.preview_pressure_clicked).pack(side="left", padx=4)

        self._fm_status = ttk.Label(frame, text="", foreground=MUTED)
        self._fm_status.pack(anchor="w", padx=6, pady=(0, 4))

        # Editable per-frame metadata table
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="x", padx=6, pady=(0, 4))

        fm_cols = ("frame", "file", "pressure", "sigma", "temp", "src")
        self._fm_table = ttk.Treeview(tree_frame, columns=fm_cols, show="headings",
                                      height=10, selectmode="extended")
        fm_vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                               command=self._fm_table.yview)
        self._fm_table.configure(yscrollcommand=fm_vsb.set)
        fm_vsb.pack(side="right", fill="y")
        self._fm_table.pack(side="left", fill="both", expand=True)

        for c, txt, w, anc, stretch in (
            ("frame", "Frame", 60, "center", False),
            ("file", "Filename", 220, "w", True),
            ("pressure", "P (GPa)", 80, "center", False),
            ("sigma", "σ (GPa)", 80, "center", False),
            ("temp", "T (K)", 80, "center", False),
            ("src", "Src", 50, "center", False),
        ):
            self._fm_table.heading(c, text=txt)
            self._fm_table.column(c, width=w, minwidth=40, anchor=anc, stretch=stretch)
        self._fm_table.tag_configure("user", foreground=ACCENT)

        # Editor row
        editor = ttk.Frame(frame)
        editor.pack(fill="x", padx=6, pady=(0, 2))
        ttk.Label(editor, text="P (GPa):", foreground=MUTED).pack(side="left", padx=(0, 2))
        self._fm_edit_p = self.tk.StringVar(value="")
        ttk.Entry(editor, textvariable=self._fm_edit_p, width=10).pack(side="left", padx=(0, 8))
        ttk.Label(editor, text="σ:", foreground=MUTED).pack(side="left", padx=(0, 2))
        self._fm_edit_sig = self.tk.StringVar(value="")
        ttk.Entry(editor, textvariable=self._fm_edit_sig, width=10).pack(side="left", padx=(0, 8))
        ttk.Label(editor, text="T (K):", foreground=MUTED).pack(side="left", padx=(0, 2))
        self._fm_edit_t = self.tk.StringVar(value="")
        ttk.Entry(editor, textvariable=self._fm_edit_t, width=10).pack(side="left", padx=(0, 8))
        ttk.Button(editor, text="Apply to selected",
                   command=self.fm_apply_selected_clicked).pack(side="left", padx=4)
        ttk.Button(editor, text="Refresh table",
                   command=self.fm_refresh_table_clicked).pack(side="left", padx=4)
        _exp_btn = ttk.Button(editor, text="Export selected…",
                              command=self.fm_export_selected_clicked)
        _exp_btn.pack(side="left", padx=(12, 4))
        _ToolTip(_exp_btn, (
            "Export the selected frame(s): the chosen reduction/fit channel "
            "as two-column .xy patterns (native axis always; 2θ too when the "
            "wavelength is known) and, optionally, every fitted peak of those "
            "frames as one peaks.csv (center/amplitude/fwhm ± esd, eta, area, "
            "chi2, flag, phase). Rietveld hand-off is the separate "
            "bulkxrd-export-refinement bundle."))

        _fm_hint = ttk.Label(
            frame,
            text=("Select frame(s), enter values (blank = leave unchanged), Apply. "
                  "Applied values are marked 'user' — filename re-parsing and "
                  "Step-1 re-runs will not overwrite them."),
            foreground=MUTED, wraplength=760, justify="left",
        )
        _fm_hint.pack(anchor="w", padx=6, pady=(0, 4))
        self.autowrap(_fm_hint)

        self.fm_plot_frame = ttk.Frame(frame)
        self.fm_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.fm_plot_frame,
            text="Extract from filenames or Import CSV to populate frame pressures.",
            foreground=MUTED,
        ).pack(anchor="center", expand=True)

        _fm_csv = ttk.Label(
            frame,
            text=(
                "CSV columns: `frame` (0-based) or `filename`, plus any of "
                "`pressure_gpa`, `pressure_sigma_gpa`, `temperature_K`, "
                "`pos_x_mm`, `pos_y_mm`. Step 1 also auto-parses pressures from "
                "filenames (e.g. sample-1p5GPa → 1.5). Use this tab to override "
                "those or to enter gauge readings (ruby, membrane, thermocouple)."
            ),
            foreground=MUTED, justify="left", wraplength=700,
        )
        _fm_csv.pack(anchor="w", padx=6, pady=(4, 4))
        self.autowrap(_fm_csv)

    def extract_pressures_clicked(self):
        from .frame_metadata import extract_to_analysis, import_csv_to_analysis, read_frame_metadata
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        if not path or not Path(path).is_file():
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text="Run Step 1 first (no analysis file yet).")
            return
        try:
            result = extract_to_analysis(path)
            summary = result.get("summary", {})
            n_parsed = summary.get("n_parsed", 0)
            n_frames = summary.get("n_frames", 0)
            p_min = summary.get("p_min")
            p_max = summary.get("p_max")
            p_range = (
                f"P {p_min:.2f}–{p_max:.2f} GPa"
                if p_min is not None and p_max is not None
                else "P unknown"
            )
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text=(
                        f"Parsed {n_parsed}/{n_frames} frames from filenames "
                        f"({p_range})."
                    )
                )
            self._draw_pressure_preview(path)
            try:
                self.fm_refresh_table_clicked()
            except Exception:
                pass
        except Exception as e:
            self.log(f"extract_to_analysis failed: {e!r}", "WARN")
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text=str(e))

    def import_pressure_csv_clicked(self):
        from .frame_metadata import extract_to_analysis, import_csv_to_analysis, read_frame_metadata
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        if not path or not Path(path).is_file():
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text="Run Step 1 first (no analysis file yet).")
            return
        csv_path = self.filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("All", "*.*")]
        )
        if not csv_path:
            return
        try:
            result = import_csv_to_analysis(path, csv_path)
            summary = result.get("summary", {})
            csv_info = result.get("csv", {})
            n_parsed = summary.get("n_parsed", 0)
            n_frames = summary.get("n_frames", 0)
            p_min = summary.get("p_min")
            p_max = summary.get("p_max")
            p_range = (
                f"P {p_min:.2f}–{p_max:.2f} GPa"
                if p_min is not None and p_max is not None
                else "P unknown"
            )
            cols = csv_info.get("columns", [])
            n_rows = csv_info.get("n_rows", 0)
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text=(
                        f"Imported {n_rows}-row CSV ({', '.join(cols)}): "
                        f"{n_parsed}/{n_frames} frames have pressure ({p_range})."
                    )
                )
            self._draw_pressure_preview(path)
            try:
                self.fm_refresh_table_clicked()
            except Exception:
                pass
        except Exception as e:
            self.log(f"import_csv_to_analysis failed: {e!r}", "WARN")
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text=str(e))

    def _do_export_frames(self, indices, out_dir, *, source="fit", peaks=True,
                          residual_unknowns=True, status_label=None):
        """Run the frame export (patterns + optional peaks.csv). Returns the
        manifest, or None on failure (logged + status)."""
        status = status_label or getattr(self, "_fm_status", None)
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            if status is not None:
                status.configure(text="Run Step 1 first (no analysis file yet).")
            return None
        from .refine_export import export_frames
        try:
            man = export_frames(path, out_dir, frames=indices,
                                source=source, peaks=peaks,
                                residual_peaks=residual_unknowns,
                                unknowns=residual_unknowns)
        except Exception as e:
            self.log(f"Frame export failed: {e!r}", "WARN")
            if status is not None:
                status.configure(text=f"Export failed: {e}")
            return None
        msg = (f"Exported {man['n_frames']} frame(s) ({man['source']}) "
               f"+ {man['n_peaks']} peak row(s)")
        extra = []
        if man.get("n_residual_peaks"):
            extra.append(f"{man['n_residual_peaks']} residual peak row(s)")
        if man.get("n_unknown_obs"):
            extra.append(f"{man['n_unknown_obs']} unknown row(s)")
        if extra:
            msg += " + " + " + ".join(extra)
        msg += f" -> {out_dir}"
        if status is not None:
            status.configure(text=msg)
        self.log(msg)
        return man

    def _export_frames_dialog(self, indices):
        """Small options dialog (channel, peaks.csv, destination), then export."""
        tk, ttk = self.tk, self.ttk
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self._fm_status.configure(text="Run Step 1 first (no analysis file yet).")
            return
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Export {len(indices)} frame(s)")
        dlg.configure(bg=BG)
        dlg.transient(self.root)
        dlg.grab_set()
        content = ttk.Frame(dlg, padding=10)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text=(
            "Writes each frame as a two-column .xy pattern (native axis "
            "always; 2θ too when the wavelength is known) and optional CSVs "
            "for fitted, residual, and unknown peaks."),
            foreground=MUTED, wraplength=430, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(content, text="Pattern channel").grid(row=1, column=0,
                                                        sticky="w", pady=2)
        v_src = tk.StringVar(value="fit")
        _src = ttk.Combobox(content, textvariable=v_src, state="readonly",
                            width=12,
                            values=["fit", "clean", "mean", "hybrid",
                                    "sigmaclip", "robust", "residual"])
        _src.grid(row=1, column=1, sticky="w")
        _ToolTip(_src, "fit = the channel Step 2 actually fitted (default). "
                       "residual = /residual/clean after phase subtraction. "
                       "The other entries are reduction-side channels "
                       "reconstructed exactly as the pipeline does.")
        v_peaks = tk.BooleanVar(value=True)
        ttk.Checkbutton(content, text="Include fitted peaks (peaks.csv)",
                        variable=v_peaks).grid(row=2, column=0, columnspan=2,
                                               sticky="w", pady=2)
        v_resunk = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            content,
            text="Include residual/unknown peaks when available",
            variable=v_resunk,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(content, text="Destination").grid(row=4, column=0,
                                                    sticky="w", pady=2)
        v_dir = tk.StringVar(value=str(self.config.get("export_frames_dir", "")))
        ttk.Entry(content, textvariable=v_dir, width=36).grid(
            row=4, column=1, sticky="we")

        def _browse():
            d = self.filedialog.askdirectory(title="Export destination folder")
            if d:
                v_dir.set(d)
        ttk.Button(content, text="Browse", command=_browse).grid(
            row=4, column=2, padx=4)

        def _go():
            dest = v_dir.get().strip()
            if not dest:
                self.messagebox.showerror("Export frames",
                                          "Pick a destination folder.",
                                          parent=dlg)
                return
            self.config["export_frames_dir"] = dest
            self.save_config(silent=True)
            dlg.destroy()
            self._do_export_frames(indices, dest, source=v_src.get(),
                                   peaks=bool(v_peaks.get()),
                                   residual_unknowns=bool(v_resunk.get()))

        btns = ttk.Frame(content)
        btns.grid(row=5, column=0, columnspan=3, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="Export", command=_go).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left",
                                                                  padx=4)
        content.columnconfigure(1, weight=1)

    def review_export_frame_clicked(self):
        """Export the frame currently shown on the Review tab."""
        try:
            idx = int(self._review_idx_var.get())
        except (ValueError, TypeError):
            idx = 0
        self._export_frames_dialog([idx])

    def fm_export_selected_clicked(self):
        """Export the frames selected in the Frame meta table."""
        self.pull_vars()
        tbl = getattr(self, "_fm_table", None)
        sel = list(tbl.selection()) if tbl is not None else []
        if not sel:
            self._fm_status.configure(
                text="Select one or more frames in the table first.")
            return
        try:
            indices = sorted(int(iid) for iid in sel)
        except ValueError:
            return
        self._export_frames_dialog(indices)

    def import_positions_clicked(self):
        """Dialog: read per-frame stage positions from the raw frames' headers."""
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        if not path or not Path(path).is_file():
            self._fm_status.configure(text="Run Step 1 first (no analysis file yet).")
            return
        tk, ttk = self.tk, self.ttk
        dlg = tk.Toplevel(self.root)
        dlg.title("Read X/Y from frame headers")
        dlg.configure(bg=BG)
        dlg.transient(self.root)
        dlg.grab_set()
        content = ttk.Frame(dlg, padding=10)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text=(
            "Reads each frame's raw image header (via fabio) and stores the "
            "two motor values as /frames/pos_x and pos_y. Key names are "
            "case-insensitive; the motor_mne/motor_pos pair convention is "
            "understood. 'List keys' shows what the first frame's header "
            "offers."), foreground=MUTED, wraplength=460, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        v_kx = tk.StringVar(value=str(self.config.get("pos_header_x", "")))
        v_ky = tk.StringVar(value=str(self.config.get("pos_header_y", "")))
        v_dir = tk.StringVar(value=str(self.config.get("pos_header_dir", "")))
        ttk.Label(content, text="X header key").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(content, textvariable=v_kx, width=20).grid(row=1, column=1, sticky="w")
        ttk.Label(content, text="Y header key").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(content, textvariable=v_ky, width=20).grid(row=2, column=1, sticky="w")
        ttk.Label(content, text="Frames folder").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(content, textvariable=v_dir, width=36).grid(row=3, column=1, sticky="we")

        def _browse():
            d = self.filedialog.askdirectory(title="Folder holding the raw frames")
            if d:
                v_dir.set(d)
        ttk.Button(content, text="Browse", command=_browse).grid(row=3, column=2, padx=4)
        ttk.Label(content, text=(
            "Folder is only needed when /frames/filename holds bare names "
            "instead of full paths."), foreground=MUTED, wraplength=460,
            justify="left").grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 8))

        def _list_keys():
            from .frame_metadata import frame_header_keys
            probe = frame_header_keys(path, search_dir=v_dir.get().strip() or None)
            if probe.get("ok"):
                self.messagebox.showinfo(
                    "Header keys",
                    f"{Path(probe['path']).name}:\n\n" + ", ".join(probe["keys"]),
                    parent=dlg)
            else:
                self.messagebox.showerror("Header keys", probe.get("error", "?"),
                                          parent=dlg)

        def _go():
            kx, ky = v_kx.get().strip(), v_ky.get().strip()
            if not kx or not ky:
                self.messagebox.showerror("Read positions",
                                          "Enter both header keys.", parent=dlg)
                return
            from .frame_metadata import import_positions_from_headers
            try:
                man = import_positions_from_headers(
                    path, kx, ky, search_dir=v_dir.get().strip() or None)
            except Exception as e:
                self.messagebox.showerror("Read positions failed", str(e), parent=dlg)
                return
            self.config["pos_header_x"] = kx
            self.config["pos_header_y"] = ky
            self.config["pos_header_dir"] = v_dir.get().strip()
            self.save_config(silent=True)
            dlg.destroy()
            msg = f"Positions read for {man['n_mapped']} frame(s)."
            if man.get("n_missing_file"):
                msg += f" {man['n_missing_file']} frame file(s) not found."
            self._fm_status.configure(text=msg)
            self.log(msg)
            self.fm_refresh_table_clicked()

        btns = ttk.Frame(content)
        btns.grid(row=5, column=0, columnspan=3, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="List keys", command=_list_keys).pack(side="left", padx=4)
        ttk.Button(btns, text="Read positions", command=_go).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left", padx=4)
        content.columnconfigure(1, weight=1)

    def preview_pressure_clicked(self):
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        if not path or not Path(path).is_file():
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text="Run Step 1 first (no analysis file yet).")
            return
        self._draw_pressure_preview(path)

    def fm_refresh_table_clicked(self):
        from .frame_metadata import read_frame_metadata
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        tbl = getattr(self, "_fm_table", None)
        if tbl is None:
            return
        if not path or not Path(path).is_file():
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text="Run Step 1 first (no analysis file yet).")
            return
        tbl.delete(*tbl.get_children())
        meta = read_frame_metadata(path)
        if not meta.get("ok"):
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text=meta.get("error", "Failed to read metadata."))
            return
        names = meta.get("filename") or []
        pressure = meta.get("pressure")
        sigma = meta.get("pressure_sigma")
        temp = meta.get("temperature")
        user = meta.get("user_edited")
        n = int(meta.get("n_frames", 0) or 0)

        def _fmt(arr, i):
            if arr is None or i >= len(arr):
                return "—"
            v = float(arr[i])
            return f"{v:.3g}" if v == v else "—"

        for i in range(n):
            fname = names[i] if i < len(names) else ""
            base = fname.rsplit("/", 1)[-1] if fname else ""
            is_user = bool(user[i]) if user is not None and i < len(user) else False
            tbl.insert("", "end", iid=str(i), values=(
                i, base, _fmt(pressure, i), _fmt(sigma, i), _fmt(temp, i),
                "user" if is_user else "auto"),
                tags=(("user",) if is_user else ()))

    def fm_apply_selected_clicked(self):
        from .frame_metadata import read_frame_metadata, apply_to_analysis
        import numpy as np
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        if not path or not Path(path).is_file():
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text="Run Step 1 first (no analysis file yet).")
            return
        tbl = getattr(self, "_fm_table", None)
        sel = list(tbl.selection()) if tbl is not None else []
        if not sel:
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text="Select one or more frames first.")
            return
        try:
            indices = [int(iid) for iid in sel]
        except ValueError:
            indices = []

        def _parse(var):
            raw = (var.get() if var is not None else "").strip()
            if not raw:
                return None
            return float(raw)

        try:
            p_val = _parse(getattr(self, "_fm_edit_p", None))
            sig_val = _parse(getattr(self, "_fm_edit_sig", None))
            t_val = _parse(getattr(self, "_fm_edit_t", None))
        except ValueError:
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text="P/σ/T must be a number (or blank).")
            return

        if p_val is None and sig_val is None and t_val is None:
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text="Enter at least one value.")
            return

        try:
            meta = read_frame_metadata(path)
            if not meta.get("ok"):
                raise RuntimeError(meta.get("error", "Failed to read metadata."))
            pressure = np.array(meta.get("pressure"), dtype=float, copy=True)
            sigma = np.array(meta.get("pressure_sigma"), dtype=float, copy=True)
            temperature = np.array(meta.get("temperature"), dtype=float, copy=True)

            parts = []
            kwargs = {}
            if p_val is not None:
                pressure[indices] = p_val
                kwargs["pressure"] = pressure
                parts.append("P")
            if sig_val is not None:
                sigma[indices] = sig_val
                kwargs["pressure_sigma"] = sigma
                parts.append("σ")
            if t_val is not None:
                temperature[indices] = t_val
                kwargs["temperature"] = temperature
                parts.append("T")

            apply_to_analysis(path, user_frames=indices, **kwargs)
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text=f"Set {', '.join(parts)} on {len(sel)} frame(s) "
                         "(marked as user edits — they survive re-runs).")
            self.fm_refresh_table_clicked()
            self._draw_pressure_preview(path)
        except Exception as e:
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text=str(e))
            self.log(f"fm_apply_selected_clicked failed: {e!r}", "WARN")

    def _draw_pressure_preview(self, path):
        from .frame_metadata import read_frame_metadata
        import numpy as np

        meta = read_frame_metadata(path)

        for w in self.fm_plot_frame.winfo_children():
            w.destroy()

        # Close previous figure if any
        prev = getattr(self, "_fm_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._fm_fig = None

        pressure = meta.get("pressure")
        if pressure is None:
            pressure = np.array([])
        pressure = np.asarray(pressure, dtype=float)

        if pressure.size == 0 or not np.any(np.isfinite(pressure)):
            self.ttk.Label(
                self.fm_plot_frame,
                text="No pressures yet — Extract from filenames or Import CSV.",
                foreground=MUTED,
            ).pack(anchor="center", expand=True)
            return

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            # matplotlib unavailable — show text summary instead
            n_frames = int(meta.get("n_frames", 0))
            from .frame_metadata import summarize_pressures
            try:
                summ = summarize_pressures(pressure)
            except Exception:
                summ = {}
            n_parsed = int(summ.get("n_parsed", 0)) if summ else 0
            p_min = summ.get("p_min")
            p_max = summ.get("p_max")
            p_txt = (
                f"P {p_min:.2f}–{p_max:.2f} GPa"
                if p_min is not None and p_max is not None else "P unknown"
            )
            self.ttk.Label(
                self.fm_plot_frame,
                text=(
                    f"matplotlib unavailable: {e}\n\n"
                    f"{n_parsed}/{n_frames} frames have pressure ({p_txt})."
                ),
                foreground=MUTED, justify="left",
            ).pack(anchor="center", expand=True)
            return

        fig = Figure(figsize=(7, 4), dpi=100, layout="constrained")
        self._fm_fig = fig
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(1, 1, 1)
        x = np.arange(pressure.size, dtype=float)
        ax.plot(x, pressure, marker=".", markersize=3, linewidth=0.8, color=ACCENT2)
        ax.set_xlabel("frame index")
        ax.set_ylabel("pressure (GPa)")
        ax.set_title("Frame pressure", color=FG)
        self._style_ax(ax)

        self._fm_canvas = self._embed_figure(self.fm_plot_frame, fig)

    # ------------------------------------------------------------------
    # Tab 9 — Identify (Step 3a: deterministic EOS phase matching)
    # ------------------------------------------------------------------

    def _tab_identify(self, frame):
        tk, ttk = self.tk, self.ttk

        # -- params area --------------------------------------------------
        title_row = ttk.Frame(frame)
        title_row.grid(row=0, column=0, columnspan=6, sticky="we", padx=6, pady=(0, 2))
        ttk.Label(
            title_row, text="Phase identification (EOS matching)",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side="left")
        self._identify_help_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(title_row, text="Show instructions",
                        variable=self._identify_help_var,
                        command=self._toggle_identify_help).pack(side="left", padx=12)

        self.checkbox(frame, "run_step3",
                      "Enable phase identification in the next run", row=1)
        self.checkbox(frame, "identify_all_phases",
                      "Search entire library (identify without pre-selecting candidates)",
                      row=2)
        # Two column-groups keep the tab short enough that the results area
        # below stays fully visible on ~700px-tall screens.
        self.field(frame, "p_min", "Pressure min (GPa)", row=3, width=10)
        self.field(frame, "p_max", "Pressure max (GPa)", row=4, width=10)
        self.field(frame, "rel_tol", "Match tolerance (Δd/d)", row=5, width=10)
        self.field(frame, "seen_conf", "Present-in-frame confidence", row=6, width=10)
        self.field(frame, "identify_wavelength",
                   "Wavelength (Å, blank=auto)", row=7, width=10)
        self.field(frame, "pressure_window",
                   "Pressure window ± (GPa)", row=3, width=10, col=1)
        self.field(frame, "pressure_sigma_k",
                   "Window = k·σ (when σ known)", row=4, width=10, col=1)
        self.field(frame, "min_matched",
                   "Min matched reflections", row=5, width=10, col=1)
        self.field(frame, "intensity_k",
                   "Intensity weight (0 = positions only)", row=6, width=10, col=1)
        self.checkbox(frame, "use_pressure_prior",
                      "Use frame-pressure prior (confine fit to ±window)", row=8)
        self.checkbox(frame, "marker_prior",
                      "Estimate pressure from marker phases first", row=8, col=1)
        self.checkbox(frame, "allow_sparse",
                      "Allow sparse/marker-only matches in residual", row=9)
        self.checkbox(frame, "use_frame_temperature",
                      "Apply frame temperatures (thermal expansion)", row=9, col=1)

        # -- Step 3b proposer: ML candidate ranking -----------------------
        mlrow = ttk.Frame(frame)
        mlrow.grid(row=10, column=0, columnspan=6, sticky="w", pady=(6, 2))
        self.vars["run_ml_rank"] = tk.BooleanVar(value=bool(self.config.get("run_ml_rank", False)))
        _mlcb = ttk.Checkbutton(
            mlrow, text="ML candidate ranking (top-K from library → Step 3a verifies)",
            variable=self.vars["run_ml_rank"])
        _mlcb.pack(side="left")
        _ToolTip(_mlcb, (
            "Before the deterministic match, rank the WHOLE library against each frame "
            "(cosine of the measured residual/fit pattern vs each phase simulated at the "
            "frame's pressure) and verify only the top-K with Step 3a. 'ML proposes, "
            "physics verifies.' Deterministic (no torch); needs pymatgen to simulate."))
        ttk.Label(mlrow, text="top-K:", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["ml_rank_top_k"] = tk.StringVar(value=str(self.config.get("ml_rank_top_k", "5")))
        ttk.Entry(mlrow, textvariable=self.vars["ml_rank_top_k"], width=5).pack(side="left")
        ttk.Label(mlrow, text="rank vs:", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["ml_rank_source"] = tk.StringVar(value=str(self.config.get("ml_rank_source", "auto")))
        ttk.Combobox(mlrow, textvariable=self.vars["ml_rank_source"],
                     values=["auto", "residual", "fit"], state="readonly", width=8).pack(side="left")
        ttk.Label(mlrow, text="scorer:", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["ml_scorer"] = tk.StringVar(value=str(self.config.get("ml_scorer", "")))
        _mlsc = ttk.Entry(mlrow, textvariable=self.vars["ml_scorer"], width=22)
        _mlsc.pack(side="left")
        _ToolTip(_mlsc, (
            "Similarity scorer for the ranking. Blank/'cosine' = the deterministic "
            "baseline. 'torch:<path to scorer.pt>' = a trained bulkxrd-ml-train "
            "export (needs bulkxrd[ml]; see docs/ml-training.md). Whatever the "
            "scorer proposes, Step 3a still verifies."))

        # -- Step 3c unknown tracking ------------------------------------
        unkrow = ttk.Frame(frame)
        unkrow.grid(row=11, column=0, columnspan=6, sticky="w", pady=(4, 2))
        ttk.Label(unkrow, text="Unknown tracking:", foreground=MUTED).pack(
            side="left", padx=(0, 6))
        ttk.Label(unkrow, text="track by", foreground=MUTED).pack(side="left", padx=(4, 2))
        self.vars["unknown_tracking_axis"] = tk.StringVar(
            value=str(self.config.get("unknown_tracking_axis", "same") or "same"))
        _ut_axis = ttk.Combobox(
            unkrow, textvariable=self.vars["unknown_tracking_axis"],
            values=["same", "frame", "pressure", "temperature", "time"],
            state="readonly", width=11)
        _ut_axis.pack(side="left")
        _ToolTip(_ut_axis, HELP["unknown_tracking_axis"])
        ttk.Label(unkrow, text="group by", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["unknown_group_by"] = tk.StringVar(
            value=str(self.config.get("unknown_group_by", "same") or "same"))
        _ut_group = ttk.Combobox(
            unkrow, textvariable=self.vars["unknown_group_by"],
            values=["same", "none", "scan", "folder"], state="readonly", width=8)
        _ut_group.pack(side="left")
        _ToolTip(_ut_group, HELP["unknown_group_by"])
        ttk.Label(unkrow, text="tol×FWHM", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["unknown_link_tol_fwhm"] = tk.StringVar(
            value=str(self.config.get("unknown_link_tol_fwhm", "1.5")))
        _ut_tol = ttk.Entry(unkrow, textvariable=self.vars["unknown_link_tol_fwhm"], width=5)
        _ut_tol.pack(side="left")
        _ToolTip(_ut_tol, HELP["unknown_link_tol_fwhm"])
        ttk.Label(unkrow, text="missing", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["unknown_max_gap"] = tk.StringVar(
            value=str(self.config.get("unknown_max_gap", "2")))
        _ut_gap = ttk.Entry(unkrow, textvariable=self.vars["unknown_max_gap"], width=5)
        _ut_gap.pack(side="left")
        _ToolTip(_ut_gap, HELP["unknown_max_gap"])
        ttk.Label(unkrow, text="axis gap", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["unknown_max_axis_gap"] = tk.StringVar(
            value=str(self.config.get("unknown_max_axis_gap", "")))
        _ut_agap = ttk.Entry(unkrow, textvariable=self.vars["unknown_max_axis_gap"], width=7)
        _ut_agap.pack(side="left")
        _ToolTip(_ut_agap, HELP["unknown_max_axis_gap"])
        ttk.Label(unkrow, text="min frames", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["unknown_min_frames"] = tk.StringVar(
            value=str(self.config.get("unknown_min_frames", "3")))
        _ut_min = ttk.Entry(unkrow, textvariable=self.vars["unknown_min_frames"], width=5)
        _ut_min.pack(side="left")
        _ToolTip(_ut_min, HELP["unknown_min_frames"])
        ttk.Label(unkrow, text="Jaccard", foreground=MUTED).pack(side="left", padx=(10, 2))
        self.vars["unknown_jaccard"] = tk.StringVar(
            value=str(self.config.get("unknown_jaccard", "0.6")))
        _ut_j = ttk.Entry(unkrow, textvariable=self.vars["unknown_jaccard"], width=5)
        _ut_j.pack(side="left")
        _ToolTip(_ut_j, HELP["unknown_jaccard"])
        self.vars["unknown_axis_predictor"] = tk.BooleanVar(
            value=bool(self.config.get("unknown_axis_predictor", True)))
        _ut_pred = ttk.Checkbutton(
            unkrow, text="predict drift", variable=self.vars["unknown_axis_predictor"])
        _ut_pred.pack(side="left", padx=(10, 2))
        _ToolTip(_ut_pred, HELP["unknown_axis_predictor"])

        self._identify_help = ttk.Label(
            frame,
            text=(
                "Step 3a fits each candidate phase's equation of state to every "
                "frame's peak list and reports a match confidence (and, for "
                "pressure series, a best-fit pressure). Needs pymatgen.\n\n"
                "To run it:\n"
                "  1. Phases tab — enable the candidate phases (or tick Search "
                "entire library here).\n"
                "  2. Tick 'Enable phase identification' above.\n"
                "  3. Run tab — Run analysis.\n"
                "  4. Click 'Load identification' to see the results.\n"
                "The residual step then subtracts confirmed phases and re-fits, "
                "and the unknowns step clusters whatever is left."
            ),
            foreground=MUTED, justify="left", wraplength=640,
        )
        self._identify_help.grid(row=13, column=0, columnspan=6, sticky="w",
                                 padx=6, pady=(8, 4))
        self._identify_help.grid_remove()   # hidden until the checkbox reveals it
        self.autowrap(self._identify_help)

        # -- controls row -------------------------------------------------
        ctrl = ttk.Frame(frame)
        ctrl.grid(row=12, column=0, columnspan=6, sticky="w", pady=(4, 2))

        ttk.Button(ctrl, text="Load identification",
                   command=self.load_identify).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_identify_fig", None), "Phase identification")
                   ).pack(side="left", padx=4)

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

        # -- body: per-frame phase table (left) + plot (right) ------------
        # NOTE: the body has its own row. The help label above shares no cell
        # with it — sharing a row was the bug that made the 'Show
        # instructions' checkbox appear to do nothing (the label toggled
        # underneath the body frame).
        body = ttk.Frame(frame)
        body.grid(row=13, column=0, columnspan=6, sticky="nsew")
        frame.rowconfigure(13, weight=1)
        frame.columnconfigure(0, weight=1)

        # Left: two browse modes in a sub-notebook. Stacking the per-frame
        # table, the materials summary, AND the frames list vertically used to
        # need ~470px and pushed the tab bottom off short screens.
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 6))
        lnb = ttk.Notebook(left)
        lnb.pack(fill="both", expand=True)
        page_frame = ttk.Frame(lnb, padding=4)
        page_mat = ttk.Frame(lnb, padding=4)
        lnb.add(page_frame, text="This frame")
        lnb.add(page_mat, text="Materials")

        # Page 1 — a frame selector and the ranked phase table for that frame.
        sel = ttk.Frame(page_frame)
        sel.pack(fill="x", pady=(0, 4))
        ttk.Label(sel, text="Frame", foreground=MUTED).pack(side="left", padx=(0, 4))
        ttk.Button(sel, text="◀", width=2,
                   command=lambda: self._step_identify_frame(-1)).pack(side="left")
        self._identify_frame_var = tk.StringVar(value="0")
        self._identify_frame_spin = ttk.Spinbox(
            sel, from_=0, to=0, width=6, textvariable=self._identify_frame_var,
            command=self._update_identify_table)
        self._identify_frame_spin.pack(side="left", padx=2)
        self._identify_frame_spin.bind("<Return>", lambda e: self._update_identify_table())
        ttk.Button(sel, text="▶", width=2,
                   command=lambda: self._step_identify_frame(1)).pack(side="left")

        tbl_frame = ttk.Frame(page_frame)
        tbl_frame.pack(fill="both", expand=True)
        cols = ("phase", "model", "conf", "recall", "prec", "pressure", "lines")
        tbl = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=6,
                           selectmode="browse")
        tbl_vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=tbl.yview)
        tbl.configure(yscrollcommand=tbl_vsb.set)
        for c, txt, w, anc in (("phase", "Phase", 140, "w"), ("model", "P-model", 78, "center"),
                               ("conf", "Conf", 52, "center"),
                               ("recall", "Recall", 52, "center"), ("prec", "Prec", 52, "center"),
                               ("pressure", "P (GPa)", 60, "center"), ("lines", "#", 36, "center")):
            tbl.heading(c, text=txt)
            tbl.column(c, width=w, minwidth=34, anchor=anc, stretch=(c == "phase"))
        tbl.tag_configure("present", foreground=ACCENT2)
        tbl.tag_configure("absent", foreground=MUTED)
        tbl_vsb.pack(side="right", fill="y")
        tbl.pack(side="left", fill="both", expand=True)
        self._identify_table = tbl

        # Page 2 — materials-found summary + frames-by-material browser.
        ttk.Label(page_mat, text="Materials found (click → frames containing it):",
                 foreground=MUTED).pack(anchor="w", pady=(0, 2))

        summary_cols = ("phase", "frames", "medP")
        summary = ttk.Treeview(page_mat, columns=summary_cols, show="headings", height=5,
                               selectmode="browse")
        for c, txt, w in (("phase", "Material", 140), ("frames", "Frames", 60),
                          ("medP", "med P", 70)):
            summary.heading(c, text=txt)
            summary.column(c, width=w, minwidth=34, anchor="center" if c != "phase" else "w",
                           stretch=(c == "phase"))
        summary.pack(fill="x", expand=False)
        summary.bind("<<TreeviewSelect>>", self._on_phase_summary_select)
        self._identify_phase_summary = summary

        ttk.Label(page_mat, text="Frames with selected material (double-click to view):",
                 foreground=MUTED).pack(anchor="w", pady=(6, 2))

        frames_list_frame = ttk.Frame(page_mat)
        frames_list_frame.pack(fill="both", expand=True)
        frames_vsb = ttk.Scrollbar(frames_list_frame, orient="vertical")
        listbox = tk.Listbox(
            frames_list_frame, height=5, bg=BG2, fg=FG,
            selectbackground=ACCENT2, yscrollcommand=frames_vsb.set,
            exportselection=False,
        )
        frames_vsb.configure(command=listbox.yview)
        frames_vsb.pack(side="right", fill="y")
        listbox.pack(side="left", fill="both", expand=True)
        listbox.bind("<Double-Button-1>", self._on_phase_frame_activate)
        self._identify_frames_list = listbox
        self._phase_frames: Dict[str, Any] = {}
        self._phase_frame_indices = []

        # Right: the (decluttered) confidence/pressure plot.
        self.identify_plot_frame = ttk.Frame(body)
        self.identify_plot_frame.pack(side="left", fill="both", expand=True)

        ttk.Label(
            self.identify_plot_frame,
            text="Enable phase identification and Run, or click \"Load "
                 "identification\" to view per-frame phases + confidence. "
                 "Tick \"Show instructions\" (top) for the step-by-step workflow.",
            foreground=MUTED, wraplength=380, justify="left",
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
        self._identify_tr = tr
        if not tr["ok"]:
            self.ttk.Label(
                self.identify_plot_frame,
                text=tr["error"],
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            if hasattr(self, "_identify_status"):
                self._identify_status.configure(text=tr["error"])
            return

        # Sync the per-frame table + its frame selector range.
        if hasattr(self, "_identify_frame_spin"):
            self._identify_frame_spin.configure(to=max(tr["n_frames"] - 1, 0))
        self._update_identify_table()

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

        # Only plot phases actually seen at least once (max confidence ≥ the bar),
        # so the figure isn't 22 overlapping flat lines + a giant legend. Fall back
        # to the strongest few if nothing clears the bar, so it's never blank.
        def _maxconf(rec):
            c = rec.get("confidence")
            return float(np.nanmax(c)) if c is not None and len(c) else 0.0
        shown = [r for r in tr["phases"] if _maxconf(r) >= conf_min]
        if not shown:
            shown = sorted(tr["phases"], key=_maxconf, reverse=True)[:5]

        for rec in shown:
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
            ax_conf.plot(x, conf_arr, linewidth=0.7, color=color, label=label)

        ax_pres.set_ylabel("pressure (GPa)")
        ax_pres.set_title(
            f"Phases seen (confidence ≥ {conf_min:.2f}) — {len(shown)} shown",
            color=FG)
        handles, labels = ax_pres.get_legend_handles_labels()
        if handles:
            ax_pres.legend(fontsize=7, framealpha=0.4, ncol=2, loc="upper right")
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
        self._attach_hover(self._identify_canvas, self._identify_status)

    def _step_identify_frame(self, delta: int):
        try:
            cur = int(float(self._identify_frame_var.get()))
        except (ValueError, AttributeError):
            cur = 0
        n = int(getattr(self, "_identify_tr", {}).get("n_frames", 0) or 0)
        cur = max(0, min(cur + delta, max(n - 1, 0)))
        self._identify_frame_var.set(str(cur))
        self._update_identify_table()

    def _update_identify_table(self):
        """Fill the per-frame table: phases ranked by confidence for the selected
        frame, with recall / precision / best-fit pressure. Present phases (≥ the
        Min-confidence bar) are highlighted."""
        import numpy as np
        tbl = getattr(self, "_identify_table", None)
        tr = getattr(self, "_identify_tr", None)
        if tbl is None or not tr or not tr.get("ok"):
            return
        tbl.delete(*tbl.get_children())
        n = int(tr.get("n_frames", 0) or 0)
        try:
            fi = max(0, min(int(float(self._identify_frame_var.get())), max(n - 1, 0)))
        except (ValueError, AttributeError):
            fi = 0
        try:
            conf_min = max(0.0, min(1.0, float(self._identify_conf_var.get())))
        except (ValueError, AttributeError):
            conf_min = 0.5

        def _at(arr, default=np.nan):
            return float(arr[fi]) if arr is not None and fi < len(arr) else default

        rows = []
        for rec in tr["phases"]:
            conf = _at(rec.get("confidence"), 0.0)
            rows.append((conf, rec))
        rows.sort(key=lambda t: (-(t[0] if t[0] == t[0] else -1), t[1]["name"].lower()))
        # Short labels for the pressure model the phase was fit under.
        # ("ambient_only" is the pre-rename value in old HDF5 files — read-compat.)
        _MODEL_LABEL = {"eos": "EOS", "axial_eos": "axial", "no_eos": "no-EOS",
                        "ambient_only": "no-EOS"}
        n_present = 0
        for conf, rec in rows:
            recall = _at(rec.get("recall"))
            prec = _at(rec.get("precision"))
            press = _at(rec.get("pressure"))
            penalty = _at(rec.get("prior_penalty"), 1.0)
            nmatch = rec.get("n_matched")
            nm = int(nmatch[fi]) if nmatch is not None and fi < len(nmatch) else 0
            present = conf >= conf_min
            n_present += int(present)
            model = rec.get("pressure_model") or ("eos" if rec.get("has_eos") else "no_eos")
            mlabel = _MODEL_LABEL.get(model, model)
            # Flag exemption from / impact of the pressure prior.
            if rec.get("pressure_assumption") == "ignore_prior":
                mlabel += " (no prior)"
            elif penalty == penalty and penalty < 0.95:
                mlabel += " ↓P"
            name = rec["name"]
            pstr = "—" if (press != press or model in ("no_eos", "ambient_only")) else f"{press:.1f}"
            tbl.insert("", "end", values=(
                name, mlabel, f"{conf:.2f}",
                "—" if recall != recall else f"{recall:.2f}",
                "—" if prec != prec else f"{prec:.2f}",
                pstr, nm),
                tags=("present" if present else "absent",))
        if hasattr(self, "_identify_status"):
            self._identify_status.configure(
                text=f"frame {fi}: {n_present} phase(s) ≥ {conf_min:.2f} "
                     f"of {len(tr['phases'])}")

        self._update_phase_summary()

    def _update_phase_summary(self):
        """Fill the materials-found summary: one row per phase with the number of
        frames it's present in (confidence >= bar AND >=3 matched reflections) and
        its median pressure over those frames. Also stashes per-phase present-frame
        index lists on self._phase_frames for the frames-list browser."""
        import numpy as np
        summary = getattr(self, "_identify_phase_summary", None)
        tr = getattr(self, "_identify_tr", None)
        if summary is None:
            return
        summary.delete(*summary.get_children())
        self._phase_frames = {}
        if not tr or not tr.get("ok"):
            return
        try:
            conf_min = max(0.0, min(1.0, float(self._identify_conf_var.get())))
        except (ValueError, AttributeError):
            conf_min = 0.5

        rows = []
        for rec in tr["phases"]:
            name = rec["name"]
            conf = rec.get("confidence")
            if conf is None:
                continue
            conf = np.asarray(conf, dtype=float)
            present = conf >= conf_min
            nmatch = rec.get("n_matched")
            if nmatch is not None:
                nmatch = np.asarray(nmatch)
                present = present & (nmatch >= 3)
            present_idx = np.nonzero(present)[0]
            self._phase_frames[name] = [int(i) for i in present_idx]
            n_present = int(present_idx.size)
            if n_present:
                pressure = np.asarray(rec.get("pressure"), dtype=float)
                med_p = float(np.nanmedian(pressure[present_idx]))
                med_p_str = "—" if med_p != med_p else f"{med_p:.1f}"
            else:
                med_p_str = "—"
            rows.append((n_present, name, med_p_str))

        rows.sort(key=lambda t: (-t[0], t[1].lower()))
        for n_present, name, med_p_str in rows:
            summary.insert("", "end", iid=name, values=(name, n_present, med_p_str))

    def _on_phase_summary_select(self, event=None):
        import numpy as np
        summary = getattr(self, "_identify_phase_summary", None)
        listbox = getattr(self, "_identify_frames_list", None)
        tr = getattr(self, "_identify_tr", None)
        if summary is None or listbox is None:
            return
        sel = summary.selection()
        listbox.delete(0, "end")
        self._phase_frame_indices = []
        if not sel:
            return
        name = sel[0]
        indices = self._phase_frames.get(name, [])

        conf_arr = None
        pressure_arr = None
        if tr and tr.get("ok"):
            for rec in tr["phases"]:
                if rec["name"] == name:
                    if rec.get("confidence") is not None:
                        conf_arr = np.asarray(rec["confidence"], dtype=float)
                    if rec.get("pressure") is not None:
                        pressure_arr = np.asarray(rec["pressure"], dtype=float)
                    break

        for i in indices:
            conf_txt = ""
            if conf_arr is not None and i < len(conf_arr):
                conf_txt = f"   conf {conf_arr[i]:.2f}"
            p_txt = ""
            if pressure_arr is not None and i < len(pressure_arr):
                p = pressure_arr[i]
                if p == p:
                    p_txt = f"   P {p:.1f}"
            listbox.insert("end", f"frame {i}{conf_txt}{p_txt}")
            self._phase_frame_indices.append(int(i))

    def _on_phase_frame_activate(self, event=None):
        listbox = getattr(self, "_identify_frames_list", None)
        if listbox is None:
            return
        sel = listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        indices = getattr(self, "_phase_frame_indices", [])
        if idx >= len(indices):
            return
        frame = indices[idx]
        self._identify_frame_var.set(str(frame))
        self._update_identify_table()

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

    # Result caches so toggling a plot option doesn't recompute reflection
    # simulation / ROI integration. Keyed on the file's mtime — a re-run
    # invalidates them automatically.

    @staticmethod
    def _analysis_mtime(path) -> int:
        try:
            return Path(path).stat().st_mtime_ns
        except OSError:
            return 0

    def _cached_tracks(self, path, phase):
        cache = getattr(self, "_tracks_cache", None)
        if cache is None:
            cache = self._tracks_cache = {}
        key = (str(path), self._analysis_mtime(path), phase.name)
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            from .heatmap import reflection_tracks
            cache[key] = reflection_tracks(path, phase)
        return cache[key]

    def _cached_layers(self, path, phases):
        cache = getattr(self, "_layers_cache", None)
        if cache is None:
            cache = self._layers_cache = {}
        key = (str(path), self._analysis_mtime(path),
               tuple(sorted(p.name for p in phases)))
        if key not in cache:
            if len(cache) > 16:
                cache.clear()
            from .heatmap import phase_layers
            cache[key] = phase_layers(path, phases)
        return cache[key]

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
        ttk.Button(row1, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_patternmap_fig", None), "Pattern map")
                   ).pack(side="left", padx=4)

        ttk.Label(row1, text="Source:", foreground=MUTED).pack(side="left", padx=(12, 2))
        from .heatmap import SOURCES as _PM_SOURCES
        self._pm_source = ttk.Combobox(
            row1,
            values=list(_PM_SOURCES),
            state="readonly", width=12,
        )
        self._pm_source.set("clean")
        self._pm_source.pack(side="left", padx=2)
        self._pm_source.bind("<<ComboboxSelected>>",
                             lambda e: self.load_pattern_map())

        ttk.Label(row1, text="X axis:", foreground=MUTED).pack(side="left", padx=(12, 2))
        self._pm_xaxis = ttk.Combobox(
            row1,
            values=["frame", "pressure", "temperature", "time"],
            state="readonly", width=11,
        )
        self._pm_xaxis.set("frame")
        self._pm_xaxis.pack(side="left", padx=2)
        self._pm_xaxis.bind("<<ComboboxSelected>>",
                            lambda e: self.load_pattern_map())

        self._pm_tracks = tk.BooleanVar(value=True)
        _trk_cb = ttk.Checkbutton(
            row1, text="Reflection tracks",
            variable=self._pm_tracks, command=self.load_pattern_map,
        )
        _trk_cb.pack(side="left", padx=8)
        _ToolTip(_trk_cb, "Predicted reflection positions of the enabled phases. "
                          "Drawn on the frame axis only.")

        self._pm_layers = tk.BooleanVar(value=False)
        _lay_cb = ttk.Checkbutton(
            row1, text="Phase layers",
            variable=self._pm_layers, command=self.load_pattern_map,
        )
        _lay_cb.pack(side="left", padx=4)
        _ToolTip(_lay_cb, "Second panel: per-phase matched-reflection intensity "
                          "vs the chosen x variable.")

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

        from .heatmap import pattern_image
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

        x_axis = getattr(self._pm_xaxis, "get", lambda: "frame")()
        if not x_axis:
            x_axis = "frame"
        img = pattern_image(path, source=self._pm_source.get(), x_axis=x_axis)
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
        Z = img["Z"]                      # (n_bins, n_frames)
        radial = img["radial"]
        n = img["n_frames"]
        x_label = img.get("x_label") or "frame index"
        xv = (np.asarray(img["x"], dtype=float) if img.get("x") is not None
              else np.arange(n, dtype=float))

        pos = Z[np.isfinite(Z) & (Z > 0)]
        if pos.size:
            vmin = float(np.percentile(pos, 5))
            vmax = float(np.percentile(pos, 99))
        else:
            vmin = None
            vmax = None

        if x_axis == "frame":
            # Uniform grid → imshow; nearest keeps frame columns crisp.
            ax.imshow(
                Z, aspect="auto", origin="lower", cmap="magma",
                interpolation="nearest",
                extent=[-0.5, float(n) - 0.5,
                        float(radial.min()), float(radial.max())],
                vmin=vmin, vmax=vmax,
            )
        else:
            # Physical (possibly non-uniform) coordinates → pcolormesh on the
            # frames sorted by x; imshow would silently pretend the values are
            # evenly spaced.
            fin = np.isfinite(xv)
            if fin.sum() < 2:
                msg = f"Fewer than two frames have a finite {x_axis} value."
                self.ttk.Label(self.patternmap_plot_frame, text=msg,
                               foreground=WARN).pack(anchor="center", expand=True)
                if hasattr(self, "_pm_status"):
                    self._pm_status.configure(text=msg)
                return
            order = np.argsort(xv[fin], kind="stable")
            xs = xv[fin][order]
            Zs = Z[:, fin][:, order]
            ax.pcolormesh(xs, radial, Zs, cmap="magma", shading="nearest",
                          vmin=vmin, vmax=vmax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(img["unit"] or "radial")
        ax.set_title(f"Pattern waterfall — {img['source']}", color=FG)
        self._style_ax(ax)

        # Reflection-track overlays (frame x-axis only — on a sorted physical
        # axis, frames are reordered and track curves would not align)
        if self._pm_tracks.get() and pymatgen_available() and x_axis == "frame":
            any_phase_plotted = False
            for phase_obj in self._enabled_phase_objects():
                tr = self._cached_tracks(path, phase_obj)
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

        # Per-phase intensity on the bottom axis, vs the same x variable.
        if show_layers and ax2 is not None:
            pl = self._cached_layers(path, self._enabled_phase_objects())
            if pl["ok"]:
                for layer in pl["layers"]:
                    y = np.asarray(layer["intensity"], dtype=float)
                    lx = xv[:y.size]
                    m = np.isfinite(lx) & np.isfinite(y)
                    o = np.argsort(lx[m], kind="stable")
                    ax2.plot(lx[m][o], y[m][o], lw=0.8, marker=".",
                             markersize=2, label=layer["name"])
                ax2.set_xlabel(x_label)
                ax2.set_ylabel("phase intensity (norm.)")
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
        self._attach_hover(self._patternmap_canvas, self._pm_status)

    # ------------------------------------------------------------------
    # Tab 11 — Unknowns (Step-3c stacked cluster diagram)
    # ------------------------------------------------------------------

    def _tab_unknowns(self, frame):
        tk, ttk = self.tk, self.ttk

        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=(0, 4))
        ttk.Button(row1, text="Load unknowns",
                   command=self.load_unknowns).pack(side="left", padx=4)
        ttk.Button(row1, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_unknowns_fig", None), "Unknowns")
                   ).pack(side="left", padx=4)
        ttk.Label(row1, text="Show:", foreground=MUTED).pack(side="left", padx=(12, 2))
        self._unk_show = ttk.Combobox(
            row1, values=["unknown clusters", "spot tracks d(P)"],
            state="readonly", width=16)
        self._unk_show.set("unknown clusters")
        self._unk_show.pack(side="left", padx=2)
        self._unk_show.bind("<<ComboboxSelected>>", lambda e: self.load_unknowns())
        _ToolTip(self._unk_show, (
            "unknown clusters — the Step-3c residual-peak co-occurrence "
            "diagram.  spot tracks d(P) — the bulkxrd-spots crystallite "
            "reflections as d-spacing vs pressure curves (one per grain "
            "reflection; RISING curves = d grows under pressure, negative "
            "linear compressibility). Pick an hkl table to label the curves."))
        _hkl_btn = ttk.Button(row1, text="hkl table…",
                              command=self.pick_spot_match_table)
        _hkl_btn.pack(side="left", padx=2)
        _ToolTip(_hkl_btn, (
            "Calculated reflection list (d/I pairs or an 'h k l d … I' table) "
            "used to label the spot-track d(P) curves with hkl assignments. "
            "Remembered in the session config."))
        ttk.Label(row1, text="X axis:", foreground=MUTED).pack(side="left", padx=(12, 2))
        self._unk_xaxis = ttk.Combobox(
            row1,
            values=["frame", "pressure", "temperature", "time"],
            state="readonly", width=11,
        )
        self._unk_xaxis.set("frame")
        self._unk_xaxis.pack(side="left", padx=2)
        self._unk_xaxis.bind("<<ComboboxSelected>>", lambda e: self.load_unknowns())

        ttk.Label(row1, text="Color by:", foreground=MUTED).pack(side="left", padx=(12, 2))
        self._unk_color = ttk.Combobox(
            row1,
            values=["center", "amplitude", "track", "group"],
            state="readonly", width=10,
        )
        self._unk_color.set("center")
        self._unk_color.pack(side="left", padx=2)
        self._unk_color.bind("<<ComboboxSelected>>", lambda e: self.load_unknowns())

        ttk.Label(row1, text="Min obs/cluster:", foreground=MUTED).pack(
            side="left", padx=(12, 2))
        self._unk_min_obs = tk.StringVar(value="1")
        _min_entry = ttk.Entry(row1, textvariable=self._unk_min_obs, width=5)
        _min_entry.pack(side="left", padx=2)
        _min_entry.bind("<Return>", lambda e: self.load_unknowns())
        _ToolTip(_min_entry, "Minimum residual-peak observations per unknown cluster.")
        ttk.Label(row1, text="Min frames/cluster:", foreground=MUTED).pack(
            side="left", padx=(12, 2))
        self._unk_min_frames = tk.StringVar(value="1")
        _min_frames_entry = ttk.Entry(row1, textvariable=self._unk_min_frames, width=5)
        _min_frames_entry.pack(side="left", padx=2)
        _min_frames_entry.bind("<Return>", lambda e: self.load_unknowns())
        _ToolTip(
            _min_frames_entry,
            "Minimum distinct frames supporting the cluster; useful for hiding short bursts.",
        )
        ttk.Button(row1, text="Refresh", command=self.load_unknowns).pack(
            side="left", padx=4)

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=(0, 4))
        ttk.Label(row2, text="Export →", foreground=MUTED).pack(side="left", padx=(4, 6))
        ttk.Button(row2, text="Diagram CSV…",
                   command=self.export_unknown_diagram_clicked).pack(side="left", padx=2)
        ttk.Button(row2, text="Frames with unknowns…",
                   command=self.export_unknown_frames_clicked).pack(side="left", padx=2)
        _spot_btn = ttk.Button(row2, text="Spot tracks…",
                               command=self.export_spot_tracks_clicked)
        _spot_btn.pack(side="left", padx=2)
        _ToolTip(_spot_btn, (
            "Export the /spots single-crystal tracks (bulkxrd-spots) as a "
            "handoff CSV bundle: per-track summary (+ optional hkl matches "
            "against a calculated reflection list), long-format d(P) point "
            "tables, untracked single-band reflections, and a README with "
            "provenance."))
        self._unknowns_status = ttk.Label(row2, text="", foreground=MUTED)
        self._unknowns_status.pack(side="left", padx=12)

        self.unknowns_plot_frame = ttk.Frame(frame)
        self.unknowns_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.unknowns_plot_frame,
            text="Load unknowns after Step 3a residual + Step 3c has run.",
            foreground=MUTED,
        ).pack(anchor="center", expand=True)

    def _unknown_min_obs_value(self) -> int:
        try:
            return max(1, int(float(self._unk_min_obs.get())))
        except Exception:
            return 1

    def _unknown_min_frames_value(self) -> int:
        try:
            return max(1, int(float(self._unk_min_frames.get())))
        except Exception:
            return 1

    def _unknown_filtered_clusters(self, data=None):
        if data is None:
            data = getattr(self, "_unknowns_data", None)
        if not data or not data.get("ok"):
            return []
        min_obs = self._unknown_min_obs_value()
        min_frames = self._unknown_min_frames_value()
        return [
            c for c in data.get("clusters", [])
            if int(c.get("n_obs", 0)) >= min_obs
            and int(c.get("n_frames_observed", 0)) >= min_frames
        ]

    def _unknown_selected_frames(self, data=None):
        import numpy as np
        if data is None:
            data = getattr(self, "_unknowns_data", None)
        if not data or not data.get("ok"):
            return []
        keep_clusters = {
            int(c["cluster"]) for c in self._unknown_filtered_clusters(data)
        }
        frame = np.asarray(data["frame"], dtype=int)
        cluster = np.asarray(data["cluster"], dtype=int)
        keep = np.array([int(c) in keep_clusters for c in cluster], dtype=bool)
        return sorted(set(int(f) for f in frame[keep].tolist()))

    def load_unknowns(self):
        """Render Step-3c unknown clusters as a stacked phase diagram."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            return

        prev = getattr(self, "_unknowns_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._unknowns_fig = None

        for w in self.unknowns_plot_frame.winfo_children():
            w.destroy()

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            self.ttk.Label(
                self.unknowns_plot_frame,
                text=f"matplotlib unavailable: {e}",
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            return

        import numpy as np

        show = getattr(self._unk_show, "get", lambda: "unknown clusters")()
        if (show or "").startswith("spot tracks"):
            self._render_spot_tracks(path, Figure, FigureCanvasTkAgg, np)
            return

        from .heatmap import unknown_diagram
        x_axis = getattr(self._unk_xaxis, "get", lambda: "frame")() or "frame"
        data = unknown_diagram(path, x_axis=x_axis)
        self._unknowns_data = data
        if not data.get("ok"):
            err = data.get("error", "unknown error")
            self.ttk.Label(
                self.unknowns_plot_frame,
                text=f"Unknowns: {err}",
                foreground=WARN,
            ).pack(anchor="center", expand=True)
            self._unknowns_status.configure(text=err)
            return

        clusters = self._unknown_filtered_clusters(data)
        cluster_ids = [int(c["cluster"]) for c in clusters]
        row_of = {cid: i for i, cid in enumerate(cluster_ids)}

        frame = np.asarray(data["frame"], dtype=int)
        x = np.asarray(data["x"], dtype=float)
        cluster = np.asarray(data["cluster"], dtype=int)
        center = np.asarray(data["center"], dtype=float)
        amp = np.asarray(data["amplitude"], dtype=float)
        track = np.asarray(data["track"], dtype=float)
        group = np.asarray(data.get("group", np.zeros(frame.size)), dtype=float)
        keep = np.isfinite(x) & np.array([int(c) in row_of for c in cluster], dtype=bool)
        x_plot = x[keep]
        y_plot = np.array([row_of[int(c)] for c in cluster[keep]], dtype=float)
        color_by = getattr(self._unk_color, "get", lambda: "center")() or "center"
        c_plot = {"amplitude": amp, "track": track, "group": group}.get(color_by, center)[keep]

        fig_h = 5.0 if len(clusters) <= 80 else 6.5
        fig = Figure(figsize=(8, fig_h), dpi=100, layout="constrained")
        self._unknowns_fig = fig
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(1, 1, 1)
        self._style_ax(ax)

        if not clusters or x_plot.size == 0:
            ax.set_title("No unknown clusters to display with current filter", color=FG)
            ax.set_xlabel(data.get("x_label") or "frame index")
            ax.set_ylabel("unknown cluster")
        else:
            for c in clusters:
                ci = int(c["cluster"])
                row = row_of[ci]
                x0, x1 = c.get("x_min"), c.get("x_max")
                if x0 == x0 and x1 == x1:
                    ax.hlines(row, float(x0), float(x1), color=MUTED,
                              lw=0.8, alpha=0.35)

            sizes = np.full(x_plot.size, 26.0)
            amp_plot = amp[keep]
            finite_amp = amp_plot[np.isfinite(amp_plot) & (amp_plot > 0)]
            if finite_amp.size:
                lo, hi = float(np.percentile(finite_amp, 10)), float(np.percentile(finite_amp, 95))
                if hi > lo:
                    sizes = 18.0 + 34.0 * np.clip((amp_plot - lo) / (hi - lo), 0, 1)
            sc = ax.scatter(
                x_plot, y_plot, c=c_plot, s=sizes,
                cmap="viridis", alpha=0.9, edgecolors=FG, linewidths=0.25,
            )
            try:
                cb = fig.colorbar(sc, ax=ax, label=color_by)
                self._style_colorbar(cb)
            except Exception:
                pass
            ax.set_xlabel(data.get("x_label") or "frame index")
            ax.set_ylabel("unknown cluster")
            ax.set_title(
                f"Unknown clusters — {len(clusters)} cluster(s), "
                f"{int(x_plot.size)} observation(s)",
                color=FG,
            )
            if len(clusters) <= 35:
                ax.set_yticks(np.arange(len(clusters)))
                labels = []
                for rec in clusters:
                    gl = str(rec.get("group_label", "") or "")
                    labels.append(f"{gl}:{rec['cluster']}" if gl else str(rec["cluster"]))
                ax.set_yticklabels(labels)
            else:
                ticks = np.unique(np.linspace(0, len(clusters) - 1,
                                             min(18, len(clusters))).astype(int))
                ax.set_yticks(ticks)
                ax.set_yticklabels([str(cluster_ids[i]) for i in ticks])
            ax.set_ylim(-0.75, len(clusters) - 0.25)

        canvas = self._embed_figure(self.unknowns_plot_frame, fig)
        n_frames = len(self._unknown_selected_frames(data))
        self._unknowns_status.configure(
            text=(f"{len(clusters)} clusters, {int(x_plot.size)}/{data['n_obs']} obs, "
                  f"{n_frames} frame(s) with unknowns"))
        self._attach_hover(canvas, self._unknowns_status)

    def pick_spot_match_table(self):
        """Choose the calculated-reflection table used to hkl-label spot tracks."""
        path = self.filedialog.askopenfilename(
            title="Calculated reflection table (Cancel = clear)",
            filetypes=[("Reflection tables", "*.txt *.csv *.dat"),
                       ("All files", "*.*")],
        )
        self.config["spot_match_file"] = path or ""
        self.save_config(silent=True)
        if (getattr(self._unk_show, "get", lambda: "")() or "").startswith("spot"):
            self.load_unknowns()

    def _render_spot_tracks(self, path, Figure, FigureCanvasTkAgg, np):
        """d(P) curves for the /spots crystallite tracks — the group-meeting
        view: one polyline per grain reflection, rising curves = NLC."""
        from .spots import load_spot_tracks
        match = str(self.config.get("spot_match_file", "") or "").strip() or None
        data = load_spot_tracks(path, min_points=self._unknown_min_frames_value(),
                                match=match)
        if not data.get("ok"):
            err = data.get("error", "unknown error")
            self.ttk.Label(self.unknowns_plot_frame,
                           text=f"Spot tracks: {err}", foreground=WARN
                           ).pack(anchor="center", expand=True)
            self._unknowns_status.configure(text=err)
            return
        tracks = data["tracks"]

        fig = Figure(figsize=(8, 5.5), dpi=100, layout="constrained")
        self._unknowns_fig = fig
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(1, 1, 1)
        self._style_ax(ax)

        if not tracks:
            ax.set_title("No spot tracks pass the filter — lower 'Min "
                         "frames/cluster' or run bulkxrd-spots", color=FG)
        else:
            try:
                from matplotlib import colormaps
                cmap = colormaps["twilight"]         # azimuth is periodic
            except Exception:                        # older matplotlib
                from matplotlib import cm
                cmap = cm.get_cmap("twilight")
            n_pts = int(sum(t["n_points"] for t in tracks))
            n_nlc = 0
            for t in tracks:
                rising = t["dd_dp"] > 5e-4
                n_nlc += int(rising)
                color = cmap(((t["azim"] + 180.0) % 360.0) / 360.0)
                ax.plot(t["pressure"], t["d"], "-", color=color,
                        lw=2.2 if rising else 1.1,
                        alpha=0.95 if rising else 0.75, zorder=3 if rising else 2)
                ax.plot(t["pressure"], t["d"], "^" if rising else "o",
                        color=color, ms=5.5 if rising else 3.5,
                        mec=FG, mew=0.3, ls="none", zorder=4)
                label = t["hkl"] or (f"az{t['azim']:+.0f}°" if len(tracks) <= 25 else "")
                if label:
                    ax.annotate(f" {label}", (t["pressure"][-1], t["d"][-1]),
                                color=color, fontsize=7.5, va="center")
            un = data["untracked"]
            if un["pressure"].size:
                ax.plot(un["pressure"], un["d"], "x", color=MUTED, ms=4,
                        alpha=0.5, ls="none", zorder=1,
                        label=f"untracked ({un['pressure'].size})")
                ax.legend(loc="best", fontsize=8, framealpha=0.3,
                          labelcolor=FG, facecolor=BG)
            ax.set_xlabel("pressure (GPa)")
            ax.set_ylabel("d-spacing (Å)")
            ax.set_title(
                f"Crystallite spot tracks — {len(tracks)}/{data['n_tracks_total']} "
                f"track(s), {n_pts} points; ▲ rising = d grows with P "
                f"({n_nlc} NLC candidate(s))", color=FG)

        canvas = self._embed_figure(self.unknowns_plot_frame, fig)
        self._unknowns_status.configure(
            text=(f"{len(tracks)} spot track(s) shown"
                  + (f", hkl labels from {Path(match).name}" if match
                     else " — pick an hkl table to label curves")))
        self._attach_hover(canvas, self._unknowns_status)

    def export_unknown_diagram_clicked(self):
        """Export observation + cluster summary CSVs for the Unknowns tab."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self._unknowns_status.configure(text="No analysis file loaded.")
            return
        default = Path(self.config.get("export_frames_dir", "") or
                       (Path(path).parent / "unknown_diagram"))
        dest = self.filedialog.askdirectory(
            title="Export unknown diagram CSVs",
            initialdir=str(default.parent if default.parent.exists() else Path(path).parent),
        )
        if not dest:
            return
        x_axis = getattr(self._unk_xaxis, "get", lambda: "frame")() or "frame"
        try:
            from .heatmap import write_unknown_diagram_csv
            man = write_unknown_diagram_csv(
                path,
                dest,
                x_axis=x_axis,
                min_obs_per_cluster=self._unknown_min_obs_value(),
                min_frames_per_cluster=self._unknown_min_frames_value(),
            )
        except Exception as e:
            self.log(f"Unknown diagram export failed: {e!r}", "WARN")
            self._unknowns_status.configure(text=f"Export failed: {e}")
            return
        msg = (f"Exported unknown diagram: {man['n_clusters']} clusters, "
               f"{man['n_obs']} obs -> {dest}")
        self._unknowns_status.configure(text=msg)
        self.log(msg)

    def export_spot_tracks_clicked(self):
        """Export /spots single-crystal tracks as the group-handoff CSV bundle."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self._unknowns_status.configure(text="No analysis file loaded.")
            return
        try:
            import h5py
            with h5py.File(path, "r") as h:
                has_spots = "spots" in h and "tracks" in h["spots"]
        except Exception:
            has_spots = False
        if not has_spots:
            self._unknowns_status.configure(
                text="No /spots in the analysis file — run bulkxrd-spots first.")
            return
        dest = self.filedialog.askdirectory(
            title="Export spot-track CSV bundle",
            initialdir=str(self.config.get("export_frames_dir", "")
                           or Path(path).parent),
        )
        if not dest:
            return
        match = self.filedialog.askopenfilename(
            title="Calculated reflection list for hkl matching (Cancel = skip)",
            filetypes=[("Reflection tables", "*.txt *.csv *.dat"), ("All files", "*.*")],
        ) or None
        try:
            from .spots import export_spot_tracks
            man = export_spot_tracks(path, dest, match=match,
                                     include_observations=True)
        except Exception as e:
            self.log(f"Spot-track export failed: {e!r}", "WARN")
            self._unknowns_status.configure(text=f"Export failed: {e}")
            return
        self.config["export_frames_dir"] = dest
        self.save_config(silent=True)
        msg = (f"Exported {man['n_tracks']} spot track(s), "
               f"{man['n_track_points']} points "
               f"(+{man['n_untracked_points']} untracked) -> {dest}")
        self._unknowns_status.configure(text=msg)
        self.log(msg)

    def export_unknown_frames_clicked(self):
        """Export residual patterns for frames carrying unknown observations."""
        data = getattr(self, "_unknowns_data", None)
        if not data or not data.get("ok"):
            self.load_unknowns()
            data = getattr(self, "_unknowns_data", None)
        frames = self._unknown_selected_frames(data)
        if not frames:
            self._unknowns_status.configure(text="No unknown frames to export.")
            return
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        stem = Path(path).stem.replace("_analysis", "")
        default_root = Path(self.config.get("export_frames_dir", "") or "outputs")
        initial = default_root if default_root.exists() else Path(path).parent
        dest = self.filedialog.askdirectory(
            title=f"Export {len(frames)} frame(s) with unknowns",
            initialdir=str(initial),
        )
        if not dest:
            return
        self.config["export_frames_dir"] = dest
        self.save_config(silent=True)
        out_dir = Path(dest) / f"unknown_frames_{stem}"
        self._do_export_frames(
            frames, out_dir, source="residual", peaks=True,
            residual_unknowns=True, status_label=self._unknowns_status,
        )

    # ------------------------------------------------------------------
    # Tab 12 — Grid map (per-frame scalars on the 2D scan grid)
    # ------------------------------------------------------------------

    def _tab_gridmap(self, frame):
        tk, ttk = self.tk, self.ttk

        _gm_intro = ttk.Label(
            frame,
            text=("For mapping runs: frames collected as a raster over the sample "
                  "are refolded onto their 2D scan grid and coloured by a "
                  "per-frame value (total/ROI intensity, contamination, peak "
                  "count, P, T, or one phase's matched intensity)."),
            foreground=MUTED, justify="left", wraplength=760,
        )
        _gm_intro.pack(anchor="w", padx=4, pady=(0, 6))
        self.autowrap(_gm_intro)

        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=(0, 2))
        ttk.Button(row1, text="Load grid map",
                   command=self.load_grid_map).pack(side="left", padx=4)
        ttk.Button(row1, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_gridmap_fig", None), "Grid map")
                   ).pack(side="left", padx=4)

        ttk.Label(row1, text="Value:", foreground=MUTED).pack(side="left", padx=(12, 2))
        self.vars["map_value"] = tk.StringVar(
            value=str(self.config.get("map_value", "total")))
        self._gm_value = ttk.Combobox(
            row1, textvariable=self.vars["map_value"],
            values=["total", "max", "contamination", "n_peaks",
                    "pressure", "temperature"],
            state="readonly", width=16)
        self._gm_value.pack(side="left", padx=2)
        _ToolTip(self._gm_value, HELP["map_value"])

        ttk.Label(row1, text="ROI min/max:", foreground=MUTED).pack(
            side="left", padx=(12, 2))
        self.vars["map_roi_min"] = tk.StringVar(
            value=str(self.config.get("map_roi_min", "")))
        _roi_lo = ttk.Entry(row1, textvariable=self.vars["map_roi_min"], width=8)
        _roi_lo.pack(side="left", padx=1)
        _ToolTip(_roi_lo, HELP["map_roi_min"])
        self.vars["map_roi_max"] = tk.StringVar(
            value=str(self.config.get("map_roi_max", "")))
        _roi_hi = ttk.Entry(row1, textvariable=self.vars["map_roi_max"], width=8)
        _roi_hi.pack(side="left", padx=1)
        _ToolTip(_roi_hi, HELP["map_roi_max"])

        self._gm_status = ttk.Label(row1, text="", foreground=MUTED)
        self._gm_status.pack(side="right", padx=8)

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=(0, 4))
        ttk.Label(row2, text="Layout:", foreground=MUTED).pack(side="left", padx=(4, 2))
        self.vars["map_layout"] = tk.StringVar(
            value=str(self.config.get("map_layout", "scan lines")))
        _lay_c = ttk.Combobox(row2, textvariable=self.vars["map_layout"],
                              values=["scan lines", "coordinates"],
                              state="readonly", width=11)
        _lay_c.pack(side="left", padx=2)
        _ToolTip(_lay_c, HELP["map_layout"])

        ttk.Label(row2, text="Frames per line:", foreground=MUTED).pack(
            side="left", padx=(12, 2))
        self.vars["map_line_len"] = tk.StringVar(
            value=str(self.config.get("map_line_len", "")))
        _len_e = ttk.Entry(row2, textvariable=self.vars["map_line_len"], width=8)
        _len_e.pack(side="left", padx=2)
        _ToolTip(_len_e, HELP["map_line_len"])

        ttk.Label(row2, text="Scan lines:", foreground=MUTED).pack(
            side="left", padx=(12, 2))
        self.vars["map_order"] = tk.StringVar(
            value=str(self.config.get("map_order", "horizontal")))
        _ord_c = ttk.Combobox(row2, textvariable=self.vars["map_order"],
                              values=["horizontal", "vertical"],
                              state="readonly", width=11)
        _ord_c.pack(side="left", padx=2)
        _ToolTip(_ord_c, HELP["map_order"])

        self.vars["map_serpentine"] = tk.BooleanVar(
            value=bool(self.config.get("map_serpentine", True)))
        _serp = ttk.Checkbutton(row2, text="Boustrophedon (serpentine)",
                                variable=self.vars["map_serpentine"])
        _serp.pack(side="left", padx=12)
        _ToolTip(_serp, HELP["map_serpentine"])

        self.gridmap_plot_frame = ttk.Frame(frame)
        self.gridmap_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.gridmap_plot_frame,
            text="Set the frames-per-line to your scan width and Load grid map.",
            foreground=MUTED,
        ).pack(anchor="center", expand=True)

    def load_grid_map(self):
        """Render the scan-grid map of the selected per-frame value."""
        self.pull_vars()
        self.save_config(silent=True)
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self._gm_status.configure(text="No analysis HDF5 — run Step 1 first.")
            return

        prev = getattr(self, "_gridmap_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._gridmap_fig = None
        for w in self.gridmap_plot_frame.winfo_children():
            w.destroy()

        def _fail(msg):
            self.ttk.Label(self.gridmap_plot_frame, text=msg, wraplength=520,
                           justify="left", foreground=WARN).pack(
                anchor="center", expand=True)
            self._gm_status.configure(text=msg)

        layout = str(self.config.get("map_layout", "scan lines") or "scan lines")
        line_len = 0
        if not layout.startswith("coord"):
            try:
                line_len = int(str(self.config.get("map_line_len", "")).strip())
                if line_len <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                _fail("Enter the frames-per-line of your scan (a positive "
                      "integer), or switch Layout to 'coordinates' if your "
                      "frames carry stage positions.")
                return

        def _opt(key):
            raw = str(self.config.get(key, "")).strip()
            return float(raw) if raw else None
        try:
            roi_lo, roi_hi = _opt("map_roi_min"), _opt("map_roi_max")
        except ValueError:
            _fail("ROI min/max must be numbers (or blank).")
            return

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
        except Exception as e:
            _fail(f"matplotlib unavailable: {e}")
            return
        import numpy as np
        from .heatmap import frame_values, frame_grid, grid_map

        kind = str(self.config.get("map_value", "total") or "total")
        if kind.startswith("phase:"):
            name = kind.split(":", 1)[1].strip()
            match = [p for p in self._enabled_phase_objects() if p.name == name]
            if not match:
                _fail(f"Phase {name!r} is not enabled on the Phases tab.")
                return
            pl = self._cached_layers(path, match)
            if not pl["ok"] or not pl["layers"]:
                _fail(pl.get("error") or f"No layer for {name!r} — run Step 3a.")
                return
            values = np.asarray(pl["layers"][0]["intensity_raw"], dtype=float)
            label = f"phase intensity — {name}"
        else:
            fv = frame_values(path, kind, radial_min=roi_lo, radial_max=roi_hi)
            if not fv["ok"]:
                _fail(fv["error"])
                return
            values = fv["values"]
            label = fv["label"]

        if layout.startswith("coord"):
            # Automatic placement from per-frame stage coordinates.
            from .frame_metadata import read_frame_metadata
            from .heatmap import coordinate_grid
            meta = read_frame_metadata(path)
            if not meta.get("ok"):
                _fail(meta.get("error") or "Could not read frame metadata.")
                return
            cg = coordinate_grid(meta["pos_x"], meta["pos_y"])
            if not cg["ok"]:
                _fail(cg["error"] + " (Frame meta tab: Import CSV with "
                      "pos_x/pos_y columns, or Read X/Y from headers.)")
                return
            gidx = cg["grid"]
            grid = np.full(gidx.shape, np.nan)
            m = gidx >= 0
            grid[m] = values[gidx[m]]
            xc = np.asarray(cg["x_centers"], dtype=float)
            yc = np.asarray(cg["y_centers"], dtype=float)

            def _half(c):
                return float(np.median(np.diff(c))) / 2.0 if c.size > 1 else 0.5
            hx, hy = _half(xc), _half(yc)
            extent = [xc[0] - hx, xc[-1] + hx, yc[0] - hy, yc[-1] + hy]
            origin = "lower"
            xlab, ylab = "stage x", "stage y"
            title = f"{label} — from frame coordinates"
            base_txt = (f"{cg['n_placed']} frames on a "
                        f"{grid.shape[0]}×{grid.shape[1]} coordinate grid")
            if cg["n_collisions"]:
                base_txt += f" ({cg['n_collisions']} collision(s))"

            def _cell(event):
                c = int(np.argmin(np.abs(xc - event.xdata)))
                r = int(np.argmin(np.abs(yc - event.ydata)))
                return r, c
        else:
            order = str(self.config.get("map_order", "horizontal") or "horizontal")
            serp = bool(self.config.get("map_serpentine", True))
            kwargs = ({"n_cols": line_len} if order == "horizontal"
                      else {"n_rows": line_len})
            try:
                grid = grid_map(values, order=order, serpentine=serp, **kwargs)
                gidx = frame_grid(values.size, order=order, serpentine=serp,
                                  **kwargs)
            except ValueError as e:
                _fail(str(e))
                return
            extent = None
            origin = "upper"
            xlab, ylab = "scan column", "scan row"
            path_txt = "boustrophedon" if serp else "unidirectional"
            title = f"{label} — {order} lines, {path_txt}"
            n_pad = int(np.sum(gidx < 0))
            base_txt = (f"{values.size} frames on a {grid.shape[0]}×"
                        f"{grid.shape[1]} grid"
                        + (f" ({n_pad} empty cells)" if n_pad else ""))

            def _cell(event):
                return int(round(event.ydata)), int(round(event.xdata))

        fig = Figure(figsize=(7, 6), dpi=100, layout="constrained")
        self._gridmap_fig = fig
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(grid, origin=origin, cmap="viridis",
                       interpolation="nearest", aspect="equal", extent=extent)
        try:
            cb = fig.colorbar(im, ax=ax, label=label)
            self._style_colorbar(cb)
        except Exception:
            pass
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, color=FG)
        self._style_ax(ax)

        canvas = self._embed_figure(self.gridmap_plot_frame, fig)
        self._gridmap_canvas = canvas
        self._gm_status.configure(text=base_txt)

        # Hover: resolve the cursor's grid cell back to the frame index.
        def _move(event):
            if event.inaxes is None or event.xdata is None or event.ydata is None:
                return
            r, c = _cell(event)
            if 0 <= r < gidx.shape[0] and 0 <= c < gidx.shape[1]:
                fi = int(gidx[r, c])
                if fi >= 0:
                    v = grid[r, c]
                    self._gm_status.configure(
                        text=f"{base_txt}   |   frame {fi}, value {v:.5g}")
        def _leave(event):
            self._gm_status.configure(text=base_txt)
        try:
            canvas.mpl_connect("motion_notify_event", _move)
            canvas.mpl_connect("axes_leave_event", _leave)
        except Exception:
            pass

    def _refresh_gridmap_values(self):
        """Extend the Value combo with per-phase entries for enabled phases."""
        base = ["total", "max", "contamination", "n_peaks",
                "pressure", "temperature"]
        names = [f"phase: {p.name}" for p in self._enabled_phase_objects()]
        try:
            self._gm_value.configure(values=base + names)
        except Exception:
            pass

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
            terminate_process_tree(self._run_proc)
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
