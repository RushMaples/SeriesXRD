"""Tabbed calibration GUI: stage 1 of the pipeline.

Pipeline order: calibrate (this stage) -> accept a PONI -> reduce
(reduce/gui.py) -> analysis (analysis/gui.py). Accepting a calibration here
hands its PONI + mask to the Reduction stage automatically.

Runs standalone (`seriesxrd-calib-gui`) or embedded as a pane in the unified
app.py window; either way it prints every major action to stdout, which the
Console Logs window captures live.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import os
import re
import sys
import traceback
import threading
import queue
import subprocess
import numpy as np

from ..core.config import TOOL_NAME, read_json, write_json, ensure_dir, now_iso, now_timestamp, safe_stem, default_workspace_paths, output_base
from ..core.env import check_dependencies, package_install_command, run_install_command
from ..core.io import read_detector_image
from ..core.masks import automatic_mask, save_mask_npz, save_mask_preview_png, load_mask_npz, polygon_to_mask
from ..core.naming import next_available_path
from ..core.processes import terminate_process_tree, worker_popen
from ..guikit.theme import BG, BG2, FG, ACCENT, ACCENT2, WARN, BORDER, ENTRY_BG, BTN_BG, BTN_ACT, MUTED
from ..guikit.tooltip import ToolTip as _ToolTip
from .dioptas import launch_dioptas, dioptas_manual_instructions
from .processing import export_accepted_generation, runtime_versions, read_poni_info, suggest_integration_settings


def _tk_imports():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    return tk, ttk, filedialog, messagebox


# Single source of truth for keV <-> Angstrom conversion (E[keV] = _HC / lambda[A]).
_HC_KEV_ANGSTROM = 12.398420


HELP = {
    "workspace_root":    "Single folder that holds this session's data and outputs. Raw/processed/figures/metadata/accepted/logs paths follow it automatically; use 'Show detailed paths' to override individually.",
    "backend_dir":       "Folder containing the seriesxrd package code (auto-detected). Only change this to run the GUI/worker from a different code copy.",
    "python_exe":        "Python interpreter used to run the GUI and pyFAI processing. Prefer the seriesxrd conda env.",
    "conda_exe":         "Path to conda or mamba. Used to launch the worker subprocess with the correct DLL path on Windows.",
    "dioptas_command":   "Optional full Dioptas launch command. If blank, the app tries the Dioptas Python field.",
    "image_file":        "Calibration detector image, usually a CeO2 TIFF/EDF frame.",
    "poni_file":         "Input PONI geometry file. The final accepted PONI is copied from this file.",
    "mask_negative":     "Masks pixels below zero. Useful for bad background corrections or invalid detector values.",
    "mask_zero":         "Masks exactly zero-valued pixels. Useful when zeros represent invalid/no-data detector areas.",
    "mask_nonfinite":    "Masks NaN and infinity values. This should usually stay on.",
    "saturated_threshold": "Optional upper threshold. Pixels above this value are masked as saturated/outlier pixels.",
    "npt_1d":            "Number of bins in the 1D intensity vs 2θ integration. Auto-set from the detector geometry (~1 bin per pixel of radial extent) when image and PONI are selected.",
    "npt_radial":        "Number of radial bins in the 2D cake plot. Auto-set to about half the 1D bins.",
    "npt_azimuthal":     "Number of azimuthal bins in the 2D cake plot (360 = 1° per bin).",
    "fast_qa":           "Optional flag to skip cake integration.",
    "radial_min":        "Optional lower 2θ limit (degrees). Applies ONLY when unit is 2th_deg; ignored for q units. Blank = automatic.",
    "radial_max":        "Optional upper 2θ limit (degrees). Applies ONLY when unit is 2th_deg; ignored for q units. Blank = automatic.",
    "coverage_threshold_pct": "Radial bins with azimuthal coverage below this % are set to NaN in 1D intensity.",
    "calibrant":         "Calibrant name for pyFAI (e.g. CeO2). Used to draw reference lines on intensity plots.",
}

class CalibrationApp:
    def __init__(self, config_path: "str | Path", parent=None):
        tk, ttk, filedialog, messagebox = _tk_imports()
        self.tk, self.ttk, self.filedialog, self.messagebox = tk, ttk, filedialog, messagebox
        self.config_path = Path(config_path).expanduser().resolve()
        self.config: Dict[str, Any] = read_json(self.config_path)
        self.config.setdefault("session_config_path", str(self.config_path))
        if parent is None:
            self._owns_root = True
            self.root = tk.Tk()
            self.root.title(f"{TOOL_NAME} Calibration")
            # Initial geometry derived from screen size, capped so the window always fits.
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            # FIX C7: reduced from 1420 to 1100 since help column is replaced by tooltips
            win_w = min(1100, sw - 80)
            win_h = min(920,  sh - 120)
            self.root.geometry(f"{win_w}x{win_h}")
            self.root.minsize(min(1100, win_w), min(760, win_h))
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        else:
            self._owns_root = False
            self.root = parent.winfo_toplevel()
        self._embed_parent = parent  # None when standalone, ttk.Frame when embedded
        self.vars: Dict[str, Any] = {}
        self.entry_widgets: Dict[str, Any] = {}
        self._suspend_wl_sync = False
        self.generations: List[Dict[str, Any]] = []
        self.current_generation_idx = -1
        self.manual_mask = None
        self.active_mask = None
        self.image_cache = None
        self.mask_polygons: List[Tuple[str, List[Tuple[float, float]]]] = []
        self.mpl = None
        self._viewer_photos: List[Any] = []
        self._viewer_zoom = 1.0
        self._final_item_vars: Dict[str, Any] = {}
        # Thread-safe logging: worker threads push lines onto this queue; a poller
        # on the Tk thread drains them into the text widget.
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        # History buffer so no lines are lost before the console window is opened.
        self._log_history: "list[str]" = []
        # QA generation busy guard + mask-acceptance dirty flag.
        self._qa_running = False
        # Handle to the running pyFAI worker subprocess (None when idle) so the
        # window can terminate it on close instead of orphaning it.
        self._qa_proc: "subprocess.Popen | None" = None
        self._mask_dirty = False
        # Callbacks fired with the handoff-JSON path after a successful accepted
        # export — the unified app uses this to live-populate the reduction pane.
        self._accept_listeners: "list" = []
        # Per-polygon mask rasterization cache for current_combined_mask.
        self._polygon_mask_cache: Dict[Any, np.ndarray] = {}
        # Last-applied workspace string, used to detect user-customized paths.
        self._last_workspace = ""
        # FIX A17b: debounce viewer resize — pending after-id and last width.
        self._resize_after_id: Optional[str] = None
        self._last_viewer_width: int = 0
        self._build_gui()
        self._drain_log_queue()
        self.log("GUI initialized")
        self.save_config(silent=True)
        # FIX A9: load prior generations from metadata folder after GUI is ready.
        self._load_existing_generations()
        # FIX C2: initial status bar refresh.
        self._update_status_bar()

    # ------------------------------------------------------------------
    # FIX C2: Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self, _parent=None):
        """Populate the persistent status bar (frame already packed at bottom)."""
        # FIX C2: the frame was pre-packed as side="bottom" before the notebook,
        # ensuring Tk allocates space for it before the expanding notebook fills the rest.
        ttk = self.ttk
        bar = self._status_bar_frame
        # Left: session name
        self.status_session = ttk.Label(bar, text="", foreground=MUTED, anchor="w")
        self.status_session.pack(side="left", padx=(6, 12))
        # Mask state
        self.status_mask = ttk.Label(bar, text="mask: none", foreground=MUTED, anchor="w")
        self.status_mask.pack(side="left", padx=(0, 12))
        # Current generation
        self.status_gen = ttk.Label(bar, text="", foreground=MUTED, anchor="w")
        self.status_gen.pack(side="left", padx=(0, 12))
        # Worker status (right-aligned)
        self.status_worker = ttk.Label(bar, text="idle", foreground=MUTED, anchor="e")
        self.status_worker.pack(side="right", padx=6)

    def _update_status_bar(self):
        """Refresh all status bar labels from current app state."""
        try:
            # Session name
            session = self.config.get("session_name", "")
            if hasattr(self, "status_session"):
                self.status_session.configure(text=f"session: {session}" if session else "session: (unnamed)")
            # Mask state
            amf = self.config.get("active_mask_file", "")
            aat = self.config.get("accepted_at", "") or ""
            if hasattr(self, "status_mask"):
                if amf:
                    ts_part = f" {aat}" if aat else ""
                    self.status_mask.configure(text=f"mask: accepted{ts_part}")
                else:
                    self.status_mask.configure(text="mask: none")
            # Generation counter
            if hasattr(self, "status_gen"):
                n = len(self.generations)
                if n > 0:
                    idx = max(0, self.current_generation_idx)
                    self.status_gen.configure(text=f"gen {idx+1}/{n}")
                else:
                    self.status_gen.configure(text="gen —/—")
            # Worker status
            if hasattr(self, "status_worker"):
                self.status_worker.configure(text="running" if self._qa_running else "idle")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # FIX A9: Reload prior generations on startup
    # ------------------------------------------------------------------

    def _load_existing_generations(self):
        """Scan the metadata tree for existing generation JSON files and populate
        self.generations so that numbering continues correctly after a relaunch.
        Mirrors the workflow path logic in processing.generate_qa_run.
        """
        try:
            metadata_root = self.config.get("metadata_root", "")
            if not metadata_root:
                return
            session_name = safe_stem(self.config.get("session_name", "calibration"))
            workflow_name = safe_stem(
                self.config.get("workflow_name") or f"calibration_review_{session_name}"
            )
            workflow_dir = Path(metadata_root) / workflow_name
            if not workflow_dir.is_dir():
                return
            # Glob all *_metadata_*.json files under genNNN sub-directories.
            candidates = list(workflow_dir.glob("gen*/*_metadata_*.json"))
            if not candidates:
                return
            loaded: List[Dict[str, Any]] = []
            seen_folders: set = set()
            for p in candidates:
                gen_folder = p.parent.name
                if gen_folder in seen_folders:
                    continue
                try:
                    md = read_json(p)
                    if not isinstance(md, dict) or "generation" not in md:
                        continue
                    seen_folders.add(gen_folder)
                    loaded.append(md)
                except Exception:
                    continue
            if not loaded:
                return
            # Sort by generation integer (genNNN → NNN), fall back to created_at.
            def _gen_sort_key(md):
                import re as _re
                lbl = md.get("generation", "")
                m = _re.match(r"gen(\d+)", str(lbl))
                if m:
                    return (int(m.group(1)), "")
                return (9999, md.get("created_at", ""))
            loaded.sort(key=_gen_sort_key)
            self.generations = loaded
            self.current_generation_idx = len(self.generations) - 1
            self.log(f"Loaded {len(loaded)} existing generation(s) from {workflow_dir}")
            # Refresh the final-save list if it already exists (it does after _build_gui).
            if hasattr(self, "_refresh_final_save_list"):
                self._refresh_final_save_list()
        except Exception as e:
            self.log(f"_load_existing_generations failed: {e}", "WARN")

    def _next_generation_index(self) -> int:
        """Next generation number = max(existing genNNN) + 1, not len()+1, so
        reloading generations with gaps (gen001+gen004) never re-creates gen003."""
        import re as _re
        mx = 0
        for md in self.generations:
            m = _re.match(r"gen(\d+)", str(md.get("generation", "")))
            if m:
                mx = max(mx, int(m.group(1)))
        return mx + 1

    # ------------------------------------------------------------------
    # Theme + build
    # ------------------------------------------------------------------

    def _apply_theme(self):
        # When embedded in the unified app the host already applied the shared
        # dark ttk theme; re-applying it here would restyle the whole host
        # window and its other panes. Only style globally when we own the root.
        if not self._owns_root:
            return
        tk, ttk = self.tk, self.ttk
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".",             background=BG,       foreground=FG,      fieldbackground=ENTRY_BG,
                         bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        style.configure("TFrame",        background=BG)
        style.configure("TLabel",        background=BG,       foreground=FG)
        style.configure("TButton",       background=BTN_BG,   foreground=FG,      bordercolor=BORDER, relief="flat", padding=4)
        style.map(       "TButton",      background=[("active", BTN_ACT), ("pressed", BTN_ACT)])
        style.configure("TEntry",        fieldbackground=ENTRY_BG, foreground=FG,  insertcolor=FG, bordercolor=BORDER)
        style.configure("TCombobox",     fieldbackground=ENTRY_BG, foreground=FG,  background=BTN_BG, arrowcolor=FG)
        style.map(       "TCombobox",    fieldbackground=[("readonly", ENTRY_BG)])
        style.configure("TCheckbutton",  background=BG,       foreground=FG,      indicatorcolor=ACCENT)
        style.map(       "TCheckbutton", background=[("active", BG2)])
        style.configure("TRadiobutton",  background=BG,       foreground=FG,      indicatorcolor=ACCENT)
        style.map(       "TRadiobutton", background=[("active", BG2)])
        style.configure("TScrollbar",    background=BG2,      troughcolor=BG,     arrowcolor=FG)
        style.configure("TNotebook",     background=BG,       tabmargins=[2, 5, 2, 0])
        style.configure("TNotebook.Tab", background=BG2,      foreground=FG,      padding=[10, 4])
        style.map(       "TNotebook.Tab",background=[("selected", BG)], foreground=[("selected", ACCENT)])
        style.configure("TSeparator",    background=BORDER)
        if self._owns_root:
            self.root.configure(bg=BG)
        try:
            self.root.option_add("*TCombobox*Listbox.background", ENTRY_BG)
            self.root.option_add("*TCombobox*Listbox.foreground", FG)
            self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        except Exception:
            pass

    def _build_gui(self):
        tk, ttk = self.tk, self.ttk
        self._apply_theme()

        # ---------------------------------------------------------
        # 1. Global Menu Bar (Native OS) — standalone mode only.
        #    In embedded mode the host window provides the menubar and
        #    calls register_menus() after construction.
        # ---------------------------------------------------------
        if self._owns_root:
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)

            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)

            self.register_menus(menubar, file_menu, None, help_menu)

        # ---------------------------------------------------------
        # 2. Main Window Wrapper
        # ---------------------------------------------------------
        _container = self._embed_parent if self._embed_parent is not None else self.root
        outer = ttk.Frame(_container, padding=6)
        outer.pack(fill="both", expand=True)
        
        # Minimal top bar for Window Controls
        topbar = ttk.Frame(outer)
        topbar.pack(fill="x", pady=(0, 6))
        # FIX C8: use portable cross-platform font instead of "Segoe UI"
        ttk.Label(topbar, text="Calibration", font=("TkDefaultFont", 14, "bold")).pack(side="left")
        ttk.Button(topbar, text="View log", command=self.open_console_logs).pack(side="right", padx=4)

        # FIX C2: status bar frame must be packed before the notebook so that
        # side="bottom" carves space before the expanding notebook fills the rest.
        self._status_bar_frame = ttk.Frame(outer, relief="sunken")
        self._status_bar_frame.pack(side="bottom", fill="x", pady=(2, 0))

        # ---------------------------------------------------------
        # 3. The 5-Step Notebook
        # ---------------------------------------------------------
        self.nb = ttk.Notebook(outer)
        self.nb.pack(fill="both", expand=True)
        self.tabs = {}
        
        for name in [
            "1 Inputs",
            "2 Mask",
            "3 Generate",
            "4 Review",
            "5 Accept calibration",
        ]:
            frame = ttk.Frame(self.nb, padding=8)
            self.nb.add(frame, text=name)
            self.tabs[name] = frame

        # ---------------------------------------------------------
        # 4. Routing logic to combine old tabs into new ones
        # ---------------------------------------------------------
        
        # TAB 1: Combine Paths and Inputs into ONE flowing page
        t1_master = self._scrollable(self.tabs["1 Inputs"])
        
        # Create isolated sub-frames so their internal grid rows never collide
        t1_paths_frame = ttk.Frame(t1_master)
        t1_paths_frame.pack(fill="x", expand=True)
        
        t1_inputs_frame = ttk.Frame(t1_master)
        t1_inputs_frame.pack(fill="both", expand=True)
        
        # Pass the isolated frames to the builder methods
        self._tab_data_paths(t1_paths_frame)
        self._tab_input(t1_inputs_frame)

        # TAB 2: Combine Masking (main) and Dioptas Launch (bottom)
        paned_2 = ttk.PanedWindow(self.tabs["2 Mask"], orient="vertical")
        paned_2.pack(fill="both", expand=True)
        
        t2_top = ttk.Frame(paned_2)
        t2_bot = ttk.Frame(paned_2)
        paned_2.add(t2_top, weight=4)
        paned_2.add(t2_bot, weight=1)
        
        self._tab_mask(t2_top)
        self._tab_dioptas(t2_bot)

        # TAB 3: Generation
        self._tab_generate(self.tabs["3 Generate"])

        # TAB 4: Viewer fills the full tab (log is now a separate top-bar window)
        self._tab_viewer(self.tabs["4 Review"])

        # TAB 5: Accept calibration
        self._tab_final(self.tabs["5 Accept calibration"])

        # FIX C2: persistent status bar at the very bottom of the main window.
        self._build_status_bar(outer)

        # Bind the save shortcut (standalone only; embedded, the host owns Ctrl-S).
        if self._owns_root:
            self.root.bind("<Control-s>", lambda e: self.save_config())

    def show_about(self):
        """Show an About dialog with runtime versions and the config path."""
        try:
            versions = runtime_versions()
        except Exception as e:
            versions = {"error": repr(e)}
        lines = [f"{TOOL_NAME} Calibration Review", ""]
        for k in ("seriesxrd", "pyFAI", "numpy", "python", "platform"):
            if k in versions:
                lines.append(f"{k}: {versions[k]}")
        for k, v in versions.items():
            if k not in ("seriesxrd", "pyFAI", "numpy", "python", "platform"):
                lines.append(f"{k}: {v}")
        lines.append("")
        lines.append(f"Config: {self.config_path}")
        self.messagebox.showinfo("About", "\n".join(lines))

    # ------------------------------------------------------------------
    # Reusable field helpers
    # ------------------------------------------------------------------

    def _scrollable(self, parent):
        tk, ttk = self.tk, self.ttk
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0, bg=BG)
        # Use tk.Frame with explicit bg to prevent white-region bleedthrough on Windows
        frame  = tk.Frame(canvas, bg=BG, borderwidth=0, highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        win = canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))

        def _on_mousewheel(event):
            # Windows and Mac handle scroll deltas differently.
            # Mac sends 1 or -1. Windows sends multiples of 120.
            import sys
            if sys.platform == "darwin":
                canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_linux_scroll_up(event):
            canvas.yview_scroll(-1, "units")

        def _on_linux_scroll_down(event):
            canvas.yview_scroll(1, "units")

        
        def _bind_to_mousewheel(event):
            # Bind globally only when hovering over this specific scroll area
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_linux_scroll_up)
            canvas.bind_all("<Button-5>", _on_linux_scroll_down)

        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_to_mousewheel)
        canvas.bind("<Leave>", _unbind_from_mousewheel)

        return frame
    
    def field(self, parent, key, label, browse=None, row=None, width=80, help_key=None):
        # FIX C7: replaced permanent column-3 help label with a _ToolTip on the
        # label and entry widgets, narrowing the window layout.
        tk, ttk = self.tk, self.ttk
        if row is None:
            row = len(parent.grid_slaves())
        var = tk.StringVar(value=str(self.config.get(key, "")))
        self.vars[key] = var
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=4, pady=3)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
        self.entry_widgets[key] = entry
        if browse:
            ttk.Button(parent, text="Browse", command=lambda: self.browse_into(key, browse)).grid(row=row, column=2, sticky="ew", padx=4, pady=3)
        txt = HELP.get(help_key or key, "")
        if txt:
            _ToolTip(lbl,   txt)
            _ToolTip(entry, txt)
        parent.columnconfigure(1, weight=1)
        return var

    def checkbox(self, parent, key, label, row=None, help_text=""):
        # FIX C7: replaced permanent column-2 help label with a _ToolTip on the
        # checkbutton widget, narrowing the window layout.
        tk, ttk = self.tk, self.ttk
        if row is None:
            row = len(parent.grid_slaves())
        var = tk.BooleanVar(value=bool(self.config.get(key, True)))
        self.vars[key] = var
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=3)
        txt = help_text or HELP.get(key, "")
        if txt:
            _ToolTip(cb, txt)
        return var

    def combobox(self, parent, key, label, values, row=None, width=30, help_key=None):
        # FIX C7: replaced permanent column-3 help label with a _ToolTip.
        tk, ttk = self.tk, self.ttk
        if row is None:
            row = len(parent.grid_slaves())
        current = str(self.config.get(key, values[0] if values else ""))
        if current not in values and values:
            current = values[0]
        var = tk.StringVar(value=current)
        self.vars[key] = var
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=4, pady=3)
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
        cb.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
        self.entry_widgets[key] = cb
        txt = HELP.get(help_key or key, "")
        if txt:
            _ToolTip(lbl, txt)
            _ToolTip(cb,  txt)
        parent.columnconfigure(1, weight=1)
        return var

    def browse_into(self, key, mode):
        current = self.vars[key].get() if key in self.vars else ""
        if mode == "dir":
            value = self.filedialog.askdirectory(title=f"Select {key}", initialdir=current or os.getcwd())
        elif mode == "python":
            fts   = [("Python", "python.exe"), ("All files", "*.*")] if os.name == "nt" else [("All files", "*")]
            value = self.filedialog.askopenfilename(title=f"Select {key}", initialdir=str(Path(current).parent if current else Path.cwd()), filetypes=fts)
        elif mode == "image":
            value = self.filedialog.askopenfilename(title=f"Select {key}", initialdir=current or self.config.get("raw_data_dir", os.getcwd()),
                                                    filetypes=[("Detector images", "*.tif *.tiff *.edf *.cbf *.img *.png"), ("All files", "*.*")])
        elif mode == "poni":
            value = self.filedialog.askopenfilename(title=f"Select {key}", initialdir=current or os.getcwd(),
                                                    filetypes=[("PONI files", "*.poni"), ("All files", "*.*")])
        elif mode == "mask":
            value = self.filedialog.askopenfilename(title=f"Select {key}", initialdir=current or os.getcwd(),
                                                    filetypes=[("Mask files", "*.npy *.npz *.tif *.tiff *.edf *.png"), ("All files", "*.*")])
        else:
            value = self.filedialog.askopenfilename(title=f"Select {key}", initialdir=current or os.getcwd())
        if value:
            self.vars[key].set(value)
            self.config[key] = value
            self.log(f"Set {key} = {value}")
            # FIX C10a: track poni/image recently used files.
            if mode == "poni":
                self._add_recent("poni", value)
            elif mode == "image":
                self._add_recent("image", value)

    def pull_vars(self):
        for key, var in self.vars.items():
            try:
                self.config[key] = var.get()
            except Exception:
                pass
        self.config["updated_at"] = now_iso()
        self.config["session_config_path"] = str(self.config_path)

    def save_config(self, silent=False):
        self.pull_vars()
        write_json(self.config_path, self.config)
        if not silent:
            self.log(f"Saved session config: {self.config_path}")

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
        self.root.after(100, self._drain_log_queue)

    # ------------------------------------------------------------------
    # Tab 1 — Session / Paths
    # ------------------------------------------------------------------

    def _tab_data_paths(self, frame): 
        
        self.field(frame, "session_name", "Session / run name")
        self.ttk.Separator(frame, orient="horizontal").grid(row=len(frame.grid_slaves()), column=0, columnspan=4, sticky="ew", pady=10)
        
        
        # 1. The Main Workspace Path
        if "workspace_root" not in self.config:
            self.config["workspace_root"] = str(output_base(self.config))
            
        ws_var = self.field(frame, "workspace_root", "Main workspace folder", "dir")
        # FIX E: initialise _last_workspace from config so we know the original baseline.
        self._last_workspace = self.config.get("workspace_root", "")

        # 2. Auto-Update Logic — only overwrites derived paths when their current value
        # matches the default derived from the PREVIOUS workspace (or is empty), so
        # hand-customised paths are never clobbered on every keystroke.
        def _on_workspace_commit(*args):
            try:
                new_ws = ws_var.get().strip()
                if not new_ws:
                    return
                new_defaults = default_workspace_paths(new_ws)
                old_defaults = default_workspace_paths(self._last_workspace) if self._last_workspace else {}
                derived_keys = ["raw_data_dir", "processed_root", "figures_root",
                                "metadata_root", "accepted_output_root", "logs_root"]
                for k in derived_keys:
                    if k in self.vars:
                        current = self.vars[k].get()
                        old_default = old_defaults.get(k, "")
                        # Only overwrite if the field is empty or still shows the old default.
                        if current == "" or current == old_default:
                            self.vars[k].set(new_defaults.get(k, current))
                    self.config[k] = self.vars[k].get() if k in self.vars else new_defaults.get(k, "")
                self._last_workspace = new_ws
            except Exception as e:
                self.log(f"Path auto-update failed: {e}", "WARN")

        ws_entry = self.entry_widgets.get("workspace_root")
        if ws_entry is not None:
            ws_entry.bind("<FocusOut>", _on_workspace_commit)
            ws_entry.bind("<Return>",   _on_workspace_commit)
        # Wrap the Browse button so it also triggers the update after selection.
        _orig_browse = self.browse_into
        def _ws_browse_and_commit(key, mode):
            _orig_browse(key, mode)
            if key == "workspace_root":
                _on_workspace_commit()
        # Monkey-patch the Browse button for workspace_root only — we re-bind
        # the button command to go through the commit hook.
        if ws_entry is not None:
            # Find the Browse button (it was added by self.field) and rebind it.
            try:
                parent_frame = ws_entry.master
                for child in parent_frame.grid_slaves():
                    if hasattr(child, "configure") and hasattr(child, "invoke"):
                        # It's a Button on the same row as the workspace entry
                        if child.grid_info().get("row") == ws_entry.grid_info().get("row") and \
                           child.grid_info().get("column") == 2:
                            child.configure(command=lambda: _ws_browse_and_commit("workspace_root", "dir"))
                            break
            except Exception:
                pass

        # 3. The Toggle Button
        self.show_advanced_paths = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(
            frame, 
            text="Show detailed paths", 
            variable=self.show_advanced_paths,
            command=self._toggle_advanced_paths
        ).grid(row=len(frame.grid_slaves()), column=0, columnspan=2, sticky="w", padx=4, pady=10)

        # 4. The Advanced Frame
        self._adv_paths_row = len(frame.grid_slaves())
        self.adv_paths_frame = self.ttk.Frame(frame)
        
        # Add the granular fields inside the advanced frame
        self.field(self.adv_paths_frame, "raw_data_dir",         "↳ Raw data folder",      "dir")
        self.field(self.adv_paths_frame, "processed_root",       "↳ Processed data root",  "dir")
        self.field(self.adv_paths_frame, "figures_root",         "↳ Figures root",         "dir")
        self.field(self.adv_paths_frame, "metadata_root",        "↳ Metadata root",        "dir")
        self.field(self.adv_paths_frame, "accepted_output_root", "↳ Accepted root",        "dir")
        self.field(self.adv_paths_frame, "logs_root",            "↳ Logs root",            "dir")

    def _toggle_advanced_paths(self):
        """Shows or hides the advanced path settings."""
        if self.show_advanced_paths.get():
            # Dynamically allocate the space
            self.adv_paths_frame.grid(row=self._adv_paths_row, column=0, columnspan=4, sticky="ew")
        else:
            self.adv_paths_frame.grid_forget()

    # Toplevel: env settings
    def open_env_settings(self):

        if hasattr(self, "env_window") and self.env_window.winfo_exists():
            self.env_window.lift()
            return

        tk, ttk = self.tk, self.ttk
        self.env_window = tk.Toplevel(self.root)
        self.env_window.title("Environment & Executable Settings")
        self.env_window.geometry("900x450")
        self.env_window.configure(bg=BG)

        self.env_window.grab_set() 

        frame = self._scrollable(self.env_window)
        
        self.field(frame, "notebook_dir",        "Workspace folder",             "dir")
        self.field(frame, "backend_dir",         "Package folder",               "dir")
        self.field(frame, "python_exe",          "Python interpreter",           "python")
        self.field(frame, "conda_exe",           "Conda / mamba executable",     "python")
        self.field(frame, "conda_env_name",      "Conda env name")
        self.field(frame, "dioptas_command",     "Dioptas command")
        self.field(frame, "dioptas_python",      "Dioptas Python",               "python")
        
        self.ttk.Separator(frame, orient="horizontal").grid(row=len(frame.grid_slaves()), column=0, columnspan=4, sticky="ew", pady=15)
        
        # Dependency check buttons
        row = len(frame.grid_slaves())
        ttk.Button(frame, text="Check dependencies",     command=self.check_deps).grid(row=row, column=0, padx=4, pady=8, sticky="w")
        ttk.Button(frame, text="Install missing packages", command=self.install_missing).grid(row=row, column=1, padx=4, pady=8, sticky="w")
        
        # FIX C8: portable font
        ttk.Label(frame, text="Checks numpy, pyFAI, fabio, matplotlib, Pillow, tkinter.", foreground=MUTED).grid(row=row+1, column=0, columnspan=4, sticky="w", padx=4)
        ttk.Label(frame, text="Installs any missing required packages using conda or pip.", foreground=MUTED).grid(row=row+2, column=0, columnspan=4, sticky="w", padx=4)

    # ------------------------------------------------------------------
    # FIX B6: PONI inspector
    # ------------------------------------------------------------------

    def _inspect_poni(self, force: bool = True):
        """Display key PONI parameters in poni_info_label.

        Uses the shared processing.read_poni_info() text parser (no pyFAI
        import in the Tk process) so there is a single PONI reader across the
        package.
        """
        if not hasattr(self, "poni_info_label"):
            return
        self.pull_vars()
        path = self.config.get("poni_file", "").strip()
        if not path or not Path(path).exists():
            self.poni_info_label.configure(text="PONI file not found or not set.")
            return
        try:
            info = read_poni_info(path)
            dist     = info.get("dist")
            wl_m     = info.get("wavelength_m")
            p1       = info.get("poni1")
            p2       = info.get("poni2")
            det_name = info.get("detector") or ""
            shape    = info.get("shape")
            wl_ang   = wl_m * 1e10 if isinstance(wl_m, (int, float)) else None
            e_kev    = (_HC_KEV_ANGSTROM / wl_ang) if (wl_ang and wl_ang > 0) else None
            def _g(v, fmt="{:.6g}"):
                return fmt.format(v) if isinstance(v, (int, float)) else "?"
            lines_out = [
                f"Detector : {det_name or '?'}" + (f"  shape={tuple(shape)}" if shape else ""),
                f"Distance : {_g(dist)} m",
                f"Wavelength: {_g(wl_m)} m"
                + (f"  ({wl_ang:.4g} A)  => Energy: {e_kev:.4f} keV" if e_kev else ""),
                f"Poni1={_g(p1)}  Poni2={_g(p2)}",
            ]
            # FIX B6: compare with form energy_kev / wavelength_m and warn on mismatch.
            form_e   = str(self.config.get("energy_kev",   "")).strip()
            form_wl  = str(self.config.get("wavelength_m", "")).strip()
            mismatches = []
            try:
                if form_e and e_kev:
                    fe = float(form_e)
                    if abs(fe - e_kev) / max(abs(e_kev), 1e-9) > 0.001:
                        mismatches.append(f"  energy: PONI={e_kev:.4f} keV vs form={fe:.4f} keV")
            except Exception:
                pass
            try:
                if form_wl and isinstance(wl_m, (int, float)):
                    fwl = float(form_wl)
                    if abs(fwl - wl_m) / max(abs(wl_m), 1e-20) > 0.001:
                        mismatches.append(f"  wavelength: PONI={wl_m:.6g} m vs form={fwl:.6g} m")
            except Exception:
                pass
            if mismatches:
                lines_out.append("WARNING - mismatch with form fields:")
                lines_out.extend(mismatches)
            self._autofill_poni_fields(info, force=force)
            self.poni_info_label.configure(text="\n".join(lines_out))
            self.log(f"PONI inspected: {path}" + (f"  E={e_kev:.4f} keV" if e_kev else ""))
        except Exception as e:
            self.poni_info_label.configure(text=f"PONI parse error: {e}")
            self.log(f"PONI inspect error: {e}", "WARN")

    def _auto_inspect_poni(self):
        """Called from poni_file var trace — silently inspect if path is valid."""
        try:
            self.pull_vars()
            path = self.config.get("poni_file", "").strip()
            if path and Path(path).exists():
                self._inspect_poni(force=False)
        except Exception:
            pass

    def _autofill_poni_fields(self, info: dict, force: bool = True):
        """Populate the editable geometry fields from a parsed PONI.

        force=True (explicit Inspect button) overwrites existing values;
        force=False (auto-inspect on path change) fills only blank fields so
        manual edits are preserved.
        """
        wl_m = info.get("wavelength_m")
        e_kev = None
        if isinstance(wl_m, (int, float)) and wl_m > 0:
            e_kev = _HC_KEV_ANGSTROM / (wl_m * 1e10)
        values = {
            "detector_name": info.get("detector") or "",
            "wavelength_m":  wl_m,
            "energy_kev":    e_kev,
            "pixel1":        info.get("pixel1"),
            "pixel2":        info.get("pixel2"),
            "dist":          info.get("dist"),
            "poni1":         info.get("poni1"),
            "poni2":         info.get("poni2"),
            "rot1":          info.get("rot1"),
            "rot2":          info.get("rot2"),
            "rot3":          info.get("rot3"),
        }
        self._suspend_wl_sync = True
        try:
            for key, val in values.items():
                if val is None or val == "":
                    continue
                if key not in self.vars:
                    continue
                current = str(self.vars[key].get()).strip()
                if not force and current:
                    continue
                sval = val if isinstance(val, str) else f"{val:.8g}"
                self.vars[key].set(sval)
                self.config[key] = sval
        finally:
            self._suspend_wl_sync = False

    def _orientation_text(self) -> str:
        """Return the current orientation status text for the label."""
        flip = bool(self.config.get("dioptas_image_flip", True))
        return ("Flip ON (Dioptas alignment)" if flip
                else "Flip OFF (file orientation)  — click Preview to compare both orientations")

    def preview_cake_orientation(self):
        """Launch a worker preview of both flip orientations, then open a dialog
        so the user can pick the orientation with straight calibrant rings. The
        choice sets ``dioptas_image_flip`` (this replaces the old manual toggle)."""
        if self._qa_running:
            self.log("A worker is already running", "WARN")
            return
        self.pull_vars()
        image_file = str(self.config.get("image_file", "") or "").strip()
        poni_file  = str(self.config.get("poni_file",  "") or "").strip()
        if not image_file or not Path(image_file).is_file():
            self.messagebox.showerror("Preview", "Select a valid calibration image first.")
            return
        if not poni_file or not Path(poni_file).is_file():
            self.messagebox.showerror("Preview", "Select a valid PONI file first.")
            return
        self.save_config(silent=True)
        python_exe  = Path(self.config.get("python_exe", sys.executable))
        backend_dir = self.config.get("backend_dir", str(Path(__file__).resolve().parents[1]))
        logs_root   = self.config.get("logs_root", "") or str(output_base(self.config) / "logs")
        ensure_dir(Path(logs_root))
        out_json    = str(next_available_path(Path(logs_root) / f"worker_preview_{now_timestamp()}.json"))
        worker_script = str(Path(backend_dir) / "calib" / "worker.py")
        cmd = [str(python_exe), worker_script,
               "--config", str(self.config_path),
               "--output-json", out_json,
               "--mode", "preview"]
        import os as _os
        worker_env = dict(_os.environ)
        _prefix_dir = python_exe.parent
        for _sub in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin"):
            _d = str(_prefix_dir / _sub)
            if Path(_d).is_dir() and _d.lower() not in worker_env.get("PATH", "").lower():
                worker_env["PATH"] = _d + _os.pathsep + worker_env.get("PATH", "")

        self._qa_running = True
        self._update_status_bar()
        self.log("Starting cake orientation preview")

        def _worker_thread():
            try:
                proc = worker_popen(
                    cmd, cwd=backend_dir, env=worker_env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                try:
                    for line in proc.stdout:
                        self.log(line.rstrip())
                except (ValueError, OSError):
                    pass
                rc = proc.wait()
                self.log(f"Preview worker returncode={rc}")
                self.root.after(0, lambda: self._preview_done(rc, out_json))
            except Exception as e:
                err = repr(e)
                self.log(f"Preview worker launch error: {err}", "ERROR")
                self.root.after(0, lambda: self._preview_error(err))
        threading.Thread(target=_worker_thread, daemon=True).start()

    def _preview_error(self, err_msg: str):
        self._qa_running = False
        self._update_status_bar()
        self.messagebox.showerror("Preview error", err_msg)

    def _preview_done(self, rc: int, out_json: str):
        self._qa_running = False
        self._update_status_bar()
        if rc != 0 or not Path(out_json).exists():
            self.messagebox.showerror(
                "Preview failed",
                f"Preview worker return code {rc}.\n"
                "See the console log window (Open Console Logs, top of window) for details.")
            return
        try:
            res = read_json(out_json)
        except Exception as e:
            self.messagebox.showerror("Preview failed", f"Could not read preview output: {e!r}")
            return
        self._show_orientation_dialog(res)

    def _show_orientation_dialog(self, res: dict):
        try:
            from PIL import Image, ImageTk  # type: ignore
        except ImportError:
            self.messagebox.showerror("Preview", "Pillow is required to display the preview.")
            return
        tk, ttk = self.tk, self.ttk
        win = tk.Toplevel(self.root)
        win.title("Choose cake orientation")
        win.configure(bg=BG)
        win.grab_set()
        ttk.Label(win, text="Pick the orientation whose calibrant rings are straight, vertical lines. "
                  "Wavy/sinusoidal rings mean the wrong orientation (or geometry that needs refining).",
                  wraplength=1080, justify="left").grid(row=0, column=0, columnspan=2, padx=10, pady=8, sticky="w")
        self._orient_photos = []

        def _panel(col, png, title, flip_value, btn_text):
            frm = ttk.Frame(win)
            frm.grid(row=1, column=col, padx=10, pady=6, sticky="n")
            ttk.Label(frm, text=title, font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
            if png and Path(png).exists():
                img = Image.open(png)
                maxw = 540
                if img.width > maxw:
                    ratio = maxw / img.width
                    img = img.resize((maxw, int(img.height * ratio)), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._orient_photos.append(photo)
                ttk.Label(frm, image=photo).pack()
            else:
                ttk.Label(frm, text="(preview image missing)").pack()
            ttk.Button(frm, text=btn_text,
                       command=lambda: self._apply_orientation_choice(flip_value, win)).pack(pady=6, fill="x")

        _panel(0, res.get("flip_off_png", ""), "Flip OFF (file orientation)", False, "Use this  (Flip OFF)")
        _panel(1, res.get("flip_on_png", ""),  "Flip ON (Dioptas alignment)", True,  "Use this  (Flip ON)")
        ttk.Button(win, text="Cancel", command=win.destroy).grid(row=2, column=0, columnspan=2, pady=8)

    def _apply_orientation_choice(self, flip_value: bool, win):
        self.config["dioptas_image_flip"] = bool(flip_value)
        # Mask editor cache depends on orientation — clear it (old toggle behavior).
        if getattr(self, "image_cache", None) is not None:
            self.image_cache = None
            self.manual_mask = None
            self.mask_polygons = []
            try:
                self.refresh_mask_display()
            except Exception:
                pass
        if hasattr(self, "orientation_label"):
            self.orientation_label.configure(text=self._orientation_text())
        self.save_config(silent=True)
        self.log(f"Cake orientation set: flip={'ON' if flip_value else 'OFF'}")
        try:
            win.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tab 2 — Calibration Input
    # ------------------------------------------------------------------

    def _tab_input(self, frame):
        from pathlib import Path
        
        self.ttk.Separator(frame, orient="horizontal").grid(row=len(frame.grid_slaves()), column=0, columnspan=4, sticky="ew", pady=15)
        img_var = self.field(frame, "image_file", "Calibration image", "image")
        self.field(frame, "poni_file",      "Input PONI",         "poni")

        # FIX B6: Inspect PONI button + read-only info label below the poni_file field.
        poni_btn_row = len(frame.grid_slaves())
        self.ttk.Button(frame, text="Inspect PONI", command=self._inspect_poni).grid(
            row=poni_btn_row, column=0, padx=4, pady=4, sticky="w")
        self.poni_info_label = self.ttk.Label(frame, text="(no PONI loaded)", foreground=MUTED, wraplength=600, justify="left")
        self.poni_info_label.grid(row=poni_btn_row, column=1, columnspan=3, sticky="w", padx=4, pady=4)
        # Auto-inspect when poni_file var changes.
        if "poni_file" in self.vars:
            self.vars["poni_file"].trace_add("write", lambda *_: self._auto_inspect_poni())

        # Cake orientation preview — replaces the old manual flip checkbox. Runs
        # a fast low-res dual-orientation cake in the worker so the user can pick
        # the orientation with straight calibrant rings before a full generation.
        prev_row = len(frame.grid_slaves())
        self.ttk.Button(frame, text="Preview cake orientation",
                        command=self.preview_cake_orientation).grid(
            row=prev_row, column=0, padx=4, pady=4, sticky="w")
        self.orientation_label = self.ttk.Label(
            frame, text=self._orientation_text(), foreground=MUTED,
            wraplength=600, justify="left")
        self.orientation_label.grid(row=prev_row, column=1, columnspan=3, sticky="w", padx=4, pady=4)

        self.field(frame, "input_mask_file","Optional input mask", "mask")
        
        pyfai_calibrants = [
            "CeO2", "LaB6", "Si", "AgBh", "Al", "Au", "Cr2O3", "CuO", 
            "LaB6_SRM660a", "LaB6_SRM660b", "LaB6_SRM660c", "NaCl", "Ni", 
            "Pt", "Si_SRM640", "Si_SRM640a", "Si_SRM640b", "Si_SRM640c", 
            "Si_SRM640d", "Si_SRM640e", "TiO2", "ZnO"
        ]
        
        self.config.setdefault("calibrant", "CeO2")
        cal_var = self.combobox(frame, "calibrant", "Calibrant", values=pyfai_calibrants)

        def _autofill_poni(img_path_obj: Path):
            current = self.vars["poni_file"].get().strip() if "poni_file" in self.vars else ""
            if current and Path(current).exists():
                return
            stem = img_path_obj.stem                 # e.g. "CeO2_30keV_168mm_0deg_001"
            base_name = re.sub(r"_\d+$", "", stem)   # strip trailing frame number
            # Preference: exact stem match > base-name match > the only PONI in the folder.
            exact = img_path_obj.with_suffix(".poni")
            candidates = [exact] if exact.exists() else []
            if not candidates:
                candidates = [p for p in sorted(img_path_obj.parent.glob("*.poni")) if base_name in p.name]
            if not candidates:
                candidates = sorted(img_path_obj.parent.glob("*.poni"))
                if len(candidates) != 1:
                    return
            poni = candidates[0]
            self.vars["poni_file"].set(str(poni))
            self.config["poni_file"] = str(poni)
            self.log(f"Auto-loaded matching PONI: {poni.name}")

        def _autofill_mask(img_path_obj: Path):
            current = self.vars["input_mask_file"].get().strip() if "input_mask_file" in self.vars else ""
            if current:
                return
            stem = img_path_obj.with_suffix("")
            for suffix in ("_mask.npz", "_mask.npy", ".npz"):
                cand = Path(str(stem) + suffix)
                if cand.exists():
                    self.vars["input_mask_file"].set(str(cand))
                    self.config["input_mask_file"] = str(cand)
                    self.log(f"Auto-loaded matching mask: {cand.name}")
                    return

        def _auto_detect_calibrant(*args):
            # A failed autofill must never break the input tab, and Tk swallows
            # trace-callback exceptions silently — so catch and log everything.
            try:
                img_path = img_var.get().strip()
                if not img_path:
                    return
                # Match against the filename only, not parent folder names.
                filename = Path(img_path).name.lower()
                # Sort by length descending so "LaB6_SRM660a" is checked before "LaB6".
                sorted_calibrants = sorted(pyfai_calibrants, key=len, reverse=True)
                detected = next((cal for cal in sorted_calibrants if cal.lower() in filename), None)
                # Only act on a positive match — don't reset a manual selection.
                if detected and cal_var.get() != detected:
                    cal_var.set(detected)
                    self.config["calibrant"] = detected
                    self.log(f"Auto-detected calibrant '{detected}' from filename")
                img_path_obj = Path(img_path)
                if img_path_obj.is_file():
                    _autofill_poni(img_path_obj)
                    _autofill_mask(img_path_obj)
                    self._suggest_generation_bins(force=False)
            except Exception as e:
                self.log(f"Input autocomplete failed: {e!r}", "WARN")

        img_var.trace_add("write", _auto_detect_calibrant)


        for key, label in [
            ("detector_name", "Detector name"), ("energy_kev", "Energy (keV)"), ("wavelength_m", "Wavelength (m)"),
            ("pixel1", "Pixel size 1 (m)"),     ("pixel2", "Pixel size 2 (m)"), ("dist", "Sample-detector distance (m)"),
            ("poni1",  "PONI1 (m)"),             ("poni2", "PONI2 (m)"),
            ("rot1",   "rot1"),                  ("rot2",  "rot2"),              ("rot3", "rot3"),
        ]:
            self.field(frame, key, label)

        # Keep energy_kev and wavelength_m in sync (E[keV] = _HC_KEV_ANGSTROM / λ[Å]).
        def _sync_from_wavelength(*_):
            if getattr(self, "_suspend_wl_sync", False):
                return
            self._suspend_wl_sync = True
            try:
                wl = float(self.vars["wavelength_m"].get().strip())
                if wl > 0:
                    self.vars["energy_kev"].set(f"{_HC_KEV_ANGSTROM / (wl * 1e10):.6g}")
            except (ValueError, ZeroDivisionError, KeyError):
                pass
            finally:
                self._suspend_wl_sync = False

        def _sync_from_energy(*_):
            if getattr(self, "_suspend_wl_sync", False):
                return
            self._suspend_wl_sync = True
            try:
                e = float(self.vars["energy_kev"].get().strip())
                if e > 0:
                    self.vars["wavelength_m"].set(f"{_HC_KEV_ANGSTROM / e * 1e-10:.6g}")
            except (ValueError, ZeroDivisionError, KeyError):
                pass
            finally:
                self._suspend_wl_sync = False

        if "wavelength_m" in self.vars:
            self.vars["wavelength_m"].trace_add("write", _sync_from_wavelength)
        if "energy_kev" in self.vars:
            self.vars["energy_kev"].trace_add("write", _sync_from_energy)

        self.ttk.Label(frame, text="Distance, PONI, rotations, and wavelength are applied to integration. "
                       "Detector name and pixel size come from the PONI file.",
                       foreground=MUTED, wraplength=600, justify="left").grid(
            row=len(frame.grid_slaves()), column=0, columnspan=4, sticky="w", padx=4, pady=(0, 6))

        self.ttk.Button(frame, text="Load image preview into Mask tab", command=self.load_image_for_mask).grid(
            row=len(frame.grid_slaves()), column=0, padx=4, pady=8, sticky="w")

    # ------------------------------------------------------------------
    # Tab 3 — Masking
    # ------------------------------------------------------------------

    def _tab_mask(self, parent):
        tk, ttk = self.tk, self.ttk
        
        # Use a horizontal PanedWindow for an adjustable sidebar split
        paned = ttk.PanedWindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True)
        
        # Create a scrollable container for the sidebar so buttons are never cut off vertically
        left_container = ttk.Frame(paned)
        right = ttk.Frame(paned)
        
        # Add a light padding to the sidebar elements
        left = self._scrollable(left_container)
        
        paned.add(left_container, weight=0)  # Sidebar holds its ideal width
        paned.add(right, weight=1)           # Matplotlib canvas claims all remaining space
        
        # --- 1. Compact Checkboxes---
        def _compact_check(key, label, default=True):
            var = tk.BooleanVar(value=bool(self.config.get(key, default)))
            self.vars[key] = var
            cb = ttk.Checkbutton(left, text=label, variable=var)
            cb.pack(anchor="w", padx=6, pady=4, fill="x")
            var.trace_add("write", lambda *_: self.refresh_mask_display())
            return var

        _compact_check("mask_negative",  "Mask negative pixels")
        _compact_check("mask_zero",      "Mask zero pixels")
        _compact_check("mask_nonfinite", "Mask NaN / inf pixels")
        _compact_check("mask_log_scale", "Log scale", default=False)
        
        # --- 2. Compact Saturation Threshold ---
        sat_frame = ttk.Frame(left)
        sat_frame.pack(fill="x", padx=6, pady=6)
        ttk.Label(sat_frame, text="Sat. threshold:").pack(side="left")
        
        sat_var = tk.StringVar(value=str(self.config.get("saturated_threshold", "")))
        self.vars["saturated_threshold"] = sat_var
        sat_entry = ttk.Entry(sat_frame, textvariable=sat_var, width=12)
        sat_entry.pack(side="right", fill="x", expand=True, padx=(6, 0))
        self.entry_widgets["saturated_threshold"] = sat_entry

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8, padx=4)
        
        ttk.Button(left, text="Accept Final Mask", command=self.accept_final_mask).pack(fill="x", padx=6, pady=4)
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10, padx=4)
        # --- 3. Sidebar Buttons (Unified padding and expansion) ---
        btn_configs = [
            ("Load image",         self.load_image_for_mask),
            ("Load mask",          self.load_mask_file),
            ("Add polygon mode",   lambda: self.set_mask_mode("add")),
            ("Erase polygon mode", lambda: self.set_mask_mode("erase")),
            ("Pan mode",           lambda: self.set_mask_mode("pan")),
            ("Zoom box mode",      lambda: self.set_mask_mode("zoom")),
            ("Undo polygon",       self.undo_mask_polygon),
            ("Clear manual mask",  self.clear_manual_mask),
            ("Save current mask",  self.save_current_mask),
            # FIX B4: load accepted mask back, and intensity histogram for threshold choice.
            ("Load accepted mask", self._load_accepted_mask_into_editor),
            ("Intensity histogram", self._show_intensity_histogram),
        ]

        for text, cmd in btn_configs:
            ttk.Button(left, text=text, command=cmd).pack(fill="x", padx=6, pady=3)

        # --- 4. Status Label + FIX B4: mask stats ---
        self.mask_mode = "None"
        self.mask_points: List[Tuple[float, float]] = []
        self.mask_status = ttk.Label(left, text="Mode: None", foreground=MUTED, font=("TkDefaultFont", 9, "italic"))
        self.mask_status.pack(fill="x", padx=6, pady=8)
        # FIX B4a: masked-pixel stats label — updated in refresh_mask_display.
        self.mask_stats_label = ttk.Label(left, text="", foreground=MUTED, wraplength=170)
        self.mask_stats_label.pack(fill="x", padx=6, pady=2)
        
        # --- 5. Main Canvas Area ---
        self.mask_view_frame = right
        self.mask_placeholder = ttk.Label(right, text="Load an image to edit/accept a mask. Use toolbar pan/zoom, polygon add/erase modes, then Accept final mask.")
        self.mask_placeholder.pack(expand=True)

    # ------------------------------------------------------------------
    # Tab 4 — Dioptas
    # ------------------------------------------------------------------

    def _tab_dioptas(self, parent):
        frame = self._scrollable(parent)
        self.field(frame, "dioptas_image_file", "Image for Dioptas", "image")
        self.field(frame, "dioptas_poni_file",  "PONI for Dioptas",  "poni")
        self.field(frame, "dioptas_mask_file",  "Optional mask for Dioptas", "mask")
        r = len(frame.grid_slaves())
        self.ttk.Button(frame, text="Use current input files",         command=self.sync_dioptas_fields).grid(row=r,   column=0, padx=4, pady=8, sticky="w")
        self.ttk.Button(frame, text="Launch Dioptas with selected files", command=self.open_dioptas).grid(row=r,   column=1, padx=4, pady=8, sticky="w")
        self.ttk.Label(frame, text="Mask preference: .npy > .tif > .npz.  Dioptas reads .npy (flipud) natively.", foreground=MUTED, wraplength=600).grid(
            row=r+1, column=0, columnspan=4, sticky="w", padx=4, pady=2)

    # ------------------------------------------------------------------
    # Tab 3 — Generate
    # ------------------------------------------------------------------

    def _tab_generate(self, parent):
        frame = self._scrollable(parent)
        # Factory fallbacks only — geometry-derived values replace any field
        # still holding one of these once image + PONI are selected.
        self._factory_generate_defaults = {
            "npt_1d": "1500", "npt_radial": "500", "npt_azimuthal": "360",
        }
        defaults = {
            **self._factory_generate_defaults,
            "unit": "2th_deg", "method": "csr", "coverage_threshold_pct": "10",
            "fast_qa": False,
        }
        for k, v in defaults.items():
            self.config.setdefault(k, v)
        self.field(frame, "npt_1d",      "1D bins",            row=0)
        self.field(frame, "npt_radial",  "Cake radial bins",   row=1)
        self.field(frame, "npt_azimuthal","Cake azimuth bins", row=2)
        self.combobox(frame, "unit",   "Integration unit",
                      ["2th_deg", "2th_rad", "q_A^-1", "q_nm^-1"], row=3)
        self.combobox(frame, "method", "pyFAI 1D method",
                      ["csr", "lut", "bbox", "numpy"], row=4)
        self.field(frame, "radial_min",             "2θ min (deg, 2θ only)", row=5)
        self.field(frame, "radial_max",             "2θ max (deg, 2θ only)", row=6)
        self.field(frame, "coverage_threshold_pct", "Coverage threshold %", row=7)
        # Main: derive bin counts from detector geometry.
        self.ttk.Button(frame, text="Auto-set bins from image/PONI",
                        command=lambda: self._suggest_generation_bins(force=True)).grid(row=8, column=0, padx=4, pady=6, sticky="w")
        self.ttk.Label(frame, text="Derives bin counts from the detector geometry (about 1 bin per pixel of radial extent). Won't overwrite values you've already edited.",
                       foreground=MUTED, wraplength=700).grid(row=8, column=1, columnspan=3, sticky="w", padx=4)
        self.entry_widgets.get("radial_min") and self.entry_widgets["radial_min"].configure(state="normal")
        self.entry_widgets.get("radial_max") and self.entry_widgets["radial_max"].configure(state="normal")
        self.generation_label = self.ttk.Label(frame, text="No generations yet")
        self.generation_label.grid(row=50, column=0, columnspan=4, sticky="w", padx=4, pady=8)
        # FIX A2: store button ref + indeterminate progress bar (busy state during a run).
        self.generate_btn = self.ttk.Button(frame, text="Generate QA run", command=self.generate_qa)
        self.generate_btn.grid(row=51, column=0, padx=4, pady=8, sticky="w")
        self.generate_progress = self.ttk.Progressbar(frame, mode="indeterminate", length=180)
        self.generate_progress.grid(row=51, column=1, padx=4, pady=8, sticky="w")
        self.generate_progress.grid_remove()  # hidden initially
        self.checkbox(frame, "fast_qa", "Fast QA (skip cake integration)")
        self.ttk.Label(frame, text="Processing runs in the background so the application remains responsive.",
                       foreground=MUTED).grid(row=52, column=0, columnspan=4, sticky="w", padx=4)
        self.ttk.Label(frame, text="2D cake uses no-split CSR (low memory). The pyFAI method above applies to 1D integration only.",
                       foreground=MUTED).grid(row=53, column=0, columnspan=4, sticky="w", padx=4)

    # ------------------------------------------------------------------
    # Tab 6 — QA Viewer
    # ------------------------------------------------------------------

    def _tab_viewer(self, parent):
        tk, ttk = self.tk, self.ttk
        # ---- top control bar ----
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(0, 4))
        # FIX C8: replace emoji with plain ASCII text
        ttk.Button(top, text="← Previous gen", command=self.show_previous_generation).pack(side="left", padx=3)
        ttk.Button(top, text="Next gen →",     command=self.show_next_generation).pack(side="left", padx=3)
        ttk.Button(top, text="Zoom +",  command=lambda: self._viewer_zoom_by(1.2)).pack(side="left", padx=2)
        ttk.Button(top, text="Zoom -",  command=lambda: self._viewer_zoom_by(1/1.2)).pack(side="left", padx=2)
        ttk.Button(top, text="Fit",     command=self._viewer_fit).pack(side="left", padx=2)
        self.viewer_label = ttk.Label(top, text="No generation loaded. Use arrow keys after clicking this tab.")
        self.viewer_label.pack(side="left", padx=12)
        # ---- main area: left sidebar | canvas | right sidebar ----
        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        # Left sidebar: panel visibility
        left_sb = ttk.Frame(body, width=180)
        left_sb.pack(side="left", fill="y", padx=(0, 4))
        left_sb.pack_propagate(False)
        # FIX C8: portable font
        ttk.Label(left_sb, text="Show panels:", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=4, pady=(4, 2))
        self._panel_vars: Dict[str, Any] = {}
        _panel_names = [
            ("raw_detector_png",       "Raw detector"),
            ("masked_detector_png",    "Masked detector"),
            ("mask_only_png",          "Mask only"),
            ("intensity_difference_png","Intensity + diff"),
            ("intensity_normalized_png","Intensity (norm.)"),
            ("cake_png",               "Cake"),
            ("coverage_png",           "Coverage diag."),
        ]
        for key, label in _panel_names:
            v = tk.BooleanVar(value=True)
            self._panel_vars[key] = v
            ttk.Checkbutton(left_sb, text=label, variable=v,
                            command=self.render_current_generation).pack(anchor="w", padx=6, pady=1)
        # Right sidebar: replot / axis controls
        right_sb = ttk.Frame(body, width=210)
        right_sb.pack(side="right", fill="y", padx=(4, 0))
        right_sb.pack_propagate(False)
        # FIX C8: portable font
        ttk.Label(right_sb, text="Replot controls:", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=4, pady=(4, 2))
        ctrl_frame = ttk.Frame(right_sb)
        ctrl_frame.pack(fill="x", padx=4)
        def _lbl_ent(label, key_var):
            r = len(ctrl_frame.grid_slaves())
            ttk.Label(ctrl_frame, text=label).grid(row=r, column=0, sticky="w", pady=2)
            v = tk.StringVar()
            ttk.Entry(ctrl_frame, textvariable=v, width=10).grid(row=r, column=1, sticky="ew", padx=2)
            ctrl_frame.columnconfigure(1, weight=1)
            return v
        self._replot_xmin = _lbl_ent("x min", "xmin")
        self._replot_xmax = _lbl_ent("x max", "xmax")
        self._replot_ymin = _lbl_ent("y min", "ymin")
        self._replot_ymax = _lbl_ent("y max", "ymax")
        self._replot_xscale = tk.StringVar(value="linear")
        self._replot_yscale = tk.StringVar(value="linear")
        ttk.Label(ctrl_frame, text="x scale").grid(row=len(ctrl_frame.grid_slaves()), column=0, sticky="w")
        sf = ttk.Frame(ctrl_frame)
        sf.grid(row=len(ctrl_frame.grid_slaves()), column=1, sticky="ew")
        ttk.Radiobutton(sf, text="lin", variable=self._replot_xscale, value="linear").pack(side="left")
        ttk.Radiobutton(sf, text="log", variable=self._replot_xscale, value="log").pack(side="left")
        ttk.Label(ctrl_frame, text="y scale").grid(row=len(ctrl_frame.grid_slaves()), column=0, sticky="w")
        sf2 = ttk.Frame(ctrl_frame)
        sf2.grid(row=len(ctrl_frame.grid_slaves()), column=1, sticky="ew")
        ttk.Radiobutton(sf2, text="lin", variable=self._replot_yscale, value="linear").pack(side="left")
        ttk.Radiobutton(sf2, text="log", variable=self._replot_yscale, value="log").pack(side="left")
        ttk.Button(right_sb, text="Replot from CSV", command=self._replot_from_csv).pack(fill="x", padx=4, pady=4)
        ttk.Button(right_sb, text="Reset limits",    command=self._replot_reset).pack(fill="x", padx=4, pady=2)
        # Centre: scrollable image canvas
        centre = ttk.Frame(body)
        centre.pack(side="left", fill="both", expand=True)
        self.viewer_canvas = tk.Canvas(centre, bg=BG, highlightthickness=0)
        self.viewer_vsb    = ttk.Scrollbar(centre, orient="vertical",   command=self.viewer_canvas.yview)
        self.viewer_hsb    = ttk.Scrollbar(centre, orient="horizontal",  command=self.viewer_canvas.xview)
        self.viewer_canvas.configure(yscrollcommand=self.viewer_vsb.set, xscrollcommand=self.viewer_hsb.set)
        self.viewer_vsb.pack(side="right",  fill="y")
        self.viewer_hsb.pack(side="bottom", fill="x")
        self.viewer_canvas.pack(fill="both", expand=True)
        # FIX A17b: debounce viewer resize — avoid LANCZOS re-render on every pixel.
        self.viewer_canvas.bind("<Configure>", self._on_viewer_configure)
        self.viewer_canvas.bind("<MouseWheel>",    self._viewer_mousewheel)
        self.viewer_canvas.bind("<Button-4>",      self._viewer_mousewheel)
        self.viewer_canvas.bind("<Button-5>",      self._viewer_mousewheel)
        # FIX J: bind arrow keys on canvas widget only (not globally) to avoid
        # firing while typing in Entry fields.  Canvas must take focus first.
        self.viewer_canvas.bind("<Button-1>",      lambda e: self.viewer_canvas.focus_set())
        self.viewer_canvas.bind("<Left>",          lambda e: self.show_previous_generation())
        self.viewer_canvas.bind("<Right>",         lambda e: self.show_next_generation())

    # ------------------------------------------------------------------
    # Console log window (opened via top-bar "Open Console Logs" button)
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
        self._log_window.title("SeriesXRD — Calibration log")
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
    # Tab 8 — Final Save
    # ------------------------------------------------------------------

    def _tab_final(self, parent):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(0, 6))
        if not self.config.get("final_output_root"):
            self.config["final_output_root"] = self.config.get("accepted_output_root", "")
        self.field(top, "final_output_root",  "Final accepted output root", "dir", row=0)
        self.field(top, "final_folder_name",  "Final folder name",          None,  row=1)
        
        # Provenance panel
        prov_lf = ttk.Frame(parent)
        prov_lf.pack(fill="x", pady=(0, 4))
        ttk.Label(prov_lf, text="Source generation:", font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=4)
        self.final_prov_label = ttk.Label(prov_lf, text="none", foreground=ACCENT)
        self.final_prov_label.pack(side="left", padx=4)
        ttk.Button(prov_lf, text="Use current viewer generation", command=self._final_use_current_gen).pack(side="left", padx=4)
        ttk.Button(prov_lf, text="Use latest generation",         command=self._final_use_latest_gen).pack(side="left", padx=4)
        ttk.Button(prov_lf, text="Refresh checklist",             command=self._refresh_final_save_list).pack(side="left", padx=4)
        # Checklist
        cl_container = ttk.Frame(parent)
        cl_container.pack(fill="both", expand=True)
        self.final_checklist_frame = self._scrollable(cl_container)
        # Status + buttons
        bot = ttk.Frame(parent)
        bot.pack(fill="x", pady=(4, 0))
        self.final_status = ttk.Label(bot, text="No accepted save yet.", wraplength=900)
        self.final_status.pack(anchor="w", padx=4, pady=4)
        ttk.Button(bot, text="Save selected items",     command=self.save_accepted).pack(side="left", padx=4, pady=4)
        ttk.Button(bot, text="Open accepted output root", command=self.open_final_root).pack(side="left", padx=4, pady=4)
        self._final_generation_idx = -1

    def _final_use_current_gen(self):
        self._final_generation_idx = self.current_generation_idx
        self._refresh_final_save_list()

    def _final_use_latest_gen(self):
        self._final_generation_idx = len(self.generations) - 1
        self._refresh_final_save_list()

    def _refresh_final_save_list(self):
        frame = self.final_checklist_frame
        for w in frame.winfo_children():
            w.destroy()
        self._final_item_vars = {}
        if self._final_generation_idx < 0 and self.generations:
            self._final_generation_idx = len(self.generations) - 1
        if self._final_generation_idx < 0 or self._final_generation_idx >= len(self.generations):
            self.ttk.Label(frame, text="No generation available. Generate a QA run first.").grid(row=0, column=0, padx=4, pady=4)
            self.final_prov_label.configure(text="none")
            return
        md   = self.generations[self._final_generation_idx]
        gen  = md.get("generation", "???")
        self.final_prov_label.configure(text=f"{gen}  —  {md.get('created_at','')}")
        paths = {k: Path(v) for k, v in md.get("paths", {}).items() if isinstance(v, str)}
        static_items = [
            ("original_calibration_image", "Source calibration TIFF",    Path(md.get("image_file", ""))),
            ("source_poni",                "Source PONI",                 Path(md.get("poni_file",  ""))),
            ("original_input_mask",        "Original input mask",         Path(self.config.get("input_mask_file",""))),
            ("session_config",             "Session config JSON",         self.config_path),
        ]
        figure_items = [
            ("compilation_png",          "Compilation PNG"),
            ("compilation_pdf",          "Compilation PDF"),
            ("raw_detector_png",         "Raw detector PNG"),
            ("masked_detector_png",      "Masked detector PNG"),
            ("mask_only_png",            "Mask-only PNG"),
            ("intensity_difference_png", "Intensity+diff PNG"),
            ("intensity_normalized_png", "Intensity (norm.) PNG"),
            ("cake_png",                 "Cake PNG"),
            ("coverage_png",             "Coverage PNG"),
        ]
        data_items = [
            ("mask_npz",       "Mask NPZ"),
            ("intensity_csv",  "Intensity CSV"),
            ("difference_csv", "Difference CSV"),
            ("coverage_csv",   "Coverage CSV"),
            ("cake_npz",       "Cake NPZ"),
            ("master_csv",     "Master CSV"),
            ("report_txt",     "Report TXT"),
            ("metadata_json",  "Metadata JSON"),
        ]
        row = 0
        self.ttk.Label(frame, text="Static sources:", font=("TkDefaultFont", 9, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 2))
        row += 1
        for key, label, p in static_items:
            exists = p.exists() if p and str(p) else False
            v = self.tk.BooleanVar(value=exists)
            self._final_item_vars[key] = v
            self.ttk.Checkbutton(frame, text=label, variable=v).grid(row=row, column=0, sticky="w", padx=8, pady=1)
            self.ttk.Label(frame, text=str(p) if p else "—", foreground=MUTED, wraplength=700).grid(row=row, column=1, sticky="w", padx=4)
            self.ttk.Label(frame, text="✓" if exists else "missing", foreground=ACCENT2 if exists else WARN).grid(row=row, column=2, padx=4)
            row += 1
        self.ttk.Label(frame, text="Figures:", font=("TkDefaultFont", 9, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 2))
        row += 1
        for key, label in figure_items:
            p      = paths.get(key)
            exists = p.exists() if p else False
            v      = self.tk.BooleanVar(value=exists)
            self._final_item_vars[key] = v
            self.ttk.Checkbutton(frame, text=label, variable=v).grid(row=row, column=0, sticky="w", padx=8, pady=1)
            self.ttk.Label(frame, text=str(p) if p else "—", foreground=MUTED, wraplength=700).grid(row=row, column=1, sticky="w", padx=4)
            self.ttk.Label(frame, text="✓" if exists else "missing", foreground=ACCENT2 if exists else WARN).grid(row=row, column=2, padx=4)
            row += 1
        self.ttk.Label(frame, text="Data:", font=("TkDefaultFont", 9, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 2))
        row += 1
        for key, label in data_items:
            p      = paths.get(key)
            exists = p.exists() if p else False
            v      = self.tk.BooleanVar(value=exists)
            self._final_item_vars[key] = v
            self.ttk.Checkbutton(frame, text=label, variable=v).grid(row=row, column=0, sticky="w", padx=8, pady=1)
            self.ttk.Label(frame, text=str(p) if p else "—", foreground=MUTED, wraplength=700).grid(row=row, column=1, sticky="w", padx=4)
            self.ttk.Label(frame, text="✓" if exists else "missing", foreground=ACCENT2 if exists else WARN).grid(row=row, column=2, padx=4)
            row += 1

    # ------------------------------------------------------------------
    # Business logic — dependency checks
    # ------------------------------------------------------------------

    def check_deps(self):
        self.pull_vars()
        dep = check_dependencies(self.config.get("python_exe", sys.executable),
                                 self.config.get("conda_exe", ""),
                                 self.config.get("conda_env_name", ""))
        self.log("Dependency check: " + json.dumps(dep.to_dict()))
        if dep.missing_required:
            self.messagebox.showwarning("Missing packages", "Missing required packages:\n" + "\n".join(dep.missing_required))
        else:
            self.messagebox.showinfo("Dependencies", "Required packages found.")

    def install_missing(self):
        self.pull_vars()
        dep = check_dependencies(self.config.get("python_exe", sys.executable),
                                 self.config.get("conda_exe", ""),
                                 self.config.get("conda_env_name", ""))
        if not dep.missing_required:
            self.log("No required packages missing")
            return
        cmd = package_install_command(dep.missing_required, self.config.get("python_exe", sys.executable),
                                      self.config.get("conda_exe", ""), self.config.get("conda_env_name", ""))
        if not self.messagebox.askyesno("Install missing packages", "Run this command?\n\n" + " ".join(cmd)):
            self.log("User declined package installation", "WARN")
            return
        def worker():
            rc = run_install_command(cmd)
            self.log(f"Install command finished with return code {rc}")
        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Mask tab logic
    # ------------------------------------------------------------------

    def load_image_for_mask(self):
        self.pull_vars()
        image_file = self.config.get("image_file", "")
        if not image_file or not Path(image_file).exists():
            self.messagebox.showerror("Missing image", "Select a calibration image in Tab 1 first.")
            return
        try:
            flip = bool(self.config.get("dioptas_image_flip", True))
            self.image_cache = read_detector_image(image_file, flip_up_down=flip)
            
            self.log(f"Loaded image for masking: {image_file} shape={getattr(self.image_cache, 'shape', None)}")
            self._draw_mask_editor()
        except Exception as e:
            self.log("Failed to load image for masking: " + repr(e), "ERROR")
            self.messagebox.showerror("Image load failed", str(e))

    def _draw_mask_editor(self):
        for child in self.mask_view_frame.winfo_children():
            child.destroy()
        try:
            import matplotlib
            matplotlib.use("TkAgg", force=True)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            self.mpl = (Figure, FigureCanvasTkAgg, NavigationToolbar2Tk)
        except Exception as e:
            self.ttk.Label(self.mask_view_frame, text="Matplotlib Tk canvas unavailable: " + str(e)).pack(fill="both", expand=True)
            return
        Figure, FigureCanvasTkAgg, NavigationToolbar2Tk = self.mpl
        fig = Figure(figsize=(7, 6), dpi=100)
        ax  = fig.add_subplot(111)
        self.mask_fig, self.mask_ax = fig, ax
        self.mask_canvas = FigureCanvasTkAgg(fig, master=self.mask_view_frame)
        self.mask_canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.mask_canvas, self.mask_view_frame)
        toolbar.update()
        self.mask_toolbar = toolbar
        self.mask_canvas.mpl_connect("button_press_event", self._mask_click)
        self.refresh_mask_display()

    def refresh_mask_display(self):
        self.pull_vars()
        if self.image_cache is None or not hasattr(self, "mask_ax"):
            return
        ax = self.mask_ax
        ax.clear()
        arr  = self.image_cache
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            vmin = float(np.percentile(finite, 1))
            vmax = float(np.percentile(finite, 99.5))
        else:
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmin, vmax = None, None
        if "mask_log_scale" in self.vars:
            use_log = self.vars["mask_log_scale"].get()
        else:
            use_log = bool(self.config.get("mask_log_scale", False))
        if use_log and vmin is not None and vmax is not None:
            from matplotlib.colors import LogNorm  # type: ignore
            safe_vmin = max(float(vmin), 1.0)
            safe_vmax = max(float(vmax), safe_vmin + 1.0)
            norm = LogNorm(vmin=safe_vmin, vmax=safe_vmax)
            ax.imshow(arr, cmap="gray", origin="upper", norm=norm)
        else:
            ax.imshow(arr, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        mask = self.current_combined_mask(preview=True)
        if mask is not None:
            overlay = np.zeros((*mask.shape, 4), dtype=float)
            overlay[..., 0] = 1.0
            overlay[..., 3] = mask.astype(float) * 0.35
            ax.imshow(overlay, origin="upper")
        if self.mask_points:
            xs, ys = zip(*self.mask_points)
            ax.plot(xs, ys, "o-", lw=1.2)
        ax.set_title(f"Mask editor — mode: {self.mask_mode}. Double-click to finish polygon.")
        self.mask_canvas.draw_idle()
        # FIX B4a: update mask stats label.
        if hasattr(self, "mask_stats_label") and mask is not None:
            n_masked = int(mask.sum())
            pct = 100.0 * mask.mean()
            self.mask_stats_label.configure(
                text=f"Masked: {n_masked:,} px  ({pct:.2f}%)"
            )

    def set_mask_mode(self, mode: str):
        self.mask_mode   = mode
        self.mask_points = []
        self.mask_status.configure(text=f"Mode: {mode}")
        self.log(f"Mask editor mode set to {mode}")

        # Synchronize with Matplotlib's native toolbar
        if hasattr(self, "mask_toolbar") and self.mask_toolbar:
            current_mpl_mode = getattr(self.mask_toolbar.mode, "value", self.mask_toolbar.mode)
            
            if mode in {"add", "erase"}:
                # Turn OFF any active pan/zoom tools
                if current_mpl_mode == "pan/zoom":
                    self.mask_toolbar.pan() 
                elif current_mpl_mode == "zoom rect":
                    self.mask_toolbar.zoom() 
            
            elif mode == "pan":
                # Turn ON pan mode
                if current_mpl_mode != "pan/zoom":
                    self.mask_toolbar.pan() 
                    
            elif mode == "zoom":
                # Turn ON zoom rectangle mode
                if current_mpl_mode != "zoom rect":
                    self.mask_toolbar.zoom()
                    
        self.refresh_mask_display()

    def _mask_click(self, event):
        if self.mask_mode not in {"add", "erase"}:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.mask_points.append((float(event.xdata), float(event.ydata)))
        # FIX B: a polygon point was added → mask is now different from last accepted.
        self._mask_dirty = True
        if getattr(event, "dblclick", False) and len(self.mask_points) >= 3:
            self.mask_polygons.append((self.mask_mode, list(self.mask_points)))
            self.log(f"Added {self.mask_mode} polygon with {len(self.mask_points)} points")
            self.mask_points = []
            # FIX F2: clear polygon cache when a new polygon is completed.
            self._polygon_mask_cache.clear()
        self.refresh_mask_display()

    def current_combined_mask(self, preview=False):
        if self.image_cache is None:
            return None
        self.pull_vars()
        base = automatic_mask(
            self.image_cache,
            mask_negative=bool(self.config.get("mask_negative",  True)),
            mask_zero=bool(self.config.get("mask_zero",          True)),
            mask_nonfinite=bool(self.config.get("mask_nonfinite", True)),
            saturated_threshold=self.config.get("saturated_threshold", ""),
        )
        if self.manual_mask is not None and self.manual_mask.shape == base.shape:
            base |= self.manual_mask
        for mode, pts in self.mask_polygons:
            # FIX F2: cache each polygon's rasterized mask keyed by (points, shape).
            cache_key = (tuple(map(tuple, pts)), base.shape)
            pm = self._polygon_mask_cache.get(cache_key)
            if pm is None:
                pm = polygon_to_mask(base.shape, pts)
                self._polygon_mask_cache[cache_key] = pm
            if mode == "add":
                base |= pm
            elif mode == "erase":
                base &= ~pm
        return base

    def load_mask_file(self):
        p = self.filedialog.askopenfilename(title="Load mask", filetypes=[("NumPy mask", "*.npz *.npy"), ("All files", "*.*")])
        if not p:
            return
        try:
            if p.lower().endswith(".npy"):
                loaded = np.load(p).astype(bool)
            else:
                loaded = load_mask_npz(p)
            # FIX C: shape check against loaded image.
            if self.image_cache is not None and loaded.shape != self.image_cache.shape:
                self.messagebox.showerror(
                    "Shape mismatch",
                    f"Mask shape {loaded.shape} does not match image shape {self.image_cache.shape}."
                )
                return
            self.manual_mask = loaded
            self.log(f"Loaded manual mask: {p}  shape={loaded.shape}")
            self.config["input_mask_file"] = p
            if "input_mask_file" in self.vars:
                self.vars["input_mask_file"].set(p)
            # FIX B: mark mask as dirty (un-accepted edits present).
            self._mask_dirty = True
            self.refresh_mask_display()
        except Exception as e:
            self.log("Failed to load mask: " + repr(e), "ERROR")
            self.messagebox.showerror("Mask load failed", str(e))

    def undo_mask_polygon(self):
        if self.mask_polygons:
            self.mask_polygons.pop()
            self.log("Undid last mask polygon")
        # FIX B: mark dirty; FIX F2: clear polygon cache.
        self._mask_dirty = True
        self._polygon_mask_cache.clear()
        self.refresh_mask_display()

    def clear_manual_mask(self):
        self.manual_mask  = None
        self.mask_polygons = []
        self.active_mask   = None
        self.log("Cleared manual mask and polygons")
        # FIX B: mark dirty; FIX F2: clear polygon cache.
        self._mask_dirty = True
        self._polygon_mask_cache.clear()
        self.refresh_mask_display()

    @staticmethod
    def _sibling(base: Path, ext: str) -> Path:
        """Return base.parent / (base.name + ext), stripping one known trailing
        extension from base.name first so that user paths with dots don't collide.
        E.g. _sibling(Path("/foo/bar.npz"), ".png") → /foo/bar.png
        """
        _known = {".npz", ".npy", ".png", ".tif", ".tiff"}
        name = base.name
        for k in _known:
            if name.lower().endswith(k):
                name = name[: -len(k)]
                break
        return base.parent / (name + ext)

    def _save_mask_all_formats(self, base_path: Path, mask: np.ndarray, meta: dict) -> dict:
        """Save mask in .npz, .npy (plain), _dioptas.npy (flipud), _dioptas.tif, .png.
        Returns dict of paths."""
        from PIL import Image  # type: ignore
        saved = {}
        npz_path = self._sibling(base_path, ".npz")
        save_mask_npz(npz_path, mask, metadata=meta)
        saved["npz"] = str(npz_path)
        self.log(f"Saved mask NPZ: {npz_path}")
        # FIX C: plain .npy (unflipped) so reloading the GUI's own export is lossless.
        npy_path = self._sibling(base_path, ".npy")
        np.save(str(npy_path), mask.astype(bool))
        saved["npy"] = str(npy_path)
        self.log(f"Saved mask NPY: {npy_path}")
        # FIX C: separate flipud copy for Dioptas (bottom-left origin).
        stem = self._sibling(base_path, "").name  # strip extension to get bare stem
        dioptas_npy_path = base_path.parent / (stem + "_dioptas.npy")
        np.save(str(dioptas_npy_path), np.flipud(mask.astype(bool)))
        saved["npy_dioptas"] = str(dioptas_npy_path)
        self.log(f"Saved mask NPY (dioptas/flipud): {dioptas_npy_path}")
        # _dioptas.tif (uint8 TIFF, flipud, white=masked)
        tif_path = base_path.parent / (stem + "_dioptas.tif")
        Image.fromarray(np.flipud(mask.astype(np.uint8) * 255), mode="L").save(str(tif_path))
        saved["tif"] = str(tif_path)
        self.log(f"Saved mask TIFF (flipud): {tif_path}")
        # .png preview (no flip — for human inspection in pyFAI convention)
        png_path = self._sibling(base_path, ".png")
        save_mask_preview_png(png_path, mask)
        saved["png"] = str(png_path)
        self.log(f"Saved mask preview PNG: {png_path}")
        return saved

    def save_current_mask(self):
        if self.image_cache is None:
            self.messagebox.showerror("No image", "Load an image first.")
            return
        mask = self.current_combined_mask()
        p    = self.filedialog.asksaveasfilename(title="Save mask", defaultextension=".npz",
                                                 filetypes=[("NumPy mask", "*.npz")])
        if not p:
            return
        # FIX C: use _sibling to strip one known extension, so we don't double-suffix.
        base = Path(p)
        self._save_mask_all_formats(base, mask, {"created_at": now_iso(), "source": "mask tab"})
        self.messagebox.showinfo("Mask saved", f"Saved formats to:\n{self._sibling(base, '.*')}")

    def accept_final_mask(self):
        if self.image_cache is None:
            self.load_image_for_mask()
            if self.image_cache is None:
                return
        mask = self.current_combined_mask()
        self.active_mask = mask
        root = Path(self.config.get("metadata_root") or (output_base(self.config) / "metadata")) / "active_masks"
        ensure_dir(root)
        stem      = f"accepted_active_mask_{safe_stem(self.config.get('session_name','calibration'))}_{now_timestamp()}"
        base_path = next_available_path(root / stem)  # no suffix yet
        saved = self._save_mask_all_formats(base_path, mask, {"accepted_at": now_iso()})
        self.config["active_mask_file"]     = saved["npz"]
        self.config["active_mask_npy"]      = saved["npy"]          # plain (unflipped)
        self.config["active_mask_npy_dioptas"] = saved["npy_dioptas"]  # flipud for Dioptas
        self.config["accepted_mask_tiff"]   = saved["tif"]
        self.config["active_mask_preview"]  = saved["png"]
        # FIX B: accepted — no un-accepted edits pending.
        self._mask_dirty = False
        self.save_config(silent=True)
        self.log(f"Accepted final mask: {saved['npz']}")
        # FIX C2: refresh status bar after mask acceptance.
        self._update_status_bar()
        self.messagebox.showinfo("Mask accepted",
                                 f"Active mask saved:\n{base_path}.*\n\n"
                                 f"NPZ: {saved['npz']}\nNPY: {saved['npy']}\n"
                                 f"NPY (Dioptas): {saved['npy_dioptas']}\nTIF: {saved['tif']}")

    # FIX B4b: Load accepted mask back into the editor.
    def _load_accepted_mask_into_editor(self):
        amf = self.config.get("active_mask_file", "")
        if not amf or not Path(amf).exists():
            self.messagebox.showerror("No accepted mask",
                                      "No accepted mask file found in config (active_mask_file).")
            return
        try:
            loaded = load_mask_npz(amf)
            if self.image_cache is not None and loaded.shape != self.image_cache.shape:
                self.messagebox.showerror(
                    "Shape mismatch",
                    f"Accepted mask shape {loaded.shape} != current image shape {self.image_cache.shape}."
                )
                return
            self.manual_mask = loaded
            self._mask_dirty = True
            self._polygon_mask_cache.clear()
            self.log(f"Loaded accepted mask into editor: {amf}")
            self.refresh_mask_display()
        except Exception as e:
            self.log(f"Failed to load accepted mask into editor: {e}", "ERROR")
            self.messagebox.showerror("Load failed", str(e))

    # FIX B4c: Intensity histogram popup.
    def _show_intensity_histogram(self):
        if self.image_cache is None:
            self.messagebox.showerror("No image", "Load a calibration image first.")
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg", force=True)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            self.messagebox.showerror("Matplotlib unavailable", str(e))
            return
        try:
            tk = self.tk
            top = tk.Toplevel(self.root)
            top.title("Pixel intensity histogram")
            top.configure(bg=BG)
            arr = self.image_cache
            finite_vals = arr[np.isfinite(arr)].ravel()
            if finite_vals.size == 0:
                self.messagebox.showerror("No finite pixels", "Image has no finite pixel values.")
                top.destroy()
                return
            fig = Figure(figsize=(7, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.hist(finite_vals, bins=512, log=True, color="#5599bb", edgecolor="none")
            ax.set_xlabel("Pixel value")
            ax.set_ylabel("Count (log scale)")
            ax.set_title("Pixel intensity distribution")
            # Vertical line at saturated_threshold if set.
            sat_str = str(self.config.get("saturated_threshold", "")).strip()
            try:
                sat_val = float(sat_str) if sat_str else None
            except ValueError:
                sat_val = None
            if sat_val is not None:
                ax.axvline(sat_val, color="red", lw=1.2, label=f"sat. threshold={sat_val}")
                ax.legend(fontsize=8)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            self.log(f"Histogram error: {e}", "ERROR")
            self.messagebox.showerror("Histogram error", str(e))

    # ------------------------------------------------------------------
    # Dioptas tab logic
    # ------------------------------------------------------------------

    def sync_dioptas_fields(self):
        for dst, src in [("dioptas_image_file", "image_file"), ("dioptas_poni_file", "poni_file")]:
            val = self.vars.get(src).get() if src in self.vars else self.config.get(src, "")
            if dst in self.vars and val:
                self.vars[dst].set(val)
                self.config[dst] = val
        # Prefer _dioptas.npy > .tif > plain .npy > .npz for Dioptas mask
        mask_candidates = [
            self.config.get("active_mask_npy_dioptas", ""),
            self.config.get("active_mask_npy",         ""),
            self.config.get("accepted_mask_tiff",       ""),
            self.config.get("active_mask_file",         ""),
        ]
        for candidate in mask_candidates:
            if candidate and Path(candidate).exists():
                if "dioptas_mask_file" in self.vars:
                    self.vars["dioptas_mask_file"].set(candidate)
                self.config["dioptas_mask_file"] = candidate
                self.log(f"Dioptas mask set to: {candidate}")
                break
        self.log("Synced Dioptas fields from current input/mask settings")

    def open_dioptas(self):
        self.pull_vars()
        img_f  = self.config.get("dioptas_image_file")  or self.config.get("image_file",      "")
        poni_f = self.config.get("dioptas_poni_file")   or self.config.get("poni_file",        "")
        mask_f = self.config.get("dioptas_mask_file")   or self.config.get("active_mask_npy", "")
        try:
            proc = launch_dioptas(
                dioptas_command=self.config.get("dioptas_command", ""),
                dioptas_python=self.config.get("dioptas_python",   ""),
                image_file=img_f, poni_file=poni_f, mask_file=mask_f,
            )
            self.log(f"Launched Dioptas PID {proc.pid}")
            # FIX C10b: show manual loading instructions after launch.
            instructions = dioptas_manual_instructions(img_f, poni_f, mask_f)
            self.messagebox.showinfo("Load files in Dioptas manually", instructions)
        except Exception as e:
            self.log("Dioptas launch failed: " + repr(e), "ERROR")
            self.messagebox.showerror("Dioptas launch failed", str(e))

    # ------------------------------------------------------------------
    # FIX C10a: Recent files helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recent_json_path() -> Path:
        return Path.home() / ".seriesxrd_recent.json"

    def _load_recent(self) -> dict:
        """Read ~/.seriesxrd_recent.json safely."""
        try:
            p = self._recent_json_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_recent(self, data: dict):
        """Write ~/.seriesxrd_recent.json safely."""
        try:
            p = self._recent_json_path()
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _add_recent(self, kind: str, path: str):
        """Add path to the recent list (poni or image) and refresh the menu."""
        try:
            data = self._load_recent()
            lst  = data.get(kind, [])
            # Deduplicate, keep most recent first, cap at 10.
            lst  = [x for x in lst if x != path]
            lst.insert(0, path)
            lst  = lst[:10]
            data[kind] = lst
            self._save_recent(data)
            self._populate_recent_menus()
        except Exception:
            pass

    def register_menus(self, menubar, file_menu, tools_menu=None, help_menu=None):
        """Populate shared menus with this stage's commands.

        Called both in standalone mode (the App builds its own menubar) and in
        embedded mode (the unified host passes its shared menus). ``tools_menu``
        may be None in standalone mode.
        """
        tk = self.tk
        file_menu.add_command(label="Global Environment Settings...", command=self.open_env_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Open session config...", command=self._open_session_config_dialog)
        file_menu.add_command(label="Save Session Config", accelerator="Ctrl+S", command=self.save_config)
        file_menu.add_separator()
        self._recent_poni_menu  = tk.Menu(file_menu, tearoff=0)
        self._recent_image_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent PONI",  menu=self._recent_poni_menu)
        file_menu.add_cascade(label="Recent image", menu=self._recent_image_menu)
        self._populate_recent_menus()
        if self._owns_root:
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self.on_close)
        if tools_menu is not None:
            tools_menu.add_command(label="Launch Dioptas", command=self.open_dioptas)
        if help_menu is not None:
            help_menu.add_command(label="About...", command=self.show_about)

    def _populate_recent_menus(self):
        """Rebuild Recent PONI and Recent image menus from stored history."""
        if not hasattr(self, "_recent_poni_menu") or not hasattr(self, "_recent_image_menu"):
            return
        try:
            data = self._load_recent()
            menus = [
                ("poni",  self._recent_poni_menu,  "poni_file"),
                ("image", self._recent_image_menu, "image_file"),
            ]
            for kind, menu, target_key in menus:
                menu.delete(0, "end")
                lst = data.get(kind, [])
                if not lst:
                    menu.add_command(label="(none)", state="disabled")
                for p in lst:
                    label = Path(p).name
                    menu.add_command(
                        label=label,
                        command=lambda _p=p, _k=target_key: self._set_var_from_recent(_k, _p)
                    )
        except Exception:
            pass

    def _set_var_from_recent(self, key: str, path: str):
        """Set a field var from a recently-used file selection."""
        try:
            if key in self.vars:
                self.vars[key].set(path)
            self.config[key] = path
            self.log(f"Recent: set {key} = {path}")
        except Exception:
            pass

    def _open_session_config_dialog(self):
        """FIX C10a: Open a different session config — show safe relaunch instructions."""
        try:
            path = self.filedialog.askopenfilename(
                title="Open session config",
                filetypes=[("JSON config", "*.json"), ("All files", "*.*")]
            )
            if not path:
                return
            relaunch_cmd = f"python -m seriesxrd.calib.gui --config \"{path}\""
            self.messagebox.showinfo(
                "Switching session config requires relaunch",
                f"To switch to the selected config, relaunch the GUI with:\n\n"
                f"{relaunch_cmd}\n\n"
                f"Selected config:\n{path}"
            )
        except Exception as e:
            self.log(f"Open config dialog error: {e}", "WARN")

    # ------------------------------------------------------------------
    # QA generation (worker subprocess in background thread)
    # ------------------------------------------------------------------

    def _check_generation_inputs(self) -> bool:
        """Validate image/PONI/mask compatibility before launching the worker.

        Uses a lightweight .poni text parse instead of importing pyFAI into the
        Tk process — a pyFAI DLL crash here would take down the whole GUI,
        which is exactly what the worker subprocess exists to prevent.
        """
        problems: List[str] = []
        image_file = str(self.config.get("image_file", "") or "").strip()
        poni_file  = str(self.config.get("poni_file",  "") or "").strip()
        if not image_file or not Path(image_file).is_file():
            problems.append(f"Calibration image not found: {image_file or '(not set)'}")
        if not poni_file or not Path(poni_file).is_file():
            problems.append(f"PONI file not found: {poni_file or '(not set)'}")
        image_shape = None
        if not problems:
            try:
                image_shape = tuple(read_detector_image(image_file).shape)
            except Exception as e:
                problems.append(f"Cannot read calibration image: {e!r}")
            info = read_poni_info(poni_file)
            if image_shape and info["shape"] and image_shape != info["shape"]:
                problems.append(
                    f"Image shape {image_shape} does not match PONI detector shape "
                    f"{info['shape']} ({info['detector'] or 'unknown detector'})."
                )
            if info["wavelength_m"] and not str(self.config.get("wavelength_m", "")).strip():
                self.config["wavelength_m"] = str(info["wavelength_m"])
                if "wavelength_m" in self.vars:
                    self.vars["wavelength_m"].set(str(info["wavelength_m"]))
                self.log(f"Auto-filled wavelength_m from PONI: {info['wavelength_m']}")
            elif not info["wavelength_m"] and self.config.get("calibrant"):
                self.log("PONI has no wavelength; calibrant reference lines may be skipped.", "WARN")
            active = str(self.config.get("active_mask_file", "") or "").strip()
            if active and Path(active).exists() and image_shape:
                try:
                    mask_shape = tuple(load_mask_npz(active).shape)
                    if mask_shape != image_shape:
                        problems.append(
                            f"Accepted mask shape {mask_shape} does not match image shape "
                            f"{image_shape}. Re-accept the mask in Tab 2."
                        )
                except Exception as e:
                    self.log(f"Could not read accepted mask {active}: {e!r}", "WARN")
        if problems:
            self.log("Generation input check failed: " + " | ".join(problems), "ERROR")
            self.messagebox.showerror("Cannot generate QA run", "\n\n".join(problems))
            return False
        return True

    def _suggest_generation_bins(self, force: bool = False):
        """Set npt_* from the detector geometry of the selected image + PONI.

        With force=False (the autocomplete hook) only fields that are blank or
        still holding a factory default are touched, so user-tuned values
        survive. The Tab 3 button passes force=True to overwrite everything.
        """
        image_file = str(self.vars["image_file"].get() if "image_file" in self.vars else self.config.get("image_file", "")).strip()
        poni_file  = str(self.vars["poni_file"].get()  if "poni_file"  in self.vars else self.config.get("poni_file",  "")).strip()
        if not (image_file and Path(image_file).is_file() and poni_file and Path(poni_file).is_file()):
            if force:
                self.messagebox.showinfo("Auto-set bins", "Select a calibration image and PONI in Tab 1 first.")
            return
        try:
            shape = tuple(read_detector_image(image_file).shape)
            suggested = suggest_integration_settings(shape, read_poni_info(poni_file))
        except Exception as e:
            self.log(f"Bin suggestion failed: {e!r}", "WARN")
            return
        self.log(f"Geometry: max radial extent = {suggested.get('r_max_px', '?')} px -> npt_1d {suggested['npt_1d']} (rounded up to 50, capped 4000)")
        factory = getattr(self, "_factory_generate_defaults", {})
        changed = []
        for key, value in suggested.items():
            current = str(self.vars[key].get() if key in self.vars else self.config.get(key, "")).strip()
            if not (force or not current or current == factory.get(key)):
                continue
            if current != str(value):
                if key in self.vars:
                    self.vars[key].set(str(value))
                self.config[key] = str(value)
                changed.append(f"{key}={value}")
        if changed:
            self.log(f"Integration bins from geometry ({shape[0]}x{shape[1]} px): " + ", ".join(changed))

    def generate_qa(self):
        # FIX A1: busy guard.
        if self._qa_running:
            self.log("A QA generation is already running", "WARN")
            return
        self.pull_vars()
        # Check first: this may also auto-fill wavelength_m from the PONI,
        # which the save below then persists for the worker to read.
        if not self._check_generation_inputs():
            return
        self.save_config(silent=True)

        # FIX A4: validate inputs before launching.
        errors = []
        for key in ("npt_1d", "npt_radial", "npt_azimuthal"):
            val = str(self.config.get(key, "")).strip()
            try:
                ival = int(val)
                if ival <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                errors.append(f"  • {key}: must be a positive integer (got {val!r})")
        cov_str = str(self.config.get("coverage_threshold_pct", "")).strip()
        try:
            cov = float(cov_str)
            if not (0.0 <= cov <= 100.0):
                raise ValueError
        except (ValueError, TypeError):
            errors.append(f"  • coverage_threshold_pct: must be a float in [0, 100] (got {cov_str!r})")
        rmin_str = str(self.config.get("radial_min", "")).strip()
        rmax_str = str(self.config.get("radial_max", "")).strip()
        rmin = rmax = None
        if rmin_str:
            try:
                rmin = float(rmin_str)
            except (ValueError, TypeError):
                errors.append(f"  • radial_min: must be a float if set (got {rmin_str!r})")
        if rmax_str:
            try:
                rmax = float(rmax_str)
            except (ValueError, TypeError):
                errors.append(f"  • radial_max: must be a float if set (got {rmax_str!r})")
        if rmin is not None and rmax is not None and rmin >= rmax:
            errors.append(f"  • radial_min ({rmin}) must be less than radial_max ({rmax})")
        if errors:
            self.messagebox.showerror("Validation errors",
                                      "Please fix the following before generating:\n\n" + "\n".join(errors))
            return

        # FIX B / Accept & Generate: handle un-accepted mask edits with a 3-way
        # choice instead of a bare warning.
        if self._mask_dirty:
            choice = self.messagebox.askyesnocancel(
                "Un-accepted mask edits",
                "You have mask edits that have not been accepted.\n\n"
                "Yes     = accept the current mask, then generate\n"
                "No      = generate with the last accepted mask (or automatic mask only)\n"
                "Cancel  = abort"
            )
            if choice is None:
                return
            if choice:  # Yes → accept the current mask first
                try:
                    self.accept_final_mask()
                    self.pull_vars()
                    self.save_config(silent=True)
                except Exception as e:
                    self.messagebox.showerror("Accept mask failed", repr(e))
                    return

        gen_idx     = self._next_generation_index()
        python_exe  = Path(self.config.get("python_exe", sys.executable))
        conda_exe   = self.config.get("conda_exe", "")
        backend_dir = self.config.get("backend_dir", str(Path(__file__).resolve().parents[1]))
        logs_root   = self.config.get("logs_root", "") or str(output_base(self.config) / "logs")
        ensure_dir(Path(logs_root))
        out_json    = str(next_available_path(Path(logs_root) / f"worker_gen{gen_idx:03d}_{now_timestamp()}.json"))
        worker_script = str(Path(backend_dir) / "calib" / "worker.py")
        # Always launch worker directly — DLL path is fixed inside the worker itself.
        # conda run via WSL bash was causing segfaults on Windows.
        cmd = [
            str(python_exe), worker_script,
            "--config", str(self.config_path),
            "--generation", str(gen_idx),
            "--output-json", out_json,
        ]
        # Belt-and-suspenders: also set Library/bin in the subprocess environment.
        import os as _os
        
        worker_env = dict(_os.environ)
        _prefix_dir = python_exe.parent
        for _sub in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin"):
            _d = str(_prefix_dir / _sub)
            if Path(_d).is_dir() and _d.lower() not in worker_env.get("PATH", "").lower():
                worker_env["PATH"] = _d + _os.pathsep + worker_env.get("PATH", "")

        # FIX A1: mark as running; FIX A2: disable button, start progress bar.
        self._qa_running = True
        if hasattr(self, "generate_btn"):
            self.generate_btn.configure(state="disabled")
        if hasattr(self, "generate_progress"):
            self.generate_progress.grid()
            self.generate_progress.start(10)

        self.log(f"Starting QA generation gen{gen_idx:03d}")
        self.log(f"Worker command: {' '.join(cmd)}")
        def _worker_thread():
            try:
                # FIX A3: stream stdout line-by-line into the log.
                proc = worker_popen(
                    cmd, cwd=backend_dir, env=worker_env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                self._qa_proc = proc  # retained so shutdown() can terminate it
                try:
                    for line in proc.stdout:
                        self.log(line.rstrip())
                except (ValueError, OSError):
                    pass
                rc = proc.wait()
                self.log(f"Worker gen{gen_idx:03d} returncode={rc}")
                self.root.after(0, lambda: self._qa_worker_done(rc, out_json, gen_idx))
            except Exception as e:
                err = repr(e)
                self.log(f"Worker gen{gen_idx:03d} launch error: {err}", "ERROR")
                self.root.after(0, lambda: self._qa_worker_error(err, gen_idx))
        threading.Thread(target=_worker_thread, daemon=True).start()

    def _qa_worker_done(self, returncode: int, out_json: str, gen_idx: int):
        # FIX A1/A2: clear busy flag, re-enable button, stop progress bar.
        self._qa_running = False
        self._qa_proc = None
        if hasattr(self, "generate_btn"):
            self.generate_btn.configure(state="normal")
        if hasattr(self, "generate_progress"):
            self.generate_progress.stop()
            self.generate_progress.grid_remove()
        # FIX C2: update status bar worker state.
        self._update_status_bar()
        self.log(f"Worker gen{gen_idx:03d} finished, returncode={returncode}")
        if returncode != 0:
            # FIX A5: removed stale "Tab 7" reference.
            self.messagebox.showerror(
                "QA generation failed",
                f"Worker return code {returncode}\n"
                "See the console log window (Open Console Logs, top of window) for details."
            )
            return
        if not Path(out_json).exists():
            self.messagebox.showerror("QA generation failed", f"Worker succeeded but output JSON missing:\n{out_json}")
            return
        try:
            md = read_json(out_json)
            self.generations.append(md)
            self.current_generation_idx = len(self.generations) - 1
            self._final_generation_idx  = self.current_generation_idx
            if hasattr(self, "generation_label"):
                self.generation_label.configure(text=f"Generated {md.get('generation')} — {md.get('paths', {}).get('compilation_png', '')}")
            self.log(f"Generated {md.get('generation')}: {md.get('paths', {}).get('compilation_png', '')}")
            self.render_current_generation()
            self.nb.select(self.tabs["4 Review"])
            self._refresh_final_save_list()
        except Exception as e:
            self.log(f"Failed to load worker output JSON: {repr(e)}", "ERROR")
            self.messagebox.showerror("QA output load failed", str(e))

    def _qa_worker_error(self, err_msg: str, gen_idx: int):
        # FIX A1/A2: clear busy flag, re-enable button, stop progress bar.
        self._qa_running = False
        self._qa_proc = None
        if hasattr(self, "generate_btn"):
            self.generate_btn.configure(state="normal")
        if hasattr(self, "generate_progress"):
            self.generate_progress.stop()
            self.generate_progress.grid_remove()
        # FIX C2: update status bar worker state.
        self._update_status_bar()
        self.log(f"Worker gen{gen_idx:03d} error: {err_msg}", "ERROR")
        self.messagebox.showerror("QA worker error", err_msg)

    # ------------------------------------------------------------------
    # Viewer rendering
    # ------------------------------------------------------------------

    def _on_viewer_configure(self, event):
        # FIX A17b: debounce <Configure> so we don't LANCZOS-resize every panel
        # PNG on every intermediate pixel during a drag resize.
        # Only re-render when width changed by more than 4 px (height-only changes
        # like scrollbar appearance/disappearance are ignored).
        new_w = event.width
        if abs(new_w - self._last_viewer_width) <= 4:
            return
        if self._resize_after_id is not None:
            try:
                self.root.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.root.after(150, self._do_debounced_render)

    def _do_debounced_render(self):
        """Called 150 ms after the last viewer Configure event."""
        self._resize_after_id = None
        if hasattr(self, "viewer_canvas"):
            self._last_viewer_width = self.viewer_canvas.winfo_width()
        self.render_current_generation()

    def render_current_generation(self):
        if not hasattr(self, "viewer_canvas"):
            return
        # FIX C2: refresh status bar on render.
        self._update_status_bar()
        self.viewer_canvas.delete("all")
        self._viewer_photos = []
        if self.current_generation_idx < 0 or self.current_generation_idx >= len(self.generations):
            return
        md   = self.generations[self.current_generation_idx]
        gen  = md.get("generation", "?")
        n    = len(self.generations)
        self.viewer_label.configure(text=f"{self.current_generation_idx+1}/{n} — {gen} — use ←/→ arrows")
        raw_paths = md.get("paths", {})
        visible_panels = [k for k, v in self._panel_vars.items() if v.get()]
        if not visible_panels:
            self.viewer_canvas.create_text(20, 20, anchor="nw", text="No panels selected.", fill=FG)
            return
        try:
            from PIL import Image, ImageTk  # type: ignore
        except ImportError:
            self.viewer_canvas.create_text(20, 20, anchor="nw", text="Pillow not available.", fill=FG)
            return
        cw = max(200, self.viewer_canvas.winfo_width()  - 24)
        panel_w = int(cw * self._viewer_zoom)
        y_offset = 10
        for key in visible_panels:
            p = raw_paths.get(key, "")
            if not p or not Path(p).exists():
                continue
            try:
                img = Image.open(p)
                ratio = panel_w / max(img.width, 1)
                ph    = int(img.height * ratio)
                img   = img.resize((panel_w, ph), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._viewer_photos.append(photo)
                self.viewer_canvas.create_image(panel_w // 2 + 12, y_offset + ph // 2,
                                                image=photo, anchor="center")
                y_offset += ph + 10
            except Exception as e:
                self.viewer_canvas.create_text(12, y_offset, anchor="nw",
                                               text=f"{key}: {repr(e)}", fill=WARN)
                y_offset += 20
        self.viewer_canvas.configure(scrollregion=(0, 0, panel_w + 24, y_offset + 10))

    def _viewer_zoom_by(self, factor: float):
        self._viewer_zoom = max(0.2, min(5.0, self._viewer_zoom * factor))
        self.render_current_generation()

    def _viewer_fit(self):
        self._viewer_zoom = 1.0
        self.render_current_generation()

    def _viewer_mousewheel(self, event):
        # FIX G: mirror _scrollable's platform handling.
        # On macOS delta is ±1..±3 (not multiples of 120), on Linux use Button-4/5.
        import sys as _sys
        if event.num == 4:
            self.viewer_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.viewer_canvas.yview_scroll(1, "units")
        elif event.delta:
            if _sys.platform == "darwin":
                self.viewer_canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                self.viewer_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def show_previous_generation(self):
        if not self.generations:
            return
        self.current_generation_idx = max(0, self.current_generation_idx - 1)
        self.render_current_generation()

    def show_next_generation(self):
        if not self.generations:
            return
        self.current_generation_idx = min(len(self.generations) - 1, self.current_generation_idx + 1)
        self.render_current_generation()

    # ------------------------------------------------------------------
    # Viewer replot
    # ------------------------------------------------------------------

    def _replot_from_csv(self):
        if self.current_generation_idx < 0 or self.current_generation_idx >= len(self.generations):
            self.messagebox.showerror("No generation", "Generate a QA run first.")
            return
        md   = self.generations[self.current_generation_idx]
        csv  = md.get("paths", {}).get("intensity_csv", "")
        if not csv or not Path(csv).exists():
            self.messagebox.showerror("Missing CSV", f"Intensity CSV not found:\n{csv}")
            return
        try:
            import csv as _csv
            rows = []
            with open(csv, newline="") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            if not rows:
                return
            keys   = list(rows[0].keys())
            x_key  = keys[0]
            y_key  = keys[1] if len(keys) > 1 else keys[0]
            xs = np.array([float(r[x_key]) for r in rows])
            ys = np.array([float(r[y_key]) if r[y_key] not in ("nan", "NaN", "") else np.nan for r in rows])
            xmin = self._replot_xmin.get().strip()
            xmax = self._replot_xmax.get().strip()
            ymin = self._replot_ymin.get().strip()
            ymax = self._replot_ymax.get().strip()
            xsc  = self._replot_xscale.get()
            ysc  = self._replot_yscale.get()
            # FIX H: use non-interactive Figure + FigureCanvasAgg — never touch
            # matplotlib.use() which would flip the global backend under TkAgg.
            from matplotlib.figure import Figure as _MplFig
            from matplotlib.backends.backend_agg import FigureCanvasAgg as _AggCanvas
            fig = _MplFig(figsize=(7.8, 4.5), dpi=150)
            _AggCanvas(fig)          # attach a non-interactive renderer
            ax = fig.add_subplot(111)
            ax.plot(xs, ys, lw=1.1)
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(f"Replot — {md.get('generation','?')}")
            if xmin and xmax:
                ax.set_xlim(float(xmin), float(xmax))
            if ymin and ymax:
                ax.set_ylim(float(ymin), float(ymax))
            ax.set_xscale(xsc)
            ax.set_yscale(ysc)
            fig.tight_layout()
            # FIX H: save into the generation's figures dir, not the CSV's parent dir.
            official_png = md.get("paths", {}).get("intensity_difference_png", "")
            if official_png:
                fig_dir = Path(official_png).parent
            else:
                fig_dir = Path(csv).parent
            out_png = next_available_path(fig_dir / f"replot_{now_timestamp()}.png")
            fig.savefig(str(out_png))
            self.log(f"Replot saved: {out_png}")
            # FIX H: store under a NEW key so the official figure is not clobbered.
            if "paths" not in md or md["paths"] is None:
                md["paths"] = {}
            md["paths"]["replot_png"] = str(out_png)
            self.render_current_generation()
            self.log(f"Replot saved: {out_png}")
            self.messagebox.showinfo("Replot saved", f"Saved replot to:\n{out_png}")
        except Exception as e:
            self.log("Replot failed: " + repr(e), "ERROR")
            self.messagebox.showerror("Replot failed", str(e))

    def _replot_reset(self):
        for v in (self._replot_xmin, self._replot_xmax, self._replot_ymin, self._replot_ymax):
            v.set("")
        self._replot_xscale.set("linear")
        self._replot_yscale.set("linear")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------

    def save_accepted(self):
        self.pull_vars()
        if not self.generations:
            self.messagebox.showerror("No generation", "Generate at least one QA run first.")
            return
        idx = self._final_generation_idx
        if idx < 0 or idx >= len(self.generations):
            idx = self.current_generation_idx
        if idx < 0:
            self.messagebox.showerror("No generation", "Select a generation first.")
            return
        md         = self.generations[idx]
        out_root   = self._resolve_final_root()
        folder_name = self.config.get("final_folder_name") or f"accepted_calibration_{safe_stem(self.config.get('session_name','calibration'))}_{now_timestamp()}"
        # FIX D: if nothing is checked, abort with an error instead of silently
        # passing None (which means "export ALL") downstream.
        selected_keys = [k for k, v in self._final_item_vars.items() if v.get()]
        if not selected_keys:
            self.messagebox.showerror("Nothing selected", "No items are selected for export.")
            return
        n_selected = len(selected_keys)
        self.log(f"Saving accepted calibration — generation={md.get('generation')} selected_keys={n_selected}")
        try:
            handoff = export_accepted_generation(self.config, md, out_root, folder_name, selected_keys=selected_keys)
            ver     = handoff.get("verification", {})
            n_copied  = len(handoff.get("copied_files", {}))
            n_miss_req = len(ver.get("missing", []))
            self.log("Saved accepted calibration: " + json.dumps(ver))
            self.final_status.configure(
                text=(
                    f"Folder: {handoff.get('accepted_folder')}\n"
                    f"Selected: {n_selected}  Copied: {n_copied}  Missing required: {n_miss_req}\n"
                    f"Missing: {ver.get('missing', [])}"
                )
            )
            handoff_json = str(Path(handoff["accepted_folder"]) / "metadata" / "calibration_handoff.json")
            self.config["latest_accepted_handoff"] = handoff_json
            self.save_config(silent=True)
            # Notify listeners (e.g. the reduction pane) so the accepted
            # calibration flows straight into the next stage.
            for fn in self._accept_listeners:
                try:
                    fn(handoff_json)
                except Exception as _e:
                    self.log(f"Accept listener failed: {_e!r}", "WARN")
            if ver.get("ok"):
                self.messagebox.showinfo("Saved", f"Accepted calibration saved and verified.\n{handoff.get('accepted_folder')}")
            else:
                self.messagebox.showwarning("Saved with missing files", json.dumps(ver, indent=2))
        except Exception as e:
            tb = traceback.format_exc()
            self.log("Accepted save failed: " + repr(e), "ERROR")
            print(tb, flush=True)
            self.messagebox.showerror("Accepted save failed", str(e))

    def _resolve_final_root(self) -> str:
        """Single fallback chain for the accepted-export root, used by both save
        and 'open folder' so they never disagree."""
        return (self.config.get("final_output_root")
                or self.config.get("accepted_output_root")
                or self.config.get("metadata_root")
                or str(output_base(self.config) / "accepted_calibrations"))

    def open_final_root(self):
        self.pull_vars()
        p = self._resolve_final_root()
        if not p:
            return
        if os.name == "nt":
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])

    # ------------------------------------------------------------------
    # App lifecycle
    # ------------------------------------------------------------------

    def confirm_shutdown(self) -> bool:
        """Return whether the pane may close without changing pane state."""
        if self._qa_running:
            return self.messagebox.askyesno(
                "Calibration running",
                "Calibration processing is still running. Stop it and close?",
            )
        return True

    def shutdown(self, confirm: bool = True) -> bool:
        """Save config and tear down. Returns False if the user cancelled.

        The unified host calls this for the embedded pane; it never destroys
        the shared root (only the owner does that in on_close).
        """
        if confirm and not self.confirm_shutdown():
            return False
        # Don't orphan the pyFAI worker: terminate it if still alive.
        proc = getattr(self, "_qa_proc", None)
        if proc is not None and proc.poll() is None:
            terminate_process_tree(proc)
        self._closing = True  # stop the log-drain poller from rescheduling
        self.save_config(silent=True)
        self.log("Calibration pane closed")
        return True

    def on_close(self):
        if not self.shutdown(confirm=True):
            return
        if self._owns_root:
            self.root.destroy()

    def add_accept_listener(self, fn) -> None:
        """Register fn(handoff_json_path) to be called after a successful
        accepted-calibration export (used to live-populate the reduction pane)."""
        self._accept_listeners.append(fn)

    def run(self) -> int:
        assert self._owns_root, "run() is only valid for a standalone (root-owning) app"
        self.root.mainloop()
        return 0


def make_calib_pane(parent_frame, config_path: "str | Path") -> "CalibrationApp":
    """Construct CalibrationApp embedded in a parent frame (for the unified app)."""
    return CalibrationApp(config_path=config_path, parent=parent_frame)


def run_app(config_path: "str | Path") -> int:
    from ..guikit.dpi import enable_hi_dpi
    enable_hi_dpi()
    app = CalibrationApp(config_path=config_path)
    return app.run()
