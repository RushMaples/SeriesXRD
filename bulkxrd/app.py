"""Unified Bulk-XRD desktop application.

A single window with a native menubar and one tab per pipeline stage:
Calibration, Reduction, and (planned) Analysis. Each stage's existing App is
embedded as a pane in a shared Tk root — the per-stage standalone entry points
(`bulkxrd-calib-gui`, `bulkxrd-reduce-gui`) keep working unchanged.

All Tk imports are deferred into functions so this module imports headlessly
(e.g. for `tests/test_imports.py` on a box without tkinter).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .core.config import (
    TOOL_NAME, VERSION, ensure_dir, write_json, read_json,
    SessionConfig, config_path_for_notebook,
)
from .reduce.session import seed_reduction_config

DEFAULT_WORKSPACE = Path.home() / "bulkxrd_workspace"


def _seed_calibration_config(workspace: Path) -> Path:
    """Ensure a calibration config exists in the workspace; return its path."""
    cfg_path = config_path_for_notebook(workspace)
    if not cfg_path.exists():
        cfg = SessionConfig.default(workspace).to_dict()
        cfg["workspace_root"] = str(workspace)
        cfg["session_config_path"] = str(cfg_path)
        write_json(cfg_path, cfg)
    return cfg_path


class BulkXRDApp:
    """The unified host window."""

    def __init__(self, workspace: "str | Path"):
        import tkinter as tk
        from tkinter import ttk
        from .guikit.tkstyle import apply_dark_theme
        from .guikit.theme import BG, MUTED
        from .calib.gui import make_calib_pane
        from .reduce.gui import make_reduce_pane

        self.tk = tk
        self.workspace = ensure_dir(workspace)
        calib_cfg = _seed_calibration_config(self.workspace)
        reduce_cfg = seed_reduction_config(self.workspace)

        self.root = tk.Tk()
        self.root.title(f"{TOOL_NAME} {VERSION}  —  {self.workspace}")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(1280, sw - 80)}x{min(900, sh - 120)}")
        self.root.minsize(min(1000, sw - 80), min(700, sh - 120))
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        apply_dark_theme(self.root, ttk)
        self.root.configure(bg=BG)

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)

        calib_tab = ttk.Frame(nb)
        reduce_tab = ttk.Frame(nb)
        analysis_tab = ttk.Frame(nb)
        nb.add(calib_tab, text="Calibration")
        nb.add(reduce_tab, text="Reduction")
        nb.add(analysis_tab, text="Analysis")

        # Embed the two stage panes.
        self.calib_pane = make_calib_pane(calib_tab, calib_cfg)
        self.reduce_pane = make_reduce_pane(reduce_tab, reduce_cfg)
        # Accepting a calibration flows its handoff straight into the reduction
        # pane (no manual JSON picking needed in the normal calibrate→reduce flow).
        self.calib_pane.add_accept_listener(self.reduce_pane.set_handoff)

        # Analysis placeholder.
        ttk.Label(analysis_tab, text="Analysis stage — planned",
                  foreground=MUTED).pack(expand=True)
        nb.tab(2, state="disabled")

        self._build_menubar()

    # ------------------------------------------------------------------

    def _build_menubar(self):
        tk = self.tk
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save All Configs", accelerator="Ctrl+S", command=self._save_all)
        file_menu.add_command(label="Open Workspace...", command=self._open_workspace)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_quit)
        self.root.bind("<Control-s>", lambda e: self._save_all())

        calib_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Calibration", menu=calib_menu)
        calib_menu.add_command(label="Environment Settings...", command=self.calib_pane.open_env_settings)
        calib_menu.add_command(label="Launch Dioptas", command=self.calib_pane.open_dioptas)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Check Environment", command=self._check_environment)
        tools_menu.add_command(label="Inspect Detector Image...", command=self._inspect_image)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._about)

    # ------------------------------------------------------------------

    def _save_all(self):
        try:
            self.calib_pane.save_config(silent=True)
            self.reduce_pane.save_config(silent=True)
            self.calib_pane.log("Saved all configs")
        except Exception as e:
            self.calib_pane.log(f"Save all failed: {e!r}", "WARN")

    def _open_workspace(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askdirectory(title="Open workspace folder")
        if not path:
            return
        messagebox.showinfo(
            "Switching workspace requires relaunch",
            f'Relaunch with:\n\n  bulkxrd --workspace "{path}"\n\nSelected:\n{path}')

    def _check_environment(self):
        from tkinter import messagebox
        from .core.env import check_dependencies
        dep = check_dependencies(sys.executable)
        lines = [f"Python checked: {getattr(dep, 'checked_in', '') or sys.executable}", ""]
        lines += [f"  {'OK ' if ok else 'MISSING'}  {pkg}" for pkg, ok in dep.required.items()]
        if dep.missing_required:
            lines += ["", "Missing: " + ", ".join(dep.missing_required)]
        else:
            lines += ["", "All required packages present."]
        messagebox.showinfo("Environment check", "\n".join(lines))

    def _inspect_image(self):
        import subprocess
        import tkinter as tk
        from tkinter import filedialog
        from .guikit.theme import BG2, FG
        path = filedialog.askopenfilename(title="Inspect detector image")
        if not path:
            return
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "bulkxrd.core.inspect", path],
                capture_output=True, text=True, timeout=60)
            out = (proc.stdout or "") + (proc.stderr or "")
        except Exception as e:
            out = f"Inspection failed: {e!r}"
        win = tk.Toplevel(self.root)
        win.title(f"Inspect: {Path(path).name}")
        win.geometry("760x520")
        txt = tk.Text(win, bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                      font=("TkFixedFont", 9), wrap="none")
        txt.insert("end", out)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)

    def _about(self):
        from tkinter import messagebox
        try:
            from .calib.processing import runtime_versions
            v = runtime_versions()
            ver = "\n".join(f"  {k}: {val}" for k, val in v.items())
        except Exception:
            ver = f"  bulkxrd: {VERSION}"
        messagebox.showinfo(
            f"About {TOOL_NAME}",
            f"{TOOL_NAME} {VERSION}\n\nWorkspace:\n  {self.workspace}\n\nVersions:\n{ver}")

    # ------------------------------------------------------------------

    def _on_quit(self):
        # Both panes must agree to close (a running worker can veto).
        if not self.calib_pane.shutdown(confirm=True):
            return
        if not self.reduce_pane.shutdown(confirm=True):
            return
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="bulkxrd", description="Unified Bulk-XRD application")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE),
                        help=f"Workspace folder for configs and outputs (default: {DEFAULT_WORKSPACE})")
    args = parser.parse_args(argv)

    from .guikit.dpi import enable_hi_dpi
    enable_hi_dpi()
    app = BulkXRDApp(workspace=args.workspace)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
