"""Unified SeriesXRD desktop application.

The application presents calibration, reduction, and analysis as one guided
workflow. Tk imports remain deferred so the package can be imported and tested
on headless systems.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .core.config import (
    TOOL_NAME, VERSION, ensure_dir, write_json,
    SessionConfig, config_path_for_notebook,
)
from .reduce.session import seed_reduction_config
from .analysis.session import seed_analysis_config

DEFAULT_WORKSPACE = Path.home() / "seriesxrd_workspace"
STAGE_TABS = ("1 Calibration", "2 Reduction", "3 Analysis")


def workspace_launch_args(workspace: "str | Path", executable: str | None = None) -> list[str]:
    """Return the platform-neutral command used to open another workspace."""
    path = Path(workspace).expanduser().resolve()
    return [executable or sys.executable, "-m", "seriesxrd.app", "--workspace", str(path)]


def _seed_calibration_config(workspace: Path) -> Path:
    """Ensure a calibration config exists in the workspace; return its path."""
    cfg_path = config_path_for_notebook(workspace)
    if not cfg_path.exists():
        cfg = SessionConfig.default(workspace).to_dict()
        cfg["workspace_root"] = str(workspace)
        cfg["session_config_path"] = str(cfg_path)
        write_json(cfg_path, cfg)
    return cfg_path


class SeriesXRDApp:
    """The unified host window."""

    def __init__(self, workspace: "str | Path"):
        import tkinter as tk
        from tkinter import ttk
        from .guikit.tkstyle import apply_dark_theme
        from .guikit.theme import BG, MUTED
        from .calib.gui import make_calib_pane
        from .reduce.gui import make_reduce_pane
        from .analysis.gui import make_analysis_pane

        self.tk = tk
        self.workspace = ensure_dir(workspace)
        calib_cfg = _seed_calibration_config(self.workspace)
        reduce_cfg = seed_reduction_config(self.workspace)
        analysis_cfg = seed_analysis_config(self.workspace)

        self.root = tk.Tk()
        self.root.title(f"{TOOL_NAME} — {self.workspace.name}")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(1280, sw - 80)}x{min(900, sh - 120)}")
        self.root.minsize(min(1000, sw - 80), min(700, sh - 120))
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        apply_dark_theme(self.root, ttk)
        self.root.configure(bg=BG)

        header = ttk.Frame(self.root, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(
            header, text=TOOL_NAME, font=("TkDefaultFont", 16, "bold"),
        ).pack(side="left")
        ttk.Label(
            header, text=f"  {self.workspace}", foreground=MUTED,
        ).pack(side="left")
        ttk.Label(
            header, text="1  Calibrate   →   2  Reduce   →   3  Analyze",
            foreground=MUTED,
        ).pack(side="right")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True)

        calib_tab = ttk.Frame(self.nb)
        reduce_tab = ttk.Frame(self.nb)
        analysis_tab = ttk.Frame(self.nb)
        self.nb.add(calib_tab, text=STAGE_TABS[0])
        self.nb.add(reduce_tab, text=STAGE_TABS[1])
        self.nb.add(analysis_tab, text=STAGE_TABS[2])

        self.calib_pane = make_calib_pane(calib_tab, calib_cfg)
        self.reduce_pane = make_reduce_pane(reduce_tab, reduce_cfg)
        self.calib_pane.add_accept_listener(self._on_calibration_accepted)

        self.analysis_pane = make_analysis_pane(analysis_tab, analysis_cfg)
        self.reduce_pane.add_reduced_listener(self._on_reduction_ready)

        self._build_menubar()

    # ------------------------------------------------------------------

    def _build_menubar(self):
        tk = self.tk
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open workspace…", command=self._open_workspace)
        file_menu.add_command(label="Save all settings", accelerator="Ctrl+S", command=self._save_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_quit)
        self.root.bind("<Control-s>", lambda e: self._save_all())

        go_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Go", menu=go_menu)
        for index, label in enumerate(STAGE_TABS):
            go_menu.add_command(
                label=label, accelerator=f"Ctrl+{index + 1}",
                command=lambda i=index: self._select_stage(i),
            )
            self.root.bind(
                f"<Control-Key-{index + 1}>",
                lambda _event, i=index: self._select_stage(i),
            )

        calib_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Calibration", menu=calib_menu)
        calib_menu.add_command(label="Environment settings…", command=self.calib_pane.open_env_settings)
        calib_menu.add_command(label="Launch Dioptas", command=self.calib_pane.open_dioptas)
        calib_menu.add_command(label="View log", command=self.calib_pane.open_console_logs)

        reduction_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Reduction", menu=reduction_menu)
        reduction_menu.add_command(
            label="Use latest calibration", command=self.reduce_pane._use_latest_calibration)
        reduction_menu.add_command(label="Scan data", command=self.reduce_pane.scan_dataset_clicked)
        reduction_menu.add_command(
            label="Review reduced data", command=self.reduce_pane.inspect_h5_clicked)
        reduction_menu.add_command(label="View log", command=self.reduce_pane.open_console_logs)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Use latest reduced data",
                                  command=self._handoff_reduced_to_analysis)
        analysis_menu.add_command(label="Inspect input", command=self.analysis_pane.inspect_input_clicked)
        analysis_menu.add_command(
            label="Export refinement bundle…",
            command=self.analysis_pane.export_refinement_clicked)
        analysis_menu.add_command(
            label="Export GSAS-ready raw patterns…",
            command=self.analysis_pane.export_gsas_raw_clicked)
        analysis_menu.add_command(label="View log", command=self.analysis_pane.open_console_logs)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Check environment", command=self._check_environment)
        tools_menu.add_command(label="Inspect detector image…", command=self._inspect_image)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._about)

    # ------------------------------------------------------------------

    def _select_stage(self, index: int):
        self.nb.select(max(0, min(int(index), len(STAGE_TABS) - 1)))

    def _on_calibration_accepted(self, handoff_path) -> None:
        """Pass an accepted calibration forward and reveal the next stage."""
        self.reduce_pane.set_handoff(handoff_path)
        self._select_stage(1)

    def _on_reduction_ready(self, reduced_path) -> None:
        """Pass reduced data forward and reveal the analysis stage."""
        self.analysis_pane.set_reduced(reduced_path)
        self._select_stage(2)

    def _save_all(self):
        try:
            self.calib_pane.save_config(silent=True)
            self.reduce_pane.save_config(silent=True)
            self.analysis_pane.save_config(silent=True)
            self.calib_pane.log("Saved all configs")
        except Exception as e:
            self.calib_pane.log(f"Save all failed: {e!r}", "WARN")

    def _handoff_reduced_to_analysis(self):
        """Push the reduction stage's most recent reduced .h5 into the analysis pane."""
        from tkinter import messagebox
        self.reduce_pane.pull_vars()
        reduced = str(self.reduce_pane.config.get("reduced_h5_file", "") or "").strip()
        if not reduced or not Path(reduced).is_file():
            messagebox.showinfo(
                "No reduced output",
                "No reduced .h5 is available yet. Run a reduction (or pick one on the "
                "Reduction → Review tab) first.")
            return
        self._on_reduction_ready(reduced)

    def _open_workspace(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askdirectory(title="Open workspace folder")
        if not path:
            return
        selected = Path(path).expanduser().resolve()
        if selected == self.workspace:
            messagebox.showinfo("Workspace", "That workspace is already open.")
            return
        if not self._confirm_shutdown_panes():
            return
        try:
            subprocess.Popen(workspace_launch_args(selected))
        except OSError as exc:
            messagebox.showerror(
                "Could not open workspace",
                f"SeriesXRD could not start a new window.\n\n{exc}",
            )
            return
        self._shutdown_panes(confirm=False)
        self.root.destroy()

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
        import tkinter as tk
        from tkinter import filedialog
        from .guikit.theme import BG2, FG
        path = filedialog.askopenfilename(title="Inspect detector image")
        if not path:
            return
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "seriesxrd.core.inspect", path],
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
            ver = f"  seriesxrd: {VERSION}"
        messagebox.showinfo(
            f"About {TOOL_NAME}",
            f"{TOOL_NAME} {VERSION}\n\nWorkspace:\n  {self.workspace}\n\nVersions:\n{ver}")

    # ------------------------------------------------------------------

    def _stage_panes(self):
        return (self.calib_pane, self.reduce_pane, self.analysis_pane)

    def _confirm_shutdown_panes(self) -> bool:
        """Ask every stage before mutating any stage's lifecycle state."""
        for pane in self._stage_panes():
            confirm = getattr(pane, "confirm_shutdown", None)
            if callable(confirm) and not confirm():
                return False
        return True

    def _shutdown_panes(self, confirm: bool = True) -> bool:
        if confirm and not self._confirm_shutdown_panes():
            return False
        for pane in self._stage_panes():
            if not pane.shutdown(confirm=False):
                return False
        return True

    def _on_quit(self):
        if self._shutdown_panes(confirm=True):
            self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="seriesxrd", description="SeriesXRD desktop application")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE),
                        help=f"Workspace folder for configs and outputs (default: {DEFAULT_WORKSPACE})")
    args = parser.parse_args(argv)

    from .guikit.dpi import enable_hi_dpi
    enable_hi_dpi()
    app = SeriesXRDApp(workspace=args.workspace)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
