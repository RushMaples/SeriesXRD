"""Headless tests for unified-application coordination."""
from __future__ import annotations

from pathlib import Path

from seriesxrd.app import SeriesXRDApp, workspace_launch_args
from seriesxrd.analysis.gui import AnalysisApp
from seriesxrd.reduce.gui import ReductionApp


class _FakePane:
    def __init__(self, allow_close: bool):
        self.allow_close = allow_close
        self.confirm_calls = 0
        self.shutdown_calls = []

    def confirm_shutdown(self) -> bool:
        self.confirm_calls += 1
        return self.allow_close

    def shutdown(self, confirm: bool = True) -> bool:
        self.shutdown_calls.append(confirm)
        return self.allow_close


class _FakeRoot:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def test_unified_close_confirmation_is_transactional():
    """A later veto must not partially shut down an earlier stage."""
    app = SeriesXRDApp.__new__(SeriesXRDApp)
    app.calib_pane = _FakePane(True)
    app.reduce_pane = _FakePane(False)
    app.analysis_pane = _FakePane(True)
    app.root = _FakeRoot()

    app._on_quit()

    assert app.calib_pane.confirm_calls == 1
    assert app.reduce_pane.confirm_calls == 1
    assert app.analysis_pane.confirm_calls == 0
    assert app.calib_pane.shutdown_calls == []
    assert app.reduce_pane.shutdown_calls == []
    assert app.analysis_pane.shutdown_calls == []
    assert not app.root.destroyed


def test_unified_close_shuts_all_panes_after_all_confirm():
    app = SeriesXRDApp.__new__(SeriesXRDApp)
    app.calib_pane = _FakePane(True)
    app.reduce_pane = _FakePane(True)
    app.analysis_pane = _FakePane(True)
    app.root = _FakeRoot()

    app._on_quit()

    for pane in (app.calib_pane, app.reduce_pane, app.analysis_pane):
        assert pane.confirm_calls == 1
        assert pane.shutdown_calls == [False]
    assert app.root.destroyed


def test_workspace_launch_args_use_module_entry_point(tmp_path):
    args = workspace_launch_args(tmp_path, executable="python-test")
    assert args[:4] == ["python-test", "-m", "seriesxrd.app", "--workspace"]
    assert Path(args[4]) == tmp_path.resolve()


def test_scientific_tools_are_exposed_by_gui_controllers():
    for name in (
        "export_refinement_clicked",
        "export_gsas_raw_clicked",
        "run_microstructure_clicked",
        "run_phase_fractions_clicked",
        "run_spot_tracking_clicked",
    ):
        assert callable(getattr(AnalysisApp, name))
    assert callable(getattr(ReductionApp, "_run_texture_job"))
