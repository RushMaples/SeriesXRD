"""Worker bootstrap regressions.

The GUI launches ``seriesxrd/analysis/worker.py`` directly by file path, not with
``python -m``. In that mode ``__package__`` is empty, so lazy relative imports
inside ``run_analysis`` fail even though the top-of-file bootstrap imports work.
"""
import json
import runpy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_worker_script_path_handles_lazy_ml_imports(monkeypatch):
    """Script-path worker execution must not use lazy relative imports.

    This reproduces the GUI failure mode: run Step 3 with ML ranking enabled from
    a direct ``worker.py`` launch. The heavy analysis functions are monkeypatched
    so the test isolates the import/bootstrap seam instead of doing real phase
    fitting.
    """
    import seriesxrd.analysis.identify as identify_mod
    import seriesxrd.analysis.phases as phases_mod
    import seriesxrd.analysis.residual as residual_mod

    monkeypatch.setattr(phases_mod, "pymatgen_available", lambda: False)

    def _fake_identify(path, phases, **kwargs):
        return {"out_h5": str(path), "phases": [p.name for p in phases]}

    def _fake_residual(path, phases, **kwargs):
        return {"out_h5": str(path)}

    monkeypatch.setattr(identify_mod, "run_identification", _fake_identify)
    monkeypatch.setattr(residual_mod, "run_residual", _fake_residual)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        analysis_h5 = td / "analysis.h5"
        analysis_h5.write_bytes(b"placeholder: worker only checks existence here")
        cfg_path = td / "analysis_config.json"
        out_json = td / "manifest.json"
        cfg_path.write_text(json.dumps({
            "analysis_h5_file": str(analysis_h5),
            "workspace_root": str(td),
            "candidate_phases": ["Au"],
            "run_step1": False,
            "run_step2": False,
            "run_step3": True,
            "run_ml_rank": True,
            "num_workers": "1",
        }), encoding="utf-8")

        monkeypatch.setattr(sys, "argv", [
            str(Path("seriesxrd") / "analysis" / "worker.py"),
            "--config", str(cfg_path),
            "--output-json", str(out_json),
        ])

        worker_path = Path(__file__).resolve().parents[1] / "seriesxrd" / "analysis" / "worker.py"
        try:
            runpy.run_path(str(worker_path), run_name="__main__")
        except SystemExit as e:
            assert int(e.code or 0) == 0

        manifest = json.loads(out_json.read_text(encoding="utf-8"))
        assert manifest["steps"] == ["identify", "residual"]
        assert manifest["analysis_h5_file"] == str(analysis_h5)


def test_worker_script_path_does_not_shadow_stdlib_fractions(monkeypatch):
    """Direct worker launch must not put seriesxrd/analysis ahead of stdlib.

    The analysis package has a sibling ``fractions.py`` module for phase
    fractions. When ``worker.py`` is launched directly by path, Python places
    that sibling directory on ``sys.path``; optional simulators then resolve
    ``import fractions`` to the project file instead of the standard library.
    """
    worker_path = Path(__file__).resolve().parents[1] / "seriesxrd" / "analysis" / "worker.py"
    script_dir = str(worker_path.parent)
    repo_root = str(Path(__file__).resolve().parents[1])

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg_path = td / "analysis_config.json"
        out_json = td / "manifest.json"
        cfg_path.write_text(json.dumps({
            "run_step1": False,
            "run_step2": False,
            "run_step3": False,
        }), encoding="utf-8")

        monkeypatch.setattr(sys, "path", [
            script_dir,
            repo_root,
            *[p for p in sys.path if p not in (script_dir, repo_root)],
        ])
        monkeypatch.setattr(sys, "argv", [
            str(worker_path),
            "--config", str(cfg_path),
            "--output-json", str(out_json),
        ])

        sys.modules.pop("fractions", None)
        try:
            runpy.run_path(str(worker_path), run_name="__main__")
        except SystemExit as e:
            assert int(e.code or 0) == 0

        sys.modules.pop("fractions", None)
        from fractions import Fraction
        import fractions

        assert Fraction(1, 2).numerator == 1
        assert Path(fractions.__file__).resolve().parent != worker_path.parent

        manifest = json.loads(out_json.read_text(encoding="utf-8"))
        assert manifest["steps"] == []
