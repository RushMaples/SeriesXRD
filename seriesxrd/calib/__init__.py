"""Calibration review stage: pyFAI QA generation, masking GUI, accepted export.

Pattern shared by every seriesxrd stage: pure logic modules (`processing`),
a crash-isolated `worker` subprocess, an optional `gui`, and a `run_gui`
CLI entry point.

PEP 562 lazy loading — submodules are NOT imported at package import time so
that importing seriesxrd.calib works even when numpy/pyFAI are absent. Only
accessing a name from .processing or .gui will trigger those heavy imports.
"""
from __future__ import annotations

__all__ = [
    # processing (requires numpy / pyFAI)
    "export_accepted_generation",
    "generate_qa_run",
    "load_pyfai_integrator",
    "read_poni_info",
    "suggest_integration_settings",
    "preview_cake_orientations",
    "runtime_versions",
    # dioptas (stdlib only)
    "build_dioptas_command",
    "launch_dioptas",
    "dioptas_manual_instructions",
    # gui (requires tkinter)
    "CalibrationApp",
    "make_calib_pane",
    "run_app",
]

# Map each exported name to the submodule that owns it.
_NAME_TO_MODULE: dict[str, str] = {
    # processing
    "export_accepted_generation":   ".processing",
    "generate_qa_run":              ".processing",
    "load_pyfai_integrator":        ".processing",
    "read_poni_info":               ".processing",
    "suggest_integration_settings": ".processing",
    "preview_cake_orientations":    ".processing",
    "runtime_versions":             ".processing",
    # dioptas
    "build_dioptas_command":        ".dioptas",
    "launch_dioptas":               ".dioptas",
    "dioptas_manual_instructions":  ".dioptas",
    # gui
    "CalibrationApp":               ".gui",
    "make_calib_pane":              ".gui",
    "run_app":                      ".gui",
}


def __getattr__(name: str):
    """Lazily import and return a name from the owning submodule (PEP 562)."""
    if name not in _NAME_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    submod = importlib.import_module(_NAME_TO_MODULE[name], package=__name__)
    attr = getattr(submod, name)
    # Cache in module namespace so subsequent accesses skip __getattr__.
    globals()[name] = attr
    return attr


def __dir__():
    return list(globals().keys()) + __all__
