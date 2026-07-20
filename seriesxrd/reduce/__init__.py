"""Dataset reduction stage: batch azimuthal integration of image series.

Applies an accepted calibration (PONI + mask via the
``calibration_handoff.json`` written by `seriesxrd.calib`) to a sample
dataset: every frame is integrated into a 1D pattern (mean and optional
azimuthal-median "robust" pattern) and optionally a 2D cake, written to one
HDF5 file plus a JSON manifest.

Follows the stage pattern shared by every seriesxrd stage: pure logic modules
(`processing`), a crash-isolated `worker` subprocess, an optional `gui`, and
a `run_gui` CLI entry point.
"""
from __future__ import annotations

from .processing import (
    DEFAULT_PATTERNS,
    reduce_dataset,
    scan_dataset,
)
from .session import seed_reduction_config
from .review import review_reduction, gallery_frames, set_excluded
from .gui import make_reduce_pane, run_app

__all__ = [
    "DEFAULT_PATTERNS", "reduce_dataset", "scan_dataset",
    "seed_reduction_config",
    "review_reduction", "gallery_frames", "set_excluded",
    "make_reduce_pane", "run_app",
]
