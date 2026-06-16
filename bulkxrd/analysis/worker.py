"""Worker subprocess for the analysis stage (Step 1 background + Step 2 peaks).

Mirrors reduce/worker.py: the GUI launches this in a subprocess so a hard crash
in numpy/scipy/h5py is isolated from the supervising UI. Reads the analysis
session config, runs the requested step(s), and writes a manifest JSON.

Progress is emitted by the underlying drivers as ``[ANALYSIS] <done> <total>``
(Step 1) and ``[PEAKS] <done> <total>`` (Step 2) lines on stdout, which the GUI
parses to drive its progress bar.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import traceback

# Works both as a package module (python -m bulkxrd.analysis.worker) and as a
# directly-launched script (the GUI runs this file by path in a subprocess).
if __package__ in (None, ""):
    _pkg_parent = str(Path(__file__).resolve().parents[2])
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    from bulkxrd.core.config import read_json, write_json, print_status
    from bulkxrd.analysis.background import run_background_separation
    from bulkxrd.analysis.peaks import run_peak_fitting
else:
    from ..core.config import read_json, write_json, print_status
    from .background import run_background_separation
    from .peaks import run_peak_fitting


def _as_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_int(v, default: int) -> int:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


def _as_float(v, default: float) -> float:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _opt_float(v):
    s = str(v).strip() if v is not None else ""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def run_analysis(cfg: dict) -> dict:
    """Drive Step 1 and/or Step 2 from a config dict. Returns a merged manifest."""
    reduced = str(cfg.get("reduced_h5_file", "") or "").strip()
    out_path = str(cfg.get("analysis_h5_file", "") or "").strip()
    run_step1 = _as_bool(cfg.get("run_step1", True), True)
    run_step2 = _as_bool(cfg.get("run_step2", True), True)

    manifest: dict = {"steps": []}

    if run_step1:
        if not reduced or not Path(reduced).expanduser().is_file():
            raise FileNotFoundError(f"Reduced HDF5 not found: {reduced!r}")
        out_arg = out_path or None
        thr = _opt_float(cfg.get("contamination_threshold"))
        m1 = run_background_separation(
            reduced, out_arg,
            max_half_window=_as_int(cfg.get("max_half_window"), 40),
            n_passes=_as_int(cfg.get("n_passes"), 1),
            use_lls=_as_bool(cfg.get("use_lls", True), True),
            contamination_threshold=thr,
        )
        out_path = m1["out_h5"]
        manifest["step1"] = m1
        manifest["steps"].append("background")
    elif not out_path:
        # Step 2 only: operate on an already-written analysis file.
        raise ValueError("run_step1 is off and no analysis_h5_file given to fit peaks into.")

    if run_step2:
        target = out_path
        if not target or not Path(target).expanduser().is_file():
            raise FileNotFoundError(f"Analysis HDF5 not found for peak fitting: {target!r}")
        m2 = run_peak_fitting(
            target, None,
            min_snr=_as_float(cfg.get("min_snr"), 5.0),
            window_factor=_as_float(cfg.get("window_factor"), 3.0),
            max_chi2=_as_float(cfg.get("max_chi2"), 25.0),
            propagate_seeds=_as_bool(cfg.get("propagate_seeds", True), True),
        )
        out_path = m2["out_h5"]
        manifest["step2"] = m2
        manifest["steps"].append("peaks")

    manifest["analysis_h5_file"] = out_path
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    try:
        cfg = read_json(args.config)
        print_status("Analysis worker starting")
        manifest = run_analysis(cfg)
        write_json(args.output_json, manifest)
        print_status(f"Analysis worker completed -> {args.output_json}")
        return 0
    except Exception as e:
        print_status("Analysis worker failed: " + repr(e), "ERROR")
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
