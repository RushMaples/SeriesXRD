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
    from bulkxrd.analysis.identify import run_identification
    from bulkxrd.analysis.phases import load_library
else:
    from ..core.config import read_json, write_json, print_status
    from .background import run_background_separation
    from .peaks import run_peak_fitting
    from .identify import run_identification
    from .phases import load_library


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
    """Drive Step 1 (background), Step 2 (peaks), and/or Step 3a (EOS phase
    matching) from a config dict. Returns a merged manifest."""
    reduced = str(cfg.get("reduced_h5_file", "") or "").strip()
    out_path = str(cfg.get("analysis_h5_file", "") or "").strip()
    run_step1 = _as_bool(cfg.get("run_step1", True), True)
    run_step2 = _as_bool(cfg.get("run_step2", True), True)
    run_step3 = _as_bool(cfg.get("run_step3", False), False)
    num_workers = _as_int(cfg.get("num_workers"), 1)

    manifest: dict = {"steps": []}

    if run_step1:
        if not reduced or not Path(reduced).expanduser().is_file():
            raise FileNotFoundError(f"Reduced HDF5 not found: {reduced!r}")
        thr = _opt_float(cfg.get("contamination_threshold"))
        m1 = run_background_separation(
            reduced, out_path or None,
            max_half_window=_as_int(cfg.get("max_half_window"), 40),
            n_passes=_as_int(cfg.get("n_passes"), 1),
            use_lls=_as_bool(cfg.get("use_lls", True), True),
            contamination_threshold=thr,
            num_workers=num_workers,
        )
        out_path = m1["out_h5"]
        manifest["step1"] = m1
        manifest["steps"].append("background")
    elif (run_step2 or run_step3) and not out_path:
        # Steps 2/3 operate on an already-written analysis file.
        raise ValueError("Step 1 is off and no analysis_h5_file given to operate on.")

    if run_step2:
        if not out_path or not Path(out_path).expanduser().is_file():
            raise FileNotFoundError(f"Analysis HDF5 not found for peak fitting: {out_path!r}")
        m2 = run_peak_fitting(
            out_path, None,
            min_snr=_as_float(cfg.get("min_snr"), 5.0),
            min_prominence_snr=_opt_float(cfg.get("min_prominence_snr")),
            window_factor=_as_float(cfg.get("window_factor"), 3.0),
            max_chi2=_as_float(cfg.get("max_chi2"), 25.0),
            edge_bins=_as_int(cfg.get("edge_bins"), 0),
            fit_min=_opt_float(cfg.get("fit_min")),
            fit_max=_opt_float(cfg.get("fit_max")),
            min_fwhm_bins=_as_float(cfg.get("min_fwhm_bins"), 0.0),
            local_baseline_bins=_as_int(cfg.get("detrend_bins"), 0),
            propagate_seeds=_as_bool(cfg.get("propagate_seeds", True), True),
            num_workers=num_workers,
        )
        out_path = m2["out_h5"]
        manifest["step2"] = m2
        manifest["steps"].append("peaks")

    if run_step3:
        if not out_path or not Path(out_path).expanduser().is_file():
            raise FileNotFoundError(f"Analysis HDF5 not found for phase matching: {out_path!r}")
        names = [str(n) for n in (cfg.get("candidate_phases") or [])]
        if not names:
            raise ValueError(
                "Step 3a needs candidate phases — enable some on the Phases tab.")
        workspace = cfg.get("workspace_root") or str(Path(out_path).expanduser().parent)
        lib = load_library(workspace)
        phases = [lib[n] for n in names if n in lib]
        missing = [n for n in names if n not in lib]
        if missing:
            print_status(f"Candidate phases not found in library, skipped: {missing}", "WARN")
        if not phases:
            raise ValueError("None of the candidate phases resolve in the reference library.")
        m3 = run_identification(
            out_path, phases,
            wavelength=_opt_float(cfg.get("identify_wavelength")),
            p_min=_as_float(cfg.get("p_min"), 0.0),
            p_max=_as_float(cfg.get("p_max"), 100.0),
            rel_tol=_as_float(cfg.get("rel_tol"), 0.01),
            num_workers=num_workers,
        )
        out_path = m3["out_h5"]
        manifest["step3"] = m3
        manifest["steps"].append("identify")

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
