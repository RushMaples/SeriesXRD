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
    from bulkxrd.core.config import read_json, write_json, print_status, make_stdio_robust
    from bulkxrd.analysis.background import run_background_separation
    from bulkxrd.analysis.peaks import run_peak_fitting
    from bulkxrd.analysis.identify import run_identification
    from bulkxrd.analysis.residual import run_residual
    from bulkxrd.analysis.phases import load_library, pymatgen_available
    from bulkxrd.analysis.frame_metadata import import_csv_to_analysis
    from bulkxrd.analysis.ml_rank import rank_candidates
else:
    from ..core.config import read_json, write_json, print_status, make_stdio_robust
    from .background import run_background_separation
    from .peaks import run_peak_fitting
    from .identify import run_identification
    from .residual import run_residual
    from .phases import load_library, pymatgen_available
    from .frame_metadata import import_csv_to_analysis
    from .ml_rank import rank_candidates


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


def _opt_int(v):
    f = _opt_float(v)
    return None if f is None else int(round(f))


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

    # Optional pressure-CSV import onto /frames before Step 3 (the metadata seam).
    # Filename-parsed pressures are already populated by Step 1; a CSV overrides
    # (merging — only the frames the CSV provides). Because the user supplied an
    # explicit prior, a failure is FATAL by default: silently continuing with
    # filename/stale/no pressure would run Step 3 against the wrong prior. Set
    # pressure_csv_required=False to fall back to a warning instead.
    csv_path = str(cfg.get("pressure_csv", "") or "").strip()
    if csv_path:
        csv_required = _as_bool(cfg.get("pressure_csv_required", True), True)
        if not (out_path and Path(out_path).expanduser().is_file()):
            msg = (f"pressure_csv given but there is no analysis file to apply it to "
                   f"({out_path!r}); run Step 1 or set analysis_h5_file.")
            if csv_required:
                raise FileNotFoundError(msg)
            print_status(msg + " — skipped", "WARN")
        else:
            try:
                mm = import_csv_to_analysis(out_path, csv_path)
                manifest["frame_metadata"] = {"csv": mm.get("csv"),
                                              "summary": mm.get("summary"),
                                              "n_mapped": mm.get("n_mapped")}
                print_status(f"Imported pressure CSV ({mm.get('n_mapped')} frames): "
                             f"{mm.get('summary')}")
            except Exception as e:
                if csv_required:
                    raise RuntimeError(
                        f"Pressure CSV import failed for {csv_path!r}: {e}. Fix the CSV "
                        f"or set pressure_csv_required=False to continue without it.") from e
                print_status(f"Pressure CSV import skipped: {e!r}", "WARN")

    if run_step2:
        if not out_path or not Path(out_path).expanduser().is_file():
            raise FileNotFoundError(f"Analysis HDF5 not found for peak fitting: {out_path!r}")
        m2 = run_peak_fitting(
            out_path, None,
            source=(str(cfg.get("peak_source", "auto") or "auto").strip() or "auto"),
            sensitivity=(str(cfg.get("sensitivity", "normal") or "normal").strip() or "normal"),
            auto_range=_as_bool(cfg.get("auto_range", True), True),
            hybrid_spike_bins=_as_int(cfg.get("hybrid_spike_bins"), 5),
            # Blank detection knobs fall back to the sensitivity preset; an
            # explicit value overrides it.
            min_snr=_opt_float(cfg.get("min_snr")),
            min_prominence_snr=_opt_float(cfg.get("min_prominence_snr")),
            window_factor=_as_float(cfg.get("window_factor"), 3.0),
            max_chi2=_as_float(cfg.get("max_chi2"), 25.0),
            edge_bins=_opt_int(cfg.get("edge_bins")),
            fit_min=_opt_float(cfg.get("fit_min")),
            fit_max=_opt_float(cfg.get("fit_max")),
            min_fwhm_bins=_opt_float(cfg.get("min_fwhm_bins")),
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
        workspace = cfg.get("workspace_root") or str(Path(out_path).expanduser().parent)
        lib = load_library(workspace)
        identify_all = _as_bool(cfg.get("identify_all_phases", False), False)
        run_ml_rank = _as_bool(cfg.get("run_ml_rank", False), False)
        phases = None

        # ML candidate ranking (Step 3b proposer): rank the WHOLE library against
        # each frame, then VERIFY only the top-K with the deterministic matcher
        # below — "ML proposes, physics verifies". Pure-numpy ranker (no torch);
        # needs pymatgen to simulate, so it's skipped with a warning when absent.
        # NOTE: pymatgen_available / rank_candidates are imported at module scope
        # (the dual bootstrap block up top handles the GUI's direct script-path
        # launch). Do NOT re-import them here: a function-local import bypasses
        # test monkeypatching AND, if written relative, crashes in script mode.
        if run_ml_rank:
            if not pymatgen_available():
                print_status("ML candidate ranking needs pymatgen — falling back to the "
                             "candidate selection.", "WARN")
            else:
                pool = list(lib.values())
                if not pool:
                    raise ValueError("Reference library is empty — add or bundle phases first.")
                mrank = rank_candidates(
                    out_path, pool,
                    source=str(cfg.get("ml_rank_source", "auto") or "auto").strip() or "auto",
                    top_k=_as_int(cfg.get("ml_rank_top_k"), 5),
                    # 'cosine' (default) or 'torch:<model.pt>' — a trained
                    # bulkxrd-ml-train export (see docs/ml-training.md).
                    scorer=(str(cfg.get("ml_scorer", "") or "").strip() or None))
                manifest["ml_rank"] = mrank
                manifest["steps"].append("ml_rank")
                phases = [lib[n] for n in mrank["candidates"] if n in lib]
                if phases:
                    print_status(f"ML ranker shortlisted {len(phases)} phase(s) for "
                                 f"verification: {[p.name for p in phases]}")
                else:
                    print_status("ML ranker produced no candidates — falling back to "
                                 "the candidate selection.", "WARN")
                    phases = None

        # No ML shortlist: open-set (whole library) or the explicit Phases-tab selection.
        if phases is None:
            if identify_all:
                phases = list(lib.values())
                if not phases:
                    raise ValueError("Reference library is empty — add or bundle phases first.")
            else:
                names = [str(n) for n in (cfg.get("candidate_phases") or [])]
                if not names:
                    raise ValueError(
                        "Step 3a needs candidate phases — enable some on the Phases tab, "
                        "turn on 'Search entire library', or enable ML candidate ranking.")
                phases = [lib[n] for n in names if n in lib]
                missing = [n for n in names if n not in lib]
                if missing:
                    print_status(f"Candidate phases not found in library, skipped: {missing}", "WARN")
                if not phases:
                    raise ValueError("None of the candidate phases resolve in the reference library.")
        rel_tol = _as_float(cfg.get("rel_tol"), 0.01)
        min_matched = _as_int(cfg.get("min_matched"), 3)
        m3 = run_identification(
            out_path, phases,
            wavelength=_opt_float(cfg.get("identify_wavelength")),
            p_min=_as_float(cfg.get("p_min"), 0.0),
            p_max=_as_float(cfg.get("p_max"), 100.0),
            rel_tol=rel_tol,
            num_workers=num_workers,
            use_frame_pressure=_as_bool(cfg.get("use_pressure_prior", True), True),
            pressure_window=_as_float(cfg.get("pressure_window"), 2.0),
            pressure_sigma_k=_as_float(cfg.get("pressure_sigma_k"), 2.0),
            min_matched=min_matched,
            marker_prior=_as_bool(cfg.get("marker_prior", False), False),
            # Soft intensity-agreement factor (0 = position-only confidence).
            intensity_k=_as_float(cfg.get("intensity_k"), 0.3),
            use_frame_temperature=_as_bool(cfg.get("use_frame_temperature", True), True),
        )
        out_path = m3["out_h5"]
        manifest["step3"] = m3
        manifest["steps"].append("identify")

        # Remove the identified phases and re-detect on the residual, so weaker
        # and unknown features the strong peaks were masking become readable.
        m3r = run_residual(
            out_path, phases,
            seen_conf=_as_float(cfg.get("seen_conf"), 0.5),
            rel_tol=rel_tol,
            min_snr=_as_float(cfg.get("min_snr"), 5.0),
            min_matched=min_matched,
            allow_sparse=_as_bool(cfg.get("allow_sparse", False), False),
        )
        out_path = m3r["out_h5"]
        manifest["step3_residual"] = m3r
        manifest["steps"].append("residual")

        # Step 3c: cluster the residual peaks into coherent unknown-phase
        # candidates (cheap; skipped when the residual left nothing behind).
        if _as_bool(cfg.get("run_step3c", True), True) and m3r.get("n_residual_peaks"):
            from .unknowns import run_unknowns   # local: keeps script-mode bootstrap simple
            m3c = run_unknowns(
                out_path,
                min_track_frames=_as_int(cfg.get("unknown_min_frames"), 3),
                jaccard_threshold=_as_float(cfg.get("unknown_jaccard"), 0.6),
            )
            manifest["step3_unknowns"] = m3c
            manifest["steps"].append("unknowns")

    manifest["analysis_h5_file"] = out_path
    return manifest


def main() -> int:
    make_stdio_robust()   # never let a non-ASCII log line crash on a cp1252 console
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
