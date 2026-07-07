"""Headless batch driver for the analysis stage — no GUI.

Runs Steps 1-3 (background → peaks → EOS phase matching) over a reduced HDF5
with multiprocessing, for high-throughput processing of thousands of frames on
a workstation or cluster. Optionally exports an ML-ready dataset afterwards.

Examples
--------
    bulkxrd-analyze reduced.h5 --phases Au,Re --workers 0
    bulkxrd-analyze reduced.h5 --steps 12            # background + peaks only
    bulkxrd-analyze reduced.h5 --phases Au -o out.h5 --ml-export ds.npz

Phase names resolve against the reference library in ``--workspace`` (bundled
baseline + that workspace's user phases). Step 3a (and any phase work) needs
pymatgen; Steps 1-2 do not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _run(args) -> int:
    # Imports are deferred so --help works without numpy/h5py installed.
    from .background import run_background_separation
    from .peaks import run_peak_fitting
    from .identify import run_identification

    reduced = Path(args.reduced).expanduser().resolve()
    if not reduced.is_file():
        print(f"[ERROR] reduced HDF5 not found: {reduced}", flush=True)
        return 1
    steps = set(args.steps)
    out = args.out
    analysis_path = out  # tracks the current analysis file across steps

    if "1" in steps:
        m1 = run_background_separation(
            reduced, out or None,
            max_half_window=args.max_half_window, n_passes=args.n_passes,
            use_lls=not args.no_lls,
            contamination_threshold=args.contamination_threshold,
            num_workers=args.workers)
        analysis_path = m1["out_h5"]
    elif not analysis_path:
        analysis_path = str(reduced.with_name(reduced.stem + "_analysis.h5"))
    if not analysis_path or not Path(analysis_path).is_file():
        print(f"[ERROR] analysis file not available for steps {sorted(steps)}: "
              f"{analysis_path!r} (run step 1 or pass --out)", flush=True)
        return 1

    # Optional pressure-CSV import onto /frames before Step 3 (the metadata seam).
    # Filenames are already parsed at Step 1; a CSV merges/overrides. Fatal on
    # failure — the user supplied an explicit prior.
    if args.pressure_csv:
        from .frame_metadata import import_csv_to_analysis
        try:
            mm = import_csv_to_analysis(analysis_path, args.pressure_csv)
            print(f"[ANALYZE] imported pressure CSV ({mm.get('n_mapped')} frames): "
                  f"{mm.get('summary')}", flush=True)
        except Exception as e:
            print(f"[ERROR] pressure CSV import failed: {e}", flush=True)
            return 1

    if "2" in steps:
        run_peak_fitting(
            analysis_path, None,
            source=args.source, sensitivity=args.sensitivity,
            auto_range=not args.no_auto_range, hybrid_spike_bins=args.hybrid_spike_bins,
            min_snr=args.min_snr, min_prominence_snr=args.min_prominence_snr,
            window_factor=args.window_factor,
            max_chi2=args.max_chi2, edge_bins=args.edge_bins,
            fit_min=args.fit_min, fit_max=args.fit_max,
            min_fwhm_bins=args.min_fwhm_bins,
            local_baseline_bins=args.detrend_bins,
            propagate_seeds=not args.no_seeds,
            num_workers=args.workers)

    if "3" in steps:
        from .phases import load_library, pymatgen_available
        if not pymatgen_available():
            print("[ERROR] Step 3a needs pymatgen (pip install pymatgen).", flush=True)
            return 1
        lib = load_library(args.workspace or Path.cwd())
        phases = None

        # ML candidate ranking (Step 3b proposer): rank the WHOLE library, verify
        # only the top-K below ("ML proposes, physics verifies"). Candidate-FREE —
        # no --phases needed.
        if args.ml_rank:
            from .ml_rank import rank_candidates
            if not list(lib.values()):
                print("[ERROR] reference library is empty — add or bundle phases first.", flush=True)
                return 1
            try:
                mrank = rank_candidates(analysis_path, list(lib.values()),
                                        source=args.ml_rank_source,
                                        top_k=args.ml_rank_top_k,
                                        scorer=(args.ml_scorer or None))
            except (RuntimeError, ValueError) as e:
                print(f"[ERROR] ML ranking failed: {e}", flush=True)
                return 1
            phases = [lib[n] for n in mrank["candidates"] if n in lib]
            if phases:
                print(f"[ANALYZE] ML ranker shortlist: {[p.name for p in phases]}", flush=True)
            else:
                print("[WARN] ML ranker produced no candidates — needs --phases.", flush=True)
                phases = None

        if phases is None:
            names = [s.strip() for s in (args.phases or "").split(",") if s.strip()]
            if not names:
                print("[ERROR] Step 3a needs --phases (comma-separated names), or --ml-rank "
                      "to rank the whole library.", flush=True)
                return 1
            phases = [lib[n] for n in names if n in lib]
            missing = [n for n in names if n not in lib]
            if missing:
                print(f"[WARN] phases not in library, skipped: {missing}", flush=True)
            if not phases:
                print("[ERROR] none of the requested phases resolve in the library.", flush=True)
                return 1
        run_identification(
            analysis_path, phases,
            wavelength=args.wavelength, p_min=args.p_min, p_max=args.p_max,
            rel_tol=args.rel_tol, num_workers=args.workers,
            use_frame_pressure=not args.no_pressure_prior,
            pressure_window=args.pressure_window,
            pressure_sigma_k=args.pressure_sigma_k,
            min_matched=args.min_matched,
            marker_prior=args.marker_prior,
            intensity_k=args.intensity_k,
            use_frame_temperature=not args.no_temperature)

        # Remove identified phases and re-fit the residual (Step 3a removal), so a
        # headless run produces the same /residual the GUI/worker does.
        from .residual import run_residual
        mres = run_residual(
            analysis_path, phases,
            seen_conf=args.seen_conf, rel_tol=args.rel_tol, min_snr=(args.min_snr or 5.0),
            min_matched=args.min_matched, allow_sparse=args.allow_sparse)

        # Step 3c: coherent-track clustering of whatever the knowns didn't explain.
        if not args.no_unknowns and mres.get("n_residual_peaks"):
            from .unknowns import run_unknowns
            run_unknowns(analysis_path)

        # Optional semi-quantitative phase fractions from the attribution.
        if args.fractions:
            from .fractions import run_fractions
            mfrac = run_fractions(analysis_path)
            if mfrac.get("ok"):
                print(f"[ANALYZE] fractions -> /fractions "
                      f"({len(mfrac.get('phases', []))} phase(s))", flush=True)
            else:
                print(f"[WARN] fractions skipped: {mfrac.get('error')}", flush=True)

    if args.ml_export:
        from .mldata import export_ml_dataset
        chans = tuple(c.strip() for c in args.ml_channels.split(",") if c.strip())
        export_ml_dataset(analysis_path, args.ml_export, channels=chans)

    print(f"[ANALYZE] done -> {analysis_path}", flush=True)
    return 0


def main(argv: "list[str] | None" = None) -> int:
    from ..core.config import make_stdio_robust
    make_stdio_robust()   # tolerate non-ASCII log lines on a cp1252 console
    p = argparse.ArgumentParser(
        prog="bulkxrd-analyze",
        description="Headless batch analysis (background -> peaks -> EOS phase matching).")
    p.add_argument("reduced", help="Path to a reduced_*.h5 (output of the reduce stage).")
    p.add_argument("-o", "--out", default="",
                   help="Output analysis .h5 (default: <reduced_stem>_analysis.h5).")
    p.add_argument("--steps", default="123",
                   help="Which steps to run, e.g. '12' or '3' (default: 123).")
    p.add_argument("--workers", type=int, default=0,
                   help="Worker processes (0=auto=CPUs-1, 1=serial). Default 0.")
    # Step 1
    p.add_argument("--max-half-window", type=int, default=40)
    p.add_argument("--n-passes", type=int, default=1)
    p.add_argument("--no-lls", action="store_true", help="Disable the LLS transform.")
    p.add_argument("--contamination-threshold", type=float, default=None)
    # Step 2
    p.add_argument("--source", default="auto",
                   choices=["auto", "hybrid", "sigmaclip", "clean", "mean"],
                   help="Peak-fit source. auto = reduce-side sigmaclip if present, else hybrid. "
                        "clean (azimuthal median) is conservative; hybrid/sigmaclip keep "
                        "spotty/textured-ring peaks. Default auto.")
    p.add_argument("--sensitivity", default="normal",
                   choices=["conservative", "normal", "sensitive"],
                   help="Detection-knob preset (fills any knob not explicitly set). Default normal.")
    p.add_argument("--no-auto-range", action="store_true",
                   help="Disable automatic valid-range inference; fit the full pattern.")
    p.add_argument("--hybrid-spike-bins", type=int, default=5,
                   help="Hybrid source: radial width (bins) below which mean-excess is a "
                        "diamond spike and removed. Default 5.")
    p.add_argument("--min-snr", type=float, default=None,
                   help="Override the sensitivity preset's min SNR (height). Default: preset.")
    p.add_argument("--min-prominence-snr", type=float, default=None,
                   help="Override the preset's min prominence SNR (controls whether a "
                        "shoulder on a stronger peak counts). Default: preset.")
    p.add_argument("--window-factor", type=float, default=3.0)
    p.add_argument("--max-chi2", type=float, default=25.0)
    p.add_argument("--edge-bins", type=int, default=None,
                   help="Drop peaks within this many bins of either pattern end. "
                        "Default: preset.")
    p.add_argument("--fit-min", type=float, default=None,
                   help="Lower fit bound (q or 2θ). Default: auto-inferred range "
                        "(full pattern with --no-auto-range).")
    p.add_argument("--fit-max", type=float, default=None,
                   help="Upper fit bound (q or 2θ). Default: auto-inferred range.")
    p.add_argument("--min-fwhm-bins", type=float, default=None,
                   help="Reject peaks narrower than this many bins. Default: preset.")
    p.add_argument("--detrend-bins", type=int, default=81,
                   help="Detection-only local-baseline window (bins); 0 = off. "
                        "Default 81, same as the GUI.")
    p.add_argument("--no-seeds", action="store_true", help="Disable seed propagation.")
    # Step 3a
    p.add_argument("--phases", default="", help="Comma-separated candidate phase names.")
    p.add_argument("--workspace", default="", help="Workspace holding the user phase library.")
    p.add_argument("--wavelength", type=float, default=None, help="Å (2θ data only).")
    p.add_argument("--p-min", type=float, default=0.0)
    p.add_argument("--p-max", type=float, default=100.0)
    p.add_argument("--rel-tol", type=float, default=0.01)
    p.add_argument("--seen-conf", type=float, default=0.5,
                   help="Confidence bar for 'phase present in frame' (and residual removal).")
    # Frame-pressure prior + evidence (the DAC accuracy controls).
    p.add_argument("--pressure-csv", default="",
                   help="CSV (frame|filename, pressure_gpa[, pressure_sigma_gpa, "
                        "temperature_K]) imported onto /frames before Step 3 (merges).")
    p.add_argument("--no-pressure-prior", action="store_true",
                   help="Ignore /frames/pressure; use the full p_min..p_max free search.")
    p.add_argument("--pressure-window", type=float, default=2.0,
                   help="GPa half-window for the prior where no per-frame sigma is known.")
    p.add_argument("--pressure-sigma-k", type=float, default=2.0,
                   help="Window half-width = k·sigma where a pressure_sigma is present.")
    p.add_argument("--marker-prior", action="store_true",
                   help="With no metadata pressure, estimate it from marker phases first.")
    p.add_argument("--min-matched", type=int, default=3,
                   help="Min one-to-one matched reflections to call a phase present. Default 3.")
    p.add_argument("--intensity-k", type=float, default=0.3,
                   help="Weight of the soft intensity-agreement factor in the confidence "
                        "(0 = position-only; DAC texture makes intensities unreliable, so "
                        "keep it gentle). Default 0.3.")
    p.add_argument("--no-temperature", action="store_true",
                   help="Ignore /frames/temperature (skip the thermal-expansion seam).")
    p.add_argument("--allow-sparse", action="store_true",
                   help="Permit phases below --min-matched to be subtracted in the residual.")
    p.add_argument("--no-unknowns", action="store_true",
                   help="Skip Step 3c (co-occurrence clustering of residual peaks).")
    p.add_argument("--fractions", action="store_true",
                   help="After the residual, write /fractions: per-frame "
                        "semi-quantitative intensity-share phase fractions "
                        "(see analysis/fractions.py for the caveats).")
    # Step 3b proposer
    p.add_argument("--ml-rank", action="store_true",
                   help="Rank the whole library per frame (deterministic cosine vs simulated "
                        "pattern at the frame pressure) and verify only the top-K with Step 3a.")
    p.add_argument("--ml-rank-top-k", type=int, default=5,
                   help="How many ranked candidates per frame to verify. Default 5.")
    p.add_argument("--ml-rank-source", default="auto",
                   help="What to rank against: auto|residual|fit. Default auto.")
    p.add_argument("--ml-scorer", default="",
                   help="Similarity scorer for --ml-rank: 'cosine' (default) or "
                        "'torch:<model.pt>' (a trained bulkxrd-ml-train export; "
                        "needs bulkxrd[ml]). Whatever it proposes, Step 3a verifies.")
    # ML export
    p.add_argument("--ml-export", default="", help="Also export an ML .npz to this path.")
    p.add_argument("--ml-channels", default="fit,spot_residual",
                   help="Comma-separated channels for the ML export: any of "
                        "fit/residual/clean/hybrid/robust/mean/sigmaclip/baseline/"
                        "spot_residual. 'fit' = the channel Step 2 actually fit "
                        "(recommended). Default fit,spot_residual.")
    args = p.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
