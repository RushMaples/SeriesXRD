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

    if "2" in steps:
        run_peak_fitting(
            analysis_path, None,
            min_snr=args.min_snr, window_factor=args.window_factor,
            max_chi2=args.max_chi2, propagate_seeds=not args.no_seeds,
            num_workers=args.workers)

    if "3" in steps:
        from .phases import load_library, pymatgen_available
        if not pymatgen_available():
            print("[ERROR] Step 3a needs pymatgen (pip install pymatgen).", flush=True)
            return 1
        names = [s.strip() for s in (args.phases or "").split(",") if s.strip()]
        if not names:
            print("[ERROR] Step 3a needs --phases (comma-separated names from the "
                  "reference library).", flush=True)
            return 1
        lib = load_library(args.workspace or Path.cwd())
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
            rel_tol=args.rel_tol, num_workers=args.workers)

    if args.ml_export:
        from .mldata import export_ml_dataset
        chans = tuple(c.strip() for c in args.ml_channels.split(",") if c.strip())
        export_ml_dataset(analysis_path, args.ml_export, channels=chans)

    print(f"[ANALYZE] done -> {analysis_path}", flush=True)
    return 0


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(
        prog="bulkxrd-analyze",
        description="Headless batch analysis (background → peaks → EOS phase matching).")
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
    p.add_argument("--min-snr", type=float, default=5.0)
    p.add_argument("--window-factor", type=float, default=3.0)
    p.add_argument("--max-chi2", type=float, default=25.0)
    p.add_argument("--no-seeds", action="store_true", help="Disable seed propagation.")
    # Step 3a
    p.add_argument("--phases", default="", help="Comma-separated candidate phase names.")
    p.add_argument("--workspace", default="", help="Workspace holding the user phase library.")
    p.add_argument("--wavelength", type=float, default=None, help="Å (2θ data only).")
    p.add_argument("--p-min", type=float, default=0.0)
    p.add_argument("--p-max", type=float, default=100.0)
    p.add_argument("--rel-tol", type=float, default=0.01)
    # ML export
    p.add_argument("--ml-export", default="", help="Also export an ML .npz to this path.")
    p.add_argument("--ml-channels", default="clean,spot_residual",
                   help="Comma-separated channels for the ML export.")
    args = p.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
