"""Analysis session-config seeding (dependency-light: stdlib + core only).

Mirrors reduce/session.py. Creates/refreshes ``analysis_session_config.json``
in a workspace and pre-fills the input reduced-HDF5 path from the workspace's
reduction config (its ``reduced_h5_file``) when present. Safe to import without
numpy/h5py/Tk.
"""
from __future__ import annotations

from pathlib import Path

from ..core.config import (
    read_json, write_json, print_status, default_workspace_paths,
    default_python_exe,
)

CONFIG_FILENAME = "analysis_session_config.json"
_REDUCTION_CONFIG_FILENAME = "reduction_session_config.json"

_DEFAULTS = {
    "session_name": "analysis",
    # Input / output
    "reduced_h5_file": "",        # reduced_*.h5 with intensity + intensity_robust
    "analysis_h5_file": "",       # output; blank -> <reduced_stem>_analysis.h5 beside input
    # Run scope
    "run_step1": True,            # background separation
    "run_step2": True,            # peak fitting
    # Step 1 — background separation (SNIP + spot residual)
    "max_half_window": "40",
    "n_passes": "1",
    "use_lls": True,
    "contamination_threshold": "",  # optional float; blank = don't flag
    # Step 2 — pseudo-Voigt peak fitting.
    # Source + Sensitivity are the primary controls; the individual knobs below
    # are advanced overrides (blank = follow the sensitivity preset).
    "peak_source": "auto",        # auto|clean|hybrid|mean|sigmaclip. auto = the reduce-side
                                  # sigmaclip channel if present, else the analysis-side hybrid.
                                  # clean (azimuthal median) is conservative but drops real
                                  # spotty/textured/incomplete-ring peaks; hybrid/sigmaclip keep them.
    "sensitivity": "normal",      # conservative|normal|sensitive — sets min_snr / min_prominence_snr
                                  # / min_fwhm_bins / edge_bins for any left blank below.
    "auto_range": True,           # blank fit_min/fit_max -> inferred valid q/2θ range
                                  # (conservative: trims only the beamstop ramp + dead tail).
    "hybrid_spike_bins": "5",     # hybrid source: radial width (bins) below which mean-excess
                                  # is treated as a diamond spike and removed; broader = real texture, kept.
    "min_snr": "",                # blank = from sensitivity preset
    "min_prominence_snr": "",     # blank = from sensitivity preset
    "window_factor": "3.0",
    "max_chi2": "25.0",
    "edge_bins": "",              # blank = from sensitivity preset
    "fit_min": "",               # optional radial-unit (2θ or q) lower fit bound; blank = auto
    "fit_max": "",               # optional upper fit bound; blank = auto
    "min_fwhm_bins": "",          # blank = from sensitivity preset
    "detrend_bins": "81",        # local-baseline window (bins) for detection; 0 = off.
                                 # The proven fix: removes residual broad background so the
                                 # noise floor reflects real noise and small peaks aren't
                                 # lost under an inflated global threshold.
    "propagate_seeds": True,
    # Step 3 prep — candidate phases (names from the reference-phase library)
    # enabled for compound identification. Edited on the GUI's Phases tab.
    "candidate_phases": [],
    # Step 3a — deterministic EOS phase matching
    "run_step3": False,
    "identify_all_phases": False,  # open-set: score the whole library, ignore the
                                   # candidate selection (identify without pre-marking).
    "p_min": "0",
    "p_max": "100",
    "rel_tol": "0.01",
    "seen_conf": "0.5",         # confidence bar for "phase present in frame" — also
                                # which phases get removed in the residual step.
    "identify_wavelength": "",  # Å; blank = auto-read from reduced PONI (2θ data only)
    # Frame metadata (pressure prior). pressure is auto-parsed from filenames at
    # Step 1; a CSV here overrides it before Step 3 (frame|filename, pressure_gpa
    # [, pressure_sigma_gpa, temperature_K]). The CSV merges (only the frames it
    # provides). A failed import is fatal unless pressure_csv_required is False.
    "pressure_csv": "",
    "pressure_csv_required": True,
    # Step 3a pressure prior + evidence — the DAC accuracy controls.
    "use_pressure_prior": True,    # confine each phase's fit to the frame's pressure ± window
                                   # (turns pressure from a free per-phase parameter into a prior).
    "pressure_window": "2.0",      # GPa half-window used where no per-frame sigma is known
    "pressure_sigma_k": "2.0",     # window = k·sigma where a pressure_sigma is present
    "marker_prior": False,         # no metadata pressure -> estimate it from marker phases first,
                                   # then reuse that as the prior for all other phases.
    "min_matched": "3",            # min one-to-one matched reflections to call a phase "present"
    "allow_sparse": False,         # permit marker/sparse phases below min_matched in the residual
    "intensity_k": "0.3",          # weight of the intensity-agreement factor (0 = positions only)
    "use_frame_temperature": True, # apply /frames/temperature to predicted d's (thermal seam)
    # Step 3b proposer: ML candidate ranking. Ranks the whole library against each
    # frame (deterministic cosine vs simulated pattern at the frame pressure) and
    # verifies only the top-K with Step 3a — "ML proposes, physics verifies".
    "run_ml_rank": False,
    "ml_rank_top_k": "5",
    "ml_rank_source": "auto",      # auto|residual|fit — what to rank against
    "ml_scorer": "",               # ''/'cosine' = deterministic; 'torch:<model.pt>' = trained
    # Grid map (view-only): scan geometry for mapping runs.
    "map_value": "total",          # per-frame scalar shown on the grid
    "map_layout": "scan lines",    # scan lines (order-based) | coordinates (pos_x/pos_y)
    "map_line_len": "",            # frames per scan line (user's raster width/height)
    "map_order": "horizontal",     # horizontal rows | vertical columns
    "map_serpentine": True,        # boustrophedon vs unidirectional raster
    "map_roi_min": "",
    "map_roi_max": "",
    # Stage-position header keys (mapping runs; Frame meta tab dialog).
    "pos_header_x": "",
    "pos_header_y": "",
    "pos_header_dir": "",
    # Parallelism: 0 = auto (CPU count − 1), 1 = serial, N = N processes.
    "num_workers": "0",
}


def analysis_config_path(workspace_dir: "str | Path") -> Path:
    return Path(workspace_dir).expanduser().resolve() / CONFIG_FILENAME


def seed_analysis_config(workspace_dir: "str | Path") -> Path:
    """Create/refresh the analysis config in a workspace and return its path.

    Pre-fills shared paths from the workspace and auto-locates the reduced HDF5
    produced by the reduction stage (the ``reduced_h5_file`` key in the
    workspace's reduction config). Existing user-set keys are never overwritten.
    """
    ws = Path(workspace_dir).expanduser().resolve()
    cfg_path = analysis_config_path(ws)
    cfg = read_json(cfg_path)

    paths = default_workspace_paths(ws)
    seed = dict(_DEFAULTS)
    seed.update({
        "workspace_root": str(ws),
        "backend_dir": str(Path(__file__).resolve().parents[1]),
        "python_exe": default_python_exe(),
        "logs_root": paths["logs_root"],
    })
    for k, v in seed.items():
        cfg.setdefault(k, v)

    # Hand off the reduced file from the reduction config if we don't have one.
    if not cfg.get("reduced_h5_file"):
        reduce_cfg = read_json(ws / _REDUCTION_CONFIG_FILENAME)
        reduced = reduce_cfg.get("reduced_h5_file", "")
        if reduced and Path(reduced).expanduser().is_file():
            cfg["reduced_h5_file"] = reduced
            print_status(f"Auto-found reduced HDF5 from reduction config: {reduced}")

    cfg["session_config_path"] = str(cfg_path)
    write_json(cfg_path, cfg)
    return cfg_path
