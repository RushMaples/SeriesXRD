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
    # Step 2 — pseudo-Voigt peak fitting
    "min_snr": "5.0",
    "window_factor": "3.0",
    "max_chi2": "25.0",
    "propagate_seeds": True,
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
