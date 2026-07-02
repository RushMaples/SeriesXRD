"""Reduction session-config seeding (dependency-light: stdlib + core only).

Replaces the old notebook ``configure_session`` helper now that the unified
app is the entry point. Safe to import without numpy/pyFAI/Tk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..core.config import (
    read_json, write_json, print_status, default_workspace_paths,
    default_python_exe, load_session_config,
)
from ..core.handoff import find_latest_handoff

CONFIG_FILENAME = "reduction_session_config.json"

_DEFAULTS = {
    "session_name": "reduction",
    "file_patterns": "*.tif;*.tiff;*.edf;*.cbf;*.mar3450;*.h5",
    "recursive": False,
    "npt_1d": "",                # blank = auto: ~1 bin per pixel of radial extent
                                 # (pyFAI rule of thumb, from the accepted PONI +
                                 # first frame). A fixed low value under-samples
                                 # sharp peaks -> stepped patterns, poor fits.
    "unit": "q_A^-1",            # the pipeline's design decision: fit in q, not 2θ
                                 # (peak widths ~constant in q -> uniform windows;
                                 # d-conversion needs no wavelength downstream).
                                 # 2th_deg remains selectable for Dioptas parity.
    "method": "csr",
    "polarization_factor": "",
    "azimuth_range": "",         # optional 'min,max' (deg) sector for ALL 1D
                                 # channels (stopgap for wavy rings; cakes stay
                                 # full-azimuth). Blank = full azimuth.
    "robust_1d": True,
    "robust_quant_halfwidth": "0.05",  # robust channel = mean of the 45-55% azimuthal
                                       # quantile band. 0 = pure median, which is
                                       # QUANTIZED on integer counts (staircase-looking
                                       # patterns at low intensity).
    "sigmaclip_1d": True,        # azimuthal sigma-clipped (trimmed-mean) 1D channel —
                                 # the less-lossy fit source for spotty/textured rings
    "sigmaclip_thresh": "3.0",   # sigma threshold for azimuthal outlier rejection
    "sigmaclip_maxiter": "5",
    "save_cakes": False,
    "npt_radial": "500",
    "npt_azimuthal": "360",
    "cake_every": "1",
    "num_workers": "0",
    "handoff_file": "",
    "dataset_dir": "",
}


def reduction_config_path(workspace_dir: "str | Path") -> Path:
    return Path(workspace_dir).expanduser().resolve() / CONFIG_FILENAME


def seed_reduction_config(workspace_dir: "str | Path") -> Path:
    """Create/refresh the reduction config in a workspace and return its path.

    Pre-fills shared paths from the workspace's calibration config (if present)
    and auto-locates the newest calibration handoff under the accepted-output
    tree. Existing user-set keys are never overwritten.
    """
    ws = Path(workspace_dir).expanduser().resolve()
    cfg_path = reduction_config_path(ws)
    cfg = read_json(cfg_path)

    calib_cfg = load_session_config(ws)  # falls back to defaults if absent
    paths = default_workspace_paths(ws)
    seed = dict(_DEFAULTS)
    seed.update({
        "workspace_root": str(ws),
        "backend_dir": calib_cfg.backend_dir or str(Path(__file__).resolve().parents[1]),
        "python_exe": calib_cfg.python_exe or default_python_exe(),
        "processed_root": calib_cfg.processed_root or paths["processed_root"],
        "logs_root": calib_cfg.logs_root or paths["logs_root"],
    })
    for k, v in seed.items():
        cfg.setdefault(k, v)

    if not cfg.get("handoff_file"):
        search_root = calib_cfg.accepted_output_root or paths["accepted_output_root"]
        latest = find_latest_handoff(search_root)
        if latest:
            cfg["handoff_file"] = str(latest)
            print_status(f"Auto-found latest calibration handoff: {latest}")

    cfg["session_config_path"] = str(cfg_path)
    write_json(cfg_path, cfg)
    return cfg_path
