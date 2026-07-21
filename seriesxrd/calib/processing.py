"""pyFAI generation/export backend.

The module imports heavy scientific packages lazily so notebook preflight can
show meaningful errors instead of failing during import.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import math
import os
import re
import sys
import numpy as np

from ..core.config import VERSION, ensure_dir, write_json, sha256_file, now_iso, now_timestamp, safe_stem, copy_file, output_base
from ..core.io import read_detector_image, write_xy_csv, write_table_csv
from ..core.masks import automatic_mask, save_mask_npz, save_mask_preview_png, load_mask_npz
from ..core.naming import generation_paths, gen_label, next_available_path


def _ensure_conda_dlls() -> None:
    """Prepend conda Library/bin to PATH on Windows so pyFAI C extensions resolve."""
    if not sys.platform.startswith("win"):
        return
    prefix = Path(sys.executable).parent
    for subdir in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin"):
        dll_dir = prefix / subdir
        if dll_dir.is_dir():
            dll_str = str(dll_dir)
            if dll_str.lower() not in os.environ.get("PATH", "").lower():
                os.environ["PATH"] = dll_str + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# Global matplotlib configuration (applied once per process)
# ---------------------------------------------------------------------------
_mpl_configured = [False]

# Mutable module-level figure DPI; generate_qa_run lowers it for Fast QA.
_FIG_DPI = [180]

def _configure_mpl() -> None:
    if _mpl_configured[0]:
        return
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
    })
    _mpl_configured[0] = True

# Dark palette for line plots — shared with the GUIs (guikit.theme)
from ..guikit.theme import (
    BG as _DARK_BG, BG2 as _DARK_AX, FG as _DARK_FG,
    CLR_RAW as _CLR_RAW, CLR_MSKD as _CLR_MSKD,
    CLR_DIFF as _CLR_DIFF, CLR_SMTH as _CLR_SMTH,
)


def _apply_dark_axes(fig, *axes) -> None:
    fig.patch.set_facecolor(_DARK_BG)
    for ax in axes:
        ax.set_facecolor(_DARK_AX)
        ax.tick_params(colors=_DARK_FG, which="both")
        ax.xaxis.label.set_color(_DARK_FG)
        ax.yaxis.label.set_color(_DARK_FG)
        ax.title.set_color(_DARK_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_DARK_FG)
        ax.xaxis.set_tick_params(labelcolor=_DARK_FG)
        ax.yaxis.set_tick_params(labelcolor=_DARK_FG)


# ---------------------------------------------------------------------------
# pyFAI helpers
# ---------------------------------------------------------------------------

def load_pyfai_integrator(poni_file: str):
    import pyFAI  # type: ignore
    return pyFAI.load(str(poni_file))


def read_poni_info(poni_file: "str | Path") -> Dict[str, Any]:
    """Lightweight text parse of a .poni file — no pyFAI import.

    Returns detector name, detector shape (rows, cols) from Detector_config
    max_shape, and wavelength in metres; None for anything missing. Safe to
    call from the Tk GUI process for pre-flight compatibility checks (a pyFAI
    DLL crash there would take down the whole GUI).
    """
    info: Dict[str, Any] = {
        "detector": None, "shape": None, "wavelength_m": None,
        "pixel1": None, "pixel2": None, "dist": None, "poni1": None, "poni2": None,
        "rot1": None, "rot2": None, "rot3": None,
    }
    try:
        text = Path(poni_file).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return info

    def _float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key == "detector":
            info["detector"] = value
        elif key == "wavelength":
            info["wavelength_m"] = _float(value)
        elif key in ("distance", "dist"):
            info["dist"] = _float(value)
        elif key == "poni1":
            info["poni1"] = _float(value)
        elif key == "poni2":
            info["poni2"] = _float(value)
        elif key == "rot1":
            info["rot1"] = _float(value)
        elif key == "rot2":
            info["rot2"] = _float(value)
        elif key == "rot3":
            info["rot3"] = _float(value)
        elif key in ("pixelsize1", "pixel1"):   # legacy poni v1 keys
            info["pixel1"] = _float(value)
        elif key in ("pixelsize2", "pixel2"):
            info["pixel2"] = _float(value)
        elif key == "detector_config":
            try:
                cfg = json.loads(value)
                shape = cfg.get("max_shape")
                if isinstance(shape, (list, tuple)) and len(shape) == 2:
                    info["shape"] = (int(shape[0]), int(shape[1]))
                _p1 = _float(cfg.get("pixel1"))
                if _p1 is not None:
                    info["pixel1"] = _p1
                _p2 = _float(cfg.get("pixel2"))
                if _p2 is not None:
                    info["pixel2"] = _p2
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                pass
    return info


def suggest_integration_settings(image_shape, poni_info: Dict[str, Any]) -> Dict[str, int]:
    """Bin counts derived from the detector geometry instead of fixed factory values.

    npt_1d follows the pyFAI rule of thumb of ~one bin per pixel of maximum
    radial extent (beam centre to the farthest image corner, in pixels), so
    peak resolution matches what the detector actually recorded. The cake
    radial axis uses half of that (it feeds QA figures, not analysis data)
    and the azimuth uses 1-degree bins; both are capped to keep 2D memory
    bounded. Falls back to a centred-beam half-diagonal when the PONI lacks
    geometry fields.
    """
    nrows, ncols = int(image_shape[0]), int(image_shape[1])
    r_max_px = None
    p1, p2 = poni_info.get("pixel1"), poni_info.get("pixel2")
    c1, c2 = poni_info.get("poni1"), poni_info.get("poni2")
    if p1 and p2 and c1 is not None and c2 is not None:
        beam_row, beam_col = c1 / p1, c2 / p2
        corners = [(0, 0), (0, ncols - 1), (nrows - 1, 0), (nrows - 1, ncols - 1)]
        r_max_px = max(math.hypot(r - beam_row, c - beam_col) for r, c in corners)
    if not r_max_px:
        r_max_px = math.hypot(nrows, ncols) / 2.0

    def _round_up(v: float, step: int = 50) -> int:
        return int(math.ceil(v / step) * step)

    return {
        "npt_1d":        min(max(_round_up(r_max_px), 500), 4000),
        "npt_radial":    min(max(_round_up(r_max_px / 2), 300), 1000),
        "npt_azimuthal": 360,
        "r_max_px":      int(round(r_max_px)),
    }


def _method_value(method: str) -> str:
    return method or "csr"


def _safe_percentile(data, q):
    arr = np.asarray(data)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def _compute_poni_radial_range(ai) -> "tuple[float, float] | None":
    """Compute 2theta range from pyFAI's own geometry (tilt-aware)."""
    try:
        shape = ai.detector.shape
        if shape is None or len(shape) < 2:
            return None
        try:
            arr = ai.center_array(shape, unit="2th_deg")
        except AttributeError:
            arr = np.degrees(ai.twoThetaArray(shape))
        return (0.0, float(arr.max()) * 1.005)
    except Exception:
        return None


def unit_axis_label(unit: str) -> str:
    """Return a matplotlib-ready x-axis label for the given pyFAI unit string."""
    return {
        "2th_deg": r"2$\theta$ (deg)",
        "2th_rad": r"2$\theta$ (rad)",
        "q_A^-1":  r"q ($\AA^{-1}$)",
        "q_nm^-1": r"q (nm$^{-1}$)",
    }.get(unit, unit)


def unit_column_name(unit: str) -> str:
    """Return a CSV-safe column name for the given pyFAI unit string."""
    return {
        "2th_deg": "two_theta_deg",
        "2th_rad": "two_theta_rad",
        "q_A^-1":  "q_per_angstrom",
        "q_nm^-1": "q_per_nm",
    }.get(unit, re.sub(r"[^A-Za-z0-9_]", "_", unit))


def runtime_versions() -> Dict[str, str]:
    """Return a dict of key library/runtime versions for provenance recording."""
    versions: Dict[str, str] = {
        "seriesxrd":  VERSION,
        "numpy":    np.__version__,
        "python":   sys.version.split()[0],
        "platform": sys.platform,
    }
    try:
        import pyFAI  # type: ignore
        versions["pyFAI"] = pyFAI.version
    except Exception:
        versions["pyFAI"] = "unavailable"
    return versions


def _get_calibrant_tth(
    calibrant_name: str, wavelength_m: float,
    radial_range: "tuple | None" = None,
    unit: str = "2th_deg",
) -> "List[Tuple[float, str]]":
    """Return list of (position_in_unit, label) for calibrant peaks using pyFAI database."""
    try:
        import pyFAI.calibrant as pc  # type: ignore
        cal = pc.get_calibrant(calibrant_name)
        cal.wavelength = float(wavelength_m)
        peaks = []
        dspacing_attr = getattr(cal, "dspacing", None) or getattr(cal, "dSpacing", [])
        for dsp in dspacing_attr:
            if dsp <= 0:
                continue
            sin_th = wavelength_m / (2.0 * dsp)
            if abs(sin_th) > 1.0:
                continue
            tth_deg = math.degrees(2.0 * math.asin(sin_th))
            # Convert from 2theta_deg to the active unit.
            if unit == "2th_deg":
                pos = tth_deg
            elif unit == "2th_rad":
                pos = math.radians(tth_deg)
            elif unit == "q_A^-1":
                # q = 4*pi/lambda * sin(theta);  lambda in Angstrom
                pos = (4.0 * math.pi / (wavelength_m * 1e10)) * math.sin(math.radians(tth_deg / 2.0))
            elif unit == "q_nm^-1":
                pos = (4.0 * math.pi / (wavelength_m * 1e9)) * math.sin(math.radians(tth_deg / 2.0))
            else:
                pos = tth_deg
            if radial_range is not None and not (radial_range[0] <= pos <= radial_range[1]):
                continue
            peaks.append((pos, calibrant_name))
        return sorted(peaks)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _save_image_png(
    path: Path, data, title: str = "", cmap: str = "magma",
    mask=None, vmin=None, vmax=None,
) -> Path:
    _configure_mpl()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=_FIG_DPI[0])
    arr = np.asarray(data)
    if vmin is None:
        vmin = _safe_percentile(arr, 1)
    if vmax is None:
        vmax = _safe_percentile(arr, 99.5)
        if vmax <= vmin:
            vmax = None
    im = ax.imshow(arr, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        overlay = np.zeros((*m.shape, 4), dtype=float)
        overlay[..., 0] = 1.0
        overlay[..., 3] = m.astype(float) * 0.35
        ax.imshow(overlay, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Detector x pixel")
    ax.set_ylabel("Detector y pixel")
    fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def _write_mask_only_png(path: Path, mask) -> Path:
    _configure_mpl()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    p = Path(path)
    fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=_FIG_DPI[0])
    ax.imshow(np.asarray(mask, dtype=bool), cmap="gray", origin="upper", vmin=0, vmax=1)
    ax.set_title("Mask used for integration")
    ax.set_xlabel("Detector x pixel")
    ax.set_ylabel("Detector y pixel")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def _make_intensity_difference_plot(
    path: Path, tth, intensity, raw_like=None,
    title: str = "Intensity and difference",
    calibrant_tth: "List | None" = None,
    x_label: str = r"2$\theta$ (deg)",
) -> Path:
    _configure_mpl()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    p = Path(path)
    tth = np.asarray(tth)
    intensity = np.asarray(intensity, dtype=float)
    if raw_like is None:
        win = max(5, int(len(intensity) / 80))
        kernel = np.ones(win) / win
        smooth = np.convolve(np.where(np.isfinite(intensity), intensity, 0), kernel, mode="same")
        diff = intensity - smooth
    else:
        diff = intensity - np.asarray(raw_like, dtype=float)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.8, 5.6), dpi=_FIG_DPI[0], sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    _apply_dark_axes(fig, ax1, ax2)
    ax1.plot(tth, intensity, lw=1.2, color=_CLR_MSKD, label="masked integration")
    if raw_like is not None:
        ax1.plot(tth, np.asarray(raw_like, dtype=float), lw=0.9, alpha=0.75, color=_CLR_RAW, label="raw/no-mask")
    if calibrant_tth:
        for ct, _lbl in calibrant_tth:
            ax1.axvline(ct, lw=0.7, alpha=0.55, color="#f5c2e7", linestyle="--")
    ax1.set_ylabel("Intensity")
    ax1.set_title(title)
    ax1.legend(loc="best")
    ax1.tick_params(labelbottom=False)
    ax2.plot(tth, diff, lw=1.0, color=_CLR_DIFF, label="masked - raw/smoothed")
    ax2.axhline(0, lw=0.8, alpha=0.5, color=_DARK_FG)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Difference")
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    return p


def _make_normalized_intensity_plot(
    path: Path, tth, masked, raw,
    title: str = "Normalized intensity vs 2theta",
    calibrant_tth: "List | None" = None,
    x_label: str = r"2$\theta$ (deg)",
) -> Path:
    """Independent per-trace normalization: each trace reaches 1 at its own peak."""
    _configure_mpl()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    p = Path(path)
    tth    = np.asarray(tth)
    masked = np.asarray(masked, dtype=float)
    raw    = np.asarray(raw, dtype=float)
    max_masked = np.nanmax(np.abs(masked)) or 1.0
    max_raw    = np.nanmax(np.abs(raw))    or 1.0
    masked_norm = masked / max_masked
    raw_norm    = raw    / max_raw
    eps = 1e-10
    norm_diff = np.where(
        np.isfinite(raw) & (np.abs(raw) > eps),
        (raw - masked) / raw,
        np.nan,
    )
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.8, 5.6), dpi=_FIG_DPI[0], sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    _apply_dark_axes(fig, ax1, ax2)
    ax1.plot(tth, raw_norm,    lw=0.9, alpha=0.75, color=_CLR_RAW,  label="raw (normalized)")
    ax1.plot(tth, masked_norm, lw=1.2,             color=_CLR_MSKD, label="masked (normalized)")
    if calibrant_tth:
        for ct, _lbl in calibrant_tth:
            ax1.axvline(ct, lw=0.7, alpha=0.55, color="#f5c2e7", linestyle="--")
    ax1.set_ylabel("Normalized intensity")
    ax1.set_title(title)
    ax1.legend(loc="best")
    ax1.tick_params(labelbottom=False)
    ax2.plot(tth, norm_diff, lw=1.0, color=_CLR_DIFF, label="(raw - masked) / raw")
    ax2.axhline(0, lw=0.8, alpha=0.5, color=_DARK_FG)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Norm. difference")
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    return p


def _make_cake_png(
    path: Path, cake, radial, azimuthal=None, title: str = "Cake plot",
    x_label: str = r"2$\theta$ (deg)",
    calibrant_tth: "List | None" = None,
) -> Path:
    _configure_mpl()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    p = Path(path)
    cake = np.array(cake, dtype=float)  # copy: caller's array is also saved to .npz
    cake[cake <= 0] = np.nan           # zero/dummy bins -> NaN -> dark background
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=_FIG_DPI[0])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color=_DARK_BG)
    finite_pos = cake[np.isfinite(cake)]
    if finite_pos.size > 0:
        vmin = float(np.percentile(finite_pos, 1))
        vmax = float(np.percentile(finite_pos, 99))
    else:
        vmin, vmax = None, None
    if vmax is not None and vmin is not None and vmax <= vmin:
        vmax = None
    extent = None
    if radial is not None:
        r = np.asarray(radial)
        extent = [float(r.min()), float(r.max()), -180, 180]
    im = ax.imshow(cake, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    # Calibrant rings should be perfectly straight vertical lines in cake space:
    # any waviness against these references means the geometry needs refining.
    # Only drawable when the x-axis is in data units (extent set).
    if calibrant_tth and extent is not None:
        for i, (ct, _lbl) in enumerate(calibrant_tth):
            ax.axvline(ct, lw=0.8, alpha=0.8, color="#f5c2e7", linestyle="--",
                       label="calibrant" if i == 0 else None)
        ax.legend(loc="upper right", framealpha=0.6, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Azimuth (deg)")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="Intensity")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def _make_coverage_plot(
    path: Path, radial, coverage_degrees,
    title: str = "Detector coverage",
    threshold_degrees: "float | None" = None,
    x_label: str = r"2$\theta$ (deg)",
) -> Path:
    """Detector coverage range diagnostic: raw + smoothed azimuthal coverage vs 2theta."""
    _configure_mpl()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    p = Path(path)
    radial = np.asarray(radial, dtype=float)
    cov    = np.asarray(coverage_degrees, dtype=float)
    npts   = len(cov)
    # Smooth with Savitzky-Golay (fall back to boxcar)
    try:
        from scipy.signal import savgol_filter  # type: ignore
        win = min(51, max(5, (npts // 20) | 1))
        if win % 2 == 0:
            win += 1
        smooth = savgol_filter(cov, win, 3)
    except Exception:
        win = max(5, npts // 20)
        kernel = np.ones(win) / win
        smooth = np.convolve(cov, kernel, mode="same")
    thr   = threshold_degrees if threshold_degrees is not None else 0.0
    valid = smooth >= thr
    if valid.any():
        first_valid = int(np.where(valid)[0][0])
        last_valid  = int(np.where(valid)[0][-1])
    else:
        first_valid, last_valid = 0, len(radial) - 1
    cov_start = float(radial[first_valid])
    cov_end   = float(radial[last_valid])
    auto_title = f"Detector coverage range: {cov_start:.3f}–{cov_end:.3f}°"
    fig, ax = plt.subplots(figsize=(7.8, 3.4), dpi=_FIG_DPI[0])
    _apply_dark_axes(fig, ax)
    ax.fill_between(radial, 0, np.where(valid, cov, 0), alpha=0.12, color=_CLR_RAW)
    ax.plot(radial, cov,    lw=1.0, alpha=0.45, color=_CLR_RAW,  label="raw coverage")
    ax.plot(radial, smooth, lw=1.8,             color=_CLR_SMTH, label="smoothed")
    if threshold_degrees is not None:
        ax.axhline(threshold_degrees, lw=1.0, linestyle="--", color=_CLR_DIFF, alpha=0.85,
                   label=f"threshold {threshold_degrees:.1f}°")
        ax.axvline(cov_end, lw=0.8, linestyle=":", color=_CLR_DIFF, alpha=0.7)
    ax.set_xlim(left=0, right=min(cov_end + 5.0, float(radial[-1])))
    ax.set_ylim(0, 370)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Coverage (°)")
    ax.set_title(auto_title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    return p


def _make_compilation(
    path_png: Path, path_pdf: Path, paths: Dict[str, Path],
    title: str = "Calibration QA compilation",
) -> None:
    _configure_mpl()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    from PIL import Image  # type: ignore
    slots = [
        ("Raw detector",        paths.get("raw_detector_png")),
        ("Masked detector",     paths.get("masked_detector_png")),
        ("Mask",                paths.get("mask_only_png")),
        ("Intensity + diff",    paths.get("intensity_difference_png")),
        ("Intensity (norm.)",   paths.get("intensity_normalized_png")),
        ("Cake",                paths.get("cake_png")),
        ("Coverage diagnostic", paths.get("coverage_png")),
        (None, None),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(20, 20), dpi=_FIG_DPI[0])
    fig.suptitle(title, y=0.998)
    for ax, (label, p) in zip(axes.ravel(), slots):
        ax.axis("off")
        if label is None:
            continue
        ax.set_title(label)
        if p and Path(p).exists():
            img = Image.open(p)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
    fig.tight_layout(pad=2.5, rect=[0, 0, 1, 0.997])
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

# 2D cake methods. The bare "csr" string defaults to bbox pixel splitting,
# whose 2D sparse matrix can exceed 18-21 GB RAM on large detectors — that is
# what previously forced the slow "numpy" engine here. CSR *without* pixel
# splitting ("no") has a small matrix, stays fast, and is the same engine
# family Dioptas uses for its cakes. Pixel splitting is opt-in via the
# "cake_pixel_split" config key for users who want smoother low-angle bins
# and have the RAM for it.
_CAKE_METHODS_2D = [
    ("no", "csr", "cython"),
    ("no", "histogram", "cython"),
    ("no", "histogram", "python"),
]
_CAKE_METHODS_2D_SPLIT = [
    ("bbox", "csr", "cython"),
    ("bbox", "histogram", "cython"),
] + _CAKE_METHODS_2D


def _apply_geometry_overrides(ai, config: Dict[str, Any]) -> List[str]:
    """Override the loaded PONI geometry with edited form values.

    The GUI autofills the geometry fields from the PONI, then lets the user
    tweak distance / beam centre / rotations / wavelength to refine alignment
    without rewriting the .poni file. Any non-blank field that differs from the
    loaded value is applied to the integrator; blank fields keep the PONI value.
    Detector and pixel size are intrinsic to the PONI's detector object and are
    not overridden here (they are shown for reference only).
    """
    applied: List[str] = []
    for key in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3"):
        raw = config.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        try:
            cur = float(getattr(ai, key))
        except Exception:
            cur = None
        if cur is None or abs(cur - val) > 1e-12:
            try:
                setattr(ai, key, val)
                applied.append(f"{key}={val:g}")
            except Exception as e:
                print(f"[QA] could not override {key}: {e}", flush=True)
    wl_raw = config.get("wavelength_m")
    if wl_raw is not None and str(wl_raw).strip():
        try:
            wl = float(wl_raw)
            if wl > 0 and (ai.wavelength is None or abs(float(ai.wavelength) - wl) > 1e-16):
                ai.wavelength = wl
                applied.append(f"wavelength={wl:g}")
        except (TypeError, ValueError):
            pass
    if applied:
        print("[QA] geometry overrides applied: " + ", ".join(applied), flush=True)
    return applied


def _integrate_with_pyfai(image, mask, poni_file, settings: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_conda_dlls()
    print("[QA] pyFAI load PONI", flush=True)
    ai       = load_pyfai_integrator(poni_file)
    _apply_geometry_overrides(ai, settings)
    # Blank bin counts fall back to geometry-derived suggestions, not fixed values.
    suggested = suggest_integration_settings(np.asarray(image).shape, read_poni_info(poni_file))
    npt_1d   = int(settings.get("npt_1d")        or suggested["npt_1d"])
    npt_rad  = int(settings.get("npt_radial")    or suggested["npt_radial"])
    npt_azim = int(settings.get("npt_azimuthal") or suggested["npt_azimuthal"])
    unit     = settings.get("unit",   "2th_deg") or "2th_deg"
    method   = _method_value(settings.get("method", "csr"))
    fast     = bool(settings.get("fast_qa", False))

    # Radial range: use explicit settings if valid (interpreted in the chosen
    # unit), else compute from PONI corners. The corner computation yields
    # 2theta degrees, so it only applies when unit is 2th_deg — for any other
    # unit pyFAI auto-ranges from the data.
    radial_range = None
    rmin = settings.get("radial_min")
    rmax = settings.get("radial_max")
    try:
        if str(rmin).strip() and str(rmax).strip():
            if unit == "2th_deg":
                radial_range = (float(rmin), float(rmax))
            else:
                print("[QA] radial_min/max ignored: only applies to 2th_deg unit", flush=True)
    except Exception:
        radial_range = None
    if radial_range is None and unit == "2th_deg":
        radial_range = _compute_poni_radial_range(ai)

    # 1D masked integration (exactly one call)
    print("[QA] integrate1d masked start", flush=True)
    res1      = ai.integrate1d(image, npt_1d, mask=mask, unit=unit, method=method, radial_range=radial_range)
    radial    = np.asarray(res1.radial)
    intensity = np.asarray(res1.intensity)
    print("[QA] integrate1d masked done", flush=True)

    # 1D raw integration — no mask (exactly one call)
    print("[QA] integrate1d raw start", flush=True)
    raw1          = ai.integrate1d(image, npt_1d, mask=None, unit=unit, method=method, radial_range=radial_range)
    raw_intensity = np.asarray(raw1.intensity)
    print("[QA] integrate1d raw done", flush=True)

    # 2D cake — fallback chain of low-RAM methods. correctSolidAngle stays
    # True throughout: the 1D integrations above use it, and silently
    # disabling it here would make the cake inconsistent with the 1D pattern
    # it is compared against.
    cake         = None
    cake_radial  = None
    cake_azim    = None
    cake_method_used = None
    cake_warning = ""
    if fast:
        # Fast QA: skip the (expensive) 2D cake entirely. cake stays None and
        # generate_qa_run drops the cake/coverage outputs accordingly.
        cake_method_used = "skipped (fast QA)"
        print("[QA] fast mode: skipping 2D cake integration", flush=True)
    else:
        methods_2d = _CAKE_METHODS_2D_SPLIT if settings.get("cake_pixel_split") else _CAKE_METHODS_2D

        for method2d in methods_2d:
            print(f"[QA] integrate2d cake start method={method2d}", flush=True)
            try:
                res2 = ai.integrate2d(
                    image, npt_rad, npt_azim,
                    mask=mask, unit=unit, method=method2d,
                    radial_range=radial_range, azimuth_range=(-180, 180),
                    correctSolidAngle=True,
                )
                cake         = np.asarray(res2.intensity)
                cake_radial  = np.asarray(res2.radial)
                cake_azim    = np.asarray(res2.azimuthal)
                cake_method_used = "/".join(method2d)
                print(f"[QA] integrate2d cake done method={method2d} shape={cake.shape}", flush=True)
                break
            except Exception as _e2d:
                print(f"[QA] integrate2d method={method2d} FAILED: {_e2d}", flush=True)
                cake_warning += f"method={method2d} failed: {_e2d}; "

        if cake is None:
            cake_warning = (
                "All 2D cake methods failed — using NaN placeholder. "
                "1D QA outputs are still valid. " + cake_warning
            )
            print(f"[QA] WARNING: {cake_warning}", flush=True)
            cake        = np.full((npt_azim, npt_rad), np.nan)
            r0, r1      = (radial_range[0], radial_range[1]) if radial_range else (0.0, 1.0)
            cake_radial = np.linspace(r0, r1, npt_rad)
            cake_azim   = np.linspace(-180.0, 180.0, npt_azim)
            cake_method_used = "none (all failed)"

    try:
        ai_params = dict(ai.get_config())
    except Exception:
        try:
            ai_params = dict(ai.getPyFAI())
        except Exception:
            ai_params = {}

    return {
        "radial":         radial,
        "intensity":      intensity,
        "raw_intensity":  raw_intensity,
        "difference":     intensity - raw_intensity,
        "cake":           cake,
        "cake_radial":    cake_radial,
        "cake_azimuthal": cake_azim,
        "ai_parameters":  ai_params,
        "radial_range":   radial_range,
        "cake_method":    cake_method_used,
        "cake_warning":   cake_warning,
    }


# ---------------------------------------------------------------------------
# Top-level generation function (called by calib/worker.py)
# ---------------------------------------------------------------------------

def generate_qa_run(config: Dict[str, Any], generation_index: int) -> Dict[str, Any]:
    session_name = safe_stem(config.get("session_name", "calibration"))
    fast = bool(config.get("fast_qa", False))
    _FIG_DPI[0] = 100 if fast else 180
    _image_str   = str(config.get("image_file", "") or "").strip()
    _poni_str    = str(config.get("poni_file",  "") or "").strip()
    if not _image_str:
        raise FileNotFoundError("No calibration image specified in session config (image_file is empty).")
    if not _poni_str:
        raise FileNotFoundError("No PONI file specified in session config (poni_file is empty).")
    image_file = Path(_image_str)
    poni_file  = Path(_poni_str)
    if not image_file.is_file():
        raise FileNotFoundError(f"Calibration image not found: {image_file}")
    if not poni_file.is_file():
        raise FileNotFoundError(f"PONI file not found: {poni_file}")
    processed_root = ensure_dir(config.get("processed_root") or output_base(config) / "data" / "processed")
    figures_root   = ensure_dir(config.get("figures_root")   or output_base(config) / "figures")
    metadata_root  = ensure_dir(config.get("metadata_root")  or output_base(config) / "metadata")
    workflow_name  = safe_stem(config.get("workflow_name") or f"calibration_review_{session_name}")
    base_dirs = {
        "data":     ensure_dir(processed_root / workflow_name),
        "figures":  ensure_dir(figures_root   / workflow_name),
        "metadata": ensure_dir(metadata_root  / workflow_name),
    }
    ts    = now_timestamp()
    paths = generation_paths(base_dirs, generation_index, session_name, ts)
    print("[QA] load detector image", flush=True)
    image = read_detector_image(image_file, flip_up_down = bool(config.get("dioptas_image_flip", True)))
    poni_info = read_poni_info(poni_file)
    if poni_info["shape"] and tuple(image.shape) != poni_info["shape"]:
        raise ValueError(
            f"Image shape {tuple(image.shape)} does not match PONI detector shape "
            f"{poni_info['shape']} ({poni_info['detector'] or 'unknown detector'})."
        )
    print("[QA] load/merge mask", flush=True)
    auto  = automatic_mask(
        image,
        mask_negative=bool(config.get("mask_negative",  True)),
        mask_zero=bool(config.get("mask_zero",          True)),
        mask_nonfinite=bool(config.get("mask_nonfinite", True)),
        saturated_threshold=config.get("saturated_threshold", ""),
    )
    active_mask_path = config.get("active_mask_file", "")
    if active_mask_path and Path(active_mask_path).exists():
        loaded = load_mask_npz(active_mask_path)
        if loaded.shape != auto.shape:
            raise ValueError(f"Loaded mask shape {loaded.shape} != image shape {auto.shape}")
        mask = auto | loaded
    else:
        mask = auto
    integration   = _integrate_with_pyfai(image, mask, poni_file, config)
    radial        = integration["radial"]
    intensity     = integration["intensity"]
    raw_intensity = integration["raw_intensity"]
    cake          = integration["cake"]
    cake_radial   = integration["cake_radial"]
    coverage_threshold_pct = float(config.get("coverage_threshold_pct", 10) or 10)
    threshold_degrees      = coverage_threshold_pct / 100.0 * 360.0
    if cake is not None:
        # Use actual cake shape so coverage math matches what was integrated
        npt_azim      = cake.shape[0] if cake.ndim == 2 else int(config.get("npt_azimuthal", 360) or 360)
        # Azimuthal coverage per radial bin (degrees)
        _cake_arr         = np.asarray(cake, dtype=float)
        _bins_with_signal = np.sum((_cake_arr > 0) & np.isfinite(_cake_arr), axis=0)
        coverage_degrees  = (_bins_with_signal / npt_azim) * 360.0
        # Coverage-threshold masking of 1D intensity
        cov_interp             = np.interp(radial, cake_radial, coverage_degrees)
        below_threshold        = cov_interp < threshold_degrees
        n_masked_cov           = int(np.sum(below_threshold))
        intensity_filtered     = intensity.copy().astype(float)
        intensity_filtered[below_threshold] = np.nan
        print(
            f"Coverage threshold: {threshold_degrees:.1f}° ({coverage_threshold_pct}%); "
            f"{n_masked_cov}/{len(radial)} bins below threshold set to NaN",
            flush=True,
        )
    else:
        # Fast QA: no 2D cake, so no coverage diagnostic or coverage masking.
        coverage_degrees   = None
        cov_interp         = None
        n_masked_cov       = 0
        intensity_filtered = intensity.copy().astype(float)
        print("[QA] fast mode: skipped coverage diagnostic and coverage masking", flush=True)
    # Optional calibrant reference lines
    calibrant_tth  = []
    calibrant_name = config.get("calibrant", "")
    wavelength_m   = None
    try:
        wl = config.get("wavelength_m")
        if wl:
            wavelength_m = float(wl)
    except Exception:
        pass
    # Active unit from config — used for axis labels, CSV column names, and
    # calibrant peak conversion.
    active_unit = str(config.get("unit", "2th_deg") or "2th_deg")
    x_label     = unit_axis_label(active_unit)
    radial_col  = unit_column_name(active_unit)
    if calibrant_name and wavelength_m:
        calibrant_tth = _get_calibrant_tth(
            calibrant_name, wavelength_m,
            integration.get("radial_range"),
            unit=active_unit,
        )
    masked_image = np.where(mask, np.nan, image.astype(float))
    print("[QA] save figures start", flush=True)
    # Save all figures
    _save_image_png(paths["raw_detector_png"],    image,        "Raw detector image",                cmap="magma")
    _save_image_png(paths["masked_detector_png"], masked_image, "Detector image with accepted mask", cmap="magma", mask=mask)
    _write_mask_only_png(paths["mask_only_png"], mask)
    save_mask_npz(paths["mask_npz"], mask, metadata={"generation": gen_label(generation_index), "created_at": now_iso()})
    _make_intensity_difference_plot(
        paths["intensity_difference_png"], radial, intensity_filtered, raw_intensity,
        title="Intensity vs 2θ and masked-minus-raw difference",
        calibrant_tth=calibrant_tth,
        x_label=x_label,
    )
    _make_normalized_intensity_plot(
        paths["intensity_normalized_png"], radial, intensity_filtered, raw_intensity,
        title="Normalized intensity vs 2θ  (independent normalization)",
        calibrant_tth=calibrant_tth,
        x_label=x_label,
    )
    # Cake/coverage only when a cake was produced (main's None-safety);
    # labels/columns are unit-aware and the cake gets calibrant rings.
    if cake is not None:
        _make_cake_png(paths["cake_png"], cake, cake_radial, integration.get("cake_azimuthal"),
                       title="2D cake integration", x_label=x_label, calibrant_tth=calibrant_tth)
        _make_coverage_plot(paths["coverage_png"], cake_radial, coverage_degrees,
                            threshold_degrees=threshold_degrees, x_label=x_label)
    if not fast:
        _make_compilation(paths["compilation_png"], paths["compilation_pdf"], paths,
                          title=f"Calibration QA {gen_label(generation_index)}")
    print("[QA] save figures done", flush=True)
    # Difference computed from the FILTERED intensity so all columns refer to the
    # same masked signal.  NaN propagates where intensity_filtered is NaN (e.g.
    # below the coverage threshold), keeping the master CSV internally consistent.
    difference_filtered = intensity_filtered - raw_intensity
    # Data files — unit-aware column names for all CSVs.
    write_xy_csv(paths["intensity_csv"],  radial, intensity_filtered,  radial_col, "intensity_masked")
    write_xy_csv(paths["difference_csv"], radial, difference_filtered, radial_col, "masked_minus_raw")
    if cake is not None:
        write_xy_csv(paths["coverage_csv"],   cake_radial, coverage_degrees,          radial_col, "coverage_degrees")
        np.savez_compressed(
            paths["cake_npz"],
            cake=cake, radial=cake_radial, azimuthal=integration.get("cake_azimuthal"),
        )
    rows = []
    for i, r in enumerate(radial):
        row = {
            radial_col:         float(r),
            "intensity_masked": float(intensity_filtered[i]),
            "intensity_raw":    float(raw_intensity[i]),
            "masked_minus_raw": float(difference_filtered[i]),
        }
        if cov_interp is not None:
            row["coverage_degrees"] = float(cov_interp[i])
            row["coverage_pct"]     = float(cov_interp[i]) / 360.0 * 100.0
        rows.append(row)
    write_table_csv(paths["master_csv"], rows)
    # Provenance versions (Fix 7)
    versions = runtime_versions()
    # Report
    report_lines = [
        f"{gen_label(generation_index)} calibration QA report",
        "=" * 40,
        f"Created: {now_iso()}",
        f"Image: {image_file}",
        f"Input PONI: {poni_file}",
        f"Image SHA256: {sha256_file(image_file)}",
        f"PONI SHA256: {sha256_file(poni_file)}",
        f"Mask pixels: {int(np.asarray(mask).sum())} / {mask.size}",
        f"Coverage threshold: {threshold_degrees:.1f}° ({coverage_threshold_pct}%)",
        f"Coverage-masked bins: {n_masked_cov} / {len(radial)}",
        f"Unit: {active_unit}",
        f"1D method: {config.get('method', 'csr')}",
        f"2D cake method: {integration.get('cake_method', 'unknown')}",
        f"Calibrant: {calibrant_name or 'none'} ({len(calibrant_tth)} peaks in range)",
        f"Versions: seriesxrd={versions['seriesxrd']} pyFAI={versions['pyFAI']} "
        f"numpy={versions['numpy']} python={versions['python']} platform={versions['platform']}",
        "",
        "Generated files:",
    ]
    for k, pp in paths.items():
        if isinstance(pp, Path):
            report_lines.append(f"  {k}: {pp}")
    if integration.get("cake_warning"):
        report_lines.append("")
        report_lines.append(f"CAKE WARNING: {integration['cake_warning']}")
    paths["report_txt"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    rr = integration.get("radial_range")
    metadata = {
        "tool_version":     VERSION,
        "seriesxrd_version": VERSION,
        "generation":       gen_label(generation_index),
        "generation_index": generation_index,
        "created_at":       now_iso(),
        "session_name":     session_name,
        "image_file":       str(image_file),
        "poni_file":        str(poni_file),
        "image_sha256":     sha256_file(image_file),
        "poni_sha256":      sha256_file(poni_file),
        "config":           config,
        "paths":            {k: str(v) for k, v in paths.items()},
        "ai_parameters":    integration.get("ai_parameters", {}),
        "radial_range":     list(rr) if rr else None,
        "cake_method":      integration.get("cake_method"),
        "cake_warning":     integration.get("cake_warning", ""),
        "versions":         versions,
    }
    write_json(paths["metadata_json"], metadata)
    return metadata


# ---------------------------------------------------------------------------
# Final accepted-generation export
# ---------------------------------------------------------------------------

def export_accepted_generation(
    config: Dict[str, Any],
    generation_metadata: Dict[str, Any],
    output_folder: "str | Path",
    folder_name: "str | None" = None,
    selected_keys: "list | None" = None,
) -> Dict[str, Any]:
    output_root = ensure_dir(output_folder)
    session_name = safe_stem(config.get("session_name", "calibration"))
    folder_base  = output_root / safe_stem(folder_name or f"accepted_calibration_{session_name}_{now_timestamp()}")
    folder       = next_available_path(folder_base, is_dir=True)
    print(f"[export] Writing accepted calibration to: {folder}", flush=True)
    sub = {
        "source_raw":           ensure_dir(folder / "source_raw"),
        "accepted_calibration": ensure_dir(folder / "accepted_calibration"),
        "data":                 ensure_dir(folder / "data"),
        "figures":              ensure_dir(folder / "figures"),
        "metadata":             ensure_dir(folder / "metadata"),
    }
    paths        = {k: Path(v) for k, v in generation_metadata.get("paths", {}).items()}
    source_image = Path(generation_metadata.get("image_file") or config.get("image_file", ""))
    source_poni  = Path(generation_metadata.get("poni_file")  or config.get("poni_file",  ""))
    accepted_poni = sub["accepted_calibration"] / "accepted_calibration.poni"
    copied: Dict[str, str] = {}

    def _want(key: str) -> bool:
        return selected_keys is None or key in selected_keys

    if _want("original_calibration_image") and source_image.exists():
        copied["original_calibration_image"] = str(copy_file(source_image, sub["source_raw"] / source_image.name))
    _accepted_poni_carries_overrides = False
    _accepted_poni_note = ""
    if _want("source_poni") and source_poni.exists():
        # Always preserve the verbatim input PONI for provenance.
        copied["original_input_poni"] = str(copy_file(source_poni, sub["source_raw"] / source_poni.name))
        # Write the accepted PONI from the geometry actually used (i.e. with any
        # overrides from the QA config applied), so downstream analysis receives
        # the tuned geometry rather than the unmodified input file.
        try:
            import pyFAI as _pyFAI  # type: ignore
            _ai_export = _pyFAI.load(str(source_poni))
            _apply_geometry_overrides(_ai_export, config)
            # pyFAI AzimuthalIntegrator.save(filename) writes a .poni file.
            _ai_export.save(str(accepted_poni))
            copied["accepted_calibration_poni"] = str(accepted_poni)
            _accepted_poni_carries_overrides = True
            print("[export] accepted_calibration.poni written from tuned geometry", flush=True)
        except Exception as _e_poni:
            print(f"[export] WARNING: could not write tuned PONI ({_e_poni}); "
                  "falling back to verbatim copy", flush=True)
            copied["accepted_calibration_poni"] = str(copy_file(source_poni, accepted_poni))
            _accepted_poni_note = f"verbatim copy — geometry overrides may not be applied ({_e_poni})"
    optional_mask = config.get("input_mask_file", "")
    if optional_mask and Path(optional_mask).exists() and _want("original_input_mask"):
        copied["original_input_mask"] = str(copy_file(optional_mask, sub["source_raw"] / Path(optional_mask).name, required=False))
    if _want("mask_npz") and "mask_npz" in paths and paths["mask_npz"].exists():
        copied["accepted_mask_npz"] = str(copy_file(paths["mask_npz"], sub["accepted_calibration"] / "accepted_mask.npz"))
        try:
            mask_arr = load_mask_npz(paths["mask_npz"])
            save_mask_preview_png(sub["accepted_calibration"] / "accepted_mask_preview.png", mask_arr)
            copied["accepted_mask_preview_png"] = str(sub["accepted_calibration"] / "accepted_mask_preview.png")
        except Exception:
            pass
    for key in ["intensity_csv", "difference_csv", "coverage_csv", "cake_npz", "master_csv"]:
        if _want(key) and key in paths and paths[key].exists():
            copied[key] = str(copy_file(paths[key], sub["data"] / paths[key].name))
    figure_keys = [
        "compilation_png", "compilation_pdf",
        "raw_detector_png", "masked_detector_png", "mask_only_png",
        "intensity_difference_png", "intensity_normalized_png",
        "cake_png", "coverage_png",
    ]
    for key in figure_keys:
        if _want(key) and key in paths and paths[key].exists():
            copied[key] = str(copy_file(paths[key], sub["figures"] / paths[key].name))
    for key in ["report_txt", "metadata_json"]:
        if _want(key) and key in paths and paths[key].exists():
            copied[key] = str(copy_file(paths[key], sub["metadata"] / paths[key].name))
    gen_config_snapshot = generation_metadata.get("config")
    if gen_config_snapshot is not None and _want("session_config"):
        write_json(
            sub["metadata"] / "calibration_session_config.json",
            gen_config_snapshot,
        )
        copied["calibration_session_config"] = str(sub["metadata"] / "calibration_session_config.json")
        config_source = "generation_snapshot"
    else:
        session_config_path = config.get("session_config_path", "")
        if session_config_path and Path(session_config_path).exists() and _want("session_config"):
            copied["calibration_session_config"] = str(copy_file(
                session_config_path, sub["metadata"] / "calibration_session_config.json", required=False,
            ))
        config_source = "live_session_file"
    handoff = {
        "handoff_schema_version": "1",
        "tool_version":      VERSION,
        "seriesxrd_version": VERSION,
        "created_at":        now_iso(),
        "accepted_generation": generation_metadata.get("generation"),
        "accepted_folder":   str(folder),
        "accepted_poni":     copied.get("accepted_calibration_poni", ""),
        "accepted_mask_npz": copied.get("accepted_mask_npz", ""),
        "source_image":      str(source_image),
        "source_poni":       str(source_poni),
        "config_source":     config_source,
        "copied_files":      copied,
        "original_generation_metadata": generation_metadata,
        "accepted_poni_carries_overrides": _accepted_poni_carries_overrides,
    }
    if not _accepted_poni_carries_overrides:
        handoff["accepted_poni_note"] = _accepted_poni_note
    write_json(sub["metadata"] / "master_metadata.json",          handoff)
    write_json(sub["metadata"] / "calibration_handoff.json", handoff)
    required = ["accepted_calibration_poni", "accepted_mask_npz", "master_csv", "compilation_png"]
    missing  = [k for k in required if not copied.get(k) or not Path(copied[k]).exists()]
    handoff["verification"] = {"ok": not missing, "missing": missing}
    write_json(sub["metadata"] / "master_metadata.json",          handoff)
    write_json(sub["metadata"] / "calibration_handoff.json", handoff)
    return handoff


# ---------------------------------------------------------------------------
# Cake orientation preview (fast, low-resolution, dual orientation)
# ---------------------------------------------------------------------------

def preview_cake_orientations(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fast low-resolution dual-orientation cake preview.

    Integrates the calibration image both unflipped and vertically flipped at
    low resolution so the user can see which orientation yields straight
    (vertical) calibrant rings before committing to a full QA generation.
    Returns the two preview PNG paths; nothing else is written.
    """
    _FIG_DPI[0] = 110
    _image_str = str(config.get("image_file", "") or "").strip()
    _poni_str  = str(config.get("poni_file",  "") or "").strip()
    if not _image_str or not Path(_image_str).is_file():
        raise FileNotFoundError(f"Calibration image not found: {_image_str!r}")
    if not _poni_str or not Path(_poni_str).is_file():
        raise FileNotFoundError(f"PONI file not found: {_poni_str!r}")
    image_file = Path(_image_str)
    poni_file  = Path(_poni_str)

    _ensure_conda_dlls()
    ai = load_pyfai_integrator(str(poni_file))
    _apply_geometry_overrides(ai, config)

    unit         = str(config.get("unit", "2th_deg") or "2th_deg")
    npt_rad      = 220
    npt_azim     = 180
    method2d     = _CAKE_METHODS_2D[0]  # ("no", "csr", "cython") — fast, low RAM
    radial_range = _compute_poni_radial_range(ai) if unit == "2th_deg" else None
    x_label      = unit_axis_label(unit)

    # Calibrant reference lines — vertical only when geometry/orientation is right.
    calibrant_tth: List = []
    calibrant_name = config.get("calibrant", "")
    wl = None
    try:
        wl = float(config.get("wavelength_m")) if str(config.get("wavelength_m", "")).strip() else None
    except (TypeError, ValueError):
        wl = None
    if wl is None:
        wl = read_poni_info(poni_file).get("wavelength_m")
    if calibrant_name and wl:
        calibrant_tth = _get_calibrant_tth(calibrant_name, float(wl), radial_range, unit=unit)

    out_dir = ensure_dir(output_base(config) / "previews")
    ts = now_timestamp()
    results: Dict[str, Any] = {
        "image_file": str(image_file), "poni_file": str(poni_file),
        "current_flip": bool(config.get("dioptas_image_flip", True)),
    }
    for flip, key, label in ((False, "flip_off_png", "Flip OFF (file orientation)"),
                             (True,  "flip_on_png",  "Flip ON (Dioptas alignment)")):
        img = read_detector_image(image_file, flip_up_down=flip)
        mask = automatic_mask(
            img,
            mask_negative=bool(config.get("mask_negative", True)),
            mask_zero=bool(config.get("mask_zero", True)),
            mask_nonfinite=bool(config.get("mask_nonfinite", True)),
            saturated_threshold=config.get("saturated_threshold", ""),
        )
        print(f"[PREVIEW] integrate2d {label} method={method2d}", flush=True)
        res2 = ai.integrate2d(
            img, npt_rad, npt_azim, mask=mask, unit=unit, method=method2d,
            radial_range=radial_range, azimuth_range=(-180, 180), correctSolidAngle=True,
        )
        png = out_dir / f"cake_preview_{'on' if flip else 'off'}_{ts}.png"
        _make_cake_png(png, np.asarray(res2.intensity), np.asarray(res2.radial),
                       np.asarray(res2.azimuthal), title=f"Cake preview — {label}",
                       x_label=x_label, calibrant_tth=calibrant_tth)
        results[key] = str(png)
    print(f"[PREVIEW] done: {results.get('flip_off_png')} | {results.get('flip_on_png')}", flush=True)
    return results
