"""Multimode heatmap data layer for the analysis stage (Hrubiak/XDI-style).

Following Hrubiak, Smith & Shen (Rev. Sci. Instrum. 90, 025109, 2019): keep the
full I(q) for every frame and compute *functionals* F(I) on demand, rather than
precomputing one reduction. Our series axis is the frame index (or the Step-3a
pressure) in place of their 2D spatial grid.

This module turns the analysis HDF5 into the arrays the GUI plots:

  * ``pattern_image``  — the I(q)/I(2θ) stack as a waterfall image
    (radial axis × frame|pressure), the base "SXDM image".
  * ``reflection_tracks`` — predicted hkl positions of a phase across frames
    (d0·s(P) → radial unit), to overlay on the waterfall.
  * ``phase_layers``  — per-substance ROI-integrated intensity F(I) vs frame
    (Eq. 1 of the paper), i.e. the filterable per-phase "layers" / false-color
    composite source.

Pure numpy + h5py (h5py lazy). ``reflection_tracks``/``phase_layers`` need the
phase reflection lists (pymatgen, via ``identify``); ``pattern_image`` does not.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .phases import Phase
from .identify import radial_to_d, scale_at_pressure, phase_reflections

# Background channels available as the image source. robust/mean/sigmaclip/hybrid
# are reconstructed from the stored clean/baseline/spot_residual(/sigmaclip_residual):
#   robust = clean + baseline,  mean = robust + spot_residual,
#   sigmaclip = clean + sigmaclip_residual,  hybrid = clean + winsorized(spot_residual).
_DIRECT = ("clean", "baseline", "spot_residual")
SOURCES = ("clean", "hybrid", "robust", "mean", "sigmaclip", "baseline", "spot_residual")


def _open(path):
    import h5py  # type: ignore
    return h5py.File(str(Path(path).expanduser()), "r")


def _pressure_track(h5, phase_name: str) -> "Optional[np.ndarray]":
    gid = h5.get("identify")
    if gid is None:
        return None
    for key in gid:
        g = gid[key]
        if hasattr(g, "attrs") and str(g.attrs.get("name", key)) == phase_name and "pressure" in g:
            return np.asarray(g["pressure"][:], dtype=float)
    return None


def _stored_reflections(h5, phase_name: str):
    """Reflections cached under /identify/<phase> by Step 3a, or None.

    Returns ``(d, weight, hkl)`` so the overlay can be built without pymatgen.
    """
    gid = h5.get("identify")
    if gid is None:
        return None
    for key in gid:
        g = gid[key]
        if not hasattr(g, "attrs") or str(g.attrs.get("name", key)) != phase_name:
            continue
        if "refl_d" not in g:
            return None
        d = np.asarray(g["refl_d"][:], dtype=float)
        w = (np.asarray(g["refl_w"][:], dtype=float)
             if "refl_w" in g else np.ones_like(d))
        if "refl_hkl" in g:
            hkl = [x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else str(x)
                   for x in g["refl_hkl"][:]]
        else:
            hkl = [""] * d.size
        return d, w, hkl
    return None


def pattern_image(analysis_h5: "str | Path", *, source: str = "clean",
                  x_axis: str = "frame", pressure_phase: "Optional[str]" = None
                  ) -> Dict[str, Any]:
    """The full pattern stack as a 2D image for a waterfall/heatmap view.

    Returns ``{ok, error, Z, radial, x, x_label, unit, source, n_frames}`` where
    ``Z`` has shape (n_bins, n_frames) — radial down the rows, frames across the
    columns — ready for ``imshow``/``pcolormesh``. With ``x_axis="pressure"`` and
    a ``pressure_phase`` that has a Step-3a track, ``x`` holds that phase's
    per-frame pressure (else the frame index, with an explanatory ``x_label``).
    """
    p = Path(analysis_h5).expanduser()
    out: Dict[str, Any] = {"ok": False, "error": "", "Z": None, "radial": None,
                           "x": None, "x_label": "frame index", "unit": "",
                           "source": source, "n_frames": 0}
    if not p.is_file():
        out["error"] = f"File does not exist: {p}"
        return out
    if source not in SOURCES:
        out["error"] = f"Unknown source {source!r} (choose from {SOURCES})."
        return out
    try:
        import h5py  # type: ignore  # noqa: F401
    except ImportError:
        out["error"] = "h5py is not installed."
        return out
    try:
        with _open(p) as h5:
            out["unit"] = str(h5.attrs.get("unit", ""))
            bg = h5.get("background")
            if bg is None or "clean" not in bg:
                out["error"] = "No /background/clean — run Step 1 first."
                return out
            clean = np.asarray(bg["clean"][:], dtype=float)
            if source == "clean":
                data = clean
            elif source in ("baseline", "spot_residual"):
                if source not in bg:
                    out["error"] = f"/background/{source} not present."
                    return out
                data = np.asarray(bg[source][:], dtype=float)
            elif source == "robust":
                data = clean + np.asarray(bg["baseline"][:], dtype=float)
            elif source == "mean":  # mean = robust + spot_residual
                data = clean + np.asarray(bg["baseline"][:], dtype=float) \
                    + np.asarray(bg["spot_residual"][:], dtype=float)
            elif source == "hybrid":  # clean + winsorized mean-excess (fit default)
                from .peaks import winsorize_excess
                if "spot_residual" not in bg:
                    out["error"] = "/background/spot_residual not present."
                    return out
                data = clean + winsorize_excess(np.asarray(bg["spot_residual"][:], dtype=float))
            else:  # sigmaclip = clean + sigmaclip_residual (reduce-side trimmed mean)
                if "sigmaclip_residual" not in bg:
                    out["error"] = ("/background/sigmaclip_residual not present — re-run "
                                    "reduction with the sigma-clip channel, then Step 1.")
                    return out
                data = clean + np.asarray(bg["sigmaclip_residual"][:], dtype=float)
            n = data.shape[0]
            radial = np.asarray(h5["radial"][:], dtype=float) if "radial" in h5 \
                else np.arange(data.shape[1], dtype=float)
            x = np.arange(n, dtype=float)
            x_label = "frame index"
            if x_axis == "pressure":
                if not pressure_phase:
                    out["error"] = "x_axis='pressure' needs a pressure_phase."
                    return out
                pr = _pressure_track(h5, pressure_phase)
                if pr is None:
                    out["error"] = (f"No Step-3a pressure track for {pressure_phase!r} "
                                    "— run Step 3a first.")
                    return out
                x = pr
                x_label = f"pressure (GPa) — {pressure_phase}"
            out.update({"ok": True, "Z": data.T, "radial": radial, "x": x,
                        "x_label": x_label, "n_frames": int(n)})
    except Exception as e:
        out["error"] = f"Failed to read HDF5: {e!r}"
    return out


def _radial_centers_from_d(d_values: np.ndarray, unit: str,
                           wavelength: "Optional[float]") -> np.ndarray:
    """Inverse of identify.radial_to_d: d-spacing (Å) → reduced radial axis."""
    u = (unit or "").strip().lower()
    d = np.asarray(d_values, float)
    if u in ("q_a^-1", "q_a-1", "q_a", "q"):
        return 2.0 * np.pi / d
    if u in ("q_nm^-1", "q_nm-1", "q_nm"):
        return 2.0 * np.pi / d / 0.1                       # Å^-1 → nm^-1
    if u in ("2th_deg", "2th_rad"):
        if not wavelength:
            raise ValueError("wavelength required to map d-spacing onto a 2θ axis.")
        theta = np.arcsin(np.clip(float(wavelength) / (2.0 * d), -1.0, 1.0))
        return np.degrees(2.0 * theta) if u == "2th_deg" else 2.0 * theta
    raise ValueError(f"Unsupported unit: {unit!r}")


def reflection_tracks(analysis_h5: "str | Path", phase: Phase, *,
                      max_reflections: int = 12) -> Dict[str, Any]:
    """Predicted reflection positions of ``phase`` across frames, on the reduced
    radial axis — curves to overlay on :func:`pattern_image`.

    Uses the phase's Step-3a per-frame pressure (``/identify``) to scale d0·s(P);
    falls back to ambient where no track exists. Returns ``{ok, error, unit,
    n_frames, tracks}`` with ``tracks`` a list (one per kept reflection) of
    ``{hkl, d0, centers}`` where ``centers`` is length-n_frames on the radial
    axis (NaN where pressure is unknown). Requires pymatgen.
    """
    p = Path(analysis_h5).expanduser()
    out: Dict[str, Any] = {"ok": False, "error": "", "unit": "", "n_frames": 0,
                           "tracks": []}
    if not p.is_file():
        out["error"] = f"File does not exist: {p}"
        return out
    try:
        cached = None
        with _open(p) as h5:
            unit = str(h5.attrs.get("unit", ""))
            gid = h5.get("identify")
            wl = float(gid.attrs.get("wavelength", 0.0)) if gid is not None else 0.0
            pr = _pressure_track(h5, phase.name) if gid is not None else None
            cached = _stored_reflections(h5, phase.name)
            bg = h5.get("background")
            n = int(bg["clean"].shape[0]) if bg is not None and "clean" in bg \
                else (int(pr.size) if pr is not None else 0)
        # Prefer the reflections cached at Step 3a (no pymatgen → no GUI freeze);
        # fall back to simulating only if they're absent (e.g. a pre-cache file).
        if cached is not None:
            d0, w, hkl = cached
        else:
            d0, w, hkl = phase_reflections(phase, max_reflections=max_reflections)
        if d0.size == 0 or n == 0:
            out["error"] = "No reflections or frames to track."
            return out
        if pr is None:
            pr = np.zeros(n)  # ambient everywhere
        s = np.array([scale_at_pressure(phase, P) if np.isfinite(P) else np.nan for P in pr])
        tracks = []
        for di, hi in zip(d0, hkl):
            d_at_P = di * s                      # (n,)
            centers = np.full(n, np.nan)
            ok = np.isfinite(d_at_P)
            if ok.any():
                centers[ok] = _radial_centers_from_d(d_at_P[ok], unit, wl or None)
            tracks.append({"hkl": hi, "d0": float(di), "centers": centers})
        out.update({"ok": True, "unit": unit, "n_frames": n, "tracks": tracks})
    except Exception as e:
        out["error"] = f"Failed to build reflection tracks: {e!r}"
    return out


def phase_layers(analysis_h5: "str | Path", phases: "Sequence[Phase]", *,
                 rel_tol: float = 0.01, max_reflections: int = 12) -> Dict[str, Any]:
    """Per-substance ROI-integrated intensity F(I) vs frame (Hrubiak Eq. 1).

    For each phase and frame, integrate the ``clean`` pattern over a narrow window
    (±rel_tol·center) around each predicted reflection (positioned by the phase's
    Step-3a pressure track) and sum — giving one intensity curve per phase, the
    filterable "layer" / false-color composite source.

    Returns ``{ok, error, unit, n_frames, layers}`` with ``layers`` a list of
    ``{name, category, intensity, n_pred}`` (intensity length n_frames, max-
    normalised). Requires pymatgen + a Step-3a ``/identify`` group.
    """
    p = Path(analysis_h5).expanduser()
    out: Dict[str, Any] = {"ok": False, "error": "", "unit": "", "n_frames": 0,
                           "layers": []}
    if not p.is_file():
        out["error"] = f"File does not exist: {p}"
        return out
    try:
        with _open(p) as h5:
            unit = str(h5.attrs.get("unit", ""))
            bg = h5.get("background")
            if bg is None or "clean" not in bg:
                out["error"] = "No /background/clean — run Step 1 first."
                return out
            clean = np.asarray(bg["clean"][:], dtype=float)
            radial = np.asarray(h5["radial"][:], dtype=float)
            if h5.get("identify") is None:
                out["error"] = "No /identify — run Step 3a first for phase layers."
                return out
        n = clean.shape[0]
        order = np.argsort(radial)
        rsorted = radial[order]
        layers = []
        for ph in phases:
            tr = reflection_tracks(analysis_h5, ph, max_reflections=max_reflections)
            if not tr["ok"]:
                continue
            inten = np.zeros(n)
            for i in range(n):
                total = 0.0
                for t in tr["tracks"]:
                    c = t["centers"][i]
                    if not np.isfinite(c):
                        continue
                    half = max(rel_tol * abs(c), 1e-9)
                    lo, hi = c - half, c + half
                    a, b = np.searchsorted(rsorted, [lo, hi])
                    if b > a:
                        total += float(np.nansum(clean[i][order][a:b]))
                inten[i] = total
            mx = inten.max()
            layers.append({"name": ph.name, "category": ph.category,
                           "intensity": inten / mx if mx > 0 else inten,
                           "n_pred": len(tr["tracks"])})
        out.update({"ok": True, "unit": unit, "n_frames": n, "layers": layers})
    except Exception as e:
        out["error"] = f"Failed to build phase layers: {e!r}"
    return out
