"""Multimode heatmap data layer for the analysis stage (Hrubiak/XDI-style).

Following Hrubiak, Smith & Shen (Rev. Sci. Instrum. 90, 025109, 2019): keep the
full I(q) for every frame and compute *functionals* F(I) on demand, rather than
precomputing one reduction. Our series axis is the frame index (or the Step-3a
pressure) in place of their 2D spatial grid.

This module turns the analysis HDF5 into the arrays the GUI plots:

  * ``pattern_image``  — the I(q)/I(2θ) stack as a waterfall image
    (radial axis × series axis), the base "SXDM image".
  * ``reflection_tracks`` — predicted hkl positions of a phase across frames
    (d0·s(P) → radial unit), to overlay on the waterfall.
  * ``phase_layers``  — per-substance ROI-integrated intensity F(I) vs frame
    (Eq. 1 of the paper), i.e. the filterable per-phase "layers" / false-color
    composite source.
  * ``series_axis``   — the per-frame independent variable (frame index,
    pressure, temperature, or elapsed time) any series plot can use as x.
  * ``frame_grid`` / ``grid_map`` — refold a linear frame series onto the 2D
    scan grid it was collected on (horizontal/vertical scan lines,
    boustrophedon or unidirectional), for mapping experiments.
  * ``frame_values``  — per-frame scalars (integrated/max intensity, optional
    ROI, contamination, peak count, P, T) to display on that grid.

Pure numpy + h5py (h5py lazy). ``reflection_tracks``/``phase_layers`` need the
phase reflection lists (pymatgen, via ``identify``); ``pattern_image`` does not.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .phases import Phase
from .identify import radial_to_d, phase_reflections, predicted_d, _parse_hkl

# Background channels available as the image source. robust/mean/sigmaclip/hybrid
# are reconstructed from the stored clean/baseline/spot_residual(/sigmaclip_residual):
#   robust = clean + baseline,  mean = robust + spot_residual,
#   sigmaclip = clean + sigmaclip_residual,  hybrid = clean + winsorized(spot_residual).
_DIRECT = ("clean", "baseline", "spot_residual")
SOURCES = ("clean", "hybrid", "robust", "mean", "sigmaclip", "baseline", "spot_residual")


def _open(path):
    import h5py  # type: ignore
    return h5py.File(str(Path(path).expanduser()), "r")


def _peaks_fit_source(h5) -> np.ndarray:
    """Reconstruct the channel Step 2 fit the peaks on (``/peaks.attrs source``),
    falling back to ``clean`` when a needed residual channel is missing. Phase
    layers must integrate this same source — integrating ``clean`` would
    under-report textured/spotty rings the median dropped but the fit kept."""
    bg = h5.get("background")
    clean = np.asarray(bg["clean"][:], dtype=float)
    pkg = h5.get("peaks")
    want = str(pkg.attrs.get("source", "clean")) if pkg is not None else "clean"
    spike = int(pkg.attrs.get("hybrid_spike_bins", 5)) if pkg is not None else 5
    spot = np.asarray(bg["spot_residual"][:], dtype=float) if "spot_residual" in bg else None
    sc = np.asarray(bg["sigmaclip_residual"][:], dtype=float) if "sigmaclip_residual" in bg else None
    from .peaks import build_fit_source
    try:
        data, _ = build_fit_source(want, clean, spot_residual=spot, sigmaclip_residual=sc,
                                   hybrid_spike_bins=spike)
        return np.asarray(data, dtype=float)
    except ValueError:
        return clean


def _frame_pressure(h5) -> "Optional[np.ndarray]":
    """Per-frame metadata pressure (GPa) from ``/frames/pressure``, or None when
    absent / entirely NaN (i.e. nothing was parsed or imported)."""
    fr = h5.get("frames")
    if fr is None or "pressure" not in fr:
        return None
    pr = np.asarray(fr["pressure"][:], dtype=float)
    return pr if np.any(np.isfinite(pr)) else None


def _frame_temperature(h5) -> "Optional[np.ndarray]":
    """Per-frame metadata temperature (K), or None when absent / all-NaN."""
    fr = h5.get("frames")
    if fr is None or "temperature" not in fr:
        return None
    tt = np.asarray(fr["temperature"][:], dtype=float)
    return tt if np.any(np.isfinite(tt)) else None


def _frame_elapsed_seconds(h5) -> "Optional[np.ndarray]":
    """Per-frame elapsed time (s) from ``/frames/timestamp`` ISO strings,
    relative to the first parseable stamp. None when absent/unparseable."""
    from datetime import datetime
    fr = h5.get("frames")
    if fr is None or "timestamp" not in fr:
        return None
    raw = fr["timestamp"][:]
    secs = np.full(len(raw), np.nan)
    for i, s in enumerate(raw):
        txt = s.decode("utf-8", "replace") if isinstance(s, (bytes, bytearray)) else str(s)
        txt = txt.strip()
        if not txt:
            continue
        try:
            secs[i] = datetime.fromisoformat(txt).timestamp()
        except ValueError:
            continue
    if not np.any(np.isfinite(secs)):
        return None
    return secs - np.nanmin(secs)


# Independent variables a series plot can use on its x-axis.
SERIES_AXES = ("frame", "pressure", "temperature", "time")


def _series_x(h5, kind: str, pressure_phase: "Optional[str]" = None,
              n: "Optional[int]" = None):
    """Resolve the per-frame independent variable on an OPEN analysis file.

    Returns ``(x, label)``. Raises ValueError with an instructive message when
    the requested variable is not in the file.
    """
    k = (kind or "frame").strip().lower()
    if k in ("frame", "index", ""):
        if n is None:
            raise ValueError("frame axis needs the frame count")
        return np.arange(int(n), dtype=float), "frame index"
    if k == "pressure":
        if pressure_phase:
            pr = _pressure_track(h5, pressure_phase)
            if pr is None:
                raise ValueError(f"No Step-3a pressure track for {pressure_phase!r} "
                                 "— run Step 3a first.")
            return pr, f"pressure (GPa) — {pressure_phase}"
        pr = _frame_pressure(h5)
        if pr is None:
            raise ValueError("No frame pressure — extract/import one on the "
                             "Frame metadata tab, or pass a pressure_phase "
                             "with a Step-3a track.")
        return pr, "pressure (GPa)"
    if k == "temperature":
        tt = _frame_temperature(h5)
        if tt is None:
            raise ValueError("No frame temperature — import a CSV with a "
                             "temperature_K column on the Frame metadata tab.")
        return tt, "temperature (K)"
    if k in ("time", "timestamp"):
        ts = _frame_elapsed_seconds(h5)
        if ts is None:
            raise ValueError("No parseable /frames/timestamp in this file.")
        return ts, "elapsed time (s)"
    raise ValueError(f"Unknown series axis {kind!r} (choose from {SERIES_AXES}).")


def series_axis(analysis_h5: "str | Path", kind: str = "frame", *,
                pressure_phase: "Optional[str]" = None) -> Dict[str, Any]:
    """Per-frame independent variable for plotting.

    Returns ``{ok, error, x, label, kind, n_frames}``. ``kind`` is one of
    :data:`SERIES_AXES`; "pressure" uses ``/frames/pressure`` (or the
    ``pressure_phase`` Step-3a track), "temperature" uses
    ``/frames/temperature``, "time" parses ``/frames/timestamp``.
    """
    out: Dict[str, Any] = {"ok": False, "error": "", "x": None,
                           "label": "", "kind": kind, "n_frames": 0}
    p = Path(analysis_h5).expanduser()
    if not p.is_file():
        out["error"] = f"File does not exist: {p}"
        return out
    try:
        with _open(p) as h5:
            bg = h5.get("background")
            fr = h5.get("frames")
            if bg is not None and "clean" in bg:
                n = int(bg["clean"].shape[0])
            elif fr is not None and "filename" in fr:
                n = int(fr["filename"].shape[0])
            else:
                out["error"] = "No frames in file — run Step 1 first."
                return out
            x, label = _series_x(h5, kind, pressure_phase, n=n)
        out.update({"ok": True, "x": np.asarray(x, dtype=float),
                    "label": label, "n_frames": n})
    except ValueError as e:
        out["error"] = str(e)
    except Exception as e:
        out["error"] = f"Failed to read HDF5: {e!r}"
    return out


# ---------------------------------------------------------------------------
# Scan-grid mapping (2D raster/mapping experiments)
# ---------------------------------------------------------------------------

def frame_grid(n_frames: int, *, n_cols: "Optional[int]" = None,
               n_rows: "Optional[int]" = None, order: str = "horizontal",
               serpentine: bool = True) -> np.ndarray:
    """Map a linear frame series onto a 2D scan grid.

    ``order="horizontal"`` fills scan lines as ROWS (give ``n_cols`` = frames
    per line); ``order="vertical"`` fills scan lines as COLUMNS (give
    ``n_rows`` = frames per line). Only one of ``n_cols``/``n_rows`` is
    needed; the other is derived. ``serpentine=True`` (boustrophedon) reverses
    every second scan line, as collected by a back-and-forth stage raster;
    ``False`` is a unidirectional raster (every line same direction).

    Returns an int array (n_rows, n_cols) of frame indices, ``-1`` where the
    last line is not filled. Frame 0 is the top-left of the first scan line.
    """
    n = int(n_frames)
    if n <= 0:
        raise ValueError("n_frames must be positive")
    nc = int(n_cols) if n_cols else 0
    nr = int(n_rows) if n_rows else 0
    if nc < 0 or nr < 0:
        raise ValueError("n_cols/n_rows must be positive")
    if not nc and not nr:
        raise ValueError("give n_cols (horizontal) or n_rows (vertical) — "
                         "the frames-per-scan-line count")
    o = (order or "horizontal").strip().lower()
    if o not in ("horizontal", "vertical"):
        raise ValueError(f"order must be 'horizontal' or 'vertical', got {order!r}")
    # Scan-line length and number of lines, regardless of orientation.
    if o == "horizontal":
        line = nc or int(np.ceil(n / nr))
    else:
        line = nr or int(np.ceil(n / nc))
    if line <= 0:
        raise ValueError("scan-line length must be positive")
    n_lines = int(np.ceil(n / line))
    flat = np.full(n_lines * line, -1, dtype=int)
    flat[:n] = np.arange(n)
    g = flat.reshape(n_lines, line)
    if serpentine:
        g[1::2] = g[1::2, ::-1]
    return g if o == "horizontal" else g.T


def grid_map(values, *, n_cols: "Optional[int]" = None,
             n_rows: "Optional[int]" = None, order: str = "horizontal",
             serpentine: bool = True) -> np.ndarray:
    """Arrange a per-frame scalar onto the scan grid (see :func:`frame_grid`).

    Returns a float array (n_rows, n_cols); padding cells are NaN.
    """
    v = np.asarray(values, dtype=float).ravel()
    g = frame_grid(v.size, n_cols=n_cols, n_rows=n_rows, order=order,
                   serpentine=serpentine)
    out = np.full(g.shape, np.nan)
    ok = g >= 0
    out[ok] = v[g[ok]]
    return out


def _cluster_1d(vals: np.ndarray, tol: "Optional[float]" = None):
    """Cluster 1D stage coordinates into grid lines.

    Motor read-back jitters around the commanded positions, so equal
    coordinates repeat with small scatter while distinct scan lines sit a
    full pitch apart. With no explicit ``tol``, the split point is found at
    the largest ratio jump in the sorted consecutive differences (jitter ≪
    pitch); a clean jump-free spacing means every distinct value is its own
    line. Returns ``(labels, centers)`` — cluster index per input value and
    the sorted cluster centers.
    """
    v = np.asarray(vals, dtype=float)
    order = np.argsort(v, kind="stable")
    s = v[order]
    d = np.diff(s)
    if tol is None:
        pos = np.sort(d[d > 0])
        if pos.size == 0:
            tol = 0.0
        elif pos.size == 1:
            tol = 0.5 * pos[0]
        else:
            ratios = pos[1:] / pos[:-1]
            k = int(np.argmax(ratios))
            if ratios[k] > 4.0:      # clear jitter → pitch boundary
                tol = float(np.sqrt(pos[k] * pos[k + 1]))
            else:                    # clean grid: every distinct value a line
                tol = 0.5 * float(pos[0])
    labels_sorted = np.zeros(s.size, dtype=int)
    if s.size > 1:
        labels_sorted[1:] = np.cumsum(d > tol)
    centers = np.array([float(s[labels_sorted == c].mean())
                        for c in range(labels_sorted.max() + 1)])
    labels = np.empty(v.size, dtype=int)
    labels[order] = labels_sorted
    return labels, centers


def coordinate_grid(pos_x, pos_y, *, tol_x: "Optional[float]" = None,
                    tol_y: "Optional[float]" = None) -> Dict[str, Any]:
    """Place frames on the 2D grid implied by their stage coordinates.

    The automatic alternative to :func:`frame_grid` when per-frame positions
    exist (``/frames/pos_x``/``pos_y``): no frames-per-line or scan-direction
    input needed — collection order becomes irrelevant because every frame
    carries its own (x, y). Coordinates are clustered per axis (see
    :func:`_cluster_1d`), so an irregular collection order, missing frames,
    or serpentine vs raster all land correctly.

    Returns ``{ok, error, grid, x_centers, y_centers, n_placed, n_collisions,
    fill_frac}`` — ``grid[row, col]`` is the frame index (−1 empty), row 0 at
    the smallest y, col 0 at the smallest x; a collision (two frames on one
    cell) keeps the later frame.
    """
    out: Dict[str, Any] = {"ok": False, "error": "", "grid": None,
                           "x_centers": None, "y_centers": None,
                           "n_placed": 0, "n_collisions": 0, "fill_frac": 0.0}
    x = np.asarray(pos_x, dtype=float)
    y = np.asarray(pos_y, dtype=float)
    if x.size != y.size:
        out["error"] = "pos_x and pos_y differ in length."
        return out
    fin = np.isfinite(x) & np.isfinite(y)
    if fin.sum() < 2:
        out["error"] = ("Fewer than two frames have both x and y positions — "
                        "import them from a CSV or the frame headers first.")
        return out
    idx = np.nonzero(fin)[0]
    cx, x_centers = _cluster_1d(x[fin], tol_x)
    cy, y_centers = _cluster_1d(y[fin], tol_y)
    grid = np.full((y_centers.size, x_centers.size), -1, dtype=int)
    n_coll = 0
    for k, fi in enumerate(idx):
        r, c = int(cy[k]), int(cx[k])
        if grid[r, c] >= 0:
            n_coll += 1
        grid[r, c] = int(fi)
    filled = int(np.sum(grid >= 0))
    out.update({"ok": True, "grid": grid, "x_centers": x_centers,
                "y_centers": y_centers, "n_placed": int(idx.size),
                "n_collisions": n_coll,
                "fill_frac": filled / grid.size if grid.size else 0.0})
    return out


# Per-frame scalars the scan-grid map can display (besides phase layers).
FRAME_VALUES = ("total", "max", "contamination", "n_peaks",
                "pressure", "temperature")


def frame_values(analysis_h5: "str | Path", kind: str = "total", *,
                 radial_min: "Optional[float]" = None,
                 radial_max: "Optional[float]" = None) -> Dict[str, Any]:
    """One scalar per frame, for the scan-grid map or a series plot.

    ``kind``: "total"/"max" integrate/peak the Step-2 fit source, optionally
    restricted to ``radial_min..radial_max`` (an ROI on the radial axis);
    "contamination" reads ``/frames/contamination``; "n_peaks" reads
    ``/peaks/counts``; "pressure"/"temperature" read the frame metadata.
    Returns ``{ok, error, values, label, n_frames}``.
    """
    out: Dict[str, Any] = {"ok": False, "error": "", "values": None,
                           "label": kind, "n_frames": 0}
    p = Path(analysis_h5).expanduser()
    if not p.is_file():
        out["error"] = f"File does not exist: {p}"
        return out
    k = (kind or "total").strip().lower()
    try:
        with _open(p) as h5:
            if k in ("total", "max"):
                bg = h5.get("background")
                if bg is None or "clean" not in bg:
                    out["error"] = "No /background/clean — run Step 1 first."
                    return out
                data = _peaks_fit_source(h5)
                radial = np.asarray(h5["radial"][:], dtype=float) \
                    if "radial" in h5 else np.arange(data.shape[1], dtype=float)
                sel = np.ones(radial.size, dtype=bool)
                if radial_min is not None:
                    sel &= radial >= float(radial_min)
                if radial_max is not None:
                    sel &= radial <= float(radial_max)
                if not sel.any():
                    out["error"] = "Radial window selects no bins."
                    return out
                roi = (radial_min is not None or radial_max is not None)
                if k == "total":
                    vals = np.nansum(data[:, sel], axis=1)
                    label = "integrated intensity" + (" (ROI)" if roi else "")
                else:
                    vals = np.nanmax(data[:, sel], axis=1)
                    label = "max intensity" + (" (ROI)" if roi else "")
            elif k == "contamination":
                fr = h5.get("frames")
                if fr is None or "contamination" not in fr:
                    out["error"] = "No /frames/contamination — run Step 1 first."
                    return out
                vals = np.asarray(fr["contamination"][:], dtype=float)
                label = "contamination score"
            elif k == "n_peaks":
                pk = h5.get("peaks")
                if pk is None or "counts" not in pk:
                    out["error"] = "No /peaks — run Step 2 first."
                    return out
                vals = np.asarray(pk["counts"][:], dtype=float)
                label = "peaks per frame"
            elif k in ("pressure", "temperature"):
                arr = (_frame_pressure(h5) if k == "pressure"
                       else _frame_temperature(h5))
                if arr is None:
                    out["error"] = (f"No frame {k} — populate it on the "
                                    "Frame metadata tab.")
                    return out
                vals = arr
                label = "pressure (GPa)" if k == "pressure" else "temperature (K)"
            else:
                out["error"] = (f"Unknown value {kind!r} "
                                f"(choose from {FRAME_VALUES}).")
                return out
        out.update({"ok": True, "values": np.asarray(vals, dtype=float),
                    "label": label, "n_frames": int(np.asarray(vals).size)})
    except Exception as e:
        out["error"] = f"Failed to read HDF5: {e!r}"
    return out


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
    columns — ready for ``imshow``/``pcolormesh``. ``x_axis`` is any of
    :data:`SERIES_AXES` ("frame", "pressure", "temperature", "time");
    ``x`` then holds that per-frame variable. For "pressure", the frame
    metadata (``/frames/pressure``) is used unless a ``pressure_phase`` with a
    Step-3a track is given. An unavailable variable is an error, not a silent
    fallback.
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
            try:
                x, x_label = _series_x(h5, x_axis, pressure_phase, n=n)
            except ValueError as e:
                out["error"] = str(e)
                return out
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

    Pressure per frame comes from the phase's Step-3a track (``/identify``) when
    present, else the frame-metadata pressure (``/frames/pressure``), else
    ambient. Compression uses the same anisotropic :func:`identify.predicted_d`
    as identification (per-axis EOS + hkl when available, isotropic otherwise),
    so a softer axis's reflections shift correctly relative to a stiff one.
    Returns ``{ok, error, unit, n_frames, tracks}`` with ``tracks`` a list (one
    per kept reflection) of ``{hkl, d0, centers}`` where ``centers`` is
    length-n_frames on the radial axis (NaN where pressure is unknown). Requires
    pymatgen only when the Step-3a reflection cache is absent.
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
            if pr is None:                       # no Step-3a track → metadata pressure
                pr = _frame_pressure(h5)
            tK = _frame_temperature(h5)          # thermal-expansion seam (if any)
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
        d0 = np.asarray(d0, dtype=float)
        hkls = [_parse_hkl(h) for h in hkl]
        # Predicted d for every reflection at each frame's pressure (anisotropic
        # via predicted_d). dmat[reflection, frame].
        dmat = np.full((d0.size, n), np.nan)
        for fi, P in enumerate(pr):
            if not np.isfinite(P):
                continue
            T_i = (float(tK[fi]) if tK is not None and fi < tK.size
                   and np.isfinite(tK[fi]) else None)
            dmat[:, fi] = predicted_d(phase, d0, hkls, float(max(P, 0.0)), T_i)
        tracks = []
        for ri, (di, hi) in enumerate(zip(d0, hkl)):
            d_at_P = dmat[ri]                     # (n,)
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

    For each phase and frame, integrate the Step-2 **fit source** (``/peaks.attrs
    source`` — auto→sigmaclip/hybrid by default, not the median ``clean``) over a
    narrow window (±rel_tol·center) around each predicted reflection (positioned by
    the phase's Step-3a pressure track) and sum — giving one intensity curve per
    phase, the filterable "layer" / false-color composite source.

    Returns ``{ok, error, unit, n_frames, layers}`` with ``layers`` a list of
    ``{name, category, intensity, intensity_raw, n_pred}`` (both length
    n_frames; ``intensity`` is max-normalised, ``intensity_raw`` is the raw
    ROI-integrated counts for cross-phase comparison). Requires pymatgen + a
    Step-3a ``/identify`` group.
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
            if h5.get("identify") is None:
                out["error"] = "No /identify — run Step 3a first for phase layers."
                return out
            clean = _peaks_fit_source(h5)          # integrate the fit source, not median clean
            radial = np.asarray(h5["radial"][:], dtype=float)
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
                           "intensity_raw": inten.copy(),
                           "n_pred": len(tr["tracks"])})
        out.update({"ok": True, "unit": unit, "n_frames": n, "layers": layers})
    except Exception as e:
        out["error"] = f"Failed to build phase layers: {e!r}"
    return out
