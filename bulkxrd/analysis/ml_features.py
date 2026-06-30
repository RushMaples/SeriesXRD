"""Step 3b, part 1 — model-ready frame features from an analysis HDF5.

The canonical "give me an analysis file, return features a model (or the
deterministic ranker in :mod:`ml_rank`) can consume" interface. One frame → one
fixed-length pattern on a wavelength-independent d-grid plus the per-frame
conditioning the rest of the pipeline already computed:

    X                resampled pattern on the shared d-grid (a chosen source)
    pressure(+sigma) the frame-metadata pressure prior (frame_metadata.py)
    contamination    diamond-spot score (background.py)
    n_peaks          good fitted-peak count (peaks.py)
    excluded         reduce-stage bad-frame mask
    candidate_phases the library/Step-3a phases in play

Design choices follow the literature gathered for this pipeline (SimXRD-4M:
preprocess the model input exactly like the experimental data; RADAR-PD: rank
against the *residual*, what is left after the known phases are removed). So the
source can be the recorded Step-2 fit channel, the Step-3a residual, or any raw
background channel — whichever the downstream model was trained on.

Pure numpy + h5py (lazy). Resampling/grid are shared with :mod:`mldata` so the
experimental and simulated patterns land on the identical axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .mldata import make_d_grid, resample_to_d, _normalize_rows
from .peaks import build_fit_source

# A pattern source is any background channel, the recorded Step-2 fit source, or
# the Step-3a residual (what the known phases did not explain — the natural input
# for candidate ranking of impurities/unknowns).
_BG_SOURCES = ("clean", "hybrid", "robust", "mean", "sigmaclip", "baseline", "spot_residual")
SOURCES = ("fit", "residual") + _BG_SOURCES


@dataclass
class FrameFeatures:
    """Per-frame features on a shared d-grid. Array fields are length ``n_frames``
    (rows of ``X`` are length ``len(d_grid)``)."""
    d_grid: np.ndarray
    X: np.ndarray                       # (N, P) resampled pattern of `source`
    frame_index: np.ndarray             # (N,)
    pressure: np.ndarray                # (N,) GPa, NaN where unknown
    pressure_sigma: np.ndarray          # (N,) GPa
    contamination: np.ndarray           # (N,)
    n_peaks: np.ndarray                 # (N,) good fitted peaks
    excluded: np.ndarray                # (N,) bool
    source: str
    unit: str
    wavelength: float
    candidate_phases: List[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.X.shape[0])


def _bg_channel(bg, name: str) -> "Optional[np.ndarray]":
    return np.asarray(bg[name][:], dtype=float) if name in bg else None


def _build_source_stack(h5, source: str) -> "tuple[np.ndarray, str]":
    """Return the (N, n_bins) intensity stack for ``source`` and its resolved name.

    ``fit`` rebuilds the channel Step 2 actually fit (``/peaks.attrs source``);
    ``residual`` reads ``/residual/clean``; otherwise a background channel is
    rebuilt the same way :mod:`heatmap` does (clean + a baseline-subtracted
    residual). Falls back to ``clean`` when a needed channel is absent.
    """
    bg = h5.get("background")
    if bg is None or "clean" not in bg:
        raise ValueError("No /background/clean — run Step 1 first.")
    clean = np.asarray(bg["clean"][:], dtype=float)
    s = (source or "fit").strip().lower()

    if s == "residual":
        rg = h5.get("residual")
        if rg is None or "clean" not in rg:
            raise ValueError("No /residual/clean — run Step 3a (residual) first, "
                             "or choose source='fit'.")
        return np.asarray(rg["clean"][:], dtype=float), "residual"

    spot = _bg_channel(bg, "spot_residual")
    sc = _bg_channel(bg, "sigmaclip_residual")
    if s == "fit":
        pkg = h5.get("peaks")
        want = str(pkg.attrs.get("source", "clean")) if pkg is not None else "clean"
        spike = int(pkg.attrs.get("hybrid_spike_bins", 5)) if pkg is not None else 5
        try:
            data, resolved = build_fit_source(want, clean, spot_residual=spot,
                                              sigmaclip_residual=sc, hybrid_spike_bins=spike)
            return np.asarray(data, dtype=float), resolved
        except ValueError:
            return clean, "clean"
    if s in _BG_SOURCES:
        try:
            data, resolved = build_fit_source(s, clean, spot_residual=spot,
                                              sigmaclip_residual=sc)
            return np.asarray(data, dtype=float), resolved
        except ValueError as e:
            raise ValueError(str(e))
    raise ValueError(f"Unknown source {source!r} (choose from {SOURCES}).")


def frame_features(
    analysis_h5: "str | Path",
    *,
    source: str = "fit",
    d_grid: "Optional[np.ndarray]" = None,
    wavelength: "Optional[float]" = None,
    normalize: bool = True,
) -> FrameFeatures:
    """Build :class:`FrameFeatures` from an analysis HDF5.

    ``source`` selects the pattern fed to the model: ``"fit"`` (the recorded
    Step-2 fit channel, default), ``"residual"`` (Step-3a leftover — the ranking
    input for unknowns), or any raw background channel. Resampling lands on the
    shared d-grid (SimXRD-4M format by default), so these line up bin-for-bin with
    :func:`ml_simulate` patterns.
    """
    import h5py  # type: ignore

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    grid = make_d_grid() if d_grid is None else np.asarray(d_grid, dtype=float)

    with h5py.File(str(src), "r") as h5:
        unit = str(h5.attrs.get("unit", ""))
        stored_wl = float(h5.attrs.get("wavelength", 0.0) or 0.0)
        if wavelength is None and stored_wl > 0:
            wavelength = stored_wl
        if wavelength is None and unit.strip().lower() in ("2th_deg", "2th_rad"):
            raise ValueError("2theta axis needs a wavelength (none stored); pass wavelength=.")
        radial = np.asarray(h5["radial"][:], dtype=float)
        stack, resolved = _build_source_stack(h5, source)
        n = stack.shape[0]

        frames = h5.get("frames")

        def _vec(name, dtype=float, default=np.nan):
            if frames is not None and name in frames:
                return np.asarray(frames[name][:], dtype=dtype)
            return np.full(n, default, dtype=dtype)

        pressure = _vec("pressure")
        pressure_sigma = _vec("pressure_sigma")
        contamination = _vec("contamination", default=0.0)
        excluded = (np.asarray(frames["excluded"][:], dtype=bool)
                    if frames is not None and "excluded" in frames else np.zeros(n, bool))
        n_peaks = _good_peak_counts(h5, n)
        candidate_phases = _candidate_phases(h5)

    X = np.zeros((n, grid.size), dtype="f4")
    for i in range(n):
        X[i] = resample_to_d(radial, stack[i], unit, wavelength, grid)
    if normalize:
        X = _normalize_rows(X)

    return FrameFeatures(
        d_grid=grid, X=X, frame_index=np.arange(n),
        pressure=pressure.astype("f8"), pressure_sigma=pressure_sigma.astype("f8"),
        contamination=contamination.astype("f8"), n_peaks=n_peaks,
        excluded=excluded, source=resolved, unit=unit,
        wavelength=float(wavelength) if wavelength else 0.0,
        candidate_phases=candidate_phases)


def _good_peak_counts(h5, n: int) -> np.ndarray:
    pk = h5.get("peaks")
    if pk is None or "frame" not in pk:
        return np.zeros(n, "i4")
    frame = np.asarray(pk["frame"][:], dtype=int)
    flag = np.asarray(pk["flag"][:], dtype=int) if "flag" in pk else np.zeros_like(frame)
    out = np.zeros(n, "i4")
    for f in frame[flag == 0]:
        if 0 <= f < n:
            out[f] += 1
    return out


def _candidate_phases(h5) -> List[str]:
    gid = h5.get("identify")
    if gid is None:
        return []
    names = []
    for key in gid:
        g = gid[key]
        if hasattr(g, "attrs") and "name" in g.attrs:
            names.append(str(g.attrs["name"]))
    return sorted(set(names))
