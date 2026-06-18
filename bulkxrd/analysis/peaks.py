"""Step 2 of the analysis pipeline: peak / profile fitting.

Takes the background-separated ``clean`` patterns from Step 1 (see
``background.py``) and fits every Bragg peak with a pseudo-Voigt profile, so
each reflection is reduced to a small, physically meaningful parameter set:

    center      peak position on the radial axis (q or 2theta, as reduced)
    amplitude   peak height of the profile
    fwhm        full width at half maximum  -> crystallite size & microstrain
    eta         Lorentzian fraction in [0,1] (size vs strain character)
    area        integrated intensity (texture / phase fraction)
    chi2        reduced chi-square of the local fit (goodness)
    flag        0 = good, nonzero = a rejection reason (see FLAG_*)

Pipeline per frame:
    1. detect candidate peaks (local maxima with SNR above a MAD noise floor),
    2. group peaks whose windows overlap and fit each group jointly as a sum of
       pseudo-Voigts plus a local constant baseline (Levenberg-Marquardt),
    3. reject implausible fits, record everything.

Across a frame series the peak centers drift slowly (the lattice compresses),
so ``fit_dataset`` optionally *seeds* each frame's detection with the previous
frame's fitted centers — this keeps peak identity coherent for the Step-3
heatmap. The model evaluation is vectorized (numpy broadcasting); per-frame
fits are independent and could be parallelized, but scipy least-squares on
~10^3 frames runs in seconds, so the default backend is plain serial scipy.

Depends on numpy + scipy (scipy is imported lazily). ``run_peak_fitting``
drives a whole analysis HDF5 written by Step 1.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .parallel import resolve_workers, chunk_ranges


# Rejection flags (bitwise-OR-able).
FLAG_OK = 0
FLAG_LOW_AMP = 1        # amplitude fell below the noise floor during fitting
FLAG_BAD_CHI2 = 2       # reduced chi-square above threshold (poor fit)
FLAG_CENTER_DRIFT = 4   # center moved more than ~1 FWHM from its seed
FLAG_WIDTH_BOUND = 8    # fwhm pinned to a bound (degenerate width)
FLAG_NO_CONVERGE = 16   # optimizer did not converge

_GAUSS_C = 4.0 * np.log(2.0)          # exp(-_GAUSS_C * (dx/fwhm)^2) has the given FWHM
_GAUSS_AREA = np.sqrt(np.pi / _GAUSS_C)   # integral of unit-height gaussian / fwhm


# ---------------------------------------------------------------------------
# Profile model (vectorized)
# ---------------------------------------------------------------------------

def pseudo_voigt(x, center, amplitude, fwhm, eta) -> np.ndarray:
    """Pseudo-Voigt profile ``A * (eta*L + (1-eta)*G)`` with peak height ``A``.

    All of ``center, amplitude, fwhm, eta`` may be scalars or broadcastable
    arrays; with ``x`` of shape (N,) and the parameters of shape (K,) passing
    ``center[None, :]`` etc. yields an (N, K) stack of profiles.
    """
    x = np.asarray(x, float)
    w = np.asarray(fwhm, float)
    dx = x - np.asarray(center, float)
    z = (2.0 * dx / w) ** 2                       # = 4 dx^2 / fwhm^2
    lor = 1.0 / (1.0 + z)
    gau = np.exp(-np.log(2.0) * z)                # exp(-4 ln2 dx^2 / fwhm^2)
    eta = np.asarray(eta, float)
    return np.asarray(amplitude, float) * (eta * lor + (1.0 - eta) * gau)


def pseudo_voigt_area(amplitude, fwhm, eta) -> np.ndarray:
    """Analytic integral of :func:`pseudo_voigt` over all x."""
    a = np.asarray(amplitude, float)
    w = np.asarray(fwhm, float)
    e = np.asarray(eta, float)
    lor_area = 0.5 * np.pi * w          # integral of unit-height lorentzian
    gau_area = _GAUSS_AREA * w          # integral of unit-height gaussian
    return a * (e * lor_area + (1.0 - e) * gau_area)


# ---------------------------------------------------------------------------
# Noise floor & peak detection
# ---------------------------------------------------------------------------

def mad_sigma(y) -> float:
    """Robust noise estimate: 1.4826 * MAD, computed on finite values only.

    Robust to the peaks themselves because the median ignores the (sparse)
    high-intensity bins.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return float(1.4826 * mad)


def _half_max_width(x, y, k) -> float:
    """Estimate FWHM at local-max index ``k`` from the half-height crossings."""
    n = y.size
    half = 0.5 * y[k]
    li = k
    while li > 0 and y[li] > half:
        li -= 1
    ri = k
    while ri < n - 1 and y[ri] > half:
        ri += 1
    w = abs(x[ri] - x[li])
    if not np.isfinite(w) or w <= 0:
        # fall back to a couple of bins' worth
        dx = np.median(np.abs(np.diff(x))) if n > 1 else 1.0
        w = 3.0 * float(dx)
    return float(w)


def detect_peaks(x, y, *, min_snr: float = 5.0,
                 min_prominence_snr: Optional[float] = None,
                 sigma: Optional[float] = None,
                 seed_centers: Optional[Sequence[float]] = None,
                 ) -> List[Dict[str, float]]:
    """Find candidate peaks and seed initial fit parameters.

    Returns a list of ``{center, amplitude, fwhm}`` dicts (one per candidate),
    sorted by center. Uses ``scipy.signal.find_peaks`` with a height threshold
    AND a prominence threshold, both in units of the MAD noise floor ``σ``:

    * ``min_snr·σ`` — minimum height (a peak must clear the noise).
    * ``min_prominence_snr·σ`` — minimum prominence. Decoupled from height
      because prominence is measured against the *taller* neighbour: a real peak
      sitting on the shoulder of a stronger one has low prominence and was being
      rejected even though its height was fine. Defaults to ``min_snr`` (the old
      coupled behaviour) when not given; set it lower to keep shoulder/adjacent
      peaks.

    If ``seed_centers`` is given, seeds with no detection are added back so peak
    identity survives across a frame series.
    """
    from scipy.signal import find_peaks  # lazy

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yf = np.where(np.isfinite(y), y, 0.0)
    sig = float(sigma) if sigma is not None else mad_sigma(y)
    sig = sig if sig > 0 else (np.nanstd(y) or 1.0)
    prom_snr = min_snr if min_prominence_snr is None else float(min_prominence_snr)
    height = min_snr * sig
    prominence = max(prom_snr * sig, 1e-12)
    idx, _ = find_peaks(yf, height=height, prominence=prominence)

    cands: List[Dict[str, float]] = []
    for k in idx:
        cands.append({"center": float(x[k]), "amplitude": float(yf[k]),
                      "fwhm": _half_max_width(x, yf, int(k))})

    if seed_centers is not None and len(seed_centers):
        seeds = np.sort(np.asarray(seed_centers, float))
        got = np.array([c["center"] for c in cands], float)
        # Re-add only seeds with no detection within ~half a peak width — a peak
        # that merely drifted between frames is already detected here; this
        # recovers reflections that dipped below SNR, without duplicating them.
        dx = np.median(np.abs(np.diff(x))) if x.size > 1 else 1.0
        typ_w = np.median([c["fwhm"] for c in cands]) if cands else 3.0 * float(dx)
        tol = max(3.0 * float(dx), 0.5 * float(typ_w))
        for s in seeds:
            if got.size == 0 or np.min(np.abs(got - s)) > tol:
                k = int(np.argmin(np.abs(x - s)))
                if np.isfinite(yf[k]):
                    cands.append({"center": float(x[k]),
                                  "amplitude": float(max(yf[k], height)),
                                  "fwhm": _half_max_width(x, yf, k)})
    cands.sort(key=lambda c: c["center"])
    return cands


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _group_peaks(cands: List[Dict[str, float]], window_factor: float
                 ) -> List[List[Dict[str, float]]]:
    """Cluster peaks whose +-window_factor*fwhm windows overlap, for joint fit."""
    groups: List[List[Dict[str, float]]] = []
    cur: List[Dict[str, float]] = []
    cur_hi = -np.inf
    for c in cands:
        lo = c["center"] - window_factor * c["fwhm"]
        hi = c["center"] + window_factor * c["fwhm"]
        if cur and lo <= cur_hi:
            cur.append(c)
            cur_hi = max(cur_hi, hi)
        else:
            if cur:
                groups.append(cur)
            cur = [c]
            cur_hi = hi
    if cur:
        groups.append(cur)
    return groups


def _fit_group(x, y, group, sigma, *, window_factor: float, max_chi2: float
               ) -> List[Dict[str, Any]]:
    """Jointly fit one cluster of peaks (sum of pseudo-Voigts + constant)."""
    from scipy.optimize import least_squares  # lazy

    centers = np.array([g["center"] for g in group], float)
    amps = np.array([max(g["amplitude"], 1e-9) for g in group], float)
    widths = np.array([max(g["fwhm"], 1e-9) for g in group], float)
    lo = float(centers.min() - window_factor * widths.max())
    hi = float(centers.max() + window_factor * widths.max())
    m = (x >= lo) & (x <= hi) & np.isfinite(y)
    xw, yw = x[m], y[m]
    K = len(group)
    if xw.size < 4 * K + 1:                       # too few points to fit
        return [_failed_peak(g, FLAG_NO_CONVERGE) for g in group]

    # parameter vector: [c,a,w,eta]*K + [baseline]
    p0, lb, ub = [], [], []
    for c, a, w in zip(centers, amps, widths):
        p0 += [c, a, w, 0.5]
        lb += [c - 0.5 * w, 0.0, 0.2 * w, 0.0]
        ub += [c + 0.5 * w, 5.0 * a + 1e-9, 5.0 * w, 1.0]
    p0.append(0.0); lb.append(-np.inf); ub.append(np.inf)   # constant baseline
    p0 = np.array(p0); lb = np.array(lb); ub = np.array(ub)
    p0 = np.clip(p0, lb, ub)
    sig = sigma if sigma > 0 else 1.0

    def model(p):
        c = p[0:4 * K:4]; a = p[1:4 * K:4]; w = p[2:4 * K:4]; e = p[3:4 * K:4]
        prof = pseudo_voigt(xw[:, None], c[None, :], a[None, :], w[None, :], e[None, :])
        return prof.sum(axis=1) + p[-1]

    def resid(p):
        return (model(p) - yw) / sig

    try:
        sol = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=200 * (K + 1))
        converged = bool(sol.success)
        p = sol.x
    except Exception:
        return [_failed_peak(g, FLAG_NO_CONVERGE) for g in group]

    dof = max(xw.size - p.size, 1)
    chi2 = float(np.sum(resid(p) ** 2) / dof)

    out: List[Dict[str, Any]] = []
    for j, g in enumerate(group):
        c, a, w, e = p[4 * j:4 * j + 4]
        flag = FLAG_OK if converged else FLAG_NO_CONVERGE
        if a < 2.0 * sig:
            flag |= FLAG_LOW_AMP
        if chi2 > max_chi2:
            flag |= FLAG_BAD_CHI2
        if abs(c - g["center"]) > max(g["fwhm"], 1e-9):
            flag |= FLAG_CENTER_DRIFT
        if w <= 0.2 * g["fwhm"] * 1.001 or w >= 5.0 * g["fwhm"] * 0.999:
            flag |= FLAG_WIDTH_BOUND
        out.append({
            "center": float(c), "amplitude": float(a), "fwhm": float(abs(w)),
            "eta": float(np.clip(e, 0.0, 1.0)),
            "area": float(pseudo_voigt_area(a, abs(w), np.clip(e, 0.0, 1.0))),
            "chi2": chi2, "flag": int(flag),
        })
    return out


def _failed_peak(g: Dict[str, float], flag: int) -> Dict[str, Any]:
    return {"center": float(g["center"]), "amplitude": float(g["amplitude"]),
            "fwhm": float(g["fwhm"]), "eta": 0.5, "area": 0.0,
            "chi2": np.inf, "flag": int(flag)}


def fit_pattern(x, y, *, min_snr: float = 5.0, window_factor: float = 3.0,
                max_chi2: float = 25.0, min_prominence_snr: Optional[float] = None,
                sigma: Optional[float] = None,
                seed_centers: Optional[Sequence[float]] = None,
                keep_flagged: bool = True) -> List[Dict[str, Any]]:
    """Detect and fit every peak in one 1D pattern.

    Returns a list of peak dicts (``center, amplitude, fwhm, eta, area, chi2,
    flag``) sorted by center. If ``keep_flagged`` is False, peaks whose flag is
    nonzero are dropped from the result.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sig = float(sigma) if sigma is not None else mad_sigma(y)
    cands = detect_peaks(x, y, min_snr=min_snr, min_prominence_snr=min_prominence_snr,
                         sigma=sig, seed_centers=seed_centers)
    peaks: List[Dict[str, Any]] = []
    for group in _group_peaks(cands, window_factor):
        peaks.extend(_fit_group(x, y, group, sig, window_factor=window_factor,
                                max_chi2=max_chi2))
    peaks.sort(key=lambda p: p["center"])
    if not keep_flagged:
        peaks = [p for p in peaks if p["flag"] == FLAG_OK]
    return peaks


def fit_dataset(radial, clean, *, min_snr: float = 5.0, window_factor: float = 3.0,
                max_chi2: float = 25.0, min_prominence_snr: Optional[float] = None,
                propagate_seeds: bool = True,
                keep_flagged: bool = True) -> List[List[Dict[str, Any]]]:
    """Fit every frame of a (N_frames, N_bins) ``clean`` stack.

    With ``propagate_seeds`` the good centers of frame *i* seed detection for
    frame *i+1*, so a reflection keeps its identity as it drifts across the
    series (helps the Step-3 heatmap). Returns a list (per frame) of peak lists.
    """
    radial = np.asarray(radial, float)
    clean = np.asarray(clean, float)
    results: List[List[Dict[str, Any]]] = []
    seeds: Optional[List[float]] = None
    for i in range(clean.shape[0]):
        peaks = fit_pattern(radial, clean[i], min_snr=min_snr,
                            window_factor=window_factor, max_chi2=max_chi2,
                            min_prominence_snr=min_prominence_snr,
                            seed_centers=seeds, keep_flagged=keep_flagged)
        results.append(peaks)
        if propagate_seeds:
            good = [p["center"] for p in peaks if p["flag"] == FLAG_OK]
            seeds = good if good else seeds
    return results


# ---------------------------------------------------------------------------
# Dataset driver (analysis HDF5 -> peaks appended)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1"
_PEAK_COLS = ("center", "amplitude", "fwhm", "eta", "area", "chi2", "flag")


def _peaks_chunk(payload):
    """Worker: fit a contiguous chunk of frames with internal seed propagation.

    Returns ``(counts, frames_local, cols)`` where ``frames_local`` are indices
    within the chunk (the parent offsets them to global frame indices). Excluded
    frames are skipped (count 0) and do not seed their neighbours.
    """
    (radial, clean_c, excluded_c, min_snr, window_factor, max_chi2,
     propagate, min_prominence_snr) = payload
    m = clean_c.shape[0]
    counts = [0] * m
    frames_local: List[int] = []
    cols: Dict[str, list] = {c: [] for c in _PEAK_COLS}
    seeds: Optional[List[float]] = None
    for j in range(m):
        if excluded_c[j]:
            continue
        peaks = fit_pattern(radial, clean_c[j], min_snr=min_snr,
                            window_factor=window_factor, max_chi2=max_chi2,
                            min_prominence_snr=min_prominence_snr,
                            seed_centers=seeds, keep_flagged=True)
        counts[j] = len(peaks)
        for p in peaks:
            frames_local.append(j)
            for c in _PEAK_COLS:
                cols[c].append(p[c])
        if propagate:
            good = [p["center"] for p in peaks if p["flag"] == FLAG_OK]
            seeds = good if good else seeds
    return counts, frames_local, cols


def run_peak_fitting(
    analysis_h5: "str | Path",
    out_h5: "Optional[str | Path]" = None,
    *,
    min_snr: float = 5.0,
    window_factor: float = 3.0,
    max_chi2: float = 25.0,
    min_prominence_snr: Optional[float] = None,
    propagate_seeds: bool = True,
    num_workers: int = 1,
) -> Dict[str, Any]:
    """Fit peaks for every frame of a Step-1 analysis HDF5 and store the result.

    Reads ``/radial`` and ``/background/clean`` (written by
    ``background.run_background_separation``). Writes a ``/peaks`` group with a
    flat, ragged layout so the per-frame peak count can vary:

        /peaks/counts      (N_frames,)   peaks fitted per frame
        /peaks/frame       (P,)          frame index of each peak (0..N-1)
        /peaks/center      (P,)          \\
        /peaks/amplitude   (P,)           |
        /peaks/fwhm        (P,)           |  one row per fitted peak,
        /peaks/eta         (P,)           |  grouped/ordered by frame then center
        /peaks/area        (P,)           |
        /peaks/chi2        (P,)          /
        /peaks/flag        (P,) int      0 = good, else FLAG_* bitmask

    where P = sum(counts). If ``out_h5`` is given the source is copied there
    first and peaks are written into the copy; otherwise ``/peaks`` is added to
    the analysis file in place (replacing any existing group). Returns a
    manifest dict; prints ``[PEAKS] <done> <total>`` progress.
    """
    import h5py  # type: ignore
    import os
    import shutil

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    dst = Path(out_h5).expanduser().resolve() if out_h5 else src

    with h5py.File(str(src), "r") as h5:
        bg = h5.get("background")
        if bg is None or "clean" not in bg:
            raise ValueError(
                "Analysis file lacks /background/clean — run Step 1 "
                "(background.run_background_separation) first.")
        clean = np.asarray(bg["clean"][:], dtype=float)
        radial = np.asarray(h5["radial"][:], dtype=float) if "radial" in h5 else \
            np.arange(clean.shape[1], dtype=float)
        unit = str(h5.attrs.get("unit", ""))
        frames = h5.get("frames")
        excluded = (np.asarray(frames["excluded"][:], dtype=bool)
                    if frames is not None and "excluded" in frames else None)

    n = clean.shape[0]
    if excluded is None or excluded.size != n:
        excluded = np.zeros(n, dtype=bool)
    workers = resolve_workers(num_workers)
    prom_txt = min_snr if min_prominence_snr is None else min_prominence_snr
    print(f"[PEAKS] fitting {n} frames ({int(excluded.sum())} excluded), "
          f"radial[{radial.size}] unit={unit or '?'} min_snr={min_snr} "
          f"min_prom={prom_txt} window={window_factor} workers={workers}", flush=True)

    counts = np.zeros(n, dtype="i4")
    cols: Dict[str, list] = {c: [] for c in _PEAK_COLS}
    frame_idx: list = []

    def _absorb(a, result):
        cc, fl, cols_c = result
        counts[a:a + len(cc)] = cc
        frame_idx.extend(x + a for x in fl)
        for c in _PEAK_COLS:
            cols[c].extend(cols_c[c])

    if workers > 1 and n > 1:
        ranges = chunk_ranges(n, workers)
        payloads = [(radial, clean[a:b], excluded[a:b], min_snr, window_factor,
                     max_chi2, propagate_seeds, min_prominence_snr) for a, b in ranges]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (a, b), result in zip(ranges, ex.map(_peaks_chunk, payloads)):
                _absorb(a, result)
                done += (b - a)
                print(f"[PEAKS] {done} {n}", flush=True)
    else:
        _absorb(0, _peaks_chunk((radial, clean, excluded, min_snr, window_factor,
                                 max_chi2, propagate_seeds, min_prominence_snr)))
        print(f"[PEAKS] {n} {n}", flush=True)

    P = int(counts.sum())
    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            if "peaks" in o:
                del o["peaks"]
            gp = o.create_group("peaks")
            gp.attrs.update({"schema_version": SCHEMA_VERSION, "min_snr": float(min_snr),
                             "min_prominence_snr": float(
                                 min_snr if min_prominence_snr is None else min_prominence_snr),
                             "window_factor": float(window_factor),
                             "max_chi2": float(max_chi2),
                             "propagate_seeds": bool(propagate_seeds)})
            gp.create_dataset("counts", data=counts)
            gp.create_dataset("frame", data=np.asarray(frame_idx, dtype="i4"))
            for c in _PEAK_COLS:
                dt = "i4" if c == "flag" else "f8"
                gp.create_dataset(c, data=np.asarray(cols[c], dtype=dt))
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    n_good = int(np.sum(np.asarray(cols["flag"], dtype=int) == FLAG_OK)) if P else 0
    manifest = {
        "tool_version": SCHEMA_VERSION, "source": str(src), "out_h5": str(dst),
        "n_frames": int(n), "n_peaks": P, "n_good": n_good,
        "n_flagged": P - n_good, "unit": unit,
        "min_snr": float(min_snr), "window_factor": float(window_factor),
        "max_chi2": float(max_chi2),
        "peaks_per_frame_mean": float(counts.mean()) if n else 0.0,
    }
    print(f"[PEAKS] done -> {dst}  ({P} peaks, {n_good} good)", flush=True)
    return manifest
