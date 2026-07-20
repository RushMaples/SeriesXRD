"""Cake waviness: diagnose ring wobble and straighten cakes into sharp 1D.

A Debye ring should be a straight vertical line in a cake (intensity vs radial
× azimuth). When the scattering sample is not exactly where the calibrant was
— the routine case in a DAC, where the calibrant is measured outside the cell —
the ring wobbles sinusoidally in azimuth even though the PONI is "correct":

    r(φ) ≈ r0 + A1·cos(φ − φ1)  +  A2·cos 2(φ − φ2)

* ``A1`` (one period per turn) ⇔ a transverse sample/beam-center offset. On
  azimuthal integration the ring's radial histogram is bimodal (intensity piles
  at the two turning points r0 ± A1) → every peak becomes a DOUBLE-horned peak
  of constant splitting ~2·A1 across the whole pattern.
* ``A2`` (two periods) ⇔ residual detector tilt; grows with radius.

Two remedies, in order of preference:
1. Re-refine the geometry on a ring measured at the *sample* position and
   re-reduce (:func:`fit_waviness` returns the offset in axis units; with the
   detector distance that converts to millimetres of sample displacement).
2. When re-reduction isn't possible, :func:`straighten_cake` shifts each
   azimuthal row by the fitted wobble and collapses the cake into a corrected
   1D pattern — recovering single sharp peaks from the same data.

Pure numpy + scipy; h5py only in :func:`diagnose_reduced`. Operates on the
``/cakes`` group the reduce stage writes with ``save_cakes=True``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def ring_centroids(cake: np.ndarray, radial: np.ndarray, r0: float,
                   halfwidth: float, *, min_height_sigma: float = 3.0
                   ) -> np.ndarray:
    """Per-azimuthal-row intensity centroid of the ring near ``r0``.

    ``cake`` is (n_azimuthal, n_radial) — pyFAI's integrate2d layout. Rows where
    the window holds no significant intensity (gaps, masked sectors, spotty
    rings) return NaN so they simply don't constrain the fit. The centroid is
    computed on the window's baseline-subtracted excess so a sloping background
    can't drag it.
    """
    radial = np.asarray(radial, float)
    m = (radial >= r0 - halfwidth) & (radial <= r0 + halfwidth)
    if m.sum() < 3:
        return np.full(cake.shape[0], np.nan)
    x = radial[m]
    out = np.full(cake.shape[0], np.nan)
    for j in range(cake.shape[0]):
        y = np.asarray(cake[j][m], float)
        fin = np.isfinite(y)
        if fin.sum() < 3:
            continue
        y = np.where(fin, y, 0.0)
        base = np.median(y[fin])
        exc = np.clip(y - base, 0.0, None)
        noise = 1.4826 * np.median(np.abs(y[fin] - base)) or 1.0
        if exc.max() < min_height_sigma * noise:
            continue
        out[j] = float(np.sum(x * exc) / np.sum(exc))
    return out


def fit_waviness(azimuthal_deg: np.ndarray, centers: np.ndarray
                 ) -> Dict[str, Any]:
    """Fit ``r(φ) = r0 + a·cosφ + b·sinφ + c·cos2φ + d·sin2φ`` by linear LS.

    Returns ``{ok, r0, A1, phi1_deg, A2, phi2_deg, rms, n}`` — ``A1`` is the
    first-harmonic amplitude (transverse offset signature; the doublet splitting
    it produces in 1D is ≈ 2·A1), ``A2`` the second (tilt signature), both in
    the radial-axis units; ``rms`` is the residual scatter of the used rows.
    """
    phi = np.radians(np.asarray(azimuthal_deg, float))
    c = np.asarray(centers, float)
    ok = np.isfinite(c)
    out = {"ok": False, "r0": float("nan"), "A1": float("nan"),
           "phi1_deg": float("nan"), "A2": float("nan"),
           "phi2_deg": float("nan"), "rms": float("nan"), "n": int(ok.sum())}
    if ok.sum() < 8:                       # need real azimuthal coverage
        return out
    P, y = phi[ok], c[ok]
    M = np.column_stack([np.ones_like(P), np.cos(P), np.sin(P),
                         np.cos(2 * P), np.sin(2 * P)])
    sol, *_ = np.linalg.lstsq(M, y, rcond=None)
    r0, a, b, cc, dd = sol
    resid = y - M @ sol
    out.update(ok=True, r0=float(r0),
               A1=float(math.hypot(a, b)), phi1_deg=float(np.degrees(math.atan2(b, a))),
               A2=float(math.hypot(cc, dd)),
               phi2_deg=float(np.degrees(math.atan2(dd, cc)) / 2.0),
               rms=float(np.sqrt(np.mean(resid ** 2))))
    return out


def _row_shifts(azimuthal_deg: np.ndarray, fits: "Sequence[Dict[str, Any]]",
                ring_r0: "Sequence[float]") -> np.ndarray:
    """Per-row radial shift δ(φ) from one or more ring fits (amplitude-averaged,
    weighted by each ring's fit quality). Positive = ring sits above r0 there."""
    phi = np.radians(np.asarray(azimuthal_deg, float))
    num = np.zeros_like(phi)
    den = 0.0
    for f in fits:
        if not f.get("ok"):
            continue
        w = 1.0 / max(f["rms"], 1e-6)
        num += w * (f["A1"] * np.cos(phi - math.radians(f["phi1_deg"]))
                    + f["A2"] * np.cos(2 * (phi - math.radians(f["phi2_deg"]))))
        den += w
    return num / den if den > 0 else np.zeros_like(phi)


def straighten_cake(cake: np.ndarray, radial: np.ndarray,
                    azimuthal_deg: np.ndarray, *,
                    ring_r0: "Optional[Sequence[float]]" = None,
                    halfwidth: "Optional[float]" = None,
                    n_rings: int = 3) -> Dict[str, Any]:
    """Fit the wobble on the strongest rings and collapse a corrected 1D pattern.

    Each azimuthal row is resampled onto ``radial - δ(φ)`` (the fitted wobble),
    so rings align before averaging. Returns ``{ok, intensity, intensity_median,
    fits, shifts, rings}`` — ``intensity`` is the straightened azimuthal mean on
    the original radial axis (NaN-aware), ``intensity_median`` the straightened
    median (spot-suppressed), ``fits`` the per-ring :func:`fit_waviness` results.

    ``ring_r0`` picks the rings explicitly; otherwise the ``n_rings`` strongest
    maxima of the collapsed pattern are used. ``halfwidth`` defaults to 8 radial
    bins.
    """
    cake = np.asarray(cake, float)
    radial = np.asarray(radial, float)
    az = np.asarray(azimuthal_deg, float)
    dr = float(np.median(np.abs(np.diff(radial)))) or 1.0
    hw = float(halfwidth) if halfwidth else 8.0 * dr

    import warnings
    if ring_r0 is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmean(np.where(cake > 0, cake, np.nan), axis=0)
        prof = np.nan_to_num(prof, nan=0.0)
        from scipy.signal import find_peaks
        med = np.median(prof)
        mad = 1.4826 * np.median(np.abs(prof - med)) or 1.0
        idx, props = find_peaks(prof, height=med + 5 * mad, prominence=3 * mad,
                                distance=max(3, int(hw / dr)))
        if idx.size == 0:
            return {"ok": False, "error": "no rings found to fit", "fits": [],
                    "intensity": None, "intensity_median": None,
                    "shifts": None, "rings": []}
        order = np.argsort(props["peak_heights"])[::-1][:max(1, int(n_rings))]
        ring_r0 = sorted(float(radial[k]) for k in idx[order])

    fits: List[Dict[str, Any]] = []
    for r0 in ring_r0:
        cent = ring_centroids(cake, radial, r0, hw)
        f = fit_waviness(az, cent)
        f["ring_r0"] = float(r0)
        fits.append(f)
    if not any(f["ok"] for f in fits):
        return {"ok": False, "error": "no ring had enough azimuthal coverage",
                "fits": fits, "intensity": None, "intensity_median": None,
                "shifts": None, "rings": list(ring_r0)}

    shifts = _row_shifts(az, fits, ring_r0)
    rows = np.full_like(cake, np.nan)
    for j in range(cake.shape[0]):
        y = np.asarray(cake[j], float)
        fin = np.isfinite(y) & (y > 0)          # pyFAI cakes use 0 for empty
        if fin.sum() < 4:
            continue
        rows[j] = np.interp(radial, radial[fin] - shifts[j], y[fin],
                            left=np.nan, right=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN bins expected
        mean1d = np.nanmean(rows, axis=0)
        med1d = np.nanmedian(rows, axis=0)
    return {"ok": True, "intensity": mean1d, "intensity_median": med1d,
            "fits": fits, "shifts": shifts, "rings": list(ring_r0)}


def straighten_reduced(reduced_h5: "str | Path", *, n_rings: int = 3
                       ) -> Dict[str, Any]:
    """Straighten every cake in a reduced HDF5 and write the corrected 1D
    channels back (atomically): ``/patterns/intensity_straightened`` (azimuthal
    MEAN), ``/patterns/intensity_straightened_robust`` (azimuthal MEDIAN,
    spot-suppressed) and ``/frames/waviness_A1``.

    Two channels mirror ``intensity`` / ``intensity_robust``: the mean keeps
    azimuthally-sparse intensity but also the diamond single-crystal spots, the
    median rejects the spots. The analysis Step-1 ``robust_source="straightened"``
    consumes the median (de-waved AND spot-suppressed) in place of the ordinary
    ``intensity_robust`` so peak fitting sees single, un-split rings.

    The straightened pattern lives on the CAKE's radial grid internally and is
    interpolated onto ``/patterns/radial``; frames without a saved cake (a
    ``cake_every`` > 1 reduction) stay NaN. This is the *rescue* path for data
    already collected with a sample-position offset — the proper fix remains
    re-refining the geometry on a sample-position ring and re-reducing, which
    also restores full radial resolution (cakes are usually coarser than the
    1D axis). Returns a manifest with per-frame amplitudes.
    """
    import os
    import shutil
    import h5py  # type: ignore

    src = Path(reduced_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Reduced HDF5 not found: {src}")
    with h5py.File(str(src), "r") as h5:
        g = h5.get("cakes")
        if g is None or "intensity" not in g:
            raise ValueError("No /cakes — re-run reduction with save_cakes=True.")
        radial_1d = np.asarray(h5["patterns/radial"][:], float)
        n_frames = h5["patterns/intensity"].shape[0]
        cake_radial = np.asarray(g["radial"][:], float)
        az = np.asarray(g["azimuthal"][:], float)
        fidx = (np.asarray(g["frame_index"][:], int) if "frame_index" in g
                else np.arange(g["intensity"].shape[0]))
        straight = np.full((n_frames, radial_1d.size), np.nan, "f4")
        straight_med = np.full((n_frames, radial_1d.size), np.nan, "f4")
        A1 = np.full(n_frames, np.nan, "f8")

        def _onto_1d(y):
            """Interpolate a cake-grid 1D profile onto the pattern radial axis."""
            fin = np.isfinite(y)
            if fin.sum() <= 4:
                return None
            return np.interp(radial_1d, cake_radial[fin], np.asarray(y)[fin],
                             left=np.nan, right=np.nan)

        for k in range(g["intensity"].shape[0]):
            fr = int(fidx[k])
            if not (0 <= fr < n_frames):
                continue
            res = straighten_cake(np.asarray(g["intensity"][k], float),
                                  cake_radial, az, n_rings=n_rings)
            if not res["ok"]:
                continue
            ym = _onto_1d(res["intensity"])
            if ym is not None:
                straight[fr] = ym
            ymed = res.get("intensity_median")
            if ymed is not None:
                yd = _onto_1d(ymed)
                if yd is not None:
                    straight_med[fr] = yd
            best = max((f for f in res["fits"] if f["ok"]),
                       key=lambda f: f["A1"], default=None)
            if best:
                A1[fr] = best["A1"]

    tmp = src.with_name(src.name + ".tmp")
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            gp = o["patterns"]
            if "intensity_straightened" in gp:
                del gp["intensity_straightened"]
            gp.create_dataset("intensity_straightened", data=straight,
                              compression="gzip", compression_opts=1)
            if "intensity_straightened_robust" in gp:
                del gp["intensity_straightened_robust"]
            gp.create_dataset("intensity_straightened_robust", data=straight_med,
                              compression="gzip", compression_opts=1)
            gf = o.require_group("frames")
            if "waviness_A1" in gf:
                del gf["waviness_A1"]
            gf.create_dataset("waviness_A1", data=A1)
        os.replace(tmp, src)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    done = int(np.isfinite(A1).sum())
    print(f"[STRAIGHTEN] {done}/{n_frames} frames straightened -> "
          f"/patterns/intensity_straightened (+_robust) (median A1="
          f"{np.nanmedian(A1) if done else float('nan'):.4g})", flush=True)
    return {"out_h5": str(src), "n_frames": int(n_frames),
            "n_straightened": done,
            "A1_median": float(np.nanmedian(A1)) if done else None}


def diagnose_reduced(reduced_h5: "str | Path", *, n_rings: int = 3,
                     max_cakes: "Optional[int]" = None) -> Dict[str, Any]:
    """Waviness report for every cake in a reduced HDF5 (needs ``save_cakes``).

    Returns ``{ok, error, unit, n_cakes, per_frame, summary}``. ``per_frame``
    rows carry the strongest-ring first/second harmonic amplitudes (axis units);
    ``summary`` reports their medians, the implied 1D doublet splitting (~2·A1),
    and — when the PONI text yields a distance — the transverse sample offset in
    mm (``offset_mm = distance · Δ2θ_amplitude``, small-angle).
    """
    import h5py  # type: ignore
    import re

    p = Path(reduced_h5).expanduser()
    out: Dict[str, Any] = {"ok": False, "error": "", "unit": "", "n_cakes": 0,
                           "per_frame": [], "summary": {}}
    if not p.is_file():
        out["error"] = f"File not found: {p}"
        return out
    with h5py.File(str(p), "r") as h5:
        g = h5.get("cakes")
        if g is None or "intensity" not in g:
            out["error"] = ("No /cakes in this file — re-run reduction with "
                            "save_cakes=True to diagnose ring waviness.")
            return out
        unit = str(h5.attrs.get("unit", ""))
        radial = np.asarray(g["radial"][:], float)
        az = np.asarray(g["azimuthal"][:], float)
        fidx = np.asarray(g["frame_index"][:], int) if "frame_index" in g else None
        n = g["intensity"].shape[0]
        take = range(n if max_cakes is None else min(n, int(max_cakes)))
        wl = 0.0
        m = re.search(r"wavelength\s*:\s*([0-9eE.+-]+)",
                      str(h5.attrs.get("poni_text", "")), re.IGNORECASE)
        if m:
            wl = float(m.group(1))
            wl = wl * 1e10 if 0 < wl < 1e-6 else wl
        md = re.search(r"distance\s*:\s*([0-9eE.+-]+)",
                       str(h5.attrs.get("poni_text", "")), re.IGNORECASE)
        dist_m = float(md.group(1)) if md else 0.0

        A1s, A2s = [], []
        for k in take:
            cake = np.asarray(g["intensity"][k], float)
            res = straighten_cake(cake, radial, az, n_rings=n_rings)
            row = {"cake": int(k),
                   "frame": int(fidx[k]) if fidx is not None else int(k),
                   "ok": bool(res["ok"])}
            if res["ok"]:
                best = max((f for f in res["fits"] if f["ok"]),
                           key=lambda f: f["A1"])
                row.update(A1=best["A1"], phi1_deg=best["phi1_deg"],
                           A2=best["A2"], ring_r0=best["ring_r0"],
                           rms=best["rms"])
                A1s.append(best["A1"]); A2s.append(best["A2"])
            out["per_frame"].append(row)

    out["unit"] = unit
    out["n_cakes"] = len(out["per_frame"])
    if A1s:
        A1 = float(np.median(A1s))
        summ = {"A1_median": A1, "A2_median": float(np.median(A2s)),
                "doublet_splitting": 2.0 * A1, "axis_unit": unit}
        # Convert the first harmonic to a physical transverse offset.
        u = unit.strip().lower()
        d2t = None
        if u.startswith("q") and wl > 0:
            d2t = A1 * wl / (2.0 * math.pi)            # rad, small-angle
        elif u == "2th_deg":
            d2t = math.radians(A1)
        elif u == "2th_rad":
            d2t = A1
        if d2t is not None:
            summ["waviness_2theta_deg"] = math.degrees(d2t)
            if dist_m > 0:
                summ["offset_mm"] = d2t * dist_m * 1000.0
                summ["distance_mm"] = dist_m * 1000.0
        out["summary"] = summ
    out["ok"] = bool(A1s)
    if not out["ok"] and not out["error"]:
        out["error"] = "No cake produced a usable ring fit."
    return out
