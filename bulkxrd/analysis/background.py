"""Step 1 of the analysis pipeline: background-scattering isolation.

Implements the first "identify → record → remove" pass of the categorization
workflow (see categorization.py): separate, per frame, the two background-like
contributions of a diamond-anvil-cell pattern from the sample/marker signal.

Two distinct operations, both recorded so nothing is silently discarded:

1. Single-crystal "spot" residual (diamond anvil reflections, coarse-grain
   spottiness). Powder rings are uniform in azimuth; single-crystal spots are
   not. The reduce stage already produced both the azimuthal MEAN
   (``patterns/intensity``) and the azimuthal MEDIAN (``patterns/intensity_robust``)
   integrations, so:
       spot_residual = mean - robust          (the spot/texture contribution)
       robust        = spot-suppressed powder signal
   A per-frame ``contamination_score`` (integrated positive residual) flags
   frames dominated by diamond spots.

2. Smooth + amorphous background (Compton/air + gasket/pressure-medium humps).
   Estimated from the robust pattern with SNIP (Statistics-sensitive Non-linear
   Iterative Peak-clipping) on an LLS-transformed intensity, then:
       baseline = SNIP(robust)
       clean    = robust - baseline

Pure-numpy logic (no pyFAI). ``separate_background`` works on a single pattern;
``run_background_separation`` drives a whole reduced HDF5.
"""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .parallel import resolve_workers, chunk_ranges


# ---------------------------------------------------------------------------
# SNIP baseline
# ---------------------------------------------------------------------------

def _lls(y: np.ndarray) -> np.ndarray:
    """Log-Log-Sqrt operator — compresses dynamic range so strong peaks do not
    shadow the SNIP background estimate (Morhac 2000)."""
    return np.log(np.log(np.sqrt(np.clip(y, 0.0, None) + 1.0) + 1.0) + 1.0)


def _lls_inv(z: np.ndarray) -> np.ndarray:
    return (np.exp(np.exp(z) - 1.0) - 1.0) ** 2 - 1.0


def snip_baseline(y, max_half_window: int = 40, n_passes: int = 1,
                  use_lls: bool = True) -> np.ndarray:
    """Estimate a smooth baseline under the peaks of a 1D pattern via SNIP.

    For increasing half-window ``m`` = 1..max_half_window, each point is replaced
    by ``min(y[i], (y[i-m] + y[i+m]) / 2)``; anything narrower than the window
    survives as a peak, anything broader is clipped to the baseline.

    Parameters
    ----------
    y : array
        Intensity vs radial bin (work in q for uniform peak widths). NaNs are
        interpolated across for the estimate, then restored.
    max_half_window : int
        Widest feature (in BINS) treated as a peak. Set ~1.5-2x the broadest
        Bragg peak half-width; wider than this is treated as background.
    n_passes : int
        Repeat the full 1..M sweep this many times (>1 pushes the baseline
        lower; risks eroding broad peaks).
    use_lls : bool
        Apply the LLS transform (recommended when intensity spans >2 decades).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()
    finite = np.isfinite(y)
    if not finite.any():
        return np.full(n, np.nan)
    yf = y.copy()
    if not finite.all():
        idx = np.arange(n)
        yf = np.interp(idx, idx[finite], y[finite])

    work = _lls(yf) if use_lls else yf.copy()
    idx = np.arange(n)
    m_max = max(1, int(max_half_window))
    for _ in range(max(1, int(n_passes))):
        for m in range(1, m_max + 1):
            lo = np.clip(idx - m, 0, n - 1)
            hi = np.clip(idx + m, 0, n - 1)
            work = np.minimum(work, 0.5 * (work[lo] + work[hi]))

    base = _lls_inv(work) if use_lls else work
    base = np.minimum(base, yf)        # baseline never sits above the data
    base[~finite] = np.nan             # don't invent values where data was absent
    return base


# ---------------------------------------------------------------------------
# Per-pattern separation
# ---------------------------------------------------------------------------

def spot_residual(mean: np.ndarray, robust: np.ndarray) -> np.ndarray:
    """Single-crystal/spot contribution = azimuthal mean - azimuthal median."""
    return np.asarray(mean, float) - np.asarray(robust, float)


def contamination_score(residual: np.ndarray) -> float:
    """Integrated POSITIVE spot residual for one frame (a diamond-contamination
    metric; negative values are just noise and are ignored)."""
    r = np.asarray(residual, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    return float(np.sum(np.clip(r, 0.0, None)))


def separate_background(intensity, intensity_robust,
                        max_half_window: int = 40, n_passes: int = 1,
                        use_lls: bool = True) -> Dict[str, Any]:
    """Run the full Step-1 separation on one frame's pair of 1D patterns.

    Returns a dict with ``spot_residual``, ``baseline``, ``clean`` (all arrays)
    and ``contamination`` (scalar). ``clean`` is the spot-suppressed,
    background-subtracted powder signal handed to peak fitting.
    """
    mean = np.asarray(intensity, float)
    robust = np.asarray(intensity_robust, float)
    if robust.shape != mean.shape:
        raise ValueError(f"shape mismatch: intensity {mean.shape} vs robust {robust.shape}")
    spots = spot_residual(mean, robust)
    baseline = snip_baseline(robust, max_half_window=max_half_window,
                             n_passes=n_passes, use_lls=use_lls)
    clean = robust - baseline
    return {
        "spot_residual": spots,
        "baseline": baseline,
        "clean": clean,
        "contamination": contamination_score(spots),
    }


# ---------------------------------------------------------------------------
# Dataset driver (reduced HDF5 -> analysis HDF5)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1"


def _parse_wavelength(poni_text: str) -> float:
    """Extract the wavelength in Å from pyFAI PONI text (stored in metres)."""
    m = re.search(r"wavelength\s*:\s*([0-9eE.+-]+)", str(poni_text), re.IGNORECASE)
    if not m:
        return 0.0
    try:
        wl = float(m.group(1))
    except ValueError:
        return 0.0
    return wl * 1e10 if 0 < wl < 1e-6 else wl


def _bg_chunk(payload):
    """Worker: run separate_background over a contiguous chunk of frames."""
    mean_c, robust_c, mhw, npasses, lls = payload
    m, nb = mean_c.shape
    clean = np.full((m, nb), np.nan, "f4")
    base = np.full((m, nb), np.nan, "f4")
    spots = np.full((m, nb), np.nan, "f4")
    contam = np.zeros(m, "f8")
    for j in range(m):
        res = separate_background(mean_c[j], robust_c[j], max_half_window=mhw,
                                  n_passes=npasses, use_lls=lls)
        clean[j] = res["clean"]; base[j] = res["baseline"]
        spots[j] = res["spot_residual"]; contam[j] = res["contamination"]
    return clean, base, spots, contam


def run_background_separation(
    reduced_h5: "str | Path",
    out_h5: "Optional[str | Path]" = None,
    *,
    max_half_window: int = 40,
    n_passes: int = 1,
    use_lls: bool = True,
    contamination_threshold: "Optional[float]" = None,
    num_workers: int = 1,
) -> Dict[str, Any]:
    """Apply Step-1 background separation to every frame of a reduced HDF5.

    Reads ``patterns/intensity``, ``patterns/intensity_robust``,
    ``patterns/radial`` (and ``frames/...`` for provenance). Writes a
    self-contained analysis HDF5:

        /  attrs: schema_version, source_reduced, unit, wavelength, max_half_window, n_passes
        /radial                      (N_bins,)
        /frames/filename             (N,)   copied from the reduced file
        /frames/contamination        (N,)   per-frame spot score
        /frames/excluded             (N,)   bool, carried over from the reduce stage
        /frames/flagged              (N,)   contamination > threshold (if given)
        /frames/pressure             (N,)   GPa; carried from the reduced file, or
                                            parsed from the filenames when that is
                                            empty. NaN where unknown. (Step-3 prior.)
        /frames/temperature          (N,)   K; carried from the reduced file (if any)
        /frames/timestamp            (N,)   str; carried from the reduced file (if any)
        /background/clean            (N, N_bins)
        /background/baseline         (N, N_bins)
        /background/spot_residual    (N, N_bins)
        /background/sigmaclip_residual (N, N_bins)  only if the reduced file has
                                     patterns/intensity_sigmaclip (= sigmaclip − robust)

    Returns a manifest dict (also has ``out_h5`` and per-run stats). Progress
    lines ``[ANALYSIS] <done> <total>`` go to stdout for a supervising UI.
    """
    import h5py  # type: ignore

    src = Path(reduced_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Reduced HDF5 not found: {src}")
    out = Path(out_h5).expanduser().resolve() if out_h5 else src.with_name(src.stem + "_analysis.h5")
    tmp = out.with_name(out.name + ".tmp")

    with h5py.File(str(src), "r") as h5:
        pat = h5.get("patterns")
        if pat is None or "intensity" not in pat or "intensity_robust" not in pat:
            raise ValueError(
                "Reduced file lacks patterns/intensity_robust — re-run reduction "
                "with the robust (azimuthal-median) pattern enabled; it is required "
                "for diamond-spot separation.")
        mean_all = np.asarray(pat["intensity"][:], dtype=float)
        robust_all = np.asarray(pat["intensity_robust"][:], dtype=float)
        # Optional reduce-side azimuthal sigma-clipped (trimmed-mean) channel —
        # keeps azimuthally-sparse real sample intensity that the median drops,
        # while still rejecting diamond single-crystal spots. Carried into the
        # analysis file as a residual so Step 2 can fit on it.
        sigmaclip_all = (np.asarray(pat["intensity_sigmaclip"][:], dtype=float)
                         if "intensity_sigmaclip" in pat else None)
        radial = np.asarray(pat["radial"][:], dtype=float) if "radial" in pat else None
        unit = str(h5.attrs.get("unit", ""))
        poni = h5.attrs.get("poni_text", "")
        if isinstance(poni, (bytes, bytearray)):
            poni = poni.decode("utf-8", "replace")
        wavelength = _parse_wavelength(poni)
        frames = h5.get("frames")
        names = None
        if frames is not None and "filename" in frames:
            names = [x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else str(x)
                     for x in frames["filename"][:]]
        excluded = (np.asarray(frames["excluded"][:], dtype=bool)
                    if frames is not None and "excluded" in frames else None)
        # Per-frame metadata carried straight through to the analysis file so
        # Step 3 can use pressure as a prior (see frame_metadata.py). The reduce
        # stage seeds /frames/pressure as an all-NaN placeholder; we backfill it
        # from the filenames below when it arrives empty.
        red_pressure = (np.asarray(frames["pressure"][:], dtype=float)
                        if frames is not None and "pressure" in frames else None)
        red_temperature = (np.asarray(frames["temperature"][:], dtype=float)
                           if frames is not None and "temperature" in frames else None)
        red_timestamp = (
            [x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else str(x)
             for x in frames["timestamp"][:]]
            if frames is not None and "timestamp" in frames else None)

    n, nb = mean_all.shape
    if excluded is None or excluded.size != n:
        excluded = np.zeros(n, dtype=bool)
    clean = np.full((n, nb), np.nan, dtype="f4")
    baseline = np.full((n, nb), np.nan, dtype="f4")
    spots = np.full((n, nb), np.nan, dtype="f4")
    contam = np.zeros(n, dtype="f8")
    workers = resolve_workers(num_workers)
    print(f"[ANALYSIS] background separation: {n} frames, {nb} bins, "
          f"SNIP M={max_half_window} passes={n_passes} workers={workers}", flush=True)

    if workers > 1 and n > 1:
        ranges = chunk_ranges(n, workers)
        payloads = [(mean_all[a:b], robust_all[a:b], max_half_window, n_passes, use_lls)
                    for a, b in ranges]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (a, b), (c, bs, sp, ct) in zip(ranges, ex.map(_bg_chunk, payloads)):
                clean[a:b] = c; baseline[a:b] = bs; spots[a:b] = sp; contam[a:b] = ct
                done += (b - a)
                print(f"[ANALYSIS] {done} {n}", flush=True)
    else:
        for i in range(n):
            res = separate_background(mean_all[i], robust_all[i],
                                      max_half_window=max_half_window,
                                      n_passes=n_passes, use_lls=use_lls)
            clean[i] = res["clean"]
            baseline[i] = res["baseline"]
            spots[i] = res["spot_residual"]
            contam[i] = res["contamination"]
            if (i + 1) % 25 == 0 or i + 1 == n:
                print(f"[ANALYSIS] {i + 1} {n}", flush=True)

    # sigmaclip channel as a residual on the spot-suppressed median (mirrors how
    # spot_residual = mean − robust is stored), so any source can be rebuilt as
    # clean + <residual> in Step 2 without re-reading the reduced file.
    sigmaclip_residual = None
    if sigmaclip_all is not None and sigmaclip_all.shape == robust_all.shape:
        sigmaclip_residual = (sigmaclip_all - robust_all).astype("f4")

    flagged = None
    if contamination_threshold is not None:
        flagged = contam > float(contamination_threshold)

    # Resolve the per-frame pressure channel: carry the reduced value through,
    # but if it is absent / all-NaN (the usual placeholder case) parse it from
    # the filenames so phase identification has a pressure prior with no manual
    # step. A later CSV import (frame_metadata.import_csv_to_analysis) overrides.
    pressure = red_pressure if (red_pressure is not None and red_pressure.size == n) else None
    n_pressure_parsed = 0
    if (pressure is None or not np.any(np.isfinite(pressure))) and names is not None:
        from .frame_metadata import extract_pressures
        parsed = extract_pressures(names)
        if np.any(np.isfinite(parsed)):
            pressure = parsed
    if pressure is not None:
        n_pressure_parsed = int(np.sum(np.isfinite(pressure)))

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(str(tmp), "w") as o:
            o.attrs.update({
                "tool": "bulkxrd.analysis.background", "schema_version": SCHEMA_VERSION,
                "source_reduced": str(src), "unit": unit,
                "wavelength": float(wavelength),
                "max_half_window": int(max_half_window), "n_passes": int(n_passes),
                "use_lls": bool(use_lls),
                "has_sigmaclip": bool(sigmaclip_residual is not None),
            })
            if radial is not None:
                o.create_dataset("radial", data=radial)
            gf = o.create_group("frames")
            if names is not None:
                import h5py as _h5
                gf.create_dataset("filename", data=np.array(names, dtype=object),
                                  dtype=_h5.string_dtype(encoding="utf-8"))
            gf.create_dataset("contamination", data=contam)
            gf.create_dataset("excluded", data=excluded)
            if flagged is not None:
                gf.create_dataset("flagged", data=flagged)
            # Frame metadata (pressure prior + provenance). pressure is always
            # written (NaN where unknown) so Step 3 can read a consistent channel.
            gf.create_dataset("pressure",
                              data=(pressure if pressure is not None
                                    else np.full(n, np.nan, "f8")).astype("f8"))
            if red_temperature is not None and red_temperature.size == n:
                gf.create_dataset("temperature", data=red_temperature.astype("f8"))
            if red_timestamp is not None and len(red_timestamp) == n:
                import h5py as _h5t
                gf.create_dataset("timestamp",
                                  data=np.array(red_timestamp, dtype=object),
                                  dtype=_h5t.string_dtype(encoding="utf-8"))
            gb = o.create_group("background")
            gb.create_dataset("clean", data=clean, compression="gzip", compression_opts=1)
            gb.create_dataset("baseline", data=baseline, compression="gzip", compression_opts=1)
            gb.create_dataset("spot_residual", data=spots, compression="gzip", compression_opts=1)
            if sigmaclip_residual is not None:
                gb.create_dataset("sigmaclip_residual", data=sigmaclip_residual,
                                  compression="gzip", compression_opts=1)
        import os
        os.replace(tmp, out)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    manifest = {
        "tool_version": SCHEMA_VERSION,
        "source_reduced": str(src),
        "out_h5": str(out),
        "n_frames": int(n),
        "n_bins": int(nb),
        "unit": unit,
        "wavelength": float(wavelength) or None,
        "n_excluded": int(excluded.sum()),
        "n_pressure": int(n_pressure_parsed),
        "max_half_window": int(max_half_window),
        "n_passes": int(n_passes),
        "contamination_threshold": contamination_threshold,
        "n_flagged": int(flagged.sum()) if flagged is not None else None,
        "has_sigmaclip": bool(sigmaclip_residual is not None),
        "contamination_min": float(np.min(contam)) if n else 0.0,
        "contamination_max": float(np.max(contam)) if n else 0.0,
    }
    print(f"[ANALYSIS] done -> {out}", flush=True)
    return manifest
