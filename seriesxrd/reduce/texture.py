"""Azimuthal texture analysis: per-ring intensity vs azimuth.

A Debye ring from an ideal, randomly-oriented powder has intensity uniform in
azimuth. Two things break that uniformity, and this module distinguishes them
by how the intensity varies with azimuthal angle ``φ``:

* **Preferred orientation** (texture) — grains aren't randomly oriented, so
  the ring intensity is modulated smoothly, dominantly at the second harmonic:
  ``I(φ) ≈ c0 + c2·cos 2(φ − φ2)``. In a DAC the same signature can also arise
  from uniaxial stress along the load axis (radial vs axial diffraction
  geometry produce similar-looking 2-fold modulation); telling the two apart
  needs the experiment geometry, which is out of scope here — this module
  only measures the modulation.
* **Coarse grains / near-single-crystal spots** — too few crystallites in the
  beam means the ring is not smooth at all: intensity concentrates in a few
  bright azimuthal rows ("spotty") rather than following a smooth harmonic.

This module MEASURES both (per-ring azimuthal profile, an overall texture
index, a spot fraction, and the second-harmonic preferred-orientation
amplitude/phase) — it does not attribute the modulation to a physical cause.

Same ``/cakes`` layout and ring auto-pick heuristic as
:mod:`seriesxrd.reduce.straighten`: a cake is ``(n_azimuthal, n_radial)``
(pyFAI's ``integrate2d`` convention), with ``/cakes/radial``,
``/cakes/azimuthal``, ``/cakes/frame_index`` alongside ``/cakes/intensity``.

Pure numpy + scipy (ring auto-pick uses ``scipy.signal.find_peaks``); h5py
only in :func:`run_texture`.

References: H.-R. Wenk & S. Grigull, J. Appl. Cryst. 36 (2003) 1040
(quantitative texture from synchrotron area-detector images — the azimuthal
intensity variation this module measures); A. K. Singh, C. Balasingh, H.-K.
Mao, R. J. Hemley & J. Shu, J. Appl. Phys. 83 (1998) 7567 (lattice strains
under nonhydrostatic DAC compression — the stress-driven 2-fold analogue;
note Singh's theory modulates peak POSITION with azimuth, while texture
modulates INTENSITY, which is one practical way to tell them apart).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import VERSION, now_iso


def ring_profile(cake: np.ndarray, radial: np.ndarray, azimuthal_deg: np.ndarray,
                 r0: float, halfwidth: float, *, min_height_sigma: float = 3.0
                 ) -> Dict[str, Any]:
    """Per-azimuthal-row intensity of the ring near ``r0``, integrated over
    ``radial in [r0-halfwidth, r0+halfwidth]``.

    Mirrors :func:`seriesxrd.reduce.straighten.ring_centroids`'s baseline and
    significance gating: each row's window is baseline-subtracted (its own
    median) before summing the positive excess, and rows whose window holds
    no significant intensity (gaps, masked sectors, spotty rings) come back
    NaN so they don't distort the azimuthal statistics — only the final
    reduction differs (integrated intensity here, a radial centroid there).

    Returns ``{ok, phi (deg), intensity}``.
    """
    cake = np.asarray(cake, float)
    radial = np.asarray(radial, float)
    phi = np.asarray(azimuthal_deg, float)
    m = (radial >= r0 - halfwidth) & (radial <= r0 + halfwidth)
    intensity = np.full(cake.shape[0], np.nan)
    if m.sum() < 3:
        return {"ok": False, "phi": phi, "intensity": intensity}
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
        intensity[j] = float(np.sum(exc))
    ok = bool(np.isfinite(intensity).sum() >= 3)
    return {"ok": ok, "phi": phi, "intensity": intensity}


def texture_metrics(phi_deg: np.ndarray, intensity: np.ndarray) -> Dict[str, Any]:
    """Texture/stress indicators for one azimuthal ring profile (NaN-aware).

    ``coverage`` — fraction of azimuth rows with finite signal.
    ``texture_index`` — ``std(I)/mean(I)`` over finite rows: 0 for an ideal
    powder ring, rising with any azimuthal non-uniformity (texture, stress,
    or spottiness alike).
    ``spotty_frac`` — fraction of finite rows with ``I > 3·median(I)``: a
    single-crystal/coarse-grain spot indicator, distinct from the smooth
    harmonic below.
    ``po_amplitude``/``po_phase_deg`` — preferred-orientation second harmonic
    ``I(φ) ≈ c0 + c2·cos 2(φ−φ2)`` fitted by linear least squares (same
    approach as :func:`seriesxrd.reduce.straighten.fit_waviness`, but on
    intensity rather than ring position). ``po_amplitude = c2/c0``
    (dimensionless; NaN when ``c0 <= 0``); ``po_phase_deg = φ2`` wrapped into
    ``[0, 180)`` (a 2-fold pattern repeats every 180°).

    Returns ``{ok, n, coverage, texture_index, spotty_frac, po_amplitude,
    po_phase_deg}``.
    """
    phi = np.radians(np.asarray(phi_deg, float))
    I = np.asarray(intensity, float)
    fin = np.isfinite(I)
    n = int(fin.sum())
    out: Dict[str, Any] = {
        "ok": False, "n": n,
        "coverage": float(fin.mean()) if I.size else float("nan"),
        "texture_index": float("nan"), "spotty_frac": float("nan"),
        "po_amplitude": float("nan"), "po_phase_deg": float("nan"),
    }
    if n < 3:
        return out
    Iv = I[fin]
    mean = float(np.mean(Iv))
    out["texture_index"] = float(np.std(Iv) / mean) if mean > 0 else float("nan")
    med = float(np.median(Iv))
    out["spotty_frac"] = float(np.mean(Iv > 3.0 * med)) if med > 0 else float("nan")
    if n >= 5:
        P = phi[fin]
        M = np.column_stack([np.ones_like(P), np.cos(2 * P), np.sin(2 * P)])
        sol, *_ = np.linalg.lstsq(M, Iv, rcond=None)
        c0, cc, dd = sol
        c2 = math.hypot(cc, dd)
        phi2 = (math.degrees(math.atan2(dd, cc)) / 2.0) % 180.0
        out["po_amplitude"] = float(c2 / c0) if c0 > 0 else float("nan")
        out["po_phase_deg"] = float(phi2)
        out["ok"] = True
    return out


def _pick_rings(cake: np.ndarray, radial: np.ndarray, dr: float, halfwidth: float,
                n_rings: int) -> List[float]:
    """Auto-pick the ``n_rings`` strongest Debye rings in a cake.

    Same heuristic :func:`seriesxrd.reduce.straighten.straighten_cake` uses when
    ``ring_r0`` isn't given explicitly (replicated here rather than imported —
    it isn't factored out as a standalone helper there): MAD-thresholded
    maxima of the azimuthally-collapsed pattern, spaced apart by about one
    ring window, keeping the strongest ``n_rings`` by peak height.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.nanmean(np.where(cake > 0, cake, np.nan), axis=0)
    prof = np.nan_to_num(prof, nan=0.0)
    from scipy.signal import find_peaks
    med = np.median(prof)
    mad = 1.4826 * np.median(np.abs(prof - med)) or 1.0
    idx, props = find_peaks(prof, height=med + 5 * mad, prominence=3 * mad,
                            distance=max(3, int(halfwidth / dr)))
    if idx.size == 0:
        return []
    order = np.argsort(props["peak_heights"])[::-1][:max(1, int(n_rings))]
    return sorted(float(radial[k]) for k in idx[order])


def run_texture(reduced_h5: "str | Path", *, n_rings: int = 3,
                halfwidth: "Optional[float]" = None, write: bool = True
                ) -> Dict[str, Any]:
    """Azimuthal texture analysis for every cake in a reduced HDF5.

    For each saved cake, auto-picks the ``n_rings`` strongest rings
    (:func:`_pick_rings`) and computes :func:`ring_profile` +
    :func:`texture_metrics` for each. ``halfwidth`` defaults to 8 radial bins
    (same default as :mod:`straighten`). When ``write=True`` the results are
    written atomically (copy to ``.tmp``, edit, ``os.replace``) into a
    ``/texture`` group in ``reduced_h5``, replacing any existing one:

        frame          (C,)     cake's frame index (from /cakes/frame_index)
        ring_r0        (C, R)   picked ring position per cake
        texture_index  (C, R)
        po_amplitude   (C, R)
        po_phase_deg   (C, R)
        spotty_frac    (C, R)
        coverage       (C, R)
        attrs: unit, n_rings, halfwidth

    ``C`` = number of saved cakes (not total frames) — a ``cake_every > 1``
    reduction simply has fewer rows; frames without a saved cake are absent
    rather than NaN-padded. ``R`` = ``n_rings``; a cake with fewer detected
    rings leaves the remaining ring columns NaN. Re-running replaces the
    group (idempotent for unchanged inputs).

    Returns a manifest: the per-cake arrays above, per-ring medians across
    cakes (``ring_medians``), and a ``per_frame`` list of per-ring dicts.
    """
    import h5py  # type: ignore

    src = Path(reduced_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Reduced HDF5 not found: {src}")

    with h5py.File(str(src), "r") as h5:
        g = h5.get("cakes")
        if g is None or "intensity" not in g:
            raise ValueError("No /cakes — re-run reduction with save_cakes=True.")
        unit = str(h5.attrs.get("unit", ""))
        cake_radial = np.asarray(g["radial"][:], float)
        az = np.asarray(g["azimuthal"][:], float)
        fidx = (np.asarray(g["frame_index"][:], int) if "frame_index" in g
               else np.arange(g["intensity"].shape[0]))
        n_cakes = int(g["intensity"].shape[0])
        dr = float(np.median(np.abs(np.diff(cake_radial)))) or 1.0
        hw = float(halfwidth) if halfwidth else 8.0 * dr
        R = max(1, int(n_rings))

        frame = np.full(n_cakes, -1, "i8")
        ring_r0_arr = np.full((n_cakes, R), np.nan)
        texture_index = np.full((n_cakes, R), np.nan)
        po_amplitude = np.full((n_cakes, R), np.nan)
        po_phase_deg = np.full((n_cakes, R), np.nan)
        spotty_frac = np.full((n_cakes, R), np.nan)
        coverage = np.full((n_cakes, R), np.nan)
        per_frame: List[Dict[str, Any]] = []

        for k in range(n_cakes):
            cake = np.asarray(g["intensity"][k], float)
            fr = int(fidx[k])
            frame[k] = fr
            rings = _pick_rings(cake, cake_radial, dr, hw, R)
            row: Dict[str, Any] = {"cake": k, "frame": fr, "rings": []}
            for ridx, r0 in enumerate(rings[:R]):
                prof = ring_profile(cake, cake_radial, az, r0, hw)
                met = texture_metrics(prof["phi"], prof["intensity"])
                ring_r0_arr[k, ridx] = r0
                texture_index[k, ridx] = met["texture_index"]
                po_amplitude[k, ridx] = met["po_amplitude"]
                po_phase_deg[k, ridx] = met["po_phase_deg"]
                spotty_frac[k, ridx] = met["spotty_frac"]
                coverage[k, ridx] = met["coverage"]
                row["rings"].append({"r0": r0, **met})
            per_frame.append(row)

    def _ring_medians(a: np.ndarray) -> List[float]:
        if n_cakes == 0:
            return []
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column ok
            return [float(v) for v in np.nanmedian(a, axis=0)]

    manifest: Dict[str, Any] = {
        "out_h5": str(src), "unit": unit, "n_cakes": n_cakes, "n_rings": R,
        "halfwidth": hw, "frame": frame.tolist(),
        "ring_r0": ring_r0_arr, "texture_index": texture_index,
        "po_amplitude": po_amplitude, "po_phase_deg": po_phase_deg,
        "spotty_frac": spotty_frac, "coverage": coverage,
        "per_frame": per_frame,
        "ring_medians": {
            "texture_index": _ring_medians(texture_index),
            "po_amplitude": _ring_medians(po_amplitude),
            "spotty_frac": _ring_medians(spotty_frac),
            "coverage": _ring_medians(coverage),
        },
    }

    if write:
        import os
        import shutil
        tmp = src.with_name(src.name + ".tmp")
        shutil.copy2(src, tmp)
        try:
            with h5py.File(str(tmp), "r+") as o:
                if "texture" in o:
                    del o["texture"]
                gt = o.create_group("texture")
                gt.create_dataset("frame", data=frame)
                gt.create_dataset("ring_r0", data=ring_r0_arr)
                gt.create_dataset("texture_index", data=texture_index)
                gt.create_dataset("po_amplitude", data=po_amplitude)
                gt.create_dataset("po_phase_deg", data=po_phase_deg)
                gt.create_dataset("spotty_frac", data=spotty_frac)
                gt.create_dataset("coverage", data=coverage)
                gt.attrs["unit"] = unit
                gt.attrs["n_rings"] = R
                gt.attrs["halfwidth"] = hw
                gt.attrs["seriesxrd_version"] = VERSION
                gt.attrs["created_at"] = now_iso()
            os.replace(tmp, src)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    print(f"[TEXTURE] {n_cakes} cakes analyzed, {R} ring(s) each -> /texture "
          f"(median texture_index={manifest['ring_medians']['texture_index']})",
          flush=True)
    return manifest


def main(argv: "list[str] | None" = None) -> int:
    """CLI: ``seriesxrd-texture reduced.h5 [--rings N] [--halfwidth W] [--dry-run]``."""
    import argparse
    p = argparse.ArgumentParser(
        prog="seriesxrd-texture",
        description="Azimuthal texture analysis of the cakes in a reduced HDF5 "
                    "(writes /texture; needs a save_cakes reduction).")
    p.add_argument("reduced", help="Path to a reduced_*.h5 with /cakes.")
    p.add_argument("--rings", type=int, default=3,
                   help="Strongest rings analyzed per cake. Default 3.")
    p.add_argument("--halfwidth", type=float, default=None,
                   help="Ring window half-width (radial units). Default: auto.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report metrics without writing /texture.")
    args = p.parse_args(argv)
    try:
        man = run_texture(args.reduced, n_rings=args.rings,
                          halfwidth=args.halfwidth, write=not args.dry_run)
    except (OSError, ValueError, KeyError) as e:
        print(f"[ERROR] {e}", flush=True)
        return 1
    meds = man.get("ring_medians", {})
    for k in ("texture_index", "po_amplitude", "spotty_frac"):
        print(f"[TEXTURE] median {k}: {meds.get(k)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
