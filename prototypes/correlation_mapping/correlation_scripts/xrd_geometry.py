#!/usr/bin/env python3
"""Shared XRD geometry / masking / frame-inventory helpers.

Single source of truth for the v2 analysis (per-cell 2D static-vs-moving work,
d-spacing / EOS, per-cell correlation). Keeps Phases B/C/D from duplicating
PONI loading, Bragg conversion, detector masking and filename parsing.

Detector here is a Pilatus CdTe 1M (1043x981, 172 um) at lambda = 0.4133 A
(30 keV). TIFFs store dead/hot pixels as int32 sentinels (+/- ~2.1e9) and
inter-module gaps as <=0, so a real mask must combine three things:
  1. the detector module-gap mask (pyFAI detector.calc_mask()),
  2. non-positive pixels (gaps / beamstop / unexposed),
  3. saturated / sentinel pixels (|value| above a sane ceiling).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pyFAI
import tifffile

DEFAULT_PONI = "Data/Calibration/CeO2_30keV_168mm_0deg_001.poni"

# Diamond is cubic (Fd-3m), a = 3.567 A. Allowed reflections: h,k,l all even or
# all odd, with 2 not allowed for the "all even" family unless h+k+l = 4n.
DIAMOND_A = 3.567
DIAMOND_HKL = [(1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0), (3, 3, 1)]

# Neon (fcc, a ~ 4.43 A at low P) and rhenium gasket (hcp) reference d-spacings
# are pressure dependent; we only hard-code diamond (nearly incompressible) as a
# fixed anchor. Ne/gasket lines are drawn from their ambient cells and flagged
# as "shifts with P".
NE_A0 = 4.43  # A, approx ambient-ish fcc Ne (will compress; indicative only)
NE_HKL = [(1, 1, 1), (2, 0, 0), (2, 2, 0)]
RE_HCP = dict(a=2.761, c=4.456)  # rhenium gasket, ambient
RE_HKL = [(1, 0, 0), (0, 0, 2), (1, 0, 1), (1, 0, 2), (1, 1, 0)]


def load_ai(poni: str = DEFAULT_PONI):
    return pyFAI.load(poni)


def wavelength_A(ai) -> float:
    return float(ai.wavelength) * 1e10


def d_from_2theta(two_theta_deg, wavelength_a: float) -> np.ndarray:
    """Bragg: d = lambda / (2 sin(theta)), theta = 2theta/2."""
    tt = np.deg2rad(np.asarray(two_theta_deg, dtype=float))
    return wavelength_a / (2.0 * np.sin(tt / 2.0))


def two_theta_from_d(d_a, wavelength_a: float) -> np.ndarray:
    d = np.asarray(d_a, dtype=float)
    arg = np.clip(wavelength_a / (2.0 * d), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arcsin(arg))


def cubic_d(a: float, hkl) -> float:
    h, k, l = hkl
    return a / np.sqrt(h * h + k * k + l * l)


def hcp_d(a: float, c: float, hkl) -> float:
    h, k, l = hkl
    inv = (4.0 / 3.0) * (h * h + h * k + k * k) / (a * a) + (l * l) / (c * c)
    return 1.0 / np.sqrt(inv)


def reference_lines(wavelength_a: float) -> list[dict]:
    """Known-phase 2theta lines for identifying static contributors.

    diamond: fixed (nearly incompressible) -> the anchor for 'is it diamond'.
    Ne / Re: pressure-dependent, drawn from ambient cells as indicative only.
    """
    lines = []
    for hkl in DIAMOND_HKL:
        d = cubic_d(DIAMOND_A, hkl)
        lines.append(dict(phase="diamond", hkl=hkl, d=d,
                          two_theta=float(two_theta_from_d(d, wavelength_a)),
                          fixed=True))
    for hkl in NE_HKL:
        d = cubic_d(NE_A0, hkl)
        lines.append(dict(phase="Ne(ambient)", hkl=hkl, d=d,
                          two_theta=float(two_theta_from_d(d, wavelength_a)),
                          fixed=False))
    for hkl in RE_HKL:
        d = hcp_d(RE_HCP["a"], RE_HCP["c"], hkl)
        lines.append(dict(phase="Re-gasket(ambient)", hkl=hkl, d=d,
                          two_theta=float(two_theta_from_d(d, wavelength_a)),
                          fixed=False))
    return lines


def build_mask(img: np.ndarray, ai, sat_ceiling: float = 1e7) -> np.ndarray:
    """Combined mask: True = ignore this pixel.

    module gaps (detector.calc_mask) | non-positive | saturated/sentinel.
    """
    m = np.zeros(img.shape, dtype=bool)
    try:
        dm = ai.detector.calc_mask()
        if dm is not None and dm.shape == img.shape:
            m |= dm.astype(bool)
    except Exception:
        pass
    m |= (img <= 0)
    m |= (np.abs(img) >= sat_ceiling)
    return m


def read_image(path: str) -> np.ndarray:
    return tifffile.imread(path).astype(np.float32)


def cake(img: np.ndarray, ai, mask=None, n_rad: int = 1400, n_azim: int = 360,
         unit: str = "2th_deg"):
    """2D unrolled (azimuth x 2theta) intensity plus axes. Returns (I, rad, chi)."""
    if mask is None:
        mask = build_mask(img, ai)
    res = ai.integrate2d(img, n_rad, n_azim, unit=unit, mask=mask)
    return res.intensity, res.radial, res.azimuthal


def integrate1d(img: np.ndarray, ai, mask=None, n_rad: int = 3000,
                unit: str = "2th_deg"):
    if mask is None:
        mask = build_mask(img, ai)
    res = ai.integrate1d(img, n_rad, unit=unit, mask=mask)
    return res.radial, res.intensity


# ---------------------------------------------------------------------------
# Frame inventory: parse pressure + exposure geometry from filenames.
# Cell_29/Cell_14 store several exposures per pressure: a static "0deg" frame
# and rocked/oscillated "-5deg"/"-10deg[_rot]" frames (stage rocked to powder-
# average single-crystal spots). Only same-oscillation frames have comparable
# azimuth (chi), which matters for the diamond chi-fixedness test.
# ---------------------------------------------------------------------------
PRESSURE_RE = re.compile(r"(\d+p\d+|\d+)\s*G[PO]a", re.I)  # tolerate 'GOa' typo
OSC_RE = re.compile(r"(\d+)\s*deg", re.I)
SERIAL_RE = re.compile(r"_(\d+)\.tif$", re.I)


@dataclass
class Frame:
    path: str
    cell: str
    pressure: float
    label: str          # e.g. '9.8GPa'
    osc_deg: int        # 0 = static, else rocking range in degrees
    rot: bool           # 'rot' token present
    decomp: bool
    serial: int

    @property
    def static(self) -> bool:
        return self.osc_deg == 0


def parse_frame(path: str) -> Frame | None:
    base = os.path.basename(path)
    cell = "Cell_14" if "Cell_14" in path else ("Cell_29" if "Cell_29" in path else "all")
    # pressure: prefer the value in the *filename* (folder can be mislabeled,
    # e.g. '12p0 GOa' holding UOTe-12p8GPa files).
    pm = PRESSURE_RE.search(base)
    if not pm:
        return None
    pressure = float(pm.group(1).replace("p", "."))
    osc = OSC_RE.search(base)
    osc_deg = int(osc.group(1)) if osc else 0
    sm = SERIAL_RE.search(base)
    serial = int(sm.group(1)) if sm else -1
    decomp = "decomp" in path.lower() or "noNe" in base
    label = f"{pressure:g}GPa" + ("_decomp" if decomp else "")
    return Frame(path=path, cell=cell, pressure=pressure, label=label,
                 osc_deg=osc_deg, rot=("rot" in base.lower()), decomp=decomp,
                 serial=serial)


def inventory(root: str = "Data", cells=("Cell_14", "Cell_29")) -> list[Frame]:
    frames = []
    for cell in cells:
        for p in sorted(glob.glob(os.path.join(root, cell, "**", "*.tif"), recursive=True)):
            fr = parse_frame(p)
            if fr:
                frames.append(fr)
    return frames


def select_series(frames: list[Frame], cell: str, prefer_static: bool = True) -> list[Frame]:
    """One frame per (pressure,decomp) for a cell, preferring static (0deg).

    For the diamond chi-fixedness test we want a consistent-orientation series;
    static exposures preserve sharp fixed-chi spots and are all at 0deg.
    Falls back to the lowest-oscillation frame when no static one exists.
    """
    by_key: dict[tuple, Frame] = {}
    for fr in frames:
        if fr.cell != cell:
            continue
        key = (round(fr.pressure, 3), fr.decomp)
        cand = by_key.get(key)
        if cand is None:
            by_key[key] = fr
            continue
        if prefer_static:
            # prefer smaller oscillation, then lower serial for determinism
            if (fr.osc_deg, fr.serial) < (cand.osc_deg, cand.serial):
                by_key[key] = fr
    return [by_key[k] for k in sorted(by_key)]


if __name__ == "__main__":
    ai = load_ai()
    wl = wavelength_A(ai)
    print(f"PONI={DEFAULT_PONI}  lambda={wl:.4f} A  detector={ai.detector}")
    print("\nReference lines (2theta deg):")
    for r in reference_lines(wl):
        tag = "FIXED" if r["fixed"] else "shifts-with-P"
        print(f"  {r['phase']:20s} {str(r['hkl']):10s} d={r['d']:.4f}A  2th={r['two_theta']:6.3f}  [{tag}]")
    inv = inventory()
    print(f"\nInventory: {len(inv)} tiff frames")
    for cell in ("Cell_14", "Cell_29"):
        ser = select_series(inv, cell)
        print(f"  {cell}: {sum(f.cell==cell for f in inv)} exposures -> {len(ser)} selected series frames")
