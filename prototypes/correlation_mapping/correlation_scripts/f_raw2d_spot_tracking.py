#!/usr/bin/env python3
"""Definitive static test: track individual SPOT 2theta centroids in the raw 2D
image across pressure.

Why this and not 1D: azimuthal integration turns the pattern into rings and
either averages out or fragments the sharp single-crystal SPOTS (diamond anvil,
gasket grains). Those spots are exactly where a genuinely-static peak hides. In
2D each spot is a compact blob at a well-defined (2theta, chi); its 2theta
centroid is clean (not confused by neighbouring powder peaks). A crystallite
keeps its chi as pressure changes (orientation fixed), so:

  track spots by (chi ~fixed, 2theta within motion tolerance) across the static
  (0deg) series, then judge each spot trajectory's 2theta-vs-P slope:
      slope ~ 0  -> STATIC spot   (fixed d = incompressible/fixed crystal)
      slope > 0  -> MOVING spot   (sample crystallite compressing)

Runs on the static (0deg) series (chi comparable across P; see A_rotcheck).
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

import xrd_geometry as G

OUT = "outputs/analysis_v2_20260701/F_raw2d_spot_tracking"
os.makedirs(OUT, exist_ok=True)

N_RAD = 2500          # fine 2theta bins for centroid precision
N_AZIM = 720          # 0.5 deg chi bins
TT_LO, TT_HI = 4.0, 22.0
SPOT_PCTL = 99.3      # intensity percentile for spot seeds
MAXFILT = (9, 9)      # local-max footprint (chi, 2theta) bins
CHI_FWHM_MAX = 18.0   # deg: a spot must be chi-localized (else it's a ring)
LINK_CHI_TOL = 8.0    # deg
LINK_TT_TOL = 0.25    # deg (allow pressure motion between frames)
MIN_FRAMES = 4
MIN_DP = 3.0
MOTION_THRESH = 0.008 # deg/GPa
STATIC_SLOPE_MAX = 0.005


def detect_spots(I, rad, chi):
    """Return list of (2theta_centroid, chi, intensity) for chi-localized spots."""
    m = (rad >= TT_LO) & (rad <= TT_HI)
    sub = I[:, m]
    radm = rad[m]
    pos = sub[sub > 0]
    if pos.size < 50:
        return []
    thr = np.percentile(pos, SPOT_PCTL)
    mx = maximum_filter(sub, size=MAXFILT)
    seeds = np.argwhere((sub == mx) & (sub > thr))
    dchi = float(np.median(np.diff(chi)))
    nchi, ntt = sub.shape
    spots = []
    for ic, it in seeds:
        # reject spots adjacent to masked bins (module-gap edge glow would
        # otherwise look static because the gap is at a fixed detector position)
        cl, cr = max(0, it - 3), min(ntt, it + 4)
        rl, rr = ic - 3, ic + 4
        chi_win = np.take(sub[:, cl:cr], range(rl, rr), axis=0, mode="wrap")
        if np.any(chi_win <= 0):
            continue
        # 2theta centroid in a small window
        lo, hi = max(0, it - 3), min(sub.shape[1], it + 4)
        w = sub[ic, lo:hi].astype(float)
        w = np.clip(w - w.min(), 0, None)
        if w.sum() <= 0:
            continue
        tt_cen = float(np.sum(radm[lo:hi] * w) / w.sum())
        # chi FWHM at this 2theta column (localization test)
        col = sub[:, it].astype(float)
        peak = col[ic]
        if peak <= 0:
            continue
        half = peak / 2.0
        # walk out in chi from ic until below half-max (wrap-around)
        n = len(col)
        left = 0
        for s in range(1, n // 2):
            if col[(ic - s) % n] < half:
                left = s
                break
        right = 0
        for s in range(1, n // 2):
            if col[(ic + s) % n] < half:
                right = s
                break
        fwhm_chi = (left + right) * dchi
        if fwhm_chi == 0 or fwhm_chi > CHI_FWHM_MAX:
            continue  # ring / too broad -> not a single-crystal spot
        spots.append((tt_cen, float(chi[ic]), float(peak), fwhm_chi))
    return spots


def link_spots(frame_spots, pressures):
    """Greedy tracking by (chi within tol, 2theta within motion tol)."""
    trajs = []  # each: list of (fi, tt, chi, I)
    for fi, spots in enumerate(frame_spots):
        used = set()
        # try to extend existing open trajectories
        for tr in trajs:
            if tr[-1][0] == fi - 1 or fi - tr[-1][0] <= 2:
                lt = tr[-1]
                best, bj = None, -1
                for j, (tt, chi, I, _) in enumerate(spots):
                    if j in used:
                        continue
                    if abs(chi - lt[2]) <= LINK_CHI_TOL and abs(tt - lt[1]) <= LINK_TT_TOL:
                        d = abs(tt - lt[1]) + abs(chi - lt[2]) / 50.0
                        if best is None or d < best:
                            best, bj = d, j
                if bj >= 0:
                    tt, chi, I, _ = spots[bj]
                    tr.append((fi, tt, chi, I))
                    used.add(bj)
        for j, (tt, chi, I, _) in enumerate(spots):
            if j not in used:
                trajs.append([(fi, tt, chi, I)])
    return trajs


def fit_slope(P, tt):
    P = np.asarray(P, float); tt = np.asarray(tt, float)
    if len(P) < 3 or np.ptp(P) == 0:
        return np.nan, np.nan, np.nan
    (a, b), cov = np.polyfit(P, tt, 1, cov=True)
    se = float(np.sqrt(cov[0, 0]))
    return float(a), se, 1.96 * se


def verdict(slope, ci, n, dP):
    if n < MIN_FRAMES or dP < MIN_DP or not np.isfinite(slope):
        return "UNDETERMINED"
    lo, hi = slope - ci, slope + ci
    if slope > MOTION_THRESH and lo > 0:
        return "MOVING"
    if abs(slope) < STATIC_SLOPE_MAX and hi < MOTION_THRESH and lo > -MOTION_THRESH:
        return "STATIC"
    return "UNDETERMINED"


def main():
    ai = G.load_ai()
    wl = G.wavelength_A(ai)
    ref = G.reference_lines(wl)
    rows = []
    for cell in ("Cell_29", "Cell_14"):
        inv = G.inventory()
        series = [f for f in G.select_series(inv, cell) if f.static and not f.decomp]
        pressures, frame_spots = [], []
        for fr in series:
            img = G.read_image(fr.path)
            mask = G.build_mask(img, ai)
            I, rad, chi = G.cake(img, ai, mask, n_rad=N_RAD, n_azim=N_AZIM)
            frame_spots.append(detect_spots(I, rad, chi))
            pressures.append(fr.pressure)
        trajs = link_spots(frame_spots, pressures)
        cell_rows = []
        plot_trajs = []  # (verdict, P, tt) for multi-frame tracks
        for tr in trajs:
            P = [pressures[p[0]] for p in tr]
            tt = [p[1] for p in tr]
            chi = np.median([p[2] for p in tr])
            n = len(tr); dP = float(np.ptp(P))
            slope, se, ci = fit_slope(P, tt)
            v = verdict(slope, ci, n, dP)
            med_tt = float(np.median(tt))
            r = min(ref, key=lambda r: abs(r["two_theta"] - med_tt))
            rd = abs(r["two_theta"] - med_tt)
            cell_rows.append(dict(cell=cell, median_2theta=round(med_tt, 4),
                                  d_A=round(float(G.d_from_2theta(med_tt, wl)), 4),
                                  chi=round(float(chi), 1), n_frames=n, dP_GPa=round(dP, 1),
                                  span_deg=round(float(np.ptp(tt)), 4),
                                  slope=round(slope, 5) if np.isfinite(slope) else "",
                                  slope_95CI=round(ci, 5) if np.isfinite(ci) else "",
                                  ref=(f"{r['phase']}{r['hkl']}" if rd <= 0.15 else ""),
                                  ref_fixed=(r["fixed"] if rd <= 0.15 else ""),
                                  verdict=v))
            if n >= MIN_FRAMES:
                plot_trajs.append((v, list(P), list(tt)))
        # validation figure: 2theta vs P for all >=4-frame spot tracks
        if plot_trajs:
            fig, ax = plt.subplots(figsize=(9, 6))
            col = {"STATIC": "tab:blue", "MOVING": "tab:green", "UNDETERMINED": "0.8"}
            for v, P, tt in plot_trajs:
                ax.plot(P, tt, "-o", ms=3, lw=0.8, color=col.get(v, "0.8"),
                        alpha=0.9 if v != "UNDETERMINED" else 0.4,
                        zorder=3 if v == "STATIC" else 2)
            for lab, c in [("STATIC (fixed 2th)", "tab:blue"),
                           ("MOVING (sample)", "tab:green"),
                           ("undetermined", "0.8")]:
                ax.plot([], [], "-o", color=c, label=lab)
            ax.set_xlabel("pressure (GPa)")
            ax.set_ylabel("spot 2theta (deg)")
            ax.set_title(f"{cell}: raw-2D spot 2theta vs pressure. "
                         f"flat line = STATIC peak; rising = compressing sample.")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{OUT}/{cell}_spot_2theta_vs_pressure.png", dpi=150)
            plt.close(fig)
        cell_rows.sort(key=lambda r: (r["verdict"], r["median_2theta"]))
        rows.extend(cell_rows)
        from collections import Counter
        c = Counter(r["verdict"] for r in cell_rows)
        n_multi = sum(1 for r in cell_rows if r["n_frames"] >= MIN_FRAMES)
        print(f"=== {cell}: {len(cell_rows)} spot tracks ({n_multi} with >={MIN_FRAMES} frames) ===")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"   {v:3d}  {k}")
        for r in cell_rows:
            if r["verdict"] == "STATIC":
                print(f"   STATIC: 2th={r['median_2theta']:.3f} d={r['d_A']:.4f} chi={r['chi']} "
                      f"n={r['n_frames']} dP={r['dP_GPa']} slope={r['slope']}+/-{r['slope_95CI']} "
                      f"span={r['span_deg']} ref={r['ref']}")
    with open(f"{OUT}/spot_static_verdicts.csv", "w", newline="") as h:
        cols = ["cell", "median_2theta", "d_A", "chi", "n_frames", "dP_GPa", "span_deg",
                "slope", "slope_95CI", "ref", "ref_fixed", "verdict"]
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote spot_static_verdicts.csv to {OUT}/")


if __name__ == "__main__":
    main()
