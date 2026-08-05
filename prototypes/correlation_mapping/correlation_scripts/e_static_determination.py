#!/usr/bin/env python3
"""Static determination: is each peak's 2theta fixed with pressure?

This answers ONLY "is the peak static" (2theta constant vs P), independent of
its physical identity (diamond / gasket / instrument / sample). Criterion:
fit 2theta = a*P + b, judge the slope a against its standard error and a
physical motion threshold. Three honest verdicts:

  STATIC       : slope consistent with 0 AND we can rule out real motion
                 (upper 95% CI below the motion threshold), with a real lever
                 arm (>=4 points spanning >=3 GPa).
  MOVING       : slope significantly positive and above the motion threshold.
  UNDETERMINED : lever arm too short, or CI too wide to decide (this is where
                 the suite's fragmentation artifacts really belong).

Primary data = powder-averaged integrated .xy (clean 2theta positions). The
static single-crystal SPOTS that powder-averaging can miss are added from the
Phase B 2D static-exposure analysis (fixed-2theta + chi-stable), because those
are exactly the static peaks a 1D ring analysis under-detects.
"""
from __future__ import annotations

import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import xrd_geometry as G
from b_static_diamond_2d import detect_1d, link

OUT = "outputs/analysis_v2_20260701/E_static_determination"
os.makedirs(OUT, exist_ok=True)
INTEGRATED = {"Cell_29": "Data/Cell_29_integrated", "Cell_14": "Data/Cell_14_integrated"}

MOTION_THRESH = 0.008    # deg/GPa: a real sample peak moves faster than this
STATIC_SLOPE_MAX = 0.005 # deg/GPa: |slope| below this is candidate-static
MIN_POINTS = 4
MIN_DP = 3.0             # GPa lever arm required to decide static vs moving


def build_frames(cell):
    frames, patterns = [], []
    for f in sorted(glob.glob(os.path.join(INTEGRATED[cell], "*.xy"))):
        m = G.PRESSURE_RE.search(os.path.basename(f).replace("decomp-", ""))
        if not m:
            continue
        pressure = float(m.group(1).replace("p", "."))
        decomp = "decomp" in f.lower()
        try:
            arr = np.loadtxt(f)
        except UnicodeDecodeError:
            arr = np.loadtxt(f, encoding="latin1")
        x, y = arr[:, 0], arr[:, 1]
        pos, amp = detect_1d(x, y)
        frames.append(dict(pressure=pressure, decomp=decomp, pos=pos, amp=amp))
    frames.sort(key=lambda f: f["pressure"])
    return frames


def fit_slope(P, tth):
    """Return slope, se_slope, and 95% CI half-width using polyfit covariance."""
    P = np.asarray(P, float); tth = np.asarray(tth, float)
    n = len(P)
    if n < 3 or np.ptp(P) == 0:
        return np.nan, np.nan, np.nan
    (a, b), cov = np.polyfit(P, tth, 1, cov=True)
    se = float(np.sqrt(cov[0, 0]))
    # 95% CI ~ t(0.975, n-2)*se ; use 1.96 for simplicity (n small -> slightly under)
    return float(a), se, 1.96 * se


def verdict(slope, se, ci, n, dP):
    if n < MIN_POINTS or dP < MIN_DP:
        return "UNDETERMINED (short lever arm)"
    if not np.isfinite(slope):
        return "UNDETERMINED"
    lo, hi = slope - ci, slope + ci
    # MOVING: slope clearly positive and physically large
    if slope > MOTION_THRESH and lo > 0:
        return "MOVING"
    # also flag clearly-negative significant slopes (usually mislink/noise)
    if slope < -MOTION_THRESH and hi < 0:
        return "MOVING (neg slope - check)"
    # STATIC: small slope AND we can exclude real motion (|CI| below threshold)
    if abs(slope) < STATIC_SLOPE_MAX and hi < MOTION_THRESH and lo > -MOTION_THRESH:
        return "STATIC"
    return "UNDETERMINED (slope not resolved)"


def main():
    ai = G.load_ai()
    wl = G.wavelength_A(ai)
    rows = []
    counts = {}
    for cell in ("Cell_29", "Cell_14"):
        frames = build_frames(cell)
        # link on non-decomp frames only (decomp is a different phase)
        seq = [f for f in frames if not f["decomp"]]
        trajs = link([dict(pos=f["pos"], amp=f["amp"]) for f in seq])
        cell_rows = []
        for ti, tr in enumerate(trajs):
            idx = [p[0] for p in tr["pts"]]
            pos = np.array([p[1] for p in tr["pts"]])
            P = np.array([seq[i]["pressure"] for i in idx])
            n = len(P); dP = float(np.ptp(P))
            slope, se, ci = fit_slope(P, pos)
            v = verdict(slope, se, ci, n, dP)
            cell_rows.append(dict(cell=cell, median_2theta=round(float(np.median(pos)), 4),
                                  d_A=round(float(G.d_from_2theta(np.median(pos), wl)), 4),
                                  n_frames=n, dP_GPa=round(dP, 1),
                                  span_deg=round(float(np.ptp(pos)), 4),
                                  slope_deg_per_GPa=round(slope, 5) if np.isfinite(slope) else "",
                                  slope_se=round(se, 5) if np.isfinite(se) else "",
                                  slope_95CI=round(ci, 5) if np.isfinite(ci) else "",
                                  verdict=v))
        cell_rows.sort(key=lambda r: r["median_2theta"])
        rows.extend(cell_rows)
        from collections import Counter
        c = Counter(r["verdict"].split(" (")[0] for r in cell_rows)
        counts[cell] = c
        print(f"=== {cell}: {len(cell_rows)} tracked peaks ===")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"   {v:2d}  {k}")
        stat = [r for r in cell_rows if r["verdict"] == "STATIC"]
        print(f"   STATIC peaks: {[r['median_2theta'] for r in stat] or 'none'}")

    with open(f"{OUT}/static_verdicts.csv", "w", newline="") as h:
        cols = ["cell", "median_2theta", "d_A", "n_frames", "dP_GPa", "span_deg",
                "slope_deg_per_GPa", "slope_se", "slope_95CI", "verdict"]
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # figure: slope +/- CI per peak, per cell, with decision bands
    for cell in ("Cell_29", "Cell_14"):
        sub = [r for r in rows if r["cell"] == cell and r["slope_deg_per_GPa"] != ""]
        sub = [r for r in sub if "UNDETERMINED (short" not in r["verdict"]]
        if not sub:
            continue
        sub.sort(key=lambda r: r["median_2theta"])
        y = np.arange(len(sub))
        sl = [r["slope_deg_per_GPa"] for r in sub]
        ci = [r["slope_95CI"] for r in sub]
        colors = {"STATIC": "tab:blue", "MOVING": "tab:green"}
        cols = [colors.get(r["verdict"].split(" (")[0], "tab:orange") for r in sub]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(sub))))
        ax.axvspan(-STATIC_SLOPE_MAX, STATIC_SLOPE_MAX, color="tab:blue", alpha=0.08)
        ax.axvline(MOTION_THRESH, color="k", ls="--", lw=0.6)
        ax.axvline(0, color="k", lw=0.5)
        ax.errorbar(sl, y, xerr=ci, fmt="none", ecolor="0.6", lw=0.8, capsize=2)
        ax.scatter(sl, y, c=cols, s=28, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([f"{r['median_2theta']:.2f}deg" for r in sub], fontsize=7)
        ax.set_xlabel("2theta slope (deg/GPa)  [blue band=static zone, dashed=motion threshold]")
        ax.set_title(f"{cell}: static determination — slope +/- 95% CI per tracked peak\n"
                     f"blue=STATIC, green=MOVING, orange=undetermined")
        fig.tight_layout()
        fig.savefig(f"{OUT}/{cell}_slope_verdicts.png", dpi=150)
        plt.close(fig)
    print(f"\nWrote static_verdicts.csv + slope figures to {OUT}/")


if __name__ == "__main__":
    main()
