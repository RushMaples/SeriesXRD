#!/usr/bin/env python3
"""Better window-comparison metrics than ACF-Pearson (for suite modes 02/03).

The ACF fingerprint is translation-invariant, so it can't tell "same peaks" from
"peaks at different positions" (proven: single-peak-diff-position -> 0.92). For
comparing 5-degree windows we want a metric that (a) is POSITION sensitive, (b)
handles the pressure-induced shift explicitly, (c) is robust to intensity texture.

Candidates compared here:
  ACF_pearson  : current suite metric (baseline).
  cosine0      : cosine similarity of bg-subtracted windows at zero lag
                 (position sensitive, no shift allowance).
  NCC_shift    : normalized cross-correlation MAX over a bounded lag search
                 -> similarity AND the best-fit shift (physically = peak motion).
                 This is the recommended replacement: shift-aware but still
                 penalizes genuinely different peak arrangements.
  peakJaccard  : detect peaks in each window, Jaccard of positions within tol
                 (intensity-robust; coarse for narrow windows).

Part A: controls (synthetic) showing which metrics discriminate.
Part B: real same-window-across-pressure (mode-02 replacement) with NCC_shift,
        which yields a similarity that decays with pressure AND recovers the
        window's peak shift (compression) as the best lag.
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import xrd_geometry as G
import window_autocorrelation_correlations as W

OUT = "outputs/analysis_v2_20260701/H_window_metrics"
os.makedirs(OUT, exist_ok=True)
STEP = 0.02  # deg/bin for synthetic + resampled real windows


def zmean(a):
    a = np.asarray(a, float)
    return a - a.mean()


def cosine0(a, b):
    a, b = zmean(a), zmean(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def ncc_shift(a, b, max_lag_bins):
    """Normalized cross-correlation max over +/-max_lag; returns (sim, lag_bins)."""
    a, b = zmean(a), zmean(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan, 0
    full = np.correlate(a, b, mode="full") / (na * nb)
    lags = np.arange(-len(b) + 1, len(a))
    keep = np.abs(lags) <= max_lag_bins
    idx = np.argmax(full[keep])
    return float(full[keep][idx]), int(lags[keep][idx])


def acf_pearson(a, b):
    fa = W.autocorrelation_fingerprint(zmean(a))
    fb = W.autocorrelation_fingerprint(zmean(b))
    return W.pearson(fa, fb)


def peak_jaccard(a, b, x, tol=0.08):
    pa = x[find_peaks(a, prominence=0.15 * (a.max() or 1))[0]]
    pb = x[find_peaks(b, prominence=0.15 * (b.max() or 1))[0]]
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    m = 0
    pb2 = list(pb)
    for p in pa:
        for q in pb2:
            if abs(p - q) <= tol:
                m += 1; pb2.remove(q); break
    return m / (len(pa) + len(pb) - m)


def gauss(x, c, w, amp=1.0):
    return amp * np.exp(-0.5 * ((x - c) / w) ** 2)


def controls():
    x = np.arange(0, 5, STEP)
    max_lag = int(round(0.4 / STEP))
    rng = np.random.default_rng(0)
    cases = {}
    # single peak, same position
    a = gauss(x, 2.5, 0.05); cases["1peak same pos"] = (a, gauss(x, 2.5, 0.05))
    # single peak shifted 0.15 deg (pressure shift)
    cases["1peak shift 0.15"] = (a, gauss(x, 2.65, 0.05))
    # single peak very different position  (the ACF killer)
    cases["1peak diff pos"] = (gauss(x, 1.5, 0.05), gauss(x, 3.5, 0.06))
    # 3 peaks same
    m = gauss(x, 1.5, .05) + gauss(x, 2.5, .05) + gauss(x, 3.5, .05)
    cases["3peaks same"] = (m, m.copy())
    # 3 peaks all shifted +0.12 (uniform compression)
    ms = gauss(x, 1.62, .05) + gauss(x, 2.62, .05) + gauss(x, 3.62, .05)
    cases["3peaks shift 0.12"] = (m, ms)
    # 3 peaks at different positions (same count/spacing-ish, different absolute)
    md = gauss(x, 1.2, .05) + gauss(x, 2.15, .05) + gauss(x, 3.05, .05)
    cases["3peaks diff pos"] = (m, md)
    # pure noise
    cases["noise vs noise"] = (rng.standard_normal(len(x)), rng.standard_normal(len(x)))
    print("=== Part A: metric behaviour on controls (want HIGH only for 'same') ===")
    print(f"{'case':22} {'ACF_pearson':>12} {'cosine0':>9} {'NCC_shift':>10} {'lag(deg)':>9} {'peakJacc':>9}")
    for name, (a, b) in cases.items():
        s, lag = ncc_shift(a, b, max_lag)
        print(f"{name:22} {acf_pearson(a,b):12.2f} {cosine0(a,b):9.2f} "
              f"{s:10.2f} {lag*STEP:9.2f} {peak_jaccard(a,b,x):9.2f}")


def load_cell(cell):
    frames = []
    for f in sorted(glob.glob(f"Data/{cell}_integrated/*.xy")):
        m = G.PRESSURE_RE.search(os.path.basename(f).replace("decomp-", ""))
        if not m or "decomp" in f.lower():
            continue
        P = float(m.group(1).replace("p", "."))
        try:
            arr = np.loadtxt(f)
        except UnicodeDecodeError:
            arr = np.loadtxt(f, encoding="latin1")
        frames.append((P, arr[:, 0], arr[:, 1]))
    frames.sort(key=lambda t: t[0])
    return frames


def window_signal(x, y, lo, hi):
    grid = np.arange(lo, hi, STEP)
    yi = np.interp(grid, x, y)
    from scipy.signal import savgol_filter
    n = len(yi)
    sm = savgol_filter(yi, min(9, n // 2 * 2 - 1), 2)
    base = savgol_filter(yi, min(51, n // 2 * 2 - 1), 2)
    sig = np.clip(sm - base, 0, None)
    return grid, sig


def real_same_window(cell="Cell_29", lo=9.0, hi=14.0):
    frames = load_cell(cell)
    ref_P, rx, ry = frames[0]
    grid, ref = window_signal(rx, ry, lo, hi)
    max_lag = int(round(0.6 / STEP))
    print(f"\n=== Part B: mode-02 replacement, {cell} window {lo}-{hi} deg, "
          f"NCC-shift vs {ref_P:g}GPa reference ===")
    print(f"{'P(GPa)':>7} {'ACF':>6} {'NCC_sim':>8} {'shift(deg)':>11}")
    Ps, sims, shifts, acfs = [], [], [], []
    for P, x, y in frames:
        _, sig = window_signal(x, y, lo, hi)
        s, lag = ncc_shift(ref, sig, max_lag)
        a = acf_pearson(ref, sig)
        print(f"{P:7.1f} {a:6.2f} {s:8.2f} {lag*STEP:11.3f}")
        Ps.append(P); sims.append(s); shifts.append(lag * STEP); acfs.append(a)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(Ps, sims, "o-", label="NCC similarity (shift-aware)")
    ax1.plot(Ps, acfs, "s--", color="0.6", label="ACF-Pearson (old)")
    ax1.set_xlabel("pressure (GPa)"); ax1.set_ylabel("similarity to lowest-P frame")
    ax1.set_title(f"{cell} {lo}-{hi}deg: similarity vs P"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(Ps, shifts, "o-", color="tab:red")
    ax2.set_xlabel("pressure (GPa)"); ax2.set_ylabel("best-fit 2theta shift (deg)")
    ax2.set_title("recovered peak shift (compression) = the physics ACF hides")
    ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/{cell}_window_{lo:g}_{hi:g}_NCC.png", dpi=150)
    plt.close(fig)
    print(f"wrote {cell}_window_{lo:g}_{hi:g}_NCC.png")


if __name__ == "__main__":
    controls()
    real_same_window()
