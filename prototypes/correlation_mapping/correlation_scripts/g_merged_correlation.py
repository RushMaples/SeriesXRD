#!/usr/bin/env python3
"""Corrected MERGED frame correlation for the combined UOTe series (Cell_14 +
Cell_29 treated as one material, all 17 frames).

Why not the suite's 01-04:
  - 01 uses per-peak ROI-area agreement -> intensity based -> not reproducible
    for spotty single-crystal-like data, and runs on ~625 over-detected groups.
  - 02-04 use Pearson of per-window autocorrelation "fingerprints". The ACF is
    translation-invariant, so two windows each containing a peak score ~0.9
    REGARDLESS of peak position (verified: single-peak-diff-position -> 0.92,
    random noise -> 0.0). Its floor is ~0.6-0.9 whenever peaks are present, so
    within/cross/unrelated all collapse to 0.66-0.79 = no real discrimination.

The only texture-robust, position-preserving, cross-cell-comparable observable
is PEAK POSITION (d-spacing). This script:
  1. merges both cells, detects peaks, converts to d-spacing;
  2. frame-vs-frame similarity = position Jaccard (peaks matching within a small
     2theta tolerance) -> high at similar pressure, decays as peaks compress;
  3. tests the merge directly: do cross-cell frames at SIMILAR pressure match as
     well as within-cell neighbours? and do all peaks fall on common d(P) curves?
"""
from __future__ import annotations

import csv
import glob
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import xrd_geometry as G
from b_static_diamond_2d import detect_1d

OUT = "outputs/analysis_v2_20260701/G_merged_correlation"
os.makedirs(OUT, exist_ok=True)
INTEGRATED = {"Cell_29": "Data/Cell_29_integrated", "Cell_14": "Data/Cell_14_integrated"}
TOL_2TH = 0.05   # deg: two peaks count as the same position within this


def load_all():
    frames = []
    for cell, d in INTEGRATED.items():
        for f in sorted(glob.glob(os.path.join(d, "*.xy"))):
            m = G.PRESSURE_RE.search(os.path.basename(f).replace("decomp-", ""))
            if not m:
                continue
            P = float(m.group(1).replace("p", "."))
            decomp = "decomp" in f.lower()
            try:
                arr = np.loadtxt(f)
            except UnicodeDecodeError:
                arr = np.loadtxt(f, encoding="latin1")
            pos, amp = detect_1d(arr[:, 0], arr[:, 1])
            frames.append(dict(cell=cell, P=P, decomp=decomp,
                               label=f"{P:g}{'d' if decomp else ''}[{cell[-2:]}]",
                               pos=np.sort(pos)))
    frames.sort(key=lambda f: (f["P"], f["cell"]))
    return frames


def jaccard(a, b, tol=TOL_2TH):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    i = j = match = 0
    a = np.sort(a); b = np.sort(b)
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tol:
            match += 1; i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return match / (len(a) + len(b) - match)


def main():
    ai = G.load_ai(); wl = G.wavelength_A(ai)
    frames = load_all()
    n = len(frames)
    labels = [f["label"] for f in frames]
    cells = [f["cell"] for f in frames]
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            M[i, j] = jaccard(frames[i]["pos"], frames[j]["pos"])

    # heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6)
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5,
                        color="w" if M[i, j] < 0.5 else "k")
    fig.colorbar(im, ax=ax, fraction=0.046, label="peak-position Jaccard")
    ax.set_title("Merged UOTe (Cell_14+Cell_29): peak-POSITION Jaccard frame correlation\n"
                 "[XX]=cell; texture-robust; high=same peak positions")
    fig.tight_layout(); fig.savefig(f"{OUT}/merged_position_jaccard.png", dpi=150); plt.close(fig)

    # within vs cross, and the crucial "cross-cell at similar pressure" test
    within, cross, cross_closeP = [], [], []
    for i, j in itertools.combinations(range(n), 2):
        if frames[i]["decomp"] or frames[j]["decomp"]:
            continue
        v = M[i, j]; dP = abs(frames[i]["P"] - frames[j]["P"])
        if cells[i] == cells[j]:
            within.append((dP, v))
        else:
            cross.append((dP, v))
            if dP <= 1.0:
                cross_closeP.append(v)
    within = np.array(within); cross = np.array(cross)

    # corr vs dP figure (merged): within and cross overlaid
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(within[:, 0], within[:, 1], s=28, alpha=0.7, label="within-cell pair")
    ax.scatter(cross[:, 0], cross[:, 1], s=28, alpha=0.7, marker="^", label="cross-cell pair")
    for arr, c in [(within, "tab:blue"), (cross, "tab:orange")]:
        if len(arr) >= 3:
            sl, b = np.polyfit(arr[:, 0], arr[:, 1], 1)
            xx = np.linspace(0, arr[:, 0].max(), 20); ax.plot(xx, sl * xx + b, color=c, lw=1)
    ax.set_xlabel("|pressure difference| (GPa)"); ax.set_ylabel("peak-position Jaccard")
    ax.set_title("Merged: position similarity vs pressure gap. Physical series -> decays.\n"
                 "cross-cell at small dP overlapping within-cell = same material (merge valid)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/merged_jaccard_vs_dP.png", dpi=150); plt.close(fig)

    # d(P) overlay of BOTH cells (direct merge validation)
    fig, ax = plt.subplots(figsize=(11, 7))
    for f in frames:
        if f["decomp"]:
            continue
        d = G.d_from_2theta(f["pos"], wl)
        col = "tab:blue" if f["cell"] == "Cell_29" else "tab:red"
        ax.scatter([f["P"]] * len(d), d, s=12, color=col, alpha=0.7)
    ax.scatter([], [], color="tab:blue", label="Cell_29"); ax.scatter([], [], color="tab:red", label="Cell_14")
    ax.set_xlabel("pressure (GPa)"); ax.set_ylabel("peak d-spacing (Angstrom)"); ax.set_ylim(1.0, 4.5)
    ax.set_title("Merge validation: all detected peaks, both cells, in d-spacing vs P.\n"
                 "Both cells tracing the same compression curves = same material.")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/merged_d_vs_P_bothcells.png", dpi=150); plt.close(fig)

    with open(f"{OUT}/merged_correlation_summary.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["group", "n_pairs", "mean_jaccard", "slope_vs_dP"])
        for name, arr in [("within_cell", within), ("cross_cell", cross)]:
            sl = np.polyfit(arr[:, 0], arr[:, 1], 1)[0] if len(arr) >= 3 else float("nan")
            w.writerow([name, len(arr), round(float(arr[:, 1].mean()), 3), round(float(sl), 4)])

    print(f"MERGED peak-position Jaccard (17 UOTe frames, tol={TOL_2TH} deg):")
    print(f"  within-cell mean={within[:,1].mean():.3f} (n={len(within)})")
    print(f"  cross-cell  mean={cross[:,1].mean():.3f} (n={len(cross)})")
    print(f"  cross-cell at |dP|<=1 GPa mean={np.mean(cross_closeP):.3f} (n={len(cross_closeP)})  <- merge test")
    print(f"  within-cell slope vs dP = {np.polyfit(within[:,0],within[:,1],1)[0]:+.4f}/GPa")
    print(f"  cross-cell  slope vs dP = {np.polyfit(cross[:,0],cross[:,1],1)[0]:+.4f}/GPa")
    print(f"\nWrote heatmap, jaccard-vs-dP, d(P) overlay, summary to {OUT}/")


if __name__ == "__main__":
    main()
