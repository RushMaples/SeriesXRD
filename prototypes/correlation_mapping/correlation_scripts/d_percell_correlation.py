#!/usr/bin/env python3
"""Phase D / Task 1: per-cell frame correlation + retire the pooled heatmap.

The suite's pooled 17x17 overall-correlation heatmap is non-physical: it mixes
two cells (Cell_14 vs Cell_29 behave like different experiments) AND, even
within one cell, the correlation does NOT decay with pressure gap. This script:
  1. computes whole-pattern Pearson correlation on a common 2theta grid,
     background-subtracted, PER CELL (never pooled),
  2. plots correlation vs |dP| per cell to test whether it tracks pressure,
  3. writes an explicit cross-cell-vs-within-cell contrast so the pooled
     heatmap can be dropped with evidence.
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
from scipy.signal import savgol_filter

import xrd_geometry as G

OUT = "outputs/analysis_v2_20260701/D_percell_correlation"
os.makedirs(OUT, exist_ok=True)
INTEGRATED = {"Cell_29": "Data/Cell_29_integrated", "Cell_14": "Data/Cell_14_integrated"}
GRID = np.arange(4.0, 22.0, 0.02)


def load_cell(cell):
    frames = []
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
        yi = np.interp(GRID, x, y, left=np.nan, right=np.nan)
        # background subtract + normalize
        good = np.isfinite(yi)
        yy = yi.copy()
        yy[~good] = np.nanmedian(yi)
        base = savgol_filter(yy, 151, 2)
        resid = np.clip(yy - base, 0, None)
        resid[~good] = 0.0
        n = np.linalg.norm(resid)
        resid = resid / n if n > 0 else resid
        frames.append(dict(cell=cell, pressure=pressure, decomp=decomp,
                           label=f"{pressure:g}GPa" + ("_dec" if decomp else ""),
                           vec=resid))
    frames.sort(key=lambda f: f["pressure"])
    return frames


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def heatmap(frames, cell):
    n = len(frames)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            M[i, j] = corr(frames[i]["vec"], frames[j]["vec"])
    labels = [f["label"] for f in frames]
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * n, 0.8 + 0.6 * n))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="w" if M[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"{cell}: within-cell whole-pattern correlation (bg-sub, common grid)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{cell}_within_correlation.png", dpi=150)
    plt.close(fig)
    return M, labels


def main():
    all_frames = {c: load_cell(c) for c in ("Cell_29", "Cell_14")}
    summary = []

    # per-cell heatmaps + corr-vs-dP
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, cell in zip(axes, ("Cell_29", "Cell_14")):
        frames = [f for f in all_frames[cell] if not f["decomp"]]
        heatmap(all_frames[cell], cell)
        dps, cs = [], []
        for a, b in itertools.combinations(frames, 2):
            dps.append(abs(a["pressure"] - b["pressure"]))
            cs.append(corr(a["vec"], b["vec"]))
        dps, cs = np.array(dps), np.array(cs)
        ax.scatter(dps, cs, s=25, alpha=0.7)
        # linear trend
        good = np.isfinite(cs)
        if good.sum() >= 3:
            sl, b = np.polyfit(dps[good], cs[good], 1)
            xx = np.linspace(0, dps.max(), 20)
            ax.plot(xx, sl * xx + b, "r-", lw=1)
            r = float(np.corrcoef(dps[good], cs[good])[0, 1])
            ax.set_title(f"{cell}: corr vs |dP|  (slope={sl:+.4f}/GPa, r={r:+.2f})\n"
                         f"physical series -> should DECAY (negative slope)")
            summary.append(dict(cell=cell, n_pairs=int(good.sum()),
                                mean_corr=round(float(np.nanmean(cs)), 3),
                                corr_vs_dP_slope=round(sl, 5),
                                corr_vs_dP_r=round(r, 3)))
        ax.set_xlabel("|pressure difference| (GPa)")
        ax.set_ylabel("whole-pattern correlation")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/corr_vs_pressure_gap.png", dpi=150)
    plt.close(fig)

    # explicit within vs cross contrast (evidence to drop the pooled heatmap)
    pooled = all_frames["Cell_29"] + all_frames["Cell_14"]
    within, cross = [], []
    for a, b in itertools.combinations(pooled, 2):
        c = corr(a["vec"], b["vec"])
        if not np.isfinite(c):
            continue
        (within if a["cell"] == b["cell"] else cross).append(c)
    contrast = dict(within_mean=round(float(np.mean(within)), 3),
                    within_n=len(within),
                    cross_mean=round(float(np.mean(cross)), 3),
                    cross_n=len(cross),
                    within_over_cross=round(float(np.mean(within) / np.mean(cross)), 2))

    with open(f"{OUT}/percell_correlation_summary.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["cell", "n_pairs", "mean_corr",
                                          "corr_vs_dP_slope", "corr_vs_dP_r"])
        w.writeheader()
        for r in summary:
            w.writerow(r)
    with open(f"{OUT}/pooled_vs_percell_contrast.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(contrast.keys()))
        w.writeheader(); w.writerow(contrast)

    print("=== per-cell correlation-vs-pressure-gap ===")
    for r in summary:
        verdict = "DOES NOT track P (near-flat/positive)" if r["corr_vs_dP_slope"] >= -0.005 else "decays with P (ok)"
        print(f"  {r['cell']}: mean_corr={r['mean_corr']}  slope={r['corr_vs_dP_slope']:+}/GPa "
              f"r={r['corr_vs_dP_r']:+}  -> {verdict}")
    print(f"\n=== pooled within-vs-cross (why the 17x17 heatmap is non-physical) ===")
    print(f"  within-cell mean={contrast['within_mean']} (n={contrast['within_n']})  "
          f"cross-cell mean={contrast['cross_mean']} (n={contrast['cross_n']})  "
          f"ratio={contrast['within_over_cross']}x")
    print(f"\nWrote per-cell heatmaps, corr_vs_pressure_gap.png, summaries to {OUT}/")


if __name__ == "__main__":
    main()
