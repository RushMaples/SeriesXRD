#!/usr/bin/env python3
"""Phase B / Task 2: 2D (2theta, chi) static-vs-moving-vs-diamond classifier.

Fixes the under-detection of the 1D trajectory method (which dumps intermittent
single-crystal diamond into 'other' because it never reaches coverage>=0.60).
Works on the consistent STATIC (0deg) exposure series per cell so azimuth (chi)
is comparable across pressure (see xrd-exposure-inventory).

Pipeline per cell:
  1. static series -> per frame: mask, 1D pattern (detect 2theta peaks) + 2D cake.
  2. link peaks across pressure into 2theta trajectories (Hungarian, motion-tol).
  3. motion class from the 2theta trajectory: MOVING vs FLAT vs sparse.
  4. azimuthal character per trajectory from the cake: n chi-spots, ring coverage,
     and chi-STABILITY across pressure (does a spot recur at the same chi?).
  5. 2-axis assignment:
       MOVING 2theta                         -> sample
       FLAT 2theta + spotty + chi-stable      -> diamond / fixed single crystal
                                                 (strengthened if it matches a
                                                  diamond HKL line)
       FLAT 2theta + continuous ring          -> static background ring (gasket/Ne/instr)
       FLAT 2theta + spotty + chi NOT stable  -> textured / ambiguous
  6. targeted chi-stability figures for 18.4 / 11.518(diamond-111) / 18.862(diamond-220).
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, savgol_filter

import xrd_geometry as G

OUT = "outputs/analysis_v2_20260701/B_2d_static_diamond"
os.makedirs(OUT, exist_ok=True)

# detection (kept consistent with peak_trajectory_tracking.py)
PROM = 0.04
MIN_DIST_PTS = 12
SMOOTH_W = 9
BASE_W = 151
TT_LO, TT_HI = 4.0, 22.0
N_RAD_1D = 3000
# cake
N_RAD_CAKE = 1400
N_AZIM = 360
BAND = 0.06          # deg half-width for azimuthal profile
# linking
LINK_TOL = 0.18
MAX_GAP = 2
# motion classification
FLAT_MAX_SPAN = 0.10
FLAT_MAX_SLOPE = 0.010
MOVING_MIN_SPAN = 0.12
MOVING_MIN_SLOPE = 0.015
MOVING_MIN_R2 = 0.60
MIN_FRAMES_MOTION = 3
# azimuthal character
CHI_SPOT_PROM = 0.25
CHI_SPOT_DIST = 5          # bins (~5 deg at 360 bins)
CHI_MATCH_TOL = 10.0       # deg, to call two spots "the same chi" across frames
SPOTTY_MAX_COVER = 0.15
RING_MIN_COVER = 0.40
SPOTTY_MAX_NSPOTS = 3
CHI_STABLE_MIN_PERSIST = 0.50
CHI_STABLE_MAX_MAD = 10.0
REF_MATCH_TOL = 0.08       # deg to attach a reference HKL line

TARGETS = [(18.40, "18.4 (fixed, unidentified)"),
           (11.518, "diamond-111 (theory)"),
           (18.862, "diamond-220 (theory)")]


def detect_1d(x, y):
    y = np.asarray(y, float)
    sm = savgol_filter(y, SMOOTH_W, 2)
    base = savgol_filter(y, BASE_W, 2)
    resid = np.clip(sm - base, 0.0, None)
    hi = np.nanpercentile(resid, 99.5)
    norm = resid / hi if hi > 0 else resid
    pk, _ = find_peaks(norm, prominence=PROM, distance=MIN_DIST_PTS)
    keep = (x[pk] >= TT_LO) & (x[pk] <= TT_HI)
    return x[pk][keep], norm[pk][keep]


def link(frames):
    """Hungarian nearest-neighbour linking across pressure-ordered frames."""
    trajs = []
    for fi, fr in enumerate(frames):
        pos = fr["pos"]
        active = [t for t in trajs if not t["closed"]]
        preds = []
        for t in active:
            pts = t["pts"]
            if len(pts) >= 2:
                (f0, p0, _), (f1, p1, _) = pts[-2], pts[-1]
                preds.append(p1 + (p1 - p0) * ((fi - f1) / max(f1 - f0, 1)))
            else:
                preds.append(pts[-1][1])
        preds = np.array(preds, float)
        matched = set()
        if len(active) and len(pos):
            cost = np.abs(preds[:, None] - pos[None, :])
            cost[cost > LINK_TOL] = 1e6
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] < 1e6:
                    active[r]["pts"].append((fi, float(pos[c]), float(fr["amp"][c])))
                    active[r]["last_fi"] = fi
                    matched.add(c)
        for t in active:
            if fi - t["last_fi"] > MAX_GAP:
                t["closed"] = True
        for c in range(len(pos)):
            if c not in matched:
                trajs.append({"pts": [(fi, float(pos[c]), float(fr["amp"][c]))],
                              "last_fi": fi, "closed": False})
    return trajs


def _erode_gap_edges(prof, n=2):
    """Blank chi bins within n bins of a no-data (nan) gap. pyFAI bleeds signal
    at module-gap boundaries, producing spurious spikes; drop those bins."""
    valid = np.isfinite(prof)
    eroded = valid.copy()
    for s in range(1, n + 1):
        eroded &= np.roll(valid, s) & np.roll(valid, -s)
    out = prof.copy()
    out[~eroded] = np.nan
    return out


def azim_profile(cake_I, rad, target):
    sel = np.abs(rad - target) <= BAND
    if sel.sum() == 0:
        return None
    block = cake_I[:, sel]
    prof = np.where(block > 0, block, np.nan)
    prof = np.nanmean(prof, axis=1)
    return _erode_gap_edges(prof, n=2)  # length N_AZIM, nan where no/edge data


def chi_spots(chi, prof):
    """Return (spot_chi_positions, n_spots, coverage) for one azimuthal profile."""
    valid = np.isfinite(prof)
    if valid.sum() < 10:
        return np.array([]), 0, 0.0
    p = prof[valid]
    base = np.nanpercentile(p, 20)
    b = np.clip(p - base, 0, None)
    mx = b.max()
    if mx <= 0:
        return np.array([]), 0, 1.0
    bn = b / mx
    pk, _ = find_peaks(bn, prominence=CHI_SPOT_PROM, distance=CHI_SPOT_DIST)
    coverage = float(np.mean(bn > 0.30))
    return chi[valid][pk], len(pk), coverage


def chi_stability(spot_lists):
    """Given per-frame spot-chi arrays, find the dominant recurring chi cluster.

    Returns (persist_fraction, chi_mad, dominant_chi). persist = fraction of
    frames (that had >=1 spot) contributing a spot to the dominant cluster.
    """
    frames_with = [s for s in spot_lists if len(s)]
    if not frames_with:
        return 0.0, np.nan, np.nan, 0
    allspots = np.concatenate(frames_with)
    if allspots.size == 0:
        return 0.0, np.nan, np.nan, 0
    # greedy 1D clustering by CHI_MATCH_TOL
    order = np.sort(allspots)
    clusters = [[order[0]]]
    for v in order[1:]:
        if v - clusters[-1][-1] <= CHI_MATCH_TOL:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    # score each cluster by how many distinct frames contribute
    best = None
    for cl in clusters:
        c_center = np.median(cl)
        n_frames_hit = sum(any(abs(s - c_center) <= CHI_MATCH_TOL for s in fr)
                           for fr in frames_with)
        if best is None or n_frames_hit > best[0]:
            members = [s for fr in frames_with for s in fr if abs(s - c_center) <= CHI_MATCH_TOL]
            mad = float(np.median(np.abs(np.array(members) - c_center))) if members else np.nan
            best = (n_frames_hit, c_center, mad)
    persist = best[0] / len(frames_with)
    return persist, best[2], best[1], best[0]  # persist, chi_mad, dom_chi, n_frames_hit


def classify(tr, wl, ref_lines):
    pts = tr["pts"]
    pos = np.array([p[1] for p in pts])
    press = np.array([p[3] for p in pts])  # filled later
    n = len(pts)
    span = float(pos.max() - pos.min())
    slope = r2 = np.nan
    if n >= 2 and np.ptp(press) > 0:
        slope, b = np.polyfit(press, pos, 1)
        pred = slope * press + b
        ss = np.sum((pos - pred) ** 2)
        tot = np.sum((pos - pos.mean()) ** 2)
        r2 = 1 - ss / tot if tot > 0 else np.nan
    motion = "sparse"
    if n >= MIN_FRAMES_MOTION:
        if (span <= FLAT_MAX_SPAN and (not np.isfinite(slope) or abs(slope) <= FLAT_MAX_SLOPE)):
            motion = "FLAT"
        elif (span >= MOVING_MIN_SPAN and np.isfinite(slope) and abs(slope) >= MOVING_MIN_SLOPE
              and np.isfinite(r2) and r2 >= MOVING_MIN_R2):
            motion = "MOVING"
        else:
            motion = "irregular"
    med_2th = float(np.median(pos))
    # nearest reference line
    ref = min(ref_lines, key=lambda r: abs(r["two_theta"] - med_2th))
    ref_dist = abs(ref["two_theta"] - med_2th)
    return dict(motion=motion, n=n, span=round(span, 4), median_2theta=round(med_2th, 4),
                slope=round(float(slope), 5) if np.isfinite(slope) else "",
                r2=round(float(r2), 3) if np.isfinite(r2) else "",
                ref_phase=ref["phase"] if ref_dist <= 0.20 else "",
                ref_hkl=str(ref["hkl"]) if ref_dist <= 0.20 else "",
                ref_dist=round(ref_dist, 4) if ref_dist <= 0.20 else "")


CHI_STABLE_MIN_FRAMES = 3   # a fixed-crystal call needs a chi-spot in >=3 static frames


def assign_class(motion, n_spots_med, coverage_med, persist, chi_mad, n_chi_hit,
                 ref_phase, ref_fixed):
    if motion == "sparse":
        return "isolated/sparse"
    if motion == "MOVING":
        return "sample (moving)"
    if motion == "irregular":
        return "sample? (irregular motion)"
    # FLAT branch
    spotty = (n_spots_med <= SPOTTY_MAX_NSPOTS and coverage_med < SPOTTY_MAX_COVER)
    ring = coverage_med >= RING_MIN_COVER
    chi_stable = (persist >= CHI_STABLE_MIN_PERSIST and np.isfinite(chi_mad)
                  and chi_mad <= CHI_STABLE_MAX_MAD)
    enough = n_chi_hit >= CHI_STABLE_MIN_FRAMES
    if ring:
        return "static background ring (gasket/Ne/instrument)"
    if spotty and chi_stable and enough:
        if ref_phase == "diamond":
            return "DIAMOND (fixed single-crystal, matches HKL)"
        return "fixed single-crystal (diamond/instrument-like)"
    if spotty and chi_stable and not enough:
        return "weak fixed-spot (chi-stable but <3 static frames)"
    if spotty and not chi_stable:
        return "textured/ambiguous (flat-2th, chi not stable)"
    return "textured static arc"


def process_cell(cell, ai, wl, ref_lines):
    inv = G.inventory()
    series = G.select_series(inv, cell)
    frames = []
    cakes = []
    for fr in series:
        img = G.read_image(fr.path)
        mask = G.build_mask(img, ai)
        x, y = G.integrate1d(img, ai, mask, n_rad=N_RAD_1D)
        pos, amp = detect_1d(x, y)
        I, rad, chi = G.cake(img, ai, mask, n_rad=N_RAD_CAKE, n_azim=N_AZIM)
        frames.append(dict(pressure=fr.pressure, label=fr.label, static=fr.static,
                           pos=pos, amp=amp))
        cakes.append(dict(I=I, rad=rad, chi=chi, label=fr.label, pressure=fr.pressure,
                          static=fr.static))
    n_static = sum(f["static"] for f in frames)
    trajs = link(frames)
    rows = []
    for ti, tr in enumerate(trajs):
        # attach pressure to each point
        tr["pts"] = [(fi, pos, amp, frames[fi]["pressure"]) for (fi, pos, amp) in tr["pts"]]
        meta = classify(tr, wl, ref_lines)
        # azimuthal character across frames the trajectory was seen in
        spot_lists = []
        n_spots_list = []
        cover_list = []
        for (fi, pos, amp, p) in tr["pts"]:
            if not cakes[fi]["static"]:
                continue  # chi only comparable on static frames
            prof = azim_profile(cakes[fi]["I"], cakes[fi]["rad"], pos)
            if prof is None:
                continue
            sc, ns, cov = chi_spots(cakes[fi]["chi"], prof)
            spot_lists.append(sc)
            n_spots_list.append(ns)
            cover_list.append(cov)
        persist, chi_mad, dom_chi, n_chi_hit = chi_stability(spot_lists)
        n_spots_med = float(np.median(n_spots_list)) if n_spots_list else np.nan
        cover_med = float(np.median(cover_list)) if cover_list else np.nan
        ref_fixed = any(r["phase"] == "diamond" and abs(r["two_theta"] - meta["median_2theta"]) <= REF_MATCH_TOL
                        for r in ref_lines)
        cls = assign_class(meta["motion"], n_spots_med if np.isfinite(n_spots_med) else 99,
                           cover_med if np.isfinite(cover_med) else 1.0,
                           persist, chi_mad, n_chi_hit, meta["ref_phase"], ref_fixed)
        d = float(G.d_from_2theta(meta["median_2theta"], wl))
        rows.append(dict(cell=cell, traj_id=ti, **meta, d_spacing_A=round(d, 4),
                         n_static_frames=len(n_spots_list),
                         n_chi_spots_med=round(n_spots_med, 1) if np.isfinite(n_spots_med) else "",
                         ring_coverage_med=round(cover_med, 3) if np.isfinite(cover_med) else "",
                         chi_persist=round(persist, 3), n_chi_frames_hit=n_chi_hit,
                         chi_mad_deg=round(chi_mad, 2) if np.isfinite(chi_mad) else "",
                         dominant_chi=round(dom_chi, 1) if np.isfinite(dom_chi) else "",
                         assigned_class=cls))
    rows.sort(key=lambda r: r["median_2theta"])
    return rows, cakes, n_static, len(series)


def targeted_figs(cell, cakes, wl):
    static_cakes = [c for c in cakes if c["static"]]
    if not static_cakes:
        return
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(11, 3.0 * len(TARGETS)))
    if len(TARGETS) == 1:
        axes = [axes]
    for ax, (t, name) in zip(axes, TARGETS):
        for c in static_cakes:
            prof = azim_profile(c["I"], c["rad"], t)
            if prof is None:
                continue
            valid = np.isfinite(prof)
            if valid.sum() < 10:
                continue
            p = prof.copy()
            base = np.nanpercentile(p[valid], 20)
            p = np.clip(p - base, 0, None)
            mx = np.nanmax(p[valid])
            if mx and mx > 0:
                p = p / mx
            ax.plot(c["chi"], p, lw=0.9, label=f"{c['pressure']:g}GPa")
        ax.set_title(f"{cell}: azimuthal profile at 2th={t:.3f} deg  [{name}]  "
                     f"(overlaid across all static frames)", fontsize=10)
        ax.set_xlabel("azimuth chi (deg)")
        ax.set_ylabel("norm I")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{cell}: chi-fixedness across pressure — a spot that stays at the "
                 f"same chi at all P = fixed single crystal (diamond)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{cell}_azimuthal_stability_targets.png", dpi=150)
    plt.close(fig)


def main():
    ai = G.load_ai()
    wl = G.wavelength_A(ai)
    ref_lines = G.reference_lines(wl)
    all_rows = []
    for cell in ("Cell_29", "Cell_14"):
        rows, cakes, n_static, n_series = process_cell(cell, ai, wl, ref_lines)
        all_rows.extend(rows)
        targeted_figs(cell, cakes, wl)
        # summary counts
        from collections import Counter
        cnt = Counter(r["assigned_class"] for r in rows)
        print(f"\n=== {cell}: {n_series} series frames ({n_static} static) -> "
              f"{len(rows)} trajectories ===")
        for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"   {v:2d}  {k}")
        # show the flat / diamond candidates
        print("   -- FLAT / fixed candidates --")
        for r in rows:
            if r["motion"] == "FLAT" or "DIAMOND" in r["assigned_class"] or "single-crystal" in r["assigned_class"]:
                print(f"      2th={r['median_2theta']:7.3f} d={r['d_spacing_A']:.4f} "
                      f"span={r['span']:.3f} nspot={r['n_chi_spots_med']} cov={r['ring_coverage_med']} "
                      f"persist={r['chi_persist']} chimad={r['chi_mad_deg']} ref={r['ref_phase']}{r['ref_hkl']} "
                      f"-> {r['assigned_class']}")
    cols = ["cell", "traj_id", "median_2theta", "d_spacing_A", "motion", "span", "slope",
            "r2", "n", "n_static_frames", "n_chi_spots_med", "ring_coverage_med",
            "chi_persist", "n_chi_frames_hit", "chi_mad_deg", "dominant_chi",
            "ref_phase", "ref_hkl", "ref_dist", "assigned_class"]
    with open(f"{OUT}/classification_table.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nWrote classification_table.csv + azimuthal_stability_targets figures to {OUT}/")


if __name__ == "__main__":
    main()
