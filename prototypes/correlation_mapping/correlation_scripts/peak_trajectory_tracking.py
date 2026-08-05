#!/usr/bin/env python3
"""Pressure-trajectory peak tracking for XRD (robust static/moving classifier).

Why this exists
---------------
The correlation suite's per-peak grouping uses a fixed ~0.02 deg tolerance, which
is far smaller than real pressure-driven peak motion (~0.5-0.7 deg over the run).
That fragments every moving peak into a stack of near-stationary group IDs, so the
suite reports many "static" groups and zero "shifting" groups -- both artifacts.
Its 1D detector is also run at high recall (~230 peaks/frame vs ~12 real rings).

This script fixes both:
  1. Detect peaks per frame with a SANE prominence on baseline-subtracted,
     self-normalized patterns (~10-20 peaks/frame, matching the raw 2D rings).
  2. Link peaks across ascending pressure with a motion-tolerant GLOBAL
     assignment (Hungarian) using linear extrapolation of each trajectory, so
     adjacent moving peaks do not cross-link.
  3. Classify each trajectory: STATIC (persists at ~constant 2theta), MOVING
     (smooth monotonic shift), or OTHER (short/transient/ambiguous).

Cells are processed SEPARATELY by default (Cell_14 and Cell_29 behave like two
different datasets in intensity/shape; only peak *positions* are cross-cell
comparable). A combined run is also written for reference.

Outputs (per group, under --out-dir/<group>/):
  peaks_per_frame.csv, trajectories.csv, static_candidates.csv,
  moving_candidates.csv, waterfall_trajectories.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, savgol_filter

PRESSURE_RE = re.compile(r"(\d+p\d+|\d+)\s*GPa", re.I)


def parse_pressure(name: str) -> float:
    m = PRESSURE_RE.search(os.path.basename(name).replace("decomp-", ""))
    return float(m.group(1).replace("p", ".")) if m else float("nan")


def infer_cell(path: str) -> str:
    if "Cell_14" in path:
        return "Cell_14"
    if "Cell_29" in path:
        return "Cell_29"
    return "all"


def load_xy(path: str) -> np.ndarray:
    try:
        return np.loadtxt(path)
    except UnicodeDecodeError:
        return np.loadtxt(path, encoding="latin1")


def detect_peaks(x: np.ndarray, y: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    sm = savgol_filter(y, args.smooth_window, 2)
    base = savgol_filter(y, args.baseline_window, 2)
    resid = np.clip(sm - base, 0.0, None)
    hi = np.nanpercentile(resid, 99.5)
    norm = resid / hi if hi > 0 else resid
    pk, _ = find_peaks(norm, prominence=args.prominence, distance=args.min_distance_pts)
    keep = (x[pk] >= args.two_theta_lo) & (x[pk] <= args.two_theta_hi)
    return x[pk][keep], norm[pk][keep]


def link_trajectories(frames: list[dict], args) -> list[dict]:
    """Global-assignment nearest-neighbour linking across ascending pressure."""
    trajs: list[dict] = []  # {pts:[(fi,pos,amp)], last_fi, closed}
    for fi, fr in enumerate(frames):
        pos = fr["pos"]
        active = [t for t in trajs if not t["closed"]]
        # predicted next position for each active trajectory (linear extrapolation)
        preds = []
        for t in active:
            pts = t["pts"]
            if len(pts) >= 2:
                (f0, p0, _), (f1, p1, _) = pts[-2], pts[-1]
                # extrapolate by frame index step (frames are pressure-ordered)
                preds.append(p1 + (p1 - p0) * ((fi - f1) / max(f1 - f0, 1)))
            else:
                preds.append(pts[-1][1])
        preds = np.array(preds, dtype=float)

        matched_peaks: set[int] = set()
        if len(active) and len(pos):
            cost = np.abs(preds[:, None] - pos[None, :])
            cost[cost > args.link_tolerance] = 1e6
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] < 1e6:
                    active[r]["pts"].append((fi, float(pos[c]), float(fr["amp"][c])))
                    active[r]["last_fi"] = fi
                    matched_peaks.add(c)
        # close trajectories that have gone stale
        for t in active:
            if fi - t["last_fi"] > args.max_gap:
                t["closed"] = True
        # unmatched peaks start new trajectories
        for c in range(len(pos)):
            if c not in matched_peaks:
                trajs.append({"pts": [(fi, float(pos[c]), float(fr["amp"][c]))],
                              "last_fi": fi, "closed": False})
    return trajs


def classify(traj: dict, frames: list[dict], args) -> dict:
    fis = np.array([p[0] for p in traj["pts"]])
    pos = np.array([p[1] for p in traj["pts"]])
    amp = np.array([p[2] for p in traj["pts"]])
    press = np.array([frames[i]["pressure"] for i in fis])
    # use only compression points for the shift fit
    comp = np.array([not frames[i]["decomp"] for i in fis])
    n_frames_total = sum(1 for fr in frames if not fr["decomp"])
    coverage = len(set(fis[comp])) / max(n_frames_total, 1)
    span = float(pos.max() - pos.min())
    slope = r2 = float("nan")
    if comp.sum() >= 3 and np.ptp(press[comp]) > 0:
        slope, intercept = np.polyfit(press[comp], pos[comp], 1)
        pred = slope * press[comp] + intercept
        ss = float(np.sum((pos[comp] - pred) ** 2))
        tot = float(np.sum((pos[comp] - pos[comp].mean()) ** 2))
        r2 = 1.0 - ss / tot if tot > 0 else float("nan")
    kind = "other"
    if (coverage >= args.static_min_coverage and span <= args.static_max_span
            and (not math.isfinite(slope) or abs(slope) <= args.static_max_abs_slope)):
        kind = "STATIC"
    elif (comp.sum() >= 4 and math.isfinite(slope) and abs(slope) >= args.moving_min_abs_slope
          and span >= args.moving_min_span and math.isfinite(r2) and r2 >= args.moving_min_r2):
        kind = "MOVING"
    return {
        "kind": kind,
        "median_2theta": round(float(np.median(pos)), 4),
        "min_2theta": round(float(pos.min()), 4),
        "max_2theta": round(float(pos.max()), 4),
        "span_deg": round(span, 4),
        "n_frames": int(len(set(fis))),
        "coverage": round(coverage, 3),
        "slope_deg_per_GPa": round(float(slope), 5) if math.isfinite(slope) else "",
        "r2": round(float(r2), 3) if math.isfinite(r2) else "",
        "mean_amp": round(float(amp.mean()), 4),
        "pressure_min": round(float(press.min()), 3),
        "pressure_max": round(float(press.max()), 3),
        "frames": ";".join(f"{frames[i]['pressure']:g}" for i in fis),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def waterfall(frames: list[dict], trajs: list[dict], classified: list[dict], out: Path, args) -> None:
    gx = np.arange(args.two_theta_lo, args.two_theta_hi, 0.014)

    def bg(x, y):
        sm = savgol_filter(np.asarray(y, float), args.smooth_window, 2)
        base = savgol_filter(np.asarray(y, float), args.baseline_window, 2)
        r = np.clip(sm - base, 0, None)
        v = np.interp(gx, x, r)
        hi = np.nanpercentile(v, 99.5)
        return v / hi if hi > 0 else v

    comp = [fr for fr in frames if not fr["decomp"]]
    if len(comp) < 2:
        return
    M = np.vstack([bg(fr["x"], fr["y"]) for fr in comp])
    P = np.array([fr["pressure"] for fr in comp])
    pedges = np.concatenate([[P[0] - 0.3], 0.5 * (P[:-1] + P[1:]), [P[-1] + 0.3]])
    xedges = np.concatenate([gx - 0.007, [gx[-1] + 0.007]])
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.pcolormesh(xedges, pedges, np.clip(M, 0, 0.5), cmap="Greys", vmin=0, vmax=0.5, shading="flat")
    ns = nm = 0
    for traj, cls in zip(trajs, classified):
        pts = [(frames[i]["pressure"], pos) for i, pos, _ in traj["pts"] if not frames[i]["decomp"]]
        if len(pts) < 2:
            continue
        pr = [p for p, _ in pts]
        po = [o for _, o in pts]
        if cls["kind"] == "STATIC":
            ax.plot(po, pr, "-o", color="tab:red", ms=3, lw=1.3, zorder=5)
            ns += 1
        elif cls["kind"] == "MOVING":
            ax.plot(po, pr, "-o", color="tab:green", ms=3, lw=1.3, zorder=5)
            nm += 1
    ax.plot([], [], "tab:red", label=f"STATIC ({sum(c['kind']=='STATIC' for c in classified)})")
    ax.plot([], [], "tab:green", label=f"MOVING ({sum(c['kind']=='MOVING' for c in classified)})")
    ax.set_xlabel("2theta (deg)")
    ax.set_ylabel("pressure (GPa)")
    ax.set_title(f"{out.parent.name}: trajectory tracking (red=static, green=moving)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def process_group(name: str, paths: list[str], out_root: Path, args) -> dict:
    frames = []
    for p in paths:
        arr = load_xy(p)
        px, amp = detect_peaks(arr[:, 0], arr[:, 1], args)
        frames.append({"path": p, "pressure": parse_pressure(p), "cell": infer_cell(p),
                       "decomp": "decomp" in p.lower(), "x": arr[:, 0], "y": arr[:, 1],
                       "pos": px, "amp": amp})
    frames.sort(key=lambda f: (f["decomp"], f["pressure"]))
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # peaks per frame
    per_frame = []
    for fr in frames:
        for pos, amp in zip(fr["pos"], fr["amp"]):
            per_frame.append({"frame_pressure": fr["pressure"], "cell": fr["cell"],
                              "decomp": int(fr["decomp"]), "two_theta": round(float(pos), 4),
                              "norm_amplitude": round(float(amp), 4)})
    write_csv(out_dir / "peaks_per_frame.csv", per_frame)

    trajs = link_trajectories(frames, args)
    classified = [classify(t, frames, args) for t in trajs]
    order = np.argsort([c["median_2theta"] for c in classified])
    trajs = [trajs[i] for i in order]
    classified = [classified[i] for i in order]
    for tid, c in enumerate(classified):
        c["traj_id"] = tid
    cols = ["traj_id", "kind", "median_2theta", "min_2theta", "max_2theta", "span_deg",
            "n_frames", "coverage", "slope_deg_per_GPa", "r2", "mean_amp",
            "pressure_min", "pressure_max", "frames"]
    write_csv(out_dir / "trajectories.csv", [{k: c[k] for k in cols} for c in classified])
    write_csv(out_dir / "static_candidates.csv", [{k: c[k] for k in cols} for c in classified if c["kind"] == "STATIC"])
    write_csv(out_dir / "moving_candidates.csv", [{k: c[k] for k in cols} for c in classified if c["kind"] == "MOVING"])
    waterfall(frames, trajs, classified, out_dir / "waterfall_trajectories.png", args)

    summary = {
        "group": name,
        "n_frames": len(frames),
        "mean_peaks_per_frame": round(float(np.mean([len(fr["pos"]) for fr in frames])), 1),
        "n_trajectories": len(classified),
        "n_static": sum(c["kind"] == "STATIC" for c in classified),
        "n_moving": sum(c["kind"] == "MOVING" for c in classified),
        "n_other": sum(c["kind"] == "other" for c in classified),
    }
    return summary


def collect_inputs(inputs: list[str]) -> list[str]:
    paths: list[str] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(glob.glob(str(p / "*.xy"))))
        elif any(ch in item for ch in "*?["):
            paths.extend(sorted(glob.glob(item)))
        elif p.exists():
            paths.append(str(p))
    return sorted(set(paths))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", default=["Data/Cell_14_integrated", "Data/Cell_29_integrated"])
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/peak_trajectories_20260701"))
    ap.add_argument("--combined", action="store_true", help="Also run all cells together.")
    # detection
    ap.add_argument("--prominence", type=float, default=0.04)
    ap.add_argument("--min-distance-pts", type=int, default=12)
    ap.add_argument("--smooth-window", type=int, default=9)
    ap.add_argument("--baseline-window", type=int, default=151)
    ap.add_argument("--two-theta-lo", type=float, default=4.0)
    ap.add_argument("--two-theta-hi", type=float, default=22.0)
    # linking
    ap.add_argument("--link-tolerance", type=float, default=0.18, help="Max 2theta jump between adjacent frames (deg).")
    ap.add_argument("--max-gap", type=int, default=2, help="Frames a trajectory may miss before it is closed.")
    # classification
    ap.add_argument("--static-min-coverage", type=float, default=0.60)
    ap.add_argument("--static-max-span", type=float, default=0.06)
    ap.add_argument("--static-max-abs-slope", type=float, default=0.006)
    ap.add_argument("--moving-min-abs-slope", type=float, default=0.008)
    ap.add_argument("--moving-min-span", type=float, default=0.08)
    ap.add_argument("--moving-min-r2", type=float, default=0.60)
    args = ap.parse_args()

    paths = collect_inputs(args.inputs)
    if not paths:
        raise SystemExit("No .xy inputs found.")
    by_cell: dict[str, list[str]] = {}
    for p in paths:
        by_cell.setdefault(infer_cell(p), []).append(p)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cell, cell_paths in sorted(by_cell.items()):
        if len(cell_paths) >= 3:
            summaries.append(process_group(cell, cell_paths, args.out_dir, args))
    if args.combined or len(by_cell) == 1:
        summaries.append(process_group("combined", paths, args.out_dir, args))

    write_csv(args.out_dir / "run_summary.csv", summaries)
    print("=== peak trajectory tracking summary ===")
    for s in summaries:
        print(f"  {s['group']:10s}: {s['n_frames']} frames, ~{s['mean_peaks_per_frame']} peaks/frame, "
              f"{s['n_trajectories']} trajs -> STATIC={s['n_static']} MOVING={s['n_moving']} other={s['n_other']}")
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
