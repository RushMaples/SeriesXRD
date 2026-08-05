#!/usr/bin/env python3
"""Scientific validation summaries for high-recall XRD peak groups.

This script does not rerun or tune peak detection. It reads an existing
correlation suite and asks which candidate groups deserve scientific review.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SUITE = Path("outputs/correlation_suite_20260621_high_recall_scored_v2")
PRESSURE_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)GPa")


@dataclass(frozen=True)
class FrameInfo:
    label: str
    pressure: float
    branch: str
    order: int


def parse_pressure(label: str) -> float:
    match = PRESSURE_RE.search(str(label))
    if not match:
        return float("nan")
    return float(match.group("value"))


def parse_frame_info(labels: Iterable[str]) -> list[FrameInfo]:
    infos: list[FrameInfo] = []
    compression = []
    decompression = []
    for label in sorted(set(str(label) for label in labels), key=lambda item: (math.isnan(parse_pressure(item)), parse_pressure(item), item)):
        branch = "decompression" if "decomp" in label.lower() else "compression"
        info = FrameInfo(label=label, pressure=parse_pressure(label), branch=branch, order=-1)
        if branch == "decompression":
            decompression.append(info)
        else:
            compression.append(info)
    compression.sort(key=lambda item: (item.pressure, item.label))
    decompression.sort(key=lambda item: (item.pressure, item.label))
    ordered: list[FrameInfo] = []
    for idx, info in enumerate(compression + decompression):
        ordered.append(FrameInfo(info.label, info.pressure, info.branch, idx))
    return ordered


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or float(np.ptp(x)) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = math.sqrt(ss_res / len(x))
    return (float(slope), float(intercept), float(r2), float(rmse))


def longest_run(labels: Iterable[str], frame_order: dict[str, FrameInfo]) -> int:
    by_branch: dict[str, list[int]] = {"compression": [], "decompression": []}
    for label in labels:
        info = frame_order.get(str(label))
        if info:
            by_branch[info.branch].append(info.order)
    longest = 0
    for values in by_branch.values():
        values = sorted(set(values))
        if not values:
            continue
        current = 1
        best = 1
        for left, right in zip(values, values[1:]):
            if right == left + 1:
                current += 1
            else:
                best = max(best, current)
                current = 1
        longest = max(longest, best, current)
    return longest


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def source_methods(series: pd.Series) -> str:
    sources: set[str] = set()
    for value in series.dropna().astype(str):
        sources.update(part for part in value.split(";") if part and part != "nan")
    return ";".join(sorted(sources))


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        arr = np.loadtxt(path)
    except UnicodeDecodeError:
        arr = np.loadtxt(path, encoding="latin1")
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Cannot read two-column xy data from {path}")
    return arr[:, 0], arr[:, 1]


def normalize_intensity(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    lo, hi = np.nanpercentile(y, [1, 99.5])
    span = hi - lo
    if not math.isfinite(span) or span <= 0:
        return y
    return (y - lo) / span


def discover_raw_tiffs(data_root: Path, label: str) -> list[str]:
    pressure = parse_pressure(label)
    if not math.isfinite(pressure) or not data_root.exists():
        return []
    token = f"{pressure:g}".replace(".", "p")
    patterns = [f"*{token}*GPa*.tif", f"*{token} GPa*/*.tif"]
    if "decomp" in label.lower():
        patterns.insert(0, "*decomp*/*.tif")
        patterns.insert(1, "*noNe*/*.tif")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(data_root.glob(f"**/{pattern}"))
    found = sorted(set(found), key=lambda path: ("10deg" not in path.name, path.name))
    return [str(path) for path in found[:6]]


def scientific_category_notes(row: pd.Series) -> tuple[str, str, int]:
    categories: list[str] = []
    notes: list[str] = []
    priority = 3
    frame_count = int(row.get("frame_count", 0))
    coverage = safe_float(row.get("coverage_fraction"))
    longest = int(row.get("longest_consecutive_run", 0))
    max_area = safe_float(row.get("max_roi_area"))
    slope = safe_float(row.get("pressure_position_slope_deg_per_gpa"), float("nan"))
    r2 = safe_float(row.get("position_fit_r_squared"), float("nan"))
    shift = safe_float(row.get("total_shift_deg"))
    mad = safe_float(row.get("position_mad_deg"))
    static = bool(row.get("suspected_static", False))
    tier_b = int(row.get("tier_b_count", 0))
    tier_a = int(row.get("tier_a_count", 0))
    comp_count = int(row.get("compression_frame_count", 0))
    decomp_count = int(row.get("decompression_frame_count", 0))
    pressure_min = safe_float(row.get("pressure_min"), float("nan"))
    pressure_max = safe_float(row.get("pressure_max"), float("nan"))

    if coverage >= 0.45 or frame_count >= 8:
        categories.append("recurring")
        notes.append("appears in many scans; recurrence alone does not identify phase")
        priority = min(priority, 2)
    if static or (frame_count >= 8 and mad <= 0.015 and abs(slope) <= 0.0015):
        categories.append("likely_static_background")
        notes.append("nearly fixed 2theta across pressure; possible static/background/diamond-like candidate")
        priority = min(priority, 2)
    if frame_count >= 4 and math.isfinite(r2) and abs(slope) >= 0.003 and shift >= 0.04 and r2 >= 0.45 and longest >= 3:
        categories.append("systematic_shift")
        notes.append("position changes smoothly enough to inspect as a possible pressure-dependent feature")
        priority = 1
    if comp_count > 0 and decomp_count > 0 and frame_count <= 4:
        categories.append("possible_hysteresis_or_decomp_link")
        notes.append("limited detections across compression/decompression; inspect before connecting trajectories")
        priority = min(priority, 2)
    if comp_count == 0 and decomp_count > 0:
        categories.append("decompression_only")
        notes.append("appears only in decompression scan")
        priority = min(priority, 2)
    if comp_count > 0 and pressure_max <= 3.0 and frame_count >= 2:
        categories.append("low_pressure_only")
        notes.append("limited to low pressure range")
        priority = min(priority, 2)
    if comp_count > 0 and pressure_min >= 8.0 and frame_count >= 2:
        categories.append("high_pressure_only")
        notes.append("limited to high pressure range")
        priority = min(priority, 2)
    if frame_count <= 2 and (tier_a > 0 or max_area >= 0.02):
        categories.append("isolated_strong")
        notes.append("few-scan candidate with strong Tier/ROI evidence; may be transient or grouping failure")
        priority = min(priority, 2)
    if tier_b > tier_a and longest >= 3:
        categories.append("tier_b_dependent")
        notes.append("trajectory depends strongly on weak/shoulder Tier B detections")
        priority = min(priority, 2)
    if not categories:
        categories.append("uncertain")
        notes.append("not enough recurrence, static behavior, or smooth shift for automatic prioritization")
    return ";".join(categories), " | ".join(notes), priority


def build_metrics(
    peak_table: pd.DataFrame,
    group_summary: pd.DataFrame,
    map_index: pd.DataFrame,
    static_scores: pd.DataFrame,
    frame_infos: list[FrameInfo],
) -> pd.DataFrame:
    frame_order = {info.label: info for info in frame_infos}
    total_frames = len(frame_infos)
    static_cols = static_scores.copy()
    if not static_cols.empty:
        static_cols = static_cols.add_prefix("static_")
        static_cols = static_cols.rename(columns={"static_peak_group": "peak_group"})

    rows: list[dict[str, object]] = []
    for group_id, sub in peak_table.groupby("peak_group"):
        group_id = int(group_id)
        labels = [str(value) for value in sub["frame"].dropna().unique()]
        infos = [frame_order[label] for label in labels if label in frame_order]
        comp = [info for info in infos if info.branch == "compression"]
        decomp = [info for info in infos if info.branch == "decompression"]
        positions = pd.to_numeric(sub["two_theta"], errors="coerce").dropna().to_numpy(dtype=float)
        pressures = np.array([frame_order[str(row.frame)].pressure for row in sub.itertuples() if str(row.frame) in frame_order], dtype=float)
        pos_for_fit = np.array([float(row.two_theta) for row in sub.itertuples() if str(row.frame) in frame_order], dtype=float)
        branches_for_fit = [frame_order[str(row.frame)].branch for row in sub.itertuples() if str(row.frame) in frame_order]
        comp_mask = np.array([branch == "compression" for branch in branches_for_fit], dtype=bool)
        slope, intercept, r2, rmse = linear_fit(pressures[comp_mask], pos_for_fit[comp_mask])
        sp = spearman(pressures[comp_mask], pos_for_fit[comp_mask])
        row = {
            "peak_group": group_id,
            "median_2theta": float(np.nanmedian(positions)) if len(positions) else float("nan"),
            "min_2theta": float(np.nanmin(positions)) if len(positions) else float("nan"),
            "max_2theta": float(np.nanmax(positions)) if len(positions) else float("nan"),
            "total_shift_deg": float(np.nanmax(positions) - np.nanmin(positions)) if len(positions) else float("nan"),
            "position_mad_deg": float(np.nanmedian(np.abs(positions - np.nanmedian(positions)))) if len(positions) else float("nan"),
            "frame_count": len(labels),
            "coverage_fraction": len(labels) / total_frames if total_frames else float("nan"),
            "longest_consecutive_run": longest_run(labels, frame_order),
            "pressure_min": min((info.pressure for info in infos), default=float("nan")),
            "pressure_max": max((info.pressure for info in infos), default=float("nan")),
            "compression_frame_count": len(comp),
            "decompression_frame_count": len(decomp),
            "tier_a_count": int((sub["confidence_tier"] == "A").sum()),
            "tier_b_count": int((sub["confidence_tier"] == "B").sum()),
            "tier_c_count": int((sub["confidence_tier"] == "C").sum()),
            "max_roi_area": float("nan"),
            "median_roi_area": float("nan"),
            "max_prominence": float(sub["prominence"].max()) if "prominence" in sub else float("nan"),
            "median_prominence": float(sub["prominence"].median()) if "prominence" in sub else float("nan"),
            "pressure_position_slope_deg_per_gpa": slope,
            "position_fit_intercept": intercept,
            "position_fit_r_squared": r2,
            "spearman_pressure_position": sp,
            "position_fit_rmse_deg": rmse,
            "source_methods": source_methods(sub["source_methods"]) if "source_methods" in sub else "",
            "frames_present": ";".join(sorted(labels, key=lambda label: frame_order.get(label, FrameInfo(label, 999, "unknown", 999)).order)),
            "xy_paths": ";".join(sorted(set(str(value) for value in sub["path"].dropna()))),
            "top_peaks_truncated": int(pd.to_numeric(sub.get("top_peaks_truncated", 0), errors="coerce").fillna(0).sum()),
        }
        rows.append(row)
    metrics = pd.DataFrame(rows)

    for extra in [group_summary, map_index]:
        if not extra.empty:
            cols = [col for col in extra.columns if col not in metrics.columns or col == "peak_group"]
            metrics = metrics.merge(extra[cols], on="peak_group", how="left", suffixes=("", "_source"))
    if not static_scores.empty:
        keep = [
            "peak_group",
            "suspected_static",
            "coverage_fraction",
            "detected_frame_count",
            "roi_presence_count",
            "roi_fraction",
            "median_detected_two_theta",
            "position_range_deg",
            "slope_deg_per_gpa",
        ]
        keep = [col for col in keep if col in static_scores.columns]
        renamed = static_scores[keep].rename(
            columns={
                "suspected_static": "suspected_static",
                "coverage_fraction": "static_score_coverage_fraction",
                "detected_frame_count": "static_detected_frame_count",
                "roi_presence_count": "static_roi_presence_count",
                "roi_fraction": "static_roi_fraction",
                "median_detected_two_theta": "static_median_detected_two_theta",
                "position_range_deg": "static_position_range_deg",
                "slope_deg_per_gpa": "static_slope_deg_per_gpa",
            }
        )
        metrics = metrics.merge(renamed, on="peak_group", how="left")
    if "suspected_static" not in metrics.columns:
        metrics["suspected_static"] = False
    metrics["suspected_static"] = metrics["suspected_static"].fillna(0).astype(bool)

    return assign_scientific_categories(metrics)


def assign_scientific_categories(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    categories = metrics.apply(scientific_category_notes, axis=1)
    metrics["scientific_category_labels"] = [item[0] for item in categories]
    metrics["notes"] = [item[1] for item in categories]
    metrics["manual_review_priority"] = [item[2] for item in categories]
    return metrics.sort_values(["manual_review_priority", "median_2theta"]).reset_index(drop=True)


def attach_roi_feature_stats(metrics: pd.DataFrame, map_index: pd.DataFrame, roi_features: pd.DataFrame) -> pd.DataFrame:
    """Populate ROI-area stats from the feature matrix when available."""
    if roi_features.empty or map_index.empty or "frame" not in roi_features.columns:
        return metrics
    roi = roi_features.set_index("frame")
    theta_col_by_group: dict[int, str] = {}
    for row in map_index.itertuples():
        group = int(row.peak_group)
        theta = safe_float(row.group_two_theta, float("nan"))
        if not math.isfinite(theta):
            continue
        target = f"{theta:.4f}"
        if target in roi.columns:
            theta_col_by_group[group] = target
            continue
        try:
            closest = min(roi.columns, key=lambda col: abs(float(col) - theta))
        except ValueError:
            continue
        if abs(float(closest) - theta) <= 0.0002:
            theta_col_by_group[group] = closest

    max_values = []
    med_values = []
    roi_presence_counts = []
    for group in metrics["peak_group"].astype(int):
        col = theta_col_by_group.get(group)
        if not col:
            max_values.append(float("nan"))
            med_values.append(float("nan"))
            roi_presence_counts.append(0)
            continue
        values = pd.to_numeric(roi[col], errors="coerce").fillna(0.0)
        nonzero = values[values > 0]
        max_values.append(float(values.max()))
        med_values.append(float(nonzero.median()) if not nonzero.empty else 0.0)
        roi_presence_counts.append(int((values > 0).sum()))
    metrics = metrics.copy()
    metrics["max_roi_area"] = max_values
    metrics["median_roi_area"] = med_values
    metrics["roi_presence_count"] = roi_presence_counts
    return metrics


def tier_a_only_comparison(metrics: pd.DataFrame, peak_table: pd.DataFrame, frame_infos: list[FrameInfo]) -> pd.DataFrame:
    frame_order = {info.label: info for info in frame_infos}
    rows = []
    for group_id, sub in peak_table.groupby("peak_group"):
        group_id = int(group_id)
        tier_a = sub[sub["confidence_tier"] == "A"]
        ab = metrics[metrics["peak_group"] == group_id].iloc[0]
        a_labels = [str(value) for value in tier_a["frame"].dropna().unique()]
        a_positions = pd.to_numeric(tier_a["two_theta"], errors="coerce").dropna().to_numpy(dtype=float)
        a_pressures = np.array([frame_order[label].pressure for label in a_labels if label in frame_order], dtype=float)
        if len(a_positions) == len(a_pressures):
            a_slope, _, a_r2, _ = linear_fit(a_pressures, a_positions)
        else:
            a_slope, a_r2 = float("nan"), float("nan")
        b_count = int((sub["confidence_tier"] == "B").sum())
        interpretation = "robust in Tier A-only"
        priority = "medium"
        if int(ab["longest_consecutive_run"]) >= 3 and len(a_labels) < int(ab["frame_count"]) / 2 and b_count > 0:
            interpretation = "Tier-B-dependent weak trajectory"
            priority = "high"
        elif len(a_labels) >= 5 and abs(safe_float(a_slope, float("nan"))) >= 0.003:
            interpretation = "convincing Tier A-only shifting candidate"
            priority = "high"
        elif len(a_labels) == 0 and b_count >= 3:
            interpretation = "visible only through weak/shoulder Tier B detections"
            priority = "medium"
        rows.append(
            {
                "peak_group": group_id,
                "median_2theta": ab["median_2theta"],
                "tier_ab_frame_count": int(ab["frame_count"]),
                "tier_a_only_frame_count": len(set(a_labels)),
                "tier_b_contribution": b_count,
                "tier_ab_longest_run": int(ab["longest_consecutive_run"]),
                "tier_a_only_longest_run": longest_run(a_labels, frame_order),
                "tier_ab_position_r_squared": ab["position_fit_r_squared"],
                "tier_a_position_r_squared": a_r2,
                "tier_ab_slope_deg_per_gpa": ab["pressure_position_slope_deg_per_gpa"],
                "tier_a_slope_deg_per_gpa": a_slope,
                "position_trend_quality": "smooth" if safe_float(ab["position_fit_r_squared"], 0) >= 0.45 else "weak/uncertain",
                "interpretation": interpretation,
                "manual_review_priority": priority,
            }
        )
    return pd.DataFrame(rows).sort_values(["manual_review_priority", "tier_ab_frame_count"], ascending=[True, False])


def classify_appearing_disappearing(metrics: pd.DataFrame, frame_infos: list[FrameInfo]) -> pd.DataFrame:
    rows = []
    columns = [
        "peak_group",
        "median_2theta",
        "transition_category",
        "pressure_interval",
        "frame_count",
        "frames_present",
        "max_roi_area",
        "tier_a_count",
        "tier_b_count",
        "evidence",
        "caution",
    ]
    for row in metrics.itertuples():
        labels = str(row.frames_present).split(";") if pd.notna(row.frames_present) else []
        pressures = [parse_pressure(label) for label in labels if math.isfinite(parse_pressure(label))]
        if not pressures:
            continue
        category = ""
        evidence = ""
        if row.decompression_frame_count > 0 and row.compression_frame_count == 0:
            category = "decompression-only appearance"
            evidence = "detected only in decompression scan(s)"
        elif row.compression_frame_count > 0 and row.decompression_frame_count == 0 and row.pressure_max <= 3.0 and row.frame_count >= 2:
            category = "low-pressure-only"
            evidence = f"present from {row.pressure_min:g} to {row.pressure_max:g} GPa only"
        elif row.compression_frame_count > 0 and row.pressure_min >= 8.0 and row.frame_count >= 2:
            category = "high-pressure-only / onset candidate"
            evidence = f"first detected near {row.pressure_min:g} GPa"
        elif row.frame_count <= 3 and row.max_roi_area >= metrics["max_roi_area"].quantile(0.90):
            category = "isolated strong / possible transient"
            evidence = "few detections but high ROI/prominence"
        if category:
            rows.append(
                {
                    "peak_group": row.peak_group,
                    "median_2theta": row.median_2theta,
                    "transition_category": category,
                    "pressure_interval": f"{row.pressure_min:g}-{row.pressure_max:g} GPa",
                    "frame_count": row.frame_count,
                    "frames_present": row.frames_present,
                    "max_roi_area": row.max_roi_area,
                    "tier_a_count": row.tier_a_count,
                    "tier_b_count": row.tier_b_count,
                    "evidence": evidence,
                    "caution": "Could reflect detection threshold, grouping failure, splitting, or changing intensity.",
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["transition_category", "median_2theta"])


def possible_linkages(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [
        "group_id_1",
        "group_id_2",
        "reason",
        "end_pressure_group_1",
        "start_pressure_group_2",
        "end_position",
        "start_position",
        "implied_shift",
        "confidence",
        "possible_event_type",
    ]
    candidates = metrics[metrics["frame_count"] >= 2].sort_values("median_2theta").reset_index(drop=True)
    for i, left in candidates.iterrows():
        for _, right in candidates.iloc[i + 1 : i + 12].iterrows():
            theta_gap = float(right["median_2theta"] - left["median_2theta"])
            if theta_gap > 0.12:
                break
            left_end = safe_float(left["pressure_max"], float("nan"))
            right_start = safe_float(right["pressure_min"], float("nan"))
            overlap = min(safe_float(left["pressure_max"]), safe_float(right["pressure_max"])) - max(safe_float(left["pressure_min"]), safe_float(right["pressure_min"]))
            adjacent = abs(right_start - left_end) <= 2.5 or overlap >= 0
            if not adjacent:
                continue
            reason = []
            if theta_gap <= 0.04:
                reason.append("nearby 2theta groups")
            if overlap < 0:
                reason.append("adjacent pressure ranges")
            else:
                reason.append("overlapping pressure ranges")
            if abs(safe_float(left["pressure_position_slope_deg_per_gpa"])) > 0.003 or abs(safe_float(right["pressure_position_slope_deg_per_gpa"])) > 0.003:
                reason.append("one or both groups show shift")
            event = "possible duplicate/over-merge" if overlap >= 0 and theta_gap <= 0.04 else "possible fragmented moving trajectory"
            confidence = "medium" if theta_gap <= 0.06 else "low"
            rows.append(
                {
                    "group_id_1": int(left["peak_group"]),
                    "group_id_2": int(right["peak_group"]),
                    "reason": "; ".join(reason),
                    "end_pressure_group_1": left_end,
                    "start_pressure_group_2": right_start,
                    "end_position": left["median_2theta"],
                    "start_position": right["median_2theta"],
                    "implied_shift": theta_gap,
                    "confidence": confidence,
                    "possible_event_type": event,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["confidence", "implied_shift"]).head(200)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def plot_persistence(metrics: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    df = metrics.sort_values("median_2theta").reset_index(drop=True)
    colors = np.where(df["suspected_static"], "tab:red", "tab:blue")
    sizes = 20 + 180 * df["coverage_fraction"].fillna(0)
    ax.scatter(df.index, df["median_2theta"], s=sizes, c=colors, alpha=0.7, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("Peak group rank by 2theta")
    ax.set_ylabel("Median 2theta (deg)")
    ax.set_title("Peak-group persistence; marker size = frame coverage, red = likely static/background")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_presence_map(metrics: pd.DataFrame, peak_table: pd.DataFrame, frame_infos: list[FrameInfo], out: Path, tier_a_only: bool = False) -> None:
    selected = metrics.sort_values(["coverage_fraction", "max_roi_area"], ascending=False).head(80)
    frames = [info.label for info in frame_infos]
    matrix = np.zeros((len(selected), len(frames)), dtype=float)
    for i, group_id in enumerate(selected["peak_group"].astype(int)):
        sub = peak_table[peak_table["peak_group"].astype(int) == group_id]
        if tier_a_only:
            sub = sub[sub["confidence_tier"] == "A"]
        for _, row in sub.iterrows():
            if row["frame"] in frames:
                value = 1.0 if row["confidence_tier"] == "A" else 0.55
                matrix[i, frames.index(row["frame"])] = max(matrix[i, frames.index(row["frame"])], value)
    fig, ax = plt.subplots(figsize=(12, max(5, len(selected) * 0.12)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels(frames, rotation=45, ha="right")
    ylabels = [f"{int(row.peak_group)} @ {row.median_2theta:.3f}" for row in selected.itertuples()]
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_title("Peak presence map (" + ("Tier A only" if tier_a_only else "Tier A+B") + ")")
    fig.colorbar(im, ax=ax, label="presence / confidence tier")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_category_summary(metrics: pd.DataFrame, out: Path) -> None:
    labels = ["recurring", "appearing/disappearing", "systematic_shift", "isolated_strong", "likely_static_background", "uncertain"]
    counts = []
    for label in labels:
        if label == "appearing/disappearing":
            mask = metrics["scientific_category_labels"].str.contains("low_pressure_only|high_pressure_only|decompression_only|hysteresis", regex=True)
        else:
            mask = metrics["scientific_category_labels"].str.contains(label, regex=False)
        counts.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, counts, color=["#4477aa", "#66c2a5", "#cc6677", "#ddcc77", "#aa3377", "#999999"])
    ax.set_ylabel("Number of groups")
    ax.set_title("Scientific category counts")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_position_trends(metrics: pd.DataFrame, peak_table: pd.DataFrame, frame_infos: list[FrameInfo], out_dir: Path, max_groups: int = 12) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shifting = metrics[metrics["scientific_category_labels"].str.contains("systematic_shift", regex=False)]
    shifting = shifting.sort_values(["position_fit_r_squared", "total_shift_deg"], ascending=False).head(max_groups)
    outputs: list[Path] = []
    frame_order = {info.label: info for info in frame_infos}
    for row in shifting.itertuples():
        sub = peak_table[peak_table["peak_group"].astype(int) == int(row.peak_group)].copy()
        sub["pressure"] = sub["frame"].map(lambda label: frame_order[str(label)].pressure if str(label) in frame_order else np.nan)
        sub["branch"] = sub["frame"].map(lambda label: frame_order[str(label)].branch if str(label) in frame_order else "unknown")
        fig, ax = plt.subplots(figsize=(6, 4))
        for branch, marker in [("compression", "o"), ("decompression", "s")]:
            part = sub[sub["branch"] == branch]
            if not part.empty:
                ax.scatter(part["pressure"], part["two_theta"], marker=marker, label=branch)
        comp = sub[sub["branch"] == "compression"]
        if len(comp) >= 3 and math.isfinite(row.pressure_position_slope_deg_per_gpa):
            xs = np.linspace(float(comp["pressure"].min()), float(comp["pressure"].max()), 80)
            ys = row.pressure_position_slope_deg_per_gpa * xs + row.position_fit_intercept
            ax.plot(xs, ys, color="black", lw=1, alpha=0.7, label=f"fit R2={row.position_fit_r_squared:.2f}")
        ax.set_xlabel("Pressure (GPa)")
        ax.set_ylabel("2theta (deg)")
        ax.set_title(f"Group {int(row.peak_group)} @ {row.median_2theta:.3f} deg")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = out_dir / f"group_{int(row.peak_group):03d}_position_vs_pressure.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def plot_roi_trends(shortlist: pd.DataFrame, peak_table: pd.DataFrame, frame_infos: list[FrameInfo], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_order = {info.label: info for info in frame_infos}
    outputs = []
    for row in shortlist.itertuples():
        sub = peak_table[peak_table["peak_group"].astype(int) == int(row.peak_group)].copy()
        sub["pressure"] = sub["frame"].map(lambda label: frame_order[str(label)].pressure if str(label) in frame_order else np.nan)
        sub["branch"] = sub["frame"].map(lambda label: frame_order[str(label)].branch if str(label) in frame_order else "unknown")
        fig, ax = plt.subplots(figsize=(6, 4))
        for branch, marker in [("compression", "o"), ("decompression", "s")]:
            part = sub[sub["branch"] == branch]
            if not part.empty:
                ax.scatter(part["pressure"], part["prominence"], marker=marker, label=branch)
                ax.plot(part["pressure"], part["prominence"], alpha=0.4)
        ax.set_xlabel("Pressure (GPa)")
        ax.set_ylabel("Peak prominence / ROI proxy")
        ax.set_title(f"Group {int(row.peak_group)} ROI proxy")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = out_dir / f"group_{int(row.peak_group):03d}_roi_vs_pressure.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def make_contact_sheet(row: pd.Series, peak_table: pd.DataFrame, frame_infos: list[FrameInfo], out_dir: Path, half_width: float = 0.28) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    group_id = int(row["peak_group"])
    sub = peak_table[peak_table["peak_group"].astype(int) == group_id].copy()
    frame_order = {info.label: info for info in frame_infos}
    sub["order"] = sub["frame"].map(lambda label: frame_order[str(label)].order if str(label) in frame_order else 999)
    sub = sub.sort_values("order")
    if sub.empty:
        return None
    center = float(row["median_2theta"])
    n = len(sub)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, peak in zip(axes.flat, sub.itertuples()):
        path = Path(str(peak.path))
        if not path.exists():
            path = Path.cwd() / str(peak.path)
        ax.axis("on")
        try:
            x, y = load_xy(path)
            yn = normalize_intensity(y)
            mask = (x >= center - half_width) & (x <= center + half_width)
            ax.plot(x[mask], yn[mask], color="black", lw=1)
            ax.axvline(float(peak.two_theta), color="tab:orange" if peak.confidence_tier == "A" else "tab:cyan", lw=1.4)
            ax.scatter([float(peak.two_theta)], [float(peak.normalized_intensity)], s=30, marker="o" if peak.confidence_tier == "A" else "x")
        except Exception as exc:  # pragma: no cover - diagnostic plot should continue.
            ax.text(0.05, 0.5, f"Could not read\\n{path}\\n{exc}", transform=ax.transAxes, fontsize=8)
        ax.set_title(f"{peak.frame} | Tier {peak.confidence_tier}", fontsize=9)
        ax.set_xlabel("2theta")
        ax.set_ylabel("norm. I")
    fig.suptitle(f"Manual review group {group_id} @ {center:.3f} deg", y=1.0)
    fig.tight_layout()
    out = out_dir / f"group_{group_id:03d}_contact_sheet.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def make_manual_shortlist(metrics: pd.DataFrame, peak_table: pd.DataFrame, map_index: pd.DataFrame, data_root: Path, out_dir: Path, frame_infos: list[FrameInfo]) -> pd.DataFrame:
    selections: list[pd.Series] = []

    def add_rows(df: pd.DataFrame, category: str, n: int) -> None:
        for _, row in df.head(n).iterrows():
            if int(row["peak_group"]) in {int(existing["peak_group"]) for existing in selections}:
                continue
            row = row.copy()
            row["selection_category"] = category
            selections.append(row)

    add_rows(metrics[metrics["scientific_category_labels"].str.contains("recurring")].sort_values(["coverage_fraction", "max_roi_area"], ascending=False), "recurring", 2)
    add_rows(metrics[metrics["scientific_category_labels"].str.contains("low_pressure_only")].sort_values("max_roi_area", ascending=False), "low-pressure-only", 2)
    add_rows(metrics[metrics["scientific_category_labels"].str.contains("high_pressure_only")].sort_values("max_roi_area", ascending=False), "high-pressure-only", 2)
    add_rows(metrics[metrics["scientific_category_labels"].str.contains("systematic_shift")].sort_values(["position_fit_r_squared", "total_shift_deg"], ascending=False), "systematic-shift", 2)
    add_rows(metrics[metrics["scientific_category_labels"].str.contains("isolated_strong")].sort_values("max_roi_area", ascending=False), "isolated-strong", 1)
    add_rows(metrics[metrics["scientific_category_labels"].str.contains("likely_static_background")].sort_values(["coverage_fraction", "position_mad_deg"], ascending=[False, True]), "likely-static-background", 1)
    add_rows(metrics[metrics["scientific_category_labels"].str.contains("tier_b_dependent")].sort_values(["longest_consecutive_run", "tier_b_count"], ascending=False), "tier-B-dependent", 1)

    shortlist = pd.DataFrame(selections).head(12)
    if shortlist.empty:
        return shortlist
    paths = []
    heatmaps = []
    review_questions = []
    raw_tiffs = []
    for _, row in shortlist.iterrows():
        sheet = make_contact_sheet(row, peak_table, frame_infos, out_dir / "manual_review_contact_sheets")
        paths.append(str(sheet) if sheet else "")
        group = int(row["peak_group"])
        heatmap = ""
        match = map_index[map_index["peak_group"].astype(int) == group]
        if not match.empty and "heatmap" in match.columns:
            heatmap = str(Path("01_per_peak_frame_correlation") / str(match.iloc[0]["heatmap"]))
        heatmaps.append(heatmap)
        labels = str(row.get("frames_present", "")).split(";")
        raw_tiffs.append(";".join(discover_raw_tiffs(data_root, labels[0]) if labels and labels[0] else []))
        category = str(row.get("selection_category", ""))
        if "shift" in category:
            review_questions.append("Does this feature move continuously with pressure, or is it split/merged across group IDs?")
        elif "static" in category:
            review_questions.append("Is this a fixed detector/background/diamond-like feature rather than sample behavior?")
        elif "isolated" in category:
            review_questions.append("Is this a real local peak in the raw .xy and 2D image, or a spotty artifact/grouping failure?")
        else:
            review_questions.append("Does a real local peak exist at this position in each relevant scan?")
    shortlist["contact_sheet"] = paths
    shortlist["per_peak_heatmap"] = heatmaps
    shortlist["source_raw_tiff_candidates"] = raw_tiffs
    shortlist["human_review_question"] = review_questions
    shortlist["source_xy_paths"] = shortlist["xy_paths"]
    return shortlist


def create_report(
    out_dir: Path,
    suite: Path,
    frame_infos: list[FrameInfo],
    metrics: pd.DataFrame,
    review: pd.DataFrame,
    recurring: pd.DataFrame,
    appearing: pd.DataFrame,
    shifting: pd.DataFrame,
    isolated: pd.DataFrame,
    static: pd.DataFrame,
    tier_comp: pd.DataFrame,
    linkages: pd.DataFrame,
    shortlist: pd.DataFrame,
    missing_notes: list[str],
) -> None:
    def table_preview(df: pd.DataFrame, cols: list[str], n: int = 8) -> str:
        if df.empty:
            return "_None found by current heuristic._"
        return df[cols].head(n).to_markdown(index=False)

    pressure_order = ", ".join(f"{info.label} ({info.branch})" for info in frame_infos)
    category_counts = {
        "recurring": int(metrics["scientific_category_labels"].str.contains("recurring").sum()),
        "appearing/disappearing": int(metrics["scientific_category_labels"].str.contains("low_pressure_only|high_pressure_only|decompression_only|hysteresis", regex=True).sum()),
        "systematic_shift": int(metrics["scientific_category_labels"].str.contains("systematic_shift").sum()),
        "isolated_strong": int(metrics["scientific_category_labels"].str.contains("isolated_strong").sum()),
        "likely_static_background": int(metrics["scientific_category_labels"].str.contains("likely_static_background").sum()),
        "uncertain": int(metrics["scientific_category_labels"].str.contains("uncertain").sum()),
    }
    lines = [
        "# Scientific Peak Group Validation",
        "",
        "## Executive Summary",
        "",
        f"- Analyzed `{suite}` with {len(frame_infos)} scans and {len(metrics)} peak groups.",
        "- This report does not retune the detector; it interprets the existing high-recall candidates using cross-pressure behavior.",
        f"- Category counts: {category_counts}.",
        f"- Top persistent groups include: {', '.join(str(int(x)) for x in recurring.head(5)['peak_group']) if not recurring.empty else 'none'}.",
        f"- Top shifting groups include: {', '.join(str(int(x)) for x in shifting.head(5)['peak_group']) if not shifting.empty else 'none'}.",
        "- Labels are cautious: recurring/static/shift candidates are priorities for inspection, not phase assignments.",
        "",
        "## 1. Dataset and Suite Analyzed",
        "",
        f"- Suite: `{suite}`",
        f"- Output folder: `{out_dir}`",
        f"- Pressure order used: {pressure_order}",
        f"- Compression and decompression are distinguishable; decompression is not automatically connected to compression trajectories.",
        "",
        "## 2. Review-Plot Sanity Check",
        "",
        review.to_markdown(index=False),
        "",
        "## 3. Pressure Ordering and Methodology",
        "",
        "Pressure labels were parsed numerically from the frame label. Compression scans are ordered by increasing pressure. Decompression scans are kept as a separate branch after compression for plotting and coverage, but global position fits use compression points unless otherwise noted.",
        "",
        "Correlation maps in this suite are per-peak frame-pair similarity maps: for each peak group, the implementation compares the local/ROI peak signal across frames. Strong correlation means similar behavior for that grouped candidate in those frames; it does not prove that the peak is real or belongs to the sample.",
        "",
        "## 4. Most Persistent Peak Groups",
        "",
        table_preview(recurring, ["peak_group", "median_2theta", "frame_count", "coverage_fraction", "longest_consecutive_run", "tier_a_count", "tier_b_count", "max_roi_area", "notes"]),
        "",
        "## 5. Pressure-Induced Appearance/Disappearance",
        "",
        table_preview(appearing, ["peak_group", "median_2theta", "transition_category", "pressure_interval", "frame_count", "evidence"]),
        "",
        "## 6. Systematically Shifting Peak Groups",
        "",
        table_preview(shifting, ["peak_group", "median_2theta", "frame_count", "pressure_position_slope_deg_per_gpa", "position_fit_r_squared", "spearman_pressure_position", "total_shift_deg", "position_fit_rmse_deg"]),
        "",
        "## 7. Isolated Strong Candidates",
        "",
        table_preview(isolated, ["peak_group", "median_2theta", "frame_count", "max_roi_area", "tier_a_count", "tier_b_count", "frames_present", "notes"]),
        "",
        "## 8. Likely Static/Background Candidates",
        "",
        table_preview(static, ["peak_group", "median_2theta", "frame_count", "coverage_fraction", "position_mad_deg", "pressure_position_slope_deg_per_gpa", "notes"]),
        "",
        "## 9. Tier A+B Versus Tier A-Only Findings",
        "",
        table_preview(tier_comp, ["peak_group", "median_2theta", "tier_ab_frame_count", "tier_a_only_frame_count", "tier_b_contribution", "tier_ab_longest_run", "tier_a_only_longest_run", "interpretation"]),
        "",
        "## 10. Possible Grouping Failures, Splits, and Merges",
        "",
        table_preview(linkages, ["group_id_1", "group_id_2", "reason", "end_pressure_group_1", "start_pressure_group_2", "implied_shift", "confidence", "possible_event_type"]),
        "",
        "## 11. Manual-Review Shortlist",
        "",
        table_preview(shortlist, ["peak_group", "selection_category", "median_2theta", "pressure_min", "pressure_max", "frame_count", "human_review_question", "contact_sheet"], n=12),
        "",
        "## 12. Main Limitations",
        "",
        "- ROI area is taken from available suite peak/prominence-style fields where exact integrated ROI area is not present for every table.",
        "- Fixed-tolerance grouping may fragment moving peaks or merge nearby peaks; `possible_group_linkages.csv` is a warning list, not an automatic correction.",
        "- Static/background labels are hypotheses for prioritization, not definite diamond or gasket assignments.",
        "- Decompression has only one labeled scan in this suite, so hysteresis claims are especially uncertain.",
        *(f"- {note}" for note in missing_notes),
        "",
        "## 13. Recommended Next Experimental Checks",
        "",
        "- Open the manual-review contact sheets and original `.xy` review plots for the shortlisted groups.",
        "- For shifting candidates, verify continuity in local windows and inspect possible neighboring group linkages.",
        "- For likely static/background candidates, compare against raw 2D images and known diamond/gasket/reference peak positions.",
        "- For isolated strong candidates, check raw TIFFs for spotty diffraction, masking artifacts, or peak movement beyond the grouping tolerance.",
    ]
    (out_dir / "scientific_peak_group_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--max-groups", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    suite = args.suite
    peak_dir = suite / "01_per_peak_frame_correlation"
    static_dir = suite / "05_static_peaks_and_overall_correlation"
    out_dir = args.out_dir or (suite / "06_scientific_peak_group_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    peak_table = pd.read_csv(peak_dir / "peak_table.csv")
    if args.max_groups:
        keep = sorted(peak_table["peak_group"].dropna().astype(int).unique())[: args.max_groups]
        peak_table = peak_table[peak_table["peak_group"].astype(int).isin(keep)]
    group_summary = read_optional_csv(peak_dir / "peak_group_summary.csv")
    map_index = read_optional_csv(peak_dir / "per_peak_map_index.csv")
    roi_features = read_optional_csv(peak_dir / "peak_roi_area_features.csv")
    static_scores = read_optional_csv(static_dir / "all_peak_static_scores.csv")
    suspected_static = read_optional_csv(static_dir / "suspected_static_peak_groups.csv")

    frame_infos = parse_frame_info(peak_table["frame"].dropna().unique())
    metrics = build_metrics(peak_table, group_summary, map_index, static_scores, frame_infos)
    metrics = attach_roi_feature_stats(metrics, map_index, roi_features)
    metrics = assign_scientific_categories(metrics)

    if args.max_groups:
        metrics = metrics[metrics["peak_group"].astype(int).isin(peak_table["peak_group"].astype(int).unique())]

    review_rows = []
    all_candidates = read_optional_csv(peak_dir / "all_candidate_table.csv")
    tier_c = read_optional_csv(peak_dir / "tier_c_candidate_table.csv")
    for frame, retained, artifact_note, conclusion in [
        (
            "10.4GPa",
            "weak shoulder/small features around 8-9 deg and 12-16 deg remain marked in Tier A/B",
            "some high-angle and broad-tail markers remain in A+B, consistent with high-recall setting",
            "passes high-recall sanity check; no top-peaks truncation",
        ),
        (
            "2.4GPa_decomp",
            "plausible protrusions remain in A/B across the broad decomp background",
            "many obvious slope/plateau/tail candidates were downgraded to Tier C, but A/B is still intentionally permissive",
            "usable for cross-scan prioritization, not single-scan phase assignment",
        ),
    ]:
        sub_all = all_candidates[all_candidates["frame"] == frame] if not all_candidates.empty else pd.DataFrame()
        sub_ab = peak_table[peak_table["frame"] == frame]
        sub_c = tier_c[tier_c["frame"] == frame] if not tier_c.empty else pd.DataFrame()
        review_rows.append(
            {
                "frame": frame,
                "total_candidates": len(sub_all) if not sub_all.empty else len(sub_ab) + len(sub_c),
                "tier_a_count": int((sub_ab["confidence_tier"] == "A").sum()),
                "tier_b_count": int((sub_ab["confidence_tier"] == "B").sum()),
                "tier_c_count": len(sub_c),
                "visually_retained_weak_features": retained,
                "obvious_artifacts_remaining_in_A+B": artifact_note,
                "conclusion": conclusion,
            }
        )
    review = pd.DataFrame(review_rows)

    recurring = metrics[
        (metrics["coverage_fraction"] >= 0.45) | (metrics["frame_count"] >= 8)
    ].sort_values(["coverage_fraction", "longest_consecutive_run", "max_roi_area"], ascending=False)
    appearing = classify_appearing_disappearing(metrics, frame_infos)
    shifting = metrics[
        metrics["scientific_category_labels"].str.contains("systematic_shift", regex=False)
    ].sort_values(["position_fit_r_squared", "total_shift_deg"], ascending=False)
    isolated = metrics[
        metrics["scientific_category_labels"].str.contains("isolated_strong", regex=False)
    ].sort_values(["max_roi_area", "tier_a_count"], ascending=False)
    static = metrics[
        metrics["scientific_category_labels"].str.contains("likely_static_background", regex=False)
    ].sort_values(["coverage_fraction", "position_mad_deg"], ascending=[False, True])
    tier_comp = tier_a_only_comparison(metrics, peak_table, frame_infos)
    linkages = possible_linkages(metrics)
    shortlist = make_manual_shortlist(metrics, peak_table, map_index, args.data_root, out_dir, frame_infos)

    write_csv(review, out_dir / "review_plot_sanity_check.csv")
    write_csv(recurring, out_dir / "recurring_peak_groups.csv")
    write_csv(appearing, out_dir / "appearing_disappearing_peak_groups.csv")
    write_csv(shifting, out_dir / "shifting_peak_groups.csv")
    write_csv(isolated, out_dir / "isolated_strong_peak_groups.csv")
    write_csv(static, out_dir / "likely_static_background_groups.csv")
    write_csv(tier_comp, out_dir / "tier_ab_vs_tier_a_comparison.csv")
    write_csv(linkages, out_dir / "possible_group_linkages.csv")
    write_csv(shortlist, out_dir / "manual_review_shortlist.csv")
    write_csv(metrics, out_dir / "all_group_scientific_metrics.csv")

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)
    plot_persistence(metrics, plots / "peak_group_persistence.png")
    plot_presence_map(metrics, peak_table, frame_infos, plots / "peak_presence_map_tier_ab.png", tier_a_only=False)
    plot_presence_map(metrics, peak_table, frame_infos, plots / "peak_presence_map_tier_a_only.png", tier_a_only=True)
    plot_category_summary(metrics, plots / "category_summary.png")
    plot_position_trends(metrics, peak_table, frame_infos, plots / "position_vs_pressure")
    if not shortlist.empty:
        plot_roi_trends(shortlist, peak_table, frame_infos, plots / "roi_vs_pressure")

    missing_notes = []
    required = {"confidence_tier", "source_methods", "prominence", "two_theta", "peak_group", "frame", "path"}
    missing = sorted(required - set(peak_table.columns))
    if missing:
        missing_notes.append(f"Missing expected peak_table columns: {', '.join(missing)}")
    if suspected_static.empty:
        missing_notes.append("No suspected_static_peak_groups.csv was available.")
    create_report(out_dir, suite, frame_infos, metrics, review, recurring, appearing, shifting, isolated, static, tier_comp, linkages, shortlist, missing_notes)

    summary = {
        "suite": str(suite),
        "out_dir": str(out_dir),
        "scan_count": len(frame_infos),
        "pressure_order": ";".join(info.label for info in frame_infos),
        "total_candidate_detections": len(peak_table),
        "total_peak_groups": len(metrics),
        "recurring_groups": len(recurring),
        "appearing_disappearing_groups": len(appearing),
        "shifting_groups": len(shifting),
        "isolated_strong_groups": len(isolated),
        "likely_static_background_groups": len(static),
        "manual_review_groups": len(shortlist),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "validation_run_summary.csv", index=False)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
