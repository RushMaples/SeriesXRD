#!/usr/bin/env python3
"""XDI-inspired pressure-window feature maps for high-pressure XRD scans.

This adapts the SXDM/XDI idea from spatial pixels to pressure-series XRD:
each integrated .xy scan is an I(2theta) pattern, each sliding 2theta window is
an ROI, and every pressure/window cell gets physically interpretable
functionals. Existing Tier A/B/C peak tables are consumed as-is; this script
does not tune or rerun the peak detector.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


PRESSURE_RE = re.compile(r"(?P<value>\d+(?:p\d+)?|\d+(?:\.\d+)?)\s*G[PO]a", re.I)
DEFAULT_OUT = Path("outputs/xdi_pressure_window_maps_20260703")
DEFAULT_PEAK_TABLE = Path(
    "outputs/correlation_suite_20260621_high_recall_scored_v2/"
    "01_per_peak_frame_correlation/all_candidate_table.csv"
)
DEFAULT_RAW2D_STATIC = Path("outputs/analysis_v2_20260701/F_raw2d_spot_tracking/spot_static_verdicts.csv")


@dataclass(frozen=True)
class Pattern:
    cell: str
    label: str
    pressure_gpa: float
    path: Path
    two_theta: np.ndarray
    intensity: np.ndarray
    raw_2d_path: str
    decomp: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build XDI-inspired pressure x 2theta-window feature maps."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("Data/Cell_14_integrated"), Path("Data/Cell_29_integrated")],
        help="Integrated .xy files or folders containing .xy files.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--peak-table", type=Path, default=DEFAULT_PEAK_TABLE)
    parser.add_argument("--raw2d-static-table", type=Path, default=DEFAULT_RAW2D_STATIC)
    parser.add_argument("--window-width", type=float, default=1.0)
    parser.add_argument("--window-step", type=float, default=0.5)
    parser.add_argument("--grid-step", type=float, default=0.02)
    parser.add_argument("--min-two-theta", type=float, default=2.0)
    parser.add_argument("--max-two-theta", type=float, default=24.0)
    parser.add_argument(
        "--max-shift-deg",
        type=float,
        default=0.35,
        help="Bounded same-window NCC lag search in degrees.",
    )
    parser.add_argument("--baseline-window-deg", type=float, default=2.0)
    parser.add_argument("--smooth-window-deg", type=float, default=0.12)
    parser.add_argument(
        "--include-decomp",
        action="store_true",
        help="Include decompression scans such as decomp-2p4GPa.xy.",
    )
    return parser.parse_args()


def norm_path(path: str | Path) -> str:
    return Path(str(path)).as_posix()


def pressure_from_text(text: str) -> float | None:
    match = PRESSURE_RE.search(text)
    if not match:
        return None
    return float(match.group("value").replace("p", "."))


def pressure_label(pressure: float, decomp: bool = False) -> str:
    label = f"{pressure:.1f}GPa"
    return f"{label}_decomp" if decomp else label


def infer_cell(path: Path) -> str:
    text = path.as_posix()
    if "Cell_14" in text:
        return "Cell_14"
    if "Cell_29" in text:
        return "Cell_29"
    return "unknown"


def discover_xy(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files.extend(sorted(item.glob("*.xy")))
        elif item.suffix.lower() == ".xy":
            files.append(item)
    return sorted(dict.fromkeys(files), key=lambda p: p.as_posix())


def odd_window(requested: int, size: int) -> int:
    if size <= 3:
        return max(3, size | 1)
    window = min(requested, size if size % 2 else size - 1)
    window = max(5, window)
    if window % 2 == 0:
        window -= 1
    return min(window, size if size % 2 else size - 1)


def load_xy(path: Path, min_two_theta: float, max_two_theta: float | None) -> tuple[np.ndarray, np.ndarray]:
    try:
        arr = np.loadtxt(path)
    except UnicodeDecodeError:
        arr = np.loadtxt(path, encoding="latin1")
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected two-column .xy file: {path}")
    x = np.asarray(arr[:, 0], dtype=float)
    y = np.asarray(arr[:, 1], dtype=float)
    keep = x >= min_two_theta
    if max_two_theta is not None:
        keep &= x <= max_two_theta
    keep &= np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def raw_2d_inventory() -> dict[tuple[str, float], str]:
    candidates: dict[tuple[str, float], tuple[int, str]] = {}
    for root in [Path("Data/Cell_14"), Path("Data/Cell_29"), Path("raw_data/High_Pressure_PXRD/Data/Cell_29")]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.tif")):
            pressure = pressure_from_text(path.name)
            if pressure is None:
                continue
            cell = infer_cell(path)
            if cell == "unknown":
                continue
            name = path.name.lower()
            score = 0
            if "10deg" in name:
                score += 100
            if "rot" in name:
                score += 10
            if "none" in name or "dene" in name:
                score -= 20
            if "none" not in name and "none" not in path.as_posix().lower():
                score += 1
            key = (cell, round(float(pressure), 3))
            value = (score, norm_path(path))
            if key not in candidates or value > candidates[key]:
                candidates[key] = value
    return {key: value[1] for key, value in candidates.items()}


def load_patterns(args: argparse.Namespace) -> list[Pattern]:
    raw_map = raw_2d_inventory()
    patterns: list[Pattern] = []
    for path in discover_xy(args.inputs):
        decomp = "decomp" in path.name.lower()
        if decomp and not args.include_decomp:
            continue
        pressure = pressure_from_text(path.name)
        if pressure is None:
            continue
        cell = infer_cell(path)
        x, y = load_xy(path, args.min_two_theta, args.max_two_theta)
        if len(x) < 20:
            continue
        raw_path = raw_map.get((cell, round(float(pressure), 3)), "")
        patterns.append(
            Pattern(
                cell=cell,
                label=pressure_label(pressure, decomp),
                pressure_gpa=float(pressure),
                path=path,
                two_theta=x,
                intensity=y,
                raw_2d_path=raw_path,
                decomp=decomp,
            )
        )
    return sorted(patterns, key=lambda p: (p.cell, p.pressure_gpa, p.label))


def load_peak_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    peaks = pd.read_csv(path)
    if "path" not in peaks.columns or "two_theta" not in peaks.columns:
        return pd.DataFrame()
    peaks = peaks.copy()
    peaks["path_norm"] = peaks["path"].astype(str).map(norm_path)
    peaks["two_theta"] = pd.to_numeric(peaks["two_theta"], errors="coerce")
    peaks["confidence_tier"] = peaks.get("confidence_tier", "").astype(str).str.upper()
    for column in ["normalized_intensity", "raw_intensity", "prominence", "width_deg", "width_estimate_deg"]:
        if column in peaks.columns:
            peaks[column] = pd.to_numeric(peaks[column], errors="coerce")
    return peaks[np.isfinite(peaks["two_theta"])].copy()


def cell_grid(patterns: list[Pattern], step: float, min_tt: float, max_tt: float | None) -> np.ndarray:
    lower = max(float(p.two_theta.min()) for p in patterns)
    upper = min(float(p.two_theta.max()) for p in patterns)
    lower = max(lower, min_tt)
    if max_tt is not None:
        upper = min(upper, max_tt)
    lower = math.ceil(lower / step) * step
    upper = math.floor(upper / step) * step
    return np.arange(lower, upper + step / 2.0, step)


def window_starts(grid: np.ndarray, width: float, step: float) -> np.ndarray:
    first = math.ceil(float(grid.min()) / step) * step
    last_allowed = float(grid.max()) - width
    last = first + math.floor((last_allowed - first) / step) * step
    if last < first:
        return np.array([], dtype=float)
    return np.arange(first, last + step / 2.0, step)


def preprocess_full(y: np.ndarray, grid_step: float, smooth_deg: float, baseline_deg: float) -> np.ndarray:
    smooth_bins = odd_window(max(5, int(round(smooth_deg / grid_step))), len(y))
    baseline_bins = odd_window(max(smooth_bins + 2, int(round(baseline_deg / grid_step))), len(y))
    smoothed = savgol_filter(y, smooth_bins, polyorder=2)
    baseline = savgol_filter(y, baseline_bins, polyorder=2)
    return smoothed - baseline


def zscore(signal: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 8 or not np.isfinite(arr).any():
        return None
    arr = arr - np.nanmean(arr)
    std = float(np.nanstd(arr))
    if std < 1e-12:
        return None
    return arr / std


def zero_shift_ncc(a: np.ndarray | None, b: np.ndarray | None) -> float:
    aa = zscore(a) if a is not None else None
    bb = zscore(b) if b is not None else None
    if aa is None or bb is None:
        return np.nan
    n = min(len(aa), len(bb))
    if n < 8:
        return np.nan
    return float(np.nanmean(aa[:n] * bb[:n]))


def shifted_ncc(a: np.ndarray | None, b: np.ndarray | None, max_lag_bins: int) -> tuple[float, int]:
    if a is None or b is None:
        return np.nan, 0
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    n = min(len(aa), len(bb))
    if n < 8:
        return np.nan, 0
    aa = aa[:n]
    bb = bb[:n]
    best_score = np.nan
    best_lag = 0
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        if lag < 0:
            left = aa[-lag:]
            right = bb[: n + lag]
        elif lag > 0:
            left = aa[: n - lag]
            right = bb[lag:]
        else:
            left = aa
            right = bb
        if len(left) < max(8, int(0.5 * n)):
            continue
        left = left - np.nanmean(left)
        right = right - np.nanmean(right)
        left_std = float(np.nanstd(left))
        right_std = float(np.nanstd(right))
        if left_std < 1e-12 or right_std < 1e-12:
            continue
        score = float(np.nanmean((left / left_std) * (right / right_std)))
        if np.isnan(best_score) or score > best_score:
            best_score = score
            best_lag = lag
    return best_score, best_lag


def acf_fingerprint(signal: np.ndarray | None) -> np.ndarray | None:
    zz = zscore(signal) if signal is not None else None
    if zz is None:
        return None
    corr = np.correlate(zz, zz, mode="full")
    corr = corr[len(corr) // 2 :]
    if len(corr) < 4 or corr[0] <= 0:
        return None
    fp = corr[1:] / corr[0]
    fp = fp - np.nanmean(fp)
    std = float(np.nanstd(fp))
    if std < 1e-12:
        return None
    return fp / std


def pearson(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return np.nan
    n = min(len(a), len(b))
    if n < 4:
        return np.nan
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def acf_summary(fp: np.ndarray | None, grid_step: float) -> dict[str, float]:
    if fp is None or len(fp) < 3:
        return {
            "acf_lag1": np.nan,
            "acf_mean_abs": np.nan,
            "acf_dom_lag_deg": np.nan,
            "acf_dom_value": np.nan,
        }
    dom_idx = int(np.nanargmax(fp[: min(len(fp), 100)]))
    return {
        "acf_lag1": float(fp[0]),
        "acf_mean_abs": float(np.nanmean(np.abs(fp))),
        "acf_dom_lag_deg": float((dom_idx + 1) * grid_step),
        "acf_dom_value": float(fp[dom_idx]),
    }


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def dominant_peak(peaks: pd.DataFrame) -> dict[str, object]:
    if peaks.empty:
        return {
            "dominant_peak_position": np.nan,
            "dominant_peak_fwhm_deg": np.nan,
            "dominant_peak_tier": "",
            "dominant_peak_group": "",
            "dominant_peak_source": "",
        }
    score_col = "raw_intensity" if "raw_intensity" in peaks.columns else "normalized_intensity"
    scores = pd.to_numeric(peaks.get(score_col, pd.Series(dtype=float)), errors="coerce")
    if scores.notna().any():
        idx = scores.idxmax()
    else:
        idx = peaks.index[0]
    row = peaks.loc[idx]
    width = safe_float(row.get("width_deg", np.nan))
    if not np.isfinite(width) or width <= 0:
        width = safe_float(row.get("width_estimate_deg", np.nan))
    return {
        "dominant_peak_position": safe_float(row.get("two_theta", np.nan)),
        "dominant_peak_fwhm_deg": width,
        "dominant_peak_tier": str(row.get("confidence_tier", "")),
        "dominant_peak_group": row.get("peak_group", ""),
        "dominant_peak_source": str(row.get("source_methods", "")),
    }


def static_table_by_window(path: Path, starts: np.ndarray, width: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if table.empty or "median_2theta" not in table.columns:
        return pd.DataFrame()
    rows = []
    table["median_2theta"] = pd.to_numeric(table["median_2theta"], errors="coerce")
    for cell, sub in table.groupby("cell"):
        for i, start in enumerate(starts):
            stop = start + width
            keep = sub[(sub["median_2theta"] >= start) & (sub["median_2theta"] < stop)]
            counts = keep["verdict"].value_counts().to_dict()
            rows.append(
                {
                    "cell": cell,
                    "window_id": window_id(i, start, stop),
                    "raw2d_static_tracks": int(counts.get("STATIC", 0)),
                    "raw2d_moving_tracks": int(counts.get("MOVING", 0)),
                    "raw2d_undetermined_tracks": int(counts.get("UNDETERMINED", 0)),
                }
            )
    return pd.DataFrame(rows)


def window_id(index: int, start: float, stop: float) -> str:
    return f"w{index:03d}_{start:.2f}_{stop:.2f}"


def window_label(start: float, stop: float) -> str:
    return f"{start:.2f}-{stop:.2f}"


def build_feature_tables(
    args: argparse.Namespace,
    patterns: list[Pattern],
    peaks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[tuple[int, int], np.ndarray | None]], dict[str, dict[tuple[int, int], np.ndarray | None]], dict[str, list[Pattern]], dict[str, np.ndarray]]:
    feature_rows: list[dict[str, object]] = []
    cell_patterns = {cell: list(group) for cell, group in pd.Series(patterns).groupby([p.cell for p in patterns])}
    signals_by_cell: dict[str, dict[tuple[int, int], np.ndarray | None]] = {}
    acf_by_cell: dict[str, dict[tuple[int, int], np.ndarray | None]] = {}
    starts_by_cell: dict[str, np.ndarray] = {}
    max_lag_bins = max(1, int(round(args.max_shift_deg / args.grid_step)))

    for cell, group_patterns in cell_patterns.items():
        group_patterns = sorted(group_patterns, key=lambda p: p.pressure_gpa)
        grid = cell_grid(group_patterns, args.grid_step, args.min_two_theta, args.max_two_theta)
        starts = window_starts(grid, args.window_width, args.window_step)
        starts_by_cell[cell] = starts
        signals: dict[tuple[int, int], np.ndarray | None] = {}
        acfs: dict[tuple[int, int], np.ndarray | None] = {}
        grid_y: list[np.ndarray] = []
        grid_signal: list[np.ndarray] = []

        for pattern in group_patterns:
            y = np.interp(grid, pattern.two_theta, pattern.intensity)
            grid_y.append(y)
            grid_signal.append(
                preprocess_full(
                    y,
                    grid_step=args.grid_step,
                    smooth_deg=args.smooth_window_deg,
                    baseline_deg=args.baseline_window_deg,
                )
            )

        for frame_index, pattern in enumerate(group_patterns):
            path_norm = norm_path(pattern.path)
            pattern_peaks = peaks[peaks["path_norm"] == path_norm] if not peaks.empty else pd.DataFrame()
            for win_index, start in enumerate(starts):
                stop = float(start + args.window_width)
                keep = (grid >= start) & (grid < stop)
                xw = grid[keep]
                raw = grid_y[frame_index][keep]
                sig = grid_signal[frame_index][keep]
                signals[(frame_index, win_index)] = sig
                fp = acf_fingerprint(sig)
                acfs[(frame_index, win_index)] = fp
                win_peaks = pattern_peaks[
                    (pattern_peaks["two_theta"] >= start) & (pattern_peaks["two_theta"] < stop)
                ]
                tiers = win_peaks["confidence_tier"].astype(str).str.upper() if not win_peaks.empty else pd.Series(dtype=str)
                reliable = win_peaks[tiers.isin(["A", "B"])] if not win_peaks.empty else pd.DataFrame()
                dom = dominant_peak(reliable)
                pos = dom["dominant_peak_position"]
                if np.isfinite(pos):
                    local_max_position = pos
                elif len(raw):
                    local_max_position = float(xw[int(np.nanargmax(raw))])
                else:
                    local_max_position = np.nan
                summary = acf_summary(fp, args.grid_step)
                positive_sig = np.clip(sig, 0, None)
                row = {
                    "cell": cell,
                    "pressure_gpa": pattern.pressure_gpa,
                    "frame_label": pattern.label,
                    "frame_index": frame_index,
                    "xy_path": norm_path(pattern.path),
                    "raw_2d_path": pattern.raw_2d_path,
                    "window_id": window_id(win_index, start, stop),
                    "window_index": win_index,
                    "window_start": float(start),
                    "window_end": float(stop),
                    "window_center": float((start + stop) / 2.0),
                    "window_label": window_label(start, stop),
                    "roi_area": float(np.trapz(raw, xw)) if len(xw) else np.nan,
                    "max_intensity": float(np.nanmax(raw)) if len(raw) else np.nan,
                    "local_baseline_corrected_intensity": float(np.nanmean(positive_sig)) if len(sig) else np.nan,
                    "baseline_corrected_area": float(np.trapz(positive_sig, xw)) if len(xw) else np.nan,
                    "tier_a_peak_count": int((tiers == "A").sum()),
                    "tier_ab_peak_count": int(tiers.isin(["A", "B"]).sum()),
                    "tier_c_candidate_count": int((tiers == "C").sum()),
                    "tier_ab_peak_density": float(tiers.isin(["A", "B"]).sum() / args.window_width),
                    "tier_c_candidate_density": float((tiers == "C").sum() / args.window_width),
                    "local_max_position": local_max_position,
                    **dom,
                    **summary,
                    "ncc_zero_to_previous": np.nan,
                    "ncc_to_previous": np.nan,
                    "best_shift_deg_to_previous": np.nan,
                    "ncc_zero_to_reference": np.nan,
                    "ncc_to_reference": np.nan,
                    "best_shift_deg_to_reference": np.nan,
                    "acf_to_previous": np.nan,
                    "acf_similarity_change": np.nan,
                    "acf_to_reference": np.nan,
                }
                feature_rows.append(row)

        for win_index, start in enumerate(starts):
            for frame_index in range(len(group_patterns)):
                row_index = next(
                    idx
                    for idx, row in enumerate(feature_rows)
                    if row["cell"] == cell
                    and row["frame_index"] == frame_index
                    and row["window_index"] == win_index
                )
                current = signals[(frame_index, win_index)]
                current_acf = acfs[(frame_index, win_index)]
                reference = signals[(0, win_index)]
                reference_acf = acfs[(0, win_index)]
                if frame_index > 0:
                    previous = signals[(frame_index - 1, win_index)]
                    previous_acf = acfs[(frame_index - 1, win_index)]
                    best, lag = shifted_ncc(previous, current, max_lag_bins)
                    feature_rows[row_index]["ncc_zero_to_previous"] = zero_shift_ncc(previous, current)
                    feature_rows[row_index]["ncc_to_previous"] = best
                    feature_rows[row_index]["best_shift_deg_to_previous"] = float(lag * args.grid_step)
                    acf_prev = pearson(previous_acf, current_acf)
                    feature_rows[row_index]["acf_to_previous"] = acf_prev
                    feature_rows[row_index]["acf_similarity_change"] = 1.0 - acf_prev if np.isfinite(acf_prev) else np.nan
                best_ref, lag_ref = shifted_ncc(reference, current, max_lag_bins)
                feature_rows[row_index]["ncc_zero_to_reference"] = zero_shift_ncc(reference, current)
                feature_rows[row_index]["ncc_to_reference"] = best_ref
                feature_rows[row_index]["best_shift_deg_to_reference"] = float(lag_ref * args.grid_step)
                feature_rows[row_index]["acf_to_reference"] = pearson(reference_acf, current_acf)

        signals_by_cell[cell] = signals
        acf_by_cell[cell] = acfs
        cell_patterns[cell] = group_patterns

    features = pd.DataFrame(feature_rows)
    static_windows = []
    for cell, starts in starts_by_cell.items():
        static_windows.append(static_table_by_window(args.raw2d_static_table, starts, args.window_width))
    raw2d_static = pd.concat([t for t in static_windows if not t.empty], ignore_index=True) if static_windows else pd.DataFrame()
    if not raw2d_static.empty:
        features = features.merge(raw2d_static, on=["cell", "window_id"], how="left")
    for col in ["raw2d_static_tracks", "raw2d_moving_tracks", "raw2d_undetermined_tracks"]:
        if col not in features.columns:
            features[col] = 0
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0).astype(int)

    summary = summarize_windows(features)
    features = features.merge(
        summary[["cell", "window_id", "static_background_score"]],
        on=["cell", "window_id"],
        how="left",
    )
    return features, summary, signals_by_cell, acf_by_cell, cell_patterns, starts_by_cell


def finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().any():
        return float(values.median())
    return np.nan


def finite_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() > 1:
        return float(values.std())
    return np.nan


def slope_vs_pressure(group: pd.DataFrame, column: str) -> float:
    subset = group[["pressure_gpa", column]].copy()
    subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset.dropna()
    if len(subset) < 3 or subset["pressure_gpa"].nunique() < 3:
        return np.nan
    x = subset["pressure_gpa"].to_numpy(float)
    y = subset[column].to_numpy(float)
    return float(np.polyfit(x, y, 1)[0])


def robust_scale(values: pd.Series, inverse: bool = False) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(np.zeros(len(vals)), index=vals.index)
    lo = float(vals.quantile(0.05))
    hi = float(vals.quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        scaled = pd.Series(np.zeros(len(vals)), index=vals.index)
    else:
        scaled = ((vals - lo) / (hi - lo)).clip(0, 1).fillna(0)
    return 1.0 - scaled if inverse else scaled


def summarize_windows(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, window_id_value), group in features.groupby(["cell", "window_id"]):
        group = group.sort_values("pressure_gpa")
        present = pd.to_numeric(group["tier_ab_peak_count"], errors="coerce").fillna(0) > 0
        tier_ab_counts = pd.to_numeric(group["tier_ab_peak_count"], errors="coerce").fillna(0)
        tier_c_counts = pd.to_numeric(group["tier_c_candidate_count"], errors="coerce").fillna(0)
        transitions = int((present.astype(int).diff().abs() == 1).sum())
        pos_slope = slope_vs_pressure(group, "dominant_peak_position")
        local_max_slope = slope_vs_pressure(group, "local_max_position")
        mean_ab = float(tier_ab_counts.mean())
        rows.append(
            {
                "cell": cell,
                "window_id": window_id_value,
                "window_label": group["window_label"].iloc[0],
                "window_start": float(group["window_start"].iloc[0]),
                "window_end": float(group["window_end"].iloc[0]),
                "window_center": float(group["window_center"].iloc[0]),
                "n_frames": int(len(group)),
                "peak_presence_fraction": float(present.mean()),
                "presence_transitions": transitions,
                "median_roi_area": finite_median(group["roi_area"]),
                "median_baseline_corrected_area": finite_median(group["baseline_corrected_area"]),
                "median_tier_ab_peak_count": finite_median(group["tier_ab_peak_count"]),
                "max_tier_ab_peak_count": int(pd.to_numeric(group["tier_ab_peak_count"], errors="coerce").fillna(0).max()),
                "tier_ab_count_span": float(tier_ab_counts.max() - tier_ab_counts.min()),
                "tier_ab_count_cv": float(tier_ab_counts.std() / mean_ab) if mean_ab > 0 and len(tier_ab_counts) > 1 else 0.0,
                "median_tier_c_candidate_count": finite_median(group["tier_c_candidate_count"]),
                "tier_c_count_span": float(tier_c_counts.max() - tier_c_counts.min()),
                "median_ncc_zero_to_previous": finite_median(group["ncc_zero_to_previous"]),
                "median_ncc_to_previous": finite_median(group["ncc_to_previous"]),
                "median_abs_shift_to_previous": finite_median(group["best_shift_deg_to_previous"].abs()),
                "median_abs_shift_to_reference": finite_median(group["best_shift_deg_to_reference"].abs()),
                "median_acf_similarity_change": finite_median(group["acf_similarity_change"]),
                "median_dominant_fwhm_deg": finite_median(group["dominant_peak_fwhm_deg"]),
                "dominant_position_slope_deg_per_gpa": pos_slope,
                "local_max_position_slope_deg_per_gpa": local_max_slope,
                "raw2d_static_tracks": int(group["raw2d_static_tracks"].max()),
                "raw2d_moving_tracks": int(group["raw2d_moving_tracks"].max()),
                "raw2d_undetermined_tracks": int(group["raw2d_undetermined_tracks"].max()),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["signal_strength_score"] = summary.groupby("cell", group_keys=False)["median_baseline_corrected_area"].apply(robust_scale)
    summary["zero_ncc_score"] = ((summary["median_ncc_zero_to_previous"] - 0.55) / 0.40).clip(0, 1).fillna(0)
    summary["small_prev_shift_score"] = (1.0 - (summary["median_abs_shift_to_previous"] / 0.12)).clip(0, 1).fillna(0)
    slope = summary["dominant_position_slope_deg_per_gpa"].where(
        summary["dominant_position_slope_deg_per_gpa"].notna(),
        summary["local_max_position_slope_deg_per_gpa"],
    )
    summary["small_slope_score"] = (1.0 - (slope.abs() / 0.012)).clip(0, 1).fillna(0)
    summary["persistence_score"] = summary["peak_presence_fraction"].clip(0, 1).fillna(0)
    summary["static_background_score"] = (
        0.25 * summary["signal_strength_score"]
        + 0.25 * summary["zero_ncc_score"]
        + 0.25 * summary["small_prev_shift_score"]
        + 0.15 * summary["small_slope_score"]
        + 0.10 * summary["persistence_score"]
    )
    summary["motion_score"] = (
        summary["median_abs_shift_to_reference"].fillna(0).clip(0, 0.35) / 0.35
        + slope.abs().fillna(0).clip(0, 0.06) / 0.06
    ) / 2.0
    max_count = summary["max_tier_ab_peak_count"].replace(0, np.nan)
    span_score = (summary["tier_ab_count_span"] / max_count).clip(0, 1).fillna(0)
    cv_score = (summary["tier_ab_count_cv"] / 0.6).clip(0, 1).fillna(0)
    presence_score = (
        0.5 * (summary["presence_transitions"].clip(0, 3) / 3.0)
        + 0.5 * (1.0 - (summary["peak_presence_fraction"] - 0.5).abs() * 2.0).clip(0, 1)
    )
    tier_c_span_score = (summary["tier_c_count_span"] / 8.0).clip(0, 1).fillna(0)
    summary["appearance_disappearance_score"] = (
        0.35 * span_score
        + 0.35 * cv_score
        + 0.20 * presence_score
        + 0.10 * tier_c_span_score
    )
    return summary.sort_values(["cell", "window_start"]).reset_index(drop=True)


def write_matrix_csv(path: Path, row_labels: list[str], col_labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *col_labels])
        for label, row in zip(row_labels, matrix):
            writer.writerow([label, *[("" if not np.isfinite(v) else f"{v:.6g}") for v in row]])


def plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    center_zero: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(matrix, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size and (vmin is None or vmax is None):
        if center_zero:
            limit = float(np.nanpercentile(np.abs(finite), 95))
            vmin = -limit
            vmax = limit
        else:
            vmin = float(np.nanpercentile(finite, 5))
            vmax = float(np.nanpercentile(finite, 95))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    fig_w = min(18.0, max(7.0, 0.28 * len(col_labels) + 2.5))
    fig_h = min(12.0, max(4.8, 0.35 * len(row_labels) + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("2theta window" if len(col_labels) != len(row_labels) else "pressure")
    ax.set_ylabel("pressure")
    xtick_step = max(1, int(math.ceil(len(col_labels) / 24)))
    ytick_step = max(1, int(math.ceil(len(row_labels) / 24)))
    ax.set_xticks(range(0, len(col_labels), xtick_step))
    ax.set_xticklabels([col_labels[i] for i in range(0, len(col_labels), xtick_step)], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(0, len(row_labels), ytick_step))
    ax.set_yticklabels([row_labels[i] for i in range(0, len(row_labels), ytick_step)], fontsize=8)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_pressure_window_heatmaps(out_dir: Path, features: pd.DataFrame) -> pd.DataFrame:
    heatmap_dir = out_dir / "heatmaps"
    matrix_dir = out_dir / "matrices"
    specs = [
        ("roi_area", "ROI integrated area", "magma", None, None, False),
        ("tier_ab_peak_density", "Tier A+B peak density", "magma", 0.0, None, False),
        ("ncc_to_previous", "NCC to previous pressure", "viridis", -1.0, 1.0, False),
        ("best_shift_deg_to_previous", "Best shift to previous pressure (deg)", "coolwarm", None, None, True),
        ("acf_similarity_change", "ACF similarity change vs previous", "magma", 0.0, None, False),
        ("dominant_peak_fwhm_deg", "Dominant peak FWHM (deg)", "magma", 0.0, None, False),
        ("static_background_score", "Static/background score", "viridis", 0.0, 1.0, False),
    ]
    rows = []
    for cell, group in features.groupby("cell"):
        group = group.sort_values(["pressure_gpa", "window_start"])
        row_order = (
            group[["pressure_gpa", "frame_label"]]
            .drop_duplicates()
            .sort_values("pressure_gpa")["frame_label"]
            .tolist()
        )
        col_order = (
            group[["window_start", "window_label"]]
            .drop_duplicates()
            .sort_values("window_start")["window_label"]
            .tolist()
        )
        for feature, title, cmap, vmin, vmax, center_zero in specs:
            matrix_df = group.pivot(index="frame_label", columns="window_label", values=feature)
            matrix_df = matrix_df.reindex(index=row_order, columns=col_order)
            csv_path = matrix_dir / cell / f"{feature}.csv"
            png_path = heatmap_dir / cell / f"{feature}.png"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            matrix_df.to_csv(csv_path)
            plot_heatmap(
                png_path,
                matrix_df.to_numpy(float),
                row_order,
                col_order,
                f"{cell}: {title}",
                feature,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center_zero=center_zero,
            )
            rows.append(
                {
                    "cell": cell,
                    "feature": feature,
                    "title": title,
                    "csv": norm_path(csv_path),
                    "heatmap": norm_path(png_path),
                }
            )
    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "heatmap_index.csv", index=False)
    return index


def save_cross_pressure_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    cell_patterns: dict[str, list[Pattern]],
    starts_by_cell: dict[str, np.ndarray],
    signals_by_cell: dict[str, dict[tuple[int, int], np.ndarray | None]],
) -> pd.DataFrame:
    rows = []
    max_lag_bins = max(1, int(round(args.max_shift_deg / args.grid_step)))
    base = out_dir / "cross_pressure"
    for cell, patterns in cell_patterns.items():
        labels = [p.label for p in patterns]
        starts = starts_by_cell[cell]
        signals = signals_by_cell[cell]
        for win_index, start in enumerate(starts):
            stop = float(start + args.window_width)
            n = len(patterns)
            zero = np.full((n, n), np.nan)
            best = np.full((n, n), np.nan)
            shift = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(n):
                    a = signals[(i, win_index)]
                    b = signals[(j, win_index)]
                    zero[i, j] = zero_shift_ncc(a, b)
                    score, lag = shifted_ncc(a, b, max_lag_bins)
                    best[i, j] = score
                    shift[i, j] = lag * args.grid_step
            wid = window_id(win_index, start, stop)
            label = window_label(start, stop)
            for matrix_name, matrix, cmap, vmin, vmax, center_zero, cbar in [
                ("zero_shift_ncc", zero, "viridis", -1.0, 1.0, False, "zero-shift NCC"),
                ("best_shift_ncc", best, "viridis", -1.0, 1.0, False, "best-shift NCC"),
                ("best_shift_deg", shift, "coolwarm", None, None, True, "best shift (deg)"),
            ]:
                csv_path = base / cell / f"{matrix_name}_matrices" / f"{wid}_{matrix_name}.csv"
                png_path = base / cell / f"{matrix_name}_heatmaps" / f"{wid}_{matrix_name}.png"
                write_matrix_csv(csv_path, labels, labels, matrix)
                plot_heatmap(
                    png_path,
                    matrix,
                    labels,
                    labels,
                    f"{cell} {label}: {matrix_name}",
                    cbar,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    center_zero=center_zero,
                )
                rows.append(
                    {
                        "cell": cell,
                        "window_id": wid,
                        "window_label": label,
                        "window_start": float(start),
                        "window_end": float(stop),
                        "matrix_type": matrix_name,
                        "csv": norm_path(csv_path),
                        "heatmap": norm_path(png_path),
                        "max_shift_deg": args.max_shift_deg,
                    }
                )
    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "same_window_cross_pressure_index.csv", index=False)
    return index


def overlap_fraction(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    denom = min(a_end - a_start, b_end - b_start)
    return float(overlap / denom) if denom > 0 else 0.0


def save_within_frame_acf_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    cell_patterns: dict[str, list[Pattern]],
    starts_by_cell: dict[str, np.ndarray],
    acf_by_cell: dict[str, dict[tuple[int, int], np.ndarray | None]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    overlap_rows = []
    base = out_dir / "within_frame_acf"
    for cell, patterns in cell_patterns.items():
        starts = starts_by_cell[cell]
        labels = [window_label(start, start + args.window_width) for start in starts]
        n_windows = len(starts)
        overlap = np.zeros((n_windows, n_windows), dtype=float)
        for i, a_start in enumerate(starts):
            for j, b_start in enumerate(starts):
                frac = overlap_fraction(a_start, a_start + args.window_width, b_start, b_start + args.window_width)
                overlap[i, j] = frac
                overlap_rows.append(
                    {
                        "cell": cell,
                        "window_a": labels[i],
                        "window_b": labels[j],
                        "overlap_fraction": frac,
                    }
                )
        overlap_csv = base / cell / "window_overlap_fraction.csv"
        write_matrix_csv(overlap_csv, labels, labels, overlap)
        for frame_index, pattern in enumerate(patterns):
            full = np.full((n_windows, n_windows), np.nan)
            for i in range(n_windows):
                for j in range(n_windows):
                    full[i, j] = pearson(acf_by_cell[cell][(frame_index, i)], acf_by_cell[cell][(frame_index, j)])
            nonoverlap = full.copy()
            nonoverlap[overlap > 0] = np.nan
            full_csv = base / cell / "all_windows_matrices" / f"{pattern.label}_acf_window_window.csv"
            non_csv = base / cell / "nonoverlap_matrices" / f"{pattern.label}_acf_window_window_nonoverlap.csv"
            full_png = base / cell / "all_windows_heatmaps" / f"{pattern.label}_acf_window_window.png"
            non_png = base / cell / "nonoverlap_heatmaps" / f"{pattern.label}_acf_window_window_nonoverlap.png"
            write_matrix_csv(full_csv, labels, labels, full)
            write_matrix_csv(non_csv, labels, labels, nonoverlap)
            plot_heatmap(
                full_png,
                full,
                labels,
                labels,
                f"{cell} {pattern.label}: window-window ACF similarity",
                "ACF Pearson",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
            )
            plot_heatmap(
                non_png,
                nonoverlap,
                labels,
                labels,
                f"{cell} {pattern.label}: non-overlap window ACF similarity",
                "ACF Pearson",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
            )
            rows.extend(
                [
                    {
                        "cell": cell,
                        "frame_label": pattern.label,
                        "pressure_gpa": pattern.pressure_gpa,
                        "mode": "all_windows",
                        "csv": norm_path(full_csv),
                        "heatmap": norm_path(full_png),
                    },
                    {
                        "cell": cell,
                        "frame_label": pattern.label,
                        "pressure_gpa": pattern.pressure_gpa,
                        "mode": "nonoverlap_only",
                        "csv": norm_path(non_csv),
                        "heatmap": norm_path(non_png),
                    },
                ]
            )
    index = pd.DataFrame(rows)
    overlap_table = pd.DataFrame(overlap_rows)
    index.to_csv(out_dir / "within_frame_acf_index.csv", index=False)
    overlap_table.to_csv(out_dir / "window_overlap_fractions_long.csv", index=False)
    return index, overlap_table


def top_table_markdown(df: pd.DataFrame, columns: list[str], n: int = 8) -> str:
    if df.empty:
        return "_No rows._"
    shown = df[columns].head(n).copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in shown.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}" if np.isfinite(value) else "")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_summary_tables(out_dir: Path, features: pd.DataFrame, summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    feature_path = tables_dir / "pressure_window_feature_table.csv"
    summary_path = tables_dir / "window_motion_summary.csv"
    features.to_csv(feature_path, index=False)
    summary.to_csv(summary_path, index=False)

    static_candidates = summary.sort_values("static_background_score", ascending=False).copy()
    static_candidates.to_csv(tables_dir / "static_background_window_candidates.csv", index=False)

    moving = summary.sort_values("motion_score", ascending=False).copy()
    moving.to_csv(tables_dir / "systematically_shifting_windows.csv", index=False)

    appearance = summary.sort_values("appearance_disappearance_score", ascending=False).copy()
    appearance.to_csv(tables_dir / "appearance_disappearance_windows.csv", index=False)

    raw2d = pd.concat(
        [
            static_candidates.head(12),
            moving.head(12),
            appearance.head(12),
            summary[(summary["raw2d_static_tracks"] > 0) | (summary["raw2d_moving_tracks"] > 0)],
        ],
        ignore_index=True,
    ).drop_duplicates(["cell", "window_id"])
    raw2d["raw2d_priority"] = raw2d[
        ["static_background_score", "motion_score", "appearance_disappearance_score"]
    ].max(axis=1)
    raw2d["raw2d_priority"] += (raw2d["raw2d_static_tracks"] > 0).astype(float) * 0.50
    raw2d["raw2d_priority"] += (raw2d["raw2d_moving_tracks"] > 0).astype(float) * 0.15
    raw2d = raw2d.sort_values("raw2d_priority", ascending=False)
    raw2d.to_csv(tables_dir / "windows_to_check_in_raw2d.csv", index=False)

    return {
        "features": features,
        "summary": summary,
        "static": static_candidates,
        "moving": moving,
        "appearance": appearance,
        "raw2d": raw2d,
    }


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    patterns: list[Pattern],
    tables: dict[str, pd.DataFrame],
    heatmap_index: pd.DataFrame,
    cross_index: pd.DataFrame,
    within_index: pd.DataFrame,
) -> None:
    summary = tables["summary"]
    static = tables["static"]
    moving = tables["moving"]
    appearance = tables["appearance"]
    raw2d = tables["raw2d"]
    cells = sorted({p.cell for p in patterns})
    pattern_count = len(patterns)
    pressure_text = ", ".join(
        f"{cell}: {min(p.pressure_gpa for p in patterns if p.cell == cell):.1f}-"
        f"{max(p.pressure_gpa for p in patterns if p.cell == cell):.1f} GPa"
        for cell in cells
    )
    readme = f"""# XDI-Inspired Pressure-Window Maps

This folder contains an XDI-inspired feature-mapping workflow for the pressure
series XRD data. The Hrubiak/Smith/Shen SXDM-XDI paper treats each spatial
pixel as a diffraction pattern and maps physically meaningful functionals. Here
the spatial axis is replaced by pressure, and each sliding 2theta window is an
ROI.

## What Was Computed

- Integrated scans processed: {pattern_count}
- Pressure coverage: {pressure_text}
- Window width: {args.window_width:.2f} deg
- Window step: {args.window_step:.2f} deg
- NCC shift bound: +/- {args.max_shift_deg:.2f} deg
- Peak detector status: existing Tier A/B/C tables were read as-is; detector
  thresholds were not tuned or rerun.

## Main Outputs

- `tables/pressure_window_feature_table.csv`: one row per pressure x window.
- `heatmap_index.csv`: pressure x window heatmaps for area, peak density, NCC,
  shift, ACF change, FWHM, and static/background score.
- `same_window_cross_pressure_index.csv`: pressure x pressure zero-shift NCC,
  best-shift NCC, and best-shift matrices for every same 2theta window.
- `within_frame_acf_index.csv`: window x window ACF maps for every pressure,
  including all-window and non-overlapping-window versions.
- `tables/static_background_window_candidates.csv`: correlation-based
  static/background candidates.
- `tables/windows_to_check_in_raw2d.csv`: windows that deserve raw detector
  inspection.

## Important Interpretation

The static/background score is a triage score, not a final proof. A window gets
a high score when it has signal, high zero-shift correlation, small bounded NCC
shift, stable dominant/local position, and persistent peaks. That can find
candidate static regions, but true static peak calls still require raw-2D spot
centroid tracking because 1D integration mixes sample spots, diamond/gasket
spots, texture, and overlapping reflections.
"""
    report = f"""# Concise Scientific Report

## How This Adapts SXDM/XDI

SXDM-XDI maps a diffraction pattern at every spatial position and then maps
functionals such as phase, strain, or texture. This project uses the same
philosophy, but the map axes are pressure and 2theta window. Each heatmap cell
answers: "what does this local piece of the XRD pattern do at this pressure?"

That means correlation is no longer the final claim. It is a diagnostic layer
that tells us where to look for compression, phase changes, static artifacts,
or raw-2D checks.

## Pressure-Stable Regions

These are windows with high static/background score. They are stable in 1D, so
they are candidates for static/background features, but not automatically
confirmed static peaks:

{top_table_markdown(static, ["cell", "window_label", "static_background_score", "median_ncc_zero_to_previous", "median_abs_shift_to_previous", "dominant_position_slope_deg_per_gpa", "raw2d_static_tracks"], 10)}

## Systematically Shifting Regions

These windows show the strongest correlation-based motion. Positive dominant
position or local-maximum slope generally means peaks move to higher 2theta
under pressure, which is the expected compression signature:

{top_table_markdown(moving, ["cell", "window_label", "motion_score", "median_abs_shift_to_reference", "dominant_position_slope_deg_per_gpa", "local_max_position_slope_deg_per_gpa", "raw2d_moving_tracks"], 10)}

## Appearance / Disappearance Candidates

These windows have changing peak counts/candidate density across pressure and
should be checked against indexed/refined phase assignments:

{top_table_markdown(appearance, ["cell", "window_label", "appearance_disappearance_score", "peak_presence_fraction", "presence_transitions", "max_tier_ab_peak_count"], 10)}

## Likely Static / Background Regions

The correlation workflow can shortlist stable regions, but the existing raw-2D
tracking remains the authority. In the current v2 result, Cell_29 has one
confirmed raw-2D static spot near 19.27 deg; Cell_14 is underpowered for static
spot judgment. Any 1D static candidate away from that confirmed region should
be treated as "manual-review required" rather than masked blindly.

## Raw 2D Manual Checks

Start with these windows because they are either static-like, strongly moving,
appearance/disappearance-like, or overlap with existing raw-2D track evidence:

{top_table_markdown(raw2d, ["cell", "window_label", "raw2d_priority", "static_background_score", "motion_score", "appearance_disappearance_score", "raw2d_static_tracks", "raw2d_moving_tracks"], 16)}

## Static-Peak Caveat

Correlation can overestimate static peaks when a strong narrow artifact
dominates a window, when windows overlap a fixed diamond/gasket spot, or when a
sample peak happens to stay inside the same broad window. It can underestimate
static peaks when spotty texture makes a static spot intermittent, when a
static spot overlaps a moving sample peak, or when azimuthal integration smears
localized detector spots. The safest workflow is therefore:

1. use these pressure-window maps to shortlist stable windows;
2. inspect the local .xy plot and raw 2D detector image;
3. confirm static behavior with raw-2D centroid tracking;
4. only then mask confirmed non-sample peaks for final phase/EOS analysis.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "xdi_pressure_window_report.md").write_text(report, encoding="utf-8")

    manifest = pd.DataFrame(
        [
            {"name": "feature_rows", "count": len(tables["features"])},
            {"name": "window_summary_rows", "count": len(summary)},
            {"name": "pressure_window_heatmaps", "count": len(heatmap_index)},
            {"name": "cross_pressure_matrices", "count": len(cross_index)},
            {"name": "within_frame_acf_matrices", "count": len(within_index)},
        ]
    )
    manifest.to_csv(out_dir / "manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns(args)
    if not patterns:
        raise SystemExit("No integrated .xy pressure scans found.")
    peaks = load_peak_table(args.peak_table)
    features, summary, signals, acfs, cell_patterns, starts = build_feature_tables(args, patterns, peaks)
    tables = save_summary_tables(args.out_dir, features, summary)
    heatmap_index = save_pressure_window_heatmaps(args.out_dir, features)
    cross_index = save_cross_pressure_outputs(args.out_dir, args, cell_patterns, starts, signals)
    within_index, _ = save_within_frame_acf_outputs(args.out_dir, args, cell_patterns, starts, acfs)
    write_report(args.out_dir, args, patterns, tables, heatmap_index, cross_index, within_index)
    print(f"Wrote XDI-inspired pressure-window maps to {args.out_dir}")
    print(f"Feature rows: {len(features)}")
    print(f"Window summaries: {len(summary)}")
    print(f"Heatmaps: {len(heatmap_index)}")
    print(f"Cross-pressure matrix outputs: {len(cross_index)}")
    print(f"Within-frame ACF outputs: {len(within_index)}")


if __name__ == "__main__":
    main()
