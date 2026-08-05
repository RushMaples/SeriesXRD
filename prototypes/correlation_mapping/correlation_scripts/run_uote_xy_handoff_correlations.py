#!/usr/bin/env python3
"""Run the four correlation families on the UOTe processed-XY handoff.

The handoff contains 56 independent pressure ladders. This runner keeps those
ladders separate, uses the teammate spot-track table as the per-peak registry,
and aggregates like pressure pairs only after each scan has been analyzed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import rankdata


WAVELENGTH_A = 0.3066
PREVIOUS_WAVELENGTH_A = 0.4133


@dataclass(frozen=True)
class Frame:
    frame: int
    scan: str
    pressure: float
    pressure_index: int
    original_filename: str


@dataclass(frozen=True)
class Track:
    track: int
    n_points: int
    n_frames: int
    p_min: float
    p_max: float
    d0: float
    d_min: float
    d_max: float
    dd_dp: float
    best_frame: int
    hkl: str
    match_d: float
    match_delta_pct: float
    intensity_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff_dir", type=Path)
    parser.add_argument("tracks_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--previous-root",
        type=Path,
        default=Path("outputs/analysis_v2_20260701"),
        help="Corrected previous UOTe result used for physical comparison.",
    )
    parser.add_argument("--channels", default="spots,fit")
    parser.add_argument("--snr-threshold", type=float, default=2.5)
    parser.add_argument("--peak-search-half-width", type=float, default=0.12)
    parser.add_argument("--roi-half-width", type=float, default=0.06)
    parser.add_argument("--position-tolerance", type=float, default=0.06)
    parser.add_argument("--window-width", type=float, default=5.0)
    parser.add_argument("--window-step", type=float, default=1.0)
    parser.add_argument("--window-shift-tolerance", type=float, default=1.0)
    parser.add_argument("--grid-step", type=float, default=0.02)
    parser.add_argument("--min-two-theta", type=float, default=2.0)
    parser.add_argument("--max-two-theta", type=float, default=32.0)
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Optional smoke-test limit. Full analysis uses every scan.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.") or "item"


def format_value(value: float | int | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return ""
    return f"{float(value):.8g}"


def read_manifest(path: Path, max_scans: int | None) -> tuple[list[Frame], list[dict[str, str]], list[float], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    scans = sorted({row["scan"] for row in rows})
    if max_scans is not None:
        scans = scans[: max(1, max_scans)]
    selected = [row for row in rows if row["scan"] in scans]
    good = [row for row in selected if row["cover_excluded"] == "0"]
    pressures = sorted({float(row["pressure_GPa"]) for row in good})
    pressure_to_index = {pressure: index for index, pressure in enumerate(pressures)}
    frames = [
        Frame(
            frame=int(row["frame"]),
            scan=row["scan"],
            pressure=float(row["pressure_GPa"]),
            pressure_index=pressure_to_index[float(row["pressure_GPa"])],
            original_filename=row["filename"],
        )
        for row in good
    ]
    frames.sort(key=lambda item: item.frame)
    return frames, selected, pressures, scans


def read_tracks(path: Path) -> list[Track]:
    tracks: list[Track] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if not row.get("track", "").strip():
                continue
            tracks.append(
                Track(
                    track=int(row["track"]),
                    n_points=int(row["n_points"]),
                    n_frames=int(row["n_frames"]),
                    p_min=float(row["p_min_gpa"]),
                    p_max=float(row["p_max_gpa"]),
                    d0=float(row["d0_A"]),
                    d_min=float(row["d_min_A"]),
                    d_max=float(row["d_max_A"]),
                    dd_dp=float(row["dd_dp_A_per_gpa"]),
                    best_frame=int(row["best_frame"]),
                    hkl=row["match_hkl"].strip(),
                    match_d=float(row["match_d_calc_A"]),
                    match_delta_pct=float(row["match_delta_pct"]),
                    intensity_max=float(row["intensity_max"]),
                )
            )
    return sorted(tracks, key=lambda item: item.track)


def channel_folder(channel: str) -> str:
    if channel == "spots":
        return "spots_channel"
    if channel == "fit":
        return "fit_channel"
    raise ValueError(f"Unknown channel: {channel}")


def load_channel(handoff_dir: Path, channel: str, frames: list[Frame]) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    paths: list[Path] = []
    missing: list[Path] = []
    ambiguous: list[tuple[Path, list[Path]]] = []
    for frame in frames:
        folder = handoff_dir / channel_folder(channel) / frame.scan
        legacy_path = folder / f"frame_{frame.frame:04d}.xy"
        if legacy_path.is_file():
            paths.append(legacy_path)
            continue
        matches = sorted(folder.glob(f"frame_{frame.frame:04d}_*GPa.xy"))
        if len(matches) == 1:
            paths.append(matches[0])
        elif not matches:
            missing.append(legacy_path)
        else:
            ambiguous.append((legacy_path, matches))
    if missing:
        raise FileNotFoundError(f"{len(missing)} expected {channel} files are missing; first: {missing[0]}")
    if ambiguous:
        expected, matches = ambiguous[0]
        raise FileNotFoundError(
            f"{len(ambiguous)} expected {channel} frames have ambiguous filename matches; "
            f"first frame {expected.stem} matched {len(matches)} files"
        )
    first = np.loadtxt(paths[0], comments="#")
    x = first[:, 0].astype(float)
    intensity = np.empty((len(paths), len(x)), dtype=np.float32)
    intensity[0] = first[:, 1]
    for index, path in enumerate(paths[1:], start=1):
        data = np.loadtxt(path, comments="#")
        if data.shape != first.shape or not np.array_equal(data[:, 0], x):
            raise ValueError(f"Inconsistent 2theta grid in {path}")
        intensity[index] = data[:, 1]
    if not np.all(np.isfinite(intensity)):
        raise ValueError(f"Non-finite values found in {channel} channel")
    return x, intensity, paths


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(float, copy=True)
    floor = np.nanpercentile(values, 5, axis=1, keepdims=True)
    values -= floor
    scale = np.nanpercentile(values, 99, axis=1, keepdims=True)
    fallback = np.nanmax(np.abs(values), axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, fallback)
    scale = np.where(scale > 0, scale, 1.0)
    return values / scale


def resample_rows(x: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(grid, x, row) for row in values])


def d_to_two_theta(d_spacing: np.ndarray | float, wavelength: float = WAVELENGTH_A) -> np.ndarray:
    d = np.asarray(d_spacing, dtype=float)
    ratio = wavelength / (2.0 * d)
    out = np.full(d.shape, np.nan, dtype=float)
    valid = (d > 0) & (np.abs(ratio) < 1.0)
    out[valid] = np.degrees(2.0 * np.arcsin(ratio[valid]))
    return out


def two_theta_to_d(two_theta: np.ndarray | float, wavelength: float = WAVELENGTH_A) -> np.ndarray:
    angle = np.radians(np.asarray(two_theta, dtype=float) / 2.0)
    out = np.full(angle.shape, np.nan, dtype=float)
    valid = np.abs(np.sin(angle)) > 1e-12
    out[valid] = wavelength / (2.0 * np.sin(angle[valid]))
    return out


def auc_probability(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return np.nan
    combined = np.concatenate([positive, negative])
    ranks = rankdata(combined, method="average")
    rank_sum = float(np.sum(ranks[: len(positive)]))
    return (rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative))


def linear_summary(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3 or np.nanstd(x[keep]) <= 0:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    predicted = slope * x[keep] + intercept
    ss_res = float(np.sum((y[keep] - predicted) ** 2))
    ss_tot = float(np.sum((y[keep] - np.mean(y[keep])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r = float(np.corrcoef(x[keep], y[keep])[0, 1])
    return float(slope), r, r2


def lower_triangle_values(matrix: np.ndarray) -> np.ndarray:
    rows, cols = np.tril_indices(matrix.shape[-1], k=-1)
    return matrix[..., rows, cols]


def nanmedian(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def write_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[format_value(value) for value in row]])


def plot_heatmap(
    path: Path,
    labels: list[str],
    matrix: np.ndarray,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "viridis",
    colorbar_label: str = "similarity",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shown = np.asarray(matrix, dtype=float).copy()
    shown[np.triu_indices_from(shown, k=0)] = np.nan
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")
    size = min(12.0, max(6.5, 0.38 * len(labels) + 2.0))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def scan_frame_rows(frames: list[Frame], scans: list[str]) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = {scan: [] for scan in scans}
    for index, frame in enumerate(frames):
        rows[frame.scan].append(index)
    for scan in rows:
        rows[scan].sort(key=lambda index: frames[index].pressure)
    return rows


def measure_tracks(
    grid: np.ndarray,
    normalized: np.ndarray,
    frames: list[Frame],
    tracks: list[Track],
    search_half_width: float,
    roi_half_width: float,
    snr_threshold: float,
) -> dict[str, np.ndarray]:
    smoothed = savgol_filter(normalized, 7, polyorder=2, axis=1)
    shape = (len(frames), len(tracks))
    expected_d = np.full(shape, np.nan, dtype=float)
    expected_center = np.full(shape, np.nan, dtype=float)
    center = np.full(shape, np.nan, dtype=float)
    area = np.full(shape, np.nan, dtype=float)
    snr = np.full(shape, np.nan, dtype=float)
    peak_height = np.full(shape, np.nan, dtype=float)
    in_scope = np.zeros(shape, dtype=bool)
    present = np.zeros(shape, dtype=bool)
    band_lower_values = np.full(len(tracks), np.nan, dtype=float)
    band_upper_values = np.full(len(tracks), np.nan, dtype=float)
    grid_step = float(np.median(np.diff(grid)))

    for track_index, track in enumerate(tracks):
        band_centers = d_to_two_theta(np.asarray([track.d_min, track.d_max]))
        band_lower = float(np.nanmin(band_centers) - search_half_width)
        band_upper = float(np.nanmax(band_centers) + search_half_width)
        band_lower_values[track_index] = band_lower
        band_upper_values[track_index] = band_upper
        search = np.flatnonzero((grid >= band_lower) & (grid <= band_upper))
        for frame_index, frame in enumerate(frames):
            if frame.pressure < track.p_min - 1e-9 or frame.pressure > track.p_max + 1e-9:
                continue
            predicted_d = track.d0 + track.dd_dp * frame.pressure
            predicted_center = float(d_to_two_theta(predicted_d))
            if not np.isfinite(predicted_center):
                continue
            in_scope[frame_index, track_index] = True
            expected_d[frame_index, track_index] = predicted_d
            expected_center[frame_index, track_index] = predicted_center
            if len(search) < 3:
                continue
            local = smoothed[frame_index, search]
            local_index = int(np.nanargmax(local))
            bin_index = int(search[local_index])
            if local_index == 0 or local_index == len(search) - 1 or bin_index <= 0 or bin_index >= len(grid) - 1:
                continue
            left_y, mid_y, right_y = smoothed[frame_index, bin_index - 1 : bin_index + 2]
            denom = left_y - 2.0 * mid_y + right_y
            offset = 0.0 if abs(float(denom)) < 1e-12 else 0.5 * float(left_y - right_y) / float(denom)
            offset = float(np.clip(offset, -0.75, 0.75))
            measured_center = float(grid[bin_index] + offset * grid_step)
            left_side = (
                (grid >= measured_center - roi_half_width - 0.03 - 0.08)
                & (grid < measured_center - roi_half_width - 0.03)
            )
            right_side = (
                (grid > measured_center + roi_half_width + 0.03)
                & (grid <= measured_center + roi_half_width + 0.03 + 0.08)
            )
            side = left_side | right_side
            roi = np.abs(grid - measured_center) <= roi_half_width
            if np.count_nonzero(roi) < 3:
                continue
            side_values = normalized[frame_index, side]
            if len(side_values):
                background = float(np.nanmedian(side_values))
                mad = float(np.nanmedian(np.abs(side_values - background)))
                noise = max(1.4826 * mad, 1e-8)
            else:
                background = float(np.nanpercentile(normalized[frame_index, roi], 10))
                noise = max(float(np.nanstd(normalized[frame_index, roi])), 1e-8)
            height = float(smoothed[frame_index, bin_index] - background)
            corrected = np.clip(normalized[frame_index, roi] - background, 0.0, None)
            measured_area = float(np.trapezoid(corrected, grid[roi]))
            measured_snr = height / noise
            center[frame_index, track_index] = measured_center
            area[frame_index, track_index] = measured_area
            snr[frame_index, track_index] = measured_snr
            peak_height[frame_index, track_index] = height
            present[frame_index, track_index] = bool(
                measured_snr >= snr_threshold
                and height > 0
            )

    return {
        "expected_d": expected_d,
        "expected_center": expected_center,
        "center": center,
        "area": area,
        "snr": snr,
        "peak_height": peak_height,
        "in_scope": in_scope,
        "present": present,
        "band_lower": band_lower_values,
        "band_upper": band_upper_values,
    }


def per_peak_similarity(values: np.ndarray, present: np.ndarray, location: bool, tolerance: float) -> np.ndarray:
    n = len(values)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(i):
            if present[i] and present[j] and np.isfinite(values[i]) and np.isfinite(values[j]):
                if location:
                    score = 1.0 - abs(float(values[i]) - float(values[j])) / max(tolerance, 1e-12)
                else:
                    denom = max(float(values[i]), float(values[j]), 1e-12)
                    score = 1.0 - abs(float(values[i]) - float(values[j])) / denom
                matrix[i, j] = float(np.clip(score, 0.0, 1.0))
            elif present[i] != present[j]:
                matrix[i, j] = 0.0
    return matrix


def classify_near_far(
    matrices: np.ndarray,
    pressures: list[float],
    near_gap: float = 1.5,
    far_gap: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    near: list[float] = []
    far: list[float] = []
    for i, p_i in enumerate(pressures):
        for j, p_j in enumerate(pressures[:i]):
            values = matrices[..., i, j].reshape(-1)
            values = values[np.isfinite(values)]
            gap = abs(p_i - p_j)
            if gap <= near_gap:
                near.extend(values.tolist())
            elif gap >= far_gap:
                far.extend(values.tolist())
    return np.asarray(near, dtype=float), np.asarray(far, dtype=float)


def analyze_per_peak(
    out_dir: Path,
    channel: str,
    frames: list[Frame],
    scans: list[str],
    pressures: list[float],
    tracks: list[Track],
    features: dict[str, np.ndarray],
    position_tolerance: float,
    make_plots: bool,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_rows = scan_frame_rows(frames, scans)
    n_tracks, n_scans, n_pressures = len(tracks), len(scans), len(pressures)
    area_matrices = np.full((n_tracks, n_scans, n_pressures, n_pressures), np.nan, dtype=np.float32)
    location_matrices = np.full_like(area_matrices, np.nan)

    for scan_index, scan in enumerate(scans):
        rows = scan_rows[scan]
        pressure_indices = [frames[row].pressure_index for row in rows]
        for track_index in range(n_tracks):
            area_matrix = per_peak_similarity(
                features["area"][rows, track_index],
                features["present"][rows, track_index],
                location=False,
                tolerance=position_tolerance,
            )
            location_matrix = per_peak_similarity(
                features["center"][rows, track_index],
                features["present"][rows, track_index],
                location=True,
                tolerance=position_tolerance,
            )
            for local_i, p_i in enumerate(pressure_indices):
                for local_j, p_j in enumerate(pressure_indices[:local_i]):
                    area_matrices[track_index, scan_index, p_i, p_j] = area_matrix[local_i, local_j]
                    location_matrices[track_index, scan_index, p_i, p_j] = location_matrix[local_i, local_j]

    aggregate_area = nanmedian(area_matrices, axis=1)
    aggregate_location = nanmedian(location_matrices, axis=1)
    np.savez_compressed(
        out_dir / "per_peak_matrices.npz",
        pressure_gpa=np.asarray(pressures),
        scan_names=np.asarray(scans),
        track_ids=np.asarray([track.track for track in tracks]),
        area_by_scan=area_matrices,
        location_by_scan=location_matrices,
        area_aggregate=aggregate_area,
        location_aggregate=aggregate_location,
    )

    frame_lookup = {frame.frame: index for index, frame in enumerate(frames)}
    long_fields = [
        "channel", "track", "frame", "scan", "pressure_GPa", "in_scope", "present",
        "expected_d_A", "expected_two_theta_deg", "measured_two_theta_deg", "delta_two_theta_deg",
        "measured_d_A", "roi_area_normalized", "peak_height_normalized", "snr",
    ]
    with (out_dir / "track_feature_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        for frame_index, frame in enumerate(frames):
            for track_index, track in enumerate(tracks):
                measured_center = features["center"][frame_index, track_index]
                expected_center = features["expected_center"][frame_index, track_index]
                writer.writerow({
                    "channel": channel,
                    "track": track.track,
                    "frame": frame.frame,
                    "scan": frame.scan,
                    "pressure_GPa": frame.pressure,
                    "in_scope": int(features["in_scope"][frame_index, track_index]),
                    "present": int(features["present"][frame_index, track_index]),
                    "expected_d_A": format_value(features["expected_d"][frame_index, track_index]),
                    "expected_two_theta_deg": format_value(expected_center),
                    "measured_two_theta_deg": format_value(measured_center),
                    "delta_two_theta_deg": format_value(measured_center - expected_center),
                    "measured_d_A": format_value(float(two_theta_to_d(measured_center))),
                    "roi_area_normalized": format_value(features["area"][frame_index, track_index]),
                    "peak_height_normalized": format_value(features["peak_height"][frame_index, track_index]),
                    "snr": format_value(features["snr"][frame_index, track_index]),
                })

    pressure_array = np.asarray(pressures, dtype=float)
    pressure_labels = [f"{value:g}" for value in pressures]
    summary_rows: list[dict[str, object]] = []
    for track_index, track in enumerate(tracks):
        present = features["present"][:, track_index]
        in_scope = features["in_scope"][:, track_index]
        measured_d = two_theta_to_d(features["center"][:, track_index])
        median_d = np.full(n_pressures, np.nan)
        median_center = np.full(n_pressures, np.nan)
        for pressure_index in range(n_pressures):
            keep = present & np.array([frame.pressure_index == pressure_index for frame in frames])
            if np.any(keep):
                median_d[pressure_index] = float(np.nanmedian(measured_d[keep]))
                median_center[pressure_index] = float(np.nanmedian(features["center"][keep, track_index]))
        slope, slope_r, slope_r2 = linear_summary(pressure_array, median_d)
        location_error = np.abs(
            features["center"][:, track_index] - features["expected_center"][:, track_index]
        )
        best_row = frame_lookup.get(track.best_frame)
        best_present = bool(features["present"][best_row, track_index]) if best_row is not None else False
        best_snr = float(features["snr"][best_row, track_index]) if best_row is not None else np.nan
        area_near, area_far = classify_near_far(area_matrices[track_index], pressures)
        loc_near, loc_far = classify_near_far(location_matrices[track_index], pressures)
        summary_rows.append({
            "channel": channel,
            "track": track.track,
            "hkl": track.hkl,
            "teammate_n_points": track.n_points,
            "teammate_n_frames": track.n_frames,
            "teammate_reliable": int(track.n_points >= 5),
            "p_min_GPa": track.p_min,
            "p_max_GPa": track.p_max,
            "d0_A": track.d0,
            "blind_search_band_min_deg": float(features["band_lower"][track_index]),
            "blind_search_band_max_deg": float(features["band_upper"][track_index]),
            "teammate_dd_dp_A_per_GPa": track.dd_dp,
            "in_scope_frames": int(np.count_nonzero(in_scope)),
            "present_frames": int(np.count_nonzero(present)),
            "presence_fraction": float(np.count_nonzero(present) / max(np.count_nonzero(in_scope), 1)),
            "best_frame_available": int(best_row is not None),
            "best_frame_present": int(best_present),
            "best_frame_snr": best_snr,
            "median_snr_present": float(np.nanmedian(features["snr"][present, track_index])) if np.any(present) else np.nan,
            "median_location_mae_deg": float(np.nanmedian(location_error[present])) if np.any(present) else np.nan,
            "extracted_dd_dp_A_per_GPa": slope,
            "extracted_slope_r": slope_r,
            "extracted_slope_r2": slope_r2,
            "slope_sign_match": int(np.isfinite(slope) and np.sign(slope) == np.sign(track.dd_dp)),
            "area_near_median": float(np.nanmedian(area_near)) if len(area_near) else np.nan,
            "area_far_median": float(np.nanmedian(area_far)) if len(area_far) else np.nan,
            "area_near_vs_far_auc": auc_probability(area_near, area_far),
            "location_near_median": float(np.nanmedian(loc_near)) if len(loc_near) else np.nan,
            "location_far_median": float(np.nanmedian(loc_far)) if len(loc_far) else np.nan,
            "location_near_vs_far_auc": auc_probability(loc_near, loc_far),
        })

        stem = f"track_{track.track:02d}_{safe_name(track.hkl)}"
        write_matrix_csv(out_dir / "area_matrices" / f"{stem}.csv", pressure_labels, aggregate_area[track_index])
        write_matrix_csv(out_dir / "location_matrices" / f"{stem}.csv", pressure_labels, aggregate_location[track_index])
        if make_plots:
            plot_heatmap(
                out_dir / "area_heatmaps" / f"{stem}.png",
                pressure_labels,
                aggregate_area[track_index],
                f"{channel}: track {track.track} ({track.hkl}) area",
            )
            plot_heatmap(
                out_dir / "location_heatmaps" / f"{stem}.png",
                pressure_labels,
                aggregate_location[track_index],
                f"{channel}: track {track.track} ({track.hkl}) location",
            )

    fields = list(summary_rows[0])
    with (out_dir / "track_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    area_near, area_far = classify_near_far(area_matrices, pressures)
    loc_near, loc_far = classify_near_far(location_matrices, pressures)
    reliable = [row for row in summary_rows if row["teammate_reliable"]]
    sign_rows = [row for row in reliable if np.isfinite(float(row["extracted_dd_dp_A_per_GPa"]))]
    strong_sign_rows = [
        row
        for row in sign_rows
        if np.isfinite(float(row["extracted_slope_r2"]))
        and float(row["extracted_slope_r2"]) >= 0.2
    ]
    best_available = [row for row in summary_rows if row["best_frame_available"]]
    return {
        "tracks": len(tracks),
        "reliable_tracks": len(reliable),
        "best_frame_detection_fraction": (
            sum(int(row["best_frame_present"]) for row in best_available) / max(len(best_available), 1)
            if best_available
            else np.nan
        ),
        "best_frame_detection_count": sum(int(row["best_frame_present"]) for row in best_available),
        "best_frame_available_count": len(best_available),
        "reliable_slope_sign_matches": sum(int(row["slope_sign_match"]) for row in sign_rows),
        "reliable_slope_sign_evaluable": len(sign_rows),
        "strong_slope_sign_matches": sum(int(row["slope_sign_match"]) for row in strong_sign_rows),
        "strong_slope_sign_evaluable": len(strong_sign_rows),
        "strong_slope_r2_min": 0.2,
        "area_near_median": float(np.nanmedian(area_near)) if len(area_near) else np.nan,
        "area_far_median": float(np.nanmedian(area_far)) if len(area_far) else np.nan,
        "area_near_vs_far_auc": auc_probability(area_near, area_far),
        "location_near_median": float(np.nanmedian(loc_near)) if len(loc_near) else np.nan,
        "location_far_median": float(np.nanmedian(loc_far)) if len(loc_far) else np.nan,
        "location_near_vs_far_auc": auc_probability(loc_near, loc_far),
        "summary_rows": summary_rows,
    }


def standardized_signals(values: np.ndarray, smooth_window: int, baseline_window: int) -> tuple[np.ndarray, np.ndarray]:
    n = values.shape[1]
    smooth_window = min(smooth_window, n if n % 2 else n - 1)
    baseline_window = min(baseline_window, n if n % 2 else n - 1)
    smooth_window = max(5, smooth_window | 1)
    baseline_window = max(smooth_window + 2, baseline_window | 1)
    baseline_window = min(baseline_window, n if n % 2 else n - 1)
    smooth = savgol_filter(values, smooth_window, polyorder=2, axis=1)
    baseline = savgol_filter(values, baseline_window, polyorder=2, axis=1)
    signal = smooth - baseline
    signal -= np.nanmean(signal, axis=1, keepdims=True)
    std = np.nanstd(signal, axis=1, keepdims=True)
    valid = std[:, 0] >= 1e-6
    std = np.where(std > 0, std, 1.0)
    signal /= std
    signal[~valid] = np.nan
    return signal.astype(np.float32), valid


def autocorrelation_fingerprints(signals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = signals.shape[1]
    clean = np.nan_to_num(signals, nan=0.0)
    transformed = np.fft.rfft(clean, n=2 * n, axis=1)
    corr = np.fft.irfft(transformed * np.conjugate(transformed), n=2 * n, axis=1)[:, :n]
    zero = corr[:, :1]
    zero = np.where(zero > 0, zero, 1.0)
    fingerprint = corr[:, 1:] / zero
    fingerprint -= np.nanmean(fingerprint, axis=1, keepdims=True)
    std = np.nanstd(fingerprint, axis=1, keepdims=True)
    fp_valid = valid & (std[:, 0] >= 1e-12)
    std = np.where(std > 0, std, 1.0)
    fingerprint /= std
    fingerprint[~fp_valid] = np.nan
    return fingerprint.astype(np.float32)


def build_window_fingerprints(
    grid: np.ndarray,
    normalized: np.ndarray,
    starts: np.ndarray,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    fingerprints: list[np.ndarray] = []
    signals: list[np.ndarray] = []
    for start in starts:
        keep = (grid >= start) & (grid < start + width)
        signal, valid = standardized_signals(normalized[:, keep], smooth_window=9, baseline_window=101)
        signals.append(signal)
        fingerprints.append(autocorrelation_fingerprints(signal, valid))
    return np.stack(signals, axis=1), np.stack(fingerprints, axis=1)


def row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_valid = np.all(np.isfinite(left), axis=1)
    right_valid = np.all(np.isfinite(right), axis=1)
    clean_left = np.nan_to_num(left, nan=0.0)
    clean_right = np.nan_to_num(right, nan=0.0)
    matrix = clean_left @ clean_right.T / max(left.shape[1], 1)
    matrix[~left_valid, :] = np.nan
    matrix[:, ~right_valid] = np.nan
    return np.clip(matrix, -1.0, 1.0)


def analyze_windows(
    out_dir: Path,
    channel: str,
    frames: list[Frame],
    scans: list[str],
    pressures: list[float],
    starts: np.ndarray,
    fingerprints: np.ndarray,
    window_width: float,
    shift_tolerance: float,
    window_step: float,
    make_plots: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    across_dir = out_dir / "across_frames"
    within_dir = out_dir / "within_frame"
    across_dir.mkdir(parents=True, exist_ok=True)
    within_dir.mkdir(parents=True, exist_ok=True)
    scan_rows = scan_frame_rows(frames, scans)
    n_scans, n_windows, n_pressures = len(scans), len(starts), len(pressures)
    across = np.full((n_scans, n_windows, n_pressures, n_pressures), np.nan, dtype=np.float32)
    neighbor = int(round(shift_tolerance / window_step))

    for scan_index, scan in enumerate(scans):
        rows = scan_rows[scan]
        pressure_indices = [frames[row].pressure_index for row in rows]
        for window_index in range(n_windows):
            best = np.full((len(rows), len(rows)), np.nan, dtype=float)
            for delta in range(-neighbor, neighbor + 1):
                shifted = window_index + delta
                if shifted < 0 or shifted >= n_windows:
                    continue
                candidate = row_correlation(fingerprints[rows, window_index], fingerprints[rows, shifted])
                best = np.fmax(best, candidate)
            for local_i, p_i in enumerate(pressure_indices):
                for local_j, p_j in enumerate(pressure_indices[:local_i]):
                    across[scan_index, window_index, p_i, p_j] = best[local_i, local_j]

    aggregate = nanmedian(across, axis=0)
    pressure_labels = [f"{value:g}" for value in pressures]
    summary_rows: list[dict[str, object]] = []
    all_diffs: list[float] = []
    all_scores: list[float] = []
    for window_index, start in enumerate(starts):
        near, far = classify_near_far(across[:, window_index], pressures)
        diffs: list[float] = []
        scores: list[float] = []
        for i, p_i in enumerate(pressures):
            for j, p_j in enumerate(pressures[:i]):
                values = across[:, window_index, i, j]
                values = values[np.isfinite(values)]
                diffs.extend([abs(p_i - p_j)] * len(values))
                scores.extend(values.tolist())
        slope, r, r2 = linear_summary(np.asarray(diffs), np.asarray(scores))
        all_diffs.extend(diffs)
        all_scores.extend(scores)
        summary_rows.append({
            "channel": channel,
            "window_index": window_index,
            "start_deg": float(start),
            "end_deg": float(start + window_width),
            "median_similarity": float(np.nanmedian(scores)) if scores else np.nan,
            "near_median": float(np.nanmedian(near)) if len(near) else np.nan,
            "far_median": float(np.nanmedian(far)) if len(far) else np.nan,
            "near_vs_far_auc": auc_probability(near, far),
            "score_vs_pressure_gap_slope": slope,
            "score_vs_pressure_gap_r": r,
            "score_vs_pressure_gap_r2": r2,
        })
        stem = f"window_{start:.1f}_{start + window_width:.1f}"
        write_matrix_csv(across_dir / "aggregate_matrices" / f"{stem}.csv", pressure_labels, aggregate[window_index])
        if make_plots:
            plot_heatmap(
                across_dir / "aggregate_heatmaps" / f"{stem}.png",
                pressure_labels,
                aggregate[window_index],
                f"{channel}: {start:.1f}-{start + window_width:.1f} deg across frames",
                vmin=-1.0,
                vmax=1.0,
                cmap_name="coolwarm",
                colorbar_label="ACF Pearson similarity",
            )

    with (across_dir / "window_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    np.savez_compressed(
        across_dir / "across_frame_matrices.npz",
        pressure_gpa=np.asarray(pressures),
        scan_names=np.asarray(scans),
        window_starts_deg=starts,
        matrices_by_scan=across,
        aggregate=aggregate,
    )

    within = np.full((len(frames), n_windows, n_windows), np.nan, dtype=np.float32)
    for frame_index in range(len(frames)):
        matrix = row_correlation(fingerprints[frame_index], fingerprints[frame_index])
        matrix[np.triu_indices_from(matrix, k=0)] = np.nan
        within[frame_index] = matrix
    within_aggregate = nanmedian(within, axis=0)
    within_by_pressure = np.full((n_pressures, n_windows, n_windows), np.nan, dtype=np.float32)
    for pressure_index in range(n_pressures):
        rows = [index for index, frame in enumerate(frames) if frame.pressure_index == pressure_index]
        within_by_pressure[pressure_index] = nanmedian(within[rows], axis=0)
    nonoverlap_indices = np.asarray(
        [
            index
            for index, start in enumerate(starts)
            if abs((start - 3.0) % window_width) < 1e-8
        ],
        dtype=int,
    )
    nonoverlap = within[:, nonoverlap_indices][:, :, nonoverlap_indices]
    within_values = lower_triangle_values(within).reshape(-1)
    nonoverlap_values = lower_triangle_values(nonoverlap).reshape(-1)
    within_values = within_values[np.isfinite(within_values)]
    nonoverlap_values = nonoverlap_values[np.isfinite(nonoverlap_values)]
    window_labels = [f"{start:.0f}-{start + window_width:.0f}" for start in starts]
    nonoverlap_labels = [window_labels[index] for index in nonoverlap_indices]
    write_matrix_csv(within_dir / "aggregate_matrix.csv", window_labels, within_aggregate)
    write_matrix_csv(
        within_dir / "aggregate_nonoverlap_matrix.csv",
        nonoverlap_labels,
        within_aggregate[np.ix_(nonoverlap_indices, nonoverlap_indices)],
    )
    if make_plots:
        plot_heatmap(
            within_dir / "aggregate_heatmap.png",
            window_labels,
            within_aggregate,
            f"{channel}: window-to-window within frame",
            vmin=-1.0,
            vmax=1.0,
            cmap_name="coolwarm",
            colorbar_label="ACF Pearson similarity",
        )
        plot_heatmap(
            within_dir / "aggregate_nonoverlap_heatmap.png",
            nonoverlap_labels,
            within_aggregate[np.ix_(nonoverlap_indices, nonoverlap_indices)],
            f"{channel}: non-overlap within-frame control",
            vmin=-1.0,
            vmax=1.0,
            cmap_name="coolwarm",
            colorbar_label="ACF Pearson similarity",
        )
        for pressure_index, pressure in enumerate(pressures):
            plot_heatmap(
                within_dir / "pressure_heatmaps" / f"{pressure:g}GPa.png",
                window_labels,
                within_by_pressure[pressure_index],
                f"{channel}: within-frame median at {pressure:g} GPa",
                vmin=-1.0,
                vmax=1.0,
                cmap_name="coolwarm",
                colorbar_label="ACF Pearson similarity",
            )
    pair_rows: list[dict[str, object]] = []
    overlapping_values: list[float] = []
    for i, left in enumerate(starts):
        for j, right in enumerate(starts[:i]):
            values = within[:, i, j]
            overlap = max(0.0, window_width - abs(float(left - right)))
            if overlap > 0:
                overlapping_values.extend(values[np.isfinite(values)].tolist())
            pair_rows.append({
                "channel": channel,
                "window_a": f"{left:.1f}-{left + window_width:.1f}",
                "window_b": f"{right:.1f}-{right + window_width:.1f}",
                "overlap_deg": overlap,
                "median_similarity": float(np.nanmedian(values)),
                "mean_similarity": float(np.nanmean(values)),
                "finite_frames": int(np.count_nonzero(np.isfinite(values))),
            })
    with (within_dir / "window_pair_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    np.savez_compressed(
        within_dir / "within_frame_matrices.npz",
        frame_indices=np.asarray([frame.frame for frame in frames]),
        pressure_gpa=np.asarray([frame.pressure for frame in frames]),
        window_starts_deg=starts,
        matrices=within,
        aggregate=within_aggregate,
        aggregate_by_pressure=within_by_pressure,
        nonoverlap_indices=nonoverlap_indices,
    )

    all_near, all_far = classify_near_far(across, pressures)
    slope, r, r2 = linear_summary(np.asarray(all_diffs), np.asarray(all_scores))
    across_metrics = {
        "windows": n_windows,
        "near_median": float(np.nanmedian(all_near)) if len(all_near) else np.nan,
        "far_median": float(np.nanmedian(all_far)) if len(all_far) else np.nan,
        "near_vs_far_auc": auc_probability(all_near, all_far),
        "score_vs_pressure_gap_slope": slope,
        "score_vs_pressure_gap_r": r,
        "score_vs_pressure_gap_r2": r2,
        "summary_rows": summary_rows,
    }
    within_metrics = {
        "frames": len(frames),
        "windows": n_windows,
        "nonoverlap_windows": int(len(nonoverlap_indices)),
        "all_pair_median": float(np.nanmedian(within_values)) if len(within_values) else np.nan,
        "overlapping_pair_median": (
            float(np.nanmedian(overlapping_values)) if len(overlapping_values) else np.nan
        ),
        "nonoverlap_pair_median": float(np.nanmedian(nonoverlap_values)) if len(nonoverlap_values) else np.nan,
        "overlap_inflation": (
            float(np.nanmedian(overlapping_values) - np.nanmedian(nonoverlap_values))
            if len(overlapping_values) and len(nonoverlap_values)
            else np.nan
        ),
    }
    return across_metrics, within_metrics


def analyze_whole_pattern(
    out_dir: Path,
    channel: str,
    frames: list[Frame],
    scans: list[str],
    pressures: list[float],
    normalized: np.ndarray,
    make_plots: bool,
) -> dict[str, float | int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    signals, valid = standardized_signals(normalized, smooth_window=9, baseline_window=101)
    scan_rows = scan_frame_rows(frames, scans)
    matrices = np.full((len(scans), len(pressures), len(pressures)), np.nan, dtype=np.float32)
    pair_rows: list[dict[str, object]] = []
    for scan_index, scan in enumerate(scans):
        rows = scan_rows[scan]
        matrix = row_correlation(signals[rows], signals[rows])
        for local_i, row_i in enumerate(rows):
            for local_j, row_j in enumerate(rows[:local_i]):
                p_i = frames[row_i].pressure_index
                p_j = frames[row_j].pressure_index
                score = float(matrix[local_i, local_j])
                matrices[scan_index, p_i, p_j] = score
                pair_rows.append({
                    "channel": channel,
                    "scan": scan,
                    "frame_a": frames[row_i].frame,
                    "frame_b": frames[row_j].frame,
                    "pressure_a_GPa": frames[row_i].pressure,
                    "pressure_b_GPa": frames[row_j].pressure,
                    "pressure_gap_GPa": abs(frames[row_i].pressure - frames[row_j].pressure),
                    "correlation": score,
                })
    with (out_dir / "whole_pattern_pair_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    aggregate = nanmedian(matrices, axis=0)
    labels = [f"{pressure:g}" for pressure in pressures]
    write_matrix_csv(out_dir / "aggregate_matrix.csv", labels, aggregate)
    if make_plots:
        plot_heatmap(
            out_dir / "aggregate_heatmap.png",
            labels,
            aggregate,
            f"{channel}: whole-pattern correlation",
            vmin=-1.0,
            vmax=1.0,
            cmap_name="coolwarm",
            colorbar_label="Pearson correlation",
        )
        x = np.asarray([float(row["pressure_gap_GPa"]) for row in pair_rows])
        y = np.asarray([float(row["correlation"]) for row in pair_rows])
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.scatter(x, y, s=7, alpha=0.18, color="#2A6F97", edgecolors="none")
        slope, _, _ = linear_summary(x, y)
        intercept = float(np.nanmean(y) - slope * np.nanmean(x)) if np.isfinite(slope) else np.nan
        if np.isfinite(slope):
            line_x = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            ax.plot(line_x, slope * line_x + intercept, color="#B23A48", linewidth=2)
        ax.set_xlabel("Pressure gap (GPa)")
        ax.set_ylabel("Whole-pattern Pearson correlation")
        ax.set_title(f"{channel}: correlation vs pressure gap")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "correlation_vs_pressure_gap.png", dpi=200)
        plt.close(fig)
    np.savez_compressed(
        out_dir / "whole_pattern_matrices.npz",
        pressure_gpa=np.asarray(pressures),
        scan_names=np.asarray(scans),
        matrices_by_scan=matrices,
        aggregate=aggregate,
        valid_frames=valid,
    )
    gaps = np.asarray([float(row["pressure_gap_GPa"]) for row in pair_rows])
    scores = np.asarray([float(row["correlation"]) for row in pair_rows])
    slope, r, r2 = linear_summary(gaps, scores)
    near = scores[gaps <= 1.5]
    far = scores[gaps >= 15.0]
    return {
        "pairs": len(pair_rows),
        "mean_correlation": float(np.nanmean(scores)),
        "near_median": float(np.nanmedian(near)),
        "far_median": float(np.nanmedian(far)),
        "near_vs_far_auc": auc_probability(near, far),
        "corr_vs_pressure_gap_slope": slope,
        "corr_vs_pressure_gap_r": r,
        "corr_vs_pressure_gap_r2": r2,
    }


def compare_previous_tracks(previous_root: Path, tracks: list[Track]) -> list[dict[str, object]]:
    previous_path = previous_root / "C_dspacing_eos" / "eos_moving_peaks.csv"
    if not previous_path.is_file():
        return []
    with previous_path.open(newline="", encoding="utf-8") as handle:
        previous = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for track in tracks:
        best = min(previous, key=lambda row: abs(float(row["d0_A"]) - track.d0))
        previous_d = float(best["d0_A"])
        delta_pct = 100.0 * (track.d0 - previous_d) / previous_d
        rows.append({
            "track": track.track,
            "hkl": track.hkl,
            "new_d0_A": track.d0,
            "new_dd_dp_A_per_GPa": track.dd_dp,
            "teammate_reliable": int(track.n_points >= 5),
            "previous_cell": best["cell"],
            "previous_traj_id": int(best["traj_id"]),
            "previous_d0_A": previous_d,
            "previous_dd_dp_A_per_GPa": float(best["dd_dP"]),
            "d0_delta_pct": delta_pct,
            "d_match_within_2pct": int(abs(delta_pct) <= 2.0),
            "compression_sign_match": int(np.sign(track.dd_dp) == np.sign(float(best["dd_dP"]))),
        })
    return rows


def read_previous_correlation(previous_root: Path) -> list[dict[str, str]]:
    path = previous_root / "D_percell_correlation" / "percell_correlation_summary.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items() if key != "summary_rows"}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def verdict(auc: float | None, higher_is_good: bool = True) -> str:
    if auc is None or not np.isfinite(float(auc)):
        return "not_evaluable"
    value = float(auc)
    if not higher_is_good:
        value = 1.0 - value
    if value >= 0.70:
        return "strong_match"
    if value >= 0.58:
        return "partial_match"
    return "weak_or_no_match"


def write_comparison(
    out_dir: Path,
    metrics: dict[str, dict[str, object]],
    previous_corr: list[dict[str, str]],
    previous_track_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    previous_r = "; ".join(
        f"{row['cell']} r={float(row['corr_vs_dP_r']):.3f}" for row in previous_corr
    ) or "not available"
    rows: list[dict[str, object]] = []
    for family, key in [
        ("Per-peak area", "area_near_vs_far_auc"),
        ("Per-peak location", "location_near_vs_far_auc"),
    ]:
        rows.append({
            "test": family,
            "spots_result": metrics["spots"]["per_peak"].get(key),
            "fit_result": metrics.get("fit", {}).get("per_peak", {}).get(key),
            "previous_result": "Measured-label benchmark AUC=1.000 (different truth task)",
            "verdict": verdict(metrics["spots"]["per_peak"].get(key)),
            "notes": "New AUC tests near-pressure pairs above far-pressure pairs within each scan.",
        })
    rows.append({
        "test": "Same window across frames",
        "spots_result": metrics["spots"]["across_frames"].get("near_vs_far_auc"),
        "fit_result": metrics.get("fit", {}).get("across_frames", {}).get("near_vs_far_auc"),
        "previous_result": "Measured-label benchmark AUC=0.598; supporting descriptor only",
        "verdict": verdict(metrics["spots"]["across_frames"].get("near_vs_far_auc")),
        "notes": "Uses the same 5-degree ACF windows with +/-1-degree neighboring-window shift.",
    })
    spots_slope_total = int(metrics["spots"]["per_peak"].get("strong_slope_sign_evaluable", 0))
    spots_slope_match = int(metrics["spots"]["per_peak"].get("strong_slope_sign_matches", 0))
    fit_slope_total = int(metrics.get("fit", {}).get("per_peak", {}).get("strong_slope_sign_evaluable", 0))
    fit_slope_match = int(metrics.get("fit", {}).get("per_peak", {}).get("strong_slope_sign_matches", 0))
    spots_slope_fraction = spots_slope_match / spots_slope_total if spots_slope_total else np.nan
    fit_slope_fraction = fit_slope_match / fit_slope_total if fit_slope_total else np.nan
    rows.append({
        "test": "Blind per-peak d-slope sign",
        "spots_result": format_value(spots_slope_fraction),
        "fit_result": format_value(fit_slope_fraction),
        "previous_result": "Teammate track-summary dd/dP signs",
        "verdict": (
            "supports_spots"
            if np.isfinite(spots_slope_fraction)
            and spots_slope_fraction >= 0.65
            and (not np.isfinite(fit_slope_fraction) or spots_slope_fraction > fit_slope_fraction + 0.10)
            else "not_channel_specific"
        ),
        "notes": "R2>=0.2 trends only. Peaks use a fixed d_min-d_max band; the teammate pressure slope is not used during selection.",
    })
    rows.append({
        "test": "Whole-pattern correlation decay",
        "spots_result": metrics["spots"]["whole_pattern"].get("corr_vs_pressure_gap_r"),
        "fit_result": metrics.get("fit", {}).get("whole_pattern", {}).get("corr_vs_pressure_gap_r"),
        "previous_result": previous_r,
        "verdict": (
            "qualitative_direction_only"
            if float(metrics["spots"]["whole_pattern"].get("corr_vs_pressure_gap_r", np.nan)) < 0
            else "direction_does_not_match"
        ),
        "notes": "Only the negative direction is comparable. The r magnitudes are not directly comparable because the experiments, masks/channels, wavelength, radial support, and aggregation units differ.",
    })
    rows.append({
        "test": "Within-frame window control",
        "spots_result": metrics["spots"]["within_frame"].get("nonoverlap_pair_median"),
        "fit_result": metrics.get("fit", {}).get("within_frame", {}).get("nonoverlap_pair_median"),
        "previous_result": "Previously structurally valid but scientifically unlabelled",
        "verdict": "exploratory_only",
        "notes": "Non-overlap control is reported because sliding windows share signal by construction.",
    })
    reliable_previous = [row for row in previous_track_rows if row["teammate_reliable"]]
    matched = sum(int(row["d_match_within_2pct"]) for row in reliable_previous)
    rows.append({
        "test": "d-spacing overlap with corrected previous UOTe moving peaks",
        "spots_result": f"{matched}/{len(reliable_previous)} reliable tracks within 2%",
        "fit_result": "not applicable",
        "previous_result": "Corrected v2 EOS moving-peak table",
        "verdict": "physical_overlap" if matched else "no_close_overlap",
        "notes": "This is a d-spacing comparison, not a direct frame-to-frame identity claim.",
    })
    with (out_dir / "comparison_to_previous.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_report(
    path: Path,
    metrics: dict[str, dict[str, object]],
    comparison: list[dict[str, object]],
    frames: list[Frame],
    scans: list[str],
    pressures: list[float],
    tracks: list[Track],
) -> None:
    spots = metrics["spots"]
    fit = metrics.get("fit")
    slope_matches = int(spots["per_peak"]["strong_slope_sign_matches"])
    slope_total = int(spots["per_peak"]["strong_slope_sign_evaluable"])
    all_slope_matches = int(spots["per_peak"]["reliable_slope_sign_matches"])
    all_slope_total = int(spots["per_peak"]["reliable_slope_sign_evaluable"])
    family_verdicts = {
        str(row["test"]): str(row["verdict"])
        for row in comparison
    }
    decay_matches = family_verdicts.get("Whole-pattern correlation decay") == "qualitative_direction_only"
    strong_families = sum(
        family_verdicts.get(name) == "strong_match"
        for name in ("Per-peak area", "Per-peak location", "Same window across frames")
    )
    if decay_matches and strong_families >= 2:
        headline = "The new processed-frame run broadly matches the expected correlation behavior."
    elif decay_matches:
        headline = "The new run agrees with the old result only in qualitative direction; the r magnitudes are not a direct replication."
    else:
        headline = "The new run does not reproduce the previous pressure-decay trend cleanly."
    fit_text = ""
    if fit:
        fit_text = (
            f"- Fit-channel whole-pattern r vs |dP|: {float(fit['whole_pattern']['corr_vs_pressure_gap_r']):.3f}.\n"
            f"- Fit-channel per-peak area/location AUC: {float(fit['per_peak']['area_near_vs_far_auc']):.3f} / "
            f"{float(fit['per_peak']['location_near_vs_far_auc']):.3f}.\n"
            f"- Fit-channel blind slope signs (R2>=0.2): {int(fit['per_peak']['strong_slope_sign_matches'])}/"
            f"{int(fit['per_peak']['strong_slope_sign_evaluable'])}; all finite trends "
            f"{int(fit['per_peak']['reliable_slope_sign_matches'])}/"
            f"{int(fit['per_peak']['reliable_slope_sign_evaluable'])}.\n"
        )
    content = f"""# UOTe processed-frame correlation report

## Scope

- {len(scans)} independent scan positions.
- {len(frames)} usable processed frames after cover/beamstop exclusions.
- {len(pressures)} pressure values from {min(pressures):g} to {max(pressures):g} GPa.
- {len(tracks)} teammate spot tracks used as the per-peak registry.
- Spots is the sample channel; fit is a tungsten/background-dominated control.

## Main conclusion

{headline}

The old and new r values use different experiments, masking/channel definitions, wavelength, radial support, and aggregation units. Only the negative direction can be compared qualitatively. Peak and window AUC values use near-pressure versus far-pressure pairs, so they should not be read as the same ground-truth task as the older phase-label benchmark.

## Four correlation families

1. **Per-peak area across frames**
   - Spots near/far median: {float(spots['per_peak']['area_near_median']):.3f} / {float(spots['per_peak']['area_far_median']):.3f}.
   - Spots near-vs-far AUC: {float(spots['per_peak']['area_near_vs_far_auc']):.3f}.
2. **Per-peak location across frames**
   - Spots near/far median: {float(spots['per_peak']['location_near_median']):.3f} / {float(spots['per_peak']['location_far_median']):.3f}.
   - Spots near-vs-far AUC: {float(spots['per_peak']['location_near_vs_far_auc']):.3f}.
   - Blind teammate-track slope signs (R2>=0.2): {slope_matches}/{slope_total}; all finite trends: {all_slope_matches}/{all_slope_total}.
   - Teammate best-frame peaks detected in spots: {int(spots['per_peak']['best_frame_detection_count'])}/{int(spots['per_peak']['best_frame_available_count'])} ({100.0 * float(spots['per_peak']['best_frame_detection_fraction']):.1f}%).
3. **Same window across frames**
   - Spots near/far median: {float(spots['across_frames']['near_median']):.3f} / {float(spots['across_frames']['far_median']):.3f}.
   - Spots near-vs-far AUC: {float(spots['across_frames']['near_vs_far_auc']):.3f}.
4. **Window-to-window within the same frame**
   - Spots overlapping-window median: {float(spots['within_frame']['overlapping_pair_median']):.3f}.
   - Spots non-overlap median: {float(spots['within_frame']['nonoverlap_pair_median']):.3f}.
   - This family remains exploratory because no independent truth labels relate one radial window to another inside a frame.

## Qualitative comparison to the previous UOTe result

- Spots whole-pattern correlation vs |dP|: slope {float(spots['whole_pattern']['corr_vs_pressure_gap_slope']):.5f}/GPa, r={float(spots['whole_pattern']['corr_vs_pressure_gap_r']):.3f}.
{fit_text}- The corrected previous result had negative r in both cells (Cell_29 -0.594; Cell_14 -0.404). The negative direction is compatible, but the numerical proximity of -0.585 and -0.594 is not evidence of replication.
- All {len(scans)} pressure ladders were analyzed separately before aggregation. No cross-position frame pairs were allowed into the main matrices.
- A scan-level bootstrap, pressure-label permutation, acquisition-order check, and fit-control adjustment are reported in `robustness/ROBUSTNESS_EVALUATION.md`.

## Scientific cautions

- The teammate track table is a summary, not the per-observation spot list. A fixed d_min-d_max band is used to prevent pressure-shift fragmentation without feeding the teammate slope into peak selection, but a nearby unrelated spot can still be selected in a 1D projection.
- The spots channel is intentionally azimuthally sparse and its absolute heights are diluted; location results are more trustworthy than absolute area.
- Sliding 5-degree windows overlap heavily. Use the non-overlap within-frame control before interpreting a high window-to-window value.
- The fit channel is a control dominated by the tungsten pressure marker. A strong fit-channel result is not automatically evidence for UOTe.
- The old corrected-v2 pipeline did use detector gap/bad-pixel masking, but it did not use the handoff's spots/fit separation or cover-frame exclusions. The two r magnitudes are therefore not directly comparable.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    handoff_dir = args.handoff_dir.expanduser().resolve()
    tracks_csv = args.tracks_csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = [item.strip().lower() for item in args.channels.split(",") if item.strip()]
    if "spots" not in channels:
        raise SystemExit("The primary spots channel must be included.")

    frames, manifest_rows, pressures, scans = read_manifest(
        handoff_dir / "manifest.csv", args.max_scans
    )
    tracks = read_tracks(tracks_csv)
    scan_counts = defaultdict(int)
    for frame in frames:
        scan_counts[frame.scan] += 1
    grid = np.arange(args.min_two_theta, args.max_two_theta + args.grid_step / 2.0, args.grid_step)
    last_start = math.floor((args.max_two_theta - args.window_width) / args.window_step) * args.window_step
    first_start = math.ceil(args.min_two_theta / args.window_step) * args.window_step
    starts = np.arange(first_start, last_start + args.window_step / 2.0, args.window_step)
    metrics: dict[str, dict[str, object]] = {}
    channel_qc: dict[str, dict[str, object]] = {}

    for channel in channels:
        channel_out = out_dir / channel
        x_native, raw, paths = load_channel(handoff_dir, channel, frames)
        normalized_native = normalize_rows(raw)
        normalized = resample_rows(x_native, normalized_native, grid)
        features = measure_tracks(
            grid,
            normalized,
            frames,
            tracks,
            args.peak_search_half_width,
            args.roi_half_width,
            args.snr_threshold,
        )
        per_peak = analyze_per_peak(
            channel_out / "per_peak",
            channel,
            frames,
            scans,
            pressures,
            tracks,
            features,
            args.position_tolerance,
            not args.no_plots,
        )
        _, fingerprints = build_window_fingerprints(
            grid, normalized, starts, args.window_width
        )
        across, within = analyze_windows(
            channel_out,
            channel,
            frames,
            scans,
            pressures,
            starts,
            fingerprints,
            args.window_width,
            args.window_shift_tolerance,
            args.window_step,
            not args.no_plots,
        )
        whole = analyze_whole_pattern(
            channel_out / "whole_pattern",
            channel,
            frames,
            scans,
            pressures,
            normalized,
            not args.no_plots,
        )
        metrics[channel] = {
            "per_peak": per_peak,
            "across_frames": across,
            "within_frame": within,
            "whole_pattern": whole,
        }
        channel_qc[channel] = {
            "files": len(paths),
            "native_bins": int(len(x_native)),
            "native_min_two_theta": float(x_native[0]),
            "native_max_two_theta": float(x_native[-1]),
            "analysis_bins": int(len(grid)),
            "raw_min": float(np.min(raw)),
            "raw_max": float(np.max(raw)),
            "negative_fraction": float(np.mean(raw < 0)),
            "zero_fraction": float(np.mean(raw == 0)),
            "finite": bool(np.all(np.isfinite(raw))),
        }

    previous_root = args.previous_root.expanduser().resolve()
    previous_track_rows = compare_previous_tracks(previous_root, tracks)
    if previous_track_rows:
        with (out_dir / "track_dspacing_match_to_previous.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(previous_track_rows[0]))
            writer.writeheader()
            writer.writerows(previous_track_rows)
    previous_corr = read_previous_correlation(previous_root)
    comparison = write_comparison(out_dir, metrics, previous_corr, previous_track_rows)
    write_report(out_dir / "REPORT.md", metrics, comparison, frames, scans, pressures, tracks)

    qc_rows = [
        {"check": "manifest_rows_selected", "value": len(manifest_rows), "passed": 1},
        {"check": "usable_frames", "value": len(frames), "passed": int(len(frames) > 0)},
        {"check": "scan_count", "value": len(scans), "passed": int(len(scans) > 0)},
        {"check": "pressure_count", "value": len(pressures), "passed": int(len(pressures) > 1)},
        {"check": "track_count", "value": len(tracks), "passed": int(len(tracks) > 0)},
        {"check": "min_frames_per_scan", "value": min(scan_counts.values()), "passed": int(min(scan_counts.values()) >= 3)},
        {"check": "max_frames_per_scan", "value": max(scan_counts.values()), "passed": 1},
    ]
    for channel, values in channel_qc.items():
        qc_rows.extend([
            {"check": f"{channel}_file_count", "value": values["files"], "passed": int(values["files"] == len(frames))},
            {"check": f"{channel}_finite", "value": values["finite"], "passed": int(values["finite"])},
        ])
    with (out_dir / "data_qc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "value", "passed"])
        writer.writeheader()
        writer.writerows(qc_rows)

    validation = {
        "passed": all(int(row["passed"]) == 1 for row in qc_rows),
        "frames": len(frames),
        "scans": len(scans),
        "pressures": len(pressures),
        "tracks": len(tracks),
        "channels": channels,
        "checks": qc_rows,
        "matrix_families": {
            channel: {
                "per_peak_area": len(tracks),
                "per_peak_location": len(tracks),
                "same_window_across_frames": len(starts),
                "window_to_window_within_frame": len(frames),
            }
            for channel in channels
        },
    }
    (out_dir / "validation_report.json").write_text(
        json.dumps(json_ready(validation), indent=2), encoding="utf-8"
    )
    run_manifest = {
        "handoff_dir": str(handoff_dir),
        "tracks_csv": str(tracks_csv),
        "previous_root": str(previous_root),
        "out_dir": str(out_dir),
        "parameters": vars(args),
        "frames": len(frames),
        "scans": scans,
        "pressures_gpa": pressures,
        "tracks": len(tracks),
        "channel_qc": channel_qc,
        "metrics": metrics,
        "comparison": comparison,
    }
    run_manifest["parameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_manifest["parameters"].items()
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(json_ready(run_manifest), indent=2), encoding="utf-8"
    )
    print(f"Wrote UOTe handoff correlations to {out_dir}")


if __name__ == "__main__":
    main()
