#!/usr/bin/env python3
"""Identify static peak groups and build overall frame correlation maps."""

from __future__ import annotations

import argparse
import csv
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PRESSURE_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)GPa")
WINDOW_LABEL_RE = re.compile(r"(?P<frame>.+)_(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a correlation suite: identify high-coverage non-moving "
            "peak groups and build overall frame-vs-frame correlation maps."
        )
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=Path("outputs/correlation_suite_20260617"),
        help="Correlation suite folder containing 01/02/03/04 subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to SUITE/05_static_peaks_and_overall_correlation.",
    )
    parser.add_argument("--min-static-frame-fraction", type=float, default=0.65)
    parser.add_argument("--max-static-mad-deg", type=float, default=0.045)
    parser.add_argument("--max-static-abs-slope", type=float, default=0.005)
    parser.add_argument("--min-static-position-frames", type=int, default=5)
    parser.add_argument("--min-static-max-roi-area", type=float, default=5e-4)
    parser.add_argument(
        "--min-reliable-frame-fraction",
        type=float,
        default=0.24,
        help=(
            "Minimum frame coverage for a peak group to be included in reliable-peak "
            "overall maps. With 17 frames, 0.24 keeps groups detected in roughly 5+ frames."
        ),
    )
    parser.add_argument(
        "--min-reliable-max-roi-area",
        type=float,
        default=1e-4,
        help="Minimum max ROI area for a peak group to be included in reliable-peak maps.",
    )
    parser.add_argument("--roi-presence-fraction-of-max", type=float, default=0.10)
    parser.add_argument("--min-roi-presence-area", type=float, default=1e-4)
    parser.add_argument("--weight-peak-presence", type=float, default=0.35)
    parser.add_argument("--weight-peak-area", type=float, default=0.30)
    parser.add_argument("--weight-same-window-acf", type=float, default=0.20)
    parser.add_argument("--weight-all-window-acf", type=float, default=0.15)
    return parser.parse_args()


def format_value(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.6g}"


def parse_float(text: str) -> float:
    if text == "":
        return np.nan
    return float(text)


def pressure_for_label(label: str) -> float | None:
    match = PRESSURE_RE.search(label)
    if not match:
        return None
    return float(match.group("value"))


def lower_triangle(matrix: np.ndarray) -> np.ndarray:
    shown = matrix.copy()
    shown[np.triu_indices_from(shown, k=0)] = np.nan
    return shown


def write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[format_value(float(value)) for value in row]])


def plot_heatmap(
    path: Path,
    labels: list[str],
    matrix: np.ndarray,
    title: str,
    colorbar_label: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "viridis",
) -> None:
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")
    size = min(14.0, max(6.0, 0.42 * len(labels) + 2.0))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def read_feature_table(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        features = header[1:]
        labels: list[str] = []
        rows: list[list[float]] = []
        for row in reader:
            labels.append(row[0])
            rows.append([parse_float(value) for value in row[1:]])
    return labels, features, np.array(rows, dtype=float)


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        labels = header[1:]
        rows = []
        row_labels = []
        for row in reader:
            row_labels.append(row[0])
            rows.append([parse_float(value) for value in row[1:]])
    if row_labels != labels:
        raise ValueError(f"Row/column labels do not match in {path}")
    return labels, np.array(rows, dtype=float)


def symmetrize_lower_matrix(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    n = out.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i, j] = np.nan
            elif np.isnan(out[i, j]) and np.isfinite(out[j, i]):
                out[i, j] = out[j, i]
            elif np.isfinite(out[i, j]) and np.isfinite(out[j, i]):
                out[i, j] = 0.5 * (out[i, j] + out[j, i])
    return out


def cosine_similarity(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(np.nan_to_num(features), axis=1, keepdims=True)
    safe = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)
    matrix = safe @ safe.T
    np.fill_diagonal(matrix, np.nan)
    return matrix


def jaccard_similarity(binary_features: np.ndarray) -> np.ndarray:
    binary = binary_features.astype(bool)
    n = binary.shape[0]
    matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            union = np.count_nonzero(binary[i] | binary[j])
            if union == 0:
                continue
            intersection = np.count_nonzero(binary[i] & binary[j])
            matrix[i, j] = intersection / union
    return matrix


def peak_group_reliability_mask(
    path: Path,
    labels: list[str],
    group_labels: list[str],
    roi_area: np.ndarray,
    presence: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    records: list[dict[str, object]] = []
    reliable_mask = np.zeros(len(group_labels), dtype=bool)
    total_frames = len(labels)

    for col, group_label in enumerate(group_labels):
        max_area = float(np.nanmax(roi_area[:, col]))
        roi_threshold = max(args.min_roi_presence_area, args.roi_presence_fraction_of_max * max_area)
        detected_count = int(np.count_nonzero(presence[:, col] > 0))
        roi_count = int(np.count_nonzero(roi_area[:, col] >= roi_threshold))
        detected_fraction = detected_count / total_frames
        roi_fraction = roi_count / total_frames
        coverage_fraction = max(detected_fraction, roi_fraction)
        reliable = bool(
            detected_fraction >= args.min_reliable_frame_fraction
            and max_area >= args.min_reliable_max_roi_area
        )
        reliable_mask[col] = reliable
        records.append(
            {
                "peak_group": col + 1,
                "group_two_theta": float(group_label),
                "reliable": int(reliable),
                "coverage_fraction": coverage_fraction,
                "detected_frame_count": detected_count,
                "detected_fraction": detected_fraction,
                "roi_presence_count": roi_count,
                "roi_fraction": roi_fraction,
                "roi_presence_threshold": roi_threshold,
                "max_roi_area": max_area,
            }
        )

    with path.open("w", newline="") as handle:
        fieldnames = list(records[0].keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: format_value(value) if isinstance(value, float) else value for key, value in record.items()})

    return reliable_mask


def robust_slope(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return np.nan
    x = np.array([point[0] for point in points], dtype=float)
    y = np.array([point[1] for point in points], dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return np.nan
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def identify_static_peaks(args: argparse.Namespace, out_dir: Path) -> tuple[list[str], np.ndarray]:
    peak_dir = args.suite_dir / "01_per_peak_frame_correlation"
    labels, group_labels, roi_area = read_feature_table(peak_dir / "peak_roi_area_features.csv")
    _, _, presence = read_feature_table(peak_dir / "peak_presence_features.csv")
    group_centers = [float(value) for value in group_labels]
    total_frames = len(labels)

    peak_rows = list(csv.DictReader((peak_dir / "peak_table.csv").open()))
    peaks_by_group_frame: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in peak_rows:
        peaks_by_group_frame[int(row["peak_group"])][row["frame"]].append(row)

    records: list[dict[str, object]] = []
    static_mask = np.zeros(len(group_centers), dtype=bool)

    for col, center in enumerate(group_centers):
        group_index = col + 1
        detected_frames = peaks_by_group_frame.get(group_index, {})
        positions: list[float] = []
        pressure_positions: list[tuple[float, float]] = []
        for frame, rows in detected_frames.items():
            best = max(rows, key=lambda row: float(row["prominence"]))
            position = float(best["two_theta"])
            positions.append(position)
            pressure = pressure_for_label(frame)
            if pressure is not None:
                pressure_positions.append((pressure, position))

        if positions:
            median_position = float(np.median(positions))
            mad_position = float(np.median(np.abs(np.array(positions) - median_position)))
            range_position = float(np.max(positions) - np.min(positions))
        else:
            median_position = np.nan
            mad_position = np.nan
            range_position = np.nan

        slope = robust_slope(pressure_positions)
        max_area = float(np.nanmax(roi_area[:, col]))
        roi_threshold = max(args.min_roi_presence_area, args.roi_presence_fraction_of_max * max_area)
        roi_present = roi_area[:, col] >= roi_threshold
        detected_count = int(np.count_nonzero(presence[:, col] > 0))
        roi_count = int(np.count_nonzero(roi_present))
        detected_fraction = detected_count / total_frames
        roi_fraction = roi_count / total_frames
        coverage_fraction = max(detected_fraction, roi_fraction)

        has_enough_positions = len(positions) >= args.min_static_position_frames
        position_stable = has_enough_positions and mad_position <= args.max_static_mad_deg
        slope_stable = has_enough_positions and (not np.isfinite(slope) or abs(slope) <= args.max_static_abs_slope)
        enough_signal = max_area >= args.min_static_max_roi_area
        high_coverage = coverage_fraction >= args.min_static_frame_fraction
        suspected_static = bool(high_coverage and enough_signal and position_stable and slope_stable)
        static_mask[col] = suspected_static

        records.append(
            {
                "peak_group": group_index,
                "group_two_theta": center,
                "suspected_static": int(suspected_static),
                "coverage_fraction": coverage_fraction,
                "detected_frame_count": detected_count,
                "detected_fraction": detected_fraction,
                "roi_presence_count": roi_count,
                "roi_fraction": roi_fraction,
                "roi_presence_threshold": roi_threshold,
                "max_roi_area": max_area,
                "position_frames": len(positions),
                "median_detected_two_theta": median_position,
                "position_mad_deg": mad_position,
                "position_range_deg": range_position,
                "slope_deg_per_gpa": slope,
                "frames_detected": ";".join(sorted(detected_frames)),
            }
        )

    fieldnames = list(records[0].keys()) if records else []
    with (out_dir / "all_peak_static_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: format_value(value) if isinstance(value, float) else value for key, value in record.items()})

    static_records = [record for record in records if record["suspected_static"]]
    with (out_dir / "suspected_static_peak_groups.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in static_records:
            writer.writerow({key: format_value(value) if isinstance(value, float) else value for key, value in record.items()})

    plot_static_summary(out_dir / "suspected_static_peak_summary.png", records)
    plot_static_positions(out_dir / "suspected_static_peak_positions_vs_pressure.png", static_records, peaks_by_group_frame)
    return group_labels, static_mask


def plot_static_summary(path: Path, records: list[dict[str, object]]) -> None:
    x = np.array([float(record["group_two_theta"]) for record in records])
    y = np.array([float(record["coverage_fraction"]) for record in records])
    colors = np.array([float(record["position_mad_deg"]) if record["position_mad_deg"] != "" else np.nan for record in records])
    static = np.array([bool(record["suspected_static"]) for record in records])

    fig, ax = plt.subplots(figsize=(11, 5))
    scatter = ax.scatter(x, y, c=colors, cmap="magma_r", s=28, alpha=0.75, label="peak groups")
    if np.any(static):
        ax.scatter(x[static], y[static], facecolors="none", edgecolors="cyan", s=110, linewidths=1.8, label="suspected static")
    ax.set_xlabel("2theta (deg)")
    ax.set_ylabel("coverage fraction")
    ax.set_title("High-coverage non-moving peak candidates")
    fig.colorbar(scatter, ax=ax, label="position MAD (deg)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_static_positions(
    path: Path,
    static_records: list[dict[str, object]],
    peaks_by_group_frame: dict[int, dict[str, list[dict[str, str]]]],
) -> None:
    if not static_records:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No suspected static peak groups passed the current thresholds.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for record in static_records:
        group_index = int(record["peak_group"])
        points: list[tuple[float, float]] = []
        for frame, rows in peaks_by_group_frame[group_index].items():
            pressure = pressure_for_label(frame)
            if pressure is None:
                continue
            best = max(rows, key=lambda row: float(row["prominence"]))
            points.append((pressure, float(best["two_theta"])))
        if len(points) < 2:
            continue
        points = sorted(points)
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=1,
            markersize=3,
            label=f"g{group_index}: {float(record['group_two_theta']):.3f} deg",
        )
    ax.set_xlabel("pressure (GPa)")
    ax.set_ylabel("detected 2theta (deg)")
    ax.set_title("Suspected static peak positions")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def aggregate_same_window_acf(suite_dir: Path, labels: list[str]) -> np.ndarray:
    matrix_dir = suite_dir / "02_same_window_acf_across_frames" / "matrices"
    matrices = []
    for path in sorted(matrix_dir.glob("*_same_window_acf.csv")):
        matrix_labels, matrix = read_matrix(path)
        if matrix_labels != labels:
            raise ValueError(f"Frame labels differ in {path}")
        matrices.append(symmetrize_lower_matrix(matrix))
    if not matrices:
        return np.full((len(labels), len(labels)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(matrices), axis=0)


def frame_from_window_label(label: str) -> str:
    match = WINDOW_LABEL_RE.match(label)
    if not match:
        return label
    return match.group("frame")


def aggregate_all_window_acf(suite_dir: Path, frame_labels: list[str]) -> np.ndarray:
    path = suite_dir / "04_all_window_to_all_window_acf" / "all_window_to_all_window_acf_matrix.csv"
    window_labels, raw_matrix = read_matrix(path)
    matrix = symmetrize_lower_matrix(raw_matrix)
    frame_for_window = [frame_from_window_label(label) for label in window_labels]
    output = np.full((len(frame_labels), len(frame_labels)), np.nan, dtype=float)

    for i, frame_a in enumerate(frame_labels):
        idx_a = [idx for idx, label in enumerate(frame_for_window) if label == frame_a]
        for j, frame_b in enumerate(frame_labels):
            if i == j:
                continue
            idx_b = [idx for idx, label in enumerate(frame_for_window) if label == frame_b]
            values = matrix[np.ix_(idx_a, idx_b)].ravel()
            values = values[np.isfinite(values)]
            if len(values):
                output[i, j] = float(np.nanmedian(values))
    return output


def scale_pearson_to_similarity(matrix: np.ndarray) -> np.ndarray:
    return np.clip((matrix + 1.0) / 2.0, 0.0, 1.0)


def weighted_average(components: list[tuple[str, np.ndarray, float]]) -> np.ndarray:
    n = components[0][1].shape[0]
    output = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            numerator = 0.0
            denominator = 0.0
            for _name, matrix, weight in components:
                value = matrix[i, j]
                if np.isfinite(value):
                    numerator += weight * float(value)
                    denominator += weight
            if denominator > 0:
                output[i, j] = numerator / denominator
    return output


def write_component_pair_table(path: Path, labels: list[str], components: list[tuple[str, np.ndarray, float]], overall: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_a", "frame_b", "overall", *[name for name, _matrix, _weight in components]])
        for i, frame_a in enumerate(labels):
            for j, frame_b in enumerate(labels):
                if i <= j:
                    continue
                writer.writerow(
                    [
                        frame_a,
                        frame_b,
                        format_value(float(overall[i, j])),
                        *[format_value(float(matrix[i, j])) for _name, matrix, _weight in components],
                    ]
                )


def build_overall_correlations(args: argparse.Namespace, out_dir: Path, static_mask: np.ndarray) -> None:
    peak_dir = args.suite_dir / "01_per_peak_frame_correlation"
    labels, group_labels, roi_area = read_feature_table(peak_dir / "peak_roi_area_features.csv")
    _, _, presence = read_feature_table(peak_dir / "peak_presence_features.csv")
    reliable_mask = peak_group_reliability_mask(
        out_dir / "reliable_peak_groups.csv",
        labels,
        group_labels,
        roi_area,
        presence,
        args,
    )

    same_window_raw = aggregate_same_window_acf(args.suite_dir, labels)
    all_window_raw = aggregate_all_window_acf(args.suite_dir, labels)
    same_window = scale_pearson_to_similarity(same_window_raw)
    all_window = scale_pearson_to_similarity(all_window_raw)

    component_specs = []
    for prefix, mask in [
        ("all_peaks", np.ones(roi_area.shape[1], dtype=bool)),
        ("reliable_peaks", reliable_mask),
        ("static_removed", ~static_mask),
        ("reliable_static_removed", reliable_mask & ~static_mask),
    ]:
        area = roi_area[:, mask]
        peak_presence = presence[:, mask]
        area_cosine = cosine_similarity(area)
        presence_jaccard = jaccard_similarity(peak_presence > 0)
        components = [
            ("peak_presence_jaccard", presence_jaccard, args.weight_peak_presence),
            ("peak_roi_area_cosine", area_cosine, args.weight_peak_area),
            ("same_window_acf_shifted_median", same_window, args.weight_same_window_acf),
            ("all_window_acf_frame_median", all_window, args.weight_all_window_acf),
        ]
        overall = weighted_average(components)
        write_matrix(out_dir / f"overall_{prefix}_correlation_matrix.csv", labels, lower_triangle(overall))
        plot_heatmap(
            out_dir / f"overall_{prefix}_correlation_heatmap.png",
            labels,
            lower_triangle(overall),
            f"Overall frame correlation ({prefix.replace('_', ' ')})",
            "weighted similarity",
        )
        write_component_pair_table(out_dir / f"overall_{prefix}_pair_components.csv", labels, components, overall)
        component_specs.append((prefix, components, overall))

    raw_components = [
        ("same_window_acf_shifted_median_raw_pearson", same_window_raw),
        ("all_window_acf_frame_median_raw_pearson", all_window_raw),
        ("same_window_acf_shifted_median_scaled", same_window),
        ("all_window_acf_frame_median_scaled", all_window),
    ]
    for name, matrix in raw_components:
        write_matrix(out_dir / f"component_{name}_matrix.csv", labels, lower_triangle(matrix))
        plot_heatmap(
            out_dir / f"component_{name}_heatmap.png",
            labels,
            lower_triangle(matrix),
            name.replace("_", " "),
            "similarity" if "scaled" in name else "Pearson",
            vmin=0.0 if "scaled" in name else -1.0,
            vmax=1.0,
            cmap_name="viridis" if "scaled" in name else "coolwarm",
        )

    for prefix, components, _overall in component_specs:
        for name, matrix, _weight in components[:2]:
            write_matrix(out_dir / f"component_{prefix}_{name}_matrix.csv", labels, lower_triangle(matrix))
            plot_heatmap(
                out_dir / f"component_{prefix}_{name}_heatmap.png",
                labels,
                lower_triangle(matrix),
                f"{prefix.replace('_', ' ')} {name.replace('_', ' ')}",
                "similarity",
            )

    with (out_dir / "overall_component_weights.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "weight", "used_in_overall"])
        writer.writerow(["peak_presence_jaccard", args.weight_peak_presence, "yes"])
        writer.writerow(["peak_roi_area_cosine", args.weight_peak_area, "yes"])
        writer.writerow(["same_window_acf_shifted_median_scaled", args.weight_same_window_acf, "yes"])
        writer.writerow(["all_window_acf_frame_median_scaled", args.weight_all_window_acf, "yes"])
        writer.writerow(["single_frame_window_acf", "", "no; it compares windows inside one frame, not frame-to-frame"])


def write_readme(args: argparse.Namespace, out_dir: Path, static_mask: np.ndarray) -> None:
    with (out_dir / "README.txt").open("w") as handle:
        handle.write("Static peak and overall correlation post-processing\n")
        handle.write("===================================================\n\n")
        handle.write("Static peak identification:\n")
        handle.write("  A suspected static peak group must have high coverage, enough ROI signal, small 2theta MAD, and small pressure slope.\n")
        handle.write("  This intentionally does not allow +/-1 deg shifting; that would hide true pressure-dependent material peak motion.\n\n")
        handle.write(f"  min-static-frame-fraction: {args.min_static_frame_fraction:g}\n")
        handle.write(f"  max-static-mad-deg: {args.max_static_mad_deg:g}\n")
        handle.write(f"  max-static-abs-slope: {args.max_static_abs_slope:g} deg/GPa\n")
        handle.write(f"  suspected static groups: {int(np.count_nonzero(static_mask))}\n\n")
        handle.write("Reliable peak set:\n")
        handle.write("  This filters the exhaustive raw-local-max peak list before peak-based overall maps.\n")
        handle.write(f"  min-reliable-frame-fraction: {args.min_reliable_frame_fraction:g}\n")
        handle.write(f"  min-reliable-max-roi-area: {args.min_reliable_max_roi_area:g}\n\n")
        handle.write("Overall frame correlation:\n")
        handle.write("  The overall score is a weighted average of normalized component similarities.\n")
        handle.write("  Components: peak presence Jaccard, peak ROI area cosine, same-window ACF median, all-window ACF frame median.\n")
        handle.write("  single_frame_window_acf is excluded because it compares windows within one frame rather than two frames.\n\n")
        handle.write("Main outputs:\n")
        handle.write("  suspected_static_peak_groups.csv\n")
        handle.write("  reliable_peak_groups.csv\n")
        handle.write("  all_peak_static_scores.csv\n")
        handle.write("  overall_all_peaks_correlation_heatmap.png/.csv\n")
        handle.write("  overall_reliable_peaks_correlation_heatmap.png/.csv\n")
        handle.write("  overall_static_removed_correlation_heatmap.png/.csv\n")
        handle.write("  overall_reliable_static_removed_correlation_heatmap.png/.csv\n")
        handle.write("  overall_*_pair_components.csv\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.suite_dir / "05_static_peaks_and_overall_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    _group_labels, static_mask = identify_static_peaks(args, out_dir)
    build_overall_correlations(args, out_dir, static_mask)
    write_readme(args, out_dir, static_mask)
    print(f"Wrote static peak and overall correlation outputs to {out_dir}")


if __name__ == "__main__":
    main()
