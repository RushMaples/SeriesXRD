#!/usr/bin/env python3
"""Create one frame-vs-frame correlation map for each detected XRD peak group."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import compare_integrated_peaks as cip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect as many integrated-pattern peaks as possible, group them by "
            "2theta, and write one lower-triangle frame correlation map per peak group."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("Data/Cell_14_integrated"), Path("Data/Cell_29_integrated")],
        help="Integrated .xy files or directories containing .xy files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/per_peak_correlation_maps_latest"),
        help="Output directory.",
    )
    parser.add_argument("--min-two-theta", type=float, default=2.0)
    parser.add_argument("--max-two-theta", type=float, default=None)
    parser.add_argument("--peak-match-tolerance", type=float, default=0.02)
    parser.add_argument("--prominence", type=float, default=0.2)
    parser.add_argument("--bump-prominence", type=float, default=0.05)
    parser.add_argument("--micro-prominence", type=float, default=0.005)
    parser.add_argument("--raw-prominence", type=float, default=0.0)
    parser.add_argument("--shoulder-prominence", type=float, default=0.0)
    parser.add_argument("--matched-filter-prominence", type=float, default=0.0)
    parser.add_argument("--matched-filter-widths", default="2,3,5,8")
    parser.add_argument("--min-shape-snr", type=float, default=0.0)
    parser.add_argument("--min-shape-contrast", type=float, default=0.0)
    parser.add_argument("--shape-half-window", type=float, default=0.16)
    parser.add_argument("--min-micro-snr", type=float, default=0.0)
    parser.add_argument("--top-peaks", type=int, default=4000)
    parser.add_argument("--distance", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=21)
    parser.add_argument("--micro-smooth-window", type=int, default=7)
    parser.add_argument("--baseline-window", type=int, default=151)
    parser.add_argument("--min-peak-height", type=float, default=0.0)
    parser.add_argument("--bump-rise-window", type=float, default=0.25)
    parser.add_argument("--min-bump-rise", type=float, default=0.0)
    parser.add_argument("--merge-tolerance", type=float, default=0.0)
    parser.add_argument("--roi-half-width", type=float, default=0.06)
    parser.add_argument("--roi-sideband-gap", type=float, default=0.03)
    parser.add_argument("--roi-sideband-width", type=float, default=0.08)
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.06,
        help="2theta difference that maps to zero in per-peak position similarity.",
    )
    parser.add_argument("--small-peak-max-prominence", type=float, default=999.0)
    parser.add_argument("--small-peak-max-width", type=float, default=999.0)
    parser.add_argument(
        "--min-frame-count",
        type=int,
        default=1,
        help="Only plot peak groups detected in at least this many frames.",
    )
    parser.add_argument(
        "--review-plots",
        action="store_true",
        help="Also write .xy review plots with all detected peaks marked.",
    )
    parser.add_argument(
        "--include-tier-c",
        action="store_true",
        help="Include likely-artifact Tier C candidates in peak_table and correlation maps.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def per_peak_similarity(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    n = len(values)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(i):
            if present[i] and present[j]:
                denom = max(float(values[i]), float(values[j]), 1e-12)
                score = 1.0 - abs(float(values[i]) - float(values[j])) / denom
                matrix[i, j] = float(np.clip(score, 0.0, 1.0))
            elif present[i] != present[j]:
                matrix[i, j] = 0.0
    return matrix


def build_peak_position_vectors(
    patterns: list[cip.Pattern],
    peaks: list[cip.Peak],
    group_centers: list[float],
) -> np.ndarray:
    label_to_row = {pattern.label: index for index, pattern in enumerate(patterns)}
    positions = np.full((len(patterns), len(group_centers)), np.nan, dtype=float)
    best_score = np.full_like(positions, -np.inf)

    for peak in peaks:
        if peak.group_index is None:
            continue
        row = label_to_row[peak.pattern_label]
        col = peak.group_index - 1
        score = (
            float(peak.tier_score),
            float(peak.source_count),
            float(peak.prominence),
            float(peak.intensity),
        )
        scalar_score = score[0] * 1e6 + score[1] * 1e3 + score[2] + score[3] * 1e-3
        if scalar_score > best_score[row, col]:
            best_score[row, col] = scalar_score
            positions[row, col] = peak.two_theta
    return positions


def per_peak_position_similarity(
    positions: np.ndarray,
    present: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    n = len(positions)
    matrix = np.full((n, n), np.nan, dtype=float)
    tolerance = max(float(tolerance), 1e-12)
    finite = present & np.isfinite(positions)
    for i in range(n):
        for j in range(i):
            if finite[i] and finite[j]:
                delta = abs(float(positions[i]) - float(positions[j]))
                score = 1.0 - delta / tolerance
                matrix[i, j] = float(np.clip(score, 0.0, 1.0))
            elif finite[i] != finite[j]:
                matrix[i, j] = 0.0
    return matrix


def write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[cip.format_value(value) for value in row]])


def plot_lower_triangle_heatmap(
    path: Path,
    labels: list[str],
    matrix: np.ndarray,
    title: str,
    colorbar_label: str = "peak area similarity",
) -> None:
    size = max(6.0, 0.42 * len(labels) + 2.0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_long_pair_table(
    path: Path,
    labels: list[str],
    group_index: int,
    center: float,
    values: np.ndarray,
    present: np.ndarray,
    matrix: np.ndarray,
) -> None:
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                [
                    "peak_group",
                    "group_two_theta",
                    "frame_a",
                    "frame_b",
                    "present_a",
                    "present_b",
                    "area_a",
                    "area_b",
                    "similarity",
                ]
            )
        for i in range(len(labels)):
            for j in range(i):
                writer.writerow(
                    [
                        group_index,
                        f"{center:.6f}",
                        labels[i],
                        labels[j],
                        int(bool(present[i])),
                        int(bool(present[j])),
                        cip.format_value(float(values[i])),
                        cip.format_value(float(values[j])),
                        cip.format_value(float(matrix[i, j])),
                    ]
                )


def write_long_position_pair_table(
    path: Path,
    labels: list[str],
    group_index: int,
    center: float,
    positions: np.ndarray,
    present: np.ndarray,
    matrix: np.ndarray,
) -> None:
    file_exists = path.exists()
    finite = present & np.isfinite(positions)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                [
                    "peak_group",
                    "group_two_theta",
                    "frame_a",
                    "frame_b",
                    "present_a",
                    "present_b",
                    "position_a",
                    "position_b",
                    "delta_two_theta",
                    "similarity",
                ]
            )
        for i in range(len(labels)):
            for j in range(i):
                delta = (
                    abs(float(positions[i]) - float(positions[j]))
                    if finite[i] and finite[j]
                    else np.nan
                )
                writer.writerow(
                    [
                        group_index,
                        f"{center:.6f}",
                        labels[i],
                        labels[j],
                        int(bool(finite[i])),
                        int(bool(finite[j])),
                        cip.format_value(float(positions[i])),
                        cip.format_value(float(positions[j])),
                        cip.format_value(float(delta)),
                        cip.format_value(float(matrix[i, j])),
                    ]
                )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.out_dir / "per_peak_heatmaps"
    matrix_dir = args.out_dir / "per_peak_matrices"
    position_heatmap_dir = args.out_dir / "per_peak_position_heatmaps"
    position_matrix_dir = args.out_dir / "per_peak_position_matrices"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    position_heatmap_dir.mkdir(parents=True, exist_ok=True)
    position_matrix_dir.mkdir(parents=True, exist_ok=True)

    detector_args = SimpleNamespace(**vars(args))
    xy_files = cip.discover_xy_files(args.inputs)
    if not xy_files:
        raise SystemExit("No .xy files found.")

    patterns = [cip.load_pattern(path, detector_args) for path in xy_files]
    patterns = sorted(
        patterns,
        key=lambda pattern: (
            float("inf") if pattern.pressure_gpa is None else pattern.pressure_gpa,
            pattern.label,
        ),
    )
    cip.ensure_unique_pattern_labels(patterns)
    labels = [pattern.label for pattern in patterns]
    all_candidates = [peak for pattern in patterns for peak in cip.detect_peaks(pattern, detector_args)]
    tier_c_candidates = [peak for peak in all_candidates if peak.confidence_tier == "C"]
    peaks = all_candidates if args.include_tier_c else [
        peak for peak in all_candidates if peak.confidence_tier != "C"
    ]
    group_centers = cip.group_peaks(peaks, args.peak_match_tolerance)
    presence, peak_intensity = cip.build_peak_vectors(patterns, peaks, group_centers)
    roi_area = cip.build_peak_roi_area_vectors(patterns, group_centers, detector_args)
    peak_position = build_peak_position_vectors(patterns, peaks, group_centers)
    peak_position_delta = peak_position - np.array(group_centers, dtype=float)
    small_peaks = cip.select_small_peak_candidates(peaks, detector_args)

    cip.write_peak_table(args.out_dir / "all_candidate_table.csv", all_candidates)
    cip.write_peak_table(args.out_dir / "tier_c_candidate_table.csv", tier_c_candidates)
    cip.write_peak_table(args.out_dir / "peak_table.csv", peaks)
    cip.write_peak_table(args.out_dir / "small_peak_table.csv", small_peaks)
    cip.write_group_table(args.out_dir / "peak_group_table.csv", labels, group_centers, presence, peak_intensity, peaks)
    cip.write_peak_group_summary(
        args.out_dir / "peak_group_summary.csv",
        patterns,
        group_centers,
        peaks,
        roi_area,
        presence,
    )
    cip.write_feature_table(args.out_dir / "peak_roi_area_features.csv", labels, group_centers, roi_area)
    cip.write_feature_table(args.out_dir / "peak_presence_features.csv", labels, group_centers, presence)
    cip.write_feature_table(args.out_dir / "peak_position_features.csv", labels, group_centers, peak_position)
    cip.write_feature_table(
        args.out_dir / "peak_position_delta_features.csv",
        labels,
        group_centers,
        peak_position_delta,
    )
    if args.review_plots:
        cip.plot_peak_review(args.out_dir, patterns, peaks, small_peaks)

    long_pairs = args.out_dir / "all_per_peak_pair_similarities.csv"
    if long_pairs.exists():
        long_pairs.unlink()
    long_position_pairs = args.out_dir / "all_per_peak_position_pair_similarities.csv"
    if long_position_pairs.exists():
        long_position_pairs.unlink()

    index_path = args.out_dir / "per_peak_map_index.csv"
    position_index_path = args.out_dir / "per_peak_position_map_index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "peak_group",
                "group_two_theta",
                "frame_count",
                "frames_present",
                "max_roi_area",
                "frame_coverage_fraction",
                "tier_a_count",
                "tier_b_count",
                "tier_c_count",
                "source_methods",
                "matched_filter_scales",
                "heatmap",
                "matrix",
            ]
        )
        for col, center in enumerate(group_centers):
            present = presence[:, col] > 0
            frame_count = int(np.count_nonzero(present))
            if frame_count < args.min_frame_count:
                continue
            members = [peak for peak in peaks if peak.group_index == col + 1]
            tier_counts = {
                tier: sum(1 for peak in members if peak.confidence_tier == tier)
                for tier in ("A", "B", "C")
            }
            sources = sorted(
                {
                    source
                    for peak in members
                    for source in peak.source_methods.split(";")
                    if source
                }
            )
            scales = sorted(
                {
                    str(peak.best_matched_scale)
                    for peak in members
                    if peak.best_matched_scale is not None
                }
            )
            values = roi_area[:, col]
            matrix = per_peak_similarity(values, present)
            stem = f"peak_group_{col + 1:03d}_{center:.3f}deg"
            heatmap_path = heatmap_dir / f"{safe_name(stem)}_correlation.png"
            matrix_path = matrix_dir / f"{safe_name(stem)}_correlation.csv"
            title = f"Peak group {col + 1}: {center:.3f} deg"
            plot_lower_triangle_heatmap(heatmap_path, labels, matrix, title)
            write_matrix(matrix_path, labels, matrix)
            write_long_pair_table(long_pairs, labels, col + 1, center, values, present, matrix)
            writer.writerow(
                [
                    col + 1,
                    f"{center:.6f}",
                    frame_count,
                    ";".join(label for label, keep in zip(labels, present) if keep),
                    cip.format_value(float(np.nanmax(values))),
                    cip.format_value(float(frame_count / max(len(labels), 1))),
                    tier_counts["A"],
                    tier_counts["B"],
                    tier_counts["C"],
                    ";".join(sources),
                    ";".join(scales),
                    str(heatmap_path.relative_to(args.out_dir)),
                    str(matrix_path.relative_to(args.out_dir)),
                ]
            )

    with position_index_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "peak_group",
                "group_two_theta",
                "frame_count",
                "frames_present",
                "position_tolerance_deg",
                "position_range_deg",
                "position_mad_deg",
                "heatmap",
                "matrix",
            ]
        )
        for col, center in enumerate(group_centers):
            present = presence[:, col] > 0
            finite = present & np.isfinite(peak_position[:, col])
            frame_count = int(np.count_nonzero(finite))
            if frame_count < args.min_frame_count:
                continue
            values = peak_position[:, col]
            matrix = per_peak_position_similarity(values, present, args.position_tolerance)
            stem = f"peak_group_{col + 1:03d}_{center:.3f}deg"
            heatmap_path = position_heatmap_dir / f"{safe_name(stem)}_position_correlation.png"
            matrix_path = position_matrix_dir / f"{safe_name(stem)}_position_correlation.csv"
            title = f"Peak position group {col + 1}: {center:.3f} deg"
            plot_lower_triangle_heatmap(
                heatmap_path,
                labels,
                matrix,
                title,
                colorbar_label="peak position similarity",
            )
            write_matrix(matrix_path, labels, matrix)
            write_long_position_pair_table(
                long_position_pairs,
                labels,
                col + 1,
                center,
                values,
                present,
                matrix,
            )
            finite_values = values[finite]
            median = float(np.nanmedian(finite_values)) if finite_values.size else np.nan
            mad = (
                float(np.nanmedian(np.abs(finite_values - median)))
                if finite_values.size and np.isfinite(median)
                else np.nan
            )
            value_range = (
                float(np.nanmax(finite_values) - np.nanmin(finite_values))
                if finite_values.size
                else np.nan
            )
            writer.writerow(
                [
                    col + 1,
                    f"{center:.6f}",
                    frame_count,
                    ";".join(label for label, keep in zip(labels, finite) if keep),
                    cip.format_value(args.position_tolerance),
                    cip.format_value(value_range),
                    cip.format_value(mad),
                    str(heatmap_path.relative_to(args.out_dir)),
                    str(matrix_path.relative_to(args.out_dir)),
                ]
            )

    with (args.out_dir / "README.txt").open("w") as handle:
        handle.write("Per-peak frame correlation maps\n")
        handle.write("================================\n\n")
        handle.write(
            "Each heatmap is one peak group. The score compares the ROI area of that peak "
            "between two frames.\n"
        )
        handle.write("If both frames have the peak: score = 1 - abs(area_a-area_b)/max(area_a,area_b).\n")
        handle.write("If only one frame has the peak: score = 0.\n")
        handle.write("If neither frame has the peak: blank/NaN.\n")
        handle.write("The diagonal and upper triangle are blank because they are redundant.\n\n")
        handle.write(
            "A second set of maps compares detected peak positions for each peak group. "
            "If both frames have the peak: position score = "
            "1 - abs(two_theta_a-two_theta_b)/position_tolerance, clipped to 0..1.\n"
        )
        handle.write("Position maps are written under per_peak_position_heatmaps/ and per_peak_position_matrices/.\n\n")
        handle.write(f"Input files: {len(patterns)}\n")
        handle.write(f"All generated candidates: {len(all_candidates)}\n")
        handle.write(f"Detected peaks used in this suite: {len(peaks)}\n")
        handle.write(f"Tier A candidates used: {sum(1 for peak in peaks if peak.confidence_tier == 'A')}\n")
        handle.write(f"Tier B candidates used: {sum(1 for peak in peaks if peak.confidence_tier == 'B')}\n")
        handle.write(f"Tier C candidates generated: {len(tier_c_candidates)}\n")
        handle.write(f"Tier C included in correlation maps: {int(bool(args.include_tier_c))}\n")
        handle.write(f"Small-peak candidates: {len(small_peaks)}\n")
        handle.write(f"Peak groups: {len(group_centers)}\n")
        handle.write(f"Peak match tolerance: {args.peak_match_tolerance:g} deg\n")
        handle.write(f"ROI half-width: {args.roi_half_width:g} deg\n")
        handle.write(f"Position tolerance: {args.position_tolerance:g} deg\n")
        handle.write(f"micro-prominence: {args.micro_prominence:g}\n")
        handle.write(f"raw-prominence: {args.raw_prominence:g}\n")
        handle.write(f"shoulder-prominence: {args.shoulder_prominence:g}\n")
        handle.write(f"matched-filter-prominence: {args.matched_filter_prominence:g}\n")
        handle.write(f"matched-filter-widths: {args.matched_filter_widths}\n")
        handle.write(f"min-micro-snr: {args.min_micro_snr:g}\n")
        handle.write(f"min-shape-snr: {args.min_shape_snr:g}\n")
        handle.write(f"min-shape-contrast: {args.min_shape_contrast:g}\n")
        handle.write(f"shape-half-window: {args.shape_half_window:g}\n")
        handle.write(f"merge-tolerance: {args.merge_tolerance:g}\n")
        handle.write(f"top-peaks: {args.top_peaks:g}\n")
        handle.write(f"Top-peaks truncation occurred: {int(any(peak.top_peaks_truncated for peak in all_candidates))}\n")

    print(f"Wrote {len(group_centers)} per-peak maps to {args.out_dir}")


if __name__ == "__main__":
    main()
