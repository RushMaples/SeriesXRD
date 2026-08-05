#!/usr/bin/env python3
"""Validate a powder/spots BulkXRD four-map correlation run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument(
        "--known-repeat-pairs",
        default="",
        help="Comma-separated zero-based pairs, for example 0-1,2-3.",
    )
    parser.add_argument(
        "--known-different-pairs",
        default="",
        help="Comma-separated zero-based pairs expected to be less similar.",
    )
    parser.add_argument("--require-known-ordering", action="store_true")
    return parser.parse_args()


def parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        left, right = token.split("-", 1)
        pairs.append((int(left), int(right)))
    return pairs


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Empty or malformed matrix: {path}")
    labels = rows[0][1:]
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for row_index, row in enumerate(rows[1:len(labels) + 1]):
        for col_index, value in enumerate(row[1:len(labels) + 1]):
            try:
                matrix[row_index, col_index] = float(value)
            except ValueError:
                pass
    return labels, matrix


def matrix_family_stats(paths: list[Path]) -> dict[str, float | int | bool]:
    total_cells = 0
    finite_cells = 0
    shapes_valid = True
    labels_unique = True
    for path in paths:
        labels, matrix = read_matrix(path)
        shapes_valid &= matrix.shape == (len(labels), len(labels))
        labels_unique &= len(labels) == len(set(labels))
        total_cells += matrix.size
        finite_cells += int(np.count_nonzero(np.isfinite(matrix)))
    return {
        "count": len(paths),
        "square_shapes": bool(shapes_valid),
        "unique_labels": bool(labels_unique),
        "finite_values": finite_cells,
        "finite_fraction": finite_cells / total_cells if total_cells else 0.0,
    }


def aggregate_pairs(paths: list[Path], pairs: list[tuple[int, int]]) -> float:
    values: list[float] = []
    for path in paths:
        labels, matrix = read_matrix(path)
        for first, second in pairs:
            if first >= len(labels) or second >= len(labels):
                continue
            row, col = max(first, second), min(first, second)
            value = float(matrix[row, col])
            if np.isfinite(value):
                values.append(value)
    return float(np.median(values)) if values else float("nan")


def load_xy(path: Path) -> np.ndarray:
    values = np.loadtxt(path, comments="#")
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Malformed XY file: {path}")
    return np.asarray(values[:, :2], dtype=float)


def channel_paths(channel_dir: Path) -> dict[str, list[Path]]:
    return {
        "area": sorted((channel_dir / "01_per_peak_frame_correlation").rglob("per_peak_matrices/*.csv")),
        "location": sorted((channel_dir / "01_per_peak_frame_correlation").rglob("per_peak_position_matrices/*.csv")),
        "same_window": sorted((channel_dir / "02_same_window_acf_across_frames").rglob("matrices/*.csv")),
        "within_frame": sorted((channel_dir / "03_single_frame_window_acf").rglob("matrices/*.csv")),
    }


def source_and_good_peaks(path: Path) -> tuple[str, int]:
    with h5py.File(path, "r") as h5:
        peaks = h5["peaks"]
        source = peaks.attrs.get("source", "unknown")
        if isinstance(source, bytes):
            source = source.decode("utf-8", "replace")
        flags = np.asarray(peaks["flag"][:], dtype=int)
    return str(source), int(np.count_nonzero(flags == 0))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repeat_pairs = parse_pairs(args.known_repeat_pairs)
    different_pairs = parse_pairs(args.known_different_pairs)

    checks: dict[str, bool] = {}
    channels: dict[str, object] = {}
    family_paths: dict[str, dict[str, list[Path]]] = {}
    for channel in ("powder", "spots"):
        channel_dir = run_dir / channel
        paths = channel_paths(channel_dir)
        family_paths[channel] = paths
        with (channel_dir / "00_bulkxrd_xy/conversion_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            exported = list(csv.DictReader(handle))
        analysis_path = Path(manifest["channels"][channel]["analysis_h5"])
        source, good_peaks = source_and_good_peaks(analysis_path)
        family_stats = {name: matrix_family_stats(items) for name, items in paths.items()}
        channel_checks = {
            "analysis_h5_present": analysis_path.is_file(),
            "good_fitted_peaks_present": good_peaks > 0,
            "area_maps_present": bool(paths["area"]),
            "location_maps_present": bool(paths["location"]),
            "same_window_maps_present": bool(paths["same_window"]),
            "within_frame_maps_present": bool(paths["within_frame"]),
            "all_matrix_shapes_valid": all(bool(item["square_shapes"]) for item in family_stats.values()),
            "all_matrix_labels_unique": all(bool(item["unique_labels"]) for item in family_stats.values()),
            "all_families_have_finite_values": all(int(item["finite_values"]) > 0 for item in family_stats.values()),
        }
        if args.expected_frames is not None:
            channel_checks["exported_frame_count"] = len(exported) == args.expected_frames
            channel_checks["within_frame_map_count"] = len(paths["within_frame"]) == args.expected_frames
        for name, passed in channel_checks.items():
            checks[f"{channel}_{name}"] = bool(passed)
        channels[channel] = {
            "fit_source": source,
            "good_fitted_peaks": good_peaks,
            "frames_exported": len(exported),
            "families": family_stats,
            "checks": channel_checks,
        }

    source_checks = {
        "powder_source_is_not_spots": channels["powder"]["fit_source"] != "spots",
        "spots_source_is_spots": channels["spots"]["fit_source"] == "spots",
    }
    checks.update(source_checks)

    powder_xy = sorted((run_dir / "powder/00_bulkxrd_xy").glob("*.xy"))
    spots_xy = sorted((run_dir / "spots/00_bulkxrd_xy").glob("*.xy"))
    channel_difference = float("nan")
    if powder_xy and spots_xy:
        powder = load_xy(powder_xy[0])
        spots = load_xy(spots_xy[0])
        same_axis = powder.shape == spots.shape and np.allclose(powder[:, 0], spots[:, 0])
        if same_axis:
            scale = max(float(np.nanmax(np.abs(powder[:, 1]))), 1e-12)
            channel_difference = float(np.nanmedian(np.abs(powder[:, 1] - spots[:, 1])) / scale)
        checks["powder_spots_axes_match"] = bool(same_axis)
        checks["powder_spots_signals_differ"] = bool(
            same_axis and not np.allclose(powder[:, 1], spots[:, 1], equal_nan=True)
        )
    else:
        checks["powder_spots_axes_match"] = False
        checks["powder_spots_signals_differ"] = False

    known_metrics: dict[str, dict[str, dict[str, float | bool]]] = {}
    if repeat_pairs and different_pairs:
        for channel in ("powder", "spots"):
            known_metrics[channel] = {}
            for family in ("area", "location", "same_window"):
                repeat_value = aggregate_pairs(family_paths[channel][family], repeat_pairs)
                different_value = aggregate_pairs(family_paths[channel][family], different_pairs)
                ordering = bool(
                    np.isfinite(repeat_value)
                    and np.isfinite(different_value)
                    and repeat_value > different_value
                )
                known_metrics[channel][family] = {
                    "repeat_pair_median": repeat_value,
                    "different_pair_median": different_value,
                    "repeat_above_different": ordering,
                }
                if args.require_known_ordering:
                    checks[f"{channel}_{family}_known_pair_ordering"] = ordering

    report = {
        "run_dir": str(run_dir),
        "channels": channels,
        "source_checks": source_checks,
        "powder_spots_first_frame_median_normalized_difference": channel_difference,
        "known_pair_metrics": known_metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    report_path = run_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True))
    if not report["passed"]:
        raise SystemExit("Dual-channel correlation validation failed")


if __name__ == "__main__":
    main()
