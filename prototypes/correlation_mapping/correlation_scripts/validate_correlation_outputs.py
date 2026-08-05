#!/usr/bin/env python3
"""Validate the four mapping families and known-similar synthetic frame pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-frames", type=int, default=8)
    return parser.parse_args()


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    labels = rows[0][1:]
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for row_index, row in enumerate(rows[1:]):
        for col_index, value in enumerate(row[1:]):
            try:
                matrix[row_index, col_index] = float(value)
            except (TypeError, ValueError):
                pass
    return labels, matrix


def pair_value(matrix: np.ndarray, first: int, second: int) -> float:
    row, col = max(first, second), min(first, second)
    return float(matrix[row, col])


def aggregate_pair(paths: list[Path], first: int, second: int) -> float:
    values = []
    for path in paths:
        _, matrix = read_matrix(path)
        value = pair_value(matrix, first, second)
        if np.isfinite(value):
            values.append(value)
    return float(np.median(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    suite = run_dir / "correlation_suite"
    area = sorted((suite / "01_per_peak_frame_correlation/per_peak_matrices").glob("*.csv"))
    location = sorted((suite / "01_per_peak_frame_correlation/per_peak_position_matrices").glob("*.csv"))
    across = sorted((suite / "02_same_window_acf_across_frames/matrices").glob("*.csv"))
    within = sorted((suite / "03_single_frame_window_acf/matrices").glob("*.csv"))
    with (run_dir / "00_bulkxrd_xy/conversion_manifest.csv").open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))

    checks = {
        "exported_frame_count": len(exported) == args.expected_frames,
        "per_peak_area_maps_present": bool(area),
        "per_peak_location_maps_present": bool(location),
        "same_window_across_frames_maps_present": bool(across),
        "within_frame_map_count": len(within) == args.expected_frames,
    }
    labels, first_area_matrix = read_matrix(area[0]) if area else ([], np.empty((0, 0)))
    checks["frame_labels_unique"] = len(labels) == len(set(labels)) == args.expected_frames
    checks["area_matrix_has_finite_scores"] = bool(np.isfinite(first_area_matrix).any())
    if within:
        _, within_matrix = read_matrix(within[0])
        checks["within_frame_matrix_has_finite_scores"] = bool(np.isfinite(within_matrix).any())
    else:
        checks["within_frame_matrix_has_finite_scores"] = False

    metrics = {
        "area_similar_pair_median": aggregate_pair(area, 1, 0),
        "area_different_pair_median": aggregate_pair(area, 6, 0),
        "location_similar_pair_median": aggregate_pair(location, 1, 0),
        "location_different_pair_median": aggregate_pair(location, 6, 0),
        "across_window_similar_pair_median": aggregate_pair(across, 1, 0),
        "across_window_different_pair_median": aggregate_pair(across, 6, 0),
    }
    for family in ("area", "location", "across_window"):
        similar = metrics[f"{family}_similar_pair_median"]
        different = metrics[f"{family}_different_pair_median"]
        checks[f"{family}_known_pair_ordering"] = bool(
            np.isfinite(similar) and np.isfinite(different) and similar > different
        )

    report = {
        "run_dir": str(run_dir),
        "map_counts": {
            "per_peak_area": len(area),
            "per_peak_location": len(location),
            "same_window_across_frames": len(across),
            "window_to_window_within_frame": len(within),
        },
        "checks": checks,
        "metrics": metrics,
        "passed": all(checks.values()),
    }
    report_path = run_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True))
    if not report["passed"]:
        raise SystemExit("Correlation validation failed")


if __name__ == "__main__":
    main()
