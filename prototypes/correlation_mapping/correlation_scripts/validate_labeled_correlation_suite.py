#!/usr/bin/env python3
"""Validate correlation maps against known shared-phase labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
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


def read_truth(path: Path) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            filename = str(row.get("filename", "")).strip()
            phases = {item.strip() for item in str(row.get("phases", "")).split(";") if item.strip()}
            if filename and phases:
                truth[Path(filename).stem] = phases
    return truth


def canonical_label(label: str) -> str:
    return label.split("[", 1)[0].strip()


def pair_value(matrix: np.ndarray, first: int, second: int) -> float:
    row, col = max(first, second), min(first, second)
    return float(matrix[row, col])


def aggregate_pair_scores(paths: list[Path], labels: list[str]) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], list[float]] = {
        (i, j): [] for i in range(len(labels)) for j in range(i)
    }
    for path in paths:
        current_labels, matrix = read_matrix(path)
        if current_labels != labels:
            raise ValueError(f"Frame-label mismatch in {path}")
        for pair in values:
            value = pair_value(matrix, *pair)
            if np.isfinite(value):
                values[pair].append(value)
    return {
        pair: float(np.mean(pair_values)) if pair_values else float("nan")
        for pair, pair_values in values.items()
    }


def auc(shared: list[float], disjoint: list[float]) -> float:
    if not shared or not disjoint:
        return float("nan")
    wins = 0.0
    for left in shared:
        for right in disjoint:
            wins += float(left > right) + 0.5 * float(left == right)
    return wins / (len(shared) * len(disjoint))


def family_metrics(
    paths: list[Path],
    labels: list[str],
    truth: dict[str, set[str]],
) -> dict[str, object]:
    pair_scores = aggregate_pair_scores(paths, labels)
    shared: list[float] = []
    disjoint: list[float] = []
    pairs: list[dict[str, object]] = []
    for (i, j), score in pair_scores.items():
        left = canonical_label(labels[i])
        right = canonical_label(labels[j])
        left_truth = truth.get(left, set())
        right_truth = truth.get(right, set())
        if not left_truth or not right_truth or not np.isfinite(score):
            continue
        overlap = sorted(left_truth & right_truth)
        target = shared if overlap else disjoint
        target.append(score)
        pairs.append({
            "left": left,
            "right": right,
            "shared_phases": overlap,
            "score": score,
        })
    shared_median = float(np.median(shared)) if shared else float("nan")
    disjoint_median = float(np.median(disjoint)) if disjoint else float("nan")
    separation_auc = auc(shared, disjoint)
    return {
        "n_maps": len(paths),
        "n_shared_pairs": len(shared),
        "n_disjoint_pairs": len(disjoint),
        "shared_pair_median": shared_median,
        "disjoint_pair_median": disjoint_median,
        "shared_vs_disjoint_auc": separation_auc,
        "shared_median_above_disjoint": bool(
            np.isfinite(shared_median)
            and np.isfinite(disjoint_median)
            and shared_median > disjoint_median
        ),
        "auc_above_chance": bool(np.isfinite(separation_auc) and separation_auc > 0.5),
        "pair_scores": sorted(pairs, key=lambda row: float(row["score"]), reverse=True),
    }


def main() -> None:
    args = parse_args()
    suite = args.suite_dir.expanduser().resolve()
    if (suite / "correlation_suite").is_dir():
        suite = suite / "correlation_suite"
    truth = read_truth(args.labels.expanduser().resolve())

    area = sorted((suite / "01_per_peak_frame_correlation/per_peak_matrices").glob("*.csv"))
    location = sorted((suite / "01_per_peak_frame_correlation/per_peak_position_matrices").glob("*.csv"))
    across = sorted((suite / "02_same_window_acf_across_frames/matrices").glob("*.csv"))
    within = sorted((suite / "03_single_frame_window_acf/matrices").glob("*.csv"))
    if not area:
        raise SystemExit(f"No per-peak area matrices under {suite}")
    labels, first_matrix = read_matrix(area[0])

    structural = {
        "per_peak_area_maps_present": bool(area),
        "per_peak_location_map_count_matches": len(location) == len(area),
        "same_window_across_frames_maps_present": bool(across),
        "within_frame_map_count_matches_frames": len(within) == len(labels),
        "frame_labels_unique": len(labels) == len(set(labels)),
        "all_frames_have_truth_labels": all(canonical_label(label) in truth for label in labels),
        "area_matrix_has_finite_scores": bool(np.isfinite(first_matrix).any()),
    }
    families = {
        "per_peak_area": family_metrics(area, labels, truth),
        "per_peak_location": family_metrics(location, labels, truth),
        "same_window_across_frames": family_metrics(across, labels, truth),
    }
    semantic = {
        name: bool(metrics["shared_median_above_disjoint"] and metrics["auc_above_chance"])
        for name, metrics in families.items()
    }
    report = {
        "suite_dir": str(suite),
        "labels_csv": str(args.labels.expanduser().resolve()),
        "frames": labels,
        "map_counts": {
            "per_peak_area": len(area),
            "per_peak_location": len(location),
            "same_window_across_frames": len(across),
            "window_to_window_within_frame": len(within),
        },
        "structural_checks": structural,
        "semantic_checks": semantic,
        "families": families,
        "passed": all(structural.values()) and all(semantic.values()),
    }
    out = args.out or (suite / "labeled_validation_report.json")
    out = out.expanduser().resolve()
    out.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    summary = {
        "map_counts": report["map_counts"],
        "structural_checks": structural,
        "semantic_checks": semantic,
        "family_summary": {
            name: {key: metrics[key] for key in (
                "n_shared_pairs", "n_disjoint_pairs", "shared_pair_median",
                "disjoint_pair_median", "shared_vs_disjoint_auc",
            )}
            for name, metrics in families.items()
        },
        "passed": report["passed"],
        "report": str(out),
    }
    print(json.dumps(summary, indent=2, allow_nan=True))
    if not report["passed"]:
        raise SystemExit("Labeled correlation validation failed")


if __name__ == "__main__":
    main()
