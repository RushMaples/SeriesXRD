#!/usr/bin/env python3
"""Summarize and visually compare peak-correlation suite outputs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SUITES = [
    Path("outputs/correlation_suite_20260619_raw_ultra_peaks"),
    Path("outputs/correlation_suite_20260621_exhaustive_shoulders"),
    Path("outputs/correlation_suite_20260621_high_recall_scored_v2"),
    Path("outputs/correlation_suite_20260621_all_candidate_diagnostic_v2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare XRD peak-correlation suite outputs.")
    parser.add_argument("suites", nargs="*", type=Path, default=DEFAULT_SUITES)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/correlation_suite_20260621_scored_comparison"),
    )
    return parser.parse_args()


def per_peak_dir(suite: Path) -> Path:
    nested = suite / "01_per_peak_frame_correlation"
    return nested if nested.exists() else suite


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def suite_metrics(suite: Path) -> dict[str, object]:
    folder = per_peak_dir(suite)
    peaks = read_csv(folder / "peak_table.csv")
    all_candidates = read_csv(folder / "all_candidate_table.csv")
    tier_c = read_csv(folder / "tier_c_candidate_table.csv")
    groups = read_csv(folder / "per_peak_map_index.csv")
    source = all_candidates if not all_candidates.empty else peaks
    tier_counts = (
        source["confidence_tier"].value_counts().to_dict()
        if "confidence_tier" in source.columns
        else {}
    )
    used_tier_counts = (
        peaks["confidence_tier"].value_counts().to_dict()
        if "confidence_tier" in peaks.columns
        else {}
    )
    group_frames = groups["frame_count"] if "frame_count" in groups.columns else pd.Series(dtype=float)
    top_truncated = False
    if "top_peaks_truncated" in source.columns:
        top_truncated = bool(pd.to_numeric(source["top_peaks_truncated"], errors="coerce").fillna(0).max())
    return {
        "suite": str(suite),
        "per_peak_dir": str(folder),
        "used_candidates": int(len(peaks)),
        "all_generated_candidates": int(len(source)),
        "tier_a_generated": int(tier_counts.get("A", 0)),
        "tier_b_generated": int(tier_counts.get("B", 0)),
        "tier_c_generated": int(tier_counts.get("C", len(tier_c))),
        "tier_a_used": int(used_tier_counts.get("A", 0)),
        "tier_b_used": int(used_tier_counts.get("B", 0)),
        "tier_c_used": int(used_tier_counts.get("C", 0)),
        "peak_groups": int(len(groups)),
        "groups_in_1plus_frames": int((group_frames >= 1).sum()),
        "groups_in_2plus_frames": int((group_frames >= 2).sum()),
        "groups_in_3plus_frames": int((group_frames >= 3).sum()),
        "groups_in_5plus_frames": int((group_frames >= 5).sum()),
        "top_peaks_truncation_occurred": int(top_truncated),
    }


def discover_labels(suites: list[Path]) -> list[str]:
    labels = ["10.4GPa", "2.4GPa_decomp"]
    first = per_peak_dir(suites[-1])
    peaks = read_csv(first / "peak_table.csv")
    if "frame" in peaks.columns:
        seen = list(dict.fromkeys(peaks["frame"].astype(str).tolist()))
        for index in [0, len(seen) // 2, len(seen) - 1]:
            if 0 <= index < len(seen) and seen[index] not in labels:
                labels.append(seen[index])
    return labels[:5]


def review_path(suite: Path, label: str) -> Path:
    return per_peak_dir(suite) / "xy_peak_review" / f"{safe_name(label)}_xy_peak_review.png"


def write_review_comparison(out_dir: Path, suites: list[Path], labels: list[str]) -> None:
    for label in labels:
        images = [(suite, review_path(suite, label)) for suite in suites]
        existing = [(suite, path) for suite, path in images if path.exists()]
        if not existing:
            continue
        fig, axes = plt.subplots(
            len(existing),
            1,
            figsize=(12, 4.0 * len(existing)),
            squeeze=False,
        )
        for ax, (suite, path) in zip(axes[:, 0], existing):
            ax.imshow(plt.imread(path))
            ax.set_title(f"{suite.name} | {label}")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe_name(label)}_review_comparison.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suites = [suite for suite in args.suites if suite.exists()]
    rows = [suite_metrics(suite) for suite in suites]
    with (args.out_dir / "suite_comparison_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["suite"])
        writer.writeheader()
        writer.writerows(rows)
    labels = discover_labels(suites) if suites else []
    write_review_comparison(args.out_dir, suites, labels)
    with (args.out_dir / "README.txt").open("w") as handle:
        handle.write("Peak suite comparison\n")
        handle.write("=====================\n\n")
        handle.write("This folder compares candidate counts and review plots across suites.\n")
        handle.write("Metrics: suite_comparison_metrics.csv\n")
        handle.write("Review plot comparisons: *_review_comparison.png\n")
    print(f"Wrote comparison outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
