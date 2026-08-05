#!/usr/bin/env python3
"""Independent final audit for the organized UOTe correlation package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_CATEGORIES = sorted(
    (
        "roi_area",
        "location",
        "window_to_window_across_frames",
        "window_to_window_within_same_frame",
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_lower_audit(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    column_labels = rows[0][1:]
    data_rows = rows[1:]
    dimension = len(column_labels)
    shape_ok = len(data_rows) == dimension and all(
        len(row[1:]) == dimension for row in data_rows
    )
    upper_or_diagonal_values: list[tuple[int, int, str]] = []
    missing_lower: list[tuple[int, int]] = []
    invalid_lower: list[tuple[int, int, str]] = []
    finite_lower = 0
    if shape_ok:
        for row_index, row in enumerate(data_rows):
            for column_index, raw_value in enumerate(row[1:]):
                value = raw_value.strip()
                if column_index >= row_index:
                    if value and value.lower() not in {"nan", "na", "null"}:
                        upper_or_diagonal_values.append(
                            (row_index, column_index, value)
                        )
                    continue
                if not value or value.lower() in {"nan", "na", "null"}:
                    missing_lower.append((row_index, column_index))
                    continue
                try:
                    numeric = float(value)
                except ValueError:
                    invalid_lower.append((row_index, column_index, value))
                    continue
                if not math.isfinite(numeric):
                    invalid_lower.append((row_index, column_index, value))
                else:
                    finite_lower += 1
    return {
        "path": str(path),
        "dimension": dimension,
        "shape_ok": shape_ok,
        "upper_or_diagonal_value_count": len(upper_or_diagonal_values),
        "missing_lower_count": len(missing_lower),
        "invalid_lower_count": len(invalid_lower),
        "finite_lower_count": finite_lower,
        "passed": bool(
            shape_ok
            and not upper_or_diagonal_values
            and not missing_lower
            and not invalid_lower
        ),
    }


def find_window_matrices(root: Path) -> tuple[list[Path], list[Path]]:
    across: list[Path] = []
    within: list[Path] = []
    for sample in ("powder", "single_crystal"):
        across_root = root / sample / "window_to_window_across_frames"
        across.extend(
            path
            for path in across_root.rglob("*.csv")
            if "matrices" in path.parts
            and "_audit_full_symmetric" not in path.parts
        )
        within_root = root / sample / "window_to_window_within_same_frame"
        within.extend(
            path
            for path in within_root.rglob("*.csv")
            if "_audit_full_symmetric" not in path.parts
            and (
                path.name == "matrix.csv"
                or "by_pressure/matrices" in path.as_posix()
            )
        )
    return sorted(across), sorted(within)


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for item in path.glob(pattern) if item.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(
            "/Users/stanley/x-ray/correlations/results/"
            "uote_pressure_level_peak_spots_absolute_anchor_iou_"
            "integer_window_suite_20260730_v8/peak_maps/"
            "complete_correlation_results_by_sample"
        ),
    )
    return parser.parse_args()


def main() -> None:
    root = parse_args().root.resolve()
    with (root / "PACKAGE_INDEX.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        index_rows = list(csv.DictReader(handle))

    hash_failures: list[dict[str, str]] = []
    missing_files: list[str] = []
    for row in index_rows:
        destination = root / row["destination_path"]
        source = Path(row["source_path"])
        if not destination.is_file() or not source.is_file():
            missing_files.append(row["destination_path"])
            continue
        destination_sha = sha256_file(destination)
        source_sha = sha256_file(source)
        if (
            destination_sha != row["copied_sha256"]
            or source_sha != row["source_sha256"]
            or destination_sha != source_sha
        ):
            hash_failures.append(
                {
                    "destination": row["destination_path"],
                    "expected_destination_sha256": row["copied_sha256"],
                    "actual_destination_sha256": destination_sha,
                    "expected_source_sha256": row["source_sha256"],
                    "actual_source_sha256": source_sha,
                }
            )

    across_matrices, within_matrices = find_window_matrices(root)
    triangle_results = [
        strict_lower_audit(path)
        for path in across_matrices + within_matrices
    ]
    triangle_failures = [
        result for result in triangle_results if not result["passed"]
    ]

    science_dirs = sorted(
        path.name for path in root.iterdir() if path.is_dir()
    )
    category_dirs = {
        sample: sorted(
            path.name
            for path in (root / sample).iterdir()
            if path.is_dir()
        )
        for sample in ("powder", "single_crystal")
    }
    ds_store_paths = sorted(
        str(path.relative_to(root))
        for path in root.rglob(".DS_Store")
        if path.is_file()
    )
    counts = {
        "package_index_rows": len(index_rows),
        "window_across_strict_lower_matrices": len(across_matrices),
        "window_within_strict_lower_matrices": len(within_matrices),
        "window_total_strict_lower_matrices": len(triangle_results),
        "powder_roi_heatmaps": count_files(
            root / "powder" / "roi_area" / "heatmaps", "*.png"
        ),
        "powder_roi_matrices": count_files(
            root / "powder" / "roi_area" / "matrices", "*.csv"
        ),
        "powder_location_heatmaps": count_files(
            root / "powder" / "location" / "heatmaps", "*.png"
        ),
        "powder_location_matrices": count_files(
            root / "powder" / "location" / "matrices", "*.csv"
        ),
        "single_primary_roi_heatmaps": count_files(
            root / "single_crystal" / "roi_area" / "heatmaps", "*.png"
        ),
        "single_primary_roi_matrices": count_files(
            root / "single_crystal" / "roi_area" / "matrices", "*.csv"
        ),
        "single_primary_location_heatmaps": count_files(
            root / "single_crystal" / "location" / "heatmaps", "*.png"
        ),
        "single_primary_location_matrices": count_files(
            root / "single_crystal" / "location" / "matrices", "*.csv"
        ),
    }
    checks = {
        "exact_two_science_directories": science_dirs
        == ["powder", "single_crystal"],
        "powder_exact_four_categories": category_dirs["powder"]
        == REQUIRED_CATEGORIES,
        "single_crystal_exact_four_categories": category_dirs[
            "single_crystal"
        ]
        == REQUIRED_CATEGORIES,
        "package_index_has_5734_copied_artifacts": len(index_rows) == 5734,
        "all_indexed_sources_and_destinations_exist": not missing_files,
        "all_5734_source_and_destination_hashes_match": not hash_failures,
        "all_305_window_matrices_audited": len(triangle_results) == 305,
        "across_window_matrix_count_253": len(across_matrices) == 253,
        "within_window_matrix_count_52": len(within_matrices) == 52,
        "all_window_matrices_strict_lower": not triangle_failures,
        "powder_roi_counts_280_plus_280": counts["powder_roi_heatmaps"]
        == counts["powder_roi_matrices"]
        == 280,
        "powder_location_counts_280_plus_280": counts[
            "powder_location_heatmaps"
        ]
        == counts["powder_location_matrices"]
        == 280,
        "single_primary_roi_counts_75_plus_75": counts[
            "single_primary_roi_heatmaps"
        ]
        == counts["single_primary_roi_matrices"]
        == 75,
        "single_primary_location_counts_75_plus_75": counts[
            "single_primary_location_heatmaps"
        ]
        == counts["single_primary_location_matrices"]
        == 75,
        "macos_metadata_excluded_from_science_index": not any(
            row["destination_path"].endswith(".DS_Store") for row in index_rows
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "checks": checks,
        "counts": counts,
        "science_directories": science_dirs,
        "category_directories": category_dirs,
        "missing_file_count": len(missing_files),
        "missing_file_examples": missing_files[:20],
        "hash_failure_count": len(hash_failures),
        "hash_failure_examples": hash_failures[:20],
        "strict_lower_failure_count": len(triangle_failures),
        "strict_lower_failure_examples": triangle_failures[:20],
        "ignored_macos_ds_store_count": len(ds_store_paths),
        "ignored_macos_ds_store_paths": ds_store_paths,
    }
    (root / "FINAL_AUDIT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_complete_path = root / "RUN_COMPLETE.json"
    run_complete = json.loads(run_complete_path.read_text(encoding="utf-8"))
    run_complete["final_audit"] = "FINAL_AUDIT.json"
    run_complete["final_audit_status"] = report["status"]
    run_complete_path.write_text(
        json.dumps(run_complete, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
