#!/usr/bin/env python3
"""Validate and manifest the Log²-denoised UOTe correlation result suite.

The computation pipelines are intentionally not imported.  This program is an
independent delivery gate for one already-produced Log² root and the previous
formal, untransformed package used as the location oracle.

It verifies:

* the exact powder/single-crystal/four-category science hierarchy;
* the expected primary matrix and heatmap counts;
* score ranges in every primary CSV;
* strict-lower-triangle window presentation;
* one-to-one CSV/PNG pairing;
* exclusion of supplementary one-minus-ACF diagnostic renderings;
* byte and numerical equality of powder location matrices to the baseline;
  single-crystal all-peak location is validated by its own Cartesian contract.

With ``--output-dir`` it also writes a SHA256 package index, a JSON validation
report, and a completion marker.  ``--dry-run`` performs every validation but
writes nothing.  ``--self-test`` builds small synthetic suites in a temporary
directory and exercises both passing and deliberately failing cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SCIENCE_SAMPLES = ("powder", "single_crystal")
CATEGORIES = (
    "location",
    "roi_area",
    "window_to_window_across_frames",
    "window_to_window_within_same_frame",
)
INTENSITY_CATEGORIES = (
    "roi_area",
    "window_to_window_across_frames",
    "window_to_window_within_same_frame",
)
BLANK_TOKENS = {"", "nan", "na", "null", "none"}


@dataclass(frozen=True)
class ExpectedCounts:
    """Expected primary matrix/heatmap counts for one transform suite."""

    peak_maps: dict[tuple[str, str], int]
    across: dict[str, int]
    within: dict[str, int]


FORMAL_EXPECTATIONS = ExpectedCounts(
    peak_maps={
        ("powder", "roi_area"): 280,
        ("powder", "location"): 280,
        ("single_crystal", "roi_area"): 275,
        ("single_crystal", "location"): 275,
    },
    # Powder has 84 spots + 84 fit-control primary across maps.  The separate
    # 28 one-minus-ACF renderings are supplementary diagnostics, not a fourth
    # correlation family.  Single-crystal contributes 57 primary maps.
    across={"powder": 168, "single_crystal": 57},
    within={"powder": 40, "single_crystal": 12},
)


@dataclass
class CsvAudit:
    path: str
    numeric_count: int
    blank_count: int
    invalid_count: int
    out_of_range_count: int
    minimum: float | None
    maximum: float | None
    errors: list[str]


@dataclass
class TriangleAudit:
    path: str
    dimension: int
    finite_lower_count: int
    shape_ok: bool
    labels_match: bool
    upper_or_diagonal_value_count: int
    missing_lower_count: int
    invalid_lower_count: int
    out_of_range_count: int
    passed: bool
    errors: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_blank(raw: str) -> bool:
    return raw.strip().lower() in BLANK_TOKENS


def read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def peak_data_start_column(sample: str) -> int:
    # Powder anchor maps have pressure and peak-count metadata columns.
    # Single-crystal all-peak rectangular maps have only the row-label column.
    return 2 if sample == "powder" else 1


def audit_numeric_csv(
    path: Path,
    *,
    start_column: int,
    lower: float,
    upper: float,
    tolerance: float,
) -> CsvAudit:
    rows = read_csv(path)
    errors: list[str] = []
    if not rows:
        return CsvAudit(
            str(path), 0, 0, 1, 0, None, None, ["empty CSV"]
        )

    numeric: list[float] = []
    blanks = 0
    invalid = 0
    out_of_range = 0
    expected_width = len(rows[0])
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_width:
            errors.append(
                f"row {row_number} has {len(row)} columns; expected {expected_width}"
            )
        padded = row + [""] * max(0, expected_width - len(row))
        for column_number, raw in enumerate(
            padded[start_column:expected_width], start=start_column + 1
        ):
            if is_blank(raw):
                blanks += 1
                continue
            try:
                value = float(raw)
            except ValueError:
                invalid += 1
                errors.append(
                    f"row {row_number}, column {column_number}: nonnumeric {raw!r}"
                )
                continue
            if not math.isfinite(value):
                invalid += 1
                errors.append(
                    f"row {row_number}, column {column_number}: nonfinite {raw!r}"
                )
                continue
            numeric.append(value)
            if value < lower - tolerance or value > upper + tolerance:
                out_of_range += 1
                errors.append(
                    f"row {row_number}, column {column_number}: {value} outside "
                    f"[{lower}, {upper}]"
                )
    return CsvAudit(
        path=str(path),
        numeric_count=len(numeric),
        blank_count=blanks,
        invalid_count=invalid,
        out_of_range_count=out_of_range,
        minimum=min(numeric) if numeric else None,
        maximum=max(numeric) if numeric else None,
        errors=errors,
    )


def audit_strict_lower_triangle(
    path: Path,
    *,
    lower: float,
    upper: float,
    tolerance: float,
) -> TriangleAudit:
    rows = read_csv(path)
    errors: list[str] = []
    if not rows:
        return TriangleAudit(
            str(path), 0, 0, False, False, 0, 0, 1, 0, False, ["empty CSV"]
        )

    column_labels = [item.strip() for item in rows[0][1:]]
    dimension = len(column_labels)
    data_rows = rows[1:]
    shape_ok = len(data_rows) == dimension and all(
        len(row) == dimension + 1 for row in data_rows
    )
    if not shape_ok:
        errors.append(
            f"matrix is not square: header dimension={dimension}, "
            f"data rows={len(data_rows)}"
        )

    row_labels = [row[0].strip() if row else "" for row in data_rows]
    labels_match = len(row_labels) == dimension and row_labels == column_labels
    if not labels_match:
        errors.append("row labels do not exactly match column labels")

    upper_or_diagonal_values = 0
    missing_lower = 0
    invalid_lower = 0
    out_of_range = 0
    finite_lower = 0
    for row_index in range(min(dimension, len(data_rows))):
        row = data_rows[row_index]
        values = row[1:] + [""] * max(0, dimension - max(0, len(row) - 1))
        for column_index, raw in enumerate(values[:dimension]):
            if column_index >= row_index:
                if not is_blank(raw):
                    upper_or_diagonal_values += 1
                    if upper_or_diagonal_values <= 10:
                        errors.append(
                            f"nonblank diagonal/upper cell ({row_index}, {column_index})"
                        )
                continue
            if is_blank(raw):
                missing_lower += 1
                if missing_lower <= 10:
                    errors.append(
                        f"blank strict-lower cell ({row_index}, {column_index})"
                    )
                continue
            try:
                value = float(raw)
            except ValueError:
                invalid_lower += 1
                if invalid_lower <= 10:
                    errors.append(
                        f"nonnumeric strict-lower cell ({row_index}, {column_index})"
                    )
                continue
            if not math.isfinite(value):
                invalid_lower += 1
                if invalid_lower <= 10:
                    errors.append(
                        f"nonfinite strict-lower cell ({row_index}, {column_index})"
                    )
                continue
            finite_lower += 1
            if value < lower - tolerance or value > upper + tolerance:
                out_of_range += 1
                if out_of_range <= 10:
                    errors.append(
                        f"strict-lower cell ({row_index}, {column_index})={value} "
                        f"outside [{lower}, {upper}]"
                    )

    passed = bool(
        shape_ok
        and labels_match
        and upper_or_diagonal_values == 0
        and missing_lower == 0
        and invalid_lower == 0
        and out_of_range == 0
        and finite_lower == dimension * (dimension - 1) // 2
    )
    return TriangleAudit(
        path=str(path),
        dimension=dimension,
        finite_lower_count=finite_lower,
        shape_ok=shape_ok,
        labels_match=labels_match,
        upper_or_diagonal_value_count=upper_or_diagonal_values,
        missing_lower_count=missing_lower,
        invalid_lower_count=invalid_lower,
        out_of_range_count=out_of_range,
        passed=passed,
        errors=errors,
    )


def immediate_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        item for item in directory.iterdir() if item.is_file() and item.suffix == suffix
    )


def collect_peak_files(
    root: Path, sample: str, category: str
) -> tuple[list[Path], list[Path]]:
    category_root = root / sample / category
    return (
        immediate_files(category_root / "matrices", ".csv"),
        immediate_files(category_root / "heatmaps", ".png"),
    )


def collect_across_files(root: Path, sample: str) -> tuple[list[Path], list[Path]]:
    category_root = root / sample / "window_to_window_across_frames"
    csvs = sorted(
        path
        for path in category_root.rglob("*.csv")
        if "matrices" in path.parts
        and "_audit_full_symmetric" not in path.parts
        and "one_minus_similarity_diagnostics" not in path.parts
    ) if category_root.is_dir() else []
    pngs = sorted(
        path
        for path in category_root.rglob("*.png")
        if "heatmaps" in path.parts
        and "_audit_full_symmetric" not in path.parts
        and "one_minus_similarity_diagnostics" not in path.parts
    ) if category_root.is_dir() else []
    return csvs, pngs


def collect_unexpected_across_diagnostics(root: Path, sample: str) -> list[Path]:
    """Return supplementary 1-r files accidentally placed in a core suite."""

    category_root = root / sample / "window_to_window_across_frames"
    if not category_root.is_dir():
        return []
    return sorted(
        path
        for path in category_root.rglob("*")
        if path.is_file() and "one_minus_similarity_diagnostics" in path.parts
    )


def collect_within_files(root: Path, sample: str) -> tuple[list[Path], list[Path]]:
    category_root = root / sample / "window_to_window_within_same_frame"
    if not category_root.is_dir():
        return [], []
    csvs = sorted(
        path
        for path in category_root.rglob("*.csv")
        if "_audit_full_symmetric" not in path.parts
        and (
            path.name == "matrix.csv"
            or "by_pressure/matrices" in path.as_posix()
        )
    )
    pngs = sorted(
        path
        for path in category_root.rglob("*.png")
        if "_audit_full_symmetric" not in path.parts
        and (
            path.name == "heatmap.png"
            or "by_pressure/heatmaps" in path.as_posix()
        )
    )
    return csvs, pngs


def expected_png_for_csv(path: Path) -> Path:
    if path.name == "matrix.csv":
        return path.with_name("heatmap.png")
    parts = list(path.parts)
    try:
        matrix_index = len(parts) - 1 - parts[::-1].index("matrices")
    except ValueError:
        return path.with_suffix(".png")
    parts[matrix_index] = "heatmaps"
    return Path(*parts).with_suffix(".png")


def expected_csv_for_png(path: Path) -> Path:
    if path.name == "heatmap.png":
        return path.with_name("matrix.csv")
    parts = list(path.parts)
    try:
        heatmap_index = len(parts) - 1 - parts[::-1].index("heatmaps")
    except ValueError:
        return path.with_suffix(".csv")
    parts[heatmap_index] = "matrices"
    return Path(*parts).with_suffix(".csv")


def numeric_tokens(path: Path, *, start_column: int) -> list[float | None | str]:
    rows = read_csv(path)
    tokens: list[float | None | str] = []
    if not rows:
        return tokens
    width = len(rows[0])
    for row in rows[1:]:
        values = row + [""] * max(0, width - len(row))
        for raw in values[start_column:width]:
            if is_blank(raw):
                tokens.append(None)
                continue
            try:
                value = float(raw)
            except ValueError:
                tokens.append(f"INVALID:{raw}")
                continue
            tokens.append(value if math.isfinite(value) else f"NONFINITE:{raw}")
    return tokens


def tokens_numerically_equal(
    left: Sequence[float | None | str],
    right: Sequence[float | None | str],
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> bool:
    if len(left) != len(right):
        return False
    for left_value, right_value in zip(left, right):
        if left_value is None or right_value is None:
            if left_value is not right_value:
                return False
            continue
        if isinstance(left_value, str) or isinstance(right_value, str):
            if left_value != right_value:
                return False
            continue
        if not math.isclose(
            left_value,
            right_value,
            rel_tol=rel_tolerance,
            abs_tol=abs_tolerance,
        ):
            return False
    return True


def count_token_differences(
    left: Sequence[float | None | str],
    right: Sequence[float | None | str],
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    differences = 0
    for left_value, right_value in zip(left, right):
        if left_value is None or right_value is None:
            differences += int(left_value is not right_value)
        elif isinstance(left_value, str) or isinstance(right_value, str):
            differences += int(left_value != right_value)
        elif not math.isclose(
            left_value,
            right_value,
            rel_tol=rel_tolerance,
            abs_tol=abs_tolerance,
        ):
            differences += 1
    return differences


def core_files(
    root: Path, sample: str, category: str
) -> tuple[list[Path], list[Path]]:
    if category in {"roi_area", "location"}:
        return collect_peak_files(root, sample, category)
    if category == "window_to_window_across_frames":
        return collect_across_files(root, sample)
    if category == "window_to_window_within_same_frame":
        return collect_within_files(root, sample)
    raise ValueError(category)


def core_start_column(sample: str, category: str) -> int:
    if category in {"roi_area", "location"}:
        return peak_data_start_column(sample)
    return 1


def check_exact_hierarchy(root: Path) -> tuple[bool, dict[str, object], list[str]]:
    errors: list[str] = []
    if not root.is_dir():
        return False, {}, [f"result root is not a directory: {root}"]
    science_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    if science_dirs != sorted(SCIENCE_SAMPLES):
        errors.append(
            f"science directories are {science_dirs}; expected {sorted(SCIENCE_SAMPLES)}"
        )
    category_dirs: dict[str, list[str]] = {}
    for sample in SCIENCE_SAMPLES:
        sample_root = root / sample
        if not sample_root.is_dir():
            category_dirs[sample] = []
            errors.append(f"missing sample directory: {sample_root}")
            continue
        found = sorted(path.name for path in sample_root.iterdir() if path.is_dir())
        category_dirs[sample] = found
        if found != sorted(CATEGORIES):
            errors.append(
                f"{sample} categories are {found}; expected {sorted(CATEGORIES)}"
            )
    return not errors, {
        "science_directories": science_dirs,
        "category_directories": category_dirs,
    }, errors


def validate_one_suite(
    root: Path,
    *,
    label: str,
    expected: ExpectedCounts,
    tolerance: float,
) -> dict[str, object]:
    errors: list[str] = []
    hierarchy_ok, hierarchy, hierarchy_errors = check_exact_hierarchy(root)
    errors.extend(hierarchy_errors)
    counts: dict[str, dict[str, int]] = {}
    pairing: dict[str, dict[str, object]] = {}
    numeric_audits: list[dict[str, object]] = []
    triangle_audits: list[dict[str, object]] = []
    expected_counts_ok = True
    unexpected_diagnostics: dict[str, list[str]] = {}

    for sample in SCIENCE_SAMPLES:
        diagnostic_files = collect_unexpected_across_diagnostics(root, sample)
        unexpected_diagnostics[sample] = [
            str(path.relative_to(root)) for path in diagnostic_files
        ]
        if diagnostic_files:
            errors.append(
                f"{sample}/window_to_window_across_frames contains "
                f"{len(diagnostic_files)} supplementary one-minus-ACF diagnostic files"
            )

    for sample in SCIENCE_SAMPLES:
        for category in CATEGORIES:
            csvs, pngs = core_files(root, sample, category)
            key = f"{sample}/{category}"
            counts[key] = {"csv": len(csvs), "png": len(pngs)}
            if category in {"roi_area", "location"}:
                expected_count = expected.peak_maps[(sample, category)]
            elif category == "window_to_window_across_frames":
                expected_count = expected.across[sample]
            else:
                expected_count = expected.within[sample]
            if len(csvs) != expected_count or len(pngs) != expected_count:
                expected_counts_ok = False
                errors.append(
                    f"{key}: got {len(csvs)} CSV/{len(pngs)} PNG; "
                    f"expected {expected_count}/{expected_count}"
                )

            missing_pngs = [
                str(expected_png_for_csv(path).relative_to(root))
                for path in csvs
                if not expected_png_for_csv(path).is_file()
            ]
            orphan_pngs = [
                str(path.relative_to(root))
                for path in pngs
                if not expected_csv_for_png(path).is_file()
            ]
            pairing[key] = {
                "paired": not missing_pngs and not orphan_pngs,
                "missing_png_count": len(missing_pngs),
                "orphan_png_count": len(orphan_pngs),
                "missing_png_examples": missing_pngs[:10],
                "orphan_png_examples": orphan_pngs[:10],
            }
            if missing_pngs or orphan_pngs:
                errors.append(
                    f"{key}: {len(missing_pngs)} matrices lack PNGs and "
                    f"{len(orphan_pngs)} PNGs lack matrices"
                )

            if category in {"roi_area", "location"}:
                for path in csvs:
                    audit = audit_numeric_csv(
                        path,
                        start_column=core_start_column(sample, category),
                        lower=0.0,
                        upper=1.0,
                        tolerance=tolerance,
                    )
                    numeric_audits.append(audit.__dict__)
                    if audit.invalid_count or audit.out_of_range_count or audit.errors:
                        errors.append(
                            f"{path.relative_to(root)} failed numeric range/format audit"
                        )
            else:
                for path in csvs:
                    audit = audit_strict_lower_triangle(
                        path,
                        lower=-1.0,
                        upper=1.0,
                        tolerance=tolerance,
                    )
                    triangle_audits.append(audit.__dict__)
                    if not audit.passed:
                        errors.append(
                            f"{path.relative_to(root)} failed strict-lower audit"
                        )

    all_pairing_ok = all(item["paired"] for item in pairing.values())
    all_numeric_ok = all(
        item["invalid_count"] == 0
        and item["out_of_range_count"] == 0
        and not item["errors"]
        for item in numeric_audits
    )
    all_triangles_ok = all(item["passed"] for item in triangle_audits)
    total_windows_expected = sum(expected.across.values()) + sum(expected.within.values())
    counts_ok = all(
        item["csv"] == item["png"]
        for item in counts.values()
    ) and len(triangle_audits) == total_windows_expected
    checks = {
        "exact_hierarchy": hierarchy_ok,
        "expected_primary_counts": counts_ok and expected_counts_ok,
        "all_csv_png_pairs_one_to_one": all_pairing_ok,
        "all_roi_location_scores_in_0_1": all_numeric_ok,
        "all_window_matrices_strict_lower_and_in_range": all_triangles_ok,
        "window_matrix_count_matches_expected": len(triangle_audits)
        == total_windows_expected,
        "no_supplementary_one_minus_acf_diagnostics": not any(
            unexpected_diagnostics.values()
        ),
    }
    return {
        "label": label,
        "root": str(root),
        "status": "PASS" if all(checks.values()) and not errors else "FAIL",
        "checks": checks,
        "hierarchy": hierarchy,
        "counts": counts,
        "numeric_audit_summary": {
            "files": len(numeric_audits),
            "invalid_values": sum(item["invalid_count"] for item in numeric_audits),
            "out_of_range_values": sum(
                item["out_of_range_count"] for item in numeric_audits
            ),
        },
        "triangle_audit_summary": {
            "files": len(triangle_audits),
            "failures": sum(not item["passed"] for item in triangle_audits),
            "finite_lower_values": sum(
                item["finite_lower_count"] for item in triangle_audits
            ),
            "expected_formal_finite_lower_values": (
                28728 + 3135 + 15120 + 2052
                if expected == FORMAL_EXPECTATIONS
                else None
            ),
        },
        "pairing": pairing,
        "unexpected_one_minus_acf_diagnostics": {
            sample: {
                "file_count": len(paths),
                "examples": paths[:20],
            }
            for sample, paths in unexpected_diagnostics.items()
        },
        "errors": errors,
        "error_count": len(errors),
        "numeric_audit_failures": [
            item
            for item in numeric_audits
            if item["invalid_count"]
            or item["out_of_range_count"]
            or item["errors"]
        ][:50],
        "triangle_audit_failures": [
            item for item in triangle_audits if not item["passed"]
        ][:50],
    }


def validate_location_against_baseline(
    suite_root: Path,
    baseline_root: Path,
    *,
    label: str,
    abs_tolerance: float,
    rel_tolerance: float,
    samples: Sequence[str] = SCIENCE_SAMPLES,
) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    checked = 0
    expected_total = 0
    for sample in samples:
        suite_csvs, _ = collect_peak_files(suite_root, sample, "location")
        baseline_csvs, _ = collect_peak_files(baseline_root, sample, "location")
        suite_category = suite_root / sample / "location"
        baseline_category = baseline_root / sample / "location"
        suite_by_relative = {
            path.relative_to(suite_category).as_posix(): path for path in suite_csvs
        }
        baseline_by_relative = {
            path.relative_to(baseline_category).as_posix(): path
            for path in baseline_csvs
        }
        expected_total += len(baseline_by_relative)
        if set(suite_by_relative) != set(baseline_by_relative):
            mismatches.append(
                {
                    "sample": sample,
                    "kind": "relative_file_set",
                    "suite_only": sorted(
                        set(suite_by_relative) - set(baseline_by_relative)
                    )[:20],
                    "baseline_only": sorted(
                        set(baseline_by_relative) - set(suite_by_relative)
                    )[:20],
                }
            )
        for relative in sorted(set(suite_by_relative) & set(baseline_by_relative)):
            checked += 1
            suite_path = suite_by_relative[relative]
            baseline_path = baseline_by_relative[relative]
            hashes_equal = sha256_file(suite_path) == sha256_file(baseline_path)
            suite_tokens = numeric_tokens(
                suite_path, start_column=peak_data_start_column(sample)
            )
            baseline_tokens = numeric_tokens(
                baseline_path, start_column=peak_data_start_column(sample)
            )
            numerics_equal = tokens_numerically_equal(
                suite_tokens,
                baseline_tokens,
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            )
            if not hashes_equal or not numerics_equal:
                mismatches.append(
                    {
                        "sample": sample,
                        "relative_path": relative,
                        "kind": "content",
                        "hashes_equal": hashes_equal,
                        "numerics_equal": numerics_equal,
                    }
                )
    return {
        "label": label,
        "checked_files": checked,
        "expected_files": expected_total,
        "all_relative_files_present": checked == expected_total,
        "all_hashes_and_numerics_equal": not mismatches and checked == expected_total,
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:50],
    }


def build_package_index(roots: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, root in roots.items():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == ".DS_Store":
                continue
            relative = path.relative_to(root)
            parts = relative.parts
            sample = parts[0] if parts and parts[0] in SCIENCE_SAMPLES else ""
            category = parts[1] if len(parts) > 1 and parts[1] in CATEGORIES else ""
            rows.append(
                {
                    "transform": label,
                    "sample": sample,
                    "category": category,
                    "relative_path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "absolute_path": str(path),
                }
            )
    return rows


def default_output_dir(log_root: Path) -> Path:
    return log_root.parent / "log_denoised_validation_package"


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def write_package(
    output_dir: Path,
    report: dict[str, object],
    index_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "VALIDATION_REPORT.json"
    index_path = output_dir / "PACKAGE_INDEX.csv"
    completion_path = output_dir / "RUN_COMPLETE.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = (
        "transform",
        "sample",
        "category",
        "relative_path",
        "bytes",
        "sha256",
        "absolute_path",
    )
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)
    completion = {
        "status": "complete" if report["status"] == "PASS" else "failed_validation",
        "created_at_utc": utc_now(),
        "validation_status": report["status"],
        "validation_report": report_path.name,
        "validation_report_sha256": sha256_file(report_path),
        "package_index": index_path.name,
        "package_index_rows": len(index_rows),
        "package_index_sha256": sha256_file(index_path),
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_validation(
    *,
    log_root: Path,
    baseline_root: Path,
    expected: ExpectedCounts,
    tolerance: float,
    comparison_abs_tolerance: float,
    comparison_rel_tolerance: float,
) -> dict[str, object]:
    log_report = validate_one_suite(
        log_root, label="log_square", expected=expected, tolerance=tolerance
    )
    log_location = validate_location_against_baseline(
        log_root,
        baseline_root,
        label="log_square_vs_baseline",
        abs_tolerance=comparison_abs_tolerance,
        rel_tolerance=comparison_rel_tolerance,
        samples=("powder",),
    )
    checks = {
        "log_suite_passes": log_report["status"] == "PASS",
        "log_location_hashes_and_numerics_equal_baseline": log_location[
            "all_hashes_and_numerics_equal"
        ],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "validated_at_utc": utc_now(),
        "roots": {
            "log_square": str(log_root),
            "baseline": str(baseline_root),
        },
        "formal_expected_counts_per_transform": {
            "powder_roi_area": expected.peak_maps[("powder", "roi_area")],
            "powder_location": expected.peak_maps[("powder", "location")],
            "single_crystal_roi_area": expected.peak_maps[
                ("single_crystal", "roi_area")
            ],
            "single_crystal_location": expected.peak_maps[
                ("single_crystal", "location")
            ],
            "powder_across": expected.across["powder"],
            "single_crystal_across": expected.across["single_crystal"],
            "powder_within": expected.within["powder"],
            "single_crystal_within": expected.within["single_crystal"],
            "total_window_matrices": sum(expected.across.values())
            + sum(expected.within.values()),
        },
        "checks": checks,
        "suites": {"log_square": log_report},
        "location_baseline_comparisons": {
            "log_square": log_location,
        },
    }


def write_table(path: Path, rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def write_png_placeholder(path: Path) -> None:
    # Pairing validation only requires a file.  This is not intended to be a
    # decodable image; it keeps the self-test dependency-free.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic-png-placeholder\n")


def synthetic_lower_matrix(path: Path, value: float, dimension: int = 3) -> None:
    labels = [f"x{index}" for index in range(dimension)]
    rows: list[list[object]] = [["row", *labels]]
    for row_index, label in enumerate(labels):
        row: list[object] = [label]
        for column_index in range(dimension):
            row.append(value + 0.01 * row_index + 0.001 * column_index if column_index < row_index else "")
        rows.append(row)
    write_table(path, rows)


def build_synthetic_suite(
    root: Path,
    *,
    roi_value: float,
    window_value: float,
    location_source: Path | None = None,
) -> None:
    for sample in SCIENCE_SAMPLES:
        for category in CATEGORIES:
            (root / sample / category).mkdir(parents=True, exist_ok=True)

        for category, value in (("roi_area", roi_value), ("location", 0.75)):
            matrix = root / sample / category / "matrices" / "map_000.csv"
            heatmap = root / sample / category / "heatmaps" / "map_000.png"
            if sample == "powder":
                rows = [
                    ["pressure_gpa", "peak_count_at_pressure", "peak 1"],
                    [3.5, 1, value],
                ]
            else:
                rows = [["row", "f0"], ["f0", value]]
            write_table(matrix, rows)
            write_png_placeholder(heatmap)

        across = (
            root
            / sample
            / "window_to_window_across_frames"
            / "spots"
            / "acf_strict"
            / "matrices"
            / "window_00_0_5.csv"
        )
        synthetic_lower_matrix(across, window_value)
        write_png_placeholder(expected_png_for_csv(across))

        within = (
            root
            / sample
            / "window_to_window_within_same_frame"
            / "spots"
            / "aggregate"
            / "matrix.csv"
        )
        synthetic_lower_matrix(within, window_value + 0.1)
        write_png_placeholder(expected_png_for_csv(within))

    if location_source is not None:
        for sample in SCIENCE_SAMPLES:
            source = location_source / sample / "location" / "matrices" / "map_000.csv"
            destination = root / sample / "location" / "matrices" / "map_000.csv"
            shutil.copy2(source, destination)


def run_self_test() -> None:
    synthetic_expected = ExpectedCounts(
        peak_maps={
            ("powder", "roi_area"): 1,
            ("powder", "location"): 1,
            ("single_crystal", "roi_area"): 1,
            ("single_crystal", "location"): 1,
        },
        across={"powder": 1, "single_crystal": 1},
        within={"powder": 1, "single_crystal": 1},
    )
    with tempfile.TemporaryDirectory(prefix="denoised-suite-validator-") as temporary:
        base = Path(temporary)
        baseline = base / "baseline"
        log_root = base / "log"
        build_synthetic_suite(baseline, roi_value=0.2, window_value=0.1)
        build_synthetic_suite(
            log_root,
            roi_value=0.3,
            window_value=0.2,
            location_source=baseline,
        )
        passing = run_validation(
            log_root=log_root,
            baseline_root=baseline,
            expected=synthetic_expected,
            tolerance=1e-12,
            comparison_abs_tolerance=1e-12,
            comparison_rel_tolerance=1e-10,
        )
        if passing["status"] != "PASS":
            raise AssertionError(
                "synthetic passing fixture failed:\n"
                + json.dumps(passing, indent=2, ensure_ascii=False)
            )

        # A supplementary 1-r rendering must not be accepted as part of the
        # compact across-frame science category.
        diagnostic = (
            log_root
            / "powder"
            / "window_to_window_across_frames"
            / "fit_control"
            / "acf_strict"
            / "one_minus_similarity_diagnostics"
            / "matrices"
            / "window_00_0_5.csv"
        )
        synthetic_lower_matrix(diagnostic, 0.4)
        write_png_placeholder(expected_png_for_csv(diagnostic))
        failing_diagnostic = run_validation(
            log_root=log_root,
            baseline_root=baseline,
            expected=synthetic_expected,
            tolerance=1e-12,
            comparison_abs_tolerance=1e-12,
            comparison_rel_tolerance=1e-10,
        )
        if (
            failing_diagnostic["status"] != "FAIL"
            or failing_diagnostic["suites"]["log_square"]["checks"][
                "no_supplementary_one_minus_acf_diagnostics"
            ]
        ):
            raise AssertionError("supplementary one-minus-ACF files were not rejected")
        shutil.rmtree(diagnostic.parents[1])

        # A nonblank diagonal must be rejected.
        bad_triangle = (
            log_root
            / "powder"
            / "window_to_window_across_frames"
            / "spots"
            / "acf_strict"
            / "matrices"
            / "window_00_0_5.csv"
        )
        rows = read_csv(bad_triangle)
        rows[1][1] = "1"
        write_table(bad_triangle, rows)
        failing_triangle = run_validation(
            log_root=log_root,
            baseline_root=baseline,
            expected=synthetic_expected,
            tolerance=1e-12,
            comparison_abs_tolerance=1e-12,
            comparison_rel_tolerance=1e-10,
        )
        if failing_triangle["status"] != "FAIL":
            raise AssertionError("nonblank diagonal was not rejected")
        synthetic_lower_matrix(bad_triangle, 0.5)

        # A changed powder location value must be rejected even if its shape is valid.
        bad_location = (
            log_root
            / "powder"
            / "location"
            / "matrices"
            / "map_000.csv"
        )
        rows = read_csv(bad_location)
        rows[1][1] = "0.5"
        write_table(bad_location, rows)
        failing_location = run_validation(
            log_root=log_root,
            baseline_root=baseline,
            expected=synthetic_expected,
            tolerance=1e-12,
            comparison_abs_tolerance=1e-12,
            comparison_rel_tolerance=1e-10,
        )
        if failing_location["status"] != "FAIL":
            raise AssertionError("changed location matrix was not rejected")

        # A score outside its scientific range must be rejected.
        baseline_location = (
            baseline
            / "powder"
            / "location"
            / "matrices"
            / "map_000.csv"
        )
        shutil.copy2(baseline_location, bad_location)
        bad_roi = log_root / "powder" / "roi_area" / "matrices" / "map_000.csv"
        rows = read_csv(bad_roi)
        rows[1][2] = "1.5"
        write_table(bad_roi, rows)
        failing_range = run_validation(
            log_root=log_root,
            baseline_root=baseline,
            expected=synthetic_expected,
            tolerance=1e-12,
            comparison_abs_tolerance=1e-12,
            comparison_rel_tolerance=1e-10,
        )
        if failing_range["status"] != "FAIL":
            raise AssertionError("out-of-range ROI score was not rejected")
        rows[1][2] = "0.6"
        write_table(bad_roi, rows)

        # Exercise the manifest writer on the known-good Log² suite.
        package_dir = base / "validation_package"
        package_index = build_package_index({"log_square": log_root})
        write_package(package_dir, passing, package_index)
        for expected_file in (
            "VALIDATION_REPORT.json",
            "PACKAGE_INDEX.csv",
            "RUN_COMPLETE.json",
        ):
            if not (package_dir / expected_file).is_file():
                raise AssertionError(f"package writer omitted {expected_file}")

    print(
        "SELF-TEST PASS: valid fixture/package accepted; triangle, location, "
        "range, and supplementary 1-r failures rejected"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and manifest the Log² UOTe suite."
    )
    parser.add_argument("--log-root", type=Path, help="log-square result root")
    parser.add_argument(
        "--baseline-root", type=Path, help="previous formal untransformed package root"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for VALIDATION_REPORT.json, PACKAGE_INDEX.csv, RUN_COMPLETE.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform all checks and print JSON, but write no package files",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run dependency-free synthetic passing/failing fixtures",
    )
    parser.add_argument(
        "--range-tolerance", type=float, default=1e-12
    )
    parser.add_argument(
        "--comparison-abs-tolerance", type=float, default=1e-12
    )
    parser.add_argument(
        "--comparison-rel-tolerance", type=float, default=1e-10
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    missing_arguments = [
        name
        for name, value in (
            ("--log-root", args.log_root),
            ("--baseline-root", args.baseline_root),
        )
        if value is None
    ]
    if missing_arguments:
        raise SystemExit(
            "required unless --self-test: " + ", ".join(missing_arguments)
        )
    log_root = args.log_root.resolve()
    baseline_root = args.baseline_root.resolve()
    report = run_validation(
        log_root=log_root,
        baseline_root=baseline_root,
        expected=FORMAL_EXPECTATIONS,
        tolerance=args.range_tolerance,
        comparison_abs_tolerance=args.comparison_abs_tolerance,
        comparison_rel_tolerance=args.comparison_rel_tolerance,
    )
    if not args.dry_run:
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else default_output_dir(log_root)
        )
        if is_within(output_dir, log_root):
            raise SystemExit("--output-dir must be outside the science result root")
        index_rows = build_package_index({"log_square": log_root})
        write_package(output_dir, report, index_rows)
        print(f"Validation package written to: {output_dir}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
