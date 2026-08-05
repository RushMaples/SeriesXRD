#!/usr/bin/env python3
"""Compare two uniform-correlation-v2 runs for numeric reproducibility.

The comparison covers every public scientific CSV and NPZ under ``spots`` and
``fit``, plus the input inventory and robustness tables.  Text/labels must be
identical, missing-value masks must be identical, and every finite numeric
value must agree within the requested absolute tolerance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_TOLERANCE = 1.0e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _scientific_files(root: Path) -> set[Path]:
    files: set[Path] = set()
    for channel in ("spots", "fit"):
        channel_root = root / channel
        if channel_root.is_dir():
            files.update(path.relative_to(root) for path in channel_root.rglob("*.csv"))
            files.update(path.relative_to(root) for path in channel_root.rglob("*.npz"))
    for relative in (Path("input_inventory.csv"), Path("robustness/scan_level_metrics.csv")):
        if (root / relative).is_file():
            files.add(relative)
    return files


def _numeric(text: str) -> float | None:
    value = text.strip()
    if value.lower() in {"", "nan", "na", "n/a", "null", "none"}:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return None


def _compare_csv(first: Path, second: Path, tolerance: float) -> dict[str, Any]:
    with first.open(newline="", encoding="utf-8-sig") as handle:
        left = list(csv.reader(handle))
    with second.open(newline="", encoding="utf-8-sig") as handle:
        right = list(csv.reader(handle))
    if len(left) != len(right):
        return {"passed": False, "reason": "row_count", "left": len(left), "right": len(right)}
    maximum = 0.0
    numeric_cells = 0
    for row_index, (left_row, right_row) in enumerate(zip(left, right, strict=True), start=1):
        if len(left_row) != len(right_row):
            return {
                "passed": False,
                "reason": "column_count",
                "row": row_index,
                "left": len(left_row),
                "right": len(right_row),
            }
        for column_index, (left_cell, right_cell) in enumerate(
            zip(left_row, right_row, strict=True), start=1
        ):
            left_number = _numeric(left_cell)
            right_number = _numeric(right_cell)
            if left_number is not None and right_number is not None:
                numeric_cells += 1
                left_finite = math.isfinite(left_number)
                right_finite = math.isfinite(right_number)
                if left_finite != right_finite:
                    return {
                        "passed": False,
                        "reason": "finite_mask",
                        "row": row_index,
                        "column": column_index,
                        "left": left_cell,
                        "right": right_cell,
                    }
                if left_finite:
                    difference = abs(left_number - right_number)
                    maximum = max(maximum, difference)
                    if difference > tolerance:
                        return {
                            "passed": False,
                            "reason": "numeric_difference",
                            "row": row_index,
                            "column": column_index,
                            "difference": difference,
                            "left": left_number,
                            "right": right_number,
                        }
            elif left_cell != right_cell:
                return {
                    "passed": False,
                    "reason": "text_difference",
                    "row": row_index,
                    "column": column_index,
                    "left": left_cell,
                    "right": right_cell,
                }
    return {
        "passed": True,
        "rows": len(left),
        "numeric_cells": numeric_cells,
        "max_abs_difference": maximum,
    }


def _compare_numeric_arrays(
    left: np.ndarray, right: np.ndarray, tolerance: float
) -> tuple[bool, float, str]:
    if left.shape != right.shape:
        return False, math.inf, "shape"
    if left.dtype.kind in "biufc" and right.dtype.kind in "biufc":
        left_values = np.asarray(left, dtype=float)
        right_values = np.asarray(right, dtype=float)
        if not np.array_equal(np.isfinite(left_values), np.isfinite(right_values)):
            return False, math.inf, "finite_mask"
        finite = np.isfinite(left_values)
        maximum = (
            float(np.max(np.abs(left_values[finite] - right_values[finite])))
            if np.any(finite)
            else 0.0
        )
        return maximum <= tolerance, maximum, "numeric"
    equal = np.array_equal(left, right)
    return bool(equal), 0.0 if equal else math.inf, "exact"


def _compare_npz(first: Path, second: Path, tolerance: float) -> dict[str, Any]:
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        left_keys = sorted(left.files)
        right_keys = sorted(right.files)
        if left_keys != right_keys:
            return {"passed": False, "reason": "keys", "left": left_keys, "right": right_keys}
        maximum = 0.0
        for key in left_keys:
            passed, difference, reason = _compare_numeric_arrays(left[key], right[key], tolerance)
            maximum = max(maximum, difference)
            if not passed:
                return {
                    "passed": False,
                    "reason": reason,
                    "key": key,
                    "left_shape": list(left[key].shape),
                    "right_shape": list(right[key].shape),
                    "max_abs_difference": difference,
                }
    return {"passed": True, "arrays": len(left_keys), "max_abs_difference": maximum}


def compare_runs(first_root: Path, second_root: Path, tolerance: float) -> dict[str, Any]:
    try:
        first_manifest = json.loads((first_root / "run_manifest.json").read_text(encoding="utf-8"))
        second_manifest = json.loads((second_root / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        first_manifest = {}
        second_manifest = {}
    first_profile_sha = str(first_manifest.get("profile_sha256", ""))
    second_profile_sha = str(second_manifest.get("profile_sha256", ""))
    first_code_sha = first_manifest.get("environment", {}).get("code_sha256", {})
    second_code_sha = second_manifest.get("environment", {}).get("code_sha256", {})
    try:
        first_inventory_sha = _sha256(first_root / "input_inventory.csv")
        second_inventory_sha = _sha256(second_root / "input_inventory.csv")
    except OSError:
        first_inventory_sha = ""
        second_inventory_sha = ""
    first_files = _scientific_files(first_root)
    second_files = _scientific_files(second_root)
    missing_first = sorted(str(path) for path in second_files - first_files)
    missing_second = sorted(str(path) for path in first_files - second_files)
    common = sorted(first_files & second_files)
    failures: list[dict[str, Any]] = []
    maximum = 0.0
    csv_count = 0
    npz_count = 0
    array_count = 0
    for relative in common:
        if relative.suffix.lower() == ".csv":
            csv_count += 1
            result = _compare_csv(first_root / relative, second_root / relative, tolerance)
        else:
            npz_count += 1
            result = _compare_npz(first_root / relative, second_root / relative, tolerance)
            array_count += int(result.get("arrays", 0))
        difference = result.get("max_abs_difference", 0.0)
        if isinstance(difference, (int, float)) and math.isfinite(float(difference)):
            maximum = max(maximum, float(difference))
        if not result.get("passed", False):
            failures.append({"file": str(relative), **result})
            if len(failures) >= 50:
                break
    same_profile = bool(first_profile_sha) and first_profile_sha == second_profile_sha
    same_code = bool(first_code_sha) and first_code_sha == second_code_sha
    same_inventory = bool(first_inventory_sha) and first_inventory_sha == second_inventory_sha
    complete_coverage = (
        bool(common)
        and len(common) == len(first_files)
        and len(common) == len(second_files)
        and not missing_first
        and not missing_second
    )
    passed = (
        same_profile
        and same_code
        and same_inventory
        and complete_coverage
        and not failures
    )
    return {
        "validator": "compare_uniform_correlation_runs-v1",
        "profile": "uniform-correlation-v2",
        "first_run": str(first_root.resolve()),
        "second_run": str(second_root.resolve()),
        "tolerance": tolerance,
        "checks": {
            "same_profile_sha256": same_profile,
            "same_code_sha256": same_code,
            "same_input_inventory_sha256": same_inventory,
            "same_scientific_file_set": not missing_first and not missing_second,
            "complete_scientific_file_coverage": complete_coverage,
            "all_scientific_values_reproducible": not failures and bool(common),
            "max_abs_difference_within_tolerance": maximum <= tolerance,
        },
        "details": {
            "first_profile_sha256": first_profile_sha,
            "second_profile_sha256": second_profile_sha,
            "first_input_inventory_sha256": first_inventory_sha,
            "second_input_inventory_sha256": second_inventory_sha,
            "same_code_sha256": same_code,
            "expected_scientific_files_first": len(first_files),
            "expected_scientific_files_second": len(second_files),
            "matched_scientific_files": len(common),
            "scientific_csv_files_compared": csv_count,
            "scientific_npz_files_compared": npz_count,
            "npz_arrays_compared": array_count,
            "max_abs_difference": maximum,
            "unmatched_scientific_files": sorted(set(missing_first + missing_second))[:100],
            "missing_from_first": missing_first[:50],
            "missing_from_second": missing_second[:50],
            "failures": failures,
        },
        "errors": [item.get("file", "comparison failure") for item in failures]
        + (["scientific file sets differ"] if missing_first or missing_second else []),
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_run", type=Path)
    parser.add_argument("second_run", type=Path)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_runs(args.first_run, args.second_run, args.tolerance)
    payload = json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().resolve().write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
