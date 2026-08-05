#!/usr/bin/env python3
"""Validate a strict-lower-triangle visualization-only UOTe rerun."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from single_global_per_peak import (  # noqa: E402
    HEATMAP_TRIANGLE_POLICY,
    strict_lower_triangle_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def matrix_csvs(root: Path) -> dict[Path, Path]:
    found: dict[Path, Path] = {}
    for file_path in root.rglob("*.csv"):
        relative = file_path.relative_to(root)
        if "validation" in relative.parts:
            continue
        if (
            any("matrices" in part for part in relative.parts)
            or "matrix" in file_path.stem
        ):
            found[relative] = file_path
    return found


def npz_files(root: Path) -> dict[Path, Path]:
    return {
        file_path.relative_to(root): file_path
        for file_path in root.rglob("*.npz")
        if "validation" not in file_path.relative_to(root).parts
    }


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if np.issubdtype(left.dtype, np.number):
        return bool(np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))
    return bool(np.array_equal(left, right))


def compare_npz(reference: Path, current: Path) -> tuple[bool, int, list[str]]:
    problems: list[str] = []
    arrays = 0
    with np.load(reference, allow_pickle=True) as old, np.load(current, allow_pickle=True) as new:
        if set(old.files) != set(new.files):
            problems.append(
                f"keys differ: old={sorted(old.files)}, new={sorted(new.files)}"
            )
            return False, 0, problems
        for key in old.files:
            arrays += 1
            if not arrays_equal(old[key], new[key]):
                problems.append(f"array differs: {key}")
    return not problems, arrays, problems


def main() -> None:
    args = parse_args()
    reference_root = args.reference_root.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else result_root / "validation" / "strict_lower_triangle_validation.json"
    )

    checks: list[dict[str, Any]] = []

    old_matrix_csvs = matrix_csvs(reference_root)
    new_matrix_csvs = matrix_csvs(result_root)
    same_matrix_set = set(old_matrix_csvs) == set(new_matrix_csvs)
    differing_matrix_csvs = [
        str(relative)
        for relative in sorted(set(old_matrix_csvs) & set(new_matrix_csvs))
        if old_matrix_csvs[relative].read_bytes() != new_matrix_csvs[relative].read_bytes()
    ]
    checks.append({
        "check": "matrix_csv_file_set_exact",
        "value": len(new_matrix_csvs),
        "expected": len(old_matrix_csvs),
        "passed": int(same_matrix_set),
    })
    checks.append({
        "check": "matrix_csv_values_exact",
        "value": len(differing_matrix_csvs),
        "expected": 0,
        "passed": int(same_matrix_set and not differing_matrix_csvs),
        "details": differing_matrix_csvs[:20],
    })

    old_npz = npz_files(reference_root)
    new_npz = npz_files(result_root)
    same_npz_set = set(old_npz) == set(new_npz)
    npz_array_count = 0
    npz_problems: dict[str, list[str]] = {}
    for relative in sorted(set(old_npz) & set(new_npz)):
        equal, count, problems = compare_npz(old_npz[relative], new_npz[relative])
        npz_array_count += count
        if not equal:
            npz_problems[str(relative)] = problems
    checks.append({
        "check": "npz_file_set_exact",
        "value": len(new_npz),
        "expected": len(old_npz),
        "passed": int(same_npz_set),
    })
    checks.append({
        "check": "npz_arrays_exact",
        "value": npz_array_count,
        "expected": "all arrays exact",
        "passed": int(same_npz_set and not npz_problems),
        "details": npz_problems,
    })

    pair_relative = Path("single_crystal/per_peak_all_frames/all_pair_scores.csv")
    pair_exact = (
        (reference_root / pair_relative).read_bytes()
        == (result_root / pair_relative).read_bytes()
    )
    checks.append({
        "check": "single_all_pair_scores_exact",
        "value": int(pair_exact),
        "expected": 1,
        "passed": int(pair_exact),
    })

    payload = np.load(
        result_root / "single_crystal/per_peak_all_frames/per_track_matrices.npz"
    )
    location = np.asarray(payload["location_similarity"], dtype=float)
    area = np.asarray(payload["normalized_area_similarity"], dtype=float)
    observed = np.asarray(payload["observed_mask"], dtype=bool)
    location_diag = np.diagonal(location, axis1=1, axis2=2)
    area_diag = np.diagonal(area, axis1=1, axis2=2)
    full_matrix_intact = (
        np.allclose(location, np.swapaxes(location, 1, 2), equal_nan=True)
        and np.allclose(area, np.swapaxes(area, 1, 2), equal_nan=True)
        and np.array_equal(np.isfinite(location_diag), observed)
        and np.array_equal(np.isfinite(area_diag), observed)
        and np.allclose(location_diag[observed], 1.0)
        and np.allclose(area_diag[observed], 1.0)
    )
    checks.append({
        "check": "underlying_full_symmetric_matrices_unchanged",
        "value": int(full_matrix_intact),
        "expected": 1,
        "passed": int(full_matrix_intact),
    })

    probe = np.arange(25, dtype=float).reshape(5, 5)
    data_layer, missing_layer = strict_lower_triangle_layers(probe)
    data_mask = np.ma.getmaskarray(data_layer)
    missing_mask = np.ma.getmaskarray(missing_layer)
    structural_hidden = data_mask & missing_mask
    upper = np.triu_indices(5, k=0)
    lower = np.tril_indices(5, k=-1)
    renderer_policy_ok = (
        np.all(structural_hidden[upper])
        and np.all(~data_mask[lower])
        and np.allclose(np.asarray(data_layer)[lower], probe[lower])
    )
    checks.append({
        "check": "renderer_strict_lower_no_diagonal",
        "value": HEATMAP_TRIANGLE_POLICY,
        "expected": "strict_lower_only_no_diagonal",
        "passed": int(renderer_policy_ok),
    })

    single_root = result_root / "single_crystal/per_peak_all_frames"
    single_plot_counts = {
        "location": len(list((single_root / "location_heatmaps").glob("track_*.png"))),
        "normalized_area": len(list((single_root / "normalized_area_heatmaps").glob("track_*.png"))),
        "paired": len(list((single_root / "paired_heatmaps").glob("track_*_location_area.png"))),
        "gallery": len(list((single_root / "gallery").glob("paired_heatmaps_page_*.png"))),
        "aggregate": sum(
            int((single_root / name).is_file())
            for name in (
                "aggregate_location_heatmap.png",
                "aggregate_normalized_area_heatmap.png",
            )
        ),
    }
    checks.append({
        "check": "single_global_matrix_plot_files",
        "value": single_plot_counts,
        "expected": {
            "location": 75,
            "normalized_area": 75,
            "paired": 75,
            "gallery": 13,
            "aggregate": 2,
        },
        "passed": int(single_plot_counts == {
            "location": 75,
            "normalized_area": 75,
            "paired": 75,
            "gallery": 13,
            "aggregate": 2,
        }),
    })

    correlation_pngs = [
        file_path
        for file_path in result_root.rglob("*.png")
        if "validation" not in file_path.relative_to(result_root).parts
        and (
            "heatmap" in file_path.name.lower()
            or any("heatmap" in part.lower() for part in file_path.parts)
        )
    ]
    checks.append({
        "check": "correlation_matrix_png_inventory",
        "value": len(correlation_pngs),
        "expected": 431,
        "passed": int(len(correlation_pngs) == 431),
    })

    validation_rows = read_csv(result_root / "validation/validation_checks.csv")
    validation_headers = validation_rows[0]
    passed_index = validation_headers.index("passed")
    failed_main = [
        row[0]
        for row in validation_rows[1:]
        if int(float(row[passed_index])) != 1
    ]
    gui_rows = read_csv(
        result_root / "validation/gui_crosscheck/gui_crosscheck_checks.csv"
    )
    gui_passed_index = gui_rows[0].index("passed")
    failed_gui = [
        row[0]
        for row in gui_rows[1:]
        if int(float(row[gui_passed_index])) != 1
    ]
    checks.append({
        "check": "main_and_gui_checks_all_pass",
        "value": {
            "main_checks": len(validation_rows) - 1,
            "gui_checks": len(gui_rows) - 1,
            "failed_main": failed_main,
            "failed_gui": failed_gui,
        },
        "expected": "no failed checks",
        "passed": int(not failed_main and not failed_gui),
    })

    manifest = json.loads((result_root / "run_manifest.json").read_text())
    single_metrics = manifest["single_crystal"]["per_peak_all_frames"]
    manifest_policy_ok = (
        manifest["validation_passed"] is True
        and single_metrics["heatmap_triangle_policy"]
        == "strict_lower_only_no_diagonal"
        and single_metrics["heatmap_diagonal_and_upper_hidden"] is True
        and single_metrics["heatmap_lower_triangle_preserved"] is True
    )
    checks.append({
        "check": "manifest_triangle_policy_and_validation",
        "value": {
            "validation_passed": manifest["validation_passed"],
            "policy": single_metrics["heatmap_triangle_policy"],
            "upper_diagonal_hidden": single_metrics[
                "heatmap_diagonal_and_upper_hidden"
            ],
            "lower_preserved": single_metrics[
                "heatmap_lower_triangle_preserved"
            ],
        },
        "expected": "strict lower policy and all true",
        "passed": int(manifest_policy_ok),
    })

    workbook_verification = json.loads(
        (
            result_root
            / "validation/workbook_qa/workbook_verification.json"
        ).read_text()
    )
    workbook_ok = (
        workbook_verification["sheet_count"] == 15
        and workbook_verification["rendered_sheet_count"] == 15
        and workbook_verification["pre_export_formula_error_count"] == 0
        and workbook_verification["post_import_formula_error_count"] == 0
    )
    checks.append({
        "check": "workbook_render_reimport",
        "value": workbook_verification,
        "expected": "15 sheets, 15 renders, 0 formula errors before/after",
        "passed": int(workbook_ok),
    })

    passed = all(int(row["passed"]) == 1 for row in checks)
    report = {
        "passed": passed,
        "reference_root": str(reference_root),
        "result_root": str(result_root),
        "policy": HEATMAP_TRIANGLE_POLICY,
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError(f"Strict-lower validation failed; inspect {output}")


if __name__ == "__main__":
    main()
