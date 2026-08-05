#!/usr/bin/env python3
"""Independent read-only audit for the all-peak integer-window suite."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PEAK_DATASETS = {
    "single_crystal_all_frames": Path(
        "single_crystal/all_frame_1d_peak_maps"
    ),
    "single_crystal_curated_2d": Path(
        "single_crystal/curated_2d_roi_peak_maps"
    ),
    "powder_all_spots": Path("powder/all_detected_spot_peak_maps"),
}

WINDOW_ROLES = {
    "single_spots": (
        Path("window_full_symmetric_audit/single_crystal/spots"),
        Path("single_crystal/windows/spots"),
    ),
    "powder_spots": (
        Path("window_full_symmetric_audit/powder/spots"),
        Path("powder/windows/spots"),
    ),
    "powder_fit_control": (
        Path("window_full_symmetric_audit/powder/fit_control"),
        Path("powder/windows/fit_control"),
    ),
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _matrix(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        materialized = list(reader)
    values = np.full((len(materialized), len(header) - 1), np.nan, dtype=float)
    for row_index, row in enumerate(materialized):
        if len(row) != len(header):
            raise AssertionError(f"ragged matrix CSV: {path}")
        for column_index, value in enumerate(row[1:]):
            if value.strip():
                values[row_index, column_index] = float(value)
    return values


def _assert_strict_lower(
    presented: np.ndarray,
    source: np.ndarray | None = None,
    *,
    tolerance: float = 2.0e-9,
) -> None:
    if presented.ndim != 2 or presented.shape[0] != presented.shape[1]:
        raise AssertionError(f"matrix is not square: {presented.shape}")
    n = presented.shape[0]
    structural = np.triu(np.ones((n, n), dtype=bool), k=0)
    if np.any(np.isfinite(presented[structural])):
        raise AssertionError("diagonal or mirrored upper triangle is not blank")
    if source is None:
        return
    if source.shape != presented.shape:
        raise AssertionError(f"source/presentation shapes differ: {source.shape}")
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    if not np.array_equal(
        np.isfinite(presented[lower]), np.isfinite(source[lower])
    ):
        raise AssertionError("lower-triangle finite mask differs from source")
    finite = lower & np.isfinite(source)
    if np.any(finite):
        difference = float(np.max(np.abs(presented[finite] - source[finite])))
        if difference > tolerance:
            raise AssertionError(f"presented/source maximum difference={difference}")


def _symmetric(array: np.ndarray, tolerance: float = 1.0e-10) -> bool:
    return bool(
        array.shape[-1] == array.shape[-2]
        and np.array_equal(
            np.isfinite(array), np.isfinite(np.swapaxes(array, -1, -2))
        )
        and np.allclose(
            array,
            np.swapaxes(array, -1, -2),
            rtol=0.0,
            atol=tolerance,
            equal_nan=True,
        )
    )


def _finite_scores_valid(array: np.ndarray) -> bool:
    values = np.asarray(array, dtype=float)
    if np.any(np.isinf(values)):
        return False
    finite = values[np.isfinite(values)]
    return bool(
        finite.size == 0
        or (float(np.min(finite)) >= -1.0 and float(np.max(finite)) <= 1.0)
    )


def _line_rows(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _line in handle) - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_peak_dataset(
    root: Path,
    dataset_name: str,
    relative: Path,
) -> dict[str, Any]:
    dataset_root = root / relative
    layout = _rows(dataset_root / "frame_slot_layout.csv")
    index = _rows(dataset_root / "per_anchor_peak_map_index.csv")
    registry = _rows(dataset_root / "peak_registry.csv")
    frames = [int(row["frame"]) for row in layout]
    counts = [int(row["peak_count"]) for row in layout]
    maximum_slots = max(int(row["max_local_peak_slots"]) for row in layout)
    if sum(counts) != len(registry) or len(index) != len(registry):
        raise AssertionError(f"{dataset_name}: registry/layout/index mismatch")
    if sum(int(row["zero_peak_frame"]) for row in layout) != sum(
        count == 0 for count in counts
    ):
        raise AssertionError(f"{dataset_name}: zero-frame count mismatch")

    matrix_files = 0
    heatmap_files = 0
    genuine_zero_cells = 0
    for anchor in index:
        anchor_frame = int(anchor["anchor_frame"])
        expected_mask = np.zeros((len(frames), maximum_slots), dtype=bool)
        for row_index, (target_frame, count) in enumerate(zip(frames, counts, strict=True)):
            if target_frame != anchor_frame:
                expected_mask[row_index, :count] = True
        metric_masks: list[np.ndarray] = []
        for metric in ("location", "area"):
            csv_path = dataset_root / anchor[f"{metric}_csv"]
            png_path = dataset_root / anchor[f"{metric}_png"]
            if not csv_path.is_file() or not png_path.is_file():
                raise AssertionError(f"{dataset_name}: missing map artifact")
            values = _matrix(csv_path)
            if values.shape != expected_mask.shape:
                raise AssertionError(
                    f"{dataset_name}: {csv_path.name} shape {values.shape} "
                    f"!= {expected_mask.shape}"
                )
            finite = np.isfinite(values)
            if not np.array_equal(finite, expected_mask):
                raise AssertionError(
                    f"{dataset_name}: structural blank mask mismatch in {csv_path}"
                )
            numeric = values[finite]
            if numeric.size and (
                np.min(numeric) < 0.0 or np.max(numeric) > 1.0
            ):
                raise AssertionError(f"{dataset_name}: score outside [0,1]")
            genuine_zero_cells += int(np.count_nonzero(numeric == 0.0))
            metric_masks.append(finite)
            matrix_files += 1
            heatmap_files += 1
        if not np.array_equal(metric_masks[0], metric_masks[1]):
            raise AssertionError(f"{dataset_name}: location/area masks differ")
        expected_finite = int(anchor["expected_cross_frame_peak_comparisons"])
        if int(np.count_nonzero(metric_masks[0])) != expected_finite:
            raise AssertionError(f"{dataset_name}: finite cell count mismatch")

    pressure_values = [float(row["pressure_GPa"]) for row in layout]
    return {
        "registered_frames": len(layout),
        "peaks_and_anchor_maps": len(index),
        "maximum_local_peak_slots": maximum_slots,
        "zero_peak_frames": sum(count == 0 for count in counts),
        "matrix_csv_files_verified": matrix_files,
        "heatmap_png_files_verified": heatmap_files,
        "genuine_numeric_zero_cells_seen": genuine_zero_cells,
        "pressure_min_GPa": min(pressure_values),
        "pressure_max_GPa": max(pressure_values),
        "all_anchor_rows_and_missing_slots_blank": True,
    }


def _audit_geometry(root: Path) -> dict[str, Any]:
    rows = _rows(root / "window_provenance/integer_window_geometry.csv")
    expected_counts = {"single_spots": 19, "powder_spots": 28, "powder_fit_control": 28}
    effective_first: dict[str, float] = {}
    for role, count in expected_counts.items():
        selected = [row for row in rows if row["role"] == role]
        if len(selected) != count:
            raise AssertionError(f"{role}: geometry row count mismatch")
        for index, row in enumerate(selected):
            start = float(row["nominal_start_deg"])
            end = float(row["nominal_end_deg"])
            if start != float(index) or end != float(index + 5):
                raise AssertionError(f"{role}: non-integer geometry at {index}")
            if float(row["nominal_width_deg"]) != 5.0:
                raise AssertionError(f"{role}: nominal width is not five degrees")
            if int(row["extrapolated"]) != 0:
                raise AssertionError(f"{role}: extrapolation flag is set")
        effective_first[role] = float(selected[0]["effective_observed_start_deg"])
    return {
        "geometry_rows": len(rows),
        "window_counts": expected_counts,
        "first_effective_start_deg": effective_first,
        "all_nominal_windows_are_0_5_1_6_sequence": True,
        "no_extrapolation": True,
    }


def _audit_window_sources(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, (source_relative, _destination_relative) in WINDOW_ROLES.items():
        source = root / source_relative
        with np.load(
            source / "across_frames/across_frame_matrices.npz",
            allow_pickle=False,
        ) as archive:
            across_arrays = {
                key: np.asarray(archive[key], dtype=float)
                for key in (
                    "acf_strict_aggregate",
                    "direct_strict_aggregate",
                    "shift_tolerant_secondary_aggregate",
                    "acf_strict_by_scan",
                    "direct_strict_by_scan",
                    "shift_tolerant_secondary_by_scan",
                )
            }
            starts = np.asarray(archive["window_starts_deg"], dtype=float)
            ends = np.asarray(archive["window_ends_deg"], dtype=float)
        with np.load(
            source / "within_frame/within_frame_matrices.npz",
            allow_pickle=False,
        ) as archive:
            within_arrays = {
                key: np.asarray(archive[key], dtype=float)
                for key in (
                    "aggregate",
                    "aggregate_by_pressure",
                    "matrices_by_frame",
                )
            }
        arrays: Iterable[np.ndarray] = (
            *across_arrays.values(),
            *within_arrays.values(),
        )
        if not all(_symmetric(array) for array in arrays):
            raise AssertionError(f"{role}: a full source matrix is not symmetric")
        if not all(_finite_scores_valid(array) for array in arrays):
            raise AssertionError(f"{role}: invalid source similarity score")
        if not np.array_equal(starts, np.arange(starts.size, dtype=float)):
            raise AssertionError(f"{role}: starts are not exact integer degrees")
        if not np.array_equal(ends, starts + 5.0):
            raise AssertionError(f"{role}: ends are not starts + five degrees")
        result[role] = {
            "windows": int(starts.size),
            "first_window": f"{starts[0]:g}-{ends[0]:g}",
            "last_window": f"{starts[-1]:g}-{ends[-1]:g}",
            "full_sources_symmetric": True,
            "finite_scores_in_minus1_plus1": True,
        }
    return result


def _audit_presented_windows(root: Path) -> dict[str, Any]:
    index = _rows(root / "window_lower_triangle_index.csv")
    if len(index) != 277:
        raise AssertionError(f"primary window map count {len(index)} != 277")
    source_cache: dict[str, dict[str, np.ndarray]] = {}
    pressure_cache: dict[str, np.ndarray] = {}
    for role, (source_relative, _destination_relative) in WINDOW_ROLES.items():
        source = root / source_relative
        with np.load(
            source / "across_frames/across_frame_matrices.npz",
            allow_pickle=False,
        ) as archive:
            source_cache[f"{role}:across"] = {
                key: np.asarray(archive[key], dtype=float)
                for key in (
                    "acf_strict_aggregate",
                    "direct_strict_aggregate",
                    "shift_tolerant_secondary_aggregate",
                )
            }
        with np.load(
            source / "within_frame/within_frame_matrices.npz",
            allow_pickle=False,
        ) as archive:
            source_cache[f"{role}:within"] = {
                "aggregate": np.asarray(archive["aggregate"], dtype=float),
                "aggregate_by_pressure": np.asarray(
                    archive["aggregate_by_pressure"], dtype=float
                ),
            }
            pressure_cache[role] = np.asarray(
                archive["pressure_gpa"], dtype=float
            )

    across_maps = 0
    within_maps = 0
    diagnostic_maps = 0
    for row in index:
        role = row["role"]
        presented = _matrix(root / row["matrix_csv"])
        if row["comparison"] == "across_frames":
            method = row["method"]
            window_index = int(row["scope"].split("_")[-1])
            source = source_cache[f"{role}:across"][
                f"{method}_aggregate"
            ][window_index]
            across_maps += 1
        else:
            if row["scope"] == "aggregate":
                source = source_cache[f"{role}:within"]["aggregate"]
            else:
                pressure = float(
                    row["scope"].removeprefix("pressure_").removesuffix("GPa")
                )
                matches = np.flatnonzero(
                    np.isclose(
                        pressure_cache[role],
                        pressure,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
                if matches.size != 1:
                    raise AssertionError(f"{role}: pressure lookup failed")
                source = source_cache[f"{role}:within"][
                    "aggregate_by_pressure"
                ][matches[0]]
            within_maps += 1
        _assert_strict_lower(presented, source)

        diagnostic_csv = row.get("one_minus_similarity_diagnostic_csv", "")
        diagnostic_png = row.get("one_minus_similarity_diagnostic_png", "")
        if diagnostic_csv or diagnostic_png:
            if not diagnostic_csv or not diagnostic_png:
                raise AssertionError("incomplete 1-r diagnostic pair")
            diagnostic = _matrix(root / diagnostic_csv)
            _assert_strict_lower(diagnostic, 1.0 - source)
            if not (root / diagnostic_png).is_file():
                raise AssertionError("missing 1-r diagnostic image")
            diagnostic_maps += 1

    quicklook_rows = _rows(root / "window_quicklooks/quicklook_index.csv")
    if len(quicklook_rows) != 6:
        raise AssertionError("quicklook map count is not six")
    for row in quicklook_rows:
        _assert_strict_lower(_matrix(root / row["matrix_csv"]))

    across_pair_rows = 0
    within_summary_rows = 0
    within_frame_rows = 0
    for _role, (_source_relative, destination_relative) in WINDOW_ROLES.items():
        destination = root / destination_relative
        across_pair_rows += _line_rows(
            destination / "across_frames/unique_lower_triangle_pairs.csv"
        )
        within_summary_rows += _line_rows(
            destination / "within_frame/unique_summary_pairs.csv"
        )
        within_frame_rows += _line_rows(
            destination / "within_frame/per_frame_unique_pairs.csv.gz"
        )
    if (across_pair_rows, within_summary_rows, within_frame_rows) != (
        31863,
        17172,
        805122,
    ):
        raise AssertionError("window unique-pair row counts are incomplete")
    if (across_maps, within_maps, diagnostic_maps) != (225, 52, 28):
        raise AssertionError("window map-family counts are incomplete")

    old_float_names = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            "0.0435_5.3876" in path.name
            or "0p0435_5p3876" in path.name
        )
    ]
    if old_float_names:
        raise AssertionError("old span-scaled window filenames remain")
    return {
        "primary_across_maps": across_maps,
        "primary_within_maps": within_maps,
        "one_minus_diagnostic_maps": diagnostic_maps,
        "quicklook_maps": len(quicklook_rows),
        "across_unique_pair_rows": across_pair_rows,
        "within_summary_unique_pair_rows": within_summary_rows,
        "within_frame_unique_pair_rows": within_frame_rows,
        "all_presentations_are_strict_lower_and_match_source": True,
        "old_span_scaled_window_names_present": False,
    }


def audit(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    validation = json.loads((root / "validation_report.json").read_text())
    completion = json.loads((root / "RUN_COMPLETE.json").read_text())
    if validation["status"] != "PASS":
        raise AssertionError("suite validation status is not PASS")
    failed = [key for key, value in validation["requirements"].items() if not value]
    if failed:
        raise AssertionError(f"suite reports failed requirements: {failed}")
    for name, expected in (
        ("validation_report.json", "validation_report_sha256"),
        ("run_manifest.json", "run_manifest_sha256"),
        ("artifact_index.csv", "artifact_index_sha256"),
    ):
        if _sha256(root / name) != completion[expected]:
            raise AssertionError(f"completion hash mismatch: {name}")

    peak_audits = {
        name: _audit_peak_dataset(root, name, relative)
        for name, relative in PEAK_DATASETS.items()
    }
    if sum(item["heatmap_png_files_verified"] for item in peak_audits.values()) != 2998:
        raise AssertionError("total peak heatmap count is not 2,998")
    if peak_audits["single_crystal_all_frames"]["pressure_min_GPa"] != 1.0:
        raise AssertionError("single-crystal minimum pressure mismatch")
    if peak_audits["single_crystal_all_frames"]["pressure_max_GPa"] != 12.8:
        raise AssertionError("single-crystal maximum pressure mismatch")
    if peak_audits["powder_all_spots"]["pressure_min_GPa"] != 3.5:
        raise AssertionError("powder minimum pressure mismatch")
    if peak_audits["powder_all_spots"]["pressure_max_GPa"] != 50.7:
        raise AssertionError("powder maximum pressure mismatch")

    return {
        "status": "PASS",
        "root": str(root),
        "completion_hashes_verified": True,
        "peak_datasets": peak_audits,
        "window_geometry": _audit_geometry(root),
        "window_full_sources": _audit_window_sources(root),
        "window_presentations": _audit_presented_windows(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.suite), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
