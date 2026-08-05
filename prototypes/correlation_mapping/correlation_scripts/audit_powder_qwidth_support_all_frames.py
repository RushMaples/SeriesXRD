#!/usr/bin/env python3
"""Audit powder q-width supports across all accepted and observed frames.

The audit separates two questions:

1. Was the requested support ``[qi-c*q_width, qi+c*q_width]`` implemented
   exactly and transferred unchanged to every waterfall?
2. Even when implemented exactly, how much local transformed positive signal
   is retained at ``c=0.6`` compared with wider candidate supports?

All 1,060 accepted powder frames are inventoried.  Signal-support metrics are
defined only for the 519 formal observations in the 360 frames that contain a
registered observation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import nonlinear_intensity_preprocessing as nonlinear  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_ROOT = (
    ROOT
    / "correlations/results/"
    "uote_nonlinear_squared_preprocessed_comparison_20260802"
)
DEFAULT_REFINEMENT_ROOT = ROOT / "correlations/UOTe XRD Data Refinement"
DEFAULT_WATERFALL_ROOT = (
    DEFAULT_COMPARISON_ROOT / "waterfall_complete_formal_composite_20260803"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_COMPARISON_ROOT
    / "validation/qwidth_support_all_frames_20260803"
)
POWDER_WAVELENGTH_A = 0.3066
FACTORS = (0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
PRIMARY_FACTOR = 0.6
REFERENCE_FACTOR = 1.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT
    )
    parser.add_argument(
        "--refinement-root", type=Path, default=DEFAULT_REFINEMENT_ROOT
    )
    parser.add_argument(
        "--waterfall-root", type=Path, default=DEFAULT_WATERFALL_ROOT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def q_to_two_theta(q: float | np.ndarray) -> np.ndarray:
    values = np.asarray(q, dtype=float)
    argument = values * POWDER_WAVELENGTH_A / (4.0 * np.pi)
    if np.any(np.abs(argument) > 1.0):
        raise ValueError("q is incompatible with the powder wavelength")
    return np.degrees(2.0 * np.arcsin(argument))


def infer_spots_path(fit_path: str) -> Path:
    path = Path(fit_path)
    parts = list(path.parts)
    try:
        index = parts.index("fit_channel")
    except ValueError as exc:
        raise ValueError(f"not a fit-channel path: {fit_path}") from exc
    parts[index] = "spots_channel"
    return Path(*parts)


def insert_zero_crossings(
    coordinate: np.ndarray, raw_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(coordinate, dtype=float)
    y = np.asarray(raw_values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("invalid piecewise-linear profile")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("profile coordinate is not strictly increasing")
    new_x: list[float] = [float(x[0])]
    new_y: list[float] = [float(y[0])]
    for index in range(x.size - 1):
        x0 = float(x[index])
        x1 = float(x[index + 1])
        y0 = float(y[index])
        y1 = float(y[index + 1])
        if y0 * y1 < 0.0:
            fraction = -y0 / (y1 - y0)
            crossing = x0 + fraction * (x1 - x0)
            if x0 < crossing < x1:
                new_x.append(crossing)
                new_y.append(0.0)
        new_x.append(x1)
        new_y.append(y1)
    return (
        np.asarray(new_x, dtype=float),
        np.maximum(np.asarray(new_y, dtype=float), 0.0),
    )


def component_metrics(
    x_two_theta: np.ndarray,
    raw_intensity: np.ndarray,
    *,
    q_center: float,
    q_width: float,
    factor: float,
    multiplier: float,
    spec: nonlinear.ROITransformSpec,
) -> dict[str, float | int]:
    q_bounds = np.asarray(
        [q_center - factor * q_width, q_center + factor * q_width], dtype=float
    )
    theta_bounds = q_to_two_theta(q_bounds)
    lower = float(theta_bounds[0])
    upper = float(theta_bounds[1])
    native = (x_two_theta > lower) & (x_two_theta < upper)
    coordinate = np.concatenate(
        (np.asarray([lower]), x_two_theta[native], np.asarray([upper]))
    )
    raw = np.interp(
        coordinate, x_two_theta, raw_intensity, left=0.0, right=0.0
    )
    coordinate, positive = insert_zero_crossings(coordinate, raw)
    transformed = np.asarray(spec.transform(positive * multiplier), dtype=float)
    if transformed.shape != coordinate.shape or np.any(~np.isfinite(transformed)):
        raise ValueError("invalid transformed component")
    if np.any(transformed < 0.0) or np.any(transformed > 1.0):
        raise ValueError("transformed component outside [0,1]")
    maximum = float(np.max(transformed)) if transformed.size else 0.0
    return {
        "q_lower": float(q_bounds[0]),
        "q_upper": float(q_bounds[1]),
        "theta_lower": lower,
        "theta_upper": upper,
        "theta_width": upper - lower,
        "native_points": int(np.count_nonzero((x_two_theta >= lower) & (x_two_theta <= upper))),
        "integral": float(np.trapezoid(transformed, coordinate)),
        "maximum": maximum,
        "left_value": float(transformed[0]),
        "right_value": float(transformed[-1]),
    }


def intervals_overlap(
    q_i: float,
    width_i: float,
    q_j: float,
    width_j: float,
    factor: float,
) -> bool:
    left_i = q_i - factor * width_i
    right_i = q_i + factor * width_i
    left_j = q_j - factor * width_j
    right_j = q_j + factor * width_j
    return min(right_i, right_j) > max(left_i, left_j)


def merge_intervals(
    intervals: Iterable[tuple[float, float]], tolerance: float = 1.0e-12
) -> list[tuple[float, float]]:
    ordered = sorted((float(left), float(right)) for left, right in intervals)
    merged: list[list[float]] = []
    for left, right in ordered:
        if right <= left:
            continue
        if not merged or left > merged[-1][1] + tolerance:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [(left, right) for left, right in merged]


def finite_quantile(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.quantile(finite, quantile)) if finite.size else math.nan


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else math.nan


def load_specs(comparison_root: Path) -> dict[str, nonlinear.ROITransformSpec]:
    specs: dict[str, nonlinear.ROITransformSpec] = {}
    for mode in ("log_squared", "exp_squared"):
        path = (
            comparison_root
            / "_sources"
            / mode
            / "powder_roi/intensity_transform_provenance.json"
        )
        provenance = json.loads(path.read_text(encoding="utf-8"))
        specs[mode] = nonlinear.ROITransformSpec(**provenance["transform"])
    return specs


def verify_waterfall_mappings(
    *,
    waterfall_root: Path,
    expected: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("log_squared", "exp_squared"):
        path = waterfall_root / "powder" / mode / "PEAK_COLOR_MAPPING.csv.gz"
        row_count = 0
        anchors: set[str] = set()
        point_components: set[tuple[str, int]] = set()
        max_left_error = 0.0
        max_right_error = 0.0
        mismatches = 0
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row_count += 1
                anchors.add(row["anchor_token"])
                uid = row["point_uid"]
                component_index = int(row["support_component_index"])
                point_components.add((uid, component_index))
                expected_items = expected.get(uid, ())
                if component_index >= len(expected_items):
                    mismatches += 1
                    continue
                expected_left, expected_right = expected_items[component_index]
                left_error = abs(float(row["support_left_deg"]) - expected_left)
                right_error = abs(float(row["support_right_deg"]) - expected_right)
                max_left_error = max(max_left_error, left_error)
                max_right_error = max(max_right_error, right_error)
                mismatches += int(left_error > 1.0e-10 or right_error > 1.0e-10)
                if row["trace_source"] != "formal_composite":
                    mismatches += 1
        result[mode] = {
            "mapping_path": str(path.resolve()),
            "mapping_rows": row_count,
            "anchors": len(anchors),
            "unique_point_support_components": len(point_components),
            "max_support_left_abs_error_deg": max_left_error,
            "max_support_right_abs_error_deg": max_right_error,
            "mismatch_rows": mismatches,
            "status": "PASS" if mismatches == 0 else "FAIL",
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_root = args.comparison_root / "_sources/log_squared/powder_roi"
    audit_rows = read_rows(
        source_root / "observation_spots_absolute_profile_audit.csv"
    )
    normalization_rows = read_rows(source_root / "frame_measurement_normalization.csv")
    observation_rows = read_rows(
        args.refinement_root / "Powder Scan/Track Analysis/spot_observations.csv"
    )
    if len(audit_rows) != 519 or len(observation_rows) != 519:
        raise ValueError(
            f"expected 519 observations, found audit={len(audit_rows)}, "
            f"source={len(observation_rows)}"
        )
    if len(normalization_rows) != 1060:
        raise ValueError(f"expected 1060 accepted frames, found {len(normalization_rows)}")

    normalization_by_frame = {int(row["frame"]): row for row in normalization_rows}
    if len(normalization_by_frame) != len(normalization_rows):
        raise ValueError("accepted frame IDs are not unique")
    specs = load_specs(args.comparison_root)

    observations_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in audit_rows:
        observations_by_frame[int(row["frame"])].append(row)

    source_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    frame_rows: list[dict[str, Any]] = []
    invalid_frame_files = 0
    for frame, normalization in sorted(normalization_by_frame.items()):
        source = infer_spots_path(normalization["fit_channel_file"])
        source_exists = source.is_file()
        points = 0
        x_min = math.nan
        x_max = math.nan
        median_step = math.nan
        monotonic = False
        if source_exists:
            data = np.loadtxt(source, comments="#", dtype=float)
            valid = (
                data.ndim == 2
                and data.shape[1] >= 2
                and data.shape[0] >= 3
                and np.all(np.isfinite(data[:, :2]))
            )
            if valid:
                x = np.asarray(data[:, 0], dtype=float)
                y = np.asarray(data[:, 1], dtype=float)
                monotonic = bool(np.all(np.diff(x) > 0.0))
                valid = valid and monotonic
            if valid:
                source_cache[source] = (x, y)
                points = int(x.size)
                x_min = float(x[0])
                x_max = float(x[-1])
                median_step = float(np.median(np.diff(x)))
            else:
                invalid_frame_files += 1
        else:
            invalid_frame_files += 1
        frame_rows.append(
            {
                "frame": frame,
                "scan": normalization["scan"],
                "pressure_gpa": normalization["pressure_gpa"],
                "exposure_token": normalization["exposure_token"],
                "formal_observation_count": len(observations_by_frame.get(frame, ())),
                "has_formal_observation": int(frame in observations_by_frame),
                "spots_channel_file": str(source),
                "source_exists": int(source_exists),
                "xy_points": points,
                "two_theta_min_deg": x_min,
                "two_theta_max_deg": x_max,
                "median_two_theta_step_deg": median_step,
                "strictly_increasing_axis": int(monotonic),
            }
        )
    write_rows(args.out_dir / "accepted_frame_coverage_audit.csv", frame_rows)
    if invalid_frame_files:
        raise RuntimeError(f"invalid accepted frame XY files: {invalid_frame_files}")

    detailed_rows: list[dict[str, Any]] = []
    q_lower_errors: list[float] = []
    q_upper_errors: list[float] = []
    theta_lower_errors: list[float] = []
    theta_upper_errors: list[float] = []
    native_count_mismatches = 0
    source_observation_mismatches = 0
    source_path_mismatches = 0
    multiplier_mismatches = 0

    for index, row in enumerate(audit_rows):
        obs_index = int(row["obs_index_0based"])
        if obs_index != index:
            raise ValueError("audit observation index is not row-order stable")
        source_observation = observation_rows[obs_index]
        frame = int(row["frame"])
        normalization = normalization_by_frame[frame]
        q_center = float(row["q_center"])
        q_width = float(row["q_width"])
        multiplier = float(row["original_main_measurement_multiplier"])
        source = Path(row["spots_channel_file"])
        expected_source = infer_spots_path(normalization["fit_channel_file"])
        source_path_mismatches += int(source != expected_source)
        multiplier_mismatches += int(
            not math.isclose(
                multiplier,
                float(normalization["main_area_multiplier"]),
                rel_tol=0.0,
                abs_tol=1.0e-11,
            )
        )
        source_observation_mismatches += int(
            frame != int(source_observation["frame"])
            or not math.isclose(
                q_center,
                float(source_observation["q"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                q_width,
                float(source_observation["q_width"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        x, y = source_cache[source]

        theoretical_q = np.asarray(
            [q_center - PRIMARY_FACTOR * q_width, q_center + PRIMARY_FACTOR * q_width]
        )
        theoretical_theta = q_to_two_theta(theoretical_q)
        q_lower_error = abs(float(row["q_lower"]) - theoretical_q[0])
        q_upper_error = abs(float(row["q_upper"]) - theoretical_q[1])
        theta_lower_error = abs(
            float(row["two_theta_lower_deg"]) - theoretical_theta[0]
        )
        theta_upper_error = abs(
            float(row["two_theta_upper_deg"]) - theoretical_theta[1]
        )
        q_lower_errors.append(q_lower_error)
        q_upper_errors.append(q_upper_error)
        theta_lower_errors.append(theta_lower_error)
        theta_upper_errors.append(theta_upper_error)

        same_frame = observations_by_frame[frame]
        neighbor_thresholds = [
            abs(q_center - float(neighbor["q_center"]))
            / (q_width + float(neighbor["q_width"]))
            for neighbor in same_frame
            if int(neighbor["obs_index_0based"]) != obs_index
        ]
        first_equal_factor_neighbor_overlap = (
            min(neighbor_thresholds) if neighbor_thresholds else math.inf
        )
        detailed: dict[str, Any] = {
            "obs_index_0based": obs_index,
            "point_uid": row["point_uid"],
            "frame": frame,
            "scan": normalization["scan"],
            "pressure_gpa": normalization["pressure_gpa"],
            "q_center": q_center,
            "q_width": q_width,
            "stored_half_width_factor": float(row["half_width_factor"]),
            "stored_q_lower": float(row["q_lower"]),
            "stored_q_upper": float(row["q_upper"]),
            "stored_two_theta_lower_deg": float(row["two_theta_lower_deg"]),
            "stored_two_theta_upper_deg": float(row["two_theta_upper_deg"]),
            "q_lower_abs_error": q_lower_error,
            "q_upper_abs_error": q_upper_error,
            "two_theta_lower_abs_error_deg": theta_lower_error,
            "two_theta_upper_abs_error_deg": theta_upper_error,
            "first_equal_factor_neighbor_overlap": first_equal_factor_neighbor_overlap,
            "spots_channel_file": str(source),
        }

        metrics_by_mode_factor: dict[tuple[str, float], dict[str, float | int]] = {}
        for factor in FACTORS:
            overlapping = any(
                intervals_overlap(
                    q_center,
                    q_width,
                    float(neighbor["q_center"]),
                    float(neighbor["q_width"]),
                    factor,
                )
                for neighbor in same_frame
                if int(neighbor["obs_index_0based"]) != obs_index
            )
            tag = str(factor).replace(".", "p")
            detailed[f"neighbor_overlap_c{tag}"] = int(overlapping)
            for mode, spec in specs.items():
                metrics = component_metrics(
                    x,
                    y,
                    q_center=q_center,
                    q_width=q_width,
                    factor=factor,
                    multiplier=multiplier,
                    spec=spec,
                )
                metrics_by_mode_factor[(mode, factor)] = metrics
                prefix = "log" if mode == "log_squared" else "exp"
                detailed[f"native_points_c{tag}"] = metrics["native_points"]
                detailed[f"theta_width_deg_c{tag}"] = metrics["theta_width"]
                detailed[f"{prefix}_integral_c{tag}"] = metrics["integral"]

        current_native = int(
            metrics_by_mode_factor[("log_squared", PRIMARY_FACTOR)]["native_points"]
        )
        native_count_mismatches += int(
            current_native != int(row["native_points_in_interval"])
        )
        for mode in ("log_squared", "exp_squared"):
            prefix = "log" if mode == "log_squared" else "exp"
            reference = metrics_by_mode_factor[(mode, REFERENCE_FACTOR)]
            reference_integral = float(reference["integral"])
            reference_max = float(reference["maximum"])
            for factor in FACTORS:
                tag = str(factor).replace(".", "p")
                metrics = metrics_by_mode_factor[(mode, factor)]
                integral = float(metrics["integral"])
                capture = (
                    integral / reference_integral
                    if reference_integral > 0.0
                    else math.nan
                )
                boundary_ratio = (
                    max(float(metrics["left_value"]), float(metrics["right_value"]))
                    / reference_max
                    if reference_max > 0.0
                    else math.nan
                )
                detailed[f"{prefix}_capture_vs_c1_c{tag}"] = capture
                detailed[f"{prefix}_boundary_to_c1_max_c{tag}"] = boundary_ratio
        detailed_rows.append(detailed)

    write_rows(args.out_dir / "observation_support_audit.csv", detailed_rows)

    factor_rows: list[dict[str, Any]] = []
    for factor in FACTORS:
        tag = str(factor).replace(".", "p")
        native = [float(row[f"native_points_c{tag}"]) for row in detailed_rows]
        overlap = [int(row[f"neighbor_overlap_c{tag}"]) for row in detailed_rows]
        safe_at_reference = [
            row for row in detailed_rows if int(row["neighbor_overlap_c1p0"]) == 0
        ]
        summary_row: dict[str, Any] = {
            "half_width_factor": factor,
            "full_q_width_multiplier": 2.0 * factor,
            "observations": len(detailed_rows),
            "native_points_min": int(min(native)),
            "native_points_q05": finite_quantile(native, 0.05),
            "native_points_median": finite_quantile(native, 0.5),
            "native_points_q95": finite_quantile(native, 0.95),
            "observations_with_fewer_than_4_native_points": int(
                np.count_nonzero(np.asarray(native) < 4)
            ),
            "fraction_with_fewer_than_4_native_points": float(
                np.mean(np.asarray(native) < 4)
            ),
            "neighbor_overlap_observations": int(sum(overlap)),
            "neighbor_overlap_fraction": float(np.mean(overlap)),
            "reference_safe_observations_no_neighbor_overlap_at_c1": len(
                safe_at_reference
            ),
        }
        for prefix in ("log", "exp"):
            captures = [
                float(row[f"{prefix}_capture_vs_c1_c{tag}"])
                for row in safe_at_reference
            ]
            boundaries = [
                float(row[f"{prefix}_boundary_to_c1_max_c{tag}"])
                for row in safe_at_reference
            ]
            summary_row[f"{prefix}_capture_vs_c1_q05"] = finite_quantile(captures, 0.05)
            summary_row[f"{prefix}_capture_vs_c1_q25"] = finite_quantile(captures, 0.25)
            summary_row[f"{prefix}_capture_vs_c1_median"] = finite_quantile(captures, 0.5)
            summary_row[f"{prefix}_capture_vs_c1_q75"] = finite_quantile(captures, 0.75)
            summary_row[f"{prefix}_capture_vs_c1_mean"] = finite_mean(captures)
            summary_row[f"{prefix}_capture_below_0p8"] = int(
                np.count_nonzero(np.asarray(captures) < 0.8)
            )
            summary_row[f"{prefix}_capture_below_0p9"] = int(
                np.count_nonzero(np.asarray(captures) < 0.9)
            )
            summary_row[f"{prefix}_boundary_ratio_median"] = finite_quantile(
                boundaries, 0.5
            )
            summary_row[f"{prefix}_boundary_ratio_above_0p25"] = int(
                np.count_nonzero(np.asarray(boundaries) > 0.25)
            )
        factor_rows.append(summary_row)
    write_rows(args.out_dir / "factor_sensitivity.csv", factor_rows)

    expected_supports: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in audit_rows:
        expected_supports[row["point_uid"]].append(
            (
                float(row["two_theta_lower_deg"]),
                float(row["two_theta_upper_deg"]),
            )
        )
    merged_supports = {
        uid: merge_intervals(intervals) for uid, intervals in expected_supports.items()
    }
    mapping_audit = verify_waterfall_mappings(
        waterfall_root=args.waterfall_root, expected=merged_supports
    )

    primary_factor_row = next(
        row for row in factor_rows if math.isclose(row["half_width_factor"], 0.6)
    )
    accepted_with_observation = sum(
        int(row["has_formal_observation"]) for row in frame_rows
    )
    all_checks_pass = (
        len(detailed_rows) == 519
        and len(frame_rows) == 1060
        and invalid_frame_files == 0
        and source_observation_mismatches == 0
        and source_path_mismatches == 0
        and multiplier_mismatches == 0
        and native_count_mismatches == 0
        and max(q_lower_errors + q_upper_errors) <= 1.0e-12
        and max(theta_lower_errors + theta_upper_errors) <= 1.0e-9
        and all(item["status"] == "PASS" for item in mapping_audit.values())
        and all(math.isclose(float(row["half_width_factor"]), 0.6) for row in audit_rows)
    )
    summary = {
        "status": "PASS" if all_checks_pass else "FAIL",
        "scope": {
            "accepted_frames_checked": len(frame_rows),
            "accepted_frames_with_formal_observations": accepted_with_observation,
            "accepted_frames_without_formal_observations": (
                len(frame_rows) - accepted_with_observation
            ),
            "formal_observations_checked": len(detailed_rows),
            "unique_formal_observation_source_frames": len(observations_by_frame),
            "formal_pressure_level_points": len(merged_supports),
            "transform_modes_checked": ["log_squared", "exp_squared"],
            "candidate_half_width_factors": list(FACTORS),
        },
        "implementation_checks": {
            "all_stored_half_width_factors_equal_0p6": all(
                math.isclose(float(row["half_width_factor"]), 0.6)
                for row in audit_rows
            ),
            "max_q_bound_abs_error_A_inverse": max(q_lower_errors + q_upper_errors),
            "max_two_theta_bound_abs_error_deg": max(
                theta_lower_errors + theta_upper_errors
            ),
            "native_point_count_mismatches": native_count_mismatches,
            "source_observation_mismatches": source_observation_mismatches,
            "source_path_mismatches": source_path_mismatches,
            "measurement_multiplier_mismatches": multiplier_mismatches,
            "waterfall_mapping_audit": mapping_audit,
        },
        "primary_factor_0p6": primary_factor_row,
        "interpretation_guardrail": (
            "Capture ratios compare local transformed positive signal inside a "
            "candidate interval with c=1.0. They are restricted in the factor "
            "summary to observations whose c=1.0 interval does not overlap another "
            "detected observation in the same frame. This is a clipping diagnostic, "
            "not proof that every additional tail sample belongs to the same peak."
        ),
        "outputs": {
            "accepted_frame_coverage_csv": str(
                (args.out_dir / "accepted_frame_coverage_audit.csv").resolve()
            ),
            "observation_support_audit_csv": str(
                (args.out_dir / "observation_support_audit.csv").resolve()
            ),
            "factor_sensitivity_csv": str(
                (args.out_dir / "factor_sensitivity.csv").resolve()
            ),
        },
    }
    write_json(args.out_dir / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all_checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
