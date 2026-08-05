#!/usr/bin/env python3
"""Pressure-level peak maps using 1D spots-channel q-width core profiles.

This is the second requested result group.  Unlike v6, it does not compare
two-dimensional cake coordinates.  Each raw spot observation locates a narrow
segment in its frame's one-dimensional ``spots_channel`` profile.  The segment
is recentered and width-normalized, so absolute q position does not contribute
to the ROI-area score (absolute position remains a separate location metric).

The retained core is

    [q_i - c*q_width_i, q_i + c*q_width_i],  c = 0.6

chosen as the narrowest tested range with adequate native sampling for almost
all observations.  Nonnegative, empirical measurement-normalized profiles are
compared by continuous min/max integrated IoU.  Every one of the existing 280
pressure-level points is an anchor in a 19 pressure x 22 local-peak-slot map.
The unchanged location and strict-lower-triangle across/within-frame integer
window outputs are copied byte-for-byte from the independently audited v6
suite and verified by SHA256 in this deliverable.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pressure_level_peak_correlations_v6 as v6


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = WORKSPACE_ROOT / "correlations" / "UOTe XRD Data Refinement"
DEFAULT_SPOTS_ROOT = DATA_ROOT / "Powder Scan" / "Reduced .xy" / "spots_channel"
DEFAULT_V6_SOURCE = (
    WORKSPACE_ROOT
    / "correlations"
    / "results"
    / "uote_pressure_level_peak_ellipse_iou_integer_window_suite_20260729_v6"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "correlations"
    / "results"
    / "uote_pressure_level_peak_spots_qwidth_iou_integer_window_suite_20260730_v7"
)

HALF_WIDTH_FACTOR = 0.6
RELATIVE_GRID_POINTS = 241
RELATIVE_GRID = np.linspace(
    -HALF_WIDTH_FACTOR,
    HALF_WIDTH_FACTOR,
    RELATIVE_GRID_POINTS,
)
FRAME_PATTERN = re.compile(r"frame_(\d+)_")
COPY_PAYLOADS = (
    Path("peak_maps/location"),
    Path("single_crystal/windows"),
    Path("powder/windows"),
    Path("window_full_symmetric_audit"),
    Path("window_quicklooks"),
    Path("window_provenance"),
    Path("LOWER_TRIANGLE_METHODS.md"),
    Path("WINDOW_METHODS.md"),
    Path("window_lower_triangle_index.csv"),
    Path("window_similarity_diagnostics.csv"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spots-root", type=Path, default=DEFAULT_SPOTS_ROOT)
    parser.add_argument("--v6-source", type=Path, default=DEFAULT_V6_SOURCE)
    parser.add_argument("--observations", type=Path, default=v6.DEFAULT_OBSERVATIONS)
    parser.add_argument("--track-points", type=Path, default=v6.DEFAULT_TRACK_POINTS)
    parser.add_argument(
        "--untracked-points", type=Path, default=v6.DEFAULT_UNTRACKED_POINTS
    )
    parser.add_argument("--manifest", type=Path, default=v6.DEFAULT_MANIFEST)
    parser.add_argument("--fit-root", type=Path, default=v6.DEFAULT_FIT_ROOT)
    parser.add_argument("--half-width-factor", type=float, default=HALF_WIDTH_FACTOR)
    parser.add_argument("--relative-grid-points", type=int, default=RELATIVE_GRID_POINTS)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--max-anchors", type=int, default=None)
    return parser.parse_args()


def _progress(message: str) -> None:
    print(f"[spots-qwidth-v7] {message}", flush=True)


def q_from_two_theta(
    two_theta_deg: float | np.ndarray,
    wavelength_a: float = v6.POWDER_WAVELENGTH_A,
) -> np.ndarray:
    values = np.asarray(two_theta_deg, dtype=float)
    return 4.0 * np.pi * np.sin(np.radians(values / 2.0)) / wavelength_a


def two_theta_from_q(
    q_a_inv: float | np.ndarray,
    wavelength_a: float = v6.POWDER_WAVELENGTH_A,
) -> np.ndarray:
    values = np.asarray(q_a_inv, dtype=float)
    argument = values * wavelength_a / (4.0 * np.pi)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(argument) & (np.abs(argument) <= 1.0)
    result[valid] = np.degrees(2.0 * np.arcsin(argument[valid]))
    return result


def extract_relative_profile(
    q_axis: np.ndarray,
    intensity: np.ndarray,
    *,
    q_center: float,
    q_width: float,
    relative_grid: np.ndarray = RELATIVE_GRID,
    clip_negative: bool = True,
    preserve_two_theta_area: bool = True,
) -> np.ndarray:
    """Extract/recenter a local profile and express area density on u.

    ``u=(q-q_center)/q_width``.  With ``preserve_two_theta_area=True``, the
    returned values include ``d(2theta)/du`` so integrating over u exactly
    reproduces the interpolated physical area in degrees.
    """
    q = np.asarray(q_axis, dtype=float)
    y = np.asarray(intensity, dtype=float)
    u = np.asarray(relative_grid, dtype=float)
    if q.ndim != 1 or y.shape != q.shape or u.ndim != 1:
        raise ValueError("q, intensity, and relative grid must be 1D")
    if q.size < 2 or not np.all(np.diff(q) > 0.0):
        raise ValueError("q axis must be strictly increasing")
    if not np.isfinite(q_center) or not np.isfinite(q_width) or q_width <= 0.0:
        raise ValueError("q center/width must be finite and width positive")
    target_q = q_center + u * q_width
    sampled = np.interp(target_q, q, y, left=0.0, right=0.0)
    if clip_negative:
        sampled = np.maximum(sampled, 0.0)
    if preserve_two_theta_area:
        theta = two_theta_from_q(target_q)
        derivative = np.gradient(theta, u, edge_order=2)
        if np.any(~np.isfinite(derivative)) or np.any(derivative <= 0.0):
            raise ValueError("invalid q-to-two-theta Jacobian")
        sampled = sampled * derivative
    return sampled


def profile_integrated_iou(
    left: np.ndarray,
    right: np.ndarray,
    coordinate: np.ndarray = RELATIVE_GRID,
) -> float:
    """Continuous min/max IoU for two nonnegative sampled 1D fields."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    x = np.asarray(coordinate, dtype=float)
    if a.shape != b.shape or a.shape != x.shape or a.ndim != 1:
        raise ValueError("profile and coordinate arrays must be equal-shape 1D")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValueError("profiles must be finite")
    if np.any(a < 0.0) or np.any(b < 0.0):
        raise ValueError("profiles must be nonnegative")
    intersection = float(np.trapezoid(np.minimum(a, b), x))
    union = float(np.trapezoid(np.maximum(a, b), x))
    if union <= 0.0:
        return math.nan
    score = intersection / union
    if score < -1.0e-12 or score > 1.0 + 1.0e-12:
        raise RuntimeError(f"profile IoU outside [0,1]: {score}")
    return float(min(1.0, max(0.0, score)))


def _load_spots_profiles(
    root: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray, Path]]:
    result: dict[int, tuple[np.ndarray, np.ndarray, Path]] = {}
    for path in root.rglob("frame_*.xy"):
        match = FRAME_PATTERN.search(path.name)
        if not match:
            continue
        frame = int(match.group(1))
        if frame in result:
            raise ValueError(f"duplicate spots-channel frame {frame}")
        data = np.loadtxt(path, comments="#", dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"invalid spots-channel XY: {path}")
        q_axis = q_from_two_theta(data[:, 0])
        if not np.all(np.diff(q_axis) > 0.0):
            raise ValueError(f"nonmonotonic q axis: {path}")
        result[frame] = (q_axis, data[:, 1], path)
    if len(result) != 1060:
        raise ValueError(f"expected 1060 spots-channel frames, found {len(result)}")
    return result


def build_spots_point_profiles(
    observations: Sequence[Mapping[str, Any]],
    points: Sequence[dict[str, Any]],
    normalization_by_frame: Mapping[int, Mapping[str, Any]],
    spots_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray, Path]],
    *,
    half_width_factor: float,
    relative_grid_points: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, Any],
    np.ndarray,
]:
    if not (0.0 < half_width_factor <= 2.0):
        raise ValueError("half-width factor must be in (0,2]")
    if relative_grid_points < 41 or relative_grid_points % 2 == 0:
        raise ValueError("relative grid point count must be odd and >=41")
    relative_grid = np.linspace(
        -half_width_factor,
        half_width_factor,
        relative_grid_points,
    )
    obs_by_index = {
        int(observation["obs_index_0based"]): observation
        for observation in observations
    }
    observation_profiles: dict[int, dict[str, Any]] = {}
    observation_audit: list[dict[str, Any]] = []
    native_counts: list[int] = []
    negative_fractions: list[float] = []
    observations_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        observations_by_frame[int(observation["frame"])].append(observation)
    neighbor_overlap_count = 0
    for observation in observations:
        obs_index = int(observation["obs_index_0based"])
        frame = int(observation["frame"])
        if frame not in spots_by_frame or frame not in normalization_by_frame:
            raise ValueError(f"missing spots/normalization for frame {frame}")
        q_axis, raw_intensity, path = spots_by_frame[frame]
        q_center = float(observation["q"])
        q_width = float(observation["q_width"])
        lower = q_center - half_width_factor * q_width
        upper = q_center + half_width_factor * q_width
        native_mask = (q_axis >= lower) & (q_axis <= upper)
        native_count = int(np.count_nonzero(native_mask))
        native_counts.append(native_count)
        native_values = raw_intensity[native_mask]
        negative_fraction = (
            float(np.mean(native_values < 0.0)) if native_values.size else math.nan
        )
        negative_fractions.append(negative_fraction)
        overlapping_neighbor_indices: list[int] = []
        for neighbor in observations_by_frame[frame]:
            neighbor_index = int(neighbor["obs_index_0based"])
            if neighbor_index == obs_index:
                continue
            neighbor_lower = float(neighbor["q"]) - (
                half_width_factor * float(neighbor["q_width"])
            )
            neighbor_upper = float(neighbor["q"]) + (
                half_width_factor * float(neighbor["q_width"])
            )
            if max(lower, neighbor_lower) <= min(upper, neighbor_upper):
                overlapping_neighbor_indices.append(neighbor_index)
        neighbor_overlap_count += int(bool(overlapping_neighbor_indices))
        base_profile = extract_relative_profile(
            q_axis,
            raw_intensity,
            q_center=q_center,
            q_width=q_width,
            relative_grid=relative_grid,
            clip_negative=True,
            preserve_two_theta_area=True,
        )
        normalization = normalization_by_frame[frame]
        main_multiplier = float(normalization["main_area_multiplier"])
        sensitivity_multiplier = float(
            normalization["fixed_token_sensitivity_multiplier"]
        )
        observation_profiles[obs_index] = {
            "frame": frame,
            "profile_main": base_profile * main_multiplier,
            "profile_sensitivity": base_profile * sensitivity_multiplier,
            "intensity_weight": float(observation["intensity"]),
        }
        observation_audit.append(
            {
                "obs_index_0based": obs_index,
                "point_uid": observation["point_uid"],
                "frame": frame,
                "q_center": q_center,
                "q_width": q_width,
                "half_width_factor": half_width_factor,
                "q_lower": lower,
                "q_upper": upper,
                "native_points_in_interval": native_count,
                "native_negative_fraction": negative_fraction,
                "negative_values_clipped_to_zero": 1,
                "overlaps_another_detected_peak_interval_in_same_frame": int(
                    bool(overlapping_neighbor_indices)
                ),
                "overlapping_neighbor_obs_indices": "|".join(
                    str(index) for index in sorted(overlapping_neighbor_indices)
                ),
                "main_measurement_multiplier": main_multiplier,
                "fixed_D1w_sensitivity_multiplier": sensitivity_multiplier,
                "spots_channel_file": str(path.resolve()),
            }
        )

    main_profiles: dict[str, np.ndarray] = {}
    sensitivity_profiles: dict[str, np.ndarray] = {}
    zero_integral_points: list[str] = []
    for point in points:
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for obs_index in point["member_obs_indices"]:
            by_frame[int(obs_by_index[int(obs_index)]["frame"])].append(
                observation_profiles[int(obs_index)]
            )
        frame_main: list[np.ndarray] = []
        frame_sensitivity: list[np.ndarray] = []
        for items in by_frame.values():
            frame_main.append(
                np.sum(
                    np.stack([item["profile_main"] for item in items]),
                    axis=0,
                )
            )
            frame_sensitivity.append(
                np.sum(
                    np.stack([item["profile_sensitivity"] for item in items]),
                    axis=0,
                )
            )
        main = np.mean(np.stack(frame_main), axis=0)
        sensitivity = np.mean(np.stack(frame_sensitivity), axis=0)
        uid = str(point["point_uid"])
        main_profiles[uid] = main
        sensitivity_profiles[uid] = sensitivity
        point["spots_qwidth_integral_main"] = float(
            np.trapezoid(main, relative_grid)
        )
        point["spots_qwidth_integral_fixed_D1w_sensitivity"] = float(
            np.trapezoid(sensitivity, relative_grid)
        )
        if point["spots_qwidth_integral_main"] <= 0.0:
            zero_integral_points.append(uid)

    counts = np.asarray(native_counts, dtype=int)
    audit = {
        "source_channel": "spots_channel",
        "two_dimensional_cake_coordinates_used": False,
        "absolute_q_position_used_in_ROI_score": False,
        "absolute_q_position_role": (
            "used only to locate each source segment; profiles are recentered "
            "to u=(q-qi)/q_width before comparison"
        ),
        "half_width_factor": half_width_factor,
        "interval_formula": "[qi-c*q_width, qi+c*q_width]",
        "relative_coordinate": "u=(q-qi)/q_width",
        "relative_grid_points": relative_grid_points,
        "negative_values": "clip to zero before nonnegative min/max integration",
        "area_measure": (
            "profile is multiplied by d(2theta)/du, so integral over u "
            "preserves the physical d(2theta) area"
        ),
        "same_frame_multiple_observations": (
            "sum observation-specific, nonoverlapping q-core segments within "
            "the physical frame; then count that frame once in the frame mean"
        ),
        "across_frames": "arithmetic mean over distinct physical frames",
        "native_points_min": int(np.min(counts)),
        "native_points_q05": float(np.quantile(counts, 0.05)),
        "native_points_median": float(np.median(counts)),
        "native_points_q95": float(np.quantile(counts, 0.95)),
        "fraction_with_fewer_than_4_native_points": float(np.mean(counts < 4)),
        "fraction_with_fewer_than_5_native_points": float(np.mean(counts < 5)),
        "median_native_negative_fraction": float(
            np.nanmedian(np.asarray(negative_fractions))
        ),
        "observations_whose_1D_interval_overlaps_another_detected_peak": (
            neighbor_overlap_count
        ),
        "one_dimensional_projection_caveat": (
            "overlapping q intervals cannot be separated by spots_channel; "
            "the audit CSV identifies every affected observation"
        ),
        "point_profiles": len(main_profiles),
        "zero_integral_points": zero_integral_points,
    }
    if len(main_profiles) != len(points):
        raise RuntimeError(f"invalid spots point profiles: {audit}")
    return (
        main_profiles,
        sensitivity_profiles,
        observation_audit,
        audit,
        relative_grid,
    )


def native_sampling_factor_optimization(
    observations: Sequence[Mapping[str, Any]],
    spots_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray, Path]],
    *,
    candidates: Sequence[float] = (0.4, 0.5, 0.6, 0.75, 1.0, 1.25),
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    """Choose the narrowest core with >=4 native points for >=90% of rows."""
    rows: list[dict[str, Any]] = []
    observations_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        observations_by_frame[int(observation["frame"])].append(observation)
    for factor in candidates:
        counts: list[int] = []
        interval_overlap_flags: list[bool] = []
        for observation in observations:
            q_axis = spots_by_frame[int(observation["frame"])][0]
            center = float(observation["q"])
            width = float(observation["q_width"])
            lower = center - factor * width
            upper = center + factor * width
            counts.append(
                int(
                    np.count_nonzero(
                        (q_axis >= lower)
                        & (q_axis <= upper)
                    )
                )
            )
            overlap = False
            for neighbor in observations_by_frame[int(observation["frame"])]:
                if neighbor is observation:
                    continue
                neighbor_center = float(neighbor["q"])
                neighbor_width = float(neighbor["q_width"])
                neighbor_lower = neighbor_center - factor * neighbor_width
                neighbor_upper = neighbor_center + factor * neighbor_width
                if max(lower, neighbor_lower) <= min(upper, neighbor_upper):
                    overlap = True
                    break
            interval_overlap_flags.append(overlap)
        values = np.asarray(counts, dtype=int)
        rows.append(
            {
                "half_width_factor": factor,
                "interval": f"[qi-{factor:g}*q_width, qi+{factor:g}*q_width]",
                "native_points_min": int(np.min(values)),
                "native_points_q05": float(np.quantile(values, 0.05)),
                "native_points_median": float(np.median(values)),
                "native_points_q95": float(np.quantile(values, 0.95)),
                "fraction_fewer_than_4_points": float(np.mean(values < 4)),
                "fraction_fewer_than_5_points": float(np.mean(values < 5)),
                "observations_overlapping_another_detected_interval": int(
                    np.count_nonzero(interval_overlap_flags)
                ),
                "fraction_overlapping_another_detected_interval": float(
                    np.mean(interval_overlap_flags)
                ),
                "passes_90pct_at_least_4_points": bool(np.mean(values < 4) <= 0.10),
            }
        )
    eligible = [
        float(row["half_width_factor"])
        for row in rows
        if row["passes_90pct_at_least_4_points"]
    ]
    if not eligible:
        raise RuntimeError("no q-width factor meets native-sampling requirement")
    selected = min(eligible)
    audit = {
        "objective": (
            "user prioritizes excluding neighboring signal: select the smallest "
            "tested half-width for which at least 90% of 519 observations "
            "retain four or more native spots-channel samples"
        ),
        "selected_half_width_factor": selected,
        "FWHM_like_reference_factor": 0.5,
        "selected_is_narrower_than_0p75": selected < 0.75,
        "candidate_count": len(rows),
    }
    return rows, selected, audit


PAIR_FIELDS = (
    "left_point_uid",
    "right_point_uid",
    "left_pressure_gpa",
    "right_pressure_gpa",
    "left_local_peak_index",
    "right_local_peak_index",
    "left_q",
    "right_q",
    "left_two_theta_deg",
    "right_two_theta_deg",
    "location_similarity",
    "ROI_profile_valid",
    "ROI_invalid_reason",
    "spots_qwidth_ROI_integrated_iou",
    "fixed_D1w_sensitivity_iou",
    "sensitivity_abs_difference",
    "left_integrated_area",
    "right_integrated_area",
)


def compute_pair_scores(
    points: Sequence[Mapping[str, Any]],
    main_profiles: Mapping[str, np.ndarray],
    sensitivity_profiles: Mapping[str, np.ndarray],
    relative_grid: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
]:
    count = len(points)
    location = np.full((count, count), np.nan, dtype=float)
    roi = np.full((count, count), np.nan, dtype=float)
    sensitivity = np.full((count, count), np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    differences: list[float] = []
    zero_pairs = 0
    invalid_pairs = 0
    zero_profile_uids = {
        uid
        for uid, profile in main_profiles.items()
        if float(np.trapezoid(profile, relative_grid)) <= 0.0
    }
    for left_index in range(count):
        left = points[left_index]
        left_uid = str(left["point_uid"])
        left_area = float(
            np.trapezoid(main_profiles[left_uid], relative_grid)
        )
        for right_index in range(left_index + 1, count):
            right = points[right_index]
            if float(left["pressure_gpa"]) == float(right["pressure_gpa"]):
                continue
            right_uid = str(right["point_uid"])
            right_area = float(
                np.trapezoid(main_profiles[right_uid], relative_grid)
            )
            loc = float(
                v6.location_similarity(
                    float(left["two_theta_deg"]),
                    float(right["two_theta_deg"]),
                )
            )
            profile_valid = (
                left_uid not in zero_profile_uids
                and right_uid not in zero_profile_uids
            )
            if profile_valid:
                score = profile_integrated_iou(
                    main_profiles[left_uid],
                    main_profiles[right_uid],
                    relative_grid,
                )
                sensitivity_score = profile_integrated_iou(
                    sensitivity_profiles[left_uid],
                    sensitivity_profiles[right_uid],
                    relative_grid,
                )
                if not np.isfinite(score) or not np.isfinite(sensitivity_score):
                    raise RuntimeError(
                        f"valid pair produced undefined score: "
                        f"{left_uid}, {right_uid}"
                    )
                difference = abs(score - sensitivity_score)
                differences.append(difference)
                zero_pairs += int(score == 0.0)
                for matrix, value in (
                    (roi, score),
                    (sensitivity, sensitivity_score),
                ):
                    matrix[left_index, right_index] = value
                    matrix[right_index, left_index] = value
                invalid_reason = ""
            else:
                score = math.nan
                sensitivity_score = math.nan
                difference = math.nan
                invalid_pairs += 1
                zero_sides = [
                    uid
                    for uid in (left_uid, right_uid)
                    if uid in zero_profile_uids
                ]
                invalid_reason = (
                    "no positive spots_channel signal in q-width core: "
                    + "|".join(zero_sides)
                )
            location[left_index, right_index] = loc
            location[right_index, left_index] = loc
            rows.append(
                {
                    "left_point_uid": left_uid,
                    "right_point_uid": right_uid,
                    "left_pressure_gpa": left["pressure_gpa"],
                    "right_pressure_gpa": right["pressure_gpa"],
                    "left_local_peak_index": left["local_peak_index"],
                    "right_local_peak_index": right["local_peak_index"],
                    "left_q": left["q"],
                    "right_q": right["q"],
                    "left_two_theta_deg": left["two_theta_deg"],
                    "right_two_theta_deg": right["two_theta_deg"],
                    "location_similarity": loc,
                    "ROI_profile_valid": int(profile_valid),
                    "ROI_invalid_reason": invalid_reason,
                    "spots_qwidth_ROI_integrated_iou": score,
                    "fixed_D1w_sensitivity_iou": sensitivity_score,
                    "sensitivity_abs_difference": difference,
                    "left_integrated_area": left_area,
                    "right_integrated_area": right_area,
                }
            )
    diff = np.asarray(differences, dtype=float)
    audit = {
        "cross_pressure_unordered_pairs": len(rows),
        "expected_cross_pressure_unordered_pairs": 37038,
        "same_pressure_pairs_omitted": count * (count - 1) // 2 - len(rows),
        "zero_profile_point_uids": sorted(zero_profile_uids),
        "zero_profile_points": len(zero_profile_uids),
        "ROI_invalid_cross_pressure_pairs": invalid_pairs,
        "ROI_valid_cross_pressure_pairs": len(rows) - invalid_pairs,
        "zero_profile_policy": (
            "every ROI cell involving a zero-positive-signal point is NaN/blank; "
            "location remains valid"
        ),
        "exact_zero_pairs": zero_pairs,
        "location_scores_in_0_1": bool(
            np.all(
                (location[np.isfinite(location)] >= 0.0)
                & (location[np.isfinite(location)] <= 1.0)
            )
        ),
        "ROI_scores_in_0_1": bool(
            np.all(
                (roi[np.isfinite(roi)] >= 0.0)
                & (roi[np.isfinite(roi)] <= 1.0)
            )
        ),
        "matrices_symmetric": bool(
            np.allclose(location, location.T, equal_nan=True)
            and np.allclose(roi, roi.T, equal_nan=True)
            and np.allclose(sensitivity, sensitivity.T, equal_nan=True)
        ),
        "fixed_D1w_sensitivity_abs_difference_median": (
            float(np.median(diff)) if diff.size else math.nan
        ),
        "fixed_D1w_sensitivity_abs_difference_q95": (
            float(np.quantile(diff, 0.95)) if diff.size else math.nan
        ),
        "fixed_D1w_sensitivity_abs_difference_max": (
            float(np.max(diff)) if diff.size else math.nan
        ),
    }
    if (
        len(rows) != 37038
        or not audit["location_scores_in_0_1"]
        or not audit["ROI_scores_in_0_1"]
        or not audit["matrices_symmetric"]
    ):
        raise RuntimeError(f"pair score audit failed: {audit}")
    return location, roi, sensitivity, rows, audit


def plot_roi_heatmap(
    path: Path,
    matrix: np.ndarray,
    pressure_layout: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
    *,
    half_width_factor: float,
) -> None:
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(12.5, 8.8))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    anchor_pressure = float(anchor["pressure_gpa"])
    anchor_row = v6.PRESSURES_DESCENDING.index(anchor_pressure)
    ax.set_title(
        "Powder spots-channel pressure-level comparison — "
        "q-width-core 1D integrated IoU\n"
        f"anchor {anchor['point_uid']} | {anchor_pressure:g} GPa, "
        f"peak {int(anchor['local_peak_index'])}, "
        f"2θ={float(anchor['two_theta_deg']):.4f}° | "
        f"u∈[-{half_width_factor:g}, {half_width_factor:g}], "
        "u=(q−qi)/q_width",
        fontsize=11,
    )
    ax.set_xlabel(
        "peak number within each target pressure (increasing 2θ; not a track ID)"
    )
    ax.set_ylabel("Pressure (GPa), descending")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [f"peak {index}" for index in range(1, matrix.shape[1] + 1)],
        rotation=55,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(pressure_layout)))
    ax.set_yticklabels(
        [
            (
                f"{float(row['pressure_gpa']):g} GPa"
                f" ({int(row['peak_count'])} peaks)"
                f"{' [anchor row omitted]' if index == anchor_row else ''}"
            )
            for index, row in enumerate(pressure_layout)
        ],
        fontsize=8,
    )
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1.0), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.25, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axhline(anchor_row - 0.5, color="#777777", linewidth=0.8)
    ax.axhline(anchor_row + 0.5, color="#777777", linewidth=0.8)
    fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.025,
        label="spots-channel q-width-core ROI integrated IoU",
    )
    fig.text(
        0.5,
        0.012,
        (
            "White = missing slot, omitted anchor-pressure row, or no positive "
            "spots-channel signal in the q-width core.\n"
            "Dark purple = valid similarity near 0; absolute qi is represented "
            "separately by location."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_anchor_outputs(
    output_root: Path,
    points: Sequence[Mapping[str, Any]],
    location_matrix: np.ndarray,
    roi_matrix: np.ndarray,
    pressure_layout: Sequence[Mapping[str, Any]],
    slot_lookup: Mapping[str, tuple[int, int]],
    maximum_slots: int,
    *,
    half_width_factor: float,
    make_plots: bool,
    max_anchors: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limit = len(points) if max_anchors is None else min(max_anchors, len(points))
    rows: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(points[:limit]):
        token = (
            f"anchor_{anchor_index:03d}_"
            f"P{v6.pressure_token(float(anchor['pressure_gpa']))}_"
            f"peak{int(anchor['local_peak_index']):02d}_"
            f"{anchor['point_uid']}"
        )
        location = v6.build_anchor_matrix(
            anchor_index,
            points,
            location_matrix,
            slot_lookup,
            maximum_slots,
        )
        roi = v6.build_anchor_matrix(
            anchor_index,
            points,
            roi_matrix,
            slot_lookup,
            maximum_slots,
        )
        expected_location_finite = len(points) - sum(
            float(point["pressure_gpa"]) == float(anchor["pressure_gpa"])
            for point in points
        )
        expected_finite = int(
            np.count_nonzero(np.isfinite(roi_matrix[anchor_index]))
        )
        if np.count_nonzero(np.isfinite(roi)) != expected_finite:
            raise RuntimeError(f"ROI anchor finite-count mismatch: {anchor['point_uid']}")
        anchor_row = v6.PRESSURES_DESCENDING.index(float(anchor["pressure_gpa"]))
        if np.any(np.isfinite(roi[anchor_row])):
            raise RuntimeError(f"anchor pressure row is not blank: {anchor['point_uid']}")
        roi_csv = (
            output_root
            / "peak_maps"
            / "roi_spots_qwidth_integrated_iou"
            / "matrices"
            / f"{token}.csv"
        )
        roi_png = (
            output_root
            / "peak_maps"
            / "roi_spots_qwidth_integrated_iou"
            / "heatmaps"
            / f"{token}.png"
        )
        v6.write_matrix_csv(roi_csv, roi, pressure_layout)
        if make_plots:
            plot_roi_heatmap(
                roi_png,
                roi,
                pressure_layout,
                anchor,
                half_width_factor=half_width_factor,
            )
        location_csv = (
            output_root
            / "peak_maps"
            / "location"
            / "matrices"
            / f"{token}.csv"
        )
        location_png = (
            output_root
            / "peak_maps"
            / "location"
            / "heatmaps"
            / f"{token}.png"
        )
        rows.append(
            {
                "anchor_index_0based": anchor_index,
                "point_uid": anchor["point_uid"],
                "pressure_gpa": anchor["pressure_gpa"],
                "local_peak_index": anchor["local_peak_index"],
                "q": anchor["q"],
                "two_theta_deg": anchor["two_theta_deg"],
                "n_observations": anchor["n_observations"],
                "distinct_frames": anchor["distinct_frames"],
                "finite_target_cells": expected_finite,
                "finite_location_target_cells": expected_location_finite,
                "anchor_ROI_profile_valid": int(expected_finite > 0),
                "invalid_ROI_target_cells_from_zero_profiles": (
                    expected_location_finite - expected_finite
                ),
                "location_matrix_csv": str(location_csv.relative_to(output_root)),
                "location_heatmap_png": str(location_png.relative_to(output_root)),
                "roi_matrix_csv": str(roi_csv.relative_to(output_root)),
                "roi_heatmap_png": (
                    str(roi_png.relative_to(output_root)) if make_plots else ""
                ),
            }
        )
        if (anchor_index + 1) % 25 == 0 or anchor_index + 1 == limit:
            _progress(f"wrote ROI anchor maps {anchor_index + 1}/{limit}")
    audit = {
        "anchors_written": limit,
        "complete_anchor_run": limit == len(points),
        "map_shape": [len(v6.PRESSURES_DESCENDING), maximum_slots],
        "ROI_matrix_csv_files": limit,
        "ROI_heatmap_png_files": limit if make_plots else 0,
        "location_matrix_csv_files": 280 if (output_root / "peak_maps/location").exists() else 0,
        "location_heatmap_png_files": 280 if (output_root / "peak_maps/location").exists() else 0,
        "same_pressure_entire_row_blank": True,
        "missing_slots_NaN": True,
    }
    return rows, audit


def copy_verified_unchanged_payloads(
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    marker = json.loads(
        (source_root / "RUN_COMPLETE.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (source_root / "validation_report.json").read_text(encoding="utf-8")
    )
    if (
        marker.get("status") != "complete"
        or marker.get("all_validation_checks_passed") is not True
        or validation.get("status") != "PASS"
    ):
        raise ValueError("v6 source is not a complete validated run")
    source_rows, source_digest = v6._payload_inventory(
        source_root, COPY_PAYLOADS
    )
    for relative in COPY_PAYLOADS:
        source = source_root / relative
        destination = output_root / relative
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite payload: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                copy_function=shutil.copy2,
                ignore=shutil.ignore_patterns(".DS_Store"),
            )
        else:
            shutil.copy2(source, destination)
    destination_rows, destination_digest = v6._payload_inventory(
        output_root, COPY_PAYLOADS
    )
    if source_rows != destination_rows or source_digest != destination_digest:
        raise RuntimeError("unchanged location/window copy failed SHA verification")
    v6.write_csv(
        output_root / "unchanged_location_window_sha256.csv",
        destination_rows,
        ["relative_path", "bytes", "sha256"],
    )
    location_files = sum(
        str(row["relative_path"]).startswith("peak_maps/location/")
        for row in destination_rows
    )
    window_files = len(destination_rows) - location_files
    audit = {
        "source_suite": str(source_root.resolve()),
        "reason": (
            "location formula and integer-window inputs/algorithms are unchanged "
            "by replacing only the ROI-area definition"
        ),
        "source_validation_status": validation.get("status"),
        "source_completion_status": marker.get("status"),
        "copied_files": len(destination_rows),
        "copied_bytes": sum(int(row["bytes"]) for row in destination_rows),
        "payload_digest": source_digest,
        "all_paths_sizes_sha256_match": True,
        "location_payload_files": location_files,
        "window_payload_files": window_files,
        "location_maps": 280,
        "location_matrices": 280,
        "window_across_frames_included": True,
        "window_within_frames_included": True,
        "window_sequence": "0-5, 1-6, ..., 27-32 degrees",
        "window_presentation": (
            "strict lower triangle only; diagonal and mirrored upper triangle omitted"
        ),
    }
    v6.write_json(output_root / "unchanged_payload_reuse_manifest.json", audit)
    return audit


def grid_convergence_audit(
    observations: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    normalization_by_frame: Mapping[int, Mapping[str, Any]],
    spots_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray, Path]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    half_width_factor: float,
    base_grid_points: int,
    sample_count: int = 128,
) -> dict[str, Any]:
    fine_points = [dict(point) for point in points]
    (
        fine_profiles,
        _fine_sensitivity,
        _fine_rows,
        _fine_profile_audit,
        fine_grid,
    ) = build_spots_point_profiles(
        observations,
        fine_points,
        normalization_by_frame,
        spots_by_frame,
        half_width_factor=half_width_factor,
        relative_grid_points=base_grid_points * 2 - 1,
    )
    valid_rows = [
        row for row in pair_rows if int(row["ROI_profile_valid"]) == 1
    ]
    indices = np.unique(
        np.linspace(
            0,
            len(valid_rows) - 1,
            min(sample_count, len(valid_rows)),
            dtype=int,
        )
    )
    differences: list[float] = []
    for index in indices:
        row = valid_rows[int(index)]
        fine = profile_integrated_iou(
            fine_profiles[str(row["left_point_uid"])],
            fine_profiles[str(row["right_point_uid"])],
            fine_grid,
        )
        differences.append(
            abs(fine - float(row["spots_qwidth_ROI_integrated_iou"]))
        )
    values = np.asarray(differences, dtype=float)
    audit = {
        "method": "rebuild all point profiles with doubled relative-grid resolution",
        "sample_pairs": len(values),
        "base_grid_points": base_grid_points,
        "fine_grid_points": base_grid_points * 2 - 1,
        "absolute_difference_median": float(np.median(values)),
        "absolute_difference_q95": float(np.quantile(values, 0.95)),
        "absolute_difference_max": float(np.max(values)),
        "q95_le_0p001": bool(np.quantile(values, 0.95) <= 0.001),
    }
    if not audit["q95_le_0p001"]:
        raise RuntimeError(f"1D integration grid convergence failed: {audit}")
    return audit


def _point_registry_rows(
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "point_index",
        "point_uid",
        "source_table",
        "source_row_0based",
        "track",
        "pressure_gpa",
        "local_peak_index",
        "d_A",
        "q",
        "two_theta_deg",
        "azim_deg",
        "intensity",
        "area",
        "n_observations",
        "distinct_frames",
        "best_frame",
        "best_frame_file",
        "obs_indices_0based",
        "frames",
        "spots_qwidth_integral_main",
        "spots_qwidth_integral_fixed_D1w_sensitivity",
    )
    return [{key: point.get(key) for key in fields} for point in points]


def _write_readme(
    path: Path,
    *,
    half_width_factor: float,
) -> None:
    path.write_text(
        f"""# UOTe spots-channel q-width-core correlation suite (v7)

This is the second requested result group.  It includes all four required
components: ROI area, location, window-to-window across frames, and
window-to-window within frames.

## Peak layout

There are 280 pressure-level anchors reconstructed from all 519 observations,
including all 52 untracked pressure-level peaks.  Each map has 19 pressure
rows (50.7 down to 3.5 GPa) and 22 possible local peak slots.  At every
pressure, peak numbers are reassigned by increasing 2theta.  Missing slots and
the complete anchor-pressure row are NaN/white.

## One-dimensional spots-channel ROI definition

No two-dimensional cake coordinate enters this ROI score.  For every
observation, the corresponding `spots_channel` curve is cropped to

`[qi - {half_width_factor:g}*q_width, qi + {half_width_factor:g}*q_width]`.

The crop is then recentered to `u=(q-qi)/q_width`.  Thus absolute qi does not
penalize ROI similarity; absolute peak position is handled only by the
separate location map.  The profile retains its physical d(2theta) integrated
area through the coordinate Jacobian.  Negative spots-channel residuals are
clipped to zero because min/max IoU requires nonnegative intensity.

Each frame is corrected by the same tungsten-dominated internal measurement
reference used in v6.  If one pressure-level point has multiple observation
rows in one physical frame, their nonoverlapping extracted q-core segments are
summed before that frame is counted once.  Distinct frames are averaged
equally.

The final score is:

`integral min(I1,I2) d(2theta) / integral max(I1,I2) d(2theta)`,

after both local peak intervals have been recentered onto the same relative
width coordinate.

Ten of the 280 pressure-level points have no positive `spots_channel` signal
inside this narrow core.  Per the requested policy, every ROI cell involving
one of those points is blank/NaN while its location cells remain available.
No artificial offset is added.  A numerical zero is reserved for two valid
profiles whose integrated overlap is genuinely zero.

The factor {half_width_factor:g} was selected as the narrowest tested core for
which at least 90% of observations retain four or more native spots-channel
samples.  This prioritizes excluding neighboring signal while avoiding the
severe undersampling of the 0.5 candidate.

## Location and windows

Location is unchanged:
`clip(1-abs(delta_2theta)/0.06 degrees,0,1)`.

Powder windows remain exactly 0-5, 1-6, ..., 27-32 degrees.  Across-frame
results compare pressure frames within the same scan before taking the median
across scans.  Within-frame results begin with each raw frame's 28 x 28
window matrix.  All user-facing window matrices keep only the strict lower
triangle; the diagonal and upper triangle are blank.  These unchanged,
previously audited artifacts were copied byte-for-byte from v6 and rechecked
by SHA256.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    started = time.time()
    output_root = args.out_dir.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output_root}")
    output_root.mkdir(parents=True)

    _progress("loading authoritative 519-to-280 mapping")
    observations, points, assignments, mapping_audit = (
        v6.assign_observations_to_points(
            args.observations,
            args.track_points,
            args.untracked_points,
        )
    )
    v6.write_csv(output_root / "observation_assignment.csv", assignments)

    _progress("loading all 1060 spots-channel profiles")
    spots_by_frame = _load_spots_profiles(args.spots_root)
    normalization_rows, normalization_by_frame, d1w_assessment = (
        v6.build_measurement_normalization(args.manifest, args.fit_root)
    )
    v6.write_csv(
        output_root / "frame_measurement_normalization.csv",
        normalization_rows,
    )

    optimization_rows, selected_factor, optimization_audit = (
        native_sampling_factor_optimization(observations, spots_by_frame)
    )
    v6.write_csv(
        output_root / "qwidth_factor_optimization.csv",
        optimization_rows,
    )
    if math.isclose(args.half_width_factor, HALF_WIDTH_FACTOR) and not math.isclose(
        selected_factor, args.half_width_factor
    ):
        raise RuntimeError(
            f"default factor no longer matches optimization: "
            f"{args.half_width_factor} != {selected_factor}"
        )

    _progress(
        f"building recentered spots-channel profiles with c={args.half_width_factor:g}"
    )
    (
        main_profiles,
        sensitivity_profiles,
        observation_profile_rows,
        profile_audit,
        relative_grid,
    ) = build_spots_point_profiles(
        observations,
        points,
        normalization_by_frame,
        spots_by_frame,
        half_width_factor=args.half_width_factor,
        relative_grid_points=args.relative_grid_points,
    )
    v6.write_csv(
        output_root / "observation_spots_qwidth_profile_audit.csv",
        observation_profile_rows,
    )
    v6.write_csv(output_root / "point_registry.csv", _point_registry_rows(points))

    pressure_layout, slot_lookup, maximum_slots = v6.build_pressure_slot_layout(points)
    v6.write_csv(output_root / "pressure_row_layout.csv", pressure_layout)
    pressure_grid_rows = sorted(
        (
            {
                "pressure_gpa": point["pressure_gpa"],
                "local_peak_index": point["local_peak_index"],
                "point_uid": point["point_uid"],
                "q": point["q"],
                "two_theta_deg": point["two_theta_deg"],
                "source_table": point["source_table"],
                "track": point["track"],
            }
            for point in points
        ),
        key=lambda row: (
            -float(row["pressure_gpa"]),
            int(row["local_peak_index"]),
        ),
    )
    v6.write_csv(output_root / "pressure_peak_grid.csv", pressure_grid_rows)

    _progress("computing 37,038 cross-pressure q-width-core ROI pairs")
    (
        location_matrix,
        roi_matrix,
        sensitivity_matrix,
        pair_rows,
        pair_audit,
    ) = compute_pair_scores(
        points,
        main_profiles,
        sensitivity_profiles,
        relative_grid,
    )
    pair_count = v6.write_csv_gz(
        output_root / "all_cross_pressure_peak_pairs.csv.gz",
        pair_rows,
        PAIR_FIELDS,
    )
    if pair_count != 37038:
        raise RuntimeError(f"pair row count mismatch: {pair_count}")
    np.savez_compressed(
        output_root / "pressure_level_similarity_matrices.npz",
        point_uids=np.asarray([str(point["point_uid"]) for point in points]),
        relative_grid=relative_grid,
        location=location_matrix,
        spots_qwidth_ROI_integrated_iou=roi_matrix,
        fixed_D1w_sensitivity_iou=sensitivity_matrix,
    )

    _progress("checking doubled-grid numerical convergence")
    convergence_audit = grid_convergence_audit(
        observations,
        points,
        normalization_by_frame,
        spots_by_frame,
        pair_rows,
        half_width_factor=args.half_width_factor,
        base_grid_points=args.relative_grid_points,
    )

    reuse_audit: dict[str, Any] | None = None
    if not args.no_copy:
        _progress("copying and SHA-verifying unchanged location/windows")
        reuse_audit = copy_verified_unchanged_payloads(
            args.v6_source.resolve(),
            output_root,
        )

    _progress("writing 280 per-anchor q-width-core ROI maps")
    anchor_rows, map_audit = write_anchor_outputs(
        output_root,
        points,
        location_matrix,
        roi_matrix,
        pressure_layout,
        slot_lookup,
        maximum_slots,
        half_width_factor=args.half_width_factor,
        make_plots=not args.no_plots,
        max_anchors=args.max_anchors,
    )
    v6.write_csv(output_root / "anchor_map_index.csv", anchor_rows)

    complete = (
        map_audit["complete_anchor_run"]
        and not args.no_plots
        and reuse_audit is not None
    )
    required = {
        "ROI_area": map_audit["ROI_heatmap_png_files"] == 280,
        "location": bool(reuse_audit and reuse_audit["location_maps"] == 280),
        "window_across_frames": bool(
            reuse_audit and reuse_audit["window_across_frames_included"]
        ),
        "window_within_frames": bool(
            reuse_audit and reuse_audit["window_within_frames_included"]
        ),
        "window_strict_lower_triangle_only": bool(
            reuse_audit
            and reuse_audit["window_presentation"].startswith("strict lower")
        ),
    }
    if complete and not all(required.values()):
        raise RuntimeError(f"required component missing: {required}")
    d1w_json = dict(d1w_assessment)
    d1w_json.pop("pair_rows")
    validation = {
        "status": "PASS" if complete else "PARTIAL_PASS",
        "complete_required_run": complete,
        "required_components": required,
        "source_counts": {
            "raw_observations": len(observations),
            "pressure_level_points": len(points),
            "tracked_points": 228,
            "untracked_points": 52,
            "spots_channel_frames": len(spots_by_frame),
        },
        "mapping": mapping_audit,
        "qwidth_factor_optimization": optimization_audit,
        "profile_definition": profile_audit,
        "measurement_normalization": d1w_json,
        "pair_scores": pair_audit,
        "grid_convergence": convergence_audit,
        "peak_maps": map_audit,
        "unchanged_location_windows": reuse_audit,
    }
    v6.write_json(output_root / "validation_report.json", validation)
    manifest = {
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "spots_root": str(args.spots_root.resolve()),
            "observations": str(args.observations.resolve()),
            "track_points": str(args.track_points.resolve()),
            "untracked_points": str(args.untracked_points.resolve()),
            "v6_source": str(args.v6_source.resolve()),
        },
        "parameters": {
            "half_width_factor": args.half_width_factor,
            "relative_grid_points": args.relative_grid_points,
            "relative_coordinate": "u=(q-qi)/q_width",
            "absolute_q_used_in_ROI_matching": False,
            "negative_spots_values": "clip_to_zero",
            "ROI_formula": "integral(min(I1,I2))/integral(max(I1,I2))",
            "integration_measure": "physical d(2theta) area retained via Jacobian",
            "pressure_order": "descending",
            "local_peak_order": "increasing 2theta independently per pressure",
            "window_presentation": "strict lower triangle; diagonal omitted",
        },
        "elapsed_seconds": time.time() - started,
        "output_root": str(output_root),
    }
    v6.write_json(output_root / "run_manifest.json", manifest)
    _write_readme(
        output_root / "README.md",
        half_width_factor=args.half_width_factor,
    )
    artifact_rows = v6.build_artifact_index(output_root)
    v6.write_csv(
        output_root / "artifact_index.csv",
        artifact_rows,
        ["relative_path", "bytes", "sha256"],
    )
    completion = {
        "status": "complete" if complete else "partial",
        "all_validation_checks_passed": complete,
        "required_components": required,
        "validation_report_sha256": v6.file_sha256(
            output_root / "validation_report.json"
        ),
        "artifact_index_sha256": v6.file_sha256(
            output_root / "artifact_index.csv"
        ),
        "artifact_count_excluding_index_and_marker": len(artifact_rows),
        "elapsed_seconds": time.time() - started,
    }
    v6.write_json(output_root / "RUN_COMPLETE.json", completion)
    _progress(
        f"finished status={completion['status']} at {output_root} "
        f"in {completion['elapsed_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
