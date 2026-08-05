#!/usr/bin/env python3
"""Directional absolute-q spots-channel ROI correlations (v8).

For every detected pressure-level peak, each raw observation contributes a
compact profile on its *absolute* q-width core

    [q_i - c*q_width_i, q_i + c*q_width_i].

The factor ``c`` is an explicit CLI parameter. The historical default is
0.6; the current formal UOTe q-width suite is reproduced with
``--half-width-factor 0.75``.

The q bounds are converted to absolute 2theta because the requested integral
is with respect to d(2theta) and the source ``spots_channel`` files are sampled
on that coordinate.  No recentering or width normalization is performed.

For an anchor A and target B, B is zero outside its own physical support and
the continuous min/max score is integrated only over the anchor support:

    S(A -> B) = integral_A min(J_A, J_B) d(2theta)
                ---------------------------------
                integral_A max(J_A, J_B) d(2theta).

This score is directional.  Disjoint supports, detected zero-signal peaks, and
zero denominators are represented by the finite value 0.  NaN is reserved for
structurally missing local slots and the deliberately omitted anchor-pressure
row in each user-facing heatmap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import nonlinear_intensity_preprocessing as nonlinear
import pressure_level_peak_spots_qwidth_correlations_v7 as legacy


v6 = legacy.v6
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SPOTS_ROOT = legacy.DEFAULT_SPOTS_ROOT
DEFAULT_V6_SOURCE = legacy.DEFAULT_V6_SOURCE
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "correlations"
    / "results"
    / (
        "uote_pressure_level_peak_spots_absolute_anchor_iou_"
        "integer_window_suite_20260730_v8"
    )
)
HALF_WIDTH_FACTOR = 0.6


@dataclass(frozen=True)
class AbsoluteComponent:
    """One clipped observation segment on absolute 2theta."""

    coordinate: np.ndarray
    positive_intensity: np.ndarray
    support: tuple[float, float]
    q_support: tuple[float, float]
    frame: int
    main_weight: float
    sensitivity_weight: float


@dataclass(frozen=True)
class AbsolutePointProfile:
    """A pressure-level peak assembled from frame-weighted components."""

    point_uid: str
    components: tuple[AbsoluteComponent, ...]
    support_intervals: tuple[tuple[float, float], ...]
    q_support_intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DirectedIoUResult:
    score: float
    numerator: float
    denominator: float
    supports_overlap: bool


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
    parser.add_argument(
        "--half-width-factor",
        type=float,
        default=HALF_WIDTH_FACTOR,
        help=(
            "Half-width factor c in [qi-c*q_width, qi+c*q_width]. "
            "Historical default: 0.6; current formal q-width suite: 0.75."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument(
        "--intensity-transform",
        choices=("none", *nonlinear.SUPPORTED_METHODS),
        default="none",
        help=(
            "Optional bounded squared-intensity preprocessing applied after "
            "the existing positive-residual clipping and frame measurement "
            "normalization, before point-profile aggregation."
        ),
    )
    parser.add_argument(
        "--transform-scale-quantile",
        type=float,
        default=nonlinear.DEFAULT_SCALE_QUANTILE,
    )
    parser.add_argument(
        "--transform-noise-floor",
        type=float,
        default=None,
        help="Required physical noise floor for log_squared; unused by exp_squared.",
    )
    return parser.parse_args()


def _progress(message: str) -> None:
    print(f"[spots-absolute-anchor-v8] {message}", flush=True)


q_from_two_theta = legacy.q_from_two_theta
two_theta_from_q = legacy.two_theta_from_q
native_sampling_factor_optimization = legacy.native_sampling_factor_optimization
build_anchor_matrix = v6.build_anchor_matrix


def _merge_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    tolerance: float = 1.0e-12,
) -> tuple[tuple[float, float], ...]:
    clean = sorted(
        (float(left), float(right))
        for left, right in intervals
        if np.isfinite(left) and np.isfinite(right) and right > left
    )
    if not clean:
        return ()
    merged: list[list[float]] = [[clean[0][0], clean[0][1]]]
    for left, right in clean[1:]:
        if left <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return tuple((left, right) for left, right in merged)


def _insert_zero_crossings(
    coordinate: np.ndarray,
    raw_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return knots that exactly represent max(piecewise-linear y, 0)."""
    x = np.asarray(coordinate, dtype=float)
    y = np.asarray(raw_values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("coordinate and raw values must be equal-shape 1D")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("component knots must be finite")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("component coordinate must be strictly increasing")
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


def _build_observation_component(
    q_axis: np.ndarray,
    raw_intensity: np.ndarray,
    *,
    q_center: float,
    q_width: float,
    half_width_factor: float,
    frame: int,
    main_weight: float,
    sensitivity_weight: float,
    transform_spec: nonlinear.ROITransformSpec | None = None,
) -> AbsoluteComponent:
    q = np.asarray(q_axis, dtype=float)
    raw = np.asarray(raw_intensity, dtype=float)
    if q.ndim != 1 or raw.shape != q.shape or q.size < 2:
        raise ValueError("invalid source q/intensity arrays")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("source q axis must be strictly increasing")
    lower_q = float(q_center - half_width_factor * q_width)
    upper_q = float(q_center + half_width_factor * q_width)
    if not np.isfinite(lower_q) or not np.isfinite(upper_q) or upper_q <= lower_q:
        raise ValueError("invalid observation q support")
    source_two_theta = two_theta_from_q(q)
    support_two_theta = two_theta_from_q(np.asarray([lower_q, upper_q]))
    lower_theta = float(support_two_theta[0])
    upper_theta = float(support_two_theta[1])
    native = (source_two_theta > lower_theta) & (source_two_theta < upper_theta)
    coordinate = np.concatenate(
        (
            np.asarray([lower_theta]),
            source_two_theta[native],
            np.asarray([upper_theta]),
        )
    )
    values = np.interp(coordinate, source_two_theta, raw, left=0.0, right=0.0)
    coordinate, positive = _insert_zero_crossings(coordinate, values)
    if transform_spec is not None:
        # The existing frame multiplier is part of the physical intensity
        # normalization and must precede a nonlinear mapping.  Once it has
        # been absorbed here, downstream component aggregation remains the
        # unchanged linear sum/mean with unit weights.
        physical_intensity = positive * float(main_weight)
        positive = np.asarray(
            transform_spec.transform(physical_intensity),
            dtype=float,
        )
        main_weight = 1.0
        # The fixed-D1w sensitivity branch is not a requested transformed
        # result.  Keep the legacy archive shape while making that diagnostic
        # an explicit duplicate of the transformed primary profile.
        sensitivity_weight = 1.0
    return AbsoluteComponent(
        coordinate=coordinate,
        positive_intensity=positive,
        support=(lower_theta, upper_theta),
        q_support=(lower_q, upper_q),
        frame=int(frame),
        main_weight=float(main_weight),
        sensitivity_weight=float(sensitivity_weight),
    )


def _component_values_on_segment(
    component: AbsoluteComponent,
    left: float,
    right: float,
    *,
    mode: str,
) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    if not (component.support[0] < midpoint < component.support[1]):
        return np.zeros(2, dtype=float)
    weight = (
        component.main_weight if mode == "main" else component.sensitivity_weight
    )
    return (
        np.interp(
            np.asarray([left, right]),
            component.coordinate,
            component.positive_intensity,
        )
        * weight
    )


def _point_values_on_segment(
    profile: AbsolutePointProfile,
    left: float,
    right: float,
    *,
    mode: str,
) -> np.ndarray:
    result = np.zeros(2, dtype=float)
    for component in profile.components:
        result += _component_values_on_segment(
            component,
            left,
            right,
            mode=mode,
        )
    return result


def _segment_minmax_integrals(
    left: float,
    right: float,
    anchor_values: np.ndarray,
    target_values: np.ndarray,
) -> tuple[float, float]:
    """Exact min/max integrals for two linear functions on one segment."""
    a0, a1 = (float(value) for value in anchor_values)
    b0, b1 = (float(value) for value in target_values)
    if min(a0, a1, b0, b1) < -1.0e-12:
        raise ValueError("profiles must be nonnegative")
    width = float(right - left)
    if width <= 0.0:
        return 0.0, 0.0
    difference0 = a0 - b0
    difference1 = a1 - b1
    if difference0 * difference1 < 0.0:
        fraction = difference0 / (difference0 - difference1)
        anchor_cross = a0 + fraction * (a1 - a0)
        target_cross = b0 + fraction * (b1 - b0)
        crossing_value = 0.5 * (anchor_cross + target_cross)
        first_width = fraction * width
        second_width = (1.0 - fraction) * width
        if difference0 < 0.0:
            first_minimum = 0.5 * first_width * (a0 + crossing_value)
            first_maximum = 0.5 * first_width * (b0 + crossing_value)
        else:
            first_minimum = 0.5 * first_width * (b0 + crossing_value)
            first_maximum = 0.5 * first_width * (a0 + crossing_value)
        if difference1 < 0.0:
            second_minimum = 0.5 * second_width * (crossing_value + a1)
            second_maximum = 0.5 * second_width * (crossing_value + b1)
        else:
            second_minimum = 0.5 * second_width * (crossing_value + b1)
            second_maximum = 0.5 * second_width * (crossing_value + a1)
        return (
            float(first_minimum + second_minimum),
            float(first_maximum + second_maximum),
        )
    intersection = 0.5 * width * (
        min(a0, b0) + min(a1, b1)
    )
    union = 0.5 * width * (
        max(a0, b0) + max(a1, b1)
    )
    return float(intersection), float(union)


def _supports_overlap(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> bool:
    return any(
        min(left_end, right_end) > max(left_start, right_start) + 1.0e-14
        for left_start, left_end in left
        for right_start, right_end in right
    )


def _integration_knots(
    anchor: AbsolutePointProfile,
    target: AbsolutePointProfile,
    domain: tuple[float, float],
) -> np.ndarray:
    lower, upper = domain
    knots: list[float] = [lower, upper]
    for component in (*anchor.components, *target.components):
        knots.extend(
            float(value)
            for value in component.coordinate
            if lower < float(value) < upper
        )
    result = np.unique(np.asarray(knots, dtype=float))
    if result.size < 2 or result[0] != lower or result[-1] != upper:
        raise RuntimeError("failed to construct anchor-domain integration knots")
    return result


def directed_profile_iou(
    anchor: AbsolutePointProfile,
    target: AbsolutePointProfile,
    *,
    mode: str = "main",
    anchor_area: float | None = None,
) -> DirectedIoUResult:
    """Exact piecewise-linear IoU integrated only on the anchor support."""
    if mode not in {"main", "sensitivity"}:
        raise ValueError("mode must be 'main' or 'sensitivity'")
    overlap = _supports_overlap(
        anchor.support_intervals,
        target.support_intervals,
    )
    if not overlap and anchor_area is not None:
        return DirectedIoUResult(
            score=0.0,
            numerator=0.0,
            denominator=float(anchor_area),
            supports_overlap=False,
        )
    numerator = 0.0
    denominator = 0.0
    for domain in anchor.support_intervals:
        knots = _integration_knots(anchor, target, domain)
        for left, right in zip(knots[:-1], knots[1:]):
            anchor_values = _point_values_on_segment(
                anchor,
                float(left),
                float(right),
                mode=mode,
            )
            target_values = _point_values_on_segment(
                target,
                float(left),
                float(right),
                mode=mode,
            )
            intersection, union = _segment_minmax_integrals(
                float(left),
                float(right),
                anchor_values,
                target_values,
            )
            numerator += intersection
            denominator += union
    score = 0.0 if denominator <= 0.0 else numerator / denominator
    if score < -1.0e-12 or score > 1.0 + 1.0e-12:
        raise RuntimeError(f"directional ROI score outside [0,1]: {score}")
    return DirectedIoUResult(
        score=float(min(1.0, max(0.0, score))),
        numerator=float(max(0.0, numerator)),
        denominator=float(max(0.0, denominator)),
        supports_overlap=overlap,
    )


def _integrate_point_area(
    profile: AbsolutePointProfile,
    *,
    mode: str,
) -> float:
    empty = AbsolutePointProfile(
        point_uid="__zero__",
        components=(),
        support_intervals=(),
        q_support_intervals=(),
    )
    return directed_profile_iou(profile, empty, mode=mode).denominator


def anchor_restricted_integrated_iou(
    coordinate: np.ndarray,
    anchor_profile: np.ndarray,
    target_profile: np.ndarray,
    *,
    anchor_support: tuple[float, float],
    target_support: tuple[float, float],
) -> float:
    """Synthetic-grid helper implementing the public directional contract."""
    x = np.asarray(coordinate, dtype=float)
    anchor = np.asarray(anchor_profile, dtype=float)
    target = np.asarray(target_profile, dtype=float)
    if x.ndim != 1 or anchor.shape != x.shape or target.shape != x.shape:
        raise ValueError("coordinate and profiles must be equal-shape 1D")
    if x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("coordinate must be strictly increasing")
    if np.any(~np.isfinite(anchor)) or np.any(~np.isfinite(target)):
        raise ValueError("profiles must be finite")
    if np.any(anchor < 0.0) or np.any(target < 0.0):
        raise ValueError("profiles must be nonnegative")
    anchor_lower, anchor_upper = (float(value) for value in anchor_support)
    target_lower, target_upper = (float(value) for value in target_support)
    if anchor_upper <= anchor_lower or target_upper <= target_lower:
        raise ValueError("supports must have positive width")
    knots = np.unique(
        np.concatenate(
            (
                x[(x > anchor_lower) & (x < anchor_upper)],
                np.asarray([anchor_lower, anchor_upper]),
                np.asarray(
                    [
                        value
                        for value in (target_lower, target_upper)
                        if anchor_lower < value < anchor_upper
                    ]
                ),
            )
        )
    )
    numerator = 0.0
    denominator = 0.0
    for left, right in zip(knots[:-1], knots[1:]):
        midpoint = 0.5 * (left + right)
        anchor_values = np.interp(
            np.asarray([left, right]), x, anchor, left=0.0, right=0.0
        )
        if not (anchor_lower < midpoint < anchor_upper):
            anchor_values[:] = 0.0
        target_values = np.interp(
            np.asarray([left, right]), x, target, left=0.0, right=0.0
        )
        if not (target_lower < midpoint < target_upper):
            target_values[:] = 0.0
        intersection, union = _segment_minmax_integrals(
            float(left),
            float(right),
            anchor_values,
            target_values,
        )
        numerator += intersection
        denominator += union
    return 0.0 if denominator <= 0.0 else float(numerator / denominator)


def build_absolute_point_profiles(
    observations: Sequence[Mapping[str, Any]],
    points: Sequence[dict[str, Any]],
    normalization_by_frame: Mapping[int, Mapping[str, Any]],
    spots_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray, Path]],
    *,
    half_width_factor: float,
    transform_spec: nonlinear.ROITransformSpec | None = None,
) -> tuple[
    dict[str, AbsolutePointProfile],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not (0.0 < half_width_factor <= 2.0):
        raise ValueError("half-width factor must be in (0,2]")
    obs_by_index = {
        int(observation["obs_index_0based"]): observation
        for observation in observations
    }
    observations_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        observations_by_frame[int(observation["frame"])].append(observation)
    components: dict[int, AbsoluteComponent] = {}
    audit_rows: list[dict[str, Any]] = []
    native_counts: list[int] = []
    negative_fractions: list[float] = []
    neighbor_overlap_count = 0
    for observation in observations:
        obs_index = int(observation["obs_index_0based"])
        frame = int(observation["frame"])
        q_axis, raw_intensity, path = spots_by_frame[frame]
        q_center = float(observation["q"])
        q_width = float(observation["q_width"])
        lower_q = q_center - half_width_factor * q_width
        upper_q = q_center + half_width_factor * q_width
        native = (q_axis >= lower_q) & (q_axis <= upper_q)
        native_values = raw_intensity[native]
        native_count = int(np.count_nonzero(native))
        native_counts.append(native_count)
        negative_fraction = (
            float(np.mean(native_values < 0.0)) if native_values.size else math.nan
        )
        negative_fractions.append(negative_fraction)
        overlapping_neighbors: list[int] = []
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
            if min(upper_q, neighbor_upper) > max(lower_q, neighbor_lower):
                overlapping_neighbors.append(neighbor_index)
        neighbor_overlap_count += int(bool(overlapping_neighbors))
        normalization = normalization_by_frame[frame]
        component = _build_observation_component(
            q_axis,
            raw_intensity,
            q_center=q_center,
            q_width=q_width,
            half_width_factor=half_width_factor,
            frame=frame,
            main_weight=float(normalization["main_area_multiplier"]),
            sensitivity_weight=float(
                normalization["fixed_token_sensitivity_multiplier"]
            ),
            transform_spec=transform_spec,
        )
        components[obs_index] = component
        audit_rows.append(
            {
                "obs_index_0based": obs_index,
                "point_uid": observation["point_uid"],
                "frame": frame,
                "q_center": q_center,
                "q_width": q_width,
                "half_width_factor": half_width_factor,
                "q_lower": component.q_support[0],
                "q_upper": component.q_support[1],
                "two_theta_lower_deg": component.support[0],
                "two_theta_upper_deg": component.support[1],
                "native_points_in_interval": native_count,
                "native_negative_fraction": negative_fraction,
                "negative_values_clipped_to_zero": 1,
                "overlaps_another_detected_peak_interval_in_same_frame": int(
                    bool(overlapping_neighbors)
                ),
                "overlapping_neighbor_obs_indices": "|".join(
                    str(index) for index in sorted(overlapping_neighbors)
                ),
                "main_measurement_multiplier": component.main_weight,
                "original_main_measurement_multiplier": float(
                    normalization["main_area_multiplier"]
                ),
                "fixed_D1w_sensitivity_multiplier": (
                    component.sensitivity_weight
                ),
                "spots_channel_file": str(path.resolve()),
            }
        )
    profiles: dict[str, AbsolutePointProfile] = {}
    zero_integral_points: list[str] = []
    for point in points:
        by_frame: dict[int, list[int]] = defaultdict(list)
        for obs_index in point["member_obs_indices"]:
            frame = int(obs_by_index[int(obs_index)]["frame"])
            by_frame[frame].append(int(obs_index))
        frame_count = len(by_frame)
        point_components = tuple(
            replace(
                components[obs_index],
                main_weight=components[obs_index].main_weight / frame_count,
                sensitivity_weight=(
                    components[obs_index].sensitivity_weight / frame_count
                ),
            )
            for frame_items in by_frame.values()
            for obs_index in frame_items
        )
        uid = str(point["point_uid"])
        profile = AbsolutePointProfile(
            point_uid=uid,
            components=point_components,
            support_intervals=_merge_intervals(
                [component.support for component in point_components]
            ),
            q_support_intervals=_merge_intervals(
                [component.q_support for component in point_components]
            ),
        )
        profiles[uid] = profile
        main_area = _integrate_point_area(profile, mode="main")
        sensitivity_area = _integrate_point_area(profile, mode="sensitivity")
        point["spots_absolute_integral_main"] = main_area
        point["spots_absolute_integral_fixed_D1w_sensitivity"] = sensitivity_area
        point["support_component_count"] = len(profile.support_intervals)
        point["support_q_min"] = min(
            interval[0] for interval in profile.q_support_intervals
        )
        point["support_q_max"] = max(
            interval[1] for interval in profile.q_support_intervals
        )
        if main_area <= 0.0:
            zero_integral_points.append(uid)
    counts = np.asarray(native_counts, dtype=int)
    profile_audit = {
        "source_channel": "spots_channel",
        "two_dimensional_cake_coordinates_used": False,
        "absolute_q_position_used_in_ROI_score": True,
        "recentered_or_width_normalized": False,
        "directional_anchor_domain": True,
        "half_width_factor": half_width_factor,
        "interval_formula": "[qi-c*q_width, qi+c*q_width]",
        "integration_coordinate": "absolute 2theta converted from q support",
        "integration_measure": "d(2theta)",
        "target_outside_own_support": "exactly zero",
        "negative_values": "clip piecewise-linear residual to zero",
        "same_frame_multiple_observations": "sum components",
        "across_frames": "arithmetic mean over distinct physical frames",
        "intensity_transform": (
            "none" if transform_spec is None else transform_spec.method
        ),
        "transform_after_measurement_normalization": bool(
            transform_spec is not None
        ),
        "native_points_min": int(np.min(counts)),
        "native_points_q05": float(np.quantile(counts, 0.05)),
        "native_points_median": float(np.median(counts)),
        "native_points_q95": float(np.quantile(counts, 0.95)),
        "fraction_with_fewer_than_4_native_points": float(np.mean(counts < 4)),
        "observations_whose_1D_interval_overlaps_another_detected_peak": (
            neighbor_overlap_count
        ),
        "point_profiles": len(profiles),
        "zero_integral_points": zero_integral_points,
        "zero_integral_point_count": len(zero_integral_points),
    }
    if len(profiles) != 280:
        raise RuntimeError(f"expected 280 absolute point profiles: {profile_audit}")
    return profiles, audit_rows, profile_audit


DIRECTED_PAIR_FIELDS = (
    "anchor_point_uid",
    "target_point_uid",
    "anchor_pressure_gpa",
    "target_pressure_gpa",
    "anchor_local_peak_index",
    "target_local_peak_index",
    "anchor_q",
    "target_q",
    "anchor_two_theta_deg",
    "target_two_theta_deg",
    "location_similarity",
    "supports_overlap",
    "anchor_has_positive_signal",
    "target_has_positive_signal",
    "spots_absolute_anchor_ROI_iou",
    "intersection_integral",
    "anchor_domain_union_integral",
    "zero_reason",
    "fixed_D1w_sensitivity_iou",
    "sensitivity_abs_difference",
)


def compute_directed_pair_scores(
    points: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, AbsolutePointProfile],
) -> tuple[
    np.ndarray,
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
    support_overlap = np.zeros((count, count), dtype=bool)
    main_areas = {
        uid: _integrate_point_area(profile, mode="main")
        for uid, profile in profiles.items()
    }
    sensitivity_areas = {
        uid: _integrate_point_area(profile, mode="sensitivity")
        for uid, profile in profiles.items()
    }
    rows: list[dict[str, Any]] = []
    differences: list[float] = []
    zeros = 0
    positives = 0
    overlaps = 0
    disjoint = 0
    zero_denominators = 0
    overlap_zero_scores = 0
    for anchor_index, anchor_point in enumerate(points):
        anchor_uid = str(anchor_point["point_uid"])
        anchor_profile = profiles[anchor_uid]
        for target_index, target_point in enumerate(points):
            if anchor_index == target_index or float(
                anchor_point["pressure_gpa"]
            ) == float(target_point["pressure_gpa"]):
                continue
            target_uid = str(target_point["point_uid"])
            target_profile = profiles[target_uid]
            loc = float(
                v6.location_similarity(
                    float(anchor_point["two_theta_deg"]),
                    float(target_point["two_theta_deg"]),
                )
            )
            main = directed_profile_iou(
                anchor_profile,
                target_profile,
                mode="main",
                anchor_area=main_areas[anchor_uid],
            )
            fixed = directed_profile_iou(
                anchor_profile,
                target_profile,
                mode="sensitivity",
                anchor_area=sensitivity_areas[anchor_uid],
            )
            if not np.isfinite(main.score) or not np.isfinite(fixed.score):
                raise RuntimeError(
                    f"nonfinite directed score: {anchor_uid}->{target_uid}"
                )
            if main.supports_overlap != fixed.supports_overlap:
                raise RuntimeError("support geometry changed between modes")
            difference = abs(main.score - fixed.score)
            differences.append(difference)
            location[anchor_index, target_index] = loc
            roi[anchor_index, target_index] = main.score
            sensitivity[anchor_index, target_index] = fixed.score
            support_overlap[anchor_index, target_index] = main.supports_overlap
            zeros += int(main.score == 0.0)
            positives += int(main.score > 0.0)
            overlaps += int(main.supports_overlap)
            disjoint += int(not main.supports_overlap)
            zero_denominators += int(main.denominator <= 0.0)
            overlap_zero_scores += int(
                main.supports_overlap and main.score == 0.0
            )
            zero_reason = ""
            if not main.supports_overlap:
                zero_reason = "absolute supports disjoint"
            elif main.denominator <= 0.0:
                zero_reason = "anchor-domain denominator is zero; defined as 0"
            elif main.score == 0.0:
                zero_reason = "no positive integrated overlap"
            rows.append(
                {
                    "anchor_point_uid": anchor_uid,
                    "target_point_uid": target_uid,
                    "anchor_pressure_gpa": anchor_point["pressure_gpa"],
                    "target_pressure_gpa": target_point["pressure_gpa"],
                    "anchor_local_peak_index": anchor_point["local_peak_index"],
                    "target_local_peak_index": target_point["local_peak_index"],
                    "anchor_q": anchor_point["q"],
                    "target_q": target_point["q"],
                    "anchor_two_theta_deg": anchor_point["two_theta_deg"],
                    "target_two_theta_deg": target_point["two_theta_deg"],
                    "location_similarity": loc,
                    "supports_overlap": int(main.supports_overlap),
                    "anchor_has_positive_signal": int(
                        main_areas[anchor_uid] > 0.0
                    ),
                    "target_has_positive_signal": int(
                        main_areas[target_uid] > 0.0
                    ),
                    "spots_absolute_anchor_ROI_iou": main.score,
                    "intersection_integral": main.numerator,
                    "anchor_domain_union_integral": main.denominator,
                    "zero_reason": zero_reason,
                    "fixed_D1w_sensitivity_iou": fixed.score,
                    "sensitivity_abs_difference": difference,
                }
            )
    finite_roi = roi[np.isfinite(roi)]
    diff = np.asarray(differences, dtype=float)
    cross_pressure_entries = len(rows)
    asymmetry = np.abs(roi - roi.T)
    finite_asymmetry = asymmetry[np.isfinite(asymmetry)]
    audit = {
        "directed_cross_pressure_pairs": cross_pressure_entries,
        "expected_directed_cross_pressure_pairs": 74076,
        "directed_positive_scores": positives,
        "directed_exact_zero_scores": zeros,
        "support_overlap_directed_pairs": overlaps,
        "support_disjoint_directed_pairs": disjoint,
        "overlapping_support_zero_scores": overlap_zero_scores,
        "zero_denominator_pairs_defined_as_zero": zero_denominators,
        "all_cross_pressure_scores_finite": bool(finite_roi.size == 74076),
        "ROI_scores_in_0_1": bool(
            np.all((finite_roi >= 0.0) & (finite_roi <= 1.0))
        ),
        "all_disjoint_support_scores_zero": bool(
            np.all(roi[(~support_overlap) & np.isfinite(roi)] == 0.0)
        ),
        "directional_not_forced_symmetric": True,
        "directional_asymmetry_max": float(np.max(finite_asymmetry)),
        "directional_asymmetry_nonzero_pairs": int(
            np.count_nonzero(finite_asymmetry > 1.0e-12)
        ),
        "location_matrix_symmetric": bool(
            np.allclose(location, location.T, equal_nan=True)
        ),
        "fixed_D1w_sensitivity_abs_difference_median": float(
            np.median(diff)
        ),
        "fixed_D1w_sensitivity_abs_difference_q95": float(
            np.quantile(diff, 0.95)
        ),
        "fixed_D1w_sensitivity_abs_difference_max": float(np.max(diff)),
        "blank_policy": (
            "NaN only for same-pressure/structurally omitted comparisons; "
            "every detected cross-pressure target is finite"
        ),
        "zero_policy": (
            "disjoint support, zero positive signal, or zero denominator is "
            "the finite value 0"
        ),
    }
    if (
        cross_pressure_entries != 74076
        or not audit["all_cross_pressure_scores_finite"]
        or not audit["ROI_scores_in_0_1"]
        or not audit["all_disjoint_support_scores_zero"]
        or not audit["location_matrix_symmetric"]
    ):
        raise RuntimeError(f"directed pair score audit failed: {audit}")
    return location, roi, sensitivity, support_overlap, rows, audit


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
        "support_component_count",
        "support_q_min",
        "support_q_max",
        "spots_absolute_integral_main",
        "spots_absolute_integral_fixed_D1w_sensitivity",
    )
    return [{field: point.get(field) for field in fields} for point in points]


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
        "Powder spots-channel ROI — absolute-q anchor-domain integrated IoU\n"
        f"anchor {anchor['point_uid']} | {anchor_pressure:g} GPa, "
        f"peak {int(anchor['local_peak_index'])}, "
        f"2θ={float(anchor['two_theta_deg']):.4f}° | "
        f"q core = qi ± {half_width_factor:g} q_width; no recentering",
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
        label="directional ROI similarity S(anchor→target)",
    )
    fig.text(
        0.5,
        0.012,
        (
            "White = missing local peak slot or omitted anchor-pressure row.\n"
            "Dark purple = numeric 0: disjoint absolute-q supports, no positive "
            "overlap, or a detected zero-signal peak."
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
    support_overlap_matrix: np.ndarray,
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
    total_finite = 0
    total_zero = 0
    total_positive = 0
    for anchor_index, anchor in enumerate(points[:limit]):
        token = (
            f"anchor_{anchor_index:03d}_"
            f"P{v6.pressure_token(float(anchor['pressure_gpa']))}_"
            f"peak{int(anchor['local_peak_index']):02d}_"
            f"{anchor['point_uid']}"
        )
        roi = v6.build_anchor_matrix(
            anchor_index,
            points,
            roi_matrix,
            slot_lookup,
            maximum_slots,
        )
        expected_finite = len(points) - sum(
            float(point["pressure_gpa"]) == float(anchor["pressure_gpa"])
            for point in points
        )
        finite = int(np.count_nonzero(np.isfinite(roi)))
        if finite != expected_finite:
            raise RuntimeError(
                f"ROI finite-count mismatch for {anchor['point_uid']}: "
                f"{finite} != {expected_finite}"
            )
        anchor_row = v6.PRESSURES_DESCENDING.index(float(anchor["pressure_gpa"]))
        if np.any(np.isfinite(roi[anchor_row])):
            raise RuntimeError(f"anchor row not blank: {anchor['point_uid']}")
        zero_count = int(np.count_nonzero(roi[np.isfinite(roi)] == 0.0))
        positive_count = int(np.count_nonzero(roi[np.isfinite(roi)] > 0.0))
        overlap_count = int(
            np.count_nonzero(support_overlap_matrix[anchor_index])
        )
        disjoint_count = expected_finite - overlap_count
        total_finite += finite
        total_zero += zero_count
        total_positive += positive_count
        roi_csv = (
            output_root
            / "peak_maps"
            / "roi_spots_absolute_anchor_integrated_iou"
            / "matrices"
            / f"{token}.csv"
        )
        roi_png = (
            output_root
            / "peak_maps"
            / "roi_spots_absolute_anchor_integrated_iou"
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
                "finite_target_cells": finite,
                "positive_target_cells": positive_count,
                "zero_target_cells": zero_count,
                "overlapping_support_target_cells": overlap_count,
                "disjoint_support_target_cells": disjoint_count,
                "location_matrix_csv": str(location_csv.relative_to(output_root)),
                "location_heatmap_png": str(location_png.relative_to(output_root)),
                "roi_matrix_csv": str(roi_csv.relative_to(output_root)),
                "roi_heatmap_png": (
                    str(roi_png.relative_to(output_root)) if make_plots else ""
                ),
            }
        )
        if (anchor_index + 1) % 25 == 0 or anchor_index + 1 == limit:
            _progress(f"wrote directed ROI anchor maps {anchor_index + 1}/{limit}")
    audit = {
        "anchors_written": limit,
        "complete_anchor_run": limit == len(points),
        "map_shape": [len(v6.PRESSURES_DESCENDING), maximum_slots],
        "ROI_matrix_csv_files": limit,
        "ROI_heatmap_png_files": limit if make_plots else 0,
        "location_matrix_csv_files": (
            280 if (output_root / "peak_maps/location").exists() else 0
        ),
        "location_heatmap_png_files": (
            280 if (output_root / "peak_maps/location").exists() else 0
        ),
        "same_pressure_entire_row_blank": True,
        "missing_slots_NaN": True,
        "detected_cross_pressure_cells_all_finite": True,
        "total_finite_detected_cross_pressure_cells": total_finite,
        "total_exact_zero_cells": total_zero,
        "total_positive_cells": total_positive,
        "white_semantics": (
            "only missing local slots and the complete anchor-pressure row"
        ),
    }
    return rows, audit


def _write_readme(
    path: Path,
    *,
    half_width_factor: float,
    transform_spec: nonlinear.ROITransformSpec | None = None,
) -> None:
    transform_text = (
        "No nonlinear intensity transform is applied."
        if transform_spec is None
        else (
            "Before the unchanged profile aggregation and directional IoU, "
            "the positive spots residual is multiplied by the existing "
            "frame measurement-normalization factor and mapped with the "
            f"documented `{transform_spec.method}` preprocessing. Its fixed "
            f"pooled scale is {transform_spec.scale:.12g} and its logarithmic "
            f"epsilon is {transform_spec.epsilon!r}. The transform changes "
            "intensity only; q positions, q-width supports, peak identities, "
            "pressure rows, and local 2theta ordering are frozen."
        )
    )
    path.write_text(
        f"""# UOTe absolute-q directional anchor-domain ROI suite (v8)

This corrected suite implements the requested *absolute interval* definition.
It does not recenter peaks.  Every observation is supported only on

`[qi - {half_width_factor:g}*q_width, qi + {half_width_factor:g}*q_width]`.

For each anchor A and target B, the target is exactly zero outside its own
support and the continuous min/max integral is evaluated only over A's
support:

`S(A->B) = integral_A min(JA,JB) d(2theta) / integral_A max(JA,JB) d(2theta)`.

The score is directional: swapping anchor and target can change the value.
Therefore all 74,076 ordered cross-pressure comparisons are stored.  Disjoint
supports produce the exact numeric value 0.  Detected peaks with no positive
spots-channel residual also remain numeric 0; a zero denominator is
conservatively defined as 0.  White/NaN is reserved for a structurally missing
local peak slot or the intentionally omitted complete anchor-pressure row.

All 519 observations are retained and mapped to 280 pressure-level peaks,
including all 52 untracked points.  Same-frame observation components are
summed; distinct physical frames are equally averaged after the existing
measurement normalization.  Negative spots-channel residuals are clipped to
zero.

{transform_text}

The horizontal axis is the local peak number independently assigned by
increasing 2theta within every pressure.  The vertical axis is the 19 pressure
levels from 50.7 down to 3.5 GPa.

Location and the required window-to-window results are unchanged from v6 and
copied with SHA256 verification.  Windows are exactly 0-5, 1-6, ..., 27-32
degrees.  User-facing across-frame and within-frame matrices contain only the
strict lower triangle; diagonal and mirrored upper triangle are omitted.
""",
        encoding="utf-8",
    )


def _regression_oracle(
    points: Sequence[Mapping[str, Any]],
    roi: np.ndarray,
    pair_audit: Mapping[str, Any],
) -> dict[str, Any]:
    indices = {
        str(point["point_uid"]): index for index, point in enumerate(points)
    }
    anchor = indices["T00_P5p81"]
    checks = {
        "T00_P5p81_to_U36_P33p6_is_zero": (
            float(roi[anchor, indices["U36_P33p6"]]) == 0.0
        ),
        "T00_P5p81_to_T15_P50p7_is_zero": (
            float(roi[anchor, indices["T15_P50p7"]]) == 0.0
        ),
        "anchor_007_finite_targets": int(
            np.count_nonzero(np.isfinite(roi[anchor]))
        ),
        "anchor_007_positive_targets": int(
            np.count_nonzero(roi[anchor][np.isfinite(roi[anchor])] > 0.0)
        ),
        "anchor_007_zero_targets": int(
            np.count_nonzero(roi[anchor][np.isfinite(roi[anchor])] == 0.0)
        ),
        "anchor_007_expected_finite": 266,
        "anchor_007_expected_positive": 67,
        "anchor_007_expected_zero": 199,
        "global_expected_positive": 8640,
        "global_expected_zero": 65436,
        "global_expected_support_overlap": 9094,
        "global_expected_support_disjoint": 64982,
    }
    checks["anchor_007_counts_match"] = (
        checks["anchor_007_finite_targets"] == 266
        and checks["anchor_007_positive_targets"] == 67
        and checks["anchor_007_zero_targets"] == 199
    )
    checks["global_oracle_counts_match"] = (
        int(pair_audit["directed_positive_scores"]) == 8640
        and int(pair_audit["directed_exact_zero_scores"]) == 65436
        and int(pair_audit["support_overlap_directed_pairs"]) == 9094
        and int(pair_audit["support_disjoint_directed_pairs"]) == 64982
    )
    checks["all_regressions_pass"] = all(
        value
        for key, value in checks.items()
        if key.endswith("_is_zero")
        or key.endswith("_counts_match")
    )
    if not checks["all_regressions_pass"]:
        raise RuntimeError(f"v8 regression oracle failed: {checks}")
    return checks


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
    spots_by_frame = legacy._load_spots_profiles(args.spots_root)
    normalization_rows, normalization_by_frame, d1w_assessment = (
        v6.build_measurement_normalization(args.manifest, args.fit_root)
    )
    v6.write_csv(
        output_root / "frame_measurement_normalization.csv",
        normalization_rows,
    )

    transform_spec: nonlinear.ROITransformSpec | None = None
    transform_scale_estimate: nonlinear.PooledScaleEstimate | None = None
    transform_audit: dict[str, Any] | None = None
    if args.intensity_transform != "none":
        method = args.intensity_transform
        if method == nonlinear.LOG_SQUARED and args.transform_noise_floor is None:
            raise ValueError(
                "--transform-noise-floor is required for log_squared"
            )
        physical_arrays = [
            np.maximum(np.asarray(spots_by_frame[frame][1], dtype=float), 0.0)
            * float(normalization_by_frame[frame]["main_area_multiplier"])
            for frame in sorted(spots_by_frame)
        ]
        transform_spec, transform_scale_estimate = nonlinear.fit_roi_transform(
            physical_arrays,
            method,
            noise_floor=args.transform_noise_floor,
            scale_quantile=args.transform_scale_quantile,
        )
        pooled_physical = np.concatenate(physical_arrays)
        transform_audit = transform_spec.audit(pooled_physical)
        nonlinear.write_transform_provenance(
            output_root / "intensity_transform_provenance.json",
            transform_spec,
            scale_estimate=transform_scale_estimate,
            audits={"powder_spots_after_measurement_normalization": transform_audit},
            context={
                "sample": "powder",
                "channel": "spots",
                "operation_order": [
                    "piecewise-linear positive residual clipping",
                    "frame measurement normalization",
                    "fixed pooled scaling",
                    "squared-intensity transform",
                    "same-frame component sum",
                    "distinct-frame arithmetic mean",
                    "unchanged directional anchor-domain IoU",
                ],
            },
        )
        del pooled_physical
        del physical_arrays

    optimization_rows, selected_factor, optimization_audit = (
        native_sampling_factor_optimization(observations, spots_by_frame)
    )
    v6.write_csv(
        output_root / "qwidth_factor_optimization.csv",
        optimization_rows,
    )
    if math.isclose(args.half_width_factor, HALF_WIDTH_FACTOR) and not math.isclose(
        selected_factor,
        args.half_width_factor,
    ):
        raise RuntimeError(
            f"default half-width factor mismatch: "
            f"{args.half_width_factor} != {selected_factor}"
        )

    _progress("building absolute-q compact observation profiles")
    profiles, observation_audit_rows, profile_audit = (
        build_absolute_point_profiles(
            observations,
            points,
            normalization_by_frame,
            spots_by_frame,
            half_width_factor=args.half_width_factor,
            transform_spec=transform_spec,
        )
    )
    v6.write_csv(
        output_root / "observation_spots_absolute_profile_audit.csv",
        observation_audit_rows,
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

    _progress("computing 74,076 directed anchor-domain comparisons")
    (
        location_matrix,
        roi_matrix,
        sensitivity_matrix,
        support_overlap_matrix,
        directed_pair_rows,
        pair_audit,
    ) = compute_directed_pair_scores(points, profiles)
    pair_count = v6.write_csv_gz(
        output_root / "all_directed_cross_pressure_peak_pairs.csv.gz",
        directed_pair_rows,
        DIRECTED_PAIR_FIELDS,
    )
    if pair_count != 74076:
        raise RuntimeError(f"directed pair row count mismatch: {pair_count}")
    if transform_spec is None:
        regression_audit = _regression_oracle(points, roi_matrix, pair_audit)
    else:
        finite_scores = roi_matrix[np.isfinite(roi_matrix)]
        regression_audit = {
            "baseline_numeric_oracle_applicable": False,
            "reason": "intensity was deliberately nonlinearly preprocessed",
            "finite_scores_in_unit_interval": bool(
                finite_scores.size
                and np.all(finite_scores >= 0.0)
                and np.all(finite_scores <= 1.0)
            ),
            "support_disjoint_pairs_remain_exact_zero": bool(
                int(pair_audit["directed_exact_zero_scores"])
                >= int(pair_audit["support_disjoint_directed_pairs"])
            ),
        }
        if not (
            regression_audit["finite_scores_in_unit_interval"]
            and regression_audit["support_disjoint_pairs_remain_exact_zero"]
        ):
            raise RuntimeError(
                f"transformed ROI validation failed: {regression_audit}"
            )
    np.savez_compressed(
        output_root / "pressure_level_directional_similarity_matrices.npz",
        point_uids=np.asarray([str(point["point_uid"]) for point in points]),
        location=location_matrix,
        spots_absolute_anchor_ROI_iou=roi_matrix,
        fixed_D1w_sensitivity_iou=sensitivity_matrix,
        support_overlap=support_overlap_matrix,
    )

    reuse_audit: dict[str, Any] | None = None
    if not args.no_copy:
        _progress("copying and SHA-verifying unchanged location/windows")
        reuse_audit = legacy.copy_verified_unchanged_payloads(
            args.v6_source.resolve(),
            output_root,
        )

    _progress("writing 280 directional anchor ROI maps")
    anchor_rows, map_audit = write_anchor_outputs(
        output_root,
        points,
        location_matrix,
        roi_matrix,
        support_overlap_matrix,
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
        "intensity_transform": (
            None
            if transform_spec is None
            else {
                "spec": transform_spec.to_dict(),
                "scale_estimate": (
                    transform_scale_estimate.to_dict()
                    if transform_scale_estimate is not None
                    else None
                ),
                "audit": transform_audit,
            }
        ),
        "directed_pair_scores": pair_audit,
        "regression_oracle": regression_audit,
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
            "absolute_q_support": True,
            "recentered": False,
            "directional": True,
            "integration_domain": "anchor support only",
            "target_outside_own_support": 0,
            "zero_denominator": 0,
            "ROI_formula": (
                "integral_anchor(min(anchor,target))/"
                "integral_anchor(max(anchor,target))"
            ),
            "pressure_order": "descending",
            "local_peak_order": "increasing 2theta independently per pressure",
            "window_presentation": "strict lower triangle; diagonal omitted",
            "intensity_transform": (
                "none" if transform_spec is None else transform_spec.method
            ),
        },
        "elapsed_seconds": time.time() - started,
        "output_root": str(output_root),
    }
    v6.write_json(output_root / "run_manifest.json", manifest)
    _write_readme(
        output_root / "README.md",
        half_width_factor=args.half_width_factor,
        transform_spec=transform_spec,
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
