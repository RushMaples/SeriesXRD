#!/usr/bin/env python3
"""Overlay denoised powder peak-correlation maps on pressure waterfalls.

The formal mode reconstructs pressure-level peak profiles from the same
observation components used by the ROI calculation and sums them into one
composite trace per pressure. The height domain is explicit: either the
correlation transform or the measurement-normalized positive signal before
that nonlinear transform. Correlation colors are joined exactly by
``(pressure_gpa, local_peak_index)`` to the pressure-level peak registry.

The colored under-peak fill is intuitive but not lossless when distinct
azimuthal spots overlap after projection to 1D 2theta.  Therefore every
registered peak is also drawn in a separate interval-graph ribbon lane below
its pressure trace.  The ribbon is the authoritative cell-to-peak encoding.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import nonlinear_intensity_preprocessing as nonlinear  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_ROOT = (
    ROOT
    / "correlations/results/"
    "uote_nonlinear_squared_qwidth075_comparison_20260803"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_COMPARISON_ROOT / "waterfall_prototypes"
DEFAULT_ANCHOR = "anchor_007_P5p81_peak08_T00_P5p81"
PRESSURE_TOLERANCE = 1.0e-7
TRACE_HEIGHT = 0.62
ROW_SPACING = 1.0
RIBBON_TOP_GAP = 0.035
RIBBON_HEIGHT = 0.026
RIBBON_GAP = 0.006
MAX_EXPECTED_RIBBON_LANES = 6


@dataclass(frozen=True)
class Peak:
    pressure_gpa: float
    local_peak_index: int
    point_uid: str
    q: float
    two_theta_deg: float
    source_table: str
    track: int


@dataclass(frozen=True)
class PressureTrace:
    pressure_gpa: float
    frame: int
    scan: str
    source: Path
    x: np.ndarray
    transformed: np.ndarray
    displayed: np.ndarray


@dataclass(frozen=True)
class ReconstructedProfiles:
    traces: tuple[PressureTrace, ...]
    displayed_point_profiles: Mapping[str, np.ndarray]
    audit: Mapping[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT
    )
    parser.add_argument(
        "--mode", choices=("log_squared", "exp_squared"), default="log_squared"
    )
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR)
    parser.add_argument(
        "--all-anchors",
        action="store_true",
        help=(
            "Generate every anchor listed in anchor_map_index.csv while "
            "reusing one reconstructed formal-composite profile cache."
        ),
    )
    parser.add_argument(
        "--scan",
        default="scan036",
        help="Coherent representative powder scan with all 19 pressures.",
    )
    parser.add_argument(
        "--trace-source",
        choices=("formal_composite", "representative_scan"),
        default="formal_composite",
        help=(
            "formal_composite reconstructs all registered pressure-level peak "
            "profiles from the formal observations in the selected display domain; "
            "representative_scan is the older one-raw-XY-per-pressure diagnostic."
        ),
    )
    parser.add_argument(
        "--display-profile-domain",
        choices=("correlation_transform", "original_positive"),
        default="correlation_transform",
        help=(
            "Choose only the vertical waterfall profile. "
            "correlation_transform shows the same Log/Exp-transformed profiles "
            "used to calculate the ROI scores. original_positive keeps those "
            "scores unchanged but draws the measurement-normalized positive "
            "spots-channel signal before the nonlinear transform."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=190)
    parser.add_argument(
        "--palette-colors",
        type=int,
        default=0,
        help=(
            "Quantize the final PNG to this many palette colors (0 disables). "
            "Useful for complete suites on storage-constrained volumes."
        ),
    )
    parser.add_argument(
        "--compact-batch",
        action="store_true",
        help=(
            "With --all-anchors, store one gzip-compressed global mapping table "
            "and one suite validation instead of duplicate per-anchor CSV/JSON."
        ),
    )
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty row table")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quantize_png(path: Path, colors: int) -> None:
    if colors == 0:
        return
    if colors < 16 or colors > 256:
        raise ValueError("--palette-colors must be 0 or an integer from 16 to 256")
    temporary = path.with_name(f".{path.stem}.quantized.png")
    with Image.open(path) as source:
        quantized = source.convert("RGB").quantize(
            colors=colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        quantized.save(temporary, optimize=True, compress_level=9)
    temporary.replace(path)


def pressure_key(value: float) -> float:
    return round(float(value), 6)


def parse_float(value: str) -> float:
    return float(value) if value.strip() else math.nan


def load_peaks(path: Path) -> list[Peak]:
    peaks = [
        Peak(
            pressure_gpa=float(row["pressure_gpa"]),
            local_peak_index=int(row["local_peak_index"]),
            point_uid=row["point_uid"],
            q=float(row["q"]),
            two_theta_deg=float(row["two_theta_deg"]),
            source_table=row["source_table"],
            track=int(row["track"]),
        )
        for row in read_rows(path)
    ]
    if len(peaks) != 280:
        raise ValueError(f"expected 280 formal powder peaks, found {len(peaks)}")
    keys = [(pressure_key(item.pressure_gpa), item.local_peak_index) for item in peaks]
    if len(set(keys)) != len(keys):
        raise ValueError("pressure/local-peak registry keys are not unique")
    return peaks


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged: list[list[float]] = []
    for left, right in ordered:
        if not merged or left > merged[-1][1] + 1.0e-12:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [(item[0], item[1]) for item in merged]


def load_supports(path: Path) -> tuple[dict[str, list[tuple[float, float]]], dict[str, int]]:
    raw: dict[str, list[tuple[float, float]]] = defaultdict(list)
    frames: dict[str, set[int]] = defaultdict(set)
    for row in read_rows(path):
        uid = row["point_uid"]
        raw[uid].append(
            (float(row["two_theta_lower_deg"]), float(row["two_theta_upper_deg"]))
        )
        frames[uid].add(int(row["frame"]))
    supports = {uid: merge_intervals(items) for uid, items in raw.items()}
    return supports, {uid: len(items) for uid, items in frames.items()}


def load_matrix(path: Path) -> tuple[dict[tuple[float, int], float], dict[float, int]]:
    scores: dict[tuple[float, int], float] = {}
    counts: dict[float, int] = {}
    for row in read_rows(path):
        pressure = pressure_key(float(row["pressure_gpa"]))
        count = int(row["peak_count_at_pressure"])
        counts[pressure] = count
        for peak_index in range(1, count + 1):
            scores[(pressure, peak_index)] = parse_float(row[f"peak {peak_index}"])
    return scores, counts


def infer_spots_path(fit_path: str) -> Path:
    path = Path(fit_path)
    parts = list(path.parts)
    try:
        index = parts.index("fit_channel")
    except ValueError as exc:
        raise ValueError(f"not a fit-channel path: {fit_path}") from exc
    parts[index] = "spots_channel"
    return Path(*parts)


def load_scan_traces(
    normalization_csv: Path,
    scan: str,
    spec: nonlinear.ROITransformSpec,
) -> list[PressureTrace]:
    selected = [row for row in read_rows(normalization_csv) if row["scan"] == scan]
    if len(selected) != 19:
        raise ValueError(f"expected 19 pressure files in {scan}, found {len(selected)}")
    prepared: list[tuple[dict[str, str], Path, np.ndarray, np.ndarray]] = []
    for row in selected:
        source = infer_spots_path(row["fit_channel_file"])
        if not source.is_file():
            raise FileNotFoundError(source)
        data = np.loadtxt(source, comments="#", dtype=float)
        if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 3:
            raise ValueError(f"invalid XY data: {source}")
        x = np.asarray(data[:, 0], dtype=float)
        physical = np.maximum(np.asarray(data[:, 1], dtype=float), 0.0) * float(
            row["main_area_multiplier"]
        )
        transformed = np.asarray(spec.transform(physical), dtype=float)
        if np.any(~np.isfinite(transformed)) or np.any(transformed < 0.0) or np.any(
            transformed > 1.0
        ):
            raise ValueError(f"transformed trace outside [0,1]: {source}")
        prepared.append((row, source, x, transformed))

    # Align each trace to its own low positive background for display only, but
    # retain one shared amplitude scale across every pressure.
    corrected: list[np.ndarray] = []
    for _, _, _, transformed in prepared:
        finite = transformed[np.isfinite(transformed)]
        floor = float(np.quantile(finite, 0.05)) if finite.size else 0.0
        corrected.append(np.maximum(transformed - floor, 0.0))
    shared_max = max(float(np.max(item)) for item in corrected)
    if not np.isfinite(shared_max) or shared_max <= 0.0:
        raise ValueError("representative scan has no positive transformed signal")

    traces: list[PressureTrace] = []
    for (row, source, x, transformed), corrected_values in zip(prepared, corrected):
        traces.append(
            PressureTrace(
                pressure_gpa=float(row["pressure_gpa"]),
                frame=int(row["frame"]),
                scan=row["scan"],
                source=source,
                x=x,
                transformed=transformed,
                displayed=np.clip(corrected_values / shared_max, 0.0, 1.0),
            )
        )
    return sorted(traces, key=lambda item: item.pressure_gpa, reverse=True)


def insert_zero_crossings(
    coordinate: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match the formal ROI builder's piecewise-linear positive clipping."""

    x = np.asarray(coordinate, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("invalid piecewise-linear component")
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


def load_point_registry_areas(path: Path) -> dict[str, float]:
    return {
        row["point_uid"]: float(row["spots_absolute_integral_main"])
        for row in read_rows(path)
    }


def reconstruct_formal_pressure_profiles(
    *,
    profile_audit_csv: Path,
    point_registry_csv: Path,
    peaks: Sequence[Peak],
    spec: nonlinear.ROITransformSpec,
    profile_domain: str = "correlation_transform",
) -> ReconstructedProfiles:
    """Rebuild formal observation profiles in a selected intensity domain.

    Components for multiple observations in one physical frame are summed;
    resulting frame profiles are averaged over distinct frames for each formal
    pressure-level point.  The 12--22 point profiles at one pressure are then
    summed to form the one-row waterfall composite requested by the user.

    ``correlation_transform`` exactly reproduces the profiles used by the ROI
    calculation.  ``original_positive`` uses the same observations, supports,
    positive clipping, and measurement normalization, but deliberately omits
    the nonlinear Log/Exp transform so the waterfall amplitude is shown in the
    original pre-denoise signal domain.
    """

    if profile_domain not in {"correlation_transform", "original_positive"}:
        raise ValueError(f"unknown formal profile domain: {profile_domain}")

    rows = read_rows(profile_audit_csv)
    if len(rows) != 519:
        raise ValueError(f"expected 519 formal observation components, found {len(rows)}")
    half_width_factors = sorted(
        {round(float(row["half_width_factor"]), 12) for row in rows}
    )
    if len(half_width_factors) != 1:
        raise ValueError(
            "formal observation components do not share one half-width factor: "
            f"{half_width_factors}"
        )
    half_width_factor = half_width_factors[0]
    support_left = min(float(row["two_theta_lower_deg"]) for row in rows)
    support_right = max(float(row["two_theta_upper_deg"]) for row in rows)
    # A dense shared plotting grid evaluates the same piecewise-linear
    # components without changing the formal integration or aggregation rules.
    grid = np.linspace(support_left, support_right, 7201, dtype=float)
    source_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    frame_profiles: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    frame_integrals: dict[str, dict[int, float]] = defaultdict(dict)
    component_integrals: list[float] = []

    for row in rows:
        source = Path(row["spots_channel_file"])
        if source not in source_cache:
            data = np.loadtxt(source, comments="#", dtype=float)
            if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 3:
                raise ValueError(f"invalid formal component XY: {source}")
            source_cache[source] = (
                np.asarray(data[:, 0], dtype=float),
                np.asarray(data[:, 1], dtype=float),
            )
        source_x, source_y = source_cache[source]
        lower = float(row["two_theta_lower_deg"])
        upper = float(row["two_theta_upper_deg"])
        native = (source_x > lower) & (source_x < upper)
        coordinate = np.concatenate(
            (np.asarray([lower]), source_x[native], np.asarray([upper]))
        )
        raw_values = np.interp(
            coordinate, source_x, source_y, left=0.0, right=0.0
        )
        coordinate, positive = insert_zero_crossings(coordinate, raw_values)
        multiplier = float(row["original_main_measurement_multiplier"])
        normalized_positive = np.asarray(positive * multiplier, dtype=float)
        if profile_domain == "correlation_transform":
            profile_values = np.asarray(
                spec.transform(normalized_positive), dtype=float
            )
            if np.any(profile_values > 1.0):
                raise ValueError(
                    f"transformed component exceeds 1 for {row['point_uid']}"
                )
        else:
            profile_values = normalized_positive
        if np.any(~np.isfinite(profile_values)) or np.any(profile_values < 0.0):
            raise ValueError(
                f"invalid {profile_domain} component for {row['point_uid']}"
            )
        component_profile = np.interp(
            grid, coordinate, profile_values, left=0.0, right=0.0
        )
        component_integral = float(np.trapezoid(profile_values, coordinate))
        component_integrals.append(component_integral)
        uid = row["point_uid"]
        frame = int(row["frame"])
        if frame in frame_profiles[uid]:
            frame_profiles[uid][frame] += component_profile
            frame_integrals[uid][frame] += component_integral
        else:
            frame_profiles[uid][frame] = component_profile
            frame_integrals[uid][frame] = component_integral

    point_profiles: dict[str, np.ndarray] = {}
    exact_point_integrals: dict[str, float] = {}
    for peak in peaks:
        uid = peak.point_uid
        per_frame = frame_profiles.get(uid, {})
        if not per_frame:
            raise ValueError(f"formal point has no reconstructed component: {uid}")
        point_profiles[uid] = np.mean(
            np.stack(list(per_frame.values()), axis=0), axis=0
        )
        exact_point_integrals[uid] = float(
            np.mean(list(frame_integrals[uid].values()))
        )
    if len(point_profiles) != 280:
        raise ValueError(f"expected 280 reconstructed point profiles, got {len(point_profiles)}")

    max_abs_error: float | None = None
    max_relative_error: float | None = None
    if profile_domain == "correlation_transform":
        formal_areas = load_point_registry_areas(point_registry_csv)
        errors: list[float] = []
        relative_errors: list[float] = []
        for uid, reconstructed in exact_point_integrals.items():
            expected = formal_areas[uid]
            error = abs(reconstructed - expected)
            errors.append(error)
            if expected > 0.0:
                relative_errors.append(error / expected)
            elif error != 0.0:
                relative_errors.append(math.inf)
        max_abs_error = max(errors, default=math.inf)
        max_relative_error = max(relative_errors, default=0.0)
        # The formal registry stores decimal text, so the reconstruction is checked
        # to sub-nanounit absolute and sub-0.1-ppm relative agreement rather than
        # against an unattainable binary round-trip identity.
        if max_abs_error > 1.0e-9 or max_relative_error > 1.0e-7:
            raise RuntimeError(
                "reconstructed formal profile areas do not match point_registry: "
                f"max_abs={max_abs_error}, max_relative={max_relative_error}"
            )

    by_pressure: dict[float, list[Peak]] = defaultdict(list)
    for peak in peaks:
        by_pressure[pressure_key(peak.pressure_gpa)].append(peak)
    composites: dict[float, np.ndarray] = {
        pressure: np.sum(
            np.stack([point_profiles[item.point_uid] for item in items], axis=0),
            axis=0,
        )
        for pressure, items in by_pressure.items()
    }
    shared_max = max(float(np.max(values)) for values in composites.values())
    if not np.isfinite(shared_max) or shared_max <= 0.0:
        raise ValueError("formal pressure composites have no positive intensity")
    displayed_points = {
        uid: np.clip(values / shared_max, 0.0, 1.0)
        for uid, values in point_profiles.items()
    }
    traces = tuple(
        PressureTrace(
            pressure_gpa=float(pressure),
            frame=-1,
            scan=(
                "formal pressure-level composite"
                if profile_domain == "correlation_transform"
                else "original positive pressure-level composite (pre-denoise)"
            ),
            source=profile_audit_csv,
            x=grid,
            transformed=composites[pressure],
            displayed=np.clip(composites[pressure] / shared_max, 0.0, 1.0),
        )
        for pressure in sorted(composites, reverse=True)
    )
    zero_profile_uids = sorted(
        uid for uid, value in exact_point_integrals.items() if value <= 0.0
    )
    audit: dict[str, Any] = {
        "trace_source": "formal_composite",
        "profile_domain": profile_domain,
        "nonlinear_transform_applied": profile_domain == "correlation_transform",
        "positive_clipping_applied": True,
        "measurement_normalization_applied": True,
        "half_width_factor": half_width_factor,
        "q_support_formula": "[qi-c*q_width, qi+c*q_width]",
        "observation_components": len(rows),
        "source_xy_files": len(source_cache),
        "pressure_level_point_profiles": len(point_profiles),
        "pressure_composite_traces": len(traces),
        "aggregation_within_frame": "sum",
        "aggregation_across_distinct_frames": "arithmetic mean per point",
        "aggregation_within_pressure": "sum 12-22 formal point profiles",
        "shared_display_scale": shared_max,
        "point_area_max_abs_error_vs_formal_registry": max_abs_error,
        "point_area_max_relative_error_vs_formal_registry": max_relative_error,
        "point_integral_min": min(exact_point_integrals.values()),
        "point_integral_max": max(exact_point_integrals.values()),
        "point_integral_sum": sum(exact_point_integrals.values()),
        "zero_integral_point_count": len(zero_profile_uids),
        "zero_integral_point_uids": zero_profile_uids,
        "grid_points": int(grid.size),
        "grid_min_two_theta_deg": float(grid[0]),
        "grid_max_two_theta_deg": float(grid[-1]),
    }
    return ReconstructedProfiles(
        traces=traces,
        displayed_point_profiles=displayed_points,
        audit=audit,
    )


def interval_sets_overlap(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]
) -> bool:
    return any(min(b, d) > max(a, c) for a, b in left for c, d in right)


def assign_interval_lanes(
    peaks: Sequence[Peak], supports: Mapping[str, Sequence[tuple[float, float]]]
) -> dict[str, int]:
    lanes: list[list[tuple[float, float]]] = []
    assignment: dict[str, int] = {}
    ordered = sorted(
        peaks,
        key=lambda item: min(left for left, _ in supports[item.point_uid]),
    )
    for peak in ordered:
        intervals = list(supports[peak.point_uid])
        for lane_index, occupied in enumerate(lanes):
            if not interval_sets_overlap(intervals, occupied):
                assignment[peak.point_uid] = lane_index
                occupied.extend(intervals)
                break
        else:
            assignment[peak.point_uid] = len(lanes)
            lanes.append(list(intervals))
    return assignment


def score_status(
    peak: Peak,
    score: float,
    anchor_pressure: float,
    anchor_peak_index: int,
) -> tuple[str, float]:
    same_pressure = math.isclose(
        peak.pressure_gpa, anchor_pressure, abs_tol=PRESSURE_TOLERANCE
    )
    if same_pressure and peak.local_peak_index == anchor_peak_index:
        return "anchor_self", 1.0
    if same_pressure:
        return "same_pressure_not_compared", math.nan
    if not np.isfinite(score):
        raise ValueError(
            f"missing cross-pressure score for P={peak.pressure_gpa}, "
            f"peak={peak.local_peak_index}"
        )
    return ("compared_zero" if score == 0.0 else "compared_positive"), score


def plot_waterfall(
    *,
    output: Path,
    mode: str,
    anchor_token: str,
    anchor_peak: Peak,
    traces: Sequence[PressureTrace],
    peaks: Sequence[Peak],
    supports: Mapping[str, Sequence[tuple[float, float]]],
    frame_support: Mapping[str, int],
    scores: Mapping[tuple[float, int], float],
    trace_source: str,
    display_profile_domain: str,
    displayed_point_profiles: Mapping[str, np.ndarray] | None,
    correlation_reconstruction_audit: Mapping[str, Any] | None,
    display_reconstruction_audit: Mapping[str, Any] | None,
    dpi: int,
    palette_colors: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_pressure: dict[float, list[Peak]] = defaultdict(list)
    for peak in peaks:
        by_pressure[pressure_key(peak.pressure_gpa)].append(peak)
    trace_pressures = [pressure_key(item.pressure_gpa) for item in traces]
    if trace_pressures != sorted(by_pressure, reverse=True):
        raise ValueError("representative trace pressure order does not match registry")

    cmap = plt.colormaps["viridis"]
    norm = Normalize(vmin=0.0, vmax=1.0)
    max_lanes = 0
    mappings: list[dict[str, Any]] = []

    fig, ax = plt.subplots(figsize=(18.0, 14.2), constrained_layout=False)
    left_limit = min(left for values in supports.values() for left, _ in values) - 0.18
    right_limit = max(right for values in supports.values() for _, right in values) + 0.28

    anchor_intervals = list(supports[anchor_peak.point_uid])
    for left, right in anchor_intervals:
        ax.axvspan(left, right, color="#d8a620", alpha=0.075, zorder=0)
        ax.axvline(left, color="#9a6b00", linewidth=0.75, linestyle="--", zorder=0)
        ax.axvline(right, color="#9a6b00", linewidth=0.75, linestyle="--", zorder=0)

    baselines: list[float] = []
    ylabels: list[str] = []
    positive_count = 0
    zero_count = 0
    anchor_count = 0
    omitted_count = 0
    compared_count = 0

    for row_index, trace in enumerate(traces):
        baseline = float(len(traces) - 1 - row_index) * ROW_SPACING
        baselines.append(baseline)
        pressure = pressure_key(trace.pressure_gpa)
        row_peaks = sorted(
            by_pressure[pressure], key=lambda item: item.local_peak_index
        )
        lanes = assign_interval_lanes(row_peaks, supports)
        row_lane_count = max(lanes.values(), default=-1) + 1
        max_lanes = max(max_lanes, row_lane_count)
        ylabels.append(f"{trace.pressure_gpa:g} GPa  ·  {len(row_peaks)} peaks")

        ax.axhline(baseline, color="#dddddd", linewidth=0.55, zorder=0)
        y_curve = baseline + TRACE_HEIGHT * trace.displayed

        # Contextual under-curve colors. For a formal composite, each colored
        # top is the target point's own reconstructed profile, rather than the
        # total pressure composite. Overlapping spatial peaks can still
        # overpaint here; the separate ribbon lanes below are authoritative.
        for peak in row_peaks:
            raw_score = scores[(pressure, peak.local_peak_index)]
            status, shown_score = score_status(
                peak,
                raw_score,
                anchor_peak.pressure_gpa,
                anchor_peak.local_peak_index,
            )
            lane = lanes[peak.point_uid]
            intervals = list(supports[peak.point_uid])
            if status == "same_pressure_not_compared":
                color = "#b7b7b7"
                face_alpha = 0.0
                omitted_count += 1
            else:
                color = cmap(norm(shown_score))
                face_alpha = 0.48
                compared_count += int(status.startswith("compared"))
                positive_count += int(status == "compared_positive")
                zero_count += int(status == "compared_zero")
                anchor_count += int(status == "anchor_self")

            for component_index, (left, right) in enumerate(intervals):
                native = (trace.x >= left) & (trace.x <= right)
                if np.count_nonzero(native) >= 2 and face_alpha > 0.0:
                    if displayed_point_profiles is None:
                        colored_top = y_curve[native]
                    else:
                        point_display = displayed_point_profiles[peak.point_uid]
                        if point_display.shape != trace.x.shape:
                            raise ValueError(
                                f"point/trace grid mismatch for {peak.point_uid}"
                            )
                        colored_top = (
                            baseline + TRACE_HEIGHT * point_display[native]
                        )
                    ax.fill_between(
                        trace.x[native],
                        baseline,
                        colored_top,
                        color=color,
                        alpha=face_alpha,
                        linewidth=0.0,
                        zorder=2,
                    )
                ribbon_top = baseline - RIBBON_TOP_GAP - lane * (
                    RIBBON_HEIGHT + RIBBON_GAP
                )
                rectangle = Rectangle(
                    (left, ribbon_top - RIBBON_HEIGHT),
                    right - left,
                    RIBBON_HEIGHT,
                    facecolor=(color if face_alpha > 0.0 else "none"),
                    edgecolor=("#222222" if status == "anchor_self" else color),
                    linewidth=(1.05 if status == "anchor_self" else 0.5),
                    alpha=(0.98 if face_alpha > 0.0 else 0.75),
                    zorder=4,
                )
                ax.add_patch(rectangle)
                mappings.append(
                    {
                        "anchor_token": anchor_token,
                        "mode": mode,
                        "trace_source": trace_source,
                        "display_profile_domain": display_profile_domain,
                        "pressure_gpa": peak.pressure_gpa,
                        "trace_label": trace.scan,
                        "trace_frame": ("" if trace.frame < 0 else trace.frame),
                        "trace_input": str(trace.source),
                        "local_peak_index": peak.local_peak_index,
                        "point_uid": peak.point_uid,
                        "track": peak.track,
                        "two_theta_center_deg": peak.two_theta_deg,
                        "support_component_index": component_index,
                        "support_left_deg": left,
                        "support_right_deg": right,
                        "ribbon_lane_0based": lane,
                        "correlation": (
                            "" if not np.isfinite(shown_score) else shown_score
                        ),
                        "status": status,
                        "distinct_profile_frames": frame_support[peak.point_uid],
                    }
                )

            if status == "anchor_self":
                ax.annotate(
                    "REF",
                    xy=(peak.two_theta_deg, baseline + TRACE_HEIGHT + 0.015),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    fontweight="bold",
                    color="#6b4a00",
                    zorder=7,
                )

        # Redraw the selected waterfall profile after all contextual fills.
        is_anchor_row = math.isclose(
            trace.pressure_gpa,
            anchor_peak.pressure_gpa,
            abs_tol=PRESSURE_TOLERANCE,
        )
        ax.plot(
            trace.x,
            y_curve,
            color=("#151515" if is_anchor_row else "#4a4a4a"),
            linewidth=(1.25 if is_anchor_row else 0.82),
            zorder=5,
        )
        if is_anchor_row:
            ax.axhline(
                baseline,
                color="#9a6b00",
                linewidth=0.85,
                linestyle=":",
                zorder=1,
            )

    if max_lanes > MAX_EXPECTED_RIBBON_LANES:
        raise RuntimeError(
            f"ribbon layout exceeded audited maximum: {max_lanes} > "
            f"{MAX_EXPECTED_RIBBON_LANES}"
        )

    ax.set_xlim(left_limit, right_limit)
    ax.set_ylim(
        -RIBBON_TOP_GAP
        - max_lanes * (RIBBON_HEIGHT + RIBBON_GAP)
        - 0.12,
        baselines[-1] + (len(traces) - 1) * ROW_SPACING + TRACE_HEIGHT + 0.12,
    )
    ax.set_yticks(baselines, ylabels, fontsize=8.4)
    ax.set_xlabel(r"$2\theta$ (degrees)", fontsize=12)
    ax.set_ylabel(
        "Pressure rows (descending); fixed offsets prevent trace overlap",
        fontsize=11,
    )
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    mode_label = "Log-squared" if mode == "log_squared" else "Exp-squared"
    half_width_factor = (
        None
        if correlation_reconstruction_audit is None
        else correlation_reconstruction_audit.get("half_width_factor")
    )
    support_suffix = (
        ""
        if half_width_factor is None
        else f"  |  q-core=qi±{float(half_width_factor):g} q_width"
    )
    original_display = display_profile_domain == "original_positive"
    figure_title = (
        f"Powder {mode_label} ROI correlation on original profiles"
        if original_display
        else f"Powder {mode_label} ROI-correlation waterfall"
    )
    fig.suptitle(
        figure_title,
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    ax.set_title(
        f"anchor {anchor_peak.point_uid} · {anchor_peak.pressure_gpa:g} GPa "
        f"peak {anchor_peak.local_peak_index} · "
        f"2θ={anchor_peak.two_theta_deg:.4f}°  |  "
        f"correlation={mode_label}  |  "
        f"display={'original positive signal (pre-denoise)' if original_display else mode_label + ' transformed profiles'}"
        f"{support_suffix}",
        fontsize=11,
        pad=12,
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.018, fraction=0.028
    )
    colorbar.set_label(f"{mode_label} ROI integrated IoU", fontsize=10)
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))

    if trace_source == "formal_composite" and original_display:
        footnote = (
            f"Color = {mode_label} ROI correlation calculated from the nonlinear-"
            "transformed profiles.  Gray line and colored-fill height = the same "
            "519 formal spots-channel observations before that nonlinear transform: "
            "piecewise-linear positive signal × measurement multiplier, summed "
            "within frame, averaged across distinct frames per peak, then the "
            "12–22 formal peaks are summed at each pressure.  One shared amplitude "
            "scale is used.  Dark purple = correlation 0; gray anchor-row outlines "
            "= intentionally not compared."
        )
    elif trace_source == "formal_composite":
        footnote = (
            "Gray line = reconstructed pressure-level composite: the exact "
            "transformed observation components used by ROI correlation are "
            "summed within frame, averaged across distinct frames per peak, "
            "then the 12–22 formal peak profiles are summed at each pressure.  "
            "Colored fill = that target peak's own profile; stacked ribbons "
            "preserve separate cells where azimuthal spots overlap in 1D.  "
            "Dark purple = correlation 0; gray anchor-row outlines = intentionally "
            "not compared."
        )
    else:
        footnote = (
            "Gray line = one actual spots-channel XY per pressure from scan036, "
            "after the same fixed denoise transform; one shared amplitude scale.  "
            "Colored peak fill + stacked ribbons = anchor-to-target ROI correlation.  "
            "Dark purple = detected peak with correlation 0; gray outline on the "
            "anchor-pressure row = intentionally not compared.  Dashed gold limits "
            "mark the anchor integration support."
        )
    fig.text(
        0.12,
        0.012,
        footnote,
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#555555",
        wrap=True,
    )
    fig.subplots_adjust(left=0.17, right=0.90, top=0.925, bottom=0.075)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    quantize_png(output, palette_colors)

    audit = {
        "status": "PASS",
        "mode": mode,
        "anchor": anchor_token,
        "anchor_point_uid": anchor_peak.point_uid,
        "anchor_pressure_gpa": anchor_peak.pressure_gpa,
        "anchor_local_peak_index": anchor_peak.local_peak_index,
        "pressure_rows": len(traces),
        "pressure_descending": trace_pressures == sorted(trace_pressures, reverse=True),
        "registered_peaks": len(peaks),
        "cross_pressure_colored_cells": compared_count,
        "positive_cross_pressure_cells": positive_count,
        "zero_cross_pressure_cells": zero_count,
        "anchor_self_cells": anchor_count,
        "same_pressure_not_compared_peaks": omitted_count,
        "maximum_ribbon_lanes": max_lanes,
        "strictly_nonoverlapping_trace_bands": True,
        "trace_height": TRACE_HEIGHT,
        "row_spacing": ROW_SPACING,
        "fixed_color_range": [0.0, 1.0],
        "trace_source": trace_source,
        "display_profile_domain": display_profile_domain,
        "trace_label": traces[0].scan,
        "trace_frames": [item.frame for item in traces if item.frame >= 0],
        "trace_inputs": sorted({str(item.source) for item in traces}),
        "anchor_support_components": [list(item) for item in anchor_intervals],
        "mapping_rows": len(mappings),
        "output_png": str(output.resolve()),
        "png_palette_colors": palette_colors,
        "formal_profile_reconstruction": (
            None
            if correlation_reconstruction_audit is None
            else dict(correlation_reconstruction_audit)
        ),
        "correlation_profile_reconstruction": (
            None
            if correlation_reconstruction_audit is None
            else dict(correlation_reconstruction_audit)
        ),
        "display_profile_reconstruction": (
            None
            if display_reconstruction_audit is None
            else dict(display_reconstruction_audit)
        ),
    }
    expected_cross_pressure = len(peaks) - len(
        by_pressure[pressure_key(anchor_peak.pressure_gpa)]
    )
    if compared_count != expected_cross_pressure:
        raise RuntimeError(f"cross-pressure cell count mismatch: {audit}")
    # Preserve the original c=0.6 numerical oracle, but do not apply its
    # fixed positive/zero partition to scientifically different q supports.
    if (
        anchor_token == DEFAULT_ANCHOR
        and half_width_factor is not None
        and math.isclose(float(half_width_factor), 0.6, abs_tol=1.0e-12)
        and (compared_count != 266 or positive_count != 67 or zero_count != 199)
    ):
        raise RuntimeError(f"anchor-007 regression count mismatch: {audit}")
    if anchor_count != 1:
        raise RuntimeError(f"anchor self mapping mismatch: {audit}")
    return mappings, audit


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compact_batch and not args.all_anchors:
        raise ValueError("--compact-batch requires --all-anchors")
    if (
        args.trace_source != "formal_composite"
        and args.display_profile_domain != "correlation_transform"
    ):
        raise ValueError(
            "--display-profile-domain original_positive currently requires "
            "--trace-source formal_composite"
        )
    source_root = args.comparison_root / "_sources" / args.mode / "powder_roi"
    formal_root = args.comparison_root / args.mode / "powder" / "roi_area"
    provenance = json.loads(
        (source_root / "intensity_transform_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    spec = nonlinear.ROITransformSpec(**provenance["transform"])
    peaks = load_peaks(source_root / "pressure_peak_grid.csv")
    supports, frame_support = load_supports(
        source_root / "observation_spots_absolute_profile_audit.csv"
    )
    missing_support = sorted({item.point_uid for item in peaks} - set(supports))
    if missing_support:
        raise ValueError(f"registered peaks without support: {missing_support[:5]}")
    anchor_index = read_rows(source_root / "anchor_map_index.csv")
    anchor_rows: dict[str, dict[str, str]] = {}
    for row in anchor_index:
        token = Path(row["roi_matrix_csv"]).stem
        if token in anchor_rows:
            raise ValueError(f"duplicate anchor token in index: {token}")
        anchor_rows[token] = row
    if len(anchor_rows) != len(peaks):
        raise ValueError(
            f"expected one anchor per formal peak ({len(peaks)}), "
            f"found {len(anchor_rows)}"
        )
    anchor_tokens = list(anchor_rows) if args.all_anchors else [args.anchor]
    missing_anchors = [token for token in anchor_tokens if token not in anchor_rows]
    if missing_anchors:
        raise ValueError(f"unknown anchors: {missing_anchors[:5]}")

    peak_lookup = {
        (pressure_key(item.pressure_gpa), item.local_peak_index): item for item in peaks
    }
    if args.trace_source == "formal_composite":
        correlation_reconstructed = reconstruct_formal_pressure_profiles(
            profile_audit_csv=(
                source_root / "observation_spots_absolute_profile_audit.csv"
            ),
            point_registry_csv=source_root / "point_registry.csv",
            peaks=peaks,
            spec=spec,
            profile_domain="correlation_transform",
        )
        if args.display_profile_domain == "correlation_transform":
            display_reconstructed = correlation_reconstructed
        else:
            display_reconstructed = reconstruct_formal_pressure_profiles(
                profile_audit_csv=(
                    source_root / "observation_spots_absolute_profile_audit.csv"
                ),
                point_registry_csv=source_root / "point_registry.csv",
                peaks=peaks,
                spec=spec,
                profile_domain="original_positive",
            )
        traces = list(display_reconstructed.traces)
        displayed_point_profiles: Mapping[str, np.ndarray] | None = (
            display_reconstructed.displayed_point_profiles
        )
        correlation_reconstruction_audit: Mapping[str, Any] | None = (
            correlation_reconstructed.audit
        )
        display_reconstruction_audit: Mapping[str, Any] | None = (
            display_reconstructed.audit
        )
    else:
        traces = load_scan_traces(
            source_root / "frame_measurement_normalization.csv", args.scan, spec
        )
        displayed_point_profiles = None
        correlation_reconstruction_audit = None
        display_reconstruction_audit = None
    expected_counts = {
        pressure_key(item.pressure_gpa): len(
            [peak for peak in peaks if pressure_key(peak.pressure_gpa) == pressure_key(item.pressure_gpa)]
        )
        for item in traces
    }

    suite_rows: list[dict[str, Any]] = []
    suite_root = args.out_dir / "powder" / args.mode
    compact_mapping_path = suite_root / "PEAK_COLOR_MAPPING.csv.gz"
    compact_handle = None
    compact_writer = None
    if args.compact_batch:
        suite_root.mkdir(parents=True, exist_ok=True)
        compact_handle = gzip.open(
            compact_mapping_path, "wt", newline="", encoding="utf-8"
        )
    try:
        for sequence, anchor_token in enumerate(anchor_tokens, start=1):
            matrix_path = formal_root / "matrices" / f"{anchor_token}.csv"
            if not matrix_path.is_file():
                raise FileNotFoundError(matrix_path)
            scores, counts = load_matrix(matrix_path)
            if counts != expected_counts:
                raise ValueError(
                    f"matrix peak counts do not match formal registry: {anchor_token}"
                )
            anchor_row = anchor_rows[anchor_token]
            anchor_key = (
                pressure_key(float(anchor_row["pressure_gpa"])),
                int(anchor_row["local_peak_index"]),
            )
            anchor_peak = peak_lookup[anchor_key]
            trace_variant = (
                args.trace_source
                if args.display_profile_domain == "correlation_transform"
                else f"{args.trace_source}_original_positive"
            )
            out_dir = (
                args.out_dir
                / "powder"
                / args.mode
                / anchor_token
                / trace_variant
            )
            output_png = (
                out_dir
                / f"{anchor_token}_{trace_variant}_correlation_waterfall.png"
            )
            mappings, audit = plot_waterfall(
                output=output_png,
                mode=args.mode,
                anchor_token=anchor_token,
                anchor_peak=anchor_peak,
                traces=traces,
                peaks=peaks,
                supports=supports,
                frame_support=frame_support,
                scores=scores,
                trace_source=args.trace_source,
                display_profile_domain=args.display_profile_domain,
                displayed_point_profiles=displayed_point_profiles,
                correlation_reconstruction_audit=(
                    correlation_reconstruction_audit
                ),
                display_reconstruction_audit=display_reconstruction_audit,
                dpi=args.dpi,
                palette_colors=args.palette_colors,
            )
            if compact_handle is not None:
                if compact_writer is None:
                    compact_writer = csv.DictWriter(
                        compact_handle, fieldnames=list(mappings[0])
                    )
                    compact_writer.writeheader()
                compact_writer.writerows(mappings)
            else:
                write_rows(out_dir / "peak_color_mapping.csv", mappings)
                write_json(out_dir / "VALIDATION.json", audit)
            suite_rows.append(
                {
                    "sequence": sequence,
                    "anchor_token": anchor_token,
                    "anchor_point_uid": audit["anchor_point_uid"],
                    "anchor_pressure_gpa": audit["anchor_pressure_gpa"],
                    "anchor_local_peak_index": audit["anchor_local_peak_index"],
                    "cross_pressure_colored_cells": audit[
                        "cross_pressure_colored_cells"
                    ],
                    "positive_cross_pressure_cells": audit[
                        "positive_cross_pressure_cells"
                    ],
                    "zero_cross_pressure_cells": audit[
                        "zero_cross_pressure_cells"
                    ],
                    "mapping_rows": audit["mapping_rows"],
                    "output_png": audit["output_png"],
                    "status": audit["status"],
                }
            )
            print(
                f"[{sequence}/{len(anchor_tokens)}] {args.mode} powder "
                f"{anchor_token}: PASS",
                flush=True,
            )
    finally:
        if compact_handle is not None:
            compact_handle.close()

    if args.all_anchors:
        write_rows(suite_root / "WATERFALL_INDEX.csv", suite_rows)
        write_json(
            suite_root / "SUITE_VALIDATION.json",
            {
                "status": "PASS",
                "sample": "powder",
                "mode": args.mode,
                "trace_source": args.trace_source,
                "display_profile_domain": args.display_profile_domain,
                "expected_anchors": len(peaks),
                "generated_anchors": len(suite_rows),
                "pressure_rows_per_figure": len(traces),
                "registered_peaks": len(peaks),
                "compact_batch": args.compact_batch,
                "png_palette_colors": args.palette_colors,
                "combined_mapping_csv_gz": (
                    str(compact_mapping_path.resolve())
                    if args.compact_batch
                    else None
                ),
                "all_anchor_validations_pass": all(
                    row["status"] == "PASS" for row in suite_rows
                ),
                "index_csv": str((suite_root / "WATERFALL_INDEX.csv").resolve()),
                "formal_profile_reconstruction": (
                    None
                    if correlation_reconstruction_audit is None
                    else dict(correlation_reconstruction_audit)
                ),
                "correlation_profile_reconstruction": (
                    None
                    if correlation_reconstruction_audit is None
                    else dict(correlation_reconstruction_audit)
                ),
                "display_profile_reconstruction": (
                    None
                    if display_reconstruction_audit is None
                    else dict(display_reconstruction_audit)
                ),
            },
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "sample": "powder",
                    "mode": args.mode,
                    "anchors": len(suite_rows),
                    "output_root": str(suite_root.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
