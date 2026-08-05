#!/usr/bin/env python3
"""Render a formal-composite single-crystal correlation waterfall.

The existing transformed single-crystal analysis compares occurrences of one
global track across the twelve curated frames.  This renderer therefore colors
only that track's observed occurrence in each frame; all other curated peaks
remain visible as neutral context and are never presented as missing scores.

For display, every collapsed frame/track feature is represented by the radial
projection of a unit elliptical kernel on its formal q +/- halfwidth support.
The kernel is normalized in 2theta and scaled so its integral is exactly the
median transformed scalar consumed by the existing correlation algorithm.
All 3--31 collapsed profiles in a frame are summed to form its one pressure-row
composite.  The kernel is explicitly a faithful visualization of the formal
support and scalar, not a claim that a raw 1-D line profile was measured.
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
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_refinement_legacy_correlations as formal  # noqa: E402
from generate_denoised_peak_correlation_waterfall import (  # noqa: E402
    interval_sets_overlap,
    quantize_png,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_ROOT = (
    ROOT
    / "correlations/results/"
    "uote_nonlinear_squared_preprocessed_comparison_20260802"
)
DEFAULT_DATA_ROOT = ROOT / "correlations/UOTe XRD Data Refinement"
DEFAULT_OUTPUT_ROOT = DEFAULT_COMPARISON_ROOT / "waterfall_prototypes"
DEFAULT_TRACK = 18
DEFAULT_ANCHOR_FRAME = 0
TRACE_HEIGHT = 0.62
ROW_SPACING = 1.0
RIBBON_TOP_GAP = 0.035
RIBBON_HEIGHT = 0.026
RIBBON_GAP = 0.006
GRID_POINTS = 4801


@dataclass(frozen=True)
class Feature:
    frame: int
    pressure_gpa: float
    orientation: str
    branch: str
    track: int
    local_peak_index: int
    q: float
    halfwidth_q: float
    two_theta_deg: float
    scalar_area: float
    n_observations: int
    duplicate: bool

    @property
    def uid(self) -> str:
        return f"f{self.frame:04d}_track{self.track:03d}"


@dataclass(frozen=True)
class FrameTrace:
    frame: int
    pressure_gpa: float
    orientation: str
    branch: str
    x: np.ndarray
    displayed: np.ndarray


@dataclass(frozen=True)
class ProfileBundle:
    grid: np.ndarray
    displayed_profiles: Mapping[str, np.ndarray]
    traces: tuple[FrameTrace, ...]
    supports: Mapping[str, tuple[tuple[float, float], ...]]
    audit: Mapping[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--mode", choices=("log_squared", "exp_squared"), default="log_squared"
    )
    parser.add_argument(
        "--family", choices=("roi_area", "location"), default="roi_area"
    )
    parser.add_argument("--track", type=int, default=DEFAULT_TRACK)
    parser.add_argument("--anchor-frame", type=int, default=DEFAULT_ANCHOR_FRAME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=190)
    parser.add_argument(
        "--all-anchors",
        action="store_true",
        help="Render every one of the 263 observed frame/track anchors.",
    )
    parser.add_argument(
        "--compact-batch",
        action="store_true",
        help=(
            "With --all-anchors, keep one global gzip mapping table and one "
            "suite validation instead of per-anchor sidecars."
        ),
    )
    parser.add_argument(
        "--palette-colors",
        type=int,
        default=128,
        help="Number of discrete viridis colors used over the fixed [0,1] range.",
    )
    parser.add_argument(
        "--rebuild-profile-cache",
        action="store_true",
        help="Rebuild the formal projected-ellipse profile cache.",
    )
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def write_rows_gzip(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_features(
    path: Path, kept_rows: Sequence[Mapping[str, str]]
) -> list[Feature]:
    rows = read_rows(path)
    if len(rows) != 263:
        raise ValueError(f"expected 263 collapsed frame/track features, got {len(rows)}")
    halfwidths: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in kept_rows:
        halfwidths[(int(row["frame"]), int(row["track"]))].append(
            float(row["halfwidth_q"])
        )
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)
    features: list[Feature] = []
    for frame, items in by_frame.items():
        ordered = sorted(items, key=lambda row: float(row["two_theta_median_deg"]))
        for local_index, row in enumerate(ordered, start=1):
            features.append(
                Feature(
                    frame=frame,
                    pressure_gpa=float(row["pressure_GPa"]),
                    orientation=row["orientation"],
                    branch=row["branch"],
                    track=int(row["track"]),
                    local_peak_index=local_index,
                    q=float(row["q_median_A^-1"]),
                    halfwidth_q=float(
                        np.median(halfwidths[(frame, int(row["track"]))])
                    ),
                    two_theta_deg=float(row["two_theta_median_deg"]),
                    scalar_area=float(
                        row["normalized_area_median_counts_per_s_per_pixel"]
                    ),
                    n_observations=int(row["n_observations"]),
                    duplicate=bool(int(row["duplicate_observation_flag"])),
                )
            )
    keys = {(item.frame, item.track) for item in features}
    if len(keys) != len(features):
        raise ValueError("frame/track feature keys are not unique")
    return features


def build_supports(
    features: Sequence[Feature],
) -> dict[str, tuple[tuple[float, float], ...]]:
    supports: dict[str, tuple[tuple[float, float], ...]] = {}
    for feature in features:
        q = feature.q
        half = feature.halfwidth_q
        left = float(
            formal.q_to_two_theta(q - half, formal.SINGLE_WAVELENGTH_A)
        )
        right = float(
            formal.q_to_two_theta(q + half, formal.SINGLE_WAVELENGTH_A)
        )
        supports[feature.uid] = ((min(left, right), max(left, right)),)
    return supports


def assign_lanes(
    features: Sequence[Feature],
    supports: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, int]:
    occupied_by_lane: list[list[tuple[float, float]]] = []
    assignment: dict[str, int] = {}
    ordered = sorted(
        features,
        key=lambda item: min(left for left, _ in supports[item.uid]),
    )
    for feature in ordered:
        intervals = list(supports[feature.uid])
        for lane, occupied in enumerate(occupied_by_lane):
            if not interval_sets_overlap(intervals, occupied):
                assignment[feature.uid] = lane
                occupied.extend(intervals)
                break
        else:
            assignment[feature.uid] = len(occupied_by_lane)
            occupied_by_lane.append(intervals)
    return assignment


def elliptical_projection(
    q_center: float,
    halfwidth_q: float,
    grid: np.ndarray,
    support: Sequence[tuple[float, float]],
    exact_scalar: float,
) -> np.ndarray:
    """Return a projected elliptical kernel with exact 2theta integral."""

    if halfwidth_q <= 0.0 or exact_scalar < 0.0:
        raise ValueError("formal halfwidth/scalar must be nonnegative")
    theta_rad = np.radians(grid)
    q_values = 4.0 * np.pi * np.sin(0.5 * theta_rad) / formal.SINGLE_WAVELENGTH_A
    u = (q_values - q_center) / halfwidth_q
    profile = np.zeros_like(grid)
    support_mask = np.zeros_like(grid, dtype=bool)
    for left, right in support:
        support_mask |= (grid >= left - 1e-12) & (grid <= right + 1e-12)
    inside = (np.abs(u) <= 1.0) & support_mask
    # Projection of a uniform 2-D ellipse onto its radial coordinate is a
    # semicircle.  dq/d(2theta_deg) converts the q-density to a 2theta-density.
    dq_dtheta_deg = (
        (2.0 * np.pi / formal.SINGLE_WAVELENGTH_A)
        * np.cos(0.5 * theta_rad)
        * (np.pi / 180.0)
    )
    profile[inside] = (
        np.sqrt(np.maximum(1.0 - u[inside] ** 2, 0.0))
        * dq_dtheta_deg[inside]
    )
    integral = float(np.trapezoid(profile, grid))
    if exact_scalar > 0.0 and integral > 0.0:
        profile *= exact_scalar / integral
    elif exact_scalar == 0.0:
        profile[:] = 0.0
    else:
        raise RuntimeError(
            f"positive formal scalar cannot be represented: {exact_scalar=} {integral=}"
        )
    return profile


def _read_profile_cache(
    npz_path: Path,
    audit_path: Path,
    features: Sequence[Feature],
    supports: Mapping[str, Sequence[tuple[float, float]]],
) -> ProfileBundle:
    loaded = np.load(npz_path, allow_pickle=False)
    grid = np.asarray(loaded["grid"], dtype=float)
    uids = [str(value) for value in loaded["uids"]]
    profiles = np.asarray(loaded["profiles"], dtype=float)
    if profiles.shape != (len(features), grid.size) or len(uids) != len(features):
        raise ValueError("single-crystal profile cache shape mismatch")
    displayed = {uid: profiles[index] for index, uid in enumerate(uids)}
    feature_by_uid = {item.uid: item for item in features}
    if set(displayed) != set(feature_by_uid):
        raise ValueError("single-crystal profile cache feature keys drifted")
    by_frame: dict[int, list[Feature]] = defaultdict(list)
    for item in features:
        by_frame[item.frame].append(item)
    composites = {
        frame: np.sum(
            np.stack([displayed[item.uid] for item in items], axis=0), axis=0
        )
        for frame, items in by_frame.items()
    }
    traces = tuple(
        FrameTrace(
            frame=frame,
            pressure_gpa=by_frame[frame][0].pressure_gpa,
            orientation=by_frame[frame][0].orientation,
            branch=by_frame[frame][0].branch,
            x=grid,
            displayed=composites[frame],
        )
        for frame in sorted(
            by_frame,
            key=lambda value: by_frame[value][0].pressure_gpa,
            reverse=True,
        )
    )
    return ProfileBundle(
        grid=grid,
        displayed_profiles=displayed,
        traces=traces,
        supports={key: tuple(value) for key, value in supports.items()},
        audit=json.loads(audit_path.read_text(encoding="utf-8")),
    )


def reconstruct_profiles(
    *,
    mode: str,
    output_root: Path,
    features: Sequence[Feature],
    supports: Mapping[str, Sequence[tuple[float, float]]],
    rebuild: bool,
) -> ProfileBundle:
    """Build/cache 263 exact-scalar ellipse projections and 12 composites."""

    cache_root = output_root / "single_crystal" / mode / "_formal_profile_cache"
    npz_path = cache_root / "profiles.npz"
    audit_path = cache_root / "PROFILE_VALIDATION.json"
    if npz_path.is_file() and audit_path.is_file() and not rebuild:
        return _read_profile_cache(npz_path, audit_path, features, supports)

    support_left = min(left for values in supports.values() for left, _ in values)
    support_right = max(right for values in supports.values() for _, right in values)
    grid = np.linspace(support_left, support_right, GRID_POINTS, dtype=float)
    collapsed: dict[str, np.ndarray] = {}
    area_errors: list[float] = []
    for feature in features:
        profile = elliptical_projection(
            feature.q,
            feature.halfwidth_q,
            grid,
            supports[feature.uid],
            feature.scalar_area,
        )
        collapsed[feature.uid] = profile
        area_errors.append(
            abs(float(np.trapezoid(profile, grid)) - feature.scalar_area)
        )

    by_frame: dict[int, list[Feature]] = defaultdict(list)
    for item in features:
        by_frame[item.frame].append(item)
    composites = {
        frame: np.sum(
            np.stack([collapsed[item.uid] for item in items], axis=0), axis=0
        )
        for frame, items in by_frame.items()
    }
    shared_max = max(float(np.max(value)) for value in composites.values())
    if not np.isfinite(shared_max) or shared_max <= 0.0:
        raise RuntimeError("single-crystal composites have no positive profile")
    displayed = {
        uid: np.clip(profile / shared_max, 0.0, 1.0)
        for uid, profile in collapsed.items()
    }
    displayed_composites = {
        frame: np.clip(profile / shared_max, 0.0, 1.0)
        for frame, profile in composites.items()
    }
    traces = tuple(
        FrameTrace(
            frame=frame,
            pressure_gpa=by_frame[frame][0].pressure_gpa,
            orientation=by_frame[frame][0].orientation,
            branch=by_frame[frame][0].branch,
            x=grid,
            displayed=displayed_composites[frame],
        )
        for frame in sorted(
            by_frame,
            key=lambda value: by_frame[value][0].pressure_gpa,
            reverse=True,
        )
    )
    audit: dict[str, Any] = {
        "status": "PASS",
        "mode": mode,
        "collapsed_frame_track_profiles": len(collapsed),
        "pressure_frame_composites": len(traces),
        "profile_definition": (
            "unit-integral radial projection of a uniform elliptical kernel on "
            "the formal q_median +/- median(halfwidth_q) support, including the "
            "q-to-2theta Jacobian; scaled to the exact frame_track_features "
            "normalized-area median consumed by correlation"
        ),
        "raw_profile_claim": False,
        "duplicate_halfwidth_rule": "median formal halfwidth_q per frame/track",
        "aggregation_within_pressure_frame": "sum all collapsed formal profiles",
        "shared_display_scale": shared_max,
        "max_abs_collapsed_profile_area_error": max(area_errors, default=math.inf),
        "grid_points": int(grid.size),
        "grid_min_two_theta_deg": float(grid[0]),
        "grid_max_two_theta_deg": float(grid[-1]),
        "cache_npz": str(npz_path.resolve()),
    }
    if audit["max_abs_collapsed_profile_area_error"] > 1e-10:
        raise RuntimeError(f"profile scalar reconstruction failed: {audit}")

    ordered_uids = [item.uid for item in features]
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        grid=grid,
        uids=np.asarray(ordered_uids),
        profiles=np.stack([displayed[uid] for uid in ordered_uids], axis=0),
    )
    write_json(audit_path, audit)
    return ProfileBundle(
        grid=grid,
        displayed_profiles=displayed,
        traces=traces,
        supports={key: tuple(value) for key, value in supports.items()},
        audit=audit,
    )


def load_scores(
    npz_path: Path,
    family: str,
    track: int,
    anchor_frame: int,
) -> tuple[dict[int, float], list[int]]:
    data = np.load(npz_path, allow_pickle=False)
    tracks = [int(value) for value in data["track_ids"]]
    frames = [int(value) for value in data["frame_ids"]]
    if track not in tracks or anchor_frame not in frames:
        raise ValueError("anchor track/frame is outside the formal matrix registry")
    matrix_key = (
        "normalized_area_similarity" if family == "roi_area" else "location_similarity"
    )
    matrix = np.asarray(data[matrix_key], dtype=float)[tracks.index(track)]
    anchor_index = frames.index(anchor_frame)
    if not np.isfinite(matrix[anchor_index, anchor_index]):
        raise ValueError("anchor track is not observed in the requested anchor frame")
    return {
        frame: float(matrix[anchor_index, index])
        for index, frame in enumerate(frames)
        if np.isfinite(matrix[anchor_index, index])
    }, frames


def load_all_score_rows(
    npz_path: Path, family: str
) -> tuple[dict[tuple[int, int], dict[int, float]], list[int]]:
    data = np.load(npz_path, allow_pickle=False)
    tracks = [int(value) for value in data["track_ids"]]
    frames = [int(value) for value in data["frame_ids"]]
    matrix_key = (
        "normalized_area_similarity" if family == "roi_area" else "location_similarity"
    )
    cube = np.asarray(data[matrix_key], dtype=float)
    rows: dict[tuple[int, int], dict[int, float]] = {}
    for track_index, track in enumerate(tracks):
        for anchor_index, anchor_frame in enumerate(frames):
            if not np.isfinite(cube[track_index, anchor_index, anchor_index]):
                continue
            rows[(track, anchor_frame)] = {
                frame: float(cube[track_index, anchor_index, target_index])
                for target_index, frame in enumerate(frames)
                if np.isfinite(cube[track_index, anchor_index, target_index])
            }
    return rows, frames


def plot_waterfall(
    *,
    output: Path,
    mode: str,
    family: str,
    track: int,
    anchor_frame: int,
    features: Sequence[Feature],
    profiles: ProfileBundle,
    scores: Mapping[int, float],
    palette_colors: int,
    dpi: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_frame: dict[int, list[Feature]] = defaultdict(list)
    for item in features:
        by_frame[item.frame].append(item)
    lookup = {(item.frame, item.track): item for item in features}
    anchor = lookup.get((anchor_frame, track))
    if anchor is None:
        raise ValueError("anchor feature is absent")

    if palette_colors < 2:
        raise ValueError("palette_colors must be at least 2")
    cmap = plt.colormaps["viridis"].resampled(palette_colors)
    norm = Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(18.0, 11.4), constrained_layout=False)
    left_limit = min(
        left for values in profiles.supports.values() for left, _ in values
    ) - 0.2
    right_limit = max(
        right for values in profiles.supports.values() for _, right in values
    ) + 0.35
    for left, right in profiles.supports[anchor.uid]:
        ax.axvspan(left, right, color="#d8a620", alpha=0.075, zorder=0)
        ax.axvline(left, color="#9a6b00", linewidth=0.75, linestyle="--", zorder=0)
        ax.axvline(right, color="#9a6b00", linewidth=0.75, linestyle="--", zorder=0)

    baselines: list[float] = []
    ylabels: list[str] = []
    mappings: list[dict[str, Any]] = []
    max_lanes = 0
    compared_positive = 0
    compared_zero = 0
    anchor_cells = 0
    neutral_features = 0
    observed_target_frames: list[int] = []

    for row_index, trace in enumerate(profiles.traces):
        baseline = float(len(profiles.traces) - 1 - row_index) * ROW_SPACING
        baselines.append(baseline)
        row_features = sorted(
            by_frame[trace.frame], key=lambda item: item.local_peak_index
        )
        lanes = assign_lanes(row_features, profiles.supports)
        max_lanes = max(max_lanes, max(lanes.values(), default=-1) + 1)
        branch = trace.branch.replace("deg", "°").replace("_", " ")
        ylabels.append(
            f"{trace.pressure_gpa:g} GPa · f{trace.frame:04d} · {branch} · "
            f"{len(row_features)} peaks"
        )
        ax.axhline(baseline, color="#dddddd", linewidth=0.55, zorder=0)
        y_curve = baseline + TRACE_HEIGHT * trace.displayed

        for feature in row_features:
            lane = lanes[feature.uid]
            if feature.track != track:
                status = "not_in_selected_track_matrix"
                shown_score = math.nan
                color: Any = "#b7b7b7"
                fill_alpha = 0.0
                neutral_features += 1
            elif feature.frame == anchor_frame:
                status = "anchor_self"
                shown_score = 1.0
                color = cmap(norm(shown_score))
                fill_alpha = 0.5
                anchor_cells += 1
                observed_target_frames.append(feature.frame)
            else:
                shown_score = scores.get(feature.frame, math.nan)
                if not np.isfinite(shown_score):
                    raise RuntimeError(
                        f"observed selected-track feature has no matrix score: {feature.uid}"
                    )
                status = "compared_zero" if shown_score == 0.0 else "compared_positive"
                color = cmap(norm(shown_score))
                fill_alpha = 0.5
                compared_zero += int(status == "compared_zero")
                compared_positive += int(status == "compared_positive")
                observed_target_frames.append(feature.frame)

            for component_index, (left, right) in enumerate(
                profiles.supports[feature.uid]
            ):
                native = (trace.x >= left) & (trace.x <= right)
                if fill_alpha > 0.0 and np.count_nonzero(native) >= 2:
                    point = profiles.displayed_profiles[feature.uid]
                    ax.fill_between(
                        trace.x[native],
                        baseline,
                        baseline + TRACE_HEIGHT * point[native],
                        color=color,
                        alpha=fill_alpha,
                        linewidth=0.0,
                        zorder=2,
                    )
                ribbon_top = baseline - RIBBON_TOP_GAP - lane * (
                    RIBBON_HEIGHT + RIBBON_GAP
                )
                ax.add_patch(
                    Rectangle(
                        (left, ribbon_top - RIBBON_HEIGHT),
                        right - left,
                        RIBBON_HEIGHT,
                        facecolor=(color if fill_alpha > 0.0 else "none"),
                        edgecolor=("#222222" if status == "anchor_self" else color),
                        linewidth=(1.05 if status == "anchor_self" else 0.5),
                        alpha=(0.98 if fill_alpha > 0.0 else 0.75),
                        zorder=4,
                    )
                )
                mappings.append(
                    {
                        "anchor_uid": anchor.uid,
                        "mode": mode,
                        "family": family,
                        "anchor_track": track,
                        "anchor_frame": anchor_frame,
                        "frame": feature.frame,
                        "pressure_gpa": feature.pressure_gpa,
                        "orientation": feature.orientation,
                        "branch": feature.branch,
                        "local_peak_index": feature.local_peak_index,
                        "track": feature.track,
                        "feature_uid": feature.uid,
                        "two_theta_center_deg": feature.two_theta_deg,
                        "support_component_index": component_index,
                        "support_left_deg": left,
                        "support_right_deg": right,
                        "ribbon_lane_0based": lane,
                        "correlation": (
                            "" if not np.isfinite(shown_score) else shown_score
                        ),
                        "status": status,
                    }
                )
            if status == "anchor_self":
                ax.annotate(
                    "REF",
                    xy=(feature.two_theta_deg, baseline + TRACE_HEIGHT + 0.015),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    fontweight="bold",
                    color="#6b4a00",
                    zorder=7,
                )

        is_anchor_row = trace.frame == anchor_frame
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

    if RIBBON_TOP_GAP + max_lanes * (RIBBON_HEIGHT + RIBBON_GAP) >= ROW_SPACING:
        raise RuntimeError("single-crystal ribbon bands would overlap adjacent rows")
    ax.set_xlim(left_limit, right_limit)
    ax.set_ylim(
        -RIBBON_TOP_GAP - max_lanes * (RIBBON_HEIGHT + RIBBON_GAP) - 0.12,
        baselines[-1]
        + (len(profiles.traces) - 1) * ROW_SPACING
        + TRACE_HEIGHT
        + 0.12,
    )
    ax.set_yticks(baselines, ylabels, fontsize=8.2)
    ax.set_xlabel(r"$2\theta$ (degrees)", fontsize=12)
    ax.set_ylabel(
        "Pressure/frame rows (descending); fixed offsets prevent overlap",
        fontsize=11,
    )
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    mode_label = "Log-squared" if mode == "log_squared" else "Exp-squared"
    family_label = "ROI-area" if family == "roi_area" else "location"
    fig.suptitle(
        f"Single-crystal {mode_label} {family_label} correlation waterfall",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    ax.set_title(
        f"anchor track {track} · f{anchor_frame:04d} · "
        f"{anchor.pressure_gpa:g} GPa · peak {anchor.local_peak_index} · "
        f"2θ={anchor.two_theta_deg:.4f}°",
        fontsize=11,
        pad=12,
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.018, fraction=0.028
    )
    colorbar.set_label(f"Denoised {family_label} similarity", fontsize=10)
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    fig.text(
        0.12,
        0.012,
        "Gray line = sum of all formal single-crystal projected elliptical "
        "kernels in that frame; every kernel uses the formal q-width support "
        "and integrates to the exact transformed scalar used by correlation. "
        "Color marks only the selected global track's "
        "cross-frame score. Gray outline ribbons are other formal tracks, not "
        "missing values. Dark purple = a real score of 0; an absent selected "
        "track produces no colored ribbon in that frame.",
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#555555",
        wrap=True,
    )
    fig.subplots_adjust(left=0.20, right=0.90, top=0.92, bottom=0.085)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    quantize_png(output, palette_colors)

    expected_target_frames = sorted(
        item.frame for item in features if item.track == track
    )
    if sorted(observed_target_frames) != expected_target_frames:
        raise RuntimeError("selected-track target mapping is incomplete")
    audit: dict[str, Any] = {
        "status": "PASS",
        "mode": mode,
        "family": family,
        "anchor_track": track,
        "anchor_frame": anchor_frame,
        "anchor_pressure_gpa": anchor.pressure_gpa,
        "anchor_local_peak_index": anchor.local_peak_index,
        "pressure_frame_rows": len(profiles.traces),
        "registered_collapsed_features": len(features),
        "selected_track_observed_frames": expected_target_frames,
        "selected_track_support": len(expected_target_frames),
        "compared_positive_cells": compared_positive,
        "compared_zero_cells": compared_zero,
        "anchor_self_cells": anchor_cells,
        "neutral_other_track_features": neutral_features,
        "maximum_ribbon_lanes": max_lanes,
        "strictly_nonoverlapping_trace_and_ribbon_bands": True,
        "trace_height": TRACE_HEIGHT,
        "row_spacing": ROW_SPACING,
        "fixed_color_range": [0.0, 1.0],
        "palette_colors": palette_colors,
        "mapping_rows": len(mappings),
        "formal_profile_reconstruction": dict(profiles.audit),
        "output_png": str(output.resolve()),
    }
    if anchor_cells != 1:
        raise RuntimeError("single-crystal anchor self mapping failed")
    if compared_positive + compared_zero != len(expected_target_frames) - 1:
        raise RuntimeError("single-crystal compared-cell count failed")
    if neutral_features != len(features) - len(expected_target_frames):
        raise RuntimeError("single-crystal neutral feature count failed")
    return mappings, audit


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compact_batch and not args.all_anchors:
        raise ValueError("--compact-batch requires --all-anchors")
    source_root = (
        args.comparison_root / "_sources" / args.mode / "single_roi"
    )
    analysis_root = source_root / "single_crystal/per_peak_all_frames"
    kept_path = (
        args.data_root / "Single Crystal (Cell 29)/Masked/kept_obs.csv"
    )
    kept_rows = read_rows(kept_path)
    if len(kept_rows) != 275:
        raise ValueError("expected the formal 275 single-crystal observations")
    features = load_features(
        analysis_root / "frame_track_features.csv", kept_rows
    )
    supports = build_supports(features)
    missing_supports = sorted({item.uid for item in features} - set(supports))
    if missing_supports:
        raise ValueError(f"features without formal support: {missing_supports[:5]}")
    profiles = reconstruct_profiles(
        mode=args.mode,
        output_root=args.out_dir,
        features=features,
        supports=supports,
        rebuild=args.rebuild_profile_cache,
    )
    matrix_path = analysis_root / "per_track_matrices.npz"

    if not args.all_anchors:
        scores, _ = load_scores(
            matrix_path,
            args.family,
            args.track,
            args.anchor_frame,
        )
        token = (
            f"track_{args.track:03d}_anchor_f{args.anchor_frame:04d}_"
            f"{args.family}"
        )
        out_dir = (
            args.out_dir
            / "single_crystal"
            / args.mode
            / token
            / "formal_composite"
        )
        output = out_dir / f"{token}_formal_composite_correlation_waterfall.png"
        mappings, audit = plot_waterfall(
            output=output,
            mode=args.mode,
            family=args.family,
            track=args.track,
            anchor_frame=args.anchor_frame,
            features=features,
            profiles=profiles,
            scores=scores,
            palette_colors=args.palette_colors,
            dpi=args.dpi,
        )
        write_rows(out_dir / "peak_color_mapping.csv", mappings)
        write_json(out_dir / "VALIDATION.json", audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0

    score_rows, _ = load_all_score_rows(matrix_path, args.family)
    anchors = sorted(
        features,
        key=lambda item: (item.pressure_gpa, item.frame, item.local_peak_index),
    )
    if len(anchors) != 263 or len(score_rows) != 263:
        raise RuntimeError("all-anchor registry must contain exactly 263 entries")
    suite_root = (
        args.out_dir
        / "single_crystal"
        / args.mode
        / args.family
        / "formal_composite"
    )
    heatmap_root = suite_root / "heatmaps"
    all_mappings: list[dict[str, Any]] = []
    anchor_index_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    support_by_track: dict[int, int] = defaultdict(int)
    for feature in features:
        support_by_track[feature.track] += 1

    for sequence, anchor in enumerate(anchors, start=1):
        token = (
            f"anchor_{sequence:03d}_{anchor.uid}_P{anchor.pressure_gpa:g}_"
            f"peak{anchor.local_peak_index:02d}_{args.family}"
        ).replace(".", "p")
        output = heatmap_root / f"{token}.png"
        mappings, audit = plot_waterfall(
            output=output,
            mode=args.mode,
            family=args.family,
            track=anchor.track,
            anchor_frame=anchor.frame,
            features=features,
            profiles=profiles,
            scores=score_rows[(anchor.track, anchor.frame)],
            palette_colors=args.palette_colors,
            dpi=args.dpi,
        )
        for row in mappings:
            row["anchor_sequence_1based"] = sequence
            row["waterfall_png"] = str(output.resolve())
        all_mappings.extend(mappings)
        audits.append(audit)
        anchor_index_rows.append(
            {
                "anchor_sequence_1based": sequence,
                "anchor_uid": anchor.uid,
                "track": anchor.track,
                "frame": anchor.frame,
                "pressure_gpa": anchor.pressure_gpa,
                "local_peak_index": anchor.local_peak_index,
                "two_theta_deg": anchor.two_theta_deg,
                "track_frame_support": support_by_track[anchor.track],
                "compared_cross_frame_cells": (
                    audit["compared_positive_cells"]
                    + audit["compared_zero_cells"]
                ),
                "waterfall_png": str(output.resolve()),
                "validation_status": audit["status"],
            }
        )
        if not args.compact_batch:
            sidecar = suite_root / "per_anchor" / token
            write_rows(sidecar / "peak_color_mapping.csv", mappings)
            write_json(sidecar / "VALIDATION.json", audit)

    directed_compared = sum(
        int(item["compared_positive_cells"])
        + int(item["compared_zero_cells"])
        for item in audits
    )
    singleton_tracks = sorted(
        track for track, support in support_by_track.items() if support == 1
    )
    singleton_anchors = sum(
        support_by_track[item.track] == 1 for item in anchors
    )
    suite_audit: dict[str, Any] = {
        "status": "PASS",
        "mode": args.mode,
        "family": args.family,
        "formal_anchor_count": len(anchors),
        "formal_feature_count": len(features),
        "pressure_frame_rows_per_waterfall": len(profiles.traces),
        "directed_cross_frame_compared_cells": directed_compared,
        "expected_directed_cross_frame_compared_cells": 1306,
        "anchor_self_cells": sum(int(item["anchor_self_cells"]) for item in audits),
        "singleton_track_count": len(singleton_tracks),
        "singleton_anchor_count": singleton_anchors,
        "singleton_tracks": singleton_tracks,
        "global_mapping_rows": len(all_mappings),
        "expected_global_mapping_rows": len(features) * len(anchors),
        "palette_colors": args.palette_colors,
        "fixed_color_range": [0.0, 1.0],
        "strictly_nonoverlapping_all_waterfalls": all(
            bool(item["strictly_nonoverlapping_trace_and_ribbon_bands"])
            for item in audits
        ),
        "formal_profile_reconstruction": dict(profiles.audit),
        "heatmap_directory": str(heatmap_root.resolve()),
    }
    expected_checks = (
        directed_compared == 1306
        and len(singleton_tracks) == 26
        and singleton_anchors == 26
        and suite_audit["anchor_self_cells"] == 263
        and len(all_mappings) == 263 * 263
        and suite_audit["strictly_nonoverlapping_all_waterfalls"]
    )
    if not expected_checks:
        raise RuntimeError(f"single-crystal suite validation failed: {suite_audit}")
    write_rows(suite_root / "ANCHOR_INDEX.csv", anchor_index_rows)
    write_rows_gzip(suite_root / "peak_color_mapping.csv.gz", all_mappings)
    write_json(suite_root / "SUITE_VALIDATION.json", suite_audit)
    print(json.dumps(suite_audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
