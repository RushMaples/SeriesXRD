#!/usr/bin/env python3
"""Build track-independent, all-peak cross-frame correlation maps.

Each peak is a frame-local observation.  For every distinct frame pair
``(i, j)`` this program evaluates the complete Cartesian product

    C_ij[m, n] = similarity(p_i,m, p_j,n)

without first matching angles, grouping peaks, or requiring a peak to recur.
The canonical pair table stores only ``i < j``.  Presentation maps use one
anchor peak at a time: every registered frame occupies one row and columns are
the frame-local slots ``peak 1 ... peak M``, where ``M`` is the largest peak
count in any frame in that dataset.  Missing slots, zero-peak frames, and the
anchor peak's own frame are NaN/white rather than numerical zero.

Three peak registries are exported:

* single-crystal all-frame 1D spot-channel candidates, freshly detected in
  every one of the 28 available frames and numerically integrated from the
  background-subtracted signal;
* the existing single-crystal curated 2D ROI observations, retained as an
  auditable companion and using raw-TIFF ROI integrations; and
* all powder 2D spot observations, including untracked/short-lived peaks,
  using the source pipeline's integrated counts above the ring median.

Track-independent uniform window results are recomputed from the original XY
profiles on exact nominal angle windows ``0-5, 1-6, 2-7, ...``.  The same suite
contains across-frame and within-frame window-to-window correlations for
single crystal and powder (sample and fit-control channels).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]

POSITION_TOLERANCE_DEG = 0.06
SINGLE_WAVELENGTH_A = 0.4133
POWDER_WAVELENGTH_A = 0.3066

DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "correlations" / "UOTe XRD Data Refinement"
DEFAULT_SINGLE_MANIFEST = (
    WORKSPACE_ROOT
    / "correlations"
    / "manifests"
    / "uote_single_crystal_uniform_v2_1_manifest.csv"
)
DEFAULT_SINGLE_PROFILE = SCRIPT_DIR / "configs" / "uniform-correlation-v2.json"
DEFAULT_SINGLE_UNCERTAINTY = (
    WORKSPACE_ROOT
    / "correlations"
    / "results"
    / "uote_all_peak_frame_correlation_suite_20260729_v3_lower_triangle"
    / "single_crystal"
    / "curated_2d_roi_peak_maps"
    / "peak_registry.csv"
)
DEFAULT_POWDER_OBSERVATIONS = (
    DEFAULT_DATA_ROOT / "Powder Scan" / "Track Analysis" / "spot_observations.csv"
)
DEFAULT_POWDER_MANIFEST = DEFAULT_DATA_ROOT / "Powder Scan" / "Reduced .xy" / "manifest.csv"
DEFAULT_SINGLE_WINDOW_INPUT_ROOT = (
    WORKSPACE_ROOT / "correlations" / "UOTe Single Crystal Reduced"
)
DEFAULT_SINGLE_WINDOW_PROFILE = (
    SCRIPT_DIR / "configs" / "uniform-correlation-v2.1.json"
)
DEFAULT_POWDER_WINDOW_INPUT_ROOT = (
    WORKSPACE_ROOT / "correlations" / "uote_xy_handoff 2"
)
DEFAULT_POWDER_WINDOW_MANIFEST = (
    DEFAULT_POWDER_WINDOW_INPUT_ROOT / "manifest.csv"
)
DEFAULT_POWDER_WINDOW_PROFILE = (
    SCRIPT_DIR / "configs" / "uniform-correlation-v2.json"
)
DEFAULT_SINGLE_WINDOW_SOURCE = (
    WORKSPACE_ROOT
    / "correlations"
    / "results"
    / "uote_single_crystal_correlations_uniform_v2_1_20260714"
)
DEFAULT_POWDER_WINDOW_SOURCE = (
    WORKSPACE_ROOT
    / "correlations"
    / "results"
    / "uote_xy_handoff2_correlations_uniform_v2_20260714"
)
DEFAULT_WINDOW_SUITE_SOURCE = (
    WORKSPACE_ROOT
    / "correlations"
    / "results"
    / "uote_all_peak_frame_correlation_suite_20260729_v3_lower_triangle"
)
WINDOW_SUITE_PAYLOADS = (
    Path("single_crystal/windows"),
    Path("powder/windows"),
    Path("window_full_symmetric_audit"),
    Path("window_quicklooks"),
    Path("window_provenance"),
    Path("LOWER_TRIANGLE_METHODS.md"),
    Path("WINDOW_METHODS.md"),
    Path("window_lower_triangle_index.csv"),
)
EXPECTED_V3_WINDOW_PAYLOAD_FILES = 4724
EXPECTED_V3_WINDOW_PAYLOAD_BYTES = 203_472_516
EXPECTED_V3_WINDOW_PAYLOAD_DIGEST = (
    "0f8991d6d8882bda93d870e0386c8702287aa03673e35d557f4474e068e5eddb"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--single-manifest", type=Path, default=DEFAULT_SINGLE_MANIFEST)
    parser.add_argument("--single-profile", type=Path, default=DEFAULT_SINGLE_PROFILE)
    parser.add_argument(
        "--single-uncertainty-table",
        type=Path,
        default=DEFAULT_SINGLE_UNCERTAINTY,
    )
    parser.add_argument(
        "--powder-observations",
        type=Path,
        default=DEFAULT_POWDER_OBSERVATIONS,
    )
    parser.add_argument("--powder-manifest", type=Path, default=DEFAULT_POWDER_MANIFEST)
    parser.add_argument(
        "--single-window-input-root",
        type=Path,
        default=DEFAULT_SINGLE_WINDOW_INPUT_ROOT,
    )
    parser.add_argument(
        "--single-window-profile",
        type=Path,
        default=DEFAULT_SINGLE_WINDOW_PROFILE,
    )
    parser.add_argument(
        "--powder-window-input-root",
        type=Path,
        default=DEFAULT_POWDER_WINDOW_INPUT_ROOT,
    )
    parser.add_argument(
        "--powder-window-manifest",
        type=Path,
        default=DEFAULT_POWDER_WINDOW_MANIFEST,
    )
    parser.add_argument(
        "--powder-window-profile",
        type=Path,
        default=DEFAULT_POWDER_WINDOW_PROFILE,
    )
    parser.add_argument(
        "--window-workers",
        type=int,
        default=8,
        help="Parallel workers used while rebuilding the integer-angle windows.",
    )
    parser.add_argument(
        "--position-tolerance-deg",
        type=float,
        default=POSITION_TOLERANCE_DEG,
    )
    parser.add_argument(
        "--no-window-copy",
        action="store_true",
        help=(
            "Skip window recomputation (deprecated option name; produces only a "
            "partial peak-map run)."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if fields is None:
        fields = list(materialized[0]) if materialized else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if fields:
            writer.writeheader()
            for row in materialized:
                writer.writerow({key: csv_value(row.get(key)) for key in fields})


def write_csv_gz(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if fields is None:
        fields = list(materialized[0]) if materialized else []
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if fields:
            writer.writeheader()
            for row in materialized:
                writer.writerow({key: csv_value(row.get(key)) for key in fields})


def csv_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return "" if not np.isfinite(float(value)) else f"{float(value):.12g}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return int(bool(value))
    if value is None:
        return ""
    return value


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def q_to_two_theta(q_a_inv: float | np.ndarray, wavelength_a: float) -> np.ndarray:
    q = np.asarray(q_a_inv, dtype=float)
    argument = q * float(wavelength_a) / (4.0 * np.pi)
    result = np.full(q.shape, np.nan, dtype=float)
    valid = np.isfinite(argument) & (np.abs(argument) <= 1.0)
    result[valid] = np.degrees(2.0 * np.arcsin(argument[valid]))
    return result


def location_similarity_matrix(
    left_two_theta: Sequence[float],
    right_two_theta: Sequence[float],
    tolerance: float = POSITION_TOLERANCE_DEG,
) -> np.ndarray:
    """Return the full rectangular location-similarity Cartesian product."""
    left = np.asarray(left_two_theta, dtype=float).reshape(-1, 1)
    right = np.asarray(right_two_theta, dtype=float).reshape(1, -1)
    resolved = max(float(tolerance), np.finfo(float).eps)
    matrix = 1.0 - np.abs(left - right) / resolved
    matrix = np.clip(matrix, 0.0, 1.0)
    matrix[~(np.isfinite(left) & np.isfinite(right))] = np.nan
    return matrix


def area_similarity_matrix(
    left_area: Sequence[float],
    right_area: Sequence[float],
) -> np.ndarray:
    """Return ``min(integrated area)/max(integrated area)`` for every pair."""
    left = np.asarray(left_area, dtype=float).reshape(-1, 1)
    right = np.asarray(right_area, dtype=float).reshape(1, -1)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    finite_nonnegative = (
        np.isfinite(left) & np.isfinite(right) & (left >= 0.0) & (right >= 0.0)
    )
    valid = finite_nonnegative & (high > 0.0)
    result = np.full(np.broadcast_shapes(left.shape, right.shape), np.nan, dtype=float)
    np.divide(low, high, out=result, where=valid)
    # Two numerically integrated zero-area peaks are equal in area.  Keeping
    # this convention explicit avoids an undefined 0/0 cell.
    result[finite_nonnegative & (high == 0.0)] = 1.0
    return result


def strict_lower_triangle(matrix: np.ndarray) -> np.ndarray:
    """Keep only cells below the diagonal of a square matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("strict lower-triangle view requires a square 2D matrix")
    result = values.copy()
    result[np.triu_indices(result.shape[0], k=0)] = np.nan
    return result


def _assert_symmetric_matrix(matrix: np.ndarray, label: str) -> None:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{label} is not square: {values.shape}")
    if not np.array_equal(np.isfinite(values), np.isfinite(values.T)):
        raise ValueError(f"{label} has an asymmetric finite-value mask")
    finite = np.isfinite(values) & np.isfinite(values.T)
    difference = (
        float(np.max(np.abs(values[finite] - values.T[finite])))
        if np.any(finite)
        else 0.0
    )
    if difference > 1.0e-10:
        raise ValueError(f"{label} is not symmetric; max abs difference={difference}")


def _peak_sort_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
    two_theta = as_float(row.get("two_theta_deg"))
    azimuth = as_float(row.get("azim_deg"))
    source_row = as_int(row.get("obs_row"), 10**12)
    return (
        two_theta if np.isfinite(two_theta) else math.inf,
        azimuth if np.isfinite(azimuth) else math.inf,
        source_row,
    )


def assign_local_peak_ids(
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
) -> list[dict[str, Any]]:
    """Assign deterministic frame-local IDs without consulting ``track``."""
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        item = dict(source)
        frame = as_int(item.get("frame"))
        if frame < 0:
            raise ValueError(f"invalid frame in peak row: {source}")
        by_frame[frame].append(item)
    result: list[dict[str, Any]] = []
    for frame in sorted(by_frame):
        ordered = sorted(by_frame[frame], key=_peak_sort_key)
        for local_index, item in enumerate(ordered, start=1):
            item["dataset"] = dataset
            item["local_peak_index"] = local_index
            item["peak_id"] = f"p{frame},{local_index}"
            result.append(item)
    return result


def group_peaks_by_frame(
    peaks: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in peaks:
        grouped[as_int(row.get("frame"))].append(dict(row))
    for frame in grouped:
        grouped[frame].sort(key=lambda row: as_int(row.get("local_peak_index")))
    return dict(sorted(grouped.items()))


PAIR_FIELDS = [
    "dataset",
    "frame_a",
    "frame_b",
    "scan_a",
    "scan_b",
    "pressure_a_GPa",
    "pressure_b_GPa",
    "orientation_a",
    "orientation_b",
    "cross_orientation",
    "peak_a_id",
    "peak_b_id",
    "local_peak_a",
    "local_peak_b",
    "two_theta_a_deg",
    "two_theta_b_deg",
    "delta_two_theta_deg",
    "q_a_A^-1",
    "q_b_A^-1",
    "azim_a_deg",
    "azim_b_deg",
    "integrated_area_a",
    "integrated_area_b",
    "area_unit_a",
    "area_unit_b",
    "location_similarity",
    "area_similarity",
    "source_track_a",
    "source_track_b",
    "source_state_a",
    "source_state_b",
]


def build_cross_frame_pair_rows(
    peaks_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    tolerance: float = POSITION_TOLERANCE_DEG,
) -> list[dict[str, Any]]:
    """Emit every cross-frame peak Cartesian-product cell for canonical pairs."""
    frames = sorted(int(frame) for frame in peaks_by_frame)
    result: list[dict[str, Any]] = []
    for frame_a, frame_b in combinations(frames, 2):
        left = list(peaks_by_frame[frame_a])
        right = list(peaks_by_frame[frame_b])
        positions = location_similarity_matrix(
            [as_float(row.get("two_theta_deg")) for row in left],
            [as_float(row.get("two_theta_deg")) for row in right],
            tolerance,
        )
        areas = area_similarity_matrix(
            [as_float(row.get("integrated_area")) for row in left],
            [as_float(row.get("integrated_area")) for row in right],
        )
        for left_index, left_row in enumerate(left):
            for right_index, right_row in enumerate(right):
                theta_a = as_float(left_row.get("two_theta_deg"))
                theta_b = as_float(right_row.get("two_theta_deg"))
                orientation_a = str(left_row.get("orientation", ""))
                orientation_b = str(right_row.get("orientation", ""))
                result.append(
                    {
                        "dataset": left_row.get("dataset", right_row.get("dataset", "")),
                        "frame_a": frame_a,
                        "frame_b": frame_b,
                        "scan_a": left_row.get("scan", ""),
                        "scan_b": right_row.get("scan", ""),
                        "pressure_a_GPa": as_float(left_row.get("pressure_GPa")),
                        "pressure_b_GPa": as_float(right_row.get("pressure_GPa")),
                        "orientation_a": orientation_a,
                        "orientation_b": orientation_b,
                        "cross_orientation": int(
                            bool(orientation_a)
                            and bool(orientation_b)
                            and orientation_a != orientation_b
                        ),
                        "peak_a_id": left_row.get("peak_id", ""),
                        "peak_b_id": right_row.get("peak_id", ""),
                        "local_peak_a": as_int(left_row.get("local_peak_index")),
                        "local_peak_b": as_int(right_row.get("local_peak_index")),
                        "two_theta_a_deg": theta_a,
                        "two_theta_b_deg": theta_b,
                        "delta_two_theta_deg": (
                            abs(theta_a - theta_b)
                            if np.isfinite(theta_a) and np.isfinite(theta_b)
                            else math.nan
                        ),
                        "q_a_A^-1": as_float(left_row.get("q_A^-1")),
                        "q_b_A^-1": as_float(right_row.get("q_A^-1")),
                        "azim_a_deg": as_float(left_row.get("azim_deg")),
                        "azim_b_deg": as_float(right_row.get("azim_deg")),
                        "integrated_area_a": as_float(left_row.get("integrated_area")),
                        "integrated_area_b": as_float(right_row.get("integrated_area")),
                        "area_unit_a": left_row.get("area_unit", ""),
                        "area_unit_b": right_row.get("area_unit", ""),
                        "location_similarity": float(positions[left_index, right_index]),
                        "area_similarity": float(areas[left_index, right_index]),
                        "source_track_a": left_row.get("track", ""),
                        "source_track_b": right_row.get("track", ""),
                        "source_state_a": left_row.get("source_state", ""),
                        "source_state_b": right_row.get("source_state", ""),
                    }
                )
    return result


def _trapezoid(values: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, x))
    return float(np.trapz(values, x))


def integrate_independent_peak_areas(
    x: np.ndarray,
    positive_residual: np.ndarray,
    centers: Sequence[float],
    fwhm_values: Sequence[float],
) -> tuple[list[float], list[tuple[float, float]]]:
    """Numerically integrate each local peak without using a fitted-area value."""
    centers_array = np.asarray(centers, dtype=float)
    widths = np.asarray(fwhm_values, dtype=float)
    order = np.argsort(centers_array, kind="stable")
    areas = np.full(len(centers_array), np.nan, dtype=float)
    bounds: list[tuple[float, float]] = [(math.nan, math.nan)] * len(centers_array)
    x = np.asarray(x, dtype=float)
    positive_residual = np.asarray(positive_residual, dtype=float)
    if (
        x.ndim != 1
        or positive_residual.shape != x.shape
        or x.size < 2
        or not np.all(np.diff(x) > 0.0)
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(positive_residual))
    ):
        raise ValueError("integration requires finite, increasing x and residual arrays")
    dx = float(np.median(np.diff(x)))
    for order_index, peak_index in enumerate(order):
        center = float(centers_array[peak_index])
        width = float(widths[peak_index])
        half_width = max(1.5 * width if np.isfinite(width) and width > 0 else 0.0, 3.0 * dx)
        left = center - half_width
        right = center + half_width
        if order_index:
            previous = float(centers_array[order[order_index - 1]])
            left = max(left, 0.5 * (previous + center))
        if order_index + 1 < len(order):
            following = float(centers_array[order[order_index + 1]])
            right = min(right, 0.5 * (center + following))
        left = max(left, float(x[0]))
        right = min(right, float(x[-1]))
        if right > left:
            # Include linearly interpolated ROI endpoints.  This still gives a
            # proper numerical integral when a narrow, neighbor-bounded ROI
            # contains only one sampled x point.
            interior = (x > left) & (x < right)
            integration_x = np.concatenate(([left], x[interior], [right]))
            integration_y = np.interp(integration_x, x, positive_residual)
            areas[peak_index] = _trapezoid(integration_y, integration_x)
        bounds[peak_index] = (left, right)
    return areas.tolist(), bounds


def load_single_all_frame_peaks(
    manifest_path: Path,
    profile_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Blindly redetect every candidate in all 28 single-crystal XY frames."""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import uniform_peak_core as peak_core  # noqa: WPS433
    from uniform_profile_binding import bind_frozen_profile  # noqa: WPS433
    from uniform_xy_input import read_xy_clean  # noqa: WPS433

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    bound = bind_frozen_profile(profile, SINGLE_WAVELENGTH_A)
    config = bound.peak_config
    manifest_rows = read_csv(manifest_path)
    peaks: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    candidate_states: Counter[str] = Counter()
    for source in manifest_rows:
        frame = as_int(source.get("frame"))
        relative = Path(str(source.get("file_path", "")).replace("\\", "/"))
        pattern_path = (manifest_path.parent / relative).resolve()
        x_raw, y_raw, metadata = read_xy_clean(
            pattern_path,
            minimum_points=bound.minimum_points_per_pattern,
        )
        preprocessed = peak_core.preprocess_pattern(x_raw, y_raw, config)
        detected = peak_core.detect_pattern_peaks(
            preprocessed,
            frame=frame,
            scan=str(source.get("scan", "")),
            pressure=as_float(source.get("pressure_GPa")),
            channel="spots",
            config=config,
        )
        usable_fits = [
            fit
            for fit in detected.peaks
            if np.isfinite(fit.two_theta) and np.isfinite(fit.q)
        ]
        areas, bounds = integrate_independent_peak_areas(
            preprocessed.x,
            preprocessed.positive_residual,
            [fit.two_theta for fit in usable_fits],
            [fit.fwhm_two_theta for fit in usable_fits],
        )
        for fit, integrated_area, (left, right) in zip(
            usable_fits,
            areas,
            bounds,
            strict=True,
        ):
            candidate_states[fit.state] += 1
            peaks.append(
                {
                    "frame": frame,
                    "scan": source.get("scan", ""),
                    "orientation": source.get("scan", ""),
                    "pressure_GPa": as_float(source.get("pressure_GPa")),
                    "two_theta_deg": float(fit.two_theta),
                    "q_A^-1": float(fit.q),
                    "azim_deg": math.nan,
                    "integrated_area": integrated_area,
                    "integrated_area_raw": integrated_area,
                    "area_unit": "background_subtracted_1D_intensity_deg",
                    "area_method": (
                        "trapezoid_positive_AsLS_residual_with_interpolated_endpoints"
                        "_and_neighbor_midpoint_boundaries"
                    ),
                    "integration_left_deg": left,
                    "integration_right_deg": right,
                    "fwhm_two_theta_deg": float(fit.fwhm_two_theta),
                    "fwhm_q_A^-1": float(fit.fwhm_q),
                    "obs_row": int(fit.peak_id),
                    "track": "",
                    "source_state": fit.state,
                    "source_reason": fit.reason,
                    "fit_area_not_used": float(fit.area),
                    "fit_model": fit.fit_model,
                    "source_file": str(pattern_path),
                    "source_excluded_from_formal_ladder": as_int(source.get("excluded"), 0),
                    "source_exclusion_reason": source.get("exclusion_reason", ""),
                }
            )
        frame_rows.append(
            {
                "frame": frame,
                "scan": source.get("scan", ""),
                "orientation": source.get("scan", ""),
                "pressure_GPa": as_float(source.get("pressure_GPa")),
                "formal_ladder_included": int(str(source.get("excluded", "0")) == "0"),
                "formal_ladder_exclusion_reason": source.get("exclusion_reason", ""),
                "peak_count": len(usable_fits),
                "pattern_valid": int(detected.pattern_valid),
                "source_file": str(pattern_path),
                "source_header_channel": metadata.get("source_channel", ""),
                "source_header_wavelength_A": metadata.get("wavelength_A", ""),
            }
        )
    audit = {
        "source": "fresh_frame_local_uniform_v2_candidates",
        "manifest_rows": len(manifest_rows),
        "frames": len(frame_rows),
        "peaks": len(peaks),
        "candidate_states": dict(candidate_states),
        "reliable_candidates": candidate_states.get("reliable", 0),
        "audit_state_candidates": len(peaks) - candidate_states.get("reliable", 0),
        "track_used_for_detection_grouping_or_filtering": False,
        "fitted_area_used_for_correlation": False,
        "area_is_numerical_integration": True,
        "profile": str(profile_path.resolve()),
        "profile_semantic_sha256": bound.semantic_sha256,
    }
    return peaks, frame_rows, audit


def load_single_curated_roi_peaks(
    uncertainty_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    observations = read_csv(uncertainty_path)
    manifest = read_csv(manifest_path)
    source_schema = (
        "exported_curated_peak_registry"
        if observations
        and "integrated_area" in observations[0]
        and "integrated_roi_area_counts_per_s" not in observations[0]
        else "raw_uncertainty_observations"
    )
    peaks: list[dict[str, Any]] = []
    for source in observations:
        if source_schema == "exported_curated_peak_registry":
            integrated_area = as_float(source.get("integrated_area"))
            integrated_area_raw = as_float(source.get("integrated_area_raw"))
            source_file = source.get("source_file", "")
        else:
            integrated_area = as_float(
                source.get("integrated_roi_area_counts_per_s")
            )
            integrated_area_raw = as_float(
                source.get("area_counts_recomputed")
            )
            source_file = source.get("raw_tiff", "")
        peaks.append(
            {
                "frame": as_int(source.get("frame")),
                "scan": source.get("scan", ""),
                "orientation": source.get("orientation", ""),
                "pressure_GPa": as_float(source.get("pressure_GPa")),
                "two_theta_deg": as_float(source.get("two_theta_deg")),
                "q_A^-1": as_float(source.get("q_A^-1")),
                "azim_deg": as_float(source.get("azim_deg")),
                "integrated_area": integrated_area,
                "integrated_area_raw": integrated_area_raw,
                "area_unit": source.get(
                    "area_unit", "background_subtracted_2D_ROI_counts_per_s"
                ),
                "area_method": source.get(
                    "area_method",
                    "pixel_sum_max(raw-sideband_median,0)/TIFF_exposure",
                ),
                "exposure_s": as_float(source.get("exposure_s")),
                "exposure_status": source.get(
                    "exposure_status", "verified_TIFF_ImageDescription"
                ),
                "roi_pixels": as_int(source.get("roi_pixels")),
                "sideband_pixels": as_int(source.get("sideband_pixels")),
                "fwhm_two_theta_deg": as_float(source.get("fwhm_two_theta_deg")),
                "fwhm_q_A^-1": as_float(source.get("fwhm_q_A^-1")),
                "obs_row": as_int(source.get("obs_row")),
                "track": source.get("track", ""),
                "source_state": source.get(
                    "source_state", "curated_kept_observation"
                ),
                "source_reason": source.get("source_reason", ""),
                "source_file": source_file,
            }
        )
    counts = Counter(as_int(row.get("frame")) for row in peaks)
    frame_rows = []
    for source in manifest:
        frame = as_int(source.get("frame"))
        frame_rows.append(
            {
                "frame": frame,
                "scan": source.get("scan", ""),
                "orientation": source.get("scan", ""),
                "pressure_GPa": as_float(source.get("pressure_GPa")),
                "formal_ladder_included": int(str(source.get("excluded", "0")) == "0"),
                "formal_ladder_exclusion_reason": source.get("exclusion_reason", ""),
                "peak_count": counts.get(frame, 0),
                "peak_registry_status": (
                    "curated_2D_ROIs_available"
                    if counts.get(frame, 0)
                    else "no_curated_2D_ROI_registry"
                ),
            }
        )
    track_counts = Counter(str(row.get("track", "")) for row in peaks)
    audit = {
        "source": str(uncertainty_path.resolve()),
        "source_sha256": file_sha256(uncertainty_path),
        "source_schema": source_schema,
        "source_rows": len(observations),
        "frames_in_manifest": len(frame_rows),
        "frames_with_curated_peaks": len(counts),
        "peaks": len(peaks),
        "all_source_rows_retained": len(peaks) == len(observations),
        "singleton_source_tracks_retained": sum(value == 1 for value in track_counts.values()),
        "track_used_for_detection_grouping_or_filtering": False,
        "area_is_raw_TIFF_2D_ROI_integration": True,
        "scope_caveat": (
            "This companion contains every row in the available curated ROI registry; "
            "it is not claimed to be a fresh all-28-frame 2D redetection."
        ),
    }
    return peaks, frame_rows, audit


def load_powder_all_spot_peaks(
    observations_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    observations = read_csv(observations_path)
    manifest = read_csv(manifest_path)
    included_manifest = {
        as_int(row.get("frame")): row
        for row in manifest
        if str(row.get("cover_excluded", "")).strip() == "0"
    }
    peaks: list[dict[str, Any]] = []
    for source_index, source in enumerate(observations):
        frame = as_int(source.get("frame"))
        if frame not in included_manifest:
            raise ValueError(f"powder peak row refers to excluded/unknown frame {frame}")
        q_value = as_float(source.get("q"))
        two_theta = float(q_to_two_theta(q_value, POWDER_WAVELENGTH_A).item())
        filename = str(source.get("filename", ""))
        exposure_status = (
            "D1s_filename_token"
            if "D1s" in filename
            else "unknown_D1w_token"
            if "D1w" in filename
            else "unknown"
        )
        peaks.append(
            {
                "frame": frame,
                "scan": source.get("scan", ""),
                "orientation": "not_applicable",
                "pressure_GPa": as_float(source.get("pressure_gpa")),
                "two_theta_deg": two_theta,
                "q_A^-1": q_value,
                "azim_deg": as_float(source.get("azim_deg")),
                "integrated_area": as_float(source.get("area")),
                "integrated_area_raw": as_float(source.get("area")),
                "area_unit": "background_subtracted_2D_blob_counts",
                "area_method": "source_blob_integrated_counts_above_ring_median",
                "exposure_status": exposure_status,
                "n_pixels": as_int(source.get("n_pixels")),
                "fwhm_q_A^-1": as_float(source.get("q_width")),
                "obs_row": source_index,
                "source_point": source.get("point", ""),
                "track": source.get("track", ""),
                "source_state": (
                    "untracked_short_lived"
                    if as_int(source.get("track")) == -1
                    else "tracked_source_provenance_only"
                ),
                "source_reason": "",
                "source_file": filename,
            }
        )
    counts = Counter(as_int(row.get("frame")) for row in peaks)
    frame_rows = []
    for frame, source in sorted(included_manifest.items()):
        frame_rows.append(
            {
                "frame": frame,
                "scan": source.get("scan", ""),
                "orientation": "not_applicable",
                "pressure_GPa": as_float(source.get("pressure_GPa")),
                "formal_ladder_included": 1,
                "peak_count": counts.get(frame, 0),
                "peak_registry_status": (
                    "detected_spots_available" if counts.get(frame, 0) else "zero_detected_spots"
                ),
                "source_file": source.get("filename", ""),
            }
        )
    audit = {
        "source": str(observations_path.resolve()),
        "source_rows": len(observations),
        "included_frames_in_manifest": len(frame_rows),
        "frames_with_spots": len(counts),
        "zero_spot_frames": len(frame_rows) - len(counts),
        "peaks": len(peaks),
        "all_source_rows_retained": len(peaks) == len(observations),
        "untracked_short_lived_observations_retained": sum(
            as_int(row.get("track")) == -1 for row in peaks
        ),
        "D1w_rows_retained": sum(
            str(row.get("exposure_status")) == "unknown_D1w_token" for row in peaks
        ),
        "track_used_for_detection_grouping_or_filtering": False,
        "area_is_source_2D_blob_integration": True,
        "area_exposure_normalization": (
            "not applied because D1w rows have no trustworthy exposure mapping; "
            "raw integrated counts are retained for every peak"
        ),
    }
    return peaks, frame_rows, audit


def write_matrix_csv(
    path: Path,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    matrix: np.ndarray,
    *,
    row_header: str = "row_peak",
) -> None:
    if matrix.shape != (len(row_labels), len(col_labels)):
        raise ValueError(
            f"matrix/label shape mismatch: {matrix.shape} != "
            f"({len(row_labels)}, {len(col_labels)})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([row_header, *col_labels])
        for label, values in zip(row_labels, matrix, strict=True):
            writer.writerow([label, *[csv_value(value) for value in values]])


def _sample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=int))


def build_frame_slot_grids(
    peaks_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    frame_registry: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Place every registered frame on a row and every local peak in its slot."""
    registry_by_frame: dict[int, dict[str, Any]] = {}
    for source in frame_registry:
        frame = as_int(source.get("frame"))
        if frame < 0:
            raise ValueError(f"invalid frame registry row: {source}")
        if frame in registry_by_frame:
            raise ValueError(f"duplicate registered frame: {frame}")
        registry_by_frame[frame] = dict(source)

    missing_frames = sorted(
        set(int(frame) for frame in peaks_by_frame) - set(registry_by_frame)
    )
    if missing_frames:
        raise ValueError(f"peak frames absent from frame registry: {missing_frames}")
    if not registry_by_frame:
        raise ValueError("frame registry is empty")

    frame_ids = sorted(registry_by_frame)
    max_peak_count = max(
        (len(peaks_by_frame.get(frame, ())) for frame in frame_ids),
        default=0,
    )
    if max_peak_count < 1:
        raise ValueError("cannot build peak maps for a dataset with no peaks")

    positions = np.full((len(frame_ids), max_peak_count), np.nan, dtype=float)
    areas = np.full_like(positions, np.nan)
    layout: list[dict[str, Any]] = []
    for row_index, frame in enumerate(frame_ids):
        source = registry_by_frame[frame]
        group = list(peaks_by_frame.get(frame, ()))
        for expected_index, peak in enumerate(group, start=1):
            local_index = as_int(peak.get("local_peak_index"))
            if local_index != expected_index:
                raise ValueError(
                    f"frame {frame} local peak slots are not contiguous: "
                    f"expected {expected_index}, got {local_index}"
                )
            positions[row_index, expected_index - 1] = as_float(
                peak.get("two_theta_deg")
            )
            areas[row_index, expected_index - 1] = as_float(
                peak.get("integrated_area")
            )
        layout.append(
            {
                "row_index_0based": row_index,
                "frame": frame,
                "scan": source.get("scan", ""),
                "pressure_GPa": as_float(source.get("pressure_GPa")),
                "orientation": source.get("orientation", ""),
                "peak_count": len(group),
                "max_local_peak_slots": max_peak_count,
                "zero_peak_frame": int(not group),
                "first_peak_id": group[0].get("peak_id", "") if group else "",
                "last_peak_id": group[-1].get("peak_id", "") if group else "",
            }
        )
    return layout, positions, areas


def build_anchor_peak_frame_slot_matrices(
    anchor_peak: Mapping[str, Any],
    frame_layout: Sequence[Mapping[str, Any]],
    target_two_theta: np.ndarray,
    target_areas: np.ndarray,
    tolerance: float = POSITION_TOLERANCE_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare one anchor peak with every local peak slot in every other frame."""
    positions = np.asarray(target_two_theta, dtype=float)
    areas = np.asarray(target_areas, dtype=float)
    expected_shape = (len(frame_layout), positions.shape[1])
    if positions.ndim != 2 or areas.shape != positions.shape:
        raise ValueError("target position and area grids must be equal-shape 2D arrays")
    if positions.shape != expected_shape:
        raise ValueError(
            f"frame layout/grid mismatch: {positions.shape} != {expected_shape}"
        )

    location = location_similarity_matrix(
        [as_float(anchor_peak.get("two_theta_deg"))],
        positions.reshape(-1),
        tolerance,
    ).reshape(positions.shape)
    area = area_similarity_matrix(
        [as_float(anchor_peak.get("integrated_area"))],
        areas.reshape(-1),
    ).reshape(areas.shape)

    anchor_frame = as_int(anchor_peak.get("frame"))
    anchor_rows = [
        index
        for index, row in enumerate(frame_layout)
        if as_int(row.get("frame")) == anchor_frame
    ]
    if len(anchor_rows) != 1:
        raise ValueError(
            f"anchor frame {anchor_frame} occurs {len(anchor_rows)} times in layout"
        )
    location[anchor_rows[0], :] = np.nan
    area[anchor_rows[0], :] = np.nan
    return location, area


def plot_anchor_peak_frame_slot_heatmap(
    path: Path,
    matrix: np.ndarray,
    frame_layout: Sequence[Mapping[str, Any]],
    title: str,
    metric_label: str,
    *,
    anchor_frame: int,
) -> None:
    if not matrix.size:
        return
    rows, columns = matrix.shape
    if rows != len(frame_layout):
        raise ValueError("plot matrix row count does not match frame layout")
    width = min(14.0, max(7.5, 3.8 + columns * 0.38))
    height = min(18.0, max(6.0, 3.5 + rows * (0.16 if rows <= 60 else 0.013)))
    fig, ax = plt.subplots(figsize=(width, height))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title, fontsize=11, wrap=True)
    ax.set_ylabel("target frame")
    ax.set_xlabel("local peak number within target frame")

    anchor_row = next(
        index
        for index, row in enumerate(frame_layout)
        if as_int(row.get("frame")) == anchor_frame
    )
    sampled_y = _sample_indices(rows, 38)
    if rows > 60:
        minimum_gap = max(6, rows // 100)
        sampled_y = sampled_y[np.abs(sampled_y - anchor_row) >= minimum_gap]
    y_indices = np.unique(
        np.concatenate((sampled_y, np.asarray([anchor_row], dtype=int)))
    )
    ax.set_yticks(y_indices)
    ax.set_yticklabels(
        [
            (
                f"f{as_int(frame_layout[index].get('frame'))}"
                f"{' [anchor]' if index == anchor_row else ''}\n"
                f"{as_float(frame_layout[index].get('pressure_GPa')):g} GPa"
            )
            for index in y_indices
        ],
        fontsize=6,
    )
    x_indices = np.arange(columns, dtype=int)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(
        [f"peak {index + 1}" for index in x_indices],
        rotation=60 if columns > 8 else 0,
        ha="right" if columns > 8 else "center",
        fontsize=6 if columns > 20 else 8,
    )
    ax.axhline(anchor_row - 0.5, color="#777777", lw=0.6)
    ax.axhline(anchor_row + 0.5, color="#777777", lw=0.6)
    fig.colorbar(
        image,
        ax=ax,
        fraction=0.06,
        pad=0.04,
        shrink=0.72,
        aspect=36,
        label=metric_label,
    )
    fig.text(
        0.5,
        0.008,
        (
            "White = no peak in that local slot, a zero-peak frame, or the "
            "excluded anchor frame; dark purple = a real similarity near 0."
        )
        + " No track/group/angle prefilter.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_global_overview(
    path: Path,
    matrix: np.ndarray,
    peaks: Sequence[Mapping[str, Any]],
    frame_blocks: Sequence[Mapping[str, Any]],
    title: str,
    metric_label: str,
) -> None:
    if not matrix.size:
        return
    fig, ax = plt.subplots(figsize=(13.0, 11.0))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    block_indices = _sample_indices(len(frame_blocks), 22)
    positions: list[float] = []
    labels: list[str] = []
    for block_index in block_indices:
        block = frame_blocks[int(block_index)]
        positions.append(
            0.5
            * (
                as_int(block.get("global_start"))
                + as_int(block.get("global_stop"))
            )
        )
        labels.append(
            f"f{as_int(block.get('frame'))}\n"
            f"{as_float(block.get('pressure_GPa')):g}GPa"
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=6)
    if len(frame_blocks) <= 100:
        for block in frame_blocks[:-1]:
            boundary = as_int(block.get("global_stop")) + 0.5
            ax.axvline(boundary, color="white", lw=0.25)
            ax.axhline(boundary, color="white", lw=0.25)
    ax.set_title(title)
    ax.set_xlabel("frame-grouped peaks")
    ax.set_ylabel("frame-grouped peaks")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label=metric_label)
    fig.text(
        0.5,
        0.01,
        "White diagonal blocks are same-frame comparisons, intentionally excluded.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _frame_metadata(row: Mapping[str, Any]) -> tuple[str, float, str]:
    return (
        str(row.get("scan", "")),
        as_float(row.get("pressure_GPa")),
        str(row.get("orientation", "")),
    )


def generate_peak_dataset(
    root: Path,
    dataset: str,
    raw_peaks: Sequence[Mapping[str, Any]],
    frame_registry: Sequence[Mapping[str, Any]],
    source_audit: Mapping[str, Any],
    tolerance: float,
    make_plots: bool,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    peaks = assign_local_peak_ids(raw_peaks, dataset)
    peaks_by_frame = group_peaks_by_frame(peaks)
    frame_registry_rows = [dict(row) for row in frame_registry]
    write_csv(root / "frame_registry.csv", frame_registry_rows)
    peak_fields: list[str] = []
    for row in peaks:
        for key in row:
            if key not in peak_fields:
                peak_fields.append(key)
    write_csv(root / "peak_registry.csv", peaks, peak_fields)

    pair_rows = build_cross_frame_pair_rows(peaks_by_frame, tolerance)
    write_csv_gz(root / "all_cross_frame_peak_pairs.csv.gz", pair_rows, PAIR_FIELDS)

    pair_index_rows: list[dict[str, Any]] = []
    offset = 0
    for frame_a, frame_b in combinations(sorted(peaks_by_frame), 2):
        left = peaks_by_frame[frame_a]
        right = peaks_by_frame[frame_b]
        cells = len(left) * len(right)
        scan_a, pressure_a, orientation_a = _frame_metadata(left[0])
        scan_b, pressure_b, orientation_b = _frame_metadata(right[0])
        pair_index_rows.append(
            {
                "frame_a": frame_a,
                "frame_b": frame_b,
                "scan_a": scan_a,
                "scan_b": scan_b,
                "pressure_a_GPa": pressure_a,
                "pressure_b_GPa": pressure_b,
                "orientation_a": orientation_a,
                "orientation_b": orientation_b,
                "rows_Na": len(left),
                "columns_Nb": len(right),
                "cells_Na_times_Nb": cells,
                "canonical_pair_table_row_start_0based": offset,
                "canonical_pair_table_row_stop_exclusive_0based": offset + cells,
            }
        )
        offset += cells
    write_csv(root / "frame_pair_index.csv", pair_index_rows)

    ordered_peaks = sorted(
        peaks,
        key=lambda row: (as_int(row.get("frame")), as_int(row.get("local_peak_index"))),
    )
    global_index = {str(row["peak_id"]): index for index, row in enumerate(ordered_peaks)}
    total = len(ordered_peaks)
    global_location = np.full((total, total), np.nan, dtype=float)
    global_area = np.full((total, total), np.nan, dtype=float)
    for row in pair_rows:
        left_index = global_index[str(row["peak_a_id"])]
        right_index = global_index[str(row["peak_b_id"])]
        global_location[left_index, right_index] = as_float(row.get("location_similarity"))
        global_location[right_index, left_index] = as_float(row.get("location_similarity"))
        global_area[left_index, right_index] = as_float(row.get("area_similarity"))
        global_area[right_index, left_index] = as_float(row.get("area_similarity"))
    np.savez_compressed(
        root / "all_peak_similarity_matrices.npz",
        peak_ids=np.asarray([str(row["peak_id"]) for row in ordered_peaks]),
        frames=np.asarray([as_int(row.get("frame")) for row in ordered_peaks], dtype=int),
        location=global_location,
        area=global_area,
    )

    frame_blocks: list[dict[str, Any]] = []
    cursor = 0
    for frame in sorted(peaks_by_frame):
        group = peaks_by_frame[frame]
        frame_blocks.append(
            {
                "frame": frame,
                "scan": group[0].get("scan", ""),
                "pressure_GPa": as_float(group[0].get("pressure_GPa")),
                "orientation": group[0].get("orientation", ""),
                "peak_count": len(group),
                "global_start": cursor,
                "global_stop": cursor + len(group) - 1,
            }
        )
        cursor += len(group)
    write_csv(root / "global_frame_blocks.csv", frame_blocks)

    frame_layout, slot_positions, slot_areas = build_frame_slot_grids(
        peaks_by_frame,
        frame_registry_rows,
    )
    write_csv(root / "frame_slot_layout.csv", frame_layout)
    max_peak_slots = slot_positions.shape[1]
    local_peak_labels = [f"peak {index}" for index in range(1, max_peak_slots + 1)]
    np.savez_compressed(
        root / "frame_slot_target_values.npz",
        frame_ids=np.asarray([as_int(row.get("frame")) for row in frame_layout]),
        peak_counts=np.asarray(
            [as_int(row.get("peak_count")) for row in frame_layout], dtype=int
        ),
        local_peak_labels=np.asarray(local_peak_labels),
        two_theta_deg=slot_positions,
        integrated_area=slot_areas,
    )

    frame_row_lookup = {
        as_int(row.get("frame")): as_int(row.get("row_index_0based"))
        for row in frame_layout
    }
    base_valid_mask = np.isfinite(slot_positions) & np.isfinite(slot_areas)
    expected_base_valid = np.zeros_like(base_valid_mask, dtype=bool)
    for row_index, row in enumerate(frame_layout):
        expected_base_valid[row_index, : as_int(row.get("peak_count"))] = True
    slot_layout_verified = bool(np.array_equal(base_valid_mask, expected_base_valid))

    anchor_index_rows: list[dict[str, Any]] = []
    all_anchor_masks_verified = True
    all_anchor_shapes_verified = True
    all_anchor_same_frame_rows_blank = True
    total_location_finite = 0
    total_area_finite = 0
    for anchor_peak in ordered_peaks:
        anchor_frame = as_int(anchor_peak.get("frame"))
        anchor_slot = as_int(anchor_peak.get("local_peak_index"))
        anchor_frame_row = frame_row_lookup[anchor_frame]
        location, area = build_anchor_peak_frame_slot_matrices(
            anchor_peak,
            frame_layout,
            slot_positions,
            slot_areas,
            tolerance,
        )
        expected_mask = expected_base_valid.copy()
        expected_mask[anchor_frame_row, :] = False
        all_anchor_masks_verified = bool(
            all_anchor_masks_verified
            and np.array_equal(np.isfinite(location), expected_mask)
            and np.array_equal(np.isfinite(area), expected_mask)
        )
        all_anchor_shapes_verified = bool(
            all_anchor_shapes_verified
            and location.shape == slot_positions.shape
            and area.shape == slot_positions.shape
        )
        all_anchor_same_frame_rows_blank = bool(
            all_anchor_same_frame_rows_blank
            and np.all(np.isnan(location[anchor_frame_row, :]))
            and np.all(np.isnan(area[anchor_frame_row, :]))
        )
        location_finite_count = int(np.count_nonzero(np.isfinite(location)))
        area_finite_count = int(np.count_nonzero(np.isfinite(area)))
        total_location_finite += location_finite_count
        total_area_finite += area_finite_count

        stem = f"anchor_f{anchor_frame:04d}_s{anchor_slot:02d}"
        frame_directory = f"frame_{anchor_frame:04d}"
        location_csv = (
            root
            / "per_anchor_peak_matrices"
            / "location"
            / frame_directory
            / f"{stem}.csv"
        )
        area_csv = (
            root
            / "per_anchor_peak_matrices"
            / "area"
            / frame_directory
            / f"{stem}.csv"
        )
        row_labels = [f"frame {as_int(row.get('frame'))}" for row in frame_layout]
        write_matrix_csv(
            location_csv,
            row_labels,
            local_peak_labels,
            location,
            row_header="target_frame",
        )
        write_matrix_csv(
            area_csv,
            row_labels,
            local_peak_labels,
            area,
            row_header="target_frame",
        )

        location_png = (
            root
            / "per_anchor_peak_heatmaps"
            / "location"
            / frame_directory
            / f"{stem}.png"
        )
        area_png = (
            root
            / "per_anchor_peak_heatmaps"
            / "area"
            / frame_directory
            / f"{stem}.png"
        )
        if make_plots:
            pressure = as_float(anchor_peak.get("pressure_GPa"))
            two_theta = as_float(anchor_peak.get("two_theta_deg"))
            descriptor = (
                f"{dataset}: anchor {anchor_peak['peak_id']} | "
                f"frame {anchor_frame}, peak {anchor_slot} | "
                f"{pressure:g} GPa, 2theta={two_theta:.4f} deg | "
                f"{len(frame_layout)} frames x {max_peak_slots} local slots"
            )
            plot_anchor_peak_frame_slot_heatmap(
                location_png,
                location,
                frame_layout,
                descriptor + " — location",
                "location similarity",
                anchor_frame=anchor_frame,
            )
            plot_anchor_peak_frame_slot_heatmap(
                area_png,
                area,
                frame_layout,
                descriptor + " — ROI / integrated area",
                "ROI / integrated-area similarity",
                anchor_frame=anchor_frame,
            )
        anchor_index_rows.append(
            {
                "dataset": dataset,
                "anchor_peak_id": anchor_peak.get("peak_id", ""),
                "anchor_frame": anchor_frame,
                "anchor_local_peak": anchor_slot,
                "anchor_frame_row_index_0based": anchor_frame_row,
                "anchor_scan": anchor_peak.get("scan", ""),
                "anchor_pressure_GPa": as_float(anchor_peak.get("pressure_GPa")),
                "anchor_orientation": anchor_peak.get("orientation", ""),
                "anchor_two_theta_deg": as_float(anchor_peak.get("two_theta_deg")),
                "anchor_integrated_area": as_float(
                    anchor_peak.get("integrated_area")
                ),
                "anchor_frame_peak_count": len(peaks_by_frame[anchor_frame]),
                "matrix_registered_frame_rows": len(frame_layout),
                "matrix_local_peak_slot_columns": max_peak_slots,
                "expected_cross_frame_peak_comparisons": (
                    len(peaks) - len(peaks_by_frame[anchor_frame])
                ),
                "location_finite_cells": location_finite_count,
                "area_finite_cells": area_finite_count,
                "structural_blank_cells": int(location.size - location_finite_count),
                "same_frame_row_blank": int(
                    np.all(np.isnan(location[anchor_frame_row, :]))
                    and np.all(np.isnan(area[anchor_frame_row, :]))
                ),
                "location_csv": str(location_csv.relative_to(root)),
                "area_csv": str(area_csv.relative_to(root)),
                "location_png": (
                    str(location_png.relative_to(root)) if make_plots else ""
                ),
                "area_png": str(area_png.relative_to(root)) if make_plots else "",
            }
        )
    write_csv(root / "per_anchor_peak_map_index.csv", anchor_index_rows)

    frame_counts = {frame: len(rows) for frame, rows in peaks_by_frame.items()}
    expected_cells = sum(
        frame_counts[left] * frame_counts[right]
        for left, right in combinations(sorted(frame_counts), 2)
    )
    finite_location = np.asarray(
        [as_float(row.get("location_similarity")) for row in pair_rows], dtype=float
    )
    finite_area = np.asarray(
        [as_float(row.get("area_similarity")) for row in pair_rows], dtype=float
    )
    integrated_areas = np.asarray(
        [as_float(row.get("integrated_area")) for row in peaks], dtype=float
    )
    expected_ordered_anchor_cells = 2 * expected_cells
    zero_peak_frames = sum(
        as_int(row.get("peak_count")) == 0 for row in frame_layout
    )
    validation = {
        "dataset": dataset,
        "source_audit": dict(source_audit),
        "registered_frames": len(frame_registry_rows),
        "nonempty_frames": len(peaks_by_frame),
        "zero_peak_frames": zero_peak_frames,
        "peaks": len(peaks),
        "max_local_peak_slots": max_peak_slots,
        "per_anchor_peak_map_shape": [len(frame_layout), max_peak_slots],
        "per_anchor_peak_maps": len(anchor_index_rows),
        "per_anchor_peak_matrix_csv_files": 2 * len(anchor_index_rows),
        "per_anchor_peak_heatmap_png_files": (
            2 * len(anchor_index_rows) if make_plots else 0
        ),
        "canonical_nonempty_frame_pairs": len(pair_index_rows),
        "expected_cross_frame_peak_cells_sum_NiNj": expected_cells,
        "written_cross_frame_peak_rows": len(pair_rows),
        "all_cartesian_cells_written": expected_cells == len(pair_rows),
        "expected_ordered_anchor_finite_cells": expected_ordered_anchor_cells,
        "location_ordered_anchor_finite_cells": total_location_finite,
        "area_ordered_anchor_finite_cells": total_area_finite,
        "every_registered_frame_is_a_map_row": len(frame_layout)
        == len(frame_registry_rows),
        "frame_slot_layout_verified": slot_layout_verified,
        "all_anchor_peak_shapes_verified": all_anchor_shapes_verified,
        "all_anchor_peak_masks_verified": all_anchor_masks_verified,
        "all_anchor_same_frame_rows_blank": all_anchor_same_frame_rows_blank,
        "all_anchor_peak_finite_counts_complete": (
            total_location_finite == expected_ordered_anchor_cells
            and total_area_finite == expected_ordered_anchor_cells
        ),
        "local_peak_columns_only": local_peak_labels,
        "track_used_for_selection_grouping_or_scoring": False,
        "same_frame_cells_excluded": True,
        "finite_nonnegative_integrated_area_count": int(
            np.count_nonzero(
                np.isfinite(integrated_areas) & (integrated_areas >= 0.0)
            )
        ),
        "every_peak_has_finite_nonnegative_integrated_area": bool(
            integrated_areas.size == len(peaks)
            and np.all(np.isfinite(integrated_areas))
            and np.all(integrated_areas >= 0.0)
        ),
        "all_location_scores_finite": bool(np.all(np.isfinite(finite_location))),
        "all_area_scores_finite": bool(np.all(np.isfinite(finite_area))),
        "finite_location_scores_in_0_1": bool(
            np.all(
                (finite_location[np.isfinite(finite_location)] >= 0.0)
                & (finite_location[np.isfinite(finite_location)] <= 1.0)
            )
        ),
        "finite_area_scores_in_0_1": bool(
            np.all(
                (finite_area[np.isfinite(finite_area)] >= 0.0)
                & (finite_area[np.isfinite(finite_area)] <= 1.0)
            )
        ),
        "rectangular_anchor_shapes_verified": all_anchor_shapes_verified,
        "position_tolerance_deg": tolerance,
        "location_formula": "radial 2theta: clip(1-abs(delta_2theta)/tolerance,0,1)",
        "area_formula": (
            "min(integrated_area_a,integrated_area_b)/max(...); both zero -> 1"
        ),
    }
    if not all(
        [
            validation["all_cartesian_cells_written"],
            validation["every_peak_has_finite_nonnegative_integrated_area"],
            validation["all_location_scores_finite"],
            validation["all_area_scores_finite"],
            validation["finite_location_scores_in_0_1"],
            validation["finite_area_scores_in_0_1"],
            validation["rectangular_anchor_shapes_verified"],
            validation["every_registered_frame_is_a_map_row"],
            validation["frame_slot_layout_verified"],
            validation["all_anchor_peak_masks_verified"],
            validation["all_anchor_same_frame_rows_blank"],
            validation["all_anchor_peak_finite_counts_complete"],
        ]
    ):
        raise RuntimeError(f"validation failed for {dataset}: {validation}")
    (root / "validation_report.json").write_text(
        json.dumps(json_ready(validation), indent=2),
        encoding="utf-8",
    )
    return validation


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != ".DS_Store"
    }


def _window_payload_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    paths: list[Path] = []
    for relative in WINDOW_SUITE_PAYLOADS:
        source = root / relative
        if not source.exists():
            raise FileNotFoundError(f"frozen window payload is missing: {source}")
        if source.is_symlink():
            raise ValueError(f"window payload may not be a symbolic link: {source}")
        if source.is_dir():
            for path in source.rglob("*"):
                if path.is_symlink():
                    raise ValueError(
                        f"window payload may not contain symbolic links: {path}"
                    )
                if path.is_file() and path.name != ".DS_Store":
                    paths.append(path)
        elif source.is_file():
            paths.append(source)
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
        relative = str(path.relative_to(root))
        size = path.stat().st_size
        sha256 = file_sha256(path)
        rows.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": sha256,
            }
        )
        digest.update(f"{relative}\0{size}\0{sha256}\n".encode("utf-8"))
    return rows, digest.hexdigest()


def copy_unchanged_window_suite(
    output_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Copy the completed v3 window deliverable byte-for-byte into a new suite."""
    source_root = source_root.resolve()
    marker_path = source_root / "RUN_COMPLETE.json"
    validation_path = source_root / "validation_report.json"
    if not marker_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            f"v3 window suite is missing completion metadata: {source_root}"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "complete"
        or marker.get("all_validation_checks_passed") is not True
        or validation.get("status") != "PASS"
    ):
        raise ValueError(f"v3 window suite is not complete/PASS: {source_root}")
    expected_validation_hash = marker.get("validation_report_sha256")
    if expected_validation_hash != file_sha256(validation_path):
        raise ValueError("v3 validation report hash does not match completion marker")

    source_rows, source_digest = _window_payload_inventory(source_root)
    source_bytes = sum(as_int(row.get("bytes"), 0) for row in source_rows)
    if (
        len(source_rows) != EXPECTED_V3_WINDOW_PAYLOAD_FILES
        or source_bytes != EXPECTED_V3_WINDOW_PAYLOAD_BYTES
        or source_digest != EXPECTED_V3_WINDOW_PAYLOAD_DIGEST
    ):
        raise ValueError(
            "v3 window payload no longer matches the frozen audited inventory: "
            f"files={len(source_rows)}, bytes={source_bytes}, digest={source_digest}"
        )

    for relative in WINDOW_SUITE_PAYLOADS:
        source = source_root / relative
        destination = output_root / relative
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite window payload: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, copy_function=shutil.copy2)
        else:
            shutil.copy2(source, destination)

    destination_rows, destination_digest = _window_payload_inventory(output_root)
    source_by_path = {
        str(row["relative_path"]): (as_int(row["bytes"]), str(row["sha256"]))
        for row in source_rows
    }
    destination_by_path = {
        str(row["relative_path"]): (as_int(row["bytes"]), str(row["sha256"]))
        for row in destination_rows
    }
    payloads_match = (
        source_by_path == destination_by_path
        and source_digest == destination_digest
    )
    if not payloads_match:
        raise RuntimeError("copied v3 window payload differs from its source")
    write_csv(
        output_root / "window_unchanged_v3_hashes.csv",
        source_rows,
        ["relative_path", "bytes", "sha256"],
    )

    source_window_audit = validation.get("window_results")
    if not isinstance(source_window_audit, Mapping):
        raise ValueError("v3 validation report has no window_results object")
    audit = json.loads(json.dumps(source_window_audit))
    for item in audit.get("copied", []):
        old_destination = Path(str(item.get("destination", "")))
        try:
            relative_destination = old_destination.relative_to(source_root)
        except ValueError:
            continue
        item["destination"] = str((output_root / relative_destination).resolve())
    audit.update(
        {
            "window_calculations_regenerated": False,
            "unchanged_from_v3_suite": True,
            "v3_suite_source": str(source_root),
            "v3_payload_files": len(source_rows),
            "v3_payload_bytes": source_bytes,
            "v3_payload_digest": source_digest,
            "copied_payload_files": len(destination_rows),
            "copied_payload_bytes": sum(
                as_int(row.get("bytes"), 0) for row in destination_rows
            ),
            "copied_payload_digest": destination_digest,
            "all_v3_payload_paths_sizes_and_sha256_match": payloads_match,
            "hash_manifest": "window_unchanged_v3_hashes.csv",
        }
    )
    return audit


def _validate_completed_window_source(source_root: Path) -> dict[str, Any]:
    marker_path = source_root / "RUN_COMPLETE.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"window source completion marker is missing: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("all_validation_checks_passed") is not True:
        raise ValueError(f"window source is not a validated complete run: {marker_path}")
    hash_targets = {
        "algorithm_config_sha256": source_root / "algorithm_config.json",
        "run_manifest_sha256": source_root / "run_manifest.json",
        "validation_report_sha256": source_root / "validation" / "validation_report.json",
    }
    verified_hashes: dict[str, str] = {}
    for marker_key, target in hash_targets.items():
        if not target.is_file():
            raise FileNotFoundError(f"window provenance file is missing: {target}")
        actual = file_sha256(target)
        if actual != marker.get(marker_key):
            raise ValueError(f"window provenance hash mismatch for {target}")
        verified_hashes[str(target.relative_to(source_root))] = actual
    return {
        "source_root": str(source_root.resolve()),
        "profile": marker.get("profile", ""),
        "check_count": as_int(marker.get("check_count")),
        "completion_marker_verified": True,
        "verified_provenance_sha256": verified_hashes,
    }


def _validate_window_role(
    root: Path,
    *,
    kind: str,
    expected_files: int,
    expected_csv: int,
    expected_png: int,
    expected_frames: int,
    expected_scans: int,
    expected_pressures: int,
) -> dict[str, Any]:
    inventory = _file_inventory(root)
    suffix_counts = Counter(Path(relative).suffix.lower() for relative in inventory)
    if len(inventory) != expected_files:
        raise ValueError(
            f"window role file count mismatch for {root}: "
            f"{len(inventory)} != {expected_files}"
        )
    expected_suffixes = {
        ".csv": expected_csv,
        ".png": expected_png,
        ".npz": 1,
    }
    for suffix, expected in expected_suffixes.items():
        if suffix_counts.get(suffix, 0) != expected:
            raise ValueError(
                f"window role {suffix} count mismatch for {root}: "
                f"{suffix_counts.get(suffix, 0)} != {expected}"
            )

    if kind == "across":
        required_dirs = ("acf_strict", "direct_strict", "shift_tolerant_secondary")
        for dirname in required_dirs:
            if not (root / dirname).is_dir():
                raise FileNotFoundError(f"missing across-frame family: {root / dirname}")
        archive_path = root / "across_frame_matrices.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            expected_shapes = {
                "scan_names": (expected_scans,),
                "pressure_gpa": (expected_pressures,),
                "window_starts_deg": (26,),
                "acf_strict_by_scan": (
                    expected_scans,
                    26,
                    expected_pressures,
                    expected_pressures,
                ),
                "direct_strict_by_scan": (
                    expected_scans,
                    26,
                    expected_pressures,
                    expected_pressures,
                ),
                "shift_tolerant_secondary_by_scan": (
                    expected_scans,
                    26,
                    expected_pressures,
                    expected_pressures,
                ),
            }
            for key, expected_shape in expected_shapes.items():
                if key not in archive or archive[key].shape != expected_shape:
                    actual = archive[key].shape if key in archive else None
                    raise ValueError(
                        f"window NPZ shape mismatch {archive_path}:{key}: "
                        f"{actual} != {expected_shape}"
                    )
        coverage = {
            "scans": expected_scans,
            "pressures": expected_pressures,
            "windows": 26,
        }
    elif kind == "within":
        required_dirs = (
            "all_windows",
            "nonoverlap_control",
            "by_pressure",
            "per_frame_matrices",
        )
        for dirname in required_dirs:
            if not (root / dirname).is_dir():
                raise FileNotFoundError(f"missing within-frame family: {root / dirname}")
        archive_path = root / "within_frame_matrices.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            expected_shapes = {
                "frame_ids": (expected_frames,),
                "pressure_gpa": (expected_pressures,),
                "scan_names": (expected_scans,),
                "window_starts_deg": (26,),
                "matrices_by_frame": (expected_frames, 26, 26),
            }
            for key, expected_shape in expected_shapes.items():
                if key not in archive or archive[key].shape != expected_shape:
                    actual = archive[key].shape if key in archive else None
                    raise ValueError(
                        f"window NPZ shape mismatch {archive_path}:{key}: "
                        f"{actual} != {expected_shape}"
                    )
        per_frame_csv = list((root / "per_frame_matrices").glob("*.csv"))
        if len(per_frame_csv) != expected_frames:
            raise ValueError(
                f"per-frame window matrix count mismatch for {root}: "
                f"{len(per_frame_csv)} != {expected_frames}"
            )
        coverage = {
            "frames": expected_frames,
            "scans": expected_scans,
            "pressures": expected_pressures,
            "windows_per_axis": 26,
            "per_frame_numeric_matrices": expected_frames,
        }
    else:
        raise ValueError(f"unknown window role kind: {kind}")
    return {
        "files": len(inventory),
        "csv_files": suffix_counts.get(".csv", 0),
        "png_files": suffix_counts.get(".png", 0),
        "npz_files": suffix_counts.get(".npz", 0),
        "coverage": coverage,
        "sha256_by_relative_path": inventory,
    }


def copy_window_results(
    output_root: Path,
    single_source: Path,
    powder_source: Path,
) -> dict[str, Any]:
    mappings = [
        {
            "role": "single_spots_across",
            "kind": "across",
            "source": single_source / "spots" / "across_frames",
            "destination": (
                output_root
                / "window_full_symmetric_audit"
                / "single_crystal"
                / "spots"
                / "across_frames"
            ),
            "expected_files": 473,
            "expected_csv": 316,
            "expected_png": 156,
            "expected_frames": 22,
            "expected_scans": 2,
            "expected_pressures": 11,
        },
        {
            "role": "single_spots_within",
            "kind": "within",
            "source": single_source / "spots" / "within_frame",
            "destination": (
                output_root
                / "window_full_symmetric_audit"
                / "single_crystal"
                / "spots"
                / "within_frame"
            ),
            "expected_files": 152,
            "expected_csv": 99,
            "expected_png": 52,
            "expected_frames": 22,
            "expected_scans": 2,
            "expected_pressures": 11,
        },
        {
            "role": "powder_spots_across",
            "kind": "across",
            "source": powder_source / "spots" / "across_frames",
            "destination": (
                output_root
                / "window_full_symmetric_audit"
                / "powder"
                / "spots"
                / "across_frames"
            ),
            "expected_files": 473,
            "expected_csv": 316,
            "expected_png": 156,
            "expected_frames": 1060,
            "expected_scans": 56,
            "expected_pressures": 19,
        },
        {
            "role": "powder_spots_within",
            "kind": "within",
            "source": powder_source / "spots" / "within_frame",
            "destination": (
                output_root
                / "window_full_symmetric_audit"
                / "powder"
                / "spots"
                / "within_frame"
            ),
            "expected_files": 1270,
            "expected_csv": 1185,
            "expected_png": 84,
            "expected_frames": 1060,
            "expected_scans": 56,
            "expected_pressures": 19,
        },
        {
            "role": "powder_fit_control_across",
            "kind": "across",
            "source": powder_source / "fit" / "across_frames",
            "destination": (
                output_root
                / "window_full_symmetric_audit"
                / "powder"
                / "fit_control"
                / "across_frames"
            ),
            "expected_files": 473,
            "expected_csv": 316,
            "expected_png": 156,
            "expected_frames": 1060,
            "expected_scans": 56,
            "expected_pressures": 19,
        },
        {
            "role": "powder_fit_control_within",
            "kind": "within",
            "source": powder_source / "fit" / "within_frame",
            "destination": (
                output_root
                / "window_full_symmetric_audit"
                / "powder"
                / "fit_control"
                / "within_frame"
            ),
            "expected_files": 1270,
            "expected_csv": 1185,
            "expected_png": 84,
            "expected_frames": 1060,
            "expected_scans": 56,
            "expected_pressures": 19,
        },
    ]
    source_runs = {
        "single_crystal": _validate_completed_window_source(single_source),
        "powder": _validate_completed_window_source(powder_source),
    }
    copied: list[dict[str, Any]] = []
    for mapping in mappings:
        source = Path(mapping["source"])
        destination = Path(mapping["destination"])
        if not source.is_dir():
            raise FileNotFoundError(f"completed window source is missing: {source}")
        source_audit = _validate_window_role(
            source,
            kind=str(mapping["kind"]),
            expected_files=as_int(mapping["expected_files"]),
            expected_csv=as_int(mapping["expected_csv"]),
            expected_png=as_int(mapping["expected_png"]),
            expected_frames=as_int(mapping["expected_frames"]),
            expected_scans=as_int(mapping["expected_scans"]),
            expected_pressures=as_int(mapping["expected_pressures"]),
        )
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(".DS_Store"),
        )
        destination_audit = _validate_window_role(
            destination,
            kind=str(mapping["kind"]),
            expected_files=as_int(mapping["expected_files"]),
            expected_csv=as_int(mapping["expected_csv"]),
            expected_png=as_int(mapping["expected_png"]),
            expected_frames=as_int(mapping["expected_frames"]),
            expected_scans=as_int(mapping["expected_scans"]),
            expected_pressures=as_int(mapping["expected_pressures"]),
        )
        hash_match = (
            source_audit["sha256_by_relative_path"]
            == destination_audit["sha256_by_relative_path"]
        )
        if not hash_match:
            raise ValueError(f"window copy hash mismatch: {source} -> {destination}")
        copied.append(
            {
                "role": mapping["role"],
                "kind": mapping["kind"],
                "source": str(source.resolve()),
                "destination": str(destination.resolve()),
                "files": destination_audit["files"],
                "png_files": destination_audit["png_files"],
                "csv_files": destination_audit["csv_files"],
                "npz_files": destination_audit["npz_files"],
                "coverage": destination_audit["coverage"],
                "source_destination_sha256_match": hash_match,
            }
        )
    provenance_root = output_root / "window_provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    for label, source in (
        ("single_crystal", single_source),
        ("powder", powder_source),
    ):
        for filename in (
            "RUN_COMPLETE.json",
            "algorithm_config.json",
            "run_manifest.json",
            "input_inventory.csv",
            "artifact_index.csv",
        ):
            source_file = source / filename
            if source_file.is_file():
                shutil.copy2(source_file, provenance_root / f"{label}_{filename}")
        validation_source = source / "validation" / "validation_report.json"
        shutil.copy2(
            validation_source,
            provenance_root / f"{label}_validation_report.json",
        )
    window_readme = """# Window correlations

The primary `single_crystal/windows/` and `powder/windows/` results do not use
peak tracks.  Every user-facing square correlation map contains only the strict
lower triangle: the diagonal is omitted because valid self-correlation is one,
and the mirrored upper triangle is omitted as redundant.

## Across frames

Twenty-six angle windows are compared across pressure frames within each scan.
`acf_strict` is the primary same-window ACF-fingerprint Pearson score;
`direct_strict` is a direct residual-signal validation; and
`shift_tolerant_secondary` permits the same window or one neighboring window.
Every family has lower-triangle matrices, PNGs, and a canonical unique-pair
table carrying support and confidence intervals.

## Within one frame

Each included frame has 325 unique off-diagonal window pairs from the original
26 x 26 window-to-window ACF matrix.  Aggregate and pressure-specific
lower-triangle maps are plotted; every per-frame unique pair is retained in
compressed CSV and NPZ form.

The single-crystal window suite uses the official 22-frame compression ladder
(two orientations x eleven pressures, 1.0--12.8 GPa).  The six deliberately
excluded frames (isolated 5-degree orientation, decompression-only frames, and
repeat exposures) remain in the new 28-frame peak maps but are not mixed into
the official pressure-ladder window aggregates.  Powder windows use all 1060
accepted frames across 3.5--50.7 GPa.

Powder `spots` is the sample channel.  Powder `fit_control` is the
tungsten-dominated control and must not be interpreted as sample-only evidence.

The byte-identical full symmetric source matrices and legacy full-square PNGs
are retained only in `window_full_symmetric_audit/` for reproducibility.  They
are not the primary presentation results.
"""
    (output_root / "WINDOW_METHODS.md").write_text(window_readme, encoding="utf-8")
    return {
        "source_runs": source_runs,
        "copied": copied,
        "roles_verified": sorted(str(item["role"]) for item in copied),
        "total_files": sum(as_int(item["files"]) for item in copied),
        "total_csv_files": sum(as_int(item["csv_files"]) for item in copied),
        "total_png_files": sum(as_int(item["png_files"]) for item in copied),
        "total_npz_files": sum(as_int(item["npz_files"]) for item in copied),
        "all_source_destination_hashes_match": all(
            bool(item["source_destination_sha256_match"]) for item in copied
        ),
        "single_window_scope": {
            "official_frames": 22,
            "all_available_peak_map_frames": 28,
            "excluded_from_official_window_ladder_frame_ids": [1, 11, 12, 24, 25, 27],
        },
        "powder_window_scope": {
            "accepted_frames": 1060,
            "pressure_min_GPa": 3.5,
            "pressure_max_GPa": 50.7,
        },
    }


def _plot_window_quicklook(
    path: Path,
    matrix: np.ndarray,
    labels: Sequence[str],
    *,
    title: str,
    axis_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    image = ax.imshow(matrix, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")
    indices = _sample_indices(len(labels), 26)
    ax.set_xticks(indices)
    ax.set_yticks(indices)
    ax.set_xticklabels([labels[index] for index in indices], rotation=90, fontsize=7)
    ax.set_yticklabels([labels[index] for index in indices], fontsize=7)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    ax.set_title(title)
    fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.025,
        label="Pearson similarity",
    )
    fig.text(
        0.5,
        0.012,
        "Strict lower triangle only; diagonal and mirrored upper triangle omitted.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _compact_number(value: float) -> str:
    return f"{float(value):g}"


def _filename_number(value: float) -> str:
    return _compact_number(value).replace("-", "m").replace(".", "p")


def _plot_one_minus_similarity_diagnostic(
    path: Path,
    similarity: np.ndarray,
    labels: Sequence[str],
    *,
    title: str,
    axis_label: str,
) -> None:
    dissimilarity = 1.0 - np.asarray(similarity, dtype=float)
    lower = strict_lower_triangle(dissimilarity)
    finite = lower[np.isfinite(lower)]
    maximum = float(np.max(finite)) if finite.size else 1.0
    vmax = max(maximum, 1.0e-6)
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    image = ax.imshow(
        lower,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    indices = _sample_indices(len(labels), 26)
    ax.set_xticks(indices)
    ax.set_yticks(indices)
    ax.set_xticklabels([labels[index] for index in indices], rotation=90, fontsize=7)
    ax.set_yticklabels([labels[index] for index in indices], fontsize=7)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    ax.set_title(title + f"\nDiagnostic scale: 1 - r, maximum={maximum:.6g}")
    fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.025,
        label="1 - Pearson similarity (0 = identical shape)",
    )
    fig.text(
        0.5,
        0.012,
        "Diagnostic contrast only; use the fixed [-1, 1] correlation map as primary.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _window_role_specs(output_root: Path) -> list[dict[str, Any]]:
    audit_root = output_root / "window_full_symmetric_audit"
    return [
        {
            "role": "single_spots",
            "source": audit_root / "single_crystal" / "spots",
            "destination": output_root / "single_crystal" / "windows" / "spots",
            "title": "Single crystal spots — official 22-frame compression ladder",
        },
        {
            "role": "powder_spots",
            "source": audit_root / "powder" / "spots",
            "destination": output_root / "powder" / "windows" / "spots",
            "title": "Powder spots — 1060 accepted frames",
        },
        {
            "role": "powder_fit_control",
            "source": audit_root / "powder" / "fit_control",
            "destination": output_root / "powder" / "windows" / "fit_control",
            "title": "Powder tungsten-dominated fit control — 1060 accepted frames",
        },
    ]


def _write_within_frame_unique_pairs_gz(
    path: Path,
    *,
    role: str,
    frame_ids: np.ndarray,
    frame_scans: np.ndarray,
    frame_pressures: np.ndarray,
    window_labels: Sequence[str],
    matrices: np.ndarray,
) -> int:
    fields = [
        "role",
        "frame",
        "scan",
        "pressure_GPa",
        "window_row_index",
        "window_col_index",
        "window_row",
        "window_col",
        "similarity",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_index in range(matrices.shape[0]):
            matrix = matrices[frame_index]
            for row_index in range(1, matrix.shape[0]):
                for col_index in range(row_index):
                    writer.writerow(
                        {
                            "role": role,
                            "frame": int(frame_ids[frame_index]),
                            "scan": str(frame_scans[frame_index]),
                            "pressure_GPa": csv_value(frame_pressures[frame_index]),
                            "window_row_index": row_index,
                            "window_col_index": col_index,
                            "window_row": window_labels[row_index],
                            "window_col": window_labels[col_index],
                            "similarity": csv_value(matrix[row_index, col_index]),
                        }
                    )
                    written += 1
    return written


def write_lower_triangle_window_results(output_root: Path) -> dict[str, Any]:
    method_specs = [
        (
            "acf_strict",
            "acf_strict_aggregate",
            "acf_strict_support",
            "acf_strict_ci_low",
            "acf_strict_ci_high",
        ),
        (
            "direct_strict",
            "direct_strict_aggregate",
            "direct_strict_support",
            "direct_strict_ci_low",
            "direct_strict_ci_high",
        ),
        (
            "shift_tolerant_secondary",
            "shift_tolerant_secondary_aggregate",
            "shift_tolerant_secondary_support",
            "shift_tolerant_secondary_ci_low",
            "shift_tolerant_secondary_ci_high",
        ),
    ]
    output_index_rows: list[dict[str, Any]] = []
    role_audits: list[dict[str, Any]] = []
    total_across_maps = 0
    total_within_maps = 0
    total_across_pair_rows = 0
    total_within_frame_pair_rows = 0
    total_within_summary_pair_rows = 0
    diagnostic_rows: list[dict[str, Any]] = []

    for spec in _window_role_specs(output_root):
        role = str(spec["role"])
        source = Path(spec["source"])
        destination = Path(spec["destination"])
        title_prefix = str(spec["title"])
        destination.mkdir(parents=True, exist_ok=True)

        across_archive_path = source / "across_frames" / "across_frame_matrices.npz"
        with np.load(across_archive_path, allow_pickle=False) as archive:
            pressure_values = np.asarray(archive["pressure_gpa"], dtype=float)
            window_starts = np.asarray(archive["window_starts_deg"], dtype=float)
            window_ends = np.asarray(archive["window_ends_deg"], dtype=float)
            across_arrays = {
                key: np.asarray(archive[key])
                for method in method_specs
                for key in method[1:]
            }
        pressure_labels = [f"{value:g}" for value in pressure_values]
        across_pair_rows: list[dict[str, Any]] = []
        role_across_maps = 0
        for (
            method,
            similarity_key,
            support_key,
            ci_low_key,
            ci_high_key,
        ) in method_specs:
            similarity_cube = np.asarray(across_arrays[similarity_key], dtype=float)
            support_cube = np.asarray(across_arrays[support_key])
            ci_low_cube = np.asarray(across_arrays[ci_low_key], dtype=float)
            ci_high_cube = np.asarray(across_arrays[ci_high_key], dtype=float)
            for window_index in range(similarity_cube.shape[0]):
                full = similarity_cube[window_index]
                _assert_symmetric_matrix(
                    full,
                    f"{role}:{method}:window={window_index}",
                )
                lower = strict_lower_triangle(full)
                unique_values = full[np.tril_indices(full.shape[0], k=-1)]
                finite_unique = unique_values[np.isfinite(unique_values)]
                offdiag_min = (
                    float(np.min(finite_unique)) if finite_unique.size else math.nan
                )
                offdiag_median = (
                    float(np.median(finite_unique)) if finite_unique.size else math.nan
                )
                offdiag_max = (
                    float(np.max(finite_unique)) if finite_unique.size else math.nan
                )
                offdiag_std = (
                    float(np.std(finite_unique)) if finite_unique.size else math.nan
                )
                fraction_above_0p99 = (
                    float(np.mean(finite_unique > 0.99))
                    if finite_unique.size
                    else math.nan
                )
                window_token = (
                    f"window_{window_index:02d}_"
                    f"{_filename_number(window_starts[window_index])}_"
                    f"{_filename_number(window_ends[window_index])}"
                )
                matrix_path = (
                    destination
                    / "across_frames"
                    / method
                    / "matrices"
                    / f"{window_token}.csv"
                )
                heatmap_path = (
                    destination
                    / "across_frames"
                    / method
                    / "heatmaps"
                    / f"{window_token}.png"
                )
                write_matrix_csv(
                    matrix_path,
                    pressure_labels,
                    pressure_labels,
                    lower,
                    row_header="pressure_GPa",
                )
                _plot_window_quicklook(
                    heatmap_path,
                    lower,
                    pressure_labels,
                    title=(
                        f"{title_prefix}\nAcross frames — {method}, "
                        f"{_compact_number(window_starts[window_index])}–"
                        f"{_compact_number(window_ends[window_index])}°"
                        f"\nOff-diagonal r range: "
                        f"{offdiag_min:.4f} to {offdiag_max:.4f}"
                    ),
                    axis_label="Pressure (GPa)",
                )
                diagnostic_csv = ""
                diagnostic_png = ""
                if role == "powder_fit_control" and method == "acf_strict":
                    diagnostic_matrix_path = (
                        destination
                        / "across_frames"
                        / method
                        / "one_minus_similarity_diagnostics"
                        / "matrices"
                        / f"{window_token}.csv"
                    )
                    diagnostic_heatmap_path = (
                        destination
                        / "across_frames"
                        / method
                        / "one_minus_similarity_diagnostics"
                        / "heatmaps"
                        / f"{window_token}.png"
                    )
                    write_matrix_csv(
                        diagnostic_matrix_path,
                        pressure_labels,
                        pressure_labels,
                        strict_lower_triangle(1.0 - full),
                        row_header="pressure_GPa",
                    )
                    _plot_one_minus_similarity_diagnostic(
                        diagnostic_heatmap_path,
                        full,
                        pressure_labels,
                        title=(
                            f"{title_prefix}\nAcross frames — {method}, "
                            f"{_compact_number(window_starts[window_index])}–"
                            f"{_compact_number(window_ends[window_index])}°"
                        ),
                        axis_label="Pressure (GPa)",
                    )
                    diagnostic_csv = str(
                        diagnostic_matrix_path.relative_to(output_root)
                    )
                    diagnostic_png = str(
                        diagnostic_heatmap_path.relative_to(output_root)
                    )
                output_index_rows.append(
                    {
                        "role": role,
                        "comparison": "across_frames",
                        "method": method,
                        "scope": f"window_{window_index:02d}",
                        "matrix_csv": str(matrix_path.relative_to(output_root)),
                        "heatmap_png": str(heatmap_path.relative_to(output_root)),
                        "one_minus_similarity_diagnostic_csv": diagnostic_csv,
                        "one_minus_similarity_diagnostic_png": diagnostic_png,
                    }
                )
                diagnostic_rows.append(
                    {
                        "role": role,
                        "method": method,
                        "window_index": window_index,
                        "window_start_deg": window_starts[window_index],
                        "window_end_deg": window_ends[window_index],
                        "finite_unique_pressure_pairs": finite_unique.size,
                        "offdiagonal_min": offdiag_min,
                        "offdiagonal_median": offdiag_median,
                        "offdiagonal_max": offdiag_max,
                        "offdiagonal_std": offdiag_std,
                        "fraction_above_0p99": fraction_above_0p99,
                        "primary_scale": "fixed_-1_to_1",
                        "one_minus_similarity_diagnostic_csv": diagnostic_csv,
                        "one_minus_similarity_diagnostic_png": diagnostic_png,
                    }
                )
                role_across_maps += 1
                for row_index in range(1, pressure_values.size):
                    for col_index in range(row_index):
                        across_pair_rows.append(
                            {
                                "role": role,
                                "method": method,
                                "window_index": window_index,
                                "window_start_deg": window_starts[window_index],
                                "window_end_deg": window_ends[window_index],
                                "pressure_row_GPa": pressure_values[row_index],
                                "pressure_col_GPa": pressure_values[col_index],
                                "similarity": full[row_index, col_index],
                                "support": support_cube[
                                    window_index, row_index, col_index
                                ],
                                "ci_low": ci_low_cube[
                                    window_index, row_index, col_index
                                ],
                                "ci_high": ci_high_cube[
                                    window_index, row_index, col_index
                                ],
                            }
                        )
        across_pairs_path = (
            destination / "across_frames" / "unique_lower_triangle_pairs.csv"
        )
        write_csv(across_pairs_path, across_pair_rows)
        total_across_pair_rows += len(across_pair_rows)
        total_across_maps += role_across_maps

        within_archive_path = source / "within_frame" / "within_frame_matrices.npz"
        with np.load(within_archive_path, allow_pickle=False) as archive:
            frame_ids = np.asarray(archive["frame_ids"], dtype=int)
            frame_scans = np.asarray(archive["frame_scans"])
            frame_pressures = np.asarray(archive["frame_pressure_gpa"], dtype=float)
            within_pressures = np.asarray(archive["pressure_gpa"], dtype=float)
            within_starts = np.asarray(archive["window_starts_deg"], dtype=float)
            within_ends = np.asarray(archive["window_ends_deg"], dtype=float)
            aggregate = np.asarray(archive["aggregate"], dtype=float)
            aggregate_ci_low = np.asarray(archive["aggregate_ci_low"], dtype=float)
            aggregate_ci_high = np.asarray(archive["aggregate_ci_high"], dtype=float)
            aggregate_support = np.asarray(archive["support"])
            aggregate_by_pressure = np.asarray(
                archive["aggregate_by_pressure"], dtype=float
            )
            by_pressure_ci_low = np.asarray(
                archive["aggregate_by_pressure_ci_low"], dtype=float
            )
            by_pressure_ci_high = np.asarray(
                archive["aggregate_by_pressure_ci_high"], dtype=float
            )
            by_pressure_support = np.asarray(archive["support_by_pressure"])
            matrices_by_frame = np.asarray(archive["matrices_by_frame"], dtype=float)
        window_labels = [
            f"{_compact_number(start)}–{_compact_number(end)}°"
            for start, end in zip(within_starts, within_ends, strict=True)
        ]
        _assert_symmetric_matrix(aggregate, f"{role}:within:aggregate")
        aggregate_lower = strict_lower_triangle(aggregate)
        within_aggregate_csv = (
            destination / "within_frame" / "aggregate" / "matrix.csv"
        )
        within_aggregate_png = (
            destination / "within_frame" / "aggregate" / "heatmap.png"
        )
        write_matrix_csv(
            within_aggregate_csv,
            window_labels,
            window_labels,
            aggregate_lower,
            row_header="two_theta_window",
        )
        _plot_window_quicklook(
            within_aggregate_png,
            aggregate_lower,
            window_labels,
            title=title_prefix + "\nWithin frame — aggregate window ACF",
            axis_label="2θ window (degrees)",
        )
        output_index_rows.append(
            {
                "role": role,
                "comparison": "within_frame",
                "method": "acf",
                "scope": "aggregate",
                "matrix_csv": str(within_aggregate_csv.relative_to(output_root)),
                "heatmap_png": str(within_aggregate_png.relative_to(output_root)),
            }
        )
        role_within_maps = 1
        within_summary_rows: list[dict[str, Any]] = []
        for row_index in range(1, len(window_labels)):
            for col_index in range(row_index):
                within_summary_rows.append(
                    {
                        "role": role,
                        "scope": "aggregate",
                        "pressure_GPa": math.nan,
                        "window_row_index": row_index,
                        "window_col_index": col_index,
                        "window_row": window_labels[row_index],
                        "window_col": window_labels[col_index],
                        "similarity": aggregate[row_index, col_index],
                        "support": aggregate_support[row_index, col_index],
                        "ci_low": aggregate_ci_low[row_index, col_index],
                        "ci_high": aggregate_ci_high[row_index, col_index],
                    }
                )

        for pressure_index, pressure in enumerate(within_pressures):
            full = aggregate_by_pressure[pressure_index]
            _assert_symmetric_matrix(
                full,
                f"{role}:within:pressure={pressure:g}",
            )
            lower = strict_lower_triangle(full)
            pressure_token = f"{pressure:g}GPa"
            matrix_path = (
                destination
                / "within_frame"
                / "by_pressure"
                / "matrices"
                / f"{pressure_token}.csv"
            )
            heatmap_path = (
                destination
                / "within_frame"
                / "by_pressure"
                / "heatmaps"
                / f"{pressure_token}.png"
            )
            write_matrix_csv(
                matrix_path,
                window_labels,
                window_labels,
                lower,
                row_header="two_theta_window",
            )
            _plot_window_quicklook(
                heatmap_path,
                lower,
                window_labels,
                title=title_prefix + f"\nWithin frame at {pressure:g} GPa",
                axis_label="2θ window (degrees)",
            )
            output_index_rows.append(
                {
                    "role": role,
                    "comparison": "within_frame",
                    "method": "acf",
                    "scope": f"pressure_{pressure:g}GPa",
                    "matrix_csv": str(matrix_path.relative_to(output_root)),
                    "heatmap_png": str(heatmap_path.relative_to(output_root)),
                }
            )
            role_within_maps += 1
            for row_index in range(1, len(window_labels)):
                for col_index in range(row_index):
                    within_summary_rows.append(
                        {
                            "role": role,
                            "scope": "by_pressure",
                            "pressure_GPa": pressure,
                            "window_row_index": row_index,
                            "window_col_index": col_index,
                            "window_row": window_labels[row_index],
                            "window_col": window_labels[col_index],
                            "similarity": full[row_index, col_index],
                            "support": by_pressure_support[
                                pressure_index, row_index, col_index
                            ],
                            "ci_low": by_pressure_ci_low[
                                pressure_index, row_index, col_index
                            ],
                            "ci_high": by_pressure_ci_high[
                                pressure_index, row_index, col_index
                            ],
                        }
                    )
        within_summary_path = (
            destination / "within_frame" / "unique_summary_pairs.csv"
        )
        write_csv(within_summary_path, within_summary_rows)
        total_within_summary_pair_rows += len(within_summary_rows)

        for frame_index in range(matrices_by_frame.shape[0]):
            _assert_symmetric_matrix(
                matrices_by_frame[frame_index],
                f"{role}:within:frame={frame_ids[frame_index]}",
            )
        triangle_mask = np.tril(
            np.ones((len(window_labels), len(window_labels)), dtype=bool),
            k=-1,
        )
        lower_by_frame = np.where(
            triangle_mask[None, :, :],
            matrices_by_frame,
            np.nan,
        )
        lower_npz_path = (
            destination / "within_frame" / "per_frame_lower_triangle.npz"
        )
        lower_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            lower_npz_path,
            frame_ids=frame_ids,
            frame_scans=frame_scans,
            frame_pressure_gpa=frame_pressures,
            window_starts_deg=within_starts,
            window_ends_deg=within_ends,
            matrices_by_frame=lower_by_frame,
        )
        per_frame_pairs_path = (
            destination / "within_frame" / "per_frame_unique_pairs.csv.gz"
        )
        within_frame_rows = _write_within_frame_unique_pairs_gz(
            per_frame_pairs_path,
            role=role,
            frame_ids=frame_ids,
            frame_scans=frame_scans,
            frame_pressures=frame_pressures,
            window_labels=window_labels,
            matrices=matrices_by_frame,
        )
        expected_frame_rows = matrices_by_frame.shape[0] * (
            len(window_labels) * (len(window_labels) - 1) // 2
        )
        if within_frame_rows != expected_frame_rows:
            raise RuntimeError(
                f"within-frame unique pair count mismatch for {role}: "
                f"{within_frame_rows} != {expected_frame_rows}"
            )
        total_within_frame_pair_rows += within_frame_rows
        total_within_maps += role_within_maps
        role_audits.append(
            {
                "role": role,
                "across_maps": role_across_maps,
                "across_unique_pairs": len(across_pair_rows),
                "within_maps": role_within_maps,
                "within_summary_unique_pairs": len(within_summary_rows),
                "within_frames": matrices_by_frame.shape[0],
                "within_frame_unique_pairs": within_frame_rows,
                "strict_lower_triangle_only": True,
                "diagonal_omitted": True,
                "full_symmetric_source_retained_in_audit": True,
            }
        )

    index_path = output_root / "window_lower_triangle_index.csv"
    write_csv(index_path, output_index_rows)
    diagnostics_path = output_root / "window_similarity_diagnostics.csv"
    write_csv(diagnostics_path, diagnostic_rows)
    methods_path = output_root / "LOWER_TRIANGLE_METHODS.md"
    methods_path.write_text(
        """# Primary window-correlation outputs

All user-facing window correlation matrices and heatmaps retain only the
strict lower triangle.  The diagonal is omitted because a valid
self-correlation is one, and the upper triangle is omitted because it mirrors
the lower triangle.

`across_frames/*/unique_lower_triangle_pairs.csv` contains exactly one row per
pressure pair, method, and angle window, with support and confidence intervals.
`within_frame/unique_summary_pairs.csv` contains aggregate and pressure-specific
unique window pairs.  `within_frame/per_frame_unique_pairs.csv.gz` and
`per_frame_lower_triangle.npz` retain every unique pair in every included frame.

The freshly recomputed full symmetric numerical source products are retained
only under `window_full_symmetric_audit/` for provenance and reproducibility.
""",
        encoding="utf-8",
    )
    fit_first_acf = next(
        (
            row
            for row in diagnostic_rows
            if row["role"] == "powder_fit_control"
            and row["method"] == "acf_strict"
            and row["window_index"] == 0
        ),
        {},
    )
    return {
        "roles": len(role_audits),
        "role_audits": role_audits,
        "across_maps": total_across_maps,
        "within_maps": total_within_maps,
        "total_maps": total_across_maps + total_within_maps,
        "across_unique_pair_rows": total_across_pair_rows,
        "within_summary_unique_pair_rows": total_within_summary_pair_rows,
        "within_frame_unique_pair_rows": total_within_frame_pair_rows,
        "index": str(index_path.relative_to(output_root)),
        "diagnostics": str(diagnostics_path.relative_to(output_root)),
        "powder_fit_control_one_minus_acf_maps": sum(
            bool(row.get("one_minus_similarity_diagnostic_png"))
            for row in diagnostic_rows
        ),
        "powder_fit_control_first_acf_window": fit_first_acf,
        "strict_lower_triangle_only": True,
        "diagonal_omitted": True,
        "full_symmetric_source_retained_in_audit": True,
    }


def write_window_quicklooks(output_root: Path) -> dict[str, Any]:
    quicklook_root = output_root / "window_quicklooks"
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for spec in _window_role_specs(output_root):
        role = str(spec["role"])
        source = Path(spec["source"])
        title_prefix = str(spec["title"])
        across_archive = source / "across_frames" / "across_frame_matrices.npz"
        with np.load(across_archive, allow_pickle=False) as archive:
            pressures = np.asarray(archive["pressure_gpa"], dtype=float)
            by_window = np.asarray(archive["acf_strict_aggregate"], dtype=float)
            across_window_count = len(
                np.asarray(archive["window_starts_deg"], dtype=float)
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            across_matrix = np.nanmedian(by_window, axis=0)
        _assert_symmetric_matrix(across_matrix, f"{role}:quicklook:across")
        across_lower = strict_lower_triangle(across_matrix)
        pressure_labels = [f"{value:g}" for value in pressures]
        across_csv = quicklook_root / f"{role}_across_frames_median_acf.csv"
        across_png = quicklook_root / f"{role}_across_frames_median_acf.png"
        write_matrix_csv(
            across_csv,
            pressure_labels,
            pressure_labels,
            across_lower,
            row_header="pressure_GPa",
        )
        _plot_window_quicklook(
            across_png,
            across_lower,
            pressure_labels,
            title=(
                title_prefix
                + f"\nAcross frames: median strict ACF over {across_window_count} "
                "integer windows — lower triangle"
            ),
            axis_label="Pressure (GPa)",
        )
        rows.append(
            {
                "role": role,
                "comparison": "across_frames",
                "scope": (
                    f"median strict ACF over {across_window_count} "
                    "same-angle integer windows"
                ),
                "matrix_csv": str(across_csv.relative_to(output_root)),
                "heatmap_png": str(across_png.relative_to(output_root)),
                "triangle_policy": "strict_lower_no_diagonal",
            }
        )
        for row_index in range(1, len(pressure_labels)):
            for col_index in range(row_index):
                pair_rows.append(
                    {
                        "role": role,
                        "comparison": "across_frames",
                        "row_index_0based": row_index,
                        "column_index_0based": col_index,
                        "row_label": pressure_labels[row_index],
                        "column_label": pressure_labels[col_index],
                        "similarity": across_matrix[row_index, col_index],
                    }
                )

        within_archive = source / "within_frame" / "within_frame_matrices.npz"
        with np.load(within_archive, allow_pickle=False) as archive:
            starts = np.asarray(archive["window_starts_deg"], dtype=float)
            ends = np.asarray(archive["window_ends_deg"], dtype=float)
            within_matrix = np.asarray(archive["aggregate"], dtype=float)
        _assert_symmetric_matrix(within_matrix, f"{role}:quicklook:within")
        within_lower = strict_lower_triangle(within_matrix)
        window_labels = [
            f"{_compact_number(start)}–{_compact_number(end)}°"
            for start, end in zip(starts, ends, strict=True)
        ]
        within_csv = quicklook_root / f"{role}_within_frame_window_acf.csv"
        within_png = quicklook_root / f"{role}_within_frame_window_acf.png"
        write_matrix_csv(
            within_csv,
            window_labels,
            window_labels,
            within_lower,
            row_header="two_theta_window",
        )
        _plot_window_quicklook(
            within_png,
            within_lower,
            window_labels,
            title=(
                title_prefix
                + f"\nWithin frame: {len(window_labels)} × {len(window_labels)} "
                "integer-window ACF — lower triangle"
            ),
            axis_label="2θ window (degrees)",
        )
        rows.append(
            {
                "role": role,
                "comparison": "within_frame",
                "scope": (
                    f"aggregate {len(window_labels)} x {len(window_labels)} "
                    "integer angle-window ACF"
                ),
                "matrix_csv": str(within_csv.relative_to(output_root)),
                "heatmap_png": str(within_png.relative_to(output_root)),
                "triangle_policy": "strict_lower_no_diagonal",
            }
        )
        for row_index in range(1, len(window_labels)):
            for col_index in range(row_index):
                pair_rows.append(
                    {
                        "role": role,
                        "comparison": "within_frame",
                        "row_index_0based": row_index,
                        "column_index_0based": col_index,
                        "row_label": window_labels[row_index],
                        "column_label": window_labels[col_index],
                        "similarity": within_matrix[row_index, col_index],
                    }
                )
    write_csv(quicklook_root / "quicklook_index.csv", rows)
    write_csv(quicklook_root / "unique_lower_triangle_pairs.csv", pair_rows)
    return {
        "roles": len(_window_role_specs(output_root)),
        "maps": len(rows),
        "unique_pair_rows": len(pair_rows),
        "index": str((quicklook_root / "quicklook_index.csv").relative_to(output_root)),
        "axis_labels_corrected_in_quicklooks": True,
        "strict_lower_triangle_only": True,
        "diagonal_omitted": True,
        "frozen_source_files_modified": False,
    }


def build_artifact_index(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "suffix": path.suffix.lower(),
            }
        )
    return rows


def write_suite_readme(
    path: Path,
    validations: Mapping[str, Any],
    window_audit: Mapping[str, Any],
    *,
    tolerance: float,
    peak_plots_delivered: bool,
) -> None:
    single_all = validations["single_crystal_all_frames"]
    single_roi = validations["single_crystal_curated_2d"]
    powder = validations["powder_all_spots"]
    lower_triangle = window_audit.get("lower_triangle", {})
    expected_window_roles = {
        "single_spots_across",
        "single_spots_within",
        "powder_spots_across",
        "powder_spots_within",
        "powder_fit_control_across",
        "powder_fit_control_within",
    }
    windows_delivered = (
        set(window_audit.get("roles_verified", [])) == expected_window_roles
        and window_audit.get("window_calculations_regenerated") is True
    )
    fit_first = lower_triangle.get("powder_fit_control_first_acf_window", {})
    window_delivery_text = (
        """Six freshly recomputed and independently validated result roles are
bundled: single-crystal
spots across frames and within frame; powder spots across/within; and powder
tungsten-dominated fit-control across/within."""
        if windows_delivered
        else """Window recomputation was intentionally skipped for this partial
run; it is not a complete deliverable."""
    )
    text = f"""# Track-independent all-peak correlation suite

## What is new

No peak is grouped or filtered by track lifetime.  For every non-empty frame
pair, all `N_i * N_j` peak combinations are evaluated.  The canonical long
tables contain each unordered frame pair once.

The presentation layout is now one map per anchor peak.  Every registered
frame occupies one row.  Columns are only `peak 1`, `peak 2`, ... up to that
dataset's maximum number of peaks in one frame.  A column is a frame-local
slot sorted by 2theta; it is not a cross-frame track identity.  The anchor
frame is entirely blank because same-frame comparison is excluded.

Radial-location similarity (2theta only; azimuth is provenance, not part of the score):

`clip(1 - abs(delta 2theta) / {tolerance:g} deg, 0, 1)`

Integrated-area similarity:

`min(A_i, A_j) / max(A_i, A_j)`

The area input is always an integration, never peak height and never ROI mean
intensity per pixel.  If both numerical integrals are exactly zero, their area
similarity is defined as one.

## Peak-map coverage

- Single crystal all-frame 1D spot channel: {single_all['registered_frames']}
  frames, {single_all['peaks']} frame-local candidates,
  {single_all['written_cross_frame_peak_rows']} complete cross-frame cells.
  Each anchor map is `{single_all['registered_frames']} x
  {single_all['max_local_peak_slots']}`.
  All 28 available frames are freshly processed independently.  Both reliable
  and audit-state candidates are retained because recurrence and track support
  are forbidden filters ({single_all['source_audit']['reliable_candidates']}
  reliable and {single_all['source_audit']['audit_state_candidates']} audit-state
  candidates).  Areas are trapezoidal integrals of the positive
  AsLS-background-subtracted `spots_masked` 1D signal with interpolated ROI
  endpoints; fitted areas are recorded but not scored.  This is not a claim of
  raw-2D redetection in all 28 frames.
- Single crystal curated 2D ROI companion: {single_roi['peaks']} observations
  in {single_roi['nonempty_frames']} frames,
  {single_roi['written_cross_frame_peak_rows']} cells.  All
  {single_roi['registered_frames']} registered frames remain visible, including
  {single_roi['zero_peak_frames']} all-white zero-peak rows; each anchor map is
  `{single_roi['registered_frames']} x {single_roi['max_local_peak_slots']}`.
  Areas are raw-TIFF
  background-subtracted ROI pixel sums divided by verified exposure.  This is
  an audit companion, not a claim of fresh 2D detection in all 28 frames.
- Powder all detected 2D spots: {powder['peaks']} observations in
  {powder['nonempty_frames']} non-empty frames,
  {powder['written_cross_frame_peak_rows']} cells.  This includes every
  exported untracked/short-lived observation.  The source `area` is integrated
  counts above the ring median.  Raw integrated counts are used because D1w
  exposure semantics are unavailable; no peak is dropped.  All
  {powder['registered_frames']} accepted frames remain visible, including
  {powder['zero_peak_frames']} all-white zero-peak rows; each anchor map is
  `{powder['registered_frames']} x {powder['max_local_peak_slots']}`.

The single-crystal pressure range is 1.0--12.8 GPa.  The powder experiment is a
different pressure series spanning 3.5--50.7 GPa; this is why powder maps show
values above 50 GPa while single-crystal maps stop near 12.8 GPa.

The powder registry has 519 peaks in total, but they are not placed into 519
horizontal columns.  Its maximum simultaneous count is only
{powder['max_local_peak_slots']}, so every powder anchor map has only
{powder['max_local_peak_slots']} columns.  For example, if a target frame has
two peaks, only `peak 1` and `peak 2` contain scores and the remaining slots
are blank.  A zero-peak frame is wholly blank.  Blank is stored as NaN, while
a genuine similarity of zero remains a dark-purple numerical cell.

## Window-to-window coverage

{window_delivery_text}

The nominal windows are now fixed in absolute 2theta degrees: `0-5`, `1-6`,
`2-7`, and so on.  Single crystal has 19 windows through `18-23`; powder has
28 windows through `27-32`.  The detector coordinates begin slightly above
zero (single: 0.01577496 degrees; powder: 0.04347043 degrees), so the first
nominal window uses only its observed support through 5 degrees.  No value is
invented, extrapolated, or padded below the detector edge.  Both nominal and
effective bounds are exported in `window_definition.csv` and
`window_provenance/integer_window_geometry.csv`.

Across-frame results include strict ACF, direct-signal validation, and a
+/-1-degree neighboring-window secondary analysis.  The primary presentation
retains only the strict lower triangle and omits the diagonal.  It contains
{lower_triangle.get('across_maps', 0)} across-frame maps and
{lower_triangle.get('within_maps', 0)} within-frame aggregate/by-pressure maps.
Canonical unique-pair tables retain support and confidence intervals, plus all
{lower_triangle.get('within_frame_unique_pair_rows', 0)} per-frame window
pairs.

The single-crystal window analysis intentionally uses the official 22-frame
compression ladder (two orientations x eleven pressures).  The six isolated,
decompression, or repeat-exposure frames remain in the all-28 peak maps but are
not mixed into official pressure-ladder window aggregates.  Powder window
analysis uses all 1060 accepted frames.  See `WINDOW_METHODS.md` and
`window_provenance/`.  The six `window_quicklooks/` maps use explicit
`Pressure (GPa)` axes for across-frame summaries and `2theta window` axes for
within-frame summaries.  Freshly recomputed full symmetric numerical matrices
are isolated under `window_full_symmetric_audit/`.

The powder fit-control ACF map can still look uniformly red on the fixed
`[-1, 1]` scale.  In its new `0-5` window, the off-diagonal pressure-pair
values are min/median/max =
{fit_first.get('offdiagonal_min', math.nan):.6f}/
{fit_first.get('offdiagonal_median', math.nan):.6f}/
{fit_first.get('offdiagonal_max', math.nan):.6f}; they are close to one but
are not all exactly one.  This channel is a highly repetitive
`fit:sigmaclip` control/background shape, and ACF normalization deliberately
removes mean and amplitude.  It therefore means “the autocorrelation shapes
are very similar,” not “all raw patterns or peaks are identical.”  Every
fit-control ACF map has a companion `1-r` diagnostic map and the exact
off-diagonal statistics are in `window_similarity_diagnostics.csv`.

## How to read the outputs

- `peak_registry.csv`: frame-local IDs, coordinates, integrated areas, and
  source provenance.  `track` is provenance only.
- `all_cross_frame_peak_pairs.csv.gz`: authoritative all-cell table.
- `frame_pair_index.csv`: dimensions and row offsets for every rectangular
  `N_i x N_j` frame-pair block.
- `frame_slot_layout.csv`: exact map row order, pressure, peak count, and
  zero-peak status for every registered frame.
- `per_anchor_peak_matrices/` and `per_anchor_peak_heatmaps/`: one location
  and one ROI/integrated-area map for every anchor peak.  Files are grouped by
  anchor frame; rows are frames and columns are local peak numbers only.
- `per_anchor_peak_map_index.csv`: anchor metadata, dimensions, finite/blank
  counts, and paths to both metrics.
- `all_peak_similarity_matrices.npz`: compact global overview arrays; same-frame
  diagonal blocks are NaN because the request specifies other frames.
- `windows/`: complete track-independent across-frame and within-frame window
  correlations, shown as strict lower triangles with no diagonal.
- `window_quicklooks/`: six presentation maps with unambiguous pressure versus
  angle-window axis labels, plus canonical unique-pair CSV data.
- `window_full_symmetric_audit/`: freshly recomputed full-square numerical
  source products kept for reproducibility.
- `window_similarity_diagnostics.csv`: numerical off-diagonal ranges for every
  across-frame map; fit-control ACF rows also link to `1-r` diagnostic maps.
- `window_provenance/`: exact input hashes, nominal/effective window bounds,
  profiles, and integer-window run metadata.
- `artifact_index.csv`: hashes the payload written before the index and final
  `RUN_COMPLETE.json`; the completion marker records the index's own hash.

White in peak heatmaps means a missing local slot, a zero-peak frame, or the
excluded anchor frame; it does not mean correlation zero.  In square window
maps, white on the diagonal and upper triangle is structural omission, not a
measured zero.  A colored lower-triangle cell is the one retained value for
that pair.

Peak PNG heatmaps delivered in this run: {peak_plots_delivered}
({2 * (single_all['peaks'] + single_roi['peaks'] + powder['peaks'])} files).
Integer-window calculations regenerated from raw XY:
{window_audit.get('window_calculations_regenerated', False)}.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tolerance = float(args.position_tolerance_deg)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise SystemExit("--position-tolerance-deg must be finite and positive")

    single_all_peaks, single_all_frames, single_all_audit = load_single_all_frame_peaks(
        args.single_manifest.expanduser().resolve(),
        args.single_profile.expanduser().resolve(),
    )
    single_roi_peaks, single_roi_frames, single_roi_audit = load_single_curated_roi_peaks(
        args.single_uncertainty_table.expanduser().resolve(),
        args.single_manifest.expanduser().resolve(),
    )
    powder_peaks, powder_frames, powder_audit = load_powder_all_spot_peaks(
        args.powder_observations.expanduser().resolve(),
        args.powder_manifest.expanduser().resolve(),
    )

    validations = {
        "single_crystal_all_frames": generate_peak_dataset(
            out_dir / "single_crystal" / "all_frame_1d_peak_maps",
            "single_crystal_all_frames",
            single_all_peaks,
            single_all_frames,
            single_all_audit,
            tolerance,
            not args.no_plots,
        ),
        "single_crystal_curated_2d": generate_peak_dataset(
            out_dir / "single_crystal" / "curated_2d_roi_peak_maps",
            "single_crystal_curated_2d",
            single_roi_peaks,
            single_roi_frames,
            single_roi_audit,
            tolerance,
            not args.no_plots,
        ),
        "powder_all_spots": generate_peak_dataset(
            out_dir / "powder" / "all_detected_spot_peak_maps",
            "powder_all_detected_spots",
            powder_peaks,
            powder_frames,
            powder_audit,
            tolerance,
            not args.no_plots,
        ),
    }

    window_audit: dict[str, Any] = {
        "roles_verified": [],
        "total_files": 0,
        "window_calculations_regenerated": False,
        "integer_window_geometry_verified": False,
        "all_source_matrices_symmetric": False,
        "all_scores_in_minus1_plus1": False,
        "all_signal_and_acf_windows_valid": False,
        "quicklooks": {
            "roles": 0,
            "maps": 0,
            "axis_labels_corrected_in_quicklooks": False,
        },
        "lower_triangle": {
            "roles": 0,
            "total_maps": 0,
            "strict_lower_triangle_only": False,
            "diagonal_omitted": False,
        },
    }
    if not args.no_window_copy:
        from integer_window_correlations import (  # noqa: WPS433
            generate_integer_window_sources,
        )

        window_audit = generate_integer_window_sources(
            out_dir,
            workers=max(1, int(args.window_workers)),
            make_full_symmetric_plots=False,
            single_root=args.single_window_input_root.expanduser().resolve(),
            single_manifest=args.single_manifest.expanduser().resolve(),
            single_profile_path=args.single_window_profile.expanduser().resolve(),
            single_wavelength_A=SINGLE_WAVELENGTH_A,
            powder_root=args.powder_window_input_root.expanduser().resolve(),
            powder_manifest=args.powder_window_manifest.expanduser().resolve(),
            powder_profile_path=args.powder_window_profile.expanduser().resolve(),
            powder_wavelength_A=POWDER_WAVELENGTH_A,
        )
        window_audit["lower_triangle"] = write_lower_triangle_window_results(out_dir)
        window_audit["quicklooks"] = write_window_quicklooks(out_dir)
    expected_window_roles = {
        "single_spots_across",
        "single_spots_within",
        "powder_spots_across",
        "powder_spots_within",
        "powder_fit_control_across",
        "powder_fit_control_within",
    }
    generated_roles = set(
        str(item) for item in window_audit.get("roles_verified", [])
    )
    requirements = {
        "no_track_based_peak_selection": all(
            not item["track_used_for_selection_grouping_or_scoring"]
            for item in validations.values()
        ),
        "all_cross_frame_peak_cartesian_products": all(
            item["all_cartesian_cells_written"] for item in validations.values()
        ),
        "per_anchor_peak_frame_slot_maps": all(
            item["per_anchor_peak_maps"] == item["peaks"]
            and item["every_registered_frame_is_a_map_row"]
            and item["frame_slot_layout_verified"]
            and item["all_anchor_peak_shapes_verified"]
            and item["all_anchor_peak_masks_verified"]
            and item["all_anchor_same_frame_rows_blank"]
            and item["all_anchor_peak_finite_counts_complete"]
            for item in validations.values()
        ),
        "every_peak_has_integrated_area_and_area_score": all(
            item["every_peak_has_finite_nonnegative_integrated_area"]
            and item["all_area_scores_finite"]
            for item in validations.values()
        ),
        "single_all_28_manifest_frames_processed": (
            validations["single_crystal_all_frames"]["registered_frames"] == 28
            and validations["single_crystal_all_frames"]["nonempty_frames"] == 28
        ),
        "single_curated_all_source_rows_retained": (
            validations["single_crystal_curated_2d"]["peaks"] == 275
            and validations["single_crystal_curated_2d"]["source_audit"].get(
                "all_source_rows_retained"
            )
            is True
        ),
        "powder_all_source_rows_retained": (
            validations["powder_all_spots"]["peaks"] == 519
            and validations["powder_all_spots"]["source_audit"].get(
                "all_source_rows_retained"
            )
            is True
            and validations["powder_all_spots"]["source_audit"].get(
                "untracked_short_lived_observations_retained"
            )
            == 58
        ),
        "single_crystal_delivered": True,
        "powder_delivered": True,
        "area_uses_integration": all(
            item["source_audit"].get("area_is_numerical_integration")
            or item["source_audit"].get("area_is_raw_TIFF_2D_ROI_integration")
            or item["source_audit"].get("area_is_source_2D_blob_integration")
            for item in validations.values()
        ),
        "peak_heatmap_pngs_delivered": (
            not args.no_plots
            and all(
                item["per_anchor_peak_heatmap_png_files"] == 2 * item["peaks"]
                for item in validations.values()
            )
        ),
        "all_six_window_roles_verified": generated_roles == expected_window_roles,
        "window_results_recomputed_from_raw_xy": (
            window_audit.get("window_calculations_regenerated") is True
            and window_audit.get("unchanged_from_v3_suite") is False
        ),
        "window_integer_geometry_verified": (
            window_audit.get("integer_window_geometry_verified") is True
            and window_audit.get("single_window_scope", {}).get("nominal_windows")
            == 19
            and window_audit.get("single_window_scope", {}).get("first_window")
            == "0-5"
            and window_audit.get("single_window_scope", {}).get("last_window")
            == "18-23"
            and window_audit.get("powder_window_scope", {}).get("nominal_windows")
            == 28
            and window_audit.get("powder_window_scope", {}).get("first_window")
            == "0-5"
            and window_audit.get("powder_window_scope", {}).get("last_window")
            == "27-32"
        ),
        "window_numerical_sources_valid": (
            window_audit.get("all_source_matrices_symmetric") is True
            and window_audit.get("all_scores_in_minus1_plus1") is True
            and window_audit.get("all_signal_and_acf_windows_valid") is True
            and window_audit.get("all_intended_scopes_verified") is True
        ),
        "window_file_inventory_complete": (
            as_int(window_audit.get("total_files")) > 0
            and as_int(window_audit.get("total_bytes")) > 0
            and as_int(window_audit.get("suffix_counts", {}).get(".npz")) == 9
        ),
        "window_across_frames_delivered": {
            "single_spots_across",
            "powder_spots_across",
            "powder_fit_control_across",
        }.issubset(generated_roles),
        "window_within_frame_delivered": {
            "single_spots_within",
            "powder_spots_within",
            "powder_fit_control_within",
        }.issubset(generated_roles),
        "window_quicklooks_with_correct_axis_semantics": (
            window_audit.get("quicklooks", {}).get("maps") == 6
            and window_audit.get("quicklooks", {}).get("unique_pair_rows") == 1324
            and window_audit.get("quicklooks", {}).get(
                "axis_labels_corrected_in_quicklooks"
            )
            is True
            and window_audit.get("quicklooks", {}).get(
                "strict_lower_triangle_only"
            )
            is True
            and window_audit.get("quicklooks", {}).get("diagonal_omitted") is True
        ),
        "window_primary_maps_are_strict_lower_triangle": (
            window_audit.get("lower_triangle", {}).get("roles") == 3
            and window_audit.get("lower_triangle", {}).get("across_maps") == 225
            and window_audit.get("lower_triangle", {}).get("within_maps") == 52
            and window_audit.get("lower_triangle", {}).get("total_maps") == 277
            and window_audit.get("lower_triangle", {}).get(
                "strict_lower_triangle_only"
            )
            is True
            and window_audit.get("lower_triangle", {}).get("diagonal_omitted")
            is True
        ),
        "window_unique_pair_tables_complete": (
            window_audit.get("lower_triangle", {}).get("across_unique_pair_rows")
            == 31863
            and window_audit.get("lower_triangle", {}).get(
                "within_summary_unique_pair_rows"
            )
            == 17172
            and window_audit.get("lower_triangle", {}).get(
                "within_frame_unique_pair_rows"
            )
            == 805122
        ),
        "fit_control_acf_diagnostics_delivered": (
            window_audit.get("lower_triangle", {}).get(
                "powder_fit_control_one_minus_acf_maps"
            )
            == 28
            and bool(
                window_audit.get("lower_triangle", {}).get(
                    "powder_fit_control_first_acf_window"
                )
            )
        ),
    }
    suite_status = "PASS" if all(requirements.values()) else "PARTIAL"
    suite_validation = {
        "status": suite_status,
        "position_tolerance_deg": tolerance,
        "peak_datasets": validations,
        "window_results": window_audit,
        "requirements": requirements,
        "partial_run_flags": {
            "no_plots": bool(args.no_plots),
            "no_window_recomputation": bool(args.no_window_copy),
        },
    }
    if not args.no_plots and not args.no_window_copy and suite_status != "PASS":
        failed = sorted(key for key, passed in requirements.items() if not passed)
        raise RuntimeError(f"complete suite validation failed: {failed}")
    (out_dir / "validation_report.json").write_text(
        json.dumps(json_ready(suite_validation), indent=2),
        encoding="utf-8",
    )
    write_suite_readme(
        out_dir / "README.md",
        validations,
        window_audit,
        tolerance=tolerance,
        peak_plots_delivered=not args.no_plots,
    )

    manifest = {
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "single_manifest": str(args.single_manifest.expanduser().resolve()),
            "single_profile": str(args.single_profile.expanduser().resolve()),
            "single_uncertainty_table": str(
                args.single_uncertainty_table.expanduser().resolve()
            ),
            "powder_observations": str(args.powder_observations.expanduser().resolve()),
            "powder_manifest": str(args.powder_manifest.expanduser().resolve()),
            "single_window_input_root": str(
                args.single_window_input_root.expanduser().resolve()
            ),
            "single_window_profile": str(
                args.single_window_profile.expanduser().resolve()
            ),
            "powder_window_input_root": str(
                args.powder_window_input_root.expanduser().resolve()
            ),
            "powder_window_manifest": str(
                args.powder_window_manifest.expanduser().resolve()
            ),
            "powder_window_profile": str(
                args.powder_window_profile.expanduser().resolve()
            ),
        },
        "parameters": {
            "position_tolerance_deg": tolerance,
            "same_frame_peak_comparisons": False,
            "track_or_recurrence_filter": False,
            "anchor_peak_map_layout": (
                "one anchor peak per map; rows=all registered frames; "
                "columns=frame-local peak slots 1..dataset maximum"
            ),
            "missing_peak_slots": "NaN/white, never numerical zero",
            "location_formula": "radial 2theta: clip(1-abs(delta_2theta)/tolerance,0,1)",
            "area_formula": "min(integrated_area_a,integrated_area_b)/max(...)",
            "both_zero_area_similarity": 1.0,
            "window_presentation_triangle": "strict_lower",
            "window_diagonal_presented": False,
            "window_nominal_sequence": "0-5, 1-6, 2-7, ...",
            "window_width_deg": 5.0,
            "window_step_deg": 1.0,
            "window_results_regenerated": not args.no_window_copy,
            "window_workers": max(1, int(args.window_workers)),
            "full_symmetric_window_sources": "retained_under_window_full_symmetric_audit",
        },
        "validation": suite_validation,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2),
        encoding="utf-8",
    )

    artifact_rows = build_artifact_index(out_dir)
    write_csv(out_dir / "artifact_index.csv", artifact_rows)
    if suite_status == "PASS":
        completion = {
            "status": "complete",
            "all_validation_checks_passed": True,
            "validation_report_sha256": file_sha256(out_dir / "validation_report.json"),
            "run_manifest_sha256": file_sha256(out_dir / "run_manifest.json"),
            "artifact_index_sha256": file_sha256(out_dir / "artifact_index.csv"),
            "payload_artifacts_indexed": len(artifact_rows),
            "note": (
                "artifact_index.csv indexes the payload written before the index "
                "and completion marker; its own hash is recorded here"
            ),
        }
        temporary_marker = out_dir / ".RUN_COMPLETE.json.tmp"
        temporary_marker.write_text(
            json.dumps(json_ready(completion), sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        temporary_marker.replace(out_dir / "RUN_COMPLETE.json")
    print(
        json.dumps(
            {
                "status": suite_status,
                "out_dir": str(out_dir),
                "artifacts": len(artifact_rows),
                "peak_datasets": {
                    key: {
                        "frames": value["nonempty_frames"],
                        "peaks": value["peaks"],
                        "cells": value["written_cross_frame_peak_rows"],
                    }
                    for key, value in validations.items()
                },
                "window_roles": len(window_audit.get("roles_verified", [])),
                "completion_marker": suite_status == "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
