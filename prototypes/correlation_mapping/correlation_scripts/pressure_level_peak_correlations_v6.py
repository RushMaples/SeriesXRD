#!/usr/bin/env python3
"""Pressure-level powder-spot correlations with compact 2D ROI integration.

This program replaces the earlier 519-observation x 1060-frame presentation
with the pressure-level layout requested for the powder data:

* 519 raw spot observations are assigned exactly once to the existing 280
  pressure-level points (228 tracked rows plus all 52 untracked rows);
* every pressure-level point is an anchor, producing one location map and one
  2D ROI integrated-IoU map;
* rows are the 19 pressure levels in descending order and columns are the
  pressure-local peak numbers sorted by increasing 2theta (maximum 22);
* the anchor pressure row and structurally absent slots are NaN/white, while a
  valid pair with disjoint compact support is a real numerical zero;
* unchanged, independently audited integer-window across/within-frame results
  can be copied byte-for-byte from the completed v5 suite into the same run.

The formal ROI field uses every raw observation.  Each observation becomes a
normalized compact biweight ellipse in (q, periodic azimuth), weighted by its
fit-control-normalized integrated area.  Split blobs first sum within a
physical frame; distinct frames then receive equal weight.  Pair similarity is
the numerically integrated intersection-over-union of the two nonnegative
fields.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = WORKSPACE_ROOT / "correlations" / "UOTe XRD Data Refinement"
TRACK_ROOT = DATA_ROOT / "Powder Scan" / "Track Analysis"
REDUCED_ROOT = DATA_ROOT / "Powder Scan" / "Reduced .xy"

DEFAULT_OBSERVATIONS = TRACK_ROOT / "spot_observations.csv"
DEFAULT_TRACK_POINTS = TRACK_ROOT / "spot_track_points.csv"
DEFAULT_UNTRACKED_POINTS = TRACK_ROOT / "spot_untracked_points.csv"
DEFAULT_MANIFEST = REDUCED_ROOT / "manifest.csv"
DEFAULT_FIT_ROOT = REDUCED_ROOT / "fit_channel"
DEFAULT_WINDOW_SOURCE = (
    WORKSPACE_ROOT
    / "correlations"
    / "results"
    / "uote_all_peak_frame_slot_integer_window_suite_20260729_v5"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "correlations"
    / "results"
    / "uote_pressure_level_peak_ellipse_iou_integer_window_suite_20260729_v6"
)

POWDER_WAVELENGTH_A = 0.3066
POSITION_TOLERANCE_DEG = 0.06
UNTRACKED_Q_TOLERANCE = 0.06784655333637069
UNTRACKED_AZIM_TOLERANCE_DEG = 6.0
FIT_NORMALIZATION_MIN_DEG = 2.0
FIT_NORMALIZATION_MAX_DEG = 25.0
Q_GRID_MIN = 0.0
Q_GRID_STEP = 0.001
AZIM_GRID_STEP_DEG = 0.1
AZIM_GRID_COUNT = 3600
KERNEL_POWER = 2
PRESSURES_DESCENDING = (
    50.7,
    41.4,
    33.6,
    23.5,
    18.3,
    14.7,
    11.5,
    9.1,
    7.6,
    7.0,
    6.81,
    5.81,
    5.37,
    4.97,
    4.71,
    4.15,
    3.85,
    3.75,
    3.5,
)
WINDOW_PAYLOADS = (
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
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--track-points", type=Path, default=DEFAULT_TRACK_POINTS)
    parser.add_argument(
        "--untracked-points", type=Path, default=DEFAULT_UNTRACKED_POINTS
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fit-root", type=Path, default=DEFAULT_FIT_ROOT)
    parser.add_argument(
        "--reuse-window-suite", type=Path, default=DEFAULT_WINDOW_SOURCE
    )
    parser.add_argument("--no-windows", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=None,
        help="Smoke-test option; omit for the required complete 280-anchor run.",
    )
    parser.add_argument("--q-step", type=float, default=Q_GRID_STEP)
    parser.add_argument("--azim-step-deg", type=float, default=AZIM_GRID_STEP_DEG)
    return parser.parse_args()


def _progress(message: str) -> None:
    print(f"[pressure-peaks-v6] {message}", flush=True)


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
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})
            count += 1
    return count


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pressure_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def q_to_two_theta(q_a_inv: float | np.ndarray) -> np.ndarray:
    q = np.asarray(q_a_inv, dtype=float)
    argument = q * POWDER_WAVELENGTH_A / (4.0 * np.pi)
    result = np.full(q.shape, np.nan, dtype=float)
    valid = np.isfinite(argument) & (np.abs(argument) <= 1.0)
    result[valid] = np.degrees(2.0 * np.arcsin(argument[valid]))
    return result


def circular_delta_deg(
    left: float | np.ndarray,
    right: float | np.ndarray,
) -> np.ndarray:
    """Return the signed shortest angular difference ``left-right``."""
    return (np.asarray(left, dtype=float) - np.asarray(right, dtype=float) + 180.0) % 360.0 - 180.0


def location_similarity(
    left_two_theta: float | np.ndarray,
    right_two_theta: float | np.ndarray,
    tolerance: float = POSITION_TOLERANCE_DEG,
) -> np.ndarray:
    resolved = max(float(tolerance), np.finfo(float).eps)
    result = 1.0 - np.abs(
        np.asarray(left_two_theta, dtype=float)
        - np.asarray(right_two_theta, dtype=float)
    ) / resolved
    return np.clip(result, 0.0, 1.0)


def _rasterize_compact_ellipse(
    q_center: float,
    azim_center_deg: float,
    h_q: float,
    h_azim_deg: float,
    *,
    power: int,
    q_min: float = Q_GRID_MIN,
    q_step: float = Q_GRID_STEP,
    azim_step_deg: float = AZIM_GRID_STEP_DEG,
    n_azim: int = AZIM_GRID_COUNT,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rasterize a compact normalized ellipse on a periodic sparse grid."""
    for label, value in (
        ("q_center", q_center),
        ("azim_center_deg", azim_center_deg),
        ("h_q", h_q),
        ("h_azim_deg", h_azim_deg),
        ("q_step", q_step),
        ("azim_step_deg", azim_step_deg),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if h_q <= 0.0 or h_azim_deg <= 0.0:
        raise ValueError("ellipse semiaxes must be positive")
    if q_step <= 0.0 or azim_step_deg <= 0.0 or n_azim < 1:
        raise ValueError("invalid raster grid")
    if not np.isclose(n_azim * azim_step_deg, 360.0, atol=1.0e-9):
        raise ValueError("azimuth grid must cover exactly 360 degrees")
    if power < 1:
        raise ValueError("kernel power must be positive")

    q_first = max(0, int(math.ceil((q_center - h_q - q_min) / q_step)))
    q_last = int(math.floor((q_center + h_q - q_min) / q_step))
    q_indices = np.arange(q_first, q_last + 1, dtype=np.int64)
    q_coordinates = q_min + q_indices.astype(float) * q_step

    wrapped_center = float((azim_center_deg + 180.0) % 360.0 - 180.0)
    center_bin = int(round((wrapped_center + 180.0) / azim_step_deg)) % n_azim
    azim_radius = int(math.ceil(h_azim_deg / azim_step_deg)) + 1
    azim_indices = np.unique(
        (center_bin + np.arange(-azim_radius, azim_radius + 1)) % n_azim
    ).astype(np.int64)
    azim_coordinates = -180.0 + azim_indices.astype(float) * azim_step_deg
    azim_delta = circular_delta_deg(azim_coordinates, wrapped_center)

    r_squared = (
        ((q_coordinates[:, None] - q_center) / h_q) ** 2
        + (azim_delta[None, :] / h_azim_deg) ** 2
    )
    mask = r_squared < 1.0
    if not np.any(mask):
        raise ValueError("ellipse has no represented grid cells")
    q_mesh, azim_mesh = np.meshgrid(q_indices, azim_indices, indexing="ij")
    flat_indices = q_mesh[mask] * int(n_azim) + azim_mesh[mask]
    raw = np.power(1.0 - r_squared[mask], power)
    cell_area = float(q_step * azim_step_deg)
    density = raw / (float(np.sum(raw, dtype=float)) * cell_area)
    order = np.argsort(flat_indices)
    return (
        np.asarray(flat_indices[order], dtype=np.int64),
        np.asarray(density[order], dtype=float),
        cell_area,
    )


def rasterize_epanechnikov_ellipse(
    q_center: float,
    azim_center_deg: float,
    h_q: float,
    h_azim_deg: float,
    *,
    q_min: float = Q_GRID_MIN,
    q_step: float = Q_GRID_STEP,
    azim_step_deg: float = AZIM_GRID_STEP_DEG,
    n_azim: int = AZIM_GRID_COUNT,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Public p=1 compact-kernel contract retained for focused unit tests."""
    return _rasterize_compact_ellipse(
        q_center,
        azim_center_deg,
        h_q,
        h_azim_deg,
        power=1,
        q_min=q_min,
        q_step=q_step,
        azim_step_deg=azim_step_deg,
        n_azim=n_azim,
    )


def rasterize_biweight_ellipse(
    q_center: float,
    azim_center_deg: float,
    h_q: float,
    h_azim_deg: float,
    *,
    q_min: float = Q_GRID_MIN,
    q_step: float = Q_GRID_STEP,
    azim_step_deg: float = AZIM_GRID_STEP_DEG,
    n_azim: int = AZIM_GRID_COUNT,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rasterize the formal C1-smooth p=2 compact biweight ellipse."""
    return _rasterize_compact_ellipse(
        q_center,
        azim_center_deg,
        h_q,
        h_azim_deg,
        power=2,
        q_min=q_min,
        q_step=q_step,
        azim_step_deg=azim_step_deg,
        n_azim=n_azim,
    )


def _sum_sparse(
    sparse_items: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not sparse_items:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=float)
    all_indices = np.concatenate(
        [np.asarray(item[0], dtype=np.int64) for item in sparse_items]
    )
    all_values = np.concatenate(
        [np.asarray(item[1], dtype=float) for item in sparse_items]
    )
    unique, inverse = np.unique(all_indices, return_inverse=True)
    summed = np.zeros(unique.size, dtype=float)
    np.add.at(summed, inverse, all_values)
    keep = summed > 0.0
    return unique[keep], summed[keep]


def aggregate_frame_profiles(
    observation_profiles: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Correct scale per observation, sum within frame, mean unique frames."""
    if not observation_profiles:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=float)
    by_frame: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for item in observation_profiles:
        frame = as_int(item.get("frame"))
        scale = as_float(item.get("measurement_scale"))
        indices = np.asarray(item.get("indices"), dtype=np.int64)
        values = np.asarray(item.get("values"), dtype=float)
        if frame < 0 or not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"invalid frame/measurement scale: {frame}, {scale}")
        if indices.ndim != 1 or values.shape != indices.shape:
            raise ValueError("sparse observation arrays must be equal-shape 1D")
        if indices.size and not np.all(np.diff(indices) > 0):
            raise ValueError("sparse observation indices must be strictly sorted")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("sparse observation values must be finite/nonnegative")
        by_frame[frame].append((indices, values / scale))
    frame_profiles = [_sum_sparse(items) for items in by_frame.values()]
    indices, values = _sum_sparse(frame_profiles)
    values /= float(len(frame_profiles))
    return indices, values


def sparse_profile_iou(
    indices_a: np.ndarray,
    values_a: np.ndarray,
    indices_b: np.ndarray,
    values_b: np.ndarray,
    cell_area: float,
) -> float:
    """Numerically integrate min/max for two sparse nonnegative fields."""
    ia = np.asarray(indices_a, dtype=np.int64)
    ib = np.asarray(indices_b, dtype=np.int64)
    va = np.asarray(values_a, dtype=float)
    vb = np.asarray(values_b, dtype=float)
    if ia.shape != va.shape or ib.shape != vb.shape:
        raise ValueError("profile index/value shape mismatch")
    if not np.isfinite(cell_area) or cell_area <= 0.0:
        raise ValueError("cell_area must be positive")
    integral_a = float(np.sum(va, dtype=float) * cell_area)
    integral_b = float(np.sum(vb, dtype=float) * cell_area)
    if integral_a <= 0.0 or integral_b <= 0.0:
        return math.nan
    if ia.size == 0 or ib.size == 0 or ia[-1] < ib[0] or ib[-1] < ia[0]:
        return 0.0
    common, a_positions, b_positions = np.intersect1d(
        ia,
        ib,
        assume_unique=True,
        return_indices=True,
    )
    intersection = (
        float(np.sum(np.minimum(va[a_positions], vb[b_positions])) * cell_area)
        if common.size
        else 0.0
    )
    upper_bound = min(integral_a, integral_b)
    tolerance = 1.0e-10 * max(integral_a, integral_b, 1.0)
    if intersection < -tolerance or intersection > upper_bound + tolerance:
        raise RuntimeError(
            f"invalid integrated intersection {intersection}; upper={upper_bound}"
        )
    union = integral_a + integral_b - intersection
    if union <= 0.0:
        return math.nan
    score = intersection / union
    if score < -1.0e-12 or score > 1.0 + 1.0e-12:
        raise RuntimeError(f"ROI IoU outside [0,1]: {score}")
    return float(min(1.0, max(0.0, score)))


@dataclass(frozen=True)
class SparsePointProfile:
    indices: np.ndarray
    values: np.ndarray
    sensitivity_values: np.ndarray
    cell_area: float

    @property
    def integral(self) -> float:
        return float(np.sum(self.values, dtype=float) * self.cell_area)

    @property
    def sensitivity_integral(self) -> float:
        return float(
            np.sum(self.sensitivity_values, dtype=float) * self.cell_area
        )


def _build_target_points(
    track_points_path: Path,
    untracked_points_path: Path,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for source_table, path in (
        ("spot_track_points.csv", track_points_path),
        ("spot_untracked_points.csv", untracked_points_path),
    ):
        for source_row, raw in enumerate(read_csv(path)):
            pressure = as_float(raw.get("pressure_gpa"))
            track = as_int(raw.get("track"))
            if source_table == "spot_track_points.csv":
                uid = f"T{track:02d}_P{pressure_token(pressure)}"
            else:
                uid = f"U{source_row:02d}_P{pressure_token(pressure)}"
            point = dict(raw)
            point.update(
                {
                    "point_uid": uid,
                    "source_table": source_table,
                    "source_row_0based": source_row,
                    "track": track,
                    "pressure_gpa": pressure,
                    "q": as_float(raw.get("q")),
                    "azim_deg": as_float(raw.get("azim_deg")),
                    "member_obs_indices": [],
                }
            )
            points.append(point)
    if len(points) != 280:
        raise ValueError(f"expected 280 target points, found {len(points)}")
    if len({str(point["point_uid"]) for point in points}) != len(points):
        raise ValueError("pressure-level point_uid values are not unique")
    return points


def assign_observations_to_points(
    observations_path: Path,
    track_points_path: Path,
    untracked_points_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Assign every raw observation to one authoritative pressure-level row."""
    observations: list[dict[str, Any]] = []
    for obs_index, raw in enumerate(read_csv(observations_path)):
        item = dict(raw)
        item.update(
            {
                "obs_index_0based": obs_index,
                "obs_csv_line_1based": obs_index + 2,
                "frame": as_int(raw.get("frame")),
                "track": as_int(raw.get("track")),
                "pressure_gpa": as_float(raw.get("pressure_gpa")),
                "q": as_float(raw.get("q")),
                "azim_deg": as_float(raw.get("azim_deg")),
                "intensity": as_float(raw.get("intensity")),
                "area": as_float(raw.get("area")),
                "q_width": as_float(raw.get("q_width")),
                "azim_width_deg": as_float(raw.get("azim_width_deg")),
            }
        )
        observations.append(item)
    if len(observations) != 519:
        raise ValueError(f"expected 519 raw observations, found {len(observations)}")

    points = _build_target_points(track_points_path, untracked_points_path)
    tracked_lookup: dict[tuple[int, float], int] = {}
    untracked_by_pressure: dict[float, list[int]] = defaultdict(list)
    for point_index, point in enumerate(points):
        pressure = float(point["pressure_gpa"])
        if str(point["source_table"]) == "spot_track_points.csv":
            key = (int(point["track"]), pressure)
            if key in tracked_lookup:
                raise ValueError(f"duplicate tracked target key: {key}")
            tracked_lookup[key] = point_index
        else:
            untracked_by_pressure[pressure].append(point_index)

    assignments: list[dict[str, Any]] = []
    for observation in observations:
        track = int(observation["track"])
        pressure = float(observation["pressure_gpa"])
        assignment_distance = 0.0
        candidate_count = 1
        if track >= 0:
            key = (track, pressure)
            if key not in tracked_lookup:
                raise ValueError(f"tracked observation has no target: {key}")
            point_index = tracked_lookup[key]
        else:
            candidates: list[tuple[float, int]] = []
            for candidate_index in untracked_by_pressure.get(pressure, []):
                candidate = points[candidate_index]
                dq = abs(float(observation["q"]) - float(candidate["q"]))
                dazim = abs(
                    float(
                        circular_delta_deg(
                            float(observation["azim_deg"]),
                            float(candidate["azim_deg"]),
                        )
                    )
                )
                if (
                    dq <= UNTRACKED_Q_TOLERANCE
                    and dazim <= UNTRACKED_AZIM_TOLERANCE_DEG
                ):
                    distance = (
                        (dq / UNTRACKED_Q_TOLERANCE) ** 2
                        + (dazim / UNTRACKED_AZIM_TOLERANCE_DEG) ** 2
                    )
                    candidates.append((distance, candidate_index))
            if not candidates:
                raise ValueError(
                    "untracked observation has no target: "
                    f"obs={observation['obs_index_0based']} p={pressure:g}"
                )
            candidates.sort(
                key=lambda item: (
                    item[0],
                    int(points[item[1]]["source_row_0based"]),
                )
            )
            assignment_distance, point_index = candidates[0]
            candidate_count = len(candidates)

        point = points[point_index]
        member_rank = len(point["member_obs_indices"]) + 1
        point["member_obs_indices"].append(int(observation["obs_index_0based"]))
        observation["point_index"] = point_index
        observation["point_uid"] = str(point["point_uid"])
        assignments.append(
            {
                "point_uid": point["point_uid"],
                "source_table": point["source_table"],
                "source_row_0based": point["source_row_0based"],
                "track": point["track"],
                "pressure_gpa": pressure,
                "member_rank": member_rank,
                "obs_index_0based": observation["obs_index_0based"],
                "obs_csv_line_1based": observation["obs_csv_line_1based"],
                "frame": observation["frame"],
                "scan": observation.get("scan", ""),
                "q": observation["q"],
                "azim_deg": observation["azim_deg"],
                "intensity": observation["intensity"],
                "area": observation["area"],
                "assignment_distance": assignment_distance,
                "eligible_untracked_target_count": candidate_count,
            }
        )

    uncovered = [
        str(point["point_uid"])
        for point in points
        if not point["member_obs_indices"]
    ]
    if uncovered:
        raise ValueError(f"pressure-level targets without observations: {uncovered}")
    if len(assignments) != len(observations):
        raise RuntimeError("not every observation was assigned exactly once")

    max_errors = {
        "area": 0.0,
        "intensity": 0.0,
        "q": 0.0,
        "azim_deg": 0.0,
    }
    frame_contributions = 0
    same_frame_extra_blobs = 0
    for point in points:
        members = [
            observations[index] for index in point["member_obs_indices"]
        ]
        frames = {int(member["frame"]) for member in members}
        frame_contributions += len(frames)
        same_frame_extra_blobs += len(members) - len(frames)
        weights = np.asarray(
            [float(member["intensity"]) for member in members], dtype=float
        )
        q_values = np.asarray([float(member["q"]) for member in members])
        q_rebuilt = float(np.average(q_values, weights=weights))
        radians = np.radians(
            [float(member["azim_deg"]) for member in members]
        )
        az_rebuilt = math.degrees(
            math.atan2(
                float(np.sum(weights * np.sin(radians))),
                float(np.sum(weights * np.cos(radians))),
            )
        )
        area_rebuilt = float(sum(float(member["area"]) for member in members))
        intensity_rebuilt = max(float(member["intensity"]) for member in members)
        max_errors["area"] = max(
            max_errors["area"],
            abs(area_rebuilt - as_float(point.get("area"))),
        )
        max_errors["intensity"] = max(
            max_errors["intensity"],
            abs(intensity_rebuilt - as_float(point.get("intensity"))),
        )
        max_errors["q"] = max(
            max_errors["q"],
            abs(q_rebuilt - float(point["q"])),
        )
        max_errors["azim_deg"] = max(
            max_errors["azim_deg"],
            abs(float(circular_delta_deg(az_rebuilt, float(point["azim_deg"])))),
        )
        if len(frames) != as_int(point.get("n_frames")):
            raise RuntimeError(
                f"n_frames mismatch for {point['point_uid']}: "
                f"{len(frames)} != {point.get('n_frames')}"
            )
        point["n_observations"] = len(members)
        point["distinct_frames"] = len(frames)
        point["obs_indices_0based"] = "|".join(
            str(index) for index in point["member_obs_indices"]
        )
        point["frames"] = "|".join(str(value) for value in sorted(frames))

    mapping_validation = {
        "observations": len(observations),
        "tracked_observations": sum(
            int(observation["track"]) >= 0 for observation in observations
        ),
        "untracked_observations": sum(
            int(observation["track"]) == -1 for observation in observations
        ),
        "pressure_level_points": len(points),
        "tracked_points": sum(
            point["source_table"] == "spot_track_points.csv" for point in points
        ),
        "untracked_points": sum(
            point["source_table"] == "spot_untracked_points.csv" for point in points
        ),
        "distinct_frame_contributions": frame_contributions,
        "same_frame_extra_blobs": same_frame_extra_blobs,
        "all_observations_assigned_once": len(assignments) == 519,
        "all_targets_covered": not uncovered,
        "max_summary_reconstruction_errors": max_errors,
        "summary_reconstruction_within_export_rounding": (
            max_errors["area"] <= 0.51
            and max_errors["intensity"] <= 0.11
            and max_errors["q"] <= 4.0e-5
            and max_errors["azim_deg"] <= 0.01
        ),
        "untracked_q_tolerance": UNTRACKED_Q_TOLERANCE,
        "untracked_periodic_azim_tolerance_deg": UNTRACKED_AZIM_TOLERANCE_DEG,
    }
    if (
        frame_contributions != 509
        or same_frame_extra_blobs != 10
        or not mapping_validation["summary_reconstruction_within_export_rounding"]
    ):
        raise RuntimeError(f"519-to-280 mapping validation failed: {mapping_validation}")

    by_pressure: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_pressure[float(point["pressure_gpa"])].append(point)
    if set(by_pressure) != set(PRESSURES_DESCENDING):
        raise ValueError(
            f"unexpected pressure ladder: {sorted(by_pressure)}"
        )
    for pressure, pressure_points in by_pressure.items():
        pressure_points.sort(
            key=lambda item: (
                float(item["q"]),
                float(item["azim_deg"]),
                str(item["point_uid"]),
            )
        )
        for local_index, point in enumerate(pressure_points, start=1):
            point["local_peak_index"] = local_index
            point["two_theta_deg"] = float(q_to_two_theta(float(point["q"])))
    for point_index, point in enumerate(points):
        point["point_index"] = point_index
    return observations, points, assignments, mapping_validation


_FRAME_PATTERN = re.compile(r"frame_(\d+)_")


def _xy_positive_integral(path: Path, lower: float, upper: float) -> float:
    data = np.loadtxt(path, comments="#", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"invalid XY file: {path}")
    x = data[:, 0]
    y = data[:, 1]
    mask = np.isfinite(x) & np.isfinite(y) & (x >= lower) & (x <= upper)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"insufficient fit-control coverage: {path}")
    return float(np.trapezoid(np.maximum(y[mask], 0.0), x[mask]))


def build_measurement_normalization(
    manifest_path: Path,
    fit_root: Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    """Build frame-level tungsten fit-control normalization and D1w audit."""
    fit_by_frame: dict[int, Path] = {}
    for path in fit_root.rglob("frame_*.xy"):
        match = _FRAME_PATTERN.search(path.name)
        if match:
            frame = int(match.group(1))
            if frame in fit_by_frame:
                raise ValueError(f"duplicate fit-channel file for frame {frame}")
            fit_by_frame[frame] = path

    rows: list[dict[str, Any]] = []
    for raw in read_csv(manifest_path):
        if as_int(raw.get("cover_excluded")) != 0:
            continue
        frame = as_int(raw.get("frame"))
        if frame not in fit_by_frame:
            raise FileNotFoundError(f"missing fit-channel profile for frame {frame}")
        filename = str(raw.get("filename", ""))
        exposure_token = (
            "D1w" if "_D1w_" in filename else "D1s" if "_D1s_" in filename else "unknown"
        )
        fit_integral = _xy_positive_integral(
            fit_by_frame[frame],
            FIT_NORMALIZATION_MIN_DEG,
            FIT_NORMALIZATION_MAX_DEG,
        )
        rows.append(
            {
                "frame": frame,
                "scan": raw.get("scan", ""),
                "pressure_gpa": as_float(raw.get("pressure_GPa")),
                "exposure_token": exposure_token,
                "fit_positive_integral_2theta_2_25": fit_integral,
                "fit_channel_file": str(fit_by_frame[frame].resolve()),
                "source_filename": filename,
            }
        )
    if len(rows) != 1060:
        raise ValueError(f"expected 1060 accepted frames, found {len(rows)}")
    d1s_integrals = np.asarray(
        [
            float(row["fit_positive_integral_2theta_2_25"])
            for row in rows
            if row["exposure_token"] == "D1s"
        ],
        dtype=float,
    )
    if d1s_integrals.size == 0:
        raise ValueError("no D1s fit-control frames found")
    reference = float(np.median(d1s_integrals))

    by_scan_pressure: dict[tuple[str, float], dict[str, Any]] = {}
    for row in rows:
        by_scan_pressure[(str(row["scan"]), float(row["pressure_gpa"]))] = row
    d1w_pair_rows: list[dict[str, Any]] = []
    for scan in sorted({str(row["scan"]) for row in rows}):
        left = by_scan_pressure.get((scan, 3.5))
        right = by_scan_pressure.get((scan, 3.75))
        if left is None or right is None:
            continue
        if left["exposure_token"] != "D1w" or right["exposure_token"] != "D1s":
            continue
        ratio = float(left["fit_positive_integral_2theta_2_25"]) / float(
            right["fit_positive_integral_2theta_2_25"]
        )
        d1w_pair_rows.append(
            {
                "scan": scan,
                "frame_3p5_D1w": left["frame"],
                "frame_3p75_D1s": right["frame"],
                "D1w_over_D1s_fit_integral_ratio": ratio,
            }
        )
    ratios = np.asarray(
        [float(row["D1w_over_D1s_fit_integral_ratio"]) for row in d1w_pair_rows]
    )
    if ratios.size < 40:
        raise RuntimeError("insufficient paired D1w/D1s evidence")
    fixed_d1w_scale = float(np.median(ratios))

    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        fit_integral = float(row["fit_positive_integral_2theta_2_25"])
        row["reference_D1s_fit_integral"] = reference
        row["main_measurement_scale"] = fit_integral / reference
        row["main_area_multiplier"] = reference / fit_integral
        row["fixed_token_sensitivity_scale"] = (
            fixed_d1w_scale if row["exposure_token"] == "D1w" else 1.0
        )
        row["fixed_token_sensitivity_multiplier"] = 1.0 / float(
            row["fixed_token_sensitivity_scale"]
        )
        by_frame[int(row["frame"])] = row

    assessment = {
        "method": "frame-level internal-standard normalization",
        "fit_control_role": (
            "tungsten-dominated internal measurement reference; not an external "
            "beam monitor and not interpreted as sample signal"
        ),
        "integration_2theta_deg": [
            FIT_NORMALIZATION_MIN_DEG,
            FIT_NORMALIZATION_MAX_DEG,
        ],
        "positive_signal_only": True,
        "accepted_frames": len(rows),
        "D1s_reference_median": reference,
        "paired_3p5_D1w_to_3p75_D1s_scans": int(ratios.size),
        "D1w_over_D1s_ratio_median": fixed_d1w_scale,
        "D1w_over_D1s_ratio_q25": float(np.quantile(ratios, 0.25)),
        "D1w_over_D1s_ratio_q75": float(np.quantile(ratios, 0.75)),
        "interpretation": (
            "D1w has an empirical measurement scale near five times D1s; this "
            "is not asserted to be a literal exposure duration."
        ),
        "formal_multiplier": "normalized_area = raw_area * M_ref / M_frame",
        "sensitivity_model": (
            f"D1s scale=1; every D1w scale=paired median {fixed_d1w_scale:.8g}"
        ),
        "pair_rows": d1w_pair_rows,
    }
    return rows, by_frame, assessment


def build_point_profiles(
    observations: Sequence[Mapping[str, Any]],
    points: Sequence[dict[str, Any]],
    normalization_by_frame: Mapping[int, Mapping[str, Any]],
    *,
    q_step: float,
    azim_step_deg: float,
) -> tuple[dict[str, SparsePointProfile], list[dict[str, Any]], dict[str, Any]]:
    """Construct formal main and fixed-D1w sensitivity fields."""
    n_azim_float = 360.0 / float(azim_step_deg)
    n_azim = int(round(n_azim_float))
    if not np.isclose(n_azim_float, n_azim, atol=1.0e-10):
        raise ValueError("azimuth step must divide 360 degrees")

    observation_profiles: dict[int, dict[str, Any]] = {}
    kernel_audit_rows: list[dict[str, Any]] = []
    maximum_kernel_integral_error = 0.0
    for observation in observations:
        frame = int(observation["frame"])
        normalization = normalization_by_frame.get(frame)
        if normalization is None:
            raise ValueError(f"observation frame lacks normalization: {frame}")
        h_q = 2.0 * float(observation["q_width"])
        h_azim = 2.0 * float(observation["azim_width_deg"])
        indices, density, cell_area = rasterize_biweight_ellipse(
            float(observation["q"]),
            float(observation["azim_deg"]),
            h_q,
            h_azim,
            q_min=Q_GRID_MIN,
            q_step=q_step,
            azim_step_deg=azim_step_deg,
            n_azim=n_azim,
        )
        integral = float(np.sum(density, dtype=float) * cell_area)
        maximum_kernel_integral_error = max(
            maximum_kernel_integral_error, abs(integral - 1.0)
        )
        measured_values = density * float(observation["area"])
        obs_index = int(observation["obs_index_0based"])
        observation_profiles[obs_index] = {
            "frame": frame,
            "indices": indices,
            "values": measured_values,
            "main_measurement_scale": float(
                normalization["main_measurement_scale"]
            ),
            "sensitivity_measurement_scale": float(
                normalization["fixed_token_sensitivity_scale"]
            ),
            "cell_area": cell_area,
        }
        kernel_audit_rows.append(
            {
                "obs_index_0based": obs_index,
                "point_uid": observation["point_uid"],
                "frame": frame,
                "q_center": observation["q"],
                "azim_center_deg": observation["azim_deg"],
                "q_width": observation["q_width"],
                "azim_width_deg": observation["azim_width_deg"],
                "semiaxis_q_2x_width": h_q,
                "semiaxis_azim_deg_2x_width": h_azim,
                "kernel_power": KERNEL_POWER,
                "grid_cells": indices.size,
                "discrete_kernel_integral": integral,
                "raw_area": observation["area"],
                "main_measurement_scale": normalization[
                    "main_measurement_scale"
                ],
                "fixed_token_sensitivity_scale": normalization[
                    "fixed_token_sensitivity_scale"
                ],
            }
        )

    profiles: dict[str, SparsePointProfile] = {}
    point_profile_rows: list[dict[str, Any]] = []
    for point in points:
        member_profiles = [
            observation_profiles[int(index)]
            for index in point["member_obs_indices"]
        ]
        main_items = [
            {
                "frame": item["frame"],
                "measurement_scale": item["main_measurement_scale"],
                "indices": item["indices"],
                "values": item["values"],
            }
            for item in member_profiles
        ]
        sensitivity_items = [
            {
                "frame": item["frame"],
                "measurement_scale": item[
                    "sensitivity_measurement_scale"
                ],
                "indices": item["indices"],
                "values": item["values"],
            }
            for item in member_profiles
        ]
        main_indices, main_values = aggregate_frame_profiles(main_items)
        sensitivity_indices, sensitivity_values = aggregate_frame_profiles(
            sensitivity_items
        )
        if not np.array_equal(main_indices, sensitivity_indices):
            raise RuntimeError(
                f"normalization changed sparse support for {point['point_uid']}"
            )
        cell_areas = {float(item["cell_area"]) for item in member_profiles}
        if len(cell_areas) != 1:
            raise RuntimeError("inconsistent raster cell areas")
        profile = SparsePointProfile(
            indices=main_indices,
            values=main_values,
            sensitivity_values=sensitivity_values,
            cell_area=cell_areas.pop(),
        )
        if not np.isfinite(profile.integral) or profile.integral <= 0.0:
            raise RuntimeError(f"invalid point integral: {point['point_uid']}")
        profiles[str(point["point_uid"])] = profile
        point["profile_integral_main"] = profile.integral
        point["profile_integral_fixed_D1w_sensitivity"] = (
            profile.sensitivity_integral
        )
        point["profile_grid_cells"] = main_indices.size
        point_profile_rows.append(
            {
                "point_uid": point["point_uid"],
                "pressure_gpa": point["pressure_gpa"],
                "local_peak_index": point["local_peak_index"],
                "n_observations": point["n_observations"],
                "distinct_frames": point["distinct_frames"],
                "sparse_grid_cells": main_indices.size,
                "main_intrinsic_area": profile.integral,
                "fixed_D1w_sensitivity_intrinsic_area": (
                    profile.sensitivity_integral
                ),
                "main_over_sensitivity_area_ratio": (
                    profile.integral / profile.sensitivity_integral
                ),
            }
        )
    if len(profiles) != len(points):
        raise RuntimeError(
            f"point/profile count mismatch: {len(points)} != {len(profiles)}"
        )
    audit = {
        "kernel": "compact C1 biweight ellipse",
        "kernel_formula": (
            "K=3/(pi*a*b)*(1-r^2)^2 for r^2<1; discrete quadrature "
            "renormalized exactly on the configured grid"
        ),
        "kernel_power": KERNEL_POWER,
        "semiaxis_q": "2*q_width",
        "semiaxis_azim": "2*azim_width_deg",
        "periodic_azimuth": True,
        "q_grid_min": Q_GRID_MIN,
        "q_grid_step": q_step,
        "azim_grid_step_deg": azim_step_deg,
        "azim_grid_count": n_azim,
        "cell_area_q_times_degree": q_step * azim_step_deg,
        "maximum_discrete_kernel_integral_abs_error": (
            maximum_kernel_integral_error
        ),
        "observations_rasterized": len(observation_profiles),
        "point_profiles": len(profiles),
        "aggregation_order": (
            "measurement-normalize each observation; sum split blobs within "
            "physical frame; arithmetic mean across unique frames"
        ),
    }
    return profiles, kernel_audit_rows, audit


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
    "roi_integrated_iou",
    "roi_fixed_D1w_sensitivity_iou",
    "roi_sensitivity_abs_difference",
    "left_intrinsic_area",
    "right_intrinsic_area",
    "left_fixed_D1w_sensitivity_area",
    "right_fixed_D1w_sensitivity_area",
)


def compute_cross_pressure_pairs(
    points: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, SparsePointProfile],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
]:
    count = len(points)
    location_matrix = np.full((count, count), np.nan, dtype=float)
    roi_matrix = np.full((count, count), np.nan, dtype=float)
    sensitivity_matrix = np.full((count, count), np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    zero_roi_pairs = 0
    sensitivity_differences: list[float] = []
    for left_index in range(count):
        left = points[left_index]
        left_profile = profiles[str(left["point_uid"])]
        for right_index in range(left_index + 1, count):
            right = points[right_index]
            if float(left["pressure_gpa"]) == float(right["pressure_gpa"]):
                continue
            right_profile = profiles[str(right["point_uid"])]
            location = float(
                location_similarity(
                    float(left["two_theta_deg"]),
                    float(right["two_theta_deg"]),
                )
            )
            roi = sparse_profile_iou(
                left_profile.indices,
                left_profile.values,
                right_profile.indices,
                right_profile.values,
                left_profile.cell_area,
            )
            sensitivity = sparse_profile_iou(
                left_profile.indices,
                left_profile.sensitivity_values,
                right_profile.indices,
                right_profile.sensitivity_values,
                left_profile.cell_area,
            )
            if not np.isfinite(roi) or not np.isfinite(sensitivity):
                raise RuntimeError(
                    f"valid cross-pressure pair produced NaN: "
                    f"{left['point_uid']}, {right['point_uid']}"
                )
            difference = abs(roi - sensitivity)
            sensitivity_differences.append(difference)
            zero_roi_pairs += int(roi == 0.0)
            for matrix, value in (
                (location_matrix, location),
                (roi_matrix, roi),
                (sensitivity_matrix, sensitivity),
            ):
                matrix[left_index, right_index] = value
                matrix[right_index, left_index] = value
            rows.append(
                {
                    "left_point_uid": left["point_uid"],
                    "right_point_uid": right["point_uid"],
                    "left_pressure_gpa": left["pressure_gpa"],
                    "right_pressure_gpa": right["pressure_gpa"],
                    "left_local_peak_index": left["local_peak_index"],
                    "right_local_peak_index": right["local_peak_index"],
                    "left_q": left["q"],
                    "right_q": right["q"],
                    "left_two_theta_deg": left["two_theta_deg"],
                    "right_two_theta_deg": right["two_theta_deg"],
                    "location_similarity": location,
                    "roi_integrated_iou": roi,
                    "roi_fixed_D1w_sensitivity_iou": sensitivity,
                    "roi_sensitivity_abs_difference": difference,
                    "left_intrinsic_area": left_profile.integral,
                    "right_intrinsic_area": right_profile.integral,
                    "left_fixed_D1w_sensitivity_area": (
                        left_profile.sensitivity_integral
                    ),
                    "right_fixed_D1w_sensitivity_area": (
                        right_profile.sensitivity_integral
                    ),
                }
            )
    differences = np.asarray(sensitivity_differences, dtype=float)
    audit = {
        "cross_pressure_unordered_pairs": len(rows),
        "expected_cross_pressure_unordered_pairs": 37038,
        "same_pressure_pairs_omitted": int(
            count * (count - 1) // 2 - len(rows)
        ),
        "ROI_pairs_exactly_zero_disjoint_support": zero_roi_pairs,
        "location_scores_in_0_1": bool(
            np.all(
                (location_matrix[np.isfinite(location_matrix)] >= 0.0)
                & (location_matrix[np.isfinite(location_matrix)] <= 1.0)
            )
        ),
        "ROI_scores_in_0_1": bool(
            np.all(
                (roi_matrix[np.isfinite(roi_matrix)] >= 0.0)
                & (roi_matrix[np.isfinite(roi_matrix)] <= 1.0)
            )
        ),
        "pair_matrices_symmetric": bool(
            np.allclose(
                location_matrix,
                location_matrix.T,
                equal_nan=True,
            )
            and np.allclose(roi_matrix, roi_matrix.T, equal_nan=True)
            and np.allclose(
                sensitivity_matrix,
                sensitivity_matrix.T,
                equal_nan=True,
            )
        ),
        "fixed_D1w_sensitivity_abs_difference_median": (
            float(np.median(differences))
        ),
        "fixed_D1w_sensitivity_abs_difference_q95": (
            float(np.quantile(differences, 0.95))
        ),
        "fixed_D1w_sensitivity_abs_difference_max": float(np.max(differences)),
    }
    if (
        len(rows) != 37038
        or not audit["location_scores_in_0_1"]
        or not audit["ROI_scores_in_0_1"]
        or not audit["pair_matrices_symmetric"]
    ):
        raise RuntimeError(f"pair validation failed: {audit}")
    return location_matrix, roi_matrix, sensitivity_matrix, rows, audit


def build_pressure_slot_layout(
    points: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int]], int]:
    by_pressure: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for point in points:
        by_pressure[float(point["pressure_gpa"])].append(point)
    maximum = max(len(rows) for rows in by_pressure.values())
    layout: list[dict[str, Any]] = []
    lookup: dict[str, tuple[int, int]] = {}
    for row_index, pressure in enumerate(PRESSURES_DESCENDING):
        pressure_points = sorted(
            by_pressure[pressure], key=lambda item: int(item["local_peak_index"])
        )
        layout.append(
            {
                "row_index_0based": row_index,
                "pressure_gpa": pressure,
                "peak_count": len(pressure_points),
                "max_peak_slots": maximum,
            }
        )
        for point in pressure_points:
            column = int(point["local_peak_index"]) - 1
            lookup[str(point["point_uid"])] = (row_index, column)
    if maximum != 22 or len(lookup) != 280:
        raise RuntimeError(
            f"unexpected pressure-slot layout: max={maximum}, points={len(lookup)}"
        )
    return layout, lookup, maximum


def build_anchor_matrix(
    anchor_index: int,
    points: Sequence[Mapping[str, Any]],
    pair_matrix: np.ndarray,
    slot_lookup: Mapping[str, tuple[int, int]],
    maximum_slots: int,
) -> np.ndarray:
    anchor = points[anchor_index]
    result = np.full(
        (len(PRESSURES_DESCENDING), maximum_slots), np.nan, dtype=float
    )
    for target_index, target in enumerate(points):
        if float(target["pressure_gpa"]) == float(anchor["pressure_gpa"]):
            continue
        row, column = slot_lookup[str(target["point_uid"])]
        result[row, column] = pair_matrix[anchor_index, target_index]
    return result


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    pressure_layout: Sequence[Mapping[str, Any]],
) -> None:
    columns = [f"peak {index}" for index in range(1, matrix.shape[1] + 1)]
    rows: list[dict[str, Any]] = []
    for row_index, layout in enumerate(pressure_layout):
        row: dict[str, Any] = {
            "pressure_gpa": layout["pressure_gpa"],
            "peak_count_at_pressure": layout["peak_count"],
        }
        row.update(
            {
                label: matrix[row_index, column_index]
                for column_index, label in enumerate(columns)
            }
        )
        rows.append(row)
    write_csv(
        path,
        rows,
        ["pressure_gpa", "peak_count_at_pressure", *columns],
    )


def plot_anchor_heatmap(
    path: Path,
    matrix: np.ndarray,
    pressure_layout: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
    *,
    metric: str,
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
    anchor_row = PRESSURES_DESCENDING.index(anchor_pressure)
    metric_title = (
        "location similarity"
        if metric == "location"
        else "2D ROI continuous integrated IoU"
    )
    title = (
        "Powder pressure-level peak comparison — "
        f"{metric_title}\n"
        f"anchor {anchor['point_uid']} | {anchor_pressure:g} GPa, "
        f"peak {int(anchor['local_peak_index'])}, "
        f"2θ={float(anchor['two_theta_deg']):.4f}°, "
        f"{int(anchor['n_observations'])} observations / "
        f"{int(anchor['distinct_frames'])} frames"
    )
    ax.set_title(title, fontsize=11)
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
    label = (
        "location similarity"
        if metric == "location"
        else "ROI integrated-overlap similarity"
    )
    fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.025,
        label=label,
    )
    fig.text(
        0.5,
        0.012,
        (
            "White = structurally missing peak slot or intentionally omitted "
            "anchor-pressure row; dark purple = a valid numerical similarity "
            "near 0. Each cell compares the anchor with that pressure/peak slot."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_anchor_maps(
    output_root: Path,
    points: Sequence[Mapping[str, Any]],
    location_matrix: np.ndarray,
    roi_matrix: np.ndarray,
    pressure_layout: Sequence[Mapping[str, Any]],
    slot_lookup: Mapping[str, tuple[int, int]],
    maximum_slots: int,
    *,
    make_plots: bool,
    max_anchors: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limit = len(points) if max_anchors is None else min(max_anchors, len(points))
    index_rows: list[dict[str, Any]] = []
    finite_counts: list[int] = []
    for anchor_index, anchor in enumerate(points[:limit]):
        token = (
            f"anchor_{anchor_index:03d}_"
            f"P{pressure_token(float(anchor['pressure_gpa']))}_"
            f"peak{int(anchor['local_peak_index']):02d}_"
            f"{anchor['point_uid']}"
        )
        location = build_anchor_matrix(
            anchor_index,
            points,
            location_matrix,
            slot_lookup,
            maximum_slots,
        )
        roi = build_anchor_matrix(
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
        if (
            np.count_nonzero(np.isfinite(location)) != expected_finite
            or np.count_nonzero(np.isfinite(roi)) != expected_finite
        ):
            raise RuntimeError(f"anchor finite-cell mismatch: {anchor['point_uid']}")
        anchor_row = PRESSURES_DESCENDING.index(float(anchor["pressure_gpa"]))
        if np.any(np.isfinite(location[anchor_row])) or np.any(
            np.isfinite(roi[anchor_row])
        ):
            raise RuntimeError(f"anchor pressure row not blank: {anchor['point_uid']}")
        finite_counts.append(expected_finite)

        location_csv = (
            output_root / "peak_maps" / "location" / "matrices" / f"{token}.csv"
        )
        roi_csv = (
            output_root
            / "peak_maps"
            / "roi_integrated_iou"
            / "matrices"
            / f"{token}.csv"
        )
        location_png = (
            output_root / "peak_maps" / "location" / "heatmaps" / f"{token}.png"
        )
        roi_png = (
            output_root
            / "peak_maps"
            / "roi_integrated_iou"
            / "heatmaps"
            / f"{token}.png"
        )
        write_matrix_csv(location_csv, location, pressure_layout)
        write_matrix_csv(roi_csv, roi, pressure_layout)
        if make_plots:
            plot_anchor_heatmap(
                location_png,
                location,
                pressure_layout,
                anchor,
                metric="location",
            )
            plot_anchor_heatmap(
                roi_png,
                roi,
                pressure_layout,
                anchor,
                metric="roi",
            )
        index_rows.append(
            {
                "anchor_index_0based": anchor_index,
                "point_uid": anchor["point_uid"],
                "pressure_gpa": anchor["pressure_gpa"],
                "local_peak_index": anchor["local_peak_index"],
                "q": anchor["q"],
                "two_theta_deg": anchor["two_theta_deg"],
                "source_table": anchor["source_table"],
                "track": anchor["track"],
                "n_observations": anchor["n_observations"],
                "distinct_frames": anchor["distinct_frames"],
                "finite_target_cells": expected_finite,
                "location_matrix_csv": str(location_csv.relative_to(output_root)),
                "location_heatmap_png": (
                    str(location_png.relative_to(output_root)) if make_plots else ""
                ),
                "roi_matrix_csv": str(roi_csv.relative_to(output_root)),
                "roi_heatmap_png": (
                    str(roi_png.relative_to(output_root)) if make_plots else ""
                ),
            }
        )
        if (anchor_index + 1) % 25 == 0 or anchor_index + 1 == limit:
            _progress(f"wrote anchor maps {anchor_index + 1}/{limit}")
    audit = {
        "requested_complete_anchors": len(points),
        "anchors_written": limit,
        "complete_anchor_run": limit == len(points),
        "map_shape": [len(PRESSURES_DESCENDING), maximum_slots],
        "location_matrix_csv_files": limit,
        "ROI_matrix_csv_files": limit,
        "location_heatmap_png_files": limit if make_plots else 0,
        "ROI_heatmap_png_files": limit if make_plots else 0,
        "pressure_rows_descending": list(PRESSURES_DESCENDING),
        "maximum_local_peak_slots": maximum_slots,
        "same_pressure_entire_row_blank": True,
        "structurally_missing_slots_NaN": True,
        "real_disjoint_ROI_is_numeric_zero": True,
        "finite_target_cells_min": min(finite_counts) if finite_counts else 0,
        "finite_target_cells_max": max(finite_counts) if finite_counts else 0,
    }
    return index_rows, audit


def _payload_inventory(
    root: Path,
    relative_payloads: Sequence[Path],
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    paths: list[Path] = []
    for relative in relative_payloads:
        source = root / relative
        if not source.exists():
            raise FileNotFoundError(f"window payload missing: {source}")
        if source.is_dir():
            paths.extend(
                path
                for path in source.rglob("*")
                if path.is_file() and path.name != ".DS_Store"
            )
        else:
            paths.append(source)
    for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
        relative = str(path.relative_to(root))
        size = path.stat().st_size
        sha256 = file_sha256(path)
        rows.append(
            {"relative_path": relative, "bytes": size, "sha256": sha256}
        )
        digest.update(f"{relative}\0{size}\0{sha256}\n".encode("utf-8"))
    return rows, digest.hexdigest()


def copy_verified_window_payload(
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Copy audited v5 window outputs and verify every copied byte by SHA256."""
    marker_path = source_root / "RUN_COMPLETE.json"
    validation_path = source_root / "validation_report.json"
    if not marker_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("window source lacks completion metadata")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "complete"
        or marker.get("all_validation_checks_passed") is not True
        or validation.get("status") != "PASS"
    ):
        raise ValueError("window source is not a completed PASS run")
    source_rows, source_digest = _payload_inventory(source_root, WINDOW_PAYLOADS)
    for relative in WINDOW_PAYLOADS:
        source = source_root / relative
        destination = output_root / relative
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite window payload: {destination}")
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
    destination_rows, destination_digest = _payload_inventory(
        output_root, WINDOW_PAYLOADS
    )
    if source_rows != destination_rows or source_digest != destination_digest:
        raise RuntimeError("copied window payload differs from audited source")
    write_csv(
        output_root / "window_reuse_sha256.csv",
        destination_rows,
        ["relative_path", "bytes", "sha256"],
    )

    geometry_path = (
        output_root / "window_provenance" / "integer_window_geometry.csv"
    )
    geometry = read_csv(geometry_path)
    powder_geometry = [
        row for row in geometry if str(row.get("role", "")).startswith("powder")
    ]
    starts = sorted(
        {
            as_float(row.get("nominal_start_deg"))
            for row in powder_geometry
            if np.isfinite(as_float(row.get("nominal_start_deg")))
        }
    )
    ends = sorted(
        {
            as_float(row.get("nominal_end_deg"))
            for row in powder_geometry
            if np.isfinite(as_float(row.get("nominal_end_deg")))
        }
    )
    geometry_verified = (
        starts == [float(value) for value in range(28)]
        and ends == [float(value) for value in range(5, 33)]
    )
    if not geometry_verified:
        raise RuntimeError(
            f"copied powder window geometry mismatch: starts={starts}, ends={ends}"
        )
    audit = {
        "delivered": True,
        "reuse_reason": (
            "integer-window inputs, definitions, and algorithms are unchanged "
            "by the new pressure-level peak aggregation"
        ),
        "source_suite": str(source_root.resolve()),
        "source_validation_status": validation.get("status"),
        "source_completion_status": marker.get("status"),
        "payload_files": len(source_rows),
        "payload_bytes": sum(int(row["bytes"]) for row in source_rows),
        "payload_digest": source_digest,
        "all_copied_paths_sizes_sha256_match": True,
        "powder_nominal_windows": 28,
        "powder_window_sequence": "0-5, 1-6, 2-7, ..., 27-32 degrees",
        "powder_geometry_verified": geometry_verified,
        "first_effective_observed_start_deg": 0.04347043,
        "across_definition": (
            "same scan/beam position across pressures, then median across scans"
        ),
        "within_definition": (
            "28x28 window matrix per raw frame, then pressure-level median"
        ),
        "presentation": "strict lower triangle only; diagonal and upper omitted",
        "source_independent_audit": (
            "audit_all_peak_integer_suite.py PASS and 43 related tests PASS "
            "before reuse"
        ),
        "powder_across_shape": [28, 19, 19],
        "powder_within_per_frame_shape": [1060, 28, 28],
        "powder_within_by_pressure_shape": [19, 28, 28],
        "fit_control_0_5_acf_offdiagonal": {
            "min": 0.996410,
            "median": 0.999408,
            "max": 0.999745,
            "interpretation": (
                "red saturation reflects a stable tungsten-dominated control "
                "fingerprint, not values overwritten with one"
            ),
        },
    }
    write_json(output_root / "window_reuse_manifest.json", audit)
    return audit


def grid_convergence_sample(
    observations: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    normalization_by_frame: Mapping[int, Mapping[str, Any]],
    profiles: Mapping[str, SparsePointProfile],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    q_step: float,
    azim_step_deg: float,
    sample_count: int = 128,
) -> dict[str, Any]:
    """Re-evaluate deterministic pair samples at twice the grid resolution."""
    if not pair_rows:
        raise ValueError("cannot run convergence audit without pairs")
    sample_indices = np.unique(
        np.linspace(
            0,
            len(pair_rows) - 1,
            min(sample_count, len(pair_rows)),
            dtype=int,
        )
    )
    sampled_rows = [pair_rows[int(index)] for index in sample_indices]
    needed_uids = {
        str(row[key])
        for row in sampled_rows
        for key in ("left_point_uid", "right_point_uid")
    }
    subset_points = [
        dict(point)
        for point in points
        if str(point["point_uid"]) in needed_uids
    ]
    fine_profiles, _kernel_rows, fine_audit = build_point_profiles(
        observations,
        subset_points,
        normalization_by_frame,
        q_step=q_step / 2.0,
        azim_step_deg=azim_step_deg / 2.0,
    )
    differences: list[float] = []
    fine_scores: list[float] = []
    for row in sampled_rows:
        left_uid = str(row["left_point_uid"])
        right_uid = str(row["right_point_uid"])
        left = fine_profiles[left_uid]
        right = fine_profiles[right_uid]
        fine = sparse_profile_iou(
            left.indices,
            left.values,
            right.indices,
            right.values,
            left.cell_area,
        )
        coarse = float(row["roi_integrated_iou"])
        fine_scores.append(fine)
        differences.append(abs(fine - coarse))
    values = np.asarray(differences, dtype=float)
    audit = {
        "method": (
            "deterministic cross-pressure pair sample re-rasterized with half "
            "q and azimuth grid steps"
        ),
        "sample_pairs": len(sampled_rows),
        "base_q_step": q_step,
        "fine_q_step": q_step / 2.0,
        "base_azim_step_deg": azim_step_deg,
        "fine_azim_step_deg": azim_step_deg / 2.0,
        "absolute_difference_median": float(np.median(values)),
        "absolute_difference_q95": float(np.quantile(values, 0.95)),
        "absolute_difference_max": float(np.max(values)),
        "fine_scores_in_0_1": bool(
            np.all(
                (np.asarray(fine_scores) >= 0.0)
                & (np.asarray(fine_scores) <= 1.0)
            )
        ),
        "fine_kernel_maximum_integral_abs_error": fine_audit[
            "maximum_discrete_kernel_integral_abs_error"
        ],
        "acceptance_q95_abs_difference_le_0p01": bool(
            np.quantile(values, 0.95) <= 0.01
        ),
    }
    if (
        not audit["fine_scores_in_0_1"]
        or not audit["acceptance_q95_abs_difference_le_0p01"]
    ):
        raise RuntimeError(f"grid convergence validation failed: {audit}")
    return audit


def build_artifact_index(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded = {"artifact_index.csv", "RUN_COMPLETE.json"}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name == ".DS_Store"
            or path.name in excluded
        ):
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


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
        "q_width",
        "azim_width_deg",
        "intensity",
        "area",
        "n_observations",
        "distinct_frames",
        "best_frame",
        "best_frame_file",
        "obs_indices_0based",
        "frames",
        "profile_grid_cells",
        "profile_integral_main",
        "profile_integral_fixed_D1w_sensitivity",
    )
    return [{key: point.get(key) for key in fields} for point in points]


def _write_readme(
    path: Path,
    *,
    window_audit: Mapping[str, Any] | None,
) -> None:
    windows = (
        "Included and byte-verified from the independently audited v5 payload."
        if window_audit
        else "Not included in this explicitly partial run."
    )
    path.write_text(
        f"""# UOTe pressure-level peak + integer-window correlation suite (v6)

## Delivered peak maps

The powder peak presentation contains 280 anchors, not 519 raw-observation
columns.  The 519 rows in `spot_observations.csv` are the measurement-level
members used to reconstruct 228 tracked and all 52 untracked pressure-level
points.  Every pressure-level point has:

- one `location` map;
- one `roi_integrated_iou` map.

Each matrix is 19 pressure rows by 22 possible pressure-local peak slots.
Pressure runs from 50.7 GPa at the top to 3.5 GPa at the bottom.  At each
pressure, `peak 1 ... peak N` is newly assigned by increasing 2theta; a column
number is therefore not a track identity.  Missing slots and the complete
anchor-pressure row are blank/NaN.  A dark zero is a valid comparison whose
compact supports do not overlap.

## Location

`clip(1 - abs(delta_2theta)/0.06 degrees, 0, 1)`.

## Formal 2D ROI-area similarity

Each raw observation is represented by a normalized compact biweight ellipse
in q x periodic azimuth.  Its semiaxes are `2*q_width` and
`2*azim_width_deg`.  Its area is corrected by the frame's tungsten-dominated
fit-control integral over 2-25 degrees:

`normalized area = raw area * median(D1s fit integral) / frame fit integral`.

This is an empirical internal measurement normalization; D1w is not assigned a
fictional exposure time.  Same-frame split blobs are summed before distinct
frames are averaged.  The final score is:

`integral min(F_anchor,F_target) / integral max(F_anchor,F_target)`.

## Window-to-window

{windows}

Powder windows are exactly 0-5, 1-6, ..., 27-32 degrees (the first observed
support begins at 0.04347043 degrees; no extrapolation).  Across-frame results
compare pressures within the same scan and then take the median across scans.
Within-frame results start from each raw frame's 28 x 28 window matrix and are
then summarized by pressure.  User-facing matrices retain only the strict lower
triangle; the diagonal and mirrored upper triangle are omitted.

The red tungsten fit-control 0-5 degree map is not a constant-one coding
error: its off-diagonal ACF values range approximately 0.996410-0.999745.
Use the nearby `one_minus_similarity_diagnostics` to inspect those small
differences.  The UOTe sample result is the `spots` channel.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    started = time.time()
    output_root = args.out_dir.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing result directory: {output_root}"
        )
    output_root.mkdir(parents=True)

    _progress("assigning 519 observations to 280 authoritative pressure points")
    observations, points, assignments, mapping_audit = (
        assign_observations_to_points(
            args.observations,
            args.track_points,
            args.untracked_points,
        )
    )
    write_csv(
        output_root / "observation_assignment.csv",
        assignments,
    )

    _progress("integrating fit-control profiles for empirical frame normalization")
    normalization_rows, normalization_by_frame, d1w_assessment = (
        build_measurement_normalization(args.manifest, args.fit_root)
    )
    write_csv(
        output_root / "frame_measurement_normalization.csv",
        normalization_rows,
    )
    write_csv(
        output_root / "D1w_D1s_paired_ratios.csv",
        d1w_assessment["pair_rows"],
    )
    d1w_json = dict(d1w_assessment)
    d1w_json.pop("pair_rows")
    write_json(output_root / "D1w_measurement_scale_assessment.json", d1w_json)

    _progress("constructing normalized compact 2D elliptical point fields")
    profiles, kernel_rows, profile_audit = build_point_profiles(
        observations,
        points,
        normalization_by_frame,
        q_step=args.q_step,
        azim_step_deg=args.azim_step_deg,
    )
    write_csv(output_root / "observation_kernel_audit.csv", kernel_rows)
    point_registry_rows = _point_registry_rows(points)
    write_csv(output_root / "point_registry.csv", point_registry_rows)

    pressure_layout, slot_lookup, maximum_slots = build_pressure_slot_layout(points)
    write_csv(output_root / "pressure_row_layout.csv", pressure_layout)
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
    write_csv(output_root / "pressure_peak_grid.csv", pressure_grid_rows)

    _progress("computing all 37,038 unordered cross-pressure point pairs")
    (
        location_matrix,
        roi_matrix,
        sensitivity_matrix,
        pair_rows,
        pair_audit,
    ) = compute_cross_pressure_pairs(points, profiles)
    pair_count = write_csv_gz(
        output_root / "all_cross_pressure_peak_pairs.csv.gz",
        pair_rows,
        PAIR_FIELDS,
    )
    if pair_count != 37038:
        raise RuntimeError(f"pair table row count mismatch: {pair_count}")
    np.savez_compressed(
        output_root / "pressure_level_similarity_matrices.npz",
        point_uids=np.asarray([str(point["point_uid"]) for point in points]),
        location=location_matrix,
        roi_integrated_iou=roi_matrix,
        roi_fixed_D1w_sensitivity_iou=sensitivity_matrix,
    )

    _progress("checking numerical quadrature convergence on 128 pair samples")
    convergence_audit = grid_convergence_sample(
        observations,
        points,
        normalization_by_frame,
        profiles,
        pair_rows,
        q_step=args.q_step,
        azim_step_deg=args.azim_step_deg,
    )

    _progress("writing per-anchor pressure x local-peak maps")
    anchor_rows, map_audit = write_anchor_maps(
        output_root,
        points,
        location_matrix,
        roi_matrix,
        pressure_layout,
        slot_lookup,
        maximum_slots,
        make_plots=not args.no_plots,
        max_anchors=args.max_anchors,
    )
    write_csv(output_root / "anchor_map_index.csv", anchor_rows)

    window_audit: dict[str, Any] | None = None
    if not args.no_windows:
        _progress("copying and byte-verifying audited across/within window results")
        window_audit = copy_verified_window_payload(
            args.reuse_window_suite.resolve(),
            output_root,
        )

    complete_required_run = (
        map_audit["complete_anchor_run"]
        and not args.no_plots
        and window_audit is not None
    )
    validation = {
        "status": "PASS" if complete_required_run else "PARTIAL_PASS",
        "complete_required_run": complete_required_run,
        "source_counts": {
            "raw_observations": len(observations),
            "pressure_level_points": len(points),
            "tracked_points": 228,
            "untracked_points": 52,
            "pressure_levels": len(PRESSURES_DESCENDING),
        },
        "mapping": mapping_audit,
        "measurement_normalization": d1w_json,
        "profile_construction": profile_audit,
        "pair_scores": pair_audit,
        "grid_convergence": convergence_audit,
        "peak_maps": map_audit,
        "window_results": window_audit,
        "required_components": {
            "ROI_area": bool(map_audit["ROI_heatmap_png_files"] == 280),
            "location": bool(map_audit["location_heatmap_png_files"] == 280),
            "window_across_frames": window_audit is not None,
            "window_within_frames": window_audit is not None,
            "window_strict_lower_triangle_only": bool(
                window_audit
                and window_audit["presentation"].startswith("strict lower")
            ),
        },
    }
    if complete_required_run and not all(validation["required_components"].values()):
        raise RuntimeError("one or more required result components are missing")
    write_json(output_root / "validation_report.json", validation)

    run_manifest = {
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "observations": str(args.observations.resolve()),
            "track_points": str(args.track_points.resolve()),
            "untracked_points": str(args.untracked_points.resolve()),
            "manifest": str(args.manifest.resolve()),
            "fit_root": str(args.fit_root.resolve()),
            "window_source": (
                str(args.reuse_window_suite.resolve()) if not args.no_windows else None
            ),
        },
        "input_sha256": {
            "observations": file_sha256(args.observations),
            "track_points": file_sha256(args.track_points),
            "untracked_points": file_sha256(args.untracked_points),
            "manifest": file_sha256(args.manifest),
        },
        "parameters": {
            "powder_wavelength_A": POWDER_WAVELENGTH_A,
            "position_tolerance_deg": POSITION_TOLERANCE_DEG,
            "kernel": "compact biweight ellipse",
            "kernel_power": KERNEL_POWER,
            "ellipse_semiaxes": ["2*q_width", "2*azim_width_deg"],
            "q_grid_step": args.q_step,
            "azim_grid_step_deg": args.azim_step_deg,
            "ROI_formula": "integral(min(Fa,Fb))/integral(max(Fa,Fb))",
            "same_pressure_comparisons": "omitted entire row",
            "pressure_order": "descending",
            "local_peak_order": "increasing 2theta independently at each pressure",
            "window_presentation": "strict lower triangle; diagonal omitted",
        },
        "output_root": str(output_root),
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    _write_readme(output_root / "README.md", window_audit=window_audit)

    artifact_rows = build_artifact_index(output_root)
    write_csv(
        output_root / "artifact_index.csv",
        artifact_rows,
        ["relative_path", "bytes", "sha256"],
    )
    completion = {
        "status": "complete" if complete_required_run else "partial",
        "all_validation_checks_passed": complete_required_run,
        "validation_report_sha256": file_sha256(
            output_root / "validation_report.json"
        ),
        "artifact_index_sha256": file_sha256(output_root / "artifact_index.csv"),
        "artifact_count_excluding_index_and_marker": len(artifact_rows),
        "required_components": validation["required_components"],
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_root / "RUN_COMPLETE.json", completion)
    _progress(
        f"finished status={completion['status']} at {output_root} "
        f"in {completion['elapsed_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
