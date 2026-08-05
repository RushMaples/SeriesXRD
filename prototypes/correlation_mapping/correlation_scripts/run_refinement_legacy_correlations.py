#!/usr/bin/env python3
"""Apply the frozen 2026-07-13 legacy correlation method to refinement exports.

Whole-pattern and ACF calculations call the original legacy implementation
directly.  The per-peak adapter deliberately does *not* redetect peaks from 1D
patterns: it consumes the curated ``kept_obs.csv`` rows as requested, groups
them by physical track, and applies the legacy location similarity and min/max
normalized-area similarity to measured observations only.

The two single-crystal orientations remain independent pressure ladders for
whole-pattern and window/ACF analysis.  Single-crystal per-peak analysis instead
uses each global track across every available Masked frame, with orientation
retained only as frame metadata.  Powder frames are paired only within one scan
position and aggregated across scans by the median, matching the legacy runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_uote_xy_handoff_correlations as legacy  # noqa: E402
from single_global_per_peak import (  # noqa: E402
    analyze_single_tracks_across_frames,
    branch_label as single_branch_label,
    orientation_base as single_orientation_base,
)


LEGACY_PROFILE = "legacy-correlation-20260713"
LEGACY_EXPECTED_SHA256 = "7fead169eda80aba25a407d030695b2856df211f34f4857c384d6b9e8e41fb61"
SINGLE_WAVELENGTH_A = 0.4133
POWDER_WAVELENGTH_A = 0.3066
POSITION_TOLERANCE_DEG = 0.06
WINDOW_WIDTH_DEG = 5.0
WINDOW_STEP_DEG = 1.0
WINDOW_SHIFT_TOLERANCE_DEG = 1.0
GRID_STEP_DEG = 0.02
MIN_TWO_THETA_DEG = 2.0
POWDER_MAX_TWO_THETA_DEG = 32.0
NEAR_GAP_GPA = 1.5
FAR_GAP_GPA = 15.0

EXPECTED_SINGLE_SELECTION = {
    "orientation_0deg": [
        (2, 1.0), (8, 1.5), (10, 2.3), (14, 3.9), (16, 5.0),
        (18, 6.2), (20, 7.4), (22, 8.5), (23, 9.8), (4, 12.0), (6, 12.8),
    ],
    "orientation_10deg": [
        (0, 1.0), (7, 1.5), (9, 2.3), (13, 3.9), (15, 5.0),
        (17, 6.2), (19, 7.4), (21, 8.5), (26, 9.8), (3, 12.0), (5, 12.8),
    ],
}
EXPECTED_SINGLE_MASKED_FRAMES = {0, 4, 5, 7, 10, 11, 13, 15, 17, 19, 21, 27}
EXPECTED_SINGLE_MASKED_OBSERVATIONS = 275
EXPECTED_SINGLE_MASKED_TRACKS = 75
EXPECTED_SINGLE_FRAME_TRACK_FEATURES = 263
EXPECTED_SINGLE_COMPARABLE_TRACKS = 49
EXPECTED_SINGLE_USABLE_TRACKS = 33
EXPECTED_SINGLE_SINGLETON_TRACKS = 26
EXPECTED_SINGLE_DUPLICATE_FRAME_TRACKS = 11
EXPECTED_SINGLE_UNIQUE_PAIRS = 653


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("correlations/UOTe XRD Data Refinement"),
    )
    parser.add_argument(
        "--single-manifest",
        type=Path,
        default=Path("correlations/manifests/uote_single_crystal_uniform_v2_1_manifest.csv"),
        help="Used only as frame/orientation/pressure metadata and exclusion registry.",
    )
    parser.add_argument(
        "--single-raw-root",
        type=Path,
        default=Path("Data/Cell_29"),
        help="Raw TIFF root used only for exposure/pixel-normalized single-crystal intensity.",
    )
    parser.add_argument(
        "--legacy-reference",
        type=Path,
        default=Path("correlations/results/uote_xy_handoff2_correlations_20260713"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def q_to_two_theta(q_a_inv: float | np.ndarray, wavelength_a: float) -> np.ndarray:
    q = np.asarray(q_a_inv, dtype=float)
    ratio = q * wavelength_a / (4.0 * np.pi)
    out = np.full(q.shape, np.nan, dtype=float)
    keep = np.abs(ratio) < 1.0
    out[keep] = np.degrees(2.0 * np.arcsin(ratio[keep]))
    return out


def circular_delta_deg(left: np.ndarray | float, right: float) -> np.ndarray:
    return (np.asarray(left, dtype=float) - float(right) + 180.0) % 360.0 - 180.0


def make_grid(max_two_theta: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(
        MIN_TWO_THETA_DEG,
        max_two_theta + GRID_STEP_DEG / 2.0,
        GRID_STEP_DEG,
    )
    first_start = math.ceil(MIN_TWO_THETA_DEG / WINDOW_STEP_DEG) * WINDOW_STEP_DEG
    last_start = math.floor((max_two_theta - WINDOW_WIDTH_DEG) / WINDOW_STEP_DEG) * WINDOW_STEP_DEG
    starts = np.arange(first_start, last_start + WINDOW_STEP_DEG / 2.0, WINDOW_STEP_DEG)
    return grid, starts


def load_xy_paths(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("No XY paths supplied")
    first = np.loadtxt(paths[0], comments="#")
    x = first[:, 0].astype(float)
    values = np.empty((len(paths), len(x)), dtype=np.float32)
    values[0] = first[:, 1]
    for index, path in enumerate(paths[1:], start=1):
        data = np.loadtxt(path, comments="#")
        if data.shape != first.shape or not np.array_equal(data[:, 0], x):
            raise ValueError(f"Inconsistent XY grid: {path}")
        values[index] = data[:, 1]
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite intensity in XY input")
    return x, values


def run_whole_and_windows(
    out_root: Path,
    channel_label: str,
    frames: list[legacy.Frame],
    scans: list[str],
    pressures: list[float],
    x_native: np.ndarray,
    raw: np.ndarray,
    max_two_theta: float,
    make_plots: bool,
) -> dict[str, Any]:
    grid, starts = make_grid(max_two_theta)
    if grid[0] < x_native[0] - 1e-9 or grid[-1] > x_native[-1] + 1e-9:
        raise ValueError(
            f"Analysis grid {grid[0]}..{grid[-1]} lies outside measured support "
            f"{x_native[0]}..{x_native[-1]}"
        )
    normalized_native = legacy.normalize_rows(raw)
    normalized = legacy.resample_rows(x_native, normalized_native, grid)
    _, fingerprints = legacy.build_window_fingerprints(
        grid,
        normalized,
        starts,
        WINDOW_WIDTH_DEG,
    )
    across, within = legacy.analyze_windows(
        out_root,
        channel_label,
        frames,
        scans,
        pressures,
        starts,
        fingerprints,
        WINDOW_WIDTH_DEG,
        WINDOW_SHIFT_TOLERANCE_DEG,
        WINDOW_STEP_DEG,
        make_plots,
    )
    whole = legacy.analyze_whole_pattern(
        out_root / "whole_pattern",
        channel_label,
        frames,
        scans,
        pressures,
        normalized,
        make_plots,
    )
    return {
        "whole_pattern": whole,
        "across_frames": across,
        "within_frame": within,
        "analysis_grid": {
            "min_two_theta_deg": float(grid[0]),
            "max_two_theta_deg": float(grid[-1]),
            "bins": int(len(grid)),
            "window_count": int(len(starts)),
            "window_starts_deg": starts,
        },
    }


def load_single_metadata(manifest_path: Path, patterns_dir: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    rows = read_csv(manifest_path)
    metadata: dict[int, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for row in rows:
        frame = int(row["frame"])
        included = row["excluded"] == "0" and row["scan"] in {"orientation_0deg", "orientation_10deg"}
        path = patterns_dir / f"frame_{frame:04d}.xy"
        record = {
            "frame": frame,
            "scan": row["scan"],
            "orientation": row["scan"],
            "pressure_GPa": float(row["pressure_GPa"]),
            "included_whole_pattern": int(included),
            "exclusion_reason": row["exclusion_reason"],
            "original_filename": row["original_filename"],
            "file_path": str(path.resolve()),
            "file_exists": int(path.is_file()),
            "wavelength_A": SINGLE_WAVELENGTH_A,
        }
        metadata[frame] = record
        audit.append(record)
    return metadata, audit


def run_single_whole(
    out_root: Path,
    metadata: dict[int, dict[str, Any]],
    patterns_dir: Path,
    make_plots: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {}
    selected_rows: list[dict[str, Any]] = []
    for orientation in ("orientation_0deg", "orientation_10deg"):
        rows = sorted(
            [row for row in metadata.values() if row["included_whole_pattern"] and row["orientation"] == orientation],
            key=lambda item: item["pressure_GPa"],
        )
        pressures = [float(row["pressure_GPa"]) for row in rows]
        pressure_index = {pressure: index for index, pressure in enumerate(pressures)}
        frames = [
            legacy.Frame(
                frame=int(row["frame"]),
                scan=orientation,
                pressure=float(row["pressure_GPa"]),
                pressure_index=pressure_index[float(row["pressure_GPa"])],
                original_filename=str(row["original_filename"]),
            )
            for row in rows
        ]
        paths = [patterns_dir / f"frame_{frame.frame:04d}.xy" for frame in frames]
        x_native, raw = load_xy_paths(paths)
        max_two_theta = math.floor(float(x_native[-1]) / GRID_STEP_DEG) * GRID_STEP_DEG
        label = orientation.replace("orientation_", "")
        series_root = out_root / label
        metrics[orientation] = run_whole_and_windows(
            series_root,
            f"single_crystal_{label}",
            frames,
            [orientation],
            pressures,
            x_native,
            raw,
            max_two_theta,
            make_plots,
        )
        metrics[orientation]["frames"] = len(frames)
        metrics[orientation]["pressures_GPa"] = pressures
        metrics[orientation]["native_support_two_theta_deg"] = [float(x_native[0]), float(x_native[-1])]
        selected_rows.extend(rows)
    return metrics, selected_rows


def run_powder_whole(
    out_root: Path,
    reduced_root: Path,
    make_plots: bool,
) -> tuple[dict[str, Any], list[legacy.Frame], list[dict[str, str]], list[float], list[str]]:
    frames, manifest_rows, pressures, scans = legacy.read_manifest(reduced_root / "manifest.csv", None)
    metrics: dict[str, Any] = {}
    for channel in ("spots", "fit"):
        x_native, raw, paths = legacy.load_channel(reduced_root, channel, frames)
        channel_root = out_root / channel
        metrics[channel] = run_whole_and_windows(
            channel_root,
            f"powder_{channel}",
            frames,
            scans,
            pressures,
            x_native,
            raw,
            POWDER_MAX_TWO_THETA_DEG,
            make_plots,
        )
        metrics[channel]["files"] = len(paths)
        metrics[channel]["native_support_two_theta_deg"] = [float(x_native[0]), float(x_native[-1])]
    return metrics, frames, manifest_rows, pressures, scans


def parse_tiff_exposure_seconds(image: Image.Image) -> float:
    description = str(image.tag_v2.get(270, ""))
    match = re.search(r"Exposure_time\s+([0-9.]+)\s*s", description)
    if not match:
        raise ValueError("Exposure_time is missing from TIFF ImageDescription")
    return float(match.group(1))


def raw_tiff_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in sorted(root.rglob("*.tif")):
        key = path.name.lower()
        if key in index:
            duplicates.add(key)
        else:
            index[key] = path
    if duplicates:
        raise ValueError(f"Duplicate TIFF basenames under {root}: {sorted(duplicates)[:3]}")
    return index


def extract_single_track_observations(
    kept_path: Path,
    masked_dir: Path,
    raw_root: Path,
    metadata: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import pyFAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyFAI is required for single-crystal ROI intensity extraction") from exc

    kept = read_csv(kept_path)
    unknown_frames = sorted({int(row["frame"]) for row in kept} - set(metadata))
    if unknown_frames:
        raise ValueError(f"Masked observations have no frame metadata: {unknown_frames}")
    # Whole-pattern exclusions do not apply to per-peak tracking.  The curated
    # Masked table is the authority here, including the decompression and
    # alternate/repeat frames that have valid masks and raw TIFFs.
    included = list(kept)
    raw_index = raw_tiff_index(raw_root)
    ai = pyFAI.load(str(masked_dir / "_geometry.poni"))
    shape = (1043, 981)
    q_array = ai.qArray(shape) / 10.0
    chi_array = np.degrees(ai.center_array(shape, unit="chi_rad"))
    detector_mask = np.asarray(ai.detector.calc_mask(), dtype=bool)
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in included:
        by_frame[int(row["frame"])].append(row)

    out: list[dict[str, Any]] = []
    frame_qc: list[dict[str, Any]] = []
    for frame in sorted(by_frame):
        meta = metadata[frame]
        basename = Path(str(meta["original_filename"]).replace("\\", "/")).name.lower()
        raw_path = raw_index.get(basename)
        if raw_path is None:
            raise FileNotFoundError(f"Raw TIFF not found for frame {frame}: {basename}")
        with Image.open(raw_path) as image:
            exposure_s = parse_tiff_exposure_seconds(image)
            raw = np.asarray(image, dtype=float)
        if raw.shape != shape:
            raise ValueError(f"Unexpected TIFF shape for {raw_path}: {raw.shape}")
        frame_mask = np.load(masked_dir / f"frame_{frame:04d}_mask.npy")
        valid_detector = (~detector_mask) & np.isfinite(raw) & (raw >= 0)
        reconstructed = np.zeros(shape, dtype=bool)
        for row in by_frame[frame]:
            q0 = float(row["q"])
            azim = float(row["azim_deg"])
            half_q = float(row["halfwidth_q"])
            half_azim = float(row["halfwidth_azim_deg"])
            radial = np.abs(q_array - q0)
            azimuthal = np.abs(circular_delta_deg(chi_array, azim))
            geometric_roi = (radial <= half_q) & (azimuthal <= half_azim)
            reconstructed |= geometric_roi
            roi = geometric_roi & (~frame_mask) & valid_detector
            side = (
                (radial > half_q * 1.15)
                & (radial <= half_q * 2.15)
                & (azimuthal <= half_azim)
                & valid_detector
            )
            pixels = raw[roi]
            side_pixels = raw[side]
            effective_pixels = int(len(pixels))
            if effective_pixels == 0:
                background = np.nan
                excess_counts = np.nan
                rate_per_pixel = np.nan
            else:
                background = (
                    float(np.nanmedian(side_pixels))
                    if len(side_pixels) >= 5
                    else float(np.nanpercentile(pixels, 10))
                )
                excess_counts = float(np.sum(np.clip(pixels - background, 0.0, None)))
                rate_per_pixel = excess_counts / exposure_s / effective_pixels
            two_theta = float(q_to_two_theta(q0, SINGLE_WAVELENGTH_A))
            out.append({
                "dataset": "single_crystal",
                "orientation": meta["orientation"],
                "scan": meta["orientation"],
                "orientation_base": single_orientation_base(str(meta["orientation"])),
                "branch": single_branch_label(meta),
                "frame": frame,
                "pressure_GPa": float(meta["pressure_GPa"]),
                "whole_pattern_included": int(meta["included_whole_pattern"]),
                "whole_pattern_exclusion_reason": meta["exclusion_reason"],
                "track": int(row["track"]),
                "obs_row": int(row["obs_row"]),
                "q_A^-1": q0,
                "d_A": float(row["d_A"]),
                "two_theta_deg": two_theta,
                "azim_deg": azim,
                "halfwidth_q_A^-1": half_q,
                "halfwidth_azim_deg": half_azim,
                "matched_d_A": float(row["matched_d_A"]) if row["matched_d_A"] else np.nan,
                "raw_background_counts": background,
                "raw_excess_counts": excess_counts,
                "exposure_s": exposure_s,
                "exposure_source": "TIFF ImageDescription Exposure_time",
                "effective_pixels": effective_pixels,
                "normalized_intensity_counts_per_s_per_pixel": rate_per_pixel,
                "intensity_status": "verified_tiff_exposure_and_roi_pixels",
                "raw_tiff": str(raw_path.resolve()),
            })
        kept_pixels = ~frame_mask
        union = reconstructed | kept_pixels
        intersection = reconstructed & kept_pixels
        jaccard = float(np.count_nonzero(intersection) / max(np.count_nonzero(union), 1))
        frame_qc.append({
            "frame": frame,
            "orientation": meta["orientation"],
            "pressure_GPa": meta["pressure_GPa"],
            "regions": len(by_frame[frame]),
            "mask_kept_pixels": int(np.count_nonzero(kept_pixels)),
            "reconstructed_pixels": int(np.count_nonzero(reconstructed)),
            "roi_mask_jaccard": jaccard,
            "exposure_s": exposure_s,
            "raw_tiff": str(raw_path.resolve()),
        })
    return out, frame_qc


def filename_exposure_seconds(filename: str) -> tuple[float, str]:
    match = re.search(r"D([0-9]+(?:p[0-9]+)?)s(?:_|\.)", filename, flags=re.IGNORECASE)
    if not match:
        return np.nan, "missing"
    value = float(match.group(1).replace("p", "."))
    return value, "filename D#s token (TIFF metadata unavailable)"


def match_powder_observation(
    kept: dict[str, str],
    candidates: list[dict[str, str]],
) -> tuple[dict[str, str], str]:
    q0 = float(kept["q"])
    d0 = float(kept["d_A"])
    az0 = float(kept["azim_deg"])
    matches = [
        row
        for row in candidates
        if abs(float(row["q"]) - q0) <= 1e-4
        and abs(float(row["d_A"]) - d0) <= 1e-4
        and abs(float(circular_delta_deg(float(row["azim_deg"]), az0))) <= 0.1
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one powder observation match for frame={kept['frame']} q={kept['q']} "
            f"azim={kept['azim_deg']}; found {len(matches)}"
        )
    # ``obs_row`` is not frame-local in this export.  The coordinate tuple is
    # the join key; requiring uniqueness above prevents a first-match error.
    return matches[0], "composite_coordinate_match"


def extract_powder_track_observations(
    kept_path: Path,
    observations_path: Path,
    tracks_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = read_csv(kept_path)
    observations = read_csv(observations_path)
    manifest = {int(row["frame"]): row for row in read_csv(manifest_path)}
    track_meta = {int(row["track"]): row for row in read_csv(tracks_path)}
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in observations:
        by_frame[int(row["frame"])].append(row)

    out: list[dict[str, Any]] = []
    join_status = Counter()
    for row in kept:
        frame = int(row["frame"])
        source, status = match_powder_observation(row, by_frame[frame])
        join_status[status] += 1
        meta = manifest[frame]
        if meta["cover_excluded"] != "0":
            raise ValueError(f"Masked powder observation refers to excluded frame {frame}")
        exposure_s, exposure_source = filename_exposure_seconds(meta["filename"])
        pixels = int(source["n_pixels"])
        area = float(source["area"])
        normalized = area / exposure_s / pixels if np.isfinite(exposure_s) and exposure_s > 0 and pixels > 0 else np.nan
        track = int(row["track"])
        tmeta = track_meta.get(track, {})
        out.append({
            "dataset": "powder",
            "orientation": "not_applicable",
            "scan": meta["scan"],
            "frame": frame,
            "pressure_GPa": float(meta["pressure_GPa"]),
            "track": track,
            "obs_row": int(row["obs_row"]),
            "q_A^-1": float(row["q"]),
            "d_A": float(row["d_A"]),
            "two_theta_deg": float(q_to_two_theta(float(row["q"]), POWDER_WAVELENGTH_A)),
            "azim_deg": float(row["azim_deg"]),
            "halfwidth_q_A^-1": float(row["halfwidth_q"]),
            "halfwidth_azim_deg": float(row["halfwidth_azim_deg"]),
            "matched_d_A": float(tmeta["match_d_calc_A"]) if tmeta.get("match_d_calc_A") else np.nan,
            "match_hkl": tmeta.get("match_hkl", ""),
            "raw_background_counts": np.nan,
            "raw_excess_counts": area,
            "source_peak_height_counts": float(source["intensity"]),
            "source_snr": float(source["snr"]),
            "exposure_s": exposure_s,
            "exposure_source": exposure_source,
            "effective_pixels": pixels,
            "normalized_intensity_counts_per_s_per_pixel": normalized,
            "intensity_status": "filename_exposure_assumption_area_over_pixels",
            "join_status": status,
            "original_filename": meta["filename"],
        })
    return out, {
        "kept_rows": len(kept),
        "matched_rows": len(out),
        "join_status": dict(join_status),
        "exposure_seconds": sorted({float(row["exposure_s"]) for row in out if np.isfinite(float(row["exposure_s"]))}),
        "all_exposure_tokens_parsed": all(np.isfinite(float(row["exposure_s"])) for row in out),
        "exposure_caveat": "Powder raw TIFF metadata was not supplied; D1s is transparently interpreted as 1 s.",
    }


def similarity_matrix(values: np.ndarray, location: bool) -> np.ndarray:
    n = len(values)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(i):
            if not (np.isfinite(values[i]) and np.isfinite(values[j])):
                continue
            if location:
                score = 1.0 - abs(float(values[i]) - float(values[j])) / POSITION_TOLERANCE_DEG
            else:
                high = max(float(values[i]), float(values[j]))
                low = min(float(values[i]), float(values[j]))
                score = low / high if high > 0 and low >= 0 else np.nan
            matrix[i, j] = float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else np.nan
    return matrix


def robust_mad(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def collapse_frame_track_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scan"]), int(row["track"]), float(row["pressure_GPa"]), int(row["frame"]))].append(row)
    out: list[dict[str, Any]] = []
    for (scan, track, pressure, frame), items in sorted(grouped.items()):
        q_values = [float(item["q_A^-1"]) for item in items]
        d_values = [float(item["d_A"]) for item in items]
        tt_values = [float(item["two_theta_deg"]) for item in items]
        intensity_values = [float(item["normalized_intensity_counts_per_s_per_pixel"]) for item in items]
        finite_intensity = [value for value in intensity_values if np.isfinite(value)]
        first = items[0]
        out.append({
            "dataset": first["dataset"],
            "orientation": first["orientation"],
            "scan": scan,
            "track": track,
            "frame": frame,
            "pressure_GPa": pressure,
            "n_observations": len(items),
            "q_median_A^-1": float(np.median(q_values)),
            "q_mad_A^-1": robust_mad(q_values),
            "d_median_A": float(np.median(d_values)),
            "d_mad_A": robust_mad(d_values),
            "two_theta_median_deg": float(np.median(tt_values)),
            "two_theta_mad_deg": robust_mad(tt_values),
            "normalized_intensity_median": float(np.median(finite_intensity)) if finite_intensity else np.nan,
            "normalized_intensity_mad": robust_mad(finite_intensity),
            "duplicate_observation_flag": int(len(items) > 1),
            "matched_d_A_anchor_candidate": first.get("matched_d_A", np.nan),
            "match_hkl": first.get("match_hkl", ""),
            "intensity_status": first["intensity_status"],
        })
    return out


def matrix_pair_values(matrices: np.ndarray, pressures: list[float]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    near: list[float] = []
    far: list[float] = []
    rows: list[dict[str, Any]] = []
    for i, p_i in enumerate(pressures):
        for j, p_j in enumerate(pressures[:i]):
            values = matrices[..., i, j].reshape(-1)
            values = values[np.isfinite(values)]
            gap = abs(p_i - p_j)
            if gap <= NEAR_GAP_GPA:
                near.extend(values.tolist())
            elif gap >= FAR_GAP_GPA:
                far.extend(values.tolist())
            for value in values:
                rows.append({"pressure_a_GPa": p_i, "pressure_b_GPa": p_j, "pressure_gap_GPa": gap, "similarity": float(value)})
    return np.asarray(near), np.asarray(far), rows


def plot_track_trajectory(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pressures = np.asarray([float(row["pressure_GPa"]) for row in rows])
    d_values = np.asarray([float(row["d_median_A"]) for row in rows])
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.scatter(pressures, d_values, color="#2A6F97", s=30)
    if len(np.unique(pressures)) >= 3:
        slope, _, _ = legacy.linear_summary(pressures, d_values)
        intercept = float(np.mean(d_values) - slope * np.mean(pressures))
        line_x = np.linspace(float(np.min(pressures)), float(np.max(pressures)), 100)
        ax.plot(line_x, slope * line_x + intercept, color="#B23A48", linewidth=1.8)
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("d spacing (A)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def analyze_direct_tracks(
    out_root: Path,
    dataset_label: str,
    observations: list[dict[str, Any]],
    pressure_registry: list[float],
    scan_registry: list[str],
    make_plots: bool,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    collapsed = collapse_frame_track_rows(observations)
    write_csv(out_root / "track_observations.csv", observations)
    write_csv(out_root / "frame_track_features.csv", collapsed)
    tracks = sorted({int(row["track"]) for row in collapsed})
    pressure_index = {float(value): index for index, value in enumerate(pressure_registry)}
    scan_index = {value: index for index, value in enumerate(scan_registry)}
    location = np.full(
        (len(tracks), len(scan_registry), len(pressure_registry), len(pressure_registry)),
        np.nan,
        dtype=np.float32,
    )
    intensity = np.full_like(location, np.nan)
    track_index = {track: index for index, track in enumerate(tracks)}

    feature_lookup: dict[tuple[int, str], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in collapsed:
        key = (int(row["track"]), str(row["scan"]))
        pressure = float(row["pressure_GPa"])
        if pressure in feature_lookup[key]:
            raise ValueError(f"Multiple frame features at identical track/scan/pressure: {key} {pressure}")
        feature_lookup[key][pressure] = row

    for (track, scan), values_by_pressure in feature_lookup.items():
        available = sorted(values_by_pressure)
        loc_values = np.asarray([float(values_by_pressure[p]["two_theta_median_deg"]) for p in available])
        int_values = np.asarray([float(values_by_pressure[p]["normalized_intensity_median"]) for p in available])
        loc_matrix = similarity_matrix(loc_values, location=True)
        int_matrix = similarity_matrix(int_values, location=False)
        for local_i, p_i in enumerate(available):
            for local_j, p_j in enumerate(available[:local_i]):
                location[track_index[track], scan_index[scan], pressure_index[p_i], pressure_index[p_j]] = loc_matrix[local_i, local_j]
                intensity[track_index[track], scan_index[scan], pressure_index[p_i], pressure_index[p_j]] = int_matrix[local_i, local_j]

    aggregate_location = legacy.nanmedian(location, axis=1)
    aggregate_intensity = legacy.nanmedian(intensity, axis=1)
    np.savez_compressed(
        out_root / "per_track_matrices.npz",
        pressure_gpa=np.asarray(pressure_registry),
        scan_names=np.asarray(scan_registry),
        track_ids=np.asarray(tracks),
        location_by_scan=location,
        intensity_by_scan=intensity,
        location_aggregate=aggregate_location,
        intensity_aggregate=aggregate_intensity,
    )

    labels = [f"{value:g}" for value in pressure_registry]
    summary_rows: list[dict[str, Any]] = []
    pressure_feature_rows: list[dict[str, Any]] = []
    for track in tracks:
        track_rows = [row for row in collapsed if int(row["track"]) == track]
        by_pressure: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in track_rows:
            by_pressure[float(row["pressure_GPa"])].append(row)
        trajectory: list[dict[str, Any]] = []
        for pressure in sorted(by_pressure):
            items = by_pressure[pressure]
            d_values = [float(item["d_median_A"]) for item in items]
            q_values = [float(item["q_median_A^-1"]) for item in items]
            tt_values = [float(item["two_theta_median_deg"]) for item in items]
            intensity_values = [float(item["normalized_intensity_median"]) for item in items]
            finite_intensity = [value for value in intensity_values if np.isfinite(value)]
            record = {
                "dataset": dataset_label,
                "track": track,
                "pressure_GPa": pressure,
                "scan_support": len({str(item["scan"]) for item in items}),
                "frame_support": len(items),
                "observation_support": sum(int(item["n_observations"]) for item in items),
                "q_median_A^-1": float(np.median(q_values)),
                "q_mad_A^-1": robust_mad(q_values),
                "d_median_A": float(np.median(d_values)),
                "d_mad_A": robust_mad(d_values),
                "two_theta_median_deg": float(np.median(tt_values)),
                "normalized_intensity_median": float(np.median(finite_intensity)) if finite_intensity else np.nan,
                "normalized_intensity_mad": robust_mad(finite_intensity),
                "duplicate_or_multiscan_flag": int(len(items) > 1),
            }
            trajectory.append(record)
            pressure_feature_rows.append(record)
        p = np.asarray([float(row["pressure_GPa"]) for row in trajectory])
        d = np.asarray([float(row["d_median_A"]) for row in trajectory])
        slope, slope_r, slope_r2 = legacy.linear_summary(p, d)
        loc_near, loc_far, _ = matrix_pair_values(location[track_index[track]], pressure_registry)
        int_near, int_far, _ = matrix_pair_values(intensity[track_index[track]], pressure_registry)
        anchor_candidates = [
            float(row["matched_d_A_anchor_candidate"])
            for row in track_rows
            if np.isfinite(float(row["matched_d_A_anchor_candidate"]))
        ]
        unique_anchor_candidates = sorted({round(value, 5) for value in anchor_candidates})
        hkl_values = [str(row["match_hkl"]) for row in track_rows if str(row["match_hkl"])]
        summary_rows.append({
            "dataset": dataset_label,
            "track": track,
            "match_hkl": Counter(hkl_values).most_common(1)[0][0] if hkl_values else "",
            "matched_d_A_reference_median": float(np.median(anchor_candidates)) if anchor_candidates else np.nan,
            "matched_d_A_reference_min": float(np.min(anchor_candidates)) if anchor_candidates else np.nan,
            "matched_d_A_reference_max": float(np.max(anchor_candidates)) if anchor_candidates else np.nan,
            "matched_d_A_reference_unique_count": len(unique_anchor_candidates),
            "pressure_points": len(trajectory),
            "pressure_min_GPa": float(np.min(p)),
            "pressure_max_GPa": float(np.max(p)),
            "scan_count": len({str(row["scan"]) for row in track_rows}),
            "frame_count": len(track_rows),
            "raw_observation_count": sum(int(row["n_observations"]) for row in track_rows),
            "duplicate_frame_track_count": sum(int(row["duplicate_observation_flag"]) for row in track_rows),
            "trajectory_status": "usable" if len(trajectory) >= 3 else "insufficient_lt3_pressures",
            "dd_dp_A_per_GPa": slope,
            "d_slope_r": slope_r,
            "d_slope_r2": slope_r2,
            "location_near_median": float(np.nanmedian(loc_near)) if len(loc_near) else np.nan,
            "location_far_median": float(np.nanmedian(loc_far)) if len(loc_far) else np.nan,
            "location_near_vs_far_auc": legacy.auc_probability(loc_near, loc_far),
            "intensity_near_median": float(np.nanmedian(int_near)) if len(int_near) else np.nan,
            "intensity_far_median": float(np.nanmedian(int_far)) if len(int_far) else np.nan,
            "intensity_near_vs_far_auc": legacy.auc_probability(int_near, int_far),
        })
        stem = f"track_{track:03d}"
        legacy.write_matrix_csv(out_root / "location_matrices" / f"{stem}.csv", labels, aggregate_location[track_index[track]])
        legacy.write_matrix_csv(out_root / "intensity_matrices" / f"{stem}.csv", labels, aggregate_intensity[track_index[track]])
        legacy.write_matrix_csv(out_root / "normalized_area_matrices" / f"{stem}.csv", labels, aggregate_intensity[track_index[track]])
        if make_plots and len(trajectory) >= 2:
            legacy.plot_heatmap(
                out_root / "location_heatmaps" / f"{stem}.png",
                labels,
                aggregate_location[track_index[track]],
                f"{dataset_label}: track {track} location",
            )
            legacy.plot_heatmap(
                out_root / "intensity_heatmaps" / f"{stem}.png",
                labels,
                aggregate_intensity[track_index[track]],
                f"{dataset_label}: track {track} normalized intensity",
            )
            legacy.plot_heatmap(
                out_root / "normalized_area_heatmaps" / f"{stem}.png",
                labels,
                aggregate_intensity[track_index[track]],
                f"{dataset_label}: track {track} normalized area density",
            )
            plot_track_trajectory(
                out_root / "trajectories" / f"{stem}_d_vs_pressure.png",
                trajectory,
                f"{dataset_label}: track {track} d(P)",
            )

    write_csv(out_root / "track_pressure_features.csv", pressure_feature_rows)
    write_csv(out_root / "track_summary.csv", summary_rows)
    overall_location = legacy.nanmedian(aggregate_location, axis=0)
    overall_intensity = legacy.nanmedian(aggregate_intensity, axis=0)
    legacy.write_matrix_csv(out_root / "aggregate_location_matrix.csv", labels, overall_location)
    legacy.write_matrix_csv(out_root / "aggregate_intensity_matrix.csv", labels, overall_intensity)
    legacy.write_matrix_csv(out_root / "aggregate_normalized_area_matrix.csv", labels, overall_intensity)
    if make_plots:
        legacy.plot_heatmap(
            out_root / "aggregate_location_heatmap.png",
            labels,
            overall_location,
            f"{dataset_label}: track-median location similarity",
        )
        legacy.plot_heatmap(
            out_root / "aggregate_intensity_heatmap.png",
            labels,
            overall_intensity,
            f"{dataset_label}: track-median normalized intensity similarity",
        )
        legacy.plot_heatmap(
            out_root / "aggregate_normalized_area_heatmap.png",
            labels,
            overall_intensity,
            f"{dataset_label}: track-median normalized area similarity",
        )

    loc_near, loc_far, loc_pairs = matrix_pair_values(location, pressure_registry)
    int_near, int_far, int_pairs = matrix_pair_values(intensity, pressure_registry)
    for row in loc_pairs:
        row["family"] = "location"
    for row in int_pairs:
        row["family"] = "normalized_area"
    write_csv(out_root / "all_pair_scores.csv", loc_pairs + int_pairs)
    return {
        "tracks": len(tracks),
        "raw_observations": len(observations),
        "frame_track_features": len(collapsed),
        "usable_trajectories_ge3_pressures": sum(row["trajectory_status"] == "usable" for row in summary_rows),
        "duplicate_frame_track_features": sum(int(row["duplicate_observation_flag"]) for row in collapsed),
        "location_near_median": float(np.nanmedian(loc_near)) if len(loc_near) else np.nan,
        "location_far_median": float(np.nanmedian(loc_far)) if len(loc_far) else np.nan,
        "location_near_vs_far_auc": legacy.auc_probability(loc_near, loc_far),
        "intensity_near_median": float(np.nanmedian(int_near)) if len(int_near) else np.nan,
        "intensity_far_median": float(np.nanmedian(int_far)) if len(int_far) else np.nan,
        "intensity_near_vs_far_auc": legacy.auc_probability(int_near, int_far),
        "intensity_status_counts": dict(Counter(str(row["intensity_status"]) for row in observations)),
        "summary_rows": summary_rows,
    }


def compare_npz_arrays(current_path: Path, reference_path: Path, keys: list[str]) -> list[dict[str, Any]]:
    current = np.load(current_path)
    reference = np.load(reference_path)
    rows: list[dict[str, Any]] = []
    for key in keys:
        left = np.asarray(current[key])
        right = np.asarray(reference[key])
        shape_equal = left.shape == right.shape
        if left.dtype.kind in "bOUS" or right.dtype.kind in "bOUS":
            exact_equal = shape_equal and np.array_equal(left, right)
            finite_mask_equal = exact_equal
            max_abs = 0.0 if exact_equal else np.nan
            passed = exact_equal
        else:
            finite_mask_equal = shape_equal and np.array_equal(np.isfinite(left), np.isfinite(right))
            if shape_equal and finite_mask_equal and np.any(np.isfinite(left)):
                max_abs = float(np.max(np.abs(left[np.isfinite(left)] - right[np.isfinite(right)])))
            elif shape_equal and finite_mask_equal:
                max_abs = 0.0
            else:
                max_abs = np.nan
            passed = shape_equal and finite_mask_equal and np.isfinite(max_abs) and max_abs <= 1e-7
        rows.append({
            "current": str(current_path.resolve()),
            "reference": str(reference_path.resolve()),
            "array": key,
            "shape_equal": int(shape_equal),
            "finite_mask_equal": int(finite_mask_equal),
            "max_abs_difference": max_abs,
            "passed": int(passed),
        })
    return rows


def powder_reference_comparison(out_root: Path, reference_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel in ("spots", "fit"):
        rows.extend(compare_npz_arrays(
            out_root / channel / "whole_pattern" / "whole_pattern_matrices.npz",
            reference_root / channel / "whole_pattern" / "whole_pattern_matrices.npz",
            ["pressure_gpa", "scan_names", "matrices_by_scan", "aggregate", "valid_frames"],
        ))
        rows.extend(compare_npz_arrays(
            out_root / channel / "across_frames" / "across_frame_matrices.npz",
            reference_root / channel / "across_frames" / "across_frame_matrices.npz",
            ["pressure_gpa", "scan_names", "window_starts_deg", "matrices_by_scan", "aggregate"],
        ))
        rows.extend(compare_npz_arrays(
            out_root / channel / "within_frame" / "within_frame_matrices.npz",
            reference_root / channel / "within_frame" / "within_frame_matrices.npz",
            ["frame_indices", "pressure_gpa", "window_starts_deg", "matrices", "aggregate", "aggregate_by_pressure", "nonoverlap_indices"],
        ))
    return rows


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: compact_metrics(value) if isinstance(value, dict) else value
        for key, value in metrics.items()
        if key != "summary_rows"
    }


def write_report(
    path: Path,
    single_whole: dict[str, Any],
    powder_whole: dict[str, Any],
    single_track: dict[str, Any],
    powder_track: dict[str, Any],
    powder_intensity_qc: dict[str, Any],
    parity_rows: list[dict[str, Any]],
) -> None:
    def r_text(value: Any) -> str:
        return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.3f}"

    parity_passed = all(int(row["passed"]) == 1 for row in parity_rows)
    content = f"""# UOTe legacy-correlation refinement report

## Method lock

- Profile: `{LEGACY_PROFILE}`.
- Whole-pattern and window calculations call `run_uote_xy_handoff_correlations.py` directly.
- No uniform-v2 or uniform-v2.1 module is imported.
- Powder numerical parity against the 2026-07-13 legacy result: **{'PASS' if parity_passed else 'FAIL'}**.
- Legacy preprocessing: row P5/P99 scaling, SG9-SG101 residual, z-score, Pearson.
- Legacy windows: fixed 5 deg width, 1 deg step, plus/minus one neighboring window maximum.
- Legacy near/far definitions: <=1.5 GPa and >=15 GPa.

## Single crystal

- Whole-pattern/ACF uses Initial Reduction only, with 0 deg and 10 deg as separate 11-pressure ladders.
- Isolated 5 deg, 2.4 GPa decompression, and repeated 9.8 GPa exposures are excluded only from the whole-pattern pressure ladders.
- The measured 2theta support ends at 23.8178 deg, so the old fixed-window formulas are retained on 2.0-23.8 deg without extrapolation.
- 0 deg whole-pattern r(correlation, |dP|): {r_text(single_whole['orientation_0deg']['whole_pattern']['corr_vs_pressure_gap_r'])}.
- 10 deg whole-pattern r(correlation, |dP|): {r_text(single_whole['orientation_10deg']['whole_pattern']['corr_vs_pressure_gap_r'])}.
- Per-peak uses all available Masked/kept_obs directly: {single_track['raw_observations']} observations, {single_track['masked_frames']} frames, and {single_track['tracks']} global track identities.
- A global track is followed across 0 deg and 10 deg frames in one matrix; orientation is frame metadata, not a per-peak split key.
- The 2.4 GPa decompression and alternate/repeat 9.8 GPa frames are included in per-peak because both have curated Masked observations.
- Track location is based on measured q/d converted with wavelength 0.4133 A; masked.xy is not treated as a dense continuous spectrum.
- Normalized ROI area is background-subtracted raw ROI excess counts divided by TIFF exposure and effective unmasked ROI pixels.
- Per-peak location and normalized-area values are pairwise similarities, not Pearson correlations.
- Missing Masked observations and singleton tracks remain explicit rather than being filled by redetection.
- Correlation heatmaps show the strict lower triangle only: self-comparison diagonal cells and the duplicate upper triangle are hidden. Exact CSV/NPZ matrices remain unchanged.

## Powder

- Whole-pattern/ACF uses Reduced .xy, spots as the UOTe sample channel and fit as the tungsten/background control.
- 56 scans are processed independently; 1060 cover-accepted frames and 19 pressure values are used.
- Spots whole-pattern r(correlation, |dP|): {r_text(powder_whole['spots']['whole_pattern']['corr_vs_pressure_gap_r'])}.
- Fit-control whole-pattern r(correlation, |dP|): {r_text(powder_whole['fit']['whole_pattern']['corr_vs_pressure_gap_r'])}.
- Spots same-window ACF near/far AUC: {r_text(powder_whole['spots']['across_frames']['near_vs_far_auc'])}.
- Per-peak uses Masked Tracks/kept_obs directly: {powder_track['raw_observations']} observations, {powder_track['tracks']} tracks.
- Powder intensity uses matched spot-observation area divided by n_pixels and the filename D1s exposure token. Raw TIFF exposure tags were not supplied, so intensity results remain explicitly exploratory.
- Powder composite coordinate joins: {powder_intensity_qc['join_status']}.

## Interpretation limits

- Per-track missing Masked observations remain unknown (NaN), not zero; the curated mask is not a full absence survey.
- Location is the primary per-peak result. Normalized ROI area is secondary because ROI population and selection vary by frame.
- Singleton-track coverage is insufficient for cross-frame comparisons. This is a data-coverage result, not a failed peak search.
- High fit-channel correlations are not evidence for UOTe because that channel is tungsten/background dominated.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    single_manifest = args.single_manifest.expanduser().resolve()
    single_raw_root = args.single_raw_root.expanduser().resolve()
    legacy_reference = args.legacy_reference.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix results in non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    make_plots = not args.no_plots

    legacy_path = SCRIPT_DIR / "run_uote_xy_handoff_correlations.py"
    legacy_sha = sha256_file(legacy_path)
    if legacy_sha != LEGACY_EXPECTED_SHA256:
        raise RuntimeError(
            f"Legacy runner SHA changed: expected {LEGACY_EXPECTED_SHA256}, got {legacy_sha}"
        )

    single_root = data_root / "Single Crystal (Cell 29)"
    patterns_dir = single_root / "Initial Reduction" / "patterns"
    masked_single = single_root / "Masked"
    single_metadata, single_input_audit = load_single_metadata(single_manifest, patterns_dir)
    write_csv(out_dir / "inputs" / "single_frame_registry.csv", single_input_audit)
    single_whole, selected_single = run_single_whole(
        out_dir / "single_crystal" / "whole_and_windows",
        single_metadata,
        patterns_dir,
        make_plots,
    )
    write_csv(out_dir / "inputs" / "single_whole_selected.csv", selected_single)

    powder_root = data_root / "Powder Scan"
    powder_reduced = powder_root / "Reduced .xy"
    powder_whole, powder_frames, powder_manifest_rows, powder_pressures, powder_scans = run_powder_whole(
        out_dir / "powder" / "whole_and_windows",
        powder_reduced,
        make_plots,
    )
    powder_input_audit = [
        {
            **row,
            "included": int(row["cover_excluded"] == "0"),
            "parsed_pressure_step": re.search(r"\\(P\d+)_", row["filename"]).group(1)
            if re.search(r"\\(P\d+)_", row["filename"])
            else "",
        }
        for row in powder_manifest_rows
    ]
    write_csv(out_dir / "inputs" / "powder_frame_registry.csv", powder_input_audit)

    single_observations, single_roi_qc = extract_single_track_observations(
        masked_single / "kept_obs.csv",
        masked_single,
        single_raw_root,
        single_metadata,
    )
    write_csv(out_dir / "validation" / "single_roi_extraction_qc.csv", single_roi_qc)
    single_track_combined = analyze_single_tracks_across_frames(
        out_dir / "single_crystal" / "per_peak_all_frames",
        single_observations,
        single_metadata,
        make_plots,
    )

    powder_observations, powder_intensity_qc = extract_powder_track_observations(
        powder_root / "Masked Tracks" / "kept_obs.csv",
        powder_root / "Track Analysis" / "spot_observations.csv",
        powder_root / "Track Analysis" / "spot_tracks.csv",
        powder_reduced / "manifest.csv",
    )
    powder_track_metrics = analyze_direct_tracks(
        out_dir / "powder" / "per_peak",
        "powder_masked_tracks",
        powder_observations,
        powder_pressures,
        powder_scans,
        make_plots,
    )

    parity_rows = powder_reference_comparison(
        out_dir / "powder" / "whole_and_windows",
        legacy_reference,
    )
    write_csv(out_dir / "validation" / "powder_legacy_parity.csv", parity_rows)

    actual_single_selection = {
        orientation: sorted(
            (int(row["frame"]), float(row["pressure_GPa"]))
            for row in selected_single
            if row["orientation"] == orientation
        )
        for orientation in EXPECTED_SINGLE_SELECTION
    }
    expected_single_selection = {
        orientation: sorted(values)
        for orientation, values in EXPECTED_SINGLE_SELECTION.items()
    }
    actual_single_masked_frames = sorted({int(row["frame"]) for row in single_observations})

    validation_checks = [
        {"check": "legacy_runner_sha256", "value": legacy_sha, "passed": int(legacy_sha == LEGACY_EXPECTED_SHA256)},
        {"check": "single_whole_frames", "value": len(selected_single), "passed": int(len(selected_single) == 22)},
        {"check": "single_orientation_0deg_frames", "value": sum(row["orientation"] == "orientation_0deg" for row in selected_single), "passed": int(sum(row["orientation"] == "orientation_0deg" for row in selected_single) == 11)},
        {"check": "single_orientation_10deg_frames", "value": sum(row["orientation"] == "orientation_10deg" for row in selected_single), "passed": int(sum(row["orientation"] == "orientation_10deg" for row in selected_single) == 11)},
        {"check": "single_orientation_0deg_exact_frame_pressure_selection", "value": actual_single_selection["orientation_0deg"], "passed": int(actual_single_selection["orientation_0deg"] == expected_single_selection["orientation_0deg"])},
        {"check": "single_orientation_10deg_exact_frame_pressure_selection", "value": actual_single_selection["orientation_10deg"], "passed": int(actual_single_selection["orientation_10deg"] == expected_single_selection["orientation_10deg"])},
        {"check": "single_masked_observations_all_available", "value": len(single_observations), "passed": int(len(single_observations) == EXPECTED_SINGLE_MASKED_OBSERVATIONS)},
        {"check": "single_masked_exact_all_frames", "value": actual_single_masked_frames, "passed": int(actual_single_masked_frames == sorted(EXPECTED_SINGLE_MASKED_FRAMES))},
        {"check": "single_global_tracks", "value": single_track_combined["tracks"], "passed": int(single_track_combined["tracks"] == EXPECTED_SINGLE_MASKED_TRACKS)},
        {"check": "single_global_frame_track_features", "value": single_track_combined["frame_track_features"], "passed": int(single_track_combined["frame_track_features"] == EXPECTED_SINGLE_FRAME_TRACK_FEATURES)},
        {"check": "single_global_duplicate_frame_track_groups", "value": single_track_combined["duplicate_frame_track_features"], "passed": int(single_track_combined["duplicate_frame_track_features"] == EXPECTED_SINGLE_DUPLICATE_FRAME_TRACKS)},
        {"check": "single_global_duplicate_extra_rows", "value": single_track_combined["duplicate_extra_observations"], "passed": int(single_track_combined["duplicate_extra_observations"] == 12)},
        {"check": "single_global_tracks_ge2_frames", "value": single_track_combined["comparable_tracks_ge2_frames"], "passed": int(single_track_combined["comparable_tracks_ge2_frames"] == EXPECTED_SINGLE_COMPARABLE_TRACKS)},
        {"check": "single_global_tracks_ge3_frames", "value": single_track_combined["usable_trajectories_ge3_frames"], "passed": int(single_track_combined["usable_trajectories_ge3_frames"] == EXPECTED_SINGLE_USABLE_TRACKS)},
        {"check": "single_global_singleton_tracks", "value": single_track_combined["singleton_tracks"], "passed": int(single_track_combined["singleton_tracks"] == EXPECTED_SINGLE_SINGLETON_TRACKS)},
        {"check": "single_global_location_unique_pairs", "value": single_track_combined["location_unique_pairs"], "passed": int(single_track_combined["location_unique_pairs"] == EXPECTED_SINGLE_UNIQUE_PAIRS)},
        {"check": "single_global_normalized_area_unique_pairs", "value": single_track_combined["normalized_area_unique_pairs"], "passed": int(single_track_combined["normalized_area_unique_pairs"] == EXPECTED_SINGLE_UNIQUE_PAIRS)},
        {"check": "single_global_cross_orientation_tracks", "value": single_track_combined["cross_orientation_tracks"], "passed": int(single_track_combined["cross_orientation_tracks"] == 24)},
        {"check": "single_global_cross_orientation_pairs", "value": single_track_combined["cross_orientation_location_pairs"], "passed": int(single_track_combined["cross_orientation_location_pairs"] == 134)},
        {"check": "single_global_track18_support", "value": single_track_combined["track18_frame_support"], "passed": int(single_track_combined["track18_frame_support"] == 11)},
        {"check": "single_global_track18_pairs", "value": single_track_combined["track18_unique_pairs"], "passed": int(single_track_combined["track18_unique_pairs"] == 55)},
        {"check": "single_global_matrices_symmetric", "value": single_track_combined["matrices_symmetric"], "passed": int(single_track_combined["matrices_symmetric"])},
        {"check": "single_global_matrix_diagonal", "value": single_track_combined["matrix_diagonal_matches_observed"], "passed": int(single_track_combined["matrix_diagonal_matches_observed"])},
        {"check": "single_global_matrix_unit_interval", "value": single_track_combined["matrix_scores_in_unit_interval"], "passed": int(single_track_combined["matrix_scores_in_unit_interval"])},
        {"check": "correlation_heatmap_triangle_policy", "value": single_track_combined["heatmap_triangle_policy"], "passed": int(single_track_combined["heatmap_triangle_policy"] == "strict_lower_only_no_diagonal")},
        {"check": "correlation_heatmap_diagonal_and_upper_hidden", "value": single_track_combined["heatmap_diagonal_and_upper_hidden"], "passed": int(single_track_combined["heatmap_diagonal_and_upper_hidden"])},
        {"check": "correlation_heatmap_lower_triangle_preserved", "value": single_track_combined["heatmap_lower_triangle_preserved"], "passed": int(single_track_combined["heatmap_lower_triangle_preserved"])},
        {"check": "single_global_paired_heatmaps", "value": single_track_combined["paired_heatmaps"], "passed": int((not make_plots) or single_track_combined["paired_heatmaps"] == EXPECTED_SINGLE_MASKED_TRACKS)},
        {"check": "single_global_location_heatmaps", "value": single_track_combined["location_heatmaps"], "passed": int((not make_plots) or single_track_combined["location_heatmaps"] == EXPECTED_SINGLE_MASKED_TRACKS)},
        {"check": "single_global_normalized_area_heatmaps", "value": single_track_combined["normalized_area_heatmaps"], "passed": int((not make_plots) or single_track_combined["normalized_area_heatmaps"] == EXPECTED_SINGLE_MASKED_TRACKS)},
        {"check": "single_global_gallery_pages", "value": single_track_combined["gallery_pages"], "passed": int((not make_plots) or single_track_combined["gallery_pages"] == 13)},
        {"check": "single_roi_qc_frames", "value": len(single_roi_qc), "passed": int(len(single_roi_qc) == len(EXPECTED_SINGLE_MASKED_FRAMES))},
        {"check": "single_roi_mask_min_jaccard", "value": min(row["roi_mask_jaccard"] for row in single_roi_qc), "passed": int(min(row["roi_mask_jaccard"] for row in single_roi_qc) >= 0.99)},
        {"check": "single_exposure_all_from_tiff", "value": sorted({row["exposure_s"] for row in single_observations}), "passed": int(all(row["exposure_source"].startswith("TIFF") for row in single_observations))},
        {"check": "powder_frames", "value": len(powder_frames), "passed": int(len(powder_frames) == 1060)},
        {"check": "powder_scans", "value": len(powder_scans), "passed": int(len(powder_scans) == 56)},
        {"check": "powder_pressures", "value": len(powder_pressures), "passed": int(len(powder_pressures) == 19)},
        {"check": "powder_masked_observations", "value": len(powder_observations), "passed": int(len(powder_observations) == 167)},
        {"check": "powder_composite_join_complete", "value": powder_intensity_qc["matched_rows"], "passed": int(powder_intensity_qc["matched_rows"] == powder_intensity_qc["kept_rows"])},
        {"check": "powder_exposure_tokens_parsed", "value": powder_intensity_qc["exposure_seconds"], "passed": int(powder_intensity_qc["all_exposure_tokens_parsed"])},
        {"check": "powder_legacy_parity", "value": sum(int(row["passed"]) for row in parity_rows), "passed": int(all(int(row["passed"]) == 1 for row in parity_rows))},
    ]
    write_csv(out_dir / "validation" / "validation_checks.csv", validation_checks)
    validation = {
        "passed": all(int(row["passed"]) == 1 for row in validation_checks),
        "checks": validation_checks,
        "powder_legacy_parity": parity_rows,
        "powder_intensity_qc": powder_intensity_qc,
        "single_roi_qc": single_roi_qc,
    }
    (out_dir / "validation" / "validation_report.json").write_text(
        json.dumps(json_ready(validation), indent=2),
        encoding="utf-8",
    )

    run_manifest = {
        "profile": LEGACY_PROFILE,
        "legacy_runner": str(legacy_path.resolve()),
        "legacy_runner_sha256": legacy_sha,
        "legacy_reference": str(legacy_reference),
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "parameters": {
            "row_normalization": "subtract P5; divide shifted row by P99",
            "residual": "Savitzky-Golay(9,2) - Savitzky-Golay(101,2); row z-score",
            "whole_pattern": "same-scan Pearson",
            "window_width_deg": WINDOW_WIDTH_DEG,
            "window_step_deg": WINDOW_STEP_DEG,
            "window_shift_tolerance_deg": WINDOW_SHIFT_TOLERANCE_DEG,
            "grid_step_deg": GRID_STEP_DEG,
            "near_gap_GPa": NEAR_GAP_GPA,
            "far_gap_GPa": FAR_GAP_GPA,
            "location_similarity": "clip(1-|delta_2theta|/0.06deg,0,1)",
            "normalized_area_similarity": "min/max on background-subtracted ROI excess counts / exposure / effective pixels",
            "missing_masked_observation": "NaN (unknown; no redetection)",
            "single_per_peak_grouping": "global track across all available Masked frames; orientation retained as frame metadata",
            "correlation_heatmap_display": "strict lower triangle only; diagonal and upper triangle hidden; exact numerical matrices unchanged",
        },
        "single_crystal": {
            "whole_and_windows": compact_metrics(single_whole),
            "per_peak_all_frames": compact_metrics(single_track_combined),
            "wavelength_A": SINGLE_WAVELENGTH_A,
            "masked_frames_not_in_whole_pattern": sorted(
                int(row["frame"])
                for row in single_metadata.values()
                if int(row["frame"]) in set(actual_single_masked_frames)
                and not int(row["included_whole_pattern"])
            ),
        },
        "powder": {
            "whole_and_windows": compact_metrics(powder_whole),
            "per_track": compact_metrics(powder_track_metrics),
            "wavelength_A": POWDER_WAVELENGTH_A,
            "intensity_qc": powder_intensity_qc,
        },
        "validation_passed": validation["passed"],
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(json_ready(run_manifest), indent=2),
        encoding="utf-8",
    )
    write_report(
        out_dir / "REPORT.md",
        single_whole,
        powder_whole,
        single_track_combined,
        powder_track_metrics,
        powder_intensity_qc,
        parity_rows,
    )
    print(f"Wrote legacy-method refinement correlations to {out_dir}")
    if not validation["passed"]:
        raise RuntimeError(f"Validation failed; inspect {out_dir / 'validation' / 'validation_checks.csv'}")


if __name__ == "__main__":
    main()
