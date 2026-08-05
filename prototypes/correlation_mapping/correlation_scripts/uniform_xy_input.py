#!/usr/bin/env python3
"""Generic manifest and XY adapters for the frozen uniform correlation runner."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CHANNEL_FOLDERS = {"spots": "spots_channel", "fit": "fit_channel"}


@dataclass(frozen=True)
class FrameInput:
    frame: int
    scan: str
    pressure: float
    pressure_index: int
    original_filename: str


def read_handoff_manifest(path: Path, max_scans: int | None = None) -> tuple[list[FrameInput], list[float], list[str], list[dict[str, str]]]:
    """Read the current handoff while exposing a generic frame/scan/condition model."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"frame", "scan", "pressure_GPa", "cover_excluded", "filename"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    scans = sorted({row["scan"] for row in rows})
    if max_scans is not None:
        scans = scans[: max(1, int(max_scans))]
    selected = [row for row in rows if row["scan"] in scans]
    included = [row for row in selected if str(row["cover_excluded"]).strip() == "0"]
    pressures = sorted({float(row["pressure_GPa"]) for row in included})
    pressure_index = {value: index for index, value in enumerate(pressures)}
    frames = [
        FrameInput(
            frame=int(row["frame"]),
            scan=row["scan"],
            pressure=float(row["pressure_GPa"]),
            pressure_index=pressure_index[float(row["pressure_GPa"])],
            original_filename=row["filename"],
        )
        for row in included
    ]
    frames.sort(key=lambda item: (item.scan, item.pressure, item.frame))
    return frames, pressures, scans, selected


def resolve_channel_paths(handoff_dir: Path, frames: list[FrameInput], channel: str) -> list[Path]:
    if channel not in CHANNEL_FOLDERS:
        raise ValueError(f"unsupported channel {channel!r}; expected one of {sorted(CHANNEL_FOLDERS)}")
    paths: list[Path] = []
    failures: list[str] = []
    for frame in frames:
        folder = handoff_dir / CHANNEL_FOLDERS[channel] / frame.scan
        exact = folder / f"frame_{frame.frame:04d}.xy"
        matches = [exact] if exact.is_file() else sorted(folder.glob(f"frame_{frame.frame:04d}_*GPa.xy"))
        if len(matches) != 1:
            failures.append(f"frame {frame.frame}: {len(matches)} matching files under {folder}")
        else:
            paths.append(matches[0])
    if failures:
        raise FileNotFoundError("; ".join(failures[:8]))
    return paths


def read_xy_clean(path: Path, minimum_points: int = 128) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Load a two-column XY file while preserving raw values for audited cleanup.

    The scientific preprocessor performs the one authoritative cleanup pass so
    its QC record can report non-finite pairs, original monotonicity, and
    duplicate coordinates.  This loader only verifies that enough finite,
    unique coordinates exist to make that cleanup possible.
    """

    metadata: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if ":" in body:
                key, value = body.split(":", 1)
                metadata[key.strip()] = value.strip()
    data = np.loadtxt(path, comments="#", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} is not a two-column XY file")
    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    finite_count = int(np.count_nonzero(finite))
    unique_count = int(np.unique(x[finite]).size)
    if finite_count < minimum_points or unique_count < minimum_points:
        raise ValueError(
            f"{path} has only {finite_count} finite / {unique_count} unique points; "
            f"minimum is {minimum_points}"
        )
    return x, y, metadata


def common_coverage_grid(
    x_values: list[np.ndarray],
    minimum_coverage_fraction: float = 0.9,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the largest contiguous grid covered by the requested frame fraction.

    The grid step is the coarsest native median step, which avoids claiming a
    resolution finer than any included input.  Interpolation outside a file's
    native range is prohibited by :func:`resample_no_extrapolation`.
    """

    if not x_values:
        raise ValueError("no x axes were supplied")
    native_steps = np.asarray([np.median(np.diff(x)) for x in x_values], dtype=float)
    if np.any(~np.isfinite(native_steps)) or np.any(native_steps <= 0):
        raise ValueError("invalid native x step")
    step = float(np.max(native_steps))
    lower = float(min(x[0] for x in x_values))
    upper = float(max(x[-1] for x in x_values))
    candidate = np.arange(lower, upper + step * 0.25, step, dtype=float)
    coverage = np.zeros(len(candidate), dtype=np.int32)
    for x in x_values:
        coverage += (candidate >= x[0] - step * 1e-6) & (candidate <= x[-1] + step * 1e-6)
    fraction = coverage / len(x_values)
    valid = fraction >= float(minimum_coverage_fraction)
    if not np.any(valid):
        raise ValueError("no common x interval reaches the requested coverage")
    padded = np.r_[False, valid, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    runs = [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]
    start, stop = max(runs, key=lambda item: item[1] - item[0])
    grid = candidate[start:stop]
    if len(grid) < 128:
        raise ValueError(f"common coverage contains only {len(grid)} points")
    return grid, {
        "minimum_coverage_fraction": float(minimum_coverage_fraction),
        "actual_minimum_coverage_fraction": float(np.min(fraction[start:stop])),
        "grid_step_deg": step,
        "analysis_min_deg": float(grid[0]),
        "analysis_max_deg": float(grid[-1]),
        "analysis_span_deg": float(grid[-1] - grid[0]),
        "analysis_points": int(len(grid)),
        "native_step_min_deg": float(np.min(native_steps)),
        "native_step_median_deg": float(np.median(native_steps)),
        "native_step_max_deg": float(np.max(native_steps)),
    }


def resample_no_extrapolation(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.full(len(grid), np.nan, dtype=float)
    keep = (grid >= x[0]) & (grid <= x[-1])
    values[keep] = np.interp(grid[keep], x, y)
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_row_indices(frames: list[FrameInput], scans: list[str]) -> dict[str, list[int]]:
    result = {scan: [] for scan in scans}
    for index, frame in enumerate(frames):
        result[frame.scan].append(index)
    for scan in result:
        result[scan].sort(key=lambda index: (frames[index].pressure, frames[index].frame))
    return result
