#!/usr/bin/env python3
"""Build interactive 3D viewers for the legacy four-family XRD correlations.

The source correlation values are read from the existing legacy suite without
recalculation.  The first three viewers extract only within-cell adjacent
pressure pairs.  The fourth viewer stacks each frame's complete
window-by-window matrix as a pressure-indexed 3D correlation cube.

Outputs:
  01_roi_area_correlation_3d.html/.png
  02_per_peak_location_correlation_3d.html/.png
  03_window_across_frames_correlation_3d.html/.png
  04_window_within_frame_correlation_3d.html/.png
  README.md, artifact_index.csv, run_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

try:
    from plotly.offline import get_plotlyjs
except ImportError:  # Optional dependency; matrix/series helpers remain usable.
    get_plotlyjs = None

import compare_integrated_peaks as cip


SCRIPT_VERSION = "1.2.1"
DEFAULT_SUITE = ROOT / "outputs" / "correlation_suite_20260621_high_recall_scored_v2"
DEFAULT_OUTPUT = ROOT / "correlations" / "results" / "legacy_correlation_3d_waterfalls_20260724"
DEFAULT_INPUTS = [
    ROOT / "Data" / "Cell_14_integrated",
    ROOT / "Data" / "Cell_29_integrated",
]
PEAK_ROOT_NAME = "01_per_peak_frame_correlation"
ACROSS_ROOT_NAME = "02_same_window_acf_across_frames"
WITHIN_ROOT_NAME = "03_single_frame_window_acf"
ZERO_AREA_EPS = 1e-12
RAW_GRID_STEP = 0.04
VIRIDIS = "Viridis"
DIVERGING = "RdBu_r"


@dataclass(frozen=True)
class FeatureTable:
    labels: list[str]
    columns: list[float]
    values: np.ndarray


@dataclass(frozen=True)
class MatrixData:
    labels: list[str]
    full: np.ndarray


@dataclass(frozen=True)
class PatternInfo:
    index: int
    label: str
    path: Path
    cell: str
    pressure: float
    decomp: bool
    two_theta: np.ndarray
    intensity: np.ndarray
    normalized: np.ndarray


@dataclass(frozen=True)
class PairInfo:
    previous: int
    current: int
    label: str
    kind: str = "adjacent compression"


@dataclass(frozen=True)
class SeriesInfo:
    key: str
    label: str
    frame_indices: list[int]
    pairs: list[PairInfo]


@dataclass(frozen=True)
class DisplayRow:
    current: int
    previous: int | None
    label: str
    kind: str
    delta_p_gpa: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plotly-js",
        choices=("inline", "cdn"),
        default="inline",
        help="Embed Plotly for offline sharing (default) or load it from the CDN.",
    )
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def finite_or_none(value: float | np.floating[Any]) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def rounded_matrix(matrix: np.ndarray, digits: int = 6) -> list[list[float | None]]:
    return [
        [None if not np.isfinite(value) else round(float(value), digits) for value in row]
        for row in np.asarray(matrix, dtype=float)
    ]


def read_feature_table(path: Path) -> FeatureTable:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"Feature table is empty: {path}")
    labels = [row[0] for row in rows[1:]]
    columns = [float(value) for value in rows[0][1:]]
    values = np.full((len(labels), len(columns)), np.nan, dtype=float)
    for i, row in enumerate(rows[1:]):
        for j, value in enumerate(row[1 : len(columns) + 1]):
            if value.strip():
                values[i, j] = float(value)
    return FeatureTable(labels=labels, columns=columns, values=values)


def read_lower_triangle_matrix(path: Path) -> MatrixData:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"Matrix is empty: {path}")
    column_labels = rows[0][1:]
    row_labels = [row[0] for row in rows[1:]]
    if column_labels != row_labels:
        raise ValueError(f"Row/column labels differ: {path}")
    size = len(row_labels)
    lower = np.full((size, size), np.nan, dtype=float)
    for i, row in enumerate(rows[1:]):
        for j, value in enumerate(row[1 : size + 1]):
            if value.strip():
                lower[i, j] = float(value)
    full = lower.copy()
    copy_from_transpose = ~np.isfinite(full) & np.isfinite(lower.T)
    full[copy_from_transpose] = lower.T[copy_from_transpose]
    return MatrixData(labels=row_labels, full=full)


def load_patterns(inputs: Sequence[Path], labels: Sequence[str]) -> list[PatternInfo]:
    files = cip.discover_xy_files(list(inputs))
    if not files:
        raise FileNotFoundError("No integrated XY files were found.")
    loader_args = SimpleNamespace(min_two_theta=2.0, max_two_theta=None)
    patterns = [cip.load_pattern(path, loader_args) for path in files]
    cip.ensure_unique_pattern_labels(patterns)
    by_label = {pattern.label: pattern for pattern in patterns}
    missing = [label for label in labels if label not in by_label]
    extras = sorted(set(by_label) - set(labels))
    if missing or extras:
        raise ValueError(f"XY/suite labels differ. Missing={missing}; extras={extras}")

    output: list[PatternInfo] = []
    for index, label in enumerate(labels):
        pattern = by_label[label]
        parent = pattern.path.parent.name
        if "Cell_14" in parent:
            cell = "Cell_14"
        elif "Cell_29" in parent:
            cell = "Cell_29"
        else:
            raise ValueError(f"Cannot infer cell from {pattern.path}")
        if pattern.pressure_gpa is None or not np.isfinite(pattern.pressure_gpa):
            raise ValueError(f"Missing pressure for {pattern.path}")
        output.append(
            PatternInfo(
                index=index,
                label=label,
                path=Path(pattern.path).resolve(),
                cell=cell,
                pressure=float(pattern.pressure_gpa),
                decomp="decomp" in label.lower() or "decomp" in pattern.path.name.lower(),
                two_theta=np.asarray(pattern.two_theta, dtype=float),
                intensity=np.asarray(pattern.intensity, dtype=float),
                normalized=np.asarray(cip.normalize_for_pattern(pattern.intensity), dtype=float),
            )
        )
    return output


def build_series(patterns: Sequence[PatternInfo]) -> list[SeriesInfo]:
    def compression(cell: str) -> list[int]:
        return [
            item.index
            for item in sorted(
                (pattern for pattern in patterns if pattern.cell == cell and not pattern.decomp),
                key=lambda pattern: pattern.pressure,
            )
        ]

    def adjacent(indices: Sequence[int]) -> list[PairInfo]:
        pairs: list[PairInfo] = []
        for previous, current in zip(indices[:-1], indices[1:]):
            a = patterns[previous]
            b = patterns[current]
            pairs.append(
                PairInfo(
                    previous=previous,
                    current=current,
                    label=f"{a.pressure:g} → {b.pressure:g} GPa",
                )
            )
        return pairs

    cell14 = compression("Cell_14")
    cell29 = compression("Cell_29")
    decomp = sorted(
        (pattern for pattern in patterns if pattern.cell == "Cell_29" and pattern.decomp),
        key=lambda pattern: pattern.index,
    )
    decomp_indices = [pattern.index for pattern in decomp]
    decomp_pairs: list[PairInfo] = []
    if decomp_indices and cell29:
        previous = cell29[-1]
        for current in decomp_indices:
            a = patterns[previous]
            b = patterns[current]
            decomp_pairs.append(
                PairInfo(
                    previous=previous,
                    current=current,
                    label=f"{a.pressure:g} → {b.pressure:g} GPa decomp",
                    kind="compression-to-decompression branch step",
                )
            )
            previous = current

    return [
        SeriesInfo(
            key="cell14_compression",
            label="Cell_14 compression",
            frame_indices=cell14,
            pairs=adjacent(cell14),
        ),
        SeriesInfo(
            key="cell29_compression",
            label="Cell_29 compression",
            frame_indices=cell29,
            pairs=adjacent(cell29),
        ),
        SeriesInfo(
            key="cell29_decompression",
            label="Cell_29 decompression reference",
            frame_indices=decomp_indices,
            pairs=decomp_pairs,
        ),
    ]


def build_display_rows(
    item: SeriesInfo,
    patterns: Sequence[PatternInfo],
) -> list[DisplayRow]:
    """Return the measured-pressure rows shown by the first three viewers.

    Compression series retain their complete frame sequence.  The first frame
    is an explicit raw-XRD-only baseline because it has no previous frame.
    Decompression stays on its separate reference branch and therefore keeps
    its stored compression-to-decompression pair row.
    """

    if item.key in {"cell14_compression", "cell29_compression"}:
        if not item.frame_indices:
            return []
        pairs_by_current = {pair.current: pair for pair in item.pairs}
        rows: list[DisplayRow] = []
        for position, current in enumerate(item.frame_indices):
            current_pattern = patterns[current]
            if position == 0:
                rows.append(
                    DisplayRow(
                        current=current,
                        previous=None,
                        label=(
                            f"{current_pattern.pressure:g} GPa — "
                            "Baseline / no previous correlation"
                        ),
                        kind="baseline",
                        delta_p_gpa=None,
                    )
                )
                continue
            pair = pairs_by_current.get(current)
            if pair is None:
                raise ValueError(
                    f"Missing adjacent pair for {item.key} frame "
                    f"{current_pattern.label}"
                )
            rows.append(
                DisplayRow(
                    current=current,
                    previous=pair.previous,
                    label=pair.label,
                    kind=pair.kind,
                    delta_p_gpa=(
                        current_pattern.pressure - patterns[pair.previous].pressure
                    ),
                )
            )
        return rows

    return [
        DisplayRow(
            current=pair.current,
            previous=pair.previous,
            label=pair.label,
            kind=pair.kind,
            delta_p_gpa=(
                patterns[pair.current].pressure - patterns[pair.previous].pressure
            ),
        )
        for pair in item.pairs
    ]


def common_raw_grid(patterns: Sequence[PatternInfo]) -> np.ndarray:
    lower = min(float(np.nanmin(pattern.two_theta)) for pattern in patterns)
    upper = max(float(np.nanmax(pattern.two_theta)) for pattern in patterns)
    return np.arange(lower, upper + RAW_GRID_STEP / 2.0, RAW_GRID_STEP)


def raw_profile(pattern: PatternInfo, grid: np.ndarray) -> list[float | None]:
    values = np.full(grid.shape, np.nan, dtype=float)
    covered = (grid >= float(np.nanmin(pattern.two_theta))) & (
        grid <= float(np.nanmax(pattern.two_theta))
    )
    values[covered] = np.interp(
        grid[covered], pattern.two_theta, pattern.normalized
    )
    values = np.clip(values, -0.05, 1.2)
    return [
        None if not np.isfinite(value) else round(float(value), 5)
        for value in values
    ]


def peak_index_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_peak_payload(
    peak_root: Path,
    patterns: Sequence[PatternInfo],
    series: Sequence[SeriesInfo],
    grid: np.ndarray,
    area_features: FeatureTable,
    presence_features: FeatureTable,
    position_features: FeatureTable,
    *,
    position: bool,
) -> dict[str, Any]:
    index_name = "per_peak_position_map_index.csv" if position else "per_peak_map_index.csv"
    rows = peak_index_rows(peak_root / index_name)
    centers = [float(row["group_two_theta"]) for row in rows]
    groups = [int(row["peak_group"]) for row in rows]
    frame_counts = [int(row["frame_count"]) for row in rows]
    matrix_paths = [peak_root / row["matrix"] for row in rows]
    if len(rows) != len(area_features.columns):
        raise ValueError("Peak index and feature-table column count differ.")
    if not np.allclose(centers, area_features.columns, atol=1e-4):
        raise ValueError("Peak centers differ between index and feature table.")

    matrices: list[np.ndarray] = []
    for path in matrix_paths:
        matrix = read_lower_triangle_matrix(path)
        if matrix.labels != area_features.labels:
            raise ValueError(f"Frame order differs in {path}")
        matrices.append(matrix.full)

    datasets: dict[str, Any] = {}
    for item in series:
        display_rows = build_display_rows(item, patterns)
        score_rows: list[list[float | None]] = []
        for display_row in display_rows:
            scores: list[float | None] = []
            for column, matrix in enumerate(matrices):
                if display_row.previous is None:
                    scores.append(None)
                    continue
                score = matrix[display_row.current, display_row.previous]
                exclude_detected_zero_area_pair = bool(
                    not position
                    and np.isfinite(score)
                    and presence_features.values[display_row.previous, column] > 0
                    and presence_features.values[display_row.current, column] > 0
                    and area_features.values[display_row.previous, column] <= ZERO_AREA_EPS
                    and area_features.values[display_row.current, column] <= ZERO_AREA_EPS
                )
                scores.append(
                    None if exclude_detected_zero_area_pair else finite_or_none(score)
                )
            score_rows.append(scores)

        datasets[item.key] = {
            "label": item.label,
            "pair_labels": [row.label for row in display_rows],
            "pair_kinds": [row.kind for row in display_rows],
            "pair_pressures": [
                patterns[row.current].pressure for row in display_rows
            ],
            "pair_previous_labels": [
                None if row.previous is None else patterns[row.previous].label
                for row in display_rows
            ],
            "pair_current_labels": [
                patterns[row.current].label for row in display_rows
            ],
            "pair_delta_p_gpa": [row.delta_p_gpa for row in display_rows],
            "baseline_mask": [
                row.previous is None for row in display_rows
            ],
            "scores": score_rows,
            "raw_profiles": [
                raw_profile(patterns[row.current], grid) for row in display_rows
            ],
            "finite_score_count": int(
                sum(value is not None for row in score_rows for value in row)
            ),
        }

    return {
        "kind": "peak_position" if position else "peak_area",
        "title": (
            "Per-Peak Location Correlation — Adjacent Pressure Pairs"
            if position
            else "ROI Area Correlation — Adjacent Pressure Pairs"
        ),
        "subtitle": (
            "All measured pressures are shown; the first compression row is a raw-XRD-only "
            "baseline, followed by stored 0–1 detected-position similarities."
            if position
            else "All measured pressures are shown; the first compression row is a raw-XRD-only "
            "baseline, followed by stored 0–1 ROI-area similarities."
        ),
        "x_label": "Legacy peak-group 2θ (degrees)",
        "z_label": "Location similarity (0–1)" if position else "ROI-area similarity (0–1)",
        "score_min": 0.0,
        "score_max": 1.0,
        "colorscale": VIRIDIS,
        "groups": groups,
        "x": [round(center, 6) for center in centers],
        "frame_counts": frame_counts,
        "position_ranges": [
            finite_or_none(
                float(np.nanmax(position_features.values[:, index]) - np.nanmin(position_features.values[:, index]))
            )
            if np.any(np.isfinite(position_features.values[:, index]))
            else None
            for index in range(len(centers))
        ],
        "raw_grid": [round(float(value), 4) for value in grid],
        "datasets": datasets,
        "method": (
            "s = clip(1 − |Δ2θ| / 0.06°, 0, 1), using the stored legacy matrices."
            if position
            else "s = 1 − |A₁−A₂| / max(A₁,A₂), using the stored legacy matrices."
        ),
        "caveat": (
            "The 0.02° legacy grouping can split pressure-moving peaks; blank means unknown. "
            "The first compression row is a raw-XRD-only baseline; adjacent means adjacent "
            "measured frames and ΔP is not constant."
            if position
            else "Pairs detected in both frames but having zero ROI area in both are "
            "treated as missing and are not plotted. The first compression row is a "
            "raw-XRD-only baseline; adjacent means adjacent measured frames and ΔP is "
            "not constant."
        ),
    }


def read_same_window_rows(root: Path) -> list[dict[str, str]]:
    with (root / "same_window_summary.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_across_payload(
    root: Path,
    patterns: Sequence[PatternInfo],
    series: Sequence[SeriesInfo],
    grid: np.ndarray,
) -> dict[str, Any]:
    summary = read_same_window_rows(root)
    starts = [float(row["start_deg"]) for row in summary]
    ends = [float(row["end_deg"]) for row in summary]
    centers = [(start + end) / 2.0 for start, end in zip(starts, ends)]
    labels = [f"{start:.1f}–{end:.1f}°" for start, end in zip(starts, ends)]
    matrices: list[np.ndarray] = []
    sources: list[str] = []
    frame_labels: list[str] | None = None
    for start, end in zip(starts, ends):
        stem = f"window_{start:.1f}_{end:.1f}_same_window_acf.csv"
        path = root / "matrices" / stem
        matrix = read_lower_triangle_matrix(path)
        if frame_labels is None:
            frame_labels = matrix.labels
        elif frame_labels != matrix.labels:
            raise ValueError(f"Frame order differs in {path}")
        matrices.append(matrix.full)
        sources.append(str(path.resolve()))
    if frame_labels != [pattern.label for pattern in patterns]:
        raise ValueError("Across-frame ACF labels differ from XY labels.")

    datasets: dict[str, Any] = {}
    for item in series:
        display_rows = build_display_rows(item, patterns)
        scores: list[list[float | None]] = []
        for display_row in display_rows:
            if display_row.previous is None:
                scores.append([None for _ in matrices])
            else:
                scores.append(
                    [
                        finite_or_none(
                            matrix[display_row.current, display_row.previous]
                        )
                        for matrix in matrices
                    ]
                )
        datasets[item.key] = {
            "label": item.label,
            "pair_labels": [row.label for row in display_rows],
            "pair_kinds": [row.kind for row in display_rows],
            "pair_pressures": [
                patterns[row.current].pressure for row in display_rows
            ],
            "pair_previous_labels": [
                None if row.previous is None else patterns[row.previous].label
                for row in display_rows
            ],
            "pair_current_labels": [
                patterns[row.current].label for row in display_rows
            ],
            "pair_delta_p_gpa": [row.delta_p_gpa for row in display_rows],
            "baseline_mask": [
                row.previous is None for row in display_rows
            ],
            "scores": scores,
            "raw_profiles": [
                raw_profile(patterns[row.current], grid) for row in display_rows
            ],
            "finite_score_count": int(sum(value is not None for row in scores for value in row)),
        }
    return {
        "kind": "same_window_across_frames",
        "title": "Same Window Across Frames — Adjacent Pressure Pairs",
        "subtitle": (
            "All measured pressures are shown; the first compression row is a raw-XRD-only "
            "baseline, followed by stored 5°-window ACF/Pearson scores."
        ),
        "x_label": "5° window center, 1° step (2θ degrees)",
        "z_label": "ACF Pearson correlation (−1 to 1)",
        "score_min": -1.0,
        "score_max": 1.0,
        "colorscale": DIVERGING,
        "x": [round(value, 3) for value in centers],
        "window_starts": starts,
        "window_ends": ends,
        "window_labels": labels,
        "raw_grid": [round(float(value), 4) for value in grid],
        "datasets": datasets,
        "method": "Pearson correlation of ACF fingerprints; legacy search chooses the best neighboring window within ±1°.",
        "caveat": (
            "The 5° windows overlap by 80%; high neighboring-window similarity is partly structural. "
            "For the release row, the stored directional search is 12.8 GPa nominal versus "
            "the neighboring 2.4 GPa decompression windows. The first compression row is a "
            "raw-XRD-only baseline; adjacent means adjacent measured frames and ΔP is not constant."
        ),
        "source_matrices": sources,
    }


def build_within_payload(
    root: Path,
    patterns: Sequence[PatternInfo],
    series: Sequence[SeriesInfo],
    grid: np.ndarray,
) -> dict[str, Any]:
    matrices: dict[int, MatrixData] = {}
    window_labels: list[str] | None = None
    sources: dict[int, str] = {}
    for pattern in patterns:
        path = root / "matrices" / f"{safe_name(pattern.label)}_single_frame_window_acf.csv"
        matrix = read_lower_triangle_matrix(path)
        if window_labels is None:
            window_labels = matrix.labels
        elif window_labels != matrix.labels:
            raise ValueError(f"Window order differs in {path}")
        matrices[pattern.index] = matrix
        sources[pattern.index] = str(path.resolve())
    assert window_labels is not None
    starts = [float(label.split("-", 1)[0]) for label in window_labels]
    ends = [float(label.split("-", 1)[1]) for label in window_labels]
    centers = [(start + end) / 2.0 for start, end in zip(starts, ends)]

    datasets: dict[str, Any] = {}
    for item in series:
        frames = item.frame_indices
        if item.key == "cell29_decompression":
            frames = [pattern.index for pattern in patterns if pattern.cell == "Cell_29" and pattern.decomp]
        frame_rows = []
        for index in frames:
            pattern = patterns[index]
            frame_rows.append(
                {
                    "frame_label": pattern.label,
                    "pressure": pattern.pressure,
                    "decomp": pattern.decomp,
                    "matrix": rounded_matrix(matrices[index].full),
                    "raw_profile": raw_profile(pattern, grid),
                    "source_matrix": sources[index],
                }
            )
        datasets[item.key] = {
            "label": item.label,
            "frames": frame_rows,
            "finite_score_count": int(
                sum(
                    value is not None
                    for frame in frame_rows
                    for row in frame["matrix"]
                    for value in row
                )
            ),
        }
    return {
        "kind": "within_frame_window_to_window",
        "title": "Window-to-Window Correlation Within Each Frame",
        "subtitle": "Each horizontal layer is one pressure frame's complete 5° window × 5° window ACF/Pearson matrix.",
        "x_label": "Window A center (2θ degrees)",
        "y_label": "Window B center (2θ degrees)",
        "z_label": "Pressure (GPa)",
        "score_min": -1.0,
        "score_max": 1.0,
        "colorscale": DIVERGING,
        "x": [round(value, 3) for value in centers],
        "window_starts": starts,
        "window_ends": ends,
        "window_labels": [label.replace("-", "–") + "°" for label in window_labels],
        "raw_grid": [round(float(value), 4) for value in grid],
        "datasets": datasets,
        "method": "Pearson correlation between ACF fingerprints of every pair of 5° windows inside one frame.",
        "caveat": "This is exploratory: overlapping windows share up to 80% of their raw 2θ range.",
    }


def json_for_html(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.replace("</", "<\\/")


def viewer_html(payload: dict[str, Any], plotly_js_mode: str) -> str:
    title = html.escape(str(payload["title"]))
    subtitle = html.escape(str(payload["subtitle"]))
    method = html.escape(str(payload["method"]))
    caveat = html.escape(str(payload["caveat"]))
    if plotly_js_mode == "inline" and get_plotlyjs is None:
        raise RuntimeError(
            "plotly is required for inline legacy 3D HTML generation; "
            "install requirements-optional.txt or use --plotly-js cdn"
        )
    plotly_loader = (
        f"<script>{get_plotlyjs()}</script>"
        if plotly_js_mode == "inline"
        else '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'
    )
    payload_json = json_for_html(payload)
    peak_controls = payload["kind"] in {"peak_area", "peak_position"}
    within = payload["kind"] == "within_frame_window_to_window"
    frame_count_control = (
        """
          <label>Minimum frame count
            <input id="frame-count" type="number" min="1" step="1" value="1">
          </label>
        """
        if peak_controls
        else ""
    )
    opacity_control = (
        """
          <label>Layer opacity
            <input id="layer-opacity" type="range" min="0.2" max="1" step="0.05" value="0.72">
          </label>
        """
        if within
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  {plotly_loader}
  <style>
    :root {{
      color-scheme: light;
      --background: #ffffff;
      --foreground: #111827;
      --muted: #5f6b7a;
      --border: #d7dee8;
      --accent: #075fcc;
      --accent-soft: #eaf2ff;
      --surface: #f8fafc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--background);
      color: var(--foreground);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
    }}
    .shell {{ min-width: 320px; padding: 20px 22px 14px; }}
    header {{ border-bottom: 1px solid var(--border); padding-bottom: 14px; }}
    h1 {{ margin: 0; font-size: clamp(23px, 2.4vw, 34px); font-weight: 500; letter-spacing: -0.025em; }}
    header p {{ margin: 7px 0 0; color: var(--muted); max-width: 1000px; }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 16px;
      align-items: end;
      padding: 15px 0 12px;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 13px;
      min-width: 126px;
    }}
    select, input, button {{
      min-height: 38px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--background);
      color: var(--foreground);
      font: inherit;
    }}
    select, input {{ padding: 7px 9px; }}
    input[type="range"] {{ padding: 0; accent-color: var(--accent); }}
    button {{ padding: 8px 13px; cursor: pointer; color: var(--accent); border-color: #90b7ea; }}
    button.primary {{ background: var(--accent); color: #ffffff; border-color: var(--accent); }}
    button:focus-visible, select:focus-visible, input:focus-visible {{
      outline: 3px solid rgba(7, 95, 204, 0.2);
      outline-offset: 1px;
    }}
    .switch {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: auto;
      min-height: 38px;
      padding-top: 19px;
    }}
    .switch input {{ min-height: auto; width: 18px; height: 18px; }}
    .workspace {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 250px;
      gap: 12px;
      align-items: stretch;
      min-height: 680px;
    }}
    #plot {{ min-height: 680px; width: 100%; }}
    aside {{
      border-left: 1px solid var(--border);
      padding: 10px 0 10px 18px;
      color: var(--muted);
    }}
    aside h2 {{ margin: 0 0 12px; color: var(--accent); font-size: 17px; font-weight: 500; }}
    dl {{ margin: 0; }}
    dt {{ margin-top: 13px; font-size: 12px; }}
    dd {{ margin: 3px 0 0; color: var(--foreground); font-size: 16px; overflow-wrap: anywhere; }}
    dd.emphasis {{ color: var(--accent); font-size: 20px; font-weight: 500; }}
    footer {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      border-top: 1px solid var(--border);
      padding-top: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    footer strong {{ color: var(--foreground); font-weight: 500; }}
    .empty {{
      display: none;
      margin: 8px 0;
      padding: 10px 12px;
      background: var(--accent-soft);
      color: var(--foreground);
      border-left: 3px solid var(--accent);
    }}
    @media (max-width: 900px) {{
      .workspace {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid var(--border); padding: 14px 0; }}
      footer {{ grid-template-columns: 1fr; }}
      #plot, .workspace {{ min-height: 570px; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </header>
    <section class="controls" aria-label="Viewer controls">
      <label>Dataset
        <select id="dataset"></select>
      </label>
      <label>2θ minimum
        <input id="theta-min" type="number" step="0.1">
      </label>
      <label>2θ maximum
        <input id="theta-max" type="number" step="0.1">
      </label>
      {frame_count_control}
      <label>Correlation minimum
        <input id="score-min" type="number" step="0.05">
      </label>
      <label>Correlation maximum
        <input id="score-max" type="number" step="0.05">
      </label>
      {opacity_control}
      <label class="switch">
        <input id="raw-overlay" type="checkbox" checked>
        Show raw XRD overlay
      </label>
      <button id="reset-camera" type="button">Reset camera</button>
      <button id="export-png" class="primary" type="button">Export PNG</button>
    </section>
    <p id="empty-message" class="empty" role="status"></p>
    <section class="workspace">
      <div id="plot" aria-label="Interactive 3D correlation plot"></div>
      <aside aria-live="polite">
        <h2>Selected point details</h2>
        <dl>
          <dt>Feature</dt><dd id="detail-feature">Click a point or surface cell</dd>
          <dt>2θ / windows</dt><dd id="detail-theta">—</dd>
          <dt>Pressure / pair</dt><dd id="detail-pair">—</dd>
          <dt>Correlation</dt><dd id="detail-score" class="emphasis">—</dd>
          <dt>Support</dt><dd id="detail-support">—</dd>
          <dt>Quality note</dt><dd id="detail-note">NaN is shown as missing, never as zero.</dd>
        </dl>
      </aside>
    </section>
    <footer>
      <div><strong>Method:</strong> {method}</div>
      <div><strong>Caveat:</strong> {caveat}</div>
    </footer>
  </main>
  <script>
    const DATA = {payload_json};
    const plot = document.getElementById("plot");
    const datasetSelect = document.getElementById("dataset");
    const thetaMin = document.getElementById("theta-min");
    const thetaMax = document.getElementById("theta-max");
    const scoreMin = document.getElementById("score-min");
    const scoreMax = document.getElementById("score-max");
    const rawOverlay = document.getElementById("raw-overlay");
    const frameCount = document.getElementById("frame-count");
    const layerOpacity = document.getElementById("layer-opacity");
    const emptyMessage = document.getElementById("empty-message");
    const defaultCamera = {{eye: {{x: 1.55, y: -1.75, z: 1.18}}, up: {{x: 0, y: 0, z: 1}}}};
    let initialized = false;

    function finite(value) {{ return typeof value === "number" && Number.isFinite(value); }}
    function clamp(value, low, high) {{ return Math.max(low, Math.min(high, value)); }}
    function datasetKeys() {{ return Object.keys(DATA.datasets); }}
    function currentDataset() {{ return DATA.datasets[datasetSelect.value]; }}
    function corrScale() {{ return DATA.colorscale; }}
    function pressureText(value) {{
      return Number(value).toLocaleString("en-US", {{maximumFractionDigits: 3}});
    }}
    function deltaPText(value) {{
      if (!finite(value)) return "Baseline / no previous correlation";
      const sign = value > 0 ? "+" : "";
      return `ΔP=${{sign}}${{pressureText(value)}} GPa`;
    }}

    function fillDatasets() {{
      for (const key of datasetKeys()) {{
        const option = document.createElement("option");
        option.value = key;
        option.textContent = DATA.datasets[key].label;
        datasetSelect.appendChild(option);
      }}
      datasetSelect.value = DATA.datasets.cell29_compression ? "cell29_compression" : datasetKeys()[0];
    }}

    function initControls() {{
      const xs = DATA.raw_grid || DATA.x;
      thetaMin.value = Math.min(...xs).toFixed(1);
      thetaMax.value = Math.max(...xs).toFixed(1);
      scoreMin.value = DATA.score_min.toFixed(2);
      scoreMax.value = DATA.score_max.toFixed(2);
    }}

    function sceneBase(xTitle, yTitle, zTitle) {{
      return {{
        xaxis: {{title: xTitle, gridcolor: "#d7dee8", zerolinecolor: "#aeb8c6", backgroundcolor: "#ffffff"}},
        yaxis: {{title: yTitle, gridcolor: "#d7dee8", zerolinecolor: "#aeb8c6", backgroundcolor: "#ffffff"}},
        zaxis: {{title: zTitle, gridcolor: "#d7dee8", zerolinecolor: "#aeb8c6", backgroundcolor: "#ffffff"}},
        bgcolor: "#ffffff",
        camera: defaultCamera,
        aspectmode: "auto"
      }};
    }}

    function rawTracesForPairs(ds, zBase, zSpan) {{
      if (!rawOverlay.checked || !ds.raw_profiles) return [];
      const lo = Number(thetaMin.value);
      const hi = Number(thetaMax.value);
      const keepForProfile = profile => DATA.raw_grid
        .map((x, i) => x >= lo && x <= hi && finite(profile[i]) ? i : -1)
        .filter(i => i >= 0);
      return ds.raw_profiles.map((profile, row) => ({{
        type: "scatter3d",
        mode: "lines",
        name: `Raw XRD · ${{ds.pair_labels[row]}}`,
        meta: {{kind: "raw-xrd", row}},
        x: keepForProfile(profile).map(i => DATA.raw_grid[i]),
        y: keepForProfile(profile).map(() => ds.pair_pressures[row]),
        z: keepForProfile(profile).map(i => zBase + zSpan * clamp(profile[i], 0, 1.2)),
        line: {{color: "rgba(82,112,151,0.48)", width: 2}},
        hovertemplate: `Raw XRD<br>${{ds.pair_labels[row]}}<br>${{deltaPText(ds.pair_delta_p_gpa?.[row])}}<br>2θ=%{{x:.3f}}°<br>normalized I=%{{customdata:.3f}}<extra></extra>`,
        customdata: keepForProfile(profile).map(i => profile[i]),
        showlegend: row === 0
      }}));
    }}

    function renderPeak() {{
      const ds = currentDataset();
      const lo = Number(thetaMin.value);
      const hi = Number(thetaMax.value);
      const minScore = Number(scoreMin.value);
      const maxScore = Number(scoreMax.value);
      const minFrames = frameCount ? Number(frameCount.value) : 1;
      const regular = {{x: [], y: [], z: [], customdata: []}};
      for (let row = 0; row < ds.scores.length; row++) {{
        for (let col = 0; col < DATA.x.length; col++) {{
          const x = DATA.x[col];
          const score = ds.scores[row][col];
          if (!finite(score) || x < lo || x > hi || DATA.frame_counts[col] < minFrames ||
              score < minScore || score > maxScore) continue;
          regular.x.push(x); regular.y.push(ds.pair_pressures[row]); regular.z.push(score);
          regular.customdata.push([
            DATA.groups[col],
            x,
            ds.pair_labels[row],
            score,
            DATA.frame_counts[col],
            DATA.position_ranges?.[col],
            deltaPText(ds.pair_delta_p_gpa?.[row]),
            ds.pair_kinds?.[row]
          ]);
        }}
      }}
      const traces = [];
      if (regular.x.length) traces.push({{
        type: "scatter3d", mode: "markers", name: "Correlation values",
        x: regular.x, y: regular.y, z: regular.z, customdata: regular.customdata,
        marker: {{size: 3.2, color: regular.z, colorscale: corrScale(), cmin: 0, cmax: 1,
          colorbar: {{title: DATA.z_label, thickness: 16}}, opacity: 0.86}},
        hovertemplate: "Peak group %{{customdata[0]}}<br>2θ=%{{x:.4f}}°<br>%{{customdata[2]}}<br>%{{customdata[6]}}<br>s=%{{z:.4f}}<br>detected frames=%{{customdata[4]}}<extra></extra>"
      }});
      traces.push(...rawTracesForPairs(ds, -0.17, 0.12));
      const yTicks = ds.pair_pressures;
      const scene = sceneBase(DATA.x_label, "Measured pressure / previous measured frame (GPa)", DATA.z_label);
      scene.yaxis.tickvals = yTicks; scene.yaxis.ticktext = ds.pair_labels;
      scene.zaxis.range = [-0.2, 1.02];
      showEmpty(regular.x.length === 0, "No finite peak scores match the current filters.");
      draw(traces, scene);
    }}

    function renderAcross() {{
      const ds = currentDataset();
      const lo = Number(thetaMin.value);
      const hi = Number(thetaMax.value);
      const minScore = Number(scoreMin.value);
      const maxScore = Number(scoreMax.value);
      const xIndices = DATA.x.map((x, i) => x >= lo && x <= hi ? i : -1).filter(i => i >= 0);
      const xs = xIndices.map(i => DATA.x[i]);
      const z = ds.scores.map(row => xIndices.map(i => {{
        const value = row[i];
        return finite(value) && value >= minScore && value <= maxScore ? value : null;
      }}));
      const custom = ds.scores.map((row, r) => xIndices.map(i => [
        DATA.window_labels[i], ds.pair_labels[r], row[i], DATA.window_starts[i],
        DATA.window_ends[i], deltaPText(ds.pair_delta_p_gpa?.[r]), ds.pair_kinds?.[r]
      ]));
      const traces = [];
      const correlationRows = z
        .map((_, row) => ds.baseline_mask?.[row] ? -1 : row)
        .filter(row => row >= 0);
      const finiteRows = correlationRows.filter(row => z[row].some(finite));
      if (finiteRows.length) {{
        if (finiteRows.length === 1 || xs.length < 2) {{
          const pointRows = finiteRows.flatMap(row =>
            z[row]
              .map((value, i) => finite(value) ? {{row, i}} : null)
              .filter(item => item !== null)
          );
          traces.push({{
            type: "scatter3d",
            mode: finiteRows.length === 1 && pointRows.length > 1 ? "lines+markers" : "markers",
            name: "Across-frame ACF",
            x: pointRows.map(item => xs[item.i]),
            y: pointRows.map(item => ds.pair_pressures[item.row]),
            z: pointRows.map(item => z[item.row][item.i]),
            customdata: pointRows.map(item => custom[item.row][item.i]),
            line: {{color: "#5b6472", width: 3}},
            marker: {{
              size: 5, color: pointRows.map(item => z[item.row][item.i]), colorscale: corrScale(),
              cmin: -1, cmax: 1,
              colorbar: {{title: "ACF Pearson r", thickness: 16}}
            }},
            hovertemplate: "%{{customdata[0]}}<br>%{{customdata[1]}}<br>%{{customdata[5]}}<br>r=%{{customdata[2]:.4f}}<extra></extra>"
          }});
        }} else {{
          traces.push({{
            type: "surface", name: "Across-frame ACF", x: xs,
            y: correlationRows.map(row => ds.pair_pressures[row]),
            z: correlationRows.map(row => z[row]),
            surfacecolor: correlationRows.map(row => z[row]),
            customdata: correlationRows.map(row => custom[row]),
            colorscale: corrScale(), cmin: -1, cmax: 1, connectgaps: false,
            colorbar: {{title: "ACF Pearson r", thickness: 16}},
            hovertemplate: "%{{customdata[0]}}<br>%{{customdata[1]}}<br>%{{customdata[5]}}<br>r=%{{customdata[2]:.4f}}<extra></extra>"
          }});
        }}
      }}
      const hasCorrelation = traces.length > 0;
      traces.push(...rawTracesForPairs(ds, -1.16, 0.14));
      const scene = sceneBase(DATA.x_label, "Measured pressure / previous measured frame (GPa)", DATA.z_label);
      scene.yaxis.tickvals = ds.pair_pressures;
      scene.yaxis.ticktext = ds.pair_labels;
      scene.zaxis.range = [-1.22, 1.02];
      showEmpty(!hasCorrelation, "No finite window scores match the current filters.");
      draw(traces, scene);
    }}

    function renderWithin() {{
      const ds = currentDataset();
      const lo = Number(thetaMin.value);
      const hi = Number(thetaMax.value);
      const minScore = Number(scoreMin.value);
      const maxScore = Number(scoreMax.value);
      const opacity = layerOpacity ? Number(layerOpacity.value) : 0.72;
      const indices = DATA.x.map((x, i) => x >= lo && x <= hi ? i : -1).filter(i => i >= 0);
      const xs = indices.map(i => DATA.x[i]);
      const traces = [];
      for (let frameIndex = 0; frameIndex < ds.frames.length; frameIndex++) {{
        const frame = ds.frames[frameIndex];
        const colors = indices.map(i => indices.map(j => {{
          const value = frame.matrix[i][j];
          return finite(value) && value >= minScore && value <= maxScore ? value : null;
        }}));
        const z = colors.map(row => row.map(value => finite(value) ? frame.pressure : null));
        const custom = indices.map(i => indices.map(j => [
          DATA.window_labels[j], DATA.window_labels[i], frame.frame_label,
          frame.matrix[i][j], frame.pressure, frame.decomp
        ]));
        if (colors.some(row => row.some(finite))) traces.push({{
          type: "surface", name: frame.frame_label, x: xs, y: xs, z,
          surfacecolor: colors, customdata: custom, colorscale: corrScale(),
          cmin: -1, cmax: 1, opacity, connectgaps: false,
          showscale: frameIndex === 0,
          colorbar: {{title: "Within-frame Pearson r", thickness: 16}},
          hovertemplate: "Window A %{{customdata[0]}}<br>Window B %{{customdata[1]}}<br>%{{customdata[2]}} · %{{customdata[4]:g}} GPa<br>r=%{{customdata[3]:.4f}}<extra></extra>"
        }});
      }}
      if (rawOverlay.checked) {{
        const yWall = Math.min(...xs) - 1.0;
        for (const frame of ds.frames) {{
          const keep = DATA.raw_grid
            .map((x, i) => x >= lo && x <= hi && finite(frame.raw_profile[i]) ? i : -1)
            .filter(i => i >= 0);
          traces.push({{
            type: "scatter3d", mode: "lines", name: `Raw XRD · ${{frame.frame_label}}`,
            meta: {{kind: "raw-xrd-within", frameLabel: frame.frame_label}},
            x: keep.map(i => DATA.raw_grid[i]), y: keep.map(() => yWall),
            z: keep.map(i => frame.pressure + 0.3 * clamp(frame.raw_profile[i], 0, 1.2)),
            line: {{color: "rgba(82,112,151,0.5)", width: 2}},
            customdata: keep.map(i => frame.raw_profile[i]),
            hovertemplate: `${{frame.frame_label}}<br>2θ=%{{x:.3f}}°<br>normalized I=%{{customdata:.3f}}<extra></extra>`,
            showlegend: false
          }});
        }}
      }}
      const scene = sceneBase(DATA.x_label, DATA.y_label, DATA.z_label);
      const pressures = ds.frames.map(frame => frame.pressure);
      scene.zaxis.tickvals = pressures;
      scene.zaxis.ticktext = ds.frames.map(frame =>
        frame.decomp ? `${{pressureText(frame.pressure)}} decomp` : pressureText(frame.pressure)
      );
      if (pressures.length === 1) scene.zaxis.range = [pressures[0] - 0.55, pressures[0] + 0.75];
      showEmpty(traces.length === 0, "No finite window-pair scores match the current filters.");
      draw(traces, scene);
    }}

    function draw(traces, scene) {{
      const layout = {{
        margin: {{l: 0, r: 0, t: 18, b: 0}},
        paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff",
        font: {{family: "Inter, system-ui, sans-serif", color: "#111827", size: 12}},
        legend: {{orientation: "h", x: 0, y: 1.02}},
        scene,
        uirevision: "preserve-camera"
      }};
      const config = {{responsive: true, displaylogo: false, scrollZoom: true}};
      const method = initialized ? Plotly.react : Plotly.newPlot;
      method(plot, traces, layout, config);
      initialized = true;
    }}

    function showEmpty(isEmpty, message) {{
      emptyMessage.style.display = isEmpty ? "block" : "none";
      emptyMessage.textContent = isEmpty ? message : "";
    }}

    function render() {{
      if (DATA.kind === "peak_area" || DATA.kind === "peak_position") renderPeak();
      else if (DATA.kind === "same_window_across_frames") renderAcross();
      else renderWithin();
    }}

    function detail(id, value, emphasis = false) {{
      const node = document.getElementById(id);
      node.textContent = value ?? "—";
      node.classList.toggle("emphasis", emphasis);
    }}

    fillDatasets();
    initControls();
    render();
    for (const control of document.querySelectorAll("select, input")) {{
      control.addEventListener(control.type === "range" ? "input" : "change", render);
    }}
    document.getElementById("reset-camera").addEventListener("click", () => {{
      Plotly.relayout(plot, {{"scene.camera": defaultCamera}});
    }});
    document.getElementById("export-png").addEventListener("click", () => {{
      Plotly.downloadImage(plot, {{format: "png", width: 2200, height: 1400, filename: DATA.kind}});
    }});
    plot.on("plotly_click", event => {{
      const point = event.points?.[0];
      if (!point) return;
      const data = point?.customdata;
      if (point?.data?.meta?.kind === "raw-xrd" || point?.data?.meta?.kind === "raw-xrd-within") {{
        if (!finite(data)) return;
        detail("detail-feature", "Raw XRD intensity");
        detail("detail-theta", `${{Number(point.x).toFixed(4)}}°`);
        detail("detail-pair", point.data.name.replace("Raw XRD · ", ""));
        detail("detail-score", `I(norm)=${{Number(data).toFixed(4)}}`, true);
        detail("detail-support", "Display overlay from the measured XY pattern");
        detail("detail-note", "Raw XRD is normalized for display and does not create a correlation value.");
        return;
      }}
      if (!data) return;
      if (DATA.kind === "peak_area" || DATA.kind === "peak_position") {{
        detail("detail-feature", `Peak group ${{data[0]}}`);
        detail("detail-theta", `${{Number(data[1]).toFixed(4)}}°`);
        detail("detail-pair", data[2]);
        detail("detail-score", Number(data[3]).toFixed(4), true);
        detail("detail-support", `Detected in ${{data[4]}} frames · ${{data[6]}}`);
        detail("detail-note", "Stored finite legacy score.");
      }} else if (DATA.kind === "same_window_across_frames") {{
        detail("detail-feature", "Same 5° window across frames");
        detail("detail-theta", data[0]);
        detail("detail-pair", data[1]);
        detail("detail-score", finite(data[2]) ? Number(data[2]).toFixed(4) : "unknown", true);
        detail("detail-support", `One stored adjacent measured-frame pair · ${{data[5]}}`);
        detail("detail-note", DATA.caveat);
      }} else {{
        detail("detail-feature", `${{data[0]}} × ${{data[1]}}`);
        detail("detail-theta", `${{data[0]}} vs ${{data[1]}}`);
        detail("detail-pair", `${{data[2]}} · ${{data[4]}} GPa${{data[5] ? " decomp" : ""}}`);
        detail("detail-score", finite(data[3]) ? Number(data[3]).toFixed(4) : "unknown", true);
        detail("detail-support", "One within-frame window pair");
        detail("detail-note", DATA.caveat);
      }}
    }});
  </script>
</body>
</html>
"""


def pressure_pair_ticklabels(labels: Sequence[str]) -> list[str]:
    output: list[str] = []
    for label in labels:
        cleaned = label.replace(" GPa", "")
        if "Baseline / no previous correlation" in cleaned:
            pressure = cleaned.split(" —", 1)[0]
            output.append(
                f"{pressure}\nBaseline / no previous\ncorrelation"
            )
        else:
            output.append(cleaned)
    return output


def figure_footer(payload: dict[str, Any]) -> str:
    text = f"Method: {payload['method']}   Caveat: {payload['caveat']}"
    return "\n".join(
        textwrap.wrap(
            text,
            width=210,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def raw_line_on_pair_axis(
    ax: Any,
    raw_grid: Sequence[float],
    profile: Sequence[float | None],
    pressure: float,
    baseline: float,
    span: float,
) -> None:
    values = np.clip(np.asarray(profile, dtype=float), 0.0, 1.2)
    ax.plot(
        raw_grid,
        np.full(len(raw_grid), pressure),
        baseline + span * values,
        color="#597aa7",
        alpha=0.35,
        lw=0.5,
    )


def plot_peak_png(payload: dict[str, Any], output: Path, dpi: int) -> None:
    keys = ["cell14_compression", "cell29_compression", "cell29_decompression"]
    fig = plt.figure(figsize=(23, 7.7))
    mappable = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
    for panel, key in enumerate(keys, start=1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        ds = payload["datasets"][key]
        for row, scores in enumerate(ds["scores"]):
            values = np.asarray([np.nan if value is None else value for value in scores], dtype=float)
            keep = np.isfinite(values)
            pressure = float(ds["pair_pressures"][row])
            ax.scatter(
                np.asarray(payload["x"])[keep],
                np.full(int(keep.sum()), pressure),
                values[keep],
                c=values[keep],
                cmap="viridis",
                vmin=0,
                vmax=1,
                s=6,
                alpha=0.82,
                depthshade=False,
            )
            if row < len(ds["raw_profiles"]):
                raw_line_on_pair_axis(
                    ax,
                    payload["raw_grid"],
                    ds["raw_profiles"][row],
                    pressure,
                    -0.17,
                    0.12,
                )
        ax.set_title(ds["label"], fontsize=12, pad=12)
        ax.set_xlabel("2θ (degrees)", labelpad=8)
        ax.set_ylabel("Measured pressure (GPa)", labelpad=8)
        ax.set_zlabel(payload["z_label"], labelpad=8)
        ax.set_zlim(-0.2, 1.02)
        ax.set_yticks(ds["pair_pressures"])
        ax.set_yticklabels(pressure_pair_ticklabels(ds["pair_labels"]), fontsize=6)
        ax.view_init(elev=28, azim=-123)
        ax.grid(True, alpha=0.2)
        if not ds["pair_labels"]:
            ax.text2D(0.5, 0.5, "No adjacent pair", transform=ax.transAxes, ha="center")
    fig.suptitle(payload["title"], fontsize=18, fontweight="semibold", y=0.98)
    fig.text(
        0.5,
        0.012,
        figure_footer(payload),
        ha="center",
        va="bottom",
        fontsize=7.6,
        linespacing=1.25,
    )
    fig.colorbar(mappable, ax=fig.axes, shrink=0.58, pad=0.02, label=payload["z_label"])
    fig.subplots_adjust(left=0.02, right=0.93, top=0.90, bottom=0.14, wspace=0.04)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_across_png(payload: dict[str, Any], output: Path, dpi: int) -> None:
    keys = ["cell14_compression", "cell29_compression", "cell29_decompression"]
    fig = plt.figure(figsize=(23, 7.7))
    mappable = ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu_r")
    x = np.asarray(payload["x"], dtype=float)
    for panel, key in enumerate(keys, start=1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        ds = payload["datasets"][key]
        z = np.asarray(
            [[np.nan if value is None else value for value in row] for row in ds["scores"]],
            dtype=float,
        )
        baseline_mask = np.asarray(ds.get("baseline_mask", [False] * len(z)), dtype=bool)
        correlation_rows = np.flatnonzero(~baseline_mask)
        finite_rows = [
            int(row)
            for row in correlation_rows
            if np.any(np.isfinite(z[row]))
        ]
        if finite_rows:
            if len(finite_rows) == 1 or len(x) < 2:
                row = finite_rows[0]
                keep = np.isfinite(z[row])
                pressure = float(ds["pair_pressures"][row])
                ax.plot(
                    x[keep],
                    np.full(int(keep.sum()), pressure),
                    z[row, keep],
                    color="#5b6472",
                    linewidth=1.0,
                    alpha=0.8,
                )
                ax.scatter(
                    x[keep],
                    np.full(int(keep.sum()), pressure),
                    z[row, keep],
                    c=z[row, keep],
                    cmap="RdBu_r",
                    vmin=-1,
                    vmax=1,
                    s=18,
                    alpha=0.92,
                    depthshade=False,
                )
            else:
                correlation_z = z[correlation_rows]
                xx, _ = np.meshgrid(x, np.arange(len(correlation_rows)))
                yy = np.repeat(
                    np.asarray(ds["pair_pressures"], dtype=float)[
                        correlation_rows
                    ][:, None],
                    correlation_z.shape[1],
                    axis=1,
                )
                ax.plot_surface(
                    xx,
                    yy,
                    correlation_z,
                    cmap="RdBu_r",
                    vmin=-1,
                    vmax=1,
                    linewidth=0.15,
                    antialiased=True,
                    alpha=0.88,
                )
        for row, profile in enumerate(ds["raw_profiles"]):
            raw_line_on_pair_axis(
                ax,
                payload["raw_grid"],
                profile,
                float(ds["pair_pressures"][row]),
                -1.16,
                0.14,
            )
        ax.set_title(ds["label"], fontsize=12, pad=12)
        ax.set_xlabel("5° window center (2θ°)", labelpad=8)
        ax.set_ylabel("Measured pressure (GPa)", labelpad=8)
        ax.set_zlabel("ACF Pearson r", labelpad=8)
        ax.set_zlim(-1.22, 1.02)
        ax.set_yticks(ds["pair_pressures"])
        ax.set_yticklabels(pressure_pair_ticklabels(ds["pair_labels"]), fontsize=6)
        ax.view_init(elev=28, azim=-123)
        ax.grid(True, alpha=0.2)
        if not ds["pair_labels"]:
            ax.text2D(0.5, 0.5, "No adjacent pair", transform=ax.transAxes, ha="center")
    fig.suptitle(payload["title"], fontsize=18, fontweight="semibold", y=0.98)
    fig.text(
        0.5,
        0.012,
        figure_footer(payload),
        ha="center",
        va="bottom",
        fontsize=7.6,
        linespacing=1.25,
    )
    fig.colorbar(mappable, ax=fig.axes, shrink=0.58, pad=0.02, label="ACF Pearson r")
    fig.subplots_adjust(left=0.02, right=0.93, top=0.90, bottom=0.14, wspace=0.04)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_within_png(payload: dict[str, Any], output: Path, dpi: int) -> None:
    keys = ["cell14_compression", "cell29_compression", "cell29_decompression"]
    fig = plt.figure(figsize=(23, 7.7))
    mappable = ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu_r")
    centers = np.asarray(payload["x"], dtype=float)
    upper_i, upper_j = np.triu_indices(len(centers), k=1)
    for panel, key in enumerate(keys, start=1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        ds = payload["datasets"][key]
        for frame in ds["frames"]:
            matrix = np.asarray(
                [[np.nan if value is None else value for value in row] for row in frame["matrix"]],
                dtype=float,
            )
            values = matrix[upper_i, upper_j]
            keep = np.isfinite(values)
            ax.scatter(
                centers[upper_j][keep],
                centers[upper_i][keep],
                np.full(int(keep.sum()), float(frame["pressure"])),
                c=values[keep],
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                s=12,
                alpha=0.78,
                depthshade=False,
            )
            y_wall = float(np.nanmin(centers) - 1.0)
            profile = np.clip(np.asarray(frame["raw_profile"], dtype=float), 0.0, 1.2)
            ax.plot(
                payload["raw_grid"],
                np.full(len(payload["raw_grid"]), y_wall),
                float(frame["pressure"]) + 0.3 * profile,
                color="#597aa7",
                alpha=0.34,
                lw=0.5,
            )
        ax.set_title(ds["label"], fontsize=12, pad=12)
        ax.set_xlabel("Window A center (2θ°)", labelpad=8)
        ax.set_ylabel("Window B center (2θ°)", labelpad=8)
        ax.set_zlabel("Pressure (GPa)", labelpad=8)
        ax.view_init(elev=26, azim=-128)
        ax.grid(True, alpha=0.2)
        if not ds["frames"]:
            ax.text2D(0.5, 0.5, "No frame", transform=ax.transAxes, ha="center")
    fig.suptitle(payload["title"], fontsize=18, fontweight="semibold", y=0.98)
    fig.text(
        0.5,
        0.012,
        figure_footer(payload),
        ha="center",
        va="bottom",
        fontsize=7.6,
        linespacing=1.25,
    )
    fig.colorbar(mappable, ax=fig.axes, shrink=0.58, pad=0.02, label="Within-frame Pearson r")
    fig.subplots_adjust(left=0.02, right=0.93, top=0.90, bottom=0.14, wspace=0.04)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(path: Path, suite: Path, inputs: Sequence[Path], artifacts: Sequence[dict[str, Any]]) -> None:
    text = f"""# Legacy correlation 3D waterfalls

These four viewers use the **stored legacy correlation values** from:

`{suite.resolve()}`

The measured XRD overlays come from:

{chr(10).join(f"- `{item.resolve()}`" for item in inputs)}

## Deliverables

| Correlation family | Interactive viewer | Static overview |
|---|---|---|
| ROI area across adjacent frames | `01_roi_area_correlation_3d.html` | `01_roi_area_correlation_3d.png` |
| Per-peak detected location across adjacent frames | `02_per_peak_location_correlation_3d.html` | `02_per_peak_location_correlation_3d.png` |
| Same 5° window across adjacent frames | `03_window_across_frames_correlation_3d.html` | `03_window_across_frames_correlation_3d.png` |
| Window-to-window within each frame | `04_window_within_frame_correlation_3d.html` | `04_window_within_frame_correlation_3d.png` |

Each HTML is standalone when generated with the default `--plotly-js inline`.
Open it directly in a browser. Rotate by dragging, zoom with the wheel, use the
dataset selector to switch Cell/branch, and use **Export PNG** for the current
interactive view.

## Axis meaning

- First three viewers:
  - x = legacy peak-group 2θ or 5° window center;
  - y = every measured compression pressure;
  - z + color = the stored correlation/similarity value.
  - the first compression pressure is labeled
    `Baseline / no previous correlation`; it has raw XRD but no invented score;
  - every later pressure row stores the comparison with the previous measured
    pressure.
- Within-frame viewer:
  - x = window A center;
  - y = window B center;
  - z = pressure;
  - color = the stored within-frame ACF/Pearson value.

The first three viewers show raw XRD for all 6 Cell 14 compression frames and
all 10 Cell 29 compression frames. The raw layer is a robust-normalized display
overlay. It does not replace or recalculate the stored correlation value.

## Pair ordering

- Cell 14 compression is kept separate from Cell 29 compression.
- Six Cell 14 frames still produce five adjacent correlation intervals, and ten
  Cell 29 compression frames still produce nine. The baseline rows make the
  complete measured frame sequences visible without inventing a sixth or tenth
  correlation interval.
- Cell 29 decompression is not inserted into the numeric-pressure compression
  sequence. Its separate branch reference compares the final compression frame
  with the decompression frame.
- `Adjacent` means adjacent **measured** frames, not equal pressure spacing.
  For example, Cell 14 `1.3 → 6.6 GPa` spans 5.3 GPa; correlation magnitudes
  should not be compared across intervals without considering ΔP.
- Missing/NaN values remain missing. They are never replaced by zero.

## Scientific cautions

1. These are the requested legacy numbers, not a new or corrected correlation run.
2. The high-recall peak registry has hundreds of tightly spaced groups. A group
   is a detection bucket, not automatically a unique physical reflection.
3. The legacy 0.02° peak grouping can fragment peaks that move under pressure.
4. ROI area is sensitive to texture, orientation, exposure, overlap, and
   background. Pairs detected in both frames but having zero ROI area in both
   are treated as missing and are not plotted.
5. Sliding 5° windows use a 1° step, so adjacent windows overlap by 80%.
6. Across-frame ACF uses the legacy neighboring-window best-match search and is
   best treated as exploratory.
7. Correlation locates coordinated changes; it does not establish phase
   identity, causation, or a phase transition by itself.

## Artifact inventory

`artifact_index.csv` records size and SHA-256 for the generated HTML/PNG files.
`run_manifest.json` records source paths, frame labels, pair rules, and counts.

Generated artifacts: {len(artifacts)}
"""
    path.write_text(text, encoding="utf-8")


def output_specs() -> list[tuple[str, str]]:
    return [
        ("01_roi_area_correlation_3d", "peak_area"),
        ("02_per_peak_location_correlation_3d", "peak_position"),
        ("03_window_across_frames_correlation_3d", "same_window_across_frames"),
        ("04_window_within_frame_correlation_3d", "within_frame_window_to_window"),
    ]


def validate_output(output: Path, specs: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for stem, kind in specs:
        html_path = output / f"{stem}.html"
        png_path = output / f"{stem}.png"
        html_text = html_path.read_text(encoding="utf-8")
        checks.extend(
            [
                {
                    "check": f"{stem}: html exists and is nontrivial",
                    "passed": html_path.stat().st_size > 100_000,
                },
                {
                    "check": f"{stem}: Plotly and expected kind embedded",
                    "passed": "Plotly.newPlot" in html_text and f'"kind":"{kind}"' in html_text,
                },
                {
                    "check": f"{stem}: all three datasets embedded",
                    "passed": all(
                        key in html_text
                        for key in ("cell14_compression", "cell29_compression", "cell29_decompression")
                    ),
                },
                {
                    "check": f"{stem}: PNG signature",
                    "passed": png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
                },
            ]
        )
        if kind != "within_frame_window_to_window":
            checks.append(
                {
                    "check": f"{stem}: explicit baseline rows embedded",
                    "passed": (
                        "0.7 GPa — Baseline / no previous correlation" in html_text
                        and "1 GPa — Baseline / no previous correlation" in html_text
                    ),
                }
            )
        if kind == "peak_area":
            checks.append(
                {
                    "check": f"{stem}: no warning marker layer embedded",
                    "passed": all(
                        phrase not in html_text
                        for phrase in (
                            "Legacy zero-area warning",
                            "legacy_warning_mask",
                            "WARNING<br>",
                            "present-but-zero",
                            "#b45f06",
                        )
                    ),
                }
            )
    failures = [row for row in checks if not row["passed"]]
    if failures:
        raise RuntimeError(f"Output validation failed: {failures}")
    return checks


def main() -> None:
    args = parse_args()
    suite = args.suite_dir.resolve()
    output = args.out_dir.resolve()
    inputs = [path.resolve() for path in args.inputs]
    if output.exists():
        raise SystemExit(f"Output already exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    peak_root = suite / PEAK_ROOT_NAME
    across_root = suite / ACROSS_ROOT_NAME
    within_root = suite / WITHIN_ROOT_NAME
    area_features = read_feature_table(peak_root / "peak_roi_area_features.csv")
    presence_features = read_feature_table(peak_root / "peak_presence_features.csv")
    position_features = read_feature_table(peak_root / "peak_position_features.csv")
    if not (
        area_features.labels == presence_features.labels == position_features.labels
        and np.allclose(area_features.columns, presence_features.columns)
        and np.allclose(area_features.columns, position_features.columns)
    ):
        raise ValueError("Legacy peak feature tables are not aligned.")

    patterns = load_patterns(inputs, area_features.labels)
    series = build_series(patterns)
    grid = common_raw_grid(patterns)
    payloads = [
        build_peak_payload(
            peak_root,
            patterns,
            series,
            grid,
            area_features,
            presence_features,
            position_features,
            position=False,
        ),
        build_peak_payload(
            peak_root,
            patterns,
            series,
            grid,
            area_features,
            presence_features,
            position_features,
            position=True,
        ),
        build_across_payload(across_root, patterns, series, grid),
        build_within_payload(within_root, patterns, series, grid),
    ]
    specs = output_specs()

    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for (stem, expected_kind), payload in zip(specs, payloads):
            if payload["kind"] != expected_kind:
                raise AssertionError(f"Payload order mismatch: {payload['kind']} != {expected_kind}")
            (temp / f"{stem}.html").write_text(
                viewer_html(payload, args.plotly_js),
                encoding="utf-8",
            )
            png_path = temp / f"{stem}.png"
            if payload["kind"] in {"peak_area", "peak_position"}:
                plot_peak_png(payload, png_path, args.dpi)
            elif payload["kind"] == "same_window_across_frames":
                plot_across_png(payload, png_path, args.dpi)
            else:
                plot_within_png(payload, png_path, args.dpi)

        checks = validate_output(temp, specs)
        (temp / "validation.json").write_text(
            json.dumps(checks, indent=2),
            encoding="utf-8",
        )

        artifacts: list[dict[str, Any]] = []
        for path in sorted(temp.iterdir()):
            if path.suffix.lower() not in {".html", ".png"}:
                continue
            artifacts.append(
                {
                    "artifact": path.name,
                    "type": path.suffix.lower().lstrip("."),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        with (temp / "artifact_index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["artifact", "type", "bytes", "sha256"])
            writer.writeheader()
            writer.writerows(artifacts)
        write_readme(temp / "README.md", suite, inputs, artifacts)

        manifest = {
            "script": str(Path(__file__).resolve()),
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_suite": str(suite),
            "integrated_xy_inputs": [str(path) for path in inputs],
            "plotly_js": args.plotly_js,
            "frame_labels": [pattern.label for pattern in patterns],
            "frame_cells": [pattern.cell for pattern in patterns],
            "frame_pressures_gpa": [pattern.pressure for pattern in patterns],
            "frame_decompression": [pattern.decomp for pattern in patterns],
            "series": [
                {
                    "key": item.key,
                    "label": item.label,
                    "frames": [patterns[index].label for index in item.frame_indices],
                    "first_three_viewer_rows": [
                        {
                            "current": patterns[row.current].label,
                            "previous": (
                                None
                                if row.previous is None
                                else patterns[row.previous].label
                            ),
                            "label": row.label,
                            "kind": row.kind,
                            "delta_p_gpa": row.delta_p_gpa,
                        }
                        for row in build_display_rows(item, patterns)
                    ],
                    "pairs": [
                        {
                            "previous": patterns[pair.previous].label,
                            "current": patterns[pair.current].label,
                            "label": pair.label,
                            "kind": pair.kind,
                        }
                        for pair in item.pairs
                    ],
                }
                for item in series
            ],
            "peak_group_count": len(area_features.columns),
            "window_count": len(payloads[2]["x"]),
            "raw_grid_step_deg": RAW_GRID_STEP,
            "artifacts": artifacts,
            "validation_passed": True,
        }
        (temp / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        os.replace(temp, output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print(f"Wrote four interactive HTML viewers and four PNG overviews to {output}")


if __name__ == "__main__":
    main()
