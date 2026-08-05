#!/usr/bin/env python3
"""Generate feature-faithful waterfall plots for the legacy four-map XRD suite.

The legacy suite contains four correlation families:

* per-peak ROI-area similarity across frames;
* per-peak detected-position similarity across frames;
* same-window ACF similarity across frames (with a neighboring-window search);
* window-to-window ACF similarity within one frame.

This script deliberately plots the feature used by each correlation instead of
repeating a full-pattern intensity waterfall for every map.  It also writes
machine-readable links between source heatmaps, matrices, waterfalls, frames,
peak groups, and ACF windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from scipy.stats import spearmanr

import compare_integrated_peaks as cip
import window_autocorrelation_correlations as wacf


SCRIPT_VERSION = "1.0.1"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = DEFAULT_ROOT / "outputs/correlation_suite_20260621_high_recall_scored_v2"
DEFAULT_OUTPUT = DEFAULT_ROOT / "outputs/correlation_suite_20260621_high_recall_scored_v2_waterfalls_20260722"
DEFAULT_INPUTS = [DEFAULT_ROOT / "Data/Cell_14_integrated", DEFAULT_ROOT / "Data/Cell_29_integrated"]
CORRELATION_TYPES = ("area", "position", "same-window", "within-frame")
ZERO_AREA_EPS = 1e-12


SIMILARITY_CMAP = LinearSegmentedColormap.from_list(
    "xrd_similarity",
    ["#2166ac", "#f2f2f2", "#b2182b"],
)
SIMILARITY_NORM = Normalize(vmin=0.0, vmax=1.0)
MISSING_COLOR = "#8a8a8a"
REFERENCE_DASH = "#3f3f3f"
WARNING_COLOR = "#a05a00"


@dataclass(frozen=True)
class MatrixData:
    labels: list[str]
    lower: np.ndarray
    full: np.ndarray


@dataclass(frozen=True)
class FeatureTable:
    labels: list[str]
    columns: list[float]
    values: np.ndarray


@dataclass(frozen=True)
class PeakMap:
    group: int
    center: float
    frame_count: int
    heatmap: Path
    matrix: Path


@dataclass(frozen=True)
class WindowMap:
    index: int
    start: float
    end: float
    heatmap: Path
    matrix: Path


@dataclass
class OutputLedger:
    waterfall_index: list[dict[str, object]]
    trace_similarity: list[dict[str, object]]
    map_summary: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Integrated XY files or directories. Defaults to Cell 14 + Cell 29.",
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--types",
        nargs="+",
        choices=CORRELATION_TYPES,
        default=list(CORRELATION_TYPES),
    )
    parser.add_argument(
        "--peak-groups",
        default=None,
        help="Comma-separated peak-group ids/ranges, e.g. 184,346 or 1-20.",
    )
    parser.add_argument(
        "--window-starts",
        default=None,
        help="Comma-separated nominal ACF window starts in degrees.",
    )
    parser.add_argument(
        "--frames",
        default=None,
        help="Comma-separated frame labels or F-indices for within-frame plots.",
    )
    parser.add_argument(
        "--reference-frame",
        default=None,
        help="Optional fixed frame label/F-index. Default: medoid chosen per map.",
    )
    parser.add_argument(
        "--reference-window-start",
        type=float,
        default=None,
        help="Optional fixed within-frame reference window. Default: medoid per frame.",
    )
    parser.add_argument("--roi-half-width", type=float, default=None)
    parser.add_argument("--roi-sideband-gap", type=float, default=0.03)
    parser.add_argument("--roi-sideband-width", type=float, default=0.08)
    parser.add_argument("--position-half-width", type=float, default=0.18)
    parser.add_argument("--acf-window-width", type=float, default=5.0)
    parser.add_argument("--acf-window-step", type=float, default=1.0)
    parser.add_argument("--acf-grid-step", type=float, default=0.02)
    parser.add_argument("--acf-shift-tolerance", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated safety guard; interrupted plot runs must use a new output directory.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Refresh frame-pair and peak-to-window summary tables without rendering PNGs.",
    )
    parser.add_argument(
        "--max-peak-plots",
        type=int,
        default=None,
        help="Testing aid: cap selected peak groups; does not affect ACF plots.",
    )
    return parser.parse_args()


def parse_id_selection(text: str | None) -> set[int] | None:
    if text is None:
        return None
    selected: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            selected.update(range(min(start, end), max(start, end) + 1))
        else:
            selected.add(int(token))
    return selected


def parse_float_selection(text: str | None) -> list[float] | None:
    if text is None:
        return None
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def compact_pressure(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "unknown P"
    return f"{value:g} GPa"


def compact_number(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 1.x/2.x compatible trapezoidal integration."""
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return float(implementation(y, x))


def csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return ""
    return value


def write_dict_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def relative_to_output(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def read_key_value(path: Path, key: str, default: float) -> float:
    if not path.exists():
        return default
    pattern = re.compile(rf"^{re.escape(key)}:\s*([-+0-9.eE]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            return float(match.group(1))
    return default


def read_feature_table(path: Path) -> FeatureTable:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"Empty feature table: {path}")
    columns = [float(value) for value in rows[0][1:]]
    labels = [row[0] for row in rows[1:]]
    values = np.full((len(labels), len(columns)), np.nan, dtype=float)
    for row_index, row in enumerate(rows[1:]):
        for col_index, value in enumerate(row[1 : len(columns) + 1]):
            if value.strip():
                values[row_index, col_index] = float(value)
    return FeatureTable(labels=labels, columns=columns, values=values)


def read_lower_triangle_matrix(path: Path) -> MatrixData:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"Empty matrix: {path}")
    column_labels = rows[0][1:]
    row_labels = [row[0] for row in rows[1:]]
    if row_labels != column_labels:
        raise ValueError(f"Matrix labels differ between rows and columns: {path}")
    size = len(row_labels)
    lower = np.full((size, size), np.nan, dtype=float)
    for row_index, row in enumerate(rows[1:]):
        for col_index, value in enumerate(row[1 : size + 1]):
            if value.strip():
                lower[row_index, col_index] = float(value)
    full = lower.copy()
    transpose_finite = np.isfinite(lower.T) & ~np.isfinite(full)
    full[transpose_finite] = lower.T[transpose_finite]
    return MatrixData(labels=row_labels, lower=lower, full=full)


def read_peak_maps(root: Path, position: bool = False) -> list[PeakMap]:
    index_name = "per_peak_position_map_index.csv" if position else "per_peak_map_index.csv"
    maps: list[PeakMap] = []
    with (root / index_name).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            maps.append(
                PeakMap(
                    group=int(row["peak_group"]),
                    center=float(row["group_two_theta"]),
                    frame_count=int(row["frame_count"]),
                    heatmap=root / row["heatmap"],
                    matrix=root / row["matrix"],
                )
            )
    return maps


def read_same_window_maps(root: Path, width: float) -> list[WindowMap]:
    maps: list[WindowMap] = []
    summary_path = root / "same_window_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            start = float(row["start_deg"])
            end = float(row["end_deg"])
            stem = f"window_{start:.1f}_{end:.1f}"
            maps.append(
                WindowMap(
                    index=index,
                    start=start,
                    end=end,
                    heatmap=root / "heatmaps" / f"{stem}_same_window_acf.png",
                    matrix=root / "matrices" / f"{stem}_same_window_acf.csv",
                )
            )
    if any(not math.isclose(item.end - item.start, width, abs_tol=1e-6) for item in maps):
        raise ValueError("Stored same-window width does not match requested ACF window width.")
    return maps


def load_patterns(inputs: Sequence[Path], labels: Sequence[str]) -> list[cip.Pattern]:
    files = cip.discover_xy_files(list(inputs))
    if not files:
        raise FileNotFoundError("No XY files found.")
    loader_args = SimpleNamespace(min_two_theta=2.0, max_two_theta=None)
    patterns = [cip.load_pattern(path, loader_args) for path in files]
    patterns = sorted(
        patterns,
        key=lambda pattern: (
            float("inf") if pattern.pressure_gpa is None else pattern.pressure_gpa,
            pattern.label,
        ),
    )
    cip.ensure_unique_pattern_labels(patterns)
    by_label = {pattern.label: pattern for pattern in patterns}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ValueError(f"XY patterns missing labels from feature table: {missing}")
    ordered = [by_label[label] for label in labels]
    if len(ordered) != len(patterns):
        extras = sorted(set(by_label) - set(labels))
        raise ValueError(f"XY inputs contain extra frames not in suite: {extras}")
    return ordered


def frame_code(index: int) -> str:
    return f"F{index:02d}"


def source_series(pattern: cip.Pattern) -> str:
    name = pattern.path.parent.name.replace("_integrated", "")
    return name.replace("Cell_", "Cell ")


def resolve_frame_reference(value: str | None, labels: Sequence[str]) -> int | None:
    if value is None:
        return None
    normalized = value.strip()
    match = re.fullmatch(r"[Ff](\d+)", normalized)
    if match:
        index = int(match.group(1))
        if not 0 <= index < len(labels):
            raise ValueError(f"Reference frame is outside 0..{len(labels) - 1}: {value}")
        return index
    if normalized.isdigit():
        index = int(normalized)
        if 0 <= index < len(labels):
            return index
    if normalized in labels:
        return labels.index(normalized)
    raise ValueError(f"Unknown frame reference: {value}")


def resolve_frame_selection(value: str | None, labels: Sequence[str]) -> set[int] | None:
    if value is None:
        return None
    return {resolve_frame_reference(token.strip(), labels) for token in value.split(",") if token.strip()}


def choose_medoid(matrix: np.ndarray, candidates: np.ndarray) -> int:
    indices = np.flatnonzero(candidates)
    if not len(indices):
        finite_counts = np.sum(np.isfinite(matrix), axis=1)
        if np.max(finite_counts) == 0:
            return 0
        return int(np.argmax(finite_counts))
    if len(indices) == 1:
        return int(indices[0])
    center_index = float(np.median(indices))
    ranking: list[tuple[float, int, float, int]] = []
    for index in indices:
        values = matrix[index, indices].copy()
        values[indices == index] = np.nan
        finite = values[np.isfinite(values)]
        median = float(np.nanmedian(finite)) if len(finite) else -np.inf
        support = int(len(finite))
        ranking.append((median, support, -abs(float(index) - center_index), -int(index)))
    best = max(range(len(ranking)), key=lambda offset: ranking[offset])
    return int(indices[best])


def choose_reference(
    matrix: np.ndarray,
    candidates: np.ndarray,
    fixed_reference: int | None,
) -> tuple[int, str]:
    if fixed_reference is not None:
        return int(fixed_reference), "user"
    return choose_medoid(matrix, candidates), "medoid"


def reference_scores(matrix: np.ndarray, reference: int) -> np.ndarray:
    values = matrix[reference].copy()
    values[reference] = 1.0
    return values


def display_similarity(raw_score: float, correlation_type: str) -> float:
    if not np.isfinite(raw_score):
        return np.nan
    if correlation_type in {"same-window", "within-frame"} or "acf" in correlation_type:
        return float(np.clip((raw_score + 1.0) / 2.0, 0.0, 1.0))
    return float(np.clip(raw_score, 0.0, 1.0))


def line_color(score: float) -> tuple[float, float, float, float] | str:
    if not np.isfinite(score):
        return MISSING_COLOR
    return SIMILARITY_CMAP(SIMILARITY_NORM(score))


def figure_for_rows(
    row_count: int,
    title: str,
    subtitle: str,
    ytick_labels: Sequence[str],
) -> tuple[plt.Figure, plt.Axes]:
    height = max(7.0, 0.48 * row_count + 3.2)
    fig, ax = plt.subplots(figsize=(16.0, height))
    fig.subplots_adjust(left=0.08, right=0.60, top=0.87, bottom=0.13)
    fig.suptitle(title, x=0.08, y=0.965, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color="#4d4d4d")
    offsets = np.arange(row_count, dtype=float)
    ax.set_yticks(offsets)
    ax.set_yticklabels(ytick_labels, fontsize=8.5)
    ax.set_ylim(-0.45, row_count - 0.15)
    ax.grid(axis="x", color="#dddddd", linewidth=0.55, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", labelsize=9)
    return fig, ax


def add_colorbar(fig: plt.Figure, label: str) -> None:
    cax = fig.add_axes([0.925, 0.23, 0.014, 0.50])
    mappable = ScalarMappable(norm=SIMILARITY_NORM, cmap=SIMILARITY_CMAP)
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label(label, fontsize=8.5, labelpad=8)


def add_right_label(ax: plt.Axes, y: float, text: str, color: str = "#333333") -> None:
    ax.text(
        1.025,
        y,
        text,
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8.1,
        color=color,
        clip_on=False,
    )


def add_footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.08, 0.035, text, ha="left", va="bottom", fontsize=8, color="#5a5a5a")


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        fig.savefig(temporary, dpi=dpi, facecolor="white", format=path.suffix.lstrip("."))
        temporary.replace(path)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()


def matrix_pair_extremes(
    matrix: np.ndarray,
    labels: Sequence[str],
    invalid_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, object]:
    finite: list[tuple[float, int, int]] = []
    clean: list[tuple[float, int, int]] = []
    invalid_pairs = invalid_pairs or set()
    for i in range(len(labels)):
        for j in range(i):
            value = float(matrix[i, j])
            if not np.isfinite(value):
                continue
            item = (value, i, j)
            finite.append(item)
            if (i, j) not in invalid_pairs:
                clean.append(item)

    def describe(items: list[tuple[float, int, int]], prefix: str) -> dict[str, object]:
        if not items:
            return {
                f"{prefix}_score": np.nan,
                f"{prefix}_item_a": "",
                f"{prefix}_item_b": "",
            }
        value, i, j = max(items) if prefix.startswith("highest") else min(items)
        return {
            f"{prefix}_score": value,
            f"{prefix}_item_a": labels[i],
            f"{prefix}_item_b": labels[j],
        }

    result: dict[str, object] = {
        "finite_pair_count": len(finite),
        "raw_score_min": min((item[0] for item in finite), default=np.nan),
        "raw_score_median": float(np.median([item[0] for item in finite])) if finite else np.nan,
        "raw_score_max": max((item[0] for item in finite), default=np.nan),
    }
    result.update(describe(finite, "highest_source"))
    result.update(describe(finite, "lowest_source"))
    result.update(describe(clean, "highest_non_degenerate"))
    return result


def peak_neighbor_gaps(maps: Sequence[PeakMap]) -> dict[int, float]:
    sorted_maps = sorted(maps, key=lambda item: item.center)
    result: dict[int, float] = {}
    for index, item in enumerate(sorted_maps):
        gaps: list[float] = []
        if index > 0:
            gaps.append(item.center - sorted_maps[index - 1].center)
        if index + 1 < len(sorted_maps):
            gaps.append(sorted_maps[index + 1].center - item.center)
        result[item.group] = min(gaps) if gaps else np.nan
    return result


def area_profile(
    pattern: cip.Pattern,
    normalized: np.ndarray,
    center: float,
    roi_half_width: float,
    sideband_gap: float,
    sideband_width: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = pattern.two_theta
    outer_half_width = roi_half_width + sideband_gap + sideband_width
    local = (x >= center - outer_half_width) & (x <= center + outer_half_width)
    roi = (x >= center - roi_half_width) & (x <= center + roi_half_width)
    left_side = (
        (x >= center - roi_half_width - sideband_gap - sideband_width)
        & (x < center - roi_half_width - sideband_gap)
    )
    right_side = (
        (x > center + roi_half_width + sideband_gap)
        & (x <= center + roi_half_width + sideband_gap + sideband_width)
    )
    sideband = left_side | right_side
    if np.any(sideband):
        background = float(np.nanmedian(normalized[sideband]))
    elif np.any(roi):
        background = float(np.nanpercentile(normalized[roi], 10))
    else:
        background = 0.0
    if np.count_nonzero(roi) >= 2:
        signal = np.clip(normalized[roi] - background, 0.0, None)
        area = integrate_trapezoid(signal, x[roi])
    else:
        area = 0.0
    return x[local], normalized[local], background, area


def plot_area_waterfall(
    item: PeakMap,
    matrix: MatrixData,
    patterns: Sequence[cip.Pattern],
    normalized_patterns: Sequence[np.ndarray],
    areas: np.ndarray,
    presence: np.ndarray,
    reference: int,
    strategy: str,
    output_path: Path,
    roi_half_width: float,
    sideband_gap: float,
    sideband_width: float,
    nearest_gap: float,
    dpi: int,
    ledger: OutputLedger,
    output_root: Path,
) -> None:
    scores = reference_scores(matrix.full, reference)
    profiles = [
        area_profile(pattern, normalized, item.center, roi_half_width, sideband_gap, sideband_width)
        for pattern, normalized in zip(patterns, normalized_patterns)
    ]
    all_y = np.concatenate([profile[1] for profile in profiles if len(profile[1])])
    y_min = float(np.nanmin(all_y)) if len(all_y) else 0.0
    y_max = float(np.nanmax(all_y)) if len(all_y) else 1.0
    shared_span = max(y_max - y_min, 1e-12)
    gain = 0.72 / shared_span
    present_zero = (presence > 0) & (areas <= ZERO_AREA_EPS)
    degenerate_pairs = {
        (i, j)
        for i in range(len(patterns))
        for j in range(i)
        if present_zero[i] and present_zero[j] and np.isfinite(matrix.full[i, j])
    }
    finite_scores = scores[np.isfinite(scores)]
    score_min = float(np.nanmin(finite_scores)) if len(finite_scores) else np.nan
    score_max = float(np.nanmax(finite_scores)) if len(finite_scores) else np.nan
    title = (
        f"Peak {item.group:04d} at {item.center:.3f}° — ROI-area similarity to "
        f"{patterns[reference].label} spans {score_min:.2f}–{score_max:.2f}"
    )
    overlap_text = (
        f"nearest group Δ={nearest_gap:.3f}°; ROIs overlap"
        if np.isfinite(nearest_gap) and nearest_gap < 2.0 * roi_half_width
        else "ROI isolated at the configured half-width"
    )
    subtitle = (
        f"Reference {frame_code(reference)} ({strategy}); ROI ±{roi_half_width:.3f}°; "
        f"shared normalized-intensity scale; {overlap_text}."
    )
    fig, ax = figure_for_rows(
        len(patterns),
        title,
        subtitle,
        [frame_code(index) for index in range(len(patterns))],
    )
    ax.axvspan(
        item.center - roi_half_width,
        item.center + roi_half_width,
        color="#e8e8e8",
        alpha=0.55,
        zorder=0,
        label="integration ROI",
    )
    ax.axvline(item.center, color="#777777", linewidth=0.7, linestyle=":", zorder=0)
    area_differences: list[float] = []
    for index, (pattern, profile) in enumerate(zip(patterns, profiles)):
        x_local, y_local, background, recomputed_area = profile
        area_differences.append(abs(float(areas[index]) - recomputed_area))
        raw_score = float(scores[index])
        shown_score = display_similarity(raw_score, "area")
        color = line_color(shown_score)
        y_display = index + (y_local - y_min) * gain
        linewidth = 2.3 if index == reference else 1.25
        line = ax.plot(x_local, y_display, color=color, linewidth=linewidth, zorder=3)[0]
        if index == reference:
            line.set_path_effects([path_effects.Stroke(linewidth=3.8, foreground="#222222"), path_effects.Normal()])
        roi = (x_local >= item.center - roi_half_width) & (x_local <= item.center + roi_half_width)
        baseline_display = index + (background - y_min) * gain
        if np.any(roi):
            filled_top = index + (np.maximum(y_local[roi], background) - y_min) * gain
            ax.fill_between(
                x_local[roi],
                baseline_display,
                filled_top,
                color=color,
                alpha=0.22,
                linewidth=0,
                zorder=2,
            )
            ax.hlines(
                baseline_display,
                item.center - roi_half_width,
                item.center + roi_half_width,
                color="#666666",
                linewidth=0.55,
                linestyle="--",
                zorder=2,
            )
        is_present = bool(presence[index] > 0)
        warning = "present-zero" if present_zero[index] else ("present" if is_present else "absent")
        ref_tag = " · REF" if index == reference else ""
        label = (
            f"{frame_code(index)} · {source_series(pattern)} · {compact_pressure(pattern.pressure_gpa)} · "
            f"A={compact_number(float(areas[index]))} · s={compact_number(shown_score)} · {warning}{ref_tag}"
        )
        add_right_label(ax, float(index) + 0.22, label, WARNING_COLOR if present_zero[index] else "#333333")
        ledger.trace_similarity.append(
            {
                "correlation_type": "peak_area",
                "feature_id": f"peak_{item.group:04d}",
                "peak_group": item.group,
                "peak_center_deg": item.center,
                "reference_index": reference,
                "reference_frame": patterns[reference].label,
                "target_index": index,
                "target_frame": pattern.label,
                "pressure_GPa": pattern.pressure_gpa,
                "raw_score": raw_score,
                "display_similarity_0_1": shown_score,
                "roi_area": float(areas[index]),
                "present": int(is_present),
                "warning": "present_but_zero_area" if present_zero[index] else "",
                "source_matrix": str(item.matrix),
                "waterfall": str(output_path),
            }
        )
    outer_half = roi_half_width + sideband_gap + sideband_width
    ax.set_xlim(item.center - outer_half, item.center + outer_half)
    ax.set_xlabel(r"$2\theta$ (degrees)")
    add_colorbar(fig, "Area similarity s (0–1)")
    degenerate_note = (
        f" Source matrix contains {len(degenerate_pairs)} degenerate high pair(s): both frames are present but both ROI areas are zero."
        if degenerate_pairs
        else ""
    )
    add_footnote(
        fig,
        "Area feature = positive signal above sideband median after per-pattern P5/P99 normalization. "
        "Gray band is the integration ROI; fills show the integrated signal." + degenerate_note,
    )
    save_figure(fig, output_path, dpi)

    warnings: list[str] = []
    if degenerate_pairs:
        warnings.append(f"degenerate_zero_area_pairs={len(degenerate_pairs)}")
    if np.isfinite(nearest_gap) and nearest_gap < 2.0 * roi_half_width:
        warnings.append("overlapping_peak_roi")
    pair_summary = matrix_pair_extremes(matrix.full, matrix.labels, degenerate_pairs)
    ledger.map_summary.append(
        {
            "correlation_type": "peak_area",
            "feature_id": f"peak_{item.group:04d}",
            "peak_group": item.group,
            "peak_center_deg": item.center,
            "frame_count": item.frame_count,
            "reference_frame": patterns[reference].label,
            "reference_pressure_GPa": patterns[reference].pressure_gpa,
            "reference_strategy": strategy,
            "degenerate_zero_area_pair_count": len(degenerate_pairs),
            "nearest_peak_group_gap_deg": nearest_gap,
            "max_recomputed_area_abs_error": max(area_differences, default=np.nan),
            "warning_flags": ";".join(warnings),
            **pair_summary,
            "source_heatmap": str(item.heatmap),
            "source_matrix": str(item.matrix),
            "waterfall": str(output_path),
        }
    )
    ledger.waterfall_index.append(
        {
            "correlation_type": "peak_area",
            "feature_id": f"peak_{item.group:04d}",
            "feature_label": f"peak group {item.group} at {item.center:.6f} deg",
            "reference": patterns[reference].label,
            "reference_strategy": strategy,
            "source_heatmap": relative_to_output(item.heatmap, output_root),
            "source_matrix": relative_to_output(item.matrix, output_root),
            "waterfall": relative_to_output(output_path, output_root),
            "warning_flags": ";".join(warnings),
        }
    )


def local_normalized_profile(pattern: cip.Pattern, center: float, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    keep = (pattern.two_theta >= center - half_width) & (pattern.two_theta <= center + half_width)
    x = pattern.two_theta[keep]
    y = pattern.intensity[keep].astype(float)
    if not len(y):
        return x, y
    low = float(np.nanpercentile(y, 5))
    high = float(np.nanpercentile(y, 99))
    span = max(high - low, 1e-12)
    normalized = np.clip((y - low) / span, -0.08, 1.15)
    return x, normalized


def plot_position_waterfall(
    item: PeakMap,
    matrix: MatrixData,
    patterns: Sequence[cip.Pattern],
    positions: np.ndarray,
    presence: np.ndarray,
    reference: int,
    strategy: str,
    output_path: Path,
    half_width: float,
    position_tolerance: float,
    nearest_gap: float,
    dpi: int,
    ledger: OutputLedger,
    output_root: Path,
) -> None:
    scores = reference_scores(matrix.full, reference)
    profiles = [local_normalized_profile(pattern, item.center, half_width) for pattern in patterns]
    finite_scores = scores[np.isfinite(scores)]
    score_min = float(np.nanmin(finite_scores)) if len(finite_scores) else np.nan
    score_max = float(np.nanmax(finite_scores)) if len(finite_scores) else np.nan
    title = (
        f"Peak {item.group:04d} at {item.center:.3f}° — position similarity to "
        f"{patterns[reference].label} spans {score_min:.2f}–{score_max:.2f}"
    )
    subtitle = (
        f"Reference {frame_code(reference)} ({strategy}); each profile is locally normalized; "
        f"score reaches zero at |Δ2θ| ≥ {position_tolerance:.3f}°."
    )
    fig, ax = figure_for_rows(
        len(patterns),
        title,
        subtitle,
        [frame_code(index) for index in range(len(patterns))],
    )
    ax.axvline(item.center, color="#777777", linewidth=0.7, linestyle=":", zorder=0)
    marker_x: list[float] = []
    marker_y: list[float] = []
    marker_colors: list[object] = []
    reference_position = float(positions[reference])
    for index, (pattern, profile) in enumerate(zip(patterns, profiles)):
        x_local, y_local = profile
        raw_score = float(scores[index])
        shown_score = display_similarity(raw_score, "position")
        color = line_color(shown_score)
        y_display = index + y_local * 0.68
        linewidth = 2.3 if index == reference else 1.25
        line = ax.plot(x_local, y_display, color=color, linewidth=linewidth, zorder=3)[0]
        if index == reference:
            line.set_path_effects([path_effects.Stroke(linewidth=3.8, foreground="#222222"), path_effects.Normal()])
        finite_position = bool(presence[index] > 0 and np.isfinite(positions[index]))
        if finite_position and len(x_local):
            center = float(positions[index])
            curve_y = float(np.interp(center, x_local, y_local))
            point_y = index + curve_y * 0.68
            marker_x.append(center)
            marker_y.append(point_y)
            marker_colors.append(color)
        else:
            center = np.nan
        delta = center - reference_position if np.isfinite(center) and np.isfinite(reference_position) else np.nan
        state = "present" if finite_position else "absent"
        ref_tag = " · REF" if index == reference else ""
        label = (
            f"{frame_code(index)} · {source_series(pattern)} · {compact_pressure(pattern.pressure_gpa)} · "
            f"center={compact_number(center, 4)}° · Δ={compact_number(delta, 4)}° · "
            f"s={compact_number(shown_score)} · {state}{ref_tag}"
        )
        add_right_label(ax, float(index) + 0.20, label)
        ledger.trace_similarity.append(
            {
                "correlation_type": "peak_position",
                "feature_id": f"peak_{item.group:04d}",
                "peak_group": item.group,
                "peak_center_deg": item.center,
                "reference_index": reference,
                "reference_frame": patterns[reference].label,
                "target_index": index,
                "target_frame": pattern.label,
                "pressure_GPa": pattern.pressure_gpa,
                "raw_score": raw_score,
                "display_similarity_0_1": shown_score,
                "detected_center_deg": center,
                "delta_from_reference_deg": delta,
                "present": int(finite_position),
                "source_matrix": str(item.matrix),
                "waterfall": str(output_path),
            }
        )
    if marker_x:
        ax.plot(marker_x, marker_y, color="#222222", linewidth=0.9, alpha=0.75, zorder=4)
        for x_value, y_value, color in zip(marker_x, marker_y, marker_colors):
            ax.scatter([x_value], [y_value], s=24, color=[color], edgecolor="#222222", linewidth=0.55, zorder=5)
    ax.set_xlim(item.center - half_width, item.center + half_width)
    ax.set_xlabel(r"$2\theta$ (degrees)")
    add_colorbar(fig, "Position similarity s (0–1)")
    add_footnote(
        fig,
        "Dots and connector show the selected detected center. In this legacy XY suite these are detector-sampled "
        "positions, not nonlinear fitted centers; an apparent disappearance can also reflect the 0.02° grouping gate.",
    )
    save_figure(fig, output_path, dpi)

    warnings: list[str] = ["detected_center_not_fitted_center"]
    if np.isfinite(nearest_gap) and nearest_gap < 2.0 * position_tolerance:
        warnings.append("nearby_peak_groups_may_split_shift")
    pair_summary = matrix_pair_extremes(matrix.full, matrix.labels)
    ledger.map_summary.append(
        {
            "correlation_type": "peak_position",
            "feature_id": f"peak_{item.group:04d}",
            "peak_group": item.group,
            "peak_center_deg": item.center,
            "frame_count": item.frame_count,
            "reference_frame": patterns[reference].label,
            "reference_pressure_GPa": patterns[reference].pressure_gpa,
            "reference_strategy": strategy,
            "position_tolerance_deg": position_tolerance,
            "nearest_peak_group_gap_deg": nearest_gap,
            "warning_flags": ";".join(warnings),
            **pair_summary,
            "source_heatmap": str(item.heatmap),
            "source_matrix": str(item.matrix),
            "waterfall": str(output_path),
        }
    )
    ledger.waterfall_index.append(
        {
            "correlation_type": "peak_position",
            "feature_id": f"peak_{item.group:04d}",
            "feature_label": f"peak group {item.group} at {item.center:.6f} deg",
            "reference": patterns[reference].label,
            "reference_strategy": strategy,
            "source_heatmap": relative_to_output(item.heatmap, output_root),
            "source_matrix": relative_to_output(item.matrix, output_root),
            "waterfall": relative_to_output(output_path, output_root),
            "warning_flags": ";".join(warnings),
        }
    )


def finite_curve(curve: np.ndarray | None) -> bool:
    return curve is not None and len(curve) >= 3 and np.all(np.isfinite(curve))


def best_stored_direction_acf_pair(
    reference: int,
    target: int,
    window_index: int,
    fingerprints: dict[tuple[int, int], np.ndarray | None],
    window_count: int,
    neighbor: int,
) -> tuple[float, int, np.ndarray | None, np.ndarray | None, str]:
    """Reproduce the directional legacy lower-triangle neighboring-window search.

    Returns raw score, winning delta, colored target curve, dashed reference-side
    partner, and the side on which the window shift occurred.
    """
    if target == reference:
        curve = fingerprints[(reference, window_index)]
        return 1.0, 0, curve, curve, "none"

    best_score = np.nan
    best_delta = 0
    target_curve: np.ndarray | None = None
    reference_partner: np.ndarray | None = None
    shift_side = "target" if reference > target else "reference"
    for delta in range(-neighbor, neighbor + 1):
        shifted = window_index + delta
        if shifted < 0 or shifted >= window_count:
            continue
        if reference > target:
            reference_curve = fingerprints[(reference, window_index)]
            candidate_target = fingerprints[(target, shifted)]
        else:
            reference_curve = fingerprints[(reference, shifted)]
            candidate_target = fingerprints[(target, window_index)]
        score = wacf.pearson(reference_curve, candidate_target)
        if np.isnan(score):
            continue
        if np.isnan(best_score) or score > best_score:
            best_score = float(score)
            best_delta = delta
            target_curve = candidate_target
            reference_partner = reference_curve
    return best_score, best_delta, target_curve, reference_partner, shift_side


def acf_display_gain(curves: Iterable[np.ndarray | None]) -> float:
    arrays = [np.abs(curve) for curve in curves if finite_curve(curve)]
    if not arrays:
        return 1.0
    maximum = float(np.nanmax(np.concatenate(arrays)))
    return 0.37 / max(maximum, 1e-12)


def plot_same_window_waterfall(
    item: WindowMap,
    matrix: MatrixData,
    patterns: Sequence[cip.Pattern],
    starts: np.ndarray,
    fingerprints: dict[tuple[int, int], np.ndarray | None],
    reference: int,
    strategy: str,
    output_path: Path,
    grid_step: float,
    window_step: float,
    shift_tolerance: float,
    dpi: int,
    ledger: OutputLedger,
    output_root: Path,
) -> None:
    scores = reference_scores(matrix.full, reference)
    neighbor = int(round(shift_tolerance / window_step))
    pair_data = [
        best_stored_direction_acf_pair(reference, index, item.index, fingerprints, len(starts), neighbor)
        for index in range(len(patterns))
    ]
    gain = acf_display_gain(
        [curve for _, _, target_curve, partner, _ in pair_data for curve in (target_curve, partner)]
    )
    finite_scores = scores[np.isfinite(scores)]
    raw_min = float(np.nanmin(finite_scores)) if len(finite_scores) else np.nan
    raw_max = float(np.nanmax(finite_scores)) if len(finite_scores) else np.nan
    title = (
        f"{item.start:.1f}–{item.end:.1f}° ACF — Pearson r to {patterns[reference].label} "
        f"spans {raw_min:+.2f}–{raw_max:+.2f}"
    )
    subtitle = (
        f"Reference {frame_code(reference)} ({strategy}); colored curve = actual frame-side feature; "
        "dashed gray = actual reference-side partner selected by the legacy lower-triangle search."
    )
    fig, ax = figure_for_rows(
        len(patterns),
        title,
        subtitle,
        [frame_code(index) for index in range(len(patterns))],
    )
    max_score_error = 0.0
    for index, (pattern, pair) in enumerate(zip(patterns, pair_data)):
        recomputed_score, delta, target_curve, reference_partner, shift_side = pair
        stored_score = float(scores[index])
        shown_score = display_similarity(stored_score, "same-window")
        color = line_color(shown_score)
        if np.isfinite(stored_score) and np.isfinite(recomputed_score):
            max_score_error = max(max_score_error, abs(stored_score - recomputed_score))
        if finite_curve(reference_partner) and index != reference:
            lag = np.arange(1, len(reference_partner) + 1, dtype=float) * grid_step
            ax.plot(
                lag,
                index + reference_partner * gain,
                color=REFERENCE_DASH,
                linewidth=0.8,
                linestyle=(0, (3, 2)),
                alpha=0.58,
                zorder=2,
            )
        if finite_curve(target_curve):
            lag = np.arange(1, len(target_curve) + 1, dtype=float) * grid_step
            linewidth = 2.3 if index == reference else 1.3
            line = ax.plot(lag, index + target_curve * gain, color=color, linewidth=linewidth, zorder=3)[0]
            if index == reference:
                line.set_path_effects([path_effects.Stroke(linewidth=3.8, foreground="#222222"), path_effects.Normal()])
        if index == reference:
            shift_label = "nominal window · REF"
            actual_target_start = item.start
            actual_partner_start = item.start
        elif shift_side == "target":
            actual_target_start = item.start + delta * window_step
            actual_partner_start = item.start
            shift_label = f"frame window Δ={delta * window_step:+.1f}°"
        else:
            actual_target_start = item.start
            actual_partner_start = item.start + delta * window_step
            shift_label = f"reference partner Δ={delta * window_step:+.1f}°"
        label = (
            f"{frame_code(index)} · {source_series(pattern)} · {compact_pressure(pattern.pressure_gpa)} · "
            f"r={compact_number(stored_score)} · color s={compact_number(shown_score)} · {shift_label}"
        )
        add_right_label(ax, float(index), label)
        ledger.trace_similarity.append(
            {
                "correlation_type": "same_window_acf",
                "feature_id": f"window_{item.start:.1f}_{item.end:.1f}",
                "window_start_deg": item.start,
                "window_end_deg": item.end,
                "reference_index": reference,
                "reference_frame": patterns[reference].label,
                "target_index": index,
                "target_frame": pattern.label,
                "pressure_GPa": pattern.pressure_gpa,
                "raw_pearson_r": stored_score,
                "display_similarity_0_1": shown_score,
                "recomputed_pearson_r": recomputed_score,
                "score_abs_error": abs(stored_score - recomputed_score)
                if np.isfinite(stored_score) and np.isfinite(recomputed_score)
                else np.nan,
                "shift_side": shift_side,
                "winning_shift_deg": delta * window_step,
                "actual_target_window_start_deg": actual_target_start,
                "actual_reference_partner_window_start_deg": actual_partner_start,
                "source_matrix": str(item.matrix),
                "waterfall": str(output_path),
            }
        )
    sample_curve = next(
        (target for _, _, target, _, _ in pair_data if finite_curve(target)),
        None,
    )
    max_lag = len(sample_curve) * grid_step if sample_curve is not None else item.end - item.start
    ax.set_xlim(grid_step, max_lag)
    ax.set_xlabel(r"ACF lag $\Delta 2\theta$ (degrees)")
    add_colorbar(fig, "Display similarity s=(r+1)/2")
    add_footnote(
        fig,
        "The plotted feature is the standardized positive-lag ACF (lag zero removed). Raw Pearson r is shown "
        "in every label; only the color encoding is transformed to a fixed 0–1 scale.",
    )
    save_figure(fig, output_path, dpi)

    pair_summary = matrix_pair_extremes(matrix.full, matrix.labels)
    warnings = ["directional_neighbor_window_search"]
    if max_score_error > 1e-4:
        warnings.append("recomputed_score_mismatch")
    ledger.map_summary.append(
        {
            "correlation_type": "same_window_acf",
            "feature_id": f"window_{item.start:.1f}_{item.end:.1f}",
            "window_start_deg": item.start,
            "window_end_deg": item.end,
            "reference_frame": patterns[reference].label,
            "reference_pressure_GPa": patterns[reference].pressure_gpa,
            "reference_strategy": strategy,
            "max_recomputed_score_abs_error": max_score_error,
            "warning_flags": ";".join(warnings),
            **pair_summary,
            "source_heatmap": str(item.heatmap),
            "source_matrix": str(item.matrix),
            "waterfall": str(output_path),
        }
    )
    ledger.waterfall_index.append(
        {
            "correlation_type": "same_window_acf",
            "feature_id": f"window_{item.start:.1f}_{item.end:.1f}",
            "feature_label": f"nominal window {item.start:.3f}-{item.end:.3f} deg",
            "reference": patterns[reference].label,
            "reference_strategy": strategy,
            "source_heatmap": relative_to_output(item.heatmap, output_root),
            "source_matrix": relative_to_output(item.matrix, output_root),
            "waterfall": relative_to_output(output_path, output_root),
            "warning_flags": ";".join(warnings),
        }
    )


def plot_within_frame_waterfall(
    frame_index: int,
    matrix: MatrixData,
    patterns: Sequence[cip.Pattern],
    starts: np.ndarray,
    fingerprints: dict[tuple[int, int], np.ndarray | None],
    reference_window: int,
    strategy: str,
    source_heatmap: Path,
    source_matrix: Path,
    output_path: Path,
    grid_step: float,
    window_width: float,
    dpi: int,
    ledger: OutputLedger,
    output_root: Path,
) -> None:
    pattern = patterns[frame_index]
    scores = reference_scores(matrix.full, reference_window)
    curves = [fingerprints[(frame_index, index)] for index in range(len(starts))]
    gain = acf_display_gain(curves)
    finite_scores = scores[np.isfinite(scores)]
    raw_min = float(np.nanmin(finite_scores)) if len(finite_scores) else np.nan
    raw_max = float(np.nanmax(finite_scores)) if len(finite_scores) else np.nan
    ref_start = float(starts[reference_window])
    title = (
        f"{pattern.label} — window ACF similarity to {ref_start:.1f}–{ref_start + window_width:.1f}° "
        f"spans r={raw_min:+.2f}–{raw_max:+.2f}"
    )
    subtitle = (
        f"{frame_code(frame_index)} · {source_series(pattern)} · {compact_pressure(pattern.pressure_gpa)}; "
        f"reference window chosen by {strategy}; dashed gray repeats the reference ACF for direct comparison."
    )
    ytick_labels = [f"{start:.0f}–{start + window_width:.0f}°" for start in starts]
    fig, ax = figure_for_rows(len(starts), title, subtitle, ytick_labels)
    reference_curve = curves[reference_window]
    for index, (start, curve) in enumerate(zip(starts, curves)):
        raw_score = float(scores[index])
        shown_score = display_similarity(raw_score, "within-frame")
        color = line_color(shown_score)
        if finite_curve(reference_curve) and index != reference_window:
            lag = np.arange(1, len(reference_curve) + 1, dtype=float) * grid_step
            ax.plot(
                lag,
                index + reference_curve * gain,
                color=REFERENCE_DASH,
                linewidth=0.8,
                linestyle=(0, (3, 2)),
                alpha=0.58,
                zorder=2,
            )
        if finite_curve(curve):
            lag = np.arange(1, len(curve) + 1, dtype=float) * grid_step
            linewidth = 2.3 if index == reference_window else 1.3
            line = ax.plot(lag, index + curve * gain, color=color, linewidth=linewidth, zorder=3)[0]
            if index == reference_window:
                line.set_path_effects([path_effects.Stroke(linewidth=3.8, foreground="#222222"), path_effects.Normal()])
        ref_tag = " · REF" if index == reference_window else ""
        label = (
            f"{start:.1f}–{start + window_width:.1f}° · r={compact_number(raw_score)} · "
            f"color s={compact_number(shown_score)}{ref_tag}"
        )
        add_right_label(ax, float(index), label)
        ledger.trace_similarity.append(
            {
                "correlation_type": "within_frame_window_acf",
                "feature_id": f"frame_{frame_index:04d}",
                "frame_index": frame_index,
                "frame": pattern.label,
                "pressure_GPa": pattern.pressure_gpa,
                "reference_window_index": reference_window,
                "reference_window_start_deg": ref_start,
                "target_window_index": index,
                "target_window_start_deg": float(start),
                "target_window_end_deg": float(start + window_width),
                "raw_pearson_r": raw_score,
                "display_similarity_0_1": shown_score,
                "source_matrix": str(source_matrix),
                "waterfall": str(output_path),
            }
        )
    sample_curve = next((curve for curve in curves if finite_curve(curve)), None)
    max_lag = len(sample_curve) * grid_step if sample_curve is not None else window_width
    ax.set_xlim(grid_step, max_lag)
    ax.set_xlabel(r"ACF lag $\Delta 2\theta$ (degrees)")
    add_colorbar(fig, "Display similarity s=(r+1)/2")
    add_footnote(
        fig,
        "Overlapping 5° windows are ordered by 2θ; adjacent windows share 80% of their samples. "
        "Raw Pearson r is shown; color uses the fixed 0–1 transform s=(r+1)/2.",
    )
    save_figure(fig, output_path, dpi)

    pair_summary = matrix_pair_extremes(matrix.full, matrix.labels)
    ledger.map_summary.append(
        {
            "correlation_type": "within_frame_window_acf",
            "feature_id": f"frame_{frame_index:04d}",
            "frame_index": frame_index,
            "frame": pattern.label,
            "pressure_GPa": pattern.pressure_gpa,
            "reference_window_start_deg": ref_start,
            "reference_strategy": strategy,
            "warning_flags": "overlapping_windows_share_samples",
            **pair_summary,
            "source_heatmap": str(source_heatmap),
            "source_matrix": str(source_matrix),
            "waterfall": str(output_path),
        }
    )
    ledger.waterfall_index.append(
        {
            "correlation_type": "within_frame_window_acf",
            "feature_id": f"frame_{frame_index:04d}",
            "feature_label": f"{pattern.label} within-frame windows",
            "reference": f"{ref_start:.3f}-{ref_start + window_width:.3f} deg",
            "reference_strategy": strategy,
            "source_heatmap": relative_to_output(source_heatmap, output_root),
            "source_matrix": relative_to_output(source_matrix, output_root),
            "waterfall": relative_to_output(output_path, output_root),
            "warning_flags": "overlapping_windows_share_samples",
        }
    )


def frame_pair_summary(
    correlation_type: str,
    feature_matrices: Sequence[
        tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]
    ],
    patterns: Sequence[cip.Pattern],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(len(patterns)):
        for j in range(i):
            values: list[tuple[str, float, float]] = []
            shared_values: list[tuple[str, float, float]] = []
            joint_present = 0
            union_present = 0
            for feature_id, matrix, valid_frames, present_frames in feature_matrices:
                raw = float(matrix[i, j])
                if np.isfinite(raw):
                    item = (feature_id, raw, display_similarity(raw, correlation_type))
                    values.append(item)
                    if valid_frames is None or (bool(valid_frames[i]) and bool(valid_frames[j])):
                        shared_values.append(item)
                if present_frames is not None:
                    present_i = bool(present_frames[i])
                    present_j = bool(present_frames[j])
                    joint_present += int(present_i and present_j)
                    union_present += int(present_i or present_j)
            if not values:
                continue
            raw_values = np.array([item[1] for item in values], dtype=float)
            shown_values = np.array([item[2] for item in values], dtype=float)
            shared_raw = np.array([item[1] for item in shared_values], dtype=float)
            shared_shown = np.array([item[2] for item in shared_values], dtype=float)
            ranking_values = shared_values if shared_values else values
            top = sorted(ranking_values, key=lambda item: item[2], reverse=True)[:5]
            low = sorted(ranking_values, key=lambda item: item[2])[:5]
            rows.append(
                {
                    "correlation_type": correlation_type,
                    "frame_a_index": i,
                    "frame_a": patterns[i].label,
                    "pressure_a_GPa": patterns[i].pressure_gpa,
                    "frame_b_index": j,
                    "frame_b": patterns[j].label,
                    "pressure_b_GPa": patterns[j].pressure_gpa,
                    "finite_map_count": len(values),
                    "median_raw_score": float(np.median(raw_values)),
                    "mean_raw_score": float(np.mean(raw_values)),
                    "median_display_similarity_0_1": float(np.median(shown_values)),
                    "mean_display_similarity_0_1": float(np.mean(shown_values)),
                    "shared_valid_map_count": len(shared_values),
                    "peak_presence_intersection_count": joint_present
                    if union_present
                    else np.nan,
                    "peak_presence_union_count": union_present if union_present else np.nan,
                    "peak_presence_jaccard": joint_present / union_present
                    if union_present
                    else np.nan,
                    "median_shared_valid_raw_score": float(np.median(shared_raw))
                    if len(shared_raw)
                    else np.nan,
                    "mean_shared_valid_raw_score": float(np.mean(shared_raw))
                    if len(shared_raw)
                    else np.nan,
                    "median_shared_valid_display_similarity_0_1": float(np.median(shared_shown))
                    if len(shared_shown)
                    else np.nan,
                    "mean_shared_valid_display_similarity_0_1": float(np.mean(shared_shown))
                    if len(shared_shown)
                    else np.nan,
                    "high_similarity_map_count_s_ge_0p8": int(np.count_nonzero(shown_values >= 0.8)),
                    "low_similarity_map_count_s_le_0p2": int(np.count_nonzero(shown_values <= 0.2)),
                    "highest_shared_valid_similarity_features": ";".join(
                        f"{feature_id}:{shown:.3f}" for feature_id, _, shown in top
                    ),
                    "lowest_shared_valid_similarity_features": ";".join(
                        f"{feature_id}:{shown:.3f}" for feature_id, _, shown in low
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            row["correlation_type"],
            -float(row["mean_display_similarity_0_1"]),
            -int(row["high_similarity_map_count_s_ge_0p8"]),
            -int(row["finite_map_count"]),
        )
    )
    return rows


def collect_frame_pair_matrices(
    correlation_types: Sequence[str],
    peak_groups: Sequence[int],
    area_by_group: dict[int, PeakMap],
    position_by_group: dict[int, PeakMap],
    area_features: FeatureTable,
    presence_features: FeatureTable,
    position_features: FeatureTable,
    same_window_maps: Sequence[WindowMap],
) -> dict[
    str,
    list[tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]],
]:
    matrices: dict[
        str,
        list[tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]],
    ] = defaultdict(list)
    if "area" in correlation_types:
        for group in peak_groups:
            matrix = read_lower_triangle_matrix(area_by_group[group].matrix).full
            present = presence_features.values[:, group - 1] > 0
            valid = present & (area_features.values[:, group - 1] > ZERO_AREA_EPS)
            matrices["peak_area"].append((f"peak_{group:04d}", matrix, valid, present))
    if "position" in correlation_types:
        for group in peak_groups:
            matrix = read_lower_triangle_matrix(position_by_group[group].matrix).full
            valid = (presence_features.values[:, group - 1] > 0) & np.isfinite(
                position_features.values[:, group - 1]
            )
            matrices["peak_position"].append((f"peak_{group:04d}", matrix, valid, valid))
    if "same-window" in correlation_types:
        for item in same_window_maps:
            matrix = read_lower_triangle_matrix(item.matrix).full
            matrices["same_window_acf"].append(
                (f"window_{item.start:.1f}_{item.end:.1f}", matrix, None, None)
            )
    return matrices


def build_window_peak_membership(
    peak_maps: Sequence[PeakMap],
    window_maps: Sequence[WindowMap],
    output_root: Path,
    include_area: bool = True,
    include_position: bool = True,
    include_same_window: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in window_maps:
        for peak in peak_maps:
            if window.start <= peak.center < window.end:
                rows.append(
                    {
                        "window_id": f"window_{window.start:.1f}_{window.end:.1f}",
                        "window_start_deg": window.start,
                        "window_end_deg": window.end,
                        "peak_group": peak.group,
                        "peak_center_deg": peak.center,
                        "area_waterfall": relative_to_output(
                            output_root / "01_peak_area_waterfalls" / f"peak_{peak.group:04d}_area_waterfall.png",
                            output_root,
                        )
                        if include_area
                        else "",
                        "position_waterfall": relative_to_output(
                            output_root
                            / "02_peak_position_waterfalls"
                            / f"peak_{peak.group:04d}_position_waterfall.png",
                            output_root,
                        )
                        if include_position
                        else "",
                        "same_window_acf_waterfall": relative_to_output(
                            output_root
                            / "03_same_window_acf_waterfalls"
                            / f"window_{window.start:.0f}_{window.end:.0f}deg_acf_waterfall.png",
                            output_root,
                        )
                        if include_same_window
                        else "",
                        "relationship_note": "peak center lies inside nominal ACF window; membership is contextual, not causal attribution",
                    }
                )
    return rows


def build_window_peak_score_associations(
    peak_maps: Sequence[PeakMap],
    position_by_group: dict[int, PeakMap],
    window_maps: Sequence[WindowMap],
    area_features: FeatureTable,
    presence_features: FeatureTable,
    position_features: FeatureTable,
    output_root: Path,
    correlation_types: Sequence[str],
) -> list[dict[str, object]]:
    """Relate peak-map and ACF-map pairwise score vectors descriptively.

    A peak is first required to lie inside the nominal window.  For each such
    peak/window pair, Spearman and Pearson associations are then computed over
    the same finite frame pairs.  Area comparisons require both frames to be
    present with strictly positive ROI area, excluding legacy 0/0 highs.
    """
    peak_type_specs: list[str] = []
    if "area" in correlation_types:
        peak_type_specs.append("peak_area")
    if "position" in correlation_types:
        peak_type_specs.append("peak_position")
    if "same-window" not in correlation_types or not peak_type_specs:
        return []

    area_matrices = {
        item.group: read_lower_triangle_matrix(item.matrix).full for item in peak_maps
    }
    position_matrices = {
        group: read_lower_triangle_matrix(position_by_group[group].matrix).full
        for group in area_matrices
    }
    acf_matrices = {
        item.index: read_lower_triangle_matrix(item.matrix).full for item in window_maps
    }
    pair_i, pair_j = np.tril_indices(len(area_features.labels), k=-1)
    rows: list[dict[str, object]] = []
    for window in window_maps:
        acf_matrix = acf_matrices[window.index]
        for peak in peak_maps:
            if not (window.start <= peak.center < window.end):
                continue
            column = peak.group - 1
            present = presence_features.values[:, column] > 0
            positive_area = present & (area_features.values[:, column] > ZERO_AREA_EPS)
            finite_position = present & np.isfinite(position_features.values[:, column])
            for peak_type in peak_type_specs:
                if peak_type == "peak_area":
                    peak_matrix = area_matrices[peak.group]
                    valid_frames = positive_area
                    peak_source = peak.matrix
                    peak_waterfall = (
                        output_root
                        / "01_peak_area_waterfalls"
                        / f"peak_{peak.group:04d}_area_waterfall.png"
                    )
                    excluded_note = "both frames present and ROI area > 0; legacy 0/0 highs excluded"
                else:
                    position_map = position_by_group[peak.group]
                    peak_matrix = position_matrices[peak.group]
                    valid_frames = finite_position
                    peak_source = position_map.matrix
                    peak_waterfall = (
                        output_root
                        / "02_peak_position_waterfalls"
                        / f"peak_{peak.group:04d}_position_waterfall.png"
                    )
                    excluded_note = "both frames have a finite detected center"
                keep = (
                    valid_frames[pair_i]
                    & valid_frames[pair_j]
                    & np.isfinite(peak_matrix[pair_i, pair_j])
                    & np.isfinite(acf_matrix[pair_i, pair_j])
                )
                peak_scores = peak_matrix[pair_i[keep], pair_j[keep]]
                acf_scores = acf_matrix[pair_i[keep], pair_j[keep]]
                support = int(len(peak_scores))
                pearson = np.nan
                rho = np.nan
                p_value = np.nan
                if (
                    support >= 5
                    and float(np.nanstd(peak_scores)) > 1e-12
                    and float(np.nanstd(acf_scores)) > 1e-12
                ):
                    pearson = float(np.corrcoef(peak_scores, acf_scores)[0, 1])
                    spearman = spearmanr(peak_scores, acf_scores, nan_policy="omit")
                    rho = float(spearman.statistic)
                    p_value = float(spearman.pvalue)
                rows.append(
                    {
                        "window_id": f"window_{window.start:.1f}_{window.end:.1f}",
                        "window_start_deg": window.start,
                        "window_end_deg": window.end,
                        "peak_correlation_type": peak_type,
                        "peak_group": peak.group,
                        "peak_center_deg": peak.center,
                        "common_frame_pair_support": support,
                        "spearman_rho": rho,
                        "spearman_abs_rho": abs(rho) if np.isfinite(rho) else np.nan,
                        "spearman_p_value_descriptive": p_value,
                        "pearson_r": pearson,
                        "mean_peak_pair_score": float(np.mean(peak_scores))
                        if support
                        else np.nan,
                        "mean_acf_pair_r": float(np.mean(acf_scores)) if support else np.nan,
                        "peak_validity_rule": excluded_note,
                        "association_note": "descriptive association of pairwise score vectors; frame pairs are not independent and this is not causal attribution",
                        "peak_source_matrix": str(peak_source),
                        "acf_source_matrix": str(window.matrix),
                        "peak_waterfall": relative_to_output(peak_waterfall, output_root),
                        "acf_waterfall": relative_to_output(
                            output_root
                            / "03_same_window_acf_waterfalls"
                            / f"window_{window.start:.0f}_{window.end:.0f}deg_acf_waterfall.png",
                            output_root,
                        ),
                    }
                )
    rows.sort(
        key=lambda row: (
            float(row["window_start_deg"]),
            str(row["peak_correlation_type"]),
            -float(row["spearman_abs_rho"])
            if isinstance(row["spearman_abs_rho"], (float, np.floating))
            and np.isfinite(float(row["spearman_abs_rho"]))
            else math.inf,
            -int(row["common_frame_pair_support"]),
        )
    )
    return rows


def prepare_acf(
    patterns: Sequence[cip.Pattern],
    width: float,
    step: float,
    grid_step: float,
    shift_tolerance: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], np.ndarray | None],
    dict[tuple[int, int], np.ndarray | None],
]:
    args = SimpleNamespace(
        window_width=width,
        window_step=step,
        grid_step=grid_step,
        shift_tolerance=shift_tolerance,
        baseline_window=101,
        smooth_window=9,
        min_window_std=1e-6,
    )
    grid = wacf.common_grid(list(patterns), grid_step)
    starts = wacf.window_starts(grid, width, step, None)
    _, signals, fingerprints = wacf.build_window_signals_and_fingerprints(
        list(patterns), grid, starts, args
    )
    return grid, starts, signals, fingerprints


def write_readme(
    path: Path,
    args: argparse.Namespace,
    patterns: Sequence[cip.Pattern],
    counts: dict[str, int],
    roi_half_width: float,
    position_tolerance: float,
) -> None:
    text = f"""# Correlation-specific XRD waterfalls

Generated {datetime.now(timezone.utc).isoformat()} by
`generate_correlation_waterfalls.py` v{SCRIPT_VERSION}.

## What is here

| Correlation family | Waterfall feature | Count |
|---|---|---:|
| Peak ROI area across frames | Full-pattern-normalized local intensity, sideband baseline, and filled integration ROI | {counts.get('peak_area', 0)} |
| Peak position across frames | Locally normalized peak profile, selected center, and center trajectory | {counts.get('peak_position', 0)} |
| Same-window ACF across frames | Standardized positive-lag ACF; pair-specific winning neighboring window is shown | {counts.get('same_window_acf', 0)} |
| Window-to-window ACF within frame | Standardized positive-lag ACF for every 5° window | {counts.get('within_frame_window_acf', 0)} |

Source suite: `{args.suite_dir.resolve()}`  
Frames: {len(patterns)}  
ROI half-width: {roi_half_width:g}°  
Position tolerance: {position_tolerance:g}°

`waterfall_index.csv` links every source heatmap/matrix to its waterfall.
`trace_similarity.csv` contains the reference-to-trace score and plotted feature metadata.
`map_summary.csv` reports the strongest/weakest frame pair per map.
`frame_pair_summary.csv` ranks frame pairs across peak groups or ACF windows.
`window_peak_membership.csv` links every peak center to every nominal 5° ACF window containing it.
`window_peak_score_associations.csv` ranks the descriptive Spearman/Pearson association
between each in-window peak map and its ACF map over common frame pairs.

## Reading the colors

All plotted color bars are fixed to 0–1 (blue → light gray → red).
Peak-area and peak-position scores already live on 0–1. ACF matrices store Pearson
`r` on -1–1; each label preserves raw `r`, while the color uses
`s = (r + 1) / 2`. Missing/undefined comparisons are gray.

## Scientific caveats

1. This is the legacy XY suite. “Area” is a background-subtracted ROI integral after
   per-pattern P5/P99 normalization; it is not a fitted component area.
2. Position markers are detector-selected sample positions, not nonlinear fitted centers.
3. A detected (`present`) peak can still have ROI area zero. The legacy 0/0 formula assigns
   those pairs a score of 1; the figures and tables flag such degenerate highs.
4. Many peak groups are closer than the {2 * roi_half_width:g}° ROI diameter, so neighboring
   groups can produce overlapping, near-duplicate waterfalls.
5. Same-window ACF maps use a directional lower-triangle search over the other frame's
   nominal ±{args.acf_shift_tolerance:g}° neighboring windows. The colored curve and gray
   dashed partner reproduce the actual stored pair, including its winning shift.
6. Sliding ACF windows overlap strongly. `window_peak_membership.csv` is a spatial membership
   table, not proof that a single peak caused a window-level ACF score.
7. `window_peak_score_associations.csv` is stronger than spatial membership, but it is still
   descriptive: pairwise frame scores are not independent, and association is not causal
   attribution. A causal claim would require peak/ROI ablation or a generative model.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.suite_dir = args.suite_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.resume and not args.summary_only:
        raise SystemExit(
            "--resume is disabled because plot metadata and PNGs must remain atomic; "
            "use a new output directory."
        )
    if args.out_dir.exists() and not args.resume and not args.summary_only:
        raise SystemExit(f"Output directory already exists; use a new path or --resume: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    peak_root = args.suite_dir / "01_per_peak_frame_correlation"
    same_window_root = args.suite_dir / "02_same_window_acf_across_frames"
    within_frame_root = args.suite_dir / "03_single_frame_window_acf"
    roi_half_width = (
        float(args.roi_half_width)
        if args.roi_half_width is not None
        else read_key_value(peak_root / "README.txt", "ROI half-width", 0.06)
    )
    position_tolerance = read_key_value(peak_root / "README.txt", "Position tolerance", 0.06)

    area_features = read_feature_table(peak_root / "peak_roi_area_features.csv")
    presence_features = read_feature_table(peak_root / "peak_presence_features.csv")
    position_features = read_feature_table(peak_root / "peak_position_features.csv")
    if not (
        area_features.labels == presence_features.labels == position_features.labels
        and np.allclose(area_features.columns, presence_features.columns)
        and np.allclose(area_features.columns, position_features.columns)
    ):
        raise ValueError("Peak feature tables are not aligned.")
    patterns = load_patterns(args.inputs, area_features.labels)
    labels = [pattern.label for pattern in patterns]
    normalized_patterns = [cip.normalize_for_pattern(pattern.intensity) for pattern in patterns]
    fixed_reference = resolve_frame_reference(args.reference_frame, labels)
    selected_frames = resolve_frame_selection(args.frames, labels)

    area_maps_all = read_peak_maps(peak_root, position=False)
    position_maps_all = read_peak_maps(peak_root, position=True)
    area_by_group = {item.group: item for item in area_maps_all}
    position_by_group = {item.group: item for item in position_maps_all}
    if set(area_by_group) != set(position_by_group):
        raise ValueError("Area and position peak-map indices do not contain the same groups.")
    selected_groups = parse_id_selection(args.peak_groups)
    peak_groups = [item.group for item in area_maps_all if selected_groups is None or item.group in selected_groups]
    if args.max_peak_plots is not None:
        peak_groups = peak_groups[: max(0, args.max_peak_plots)]
    missing_selected = (selected_groups or set()) - set(area_by_group)
    if missing_selected:
        raise ValueError(f"Unknown peak groups: {sorted(missing_selected)}")
    nearest_gaps = peak_neighbor_gaps(area_maps_all)

    ledger = OutputLedger(waterfall_index=[], trace_similarity=[], map_summary=[])
    pair_matrices: dict[
        str,
        list[tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]],
    ] = defaultdict(list)
    plot_counts: dict[str, int] = defaultdict(int)
    progress_total = (
        len(peak_groups) * int("area" in args.types)
        + len(peak_groups) * int("position" in args.types)
    )
    completed = 0

    if "area" in args.types and not args.summary_only:
        output_dir = args.out_dir / "01_peak_area_waterfalls"
        for group in peak_groups:
            item = area_by_group[group]
            matrix = read_lower_triangle_matrix(item.matrix)
            if matrix.labels != labels:
                raise ValueError(f"Frame order differs in {item.matrix}")
            column = group - 1
            candidates = (presence_features.values[:, column] > 0) & (
                area_features.values[:, column] > ZERO_AREA_EPS
            )
            reference, strategy = choose_reference(matrix.full, candidates, fixed_reference)
            output_path = output_dir / f"peak_{group:04d}_area_waterfall.png"
            if not (args.resume and output_path.exists()):
                plot_area_waterfall(
                    item,
                    matrix,
                    patterns,
                    normalized_patterns,
                    area_features.values[:, column],
                    presence_features.values[:, column],
                    reference,
                    strategy,
                    output_path,
                    roi_half_width,
                    args.roi_sideband_gap,
                    args.roi_sideband_width,
                    nearest_gaps[group],
                    args.dpi,
                    ledger,
                    args.out_dir,
                )
            pair_matrices["peak_area"].append(
                (
                    f"peak_{group:04d}",
                    matrix.full,
                    candidates.copy(),
                    presence_features.values[:, column] > 0,
                )
            )
            plot_counts["peak_area"] += 1
            completed += 1
            if completed % 25 == 0 or completed == progress_total:
                print(f"peak waterfalls: {completed}/{progress_total}", flush=True)

    if "position" in args.types and not args.summary_only:
        output_dir = args.out_dir / "02_peak_position_waterfalls"
        for group in peak_groups:
            item = position_by_group[group]
            matrix = read_lower_triangle_matrix(item.matrix)
            if matrix.labels != labels:
                raise ValueError(f"Frame order differs in {item.matrix}")
            column = group - 1
            candidates = (presence_features.values[:, column] > 0) & np.isfinite(
                position_features.values[:, column]
            )
            reference, strategy = choose_reference(matrix.full, candidates, fixed_reference)
            output_path = output_dir / f"peak_{group:04d}_position_waterfall.png"
            if not (args.resume and output_path.exists()):
                plot_position_waterfall(
                    item,
                    matrix,
                    patterns,
                    position_features.values[:, column],
                    presence_features.values[:, column],
                    reference,
                    strategy,
                    output_path,
                    args.position_half_width,
                    position_tolerance,
                    nearest_gaps[group],
                    args.dpi,
                    ledger,
                    args.out_dir,
                )
            pair_matrices["peak_position"].append(
                (f"peak_{group:04d}", matrix.full, candidates.copy(), candidates.copy())
            )
            plot_counts["peak_position"] += 1
            completed += 1
            if completed % 25 == 0 or completed == progress_total:
                print(f"peak waterfalls: {completed}/{progress_total}", flush=True)

    need_acf = ("same-window" in args.types or "within-frame" in args.types) and not args.summary_only
    starts = np.array([], dtype=float)
    fingerprints: dict[tuple[int, int], np.ndarray | None] = {}
    if need_acf:
        _, starts, _, fingerprints = prepare_acf(
            patterns,
            args.acf_window_width,
            args.acf_window_step,
            args.acf_grid_step,
            args.acf_shift_tolerance,
        )

    same_window_maps_all = read_same_window_maps(same_window_root, args.acf_window_width)
    stored_starts = np.array([item.start for item in same_window_maps_all], dtype=float)
    if need_acf and not np.allclose(starts, stored_starts):
        raise ValueError(f"Recomputed ACF starts {starts} differ from stored starts {stored_starts}")
    selected_window_starts = parse_float_selection(args.window_starts)
    selected_window_maps = [
        item
        for item in same_window_maps_all
        if selected_window_starts is None
        or any(math.isclose(item.start, wanted, abs_tol=1e-6) for wanted in selected_window_starts)
    ]
    if selected_window_starts is not None:
        found = {item.start for item in selected_window_maps}
        missing = [wanted for wanted in selected_window_starts if not any(math.isclose(wanted, value) for value in found)]
        if missing:
            raise ValueError(f"Unknown same-window starts: {missing}")

    if args.summary_only:
        collected = collect_frame_pair_matrices(
            args.types,
            peak_groups,
            area_by_group,
            position_by_group,
            area_features,
            presence_features,
            position_features,
            selected_window_maps,
        )
        frame_pair_rows: list[dict[str, object]] = []
        for correlation_type, matrices in collected.items():
            frame_pair_rows.extend(frame_pair_summary(correlation_type, matrices, patterns))
        write_dict_rows(args.out_dir / "tables" / "frame_pair_summary.csv", frame_pair_rows)
        selected_peak_maps = [area_by_group[group] for group in peak_groups]
        membership_rows = build_window_peak_membership(
            selected_peak_maps,
            selected_window_maps,
            args.out_dir,
            include_area="area" in args.types,
            include_position="position" in args.types,
            include_same_window="same-window" in args.types,
        )
        write_dict_rows(args.out_dir / "tables" / "window_peak_membership.csv", membership_rows)
        association_rows = build_window_peak_score_associations(
            selected_peak_maps,
            position_by_group,
            selected_window_maps,
            area_features,
            presence_features,
            position_features,
            args.out_dir,
            args.types,
        )
        write_dict_rows(
            args.out_dir / "tables" / "window_peak_score_associations.csv",
            association_rows,
        )
        print(
            f"Refreshed {len(frame_pair_rows)} frame-pair rows and {len(membership_rows)} "
            f"window-peak rows plus {len(association_rows)} score-association rows in {args.out_dir}",
            flush=True,
        )
        return

    if "same-window" in args.types:
        output_dir = args.out_dir / "03_same_window_acf_waterfalls"
        for item in selected_window_maps:
            matrix = read_lower_triangle_matrix(item.matrix)
            if matrix.labels != labels:
                raise ValueError(f"Frame order differs in {item.matrix}")
            valid = np.array(
                [finite_curve(fingerprints[(index, item.index)]) for index in range(len(patterns))],
                dtype=bool,
            )
            reference, strategy = choose_reference(matrix.full, valid, fixed_reference)
            output_path = output_dir / f"window_{item.start:.0f}_{item.end:.0f}deg_acf_waterfall.png"
            if not (args.resume and output_path.exists()):
                plot_same_window_waterfall(
                    item,
                    matrix,
                    patterns,
                    starts,
                    fingerprints,
                    reference,
                    strategy,
                    output_path,
                    args.acf_grid_step,
                    args.acf_window_step,
                    args.acf_shift_tolerance,
                    args.dpi,
                    ledger,
                    args.out_dir,
                )
            pair_matrices["same_window_acf"].append(
                (f"window_{item.start:.1f}_{item.end:.1f}", matrix.full, None, None)
            )
            plot_counts["same_window_acf"] += 1
            print(
                f"same-window ACF waterfalls: {plot_counts['same_window_acf']}/{len(selected_window_maps)}",
                flush=True,
            )

    if "within-frame" in args.types:
        output_dir = args.out_dir / "04_within_frame_window_acf_waterfalls"
        for frame_index, pattern in enumerate(patterns):
            if selected_frames is not None and frame_index not in selected_frames:
                continue
            stem = safe_name(pattern.label)
            source_matrix = within_frame_root / "matrices" / f"{stem}_single_frame_window_acf.csv"
            source_heatmap = within_frame_root / "heatmaps" / f"{stem}_single_frame_window_acf.png"
            matrix = read_lower_triangle_matrix(source_matrix)
            expected_window_labels = [
                f"{start:.1f}-{start + args.acf_window_width:.1f}" for start in starts
            ]
            if matrix.labels != expected_window_labels:
                raise ValueError(f"Window order differs in {source_matrix}")
            valid = np.array(
                [finite_curve(fingerprints[(frame_index, index)]) for index in range(len(starts))],
                dtype=bool,
            )
            if args.reference_window_start is not None:
                matches = np.flatnonzero(np.isclose(starts, args.reference_window_start))
                if not len(matches):
                    raise ValueError(f"Unknown reference window start: {args.reference_window_start}")
                reference_window, strategy = int(matches[0]), "user"
            else:
                reference_window, strategy = choose_reference(matrix.full, valid, None)
            output_path = (
                output_dir
                / f"frame_{frame_index:04d}_{safe_name(pattern.label)}_window_acf_waterfall.png"
            )
            if not (args.resume and output_path.exists()):
                plot_within_frame_waterfall(
                    frame_index,
                    matrix,
                    patterns,
                    starts,
                    fingerprints,
                    reference_window,
                    strategy,
                    source_heatmap,
                    source_matrix,
                    output_path,
                    args.acf_grid_step,
                    args.acf_window_width,
                    args.dpi,
                    ledger,
                    args.out_dir,
                )
            plot_counts["within_frame_window_acf"] += 1
            print(
                f"within-frame ACF waterfalls: {plot_counts['within_frame_window_acf']}",
                flush=True,
            )

    frame_pair_rows: list[dict[str, object]] = []
    for correlation_type, matrices in pair_matrices.items():
        frame_pair_rows.extend(frame_pair_summary(correlation_type, matrices, patterns))

    tables_dir = args.out_dir / "tables"
    write_dict_rows(args.out_dir / "waterfall_index.csv", ledger.waterfall_index)
    write_dict_rows(tables_dir / "trace_similarity.csv", ledger.trace_similarity)
    write_dict_rows(tables_dir / "map_summary.csv", ledger.map_summary)
    write_dict_rows(tables_dir / "frame_pair_summary.csv", frame_pair_rows)
    selected_peak_maps = [area_by_group[group] for group in peak_groups]
    membership_rows = build_window_peak_membership(
        selected_peak_maps,
        selected_window_maps,
        args.out_dir,
        include_area="area" in args.types,
        include_position="position" in args.types,
        include_same_window="same-window" in args.types,
    )
    write_dict_rows(tables_dir / "window_peak_membership.csv", membership_rows)
    association_rows = build_window_peak_score_associations(
        selected_peak_maps,
        position_by_group,
        selected_window_maps,
        area_features,
        presence_features,
        position_features,
        args.out_dir,
        args.types,
    )
    write_dict_rows(tables_dir / "window_peak_score_associations.csv", association_rows)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_suite": str(args.suite_dir),
        "output_root": str(args.out_dir),
        "inputs": [str(path.resolve()) for path in args.inputs],
        "correlation_types": list(args.types),
        "frames": labels,
        "frame_pressures_GPa": [pattern.pressure_gpa for pattern in patterns],
        "selected_peak_groups": peak_groups,
        "selected_window_starts_deg": [item.start for item in selected_window_maps],
        "roi_half_width_deg": roi_half_width,
        "roi_sideband_gap_deg": args.roi_sideband_gap,
        "roi_sideband_width_deg": args.roi_sideband_width,
        "position_tolerance_deg": position_tolerance,
        "acf_window_width_deg": args.acf_window_width,
        "acf_window_step_deg": args.acf_window_step,
        "acf_grid_step_deg": args.acf_grid_step,
        "acf_shift_tolerance_deg": args.acf_shift_tolerance,
        "reference_frame": args.reference_frame or "per-map medoid",
        "reference_window_start_deg": args.reference_window_start
        if args.reference_window_start is not None
        else "per-frame medoid",
        "plot_counts": dict(plot_counts),
        "waterfall_index_rows": len(ledger.waterfall_index),
        "trace_similarity_rows": len(ledger.trace_similarity),
        "map_summary_rows": len(ledger.map_summary),
        "frame_pair_summary_rows": len(frame_pair_rows),
        "window_peak_membership_rows": len(membership_rows),
        "window_peak_score_association_rows": len(association_rows),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_readme(
        args.out_dir / "README.md",
        args,
        patterns,
        dict(plot_counts),
        roi_half_width,
        position_tolerance,
    )
    print(
        f"Wrote {sum(plot_counts.values())} waterfall plots and {len(ledger.waterfall_index)} index rows "
        f"to {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
