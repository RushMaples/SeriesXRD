#!/usr/bin/env python3
"""Create observational XRD peak-evolution waterfalls for four map families.

Only measured I(2theta) is drawn. Vertical offsets are display-only; no map
score, reference trace, or physical explanation is encoded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = ROOT / "outputs/correlation_suite_20260621_high_recall_scored_v2"
DEFAULT_OUT = ROOT / "correlations/results/correlation_peak_evolution_waterfalls_20260722"
DEFAULT_INPUTS = [ROOT / "Data/Cell_14_integrated", ROOT / "Data/Cell_29_integrated"]
KINDS = ("area", "position", "same-window", "within-frame")

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#3568a8", "#b5543a", "#4f8a5b", "#7656a3")
TEXT = "#303030"
MUTED = "#666666"


@dataclass(frozen=True)
class Pattern:
    label: str
    pressure: float | None
    series: str
    branch: str
    path: Path
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class PeakMap:
    group: int
    center: float
    area_heatmap: Path
    area_matrix: Path
    position_heatmap: Path
    position_matrix: Path


@dataclass(frozen=True)
class WindowMap:
    index: int
    start: float
    end: float
    heatmap: Path
    matrix: Path


@dataclass(frozen=True)
class Slice:
    x: np.ndarray
    y: np.ndarray
    coverage: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    p.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--types", nargs="+", choices=KINDS, default=list(KINDS))
    p.add_argument("--peak-groups", default=None)
    p.add_argument("--window-starts", default=None)
    p.add_argument("--peak-half-width", type=float, default=0.18)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--max-peak-plots", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--max-window-plots", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def parse_ids(text: str | None) -> set[int] | None:
    if text is None:
        return None
    result: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if "-" in token:
            left, right = map(int, token.split("-", 1))
            result.update(range(min(left, right), max(left, right) + 1))
        elif token:
            result.add(int(token))
    return result


def parse_floats(text: str | None) -> list[float] | None:
    return None if text is None else [float(v.strip()) for v in text.split(",") if v.strip()]


def frame_code(index: int) -> str:
    return f"F{index:02d}"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def angle_token(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def source_label(path: Path) -> tuple[str, float | None, str]:
    decomp = re.match(r"^decomp-([-+]?\d+(?:p\d+)?)GPa$", path.stem, re.I)
    if decomp:
        pressure = float(decomp.group(1).replace("p", "."))
        return f"{pressure:g}GPa_decomp", pressure, "decompression"
    match = re.match(r"^([-+]?\d+(?:p\d+)?)GPa(.*)$", path.stem, re.I)
    if not match:
        return path.stem, None, "unknown"
    pressure = float(match.group(1).replace("p", "."))
    suffix = match.group(2)
    branch = "decompression" if "decomp" in suffix.lower() else "compression"
    return f"{pressure:g}GPa{suffix}", pressure, branch


def discover(inputs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_file() and item.suffix.lower() == ".xy":
            files.append(item)
        elif item.is_dir():
            files.extend(item.rglob("*.xy"))
        else:
            raise FileNotFoundError(item)
    return sorted({path.resolve() for path in files})


def read_labels(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        return [row[0] for row in reader]


def load_patterns(inputs: Sequence[Path], labels: Sequence[str]) -> list[Pattern]:
    found: dict[str, Pattern] = {}
    for path in discover(inputs):
        label, pressure, branch = source_label(path)
        data = np.loadtxt(
            path,
            comments="#",
            usecols=(0, 1),
            dtype=float,
            encoding="latin-1",
        )
        if data.ndim == 1:
            data = data.reshape(1, 2)
        finite = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
        data = data[finite]
        order = np.argsort(data[:, 0], kind="stable")
        data = data[order]
        series = path.parent.name.replace("_integrated", "").replace("Cell_", "Cell ")
        if label in found:
            raise ValueError(f"Duplicate frame label: {label}")
        found[label] = Pattern(label, pressure, series, branch, path, data[:, 0], data[:, 1])
    missing = [label for label in labels if label not in found]
    extras = sorted(set(found) - set(labels))
    if missing or extras:
        raise ValueError(f"XY/suite mismatch; missing={missing}, extras={extras}")
    return [found[label] for label in labels]


def read_peak_maps(root: Path) -> list[PeakMap]:
    def rows(name: str) -> dict[int, dict[str, str]]:
        with (root / name).open(newline="", encoding="utf-8") as handle:
            return {int(row["peak_group"]): row for row in csv.DictReader(handle)}

    area = rows("per_peak_map_index.csv")
    position = rows("per_peak_position_map_index.csv")
    if set(area) != set(position):
        raise ValueError("Area and position indices differ")
    result: list[PeakMap] = []
    for group, a in area.items():
        p = position[group]
        result.append(
            PeakMap(
                group,
                float(a["group_two_theta"]),
                (root / a["heatmap"]).resolve(),
                (root / a["matrix"]).resolve(),
                (root / p["heatmap"]).resolve(),
                (root / p["matrix"]).resolve(),
            )
        )
    return result


def read_windows(root: Path) -> list[WindowMap]:
    result: list[WindowMap] = []
    with (root / "same_window_summary.csv").open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            start, end = float(row["start_deg"]), float(row["end_deg"])
            stem = f"window_{start:.1f}_{end:.1f}"
            result.append(
                WindowMap(
                    index,
                    start,
                    end,
                    (root / "heatmaps" / f"{stem}_same_window_acf.png").resolve(),
                    (root / "matrices" / f"{stem}_same_window_acf.csv").resolve(),
                )
            )
    return result


def raw_slice(pattern: Pattern, start: float, end: float) -> Slice:
    keep = (pattern.x >= start) & (pattern.x <= end)
    x, y = pattern.x[keep], pattern.y[keep]
    if not len(x) or pattern.x[-1] < start or pattern.x[0] > end:
        coverage = "not_measured"
    elif pattern.x[0] <= start and pattern.x[-1] >= end:
        coverage = "full"
    else:
        coverage = "partial"
    return Slice(x, y, coverage)


def shared_scale(curves: Iterable[np.ndarray]) -> tuple[float, float]:
    arrays = [curve[np.isfinite(curve)] for curve in curves if len(curve)]
    arrays = [curve for curve in arrays if len(curve)]
    if not arrays:
        return 0.0, 0.0
    values = np.concatenate(arrays)
    minimum = float(values.min())
    span = float(values.max() - minimum)
    return minimum, 0.72 / span if span > 0 else 0.0


def series_groups(patterns: Sequence[Pattern]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, pattern in enumerate(patterns):
        groups.setdefault(pattern.series, []).append(index)

    def series_key(name: str) -> tuple[int, str]:
        match = re.search(r"\d+", name)
        return (int(match.group()) if match else 10**9, name)

    ordered: dict[str, list[int]] = {}
    for name in sorted(groups, key=series_key):
        ordered[name] = sorted(
            groups[name],
            key=lambda index: (
                patterns[index].branch != "compression",
                math.inf if patterns[index].pressure is None else patterns[index].pressure,
                index,
            ),
        )
    return ordered


def series_color(name: str) -> str:
    preferred = {"Cell 14": COLORS[0], "Cell 29": COLORS[1]}
    if name in preferred:
        return preferred[name]
    match = re.search(r"\d+", name)
    index = int(match.group()) if match else 0
    return COLORS[index % len(COLORS)]


def row_layout(patterns: Sequence[Pattern]) -> tuple[dict[int, float], dict[str, list[int]]]:
    groups = series_groups(patterns)
    positions: dict[int, float] = {}
    cursor = 0.0
    for group_number, indices in enumerate(groups.values()):
        if group_number:
            cursor += 0.70
        for index in indices:
            positions[index] = cursor
            cursor += 1.0
    return positions, groups


def right_label(ax: plt.Axes, y: float, text: str, color: str = TEXT) -> None:
    ax.text(
        1.02,
        y,
        text,
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8,
        color=color,
        clip_on=False,
    )


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        fig.savefig(temporary, dpi=dpi, format="png", facecolor="white")
        temporary.replace(path)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()


def plot_frame_stack(
    patterns: Sequence[Pattern],
    start: float,
    end: float,
    title: str,
    subtitle: str,
    output: Path,
    dpi: int,
    center: float | None = None,
    frame_ids: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    if frame_ids is None:
        frame_ids = list(range(len(patterns)))
    if len(frame_ids) != len(patterns):
        raise ValueError("frame_ids must align with patterns")
    slices = [raw_slice(pattern, start, end) for pattern in patterns]
    positions, groups = row_layout(patterns)
    height = max(7.0, 0.47 * len(patterns) + 3.0)
    fig, ax = plt.subplots(figsize=(14.5, height))
    fig.subplots_adjust(left=0.11, right=0.72, top=0.87, bottom=0.13)
    fig.suptitle(title, x=0.11, y=0.965, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.11, 0.925, subtitle, ha="left", va="top", fontsize=9.2, color=MUTED)
    rows: list[dict[str, object]] = []

    for name, indices in groups.items():
        color = series_color(name)
        minimum, gain = shared_scale(slices[index].y for index in indices)
        group_y = [positions[index] for index in indices]
        ax.text(
            -0.085,
            float(np.mean(group_y)),
            name,
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.5,
            fontweight="bold",
            color=color,
            clip_on=False,
        )
        for index in indices:
            pattern, excerpt = patterns[index], slices[index]
            display_index = int(frame_ids[index])
            base = positions[index]
            if len(excerpt.x):
                ax.plot(
                    excerpt.x,
                    base + (excerpt.y - minimum) * gain,
                    color=color,
                    linewidth=1.15,
                    zorder=3,
                )
            note = ""
            if excerpt.coverage == "partial":
                note = " · partial range"
            elif excerpt.coverage == "not_measured":
                note = " · not measured in this range"
            branch = " · decomp" if pattern.branch == "decompression" else ""
            pressure = "NA" if pattern.pressure is None else f"{pattern.pressure:g} GPa"
            right_label(
                ax,
                base + 0.08,
                f"{frame_code(display_index)} · {pressure}{branch}{note}",
                MUTED if excerpt.coverage == "not_measured" else TEXT,
            )
            rows.append(
                {
                    "frame_index": display_index,
                    "frame_code": frame_code(display_index),
                    "frame_label": pattern.label,
                    "series": pattern.series,
                    "pressure_GPa": pattern.pressure,
                    "branch": pattern.branch,
                    "source_xy": str(pattern.path.resolve()),
                    "requested_start_deg": start,
                    "requested_end_deg": end,
                    "coverage": excerpt.coverage,
                    "point_count": len(excerpt.x),
                    "measured_start_deg": float(excerpt.x[0]) if len(excerpt.x) else np.nan,
                    "measured_end_deg": float(excerpt.x[-1]) if len(excerpt.x) else np.nan,
                }
            )

    if center is not None:
        ax.axvline(center, color="#777777", linewidth=0.75, linestyle=":", zorder=1)
        ax.text(
            center,
            1.005,
            f"nominal {center:.4f}°",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
    ticks = sorted((value, frame_code(int(frame_ids[index]))) for index, value in positions.items())
    ax.set_xlim(start, end)
    ax.set_ylim(-0.4, max(positions.values()) + 1.02)
    ax.set_yticks([value for value, _ in ticks])
    ax.set_yticklabels([label for _, label in ticks], fontsize=8.3)
    ax.set_xlabel(r"$2\theta$ (degrees)")
    ax.set_ylabel("Frame / pressure order (vertical offset)")
    ax.grid(axis="x", color="#dedede", linewidth=0.55, alpha=0.85)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    fig.text(
        0.11,
        0.035,
        "Measured raw intensity (a.u.). One shared linear scale is used within each Cell; "
        "vertical offsets are display-only. Color identifies Cell only.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    save(fig, output, dpi)
    return rows


def plot_cell_full_pattern(
    patterns: Sequence[Pattern],
    indices: Sequence[int],
    output: Path,
    dpi: int,
    start: float = 3.0,
    end: float = 23.0,
) -> list[dict[str, object]]:
    subset = [patterns[index] for index in indices]
    title = (
        f"{subset[0].series} — measured {start:g}–{end:g}° intensity "
        "across pressure sequence"
    )
    subtitle = "Full-pattern companion for within-frame window maps · decompression follows compression"
    return plot_frame_stack(
        subset,
        start,
        end,
        title,
        subtitle,
        output,
        dpi,
        frame_ids=indices,
    )


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean: dict[str, object] = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
                    value = ""
                clean[field] = value
            writer.writerow(clean)


def write_readme(
    path: Path,
    suite: Path,
    artifact_counts: dict[str, int],
    mapping_count: int,
    frame_count: int,
    half_width: float,
) -> None:
    text = f"""# XRD peak-evolution waterfalls

Generated by `generate_peak_evolution_waterfalls.py` v{VERSION}.

These figures show only measured XRD intensity as vertically offset spectral
traces. They are intended for observing whether peaks appear, disappear, or
move across a frame/pressure sequence. They do not encode a map value, choose a
reference, or propose a physical cause.

| Companion map family | Unique raw-spectrum artifacts |
|---|---:|
| Peak area + peak position (shared local views) | {artifact_counts.get('peak_evolution', 0)} |
| Same 5° window across frames | {artifact_counts.get('same_window', 0)} |
| Within-frame windows (shared Cell-level 3–23° views) | {artifact_counts.get('full_pattern', 0)} |

Unique PNG files: {sum(artifact_counts.values())}  
Source-map mappings in `waterfall_index.csv`: {mapping_count}  
Frames: {frame_count}  
Peak-view half-width: {half_width:g}°  
Source suite: `{suite}`

## Reading rules

- x is measured 2θ and intensity is reported in arbitrary units because the XY
  headers provide no calibrated intensity unit.
- Every trace in one Cell uses the same linear intensity scale. Cell 14 and Cell
  29 are kept as separate row groups and have independent scales.
- Vertical offsets are display-only. Color identifies Cell only.
- Rows are ordered by pressure within each Cell. Cell 29 decompression follows
  the complete compression branch rather than being inserted by pressure value.
- A raw trace remains visible whether or not the peak detector assigned a peak.
- `not measured in this range` means the source detector range does not cover
  that 2θ interval; it is not a statement that a peak disappeared.
- There is no reliable timestamp column in these source XY files, so no time
  coordinate is invented.

Area and position maps point to the same local raw-spectrum PNG because the
observational peak evolution is the same measurement. The 17 within-frame maps
point to one of two Cell-level full-pattern waterfalls because a single-frame
matrix itself has no temporal dimension.

`artifact_index.csv` lists every unique PNG. `waterfall_index.csv` preserves the
source-map-to-artifact mapping. `tables/trace_index.csv` records every displayed
raw trace, source XY path, requested range, coverage, and point count.
"""
    path.write_text(text, encoding="utf-8")


def peak_mapping_row(kind: str, item: PeakMap, output: Path) -> dict[str, object]:
    if kind == "peak_area":
        heatmap, matrix = item.area_heatmap, item.area_matrix
    else:
        heatmap, matrix = item.position_heatmap, item.position_matrix
    return {
        "correlation_type": kind,
        "feature_id": f"peak_{item.group:04d}",
        "feature_label": f"peak group {item.group} at {item.center:.6f} deg",
        "source_heatmap": str(heatmap),
        "source_matrix": str(matrix),
        "waterfall": str(output.resolve()),
    }


def add_trace_metadata(
    destination: list[dict[str, object]],
    rows: Sequence[dict[str, object]],
    artifact_id: str,
    output: Path,
) -> None:
    for row in rows:
        destination.append(
            {
                "artifact_id": artifact_id,
                "waterfall": str(output.resolve()),
                **row,
            }
        )


def main() -> None:
    args = parse_args()
    args.suite_dir = args.suite_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    peak_root = args.suite_dir / "01_per_peak_frame_correlation"
    same_root = args.suite_dir / "02_same_window_acf_across_frames"
    within_root = args.suite_dir / "03_single_frame_window_acf"
    labels = read_labels(peak_root / "peak_roi_area_features.csv")
    patterns = load_patterns(args.inputs, labels)
    peak_maps = read_peak_maps(peak_root)
    all_windows = read_windows(same_root)

    selected_groups = parse_ids(args.peak_groups)
    peaks = [item for item in peak_maps if selected_groups is None or item.group in selected_groups]
    missing_groups = (selected_groups or set()) - {item.group for item in peak_maps}
    if missing_groups:
        raise ValueError(f"Unknown peak groups: {sorted(missing_groups)}")
    if args.max_peak_plots is not None:
        peaks = peaks[: max(0, args.max_peak_plots)]

    selected_starts = parse_floats(args.window_starts)
    windows = [
        item
        for item in all_windows
        if selected_starts is None
        or any(math.isclose(item.start, value, abs_tol=1e-9) for value in selected_starts)
    ]
    if selected_starts is not None:
        missing_starts = [
            value
            for value in selected_starts
            if not any(math.isclose(item.start, value, abs_tol=1e-9) for item in windows)
        ]
        if missing_starts:
            raise ValueError(f"Unknown window starts: {missing_starts}")
    if args.max_window_plots is not None:
        windows = windows[: max(0, args.max_window_plots)]

    artifacts: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    counts: dict[str, int] = {"peak_evolution": 0, "same_window": 0, "full_pattern": 0}

    if "area" in args.types or "position" in args.types:
        folder = args.out_dir / "01_peak_evolution_waterfalls"
        for completed, item in enumerate(peaks, start=1):
            artifact_id = f"peak_{item.group:04d}"
            output = folder / f"{artifact_id}_evolution_waterfall.png"
            rows = plot_frame_stack(
                patterns,
                item.center - args.peak_half_width,
                item.center + args.peak_half_width,
                f"Peak {item.group:04d} at {item.center:.4f}° — measured intensity across frames",
                "Rows grouped by Cell and ordered by pressure; decompression follows compression",
                output,
                args.dpi,
                center=item.center,
            )
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "artifact_type": "peak_evolution",
                    "peak_group": item.group,
                    "peak_center_deg": item.center,
                    "x_start_deg": item.center - args.peak_half_width,
                    "x_end_deg": item.center + args.peak_half_width,
                    "waterfall": str(output.resolve()),
                }
            )
            add_trace_metadata(traces, rows, artifact_id, output)
            if "area" in args.types:
                mappings.append(peak_mapping_row("peak_area", item, output))
            if "position" in args.types:
                mappings.append(peak_mapping_row("peak_position", item, output))
            counts["peak_evolution"] += 1
            if completed % 25 == 0 or completed == len(peaks):
                print(f"peak-evolution waterfalls: {completed}/{len(peaks)}", flush=True)

    if "same-window" in args.types:
        folder = args.out_dir / "02_same_window_waterfalls"
        for completed, item in enumerate(windows, start=1):
            token = f"{angle_token(item.start)}_{angle_token(item.end)}"
            artifact_id = f"window_{token}deg"
            output = folder / f"{artifact_id}_waterfall.png"
            rows = plot_frame_stack(
                patterns,
                item.start,
                item.end,
                f"{item.start:g}–{item.end:g}° — measured intensity across frames",
                "Raw 5° spectral window; rows grouped by Cell and ordered by pressure",
                output,
                args.dpi,
            )
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "artifact_type": "same_window_evolution",
                    "window_start_deg": item.start,
                    "window_end_deg": item.end,
                    "x_start_deg": item.start,
                    "x_end_deg": item.end,
                    "waterfall": str(output.resolve()),
                }
            )
            mappings.append(
                {
                    "correlation_type": "same_window_across_frames",
                    "feature_id": artifact_id,
                    "feature_label": f"window {item.start:.6f}-{item.end:.6f} deg",
                    "source_heatmap": str(item.heatmap),
                    "source_matrix": str(item.matrix),
                    "waterfall": str(output.resolve()),
                }
            )
            add_trace_metadata(traces, rows, artifact_id, output)
            counts["same_window"] += 1
            print(f"same-window waterfalls: {completed}/{len(windows)}", flush=True)

    if "within-frame" in args.types:
        folder = args.out_dir / "03_within_frame_full_pattern_waterfalls"
        groups = series_groups(patterns)
        within_start = min(item.start for item in all_windows)
        within_end = max(item.end for item in all_windows)
        series_outputs: dict[str, Path] = {}
        for completed, (series, indices) in enumerate(groups.items(), start=1):
            series_token = safe_name(series.replace(" ", "_"))
            artifact_id = f"full_pattern_{series_token}"
            output = folder / f"{artifact_id}_waterfall.png"
            rows = plot_cell_full_pattern(
                patterns,
                indices,
                output,
                args.dpi,
                start=within_start,
                end=within_end,
            )
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "artifact_type": "cell_full_pattern_evolution",
                    "series": series,
                    "x_start_deg": within_start,
                    "x_end_deg": within_end,
                    "waterfall": str(output.resolve()),
                }
            )
            add_trace_metadata(traces, rows, artifact_id, output)
            series_outputs[series] = output
            counts["full_pattern"] += 1
            print(f"full-pattern waterfalls: {completed}/{len(groups)}", flush=True)

        for index, pattern in enumerate(patterns):
            stem = safe_name(pattern.label)
            source_heatmap = (within_root / "heatmaps" / f"{stem}_single_frame_window_acf.png").resolve()
            source_matrix = (within_root / "matrices" / f"{stem}_single_frame_window_acf.csv").resolve()
            if not source_heatmap.exists() or not source_matrix.exists():
                raise FileNotFoundError(f"Missing within-frame map for {pattern.label}")
            mappings.append(
                {
                    "correlation_type": "within_frame_windows",
                    "feature_id": f"frame_{index:04d}",
                    "feature_label": f"{pattern.label} within-frame windows",
                    "frame_index": index,
                    "frame_code": frame_code(index),
                    "frame_label": pattern.label,
                    "series": pattern.series,
                    "pressure_GPa": pattern.pressure,
                    "branch": pattern.branch,
                    "source_heatmap": str(source_heatmap),
                    "source_matrix": str(source_matrix),
                    "waterfall": str(series_outputs[pattern.series].resolve()),
                }
            )

    write_rows(args.out_dir / "artifact_index.csv", artifacts)
    write_rows(args.out_dir / "waterfall_index.csv", mappings)
    write_rows(args.out_dir / "tables" / "trace_index.csv", traces)
    write_readme(
        args.out_dir / "README.md",
        args.suite_dir,
        counts,
        len(mappings),
        len(patterns),
        args.peak_half_width,
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_suite": str(args.suite_dir),
        "output_root": str(args.out_dir),
        "inputs": [str(path.resolve()) for path in args.inputs],
        "requested_types": list(args.types),
        "frame_labels": labels,
        "frame_series": [pattern.series for pattern in patterns],
        "frame_pressures_GPa": [pattern.pressure for pattern in patterns],
        "frame_branches": [pattern.branch for pattern in patterns],
        "selected_peak_groups": [item.group for item in peaks],
        "selected_window_starts_deg": [item.start for item in windows],
        "artifact_counts": counts,
        "unique_png_count": sum(counts.values()),
        "source_map_mapping_count": len(mappings),
        "trace_index_rows": len(traces),
        "encodes_correlation_values": False,
        "physical_interpretation_included": False,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Wrote {sum(counts.values())} unique PNGs and {len(mappings)} map links to {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
