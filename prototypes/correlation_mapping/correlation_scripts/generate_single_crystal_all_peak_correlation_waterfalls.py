#!/usr/bin/env python3
"""Map all-peak Log² ROI correlations onto original single-crystal XY traces.

The color comes from the all-peak Log² ROI-area matrix.  The waterfall height
comes from the original positive, spot-masked XY intensity divided by the TIFF
exposure; no nonlinear transform is applied to the displayed trace.  Every
target-frame peak is joined by ``(frame, local_peak_index)`` and is shown both
as a colored under-curve region and as a lossless support ribbon.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


WAVELENGTH_A = 0.4133
TRACE_HEIGHT = 0.66
ROW_SPACING = 1.0
RIBBON_HEIGHT = 0.025
RIBBON_GAP = 0.005


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--xy-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--anchor-peak-id",
        action="append",
        default=[],
        help="Generate only this peak ID; repeat for more. Default: all anchors.",
    )
    parser.add_argument("--dpi", type=int, default=175)
    parser.add_argument("--palette-colors", type=int, default=128)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def q_to_two_theta(q_a_inv: float) -> float:
    argument = float(q_a_inv) * WAVELENGTH_A / (4.0 * math.pi)
    if not math.isfinite(argument) or abs(argument) > 1.0:
        return math.nan
    return math.degrees(2.0 * math.asin(argument))


def peak_support(peak: Mapping[str, str]) -> tuple[float, float]:
    center = float(peak["q_A^-1"])
    halfwidth = float(peak["halfwidth_q_A^-1"])
    left = q_to_two_theta(max(center - halfwidth, 0.0))
    right = q_to_two_theta(center + halfwidth)
    if not (math.isfinite(left) and math.isfinite(right) and right > left):
        raise ValueError(f"invalid q support for {peak.get('peak_id')}")
    return left, right


def assign_interval_lanes(
    peaks: Sequence[Mapping[str, str]],
) -> dict[int, int]:
    """Greedily assign non-overlapping support intervals to ribbon lanes."""

    lane_ends: list[float] = []
    result: dict[int, int] = {}
    ordered = sorted(
        enumerate(peaks), key=lambda item: (peak_support(item[1])[0], item[0])
    )
    for original_index, peak in ordered:
        left, right = peak_support(peak)
        lane = next(
            (index for index, end in enumerate(lane_ends) if left > end + 1e-12),
            len(lane_ends),
        )
        if lane == len(lane_ends):
            lane_ends.append(right)
        else:
            lane_ends[lane] = right
        result[original_index] = lane
    return result


def load_area_matrix(path: Path) -> dict[tuple[int, int], float]:
    rows = read_rows(path)
    result: dict[tuple[int, int], float] = {}
    for row in rows:
        label = row["target_frame"].strip()
        if not label.startswith("frame "):
            raise ValueError(f"unexpected target-frame label in {path}: {label}")
        frame = int(label.split()[1])
        for field, raw in row.items():
            if not field.startswith("peak ") or not raw.strip():
                continue
            score = float(raw)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"score outside [0,1] in {path}: {score}")
            result[(frame, int(field.split()[1]))] = score
    return result


def load_original_traces(
    xy_root: Path,
    frame_layout: Sequence[Mapping[str, str]],
    exposure_by_frame: Mapping[int, float],
) -> dict[int, tuple[Path, np.ndarray, np.ndarray]]:
    prepared: dict[int, tuple[Path, np.ndarray, np.ndarray]] = {}
    positive_values: list[np.ndarray] = []
    for row in frame_layout:
        frame = int(row["frame"])
        path = xy_root / f"frame_{frame:04d}_masked.xy"
        data = np.loadtxt(path, comments="#", dtype=float)
        if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 3:
            raise ValueError(f"invalid original XY file: {path}")
        x = np.asarray(data[:, 0], dtype=float)
        y = np.clip(np.asarray(data[:, 1], dtype=float), 0.0, None)
        exposure = float(exposure_by_frame[frame])
        if not math.isfinite(exposure) or exposure <= 0.0:
            raise ValueError(f"invalid TIFF exposure for frame {frame}")
        rate = y / exposure
        prepared[frame] = (path, x, rate)
        positive_values.append(rate[rate > 0.0])
    pooled = np.concatenate([item for item in positive_values if item.size])
    shared_cap = float(np.quantile(pooled, 0.995))
    if not math.isfinite(shared_cap) or shared_cap <= 0.0:
        raise ValueError("original XY traces have no positive intensity")
    return {
        frame: (path, x, np.clip(rate / shared_cap, 0.0, 1.0))
        for frame, (path, x, rate) in prepared.items()
    }


def quantize_png(path: Path, colors: int) -> None:
    if colors == 0:
        return
    if colors < 16 or colors > 256:
        raise ValueError("--palette-colors must be 0 or 16..256")
    temporary = path.with_name(f".{path.stem}.quantized.png")
    with Image.open(path) as source:
        source.convert("RGB").quantize(
            colors=colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        ).save(temporary, optimize=True, compress_level=9)
    temporary.replace(path)


def plot_anchor(
    path: Path,
    anchor: Mapping[str, str],
    frame_layout: Sequence[Mapping[str, str]],
    peaks_by_frame: Mapping[int, Sequence[Mapping[str, str]]],
    traces: Mapping[int, tuple[Path, np.ndarray, np.ndarray]],
    scores: Mapping[tuple[int, int], float],
    *,
    dpi: int,
) -> dict[str, Any]:
    ordered_frames = sorted(
        frame_layout, key=lambda row: (float(row["pressure_GPa"]), int(row["frame"]))
    )
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(15.5, 10.0))
    finite_joined = 0
    expected_joined = 0
    max_lane = 0
    for row_index, frame_row in enumerate(ordered_frames):
        frame = int(frame_row["frame"])
        baseline = float(row_index) * ROW_SPACING
        source, x, displayed = traces[frame]
        y_plot = baseline + TRACE_HEIGHT * displayed
        ax.plot(x, y_plot, color="#20242a", lw=0.58, zorder=2)
        peaks = list(peaks_by_frame[frame])
        lanes = assign_interval_lanes(peaks)
        same_frame = frame == int(anchor["frame"])
        if not same_frame:
            expected_joined += len(peaks)
        for peak_index, peak in enumerate(peaks):
            local_index = int(peak["local_peak_index"])
            score = scores.get((frame, local_index), math.nan)
            if not same_frame and not math.isfinite(score):
                raise ValueError(
                    f"missing correlation for frame {frame}, peak {local_index}"
                )
            if same_frame:
                color = "#d7dce2"
            else:
                color = cmap(norm(score))
                finite_joined += 1
            left, right = peak_support(peak)
            mask = (x >= left) & (x <= right)
            if np.count_nonzero(mask) >= 2:
                ax.fill_between(
                    x[mask], baseline, y_plot[mask], color=color, alpha=0.88, lw=0,
                    zorder=3,
                )
            lane = lanes[peak_index]
            max_lane = max(max_lane, lane + 1)
            ribbon_y = baseline - 0.045 - lane * (RIBBON_HEIGHT + RIBBON_GAP)
            ax.add_patch(
                Rectangle(
                    (left, ribbon_y), right - left, RIBBON_HEIGHT,
                    facecolor=color, edgecolor="#30343a", linewidth=0.18, zorder=4,
                )
            )
        ax.text(
            float(x.min()) - 0.20,
            baseline + 0.14,
            f"{float(frame_row['pressure_GPa']):g} GPa\nf{frame:04d}",
            ha="right",
            va="center",
            fontsize=7,
            color="#30343a",
        )
        if same_frame:
            ax.axhspan(
                baseline - 0.13,
                baseline + TRACE_HEIGHT + 0.03,
                color="#adb5bd",
                alpha=0.10,
                zorder=0,
            )

    if finite_joined != expected_joined:
        raise RuntimeError(
            f"joined {finite_joined} correlation cells; expected {expected_joined}"
        )
    anchor_two_theta = float(anchor["two_theta_deg"])
    ax.axvline(anchor_two_theta, color="#e63946", lw=0.8, ls="--", alpha=0.8)
    ax.set_xlim(min(value[1].min() for value in traces.values()), max(value[1].max() for value in traces.values()))
    ax.set_ylim(-0.22 - max_lane * 0.03, len(ordered_frames) * ROW_SPACING - 0.10)
    ax.set_yticks([])
    ax.set_xlabel(r"$2\theta$ (degrees)")
    ax.set_title(
        "Single crystal — Log² all-peak ROI correlation on original XY waterfall\n"
        f"anchor {anchor['peak_id']} | frame {int(anchor['frame'])}, "
        f"local peak {int(anchor['local_peak_index'])} | "
        f"{float(anchor['pressure_GPa']):g} GPa | {anchor_two_theta:.4f}°",
        fontsize=12,
    )
    ax.grid(axis="x", color="#c9ced6", lw=0.35, alpha=0.5)
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.025, pad=0.015
    )
    colorbar.set_label("Log² ROI-area correlation (min/max)")
    fig.text(
        0.5,
        0.008,
        "Height: original positive spot-masked XY / TIFF exposure (shared display scale). "
        "Color: Log² correlation. Ribbons preserve every peak when projected supports overlap. "
        "Grey row: anchor frame excluded from cross-frame scoring.",
        ha="center",
        fontsize=7.5,
        color="#4b5158",
    )
    fig.tight_layout(rect=(0.02, 0.025, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return {
        "joined_cross_frame_peak_cells": finite_joined,
        "expected_cross_frame_peak_cells": expected_joined,
        "max_ribbon_lanes": max_lane,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analysis_root = args.analysis_root.expanduser().resolve()
    xy_root = args.xy_root.expanduser().resolve()
    output_root = args.out_dir.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")

    transform = json.loads(
        (analysis_root.parent.parent / "preprocessing" / "TRANSFORM_PROVENANCE.json").read_text(
            encoding="utf-8"
        )
    )
    transform_method = transform.get("method") or transform.get("transform", {}).get(
        "method"
    )
    if transform_method != "log_squared":
        raise ValueError("waterfall requires a log_squared correlation source")

    peaks = read_rows(analysis_root / "peak_registry.csv")
    anchors = read_rows(analysis_root / "per_anchor_peak_map_index.csv")
    frame_layout = read_rows(analysis_root / "frame_slot_layout.csv")
    exposures = {
        int(row["frame"]): float(row["exposure_s"])
        for row in read_rows(analysis_root.parent.parent / "preprocessing" / "frame_extraction_qc.csv")
    }
    if len(peaks) != 275 or len(anchors) != 275:
        raise ValueError(f"expected 275 peaks/anchors, found {len(peaks)}/{len(anchors)}")
    selected_ids = set(args.anchor_peak_id)
    if selected_ids:
        anchors = [row for row in anchors if row["anchor_peak_id"] in selected_ids]
        missing = selected_ids - {row["anchor_peak_id"] for row in anchors}
        if missing:
            raise ValueError(f"unknown anchor peak IDs: {sorted(missing)}")
    peaks_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for peak in peaks:
        peaks_by_frame[int(peak["frame"])].append(peak)
    for frame in peaks_by_frame:
        peaks_by_frame[frame].sort(key=lambda row: int(row["local_peak_index"]))
    traces = load_original_traces(xy_root, frame_layout, exposures)

    output_root.mkdir(parents=True)
    index_rows: list[dict[str, Any]] = []
    for anchor in anchors:
        matrix_path = analysis_root / anchor["area_csv"]
        scores = load_area_matrix(matrix_path)
        frame = int(anchor["anchor_frame"])
        slot = int(anchor["anchor_local_peak"])
        peak = next(
            item
            for item in peaks
            if int(item["frame"]) == frame and int(item["local_peak_index"]) == slot
        )
        stem = matrix_path.stem
        png_path = output_root / "heatmaps" / f"frame_{frame:04d}" / f"{stem}.png"
        metrics = plot_anchor(
            png_path,
            peak,
            frame_layout,
            peaks_by_frame,
            traces,
            scores,
            dpi=args.dpi,
        )
        quantize_png(png_path, args.palette_colors)
        index_rows.append(
            {
                "anchor_peak_id": peak["peak_id"],
                "anchor_frame": frame,
                "anchor_local_peak": slot,
                "anchor_pressure_GPa": peak["pressure_GPa"],
                "anchor_two_theta_deg": peak["two_theta_deg"],
                "source_area_matrix": str(matrix_path),
                "png": str(png_path.relative_to(output_root)),
                **metrics,
            }
        )
    write_rows(output_root / "WATERFALL_INDEX.csv", index_rows)
    source_xy = [traces[int(row["frame"])][0] for row in frame_layout]
    validation = {
        "status": "PASS",
        "transform": "log_squared",
        "anchor_maps": len(index_rows),
        "all_275_anchors_generated": len(index_rows) == 275,
        "original_xy_trace_files": len(source_xy),
        "track_used_for_selection_grouping_or_scoring": False,
        "height_domain": "original_positive_spot_masked_XY_per_TIFF_exposure",
        "color_domain": "log_squared_all_peak_ROI_area_min_max_similarity",
        "every_cross_frame_peak_joined": all(
            row["joined_cross_frame_peak_cells"]
            == row["expected_cross_frame_peak_cells"]
            for row in index_rows
        ),
        "source_xy_sha256": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_xy
        ],
    }
    write_json(output_root / "SUITE_VALIDATION.json", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
