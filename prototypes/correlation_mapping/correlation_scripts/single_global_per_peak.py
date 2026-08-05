#!/usr/bin/env python3
"""Global-track per-peak analysis for the single-crystal Masked export."""

from __future__ import annotations

import csv
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import run_uote_xy_handoff_correlations as legacy


POSITION_TOLERANCE_DEG = 0.06
HEATMAP_TRIANGLE_POLICY = "strict_lower_only_no_diagonal"


def strict_lower_triangle_layers(
    matrix: np.ndarray,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """Return data and missing-value layers for a strict-lower heatmap.

    The diagonal and upper triangle are structurally hidden. Missing values in
    the lower triangle remain distinguishable as a gray diagnostic layer.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {values.shape}")
    structural = np.triu(np.ones(values.shape, dtype=bool), k=0)
    finite_lower = (~structural) & np.isfinite(values)
    missing_lower = (~structural) & (~np.isfinite(values))
    data_layer = np.ma.array(values, mask=~finite_lower, copy=True)
    missing_layer = np.ma.array(
        np.ones(values.shape, dtype=float),
        mask=~missing_lower,
        copy=True,
    )
    return data_layer, missing_layer


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def robust_mad(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def orientation_base(scan: str) -> str:
    """Return orientation without the substring bug where 10deg contains 0deg."""
    if scan in {"orientation_10deg", "decompression_orientation_10deg"}:
        return "10deg"
    if scan == "orientation_0deg":
        return "0deg"
    return scan


def branch_label(metadata_row: dict[str, Any]) -> str:
    scan = str(metadata_row["orientation"])
    reason = str(metadata_row.get("exclusion_reason", ""))
    if scan == "decompression_orientation_10deg":
        return "10deg_decomp"
    if reason.startswith("duplicate_scan_pressure"):
        return "10deg_alt"
    return orientation_base(scan)


def collapse_frame_track_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median-collapse duplicate ROI observations within one global track/frame."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["track"]), int(row["frame"]))].append(row)
    out: list[dict[str, Any]] = []
    for (track, frame), items in sorted(grouped.items()):
        first = items[0]
        q_values = [float(item["q_A^-1"]) for item in items]
        d_values = [float(item["d_A"]) for item in items]
        tt_values = [float(item["two_theta_deg"]) for item in items]
        area_values = [float(item["normalized_intensity_counts_per_s_per_pixel"]) for item in items]
        finite_area = [value for value in area_values if np.isfinite(value)]
        out.append({
            "dataset": first["dataset"],
            "track": track,
            "frame": frame,
            "pressure_GPa": float(first["pressure_GPa"]),
            "orientation": orientation_base(str(first["orientation"])),
            "branch": str(first.get("branch", orientation_base(str(first["orientation"])))),
            "n_observations": len(items),
            "q_median_A^-1": float(np.median(q_values)),
            "q_mad_A^-1": robust_mad(q_values),
            "d_median_A": float(np.median(d_values)),
            "d_mad_A": robust_mad(d_values),
            "two_theta_median_deg": float(np.median(tt_values)),
            "two_theta_mad_deg": robust_mad(tt_values),
            "normalized_area_median_counts_per_s_per_pixel": float(np.median(finite_area)) if finite_area else np.nan,
            "normalized_area_mad_counts_per_s_per_pixel": robust_mad(finite_area),
            "duplicate_observation_flag": int(len(items) > 1),
            "matched_d_A_anchor_candidate": first.get("matched_d_A", np.nan),
            "area_status": first["intensity_status"],
        })
    return out


def symmetric_similarity_matrix(values: np.ndarray, location: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    matrix = np.full((len(values), len(values)), np.nan, dtype=float)
    for i, left in enumerate(values):
        if not np.isfinite(left):
            continue
        matrix[i, i] = 1.0
        for j in range(i):
            right = values[j]
            if not np.isfinite(right):
                continue
            if location:
                score = 1.0 - abs(float(left) - float(right)) / POSITION_TOLERANCE_DEG
            else:
                high = max(float(left), float(right))
                low = min(float(left), float(right))
                if high > 0 and low >= 0:
                    score = low / high
                elif high == 0 and low == 0:
                    score = 1.0
                else:
                    score = np.nan
            if np.isfinite(score):
                score = float(np.clip(score, 0.0, 1.0))
                matrix[i, j] = score
                matrix[j, i] = score
    return matrix


def _draw_axis(
    ax: plt.Axes,
    labels: list[str],
    matrix: np.ndarray,
    title: str,
    support: int,
    compact: bool = False,
) -> Any:
    data_layer, missing_layer = strict_lower_triangle_layers(matrix)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    white_cmap = ListedColormap(["#FFFFFF"])
    missing_cmap = ListedColormap(["#D9DEE7"])
    missing_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    ax.set_facecolor("white")
    ax.imshow(
        np.ones(np.asarray(matrix).shape, dtype=float),
        cmap=white_cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.imshow(
        missing_layer,
        cmap=missing_cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    image = ax.imshow(
        data_layer,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9 if compact else 11, pad=6)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90 if compact else 55, ha="right", fontsize=5.5 if compact else 6.5)
    ax.set_yticklabels(labels, fontsize=5.5 if compact else 6.5)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    if support <= 1:
        ax.text(
            0.5,
            0.08,
            "1 observed frame\nno off-diagonal pair",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7 if compact else 9,
            color="#27364B",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#9AA6B2", "boxstyle": "round,pad=0.3"},
        )
    return image


def plot_matrix(path: Path, labels: list[str], matrix: np.ndarray, title: str, support: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 8.3))
    image = _draw_axis(ax, labels, matrix, title, support)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="similarity (0-1)")
    fig.text(
        0.5,
        0.012,
        "Strict lower triangle only: diagonal and upper triangle are hidden. Gray = missing lower-triangle observation.",
        ha="center",
        fontsize=8,
        color="#455568",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_pair(
    path: Path,
    labels: list[str],
    location: np.ndarray,
    area: np.ndarray,
    track: int,
    support: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 8.1))
    image = _draw_axis(axes[0], labels, location, "Location similarity", support)
    _draw_axis(axes[1], labels, area, "Normalized ROI area similarity", support)
    fig.suptitle(f"Single crystal global track {track} across all Masked frames ({support}/12 observed)", fontsize=14, y=0.98)
    fig.text(
        0.5,
        0.012,
        "Strict lower triangle only; diagonal/upper hidden. Global track is not split by 0deg/10deg. Gray = missing.",
        ha="center",
        fontsize=8.5,
        color="#455568",
    )
    fig.subplots_adjust(left=0.075, right=0.88, bottom=0.19, top=0.90, wspace=0.20)
    colorbar_axis = fig.add_axes([0.91, 0.19, 0.014, 0.69])
    fig.colorbar(image, cax=colorbar_axis, label="similarity (0-1)")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_trajectory(path: Path, rows: list[dict[str, Any]], title: str) -> None:
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


def plot_gallery(
    out_dir: Path,
    labels: list[str],
    ordered_tracks: list[int],
    location_by_track: dict[int, np.ndarray],
    area_by_track: dict[int, np.ndarray],
    support_by_track: dict[int, int],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_labels = [label.split("\n", 1)[0] for label in labels]
    page_size = 6
    for page, start in enumerate(range(0, len(ordered_tracks), page_size), start=1):
        chunk = ordered_tracks[start:start + page_size]
        fig, axes = plt.subplots(
            len(chunk),
            2,
            figsize=(14.8, 3.35 * len(chunk) + 0.65),
            squeeze=False,
        )
        image = None
        for row_index, track in enumerate(chunk):
            support = support_by_track[track]
            image = _draw_axis(
                axes[row_index, 0],
                compact_labels,
                location_by_track[track],
                f"Track {track} location ({support}/12)",
                support,
                compact=True,
            )
            _draw_axis(
                axes[row_index, 1],
                compact_labels,
                area_by_track[track],
                f"Track {track} normalized area ({support}/12)",
                support,
                compact=True,
            )
        fig.suptitle(
            "Single-crystal global per-peak heatmaps — strict lower triangle",
            fontsize=14,
            y=0.985,
        )
        fig.subplots_adjust(
            left=0.055,
            right=0.875,
            bottom=0.050,
            top=0.935,
            hspace=0.62,
            wspace=0.22,
        )
        if image is not None:
            colorbar_axis = fig.add_axes([0.905, 0.10, 0.014, 0.78])
            fig.colorbar(image, cax=colorbar_axis, label="similarity (0-1)")
        fig.savefig(out_dir / f"paired_heatmaps_page_{page:02d}.png", dpi=165)
        plt.close(fig)
    return math.ceil(len(ordered_tracks) / page_size)


def regenerate_gallery_from_saved_outputs(out_root: Path) -> int:
    """Rebuild only the paginated gallery from the saved exact matrices."""
    with (out_root / "frame_registry.csv").open(newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    with (out_root / "track_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    payload = np.load(out_root / "per_track_matrices.npz", allow_pickle=False)
    track_ids = [int(value) for value in payload["track_ids"]]
    labels = [str(row["display_label"]).replace(" | ", "\n") for row in frame_rows]
    location_by_track = {
        track: np.asarray(payload["location_similarity"][index], dtype=float)
        for index, track in enumerate(track_ids)
    }
    area_by_track = {
        track: np.asarray(payload["normalized_area_similarity"][index], dtype=float)
        for index, track in enumerate(track_ids)
    }
    support_by_track = {
        int(row["track"]): int(row["frame_count"])
        for row in summary_rows
    }
    ordered_tracks = [
        int(row["track"])
        for row in sorted(
            summary_rows,
            key=lambda row: (-int(row["frame_count"]), int(row["track"])),
        )
    ]
    return plot_gallery(
        out_root / "gallery",
        labels,
        ordered_tracks,
        location_by_track,
        area_by_track,
        support_by_track,
    )


def regenerate_all_heatmaps_from_saved_outputs(out_root: Path) -> dict[str, int]:
    """Rebuild every single-global heatmap without changing saved matrices."""
    with (out_root / "frame_registry.csv").open(newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    with (out_root / "track_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    payload = np.load(out_root / "per_track_matrices.npz", allow_pickle=False)
    track_ids = [int(value) for value in payload["track_ids"]]
    labels = [str(row["display_label"]).replace(" | ", "\n") for row in frame_rows]
    location = np.asarray(payload["location_similarity"], dtype=float)
    area = np.asarray(payload["normalized_area_similarity"], dtype=float)
    location_by_track = {
        track: location[index]
        for index, track in enumerate(track_ids)
    }
    area_by_track = {
        track: area[index]
        for index, track in enumerate(track_ids)
    }
    support_by_track = {
        int(row["track"]): int(row["frame_count"])
        for row in summary_rows
    }

    for track in track_ids:
        support = support_by_track[track]
        stem = f"track_{track:03d}"
        plot_matrix(
            out_root / "location_heatmaps" / f"{stem}.png",
            labels,
            location_by_track[track],
            f"Global track {track}: location similarity ({support}/12 observed)",
            support,
        )
        plot_matrix(
            out_root / "normalized_area_heatmaps" / f"{stem}.png",
            labels,
            area_by_track[track],
            f"Global track {track}: normalized ROI area similarity ({support}/12 observed)",
            support,
        )
        plot_pair(
            out_root / "paired_heatmaps" / f"{stem}_location_area.png",
            labels,
            location_by_track[track],
            area_by_track[track],
            track,
            support,
        )

    plot_matrix(
        out_root / "aggregate_location_heatmap.png",
        labels,
        legacy.nanmedian(location, axis=0),
        "Track-median location similarity across all global tracks",
        len(labels),
    )
    plot_matrix(
        out_root / "aggregate_normalized_area_heatmap.png",
        labels,
        legacy.nanmedian(area, axis=0),
        "Track-median normalized ROI area similarity across all global tracks",
        len(labels),
    )
    ordered_tracks = [
        int(row["track"])
        for row in sorted(
            summary_rows,
            key=lambda row: (-int(row["frame_count"]), int(row["track"])),
        )
    ]
    gallery_pages = plot_gallery(
        out_root / "gallery",
        labels,
        ordered_tracks,
        location_by_track,
        area_by_track,
        support_by_track,
    )
    return {
        "location": len(track_ids),
        "normalized_area": len(track_ids),
        "paired": len(track_ids),
        "aggregate": 2,
        "gallery": gallery_pages,
    }


def _finite_pair_count(matrix: np.ndarray) -> int:
    lower = np.tril_indices(matrix.shape[0], k=-1)
    return int(np.count_nonzero(np.isfinite(matrix[lower])))


def _write_heatmap_index(path: Path) -> None:
    content = """# Single-crystal global per-peak heatmaps

The primary per-peak result groups by global track across all 12 available Masked frames. It is not split by 0deg/10deg; orientation and branch remain visible in every frame label.

Every correlation heatmap shows the strict lower triangle only. The diagonal and upper triangle are intentionally hidden because they contain self-comparisons or duplicate symmetric information. Exact numerical matrices remain complete in CSV/NPZ outputs.

- `paired_heatmaps/`: one location + normalized-area card for each of 75 tracks.
- `location_heatmaps/`: one location heatmap for each track.
- `normalized_area_heatmaps/`: one normalized ROI-area heatmap for each track.
- `gallery/`: paginated overview, sorted by frame support.
- `location_matrices/` and `normalized_area_matrices/`: exact CSV matrices.
- `track_summary.csv`: support, status, cross-orientation pairs, and image paths.

White cells on/above the diagonal are intentionally hidden. Gray cells below the diagonal mean that the peak was not observed in one or both frames. Singleton tracks are retained and explicitly marked as having no off-diagonal pair.
"""
    path.write_text(content, encoding="utf-8")


def analyze_single_tracks_across_frames(
    out_root: Path,
    observations: list[dict[str, Any]],
    metadata: dict[int, dict[str, Any]],
    make_plots: bool,
) -> dict[str, Any]:
    """Analyze each global track over all actual Masked frames."""
    out_root.mkdir(parents=True, exist_ok=True)
    collapsed = collapse_frame_track_rows(observations)
    write_csv(out_root / "track_observations.csv", observations)
    write_csv(out_root / "frame_track_features.csv", collapsed)

    frame_ids = sorted(
        {int(row["frame"]) for row in observations},
        key=lambda frame: (float(metadata[frame]["pressure_GPa"]), frame),
    )
    frame_rows: list[dict[str, Any]] = []
    for frame in frame_ids:
        meta = metadata[frame]
        branch = branch_label(meta)
        orientation = orientation_base(str(meta["orientation"]))
        pressure = float(meta["pressure_GPa"])
        display_branch = branch.replace("0deg", "0°").replace("10deg", "10°").replace("_", " ")
        frame_rows.append({
            "frame": frame,
            "pressure_GPa": pressure,
            "orientation": orientation,
            "branch": branch,
            "whole_pattern_included": int(meta["included_whole_pattern"]),
            "per_peak_included": 1,
            "whole_pattern_exclusion_reason": meta["exclusion_reason"],
            "machine_label": f"f{frame:04d}|{pressure:g}GPa|{branch}",
            "display_label": f"f{frame:04d} | {pressure:g} GPa | {display_branch}",
            "heatmap_label": f"f{frame:04d} / {pressure:g} GPa / {display_branch}",
        })
    write_csv(out_root / "frame_registry.csv", frame_rows)

    frame_index = {frame: index for index, frame in enumerate(frame_ids)}
    machine_labels = [str(row["machine_label"]) for row in frame_rows]
    heatmap_labels = [str(row["display_label"]).replace(" | ", "\n") for row in frame_rows]
    tracks = sorted({int(row["track"]) for row in collapsed})
    track_index = {track: index for index, track in enumerate(tracks)}
    shape = (len(tracks), len(frame_ids), len(frame_ids))
    location = np.full(shape, np.nan, dtype=np.float32)
    normalized_area = np.full(shape, np.nan, dtype=np.float32)
    observed = np.zeros((len(tracks), len(frame_ids)), dtype=bool)

    feature_lookup: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in collapsed:
        track = int(row["track"])
        frame = int(row["frame"])
        if frame in feature_lookup[track]:
            raise ValueError(f"Multiple collapsed features for global track/frame: {track}, {frame}")
        feature_lookup[track][frame] = row

    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    support_by_track: dict[int, int] = {}
    location_by_track: dict[int, np.ndarray] = {}
    area_by_track: dict[int, np.ndarray] = {}
    lower_indices = np.tril_indices(len(frame_ids), k=-1)

    for track in tracks:
        values_by_frame = feature_lookup[track]
        loc_values = np.full(len(frame_ids), np.nan, dtype=float)
        area_values = np.full(len(frame_ids), np.nan, dtype=float)
        for frame, row in values_by_frame.items():
            index = frame_index[frame]
            observed[track_index[track], index] = True
            loc_values[index] = float(row["two_theta_median_deg"])
            area_values[index] = float(row["normalized_area_median_counts_per_s_per_pixel"])

        loc_matrix = symmetric_similarity_matrix(loc_values, location=True)
        area_matrix = symmetric_similarity_matrix(area_values, location=False)
        location[track_index[track]] = loc_matrix
        normalized_area[track_index[track]] = area_matrix
        location_by_track[track] = loc_matrix
        area_by_track[track] = area_matrix

        available_frames = sorted(values_by_frame, key=lambda frame: frame_index[frame])
        available_indices = [frame_index[frame] for frame in available_frames]
        support = len(available_frames)
        support_by_track[track] = support
        cross_pairs = 0
        for local_i, i in enumerate(available_indices):
            for j in available_indices[:local_i]:
                frame_a = frame_ids[i]
                frame_b = frame_ids[j]
                row_a = frame_rows[i]
                row_b = frame_rows[j]
                is_cross = int(row_a["orientation"] != row_b["orientation"])
                cross_pairs += is_cross
                common = {
                    "track": track,
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "pressure_a_GPa": float(row_a["pressure_GPa"]),
                    "pressure_b_GPa": float(row_b["pressure_GPa"]),
                    "pressure_gap_GPa": abs(float(row_a["pressure_GPa"]) - float(row_b["pressure_GPa"])),
                    "orientation_a": row_a["orientation"],
                    "orientation_b": row_b["orientation"],
                    "branch_a": row_a["branch"],
                    "branch_b": row_b["branch"],
                    "cross_orientation": is_cross,
                }
                if np.isfinite(loc_matrix[i, j]):
                    pair_rows.append({**common, "family": "location", "similarity": float(loc_matrix[i, j])})
                if np.isfinite(area_matrix[i, j]):
                    pair_rows.append({**common, "family": "normalized_area", "similarity": float(area_matrix[i, j])})

        track_rows = [values_by_frame[frame] for frame in available_frames]
        pressures = np.asarray([float(row["pressure_GPa"]) for row in track_rows], dtype=float)
        d_values = np.asarray([float(row["d_median_A"]) for row in track_rows], dtype=float)
        slope, slope_r, slope_r2 = legacy.linear_summary(pressures, d_values)
        anchors = [
            float(row["matched_d_A_anchor_candidate"])
            for row in track_rows
            if np.isfinite(float(row["matched_d_A_anchor_candidate"]))
        ]
        status = "usable_ge3_frames" if support >= 3 else ("comparable_2_frames" if support == 2 else "insufficient_single_frame")
        loc_pair_count = int(np.count_nonzero(np.isfinite(loc_matrix[lower_indices])))
        area_pair_count = int(np.count_nonzero(np.isfinite(area_matrix[lower_indices])))
        stem = f"track_{track:03d}"
        summary_rows.append({
            "dataset": "single_crystal_global_masked",
            "track": track,
            "observed_d_median_A": float(np.median(d_values)),
            "matched_d_A_reference_median": float(np.median(anchors)) if anchors else np.nan,
            "matched_d_A_reference_min": float(np.min(anchors)) if anchors else np.nan,
            "matched_d_A_reference_max": float(np.max(anchors)) if anchors else np.nan,
            "matched_d_A_reference_unique_count": len({round(value, 5) for value in anchors}),
            "frame_count": support,
            "pressure_points": len(np.unique(pressures)),
            "pressure_min_GPa": float(np.min(pressures)),
            "pressure_max_GPa": float(np.max(pressures)),
            "orientations_observed": ";".join(sorted({orientation_base(str(metadata[frame]["orientation"])) for frame in available_frames})),
            "branches_observed": ";".join(sorted({branch_label(metadata[frame]) for frame in available_frames})),
            "frames_observed": ";".join(f"f{frame:04d}" for frame in available_frames),
            "raw_observation_count": sum(int(row["n_observations"]) for row in track_rows),
            "duplicate_frame_track_count": sum(int(row["duplicate_observation_flag"]) for row in track_rows),
            "status": status,
            "dd_dp_A_per_GPa": slope,
            "d_slope_r": slope_r,
            "d_slope_r2": slope_r2,
            "finite_location_pairs": loc_pair_count,
            "finite_normalized_area_pairs": area_pair_count,
            "cross_orientation_pair_count": cross_pairs,
            "paired_heatmap": f"paired_heatmaps/{stem}_location_area.png",
            "location_heatmap": f"location_heatmaps/{stem}.png",
            "normalized_area_heatmap": f"normalized_area_heatmaps/{stem}.png",
        })

        legacy.write_matrix_csv(out_root / "location_matrices" / f"{stem}.csv", machine_labels, loc_matrix)
        legacy.write_matrix_csv(out_root / "normalized_area_matrices" / f"{stem}.csv", machine_labels, area_matrix)
        if make_plots:
            plot_matrix(
                out_root / "location_heatmaps" / f"{stem}.png",
                heatmap_labels,
                loc_matrix,
                f"Global track {track}: location similarity ({support}/12 observed)",
                support,
            )
            plot_matrix(
                out_root / "normalized_area_heatmaps" / f"{stem}.png",
                heatmap_labels,
                area_matrix,
                f"Global track {track}: normalized ROI area similarity ({support}/12 observed)",
                support,
            )
            plot_pair(
                out_root / "paired_heatmaps" / f"{stem}_location_area.png",
                heatmap_labels,
                loc_matrix,
                area_matrix,
                track,
                support,
            )
            if support >= 2:
                plot_trajectory(
                    out_root / "trajectories" / f"{stem}_d_vs_pressure.png",
                    track_rows,
                    f"single crystal global track {track}: d(P)",
                )

    write_csv(out_root / "track_summary.csv", summary_rows)
    write_csv(out_root / "all_pair_scores.csv", pair_rows)
    np.savez_compressed(
        out_root / "per_track_matrices.npz",
        frame_ids=np.asarray(frame_ids, dtype=int),
        pressure_gpa=np.asarray([row["pressure_GPa"] for row in frame_rows], dtype=float),
        orientation_labels=np.asarray([row["orientation"] for row in frame_rows]),
        branch_labels=np.asarray([row["branch"] for row in frame_rows]),
        machine_labels=np.asarray(machine_labels),
        track_ids=np.asarray(tracks, dtype=int),
        observed_mask=observed,
        location_similarity=location,
        normalized_area_similarity=normalized_area,
    )

    aggregate_location = legacy.nanmedian(location, axis=0)
    aggregate_area = legacy.nanmedian(normalized_area, axis=0)
    legacy.write_matrix_csv(out_root / "aggregate_location_matrix.csv", machine_labels, aggregate_location)
    legacy.write_matrix_csv(out_root / "aggregate_normalized_area_matrix.csv", machine_labels, aggregate_area)
    gallery_pages = 0
    if make_plots:
        plot_matrix(
            out_root / "aggregate_location_heatmap.png",
            heatmap_labels,
            aggregate_location,
            "Track-median location similarity across all global tracks",
            len(frame_ids),
        )
        plot_matrix(
            out_root / "aggregate_normalized_area_heatmap.png",
            heatmap_labels,
            aggregate_area,
            "Track-median normalized ROI area similarity across all global tracks",
            len(frame_ids),
        )
        ordered_tracks = [int(row["track"]) for row in sorted(summary_rows, key=lambda row: (-int(row["frame_count"]), int(row["track"])))]
        gallery_pages = plot_gallery(
            out_root / "gallery",
            heatmap_labels,
            ordered_tracks,
            location_by_track,
            area_by_track,
            support_by_track,
        )
    _write_heatmap_index(out_root / "HEATMAP_INDEX.md")

    location_diag = np.diagonal(location, axis1=1, axis2=2)
    area_diag = np.diagonal(normalized_area, axis1=1, axis2=2)
    diagonal_ok = (
        np.array_equal(np.isfinite(location_diag), observed)
        and np.array_equal(np.isfinite(area_diag), observed)
        and np.allclose(location_diag[observed], 1.0)
        and np.allclose(area_diag[observed], 1.0)
    )
    track18 = next((row for row in summary_rows if int(row["track"]) == 18), None)
    probe = np.arange(16, dtype=float).reshape(4, 4)
    probe_data, probe_missing = strict_lower_triangle_layers(probe)
    probe_data_mask = np.ma.getmaskarray(probe_data)
    probe_missing_mask = np.ma.getmaskarray(probe_missing)
    probe_structural_hidden = probe_data_mask & probe_missing_mask
    probe_lower = np.tril_indices(4, k=-1)
    probe_upper = np.triu_indices(4, k=0)
    return {
        "tracks": len(tracks),
        "raw_observations": len(observations),
        "masked_frames": len(frame_ids),
        "frame_ids": frame_ids,
        "frame_track_features": len(collapsed),
        "duplicate_frame_track_features": sum(int(row["duplicate_observation_flag"]) for row in collapsed),
        "duplicate_extra_observations": len(observations) - len(collapsed),
        "comparable_tracks_ge2_frames": sum(int(row["frame_count"]) >= 2 for row in summary_rows),
        "usable_trajectories_ge3_frames": sum(int(row["frame_count"]) >= 3 for row in summary_rows),
        "singleton_tracks": sum(int(row["frame_count"]) == 1 for row in summary_rows),
        "location_unique_pairs": int(np.count_nonzero(np.isfinite(location[:, lower_indices[0], lower_indices[1]]))),
        "normalized_area_unique_pairs": int(np.count_nonzero(np.isfinite(normalized_area[:, lower_indices[0], lower_indices[1]]))),
        "cross_orientation_tracks": sum(int(row["cross_orientation_pair_count"]) > 0 for row in summary_rows),
        "cross_orientation_location_pairs": sum(int(row["cross_orientation"]) for row in pair_rows if row["family"] == "location"),
        "track18_frame_support": int(track18["frame_count"]) if track18 else 0,
        "track18_unique_pairs": int(track18["finite_location_pairs"]) if track18 else 0,
        "matrices_symmetric": bool(
            np.allclose(location, np.swapaxes(location, 1, 2), equal_nan=True)
            and np.allclose(normalized_area, np.swapaxes(normalized_area, 1, 2), equal_nan=True)
        ),
        "matrix_diagonal_matches_observed": bool(diagonal_ok),
        "matrix_scores_in_unit_interval": bool(
            np.all((location[np.isfinite(location)] >= 0) & (location[np.isfinite(location)] <= 1))
            and np.all((normalized_area[np.isfinite(normalized_area)] >= 0) & (normalized_area[np.isfinite(normalized_area)] <= 1))
        ),
        "heatmap_triangle_policy": HEATMAP_TRIANGLE_POLICY,
        "heatmap_diagonal_and_upper_hidden": bool(
            np.all(probe_structural_hidden[probe_upper])
        ),
        "heatmap_lower_triangle_preserved": bool(
            np.all(~probe_data_mask[probe_lower])
            and np.allclose(np.asarray(probe_data)[probe_lower], probe[probe_lower])
        ),
        "paired_heatmaps": len(list((out_root / "paired_heatmaps").glob("track_*_location_area.png"))) if make_plots else 0,
        "location_heatmaps": len(list((out_root / "location_heatmaps").glob("track_*.png"))) if make_plots else 0,
        "normalized_area_heatmaps": len(list((out_root / "normalized_area_heatmaps").glob("track_*.png"))) if make_plots else 0,
        "gallery_pages": gallery_pages,
        "area_status_counts": dict(Counter(str(row["intensity_status"]) for row in observations)),
        "summary_rows": summary_rows,
    }
