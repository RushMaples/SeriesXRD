#!/usr/bin/env python3
"""Serialize uniform-correlation numeric results into auditable artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

import uniform_peak_core as up
import uniform_window_core as uw
from uniform_correlation_io import plot_matrix, safe_name, write_matrix_csv, write_rows_csv


PEAK_FIT_FIELDS = [
    "channel",
    "frame",
    "scan",
    "pressure_GPa",
    "pattern_valid",
    "peak_id",
    "state",
    "reason",
    "candidate_index",
    "two_theta_deg",
    "q_A^-1",
    "fwhm_two_theta_deg",
    "fwhm_q_A^-1",
    "raw_fitted_area",
    "area_se",
    "relative_area",
    "eta",
    "height_snr",
    "delta_bic",
    "fit_success",
    "at_parameter_boundary",
    "group_id",
    "fit_model",
]
CONSENSUS_FIELDS = [
    "channel",
    "consensus_id",
    "pressure_GPa",
    "pressure_index",
    "q_A^-1",
    "fwhm_q_A^-1",
    "relative_area",
    "scan_support",
    "total_scans",
    "required_support",
    "reliable",
    "ambiguous",
]
TRACK_FIELDS = [
    "channel",
    "track_id",
    "official",
    "ambiguous",
    "pressure_min_GPa",
    "pressure_max_GPa",
    "pressure_nodes",
    "minimum_pressure_nodes",
    "median_q_A^-1",
    "median_fwhm_q_A^-1",
    "node_ids",
]
TRAJECTORY_NODE_FIELDS = [
    "channel",
    "track_id",
    "official",
    "ambiguous",
    "consensus_id",
    "pressure_GPa",
    "pressure_index",
    "q_A^-1",
    "fwhm_q_A^-1",
    "relative_area",
    "scan_support",
]
OBSERVATION_FIELDS = [
    "channel",
    "track_id",
    "track_official",
    "scan",
    "pressure_GPa",
    "frame",
    "state",
    "reason",
    "q_A^-1",
    "fwhm_q_A^-1",
    "relative_area",
    "peak_id",
]
NEAR_FAR_FIELDS = [
    "near_gap_max",
    "far_gap_min",
    "near_median",
    "far_median",
    "near_far_median_difference",
    "auc",
    "near_count",
    "far_count",
    "distinct_gap_count",
    "reason",
    "auc_ci_low",
    "auc_ci_high",
    "near_far_median_difference_ci_low",
    "near_far_median_difference_ci_high",
    "bootstrap_iterations",
    "random_seed",
]
PEAK_SUMMARY_FIELDS = [
    *TRACK_FIELDS,
    "present_observations",
    "absent_observations",
    "unknown_observations",
    "out_of_range_observations",
    *[
        f"{family}_{field}"
        for family in ("area", "location", "presence")
        for field in NEAR_FAR_FIELDS
    ],
]


def _pressure_labels(values: Sequence[float]) -> list[str]:
    return [f"{float(value):g}" for value in values]


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else math.nan


def write_per_peak_results(
    channel_root: Path,
    channel: str,
    analysis: up.PerPeakAnalysis,
    frame_peaks: Sequence[up.FramePeaks],
    *,
    scans: Sequence[str],
    pressures: Sequence[float],
    make_plots: bool,
) -> dict[str, Any]:
    root = channel_root / "per_peak"
    root.mkdir(parents=True, exist_ok=True)
    pressure_values = np.asarray(pressures, dtype=float)
    pressure_labels = _pressure_labels(pressure_values)
    official_tracks = [
        track for track in analysis.tracks if track.official and track.track_id in analysis.correlations
    ]
    for family in ("area", "location", "presence"):
        for subdirectory in ("matrices", "confidence_intervals", "heatmaps", "support_maps"):
            (root / family / subdirectory).mkdir(parents=True, exist_ok=True)
    for subdirectory in ("matrices", "heatmaps"):
        (root / "support" / subdirectory).mkdir(parents=True, exist_ok=True)
    (root / "trajectories" / "plots").mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    for frame in frame_peaks:
        if not frame.peaks:
            candidate_rows.append(
                {
                    "channel": channel,
                    "frame": frame.frame,
                    "scan": frame.scan,
                    "pressure_GPa": frame.pressure,
                    "pattern_valid": int(frame.pattern_valid),
                    "peak_id": "",
                    "state": "no_candidate" if frame.pattern_valid else "invalid_pattern",
                    "reason": ";".join(frame.warnings),
                }
            )
        for peak in frame.peaks:
            candidate_rows.append(
                {
                    "channel": channel,
                    "frame": frame.frame,
                    "scan": frame.scan,
                    "pressure_GPa": frame.pressure,
                    "pattern_valid": int(frame.pattern_valid),
                    "peak_id": peak.peak_id,
                    "candidate_index": peak.candidate_index,
                    "state": peak.state,
                    "reason": peak.reason,
                    "two_theta_deg": peak.two_theta,
                    "q_A^-1": peak.q,
                    "fwhm_two_theta_deg": peak.fwhm_two_theta,
                    "fwhm_q_A^-1": peak.fwhm_q,
                    "raw_fitted_area": peak.area,
                    "area_se": peak.area_se,
                    "relative_area": peak.relative_area,
                    "eta": peak.eta,
                    "height_snr": peak.height_snr,
                    "delta_bic": peak.delta_bic,
                    "fit_success": int(peak.fit_success),
                    "at_parameter_boundary": int(peak.at_parameter_boundary),
                    "group_id": peak.group_id,
                    "fit_model": peak.fit_model,
                }
            )
    write_rows_csv(root / "detected_peak_fits.csv", candidate_rows, fieldnames=PEAK_FIT_FIELDS)

    consensus_rows: list[dict[str, Any]] = []
    for pressure in pressure_values:
        for item in analysis.consensus_by_pressure.get(float(pressure), ()):
            consensus_rows.append(
                {
                    "channel": channel,
                    "consensus_id": item.consensus_id,
                    "pressure_GPa": item.pressure,
                    "pressure_index": item.pressure_index,
                    "q_A^-1": item.q,
                    "fwhm_q_A^-1": item.fwhm_q,
                    "relative_area": item.relative_area,
                    "scan_support": item.support,
                    "total_scans": item.total_scans,
                    "required_support": item.required_support,
                    "reliable": int(item.reliable),
                    "ambiguous": int(item.ambiguous),
                }
            )
    write_rows_csv(
        root / "pressure_consensus_nodes.csv",
        consensus_rows,
        fieldnames=CONSENSUS_FIELDS,
    )

    track_rows: list[dict[str, Any]] = []
    for track in analysis.tracks:
        q_values = np.asarray([node.q for node in track.nodes], dtype=float)
        widths = np.asarray([node.fwhm_q for node in track.nodes], dtype=float)
        track_rows.append(
            {
                "channel": channel,
                "track_id": track.track_id,
                "official": int(track.official),
                "ambiguous": int(track.ambiguous),
                "pressure_min_GPa": track.pressure_min,
                "pressure_max_GPa": track.pressure_max,
                "pressure_nodes": len(track.nodes),
                "minimum_pressure_nodes": track.minimum_pressure_support,
                "median_q_A^-1": _finite_median(q_values),
                "median_fwhm_q_A^-1": _finite_median(widths),
                "node_ids": ";".join(node.consensus_id for node in track.nodes),
            }
        )
    write_rows_csv(root / "canonical_tracks.csv", track_rows, fieldnames=TRACK_FIELDS)
    write_rows_csv(
        root / "trajectories" / "canonical_tracks.csv",
        track_rows,
        fieldnames=TRACK_FIELDS,
    )
    trajectory_node_rows: list[dict[str, Any]] = []
    for track in analysis.tracks:
        for node in track.nodes:
            trajectory_node_rows.append(
                {
                    "channel": channel,
                    "track_id": track.track_id,
                    "official": int(track.official),
                    "ambiguous": int(track.ambiguous),
                    "consensus_id": node.consensus_id,
                    "pressure_GPa": node.pressure,
                    "pressure_index": node.pressure_index,
                    "q_A^-1": node.q,
                    "fwhm_q_A^-1": node.fwhm_q,
                    "relative_area": node.relative_area,
                    "scan_support": node.support,
                }
            )
        if make_plots and track.official:
            pressure_axis = np.asarray([node.pressure for node in track.nodes], dtype=float)
            q_axis = np.asarray([node.q for node in track.nodes], dtype=float)
            width_axis = np.asarray([node.fwhm_q for node in track.nodes], dtype=float)
            fig, ax = plt.subplots(figsize=(7.0, 4.4))
            ax.errorbar(pressure_axis, q_axis, yerr=0.5 * width_axis, fmt="o-", capsize=2)
            ax.set_xlabel("Pressure (GPa)")
            ax.set_ylabel("q (Å⁻¹)")
            ax.set_title(f"{channel}: {track.track_id} blind radial trajectory")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            plot_path = root / "trajectories" / "plots" / f"{track.track_id}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=180)
            plt.close(fig)
    write_rows_csv(
        root / "trajectories" / "trajectory_nodes.csv",
        trajectory_node_rows,
        fieldnames=TRAJECTORY_NODE_FIELDS,
    )

    observation_rows: list[dict[str, Any]] = []
    assigned_peak_keys: set[tuple[int, int]] = set()
    duplicate_present_keys: list[tuple[int, int]] = []
    for track in analysis.tracks:
        for (scan, pressure), observation in sorted(
            analysis.assignments[track.track_id].items(), key=lambda item: (item[0][0], item[0][1])
        ):
            if observation.state == "present" and observation.frame is not None and observation.peak_id is not None:
                key = (int(observation.frame), int(observation.peak_id))
                if key in assigned_peak_keys:
                    duplicate_present_keys.append(key)
                assigned_peak_keys.add(key)
            observation_rows.append(
                {
                    "channel": channel,
                    "track_id": track.track_id,
                    "track_official": int(track.official),
                    "scan": scan,
                    "pressure_GPa": pressure,
                    "frame": observation.frame,
                    "state": observation.state,
                    "reason": observation.reason,
                    "q_A^-1": observation.q,
                    "fwhm_q_A^-1": observation.fwhm_q,
                    "relative_area": observation.relative_area,
                    "peak_id": observation.peak_id,
                }
            )
    if duplicate_present_keys:
        raise ValueError(f"one detected peak was assigned to multiple tracks: {duplicate_present_keys[:5]}")
    write_rows_csv(
        root / "peak_observations.csv",
        observation_rows,
        fieldnames=OBSERVATION_FIELDS,
    )

    summary_rows: list[dict[str, Any]] = []
    npz_payload: dict[str, np.ndarray] = {
        "pressure_gpa": pressure_values,
        "scan_names": np.asarray(scans),
        "track_ids": np.asarray([track.track_id for track in official_tracks]),
    }
    array_fields = [
        "area",
        "location",
        "presence",
        "n_available",
        "n_both_present",
        "n10",
        "n01",
        "n_unknown",
        "required_support",
        "area_ci_low",
        "area_ci_high",
        "location_ci_low",
        "location_ci_high",
        "presence_ci_low",
        "presence_ci_high",
        "area_by_scan",
        "location_by_scan",
        "presence_by_scan",
    ]
    stacked: dict[str, list[np.ndarray]] = {field: [] for field in array_fields}

    for track in official_tracks:
        matrices = analysis.correlations[track.track_id]
        insufficient = matrices.n_both_present < matrices.required_support
        family_matrices = {
            "area": matrices.area,
            "location": matrices.location,
            "presence": matrices.presence,
        }
        for family, matrix in family_matrices.items():
            family_dir = root / family
            family_support = (
                matrices.n_available if family == "presence" else matrices.n_both_present
            )
            write_matrix_csv(
                family_dir / "matrices" / f"{track.track_id}.csv",
                pressure_labels,
                matrix,
            )
            write_matrix_csv(
                family_dir / "support_maps" / f"{track.track_id}_support.csv",
                pressure_labels,
                family_support,
            )
            low = getattr(matrices, f"{family}_ci_low")
            high = getattr(matrices, f"{family}_ci_high")
            write_matrix_csv(
                family_dir / "confidence_intervals" / f"{track.track_id}_ci_low.csv",
                pressure_labels,
                low,
            )
            write_matrix_csv(
                family_dir / "confidence_intervals" / f"{track.track_id}_ci_high.csv",
                pressure_labels,
                high,
            )
            if make_plots:
                plot_matrix(
                    family_dir / "heatmaps" / f"{track.track_id}.png",
                    pressure_labels,
                    matrix,
                    f"{channel}: {track.track_id} {family}",
                    vmin=0.0,
                    vmax=1.0,
                    cmap="viridis",
                    colorbar_label=(
                        f"conditional {family} similarity"
                        if family != "presence"
                        else "presence Jaccard"
                    ),
                    insufficient_mask=(
                        insufficient
                        if family in {"area", "location"}
                        else matrices.n_available < matrices.required_support
                    ),
                )
                plot_matrix(
                    family_dir / "support_maps" / f"{track.track_id}_support.png",
                    pressure_labels,
                    family_support,
                    f"{channel}: {track.track_id} {family} scan support",
                    vmin=0.0,
                    vmax=float(max(len(scans), 1)),
                    cmap="cividis",
                    colorbar_label=(
                        "paired scans" if family != "presence" else "available scans"
                    ),
                    integer_annotations=True,
                )
        support_dir = root / "support"
        support_items = {
            "n_available": matrices.n_available,
            "n_both_present": matrices.n_both_present,
            "n10": matrices.n10,
            "n01": matrices.n01,
            "n_unknown": matrices.n_unknown,
            "required_support": matrices.required_support,
        }
        for name, matrix in support_items.items():
            write_matrix_csv(
                support_dir / "matrices" / f"{track.track_id}_{name}.csv",
                pressure_labels,
                matrix,
            )
        if make_plots:
            plot_matrix(
                support_dir / "heatmaps" / f"{track.track_id}_n_both_present.png",
                pressure_labels,
                matrices.n_both_present,
                f"{channel}: {track.track_id} co-detection support",
                vmin=0.0,
                vmax=float(max(len(scans), 1)),
                cmap="cividis",
                colorbar_label="paired scans",
                integer_annotations=True,
            )

        track_meta = next(row for row in track_rows if row["track_id"] == track.track_id)
        area_stats = dict(analysis.near_far.get(track.track_id, {}).get("area", {}))
        location_stats = dict(analysis.near_far.get(track.track_id, {}).get("location", {}))
        presence_stats = dict(analysis.near_far.get(track.track_id, {}).get("presence", {}))
        states = [
            item.state for item in analysis.assignments[track.track_id].values()
        ]
        row = {
            **track_meta,
            "present_observations": states.count("present"),
            "absent_observations": states.count("absent"),
            "unknown_observations": states.count("unknown"),
            "out_of_range_observations": states.count("out_of_range"),
        }
        for prefix, stats in (
            ("area", area_stats),
            ("location", location_stats),
            ("presence", presence_stats),
        ):
            for key, value in stats.items():
                row[f"{prefix}_{key}"] = value
        summary_rows.append(row)
        for field in array_fields:
            if hasattr(matrices, field):
                stacked[field].append(np.asarray(getattr(matrices, field)))
    write_rows_csv(
        root / "peak_summary.csv",
        summary_rows,
        fieldnames=PEAK_SUMMARY_FIELDS,
    )
    for field, values in stacked.items():
        if values:
            npz_payload[field] = np.stack(values)
        elif field.endswith("_by_scan"):
            npz_payload[field] = np.empty(
                (0, len(scans), pressure_values.size, pressure_values.size),
                dtype=float,
            )
        else:
            dtype = (
                np.int32
                if field
                in {
                    "n_available",
                    "n_both_present",
                    "n10",
                    "n01",
                    "n_unknown",
                    "required_support",
                }
                else float
            )
            npz_payload[field] = np.empty(
                (0, pressure_values.size, pressure_values.size),
                dtype=dtype,
            )
    np.savez_compressed(root / "per_peak_matrices.npz", **npz_payload)

    area_auc = np.asarray([row.get("area_auc", math.nan) for row in summary_rows], dtype=float)
    location_auc = np.asarray([row.get("location_auc", math.nan) for row in summary_rows], dtype=float)
    return {
        "candidate_fits": sum(len(frame.peaks) for frame in frame_peaks),
        "reliable_candidate_fits": sum(
            int(peak.reliable) for frame in frame_peaks for peak in frame.peaks
        ),
        "all_radial_tracks": len(analysis.tracks),
        "official_radial_tracks": len(official_tracks),
        "median_area_auc": _finite_median(area_auc),
        "median_location_auc": _finite_median(location_auc),
        "peak_summary_rows": summary_rows,
    }


def write_across_results(
    channel_root: Path,
    channel: str,
    across: uw.AcrossFrameCorrelations,
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float,
    minimum_distinct_gaps: int,
    minimum_group_values: int,
    near_gap_quantile: float,
    far_gap_quantile: float,
    make_plots: bool,
) -> dict[str, Any]:
    root = channel_root / "across_frames"
    root.mkdir(parents=True, exist_ok=True)
    pressure_labels = _pressure_labels(across.pressure_values)
    families = {
        "acf_strict": across.acf_strict_by_scan,
        "direct_strict": across.direct_strict_by_scan,
        "shift_tolerant_secondary": across.shift_tolerant_by_scan,
    }
    payload: dict[str, np.ndarray] = {
        "pressure_gpa": across.pressure_values,
        "scan_names": np.asarray(across.scan_labels),
        "window_starts_deg": across.window_spec.starts_deg,
        "window_ends_deg": across.window_spec.ends_deg,
        "window_width_deg": np.asarray(across.window_spec.width_deg),
        "window_step_deg": np.asarray(across.window_spec.step_deg),
        "availability_by_scan": across.availability_by_scan,
    }
    family_metrics: dict[str, Any] = {}
    all_summary_rows: list[dict[str, Any]] = []

    for family, values_by_scan in families.items():
        aggregate = uw.aggregate_scan_matrices(
            values_by_scan,
            across.availability_by_scan,
            n_bootstrap=n_bootstrap,
            seed=seed,
            confidence=confidence,
        )
        near_far = uw.near_far_auc_summary(
            values_by_scan,
            across.pressure_values,
            across.availability_by_scan,
            n_bootstrap=n_bootstrap,
            seed=seed,
            confidence=confidence,
            minimum_distinct_gaps=minimum_distinct_gaps,
            minimum_group_values=minimum_group_values,
            near_gap_quantile=near_gap_quantile,
            far_gap_quantile=far_gap_quantile,
        )
        family_dir = root / family
        summary_rows: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(
            zip(across.window_spec.starts_deg, across.window_spec.ends_deg, strict=True)
        ):
            stem = f"window_{start:.4f}_{end:.4f}"
            matrix = aggregate.median[index]
            support = aggregate.support[index]
            insufficient = ~aggregate.sufficient_support[index]
            write_matrix_csv(family_dir / "matrices" / f"{stem}.csv", pressure_labels, matrix)
            write_matrix_csv(
                family_dir / "support_maps" / f"{stem}_support.csv",
                pressure_labels,
                support,
            )
            write_matrix_csv(
                family_dir / "confidence_intervals" / f"{stem}_ci_low.csv",
                pressure_labels,
                aggregate.ci_low[index],
            )
            write_matrix_csv(
                family_dir / "confidence_intervals" / f"{stem}_ci_high.csv",
                pressure_labels,
                aggregate.ci_high[index],
            )
            if make_plots:
                plot_matrix(
                    family_dir / "heatmaps" / f"{stem}.png",
                    pressure_labels,
                    matrix,
                    f"{channel}: {family} {start:.2f}-{end:.2f}°",
                    vmin=-1.0,
                    vmax=1.0,
                    cmap="coolwarm",
                    colorbar_label="Pearson similarity",
                    insufficient_mask=insufficient,
                )
                plot_matrix(
                    family_dir / "support_maps" / f"{stem}_support.png",
                    pressure_labels,
                    support,
                    f"{channel}: {family} paired-scan support {start:.2f}-{end:.2f}°",
                    vmin=0.0,
                    vmax=float(max(len(across.scan_labels), 1)),
                    cmap="cividis",
                    colorbar_label="paired scans",
                    integer_annotations=True,
                )
            row = {
                "channel": channel,
                "family": family,
                "window_index": index,
                "start_deg": float(start),
                "end_deg": float(end),
                "median_similarity": _finite_median(matrix),
                "near_gap_max_GPa": near_far.near_gap_max,
                "far_gap_min_GPa": near_far.far_gap_min,
                "near_median": float(near_far.near_median[index]),
                "far_median": float(near_far.far_median[index]),
                "near_vs_far_auc": float(near_far.auc[index]),
                "auc_ci_low": float(near_far.auc_ci_low[index]),
                "auc_ci_high": float(near_far.auc_ci_high[index]),
                "near_supported_cells": int(near_far.near_cells[index]),
                "far_supported_cells": int(near_far.far_cells[index]),
                "auc_reason_if_na": near_far.reasons[index],
                "finite_cells": int(np.count_nonzero(np.isfinite(matrix))),
                "insufficient_support_cells": int(np.count_nonzero(insufficient)),
            }
            summary_rows.append(row)
            all_summary_rows.append(row)
        write_rows_csv(family_dir / "window_summary.csv", summary_rows)
        payload.update(
            {
                f"{family}_by_scan": values_by_scan,
                f"{family}_aggregate": aggregate.median,
                f"{family}_ci_low": aggregate.ci_low,
                f"{family}_ci_high": aggregate.ci_high,
                f"{family}_support": aggregate.support,
                f"{family}_available": aggregate.available,
                f"{family}_support_required": aggregate.support_required,
                f"{family}_sufficient_support": aggregate.sufficient_support,
                f"{family}_near_median": near_far.near_median,
                f"{family}_far_median": near_far.far_median,
                f"{family}_near_far_auc": near_far.auc,
                f"{family}_auc_ci_low": near_far.auc_ci_low,
                f"{family}_auc_ci_high": near_far.auc_ci_high,
            }
        )
        finite_auc = near_far.auc[np.isfinite(near_far.auc)]
        family_metrics[family] = {
            "windows": len(summary_rows),
            "median_window_auc": float(np.median(finite_auc)) if finite_auc.size else math.nan,
            "best_window_auc": float(np.max(finite_auc)) if finite_auc.size else math.nan,
            "near_gap_max_GPa": near_far.near_gap_max,
            "far_gap_min_GPa": near_far.far_gap_min,
            "supported_windows": int(finite_auc.size),
        }
    np.savez_compressed(root / "across_frame_matrices.npz", **payload)
    write_rows_csv(root / "all_window_summaries.csv", all_summary_rows)
    return {
        "window_width_deg": across.window_spec.width_deg,
        "window_step_deg": across.window_spec.step_deg,
        "windows": len(across.window_spec.starts_deg),
        "families": family_metrics,
    }


def write_within_results(
    channel_root: Path,
    channel: str,
    within: uw.WithinFrameCorrelations,
    window_spec: uw.WindowSpec,
    *,
    frame_ids: Sequence[int],
    frame_scans: Sequence[str],
    frame_pressures: Sequence[float],
    n_bootstrap: int,
    seed: int,
    confidence: float,
    make_plots: bool,
) -> dict[str, Any]:
    root = channel_root / "within_frame"
    labels = list(window_spec.labels)
    nonoverlap_indices = np.asarray(within.nonoverlap_indices, dtype=int)
    nonoverlap_labels = [labels[index] for index in nonoverlap_indices]
    aggregate = uw.aggregate_scan_matrices(
        within.by_scan,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )
    all_dir = root / "all_windows"
    write_matrix_csv(all_dir / "aggregate_matrix.csv", labels, aggregate.median, row_header="window")
    write_matrix_csv(all_dir / "support_matrix.csv", labels, aggregate.support, row_header="window")
    write_matrix_csv(all_dir / "ci_low_matrix.csv", labels, aggregate.ci_low, row_header="window")
    write_matrix_csv(all_dir / "ci_high_matrix.csv", labels, aggregate.ci_high, row_header="window")
    if make_plots:
        plot_matrix(
            all_dir / "aggregate_heatmap.png",
            labels,
            aggregate.median,
            f"{channel}: within-frame all-window ACF",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            colorbar_label="ACF Pearson similarity",
            insufficient_mask=~aggregate.sufficient_support,
        )
        plot_matrix(
            all_dir / "support_heatmap.png",
            labels,
            aggregate.support,
            f"{channel}: within-frame all-window scan support",
            vmin=0.0,
            vmax=float(max(len(within.scan_labels), 1)),
            cmap="cividis",
            colorbar_label="scans",
        )
        plot_matrix(
            all_dir / "ci_low_heatmap.png",
            labels,
            aggregate.ci_low,
            f"{channel}: within-frame all-window 95% CI low",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            colorbar_label="ACF Pearson similarity",
            insufficient_mask=~aggregate.sufficient_support,
        )
        plot_matrix(
            all_dir / "ci_high_heatmap.png",
            labels,
            aggregate.ci_high,
            f"{channel}: within-frame all-window 95% CI high",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            colorbar_label="ACF Pearson similarity",
            insufficient_mask=~aggregate.sufficient_support,
        )

    non_dir = root / "nonoverlap_control"
    non_aggregate = aggregate.median[np.ix_(nonoverlap_indices, nonoverlap_indices)]
    non_support = aggregate.support[np.ix_(nonoverlap_indices, nonoverlap_indices)]
    non_available = aggregate.available[np.ix_(nonoverlap_indices, nonoverlap_indices)]
    non_required = aggregate.support_required[np.ix_(nonoverlap_indices, nonoverlap_indices)]
    non_sufficient = aggregate.sufficient_support[np.ix_(nonoverlap_indices, nonoverlap_indices)]
    non_ci_low = aggregate.ci_low[np.ix_(nonoverlap_indices, nonoverlap_indices)]
    non_ci_high = aggregate.ci_high[np.ix_(nonoverlap_indices, nonoverlap_indices)]
    write_matrix_csv(non_dir / "aggregate_matrix.csv", nonoverlap_labels, non_aggregate, row_header="window")
    write_matrix_csv(non_dir / "support_matrix.csv", nonoverlap_labels, non_support, row_header="window")
    write_matrix_csv(non_dir / "available_matrix.csv", nonoverlap_labels, non_available, row_header="window")
    write_matrix_csv(
        non_dir / "support_required_matrix.csv",
        nonoverlap_labels,
        non_required,
        row_header="window",
    )
    write_matrix_csv(non_dir / "ci_low_matrix.csv", nonoverlap_labels, non_ci_low, row_header="window")
    write_matrix_csv(non_dir / "ci_high_matrix.csv", nonoverlap_labels, non_ci_high, row_header="window")
    if make_plots:
        plot_matrix(
            non_dir / "aggregate_heatmap.png",
            nonoverlap_labels,
            non_aggregate,
            f"{channel}: within-frame non-overlap ACF control",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            colorbar_label="ACF Pearson similarity",
            insufficient_mask=~non_sufficient,
        )
        plot_matrix(
            non_dir / "support_heatmap.png",
            nonoverlap_labels,
            non_support,
            f"{channel}: within-frame non-overlap scan support",
            vmin=0.0,
            vmax=float(max(len(within.scan_labels), 1)),
            cmap="cividis",
            colorbar_label="scans",
            integer_annotations=True,
        )
        plot_matrix(
            non_dir / "ci_low_heatmap.png",
            nonoverlap_labels,
            non_ci_low,
            f"{channel}: within-frame non-overlap 95% CI low",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            colorbar_label="ACF Pearson similarity",
            insufficient_mask=~non_sufficient,
        )
        plot_matrix(
            non_dir / "ci_high_heatmap.png",
            nonoverlap_labels,
            non_ci_high,
            f"{channel}: within-frame non-overlap 95% CI high",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            colorbar_label="ACF Pearson similarity",
            insufficient_mask=~non_sufficient,
        )

    pressure_availability = np.any(
        np.isfinite(within.by_scan_pressure), axis=(-2, -1), keepdims=True
    )
    by_pressure_aggregate = uw.aggregate_scan_matrices(
        within.by_scan_pressure,
        pressure_availability,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )
    by_pressure_dir = root / "by_pressure"
    for pressure_index, pressure in enumerate(within.pressure_values):
        matrix = by_pressure_aggregate.median[pressure_index]
        support = by_pressure_aggregate.support[pressure_index]
        available = by_pressure_aggregate.available[pressure_index]
        required = by_pressure_aggregate.support_required[pressure_index]
        sufficient = by_pressure_aggregate.sufficient_support[pressure_index]
        ci_low = by_pressure_aggregate.ci_low[pressure_index]
        ci_high = by_pressure_aggregate.ci_high[pressure_index]
        insufficient = ~sufficient
        stem = f"{pressure:g}GPa"
        write_matrix_csv(by_pressure_dir / "matrices" / f"{stem}.csv", labels, matrix, row_header="window")
        write_matrix_csv(
            by_pressure_dir / "support_maps" / f"{stem}_support.csv",
            labels,
            support,
            row_header="window",
        )
        write_matrix_csv(
            by_pressure_dir / "support_maps" / f"{stem}_available.csv",
            labels,
            available,
            row_header="window",
        )
        write_matrix_csv(
            by_pressure_dir / "support_maps" / f"{stem}_support_required.csv",
            labels,
            required,
            row_header="window",
        )
        write_matrix_csv(
            by_pressure_dir / "confidence_intervals" / f"{stem}_ci_low.csv",
            labels,
            ci_low,
            row_header="window",
        )
        write_matrix_csv(
            by_pressure_dir / "confidence_intervals" / f"{stem}_ci_high.csv",
            labels,
            ci_high,
            row_header="window",
        )
        if make_plots:
            plot_matrix(
                by_pressure_dir / "heatmaps" / f"{stem}.png",
                labels,
                matrix,
                f"{channel}: within-frame median at {pressure:g} GPa",
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
                colorbar_label="ACF Pearson similarity",
                insufficient_mask=insufficient,
            )
            plot_matrix(
                by_pressure_dir / "support_maps" / f"{stem}_support.png",
                labels,
                support,
                f"{channel}: within-frame scan support at {pressure:g} GPa",
                vmin=0.0,
                vmax=float(max(len(within.scan_labels), 1)),
                cmap="cividis",
                colorbar_label="scans",
                integer_annotations=True,
            )
            plot_matrix(
                by_pressure_dir / "confidence_intervals" / f"{stem}_ci_low.png",
                labels,
                ci_low,
                f"{channel}: within-frame 95% CI low at {pressure:g} GPa",
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
                colorbar_label="ACF Pearson similarity",
                insufficient_mask=insufficient,
            )
            plot_matrix(
                by_pressure_dir / "confidence_intervals" / f"{stem}_ci_high.png",
                labels,
                ci_high,
                f"{channel}: within-frame 95% CI high at {pressure:g} GPa",
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
                colorbar_label="ACF Pearson similarity",
                insufficient_mask=insufficient,
            )

    per_frame_dir = root / "per_frame_matrices"
    for index, (frame, scan, pressure) in enumerate(
        zip(frame_ids, frame_scans, frame_pressures, strict=True)
    ):
        stem = f"frame_{int(frame):04d}_{safe_name(scan)}_{float(pressure):g}GPa"
        write_matrix_csv(per_frame_dir / f"{stem}.csv", labels, within.by_frame[index], row_header="window")

    pair_rows: list[dict[str, Any]] = []
    for left in range(len(labels)):
        for right in range(left):
            overlap = max(
                0.0,
                window_spec.width_deg
                - abs(window_spec.starts_deg[left] - window_spec.starts_deg[right]),
            )
            pair_rows.append(
                {
                    "channel": channel,
                    "window_a": labels[left],
                    "window_b": labels[right],
                    "overlap_deg": float(overlap),
                    "is_nonoverlap_control_pair": int(
                        left in set(nonoverlap_indices.tolist())
                        and right in set(nonoverlap_indices.tolist())
                    ),
                    "median_similarity": float(aggregate.median[left, right]),
                    "ci_low": float(aggregate.ci_low[left, right]),
                    "ci_high": float(aggregate.ci_high[left, right]),
                    "scan_support": int(aggregate.support[left, right]),
                    "support_required": int(aggregate.support_required[left, right]),
                    "sufficient_support": int(aggregate.sufficient_support[left, right]),
                }
            )
    write_rows_csv(root / "window_pair_summary.csv", pair_rows)
    np.savez_compressed(
        root / "within_frame_matrices.npz",
        frame_ids=np.asarray(frame_ids, dtype=int),
        frame_scans=np.asarray(frame_scans),
        frame_pressure_gpa=np.asarray(frame_pressures, dtype=float),
        pressure_gpa=within.pressure_values,
        scan_names=np.asarray(within.scan_labels),
        window_starts_deg=window_spec.starts_deg,
        window_ends_deg=window_spec.ends_deg,
        nonoverlap_indices=nonoverlap_indices,
        matrices_by_frame=within.by_frame,
        matrices_by_scan=within.by_scan,
        matrices_by_scan_pressure=within.by_scan_pressure,
        aggregate=aggregate.median,
        aggregate_ci_low=aggregate.ci_low,
        aggregate_ci_high=aggregate.ci_high,
        support=aggregate.support,
        available=aggregate.available,
        support_required=aggregate.support_required,
        sufficient_support=aggregate.sufficient_support,
        aggregate_by_pressure=by_pressure_aggregate.median,
        aggregate_by_pressure_ci_low=by_pressure_aggregate.ci_low,
        aggregate_by_pressure_ci_high=by_pressure_aggregate.ci_high,
        support_by_pressure=by_pressure_aggregate.support,
        available_by_pressure=by_pressure_aggregate.available,
        support_required_by_pressure=by_pressure_aggregate.support_required,
        sufficient_support_by_pressure=by_pressure_aggregate.sufficient_support,
        nonoverlap_aggregate=non_aggregate,
        nonoverlap_ci_low=non_ci_low,
        nonoverlap_ci_high=non_ci_high,
        nonoverlap_support=non_support,
        nonoverlap_available=non_available,
        nonoverlap_support_required=non_required,
        nonoverlap_sufficient_support=non_sufficient,
    )
    lower = np.tril_indices(len(labels), k=-1)
    non_lower = np.tril_indices(len(nonoverlap_labels), k=-1)
    overlap_values = aggregate.median[lower]
    non_values = non_aggregate[non_lower]
    return {
        "windows": len(labels),
        "nonoverlap_windows": len(nonoverlap_labels),
        "all_window_pair_median": _finite_median(overlap_values),
        "nonoverlap_pair_median": _finite_median(non_values),
        "finite_all_window_pairs": int(np.count_nonzero(np.isfinite(overlap_values))),
        "finite_nonoverlap_pairs": int(np.count_nonzero(np.isfinite(non_values))),
    }
