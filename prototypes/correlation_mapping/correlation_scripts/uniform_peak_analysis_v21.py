#!/usr/bin/env python3
"""Per-peak analysis orchestration for edge-segmented v2.1 tracking.

Peak detection, pressure consensus, observation assignment, similarity
formulas, support masking, bootstrap intervals, and near/far summaries are the
unchanged v2 implementations.  Only the consensus-to-trajectory step is
replaced by audited edge-level segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import uniform_peak_core as up
from uniform_peak_tracking_v21 import (
    SegmentedTrackingConfig,
    SegmentedTrackingResult,
    segment_consensus_bidirectional,
)


@dataclass(frozen=True)
class PerPeakAnalysisV21:
    """v2-compatible analysis product with complete segmented-link evidence."""

    consensus_by_pressure: Mapping[float, tuple[up.PressureConsensus, ...]]
    tracks: tuple[Any, ...]
    assignments: Mapping[str, Mapping[tuple[str, float], up.AssignedObservation]]
    correlations: Mapping[str, up.CorrelationMatrices]
    near_far: Mapping[str, Mapping[str, Mapping[str, float | int | str]]]
    tracking_result: SegmentedTrackingResult


def _outside_segment(track: Any, pressure: float) -> bool:
    return pressure < track.pressure_min or pressure > track.pressure_max


def _blank_nonofficial_assignments(
    tracks: Sequence[Any],
    scans: Sequence[str],
    pressure_levels: Sequence[float],
) -> dict[str, dict[tuple[str, float], up.AssignedObservation]]:
    result: dict[str, dict[tuple[str, float], up.AssignedObservation]] = {}
    for track in tracks:
        values: dict[tuple[str, float], up.AssignedObservation] = {}
        boundary_unknown = set(getattr(track, "boundary_unknown_pressure_indices", ()))
        for scan_value in scans:
            scan = str(scan_value)
            for pressure_index, pressure_value in enumerate(pressure_levels):
                pressure = float(pressure_value)
                outside = _outside_segment(track, pressure)
                boundary = pressure_index in boundary_unknown
                values[(scan, pressure)] = up.AssignedObservation(
                    track_id=track.track_id,
                    scan=scan,
                    pressure=pressure,
                    frame=None,
                    state="out_of_range" if outside else "unknown",
                    reason=(
                        "outside_supported_segment_range"
                        if outside
                        else (
                            "quarantined_ambiguity_boundary"
                            if boundary
                            else "nonofficial_segment_not_assigned"
                        )
                    ),
                )
        result[track.track_id] = values
    return result


def _apply_boundary_unknowns(
    tracks: Sequence[Any],
    assignments: dict[str, dict[tuple[str, float], up.AssignedObservation]],
    scans: Sequence[str],
    pressure_levels: Sequence[float],
) -> None:
    """Prevent a quarantined cut boundary from being reused as a measurement."""

    pressure_values = tuple(float(value) for value in pressure_levels)
    for track in tracks:
        for pressure_index in getattr(track, "boundary_unknown_pressure_indices", ()):
            if pressure_index < 0 or pressure_index >= len(pressure_values):
                raise ValueError(
                    f"segment {track.track_id} has invalid boundary pressure index {pressure_index}"
                )
            pressure = pressure_values[pressure_index]
            for scan_value in scans:
                scan = str(scan_value)
                key = (scan, pressure)
                previous = assignments[track.track_id][key]
                assignments[track.track_id][key] = replace(
                    previous,
                    frame=None,
                    state="unknown",
                    reason="quarantined_ambiguity_boundary",
                    q=float("nan"),
                    fwhm_q=float("nan"),
                    relative_area=float("nan"),
                    peak_id=None,
                )


def analyze_per_peak_v21(
    frame_peaks: Sequence[up.FramePeaks],
    pressure_levels: Sequence[float],
    scans: Sequence[str],
    peak_config: up.UniformPeakConfig,
    tracking_config: SegmentedTrackingConfig,
    *,
    bootstrap_iterations: int | None = None,
    seed: int | None = None,
    official_only: bool = True,
) -> PerPeakAnalysisV21:
    """Run unchanged v2 analysis around the v2.1 segmented tracking step."""

    consensus = up.build_pressure_consensus(frame_peaks, scans, pressure_levels, peak_config)
    tracking_result = segment_consensus_bidirectional(
        consensus,
        pressure_levels,
        tracking_config,
    )
    tracks = tuple(tracking_result.segments)
    assignment_tracks = (
        tuple(track for track in tracks if track.official) if official_only else tracks
    )
    assignments = up.assign_track_observations(
        assignment_tracks,
        frame_peaks,
        scans,
        pressure_levels,
        peak_config,
    )
    _apply_boundary_unknowns(
        assignment_tracks,
        assignments,
        scans,
        pressure_levels,
    )
    if official_only:
        assignments.update(
            _blank_nonofficial_assignments(
                [track for track in tracks if not track.official],
                scans,
                pressure_levels,
            )
        )

    iterations = (
        peak_config.bootstrap_iterations
        if bootstrap_iterations is None
        else int(bootstrap_iterations)
    )
    random_seed = peak_config.random_seed if seed is None else int(seed)
    correlations: dict[str, up.CorrelationMatrices] = {}
    summaries: dict[str, dict[str, Mapping[str, float | int | str]]] = {}
    for track in tracks:
        if official_only and not track.official:
            continue
        matrices = up.compute_track_correlations(
            assignments[track.track_id],
            scans,
            pressure_levels,
            bootstrap_iterations=iterations,
            seed=random_seed,
            ci_percentiles=peak_config.ci_percentiles,
        )
        correlations[track.track_id] = matrices
        summaries[track.track_id] = {
            "area": up.bootstrap_near_far_summary(
                matrices.area_by_scan,
                pressure_levels,
                aggregate="median",
                point_matrix=matrices.area,
                bootstrap_iterations=iterations,
                seed=random_seed,
                minimum_distinct_gaps=peak_config.minimum_distinct_pressure_gaps,
                minimum_group_values=peak_config.minimum_supported_group_values,
                near_gap_quantile=peak_config.near_gap_quantile,
                far_gap_quantile=peak_config.far_gap_quantile,
                ci_percentiles=peak_config.ci_percentiles,
            ),
            "location": up.bootstrap_near_far_summary(
                matrices.location_by_scan,
                pressure_levels,
                aggregate="median",
                point_matrix=matrices.location,
                bootstrap_iterations=iterations,
                seed=random_seed,
                minimum_distinct_gaps=peak_config.minimum_distinct_pressure_gaps,
                minimum_group_values=peak_config.minimum_supported_group_values,
                near_gap_quantile=peak_config.near_gap_quantile,
                far_gap_quantile=peak_config.far_gap_quantile,
                ci_percentiles=peak_config.ci_percentiles,
            ),
            "presence": up.bootstrap_near_far_summary(
                matrices.presence_by_scan,
                pressure_levels,
                aggregate="mean",
                point_matrix=matrices.presence,
                bootstrap_iterations=iterations,
                seed=random_seed,
                minimum_distinct_gaps=peak_config.minimum_distinct_pressure_gaps,
                minimum_group_values=peak_config.minimum_supported_group_values,
                near_gap_quantile=peak_config.near_gap_quantile,
                far_gap_quantile=peak_config.far_gap_quantile,
                ci_percentiles=peak_config.ci_percentiles,
            ),
        }
    return PerPeakAnalysisV21(
        consensus_by_pressure={key: tuple(value) for key, value in consensus.items()},
        tracks=tracks,
        assignments=assignments,
        correlations=correlations,
        near_far=summaries,
        tracking_result=tracking_result,
    )


__all__ = ["PerPeakAnalysisV21", "analyze_per_peak_v21"]
