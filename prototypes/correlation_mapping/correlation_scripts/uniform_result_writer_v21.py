#!/usr/bin/env python3
"""Versioned result writer for uniform-correlation-v2.1 segments.

Across-frame and within-frame serialization is intentionally delegated to the
unchanged v2 writer.  Per-peak matrices use the same writer contract because a
``SegmentedTrack`` deliberately duck-types ``RadialTrack``; this module adds
the edge/ambiguity/lineage/quarantine/selection audits required by v2.1.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from uniform_correlation_io import write_rows_csv
from uniform_result_writer import (
    write_across_results,
    write_per_peak_results as _write_per_peak_results_v2,
    write_within_results,
)


LINK_FIELDS = [
    "channel",
    "edge_id",
    "first_consensus_id",
    "second_consensus_id",
    "first_pressure_GPa",
    "second_pressure_GPa",
    "first_pressure_index",
    "second_pressure_index",
    "first_q_A^-1",
    "second_q_A^-1",
    "forward_evaluated",
    "backward_evaluated",
    "forward_admissible",
    "backward_admissible",
    "within_original_gate",
    "forward_matched",
    "backward_matched",
    "forward_cost",
    "backward_cost",
    "forward_source_margin",
    "forward_target_margin",
    "backward_source_margin",
    "backward_target_margin",
    "missing_pressure_levels",
    "mutual",
    "endpoint_quarantined",
    "crosses_quarantined_boundary",
    "order_crossing",
    "accepted",
    "cut_reason",
]
AMBIGUITY_FIELDS = [
    "channel",
    "event_id",
    "event_type",
    "pressure_indices",
    "node_ids",
    "edge_ids",
    "reason",
]
GAP_CUT_FIELDS = [
    "channel",
    "direction",
    "source_consensus_id",
    "source_pressure_index",
    "first_unreachable_pressure_index",
    "missing_pressure_levels",
    "reason",
]
LINEAGE_FIELDS = [
    "channel",
    "parent_track_id",
    "parent_node_ids",
    "parent_had_cuts",
    "segment_id",
    "segment_index",
    "segment_count",
    "official",
    "status",
    "pressure_min_GPa",
    "pressure_max_GPa",
    "pressure_nodes",
    "minimum_pressure_nodes",
    "node_ids",
    "boundary_unknown_pressure_indices",
    "cut_event_ids",
]
QUARANTINE_FIELDS = [
    "channel",
    "consensus_id",
    "pressure_GPa",
    "pressure_index",
    "q_A^-1",
    "fwhm_q_A^-1",
    "relative_area",
    "scan_support",
    "reason",
]
SELECTION_FIELDS = [
    "channel",
    "consensus_id",
    "pressure_GPa",
    "pressure_index",
    "q_A^-1",
    "fwhm_q_A^-1",
    "relative_area",
    "scan_support",
    "reliable",
    "selection_status",
    "official_segment_ids",
    "all_segment_ids",
    "all_reliable_nodes",
    "official_retained_nodes",
    "retained_fraction",
    "pressure_available_reliable_nodes",
    "pressure_official_retained_nodes",
    "pressure_retained_fraction",
]


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list, set)):
        return ";".join(str(_plain(item)) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _row(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        source = asdict(value)
    elif isinstance(value, Mapping):
        source = dict(value)
    elif hasattr(value, "__dict__"):
        source = dict(vars(value))
    else:
        raise TypeError(f"cannot serialize audit record of type {type(value)!r}")
    return {str(key): _plain(item) for key, item in source.items()}


def _fieldnames(required: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result = list(required)
    seen = set(result)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def _write_audit(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    required_fields: Sequence[str],
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    write_rows_csv(path, materialized, fieldnames=_fieldnames(required_fields, materialized))
    return materialized


def _consensus_lookup(analysis: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for values in analysis.consensus_by_pressure.values():
        for node in values:
            result[node.consensus_id] = node
    return result


def _selection_rows(analysis: Any, channel: str, quarantined: set[str]) -> list[dict[str, Any]]:
    membership: dict[str, list[str]] = {}
    official_membership: dict[str, list[str]] = {}
    for segment in analysis.tracks:
        for node in segment.nodes:
            membership.setdefault(node.consensus_id, []).append(segment.track_id)
            if segment.official:
                official_membership.setdefault(node.consensus_id, []).append(segment.track_id)
    duplicates = {
        consensus_id: sorted(segment_ids)
        for consensus_id, segment_ids in membership.items()
        if len(segment_ids) > 1
    }
    if duplicates:
        raise ValueError(
            "one consensus node belongs to multiple v2.1 segments: "
            f"{list(duplicates.items())[:5]}"
        )
    rows: list[dict[str, Any]] = []
    for pressure in sorted(analysis.consensus_by_pressure):
        for node in sorted(
            analysis.consensus_by_pressure[pressure], key=lambda item: (item.q, item.consensus_id)
        ):
            official_ids = sorted(official_membership.get(node.consensus_id, ()))
            all_ids = sorted(membership.get(node.consensus_id, ()))
            if node.consensus_id in quarantined:
                status = "quarantined_unknown"
            elif official_ids:
                status = "official_segment_retained"
            elif all_ids:
                status = "nonofficial_segment"
            elif not node.reliable:
                status = "unreliable_consensus"
            else:
                status = "unsegmented_reliable"
            rows.append(
                {
                    "channel": channel,
                    "consensus_id": node.consensus_id,
                    "pressure_GPa": node.pressure,
                    "pressure_index": node.pressure_index,
                    "q_A^-1": node.q,
                    "fwhm_q_A^-1": node.fwhm_q,
                    "relative_area": node.relative_area,
                    "scan_support": node.support,
                    "reliable": int(node.reliable),
                    "selection_status": status,
                    "official_segment_ids": ";".join(official_ids),
                    "all_segment_ids": ";".join(all_ids),
                }
            )
    all_reliable = sum(int(row["reliable"]) == 1 for row in rows)
    official_retained = sum(
        row["selection_status"] == "official_segment_retained" for row in rows
    )
    global_fraction = official_retained / all_reliable if all_reliable else math.nan
    pressure_counts: dict[int, tuple[int, int]] = {}
    for row in rows:
        pressure_index = int(row["pressure_index"])
        available, retained = pressure_counts.get(pressure_index, (0, 0))
        if int(row["reliable"]) == 1:
            available += 1
        if row["selection_status"] == "official_segment_retained":
            retained += 1
        pressure_counts[pressure_index] = (available, retained)
    for row in rows:
        available, retained = pressure_counts[int(row["pressure_index"])]
        row.update(
            {
                "all_reliable_nodes": all_reliable,
                "official_retained_nodes": official_retained,
                "retained_fraction": global_fraction,
                "pressure_available_reliable_nodes": available,
                "pressure_official_retained_nodes": retained,
                "pressure_retained_fraction": retained / available if available else math.nan,
            }
        )
    if not rows:
        rows.append(
            {
                "channel": channel,
                "consensus_id": "",
                "pressure_GPa": math.nan,
                "pressure_index": "",
                "q_A^-1": math.nan,
                "fwhm_q_A^-1": math.nan,
                "relative_area": math.nan,
                "scan_support": 0,
                "reliable": 0,
                "selection_status": "no_consensus_nodes",
                "official_segment_ids": "",
                "all_segment_ids": "",
                "all_reliable_nodes": 0,
                "official_retained_nodes": 0,
                "retained_fraction": math.nan,
                "pressure_available_reliable_nodes": 0,
                "pressure_official_retained_nodes": 0,
                "pressure_retained_fraction": math.nan,
            }
        )
    return rows


def _finite_summary(values: Sequence[float]) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {name: math.nan for name in ("minimum", "q25", "median", "q75", "maximum")}
    return {
        "minimum": float(np.min(finite)),
        "q25": float(np.percentile(finite, 25.0)),
        "median": float(np.median(finite)),
        "q75": float(np.percentile(finite, 75.0)),
        "maximum": float(np.max(finite)),
    }


def _selection_summary_rows(rows: Sequence[Mapping[str, Any]], channel: str) -> list[dict[str, Any]]:
    node_rows = [row for row in rows if str(row.get("consensus_id", "")).strip()]
    groups = {
        "all_reliable": [row for row in node_rows if int(row["reliable"]) == 1],
        "official_segment_retained": [
            row for row in node_rows if row["selection_status"] == "official_segment_retained"
        ],
        "not_officially_retained": [
            row for row in node_rows if row["selection_status"] != "official_segment_retained"
        ],
        "quarantined_unknown": [
            row for row in node_rows if row["selection_status"] == "quarantined_unknown"
        ],
    }
    all_reliable = sum(int(row["reliable"]) == 1 for row in node_rows)
    official_retained = sum(
        row["selection_status"] == "official_segment_retained" for row in node_rows
    )
    retained_fraction = official_retained / all_reliable if all_reliable else math.nan
    result: list[dict[str, Any]] = [
        {
            "channel": channel,
            "row_kind": "overall_retention",
            "selection_group": "retained_fraction",
            "metric": "retention",
            "node_count": all_reliable,
            "all_reliable_nodes": all_reliable,
            "official_retained_nodes": official_retained,
            "retained_fraction": retained_fraction,
        }
    ]
    for group, selected in groups.items():
        for metric in ("q_A^-1", "fwhm_q_A^-1", "relative_area", "scan_support"):
            stats = _finite_summary([float(row[metric]) for row in selected])
            result.append(
                {
                    "channel": channel,
                    "row_kind": "distribution_quantiles",
                    "selection_group": group,
                    "metric": metric,
                    "node_count": len(selected),
                    "all_reliable_nodes": all_reliable,
                    "official_retained_nodes": official_retained,
                    "retained_fraction": retained_fraction,
                    **stats,
                }
            )
    pressure_values: list[tuple[int, float]] = []
    for row in node_rows:
        try:
            item = (int(row["pressure_index"]), float(row["pressure_GPa"]))
        except (TypeError, ValueError):
            continue
        if np.isfinite(item[1]) and item not in pressure_values:
            pressure_values.append(item)
    pressure_values.sort()
    for pressure_index, pressure in pressure_values:
        selected = [row for row in node_rows if int(row["pressure_index"]) == pressure_index]
        available = sum(int(row["reliable"]) == 1 for row in selected)
        retained = sum(
            row["selection_status"] == "official_segment_retained" for row in selected
        )
        result.append(
            {
                "channel": channel,
                "row_kind": "pressure_retention",
                "selection_group": "all_reliable",
                "metric": "retention",
                "pressure_index": pressure_index,
                "pressure_GPa": pressure,
                "node_count": available,
                "all_reliable_nodes": all_reliable,
                "official_retained_nodes": official_retained,
                "retained_fraction": retained_fraction,
                "pressure_available_reliable_nodes": available,
                "pressure_official_retained_nodes": retained,
                "pressure_retained_fraction": retained / available if available else math.nan,
            }
        )
    return result


def _write_v21_audits(
    root: Path,
    channel: str,
    analysis: Any,
    tracking_result: Any,
) -> dict[str, int]:
    link_rows: list[dict[str, Any]] = []
    for item in tracking_result.link_evidence:
        raw = _row(item)
        raw["first_q_A^-1"] = raw.pop("first_q_A_inv")
        raw["second_q_A^-1"] = raw.pop("second_q_A_inv")
        normalized = {field: raw.get(field) for field in LINK_FIELDS if field != "channel"}
        link_rows.append({"channel": channel, **normalized})
    _write_audit(root / "link_evidence.csv", link_rows, LINK_FIELDS)

    ambiguity_rows = [
        {"channel": channel, **_row(item)} for item in tracking_result.ambiguity_events
    ]
    _write_audit(root / "ambiguity_events.csv", ambiguity_rows, AMBIGUITY_FIELDS)

    gap_cut_rows = [
        {"channel": channel, **_row(item)}
        for item in getattr(tracking_result, "gap_cuts", ())
    ]
    _write_audit(root / "gap_cuts.csv", gap_cut_rows, GAP_CUT_FIELDS)

    parents = {item.parent_track_id: item for item in tracking_result.parents}
    lineage_rows: list[dict[str, Any]] = []
    for segment in sorted(analysis.tracks, key=lambda item: item.track_id):
        parent = parents.get(segment.parent_track_id)
        lineage_rows.append(
            {
                "channel": channel,
                "parent_track_id": segment.parent_track_id,
                "parent_node_ids": _plain(parent.node_ids) if parent is not None else "",
                "parent_had_cuts": int(parent.had_cuts) if parent is not None else "",
                "segment_id": segment.track_id,
                "segment_index": segment.segment_index,
                "segment_count": segment.segment_count,
                "official": int(segment.official),
                "status": segment.status,
                "pressure_min_GPa": segment.pressure_min,
                "pressure_max_GPa": segment.pressure_max,
                "pressure_nodes": len(segment.nodes),
                "minimum_pressure_nodes": segment.minimum_pressure_support,
                "node_ids": _plain(tuple(node.consensus_id for node in segment.nodes)),
                "boundary_unknown_pressure_indices": _plain(
                    segment.boundary_unknown_pressure_indices
                ),
                "cut_event_ids": _plain(segment.cut_event_ids),
            }
        )
    _write_audit(root / "segment_lineage.csv", lineage_rows, LINEAGE_FIELDS)

    lookup = _consensus_lookup(analysis)
    quarantine_rows: list[dict[str, Any]] = []
    for consensus_id in sorted(tracking_result.quarantined_node_ids):
        node = lookup.get(consensus_id)
        quarantine_rows.append(
            {
                "channel": channel,
                "consensus_id": consensus_id,
                "pressure_GPa": getattr(node, "pressure", math.nan),
                "pressure_index": getattr(node, "pressure_index", ""),
                "q_A^-1": getattr(node, "q", math.nan),
                "fwhm_q_A^-1": getattr(node, "fwhm_q", math.nan),
                "relative_area": getattr(node, "relative_area", math.nan),
                "scan_support": getattr(node, "support", ""),
                "reason": "low_margin_competition_identity_unknown",
            }
        )
    _write_audit(root / "quarantined_nodes.csv", quarantine_rows, QUARANTINE_FIELDS)

    selection_rows = _selection_rows(
        analysis, channel, set(tracking_result.quarantined_node_ids)
    )
    _write_audit(root / "selection_audit.csv", selection_rows, SELECTION_FIELDS)
    selection_summary = _selection_summary_rows(selection_rows, channel)
    _write_audit(
        root / "selection_audit_summary.csv",
        selection_summary,
        [
            "channel",
            "row_kind",
            "selection_group",
            "metric",
            "pressure_index",
            "pressure_GPa",
            "node_count",
            "all_reliable_nodes",
            "official_retained_nodes",
            "retained_fraction",
            "pressure_available_reliable_nodes",
            "pressure_official_retained_nodes",
            "pressure_retained_fraction",
            "minimum",
            "q25",
            "median",
            "q75",
            "maximum",
        ],
    )
    official_cut_reasons = {
        "cut_one_way",
        "cut_low_margin",
        "cut_order_crossing",
        "cut_missing_too_long",
        "cut_outside_gate",
    }
    cut_links = sum(
        int(
            any(
                reason in official_cut_reasons
                for reason in str(row.get("cut_reason", "")).split(";")
            )
        )
        for row in link_rows
    )
    nonselected = sum(
        int(not bool(row.get("accepted")) and not str(row.get("cut_reason", "")).strip())
        for row in link_rows
    )
    return {
        "candidate_links": len(link_rows),
        "accepted_links": sum(int(bool(row.get("accepted"))) for row in link_rows),
        "cut_links": cut_links,
        "nonselected_hungarian_candidates": nonselected,
        "ambiguity_events": len(ambiguity_rows),
        "missing_too_long_gap_cuts": len(gap_cut_rows),
        "quarantined_nodes": len(quarantine_rows),
        "selection_nodes": len(selection_rows),
    }


def write_per_peak_results(
    channel_root: Path,
    channel: str,
    analysis: Any,
    frame_peaks: Sequence[Any],
    *,
    scans: Sequence[str],
    pressures: Sequence[float],
    make_plots: bool,
    tracking_result: Any | None = None,
) -> dict[str, Any]:
    """Write v2-compatible matrices plus the mandatory v2.1 audit tables."""

    result = _write_per_peak_results_v2(
        channel_root,
        channel,
        analysis,
        frame_peaks,
        scans=scans,
        pressures=pressures,
        make_plots=make_plots,
    )
    evidence = tracking_result or getattr(analysis, "tracking_result", None)
    if evidence is None:
        raise ValueError("v2.1 result writing requires SegmentedTrackingResult evidence")
    audit_metrics = _write_v21_audits(channel_root / "per_peak", channel, analysis, evidence)
    result.update(audit_metrics)
    result["all_segments"] = result.pop("all_radial_tracks")
    result["official_segments"] = result.pop("official_radial_tracks")
    # Compatibility aliases keep the generic workbook reader functional while
    # making the v2.1 scientific unit explicit in the run manifest.
    result["all_radial_tracks"] = result["all_segments"]
    result["official_radial_tracks"] = result["official_segments"]
    return result


__all__ = [
    "write_across_results",
    "write_per_peak_results",
    "write_within_results",
]
