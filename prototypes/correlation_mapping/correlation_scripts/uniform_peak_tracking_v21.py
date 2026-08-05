#!/usr/bin/env python3
"""Edge-level segmented radial-peak tracking for ``uniform-correlation-v2.1``.

The frozen v2 peak detector, fitter, pressure consensus, and correlation
formulas remain untouched.  This module replaces only the trajectory-level
ambiguity propagation rule.  Instead of invalidating an entire connected
trajectory when one link is uncertain, it:

* records every forward/backward Hungarian edge evaluation;
* accepts only the exact same edge in both directions;
* quarantines nodes involved in a frozen low-margin competition;
* cuts q-order-crossing edges that one-dimensional radial data cannot identify;
* builds deterministic, non-overlapping clean trajectory segments; and
* records cut boundaries so downstream assignment can emit ``unknown``/NaN.

No material-specific q position, pressure range, expected slope, or hand-made
track registry is consumed here.  All numeric settings are width- and
pressure-sampling-relative and must be frozen by the calling profile binder.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

import uniform_peak_core as v2


ALGORITHM_VERSION = "2.1.0"
_BLOCKED_COST = 1.0e12


@dataclass(frozen=True)
class SegmentedTrackingConfig:
    """Frozen v2.1 trajectory settings.

    The numeric defaults are exactly the v2 values.  ``unmatched_cost`` makes
    the previous Hungarian dummy cost explicit and auditable; it does not
    change the scientific decision rule.
    """

    algorithm_version: str = ALGORITHM_VERSION
    max_missing_pressure_levels: int = 2
    track_gate_factor: float = 1.5
    track_width_cost_weight: float = 0.1
    ambiguous_cost_margin: float = 0.25
    unmatched_cost: float = 10.0
    decision_unit: str = "candidate_edge"
    require_bidirectional_same_edge: bool = True
    require_margin_at_both_endpoints: bool = True
    reject_q_order_crossing: bool = True
    low_margin_competitor_state: str = "unknown_quarantined"
    cut_reasons: tuple[str, ...] = (
        "cut_one_way",
        "cut_low_margin",
        "cut_order_crossing",
        "cut_missing_too_long",
        "cut_outside_gate",
    )
    allow_interpolation_across_cut: bool = False

    def __post_init__(self) -> None:
        if self.algorithm_version != ALGORITHM_VERSION:
            raise ValueError(
                f"This module implements {ALGORITHM_VERSION!r}, not "
                f"{self.algorithm_version!r}"
            )
        numeric_positive = (
            self.track_gate_factor,
            self.ambiguous_cost_margin,
            self.unmatched_cost,
        )
        if not all(np.isfinite(value) and value > 0 for value in numeric_positive):
            raise ValueError("gate, ambiguity margin, and unmatched cost must be positive")
        if (
            not np.isfinite(self.track_width_cost_weight)
            or self.track_width_cost_weight < 0
        ):
            raise ValueError("track_width_cost_weight must be finite and non-negative")
        if self.max_missing_pressure_levels < 0:
            raise ValueError("max_missing_pressure_levels cannot be negative")
        expected = {
            "decision_unit": "candidate_edge",
            "low_margin_competitor_state": "unknown_quarantined",
        }
        for field_name, expected_value in expected.items():
            actual = getattr(self, field_name)
            if actual != expected_value:
                raise ValueError(
                    f"{field_name}={actual!r} is unsupported; expected "
                    f"{expected_value!r}"
                )
        if not (
            self.require_bidirectional_same_edge
            and self.require_margin_at_both_endpoints
            and self.reject_q_order_crossing
        ):
            raise ValueError(
                "v2.1 requires mutual edges, both-endpoint margins, and crossing rejection"
            )
        expected_cut_reasons = (
            "cut_one_way",
            "cut_low_margin",
            "cut_order_crossing",
            "cut_missing_too_long",
            "cut_outside_gate",
        )
        if tuple(self.cut_reasons) != expected_cut_reasons:
            raise ValueError(
                f"cut_reasons={self.cut_reasons!r} is unsupported; expected "
                f"{expected_cut_reasons!r}"
            )
        if self.allow_interpolation_across_cut:
            raise ValueError("v2.1 never interpolates across a cut")


@dataclass(frozen=True)
class PassLink:
    """One path-to-node edge evaluated in one directional tracking pass."""

    direction: str
    source_consensus_id: str
    target_consensus_id: str
    source_pressure: float
    target_pressure: float
    source_pressure_index: int
    target_pressure_index: int
    source_q: float
    target_q: float
    admissible: bool
    matched: bool
    cost: float
    source_margin: float
    target_margin: float
    reason: str

    @property
    def edge_key(self) -> tuple[str, str]:
        return tuple(sorted((self.source_consensus_id, self.target_consensus_id)))


@dataclass(frozen=True)
class CompetitionEvidence:
    """Frozen winner/runner-up ambiguity evidence from one pass."""

    direction: str
    competition_axis: str
    pressure_index: int
    anchor_consensus_id: str
    best_consensus_id: str
    runner_up_consensus_id: str
    best_cost: float
    runner_up_cost: float
    cost_margin: float
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryLinkEvidence:
    """Bidirectionally reconciled evidence for one candidate identity edge."""

    edge_id: str
    first_consensus_id: str
    second_consensus_id: str
    first_pressure_GPa: float
    second_pressure_GPa: float
    first_pressure_index: int
    second_pressure_index: int
    first_q_A_inv: float
    second_q_A_inv: float
    missing_pressure_levels: int
    forward_evaluated: bool
    backward_evaluated: bool
    forward_admissible: bool
    backward_admissible: bool
    within_original_gate: bool
    forward_matched: bool
    backward_matched: bool
    forward_cost: float
    backward_cost: float
    forward_source_margin: float
    forward_target_margin: float
    backward_source_margin: float
    backward_target_margin: float
    mutual: bool
    endpoint_quarantined: bool
    crosses_quarantined_boundary: bool
    order_crossing: bool
    accepted: bool
    cut_reason: str


@dataclass(frozen=True)
class AmbiguityEvent:
    """Auditable local reason why an identity continuation was not used."""

    event_id: str
    event_type: str
    direction: str
    pressure_indices: tuple[int, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    cost_margin: float
    reason: str


@dataclass(frozen=True)
class GapCutEvidence:
    """A path termination caused solely by the frozen missing-level limit."""

    direction: str
    source_consensus_id: str
    source_pressure_index: int
    first_unreachable_pressure_index: int
    missing_pressure_levels: int
    reason: str = "cut_missing_too_long"


@dataclass(frozen=True)
class SegmentedTrack:
    """A clean radial trajectory segment, compatible with v2 track consumers."""

    track_id: str
    parent_track_id: str
    channel: str
    nodes: tuple[v2.TrackNode, ...]
    official: bool
    ambiguous: bool
    minimum_pressure_support: int
    segment_index: int
    segment_count: int
    status: str
    boundary_unknown_pressure_indices: tuple[int, ...] = ()
    cut_event_ids: tuple[str, ...] = ()

    @property
    def pressure_min(self) -> float:
        return min(node.pressure for node in self.nodes)

    @property
    def pressure_max(self) -> float:
        return max(node.pressure for node in self.nodes)


@dataclass(frozen=True)
class ParentTrack:
    """Lineage record for a provisional mutual-edge component."""

    parent_track_id: str
    channel: str
    node_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    had_cuts: bool


@dataclass(frozen=True)
class SegmentTarget:
    """Resolved target state used by downstream per-frame assignment."""

    state: str
    reason: str
    q: float = math.nan
    fwhm_q: float = math.nan


@dataclass(frozen=True)
class SegmentedTrackingResult:
    """Complete v2.1 segmented tracking product and audit trail."""

    segments: tuple[SegmentedTrack, ...]
    pass_links: tuple[PassLink, ...]
    competitions: tuple[CompetitionEvidence, ...]
    link_evidence: tuple[TrajectoryLinkEvidence, ...]
    ambiguity_events: tuple[AmbiguityEvent, ...]
    gap_cuts: tuple[GapCutEvidence, ...]
    quarantined_node_ids: tuple[str, ...]
    parents: tuple[ParentTrack, ...]


@dataclass(frozen=True)
class _PassResult:
    links: tuple[PassLink, ...]
    competitions: tuple[CompetitionEvidence, ...]
    gap_cuts: tuple[GapCutEvidence, ...]
    touched_node_ids: tuple[str, ...]
    quarantined_node_ids: tuple[str, ...]


def _consensus_to_node(item: v2.PressureConsensus) -> v2.TrackNode:
    return v2.TrackNode(
        consensus_id=item.consensus_id,
        pressure=item.pressure,
        pressure_index=item.pressure_index,
        q=item.q,
        fwhm_q=item.fwhm_q,
        relative_area=item.relative_area,
        support=item.support,
        ambiguous=item.ambiguous,
    )


def _trajectory_prediction(
    nodes: Sequence[v2.TrackNode], target_pressure: float
) -> tuple[float, float]:
    last = nodes[-1]
    if len(nodes) < 2:
        return last.q, last.fwhm_q
    previous = nodes[-2]
    pressure_delta = last.pressure - previous.pressure
    if abs(pressure_delta) <= np.finfo(float).eps:
        return last.q, last.fwhm_q
    velocity = (last.q - previous.q) / pressure_delta
    return last.q + velocity * (target_pressure - last.pressure), last.fwhm_q


def _directional_pass(
    consensus_by_pressure: Mapping[float, Sequence[v2.PressureConsensus]],
    pressure_levels: Sequence[float],
    config: SegmentedTrackingConfig,
    *,
    reverse: bool,
) -> _PassResult:
    direction = "high_to_low" if reverse else "low_to_high"
    ordered = list(enumerate(float(value) for value in pressure_levels))
    if reverse:
        ordered.reverse()
    sorted_pressures = np.sort(np.asarray(pressure_levels, dtype=float))
    positive_steps = np.diff(sorted_pressures)
    positive_steps = positive_steps[positive_steps > 0]
    median_pressure_step = float(np.median(positive_steps)) if positive_steps.size else 1.0

    paths: list[list[v2.TrackNode]] = []
    evaluations: list[PassLink] = []
    competitions: list[CompetitionEvidence] = []
    gap_cuts: list[GapCutEvidence] = []
    touched: set[str] = set()
    quarantined: set[str] = set()
    expired_path_endpoints: set[str] = set()

    for level_index, pressure in ordered:
        nodes = [
            _consensus_to_node(item)
            for item in consensus_by_pressure.get(pressure, ())
            if item.reliable
        ]
        nodes.sort(key=lambda node: (node.q, node.consensus_id))
        touched.update(node.consensus_id for node in nodes)

        active: list[list[v2.TrackNode]] = []
        for path in paths:
            missing = abs(level_index - path[-1].pressure_index) - 1
            if missing <= config.max_missing_pressure_levels:
                active.append(path)
            elif path[-1].consensus_id not in expired_path_endpoints:
                expired_path_endpoints.add(path[-1].consensus_id)
                gap_cuts.append(
                    GapCutEvidence(
                        direction=direction,
                        source_consensus_id=path[-1].consensus_id,
                        source_pressure_index=path[-1].pressure_index,
                        first_unreachable_pressure_index=level_index,
                        missing_pressure_levels=missing,
                    )
                )
        if not active:
            paths.extend([[node] for node in nodes])
            continue

        n_path = len(active)
        n_node = len(nodes)
        real_cost = np.full((n_path, n_node), _BLOCKED_COST, dtype=float)
        for path_index, path in enumerate(active):
            prediction, predicted_width = _trajectory_prediction(path, pressure)
            pressure_gap_factor = max(
                1.0,
                abs(pressure - path[-1].pressure) / median_pressure_step,
            )
            for node_index, node in enumerate(nodes):
                gate = (
                    config.track_gate_factor
                    * math.sqrt(predicted_width**2 + node.fwhm_q**2)
                    * pressure_gap_factor
                )
                if not (
                    np.isfinite(gate)
                    and gate > 0
                    and predicted_width > 0
                    and node.fwhm_q > 0
                ):
                    continue
                normalized_dq = abs(node.q - prediction) / gate
                if normalized_dq <= 1.0:
                    width_term = math.log(node.fwhm_q / predicted_width)
                    real_cost[path_index, node_index] = (
                        normalized_dq**2
                        + config.track_width_cost_weight * width_term**2
                    )

        size = n_path + n_node
        cost = np.full((size, size), _BLOCKED_COST, dtype=float)
        cost[:n_path, :n_node] = real_cost
        cost[:n_path, n_node:] = config.unmatched_cost
        cost[n_path:, :n_node] = config.unmatched_cost
        cost[n_path:, n_node:] = 0.0
        rows, columns = linear_sum_assignment(cost)
        matches = {
            (int(row), int(column))
            for row, column in zip(rows, columns)
            if row < n_path
            and column < n_node
            and real_cost[row, column] < config.unmatched_cost
        }

        # Margins are evaluated for the actual globally matched edge, at both
        # endpoints.  The unmatched dummy is an explicit competitor.  Thus a
        # unique real candidate has an auditable finite margin
        # ``unmatched_cost - selected_cost`` rather than a silent NaN.
        source_margins: dict[int, float] = {}
        target_margins: dict[int, float] = {}
        locally_quarantined_node_indices: set[int] = set()
        for path_index, node_index in sorted(matches):
            selected_cost = float(real_cost[path_index, node_index])
            row_alternatives = [config.unmatched_cost]
            for candidate_index in range(n_node):
                if candidate_index == node_index:
                    continue
                candidate_cost = float(real_cost[path_index, candidate_index])
                if candidate_cost < _BLOCKED_COST / 2.0:
                    row_alternatives.append(candidate_cost)
            row_runner_cost = float(min(row_alternatives))
            row_margin = row_runner_cost - selected_cost
            source_margins[path_index] = row_margin

            column_alternatives = [config.unmatched_cost]
            for candidate_index in range(n_path):
                if candidate_index == path_index:
                    continue
                candidate_cost = float(real_cost[candidate_index, node_index])
                if candidate_cost < _BLOCKED_COST / 2.0:
                    column_alternatives.append(candidate_cost)
            column_runner_cost = float(min(column_alternatives))
            column_margin = column_runner_cost - selected_cost
            target_margins[node_index] = column_margin

            if row_margin < config.ambiguous_cost_margin:
                runner_indices = [
                    candidate_index
                    for candidate_index in range(n_node)
                    if candidate_index != node_index
                    and real_cost[path_index, candidate_index]
                    == row_runner_cost
                ]
                current_indices = {node_index, *runner_indices[:1]}
                locally_quarantined_node_indices.update(current_indices)
                node_ids = tuple(
                    sorted(nodes[index].consensus_id for index in current_indices)
                )
                runner_id = (
                    nodes[runner_indices[0]].consensus_id
                    if runner_indices
                    else "UNMATCHED_DUMMY"
                )
                competitions.append(
                    CompetitionEvidence(
                        direction=direction,
                        competition_axis="matched_source_endpoint",
                        pressure_index=level_index,
                        anchor_consensus_id=active[path_index][-1].consensus_id,
                        best_consensus_id=nodes[node_index].consensus_id,
                        runner_up_consensus_id=runner_id,
                        best_cost=selected_cost,
                        runner_up_cost=row_runner_cost,
                        cost_margin=row_margin,
                        node_ids=node_ids,
                    )
                )
            if column_margin < config.ambiguous_cost_margin:
                # The current-pressure target is the locally unresolved node;
                # do not poison the preceding source-pressure nodes.
                locally_quarantined_node_indices.add(node_index)
                runner_indices = [
                    candidate_index
                    for candidate_index in range(n_path)
                    if candidate_index != path_index
                    and real_cost[candidate_index, node_index]
                    == column_runner_cost
                ]
                runner_id = (
                    active[runner_indices[0]][-1].consensus_id
                    if runner_indices
                    else "UNMATCHED_DUMMY"
                )
                competitions.append(
                    CompetitionEvidence(
                        direction=direction,
                        competition_axis="matched_target_endpoint",
                        pressure_index=level_index,
                        anchor_consensus_id=nodes[node_index].consensus_id,
                        best_consensus_id=active[path_index][-1].consensus_id,
                        runner_up_consensus_id=runner_id,
                        best_cost=selected_cost,
                        runner_up_cost=column_runner_cost,
                        cost_margin=column_margin,
                        node_ids=(nodes[node_index].consensus_id,),
                    )
                )

        quarantined.update(
            nodes[index].consensus_id
            for index in locally_quarantined_node_indices
        )

        for path_index, path in enumerate(active):
            source = path[-1]
            for node_index, node in enumerate(nodes):
                candidate_cost = float(real_cost[path_index, node_index])
                admissible = candidate_cost < _BLOCKED_COST / 2.0
                matched = (path_index, node_index) in matches
                evaluations.append(
                    PassLink(
                        direction=direction,
                        source_consensus_id=source.consensus_id,
                        target_consensus_id=node.consensus_id,
                        source_pressure=source.pressure,
                        target_pressure=node.pressure,
                        source_pressure_index=source.pressure_index,
                        target_pressure_index=node.pressure_index,
                        source_q=source.q,
                        target_q=node.q,
                        admissible=admissible,
                        matched=matched,
                        cost=candidate_cost if admissible else math.nan,
                        source_margin=source_margins.get(path_index, math.nan),
                        target_margin=target_margins.get(node_index, math.nan),
                        reason=(
                            "selected_hungarian"
                            if matched
                            else "not_selected_hungarian"
                            if admissible
                            else "cut_outside_gate"
                        ),
                    )
                )

        propagated_node_indices: set[int] = set()
        for path_index, node_index in sorted(matches):
            if node_index in locally_quarantined_node_indices:
                continue
            active[path_index].append(nodes[node_index])
            propagated_node_indices.add(node_index)
        paths.extend(
            [node]
            for node_index, node in enumerate(nodes)
            if node_index not in propagated_node_indices
            and node_index not in locally_quarantined_node_indices
        )

        # Existing active paths are references into ``paths``.  Only new
        # singleton paths were appended, but keep identity de-duplication as a
        # deterministic guard against future path container refactors.
        unique_paths: list[list[v2.TrackNode]] = []
        seen_path_objects: set[int] = set()
        for path in paths:
            identity = id(path)
            if identity not in seen_path_objects:
                seen_path_objects.add(identity)
                unique_paths.append(path)
        paths = unique_paths

    return _PassResult(
        links=tuple(evaluations),
        competitions=tuple(competitions),
        gap_cuts=tuple(gap_cuts),
        touched_node_ids=tuple(sorted(touched)),
        quarantined_node_ids=tuple(sorted(quarantined)),
    )


def _oriented_nodes(
    edge_key: tuple[str, str],
    nodes: Mapping[str, v2.PressureConsensus],
) -> tuple[v2.PressureConsensus, v2.PressureConsensus]:
    first, second = (nodes[edge_key[0]], nodes[edge_key[1]])
    if (second.pressure_index, second.consensus_id) < (
        first.pressure_index,
        first.consensus_id,
    ):
        first, second = second, first
    return first, second


def _pass_edge_lookup(
    links: Sequence[PassLink], direction: str
) -> dict[tuple[str, str], PassLink]:
    result: dict[tuple[str, str], PassLink] = {}
    for link in links:
        if link.direction != direction:
            continue
        existing = result.get(link.edge_key)
        if existing is None:
            result[link.edge_key] = link
            continue
        # The path algorithm should evaluate an endpoint pair once per pass.
        # A duplicate is a programming error because it makes link evidence
        # non-identifiable.
        raise AssertionError(
            f"duplicate {direction} evaluation for edge {link.edge_key}: "
            f"{existing} vs {link}"
        )
    return result


def _interpolated_q(
    first: v2.PressureConsensus,
    second: v2.PressureConsensus,
    pressure: float,
) -> float:
    fraction = (pressure - first.pressure) / (second.pressure - first.pressure)
    return float(first.q + fraction * (second.q - first.q))


def _crossing_mutual_edges(
    mutual_edges: Sequence[tuple[str, str]],
    nodes: Mapping[str, v2.PressureConsensus],
) -> set[tuple[str, str]]:
    """Return mutual edges whose radial line segments cross in pressure-q."""

    oriented = {
        edge: _oriented_nodes(edge, nodes)
        for edge in mutual_edges
    }
    crossing: set[tuple[str, str]] = set()
    ordered_edges = sorted(mutual_edges)
    for first_index, first_edge in enumerate(ordered_edges):
        first_start, first_end = oriented[first_edge]
        for second_edge in ordered_edges[first_index + 1 :]:
            if set(first_edge) & set(second_edge):
                continue
            second_start, second_end = oriented[second_edge]
            overlap_start = max(first_start.pressure, second_start.pressure)
            overlap_end = min(first_end.pressure, second_end.pressure)
            if overlap_end <= overlap_start:
                continue
            difference_start = _interpolated_q(
                first_start, first_end, overlap_start
            ) - _interpolated_q(second_start, second_end, overlap_start)
            difference_end = _interpolated_q(
                first_start, first_end, overlap_end
            ) - _interpolated_q(second_start, second_end, overlap_end)
            if difference_start * difference_end <= 0.0:
                crossing.update((first_edge, second_edge))
    return crossing


def _connected_components(
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {identifier: set() for identifier in node_ids}
    for first, second in edges:
        if first in adjacency and second in adjacency:
            adjacency[first].add(second)
            adjacency[second].add(first)
    remaining = set(adjacency)
    components: list[list[str]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _component_sort_key(
    identifiers: Sequence[str],
    nodes: Mapping[str, v2.PressureConsensus],
) -> tuple[float, int, str]:
    return (
        float(np.median([nodes[identifier].q for identifier in identifiers])),
        min(nodes[identifier].pressure_index for identifier in identifiers),
        min(identifiers),
    )


def _prediction_cost_from_clean_history(
    history: Sequence[v2.TrackNode],
    target: v2.TrackNode,
    median_pressure_step: float,
    config: SegmentedTrackingConfig,
) -> float:
    prediction, predicted_width = _trajectory_prediction(history, target.pressure)
    pressure_gap_factor = max(
        1.0,
        abs(target.pressure - history[-1].pressure) / median_pressure_step,
    )
    gate = (
        config.track_gate_factor
        * math.sqrt(predicted_width**2 + target.fwhm_q**2)
        * pressure_gap_factor
    )
    if not (
        np.isfinite(gate)
        and gate > 0
        and predicted_width > 0
        and target.fwhm_q > 0
    ):
        return math.inf
    normalized_dq = abs(target.q - prediction) / gate
    if normalized_dq > 1.0:
        return math.inf
    width_term = math.log(target.fwhm_q / predicted_width)
    return float(
        normalized_dq**2 + config.track_width_cost_weight * width_term**2
    )


def _revalidate_predictions_after_cuts(
    accepted_edges: Sequence[tuple[str, str]],
    nodes: Mapping[str, v2.PressureConsensus],
    pressure_levels: Sequence[float],
    config: SegmentedTrackingConfig,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    """Replay every clean component without using history across a cut.

    The discovery passes may initially have predicted a later node using a
    trajectory history that contains a subsequently rejected one-way or
    crossing edge.  This deterministic replay starts anew at every cut.  An
    edge remains usable only when its gate/cost is reproducible from clean
    history in both pressure directions.
    """

    remaining = {tuple(sorted(edge)) for edge in accepted_edges}
    failed_forward: set[tuple[str, str]] = set()
    failed_backward: set[tuple[str, str]] = set()
    sorted_pressures = np.sort(np.asarray(pressure_levels, dtype=float))
    positive_steps = np.diff(sorted_pressures)
    positive_steps = positive_steps[positive_steps > 0]
    median_step = float(np.median(positive_steps)) if positive_steps.size else 1.0

    while True:
        node_ids = sorted({identifier for edge in remaining for identifier in edge})
        components = _connected_components(node_ids, sorted(remaining))
        newly_failed_forward: set[tuple[str, str]] = set()
        newly_failed_backward: set[tuple[str, str]] = set()
        for component in components:
            ordered_ids = sorted(
                component,
                key=lambda identifier: (
                    nodes[identifier].pressure_index,
                    nodes[identifier].q,
                    identifier,
                ),
            )
            # A mutual one-to-one identity component must be a pressure-ordered
            # path.  Reject branching/cycling rather than inventing an order.
            component_edges = {
                edge
                for edge in remaining
                if edge[0] in component and edge[1] in component
            }
            expected_edges = {
                tuple(sorted((first, second)))
                for first, second in zip(ordered_ids, ordered_ids[1:])
            }
            if component_edges != expected_edges:
                newly_failed_forward.update(component_edges)
                newly_failed_backward.update(component_edges)
                continue

            history: list[v2.TrackNode] = [
                _consensus_to_node(nodes[ordered_ids[0]])
            ]
            for identifier in ordered_ids[1:]:
                target = _consensus_to_node(nodes[identifier])
                edge = tuple(sorted((history[-1].consensus_id, identifier)))
                cost = _prediction_cost_from_clean_history(
                    history, target, median_step, config
                )
                if not np.isfinite(cost) or cost >= config.unmatched_cost:
                    newly_failed_forward.add(edge)
                    history = [target]
                else:
                    history.append(target)

            reverse_ids = list(reversed(ordered_ids))
            history = [_consensus_to_node(nodes[reverse_ids[0]])]
            for identifier in reverse_ids[1:]:
                target = _consensus_to_node(nodes[identifier])
                edge = tuple(sorted((history[-1].consensus_id, identifier)))
                cost = _prediction_cost_from_clean_history(
                    history, target, median_step, config
                )
                if not np.isfinite(cost) or cost >= config.unmatched_cost:
                    newly_failed_backward.add(edge)
                    history = [target]
                else:
                    history.append(target)

        newly_failed = newly_failed_forward | newly_failed_backward
        if not newly_failed:
            break
        failed_forward.update(newly_failed_forward)
        failed_backward.update(newly_failed_backward)
        previous_size = len(remaining)
        remaining.difference_update(newly_failed)
        if len(remaining) == previous_size:
            break
    return remaining, failed_forward, failed_backward


def _build_events(
    competitions: Sequence[CompetitionEvidence],
    evidence: Sequence[TrajectoryLinkEvidence],
    gap_cuts: Sequence[GapCutEvidence],
    nodes: Mapping[str, v2.PressureConsensus],
) -> tuple[AmbiguityEvent, ...]:
    raw_events: list[tuple[str, str, tuple[int, ...], tuple[str, ...], tuple[str, ...], float, str]] = []
    for item in competitions:
        pressure_indices = tuple(
            sorted(
                {
                    nodes[identifier].pressure_index
                    for identifier in item.node_ids
                    if identifier in nodes
                }
            )
        )
        raw_events.append(
            (
                "low_margin_competition",
                item.direction,
                pressure_indices,
                item.node_ids,
                (),
                item.cost_margin,
                f"{item.competition_axis}: margin {item.cost_margin:.12g} "
                "below frozen threshold",
            )
        )
    for item in evidence:
        if item.accepted or not (item.forward_matched or item.backward_matched):
            continue
        raw_events.append(
            (
                "cut_edge",
                "bidirectional_reconciliation",
                tuple(sorted({item.first_pressure_index, item.second_pressure_index})),
                tuple(sorted({item.first_consensus_id, item.second_consensus_id})),
                (item.edge_id,),
                math.nan,
                item.cut_reason,
            )
        )
    for item in gap_cuts:
        raw_events.append(
            (
                "cut_missing_too_long",
                item.direction,
                tuple(
                    sorted(
                        {
                            item.source_pressure_index,
                            item.first_unreachable_pressure_index,
                        }
                    )
                ),
                (item.source_consensus_id,),
                (),
                math.nan,
                item.reason,
            )
        )
    raw_events.sort(
        key=lambda item: (
            item[0],
            item[2],
            item[3],
            item[4],
            item[1],
            item[6],
        )
    )
    return tuple(
        AmbiguityEvent(
            event_id=f"ambiguity_event_{index:05d}",
            event_type=item[0],
            direction=item[1],
            pressure_indices=item[2],
            node_ids=item[3],
            edge_ids=item[4],
            cost_margin=item[5],
            reason=item[6],
        )
        for index, item in enumerate(raw_events, start=1)
    )


def segment_consensus_bidirectional(
    consensus_by_pressure: Mapping[float, Sequence[v2.PressureConsensus]],
    pressure_levels: Sequence[float],
    config: SegmentedTrackingConfig,
) -> SegmentedTrackingResult:
    """Build deterministic clean segments from bidirectional edge evidence.

    Ambiguity is local: uncertain edges are cut and low-margin nodes are
    quarantined.  It never propagates to an otherwise clean connected segment.
    """

    supplied_pressures = tuple(float(value) for value in pressure_levels)
    if not supplied_pressures or len(set(supplied_pressures)) != len(supplied_pressures):
        raise ValueError("pressure_levels must be non-empty and unique")
    # Pressure traversal is canonicalized numerically.  Reversing or permuting
    # the caller's list cannot change the accepted undirected edge set or IDs.
    pressures = tuple(sorted(supplied_pressures))
    for index, pressure in enumerate(pressures):
        for item in consensus_by_pressure.get(pressure, ()):
            if item.reliable and item.pressure_index != index:
                raise ValueError(
                    f"consensus {item.consensus_id!r} pressure_index "
                    f"{item.pressure_index} does not match pressure_levels index {index}"
                )

    forward = _directional_pass(
        consensus_by_pressure, pressures, config, reverse=False
    )
    backward = _directional_pass(
        consensus_by_pressure, pressures, config, reverse=True
    )
    all_consensus = {
        item.consensus_id: item
        for pressure in pressures
        for item in consensus_by_pressure.get(pressure, ())
        if item.reliable
    }
    if not all_consensus:
        return SegmentedTrackingResult(
            segments=(),
            pass_links=forward.links + backward.links,
            competitions=forward.competitions + backward.competitions,
            link_evidence=(),
            ambiguity_events=(),
            gap_cuts=forward.gap_cuts + backward.gap_cuts,
            quarantined_node_ids=(),
            parents=(),
        )

    forward_lookup = _pass_edge_lookup(forward.links, "low_to_high")
    backward_lookup = _pass_edge_lookup(backward.links, "high_to_low")
    edge_keys = sorted(set(forward_lookup) | set(backward_lookup))
    matched_forward = {
        key for key, value in forward_lookup.items() if value.matched
    }
    matched_backward = {
        key for key, value in backward_lookup.items() if value.matched
    }
    mutual_edges = sorted(matched_forward & matched_backward)
    quarantined = set(forward.quarantined_node_ids) | set(
        backward.quarantined_node_ids
    )
    quarantined_pressure_indices = {
        all_consensus[identifier].pressure_index
        for identifier in quarantined
        if identifier in all_consensus
    }
    crossing_edges = _crossing_mutual_edges(mutual_edges, all_consensus)

    evidence_rows: list[TrajectoryLinkEvidence] = []
    for edge_index, edge_key in enumerate(
        sorted(
            edge_keys,
            key=lambda edge: (
                _oriented_nodes(edge, all_consensus)[0].pressure_index,
                _oriented_nodes(edge, all_consensus)[1].pressure_index,
                _oriented_nodes(edge, all_consensus)[0].consensus_id,
                _oriented_nodes(edge, all_consensus)[1].consensus_id,
            ),
        ),
        start=1,
    ):
        first, second = _oriented_nodes(edge_key, all_consensus)
        forward_item = forward_lookup.get(edge_key)
        backward_item = backward_lookup.get(edge_key)
        forward_matched = bool(forward_item and forward_item.matched)
        backward_matched = bool(backward_item and backward_item.matched)
        mutual = forward_matched and backward_matched
        endpoint_quarantined = bool(set(edge_key) & quarantined)
        lower_index = min(first.pressure_index, second.pressure_index)
        upper_index = max(first.pressure_index, second.pressure_index)
        crosses_quarantined_boundary = any(
            lower_index < pressure_index < upper_index
            for pressure_index in quarantined_pressure_indices
        )
        crossing = edge_key in crossing_edges
        endpoint_margins = (
            forward_item.source_margin if forward_item is not None else math.nan,
            forward_item.target_margin if forward_item is not None else math.nan,
            backward_item.source_margin if backward_item is not None else math.nan,
            backward_item.target_margin if backward_item is not None else math.nan,
        )
        margins_pass = bool(
            mutual
            and all(
                np.isfinite(value) and value >= config.ambiguous_cost_margin
                for value in endpoint_margins
            )
        )
        accepted = (
            mutual
            and margins_pass
            and not endpoint_quarantined
            and not crosses_quarantined_boundary
            and not crossing
        )
        reasons: list[str] = []
        if forward_matched != backward_matched:
            reasons.append("cut_one_way")
        elif not mutual and not bool(
            (forward_item and forward_item.admissible)
            or (backward_item and backward_item.admissible)
        ):
            # Outside-gate evidence is a configured scientific cut.  An
            # admissible-but-unselected candidate is merely audit evidence and
            # deliberately has an empty cut reason.
            reasons.append("cut_outside_gate")
        elif not mutual:
            pass
        if mutual and not margins_pass:
            reasons.append("cut_low_margin")
        if (
            (forward_matched or backward_matched)
            and (endpoint_quarantined or crosses_quarantined_boundary)
            and "cut_low_margin" not in reasons
        ):
            reasons.append("cut_low_margin")
        if crossing:
            reasons.append("cut_order_crossing")
        missing_pressure_levels = abs(
            second.pressure_index - first.pressure_index
        ) - 1
        evidence_rows.append(
            TrajectoryLinkEvidence(
                edge_id=f"trajectory_edge_{edge_index:06d}",
                first_consensus_id=first.consensus_id,
                second_consensus_id=second.consensus_id,
                first_pressure_GPa=first.pressure,
                second_pressure_GPa=second.pressure,
                first_pressure_index=first.pressure_index,
                second_pressure_index=second.pressure_index,
                first_q_A_inv=first.q,
                second_q_A_inv=second.q,
                missing_pressure_levels=missing_pressure_levels,
                forward_evaluated=forward_item is not None,
                backward_evaluated=backward_item is not None,
                forward_admissible=bool(forward_item and forward_item.admissible),
                backward_admissible=bool(backward_item and backward_item.admissible),
                within_original_gate=bool(
                    forward_item
                    and forward_item.admissible
                    and backward_item
                    and backward_item.admissible
                ),
                forward_matched=forward_matched,
                backward_matched=backward_matched,
                forward_cost=(
                    forward_item.cost if forward_item is not None else math.nan
                ),
                backward_cost=(
                    backward_item.cost if backward_item is not None else math.nan
                ),
                forward_source_margin=endpoint_margins[0],
                forward_target_margin=endpoint_margins[1],
                backward_source_margin=endpoint_margins[2],
                backward_target_margin=endpoint_margins[3],
                mutual=mutual,
                endpoint_quarantined=endpoint_quarantined,
                crosses_quarantined_boundary=crosses_quarantined_boundary,
                order_crossing=crossing,
                accepted=accepted,
                cut_reason=";".join(reasons),
            )
        )
    initially_accepted_edges = sorted(
        tuple(sorted((item.first_consensus_id, item.second_consensus_id)))
        for item in evidence_rows
        if item.accepted
    )
    accepted_edge_set, replay_failed_forward, replay_failed_backward = (
        _revalidate_predictions_after_cuts(
            initially_accepted_edges,
            all_consensus,
            pressures,
            config,
        )
    )
    revalidated_rows: list[TrajectoryLinkEvidence] = []
    for item in evidence_rows:
        edge_key = tuple(
            sorted((item.first_consensus_id, item.second_consensus_id))
        )
        if item.accepted and edge_key not in accepted_edge_set:
            failed_forward = edge_key in replay_failed_forward
            failed_backward = edge_key in replay_failed_backward
            replay_reason = (
                "cut_outside_gate"
                if failed_forward and failed_backward
                else "cut_one_way"
            )
            revalidated_rows.append(
                replace(item, accepted=False, cut_reason=replay_reason)
            )
        else:
            revalidated_rows.append(item)
    link_evidence = tuple(revalidated_rows)
    accepted_edges = sorted(accepted_edge_set)

    # Parent lineage is based on mutual matched edges before local cuts.  It is
    # used only for stable naming/audit and never restores a rejected link.
    proposed_matched_edges = sorted(matched_forward | matched_backward)
    parent_components = _connected_components(
        tuple(sorted(all_consensus)), proposed_matched_edges
    )
    parent_components.sort(
        key=lambda identifiers: _component_sort_key(identifiers, all_consensus)
    )
    all_competitions = forward.competitions + backward.competitions
    all_gap_cuts = forward.gap_cuts + backward.gap_cuts
    events = _build_events(
        all_competitions, link_evidence, all_gap_cuts, all_consensus
    )
    minimum_levels = v2.minimum_pressure_support(len(pressures))

    segments: list[SegmentedTrack] = []
    parent_rows: list[ParentTrack] = []
    for parent_index, parent_node_ids in enumerate(parent_components, start=1):
        parent_id = f"radial_peak_{parent_index:03d}"
        parent_set = set(parent_node_ids)
        clean_node_ids = sorted(parent_set - quarantined)
        parent_accepted_edges = [
            edge
            for edge in accepted_edges
            if edge[0] in parent_set and edge[1] in parent_set
        ]
        clean_components = _connected_components(
            clean_node_ids, parent_accepted_edges
        )
        clean_components.sort(
            key=lambda identifiers: (
                min(all_consensus[item].pressure_index for item in identifiers),
                float(np.median([all_consensus[item].q for item in identifiers])),
                min(identifiers),
            )
        )
        parent_segments: list[SegmentedTrack] = []
        for segment_index, identifiers in enumerate(clean_components, start=1):
            segment_id = f"{parent_id}_segment_{segment_index:02d}"
            node_values = tuple(
                _consensus_to_node(all_consensus[identifier])
                for identifier in sorted(
                    identifiers,
                    key=lambda item: (
                        all_consensus[item].pressure_index,
                        all_consensus[item].q,
                        item,
                    ),
                )
            )
            level_count = len({node.pressure_index for node in node_values})
            official = level_count >= minimum_levels
            parent_segments.append(
                SegmentedTrack(
                    track_id=segment_id,
                    parent_track_id=parent_id,
                    channel=all_consensus[identifiers[0]].channel,
                    nodes=node_values,
                    official=official,
                    ambiguous=False,
                    minimum_pressure_support=minimum_levels,
                    segment_index=segment_index,
                    segment_count=len(clean_components),
                    status=(
                        "official"
                        if official
                        else "insufficient_pressure_support"
                    ),
                )
            )

        finalized_parent_segments: list[SegmentedTrack] = []
        for segment in parent_segments:
            segment_node_ids = {node.consensus_id for node in segment.nodes}
            segment_pressure_indices = {
                node.pressure_index for node in segment.nodes
            }
            relevant_events = [
                event
                for event in events
                if segment_node_ids & set(event.node_ids)
            ]
            boundary_indices: set[int] = set()
            for event in relevant_events:
                for identifier in event.node_ids:
                    if identifier in segment_node_ids or identifier not in all_consensus:
                        continue
                    pressure_index = all_consensus[identifier].pressure_index
                    if pressure_index not in segment_pressure_indices:
                        boundary_indices.add(pressure_index)
            finalized_parent_segments.append(
                replace(
                    segment,
                    boundary_unknown_pressure_indices=tuple(sorted(boundary_indices)),
                    cut_event_ids=tuple(event.event_id for event in relevant_events),
                )
            )
        segments.extend(finalized_parent_segments)

        cut_evidence = [
            item
            for item in link_evidence
            if not item.accepted
            and (item.forward_matched or item.backward_matched)
            and (
                item.first_consensus_id in parent_set
                or item.second_consensus_id in parent_set
            )
        ]
        parent_rows.append(
            ParentTrack(
                parent_track_id=parent_id,
                channel=all_consensus[parent_node_ids[0]].channel,
                node_ids=tuple(sorted(parent_node_ids)),
                segment_ids=tuple(
                    item.track_id for item in finalized_parent_segments
                ),
                had_cuts=bool(cut_evidence or parent_set & quarantined),
            )
        )

    if len({node.consensus_id for segment in segments for node in segment.nodes}) != sum(
        len(segment.nodes) for segment in segments
    ):
        raise AssertionError("a consensus node was assigned to multiple v2.1 segments")
    accepted_node_ids = {
        identifier for edge in accepted_edges for identifier in edge
    }
    if accepted_node_ids & quarantined:
        raise AssertionError("a quarantined node appears on an accepted edge")
    if any(item.accepted and not item.mutual for item in link_evidence):
        raise AssertionError("a non-mutual edge was accepted")
    if any(item.accepted and item.order_crossing for item in link_evidence):
        raise AssertionError("an order-crossing edge was accepted")
    for item in link_evidence:
        if not item.accepted:
            continue
        endpoint_margins = (
            item.forward_source_margin,
            item.forward_target_margin,
            item.backward_source_margin,
            item.backward_target_margin,
        )
        if not all(
            np.isfinite(value) and value >= config.ambiguous_cost_margin
            for value in endpoint_margins
        ):
            raise AssertionError(
                f"accepted edge {item.edge_id} lacks four passing endpoint margins"
            )

    return SegmentedTrackingResult(
        segments=tuple(segments),
        pass_links=forward.links + backward.links,
        competitions=all_competitions,
        link_evidence=link_evidence,
        ambiguity_events=events,
        gap_cuts=all_gap_cuts,
        quarantined_node_ids=tuple(sorted(quarantined)),
        parents=tuple(parent_rows),
    )


def resolve_segment_target(
    segment: SegmentedTrack,
    pressure: float,
    pressure_index: int,
) -> SegmentTarget:
    """Resolve an exact/interpolated target without crossing a v2.1 cut."""

    if pressure_index in segment.boundary_unknown_pressure_indices:
        return SegmentTarget(
            state="unknown",
            reason="ambiguous_trajectory_boundary",
        )
    nodes = sorted(segment.nodes, key=lambda node: node.pressure)
    if pressure < nodes[0].pressure or pressure > nodes[-1].pressure:
        return SegmentTarget(
            state="out_of_range",
            reason="outside_supported_segment_range",
        )
    for node in nodes:
        if pressure == node.pressure:
            return SegmentTarget(
                state="target",
                reason="exact_segment_consensus",
                q=node.q,
                fwhm_q=node.fwhm_q,
            )
    lower = [node for node in nodes if node.pressure < pressure]
    upper = [node for node in nodes if node.pressure > pressure]
    if not lower or not upper:
        return SegmentTarget(
            state="out_of_range",
            reason="outside_supported_segment_range",
        )
    first, second = lower[-1], upper[0]
    # A segment can bridge at most the frozen number of missing levels because
    # its accepted edge came from the directional tracker.  No cut edge can be
    # present inside this clean connected component.
    fraction = (pressure - first.pressure) / (second.pressure - first.pressure)
    return SegmentTarget(
        state="target",
        reason="interpolated_within_accepted_segment_edge",
        q=float(first.q + fraction * (second.q - first.q)),
        fwhm_q=float(
            first.fwhm_q + fraction * (second.fwhm_q - first.fwhm_q)
        ),
    )


__all__ = [
    "ALGORITHM_VERSION",
    "AmbiguityEvent",
    "CompetitionEvidence",
    "GapCutEvidence",
    "ParentTrack",
    "PassLink",
    "SegmentTarget",
    "SegmentedTrack",
    "SegmentedTrackingConfig",
    "SegmentedTrackingResult",
    "TrajectoryLinkEvidence",
    "resolve_segment_target",
    "segment_consensus_bidirectional",
]
