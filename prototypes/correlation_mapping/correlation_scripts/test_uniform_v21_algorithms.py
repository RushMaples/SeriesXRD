#!/usr/bin/env python3
"""Synthetic and regression gates for ``uniform-correlation-v2.1``.

The suite is intentionally independent of every real UOTe XY file and result
directory.  It locks the scientific contract that v2.1 changes only the
pressure-trajectory ambiguity scope: uncertain links are cut locally, while
all v2 preprocessing, detection, consensus, similarity, missingness, and
support semantics remain frozen.
"""

from __future__ import annotations

from dataclasses import asdict, fields
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import uniform_peak_core as peak  # noqa: E402
import uniform_peak_tracking_v21 as tracking  # noqa: E402
import uniform_profile_binding as profile_v2  # noqa: E402
import uniform_profile_binding_v21 as profile_v21  # noqa: E402


WAVELENGTH = 0.3066
V2_PROFILE_PATH = SCRIPT_DIR / "configs" / "uniform-correlation-v2.json"
V21_PROFILE_PATH = SCRIPT_DIR / "configs" / "uniform-correlation-v2.1.json"


def tracking_config() -> tracking.SegmentedTrackingConfig:
    """Return the exact frozen v2.1 tracking policy used by all cases."""

    return tracking.SegmentedTrackingConfig(
        algorithm_version="2.1.0",
        max_missing_pressure_levels=2,
        track_gate_factor=1.5,
        track_width_cost_weight=0.1,
        ambiguous_cost_margin=0.25,
        unmatched_cost=10.0,
        decision_unit="candidate_edge",
        require_bidirectional_same_edge=True,
        require_margin_at_both_endpoints=True,
        reject_q_order_crossing=True,
        low_margin_competitor_state="unknown_quarantined",
        cut_reasons=(
            "cut_one_way",
            "cut_low_margin",
            "cut_order_crossing",
            "cut_missing_too_long",
            "cut_outside_gate",
        ),
        allow_interpolation_across_cut=False,
    )


def consensus(
    identifier: str,
    pressure_index: int,
    pressure: float,
    q: float,
    *,
    width: float = 0.05,
    relative_area: float = 0.1,
) -> peak.PressureConsensus:
    return peak.PressureConsensus(
        consensus_id=identifier,
        channel="spots",
        pressure=float(pressure),
        pressure_index=int(pressure_index),
        q=float(q),
        fwhm_q=float(width),
        relative_area=float(relative_area),
        support=5,
        total_scans=5,
        required_support=5,
        member_keys=(),
        reliable=True,
        ambiguous=False,
    )


def one_branch(
    pressure_levels: Sequence[float],
    present_indices: Iterable[int] | None = None,
    *,
    prefix: str = "A",
    q0: float = 1.0,
    slope: float = 0.015,
    width: float = 0.05,
    relative_area_scale: float = 1.0,
) -> dict[float, tuple[peak.PressureConsensus, ...]]:
    present = (
        set(range(len(pressure_levels)))
        if present_indices is None
        else {int(value) for value in present_indices}
    )
    return {
        float(pressure): (
            (
                consensus(
                    f"{prefix}_p{index:02d}",
                    index,
                    pressure,
                    q0 + slope * float(pressure),
                    width=width,
                    relative_area=relative_area_scale * (0.1 + 0.001 * index),
                ),
            )
            if index in present
            else ()
        )
        for index, pressure in enumerate(pressure_levels)
    }


def accepted_edges(
    result: tracking.SegmentedTrackingResult,
) -> frozenset[tuple[str, str]]:
    return frozenset(
        tuple(sorted((item.first_consensus_id, item.second_consensus_id)))
        for item in result.link_evidence
        if item.accepted
    )


def segment_node_sets(
    result: tracking.SegmentedTrackingResult,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(
            tuple(sorted(node.consensus_id for node in segment.nodes))
            for segment in result.segments
        )
    )


def official_segments(
    result: tracking.SegmentedTrackingResult,
) -> list[tracking.SegmentedTrack]:
    return [segment for segment in result.segments if segment.official]


def assigned_observation(
    scan: str,
    pressure_value: float,
    state: str,
    *,
    area: float = math.nan,
    q: float = math.nan,
    width: float = math.nan,
) -> peak.AssignedObservation:
    return peak.AssignedObservation(
        track_id="synthetic_segment",
        scan=scan,
        pressure=float(pressure_value),
        frame=0,
        state=state,
        reason=f"synthetic_{state}",
        relative_area=float(area),
        q=float(q),
        fwhm_q=float(width),
    )


def fitted_peak(peak_id: int, q: float, *, width: float = 0.05) -> peak.PeakFit:
    return peak.PeakFit(
        peak_id=int(peak_id),
        candidate_index=int(peak_id),
        state="reliable",
        reason="synthetic_reliable",
        two_theta=10.0 + peak_id,
        q=float(q),
        fwhm_two_theta=0.1,
        fwhm_q=float(width),
        area=1.0,
        area_se=0.1,
        relative_area=0.1,
        eta=0.5,
        height_snr=20.0,
        delta_bic=30.0,
        fit_success=True,
        at_parameter_boundary=False,
        group_id=int(peak_id),
    )


def pseudo_voigt(
    x: np.ndarray,
    center: float,
    fwhm: float,
    area: float,
    eta: float = 0.4,
) -> np.ndarray:
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gamma = fwhm / 2.0
    gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi)
    )
    lorentzian = gamma / (math.pi * ((x - center) ** 2 + gamma**2))
    return area * ((1.0 - eta) * gaussian + eta * lorentzian)


class V21UpstreamFreezeTests(unittest.TestCase):
    def test_v21_upstream_profile_is_exactly_v2(self) -> None:
        v2 = json.loads(V2_PROFILE_PATH.read_text(encoding="utf-8"))
        v21 = json.loads(V21_PROFILE_PATH.read_text(encoding="utf-8"))
        for section in (
            "input",
            "baseline",
            "peak_detection",
            "peak_fit",
            "consensus",
            "similarity",
            "statistics",
            "windows",
            "plotting",
        ):
            self.assertEqual(v21[section], v2[section], section)
        for key in (
            "gate_factor",
            "width_cost_weight",
            "gate_rule",
            "cost_rule",
            "maximum_missing_pressure_levels",
            "minimum_pressure_support_rule",
            "allow_extrapolation",
            "ambiguous_cost_margin",
        ):
            self.assertEqual(v21["tracking"][key], v2["tracking"][key], key)

        bound_v2 = profile_v2.bind_frozen_profile(v2, WAVELENGTH)
        bound_v21 = profile_v21.bind_frozen_profile_v21(v21, WAVELENGTH)
        self.assertEqual(asdict(bound_v21.peak_config), asdict(bound_v2.peak_config))
        self.assertEqual(asdict(bound_v21.window_config), asdict(bound_v2.window_config))
        self.assertEqual(
            set(bound_v21.binding_audit["segmented_tracking_config"]),
            {item.name for item in fields(tracking.SegmentedTrackingConfig)},
        )
        self.assertEqual(bound_v21.tracking_config.ambiguous_cost_margin, 0.25)
        self.assertEqual(bound_v21.tracking_config.unmatched_cost, 10.0)
        self.assertTrue(bound_v21.tracking_config.require_margin_at_both_endpoints)


class V21SegmentTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tracking_config()

    def test_continuous_peak_is_one_complete_official_segment(self) -> None:
        pressures = tuple(float(value) for value in range(7))
        result = tracking.segment_consensus_bidirectional(
            one_branch(pressures), pressures, self.config
        )
        official = official_segments(result)
        self.assertEqual(len(official), 1)
        self.assertEqual(len(official[0].nodes), len(pressures))
        self.assertEqual(len(accepted_edges(result)), len(pressures) - 1)
        self.assertFalse(result.quarantined_node_ids)
        self.assertTrue(all(item.mutual for item in result.link_evidence if item.accepted))

    def test_local_low_margin_ambiguity_splits_without_poisoning_clean_sides(self) -> None:
        pressures = tuple(float(value) for value in range(9))
        by_pressure = one_branch(pressures)
        boundary = 4
        main = by_pressure[pressures[boundary]][0]
        by_pressure[pressures[boundary]] = (
            main,
            consensus(
                "decoy_p04",
                boundary,
                pressures[boundary],
                main.q + 0.002,
                width=main.fwhm_q,
            ),
        )
        result = tracking.segment_consensus_bidirectional(
            by_pressure, pressures, self.config
        )
        self.assertTrue(result.competitions)
        self.assertTrue(result.quarantined_node_ids)
        self.assertTrue(any("cut_low_margin" in item.cut_reason for item in result.link_evidence))
        for item in result.link_evidence:
            if item.endpoint_quarantined:
                self.assertFalse(item.accepted)
        for segment in result.segments:
            ids = {node.consensus_id for node in segment.nodes}
            self.assertFalse(ids & set(result.quarantined_node_ids))
            indices = {node.pressure_index for node in segment.nodes}
            self.assertFalse(any(i < boundary for i in indices) and any(i > boundary for i in indices))

        left = [
            item
            for item in official_segments(result)
            if max(node.pressure_index for node in item.nodes) < boundary
        ]
        right = [
            item
            for item in official_segments(result)
            if min(node.pressure_index for node in item.nodes) > boundary
        ]
        self.assertTrue(left, "clean low-pressure side should survive locally")
        self.assertTrue(right, "clean high-pressure side should survive locally")

    def test_crossing_peaks_are_cut_and_never_switch_identity(self) -> None:
        pressures = tuple(float(value) for value in range(6))
        branch_a = (0.90, 0.94, 0.98, 1.02, 1.06, 1.10)
        branch_b = (1.10, 1.06, 1.02, 0.98, 0.94, 0.90)
        by_pressure = {
            pressure_value: (
                consensus(f"A_p{index:02d}", index, pressure_value, branch_a[index], width=0.08),
                consensus(f"B_p{index:02d}", index, pressure_value, branch_b[index], width=0.08),
            )
            for index, pressure_value in enumerate(pressures)
        }
        result = tracking.segment_consensus_bidirectional(
            by_pressure, pressures, self.config
        )
        crossing = [item for item in result.link_evidence if item.order_crossing]
        self.assertTrue(crossing)
        self.assertTrue(all(not item.accepted for item in crossing))
        self.assertTrue(any("cut_order_crossing" in item.cut_reason for item in crossing))
        for segment in result.segments:
            truth_branches = {node.consensus_id.split("_", 1)[0] for node in segment.nodes}
            self.assertLessEqual(
                len(truth_branches),
                1,
                f"identity switch in {segment.track_id}: {truth_branches}",
            )

    def test_missing_one_or_two_levels_bridge_but_three_levels_cut(self) -> None:
        pressures = tuple(float(value) for value in range(9))
        cases = (
            ({0, 1, 3, 4, 5, 6, 7, 8}, 1, True),
            ({0, 1, 4, 5, 6, 7, 8}, 2, True),
            ({0, 1, 5, 6, 7, 8}, 3, False),
        )
        for indices, missing_count, should_bridge in cases:
            with self.subTest(missing_levels=missing_count):
                result = tracking.segment_consensus_bidirectional(
                    one_branch(pressures, indices), pressures, self.config
                )
                first = "A_p01"
                second = f"A_p{missing_count + 2:02d}"
                bridge = tuple(sorted((first, second)))
                self.assertEqual(bridge in accepted_edges(result), should_bridge)
                if should_bridge:
                    containing = [
                        segment
                        for segment in result.segments
                        if {first, second}.issubset(
                            {node.consensus_id for node in segment.nodes}
                        )
                    ]
                    self.assertEqual(len(containing), 1)
                    for missing_index in range(2, missing_count + 2):
                        target = tracking.resolve_segment_target(
                            containing[0], pressures[missing_index], missing_index
                        )
                        self.assertEqual(target.state, "target")
                        self.assertEqual(
                            target.reason, "interpolated_within_accepted_segment_edge"
                        )
                else:
                    self.assertFalse(
                        any(
                            {first, second}.issubset(
                                {node.consensus_id for node in segment.nodes}
                            )
                            for segment in result.segments
                        )
                    )

    def test_no_target_is_interpolated_across_an_ambiguity_cut(self) -> None:
        pressures = tuple(float(value) for value in range(9))
        by_pressure = one_branch(pressures)
        boundary = 4
        main = by_pressure[pressures[boundary]][0]
        by_pressure[pressures[boundary]] = (
            main,
            consensus(
                "decoy_boundary",
                boundary,
                pressures[boundary],
                main.q + 0.001,
                width=main.fwhm_q,
            ),
        )
        result = tracking.segment_consensus_bidirectional(
            by_pressure, pressures, self.config
        )
        self.assertTrue(result.quarantined_node_ids)
        for segment in result.segments:
            target = tracking.resolve_segment_target(
                segment, pressures[boundary], boundary
            )
            self.assertNotEqual(target.state, "target")
            self.assertTrue(
                target.state in {"unknown", "out_of_range"},
                (segment.track_id, target),
            )

    def test_consensus_nodes_and_frame_detections_are_one_to_one(self) -> None:
        pressures = tuple(float(value) for value in range(5))
        by_pressure = {
            pressure_value: (
                consensus(f"A_p{index:02d}", index, pressure_value, 1.0 + 0.01 * index),
                consensus(f"B_p{index:02d}", index, pressure_value, 1.4 + 0.01 * index),
            )
            for index, pressure_value in enumerate(pressures)
        }
        result = tracking.segment_consensus_bidirectional(
            by_pressure, pressures, self.config
        )
        official = official_segments(result)
        self.assertEqual(len(official), 2)
        node_ids = [node.consensus_id for segment in result.segments for node in segment.nodes]
        self.assertEqual(len(node_ids), len(set(node_ids)))

        pressure_index = 2
        frame = peak.FramePeaks(
            frame=100,
            scan="scanA",
            pressure=pressures[pressure_index],
            channel="spots",
            peaks=(
                fitted_peak(1, 1.0 + 0.01 * pressure_index),
                fitted_peak(2, 1.4 + 0.01 * pressure_index),
            ),
            pattern_valid=True,
            noise=1.0,
            total_positive_area=1.0,
            measured_q_min=0.5,
            measured_q_max=2.0,
        )
        assignments = peak.assign_track_observations(
            official,
            [frame],
            ["scanA"],
            pressures,
            peak.UniformPeakConfig(wavelength=WAVELENGTH, bootstrap_iterations=0),
        )
        owners: list[tuple[int, int]] = []
        for values in assignments.values():
            item = values[("scanA", pressures[pressure_index])]
            if item.state == "present":
                owners.append((int(item.frame), int(item.peak_id)))
        self.assertEqual(len(owners), 2)
        self.assertEqual(len(owners), len(set(owners)))

    def test_tracking_is_independent_of_peak_area_scale(self) -> None:
        pressures = tuple(float(value) for value in range(7))
        original = tracking.segment_consensus_bidirectional(
            one_branch(pressures, relative_area_scale=1.0),
            pressures,
            self.config,
        )
        scaled = tracking.segment_consensus_bidirectional(
            one_branch(pressures, relative_area_scale=100.0),
            pressures,
            self.config,
        )
        self.assertEqual(accepted_edges(original), accepted_edges(scaled))
        self.assertEqual(segment_node_sets(original), segment_node_sets(scaled))


class V21InvariantAndCorrelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracking_config = tracking_config()
        self.peak_config = peak.UniformPeakConfig(
            wavelength=WAVELENGTH,
            bootstrap_iterations=0,
        )

    def test_relative_area_is_invariant_to_global_intensity_scaling(self) -> None:
        x = np.linspace(5.0, 15.0, 2001)
        baseline = 40.0 + 1.5 * x + 0.015 * (x - 10.0) ** 2
        signal = baseline + pseudo_voigt(x, 9.0, 0.15, 8.0)
        signal += np.random.default_rng(23).normal(0.0, 0.02, x.size)
        relative_areas: list[float] = []
        for frame_index, scale in enumerate((1.0, 7.0)):
            preprocessed = peak.preprocess_pattern(x, scale * signal, self.peak_config)
            frame = peak.detect_pattern_peaks(
                preprocessed,
                frame=frame_index,
                scan="scanA",
                pressure=0.0,
                channel="spots",
                config=self.peak_config,
            )
            reliable = [item for item in frame.peaks if item.reliable]
            self.assertEqual(len(reliable), 1)
            relative_areas.append(reliable[0].relative_area)
        self.assertLess(abs(relative_areas[0] - relative_areas[1]), 1.0e-3)

    def test_correlations_pair_only_within_the_same_scan(self) -> None:
        scans = [f"scan{index}" for index in range(5)]
        values: dict[tuple[str, float], peak.AssignedObservation] = {}
        for index, scan in enumerate(scans):
            first_area = 0.1 * (2.0**index)
            values[(scan, 0.0)] = assigned_observation(
                scan, 0.0, "present", area=first_area, q=1.0, width=0.04
            )
            values[(scan, 1.0)] = assigned_observation(
                scan, 1.0, "present", area=2.0 * first_area, q=1.0, width=0.04
            )
        result = peak.compute_track_correlations(
            values, scans, [0.0, 1.0], bootstrap_iterations=0, seed=0
        )
        self.assertAlmostEqual(result.area[0, 1], 0.5, places=15)
        self.assertAlmostEqual(result.location[0, 1], 1.0, places=15)
        self.assertEqual(result.n_both_present[0, 1], len(scans))

    def test_missing_and_insufficient_support_are_nan_not_zero(self) -> None:
        scans = [f"scan{index}" for index in range(5)]
        values: dict[tuple[str, float], peak.AssignedObservation] = {}
        for index, scan in enumerate(scans):
            values[(scan, 0.0)] = assigned_observation(
                scan, 0.0, "present", area=0.2, q=1.0, width=0.04
            )
            values[(scan, 1.0)] = (
                assigned_observation(scan, 1.0, "absent")
                if index == len(scans) - 1
                else assigned_observation(
                    scan, 1.0, "present", area=0.1, q=1.01, width=0.04
                )
            )
        result = peak.compute_track_correlations(
            values, scans, [0.0, 1.0], bootstrap_iterations=0, seed=0
        )
        self.assertEqual(result.n_both_present[0, 1], 4)
        self.assertEqual(result.n10[0, 1], 1)
        self.assertEqual(result.required_support[0, 1], 5)
        self.assertTrue(math.isnan(result.area[0, 1]))
        self.assertTrue(math.isnan(result.location[0, 1]))
        self.assertNotEqual(result.area[0, 1], 0.0)
        self.assertAlmostEqual(result.presence[0, 1], 0.8)

    def test_permutation_pressure_reversal_and_rerun_are_deterministic(self) -> None:
        pressures = tuple(float(value) for value in range(7))
        canonical: dict[float, tuple[peak.PressureConsensus, ...]] = {}
        for index, pressure_value in enumerate(pressures):
            canonical[pressure_value] = (
                consensus(f"A_p{index:02d}", index, pressure_value, 1.0 + 0.01 * index),
                consensus(f"B_p{index:02d}", index, pressure_value, 1.4 + 0.012 * index),
            )
        permuted = {
            pressure_value: tuple(reversed(canonical[pressure_value]))
            for pressure_value in reversed(pressures)
        }
        first = tracking.segment_consensus_bidirectional(
            canonical, pressures, self.tracking_config
        )
        rerun = tracking.segment_consensus_bidirectional(
            canonical, pressures, self.tracking_config
        )
        shuffled = tracking.segment_consensus_bidirectional(
            permuted, pressures, self.tracking_config
        )
        reversed_pressure = tracking.segment_consensus_bidirectional(
            canonical, tuple(reversed(pressures)), self.tracking_config
        )
        for candidate in (rerun, shuffled, reversed_pressure):
            self.assertEqual(accepted_edges(first), accepted_edges(candidate))
            self.assertEqual(segment_node_sets(first), segment_node_sets(candidate))

        first_evidence = {item.edge_id: item for item in first.link_evidence}
        rerun_evidence = {item.edge_id: item for item in rerun.link_evidence}
        self.assertEqual(set(first_evidence), set(rerun_evidence))
        numeric_fields = (
            "first_pressure_GPa",
            "second_pressure_GPa",
            "first_q_A_inv",
            "second_q_A_inv",
            "forward_cost",
            "backward_cost",
            "forward_source_margin",
            "forward_target_margin",
            "backward_source_margin",
            "backward_target_margin",
        )
        maximum_difference = 0.0
        for edge_id in sorted(first_evidence):
            left = first_evidence[edge_id]
            right = rerun_evidence[edge_id]
            for name in numeric_fields:
                first_value = float(getattr(left, name))
                second_value = float(getattr(right, name))
                if math.isnan(first_value) and math.isnan(second_value):
                    continue
                maximum_difference = max(
                    maximum_difference, abs(first_value - second_value)
                )
        self.assertLess(maximum_difference, 1.0e-10)

        scans = [f"scan{index}" for index in range(6)]
        assignments: dict[tuple[str, float], peak.AssignedObservation] = {}
        for index, scan in enumerate(scans):
            assignments[(scan, 0.0)] = assigned_observation(
                scan, 0.0, "present", area=0.2 + 0.01 * index, q=1.0, width=0.04
            )
            assignments[(scan, 1.0)] = assigned_observation(
                scan, 1.0, "present", area=0.1 + 0.005 * index, q=1.01, width=0.04
            )
        matrices_a = peak.compute_track_correlations(
            assignments,
            scans,
            [0.0, 1.0],
            bootstrap_iterations=50,
            seed=0,
        )
        matrices_b = peak.compute_track_correlations(
            assignments,
            scans,
            [0.0, 1.0],
            bootstrap_iterations=50,
            seed=0,
        )
        for name in (
            "area",
            "location",
            "presence",
            "area_ci_low",
            "area_ci_high",
            "location_ci_low",
            "location_ci_high",
            "presence_ci_low",
            "presence_ci_high",
        ):
            np.testing.assert_allclose(
                getattr(matrices_a, name),
                getattr(matrices_b, name),
                atol=1.0e-10,
                rtol=0.0,
                equal_nan=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
