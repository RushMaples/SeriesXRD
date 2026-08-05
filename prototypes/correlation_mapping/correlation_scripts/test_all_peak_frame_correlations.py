#!/usr/bin/env python3
"""Contract tests for track-independent, all-peak cross-frame correlations.

The public row contract exercised here is intentionally small:

* input peak rows contain ``frame``, ``two_theta_deg``, ``integrated_area``,
  ``azim_deg``, and ``obs_row``; ``track`` may be present as provenance only;
* ``assign_local_peak_ids`` returns rows augmented with ``dataset``, a one-based
  ``local_peak_index``, and ``peak_id="p{frame},{local_peak_index}"``;
* ``build_cross_frame_pair_rows`` returns one row for every peak Cartesian
  product in every canonical frame pair ``frame_a < frame_b``.

No test permits angle matching or track membership to remove a comparison.
"""

from __future__ import annotations

import copy
import sys
import unittest
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import all_peak_frame_correlations as correlations  # noqa: E402


def peak_row(
    frame: int,
    two_theta_deg: float,
    integrated_area: float,
    obs_row: int,
    *,
    azim_deg: float = 0.0,
    track: int = -1,
) -> dict[str, object]:
    return {
        "frame": frame,
        "two_theta_deg": two_theta_deg,
        "integrated_area": integrated_area,
        "azim_deg": azim_deg,
        "obs_row": obs_row,
        "track": track,
    }


def assign_and_group(
    rows: list[dict[str, object]],
    dataset: str = "synthetic",
) -> tuple[list[dict[str, object]], dict[int, list[dict[str, object]]]]:
    assigned = correlations.assign_local_peak_ids(copy.deepcopy(rows), dataset)
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in assigned:
        grouped[int(row["frame"])].append(row)
    return assigned, dict(grouped)


def three_frame_fixture() -> list[dict[str, object]]:
    return [
        peak_row(1, 5.00, 2.0, 101, azim_deg=-10.0, track=7),
        peak_row(1, 8.00, 4.0, 102, azim_deg=20.0, track=8),
        peak_row(2, 5.02, 1.0, 201, azim_deg=-8.0, track=7),
        peak_row(2, 12.0, 8.0, 202, azim_deg=0.0, track=44),
        peak_row(2, 50.0, 16.0, 203, azim_deg=30.0, track=-1),
        peak_row(3, 5.04, 2.0, 301, azim_deg=-7.0, track=-1),
    ]


class RectangularSimilarityTests(unittest.TestCase):
    def test_location_similarity_is_full_two_by_three_matrix(self) -> None:
        left = np.asarray([10.0, 20.0])
        right = np.asarray([10.0, 10.05, 40.0])

        actual = correlations.location_similarity_matrix(left, right, tolerance=0.1)

        expected = np.asarray(
            [
                [1.0, 0.5, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        self.assertEqual(actual.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)

    def test_area_similarity_uses_min_over_max_for_every_cell(self) -> None:
        left = np.asarray([2.0, 8.0])
        right = np.asarray([4.0, 1.0, 16.0])

        actual = correlations.area_similarity_matrix(left, right)

        expected = np.asarray(
            [
                [0.5, 0.5, 0.125],
                [0.5, 0.125, 0.5],
            ]
        )
        self.assertEqual(actual.shape, (2, 3))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)

    def test_two_zero_integrated_areas_are_equal(self) -> None:
        actual = correlations.area_similarity_matrix([0.0, 2.0], [0.0])
        expected = np.asarray([[1.0], [0.0]])

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


class LowerTriangleTests(unittest.TestCase):
    def test_strict_lower_triangle_removes_diagonal_and_upper_half(self) -> None:
        source = np.asarray(
            [
                [1.0, 0.2, 0.3],
                [0.2, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ]
        )

        actual = correlations.strict_lower_triangle(source)

        self.assertTrue(np.all(np.isnan(actual[np.triu_indices(3, k=0)])))
        np.testing.assert_allclose(actual[1, 0], 0.2)
        np.testing.assert_allclose(actual[2, 0], 0.3)
        np.testing.assert_allclose(actual[2, 1], 0.4)
        np.testing.assert_allclose(source[0, 0], 1.0)

    def test_strict_lower_triangle_rejects_rectangles(self) -> None:
        with self.assertRaises(ValueError):
            correlations.strict_lower_triangle(np.zeros((2, 3)))


class NumericalIntegrationTests(unittest.TestCase):
    def test_narrow_neighbor_bounded_peak_still_gets_finite_area(self) -> None:
        x = np.asarray([0.0, 1.0, 2.0])
        residual = np.asarray([0.0, 4.0, 0.0])

        areas, bounds = correlations.integrate_independent_peak_areas(
            x,
            residual,
            centers=[0.9, 1.0, 1.1],
            fwhm_values=[0.01, 0.01, 0.01],
        )

        self.assertTrue(np.isfinite(areas[1]))
        self.assertGreater(areas[1], 0.0)
        self.assertLess(bounds[1][0], bounds[1][1])


class LocalPeakIdTests(unittest.TestCase):
    def test_local_ids_are_stable_under_input_reordering(self) -> None:
        rows = [
            peak_row(2, 6.1, 3.0, 22, azim_deg=4.0, track=100),
            peak_row(1, 4.0, 2.0, 12, azim_deg=20.0, track=500),
            peak_row(2, 5.9, 1.0, 21, azim_deg=9.0, track=-1),
            peak_row(1, 4.0, 4.0, 11, azim_deg=-10.0, track=501),
        ]

        forward = correlations.assign_local_peak_ids(copy.deepcopy(rows), "single")
        reversed_rows = correlations.assign_local_peak_ids(
            copy.deepcopy(list(reversed(rows))),
            "single",
        )

        def by_observation(
            values: list[dict[str, object]],
        ) -> dict[int, tuple[str, int, str]]:
            return {
                int(row["obs_row"]): (
                    str(row["dataset"]),
                    int(row["local_peak_index"]),
                    str(row["peak_id"]),
                )
                for row in values
            }

        expected = {
            11: ("single", 1, "p1,1"),
            12: ("single", 2, "p1,2"),
            21: ("single", 1, "p2,1"),
            22: ("single", 2, "p2,2"),
        }
        self.assertEqual(by_observation(forward), expected)
        self.assertEqual(by_observation(reversed_rows), expected)


class CrossFramePairRowTests(unittest.TestCase):
    REQUIRED_FIELDS = {
        "frame_a",
        "frame_b",
        "peak_a_id",
        "peak_b_id",
        "two_theta_a_deg",
        "two_theta_b_deg",
        "integrated_area_a",
        "integrated_area_b",
        "delta_two_theta_deg",
        "location_similarity",
        "area_similarity",
    }

    def setUp(self) -> None:
        self.raw_rows = three_frame_fixture()
        _, self.peaks_by_frame = assign_and_group(self.raw_rows)
        self.pairs = correlations.build_cross_frame_pair_rows(
            self.peaks_by_frame,
            tolerance=0.1,
        )

    def test_every_cross_frame_cartesian_product_is_emitted(self) -> None:
        counts = {frame: len(rows) for frame, rows in self.peaks_by_frame.items()}
        expected_total = sum(
            counts[left] * counts[right]
            for left, right in combinations(sorted(counts), 2)
        )

        self.assertEqual(expected_total, 11)
        self.assertEqual(len(self.pairs), expected_total)
        self.assertEqual(
            Counter((int(row["frame_a"]), int(row["frame_b"])) for row in self.pairs),
            Counter({(1, 2): 6, (1, 3): 2, (2, 3): 3}),
        )
        self.assertTrue(
            all(int(row["frame_a"]) < int(row["frame_b"]) for row in self.pairs)
        )
        self.assertTrue(
            all(self.REQUIRED_FIELDS.issubset(row) for row in self.pairs)
        )

    def test_far_angle_pair_is_retained_with_zero_location_similarity(self) -> None:
        far = next(
            row
            for row in self.pairs
            if row["peak_a_id"] == "p1,1" and row["peak_b_id"] == "p2,3"
        )

        self.assertAlmostEqual(float(far["delta_two_theta_deg"]), 45.0)
        self.assertEqual(float(far["location_similarity"]), 0.0)
        self.assertAlmostEqual(float(far["area_similarity"]), 0.125)

    def test_track_values_do_not_change_ids_membership_or_scores(self) -> None:
        changed_tracks = copy.deepcopy(self.raw_rows)
        for index, row in enumerate(changed_tracks):
            row["track"] = 10_000 + index
        assigned_changed, changed_by_frame = assign_and_group(changed_tracks)
        changed_pairs = correlations.build_cross_frame_pair_rows(
            changed_by_frame,
            tolerance=0.1,
        )

        original_assigned, _ = assign_and_group(self.raw_rows)
        original_ids = {
            int(row["obs_row"]): str(row["peak_id"]) for row in original_assigned
        }
        changed_ids = {
            int(row["obs_row"]): str(row["peak_id"]) for row in assigned_changed
        }
        self.assertEqual(changed_ids, original_ids)

        core_fields = (
            "frame_a",
            "frame_b",
            "peak_a_id",
            "peak_b_id",
            "delta_two_theta_deg",
            "location_similarity",
            "area_similarity",
        )

        def core(values: list[dict[str, object]]) -> list[tuple[object, ...]]:
            return sorted(tuple(row[field] for field in core_fields) for row in values)

        self.assertEqual(core(changed_pairs), core(self.pairs))


class PerAnchorPeakFrameSlotMapTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = [
            peak_row(2, 5.00, 2.0, 201, azim_deg=-10.0),
            peak_row(2, 8.00, 4.0, 202, azim_deg=20.0),
            peak_row(10, 5.05, 1.0, 1001),
            peak_row(10, 12.0, 8.0, 1002),
            peak_row(10, 50.0, 16.0, 1003),
            peak_row(50, 5.02, 2.0, 5001),
        ]
        self.peaks, self.peaks_by_frame = assign_and_group(rows)
        self.registry = [
            {"frame": 2, "pressure_GPa": 3.5, "peak_count": 2},
            {"frame": 10, "pressure_GPa": 6.0, "peak_count": 3},
            {"frame": 50, "pressure_GPa": 12.0, "peak_count": 1},
            {"frame": 99, "pressure_GPa": 20.0, "peak_count": 0},
        ]
        (
            self.layout,
            self.position_grid,
            self.area_grid,
        ) = correlations.build_frame_slot_grids(
            self.peaks_by_frame,
            self.registry,
        )

    def test_layout_keeps_every_registered_frame_and_pads_missing_slots(self) -> None:
        self.assertEqual(self.position_grid.shape, (4, 3))
        self.assertEqual([row["frame"] for row in self.layout], [2, 10, 50, 99])
        self.assertTrue(np.all(np.isfinite(self.position_grid[1, :])))
        self.assertTrue(np.isfinite(self.position_grid[2, 0]))
        self.assertTrue(np.all(np.isnan(self.position_grid[2, 1:])))
        self.assertTrue(np.all(np.isnan(self.position_grid[3, :])))
        self.assertEqual([row["peak_count"] for row in self.layout], [2, 3, 1, 0])

    def test_anchor_frame_and_zero_peak_frame_are_blank_but_real_zero_is_finite(
        self,
    ) -> None:
        anchor = next(row for row in self.peaks if row["peak_id"] == "p2,1")
        location, area = correlations.build_anchor_peak_frame_slot_matrices(
            anchor,
            self.layout,
            self.position_grid,
            self.area_grid,
            tolerance=0.1,
        )

        self.assertTrue(np.all(np.isnan(location[0, :])))
        self.assertTrue(np.all(np.isnan(area[0, :])))
        self.assertTrue(np.all(np.isnan(location[3, :])))
        self.assertTrue(np.all(np.isnan(area[3, :])))
        self.assertEqual(np.count_nonzero(np.isfinite(location)), 4)
        self.assertEqual(np.count_nonzero(np.isfinite(area)), 4)
        self.assertEqual(location[1, 1], 0.0)
        self.assertTrue(np.isfinite(location[1, 1]))
        self.assertTrue(np.all(np.isnan(location[2, 1:])))
        np.testing.assert_allclose(area[1, :], [0.5, 0.25, 0.125])

    def test_all_anchors_have_twice_the_canonical_cross_frame_cell_count(self) -> None:
        total_finite = 0
        matrices: dict[str, np.ndarray] = {}
        for anchor in self.peaks:
            location, _ = correlations.build_anchor_peak_frame_slot_matrices(
                anchor,
                self.layout,
                self.position_grid,
                self.area_grid,
                tolerance=0.1,
            )
            matrices[str(anchor["peak_id"])] = location
            total_finite += int(np.count_nonzero(np.isfinite(location)))

        canonical = correlations.build_cross_frame_pair_rows(
            self.peaks_by_frame,
            tolerance=0.1,
        )
        self.assertEqual(len(canonical), 11)
        self.assertEqual(total_finite, 2 * len(canonical))
        self.assertEqual(matrices["p2,1"][1, 0], matrices["p10,1"][0, 0])

    def test_invalid_registry_or_local_slot_layout_is_rejected(self) -> None:
        duplicate_registry = [*self.registry, dict(self.registry[0])]
        with self.assertRaises(ValueError):
            correlations.build_frame_slot_grids(
                self.peaks_by_frame,
                duplicate_registry,
            )

        broken = copy.deepcopy(self.peaks_by_frame)
        broken[10][1]["local_peak_index"] = 4
        with self.assertRaises(ValueError):
            correlations.build_frame_slot_grids(broken, self.registry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
