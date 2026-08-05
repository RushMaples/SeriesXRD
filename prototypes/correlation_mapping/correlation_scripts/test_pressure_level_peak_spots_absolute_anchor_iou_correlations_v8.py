#!/usr/bin/env python3
"""Numerical contracts for the directional absolute-q v8 ROI definition."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Importing the production module imports matplotlib.  Keep its cache in a
# writable temporary location so the tests do not depend on a user's home.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "uotexrd-v8-test-matplotlib"),
)

import pressure_level_peak_spots_absolute_anchor_iou_correlations_v8 as correlations  # noqa: E402


class AnchorRestrictedIntegratedIoUTests(unittest.TestCase):
    """Contracts for integration on the anchor's absolute-q support only."""

    def test_absolute_disjoint_intervals_have_zero_similarity(self) -> None:
        coordinate = np.linspace(0.0, 5.0, 5001)
        anchor = np.maximum(1.0 - np.abs(coordinate - 1.0), 0.0)
        target = np.maximum(1.0 - np.abs(coordinate - 4.0), 0.0)

        score = correlations.anchor_restricted_integrated_iou(
            coordinate,
            anchor,
            target,
            anchor_support=(0.0, 2.0),
            target_support=(3.0, 5.0),
        )

        self.assertEqual(float(score), 0.0)

    def test_partial_absolute_overlap_uses_only_the_common_signal(self) -> None:
        coordinate = np.linspace(0.0, 3.0, 3001)
        anchor = np.maximum(1.0 - np.abs(coordinate - 1.0), 0.0)
        target = np.maximum(1.0 - np.abs(coordinate - 2.0), 0.0)

        score = correlations.anchor_restricted_integrated_iou(
            coordinate,
            anchor,
            target,
            anchor_support=(0.0, 2.0),
            target_support=(1.0, 3.0),
        )

        # On anchor support [0,2], integral(min)=1/4 and
        # integral(max)=5/4 for these two piecewise-linear tents.
        self.assertAlmostEqual(float(score), 1.0 / 5.0, places=12)
        self.assertGreater(float(score), 0.0)
        self.assertLess(float(score), 1.0)

    def test_swapping_anchor_can_change_directional_score(self) -> None:
        coordinate = np.linspace(0.0, 4.0, 4001)
        broad = np.maximum(1.0 - np.abs(coordinate - 2.0) / 2.0, 0.0)
        narrow = np.maximum(1.0 - np.abs(coordinate - 2.0), 0.0)

        broad_to_narrow = correlations.anchor_restricted_integrated_iou(
            coordinate,
            broad,
            narrow,
            anchor_support=(0.0, 4.0),
            target_support=(1.0, 3.0),
        )
        narrow_to_broad = correlations.anchor_restricted_integrated_iou(
            coordinate,
            narrow,
            broad,
            anchor_support=(1.0, 3.0),
            target_support=(0.0, 4.0),
        )

        self.assertAlmostEqual(float(broad_to_narrow), 1.0 / 2.0, places=12)
        self.assertAlmostEqual(float(narrow_to_broad), 2.0 / 3.0, places=12)
        self.assertNotAlmostEqual(
            float(broad_to_narrow),
            float(narrow_to_broad),
            places=12,
        )

    def test_target_values_outside_declared_support_are_masked(self) -> None:
        coordinate = np.linspace(0.0, 5.0, 5001)
        anchor = np.maximum(1.0 - np.abs(coordinate - 1.0), 0.0)

        # These values deliberately duplicate the anchor inside [0,2].
        # They must not contribute because the target's physical support is
        # declared to be [3,5].
        target_values_with_out_of_support_leakage = anchor.copy()
        score = correlations.anchor_restricted_integrated_iou(
            coordinate,
            anchor,
            target_values_with_out_of_support_leakage,
            anchor_support=(0.0, 2.0),
            target_support=(3.0, 5.0),
        )

        self.assertEqual(float(score), 0.0)

    def test_identical_profile_and_support_have_unit_similarity(self) -> None:
        coordinate = np.linspace(0.0, 2.0, 2001)
        profile = np.maximum(1.0 - np.abs(coordinate - 1.0), 0.0)

        score = correlations.anchor_restricted_integrated_iou(
            coordinate,
            profile,
            profile.copy(),
            anchor_support=(0.0, 2.0),
            target_support=(0.0, 2.0),
        )

        self.assertAlmostEqual(float(score), 1.0, places=12)

    def test_amplitude_scale_is_integrated_min_over_max(self) -> None:
        coordinate = np.linspace(0.0, 2.0, 2001)
        anchor = np.maximum(1.0 - np.abs(coordinate - 1.0), 0.0)
        scale = 0.35

        score = correlations.anchor_restricted_integrated_iou(
            coordinate,
            anchor,
            scale * anchor,
            anchor_support=(0.0, 2.0),
            target_support=(0.0, 2.0),
        )

        self.assertAlmostEqual(float(score), scale, places=12)

    def test_two_zero_profiles_are_conservatively_numeric_zero(self) -> None:
        coordinate = np.linspace(0.0, 2.0, 21)
        zero = np.zeros_like(coordinate)

        score = correlations.anchor_restricted_integrated_iou(
            coordinate,
            zero,
            zero.copy(),
            anchor_support=(0.0, 2.0),
            target_support=(0.0, 2.0),
        )

        self.assertEqual(float(score), 0.0)
        self.assertTrue(np.isfinite(score))

    def test_one_zero_profile_is_numeric_zero_in_either_direction(self) -> None:
        coordinate = np.linspace(0.0, 2.0, 2001)
        positive = np.maximum(1.0 - np.abs(coordinate - 1.0), 0.0)
        zero = np.zeros_like(coordinate)

        for anchor, target in ((zero, positive), (positive, zero)):
            with self.subTest(anchor_is_zero=not np.any(anchor)):
                score = correlations.anchor_restricted_integrated_iou(
                    coordinate,
                    anchor,
                    target,
                    anchor_support=(0.0, 2.0),
                    target_support=(0.0, 2.0),
                )
                self.assertEqual(float(score), 0.0)
                self.assertTrue(np.isfinite(score))

    def test_piecewise_crossing_profiles_use_continuous_min_and_max(self) -> None:
        coordinate = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        anchor = np.asarray([0.0, 1.0, 1.0, 0.0, 0.0])
        target = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0])

        score = correlations.anchor_restricted_integrated_iou(
            coordinate,
            anchor,
            target,
            anchor_support=(0.0, 4.0),
            target_support=(0.0, 4.0),
        )

        # For the piecewise-linear fields, integral(min)=1 and
        # integral(max)=3.
        self.assertAlmostEqual(float(score), 1.0 / 3.0, places=12)


class AnchorMapBlankVersusZeroTests(unittest.TestCase):
    def test_detected_zero_is_finite_while_structural_cells_remain_blank(
        self,
    ) -> None:
        pressures = correlations.v6.PRESSURES_DESCENDING
        anchor_pressure = 5.81
        target_pressure = 33.6
        anchor_row = pressures.index(anchor_pressure)
        target_row = pressures.index(target_pressure)
        points = [
            {"point_uid": "anchor", "pressure_gpa": anchor_pressure},
            {"point_uid": "same-pressure", "pressure_gpa": anchor_pressure},
            {"point_uid": "detected-zero", "pressure_gpa": target_pressure},
        ]
        slot_lookup = {
            "anchor": (anchor_row, 0),
            "same-pressure": (anchor_row, 1),
            "detected-zero": (target_row, 0),
        }
        pair_matrix = np.full((3, 3), np.nan, dtype=float)
        pair_matrix[0, 2] = 0.0
        pair_matrix[2, 0] = 0.0

        builder = getattr(
            correlations,
            "build_anchor_matrix",
            correlations.v6.build_anchor_matrix,
        )
        result = builder(
            0,
            points,
            pair_matrix,
            slot_lookup,
            2,
        )

        self.assertEqual(float(result[target_row, 0]), 0.0)
        self.assertTrue(np.isfinite(result[target_row, 0]))
        self.assertTrue(np.isnan(result[target_row, 1]))
        self.assertTrue(np.all(np.isnan(result[anchor_row])))


if __name__ == "__main__":
    unittest.main()
