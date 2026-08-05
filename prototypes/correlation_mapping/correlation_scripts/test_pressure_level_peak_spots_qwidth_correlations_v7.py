#!/usr/bin/env python3
"""Numerical contracts for the v7 spots-channel q-width correlations."""

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
    str(Path(tempfile.gettempdir()) / "uotexrd-v7-test-matplotlib"),
)

import pressure_level_peak_spots_qwidth_correlations_v7 as correlations  # noqa: E402


class CoordinateConversionTests(unittest.TestCase):
    def test_q_two_theta_round_trip(self) -> None:
        two_theta = np.asarray([0.0, 3.5, 9.8878, 18.3, 32.0], dtype=float)

        restored = correlations.two_theta_from_q(
            correlations.q_from_two_theta(two_theta)
        )

        np.testing.assert_allclose(
            restored,
            two_theta,
            rtol=0.0,
            atol=1.0e-12,
        )


class RelativeProfileExtractionTests(unittest.TestCase):
    def test_recentering_removes_absolute_q_location(self) -> None:
        width = 0.08
        local_native_coordinate = np.linspace(-2.0, 2.0, 401)
        relative_grid = np.linspace(-0.6, 0.6, 61)

        def local_shape(coordinate: np.ndarray) -> np.ndarray:
            return (
                np.exp(-0.5 * (coordinate / 0.24) ** 2)
                * (1.0 + 0.15 * coordinate)
            )

        first_center = 1.20
        second_center = 2.40
        first_q = first_center + width * local_native_coordinate
        second_q = second_center + width * local_native_coordinate
        intensity = local_shape(local_native_coordinate)

        first = correlations.extract_relative_profile(
            first_q,
            intensity,
            q_center=first_center,
            q_width=width,
            relative_grid=relative_grid,
            preserve_two_theta_area=False,
        )
        second = correlations.extract_relative_profile(
            second_q,
            intensity,
            q_center=second_center,
            q_width=width,
            relative_grid=relative_grid,
            preserve_two_theta_area=False,
        )

        np.testing.assert_allclose(first, second, rtol=0.0, atol=1.0e-12)

    def test_negative_spots_residuals_are_clipped_to_zero(self) -> None:
        relative_grid = np.linspace(-1.0, 1.0, 5)
        q_center = 1.0
        q_width = 0.2
        q_axis = q_center + relative_grid * q_width
        intensity = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])

        clipped = correlations.extract_relative_profile(
            q_axis,
            intensity,
            q_center=q_center,
            q_width=q_width,
            relative_grid=relative_grid,
            clip_negative=True,
            preserve_two_theta_area=False,
        )
        retained = correlations.extract_relative_profile(
            q_axis,
            intensity,
            q_center=q_center,
            q_width=q_width,
            relative_grid=relative_grid,
            clip_negative=False,
            preserve_two_theta_area=False,
        )

        np.testing.assert_array_equal(
            clipped,
            np.asarray([0.0, 0.0, 0.0, 1.0, 2.0]),
        )
        np.testing.assert_allclose(retained, intensity, rtol=0.0, atol=0.0)

    def test_jacobian_preserves_physical_two_theta_area(self) -> None:
        relative_grid = np.linspace(-0.6, 0.6, 2001)
        q_center = 2.0
        q_width = 0.15
        amplitude = 2.75
        q_axis = q_center + relative_grid * q_width
        intensity = np.full_like(q_axis, amplitude)

        area_density_on_u = correlations.extract_relative_profile(
            q_axis,
            intensity,
            q_center=q_center,
            q_width=q_width,
            relative_grid=relative_grid,
            preserve_two_theta_area=True,
        )
        two_theta = correlations.two_theta_from_q(q_axis)
        actual_area = float(np.trapezoid(area_density_on_u, relative_grid))
        expected_area = float(amplitude * (two_theta[-1] - two_theta[0]))

        self.assertAlmostEqual(actual_area, expected_area, places=9)


class IntegratedProfileIoUTests(unittest.TestCase):
    def test_identical_profiles_have_unit_iou(self) -> None:
        coordinate = np.linspace(-0.6, 0.6, 101)
        profile = np.exp(-0.5 * (coordinate / 0.2) ** 2)

        score = correlations.profile_integrated_iou(
            profile,
            profile.copy(),
            coordinate,
        )

        self.assertAlmostEqual(float(score), 1.0, places=12)

    def test_amplitude_scale_is_integrated_min_over_max(self) -> None:
        coordinate = np.linspace(-0.6, 0.6, 101)
        profile = np.exp(-0.5 * (coordinate / 0.2) ** 2)
        scale = 0.35

        score = correlations.profile_integrated_iou(
            profile,
            scale * profile,
            coordinate,
        )

        self.assertAlmostEqual(float(score), scale, places=12)

    def test_partial_overlap_uses_continuous_trapezoid_integrals(self) -> None:
        coordinate = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        left = np.asarray([0.0, 1.0, 1.0, 0.0, 0.0])
        right = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0])

        score = correlations.profile_integrated_iou(left, right, coordinate)

        # Integral(min)=1 and integral(max)=3 for these piecewise-linear fields.
        self.assertAlmostEqual(float(score), 1.0 / 3.0, places=12)
        self.assertGreater(float(score), 0.0)
        self.assertLess(float(score), 1.0)

    def test_negative_input_is_rejected(self) -> None:
        coordinate = np.asarray([0.0, 1.0, 2.0])

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            correlations.profile_integrated_iou(
                np.asarray([0.0, -0.1, 1.0]),
                np.asarray([0.0, 0.5, 1.0]),
                coordinate,
            )


class NativeSamplingOptimizationTests(unittest.TestCase):
    def test_selects_narrowest_candidate_meeting_sampling_contract(self) -> None:
        q_axis = np.linspace(-1.0, 1.0, 9)
        spots_by_frame = {
            7: (q_axis, np.ones_like(q_axis), Path("synthetic-frame-7.xy"))
        }
        observations = [{"frame": 7, "q": 0.0, "q_width": 1.0}]

        rows, selected, audit = correlations.native_sampling_factor_optimization(
            observations,
            spots_by_frame,
            candidates=(0.25, 0.4, 0.6),
        )

        self.assertEqual([row["native_points_min"] for row in rows], [3, 3, 5])
        self.assertEqual(
            [row["passes_90pct_at_least_4_points"] for row in rows],
            [False, False, True],
        )
        self.assertAlmostEqual(selected, 0.6)
        self.assertAlmostEqual(audit["selected_half_width_factor"], 0.6)


if __name__ == "__main__":
    unittest.main()
