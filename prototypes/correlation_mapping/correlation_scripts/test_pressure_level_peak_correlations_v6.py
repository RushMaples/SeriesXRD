#!/usr/bin/env python3
"""Focused numerical contracts for pressure-level peak correlations v6.

These tests deliberately cover only the small public numerical API.  In
particular, an observation profile's ``values`` are its measured (not yet
exposure-corrected) sparse density values.  Aggregation must:

1. divide each observation by its own ``measurement_scale``;
2. sum blobs that belong to the same physical frame; and
3. average the resulting profiles over unique physical frames.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pressure_level_peak_correlations_v6 as correlations  # noqa: E402


class AngleAndLocationTests(unittest.TestCase):
    def test_circular_delta_uses_shortest_signed_angular_distance(self) -> None:
        actual = correlations.circular_delta_deg(
            np.asarray([179.0, -179.0, 10.0]),
            -179.0,
        )

        np.testing.assert_allclose(
            actual,
            np.asarray([-2.0, 0.0, -171.0]),
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            float(correlations.circular_delta_deg(-179.0, 179.0)),
            2.0,
        )

    def test_location_similarity_is_linear_clipped_and_symmetric(self) -> None:
        self.assertAlmostEqual(
            float(correlations.location_similarity(10.0, 10.0)),
            1.0,
        )
        self.assertAlmostEqual(
            float(correlations.location_similarity(10.0, 10.03)),
            0.5,
            places=12,
        )
        self.assertAlmostEqual(
            float(correlations.location_similarity(10.03, 10.0)),
            0.5,
            places=12,
        )
        self.assertEqual(
            float(correlations.location_similarity(10.0, 10.12)),
            0.0,
        )


class EllipticalKernelTests(unittest.TestCase):
    def test_discrete_epanechnikov_kernel_integrates_to_one(self) -> None:
        indices, density, cell_area = (
            correlations.rasterize_epanechnikov_ellipse(
                q_center=1.0,
                azim_center_deg=23.4,
                h_q=0.004,
                h_azim_deg=0.35,
            )
        )

        self.assertGreater(indices.size, 0)
        self.assertEqual(indices.ndim, 1)
        self.assertEqual(density.shape, indices.shape)
        self.assertTrue(np.issubdtype(indices.dtype, np.integer))
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertTrue(np.all(np.isfinite(density)))
        self.assertTrue(np.all(density >= 0.0))
        self.assertGreater(float(cell_area), 0.0)
        self.assertAlmostEqual(
            float(np.sum(density, dtype=float) * cell_area),
            1.0,
            places=12,
        )

    def test_kernel_wraps_continuously_across_azimuth_seam(self) -> None:
        kwargs = {
            "q_center": 1.0,
            "h_q": 0.004,
            "h_azim_deg": 0.30,
            "q_min": 0.0,
            "q_step": 0.001,
            "azim_step_deg": 0.1,
            "n_azim": 3600,
        }
        positive_indices, positive_density, positive_cell_area = (
            correlations.rasterize_epanechnikov_ellipse(
                azim_center_deg=179.95,
                **kwargs,
            )
        )
        equivalent_indices, equivalent_density, equivalent_cell_area = (
            correlations.rasterize_epanechnikov_ellipse(
                azim_center_deg=-180.05,
                **kwargs,
            )
        )

        np.testing.assert_array_equal(positive_indices, equivalent_indices)
        np.testing.assert_allclose(
            positive_density,
            equivalent_density,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertEqual(positive_cell_area, equivalent_cell_area)

        azimuth_bins = positive_indices % kwargs["n_azim"]
        self.assertTrue(np.any(azimuth_bins <= 2))
        self.assertTrue(np.any(azimuth_bins >= kwargs["n_azim"] - 3))

    def test_formal_biweight_kernel_is_normalized_and_compact(self) -> None:
        indices, density, cell_area = correlations.rasterize_biweight_ellipse(
            q_center=3.5,
            azim_center_deg=-177.9,
            h_q=0.05,
            h_azim_deg=2.0,
        )

        self.assertGreater(indices.size, 0)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertTrue(np.all(density > 0.0))
        self.assertAlmostEqual(
            float(np.sum(density, dtype=float) * cell_area),
            1.0,
            places=12,
        )


class SparseProfileIoUTests(unittest.TestCase):
    def test_identical_profiles_have_unit_iou(self) -> None:
        indices = np.asarray([7, 11, 20], dtype=np.int64)
        values = np.asarray([1.0, 3.0, 2.0])

        actual = correlations.sparse_profile_iou(
            indices,
            values,
            indices,
            values,
            cell_area=0.25,
        )

        self.assertAlmostEqual(float(actual), 1.0, places=12)

    def test_disjoint_profiles_have_exactly_zero_iou(self) -> None:
        actual = correlations.sparse_profile_iou(
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([2.0, 1.0]),
            np.asarray([8, 9], dtype=np.int64),
            np.asarray([4.0, 3.0]),
            cell_area=0.5,
        )

        self.assertEqual(float(actual), 0.0)

    def test_amplitude_mismatch_uses_integrated_min_over_max(self) -> None:
        indices = np.asarray([2, 4], dtype=np.int64)
        actual = correlations.sparse_profile_iou(
            indices,
            np.asarray([1.0, 2.0]),
            indices,
            np.asarray([0.5, 1.0]),
            cell_area=0.25,
        )

        self.assertAlmostEqual(float(actual), 0.5, places=12)

    def test_partial_overlap_is_strictly_between_zero_and_one(self) -> None:
        actual = correlations.sparse_profile_iou(
            np.asarray([1, 2, 3], dtype=np.int64),
            np.ones(3),
            np.asarray([3, 4, 5], dtype=np.int64),
            np.ones(3),
            cell_area=1.0,
        )

        self.assertAlmostEqual(float(actual), 0.2, places=12)
        self.assertGreater(float(actual), 0.0)
        self.assertLess(float(actual), 1.0)


class FrameAggregationTests(unittest.TestCase):
    def test_blobs_sum_within_frame_then_unique_frames_are_averaged(self) -> None:
        observations = [
            {
                "frame": 10,
                "measurement_scale": 2.0,
                "indices": np.asarray([1, 2], dtype=np.int64),
                "values": np.asarray([4.0, 2.0]),
            },
            {
                "frame": 10,
                "measurement_scale": 2.0,
                "indices": np.asarray([2, 3], dtype=np.int64),
                "values": np.asarray([2.0, 6.0]),
            },
            {
                "frame": 20,
                "measurement_scale": 1.0,
                "indices": np.asarray([1, 3], dtype=np.int64),
                "values": np.asarray([4.0, 1.0]),
            },
        ]

        indices, values = correlations.aggregate_frame_profiles(observations)

        np.testing.assert_array_equal(
            indices,
            np.asarray([1, 2, 3], dtype=np.int64),
        )
        # Corrected frame 10 is [2, 2, 3]; corrected frame 20 is [4, 0, 1].
        # Their equal-frame mean is [3, 1, 2], not an observation-count mean.
        np.testing.assert_allclose(
            values,
            np.asarray([3.0, 1.0, 2.0]),
            rtol=0.0,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
