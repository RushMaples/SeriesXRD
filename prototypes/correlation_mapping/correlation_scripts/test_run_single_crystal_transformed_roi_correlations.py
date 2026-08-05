#!/usr/bin/env python3
"""Focused contracts for the transformed single-crystal ROI runner."""

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
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "uotexrd-transformed-roi-test-mpl"),
)

import nonlinear_intensity_preprocessing as nonlinear  # noqa: E402
import run_single_crystal_transformed_roi_correlations as runner  # noqa: E402
import generate_single_crystal_all_peak_correlation_waterfalls as waterfall  # noqa: E402
from all_peak_frame_correlations import (  # noqa: E402
    assign_local_peak_ids,
    build_anchor_peak_frame_slot_matrices,
    build_frame_slot_grids,
    group_peaks_by_frame,
)


def fixture_observation(**overrides: str) -> dict[str, str]:
    row = {
        "frame": "0",
        "obs_row": "1",
        "q": "2.0",
        "d_A": "3.0",
        "azim_deg": "0.0",
        "halfwidth_q": "0.4",
        "halfwidth_azim_deg": "10.0",
        "track": "7",
        "matched_d_A": "3.1",
    }
    row.update(overrides)
    return row


class TransformedSingleCrystalROITests(unittest.TestCase):
    def test_exact_geometric_roi_sideband_and_noise(self) -> None:
        q_array = np.repeat(
            np.asarray([1.0, 1.5, 2.0, 2.5, 3.0])[:, None],
            5,
            axis=1,
        )
        chi = np.zeros((5, 5), dtype=float)
        raw = np.asarray(
            [
                [1, 1, 1, 1, 1],
                [5, 5, 5, 5, 5],
                [10, 20, 30, 40, 50],
                [7, 7, 7, 7, 7],
                [1, 1, 1, 1, 1],
            ],
            dtype=float,
        )
        detector_mask = np.zeros_like(raw, dtype=bool)
        frame_mask = np.ones_like(raw, dtype=bool)
        frame_mask[2] = False
        frame_mask[2, 0] = True

        item = runner.extract_observation_pixels(
            raw,
            detector_mask,
            frame_mask,
            q_array,
            chi,
            fixture_observation(),
            frame=0,
            raw_tiff=Path("fixture.tif"),
            exposure_s=2.0,
        )

        # Sideband rows are five 5s and five 7s: median=6, MAD=1.
        self.assertEqual(item.sideband_pixels, 10)
        self.assertEqual(item.background_median_counts, 6.0)
        self.assertEqual(item.sideband_mad_counts, 1.0)
        self.assertAlmostEqual(
            item.sideband_noise_counts_per_s_per_pixel,
            1.4826 / 2.0,
            places=15,
        )
        # First ROI pixel is excluded by frame mask.
        np.testing.assert_allclose(
            item.positive_excess_rate,
            np.asarray([7.0, 12.0, 17.0, 22.0]),
        )
        self.assertEqual(item.effective_roi_pixels, 4)
        self.assertEqual(item.positive_excess_pixels, 4)
        self.assertEqual(item.zero_excess_pixels, 0)

    def test_invalid_detector_and_negative_sentinel_are_excluded(self) -> None:
        q_array = np.repeat(
            np.asarray([1.5, 2.0, 2.5])[:, None],
            5,
            axis=1,
        )
        chi = np.zeros((3, 5), dtype=float)
        raw = np.asarray(
            [
                [5, 5, 5, 5, 5],
                [20, -1, 30, 40, 50],
                [7, 7, 7, 7, 7],
            ],
            dtype=float,
        )
        detector_mask = np.zeros_like(raw, dtype=bool)
        detector_mask[1, 2] = True
        frame_mask = np.ones_like(raw, dtype=bool)
        frame_mask[1] = False

        item = runner.extract_observation_pixels(
            raw,
            detector_mask,
            frame_mask,
            q_array,
            chi,
            fixture_observation(),
            frame=0,
            raw_tiff=Path("fixture.tif"),
            exposure_s=2.0,
        )

        # -1 is rejected by raw>=0 and 30 is rejected by detector mask.
        np.testing.assert_allclose(
            item.positive_excess_rate,
            np.asarray([7.0, 17.0, 22.0]),
        )
        self.assertEqual(item.effective_roi_pixels, 3)

    def test_global_scale_and_noise_are_fitted_once_across_observations(self) -> None:
        base = fixture_observation()
        first = runner.PixelObservation(
            source=base,
            frame=0,
            raw_tiff=Path("a.tif"),
            exposure_s=2.0,
            background_median_counts=5.0,
            positive_excess_rate=np.asarray([0.0, 1.0, 2.0]),
            raw_excess_counts=6.0,
            geometric_roi_pixels=3,
            effective_roi_pixels=3,
            detector_or_frame_masked_roi_pixels=0,
            raw_zero_roi_pixels=0,
            positive_excess_pixels=2,
            zero_excess_pixels=1,
            sideband_pixels=10,
            sideband_median_counts=5.0,
            sideband_mad_counts=1.0,
            sideband_noise_counts_per_s_per_pixel=0.5,
        )
        second = runner.PixelObservation(
            **{
                **first.__dict__,
                "source": fixture_observation(obs_row="2"),
                "positive_excess_rate": np.asarray([3.0, 4.0]),
                "effective_roi_pixels": 2,
                "positive_excess_pixels": 2,
                "zero_excess_pixels": 0,
                "sideband_noise_counts_per_s_per_pixel": 1.5,
            }
        )
        spec, estimate, noise = runner.fit_transform_from_pixels(
            [first, second],
            nonlinear.LOG_SQUARED,
            scale_quantile=0.5,
        )

        self.assertEqual(estimate.total_slots, 5)
        self.assertEqual(estimate.positive_finite_slots, 4)
        self.assertEqual(spec.scale, 2.5)
        self.assertEqual(noise, 1.0)
        self.assertAlmostEqual(spec.epsilon, (1.0 / 2.5) ** 2, places=15)

    def test_rows_preserve_every_peak_and_ignore_track_for_identity(self) -> None:
        metadata = {
            0: {
                "orientation": "orientation_10deg",
                "pressure_GPa": 1.0,
                "included_whole_pattern": 1,
                "exclusion_reason": "",
            }
        }
        payloads = []
        for obs_row, rates in (("1", [0.0, 1.0]), ("2", [1.0, 1.0])):
            payloads.append(
                runner.PixelObservation(
                    source=fixture_observation(obs_row=obs_row),
                    frame=0,
                    raw_tiff=Path("fixture.tif"),
                    exposure_s=1.0,
                    background_median_counts=0.0,
                    positive_excess_rate=np.asarray(rates),
                    raw_excess_counts=float(sum(rates)),
                    geometric_roi_pixels=2,
                    effective_roi_pixels=2,
                    detector_or_frame_masked_roi_pixels=0,
                    raw_zero_roi_pixels=0,
                    positive_excess_pixels=int(np.count_nonzero(rates)),
                    zero_excess_pixels=2 - int(np.count_nonzero(rates)),
                    sideband_pixels=5,
                    sideband_median_counts=0.0,
                    sideband_mad_counts=1.0,
                    sideband_noise_counts_per_s_per_pixel=1.0,
                )
            )
        spec = nonlinear.make_roi_transform_spec(
            nonlinear.LOG_SQUARED,
            scale=1.0,
            noise_floor=1.0,
        )
        rows, _ = runner.build_transformed_rows(payloads, spec, metadata)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["track"] for row in rows], [7, 7])
        self.assertEqual(
            [row["integrated_area"] for row in rows],
            [0.5, 1.0],
        )
        peaks = assign_local_peak_ids(rows, "single_crystal")
        self.assertEqual([row["peak_id"] for row in peaks], ["p0,1", "p0,2"])

    def test_all_peak_map_compares_every_other_frame_peak(self) -> None:
        raw = [
            {
                "frame": 0,
                "obs_row": 1,
                "track": 4,
                "two_theta_deg": 10.0,
                "azim_deg": 1.0,
                "integrated_area": 0.5,
            },
            {
                "frame": 0,
                "obs_row": 2,
                "track": 4,
                "two_theta_deg": 11.0,
                "azim_deg": 2.0,
                "integrated_area": 1.0,
            },
            {
                "frame": 4,
                "obs_row": 3,
                "track": 99,
                "two_theta_deg": 10.03,
                "azim_deg": 3.0,
                "integrated_area": 0.25,
            },
        ]
        peaks = assign_local_peak_ids(raw, "single_crystal")
        grouped = group_peaks_by_frame(peaks)
        layout, positions, areas = build_frame_slot_grids(
            grouped,
            [
                {"frame": 0, "scan": "a", "pressure_GPa": 1.0},
                {"frame": 4, "scan": "b", "pressure_GPa": 2.0},
            ],
        )
        location, area = build_anchor_peak_frame_slot_matrices(
            peaks[0], layout, positions, areas, 0.06
        )
        self.assertEqual(location.shape, (2, 2))
        self.assertTrue(np.all(np.isnan(location[0])))
        self.assertAlmostEqual(location[1, 0], 0.5)
        self.assertAlmostEqual(area[1, 0], 0.5)
        self.assertTrue(np.isnan(location[1, 1]))

    def test_cli_accepts_log_mode_and_output(self) -> None:
        parsed = runner.parse_args(
            [
                "--mode",
                nonlinear.LOG_SQUARED,
                "--out-dir",
                "result",
                "--no-plots",
            ]
        )
        self.assertEqual(parsed.mode, nonlinear.LOG_SQUARED)
        self.assertEqual(parsed.out_dir, Path("result"))
        self.assertTrue(parsed.no_plots)

    def test_waterfall_support_lanes_preserve_overlapping_peaks(self) -> None:
        peaks = [
            {"peak_id": "p0,1", "q_A^-1": "4.0", "halfwidth_q_A^-1": "0.2"},
            {"peak_id": "p0,2", "q_A^-1": "4.1", "halfwidth_q_A^-1": "0.2"},
            {"peak_id": "p0,3", "q_A^-1": "5.0", "halfwidth_q_A^-1": "0.1"},
        ]
        lanes = waterfall.assign_interval_lanes(peaks)
        self.assertNotEqual(lanes[0], lanes[1])
        self.assertEqual(lanes[0], lanes[2])
        left, right = waterfall.peak_support(peaks[0])
        self.assertLess(left, right)


if __name__ == "__main__":
    unittest.main()
