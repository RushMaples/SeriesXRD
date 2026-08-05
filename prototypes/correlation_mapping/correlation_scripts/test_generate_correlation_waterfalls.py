#!/usr/bin/env python3
"""Focused tests for correlation-waterfall data semantics."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import generate_correlation_waterfalls as waterfalls


class MatrixTests(unittest.TestCase):
    def test_lower_triangle_is_symmetrized_without_inventing_diagonal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame", "a", "b", "c"])
                writer.writerow(["a", "", "", ""])
                writer.writerow(["b", "0.2", "", ""])
                writer.writerow(["c", "0.8", "0.5", ""])
            matrix = waterfalls.read_lower_triangle_matrix(path)
        self.assertTrue(np.isnan(matrix.full[0, 0]))
        self.assertAlmostEqual(matrix.full[0, 1], 0.2)
        self.assertAlmostEqual(matrix.full[1, 0], 0.2)
        self.assertAlmostEqual(matrix.full[0, 2], 0.8)
        self.assertAlmostEqual(matrix.full[2, 1], 0.5)

    def test_medoid_uses_candidate_rows_and_pairwise_median(self) -> None:
        matrix = np.array(
            [
                [np.nan, 0.9, 0.8, 0.0],
                [0.9, np.nan, 0.95, 0.0],
                [0.8, 0.95, np.nan, 0.0],
                [0.0, 0.0, 0.0, np.nan],
            ]
        )
        candidates = np.array([True, True, True, False])
        self.assertEqual(waterfalls.choose_medoid(matrix, candidates), 1)


class FeatureTests(unittest.TestCase):
    def test_acf_color_transform_preserves_raw_endpoints(self) -> None:
        self.assertEqual(waterfalls.display_similarity(-1.0, "same-window"), 0.0)
        self.assertEqual(waterfalls.display_similarity(0.0, "same-window"), 0.5)
        self.assertEqual(waterfalls.display_similarity(1.0, "same-window"), 1.0)
        self.assertEqual(waterfalls.display_similarity(0.0, "same_window_acf"), 0.5)
        self.assertEqual(waterfalls.display_similarity(0.25, "area"), 0.25)

    def test_directional_acf_pair_reproduces_stored_orientation(self) -> None:
        base = np.array([1.0, -1.0, 0.5, -0.5])
        opposite = -base
        fingerprints: dict[tuple[int, int], np.ndarray | None] = {}
        for frame in range(3):
            for window in range(3):
                fingerprints[(frame, window)] = opposite.copy()
        fingerprints[(2, 1)] = base.copy()  # reference nominal
        fingerprints[(0, 2)] = base.copy()  # lower-index target wins at +1
        score, delta, target, partner, side = waterfalls.best_stored_direction_acf_pair(
            reference=2,
            target=0,
            window_index=1,
            fingerprints=fingerprints,
            window_count=3,
            neighbor=1,
        )
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(delta, 1)
        self.assertEqual(side, "target")
        np.testing.assert_allclose(target, base)
        np.testing.assert_allclose(partner, base)

    def test_area_profile_matches_positive_sideband_integral(self) -> None:
        x = np.linspace(0.0, 1.0, 101)
        y = np.ones_like(x)
        y[(x >= 0.45) & (x <= 0.55)] = 3.0
        pattern = type(
            "PatternLike",
            (),
            {"two_theta": x, "intensity": y},
        )()
        _, _, background, area = waterfalls.area_profile(
            pattern,
            y,
            center=0.5,
            roi_half_width=0.05,
            sideband_gap=0.02,
            sideband_width=0.08,
        )
        self.assertAlmostEqual(background, 1.0)
        self.assertGreater(area, 0.15)
        self.assertLess(area, 0.25)

    def test_frame_pair_ranking_excludes_zero_area_degenerate_feature(self) -> None:
        patterns = [
            type("PatternLike", (), {"label": "a", "pressure_gpa": 1.0})(),
            type("PatternLike", (), {"label": "b", "pressure_gpa": 2.0})(),
        ]
        degenerate = np.array([[np.nan, 1.0], [1.0, np.nan]])
        valid = np.array([[np.nan, 0.6], [0.6, np.nan]])
        rows = waterfalls.frame_pair_summary(
            "peak_area",
            [
                ("peak_0001", degenerate, np.array([False, False]), np.array([True, True])),
                ("peak_0002", valid, np.array([True, True]), np.array([True, True])),
            ],
            patterns,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["shared_valid_map_count"], 1)
        self.assertEqual(rows[0]["highest_shared_valid_similarity_features"], "peak_0002:0.600")
        self.assertEqual(rows[0]["peak_presence_jaccard"], 1.0)


if __name__ == "__main__":
    unittest.main()
