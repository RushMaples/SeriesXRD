#!/usr/bin/env python3
"""Focused semantic tests for relationship-first waterfall generation."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import generate_relationship_waterfalls as relationship


class MatrixTests(unittest.TestCase):
    def test_lower_triangle_is_symmetrized_without_inventing_missing_pairs(self) -> None:
        lower = np.array(
            [
                [np.nan, np.nan, np.nan],
                [0.2, np.nan, np.nan],
                [np.nan, 0.8, np.nan],
            ]
        )

        full = relationship.symmetrize_lower(lower, diagonal=1.0)

        self.assertEqual(full[0, 0], 1.0)
        self.assertEqual(full[0, 1], 0.2)
        self.assertEqual(full[1, 0], 0.2)
        self.assertEqual(full[1, 2], 0.8)
        self.assertTrue(np.isnan(full[0, 2]))


class RankingTests(unittest.TestCase):
    def test_topk_uses_signed_correlation_and_excludes_self(self) -> None:
        rows = []
        for a, b, score in ((0, 1, -0.1), (0, 2, 0.8), (1, 2, 0.3)):
            rows.append(
                {
                    "dataset": "demo",
                    "channel": "spots",
                    "scan": "scan001",
                    "frame_a": a,
                    "frame_b": b,
                    "pressure_a_GPa": float(a + 1),
                    "pressure_b_GPa": float(b + 1),
                    "pressure_gap_GPa": float(abs(a - b)),
                    "acquisition_signature_a": "a",
                    "acquisition_signature_b": "a",
                    "acquisition_protocol_changed": 0,
                    "correlation": score,
                }
            )
        pairs = pd.DataFrame(rows)

        top, medoids, _ = relationship.rank_frame_neighbors(pairs, top_k=1)

        anchor_zero = top[top["anchor_frame"].eq(0)].iloc[0]
        self.assertEqual(anchor_zero["neighbor_frame"], 2)
        self.assertAlmostEqual(anchor_zero["correlation"], 0.8)
        self.assertTrue((top["anchor_frame"] != top["neighbor_frame"]).all())
        self.assertEqual(len(medoids), 1)


class WindowTests(unittest.TestCase):
    def test_interval_matrix_preserves_structural_missingness(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "window_index": 2,
                    "p_low_GPa": 1.0,
                    "p_high_GPa": 2.0,
                    "value": 0.7,
                },
                {
                    "window_index": 7,
                    "p_low_GPa": 2.0,
                    "p_high_GPa": 3.0,
                    "value": -0.2,
                },
            ]
        )

        result = relationship.interval_matrix(
            frame, [2, 7], [(1.0, 2.0), (2.0, 3.0)], "value"
        )

        self.assertAlmostEqual(result[0, 0], 0.7)
        self.assertAlmostEqual(result[1, 1], -0.2)
        self.assertTrue(np.isnan(result[0, 1]))
        self.assertTrue(np.isnan(result[1, 0]))


class PeakEvidenceTests(unittest.TestCase):
    def test_selection_requires_support_and_keeps_high_and_low_examples(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "map_key": "demo",
                    "scan": "scan001",
                    "frame_a": 1,
                    "frame_b": 0,
                    "whole_pattern_correlation": 0.9,
                    "location_support_tracks": 2,
                },
                {
                    "map_key": "demo",
                    "scan": "scan001",
                    "frame_a": 2,
                    "frame_b": 0,
                    "whole_pattern_correlation": 0.1,
                    "location_support_tracks": 2,
                },
                {
                    "map_key": "demo",
                    "scan": "scan001",
                    "frame_a": 3,
                    "frame_b": 0,
                    "whole_pattern_correlation": 0.99,
                    "location_support_tracks": 1,
                },
            ]
        )

        selected = relationship.select_peak_evidence_pairs(frame)

        self.assertEqual(len(selected), 2)
        self.assertEqual(
            set(selected["whole_pattern_correlation"].round(2)), {0.1, 0.9}
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
