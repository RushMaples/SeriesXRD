#!/usr/bin/env python3
"""Focused tests for observational peak-evolution waterfall semantics."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import generate_peak_evolution_waterfalls as waterfalls


def pattern(
    label: str,
    pressure: float,
    series: str,
    branch: str,
    x: list[float] | None = None,
    y: list[float] | None = None,
) -> waterfalls.Pattern:
    return waterfalls.Pattern(
        label=label,
        pressure=pressure,
        series=series,
        branch=branch,
        path=Path(f"{label}.xy"),
        x=np.asarray([1.0, 2.0] if x is None else x, dtype=float),
        y=np.asarray([10.0, 20.0] if y is None else y, dtype=float),
    )


class SourceLabelTests(unittest.TestCase):
    def test_decomp_prefix_is_mapped_to_suite_label_and_branch(self) -> None:
        label, pressure, branch = waterfalls.source_label(Path("decomp-2p4GPa.xy"))

        self.assertEqual(label, "2.4GPa_decomp")
        self.assertAlmostEqual(pressure, 2.4)
        self.assertEqual(branch, "decompression")


class RawSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pattern = pattern(
            "frame",
            1.0,
            "Cell 29",
            "compression",
            x=[1.0, 2.0, 3.0, 4.0],
            y=[10.0, 21.0, 35.0, 80.0],
        )

    def test_full_slice_preserves_measured_samples_without_transform(self) -> None:
        excerpt = waterfalls.raw_slice(self.pattern, 1.5, 3.5)

        self.assertEqual(excerpt.coverage, "full")
        np.testing.assert_array_equal(excerpt.x, [2.0, 3.0])
        np.testing.assert_array_equal(excerpt.y, [21.0, 35.0])
        np.testing.assert_array_equal(self.pattern.y, [10.0, 21.0, 35.0, 80.0])

    def test_partial_and_unmeasured_coverage_are_distinct(self) -> None:
        left_partial = waterfalls.raw_slice(self.pattern, 0.5, 2.5)
        right_partial = waterfalls.raw_slice(self.pattern, 2.5, 4.5)
        outside = waterfalls.raw_slice(self.pattern, 5.0, 6.0)

        self.assertEqual(left_partial.coverage, "partial")
        np.testing.assert_array_equal(left_partial.x, [1.0, 2.0])
        np.testing.assert_array_equal(left_partial.y, [10.0, 21.0])
        self.assertEqual(right_partial.coverage, "partial")
        np.testing.assert_array_equal(right_partial.x, [3.0, 4.0])
        np.testing.assert_array_equal(right_partial.y, [35.0, 80.0])
        self.assertEqual(outside.coverage, "not_measured")
        self.assertEqual(outside.x.size, 0)
        self.assertEqual(outside.y.size, 0)


class ScaleTests(unittest.TestCase):
    def test_shared_scale_uses_one_domain_for_all_curves(self) -> None:
        first = np.array([10.0, 20.0, 30.0])
        second = np.array([20.0, 60.0, 110.0])

        minimum, gain = waterfalls.shared_scale([first, second])

        self.assertEqual(minimum, 10.0)
        self.assertAlmostEqual(gain, 0.72 / 100.0)
        self.assertAlmostEqual((first.max() - minimum) * gain, 0.144)
        self.assertAlmostEqual((second.max() - minimum) * gain, 0.72)
        self.assertAlmostEqual(
            (first[1] - minimum) * gain,
            (second[0] - minimum) * gain,
        )


class OrderingTests(unittest.TestCase):
    def test_decompression_follows_all_compression_rows_in_each_series(self) -> None:
        patterns = [
            pattern("29_decomp", 2.4, "Cell 29", "decompression"),
            pattern("14_high", 5.0, "Cell 14", "compression"),
            pattern("29_high", 7.4, "Cell 29", "compression"),
            pattern("29_low", 1.0, "Cell 29", "compression"),
            pattern("14_decomp", 2.0, "Cell 14", "decompression"),
            pattern("14_low", 0.7, "Cell 14", "compression"),
        ]

        groups = waterfalls.series_groups(patterns)

        self.assertEqual(list(groups), ["Cell 14", "Cell 29"])
        self.assertEqual(groups["Cell 14"], [5, 1, 4])
        self.assertEqual(groups["Cell 29"], [3, 2, 0])


class FilenameTests(unittest.TestCase):
    def test_distinct_angles_do_not_share_an_artifact_token(self) -> None:
        angles = [3.0, 3.0004, 3.0005, -3.0004]
        tokens = [waterfalls.angle_token(value) for value in angles]

        self.assertEqual(len(tokens), len(set(tokens)))
        for token in tokens:
            self.assertRegex(token, r"^[A-Za-z0-9]+$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
