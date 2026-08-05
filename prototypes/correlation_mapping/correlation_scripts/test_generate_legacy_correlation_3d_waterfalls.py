#!/usr/bin/env python3
"""Focused tests for the legacy 3D correlation-waterfall generator."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import generate_legacy_correlation_3d_waterfalls as viewer


class MatrixTests(unittest.TestCase):
    def test_lower_triangle_is_mirrored_without_inventing_diagonal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame", "a", "b", "c"])
                writer.writerow(["a", "", "", ""])
                writer.writerow(["b", "0.25", "", ""])
                writer.writerow(["c", "-0.5", "0.8", ""])
            matrix = viewer.read_lower_triangle_matrix(path)
        self.assertTrue(np.isnan(matrix.full[0, 0]))
        self.assertAlmostEqual(matrix.full[0, 1], 0.25)
        self.assertAlmostEqual(matrix.full[1, 0], 0.25)
        self.assertAlmostEqual(matrix.full[0, 2], -0.5)
        self.assertAlmostEqual(matrix.full[2, 1], 0.8)

    def test_rounding_keeps_missing_as_json_null(self) -> None:
        result = viewer.rounded_matrix(np.array([[np.nan, 0.12345678]]))
        self.assertIsNone(result[0][0])
        self.assertEqual(result[0][1], 0.123457)


class SeriesTests(unittest.TestCase):
    @staticmethod
    def pattern(index: int, cell: str, pressure: float, decomp: bool = False) -> viewer.PatternInfo:
        values = np.array([1.0, 2.0])
        return viewer.PatternInfo(
            index=index,
            label=f"{pressure:g}GPa" + ("_decomp" if decomp else ""),
            path=Path(f"/tmp/{cell}/{index}.xy"),
            cell=cell,
            pressure=pressure,
            decomp=decomp,
            two_theta=values,
            intensity=values,
            normalized=values,
        )

    def test_cells_are_separate_and_decompression_follows_last_compression(self) -> None:
        patterns = [
            self.pattern(0, "Cell_14", 0.7),
            self.pattern(1, "Cell_29", 1.0),
            self.pattern(2, "Cell_14", 1.3),
            self.pattern(3, "Cell_29", 1.5),
            self.pattern(4, "Cell_29", 2.4, decomp=True),
        ]
        series = {item.key: item for item in viewer.build_series(patterns)}
        self.assertEqual(
            [(pair.previous, pair.current) for pair in series["cell14_compression"].pairs],
            [(0, 2)],
        )
        self.assertEqual(
            [(pair.previous, pair.current) for pair in series["cell29_compression"].pairs],
            [(1, 3)],
        )
        self.assertEqual(
            [(pair.previous, pair.current) for pair in series["cell29_decompression"].pairs],
            [(3, 4)],
        )

    def test_compression_display_rows_include_raw_only_baseline(self) -> None:
        patterns = [
            self.pattern(0, "Cell_14", 0.7),
            self.pattern(1, "Cell_14", 1.3),
            self.pattern(2, "Cell_14", 6.6),
        ]
        item = {
            series.key: series for series in viewer.build_series(patterns)
        }["cell14_compression"]
        rows = viewer.build_display_rows(item, patterns)
        self.assertEqual(len(rows), 3)
        self.assertIsNone(rows[0].previous)
        self.assertEqual(rows[0].current, 0)
        self.assertEqual(rows[0].kind, "baseline")
        self.assertIn("Baseline / no previous correlation", rows[0].label)
        self.assertEqual((rows[1].previous, rows[1].current), (0, 1))
        self.assertAlmostEqual(rows[1].delta_p_gpa, 0.6)
        self.assertEqual((rows[2].previous, rows[2].current), (1, 2))
        self.assertAlmostEqual(rows[2].delta_p_gpa, 5.3)

    def test_decompression_display_row_remains_reference_pair(self) -> None:
        patterns = [
            self.pattern(0, "Cell_29", 1.0),
            self.pattern(1, "Cell_29", 12.8),
            self.pattern(2, "Cell_29", 2.4, decomp=True),
        ]
        item = {
            series.key: series for series in viewer.build_series(patterns)
        }["cell29_decompression"]
        rows = viewer.build_display_rows(item, patterns)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].previous, rows[0].current), (1, 2))
        self.assertEqual(rows[0].kind, "compression-to-decompression branch step")
        self.assertAlmostEqual(rows[0].delta_p_gpa, -10.4)


if __name__ == "__main__":
    unittest.main()
