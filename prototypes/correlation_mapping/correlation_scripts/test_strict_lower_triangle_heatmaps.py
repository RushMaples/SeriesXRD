#!/usr/bin/env python3
"""Tests for strict-lower correlation heatmap display layers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from single_global_per_peak import strict_lower_triangle_layers  # noqa: E402


class StrictLowerTriangleLayersTest(unittest.TestCase):
    def test_diagonal_and_upper_are_structurally_hidden(self) -> None:
        source = np.arange(25, dtype=float).reshape(5, 5)
        original = source.copy()
        data, missing = strict_lower_triangle_layers(source)
        data_mask = np.ma.getmaskarray(data)
        missing_mask = np.ma.getmaskarray(missing)
        upper = np.triu_indices(5, k=0)
        lower = np.tril_indices(5, k=-1)

        self.assertTrue(np.all(data_mask[upper]))
        self.assertTrue(np.all(missing_mask[upper]))
        self.assertTrue(np.all(~data_mask[lower]))
        self.assertTrue(np.all(missing_mask[lower]))
        np.testing.assert_array_equal(np.asarray(data)[lower], source[lower])
        np.testing.assert_array_equal(source, original)

    def test_missing_lower_cell_is_gray_layer_not_structural_white(self) -> None:
        source = np.arange(16, dtype=float).reshape(4, 4)
        source[3, 0] = np.nan
        source[0, 3] = np.nan
        data, missing = strict_lower_triangle_layers(source)
        data_mask = np.ma.getmaskarray(data)
        missing_mask = np.ma.getmaskarray(missing)

        self.assertTrue(data_mask[3, 0])
        self.assertFalse(missing_mask[3, 0])
        self.assertTrue(data_mask[0, 3])
        self.assertTrue(missing_mask[0, 3])

    def test_non_square_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            strict_lower_triangle_layers(np.zeros((3, 4), dtype=float))


if __name__ == "__main__":
    unittest.main()
