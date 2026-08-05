#!/usr/bin/env python3
"""Regression tests for the generic XY input adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uniform_peak_core import UniformPeakConfig, preprocess_pattern
from uniform_xy_input import read_xy_clean


class RawInputAuditTests(unittest.TestCase):
    def test_loader_preserves_raw_cleanup_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pattern.xy"
            x = np.linspace(5.0, 10.0, 130)
            y = 10.0 + np.sin(x)
            rows = list(zip(x, y, strict=True))
            rows[5] = (rows[4][0], rows[5][1])  # duplicate 2theta
            rows[8] = (rows[8][0], float("nan"))
            rows[10], rows[11] = rows[11], rows[10]  # non-monotonic raw order
            path.write_text(
                "# wavelength_A: 0.3066\n"
                + "\n".join(f"{x_value:.12g} {y_value:.12g}" for x_value, y_value in rows)
                + "\n",
                encoding="utf-8",
            )
            raw_x, raw_y, metadata = read_xy_clean(path, minimum_points=128)
            self.assertEqual(len(raw_x), 130)
            self.assertTrue(np.isnan(raw_y[8]))
            self.assertEqual(metadata["wavelength_A"], "0.3066")
            processed = preprocess_pattern(
                raw_x,
                raw_y,
                UniformPeakConfig(wavelength=0.3066, bootstrap_iterations=0),
            )
            self.assertEqual(processed.cleaned.finite_removed, 1)
            self.assertEqual(processed.cleaned.duplicate_points_merged, 1)
            self.assertFalse(processed.cleaned.originally_strictly_increasing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
