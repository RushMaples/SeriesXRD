#!/usr/bin/env python3
"""Focused contracts for stable squared-intensity preprocessing."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nonlinear_intensity_preprocessing as preprocessing  # noqa: E402


class NonlinearIntensityPreprocessingTests(unittest.TestCase):
    def test_log_transform_matches_documented_formula(self) -> None:
        spec = preprocessing.make_roi_transform_spec(
            preprocessing.LOG_SQUARED,
            scale=10.0,
            noise_floor=1.0,
        )
        values = np.asarray([-2.0, 0.0, 5.0, 10.0, 20.0, np.nan])
        result = np.asarray(spec.transform(values))
        epsilon = (1.0 / 10.0) ** 2
        expected_midpoint = (
            np.log(0.5**2 + epsilon) - np.log(epsilon)
        ) / (np.log(1.0 + epsilon) - np.log(epsilon))

        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], expected_midpoint, places=15)
        self.assertEqual(result[3], 1.0)
        self.assertEqual(result[4], 1.0)
        self.assertTrue(np.isnan(result[5]))

    def test_log_squared_is_the_only_supported_denoise_transform(self) -> None:
        self.assertEqual(
            preprocessing.SUPPORTED_METHODS,
            (preprocessing.LOG_SQUARED,),
        )
        with self.assertRaisesRegex(ValueError, "unsupported transform"):
            preprocessing.make_roi_transform_spec(
                "unsupported",  # type: ignore[arg-type]
                scale=4.0,
                noise_floor=1.0,
            )

    def test_signed_window_input_is_clipped_then_squared(self) -> None:
        values = np.asarray([-3.0, -0.5, 0.0, 0.5, 3.0])
        result = np.asarray(
            preprocessing.transform_bounded_squared(
                values,
                method=preprocessing.LOG_SQUARED,
                epsilon=0.01,
            )
        )

        np.testing.assert_allclose(result[0], result[-1], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result[1], result[-2], rtol=0.0, atol=0.0)
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[2], 0.0)

    def test_mask_and_unmasked_nan_are_preserved(self) -> None:
        values = np.ma.array(
            [0.0, 2.0, np.nan, 8.0],
            mask=[False, True, False, False],
        )
        spec = preprocessing.make_roi_transform_spec(
            preprocessing.LOG_SQUARED,
            scale=8.0,
            noise_floor=1.0,
        )
        result = spec.transform(values)

        self.assertTrue(np.ma.isMaskedArray(result))
        np.testing.assert_array_equal(
            np.ma.getmaskarray(result),
            np.asarray([False, True, False, False]),
        )
        self.assertTrue(np.isnan(np.ma.getdata(result)[2]))
        self.assertEqual(float(result[0]), 0.0)
        self.assertEqual(float(result[3]), 1.0)

    def test_fixed_pooled_scale_uses_positive_finite_unmasked_values(self) -> None:
        first = np.ma.array(
            [-5.0, 0.0, 1.0, 2.0, 100.0],
            mask=[False, False, False, False, True],
        )
        second = np.asarray([3.0, 4.0, np.nan, np.inf])
        estimate = preprocessing.estimate_fixed_pooled_scale(
            [first, second],
            quantile=0.5,
        )

        # Positive finite unmasked pool is [1, 2, 3, 4].
        self.assertEqual(estimate.scale, 2.5)
        self.assertEqual(estimate.array_count, 2)
        self.assertEqual(estimate.total_slots, 9)
        self.assertEqual(estimate.masked_slots, 1)
        self.assertEqual(estimate.unmasked_nan_slots, 1)
        self.assertEqual(estimate.unmasked_positive_infinity_slots, 1)
        self.assertEqual(estimate.positive_finite_slots, 4)

    def test_epsilon_is_derived_from_noise_floor_with_a_floor(self) -> None:
        self.assertAlmostEqual(
            preprocessing.epsilon_from_noise_floor(0.5, 10.0),
            0.0025,
            places=15,
        )
        self.assertEqual(
            preprocessing.epsilon_from_noise_floor(0.0, 10.0),
            preprocessing.DEFAULT_EPSILON_FLOOR,
        )

    def test_audit_reports_literal_hazards_and_output_contract(self) -> None:
        values = np.asarray(
            [-100.0, 0.0, 1.0, 100.0, np.nan, np.inf, -np.inf]
        )
        spec = preprocessing.make_roi_transform_spec(
            preprocessing.LOG_SQUARED,
            scale=10.0,
            noise_floor=1.0,
        )
        audit = spec.audit(values)

        self.assertEqual(audit["literal_log_zero_to_negative_infinity_slots"], 1)
        self.assertEqual(audit["negative_slots_clipped_to_z_zero"], 1)
        self.assertEqual(audit["above_fixed_scale_slots_clipped_to_z_one"], 1)
        self.assertEqual(audit["output_below_zero_slots"], 0)
        self.assertEqual(audit["output_above_one_slots"], 0)
        self.assertTrue(audit["mask_preserved_exactly"])

    def test_provenance_writer_is_json_safe_and_complete(self) -> None:
        pooled = [np.asarray([0.0, 1.0, 2.0, 3.0])]
        spec, estimate = preprocessing.fit_roi_transform(
            pooled,
            preprocessing.LOG_SQUARED,
            noise_floor=0.25,
            scale_quantile=0.75,
        )
        audit = spec.audit(pooled[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            returned = preprocessing.write_transform_provenance(
                path,
                spec,
                scale_estimate=estimate,
                audits={"fixture": audit},
                context={"dataset": "test"},
            )
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(returned, path)
        self.assertEqual(record["schema_version"], preprocessing.SCHEMA_VERSION)
        self.assertEqual(record["transform"]["method"], preprocessing.LOG_SQUARED)
        self.assertEqual(record["context"]["dataset"], "test")
        self.assertFalse(record["literal_raw_formula_used"])
        self.assertIn("fixture", record["audits"])

    def test_standalone_audit_writer_accepts_numpy_context(self) -> None:
        spec = preprocessing.make_roi_transform_spec(
            preprocessing.LOG_SQUARED,
            scale=2.0,
            noise_floor=0.2,
        )
        audit = spec.audit(np.asarray([0.0, 1.0, 2.0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            preprocessing.write_numerical_audit(
                path,
                audit,
                context={"pressure_gpa": np.asarray([3.5, 3.75])},
            )
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(record["schema_version"], preprocessing.SCHEMA_VERSION)
        self.assertEqual(record["audit"]["finite_unmasked_slots"], 3)
        self.assertEqual(record["context"]["pressure_gpa"], [3.5, 3.75])

    def test_invalid_log_spec_without_noise_floor_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise_floor"):
            preprocessing.make_roi_transform_spec(
                preprocessing.LOG_SQUARED,
                scale=1.0,
            )


if __name__ == "__main__":
    unittest.main()
