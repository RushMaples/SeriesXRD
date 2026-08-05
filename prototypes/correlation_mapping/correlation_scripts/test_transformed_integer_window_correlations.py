#!/usr/bin/env python3
"""Focused tests for transformed integer-window preprocessing."""

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

import integer_window_correlations as iw  # noqa: E402
import uniform_peak_core as up  # noqa: E402
import uniform_window_core as uw  # noqa: E402
from run_uniform_xy_correlations import _clip_xy_to_interval  # noqa: E402
from uniform_profile_binding import bind_frozen_profile  # noqa: E402
from uniform_xy_input import read_xy_clean  # noqa: E402


class IntensityTransformTests(unittest.TestCase):
    @staticmethod
    def _processed_pool() -> list[dict[str, object]]:
        return [
            {"residual": np.asarray([-2.0, 0.0, 1.0, 4.0]), "noise": 0.2},
            {"residual": np.asarray([0.0, -1.0, 2.0, 3.0]), "noise": 0.4},
        ]

    def test_none_mode_is_exactly_identity(self) -> None:
        processed = self._processed_pool()
        transformed, audit, frame_audits = iw._transform_preprocessed_residuals(
            processed,
            iw.IntensityTransformConfig(mode="none"),
        )
        for result, source in zip(transformed, processed, strict=True):
            self.assertTrue(np.array_equal(result, source["residual"]))
        self.assertTrue(audit["default_path_numeric_identity"])
        self.assertTrue(all(row["transform_mode"] == "none" for row in frame_audits))

    def test_log_squared_uses_one_pooled_abs_residual_scale_and_noise_epsilon(self) -> None:
        config = iw.IntensityTransformConfig(
            mode="log_squared",
            scale_quantile=0.5,
            epsilon_floor=1.0e-12,
        )
        transformed, audit, frame_audits = iw._transform_preprocessed_residuals(
            self._processed_pool(),
            config,
        )
        self.assertEqual(audit["role_scale_a"], 1.5)
        self.assertAlmostEqual(audit["role_noise_sigma"], 0.3)
        self.assertAlmostEqual(audit["derived_epsilon"], 0.04)
        self.assertEqual(audit["clipped_fraction"], 0.5)
        self.assertTrue(audit["zero_maps_to_zero"])
        self.assertTrue(audit["output_bounded_0_1"])
        self.assertEqual(transformed[0][0], transformed[1][2])
        self.assertEqual(transformed[0][1], 0.0)
        self.assertTrue(all(row["role_scale_a"] == 1.5 for row in frame_audits))

    def test_exp_squared_is_zero_preserving_bounded_and_sign_erasing(self) -> None:
        config = iw.IntensityTransformConfig(
            mode="exp_squared",
            scale_quantile=0.5,
        )
        transformed, audit, _ = iw._transform_preprocessed_residuals(
            self._processed_pool(),
            config,
        )
        self.assertIsNone(audit["derived_epsilon"])
        self.assertEqual(transformed[0][0], transformed[1][2])
        self.assertEqual(transformed[0][1], 0.0)
        all_values = np.concatenate(transformed)
        self.assertTrue(np.all(np.isfinite(all_values)))
        self.assertGreaterEqual(float(np.min(all_values)), 0.0)
        self.assertLessEqual(float(np.max(all_values)), 1.0)

    def test_transform_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            iw.IntensityTransformConfig(mode="unknown")
        with self.assertRaises(ValueError):
            iw.IntensityTransformConfig(mode="log_squared", scale_quantile=0.0)
        with self.assertRaises(ValueError):
            iw.IntensityTransformConfig(mode="log_squared", epsilon_floor=0.0)


class WindowPipelineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        profile_path = SCRIPT_DIR / "configs" / "uniform-correlation-v2.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        cls.bound = bind_frozen_profile(profile, 0.3066)

    def _xy_fixture(self, directory: Path) -> Path:
        x = np.linspace(0.04, 32.0, 1600)
        y = (
            12.0
            + 0.08 * x
            + 4.0 * np.sin(0.9 * x)
            + 35.0 * np.exp(-0.5 * ((x - 8.2) / 0.15) ** 2)
            + 18.0 * np.exp(-0.5 * ((x - 21.4) / 0.23) ** 2)
        )
        path = directory / "frame_0000.xy"
        np.savetxt(path, np.column_stack((x, y)), fmt="%.12g")
        return path

    def _payload(
        self,
        path: Path,
    ) -> tuple[str, up.UniformPeakConfig, int, float, float]:
        return (
            str(path),
            self.bound.peak_config,
            self.bound.minimum_points_per_pattern,
            0.04,
            32.0,
        )

    def test_none_preprocessing_matches_previous_pipeline_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._xy_fixture(Path(temporary))
            result = iw._preprocess_window_profile(self._payload(path))
            raw_x, raw_y, _ = read_xy_clean(
                path,
                minimum_points=self.bound.minimum_points_per_pattern,
            )
            cleaned = up.clean_xy(raw_x, raw_y)
            x, y = _clip_xy_to_interval(cleaned.x, cleaned.y, 0.04, 32.0)
            expected = up.preprocess_pattern(x, y, self.bound.peak_config)
            self.assertTrue(np.array_equal(result["x"], expected.x))
            self.assertTrue(np.array_equal(result["residual"], expected.residual))

    def test_both_transforms_feed_unchanged_window_acf_pearson_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._xy_fixture(Path(temporary))
            result = iw._preprocess_window_profile(self._payload(path))
            processed = [
                result,
                {
                    **result,
                    "residual": 1.04 * result["residual"]
                    + 0.03 * np.roll(result["residual"], 2),
                    "noise": 1.04 * result["noise"],
                },
            ]
            spec = uw.make_fixed_sliding_window_spec(
                32.0,
                width_deg=iw.WINDOW_WIDTH_DEG,
                step_deg=iw.WINDOW_STEP_DEG,
                start_deg=iw.WINDOW_START_DEG,
            )
            outputs = {}
            for mode in ("log_squared", "exp_squared"):
                transformed, audit, _ = iw._transform_preprocessed_residuals(
                    processed,
                    iw.IntensityTransformConfig(mode=mode),
                )
                features = uw.build_window_features(
                    result["x"],
                    np.stack(transformed),
                    spec,
                    config=self.bound.window_config,
                )
                across = uw.compute_across_frame_correlations(
                    features,
                    ["scan", "scan"],
                    [1.0, 2.0],
                    config=self.bound.window_config,
                )
                within = uw.compute_within_frame_correlations(
                    features.fingerprints,
                    ["scan", "scan"],
                    [1.0, 2.0],
                    nonoverlap_indices=spec.nonoverlap_indices,
                    config=self.bound.window_config,
                )
                self.assertTrue(np.all(features.signal_valid), mode)
                self.assertTrue(np.all(features.fingerprint_valid), mode)
                self.assertEqual(across.acf_strict_by_scan.shape, (1, 28, 2, 2))
                self.assertEqual(within.by_frame.shape, (2, 28, 28))
                self.assertTrue(audit["output_bounded_0_1"])
                outputs[mode] = np.stack(transformed)
            self.assertFalse(
                np.array_equal(outputs["log_squared"], outputs["exp_squared"])
            )

    def test_integer_window_geometry_remains_zero_to_five_then_one_to_six(self) -> None:
        spec = uw.make_fixed_sliding_window_spec(
            32.0,
            width_deg=iw.WINDOW_WIDTH_DEG,
            step_deg=iw.WINDOW_STEP_DEG,
            start_deg=iw.WINDOW_START_DEG,
        )
        self.assertTrue(np.array_equal(spec.starts_deg, np.arange(28, dtype=float)))
        self.assertTrue(
            np.array_equal(spec.ends_deg, np.arange(5, 33, dtype=float))
        )
        self.assertEqual(spec.labels[0], "0-5")
        self.assertEqual(spec.labels[1], "1-6")
        self.assertEqual(spec.labels[-1], "27-32")


if __name__ == "__main__":
    unittest.main()
