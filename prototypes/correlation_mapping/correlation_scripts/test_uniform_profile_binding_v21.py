#!/usr/bin/env python3
"""Regression tests that v2.1 cannot tune the frozen v2 upstream analysis."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from uniform_profile_binding_v21 import (
    ProfileBindingV21Error,
    bind_frozen_profile_v21,
)


PROFILE = Path(__file__).resolve().parent / "configs" / "uniform-correlation-v2.1.json"


def _profile() -> dict:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


class ProfileBindingV21Tests(unittest.TestCase):
    def test_binds_all_upstream_and_edge_fields(self) -> None:
        bound = bind_frozen_profile_v21(_profile(), 0.3066)
        self.assertEqual(bound.peak_config.prominence_noise_factor, 5.0)
        self.assertEqual(bound.peak_config.height_noise_factor, 3.0)
        self.assertEqual(bound.tracking_config.ambiguous_cost_margin, 0.25)
        self.assertEqual(bound.tracking_config.unmatched_cost, 10.0)
        self.assertEqual(
            set(bound.binding_audit["segmented_tracking_config"]),
            set(bound.resolved_semantics["segmented_tracking_config"]),
        )
        self.assertEqual(
            set(bound.binding_audit["tracking_policy"]),
            set(bound.resolved_semantics["tracking_policy"]),
        )

    def test_rejects_any_upstream_tuning(self) -> None:
        cases = [
            ("peak_detection", "minimum_prominence_sigma", 4.9),
            ("peak_fit", "minimum_delta_bic", 9.0),
            ("consensus", "fwhm_distance_factor", 0.6),
            ("windows", "window_count", 25),
        ]
        for section, field, value in cases:
            with self.subTest(section=section, field=field):
                profile = _profile()
                profile[section][field] = value
                with self.assertRaisesRegex(ProfileBindingV21Error, "differs from frozen"):
                    bind_frozen_profile_v21(profile, 0.3066)

    def test_rejects_tracking_policy_relaxation(self) -> None:
        profile = _profile()
        profile["tracking"]["require_margin_at_both_endpoints"] = False
        with self.assertRaises(ProfileBindingV21Error):
            bind_frozen_profile_v21(profile, 0.3066)


if __name__ == "__main__":
    unittest.main()
