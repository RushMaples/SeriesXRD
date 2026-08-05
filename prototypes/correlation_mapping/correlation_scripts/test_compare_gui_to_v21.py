#!/usr/bin/env python3
"""Small deterministic tests for the v2.1 GUI cross-check layer."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from compare_gui_to_v21 import (
    FIXED_VISUALIZATION_FILENAMES,
    GuiDataset,
    MATCH_FORMULA,
    _acf_fingerprint,
    _flag_reason,
    _match_one_group,
    _pearson,
    _require_run_profile,
    run,
    write_gui_crosscheck_report,
    write_gui_visualizations,
)


class GuiCrosscheckTests(unittest.TestCase):
    def test_profile_guard_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_manifest.json").write_text(
                json.dumps({"profile": "uniform-correlation-v2"}), encoding="utf-8"
            )
            manifest = _require_run_profile(
                root, "uniform-correlation-v2", role="legacy"
            )
            self.assertEqual(manifest["profile"], "uniform-correlation-v2")
            with self.assertRaisesRegex(ValueError, "profile mismatch"):
                _require_run_profile(root, "uniform-correlation-v2.1", role="result")

    def test_result_and_legacy_roots_must_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_manifest.json").write_text(
                json.dumps({"profile": "uniform-correlation-v2.1"}), encoding="utf-8"
            )
            args = SimpleNamespace(
                result_root=str(root), legacy_root=str(root),
                gui_inventory=str(root / "missing.csv"), output_dir=None,
            )
            with self.assertRaisesRegex(ValueError, "must be different"):
                run(args)

    def test_flag_bitmask_decoding(self) -> None:
        self.assertEqual(_flag_reason(0), "good")
        self.assertEqual(_flag_reason(1 | 8), "low_amp;width_bound")
        self.assertEqual(_flag_reason(2 | 4 | 16), "bad_chi2;center_drift;no_converge")

    def test_hungarian_matching_is_one_to_one_and_gated(self) -> None:
        corr = pd.DataFrame(
            {
                "correlation_row_id": ["c0", "c1", "c2"],
                "frame": [1, 1, 1], "peak_id": [0, 1, 2],
                "state": ["reliable"] * 3, "reason": ["passed"] * 3,
                "two_theta_deg": [10.0, 10.2, 14.0],
                "fwhm_two_theta_deg": [0.1, 0.1, 0.1],
                "raw_fitted_area": [1.0, 2.0, 3.0], "relative_area": [0.1, 0.2, 0.3],
            }
        )
        gui = pd.DataFrame(
            {
                "gui_peak_id": ["g0", "g1"], "gui_frame": [0, 0],
                "gui_filename": ["f.xy", "f.xy"], "gui_good": [1, 1],
                "gui_flag": [0, 0], "gui_rejected_reason": ["good", "good"],
                "gui_two_theta_deg": [10.01, 10.19],
                "gui_fwhm_two_theta_deg": [0.1, 0.1],
                "gui_area": [4.0, 5.0], "gui_radial_step_deg": [0.01, 0.01],
            }
        )
        rows = _match_one_group(
            corr, gui, result_version="test", view="good_only",
            channel="spots", scan="scan001", pressure=3.5,
        )
        statuses = [row["match_status"] for row in rows]
        self.assertEqual(statuses.count("matched"), 2)
        self.assertEqual(statuses.count("correlation_only"), 1)
        self.assertEqual(statuses.count("gui_only"), 0)
        self.assertTrue(all(row["matching_formula"] == MATCH_FORMULA for row in rows))

    def test_matching_prefers_nearest_center_not_broadest_gate(self) -> None:
        corr = pd.DataFrame(
            {
                "correlation_row_id": ["c0"], "frame": [1], "peak_id": [0],
                "state": ["reliable"], "reason": ["passed"],
                "two_theta_deg": [10.0], "fwhm_two_theta_deg": [0.05],
                "raw_fitted_area": [1.0], "relative_area": [0.1],
            }
        )
        gui = pd.DataFrame(
            {
                "gui_peak_id": ["near_narrow", "far_broad"], "gui_frame": [0, 0],
                "gui_filename": ["f.xy", "f.xy"], "gui_good": [1, 1],
                "gui_flag": [0, 0], "gui_rejected_reason": ["good", "good"],
                "gui_two_theta_deg": [10.02, 10.04],
                "gui_fwhm_two_theta_deg": [0.05, 0.95],
                "gui_area": [1.0, 1.0], "gui_radial_step_deg": [0.01, 0.01],
            }
        )
        rows = _match_one_group(
            corr, gui, result_version="test", view="good_only",
            channel="spots", scan="scan001", pressure=1.0,
        )
        matched = [row for row in rows if row["match_status"] == "matched"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["gui_peak_id"], "near_narrow")

    def test_acf_fingerprint_identity(self) -> None:
        x = np.linspace(0.0, 4.0 * np.pi, 101)
        signal = np.sin(x) + 0.2 * np.cos(3.0 * x)
        fp = _acf_fingerprint(signal)
        self.assertIsNotNone(fp)
        self.assertAlmostEqual(_pearson(fp, fp), 1.0, places=12)

    def test_fixed_visualization_files_are_written(self) -> None:
        radial = np.linspace(5.0, 15.0, 101)
        pressures = np.array([1.0, 2.0, 4.0])
        patterns = np.vstack([
            1.0 + np.exp(-0.5 * ((radial - (8.0 + 0.1 * p)) / 0.12) ** 2)
            for p in pressures
        ])
        peaks = pd.DataFrame(
            {
                "gui_peak_id": ["g0", "g1", "g2"],
                "channel": ["spots"] * 3, "scan": ["scan001"] * 3,
                "pressure_GPa": pressures, "gui_good": [1, 0, 1],
                "gui_two_theta_deg": [8.1, 8.2, 8.4],
                "gui_area": [10.0, 9.0, 8.0],
                "gui_fwhm_two_theta_deg": [0.10, 0.11, 0.12],
            }
        )
        dataset = GuiDataset(
            channel="spots", scan="scan001", path=Path("synthetic_gui.h5"),
            pattern_source="clean", unit="2th_deg", wavelength_A=0.3,
            radial_deg=radial, radial_step_deg=float(np.median(np.diff(radial))),
            pressure_GPa=pressures, filenames=("f0", "f1", "f2"),
            excluded=np.array([False, False, False]), patterns=patterns, peaks=peaks,
        )
        matches = pd.DataFrame(
            {
                "result_version": ["uniform-correlation-v2.1"] * 2,
                "match_view": ["good_only"] * 2,
                "match_status": ["matched"] * 2,
                "channel": ["spots"] * 2, "scan": ["scan001"] * 2,
                "pressure_GPa": [1.0, 4.0], "gui_peak_id": ["g0", "g2"],
                "gui_two_theta_deg": [8.1, 8.4],
                "correlation_two_theta_deg": [8.105, 8.395],
                "position_delta_over_gate": [0.05, 0.04],
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            manifest = write_gui_visualizations(out, [dataset], matches)
            self.assertEqual(set(manifest["filename"]), set(FIXED_VISUALIZATION_FILENAMES.values()))
            for name in FIXED_VISUALIZATION_FILENAMES.values():
                path = out / name
                self.assertTrue(path.is_file(), name)
                self.assertGreater(path.stat().st_size, 1000, name)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            report = write_gui_crosscheck_report(
                out, result_root=Path("v21"), legacy_root=Path("v2"),
                datasets=[dataset], match_summaries=pd.DataFrame(),
                boundary_summary=pd.DataFrame(), pattern_checks=pd.DataFrame(),
                control_auc=pd.DataFrame(), control_boundaries=pd.DataFrame(),
                total_spots_scans=56,
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("1/56", text)
            self.assertIn("fit/tungsten", text)


if __name__ == "__main__":
    unittest.main()
