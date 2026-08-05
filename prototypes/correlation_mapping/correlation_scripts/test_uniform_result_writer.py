#!/usr/bin/env python3
"""Small output-contract tests for the uniform correlation serializer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import uniform_result_writer as writer
import uniform_peak_core as peak
import uniform_window_core as window
import validate_uniform_xy_correlations as validator


class WithinFrameWriterContractTests(unittest.TestCase):
    def test_within_aggregate_outputs_have_support_and_ci_companions(self) -> None:
        rng = np.random.default_rng(1234)
        scan_labels = [f"scan_{index}" for index in range(5)]
        pressures = [0.0, 1.0]
        frame_scans = [scan for scan in scan_labels for _ in pressures]
        frame_pressures = pressures * len(scan_labels)
        fingerprints = rng.normal(size=(len(frame_scans), window.N_WINDOWS, 24))

        # One unavailable window must remain NaN after aggregation, never become zero.
        fingerprints[0, 3] = np.nan
        within = window.compute_within_frame_correlations(
            fingerprints,
            frame_scans,
            frame_pressures,
        )
        spec = window.make_uniform_window_spec(2.0, 32.0)

        plot_calls: dict[str, dict[str, object]] = {}

        def fake_plot(path: Path, *_args: object, **kwargs: object) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"plot-placeholder")
            plot_calls[path.as_posix()] = kwargs

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            writer, "plot_matrix", side_effect=fake_plot
        ):
            channel_root = Path(temp_dir) / "spots"
            writer.write_within_results(
                channel_root,
                "spots",
                within,
                spec,
                frame_ids=np.arange(len(frame_scans)),
                frame_scans=frame_scans,
                frame_pressures=frame_pressures,
                n_bootstrap=25,
                seed=0,
                confidence=0.95,
                make_plots=True,
            )
            root = channel_root / "within_frame"

            for stem in ("0GPa", "1GPa"):
                self.assertTrue((root / "by_pressure" / "heatmaps" / f"{stem}.png").is_file())
                self.assertTrue(
                    (root / "by_pressure" / "support_maps" / f"{stem}_support.png").is_file()
                )
                for bound in ("low", "high"):
                    csv_path = (
                        root
                        / "by_pressure"
                        / "confidence_intervals"
                        / f"{stem}_ci_{bound}.csv"
                    )
                    png_path = csv_path.with_suffix(".png")
                    self.assertTrue(csv_path.is_file())
                    self.assertTrue(png_path.is_file())
                    self.assertEqual(plot_calls[png_path.as_posix()]["vmin"], -1.0)
                    self.assertEqual(plot_calls[png_path.as_posix()]["vmax"], 1.0)

            non_dir = root / "nonoverlap_control"
            for name in (
                "aggregate_matrix.csv",
                "support_matrix.csv",
                "ci_low_matrix.csv",
                "ci_high_matrix.csv",
                "aggregate_heatmap.png",
                "support_heatmap.png",
                "ci_low_heatmap.png",
                "ci_high_heatmap.png",
            ):
                self.assertTrue((non_dir / name).is_file(), name)

            for name in ("ci_low_heatmap.png", "ci_high_heatmap.png"):
                self.assertTrue((root / "all_windows" / name).is_file())

            with np.load(root / "within_frame_matrices.npz", allow_pickle=False) as archive:
                for key in (
                    "aggregate_by_pressure_ci_low",
                    "aggregate_by_pressure_ci_high",
                    "support_by_pressure",
                    "available_by_pressure",
                    "support_required_by_pressure",
                    "sufficient_support_by_pressure",
                    "nonoverlap_ci_low",
                    "nonoverlap_ci_high",
                ):
                    self.assertIn(key, archive.files)
                aggregate_by_pressure = archive["aggregate_by_pressure"]
                np.testing.assert_allclose(
                    aggregate_by_pressure,
                    within.aggregate_by_pressure,
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
                sufficient = archive["sufficient_support_by_pressure"].astype(bool)
                for key in (
                    "aggregate_by_pressure",
                    "aggregate_by_pressure_ci_low",
                    "aggregate_by_pressure_ci_high",
                    "nonoverlap_aggregate",
                    "nonoverlap_ci_low",
                    "nonoverlap_ci_high",
                ):
                    values = archive[key]
                    finite = values[np.isfinite(values)]
                    self.assertTrue(np.all((finite >= -1.0) & (finite <= 1.0)), key)
                    np.testing.assert_allclose(
                        values,
                        np.swapaxes(values, -1, -2),
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
                self.assertTrue(np.all(np.isnan(aggregate_by_pressure[~sufficient])))
                self.assertTrue(np.isnan(aggregate_by_pressure[0, 3, 0]))


class EmptyPerPeakContractTests(unittest.TestCase):
    def test_zero_official_tracks_is_auditable_and_validator_legal(self) -> None:
        empty_analysis = peak.PerPeakAnalysis(
            consensus_by_pressure={},
            tracks=(),
            assignments={},
            correlations={},
            near_far={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            for channel in ("spots", "fit"):
                channel_root = run_dir / channel
                metrics = writer.write_per_peak_results(
                    channel_root,
                    channel,
                    empty_analysis,
                    (),
                    scans=("scan_a", "scan_b"),
                    pressures=(0.0, 1.0),
                    make_plots=False,
                )
                self.assertEqual(metrics["official_radial_tracks"], 0)
                per_peak = channel_root / "per_peak"
                for directory in ("area", "location", "presence", "support", "trajectories"):
                    self.assertTrue((per_peak / directory).is_dir(), directory)

                canonical_fields, canonical_rows = validator.read_csv_rows(
                    per_peak / "canonical_tracks.csv"
                )
                summary_fields, summary_rows = validator.read_csv_rows(
                    per_peak / "peak_summary.csv"
                )
                self.assertIn("official", canonical_fields)
                self.assertIn("track_id", summary_fields)
                self.assertEqual(canonical_rows, [])
                self.assertEqual(summary_rows, [])

                with np.load(per_peak / "per_peak_matrices.npz", allow_pickle=False) as archive:
                    self.assertEqual(archive["track_ids"].shape, (0,))
                    for key in (
                        "area",
                        "location",
                        "presence",
                        "n_available",
                        "n_both_present",
                        "n10",
                        "n01",
                        "n_unknown",
                        "required_support",
                    ):
                        self.assertEqual(archive[key].shape, (0, 2, 2), key)
                    for key in ("area_by_scan", "location_by_scan", "presence_by_scan"):
                        self.assertEqual(archive[key].shape, (0, 2, 2, 2), key)

            state = validator.ValidationState()
            validator.validate_matrices(run_dir, state, tolerance=1.0e-10)
            validator.validate_missing_semantics(run_dir, state, tolerance=1.0e-10)
            for channel in ("spots", "fit"):
                for family in ("area", "location", "presence"):
                    self.assertTrue(
                        state.checks[
                            f"matrix_family_{channel}_{family}_present_or_zero_tracks"
                        ]
                    )
                self.assertTrue(state.checks[f"matrix_family_{channel}_support_present"])
            for family in ("area", "location", "presence", "support"):
                self.assertTrue(state.checks[f"matrix_family_{family}_present"])
            self.assertTrue(state.checks["missing_count_arrays_present"])
            self.assertTrue(state.checks["missing_presence_count_semantics"])
            self.assertTrue(state.checks["missing_insufficient_support_is_nan"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
