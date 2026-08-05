#!/usr/bin/env python3
"""Synthetic regression tests for the frozen uniform XY correlation cores.

Run directly with ``python test_uniform_xy_algorithms.py`` or through unittest.
No UOTe file, hand-curated track, or result directory is used here: the tests
exercise the same public algorithms on generated signals with known answers.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, fields
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import uniform_peak_core as peak  # noqa: E402
import uniform_profile_binding as profile_binding  # noqa: E402
import uniform_window_core as window  # noqa: E402


WAVELENGTH = 0.3066
PROFILE_PATH = SCRIPT_DIR / "configs" / "uniform-correlation-v2.json"


def pseudo_voigt(
    x: np.ndarray,
    center: float,
    fwhm: float,
    area: float,
    eta: float = 0.4,
) -> np.ndarray:
    """Unit-area pseudo-Voigt used only to generate test observations."""

    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gamma = fwhm / 2.0
    gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi)
    )
    lorentzian = gamma / (math.pi * ((x - center) ** 2 + gamma**2))
    return area * ((1.0 - eta) * gaussian + eta * lorentzian)


def synthetic_pattern(
    peaks: list[tuple[float, float, float]],
    *,
    seed: int = 0,
    scale: float = 1.0,
    points: int = 2001,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(5.0, 15.0, points)
    curved_baseline = 40.0 + 1.5 * x + 0.015 * (x - 10.0) ** 2
    signal = curved_baseline.copy()
    for center, fwhm, area in peaks:
        signal += pseudo_voigt(x, center, fwhm, area)
    signal += np.random.default_rng(seed).normal(0.0, 0.02, x.size)
    return x, scale * signal


def reliable_peaks(frame: peak.FramePeaks) -> list[peak.PeakFit]:
    return [item for item in frame.peaks if item.reliable]


def assigned(
    scan: str,
    pressure: float,
    state: str,
    *,
    relative_area: float = math.nan,
    q: float = math.nan,
    fwhm_q: float = math.nan,
) -> peak.AssignedObservation:
    return peak.AssignedObservation(
        track_id="radial_peak_001",
        scan=scan,
        pressure=pressure,
        frame=0,
        state=state,
        reason=f"synthetic_{state}",
        relative_area=relative_area,
        q=q,
        fwhm_q=fwhm_q,
    )


def consensus(
    identifier: str,
    pressure_index: int,
    q: float,
    *,
    width: float = 0.05,
) -> peak.PressureConsensus:
    return peak.PressureConsensus(
        consensus_id=identifier,
        channel="spots",
        pressure=float(pressure_index),
        pressure_index=pressure_index,
        q=q,
        fwhm_q=width,
        relative_area=0.1,
        support=5,
        total_scans=5,
        required_support=5,
        member_keys=(),
        reliable=True,
        ambiguous=False,
    )


class PeakFormulaTests(unittest.TestCase):
    def test_area_ratio_formula_and_invalid_values(self) -> None:
        self.assertAlmostEqual(peak.relative_area_similarity(2.0, 4.0), 0.5, places=15)
        self.assertAlmostEqual(peak.relative_area_similarity(4.0, 2.0), 0.5, places=15)
        self.assertTrue(math.isnan(peak.relative_area_similarity(math.nan, 1.0)))
        self.assertTrue(math.isnan(peak.relative_area_similarity(-1.0, 1.0)))

    def test_one_fwhm_location_shift_is_one_quarter(self) -> None:
        fwhm = 0.02
        score = peak.location_similarity(0.5, 0.5 + fwhm, fwhm, fwhm)
        self.assertAlmostEqual(score, 0.25, places=12)

    def test_q_conversion_round_trip(self) -> None:
        angles = np.asarray([3.0, 10.0, 25.0, 50.0])
        q = peak.two_theta_to_q(angles, WAVELENGTH)
        np.testing.assert_allclose(peak.q_to_two_theta(q, WAVELENGTH), angles, atol=1e-12)

    def test_support_formulas(self) -> None:
        self.assertEqual(peak.minimum_scan_support(0), 0)
        self.assertEqual(peak.minimum_scan_support(3), 3)
        self.assertEqual(peak.minimum_scan_support(56), 6)
        self.assertEqual(peak.minimum_pressure_support(20), 3)


class FrozenProfileBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    def test_every_config_field_is_explicitly_bound_and_audited(self) -> None:
        bound = profile_binding.bind_frozen_profile(self.profile, WAVELENGTH)
        peak_fields = {item.name for item in fields(peak.UniformPeakConfig)}
        window_fields = {item.name for item in fields(window.UniformWindowConfig)}
        self.assertEqual(set(bound.binding_audit["peak_config"]), peak_fields)
        self.assertEqual(set(bound.binding_audit["window_config"]), window_fields)
        self.assertEqual(set(asdict(bound.peak_config)), peak_fields)
        self.assertEqual(set(asdict(bound.window_config)), window_fields)
        self.assertEqual(bound.peak_config.prominence_noise_factor, 5.0)
        self.assertEqual(bound.peak_config.fit_max_nfev, 4000)
        self.assertEqual(bound.window_config.shift_tolerant_neighbor_steps, 1)
        self.assertEqual(bound.window_config.nonoverlap_stride_windows, 5)

    def test_numeric_profile_change_changes_resolved_semantics_and_digest(self) -> None:
        original = profile_binding.bind_frozen_profile(self.profile, WAVELENGTH)
        changed_profile = copy.deepcopy(self.profile)
        changed_profile["peak_detection"]["minimum_prominence_sigma"] = 6.0
        changed = profile_binding.bind_frozen_profile(changed_profile, WAVELENGTH)
        self.assertEqual(changed.peak_config.prominence_noise_factor, 6.0)
        self.assertNotEqual(original.semantic_sha256, changed.semantic_sha256)

    def test_unsupported_formula_or_unbound_key_fails_before_execution(self) -> None:
        changed_formula = copy.deepcopy(self.profile)
        changed_formula["windows"]["width_rule"] = "analysis_span/7"
        with self.assertRaises(profile_binding.ProfileBindingError):
            profile_binding.bind_frozen_profile(changed_formula, WAVELENGTH)

        extra_key = copy.deepcopy(self.profile)
        extra_key["statistics"]["silent_threshold"] = 123
        with self.assertRaises(profile_binding.ProfileBindingError):
            profile_binding.bind_frozen_profile(extra_key, WAVELENGTH)


class PeakDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = peak.UniformPeakConfig(wavelength=WAVELENGTH, bootstrap_iterations=0)

    def detect(
        self,
        peaks_to_generate: list[tuple[float, float, float]],
        *,
        seed: int = 0,
        scale: float = 1.0,
    ) -> tuple[peak.PreprocessedXY, peak.FramePeaks]:
        x, y = synthetic_pattern(peaks_to_generate, seed=seed, scale=scale)
        preprocessed = peak.preprocess_xy(x, y, self.config)
        frame = peak.detect_peaks(
            preprocessed,
            frame=0,
            scan="scan000",
            pressure=0.0,
            channel="spots",
            config=self.config,
        )
        return preprocessed, frame

    def test_static_and_continuously_shifted_peaks_are_found_blindly(self) -> None:
        _, first = self.detect([(9.00, 0.15, 8.0)], seed=1)
        _, second = self.detect([(9.12, 0.15, 8.0)], seed=2)
        first_reliable = reliable_peaks(first)
        second_reliable = reliable_peaks(second)
        self.assertEqual(len(first_reliable), 1)
        self.assertEqual(len(second_reliable), 1)
        self.assertAlmostEqual(first_reliable[0].two_theta, 9.00, delta=0.01)
        self.assertAlmostEqual(second_reliable[0].two_theta, 9.12, delta=0.01)

    def test_adjacent_peaks_are_jointly_resolved(self) -> None:
        _, frame = self.detect([(9.00, 0.15, 8.0), (9.22, 0.15, 5.0)], seed=3)
        fitted = reliable_peaks(frame)
        self.assertEqual(len(fitted), 2)
        np.testing.assert_allclose(
            [item.two_theta for item in fitted], [9.00, 9.22], atol=0.015
        )
        self.assertEqual(len({item.group_id for item in fitted}), 1)

    def test_peak_appearance_and_disappearance_is_not_filled_by_nearest_maximum(self) -> None:
        _, present = self.detect([(9.0, 0.15, 8.0)], seed=4)
        _, absent = self.detect([], seed=5)
        self.assertEqual(len(reliable_peaks(present)), 1)
        self.assertEqual(len(absent.peaks), 0)

    def test_baseline_and_relative_area_are_scale_equivariant(self) -> None:
        x, y = synthetic_pattern([(9.0, 0.15, 8.0)], seed=6)
        baseline = peak.asls_baseline(x, y)
        scaled_baseline = peak.asls_baseline(x, 7.0 * y)
        # Sparse AsLS solves are scale-equivariant up to ordinary solver
        # roundoff (about 1e-7 relative in the bundled SciPy build).
        np.testing.assert_allclose(scaled_baseline, 7.0 * baseline, rtol=2e-7, atol=5e-5)

        _, original_frame = self.detect([(9.0, 0.15, 8.0)], seed=6, scale=1.0)
        _, scaled_frame = self.detect([(9.0, 0.15, 8.0)], seed=6, scale=7.0)
        original = reliable_peaks(original_frame)
        scaled = reliable_peaks(scaled_frame)
        self.assertEqual(len(original), 1)
        self.assertEqual(len(scaled), 1)
        # This is the formal acceptance bound for global intensity scaling.
        self.assertLess(abs(original[0].relative_area - scaled[0].relative_area), 1e-3)

    def test_relative_area_correlation_map_is_invariant_to_global_intensity_scale(self) -> None:
        scans = [f"scan{index}" for index in range(5)]
        pressures = [0.0, 1.0]
        maps: list[np.ndarray] = []
        for scale in (1.0, 7.0):
            measurements: dict[float, peak.PeakFit] = {}
            for pressure_value, area_value, seed_value in (
                (0.0, 8.0, 61),
                (1.0, 4.0, 62),
            ):
                _, detected = self.detect(
                    [(9.0 + 0.03 * pressure_value, 0.15, area_value)],
                    seed=seed_value,
                    scale=scale,
                )
                reliable = reliable_peaks(detected)
                self.assertEqual(len(reliable), 1)
                measurements[pressure_value] = reliable[0]

            assignments: dict[tuple[str, float], peak.AssignedObservation] = {}
            for scan in scans:
                for pressure_value in pressures:
                    measurement = measurements[pressure_value]
                    assignments[(scan, pressure_value)] = assigned(
                        scan,
                        pressure_value,
                        "present",
                        relative_area=measurement.relative_area,
                        q=measurement.q,
                        fwhm_q=measurement.fwhm_q,
                    )
            matrices = peak.compute_track_correlations(
                assignments,
                scans,
                pressures,
                bootstrap_iterations=0,
                seed=0,
            )
            maps.append(matrices.area)

        difference = np.abs(maps[0] - maps[1])
        self.assertLess(float(np.nanmax(difference)), 1e-3)

    def test_different_sampling_and_duplicate_cleanup_are_deterministic(self) -> None:
        x = np.asarray([5.0, 5.2, 5.1, 5.1, 5.3, 5.4])
        y = np.asarray([1.0, 3.0, 2.0, 4.0, 5.0, 6.0])
        cleaned = peak.clean_xy(x, y)
        self.assertTrue(np.all(np.diff(cleaned.x) > 0))
        self.assertEqual(cleaned.duplicate_points_merged, 1)
        duplicate_index = int(np.flatnonzero(cleaned.x == 5.1)[0])
        self.assertAlmostEqual(cleaned.y[duplicate_index], 3.0)


class PeakTrackingAndMissingnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = peak.UniformPeakConfig(wavelength=WAVELENGTH, bootstrap_iterations=0)

    def test_continuous_trajectory_links_in_both_directions(self) -> None:
        pressures = [0.0, 1.0, 2.0, 3.0]
        by_pressure = {
            pressure: (consensus(f"p{index}", index, 1.0 + 0.02 * index),)
            for index, pressure in enumerate(pressures)
        }
        tracks = peak.link_consensus_bidirectional(by_pressure, pressures, self.config)
        official = [track for track in tracks if track.official]
        self.assertEqual(len(official), 1)
        self.assertFalse(official[0].ambiguous)
        self.assertEqual(len(official[0].nodes), 4)

    def test_crossing_tracks_are_flagged_not_silently_swapped(self) -> None:
        q_by_pressure = ((0.90, 1.10), (0.99, 1.01), (1.10, 0.90))
        by_pressure = {
            float(index): tuple(
                consensus(f"p{index}_{branch}", index, q, width=0.20)
                for branch, q in enumerate(q_values)
            )
            for index, q_values in enumerate(q_by_pressure)
        }
        tracks = peak.link_consensus_bidirectional(
            by_pressure, [0.0, 1.0, 2.0], self.config
        )
        self.assertTrue(any(track.ambiguous for track in tracks))
        self.assertFalse(any(track.official for track in tracks))
        self.assertFalse(
            any(len({node.pressure_index for node in track.nodes}) == 3 and not track.ambiguous for track in tracks)
        )

    def test_one_sided_absence_is_nan_and_presence_is_jaccard(self) -> None:
        scans = [f"scan{index}" for index in range(5)]
        assignments: dict[tuple[str, float], peak.AssignedObservation] = {}
        for index, scan in enumerate(scans):
            assignments[(scan, 0.0)] = assigned(
                scan, 0.0, "present", relative_area=0.2, q=1.0, fwhm_q=0.04
            )
            if index < 4:
                assignments[(scan, 1.0)] = assigned(
                    scan, 1.0, "present", relative_area=0.1, q=1.01, fwhm_q=0.04
                )
            else:
                assignments[(scan, 1.0)] = assigned(scan, 1.0, "absent")
        result = peak.compute_track_correlations(
            assignments, scans, [0.0, 1.0], bootstrap_iterations=0, seed=0
        )
        self.assertEqual(result.n_both_present[0, 1], 4)
        self.assertEqual(result.n10[0, 1], 1)
        self.assertEqual(result.required_support[0, 1], 5)
        self.assertTrue(math.isnan(result.area[0, 1]))
        self.assertTrue(math.isnan(result.location[0, 1]))
        self.assertAlmostEqual(result.presence[0, 1], 4.0 / 5.0)

    def test_missing_frame_is_unknown_not_a_numeric_zero(self) -> None:
        scans = [f"scan{index}" for index in range(6)]
        assignments: dict[tuple[str, float], peak.AssignedObservation] = {}
        for index, scan in enumerate(scans):
            assignments[(scan, 0.0)] = assigned(
                scan, 0.0, "present", relative_area=0.2, q=1.0, fwhm_q=0.04
            )
            assignments[(scan, 1.0)] = (
                assigned(scan, 1.0, "unknown")
                if index == 5
                else assigned(
                    scan, 1.0, "present", relative_area=0.1, q=1.01, fwhm_q=0.04
                )
            )
        result = peak.compute_track_correlations(
            assignments, scans, [0.0, 1.0], bootstrap_iterations=0, seed=0
        )
        self.assertEqual(result.n_unknown[0, 1], 1)
        self.assertEqual(result.n_available[0, 1], 5)
        self.assertEqual(result.n_both_present[0, 1], 5)
        self.assertAlmostEqual(result.area[0, 1], 0.5)
        self.assertGreater(result.location[0, 1], 0.0)
        self.assertAlmostEqual(result.presence[0, 1], 1.0)

    def test_correlations_pair_only_within_the_same_scan(self) -> None:
        scans = [f"scan{index}" for index in range(5)]
        assignments: dict[tuple[str, float], peak.AssignedObservation] = {}
        first_areas = [0.10, 0.20, 0.40, 0.80, 1.60]
        for scan, first_area in zip(scans, first_areas, strict=True):
            assignments[(scan, 0.0)] = assigned(
                scan, 0.0, "present", relative_area=first_area, q=1.0, fwhm_q=0.04
            )
            assignments[(scan, 1.0)] = assigned(
                scan, 1.0, "present", relative_area=2.0 * first_area, q=1.0, fwhm_q=0.04
            )
        result = peak.compute_track_correlations(
            assignments, scans, [0.0, 1.0], bootstrap_iterations=0, seed=0
        )
        self.assertAlmostEqual(result.area[0, 1], 0.5, places=15)
        self.assertAlmostEqual(result.location[0, 1], 1.0, places=15)

    def test_target_outside_frame_measured_q_range_is_unknown_not_absent(self) -> None:
        node = peak.TrackNode(
            consensus_id="p0",
            pressure=0.0,
            pressure_index=0,
            q=1.50,
            fwhm_q=0.04,
            relative_area=0.1,
            support=5,
        )
        track = peak.RadialTrack(
            track_id="radial_peak_001",
            channel="spots",
            nodes=(node,),
            official=True,
            ambiguous=False,
            minimum_pressure_support=1,
        )
        frame = peak.FramePeaks(
            frame=1,
            scan="scanA",
            pressure=0.0,
            channel="spots",
            peaks=(),
            pattern_valid=True,
            noise=1.0,
            total_positive_area=1.0,
            measured_q_min=1.0,
            measured_q_max=1.2,
        )
        observations = peak.assign_track_observations(
            [track],
            [frame],
            ["scanA"],
            [0.0],
            self.config,
        )
        observation = observations[track.track_id][("scanA", 0.0)]
        self.assertEqual(observation.state, "unknown")
        self.assertEqual(observation.reason, "outside_frame_measured_q_range")


class WindowGeometryAndCorrelationTests(unittest.TestCase):
    def test_window_geometry_is_data_agnostic_and_has_nonoverlap_control(self) -> None:
        spec = window.make_uniform_window_spec(2.0, 32.0)
        self.assertEqual(spec.starts_deg.size, 26)
        self.assertAlmostEqual(spec.width_deg, 5.0)
        self.assertAlmostEqual(spec.step_deg, 1.0)
        np.testing.assert_array_equal(spec.nonoverlap_indices, [0, 5, 10, 15, 20, 25])
        self.assertAlmostEqual(spec.starts_deg[0], 2.0)
        self.assertAlmostEqual(spec.ends_deg[-1], 32.0)

    def test_fixed_sliding_geometry_is_exactly_zero_to_five_then_one_to_six(
        self,
    ) -> None:
        spec = window.make_fixed_sliding_window_spec(32.108)
        self.assertEqual(spec.starts_deg.size, 28)
        np.testing.assert_array_equal(spec.starts_deg[:3], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(spec.ends_deg[:3], [5.0, 6.0, 7.0])
        self.assertEqual(spec.labels[:3], ("0-5", "1-6", "2-7"))
        self.assertEqual(spec.labels[-1], "27-32")
        np.testing.assert_array_equal(
            spec.nonoverlap_indices,
            [0, 5, 10, 15, 20, 25],
        )

    def test_fixed_sliding_geometry_supports_dynamic_window_count_and_tiny_edge_clip(
        self,
    ) -> None:
        grid = np.linspace(0.043, 12.0, 1201)
        residuals = np.vstack((np.sin(grid), np.cos(grid)))
        spec = window.make_fixed_sliding_window_spec(float(grid[-1]))
        features = window.build_window_features(grid, residuals, spec)
        self.assertEqual(features.signals.shape[:2], (2, 8))
        self.assertTrue(np.all(features.signal_valid))
        across = window.compute_across_frame_correlations(
            features,
            ["scanA", "scanA"],
            [0.0, 1.0],
        )
        self.assertEqual(across.acf_strict_by_scan.shape, (1, 8, 2, 2))
        within = window.compute_within_frame_correlations(
            features.fingerprints,
            ["scanA", "scanA"],
            [0.0, 1.0],
            nonoverlap_indices=spec.nonoverlap_indices,
        )
        self.assertEqual(within.by_frame.shape, (2, 8, 8))

    def test_common_grid_handles_different_sampling_without_extrapolation(self) -> None:
        first_x = np.linspace(0.0, 10.0, 101)
        second_x = np.linspace(0.5, 9.5, 130)
        batch = window.resample_common_grid(
            [first_x, second_x],
            [np.sin(first_x), np.sin(second_x)],
            coverage_fraction=1.0,
        )
        self.assertAlmostEqual(batch.interval.lower_deg, 0.5)
        self.assertAlmostEqual(batch.interval.upper_deg, 9.5)
        self.assertGreaterEqual(batch.grid_deg[0], 0.5)
        self.assertLessEqual(batch.grid_deg[-1], 9.5)
        self.assertTrue(np.all(np.isfinite(batch.values)))

    def test_shared_coverage_interval_excludes_minority_only_edge_range(self) -> None:
        from run_uniform_xy_correlations import _clip_xy_to_interval

        majority = [np.linspace(2.0, 8.0, 121) for _ in range(9)]
        minority = np.linspace(0.0, 10.0, 201)
        interval = window.common_coverage_interval(majority + [minority], 0.9)
        self.assertAlmostEqual(interval.lower_deg, 2.0)
        self.assertAlmostEqual(interval.upper_deg, 8.0)
        y = np.exp(-0.5 * ((minority - 1.0) / 0.05) ** 2)
        clipped_x, clipped_y = _clip_xy_to_interval(
            minority,
            y,
            interval.lower_deg,
            interval.upper_deg,
        )
        self.assertGreaterEqual(clipped_x[0], 2.0)
        self.assertLessEqual(clipped_x[-1], 8.0)
        self.assertLess(float(np.max(clipped_y)), 1e-20)

    def test_window_standardization_is_invariant_to_global_intensity_scale(self) -> None:
        grid = np.linspace(0.0, 12.0, 1201)
        residual = (
            np.sin(2.0 * np.pi * grid / 2.1)
            + 0.25 * np.sin(2.0 * np.pi * grid / 0.7)
            + 0.01 * grid
        )[None, :]
        spec = window.make_uniform_window_spec(0.0, 12.0)
        original, original_valid = window.standardize_residual_windows(grid, residual, spec)
        scaled, scaled_valid = window.standardize_residual_windows(grid, 9.0 * residual, spec)
        np.testing.assert_array_equal(original_valid, scaled_valid)
        np.testing.assert_allclose(original, scaled, atol=1e-12, equal_nan=True)

    def test_strict_and_shift_tolerant_outputs_remain_separate(self) -> None:
        spec = window.make_uniform_window_spec(0.0, 12.0)
        n_samples = 32
        phase = np.linspace(0.0, 2.0 * np.pi, n_samples)
        target = np.sin(phase) + 0.2 * np.sin(3.0 * phase)
        filler = np.cos(1.5 * phase)
        signals = np.tile(filler, (2, 26, 1))
        fingerprints = signals.copy()
        # At pressure 1, the motif has left strict window 0 and appears in its
        # immediate neighbor.  The strict primary must not silently follow it.
        signals[0, 0] = target
        signals[0, 1] = -target
        signals[1, 0] = -target
        signals[1, 1] = target
        fingerprints[0, 0] = target
        fingerprints[0, 1] = -target
        fingerprints[1, 0] = -target
        fingerprints[1, 1] = target
        features = window.WindowFeatures(
            spec=spec,
            sample_fraction=np.linspace(0.0, 1.0, n_samples),
            signals=signals,
            signal_valid=np.ones((2, 26), dtype=bool),
            fingerprints=fingerprints,
            fingerprint_valid=np.ones((2, 26), dtype=bool),
        )
        result = window.compute_across_frame_correlations(
            features, ["scanA", "scanA"], [0.0, 1.0]
        )
        self.assertAlmostEqual(result.acf_strict_by_scan[0, 0, 0, 1], -1.0, places=12)
        self.assertAlmostEqual(result.direct_strict_by_scan[0, 0, 0, 1], -1.0, places=12)
        self.assertAlmostEqual(result.shift_tolerant_by_scan[0, 0, 0, 1], 1.0, places=12)
        np.testing.assert_allclose(
            result.acf_strict_by_scan,
            np.swapaxes(result.acf_strict_by_scan, -1, -2),
            atol=1e-12,
            equal_nan=True,
        )

    def test_within_frame_retains_negative_correlations(self) -> None:
        phase = np.linspace(0.0, 2.0 * np.pi, 32)
        target = np.sin(phase) + 0.2 * np.sin(3.0 * phase)
        fingerprints = np.tile(np.cos(1.5 * phase), (1, 26, 1))
        fingerprints[0, 0] = target
        fingerprints[0, 1] = -target
        result = window.compute_within_frame_correlations(
            fingerprints, ["scanA"], [0.0]
        )
        self.assertAlmostEqual(result.by_frame[0, 0, 1], -1.0, places=12)
        self.assertAlmostEqual(result.by_frame[0, 1, 0], -1.0, places=12)
        np.testing.assert_array_equal(result.nonoverlap_indices, [0, 5, 10, 15, 20, 25])

    def test_aggregate_is_exactly_recomputable_and_deterministic(self) -> None:
        values = np.asarray(
            [
                [[1.0, value], [value, 1.0]]
                for value in (0.1, 0.3, 0.5, 0.7, 0.9)
            ]
        )
        first = window.aggregate_scan_matrices(values, n_bootstrap=50, seed=0)
        second = window.aggregate_scan_matrices(values, n_bootstrap=50, seed=0)
        expected = np.nanmedian(values, axis=0)
        np.testing.assert_allclose(first.median, expected, atol=1e-15)
        np.testing.assert_allclose(first.median, second.median, atol=0.0)
        np.testing.assert_allclose(first.ci_low, second.ci_low, atol=0.0)
        np.testing.assert_allclose(first.ci_high, second.ci_high, atol=0.0)

    def test_missing_window_support_is_nan_not_zero(self) -> None:
        values = np.full((5, 2, 2), np.nan)
        values[:, 0, 0] = 1.0
        availability = np.ones_like(values, dtype=bool)
        result = window.aggregate_scan_matrices(
            values, availability, n_bootstrap=0, seed=0
        )
        self.assertEqual(result.support[0, 1], 0)
        self.assertTrue(math.isnan(result.median[0, 1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
