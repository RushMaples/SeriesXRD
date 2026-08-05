#!/usr/bin/env python3
"""Run the frozen uniform-v2 synthetic suite and write auditable JSON evidence."""

from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from typing import Iterable

import test_uniform_xy_algorithms


CASE_TESTS = {
    "stationary_peak": "test_static_and_continuously_shifted_peaks_are_found_blindly",
    "continuous_peak_shift": "test_static_and_continuously_shifted_peaks_are_found_blindly",
    "known_area_ratio": "test_area_ratio_formula_and_invalid_values",
    "peak_appearance_disappearance": "test_peak_appearance_and_disappearance_is_not_filled_by_nearest_maximum",
    "adjacent_peaks": "test_adjacent_peaks_are_jointly_resolved",
    "crossing_peaks": "test_crossing_tracks_are_flagged_not_silently_swapped",
    "baseline_drift": "test_baseline_and_relative_area_are_scale_equivariant",
    "different_sampling_interval": "test_different_sampling_and_duplicate_cleanup_are_deterministic",
    "global_intensity_scaling": "test_relative_area_correlation_map_is_invariant_to_global_intensity_scale",
    "missing_frame": "test_missing_frame_is_unknown_not_a_numeric_zero",
}


def iter_tests(suite: unittest.TestSuite) -> Iterable[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_uniform_xy_algorithms)
    test_ids = [test.id() for test in iter_tests(suite)]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed_ids = {test.id() for test, _traceback in result.failures + result.errors}
    skipped_ids = {test.id() for test, _reason in result.skipped}
    case_status: dict[str, bool] = {}
    case_tests: dict[str, list[str]] = {}
    for case, method_name in CASE_TESTS.items():
        matches = [identifier for identifier in test_ids if identifier.endswith(method_name)]
        case_tests[case] = matches
        case_status[case] = bool(matches) and all(
            identifier not in failed_ids and identifier not in skipped_ids for identifier in matches
        )
    passed = result.wasSuccessful() and all(case_status.values())
    report = {
        "validator": "run_uniform_synthetic_validation-v1",
        "profile": "uniform-correlation-v2",
        "passed": passed,
        "test_count": result.testsRun,
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skipped_count": len(result.skipped),
        "checks": {
            "all_unittests_passed": result.wasSuccessful(),
            "all_required_synthetic_cases_passed": all(case_status.values()),
        },
        "cases": case_status,
        "case_tests": case_tests,
        "failed_tests": sorted(failed_ids),
        "skipped_tests": sorted(skipped_ids),
        "errors": [
            f"{test.id()}: {traceback.splitlines()[-1] if traceback else 'failure'}"
            for test, traceback in result.failures + result.errors
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
