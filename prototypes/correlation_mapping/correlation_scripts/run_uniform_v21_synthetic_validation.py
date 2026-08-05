#!/usr/bin/env python3
"""Run the independent v2.1 synthetic suite and write auditable JSON."""

from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from typing import Iterable

import test_uniform_v21_algorithms


CASE_TESTS = {
    "v2_upstream_config_equivalence": "test_v21_upstream_profile_is_exactly_v2",
    "continuous_peak": "test_continuous_peak_is_one_complete_official_segment",
    "local_ambiguity_split": "test_local_low_margin_ambiguity_splits_without_poisoning_clean_sides",
    "crossing_no_identity_switch": "test_crossing_peaks_are_cut_and_never_switch_identity",
    "missing_gap_bridge_and_cut": "test_missing_one_or_two_levels_bridge_but_three_levels_cut",
    "no_cross_cut_interpolation": "test_no_target_is_interpolated_across_an_ambiguity_cut",
    "unique_node_and_detection_assignment": "test_consensus_nodes_and_frame_detections_are_one_to_one",
    "tracking_area_independence": "test_tracking_is_independent_of_peak_area_scale",
    "global_intensity_scale_invariance": "test_relative_area_is_invariant_to_global_intensity_scaling",
    "no_cross_scan_pairing": "test_correlations_pair_only_within_the_same_scan",
    "missing_is_nan": "test_missing_and_insufficient_support_are_nan_not_zero",
    "deterministic_permutation_pressure_reversal": "test_permutation_pressure_reversal_and_rerun_are_deterministic",
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
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        test_uniform_v21_algorithms
    )
    test_ids = [test.id() for test in iter_tests(suite)]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed_ids = {
        test.id() for test, _traceback in result.failures + result.errors
    }
    skipped_ids = {test.id() for test, _reason in result.skipped}

    case_tests: dict[str, list[str]] = {}
    case_status: dict[str, bool] = {}
    for case, method_name in CASE_TESTS.items():
        matches = [identifier for identifier in test_ids if identifier.endswith(method_name)]
        case_tests[case] = matches
        case_status[case] = bool(matches) and all(
            identifier not in failed_ids and identifier not in skipped_ids
            for identifier in matches
        )

    passed = result.wasSuccessful() and all(case_status.values())
    report = {
        "validator": "run_uniform_v21_synthetic_validation-v1",
        "profile": "uniform-correlation-v2.1",
        "upstream_profile": "uniform-correlation-v2",
        "real_data_used": False,
        "passed": passed,
        "test_count": result.testsRun,
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skipped_count": len(result.skipped),
        "checks": {
            "all_unittests_passed": result.wasSuccessful(),
            "all_required_synthetic_cases_passed": all(case_status.values()),
            "identity_switch_tolerance": 0,
            "cross_cut_interpolation_allowed": False,
            "numeric_rerun_tolerance": 1.0e-10,
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
