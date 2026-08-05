#!/usr/bin/env python3
"""Compare correlation peak-detection profiles without generating map images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import compare_integrated_peaks as cip
import per_peak_correlation_maps as ppm
from run_correlation_suite import PER_PEAK_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PER_PEAK_PROFILES),
        default=sorted(PER_PEAK_PROFILES),
    )
    return parser.parse_args()


def profile_namespace(inputs: list[Path], profile: str) -> argparse.Namespace:
    original = sys.argv
    try:
        sys.argv = [
            "assess_peak_profiles",
            *[str(path) for path in inputs],
            "--out-dir",
            "/tmp/correlation-profile-assessment-unused",
            *PER_PEAK_PROFILES[profile],
        ]
        return ppm.parse_args()
    finally:
        sys.argv = original


def assess(inputs: list[Path], profile: str) -> dict[str, object]:
    started = time.perf_counter()
    args = profile_namespace(inputs, profile)
    detector_args = SimpleNamespace(**vars(args))
    files = cip.discover_xy_files(args.inputs)
    patterns = [cip.load_pattern(path, detector_args) for path in files]
    patterns = sorted(
        patterns,
        key=lambda pattern: (
            float("inf") if pattern.pressure_gpa is None else pattern.pressure_gpa,
            pattern.label,
        ),
    )
    cip.ensure_unique_pattern_labels(patterns)
    frame_rows: list[dict[str, object]] = []
    accepted = []
    all_candidates = []
    for pattern in patterns:
        detected = cip.detect_peaks(pattern, detector_args)
        kept = [peak for peak in detected if peak.confidence_tier != "C"]
        all_candidates.extend(detected)
        accepted.extend(kept)
        frame_rows.append({
            "label": pattern.label,
            "path": str(pattern.path),
            "accepted_peaks": len(kept),
            "tier_a": sum(peak.confidence_tier == "A" for peak in detected),
            "tier_b": sum(peak.confidence_tier == "B" for peak in detected),
            "tier_c": sum(peak.confidence_tier == "C" for peak in detected),
            "top_peaks_truncated": any(peak.top_peaks_truncated for peak in detected),
        })
    groups = cip.group_peaks(accepted, args.peak_match_tolerance)
    counts = [int(row["accepted_peaks"]) for row in frame_rows]
    return {
        "profile": profile,
        "profile_arguments": PER_PEAK_PROFILES[profile],
        "n_frames": len(patterns),
        "accepted_peaks_total": len(accepted),
        "candidate_peaks_total": len(all_candidates),
        "accepted_peaks_per_frame": counts,
        "accepted_peaks_per_frame_median": float(np.median(counts)) if counts else float("nan"),
        "peak_groups": len(groups),
        "group_position_min": min(groups) if groups else float("nan"),
        "group_position_max": max(groups) if groups else float("nan"),
        "any_top_peaks_truncated": any(bool(row["top_peaks_truncated"]) for row in frame_rows),
        "frames": frame_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    results = [assess(args.inputs, profile) for profile in args.profiles]
    report = {
        "inputs": [str(path.expanduser().resolve()) for path in args.inputs],
        "profiles": {str(row["profile"]): row for row in results},
    }
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    summary = {
        name: {
            "accepted_peaks_total": row["accepted_peaks_total"],
            "accepted_peaks_per_frame": row["accepted_peaks_per_frame"],
            "peak_groups": row["peak_groups"],
            "any_top_peaks_truncated": row["any_top_peaks_truncated"],
        }
        for name, row in report["profiles"].items()
    }
    print(json.dumps({"summary": summary, "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
