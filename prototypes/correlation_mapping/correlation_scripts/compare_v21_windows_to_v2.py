#!/usr/bin/env python3
"""Prove that v2.1 Across/Within results are numerically unchanged from v2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from compare_uniform_correlation_runs_v21 import _compare_csv, _compare_npz, _json_safe


DEFAULT_TOLERANCE = 1.0e-10


def _channels(root: Path) -> tuple[str, ...]:
    try:
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ()
    return tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in manifest.get("channels", [])
            if str(item).strip()
        )
    )


def _window_files(root: Path, channels: tuple[str, ...]) -> set[Path]:
    found: set[Path] = set()
    for channel in channels:
        for family in ("across_frames", "within_frame"):
            family_root = root / channel / family
            if not family_root.is_dir():
                continue
            found.update(path.relative_to(root) for path in family_root.rglob("*.csv"))
            found.update(path.relative_to(root) for path in family_root.rglob("*.npz"))
    return found


def compare(v21_root: Path, v2_root: Path, tolerance: float) -> dict[str, object]:
    channels = _channels(v21_root)
    v2_channels = _channels(v2_root)
    comparable = tuple(channel for channel in channels if channel in v2_channels)
    left_files = _window_files(v21_root, comparable)
    right_files = _window_files(v2_root, comparable)
    missing_from_v21 = sorted(str(path) for path in right_files - left_files)
    missing_from_v2 = sorted(str(path) for path in left_files - right_files)
    common = sorted(left_files & right_files)
    failures: list[dict[str, object]] = []
    maximum = 0.0
    csv_files = 0
    npz_files = 0
    for relative in common:
        if relative.suffix.lower() == ".csv":
            csv_files += 1
            result = _compare_csv(v21_root / relative, v2_root / relative, tolerance)
        else:
            npz_files += 1
            result = _compare_npz(v21_root / relative, v2_root / relative, tolerance)
        difference = result.get("max_abs_difference", 0.0)
        if isinstance(difference, (int, float)) and math.isfinite(float(difference)):
            maximum = max(maximum, float(difference))
        if not result.get("passed", False):
            failures.append({"file": str(relative), **result})
    same_file_set = not missing_from_v21 and not missing_from_v2 and bool(common)
    passed = same_file_set and not failures and maximum <= tolerance
    return {
        "validator": "compare_v21_windows_to_v2-v1",
        "profile": "uniform-correlation-v2.1",
        "v21_run": str(v21_root.resolve()),
        "v2_run": str(v2_root.resolve()),
        "channels": list(comparable),
        "tolerance": tolerance,
        "passed": passed,
        "maximum_absolute_difference": maximum,
        "checks": {
            "comparable_channels_present": bool(comparable),
            "same_window_scientific_file_set": same_file_set,
            "all_window_values_match": not failures and bool(common),
            "maximum_difference_within_tolerance": maximum <= tolerance,
        },
        "details": {
            "matched_files": len(common),
            "csv_files_compared": csv_files,
            "npz_files_compared": npz_files,
            "missing_from_v21": missing_from_v21,
            "missing_from_v2": missing_from_v2,
            "failures": failures[:50],
        },
        "errors": [item.get("file", "window comparison failure") for item in failures]
        + (["window file sets differ"] if not same_file_set else []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v21_run", type=Path)
    parser.add_argument("v2_run", type=Path)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.v21_run.expanduser().resolve(),
        args.v2_run.expanduser().resolve(),
        args.tolerance,
    )
    payload = json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
