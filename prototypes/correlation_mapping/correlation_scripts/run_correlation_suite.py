#!/usr/bin/env python3
"""Run the current XRD correlation workflows into one organized folder."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

PER_PEAK_PROFILES = {
    "uotexrd-high-recall": [
        "--peak-match-tolerance", "0.02",
        "--micro-prominence", "0.005",
        "--raw-prominence", "0",
        "--shoulder-prominence", "0",
        "--matched-filter-prominence", "0.2",
        "--matched-filter-widths", "2,3,5,8",
        "--min-micro-snr", "0",
        "--min-shape-snr", "0.15",
        "--min-shape-contrast", "0.0005",
        "--shape-half-window", "0.16",
        "--top-peaks", "4000",
        "--prominence", "0.2",
        "--bump-prominence", "0.05",
        "--min-peak-height", "0",
        "--min-bump-rise", "0",
        "--distance", "1",
        "--merge-tolerance", "0.015",
        "--small-peak-max-prominence", "999",
        "--small-peak-max-width", "999",
    ],
    "portable-conservative": [
        "--peak-match-tolerance", "0.08",
        "--micro-prominence", "5",
        "--raw-prominence", "999",
        "--shoulder-prominence", "999",
        "--matched-filter-prominence", "999",
        "--matched-filter-widths", "2,3,5,8",
        "--min-micro-snr", "5",
        "--min-shape-snr", "2",
        "--min-shape-contrast", "0.01",
        "--shape-half-window", "0.16",
        "--top-peaks", "200",
        "--prominence", "10",
        "--bump-prominence", "10",
        "--min-peak-height", "3",
        "--min-bump-rise", "3",
        "--distance", "8",
        "--merge-tolerance", "0.05",
        "--small-peak-max-prominence", "999",
        "--small-peak-max-width", "999",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the organized XRD correlation suite.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/correlation_suite_20260617"),
        help="Main output folder containing the current correlation subfolders.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["Data/Cell_14_integrated", "Data/Cell_29_integrated"],
        help="Integrated .xy directories/files.",
    )
    parser.add_argument(
        "--include-tier-c",
        action="store_true",
        help="Include likely-artifact Tier C candidates in the per-peak correlation output.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PER_PEAK_PROFILES),
        default="uotexrd-high-recall",
        help=(
            "Peak-detection profile. uotexrd-high-recall preserves the latest UOTe settings; "
            "portable-conservative is a cross-dataset validation baseline."
        ),
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    common_inputs = [str(item) for item in args.inputs]

    per_peak_dir = args.out_dir / "01_per_peak_frame_correlation"
    same_window_dir = args.out_dir / "02_same_window_acf_across_frames"
    single_frame_dir = args.out_dir / "03_single_frame_window_acf"
    same_window_nonoverlap_dir = args.out_dir / "02_same_window_acf_across_frames_nonoverlap"
    single_frame_nonoverlap_dir = args.out_dir / "03_single_frame_window_acf_nonoverlap"
    per_peak_cmd = [
            python,
            str(SCRIPT_DIR / "per_peak_correlation_maps.py"),
            *common_inputs,
            "--out-dir",
            str(per_peak_dir),
            "--review-plots",
            *PER_PEAK_PROFILES[args.profile],
    ]
    if args.include_tier_c:
        per_peak_cmd.append("--include-tier-c")
    run(per_peak_cmd)

    def run_window(mode: str, out_dir: Path, step: str, extra: list[str] | None = None) -> None:
        cmd = [
            python,
            str(SCRIPT_DIR / "window_autocorrelation_correlations.py"),
            *common_inputs,
            "--out-dir",
            str(out_dir),
            "--mode",
            mode,
            "--window-width",
            "5",
            "--window-step",
            step,
            "--shift-tolerance",
            "1",
        ]
        if extra:
            cmd.extend(extra)
        run(cmd)

    run_window("same-window", same_window_dir, "1", ["--write-ncc-shift"])
    run_window("single-frame", single_frame_dir, "1")
    run_window(
        "same-window",
        same_window_nonoverlap_dir,
        "5",
        ["--window-start", "3", "--write-ncc-shift"],
    )
    run_window(
        "single-frame",
        single_frame_nonoverlap_dir,
        "5",
        ["--window-start", "3"],
    )

    with (args.out_dir / "README.txt").open("w") as handle:
        handle.write("XRD correlation suite\n")
        handle.write("=====================\n\n")
        handle.write(f"Peak-detection profile: {args.profile}\n")
        handle.write("Profile arguments: " + " ".join(PER_PEAK_PROFILES[args.profile]) + "\n\n")
        handle.write("01_per_peak_frame_correlation:\n")
        handle.write("  Two lower-triangle frame-vs-frame map sets per detected peak group:\n")
        handle.write("    per_peak_heatmaps/ and per_peak_matrices/: ROI-area similarity.\n")
        handle.write("    per_peak_position_heatmaps/ and per_peak_position_matrices/: peak-position similarity.\n\n")
        handle.write(f"    include-tier-c: {int(bool(args.include_tier_c))}\n\n")
        handle.write("02_same_window_acf_across_frames:\n")
        handle.write("  One lower-triangle frame-vs-frame ACF/Pearson map per 5 degree window.\n")
        handle.write("  Handles up to +/-1 degree pressure shift by comparing neighboring windows and taking the best match.\n")
        handle.write("  Also writes same-window NCC+shift maps under ncc_shift_heatmaps/ and ncc_shift_matrices/.\n\n")
        handle.write("03_single_frame_window_acf:\n")
        handle.write("  For each frame, compares all 5 degree windows within that single frame.\n\n")
        handle.write("02_same_window_acf_across_frames_nonoverlap and 03_single_frame_window_acf_nonoverlap:\n")
        handle.write("  Non-overlapping 5 degree windows starting at 3 deg, for checking overlap-driven high similarity.\n\n")
    print(f"Wrote correlation suite to {args.out_dir}")


if __name__ == "__main__":
    main()
