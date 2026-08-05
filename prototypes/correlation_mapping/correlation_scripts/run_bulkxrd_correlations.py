#!/usr/bin/env python3
"""Convert a BulkXRD result and run all four current correlation mappings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--source",
        choices=[
            "auto", "mean", "robust", "sigmaclip", "straightened",
            "straightened-robust", "clean", "spots", "residual",
        ],
        default="auto",
    )
    parser.add_argument("--dataset", default="")
    parser.add_argument("--wavelength-angstrom", type=float, default=None)
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--include-tier-c", action="store_true")
    parser.add_argument(
        "--profile",
        choices=["uotexrd-high-recall", "portable-conservative"],
        default="uotexrd-high-recall",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    input_path = args.input_h5.expanduser().resolve()
    out_dir = args.out_dir or (DEFAULT_RESULTS_DIR / input_path.stem)
    out_dir = out_dir.expanduser().resolve()
    xy_dir = out_dir / "00_bulkxrd_xy"
    suite_dir = out_dir / "correlation_suite"
    out_dir.mkdir(parents=True, exist_ok=True)

    convert = [
        sys.executable,
        str(SCRIPT_DIR / "bulkxrd_h5_to_xy.py"),
        str(input_path),
        "--out-dir",
        str(xy_dir),
        "--source",
        args.source,
    ]
    if args.dataset:
        convert.extend(["--dataset", args.dataset])
    if args.wavelength_angstrom is not None:
        convert.extend(["--wavelength-angstrom", str(args.wavelength_angstrom)])
    if args.include_excluded:
        convert.append("--include-excluded")
    if args.include_failed:
        convert.append("--include-failed")
    run(convert)

    suite = [
        sys.executable,
        str(SCRIPT_DIR / "run_correlation_suite.py"),
        "--out-dir",
        str(suite_dir),
        "--profile",
        args.profile,
        str(xy_dir),
    ]
    if args.include_tier_c:
        suite.append("--include-tier-c")
    run(suite)

    manifest = {
        "input_h5": str(input_path),
        "output_dir": str(out_dir),
        "source": args.source,
        "dataset": args.dataset,
        "include_tier_c": bool(args.include_tier_c),
        "profile": args.profile,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mappings": [
            "per-peak ROI-area similarity across frames",
            "per-peak location similarity across frames",
            "same-window ACF across frames",
            "window-to-window ACF within each frame",
        ],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"BulkXRD four-map correlation run complete: {out_dir}")


if __name__ == "__main__":
    main()
