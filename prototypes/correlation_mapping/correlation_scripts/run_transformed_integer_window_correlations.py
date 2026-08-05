#!/usr/bin/env python3
"""Run the Log² integer-window suite and write strict-lower outputs.

This driver intentionally covers only window-to-window correlations.  It
applies ``log_squared`` to pooled-scale, bounded residuals after the unchanged
AsLS baseline and then reuses the unchanged fixed-window, ACF, Pearson,
aggregation, and plotting code. ROI-area and location calculations are outside
this driver.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from all_peak_frame_correlations import (
    write_lower_triangle_window_results,
    write_window_quicklooks,
)
from integer_window_correlations import (
    DEFAULT_POWDER_MANIFEST,
    DEFAULT_POWDER_PROFILE,
    DEFAULT_POWDER_ROOT,
    DEFAULT_SINGLE_MANIFEST,
    DEFAULT_SINGLE_PROFILE,
    DEFAULT_SINGLE_ROOT,
    DEFAULT_TRANSFORM_EPSILON_FLOOR,
    DEFAULT_TRANSFORM_SCALE_QUANTILE,
    IntensityTransformConfig,
    generate_integer_window_sources,
)
from uniform_correlation_io import json_ready, write_json


TRANSFORMED_MODES = ("log_squared",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--transform-mode",
        choices=TRANSFORMED_MODES,
        default="log_squared",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument(
        "--transform-scale-quantile",
        type=float,
        default=DEFAULT_TRANSFORM_SCALE_QUANTILE,
    )
    parser.add_argument(
        "--transform-epsilon-floor",
        type=float,
        default=DEFAULT_TRANSFORM_EPSILON_FLOOR,
    )
    parser.add_argument("--single-root", type=Path, default=DEFAULT_SINGLE_ROOT)
    parser.add_argument(
        "--single-manifest", type=Path, default=DEFAULT_SINGLE_MANIFEST
    )
    parser.add_argument(
        "--single-profile", type=Path, default=DEFAULT_SINGLE_PROFILE
    )
    parser.add_argument("--powder-root", type=Path, default=DEFAULT_POWDER_ROOT)
    parser.add_argument(
        "--powder-manifest", type=Path, default=DEFAULT_POWDER_MANIFEST
    )
    parser.add_argument(
        "--powder-profile", type=Path, default=DEFAULT_POWDER_PROFILE
    )
    parser.add_argument(
        "--no-quicklooks",
        action="store_true",
        help="Skip optional quicklook figures; strict-lower numerical outputs remain.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    output_root = args.out_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing to mix transformed results in non-empty directory: "
            f"{output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    transform = IntensityTransformConfig(
        mode=args.transform_mode,
        scale_quantile=args.transform_scale_quantile,
        epsilon_floor=args.transform_epsilon_floor,
    )
    source_audit = generate_integer_window_sources(
        output_root,
        workers=max(1, int(args.workers)),
        make_full_symmetric_plots=False,
        max_scans=args.max_scans,
        single_root=args.single_root,
        single_manifest=args.single_manifest,
        single_profile_path=args.single_profile,
        powder_root=args.powder_root,
        powder_manifest=args.powder_manifest,
        powder_profile_path=args.powder_profile,
        intensity_transform=transform,
    )
    lower_audit = write_lower_triangle_window_results(output_root)
    quicklook_audit = (
        {"skipped": True}
        if args.no_quicklooks
        else write_window_quicklooks(output_root)
    )
    strict_lower_verified = bool(
        lower_audit.get("roles") == 3
        and lower_audit.get("total_maps", 0) > 0
        and all(
            bool(role.get("strict_lower_triangle_only"))
            and bool(role.get("diagonal_omitted"))
            for role in lower_audit.get("role_audits", [])
        )
    )
    complete = {
        "status": "PASS" if strict_lower_verified else "FAIL",
        "scope": "window_to_window_only",
        "intensity_preprocessing": json_ready(transform.__dict__),
        "fixed_window_geometry": "0-5, 1-6, 2-7, ... degrees",
        "source_audit": source_audit,
        "strict_lower_output_audit": lower_audit,
        "quicklook_audit": quicklook_audit,
        "strict_lower_verified": strict_lower_verified,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_root / "TRANSFORMED_WINDOW_RUN_COMPLETE.json", complete)
    if not strict_lower_verified:
        raise RuntimeError(f"strict-lower validation failed: {lower_audit}")
    print(json.dumps(json_ready(complete), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
