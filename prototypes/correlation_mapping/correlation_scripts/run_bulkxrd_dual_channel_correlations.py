#!/usr/bin/env python3
"""Run the four correlation mappings on separate powder and spots channels."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BULKXRD_REPO = SCRIPT_DIR.parent / "BulkXRD"
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "results"
SCAN_RE = re.compile(r"(?:^|[^A-Za-z])scan[_\- ]*0*(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_h5",
        type=Path,
        help="BulkXRD reduced HDF5 or Step-1 analysis HDF5.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bulkxrd-repo", type=Path, default=DEFAULT_BULKXRD_REPO)
    parser.add_argument(
        "--powder-source",
        choices=["auto", "clean", "hybrid", "sigmaclip"],
        default="auto",
        help="auto uses sigmaclip when available, otherwise clean.",
    )
    parser.add_argument(
        "--powder-sensitivity",
        choices=["conservative", "normal", "sensitive"],
        default="normal",
    )
    parser.add_argument(
        "--spots-sensitivity",
        choices=["conservative", "normal", "sensitive"],
        default="sensitive",
    )
    parser.add_argument(
        "--group-by",
        choices=["auto", "none", "scan", "folder"],
        default="auto",
        help="Create independent frame maps per scan/folder; auto detects multiple scan tags.",
    )
    parser.add_argument(
        "--seed-group-by",
        choices=["auto", "none", "scan", "folder"],
        default="auto",
        help="Keep BulkXRD fit-seed propagation within each pressure series.",
    )
    parser.add_argument("--peak-match-tolerance", type=float, default=0.08)
    parser.add_argument("--position-tolerance", type=float, default=0.06)
    parser.add_argument("--min-peak-frame-count", type=int, default=1)
    parser.add_argument("--window-width", type=float, default=5.0)
    parser.add_argument("--window-step", type=float, default=1.0)
    parser.add_argument("--window-grid-step", type=float, default=0.02)
    parser.add_argument("--window-shift-tolerance", type=float, default=1.0)
    parser.add_argument("--max-half-window", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--wavelength-angstrom", type=float, default=None)
    return parser.parse_args()


def decode(value: object) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.") or "series"


def scan_label(name: str) -> str:
    match = SCAN_RE.search(name)
    return f"scan{int(match.group(1)):03d}" if match else "scan_unknown"


def folder_label(name: str) -> str:
    normalized = name.replace("\\", "/")
    parent = normalized.rsplit("/", 1)[0] if "/" in normalized else ""
    return safe_name(parent.rsplit("/", 1)[-1]) if parent else "folder_unknown"


def resolved_group_by(requested: str, names: list[str]) -> str:
    if requested != "auto":
        return requested
    scans = {scan_label(name) for name in names}
    scans.discard("scan_unknown")
    return "scan" if len(scans) > 1 else "none"


def inspect_input(path: Path) -> tuple[str, list[str], np.ndarray]:
    with h5py.File(path, "r") as h5:
        patterns = h5.get("patterns")
        background = h5.get("background")
        if patterns is not None and "intensity" in patterns and "intensity_robust" in patterns:
            input_kind = "reduced"
            n = int(patterns["intensity"].shape[0])
        elif (
            background is not None
            and "clean" in background
            and "spot_residual" in background
            and "radial" in h5
        ):
            input_kind = "analysis"
            n = int(background["clean"].shape[0])
        else:
            raise ValueError(
                "Dual-channel correlation needs either a reduced BulkXRD HDF5 with "
                "/patterns/intensity plus /patterns/intensity_robust, or a Step-1 "
                "analysis HDF5 with /radial, /background/clean, and "
                "/background/spot_residual"
            )
        frames = h5.get("frames")
        if frames is not None and "filename" in frames:
            names = [decode(value) for value in frames["filename"][:]]
        else:
            names = [f"frame_{index:04d}" for index in range(n)]
        pressure = (
            np.asarray(frames["pressure"][:], dtype=float)
            if frames is not None and "pressure" in frames
            else np.full(n, np.nan)
        )
    if len(names) != n or len(pressure) != n:
        raise ValueError("BulkXRD frame metadata length does not match the pattern stack")
    return input_kind, names, pressure


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_conversion_groups(manifest_path: Path, requested: str) -> tuple[str, OrderedDict[str, list[Path]]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["original_filename"] for row in rows]
    group_by = resolved_group_by(requested, names)
    groups: OrderedDict[str, list[Path]] = OrderedDict()
    root = manifest_path.parent
    for row in rows:
        name = row["original_filename"]
        if group_by == "scan":
            key = scan_label(name)
        elif group_by == "folder":
            key = folder_label(name)
        else:
            key = "all"
        groups.setdefault(key, []).append(root / row["output_xy"])
    return group_by, groups


def run_window_maps(
    xy_dir: Path,
    output_root: Path,
    group_by: str,
    args: argparse.Namespace,
) -> dict[str, int | str]:
    resolved, groups = read_conversion_groups(xy_dir / "conversion_manifest.csv", group_by)
    multiple = len(groups) > 1
    for series, files in groups.items():
        same_out = output_root / "02_same_window_acf_across_frames"
        within_out = output_root / "03_single_frame_window_acf"
        if multiple:
            same_out = same_out / safe_name(series)
            within_out = within_out / safe_name(series)
        common = [str(path) for path in files]
        run([
            sys.executable,
            str(SCRIPT_DIR / "window_autocorrelation_correlations.py"),
            *common,
            "--out-dir", str(same_out),
            "--mode", "same-window",
            "--window-width", str(args.window_width),
            "--window-step", str(args.window_step),
            "--grid-step", str(args.window_grid_step),
            "--shift-tolerance", str(args.window_shift_tolerance),
            "--write-ncc-shift",
        ])
        run([
            sys.executable,
            str(SCRIPT_DIR / "window_autocorrelation_correlations.py"),
            *common,
            "--out-dir", str(within_out),
            "--mode", "single-frame",
            "--window-width", str(args.window_width),
            "--window-step", str(args.window_step),
            "--grid-step", str(args.window_grid_step),
            "--shift-tolerance", str(args.window_shift_tolerance),
        ])
    return {"group_by": resolved, "series": len(groups)}


def peak_stats(path: Path) -> dict[str, int | str]:
    with h5py.File(path, "r") as h5:
        peaks = h5["peaks"]
        flags = np.asarray(peaks["flag"][:], dtype=int)
        return {
            "source": decode(peaks.attrs.get("source", "unknown")),
            "total": int(flags.size),
            "good": int(np.count_nonzero(flags == 0)),
            "frames": int(len(peaks["counts"])),
        }


def map_counts(channel_dir: Path) -> dict[str, int]:
    return {
        "per_peak_area": len(list((channel_dir / "01_per_peak_frame_correlation").rglob("per_peak_matrices/*.csv"))),
        "per_peak_location": len(list((channel_dir / "01_per_peak_frame_correlation").rglob("per_peak_position_matrices/*.csv"))),
        "same_window_across_frames": len(list((channel_dir / "02_same_window_acf_across_frames").rglob("matrices/*.csv"))),
        "window_to_window_within_frame": len(list((channel_dir / "03_single_frame_window_acf").rglob("matrices/*.csv"))),
    }


def main() -> None:
    args = parse_args()
    input_h5 = args.input_h5.expanduser().resolve()
    if not input_h5.is_file():
        raise SystemExit(f"BulkXRD HDF5 not found: {input_h5}")
    input_kind, names, pressure = inspect_input(input_h5)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.out_dir or (DEFAULT_RESULTS_DIR / f"{input_h5.stem}_dual_channel_{stamp}")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    bulkxrd_repo = args.bulkxrd_repo.expanduser().resolve()
    if not (bulkxrd_repo / "bulkxrd").is_dir():
        raise SystemExit(f"BulkXRD package not found under {bulkxrd_repo}")
    sys.path.insert(0, str(bulkxrd_repo))
    from bulkxrd.analysis.background import run_background_separation
    from bulkxrd.analysis.peaks import run_peak_fitting

    base_analysis = analysis_dir / "background_analysis.h5"
    powder_analysis = analysis_dir / "powder_peaks_analysis.h5"
    spots_analysis = analysis_dir / "spots_peaks_analysis.h5"
    if input_kind == "reduced":
        run_background_separation(
            input_h5,
            base_analysis,
            max_half_window=args.max_half_window,
            num_workers=args.num_workers,
        )
    else:
        base_analysis = input_h5
    with h5py.File(base_analysis, "r") as h5:
        powder_source = args.powder_source
        if powder_source == "auto":
            powder_source = "sigmaclip" if "background/sigmaclip_residual" in h5 else "clean"

    detected_group_by = resolved_group_by(args.group_by, names)
    seed_group_by = args.seed_group_by
    if seed_group_by == "auto":
        seed_group_by = "scan" if detected_group_by == "scan" else "none"
    seed_axis = "pressure" if np.any(np.isfinite(pressure)) else "frame"
    run_peak_fitting(
        base_analysis,
        powder_analysis,
        source=powder_source,
        sensitivity=args.powder_sensitivity,
        seed_tracking_axis=seed_axis,
        seed_group_by=seed_group_by,
        num_workers=args.num_workers,
    )
    run_peak_fitting(
        base_analysis,
        spots_analysis,
        source="spots",
        sensitivity=args.spots_sensitivity,
        auto_range=False,
        seed_tracking_axis=seed_axis,
        seed_group_by=seed_group_by,
        num_workers=args.num_workers,
    )

    channel_specs = {
        "powder": (powder_analysis, powder_source),
        "spots": (spots_analysis, "spots"),
    }
    channel_results: dict[str, object] = {}
    for channel, (analysis_h5, export_source) in channel_specs.items():
        channel_dir = output_root / channel
        xy_dir = channel_dir / "00_bulkxrd_xy"
        peak_dir = channel_dir / "01_per_peak_frame_correlation"
        channel_dir.mkdir(parents=True, exist_ok=True)
        convert = [
            sys.executable,
            str(SCRIPT_DIR / "bulkxrd_h5_to_xy.py"),
            str(analysis_h5),
            "--out-dir", str(xy_dir),
            "--source", export_source,
        ]
        if args.wavelength_angstrom is not None:
            convert.extend(["--wavelength-angstrom", str(args.wavelength_angstrom)])
        run(convert)

        peak_map = [
            sys.executable,
            str(SCRIPT_DIR / "bulkxrd_fitted_peak_correlations.py"),
            str(analysis_h5),
            "--out-dir", str(peak_dir),
            "--group-by", args.group_by,
            "--peak-match-tolerance", str(args.peak_match_tolerance),
            "--position-tolerance", str(args.position_tolerance),
            "--min-frame-count", str(args.min_peak_frame_count),
        ]
        if args.wavelength_angstrom is not None:
            peak_map.extend(["--wavelength-angstrom", str(args.wavelength_angstrom)])
        run(peak_map)
        grouping = run_window_maps(xy_dir, channel_dir, args.group_by, args)
        channel_results[channel] = {
            "analysis_h5": str(analysis_h5),
            "peak_fitting": peak_stats(analysis_h5),
            "window_grouping": grouping,
            "map_counts": map_counts(channel_dir),
        }

    manifest = {
        "input_h5": str(input_h5),
        "input_kind": input_kind,
        "bulkxrd_repo": str(bulkxrd_repo),
        "bulkxrd_commit": "unknown",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "powder_source": powder_source,
        "spots_source": "background/spot_residual",
        "seed_tracking_axis": seed_axis,
        "seed_group_by": seed_group_by,
        "requested_group_by": args.group_by,
        "resolved_group_by": detected_group_by,
        "parameters": {
            "peak_match_tolerance_deg": args.peak_match_tolerance,
            "position_tolerance_deg": args.position_tolerance,
            "min_peak_frame_count": args.min_peak_frame_count,
            "window_width_deg": args.window_width,
            "window_step_deg": args.window_step,
            "window_grid_step_deg": args.window_grid_step,
            "window_shift_tolerance_deg": args.window_shift_tolerance,
            "max_half_window_bins": args.max_half_window,
            "powder_sensitivity": args.powder_sensitivity,
            "spots_sensitivity": args.spots_sensitivity,
        },
        "channels": channel_results,
        "mappings": {
            "per_peak_area_across_frames": "BulkXRD /peaks/area",
            "per_peak_location_across_frames": "BulkXRD /peaks/center and /peaks/center_err",
            "same_window_across_frames": "window ACF plus shift-tolerant NCC",
            "window_to_window_within_frame": "within-frame window ACF",
        },
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=bulkxrd_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest["bulkxrd_commit"] = commit
    except (OSError, subprocess.CalledProcessError):
        pass
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "README.txt").write_text(
        "BulkXRD dual-channel correlation run\n"
        "====================================\n\n"
        f"Input: {input_h5}\n"
        f"Input kind: {input_kind}\n"
        f"Powder fit/pattern source: {powder_source}\n"
        "Spots fit/pattern source: background/spot_residual (mean - robust)\n\n"
        "Each channel contains:\n"
        "  01_per_peak_frame_correlation: fitted area and location maps across frames\n"
        "  02_same_window_acf_across_frames: same radial window across frames\n"
        "  03_single_frame_window_acf: window-to-window maps within each frame\n\n"
        "The powder and spots results must be interpreted separately; they are not "
        "two interchangeable views of the same signal.\n",
        encoding="utf-8",
    )
    print(f"Dual-channel four-map correlation run complete: {output_root}")


if __name__ == "__main__":
    main()
