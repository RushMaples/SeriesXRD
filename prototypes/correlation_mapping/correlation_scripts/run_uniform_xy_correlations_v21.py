#!/usr/bin/env python3
"""Run frozen, data-agnostic ``uniform-correlation-v2.1`` XY analysis.

V2.1 changes only consensus trajectory tracking: edge evidence is evaluated
locally and ambiguous links become explicit CUT boundaries.  Upstream peak
finding, per-segment similarity formulas, and across/within-frame algorithms
are the frozen v2 implementations.  Both the existing Handoff layout and a
generic direct manifest are supported through input-only adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

import uniform_window_core as uw
from run_uniform_xy_correlations import (
    _posthoc_reference_matches,
    _process_channel_patterns,
    _scan_level_window_rows,
)
from uniform_correlation_io import (
    build_artifact_index,
    directory_sha256,
    file_sha256,
    json_ready,
    write_json,
    write_rows_csv,
)
from uniform_peak_analysis_v21 import analyze_per_peak_v21
from uniform_profile_binding_v21 import PROFILE_NAME, bind_frozen_profile_v21
from uniform_result_writer_v21 import (
    write_across_results,
    write_per_peak_results,
    write_within_results,
)
from uniform_xy_input_v21 import read_input_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = SCRIPT_DIR / "configs" / "uniform-correlation-v2.1.json"


def _progress(message: str) -> None:
    print(f"[uniform-v2.1] {message}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--input-mode", choices=("auto", "direct", "handoff"), default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wavelength-A", type=float, required=True)
    parser.add_argument("--channels", default="spots,fit")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--reference-tracks", type=Path, default=None)
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=None,
        help=(
            "Protected v2 result root: always record before/after integrity; "
            "compare window arrays only for Handoff input."
        ),
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-plots", action="store_true", help="Operational reproducibility run only.")
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="EXPERIMENTAL smoke-test subset; official runs use every scan.",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Required for a subset run. Scientific thresholds remain frozen.",
    )
    return parser.parse_args(argv)


def load_frozen_profile(path: Path) -> tuple[dict[str, Any], str, bytes]:
    raw = path.expanduser().resolve().read_bytes()
    profile = json.loads(raw)
    if profile.get("profile") != PROFILE_NAME:
        raise ValueError(f"expected profile {PROFILE_NAME!r}, got {profile.get('profile')!r}")
    if profile.get("status") != "OFFICIAL_FROZEN":
        raise ValueError("official runner requires a frozen profile")
    return profile, hashlib.sha256(raw).hexdigest(), raw


def _environment_manifest() -> dict[str, Any]:
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR.parent.parent,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
    except OSError:
        git_head = ""
    code_files = [
        Path(__file__).resolve(),
        SCRIPT_DIR / "uniform_peak_core.py",
        SCRIPT_DIR / "uniform_peak_tracking_v21.py",
        SCRIPT_DIR / "uniform_peak_analysis_v21.py",
        SCRIPT_DIR / "uniform_window_core.py",
        SCRIPT_DIR / "uniform_profile_binding.py",
        SCRIPT_DIR / "uniform_profile_binding_v21.py",
        SCRIPT_DIR / "uniform_result_writer.py",
        SCRIPT_DIR / "uniform_result_writer_v21.py",
        SCRIPT_DIR / "uniform_xy_input.py",
        SCRIPT_DIR / "uniform_xy_input_v21.py",
        SCRIPT_DIR / "uniform_correlation_io.py",
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "git_head": git_head,
        "code_sha256": {path.name: file_sha256(path) for path in code_files},
    }


def _array_difference(first: np.ndarray, second: np.ndarray) -> tuple[bool, float, str]:
    if first.shape != second.shape:
        return False, math.inf, "shape_mismatch"
    if first.dtype.kind in "biufc" and second.dtype.kind in "biufc":
        left = np.asarray(first, dtype=float)
        right = np.asarray(second, dtype=float)
        if not np.array_equal(np.isfinite(left), np.isfinite(right)):
            return False, math.inf, "finite_mask_mismatch"
        finite = np.isfinite(left)
        maximum = float(np.max(np.abs(left[finite] - right[finite]))) if np.any(finite) else 0.0
        return maximum <= 1.0e-10, maximum, "numeric"
    equal = bool(np.array_equal(first, second))
    return equal, 0.0 if equal else math.inf, "exact"


def _compare_window_npz(v2_root: Path, v21_root: Path, channels: Sequence[str]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked_arrays = 0
    maximum = 0.0
    for channel in channels:
        for relative in (
            Path(channel) / "across_frames" / "across_frame_matrices.npz",
            Path(channel) / "within_frame" / "within_frame_matrices.npz",
        ):
            left_path = v2_root / relative
            right_path = v21_root / relative
            if not left_path.is_file() or not right_path.is_file():
                failures.append(
                    {
                        "file": str(relative),
                        "reason": "missing_file",
                        "v2_exists": left_path.is_file(),
                        "v21_exists": right_path.is_file(),
                    }
                )
                continue
            with np.load(left_path, allow_pickle=False) as left, np.load(
                right_path, allow_pickle=False
            ) as right:
                if sorted(left.files) != sorted(right.files):
                    failures.append(
                        {
                            "file": str(relative),
                            "reason": "array_keys",
                            "v2_keys": sorted(left.files),
                            "v21_keys": sorted(right.files),
                        }
                    )
                    continue
                for key in sorted(left.files):
                    checked_arrays += 1
                    passed, difference, reason = _array_difference(left[key], right[key])
                    if math.isfinite(difference):
                        maximum = max(maximum, difference)
                    if not passed:
                        failures.append(
                            {
                                "file": str(relative),
                                "array": key,
                                "reason": reason,
                                "max_abs_difference": difference,
                            }
                        )
                        if len(failures) >= 50:
                            break
    return {
        "validator": "compare_uniform_v2_v21_windows-v1",
        "tolerance": 1.0e-10,
        "v2_root": str(v2_root),
        "v21_root": str(v21_root),
        "arrays_compared": checked_arrays,
        "maximum_absolute_difference": maximum,
        "failures": failures,
        "passed": checked_arrays > 0 and not failures and maximum <= 1.0e-10,
    }


def _write_report(
    path: Path,
    metrics: Mapping[str, Any],
    profile_hash: str,
    official: bool,
    dataset_label: str,
) -> None:
    channel_lines: list[str] = []
    for channel, channel_metrics in metrics.items():
        peak = channel_metrics.get("per_peak", {})
        across = channel_metrics.get("across_frames", {}).get("families", {})
        within = channel_metrics.get("within_frame", {})
        channel_lines.extend(
            [
                f"### {channel}",
                "",
                f"- Official unambiguous segments: {peak.get('official_segments', 0)}.",
                f"- Quarantined ambiguous nodes: {peak.get('quarantined_nodes', 0)}.",
                f"- Strict-ACF median window AUC: {across.get('acf_strict', {}).get('median_window_auc', math.nan):.3f}.",
                f"- Within-frame non-overlap median: {within.get('nonoverlap_pair_median', math.nan):.3f}.",
                "",
            ]
        )
    status = "OFFICIAL_FROZEN" if official else "EXPERIMENTAL"
    text = f"""# {dataset_label} uniform-correlation-v2.1 report

## Run identity

- Status: **{status}**
- Frozen v2.1 profile SHA256: `{profile_hash}`
- V2.1 changes only tracking: ambiguous candidate edges are cut locally.
- V2 peak detection, fitting, consensus, similarity, Across-frame, and Within-frame formulas are unchanged.
- No hand-curated peak position, pressure range, or slope guided detection or tracking.

## Beginner summary

A segment is a pressure interval over which one radial peak identity could be followed without an ambiguous link. A gray cell outside that interval is **not zero**: it means this segment does not provide a justified comparison there. Heatmaps describe only the unambiguously tracked peak subset, not every XRD peak.

{chr(10).join(channel_lines)}
## What was cut and why

Every proposed connection is recorded in `per_peak/link_evidence.csv`. The local cut reasons are `cut_one_way`, `cut_low_margin`, `cut_order_crossing`, `cut_missing_too_long`, and `cut_outside_gate`. Competing low-margin nodes are listed in `per_peak/quarantined_nodes.csv` and remain unknown. The retention distribution is in `per_peak/selection_audit.csv` and `selection_audit_summary.csv`.

## Interpretation limit

- Area/location maps are conditional on the same scan containing a reliable measurement at both pressures.
- Missing, unknown, support-insufficient, outside-segment, and CUT-separated values stay NaN.
- A mathematically valid correlation does not by itself prove a phase transition.
- Because this method revision was motivated by the UOTe analysis, the result is a method-development reanalysis; independent data are needed for confirmatory use.

See `algorithm_config.json`, `run_manifest.json`, `validation/`, `comparison_to_v2/`, and each map's support files for the audit trail.
"""
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start = time.time()
    input_root = args.input_root.expanduser().resolve()
    manifest = (args.manifest or input_root / "manifest.csv").expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {out_dir}")
    if args.max_scans is not None and not args.experimental:
        raise SystemExit("--max-scans requires --experimental; official runs use every scan")
    channels = [item.strip() for item in args.channels.split(",") if item.strip()]
    if not channels:
        raise SystemExit("--channels must contain at least one channel")

    profile, profile_hash, profile_raw = load_frozen_profile(args.profile)
    bound = bind_frozen_profile_v21(profile, float(args.wavelength_A))
    peak_config = bound.peak_config
    window_config = bound.window_config
    dataset = read_input_dataset(
        input_root,
        manifest,
        channels,
        input_mode=args.input_mode,
        max_scans=args.max_scans,
    )
    frames = list(dataset.frames)
    pressures = list(dataset.pressures)
    scans = list(dataset.scans)

    out_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("validation", "robustness", "comparison_to_v2", "comparison_to_legacy"):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
    (out_dir / "algorithm_config.json").write_bytes(profile_raw)
    official = not args.experimental and args.max_scans is None
    dataset_label = args.dataset_label or input_root.name

    v2_root: Path | None = None
    v2_integrity: dict[str, Any] | None = None
    if args.v2_root is not None:
        v2_root = args.v2_root.expanduser().resolve()
        if out_dir == v2_root or v2_root in out_dir.parents:
            raise SystemExit("v2.1 output directory must not be inside the protected v2 root")
        before_count, before_sha = directory_sha256(v2_root)
        v2_integrity = {
            "path": str(v2_root),
            "files_before": before_count,
            "sha256_before": before_sha,
        }

    metrics: dict[str, Any] = {}
    inventory_rows: list[dict[str, Any]] = []
    scan_metric_rows: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] = []

    for channel in channels:
        channel_start = time.time()
        paths = list(dataset.paths_by_channel[channel])
        _progress(f"{channel}: unchanged v2 peak extraction start ({len(paths)} frames)")
        processed, channel_inventory, coverage_interval = _process_channel_patterns(
            paths,
            frames,
            channel,
            peak_config,
            minimum_points=bound.minimum_points_per_pattern,
            coverage_fraction=window_config.coverage_fraction,
            workers=max(1, int(args.workers)),
        )
        inventory_rows.extend(channel_inventory)
        frame_peaks = [item["frame_peaks"] for item in processed]
        _progress(f"{channel}: edge-segmented per-peak analysis/write start")
        peak_analysis = analyze_per_peak_v21(
            frame_peaks,
            pressures,
            scans,
            peak_config,
            bound.tracking_config,
            bootstrap_iterations=peak_config.bootstrap_iterations,
            seed=peak_config.random_seed,
            official_only=True,
        )
        channel_root = out_dir / channel
        per_peak_metrics = write_per_peak_results(
            channel_root,
            channel,
            peak_analysis,
            frame_peaks,
            scans=scans,
            pressures=pressures,
            make_plots=not args.no_plots,
            tracking_result=peak_analysis.tracking_result,
        )
        if args.reference_tracks is not None:
            posthoc_rows.extend(
                _posthoc_reference_matches(
                    peak_analysis,
                    args.reference_tracks.expanduser().resolve(),
                    peak_config.wavelength,
                    channel,
                )
            )

        batch = uw.resample_common_grid(
            [item["x"] for item in processed],
            [item["residual"] for item in processed],
            coverage_fraction=window_config.coverage_fraction,
            coverage_interval=coverage_interval,
        )
        features = uw.build_window_features(batch.grid_deg, batch.values, config=window_config)
        frame_scans = [frame.scan for frame in frames]
        frame_pressures = [frame.pressure for frame in frames]
        across = uw.compute_across_frame_correlations(
            features, frame_scans, frame_pressures, config=window_config
        )
        across_metrics = write_across_results(
            channel_root,
            channel,
            across,
            n_bootstrap=window_config.bootstrap_iterations,
            seed=window_config.random_seed,
            confidence=window_config.confidence,
            minimum_distinct_gaps=window_config.minimum_distinct_pressure_gaps,
            minimum_group_values=window_config.minimum_supported_group_values,
            near_gap_quantile=window_config.near_gap_quantile,
            far_gap_quantile=window_config.far_gap_quantile,
            make_plots=not args.no_plots,
        )
        within = uw.compute_within_frame_correlations(
            features.fingerprints,
            frame_scans,
            frame_pressures,
            nonoverlap_indices=features.spec.nonoverlap_indices,
            config=window_config,
        )
        within_metrics = write_within_results(
            channel_root,
            channel,
            within,
            features.spec,
            frame_ids=[frame.frame for frame in frames],
            frame_scans=frame_scans,
            frame_pressures=frame_pressures,
            n_bootstrap=window_config.bootstrap_iterations,
            seed=window_config.random_seed,
            confidence=window_config.confidence,
            make_plots=not args.no_plots,
        )
        scan_metric_rows.extend(_scan_level_window_rows(across, channel))
        metrics[channel] = {
            "per_peak": per_peak_metrics,
            "across_frames": across_metrics,
            "within_frame": within_metrics,
            "input": {
                "frames": len(frames),
                "analysis_min_deg": float(batch.grid_deg[0]),
                "analysis_max_deg": float(batch.grid_deg[-1]),
                "grid_step_deg": batch.grid_step_deg,
                "coverage_required_frames": batch.interval.required_frames,
                "coverage_total_frames": batch.interval.total_frames,
                "valid_signal_windows": int(np.count_nonzero(features.signal_valid)),
                "valid_acf_windows": int(np.count_nonzero(features.fingerprint_valid)),
            },
            "elapsed_seconds": time.time() - channel_start,
        }
        _progress(f"{channel}: complete")

    write_rows_csv(out_dir / "input_inventory.csv", inventory_rows)
    write_rows_csv(out_dir / "robustness" / "scan_level_metrics.csv", scan_metric_rows)
    if posthoc_rows:
        write_rows_csv(
            out_dir / "comparison_to_legacy" / "posthoc_reference_track_matches.csv",
            posthoc_rows,
        )

    window_equivalence: dict[str, Any] | None = None
    if v2_root is not None and v2_integrity is not None:
        after_count, after_sha = directory_sha256(v2_root)
        v2_integrity.update(
            {
                "files_after": after_count,
                "sha256_after": after_sha,
                "unchanged": after_count == v2_integrity["files_before"]
                and after_sha == v2_integrity["sha256_before"],
            }
        )
        write_json(out_dir / "comparison_to_v2" / "v2_integrity.json", v2_integrity)
        if dataset.input_mode == "handoff":
            window_equivalence = _compare_window_npz(v2_root, out_dir, channels)
            write_json(
                out_dir / "comparison_to_v2" / "across_within_comparison.json",
                window_equivalence,
            )
        else:
            write_json(
                out_dir / "comparison_to_v2" / "across_within_comparison_not_applicable.json",
                {
                    "passed": None,
                    "reason": "protected_v2_root_is_not_the_same_direct_manifest_dataset",
                    "input_mode": dataset.input_mode,
                    "v2_root": str(v2_root),
                },
            )

    run_manifest = {
        "profile": PROFILE_NAME,
        "algorithm_version": "2.1.0",
        "upstream_algorithm_version": "2.0.0",
        "profile_status": "OFFICIAL_FROZEN" if official else "EXPERIMENTAL",
        "profile_sha256": profile_hash,
        "upstream_profile_sha256": bound.upstream_profile_sha256,
        "execution_semantics_sha256": bound.semantic_sha256,
        "resolved_algorithm_semantics": bound.resolved_semantics,
        "config_binding_audit": bound.binding_audit,
        "config_binding_checks": {
            "all_peak_config_fields_explicitly_bound": True,
            "all_window_config_fields_explicitly_bound": True,
            "all_segmented_tracking_fields_explicitly_bound": True,
            "all_tracking_policy_fields_explicitly_bound": True,
            "peak_config_field_count": len(bound.binding_audit["peak_config"]),
            "window_config_field_count": len(bound.binding_audit["window_config"]),
            "segmented_tracking_field_count": len(
                bound.binding_audit["segmented_tracking_config"]
            ),
            "tracking_policy_field_count": len(bound.binding_audit["tracking_policy"]),
        },
        "wavelength_A": float(args.wavelength_A),
        "dataset_label": dataset_label,
        "input_root": str(input_root),
        "input_mode": dataset.input_mode,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "channels": channels,
        "frames": len(frames),
        "scans": scans,
        "pressure_gpa": pressures,
        "excluded_manifest_rows": dataset.excluded_rows,
        "workers": max(1, int(args.workers)),
        "plots_written": not args.no_plots,
        "reference_tracks_role": "posthoc_annotation_only" if args.reference_tracks else "not_supplied",
        "reference_tracks": str(args.reference_tracks.expanduser().resolve())
        if args.reference_tracks
        else None,
        "v2_integrity": v2_integrity,
        "across_within_v2_equivalence": window_equivalence,
        "environment": _environment_manifest(),
        "metrics": metrics,
        "elapsed_seconds": time.time() - start,
    }
    write_json(out_dir / "run_manifest.json", run_manifest)
    _write_report(out_dir / "REPORT.md", metrics, profile_hash, official, dataset_label)
    index = build_artifact_index(
        out_dir,
        exclude={"artifact_index.csv", "RUN_COMPLETE.json", ".DS_Store"},
    )
    write_rows_csv(out_dir / "artifact_index.csv", index)
    print(json.dumps(json_ready({"out_dir": out_dir, "metrics": metrics}), indent=2))


if __name__ == "__main__":
    main()
