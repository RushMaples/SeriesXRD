#!/usr/bin/env python3
"""Run the frozen, data-agnostic ``uniform-correlation-v2`` XY workflow.

The current UOTe handoff layout is supported through a thin input adapter, but
neither peak detection nor correlation consumes the teammate track registry.
An optional registry is used only after analysis to annotate nearest radial
trajectories and can never change a detection, assignment, score, or mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

import uniform_peak_core as up
import uniform_window_core as uw
from uniform_profile_binding import bind_frozen_profile
from uniform_correlation_io import (
    build_artifact_index,
    directory_sha256,
    file_sha256,
    json_ready,
    write_json,
    write_rows_csv,
)
from uniform_result_writer import (
    write_across_results,
    write_per_peak_results,
    write_within_results,
)
from uniform_xy_input import (
    FrameInput,
    read_handoff_manifest,
    read_xy_clean,
    resolve_channel_paths,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = SCRIPT_DIR / "configs" / "uniform-correlation-v2.json"
PROFILE_NAME = "uniform-correlation-v2"


def _progress(message: str) -> None:
    print(f"[uniform-v2] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wavelength-A", type=float, required=True)
    parser.add_argument("--channels", default="spots,fit")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--reference-tracks", type=Path, default=None)
    parser.add_argument("--legacy-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-plots", action="store_true", help="Operational reproducibility run only.")
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="EXPERIMENTAL smoke-test subset; official runs must use every scan.",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Required for a subset run. Scientific thresholds remain frozen.",
    )
    return parser.parse_args()


def load_frozen_profile(path: Path) -> tuple[dict[str, Any], str, bytes]:
    raw = path.expanduser().resolve().read_bytes()
    profile = json.loads(raw)
    if profile.get("profile") != PROFILE_NAME:
        raise ValueError(f"expected profile {PROFILE_NAME!r}, got {profile.get('profile')!r}")
    if profile.get("status") != "OFFICIAL_FROZEN":
        raise ValueError("official runner requires a frozen profile")
    return profile, hashlib.sha256(raw).hexdigest(), raw


def _clip_xy_to_interval(
    x: np.ndarray,
    y: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip to the shared interval, interpolating endpoints without extrapolation."""

    inside = (x >= lower) & (x <= upper)
    clipped_x = np.asarray(x[inside], dtype=float)
    clipped_y = np.asarray(y[inside], dtype=float)
    if x[0] <= lower <= x[-1] and not np.any(np.isclose(clipped_x, lower, rtol=0.0, atol=1e-12)):
        clipped_x = np.insert(clipped_x, 0, lower)
        clipped_y = np.insert(clipped_y, 0, np.interp(lower, x, y))
    if x[0] <= upper <= x[-1] and not np.any(np.isclose(clipped_x, upper, rtol=0.0, atol=1e-12)):
        clipped_x = np.append(clipped_x, upper)
        clipped_y = np.append(clipped_y, np.interp(upper, x, y))
    return clipped_x, clipped_y


def _process_pattern(
    payload: tuple[
        str,
        FrameInput,
        str,
        up.UniformPeakConfig,
        int,
        float,
        float,
    ]
) -> dict[str, Any]:
    path_text, frame, channel, config, minimum_points, lower, upper = payload
    path = Path(path_text)
    raw_x, raw_y, metadata = read_xy_clean(path, minimum_points=minimum_points)
    raw_cleaned = up.clean_xy(raw_x, raw_y)
    original_points = raw_cleaned.original_count
    x, y = _clip_xy_to_interval(
        raw_cleaned.x,
        raw_cleaned.y,
        lower,
        upper,
    )
    if len(x) < minimum_points:
        raise ValueError(
            f"{path} has only {len(x)} points inside shared analysis interval "
            f"[{lower:.8g}, {upper:.8g}]"
        )
    preprocessed = up.preprocess_pattern(x, y, config)
    peaks = up.detect_pattern_peaks(
        preprocessed,
        frame=frame.frame,
        scan=frame.scan,
        pressure=frame.pressure,
        channel=channel,
        config=config,
    )
    header_wavelength = math.nan
    try:
        header_wavelength = float(metadata.get("wavelength_A", "nan"))
    except ValueError:
        pass
    return {
        "x": preprocessed.x,
        "residual": preprocessed.residual,
        "frame_peaks": peaks,
        "header_wavelength_A": header_wavelength,
        "points": len(preprocessed.x),
        "original_points": original_points,
        "dx_deg": preprocessed.dx,
        "noise": preprocessed.noise,
        "total_positive_area": preprocessed.total_positive_area,
        "finite_removed": raw_cleaned.finite_removed,
        "duplicate_points_merged": raw_cleaned.duplicate_points_merged,
        "originally_strictly_increasing": raw_cleaned.originally_strictly_increasing,
        "warnings": ";".join(raw_cleaned.warnings + preprocessed.warnings),
    }


def _process_channel_patterns(
    paths: list[Path],
    frames: list[FrameInput],
    channel: str,
    config: up.UniformPeakConfig,
    *,
    minimum_points: int,
    coverage_fraction: float,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], uw.CoverageInterval]:
    # This preflight is intentionally before any baseline/peak detection so
    # per-peak and window analyses share exactly the same angle interval.
    preflight_x = []
    for path in paths:
        raw_x, raw_y, _ = read_xy_clean(path, minimum_points=minimum_points)
        preflight_x.append(up.clean_xy(raw_x, raw_y).x)
    interval = uw.common_coverage_interval(preflight_x, coverage_fraction)
    payloads = [
        (
            str(path),
            frame,
            channel,
            config,
            minimum_points,
            interval.lower_deg,
            interval.upper_deg,
        )
        for path, frame in zip(paths, frames, strict=True)
    ]
    if workers <= 1:
        processed = [_process_pattern(payload) for payload in payloads]
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                processed = list(executor.map(_process_pattern, payloads, chunksize=1))
        except (OSError, PermissionError):
            # Some managed macOS sandboxes prohibit the semaphore limit query
            # used by ProcessPoolExecutor.  Threads preserve identical numeric
            # semantics and SciPy releases the GIL in the expensive solvers.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                processed = list(executor.map(_process_pattern, payloads))
    qc_rows: list[dict[str, Any]] = []
    for path, frame, result in zip(paths, frames, processed, strict=True):
        header_wavelength = result["header_wavelength_A"]
        if np.isfinite(header_wavelength) and not np.isclose(
            header_wavelength, config.wavelength, rtol=0.0, atol=5e-7
        ):
            raise ValueError(
                f"wavelength mismatch in {path}: header={header_wavelength}, CLI={config.wavelength}"
            )
        qc_rows.append(
            {
                "channel": channel,
                "frame": frame.frame,
                "scan": frame.scan,
                "pressure_GPa": frame.pressure,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "points": result["points"],
                "original_points": result["original_points"],
                "analysis_interval_min_deg": interval.lower_deg,
                "analysis_interval_max_deg": interval.upper_deg,
                "dx_deg": result["dx_deg"],
                "noise": result["noise"],
                "total_positive_area": result["total_positive_area"],
                "finite_removed": result["finite_removed"],
                "duplicate_points_merged": result["duplicate_points_merged"],
                "originally_strictly_increasing": int(result["originally_strictly_increasing"]),
                "header_wavelength_A": header_wavelength,
                "warnings": result["warnings"],
            }
        )
    return processed, qc_rows, interval


def _posthoc_reference_matches(
    analysis: up.PerPeakAnalysis,
    reference_csv: Path,
    wavelength: float,
    channel: str,
) -> list[dict[str, Any]]:
    with reference_csv.open(newline="", encoding="utf-8-sig") as handle:
        references = [row for row in csv.DictReader(handle) if row.get("track", "").strip()]
    rows: list[dict[str, Any]] = []
    for track in analysis.tracks:
        if not track.official:
            continue
        candidates: list[tuple[float, dict[str, str], int, float]] = []
        for reference in references:
            p_min = float(reference["p_min_gpa"])
            p_max = float(reference["p_max_gpa"])
            d_at_pmin = float(reference["d0_A"])
            slope = float(reference["dd_dp_A_per_gpa"])
            distances = []
            for node in track.nodes:
                if p_min <= node.pressure <= p_max:
                    expected_d = d_at_pmin + slope * (node.pressure - p_min)
                    if expected_d > 0:
                        expected_q = 2.0 * math.pi / expected_d
                        distances.append(abs(node.q - expected_q))
            if distances:
                candidates.append((float(np.median(distances)), reference, len(distances), float(np.max(distances))))
        if not candidates:
            rows.append(
                {
                    "channel": channel,
                    "radial_track": track.track_id,
                    "status": "no_pressure_overlap",
                    "note": "Post-hoc only; reference did not influence detection or correlation.",
                }
            )
            continue
        median_delta, reference, overlap_nodes, max_delta = min(candidates, key=lambda item: item[0])
        median_width = float(np.median([node.fwhm_q for node in track.nodes]))
        close = bool(np.isfinite(median_width) and median_delta <= median_width)
        rows.append(
            {
                "channel": channel,
                "radial_track": track.track_id,
                "status": "nearest_within_one_median_FWHM" if close else "nearest_only_not_a_match",
                "reference_track": int(reference["track"]),
                "reference_hkl": reference.get("match_hkl", ""),
                "overlap_nodes": overlap_nodes,
                "median_abs_delta_q_A^-1": median_delta,
                "max_abs_delta_q_A^-1": max_delta,
                "median_track_fwhm_q_A^-1": median_width,
                "wavelength_A": wavelength,
                "note": "Post-hoc only; reference did not influence detection, assignment, support, or correlation.",
            }
        )
    return rows


def _scan_level_window_rows(across: uw.AcrossFrameCorrelations, channel: str) -> list[dict[str, Any]]:
    pressures = across.pressure_values
    gap = np.abs(pressures[:, None] - pressures[None, :])
    lower = np.tril_indices(len(pressures), k=-1)
    rows: list[dict[str, Any]] = []
    for family, values in (
        ("acf_strict", across.acf_strict_by_scan),
        ("direct_strict", across.direct_strict_by_scan),
        ("shift_tolerant_secondary", across.shift_tolerant_by_scan),
    ):
        for scan_index, scan in enumerate(across.scan_labels):
            score = np.nanmedian(values[scan_index], axis=0)
            x = gap[lower]
            y = score[lower]
            keep = np.isfinite(x) & np.isfinite(y)
            corr = float(np.corrcoef(x[keep], y[keep])[0, 1]) if np.count_nonzero(keep) >= 3 else math.nan
            rows.append(
                {
                    "channel": channel,
                    "family": family,
                    "scan": scan,
                    "pressure_pairs": int(np.count_nonzero(keep)),
                    "median_similarity": float(np.nanmedian(y[keep])) if np.any(keep) else math.nan,
                    "pearson_r_similarity_vs_pressure_gap": corr,
                }
            )
    return rows


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
        SCRIPT_DIR / "uniform_window_core.py",
        SCRIPT_DIR / "uniform_profile_binding.py",
        SCRIPT_DIR / "uniform_result_writer.py",
        SCRIPT_DIR / "uniform_xy_input.py",
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


def _write_report(path: Path, metrics: dict[str, Any], profile_hash: str, official: bool) -> None:
    spots = metrics.get("spots", {})
    fit = metrics.get("fit", {})
    spots_peak = spots.get("per_peak", {})
    spots_across = spots.get("across_frames", {}).get("families", {})
    spots_within = spots.get("within_frame", {})
    fit_peak = fit.get("per_peak", {})
    fit_across = fit.get("across_frames", {}).get("families", {})
    status = "OFFICIAL_FROZEN" if official else "EXPERIMENTAL"
    text = f"""# UOTe uniform-correlation-v2 report

## Run identity

- Status: **{status}**
- Frozen profile SHA256: `{profile_hash}`
- Spots is the primary sample channel; fit is a tungsten-dominated control.
- No teammate p_min/p_max/d0/slope/search band was used for detection or scoring.

## Beginner summary

This run separates three ideas that the legacy heatmaps mixed together: whether a radial peak was detected, how much paired-scan support exists, and—only when both sides contain the same reliable peak—how similar its area or location is. Gray/hatched cells mean “not enough evidence”; they are not zero and they are not negative correlation.

- Spots official blind radial tracks: {spots_peak.get('official_radial_tracks', 0)}.
- Fit-control official blind radial tracks: {fit_peak.get('official_radial_tracks', 0)}.
- Spots median per-track area/location near-vs-far AUC: {spots_peak.get('median_area_auc', math.nan):.3f} / {spots_peak.get('median_location_auc', math.nan):.3f}.
- Spots strict-ACF median window AUC: {spots_across.get('acf_strict', {}).get('median_window_auc', math.nan):.3f}.
- Spots direct-strict median window AUC: {spots_across.get('direct_strict', {}).get('median_window_auc', math.nan):.3f}.
- Fit-control strict-ACF median window AUC: {fit_across.get('acf_strict', {}).get('median_window_auc', math.nan):.3f}.
- Spots within-frame non-overlap median: {spots_within.get('nonoverlap_pair_median', math.nan):.3f}.

An AUC near 0.5 means near-pressure pairs are not reliably more similar than far-pressure pairs. A high score in both spots and fit can be shared instrument/pressure-marker behavior rather than UOTe-specific evidence. Weak or unavailable results are reported as such and do not cause threshold changes.

## Method summary

1. Blind AsLS + robust-noise peak detection and pseudo-Voigt fitting.
2. Per-pressure scan consensus and bidirectional one-to-one radial trajectory tracking in q.
3. Conditional area/location maps, separate presence Jaccard, and explicit support counts.
4. Strict same-window ACF and direct residual Pearson across pressures within each scan.
5. Window-to-window ACF within frames, with the non-overlap subset as the primary control.
6. Q25/Q75 pressure-gap near/far definitions and 2,000 scan-level bootstrap resamples.

## Limitations

- One-dimensional XY recovers radial peaks only; azimuth-specific spot identities cannot be reconstructed.
- Overlapping or crossing peaks that cannot be resolved are marked unknown/split rather than silently swapped.
- Relative area is normalized to each frame's total positive residual and is not an absolute-intensity measurement.
- A mathematically valid correlation is not by itself proof of a phase transition.

See `algorithm_config.json`, `run_manifest.json`, `validation/`, and the per-map support files for the complete audit trail.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    handoff = args.handoff_dir.expanduser().resolve()
    manifest = (args.manifest or handoff / "manifest.csv").expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {out_dir}")
    if args.max_scans is not None and not args.experimental:
        raise SystemExit("--max-scans requires --experimental; official runs use every scan")
    profile, profile_hash, profile_raw = load_frozen_profile(args.profile)
    bound_profile = bind_frozen_profile(profile, float(args.wavelength_A))
    config = bound_profile.peak_config
    window_config = bound_profile.window_config
    out_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("validation", "robustness", "comparison_to_legacy"):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)

    official = not args.experimental and args.max_scans is None
    (out_dir / "algorithm_config.json").write_bytes(profile_raw)
    channels = [item.strip().lower() for item in args.channels.split(",") if item.strip()]
    if not channels or any(item not in {"spots", "fit"} for item in channels):
        raise SystemExit("--channels must contain spots and/or fit")
    frames, pressures, scans, selected_manifest = read_handoff_manifest(manifest, args.max_scans)
    metrics: dict[str, Any] = {}
    inventory_rows: list[dict[str, Any]] = []
    scan_metric_rows: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] = []

    legacy_before: dict[str, Any] | None = None
    if args.legacy_root is not None:
        legacy_root = args.legacy_root.expanduser().resolve()
        count, digest = directory_sha256(legacy_root)
        legacy_before = {"path": str(legacy_root), "files": count, "sha256_before": digest}

    for channel in channels:
        channel_start = time.time()
        paths = resolve_channel_paths(handoff, frames, channel)
        _progress(
            f"{channel}: peak extraction start ({len(paths)} frames; shared coverage preflight)"
        )
        processed, channel_inventory, coverage_interval = _process_channel_patterns(
            paths,
            frames,
            channel,
            config,
            minimum_points=bound_profile.minimum_points_per_pattern,
            coverage_fraction=window_config.coverage_fraction,
            workers=max(1, int(args.workers)),
        )
        _progress(f"{channel}: peak extraction complete")
        inventory_rows.extend(channel_inventory)
        frame_peaks = [item["frame_peaks"] for item in processed]
        _progress(f"{channel}: per-peak analysis/write start")
        peak_analysis = up.analyze_per_peak(
            frame_peaks,
            pressures,
            scans,
            config,
            bootstrap_iterations=config.bootstrap_iterations,
            seed=config.random_seed,
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
        )
        _progress(f"{channel}: per-peak analysis/write complete")
        if args.reference_tracks is not None:
            posthoc_rows.extend(
                _posthoc_reference_matches(
                    peak_analysis,
                    args.reference_tracks.expanduser().resolve(),
                    config.wavelength,
                    channel,
                )
            )

        batch = uw.resample_common_grid(
            [item["x"] for item in processed],
            [item["residual"] for item in processed],
            coverage_fraction=window_config.coverage_fraction,
            coverage_interval=coverage_interval,
        )
        features = uw.build_window_features(
            batch.grid_deg,
            batch.values,
            config=window_config,
        )
        frame_scans = [frame.scan for frame in frames]
        frame_pressures = [frame.pressure for frame in frames]
        _progress(f"{channel}: across-frame analysis/write start")
        across = uw.compute_across_frame_correlations(
            features,
            frame_scans,
            frame_pressures,
            config=window_config,
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
        _progress(f"{channel}: across-frame analysis/write complete")
        _progress(f"{channel}: within-frame analysis/write start")
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
        _progress(f"{channel}: within-frame analysis/write complete")
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

    write_rows_csv(out_dir / "input_inventory.csv", inventory_rows)
    write_rows_csv(out_dir / "robustness" / "scan_level_metrics.csv", scan_metric_rows)
    if posthoc_rows:
        write_rows_csv(out_dir / "comparison_to_legacy" / "posthoc_reference_track_matches.csv", posthoc_rows)

    if legacy_before is not None:
        count, digest = directory_sha256(Path(legacy_before["path"]))
        legacy_before.update(
            {
                "files_after": count,
                "sha256_after": digest,
                "unchanged": digest == legacy_before["sha256_before"] and count == legacy_before["files"],
            }
        )
        write_json(out_dir / "comparison_to_legacy" / "legacy_integrity.json", legacy_before)

    run_manifest = {
        "profile": PROFILE_NAME,
        "profile_status": "OFFICIAL_FROZEN" if official else "EXPERIMENTAL",
        "profile_sha256": profile_hash,
        "execution_semantics_sha256": bound_profile.semantic_sha256,
        "resolved_algorithm_semantics": bound_profile.resolved_semantics,
        "config_binding_audit": bound_profile.binding_audit,
        "config_binding_checks": {
            "all_peak_config_fields_explicitly_bound": True,
            "all_window_config_fields_explicitly_bound": True,
            "peak_config_field_count": len(bound_profile.binding_audit["peak_config"]),
            "window_config_field_count": len(bound_profile.binding_audit["window_config"]),
        },
        "wavelength_A": float(args.wavelength_A),
        "handoff_dir": str(handoff),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "channels": channels,
        "frames": len(frames),
        "scans": scans,
        "pressure_gpa": pressures,
        "excluded_manifest_rows": sum(
            str(row.get("cover_excluded", "")).strip() != "0" for row in selected_manifest
        ),
        "workers": max(1, int(args.workers)),
        "plots_written": not args.no_plots,
        "reference_tracks_role": "posthoc_annotation_only" if args.reference_tracks else "not_supplied",
        "reference_tracks": str(args.reference_tracks.expanduser().resolve()) if args.reference_tracks else None,
        "legacy_integrity": legacy_before,
        "environment": _environment_manifest(),
        "metrics": metrics,
        "elapsed_seconds": time.time() - start,
    }
    write_json(out_dir / "run_manifest.json", run_manifest)
    _write_report(out_dir / "REPORT.md", metrics, profile_hash, official)
    index = build_artifact_index(
        out_dir,
        exclude={
            "artifact_index.csv",
            "RUN_COMPLETE.json",
            "UOTe_Handoff2_Correlation_Report_uniform_v2_20260714.xlsx",
        },
    )
    write_rows_csv(out_dir / "artifact_index.csv", index)
    print(json.dumps(json_ready({"out_dir": out_dir, "metrics": metrics}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
