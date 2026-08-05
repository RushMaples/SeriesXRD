#!/usr/bin/env python3
"""Recompute XRD window correlations on absolute 5-degree / 1-degree windows.

The nominal geometry is 0-5, 1-6, 2-7, and so on.  The source data start a
small positive distance above zero, so only the first nominal window uses the
observed portion of its interval; no value is extrapolated below the detector
coverage.  Nominal and effective bounds are both exported.
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
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nonlinear_intensity_preprocessing import (
    DEFAULT_EPSILON_FLOOR,
    DEFAULT_SCALE_QUANTILE,
    LOG_SQUARED,
    epsilon_from_noise_floor,
    transform_bounded_squared,
)
import uniform_peak_core as up
import uniform_window_core as uw
from run_uniform_xy_correlations import _clip_xy_to_interval
from uniform_correlation_io import file_sha256, json_ready, write_json, write_rows_csv
from uniform_profile_binding import bind_frozen_profile
from uniform_profile_binding_v21 import bind_frozen_profile_v21
from uniform_result_writer import write_across_results, write_within_results
from uniform_xy_input import read_xy_clean
from uniform_xy_input_v21 import InputDataset, read_input_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_SINGLE_ROOT = WORKSPACE_ROOT / "correlations" / "UOTe Single Crystal Reduced"
DEFAULT_SINGLE_MANIFEST = (
    WORKSPACE_ROOT
    / "correlations"
    / "manifests"
    / "uote_single_crystal_uniform_v2_1_manifest.csv"
)
DEFAULT_SINGLE_PROFILE = SCRIPT_DIR / "configs" / "uniform-correlation-v2.1.json"
DEFAULT_POWDER_ROOT = WORKSPACE_ROOT / "correlations" / "uote_xy_handoff 2"
DEFAULT_POWDER_MANIFEST = DEFAULT_POWDER_ROOT / "manifest.csv"
DEFAULT_POWDER_PROFILE = SCRIPT_DIR / "configs" / "uniform-correlation-v2.json"

WINDOW_WIDTH_DEG = 5.0
WINDOW_STEP_DEG = 1.0
WINDOW_START_DEG = 0.0
EDGE_POLICY = "observed_support_no_extrapolation_resampled_on_normalized_coordinate"
INTENSITY_TRANSFORM_MODES = ("none", LOG_SQUARED)
DEFAULT_TRANSFORM_SCALE_QUANTILE = DEFAULT_SCALE_QUANTILE
DEFAULT_TRANSFORM_EPSILON_FLOOR = DEFAULT_EPSILON_FLOOR


@dataclass(frozen=True)
class IntensityTransformConfig:
    """Frozen role-wide preprocessing settings for window residuals."""

    mode: str = "none"
    scale_quantile: float = DEFAULT_TRANSFORM_SCALE_QUANTILE
    epsilon_floor: float = DEFAULT_TRANSFORM_EPSILON_FLOOR

    def __post_init__(self) -> None:
        if self.mode not in INTENSITY_TRANSFORM_MODES:
            raise ValueError(
                f"unsupported intensity transform {self.mode!r}; "
                f"expected one of {INTENSITY_TRANSFORM_MODES}"
            )
        if not (0.0 < self.scale_quantile <= 1.0):
            raise ValueError("transform scale quantile must be in (0, 1]")
        if not (
            np.isfinite(self.epsilon_floor)
            and self.epsilon_floor > 0.0
        ):
            raise ValueError("transform epsilon floor must be finite and positive")


def _transform_preprocessed_residuals(
    processed: Sequence[Mapping[str, Any]],
    config: IntensityTransformConfig,
) -> tuple[list[np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """Transform one complete role with a single pooled residual scale."""

    residuals = [np.asarray(item["residual"], dtype=float) for item in processed]
    if not residuals or any(
        residual.ndim != 1
        or residual.size < 2
        or np.any(~np.isfinite(residual))
        for residual in residuals
    ):
        raise ValueError("role transform requires finite 1D residual arrays")
    if config.mode == "none":
        frame_audits = [
            {
                "transform_mode": "none",
                "role_scale_a": None,
                "role_noise_sigma": None,
                "derived_epsilon": None,
                "clipped_fraction": 0.0,
                "transformed_min": float(np.min(residual)),
                "transformed_max": float(np.max(residual)),
                "zero_maps_to_zero": True,
            }
            for residual in residuals
        ]
        return residuals, {
            "mode": "none",
            "position_in_pipeline": "no nonlinear transform",
            "role_scale_a": None,
            "role_noise_sigma": None,
            "derived_epsilon": None,
            "clipped_fraction": 0.0,
            "output_min": float(min(np.min(item) for item in residuals)),
            "output_max": float(max(np.max(item) for item in residuals)),
            "default_path_numeric_identity": True,
        }, frame_audits

    pooled_absolute = np.concatenate([np.abs(residual) for residual in residuals])
    scale = float(np.quantile(pooled_absolute, config.scale_quantile))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("pooled absolute-residual quantile must be positive")
    noises = np.asarray([float(item["noise"]) for item in processed], dtype=float)
    finite_noises = noises[np.isfinite(noises) & (noises >= 0.0)]
    if finite_noises.size != len(processed):
        raise ValueError("every preprocessed frame must have finite nonnegative noise")
    sigma = float(np.median(finite_noises))
    epsilon = epsilon_from_noise_floor(
        sigma,
        scale,
        epsilon_floor=config.epsilon_floor,
    )

    transformed_residuals: list[np.ndarray] = []
    frame_audits: list[dict[str, Any]] = []
    clipped_slots = 0
    total_slots = 0
    exact_zero_slots = 0
    exact_zero_outputs = 0
    for residual in residuals:
        clipped = np.abs(residual) > scale
        z = np.clip(residual / scale, -1.0, 1.0)
        transformed = np.asarray(
            transform_bounded_squared(
                z,
                method=config.mode,
                epsilon=epsilon,
            ),
            dtype=float,
        )
        if transformed.shape != residual.shape or np.any(~np.isfinite(transformed)):
            raise ValueError(f"{config.mode} produced an invalid residual array")
        zero_mask = residual == 0.0
        clipped_count = int(np.count_nonzero(clipped))
        zero_count = int(np.count_nonzero(zero_mask))
        zero_output_count = int(np.count_nonzero(transformed[zero_mask] == 0.0))
        clipped_slots += clipped_count
        total_slots += int(residual.size)
        exact_zero_slots += zero_count
        exact_zero_outputs += zero_output_count
        transformed_residuals.append(transformed)
        frame_audits.append(
            {
                "transform_mode": config.mode,
                "role_scale_a": scale,
                "role_noise_sigma": sigma,
                "derived_epsilon": epsilon,
                "clipped_fraction": clipped_count / residual.size,
                "transformed_min": float(np.min(transformed)),
                "transformed_max": float(np.max(transformed)),
                "zero_maps_to_zero": zero_output_count == zero_count,
            }
        )
    all_values = np.concatenate(transformed_residuals)
    role_audit = {
        "mode": config.mode,
        "position_in_pipeline": (
            "after unchanged clean_xy/AsLS preprocessing and before common-grid "
            "resampling, window extraction, standardization, ACF, and Pearson"
        ),
        "role_scale_a": scale,
        "scale_quantile": config.scale_quantile,
        "role_noise_sigma": sigma,
        "derived_epsilon": epsilon,
        "epsilon_floor": config.epsilon_floor,
        "normalized_residual": "z=clip(residual/a,-1,1)",
        "sign_policy": "squaring deliberately erases residual sign",
        "clipped_slots": clipped_slots,
        "total_slots": total_slots,
        "clipped_fraction": clipped_slots / total_slots,
        "exact_zero_slots": exact_zero_slots,
        "exact_zero_outputs": exact_zero_outputs,
        "zero_maps_to_zero": exact_zero_outputs == exact_zero_slots,
        "output_min": float(np.min(all_values)),
        "output_max": float(np.max(all_values)),
        "output_bounded_0_1": bool(
            np.min(all_values) >= 0.0 and np.max(all_values) <= 1.0
        ),
        "pooled_scale_estimate": {
            "definition": "quantile(abs(signed preprocessed residual))",
            "scale": scale,
            "quantile": config.scale_quantile,
            "frame_arrays": len(residuals),
            "finite_slots": int(pooled_absolute.size),
            "zero_slots": int(np.count_nonzero(pooled_absolute == 0.0)),
        },
    }
    return transformed_residuals, role_audit, frame_audits


def _progress(message: str) -> None:
    print(f"[integer-windows] {message}", flush=True)


def _load_profile(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _clean_x(path: Path, minimum_points: int) -> np.ndarray:
    raw_x, raw_y, _metadata = read_xy_clean(path, minimum_points=minimum_points)
    return up.clean_xy(raw_x, raw_y).x


def _coverage_interval(
    paths: Sequence[Path],
    *,
    minimum_points: int,
    coverage_fraction: float,
) -> uw.CoverageInterval:
    coordinates = [_clean_x(path, minimum_points) for path in paths]
    return uw.common_coverage_interval(coordinates, coverage_fraction)


def _preprocess_window_profile(
    payload: tuple[str, up.UniformPeakConfig, int, float, float],
) -> dict[str, Any]:
    path_text, config, minimum_points, lower, upper = payload
    path = Path(path_text)
    raw_x, raw_y, metadata = read_xy_clean(path, minimum_points=minimum_points)
    header_wavelength = math.nan
    try:
        header_wavelength = float(metadata.get("wavelength_A", "nan"))
    except (TypeError, ValueError):
        pass
    if np.isfinite(header_wavelength) and not np.isclose(
        header_wavelength,
        config.wavelength,
        rtol=0.0,
        atol=5.0e-7,
    ):
        raise ValueError(
            f"wavelength mismatch in {path}: header={header_wavelength}, "
            f"profile={config.wavelength}"
        )
    cleaned = up.clean_xy(raw_x, raw_y)
    x, y = _clip_xy_to_interval(cleaned.x, cleaned.y, lower, upper)
    if len(x) < minimum_points:
        raise ValueError(
            f"{path} has only {len(x)} points inside [{lower:g}, {upper:g}]"
        )
    preprocessed = up.preprocess_pattern(x, y, config)
    return {
        "x": preprocessed.x,
        "residual": preprocessed.residual,
        "points": len(preprocessed.x),
        "grid_step_deg": preprocessed.dx,
        "noise": preprocessed.noise,
        "header_wavelength_A": header_wavelength,
    }


def _preprocess_profiles(
    paths: Sequence[Path],
    *,
    config: up.UniformPeakConfig,
    minimum_points: int,
    interval: uw.CoverageInterval,
    workers: int,
) -> list[dict[str, Any]]:
    payloads = [
        (
            str(path),
            config,
            minimum_points,
            interval.lower_deg,
            interval.upper_deg,
        )
        for path in paths
    ]
    if workers <= 1:
        return [_preprocess_window_profile(payload) for payload in payloads]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(_preprocess_window_profile, payloads, chunksize=4)
            )
    except (OSError, PermissionError):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_preprocess_window_profile, payloads))


def _all_symmetric(values: np.ndarray, tolerance: float = 1.0e-10) -> bool:
    array = np.asarray(values, dtype=float)
    if array.shape[-1] != array.shape[-2]:
        return False
    finite = np.isfinite(array) & np.isfinite(np.swapaxes(array, -1, -2))
    masks_match = np.array_equal(
        np.isfinite(array),
        np.isfinite(np.swapaxes(array, -1, -2)),
    )
    if not masks_match:
        return False
    if not np.any(finite):
        return True
    difference = np.abs(array - np.swapaxes(array, -1, -2))
    return bool(np.nanmax(difference) <= tolerance)


def _finite_in_unit_pearson_range(*arrays: np.ndarray) -> bool:
    for source in arrays:
        values = np.asarray(source, dtype=float)
        if np.any(np.isinf(values)):
            return False
        finite = values[np.isfinite(values)]
        if finite.size and (np.min(finite) < -1.0 or np.max(finite) > 1.0):
            return False
    return True


def _window_definition_rows(
    spec: uw.WindowSpec,
    grid: np.ndarray,
    sample_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(
        zip(spec.starts_deg, spec.ends_deg, strict=True)
    ):
        effective_start = max(float(start), float(grid[0]))
        effective_end = min(float(end), float(grid[-1]))
        rows.append(
            {
                "window_index": index,
                "window_label": f"{start:g}-{end:g}",
                "nominal_start_deg": start,
                "nominal_end_deg": end,
                "nominal_width_deg": end - start,
                "effective_observed_start_deg": effective_start,
                "effective_observed_end_deg": effective_end,
                "effective_observed_width_deg": effective_end - effective_start,
                "lower_edge_unobserved_deg": max(0.0, float(grid[0] - start)),
                "upper_edge_unobserved_deg": max(0.0, float(end - grid[-1])),
                "edge_clipped": int(
                    effective_start > float(start) or effective_end < float(end)
                ),
                "extrapolated": 0,
                "resampled_points": sample_count,
                "edge_policy": EDGE_POLICY,
                "is_nonoverlap_control": int(
                    index in set(spec.nonoverlap_indices.tolist())
                ),
            }
        )
    return rows


def _validate_geometry(spec: uw.WindowSpec) -> bool:
    starts = np.asarray(spec.starts_deg, dtype=float)
    ends = np.asarray(spec.ends_deg, dtype=float)
    return bool(
        starts.size >= 1
        and np.array_equal(starts, np.arange(starts.size, dtype=float))
        and np.array_equal(ends, starts + WINDOW_WIDTH_DEG)
        and np.all(np.diff(starts) == WINDOW_STEP_DEG)
        and np.all(starts == np.floor(starts))
        and np.all(ends == np.floor(ends))
    )


def _run_role(
    *,
    role: str,
    channel_label: str,
    destination: Path,
    dataset: InputDataset,
    channel: str,
    peak_config: up.UniformPeakConfig,
    window_config: uw.UniformWindowConfig,
    minimum_points: int,
    workers: int,
    make_full_symmetric_plots: bool,
    expected_scope: Mapping[str, Any] | None,
    transform_config: IntensityTransformConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = list(dataset.paths_by_channel[channel])
    frames = list(dataset.frames)
    _progress(f"{role}: common-coverage preflight for {len(paths)} frames")
    interval = _coverage_interval(
        paths,
        minimum_points=minimum_points,
        coverage_fraction=window_config.coverage_fraction,
    )
    _progress(f"{role}: baseline preprocessing (no peak fitting)")
    processed = _preprocess_profiles(
        paths,
        config=peak_config,
        minimum_points=minimum_points,
        interval=interval,
        workers=workers,
    )
    transformed_residuals, transform_audit, transform_frame_audits = (
        _transform_preprocessed_residuals(processed, transform_config)
    )
    batch = uw.resample_common_grid(
        [item["x"] for item in processed],
        transformed_residuals,
        coverage_fraction=window_config.coverage_fraction,
        coverage_interval=interval,
    )
    spec = uw.make_fixed_sliding_window_spec(
        float(batch.grid_deg[-1]),
        width_deg=WINDOW_WIDTH_DEG,
        step_deg=WINDOW_STEP_DEG,
        start_deg=WINDOW_START_DEG,
    )
    features = uw.build_window_features(
        batch.grid_deg,
        batch.values,
        spec,
        config=window_config,
    )
    frame_scans = [frame.scan for frame in frames]
    frame_pressures = [frame.pressure for frame in frames]
    frame_ids = [frame.frame for frame in frames]

    _progress(
        f"{role}: across-frame correlations on {len(spec.starts_deg)} integer windows"
    )
    across = uw.compute_across_frame_correlations(
        features,
        frame_scans,
        frame_pressures,
        config=window_config,
    )
    across_metrics = write_across_results(
        destination,
        channel_label,
        across,
        n_bootstrap=window_config.bootstrap_iterations,
        seed=window_config.random_seed,
        confidence=window_config.confidence,
        minimum_distinct_gaps=window_config.minimum_distinct_pressure_gaps,
        minimum_group_values=window_config.minimum_supported_group_values,
        near_gap_quantile=window_config.near_gap_quantile,
        far_gap_quantile=window_config.far_gap_quantile,
        make_plots=make_full_symmetric_plots,
    )
    _progress(f"{role}: within-frame window-to-window correlations")
    within = uw.compute_within_frame_correlations(
        features.fingerprints,
        frame_scans,
        frame_pressures,
        nonoverlap_indices=spec.nonoverlap_indices,
        config=window_config,
    )
    within_metrics = write_within_results(
        destination,
        channel_label,
        within,
        spec,
        frame_ids=frame_ids,
        frame_scans=frame_scans,
        frame_pressures=frame_pressures,
        n_bootstrap=window_config.bootstrap_iterations,
        seed=window_config.random_seed,
        confidence=window_config.confidence,
        make_plots=make_full_symmetric_plots,
    )

    definition_rows = _window_definition_rows(
        spec,
        batch.grid_deg,
        features.signals.shape[-1],
    )
    write_rows_csv(destination / "window_definition.csv", definition_rows)
    np.savez_compressed(
        destination / "window_feature_validity.npz",
        frame_ids=np.asarray(frame_ids, dtype=int),
        frame_scans=np.asarray(frame_scans),
        frame_pressure_gpa=np.asarray(frame_pressures, dtype=float),
        nominal_window_starts_deg=spec.starts_deg,
        nominal_window_ends_deg=spec.ends_deg,
        effective_window_starts_deg=np.asarray(
            [row["effective_observed_start_deg"] for row in definition_rows],
            dtype=float,
        ),
        effective_window_ends_deg=np.asarray(
            [row["effective_observed_end_deg"] for row in definition_rows],
            dtype=float,
        ),
        signal_valid=features.signal_valid,
        fingerprint_valid=features.fingerprint_valid,
    )

    geometry_verified = _validate_geometry(spec)
    expected_windows = len(spec.starts_deg)
    exact_shape_verified = bool(
        across.acf_strict_by_scan.shape
        == (
            len(dataset.scans),
            expected_windows,
            len(dataset.pressures),
            len(dataset.pressures),
        )
        and within.by_frame.shape
        == (len(frames), expected_windows, expected_windows)
    )
    actual_scope = {
        "frames": len(frames),
        "scans": len(dataset.scans),
        "pressures": len(dataset.pressures),
        "nominal_windows": len(spec.starts_deg),
        "pressure_min_GPa": float(min(dataset.pressures)),
        "pressure_max_GPa": float(max(dataset.pressures)),
    }
    intended_scope_verified = bool(
        expected_scope is None
        or all(actual_scope.get(key) == value for key, value in expected_scope.items())
    )
    symmetric_verified = bool(
        _all_symmetric(across.acf_strict_by_scan)
        and _all_symmetric(across.direct_strict_by_scan)
        and _all_symmetric(across.shift_tolerant_by_scan)
        and _all_symmetric(within.by_frame)
    )
    score_range_verified = _finite_in_unit_pearson_range(
        across.acf_strict_by_scan,
        across.direct_strict_by_scan,
        across.shift_tolerant_by_scan,
        within.by_frame,
    )
    all_windows_valid = bool(
        np.all(features.signal_valid) and np.all(features.fingerprint_valid)
    )
    validation = {
        "role": role,
        "frames": len(frames),
        "scans": len(dataset.scans),
        "pressures": len(dataset.pressures),
        "nominal_windows": len(spec.starts_deg),
        "first_nominal_window": spec.labels[0],
        "last_nominal_window": spec.labels[-1],
        "window_width_deg": spec.width_deg,
        "window_step_deg": spec.step_deg,
        "common_grid_min_deg": float(batch.grid_deg[0]),
        "common_grid_max_deg": float(batch.grid_deg[-1]),
        "first_effective_window_start_deg": definition_rows[0][
            "effective_observed_start_deg"
        ],
        "first_window_lower_edge_unobserved_deg": definition_rows[0][
            "lower_edge_unobserved_deg"
        ],
        "edge_policy": EDGE_POLICY,
        "intensity_preprocessing": transform_audit,
        "no_extrapolation": True,
        "integer_geometry_verified": geometry_verified,
        "expected_shapes_verified": exact_shape_verified,
        "source_matrices_symmetric": symmetric_verified,
        "finite_scores_in_minus1_plus1": score_range_verified,
        "all_signal_and_acf_windows_valid": all_windows_valid,
        "actual_scope": actual_scope,
        "expected_scope": dict(expected_scope or {}),
        "intended_scope_verified": intended_scope_verified,
        "across_metrics": across_metrics,
        "within_metrics": within_metrics,
    }
    if not all(
        (
            geometry_verified,
            exact_shape_verified,
            symmetric_verified,
            score_range_verified,
            all_windows_valid,
            intended_scope_verified,
        )
    ):
        raise RuntimeError(f"integer window validation failed: {validation}")
    write_json(destination / "integer_window_validation.json", validation)

    inventory_rows: list[dict[str, Any]] = []
    for frame, path, result, transform_frame in zip(
        frames,
        paths,
        processed,
        transform_frame_audits,
        strict=True,
    ):
        inventory_rows.append(
            {
                "role": role,
                "frame": frame.frame,
                "scan": frame.scan,
                "pressure_GPa": frame.pressure,
                "source_path": str(path.resolve()),
                "source_sha256": file_sha256(path),
                "source_bytes": path.stat().st_size,
                "points_used": result["points"],
                "grid_step_deg": result["grid_step_deg"],
                "noise": result["noise"],
                "header_wavelength_A": result["header_wavelength_A"],
                "profile_wavelength_A": peak_config.wavelength,
                **transform_frame,
            }
        )
    geometry_rows = [dict(row, role=role) for row in definition_rows]
    return validation, inventory_rows, geometry_rows


def _payload_inventory(root: Path) -> tuple[int, int, dict[str, int]]:
    files = [path for path in root.rglob("*") if path.is_file()]
    suffixes: dict[str, int] = {}
    for path in files:
        suffix = path.suffix.lower()
        suffixes[suffix] = suffixes.get(suffix, 0) + 1
    return len(files), sum(path.stat().st_size for path in files), suffixes


def _statistics_config_audit(
    config: uw.UniformWindowConfig,
) -> dict[str, Any]:
    """Separate inherited statistical settings from replaced window geometry."""

    values = asdict(config)
    profile_geometry = {
        key: values.pop(key)
        for key in ("window_count", "width_divisor", "step_divisor")
    }
    return {
        "profile_geometry_not_executed": profile_geometry,
        "executed_geometry_override": {
            "start_deg": WINDOW_START_DEG,
            "width_deg": WINDOW_WIDTH_DEG,
            "step_deg": WINDOW_STEP_DEG,
            "count": "derived from observed upper bound",
        },
        "inherited_statistics_and_feature_settings": values,
    }


def generate_integer_window_sources(
    output_root: Path,
    *,
    workers: int = min(8, os.cpu_count() or 1),
    make_full_symmetric_plots: bool = False,
    max_scans: int | None = None,
    single_root: Path = DEFAULT_SINGLE_ROOT,
    single_manifest: Path = DEFAULT_SINGLE_MANIFEST,
    single_profile_path: Path = DEFAULT_SINGLE_PROFILE,
    single_wavelength_A: float = 0.4133,
    powder_root: Path = DEFAULT_POWDER_ROOT,
    powder_manifest: Path = DEFAULT_POWDER_MANIFEST,
    powder_profile_path: Path = DEFAULT_POWDER_PROFILE,
    powder_wavelength_A: float = 0.3066,
    intensity_transform: IntensityTransformConfig | None = None,
) -> dict[str, Any]:
    """Generate three complete full-symmetric sources under one suite root."""
    start_time = time.time()
    transform_config = intensity_transform or IntensityTransformConfig()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    single_root = single_root.expanduser().resolve()
    single_manifest = single_manifest.expanduser().resolve()
    single_profile_path = single_profile_path.expanduser().resolve()
    powder_root = powder_root.expanduser().resolve()
    powder_manifest = powder_manifest.expanduser().resolve()
    powder_profile_path = powder_profile_path.expanduser().resolve()
    final_audit_root = output_root / "window_full_symmetric_audit"
    final_provenance_root = output_root / "window_provenance"
    final_methods_path = output_root / "WINDOW_METHODS.md"
    existing = [
        path
        for path in (final_audit_root, final_provenance_root, final_methods_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "integer-window targets already exist: "
            + ", ".join(str(path) for path in existing)
        )

    staging_root = Path(
        tempfile.mkdtemp(prefix=".integer-window-build-", dir=output_root)
    )
    audit_root = staging_root / "window_full_symmetric_audit"
    provenance_root = staging_root / "window_provenance"
    methods_path = staging_root / "WINDOW_METHODS.md"
    audit_root.mkdir(parents=True, exist_ok=False)
    try:
        single_profile, single_profile_hash = _load_profile(single_profile_path)
        single_bound = bind_frozen_profile_v21(
            single_profile, float(single_wavelength_A)
        )
        single_dataset = read_input_dataset(
            single_root,
            single_manifest,
            ["spots"],
            input_mode="direct",
            max_scans=max_scans,
        )
        powder_profile, powder_profile_hash = _load_profile(powder_profile_path)
        powder_bound = bind_frozen_profile(
            powder_profile, float(powder_wavelength_A)
        )
        powder_dataset = read_input_dataset(
            powder_root,
            powder_manifest,
            ["spots", "fit"],
            input_mode="handoff",
            max_scans=max_scans,
        )

        production_single_scope = (
            {
                "frames": 22,
                "scans": 2,
                "pressures": 11,
                "nominal_windows": 19,
                "pressure_min_GPa": 1.0,
                "pressure_max_GPa": 12.8,
            }
            if max_scans is None
            else None
        )
        production_powder_scope = (
            {
                "frames": 1060,
                "scans": 56,
                "pressures": 19,
                "nominal_windows": 28,
                "pressure_min_GPa": 3.5,
                "pressure_max_GPa": 50.7,
            }
            if max_scans is None
            else None
        )
        role_specs = [
            {
                "role": "single_spots",
                "channel_label": "single spots",
                "destination": audit_root / "single_crystal" / "spots",
                "dataset": single_dataset,
                "channel": "spots",
                "peak_config": single_bound.peak_config,
                "window_config": single_bound.window_config,
                "minimum_points": single_bound.minimum_points_per_pattern,
                "expected_scope": production_single_scope,
                "transform_config": transform_config,
            },
            {
                "role": "powder_spots",
                "channel_label": "powder spots",
                "destination": audit_root / "powder" / "spots",
                "dataset": powder_dataset,
                "channel": "spots",
                "peak_config": powder_bound.peak_config,
                "window_config": powder_bound.window_config,
                "minimum_points": powder_bound.minimum_points_per_pattern,
                "expected_scope": production_powder_scope,
                "transform_config": transform_config,
            },
            {
                "role": "powder_fit_control",
                "channel_label": "powder tungsten-dominated fit control",
                "destination": audit_root / "powder" / "fit_control",
                "dataset": powder_dataset,
                "channel": "fit",
                "peak_config": powder_bound.peak_config,
                "window_config": powder_bound.window_config,
                "minimum_points": powder_bound.minimum_points_per_pattern,
                "expected_scope": production_powder_scope,
                "transform_config": transform_config,
            },
        ]
        validations: list[dict[str, Any]] = []
        inventory_rows: list[dict[str, Any]] = []
        geometry_rows: list[dict[str, Any]] = []
        for spec in role_specs:
            validation, role_inventory, role_geometry = _run_role(
                **spec,
                workers=max(1, int(workers)),
                make_full_symmetric_plots=make_full_symmetric_plots,
            )
            validations.append(validation)
            inventory_rows.extend(role_inventory)
            geometry_rows.extend(role_geometry)

        provenance_root.mkdir(parents=True, exist_ok=False)
        write_rows_csv(provenance_root / "input_inventory.csv", inventory_rows)
        write_rows_csv(
            provenance_root / "integer_window_geometry.csv", geometry_rows
        )
        manifest = {
            "method": "absolute-integer-sliding-window-correlations-v1",
            "window_geometry": {
                "nominal_start_deg": WINDOW_START_DEG,
                "width_deg": WINDOW_WIDTH_DEG,
                "step_deg": WINDOW_STEP_DEG,
                "sequence": "0-5, 1-6, 2-7, ...",
                "last_window_policy": (
                    "last nominal integer end not above common-grid floor"
                ),
                "edge_policy": EDGE_POLICY,
                "no_extrapolation": True,
            },
            "configuration_separation": {
                "single": _statistics_config_audit(single_bound.window_config),
                "powder": _statistics_config_audit(powder_bound.window_config),
                "explanation": (
                    "The profile's span-scaled 26-window geometry is replaced; "
                    "its preprocessing, fingerprint, shift, support, bootstrap, "
                    "confidence, and near/far statistical settings are retained."
                ),
            },
            "intensity_preprocessing": {
                **asdict(transform_config),
                "position_in_pipeline": (
                    "after unchanged clean_xy/AsLS preprocessing and before "
                    "common-grid resampling, window extraction, "
                    "standardization, ACF, and Pearson correlation"
                ),
                "role_scale_definition": (
                    "one a=Q(scale_quantile, abs(signed residual)) pooled over "
                    "all frames independently for each role/channel"
                ),
                "noise_definition": (
                    "sigma=median(preprocessed robust noise across role frames)"
                ),
                "normalized_residual": "z=clip(residual/a,-1,1)",
                "log_squared_formula": (
                    "log1p(z^2/epsilon)/log1p(1/epsilon), "
                    "epsilon=max((sigma/a)^2,epsilon_floor)"
                ),
                "output_contract": "zero->0; finite output in [0,1]",
                "derived_role_parameters": "recorded under every roles[] entry",
            },
            "roles": validations,
            "inputs": {
                "single_root": str(single_root),
                "single_manifest": str(single_manifest),
                "single_profile": str(single_profile_path),
                "single_profile_sha256": single_profile_hash,
                "single_wavelength_A": float(single_wavelength_A),
                "powder_root": str(powder_root),
                "powder_manifest": str(powder_manifest),
                "powder_profile": str(powder_profile_path),
                "powder_profile_sha256": powder_profile_hash,
                "powder_wavelength_A": float(powder_wavelength_A),
            },
            "workers": max(1, int(workers)),
            "max_scans": max_scans,
            "full_symmetric_plots_written": bool(make_full_symmetric_plots),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "elapsed_seconds": time.time() - start_time,
        }
        write_json(
            provenance_root / "integer_window_run_manifest.json", manifest
        )
        methods_path.write_text(
            f"""# Integer sliding-window correlation methods

The nominal 2theta windows are fixed in absolute degrees:
`0-5, 1-6, 2-7, ...`.  Every window is 5 degrees wide and adjacent starts
are separated by 1 degree.  Single-crystal windows end at `18-23`; powder
windows end at `27-32`.

Intensity preprocessing mode: `{transform_config.mode}`.  The raw XY curves
first receive the unchanged cleaning and AsLS baseline subtraction.  Within
each role/channel, one shared
`a=Q{100.0 * transform_config.scale_quantile:g}(abs(residual))` and the median
preprocessed noise `sigma` are then pooled across every accepted frame.  The
bounded squared transform is applied to `z=clip(residual/a,-1,1)` before the
otherwise unchanged common-grid, window, standardization, ACF, and Pearson
pipeline.  The complete role-wide parameters and per-frame clipping audit are
recorded in
`window_provenance/integer_window_run_manifest.json` and
`window_provenance/input_inventory.csv`.

The detector grids begin slightly above zero (single:
{validations[0]['common_grid_min_deg']:.8f} degrees; powder:
{validations[1]['common_grid_min_deg']:.8f} degrees).  The first nominal `0-5`
window therefore uses
only observed support from the detector lower edge through 5 degrees, sampled
on the same normalized coordinate as the other windows.  Nothing below the
detector edge is extrapolated or padded.  Exact nominal and effective bounds
are recorded in each role's `window_definition.csv` and in
`window_provenance/integer_window_geometry.csv`.

Across-frame outputs compare the same angle window between pressure frames
within the same scan.  They include strict ACF Pearson similarity, direct
standardized-signal validation, and a +/-1-degree neighboring-window secondary
score.  Within-frame outputs compare every angle window with every other angle
window inside a frame using the ACF fingerprints.

The frozen profiles' old span-scaled 26-window geometry is not executed.
Only their preprocessing, fingerprint, shift, support, bootstrap, confidence,
and near/far statistical settings are inherited.  This separation and both
the old and executed geometry are explicit in the run manifest.

Primary presentation maps are strict lower triangles without the diagonal.
The full symmetric numerical sources remain under
`window_full_symmetric_audit/`.
""",
            encoding="utf-8",
        )

        total_files, total_bytes, suffix_counts = _payload_inventory(audit_root)
        audit_root.replace(final_audit_root)
        provenance_root.replace(final_provenance_root)
        methods_path.replace(final_methods_path)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    roles_verified = [
        "single_spots_across",
        "single_spots_within",
        "powder_spots_across",
        "powder_spots_within",
        "powder_fit_control_across",
        "powder_fit_control_within",
    ]
    return {
        "method": "absolute-integer-sliding-window-correlations-v1",
        "intensity_preprocessing": asdict(transform_config),
        "generated": validations,
        "roles_verified": roles_verified,
        "window_calculations_regenerated": True,
        "unchanged_from_v3_suite": False,
        "integer_window_geometry_verified": all(
            item["integer_geometry_verified"] for item in validations
        ),
        "all_source_matrices_symmetric": all(
            item["source_matrices_symmetric"] for item in validations
        ),
        "all_scores_in_minus1_plus1": all(
            item["finite_scores_in_minus1_plus1"] for item in validations
        ),
        "all_signal_and_acf_windows_valid": all(
            item["all_signal_and_acf_windows_valid"] for item in validations
        ),
        "all_intended_scopes_verified": all(
            item["intended_scope_verified"] for item in validations
        ),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "suffix_counts": suffix_counts,
        "single_window_scope": {
            "official_frames": validations[0]["frames"],
            "scans": validations[0]["scans"],
            "pressures": validations[0]["pressures"],
            "nominal_windows": validations[0]["nominal_windows"],
            "first_window": validations[0]["first_nominal_window"],
            "last_window": validations[0]["last_nominal_window"],
            "pressure_min_GPa": validations[0]["actual_scope"]["pressure_min_GPa"],
            "pressure_max_GPa": validations[0]["actual_scope"]["pressure_max_GPa"],
        },
        "powder_window_scope": {
            "accepted_frames": validations[1]["frames"],
            "scans": validations[1]["scans"],
            "pressures": validations[1]["pressures"],
            "nominal_windows": validations[1]["nominal_windows"],
            "first_window": validations[1]["first_nominal_window"],
            "last_window": validations[1]["last_nominal_window"],
            "pressure_min_GPa": validations[1]["actual_scope"]["pressure_min_GPa"],
            "pressure_max_GPa": validations[1]["actual_scope"]["pressure_max_GPa"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--full-symmetric-plots", action="store_true")
    parser.add_argument(
        "--transform-mode",
        choices=INTENSITY_TRANSFORM_MODES,
        default="none",
        help=(
            "Role-pooled bounded residual preprocessing after the unchanged "
            "baseline and before the window/ACF/Pearson pipeline."
        ),
    )
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
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the complete production run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.out_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audit = generate_integer_window_sources(
        output_root,
        workers=args.workers,
        make_full_symmetric_plots=args.full_symmetric_plots,
        max_scans=args.max_scans,
        intensity_transform=IntensityTransformConfig(
            mode=args.transform_mode,
            scale_quantile=args.transform_scale_quantile,
            epsilon_floor=args.transform_epsilon_floor,
        ),
    )
    print(json.dumps(json_ready(audit), indent=2))


if __name__ == "__main__":
    main()
