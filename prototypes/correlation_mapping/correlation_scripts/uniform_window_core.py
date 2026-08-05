#!/usr/bin/env python3
"""Deterministic window-correlation primitives for ``uniform-correlation-v2``.

This module contains no dataset-specific pressure, angle, or peak-track rules.
It accepts baseline-corrected one-dimensional radial-XRD residuals and keeps
independent scans separate until the explicit aggregation step.  The public
functions return NumPy arrays/dataclasses only; writing tables and figures is
the responsibility of the runner.

Array conventions
-----------------
``frames`` are ordered as supplied by the caller.  Window features have shape
``(n_frames, n_windows, n_samples)``.  Across-frame results have shape
``(n_scans, n_windows, n_pressures, n_pressures)`` and are full symmetric
matrices.  Within-frame results have shape
``(n_frames, n_windows, n_windows)`` and are also symmetric.
Missing or invalid comparisons are always represented by ``NaN``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import rankdata


ALGORITHM_VERSION = "2.0.0"
N_WINDOWS = 26
DEFAULT_COVERAGE_FRACTION = 0.90
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_RANDOM_SEED = 0
DEFAULT_MINIMUM_DISTINCT_PRESSURE_GAPS = 4
DEFAULT_MINIMUM_SUPPORTED_GROUP_VALUES = 5


@dataclass(frozen=True)
class UniformWindowConfig:
    """Resolved window/statistics settings for ``uniform-correlation-v2``.

    The official profile binder constructs this object field-by-field from the
    JSON profile.  Fixed formula names are validated by that binder; numeric
    values are passed here so the runner never falls back to module defaults.
    """

    algorithm_version: str = ALGORITHM_VERSION
    coverage_fraction: float = DEFAULT_COVERAGE_FRACTION
    allow_extrapolation: bool = False
    window_count: int = N_WINDOWS
    width_divisor: float = 6.0
    step_divisor: float = 5.0
    minimum_finite_fraction: float = 1.0
    fingerprint_method: str = "standardized_positive_lag_fft_acf"
    strict_acf_primary: bool = True
    direct_strict_validation: bool = True
    shift_tolerant_neighbor_steps: int = 1
    shift_tolerant_role: str = "SECONDARY"
    nonoverlap_stride_windows: int = 5
    bootstrap_iterations: int = DEFAULT_BOOTSTRAPS
    random_seed: int = DEFAULT_RANDOM_SEED
    confidence: float = 0.95
    near_gap_quantile: float = 0.25
    far_gap_quantile: float = 0.75
    minimum_distinct_pressure_gaps: int = DEFAULT_MINIMUM_DISTINCT_PRESSURE_GAPS
    minimum_supported_group_values: int = DEFAULT_MINIMUM_SUPPORTED_GROUP_VALUES

    def __post_init__(self) -> None:
        if self.algorithm_version != ALGORITHM_VERSION:
            raise ValueError(
                f"This module implements {ALGORITHM_VERSION!r}, not {self.algorithm_version!r}"
            )
        if not (0.0 < self.coverage_fraction <= 1.0):
            raise ValueError("coverage_fraction must be in (0, 1]")
        if self.allow_extrapolation:
            raise ValueError("uniform-correlation-v2 forbids extrapolation")
        if self.window_count != N_WINDOWS:
            raise ValueError(f"uniform-correlation-v2 requires exactly {N_WINDOWS} windows")
        if self.width_divisor != 6.0 or self.step_divisor != 5.0:
            raise ValueError("uniform-correlation-v2 requires W=L/6 and step=W/5")
        if not (0.0 < self.minimum_finite_fraction <= 1.0):
            raise ValueError("minimum_finite_fraction must be in (0, 1]")
        if self.fingerprint_method != "standardized_positive_lag_fft_acf":
            raise ValueError("unsupported ACF fingerprint method")
        if not self.strict_acf_primary or not self.direct_strict_validation:
            raise ValueError("uniform-correlation-v2 requires both strict correlation families")
        if self.shift_tolerant_role != "SECONDARY":
            raise ValueError("shift-tolerant output must remain SECONDARY")
        if self.shift_tolerant_neighbor_steps < 0:
            raise ValueError("shift_tolerant_neighbor_steps cannot be negative")
        if self.nonoverlap_stride_windows < 1:
            raise ValueError("nonoverlap_stride_windows must be positive")
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations cannot be negative")
        if not (0.0 < self.confidence < 1.0):
            raise ValueError("confidence must be in (0, 1)")
        if not (0.0 < self.near_gap_quantile < self.far_gap_quantile < 1.0):
            raise ValueError("near/far gap quantiles must be ordered inside (0, 1)")
        if self.minimum_distinct_pressure_gaps < 2:
            raise ValueError("minimum_distinct_pressure_gaps must be at least two")
        if self.minimum_supported_group_values < 1:
            raise ValueError("minimum_supported_group_values must be positive")


@dataclass(frozen=True)
class XYValidation:
    """Validation facts for one XY profile."""

    points: int
    all_finite: bool
    strictly_increasing: bool
    duplicate_coordinates: int
    median_step_deg: float
    min_step_deg: float
    max_step_deg: float

    @property
    def valid(self) -> bool:
        return self.points >= 2 and self.all_finite and self.strictly_increasing


@dataclass(frozen=True)
class CoverageInterval:
    """Largest angle interval covered by the requested fraction of frames."""

    lower_deg: float
    upper_deg: float
    required_frames: int
    total_frames: int
    coverage_fraction: float

    @property
    def span_deg(self) -> float:
        return self.upper_deg - self.lower_deg


@dataclass(frozen=True)
class ResampledBatch:
    """Profiles resampled without extrapolation onto a common uniform grid."""

    grid_deg: np.ndarray
    values: np.ndarray
    coverage_count: np.ndarray
    interval: CoverageInterval
    grid_step_deg: float
    validations: tuple[XYValidation, ...]


@dataclass(frozen=True)
class WindowSpec:
    """Window geometry used by the correlation primitives."""

    lower_deg: float
    upper_deg: float
    width_deg: float
    step_deg: float
    starts_deg: np.ndarray
    ends_deg: np.ndarray
    nonoverlap_indices: np.ndarray

    @property
    def span_deg(self) -> float:
        return self.upper_deg - self.lower_deg

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(
            f"{start:.6g}-{end:.6g}"
            for start, end in zip(self.starts_deg, self.ends_deg, strict=True)
        )


@dataclass(frozen=True)
class WindowFeatures:
    """Standardized residual windows and their standardized FFT ACFs."""

    spec: WindowSpec
    sample_fraction: np.ndarray
    signals: np.ndarray
    signal_valid: np.ndarray
    fingerprints: np.ndarray
    fingerprint_valid: np.ndarray


@dataclass(frozen=True)
class AcrossFrameCorrelations:
    """Three same-scan across-pressure correlation families."""

    scan_labels: tuple[str, ...]
    pressure_values: np.ndarray
    window_spec: WindowSpec
    availability_by_scan: np.ndarray
    acf_strict_by_scan: np.ndarray
    direct_strict_by_scan: np.ndarray
    shift_tolerant_by_scan: np.ndarray


@dataclass(frozen=True)
class WithinFrameCorrelations:
    """Window-to-window ACF correlations before and after scan aggregation."""

    scan_labels: tuple[str, ...]
    pressure_values: np.ndarray
    nonoverlap_indices: np.ndarray
    by_frame: np.ndarray
    by_scan: np.ndarray
    by_scan_pressure: np.ndarray
    aggregate: np.ndarray
    support: np.ndarray
    aggregate_by_pressure: np.ndarray
    support_by_pressure: np.ndarray
    nonoverlap_by_frame: np.ndarray
    nonoverlap_aggregate: np.ndarray


@dataclass(frozen=True)
class MatrixAggregate:
    """Median, support, and scan-bootstrap confidence interval for a map."""

    median: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    support: np.ndarray
    available: np.ndarray
    support_required: np.ndarray
    sufficient_support: np.ndarray
    confidence: float
    n_bootstrap: int
    seed: int


@dataclass(frozen=True)
class NearFarSummary:
    """Per-leading-feature near/far medians and AUC with scan-bootstrap CI."""

    near_gap_max: float
    far_gap_min: float
    near_median: np.ndarray
    far_median: np.ndarray
    auc: np.ndarray
    auc_ci_low: np.ndarray
    auc_ci_high: np.ndarray
    near_cells: np.ndarray
    far_cells: np.ndarray
    reasons: tuple[str, ...]
    feature_shape: tuple[int, ...]
    confidence: float
    n_bootstrap: int
    seed: int


def validate_xy(two_theta_deg: np.ndarray, intensity: np.ndarray) -> XYValidation:
    """Inspect one XY profile without silently sorting or merging coordinates."""

    x = np.asarray(two_theta_deg, dtype=float).reshape(-1)
    y = np.asarray(intensity, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"two_theta and intensity shapes differ: {x.shape} != {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    diffs = np.diff(x) if x.size >= 2 else np.asarray([], dtype=float)
    finite_diffs = diffs[np.isfinite(diffs)]
    positive = finite_diffs[finite_diffs > 0]
    return XYValidation(
        points=int(x.size),
        all_finite=bool(np.all(finite)),
        strictly_increasing=bool(x.size >= 2 and np.all(diffs > 0)),
        duplicate_coordinates=int(np.count_nonzero(diffs == 0)),
        median_step_deg=float(np.median(positive)) if positive.size else math.nan,
        min_step_deg=float(np.min(positive)) if positive.size else math.nan,
        max_step_deg=float(np.max(positive)) if positive.size else math.nan,
    )


def _validated_profiles(
    two_theta_frames: Sequence[np.ndarray],
    intensity_frames: Sequence[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, tuple[XYValidation, ...]]:
    if not two_theta_frames:
        raise ValueError("At least one XY frame is required")
    if intensity_frames is not None and len(two_theta_frames) != len(intensity_frames):
        raise ValueError("two_theta_frames and intensity_frames must have equal length")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] | None = [] if intensity_frames is not None else None
    facts: list[XYValidation] = []
    for index, raw_x in enumerate(two_theta_frames):
        x = np.asarray(raw_x, dtype=float).reshape(-1)
        raw_y = np.zeros_like(x) if intensity_frames is None else intensity_frames[index]
        y = np.asarray(raw_y, dtype=float).reshape(-1)
        fact = validate_xy(x, y)
        if not fact.valid:
            reasons: list[str] = []
            if fact.points < 2:
                reasons.append("fewer than two points")
            if not fact.all_finite:
                reasons.append("non-finite coordinate or intensity")
            if not fact.strictly_increasing:
                reasons.append("2theta is not strictly increasing")
            if fact.duplicate_coordinates:
                reasons.append(f"{fact.duplicate_coordinates} duplicate coordinates")
            raise ValueError(f"Invalid XY frame {index}: {', '.join(reasons)}")
        xs.append(x)
        if ys is not None:
            ys.append(y)
        facts.append(fact)
    return xs, ys, tuple(facts)


def common_coverage_interval(
    two_theta_frames: Sequence[np.ndarray],
    coverage_fraction: float = DEFAULT_COVERAGE_FRACTION,
) -> CoverageInterval:
    """Find the largest continuous interval covered by at least a frame fraction.

    Each validated 1-D profile contributes its closed ``[min, max]`` interval.
    The support count is piecewise constant between profile endpoints, so the
    exact longest qualifying segment can be found without an arbitrary grid.
    """

    if not (0.0 < coverage_fraction <= 1.0):
        raise ValueError("coverage_fraction must be in (0, 1]")
    xs, _, _ = _validated_profiles(two_theta_frames)
    total = len(xs)
    required = int(math.ceil(coverage_fraction * total))
    endpoints = np.unique(
        np.concatenate([np.asarray([x[0], x[-1]], dtype=float) for x in xs])
    )
    if endpoints.size < 2:
        raise ValueError("Profiles do not define a non-zero common angle interval")

    qualifying: list[tuple[float, float]] = []
    for left, right in zip(endpoints[:-1], endpoints[1:], strict=True):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        support = sum(bool(x[0] <= midpoint <= x[-1]) for x in xs)
        if support >= required:
            if qualifying and np.isclose(qualifying[-1][1], left, rtol=0.0, atol=1e-12):
                qualifying[-1] = (qualifying[-1][0], float(right))
            else:
                qualifying.append((float(left), float(right)))
    if not qualifying:
        raise ValueError(
            f"No positive-width interval is covered by {required}/{total} frames"
        )
    lower, upper = max(qualifying, key=lambda item: (item[1] - item[0], -item[0]))
    return CoverageInterval(
        lower_deg=lower,
        upper_deg=upper,
        required_frames=required,
        total_frames=total,
        coverage_fraction=float(coverage_fraction),
    )


def resample_common_grid(
    two_theta_frames: Sequence[np.ndarray],
    intensity_frames: Sequence[np.ndarray],
    *,
    coverage_fraction: float = DEFAULT_COVERAGE_FRACTION,
    grid_step_deg: float | None = None,
    coverage_interval: CoverageInterval | None = None,
) -> ResampledBatch:
    """Interpolate profiles onto a common grid, never beyond a profile's range."""

    xs, ys_or_none, validations = _validated_profiles(two_theta_frames, intensity_frames)
    assert ys_or_none is not None
    if coverage_interval is None:
        interval = common_coverage_interval(xs, coverage_fraction)
    else:
        interval = coverage_interval
        if interval.total_frames != len(xs):
            raise ValueError("coverage_interval frame count does not match profiles")
        if not math.isclose(
            interval.coverage_fraction, coverage_fraction, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("coverage_interval fraction does not match requested fraction")
        midpoint = 0.5 * (interval.lower_deg + interval.upper_deg)
        covered = sum(bool(x[0] <= midpoint <= x[-1]) for x in xs)
        if covered < interval.required_frames:
            raise ValueError("supplied coverage_interval is not covered by enough profiles")
    if grid_step_deg is None:
        per_frame_steps = np.asarray([item.median_step_deg for item in validations])
        grid_step_deg = float(np.median(per_frame_steps[np.isfinite(per_frame_steps)]))
    if not np.isfinite(grid_step_deg) or grid_step_deg <= 0:
        raise ValueError("grid_step_deg must be a finite positive number")
    n_points = int(math.floor(interval.span_deg / grid_step_deg + 1e-12)) + 1
    if n_points < 2:
        raise ValueError("Common interval is shorter than one grid step")
    grid = interval.lower_deg + np.arange(n_points, dtype=float) * grid_step_deg
    grid = grid[grid <= interval.upper_deg + 1e-12]
    values = np.full((len(xs), grid.size), np.nan, dtype=float)
    for index, (x, y) in enumerate(zip(xs, ys_or_none, strict=True)):
        inside = (grid >= x[0]) & (grid <= x[-1])
        values[index, inside] = np.interp(grid[inside], x, y)
    coverage = np.count_nonzero(np.isfinite(values), axis=0).astype(np.int32)
    return ResampledBatch(
        grid_deg=grid,
        values=values,
        coverage_count=coverage,
        interval=interval,
        grid_step_deg=float(grid_step_deg),
        validations=validations,
    )


def make_uniform_window_spec(
    lower_deg: float,
    upper_deg: float,
    config: UniformWindowConfig | None = None,
) -> WindowSpec:
    """Create exactly 26 windows with ``W=L/6`` and ``step=W/5``."""

    resolved = config or UniformWindowConfig()
    lower = float(lower_deg)
    upper = float(upper_deg)
    if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
        raise ValueError("upper_deg must be finite and greater than lower_deg")
    span = upper - lower
    width = span / resolved.width_divisor
    step = width / resolved.step_divisor
    # linspace pins the final end to upper and avoids cumulative rounding drift.
    starts = np.linspace(lower, upper - width, resolved.window_count, dtype=float)
    expected = lower + np.arange(resolved.window_count, dtype=float) * step
    if not np.allclose(starts, expected, rtol=1e-13, atol=1e-13):
        raise RuntimeError("Internal 26-window geometry invariant failed")
    ends = starts + width
    ends[-1] = upper
    nonoverlap = np.arange(
        0, resolved.window_count, resolved.nonoverlap_stride_windows, dtype=np.int32
    )
    return WindowSpec(
        lower_deg=lower,
        upper_deg=upper,
        width_deg=width,
        step_deg=step,
        starts_deg=starts,
        ends_deg=ends,
        nonoverlap_indices=nonoverlap,
    )


def make_fixed_sliding_window_spec(
    upper_available_deg: float,
    *,
    width_deg: float = 5.0,
    step_deg: float = 1.0,
    start_deg: float = 0.0,
) -> WindowSpec:
    """Create absolute sliding windows such as 0-5, 1-6, 2-7, and so on.

    Only windows whose nominal upper edge is no greater than the largest whole
    degree covered by the data are included.  The nominal labels remain exact
    integers; callers may clip a tiny detector-edge shortfall at the first
    sample without extrapolating.
    """

    upper_available = float(upper_available_deg)
    width = float(width_deg)
    step = float(step_deg)
    start = float(start_deg)
    if not all(np.isfinite(value) for value in (upper_available, width, step, start)):
        raise ValueError("fixed sliding-window parameters must be finite")
    if width <= 0.0 or step <= 0.0:
        raise ValueError("fixed window width and step must be positive")
    final_edge = float(math.floor(upper_available + 1.0e-12))
    if final_edge < start + width:
        raise ValueError("available angle range does not contain one complete window")
    count = int(math.floor((final_edge - start - width) / step + 1.0e-12)) + 1
    starts = start + np.arange(count, dtype=float) * step
    ends = starts + width
    stride = max(1, int(math.ceil(width / step - 1.0e-12)))
    return WindowSpec(
        lower_deg=start,
        upper_deg=float(ends[-1]),
        width_deg=width,
        step_deg=step,
        starts_deg=starts,
        ends_deg=ends,
        nonoverlap_indices=np.arange(0, count, stride, dtype=np.int32),
    )


def standardize_residual_windows(
    grid_deg: np.ndarray,
    residuals: np.ndarray,
    spec: WindowSpec | None = None,
    *,
    min_finite_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract, mean-center, and standardize baseline-corrected residuals.

    ``residuals`` must already have had its baseline removed by the shared
    profile-preprocessing stage.  All windows are sampled on an identical
    normalized coordinate grid so that direct correlations are well-defined
    even when ``width_deg / grid_step`` is not an integer.
    """

    grid = np.asarray(grid_deg, dtype=float).reshape(-1)
    values = np.asarray(residuals, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != grid.size:
        raise ValueError("residuals must have shape (n_frames, len(grid_deg))")
    if grid.size < 3 or not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
        raise ValueError("grid_deg must contain at least three finite increasing values")
    if not (0.0 < min_finite_fraction <= 1.0):
        raise ValueError("min_finite_fraction must be in (0, 1]")
    if spec is None:
        spec = make_uniform_window_spec(float(grid[0]), float(grid[-1]))
    step = float(np.median(np.diff(grid)))
    lower_shortfall = max(0.0, float(grid[0] - spec.lower_deg))
    upper_shortfall = max(0.0, float(spec.upper_deg - grid[-1]))
    edge_tolerance = min(0.05, 0.01 * float(spec.width_deg))
    if lower_shortfall > edge_tolerance + 1.0e-12 or upper_shortfall > 1.0e-12:
        raise ValueError(
            "WindowSpec extends materially beyond grid_deg; extrapolation is forbidden"
        )
    n_samples = max(3, int(math.floor(spec.width_deg / step + 1e-12)) + 1)
    sample_fraction = np.linspace(0.0, 1.0, n_samples, dtype=float)
    n_windows = len(spec.starts_deg)
    if n_windows < 1 or len(spec.ends_deg) != n_windows:
        raise ValueError("WindowSpec must contain equal non-empty start/end arrays")
    signals = np.full((values.shape[0], n_windows, n_samples), np.nan, dtype=float)
    valid = np.zeros((values.shape[0], n_windows), dtype=bool)

    for window_index, (start, end) in enumerate(
        zip(spec.starts_deg, spec.ends_deg, strict=True)
    ):
        effective_start = max(float(start), float(grid[0]))
        effective_end = min(float(end), float(grid[-1]))
        if effective_end <= effective_start:
            continue
        sample_x = effective_start + sample_fraction * (
            effective_end - effective_start
        )
        for frame_index, row in enumerate(values):
            sampled = np.interp(sample_x, grid, row, left=np.nan, right=np.nan)
            finite = np.isfinite(sampled)
            if np.count_nonzero(finite) < math.ceil(min_finite_fraction * n_samples):
                continue
            # Filling is allowed only between observed finite points.  With the
            # official default (1.0), this branch is normally a no-op.
            if not np.all(finite):
                finite_indices = np.flatnonzero(finite)
                if finite_indices[0] != 0 or finite_indices[-1] != n_samples - 1:
                    continue
                missing = ~finite
                sampled[missing] = np.interp(
                    np.flatnonzero(missing), finite_indices, sampled[finite]
                )
            centered = sampled - float(np.mean(sampled))
            scale = float(np.std(centered))
            magnitude = max(float(np.max(np.abs(sampled))), 1.0)
            if not np.isfinite(scale) or scale <= np.finfo(float).eps * magnitude:
                continue
            signals[frame_index, window_index] = centered / scale
            valid[frame_index, window_index] = True
    return signals, valid


def fft_acf_fingerprints(
    signals: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized positive-lag autocorrelation fingerprints via FFT."""

    array = np.asarray(signals, dtype=float)
    if array.ndim < 2 or array.shape[-1] < 3:
        raise ValueError("signals must have at least two dimensions and three samples")
    leading_shape = array.shape[:-1]
    if valid is None:
        valid_array = np.all(np.isfinite(array), axis=-1)
    else:
        valid_array = np.asarray(valid, dtype=bool)
        if valid_array.shape != leading_shape:
            raise ValueError("valid must have shape signals.shape[:-1]")
        valid_array = valid_array & np.all(np.isfinite(array), axis=-1)

    flat = array.reshape(-1, array.shape[-1])
    flat_valid = valid_array.reshape(-1)
    fingerprints = np.full((flat.shape[0], flat.shape[1] - 1), np.nan, dtype=float)
    if np.any(flat_valid):
        selected = flat[flat_valid]
        n = selected.shape[1]
        transformed = np.fft.rfft(selected, n=2 * n, axis=1)
        correlation = np.fft.irfft(
            transformed * np.conjugate(transformed), n=2 * n, axis=1
        )[:, :n]
        zero_lag = correlation[:, 0]
        good_zero = np.isfinite(zero_lag) & (zero_lag > np.finfo(float).eps)
        raw = np.full((selected.shape[0], n - 1), np.nan, dtype=float)
        raw[good_zero] = correlation[good_zero, 1:] / zero_lag[good_zero, None]
        means = np.nanmean(raw, axis=1, keepdims=True)
        centered = raw - means
        scales = np.nanstd(centered, axis=1)
        good_fp = good_zero & np.isfinite(scales) & (scales > 1e-12)
        standardized = np.full_like(raw, np.nan)
        standardized[good_fp] = centered[good_fp] / scales[good_fp, None]
        selected_indices = np.flatnonzero(flat_valid)
        fingerprints[selected_indices[good_fp]] = standardized[good_fp]
        flat_valid[selected_indices[~good_fp]] = False
    return fingerprints.reshape(*leading_shape, array.shape[-1] - 1), flat_valid.reshape(
        leading_shape
    )


def build_window_features(
    grid_deg: np.ndarray,
    residuals: np.ndarray,
    spec: WindowSpec | None = None,
    *,
    min_finite_fraction: float | None = None,
    config: UniformWindowConfig | None = None,
) -> WindowFeatures:
    """Build standardized residual and FFT-ACF features in one deterministic call."""

    grid = np.asarray(grid_deg, dtype=float).reshape(-1)
    resolved = config or UniformWindowConfig()
    if spec is None:
        spec = make_uniform_window_spec(float(grid[0]), float(grid[-1]), resolved)
    fraction = (
        resolved.minimum_finite_fraction
        if min_finite_fraction is None
        else float(min_finite_fraction)
    )
    signals, signal_valid = standardize_residual_windows(
        grid, residuals, spec, min_finite_fraction=fraction
    )
    fingerprints, fingerprint_valid = fft_acf_fingerprints(signals, signal_valid)
    sample_fraction = np.linspace(0.0, 1.0, signals.shape[-1], dtype=float)
    return WindowFeatures(
        spec=spec,
        sample_fraction=sample_fraction,
        signals=signals,
        signal_valid=signal_valid,
        fingerprints=fingerprints,
        fingerprint_valid=fingerprint_valid,
    )


def pearson_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Finite-only Pearson correlation; invalid vectors yield ``NaN``."""

    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.shape != b.shape or a.size < 2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return math.nan
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return math.nan
    return float(np.clip(np.dot(a, b) / norm, -1.0, 1.0))


def _row_correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlations between rows of two feature arrays."""

    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] < 2:
        raise ValueError("left/right must be 2-D arrays with equal feature length")
    valid_a = np.all(np.isfinite(a), axis=1)
    valid_b = np.all(np.isfinite(b), axis=1)
    clean_a = np.where(np.isfinite(a), a, 0.0)
    clean_b = np.where(np.isfinite(b), b, 0.0)
    clean_a -= np.mean(clean_a, axis=1, keepdims=True)
    clean_b -= np.mean(clean_b, axis=1, keepdims=True)
    norm_a = np.linalg.norm(clean_a, axis=1)
    norm_b = np.linalg.norm(clean_b, axis=1)
    valid_a &= np.isfinite(norm_a) & (norm_a > np.finfo(float).eps)
    valid_b &= np.isfinite(norm_b) & (norm_b > np.finfo(float).eps)
    denominator = norm_a[:, None] * norm_b[None, :]
    matrix = np.divide(
        clean_a @ clean_b.T,
        denominator,
        out=np.full((a.shape[0], b.shape[0]), np.nan, dtype=float),
        where=denominator > np.finfo(float).eps,
    )
    matrix[~valid_a, :] = np.nan
    matrix[:, ~valid_b] = np.nan
    return np.clip(matrix, -1.0, 1.0)


def _metadata_axes(
    scan_ids: Sequence[object], pressures: Sequence[float], n_frames: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, dict[str, int], dict[float, int]]:
    scans = np.asarray([str(value) for value in scan_ids], dtype=object)
    pressure_array = np.asarray(pressures, dtype=float).reshape(-1)
    if scans.size != n_frames or pressure_array.size != n_frames:
        raise ValueError("scan_ids and pressures must contain one value per frame")
    if not np.all(np.isfinite(pressure_array)):
        raise ValueError("pressures must be finite")
    scan_labels = tuple(sorted(set(scans.tolist())))
    pressure_values = np.unique(pressure_array)
    scan_to_index = {value: index for index, value in enumerate(scan_labels)}
    pressure_to_index = {float(value): index for index, value in enumerate(pressure_values)}
    seen: set[tuple[str, float]] = set()
    for scan, pressure in zip(scans, pressure_array, strict=True):
        key = (str(scan), float(pressure))
        if key in seen:
            raise ValueError(f"Duplicate frame for scan/pressure pair {key}")
        seen.add(key)
    return scans, pressure_array, scan_labels, pressure_values, scan_to_index, pressure_to_index


def compute_across_frame_correlations(
    features: WindowFeatures,
    scan_ids: Sequence[object],
    pressures: Sequence[float],
    *,
    config: UniformWindowConfig | None = None,
) -> AcrossFrameCorrelations:
    """Compute strict/direct/shift-tolerant maps using same-scan pairs only.

    The shift-tolerant secondary score is the maximum over the strict window
    and either immediate neighbor in *both* comparison directions.  Taking the
    symmetric candidate set prevents an arbitrary pressure ordering from
    changing the result.
    """

    resolved = config or UniformWindowConfig()
    signals = np.asarray(features.signals, dtype=float)
    fingerprints = np.asarray(features.fingerprints, dtype=float)
    if signals.ndim != 3 or fingerprints.ndim != 3:
        raise ValueError("WindowFeatures signals/fingerprints must be three-dimensional")
    if signals.shape[:2] != fingerprints.shape[:2] or signals.shape[1] < 1:
        raise ValueError("WindowFeatures shape invariant failed")
    (
        scans,
        frame_pressures,
        scan_labels,
        pressure_values,
        scan_to_index,
        pressure_to_index,
    ) = _metadata_axes(scan_ids, pressures, signals.shape[0])
    n_scans = len(scan_labels)
    n_pressures = pressure_values.size
    n_windows = signals.shape[1]
    shape = (n_scans, n_windows, n_pressures, n_pressures)
    acf_strict = np.full(shape, np.nan, dtype=float)
    direct_strict = np.full(shape, np.nan, dtype=float)
    shift_tolerant = np.full(shape, np.nan, dtype=float)
    availability = np.zeros((n_scans, n_pressures, n_pressures), dtype=bool)

    rows_by_scan: dict[str, list[int]] = {scan: [] for scan in scan_labels}
    for row, scan in enumerate(scans):
        rows_by_scan[str(scan)].append(row)
    for scan in scan_labels:
        scan_index = scan_to_index[scan]
        rows = rows_by_scan[scan]
        pressure_indices = np.asarray(
            [pressure_to_index[float(frame_pressures[row])] for row in rows], dtype=int
        )
        availability[scan_index][np.ix_(pressure_indices, pressure_indices)] = True
        pressure_ix = np.ix_(pressure_indices, pressure_indices)
        for window in range(n_windows):
            strict_acf = _row_correlation_matrix(
                fingerprints[rows, window], fingerprints[rows, window]
            )
            strict_direct = _row_correlation_matrix(
                signals[rows, window], signals[rows, window]
            )
            secondary = np.full(strict_acf.shape, np.nan, dtype=float)
            for delta in range(
                -resolved.shift_tolerant_neighbor_steps,
                resolved.shift_tolerant_neighbor_steps + 1,
            ):
                shifted = window + delta
                if not 0 <= shifted < n_windows:
                    continue
                directional = _row_correlation_matrix(
                    fingerprints[rows, window], fingerprints[rows, shifted]
                )
                symmetric_candidate = np.fmax(directional, directional.T)
                secondary = np.fmax(secondary, symmetric_candidate)
            acf_strict[scan_index, window][pressure_ix] = strict_acf
            direct_strict[scan_index, window][pressure_ix] = strict_direct
            shift_tolerant[scan_index, window][pressure_ix] = secondary
    return AcrossFrameCorrelations(
        scan_labels=scan_labels,
        pressure_values=pressure_values,
        window_spec=features.spec,
        availability_by_scan=availability,
        acf_strict_by_scan=acf_strict,
        direct_strict_by_scan=direct_strict,
        shift_tolerant_by_scan=shift_tolerant,
    )


def _nanmedian(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def support_threshold(n_available: int | np.ndarray) -> int | np.ndarray:
    """Apply ``S(N)=min[N,max(5,ceil(0.1N))]`` elementwise."""

    array = np.asarray(n_available)
    if np.any(array < 0):
        raise ValueError("n_available cannot be negative")
    result = np.minimum(array, np.maximum(5, np.ceil(0.1 * array).astype(int))).astype(int)
    if result.ndim == 0:
        return int(result)
    return result


def compute_within_frame_correlations(
    fingerprints: np.ndarray,
    scan_ids: Sequence[object],
    pressures: Sequence[float],
    *,
    nonoverlap_indices: np.ndarray | None = None,
    config: UniformWindowConfig | None = None,
) -> WithinFrameCorrelations:
    """Compute full within-frame ACF matrices and non-overlapping controls."""

    resolved = config or UniformWindowConfig()
    fp = np.asarray(fingerprints, dtype=float)
    if fp.ndim != 3 or fp.shape[1] < 1:
        raise ValueError(
            "fingerprints must have shape (n_frames, n_windows, n_lags)"
        )
    n_windows = fp.shape[1]
    (
        scans,
        frame_pressures,
        scan_labels,
        pressure_values,
        scan_to_index,
        pressure_to_index,
    ) = _metadata_axes(scan_ids, pressures, fp.shape[0])
    if nonoverlap_indices is None:
        nonoverlap = np.arange(
            0, n_windows, resolved.nonoverlap_stride_windows, dtype=np.int32
        )
    else:
        nonoverlap = np.asarray(nonoverlap_indices, dtype=np.int32).reshape(-1)
        if (
            nonoverlap.size == 0
            or np.any(nonoverlap < 0)
            or np.any(nonoverlap >= n_windows)
            or np.unique(nonoverlap).size != nonoverlap.size
        ):
            raise ValueError("nonoverlap_indices must be unique valid window indices")

    by_frame = np.full(
        (fp.shape[0], n_windows, n_windows), np.nan, dtype=float
    )
    for frame in range(fp.shape[0]):
        by_frame[frame] = _row_correlation_matrix(fp[frame], fp[frame])

    by_scan_pressure = np.full(
        (len(scan_labels), pressure_values.size, n_windows, n_windows),
        np.nan,
        dtype=float,
    )
    for frame, (scan, pressure) in enumerate(zip(scans, frame_pressures, strict=True)):
        by_scan_pressure[
            scan_to_index[str(scan)], pressure_to_index[float(pressure)]
        ] = by_frame[frame]
    by_scan = _nanmedian(by_scan_pressure, axis=1)
    support = np.count_nonzero(np.isfinite(by_scan), axis=0).astype(np.int32)
    required = support_threshold(len(scan_labels))
    aggregate = _nanmedian(by_scan, axis=0)
    aggregate[support < required] = np.nan

    support_by_pressure = np.count_nonzero(np.isfinite(by_scan_pressure), axis=0).astype(
        np.int32
    )
    available_by_pressure = np.count_nonzero(
        np.any(np.isfinite(by_scan_pressure), axis=(-2, -1)), axis=0
    )
    required_by_pressure = np.asarray(support_threshold(available_by_pressure), dtype=int)
    aggregate_by_pressure = _nanmedian(by_scan_pressure, axis=0)
    aggregate_by_pressure[
        support_by_pressure < required_by_pressure[:, None, None]
    ] = np.nan

    nonoverlap_by_frame = by_frame[:, nonoverlap][:, :, nonoverlap]
    nonoverlap_aggregate = aggregate[np.ix_(nonoverlap, nonoverlap)]
    return WithinFrameCorrelations(
        scan_labels=scan_labels,
        pressure_values=pressure_values,
        nonoverlap_indices=nonoverlap,
        by_frame=by_frame,
        by_scan=by_scan,
        by_scan_pressure=by_scan_pressure,
        aggregate=aggregate,
        support=support,
        aggregate_by_pressure=aggregate_by_pressure,
        support_by_pressure=support_by_pressure,
        nonoverlap_by_frame=nonoverlap_by_frame,
        nonoverlap_aggregate=nonoverlap_aggregate,
    )


def _broadcast_availability(values: np.ndarray, availability: np.ndarray | None) -> np.ndarray:
    if availability is None:
        return np.ones(values.shape, dtype=bool)
    available = np.asarray(availability, dtype=bool)
    if available.ndim == values.ndim - 1 and available.shape[0] == values.shape[0]:
        # Common across-frame case: (scan, pressure, pressure) -> insert windows.
        available = np.expand_dims(available, axis=1)
    try:
        return np.broadcast_to(available, values.shape)
    except ValueError as exc:
        raise ValueError(
            f"availability shape {available.shape} is not broadcastable to {values.shape}"
        ) from exc


def _bootstrap_scan_median_ci(
    values_by_scan: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float,
    sufficient: np.ndarray,
    cell_chunk: int = 192,
    draw_batch: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    output_shape = values_by_scan.shape[1:]
    low = np.full(output_shape, np.nan, dtype=float)
    high = np.full(output_shape, np.nan, dtype=float)
    if n_bootstrap <= 0 or values_by_scan.shape[0] == 0:
        return low, high
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, values_by_scan.shape[0], size=(n_bootstrap, values_by_scan.shape[0])
    )
    flat = values_by_scan.reshape(values_by_scan.shape[0], -1)
    flat_sufficient = np.asarray(sufficient, dtype=bool).reshape(-1)
    flat_low = low.reshape(-1)
    flat_high = high.reshape(-1)
    alpha = (1.0 - confidence) / 2.0
    for cell_start in range(0, flat.shape[1], cell_chunk):
        cell_stop = min(cell_start + cell_chunk, flat.shape[1])
        boot_medians = np.full((n_bootstrap, cell_stop - cell_start), np.nan, dtype=float)
        for draw_start in range(0, n_bootstrap, draw_batch):
            draw_stop = min(draw_start + draw_batch, n_bootstrap)
            sampled = flat[draws[draw_start:draw_stop], cell_start:cell_stop]
            boot_medians[draw_start:draw_stop] = _nanmedian(sampled, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            chunk_low = np.nanquantile(boot_medians, alpha, axis=0)
            chunk_high = np.nanquantile(boot_medians, 1.0 - alpha, axis=0)
        good = flat_sufficient[cell_start:cell_stop]
        flat_low[cell_start:cell_stop][good] = chunk_low[good]
        flat_high[cell_start:cell_stop][good] = chunk_high[good]
    return low, high


def aggregate_scan_matrices(
    values_by_scan: np.ndarray,
    availability_by_scan: np.ndarray | None = None,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_RANDOM_SEED,
    confidence: float = 0.95,
) -> MatrixAggregate:
    """Median-aggregate independent scans with support masking and bootstrap CI."""

    values = np.asarray(values_by_scan, dtype=float)
    if values.ndim < 2 or values.shape[0] < 1:
        raise ValueError("values_by_scan must have scan as its non-empty first axis")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap cannot be negative")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    availability = _broadcast_availability(values, availability_by_scan)
    masked = np.where(availability, values, np.nan)
    available = np.count_nonzero(availability, axis=0).astype(np.int32)
    support = np.count_nonzero(np.isfinite(masked), axis=0).astype(np.int32)
    required = np.asarray(support_threshold(available), dtype=np.int32)
    sufficient = (available > 0) & (support >= required)
    median = _nanmedian(masked, axis=0)
    median = np.where(sufficient, median, np.nan)
    ci_low, ci_high = _bootstrap_scan_median_ci(
        masked,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
        sufficient=sufficient,
    )
    return MatrixAggregate(
        median=median,
        ci_low=ci_low,
        ci_high=ci_high,
        support=support,
        available=available,
        support_required=required,
        sufficient_support=sufficient,
        confidence=float(confidence),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
    )


def auc_probability(near: np.ndarray, far: np.ndarray) -> float:
    """Return ``P(near > far) + 0.5 P(tie)`` using average ranks."""

    near_values = np.asarray(near, dtype=float).reshape(-1)
    far_values = np.asarray(far, dtype=float).reshape(-1)
    near_values = near_values[np.isfinite(near_values)]
    far_values = far_values[np.isfinite(far_values)]
    if near_values.size == 0 or far_values.size == 0:
        return math.nan
    combined = np.concatenate([far_values, near_values])
    ranks = rankdata(combined, method="average")
    rank_sum_near = float(np.sum(ranks[far_values.size :]))
    u_near = rank_sum_near - near_values.size * (near_values.size + 1) / 2.0
    return float(u_near / (near_values.size * far_values.size))


def pressure_gap_quantiles(
    pressure_values: Sequence[float],
    minimum_distinct_gaps: int = DEFAULT_MINIMUM_DISTINCT_PRESSURE_GAPS,
    near_gap_quantile: float = 0.25,
    far_gap_quantile: float = 0.75,
) -> tuple[float, float]:
    """Return Q25/Q75 of all non-zero, unique-axis pressure-pair gaps."""

    pressures = np.unique(np.asarray(pressure_values, dtype=float))
    if minimum_distinct_gaps < 2:
        raise ValueError("minimum_distinct_gaps must be at least 2")
    if not (0.0 < near_gap_quantile < far_gap_quantile < 1.0):
        raise ValueError("near/far gap quantiles must be ordered inside (0, 1)")
    if pressures.size < 2 or not np.all(np.isfinite(pressures)):
        return math.nan, math.nan
    lower = np.tril_indices(pressures.size, k=-1)
    gaps = np.abs(pressures[:, None] - pressures[None, :])[lower]
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    # Decimal pressures can produce tiny binary differences for mathematically
    # equal gaps.  Twelve decimal places is far below experimental precision
    # and keeps the distinct-gap feasibility check stable.
    if gaps.size == 0 or np.unique(np.round(gaps, 12)).size < minimum_distinct_gaps:
        return math.nan, math.nan
    return (
        float(np.quantile(gaps, near_gap_quantile)),
        float(np.quantile(gaps, far_gap_quantile)),
    )


def _batched_auc(near: np.ndarray, far: np.ndarray) -> np.ndarray:
    """Pairwise AUC for rows, with NaN-aware denominators."""

    near_values = np.asarray(near, dtype=float)
    far_values = np.asarray(far, dtype=float)
    valid = np.isfinite(near_values[:, :, None]) & np.isfinite(far_values[:, None, :])
    greater = (near_values[:, :, None] > far_values[:, None, :]) & valid
    equal = (near_values[:, :, None] == far_values[:, None, :]) & valid
    count = np.count_nonzero(valid, axis=(1, 2))
    numerator = np.count_nonzero(greater, axis=(1, 2)) + 0.5 * np.count_nonzero(
        equal, axis=(1, 2)
    )
    return np.divide(
        numerator,
        count,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=count > 0,
    )


def near_far_auc_summary(
    values_by_scan: np.ndarray,
    pressure_values: Sequence[float],
    availability_by_scan: np.ndarray | None = None,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_RANDOM_SEED,
    confidence: float = 0.95,
    minimum_distinct_gaps: int = DEFAULT_MINIMUM_DISTINCT_PRESSURE_GAPS,
    minimum_group_values: int = DEFAULT_MINIMUM_SUPPORTED_GROUP_VALUES,
    near_gap_quantile: float = 0.25,
    far_gap_quantile: float = 0.75,
) -> NearFarSummary:
    """Summarize each map feature using Q25-near/Q75-far scan-bootstrap AUC.

    The last two axes must be the same pressure axis.  Any leading axes after
    the scan axis are treated as independent features (normally 26 windows).
    AUC is computed from the scan-median aggregate map, while its confidence
    interval is obtained by resampling whole scans.
    """

    values = np.asarray(values_by_scan, dtype=float)
    pressures = np.asarray(pressure_values, dtype=float).reshape(-1)
    if values.ndim < 3 or values.shape[-2:] != (pressures.size, pressures.size):
        raise ValueError("Last two values_by_scan axes must match pressure_values")
    if n_bootstrap < 0 or not (0.0 < confidence < 1.0):
        raise ValueError("Invalid bootstrap count or confidence")
    if minimum_group_values < 1:
        raise ValueError("minimum_group_values must be at least 1")
    near_gap, far_gap = pressure_gap_quantiles(
        pressures,
        minimum_distinct_gaps,
        near_gap_quantile,
        far_gap_quantile,
    )
    feature_shape = values.shape[1:-2]
    feature_count = int(np.prod(feature_shape)) if feature_shape else 1
    out_shape = feature_shape if feature_shape else ()
    near_median = np.full(feature_count, np.nan, dtype=float)
    far_median = np.full(feature_count, np.nan, dtype=float)
    auc = np.full(feature_count, np.nan, dtype=float)
    auc_low = np.full(feature_count, np.nan, dtype=float)
    auc_high = np.full(feature_count, np.nan, dtype=float)
    near_cells = np.zeros(feature_count, dtype=np.int32)
    far_cells = np.zeros(feature_count, dtype=np.int32)
    reasons = ["" for _ in range(feature_count)]

    if not np.isfinite(near_gap) or not np.isfinite(far_gap) or near_gap >= far_gap:
        reasons = ["insufficient distinct non-zero pressure gaps" for _ in reasons]
        return NearFarSummary(
            near_gap_max=near_gap,
            far_gap_min=far_gap,
            near_median=near_median.reshape(out_shape),
            far_median=far_median.reshape(out_shape),
            auc=auc.reshape(out_shape),
            auc_ci_low=auc_low.reshape(out_shape),
            auc_ci_high=auc_high.reshape(out_shape),
            near_cells=near_cells.reshape(out_shape),
            far_cells=far_cells.reshape(out_shape),
            reasons=tuple(reasons),
            feature_shape=feature_shape,
            confidence=float(confidence),
            n_bootstrap=int(n_bootstrap),
            seed=int(seed),
        )

    availability = _broadcast_availability(values, availability_by_scan)
    masked = np.where(availability, values, np.nan)
    aggregate = aggregate_scan_matrices(
        values, availability, n_bootstrap=0, seed=seed, confidence=confidence
    )
    flat_values = masked.reshape(masked.shape[0], feature_count, pressures.size, pressures.size)
    flat_median = aggregate.median.reshape(feature_count, pressures.size, pressures.size)
    flat_sufficient = aggregate.sufficient_support.reshape(
        feature_count, pressures.size, pressures.size
    )
    lower = np.tril_indices(pressures.size, k=-1)
    gaps = np.abs(pressures[:, None] - pressures[None, :])
    lower_gaps = gaps[lower]
    near_selector = lower_gaps <= near_gap
    far_selector = lower_gaps >= far_gap
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.shape[0], size=(n_bootstrap, values.shape[0]))
    alpha = (1.0 - confidence) / 2.0

    for feature in range(feature_count):
        finite_matrix = np.where(flat_sufficient[feature], flat_median[feature], np.nan)
        lower_values = finite_matrix[lower]
        near = lower_values[near_selector]
        far = lower_values[far_selector]
        near = near[np.isfinite(near)]
        far = far[np.isfinite(far)]
        near_cells[feature] = near.size
        far_cells[feature] = far.size
        if near.size < minimum_group_values or far.size < minimum_group_values:
            reasons[feature] = "insufficient supported near or far pressure cells"
            continue
        near_median[feature] = float(np.median(near))
        far_median[feature] = float(np.median(far))
        auc[feature] = auc_probability(near, far)
        if n_bootstrap == 0:
            continue
        bootstrap_auc = np.full(n_bootstrap, np.nan, dtype=float)
        for draw_start in range(0, n_bootstrap, 64):
            draw_stop = min(draw_start + 64, n_bootstrap)
            sampled = flat_values[draws[draw_start:draw_stop], feature]
            medians = _nanmedian(sampled, axis=1)
            medians[:, ~flat_sufficient[feature]] = np.nan
            lower_boot = medians[:, lower[0], lower[1]]
            bootstrap_auc[draw_start:draw_stop] = _batched_auc(
                lower_boot[:, near_selector], lower_boot[:, far_selector]
            )
        finite_auc = bootstrap_auc[np.isfinite(bootstrap_auc)]
        if finite_auc.size:
            auc_low[feature] = float(np.quantile(finite_auc, alpha))
            auc_high[feature] = float(np.quantile(finite_auc, 1.0 - alpha))
        else:
            reasons[feature] = "bootstrap produced no finite near/far AUC"

    return NearFarSummary(
        near_gap_max=near_gap,
        far_gap_min=far_gap,
        near_median=near_median.reshape(out_shape),
        far_median=far_median.reshape(out_shape),
        auc=auc.reshape(out_shape),
        auc_ci_low=auc_low.reshape(out_shape),
        auc_ci_high=auc_high.reshape(out_shape),
        near_cells=near_cells.reshape(out_shape),
        far_cells=far_cells.reshape(out_shape),
        reasons=tuple(reasons),
        feature_shape=feature_shape,
        confidence=float(confidence),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
    )


__all__ = [
    "ALGORITHM_VERSION",
    "DEFAULT_MINIMUM_DISTINCT_PRESSURE_GAPS",
    "DEFAULT_MINIMUM_SUPPORTED_GROUP_VALUES",
    "N_WINDOWS",
    "UniformWindowConfig",
    "AcrossFrameCorrelations",
    "CoverageInterval",
    "MatrixAggregate",
    "NearFarSummary",
    "ResampledBatch",
    "WindowFeatures",
    "WindowSpec",
    "WithinFrameCorrelations",
    "XYValidation",
    "aggregate_scan_matrices",
    "auc_probability",
    "build_window_features",
    "common_coverage_interval",
    "compute_across_frame_correlations",
    "compute_within_frame_correlations",
    "fft_acf_fingerprints",
    "make_fixed_sliding_window_spec",
    "make_uniform_window_spec",
    "near_far_auc_summary",
    "pearson_similarity",
    "pressure_gap_quantiles",
    "resample_common_grid",
    "standardize_residual_windows",
    "support_threshold",
    "validate_xy",
]
