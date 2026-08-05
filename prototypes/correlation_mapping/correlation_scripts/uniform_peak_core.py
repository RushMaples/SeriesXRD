#!/usr/bin/env python3
"""Deterministic, data-agnostic per-peak core for ``uniform-correlation-v2``.

This module deliberately does *not* consume a hand-curated peak/track table.
Every one-dimensional radial XY pattern is treated with the same frozen rules:

* deterministic XY cleanup and AsLS background subtraction;
* robust noise estimation and blind peak discovery;
* joint pseudo-Voigt fitting for overlapping candidate groups;
* complete-link pressure-level consensus across independent scans;
* constant-velocity Hungarian tracking in both pressure directions;
* explicit ``present``/``absent``/``unknown``/``out_of_range`` states; and
* conditional area/location correlations with scan-level bootstrap intervals.

The functions are intentionally free of UOTe-specific peak positions, pressure
ranges, or expected slopes.  The wavelength is mandatory in
:class:`UniformPeakConfig`, so it cannot be silently inherited from an older
analysis.  One-dimensional XY data contain radial information only; this core
therefore names outputs ``radial_peak_NNN`` and never claims to recover an
azimuth-specific spot identity.

The implementation is conservative.  A failed or poorly identified fit is
``unknown`` rather than being replaced by the nearest/highest feature.  Missing
or unknown observations consequently become NaN in conditional area/location
maps; presence is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.signal import find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import rankdata


ALGORITHM_VERSION = "2.0.0"
_SQRT_2_LN_2 = math.sqrt(2.0 * math.log(2.0))
_FWHM_TO_SIGMA = 1.0 / (2.0 * _SQRT_2_LN_2)


@dataclass(frozen=True)
class UniformPeakConfig:
    """Frozen scientific settings for the uniform radial peak workflow.

    ``wavelength`` has no default by design.  The remaining values implement
    the formulas in the v2 analysis plan and should be serialized verbatim in
    a run manifest.  Trajectory gates/costs are dimensionless functions of the
    fitted widths and pressure sampling, not material-specific q tolerances.
    """

    wavelength: float
    algorithm_version: str = ALGORITHM_VERSION
    asls_p: float = 0.01
    asls_iterations: int = 10
    asls_min_span_deg: float = 0.5
    asls_min_span_bins: int = 20
    gaussian_sigma_bins: float = 1.0
    prominence_noise_factor: float = 5.0
    height_noise_factor: float = 3.0
    minimum_width_bins: float = 2.0
    delta_bic_minimum: float = 10.0
    area_over_se_minimum: float = 3.0
    fit_loss: str = "soft_l1"
    fit_max_nfev: int = 4000
    boundary_relative_tolerance: float = 1.0e-4
    reject_parameter_bound_hits: bool = True
    overlap_grouping: bool = True
    consensus_fwhm_factor: float = 0.5
    max_missing_pressure_levels: int = 2
    track_gate_factor: float = 1.5
    track_width_cost_weight: float = 0.1
    ambiguous_cost_margin: float = 0.25
    bootstrap_iterations: int = 2000
    random_seed: int = 0
    ci_percentiles: tuple[float, float] = (2.5, 97.5)
    near_gap_quantile: float = 0.25
    far_gap_quantile: float = 0.75
    minimum_distinct_pressure_gaps: int = 4
    minimum_supported_group_values: int = 5

    def __post_init__(self) -> None:
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("A finite, positive wavelength must be supplied explicitly")
        if self.algorithm_version != ALGORITHM_VERSION:
            raise ValueError(
                f"This module implements {ALGORITHM_VERSION!r}, not {self.algorithm_version!r}"
            )
        if not (0 < self.asls_p < 1):
            raise ValueError("asls_p must be between zero and one")
        if self.asls_iterations < 1 or self.minimum_width_bins <= 0:
            raise ValueError("iteration and width settings must be positive")
        if self.bootstrap_iterations < 0 or self.max_missing_pressure_levels < 0:
            raise ValueError("bootstrap/missing-level settings cannot be negative")
        if self.fit_max_nfev < 1 or not (0.0 < self.boundary_relative_tolerance < 1.0):
            raise ValueError("fit iteration/boundary settings are invalid")
        ci_low, ci_high = self.ci_percentiles
        if not (0.0 <= ci_low < ci_high <= 100.0):
            raise ValueError("ci_percentiles must be ordered within [0, 100]")
        if not (0.0 < self.near_gap_quantile < self.far_gap_quantile < 1.0):
            raise ValueError("near/far gap quantiles must be ordered inside (0, 1)")
        if self.minimum_distinct_pressure_gaps < 2:
            raise ValueError("minimum_distinct_pressure_gaps must be at least two")
        if self.minimum_supported_group_values < 1:
            raise ValueError("minimum_supported_group_values must be positive")


@dataclass(frozen=True)
class CleanXY:
    """A checked, strictly increasing XY pattern and its cleanup audit."""

    x: np.ndarray
    y: np.ndarray
    dx: float
    original_count: int
    finite_removed: int
    duplicate_points_merged: int
    originally_strictly_increasing: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreprocessedXY:
    """Reusable background-corrected representation of one XY pattern."""

    x: np.ndarray
    y: np.ndarray
    baseline: np.ndarray
    residual: np.ndarray
    positive_residual: np.ndarray
    smoothed_residual: np.ndarray
    noise: float
    dx: float
    total_positive_area: float
    cleaned: CleanXY
    valid: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PeakFit:
    """One blind candidate after grouped pseudo-Voigt fitting."""

    peak_id: int
    candidate_index: int
    state: str
    reason: str
    two_theta: float
    q: float
    fwhm_two_theta: float
    fwhm_q: float
    area: float
    area_se: float
    relative_area: float
    eta: float
    height_snr: float
    delta_bic: float
    fit_success: bool
    at_parameter_boundary: bool
    group_id: int
    fit_model: str = "pseudo_voigt"

    @property
    def reliable(self) -> bool:
        return self.state == "reliable"


@dataclass(frozen=True)
class FramePeaks:
    """All candidate fits for one scan/pressure/channel pattern."""

    frame: int
    scan: str
    pressure: float
    channel: str
    peaks: tuple[PeakFit, ...]
    pattern_valid: bool
    noise: float
    total_positive_area: float
    measured_q_min: float = math.nan
    measured_q_max: float = math.nan
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PressureConsensus:
    """A radial peak consensus at one pressure across independent scans."""

    consensus_id: str
    channel: str
    pressure: float
    pressure_index: int
    q: float
    fwhm_q: float
    relative_area: float
    support: int
    total_scans: int
    required_support: int
    member_keys: tuple[tuple[str, int, int], ...]
    reliable: bool
    ambiguous: bool = False


@dataclass(frozen=True)
class TrackNode:
    """A pressure consensus embedded in a radial trajectory."""

    consensus_id: str
    pressure: float
    pressure_index: int
    q: float
    fwhm_q: float
    relative_area: float
    support: int
    ambiguous: bool = False


@dataclass(frozen=True)
class RadialTrack:
    """A bidirectionally consistent radial trajectory."""

    track_id: str
    channel: str
    nodes: tuple[TrackNode, ...]
    official: bool
    ambiguous: bool
    minimum_pressure_support: int

    @property
    def pressure_min(self) -> float:
        return min(node.pressure for node in self.nodes)

    @property
    def pressure_max(self) -> float:
        return max(node.pressure for node in self.nodes)


@dataclass(frozen=True)
class AssignedObservation:
    """State and optional measurement for one track/scan/pressure cell."""

    track_id: str
    scan: str
    pressure: float
    frame: int | None
    state: str
    reason: str
    q: float = math.nan
    fwhm_q: float = math.nan
    relative_area: float = math.nan
    peak_id: int | None = None


@dataclass(frozen=True)
class CorrelationMatrices:
    """Conditional similarities, presence, support, and bootstrap intervals."""

    scan_labels: tuple[str, ...]
    pressure_levels: tuple[float, ...]
    area_by_scan: np.ndarray
    location_by_scan: np.ndarray
    presence_by_scan: np.ndarray
    area: np.ndarray
    location: np.ndarray
    presence: np.ndarray
    n_available: np.ndarray
    n_both_present: np.ndarray
    n10: np.ndarray
    n01: np.ndarray
    n_unknown: np.ndarray
    required_support: np.ndarray
    area_ci_low: np.ndarray
    area_ci_high: np.ndarray
    location_ci_low: np.ndarray
    location_ci_high: np.ndarray
    presence_ci_low: np.ndarray
    presence_ci_high: np.ndarray
    bootstrap_iterations: int
    random_seed: int


@dataclass(frozen=True)
class PerPeakAnalysis:
    """High-level product returned by :func:`analyze_per_peak`."""

    consensus_by_pressure: Mapping[float, tuple[PressureConsensus, ...]]
    tracks: tuple[RadialTrack, ...]
    assignments: Mapping[str, Mapping[tuple[str, float], AssignedObservation]]
    correlations: Mapping[str, CorrelationMatrices]
    near_far: Mapping[str, Mapping[str, Mapping[str, float | int | str]]]


def clean_xy(two_theta: Sequence[float], intensity: Sequence[float]) -> CleanXY:
    """Check and deterministically clean one XY pattern.

    Non-finite pairs are removed and reported.  Points are stably sorted by
    two-theta, and duplicate coordinates are merged by their arithmetic mean.
    The output is always strictly increasing.  Cleanup is audited rather than
    silently changing the scientific thresholds.
    """

    x0 = np.asarray(two_theta, dtype=float).reshape(-1)
    y0 = np.asarray(intensity, dtype=float).reshape(-1)
    if x0.size != y0.size:
        raise ValueError("two_theta and intensity must have the same length")
    if x0.size < 5:
        raise ValueError("An XY pattern needs at least five points")

    finite = np.isfinite(x0) & np.isfinite(y0)
    finite_removed = int(x0.size - np.count_nonzero(finite))
    x = x0[finite]
    y = y0[finite]
    if x.size < 5:
        raise ValueError("Fewer than five finite XY pairs remain")

    originally_increasing = bool(finite_removed == 0 and np.all(np.diff(x) > 0))
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    unique_x, first, counts = np.unique(x, return_index=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.add.reduceat(y, first)
        y = sums / counts
        x = unique_x
    duplicates = int(np.sum(counts - 1))
    if x.size < 5 or not np.all(np.diff(x) > 0):
        raise ValueError("XY cleanup did not produce a valid increasing grid")

    differences = np.diff(x)
    dx = float(np.median(differences))
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("Cannot determine a positive XY sampling interval")

    warnings: list[str] = []
    if finite_removed:
        warnings.append(f"removed_{finite_removed}_nonfinite_pairs")
    if not originally_increasing:
        warnings.append("input_not_strictly_increasing")
    if duplicates:
        warnings.append(f"merged_{duplicates}_duplicate_points")
    if np.max(np.abs(differences - dx)) > max(1.0e-10, 0.05 * dx):
        warnings.append("irregular_sampling")

    return CleanXY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        dx=dx,
        original_count=int(x0.size),
        finite_removed=finite_removed,
        duplicate_points_merged=duplicates,
        originally_strictly_increasing=originally_increasing,
        warnings=tuple(warnings),
    )


def asls_baseline(
    x: Sequence[float],
    y: Sequence[float],
    *,
    p: float = 0.01,
    n_iter: int = 10,
    minimum_span_deg: float = 0.5,
    minimum_span_bins: int = 20,
) -> np.ndarray:
    """Return the asymmetric least-squares baseline using the frozen lambda.

    ``lambda = [max(0.5 degree, 20*dx)/dx]^4`` by default.
    """

    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if xv.size != yv.size or xv.size < 5 or not np.all(np.diff(xv) > 0):
        raise ValueError("asls_baseline requires equal, increasing arrays of length >= 5")
    if not np.all(np.isfinite(yv)):
        raise ValueError("asls_baseline requires finite intensity values")
    if not (0 < p < 1) or n_iter < 1:
        raise ValueError("AsLS p and iteration count are invalid")

    dx = float(np.median(np.diff(xv)))
    span = max(float(minimum_span_deg), float(minimum_span_bins) * dx)
    lam = (span / dx) ** 4
    n = yv.size
    second_difference = diags(
        (np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)),
        (0, 1, 2),
        shape=(n - 2, n),
        format="csc",
    )
    penalty = lam * (second_difference.T @ second_difference)
    weights = np.ones(n, dtype=float)
    baseline = np.zeros(n, dtype=float)
    for _ in range(int(n_iter)):
        system = diags(weights, 0, shape=(n, n), format="csc") + penalty
        baseline = np.asarray(spsolve(system, weights * yv), dtype=float)
        weights = np.where(yv > baseline, p, 1.0 - p)
    if not np.all(np.isfinite(baseline)):
        raise ValueError("AsLS baseline solve produced non-finite values")
    return baseline


def robust_noise(residual: Sequence[float]) -> float:
    """Frozen robust noise estimator: ``1.4826*MAD(diff)/sqrt(2)``."""

    values = np.asarray(residual, dtype=float).reshape(-1)
    differences = np.diff(values[np.isfinite(values)])
    if differences.size == 0:
        return math.nan
    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    return float(1.4826 * mad / math.sqrt(2.0))


def preprocess_pattern(
    two_theta: Sequence[float],
    intensity: Sequence[float],
    config: UniformPeakConfig,
) -> PreprocessedXY:
    """Clean, background-correct, and noise-characterize one pattern."""

    cleaned = clean_xy(two_theta, intensity)
    baseline = asls_baseline(
        cleaned.x,
        cleaned.y,
        p=config.asls_p,
        n_iter=config.asls_iterations,
        minimum_span_deg=config.asls_min_span_deg,
        minimum_span_bins=config.asls_min_span_bins,
    )
    residual = cleaned.y - baseline
    positive = np.maximum(residual, 0.0)
    smooth = gaussian_filter1d(residual, sigma=config.gaussian_sigma_bins, mode="nearest")
    noise = robust_noise(residual)
    # ``numpy.trapezoid`` was introduced after ``numpy.trapz``; support both
    # the frozen analysis environment and lightweight validation environments.
    if hasattr(np, "trapezoid"):
        total_area = float(np.trapezoid(positive, cleaned.x))
    else:  # pragma: no cover - exercised only by older NumPy releases
        total_area = float(np.trapz(positive, cleaned.x))
    warnings = list(cleaned.warnings)
    valid = bool(
        np.all(np.isfinite(residual))
        and np.isfinite(noise)
        and noise > 0
        and np.isfinite(total_area)
        and total_area > 0
    )
    if not np.isfinite(noise) or noise <= 0:
        warnings.append("nonpositive_noise")
    if not np.isfinite(total_area) or total_area <= 0:
        warnings.append("nonpositive_total_positive_area")
    return PreprocessedXY(
        x=cleaned.x,
        y=cleaned.y,
        baseline=baseline,
        residual=residual,
        positive_residual=positive,
        smoothed_residual=smooth,
        noise=float(noise),
        dx=cleaned.dx,
        total_positive_area=total_area,
        cleaned=cleaned,
        valid=valid,
        warnings=tuple(warnings),
    )


# Backward-friendly descriptive alias used by validation code.
preprocess_xy = preprocess_pattern


def two_theta_to_q(two_theta: Sequence[float] | float, wavelength: float) -> np.ndarray:
    """Convert two-theta degrees to scattering vector magnitude in inverse A."""

    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be finite and positive")
    values = np.asarray(two_theta, dtype=float)
    return 4.0 * np.pi * np.sin(np.radians(values / 2.0)) / wavelength


def q_to_two_theta(q: Sequence[float] | float, wavelength: float) -> np.ndarray:
    """Convert scattering vector magnitude in inverse A to two-theta degrees."""

    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be finite and positive")
    values = np.asarray(q, dtype=float)
    argument = values * wavelength / (4.0 * np.pi)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(argument) & (np.abs(argument) <= 1.0)
    out[valid] = np.degrees(2.0 * np.arcsin(argument[valid]))
    return out


def _pseudo_voigt_unit_area(x: np.ndarray, center: float, fwhm: float, eta: float) -> np.ndarray:
    fwhm = max(float(fwhm), np.finfo(float).tiny)
    sigma = fwhm * _FWHM_TO_SIGMA
    gamma = 0.5 * fwhm
    gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    lorentzian = gamma / (math.pi * ((x - center) ** 2 + gamma**2))
    return (1.0 - eta) * gaussian + eta * lorentzian


def _group_model(x: np.ndarray, parameters: np.ndarray, n_peaks: int) -> np.ndarray:
    model = parameters[0] + parameters[1] * (x - float(np.mean(x)))
    for index in range(n_peaks):
        offset = 2 + 4 * index
        area, center, fwhm, eta = parameters[offset : offset + 4]
        model = model + area * _pseudo_voigt_unit_area(x, center, fwhm, eta)
    return model


def _bic(residual: np.ndarray, parameter_count: int) -> float:
    n = residual.size
    rss = max(float(np.dot(residual, residual)), np.finfo(float).tiny)
    return float(n * math.log(rss / n) + parameter_count * math.log(n))


def _parameter_at_boundary(
    value: float,
    lower: float,
    upper: float,
    relative_tolerance: float,
) -> bool:
    if not np.isfinite(lower) and not np.isfinite(upper):
        return False
    scale = max(1.0, abs(value))
    if np.isfinite(lower) and np.isfinite(upper):
        scale = max(scale, upper - lower)
    return bool(
        (np.isfinite(lower) and value - lower <= relative_tolerance * scale)
        or (np.isfinite(upper) and upper - value <= relative_tolerance * scale)
    )


def _fit_peak_group(
    preprocessed: PreprocessedXY,
    candidate_indices: np.ndarray,
    candidate_width_bins: np.ndarray,
    group_id: int,
    first_peak_id: int,
    config: UniformPeakConfig,
) -> list[PeakFit]:
    """Jointly fit one overlapping candidate group and retain all audit rows."""

    x_all = preprocessed.x
    y_all = preprocessed.residual
    dx = preprocessed.dx
    centers0 = x_all[candidate_indices]
    widths0 = np.maximum(candidate_width_bins * dx, config.minimum_width_bins * dx)
    n_peaks = len(candidate_indices)

    left = max(0, int(candidate_indices[0] - math.ceil(4.0 * candidate_width_bins[0]) - 3))
    right = min(
        x_all.size,
        int(candidate_indices[-1] + math.ceil(4.0 * candidate_width_bins[-1]) + 4),
    )
    minimum_points = max(12, 5 * n_peaks + 3)
    if right - left < minimum_points:
        deficit = minimum_points - (right - left)
        left = max(0, left - deficit // 2 - 1)
        right = min(x_all.size, right + deficit - deficit // 2 + 1)
    x = x_all[left:right]
    y = y_all[left:right]
    if x.size < minimum_points:
        return [
            PeakFit(
                peak_id=first_peak_id + index,
                candidate_index=int(candidate),
                state="unknown",
                reason="insufficient_fit_points",
                two_theta=float(x_all[candidate]),
                q=float(two_theta_to_q(x_all[candidate], config.wavelength)),
                fwhm_two_theta=float(widths0[index]),
                fwhm_q=float(
                    abs(
                        two_theta_to_q(x_all[candidate] + widths0[index] / 2, config.wavelength)
                        - two_theta_to_q(x_all[candidate] - widths0[index] / 2, config.wavelength)
                    )
                ),
                area=math.nan,
                area_se=math.nan,
                relative_area=math.nan,
                eta=math.nan,
                height_snr=float(preprocessed.smoothed_residual[candidate] / preprocessed.noise),
                delta_bic=math.nan,
                fit_success=False,
                at_parameter_boundary=False,
                group_id=group_id,
            )
            for index, candidate in enumerate(candidate_indices)
        ]

    x_mean = float(np.mean(x))
    edge_count = max(2, min(5, x.size // 4))
    edge_x = np.concatenate([x[:edge_count], x[-edge_count:]])
    edge_y = np.concatenate([y[:edge_count], y[-edge_count:]])
    slope0, intercept_at_zero = np.polyfit(edge_x - x_mean, edge_y, 1)
    background0 = np.array([intercept_at_zero, slope0], dtype=float)
    data_scale = max(float(np.ptp(y)), preprocessed.noise, np.finfo(float).eps)
    span = float(x[-1] - x[0])

    initial: list[float] = [background0[0], background0[1]]
    lower: list[float] = [-np.inf, -np.inf]
    upper: list[float] = [np.inf, np.inf]
    for index, (candidate, center, width) in enumerate(zip(candidate_indices, centers0, widths0)):
        if index == 0:
            center_lower = max(float(x[0]), center - max(2.0 * width, 3.0 * dx))
        else:
            center_lower = 0.5 * (centers0[index - 1] + center)
        if index == n_peaks - 1:
            center_upper = min(float(x[-1]), center + max(2.0 * width, 3.0 * dx))
        else:
            center_upper = 0.5 * (center + centers0[index + 1])
        if center_upper - center_lower <= 2.0e-8:
            center_lower = center - dx
            center_upper = center + dx
        fwhm_lower = config.minimum_width_bins * dx
        fwhm_upper = max(fwhm_lower * 1.01, min(span / 2.0, 8.0 * width))
        peak_height = max(float(preprocessed.smoothed_residual[candidate]), preprocessed.noise)
        area0 = max(peak_height * width, np.finfo(float).eps)
        area_upper = max(10.0 * data_scale * span, 10.0 * area0)
        initial.extend([area0, float(center), float(np.clip(width, fwhm_lower * 1.001, fwhm_upper * 0.999)), 0.5])
        lower.extend([0.0, center_lower, fwhm_lower, 0.0])
        upper.extend([area_upper, center_upper, fwhm_upper, 1.0])

    initial_array = np.asarray(initial, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    f_scale = max(preprocessed.noise, np.finfo(float).eps)

    fixed_eta: dict[int, float] = {}
    fit_parameters: np.ndarray | None = None
    try:
        null_result = least_squares(
            lambda parameters: parameters[0] + parameters[1] * (x - x_mean) - y,
            background0,
            loss=config.fit_loss,
            f_scale=f_scale,
            max_nfev=config.fit_max_nfev,
        )
        fit_result = least_squares(
            lambda parameters: _group_model(x, parameters, n_peaks) - y,
            initial_array,
            bounds=(lower_array, upper_array),
            loss=config.fit_loss,
            f_scale=f_scale,
            max_nfev=config.fit_max_nfev,
        )
        fit_parameters = np.asarray(fit_result.x, dtype=float).copy()

        # Eta=0 and eta=1 are legitimate pure Gaussian/Lorentzian models, but
        # accepting a constrained optimizer sitting on a bound would violate
        # the frozen boundary rule.  Promote each endpoint to a reduced model,
        # refit with eta fixed, and recompute covariance/BIC with the smaller
        # parameter count.  Repeat because fixing one component can move a
        # second component onto a physical endpoint.
        for _ in range(n_peaks):
            new_endpoint = False
            for peak_index in range(n_peaks):
                eta_index = 2 + 4 * peak_index + 3
                if peak_index in fixed_eta:
                    continue
                eta_value = float(fit_parameters[eta_index])
                if _parameter_at_boundary(
                    eta_value,
                    float(lower_array[eta_index]),
                    float(upper_array[eta_index]),
                    config.boundary_relative_tolerance,
                ):
                    fixed_eta[peak_index] = 0.0 if eta_value < 0.5 else 1.0
                    fit_parameters[eta_index] = fixed_eta[peak_index]
                    new_endpoint = True
            if not new_endpoint:
                break

            fixed_indices = {2 + 4 * peak_index + 3 for peak_index in fixed_eta}
            free_indices = np.asarray(
                [index for index in range(fit_parameters.size) if index not in fixed_indices],
                dtype=int,
            )
            template = fit_parameters.copy()

            def expand_reduced(reduced: np.ndarray) -> np.ndarray:
                expanded = template.copy()
                expanded[free_indices] = reduced
                for peak_index, endpoint in fixed_eta.items():
                    expanded[2 + 4 * peak_index + 3] = endpoint
                return expanded

            fit_result = least_squares(
                lambda reduced: _group_model(x, expand_reduced(reduced), n_peaks) - y,
                fit_parameters[free_indices],
                bounds=(lower_array[free_indices], upper_array[free_indices]),
                loss=config.fit_loss,
                f_scale=f_scale,
                max_nfev=config.fit_max_nfev,
            )
            fit_parameters = expand_reduced(fit_result.x)

        fixed_indices = {2 + 4 * peak_index + 3 for peak_index in fixed_eta}
        free_indices = np.asarray(
            [index for index in range(fit_parameters.size) if index not in fixed_indices],
            dtype=int,
        )
        fit_residual = _group_model(x, fit_parameters, n_peaks) - y
        null_residual = null_result.x[0] + null_result.x[1] * (x - x_mean) - y
        parameter_count = 2 + 4 * n_peaks - len(fixed_eta)
        delta_bic = _bic(null_residual, 2) - _bic(fit_residual, parameter_count)
        dof = max(1, x.size - parameter_count)
        covariance = np.linalg.pinv(fit_result.jac.T @ fit_result.jac)
        covariance *= float(np.dot(fit_residual, fit_residual)) / dof
        reduced_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        standard_errors = np.full(fit_parameters.size, np.nan)
        standard_errors[free_indices] = reduced_errors
        standard_errors[list(fixed_indices)] = 0.0
        fit_success = bool(fit_result.success and np.all(np.isfinite(fit_parameters)))
    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
        fit_result = None
        fit_parameters = None
        standard_errors = np.full(len(initial), np.nan)
        delta_bic = math.nan
        fit_success = False

    results: list[PeakFit] = []
    for index, candidate in enumerate(candidate_indices):
        offset = 2 + 4 * index
        if fit_parameters is None:
            area = center = fwhm = eta = area_se = math.nan
            boundary = False
            fit_model = "fit_failed"
        else:
            area, center, fwhm, eta = map(float, fit_parameters[offset : offset + 4])
            area_se = float(standard_errors[offset])
            # A fixed endpoint is now a reduced model, not a free parameter at
            # a bound.  Area, center, and FWHM must still remain interior.
            boundary_indices = list(range(offset, offset + 3))
            if index not in fixed_eta:
                boundary_indices.append(offset + 3)
            boundary = any(
                _parameter_at_boundary(
                    float(fit_parameters[param_index]),
                    float(lower_array[param_index]),
                    float(upper_array[param_index]),
                    config.boundary_relative_tolerance,
                )
                for param_index in boundary_indices
            )
            if fixed_eta.get(index) == 0.0:
                fit_model = "gaussian_endpoint"
            elif fixed_eta.get(index) == 1.0:
                fit_model = "lorentzian_endpoint"
            else:
                fit_model = "pseudo_voigt"

        area_over_se = area / area_se if np.isfinite(area_se) and area_se > 0 else math.nan
        reliable = bool(
            fit_success
            and np.isfinite(delta_bic)
            and delta_bic >= config.delta_bic_minimum
            and np.isfinite(area_over_se)
            and area_over_se >= config.area_over_se_minimum
            and (not config.reject_parameter_bound_hits or not boundary)
            and np.isfinite(center)
            and np.isfinite(fwhm)
            and fwhm > 0
            and np.isfinite(area)
            and area > 0
            and preprocessed.total_positive_area > 0
        )
        if reliable:
            state = "reliable"
            reason = "passed_all_fit_checks"
        elif not fit_success:
            state, reason = "unknown", "fit_failed"
        elif not np.isfinite(delta_bic) or delta_bic < config.delta_bic_minimum:
            state, reason = "unknown", "delta_bic_below_threshold"
        elif not np.isfinite(area_over_se) or area_over_se < config.area_over_se_minimum:
            state, reason = "unknown", "area_over_se_below_threshold"
        elif boundary and config.reject_parameter_bound_hits:
            state, reason = "unknown", "parameter_at_boundary"
        else:
            state, reason = "unknown", "invalid_fit_parameter"

        if np.isfinite(center) and np.isfinite(fwhm):
            q_value = float(two_theta_to_q(center, config.wavelength))
            fwhm_q = float(
                abs(
                    two_theta_to_q(center + fwhm / 2.0, config.wavelength)
                    - two_theta_to_q(center - fwhm / 2.0, config.wavelength)
                )
            )
        else:
            center = float(x_all[candidate])
            fwhm = float(widths0[index])
            q_value = float(two_theta_to_q(center, config.wavelength))
            fwhm_q = float(
                abs(
                    two_theta_to_q(center + fwhm / 2.0, config.wavelength)
                    - two_theta_to_q(center - fwhm / 2.0, config.wavelength)
                )
            )
        relative_area = (
            area / preprocessed.total_positive_area
            if np.isfinite(area) and preprocessed.total_positive_area > 0
            else math.nan
        )
        results.append(
            PeakFit(
                peak_id=first_peak_id + index,
                candidate_index=int(candidate),
                state=state,
                reason=reason,
                two_theta=float(center),
                q=q_value,
                fwhm_two_theta=float(fwhm),
                fwhm_q=fwhm_q,
                area=float(area),
                area_se=float(area_se),
                relative_area=float(relative_area),
                eta=float(eta),
                height_snr=float(preprocessed.smoothed_residual[candidate] / preprocessed.noise),
                delta_bic=float(delta_bic),
                fit_success=fit_success,
                at_parameter_boundary=boundary,
                group_id=group_id,
                fit_model=fit_model,
            )
        )
    return results


def detect_pattern_peaks(
    preprocessed: PreprocessedXY,
    *,
    frame: int,
    scan: str,
    pressure: float,
    channel: str,
    config: UniformPeakConfig,
) -> FramePeaks:
    """Discover and robustly fit all candidates in one preprocessed pattern."""

    measured_q = two_theta_to_q(
        np.asarray([preprocessed.x[0], preprocessed.x[-1]], dtype=float),
        config.wavelength,
    )
    measured_q_min = float(np.min(measured_q))
    measured_q_max = float(np.max(measured_q))

    if not preprocessed.valid:
        return FramePeaks(
            frame=int(frame),
            scan=str(scan),
            pressure=float(pressure),
            channel=str(channel),
            peaks=(),
            pattern_valid=False,
            noise=float(preprocessed.noise),
            total_positive_area=float(preprocessed.total_positive_area),
            measured_q_min=measured_q_min,
            measured_q_max=measured_q_max,
            warnings=preprocessed.warnings + ("peak_detection_skipped_invalid_pattern",),
        )

    candidates, properties = find_peaks(
        preprocessed.smoothed_residual,
        prominence=config.prominence_noise_factor * preprocessed.noise,
        height=config.height_noise_factor * preprocessed.noise,
        width=config.minimum_width_bins,
    )
    if not len(candidates):
        return FramePeaks(
            frame=int(frame),
            scan=str(scan),
            pressure=float(pressure),
            channel=str(channel),
            peaks=(),
            pattern_valid=True,
            noise=float(preprocessed.noise),
            total_positive_area=float(preprocessed.total_positive_area),
            measured_q_min=measured_q_min,
            measured_q_max=measured_q_max,
            warnings=preprocessed.warnings,
        )

    widths = np.asarray(properties["widths"], dtype=float)
    if config.overlap_grouping:
        groups: list[list[int]] = [[0]]
        for index in range(1, len(candidates)):
            previous = groups[-1][-1]
            separation = candidates[index] - candidates[previous]
            # Candidates closer than the sum of their estimated FWHMs have
            # materially overlapping fitting support and must be solved jointly.
            overlap_distance = widths[index] + widths[previous]
            if separation <= overlap_distance:
                groups[-1].append(index)
            else:
                groups.append([index])
    else:
        groups = [[index] for index in range(len(candidates))]

    fits: list[PeakFit] = []
    next_peak_id = 0
    for group_id, group_indices in enumerate(groups):
        selection = np.asarray(group_indices, dtype=int)
        group_fits = _fit_peak_group(
            preprocessed,
            candidates[selection],
            widths[selection],
            group_id,
            next_peak_id,
            config,
        )
        fits.extend(group_fits)
        next_peak_id += len(group_fits)
    fits.sort(key=lambda item: (item.two_theta, item.peak_id))
    # Re-number after sorting so IDs are deterministic in radial order.
    fits = [PeakFit(**{**fit.__dict__, "peak_id": index}) for index, fit in enumerate(fits)]
    return FramePeaks(
        frame=int(frame),
        scan=str(scan),
        pressure=float(pressure),
        channel=str(channel),
        peaks=tuple(fits),
        pattern_valid=True,
        noise=float(preprocessed.noise),
        total_positive_area=float(preprocessed.total_positive_area),
        measured_q_min=measured_q_min,
        measured_q_max=measured_q_max,
        warnings=preprocessed.warnings,
    )


# Short alias retained for callers that prefer the generic name.
detect_peaks = detect_pattern_peaks


def minimum_scan_support(n_scans: int) -> int:
    """Frozen scan support ``min[N, max(5, ceil(0.1*N))]``."""

    n = int(n_scans)
    if n <= 0:
        return 0
    return min(n, max(5, int(math.ceil(0.1 * n))))


def minimum_pressure_support(n_pressure_levels: int) -> int:
    """Frozen pressure support ``min[M, max(3, ceil(0.1*M))]``."""

    n = int(n_pressure_levels)
    if n <= 0:
        return 0
    return min(n, max(3, int(math.ceil(0.1 * n))))


def _peaks_compatible(first: PeakFit | PressureConsensus, second: PeakFit | PressureConsensus, factor: float) -> bool:
    if not all(np.isfinite(value) and value > 0 for value in (first.fwhm_q, second.fwhm_q)):
        return False
    return bool(abs(first.q - second.q) <= factor * (first.fwhm_q + second.fwhm_q))


def build_pressure_consensus(
    frame_peaks: Sequence[FramePeaks],
    all_scans: Sequence[str],
    pressure_levels: Sequence[float],
    config: UniformPeakConfig,
) -> dict[float, tuple[PressureConsensus, ...]]:
    """Build deterministic one-dimensional complete-link consensus clusters.

    Every member pair in a cluster satisfies
    ``|dq| <= 0.5*(FWHMq_i + FWHMq_j)`` (using the configured frozen factor),
    and no cluster contains two observations from the same scan.  The greedy
    radial ordering is deterministic and enforces the complete-link invariant.
    """

    scans = tuple(sorted({str(item) for item in all_scans}))
    pressure_tuple = tuple(float(value) for value in pressure_levels)
    channels = {frame.channel for frame in frame_peaks}
    if len(channels) > 1:
        raise ValueError("build_pressure_consensus must be called separately per channel")
    channel = next(iter(channels), "unknown")
    pressure_index = {value: index for index, value in enumerate(pressure_tuple)}
    frames_by_pressure: dict[float, list[tuple[FramePeaks, PeakFit]]] = {
        value: [] for value in pressure_tuple
    }
    for frame in frame_peaks:
        if frame.pressure not in pressure_index:
            continue
        for peak in frame.peaks:
            if peak.reliable:
                frames_by_pressure[frame.pressure].append((frame, peak))

    result: dict[float, tuple[PressureConsensus, ...]] = {}
    required = minimum_scan_support(len(scans))
    for pressure in pressure_tuple:
        observations = sorted(
            frames_by_pressure[pressure],
            key=lambda item: (item[1].q, item[0].scan, item[0].frame, item[1].peak_id),
        )
        clusters: list[list[tuple[FramePeaks, PeakFit]]] = []
        for observation in observations:
            frame, peak = observation
            compatible_clusters: list[tuple[float, int]] = []
            for cluster_index, cluster in enumerate(clusters):
                if any(existing_frame.scan == frame.scan for existing_frame, _ in cluster):
                    continue
                if all(
                    _peaks_compatible(peak, existing_peak, config.consensus_fwhm_factor)
                    for _, existing_peak in cluster
                ):
                    median_q = float(np.median([existing_peak.q for _, existing_peak in cluster]))
                    compatible_clusters.append((abs(peak.q - median_q), cluster_index))
            if compatible_clusters:
                _, selected = min(compatible_clusters, key=lambda item: (item[0], item[1]))
                clusters[selected].append(observation)
            else:
                clusters.append([observation])

        consensus: list[PressureConsensus] = []
        clusters.sort(key=lambda cluster: float(np.median([peak.q for _, peak in cluster])))
        for cluster_index, cluster in enumerate(clusters):
            q_values = np.array([peak.q for _, peak in cluster], dtype=float)
            fwhm_values = np.array([peak.fwhm_q for _, peak in cluster], dtype=float)
            area_values = np.array([peak.relative_area for _, peak in cluster], dtype=float)
            member_keys = tuple(
                sorted((frame.scan, frame.frame, peak.peak_id) for frame, peak in cluster)
            )
            support = len({frame.scan for frame, _ in cluster})
            consensus.append(
                PressureConsensus(
                    consensus_id=f"{channel}|p{pressure_index[pressure]:03d}|c{cluster_index:04d}",
                    channel=channel,
                    pressure=pressure,
                    pressure_index=pressure_index[pressure],
                    q=float(np.median(q_values)),
                    fwhm_q=float(np.median(fwhm_values)),
                    relative_area=float(np.median(area_values)),
                    support=support,
                    total_scans=len(scans),
                    required_support=required,
                    member_keys=member_keys,
                    reliable=bool(support >= required),
                    ambiguous=False,
                )
            )
        result[pressure] = tuple(consensus)
    return result


def _consensus_to_node(consensus: PressureConsensus) -> TrackNode:
    return TrackNode(
        consensus_id=consensus.consensus_id,
        pressure=consensus.pressure,
        pressure_index=consensus.pressure_index,
        q=consensus.q,
        fwhm_q=consensus.fwhm_q,
        relative_area=consensus.relative_area,
        support=consensus.support,
        ambiguous=consensus.ambiguous,
    )


def _trajectory_prediction(nodes: Sequence[TrackNode], target_pressure: float) -> tuple[float, float]:
    last = nodes[-1]
    if len(nodes) < 2:
        return last.q, last.fwhm_q
    previous = nodes[-2]
    pressure_delta = last.pressure - previous.pressure
    if abs(pressure_delta) <= np.finfo(float).eps:
        return last.q, last.fwhm_q
    velocity = (last.q - previous.q) / pressure_delta
    return last.q + velocity * (target_pressure - last.pressure), last.fwhm_q


def _forward_trajectory_links(
    consensus_by_pressure: Mapping[float, Sequence[PressureConsensus]],
    pressure_levels: Sequence[float],
    config: UniformPeakConfig,
    reverse: bool,
) -> tuple[set[tuple[str, str]], set[str], set[str]]:
    ordered = list(enumerate(float(value) for value in pressure_levels))
    if reverse:
        ordered.reverse()
    sorted_pressures = np.sort(np.asarray(pressure_levels, dtype=float))
    positive_steps = np.diff(sorted_pressures)
    positive_steps = positive_steps[positive_steps > 0]
    median_pressure_step = float(np.median(positive_steps)) if positive_steps.size else 1.0
    paths: list[list[TrackNode]] = []
    links: set[tuple[str, str]] = set()
    touched: set[str] = set()
    ambiguous_nodes: set[str] = set()

    for level_index, pressure in ordered:
        nodes = [
            _consensus_to_node(item)
            for item in consensus_by_pressure.get(pressure, ())
            if item.reliable
        ]
        nodes.sort(key=lambda item: (item.q, item.consensus_id))
        touched.update(node.consensus_id for node in nodes)
        active: list[list[TrackNode]] = []
        for path in paths:
            missing = abs(level_index - path[-1].pressure_index) - 1
            if missing <= config.max_missing_pressure_levels:
                active.append(path)

        if not active:
            paths.extend([[node] for node in nodes])
            continue
        n_path = len(active)
        n_node = len(nodes)
        size = n_path + n_node
        cost = np.full((size, size), 1.0e6, dtype=float)
        # A gated real pair is preferable to an unmatched dummy.  The gate,
        # rather than a hidden absolute q cutoff, decides admissibility.
        cost[:n_path, n_node:] = 10.0
        cost[n_path:, :n_node] = 10.0
        cost[n_path:, n_node:] = 0.0
        for path_index, path in enumerate(active):
            prediction, predicted_width = _trajectory_prediction(path, pressure)
            pressure_gap_factor = max(
                1.0,
                abs(pressure - path[-1].pressure) / median_pressure_step,
            )
            for node_index, node in enumerate(nodes):
                gate = (
                    config.track_gate_factor
                    * math.sqrt(predicted_width**2 + node.fwhm_q**2)
                    * pressure_gap_factor
                )
                if np.isfinite(gate) and gate > 0:
                    normalized_dq = abs(node.q - prediction) / gate
                    if normalized_dq <= 1.0 and predicted_width > 0 and node.fwhm_q > 0:
                        width_term = math.log(node.fwhm_q / predicted_width)
                        cost[path_index, node_index] = (
                            normalized_dq**2
                            + config.track_width_cost_weight * width_term**2
                        )

        # A small winner/runner-up margin means the identity is not uniquely
        # supported.  Flag all involved nodes instead of silently swapping IDs.
        real_cost = cost[:n_path, :n_node]
        for path_index, path in enumerate(active):
            finite_columns = np.flatnonzero(real_cost[path_index] < 1.0e5)
            if finite_columns.size >= 2:
                ordered_columns = finite_columns[
                    np.argsort(real_cost[path_index, finite_columns], kind="stable")
                ]
                if (
                    real_cost[path_index, ordered_columns[1]]
                    - real_cost[path_index, ordered_columns[0]]
                    < config.ambiguous_cost_margin
                ):
                    ambiguous_nodes.add(path[-1].consensus_id)
                    ambiguous_nodes.update(nodes[index].consensus_id for index in ordered_columns[:2])
        for node_index, node in enumerate(nodes):
            finite_rows = np.flatnonzero(real_cost[:, node_index] < 1.0e5)
            if finite_rows.size >= 2:
                ordered_rows = finite_rows[
                    np.argsort(real_cost[finite_rows, node_index], kind="stable")
                ]
                if (
                    real_cost[ordered_rows[1], node_index]
                    - real_cost[ordered_rows[0], node_index]
                    < config.ambiguous_cost_margin
                ):
                    ambiguous_nodes.add(node.consensus_id)
                    ambiguous_nodes.update(active[index][-1].consensus_id for index in ordered_rows[:2])
        row_indices, column_indices = linear_sum_assignment(cost)
        matched_nodes: set[int] = set()
        for row, column in zip(row_indices, column_indices):
            if row < n_path and column < n_node and cost[row, column] < 10.0:
                path = active[row]
                previous = path[-1]
                node = nodes[column]
                path.append(node)
                links.add(tuple(sorted((previous.consensus_id, node.consensus_id))))
                matched_nodes.add(column)
        paths.extend([[node] for index, node in enumerate(nodes) if index not in matched_nodes])
        # Remove duplicate references introduced by ``paths.extend`` on levels
        # where active paths were already in paths (only new singleton paths are
        # actually added above).
        seen_ids: set[int] = set()
        deduplicated: list[list[TrackNode]] = []
        for path in paths:
            identity = id(path)
            if identity not in seen_ids:
                seen_ids.add(identity)
                deduplicated.append(path)
        paths = deduplicated
    return links, touched, ambiguous_nodes


def link_consensus_bidirectional(
    consensus_by_pressure: Mapping[float, Sequence[PressureConsensus]],
    pressure_levels: Sequence[float],
    config: UniformPeakConfig,
) -> tuple[RadialTrack, ...]:
    """Link pressure consensuses in both directions and keep mutual edges only."""

    pressures = tuple(float(value) for value in pressure_levels)
    if len(set(pressures)) != len(pressures):
        raise ValueError("pressure_levels must be unique")
    forward, forward_nodes, forward_ambiguous = _forward_trajectory_links(
        consensus_by_pressure, pressures, config, reverse=False
    )
    backward, backward_nodes, backward_ambiguous = _forward_trajectory_links(
        consensus_by_pressure, pressures, config, reverse=True
    )
    consistent_links = forward & backward
    inconsistent_nodes = {
        node for edge in forward ^ backward for node in edge
    } | forward_ambiguous | backward_ambiguous
    all_consensus = {
        item.consensus_id: item
        for pressure in pressures
        for item in consensus_by_pressure.get(pressure, ())
        if item.reliable and item.consensus_id in (forward_nodes | backward_nodes)
    }

    adjacency: dict[str, set[str]] = {identifier: set() for identifier in all_consensus}
    for first, second in consistent_links:
        adjacency[first].add(second)
        adjacency[second].add(first)
    components: list[list[str]] = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        stack = [start]
        component: list[str] = []
        remaining.remove(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(component)

    component_nodes: list[tuple[list[TrackNode], bool]] = []
    for identifiers in components:
        nodes = sorted(
            (_consensus_to_node(all_consensus[identifier]) for identifier in identifiers),
            key=lambda node: (node.pressure_index, node.q, node.consensus_id),
        )
        ambiguous = any(
            identifier in inconsistent_nodes or all_consensus[identifier].ambiguous
            for identifier in identifiers
        )
        component_nodes.append((nodes, ambiguous))
    component_nodes.sort(
        key=lambda item: (
            float(np.median([node.q for node in item[0]])),
            item[0][0].pressure_index,
            item[0][0].consensus_id,
        )
    )

    minimum_levels = minimum_pressure_support(len(pressures))
    tracks: list[RadialTrack] = []
    for index, (nodes, ambiguous) in enumerate(component_nodes, start=1):
        channel = all_consensus[nodes[0].consensus_id].channel
        n_levels = len({node.pressure_index for node in nodes})
        tracks.append(
            RadialTrack(
                track_id=f"radial_peak_{index:03d}",
                channel=channel,
                nodes=tuple(nodes),
                official=bool(n_levels >= minimum_levels and not ambiguous),
                ambiguous=ambiguous,
                minimum_pressure_support=minimum_levels,
            )
        )
    return tuple(tracks)


def _track_target(track: RadialTrack, pressure: float) -> tuple[float, float] | None:
    """Exact or linearly interpolated target; never extrapolate."""

    nodes = sorted(track.nodes, key=lambda node: node.pressure)
    if pressure < nodes[0].pressure or pressure > nodes[-1].pressure:
        return None
    for node in nodes:
        if pressure == node.pressure:
            return node.q, node.fwhm_q
    lower = [node for node in nodes if node.pressure < pressure]
    upper = [node for node in nodes if node.pressure > pressure]
    if not lower or not upper:
        return None
    first, second = lower[-1], upper[0]
    fraction = (pressure - first.pressure) / (second.pressure - first.pressure)
    q = first.q + fraction * (second.q - first.q)
    width = first.fwhm_q + fraction * (second.fwhm_q - first.fwhm_q)
    return float(q), float(width)


def assign_track_observations(
    tracks: Sequence[RadialTrack],
    frame_peaks: Sequence[FramePeaks],
    scans: Sequence[str],
    pressure_levels: Sequence[float],
    config: UniformPeakConfig,
) -> dict[str, dict[tuple[str, float], AssignedObservation]]:
    """Assign detections once per frame with Hungarian one-to-one matching.

    A reliable :class:`PeakFit` can be ``present`` in at most one radial track
    for a given frame.  Feasible costs use only fitted widths,
    ``gate=sqrt(w_candidate^2+w_target^2)`` and ``cost=(dq/gate)^2``.  A
    best/runner-up margin below the frozen ambiguity margin makes the affected
    assignment ``unknown`` rather than silently choosing an identity.
    """

    frame_lookup: dict[tuple[str, float], list[FramePeaks]] = {}
    for frame in frame_peaks:
        frame_lookup.setdefault((frame.scan, frame.pressure), []).append(frame)
    ordered_tracks = tuple(sorted(tracks, key=lambda item: item.track_id))
    result: dict[str, dict[tuple[str, float], AssignedObservation]] = {
        track.track_id: {} for track in ordered_tracks
    }
    for scan_value in scans:
        scan = str(scan_value)
        for pressure_value in pressure_levels:
            pressure = float(pressure_value)
            key = (scan, pressure)
            targets: list[tuple[RadialTrack, float, float]] = []
            for track in ordered_tracks:
                target = _track_target(track, pressure)
                if target is None:
                    result[track.track_id][key] = AssignedObservation(
                        track_id=track.track_id,
                        scan=scan,
                        pressure=pressure,
                        frame=None,
                        state="out_of_range",
                        reason="outside_supported_track_range",
                    )
                else:
                    targets.append((track, target[0], target[1]))

            frames = frame_lookup.get(key, [])
            if len(frames) != 1 or not frames[0].pattern_valid:
                for track, _, _ in targets:
                    result[track.track_id][key] = AssignedObservation(
                        track_id=track.track_id,
                        scan=scan,
                        pressure=pressure,
                        frame=frames[0].frame if len(frames) == 1 else None,
                        state="unknown",
                        reason="missing_duplicate_or_invalid_frame",
                    )
                continue

            frame = frames[0]
            covered_targets: list[tuple[RadialTrack, float, float]] = []
            for track, target_q, target_width in targets:
                has_measured_range = np.isfinite(frame.measured_q_min) and np.isfinite(
                    frame.measured_q_max
                )
                if has_measured_range and not (
                    frame.measured_q_min - 1.0e-12
                    <= target_q
                    <= frame.measured_q_max + 1.0e-12
                ):
                    result[track.track_id][key] = AssignedObservation(
                        track_id=track.track_id,
                        scan=scan,
                        pressure=pressure,
                        frame=frame.frame,
                        state="unknown",
                        reason="outside_frame_measured_q_range",
                    )
                else:
                    covered_targets.append((track, target_q, target_width))
            targets = covered_targets
            reliable = sorted(
                (peak for peak in frame.peaks if peak.reliable),
                key=lambda peak: (peak.q, peak.peak_id),
            )
            n_track, n_peak = len(targets), len(reliable)
            real_cost = np.full((n_track, n_peak), 1.0e6, dtype=float)
            for track_index, (_, target_q, target_width) in enumerate(targets):
                for peak_index, peak in enumerate(reliable):
                    if not all(
                        np.isfinite(value) and value > 0
                        for value in (target_width, peak.fwhm_q)
                    ):
                        continue
                    gate = math.sqrt(target_width**2 + peak.fwhm_q**2)
                    normalized = abs(peak.q - target_q) / gate
                    if normalized <= 1.0:
                        real_cost[track_index, peak_index] = normalized**2

            ambiguous_tracks: set[int] = set()
            ambiguous_peaks: set[int] = set()
            for track_index in range(n_track):
                feasible = np.flatnonzero(real_cost[track_index] < 1.0e5)
                if feasible.size >= 2:
                    ordered = feasible[
                        np.argsort(real_cost[track_index, feasible], kind="stable")
                    ]
                    if (
                        real_cost[track_index, ordered[1]]
                        - real_cost[track_index, ordered[0]]
                        < config.ambiguous_cost_margin
                    ):
                        ambiguous_tracks.add(track_index)
                        ambiguous_peaks.update(int(item) for item in ordered[:2])
            for peak_index in range(n_peak):
                feasible = np.flatnonzero(real_cost[:, peak_index] < 1.0e5)
                if feasible.size >= 2:
                    ordered = feasible[
                        np.argsort(real_cost[feasible, peak_index], kind="stable")
                    ]
                    if (
                        real_cost[ordered[1], peak_index]
                        - real_cost[ordered[0], peak_index]
                        < config.ambiguous_cost_margin
                    ):
                        ambiguous_peaks.add(peak_index)
                        ambiguous_tracks.update(int(item) for item in ordered[:2])

            matches: dict[int, int] = {}
            if n_track and n_peak:
                size = n_track + n_peak
                cost = np.full((size, size), 1.0e6, dtype=float)
                cost[:n_track, :n_peak] = real_cost
                cost[:n_track, n_peak:] = 10.0
                cost[n_track:, :n_peak] = 10.0
                cost[n_track:, n_peak:] = 0.0
                rows, columns = linear_sum_assignment(cost)
                matches = {
                    int(row): int(column)
                    for row, column in zip(rows, columns)
                    if row < n_track and column < n_peak and real_cost[row, column] < 1.0e5
                }

            present_peak_keys: set[tuple[int, int]] = set()
            for track_index, (track, target_q, target_width) in enumerate(targets):
                matched_peak_index = matches.get(track_index)
                if (
                    track_index in ambiguous_tracks
                    or (
                        matched_peak_index is not None
                        and matched_peak_index in ambiguous_peaks
                    )
                ):
                    observation = AssignedObservation(
                        track_id=track.track_id,
                        scan=scan,
                        pressure=pressure,
                        frame=frame.frame,
                        state="unknown",
                        reason="ambiguous_hungarian_margin",
                    )
                elif matched_peak_index is not None:
                    peak = reliable[matched_peak_index]
                    peak_key = (frame.frame, peak.peak_id)
                    if peak_key in present_peak_keys:
                        raise AssertionError("A PeakFit was assigned present to more than one track")
                    present_peak_keys.add(peak_key)
                    observation = AssignedObservation(
                        track_id=track.track_id,
                        scan=scan,
                        pressure=pressure,
                        frame=frame.frame,
                        state="present",
                        reason="unique_hungarian_match",
                        q=peak.q,
                        fwhm_q=peak.fwhm_q,
                        relative_area=peak.relative_area,
                        peak_id=peak.peak_id,
                    )
                else:
                    unreliable_gated = False
                    for peak in frame.peaks:
                        if peak.reliable or not np.isfinite(peak.q) or not np.isfinite(peak.fwhm_q):
                            continue
                        if peak.fwhm_q <= 0 or target_width <= 0:
                            continue
                        gate = math.sqrt(target_width**2 + peak.fwhm_q**2)
                        if abs(peak.q - target_q) <= gate:
                            unreliable_gated = True
                            break
                    observation = AssignedObservation(
                        track_id=track.track_id,
                        scan=scan,
                        pressure=pressure,
                        frame=frame.frame,
                        state="unknown" if unreliable_gated else "absent",
                        reason=(
                            "compatible_candidate_failed_fit_checks"
                            if unreliable_gated
                            else "no_compatible_candidate"
                        ),
                    )
                result[track.track_id][key] = observation
    assert_unique_present_peak_assignments(result)
    return result


def assert_unique_present_peak_assignments(
    assignments_by_track: Mapping[
        str, Mapping[tuple[str, float], AssignedObservation]
    ],
) -> None:
    """Raise if one fitted frame peak is marked present in multiple tracks."""

    owner: dict[tuple[str, float, int, int], str] = {}
    for track_id, assignments in assignments_by_track.items():
        for (scan, pressure), observation in assignments.items():
            if observation.state != "present":
                continue
            if observation.frame is None or observation.peak_id is None:
                raise AssertionError("A present assignment lacks frame/peak identity")
            key = (str(scan), float(pressure), int(observation.frame), int(observation.peak_id))
            previous = owner.setdefault(key, track_id)
            if previous != track_id:
                raise AssertionError(
                    f"Peak {key} is present in both {previous!r} and {track_id!r}"
                )


def relative_area_similarity(first: float, second: float) -> float:
    """Return ``min(Ai,Aj)/max(Ai,Aj)`` for finite non-negative areas."""

    if not np.isfinite(first) or not np.isfinite(second) or first < 0 or second < 0:
        return math.nan
    maximum = max(float(first), float(second))
    if maximum <= 0:
        return math.nan
    return min(float(first), float(second)) / maximum


def location_similarity(q1: float, q2: float, fwhm_q1: float, fwhm_q2: float) -> float:
    """Width-aware Gaussian similarity for two fitted radial locations."""

    values = (q1, q2, fwhm_q1, fwhm_q2)
    if not all(np.isfinite(value) for value in values) or fwhm_q1 <= 0 or fwhm_q2 <= 0:
        return math.nan
    sigma1 = fwhm_q1 * _FWHM_TO_SIGMA
    sigma2 = fwhm_q2 * _FWHM_TO_SIGMA
    return float(math.exp(-((q1 - q2) ** 2) / (2.0 * (sigma1**2 + sigma2**2))))


def _nan_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else math.nan


def compute_track_correlations(
    assignments: Mapping[tuple[str, float], AssignedObservation],
    scans: Sequence[str],
    pressure_levels: Sequence[float],
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 0,
    ci_percentiles: tuple[float, float] = (2.5, 97.5),
) -> CorrelationMatrices:
    """Compute same-scan conditional correlations and scan-bootstrap CIs.

    ``n10[i,j]`` counts present at pressure ``i`` and absent at ``j``;
    ``n01`` has the reverse direction, so ``n10.T == n01``.  Similarity,
    presence, support, and CI matrices are symmetric.
    """

    scan_tuple = tuple(str(item) for item in scans)
    pressures = tuple(float(item) for item in pressure_levels)
    n_scan = len(scan_tuple)
    n_pressure = len(pressures)
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations cannot be negative")
    ci_low_percentile, ci_high_percentile = map(float, ci_percentiles)
    if not (0.0 <= ci_low_percentile < ci_high_percentile <= 100.0):
        raise ValueError("ci_percentiles must be ordered within [0, 100]")

    state = np.full((n_scan, n_pressure), -2, dtype=np.int8)
    area_values = np.full((n_scan, n_pressure), np.nan, dtype=float)
    q_values = np.full_like(area_values, np.nan)
    width_values = np.full_like(area_values, np.nan)
    state_code = {"out_of_range": -2, "unknown": -1, "absent": 0, "present": 1}
    for scan_index, scan in enumerate(scan_tuple):
        for pressure_index, pressure in enumerate(pressures):
            observation = assignments.get((scan, pressure))
            if observation is None:
                state[scan_index, pressure_index] = -1
                continue
            if observation.state not in state_code:
                raise ValueError(f"Unknown observation state: {observation.state}")
            state[scan_index, pressure_index] = state_code[observation.state]
            if observation.state == "present":
                area_values[scan_index, pressure_index] = observation.relative_area
                q_values[scan_index, pressure_index] = observation.q
                width_values[scan_index, pressure_index] = observation.fwhm_q

    shape = (n_pressure, n_pressure)
    area = np.full(shape, np.nan)
    location = np.full(shape, np.nan)
    presence = np.full(shape, np.nan)
    n_available = np.zeros(shape, dtype=np.int32)
    n_both = np.zeros(shape, dtype=np.int32)
    n10 = np.zeros(shape, dtype=np.int32)
    n01 = np.zeros(shape, dtype=np.int32)
    n_unknown = np.zeros(shape, dtype=np.int32)
    required = np.zeros(shape, dtype=np.int32)
    area_low = np.full(shape, np.nan)
    area_high = np.full(shape, np.nan)
    location_low = np.full(shape, np.nan)
    location_high = np.full(shape, np.nan)
    presence_low = np.full(shape, np.nan)
    presence_high = np.full(shape, np.nan)
    area_by_scan_all = np.full((n_scan, n_pressure, n_pressure), np.nan, dtype=float)
    location_by_scan_all = np.full_like(area_by_scan_all, np.nan)
    presence_by_scan_all = np.full_like(area_by_scan_all, np.nan)
    rng = np.random.default_rng(int(seed))
    bootstrap_indices = (
        rng.integers(0, n_scan, size=(bootstrap_iterations, n_scan))
        if bootstrap_iterations and n_scan
        else np.empty((0, n_scan), dtype=int)
    )

    for first in range(n_pressure):
        for second in range(first, n_pressure):
            s1 = state[:, first]
            s2 = state[:, second]
            in_range = (s1 != -2) & (s2 != -2)
            unknown = in_range & ((s1 == -1) | (s2 == -1))
            available = in_range & ~unknown
            both = available & (s1 == 1) & (s2 == 1)
            first_only = available & (s1 == 1) & (s2 == 0)
            second_only = available & (s1 == 0) & (s2 == 1)
            nav = int(np.count_nonzero(available))
            nboth = int(np.count_nonzero(both))
            n_available[first, second] = n_available[second, first] = nav
            n_both[first, second] = n_both[second, first] = nboth
            n10[first, second] = int(np.count_nonzero(first_only))
            n01[first, second] = int(np.count_nonzero(second_only))
            n10[second, first] = n01[first, second]
            n01[second, first] = n10[first, second]
            n_unknown[first, second] = n_unknown[second, first] = int(np.count_nonzero(unknown))
            support = minimum_scan_support(nav)
            required[first, second] = required[second, first] = support

            area_by_scan = np.full(n_scan, np.nan)
            location_by_scan = np.full(n_scan, np.nan)
            presence_by_scan = np.full(n_scan, np.nan)
            both_indices = np.flatnonzero(both)
            for scan_index in both_indices:
                area_by_scan[scan_index] = relative_area_similarity(
                    area_values[scan_index, first], area_values[scan_index, second]
                )
                location_by_scan[scan_index] = location_similarity(
                    q_values[scan_index, first],
                    q_values[scan_index, second],
                    width_values[scan_index, first],
                    width_values[scan_index, second],
                )
                presence_by_scan[scan_index] = 1.0
            presence_by_scan[first_only | second_only] = 0.0
            area_by_scan_all[:, first, second] = area_by_scan
            area_by_scan_all[:, second, first] = area_by_scan
            location_by_scan_all[:, first, second] = location_by_scan
            location_by_scan_all[:, second, first] = location_by_scan
            presence_by_scan_all[:, first, second] = presence_by_scan
            presence_by_scan_all[:, second, first] = presence_by_scan
            if support > 0 and nboth >= support:
                area_value = float(np.nanmedian(area_by_scan))
                location_value = float(np.nanmedian(location_by_scan))
                area[first, second] = area[second, first] = area_value
                location[first, second] = location[second, first] = location_value

            denominator = nboth + int(np.count_nonzero(first_only)) + int(np.count_nonzero(second_only))
            if denominator > 0:
                presence_value = nboth / denominator
                presence[first, second] = presence[second, first] = presence_value

            if bootstrap_iterations and n_scan:
                area_bootstrap = np.full(bootstrap_iterations, np.nan)
                location_bootstrap = np.full(bootstrap_iterations, np.nan)
                presence_bootstrap = np.full(bootstrap_iterations, np.nan)
                sampled_available = available[bootstrap_indices]
                sampled_both = both[bootstrap_indices]
                sample_n_available = np.count_nonzero(sampled_available, axis=1)
                sample_n_both = np.count_nonzero(sampled_both, axis=1)
                sample_support = np.minimum(
                    sample_n_available,
                    np.maximum(5, np.ceil(0.1 * sample_n_available).astype(int)),
                )
                conditional_valid = (sample_support > 0) & (sample_n_both >= sample_support)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    area_medians = np.nanmedian(area_by_scan[bootstrap_indices], axis=1)
                    location_medians = np.nanmedian(
                        location_by_scan[bootstrap_indices], axis=1
                    )
                area_bootstrap[conditional_valid] = area_medians[conditional_valid]
                location_bootstrap[conditional_valid] = location_medians[conditional_valid]
                sample_denominator = (
                    sample_n_both
                    + np.count_nonzero(first_only[bootstrap_indices], axis=1)
                    + np.count_nonzero(second_only[bootstrap_indices], axis=1)
                )
                presence_valid = sample_denominator > 0
                presence_bootstrap[presence_valid] = (
                    sample_n_both[presence_valid] / sample_denominator[presence_valid]
                )
                if np.isfinite(area[first, second]):
                    low = _nan_percentile(area_bootstrap, ci_low_percentile)
                    high = _nan_percentile(area_bootstrap, ci_high_percentile)
                    area_low[first, second] = area_low[second, first] = low
                    area_high[first, second] = area_high[second, first] = high
                if np.isfinite(location[first, second]):
                    low = _nan_percentile(location_bootstrap, ci_low_percentile)
                    high = _nan_percentile(location_bootstrap, ci_high_percentile)
                    location_low[first, second] = location_low[second, first] = low
                    location_high[first, second] = location_high[second, first] = high
                if np.isfinite(presence[first, second]):
                    low = _nan_percentile(presence_bootstrap, ci_low_percentile)
                    high = _nan_percentile(presence_bootstrap, ci_high_percentile)
                    presence_low[first, second] = presence_low[second, first] = low
                    presence_high[first, second] = presence_high[second, first] = high

    return CorrelationMatrices(
        scan_labels=scan_tuple,
        pressure_levels=pressures,
        area_by_scan=area_by_scan_all,
        location_by_scan=location_by_scan_all,
        presence_by_scan=presence_by_scan_all,
        area=area,
        location=location,
        presence=presence,
        n_available=n_available,
        n_both_present=n_both,
        n10=n10,
        n01=n01,
        n_unknown=n_unknown,
        required_support=required,
        area_ci_low=area_low,
        area_ci_high=area_high,
        location_ci_low=location_low,
        location_ci_high=location_high,
        presence_ci_low=presence_low,
        presence_ci_high=presence_high,
        bootstrap_iterations=int(bootstrap_iterations),
        random_seed=int(seed),
    )


def _auc_probability(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not positive.size or not negative.size:
        return math.nan
    ranks = rankdata(np.concatenate([positive, negative]), method="average")
    rank_sum = float(np.sum(ranks[: positive.size]))
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2.0)
        / (positive.size * negative.size)
    )


def near_far_auc(
    matrix: np.ndarray,
    pressure_levels: Sequence[float],
    *,
    minimum_distinct_gaps: int = 4,
    minimum_group_values: int = 5,
    near_gap_quantile: float = 0.25,
    far_gap_quantile: float = 0.75,
) -> dict[str, float | int | str]:
    """Summarize near/far separation using Q25/Q75 of all nonzero gaps.

    AUC and the median difference are withheld unless there are at least four
    distinct nonzero pressure gaps and at least five supported values in both
    the near and far groups.  The returned ``reason`` makes an unavailable
    result distinguishable from a weak scientific result.
    """

    values = np.asarray(matrix, dtype=float)
    pressures = np.asarray(pressure_levels, dtype=float)
    if values.shape != (pressures.size, pressures.size):
        raise ValueError("matrix shape must match pressure_levels")
    if not (0.0 < near_gap_quantile < far_gap_quantile < 1.0):
        raise ValueError("near/far gap quantiles must be ordered inside (0, 1)")
    gaps_all = np.abs(pressures[:, None] - pressures[None, :])
    nonzero_gaps = gaps_all[np.triu(np.ones_like(gaps_all, dtype=bool), 1)]
    distinct_gaps = np.unique(np.round(nonzero_gaps, decimals=12))
    if distinct_gaps.size < minimum_distinct_gaps:
        return {
            "near_gap_max": math.nan,
            "far_gap_min": math.nan,
            "near_median": math.nan,
            "far_median": math.nan,
            "auc": math.nan,
            "near_far_median_difference": math.nan,
            "near_count": 0,
            "far_count": 0,
            "distinct_gap_count": int(distinct_gaps.size),
            "reason": "insufficient_distinct_pressure_gaps",
        }
    near_limit, far_limit = np.quantile(
        nonzero_gaps, [near_gap_quantile, far_gap_quantile]
    )
    upper = np.triu(np.ones_like(values, dtype=bool), 1) & np.isfinite(values)
    near = values[upper & (gaps_all <= near_limit)]
    far = values[upper & (gaps_all >= far_limit)]
    near_median = float(np.median(near)) if near.size else math.nan
    far_median = float(np.median(far)) if far.size else math.nan
    enough_values = near.size >= minimum_group_values and far.size >= minimum_group_values
    return {
        "near_gap_max": float(near_limit),
        "far_gap_min": float(far_limit),
        "near_median": near_median,
        "far_median": far_median,
        "near_far_median_difference": (
            near_median - far_median if enough_values else math.nan
        ),
        "auc": _auc_probability(near, far) if enough_values else math.nan,
        "near_count": int(near.size),
        "far_count": int(far.size),
        "distinct_gap_count": int(distinct_gaps.size),
        "reason": "ok" if enough_values else "insufficient_supported_group_values",
    }


def bootstrap_near_far_summary(
    by_scan: np.ndarray,
    pressure_levels: Sequence[float],
    *,
    aggregate: str = "median",
    point_matrix: np.ndarray | None = None,
    bootstrap_iterations: int = 2000,
    seed: int = 0,
    minimum_distinct_gaps: int = 4,
    minimum_group_values: int = 5,
    near_gap_quantile: float = 0.25,
    far_gap_quantile: float = 0.75,
    ci_percentiles: tuple[float, float] = (2.5, 97.5),
) -> dict[str, float | int | str]:
    """Add scan-bootstrap AUC and near/far median-difference intervals.

    ``by_scan`` must be ``(n_scans,n_pressure,n_pressure)`` and contain only
    same-scan values.  ``aggregate='median'`` is used for conditional area and
    location; ``aggregate='mean'`` reproduces presence Jaccard from per-scan
    1/0 values.  If supplied, ``point_matrix`` carries the frozen full-sample
    support mask into every bootstrap replicate.
    """

    values = np.asarray(by_scan, dtype=float)
    pressures = np.asarray(pressure_levels, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (pressures.size, pressures.size):
        raise ValueError("by_scan must have shape (n_scans,n_pressure,n_pressure)")
    if aggregate not in {"median", "mean"}:
        raise ValueError("aggregate must be 'median' or 'mean'")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations cannot be negative")
    ci_low_percentile, ci_high_percentile = map(float, ci_percentiles)
    if not (0.0 <= ci_low_percentile < ci_high_percentile <= 100.0):
        raise ValueError("ci_percentiles must be ordered within [0, 100]")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        calculated_point = (
            np.nanmedian(values, axis=0)
            if aggregate == "median"
            else np.nanmean(values, axis=0)
        )
    if point_matrix is None:
        point = calculated_point
        support_mask = np.isfinite(point)
    else:
        point = np.asarray(point_matrix, dtype=float)
        if point.shape != calculated_point.shape:
            raise ValueError("point_matrix shape differs from by_scan pressure axes")
        support_mask = np.isfinite(point)
    base = near_far_auc(
        point,
        pressures,
        minimum_distinct_gaps=minimum_distinct_gaps,
        minimum_group_values=minimum_group_values,
        near_gap_quantile=near_gap_quantile,
        far_gap_quantile=far_gap_quantile,
    )

    auc_values = np.full(bootstrap_iterations, np.nan)
    difference_values = np.full(bootstrap_iterations, np.nan)
    if bootstrap_iterations and values.shape[0] and base["reason"] == "ok":
        rng = np.random.default_rng(int(seed))
        samples = rng.integers(
            0,
            values.shape[0],
            size=(bootstrap_iterations, values.shape[0]),
        )
        for iteration, sample in enumerate(samples):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                aggregate_matrix = (
                    np.nanmedian(values[sample], axis=0)
                    if aggregate == "median"
                    else np.nanmean(values[sample], axis=0)
                )
            aggregate_matrix[~support_mask] = np.nan
            summary = near_far_auc(
                aggregate_matrix,
                pressures,
                minimum_distinct_gaps=minimum_distinct_gaps,
                minimum_group_values=minimum_group_values,
                near_gap_quantile=near_gap_quantile,
                far_gap_quantile=far_gap_quantile,
            )
            auc_values[iteration] = float(summary["auc"])
            difference_values[iteration] = float(summary["near_far_median_difference"])

    result = dict(base)
    result.update(
        {
            "auc_ci_low": _nan_percentile(auc_values, ci_low_percentile),
            "auc_ci_high": _nan_percentile(auc_values, ci_high_percentile),
            "near_far_median_difference_ci_low": _nan_percentile(
                difference_values, ci_low_percentile
            ),
            "near_far_median_difference_ci_high": _nan_percentile(
                difference_values, ci_high_percentile
            ),
            "bootstrap_iterations": int(bootstrap_iterations),
            "random_seed": int(seed),
        }
    )
    return result


def analyze_per_peak(
    frame_peaks: Sequence[FramePeaks],
    pressure_levels: Sequence[float],
    scans: Sequence[str],
    config: UniformPeakConfig,
    *,
    bootstrap_iterations: int | None = None,
    seed: int | None = None,
    official_only: bool = True,
) -> PerPeakAnalysis:
    """Run consensus, tracking, assignment, correlations, and summaries."""

    consensus = build_pressure_consensus(frame_peaks, scans, pressure_levels, config)
    tracks = link_consensus_bidirectional(consensus, pressure_levels, config)
    assignment_tracks = (
        tuple(track for track in tracks if track.official) if official_only else tracks
    )
    assignments = assign_track_observations(
        assignment_tracks, frame_peaks, scans, pressure_levels, config
    )
    # Non-official fragments are retained for QC/trajectory export but do not
    # compete with supported official tracks for a fitted peak.
    if official_only:
        for track in tracks:
            if track.official:
                continue
            track_assignments: dict[tuple[str, float], AssignedObservation] = {}
            for scan_value in scans:
                scan = str(scan_value)
                for pressure_value in pressure_levels:
                    pressure = float(pressure_value)
                    in_range = _track_target(track, pressure) is not None
                    track_assignments[(scan, pressure)] = AssignedObservation(
                        track_id=track.track_id,
                        scan=scan,
                        pressure=pressure,
                        frame=None,
                        state="unknown" if in_range else "out_of_range",
                        reason=(
                            "nonofficial_track_not_assigned"
                            if in_range
                            else "outside_supported_track_range"
                        ),
                    )
            assignments[track.track_id] = track_assignments
    iterations = config.bootstrap_iterations if bootstrap_iterations is None else int(bootstrap_iterations)
    random_seed = config.random_seed if seed is None else int(seed)
    correlations: dict[str, CorrelationMatrices] = {}
    summaries: dict[str, dict[str, Mapping[str, float | int | str]]] = {}
    for track in tracks:
        if official_only and not track.official:
            continue
        matrices = compute_track_correlations(
            assignments[track.track_id],
            scans,
            pressure_levels,
            bootstrap_iterations=iterations,
            seed=random_seed,
            ci_percentiles=config.ci_percentiles,
        )
        correlations[track.track_id] = matrices
        summaries[track.track_id] = {
            "area": bootstrap_near_far_summary(
                matrices.area_by_scan,
                pressure_levels,
                aggregate="median",
                point_matrix=matrices.area,
                bootstrap_iterations=iterations,
                seed=random_seed,
                minimum_distinct_gaps=config.minimum_distinct_pressure_gaps,
                minimum_group_values=config.minimum_supported_group_values,
                near_gap_quantile=config.near_gap_quantile,
                far_gap_quantile=config.far_gap_quantile,
                ci_percentiles=config.ci_percentiles,
            ),
            "location": bootstrap_near_far_summary(
                matrices.location_by_scan,
                pressure_levels,
                aggregate="median",
                point_matrix=matrices.location,
                bootstrap_iterations=iterations,
                seed=random_seed,
                minimum_distinct_gaps=config.minimum_distinct_pressure_gaps,
                minimum_group_values=config.minimum_supported_group_values,
                near_gap_quantile=config.near_gap_quantile,
                far_gap_quantile=config.far_gap_quantile,
                ci_percentiles=config.ci_percentiles,
            ),
            "presence": bootstrap_near_far_summary(
                matrices.presence_by_scan,
                pressure_levels,
                aggregate="mean",
                point_matrix=matrices.presence,
                bootstrap_iterations=iterations,
                seed=random_seed,
                minimum_distinct_gaps=config.minimum_distinct_pressure_gaps,
                minimum_group_values=config.minimum_supported_group_values,
                near_gap_quantile=config.near_gap_quantile,
                far_gap_quantile=config.far_gap_quantile,
                ci_percentiles=config.ci_percentiles,
            ),
        }
    return PerPeakAnalysis(
        consensus_by_pressure={key: tuple(value) for key, value in consensus.items()},
        tracks=tracks,
        assignments=assignments,
        correlations=correlations,
        near_far=summaries,
    )


def run_smoke_checks() -> dict[str, bool]:
    """Small dependency-free numerical checks used by validation/CI runners."""

    area_ok = abs(relative_area_similarity(2.0, 4.0) - 0.5) < 1.0e-15
    width = 0.02
    location_at_one_fwhm = location_similarity(0.5, 0.5 + width, width, width)
    location_ok = abs(location_at_one_fwhm - 0.25) < 1.0e-12
    q = float(two_theta_to_q(10.0, 0.3066))
    roundtrip = float(q_to_two_theta(q, 0.3066))
    conversion_ok = abs(roundtrip - 10.0) < 1.0e-12
    support_ok = minimum_scan_support(56) == 6 and minimum_pressure_support(20) == 3
    result = {
        "area_formula": area_ok,
        "location_one_fwhm": location_ok,
        "q_roundtrip": conversion_ok,
        "support_formulas": support_ok,
    }
    if not all(result.values()):
        raise AssertionError(f"uniform peak core smoke checks failed: {result}")
    return result


__all__ = [
    "ALGORITHM_VERSION",
    "AssignedObservation",
    "CleanXY",
    "CorrelationMatrices",
    "FramePeaks",
    "PeakFit",
    "PerPeakAnalysis",
    "PreprocessedXY",
    "PressureConsensus",
    "RadialTrack",
    "TrackNode",
    "UniformPeakConfig",
    "analyze_per_peak",
    "asls_baseline",
    "assert_unique_present_peak_assignments",
    "assign_track_observations",
    "bootstrap_near_far_summary",
    "build_pressure_consensus",
    "clean_xy",
    "compute_track_correlations",
    "detect_pattern_peaks",
    "detect_peaks",
    "link_consensus_bidirectional",
    "location_similarity",
    "minimum_pressure_support",
    "minimum_scan_support",
    "near_far_auc",
    "preprocess_pattern",
    "preprocess_xy",
    "q_to_two_theta",
    "relative_area_similarity",
    "robust_noise",
    "run_smoke_checks",
    "two_theta_to_q",
]
