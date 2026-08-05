#!/usr/bin/env python3
"""Strictly bind the frozen JSON profile to executable uniform-v2 settings.

The runner must never instantiate a scientific config with silent dataclass
defaults.  This module validates the complete profile schema, rejects textual
algorithm rules that the implementation does not implement, constructs every
``UniformPeakConfig`` and ``UniformWindowConfig`` field explicitly, and emits
a canonical semantic digest suitable for the run manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import Any, Mapping

import uniform_peak_core as peak
import uniform_window_core as window


PROFILE_NAME = "uniform-correlation-v2"


class ProfileBindingError(ValueError):
    """Raised when the frozen profile and executable semantics disagree."""


@dataclass(frozen=True)
class BoundUniformProfile:
    peak_config: peak.UniformPeakConfig
    window_config: window.UniformWindowConfig
    minimum_points_per_pattern: int
    resolved_semantics: Mapping[str, Any]
    semantic_sha256: str
    binding_audit: Mapping[str, Mapping[str, str]]


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProfileBindingError(f"{path} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProfileBindingError(
            f"{path} schema mismatch; missing={missing}, unexpected={extra}"
        )


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ProfileBindingError(f"{path} must be a string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ProfileBindingError(f"{path} must be boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileBindingError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ProfileBindingError(f"{path} must be >= {minimum}")
    return int(value)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileBindingError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProfileBindingError(f"{path} must be finite")
    return result


def _number_pair(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ProfileBindingError(f"{path} must contain exactly two numbers")
    return _number(value[0], f"{path}[0]"), _number(value[1], f"{path}[1]")


def _require_equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise ProfileBindingError(
            f"{path}={actual!r} is unsupported; implementation requires {expected!r}"
        )


def _source_audit() -> dict[str, dict[str, str]]:
    peak_sources = {
        "wavelength": "runtime.--wavelength-A",
        "algorithm_version": "algorithm_version",
        "asls_p": "baseline.asls_p",
        "asls_iterations": "baseline.asls_iterations",
        "asls_min_span_deg": "baseline.lambda_minimum_span_deg",
        "asls_min_span_bins": "baseline.lambda_minimum_span_bins",
        "gaussian_sigma_bins": "peak_detection.gaussian_sigma_bins",
        "prominence_noise_factor": "peak_detection.minimum_prominence_sigma",
        "height_noise_factor": "peak_detection.minimum_height_sigma",
        "minimum_width_bins": "peak_detection.minimum_width_bins",
        "delta_bic_minimum": "peak_fit.minimum_delta_bic",
        "area_over_se_minimum": "peak_fit.minimum_area_over_se",
        "fit_loss": "peak_fit.loss",
        "fit_max_nfev": "peak_fit.maximum_function_evaluations",
        "boundary_relative_tolerance": "peak_fit.boundary_relative_tolerance",
        "reject_parameter_bound_hits": "peak_fit.reject_parameter_bound_hits",
        "overlap_grouping": "peak_fit.overlap_grouping",
        "consensus_fwhm_factor": "consensus.fwhm_distance_factor",
        "max_missing_pressure_levels": "tracking.maximum_missing_pressure_levels",
        "track_gate_factor": "tracking.gate_factor",
        "track_width_cost_weight": "tracking.width_cost_weight",
        "ambiguous_cost_margin": "tracking.ambiguous_cost_margin",
        "bootstrap_iterations": "statistics.bootstrap_resamples",
        "random_seed": "random_seed",
        "ci_percentiles": "statistics.ci_percentiles",
        "near_gap_quantile": "statistics.near_gap_quantile",
        "far_gap_quantile": "statistics.far_gap_quantile",
        "minimum_distinct_pressure_gaps": "statistics.minimum_distinct_pressure_gaps",
        "minimum_supported_group_values": "statistics.minimum_supported_group_values",
    }
    window_sources = {
        "algorithm_version": "algorithm_version",
        "coverage_fraction": "input.minimum_coverage_fraction",
        "allow_extrapolation": "input.allow_extrapolation",
        "window_count": "windows.window_count",
        "width_divisor": "windows.width_rule",
        "step_divisor": "windows.step_rule",
        "minimum_finite_fraction": "windows.minimum_finite_fraction_per_window",
        "fingerprint_method": "windows.fingerprint",
        "strict_acf_primary": "windows.strict_acf_primary",
        "direct_strict_validation": "windows.direct_strict_validation",
        "shift_tolerant_neighbor_steps": "windows.shift_tolerant_neighbor_steps",
        "shift_tolerant_role": "windows.shift_tolerant_role",
        "nonoverlap_stride_windows": "windows.nonoverlap_stride_windows",
        "bootstrap_iterations": "statistics.bootstrap_resamples",
        "random_seed": "random_seed",
        "confidence": "statistics.ci_percentiles",
        "near_gap_quantile": "statistics.near_gap_quantile",
        "far_gap_quantile": "statistics.far_gap_quantile",
        "minimum_distinct_pressure_gaps": "statistics.minimum_distinct_pressure_gaps",
        "minimum_supported_group_values": "statistics.minimum_supported_group_values",
    }
    peak_fields = {item.name for item in fields(peak.UniformPeakConfig)}
    window_fields = {item.name for item in fields(window.UniformWindowConfig)}
    if set(peak_sources) != peak_fields:
        raise RuntimeError(
            "UniformPeakConfig field binding is incomplete: "
            f"missing={sorted(peak_fields - set(peak_sources))}, "
            f"extra={sorted(set(peak_sources) - peak_fields)}"
        )
    if set(window_sources) != window_fields:
        raise RuntimeError(
            "UniformWindowConfig field binding is incomplete: "
            f"missing={sorted(window_fields - set(window_sources))}, "
            f"extra={sorted(set(window_sources) - window_fields)}"
        )
    return {"peak_config": peak_sources, "window_config": window_sources}


def bind_frozen_profile(
    profile_value: Mapping[str, Any], wavelength: float
) -> BoundUniformProfile:
    """Validate and bind every frozen profile setting to executable semantics."""

    profile = _object(profile_value, "profile")
    _exact_keys(
        profile,
        {
            "profile",
            "status",
            "algorithm_version",
            "random_seed",
            "input",
            "baseline",
            "peak_detection",
            "peak_fit",
            "consensus",
            "tracking",
            "similarity",
            "statistics",
            "windows",
            "plotting",
        },
        "profile",
    )
    _require_equal(_string(profile["profile"], "profile.profile"), PROFILE_NAME, "profile.profile")
    _require_equal(
        _string(profile["status"], "profile.status"), "OFFICIAL_FROZEN", "profile.status"
    )
    algorithm_version = _string(profile["algorithm_version"], "profile.algorithm_version")
    _require_equal(algorithm_version, peak.ALGORITHM_VERSION, "profile.algorithm_version")
    _require_equal(algorithm_version, window.ALGORITHM_VERSION, "profile.algorithm_version")
    seed = _integer(profile["random_seed"], "profile.random_seed", minimum=0)

    input_config = _object(profile["input"], "profile.input")
    _exact_keys(
        input_config,
        {
            "minimum_coverage_fraction",
            "minimum_points_per_pattern",
            "require_wavelength",
            "allow_extrapolation",
        },
        "profile.input",
    )
    coverage_fraction = _number(
        input_config["minimum_coverage_fraction"],
        "profile.input.minimum_coverage_fraction",
    )
    minimum_points = _integer(
        input_config["minimum_points_per_pattern"],
        "profile.input.minimum_points_per_pattern",
        minimum=5,
    )
    _require_equal(
        _boolean(input_config["require_wavelength"], "profile.input.require_wavelength"),
        True,
        "profile.input.require_wavelength",
    )
    allow_extrapolation = _boolean(
        input_config["allow_extrapolation"], "profile.input.allow_extrapolation"
    )
    _require_equal(allow_extrapolation, False, "profile.input.allow_extrapolation")

    baseline = _object(profile["baseline"], "profile.baseline")
    _exact_keys(
        baseline,
        {
            "method",
            "asls_p",
            "asls_iterations",
            "lambda_minimum_span_deg",
            "lambda_minimum_span_bins",
            "lambda_rule",
        },
        "profile.baseline",
    )
    _require_equal(
        _string(baseline["method"], "profile.baseline.method"),
        "asymmetric_least_squares",
        "profile.baseline.method",
    )
    _require_equal(
        _string(baseline["lambda_rule"], "profile.baseline.lambda_rule"),
        "(max(0.5_deg,20*dx)/dx)^4",
        "profile.baseline.lambda_rule",
    )
    _require_equal(
        _number(baseline["lambda_minimum_span_deg"], "profile.baseline.lambda_minimum_span_deg"),
        0.5,
        "profile.baseline.lambda_minimum_span_deg",
    )
    _require_equal(
        _integer(baseline["lambda_minimum_span_bins"], "profile.baseline.lambda_minimum_span_bins", minimum=1),
        20,
        "profile.baseline.lambda_minimum_span_bins",
    )

    detection = _object(profile["peak_detection"], "profile.peak_detection")
    _exact_keys(
        detection,
        {
            "gaussian_sigma_bins",
            "noise_method",
            "minimum_prominence_sigma",
            "minimum_height_sigma",
            "minimum_width_bins",
        },
        "profile.peak_detection",
    )
    _require_equal(
        _string(detection["noise_method"], "profile.peak_detection.noise_method"),
        "1.4826*MAD(diff(residual))/sqrt(2)",
        "profile.peak_detection.noise_method",
    )

    peak_fit = _object(profile["peak_fit"], "profile.peak_fit")
    _exact_keys(
        peak_fit,
        {
            "model",
            "loss",
            "minimum_delta_bic",
            "minimum_area_over_se",
            "maximum_function_evaluations",
            "boundary_relative_tolerance",
            "reject_parameter_bound_hits",
            "overlap_grouping",
        },
        "profile.peak_fit",
    )
    _require_equal(
        _string(peak_fit["model"], "profile.peak_fit.model"),
        "linear_background_plus_pseudo_voigt",
        "profile.peak_fit.model",
    )
    fit_loss = _string(peak_fit["loss"], "profile.peak_fit.loss")
    _require_equal(fit_loss, "soft_l1", "profile.peak_fit.loss")

    consensus = _object(profile["consensus"], "profile.consensus")
    _exact_keys(
        consensus,
        {
            "coordinate",
            "clustering",
            "fwhm_distance_factor",
            "distance_rule",
            "support_rule",
        },
        "profile.consensus",
    )
    for key, expected in {
        "coordinate": "q_A^-1",
        "clustering": "complete_link",
        "distance_rule": "abs(dq)<=0.5*(fwhm_q_i+fwhm_q_j)",
        "support_rule": "min(N,max(5,ceil(0.1*N)))",
    }.items():
        _require_equal(_string(consensus[key], f"profile.consensus.{key}"), expected, f"profile.consensus.{key}")
    _require_equal(
        _number(consensus["fwhm_distance_factor"], "profile.consensus.fwhm_distance_factor"),
        0.5,
        "profile.consensus.fwhm_distance_factor",
    )

    tracking = _object(profile["tracking"], "profile.tracking")
    _exact_keys(
        tracking,
        {
            "assignment",
            "gate_factor",
            "width_cost_weight",
            "gate_rule",
            "cost_rule",
            "maximum_missing_pressure_levels",
            "minimum_pressure_support_rule",
            "allow_extrapolation",
            "ambiguous_cost_margin",
        },
        "profile.tracking",
    )
    for key, expected in {
        "assignment": "bidirectional_hungarian_constant_velocity",
        "gate_rule": "1.5*sqrt(w_pred^2+w_obs^2)*max(1,dP/median_positive_dP)",
        "cost_rule": "(dq/gate)^2+0.1*log(w_obs/w_pred)^2",
        "minimum_pressure_support_rule": "min(M,max(3,ceil(0.1*M)))",
    }.items():
        _require_equal(_string(tracking[key], f"profile.tracking.{key}"), expected, f"profile.tracking.{key}")
    _require_equal(
        _boolean(tracking["allow_extrapolation"], "profile.tracking.allow_extrapolation"),
        False,
        "profile.tracking.allow_extrapolation",
    )
    _require_equal(
        _number(tracking["gate_factor"], "profile.tracking.gate_factor"),
        1.5,
        "profile.tracking.gate_factor",
    )
    _require_equal(
        _number(tracking["width_cost_weight"], "profile.tracking.width_cost_weight"),
        0.1,
        "profile.tracking.width_cost_weight",
    )

    similarity = _object(profile["similarity"], "profile.similarity")
    expected_similarity = {
        "area": "min(relative_area_i,relative_area_j)/max(relative_area_i,relative_area_j)",
        "location": "exp(-(q_i-q_j)^2/(2*(sigma_q_i^2+sigma_q_j^2)))",
        "sigma_q": "fwhm_q/(2*sqrt(2*ln(2)))",
        "presence": "n11/(n11+n10+n01)",
        "missing_area_location": "NaN",
    }
    _exact_keys(similarity, set(expected_similarity), "profile.similarity")
    for key, expected in expected_similarity.items():
        _require_equal(_string(similarity[key], f"profile.similarity.{key}"), expected, f"profile.similarity.{key}")

    statistics = _object(profile["statistics"], "profile.statistics")
    _exact_keys(
        statistics,
        {
            "pair_support_rule",
            "bootstrap_resamples",
            "bootstrap_unit",
            "ci_percentiles",
            "near_gap_quantile",
            "far_gap_quantile",
            "minimum_distinct_pressure_gaps",
            "minimum_supported_group_values",
        },
        "profile.statistics",
    )
    _require_equal(
        _string(statistics["pair_support_rule"], "profile.statistics.pair_support_rule"),
        "min(N,max(5,ceil(0.1*N)))",
        "profile.statistics.pair_support_rule",
    )
    _require_equal(
        _string(statistics["bootstrap_unit"], "profile.statistics.bootstrap_unit"),
        "scan",
        "profile.statistics.bootstrap_unit",
    )
    ci_percentiles = _number_pair(
        statistics["ci_percentiles"], "profile.statistics.ci_percentiles"
    )
    if not (0.0 <= ci_percentiles[0] < ci_percentiles[1] <= 100.0):
        raise ProfileBindingError("profile.statistics.ci_percentiles are invalid")
    if not math.isclose(ci_percentiles[0] + ci_percentiles[1], 100.0, abs_tol=1e-12):
        raise ProfileBindingError("CI percentiles must be symmetric")
    confidence = (ci_percentiles[1] - ci_percentiles[0]) / 100.0

    windows = _object(profile["windows"], "profile.windows")
    _exact_keys(
        windows,
        {
            "width_rule",
            "step_rule",
            "window_count",
            "fingerprint",
            "strict_acf_primary",
            "direct_strict_validation",
            "shift_tolerant_neighbor_steps",
            "shift_tolerant_role",
            "nonoverlap_stride_windows",
            "minimum_finite_fraction_per_window",
        },
        "profile.windows",
    )
    for key, expected in {
        "width_rule": "analysis_span/6",
        "step_rule": "window_width/5",
        "fingerprint": "standardized_positive_lag_fft_acf",
        "shift_tolerant_role": "SECONDARY",
    }.items():
        _require_equal(_string(windows[key], f"profile.windows.{key}"), expected, f"profile.windows.{key}")

    plotting = _object(profile["plotting"], "profile.plotting")
    _exact_keys(
        plotting,
        {
            "full_symmetric_matrices",
            "conditional_score_range",
            "pearson_score_range",
            "missing_color",
            "insufficient_support_hatch",
        },
        "profile.plotting",
    )
    _require_equal(_boolean(plotting["full_symmetric_matrices"], "profile.plotting.full_symmetric_matrices"), True, "profile.plotting.full_symmetric_matrices")
    _require_equal(_number_pair(plotting["conditional_score_range"], "profile.plotting.conditional_score_range"), (0.0, 1.0), "profile.plotting.conditional_score_range")
    _require_equal(_number_pair(plotting["pearson_score_range"], "profile.plotting.pearson_score_range"), (-1.0, 1.0), "profile.plotting.pearson_score_range")
    _require_equal(_string(plotting["missing_color"], "profile.plotting.missing_color"), "#BDBDBD", "profile.plotting.missing_color")
    _require_equal(_string(plotting["insufficient_support_hatch"], "profile.plotting.insufficient_support_hatch"), "///", "profile.plotting.insufficient_support_hatch")

    peak_config = peak.UniformPeakConfig(
        wavelength=_number(wavelength, "runtime.wavelength_A"),
        algorithm_version=algorithm_version,
        asls_p=_number(baseline["asls_p"], "profile.baseline.asls_p"),
        asls_iterations=_integer(baseline["asls_iterations"], "profile.baseline.asls_iterations", minimum=1),
        asls_min_span_deg=_number(baseline["lambda_minimum_span_deg"], "profile.baseline.lambda_minimum_span_deg"),
        asls_min_span_bins=_integer(baseline["lambda_minimum_span_bins"], "profile.baseline.lambda_minimum_span_bins", minimum=1),
        gaussian_sigma_bins=_number(detection["gaussian_sigma_bins"], "profile.peak_detection.gaussian_sigma_bins"),
        prominence_noise_factor=_number(detection["minimum_prominence_sigma"], "profile.peak_detection.minimum_prominence_sigma"),
        height_noise_factor=_number(detection["minimum_height_sigma"], "profile.peak_detection.minimum_height_sigma"),
        minimum_width_bins=_number(detection["minimum_width_bins"], "profile.peak_detection.minimum_width_bins"),
        delta_bic_minimum=_number(peak_fit["minimum_delta_bic"], "profile.peak_fit.minimum_delta_bic"),
        area_over_se_minimum=_number(peak_fit["minimum_area_over_se"], "profile.peak_fit.minimum_area_over_se"),
        fit_loss=fit_loss,
        fit_max_nfev=_integer(peak_fit["maximum_function_evaluations"], "profile.peak_fit.maximum_function_evaluations", minimum=1),
        boundary_relative_tolerance=_number(peak_fit["boundary_relative_tolerance"], "profile.peak_fit.boundary_relative_tolerance"),
        reject_parameter_bound_hits=_boolean(peak_fit["reject_parameter_bound_hits"], "profile.peak_fit.reject_parameter_bound_hits"),
        overlap_grouping=_boolean(peak_fit["overlap_grouping"], "profile.peak_fit.overlap_grouping"),
        consensus_fwhm_factor=_number(consensus["fwhm_distance_factor"], "profile.consensus.fwhm_distance_factor"),
        max_missing_pressure_levels=_integer(tracking["maximum_missing_pressure_levels"], "profile.tracking.maximum_missing_pressure_levels", minimum=0),
        track_gate_factor=_number(tracking["gate_factor"], "profile.tracking.gate_factor"),
        track_width_cost_weight=_number(tracking["width_cost_weight"], "profile.tracking.width_cost_weight"),
        ambiguous_cost_margin=_number(tracking["ambiguous_cost_margin"], "profile.tracking.ambiguous_cost_margin"),
        bootstrap_iterations=_integer(statistics["bootstrap_resamples"], "profile.statistics.bootstrap_resamples", minimum=0),
        random_seed=seed,
        ci_percentiles=ci_percentiles,
        near_gap_quantile=_number(statistics["near_gap_quantile"], "profile.statistics.near_gap_quantile"),
        far_gap_quantile=_number(statistics["far_gap_quantile"], "profile.statistics.far_gap_quantile"),
        minimum_distinct_pressure_gaps=_integer(statistics["minimum_distinct_pressure_gaps"], "profile.statistics.minimum_distinct_pressure_gaps", minimum=2),
        minimum_supported_group_values=_integer(statistics["minimum_supported_group_values"], "profile.statistics.minimum_supported_group_values", minimum=1),
    )
    window_config = window.UniformWindowConfig(
        algorithm_version=algorithm_version,
        coverage_fraction=coverage_fraction,
        allow_extrapolation=allow_extrapolation,
        window_count=_integer(windows["window_count"], "profile.windows.window_count", minimum=1),
        width_divisor=6.0,
        step_divisor=5.0,
        minimum_finite_fraction=_number(windows["minimum_finite_fraction_per_window"], "profile.windows.minimum_finite_fraction_per_window"),
        fingerprint_method=_string(windows["fingerprint"], "profile.windows.fingerprint"),
        strict_acf_primary=_boolean(windows["strict_acf_primary"], "profile.windows.strict_acf_primary"),
        direct_strict_validation=_boolean(windows["direct_strict_validation"], "profile.windows.direct_strict_validation"),
        shift_tolerant_neighbor_steps=_integer(windows["shift_tolerant_neighbor_steps"], "profile.windows.shift_tolerant_neighbor_steps", minimum=0),
        shift_tolerant_role=_string(windows["shift_tolerant_role"], "profile.windows.shift_tolerant_role"),
        nonoverlap_stride_windows=_integer(windows["nonoverlap_stride_windows"], "profile.windows.nonoverlap_stride_windows", minimum=1),
        bootstrap_iterations=_integer(statistics["bootstrap_resamples"], "profile.statistics.bootstrap_resamples", minimum=0),
        random_seed=seed,
        confidence=confidence,
        near_gap_quantile=_number(statistics["near_gap_quantile"], "profile.statistics.near_gap_quantile"),
        far_gap_quantile=_number(statistics["far_gap_quantile"], "profile.statistics.far_gap_quantile"),
        minimum_distinct_pressure_gaps=_integer(statistics["minimum_distinct_pressure_gaps"], "profile.statistics.minimum_distinct_pressure_gaps", minimum=2),
        minimum_supported_group_values=_integer(statistics["minimum_supported_group_values"], "profile.statistics.minimum_supported_group_values", minimum=1),
    )
    audit = _source_audit()
    semantics = {
        "profile": PROFILE_NAME,
        "algorithm_version": algorithm_version,
        "peak_config": asdict(peak_config),
        "window_config": asdict(window_config),
        "minimum_points_per_pattern": minimum_points,
        "fixed_rules": {
            "baseline_method": baseline["method"],
            "baseline_lambda_rule": baseline["lambda_rule"],
            "noise_method": detection["noise_method"],
            "fit_model": peak_fit["model"],
            "consensus_coordinate": consensus["coordinate"],
            "consensus_clustering": consensus["clustering"],
            "consensus_distance_rule": consensus["distance_rule"],
            "scan_support_rule": consensus["support_rule"],
            "tracking_assignment": tracking["assignment"],
            "tracking_gate_rule": tracking["gate_rule"],
            "tracking_cost_rule": tracking["cost_rule"],
            "pressure_support_rule": tracking["minimum_pressure_support_rule"],
            "pair_support_rule": statistics["pair_support_rule"],
            "bootstrap_unit": statistics["bootstrap_unit"],
            "similarity": dict(similarity),
            "plotting": dict(plotting),
        },
    }
    encoded = json.dumps(
        semantics, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return BoundUniformProfile(
        peak_config=peak_config,
        window_config=window_config,
        minimum_points_per_pattern=minimum_points,
        resolved_semantics=semantics,
        semantic_sha256=hashlib.sha256(encoded).hexdigest(),
        binding_audit=audit,
    )


__all__ = [
    "BoundUniformProfile",
    "PROFILE_NAME",
    "ProfileBindingError",
    "bind_frozen_profile",
]
