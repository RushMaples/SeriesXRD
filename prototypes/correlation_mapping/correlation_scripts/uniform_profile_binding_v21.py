#!/usr/bin/env python3
"""Strict binding for ``uniform-correlation-v2.1``.

Version 2.1 deliberately changes only the pressure-trajectory decision unit.
This binder proves that every upstream peak, consensus, correlation, window,
and plotting setting is byte-for-byte equivalent to the frozen v2 profile,
then binds the new edge-segmentation policy separately.  The v2 modules and
profile are never modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from uniform_profile_binding import BoundUniformProfile, bind_frozen_profile
from uniform_peak_tracking_v21 import SegmentedTrackingConfig


SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_NAME = "uniform-correlation-v2.1"
ALGORITHM_VERSION = "2.1.0"
UPSTREAM_VERSION = "2.0.0"
V2_PROFILE_PATH = SCRIPT_DIR / "configs" / "uniform-correlation-v2.json"


class ProfileBindingV21Error(ValueError):
    """Raised when v2.1 would change a frozen v2 upstream setting."""


@dataclass(frozen=True)
class BoundUniformProfileV21:
    """Executable v2 upstream settings plus the v2.1 tracking policy."""

    peak_config: Any
    window_config: Any
    tracking_config: SegmentedTrackingConfig
    minimum_points_per_pattern: int
    resolved_semantics: Mapping[str, Any]
    semantic_sha256: str
    binding_audit: Mapping[str, Mapping[str, str]]
    upstream_profile_sha256: str


def _canonical_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_v2_profile() -> tuple[dict[str, Any], str]:
    raw = V2_PROFILE_PATH.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _require_equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise ProfileBindingV21Error(
            f"{path} differs from frozen uniform-correlation-v2; "
            f"expected={expected!r}, actual={actual!r}"
        )


def bind_frozen_profile_v21(
    profile_value: Mapping[str, Any], wavelength: float
) -> BoundUniformProfileV21:
    """Validate v2.1 and bind it without permitting upstream threshold drift."""

    if not isinstance(profile_value, dict):
        raise ProfileBindingV21Error("profile must be a JSON object")
    profile = dict(profile_value)
    expected_top = {
        "profile",
        "status",
        "algorithm_version",
        "upstream_algorithm_version",
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
    }
    if set(profile) != expected_top:
        raise ProfileBindingV21Error(
            "profile schema mismatch; "
            f"missing={sorted(expected_top - set(profile))}, "
            f"unexpected={sorted(set(profile) - expected_top)}"
        )
    _require_equal(profile["profile"], PROFILE_NAME, "profile.profile")
    _require_equal(profile["status"], "OFFICIAL_FROZEN", "profile.status")
    _require_equal(profile["algorithm_version"], ALGORITHM_VERSION, "profile.algorithm_version")
    _require_equal(
        profile["upstream_algorithm_version"],
        UPSTREAM_VERSION,
        "profile.upstream_algorithm_version",
    )

    v2, upstream_sha = _load_v2_profile()
    _require_equal(profile["random_seed"], v2["random_seed"], "profile.random_seed")
    for section in (
        "input",
        "baseline",
        "peak_detection",
        "peak_fit",
        "consensus",
        "similarity",
        "statistics",
        "windows",
        "plotting",
    ):
        _require_equal(profile[section], v2[section], f"profile.{section}")

    tracking = profile["tracking"]
    if not isinstance(tracking, dict):
        raise ProfileBindingV21Error("profile.tracking must be an object")
    common_tracking_fields = {
        "gate_factor",
        "width_cost_weight",
        "gate_rule",
        "cost_rule",
        "maximum_missing_pressure_levels",
        "minimum_pressure_support_rule",
        "allow_extrapolation",
        "ambiguous_cost_margin",
    }
    for key in sorted(common_tracking_fields):
        _require_equal(tracking.get(key), v2["tracking"][key], f"profile.tracking.{key}")
    expected_v21_tracking = {
        *common_tracking_fields,
        "assignment",
        "unmatched_cost",
        "decision_unit",
        "require_bidirectional_same_edge",
        "require_margin_at_both_endpoints",
        "reject_q_order_crossing",
        "low_margin_competitor_state",
        "cut_reasons",
        "allow_interpolation_across_cut",
    }
    if set(tracking) != expected_v21_tracking:
        raise ProfileBindingV21Error(
            "profile.tracking schema mismatch; "
            f"missing={sorted(expected_v21_tracking - set(tracking))}, "
            f"unexpected={sorted(set(tracking) - expected_v21_tracking)}"
        )
    fixed_tracking = {
        "assignment": "bidirectional_hungarian_constant_velocity_edge_segmented",
        "decision_unit": "candidate_edge",
        "require_bidirectional_same_edge": True,
        "require_margin_at_both_endpoints": True,
        "reject_q_order_crossing": True,
        "low_margin_competitor_state": "unknown_quarantined",
        "cut_reasons": [
            "cut_one_way",
            "cut_low_margin",
            "cut_order_crossing",
            "cut_missing_too_long",
            "cut_outside_gate",
        ],
        "allow_interpolation_across_cut": False,
    }
    for key, expected in fixed_tracking.items():
        _require_equal(tracking[key], expected, f"profile.tracking.{key}")
    _require_equal(tracking["unmatched_cost"], 10.0, "profile.tracking.unmatched_cost")

    upstream_bound: BoundUniformProfile = bind_frozen_profile(v2, wavelength)
    tracking_config = SegmentedTrackingConfig(
        algorithm_version=ALGORITHM_VERSION,
        max_missing_pressure_levels=int(tracking["maximum_missing_pressure_levels"]),
        track_gate_factor=float(tracking["gate_factor"]),
        track_width_cost_weight=float(tracking["width_cost_weight"]),
        ambiguous_cost_margin=float(tracking["ambiguous_cost_margin"]),
        unmatched_cost=float(tracking["unmatched_cost"]),
        decision_unit=str(tracking["decision_unit"]),
        require_bidirectional_same_edge=bool(
            tracking["require_bidirectional_same_edge"]
        ),
        require_margin_at_both_endpoints=bool(
            tracking["require_margin_at_both_endpoints"]
        ),
        reject_q_order_crossing=bool(tracking["reject_q_order_crossing"]),
        low_margin_competitor_state=str(tracking["low_margin_competitor_state"]),
        cut_reasons=tuple(str(item) for item in tracking["cut_reasons"]),
        allow_interpolation_across_cut=bool(
            tracking["allow_interpolation_across_cut"]
        ),
    )
    resolved_semantics = {
        "profile": PROFILE_NAME,
        "algorithm_version": ALGORITHM_VERSION,
        "upstream_algorithm_version": UPSTREAM_VERSION,
        "upstream_profile_sha256": upstream_sha,
        "peak_config": asdict(upstream_bound.peak_config),
        "window_config": asdict(upstream_bound.window_config),
        "segmented_tracking_config": asdict(tracking_config),
        "tracking_policy": fixed_tracking,
        "minimum_points_per_pattern": upstream_bound.minimum_points_per_pattern,
    }
    binding_audit = {
        **upstream_bound.binding_audit,
        "segmented_tracking_config": {
            "algorithm_version": "algorithm_version",
            "max_missing_pressure_levels": "tracking.maximum_missing_pressure_levels",
            "track_gate_factor": "tracking.gate_factor",
            "track_width_cost_weight": "tracking.width_cost_weight",
            "ambiguous_cost_margin": "tracking.ambiguous_cost_margin",
            "unmatched_cost": "tracking.unmatched_cost",
            "decision_unit": "tracking.decision_unit",
            "require_bidirectional_same_edge": "tracking.require_bidirectional_same_edge",
            "require_margin_at_both_endpoints": "tracking.require_margin_at_both_endpoints",
            "reject_q_order_crossing": "tracking.reject_q_order_crossing",
            "low_margin_competitor_state": "tracking.low_margin_competitor_state",
            "cut_reasons": "tracking.cut_reasons",
            "allow_interpolation_across_cut": "tracking.allow_interpolation_across_cut",
        },
        "tracking_policy": {
            key: f"tracking.{key}" for key in fixed_tracking
        },
    }
    return BoundUniformProfileV21(
        peak_config=upstream_bound.peak_config,
        window_config=upstream_bound.window_config,
        tracking_config=tracking_config,
        minimum_points_per_pattern=upstream_bound.minimum_points_per_pattern,
        resolved_semantics=resolved_semantics,
        semantic_sha256=_canonical_sha(resolved_semantics),
        binding_audit=binding_audit,
        upstream_profile_sha256=upstream_sha,
    )


__all__ = [
    "ALGORITHM_VERSION",
    "BoundUniformProfileV21",
    "PROFILE_NAME",
    "ProfileBindingV21Error",
    "UPSTREAM_VERSION",
    "bind_frozen_profile_v21",
]
