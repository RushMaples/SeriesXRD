#!/usr/bin/env python3
"""Stable Log² intensity preprocessing for the UOTe correlations.

The literal mapping ``log(f**2)`` is singular at zero. This module implements
the agreed dimensionless, bounded Log² transform without changing any
downstream ROI-IoU, location, or window-correlation algorithm.

For physically normalized, nonnegative ROI intensity ``f`` and one fixed
pooled scale ``a`` shared by every compared frame,

    z = clip(max(f, 0) / a, 0, 1).

The transform is

    [ln(z**2 + eps) - ln(eps)]
    / [ln(1 + eps) - ln(eps)].

For the logarithmic transform, ``eps`` is derived from a documented physical
noise floor ``sigma`` as ``max((sigma/a)**2, epsilon_floor)``.  Masked values
remain masked and unmasked NaNs remain NaN.  The lower-level
``transform_bounded_squared`` function also accepts signed normalized window
data; its input is clipped to ``[-1, 1]`` before squaring.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np


TransformMethod = Literal["log_squared"]

LOG_SQUARED: TransformMethod = "log_squared"
SUPPORTED_METHODS: tuple[TransformMethod, ...] = (LOG_SQUARED,)
DEFAULT_SCALE_QUANTILE = 0.995
DEFAULT_EPSILON_FLOOR = 1.0e-12
SCHEMA_VERSION = "uotexrd-log-squared-intensity-preprocessing-v2"


ArrayLike = np.ndarray | np.ma.MaskedArray | Sequence[float]

__all__ = [
    "DEFAULT_EPSILON_FLOOR",
    "DEFAULT_SCALE_QUANTILE",
    "LOG_SQUARED",
    "PooledScaleEstimate",
    "ROITransformSpec",
    "SCHEMA_VERSION",
    "SUPPORTED_METHODS",
    "audit_roi_transform",
    "build_transform_provenance",
    "epsilon_from_noise_floor",
    "estimate_fixed_pooled_scale",
    "fit_roi_transform",
    "make_roi_transform_spec",
    "roi_z_values",
    "transform_bounded_squared",
    "transform_roi_intensity",
    "write_numerical_audit",
    "write_transform_provenance",
]


@dataclass(frozen=True)
class PooledScaleEstimate:
    """A fixed positive pooled quantile and its complete input accounting."""

    scale: float
    quantile: float
    array_count: int
    total_slots: int
    masked_slots: int
    unmasked_nan_slots: int
    unmasked_positive_infinity_slots: int
    unmasked_negative_infinity_slots: int
    finite_unmasked_slots: int
    negative_finite_slots: int
    zero_finite_slots: int
    positive_finite_slots: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return _json_ready(asdict(self))


@dataclass(frozen=True)
class ROITransformSpec:
    """Frozen parameters for one cross-frame-comparable ROI transform."""

    method: TransformMethod
    scale: float
    noise_floor: float | None
    epsilon: float | None
    scale_quantile: float | None = DEFAULT_SCALE_QUANTILE
    epsilon_floor: float = DEFAULT_EPSILON_FLOOR
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"unsupported transform method {self.method!r}; "
                f"expected one of {SUPPORTED_METHODS}"
            )
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and strictly positive")
        if self.noise_floor is not None and (
            not np.isfinite(self.noise_floor) or self.noise_floor < 0.0
        ):
            raise ValueError("noise_floor must be finite and nonnegative")
        if not np.isfinite(self.epsilon_floor) or self.epsilon_floor <= 0.0:
            raise ValueError("epsilon_floor must be finite and strictly positive")
        if self.scale_quantile is not None and not (
            0.0 < self.scale_quantile <= 1.0
        ):
            raise ValueError("scale_quantile must be in (0, 1]")
        if self.epsilon is None or not np.isfinite(self.epsilon):
            raise ValueError("log_squared requires a finite epsilon")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be strictly positive")

    def transform(self, values: ArrayLike) -> np.ndarray | np.ma.MaskedArray:
        """Apply this specification to physically normalized ROI intensity."""

        return transform_roi_intensity(values, self)

    def audit(self, values: ArrayLike) -> dict[str, Any]:
        """Return a JSON-safe numerical audit for this specification."""

        return audit_roi_transform(values, self)

    def to_dict(self) -> dict[str, Any]:
        """Return frozen parameters in JSON-safe form."""

        return _json_ready(asdict(self))


def _iter_arrays(values: ArrayLike | Iterable[ArrayLike]) -> Iterable[ArrayLike]:
    if isinstance(values, (np.ndarray, np.ma.MaskedArray)):
        yield values
        return
    # A plain numeric list is one array, whereas a list/tuple of ndarrays is a
    # pool of arrays.  Materializing the outer iterable also supports generators.
    if isinstance(values, Sequence) and values and np.isscalar(values[0]):
        yield values
        return
    yield from values  # type: ignore[misc]


def _data_and_mask(values: ArrayLike) -> tuple[np.ndarray, np.ndarray, bool]:
    is_masked = np.ma.isMaskedArray(values)
    if is_masked:
        data = np.asarray(np.ma.getdata(values), dtype=np.float64)
        mask = np.ma.getmaskarray(values).astype(bool, copy=True)
    else:
        data = np.asarray(values, dtype=np.float64)
        mask = np.zeros(data.shape, dtype=bool)
    return data, mask, bool(is_masked)


def _restore_mask_type(
    data: np.ndarray,
    mask: np.ndarray,
    was_masked: bool,
) -> np.ndarray | np.ma.MaskedArray:
    if was_masked:
        return np.ma.array(data, mask=mask, copy=False)
    return data


def estimate_fixed_pooled_scale(
    values: ArrayLike | Iterable[ArrayLike],
    *,
    quantile: float = DEFAULT_SCALE_QUANTILE,
) -> PooledScaleEstimate:
    """Estimate one fixed scale from all positive, finite, unmasked values.

    The returned scale must be computed once for a complete comparison family
    and then reused for every frame.  Per-frame ROI scales would erase the
    physical amplitude differences that the downstream ROI score is intended
    to compare.
    """

    if not np.isfinite(quantile) or not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must be finite and in (0, 1]")

    positives: list[np.ndarray] = []
    counts = {
        "array_count": 0,
        "total_slots": 0,
        "masked_slots": 0,
        "unmasked_nan_slots": 0,
        "unmasked_positive_infinity_slots": 0,
        "unmasked_negative_infinity_slots": 0,
        "finite_unmasked_slots": 0,
        "negative_finite_slots": 0,
        "zero_finite_slots": 0,
        "positive_finite_slots": 0,
    }
    for item in _iter_arrays(values):
        data, mask, _ = _data_and_mask(item)
        counts["array_count"] += 1
        counts["total_slots"] += int(data.size)
        counts["masked_slots"] += int(np.count_nonzero(mask))
        unmasked = ~mask
        counts["unmasked_nan_slots"] += int(
            np.count_nonzero(unmasked & np.isnan(data))
        )
        counts["unmasked_positive_infinity_slots"] += int(
            np.count_nonzero(unmasked & np.isposinf(data))
        )
        counts["unmasked_negative_infinity_slots"] += int(
            np.count_nonzero(unmasked & np.isneginf(data))
        )
        finite = unmasked & np.isfinite(data)
        finite_values = data[finite]
        counts["finite_unmasked_slots"] += int(finite_values.size)
        counts["negative_finite_slots"] += int(
            np.count_nonzero(finite_values < 0.0)
        )
        counts["zero_finite_slots"] += int(
            np.count_nonzero(finite_values == 0.0)
        )
        positive = finite_values[finite_values > 0.0]
        counts["positive_finite_slots"] += int(positive.size)
        if positive.size:
            positives.append(positive)

    if counts["array_count"] == 0:
        raise ValueError("at least one pooled array is required")
    if not positives:
        raise ValueError("pooled input has no positive finite unmasked values")
    positive_values = np.concatenate(positives)
    scale = float(np.quantile(positive_values, quantile))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("pooled positive quantile did not produce a valid scale")
    return PooledScaleEstimate(
        scale=scale,
        quantile=float(quantile),
        **counts,
    )


def epsilon_from_noise_floor(
    noise_floor: float,
    scale: float,
    *,
    epsilon_floor: float = DEFAULT_EPSILON_FLOOR,
) -> float:
    """Return ``max((noise_floor/scale)**2, epsilon_floor)``."""

    if not np.isfinite(noise_floor) or noise_floor < 0.0:
        raise ValueError("noise_floor must be finite and nonnegative")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and strictly positive")
    if not np.isfinite(epsilon_floor) or epsilon_floor <= 0.0:
        raise ValueError("epsilon_floor must be finite and strictly positive")
    return float(max((float(noise_floor) / float(scale)) ** 2, epsilon_floor))


def make_roi_transform_spec(
    method: TransformMethod,
    *,
    scale: float,
    noise_floor: float | None = None,
    scale_quantile: float | None = DEFAULT_SCALE_QUANTILE,
    epsilon_floor: float = DEFAULT_EPSILON_FLOOR,
) -> ROITransformSpec:
    """Freeze validated parameters for one ROI transform."""

    if noise_floor is None:
        raise ValueError(
            "log_squared requires a documented physical noise_floor"
        )
    epsilon = epsilon_from_noise_floor(
        noise_floor,
        scale,
        epsilon_floor=epsilon_floor,
    )
    return ROITransformSpec(
        method=method,
        scale=float(scale),
        noise_floor=None if noise_floor is None else float(noise_floor),
        epsilon=epsilon,
        scale_quantile=scale_quantile,
        epsilon_floor=float(epsilon_floor),
    )


def fit_roi_transform(
    values: ArrayLike | Iterable[ArrayLike],
    method: TransformMethod,
    *,
    noise_floor: float | None = None,
    scale_quantile: float = DEFAULT_SCALE_QUANTILE,
    epsilon_floor: float = DEFAULT_EPSILON_FLOOR,
) -> tuple[ROITransformSpec, PooledScaleEstimate]:
    """Estimate one pooled scale and return a frozen reusable specification."""

    estimate = estimate_fixed_pooled_scale(values, quantile=scale_quantile)
    spec = make_roi_transform_spec(
        method,
        scale=estimate.scale,
        noise_floor=noise_floor,
        scale_quantile=estimate.quantile,
        epsilon_floor=epsilon_floor,
    )
    return spec, estimate


def roi_z_values(
    values: ArrayLike,
    *,
    scale: float,
) -> np.ndarray | np.ma.MaskedArray:
    """Return ``clip(max(f, 0)/scale, 0, 1)`` while preserving masks/NaNs."""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and strictly positive")
    data, mask, was_masked = _data_and_mask(values)
    output = np.full(data.shape, np.nan, dtype=np.float64)
    valid = (~mask) & np.isfinite(data)
    output[valid] = np.clip(np.maximum(data[valid], 0.0) / scale, 0.0, 1.0)
    return _restore_mask_type(output, mask, was_masked)


def transform_bounded_squared(
    normalized_values: ArrayLike,
    *,
    method: TransformMethod,
    epsilon: float | None = None,
) -> np.ndarray | np.ma.MaskedArray:
    """Transform dimensionless values after clipping them to ``[-1, 1]``.

    This function is suitable both for ROI ``z`` in ``[0, 1]`` and for signed
    normalized window inputs in ``[-1, 1]``.  Squaring deliberately erases the
    sign in the latter case, matching the requested experimental mapping.
    """

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"unsupported transform method {method!r}; expected {SUPPORTED_METHODS}"
        )
    if epsilon is None or not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("log_squared requires finite epsilon > 0")

    data, mask, was_masked = _data_and_mask(normalized_values)
    output = np.full(data.shape, np.nan, dtype=np.float64)
    valid = (~mask) & np.isfinite(data)
    bounded = np.clip(data[valid], -1.0, 1.0)
    squared = bounded * bounded
    assert epsilon is not None
    # Algebraically identical to the documented difference-of-logs form,
    # but log1p is more accurate for small squared values.
    denominator = math.log1p(1.0 / epsilon)
    output[valid] = np.log1p(squared / epsilon) / denominator
    # Roundoff cannot materially escape the declared output domain, but this
    # clip gives downstream contracts an exact [0, 1] guarantee.
    output[valid] = np.clip(output[valid], 0.0, 1.0)
    return _restore_mask_type(output, mask, was_masked)


def transform_roi_intensity(
    values: ArrayLike,
    spec: ROITransformSpec,
) -> np.ndarray | np.ma.MaskedArray:
    """Normalize ROI intensities with a fixed scale and apply ``spec``."""

    z = roi_z_values(values, scale=spec.scale)
    return transform_bounded_squared(
        z,
        method=spec.method,
        epsilon=spec.epsilon,
    )


def _finite_range(data: np.ndarray, mask: np.ndarray) -> tuple[float | None, float | None]:
    finite = (~mask) & np.isfinite(data)
    if not np.any(finite):
        return None, None
    values = data[finite]
    return float(np.min(values)), float(np.max(values))


def audit_roi_transform(
    values: ArrayLike,
    spec: ROITransformSpec,
) -> dict[str, Any]:
    """Audit input hazards, clipping, mask preservation, and output bounds."""

    data, mask, _ = _data_and_mask(values)
    unmasked = ~mask
    finite = unmasked & np.isfinite(data)
    z_result = roi_z_values(values, scale=spec.scale)
    transformed = transform_roi_intensity(values, spec)
    z_data, z_mask, _ = _data_and_mask(z_result)
    out_data, out_mask, _ = _data_and_mask(transformed)
    input_min, input_max = _finite_range(data, mask)
    z_min, z_max = _finite_range(z_data, z_mask)
    output_min, output_max = _finite_range(out_data, out_mask)
    finite_output = (~out_mask) & np.isfinite(out_data)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "method": spec.method,
        "input_shape": list(data.shape),
        "total_slots": int(data.size),
        "masked_slots": int(np.count_nonzero(mask)),
        "unmasked_nan_slots": int(np.count_nonzero(unmasked & np.isnan(data))),
        "unmasked_positive_infinity_slots": int(
            np.count_nonzero(unmasked & np.isposinf(data))
        ),
        "unmasked_negative_infinity_slots": int(
            np.count_nonzero(unmasked & np.isneginf(data))
        ),
        "finite_unmasked_slots": int(np.count_nonzero(finite)),
        "finite_negative_slots": int(np.count_nonzero(finite & (data < 0.0))),
        "finite_zero_slots": int(np.count_nonzero(finite & (data == 0.0))),
        "finite_positive_slots": int(np.count_nonzero(finite & (data > 0.0))),
        "literal_log_zero_to_negative_infinity_slots": int(
            np.count_nonzero(finite & (data == 0.0))
        ),
        "negative_slots_clipped_to_z_zero": int(
            np.count_nonzero(finite & (data < 0.0))
        ),
        "above_fixed_scale_slots_clipped_to_z_one": int(
            np.count_nonzero(finite & (data > spec.scale))
        ),
        "input_finite_min": input_min,
        "input_finite_max": input_max,
        "z_finite_min": z_min,
        "z_finite_max": z_max,
        "z_exact_zero_slots": int(
            np.count_nonzero((~z_mask) & np.isfinite(z_data) & (z_data == 0.0))
        ),
        "z_exact_one_slots": int(
            np.count_nonzero((~z_mask) & np.isfinite(z_data) & (z_data == 1.0))
        ),
        "output_finite_min": output_min,
        "output_finite_max": output_max,
        "output_unmasked_nan_slots": int(
            np.count_nonzero((~out_mask) & np.isnan(out_data))
        ),
        "output_finite_slots": int(np.count_nonzero(finite_output)),
        "output_below_zero_slots": int(
            np.count_nonzero(finite_output & (out_data < 0.0))
        ),
        "output_above_one_slots": int(
            np.count_nonzero(finite_output & (out_data > 1.0))
        ),
        "mask_preserved_exactly": bool(np.array_equal(mask, out_mask)),
    }
    return _json_ready(audit)


def build_transform_provenance(
    spec: ROITransformSpec,
    *,
    scale_estimate: PooledScaleEstimate | None = None,
    audits: Mapping[str, Mapping[str, Any]] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete JSON-safe provenance record."""

    formula = (
        "[ln(z^2+epsilon)-ln(epsilon)]/"
        "[ln(1+epsilon)-ln(epsilon)]"
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "transform": spec.to_dict(),
        "formula": formula,
        "roi_z_definition": (
            "z=clip(max(physically_normalized_intensity,0)/"
            "fixed_pooled_scale,0,1)"
        ),
        "window_input_contract": (
            "signed dimensionless input is clipped to [-1,1] before squaring"
        ),
        "literal_raw_formula_used": False,
        "mask_nan_policy": {
            "masked_values": "remain masked",
            "unmasked_nan": "remains NaN",
            "positive_or_negative_infinity": "converted to NaN and audited",
            "mask_arrays": "metadata only; never transformed as intensity",
        },
        "numerical_safety": {
            "output_range": [0.0, 1.0],
            "zero_background_preserved": True,
        },
        "fixed_pooled_scale_estimate": (
            None if scale_estimate is None else scale_estimate.to_dict()
        ),
        "audits": {} if audits is None else dict(audits),
        "context": {} if context is None else dict(context),
    }
    return _json_ready(record)


def write_transform_provenance(
    path: str | Path,
    spec: ROITransformSpec,
    *,
    scale_estimate: PooledScaleEstimate | None = None,
    audits: Mapping[str, Mapping[str, Any]] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write a deterministic transform provenance JSON file."""

    record = build_transform_provenance(
        spec,
        scale_estimate=scale_estimate,
        audits=audits,
        context=context,
    )
    return _write_json_atomic(path, record)


def write_numerical_audit(
    path: str | Path,
    audit: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write a standalone JSON numerical-audit artifact."""

    record = {
        "schema_version": SCHEMA_VERSION,
        "audit": dict(audit),
        "context": {} if context is None else dict(context),
    }
    return _write_json_atomic(path, record)


def _write_json_atomic(path: str | Path, record: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_ready(record),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    return value


def _self_check() -> None:
    log_spec = make_roi_transform_spec(
        LOG_SQUARED,
        scale=10.0,
        noise_floor=1.0,
    )
    values = np.asarray([-5.0, 0.0, 5.0, 10.0, 20.0, np.nan])
    result = np.asarray(log_spec.transform(values))
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert 0.0 < result[2] < 1.0
    assert result[3] == 1.0
    assert result[4] == 1.0
    assert np.isnan(result[5])
    audit = log_spec.audit(values)
    assert audit["output_below_zero_slots"] == 0
    assert audit["output_above_one_slots"] == 0


if __name__ == "__main__":
    _self_check()
    print("nonlinear_intensity_preprocessing self-check: PASS")
