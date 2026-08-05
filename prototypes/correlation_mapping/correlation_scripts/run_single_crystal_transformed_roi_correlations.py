#!/usr/bin/env python3
"""Run transformed single-crystal ROI correlations on the curated 275 spots.

This runner changes only the scalar ROI intensity supplied to the existing
``analyze_single_tracks_across_frames`` implementation.  Peak identities,
locations, duplicate-observation collapse, frame ordering, min/max area
similarity, matrices, and plots remain owned by that existing implementation.

For every curated observation the raw TIFF pixels are selected with the formal
detector mask, curated frame mask, geometric ROI, and radial sideband.  The
sideband median is subtracted, positive excess is divided by TIFF exposure,
and all 90,398 ROI-pixel instances jointly fit one positive Q99.5 scale.  Each
pixel is transformed with ``nonlinear_intensity_preprocessing`` and the mean
transformed value becomes the observation's area feature.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nonlinear_intensity_preprocessing as nonlinear  # noqa: E402
import run_refinement_legacy_correlations as formal  # noqa: E402
from single_global_per_peak import (  # noqa: E402
    analyze_single_tracks_across_frames,
    branch_label,
    orientation_base,
)


DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "correlations" / "UOTe XRD Data Refinement"
DEFAULT_SINGLE_MANIFEST = (
    WORKSPACE_ROOT
    / "correlations"
    / "manifests"
    / "uote_single_crystal_uniform_v2_1_manifest.csv"
)
DEFAULT_SINGLE_RAW_ROOT = WORKSPACE_ROOT / "Data" / "Cell_29"
DEFAULT_SCALE_QUANTILE = 0.995

EXPECTED_OBSERVATIONS = 275
EXPECTED_MASKED_FRAMES = (0, 4, 5, 7, 10, 11, 13, 15, 17, 19, 21, 27)
EXPECTED_TRACKS = 75
EXPECTED_ROI_PIXEL_INSTANCES = 90_398
EXPECTED_POSITIVE_EXCESS_PIXEL_INSTANCES = 64_505
EXPECTED_ZERO_EXCESS_PIXEL_INSTANCES = 25_893
EXPECTED_Q995_POSITIVE_SCALE = 333.7071235707903
EXPECTED_SIDEBAND_NOISE_FLOOR = 0.1976865895529851
NORMAL_CONSISTENT_MAD_FACTOR = 1.4826


@dataclass(frozen=True)
class PixelObservation:
    """Formal geometric extraction for one curated ROI observation."""

    source: Mapping[str, str]
    frame: int
    raw_tiff: Path
    exposure_s: float
    background_median_counts: float
    positive_excess_rate: np.ndarray
    raw_excess_counts: float
    geometric_roi_pixels: int
    effective_roi_pixels: int
    detector_or_frame_masked_roi_pixels: int
    raw_zero_roi_pixels: int
    positive_excess_pixels: int
    zero_excess_pixels: int
    sideband_pixels: int
    sideband_median_counts: float
    sideband_mad_counts: float
    sideband_noise_counts_per_s_per_pixel: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=nonlinear.SUPPORTED_METHODS,
        help="Stable squared-intensity transform.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--single-manifest",
        type=Path,
        default=DEFAULT_SINGLE_MANIFEST,
    )
    parser.add_argument(
        "--single-raw-root",
        type=Path,
        default=DEFAULT_SINGLE_RAW_ROOT,
    )
    parser.add_argument(
        "--scale-quantile",
        type=float,
        default=DEFAULT_SCALE_QUANTILE,
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def _circular_delta_deg(left: np.ndarray, right: float) -> np.ndarray:
    return (np.asarray(left, dtype=float) - float(right) + 180.0) % 360.0 - 180.0


def geometric_masks(
    q_array: np.ndarray,
    chi_array_deg: np.ndarray,
    observation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact formal geometric ROI and radial sideband masks."""

    q = np.asarray(q_array, dtype=float)
    chi = np.asarray(chi_array_deg, dtype=float)
    if q.shape != chi.shape:
        raise ValueError("q_array and chi_array_deg must have identical shapes")
    q0 = float(observation["q"])
    azimuth = float(observation["azim_deg"])
    half_q = float(observation["halfwidth_q"])
    half_azimuth = float(observation["halfwidth_azim_deg"])
    if not all(np.isfinite(value) for value in (q0, azimuth, half_q, half_azimuth)):
        raise ValueError("observation geometry must be finite")
    if half_q <= 0.0 or half_azimuth <= 0.0:
        raise ValueError("observation half-widths must be strictly positive")
    radial = np.abs(q - q0)
    azimuthal = np.abs(_circular_delta_deg(chi, azimuth))
    geometric_roi = (radial <= half_q) & (azimuthal <= half_azimuth)
    geometric_sideband = (
        (radial > 1.15 * half_q)
        & (radial <= 2.15 * half_q)
        & (azimuthal <= half_azimuth)
    )
    return geometric_roi, geometric_sideband


def extract_observation_pixels(
    raw: np.ndarray,
    detector_mask: np.ndarray,
    frame_mask: np.ndarray,
    q_array: np.ndarray,
    chi_array_deg: np.ndarray,
    observation: Mapping[str, str],
    *,
    frame: int,
    raw_tiff: Path,
    exposure_s: float,
) -> PixelObservation:
    """Extract one observation's positive per-pixel excess-rate vector."""

    image = np.asarray(raw, dtype=np.float64)
    detector = np.asarray(detector_mask, dtype=bool)
    curated = np.asarray(frame_mask, dtype=bool)
    if image.shape != detector.shape or image.shape != curated.shape:
        raise ValueError("raw, detector_mask, and frame_mask must share a shape")
    if np.asarray(q_array).shape != image.shape or np.asarray(chi_array_deg).shape != image.shape:
        raise ValueError("geometry arrays must match the raw image shape")
    if not np.isfinite(exposure_s) or exposure_s <= 0.0:
        raise ValueError("exposure_s must be finite and strictly positive")

    geometric_roi, geometric_sideband = geometric_masks(
        q_array,
        chi_array_deg,
        observation,
    )
    valid_detector = (~detector) & np.isfinite(image) & (image >= 0.0)
    roi_mask = geometric_roi & (~curated) & valid_detector
    sideband_mask = geometric_sideband & valid_detector
    roi_pixels = image[roi_mask]
    sideband_pixels = image[sideband_mask]
    if roi_pixels.size == 0:
        raise ValueError(
            f"curated observation has no valid ROI pixels: frame={frame}, "
            f"obs_row={observation.get('obs_row')}"
        )
    if sideband_pixels.size < 5:
        raise ValueError(
            f"curated observation has fewer than five sideband pixels: "
            f"frame={frame}, obs_row={observation.get('obs_row')}"
        )

    background = float(np.median(sideband_pixels))
    positive_excess_counts = np.clip(roi_pixels - background, 0.0, None)
    positive_excess_rate = positive_excess_counts / float(exposure_s)
    sideband_median = float(np.median(sideband_pixels))
    sideband_mad = float(
        np.median(np.abs(sideband_pixels - sideband_median))
    )
    sideband_noise = float(
        NORMAL_CONSISTENT_MAD_FACTOR * sideband_mad / float(exposure_s)
    )
    return PixelObservation(
        source=dict(observation),
        frame=int(frame),
        raw_tiff=Path(raw_tiff),
        exposure_s=float(exposure_s),
        background_median_counts=background,
        positive_excess_rate=np.asarray(positive_excess_rate, dtype=np.float64),
        raw_excess_counts=float(np.sum(positive_excess_counts)),
        geometric_roi_pixels=int(np.count_nonzero(geometric_roi)),
        effective_roi_pixels=int(roi_pixels.size),
        detector_or_frame_masked_roi_pixels=int(
            np.count_nonzero(geometric_roi) - roi_pixels.size
        ),
        raw_zero_roi_pixels=int(np.count_nonzero(roi_pixels == 0.0)),
        positive_excess_pixels=int(np.count_nonzero(positive_excess_counts > 0.0)),
        zero_excess_pixels=int(np.count_nonzero(positive_excess_counts == 0.0)),
        sideband_pixels=int(sideband_pixels.size),
        sideband_median_counts=sideband_median,
        sideband_mad_counts=sideband_mad,
        sideband_noise_counts_per_s_per_pixel=sideband_noise,
    )


def _raw_tiff_path(
    metadata_row: Mapping[str, Any],
    raw_index: Mapping[str, Path],
) -> Path:
    basename = Path(
        str(metadata_row["original_filename"]).replace("\\", "/")
    ).name.lower()
    path = raw_index.get(basename)
    if path is None:
        raise FileNotFoundError(f"raw TIFF not found for {basename}")
    return path


def extract_curated_pixel_observations(
    kept_rows: Sequence[Mapping[str, str]],
    masked_dir: Path,
    raw_root: Path,
    metadata: Mapping[int, Mapping[str, Any]],
) -> tuple[list[PixelObservation], list[dict[str, Any]]]:
    """Extract all curated observations and frame-level mask reconciliation."""

    try:
        import pyFAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("pyFAI is required for transformed ROI extraction") from exc

    by_frame: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in kept_rows:
        frame = int(row["frame"])
        if frame not in metadata:
            raise ValueError(f"curated observation has no metadata: frame {frame}")
        by_frame[frame].append(row)

    raw_index = formal.raw_tiff_index(raw_root)
    ai = pyFAI.load(str(masked_dir / "_geometry.poni"))
    shape = (1043, 981)
    q_array = ai.qArray(shape) / 10.0
    chi_array = np.degrees(ai.center_array(shape, unit="chi_rad"))
    detector_mask = np.asarray(ai.detector.calc_mask(), dtype=bool)
    if q_array.shape != shape or chi_array.shape != shape or detector_mask.shape != shape:
        raise ValueError("formal detector geometry does not have shape (1043, 981)")

    extracted: list[PixelObservation] = []
    frame_qc: list[dict[str, Any]] = []
    for frame in sorted(by_frame):
        metadata_row = metadata[frame]
        raw_path = _raw_tiff_path(metadata_row, raw_index)
        with Image.open(raw_path) as image:
            exposure_s = formal.parse_tiff_exposure_seconds(image)
            raw = np.asarray(image, dtype=np.float64)
        if raw.shape != shape:
            raise ValueError(f"unexpected TIFF shape for {raw_path}: {raw.shape}")
        frame_mask_path = masked_dir / f"frame_{frame:04d}_mask.npy"
        frame_mask = np.load(frame_mask_path).astype(bool)
        if frame_mask.shape != shape:
            raise ValueError(f"unexpected frame mask shape: {frame_mask_path}")

        reconstructed = np.zeros(shape, dtype=bool)
        frame_pixel_instances = 0
        for row in by_frame[frame]:
            geometric_roi, _ = geometric_masks(q_array, chi_array, row)
            reconstructed |= geometric_roi
            item = extract_observation_pixels(
                raw,
                detector_mask,
                frame_mask,
                q_array,
                chi_array,
                row,
                frame=frame,
                raw_tiff=raw_path,
                exposure_s=exposure_s,
            )
            extracted.append(item)
            frame_pixel_instances += item.effective_roi_pixels

        kept_pixels = ~frame_mask
        union = reconstructed | kept_pixels
        intersection = reconstructed & kept_pixels
        frame_qc.append(
            {
                "frame": frame,
                "orientation": metadata_row["orientation"],
                "pressure_GPa": metadata_row["pressure_GPa"],
                "regions": len(by_frame[frame]),
                "mask_kept_pixels": int(np.count_nonzero(kept_pixels)),
                "reconstructed_pixels": int(np.count_nonzero(reconstructed)),
                "roi_mask_jaccard": float(
                    np.count_nonzero(intersection)
                    / max(np.count_nonzero(union), 1)
                ),
                "effective_roi_pixel_instances": frame_pixel_instances,
                "exposure_s": exposure_s,
                "raw_tiff": str(raw_path.resolve()),
                "frame_mask": str(frame_mask_path.resolve()),
            }
        )
    return extracted, frame_qc


def fit_transform_from_pixels(
    observations: Sequence[PixelObservation],
    method: nonlinear.TransformMethod,
    *,
    scale_quantile: float = DEFAULT_SCALE_QUANTILE,
) -> tuple[
    nonlinear.ROITransformSpec,
    nonlinear.PooledScaleEstimate,
    float,
]:
    """Fit one shared scale and the audited observation-median noise floor."""

    if not observations:
        raise ValueError("at least one pixel observation is required")
    noise_values = np.asarray(
        [item.sideband_noise_counts_per_s_per_pixel for item in observations],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(noise_values)) or np.any(noise_values < 0.0):
        raise ValueError("sideband noise estimates must be finite and nonnegative")
    noise_floor = float(np.median(noise_values))
    spec, estimate = nonlinear.fit_roi_transform(
        [item.positive_excess_rate for item in observations],
        method,
        noise_floor=noise_floor,
        scale_quantile=scale_quantile,
    )
    return spec, estimate, noise_floor


def build_transformed_rows(
    observations: Sequence[PixelObservation],
    spec: nonlinear.ROITransformSpec,
    metadata: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build analyzer-compatible rows and complete per-observation QC."""

    output_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for index, item in enumerate(observations):
        source = item.source
        meta = metadata[item.frame]
        transformed = np.asarray(spec.transform(item.positive_excess_rate), dtype=float)
        if transformed.shape != item.positive_excess_rate.shape:
            raise RuntimeError("transformed ROI pixels changed shape")
        if not np.all(np.isfinite(transformed)):
            raise RuntimeError("transformed valid ROI pixels must all be finite")
        transformed_mean = float(np.mean(transformed))
        raw_rate_mean = float(np.mean(item.positive_excess_rate))
        two_theta = float(
            formal.q_to_two_theta(
                float(source["q"]),
                formal.SINGLE_WAVELENGTH_A,
            )
        )
        status = (
            f"dimensionless_mean_of_{spec.method}_per_pixel_"
            "after_formal_mask_background_exposure"
        )
        row = {
            "dataset": "single_crystal",
            "orientation": meta["orientation"],
            "scan": meta["orientation"],
            "orientation_base": orientation_base(str(meta["orientation"])),
            "branch": branch_label(dict(meta)),
            "frame": item.frame,
            "pressure_GPa": float(meta["pressure_GPa"]),
            "whole_pattern_included": int(meta["included_whole_pattern"]),
            "whole_pattern_exclusion_reason": meta["exclusion_reason"],
            "track": int(source["track"]),
            "obs_row": int(source["obs_row"]),
            "q_A^-1": float(source["q"]),
            "d_A": float(source["d_A"]),
            "two_theta_deg": two_theta,
            "azim_deg": float(source["azim_deg"]),
            "halfwidth_q_A^-1": float(source["halfwidth_q"]),
            "halfwidth_azim_deg": float(source["halfwidth_azim_deg"]),
            "matched_d_A": (
                float(source["matched_d_A"])
                if str(source.get("matched_d_A", "")).strip()
                else np.nan
            ),
            "raw_background_counts": item.background_median_counts,
            "raw_excess_counts": item.raw_excess_counts,
            "exposure_s": item.exposure_s,
            "exposure_source": "TIFF ImageDescription Exposure_time",
            "effective_pixels": item.effective_roi_pixels,
            # This legacy key is intentionally preserved because the imported
            # analyzer consumes it.  Its value is now dimensionless.
            "normalized_intensity_counts_per_s_per_pixel": transformed_mean,
            "intensity_status": status,
            "raw_tiff": str(item.raw_tiff.resolve()),
            "untransformed_normalized_intensity_counts_per_s_per_pixel": (
                raw_rate_mean
            ),
            "transformed_intensity_dimensionless_mean": transformed_mean,
            "intensity_transform_method": spec.method,
            "intensity_transform_fixed_scale": spec.scale,
            "intensity_transform_noise_floor": spec.noise_floor,
            "intensity_transform_epsilon": spec.epsilon,
            "intensity_transform_scale_quantile": spec.scale_quantile,
        }
        output_rows.append(row)
        qc_rows.append(
            {
                "observation_index_0based": index,
                "frame": item.frame,
                "obs_row": int(source["obs_row"]),
                "track": int(source["track"]),
                "pressure_GPa": float(meta["pressure_GPa"]),
                "q_A^-1": float(source["q"]),
                "azim_deg": float(source["azim_deg"]),
                "geometric_roi_pixels": item.geometric_roi_pixels,
                "effective_roi_pixels": item.effective_roi_pixels,
                "detector_or_frame_masked_roi_pixels": (
                    item.detector_or_frame_masked_roi_pixels
                ),
                "raw_zero_roi_pixels": item.raw_zero_roi_pixels,
                "positive_excess_pixels": item.positive_excess_pixels,
                "zero_excess_pixels": item.zero_excess_pixels,
                "sideband_pixels": item.sideband_pixels,
                "background_median_counts": item.background_median_counts,
                "sideband_median_counts": item.sideband_median_counts,
                "sideband_mad_counts": item.sideband_mad_counts,
                "sideband_noise_counts_per_s_per_pixel": (
                    item.sideband_noise_counts_per_s_per_pixel
                ),
                "raw_excess_counts": item.raw_excess_counts,
                "raw_positive_excess_rate_mean": raw_rate_mean,
                "pixels_saturated_at_fixed_scale": int(
                    np.count_nonzero(item.positive_excess_rate > spec.scale)
                ),
                "transformed_intensity_dimensionless_mean": transformed_mean,
                "transform_method": spec.method,
                "raw_tiff": str(item.raw_tiff.resolve()),
            }
        )
    return output_rows, qc_rows


def validate_formal_contract(
    observations: Sequence[PixelObservation],
    spec: nonlinear.ROITransformSpec,
    estimate: nonlinear.PooledScaleEstimate,
    noise_floor: float,
) -> dict[str, Any]:
    """Fail closed if the authoritative 275-observation input has drifted."""

    frame_ids = tuple(sorted({item.frame for item in observations}))
    total_pixels = sum(item.effective_roi_pixels for item in observations)
    positive_pixels = sum(item.positive_excess_pixels for item in observations)
    zero_pixels = sum(item.zero_excess_pixels for item in observations)
    checks = {
        "observations_275": len(observations) == EXPECTED_OBSERVATIONS,
        "masked_frames_exact": frame_ids == EXPECTED_MASKED_FRAMES,
        "roi_pixel_instances_90398": total_pixels == EXPECTED_ROI_PIXEL_INSTANCES,
        "positive_excess_pixels_64505": (
            positive_pixels == EXPECTED_POSITIVE_EXCESS_PIXEL_INSTANCES
        ),
        "zero_excess_pixels_25893": (
            zero_pixels == EXPECTED_ZERO_EXCESS_PIXEL_INSTANCES
        ),
        "pooled_scale_count_matches": estimate.total_slots == total_pixels,
        "pooled_positive_count_matches": (
            estimate.positive_finite_slots == positive_pixels
        ),
        "q995_scale_matches_audit": math.isclose(
            spec.scale,
            EXPECTED_Q995_POSITIVE_SCALE,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "sideband_noise_matches_audit": math.isclose(
            noise_floor,
            EXPECTED_SIDEBAND_NOISE_FLOOR,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"formal transformed-ROI contract failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "frames": list(frame_ids),
        "observations": len(observations),
        "roi_pixel_instances": total_pixels,
        "positive_excess_pixel_instances": positive_pixels,
        "zero_excess_pixel_instances": zero_pixels,
        "fixed_positive_q995_scale": spec.scale,
        "sideband_noise_floor": noise_floor,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    fields = list(materialized[0]) if materialized else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            for row in materialized:
                writer.writerow(
                    {
                        key: ""
                        if isinstance(value, (float, np.floating))
                        and not np.isfinite(float(value))
                        else value
                        for key, value in row.items()
                    }
                )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_ready(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    data_root = args.data_root.expanduser().resolve()
    manifest_path = args.single_manifest.expanduser().resolve()
    raw_root = args.single_raw_root.expanduser().resolve()
    output_root = args.out_dir.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")

    single_root = data_root / "Single Crystal (Cell 29)"
    masked_dir = single_root / "Masked"
    patterns_dir = single_root / "Initial Reduction" / "patterns"
    kept_path = masked_dir / "kept_obs.csv"
    metadata, metadata_audit = formal.load_single_metadata(
        manifest_path,
        patterns_dir,
    )
    kept_rows = formal.read_csv(kept_path)
    pixel_observations, frame_qc = extract_curated_pixel_observations(
        kept_rows,
        masked_dir,
        raw_root,
        metadata,
    )
    spec, scale_estimate, noise_floor = fit_transform_from_pixels(
        pixel_observations,
        args.mode,
        scale_quantile=args.scale_quantile,
    )
    contract = validate_formal_contract(
        pixel_observations,
        spec,
        scale_estimate,
        noise_floor,
    )
    transformed_rows, observation_qc = build_transformed_rows(
        pixel_observations,
        spec,
        metadata,
    )

    output_root.mkdir(parents=True)
    preprocessing_root = output_root / "preprocessing"
    _write_csv(preprocessing_root / "single_frame_metadata.csv", metadata_audit)
    _write_csv(preprocessing_root / "frame_extraction_qc.csv", frame_qc)
    _write_csv(preprocessing_root / "observation_transform_qc.csv", observation_qc)
    pooled_pixels = np.concatenate(
        [item.positive_excess_rate for item in pixel_observations]
    )
    pooled_audit = spec.audit(pooled_pixels)
    nonlinear.write_numerical_audit(
        preprocessing_root / "pooled_pixel_numerical_audit.json",
        pooled_audit,
        context={"contract": contract},
    )
    nonlinear.write_transform_provenance(
        preprocessing_root / "TRANSFORM_PROVENANCE.json",
        spec,
        scale_estimate=scale_estimate,
        audits={"pooled_roi_pixel_instances": pooled_audit},
        context={
            "dataset": "single_crystal_curated_275_observations",
            "kept_observations": str(kept_path.resolve()),
            "manifest": str(manifest_path),
            "geometry": str((masked_dir / "_geometry.poni").resolve()),
            "raw_tiff_root": str(raw_root),
            "processing_order": [
                "formal detector and curated frame masks",
                "geometric ROI and radial sideband",
                "sideband median subtraction",
                "positive clip",
                "TIFF exposure division per pixel",
                "one positive Q99.5 scale over all ROI pixel instances",
                "bounded nonlinear transform per pixel",
                "mean transformed pixels per observation",
                "existing duplicate/frame median and min/max similarity",
            ],
            "downstream_analyzer": (
                "single_global_per_peak.analyze_single_tracks_across_frames"
            ),
            "legacy_intensity_field_now_dimensionless": (
                "normalized_intensity_counts_per_s_per_pixel"
            ),
            "contract": contract,
        },
    )

    analysis_root = output_root / "single_crystal" / "per_peak_all_frames"
    analysis_metrics = analyze_single_tracks_across_frames(
        analysis_root,
        transformed_rows,
        metadata,
        make_plots=not args.no_plots,
    )
    if (
        int(analysis_metrics["raw_observations"]) != EXPECTED_OBSERVATIONS
        or int(analysis_metrics["tracks"]) != EXPECTED_TRACKS
        or int(analysis_metrics["masked_frames"]) != len(EXPECTED_MASKED_FRAMES)
    ):
        raise RuntimeError("existing downstream analyzer returned unexpected scope")

    source_paths = [
        kept_path,
        manifest_path,
        masked_dir / "_geometry.poni",
        Path(__file__).resolve(),
        Path(nonlinear.__file__).resolve(),
        SCRIPT_DIR / "single_global_per_peak.py",
    ]
    source_hashes = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in source_paths
    ]
    _write_csv(preprocessing_root / "source_hashes.csv", source_hashes)
    metrics_without_rows = {
        key: value
        for key, value in analysis_metrics.items()
        if key != "summary_rows"
    }
    completion = {
        "status": "PASS",
        "completed": True,
        "mode": spec.method,
        "elapsed_seconds": time.time() - started,
        "output_root": str(output_root),
        "formal_contract": contract,
        "transform": spec.to_dict(),
        "scale_estimate": scale_estimate.to_dict(),
        "downstream_analysis": metrics_without_rows,
        "plots_generated": not args.no_plots,
        "location_semantics": (
            "unchanged frozen curated coordinates; independent of intensity transform"
        ),
    }
    _write_json(output_root / "RUN_COMPLETE.json", completion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

