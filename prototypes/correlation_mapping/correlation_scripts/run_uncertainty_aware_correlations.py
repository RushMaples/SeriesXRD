#!/usr/bin/env python3
"""Uncertainty-aware UOTe correlation reanalysis.

This runner deliberately leaves the verified 2026-07-16 legacy result intact.
It reuses the verified curated peak identities and processed XY channels, then
changes only the requested scoring/interpretation layer:

* peak position: FWHM-scaled Gaussian center-distance using per-observation width and
  centroid uncertainty (no fixed 0.06 degree cutoff),
* peak area: Gaussian log-area consistency using measurement uncertainty plus
  same-pressure repeatability,
* same-window: noise-whitened direct NCC with an explicit bounded shift,
* window-to-window: pressure-trajectory agreement and shared-change candidates,
* whole pattern: raw Pearson QC plus matched W/background and acquisition-order
  adjustment,
* presence/birth/death: separate state tables; unobserved curated peaks remain
  unknown and are never converted to absence.

The resulting run is a method-development reanalysis, not an independent
confirmatory phase-transition test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
import pandas as pd
import scipy
from PIL import Image
from scipy.stats import rankdata


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_uote_xy_handoff_correlations as legacy  # noqa: E402


DATE_TAG = "20260719"
PROFILE = "uncertainty-aware-correlation-v3-method-development"
SINGLE_WAVELENGTH_A = 0.4133
POWDER_WAVELENGTH_A = 0.3066
NEAR_GAP_GPA = 1.5
FAR_GAP_GPA = 15.0
EPS = 1.0e-12


@dataclass(frozen=True)
class SeriesData:
    label: str
    channel: str
    frames: list[legacy.Frame]
    scans: list[str]
    pressures: list[float]
    grid: np.ndarray
    normalized: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=root / "correlations/results/uote_refinement_legacy_global_per_peak_strict_lower_triangle_20260716",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=root / "correlations/UOTe XRD Data Refinement",
    )
    parser.add_argument(
        "--handoff-root",
        type=Path,
        default=root / "correlations/uote_xy_handoff 2",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-width-deg", type=float, default=5.0)
    parser.add_argument("--window-step-deg", type=float, default=1.0)
    parser.add_argument("--grid-step-deg", type=float, default=0.02)
    parser.add_argument("--max-shift-deg", type=float, default=0.12)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--overwrite-completed",
        action="store_true",
        help="Allow deterministic regeneration of an existing completed output directory.",
    )
    return parser.parse_args(argv)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_set(paths: Sequence[Path]) -> str:
    """Hash an ordered file set including paths, sizes, and content hashes."""
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def q_to_two_theta(q_a_inv: np.ndarray | float, wavelength_a: float) -> np.ndarray:
    q = np.asarray(q_a_inv, dtype=float)
    ratio = q * wavelength_a / (4.0 * np.pi)
    result = np.full(q.shape, np.nan, dtype=float)
    valid = np.isfinite(ratio) & (np.abs(ratio) < 1.0)
    result[valid] = np.degrees(2.0 * np.arcsin(ratio[valid]))
    return result


def q_width_to_two_theta(q_center: float, q_width: float, wavelength_a: float) -> float:
    if not (np.isfinite(q_center) and np.isfinite(q_width) and q_width > 0):
        return np.nan
    low = float(q_to_two_theta(max(q_center - q_width / 2.0, EPS), wavelength_a))
    high = float(q_to_two_theta(q_center + q_width / 2.0, wavelength_a))
    return high - low if np.isfinite(low) and np.isfinite(high) else np.nan


def q_error_to_two_theta(q_center: float, q_error: float, wavelength_a: float) -> float:
    if not (np.isfinite(q_center) and np.isfinite(q_error) and q_error >= 0):
        return np.nan
    step = max(q_error, 1.0e-7)
    low = float(q_to_two_theta(max(q_center - step, EPS), wavelength_a))
    high = float(q_to_two_theta(q_center + step, wavelength_a))
    derivative_error = abs(high - low) / 2.0
    return derivative_error * (q_error / step)


def circular_delta_deg(left: np.ndarray | float, right: float) -> np.ndarray:
    return (np.asarray(left, dtype=float) - float(right) + 180.0) % 360.0 - 180.0


def acquisition_signature(filename: str) -> str:
    """Extract coarse acquisition-mode tags from the source filename."""
    name = str(filename).replace("\\", "/").split("/")[-1]
    detector = re.search(r"(?:^|_)(D\d+[A-Za-z]*)(?:_|\.|$)", name, flags=re.IGNORECASE)
    tags = [detector.group(1).lower() if detector else "detector_unknown"]
    lowered = name.lower()
    if "longscan" in lowered:
        tags.append("longscan")
    if "dup" in lowered:
        tags.append("duplicate")
    return "+".join(tags)


def robust_scale(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan
    median = float(np.median(array))
    return float(1.4826 * np.median(np.abs(array - median)))


def auc_probability(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return np.nan
    combined = np.concatenate([positive, negative])
    ranks = rankdata(combined, method="average")
    rank_sum = float(np.sum(ranks[: len(positive)]))
    return (rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(keep) < 3:
        return np.nan
    left = left[keep]
    right = right[keep]
    if np.std(left) <= 0 or np.std(right) <= 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def bootstrap_median_ci(values: np.ndarray, count: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draw = rng.integers(0, len(values), size=(count, len(values)))
    medians = np.median(values[draw], axis=1)
    return tuple(float(item) for item in np.quantile(medians, [0.025, 0.975]))


def write_matrix_csv(path: Path, labels: Sequence[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(np.asarray(matrix, dtype=float), index=list(labels), columns=list(labels))
    frame.index.name = "row"
    frame.to_csv(path)


def strict_lower(matrix: np.ndarray) -> np.ndarray:
    shown = np.asarray(matrix, dtype=float).copy()
    shown[np.triu_indices_from(shown, k=0)] = np.nan
    return shown


def draw_strict_lower_axis(
    ax: Any,
    matrix: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
) -> Any:
    """Draw data, missing lower-triangle cells, and hidden cells distinctly."""
    values = np.asarray(matrix, dtype=float)
    lower = np.tril(np.ones(values.shape, dtype=bool), k=-1)
    shown = np.where(lower, values, np.nan)
    missing = np.where(lower & ~np.isfinite(values), 1.0, np.nan)

    missing_palette = ListedColormap(["#D9D9D9"])
    missing_palette.set_bad((0.0, 0.0, 0.0, 0.0))
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad((0.0, 0.0, 0.0, 0.0))
    ax.set_facecolor("white")
    ax.imshow(missing, vmin=0.0, vmax=1.0, cmap=missing_palette)
    return ax.imshow(shown, vmin=vmin, vmax=vmax, cmap=palette)


def plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = min(12.0, max(6.5, 0.38 * len(labels) + 2.0))
    fig, ax = plt.subplots(figsize=(size, size))
    image = draw_strict_lower_axis(ax, matrix, vmin, vmax, cmap)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.text(
        0.5, 0.008,
        "White = hidden diagonal/upper triangle; gray = NaN/unknown in the lower triangle.",
        ha="center", va="bottom", fontsize=7, color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 1.0))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_track_contact_sheet(
    path: Path,
    matrices: np.ndarray,
    track_ids: Sequence[int],
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
) -> None:
    """Write a compact overview while keeping full-resolution track plots separate."""
    count = len(track_ids)
    columns = 5 if count > 20 else min(5, max(1, count))
    rows = int(math.ceil(count / columns))
    fig, axes = plt.subplots(
        rows, columns, squeeze=False,
        figsize=(2.7 * columns, 2.35 * rows),
    )
    image = None
    for index, (track, matrix) in enumerate(zip(track_ids, matrices)):
        ax = axes.flat[index]
        image = draw_strict_lower_axis(ax, matrix, vmin, vmax, cmap)
        pair_count = int(np.count_nonzero(np.isfinite(strict_lower(matrix))))
        ax.set_title(f"track {int(track):03d} | pairs={pair_count}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[count:]:
        ax.axis("off")
    compact = rows <= 3
    top = 0.86 if compact else 0.975
    bottom = 0.10 if compact else 0.025
    fig.suptitle(title, fontsize=13, y=0.98 if compact else 0.995)
    if image is None:
        palette = plt.get_cmap(cmap)
        image = matplotlib.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=palette)
    colorbar_axis = fig.add_axes([0.945, bottom, 0.014, top - bottom])
    fig.colorbar(image, cax=colorbar_axis, label="similarity")
    fig.text(
        0.5, 0.01,
        "White = hidden diagonal/upper triangle; gray = NaN/unknown; full-size plots retain axis labels.",
        ha="center", va="bottom", fontsize=8, color="#555555",
    )
    fig.subplots_adjust(
        top=top, bottom=bottom, left=0.025, right=0.92,
        wspace=0.22, hspace=0.38,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def estimate_single_roi_uncertainties(
    observations: pd.DataFrame,
    masked_dir: Path,
) -> pd.DataFrame:
    """Estimate spot FWHM, centroid SE, and ROI-area SE from raw 2D pixels."""
    try:
        import pyFAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - explicit runtime requirement
        raise RuntimeError("pyFAI is required for the single-crystal ROI uncertainty pass") from exc

    ai = pyFAI.load(str(masked_dir / "_geometry.poni"))
    shape = (1043, 981)
    q_array = ai.qArray(shape) / 10.0
    chi_array = np.degrees(ai.center_array(shape, unit="chi_rad"))
    detector_mask = np.asarray(ai.detector.calc_mask(), dtype=bool)
    rows: list[dict[str, Any]] = []
    for frame, group in observations.groupby("frame", sort=True):
        raw_path = Path(str(group.iloc[0]["raw_tiff"]))
        with Image.open(raw_path) as image:
            raw = np.asarray(image, dtype=float)
        if raw.shape != shape:
            raise ValueError(f"Unexpected TIFF shape for {raw_path}: {raw.shape}")
        frame_mask = np.load(masked_dir / f"frame_{int(frame):04d}_mask.npy")
        valid_detector = (~detector_mask) & np.isfinite(raw) & (raw >= 0)
        for data in group.to_dict("records"):
            q0 = float(data["q_A^-1"])
            azim = float(data.get("azim_deg"))
            half_q = float(data.get("_15", data.get("halfwidth_q_A^-1")))
            half_azim = float(data.get("halfwidth_azim_deg"))
            radial = np.abs(q_array - q0)
            azimuthal = np.abs(circular_delta_deg(chi_array, azim))
            geometric_roi = (radial <= half_q) & (azimuthal <= half_azim)
            roi = geometric_roi & (~frame_mask) & valid_detector
            side = (
                (radial > half_q * 1.15)
                & (radial <= half_q * 2.15)
                & (azimuthal <= half_azim)
                & valid_detector
            )
            pixels = raw[roi]
            side_pixels = raw[side]
            if len(pixels) == 0:
                background = area_counts = area_se_counts = np.nan
                fwhm_q = centroid_se_q = weighted_center_q = np.nan
                effective_n = 0.0
            else:
                background = (
                    float(np.nanmedian(side_pixels))
                    if len(side_pixels) >= 5
                    else float(np.nanpercentile(pixels, 10))
                )
                weights = np.clip(pixels - background, 0.0, None)
                q_values = q_array[roi]
                total = float(np.sum(weights))
                if total > 0:
                    weighted_center_q = float(np.sum(weights * q_values) / total)
                    variance_q = float(np.sum(weights * (q_values - weighted_center_q) ** 2) / total)
                    sigma_q = math.sqrt(max(variance_q, 0.0))
                    fwhm_q = 2.354820045 * sigma_q
                    effective_n = float(total**2 / max(float(np.sum(weights**2)), EPS))
                    centroid_se_q = sigma_q / math.sqrt(max(effective_n, 1.0))
                else:
                    weighted_center_q = fwhm_q = centroid_se_q = np.nan
                    effective_n = 0.0
                area_counts = total
                noise = robust_scale(side_pixels)
                if not np.isfinite(noise) or noise <= 0:
                    noise = math.sqrt(max(float(np.nanmedian(pixels)), 1.0))
                background_se = 1.2533 * noise / math.sqrt(max(len(side_pixels), 1))
                area_se_counts = math.sqrt(len(pixels) * noise**2 + (len(pixels) * background_se) ** 2)
            rows.append(
                {
                    "frame": int(data["frame"]),
                    "track": int(data["track"]),
                    "obs_row": int(data["obs_row"]),
                    "profile_weighted_center_q_A^-1": weighted_center_q,
                    "profile_center_offset_q_A^-1": weighted_center_q - q0 if np.isfinite(weighted_center_q) else np.nan,
                    "fwhm_q_A^-1": fwhm_q,
                    "fwhm_two_theta_deg": q_width_to_two_theta(q0, fwhm_q, SINGLE_WAVELENGTH_A),
                    "centroid_se_q_A^-1": centroid_se_q,
                    "centroid_se_two_theta_deg": q_error_to_two_theta(q0, centroid_se_q, SINGLE_WAVELENGTH_A),
                    "profile_effective_n": effective_n,
                    "area_counts_recomputed": area_counts,
                    "area_se_counts": area_se_counts,
                    "integrated_roi_area_counts_per_s": (
                        area_counts / float(data["exposure_s"])
                        if np.isfinite(area_counts) and float(data["exposure_s"]) > 0
                        else np.nan
                    ),
                    "log_area_se": area_se_counts / area_counts if area_counts > 0 else np.nan,
                    "area_uncertainty_source": "raw_2D_ROI_sideband_robust_noise",
                    "location_uncertainty_source": "raw_2D_weighted_profile_FWHM_and_centroid_SE",
                    "sideband_pixels": int(len(side_pixels)),
                    "roi_pixels": int(len(pixels)),
                    "sideband_background_counts": background,
                }
            )
    enriched = observations.merge(pd.DataFrame(rows), on=["frame", "track", "obs_row"], how="left", validate="one_to_one")
    return enriched


def enrich_powder_uncertainties(observations: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Attach measured q width; area uncertainty remains repeatability-only."""
    source = pd.read_csv(source_path)
    by_frame = {int(frame): group for frame, group in source.groupby("frame", sort=False)}
    rows: list[dict[str, Any]] = []
    for item in observations.to_dict("records"):
        candidates = by_frame[int(item["frame"])]
        delta_q = np.abs(candidates["q"].to_numpy(float) - float(item["q_A^-1"]))
        delta_az = np.abs(circular_delta_deg(candidates["azim_deg"].to_numpy(float), float(item["azim_deg"])))
        keep = (delta_q <= 1.0e-4) & (delta_az <= 0.1)
        matched = candidates.loc[keep]
        if len(matched) != 1:
            raise ValueError(f"Powder uncertainty join expected one row for frame {item['frame']} track {item['track']}")
        source_row = matched.iloc[0]
        q_width = float(source_row["q_width"])
        snr = float(source_row["snr"])
        # This is a centroid-precision approximation only; peak-height SNR is
        # not silently re-labelled as an area standard error.
        centroid_se_q = q_width / (2.354820045 * max(snr, 1.0))
        rows.append(
            {
                "frame": int(item["frame"]),
                "track": int(item["track"]),
                "obs_row": int(item["obs_row"]),
                "fwhm_q_A^-1": q_width,
                "fwhm_two_theta_deg": q_width_to_two_theta(float(item["q_A^-1"]), q_width, POWDER_WAVELENGTH_A),
                "centroid_se_q_A^-1": centroid_se_q,
                "centroid_se_two_theta_deg": q_error_to_two_theta(float(item["q_A^-1"]), centroid_se_q, POWDER_WAVELENGTH_A),
                "profile_effective_n": np.nan,
                "area_se_counts": np.nan,
                "integrated_roi_area_counts_per_s": (
                    float(item["raw_excess_counts"]) / float(item["exposure_s"])
                    if float(item["exposure_s"]) > 0
                    else np.nan
                ),
                "log_area_se": np.nan,
                "area_uncertainty_source": "same_pressure_independent_scan_repeatability_only",
                "location_uncertainty_source": "source_q_width_plus_peak_height_SNR_centroid_approximation",
                "profile_width_interpretation": "source_q_width_assumed_FWHM_like_not_documented_fit_FWHM",
                "source_q_width_A^-1": q_width,
                "source_peak_height_snr": snr,
            }
        )
    return observations.merge(pd.DataFrame(rows), on=["frame", "track", "obs_row"], how="left", validate="one_to_one")


def collapse_peak_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate spots for one track/frame without inventing observations."""
    grouping = ["dataset", "scan", "track", "pressure_GPa", "frame"]
    rows: list[dict[str, Any]] = []
    for key, group in observations.groupby(grouping, sort=True, dropna=False):
        integrated_area = group["integrated_roi_area_counts_per_s"].to_numpy(float)
        log_area = np.log(integrated_area)
        log_area = log_area[np.isfinite(log_area)]
        center = group["two_theta_deg"].to_numpy(float)
        center = center[np.isfinite(center)]
        center_se = group["centroid_se_two_theta_deg"].to_numpy(float)
        center_se = center_se[np.isfinite(center_se) & (center_se >= 0)]
        fwhm = group["fwhm_two_theta_deg"].to_numpy(float)
        fwhm = fwhm[np.isfinite(fwhm) & (fwhm > 0)]
        log_area_se = group["log_area_se"].to_numpy(float)
        log_area_se = log_area_se[np.isfinite(log_area_se) & (log_area_se >= 0)]
        n = len(group)
        center_duplicate_component = robust_scale(center) / math.sqrt(max(n, 1)) if len(center) > 1 else 0.0
        log_area_duplicate_component = robust_scale(log_area) / math.sqrt(max(n, 1)) if len(log_area) > 1 else 0.0
        center_measurement = float(np.nanmedian(center_se)) if len(center_se) else np.nan
        area_measurement = float(np.nanmedian(log_area_se)) if len(log_area_se) else np.nan
        center_total = (
            math.sqrt(center_measurement**2 + center_duplicate_component**2)
            if np.isfinite(center_measurement)
            else center_duplicate_component if center_duplicate_component > 0 else np.nan
        )
        area_total = (
            math.sqrt(area_measurement**2 + log_area_duplicate_component**2)
            if np.isfinite(area_measurement)
            else log_area_duplicate_component if log_area_duplicate_component > 0 else np.nan
        )
        first = group.iloc[0]
        row = {
            "dataset": key[0],
            "scan": key[1],
            "track": int(key[2]),
            "pressure_GPa": float(key[3]),
            "frame": int(key[4]),
            "orientation": first.get("orientation", "not_applicable"),
            "orientation_base": first.get("orientation_base", "not_applicable"),
            "branch": first.get("branch", first.get("scan", "not_applicable")),
            "n_observations": n,
            "two_theta_deg": float(np.median(center)) if len(center) else np.nan,
            "centroid_se_two_theta_deg": center_total,
            "fwhm_two_theta_deg": float(np.median(fwhm)) if len(fwhm) else np.nan,
            "log_area": float(np.median(log_area)) if len(log_area) else np.nan,
            "integrated_roi_area_counts_per_s": (
                float(np.exp(np.median(log_area))) if len(log_area) else np.nan
            ),
            "area_metric": "integrated_background_subtracted_ROI_excess_counts_per_second",
            "log_area_se": area_total,
            "location_uncertainty_source": ";".join(sorted(set(group["location_uncertainty_source"].astype(str)))),
            "area_uncertainty_source": ";".join(sorted(set(group["area_uncertainty_source"].astype(str)))),
            "duplicate_observation_flag": int(n > 1),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def repeatability_pairs(features: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Build same-pressure independent-observation pair differences."""
    keys = ["track", "pressure_GPa"]
    if features["dataset"].iloc[0] == "single_crystal":
        keys.append("orientation_base")
    rows: list[dict[str, Any]] = []
    for group_key, group in features.groupby(keys, sort=True, dropna=False):
        records = group.to_dict("records")
        for i, left in enumerate(records):
            for right in records[:i]:
                left_value = float(left[value_column])
                right_value = float(right[value_column])
                if not (np.isfinite(left_value) and np.isfinite(right_value)):
                    continue
                rows.append(
                    {
                        "group": str(group_key),
                        "track": int(left["track"]),
                        "pressure_GPa": float(left["pressure_GPa"]),
                        "frame_a": int(left["frame"]),
                        "frame_b": int(right["frame"]),
                        "absolute_difference": abs(left_value - right_value),
                        "measurement_variance": (
                            float(left.get(f"{value_column}_se", np.nan)) ** 2
                            + float(right.get(f"{value_column}_se", np.nan)) ** 2
                        ),
                    }
                )
    return pd.DataFrame(rows)


def estimate_repeatability(
    features: pd.DataFrame,
    value_column: str,
    se_column: str,
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate pair scatter with equal-weight track×pressure groups and track bootstrap."""
    keys = ["track", "pressure_GPa"]
    if features["dataset"].iloc[0] == "single_crystal":
        keys.append("orientation_base")
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for group_key, group in features.groupby(keys, sort=True, dropna=False):
        records = [
            record for record in group.to_dict("records")
            if np.isfinite(float(record[value_column]))
        ]
        group_pair_rows: list[dict[str, Any]] = []
        for i, left in enumerate(records):
            for right in records[:i]:
                left_value = float(left[value_column])
                right_value = float(right[value_column])
                u_left = float(left.get(se_column, np.nan))
                u_right = float(right.get(se_column, np.nan))
                measurement_variance = (
                    (u_left**2 if np.isfinite(u_left) else 0.0)
                    + (u_right**2 if np.isfinite(u_right) else 0.0)
                )
                group_pair_rows.append(
                    {
                        "group": str(group_key), "track": int(left["track"]),
                        "pressure_GPa": float(left["pressure_GPa"]),
                        "frame_a": int(left["frame"]), "frame_b": int(right["frame"]),
                        "absolute_difference": abs(left_value - right_value),
                        "measurement_variance": measurement_variance,
                    }
                )
        if group_pair_rows:
            differences = np.asarray([row["absolute_difference"] for row in group_pair_rows], dtype=float)
            variances = np.asarray([row["measurement_variance"] for row in group_pair_rows], dtype=float)
            group_q68 = float(np.quantile(differences, 0.682689492))
            group_measurement_variance = float(np.median(variances))
            group_tau = math.sqrt(max(group_q68**2 - group_measurement_variance, 0.0))
            group_summary = {
                "group_observations": len(records),
                "group_pairs": len(group_pair_rows),
                "group_absolute_difference_q68": group_q68,
                "group_median_measurement_variance": group_measurement_variance,
                "group_extra_pair_tau": group_tau,
            }
            group_rows.append(
                {
                    "group": str(group_key), "track": int(records[0]["track"]),
                    "pressure_GPa": float(records[0]["pressure_GPa"]), **group_summary,
                }
            )
            for row in group_pair_rows:
                rows.append({**row, **group_summary})
    pair_frame = pd.DataFrame(rows)
    group_frame = pd.DataFrame(group_rows)
    if len(group_frame):
        track_tau = group_frame.groupby("track")["group_extra_pair_tau"].median()
        tau_pair = float(np.median(track_tau.to_numpy(float)))
        q68 = float(np.median(group_frame["group_absolute_difference_q68"]))
        median_measurement_variance = float(np.median(group_frame["group_median_measurement_variance"]))
        track_values = track_tau.to_numpy(float)
        if len(track_values) > 1:
            draw = rng.integers(0, len(track_values), size=(bootstrap_resamples, len(track_values)))
            boot_tau = np.median(track_values[draw], axis=1)
            ci = [float(item) for item in np.quantile(boot_tau, [0.025, 0.975])]
        else:
            ci = [tau_pair, tau_pair]
    else:
        q68 = median_measurement_variance = tau_pair = np.nan
        ci = [np.nan, np.nan]
    return (
        {
            "value_column": value_column,
            "se_column": se_column,
            "same_pressure_pairs": int(len(pair_frame)),
            "same_pressure_groups": int(len(group_frame)),
            "independent_tracks": int(group_frame["track"].nunique()) if len(group_frame) else 0,
            "absolute_difference_q68": q68,
            "median_pair_measurement_variance": median_measurement_variance,
            "extra_pair_repeatability_tau": tau_pair,
            "bootstrap_ci95": ci,
            "bootstrap_resamples": bootstrap_resamples,
            "aggregation": "median group q68; equal-weight track median; bootstrap independent tracks",
        },
        pair_frame,
    )


def classify_pair_values(values: np.ndarray, pressures: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    near: list[float] = []
    far: list[float] = []
    for i, p_i in enumerate(pressures):
        for j, p_j in enumerate(pressures[:i]):
            current = np.asarray(values[..., i, j]).reshape(-1)
            current = current[np.isfinite(current)]
            gap = abs(float(p_i) - float(p_j))
            if gap <= NEAR_GAP_GPA:
                near.extend(current.tolist())
            elif gap >= FAR_GAP_GPA:
                far.extend(current.tolist())
    return np.asarray(near), np.asarray(far)


def analyze_peak_dataset(
    out_dir: Path,
    heatmap_root: Path,
    observations: pd.DataFrame,
    features: pd.DataFrame,
    frame_registry: pd.DataFrame,
    bootstrap_resamples: int,
    rng: np.random.Generator,
    make_plots: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_root.mkdir(parents=True, exist_ok=True)
    observations.to_csv(out_dir / "observation_uncertainties.csv", index=False)
    features.to_csv(out_dir / "frame_track_features.csv", index=False)

    area_repeat, area_pairs = estimate_repeatability(
        features, "log_area", "log_area_se", bootstrap_resamples, rng
    )
    center_repeat, center_pairs = estimate_repeatability(
        features, "two_theta_deg", "centroid_se_two_theta_deg", bootstrap_resamples, rng
    )
    area_pairs.to_csv(out_dir / "area_same_pressure_repeatability_pairs.csv", index=False)
    center_pairs.to_csv(out_dir / "location_same_pressure_repeatability_pairs.csv", index=False)
    if len(area_pairs):
        area_pairs.drop_duplicates("group")[
            [
                "group", "track", "pressure_GPa", "group_observations", "group_pairs",
                "group_absolute_difference_q68", "group_median_measurement_variance",
                "group_extra_pair_tau",
            ]
        ].to_csv(out_dir / "area_same_pressure_repeatability_groups.csv", index=False)
    if len(center_pairs):
        center_pairs.drop_duplicates("group")[
            [
                "group", "track", "pressure_GPa", "group_observations", "group_pairs",
                "group_absolute_difference_q68", "group_median_measurement_variance",
                "group_extra_pair_tau",
            ]
        ].to_csv(out_dir / "location_same_pressure_repeatability_groups.csv", index=False)
    write_json(out_dir / "uncertainty_calibration.json", {"area": area_repeat, "location": center_repeat})
    tau_area_raw = float(area_repeat["extra_pair_repeatability_tau"])
    tau_center_raw = float(center_repeat["extra_pair_repeatability_tau"])
    tau_area = tau_area_raw if np.isfinite(tau_area_raw) else 0.0
    tau_center = tau_center_raw if np.isfinite(tau_center_raw) else 0.0
    area_calibrated = int(area_repeat["same_pressure_pairs"]) > 0
    center_calibrated = int(center_repeat["same_pressure_pairs"]) > 0
    area_status = (
        "track_cluster_calibrated_with_same_pressure_repeatability_secondary"
        if area_calibrated
        else "measurement_noise_only_uncalibrated_no_same_pressure_repeats"
    )
    center_status = (
        "track_cluster_calibrated_with_same_pressure_repeatability_secondary"
        if center_calibrated
        else "measurement_noise_only_uncalibrated_no_same_pressure_repeats"
    )

    dataset = str(features["dataset"].iloc[0])
    profile_width_status = (
        "raw_2D_second_moment_FWHM"
        if dataset == "single_crystal"
        else "source_q_width_FWHM_like_not_documented_as_fitted_FWHM"
    )
    tracks = sorted(features["track"].astype(int).unique())
    track_index = {track: index for index, track in enumerate(tracks)}
    if dataset == "single_crystal":
        registry = frame_registry.reset_index(drop=True)
        axes = registry["machine_label"].astype(str).tolist()
        axis_pressures = registry["pressure_GPa"].to_numpy(float)
        axis_keys = registry["frame"].astype(int).tolist()
        scans = ["global_frame_axis"]
        shape = (len(tracks), 1, len(axes), len(axes))
        lookup_axis = {int(frame): index for index, frame in enumerate(axis_keys)}
        lookup_scan = {"global_frame_axis": 0}
    else:
        registry = frame_registry[frame_registry["included"].astype(int) == 1].copy()
        pressures = sorted(registry["pressure_GPa"].astype(float).unique())
        scans = sorted(registry["scan"].astype(str).unique())
        axes = [f"{value:g}" for value in pressures]
        axis_pressures = np.asarray(pressures, dtype=float)
        lookup_axis = {float(value): index for index, value in enumerate(pressures)}
        lookup_scan = {scan: index for index, scan in enumerate(scans)}
        shape = (len(tracks), len(scans), len(axes), len(axes))

    location = np.full(shape, np.nan, dtype=np.float32)
    centroid_consistency = np.full(shape, np.nan, dtype=np.float32)
    area = np.full(shape, np.nan, dtype=np.float32)
    location_pair_rows: list[dict[str, Any]] = []
    area_pair_rows: list[dict[str, Any]] = []
    group_keys = ["track", "scan"] if dataset != "single_crystal" else ["track"]
    for group_key, group in features.groupby(group_keys, sort=True):
        group = group.sort_values(["pressure_GPa", "frame"]).reset_index(drop=True)
        track = int(group.iloc[0]["track"])
        scan = str(group.iloc[0]["scan"]) if dataset != "single_crystal" else "global_frame_axis"
        for i in range(len(group)):
            high = group.iloc[i]
            for j in range(i):
                low = group.iloc[j]
                if dataset == "single_crystal":
                    index_i = lookup_axis[int(high["frame"])]
                    index_j = lookup_axis[int(low["frame"])]
                else:
                    index_i = lookup_axis[float(high["pressure_GPa"])]
                    index_j = lookup_axis[float(low["pressure_GPa"])]
                if index_i == index_j:
                    continue
                if index_i < index_j:
                    index_i, index_j = index_j, index_i
                    high, low = low, high
                delta = abs(float(high["two_theta_deg"]) - float(low["two_theta_deg"]))
                sigma_high = float(high["fwhm_two_theta_deg"]) / 2.354820045
                sigma_low = float(low["fwhm_two_theta_deg"]) / 2.354820045
                u_high = float(high["centroid_se_two_theta_deg"])
                u_low = float(low["centroid_se_two_theta_deg"])
                denom_profile = sigma_high**2 + sigma_low**2
                if np.isfinite(u_high):
                    denom_profile += u_high**2
                if np.isfinite(u_low):
                    denom_profile += u_low**2
                profile_similarity = math.exp(-0.5 * delta**2 / max(denom_profile, EPS))
                centroid_denom = (
                    (u_high**2 if np.isfinite(u_high) else 0.0)
                    + (u_low**2 if np.isfinite(u_low) else 0.0)
                    + tau_center**2
                )
                centroid_z = delta / math.sqrt(max(centroid_denom, EPS))
                centroid_similarity = math.exp(-0.5 * centroid_z**2)
                slot = (track_index[track], lookup_scan[scan], index_i, index_j)
                location[slot] = profile_similarity
                centroid_consistency[slot] = centroid_similarity
                location_pair_rows.append(
                    {
                        "dataset": dataset,
                        "track": track,
                        "scan": scan,
                        "frame_high": int(high["frame"]),
                        "frame_low": int(low["frame"]),
                        "pressure_high_GPa": float(high["pressure_GPa"]),
                        "pressure_low_GPa": float(low["pressure_GPa"]),
                        "pressure_gap_GPa": abs(float(high["pressure_GPa"]) - float(low["pressure_GPa"])),
                        "delta_two_theta_deg": delta,
                        "profile_sigma_pair_deg": math.sqrt(max(denom_profile, 0.0)),
                        "fwhm_scaled_center_distance_similarity": profile_similarity,
                        "centroid_repeatability_tau_deg": tau_center,
                        "centroid_shift_z": centroid_z,
                            "centroid_consistency_similarity": centroid_similarity,
                            "centroid_consistency_status": center_status,
                    }
                )
                log_high = float(high["log_area"])
                log_low = float(low["log_area"])
                if (
                    dataset == "single_crystal"
                    and str(high["orientation_base"]) != str(low["orientation_base"])
                ):
                    # Intensity changes with crystal orientation even at fixed
                    # structure; do not call that a pressure-driven area change.
                    continue
                if np.isfinite(log_high) and np.isfinite(log_low):
                    area_u_high = float(high["log_area_se"])
                    area_u_low = float(low["log_area_se"])
                    area_variance = tau_area**2
                    if np.isfinite(area_u_high):
                        area_variance += area_u_high**2
                    if np.isfinite(area_u_low):
                        area_variance += area_u_low**2
                    if area_variance <= EPS:
                        continue
                    area_z = abs(log_high - log_low) / math.sqrt(area_variance)
                    area_similarity = math.exp(-0.5 * area_z**2)
                    area[slot] = area_similarity
                    area_pair_rows.append(
                        {
                            "dataset": dataset,
                            "track": track,
                            "scan": scan,
                            "frame_high": int(high["frame"]),
                            "frame_low": int(low["frame"]),
                            "pressure_high_GPa": float(high["pressure_GPa"]),
                            "pressure_low_GPa": float(low["pressure_GPa"]),
                            "pressure_gap_GPa": abs(float(high["pressure_GPa"]) - float(low["pressure_GPa"])),
                            "orientation_high": str(high["orientation_base"]),
                            "orientation_low": str(low["orientation_base"]),
                            "absolute_log_area_difference": abs(log_high - log_low),
                            "log_area_pair_sigma": math.sqrt(area_variance),
                            "log_area_z": area_z,
                            "area_similarity": area_similarity,
                            "area_similarity_status": area_status,
                            "repeatability_tau_log_area": tau_area,
                            "geometry_rule": "same_orientation_only_for_single_crystal",
                        }
                    )

    aggregate_location = legacy.nanmedian(location, axis=1)
    aggregate_centroid = legacy.nanmedian(centroid_consistency, axis=1)
    aggregate_area = legacy.nanmedian(area, axis=1)
    np.savez_compressed(
        out_dir / "per_track_matrices.npz",
        track_ids=np.asarray(tracks),
        axis_labels=np.asarray(axes),
        axis_pressure_gpa=axis_pressures,
        scan_names=np.asarray(scans),
        area_calibrated=np.asarray(area_calibrated),
        centroid_consistency_calibrated=np.asarray(center_calibrated),
        location_profile_by_scan=location,
        centroid_consistency_by_scan=centroid_consistency,
        area_by_scan=area,
        location_profile_aggregate=aggregate_location,
        centroid_consistency_aggregate=aggregate_centroid,
        area_aggregate=aggregate_area,
    )
    pd.DataFrame(location_pair_rows).to_csv(out_dir / "location_pair_scores.csv", index=False)
    pd.DataFrame(area_pair_rows).to_csv(out_dir / "area_pair_scores.csv", index=False)
    overall_location = legacy.nanmedian(aggregate_location, axis=0)
    overall_centroid = legacy.nanmedian(aggregate_centroid, axis=0)
    overall_area = legacy.nanmedian(aggregate_area, axis=0)
    write_matrix_csv(out_dir / "aggregate_location_profile_matrix.csv", axes, overall_location)
    write_matrix_csv(out_dir / "aggregate_centroid_consistency_matrix.csv", axes, overall_centroid)
    write_matrix_csv(out_dir / "aggregate_area_matrix.csv", axes, overall_area)

    heatmap_index_rows: list[dict[str, Any]] = []
    for track in tracks:
        index = track_index[track]
        track_features = features[features["track"] == track]
        location_matrix = aggregate_location[index]
        area_matrix = aggregate_area[index]
        location_pairs = int(np.count_nonzero(np.isfinite(strict_lower(location_matrix))))
        area_pairs = int(np.count_nonzero(np.isfinite(strict_lower(area_matrix))))
        frames = int(track_features["frame"].nunique())
        observations_count = int(track_features["n_observations"].sum())
        median_two_theta = float(track_features["two_theta_deg"].median())
        stem = f"track_{int(track):03d}"
        location_matrix_path = heatmap_root / "location_matrices" / f"{stem}.csv"
        area_matrix_path = heatmap_root / "area_matrices" / f"{stem}.csv"
        location_heatmap_path = heatmap_root / "location" / f"{stem}.png"
        area_heatmap_path = heatmap_root / "area" / f"{stem}.png"
        write_matrix_csv(location_matrix_path, axes, location_matrix)
        write_matrix_csv(area_matrix_path, axes, area_matrix)
        if make_plots:
            plot_heatmap(
                location_heatmap_path,
                location_matrix,
                axes,
                (
                    f"{dataset} track {int(track):03d}: FWHM-scaled peak-position similarity\n"
                    f"median 2θ={median_two_theta:.3f}° | frames={frames} | comparable pairs={location_pairs}"
                ),
                0.0,
                1.0,
                "viridis",
                "Gaussian center-distance similarity",
            )
            plot_heatmap(
                area_heatmap_path,
                area_matrix,
                axes,
                (
                    f"{dataset} track {int(track):03d}: log-area similarity "
                    f"({'repeatability-calibrated/secondary' if area_calibrated else 'uncalibrated/secondary'})\n"
                    f"median 2θ={median_two_theta:.3f}° | frames={frames} | comparable pairs={area_pairs}"
                ),
                0.0,
                1.0,
                "viridis",
                "Gaussian log-area similarity",
            )
        heatmap_index_rows.append(
            {
                "dataset": dataset,
                "track": int(track),
                "median_two_theta_deg": median_two_theta,
                "frames": frames,
                "observations": observations_count,
                "pressure_min_GPa": float(track_features["pressure_GPa"].min()),
                "pressure_max_GPa": float(track_features["pressure_GPa"].max()),
                "location_comparable_pairs": location_pairs,
                "area_comparable_pairs": area_pairs,
                "location_heatmap": f"{dataset}/location/{stem}.png",
                "area_heatmap": f"{dataset}/area/{stem}.png",
                "location_matrix": f"{dataset}/location_matrices/{stem}.csv",
                "area_matrix": f"{dataset}/area_matrices/{stem}.csv",
                "location_width_status": profile_width_status,
                "area_status": area_status,
                "missing_semantics": "white_hidden_triangle_gray_NaN_unknown_not_zero",
            }
        )
    pd.DataFrame(heatmap_index_rows).to_csv(heatmap_root / "index.csv", index=False)

    if make_plots:
        plot_heatmap(
            out_dir / "aggregate_location_profile_heatmap.png",
            overall_location,
            axes,
            f"{dataset}: FWHM-scaled Gaussian peak-position similarity",
            0.0,
            1.0,
            "viridis",
            "FWHM-scaled Gaussian center-distance similarity",
        )
        plot_heatmap(
            out_dir / "aggregate_area_heatmap.png",
            overall_area,
            axes,
            (
                f"{dataset}: uncertainty-aware log-area similarity"
                if area_calibrated
                else f"{dataset}: measurement-noise-only area similarity (uncalibrated)"
            ),
            0.0,
            1.0,
            "viridis",
            "Gaussian log-area similarity",
        )
        plot_track_contact_sheet(
            heatmap_root.parent / f"{dataset}_location_all_tracks.png",
            aggregate_location,
            tracks,
            f"{dataset}: per-peak FWHM-scaled location similarity",
        )
        plot_track_contact_sheet(
            heatmap_root.parent / f"{dataset}_area_all_tracks.png",
            aggregate_area,
            tracks,
            (
                f"{dataset}: per-peak log-area similarity "
                f"({'repeatability-calibrated/secondary' if area_calibrated else 'uncalibrated/secondary'})"
            ),
        )

    loc_near, loc_far = classify_pair_values(location, axis_pressures)
    area_near, area_far = classify_pair_values(area, axis_pressures)
    summary_rows: list[dict[str, Any]] = []
    for track in tracks:
        index = track_index[track]
        finite_location = np.count_nonzero(np.isfinite(location[index]))
        finite_area = np.count_nonzero(np.isfinite(area[index]))
        track_features = features[features["track"] == track]
        summary_rows.append(
            {
                "dataset": dataset,
                "track": track,
                "frames": int(track_features["frame"].nunique()),
                "observations": int(track_features["n_observations"].sum()),
                "pressure_min_GPa": float(track_features["pressure_GPa"].min()),
                "pressure_max_GPa": float(track_features["pressure_GPa"].max()),
                "median_fwhm_two_theta_deg": float(track_features["fwhm_two_theta_deg"].median()),
                "median_centroid_se_two_theta_deg": float(track_features["centroid_se_two_theta_deg"].median()),
                "location_pair_scores": int(finite_location),
                "area_pair_scores": int(finite_area),
                "missing_semantics": "NaN_unknown_not_zero",
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "track_summary.csv", index=False)
    area_near_median_raw = float(np.nanmedian(area_near)) if len(area_near) else np.nan
    area_far_median_raw = float(np.nanmedian(area_far)) if len(area_far) else np.nan
    area_auc_raw = auc_probability(area_near, area_far)
    return {
        "dataset": dataset,
        "raw_observations": int(len(observations)),
        "collapsed_features": int(len(features)),
        "tracks": int(len(tracks)),
        "axes": int(len(axes)),
        "scans": int(len(scans)),
        "location_pair_scores": int(np.count_nonzero(np.isfinite(location))),
        "area_pair_scores": int(np.count_nonzero(np.isfinite(area))),
        "median_fwhm_two_theta_deg": float(features["fwhm_two_theta_deg"].median()),
        "median_centroid_se_two_theta_deg": float(features["centroid_se_two_theta_deg"].median()),
        "profile_width_status": profile_width_status,
        "location_near_median": float(np.nanmedian(loc_near)) if len(loc_near) else np.nan,
        "location_far_median": float(np.nanmedian(loc_far)) if len(loc_far) else np.nan,
        "location_near_far_auc": auc_probability(loc_near, loc_far),
        "area_near_median": area_near_median_raw if area_calibrated else np.nan,
        "area_far_median": area_far_median_raw if area_calibrated else np.nan,
        "area_near_far_auc": area_auc_raw if area_calibrated else np.nan,
        "area_measurement_only_near_median": area_near_median_raw,
        "area_measurement_only_far_median": area_far_median_raw,
        "area_measurement_only_near_far_auc": area_auc_raw,
        "area_similarity_status": area_status,
        "centroid_consistency_status": center_status,
        "area_repeatability": area_repeat,
        "location_repeatability": center_repeat,
        "strict_lower_triangle_display": True,
        "exact_matrices_store_nan_missing": True,
        "per_track_heatmap_export": {
            "tracks": int(len(tracks)),
            "location_png": int(len(tracks)) if make_plots else 0,
            "area_png": int(len(tracks)) if make_plots else 0,
            "location_csv": int(len(tracks)),
            "area_csv": int(len(tracks)),
        },
    }


def write_per_peak_heatmap_landing(root: Path) -> None:
    """Create one discoverable index for all per-track peak heatmaps."""
    root.mkdir(parents=True, exist_ok=True)
    frames = [
        pd.read_csv(root / "single_crystal/index.csv"),
        pd.read_csv(root / "powder/index.csv"),
    ]
    index = pd.concat(frames, ignore_index=True)
    index.to_csv(root / "per_peak_heatmap_index.csv", index=False)
    single_tracks = int((index["dataset"] == "single_crystal").sum())
    powder_tracks = int((index["dataset"] == "powder").sum())
    readme = f"""# Per-peak location and area heatmaps

这里是逐峰（每个 track 独立）的 location 与 area heatmap，不是跨所有峰取中位数后的 aggregate 图。

- 单晶：{single_tracks} tracks，每个 track 都有 location 和 area PNG/CSV。
- 粉末：{powder_tracks} tracks，每个 track 都有 location 和 area PNG/CSV。
- 总览图：`single_crystal_location_all_tracks.png`、`single_crystal_area_all_tracks.png`、`powder_location_all_tracks.png`、`powder_area_all_tracks.png`。
- 单峰大图：`single_crystal/location/`、`single_crystal/area/`、`powder/location/`、`powder/area/`。
- 对应数值矩阵：各数据集的 `location_matrices/` 和 `area_matrices/`。
- 完整索引：`per_peak_heatmap_index.csv`。

显示规则：白色是按严格下三角规则隐藏的对角线/上三角；灰色是下三角中的 NaN（未知或不可比较），不是 0。单晶 area 没有同压力重复测量校准，因此标为 uncalibrated/secondary；没有可比较 pair 的 track 仍保留全灰图，避免把“无数据”误认为漏文件。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def write_presence_tables(
    out_dir: Path,
    single_features: pd.DataFrame,
    powder_features: pd.DataFrame,
    single_registry: pd.DataFrame,
    powder_registry: pd.DataFrame,
) -> dict[str, Any]:
    """Separate observed presence from unknown; do not infer absence from curated gaps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    single_present = {(int(row.track), int(row.frame)) for row in single_features.itertuples()}
    for track in sorted(single_features["track"].astype(int).unique()):
        # The verified registry row order is the global matrix-axis order; it
        # does not expose a separate axis_index column.
        for row in single_registry.itertuples():
            present = (track, int(row.frame)) in single_present
            state_rows.append(
                {
                    "dataset": "single_crystal",
                    "track": track,
                    "scan": str(row.orientation),
                    "frame": int(row.frame),
                    "pressure_GPa": float(row.pressure_GPa),
                    "state": "present" if present else "unknown",
                    "absence_confirmed": 0,
                    "reason": "curated_kept_observation" if present else "not_in_curated_table_is_not_absence",
                }
            )
        for branch, group in single_registry.groupby("branch", sort=True):
            previous: dict[str, Any] | None = None
            for row in group.sort_values(["pressure_GPa", "frame"]).to_dict("records"):
                present = (track, int(row["frame"])) in single_present
                current = {
                    "pressure_GPa": float(row["pressure_GPa"]),
                    "state": "present" if present else "unknown",
                }
                if previous is not None:
                    both_present = previous["state"] == "present" and current["state"] == "present"
                    transition_rows.append(
                        {
                            "dataset": "single_crystal", "track": track,
                            "scan": str(branch),
                            "p_low_GPa": float(previous["pressure_GPa"]),
                            "p_high_GPa": float(current["pressure_GPa"]),
                            "delta_p_GPa": float(current["pressure_GPa"] - previous["pressure_GPa"]),
                            "n00": np.nan, "n01_birth": np.nan, "n10_death": np.nan,
                            "n11": int(both_present), "n_unknown": int(not both_present),
                            "birth_rate": np.nan, "death_rate": np.nan,
                            "turnover_rate": np.nan, "identifiable": 0,
                            "reason": "curated_missing_is_unknown_no_confirmed_absence",
                        }
                    )
                previous = current

    powder_included = powder_registry[powder_registry["included"].astype(int) == 1].copy()
    powder_present = {(int(row.track), int(row.frame)) for row in powder_features.itertuples()}
    for track in sorted(powder_features["track"].astype(int).unique()):
        for scan, group in powder_included.groupby("scan", sort=True):
            group = group.sort_values(["pressure_GPa", "frame"])
            previous: dict[str, Any] | None = None
            for row in group.to_dict("records"):
                present = (track, int(row["frame"])) in powder_present
                current = {
                    "dataset": "powder",
                    "track": track,
                    "scan": str(scan),
                    "frame": int(row["frame"]),
                    "pressure_GPa": float(row["pressure_GPa"]),
                    "state": "present" if present else "unknown",
                    "absence_confirmed": 0,
                    "reason": "curated_kept_observation" if present else "not_in_curated_table_is_not_absence",
                }
                state_rows.append(current)
                if previous is not None:
                    both_present = previous["state"] == "present" and current["state"] == "present"
                    transition_rows.append(
                        {
                            "dataset": "powder",
                            "track": track,
                            "scan": str(scan),
                            "p_low_GPa": float(previous["pressure_GPa"]),
                            "p_high_GPa": float(current["pressure_GPa"]),
                            "delta_p_GPa": float(current["pressure_GPa"] - previous["pressure_GPa"]),
                            "n00": np.nan,
                            "n01_birth": np.nan,
                            "n10_death": np.nan,
                            "n11": int(both_present),
                            "n_unknown": int(not both_present),
                            "birth_rate": np.nan,
                            "death_rate": np.nan,
                            "turnover_rate": np.nan,
                            "identifiable": 0,
                            "reason": "curated_missing_is_unknown_no_confirmed_absence",
                        }
                    )
                previous = current
    state_frame = pd.DataFrame(state_rows)
    transition_frame = pd.DataFrame(transition_rows)
    state_frame.to_csv(out_dir / "presence_state_long.csv", index=False)
    transition_frame.to_csv(out_dir / "birth_death_transitions.csv", index=False)
    summary = {
        "states": int(len(state_frame)),
        "present": int(np.count_nonzero(state_frame["state"] == "present")),
        "unknown": int(np.count_nonzero(state_frame["state"] == "unknown")),
        "confirmed_absent": 0,
        "confirmed_births": 0,
        "confirmed_deaths": 0,
        "transition_rows_single_crystal": int(
            np.count_nonzero(transition_frame["dataset"] == "single_crystal")
        ),
        "transition_rows_powder": int(
            np.count_nonzero(transition_frame["dataset"] == "powder")
        ),
        "birth_death_identifiable": False,
        "reason": "curated kept_obs tables enumerate detected/kept peaks only; a missing row remains unknown",
    }
    write_json(out_dir / "presence_summary.json", summary)
    return summary


def load_powder_series(handoff_root: Path, grid_step: float) -> dict[str, SeriesData]:
    frames, _, pressures, scans = legacy.read_manifest(handoff_root / "manifest.csv", None)
    grid = np.arange(2.0, 32.0 + grid_step / 2.0, grid_step)
    output: dict[str, SeriesData] = {}
    for channel in ("spots", "fit"):
        x, values, _ = legacy.load_channel(handoff_root, channel, frames)
        normalized = legacy.resample_rows(x, legacy.normalize_rows(values), grid)
        output[channel] = SeriesData(
            label="powder", channel=channel, frames=frames, scans=scans,
            pressures=pressures, grid=grid, normalized=normalized.astype(np.float32),
        )
    return output


def load_single_series(selected_path: Path, grid_step: float) -> dict[str, SeriesData]:
    registry = pd.read_csv(selected_path)
    output: dict[str, SeriesData] = {}
    for orientation_base in ("0deg", "10deg"):
        orientation = f"orientation_{orientation_base}"
        selected = registry[registry["orientation"] == orientation].sort_values("pressure_GPa")
        paths = [Path(item) for item in selected["file_path"].astype(str)]
        loaded = [np.loadtxt(path, comments="#") for path in paths]
        x = loaded[0][:, 0].astype(float)
        max_grid = math.floor(min(float(np.max(item[:, 0])) for item in loaded) / grid_step) * grid_step
        grid = np.arange(2.0, max_grid + grid_step / 2.0, grid_step)
        values = np.vstack([item[:, 1].astype(float) for item in loaded])
        normalized = legacy.resample_rows(x, legacy.normalize_rows(values), grid)
        pressures = selected["pressure_GPa"].astype(float).tolist()
        frames = [
            legacy.Frame(
                frame=int(row.frame), scan=orientation, pressure=float(row.pressure_GPa),
                pressure_index=index, original_filename=str(row.original_filename),
            )
            for index, row in enumerate(selected.itertuples())
        ]
        output[orientation_base] = SeriesData(
            label=f"single_{orientation_base}", channel="spots", frames=frames,
            scans=[orientation], pressures=pressures, grid=grid,
            normalized=normalized.astype(np.float32),
        )
    return output


def standardized_and_whitened(
    series: SeriesData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Standardize, then weight only from same-pressure scan repeatability when available."""
    standardized, valid = legacy.standardized_signals(
        series.normalized, smooth_window=9, baseline_window=101
    )
    if len(series.scans) > 1:
        residuals = np.full_like(standardized, np.nan, dtype=float)
        for pressure in series.pressures:
            rows = [
                index for index, frame in enumerate(series.frames)
                if math.isclose(float(frame.pressure), float(pressure), abs_tol=1.0e-9)
            ]
            if len(rows) < 2:
                continue
            group = standardized[np.asarray(rows)].astype(float)
            residuals[np.asarray(rows)] = group - np.nanmedian(group, axis=0, keepdims=True)
        center = np.nanmedian(residuals, axis=0)
        noise = 1.4826 * np.nanmedian(np.abs(residuals - center[None, :]), axis=0)
        positive = noise[np.isfinite(noise) & (noise > 0)]
        floor = float(np.quantile(positive, 0.10)) if len(positive) else 1.0
        noise = np.where(np.isfinite(noise) & (noise > floor), noise, floor)
        weighting = "same_pressure_cross_scan_repeatability_MAD"
    else:
        noise = np.ones(standardized.shape[1], dtype=float)
        weighting = "uniform_no_independent_same_pressure_replicates"
    whitened = standardized.astype(float) / noise[None, :]
    whitened[~valid] = np.nan
    return (
        standardized.astype(np.float32), whitened.astype(np.float32),
        noise.astype(np.float32), weighting,
    )


def normalized_cross_matrix(data: np.ndarray, lag: int, fixed_margin: int = 0) -> np.ndarray:
    """Row-wise NCC on one fixed core; positive lag moves the second row right."""
    n_points = data.shape[1]
    core_start = fixed_margin
    core_end = n_points - fixed_margin
    if core_end - core_start < 3 or abs(lag) > fixed_margin:
        return np.full((len(data), len(data)), np.nan)
    left = data[:, core_start:core_end]
    right = data[:, core_start + lag:core_end + lag]
    left = left - np.nanmean(left, axis=1, keepdims=True)
    right = right - np.nanmean(right, axis=1, keepdims=True)
    left_norm = np.sqrt(np.nansum(left**2, axis=1, keepdims=True))
    right_norm = np.sqrt(np.nansum(right**2, axis=1, keepdims=True))
    left = np.divide(left, left_norm, out=np.full_like(left, np.nan), where=left_norm > EPS)
    right = np.divide(right, right_norm, out=np.full_like(right, np.nan), where=right_norm > EPS)
    matrix = np.nan_to_num(left, nan=0.0) @ np.nan_to_num(right, nan=0.0).T
    valid_left = np.all(np.isfinite(left), axis=1)
    valid_right = np.all(np.isfinite(right), axis=1)
    matrix[~valid_left, :] = np.nan
    matrix[:, ~valid_right] = np.nan
    return np.clip(matrix, -1.0, 1.0)


def shifted_ncc_matrices(
    data: np.ndarray, official_lag: int, sensitivity_lags: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    n = len(data)
    maximum = max(official_lag, max(sensitivity_lags))
    lag_matrices = {
        lag: normalized_cross_matrix(data, lag, fixed_margin=maximum)
        for lag in range(-maximum, maximum + 1)
    }

    def best_within(bound: int) -> tuple[np.ndarray, np.ndarray]:
        best = np.full((n, n), -np.inf, dtype=float)
        best_lag = np.zeros((n, n), dtype=int)
        for lag in range(-bound, bound + 1):
            current = lag_matrices[lag]
            update = np.isfinite(current) & (current > best)
            best[update] = current[update]
            best_lag[update] = lag
        best[~np.isfinite(best)] = np.nan
        return best, best_lag

    official, official_best_lag = best_within(official_lag)
    sensitivity = {bound: best_within(bound)[0] for bound in sensitivity_lags}
    return lag_matrices[0], official, official_best_lag, sensitivity


def scan_rows(series: SeriesData) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = {scan: [] for scan in series.scans}
    for index, frame in enumerate(series.frames):
        rows[frame.scan].append(index)
    for scan in rows:
        rows[scan].sort(key=lambda index: series.frames[index].pressure)
    return rows


def window_metric_summary(values: np.ndarray, pressures: Sequence[float]) -> dict[str, Any]:
    near, far = classify_pair_values(values, pressures)
    gaps: list[float] = []
    scores: list[float] = []
    for i, p_i in enumerate(pressures):
        for j, p_j in enumerate(pressures[:i]):
            current = np.asarray(values[..., i, j]).reshape(-1)
            current = current[np.isfinite(current)]
            gaps.extend([abs(float(p_i) - float(p_j))] * len(current))
            scores.extend(current.tolist())
    return {
        "finite_scores": len(scores),
        "median": float(np.median(scores)) if scores else np.nan,
        "near_median": float(np.median(near)) if len(near) else np.nan,
        "far_median": float(np.median(far)) if len(far) else np.nan,
        "near_far_auc": auc_probability(near, far),
        "r_score_vs_pressure_gap": pearson(np.asarray(gaps), np.asarray(scores)),
    }


def analyze_same_windows(
    out_dir: Path,
    series: SeriesData,
    width_deg: float,
    step_deg: float,
    max_shift_deg: float,
    make_plots: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, whitened, noise, weighting_method = standardized_and_whitened(series)
    dx = float(np.median(np.diff(series.grid)))
    max_lag = max(1, int(round(max_shift_deg / dx)))
    sensitivity_lags = sorted({max(1, int(round(value / dx))) for value in (0.06, 0.12, 0.18)})
    maximum_lag = max(max_lag, max(sensitivity_lags))
    core_bins = int(round(width_deg / dx))
    step_bins = max(1, int(round(step_deg / dx)))
    start_indices = np.arange(
        maximum_lag,
        len(series.grid) - maximum_lag - core_bins + 1,
        step_bins,
        dtype=int,
    )
    starts = series.grid[start_indices]
    primary_stride = max(1, int(round(width_deg / step_deg)))
    primary_count = max(1, len(starts) // primary_stride)
    primary_span = (primary_count - 1) * primary_stride
    primary_offset = max(0, (len(starts) - 1 - primary_span) // 2)
    n_scans, n_windows, n_pressures = len(series.scans), len(starts), len(series.pressures)
    zero = np.full((n_scans, n_windows, n_pressures, n_pressures), np.nan, dtype=np.float32)
    aligned = np.full_like(zero, np.nan)
    shift = np.full_like(zero, np.nan)
    sensitivity_arrays = {lag: np.full_like(zero, np.nan) for lag in sensitivity_lags}
    frame_rows = scan_rows(series)
    pressure_lookup = {float(value): index for index, value in enumerate(series.pressures)}
    adjacent_rows: list[dict[str, Any]] = []
    for scan_index, scan in enumerate(series.scans):
        rows = frame_rows[scan]
        for window_index, (start_index, start) in enumerate(zip(start_indices, starts)):
            extended = slice(
                int(start_index - maximum_lag),
                int(start_index + core_bins + maximum_lag),
            )
            data = whitened[np.asarray(rows)][:, extended]
            zero_local, best_local, lag_local, sensitivity = shifted_ncc_matrices(
                data, max_lag, sensitivity_lags
            )
            for high_local in range(len(rows)):
                high_frame = series.frames[rows[high_local]]
                high_p = pressure_lookup[float(high_frame.pressure)]
                for low_local in range(high_local):
                    low_frame = series.frames[rows[low_local]]
                    low_p = pressure_lookup[float(low_frame.pressure)]
                    if high_p == low_p:
                        continue
                    pair_low, pair_high = low_local, high_local
                    if high_p < low_p:
                        high_p, low_p = low_p, high_p
                        pair_low, pair_high = high_local, low_local
                    zero[scan_index, window_index, high_p, low_p] = zero_local[pair_low, pair_high]
                    aligned[scan_index, window_index, high_p, low_p] = best_local[pair_low, pair_high]
                    shift[scan_index, window_index, high_p, low_p] = lag_local[pair_low, pair_high] * dx
                    for lag, array in sensitivity.items():
                        sensitivity_arrays[lag][scan_index, window_index, high_p, low_p] = array[pair_low, pair_high]
            for high_local in range(1, len(rows)):
                low_local = high_local - 1
                low_frame = series.frames[rows[low_local]]
                high_frame = series.frames[rows[high_local]]
                low_p = pressure_lookup[float(low_frame.pressure)]
                high_p = pressure_lookup[float(high_frame.pressure)]
                adjacent_rows.append(
                    {
                        "dataset": series.label,
                        "channel": series.channel,
                        "scan": scan,
                        "window_index": window_index,
                        "window_start_deg": float(start),
                        "window_end_deg": float(start + width_deg),
                        "nonoverlap_primary": int(
                            window_index >= primary_offset
                            and (window_index - primary_offset) % primary_stride == 0
                        ),
                        "p_low_GPa": float(low_frame.pressure),
                        "p_high_GPa": float(high_frame.pressure),
                        "delta_p_GPa": float(high_frame.pressure - low_frame.pressure),
                        "frame_low": int(low_frame.frame),
                        "frame_high": int(high_frame.frame),
                        "filename_low": str(low_frame.original_filename),
                        "filename_high": str(high_frame.original_filename),
                        "acquisition_signature_low": acquisition_signature(low_frame.original_filename),
                        "acquisition_signature_high": acquisition_signature(high_frame.original_filename),
                        "acquisition_protocol_changed": int(
                            acquisition_signature(low_frame.original_filename)
                            != acquisition_signature(high_frame.original_filename)
                        ),
                        "zero_shift_ncc": float(zero[scan_index, window_index, high_p, low_p]),
                        "aligned_ncc": float(aligned[scan_index, window_index, high_p, low_p]),
                        "best_shift_deg": float(shift[scan_index, window_index, high_p, low_p]),
                        "alignment_gain": float(aligned[scan_index, window_index, high_p, low_p] - zero[scan_index, window_index, high_p, low_p]),
                        "shape_change": float(1.0 - aligned[scan_index, window_index, high_p, low_p]),
                    }
                )

    aggregate_zero = legacy.nanmedian(zero, axis=0)
    aggregate_aligned = legacy.nanmedian(aligned, axis=0)
    aggregate_shift = legacy.nanmedian(shift, axis=0)
    np.savez_compressed(
        out_dir / "same_window_matrices.npz",
        pressure_gpa=np.asarray(series.pressures), scan_names=np.asarray(series.scans),
        window_starts_deg=starts, zero_shift_by_scan=zero, aligned_by_scan=aligned,
        best_shift_deg_by_scan=shift, zero_shift_aggregate=aggregate_zero,
        aligned_aggregate=aggregate_aligned, best_shift_deg_aggregate=aggregate_shift,
        noise_sigma_by_bin=noise, grid_two_theta_deg=series.grid,
        **{f"aligned_bound_{lag}_bins": array for lag, array in sensitivity_arrays.items()},
    )
    adjacent = pd.DataFrame(adjacent_rows)
    adjusted_parts: list[pd.DataFrame] = []
    for _, group in adjacent.groupby("window_index", sort=True):
        x = np.log1p(group["delta_p_GPa"].to_numpy(float))
        y = group["shape_change"].to_numpy(float)
        keep = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(keep) >= 3 and np.std(x[keep]) > 0:
            beta, *_ = np.linalg.lstsq(
                np.column_stack([np.ones(np.count_nonzero(keep)), x[keep]]), y[keep], rcond=None
            )
            expected = beta[0] + beta[1] * x
        else:
            expected = np.full(len(group), np.nanmedian(y))
        current = group.copy()
        current["expected_change_from_delta_p"] = expected
        current["delta_p_adjusted_change"] = y - expected
        adjusted_parts.append(current)
    adjacent = pd.concat(adjusted_parts, ignore_index=True)
    adjacent.to_csv(out_dir / "adjacent_window_trajectories.csv", index=False)

    window_rows: list[dict[str, Any]] = []
    for window_index, start in enumerate(starts):
        raw_summary = window_metric_summary(zero[:, window_index], series.pressures)
        aligned_summary = window_metric_summary(aligned[:, window_index], series.pressures)
        window_rows.append(
            {
                "dataset": series.label,
                "channel": series.channel,
                "window_index": window_index,
                "start_deg": float(start),
                "end_deg": float(start + width_deg),
                **{f"zero_{key}": value for key, value in raw_summary.items()},
                **{f"aligned_{key}": value for key, value in aligned_summary.items()},
                "median_abs_best_shift_deg": float(np.nanmedian(np.abs(shift[:, window_index]))),
                "median_alignment_gain": float(np.nanmedian(aligned[:, window_index] - zero[:, window_index])),
            }
        )
    pd.DataFrame(window_rows).to_csv(out_dir / "window_summary.csv", index=False)
    sensitivity_rows: list[dict[str, Any]] = []
    for lag, array in sensitivity_arrays.items():
        sensitivity_rows.append(
            {
                "dataset": series.label,
                "channel": series.channel,
                "shift_bound_deg": lag * dx,
                **window_metric_summary(array, series.pressures),
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(out_dir / "shift_bound_sensitivity.csv", index=False)
    interval = (
        adjacent.groupby(
            ["window_index", "window_start_deg", "window_end_deg", "p_low_GPa", "p_high_GPa", "delta_p_GPa"],
            as_index=False,
        )
        .agg(
            scan_support=("scan", "nunique"),
            median_aligned_ncc=("aligned_ncc", "median"),
            median_shape_change=("shape_change", "median"),
            median_delta_p_adjusted_change=("delta_p_adjusted_change", "median"),
            q25_adjusted_change=("delta_p_adjusted_change", lambda values: np.nanquantile(values, 0.25)),
            q75_adjusted_change=("delta_p_adjusted_change", lambda values: np.nanquantile(values, 0.75)),
            median_best_shift_deg=("best_shift_deg", "median"),
        )
    )
    interval.to_csv(out_dir / "trajectory_by_interval.csv", index=False)
    trajectory_pivot = interval.pivot(
        index="window_index", columns=["p_low_GPa", "p_high_GPa"],
        values="median_delta_p_adjusted_change"
    )
    trajectory_matrix = np.full((n_windows, n_windows), np.nan)
    for i in range(n_windows):
        for j in range(i):
            trajectory_matrix[i, j] = pearson(
                trajectory_pivot.loc[i].to_numpy(float), trajectory_pivot.loc[j].to_numpy(float)
            )
    labels = [f"{start:g}-{start + width_deg:g}" for start in starts]
    write_matrix_csv(out_dir / "window_trajectory_correlation_matrix.csv", labels, trajectory_matrix)
    if make_plots:
        plot_heatmap(
            out_dir / "window_trajectory_correlation_heatmap.png", trajectory_matrix, labels,
            f"{series.label} {series.channel}: agreement of pressure-change trajectories",
            -1.0, 1.0, "coolwarm", "Pearson r of delta-P-adjusted change trajectories",
        )
        pivot_change = interval.pivot(
            index="window_index", columns=["p_low_GPa", "p_high_GPa"],
            values="median_delta_p_adjusted_change"
        )
        fig, ax = plt.subplots(figsize=(11.0, 6.0))
        image = ax.imshow(pivot_change.to_numpy(float), aspect="auto", cmap="coolwarm")
        ax.set_yticks(
            range(len(pivot_change.index)),
            [f"{starts[index]:g}-{starts[index] + width_deg:g}" for index in pivot_change.index],
            fontsize=7,
        )
        ax.set_xticks(
            range(len(pivot_change.columns)),
            [f"{low:g}→{high:g}" for low, high in pivot_change.columns],
            rotation=45, ha="right",
        )
        ax.set_xlabel("Adjacent pressure interval (GPa)")
        ax.set_ylabel("2theta window (deg)")
        ax.set_title(f"{series.label} {series.channel}: delta-P-adjusted shape change")
        fig.colorbar(image, ax=ax, label="Residual shape change")
        fig.tight_layout()
        fig.savefig(out_dir / "window_change_trajectory_map.png", dpi=190)
        plt.close(fig)
    overall = window_metric_summary(aligned, series.pressures)
    overall.update(
        {
            "dataset": series.label, "channel": series.channel,
            "frames": len(series.frames), "scans": len(series.scans),
            "pressures": len(series.pressures), "windows": n_windows,
            "window_width_deg": width_deg, "window_step_deg": step_deg,
            "max_shift_deg": max_shift_deg,
            "noise_weighting": weighting_method,
            "shape_metric": "same-window bounded direct NCC",
            "shift_reported_separately": True,
            "fixed_core_for_all_lags": True,
        }
    )
    write_json(out_dir / "window_metrics.json", overall)
    return overall, adjacent


def combine_powder_boundaries(
    out_dir: Path,
    spots: pd.DataFrame,
    fit: pd.DataFrame,
    width_deg: float,
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["scan", "window_index", "p_low_GPa", "p_high_GPa", "delta_p_GPa"]
    merged = spots.merge(
        fit[keys + ["delta_p_adjusted_change", "shape_change", "best_shift_deg"]],
        on=keys, suffixes=("_spots", "_fit"), validate="one_to_one",
    )
    merged["sample_specific_excess_change"] = (
        merged["delta_p_adjusted_change_spots"] - merged["delta_p_adjusted_change_fit"]
    )
    if "nonoverlap_primary" not in merged.columns:
        raise ValueError("Window trajectories must explicitly mark fixed-core primary windows")
    merged.to_csv(out_dir / "matched_spots_fit_window_changes.csv", index=False)
    primary = merged[merged["nonoverlap_primary"] == 1].copy()
    rows: list[dict[str, Any]] = []
    for (p_low, p_high, delta_p), group in primary.groupby(
        ["p_low_GPa", "p_high_GPa", "delta_p_GPa"], sort=True
    ):
        # The independent replication unit is a scan, not each scan-window row.
        # First collapse the six non-overlapping windows within a scan, then
        # bootstrap scan medians so correlated windows do not narrow the CI.
        by_scan = (
            group.groupby("scan", as_index=False)
            .agg(
                spots_adjusted_change=("delta_p_adjusted_change_spots", "median"),
                fit_adjusted_change=("delta_p_adjusted_change_fit", "median"),
                sample_specific_excess=("sample_specific_excess_change", "median"),
                nonoverlap_windows=("window_index", "nunique"),
            )
        )
        values = by_scan["sample_specific_excess"].to_numpy(float)
        ci_low, ci_high = bootstrap_median_ci(values, bootstrap_resamples, rng)
        rows.append(
            {
                "p_low_GPa": float(p_low), "p_high_GPa": float(p_high),
                "delta_p_GPa": float(delta_p), "scan_window_support": int(len(group)),
                "independent_scans": int(len(by_scan)),
                "nonoverlap_windows": int(group["window_index"].nunique()),
                "median_spots_adjusted_change": float(np.nanmedian(by_scan["spots_adjusted_change"])),
                "median_fit_adjusted_change": float(np.nanmedian(by_scan["fit_adjusted_change"])),
                "median_sample_specific_excess": float(np.nanmedian(values)),
                "sample_specific_excess_ci95_low": ci_low,
                "sample_specific_excess_ci95_high": ci_high,
                "positive_excess_fraction": float(np.mean(values > 0)),
                "statistical_positive_excess": int(ci_low > 0 and np.mean(values > 0) >= 0.60),
                "protocol_change_fraction": float(np.mean(group["acquisition_protocol_changed"] > 0)),
                "protocol_confounded": int(np.any(group["acquisition_protocol_changed"] > 0)),
                "protocol_low": ";".join(sorted(set(group["acquisition_signature_low"].astype(str)))),
                "protocol_high": ";".join(sorted(set(group["acquisition_signature_high"].astype(str)))),
                "bootstrap_unit": "independent_scan_median_across_nonoverlap_windows",
                "interpretation": "candidate_only_requires_peak_or_presence_confirmation",
            }
        )
    candidates = pd.DataFrame(rows)
    effect_center = float(np.nanmedian(candidates["median_sample_specific_excess"]))
    effect_scale = robust_scale(candidates["median_sample_specific_excess"])
    practical_threshold = max(0.0, effect_center + 3.0 * effect_scale)
    maximum_scan_support = int(candidates["independent_scans"].max())
    minimum_scan_support = max(10, int(math.ceil(0.80 * maximum_scan_support)))
    candidates["robust_effect_center"] = effect_center
    candidates["robust_effect_scale_mad"] = effect_scale
    candidates["practical_positive_outlier_threshold"] = practical_threshold
    candidates["practical_positive_outlier"] = (
        candidates["median_sample_specific_excess"] > practical_threshold
    ).astype(int)
    candidates["minimum_scan_support_required"] = minimum_scan_support
    candidates["scan_support_eligible"] = (
        candidates["independent_scans"] >= minimum_scan_support
    ).astype(int)
    candidates["candidate_boundary"] = (
        (candidates["statistical_positive_excess"] == 1)
        & (candidates["practical_positive_outlier"] == 1)
        & (candidates["protocol_confounded"] == 0)
        & (candidates["scan_support_eligible"] == 1)
    ).astype(int)
    candidates.loc[candidates["statistical_positive_excess"] == 0, "interpretation"] = (
        "no_scan_bootstrap_positive_excess_not_candidate"
    )
    candidates.loc[
        (candidates["statistical_positive_excess"] == 1)
        & (candidates["practical_positive_outlier"] == 0),
        "interpretation",
    ] = "below_robust_practical_effect_threshold_not_candidate"
    candidates.loc[candidates["scan_support_eligible"] == 0, "interpretation"] = (
        "insufficient_independent_scan_support_not_candidate"
    )
    candidates.loc[candidates["protocol_confounded"] == 1, "interpretation"] = (
        "acquisition_protocol_confounded_QC_signal_not_sample_candidate"
    )
    candidates["candidate_rank"] = candidates["median_sample_specific_excess"].rank(
        method="min", ascending=False
    ).astype(int)
    candidates = candidates.sort_values("candidate_rank")
    candidates.to_csv(out_dir / "boundary_candidates.csv", index=False)
    if len(candidates):
        ordered = candidates.sort_values("p_high_GPa")
        y = ordered["median_sample_specific_excess"].to_numpy(float)
        low = ordered["sample_specific_excess_ci95_low"].to_numpy(float)
        high = ordered["sample_specific_excess_ci95_high"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.errorbar(
            ordered["p_high_GPa"], y, yerr=np.vstack([y - low, high - y]),
            fmt="o-", color="#2F6F9F", ecolor="#7EA6C5", capsize=2,
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("High-pressure edge of adjacent interval (GPa)")
        ax.set_ylabel("Spots excess change after fit control")
        ax.set_title("Window trajectory candidates (95% descriptive bootstrap CI)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "boundary_candidates.png", dpi=190)
        plt.close(fig)
    eligible = candidates[candidates["candidate_boundary"] == 1].sort_values("candidate_rank")
    top_raw = candidates.sort_values("candidate_rank").iloc[0] if len(candidates) else None
    top_eligible = eligible.iloc[0] if len(eligible) else None
    summary = {
        "intervals": int(len(candidates)),
        "candidate_intervals": int(candidates["candidate_boundary"].sum()) if len(candidates) else 0,
        "statistical_positive_intervals": int(candidates["statistical_positive_excess"].sum()) if len(candidates) else 0,
        "protocol_confounded_intervals": int(candidates["protocol_confounded"].sum()) if len(candidates) else 0,
        "practical_positive_outlier_threshold": practical_threshold,
        "minimum_independent_scan_support": minimum_scan_support,
        "top_candidate_high_pressure_GPa": float(top_eligible["p_high_GPa"]) if top_eligible is not None else np.nan,
        "top_candidate_sample_specific_excess": float(top_eligible["median_sample_specific_excess"]) if top_eligible is not None else np.nan,
        "top_raw_high_pressure_GPa": float(top_raw["p_high_GPa"]) if top_raw is not None else np.nan,
        "top_raw_sample_specific_excess": float(top_raw["median_sample_specific_excess"]) if top_raw is not None else np.nan,
        "top_raw_protocol_confounded": int(top_raw["protocol_confounded"]) if top_raw is not None else 0,
        "primary_windows": "nonoverlap 5-degree windows",
        "status": "protocol-screened candidate ranking, not phase-transition proof",
    }
    write_json(out_dir / "boundary_summary.json", summary)
    return summary, candidates


def analyze_whole_pattern(
    out_dir: Path, series: SeriesData, make_plots: bool
) -> tuple[dict[str, Any], pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    standardized, _ = legacy.standardized_signals(series.normalized, smooth_window=9, baseline_window=101)
    rows_by_scan = scan_rows(series)
    pressure_lookup = {float(value): index for index, value in enumerate(series.pressures)}
    matrices = np.full(
        (len(series.scans), len(series.pressures), len(series.pressures)), np.nan, dtype=np.float32
    )
    pairs: list[dict[str, Any]] = []
    for scan_index, scan in enumerate(series.scans):
        rows = rows_by_scan[scan]
        local = legacy.row_correlation(standardized[rows], standardized[rows])
        for high_local in range(len(rows)):
            high_frame = series.frames[rows[high_local]]
            high_p = pressure_lookup[float(high_frame.pressure)]
            for low_local in range(high_local):
                low_frame = series.frames[rows[low_local]]
                low_p = pressure_lookup[float(low_frame.pressure)]
                if high_p == low_p:
                    continue
                score = float(local[high_local, low_local])
                if high_p < low_p:
                    high_p, low_p = low_p, high_p
                matrices[scan_index, high_p, low_p] = score
                pairs.append(
                    {
                        "dataset": series.label,
                        "channel": series.channel,
                        "scan": scan,
                        "frame_a": int(high_frame.frame),
                        "frame_b": int(low_frame.frame),
                        "pressure_a_GPa": float(high_frame.pressure),
                        "pressure_b_GPa": float(low_frame.pressure),
                        "pressure_gap_GPa": abs(float(high_frame.pressure) - float(low_frame.pressure)),
                        "acquisition_signature_a": acquisition_signature(
                            high_frame.original_filename
                        ),
                        "acquisition_signature_b": acquisition_signature(
                            low_frame.original_filename
                        ),
                        "acquisition_protocol_changed": int(
                            acquisition_signature(high_frame.original_filename)
                            != acquisition_signature(low_frame.original_filename)
                        ),
                        "correlation": score,
                    }
                )
    aggregate = legacy.nanmedian(matrices, axis=0)
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(out_dir / "whole_pattern_pair_scores.csv", index=False)
    labels = [f"{value:g}" for value in series.pressures]
    write_matrix_csv(out_dir / "aggregate_matrix.csv", labels, aggregate)
    np.savez_compressed(
        out_dir / "whole_pattern_matrices.npz",
        pressure_gpa=np.asarray(series.pressures), scan_names=np.asarray(series.scans),
        matrices_by_scan=matrices, aggregate=aggregate,
    )
    if make_plots:
        plot_heatmap(
            out_dir / "aggregate_heatmap.png", aggregate, labels,
            f"{series.label} {series.channel}: raw whole-pattern Pearson QC",
            -1.0, 1.0, "coolwarm", "Pearson correlation",
        )
    summary = {
        "dataset": series.label,
        "channel": series.channel,
        "frames": len(series.frames),
        "scans": len(series.scans),
        "pairs": len(pair_frame),
        "raw_r_correlation_vs_pressure_gap": pearson(
            pair_frame["pressure_gap_GPa"].to_numpy(float), pair_frame["correlation"].to_numpy(float)
        ),
        "raw_slope_correlation_vs_pressure_gap": (
            float(np.polyfit(pair_frame["pressure_gap_GPa"], pair_frame["correlation"], 1)[0])
            if len(pair_frame) >= 3 else np.nan
        ),
        "role": "QC_only",
    }
    write_json(out_dir / "whole_pattern_metrics.json", summary)
    return summary, pair_frame


def residualize(values: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    controls = np.asarray(controls, dtype=float)
    design = np.column_stack([np.ones(len(values)), controls])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta, beta


def control_adjust_whole_pattern(
    out_dir: Path,
    spots: pd.DataFrame,
    fit: pd.DataFrame,
    pressures: Sequence[float],
    bootstrap_resamples: int,
    rng: np.random.Generator,
    make_plots: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["scan", "frame_a", "frame_b", "pressure_a_GPa", "pressure_b_GPa", "pressure_gap_GPa"]
    merged = spots.merge(
        fit[keys + ["correlation"]], on=keys, suffixes=("_spots", "_fit"), validate="one_to_one"
    )
    parts: list[pd.DataFrame] = []
    scan_rows_output: list[dict[str, Any]] = []
    for scan, group in merged.groupby("scan", sort=True):
        frame_ids = sorted(set(group["frame_a"]).union(group["frame_b"]))
        order = {int(frame): index for index, frame in enumerate(frame_ids)}
        current = group.copy()
        current["acquisition_order_gap"] = np.abs(
            current["frame_a"].map(order).to_numpy(int) - current["frame_b"].map(order).to_numpy(int)
        )
        y = current["correlation_spots"].to_numpy(float)
        fit_score = current["correlation_fit"].to_numpy(float)
        order_gap = current["acquisition_order_gap"].to_numpy(float)
        pressure_gap = current["pressure_gap_GPa"].to_numpy(float)
        protocol_change = current["acquisition_protocol_changed"].to_numpy(float)
        base_controls = np.column_stack([fit_score, order_gap])
        base_residual, _ = residualize(y, base_controls)
        base_pressure_residual, _ = residualize(pressure_gap, base_controls)
        controls = np.column_stack([fit_score, order_gap, protocol_change])
        residual, beta_controls = residualize(y, controls)
        pressure_residual, _ = residualize(pressure_gap, controls)
        current["spots_residual_after_fit_order_protocol"] = residual
        current["pressure_gap_residual_after_fit_order_protocol"] = pressure_residual
        current["spots_residual_after_fit_and_order_no_protocol"] = base_residual
        current["pressure_gap_residual_after_fit_and_order_no_protocol"] = base_pressure_residual
        y_fisher = np.arctanh(np.clip(y, -1.0 + 1.0e-6, 1.0 - 1.0e-6))
        fit_fisher = np.arctanh(np.clip(fit_score, -1.0 + 1.0e-6, 1.0 - 1.0e-6))
        fisher_base_controls = np.column_stack([fit_fisher, order_gap])
        fisher_base_residual, _ = residualize(y_fisher, fisher_base_controls)
        fisher_base_pressure_residual, _ = residualize(
            pressure_gap, fisher_base_controls
        )
        fisher_controls = np.column_stack([fit_fisher, order_gap, protocol_change])
        fisher_residual, _ = residualize(y_fisher, fisher_controls)
        fisher_pressure_residual, _ = residualize(pressure_gap, fisher_controls)
        current["correlation_spots_fisher_z"] = y_fisher
        current["correlation_fit_fisher_z"] = fit_fisher
        current["spots_fisher_z_residual_after_fit_order_protocol"] = fisher_residual
        current["pressure_gap_residual_after_fit_z_order_protocol"] = fisher_pressure_residual
        current["spots_fisher_z_residual_after_fit_order_no_protocol"] = fisher_base_residual
        current["pressure_gap_residual_after_fit_z_order_no_protocol"] = (
            fisher_base_pressure_residual
        )
        standardized_predictors = np.column_stack(
            [
                (fit_score - np.mean(fit_score)) / max(np.std(fit_score), EPS),
                (order_gap - np.mean(order_gap)) / max(np.std(order_gap), EPS),
                (protocol_change - np.mean(protocol_change))
                / max(np.std(protocol_change), EPS),
                (pressure_gap - np.mean(pressure_gap)) / max(np.std(pressure_gap), EPS),
            ]
        )
        full_design = np.column_stack([np.ones(len(group)), standardized_predictors])
        y_standardized = (y - np.mean(y)) / max(np.std(y), EPS)
        beta_full, *_ = np.linalg.lstsq(full_design, y_standardized, rcond=None)
        fisher_predictors = np.column_stack(
            [
                (fit_fisher - np.mean(fit_fisher)) / max(np.std(fit_fisher), EPS),
                (order_gap - np.mean(order_gap)) / max(np.std(order_gap), EPS),
                (protocol_change - np.mean(protocol_change))
                / max(np.std(protocol_change), EPS),
                (pressure_gap - np.mean(pressure_gap)) / max(np.std(pressure_gap), EPS),
            ]
        )
        fisher_design = np.column_stack([np.ones(len(group)), fisher_predictors])
        y_fisher_standardized = (
            (y_fisher - np.mean(y_fisher)) / max(np.std(y_fisher), EPS)
        )
        beta_fisher, *_ = np.linalg.lstsq(
            fisher_design, y_fisher_standardized, rcond=None
        )
        same_protocol = protocol_change == 0
        if np.count_nonzero(same_protocol) >= 5:
            same_controls = np.column_stack(
                [fit_score[same_protocol], order_gap[same_protocol]]
            )
            same_y_residual, _ = residualize(y[same_protocol], same_controls)
            same_pressure_residual, _ = residualize(
                pressure_gap[same_protocol], same_controls
            )
            same_protocol_partial = pearson(
                same_y_residual, same_pressure_residual
            )
        else:
            same_protocol_partial = np.nan
        scan_rows_output.append(
            {
                "scan": scan,
                "pairs": len(group),
                "raw_r_spots_vs_pressure_gap": pearson(pressure_gap, y),
                "raw_r_fit_vs_pressure_gap": pearson(pressure_gap, fit_score),
                "r_spots_vs_fit": pearson(y, fit_score),
                "r_pressure_gap_vs_order_gap": pearson(pressure_gap, order_gap),
                "protocol_changed_pairs": int(np.count_nonzero(protocol_change)),
                "partial_r_spots_vs_pressure_gap_given_fit_order_protocol": pearson(
                    pressure_residual, residual
                ),
                "partial_r_without_protocol_control": pearson(
                    base_pressure_residual, base_residual
                ),
                "partial_r_same_protocol_pairs_given_fit_order": same_protocol_partial,
                "partial_r_fisher_z_spots_vs_pressure_gap_given_fit_z_order_protocol": pearson(
                    fisher_pressure_residual, fisher_residual
                ),
                "partial_r_fisher_z_without_protocol_control": pearson(
                    fisher_base_pressure_residual, fisher_base_residual
                ),
                "standardized_pressure_gap_coefficient": float(beta_full[4]),
                "standardized_fit_coefficient": float(beta_full[1]),
                "standardized_order_gap_coefficient": float(beta_full[2]),
                "standardized_protocol_change_coefficient": float(beta_full[3]),
                "fisher_z_standardized_pressure_gap_coefficient": float(beta_fisher[4]),
                "fisher_z_standardized_fit_coefficient": float(beta_fisher[1]),
                "fisher_z_standardized_order_gap_coefficient": float(beta_fisher[2]),
                "fisher_z_standardized_protocol_change_coefficient": float(beta_fisher[3]),
                "design_condition_number": float(np.linalg.cond(full_design)),
                "control_intercept": float(beta_controls[0]),
                "control_fit_coefficient": float(beta_controls[1]),
                "control_order_gap_coefficient": float(beta_controls[2]),
                "control_protocol_change_coefficient": float(beta_controls[3]),
            }
        )
        parts.append(current)
    adjusted = pd.concat(parts, ignore_index=True)
    scan_metrics = pd.DataFrame(scan_rows_output)
    adjusted.to_csv(out_dir / "adjusted_pair_scores.csv", index=False)
    scan_metrics.to_csv(out_dir / "scan_level_control_metrics.csv", index=False)

    pressure_lookup = {float(value): index for index, value in enumerate(pressures)}
    scans = sorted(adjusted["scan"].unique())
    matrices = np.full((len(scans), len(pressures), len(pressures)), np.nan, dtype=np.float32)
    scan_lookup = {scan: index for index, scan in enumerate(scans)}
    for row in adjusted.itertuples():
        high = pressure_lookup[float(max(row.pressure_a_GPa, row.pressure_b_GPa))]
        low = pressure_lookup[float(min(row.pressure_a_GPa, row.pressure_b_GPa))]
        matrices[scan_lookup[str(row.scan)], high, low] = float(
            row.spots_residual_after_fit_order_protocol
        )
    aggregate = legacy.nanmedian(matrices, axis=0)
    labels = [f"{value:g}" for value in pressures]
    write_matrix_csv(out_dir / "aggregate_adjusted_residual_matrix.csv", labels, aggregate)
    np.savez_compressed(
        out_dir / "adjusted_residual_matrices.npz",
        pressure_gpa=np.asarray(pressures), scan_names=np.asarray(scans),
        matrices_by_scan=matrices, aggregate=aggregate,
    )
    if make_plots:
        plot_heatmap(
            out_dir / "aggregate_adjusted_residual_heatmap.png", aggregate, labels,
            "Powder spots residual after fit/order/protocol adjustment",
            float(np.nanpercentile(aggregate, 5)), float(np.nanpercentile(aggregate, 95)),
            "coolwarm", "Residual sample-channel Pearson score",
        )
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        scatter = ax.scatter(
            adjusted["correlation_fit"], adjusted["correlation_spots"],
            c=adjusted["pressure_gap_GPa"], s=8, alpha=0.25, cmap="viridis", edgecolors="none",
        )
        fig.colorbar(scatter, ax=ax, label="Pressure gap (GPa)")
        ax.set_xlabel("Fit-channel Pearson (W/background control)")
        ax.set_ylabel("Spots-channel Pearson")
        ax.set_title("Whole-pattern sample channel vs matched control")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "spots_vs_fit_control.png", dpi=190)
        plt.close(fig)
    partial = scan_metrics[
        "partial_r_spots_vs_pressure_gap_given_fit_order_protocol"
    ].to_numpy(float)
    partial_no_protocol = scan_metrics["partial_r_without_protocol_control"].to_numpy(float)
    partial_same_protocol = scan_metrics[
        "partial_r_same_protocol_pairs_given_fit_order"
    ].to_numpy(float)
    partial_fisher = scan_metrics[
        "partial_r_fisher_z_spots_vs_pressure_gap_given_fit_z_order_protocol"
    ].to_numpy(float)
    partial_fisher_no_protocol = scan_metrics[
        "partial_r_fisher_z_without_protocol_control"
    ].to_numpy(float)
    coefficients = scan_metrics["standardized_pressure_gap_coefficient"].to_numpy(float)
    fisher_coefficients = scan_metrics[
        "fisher_z_standardized_pressure_gap_coefficient"
    ].to_numpy(float)
    partial_ci = bootstrap_median_ci(partial, bootstrap_resamples, rng)
    partial_no_protocol_ci = bootstrap_median_ci(
        partial_no_protocol, bootstrap_resamples, rng
    )
    partial_same_protocol_ci = bootstrap_median_ci(
        partial_same_protocol, bootstrap_resamples, rng
    )
    partial_fisher_ci = bootstrap_median_ci(partial_fisher, bootstrap_resamples, rng)
    partial_fisher_no_protocol_ci = bootstrap_median_ci(
        partial_fisher_no_protocol, bootstrap_resamples, rng
    )
    coefficient_ci = bootstrap_median_ci(coefficients, bootstrap_resamples, rng)
    fisher_coefficient_ci = bootstrap_median_ci(
        fisher_coefficients, bootstrap_resamples, rng
    )
    summary = {
        "pairs": int(len(adjusted)),
        "scans": int(len(scan_metrics)),
        "raw_spots_pooled_r_vs_pressure_gap": pearson(
            adjusted["pressure_gap_GPa"], adjusted["correlation_spots"]
        ),
        "raw_fit_pooled_r_vs_pressure_gap": pearson(
            adjusted["pressure_gap_GPa"], adjusted["correlation_fit"]
        ),
        "spots_vs_fit_pair_r": pearson(adjusted["correlation_spots"], adjusted["correlation_fit"]),
        "pressure_gap_vs_order_gap_r": pearson(
            adjusted["pressure_gap_GPa"], adjusted["acquisition_order_gap"]
        ),
        "protocol_changed_pairs": int(adjusted["acquisition_protocol_changed"].sum()),
        "median_scan_partial_r_given_fit_order_protocol": float(np.nanmedian(partial)),
        "median_scan_partial_r_given_fit_order_protocol_ci95": list(partial_ci),
        "median_scan_partial_r_without_protocol_control": float(
            np.nanmedian(partial_no_protocol)
        ),
        "median_scan_partial_r_without_protocol_control_ci95": list(
            partial_no_protocol_ci
        ),
        "median_scan_partial_r_same_protocol_pairs_given_fit_order": float(
            np.nanmedian(partial_same_protocol)
        ),
        "median_scan_partial_r_same_protocol_pairs_given_fit_order_ci95": list(
            partial_same_protocol_ci
        ),
        "median_scan_partial_r_fisher_z_given_fit_z_order_protocol": float(
            np.nanmedian(partial_fisher)
        ),
        "median_scan_partial_r_fisher_z_given_fit_z_order_protocol_ci95": list(
            partial_fisher_ci
        ),
        "median_scan_partial_r_fisher_z_without_protocol_control": float(
            np.nanmedian(partial_fisher_no_protocol)
        ),
        "median_scan_partial_r_fisher_z_without_protocol_control_ci95": list(
            partial_fisher_no_protocol_ci
        ),
        "median_standardized_pressure_gap_coefficient": float(np.nanmedian(coefficients)),
        "median_standardized_pressure_gap_coefficient_ci95": list(coefficient_ci),
        "median_fisher_z_standardized_pressure_gap_coefficient": float(
            np.nanmedian(fisher_coefficients)
        ),
        "median_fisher_z_standardized_pressure_gap_coefficient_ci95": list(
            fisher_coefficient_ci
        ),
        "median_design_condition_number": float(scan_metrics["design_condition_number"].median()),
        "interpretation": "descriptive protocol-sensitive adjustment; fit contains W pressure response and is not a pure nuisance regressor",
    }
    write_json(out_dir / "whole_pattern_control_summary.json", summary)
    return summary


def compare_npz_array(current_path: Path, reference_path: Path, key: str) -> dict[str, Any]:
    """Compare one exact numeric array while treating paired NaNs as equal."""
    with np.load(current_path, allow_pickle=False) as current, np.load(
        reference_path, allow_pickle=False
    ) as reference:
        left = np.asarray(current[key], dtype=float)
        right = np.asarray(reference[key], dtype=float)
    shape_match = left.shape == right.shape
    if not shape_match:
        return {
            "shape_match": False,
            "current_shape": list(left.shape),
            "reference_shape": list(right.shape),
            "nan_pattern_match": False,
            "max_abs_difference": np.nan,
        }
    nan_pattern_match = bool(np.array_equal(np.isnan(left), np.isnan(right)))
    finite = np.isfinite(left) & np.isfinite(right)
    maximum = float(np.max(np.abs(left[finite] - right[finite]))) if np.any(finite) else 0.0
    return {
        "shape_match": True,
        "current_shape": list(left.shape),
        "reference_shape": list(right.shape),
        "nan_pattern_match": nan_pattern_match,
        "max_abs_difference": maximum,
    }


def validate_run(out_dir: Path, baseline_root: Path) -> dict[str, Any]:
    """Run method-specific invariants and raw-Pearson baseline parity checks."""
    checks: list[dict[str, Any]] = []
    validation_config = json.loads((out_dir / "method_config.json").read_text(encoding="utf-8"))
    plots_expected = bool(validation_config.get("plots_generated", True))

    def add(check: str, passed: bool, detail: str, value: Any = None) -> None:
        checks.append(
            {
                "check": check,
                "passed": bool(passed),
                "value": json_ready(value),
                "detail": detail,
            }
        )

    for label, path in (
        ("single_crystal", out_dir / "single_crystal/per_peak/per_track_matrices.npz"),
        ("powder", out_dir / "powder/per_peak/per_track_matrices.npz"),
    ):
        with np.load(path, allow_pickle=False) as archive:
            for key in (
                "location_profile_by_scan",
                "centroid_consistency_by_scan",
                "area_by_scan",
            ):
                values = np.asarray(archive[key], dtype=float)
                finite = values[np.isfinite(values)]
                in_range = bool(len(finite) and np.all((finite >= 0.0) & (finite <= 1.0)))
                add(
                    f"{label}_{key}_finite_unit_interval",
                    in_range,
                    "Finite Gaussian similarities must lie in [0, 1]; NaN remains missing/unknown.",
                    {"finite": len(finite), "min": np.min(finite) if len(finite) else np.nan,
                     "max": np.max(finite) if len(finite) else np.nan},
                )

            track_ids = np.asarray(archive["track_ids"], dtype=int)
            expected_tracks = len(track_ids)
            export_root = out_dir / "per_peak_heatmaps" / label
            index_path = export_root / "index.csv"
            exported_index = pd.read_csv(index_path) if index_path.is_file() else pd.DataFrame()
            location_pngs = sorted((export_root / "location").glob("track_*.png"))
            area_pngs = sorted((export_root / "area").glob("track_*.png"))
            location_csvs = sorted((export_root / "location_matrices").glob("track_*.csv"))
            area_csvs = sorted((export_root / "area_matrices").glob("track_*.csv"))
            index_tracks_match = bool(
                len(exported_index) == expected_tracks
                and "track" in exported_index
                and np.array_equal(
                    np.sort(exported_index["track"].to_numpy(int)), np.sort(track_ids)
                )
            )
            export_counts_pass = bool(
                len(location_csvs) == expected_tracks
                and len(area_csvs) == expected_tracks
                and index_tracks_match
                and (
                    not plots_expected
                    or (
                        len(location_pngs) == expected_tracks
                        and len(area_pngs) == expected_tracks
                        and all(item.stat().st_size > 1000 for item in location_pngs + area_pngs)
                    )
                )
            )
            add(
                f"{label}_per_track_peak_heatmap_export_complete",
                export_counts_pass,
                "Every peak track has explicit location/area matrix exports and, when enabled, PNG heatmaps; singleton tracks remain explicit all-NaN plots.",
                {
                    "expected_tracks": expected_tracks,
                    "location_png": len(location_pngs),
                    "area_png": len(area_pngs),
                    "location_csv": len(location_csvs),
                    "area_csv": len(area_csvs),
                    "index_tracks_match": index_tracks_match,
                    "plots_expected": plots_expected,
                },
            )

            matrix_export_pass = True
            maximum_difference = 0.0
            for folder, key in (
                ("location_matrices", "location_profile_aggregate"),
                ("area_matrices", "area_aggregate"),
            ):
                expected_matrices = np.asarray(archive[key], dtype=float)
                for index, track in enumerate(track_ids):
                    csv_path = export_root / folder / f"track_{int(track):03d}.csv"
                    if not csv_path.is_file():
                        matrix_export_pass = False
                        continue
                    actual = pd.read_csv(csv_path, index_col=0).to_numpy(float)
                    expected = expected_matrices[index]
                    if actual.shape != expected.shape:
                        matrix_export_pass = False
                        continue
                    masks_equal = np.array_equal(np.isfinite(actual), np.isfinite(expected))
                    upper_hidden = not np.isfinite(
                        actual[np.triu_indices_from(actual, k=0)]
                    ).any()
                    if not masks_equal or not upper_hidden:
                        matrix_export_pass = False
                        continue
                    finite = np.isfinite(expected)
                    difference = (
                        float(np.max(np.abs(actual[finite] - expected[finite])))
                        if np.any(finite) else 0.0
                    )
                    maximum_difference = max(maximum_difference, difference)
                    if difference > 1.0e-7:
                        matrix_export_pass = False
            add(
                f"{label}_per_track_matrix_exports_match_npz_and_strict_lower",
                matrix_export_pass,
                "Per-track CSVs exactly preserve the NPZ matrices; diagonal/upper triangle and missing lower cells remain NaN.",
                {"maximum_abs_difference": maximum_difference},
            )

            overview_paths = [
                out_dir / "per_peak_heatmaps" / f"{label}_location_all_tracks.png",
                out_dir / "per_peak_heatmaps" / f"{label}_area_all_tracks.png",
            ]
            overview_pass = bool(
                not plots_expected
                or all(item.is_file() and item.stat().st_size > 1000 for item in overview_paths)
            )
            add(
                f"{label}_per_track_overview_heatmaps_present",
                overview_pass,
                "A discoverable location and area contact sheet is present at the per_peak_heatmaps root.",
                {item.name: item.stat().st_size if item.is_file() else 0 for item in overview_paths},
            )

        observations = pd.read_csv(out_dir / f"{'single_crystal' if label == 'single_crystal' else 'powder'}/per_peak/observation_uncertainties.csv")
        fwhm_coverage = float(np.mean(np.isfinite(observations["fwhm_two_theta_deg"])))
        center_coverage = float(np.mean(np.isfinite(observations["centroid_se_two_theta_deg"])))
        if label == "powder":
            add(
                "powder_source_width_scale_coverage",
                fwhm_coverage >= 0.95,
                "At least 95% have a source q-width scale; this is not documented as fitted FWHM.",
                fwhm_coverage,
            )
        else:
            add(
                "single_crystal_second_moment_fwhm_coverage",
                fwhm_coverage >= 0.95,
                "At least 95% have a raw-2D second-moment FWHM scale.",
                fwhm_coverage,
            )
        add(
            f"{label}_centroid_uncertainty_coverage",
            center_coverage >= 0.95,
            (
                "At least 95% have an approximate width/SNR-derived center precision."
                if label == "powder"
                else "At least 95% have a raw-2D centroid precision estimate."
            ),
            center_coverage,
        )

    states = pd.read_csv(out_dir / "presence/presence_state_long.csv")
    transitions = pd.read_csv(out_dir / "presence/birth_death_transitions.csv")
    add(
        "presence_has_no_invented_absence",
        bool((states["absence_confirmed"] == 0).all() and not (states["state"] == "absent").any()),
        "A missing curated row is unknown, never silently converted to absence.",
        states["state"].value_counts().to_dict(),
    )
    rate_columns = ["birth_rate", "death_rate", "turnover_rate"]
    rates_are_nan = bool(transitions[rate_columns].isna().all().all())
    add(
        "birth_death_rates_not_claimed_without_absence_labels",
        rates_are_nan,
        "Birth/death rates are not identifiable without confirmed absence labels.",
        {column: int(transitions[column].notna().sum()) for column in rate_columns},
    )

    adjusted_pairs = pd.read_csv(
        out_dir / "whole_pattern/powder_control_adjustment/adjusted_pair_scores.csv"
    )
    stored_metrics = pd.read_csv(
        out_dir / "whole_pattern/powder_control_adjustment/scan_level_control_metrics.csv"
    ).set_index("scan")
    stored_partial = stored_metrics[
        "partial_r_spots_vs_pressure_gap_given_fit_order_protocol"
    ]
    stored_fisher_partial = stored_metrics[
        "partial_r_fisher_z_spots_vs_pressure_gap_given_fit_z_order_protocol"
    ]
    partial_differences: list[float] = []
    fisher_partial_differences: list[float] = []
    for scan, group in adjusted_pairs.groupby("scan", sort=True):
        controls = group[
            ["correlation_fit", "acquisition_order_gap", "acquisition_protocol_changed"]
        ].to_numpy(float)
        sample_residual, _ = residualize(group["correlation_spots"].to_numpy(float), controls)
        pressure_residual, _ = residualize(group["pressure_gap_GPa"].to_numpy(float), controls)
        expected = pearson(sample_residual, pressure_residual)
        partial_differences.append(abs(expected - float(stored_partial.loc[scan])))
        fisher_controls = group[
            [
                "correlation_fit_fisher_z", "acquisition_order_gap",
                "acquisition_protocol_changed",
            ]
        ].to_numpy(float)
        fisher_sample_residual, _ = residualize(
            group["correlation_spots_fisher_z"].to_numpy(float), fisher_controls
        )
        fisher_pressure_residual, _ = residualize(
            group["pressure_gap_GPa"].to_numpy(float), fisher_controls
        )
        expected_fisher = pearson(fisher_sample_residual, fisher_pressure_residual)
        fisher_partial_differences.append(
            abs(expected_fisher - float(stored_fisher_partial.loc[scan]))
        )
    maximum_partial_difference = max(partial_differences) if partial_differences else np.nan
    maximum_fisher_partial_difference = (
        max(fisher_partial_differences) if fisher_partial_differences else np.nan
    )
    add(
        "whole_pattern_partial_correlation_residualizes_both_variables",
        bool(
            np.isfinite(maximum_partial_difference)
            and maximum_partial_difference <= 1.0e-10
            and np.isfinite(maximum_fisher_partial_difference)
            and maximum_fisher_partial_difference <= 1.0e-10
        ),
        "Strict partial correlation residualizes both variables against fit/order/protocol controls in raw-r and Fisher-z sensitivity models.",
        {
            "raw_r_max_difference": maximum_partial_difference,
            "fisher_z_max_difference": maximum_fisher_partial_difference,
        },
    )

    candidates = pd.read_csv(out_dir / "window_to_window/boundary_candidates.csv")
    no_confounded_candidate = bool(
        (candidates.loc[candidates["protocol_confounded"] == 1, "candidate_boundary"] == 0).all()
    )
    add(
        "protocol_changed_intervals_are_not_sample_candidates",
        no_confounded_candidate,
        "An acquisition-protocol transition may remain visible as QC but cannot be promoted to a sample candidate.",
        {
            "protocol_confounded": int(candidates["protocol_confounded"].sum()),
            "confounded_promoted": int(
                ((candidates["protocol_confounded"] == 1) & (candidates["candidate_boundary"] == 1)).sum()
            ),
        },
    )
    no_low_support_candidate = bool(
        (candidates.loc[candidates["scan_support_eligible"] == 0, "candidate_boundary"] == 0).all()
    )
    add(
        "low_scan_support_intervals_are_not_sample_candidates",
        no_low_support_candidate,
        "Sparse skipped-pressure intervals remain visible but cannot enter the formal candidate set.",
        {
            "low_support_intervals": int((candidates["scan_support_eligible"] == 0).sum()),
            "low_support_promoted": int(
                ((candidates["scan_support_eligible"] == 0) & (candidates["candidate_boundary"] == 1)).sum()
            ),
        },
    )
    method_summaries = json.loads((out_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    single_peak_summary = method_summaries["per_peak"]["single_crystal"]
    powder_peak_summary = method_summaries["per_peak"]["powder"]
    window_summaries = method_summaries["windows"]
    weighting_rules_pass = bool(
        window_summaries["single_0deg"]["noise_weighting"]
        == "uniform_no_independent_same_pressure_replicates"
        and window_summaries["single_10deg"]["noise_weighting"]
        == "uniform_no_independent_same_pressure_replicates"
        and window_summaries["powder_spots"]["noise_weighting"]
        == "same_pressure_cross_scan_repeatability_MAD"
        and window_summaries["powder_fit"]["noise_weighting"]
        == "same_pressure_cross_scan_repeatability_MAD"
        and all(item["fixed_core_for_all_lags"] for item in window_summaries.values())
    )
    add(
        "window_weighting_uses_independent_repeats_or_uniform_fallback_and_fixed_core",
        weighting_rules_pass,
        "Powder weights come from same-pressure scan repeats; single is uniform; every lag uses the same 5° core.",
        {
            key: {
                "weighting": value["noise_weighting"],
                "fixed_core": value["fixed_core_for_all_lags"],
            }
            for key, value in window_summaries.items()
        },
    )
    single_uncalibrated_is_explicit = bool(
        single_peak_summary["area_repeatability"]["same_pressure_pairs"] == 0
        and "uncalibrated" in single_peak_summary["area_similarity_status"]
        and single_peak_summary["area_near_median"] is None
        and single_peak_summary["area_near_far_auc"] is None
    )
    add(
        "single_area_without_repeatability_is_not_reported_as_calibrated",
        single_uncalibrated_is_explicit,
        "Measurement-noise-only single-crystal area scores may be retained diagnostically, but formal summaries remain NA.",
        {
            "same_pressure_pairs": single_peak_summary["area_repeatability"]["same_pressure_pairs"],
            "status": single_peak_summary["area_similarity_status"],
            "formal_near_median": single_peak_summary["area_near_median"],
        },
    )
    single_area_pairs = pd.read_csv(out_dir / "single_crystal/per_peak/area_pair_scores.csv")
    same_orientation_only = bool(
        (single_area_pairs["orientation_high"] == single_area_pairs["orientation_low"]).all()
    )
    add(
        "single_area_pairs_do_not_cross_measurement_orientation",
        same_orientation_only,
        "0°/10° intensity geometry changes are not treated as pressure-driven ROI-area differences.",
        {
            "area_pairs": int(len(single_area_pairs)),
            "cross_orientation_pairs": int(
                (single_area_pairs["orientation_high"] != single_area_pairs["orientation_low"]).sum()
            ),
        },
    )
    powder_width_limit_is_explicit = bool(
        "not_documented" in powder_peak_summary["profile_width_status"]
    )
    add(
        "powder_source_width_is_not_overstated_as_verified_fit_fwhm",
        powder_width_limit_is_explicit,
        "Sparse source components provide an FWHM-like q width, not a documented fitted FWHM/covariance.",
        powder_peak_summary["profile_width_status"],
    )
    repeatability_balanced = bool(
        powder_peak_summary["area_repeatability"]["same_pressure_groups"] > 0
        and powder_peak_summary["area_repeatability"]["independent_tracks"] > 1
        and "equal-weight track" in powder_peak_summary["area_repeatability"]["aggregation"]
        and "equal-weight track" in powder_peak_summary["location_repeatability"]["aggregation"]
    )
    add(
        "powder_repeatability_uses_group_and_track_balancing",
        repeatability_balanced,
        "Combinatorial within-group pairs are summarized before independent-track bootstrap.",
        {
            "groups": powder_peak_summary["area_repeatability"]["same_pressure_groups"],
            "tracks": powder_peak_summary["area_repeatability"]["independent_tracks"],
            "aggregation": powder_peak_summary["area_repeatability"]["aggregation"],
        },
    )

    for label in ("single_0deg", "single_10deg", "powder_spots", "powder_fit"):
        archive_path = out_dir / f"windows/{label}/same_window_matrices.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            zero = np.asarray(archive["zero_shift_by_scan"], dtype=float)
            aligned = np.asarray(archive["aligned_by_scan"], dtype=float)
            finite = np.isfinite(zero) & np.isfinite(aligned)
            aligned_dominates = bool(np.all(aligned[finite] + 1.0e-7 >= zero[finite]))
            add(
                f"{label}_aligned_ncc_not_below_zero_shift",
                aligned_dominates,
                "Bounded alignment maximizes over a set that includes zero shift.",
                {"pairs": int(np.count_nonzero(finite)),
                 "min_gain": float(np.min(aligned[finite] - zero[finite])) if np.any(finite) else np.nan},
            )
            bounds: list[tuple[int, np.ndarray]] = []
            for key in archive.files:
                if key.startswith("aligned_bound_") and key.endswith("_bins"):
                    lag = int(key.removeprefix("aligned_bound_").removesuffix("_bins"))
                    bounds.append((lag, np.asarray(archive[key], dtype=float)))
            bounds.sort(key=lambda item: item[0])
            monotonic = True
            minimum_gain = np.inf
            for (_, smaller), (_, larger) in zip(bounds, bounds[1:]):
                common = np.isfinite(smaller) & np.isfinite(larger)
                if np.any(common):
                    difference = larger[common] - smaller[common]
                    minimum_gain = min(minimum_gain, float(np.min(difference)))
                    monotonic = monotonic and bool(np.all(difference >= -1.0e-7))
            add(
                f"{label}_shift_bound_sensitivity_monotonic",
                monotonic,
                "A wider allowed shift cannot reduce the maximum NCC.",
                {"bounds_bins": [item[0] for item in bounds],
                 "minimum_increment": minimum_gain if np.isfinite(minimum_gain) else np.nan},
            )

    parity_paths = {
        "single_0deg": (
            out_dir / "whole_pattern/single_0deg/whole_pattern_matrices.npz",
            baseline_root / "single_crystal/whole_and_windows/0deg/whole_pattern/whole_pattern_matrices.npz",
        ),
        "single_10deg": (
            out_dir / "whole_pattern/single_10deg/whole_pattern_matrices.npz",
            baseline_root / "single_crystal/whole_and_windows/10deg/whole_pattern/whole_pattern_matrices.npz",
        ),
        "powder_spots": (
            out_dir / "whole_pattern/powder_spots/whole_pattern_matrices.npz",
            baseline_root / "powder/whole_and_windows/spots/whole_pattern/whole_pattern_matrices.npz",
        ),
        "powder_fit": (
            out_dir / "whole_pattern/powder_fit/whole_pattern_matrices.npz",
            baseline_root / "powder/whole_and_windows/fit/whole_pattern/whole_pattern_matrices.npz",
        ),
    }
    for label, (current, reference) in parity_paths.items():
        comparison = compare_npz_array(current, reference, "matrices_by_scan")
        passed = bool(
            comparison["shape_match"]
            and comparison["nan_pattern_match"]
            and comparison["max_abs_difference"] <= 1.0e-6
        )
        add(
            f"{label}_raw_whole_pattern_parity_with_verified_baseline",
            passed,
            "The retained raw Pearson QC must reproduce the verified 2026-07-16 matrix.",
            comparison,
        )

    check_frame = pd.DataFrame(checks)
    validation_dir = out_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    check_frame.to_csv(validation_dir / "validation_checks.csv", index=False)
    report = {
        "status": "PASS" if bool(check_frame["passed"].all()) else "FAIL",
        "checks": int(len(check_frame)),
        "passed": int(check_frame["passed"].sum()),
        "failed": int((~check_frame["passed"]).sum()),
        "failed_checks": check_frame.loc[~check_frame["passed"], "check"].tolist(),
    }
    write_json(validation_dir / "validation_report.json", report)
    return report


def metric_text(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}" if np.isfinite(value) else "NA"
    return str(value)


def write_report(out_dir: Path, summaries: dict[str, Any], validation: dict[str, Any]) -> None:
    single = summaries["per_peak"]["single_crystal"]
    powder = summaries["per_peak"]["powder"]
    windows = summaries["windows"]
    whole = summaries["whole_pattern"]
    control = summaries["whole_pattern_control"]
    boundaries = summaries["window_boundaries"]
    presence = summaries["presence"]
    lines = [
        "# UOTe uncertainty-aware correlation reanalysis",
        "",
        f"**Computational validation: {validation['status']} ({validation['passed']}/{validation['checks']} checks passed).** This verifies implementation invariants and baseline parity, not the physical hypothesis.",
        "",
        "This is a new method-development result set. The verified 2026-07-16 legacy baseline was read-only and remains the formal baseline until this method is scientifically reviewed.",
        "",
        "## What changed",
        "",
        "| Metric | Before | Now | What it addresses / remaining limit |",
        "|---|---|---|---|",
        "| Peak position | `clip(1-|Δ2θ|/0.06°,0,1)` | FWHM-scaled Gaussian center-distance similarity plus a separate centroid z-consistency diagnostic | Removes the universal 0.06° cutoff; it is not a full Gaussian-profile overlap integral |",
        "| ROI area | smaller/larger per-pixel intensity | Gaussian similarity of integrated `Δlog(ROI excess/exposure)` scaled by measurement uncertainty and repeatability when available | Single crystal has no repeat calibration and remains diagnostic/NA; cross-orientation area pairs are omitted |",
        "| Same window across pressure | ACF and maximum over neighboring windows | Direct NCC on one fixed 5° core; zero-shift, aligned score and best shift are separate. Powder uses same-pressure cross-scan repeatability weights; single uses uniform weights | Avoids neighboring-window maximization and avoids estimating noise from pressure-driven changes |",
        "| Window-to-window | ACF of overlapping fixed windows | Correlation of ΔP-adjusted pressure-change trajectories; non-overlapping windows are primary | Targets coordinated changes rather than static visual resemblance |",
        "| Whole pattern | Direct Pearson only | Direct Pearson retained as QC, plus matched fit/W-background, acquisition order and protocol-mismatch adjustment; Fisher-z and same-protocol sensitivities are reported | Shows that the residual result is protocol/model-sensitive rather than one definitive corrected number |",
        "| Peak birth/death | Mixed with similarity or missing | Dedicated presence-state and birth/death tables | Missing is not treated as zero similarity or disappearance |",
        "| Missing data | NaN | NaN retained | Unknown remains unknown |",
        "",
        "## Key results",
        "",
        f"- Single-crystal per-peak: {single['raw_observations']} observations, {single['tracks']} tracks; median second-moment FWHM {metric_text(single['median_fwhm_two_theta_deg'], 4)}°; near median {metric_text(single['location_near_median'])}. Far pairs (≥15 GPa) do not exist, so location AUC is NA—not evidence of no pressure response.",
        f"- Powder per-peak: {powder['raw_observations']} observations, {powder['tracks']} tracks; median FWHM-like source width {metric_text(powder['median_fwhm_two_theta_deg'], 4)}°; location near/far AUC {metric_text(powder['location_near_far_auc'])}. The source does not document `q_width` as a fitted FWHM, so center precision is approximate.",
        f"- Area near/far AUC: single {metric_text(single['area_near_far_auc'])} (formal status: {single['area_similarity_status']}); powder {metric_text(powder['area_near_far_auc'])}. Single has zero same-pressure repeat groups. Powder repeat calibration uses {powder['area_repeatability']['same_pressure_groups']} track×pressure groups from {powder['area_repeatability']['independent_tracks']} tracks with track-cluster bootstrap; it still lacks a defensible per-observation area SE and remains secondary.",
        f"- Same-window aligned NCC near/far AUC: single 0° {metric_text(windows['single_0deg']['near_far_auc'])}; single 10° {metric_text(windows['single_10deg']['near_far_auc'])}; powder spots {metric_text(windows['powder_spots']['near_far_auc'])}; powder fit {metric_text(windows['powder_fit']['near_far_auc'])}.",
        f"- Whole-pattern raw Pearson versus pressure gap (QC): single 0° {metric_text(whole['single_0deg']['raw_r_correlation_vs_pressure_gap'])}; single 10° {metric_text(whole['single_10deg']['raw_r_correlation_vs_pressure_gap'])}; powder spots {metric_text(whole['powder_spots']['raw_r_correlation_vs_pressure_gap'])}; powder fit {metric_text(whole['powder_fit']['raw_r_correlation_vs_pressure_gap'])}.",
        f"- Powder control adjustment is model-sensitive. Median strict partial r with fit/order/protocol controls = {metric_text(control['median_scan_partial_r_given_fit_order_protocol'])} (95% descriptive CI {metric_text(control['median_scan_partial_r_given_fit_order_protocol_ci95'][0])} to {metric_text(control['median_scan_partial_r_given_fit_order_protocol_ci95'][1])}); without protocol control = {metric_text(control['median_scan_partial_r_without_protocol_control'])}; Fisher-z sensitivity = {metric_text(control['median_scan_partial_r_fisher_z_given_fit_z_order_protocol'])}; same-protocol-pairs sensitivity = {metric_text(control['median_scan_partial_r_same_protocol_pairs_given_fit_order'])}. Interpret direction/weak residual association, not one causal effect size.",
        f"- Window trajectory ranking found {boundaries['candidate_intervals']} protocol/support/effect-screened candidate interval(s) out of {boundaries['intervals']}, while {boundaries['statistical_positive_intervals']} interval(s) were merely statistical-positive. The strongest raw edge is {metric_text(boundaries['top_raw_high_pressure_GPa'], 2)} GPa (excess {metric_text(boundaries['top_raw_sample_specific_excess'])}); protocol-confounded={boundaries['top_raw_protocol_confounded']}. Sub-threshold statistical-positive rows remain visible in the CSV/workbook but are not promoted.",
        f"- Presence table: {presence['present']} present states, {presence['unknown']} unknown states, {presence['confirmed_absent']} confirmed absences. Therefore confirmed births/deaths are not identifiable from these curated tables.",
        "",
        "## Scientific interpretation",
        "",
        "The new plots measure how patterns, peaks and windows change along the pressure/acquisition staircase with explicit uncertainty, repeatability and control channels. They still do not by themselves prove a UOTe phase transition. The fit channel contains W/background pressure response and is not a pure nuisance variable, and the whole-pattern residual is sensitive to protocol handling, so adjusted results are descriptive rather than causal. Any interval with a filename-derived acquisition-mode change is retained as QC but excluded from sample-candidate status. Remaining candidates would require confirmation from peak indexing, independently labeled presence/absence, and preferably an independent experiment.",
        "",
        "## Output map",
        "",
        "- `single_crystal/per_peak/` and `powder/per_peak/`: uncertainty-enriched observations, pair scores, matrices and summaries.",
        "- `per_peak_heatmaps/`: every peak/track exported separately for location and area (PNG + CSV), plus four all-track overview sheets and an index. White cells are hidden diagonal/upper triangle; gray lower-triangle cells are NaN/unknown.",
        "- `windows/`: same-window zero-shift/aligned/shift outputs and pressure trajectories.",
        "- `window_to_window/`: matched spots-versus-fit candidate ranking using primary non-overlapping windows.",
        "- `whole_pattern/`: raw Pearson QC and powder control-adjusted residuals.",
        "- `presence/`: explicit present/unknown states and non-identifiable birth/death tables.",
        "- `validation/`: invariant checks and verified-baseline parity results.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = args.out_dir.resolve()
    baseline_root = args.baseline_root.resolve()
    data_root = args.data_root.resolve()
    handoff_root = args.handoff_root.resolve()
    if not baseline_root.exists():
        raise FileNotFoundError(f"Verified baseline not found: {baseline_root}")
    if (
        out_dir.exists()
        and any(out_dir.iterdir())
        and (out_dir / "run_manifest.json").exists()
        and not args.overwrite_completed
    ):
        raise FileExistsError(f"Refusing to overwrite a completed output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rng = np.random.default_rng(args.seed)
    make_plots = not args.no_plots

    single_observation_path = baseline_root / "single_crystal/per_peak_all_frames/track_observations.csv"
    powder_observation_path = baseline_root / "powder/per_peak/track_observations.csv"
    single_registry_path = baseline_root / "single_crystal/per_peak_all_frames/frame_registry.csv"
    powder_registry_path = baseline_root / "inputs/powder_frame_registry.csv"
    single_selected_path = baseline_root / "inputs/single_whole_selected.csv"
    powder_source_path = data_root / "Powder Scan/Track Analysis/spot_observations.csv"
    single_masked_dir = data_root / "Single Crystal (Cell 29)/Masked"

    single_observations = pd.read_csv(single_observation_path)
    powder_observations = pd.read_csv(powder_observation_path)
    single_registry = pd.read_csv(single_registry_path)
    powder_registry = pd.read_csv(powder_registry_path)

    single_enriched = estimate_single_roi_uncertainties(single_observations, single_masked_dir)
    powder_enriched = enrich_powder_uncertainties(powder_observations, powder_source_path)
    single_features = collapse_peak_observations(single_enriched)
    powder_features = collapse_peak_observations(powder_enriched)

    summaries: dict[str, Any] = {"per_peak": {}}
    summaries["per_peak"]["single_crystal"] = analyze_peak_dataset(
        out_dir / "single_crystal/per_peak", out_dir / "per_peak_heatmaps/single_crystal",
        single_enriched, single_features,
        single_registry, args.bootstrap_resamples, rng, make_plots,
    )
    summaries["per_peak"]["powder"] = analyze_peak_dataset(
        out_dir / "powder/per_peak", out_dir / "per_peak_heatmaps/powder",
        powder_enriched, powder_features,
        powder_registry, args.bootstrap_resamples, rng, make_plots,
    )
    write_per_peak_heatmap_landing(out_dir / "per_peak_heatmaps")
    summaries["presence"] = write_presence_tables(
        out_dir / "presence", single_features, powder_features, single_registry, powder_registry
    )

    single_series = load_single_series(single_selected_path, args.grid_step_deg)
    powder_series = load_powder_series(handoff_root, args.grid_step_deg)
    all_series = {
        "single_0deg": single_series["0deg"],
        "single_10deg": single_series["10deg"],
        "powder_spots": powder_series["spots"],
        "powder_fit": powder_series["fit"],
    }
    summaries["windows"] = {}
    window_trajectories: dict[str, pd.DataFrame] = {}
    for label, series in all_series.items():
        metric, trajectory = analyze_same_windows(
            out_dir / f"windows/{label}", series,
            args.window_width_deg, args.window_step_deg, args.max_shift_deg, make_plots,
        )
        summaries["windows"][label] = metric
        window_trajectories[label] = trajectory
    boundary_summary, _ = combine_powder_boundaries(
        out_dir / "window_to_window",
        window_trajectories["powder_spots"], window_trajectories["powder_fit"],
        args.window_width_deg, args.bootstrap_resamples, rng,
    )
    summaries["window_boundaries"] = boundary_summary

    summaries["whole_pattern"] = {}
    whole_pairs: dict[str, pd.DataFrame] = {}
    for label, series in all_series.items():
        metric, pairs = analyze_whole_pattern(out_dir / f"whole_pattern/{label}", series, make_plots)
        summaries["whole_pattern"][label] = metric
        whole_pairs[label] = pairs
    summaries["whole_pattern_control"] = control_adjust_whole_pattern(
        out_dir / "whole_pattern/powder_control_adjustment",
        whole_pairs["powder_spots"], whole_pairs["powder_fit"], powder_series["spots"].pressures,
        args.bootstrap_resamples, rng, make_plots,
    )

    method_config = {
        "profile": PROFILE,
        "date_tag": DATE_TAG,
        "peak_position_fwhm_scaled_center_distance_similarity": "exp(-0.5*delta_2theta^2/(sigma_fwhm_1^2+sigma_fwhm_2^2+u_center_1^2+u_center_2^2)); center-distance score, not full profile-overlap integral",
        "peak_centroid_consistency": "exp(-0.5*(delta_2theta/sqrt(u_center_1^2+u_center_2^2+tau_repeat^2))^2)",
        "peak_area_similarity": "exp(-0.5*(delta_log_area/sqrt(u_log_area_1^2+u_log_area_2^2+tau_repeat^2))^2)",
        "peak_area_value": "integrated background-subtracted ROI excess counts divided by exposure seconds; not mean intensity per pixel",
        "single_area_geometry_rule": "same-orientation comparisons only; cross 0deg/10deg area pairs are NaN",
        "single_area_uncertainty": "raw 2D ROI sideband robust-noise propagation; no same-pressure repeats available, so formal summary is NA",
        "powder_area_uncertainty": "track-balanced same-pressure independent-scan log-area repeatability only; no fabricated per-observation area SE",
        "same_window_shape": "fixed-core direct NCC; powder bins weighted by same-pressure cross-scan repeatability MAD; single bins uniform because no independent replicates",
        "same_window_shift": {"official_bound_deg": args.max_shift_deg, "sensitivity_deg": [0.06, 0.12, 0.18]},
        "window_to_window": "delta-P-adjusted adjacent-pressure shape-change trajectories; non-overlap primary windows; candidate CI bootstraps independent scan medians",
        "window_candidate_protocol_screen": "filename-derived detector/acquisition signature changes are QC-only and excluded from sample-candidate eligibility",
        "window_candidate_practical_screen": "positive excess must exceed robust interval center plus 3 scaled MAD, in addition to scan-bootstrap CI and sign consistency",
        "window_candidate_support_screen": "at least max(10, 80% of maximum interval scan support) independent scans",
        "whole_pattern": "raw Pearson QC retained; powder spots additionally adjusted for matched fit channel, acquisition-order gap, and filename-derived protocol mismatch; raw-r, Fisher-z, no-protocol, and same-protocol sensitivities reported",
        "presence": "curated observation means present; curated missing means unknown; absence/birth/death not inferred",
        "missing": "NaN",
        "strict_lower_triangle_display": True,
        "plots_generated": make_plots,
        "window_width_deg": args.window_width_deg,
        "window_step_deg": args.window_step_deg,
        "grid_step_deg": args.grid_step_deg,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
    }
    write_json(out_dir / "method_config.json", method_config)
    write_json(out_dir / "summary_metrics.json", summaries)

    selected_registry = pd.read_csv(single_selected_path)
    input_groups: dict[str, list[Path]] = {
        "verified_baseline_report": [baseline_root / "REPORT.md"],
        "single_curated_observations": [single_observation_path],
        "powder_curated_observations": [powder_observation_path],
        "single_frame_registry": [single_registry_path],
        "powder_frame_registry": [powder_registry_path],
        "single_whole_pattern_selection": [single_selected_path],
        "powder_source_peak_table": [powder_source_path],
        "single_geometry": [single_masked_dir / "_geometry.poni"],
        "powder_handoff_manifest": [handoff_root / "manifest.csv"],
        "analysis_runner": [Path(__file__).resolve()],
        "legacy_preprocessing_runner": [
            SCRIPT_DIR / "run_uote_xy_handoff_correlations.py"
        ],
        "single_raw_tiff": sorted(
            {Path(item).resolve() for item in single_observations["raw_tiff"].astype(str)}
        ),
        "single_frame_masks": [
            single_masked_dir / f"frame_{int(frame):04d}_mask.npy"
            for frame in sorted(single_observations["frame"].astype(int).unique())
        ],
        "single_whole_pattern_xy": sorted(
            {Path(item).resolve() for item in selected_registry["file_path"].astype(str)}
        ),
        "powder_spots_xy": sorted((handoff_root / "spots_channel").rglob("*.xy")),
        "powder_fit_xy": sorted((handoff_root / "fit_channel").rglob("*.xy")),
    }
    input_file_rows: list[dict[str, Any]] = []
    input_group_rows: list[dict[str, Any]] = []
    for role, paths in input_groups.items():
        resolved = [path.resolve() for path in paths]
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing files in provenance group {role}: {missing[:3]}")
        for path in resolved:
            input_file_rows.append(
                {
                    "role": role, "path": str(path), "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        input_group_rows.append(
            {
                "role": role, "files": len(resolved),
                "total_bytes": int(sum(path.stat().st_size for path in resolved)),
                "file_set_sha256": sha256_file_set(resolved),
            }
        )
    pd.DataFrame(input_file_rows).to_csv(out_dir / "input_file_manifest.csv", index=False)
    pd.DataFrame(input_group_rows).to_csv(out_dir / "input_audit.csv", index=False)

    validation = validate_run(out_dir, baseline_root)
    write_report(out_dir, summaries, validation)
    analysis_output_rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name in {"run_manifest.json", "analysis_output_manifest.csv"}:
            continue
        if path.suffix.lower() == ".xlsx" or "workbook_qa" in path.parts:
            continue
        analysis_output_rows.append(
            {
                "relative_path": str(path.relative_to(out_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(analysis_output_rows).to_csv(
        out_dir / "analysis_output_manifest.csv", index=False
    )
    manifest = {
        "profile": PROFILE,
        "date_tag": DATE_TAG,
        "status": validation["status"],
        "created_by": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "legacy_preprocessing_runner_sha256": sha256_file(
            SCRIPT_DIR / "run_uote_xy_handoff_correlations.py"
        ),
        "baseline_root": str(baseline_root),
        "baseline_preserved": True,
        "output_root": str(out_dir),
        "elapsed_seconds": time.time() - start,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "arguments": vars(args),
        "validation": validation,
        "report_sha256": sha256_file(out_dir / "REPORT.md"),
        "method_config_sha256": sha256_file(out_dir / "method_config.json"),
        "summary_metrics_sha256": sha256_file(out_dir / "summary_metrics.json"),
        "input_file_manifest_sha256": sha256_file(out_dir / "input_file_manifest.csv"),
        "analysis_output_manifest_sha256": sha256_file(
            out_dir / "analysis_output_manifest.csv"
        ),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps(json_ready({"output": out_dir, "validation": validation}), indent=2))


if __name__ == "__main__":
    main()
