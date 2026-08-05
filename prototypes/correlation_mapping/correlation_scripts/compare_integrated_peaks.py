#!/usr/bin/env python3
"""Compare integrated XRD frames using peak and full-pattern similarity."""

from __future__ import annotations

import argparse
import csv
import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.spatial.distance import pdist, squareform


PRESSURE_RE = re.compile(r"(?P<value>\d+(?:p\d+)?|\d+(?:\.\d+)?)\s*GPa", re.IGNORECASE)


@dataclass
class Pattern:
    label: str
    path: Path
    two_theta: np.ndarray
    intensity: np.ndarray
    pressure_gpa: float | None


@dataclass
class Peak:
    pattern_label: str
    path: Path
    two_theta: float
    intensity: float
    prominence: float
    width_deg: float
    group_index: int | None = None
    group_two_theta: float | None = None
    source_methods: str = ""
    source_count: int = 0
    confidence_tier: str = "B"
    tier_score: float = 0.0
    accepted_standard: bool = True
    raw_intensity: float = 0.0
    residual_height: float = 0.0
    conventional_prominence: float = 0.0
    matched_filter_responses: str = ""
    best_matched_scale: int | None = None
    best_matched_response: float = 0.0
    local_center_height: float = 0.0
    left_side_rise: float = 0.0
    right_side_rise: float = 0.0
    two_sided_contrast: float = 0.0
    local_noise: float = 0.0
    local_shape_snr: float = 0.0
    local_slope: float = 0.0
    local_curvature: float = 0.0
    local_center_percentile: float = 0.0
    local_monotonic_fraction: float = 0.0
    local_is_max: bool = False
    nearest_stronger_peak_deg: float = math.nan
    likely_shoulder: bool = False
    top_peaks_truncated: bool = False


@dataclass
class CandidateRecord:
    index: int
    sources: set[str] = field(default_factory=set)
    prominences: dict[str, float] = field(default_factory=dict)
    matched_responses: dict[int, float] = field(default_factory=dict)

    def merge(self, other: "CandidateRecord") -> None:
        self.sources.update(other.sources)
        for source, prominence in other.prominences.items():
            self.prominences[source] = max(self.prominences.get(source, 0.0), prominence)
        for width, response in other.matched_responses.items():
            self.matched_responses[width] = max(self.matched_responses.get(width, 0.0), response)

    @property
    def max_prominence(self) -> float:
        values = list(self.prominences.values()) + list(self.matched_responses.values())
        return max(values) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build XRD frame correlation matrices from integrated .xy files: "
            "peak-presence Jaccard, peak-intensity cosine, peak-position shift, "
            "and full-pattern Pearson."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("Data/Cell_29_integrated")],
        help="Integrated .xy files or directories containing .xy files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/frame_peak_similarity_cell29"),
        help="Output directory for CSV matrices and heatmaps.",
    )
    parser.add_argument(
        "--peak-match-tolerance",
        type=float,
        default=0.08,
        help="2theta tolerance in degrees for grouping peaks across frames.",
    )
    parser.add_argument(
        "--min-two-theta",
        type=float,
        default=2.0,
        help="Ignore low-angle region below this 2theta value.",
    )
    parser.add_argument(
        "--max-two-theta",
        type=float,
        default=None,
        help="Optional upper 2theta bound.",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=1.5,
        help="Minimum peak prominence after baseline subtraction.",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=8,
        help="Minimum distance between detected peaks, in samples.",
    )
    parser.add_argument(
        "--bump-prominence",
        type=float,
        default=0.6,
        help=(
            "Minimum prominence, in percent of normalized intensity, for broad "
            "local bumps detected on the smoothed .xy curve."
        ),
    )
    parser.add_argument(
        "--min-peak-height",
        type=float,
        default=0.5,
        help=(
            "Minimum positive residual height, in percent of normalized intensity, "
            "required for any accepted peak/bump."
        ),
    )
    parser.add_argument(
        "--bump-rise-window",
        type=float,
        default=0.25,
        help="Half-window in degrees used to verify that broad bumps rise above both sides.",
    )
    parser.add_argument(
        "--min-bump-rise",
        type=float,
        default=1.0,
        help=(
            "Minimum rise, in percent of normalized intensity, above both local "
            "sides for broad bump candidates."
        ),
    )
    parser.add_argument(
        "--micro-prominence",
        type=float,
        default=1.5,
        help=(
            "Minimum prominence, in percent of normalized intensity, for tiny "
            "local protrusions detected with a narrow smoothing window."
        ),
    )
    parser.add_argument(
        "--raw-prominence",
        type=float,
        default=0.0,
        help=(
            "Minimum prominence, in percent of normalized intensity, for local "
            "maxima detected directly on the normalized raw .xy curve."
        ),
    )
    parser.add_argument(
        "--matched-filter-prominence",
        type=float,
        default=0.0,
        help=(
            "Minimum prominence, in percent-like matched-filter units, for "
            "multi-scale peak-template candidates."
        ),
    )
    parser.add_argument(
        "--matched-filter-widths",
        default="2,3,5,8",
        help="Comma-separated matched-filter kernel widths in samples.",
    )
    parser.add_argument(
        "--shoulder-prominence",
        type=float,
        default=0.0,
        help=(
            "Minimum prominence, in percent-like curvature units, for shoulder "
            "candidates detected from negative second derivative maxima."
        ),
    )
    parser.add_argument(
        "--min-shape-snr",
        type=float,
        default=0.0,
        help="Minimum local detrended peak-shape SNR for raw/shoulder/matched candidates.",
    )
    parser.add_argument(
        "--min-shape-contrast",
        type=float,
        default=0.0,
        help="Minimum local detrended two-sided contrast for raw/shoulder/matched candidates.",
    )
    parser.add_argument(
        "--shape-half-window",
        type=float,
        default=0.16,
        help="Half-window in degrees used to score whether a candidate is locally peak-shaped.",
    )
    parser.add_argument(
        "--micro-smooth-window",
        type=int,
        default=7,
        help="Odd Savitzky-Golay smoothing window for tiny local protrusions.",
    )
    parser.add_argument(
        "--min-micro-snr",
        type=float,
        default=0.5,
        help="Minimum local robust SNR required for micro-peak candidates.",
    )
    parser.add_argument(
        "--merge-tolerance",
        type=float,
        default=0.05,
        help="Merge detections closer than this many degrees 2theta.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=21,
        help="Odd Savitzky-Golay smoothing window for peak finding.",
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=151,
        help="Odd Savitzky-Golay window for broad background subtraction.",
    )
    parser.add_argument(
        "--top-peaks",
        type=int,
        default=80,
        help="Maximum peaks kept per frame, ranked by prominence.",
    )
    parser.add_argument(
        "--pattern-grid-step",
        type=float,
        default=0.02,
        help="Common 2theta grid step for full-pattern Pearson matrix.",
    )
    parser.add_argument(
        "--small-peak-max-prominence",
        type=float,
        default=0.35,
        help="Maximum normalized prominence for peaks kept in small_peak_table.csv.",
    )
    parser.add_argument(
        "--small-peak-max-width",
        type=float,
        default=2.0,
        help="Maximum FWHM-like width in degrees for peaks kept in small_peak_table.csv.",
    )
    parser.add_argument(
        "--review-plots",
        action="store_true",
        help="Write per-frame .xy review plots with all peaks and small-peak candidates marked.",
    )
    parser.add_argument(
        "--roi-half-width",
        type=float,
        default=0.06,
        help="Half-width in degrees 2theta for peak ROI area integration.",
    )
    parser.add_argument(
        "--roi-sideband-gap",
        type=float,
        default=0.03,
        help="Gap in degrees between peak ROI and sidebands for local background.",
    )
    parser.add_argument(
        "--roi-sideband-width",
        type=float,
        default=0.08,
        help="Width in degrees of each sideband for local background estimation.",
    )
    parser.add_argument(
        "--spot-csv",
        type=Path,
        default=Path(
            "outputs/auto_ring_filter_batch_radius_groups_cell29_10deg_by_pressure_legend_outside/all_spots.csv"
        ),
        help="Optional all_spots.csv from the 2D spot/radius pipeline for radius-azimuth correlation.",
    )
    parser.add_argument(
        "--spot-center-x",
        type=float,
        default=514.782,
        help="Beam center x pixel for azimuth calculation from 2D spot coordinates.",
    )
    parser.add_argument(
        "--spot-center-y",
        type=float,
        default=537.738,
        help="Beam center y pixel for azimuth calculation from 2D spot coordinates.",
    )
    parser.add_argument(
        "--spot-radius-bin",
        type=float,
        default=25.0,
        help="Radius bin width in pixels for 2D spot radius-azimuth vectors.",
    )
    parser.add_argument(
        "--spot-azimuth-bin",
        type=float,
        default=30.0,
        help="Azimuth bin width in degrees for 2D spot radius-azimuth vectors.",
    )
    parser.add_argument(
        "--spot-value",
        choices=["count", "max_intensity", "mean_intensity", "max_z"],
        default="max_intensity",
        help="Spot value accumulated in radius-azimuth bins.",
    )
    return parser.parse_args()


def discover_xy_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            files.extend(sorted(input_path.rglob("*.xy")))
        elif input_path.suffix.lower() == ".xy":
            files.append(input_path)
    return sorted(dict.fromkeys(files), key=lambda path: natural_sort_key(path.name))


def natural_sort_key(text: str) -> list[object]:
    parts = re.split(r"(\d+(?:p\d+)?|\d+(?:\.\d+)?)", text)
    key: list[object] = []
    for part in parts:
        normalized = part.replace("p", ".")
        try:
            key.append(float(normalized))
        except ValueError:
            key.append(part.lower())
    return key


def parse_pressure(path: Path) -> float | None:
    match = PRESSURE_RE.search(str(path))
    if not match:
        return None
    return float(match.group("value").replace("p", "."))


def label_for_path(path: Path) -> str:
    pressure = parse_pressure(path)
    if pressure is None:
        return path.stem
    pressure_text = f"{pressure:g}GPa"
    if "decomp" in path.stem.lower() or "decomp" in str(path.parent).lower():
        return f"{pressure_text}_decomp"
    return pressure_text


def ensure_unique_pattern_labels(patterns: list[Pattern]) -> list[Pattern]:
    """Keep pressure-first labels readable without collapsing repeated frames."""
    counts = Counter(pattern.label for pattern in patterns)
    used: set[str] = set()
    for pattern in patterns:
        base = pattern.label
        if counts[base] > 1:
            candidate = f"{base} [{pattern.path.stem}]"
        else:
            candidate = base
        suffix = 2
        unique = candidate
        while unique in used:
            unique = f"{candidate} #{suffix}"
            suffix += 1
        pattern.label = unique
        used.add(unique)
    return patterns


def load_pattern(path: Path, args: argparse.Namespace) -> Pattern:
    data = np.loadtxt(path, comments="#", encoding="latin1")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")

    two_theta = data[:, 0].astype(float)
    intensity = data[:, 1].astype(float)
    keep = np.isfinite(two_theta) & np.isfinite(intensity)
    keep &= two_theta >= args.min_two_theta
    if args.max_two_theta is not None:
        keep &= two_theta <= args.max_two_theta

    return Pattern(
        label=label_for_path(path),
        path=path,
        two_theta=two_theta[keep],
        intensity=intensity[keep],
        pressure_gpa=parse_pressure(path),
    )


def odd_window(requested: int, size: int, minimum: int = 5) -> int:
    if size < minimum:
        return max(3, size | 1)
    window = max(minimum, requested)
    window = min(window, size if size % 2 == 1 else size - 1)
    if window % 2 == 0:
        window -= 1
    return max(minimum, window)


def normalize_for_pattern(intensity: np.ndarray) -> np.ndarray:
    y = intensity.astype(float)
    y = y - np.nanpercentile(y, 5)
    scale = np.nanpercentile(y, 99)
    if scale <= 0:
        scale = np.nanmax(np.abs(y))
    if scale > 0:
        y = y / scale
    return y


def parse_width_list(text: str) -> list[int]:
    widths: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        widths.append(max(1, int(round(float(item)))))
    return widths or [2, 3, 5, 8]


def mexican_hat_kernel(width: int) -> np.ndarray:
    radius = max(3, int(math.ceil(width * 4)))
    t = np.arange(-radius, radius + 1, dtype=float) / max(float(width), 1.0)
    kernel = (1.0 - t**2) * np.exp(-0.5 * t**2)
    kernel = kernel - float(np.mean(kernel))
    norm = float(np.linalg.norm(kernel))
    if norm > 0:
        kernel = kernel / norm
    return kernel


def get_arg(args: argparse.Namespace, name: str, default: float | int | str) -> float | int | str:
    return getattr(args, name, default)


def robust_mad(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = float(np.nanmedian(finite))
    return 1.4826 * float(np.nanmedian(np.abs(finite - median)))


def local_shape_metrics(
    two_theta: np.ndarray,
    signal: np.ndarray,
    index: int,
    half_window_deg: float,
) -> dict[str, float]:
    empty = {
        "center_height": 0.0,
        "left_rise": 0.0,
        "right_rise": 0.0,
        "two_sided_contrast": 0.0,
        "local_noise": 0.0,
        "local_shape_snr": 0.0,
        "local_slope": 0.0,
        "local_curvature": 0.0,
        "center_percentile": 0.0,
        "monotonic_fraction": 0.0,
        "is_local_max": 0.0,
    }
    step = abs(float(np.nanmedian(np.diff(two_theta))))
    half_window = max(4, int(round(half_window_deg / step)))
    left = max(0, index - half_window)
    right = min(len(signal), index + half_window + 1)
    center = index - left
    if right - left < 7 or center < 2 or center > right - left - 3:
        return empty

    local_x = two_theta[left:right] - two_theta[index]
    local_y = signal[left:right]
    exclude = max(1, int(round(0.03 / step)))
    sample_indices = np.arange(len(local_y))
    side_mask = np.abs(sample_indices - center) > exclude
    if np.count_nonzero(side_mask) < 4:
        side_mask = np.ones(len(local_y), dtype=bool)
        side_mask[center] = False

    slope, intercept = np.polyfit(local_x[side_mask], local_y[side_mask], 1)
    detrended = local_y - (slope * local_x + intercept)
    left_side = detrended[:center]
    right_side = detrended[center + 1 :]
    if len(left_side) == 0 or len(right_side) == 0:
        return empty

    center_value = float(detrended[center])
    left_base = float(np.nanpercentile(left_side, 30))
    right_base = float(np.nanpercentile(right_side, 30))
    left_rise = center_value - left_base
    right_rise = center_value - right_base
    contrast = min(left_rise, right_rise)
    side_values = detrended[side_mask]
    noise = robust_mad(side_values)
    snr = contrast / (noise + 1e-9)
    if len(detrended) >= 5:
        curvature = -float(np.gradient(np.gradient(detrended))[center])
    else:
        curvature = 0.0
    finite_detrended = detrended[np.isfinite(detrended)]
    if finite_detrended.size:
        center_percentile = float(np.mean(finite_detrended <= center_value))
    else:
        center_percentile = 0.0
    diffs = np.diff(detrended)
    finite_diffs = diffs[np.isfinite(diffs)]
    if finite_diffs.size:
        pos_fraction = float(np.mean(finite_diffs > 0))
        neg_fraction = float(np.mean(finite_diffs < 0))
        monotonic_fraction = max(pos_fraction, neg_fraction)
    else:
        monotonic_fraction = 0.0
    max_window = max(1, int(round(0.025 / step)))
    peak_left = max(0, center - max_window)
    peak_right = min(len(detrended), center + max_window + 1)
    local_is_max = bool(center_value >= float(np.nanmax(detrended[peak_left:peak_right])) - 1e-12)
    return {
        "center_height": float(center_value),
        "left_rise": float(left_rise),
        "right_rise": float(right_rise),
        "two_sided_contrast": float(contrast),
        "local_noise": float(noise),
        "local_shape_snr": float(snr),
        "local_slope": float(slope),
        "local_curvature": float(curvature),
        "center_percentile": float(center_percentile),
        "monotonic_fraction": float(monotonic_fraction),
        "is_local_max": float(local_is_max),
    }


def local_shape_score(
    two_theta: np.ndarray,
    signal: np.ndarray,
    index: int,
    half_window_deg: float,
) -> tuple[float, float]:
    metrics = local_shape_metrics(two_theta, signal, index, half_window_deg)
    return metrics["two_sided_contrast"], metrics["local_shape_snr"]


def add_candidate_record(
    records: list[CandidateRecord],
    index: int,
    source: str,
    prominence: float,
    matched_width: int | None = None,
) -> None:
    record = CandidateRecord(index=int(index))
    record.sources.add(source)
    record.prominences[source] = float(prominence)
    if matched_width is not None:
        record.matched_responses[int(matched_width)] = float(prominence)
    records.append(record)


def choose_record_index(first: CandidateRecord, second: CandidateRecord) -> int:
    return first.index if first.max_prominence >= second.max_prominence else second.index


def merge_candidate_records(
    records: list[CandidateRecord],
    two_theta: np.ndarray,
    tolerance: float,
) -> list[CandidateRecord]:
    if not records:
        return []
    sorted_records = sorted(records, key=lambda record: (float(two_theta[record.index]), -record.max_prominence))
    merged: list[CandidateRecord] = []
    for record in sorted_records:
        if not merged:
            merged.append(record)
            continue
        last = merged[-1]
        same_index = record.index == last.index
        close = abs(float(two_theta[record.index] - two_theta[last.index])) <= tolerance
        if same_index or close:
            best_index = choose_record_index(last, record)
            last.merge(record)
            last.index = best_index
        else:
            merged.append(record)
    return merged


def matched_response_text(responses: dict[int, float]) -> str:
    if not responses:
        return ""
    return ";".join(f"w{width}:{response:.6g}" for width, response in sorted(responses.items()))


def candidate_tier(
    record: CandidateRecord,
    metrics: dict[str, float],
    conventional_prominence: float,
    width_deg: float,
    raw_intensity: float,
    args: argparse.Namespace,
) -> tuple[str, float, bool]:
    sources = record.sources
    source_count = len(sources)
    contrast = metrics["two_sided_contrast"]
    left_rise = metrics["left_rise"]
    right_rise = metrics["right_rise"]
    snr = metrics["local_shape_snr"]
    center_height = metrics["center_height"]
    best_matched = max(record.matched_responses.values()) if record.matched_responses else 0.0
    relative_contrast = contrast / max(raw_intensity, 1e-6)
    center_percentile = metrics.get("center_percentile", 0.0)
    monotonic_fraction = metrics.get("monotonic_fraction", 0.0)
    is_local_max = bool(metrics.get("is_local_max", 0.0))
    min_shape_snr = float(get_arg(args, "min_shape_snr", 0.0))
    min_shape_contrast = float(get_arg(args, "min_shape_contrast", 0.0))
    has_one_sided_protrusion = max(left_rise, right_rise) > max(min_shape_contrast, 0.0)
    has_two_sided_protrusion = contrast > max(min_shape_contrast, 0.0)
    protected_peak_source = bool(sources & {"residual", "bump", "micro"})
    weak_shape_ok = (
        center_height > 0
        and has_one_sided_protrusion
        and snr > max(0.05, min_shape_snr * 0.5)
        and (is_local_max or protected_peak_source or has_two_sided_protrusion)
    )
    likely_curvature_only = sources == {"shoulder"} and not weak_shape_ok
    likely_noise_only = source_count == 1 and sources <= {"raw", "shoulder", "matched"} and snr < max(0.15, min_shape_snr) and contrast <= min_shape_contrast
    monotonic_tail = (
        source_count == 1
        and sources <= {"raw", "shoulder", "matched"}
        and monotonic_fraction >= 0.82
        and not has_two_sided_protrusion
    )
    plateau_or_ripple = (
        source_count == 1
        and sources <= {"raw", "shoulder", "matched"}
        and center_percentile < 0.62
        and contrast < max(0.003, min_shape_contrast * 3.0)
        and snr < 0.75
    )
    nonmax_noise = (
        source_count == 1
        and sources <= {"raw", "shoulder", "matched"}
        and not is_local_max
        and not has_two_sided_protrusion
        and snr < 1.0
    )
    broad_matched_tail = (
        sources == {"matched"}
        and record.matched_responses
        and max(record.matched_responses) >= 8
        and snr < max(0.35, min_shape_snr)
        and not has_two_sided_protrusion
    )
    duplicate_or_tail_width = width_deg > 0.8 and sources <= {"raw", "shoulder", "matched"} and snr < 0.25
    high_background_ripple = (
        raw_intensity > 0.45
        and contrast < 0.012
        and relative_contrast < 0.015
        and not (sources & {"residual", "bump"} and conventional_prominence >= 0.01)
    )

    if (
        likely_curvature_only
        or likely_noise_only
        or monotonic_tail
        or plateau_or_ripple
        or nonmax_noise
        or broad_matched_tail
        or duplicate_or_tail_width
        or high_background_ripple
    ):
        tier = "C"
    elif (
        (source_count >= 2 and has_two_sided_protrusion and snr >= max(0.35, min_shape_snr))
        or (best_matched >= 1.5 and has_two_sided_protrusion)
        or (sources & {"residual", "bump", "micro"} and snr >= max(0.5, min_shape_snr))
    ):
        tier = "A"
    elif weak_shape_ok or sources & {"residual", "bump", "micro"} or has_one_sided_protrusion:
        tier = "B"
    else:
        tier = "C"

    tier_rank = {"A": 2.0, "B": 1.0, "C": 0.0}[tier]
    score = (
        tier_rank * 1000.0
        + source_count * 100.0
        + max(snr, 0.0) * 10.0
        + best_matched
        + max(conventional_prominence, 0.0)
    )
    return tier, float(score), tier != "C"


def nearest_stronger_distance(peaks: list[Peak], index: int) -> float:
    peak = peaks[index]
    stronger = [
        abs(other.two_theta - peak.two_theta)
        for other_index, other in enumerate(peaks)
        if other_index != index and other.tier_score > peak.tier_score
    ]
    return min(stronger) if stronger else math.nan


def detect_peaks(pattern: Pattern, args: argparse.Namespace) -> list[Peak]:
    x = pattern.two_theta
    y = normalize_for_pattern(pattern.intensity)
    if len(x) < 20:
        return []

    smooth_w = odd_window(args.smooth_window, len(y))
    baseline_w = odd_window(args.baseline_window, len(y), minimum=smooth_w + 2)
    smoothed = savgol_filter(y, smooth_w, polyorder=2)
    baseline = savgol_filter(y, baseline_w, polyorder=2)
    residual = smoothed - baseline
    micro_w = odd_window(args.micro_smooth_window, len(y), minimum=5)
    micro_smoothed = savgol_filter(y, micro_w, polyorder=2)
    micro_residual = micro_smoothed - baseline
    shoulder_signal = -np.gradient(np.gradient(micro_smoothed))
    matched_signal = micro_residual
    matched_noise = robust_mad(matched_signal)
    if matched_noise <= 0:
        matched_noise = 1.0

    residual_indices, residual_props = find_peaks(
        residual,
        prominence=args.prominence / 100.0,
        distance=args.distance,
    )
    bump_indices, bump_props = find_peaks(
        smoothed,
        prominence=args.bump_prominence / 100.0,
        distance=args.distance,
    )
    micro_indices, micro_props = find_peaks(
        micro_smoothed,
        prominence=args.micro_prominence / 100.0,
        distance=max(1, args.distance // 2),
    )
    raw_indices, raw_props = find_peaks(
        y,
        prominence=args.raw_prominence / 100.0,
        distance=args.distance,
    )
    shoulder_indices, shoulder_props = find_peaks(
        shoulder_signal,
        prominence=args.shoulder_prominence / 100.0,
        distance=args.distance,
    )
    records: list[CandidateRecord] = []
    for index, prominence in zip(residual_indices, residual_props["prominences"]):
        add_candidate_record(records, int(index), "residual", float(prominence))
    for index, prominence in zip(bump_indices, bump_props["prominences"]):
        add_candidate_record(records, int(index), "bump", float(prominence))
    for index, prominence in zip(micro_indices, micro_props["prominences"]):
        add_candidate_record(records, int(index), "micro", float(prominence))
    for index, prominence in zip(raw_indices, raw_props["prominences"]):
        add_candidate_record(records, int(index), "raw", float(prominence))
    for index, prominence in zip(shoulder_indices, shoulder_props["prominences"]):
        add_candidate_record(records, int(index), "shoulder", float(prominence))
    for width in parse_width_list(args.matched_filter_widths):
        kernel = mexican_hat_kernel(width)
        response = np.convolve(matched_signal, kernel, mode="same") / (matched_noise + 1e-9)
        indices, props = find_peaks(
            response,
            prominence=float(get_arg(args, "matched_filter_prominence", 0.0)),
            distance=args.distance,
        )
        for index, prominence in zip(indices, props["prominences"]):
            add_candidate_record(records, int(index), "matched", float(prominence), matched_width=width)

    if len(records) == 0:
        return []

    records = merge_candidate_records(records, x, float(get_arg(args, "merge_tolerance", 0.0)))
    step = float(np.nanmedian(np.diff(x)))
    peaks: list[Peak] = []
    for record in records:
        index = record.index
        sources = record.sources
        if "residual" in sources:
            signal = residual
            intensity_value = residual[index]
        elif "micro" in sources:
            signal = micro_smoothed
            intensity_value = max(micro_residual[index], record.max_prominence)
        elif "raw" in sources:
            signal = y
            intensity_value = max(y[index] - baseline[index], record.max_prominence)
        elif "shoulder" in sources:
            signal = shoulder_signal
            intensity_value = max(micro_residual[index], record.max_prominence)
        elif "matched" in sources:
            signal = y
            intensity_value = max(micro_residual[index], record.max_prominence)
        else:
            signal = smoothed
            intensity_value = smoothed[index] - baseline[index]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                width = float(peak_widths(signal, np.array([index]), rel_height=0.5)[0][0])
        except Exception:
            width = 0.0
        width_deg = float(width * abs(step))
        shape = local_shape_metrics(
            x,
            micro_smoothed,
            index,
            float(get_arg(args, "shape_half_window", 0.16)),
        )
        conventional_prominence = max(
            (value for source, value in record.prominences.items() if source != "matched"),
            default=0.0,
        )
        tier, tier_score, accepted_standard = candidate_tier(
            record,
            shape,
            conventional_prominence,
            width_deg,
            float(y[index]),
            args,
        )
        best_scale = None
        best_response = 0.0
        if record.matched_responses:
            best_scale, best_response = max(record.matched_responses.items(), key=lambda item: item[1])
        likely_shoulder = bool(
            "shoulder" in sources
            or (shape["left_rise"] > 0 and shape["right_rise"] <= 0)
            or (shape["right_rise"] > 0 and shape["left_rise"] <= 0)
        )
        peaks.append(
            Peak(
                pattern_label=pattern.label,
                path=pattern.path,
                two_theta=float(x[index]),
                intensity=float(max(intensity_value, 0.0)),
                prominence=float(record.max_prominence),
                width_deg=width_deg,
                source_methods=";".join(sorted(sources)),
                source_count=len(sources),
                confidence_tier=tier,
                tier_score=tier_score,
                accepted_standard=accepted_standard,
                raw_intensity=float(y[index]),
                residual_height=float(micro_residual[index]),
                conventional_prominence=float(conventional_prominence),
                matched_filter_responses=matched_response_text(record.matched_responses),
                best_matched_scale=best_scale,
                best_matched_response=float(best_response),
                local_center_height=shape["center_height"],
                left_side_rise=shape["left_rise"],
                right_side_rise=shape["right_rise"],
                two_sided_contrast=shape["two_sided_contrast"],
                local_noise=shape["local_noise"],
                local_shape_snr=shape["local_shape_snr"],
                local_slope=shape["local_slope"],
                local_curvature=shape["local_curvature"],
                local_center_percentile=shape["center_percentile"],
                local_monotonic_fraction=shape["monotonic_fraction"],
                local_is_max=bool(shape["is_local_max"]),
                likely_shoulder=likely_shoulder,
            )
        )
    peaks = sorted(peaks, key=lambda peak: peak.tier_score, reverse=True)
    truncated = len(peaks) > int(get_arg(args, "top_peaks", 80))
    peaks = peaks[: int(get_arg(args, "top_peaks", 80))]
    for index, peak in enumerate(peaks):
        peak.nearest_stronger_peak_deg = nearest_stronger_distance(peaks, index)
        peak.top_peaks_truncated = truncated
    return sorted(peaks, key=lambda peak: peak.two_theta)


def filter_peak_candidates(
    candidates: list[tuple[int, float, str]],
    two_theta: np.ndarray,
    smoothed: np.ndarray,
    residual: np.ndarray,
    micro_smoothed: np.ndarray,
    micro_residual: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[int, float, str]]:
    min_height = args.min_peak_height / 100.0
    min_bump_rise = args.min_bump_rise / 100.0
    step = float(np.nanmedian(np.diff(two_theta)))
    window_samples = max(2, int(round(args.bump_rise_window / abs(step))))
    noise_window_samples = max(5, int(round(0.4 / abs(step))))
    filtered: list[tuple[int, float, str]] = []

    for index, prominence, source in candidates:
        if source in {"raw", "shoulder", "matched"}:
            if args.min_shape_snr > 0 or args.min_shape_contrast > 0:
                contrast, shape_snr = local_shape_score(
                    two_theta,
                    micro_smoothed,
                    index,
                    args.shape_half_window,
                )
                if contrast < args.min_shape_contrast:
                    continue
                if shape_snr < args.min_shape_snr:
                    continue
            filtered.append((index, prominence, source))
            continue

        height_signal = micro_residual if source == "micro" else residual
        shape_signal = micro_smoothed if source == "micro" else smoothed
        residual_height = float(height_signal[index])
        if source != "micro" and residual_height < min_height:
            continue

        if source in {"bump", "micro"}:
            left = max(0, index - window_samples)
            right = min(len(shape_signal), index + window_samples + 1)
            left_min = float(np.min(shape_signal[left : index + 1]))
            right_min = float(np.min(shape_signal[index:right]))
            two_sided_rise = min(
                float(shape_signal[index]) - left_min,
                float(shape_signal[index]) - right_min,
            )
            min_rise = min_bump_rise
            if source == "micro":
                min_rise = min(min_bump_rise, args.micro_prominence / 100.0)
            if two_sided_rise < min_rise:
                continue

        if source == "micro":
            left = max(0, index - noise_window_samples)
            right = min(len(micro_residual), index + noise_window_samples + 1)
            local = micro_residual[left:right]
            local_noise = 1.4826 * float(np.median(np.abs(local - np.median(local))))
            local_baseline = float(np.median(micro_smoothed[left:right]))
            local_height = float(micro_smoothed[index]) - local_baseline
            if local_height < args.micro_prominence / 100.0:
                continue
            if local_height / (local_noise + 1e-9) < args.min_micro_snr:
                continue

        filtered.append((index, prominence, source))

    return filtered


def merge_close_peak_candidates(
    candidates: list[tuple[int, float, str]],
    two_theta: np.ndarray,
    tolerance: float,
) -> list[tuple[int, float, str]]:
    """Merge duplicate residual/bump detections while keeping distinct shoulders."""
    sorted_candidates = sorted(candidates, key=lambda item: two_theta[item[0]])
    merged: list[tuple[int, float, str]] = []

    for candidate in sorted_candidates:
        if not merged:
            merged.append(candidate)
            continue
        last = merged[-1]
        if abs(float(two_theta[candidate[0]] - two_theta[last[0]])) <= tolerance:
            if candidate[1] > last[1]:
                merged[-1] = candidate
        else:
            merged.append(candidate)
    return merged


def group_peaks(peaks: list[Peak], tolerance: float) -> list[float]:
    sorted_peaks = sorted(peaks, key=lambda peak: peak.two_theta)
    groups: list[list[Peak]] = []

    for peak in sorted_peaks:
        if not groups:
            groups.append([peak])
            continue
        center = float(np.mean([member.two_theta for member in groups[-1]]))
        if abs(peak.two_theta - center) <= tolerance:
            groups[-1].append(peak)
        else:
            groups.append([peak])

    centers: list[float] = []
    for group_index, members in enumerate(groups, start=1):
        weights = np.array([max(member.prominence, 1e-9) for member in members])
        positions = np.array([member.two_theta for member in members])
        center = float(np.average(positions, weights=weights))
        centers.append(center)
        for member in members:
            member.group_index = group_index
            member.group_two_theta = center
    return centers


def build_peak_vectors(
    patterns: list[Pattern], peaks: list[Peak], group_centers: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    label_to_row = {pattern.label: index for index, pattern in enumerate(patterns)}
    presence = np.zeros((len(patterns), len(group_centers)), dtype=float)
    intensity = np.zeros_like(presence)

    for peak in peaks:
        if peak.group_index is None:
            continue
        row = label_to_row[peak.pattern_label]
        col = peak.group_index - 1
        presence[row, col] = 1.0
        intensity[row, col] = max(intensity[row, col], peak.intensity)

    row_max = intensity.max(axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    intensity = intensity / row_max
    return presence, intensity


def build_peak_roi_area_vectors(
    patterns: list[Pattern], group_centers: list[float], args: argparse.Namespace
) -> np.ndarray:
    areas = np.zeros((len(patterns), len(group_centers)), dtype=float)
    for row, pattern in enumerate(patterns):
        x = pattern.two_theta
        y = normalize_for_pattern(pattern.intensity)
        for col, center in enumerate(group_centers):
            roi = (x >= center - args.roi_half_width) & (x <= center + args.roi_half_width)
            if not np.any(roi):
                continue
            left_side = (
                (x >= center - args.roi_half_width - args.roi_sideband_gap - args.roi_sideband_width)
                & (x < center - args.roi_half_width - args.roi_sideband_gap)
            )
            right_side = (
                (x > center + args.roi_half_width + args.roi_sideband_gap)
                & (x <= center + args.roi_half_width + args.roi_sideband_gap + args.roi_sideband_width)
            )
            sideband = left_side | right_side
            if np.any(sideband):
                background = float(np.nanmedian(y[sideband]))
            else:
                background = float(np.nanpercentile(y[roi], 10))
            signal = np.clip(y[roi] - background, 0.0, None)
            areas[row, col] = float(np.trapezoid(signal, x[roi]))
    return areas


def unique_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        if counts[label] == 1 and labels.count(label) == 1:
            unique.append(label)
        elif counts[label] == 1:
            unique.append(label)
        else:
            unique.append(f"{label}_{counts[label]}")
    if len(set(unique)) == len(unique):
        return unique
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for label in unique:
        seen[label] = seen.get(label, 0) + 1
        deduped.append(label if seen[label] == 1 else f"{label}_{seen[label]}")
    return deduped


def build_spot_radius_azimuth_vectors(
    spot_csv: Path, args: argparse.Namespace
) -> tuple[list[str], list[str], np.ndarray]:
    if not spot_csv.exists():
        return [], [], np.zeros((0, 0), dtype=float)

    rows: list[dict[str, str]] = []
    with spot_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        return [], [], np.zeros((0, 0), dtype=float)

    image_paths = list(dict.fromkeys(row["image"] for row in rows))
    labels = unique_labels([spot_label_for_image(Path(path)) for path in image_paths])
    image_to_row = {path: index for index, path in enumerate(image_paths)}

    radii = np.array([float(row["radius_px"]) for row in rows], dtype=float)
    max_radius = float(np.nanmax(radii)) if len(radii) else 0.0
    radius_edges = np.arange(0.0, max_radius + args.spot_radius_bin * 1.5, args.spot_radius_bin)
    azimuth_edges = np.arange(-180.0, 180.0 + args.spot_azimuth_bin, args.spot_azimuth_bin)
    radius_bins = len(radius_edges) - 1
    azimuth_bins = len(azimuth_edges) - 1
    features = np.zeros((len(image_paths), radius_bins * azimuth_bins), dtype=float)

    for row in rows:
        image_row = image_to_row[row["image"]]
        x = float(row["x"])
        y = float(row["y"])
        radius = float(row["radius_px"])
        azimuth = math.degrees(math.atan2(y - args.spot_center_y, x - args.spot_center_x))
        radius_index = int(np.searchsorted(radius_edges, radius, side="right") - 1)
        azimuth_index = int(np.searchsorted(azimuth_edges, azimuth, side="right") - 1)
        if not (0 <= radius_index < radius_bins and 0 <= azimuth_index < azimuth_bins):
            continue
        value = 1.0 if args.spot_value == "count" else float(row[args.spot_value])
        features[image_row, radius_index * azimuth_bins + azimuth_index] += value

    feature_names: list[str] = []
    for radius_index in range(radius_bins):
        r0 = radius_edges[radius_index]
        r1 = radius_edges[radius_index + 1]
        for azimuth_index in range(azimuth_bins):
            a0 = azimuth_edges[azimuth_index]
            a1 = azimuth_edges[azimuth_index + 1]
            feature_names.append(f"r{r0:.0f}-{r1:.0f}_az{a0:.0f}-{a1:.0f}")

    nonzero = np.any(features > 0, axis=0)
    return labels, [name for name, keep in zip(feature_names, nonzero) if keep], features[:, nonzero]


def spot_label_for_image(path: Path) -> str:
    pressure = parse_pressure(path)
    suffix = path.stem
    if pressure is None:
        return suffix
    return f"{pressure:g}GPa_{suffix}"


def jaccard_similarity(binary: np.ndarray) -> np.ndarray:
    if binary.shape[0] == 0:
        return np.zeros((0, 0))
    distances = squareform(pdist(binary.astype(bool), metric="jaccard"))
    similarity = 1.0 - distances
    np.fill_diagonal(similarity, 1.0)
    return similarity


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    similarity = safe @ safe.T
    for index, norm in enumerate(norms[:, 0]):
        similarity[index, index] = 1.0 if norm > 0 else np.nan
    return similarity


def peak_shift_matrix(
    patterns: list[Pattern], peaks: list[Peak], group_count: int
) -> tuple[np.ndarray, np.ndarray]:
    label_to_row = {pattern.label: index for index, pattern in enumerate(patterns)}
    positions = np.full((len(patterns), group_count), np.nan)
    for peak in peaks:
        if peak.group_index is None:
            continue
        row = label_to_row[peak.pattern_label]
        col = peak.group_index - 1
        positions[row, col] = peak.two_theta

    rms = np.full((len(patterns), len(patterns)), np.nan)
    mean_shift = np.full_like(rms, np.nan)
    for i in range(len(patterns)):
        for j in range(len(patterns)):
            common = np.isfinite(positions[i]) & np.isfinite(positions[j])
            if not np.any(common):
                continue
            delta = positions[j, common] - positions[i, common]
            rms[i, j] = math.sqrt(float(np.mean(delta**2)))
            mean_shift[i, j] = float(np.mean(delta))
    return rms, mean_shift


def full_pattern_pearson(patterns: list[Pattern], step: float) -> tuple[np.ndarray, np.ndarray]:
    lower = max(pattern.two_theta.min() for pattern in patterns)
    upper = min(pattern.two_theta.max() for pattern in patterns)
    grid = np.arange(lower, upper + step / 2.0, step)
    vectors = []
    for pattern in patterns:
        y = np.interp(grid, pattern.two_theta, pattern.intensity)
        y = normalize_for_pattern(y)
        y = y - np.mean(y)
        std = np.std(y)
        if std > 0:
            y = y / std
        vectors.append(y)
    matrix = np.vstack(vectors)
    return np.corrcoef(matrix), grid


def write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[format_value(value) for value in row]])


def format_value(value: float) -> str:
    if np.isnan(value):
        return ""
    return f"{value:.6g}"


def write_feature_table(
    path: Path, labels: list[str], centers: list[float], matrix: np.ndarray
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", *[f"{center:.4f}" for center in centers]])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[format_value(value) for value in row]])


def write_named_feature_table(
    path: Path, labels: list[str], feature_names: list[str], matrix: np.ndarray
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", *feature_names])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[format_value(value) for value in row]])


def write_peak_table(path: Path, peaks: list[Peak]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "path",
                "two_theta",
                "normalized_intensity",
                "prominence",
                "width_deg",
                "peak_group",
                "group_two_theta",
                "delta_from_group",
                "source_methods",
                "source_count",
                "confidence_tier",
                "tier_score",
                "accepted_standard",
                "raw_intensity",
                "residual_height",
                "conventional_prominence",
                "matched_filter_responses",
                "best_matched_scale",
                "best_matched_response",
                "local_center_height",
                "left_side_rise",
                "right_side_rise",
                "two_sided_contrast",
                "local_noise",
                "local_shape_snr",
                "local_slope",
                "local_curvature",
                "local_center_percentile",
                "local_monotonic_fraction",
                "local_is_max",
                "width_estimate_deg",
                "nearest_stronger_peak_deg",
                "likely_shoulder",
                "top_peaks_truncated",
            ]
        )
        for peak in sorted(peaks, key=lambda item: (item.pattern_label, item.two_theta)):
            group_center = peak.group_two_theta
            delta = peak.two_theta - group_center if group_center is not None else np.nan
            writer.writerow(
                [
                    peak.pattern_label,
                    str(peak.path),
                    f"{peak.two_theta:.6f}",
                    f"{peak.intensity:.6g}",
                    f"{peak.prominence:.6g}",
                    f"{peak.width_deg:.6g}",
                    peak.group_index or "",
                    f"{group_center:.6f}" if group_center is not None else "",
                    format_value(delta),
                    peak.source_methods,
                    peak.source_count,
                    peak.confidence_tier,
                    f"{peak.tier_score:.6g}",
                    int(bool(peak.accepted_standard)),
                    f"{peak.raw_intensity:.6g}",
                    f"{peak.residual_height:.6g}",
                    f"{peak.conventional_prominence:.6g}",
                    peak.matched_filter_responses,
                    peak.best_matched_scale if peak.best_matched_scale is not None else "",
                    f"{peak.best_matched_response:.6g}",
                    f"{peak.local_center_height:.6g}",
                    f"{peak.left_side_rise:.6g}",
                    f"{peak.right_side_rise:.6g}",
                    f"{peak.two_sided_contrast:.6g}",
                    f"{peak.local_noise:.6g}",
                    f"{peak.local_shape_snr:.6g}",
                    f"{peak.local_slope:.6g}",
                    f"{peak.local_curvature:.6g}",
                    f"{peak.local_center_percentile:.6g}",
                    f"{peak.local_monotonic_fraction:.6g}",
                    int(bool(peak.local_is_max)),
                    f"{peak.width_deg:.6g}",
                    format_value(peak.nearest_stronger_peak_deg),
                    int(bool(peak.likely_shoulder)),
                    int(bool(peak.top_peaks_truncated)),
                ]
            )


def select_small_peak_candidates(peaks: list[Peak], args: argparse.Namespace) -> list[Peak]:
    return [
        peak
        for peak in peaks
        if peak.prominence <= args.small_peak_max_prominence
        and peak.width_deg <= args.small_peak_max_width
    ]


def write_group_table(
    path: Path,
    labels: list[str],
    centers: list[float],
    presence: np.ndarray,
    intensity: np.ndarray,
    peaks: list[Peak] | None = None,
) -> None:
    peaks_by_group: dict[int, list[Peak]] = {}
    for peak in peaks or []:
        if peak.group_index is not None:
            peaks_by_group.setdefault(peak.group_index, []).append(peak)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "peak_group",
                "group_two_theta",
                "frame_count",
                "frames_present",
                "max_normalized_intensity",
                "tier_a_count",
                "tier_b_count",
                "tier_c_count",
                "source_methods",
            ]
        )
        for col, center in enumerate(centers):
            present_indices = np.flatnonzero(presence[:, col] > 0)
            members = peaks_by_group.get(col + 1, [])
            tier_counts = {
                tier: sum(1 for peak in members if peak.confidence_tier == tier)
                for tier in ("A", "B", "C")
            }
            sources = sorted(
                {
                    source
                    for peak in members
                    for source in peak.source_methods.split(";")
                    if source
                }
            )
            writer.writerow(
                [
                    col + 1,
                    f"{center:.6f}",
                    len(present_indices),
                    ";".join(labels[index] for index in present_indices),
                    format_value(float(np.max(intensity[:, col]))),
                    tier_counts["A"],
                    tier_counts["B"],
                    tier_counts["C"],
                    ";".join(sources),
                ]
            )


def longest_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def write_peak_group_summary(
    path: Path,
    patterns: list[Pattern],
    centers: list[float],
    peaks: list[Peak],
    roi_area: np.ndarray,
    presence: np.ndarray,
) -> None:
    peaks_by_group: dict[int, list[Peak]] = {}
    for peak in peaks:
        if peak.group_index is not None:
            peaks_by_group.setdefault(peak.group_index, []).append(peak)
    label_to_pressure = {pattern.label: pattern.pressure_gpa for pattern in patterns}
    total_frames = len(patterns)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "peak_group",
                "group_two_theta",
                "frame_count",
                "frame_coverage_fraction",
                "longest_consecutive_run",
                "median_position",
                "position_mad_deg",
                "pressure_position_slope_deg_per_gpa",
                "max_roi_area",
                "mean_confidence_tier",
                "tier_a_count",
                "tier_b_count",
                "tier_c_count",
                "source_methods",
                "matched_filter_scales",
                "frames_present",
            ]
        )
        for col, center in enumerate(centers):
            members = peaks_by_group.get(col + 1, [])
            present = presence[:, col] > 0
            positions = np.array([peak.two_theta for peak in members], dtype=float)
            pressures = np.array(
                [
                    label_to_pressure.get(peak.pattern_label, np.nan)
                    if label_to_pressure.get(peak.pattern_label) is not None
                    else np.nan
                    for peak in members
                ],
                dtype=float,
            )
            median = float(np.nanmedian(positions)) if positions.size else np.nan
            mad = (
                float(np.nanmedian(np.abs(positions - median)))
                if positions.size and np.isfinite(median)
                else np.nan
            )
            finite = np.isfinite(pressures) & np.isfinite(positions)
            slope = np.nan
            if np.count_nonzero(finite) >= 2 and float(np.nanmax(pressures[finite]) - np.nanmin(pressures[finite])) > 0:
                slope = float(np.polyfit(pressures[finite], positions[finite], 1)[0])
            tier_values = {"A": 3.0, "B": 2.0, "C": 1.0}
            tier_counts = {
                tier: sum(1 for peak in members if peak.confidence_tier == tier)
                for tier in ("A", "B", "C")
            }
            mean_tier = (
                float(np.mean([tier_values.get(peak.confidence_tier, np.nan) for peak in members]))
                if members
                else np.nan
            )
            sources = sorted(
                {
                    source
                    for peak in members
                    for source in peak.source_methods.split(";")
                    if source
                }
            )
            scales = sorted(
                {
                    str(peak.best_matched_scale)
                    for peak in members
                    if peak.best_matched_scale is not None
                }
            )
            frames = [
                pattern.label
                for pattern, is_present in zip(patterns, present)
                if bool(is_present)
            ]
            writer.writerow(
                [
                    col + 1,
                    f"{center:.6f}",
                    int(np.count_nonzero(present)),
                    format_value(float(np.count_nonzero(present) / max(total_frames, 1))),
                    longest_true_run(present),
                    format_value(median),
                    format_value(mad),
                    format_value(slope),
                    format_value(float(np.nanmax(roi_area[:, col])) if roi_area.size else np.nan),
                    format_value(mean_tier),
                    tier_counts["A"],
                    tier_counts["B"],
                    tier_counts["C"],
                    ";".join(sources),
                    ";".join(scales),
                    ";".join(frames),
                ]
            )


def plot_peak_review(
    out_dir: Path,
    patterns: list[Pattern],
    peaks: list[Peak],
    small_peaks: list[Peak],
) -> None:
    review_dir = out_dir / "xy_peak_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    peaks_by_label: dict[str, list[Peak]] = {}
    small_by_label: dict[str, list[Peak]] = {}
    for peak in peaks:
        peaks_by_label.setdefault(peak.pattern_label, []).append(peak)
    for peak in small_peaks:
        small_by_label.setdefault(peak.pattern_label, []).append(peak)

    for pattern in patterns:
        x = pattern.two_theta
        y = normalize_for_pattern(pattern.intensity)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(x, y, color="0.2", lw=0.8, label="integrated .xy")

        frame_peaks = peaks_by_label.get(pattern.label, [])
        tier_styles = {
            "A": dict(marker="o", facecolors="none", edgecolors="#f28e2b", s=34, linewidths=1.2, label="Tier A strong"),
            "B": dict(marker="x", color="#00bcd4", s=34, linewidths=1.4, label="Tier B weak/shoulder"),
            "C": dict(marker=".", color="0.55", s=16, linewidths=0.8, label="Tier C diagnostic"),
        }
        for tier in ("A", "B", "C"):
            tier_peaks = [peak for peak in frame_peaks if peak.confidence_tier == tier]
            if not tier_peaks:
                continue
            peak_x = np.array([peak.two_theta for peak in tier_peaks])
            peak_y = np.interp(peak_x, x, y)
            ax.scatter(peak_x, peak_y, **tier_styles[tier])

        frame_small = small_by_label.get(pattern.label, [])
        if frame_small and len(frame_small) < 0.85 * max(len(frame_peaks), 1):
            small_x = np.array([peak.two_theta for peak in frame_small])
            small_y = np.interp(small_x, x, y)
            ax.scatter(
                small_x,
                small_y,
                s=48,
                facecolors="none",
                edgecolors="#2ca02c",
                marker="s",
                linewidths=1.5,
                label="small-peak candidates",
            )

        ax.set_title(f"{pattern.label}: .xy peaks")
        ax.set_xlabel("2theta (deg)")
        ax.set_ylabel("normalized intensity")
        ax.legend(loc="upper right")
        fig.tight_layout()
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", pattern.label)
        fig.savefig(review_dir / f"{safe_label}_xy_peak_review.png", dpi=180)
        plt.close(fig)


def plot_heatmap(
    path: Path,
    labels: list[str],
    matrix: np.ndarray,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    size = max(6, 0.45 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xy_files = discover_xy_files(args.inputs)
    if not xy_files:
        raise SystemExit("No .xy files found.")

    patterns = [load_pattern(path, args) for path in xy_files]
    patterns = sorted(
        patterns,
        key=lambda pattern: (
            float("inf") if pattern.pressure_gpa is None else pattern.pressure_gpa,
            pattern.label,
        ),
    )
    ensure_unique_pattern_labels(patterns)

    peaks = [peak for pattern in patterns for peak in detect_peaks(pattern, args)]
    group_centers = group_peaks(peaks, args.peak_match_tolerance)
    labels = [pattern.label for pattern in patterns]

    presence, peak_intensity = build_peak_vectors(patterns, peaks, group_centers)
    roi_area = build_peak_roi_area_vectors(patterns, group_centers, args)
    roi_area_cosine = cosine_similarity(roi_area)
    jaccard = jaccard_similarity(presence)
    cosine = cosine_similarity(peak_intensity)
    rms_shift, mean_shift = peak_shift_matrix(patterns, peaks, len(group_centers))
    pearson, grid = full_pattern_pearson(patterns, args.pattern_grid_step)
    spot_labels, spot_feature_names, spot_features = build_spot_radius_azimuth_vectors(args.spot_csv, args)
    spot_cosine = cosine_similarity(spot_features) if len(spot_labels) else np.zeros((0, 0))
    spot_jaccard = (
        jaccard_similarity((spot_features > 0).astype(float)) if len(spot_labels) else np.zeros((0, 0))
    )

    write_peak_table(args.out_dir / "peak_table.csv", peaks)
    small_peaks = select_small_peak_candidates(peaks, args)
    write_peak_table(args.out_dir / "small_peak_table.csv", small_peaks)
    small_presence, small_peak_intensity = build_peak_vectors(patterns, small_peaks, group_centers)
    small_jaccard = jaccard_similarity(small_presence)
    small_cosine = cosine_similarity(small_peak_intensity)
    small_rms_shift, small_mean_shift = peak_shift_matrix(patterns, small_peaks, len(group_centers))
    write_group_table(args.out_dir / "peak_group_table.csv", labels, group_centers, presence, peak_intensity, peaks)
    write_peak_group_summary(
        args.out_dir / "peak_group_summary.csv",
        patterns,
        group_centers,
        peaks,
        roi_area,
        presence,
    )
    write_feature_table(args.out_dir / "A_peak_roi_area_features.csv", labels, group_centers, roi_area)
    write_feature_table(args.out_dir / "B_peak_presence_features.csv", labels, group_centers, presence)
    write_feature_table(args.out_dir / "aux_peak_intensity_features.csv", labels, group_centers, peak_intensity)
    write_feature_table(args.out_dir / "aux_small_peak_presence_features.csv", labels, group_centers, small_presence)
    write_feature_table(args.out_dir / "aux_small_peak_intensity_features.csv", labels, group_centers, small_peak_intensity)
    write_matrix(args.out_dir / "A_peak_roi_area_cosine_matrix.csv", labels, roi_area_cosine)
    write_matrix(args.out_dir / "B_peak_presence_jaccard_matrix.csv", labels, jaccard)
    write_matrix(args.out_dir / "C_peak_position_rms_shift_matrix_deg.csv", labels, rms_shift)
    write_matrix(args.out_dir / "aux_C_peak_position_mean_shift_matrix_deg.csv", labels, mean_shift)
    write_matrix(args.out_dir / "aux_peak_intensity_cosine_matrix.csv", labels, cosine)
    write_matrix(args.out_dir / "aux_full_pattern_pearson_matrix.csv", labels, pearson)
    write_matrix(args.out_dir / "E_full_integrated_pattern_pearson_matrix.csv", labels, pearson)
    write_matrix(args.out_dir / "aux_small_peak_presence_jaccard_matrix.csv", labels, small_jaccard)
    write_matrix(args.out_dir / "aux_small_peak_intensity_cosine_matrix.csv", labels, small_cosine)
    write_matrix(args.out_dir / "aux_small_peak_position_rms_shift_matrix_deg.csv", labels, small_rms_shift)
    write_matrix(args.out_dir / "aux_small_peak_position_mean_shift_matrix_deg.csv", labels, small_mean_shift)
    if len(spot_labels):
        write_named_feature_table(
            args.out_dir / "D_spot_radius_azimuth_features.csv",
            spot_labels,
            spot_feature_names,
            spot_features,
        )
        write_matrix(args.out_dir / "D_spot_radius_azimuth_cosine_matrix.csv", spot_labels, spot_cosine)
        write_matrix(args.out_dir / "aux_D_spot_radius_azimuth_jaccard_matrix.csv", spot_labels, spot_jaccard)
    np.savetxt(args.out_dir / "full_pattern_common_two_theta_grid.csv", grid, delimiter=",")

    plot_heatmap(
        args.out_dir / "A_peak_roi_area_cosine_heatmap.png",
        labels,
        roi_area_cosine,
        "A. Peak ROI area cosine correlation",
        "viridis",
        vmin=0,
        vmax=1,
    )
    plot_heatmap(
        args.out_dir / "B_peak_presence_jaccard_heatmap.png",
        labels,
        jaccard,
        "B. Peak presence Jaccard correlation",
        "viridis",
        vmin=0,
        vmax=1,
    )
    plot_heatmap(
        args.out_dir / "aux_peak_intensity_cosine_heatmap.png",
        labels,
        cosine,
        "Aux. Peak intensity cosine similarity",
        "viridis",
        vmin=0,
        vmax=1,
    )
    plot_heatmap(
        args.out_dir / "C_peak_position_rms_shift_heatmap_deg.png",
        labels,
        rms_shift,
        "C. Peak position RMS shift (deg 2theta)",
        "magma",
        vmin=0,
        vmax=np.nanpercentile(rms_shift, 95),
    )
    plot_heatmap(
        args.out_dir / "aux_full_pattern_pearson_heatmap.png",
        labels,
        pearson,
        "Aux. Full integrated pattern Pearson correlation",
        "viridis",
        vmin=-1,
        vmax=1,
    )
    plot_heatmap(
        args.out_dir / "E_full_integrated_pattern_pearson_heatmap.png",
        labels,
        pearson,
        "E. Full integrated pattern Pearson correlation",
        "viridis",
        vmin=-1,
        vmax=1,
    )
    plot_heatmap(
        args.out_dir / "aux_small_peak_presence_jaccard_heatmap.png",
        labels,
        small_jaccard,
        "Aux. Small peaks: presence Jaccard similarity",
        "viridis",
        vmin=0,
        vmax=1,
    )
    plot_heatmap(
        args.out_dir / "aux_small_peak_intensity_cosine_heatmap.png",
        labels,
        small_cosine,
        "Aux. Small peaks: intensity cosine similarity",
        "viridis",
        vmin=0,
        vmax=1,
    )
    plot_heatmap(
        args.out_dir / "aux_small_peak_position_rms_shift_heatmap_deg.png",
        labels,
        small_rms_shift,
        "Aux. Small peaks: RMS shift (deg 2theta)",
        "magma",
        vmin=0,
        vmax=np.nanpercentile(small_rms_shift, 95),
    )
    if len(spot_labels):
        plot_heatmap(
            args.out_dir / "D_spot_radius_azimuth_cosine_heatmap.png",
            spot_labels,
            spot_cosine,
            "D. 2D spot radius-azimuth cosine correlation",
            "viridis",
            vmin=0,
            vmax=1,
        )
        plot_heatmap(
            args.out_dir / "aux_D_spot_radius_azimuth_jaccard_heatmap.png",
            spot_labels,
            spot_jaccard,
            "Aux. 2D spot radius-azimuth presence Jaccard",
            "viridis",
            vmin=0,
            vmax=1,
        )
    if args.review_plots:
        plot_peak_review(args.out_dir, patterns, peaks, small_peaks)

    summary_path = args.out_dir / "summary.txt"
    with summary_path.open("w") as handle:
        handle.write(f"Input files: {len(patterns)}\n")
        handle.write(f"Detected peaks: {len(peaks)}\n")
        handle.write(f"Small-peak candidates: {len(small_peaks)}\n")
        handle.write(f"Matched peak groups: {len(group_centers)}\n")
        handle.write(f"Peak match tolerance: {args.peak_match_tolerance} deg 2theta\n")
        handle.write(f"Prominence threshold: {args.prominence / 100.0:g} normalized residual\n")
        handle.write(f"Small-peak max prominence: {args.small_peak_max_prominence:g}\n")
        handle.write(f"Small-peak max width: {args.small_peak_max_width:g} deg\n")
        handle.write(f"Peak ROI half-width: {args.roi_half_width:g} deg\n")
        handle.write(f"Common full-pattern grid points: {len(grid)}\n")
        if len(spot_labels):
            handle.write(f"2D spot frames: {len(spot_labels)}\n")
            handle.write(f"2D spot features: {len(spot_feature_names)}\n")
            handle.write(f"2D spot CSV: {args.spot_csv}\n")
            handle.write(f"2D spot radius bin: {args.spot_radius_bin:g} px\n")
            handle.write(f"2D spot azimuth bin: {args.spot_azimuth_bin:g} deg\n")
            handle.write(f"2D spot value: {args.spot_value}\n")
        else:
            handle.write(f"2D spot CSV not used or not found: {args.spot_csv}\n")

    print(f"Wrote {len(patterns)}-frame peak similarity outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
