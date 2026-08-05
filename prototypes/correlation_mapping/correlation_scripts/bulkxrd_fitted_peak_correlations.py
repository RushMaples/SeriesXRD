#!/usr/bin/env python3
"""Build per-peak area and location maps directly from BulkXRD fitted peaks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

import bulkxrd_h5_to_xy as bxy
import compare_integrated_peaks as cip
import per_peak_correlation_maps as ppcm


SCAN_RE = re.compile(r"(?:^|[^A-Za-z])scan[_\- ]*0*(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_h5", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--peak-match-tolerance", type=float, default=0.08)
    parser.add_argument("--position-tolerance", type=float, default=0.06)
    parser.add_argument("--min-frame-count", type=int, default=1)
    parser.add_argument(
        "--group-by",
        choices=["auto", "none", "scan", "folder"],
        default="auto",
        help="Make independent maps per scan/folder; auto uses scan when multiple scan tags exist.",
    )
    parser.add_argument("--include-flagged", action="store_true")
    parser.add_argument("--wavelength-angstrom", type=float, default=None)
    return parser.parse_args()


def decode(value: object) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.") or "series"


def scan_label(name: str) -> str:
    match = SCAN_RE.search(name)
    return f"scan{int(match.group(1)):03d}" if match else "scan_unknown"


def folder_label(name: str) -> str:
    normalized = name.replace("\\", "/")
    parent = normalized.rsplit("/", 1)[0] if "/" in normalized else ""
    return safe_name(parent.rsplit("/", 1)[-1]) if parent else "folder_unknown"


def resolve_group_by(requested: str, names: list[str]) -> str:
    if requested != "auto":
        return requested
    scans = {scan_label(name) for name in names}
    scans.discard("scan_unknown")
    return "scan" if len(scans) > 1 else "none"


def frame_label(index: int, name: str, pressure: float) -> str:
    stem = safe_name(Path(name.split("::", 1)[0]).stem) if name else f"frame_{index:04d}"
    pressure_text = f"{pressure:g}GPa" if np.isfinite(pressure) else "P_unknown"
    return f"{index:04d}_{pressure_text}_{stem}"


def h5_vector(group: h5py.Group | None, key: str, n: int, default: object) -> np.ndarray:
    if group is None or key not in group:
        return np.full(n, default)
    values = np.asarray(group[key][:])
    if len(values) != n:
        raise ValueError(f"/frames/{key} has {len(values)} rows; expected {n}")
    return values


def convert_errors_to_two_theta(
    centers: np.ndarray,
    errors: np.ndarray,
    unit: str,
    wavelength: float | None,
) -> np.ndarray:
    out = np.full(errors.shape, np.nan, dtype=float)
    finite = np.isfinite(centers) & np.isfinite(errors) & (errors >= 0)
    if not np.any(finite):
        return out
    if unit in {"2th_deg", "2theta_deg", "2theta", "2th"}:
        out[finite] = errors[finite]
        return out
    if unit in {"2th_rad", "2theta_rad"}:
        out[finite] = np.degrees(errors[finite])
        return out
    if unit in {"q_a^-1", "q_a-1", "q_angstrom^-1", "q_å^-1", "q_nm^-1", "q_nm-1"}:
        if not wavelength:
            raise ValueError("q-axis uncertainty conversion needs a wavelength")
        q_scale = 0.1 if unit in {"q_nm^-1", "q_nm-1"} else 1.0
        z = centers * q_scale * wavelength / (4.0 * math.pi)
        valid = finite & (np.abs(z) < 1.0)
        derivative = np.full(centers.shape, np.nan, dtype=float)
        derivative[valid] = np.degrees(
            2.0
            * q_scale
            * wavelength
            / (4.0 * math.pi)
            / np.sqrt(1.0 - z[valid] ** 2)
        )
        out[valid] = np.abs(derivative[valid]) * errors[valid]
        return out
    raise ValueError(f"Cannot convert uncertainty for BulkXRD unit {unit!r}")


def write_fitted_peak_table(
    path: Path,
    peaks: list[cip.Peak],
    extras: dict[int, dict[str, float | int]],
) -> None:
    fields = [
        "frame", "frame_index", "peak_group", "group_two_theta", "center_native",
        "center_two_theta", "center_err_two_theta", "area_native", "amplitude",
        "fwhm_native", "chi2", "flag", "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for peak in sorted(peaks, key=lambda item: (item.pattern_label, item.two_theta)):
            extra = extras[id(peak)]
            writer.writerow({
                "frame": peak.pattern_label,
                "frame_index": int(extra["frame_index"]),
                "peak_group": peak.group_index or "",
                "group_two_theta": cip.format_value(float(peak.group_two_theta or np.nan)),
                "center_native": cip.format_value(float(extra["center_native"])),
                "center_two_theta": cip.format_value(peak.two_theta),
                "center_err_two_theta": cip.format_value(float(extra["center_err_two_theta"])),
                "area_native": cip.format_value(float(extra["area"])),
                "amplitude": cip.format_value(float(extra["amplitude"])),
                "fwhm_native": cip.format_value(float(extra["fwhm_native"])),
                "chi2": cip.format_value(float(extra["chi2"])),
                "flag": int(extra["flag"]),
                "source": peak.source_methods,
            })


def write_group_summary(
    path: Path,
    labels: list[str],
    centers: list[float],
    presence: np.ndarray,
    areas: np.ndarray,
    positions: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "peak_group", "group_two_theta", "frame_count", "frame_coverage_fraction",
            "frames_present", "max_area_native", "median_area_native",
            "position_min_deg", "position_max_deg",
        ])
        for col, center in enumerate(centers):
            present = presence[:, col]
            area_values = areas[present, col]
            position_values = positions[present & np.isfinite(positions[:, col]), col]
            writer.writerow([
                col + 1,
                f"{center:.6f}",
                int(np.count_nonzero(present)),
                cip.format_value(float(np.mean(present))),
                ";".join(label for label, keep in zip(labels, present) if keep),
                cip.format_value(float(np.nanmax(area_values))) if area_values.size else "",
                cip.format_value(float(np.nanmedian(area_values))) if area_values.size else "",
                cip.format_value(float(np.nanmin(position_values))) if position_values.size else "",
                cip.format_value(float(np.nanmax(position_values))) if position_values.size else "",
            ])


def map_series(
    output_dir: Path,
    series_name: str,
    frame_order: list[int],
    labels_by_frame: list[str],
    peak_rows: list[dict[str, float | int]],
    source: str,
    match_tolerance: float,
    position_tolerance: float,
    min_frame_count: int,
) -> dict[str, int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "per_peak_heatmaps"
    matrix_dir = output_dir / "per_peak_matrices"
    position_heatmap_dir = output_dir / "per_peak_position_heatmaps"
    position_matrix_dir = output_dir / "per_peak_position_matrices"
    for directory in (heatmap_dir, matrix_dir, position_heatmap_dir, position_matrix_dir):
        directory.mkdir(parents=True, exist_ok=True)

    labels = [labels_by_frame[index] for index in frame_order]
    frame_to_row = {frame: row for row, frame in enumerate(frame_order)}
    peaks: list[cip.Peak] = []
    extras: dict[int, dict[str, float | int]] = {}
    for record in peak_rows:
        frame = int(record["frame_index"])
        if frame not in frame_to_row:
            continue
        peak = cip.Peak(
            pattern_label=labels_by_frame[frame],
            path=Path(record["path"]),
            two_theta=float(record["center_two_theta"]),
            intensity=float(record["area"]),
            prominence=max(float(record["amplitude"]), 1e-12),
            width_deg=float(record["fwhm_two_theta"]),
            source_methods=source,
            source_count=1,
            confidence_tier="A",
            tier_score=1.0,
        )
        peaks.append(peak)
        extras[id(peak)] = record

    centers = cip.group_peaks(peaks, match_tolerance) if peaks else []
    shape = (len(frame_order), len(centers))
    presence = np.zeros(shape, dtype=bool)
    areas = np.full(shape, np.nan, dtype=float)
    positions = np.full(shape, np.nan, dtype=float)
    position_errors = np.full(shape, np.nan, dtype=float)
    best_quality = np.full(shape, -np.inf, dtype=float)
    for peak in peaks:
        if peak.group_index is None:
            continue
        extra = extras[id(peak)]
        row = frame_to_row[int(extra["frame_index"])]
        col = peak.group_index - 1
        chi2 = float(extra["chi2"])
        quality = float(extra["amplitude"]) / (1.0 + max(chi2, 0.0) if np.isfinite(chi2) else 1.0)
        if quality <= best_quality[row, col]:
            continue
        best_quality[row, col] = quality
        presence[row, col] = True
        areas[row, col] = float(extra["area"])
        positions[row, col] = peak.two_theta
        position_errors[row, col] = float(extra["center_err_two_theta"])

    cip.write_feature_table(output_dir / "peak_fit_area_features.csv", labels, centers, areas)
    cip.write_feature_table(output_dir / "peak_presence_features.csv", labels, centers, presence.astype(float))
    cip.write_feature_table(output_dir / "peak_position_features.csv", labels, centers, positions)
    cip.write_feature_table(output_dir / "peak_position_error_features.csv", labels, centers, position_errors)
    write_fitted_peak_table(output_dir / "fitted_peak_table.csv", peaks, extras)
    write_group_summary(output_dir / "peak_group_summary.csv", labels, centers, presence, areas, positions)

    area_pairs = output_dir / "all_per_peak_pair_similarities.csv"
    position_pairs = output_dir / "all_per_peak_position_pair_similarities.csv"
    for path in (area_pairs, position_pairs):
        if path.exists():
            path.unlink()

    area_count = 0
    position_count = 0
    area_index = output_dir / "per_peak_map_index.csv"
    with area_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "peak_group", "group_two_theta", "frame_count", "frames_present",
            "max_area_native", "frame_coverage_fraction", "heatmap", "matrix",
        ])
        for col, center in enumerate(centers):
            present = presence[:, col]
            frame_count = int(np.count_nonzero(present))
            if frame_count < min_frame_count:
                continue
            values = areas[:, col]
            matrix = ppcm.per_peak_similarity(values, present)
            stem = ppcm.safe_name(f"peak_group_{col + 1:03d}_{center:.3f}deg")
            heatmap_path = heatmap_dir / f"{stem}_correlation.png"
            matrix_path = matrix_dir / f"{stem}_correlation.csv"
            ppcm.plot_lower_triangle_heatmap(
                heatmap_path,
                labels,
                matrix,
                f"{series_name} peak {col + 1}: {center:.3f} deg",
                colorbar_label="fitted peak area similarity",
            )
            ppcm.write_matrix(matrix_path, labels, matrix)
            ppcm.write_long_pair_table(area_pairs, labels, col + 1, center, values, present, matrix)
            writer.writerow([
                col + 1,
                f"{center:.6f}",
                frame_count,
                ";".join(label for label, keep in zip(labels, present) if keep),
                cip.format_value(float(np.nanmax(values[present]))),
                cip.format_value(frame_count / max(len(labels), 1)),
                str(heatmap_path.relative_to(output_dir)),
                str(matrix_path.relative_to(output_dir)),
            ])
            area_count += 1

    position_index = output_dir / "per_peak_position_map_index.csv"
    with position_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "peak_group", "group_two_theta", "frame_count", "frames_present",
            "position_tolerance_deg", "position_range_deg", "median_center_error_deg",
            "heatmap", "matrix",
        ])
        for col, center in enumerate(centers):
            present = presence[:, col] & np.isfinite(positions[:, col])
            frame_count = int(np.count_nonzero(present))
            if frame_count < min_frame_count:
                continue
            values = positions[:, col]
            matrix = ppcm.per_peak_position_similarity(values, present, position_tolerance)
            stem = ppcm.safe_name(f"peak_group_{col + 1:03d}_{center:.3f}deg")
            heatmap_path = position_heatmap_dir / f"{stem}_position_correlation.png"
            matrix_path = position_matrix_dir / f"{stem}_position_correlation.csv"
            ppcm.plot_lower_triangle_heatmap(
                heatmap_path,
                labels,
                matrix,
                f"{series_name} peak position {col + 1}: {center:.3f} deg",
                colorbar_label="fitted peak position similarity",
            )
            ppcm.write_matrix(matrix_path, labels, matrix)
            ppcm.write_long_position_pair_table(
                position_pairs, labels, col + 1, center, values, present, matrix
            )
            finite_values = values[present]
            finite_errors = position_errors[present, col]
            writer.writerow([
                col + 1,
                f"{center:.6f}",
                frame_count,
                ";".join(label for label, keep in zip(labels, present) if keep),
                cip.format_value(position_tolerance),
                cip.format_value(float(np.nanmax(finite_values) - np.nanmin(finite_values))),
                cip.format_value(float(np.nanmedian(finite_errors))) if np.isfinite(finite_errors).any() else "",
                str(heatmap_path.relative_to(output_dir)),
                str(matrix_path.relative_to(output_dir)),
            ])
            position_count += 1

    (output_dir / "README.txt").write_text(
        "BulkXRD fitted-peak correlation maps\n"
        "====================================\n\n"
        f"Series: {series_name}\n"
        f"Fit source: {source}\n"
        f"Frames: {len(labels)}\n"
        f"Good fitted peaks: {len(peaks)}\n"
        f"Peak groups: {len(centers)}\n"
        f"Peak match tolerance: {match_tolerance:g} deg 2theta\n"
        f"Position tolerance: {position_tolerance:g} deg 2theta\n\n"
        "Area maps use BulkXRD /peaks/area directly. If both frames contain the fitted "
        "reflection, score = 1 - abs(area_a-area_b)/max(area_a,area_b); if only one "
        "contains it, score = 0.\n"
        "Location maps use /peaks/center converted to 2theta. The corresponding "
        "/peaks/center_err values are exported for fit-quality review.\n",
        encoding="utf-8",
    )
    return {
        "series": series_name,
        "frames": len(labels),
        "fitted_peaks": len(peaks),
        "peak_groups": len(centers),
        "area_maps": area_count,
        "location_maps": position_count,
    }


def main() -> None:
    args = parse_args()
    input_path = args.analysis_h5.expanduser().resolve()
    output_root = args.out_dir.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"BulkXRD analysis HDF5 not found: {input_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        if "peaks" not in h5:
            raise SystemExit(f"{input_path} has no /peaks group; run BulkXRD peak fitting first")
        peak_group = h5["peaks"]
        frames_group = h5.get("frames")
        counts = np.asarray(peak_group["counts"][:], dtype=int)
        n_frames = len(counts)
        raw_names = h5_vector(frames_group, "filename", n_frames, "")
        names = [decode(value) for value in raw_names]
        pressure = h5_vector(frames_group, "pressure", n_frames, np.nan).astype(float)
        excluded = h5_vector(frames_group, "excluded", n_frames, False).astype(bool)
        frame_indices = np.asarray(peak_group["frame"][:], dtype=int)
        centers_native = np.asarray(peak_group["center"][:], dtype=float)
        areas = np.asarray(peak_group["area"][:], dtype=float)
        amplitudes = np.asarray(peak_group["amplitude"][:], dtype=float)
        fwhm_native = np.asarray(peak_group["fwhm"][:], dtype=float)
        chi2 = np.asarray(peak_group["chi2"][:], dtype=float)
        flags = np.asarray(peak_group["flag"][:], dtype=int)
        center_errors_native = (
            np.asarray(peak_group["center_err"][:], dtype=float)
            if "center_err" in peak_group
            else np.full(centers_native.shape, np.nan)
        )
        source = decode(peak_group.attrs.get("source", "unknown"))
        unit = bxy.normalized_unit(h5.attrs.get("unit", ""))
        wavelength, wavelength_source = bxy.wavelength_from_h5(h5, input_path)

    if args.wavelength_angstrom is not None:
        wavelength = float(args.wavelength_angstrom)
        wavelength_source = "command-line"
    centers_two_theta, _ = bxy.convert_axis(centers_native, unit, "two-theta", wavelength)
    center_errors_two_theta = convert_errors_to_two_theta(
        centers_native, center_errors_native, unit, wavelength
    )
    fwhm_two_theta = convert_errors_to_two_theta(
        centers_native, np.abs(fwhm_native) / 2.0, unit, wavelength
    ) * 2.0

    good = (
        (frame_indices >= 0)
        & (frame_indices < n_frames)
        & np.isfinite(centers_native)
        & np.isfinite(centers_two_theta)
        & np.isfinite(areas)
        & (areas > 0)
        & np.isfinite(amplitudes)
        & (amplitudes > 0)
    )
    if not args.include_flagged:
        good &= flags == 0
    if np.any(good):
        good &= ~excluded[np.clip(frame_indices, 0, max(n_frames - 1, 0))]

    peak_rows: list[dict[str, float | int]] = []
    for index in np.flatnonzero(good):
        peak_rows.append({
            "path": str(input_path),
            "frame_index": int(frame_indices[index]),
            "center_native": float(centers_native[index]),
            "center_two_theta": float(centers_two_theta[index]),
            "center_err_two_theta": float(center_errors_two_theta[index]),
            "area": float(areas[index]),
            "amplitude": float(amplitudes[index]),
            "fwhm_native": float(fwhm_native[index]),
            "fwhm_two_theta": float(fwhm_two_theta[index]),
            "chi2": float(chi2[index]),
            "flag": int(flags[index]),
        })

    labels = [frame_label(index, names[index], pressure[index]) for index in range(n_frames)]
    group_by = resolve_group_by(args.group_by, names)
    series: OrderedDict[str, list[int]] = OrderedDict()
    for index, name in enumerate(names):
        if excluded[index]:
            continue
        if group_by == "scan":
            key = scan_label(name)
        elif group_by == "folder":
            key = folder_label(name)
        else:
            key = "all"
        series.setdefault(key, []).append(index)
    for frames in series.values():
        frames.sort(key=lambda index: (
            float("inf") if not np.isfinite(pressure[index]) else pressure[index], index
        ))

    summaries: list[dict[str, int | str]] = []
    multiple = len(series) > 1
    for key, frame_order in series.items():
        series_dir = output_root / safe_name(key) if multiple else output_root
        summaries.append(map_series(
            series_dir,
            key,
            frame_order,
            labels,
            peak_rows,
            source,
            args.peak_match_tolerance,
            args.position_tolerance,
            args.min_frame_count,
        ))

    with (output_root / "series_index.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["series", "frames", "fitted_peaks", "peak_groups", "area_maps", "location_maps"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    manifest = {
        "analysis_h5": str(input_path),
        "fit_source": source,
        "input_unit": unit,
        "wavelength_angstrom": wavelength,
        "wavelength_source": wavelength_source,
        "group_by": group_by,
        "include_flagged": bool(args.include_flagged),
        "peak_match_tolerance_deg": args.peak_match_tolerance,
        "position_tolerance_deg": args.position_tolerance,
        "frames_total": n_frames,
        "good_fitted_peaks": len(peak_rows),
        "series": summaries,
    }
    (output_root / "mapping_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote fitted-peak maps for {len(series)} series: "
        f"{sum(int(item['area_maps']) for item in summaries)} area, "
        f"{sum(int(item['location_maps']) for item in summaries)} location"
    )


if __name__ == "__main__":
    main()
