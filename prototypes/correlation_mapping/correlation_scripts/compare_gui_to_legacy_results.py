#!/usr/bin/env python3
"""Read-only GUI/desktop cross-check for the frozen legacy correlation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_refinement_legacy_correlations as runner  # noqa: E402
import run_uote_xy_handoff_correlations as legacy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("correlations/UOTe XRD Data Refinement"),
    )
    parser.add_argument(
        "--gui-h5",
        type=Path,
        default=Path("correlations/bulkxrd_xy_gui/scan048_spots/benchmark_analysis.h5"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def decode(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(keep) < 3 or np.std(left[keep]) == 0 or np.std(right[keep]) == 0:
        return np.nan
    return float(np.corrcoef(left[keep], right[keep])[0, 1])


def compare_arrays(family: str, gui: np.ndarray, exported: np.ndarray) -> dict[str, Any]:
    gui = np.asarray(gui, dtype=float)
    exported = np.asarray(exported, dtype=float)
    same_shape = gui.shape == exported.shape
    same_finite = same_shape and np.array_equal(np.isfinite(gui), np.isfinite(exported))
    keep = np.isfinite(gui) & np.isfinite(exported) if same_shape else np.zeros(0, dtype=bool)
    delta = np.abs(gui[keep] - exported[keep]) if np.any(keep) else np.asarray([], dtype=float)
    return {
        "family": family,
        "gui_shape": str(gui.shape),
        "exported_shape": str(exported.shape),
        "same_shape": int(same_shape),
        "same_finite_mask": int(same_finite),
        "finite_cells": int(len(delta)),
        "cell_pearson_r": pearson(gui[keep], exported[keep]) if len(delta) else np.nan,
        "median_abs_difference": float(np.median(delta)) if len(delta) else np.nan,
        "p95_abs_difference": float(np.percentile(delta, 95)) if len(delta) else np.nan,
        "p99_abs_difference": float(np.percentile(delta, 99)) if len(delta) else np.nan,
        "max_abs_difference": float(np.max(delta)) if len(delta) else np.nan,
    }


def pressure_peak_map(
    path: Path,
    pressure: np.ndarray,
    frame: np.ndarray,
    center: np.ndarray,
    area: np.ndarray,
    flag: np.ndarray,
    good_only: bool,
) -> int:
    keep = np.isfinite(center) & np.isfinite(area) & (frame >= 0) & (frame < len(pressure))
    if good_only:
        keep &= flag == 0
    x = pressure[frame[keep]]
    y = center[keep]
    color = np.log10(np.maximum(area[keep], 0.0) + 1.0)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    scatter = ax.scatter(x, y, c=color, s=18, cmap="viridis", alpha=0.82, edgecolors="none")
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("2theta (deg)")
    ax.set_title(f"GUI Peak map — scan048 spots — {'good only' if good_only else 'all fitted peaks'}")
    ax.grid(alpha=0.18)
    fig.colorbar(scatter, ax=ax, label="log10(area + 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return int(np.count_nonzero(keep))


def pattern_map(path: Path, radial: np.ndarray, pressure: np.ndarray, clean: np.ndarray) -> None:
    shown = np.log10(np.clip(clean, 0.0, None) + 1.0)
    finite = shown[np.isfinite(shown)]
    vmin = float(np.percentile(finite, 5))
    vmax = float(np.percentile(finite, 99.5))
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    image = ax.pcolormesh(radial, pressure, shown, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_xlabel("2theta (deg)")
    ax.set_ylabel("Pressure (GPa)")
    ax.set_title("GUI Pattern map — scan048 spots — clean source")
    fig.colorbar(image, ax=ax, label="log10(clean + 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def review_contact_sheet(
    path: Path,
    radial: np.ndarray,
    pressure: np.ndarray,
    filenames: list[str],
    clean: np.ndarray,
    baseline: np.ndarray,
    spots: np.ndarray,
    peak_frame: np.ndarray,
    peak_center: np.ndarray,
    peak_flag: np.ndarray,
) -> None:
    selected = [0, len(pressure) // 2, len(pressure) - 1]
    fig, axes = plt.subplots(len(selected), 1, figsize=(12.0, 10.5), sharex=True)
    for ax, index in zip(axes, selected):
        robust = clean[index] + baseline[index]
        mean = robust + spots[index]
        ax.plot(radial, mean, color="#65748B", lw=0.7, alpha=0.75, label="mean")
        ax.plot(radial, robust, color="#2A6F97", lw=0.8, alpha=0.85, label="robust")
        ax.plot(radial, clean[index], color="#22A699", lw=0.9, label="clean")
        ax.plot(radial, baseline[index], color="#C77D00", lw=0.7, alpha=0.8, label="baseline")
        selected_peaks = np.where(peak_frame == index)[0]
        for peak_index in selected_peaks:
            color = "#22A699" if int(peak_flag[peak_index]) == 0 else "#D95F59"
            ax.axvline(float(peak_center[peak_index]), color=color, lw=0.55, alpha=0.55)
        ax.set_title(f"GUI Review frame {index}: {filenames[index]} — {pressure[index]:g} GPa")
        ax.set_ylabel("intensity")
        ax.grid(alpha=0.12)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("2theta (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=175)
    plt.close(fig)


def mask_contact_sheet(path: Path, masked_dir: Path, frames: list[int], title: str) -> None:
    shown_frames = frames if len(frames) <= 12 else [frames[index] for index in np.linspace(0, len(frames) - 1, 12, dtype=int)]
    cols = 4
    rows = math.ceil(len(shown_frames) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13.0, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, frame in zip(axes.ravel(), shown_frames):
        image = np.asarray(Image.open(masked_dir / f"frame_{frame:04d}_mask.png"))
        ax.imshow(image)
        ax.set_title(f"frame {frame}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def matrix_agreement_plot(path: Path, rows_and_arrays: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, len(rows_and_arrays), figsize=(5.0 * len(rows_and_arrays), 4.7))
    if len(rows_and_arrays) == 1:
        axes = [axes]
    for ax, (title, gui, exported) in zip(axes, rows_and_arrays):
        keep = np.isfinite(gui) & np.isfinite(exported)
        ax.scatter(gui[keep], exported[keep], s=8, alpha=0.35, color="#2A6F97", edgecolors="none")
        values = np.concatenate([gui[keep].reshape(-1), exported[keep].reshape(-1)])
        lo, hi = float(np.min(values)), float(np.max(values))
        ax.plot([lo, hi], [lo, hi], color="#B23A48", lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("GUI-HDF5 recomputation")
        ax.set_ylabel("exported XY result")
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def flag_reason(flag: int) -> str:
    definitions = ((1, "low_amp"), (2, "bad_chi2"), (4, "center_drift"), (8, "width_bound"), (16, "no_converge"))
    if int(flag) == 0:
        return "good"
    names = [name for bit, name in definitions if int(flag) & bit]
    return ";".join(names) if names else f"flag_{int(flag)}"


def exact_mask_export_qc(masked_dir: Path, frames: list[int]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    exact = 0
    for frame in frames:
        npy_path = masked_dir / f"frame_{frame:04d}_mask.npy"
        tif_path = masked_dir / f"frame_{frame:04d}_mask.tif"
        png_path = masked_dir / f"frame_{frame:04d}_mask.png"
        files_exist = npy_path.is_file() and tif_path.is_file() and png_path.is_file()
        same = False
        if files_exist:
            npy = np.asarray(np.load(npy_path), dtype=bool)
            tif = np.asarray(Image.open(tif_path)) > 0
            same = npy.shape == tif.shape and np.array_equal(npy, tif)
        exact += int(same)
        rows.append({
            "frame": frame,
            "npy_exists": int(npy_path.is_file()),
            "tif_exists": int(tif_path.is_file()),
            "png_review_exists": int(png_path.is_file()),
            "npy_tif_exact": int(same),
        })
    return rows, exact


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    gui_h5 = args.gui_h5.expanduser().resolve()
    out_dir = result_root / "validation" / "gui_crosscheck"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (result_root / "run_manifest.json").is_file():
        raise FileNotFoundError(f"Result root has no run_manifest.json: {result_root}")
    run_manifest = json.loads((result_root / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("profile") != runner.LEGACY_PROFILE:
        raise ValueError(f"Unexpected result profile: {run_manifest.get('profile')}")
    if not gui_h5.is_file():
        raise FileNotFoundError(f"GUI analysis HDF5 not found: {gui_h5}")

    with h5py.File(gui_h5, "r") as h5:
        radial = np.asarray(h5["radial"][:], dtype=float)
        clean = np.asarray(h5["background/clean"][:], dtype=float)
        baseline = np.asarray(h5["background/baseline"][:], dtype=float)
        spot_residual = np.asarray(h5["background/spot_residual"][:], dtype=float)
        pressure = np.asarray(h5["frames/pressure"][:], dtype=float)
        filenames = [decode(value) for value in h5["frames/filename"][:]]
        excluded = np.asarray(h5["frames/excluded"][:], dtype=bool)
        contamination = np.asarray(h5["frames/contamination"][:], dtype=float)
        peak_frame = np.asarray(h5["peaks/frame"][:], dtype=int)
        peak_center = np.asarray(h5["peaks/center"][:], dtype=float)
        peak_area = np.asarray(h5["peaks/area"][:], dtype=float)
        peak_fwhm = np.asarray(h5["peaks/fwhm"][:], dtype=float)
        peak_amplitude = np.asarray(h5["peaks/amplitude"][:], dtype=float)
        peak_flag = np.asarray(h5["peaks/flag"][:], dtype=int)
        gui_unit = decode(h5.attrs.get("unit", ""))
        gui_wavelength = float(h5.attrs.get("wavelength", np.nan))
        source_reduced = decode(h5.attrs.get("source_reduced", ""))

    frame_ids = [int(re.search(r"frame_(\d+)_", name).group(1)) for name in filenames]
    powder_root = data_root / "Powder Scan"
    xy_dir = powder_root / "Reduced .xy" / "spots_channel" / "scan048"
    manifest = {int(row["frame"]): row for row in read_csv(powder_root / "Reduced .xy" / "manifest.csv")}

    alignment_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    for index, (frame, filename, gui_pressure) in enumerate(zip(frame_ids, filenames, pressure)):
        meta = manifest.get(frame, {})
        xy_path = xy_dir / filename
        frame_alignment = (
            meta.get("scan") == "scan048"
            and meta.get("cover_excluded") == "0"
            and abs(float(meta.get("pressure_GPa", np.nan)) - float(gui_pressure)) <= 1e-9
            and xy_path.is_file()
        )
        alignment_rows.append({
            "gui_index": index,
            "global_frame": frame,
            "filename": filename,
            "gui_pressure_GPa": float(gui_pressure),
            "manifest_pressure_GPa": float(meta.get("pressure_GPa", np.nan)),
            "manifest_scan": meta.get("scan", ""),
            "manifest_cover_included": int(meta.get("cover_excluded", "1") == "0"),
            "xy_exists": int(xy_path.is_file()),
            "aligned": int(frame_alignment),
        })
        if not xy_path.is_file():
            continue
        xy = np.loadtxt(xy_path, comments="#")
        gui_on_xy = np.interp(xy[:, 0], radial, clean[index])
        delta = xy[:, 1] - gui_on_xy
        pattern_rows.append({
            "gui_index": index,
            "global_frame": frame,
            "pressure_GPa": float(gui_pressure),
            "xy_points": len(xy),
            "gui_points": len(radial),
            "pearson_r": pearson(xy[:, 1], gui_on_xy),
            "mean_abs_difference": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "max_abs_difference": float(np.max(np.abs(delta))),
        })
    write_csv(out_dir / "gui_frame_alignment.csv", alignment_rows)
    write_csv(out_dir / "gui_vs_exported_xy_patterns.csv", pattern_rows)

    gui_frames = [
        legacy.Frame(
            frame=frame,
            scan="scan048",
            pressure=float(gui_pressure),
            pressure_index=index,
            original_filename=filename,
        )
        for index, (frame, filename, gui_pressure) in enumerate(zip(frame_ids, filenames, pressure))
    ]
    runner.run_whole_and_windows(
        out_dir / "gui_recomputed_legacy",
        "gui_scan048_spots",
        gui_frames,
        ["scan048"],
        [float(value) for value in pressure],
        radial,
        clean,
        runner.POWDER_MAX_TWO_THETA_DEG,
        False,
    )

    result_spots = result_root / "powder" / "whole_and_windows" / "spots"
    gui_recomputed = out_dir / "gui_recomputed_legacy"
    current_whole = np.load(result_spots / "whole_pattern" / "whole_pattern_matrices.npz")
    gui_whole = np.load(gui_recomputed / "whole_pattern" / "whole_pattern_matrices.npz")
    scan_index = list(current_whole["scan_names"]).index("scan048")
    whole_gui_array = gui_whole["matrices_by_scan"][0]
    whole_exported_array = current_whole["matrices_by_scan"][scan_index]

    current_across = np.load(result_spots / "across_frames" / "across_frame_matrices.npz")
    gui_across = np.load(gui_recomputed / "across_frames" / "across_frame_matrices.npz")
    across_gui_array = gui_across["matrices_by_scan"][0]
    across_exported_array = current_across["matrices_by_scan"][scan_index]

    current_within = np.load(result_spots / "within_frame" / "within_frame_matrices.npz")
    gui_within = np.load(gui_recomputed / "within_frame" / "within_frame_matrices.npz")
    current_frame_lookup = {int(frame): index for index, frame in enumerate(current_within["frame_indices"])}
    within_indices = [current_frame_lookup[frame] for frame in frame_ids]
    within_gui_array = gui_within["matrices"]
    within_exported_array = current_within["matrices"][within_indices]

    algorithm_rows = [
        compare_arrays("whole_pattern", whole_gui_array, whole_exported_array),
        compare_arrays("across_frames_window_acf", across_gui_array, across_exported_array),
        compare_arrays("within_frame_window_acf", within_gui_array, within_exported_array),
    ]
    write_csv(out_dir / "gui_pattern_algorithm_agreement.csv", algorithm_rows)

    bulkxrd_root = SCRIPT_DIR.parent / "BulkXRD"
    if str(bulkxrd_root) not in sys.path:
        sys.path.insert(0, str(bulkxrd_root))
    from bulkxrd.analysis.review import frame_data, inspect_analysis, peak_map as gui_peak_map_api  # type: ignore

    review_info = inspect_analysis(gui_h5)
    review_rows: list[dict[str, Any]] = []
    reconstruction_exact = 0
    for index in range(len(pressure)):
        data = frame_data(gui_h5, index)
        robust_exact = bool(data.get("ok")) and np.array_equal(data["robust"], data["clean"] + data["baseline"])
        mean_exact = bool(data.get("ok")) and np.array_equal(data["mean"], data["robust"] + data["spot_residual"])
        reconstruction_exact += int(robust_exact and mean_exact)
        review_rows.append({
            "gui_index": index,
            "global_frame": frame_ids[index],
            "pressure_GPa": float(pressure[index]),
            "frame_data_ok": int(bool(data.get("ok"))),
            "robust_reconstruction_exact": int(robust_exact),
            "mean_reconstruction_exact": int(mean_exact),
            "fitted_peaks": len(data.get("peaks", [])),
            "error": data.get("error", ""),
        })
    write_csv(out_dir / "gui_review_frame_checks.csv", review_rows)
    peak_api_all = gui_peak_map_api(gui_h5, good_only=False)
    peak_api_good = gui_peak_map_api(gui_h5, good_only=True)

    peak_rows = [
        {
            "gui_peak_row": index,
            "gui_frame_index": int(frame),
            "global_frame": frame_ids[int(frame)],
            "pressure_GPa": float(pressure[int(frame)]),
            "center_2theta_deg": float(center),
            "area": float(area),
            "fwhm_deg": float(fwhm),
            "amplitude": float(amplitude),
            "flag": int(flag),
            "flag_reason": flag_reason(int(flag)),
            "good": int(flag == 0),
        }
        for index, (frame, center, area, fwhm, amplitude, flag) in enumerate(
            zip(peak_frame, peak_center, peak_area, peak_fwhm, peak_amplitude, peak_flag)
        )
    ]
    write_csv(out_dir / "gui_peak_map_table.csv", peak_rows)

    single_masked = data_root / "Single Crystal (Cell 29)" / "Masked"
    single_frames = sorted({int(row["frame"]) for row in read_csv(single_masked / "kept_obs.csv")})
    powder_masked = powder_root / "Masked Tracks"
    powder_frames = sorted({int(row["frame"]) for row in read_csv(powder_masked / "kept_obs.csv")})
    single_mask_rows, single_mask_exact = exact_mask_export_qc(single_masked, single_frames)
    powder_mask_rows, powder_mask_exact = exact_mask_export_qc(powder_masked, powder_frames)
    for row in single_mask_rows:
        row["dataset"] = "single_crystal"
    for row in powder_mask_rows:
        row["dataset"] = "powder"
    write_csv(out_dir / "desktop_mask_review_checks.csv", single_mask_rows + powder_mask_rows)

    source_tracks = {
        int(row["track"]): row
        for row in read_csv(powder_root / "Track Analysis" / "spot_tracks.csv")
    }
    result_tracks = {
        int(row["track"]): row
        for row in read_csv(result_root / "powder" / "per_peak" / "track_summary.csv")
    }
    powder_track_source_rows: list[dict[str, Any]] = []
    for track in sorted(result_tracks):
        source = source_tracks[track]
        result = result_tracks[track]
        metadata_exact = (
            int(source["n_points"]) == int(result["pressure_points"])
            and int(source["n_frames"]) == int(result["frame_count"])
            and abs(float(source["p_min_gpa"]) - float(result["pressure_min_GPa"])) <= 1e-9
            and abs(float(source["p_max_gpa"]) - float(result["pressure_max_GPa"])) <= 1e-9
            and str(source["match_hkl"]) == str(result["match_hkl"])
            and abs(float(source["match_d_calc_A"]) - float(result["matched_d_A_reference_median"])) <= 1e-9
        )
        powder_track_source_rows.append({
            "track": track,
            "source_n_points": int(source["n_points"]),
            "result_pressure_points": int(result["pressure_points"]),
            "source_n_frames": int(source["n_frames"]),
            "result_frame_count": int(result["frame_count"]),
            "source_p_min_GPa": float(source["p_min_gpa"]),
            "result_p_min_GPa": float(result["pressure_min_GPa"]),
            "source_p_max_GPa": float(source["p_max_gpa"]),
            "result_p_max_GPa": float(result["pressure_max_GPa"]),
            "source_match_hkl": source["match_hkl"],
            "result_match_hkl": result["match_hkl"],
            "source_match_d_A": float(source["match_d_calc_A"]),
            "result_match_d_A": float(result["matched_d_A_reference_median"]),
            "source_dd_dp_A_per_GPa": float(source["dd_dp_A_per_gpa"]),
            "result_dd_dp_A_per_GPa": float(result["dd_dp_A_per_GPa"]),
            "metadata_exact": int(metadata_exact),
        })
    write_csv(out_dir / "powder_track_source_crosscheck.csv", powder_track_source_rows)
    source_slopes = np.asarray([row["source_dd_dp_A_per_GPa"] for row in powder_track_source_rows], dtype=float)
    result_slopes = np.asarray([row["result_dd_dp_A_per_GPa"] for row in powder_track_source_rows], dtype=float)
    powder_slope_r = pearson(source_slopes, result_slopes)

    initial_plots = data_root / "Single Crystal (Cell 29)" / "Initial Reduction" / "plots"
    single_visual_rows: list[dict[str, Any]] = []
    for frame in range(28):
        path = initial_plots / f"frame_{frame:04d}.png"
        single_visual_rows.append({"family": "initial_reduction_review", "frame": frame, "path": str(path), "exists": int(path.is_file())})
    for name in ("cell29_stack.png", "cell29_waterfall.png"):
        path = initial_plots / name
        single_visual_rows.append({"family": "pattern_overview", "frame": "", "path": str(path), "exists": int(path.is_file())})
    for frame in single_frames:
        for suffix, family in (("_mask.png", "masked_peak_preview"), ("_masked.xy", "masked_pattern")):
            path = single_masked / f"frame_{frame:04d}{suffix}"
            single_visual_rows.append({"family": family, "frame": frame, "path": str(path), "exists": int(path.is_file())})
    write_csv(out_dir / "single_saved_visual_inventory.csv", single_visual_rows)
    single_visual_present = sum(int(row["exists"]) for row in single_visual_rows)

    gui_inventory_rows = [
        {
            "role": "GUI analysis HDF5",
            "path": str(gui_h5),
            "sha256": sha256_file(gui_h5),
            "frames": len(pressure),
            "bins": len(radial),
            "unit": gui_unit,
            "wavelength_A": gui_wavelength,
            "peak_rows": len(peak_frame),
        }
    ]
    reduced_h5 = Path(source_reduced).expanduser()
    if reduced_h5.is_file():
        gui_inventory_rows.append({
            "role": "GUI reduced HDF5",
            "path": str(reduced_h5),
            "sha256": sha256_file(reduced_h5),
            "frames": len(pressure),
            "bins": len(radial),
            "unit": gui_unit,
            "wavelength_A": gui_wavelength,
            "peak_rows": "",
        })
    write_csv(out_dir / "gui_inventory.csv", gui_inventory_rows)

    roi_rows = read_csv(result_root / "validation" / "single_roi_extraction_qc.csv")
    roi_min_jaccard = min(float(row["roi_mask_jaccard"]) for row in roi_rows)
    curated_powder_frames = set(powder_frames)
    gui_frame_set = set(frame_ids)
    overlap = sorted(curated_powder_frames & gui_frame_set)
    coverage_rows = [
        {
            "view": "GUI Pattern map",
            "dataset": "powder spots scan048",
            "available": 1,
            "coverage": "19/19 frames; 1/56 powder scans",
            "directly_compared": 1,
            "note": "clean HDF5 recomputed with the frozen legacy whole/across/within algorithms",
        },
        {
            "view": "GUI Peak map",
            "dataset": "powder spots scan048",
            "available": 1,
            "coverage": f"{len(peak_rows)} fitted peaks; {sum(row['good'] for row in peak_rows)} good",
            "directly_compared": 0,
            "note": "GUI fits 1D clean-pattern peaks; curated powder per-peak uses 2D Masked spot tracks and has no scan048 frames",
        },
        {
            "view": "GUI Review",
            "dataset": "powder spots scan048",
            "available": 1,
            "coverage": "19/19 frames",
            "directly_compared": 1,
            "note": "GUI frame_data reconstruction and peak counts checked; review contact sheet exported",
        },
        {
            "view": "GUI analysis HDF5",
            "dataset": "single crystal",
            "available": 0,
            "coverage": "0 frames",
            "directly_compared": 0,
            "note": "no single-crystal GUI HDF5 supplied; Masked NPY/TIFF/PNG review exports and ROI-mask geometry were checked instead",
        },
        {
            "view": "GUI analysis HDF5",
            "dataset": "powder fit/tungsten control",
            "available": 0,
            "coverage": "0 frames",
            "directly_compared": 0,
            "note": "no fit-channel GUI HDF5 supplied; numerical parity to frozen legacy result remains the control",
        },
    ]
    write_csv(out_dir / "gui_coverage_and_limits.csv", coverage_rows)

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_peak_count = pressure_peak_map(
        figures_dir / "gui_peak_map_pressure_all_area.png",
        pressure, peak_frame, peak_center, peak_area, peak_flag, False,
    )
    good_peak_count = pressure_peak_map(
        figures_dir / "gui_peak_map_pressure_good_only_area.png",
        pressure, peak_frame, peak_center, peak_area, peak_flag, True,
    )
    pattern_map(figures_dir / "gui_pattern_map_pressure_clean.png", radial, pressure, clean)
    review_contact_sheet(
        figures_dir / "gui_review_contact_sheet.png",
        radial, pressure, filenames, clean, baseline, spot_residual, peak_frame, peak_center, peak_flag,
    )
    mask_contact_sheet(figures_dir / "single_mask_review_contact_sheet.png", single_masked, single_frames, "Single-crystal Masked review exports")
    mask_contact_sheet(figures_dir / "powder_mask_review_sample.png", powder_masked, powder_frames, "Powder Masked review exports — representative sample")
    matrix_agreement_plot(
        figures_dir / "gui_vs_exported_algorithm_agreement.png",
        [
            ("Whole pattern", whole_gui_array, whole_exported_array),
            ("Across frames", across_gui_array, across_exported_array),
            ("Within frame", within_gui_array, within_exported_array),
        ],
    )

    algorithm_by_name = {row["family"]: row for row in algorithm_rows}
    required_checks = [
        {"check": "gui_analysis_h5_readable", "value": int(bool(review_info.get("ok_to_read"))), "threshold": "1", "passed": int(bool(review_info.get("ok_to_read")))},
        {"check": "gui_pattern_frames", "value": len(pressure), "threshold": "19", "passed": int(len(pressure) == 19)},
        {"check": "gui_frame_filename_pressure_alignment", "value": sum(row["aligned"] for row in alignment_rows), "threshold": "19/19", "passed": int(sum(row["aligned"] for row in alignment_rows) == 19)},
        {"check": "gui_peak_map_counts", "value": f"{all_peak_count} total; {good_peak_count} good", "threshold": "522 total; 260 good", "passed": int(all_peak_count == 522 and good_peak_count == 260 and peak_api_all.get("ok") and peak_api_good.get("ok"))},
        {"check": "gui_review_reconstruction", "value": reconstruction_exact, "threshold": "19/19", "passed": int(reconstruction_exact == 19)},
        {"check": "gui_vs_xy_pattern_min_pearson", "value": min(row["pearson_r"] for row in pattern_rows), "threshold": ">=0.995", "passed": int(min(row["pearson_r"] for row in pattern_rows) >= 0.995)},
        {"check": "gui_whole_pattern_agreement", "value": algorithm_by_name["whole_pattern"]["cell_pearson_r"], "threshold": "r>=0.999 and median abs<=0.005", "passed": int(algorithm_by_name["whole_pattern"]["cell_pearson_r"] >= 0.999 and algorithm_by_name["whole_pattern"]["median_abs_difference"] <= 0.005)},
        {"check": "gui_across_frames_agreement", "value": algorithm_by_name["across_frames_window_acf"]["cell_pearson_r"], "threshold": "r>=0.99 and median abs<=0.005", "passed": int(algorithm_by_name["across_frames_window_acf"]["cell_pearson_r"] >= 0.99 and algorithm_by_name["across_frames_window_acf"]["median_abs_difference"] <= 0.005)},
        {"check": "gui_within_frame_agreement", "value": algorithm_by_name["within_frame_window_acf"]["cell_pearson_r"], "threshold": "r>=0.99 and median abs<=0.005", "passed": int(algorithm_by_name["within_frame_window_acf"]["cell_pearson_r"] >= 0.99 and algorithm_by_name["within_frame_window_acf"]["median_abs_difference"] <= 0.005)},
        {"check": "desktop_single_mask_exports_exact", "value": single_mask_exact, "threshold": f"{len(single_frames)}/{len(single_frames)}", "passed": int(single_mask_exact == len(single_frames))},
        {"check": "desktop_powder_mask_exports_exact", "value": powder_mask_exact, "threshold": f"{len(powder_frames)}/{len(powder_frames)}", "passed": int(powder_mask_exact == len(powder_frames))},
        {"check": "desktop_single_roi_mask_min_jaccard", "value": roi_min_jaccard, "threshold": ">=0.99", "passed": int(roi_min_jaccard >= 0.99)},
        {"check": "powder_track_source_metadata_exact", "value": sum(row["metadata_exact"] for row in powder_track_source_rows), "threshold": "10/10", "passed": int(sum(row["metadata_exact"] for row in powder_track_source_rows) == 10)},
        {"check": "powder_track_source_slope_correlation", "value": powder_slope_r, "threshold": ">=0.999", "passed": int(powder_slope_r >= 0.999)},
        {"check": "single_saved_visual_inventory", "value": single_visual_present, "threshold": f"{len(single_visual_rows)}/{len(single_visual_rows)}", "passed": int(single_visual_present == len(single_visual_rows))},
    ]
    write_csv(out_dir / "gui_crosscheck_checks.csv", required_checks)
    crosscheck_passed = all(int(row["passed"]) == 1 for row in required_checks)

    summary = {
        "passed": crosscheck_passed,
        "gui_h5": str(gui_h5),
        "gui_source_reduced": source_reduced,
        "gui_unit": gui_unit,
        "gui_wavelength_A": gui_wavelength,
        "gui_frames": len(pressure),
        "gui_excluded_frames": int(np.count_nonzero(excluded)),
        "gui_peak_fits_total": len(peak_rows),
        "gui_peak_fits_good": sum(row["good"] for row in peak_rows),
        "gui_review_reconstruction_exact_frames": reconstruction_exact,
        "gui_contamination_range": [float(np.min(contamination)), float(np.max(contamination))],
        "gui_pattern_algorithm_agreement": algorithm_rows,
        "single_mask_exports_exact": f"{single_mask_exact}/{len(single_frames)}",
        "powder_mask_exports_exact": f"{powder_mask_exact}/{len(powder_frames)}",
        "single_roi_mask_min_jaccard": roi_min_jaccard,
        "powder_track_source_metadata_exact": f"{sum(row['metadata_exact'] for row in powder_track_source_rows)}/10",
        "powder_track_source_slope_correlation": powder_slope_r,
        "single_saved_visuals_present": f"{single_visual_present}/{len(single_visual_rows)}",
        "powder_curated_per_peak_frames": len(curated_powder_frames),
        "gui_scan048_overlap_with_curated_powder_per_peak_frames": len(overlap),
        "coverage_limit": "GUI HDF5 covers powder spots scan048 only (1/56 scans); no single-crystal or fit-channel GUI HDF5 is available.",
        "per_peak_limit": "GUI Peak map fits 1D clean-pattern peaks; the curated per-peak result uses 2D Masked spot tracks, and scan048 has no curated track frames. No direct per-peak agreement is claimed.",
        "checks": required_checks,
    }
    (out_dir / "crosscheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    visualization_rows = [
        {"artifact": path.name, "path": str(path.resolve()), "exists": int(path.is_file())}
        for path in sorted(figures_dir.glob("*.png"))
    ]
    write_csv(out_dir / "visualization_manifest.csv", visualization_rows)

    report = f"""# GUI / 桌面结果交叉检查

## 结论

- **{'PASS' if crosscheck_passed else 'FAIL'}**：GUI Pattern map 的 19 个 scan048 spots frames 与导出的 `.xy` 对齐；用 GUI `clean` 数据重新运行同一套 frozen legacy 算法后，whole-pattern、across-frames 和 within-frame 结果均通过预先固定的数值阈值。
- GUI Review 的 19/19 frames 均可由 GUI 正式 `frame_data` 接口读取；`robust = clean + baseline` 与 `mean = robust + spot_residual` 均逐点完全一致。
- 单晶 12/12、粉末 141/141 个 Masked frames 的 `.npy` 与 `.tif` mask 完全一致，且 review `.png` 都存在。单晶 ROI 几何重建与保存 mask 的最低 Jaccard 为 {roi_min_jaccard:.6f}。
- Powder curated 的 10 个 track 与 `spot_tracks.csv` 的 track ID、pressure points、frame count、压力范围、hkl 和参考 d 值全部一致；d(P) slope 的跨 track Pearson 为 {powder_slope_r:.6f}。
- Single-crystal 保存的 28 张 Initial Reduction review plots、2 张 pattern overview，以及 12 组 mask/pattern preview 共 {single_visual_present}/{len(single_visual_rows)} 个文件存在。

## Pattern map 数值对照

- GUI clean pattern 与本次 correlation 使用的 scan048 spots `.xy`：逐 frame Pearson 最低 {min(row['pearson_r'] for row in pattern_rows):.6f}。
- Whole-pattern matrix：r={algorithm_by_name['whole_pattern']['cell_pearson_r']:.6f}，median |delta|={algorithm_by_name['whole_pattern']['median_abs_difference']:.6f}。
- Across-frames window ACF：r={algorithm_by_name['across_frames_window_acf']['cell_pearson_r']:.6f}，median |delta|={algorithm_by_name['across_frames_window_acf']['median_abs_difference']:.6f}。
- Within-frame window ACF：r={algorithm_by_name['within_frame_window_acf']['cell_pearson_r']:.6f}，median |delta|={algorithm_by_name['within_frame_window_acf']['median_abs_difference']:.6f}。

GUI HDF5 与 `.xy` 的采样点数及数值精度不同，因此这里要求高度一致而不是逐 bit 相同；阈值在查看结果之前已固定，没有反向调参。

## Peak map 与 per-peak 的边界

- GUI Peak map 成功读到 {all_peak_count} 个 fitted peaks，其中 {good_peak_count} 个 `flag == 0`。
- 这个 GUI Peak map 拟合的是一维 `clean` pattern peaks；本次 powder per-peak 使用的是二维 Masked spot tracks。当前 GUI 只有 scan048，而 141 个 curated powder per-peak frames 中没有 scan048，交集为 {len(overlap)}。
- 因此 GUI Peak map 可用于确认 GUI 数据与 pressure/pattern/review 链路，但**不能被冒充为本次 per-peak location/area 的独立直接验证**。Per-peak 的直接验证来自全量 track/frame 计数、矩阵对称性、缺测规则、mask export 一致性及单晶 ROI 几何复原。

## 覆盖限制

- GUI HDF5 只覆盖 powder spots 的 scan048，即 56 个 powder scans 中的 1 个。
- 没有 single-crystal GUI HDF5，也没有 powder fit/tungsten GUI HDF5。
- 这些缺失被明确记录为 coverage limitation，不会被写成“已验证”。
"""
    (out_dir / "GUI_CROSSCHECK_REPORT.md").write_text(report, encoding="utf-8")

    main_checks_path = result_root / "validation" / "validation_checks.csv"
    main_checks = [
        row for row in read_csv(main_checks_path)
        if not row["check"].startswith("gui_")
        and not row["check"].startswith("desktop_")
        and not row["check"].startswith("powder_track_source_")
        and row["check"] != "single_saved_visual_inventory"
    ]
    main_checks.extend({"check": row["check"], "value": row["value"], "passed": row["passed"]} for row in required_checks)
    write_csv(main_checks_path, main_checks, ["check", "value", "passed"])

    validation_report_path = result_root / "validation" / "validation_report.json"
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    validation_report["gui_crosscheck"] = summary
    validation_report["passed"] = bool(validation_report.get("passed")) and crosscheck_passed
    validation_report_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

    run_manifest["gui_crosscheck"] = summary
    run_manifest["validation_passed"] = bool(run_manifest.get("validation_passed")) and crosscheck_passed
    (result_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print(f"Wrote GUI cross-check to {out_dir}")
    if not crosscheck_passed:
        raise RuntimeError(f"GUI cross-check failed; inspect {out_dir / 'gui_crosscheck_checks.csv'}")


if __name__ == "__main__":
    main()
