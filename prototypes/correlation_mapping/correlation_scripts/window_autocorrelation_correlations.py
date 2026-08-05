#!/usr/bin/env python3
"""Windowed autocorrelation similarity maps for integrated XRD patterns."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import compare_integrated_peaks as cip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split each integrated .xy pattern into sliding 2theta windows, "
            "compute autocorrelation fingerprints, and compare windows."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("Data/Cell_14_integrated"), Path("Data/Cell_29_integrated")],
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["same-window", "single-frame", "all-window"], required=True)
    parser.add_argument("--window-width", type=float, default=5.0)
    parser.add_argument("--window-step", type=float, default=1.0)
    parser.add_argument(
        "--window-start",
        type=float,
        default=None,
        help="Optional explicit first window start. Useful for matching overlap and non-overlap grids.",
    )
    parser.add_argument("--grid-step", type=float, default=0.02)
    parser.add_argument("--min-two-theta", type=float, default=2.0)
    parser.add_argument("--max-two-theta", type=float, default=None)
    parser.add_argument(
        "--shift-tolerance",
        type=float,
        default=1.0,
        help="For same-window mode, compare against windows shifted by up to this many degrees.",
    )
    parser.add_argument(
        "--write-ncc-shift",
        action="store_true",
        help="In same-window mode, also write direct signal NCC+shift matrices.",
    )
    parser.add_argument("--baseline-window", type=int, default=101)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument(
        "--min-window-std",
        type=float,
        default=1e-6,
        help="Windows flatter than this after preprocessing are treated as empty.",
    )
    return parser.parse_args()


def load_patterns(args: argparse.Namespace) -> list[cip.Pattern]:
    loader_args = SimpleNamespace(
        min_two_theta=args.min_two_theta,
        max_two_theta=args.max_two_theta,
    )
    files = cip.discover_xy_files(args.inputs)
    if not files:
        raise SystemExit("No .xy files found.")
    patterns = [cip.load_pattern(path, loader_args) for path in files]
    patterns = sorted(
        patterns,
        key=lambda pattern: (
            float("inf") if pattern.pressure_gpa is None else pattern.pressure_gpa,
            pattern.label,
        ),
    )
    return cip.ensure_unique_pattern_labels(patterns)


def common_grid(patterns: list[cip.Pattern], step: float) -> np.ndarray:
    lower = max(pattern.two_theta.min() for pattern in patterns)
    upper = min(pattern.two_theta.max() for pattern in patterns)
    return np.arange(lower, upper + step / 2.0, step)


def window_starts(grid: np.ndarray, width: float, step: float, start: float | None = None) -> np.ndarray:
    if start is None:
        first = math.ceil(float(grid.min()) / step) * step
    else:
        first = float(start)
        if first < float(grid.min()):
            first += math.ceil((float(grid.min()) - first) / step) * step
    max_start = float(grid.max()) - width
    last = first + math.floor((max_start - first) / step) * step
    if last < first:
        return np.array([], dtype=float)
    return np.arange(first, last + step / 2.0, step)


def odd_window(requested: int, size: int) -> int:
    if size < 5:
        return max(3, size | 1)
    window = min(requested, size if size % 2 == 1 else size - 1)
    window = max(5, window)
    if window % 2 == 0:
        window -= 1
    return window


def preprocess_window(y: np.ndarray, args: argparse.Namespace) -> np.ndarray | None:
    if len(y) < 8 or np.nanstd(y) < args.min_window_std:
        return None
    smooth_w = odd_window(args.smooth_window, len(y))
    baseline_w = odd_window(args.baseline_window, len(y))
    smoothed = savgol_filter(y, smooth_w, polyorder=2)
    if baseline_w > smooth_w + 2:
        baseline = savgol_filter(y, baseline_w, polyorder=2)
    else:
        baseline = np.full_like(y, np.nanmedian(y))
    signal = smoothed - baseline
    signal = signal - np.nanmean(signal)
    std = float(np.nanstd(signal))
    if std < args.min_window_std:
        return None
    return signal / std


def autocorrelation_fingerprint(signal: np.ndarray) -> np.ndarray | None:
    corr = np.correlate(signal, signal, mode="full")
    corr = corr[len(corr) // 2 :]
    if len(corr) <= 2 or corr[0] <= 0:
        return None
    corr = corr / corr[0]
    # Drop lag zero because it is always one and dominates the comparison.
    fingerprint = corr[1:]
    fingerprint = fingerprint - np.nanmean(fingerprint)
    std = float(np.nanstd(fingerprint))
    if std < 1e-12:
        return None
    return fingerprint / std


def pearson(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return np.nan
    n = min(len(a), len(b))
    if n < 3:
        return np.nan
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def shifted_ncc(
    a: np.ndarray | None,
    b: np.ndarray | None,
    max_lag: int,
) -> tuple[float, int]:
    if a is None or b is None:
        return np.nan, 0
    n = min(len(a), len(b))
    if n < 8:
        return np.nan, 0
    a = a[:n]
    b = b[:n]
    best_score = np.nan
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: n + lag]
        elif lag > 0:
            aa = a[: n - lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b
        if len(aa) < max(8, int(0.5 * n)):
            continue
        aa = aa - np.nanmean(aa)
        bb = bb - np.nanmean(bb)
        std_a = float(np.nanstd(aa))
        std_b = float(np.nanstd(bb))
        if std_a < 1e-12 or std_b < 1e-12:
            continue
        score = float(np.nanmean((aa / std_a) * (bb / std_b)))
        if np.isnan(best_score) or score > best_score:
            best_score = score
            best_lag = lag
    return best_score, best_lag


def build_window_signals_and_fingerprints(
    patterns: list[cip.Pattern], grid: np.ndarray, starts: np.ndarray, args: argparse.Namespace
) -> tuple[
    list[str],
    dict[tuple[int, int], np.ndarray | None],
    dict[tuple[int, int], np.ndarray | None],
]:
    labels = [pattern.label for pattern in patterns]
    signals: dict[tuple[int, int], np.ndarray | None] = {}
    fingerprints: dict[tuple[int, int], np.ndarray | None] = {}
    for frame_index, pattern in enumerate(patterns):
        y_grid = np.interp(grid, pattern.two_theta, pattern.intensity)
        y_grid = cip.normalize_for_pattern(y_grid)
        for window_index, start in enumerate(starts):
            stop = start + args.window_width
            keep = (grid >= start) & (grid < stop)
            signal = preprocess_window(y_grid[keep], args)
            signals[(frame_index, window_index)] = signal
            fingerprints[(frame_index, window_index)] = autocorrelation_fingerprint(signal) if signal is not None else None
    return labels, signals, fingerprints


def lower_triangle(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    out[np.triu_indices_from(out, k=0)] = np.nan
    return out


def write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[cip.format_value(value) for value in row]])


def plot_heatmap(
    path: Path,
    labels: list[str],
    matrix: np.ndarray,
    title: str,
    vmin=-1.0,
    vmax=1.0,
    colorbar_label: str = "ACF Pearson similarity",
) -> None:
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    size = min(18.0, max(6.0, 0.22 * len(labels) + 2.0))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    if len(labels) <= 80:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("frame-window index; full labels are in the CSV")
        ax.set_ylabel("frame-window index; full labels are in the CSV")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def same_window_mode(
    labels: list[str],
    starts: np.ndarray,
    signals: dict[tuple[int, int], np.ndarray | None],
    fingerprints: dict[tuple[int, int], np.ndarray | None],
    args: argparse.Namespace,
) -> None:
    heatmap_dir = args.out_dir / "heatmaps"
    matrix_dir = args.out_dir / "matrices"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    neighbor = int(round(args.shift_tolerance / args.window_step))
    summary_path = args.out_dir / "same_window_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window", "start_deg", "end_deg", "median_similarity", "best_shift_counts"])
        for window_index, start in enumerate(starts):
            matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
            shifts: list[int] = []
            for i in range(len(labels)):
                for j in range(i):
                    best_score = np.nan
                    best_shift = 0
                    for delta in range(-neighbor, neighbor + 1):
                        shifted = window_index + delta
                        if shifted < 0 or shifted >= len(starts):
                            continue
                        score = pearson(fingerprints[(i, window_index)], fingerprints[(j, shifted)])
                        if np.isnan(score):
                            continue
                        if np.isnan(best_score) or score > best_score:
                            best_score = score
                            best_shift = delta
                    matrix[i, j] = best_score
                    if not np.isnan(best_score):
                        shifts.append(best_shift)
            shown = lower_triangle(matrix)
            stem = f"window_{start:.1f}_{start + args.window_width:.1f}"
            labels_title = f"{start:.1f}-{start + args.window_width:.1f} deg"
            plot_heatmap(heatmap_dir / f"{stem}_same_window_acf.png", labels, shown, labels_title)
            write_matrix(matrix_dir / f"{stem}_same_window_acf.csv", labels, shown)
            counts = {shift: shifts.count(shift) for shift in sorted(set(shifts))}
            writer.writerow(
                [
                    stem,
                    f"{start:.3f}",
                    f"{start + args.window_width:.3f}",
                    cip.format_value(float(np.nanmedian(shown))),
                    ";".join(f"{k:+d}:{v}" for k, v in counts.items()),
                ]
            )
    if args.write_ncc_shift:
        same_window_ncc_shift_mode(labels, starts, signals, args)


def same_window_ncc_shift_mode(
    labels: list[str],
    starts: np.ndarray,
    signals: dict[tuple[int, int], np.ndarray | None],
    args: argparse.Namespace,
) -> None:
    heatmap_dir = args.out_dir / "ncc_shift_heatmaps"
    matrix_dir = args.out_dir / "ncc_shift_matrices"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    max_lag = max(0, int(round(args.shift_tolerance / args.grid_step)))
    summary_path = args.out_dir / "same_window_ncc_shift_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window", "start_deg", "end_deg", "median_ncc", "best_lag_deg_counts"])
        for window_index, start in enumerate(starts):
            matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
            lags: list[float] = []
            for i in range(len(labels)):
                for j in range(i):
                    score, lag = shifted_ncc(
                        signals[(i, window_index)],
                        signals[(j, window_index)],
                        max_lag,
                    )
                    matrix[i, j] = score
                    if not np.isnan(score):
                        lags.append(round(lag * args.grid_step, 6))
            shown = lower_triangle(matrix)
            stem = f"window_{start:.1f}_{start + args.window_width:.1f}"
            title = f"{start:.1f}-{start + args.window_width:.1f} deg NCC+shift"
            plot_heatmap(
                heatmap_dir / f"{stem}_same_window_ncc_shift.png",
                labels,
                shown,
                title,
                colorbar_label="NCC+shift similarity",
            )
            write_matrix(matrix_dir / f"{stem}_same_window_ncc_shift.csv", labels, shown)
            counts = {lag: lags.count(lag) for lag in sorted(set(lags))}
            writer.writerow(
                [
                    stem,
                    f"{start:.3f}",
                    f"{start + args.window_width:.3f}",
                    cip.format_value(float(np.nanmedian(shown))),
                    ";".join(f"{lag:+.3f}:{count}" for lag, count in counts.items()),
                ]
            )


def single_frame_mode(
    labels: list[str],
    starts: np.ndarray,
    fingerprints: dict[tuple[int, int], np.ndarray | None],
    args: argparse.Namespace,
) -> None:
    heatmap_dir = args.out_dir / "heatmaps"
    matrix_dir = args.out_dir / "matrices"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    window_labels = [f"{start:.1f}-{start + args.window_width:.1f}" for start in starts]
    for frame_index, label in enumerate(labels):
        matrix = np.full((len(starts), len(starts)), np.nan, dtype=float)
        for i in range(len(starts)):
            for j in range(i):
                matrix[i, j] = pearson(fingerprints[(frame_index, i)], fingerprints[(frame_index, j)])
        shown = lower_triangle(matrix)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        plot_heatmap(heatmap_dir / f"{stem}_single_frame_window_acf.png", window_labels, shown, label)
        write_matrix(matrix_dir / f"{stem}_single_frame_window_acf.csv", window_labels, shown)


def all_window_mode(
    labels: list[str],
    starts: np.ndarray,
    fingerprints: dict[tuple[int, int], np.ndarray | None],
    args: argparse.Namespace,
) -> None:
    combined_labels: list[str] = []
    keys: list[tuple[int, int]] = []
    for frame_index, label in enumerate(labels):
        for window_index, start in enumerate(starts):
            combined_labels.append(f"{label}_{start:.1f}-{start + args.window_width:.1f}")
            keys.append((frame_index, window_index))
    matrix = np.full((len(keys), len(keys)), np.nan, dtype=float)
    for i in range(len(keys)):
        for j in range(i):
            matrix[i, j] = pearson(fingerprints[keys[i]], fingerprints[keys[j]])
    shown = lower_triangle(matrix)
    # This all-window matrix can be large; keep a CSV and one overview heatmap.
    write_matrix(args.out_dir / "all_window_to_all_window_acf_matrix.csv", combined_labels, shown)
    plot_heatmap(
        args.out_dir / "all_window_to_all_window_acf_heatmap.png",
        combined_labels,
        shown,
        "All windows vs all windows ACF similarity",
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns(args)
    grid = common_grid(patterns, args.grid_step)
    starts = window_starts(grid, args.window_width, args.window_step, args.window_start)
    labels, signals, fingerprints = build_window_signals_and_fingerprints(patterns, grid, starts, args)
    if args.mode == "same-window":
        same_window_mode(labels, starts, signals, fingerprints, args)
    elif args.mode == "single-frame":
        single_frame_mode(labels, starts, fingerprints, args)
    else:
        all_window_mode(labels, starts, fingerprints, args)
    with (args.out_dir / "README.txt").open("w") as handle:
        handle.write(f"Mode: {args.mode}\n")
        handle.write(f"Frames: {len(labels)}\n")
        handle.write(f"Windows: {len(starts)}\n")
        handle.write(f"Window width: {args.window_width:g} deg\n")
        handle.write(f"Window step: {args.window_step:g} deg\n")
        if args.window_start is not None:
            handle.write(f"Window start: {args.window_start:g} deg\n")
        handle.write(f"Grid step: {args.grid_step:g} deg\n")
        if args.mode == "same-window":
            handle.write(f"Shift tolerance: +/-{args.shift_tolerance:g} deg\n")
            handle.write("Each same-window pair uses the best ACF similarity over neighboring shifted windows.\n")
            if args.write_ncc_shift:
                handle.write(
                    "NCC+shift matrices directly compare preprocessed window signals and take the best "
                    "normalized cross-correlation over +/-shift-tolerance lags.\n"
                )
        handle.write("Diagonal and upper triangle are blank in plotted matrices.\n")
    print(f"Wrote {args.mode} window autocorrelation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
