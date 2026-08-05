#!/usr/bin/env python3
"""Generate relationship-first XRD waterfall figures from the verified v3 run.

The original line waterfalls are useful for seeing spectral evolution, but they
do not answer the two relationship questions directly:

* which frames are most correlated; and
* which peak/window evidence is available for a correlation-map cell.

This generator therefore combines matrices, Top-K rankings, peak fingerprints,
selected map-cell evidence, and window-interval views.  The actual compared
whole-pattern feature is retained as a drill-down below the relationship view.

Important semantics inherited from the v3 analysis:

* NaN/unknown is never converted to zero or absence.
* Whole-pattern Pearson and direct NCC are signed quantities in [-1, 1].
* Per-peak similarities are associated evidence, not additive contributions to
  whole-pattern Pearson/NCC.
* The v3 window analysis is fixed-window direct NCC plus correlations between
  pressure-change trajectories.  It is not ACF and not a within-frame ACF map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    ROOT
    / "correlations/results/uote_uncertainty_aware_correlations_v3_20260719"
)
DEFAULT_OUTPUT = (
    ROOT / "correlations/results/uote_relationship_waterfalls_v1_20260724"
)
DEFAULT_BASELINE = (
    ROOT
    / "correlations/results/"
    "uote_refinement_legacy_global_per_peak_strict_lower_triangle_20260716"
)
DEFAULT_HANDOFF = ROOT / "correlations/uote_xy_handoff 2"
VERSION = "1.0.0"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_uncertainty_aware_correlations as v3  # noqa: E402


SIGNED_CMAP = plt.colormaps["RdBu_r"].copy()
SIGNED_CMAP.set_bad("#d9d9d9")
SIMILARITY_CMAP = plt.colormaps["viridis"].copy()
SIMILARITY_CMAP.set_bad("#d9d9d9")
CHANGE_CMAP = plt.colormaps["PuOr_r"].copy()
CHANGE_CMAP.set_bad("#d9d9d9")
SHIFT_CMAP = plt.colormaps["BrBG"].copy()
SHIFT_CMAP.set_bad("#d9d9d9")
SUPPORT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "support", ["#eeeeee", "#9ecae1", "#08519c"]
)
MISSING_COLOR = "#d9d9d9"
TEXT = "#252525"
MUTED = "#666666"
GRID = "#ececec"
FOCUS = "#c58a17"
FRAME_A = "#24557a"
FRAME_B = "#d36b32"
PROTOCOL_MARK = "†"


@dataclass(frozen=True)
class SeriesFeature:
    key: str
    series: v3.SeriesData
    standardized: np.ndarray


@dataclass(frozen=True)
class PeakArchive:
    dataset: str
    track_ids: np.ndarray
    axis_labels: np.ndarray
    axis_pressures: np.ndarray
    scan_names: np.ndarray
    location: np.ndarray
    area: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--max-powder-scans",
        type=int,
        default=None,
        help="Testing aid: cap powder scan panels; CSV rankings remain complete.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate a completed output directory in place.",
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
    path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def compact(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def frame_label(frame: int, pressure: float) -> str:
    return f"F{int(frame)} · {compact(float(pressure), 2)} GPa"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        fig.savefig(
            temporary,
            dpi=dpi,
            facecolor="white",
            bbox_inches="tight",
            format=path.suffix.lstrip("."),
        )
        temporary.replace(path)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()


def symmetrize_lower(matrix: np.ndarray, diagonal: float | None = None) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected square matrix, got {values.shape}")
    full = values.copy()
    fill = ~np.isfinite(full) & np.isfinite(values.T)
    full[fill] = values.T[fill]
    if diagonal is not None:
        np.fill_diagonal(full, diagonal)
    return full


def symmetrize_track_stack(values: np.ndarray) -> np.ndarray:
    stack = np.asarray(values, dtype=float).copy()
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        raise ValueError(f"Expected track×axis×axis stack, got {stack.shape}")
    transposed = np.swapaxes(stack, 1, 2)
    fill = ~np.isfinite(stack) & np.isfinite(transposed)
    stack[fill] = transposed[fill]
    return stack


def robust_symmetric_limit(values: Iterable[float], minimum: float = 1e-6) -> float:
    array = np.asarray(list(values), dtype=float)
    array = np.abs(array[np.isfinite(array)])
    if not len(array):
        return 1.0
    return max(minimum, float(np.quantile(array, 0.98)))


def add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    label: str,
    *,
    fraction: float = 0.045,
) -> None:
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=fraction,
        pad=0.025,
        label=label,
    )


def load_series_features(
    baseline_root: Path, handoff_root: Path
) -> dict[str, SeriesFeature]:
    selected = baseline_root / "inputs/single_whole_selected.csv"
    single = v3.load_single_series(selected, 0.02)
    powder = v3.load_powder_series(handoff_root, 0.02)
    raw = {
        "single_0deg": single["0deg"],
        "single_10deg": single["10deg"],
        "powder_spots": powder["spots"],
        "powder_fit": powder["fit"],
    }
    result: dict[str, SeriesFeature] = {}
    for key, series in raw.items():
        standardized, _ = v3.legacy.standardized_signals(
            series.normalized, smooth_window=9, baseline_window=101
        )
        result[key] = SeriesFeature(
            key=key,
            series=series,
            standardized=np.asarray(standardized, dtype=float),
        )
    return result


def whole_pair_path(source: Path, key: str) -> Path:
    return source / f"whole_pattern/{key}/whole_pattern_pair_scores.csv"


def rank_frame_neighbors(
    pairs: pd.DataFrame, top_k: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "dataset",
        "channel",
        "scan",
        "frame_a",
        "frame_b",
        "pressure_a_GPa",
        "pressure_b_GPa",
        "correlation",
        "acquisition_signature_a",
        "acquisition_signature_b",
        "acquisition_protocol_changed",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Whole-pattern pair table missing columns: {missing}")
    directed_parts: list[pd.DataFrame] = []
    for reverse in (False, True):
        current = pd.DataFrame(
            {
                "dataset": pairs["dataset"].astype(str),
                "channel": pairs["channel"].astype(str),
                "scan": pairs["scan"].astype(str),
                "anchor_frame": (
                    pairs["frame_b"] if reverse else pairs["frame_a"]
                ).astype(int),
                "anchor_pressure_GPa": (
                    pairs["pressure_b_GPa"]
                    if reverse
                    else pairs["pressure_a_GPa"]
                ).astype(float),
                "anchor_signature": (
                    pairs["acquisition_signature_b"]
                    if reverse
                    else pairs["acquisition_signature_a"]
                ).astype(str),
                "neighbor_frame": (
                    pairs["frame_a"] if reverse else pairs["frame_b"]
                ).astype(int),
                "neighbor_pressure_GPa": (
                    pairs["pressure_a_GPa"]
                    if reverse
                    else pairs["pressure_b_GPa"]
                ).astype(float),
                "neighbor_signature": (
                    pairs["acquisition_signature_a"]
                    if reverse
                    else pairs["acquisition_signature_b"]
                ).astype(str),
                "pressure_gap_GPa": pairs["pressure_gap_GPa"].astype(float),
                "correlation": pairs["correlation"].astype(float),
                "acquisition_protocol_changed": pairs[
                    "acquisition_protocol_changed"
                ].astype(int),
            }
        )
        directed_parts.append(current)
    directed = pd.concat(directed_parts, ignore_index=True)
    anchor_keys = [
        "dataset",
        "channel",
        "scan",
        "anchor_frame",
        "anchor_pressure_GPa",
        "anchor_signature",
    ]
    stats = (
        directed.groupby(anchor_keys, as_index=False)
        .agg(
            anchor_pair_count=("correlation", "size"),
            anchor_mean_correlation=("correlation", "mean"),
            anchor_median_correlation=("correlation", "median"),
            anchor_min_correlation=("correlation", "min"),
            anchor_max_correlation=("correlation", "max"),
        )
    )
    medoids = (
        stats.sort_values(
            [
                "dataset",
                "channel",
                "scan",
                "anchor_mean_correlation",
                "anchor_median_correlation",
                "anchor_frame",
            ],
            ascending=[True, True, True, False, False, True],
            kind="stable",
        )
        .groupby(["dataset", "channel", "scan"], as_index=False)
        .first()
        .rename(
            columns={
                "anchor_frame": "medoid_frame",
                "anchor_pressure_GPa": "medoid_pressure_GPa",
                "anchor_signature": "medoid_signature",
            }
        )
    )
    ranked = directed.sort_values(
        anchor_keys
        + [
            "correlation",
            "pressure_gap_GPa",
            "neighbor_pressure_GPa",
            "neighbor_frame",
        ],
        ascending=[True] * len(anchor_keys) + [False, True, True, True],
        kind="stable",
    ).copy()
    ranked["neighbor_rank"] = (
        ranked.groupby(anchor_keys, sort=False).cumcount() + 1
    )
    same = ranked["acquisition_protocol_changed"].eq(0)
    same_rank = (
        ranked.loc[same]
        .groupby(anchor_keys, sort=False)
        .cumcount()
        .add(1)
    )
    ranked["same_detected_protocol_rank"] = pd.Series(
        np.nan, index=ranked.index, dtype=float
    )
    ranked.loc[same, "same_detected_protocol_rank"] = same_rank.to_numpy(float)
    ranked = ranked.merge(stats, on=anchor_keys, how="left", validate="many_to_one")
    ranked = ranked.merge(
        medoids[
            [
                "dataset",
                "channel",
                "scan",
                "medoid_frame",
                "medoid_pressure_GPa",
            ]
        ],
        on=["dataset", "channel", "scan"],
        how="left",
        validate="many_to_one",
    )
    ranked["is_scan_medoid"] = (
        ranked["anchor_frame"].eq(ranked["medoid_frame"])
    ).astype(int)
    top = ranked[ranked["neighbor_rank"] <= top_k].copy()
    top_sets = (
        top.groupby(["dataset", "channel", "scan", "anchor_frame"])[
            "neighbor_frame"
        ]
        .agg(lambda values: set(int(item) for item in values))
        .to_dict()
    )
    top["reciprocal_topk"] = [
        int(
            int(row.anchor_frame)
            in top_sets.get(
                (
                    str(row.dataset),
                    str(row.channel),
                    str(row.scan),
                    int(row.neighbor_frame),
                ),
                set(),
            )
        )
        for row in top.itertuples()
    ]
    return top.reset_index(drop=True), medoids.reset_index(drop=True), ranked


def scan_nodes(group: pd.DataFrame) -> pd.DataFrame:
    left = group[
        [
            "frame_a",
            "pressure_a_GPa",
            "acquisition_signature_a",
        ]
    ].rename(
        columns={
            "frame_a": "frame",
            "pressure_a_GPa": "pressure_GPa",
            "acquisition_signature_a": "signature",
        }
    )
    right = group[
        [
            "frame_b",
            "pressure_b_GPa",
            "acquisition_signature_b",
        ]
    ].rename(
        columns={
            "frame_b": "frame",
            "pressure_b_GPa": "pressure_GPa",
            "acquisition_signature_b": "signature",
        }
    )
    nodes = (
        pd.concat([left, right], ignore_index=True)
        .drop_duplicates()
        .sort_values(["pressure_GPa", "frame"], kind="stable")
        .reset_index(drop=True)
    )
    conflicts = nodes.groupby("frame").size()
    if int(conflicts.max()) != 1:
        raise ValueError("A frame maps to multiple pressure/signature values")
    return nodes


def scan_full_matrix(group: pd.DataFrame, nodes: pd.DataFrame) -> np.ndarray:
    lookup = {int(frame): index for index, frame in enumerate(nodes["frame"])}
    matrix = np.full((len(nodes), len(nodes)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    for row in group.itertuples():
        a = lookup[int(row.frame_a)]
        b = lookup[int(row.frame_b)]
        matrix[a, b] = float(row.correlation)
        matrix[b, a] = float(row.correlation)
    return matrix


def series_row_lookup(feature: SeriesFeature) -> dict[tuple[str, int], int]:
    return {
        (str(frame.scan), int(frame.frame)): index
        for index, frame in enumerate(feature.series.frames)
    }


def plot_matrix_topk_feature_waterfall(
    output: Path,
    key: str,
    scan: str,
    group: pd.DataFrame,
    top_rows: pd.DataFrame,
    medoid_row: pd.Series,
    feature: SeriesFeature,
    top_k: int,
    dpi: int,
) -> None:
    nodes = scan_nodes(group)
    matrix = scan_full_matrix(group, nodes)
    order_lookup = {int(frame): index for index, frame in enumerate(nodes["frame"])}
    labels = [
        f"F{int(row.frame)}\n{compact(float(row.pressure_GPa), 2)}"
        for row in nodes.itertuples()
    ]
    n = len(nodes)
    top_grid = np.full((n, top_k), np.nan, dtype=float)
    top_text = [["" for _ in range(top_k)] for _ in range(n)]
    for row in top_rows.itertuples():
        y = order_lookup[int(row.anchor_frame)]
        x = int(row.neighbor_rank) - 1
        top_grid[y, x] = float(row.correlation)
        mark = PROTOCOL_MARK if int(row.acquisition_protocol_changed) else ""
        top_text[y][x] = (
            f"F{int(row.neighbor_frame)}{mark}\n"
            f"{compact(float(row.neighbor_pressure_GPa), 2)} GPa\n"
            f"{float(row.correlation):.3f}"
        )

    medoid_frame = int(medoid_row["medoid_frame"])
    medoid_pressure = float(medoid_row["medoid_pressure_GPa"])
    directed = pd.concat(
        [
            group.rename(
                columns={
                    "frame_a": "anchor",
                    "frame_b": "neighbor",
                    "pressure_a_GPa": "anchor_pressure",
                    "pressure_b_GPa": "neighbor_pressure",
                }
            ),
            group.rename(
                columns={
                    "frame_b": "anchor",
                    "frame_a": "neighbor",
                    "pressure_b_GPa": "anchor_pressure",
                    "pressure_a_GPa": "neighbor_pressure",
                }
            ),
        ],
        ignore_index=True,
    )
    medoid_scores = {
        int(row.neighbor): float(row.correlation)
        for row in directed[directed["anchor"].eq(medoid_frame)].itertuples()
    }
    medoid_scores[medoid_frame] = 1.0

    row_lookup = series_row_lookup(feature)
    spectral_rows: list[np.ndarray] = []
    for row in nodes.itertuples():
        spectral_rows.append(
            feature.standardized[
                row_lookup[(str(scan), int(row.frame))]
            ]
        )
    spectra = np.asarray(spectral_rows, dtype=float)
    finite = np.abs(spectra[np.isfinite(spectra)])
    display_limit = float(np.quantile(finite, 0.995)) if len(finite) else 1.0
    gain = 0.72 / max(display_limit, 1e-9)

    fig = plt.figure(figsize=(16.8, 11.0))
    grid = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.18],
        width_ratios=[1.18, 0.82],
        hspace=0.30,
        wspace=0.26,
    )
    matrix_ax = fig.add_subplot(grid[0, 0])
    top_ax = fig.add_subplot(grid[0, 1])
    waterfall_ax = fig.add_subplot(grid[1, :])
    signed_norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    image = matrix_ax.imshow(
        matrix,
        cmap=SIGNED_CMAP,
        norm=signed_norm,
        aspect="equal",
        interpolation="nearest",
    )
    matrix_ax.set_xticks(range(n), labels, rotation=90, fontsize=6.5)
    matrix_ax.set_yticks(range(n), labels, fontsize=6.5)
    matrix_ax.set_xlabel("Frame / pressure")
    matrix_ax.set_ylabel("Frame / pressure")
    matrix_ax.set_title("A · Complete within-scan Pearson map")
    medoid_index = order_lookup[medoid_frame]
    matrix_ax.add_patch(
        plt.Rectangle(
            (-0.5, medoid_index - 0.5),
            n,
            1,
            fill=False,
            edgecolor=FOCUS,
            linewidth=2.0,
        )
    )
    matrix_ax.add_patch(
        plt.Rectangle(
            (medoid_index - 0.5, -0.5),
            1,
            n,
            fill=False,
            edgecolor=FOCUS,
            linewidth=2.0,
        )
    )
    add_colorbar(fig, matrix_ax, SIGNED_CMAP, signed_norm, "Raw Pearson r")

    top_ax.imshow(
        top_grid,
        cmap=SIGNED_CMAP,
        norm=signed_norm,
        aspect="auto",
        interpolation="nearest",
    )
    top_ax.set_xticks(range(top_k), [f"Top {rank}" for rank in range(1, top_k + 1)])
    top_ax.set_yticks(range(n), labels, fontsize=6.5)
    top_ax.set_title("B · Most correlated partners for every frame")
    for y in range(n):
        for x in range(top_k):
            if not np.isfinite(top_grid[y, x]):
                continue
            color = "white" if abs(top_grid[y, x]) > 0.60 else TEXT
            top_ax.text(
                x,
                y,
                top_text[y][x],
                ha="center",
                va="center",
                fontsize=5.6,
                color=color,
                linespacing=0.92,
            )
    top_ax.set_xlabel(
        f"{PROTOCOL_MARK} filename-derived acquisition signature changed"
    )

    x = feature.series.grid
    for y, row in enumerate(nodes.itertuples()):
        score = medoid_scores.get(int(row.frame), np.nan)
        color = (
            FOCUS
            if int(row.frame) == medoid_frame
            else SIGNED_CMAP(signed_norm(score))
        )
        baseline = float(y)
        waterfall_ax.axhline(
            baseline, color="#efefef", linewidth=0.55, zorder=0
        )
        waterfall_ax.plot(
            x,
            baseline + spectra[y] * gain,
            color=color,
            linewidth=1.15 if int(row.frame) != medoid_frame else 1.8,
            zorder=2,
        )
        right = (
            f"F{int(row.frame)} · {compact(float(row.pressure_GPa), 2)} GPa"
            f" · r={score:.3f}"
        )
        if int(row.frame) == medoid_frame:
            right += " · MEDOID"
        waterfall_ax.text(
            1.004,
            baseline,
            right,
            transform=waterfall_ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.3,
            color=color if int(row.frame) != medoid_frame else "#8a5a00",
            clip_on=False,
        )
    waterfall_ax.set_ylim(-0.7, n - 0.05)
    waterfall_ax.set_yticks(range(n), labels, fontsize=6.5)
    waterfall_ax.set_xlim(float(x[0]), float(x[-1]))
    waterfall_ax.set_xlabel(r"$2\theta$ (degrees)")
    waterfall_ax.set_ylabel("Frame / pressure order; vertical offsets are display-only")
    waterfall_ax.set_title(
        "C · Actual compared whole-pattern feature, colored by similarity to the medoid"
    )
    waterfall_ax.grid(axis="x", color=GRID, linewidth=0.6)
    waterfall_ax.spines[["top", "right"]].set_visible(False)

    dataset_text = key.replace("_", " ")
    fig.suptitle(
        f"{dataset_text} · {scan}: frame relationships first, spectrum feature second",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.01,
        0.006,
        (
            f"Medoid: F{medoid_frame}, {medoid_pressure:g} GPa. "
            "Whole-pattern feature = normalized spectrum → SG smoothing (9 bins) "
            "→ SG baseline subtraction (101 bins) → per-frame centering/scaling. "
            "Raw Pearson is QC-only. Diagonal 1.0 is display-only self-correlation; "
            "vertical offsets do not encode intensity. "
            f"{PROTOCOL_MARK} means a filename-derived signature change was detected."
        ),
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
    )
    save_figure(fig, output, dpi)


def plot_best_match_overview(
    output: Path,
    key: str,
    top_rows: pd.DataFrame,
    pressure_axis: Sequence[float],
    scan_axis: Sequence[str],
    dpi: int,
) -> None:
    best = top_rows[top_rows["neighbor_rank"].eq(1)].copy()
    pressure_axis = np.asarray(pressure_axis, dtype=float)
    scan_axis = [str(item) for item in scan_axis]
    score = np.full((len(scan_axis), len(pressure_axis)), np.nan, dtype=float)
    gap = np.full_like(score, np.nan)
    reciprocal = np.full_like(score, np.nan)
    scan_lookup = {scan: index for index, scan in enumerate(scan_axis)}
    for row in best.itertuples():
        y = scan_lookup[str(row.scan)]
        candidates = np.flatnonzero(
            np.isclose(pressure_axis, float(row.anchor_pressure_GPa), atol=1e-8)
        )
        if not len(candidates):
            continue
        x = int(candidates[0])
        score[y, x] = float(row.correlation)
        gap[y, x] = float(row.pressure_gap_GPa)
        reciprocal[y, x] = float(row.reciprocal_topk)
    gap_max = float(np.nanquantile(gap, 0.98)) if np.any(np.isfinite(gap)) else 1.0

    height = max(4.2, min(14.0, 2.8 + 0.18 * len(scan_axis)))
    fig, axes = plt.subplots(1, 2, figsize=(17.0, height), sharey=True)
    signed_norm = mcolors.Normalize(-1.0, 1.0)
    gap_norm = mcolors.Normalize(0.0, max(gap_max, 1e-6))
    axes[0].imshow(
        score,
        aspect="auto",
        cmap=SIGNED_CMAP,
        norm=signed_norm,
        interpolation="nearest",
    )
    axes[1].imshow(
        gap,
        aspect="auto",
        cmap="magma_r",
        norm=gap_norm,
        interpolation="nearest",
    )
    for ax in axes:
        ax.set_xticks(
            range(len(pressure_axis)),
            [compact(value, 2) for value in pressure_axis],
            rotation=90,
            fontsize=6.5,
        )
        ax.set_xlabel("Query-frame pressure (GPa)")
        ax.set_yticks(
            range(len(scan_axis)),
            scan_axis,
            fontsize=6.0 if len(scan_axis) > 20 else 7.5,
        )
    axes[0].set_ylabel("Scan")
    axes[0].set_title("Top-1 raw Pearson r")
    axes[1].set_title("Pressure gap to Top-1 partner (GPa)")
    add_colorbar(fig, axes[0], SIGNED_CMAP, signed_norm, "Top-1 r")
    add_colorbar(fig, axes[1], plt.colormaps["magma_r"], gap_norm, "|ΔP| (GPa)")
    fig.suptitle(
        f"{key.replace('_', ' ')}: best match for every frame within its own scan",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.008,
        (
            "Rows are independent scans; no cross-scan ranking is performed. "
            "Missing cells are gray. Full partner frame IDs, reciprocity, and "
            "filename-derived protocol flags are in frame_topk_relationships.csv."
        ),
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(fig, output, dpi)


def generate_whole_pattern_outputs(
    source: Path,
    output: Path,
    features: dict[str, SeriesFeature],
    top_k: int,
    dpi: int,
    max_powder_scans: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    all_top: list[pd.DataFrame] = []
    all_medoids: list[pd.DataFrame] = []
    all_pairs: dict[str, pd.DataFrame] = {}
    generated: list[Path] = []
    for key in ("single_0deg", "single_10deg", "powder_spots", "powder_fit"):
        pairs = pd.read_csv(whole_pair_path(source, key))
        top, medoids, _ = rank_frame_neighbors(pairs, top_k)
        top.insert(0, "map_key", key)
        medoids.insert(0, "map_key", key)
        all_top.append(top)
        all_medoids.append(medoids)
        all_pairs[key] = pairs

    top_frame = pd.concat(all_top, ignore_index=True)
    medoid_frame = pd.concat(all_medoids, ignore_index=True)
    write_csv(output / "frame_topk_relationships.csv", top_frame)
    write_csv(output / "scan_medoids.csv", medoid_frame)

    for key, pairs in all_pairs.items():
        scans = sorted(pairs["scan"].astype(str).unique())
        render_scans = scans
        if key.startswith("powder") and max_powder_scans is not None:
            render_scans = scans[: max(0, max_powder_scans)]
        for scan in render_scans:
            group = pairs[pairs["scan"].astype(str).eq(scan)].copy()
            selected_top = top_frame[
                top_frame["map_key"].eq(key)
                & top_frame["scan"].astype(str).eq(scan)
            ]
            selected_medoid = medoid_frame[
                medoid_frame["map_key"].eq(key)
                & medoid_frame["scan"].astype(str).eq(scan)
            ].iloc[0]
            path = (
                output
                / "by_scan"
                / key
                / f"{safe_name(scan)}_matrix_topk_feature_waterfall.png"
            )
            plot_matrix_topk_feature_waterfall(
                path,
                key,
                scan,
                group,
                selected_top,
                selected_medoid,
                features[key],
                top_k,
                dpi,
            )
            generated.append(path)

        feature = features[key]
        overview_path = output / "overview" / f"{key}_best_match_overview.png"
        plot_best_match_overview(
            overview_path,
            key,
            top_frame[top_frame["map_key"].eq(key)],
            feature.series.pressures,
            feature.series.scans,
            dpi,
        )
        generated.append(overview_path)
    return top_frame, medoid_frame, generated


def load_peak_archive(source: Path, dataset: str) -> PeakArchive:
    path = source / f"{dataset}/per_peak/per_track_matrices.npz"
    with np.load(path, allow_pickle=False) as archive:
        return PeakArchive(
            dataset=dataset,
            track_ids=np.asarray(archive["track_ids"], dtype=int),
            axis_labels=np.asarray(archive["axis_labels"], dtype=str),
            axis_pressures=np.asarray(archive["axis_pressure_gpa"], dtype=float),
            scan_names=np.asarray(archive["scan_names"], dtype=str),
            location=symmetrize_track_stack(
                np.asarray(archive["location_profile_aggregate"], dtype=float)
            ),
            area=symmetrize_track_stack(
                np.asarray(archive["area_aggregate"], dtype=float)
            ),
        )


def parse_single_axis_label(label: str) -> dict[str, object]:
    match = re.match(
        r"^f(?P<frame>\d+)\|(?P<pressure>[-+0-9.]+)GPa\|(?P<tag>.+)$",
        str(label),
    )
    if not match:
        raise ValueError(f"Unrecognized single-crystal peak axis label: {label}")
    tag = match.group("tag")
    if tag.startswith("0deg"):
        orientation_base = "0deg"
    elif tag.startswith("10deg"):
        orientation_base = "10deg"
    else:
        orientation_base = "unknown"
    return {
        "frame": int(match.group("frame")),
        "pressure_GPa": float(match.group("pressure")),
        "axis_tag": tag,
        "orientation_base": orientation_base,
    }


def add_relative_peak_values(features: pd.DataFrame) -> pd.DataFrame:
    current = features.copy()
    current["track_center_median_deg"] = current.groupby("track")[
        "two_theta_deg"
    ].transform("median")
    current["center_shift_from_track_median_deg"] = (
        current["two_theta_deg"] - current["track_center_median_deg"]
    )
    group_columns = ["track"]
    if current["dataset"].astype(str).eq("single_crystal").all():
        group_columns.append("orientation_base")
    grouped = current.groupby(group_columns, sort=False)["log_area"]
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    current["within_track_area_rank"] = np.where(
        counts <= 1,
        0.5,
        (ranks - 1.0) / (counts - 1.0),
    )
    current["log2_area_ratio_to_track_median"] = (
        current["log_area"] - grouped.transform("median")
    ) / math.log(2.0)
    return current


def single_fingerprint_table(
    source: Path, archive: PeakArchive
) -> pd.DataFrame:
    features = pd.read_csv(
        source / "single_crystal/per_peak/frame_track_features.csv"
    )
    features = add_relative_peak_values(features)
    by_key = {
        (int(row.track), int(row.frame)): row
        for row in features.itertuples()
    }
    axes = [parse_single_axis_label(label) for label in archive.axis_labels]
    rows: list[dict[str, object]] = []
    for track in archive.track_ids:
        for axis_index, axis in enumerate(axes):
            key = (int(track), int(axis["frame"]))
            found = by_key.get(key)
            row: dict[str, object] = {
                "dataset": "single_crystal",
                "track": int(track),
                "axis_index": axis_index,
                "axis_label": str(archive.axis_labels[axis_index]),
                "frame": int(axis["frame"]),
                "pressure_GPa": float(axis["pressure_GPa"]),
                "axis_tag": str(axis["axis_tag"]),
                "orientation_base": str(axis["orientation_base"]),
                "frame_measured": 1,
                "state": "present" if found is not None else "unknown",
                "absence_confirmed": 0,
                "missing_semantics": (
                    "curated_detection"
                    if found is not None
                    else "not_determined_not_absence"
                ),
            }
            for field in (
                "two_theta_deg",
                "centroid_se_two_theta_deg",
                "fwhm_two_theta_deg",
                "log_area",
                "integrated_roi_area_counts_per_s",
                "track_center_median_deg",
                "center_shift_from_track_median_deg",
                "within_track_area_rank",
                "log2_area_ratio_to_track_median",
                "n_observations",
                "duplicate_observation_flag",
            ):
                row[field] = getattr(found, field) if found is not None else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def powder_fingerprint_table(
    source: Path,
    archive: PeakArchive,
    series: v3.SeriesData,
) -> pd.DataFrame:
    features = pd.read_csv(source / "powder/per_peak/frame_track_features.csv")
    features = add_relative_peak_values(features)
    by_key = {
        (str(row.scan), int(row.track), int(row.frame)): row
        for row in features.itertuples()
    }
    frame_by_scan_pressure: dict[tuple[str, float], int] = {}
    for frame in series.frames:
        frame_by_scan_pressure[(str(frame.scan), float(frame.pressure))] = int(
            frame.frame
        )
    rows: list[dict[str, object]] = []
    for scan in archive.scan_names:
        for track in archive.track_ids:
            for axis_index, pressure in enumerate(archive.axis_pressures):
                frame = frame_by_scan_pressure.get((str(scan), float(pressure)))
                found = (
                    by_key.get((str(scan), int(track), int(frame)))
                    if frame is not None
                    else None
                )
                if frame is None:
                    state = "not_measured"
                    semantics = "frame_not_measured"
                elif found is None:
                    state = "unknown"
                    semantics = "not_determined_not_absence"
                else:
                    state = "present"
                    semantics = "curated_detection"
                row: dict[str, object] = {
                    "dataset": "powder",
                    "scan": str(scan),
                    "track": int(track),
                    "axis_index": axis_index,
                    "axis_label": str(archive.axis_labels[axis_index]),
                    "frame": frame if frame is not None else np.nan,
                    "pressure_GPa": float(pressure),
                    "orientation_base": "not_applicable",
                    "frame_measured": int(frame is not None),
                    "state": state,
                    "absence_confirmed": 0,
                    "missing_semantics": semantics,
                }
                for field in (
                    "two_theta_deg",
                    "centroid_se_two_theta_deg",
                    "fwhm_two_theta_deg",
                    "log_area",
                    "integrated_roi_area_counts_per_s",
                    "track_center_median_deg",
                    "center_shift_from_track_median_deg",
                    "within_track_area_rank",
                    "log2_area_ratio_to_track_median",
                    "n_observations",
                    "duplicate_observation_flag",
                ):
                    row[field] = (
                        getattr(found, field) if found is not None else np.nan
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def peak_track_order(source: Path, dataset: str) -> tuple[list[int], dict[int, float]]:
    index = pd.read_csv(
        source / "per_peak_heatmaps/per_peak_heatmap_index.csv"
    )
    selected = index[index["dataset"].astype(str).eq(dataset)].copy()
    selected = selected.sort_values(
        ["median_two_theta_deg", "track"], kind="stable"
    )
    order = selected["track"].astype(int).tolist()
    medians = dict(
        zip(
            selected["track"].astype(int),
            selected["median_two_theta_deg"].astype(float),
        )
    )
    return order, medians


def marker_size(area_rank: float) -> float:
    if not np.isfinite(area_rank):
        return 22.0
    return 24.0 + 92.0 * float(np.clip(area_rank, 0.0, 1.0))


def draw_fingerprint_grid(
    ax: plt.Axes,
    table: pd.DataFrame,
    track_order: Sequence[int],
    axis_count: int,
    shift_norm: mcolors.Normalize,
    *,
    show_unknown_crosses: bool,
    connect_group: str | None = None,
) -> None:
    track_lookup = {int(track): index for index, track in enumerate(track_order)}
    background = np.zeros((len(track_order), axis_count), dtype=float)
    ax.imshow(
        background,
        cmap=mcolors.ListedColormap(["#f0f0f0"]),
        aspect="auto",
        interpolation="nearest",
        extent=(-0.5, axis_count - 0.5, len(track_order) - 0.5, -0.5),
        zorder=0,
    )
    for row in table.itertuples():
        y = track_lookup[int(row.track)]
        x = int(row.axis_index)
        if str(row.state) == "not_measured":
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor="#bdbdbd",
                    edgecolor="white",
                    linewidth=0.25,
                    zorder=1,
                )
            )
        elif str(row.state) == "unknown" and show_unknown_crosses:
            ax.plot(
                x,
                y,
                marker="x",
                markersize=2.0,
                markeredgewidth=0.35,
                color="#a6a6a6",
                zorder=2,
            )
    present = table[table["state"].astype(str).eq("present")].copy()
    if connect_group is not None and len(present):
        grouping = ["track", connect_group]
        for _, group in present.groupby(grouping, sort=False):
            group = group.sort_values("axis_index")
            indices = group["axis_index"].to_numpy(int)
            for left, right in zip(indices[:-1], indices[1:]):
                if right == left + 1:
                    y = track_lookup[int(group.iloc[0]["track"])]
                    ax.plot(
                        [left, right],
                        [y, y],
                        color="#7f7f7f",
                        linewidth=0.65,
                        zorder=2,
                    )
    if len(present):
        y = [track_lookup[int(track)] for track in present["track"]]
        colors = present["center_shift_from_track_median_deg"].to_numpy(float)
        sizes = [
            marker_size(float(value))
            for value in present["within_track_area_rank"].to_numpy(float)
        ]
        ax.scatter(
            present["axis_index"],
            y,
            c=colors,
            s=sizes,
            cmap=SHIFT_CMAP,
            norm=shift_norm,
            edgecolors="#2f2f2f",
            linewidths=0.35,
            zorder=3,
        )
    ax.set_xlim(-0.5, axis_count - 0.5)
    ax.set_ylim(len(track_order) - 0.5, -0.5)
    ax.set_xticks(range(axis_count))
    ax.set_yticks(range(len(track_order)))
    ax.grid(color="white", linewidth=0.32)
    ax.set_axisbelow(False)


def plot_single_peak_fingerprint(
    path: Path,
    table: pd.DataFrame,
    track_order: Sequence[int],
    medians: dict[int, float],
    archive: PeakArchive,
    dpi: int,
) -> None:
    limit = robust_symmetric_limit(
        table["center_shift_from_track_median_deg"].to_numpy(float),
        minimum=0.01,
    )
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, ax = plt.subplots(figsize=(14.8, 20.0))
    draw_fingerprint_grid(
        ax,
        table,
        track_order,
        len(archive.axis_labels),
        norm,
        show_unknown_crosses=True,
        connect_group="axis_tag",
    )
    axes = [parse_single_axis_label(label) for label in archive.axis_labels]
    xlabels = [
        f"F{int(item['frame'])}\n{compact(float(item['pressure_GPa']), 2)} GPa\n{item['axis_tag']}"
        for item in axes
    ]
    ylabels = [
        f"T{int(track):03d} · {medians[int(track)]:.2f}°"
        for track in track_order
    ]
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_yticklabels(ylabels, fontsize=6.2)
    ax.set_xlabel("Global curated frame axis")
    ax.set_ylabel("Peak track, ordered by median 2θ")
    ax.set_title(
        "Single crystal peak–frame fingerprint\n"
        "position = frame/track · color = center shift · size = within-track area rank",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    add_colorbar(
        fig,
        ax,
        SHIFT_CMAP,
        norm,
        "Detected center − track median (degrees)",
        fraction=0.025,
    )
    for size_value, label in ((24, "low"), (70, "mid"), (116, "high")):
        ax.scatter(
            [],
            [],
            s=size_value,
            facecolor="#bdbdbd",
            edgecolor=TEXT,
            linewidth=0.4,
            label=label,
        )
    ax.legend(
        title="Within-track area rank",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.88),
        frameon=False,
        fontsize=7,
        title_fontsize=7,
    )
    fig.text(
        0.01,
        0.007,
        (
            "Filled circle = curated detected peak. Gray cell/× = not determined, "
            "not a confirmed absence. Lines connect only adjacent detected frames "
            "inside the same orientation/branch; they never bridge unknown cells. "
            "Marker size compares area only within the same track and orientation. "
            "No causal or disappearance interpretation is encoded."
        ),
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 0.94, 1))
    save_figure(fig, path, dpi)


def plot_powder_peak_fingerprint_atlas(
    path: Path,
    table: pd.DataFrame,
    track_order: Sequence[int],
    medians: dict[int, float],
    archive: PeakArchive,
    dpi: int,
) -> None:
    limit = robust_symmetric_limit(
        table["center_shift_from_track_median_deg"].to_numpy(float),
        minimum=0.01,
    )
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    scans = archive.scan_names.tolist()
    columns = 7
    rows = math.ceil(len(scans) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(22.0, 18.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for index, scan in enumerate(scans):
        ax = axes.flat[index]
        current = table[table["scan"].astype(str).eq(str(scan))]
        draw_fingerprint_grid(
            ax,
            current,
            track_order,
            len(archive.axis_pressures),
            norm,
            show_unknown_crosses=False,
        )
        present = current["state"].astype(str).eq("present")
        ax.set_title(
            f"{scan} · {int(present.sum())} curated detections",
            fontsize=7.2,
            pad=2,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for index in range(len(scans), rows * columns):
        axes.flat[index].axis("off")
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xticks(
                range(len(archive.axis_pressures)),
                [compact(value, 2) for value in archive.axis_pressures],
                rotation=90,
                fontsize=5.2,
            )
    for ax in axes[:, 0]:
        if ax.axison:
            ax.set_yticks(
                range(len(track_order)),
                [f"T{track}" for track in track_order],
                fontsize=5.4,
            )
    fig.suptitle(
        "Powder peak–frame fingerprint atlas — every scan kept separate",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.supxlabel("Pressure axis (GPa)")
    fig.supylabel("Peak tracks ordered by median 2θ")
    color_ax = fig.add_axes([0.92, 0.16, 0.012, 0.68])
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=SHIFT_CMAP),
        cax=color_ax,
        label="Detected center − track median (degrees)",
    )
    fig.text(
        0.01,
        0.006,
        (
            "Filled circle = curated detected peak; color = center shift; size = "
            "within-track area rank. Light gray = unknown/not determined, not "
            "absence. Darker gray = that pressure frame was not measured in the "
            "scan. Scans are not collapsed to a pressure median."
        ),
        fontsize=7.5,
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.055, right=0.905, bottom=0.08, top=0.965, wspace=0.12, hspace=0.30
    )
    save_figure(fig, path, dpi)


def choose_powder_detail_scan(table: pd.DataFrame) -> str:
    present = table[table["state"].astype(str).eq("present")]
    stats = (
        present.groupby("scan", as_index=False)
        .agg(
            distinct_tracks=("track", "nunique"),
            present_cells=("track", "size"),
        )
        .sort_values(
            ["distinct_tracks", "present_cells", "scan"],
            ascending=[False, False, True],
            kind="stable",
        )
    )
    if not len(stats):
        return str(sorted(table["scan"].astype(str).unique())[0])
    return str(stats.iloc[0]["scan"])


def plot_powder_peak_fingerprint_detail(
    path: Path,
    table: pd.DataFrame,
    scan: str,
    track_order: Sequence[int],
    medians: dict[int, float],
    archive: PeakArchive,
    dpi: int,
) -> None:
    current = table[table["scan"].astype(str).eq(scan)].copy()
    limit = robust_symmetric_limit(
        table["center_shift_from_track_median_deg"].to_numpy(float),
        minimum=0.01,
    )
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, ax = plt.subplots(figsize=(17.0, 7.0))
    draw_fingerprint_grid(
        ax,
        current,
        track_order,
        len(archive.axis_pressures),
        norm,
        show_unknown_crosses=True,
    )
    ax.set_xticklabels(
        [compact(value, 2) for value in archive.axis_pressures],
        rotation=90,
        fontsize=7.5,
    )
    ax.set_yticklabels(
        [
            f"T{int(track):03d} · {medians[int(track)]:.2f}°"
            for track in track_order
        ],
        fontsize=8,
    )
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Peak track")
    ax.set_title(
        f"Powder peak–frame fingerprint detail · {scan}\n"
        "chosen for the largest number of distinct detected tracks",
        fontsize=14,
        fontweight="bold",
    )
    add_colorbar(
        fig,
        ax,
        SHIFT_CMAP,
        norm,
        "Detected center − track median (degrees)",
    )
    fig.text(
        0.01,
        0.008,
        (
            "Filled circle = curated detection; size = within-track area rank. "
            "Gray/× = unknown, not confirmed absence. This panel shows association "
            "and evolution only; it does not explain why a peak changed."
        ),
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, path, dpi)


def generate_peak_fingerprints(
    source: Path,
    output: Path,
    series_features: dict[str, SeriesFeature],
    dpi: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path], str]:
    single_archive = load_peak_archive(source, "single_crystal")
    powder_archive = load_peak_archive(source, "powder")
    single = single_fingerprint_table(source, single_archive)
    powder = powder_fingerprint_table(
        source, powder_archive, series_features["powder_spots"].series
    )
    single_order, single_medians = peak_track_order(
        source, "single_crystal"
    )
    powder_order, powder_medians = peak_track_order(source, "powder")
    write_csv(output / "single_crystal_peak_frame_fingerprint.csv", single)
    write_csv(output / "powder_peak_frame_fingerprint.csv", powder)
    generated: list[Path] = []
    single_path = output / "single_crystal_peak_frame_fingerprint.png"
    plot_single_peak_fingerprint(
        single_path,
        single,
        single_order,
        single_medians,
        single_archive,
        dpi,
    )
    generated.append(single_path)
    atlas_path = output / "powder_peak_frame_fingerprint_atlas.png"
    plot_powder_peak_fingerprint_atlas(
        atlas_path,
        powder,
        powder_order,
        powder_medians,
        powder_archive,
        dpi,
    )
    generated.append(atlas_path)
    detail_scan = choose_powder_detail_scan(powder)
    detail_path = (
        output
        / f"powder_peak_frame_fingerprint_detail_{safe_name(detail_scan)}.png"
    )
    plot_powder_peak_fingerprint_detail(
        detail_path,
        powder,
        detail_scan,
        powder_order,
        powder_medians,
        powder_archive,
        dpi,
    )
    generated.append(detail_path)
    return single, powder, generated, detail_scan


def unordered_pair_columns(frame: pd.DataFrame, high: str, low: str) -> pd.DataFrame:
    current = frame.copy()
    current["pair_frame_min"] = np.minimum(
        current[high].to_numpy(int), current[low].to_numpy(int)
    )
    current["pair_frame_max"] = np.maximum(
        current[high].to_numpy(int), current[low].to_numpy(int)
    )
    return current


def build_peak_evidence_sources(
    source: Path,
) -> dict[str, dict[str, pd.DataFrame]]:
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset in ("single_crystal", "powder"):
        root = source / f"{dataset}/per_peak"
        location = unordered_pair_columns(
            pd.read_csv(root / "location_pair_scores.csv"),
            "frame_high",
            "frame_low",
        )
        area = unordered_pair_columns(
            pd.read_csv(root / "area_pair_scores.csv"),
            "frame_high",
            "frame_low",
        )
        features = pd.read_csv(root / "frame_track_features.csv")
        result[dataset] = {
            "location": location,
            "area": area,
            "features": features,
        }
    return result


def evidence_subset(
    evidence: pd.DataFrame,
    dataset: str,
    scan: str,
    frame_a: int,
    frame_b: int,
) -> pd.DataFrame:
    minimum, maximum = sorted((int(frame_a), int(frame_b)))
    keep = evidence["pair_frame_min"].eq(minimum) & evidence[
        "pair_frame_max"
    ].eq(maximum)
    if dataset == "powder":
        keep &= evidence["scan"].astype(str).eq(str(scan))
    return evidence[keep].copy()


def build_frame_pair_peak_index(
    source: Path,
    evidence_sources: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key in ("single_0deg", "single_10deg", "powder_spots", "powder_fit"):
        whole = pd.read_csv(whole_pair_path(source, key))
        dataset = "powder" if key.startswith("powder") else "single_crystal"
        location = evidence_sources[dataset]["location"]
        area = evidence_sources[dataset]["area"]
        for item in whole.itertuples():
            loc = evidence_subset(
                location,
                dataset,
                str(item.scan),
                int(item.frame_a),
                int(item.frame_b),
            )
            ar = evidence_subset(
                area,
                dataset,
                str(item.scan),
                int(item.frame_a),
                int(item.frame_b),
            )
            rows.append(
                {
                    "map_key": key,
                    "dataset": dataset,
                    "channel": str(item.channel),
                    "scan": str(item.scan),
                    "frame_a": int(item.frame_a),
                    "frame_b": int(item.frame_b),
                    "pressure_a_GPa": float(item.pressure_a_GPa),
                    "pressure_b_GPa": float(item.pressure_b_GPa),
                    "pressure_gap_GPa": float(item.pressure_gap_GPa),
                    "whole_pattern_correlation": float(item.correlation),
                    "acquisition_protocol_changed": int(
                        item.acquisition_protocol_changed
                    ),
                    "location_support_tracks": int(loc["track"].nunique()),
                    "area_support_tracks": int(ar["track"].nunique()),
                    "median_location_similarity": (
                        float(
                            loc[
                                "fwhm_scaled_center_distance_similarity"
                            ].median()
                        )
                        if len(loc)
                        else np.nan
                    ),
                    "median_area_similarity": (
                        float(ar["area_similarity"].median())
                        if len(ar)
                        else np.nan
                    ),
                    "per_peak_evidence_semantics": (
                        "associated_non_additive_evidence_not_Pearson_contribution"
                    ),
                }
            )
    return pd.DataFrame(rows)


def select_peak_evidence_pairs(index: pd.DataFrame) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    for key, group in index.groupby("map_key", sort=False):
        maximum = int(group["location_support_tracks"].max())
        threshold = max(2, int(math.ceil(maximum / 2.0)))
        eligible = group[group["location_support_tracks"] >= threshold].copy()
        if not len(eligible):
            eligible = group[group["location_support_tracks"] > 0].copy()
        if not len(eligible):
            continue
        high = eligible.sort_values(
            [
                "whole_pattern_correlation",
                "location_support_tracks",
                "scan",
                "frame_a",
                "frame_b",
            ],
            ascending=[False, False, True, True, True],
            kind="stable",
        ).iloc[[0]].copy()
        low = eligible.sort_values(
            [
                "whole_pattern_correlation",
                "location_support_tracks",
                "scan",
                "frame_a",
                "frame_b",
            ],
            ascending=[True, False, True, True, True],
            kind="stable",
        ).iloc[[0]].copy()
        if (
            int(high.iloc[0]["frame_a"]) == int(low.iloc[0]["frame_a"])
            and int(high.iloc[0]["frame_b"]) == int(low.iloc[0]["frame_b"])
            and str(high.iloc[0]["scan"]) == str(low.iloc[0]["scan"])
        ):
            high["selection_role"] = "only_supported_selection"
            high["selection_support_threshold"] = threshold
            selected_parts.append(high)
        else:
            high["selection_role"] = "highest_correlation_with_peak_support"
            low["selection_role"] = "lowest_correlation_with_peak_support"
            high["selection_support_threshold"] = threshold
            low["selection_support_threshold"] = threshold
            selected_parts.extend([high, low])
    if not selected_parts:
        return pd.DataFrame()
    return pd.concat(selected_parts, ignore_index=True)


def evidence_rows_for_selection(
    selection: pd.Series,
    sources: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    dataset = str(selection["dataset"])
    scan = str(selection["scan"])
    frame_a = int(selection["frame_a"])
    frame_b = int(selection["frame_b"])
    location = evidence_subset(
        sources[dataset]["location"], dataset, scan, frame_a, frame_b
    )
    area = evidence_subset(
        sources[dataset]["area"], dataset, scan, frame_a, frame_b
    )
    location = location[
        [
            "track",
            "fwhm_scaled_center_distance_similarity",
            "centroid_consistency_similarity",
            "delta_two_theta_deg",
        ]
    ].drop_duplicates("track")
    area = area[["track", "area_similarity", "absolute_log_area_difference"]]
    area = area.drop_duplicates("track")
    merged = location.merge(area, on="track", how="outer", validate="one_to_one")
    features = sources[dataset]["features"]
    if dataset == "powder":
        features = features[features["scan"].astype(str).eq(scan)]
    endpoint_rows: list[dict[str, object]] = []
    for track in sorted(merged["track"].astype(int).unique()):
        row_a = features[
            features["track"].astype(int).eq(track)
            & features["frame"].astype(int).eq(frame_a)
        ]
        row_b = features[
            features["track"].astype(int).eq(track)
            & features["frame"].astype(int).eq(frame_b)
        ]
        current = merged[merged["track"].astype(int).eq(track)].iloc[0]
        endpoint_rows.append(
            {
                "track": track,
                "frame_a": frame_a,
                "frame_b": frame_b,
                "center_a_deg": (
                    float(row_a.iloc[0]["two_theta_deg"])
                    if len(row_a)
                    else np.nan
                ),
                "center_b_deg": (
                    float(row_b.iloc[0]["two_theta_deg"])
                    if len(row_b)
                    else np.nan
                ),
                "area_a_counts_per_s": (
                    float(row_a.iloc[0]["integrated_roi_area_counts_per_s"])
                    if len(row_a)
                    else np.nan
                ),
                "area_b_counts_per_s": (
                    float(row_b.iloc[0]["integrated_roi_area_counts_per_s"])
                    if len(row_b)
                    else np.nan
                ),
                "location_similarity": float(
                    current["fwhm_scaled_center_distance_similarity"]
                )
                if np.isfinite(
                    current["fwhm_scaled_center_distance_similarity"]
                )
                else np.nan,
                "centroid_consistency_similarity": float(
                    current["centroid_consistency_similarity"]
                )
                if np.isfinite(current["centroid_consistency_similarity"])
                else np.nan,
                "delta_two_theta_deg": float(current["delta_two_theta_deg"])
                if np.isfinite(current["delta_two_theta_deg"])
                else np.nan,
                "area_similarity": float(current["area_similarity"])
                if np.isfinite(current["area_similarity"])
                else np.nan,
                "absolute_log_area_difference": float(
                    current["absolute_log_area_difference"]
                )
                if np.isfinite(current["absolute_log_area_difference"])
                else np.nan,
            }
        )
    frame = pd.DataFrame(endpoint_rows)
    frame["median_center_deg"] = frame[["center_a_deg", "center_b_deg"]].median(
        axis=1
    )
    return frame.sort_values(
        ["median_center_deg", "track"], kind="stable"
    ).reset_index(drop=True)


def plot_frame_pair_peak_evidence(
    path: Path,
    selection: pd.Series,
    evidence: pd.DataFrame,
    dpi: int,
) -> None:
    count = len(evidence)
    height = max(5.0, min(15.0, 2.8 + 0.42 * count))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, height),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 0.82, 0.82], "wspace": 0.12},
    )
    y = np.arange(count)
    labels = [
        f"T{int(row.track):03d} · {row.median_center_deg:.2f}°"
        for row in evidence.itertuples()
    ]
    centers = np.concatenate(
        [
            evidence["center_a_deg"].to_numpy(float),
            evidence["center_b_deg"].to_numpy(float),
        ]
    )
    finite_centers = centers[np.isfinite(centers)]
    if len(finite_centers):
        lower, upper = float(finite_centers.min()), float(finite_centers.max())
        pad = max(0.15, 0.03 * max(upper - lower, 1.0))
        axes[0].set_xlim(lower - pad, upper + pad)
    for index, row in enumerate(evidence.itertuples()):
        if np.isfinite(row.center_a_deg) and np.isfinite(row.center_b_deg):
            axes[0].plot(
                [row.center_a_deg, row.center_b_deg],
                [index, index],
                color="#8f8f8f",
                linewidth=1.4,
                zorder=1,
            )
            axes[0].scatter(
                [row.center_a_deg],
                [index],
                color=FRAME_A,
                s=32,
                zorder=2,
            )
            axes[0].scatter(
                [row.center_b_deg],
                [index],
                color=FRAME_B,
                s=32,
                zorder=2,
            )
    axes[0].set_yticks(y, labels, fontsize=7)
    axes[0].set_xlabel(r"Detected center $2\theta$ (degrees)")
    axes[0].set_title("Detected centers")
    axes[0].grid(axis="x", color=GRID)
    similarity_norm = mcolors.Normalize(0.0, 1.0)
    for ax, column, title in (
        (axes[1], "location_similarity", "Location similarity"),
        (axes[2], "area_similarity", "Area similarity"),
    ):
        ax.set_xlim(0.0, 1.0)
        ax.axvline(0.5, color="#d4d4d4", linewidth=0.8)
        for index, value in enumerate(evidence[column].to_numpy(float)):
            if np.isfinite(value):
                ax.plot(
                    [0.0, value],
                    [index, index],
                    color="#cfcfcf",
                    linewidth=2.0,
                    zorder=1,
                )
                ax.scatter(
                    [value],
                    [index],
                    c=[value],
                    cmap=SIMILARITY_CMAP,
                    norm=similarity_norm,
                    s=46,
                    edgecolors=TEXT,
                    linewidths=0.35,
                    zorder=2,
                )
                ax.text(
                    min(value + 0.025, 0.97),
                    index,
                    f"{value:.3f}",
                    ha="left" if value < 0.9 else "right",
                    va="center",
                    fontsize=6.2,
                )
            else:
                ax.text(
                    0.5,
                    index,
                    "—",
                    ha="center",
                    va="center",
                    color="#8f8f8f",
                    fontsize=9,
                )
        ax.set_xlabel("Similarity (0–1)")
        ax.set_title(title)
        ax.grid(axis="x", color=GRID)
    for ax in axes:
        ax.set_ylim(count - 0.5, -0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    axes[0].scatter([], [], color=FRAME_A, s=32, label=f"F{int(selection['frame_a'])}")
    axes[0].scatter([], [], color=FRAME_B, s=32, label=f"F{int(selection['frame_b'])}")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    protocol = (
        f" · {PROTOCOL_MARK} filename signature changed"
        if int(selection["acquisition_protocol_changed"])
        else ""
    )
    fig.suptitle(
        (
            f"{selection['map_key']} · {selection['scan']}: "
            f"F{int(selection['frame_a'])} ({float(selection['pressure_a_GPa']):g} GPa) "
            f"↔ F{int(selection['frame_b'])} ({float(selection['pressure_b_GPa']):g} GPa)\n"
            f"whole-pattern r={float(selection['whole_pattern_correlation']):.4f}; "
            f"{count} comparable peak tracks{protocol}"
        ),
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.006,
        (
            "Per-peak evidence associated with this map cell (non-additive). "
            "It is not a decomposition or percentage contribution to whole-pattern "
            "Pearson. Gray em dash = incomparable/unknown, not similarity zero. "
            "Single-crystal area evidence is uncalibrated/secondary; powder area "
            "evidence is repeatability-calibrated but remains secondary."
        ),
        fontsize=7.4,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    save_figure(fig, path, dpi)


def generate_peak_evidence_outputs(
    source: Path,
    output: Path,
    dpi: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    sources = build_peak_evidence_sources(source)
    index = build_frame_pair_peak_index(source, sources)
    selected = select_peak_evidence_pairs(index)
    write_csv(output / "frame_pair_peak_evidence_index.csv", index)
    write_csv(output / "selected_pairs.csv", selected)
    generated: list[Path] = []
    evidence_csv_rows: list[pd.DataFrame] = []
    for selection_index, selection in selected.iterrows():
        evidence = evidence_rows_for_selection(selection, sources)
        stem = (
            f"{selection['map_key']}_{safe_name(selection['scan'])}_"
            f"F{int(selection['frame_a'])}_F{int(selection['frame_b'])}_"
            f"{safe_name(selection['selection_role'])}"
        )
        csv_path = output / str(selection["map_key"]) / f"{stem}.csv"
        enriched = evidence.copy()
        for field in (
            "map_key",
            "scan",
            "frame_a",
            "frame_b",
            "pressure_a_GPa",
            "pressure_b_GPa",
            "whole_pattern_correlation",
            "selection_role",
        ):
            enriched[field] = selection[field]
        write_csv(csv_path, enriched)
        evidence_csv_rows.append(enriched)
        png_path = output / str(selection["map_key"]) / f"{stem}.png"
        plot_frame_pair_peak_evidence(png_path, selection, evidence, dpi)
        generated.append(png_path)
    if evidence_csv_rows:
        write_csv(
            output / "selected_pair_peak_evidence_long.csv",
            pd.concat(evidence_csv_rows, ignore_index=True),
        )
    else:
        write_csv(
            output / "selected_pair_peak_evidence_long.csv", pd.DataFrame()
        )
    return index, selected, generated


def primary_window_indices(source: Path, key: str) -> list[int]:
    adjacent = pd.read_csv(
        source / f"windows/{key}/adjacent_window_trajectories.csv",
        usecols=["window_index", "nonoverlap_primary"],
    )
    selected = (
        adjacent[adjacent["nonoverlap_primary"].eq(1)]["window_index"]
        .drop_duplicates()
        .astype(int)
        .sort_values()
        .tolist()
    )
    if not selected:
        raise ValueError(f"No primary non-overlap windows found for {key}")
    return selected


def window_peak_membership(
    source: Path,
    key: str,
    starts: Sequence[float],
    primary: Sequence[int],
    width: float = 5.0,
) -> pd.DataFrame:
    if key.startswith("powder"):
        dataset = "powder"
        features = pd.read_csv(
            source / "powder/per_peak/frame_track_features.csv"
        )
    else:
        dataset = "single_crystal"
        features = pd.read_csv(
            source / "single_crystal/per_peak/frame_track_features.csv"
        )
        orientation = "0deg" if key.endswith("0deg") else "10deg"
        features = features[
            features["orientation_base"].astype(str).eq(orientation)
        ]
    centers = (
        features.groupby("track", as_index=False)
        .agg(median_two_theta_deg=("two_theta_deg", "median"))
        .sort_values(["median_two_theta_deg", "track"], kind="stable")
    )
    rows: list[dict[str, object]] = []
    primary = list(primary)
    for position, window_index in enumerate(primary):
        start = float(starts[window_index])
        end = start + width
        if position == len(primary) - 1:
            inside = centers[
                centers["median_two_theta_deg"].between(
                    start, end, inclusive="both"
                )
            ]
        else:
            inside = centers[
                centers["median_two_theta_deg"].ge(start)
                & centers["median_two_theta_deg"].lt(end)
            ]
        if not len(inside):
            rows.append(
                {
                    "map_key": key,
                    "dataset": dataset,
                    "window_index": int(window_index),
                    "window_start_deg": start,
                    "window_end_deg": end,
                    "track": np.nan,
                    "median_two_theta_deg": np.nan,
                    "membership": "no_curated_track_center_in_primary_window",
                }
            )
        else:
            for item in inside.itertuples():
                rows.append(
                    {
                        "map_key": key,
                        "dataset": dataset,
                        "window_index": int(window_index),
                        "window_start_deg": start,
                        "window_end_deg": end,
                        "track": int(item.track),
                        "median_two_theta_deg": float(
                            item.median_two_theta_deg
                        ),
                        "membership": (
                            "median_peak_center_inside_primary_nonoverlap_window"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def interval_order(frame: pd.DataFrame) -> list[tuple[float, float]]:
    intervals = (
        frame[["p_low_GPa", "p_high_GPa"]]
        .drop_duplicates()
        .sort_values(["p_low_GPa", "p_high_GPa"], kind="stable")
    )
    return [
        (float(row.p_low_GPa), float(row.p_high_GPa))
        for row in intervals.itertuples()
    ]


def interval_matrix(
    frame: pd.DataFrame,
    primary: Sequence[int],
    intervals: Sequence[tuple[float, float]],
    value: str,
) -> np.ndarray:
    lookup = {
        (
            int(row.window_index),
            float(row.p_low_GPa),
            float(row.p_high_GPa),
        ): float(getattr(row, value))
        for row in frame.itertuples()
    }
    result = np.full((len(primary), len(intervals)), np.nan, dtype=float)
    for y, window_index in enumerate(primary):
        for x, (low, high) in enumerate(intervals):
            result[y, x] = lookup.get((int(window_index), low, high), np.nan)
    return result


def merge_interval_protocol(
    source: Path, key: str, trajectory: pd.DataFrame
) -> pd.DataFrame:
    adjacent = pd.read_csv(
        source / f"windows/{key}/adjacent_window_trajectories.csv"
    )
    protocol = (
        adjacent.groupby(
            ["window_index", "p_low_GPa", "p_high_GPa"], as_index=False
        )
        .agg(
            protocol_change_fraction=(
                "acquisition_protocol_changed",
                "mean",
            ),
            independent_scan_support=("scan", "nunique"),
        )
    )
    return trajectory.merge(
        protocol,
        on=["window_index", "p_low_GPa", "p_high_GPa"],
        how="left",
        validate="one_to_one",
    )


def overlay_interval_flags(
    ax: plt.Axes,
    protocol: np.ndarray,
    support: np.ndarray,
    minimum_support: int,
) -> None:
    rows, columns = protocol.shape
    for y in range(rows):
        for x in range(columns):
            if np.isfinite(support[y, x]) and support[y, x] < minimum_support:
                ax.add_patch(
                    plt.Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#555555",
                        linewidth=0.8,
                        hatch="////",
                    )
                )
            if np.isfinite(protocol[y, x]) and protocol[y, x] > 0:
                ax.plot(
                    x + 0.30,
                    y - 0.30,
                    marker="^",
                    markersize=4.2,
                    color="#111111",
                    clip_on=False,
                )


def plot_window_interval_evidence(
    path: Path,
    key: str,
    frame: pd.DataFrame,
    primary: Sequence[int],
    starts: Sequence[float],
    membership: pd.DataFrame,
    dpi: int,
) -> None:
    intervals = interval_order(frame)
    aligned = interval_matrix(
        frame, primary, intervals, "median_aligned_ncc"
    )
    residual = interval_matrix(
        frame, primary, intervals, "median_delta_p_adjusted_change"
    )
    shift = interval_matrix(
        frame, primary, intervals, "median_best_shift_deg"
    )
    support = interval_matrix(
        frame, primary, intervals, "independent_scan_support"
    )
    protocol = interval_matrix(
        frame, primary, intervals, "protocol_change_fraction"
    )
    residual_limit = robust_symmetric_limit(residual.ravel(), minimum=0.02)
    residual_norm = mcolors.TwoSlopeNorm(
        vmin=-residual_limit, vcenter=0.0, vmax=residual_limit
    )
    shift_limit = max(
        0.02,
        float(np.nanmax(np.abs(shift))) if np.any(np.isfinite(shift)) else 0.12,
    )
    shift_norm = mcolors.TwoSlopeNorm(
        vmin=-shift_limit, vcenter=0.0, vmax=shift_limit
    )
    signed_norm = mcolors.Normalize(-1.0, 1.0)
    labels: list[str] = []
    for index in primary:
        start = float(starts[index])
        tracks = membership[
            membership["window_index"].astype(int).eq(int(index))
            & membership["track"].notna()
        ]["track"].astype(int)
        labels.append(
            f"{start:.2f}–{start + 5.0:.2f}° · {tracks.nunique()} peak tracks"
        )
    xlabels = [f"{low:g}→{high:g}" for low, high in intervals]
    minimum_support = 45 if key.startswith("powder") else 1
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16.5, 8.7),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    panels = [
        (
            aligned,
            SIGNED_CMAP,
            signed_norm,
            "Aligned direct NCC (similarity)",
            "Aligned NCC",
        ),
        (
            residual,
            CHANGE_CMAP,
            residual_norm,
            "ΔP-adjusted shape change (change evidence)",
            "Residual change",
        ),
        (
            shift,
            SHIFT_CMAP,
            shift_norm,
            "Best alignment shift (degrees)",
            "Best shift (°)",
        ),
    ]
    for ax, (values, cmap, norm, title, colorbar_label) in zip(axes, panels):
        ax.imshow(
            values,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(primary)), labels, fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        overlay_interval_flags(ax, protocol, support, minimum_support)
        add_colorbar(fig, ax, cmap, norm, colorbar_label, fraction=0.020)
    axes[-1].set_xticks(
        range(len(intervals)), xlabels, rotation=55, ha="right", fontsize=7
    )
    axes[-1].set_xlabel("Observed adjacent pressure interval (GPa; categorical)")
    axes[1].set_ylabel("Primary non-overlapping 5° windows")
    fig.suptitle(
        f"{key.replace('_', ' ')}: window × pressure-interval relationship waterfall",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.008,
        (
            "Top panel is similarity; middle panel is change evidence, not "
            "similarity. Triangle = filename-derived acquisition signature change "
            "in at least one supporting scan. Hatched cell = support below "
            f"{minimum_support} independent scans. Single-crystal support is one "
            "scan, so quartiles do not represent replicate uncertainty. Primary "
            "non-overlap windows avoid the inflated agreement of sliding windows."
        ),
        fontsize=7.3,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, path, dpi)


def matched_powder_interval_summary(source: Path) -> pd.DataFrame:
    matched = pd.read_csv(
        source
        / "window_to_window/matched_spots_fit_window_changes.csv"
    )
    primary = matched[matched["nonoverlap_primary"].eq(1)].copy()
    return (
        primary.groupby(
            [
                "window_index",
                "window_start_deg",
                "window_end_deg",
                "p_low_GPa",
                "p_high_GPa",
                "delta_p_GPa",
            ],
            as_index=False,
        )
        .agg(
            independent_scan_support=("scan", "nunique"),
            median_spots_residual=(
                "delta_p_adjusted_change_spots",
                "median",
            ),
            median_fit_residual=(
                "delta_p_adjusted_change_fit",
                "median",
            ),
            median_sample_specific_excess=(
                "sample_specific_excess_change",
                "median",
            ),
            q25_sample_specific_excess=(
                "sample_specific_excess_change",
                lambda values: np.nanquantile(values, 0.25),
            ),
            q75_sample_specific_excess=(
                "sample_specific_excess_change",
                lambda values: np.nanquantile(values, 0.75),
            ),
            protocol_change_fraction=(
                "acquisition_protocol_changed",
                "mean",
            ),
        )
    )


def plot_powder_spots_fit_interval_evidence(
    path: Path,
    summary: pd.DataFrame,
    primary: Sequence[int],
    starts: Sequence[float],
    membership: pd.DataFrame,
    dpi: int,
) -> None:
    intervals = interval_order(summary)
    spots = interval_matrix(
        summary, primary, intervals, "median_spots_residual"
    )
    fit = interval_matrix(summary, primary, intervals, "median_fit_residual")
    excess = interval_matrix(
        summary, primary, intervals, "median_sample_specific_excess"
    )
    support = interval_matrix(
        summary, primary, intervals, "independent_scan_support"
    )
    protocol = interval_matrix(
        summary, primary, intervals, "protocol_change_fraction"
    )
    limit = robust_symmetric_limit(
        np.concatenate([spots.ravel(), fit.ravel(), excess.ravel()]),
        minimum=0.02,
    )
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    labels: list[str] = []
    for index in primary:
        start = float(starts[index])
        tracks = membership[
            membership["window_index"].astype(int).eq(int(index))
            & membership["track"].notna()
        ]["track"].astype(int)
        labels.append(
            f"{start:.2f}–{start + 5.0:.2f}° · {tracks.nunique()} peak tracks"
        )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16.5, 8.4),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    for ax, values, title in zip(
        axes,
        (spots, fit, excess),
        (
            "Spots-channel ΔP-adjusted change",
            "Fit/control-channel ΔP-adjusted change",
            "Spots − fit excess change (per-scan difference, then median)",
        ),
    ):
        ax.imshow(
            values,
            aspect="auto",
            cmap=CHANGE_CMAP,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(primary)), labels, fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        overlay_interval_flags(ax, protocol, support, 45)
        add_colorbar(
            fig,
            ax,
            CHANGE_CMAP,
            norm,
            "Residual change",
            fraction=0.020,
        )
    intervals_text = [f"{low:g}→{high:g}" for low, high in intervals]
    axes[-1].set_xticks(
        range(len(intervals)),
        intervals_text,
        rotation=55,
        ha="right",
        fontsize=7,
    )
    axes[-1].set_xlabel("Observed adjacent pressure interval (GPa; categorical)")
    axes[1].set_ylabel("Primary non-overlapping 5° windows")
    fig.suptitle(
        "Powder window evidence: sample channel, matched fit channel, and excess",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.008,
        (
            "Positive excess means the spots channel changed more than its "
            "ΔP expectation relative to the matched fit channel. It is descriptive "
            "control evidence, not causal attribution or a phase/event label. "
            "Triangle = detected filename-signature change; hatch = <45 scans. "
            "The source v3 analysis promoted zero candidate boundaries."
        ),
        fontsize=7.3,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, path, dpi)


def frame_lookup_from_adjacent(
    adjacent: pd.DataFrame,
) -> dict[tuple[str, float], int]:
    lookup: dict[tuple[str, float], int] = {}
    for row in adjacent.itertuples():
        for pressure, frame in (
            (float(row.p_low_GPa), int(row.frame_low)),
            (float(row.p_high_GPa), int(row.frame_high)),
        ):
            key = (str(row.scan), pressure)
            previous = lookup.get(key)
            if previous is not None and previous != frame:
                raise ValueError(f"Conflicting frame mapping for {key}")
            lookup[key] = frame
    return lookup


def same_window_top_pairs(
    source: Path,
    key: str,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = source / f"windows/{key}/same_window_matrices.npz"
    with np.load(path, allow_pickle=False) as archive:
        pressures = np.asarray(archive["pressure_gpa"], dtype=float)
        scans = np.asarray(archive["scan_names"], dtype=str)
        starts = np.asarray(archive["window_starts_deg"], dtype=float)
        zero_scan = np.asarray(archive["zero_shift_by_scan"], dtype=float)
        aligned_scan = np.asarray(archive["aligned_by_scan"], dtype=float)
        shift_scan = np.asarray(
            archive["best_shift_deg_by_scan"], dtype=float
        )
        zero_aggregate = np.asarray(
            archive["zero_shift_aggregate"], dtype=float
        )
        aligned_aggregate = np.asarray(
            archive["aligned_aggregate"], dtype=float
        )
        shift_aggregate = np.asarray(
            archive["best_shift_deg_aggregate"], dtype=float
        )
    primary = primary_window_indices(source, key)
    aggregate_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    adjacent = pd.read_csv(
        source / f"windows/{key}/adjacent_window_trajectories.csv"
    )
    frame_lookup = frame_lookup_from_adjacent(adjacent)
    for window_index in primary:
        candidates: list[dict[str, object]] = []
        for high in range(len(pressures)):
            for low in range(high):
                aligned = float(aligned_aggregate[window_index, high, low])
                zero = float(zero_aggregate[window_index, high, low])
                shift = float(shift_aggregate[window_index, high, low])
                support = int(
                    np.count_nonzero(
                        np.isfinite(aligned_scan[:, window_index, high, low])
                    )
                )
                candidates.append(
                    {
                        "map_key": key,
                        "window_index": int(window_index),
                        "window_start_deg": float(starts[window_index]),
                        "window_end_deg": float(starts[window_index] + 5.0),
                        "pressure_low_GPa": float(pressures[low]),
                        "pressure_high_GPa": float(pressures[high]),
                        "pressure_gap_GPa": float(
                            pressures[high] - pressures[low]
                        ),
                        "aligned_ncc": aligned,
                        "zero_shift_ncc": zero,
                        "alignment_gain": aligned - zero,
                        "best_shift_deg": shift,
                        "independent_scan_support": support,
                    }
                )
        ranked = pd.DataFrame(candidates).sort_values(
            [
                "aligned_ncc",
                "pressure_gap_GPa",
                "pressure_low_GPa",
                "pressure_high_GPa",
            ],
            ascending=[False, True, True, True],
            kind="stable",
        )
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        aggregate_rows.extend(ranked.head(top_k).to_dict("records"))

        for scan_index, scan in enumerate(scans):
            local: list[dict[str, object]] = []
            for high in range(len(pressures)):
                for low in range(high):
                    aligned = float(
                        aligned_scan[scan_index, window_index, high, low]
                    )
                    if not np.isfinite(aligned):
                        continue
                    p_low = float(pressures[low])
                    p_high = float(pressures[high])
                    local.append(
                        {
                            "map_key": key,
                            "scan": str(scan),
                            "window_index": int(window_index),
                            "window_start_deg": float(starts[window_index]),
                            "window_end_deg": float(
                                starts[window_index] + 5.0
                            ),
                            "frame_low": frame_lookup.get(
                                (str(scan), p_low), np.nan
                            ),
                            "frame_high": frame_lookup.get(
                                (str(scan), p_high), np.nan
                            ),
                            "pressure_low_GPa": p_low,
                            "pressure_high_GPa": p_high,
                            "pressure_gap_GPa": p_high - p_low,
                            "aligned_ncc": aligned,
                            "zero_shift_ncc": float(
                                zero_scan[
                                    scan_index, window_index, high, low
                                ]
                            ),
                            "best_shift_deg": float(
                                shift_scan[
                                    scan_index, window_index, high, low
                                ]
                            ),
                        }
                    )
            local_frame = pd.DataFrame(local).sort_values(
                [
                    "aligned_ncc",
                    "pressure_gap_GPa",
                    "pressure_low_GPa",
                    "pressure_high_GPa",
                ],
                ascending=[False, True, True, True],
                kind="stable",
            )
            local_frame["rank"] = np.arange(1, len(local_frame) + 1)
            frame_rows.extend(local_frame.head(top_k).to_dict("records"))
    return pd.DataFrame(aggregate_rows), pd.DataFrame(frame_rows)


def plot_same_window_top_pairs(
    path: Path,
    key: str,
    rows: pd.DataFrame,
    membership: pd.DataFrame,
    dpi: int,
) -> None:
    current = rows.sort_values(
        ["window_index", "rank"], kind="stable"
    ).reset_index(drop=True)
    y = np.arange(len(current))
    labels: list[str] = []
    for row in current.itertuples():
        peak_count = membership[
            membership["window_index"].astype(int).eq(int(row.window_index))
            & membership["track"].notna()
        ]["track"].nunique()
        labels.append(
            f"{row.window_start_deg:.2f}–{row.window_end_deg:.2f}°"
            f" · #{int(row.rank)} · {row.pressure_low_GPa:g}↔{row.pressure_high_GPa:g} GPa"
            f" · {peak_count} peaks"
        )
    fig, ax = plt.subplots(
        figsize=(12.8, max(5.5, 2.6 + 0.36 * len(current)))
    )
    norm = mcolors.Normalize(-1.0, 1.0)
    ax.axvline(0.0, color="#a0a0a0", linewidth=0.8)
    for index, row in enumerate(current.itertuples()):
        ax.plot(
            [row.zero_shift_ncc, row.aligned_ncc],
            [index, index],
            color="#8d8d8d",
            linewidth=1.5,
        )
        ax.scatter(
            [row.zero_shift_ncc],
            [index],
            facecolors="white",
            edgecolors="#505050",
            linewidths=0.8,
            s=42,
            zorder=3,
        )
        ax.scatter(
            [row.aligned_ncc],
            [index],
            c=[row.aligned_ncc],
            cmap=SIGNED_CMAP,
            norm=norm,
            edgecolors=TEXT,
            linewidths=0.4,
            s=54,
            zorder=4,
        )
        ax.text(
            (
                float(row.aligned_ncc) + 0.025
                if float(row.aligned_ncc) < 0.86
                else 0.86
            ),
            index,
            f"shift={float(row.best_shift_deg):+.3f}° · n={int(row.independent_scan_support)}",
            ha="left" if float(row.aligned_ncc) < 0.86 else "right",
            va="center",
            fontsize=6.5,
        )
    ax.set_yticks(y, labels, fontsize=7.2)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(len(current) - 0.6, -0.6)
    ax.set_xlabel("Direct NCC")
    ax.set_title(
        f"{key.replace('_', ' ')}: Top correlated pressure pairs in each primary window",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter(
        [], [], facecolors="white", edgecolors="#505050", s=42, label="zero shift"
    )
    ax.scatter([], [], color="#4f7fa6", s=54, label="aligned")
    ax.legend(frameon=False, loc="lower left")
    add_colorbar(fig, ax, SIGNED_CMAP, norm, "Aligned NCC")
    fig.text(
        0.01,
        0.008,
        (
            "Ranking uses the aggregate within-scan pressure-pair median. "
            "It never compares frames across scans. Filled dot = aligned NCC; "
            "hollow dot = zero-shift NCC. Window–peak membership uses each track's "
            "median detected center; full track lists are in window_peak_membership.csv."
        ),
        fontsize=7.3,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_figure(fig, path, dpi)


def read_strict_lower_csv(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path, index_col=0)
    labels = frame.columns.astype(str).tolist()
    if frame.index.astype(str).tolist() != labels:
        raise ValueError(f"Matrix row/column labels differ: {path}")
    return labels, symmetrize_lower(frame.to_numpy(float))


def window_trajectory_pairs(
    source: Path,
    key: str,
    primary: Sequence[int],
    starts: Sequence[float],
) -> pd.DataFrame:
    labels, matrix = read_strict_lower_csv(
        source / f"windows/{key}/window_trajectory_correlation_matrix.csv"
    )
    rows: list[dict[str, object]] = []
    for high_position in range(len(primary)):
        for low_position in range(high_position):
            high = int(primary[high_position])
            low = int(primary[low_position])
            rows.append(
                {
                    "map_key": key,
                    "window_a_index": low,
                    "window_a_start_deg": float(starts[low]),
                    "window_a_end_deg": float(starts[low] + 5.0),
                    "window_b_index": high,
                    "window_b_start_deg": float(starts[high]),
                    "window_b_end_deg": float(starts[high] + 5.0),
                    "trajectory_correlation": float(matrix[high, low]),
                    "matrix_label_a": labels[low],
                    "matrix_label_b": labels[high],
                    "semantics": (
                        "Pearson_r_between_deltaP_adjusted_change_trajectories"
                    ),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        [
            "trajectory_correlation",
            "window_a_start_deg",
            "window_b_start_deg",
        ],
        ascending=[False, True, True],
        kind="stable",
    )
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def plot_window_trajectory_pairs(
    path: Path,
    key: str,
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    dpi: int,
) -> None:
    current = frame.sort_values("trajectory_correlation").reset_index(drop=True)
    y = np.arange(len(current))
    labels: list[str] = []
    for row in current.itertuples():
        count_a = membership[
            membership["window_index"].astype(int).eq(int(row.window_a_index))
            & membership["track"].notna()
        ]["track"].nunique()
        count_b = membership[
            membership["window_index"].astype(int).eq(int(row.window_b_index))
            & membership["track"].notna()
        ]["track"].nunique()
        labels.append(
            f"{row.window_a_start_deg:.2f}–{row.window_a_end_deg:.2f}°"
            f" ↔ {row.window_b_start_deg:.2f}–{row.window_b_end_deg:.2f}°"
            f" · peaks {count_a}/{count_b}"
        )
    fig, ax = plt.subplots(
        figsize=(12.8, max(4.8, 2.7 + 0.42 * len(current)))
    )
    norm = mcolors.Normalize(-1.0, 1.0)
    ax.axvline(0.0, color="#808080", linewidth=0.8)
    colors = SIGNED_CMAP(norm(current["trajectory_correlation"].to_numpy(float)))
    ax.hlines(
        y,
        0.0,
        current["trajectory_correlation"],
        color="#bdbdbd",
        linewidth=2.0,
    )
    ax.scatter(
        current["trajectory_correlation"],
        y,
        c=colors,
        s=62,
        edgecolors=TEXT,
        linewidths=0.4,
    )
    for index, value in enumerate(
        current["trajectory_correlation"].to_numpy(float)
    ):
        ax.text(
            min(value + 0.025, 0.96),
            index,
            f"r={value:.3f}",
            ha="left" if value < 0.88 else "right",
            va="center",
            fontsize=7,
        )
    ax.set_yticks(y, labels, fontsize=7.4)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Pearson r between pressure-change trajectories")
    ax.set_title(
        f"{key.replace('_', ' ')}: which primary windows change together",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    add_colorbar(fig, ax, SIGNED_CMAP, norm, "Trajectory r")
    fig.text(
        0.01,
        0.008,
        (
            "This is the v3 window-to-window correlation: Pearson r between "
            "ΔP-adjusted adjacent-pressure change trajectories. It is not a "
            "within-frame ACF comparison. Only primary non-overlapping windows "
            "are ranked; no causal or phase/event interpretation is encoded."
        ),
        fontsize=7.3,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, path, dpi)


def generate_window_outputs(
    source: Path,
    output: Path,
    top_k: int,
    dpi: int,
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    interval_parts: list[pd.DataFrame] = []
    membership_parts: list[pd.DataFrame] = []
    aggregate_top_parts: list[pd.DataFrame] = []
    frame_top_parts: list[pd.DataFrame] = []
    trajectory_parts: list[pd.DataFrame] = []
    generated: list[Path] = []
    state: dict[str, tuple[pd.DataFrame, list[int], np.ndarray, pd.DataFrame]] = {}

    for key in ("single_0deg", "single_10deg", "powder_spots", "powder_fit"):
        trajectory = pd.read_csv(
            source / f"windows/{key}/trajectory_by_interval.csv"
        )
        trajectory = merge_interval_protocol(source, key, trajectory)
        primary = primary_window_indices(source, key)
        with np.load(
            source / f"windows/{key}/same_window_matrices.npz",
            allow_pickle=False,
        ) as archive:
            starts = np.asarray(archive["window_starts_deg"], dtype=float)
        membership = window_peak_membership(
            source, key, starts, primary, width=5.0
        )
        membership_parts.append(membership)
        selected_interval = trajectory[
            trajectory["window_index"].astype(int).isin(primary)
        ].copy()
        selected_interval.insert(0, "map_key", key)
        interval_parts.append(selected_interval)
        state[key] = (trajectory, primary, starts, membership)

        interval_path = output / f"{key}_window_interval_evidence_waterfall.png"
        plot_window_interval_evidence(
            interval_path,
            key,
            trajectory,
            primary,
            starts,
            membership,
            dpi,
        )
        generated.append(interval_path)

        aggregate_top, frame_top = same_window_top_pairs(
            source, key, top_k
        )
        aggregate_top_parts.append(aggregate_top)
        frame_top_parts.append(frame_top)
        top_path = output / f"{key}_same_window_top_pressure_pairs.png"
        plot_same_window_top_pairs(
            top_path, key, aggregate_top, membership, dpi
        )
        generated.append(top_path)

        trajectory_pairs = window_trajectory_pairs(
            source, key, primary, starts
        )
        trajectory_parts.append(trajectory_pairs)
        trajectory_path = output / f"{key}_window_trajectory_top_pairs.png"
        plot_window_trajectory_pairs(
            trajectory_path, key, trajectory_pairs, membership, dpi
        )
        generated.append(trajectory_path)

    interval_summary = pd.concat(interval_parts, ignore_index=True)
    membership_frame = pd.concat(membership_parts, ignore_index=True)
    aggregate_top_frame = pd.concat(aggregate_top_parts, ignore_index=True)
    frame_top_frame = pd.concat(frame_top_parts, ignore_index=True)
    trajectory_frame = pd.concat(trajectory_parts, ignore_index=True)
    write_csv(output / "primary_window_interval_summary.csv", interval_summary)
    write_csv(output / "window_peak_membership.csv", membership_frame)
    write_csv(
        output / "same_window_top_pressure_pairs.csv", aggregate_top_frame
    )
    write_csv(output / "same_window_frame_pair_topk.csv", frame_top_frame)
    write_csv(
        output / "primary_window_trajectory_pairs.csv", trajectory_frame
    )

    matched_summary = matched_powder_interval_summary(source)
    write_csv(
        output / "powder_primary_spots_fit_interval_summary.csv",
        matched_summary,
    )
    powder_trajectory, powder_primary, powder_starts, powder_membership = state[
        "powder_spots"
    ]
    del powder_trajectory
    matched_path = output / "powder_spots_fit_interval_evidence_waterfall.png"
    plot_powder_spots_fit_interval_evidence(
        matched_path,
        matched_summary,
        powder_primary,
        powder_starts,
        powder_membership,
        dpi,
    )
    generated.append(matched_path)
    tables = {
        "interval": interval_summary,
        "membership": membership_frame,
        "same_window_aggregate_top": aggregate_top_frame,
        "same_window_frame_top": frame_top_frame,
        "trajectory_pairs": trajectory_frame,
        "matched_powder": matched_summary,
    }
    return tables, generated


def write_readme(
    output: Path,
    source: Path,
    whole_top: pd.DataFrame,
    medoids: pd.DataFrame,
    single_fingerprint: pd.DataFrame,
    powder_fingerprint: pd.DataFrame,
    peak_index: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    window_tables: dict[str, pd.DataFrame],
    detail_scan: str,
    scan_panel_count: int,
) -> None:
    supported = peak_index["location_support_tracks"].gt(0)
    powder_supported = peak_index[
        peak_index["dataset"].astype(str).eq("powder")
    ]["location_support_tracks"].gt(0)
    text = f"""# UOTe relationship-first waterfall output

This result set reorganizes the verified v3 correlation outputs around two
questions:

1. **Which frames/windows are most correlated?**
2. **Which peak evidence is actually available for a map cell?**

It does **not** infer why a peak or correlation changed.

Source: `{source}`

## Reading order

### 1. `01_whole_pattern_frame_topk/`

- Every whole-pattern map is shown as a complete symmetric matrix.
- The adjacent Top-{int(whole_top["neighbor_rank"].max())} carpet directly lists the
  most correlated partner for every frame.
- The lower panel shows the actual standardized whole-pattern feature used by
  Pearson, colored by similarity to the within-scan medoid.
- Powder rankings are always within one scan; no cross-scan frame ranking is
  invented.

Key tables:

- `frame_topk_relationships.csv`: all query-frame Top-K relationships, reciprocal
  Top-K flag, pressure gap, and filename-derived protocol flag.
- `scan_medoids.csv`: one representative medoid per map/scan.

Generated matrix-assisted scan figures: **{scan_panel_count}**.
Medoids: **{len(medoids)}**.

### 2. `02_peak_frame_fingerprints/`

- A filled circle means a curated detected peak.
- Gray/× means **unknown / not determined**, not absence.
- Color is detected-center shift from that track's median.
- Circle size is an area rank within the same track (and within orientation for
  single crystal).
- The powder atlas keeps all 56 scans separate. Its deterministic readable
  detail scan is `{detail_scan}`.

Fingerprint states:

- single crystal: {int((single_fingerprint["state"] == "present").sum())} present,
  {int((single_fingerprint["state"] == "unknown").sum())} unknown;
- powder: {int((powder_fingerprint["state"] == "present").sum())} present,
  {int((powder_fingerprint["state"] == "unknown").sum())} unknown,
  {int((powder_fingerprint["state"] == "not_measured").sum())} not-measured grid
  cells;
- confirmed absences encoded: **0**.

### 3. `03_frame_pair_peak_evidence/`

Each selected whole-pattern map cell has:

- the two detected peak-center positions;
- per-track location similarity;
- per-track area similarity where comparable;
- the number of supporting tracks.

These scores are **associated, non-additive evidence**. They are not a
decomposition or percentage contribution to whole-pattern Pearson.

Complete map-cell index rows: **{len(peak_index)}**.
Cells with at least one location-evidence track: **{int(supported.sum())}**.
Powder supported cells: **{int(powder_supported.sum())}** out of
{int((peak_index["dataset"] == "powder").sum())}; most have only one track.
Selected readable high/low examples: **{len(selected_pairs)}**.

### 4. `04_window_relationship_waterfalls/`

The v3 window method is:

- fixed 5°-window **direct NCC** across pressure frames; and
- correlation between **ΔP-adjusted pressure-change trajectories** of different
  windows.

It is not ACF and not a within-frame ACF map.

Outputs include:

- primary non-overlapping window × pressure-interval waterfalls;
- Top-K pressure/frame pairs for every primary window;
- ranked window-pair trajectory correlations;
- a powder spots/fit/excess interval comparison;
- explicit window-to-peak-track membership.

Primary window–interval rows: **{len(window_tables["interval"])}**.
Frame-level same-window Top-K rows: **{len(window_tables["same_window_frame_top"])}**.

## Color and missing-data ledger

- signed Pearson / NCC / trajectory r: blue–white–red, fixed `-1…1`;
- bounded peak similarity: viridis, fixed `0…1`;
- signed change/shift: diverging scale centered at zero;
- gray: NaN/unknown/not measured, never numeric zero;
- `†` or triangle: a filename-derived acquisition-signature change was detected.

## Scientific limits retained from v3

- Whole-pattern raw Pearson remains QC-only.
- Single-crystal area similarity has no repeat calibration and is secondary.
- Powder area repeatability is track-calibrated but lacks a defensible
  per-observation area SE and remains secondary.
- The fit channel is descriptive control evidence, not a pure nuisance or a
  causal subtraction.
- The v3 source found zero promoted candidate intervals.
- No real time field exists; figures use frame/acquisition order and pressure.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def validation_check(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    detail: str,
    value: object = None,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "detail": detail,
            "value": json_ready(value),
        }
    )


def validate_outputs(
    output: Path,
    whole_top: pd.DataFrame,
    medoids: pd.DataFrame,
    single_fingerprint: pd.DataFrame,
    powder_fingerprint: pd.DataFrame,
    peak_index: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    window_tables: dict[str, pd.DataFrame],
    generated_pngs: Sequence[Path],
    expected_scan_panels: int,
    detail_scan: str,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    rank_counts = (
        whole_top.groupby(["map_key", "scan", "anchor_frame"])[
            "neighbor_rank"
        ].nunique()
    )
    expected_top_k = int(whole_top["neighbor_rank"].max())
    validation_check(
        checks,
        "whole_pattern_every_anchor_has_top_k",
        bool(len(rank_counts) and rank_counts.eq(expected_top_k).all()),
        "Every query frame must have a complete Top-K list.",
        {
            "anchors": len(rank_counts),
            "top_k": expected_top_k,
            "minimum_ranks": int(rank_counts.min()) if len(rank_counts) else 0,
        },
    )
    validation_check(
        checks,
        "medoid_count_matches_map_scans",
        len(medoids) == 114,
        "Expected 56+56+1+1 within-scan medoids for the full v3 source.",
        len(medoids),
    )
    scan_pngs = [
        path
        for path in generated_pngs
        if "matrix_topk_feature_waterfall" in path.name
    ]
    validation_check(
        checks,
        "matrix_assisted_scan_panel_count",
        len(scan_pngs) == expected_scan_panels,
        "One rendered matrix-assisted panel is required for every selected scan.",
        {"actual": len(scan_pngs), "expected": expected_scan_panels},
    )
    validation_check(
        checks,
        "single_fingerprint_grid_complete",
        len(single_fingerprint) == 75 * 12,
        "Single fingerprint must contain every track×global-axis state.",
        len(single_fingerprint),
    )
    validation_check(
        checks,
        "powder_fingerprint_grid_complete",
        len(powder_fingerprint) == 10 * 56 * 19,
        "Powder fingerprint must contain every track×scan×pressure grid cell.",
        len(powder_fingerprint),
    )
    confirmed_absence = int(
        single_fingerprint["absence_confirmed"].sum()
        + powder_fingerprint["absence_confirmed"].sum()
    )
    validation_check(
        checks,
        "fingerprints_do_not_invent_absence",
        confirmed_absence == 0
        and not single_fingerprint["state"].eq("absent").any()
        and not powder_fingerprint["state"].eq("absent").any(),
        "Unknown/not-measured states must never become absence.",
        confirmed_absence,
    )
    validation_check(
        checks,
        "powder_detail_scan_is_deterministic",
        detail_scan == "scan033",
        "Detail scan maximizes distinct detected tracks, then present cells.",
        detail_scan,
    )
    validation_check(
        checks,
        "peak_map_cell_index_complete",
        len(peak_index) == 2 * 9504 + 2 * 55,
        "Index must cover every whole-pattern map cell in all four raw maps.",
        len(peak_index),
    )
    validation_check(
        checks,
        "selected_peak_evidence_examples",
        len(selected_pairs) == 7
        and set(selected_pairs["map_key"])
        == {"single_0deg", "single_10deg", "powder_spots", "powder_fit"},
        "Deterministic high/low supported selections should yield seven figures.",
        selected_pairs[
            ["map_key", "scan", "frame_a", "frame_b", "selection_role"]
        ].to_dict("records"),
    )
    finite_location = peak_index["median_location_similarity"].dropna()
    finite_area = peak_index["median_area_similarity"].dropna()
    validation_check(
        checks,
        "peak_similarity_ranges",
        bool(
            finite_location.between(0.0, 1.0).all()
            and finite_area.between(0.0, 1.0).all()
        ),
        "Finite per-peak similarities must remain in [0, 1].",
        {
            "location_min": finite_location.min(),
            "location_max": finite_location.max(),
            "area_min": finite_area.min(),
            "area_max": finite_area.max(),
        },
    )
    primary_counts = (
        window_tables["membership"]
        .groupby("map_key")["window_index"]
        .nunique()
        .to_dict()
    )
    validation_check(
        checks,
        "primary_nonoverlap_window_counts",
        primary_counts
        == {
            "powder_fit": 5,
            "powder_spots": 5,
            "single_0deg": 3,
            "single_10deg": 3,
        },
        "Only the v3-designated primary non-overlapping windows are ranked.",
        primary_counts,
    )
    aggregate_top = window_tables["same_window_aggregate_top"]
    expected_window_top_rows = (5 + 5 + 3 + 3) * expected_top_k
    validation_check(
        checks,
        "same_window_aggregate_topk_complete",
        len(aggregate_top) == expected_window_top_rows,
        "Every primary window must have a complete aggregate Top-K list.",
        {
            "actual": len(aggregate_top),
            "expected": expected_window_top_rows,
        },
    )
    finite_ncc = aggregate_top["aligned_ncc"].dropna()
    validation_check(
        checks,
        "same_window_ncc_signed_range",
        bool(finite_ncc.between(-1.0, 1.0).all()),
        "Direct NCC must remain signed and inside [-1, 1].",
        {"minimum": finite_ncc.min(), "maximum": finite_ncc.max()},
    )
    example = aggregate_top[
        aggregate_top["map_key"].eq("powder_spots")
        & aggregate_top["window_index"].eq(2)
        & aggregate_top["rank"].eq(1)
    ]
    example_pass = bool(
        len(example) == 1
        and math.isclose(
            float(example.iloc[0]["pressure_low_GPa"]), 41.4, abs_tol=1e-8
        )
        and math.isclose(
            float(example.iloc[0]["pressure_high_GPa"]), 50.7, abs_tol=1e-8
        )
        and math.isclose(
            float(example.iloc[0]["aligned_ncc"]),
            0.986390,
            abs_tol=2e-6,
        )
    )
    validation_check(
        checks,
        "same_window_known_example",
        example_pass,
        "Powder-spots primary window 2 Top-1 must match the verified source.",
        example.to_dict("records"),
    )
    image_errors: list[str] = []
    dimensions: list[tuple[int, int]] = []
    for path in generated_pngs:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions.append(tuple(image.size))
            if path.stat().st_size <= 1000:
                image_errors.append(f"too-small:{path}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            image_errors.append(f"{path}:{exc}")
    validation_check(
        checks,
        "all_pngs_open_and_nontrivial",
        not image_errors
        and len(dimensions) == len(generated_pngs)
        and all(width >= 700 and height >= 450 for width, height in dimensions),
        "Every generated PNG must open and have nontrivial dimensions/bytes.",
        {
            "pngs": len(generated_pngs),
            "errors": image_errors[:10],
            "minimum_width": min((item[0] for item in dimensions), default=0),
            "minimum_height": min((item[1] for item in dimensions), default=0),
        },
    )
    failed = [row for row in checks if not bool(row["passed"])]
    report = {
        "status": "PASS" if not failed else "FAIL",
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "failed_checks": [row["check"] for row in failed],
        "details": checks,
    }
    write_json(output / "validation_report.json", report)
    return report


def build_artifact_index(output: Path) -> pd.DataFrame:
    descriptions = {
        "01_whole_pattern_frame_topk": (
            "whole-pattern matrix, frame Top-K, and compared-feature waterfall"
        ),
        "02_peak_frame_fingerprints": (
            "peak-track detection/center/area fingerprint"
        ),
        "03_frame_pair_peak_evidence": (
            "selected map-cell per-peak associated evidence"
        ),
        "04_window_relationship_waterfalls": (
            "direct-NCC and pressure-change trajectory relationships"
        ),
    }
    rows: list[dict[str, object]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact_index.csv":
            continue
        relative = path.relative_to(output)
        category = relative.parts[0] if len(relative.parts) > 1 else "root"
        width = height = np.nan
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                width, height = image.size
        rows.append(
            {
                "relative_path": str(relative),
                "category": category,
                "description": descriptions.get(category, "run documentation"),
                "extension": path.suffix.lower(),
                "bytes": path.stat().st_size,
                "width_px": width,
                "height_px": height,
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    write_csv(output / "artifact_index.csv", frame)
    return frame


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = args.source_dir.resolve()
    output = args.out_dir.resolve()
    baseline = args.baseline_root.resolve()
    handoff = args.handoff_root.resolve()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    for required in (
        source / "run_manifest.json",
        source / "method_config.json",
        source / "validation/validation_report.json",
        baseline / "inputs/single_whole_selected.csv",
        handoff / "manifest.csv",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    source_manifest_before = sha256_file(source / "run_manifest.json")
    if (
        output.exists()
        and (output / "run_manifest.json").is_file()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Completed output already exists; use --overwrite: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    series_features = load_series_features(baseline, handoff)
    whole_root = output / "01_whole_pattern_frame_topk"
    whole_top, medoids, whole_pngs = generate_whole_pattern_outputs(
        source,
        whole_root,
        series_features,
        args.top_k,
        args.dpi,
        args.max_powder_scans,
    )

    fingerprint_root = output / "02_peak_frame_fingerprints"
    (
        single_fingerprint,
        powder_fingerprint,
        fingerprint_pngs,
        detail_scan,
    ) = generate_peak_fingerprints(
        source, fingerprint_root, series_features, args.dpi
    )

    evidence_root = output / "03_frame_pair_peak_evidence"
    peak_index, selected_pairs, evidence_pngs = generate_peak_evidence_outputs(
        source, evidence_root, args.dpi
    )

    window_root = output / "04_window_relationship_waterfalls"
    window_tables, window_pngs = generate_window_outputs(
        source, window_root, args.top_k, args.dpi
    )
    generated_pngs = (
        whole_pngs + fingerprint_pngs + evidence_pngs + window_pngs
    )
    expected_scan_panels = (
        114
        if args.max_powder_scans is None
        else 2 + 2 * min(56, max(0, args.max_powder_scans))
    )
    write_readme(
        output,
        source,
        whole_top,
        medoids,
        single_fingerprint,
        powder_fingerprint,
        peak_index,
        selected_pairs,
        window_tables,
        detail_scan,
        expected_scan_panels,
    )
    validation = validate_outputs(
        output,
        whole_top,
        medoids,
        single_fingerprint,
        powder_fingerprint,
        peak_index,
        selected_pairs,
        window_tables,
        generated_pngs,
        expected_scan_panels,
        detail_scan,
    )
    source_manifest_after = sha256_file(source / "run_manifest.json")
    if source_manifest_before != source_manifest_after:
        raise RuntimeError("Source v3 run manifest changed during generation")
    artifact_index = build_artifact_index(output)
    completed = datetime.now(timezone.utc)
    manifest = {
        "profile": "uote-relationship-waterfalls-v1",
        "version": VERSION,
        "status": validation["status"],
        "created_by": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_dir": source,
        "source_run_manifest_sha256": source_manifest_after,
        "baseline_root": baseline,
        "handoff_root": handoff,
        "output_dir": output,
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "arguments": vars(args),
        "artifact_rows": len(artifact_index),
        "pngs": len(generated_pngs),
        "matrix_assisted_scan_panels": expected_scan_panels,
        "top_k_rows": len(whole_top),
        "medoids": len(medoids),
        "selected_peak_evidence_pairs": len(selected_pairs),
        "powder_detail_scan": detail_scan,
        "validation": validation,
        "semantic_guards": {
            "unknown_is_not_absence": True,
            "per_peak_evidence_is_not_additive_contribution": True,
            "window_method_is_direct_ncc_not_acf": True,
            "window_to_window_is_pressure_trajectory_correlation": True,
            "no_causal_interpretation": True,
        },
    }
    write_json(output / "run_manifest.json", manifest)
    if validation["status"] != "PASS":
        raise RuntimeError(
            f"Relationship-waterfall validation failed: "
            f"{validation['failed_checks']}"
        )
    print(
        json.dumps(
            json_ready(
                {
                    "output": output,
                    "status": validation["status"],
                    "pngs": len(generated_pngs),
                    "artifacts": len(artifact_index),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
