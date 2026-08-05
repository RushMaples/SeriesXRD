#!/usr/bin/env python3
"""Render one denoised powder formal-composite waterfall without correlation shading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import generate_denoised_peak_correlation_waterfall as shaded
import nonlinear_intensity_preprocessing as nonlinear


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_ROOT = (
    ROOT
    / "correlations/results/"
    "uote_nonlinear_squared_preprocessed_comparison_20260802"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_COMPARISON_ROOT
    / "waterfall_unshaded_preview_20260803"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT
    )
    parser.add_argument(
        "--mode", choices=("log_squared", "exp_squared"), default="log_squared"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = args.comparison_root / "_sources" / args.mode / "powder_roi"
    provenance = json.loads(
        (source_root / "intensity_transform_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    spec = nonlinear.ROITransformSpec(**provenance["transform"])
    peaks = shaded.load_peaks(source_root / "pressure_peak_grid.csv")
    reconstructed = shaded.reconstruct_formal_pressure_profiles(
        profile_audit_csv=(
            source_root / "observation_spots_absolute_profile_audit.csv"
        ),
        point_registry_csv=source_root / "point_registry.csv",
        peaks=peaks,
        spec=spec,
    )

    traces = list(reconstructed.traces)
    peak_counts: dict[float, int] = {}
    for peak in peaks:
        key = shaded.pressure_key(peak.pressure_gpa)
        peak_counts[key] = peak_counts.get(key, 0) + 1

    fig, ax = plt.subplots(figsize=(17.5, 13.2), constrained_layout=False)
    baselines: list[float] = []
    labels: list[str] = []
    for row_index, trace in enumerate(traces):
        baseline = float(len(traces) - 1 - row_index) * shaded.ROW_SPACING
        baselines.append(baseline)
        count = peak_counts[shaded.pressure_key(trace.pressure_gpa)]
        labels.append(f"{trace.pressure_gpa:g} GPa  ·  {count} peaks")
        ax.axhline(baseline, color="#dddddd", linewidth=0.55, zorder=0)
        ax.plot(
            trace.x,
            baseline + shaded.TRACE_HEIGHT * trace.displayed,
            color="#333333",
            linewidth=0.95,
            zorder=2,
        )

    grid_left = float(min(trace.x[0] for trace in traces))
    grid_right = float(max(trace.x[-1] for trace in traces))
    ax.set_xlim(grid_left - 0.18, grid_right + 0.28)
    ax.set_ylim(-0.18, baselines[-1] + (len(traces) - 1) + shaded.TRACE_HEIGHT + 0.12)
    ax.set_yticks(baselines, labels, fontsize=8.8)
    ax.set_xlabel(r"$2\theta$ (degrees)", fontsize=12)
    ax.set_ylabel(
        "Pressure rows (descending); fixed offsets prevent trace overlap",
        fontsize=11,
    )
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    mode_label = "Log-squared" if args.mode == "log_squared" else "Exp-squared"
    fig.suptitle(
        f"Powder {mode_label} formal-composite waterfall — no correlation shading",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    ax.set_title(
        "One reconstructed transformed trace per pressure; shared amplitude scale",
        fontsize=11,
        pad=12,
    )
    fig.text(
        0.12,
        0.012,
        (
            "Each trace is the same pressure-level composite used by the formal ROI "
            "calculation: observation components are summed within a frame, averaged "
            "across distinct frames per peak, and the 12–22 formal peak profiles are "
            "then summed at each pressure. Correlation colors, ribbons, anchor support, "
            "and reference highlighting are intentionally omitted."
        ),
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#555555",
        wrap=True,
    )
    fig.subplots_adjust(left=0.17, right=0.97, top=0.925, bottom=0.075)

    output_dir = args.out_dir / "powder" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / f"powder_{args.mode}_formal_composite_unshaded.png"
    fig.savefig(output_png, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    validation = {
        "status": "PASS",
        "sample": "powder",
        "mode": args.mode,
        "trace_source": "formal_composite",
        "pressure_rows": len(traces),
        "registered_pressure_level_peaks": len(peaks),
        "correlation_shading": False,
        "correlation_ribbons": False,
        "anchor_highlighting": False,
        "shared_display_scale": reconstructed.audit["shared_display_scale"],
        "trace_height": shaded.TRACE_HEIGHT,
        "row_spacing": shaded.ROW_SPACING,
        "strictly_nonoverlapping_trace_bands": (
            shaded.TRACE_HEIGHT < shaded.ROW_SPACING
        ),
        "formal_profile_reconstruction": dict(reconstructed.audit),
        "output_png": str(output_png.resolve()),
    }
    (output_dir / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
