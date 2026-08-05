#!/usr/bin/env python3
"""Evaluate whether the processed-handoff whole-pattern result is robust.

The primary unit is a scan position, not an individual frame pair. The script
also tests shuffled pressure labels, acquisition-order confounding, and whether
the spots-channel trend remains after controlling for the fit-channel signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260713
PAIR_KEY = ["scan", "frame_a", "frame_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--bootstraps", type=int, default=20000)
    return parser.parse_args()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return np.nan
    x = x[keep]
    y = y[keep]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    control = np.asarray(control, dtype=float)
    design = np.column_stack([np.ones(len(control)), control])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def partial_corr(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    control = np.asarray(control, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if np.count_nonzero(keep) < 4:
        return np.nan
    return pearson(residualize(x[keep], control[keep]), residualize(y[keep], control[keep]))


def bootstrap_ci(values: np.ndarray, count: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    draws = rng.choice(values, size=(count, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return tuple(float(value) for value in np.quantile(medians, [0.025, 0.975]))


def scan_level_metrics(frame: pd.DataFrame, channel: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scan, group in frame.groupby("scan", sort=True):
        gaps = group["pressure_gap_GPa"].to_numpy(float)
        scores = group["correlation"].to_numpy(float)
        near = scores[gaps <= 1.5]
        far = scores[gaps >= 15.0]
        rows.append(
            {
                "channel": channel,
                "scan": scan,
                "pair_count": len(group),
                "r_score_vs_pressure_gap": pearson(gaps, scores),
                "near_median": float(np.median(near)) if len(near) else np.nan,
                "far_median": float(np.median(far)) if len(far) else np.nan,
                "near_minus_far": (
                    float(np.median(near) - np.median(far)) if len(near) and len(far) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def prepare_permutation_scans(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    prepared = []
    for _, group in frame.groupby("scan", sort=True):
        frame_ids = sorted(set(group["frame_a"]).union(group["frame_b"]))
        index = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
        pressure_by_frame: dict[int, float] = {}
        for row in group.itertuples(index=False):
            pressure_by_frame[int(row.frame_a)] = float(row.pressure_a_GPa)
            pressure_by_frame[int(row.frame_b)] = float(row.pressure_b_GPa)
        pressure = np.asarray([pressure_by_frame[int(frame_id)] for frame_id in frame_ids], dtype=float)
        a_index = group["frame_a"].map(index).to_numpy(int)
        b_index = group["frame_b"].map(index).to_numpy(int)
        scores = group["correlation"].to_numpy(float)
        prepared.append((pressure, a_index, b_index, scores))
    return prepared


def permutation_null(
    frame: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    prepared = prepare_permutation_scans(frame)
    null = np.empty(count, dtype=float)
    for draw in range(count):
        scan_r = []
        for pressure, a_index, b_index, scores in prepared:
            shuffled = rng.permutation(pressure)
            gaps = np.abs(shuffled[a_index] - shuffled[b_index])
            scan_r.append(pearson(gaps, scores))
        null[draw] = np.nanmedian(scan_r)
    return null


def add_order_gap(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("scan", sort=True):
        frame_ids = sorted(set(group["frame_a"]).union(group["frame_b"]))
        order = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
        group = group.copy()
        group["order_gap"] = np.abs(
            group["frame_a"].map(order).to_numpy(int) - group["frame_b"].map(order).to_numpy(int)
        )
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def channel_summary(
    frame: pd.DataFrame,
    scan_metrics: pd.DataFrame,
    null: np.ndarray,
    bootstrap_count: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    scan_r = scan_metrics["r_score_vs_pressure_gap"].to_numpy(float)
    observed = float(np.nanmedian(scan_r))
    ci_low, ci_high = bootstrap_ci(scan_r, bootstrap_count, rng)
    valid = scan_r[np.isfinite(scan_r)]
    differences = scan_metrics["near_minus_far"].to_numpy(float)
    valid_differences = differences[np.isfinite(differences)]
    return {
        "pair_pooled_r": pearson(frame["pressure_gap_GPa"], frame["correlation"]),
        "scan_median_r": observed,
        "scan_mean_r": float(np.nanmean(scan_r)),
        "scan_r_q1": float(np.nanquantile(scan_r, 0.25)),
        "scan_r_q3": float(np.nanquantile(scan_r, 0.75)),
        "scan_median_r_bootstrap_ci95": [ci_low, ci_high],
        "negative_scans": int(np.count_nonzero(valid < 0)),
        "evaluable_scans": int(len(valid)),
        "negative_scan_fraction": float(np.mean(valid < 0)),
        "near_above_far_scans": int(np.count_nonzero(valid_differences > 0)),
        "near_far_evaluable_scans": int(len(valid_differences)),
        "near_above_far_fraction": float(np.mean(valid_differences > 0)),
        "permutation_null_median": float(np.median(null)),
        "permutation_null_ci95": [float(value) for value in np.quantile(null, [0.025, 0.975])],
        "permutation_p_one_sided": float((1 + np.count_nonzero(null <= observed)) / (len(null) + 1)),
    }


def pressure_bin_summary(frame: pd.DataFrame, channel: str) -> pd.DataFrame:
    labels = ["0-1.5", "1.5-5", "5-15", "15+"]
    frame = frame.copy()
    frame["gap_bin"] = pd.cut(
        frame["pressure_gap_GPa"],
        [-np.inf, 1.5, 5.0, 15.0, np.inf],
        labels=labels,
        include_lowest=True,
    )
    per_scan = (
        frame.groupby(["scan", "gap_bin"], observed=True)["correlation"]
        .median()
        .reset_index()
    )
    rows = []
    for label in labels:
        values = per_scan.loc[per_scan["gap_bin"] == label, "correlation"].to_numpy(float)
        rows.append(
            {
                "channel": channel,
                "pressure_gap_bin_GPa": label,
                "scan_count": len(values),
                "scan_median_correlation": float(np.median(values)),
                "scan_q1_correlation": float(np.quantile(values, 0.25)),
                "scan_q3_correlation": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def plot_scan_distributions(scan_metrics: pd.DataFrame, path: Path) -> None:
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = {"spots": "#2A6F97", "fit": "#B23A48"}
    for x_pos, channel in enumerate(["spots", "fit"], start=1):
        values = scan_metrics.loc[
            scan_metrics["channel"] == channel, "r_score_vs_pressure_gap"
        ].to_numpy(float)
        ax.boxplot(values, positions=[x_pos], widths=0.45, showfliers=False)
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(np.full(len(values), x_pos) + jitter, values, s=18, alpha=0.65, color=colors[channel])
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_xticks([1, 2], ["spots (sample channel)", "fit (W control)"])
    ax.set_ylabel("Per-scan r: pattern correlation vs |dP|")
    ax.set_title("Pressure-decay trend across independent scan positions")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_permutation(nulls: dict[str, np.ndarray], observed: dict[str, float], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    colors = {"spots": "#2A6F97", "fit": "#B23A48"}
    for ax, channel in zip(axes, ["spots", "fit"]):
        ax.hist(nulls[channel], bins=45, color=colors[channel], alpha=0.8)
        ax.axvline(observed[channel], color="black", linewidth=2, label=f"observed {observed[channel]:.3f}")
        ax.axvline(0, color="white", linewidth=1, alpha=0.9)
        ax.set_title(channel)
        ax.set_xlabel("Median per-scan r after pressure shuffle")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Permutation count")
    fig.suptitle("Pressure-label permutation test")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_control_relationship(merged: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    scatter = ax.scatter(
        merged["correlation_fit"],
        merged["correlation_spots"],
        c=merged["pressure_gap_GPa"],
        s=9,
        alpha=0.35,
        cmap="viridis",
        edgecolors="none",
    )
    fig.colorbar(scatter, ax=ax, label="Pressure gap (GPa)")
    ax.set_xlabel("fit-channel correlation (W control)")
    ax.set_ylabel("spots-channel correlation")
    ax.set_title("Sample-channel similarity vs control-channel similarity")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pressure_bins(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(4)
    for channel, color in [("spots", "#2A6F97"), ("fit", "#B23A48")]:
        group = summary[summary["channel"] == channel]
        median = group["scan_median_correlation"].to_numpy(float)
        q1 = group["scan_q1_correlation"].to_numpy(float)
        q3 = group["scan_q3_correlation"].to_numpy(float)
        ax.plot(x, median, marker="o", linewidth=2, color=color, label=channel)
        ax.fill_between(x, q1, q3, color=color, alpha=0.16)
    ax.set_xticks(x, ["0-1.5", "1.5-5", "5-15", "15+"])
    ax.set_xlabel("Pressure gap (GPa)")
    ax.set_ylabel("Median within-scan pattern correlation")
    ax.set_title("Correlation decay after scan-level aggregation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(
    path: Path,
    summary: dict[str, object],
    old_cell29: dict[str, float | int],
) -> None:
    spots = summary["spots"]
    fit = summary["fit"]
    control = summary["control_and_confounding"]
    text = f"""# UOTe processed-handoff robustness evaluation

## Bottom line

- The old Cell_29 value and the new spots-channel value are **not numerically interchangeable**. They use different experiments, wavelengths, radial ranges, preprocessing, masks/channel definitions, frame counts, and aggregation units.
- The old corrected-v2 pipeline was not literally unmasked: its documentation specifies detector module-gap, hot/dead-pixel, and non-positive-pixel masking. It did not, however, use this handoff's azimuthal `spots_channel`, tungsten-dominated `fit_channel`, or the same cover-frame exclusions.
- The new result is internally coherent as evidence that **processed pattern similarity decreases along the pressure/acquisition ladder**. It is not yet clean evidence that the whole-pattern effect is UOTe-specific.

## Scan-level robustness

Individual frame pairs are dependent, so the primary statistic here is the median r across independent scan positions.

| channel | pooled pair r | median scan r | bootstrap 95% CI | negative scans | near correlation > far correlation | pressure-shuffle p |
|---|---:|---:|---:|---:|---:|---:|
| spots | {spots['pair_pooled_r']:.3f} | {spots['scan_median_r']:.3f} | [{spots['scan_median_r_bootstrap_ci95'][0]:.3f}, {spots['scan_median_r_bootstrap_ci95'][1]:.3f}] | {spots['negative_scans']}/{spots['evaluable_scans']} | {spots['near_above_far_scans']}/{spots['near_far_evaluable_scans']} | {spots['permutation_p_one_sided']:.5f} |
| fit control | {fit['pair_pooled_r']:.3f} | {fit['scan_median_r']:.3f} | [{fit['scan_median_r_bootstrap_ci95'][0]:.3f}, {fit['scan_median_r_bootstrap_ci95'][1]:.3f}] | {fit['negative_scans']}/{fit['evaluable_scans']} | {fit['near_above_far_scans']}/{fit['near_far_evaluable_scans']} | {fit['permutation_p_one_sided']:.5f} |

The pressure-label permutation checks whether the observed ordering is stronger than arbitrary pressure assignments within each scan. It does not distinguish pressure physics from monotonic acquisition-time drift.

## Specificity and confounding

- Pressure gap versus retained-frame order gap: r={control['pressure_gap_vs_order_gap_r']:.3f}. Pressure and acquisition order are therefore strongly confounded.
- Spots-pair similarity versus fit-control similarity: r={control['spots_vs_fit_pair_score_r']:.3f}.
- Spots pressure-gap partial r after controlling for fit, pooled: {control['spots_gap_partial_r_given_fit_pooled']:.3f}.
- Median per-scan partial r after controlling for fit: {control['spots_gap_partial_r_given_fit_scan_median']:.3f}, bootstrap 95% CI [{control['spots_gap_partial_r_given_fit_scan_median_ci95'][0]:.3f}, {control['spots_gap_partial_r_given_fit_scan_median_ci95'][1]:.3f}], with {control['spots_gap_partial_negative_scans']}/{control['spots_gap_partial_evaluable_scans']} scans negative.

Because the tungsten-dominated control has a stronger decay than the sample channel, whole-pattern decay alone cannot identify a UOTe transition. The spots result can still be physically plausible: sparse sample spots and residual diamond features evolve with pressure, but they are mixed with shared acquisition/processing effects.

## Why the old and new r values are not an apples-to-apples replication

| property | old corrected Cell_29 | new processed handoff |
|---|---|---|
| data unit | one Cell_29 pressure series | 56 spatial scan positions |
| pair count | {old_cell29['n_pairs']} | {summary['pair_count_per_channel']} |
| wavelength | 0.4133 A | 0.3066 A |
| radial range | 4-22 deg 2theta | 2-32 deg 2theta |
| signal | integrated, baseline-subtracted positive pattern | azimuthally sparse spots channel; fit channel as W control |
| masking | detector bad/gap/non-positive mask | teammate processing plus 228 cover/beamstop exclusions |
| pooled r | {old_cell29['corr_vs_dP_r']:.3f} | {spots['pair_pooled_r']:.3f} spots |

The defensible comparison is only qualitative: both analyses show decreasing similarity along their respective pressure series. To compare magnitudes, the same raw frames must be processed both ways, restricted to the same q/2theta support, and analyzed with the same retained frames and scan-level statistic.

## Scientific verdict

1. **Does the new number make statistical sense? Yes.** The negative trend repeats across scan positions, survives scan-level aggregation, and is far outside the pressure-label permutation null.
2. **Does it make physical sense? Plausibly yes.** Peak motion and changing sparse-spot content should reduce pattern similarity as pressure separation grows.
3. **Does it prove the UOTe structure is responsible? No.** The stronger fit-channel control and pressure-order confounding prevent that claim.
4. **Does `-0.585` reproduce `-0.594`? No, not quantitatively.** Their similar values are coincidental-compatible, not a direct validation.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    out_dir = result_dir / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    frames: dict[str, pd.DataFrame] = {}
    scan_tables = []
    nulls: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, object]] = {}
    bin_tables = []
    for channel in ["spots", "fit"]:
        path = result_dir / channel / "whole_pattern" / "whole_pattern_pair_scores.csv"
        frame = pd.read_csv(path)
        frame = add_order_gap(frame)
        frames[channel] = frame
        scan_table = scan_level_metrics(frame, channel)
        scan_tables.append(scan_table)
        null = permutation_null(frame, args.permutations, rng)
        nulls[channel] = null
        summaries[channel] = channel_summary(frame, scan_table, null, args.bootstraps, rng)
        bin_tables.append(pressure_bin_summary(frame, channel))

    scan_metrics = pd.concat(scan_tables, ignore_index=True)
    pressure_bins = pd.concat(bin_tables, ignore_index=True)
    merged = frames["spots"].merge(
        frames["fit"][PAIR_KEY + ["correlation"]],
        on=PAIR_KEY,
        how="inner",
        validate="one_to_one",
        suffixes=("_spots", "_fit"),
    )

    partial_rows = []
    for scan, group in merged.groupby("scan", sort=True):
        partial_rows.append(
            {
                "scan": scan,
                "partial_r_spots_gap_given_fit": partial_corr(
                    group["pressure_gap_GPa"],
                    group["correlation_spots"],
                    group["correlation_fit"],
                ),
            }
        )
    partial_table = pd.DataFrame(partial_rows)
    partial_values = partial_table["partial_r_spots_gap_given_fit"].to_numpy(float)
    partial_valid = partial_values[np.isfinite(partial_values)]
    partial_ci = bootstrap_ci(partial_valid, args.bootstraps, rng)

    control = {
        "pressure_gap_vs_order_gap_r": pearson(
            frames["spots"]["pressure_gap_GPa"], frames["spots"]["order_gap"]
        ),
        "spots_vs_fit_pair_score_r": pearson(
            merged["correlation_spots"], merged["correlation_fit"]
        ),
        "spots_gap_partial_r_given_fit_pooled": partial_corr(
            merged["pressure_gap_GPa"], merged["correlation_spots"], merged["correlation_fit"]
        ),
        "spots_gap_partial_r_given_fit_scan_median": float(np.median(partial_valid)),
        "spots_gap_partial_r_given_fit_scan_median_ci95": list(partial_ci),
        "spots_gap_partial_negative_scans": int(np.count_nonzero(partial_valid < 0)),
        "spots_gap_partial_evaluable_scans": int(len(partial_valid)),
    }

    previous_path = result_dir.parents[2] / "outputs" / "analysis_v2_20260701" / "D_percell_correlation" / "percell_correlation_summary.csv"
    if not previous_path.is_file():
        previous_path = Path("outputs/analysis_v2_20260701/D_percell_correlation/percell_correlation_summary.csv").resolve()
    previous = pd.read_csv(previous_path)
    old_cell29_row = previous.loc[previous["cell"] == "Cell_29"].iloc[0]
    old_cell29 = {
        "n_pairs": int(old_cell29_row["n_pairs"]),
        "corr_vs_dP_r": float(old_cell29_row["corr_vs_dP_r"]),
    }

    summary: dict[str, object] = {
        "primary_unit": "scan position",
        "pair_count_per_channel": int(len(frames["spots"])),
        "scan_count": int(frames["spots"]["scan"].nunique()),
        "permutations": args.permutations,
        "bootstraps": args.bootstraps,
        "spots": summaries["spots"],
        "fit": summaries["fit"],
        "control_and_confounding": control,
        "old_cell29_reference": old_cell29,
    }

    scan_metrics = scan_metrics.merge(partial_table, on="scan", how="left")
    scan_metrics.to_csv(out_dir / "scan_level_metrics.csv", index=False)
    pressure_bins.to_csv(out_dir / "pressure_gap_bins_scan_aggregated.csv", index=False)
    pd.DataFrame(
        {
            "draw": np.arange(args.permutations),
            "spots_null_median_scan_r": nulls["spots"],
            "fit_null_median_scan_r": nulls["fit"],
        }
    ).to_csv(out_dir / "pressure_label_permutation_null.csv", index=False)
    (out_dir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    plot_scan_distributions(scan_metrics, out_dir / "scan_r_distributions.png")
    plot_permutation(
        nulls,
        {channel: float(summaries[channel]["scan_median_r"]) for channel in ["spots", "fit"]},
        out_dir / "pressure_label_permutation.png",
    )
    plot_control_relationship(merged, out_dir / "spots_vs_fit_control.png")
    plot_pressure_bins(pressure_bins, out_dir / "pressure_gap_bins.png")
    write_report(out_dir / "ROBUSTNESS_EVALUATION.md", summary, old_cell29)

    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
