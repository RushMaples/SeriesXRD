#!/usr/bin/env python3
"""Build the final story package for the interpretable high-pressure XRD project.

This is intentionally a thin synthesis layer over `outputs/analysis_v2_20260701`.
The earlier correlation maps are treated as diagnostics; the final story is
driven by phase/refinement, EOS, and static-vs-moving peak evidence.
"""
from __future__ import annotations

import csv
import os
import shutil
import textwrap
from collections import Counter
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_SCRIPT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = _SCRIPT_ROOT
ANALYSIS = ROOT / "outputs" / "analysis_v2_20260701"
CORR_SUITE = ROOT / "outputs" / "correlation_suite_20260621_high_recall_scored_v2"
OUT = ROOT / "outputs" / "final_interpretable_xrd_20260703"
FIG = OUT / "figures"
TABLE = OUT / "tables"


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as h:
        return list(csv.DictReader(h))


def write_csv(path: Path, rows: list[dict[str, object]], cols: list[str]) -> None:
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def figure_path(name: str) -> Path:
    return FIG / name


def save_workflow() -> None:
    stages = [
        ("Raw DAC XRD", "2D TIFFs + integrated .xy\nCell_14 and Cell_29"),
        ("Feature Maps", "ROI area / peak position\nACF + NCC shift"),
        ("Quality Gate", "Retire raw-pattern\ncorrelation as final proof"),
        ("Physical Features", "indexed peaks, d(P),\nraw-2D spot tracks"),
        ("Final Interpretation", "phase evolution + EOS\nstatic artifact census"),
    ]
    colors = ["#4477aa", "#66aa88", "#ccbb44", "#aa6655", "#7755aa"]

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.set_axis_off()
    xs = np.linspace(0.08, 0.92, len(stages))
    y = 0.55
    for i, ((title, body), color) in enumerate(zip(stages, colors)):
        ax.annotate(
            "",
            xy=(xs[i] - 0.08, y),
            xytext=(xs[i - 1] + 0.08, y),
            arrowprops=dict(arrowstyle="->", lw=2.2, color="0.35"),
        ) if i else None
        box = plt.Rectangle((xs[i] - 0.08, y - 0.18), 0.16, 0.36,
                            fc=color, ec="0.15", lw=1.2, alpha=0.94)
        ax.add_patch(box)
        ax.text(xs[i], y + 0.065, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(xs[i], y - 0.07, body, ha="center", va="center",
                fontsize=8.5, color="white")
    ax.text(
        0.5,
        0.92,
        "Final project direction: use correlation maps as diagnostics, then make physically interpretable XRD maps",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.16,
        "Paper claim: physically constrained feature extraction recovers a consistent high-pressure UOTe story despite spotty texture.",
        ha="center",
        va="center",
        fontsize=11,
        color="0.2",
    )
    fig.tight_layout()
    fig.savefig(figure_path("01_workflow_schematic.png"), dpi=220)
    plt.close(fig)


def open_image(path: Path, target_width: int | None = None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if target_width is not None:
        scale = target_width / im.width
        im = im.resize((target_width, int(im.height * scale)), Image.LANCZOS)
    return im


def add_title(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 34) -> None:
    try:
        font = ImageFont.truetype("Arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(30, 30, 30), font=font)


def save_side_by_side(left: Path, right: Path, out: Path, title: str) -> None:
    w = 1200
    left_im = open_image(left, target_width=w)
    right_im = open_image(right, target_width=w)
    pad, title_h = 40, 70
    h = max(left_im.height, right_im.height)
    canvas = Image.new("RGB", (w * 2 + pad * 3, h + title_h + pad), "white")
    draw = ImageDraw.Draw(canvas)
    add_title(draw, (pad, 22), title, 32)
    canvas.paste(left_im, (pad, title_h))
    canvas.paste(right_im, (w + pad * 2, title_h))
    canvas.save(out, quality=95)


def save_waterfall_panel() -> None:
    save_side_by_side(
        ANALYSIS / "C_dspacing_eos" / "Cell_29_waterfall_reference_lines.png",
        ANALYSIS / "C_dspacing_eos" / "Cell_14_waterfall_reference_lines.png",
        figure_path("02_pressure_waterfall_indexed_panel.png"),
        "Pressure waterfalls with reference/index lines: sample peaks move; fixed references are diagnostic only",
    )


def save_refined_eos_panel() -> None:
    rows = read_csv(ANALYSIS / "G_merged_correlation" / "merged_refined_EOS_Pnmm.csv")
    p = np.array([float(r["P_GPa"]) for r in rows])
    a = np.array([float(r["a"]) for r in rows])
    b = np.array([float(r["b"]) for r in rows])
    c = np.array([float(r["c"]) for r in rows])
    v = np.array([float(r["V_A3"]) for r in rows])
    cells = np.array([r["cell"] for r in rows])
    markers = {"Cell_29": "o", "Cell_14": "s"}
    colors = {"Cell_29": "#4477aa", "Cell_14": "#cc6677"}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    ax = axes[0]
    for cell in ("Cell_29", "Cell_14"):
        m = cells == cell
        ax.plot(p[m], v[m], markers[cell], ms=7, color=colors[cell], label=cell)
    order = np.argsort(p)
    ax.plot(p[order], v[order], color="0.25", lw=1.6, alpha=0.75,
            label="pressure-sorted compression guide")
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Pnmm refined volume (A^3)")
    ax.set_title("Refined Pnmm EOS trend validates merging by physics")
    ax.grid(alpha=0.28)
    ax.legend()

    ax = axes[1]
    ax.plot(p, a, "-o", color="#4477aa", label="a")
    ax.plot(p, b, "-o", color="#66aa88", label="b")
    ax.plot(p, c, "-o", color="#aa6655", label="c")
    ax.axvspan(8.0, p.max() + 0.3, color="#ddcc77", alpha=0.22,
               label="a/b convergence regime")
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Lattice parameter (A)")
    ax.set_title("Lattice evolution: a and b converge at high pressure")
    ax.grid(alpha=0.28)
    ax.legend()

    fig.suptitle("Main physical evidence: refined structure and compression, not raw frame correlation",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_path("03_refined_eos_lattice_panel.png"), dpi=220)
    plt.close(fig)


def save_phase_summary() -> None:
    rows = read_csv(ANALYSIS / "G_merged_correlation" / "merged_refined_EOS_Pnmm.csv")
    p = np.array([float(r["P_GPa"]) for r in rows])
    a = np.array([float(r["a"]) for r in rows])
    b = np.array([float(r["b"]) for r in rows])
    delta_ab = np.abs(a - b)

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.set_xlim(0.5, 13.2)
    ax.set_ylim(0, 3)
    ax.set_yticks([])
    bands = [
        (0.7, 6.0, 2.1, "#88ccee", "Pnmm + R3c + cubic refined model"),
        (6.0, 13.0, 1.25, "#ddcc77", "Im-3m/Fm-3m appears by ~6 GPa"),
        (8.5, 13.0, 0.42, "#cc6677", "a/b convergence: orthorhombic distortion collapses"),
    ]
    for lo, hi, y, color, label in bands:
        ax.add_patch(plt.Rectangle((lo, y - 0.24), hi - lo, 0.48,
                                   fc=color, ec="0.3", lw=0.8, alpha=0.72))
        ax.text((lo + hi) / 2, y, label, ha="center", va="center",
                fontsize=11, fontweight="bold")
    ax.plot(p, 2.72 - 2.0 * delta_ab / max(delta_ab), "o-", color="0.2", lw=1.2,
            label="scaled |a-b|")
    ax.text(0.65, 2.85, "|a-b| decreases with pressure", fontsize=9, color="0.2")
    for x in p:
        ax.axvline(x, color="0.86", lw=0.6, zorder=0)
    ax.set_xlabel("Pressure (GPa)")
    ax.set_title("Phase/evolution summary to guide the final paper narrative",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_path("04_phase_evolution_summary.png"), dpi=220)
    plt.close(fig)


def save_static_evidence() -> None:
    rows = read_csv(ANALYSIS / "F_raw2d_spot_tracking" / "spot_static_verdicts.csv")
    counts = {cell: Counter(r["verdict"] for r in rows if r["cell"] == cell)
              for cell in ("Cell_29", "Cell_14")}
    static = [r for r in rows if r["verdict"] == "STATIC"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    cats = ["MOVING", "STATIC", "UNDETERMINED"]
    x = np.arange(len(cats))
    width = 0.36
    for i, cell in enumerate(("Cell_29", "Cell_14")):
        vals = [counts[cell].get(c, 0) for c in cats]
        ax.bar(x + (i - 0.5) * width, vals, width, label=cell,
               color=["#66aa88", "#4477aa"][i])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20)
    ax.set_yscale("symlog", linthresh=2)
    ax.set_ylabel("Raw-2D spot tracks (symlog count)")
    ax.set_title("Static peak determination must use raw-2D spot tracking")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    if static:
        r = static[0]
        txt = (
            "Confirmed static spot\n"
            f"Cell_29: 2theta={float(r['median_2theta']):.3f} deg\n"
            f"d={float(r['d_A']):.4f} A, chi={float(r['chi']):.1f} deg\n"
            f"frames={r['n_frames']}, dP={r['dP_GPa']} GPa\n"
            f"slope={float(r['slope']):+.4f} deg/GPa"
        )
        ax.text(0.03, 0.95, txt, transform=ax.transAxes, va="top",
                bbox=dict(fc="white", ec="0.65", boxstyle="round,pad=0.45"), fontsize=9)

    ax = axes[1]
    img_path = ANALYSIS / "F_raw2d_spot_tracking" / "Cell_29_spot_2theta_vs_pressure.png"
    ax.imshow(Image.open(img_path))
    ax.axis("off")
    ax.set_title("Cell_29 evidence: one flat track, many compressing tracks")
    fig.suptitle("Static vs moving peak evidence", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_path("05_static_vs_moving_spot_evidence.png"), dpi=220)
    plt.close(fig)


def save_correlation_diagnostic() -> None:
    summary = read_csv(ANALYSIS / "D_percell_correlation" / "percell_correlation_summary.csv")
    merged = read_csv(ANALYSIS / "G_merged_correlation" / "merged_correlation_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1, 1.2]})

    ax = axes[0]
    cells = [r["cell"] for r in summary]
    slopes = [float(r["corr_vs_dP_slope"]) for r in summary]
    rs = [float(r["corr_vs_dP_r"]) for r in summary]
    ax.bar(cells, slopes, color=["#4477aa", "#cc6677"])
    for i, r in enumerate(rs):
        ax.text(i, slopes[i], f"r={r:.2f}", ha="center",
                va="top" if slopes[i] < 0 else "bottom", fontsize=10)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_ylabel("Whole-pattern Pearson slope vs |dP|")
    ax.set_title("Useful diagnostic: per-cell correlation decays with pressure gap")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    img_path = ANALYSIS / "D_percell_correlation" / "corr_vs_pressure_gap.png"
    ax.imshow(Image.open(img_path))
    ax.axis("off")
    ax.set_title("Use this instead of pooled raw-pattern/correlation-suite heatmaps")

    fig.suptitle("Correlation maps are diagnostic, not final phase-transition evidence",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_path("06_correlation_diagnostic_appendix.png"), dpi=220)
    plt.close(fig)

    decision_rows = [
        {
            "evidence": "per-peak ROI-area / peak-position maps",
            "status": "diagnostic only",
            "reason": "spotty texture and over-detected peak groups make ROI intensity non-reproducible",
            "manuscript_role": "appendix / QC, not phase-transition proof",
        },
        {
            "evidence": "same-window and within-frame ACF/NCC maps",
            "status": "diagnostic only",
            "reason": "ACF is translation-invariant and can score unrelated peak windows too high",
            "manuscript_role": "method limitation / supporting diagnostic",
        },
        {
            "evidence": "per-cell whole-pattern Pearson",
            "status": "supporting diagnostic",
            "reason": "decays with pressure gap when cells are not pooled",
            "manuscript_role": "validates pressure-series behavior within each cell",
        },
        {
            "evidence": "refined Pnmm EOS and lattice parameters",
            "status": "main evidence",
            "reason": "Cell_14 and Cell_29 follow one compression trend",
            "manuscript_role": "central result figure",
        },
        {
            "evidence": "raw-2D spot centroid tracking",
            "status": "main evidence",
            "reason": "separates fixed non-sample spots from compressing sample spots",
            "manuscript_role": "static artifact census and data-quality claim",
        },
    ]
    write_csv(
        TABLE / "evidence_role_decision_table.csv",
        decision_rows,
        ["evidence", "status", "reason", "manuscript_role"],
    )

    with (TABLE / "correlation_summary_compact.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["source", "group", "n_pairs", "mean_or_slope", "extra"])
        for r in summary:
            w.writerow(["per-cell Pearson", r["cell"], r["n_pairs"],
                        r["corr_vs_dP_slope"], f"r={r['corr_vs_dP_r']}"])
        for r in merged:
            w.writerow(["merged position Jaccard", r["group"], r["n_pairs"],
                        r["mean_jaccard"], f"slope={r['slope_vs_dP']}"])


def copy_curated_existing_figures() -> None:
    curated = {
        "source_merged_refined_EOS.png": ANALYSIS / "G_merged_correlation" / "merged_refined_EOS.png",
        "source_merged_position_jaccard.png": ANALYSIS / "G_merged_correlation" / "merged_position_jaccard.png",
        "source_merged_d_vs_P_bothcells.png": ANALYSIS / "G_merged_correlation" / "merged_d_vs_P_bothcells.png",
        "source_Cell_29_d_vs_pressure.png": ANALYSIS / "C_dspacing_eos" / "Cell_29_d_vs_pressure.png",
        "source_Cell_14_d_vs_pressure.png": ANALYSIS / "C_dspacing_eos" / "Cell_14_d_vs_pressure.png",
        "source_Cell_29_window_9_14_NCC.png": ANALYSIS / "H_window_metrics" / "Cell_29_window_9_14_NCC.png",
    }
    for name, src in curated.items():
        if src.exists():
            shutil.copy2(src, FIG / name)


def write_report_files() -> None:
    eos_rows = read_csv(ANALYSIS / "G_merged_correlation" / "merged_refined_EOS_Pnmm.csv")
    static_rows = read_csv(ANALYSIS / "F_raw2d_spot_tracking" / "spot_static_verdicts.csv")
    static_counts = Counter(r["verdict"] for r in static_rows if r["cell"] == "Cell_29")
    cell14_static = Counter(r["verdict"] for r in static_rows if r["cell"] == "Cell_14")
    p_min = min(float(r["P_GPa"]) for r in eos_rows)
    p_max = max(float(r["P_GPa"]) for r in eos_rows)
    v0 = float(eos_rows[0]["V_A3"])
    vlast = float(eos_rows[-1]["V_A3"])
    v_drop = 100.0 * (v0 - vlast) / v0

    readme = f"""# Interpretable High-Pressure XRD Feature Mapping

This folder is the final synthesis layer for the UOTe DAC XRD project. It
implements the decision to use correlation maps as diagnostics and to make the
paper/report about physically interpretable feature extraction: phase/refinement,
EOS, peak motion, and static artifact separation.

## Central Claim

Although simple frame-to-frame XRD correlation maps are unstable for spotty,
textured DAC data, physically constrained feature extraction reveals a consistent
high-pressure structural evolution of UOTe across two cells, including
compressing sample peaks, pressure-induced phase changes, and a small number of
static non-sample artifacts.

## Figure Set

1. `figures/01_workflow_schematic.png` - final analysis logic.
2. `figures/02_pressure_waterfall_indexed_panel.png` - pressure waterfalls with reference/index lines.
3. `figures/03_refined_eos_lattice_panel.png` - main EOS/lattice evidence.
4. `figures/04_phase_evolution_summary.png` - phase-transition narrative guide.
5. `figures/05_static_vs_moving_spot_evidence.png` - raw-2D static-vs-moving proof.
6. `figures/06_correlation_diagnostic_appendix.png` - correlation maps demoted to diagnostic evidence.

## Headline Results

- Refined Pnmm volume spans {p_min:.1f}-{p_max:.1f} GPa and decreases from
  {v0:.3f} to {vlast:.3f} A^3 ({v_drop:.1f}% drop).
- Cell_14 and Cell_29 merge by refined structural/EOS trend, not by raw-pattern
  correlation.
- Cell_29 raw-2D spot tracking: {static_counts.get('STATIC', 0)} STATIC,
  {static_counts.get('MOVING', 0)} MOVING, {static_counts.get('UNDETERMINED', 0)}
  UNDETERMINED tracks.
- Cell_14 static-spot determination remains underpowered:
  {cell14_static.get('STATIC', 0)} STATIC, {cell14_static.get('MOVING', 0)} MOVING,
  {cell14_static.get('UNDETERMINED', 0)} UNDETERMINED tracks.

## How To Regenerate

Run from the repository root:

```bash
python3 scripts/final_interpretable_xrd_package.py
```

The script only reads `outputs/analysis_v2_20260701` and writes this final
package. It does not replace the upstream analysis.
"""

    outline = """# Paper / Report Outline

## Working Title

Interpretable feature mapping of high-pressure UOTe diffraction under DAC
conditions.

## Abstract Logic

1. High-pressure DAC XRD is spotty/textured, so direct frame correlation can be
   misleading.
2. We compute correlation and autocorrelation maps as exploratory diagnostics.
3. We then constrain the interpretation with physically meaningful observables:
   indexed/refined phases, d-spacing trajectories, EOS/lattice parameters, and
   raw-2D spot centroid tracking.
4. This separates compressing sample peaks from static non-sample artifacts and
   recovers a consistent pressure evolution across Cell_14 and Cell_29.

## Recommended Figure Order

1. Workflow schematic.
2. Pressure waterfall with indexed/reference lines.
3. Refined lattice parameters and volume vs pressure.
4. Phase/evolution summary.
5. Static vs moving raw-2D spot tracking.
6. Correlation-map limitations and diagnostic role.

## Interpretation Rules

- Do not claim phase transitions from ACF/NCC correlation maps alone.
- Do not merge cells by raw-pattern similarity; merge them by refined structural
  trend and EOS consistency.
- Treat 1D integrated peaks as sample compression/EOS evidence.
- Treat raw-2D spot centroid tracking as the authoritative static-peak test.
- Mark Cell_14 static-spot conclusions as underpowered unless additional static
  exposures become available.
"""

    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "paper_outline.md").write_text(outline, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    save_workflow()
    save_waterfall_panel()
    save_refined_eos_panel()
    save_phase_summary()
    save_static_evidence()
    save_correlation_diagnostic()
    copy_curated_existing_figures()
    write_report_files()
    print(f"Wrote final story package to {OUT}")
    print(f"Figures: {FIG}")
    print(f"Tables:  {TABLE}")


if __name__ == "__main__":
    main()
