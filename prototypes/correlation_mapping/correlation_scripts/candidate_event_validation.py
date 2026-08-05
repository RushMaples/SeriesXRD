#!/usr/bin/env python3
"""Candidate event validation for XDI-inspired pressure-window maps.

This script does not tune the peak detector and does not add feature-map modes.
It consumes the completed pressure-window mapping suite plus existing peak-group,
raw-2D, and reference/refinement outputs, then writes scientifically reviewable
candidate event packages.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRESSURE_RE = re.compile(r"(?P<value>\d+(?:p\d+)?|\d+(?:\.\d+)?)\s*G[PO]a", re.I)
DEFAULT_XDI_GLOB = "outputs/xdi_pressure_window_maps*"
DEFAULT_PEAK_SUITE = Path("outputs/correlation_suite_20260621_high_recall_scored_v2")
DEFAULT_ANALYSIS_V2 = Path("outputs/analysis_v2_20260701")
DEFAULT_PONI = Path("Data/Calibration/CeO2_30keV_168mm_0deg_001.poni")
EVENT_DIR_NAME = "07_candidate_event_validation"
REF_TOL_DEG = 0.12


CATALOG_COLUMNS = [
    "event_id",
    "event_type",
    "cell",
    "related_peak_group_ids",
    "related_window_start",
    "related_window_end",
    "median_2theta",
    "observed_d",
    "observed_q",
    "pressure_range",
    "compression_or_decompression",
    "frame_count",
    "strongest_frame",
    "max_roi_area",
    "max_prominence",
    "position_shift",
    "FWHM_change",
    "NCC_summary",
    "ACF_summary",
    "Tier A count",
    "Tier B count",
    "Tier C count",
    "static/background score if available",
    "initial_interpretation",
    "manual_review_priority",
    "reason_selected",
]


@dataclass
class Inputs:
    xdi_suite: Path
    peak_suite: Path
    peak_dir: Path
    analysis_v2: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdi-suite", type=Path, default=None)
    parser.add_argument("--peak-suite", type=Path, default=DEFAULT_PEAK_SUITE)
    parser.add_argument("--analysis-v2", type=Path, default=DEFAULT_ANALYSIS_V2)
    parser.add_argument("--poni", type=Path, default=DEFAULT_PONI)
    parser.add_argument("--max-events", type=int, default=18)
    return parser.parse_args()


def norm_path(path: str | Path) -> str:
    return Path(str(path)).as_posix()


def pressure_from_text(text: str | float | int | None) -> float | None:
    if text is None or (isinstance(text, float) and not np.isfinite(text)):
        return None
    match = PRESSURE_RE.search(str(text))
    if not match:
        return None
    return float(match.group("value").replace("p", "."))


def infer_cell(text: str | Path) -> str:
    value = str(text)
    if "Cell_14" in value:
        return "Cell_14"
    if "Cell_29" in value:
        return "Cell_29"
    return ""


def is_decomp_text(text: str | Path) -> bool:
    value = str(text).lower()
    return "decomp" in value or "none" in value or "no-ne" in value or "none" in value


def pressure_label(pressure: float | None, decomp: bool = False) -> str:
    if pressure is None or not np.isfinite(pressure):
        return ""
    label = f"{pressure:g}GPa"
    return f"{label}_decomp" if decomp else label


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def fmt(value, digits: int = 3, blank: str = "") -> str:
    try:
        value = float(value)
    except Exception:
        return blank
    if not np.isfinite(value):
        return blank
    return f"{value:.{digits}f}"


def q_from_d(d_a: float | np.ndarray) -> np.ndarray:
    d = np.asarray(d_a, dtype=float)
    return 2.0 * np.pi / d


def d_from_2theta(two_theta_deg, wavelength_a: float | None):
    if wavelength_a is None or not np.isfinite(wavelength_a):
        return np.full_like(np.asarray(two_theta_deg, dtype=float), np.nan, dtype=float)
    tt = np.deg2rad(np.asarray(two_theta_deg, dtype=float))
    return wavelength_a / (2.0 * np.sin(tt / 2.0))


def two_theta_from_d(d_a, wavelength_a: float) -> np.ndarray:
    d = np.asarray(d_a, dtype=float)
    arg = np.clip(wavelength_a / (2.0 * d), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arcsin(arg))


def add_d_q_columns(df: pd.DataFrame, theta_col: str, wavelength_a: float | None, prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    if theta_col not in out.columns:
        return out
    theta = pd.to_numeric(out[theta_col], errors="coerce").to_numpy()
    d = d_from_2theta(theta, wavelength_a)
    q = q_from_d(d)
    out[f"{prefix}d_A" if prefix else "d_A"] = d
    out[f"{prefix}q_invA" if prefix else "q_invA"] = q
    return out


def latest_xdi_suite(explicit: Path | None = None) -> Path:
    if explicit:
        return explicit
    suites = []
    for path in sorted(Path(".").glob(DEFAULT_XDI_GLOB)):
        if (path / "tables" / "pressure_window_feature_table.csv").exists():
            suites.append(path)
    if not suites:
        raise FileNotFoundError("No completed xdi_pressure_window_maps suite was found.")
    return max(suites, key=lambda p: p.stat().st_mtime)


def peak_dir_for_suite(peak_suite: Path) -> Path:
    peak_dir = peak_suite / "01_per_peak_frame_correlation"
    if peak_dir.exists():
        return peak_dir
    return peak_suite


def unique_output_dir(base: Path) -> Path:
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base.with_name(f"{base.name}_{stamp}")


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def run_git_status() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"git status unavailable: {exc}"


def read_wavelength_a(poni: Path) -> tuple[float | None, str]:
    if poni.exists():
        for line in poni.read_text(errors="ignore").splitlines():
            if line.lower().startswith("wavelength"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    wl_m = safe_float(parts[1])
                    if np.isfinite(wl_m):
                        return wl_m * 1e10, str(poni)
    for ref in sorted(Path("Data").rglob("*.ref")):
        for line in ref.read_text(errors="ignore").splitlines():
            if "Wavelength:" in line:
                match = re.search(r"Wavelength:\s*([0-9.]+)", line)
                if match:
                    wl = safe_float(match.group(1))
                    if np.isfinite(wl):
                        return wl, str(ref)
    return None, ""


def cubic_d(a: float, hkl: tuple[int, int, int]) -> float:
    h, k, l = hkl
    return a / math.sqrt(h * h + k * k + l * l)


def hcp_d(a: float, c: float, hkl: tuple[int, int, int]) -> float:
    h, k, l = hkl
    inv = (4.0 / 3.0) * (h * h + h * k + k * k) / (a * a) + (l * l) / (c * c)
    return 1.0 / math.sqrt(inv)


def known_reference_lines(wavelength_a: float | None) -> pd.DataFrame:
    if wavelength_a is None or not np.isfinite(wavelength_a):
        return pd.DataFrame(
            columns=["reference_phase", "reference_hkl", "reference_d_A", "reference_2theta", "fixed"]
        )
    rows = []
    for hkl in [(1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0), (3, 3, 1)]:
        d = cubic_d(3.567, hkl)
        rows.append(
            {
                "reference_phase": "diamond",
                "reference_hkl": str(hkl),
                "reference_d_A": d,
                "reference_2theta": float(two_theta_from_d(d, wavelength_a)),
                "fixed": True,
                "source": "local xrd_geometry constants",
            }
        )
    for hkl in [(1, 1, 1), (2, 0, 0), (2, 2, 0)]:
        d = cubic_d(4.43, hkl)
        rows.append(
            {
                "reference_phase": "Ne(ambient)",
                "reference_hkl": str(hkl),
                "reference_d_A": d,
                "reference_2theta": float(two_theta_from_d(d, wavelength_a)),
                "fixed": False,
                "source": "local xrd_geometry constants",
            }
        )
    for hkl in [(1, 0, 0), (0, 0, 2), (1, 0, 1), (1, 0, 2), (1, 1, 0)]:
        d = hcp_d(2.761, 4.456, hkl)
        rows.append(
            {
                "reference_phase": "Re-gasket(ambient)",
                "reference_hkl": str(hkl),
                "reference_d_A": d,
                "reference_2theta": float(two_theta_from_d(d, wavelength_a)),
                "fixed": False,
                "source": "local xrd_geometry constants",
            }
        )
    return pd.DataFrame(rows)


def reference_file_inventory() -> pd.DataFrame:
    roots = [Path("Data"), Path("outputs")]
    suffixes = {".cif", ".jcpds", ".gpx", ".lst", ".hkl", ".ref", ".prf", ".l70", ".m40", ".m41", ".m50", ".m70", ".res", ".ins"}
    keywords = ("refin", "diamond", "gasket", "background", "static", "jana", "gsas", "reflection")
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if path.suffix.lower() in suffixes or any(key in name for key in keywords):
                rows.append(
                    {
                        "path": norm_path(path),
                        "suffix": path.suffix.lower(),
                        "cell": infer_cell(path),
                        "pressure_gpa": pressure_from_text(path.as_posix()),
                        "size_bytes": path.stat().st_size,
                    }
                )
    return pd.DataFrame(rows).sort_values(["suffix", "path"], na_position="last")


def phase_names_from_ref(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(errors="ignore").splitlines()
    names = []
    for line in text:
        match = re.search(r"\*\s*Structure data -\s*(.*?)\s*\*", line)
        if match:
            name = match.group(1).strip()
            if name and name not in names:
                names.append(name)
    if names:
        return names
    for idx, line in enumerate(text[:-1]):
        if text[idx + 1].strip("=").strip() == "" and line.strip() and len(line.strip()) <= 30:
            name = line.strip()
            if name.isalpha() and name not in names:
                names.append(name)
    return names


def parse_refinement_reflections(wavelength_a: float | None) -> pd.DataFrame:
    rows = []
    for prf in sorted(Path("Data").rglob("*.prf")):
        ref = prf.with_suffix(".ref")
        phase_names = phase_names_from_ref(ref)
        cell = infer_cell(prf)
        pressure = pressure_from_text(prf.as_posix())
        for line in prf.read_text(errors="ignore").splitlines():
            tokens = line.split()
            if len(tokens) < 10:
                continue
            try:
                h, k, l = int(tokens[0]), int(tokens[1]), int(tokens[2])
                mult = float(tokens[3])
                phase_idx = int(float(tokens[4]))
                two_theta = float(tokens[5])
            except Exception:
                continue
            if not (2.0 <= two_theta <= 26.0) or phase_idx < 1:
                continue
            phase = phase_names[phase_idx - 1] if phase_idx <= len(phase_names) else f"phase_{phase_idx}"
            d_a = float(d_from_2theta(two_theta, wavelength_a)) if wavelength_a else np.nan
            rows.append(
                {
                    "cell": cell,
                    "pressure_gpa": pressure,
                    "source_prf": norm_path(prf),
                    "reference_phase": phase,
                    "reference_hkl": str((h, k, l)),
                    "multiplicity": mult,
                    "reference_2theta": two_theta,
                    "reference_d_A": d_a,
                    "reference_q_invA": float(q_from_d(d_a)) if np.isfinite(d_a) else np.nan,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.drop_duplicates(
        ["cell", "pressure_gpa", "reference_phase", "reference_hkl", "reference_2theta"]
    ).sort_values(["cell", "pressure_gpa", "reference_2theta"])


def prepare_peaks(peaks: pd.DataFrame) -> pd.DataFrame:
    if peaks.empty:
        return peaks
    out = peaks.copy()
    out["cell"] = out["path"].astype(str).map(infer_cell)
    out["pressure_gpa"] = out["frame"].map(pressure_from_text)
    missing = out["pressure_gpa"].isna()
    if missing.any():
        out.loc[missing, "pressure_gpa"] = out.loc[missing, "path"].map(pressure_from_text)
    out["decomp"] = out["frame"].astype(str).str.contains("decomp", case=False, na=False) | out["path"].astype(str).map(is_decomp_text)
    out["frame_label"] = [
        pressure_label(p, d) if p is not None else str(fr)
        for p, d, fr in zip(out["pressure_gpa"], out["decomp"], out["frame"])
    ]
    for column in ["two_theta", "prominence", "width_deg", "width_estimate_deg", "normalized_intensity", "raw_intensity", "peak_group"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["confidence_tier"] = out.get("confidence_tier", "").astype(str).str.upper()
    return out


def slope_and_r2(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    keep = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[keep], yy[keep]
    if len(xx) < 2 or np.ptp(xx) <= 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(xx, yy, 1)
    pred = slope * xx + intercept
    total = np.sum((yy - yy.mean()) ** 2)
    r2 = 1.0 - np.sum((yy - pred) ** 2) / total if total > 0 else np.nan
    return float(slope), float(r2)


def group_metrics(peaks: pd.DataFrame, group_summary: pd.DataFrame) -> pd.DataFrame:
    if peaks.empty or "peak_group" not in peaks.columns:
        return pd.DataFrame()
    rows = []
    valid = peaks.dropna(subset=["peak_group", "two_theta"]).copy()
    valid["peak_group"] = valid["peak_group"].astype(int)
    for gid, grp in valid.groupby("peak_group"):
        comp = grp[~grp["decomp"]].copy()
        strong = grp.sort_values(["prominence", "normalized_intensity"], ascending=False, na_position="last").head(1)
        if strong.empty:
            strong = grp.head(1)
        strong_row = strong.iloc[0]
        cell = strong_row.get("cell", "")
        compression_frame_count = int(comp["frame_label"].nunique()) if not comp.empty else 0
        decomp_frame_count = int(grp[grp["decomp"]]["frame_label"].nunique())
        slope, r2 = slope_and_r2(comp["pressure_gpa"], comp["two_theta"]) if not comp.empty else (np.nan, np.nan)
        fwhm_source = "width_deg" if "width_deg" in grp.columns else "width_estimate_deg"
        fwhm_values = pd.to_numeric(grp.get(fwhm_source, pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "peak_group": int(gid),
                "cell": cell,
                "group_two_theta": float(np.nanmedian(grp["two_theta"])),
                "median_2theta": float(np.nanmedian(grp["two_theta"])),
                "compression_frame_count": compression_frame_count,
                "decomp_frame_count": decomp_frame_count,
                "frame_count": int(grp["frame_label"].nunique()),
                "pressure_min": float(np.nanmin(comp["pressure_gpa"])) if not comp.empty else np.nan,
                "pressure_max": float(np.nanmax(comp["pressure_gpa"])) if not comp.empty else np.nan,
                "position_shift": float(np.nanmax(comp["two_theta"]) - np.nanmin(comp["two_theta"])) if len(comp) else np.nan,
                "position_slope": slope,
                "position_r2": r2,
                "max_prominence": float(np.nanmax(grp.get("prominence", pd.Series(dtype=float)))),
                "max_normalized_intensity": float(np.nanmax(grp.get("normalized_intensity", pd.Series(dtype=float)))),
                "max_raw_intensity": float(np.nanmax(grp.get("raw_intensity", pd.Series(dtype=float)))),
                "FWHM_change": float(np.nanmax(fwhm_values) - np.nanmin(fwhm_values)) if fwhm_values.notna().any() else np.nan,
                "median_fwhm": float(np.nanmedian(fwhm_values)) if fwhm_values.notna().any() else np.nan,
                "tier_a_count": int((grp["confidence_tier"] == "A").sum()),
                "tier_b_count": int((grp["confidence_tier"] == "B").sum()),
                "tier_c_count": int((grp["confidence_tier"] == "C").sum()),
                "frames_present": ";".join(sorted(grp["frame_label"].dropna().astype(str).unique())),
                "strongest_frame": str(strong_row.get("frame_label", strong_row.get("frame", ""))),
                "strongest_path": str(strong_row.get("path", "")),
                "strongest_pressure_gpa": safe_float(strong_row.get("pressure_gpa")),
                "source_methods": ";".join(sorted(set(grp.get("source_methods", pd.Series(dtype=str)).dropna().astype(str)))),
            }
        )
    out = pd.DataFrame(rows)
    if not group_summary.empty and "peak_group" in group_summary.columns:
        summary = group_summary.copy()
        summary["peak_group"] = pd.to_numeric(summary["peak_group"], errors="coerce").astype("Int64")
        out = out.merge(
            summary.add_prefix("summary_"),
            left_on="peak_group",
            right_on="summary_peak_group",
            how="left",
        )
        if "summary_max_roi_area" in out.columns:
            out["max_roi_area"] = pd.to_numeric(out["summary_max_roi_area"], errors="coerce")
        else:
            out["max_roi_area"] = np.nan
    else:
        out["max_roi_area"] = np.nan
    return out


def window_key(df: pd.DataFrame, cell: str, start: float, end: float) -> pd.DataFrame:
    return df[
        (df["cell"] == cell)
        & np.isclose(pd.to_numeric(df["window_start"], errors="coerce"), float(start))
        & np.isclose(pd.to_numeric(df["window_end"], errors="coerce"), float(end))
    ].copy()


def nearest_window(window_summary: pd.DataFrame, cell: str, theta: float) -> pd.Series | None:
    if window_summary.empty or not np.isfinite(theta):
        return None
    subset = window_summary[
        (window_summary["cell"] == cell)
        & (pd.to_numeric(window_summary["window_start"], errors="coerce") <= theta)
        & (pd.to_numeric(window_summary["window_end"], errors="coerce") >= theta)
    ].copy()
    if subset.empty:
        subset = window_summary[window_summary["cell"] == cell].copy()
        if subset.empty:
            return None
        subset["dist"] = (pd.to_numeric(subset["window_center"], errors="coerce") - theta).abs()
    else:
        subset["dist"] = (pd.to_numeric(subset["window_center"], errors="coerce") - theta).abs()
    return subset.sort_values(["dist", "static_background_score"], ascending=[True, False]).iloc[0]


def tier_counts_for_window(peaks: pd.DataFrame, cell: str, start: float, end: float) -> tuple[int, int, int, float, str]:
    if peaks.empty:
        return 0, 0, 0, np.nan, ""
    subset = peaks[
        (peaks["cell"] == cell)
        & (pd.to_numeric(peaks["two_theta"], errors="coerce") >= start)
        & (pd.to_numeric(peaks["two_theta"], errors="coerce") <= end)
    ]
    if subset.empty:
        return 0, 0, 0, np.nan, ""
    max_prom = float(np.nanmax(subset.get("prominence", pd.Series(dtype=float))))
    groups = sorted({int(g) for g in subset["peak_group"].dropna().unique()})[:8]
    return (
        int((subset["confidence_tier"] == "A").sum()),
        int((subset["confidence_tier"] == "B").sum()),
        int((subset["confidence_tier"] == "C").sum()),
        max_prom,
        ";".join(str(g) for g in groups),
    )


def feature_summary_for_window(features: pd.DataFrame, cell: str, start: float, end: float) -> dict:
    rows = window_key(features, cell, start, end).sort_values("pressure_gpa")
    if rows.empty:
        return {}
    pos = pd.to_numeric(rows.get("dominant_peak_position", rows.get("local_max_position")), errors="coerce")
    if pos.notna().sum() < 2:
        pos = pd.to_numeric(rows.get("local_max_position"), errors="coerce")
    fwhm = pd.to_numeric(rows.get("dominant_peak_fwhm_deg"), errors="coerce")
    roi = pd.to_numeric(rows.get("roi_area"), errors="coerce")
    return {
        "pressure_range": f"{rows['pressure_gpa'].min():g}-{rows['pressure_gpa'].max():g} GPa",
        "frame_count": int(rows["frame_label"].nunique()),
        "strongest_frame": str(rows.sort_values("roi_area", ascending=False).iloc[0].get("frame_label", "")),
        "max_roi_area": float(np.nanmax(roi)) if roi.notna().any() else np.nan,
        "position_shift": float(np.nanmax(pos) - np.nanmin(pos)) if pos.notna().sum() else np.nan,
        "FWHM_change": float(np.nanmax(fwhm) - np.nanmin(fwhm)) if fwhm.notna().sum() else np.nan,
        "NCC_summary": (
            f"median prev NCC={np.nanmedian(pd.to_numeric(rows.get('ncc_to_previous'), errors='coerce')):.3f}; "
            f"min zero-shift prev NCC={np.nanmin(pd.to_numeric(rows.get('ncc_zero_to_previous'), errors='coerce')):.3f}; "
            f"median |shift|={np.nanmedian(np.abs(pd.to_numeric(rows.get('best_shift_deg_to_previous'), errors='coerce'))):.3f} deg"
        ),
        "ACF_summary": f"median ACF change={np.nanmedian(pd.to_numeric(rows.get('acf_similarity_change'), errors='coerce')):.3f}",
        "static_score": float(np.nanmedian(pd.to_numeric(rows.get("static_background_score"), errors="coerce"))),
        "median_2theta": float(np.nanmedian(pos)) if pos.notna().any() else float(np.nanmedian(rows["window_center"])),
    }


def add_event(events: list[dict], event: dict) -> None:
    key = (
        event.get("event_type"),
        event.get("cell"),
        str(event.get("related_peak_group_ids", "")),
        round(safe_float(event.get("related_window_start")), 3)
        if np.isfinite(safe_float(event.get("related_window_start")))
        else "",
        round(safe_float(event.get("median_2theta")), 3)
        if np.isfinite(safe_float(event.get("median_2theta")))
        else "",
    )
    for existing in events:
        old_key = (
            existing.get("event_type"),
            existing.get("cell"),
            str(existing.get("related_peak_group_ids", "")),
            round(safe_float(existing.get("related_window_start")), 3)
            if np.isfinite(safe_float(existing.get("related_window_start")))
            else "",
            round(safe_float(existing.get("median_2theta")), 3)
            if np.isfinite(safe_float(existing.get("median_2theta")))
            else "",
        )
        if old_key == key:
            return
    events.append(event)


def event_from_window(
    row: pd.Series,
    event_type: str,
    reason: str,
    interpretation: str,
    priority: str,
    features: pd.DataFrame,
    peaks: pd.DataFrame,
) -> dict:
    cell = str(row.get("cell", ""))
    start = safe_float(row.get("window_start"))
    end = safe_float(row.get("window_end"))
    fs = feature_summary_for_window(features, cell, start, end)
    a, b, c, max_prom, groups = tier_counts_for_window(peaks, cell, start, end)
    median_theta = fs.get("median_2theta", safe_float(row.get("window_center")))
    return {
        "event_type": event_type,
        "cell": cell,
        "related_peak_group_ids": groups,
        "related_window_start": start,
        "related_window_end": end,
        "median_2theta": median_theta,
        "pressure_range": fs.get("pressure_range", ""),
        "compression_or_decompression": "compression",
        "frame_count": fs.get("frame_count", safe_float(row.get("n_frames"))),
        "strongest_frame": fs.get("strongest_frame", ""),
        "max_roi_area": fs.get("max_roi_area", safe_float(row.get("median_roi_area"))),
        "max_prominence": max_prom,
        "position_shift": fs.get("position_shift", np.nan),
        "FWHM_change": fs.get("FWHM_change", np.nan),
        "NCC_summary": fs.get("NCC_summary", ""),
        "ACF_summary": fs.get("ACF_summary", ""),
        "Tier A count": a,
        "Tier B count": b,
        "Tier C count": c,
        "static/background score if available": fs.get("static_score", safe_float(row.get("static_background_score"))),
        "initial_interpretation": interpretation,
        "manual_review_priority": priority,
        "reason_selected": reason,
        "source_kind": "window",
    }


def event_from_group(
    group: pd.Series,
    event_type: str,
    reason: str,
    interpretation: str,
    priority: str,
    window_summary: pd.DataFrame,
    features: pd.DataFrame,
) -> dict:
    gid = int(group["peak_group"])
    cell = str(group.get("cell", ""))
    theta = safe_float(group.get("median_2theta", group.get("group_two_theta")))
    win = nearest_window(window_summary, cell, theta)
    if win is not None:
        start, end = safe_float(win.get("window_start")), safe_float(win.get("window_end"))
        fs = feature_summary_for_window(features, cell, start, end)
        ncc_summary = fs.get("NCC_summary", "")
        acf_summary = fs.get("ACF_summary", "")
        static_score = fs.get("static_score", safe_float(win.get("static_background_score")))
        max_roi = fs.get("max_roi_area", safe_float(group.get("max_roi_area")))
    else:
        start, end = theta - 0.5, theta + 0.5
        ncc_summary = ""
        acf_summary = ""
        static_score = np.nan
        max_roi = safe_float(group.get("max_roi_area"))
    pressure_min = safe_float(group.get("pressure_min"))
    pressure_max = safe_float(group.get("pressure_max"))
    pressure_range = f"{pressure_min:g}-{pressure_max:g} GPa" if np.isfinite(pressure_min) and np.isfinite(pressure_max) else ""
    comp_or_decomp = "decompression/compression comparison" if safe_float(group.get("decomp_frame_count"), 0) > 0 else "compression"
    return {
        "event_type": event_type,
        "cell": cell,
        "related_peak_group_ids": str(gid),
        "related_window_start": start,
        "related_window_end": end,
        "median_2theta": theta,
        "pressure_range": pressure_range,
        "compression_or_decompression": comp_or_decomp,
        "frame_count": int(safe_float(group.get("frame_count"), 0)),
        "strongest_frame": str(group.get("strongest_frame", "")),
        "max_roi_area": max_roi,
        "max_prominence": safe_float(group.get("max_prominence")),
        "position_shift": safe_float(group.get("position_shift")),
        "FWHM_change": safe_float(group.get("FWHM_change")),
        "NCC_summary": ncc_summary,
        "ACF_summary": acf_summary,
        "Tier A count": int(safe_float(group.get("tier_a_count"), 0)),
        "Tier B count": int(safe_float(group.get("tier_b_count"), 0)),
        "Tier C count": int(safe_float(group.get("tier_c_count"), 0)),
        "static/background score if available": static_score,
        "initial_interpretation": interpretation,
        "manual_review_priority": priority,
        "reason_selected": reason,
        "source_kind": "peak_group",
    }


def find_acf_repeated_event(inputs: Inputs, features: pd.DataFrame, window_summary: pd.DataFrame) -> dict | None:
    index_path = inputs.xdi_suite / "within_frame_acf_index.csv"
    idx = read_csv_optional(index_path)
    if idx.empty:
        return None
    best = None
    for _, row in idx[idx["mode"].astype(str).str.contains("nonoverlap", case=False, na=False)].iterrows():
        csv_path = Path(str(row.get("csv", "")))
        if not csv_path.exists():
            continue
        mat = read_csv_optional(csv_path)
        if mat.empty:
            continue
        label_col = mat.columns[0]
        labels = mat[label_col].astype(str).tolist()
        data = mat.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")
        values = data.to_numpy(dtype=float)
        tri = np.triu(np.ones(values.shape, dtype=bool), k=1)
        valid = tri & np.isfinite(values)
        if not valid.any():
            continue
        score = float(np.nanpercentile(values[valid], 99))
        hits = np.argwhere(valid & (values >= max(0.88, score - 1e-9)))
        if hits.size == 0:
            continue
        i, j = hits[0]
        candidate = {
            "score": score,
            "cell": row.get("cell", ""),
            "pressure": safe_float(row.get("pressure_gpa")),
            "frame": row.get("frame_label", ""),
            "label_a": labels[int(i)] if int(i) < len(labels) else "",
            "label_b": list(data.columns)[int(j)] if int(j) < len(data.columns) else "",
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if not best:
        return None
    starts = []
    ends = []
    for label in [best["label_a"], best["label_b"]]:
        parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)", str(label))
        if len(parts) >= 2:
            starts.append(float(parts[-2]))
            ends.append(float(parts[-1]))
    if not starts:
        return None
    start, end = min(starts), max(ends)
    wrow = nearest_window(window_summary, best["cell"], (start + end) / 2.0)
    if wrow is None:
        return None
    event = event_from_window(
        wrow,
        "high_ACF_similarity_pattern_repeated_across_windows",
        (
            f"Non-overlapping within-frame ACF windows {best['label_a']} and {best['label_b']} "
            f"show high similarity ({best['score']:.3f}) at {best['frame']}."
        ),
        "candidate repeated texture/profile motif; it needs local XY and raw-image checking before any phase claim",
        "Medium",
        features,
        pd.DataFrame(),
    )
    event["compression_or_decompression"] = "compression"
    event["strongest_frame"] = str(best["frame"])
    event["pressure_range"] = f"{best['pressure']:g} GPa"
    event["ACF_summary"] = f"non-overlap ACF similarity={best['score']:.3f}; paired windows={best['label_a']} and {best['label_b']}"
    return event


def select_events(
    inputs: Inputs,
    features: pd.DataFrame,
    window_summary: pd.DataFrame,
    peaks: pd.DataFrame,
    groups: pd.DataFrame,
    raw_static: pd.DataFrame,
    max_events: int,
) -> pd.DataFrame:
    events: list[dict] = []

    usable_groups = groups.copy()
    if not usable_groups.empty:
        usable_groups["abs_slope"] = pd.to_numeric(usable_groups["position_slope"], errors="coerce").abs()
        usable_groups["sample_score"] = (
            pd.to_numeric(usable_groups["compression_frame_count"], errors="coerce").fillna(0)
            + 10 * pd.to_numeric(usable_groups["abs_slope"], errors="coerce").fillna(0)
            + pd.to_numeric(usable_groups["max_prominence"], errors="coerce").rank(pct=True).fillna(0)
        )

        recurring = usable_groups[
            (pd.to_numeric(usable_groups["compression_frame_count"], errors="coerce") >= 5)
            & (pd.to_numeric(usable_groups["position_shift"], errors="coerce") >= 0.05)
        ].sort_values("sample_score", ascending=False)
        for _, row in recurring.head(2).iterrows():
            add_event(
                events,
                event_from_group(
                    row,
                    "recurring_pressure_dependent_peak_group",
                    "The same peak group is detected in many compression frames and its position changes with pressure.",
                    "possible sample-related pressure-dependent feature, not a phase identity by itself",
                    "High",
                    window_summary,
                    features,
                ),
            )

        tier_b = usable_groups[
            (pd.to_numeric(usable_groups["tier_b_count"], errors="coerce") >= 3)
            & (pd.to_numeric(usable_groups["tier_b_count"], errors="coerce") >= pd.to_numeric(usable_groups["tier_a_count"], errors="coerce"))
            & (pd.to_numeric(usable_groups["compression_frame_count"], errors="coerce") >= 3)
        ].sort_values(["compression_frame_count", "max_prominence"], ascending=False)
        for _, row in tier_b.head(2).iterrows():
            add_event(
                events,
                event_from_group(
                    row,
                    "Tier-B-dependent weak trajectory",
                    "This trajectory depends strongly on Tier B detections, so it is scientifically interesting but detector-threshold sensitive.",
                    "weak candidate trajectory; needs manual local XY and raw-image confirmation",
                    "High",
                    window_summary,
                    features,
                ),
            )

        isolated = usable_groups[
            (pd.to_numeric(usable_groups["compression_frame_count"], errors="coerce") <= 1)
            & (pd.to_numeric(usable_groups["max_prominence"], errors="coerce") > 0)
        ].sort_values("max_prominence", ascending=False)
        for _, row in isolated.head(1).iterrows():
            add_event(
                events,
                event_from_group(
                    row,
                    "isolated_strong_peak",
                    "A strong local peak appears in only one compression frame.",
                    "unresolved isolated feature; could be sample, spotty texture, gasket/cell, or detector artifact",
                    "High",
                    window_summary,
                    features,
                ),
            )

        decomp = usable_groups[
            (pd.to_numeric(usable_groups["decomp_frame_count"], errors="coerce") > 0)
            & (pd.to_numeric(usable_groups["max_prominence"], errors="coerce") > 0)
        ].sort_values(["decomp_frame_count", "max_prominence"], ascending=False)
        if not decomp.empty:
            add_event(
                events,
                event_from_group(
                    decomp.iloc[0],
                    "compression_decompression_hysteresis",
                    "The peak group is detected in the decompression scan, so it is a natural pressure-path check.",
                    "possible compression/decompression history effect; compare decomp local XY against nearby compression pressures",
                    "High",
                    window_summary,
                    features,
                ),
            )

    if not window_summary.empty:
        shift = window_summary.sort_values("motion_score", ascending=False)
        for _, row in shift.head(2).iterrows():
            add_event(
                events,
                event_from_window(
                    row,
                    "systematic_2theta_shift",
                    "The pressure-window map reports a large position/motion score in this 2theta window.",
                    "possible sample compression or changing overlap; needs indexed/refined-line comparison",
                    "High",
                    features,
                    peaks,
                ),
            )

        # Appearance/disappearance from existing feature counts and ROI changes.
        app_rows = []
        for _, row in window_summary.iterrows():
            local = window_key(features, row["cell"], row["window_start"], row["window_end"]).sort_values("pressure_gpa")
            if local.empty:
                continue
            first = local.iloc[0]
            last = local.iloc[-1]
            d_count = safe_float(last.get("tier_ab_peak_count"), 0) - safe_float(first.get("tier_ab_peak_count"), 0)
            d_roi = safe_float(last.get("roi_area"), 0) - safe_float(first.get("roi_area"), 0)
            score = safe_float(row.get("appearance_disappearance_score"), 0) + abs(d_count) * 0.1 + abs(d_roi) * 0.01
            app_rows.append((score, d_count, d_roi, row))
        app_rows = sorted(app_rows, key=lambda x: x[0], reverse=True)
        app = next((item for item in app_rows if item[1] > 0 or (item[1] >= 0 and item[2] > 0)), app_rows[0] if app_rows else None)
        app_key = None
        if app:
            app_key = (app[3].get("cell"), safe_float(app[3].get("window_start")), safe_float(app[3].get("window_end")))
        dis = next(
            (
                item
                for item in app_rows
                if (item[1] < 0 or (item[1] <= 0 and item[2] < 0))
                and (item[3].get("cell"), safe_float(item[3].get("window_start")), safe_float(item[3].get("window_end"))) != app_key
            ),
            None,
        )
        if dis is None and len(app_rows) > 1:
            dis = next(
                (
                    item
                    for item in app_rows
                    if (item[3].get("cell"), safe_float(item[3].get("window_start")), safe_float(item[3].get("window_end"))) != app_key
                ),
                None,
            )
        if app:
            app_text = "increases" if app[1] > 0 else "shows an ROI-area increase while peak count is not the main signal"
            add_event(
                events,
                event_from_window(
                    app[3],
                    "sudden_appearance_with_pressure",
                    f"Tier A+B count/ROI {app_text} across the pressure series (delta count={app[1]:.1f}, delta ROI={app[2]:.2f}).",
                    "possible appearance/intensity-transfer feature; not a phase transition without line assignment",
                    "High",
                    features,
                    peaks,
                ),
            )
        if dis:
            dis_text = "decreases" if dis[1] < 0 else "shows an ROI-area decrease while peak count is not the main signal"
            add_event(
                events,
                event_from_window(
                    dis[3],
                    "disappearance_with_pressure",
                    f"Tier A+B count/ROI {dis_text} across the pressure series (delta count={dis[1]:.1f}, delta ROI={dis[2]:.2f}).",
                    "possible disappearance or overlap redistribution; needs local XY confirmation",
                    "High",
                    features,
                    peaks,
                ),
            )

        fwhm_scores = []
        roi_scores = []
        ncc_scores = []
        for _, row in window_summary.iterrows():
            local = window_key(features, row["cell"], row["window_start"], row["window_end"]).sort_values("pressure_gpa")
            if local.empty:
                continue
            fwhm = pd.to_numeric(local.get("dominant_peak_fwhm_deg"), errors="coerce")
            roi = pd.to_numeric(local.get("roi_area"), errors="coerce")
            ncc = pd.to_numeric(local.get("ncc_zero_to_previous"), errors="coerce")
            acf = pd.to_numeric(local.get("acf_similarity_change"), errors="coerce")
            fwhm_scores.append((float(np.nanmax(fwhm) - np.nanmin(fwhm)) if fwhm.notna().any() else -1, row))
            roi_cv = float(np.nanstd(roi) / (abs(np.nanmean(roi)) + 1e-9)) if roi.notna().any() else -1
            roi_scores.append((roi_cv, row))
            ncc_drop = 1.0 - float(np.nanmin(ncc)) if ncc.notna().any() else -1
            acf_change = float(np.nanmax(acf)) if acf.notna().any() else 0
            ncc_scores.append((ncc_drop + acf_change, row))
        for score, row in sorted(fwhm_scores, key=lambda x: x[0], reverse=True)[:1]:
            add_event(
                events,
                event_from_window(
                    row,
                    "FWHM_or_broadening_anomaly",
                    f"Dominant FWHM changes strongly in this window (span about {score:.3f} deg).",
                    "possible broadening, strain, overlap, or background feature; verify peak shape in local XY",
                    "Medium",
                    features,
                    peaks,
                ),
            )
        for score, row in sorted(roi_scores, key=lambda x: x[0], reverse=True)[:1]:
            add_event(
                events,
                event_from_window(
                    row,
                    "ROI_area_anomaly",
                    f"ROI area varies unusually strongly across pressure (coefficient of variation about {score:.2f}).",
                    "possible intensity redistribution or artifact; needs local XY and raw-image check",
                    "Medium",
                    features,
                    peaks,
                ),
            )
        for score, row in sorted(ncc_scores, key=lambda x: x[0], reverse=True)[:2]:
            add_event(
                events,
                event_from_window(
                    row,
                    "NCC_drop_or_window_structural_change",
                    f"Same-window correlation changes strongly (combined NCC/ACF score {score:.2f}).",
                    "candidate window-level structural/profile change; map evidence only points to where to inspect",
                    "High",
                    features,
                    peaks,
                ),
            )

        acf_event = find_acf_repeated_event(inputs, features, window_summary)
        if acf_event:
            add_event(events, acf_event)

        if raw_static is not None and not raw_static.empty and "verdict" in raw_static.columns:
            confirmed = raw_static[raw_static["verdict"].astype(str).str.upper() == "STATIC"].copy()
            for _, raw_row in confirmed.iterrows():
                theta = safe_float(raw_row.get("median_2theta"))
                cell = str(raw_row.get("cell", ""))
                win = nearest_window(window_summary, cell, theta)
                if win is None:
                    continue
                event = event_from_window(
                    win,
                    "likely_static_background_diamond_like_peak",
                    (
                        f"Raw-2D centroid tracking already confirms a static spot at {theta:.4f} deg "
                        f"(n={raw_row.get('n_frames', '')}, span={raw_row.get('span_deg', '')})."
                    ),
                    "likely static/background candidate; this is the current authoritative mask-candidate evidence",
                    "High",
                    features,
                    peaks,
                )
                event["median_2theta"] = theta
                event["position_shift"] = safe_float(raw_row.get("span_deg"))
                event["related_peak_group_ids"] = event.get("related_peak_group_ids", "")
                add_event(events, event)

        static_rows = window_summary.sort_values(
            ["raw2d_static_tracks", "static_background_score"], ascending=False
        )
        for _, row in static_rows.head(2).iterrows():
            priority = "High" if safe_float(row.get("raw2d_static_tracks"), 0) > 0 else "Medium"
            add_event(
                events,
                event_from_window(
                    row,
                    "likely_static_background_diamond_like_peak",
                    "Static/background score is high; if raw2D_static_tracks > 0 this is an especially important mask candidate.",
                    "likely static/background candidate only if raw-2D centroid evidence supports it",
                    priority,
                    features,
                    peaks,
                ),
            )

    # If a category is still missing, fill from the strongest remaining groups.
    if len(events) < max_events and not usable_groups.empty:
        filler = usable_groups.sort_values(["compression_frame_count", "max_prominence"], ascending=False)
        for _, row in filler.iterrows():
            if len(events) >= max_events:
                break
            add_event(
                events,
                event_from_group(
                    row,
                    "recurring_pressure_dependent_peak_group",
                    "Filler event: recurring peak group selected to keep the catalog scientifically broad.",
                    "possible sample-related or background-related recurring feature; unresolved without manual review",
                    "Medium",
                    window_summary,
                    features,
                ),
            )

    events = events[:max_events]
    for i, event in enumerate(events, start=1):
        event["event_id"] = f"event_{i:03d}"
    catalog = pd.DataFrame(events)
    for col in CATALOG_COLUMNS:
        if col not in catalog.columns:
            catalog[col] = ""
    return catalog[CATALOG_COLUMNS + [c for c in catalog.columns if c not in CATALOG_COLUMNS]]


def add_catalog_dq(catalog: pd.DataFrame, wavelength_a: float | None) -> pd.DataFrame:
    out = catalog.copy()
    theta = pd.to_numeric(out["median_2theta"], errors="coerce")
    d = d_from_2theta(theta, wavelength_a)
    q = q_from_d(d)
    out["observed_d"] = d
    out["observed_q"] = q
    return out


def nearest_ref(refs: pd.DataFrame, theta: float, cell: str = "", pressure: float | None = None) -> pd.Series | None:
    if refs.empty or not np.isfinite(theta) or "reference_2theta" not in refs.columns:
        return None
    subset = refs.copy()
    if cell and "cell" in subset.columns:
        cell_subset = subset[(subset["cell"] == cell) | (subset["cell"].astype(str) == "")]
        if not cell_subset.empty:
            subset = cell_subset
    if pressure is not None and np.isfinite(pressure) and "pressure_gpa" in subset.columns:
        tmp = subset.copy()
        tmp["pdist"] = (pd.to_numeric(tmp["pressure_gpa"], errors="coerce") - pressure).abs()
        close = tmp[tmp["pdist"].fillna(999) <= 1.0]
        if not close.empty:
            subset = close
    subset = subset.copy()
    subset["position_difference"] = (pd.to_numeric(subset["reference_2theta"], errors="coerce") - theta).abs()
    if subset["position_difference"].notna().sum() == 0:
        return None
    return subset.sort_values("position_difference").iloc[0]


def raw2d_evidence(raw_static: pd.DataFrame, class_table: pd.DataFrame, cell: str, theta: float) -> tuple[str, float]:
    notes = []
    static_score = np.nan
    if not raw_static.empty and "median_2theta" in raw_static.columns:
        sub = raw_static[raw_static.get("cell", "") == cell].copy()
        if not sub.empty:
            sub["dist"] = (pd.to_numeric(sub["median_2theta"], errors="coerce") - theta).abs()
            hit = sub.sort_values("dist").head(1)
            if not hit.empty and safe_float(hit.iloc[0]["dist"]) <= 0.15:
                r = hit.iloc[0]
                notes.append(
                    f"raw2D spot tracking nearest {safe_float(r['median_2theta']):.4f} deg: {r.get('verdict', '')}, n={r.get('n_frames', '')}, span={r.get('span_deg', '')}"
                )
                static_score = 1.0 if str(r.get("verdict", "")).upper() == "STATIC" else 0.0
    if not class_table.empty and "median_2theta" in class_table.columns:
        sub = class_table[class_table.get("cell", "") == cell].copy()
        if not sub.empty:
            sub["dist"] = (pd.to_numeric(sub["median_2theta"], errors="coerce") - theta).abs()
            hit = sub.sort_values("dist").head(1)
            if not hit.empty and safe_float(hit.iloc[0]["dist"]) <= 0.15:
                r = hit.iloc[0]
                notes.append(f"2D ring/spot class nearest {safe_float(r['median_2theta']):.4f} deg: {r.get('assigned_class', '')}")
    return "; ".join(notes), static_score


def reference_assignments(
    catalog: pd.DataFrame,
    known_refs: pd.DataFrame,
    refinement_refs: pd.DataFrame,
    peak_identification: pd.DataFrame,
    raw_static: pd.DataFrame,
    class_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    all_refs = []
    if not known_refs.empty:
        tmp = known_refs.copy()
        tmp["assignment_type"] = "known_background_reference"
        all_refs.append(tmp)
    if not refinement_refs.empty:
        tmp = refinement_refs.copy()
        tmp["assignment_type"] = "refinement_predicted_reflection"
        all_refs.append(tmp)
    refs = pd.concat(all_refs, ignore_index=True) if all_refs else pd.DataFrame()

    for _, event in catalog.iterrows():
        theta = safe_float(event.get("median_2theta"))
        cell = str(event.get("cell", ""))
        pressure = pressure_from_text(event.get("strongest_frame", "")) or safe_float(event.get("pressure_range", np.nan))
        raw_note, raw_static_score = raw2d_evidence(raw_static, class_table, cell, theta)
        static_score = safe_float(event.get("static/background score if available"), np.nan)
        static_like = (
            (np.isfinite(static_score) and static_score >= 0.68)
            or (np.isfinite(raw_static_score) and raw_static_score >= 1.0)
            or "static" in str(event.get("event_type", "")).lower()
        )
        event_type_lower = str(event.get("event_type", "")).lower()
        event_shift_span = safe_float(event.get("position_shift"))
        moving_like = (
            "systematic" in event_type_lower
            or ("recurring" in event_type_lower and np.isfinite(event_shift_span) and event_shift_span >= 0.05)
            or "hysteresis" in event_type_lower
            or "tier-b" in event_type_lower
            or "appearance" in event_type_lower
            or "disappearance" in event_type_lower
        )
        nearest = nearest_ref(refs, theta, cell, pressure)
        candidate_assignment = "unassigned"
        assignment_type = ""
        reference_peak_position = np.nan
        reference_phase = ""
        reference_hkl = ""
        position_difference = np.nan
        confidence = "unresolved"
        consistency = "unresolved"
        notes = []

        if nearest is not None:
            reference_peak_position = safe_float(nearest.get("reference_2theta"))
            reference_phase = str(nearest.get("reference_phase", ""))
            reference_hkl = str(nearest.get("reference_hkl", ""))
            position_difference = safe_float(nearest.get("position_difference"))
            assignment_type = str(nearest.get("assignment_type", ""))
            candidate_assignment = f"nearest {reference_phase} {reference_hkl}"
            notes.append(f"nearest reference is {position_difference:.4f} deg away")

            phase_lower = reference_phase.lower()
            if position_difference <= REF_TOL_DEG:
                if "diamond" in phase_lower:
                    confidence = "likely_diamond_or_cell" if static_like else "possible_background"
                    consistency = "fixed reference; consistent only if raw position is static"
                elif "re-gasket" in phase_lower or "ne(" in phase_lower or "ruby" in phase_lower or reference_phase == "W":
                    if np.isfinite(raw_static_score) and raw_static_score >= 1.0:
                        confidence = "likely_static_background"
                    else:
                        confidence = "likely_gasket_or_medium" if static_like or not moving_like else "possible_background"
                    consistency = "background/reference line nearby; pressure behavior must be checked manually"
                elif "uote" in phase_lower:
                    if static_like:
                        confidence = "likely_static_background" if raw_static_score == 1.0 else "possible_background"
                        consistency = "near a UOTe refinement line, but this event was selected as static-like; raw-2D confirmation takes priority before masking"
                    else:
                        confidence = "possible_sample_candidate" if moving_like else "unresolved"
                        consistency = "near a refinement-predicted UOTe reflection; pressure behavior is the key check"
                else:
                    confidence = "possible_background" if static_like else ("possible_sample_candidate" if moving_like else "unresolved")
                    consistency = "near a refinement-predicted line but identity remains cautious"
            else:
                if static_like:
                    confidence = "likely_static_background" if raw_static_score == 1.0 else "possible_background"
                    consistency = "position is not close to a listed reference, but static/background behavior is flagged"
                elif moving_like:
                    confidence = "possible_sample_candidate"
                    consistency = "moving/pressure-dependent behavior without a close listed reference"
        else:
            if static_like:
                confidence = "likely_static_background" if raw_static_score == 1.0 else "possible_background"
                consistency = "static-like behavior but no local reference line found"
            elif moving_like:
                confidence = "possible_sample_candidate"
                consistency = "pressure-dependent behavior but no local reference line found"

        if raw_note:
            notes.append(raw_note)

        # Existing trajectory identification is a second, independent check.
        if not peak_identification.empty and "median_2theta" in peak_identification.columns:
            sub = peak_identification[peak_identification.get("cell", "") == cell].copy()
            if not sub.empty:
                sub["dist"] = (pd.to_numeric(sub["median_2theta"], errors="coerce") - theta).abs()
                hit = sub.sort_values("dist").head(1)
                if not hit.empty and safe_float(hit.iloc[0]["dist"]) <= 0.15:
                    r = hit.iloc[0]
                    notes.append(
                        f"existing d/EOS table nearest trajectory {r.get('traj_id', '')}: {r.get('motion', '')}, ref={r.get('ref_phase', '')} {r.get('ref_hkl', '')}"
                    )
                    if str(r.get("motion", "")).upper() == "MOVING" and confidence == "possible_sample_candidate":
                        confidence = "strong_sample_candidate"

        rows.append(
            {
                "event_id": event["event_id"],
                "peak_group_id": event.get("related_peak_group_ids", ""),
                "observed_2theta": theta,
                "observed_d": event.get("observed_d", np.nan),
                "observed_q": event.get("observed_q", np.nan),
                "candidate_assignment": candidate_assignment,
                "assignment_type": assignment_type,
                "reference_peak_position": reference_peak_position,
                "reference_phase": reference_phase,
                "reference_hkl": reference_hkl,
                "position_difference": position_difference,
                "pressure_behavior_consistency": consistency,
                "static_background_score": static_score,
                "raw_2d_evidence": raw_note,
                "assignment_confidence": confidence,
                "notes": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def static_every_frame_candidates(peaks: pd.DataFrame, features: pd.DataFrame, raw_static: pd.DataFrame, wavelength_a: float | None) -> pd.DataFrame:
    rows = []
    if not peaks.empty:
        comp = peaks[~peaks["decomp"]].dropna(subset=["peak_group", "two_theta"]).copy()
        frame_counts = comp.groupby("cell")["frame_label"].nunique().to_dict()
        comp["peak_group"] = comp["peak_group"].astype(int)
        for (cell, gid), grp in comp.groupby(["cell", "peak_group"]):
            needed = frame_counts.get(cell, 0)
            if needed <= 0:
                continue
            n = grp["frame_label"].nunique()
            if n != needed:
                continue
            slope, r2 = slope_and_r2(grp["pressure_gpa"], grp["two_theta"])
            span = float(grp["two_theta"].max() - grp["two_theta"].min())
            theta = float(grp["two_theta"].median())
            static_like = abs(slope) <= 0.005 and span <= 0.05 if np.isfinite(slope) else span <= 0.03
            rows.append(
                {
                    "candidate_kind": "strict_peak_group_all_compression_frames",
                    "cell": cell,
                    "peak_group": int(gid),
                    "median_2theta": theta,
                    "d_A": float(d_from_2theta(theta, wavelength_a)) if wavelength_a else np.nan,
                    "q_invA": float(q_from_d(d_from_2theta(theta, wavelength_a))) if wavelength_a else np.nan,
                    "frames_present": n,
                    "required_frames": needed,
                    "span_deg": span,
                    "slope_deg_per_GPa": slope,
                    "tier_a_count": int((grp["confidence_tier"] == "A").sum()),
                    "tier_b_count": int((grp["confidence_tier"] == "B").sum()),
                    "tier_c_count": int((grp["confidence_tier"] == "C").sum()),
                    "max_prominence": float(np.nanmax(grp.get("prominence", pd.Series(dtype=float)))),
                    "correlation_static_like": bool(static_like),
                    "raw2d_confirmed": False,
                    "notes": "Strict all-frame 1D peak-group candidate; requires raw-2D confirmation.",
                }
            )
    if not raw_static.empty:
        confirmed = raw_static[raw_static.get("verdict", "").astype(str).str.upper() == "STATIC"].copy()
        for _, r in confirmed.iterrows():
            theta = safe_float(r.get("median_2theta"))
            rows.append(
                {
                    "candidate_kind": "raw2d_confirmed_static_spot",
                    "cell": r.get("cell", ""),
                    "peak_group": "",
                    "median_2theta": theta,
                    "d_A": float(d_from_2theta(theta, wavelength_a)) if wavelength_a else np.nan,
                    "q_invA": float(q_from_d(d_from_2theta(theta, wavelength_a))) if wavelength_a else np.nan,
                    "frames_present": r.get("n_frames", ""),
                    "required_frames": "",
                    "span_deg": r.get("span_deg", ""),
                    "slope_deg_per_GPa": r.get("slope", ""),
                    "tier_a_count": "",
                    "tier_b_count": "",
                    "tier_c_count": "",
                    "max_prominence": "",
                    "correlation_static_like": "",
                    "raw2d_confirmed": True,
                    "notes": "This is the authoritative static evidence; use it as mask-candidate proof before masking.",
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["raw2d_confirmed", "correlation_static_like", "cell", "median_2theta"], ascending=[False, False, True, True])


def build_covarying_clusters(peaks: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    if peaks.empty or groups.empty:
        return pd.DataFrame()
    comp = peaks[~peaks["decomp"]].dropna(subset=["peak_group", "two_theta"]).copy()
    comp["peak_group"] = comp["peak_group"].astype(int)
    rows = []
    for (cell, gid), grp in comp.groupby(["cell", "peak_group"]):
        if grp["frame_label"].nunique() < 3:
            continue
        slope, r2 = slope_and_r2(grp["pressure_gpa"], grp["two_theta"])
        fwhm_col = "width_deg" if "width_deg" in grp.columns else "width_estimate_deg"
        fwhm_slope, _ = slope_and_r2(grp["pressure_gpa"], pd.to_numeric(grp.get(fwhm_col), errors="coerce"))
        pmin, pmax = float(grp["pressure_gpa"].min()), float(grp["pressure_gpa"].max())
        slope_sign = "positive_shift" if slope > 0.01 else ("negative_shift" if slope < -0.01 else "flat")
        broadening = "broadens" if fwhm_slope > 0.002 else ("narrows" if fwhm_slope < -0.002 else "fwhm_stable")
        interval = f"{round(pmin):02.0f}-{round(pmax):02.0f}GPa"
        tier_class = "TierA_present" if (grp["confidence_tier"] == "A").sum() else "TierB_or_C_only"
        rows.append(
            {
                "cell": cell,
                "peak_group": int(gid),
                "median_2theta": float(grp["two_theta"].median()),
                "presence_interval": interval,
                "pressure_min": pmin,
                "pressure_max": pmax,
                "frame_count": int(grp["frame_label"].nunique()),
                "position_slope_deg_per_GPa": slope,
                "position_r2": r2,
                "slope_sign": slope_sign,
                "fwhm_slope_deg_per_GPa": fwhm_slope,
                "broadening_class": broadening,
                "tier_class": tier_class,
                "presence_signature": ";".join(sorted(grp["frame_label"].astype(str).unique())),
                "max_prominence": float(np.nanmax(grp.get("prominence", pd.Series(dtype=float)))),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    cluster_rows = []
    cid = 1
    keys = ["cell", "presence_interval", "slope_sign", "broadening_class", "tier_class"]
    for key, grp in detail.groupby(keys):
        if len(grp) < 2:
            continue
        cluster_id = f"cluster_{cid:03d}"
        cid += 1
        for _, row in grp.iterrows():
            out = row.to_dict()
            out["cluster_id"] = cluster_id
            out["cluster_size"] = len(grp)
            out["cluster_label"] = " | ".join(str(x) for x in key)
            out["interpretation"] = "candidate correlated peak set; do not call a phase transition without refinement/indexing"
            cluster_rows.append(out)
    return pd.DataFrame(cluster_rows).sort_values(["cluster_size", "cluster_id", "median_2theta"], ascending=[False, True, True])


def load_xy(path: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path:
        return None
    p = Path(str(path))
    if not p.exists():
        return None
    try:
        arr = np.loadtxt(p)
    except UnicodeDecodeError:
        arr = np.loadtxt(p, encoding="latin1")
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


def event_window(event: pd.Series) -> tuple[float, float]:
    start = safe_float(event.get("related_window_start"))
    end = safe_float(event.get("related_window_end"))
    theta = safe_float(event.get("median_2theta"))
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        start, end = theta - 0.5, theta + 0.5
    return start, end


def plot_local_xy(
    event: pd.Series,
    features: pd.DataFrame,
    peaks: pd.DataFrame,
    out_path: Path,
    reference_lines: pd.DataFrame | None = None,
    with_reference: bool = False,
) -> None:
    cell = str(event.get("cell", ""))
    start, end = event_window(event)
    theta = safe_float(event.get("median_2theta"))
    local_features = features[
        (features["cell"] == cell)
        & (pd.to_numeric(features["window_start"], errors="coerce") <= theta)
        & (pd.to_numeric(features["window_end"], errors="coerce") >= theta)
    ].sort_values("pressure_gpa")
    if local_features.empty:
        local_features = features[features["cell"] == cell].sort_values("pressure_gpa")

    xy_rows = []
    seen = set()
    for _, row in local_features.iterrows():
        path = str(row.get("xy_path", ""))
        if path and path not in seen:
            seen.add(path)
            xy_rows.append((path, row.get("frame_label", ""), safe_float(row.get("pressure_gpa")), False))
    if "decompression" in str(event.get("compression_or_decompression", "")).lower() and not peaks.empty:
        gids = [int(x) for x in re.findall(r"\d+", str(event.get("related_peak_group_ids", "")))]
        if gids:
            extra = peaks[(peaks["peak_group"].isin(gids)) & (peaks["decomp"])]
            for _, row in extra.iterrows():
                path = str(row.get("path", ""))
                if path and path not in seen:
                    seen.add(path)
                    xy_rows.append((path, row.get("frame_label", "decomp"), safe_float(row.get("pressure_gpa")), True))

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    slices = []
    for path, label, pressure, decomp in xy_rows:
        loaded = load_xy(path)
        if loaded is None:
            continue
        x, y = loaded
        mask = (x >= start) & (x <= end)
        if mask.sum() < 3:
            continue
        slices.append((x[mask], y[mask], str(label), pressure, decomp, path))
    if not slices:
        ax.text(0.5, 0.5, "No local .xy data could be loaded.", ha="center", va="center")
        ax.axis("off")
    else:
        all_y = np.concatenate([s[1] for s in slices])
        robust_span = np.nanpercentile(all_y, 98) - np.nanpercentile(all_y, 5)
        if not np.isfinite(robust_span) or robust_span <= 0:
            robust_span = np.nanmax(all_y) - np.nanmin(all_y)
        if not np.isfinite(robust_span) or robust_span <= 0:
            robust_span = 1.0
        offset_step = robust_span * 1.15
        tier_colors = {"A": "tab:green", "B": "tab:orange", "C": "tab:red"}
        for idx, (x, y, label, pressure, decomp, path) in enumerate(slices):
            offset = idx * offset_step
            color = "tab:purple" if decomp else "0.25"
            ax.plot(x, y + offset, lw=0.9, color=color, alpha=0.95)
            ax.text(end + 0.02 * (end - start), np.nanmedian(y + offset), label, fontsize=8, va="center")
            if not peaks.empty:
                psub = peaks[(peaks["path"].astype(str) == path) & (peaks["two_theta"] >= start) & (peaks["two_theta"] <= end)]
                for _, peak in psub.iterrows():
                    px = safe_float(peak.get("two_theta"))
                    if not np.isfinite(px):
                        continue
                    py = float(np.interp(px, x, y)) + offset
                    tier = str(peak.get("confidence_tier", "")).upper()
                    ax.scatter([px], [py], s=22, color=tier_colors.get(tier, "0.5"), zorder=4)
        if np.isfinite(theta):
            ax.axvline(theta, color="black", lw=1.0, ls="--", label="candidate position")
        if with_reference and reference_lines is not None and not reference_lines.empty:
            refs = reference_lines[
                (pd.to_numeric(reference_lines["reference_2theta"], errors="coerce") >= start)
                & (pd.to_numeric(reference_lines["reference_2theta"], errors="coerce") <= end)
            ].copy()
            for _, ref in refs.iterrows():
                rt = safe_float(ref.get("reference_2theta"))
                phase = str(ref.get("reference_phase", ""))
                ax.axvline(rt, color="tab:red" if phase == "diamond" else "tab:blue", lw=0.8, alpha=0.55)
                ax.text(rt, ax.get_ylim()[1], phase, rotation=90, va="top", ha="right", fontsize=7)
        ax.set_xlim(start, end)
        ax.set_xlabel("2theta (deg)")
        ax.set_ylabel("raw local intensity + vertical offset")
        ax.set_title(f"{event['event_id']}: local .xy pressure series, {cell}, {start:.2f}-{end:.2f} deg")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_metric_vs_pressure(
    x: Iterable[float],
    y: Iterable[float],
    out_path: Path,
    title: str,
    ylabel: str,
    marker: str = "o",
) -> bool:
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    keep = np.isfinite(xx) & np.isfinite(yy)
    if keep.sum() < 2:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xx[keep], yy[keep], marker=marker, lw=1.2)
    ax.set_xlabel("pressure (GPa)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True


def event_metrics(event: pd.Series, features: pd.DataFrame, peaks: pd.DataFrame, wavelength_a: float | None) -> pd.DataFrame:
    cell = str(event.get("cell", ""))
    gids = [int(x) for x in re.findall(r"\d+", str(event.get("related_peak_group_ids", "")))]
    if gids and not peaks.empty:
        rows = peaks[(peaks["peak_group"].isin(gids)) & ((peaks["cell"] == cell) | (cell == ""))].copy()
        if not rows.empty:
            rows = rows.rename(columns={"two_theta": "observed_2theta", "width_deg": "fwhm_deg"})
            rows = add_d_q_columns(rows, "observed_2theta", wavelength_a, prefix="observed_")
            keep = [
                "frame_label",
                "pressure_gpa",
                "cell",
                "decomp",
                "path",
                "peak_group",
                "observed_2theta",
                "observed_d_A",
                "observed_q_invA",
                "prominence",
                "normalized_intensity",
                "raw_intensity",
                "fwhm_deg",
                "width_estimate_deg",
                "confidence_tier",
                "source_methods",
            ]
            return rows[[c for c in keep if c in rows.columns]].sort_values(["pressure_gpa", "frame_label"])
    start, end = event_window(event)
    rows = window_key(features, cell, start, end).copy()
    if rows.empty:
        rows = features[(features["cell"] == cell) & (features["window_center"].sub((start + end) / 2).abs() < 0.26)].copy()
    rows["observed_2theta"] = pd.to_numeric(rows.get("dominant_peak_position"), errors="coerce")
    missing = rows["observed_2theta"].isna()
    rows.loc[missing, "observed_2theta"] = pd.to_numeric(rows.loc[missing].get("local_max_position"), errors="coerce")
    rows = add_d_q_columns(rows, "observed_2theta", wavelength_a, prefix="observed_")
    keep = [
        "frame_label",
        "pressure_gpa",
        "cell",
        "xy_path",
        "raw_2d_path",
        "window_label",
        "observed_2theta",
        "observed_d_A",
        "observed_q_invA",
        "roi_area",
        "tier_a_peak_count",
        "tier_ab_peak_count",
        "tier_c_candidate_count",
        "dominant_peak_fwhm_deg",
        "ncc_zero_to_previous",
        "ncc_to_previous",
        "best_shift_deg_to_previous",
        "acf_similarity_change",
        "static_background_score",
    ]
    return rows[[c for c in keep if c in rows.columns]].sort_values(["pressure_gpa", "frame_label"])


def create_event_package(
    event: pd.Series,
    out_dir: Path,
    inputs: Inputs,
    features: pd.DataFrame,
    peaks: pd.DataFrame,
    reference_lines: pd.DataFrame,
    wavelength_a: float | None,
) -> dict:
    event_dir = out_dir / "events" / event["event_id"]
    event_dir.mkdir(parents=True, exist_ok=True)
    metrics = event_metrics(event, features, peaks, wavelength_a)
    metrics.to_csv(event_dir / "event_metrics.csv", index=False)

    plot_local_xy(event, features, peaks, event_dir / "local_xy_pressure_series.png")
    plot_local_xy(event, features, peaks, event_dir / "reference_overlay.png", reference_lines, with_reference=True)

    cell = str(event.get("cell", ""))
    start, end = event_window(event)
    local_features = window_key(features, cell, start, end).sort_values("pressure_gpa")
    files = {
        "local_xy": event_dir / "local_xy_pressure_series.png",
        "reference_overlay": event_dir / "reference_overlay.png",
    }
    if not local_features.empty:
        if plot_metric_vs_pressure(
            local_features["pressure_gpa"],
            local_features["roi_area"],
            event_dir / "roi_area_vs_pressure.png",
            f"{event['event_id']}: ROI area vs pressure",
            "ROI area",
        ):
            files["roi_area"] = event_dir / "roi_area_vs_pressure.png"
        pos = pd.to_numeric(local_features.get("dominant_peak_position"), errors="coerce")
        if pos.notna().sum() < 2:
            pos = pd.to_numeric(local_features.get("local_max_position"), errors="coerce")
        if plot_metric_vs_pressure(
            local_features["pressure_gpa"],
            pos,
            event_dir / "peak_center_vs_pressure.png",
            f"{event['event_id']}: peak/local center vs pressure",
            "2theta (deg)",
        ):
            files["center"] = event_dir / "peak_center_vs_pressure.png"
        if plot_metric_vs_pressure(
            local_features["pressure_gpa"],
            pd.to_numeric(local_features.get("dominant_peak_fwhm_deg"), errors="coerce"),
            event_dir / "fwhm_vs_pressure.png",
            f"{event['event_id']}: dominant FWHM vs pressure",
            "FWHM (deg)",
        ):
            files["fwhm"] = event_dir / "fwhm_vs_pressure.png"

        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
        x = pd.to_numeric(local_features["pressure_gpa"], errors="coerce")
        ax1.plot(x, pd.to_numeric(local_features.get("ncc_to_previous"), errors="coerce"), "o-", color="tab:blue", label="best-shift NCC")
        ax1.plot(x, pd.to_numeric(local_features.get("ncc_zero_to_previous"), errors="coerce"), "s--", color="tab:cyan", label="zero-shift NCC")
        ax1.set_xlabel("pressure (GPa)")
        ax1.set_ylabel("NCC")
        ax2 = ax1.twinx()
        ax2.plot(x, pd.to_numeric(local_features.get("best_shift_deg_to_previous"), errors="coerce"), "^-", color="tab:red", label="best shift")
        ax2.set_ylabel("best shift (deg)")
        ax1.set_title(f"{event['event_id']}: NCC and bounded shift summary")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(event_dir / "ncc_shift_summary.png", dpi=170)
        plt.close(fig)
        files["ncc_shift"] = event_dir / "ncc_shift_summary.png"

        if plot_metric_vs_pressure(
            local_features["pressure_gpa"],
            pd.to_numeric(local_features.get("acf_similarity_change"), errors="coerce"),
            event_dir / "acf_similarity_summary.png",
            f"{event['event_id']}: ACF similarity change vs pressure",
            "ACF similarity change",
        ):
            files["acf"] = event_dir / "acf_similarity_summary.png"

        feature_name = "static_background_score" if "static" in str(event.get("event_type", "")).lower() else "roi_area"
        crop = features[(features["cell"] == cell) & (features["window_start"] >= start - 2.0) & (features["window_end"] <= end + 2.0)]
        if not crop.empty and feature_name in crop.columns:
            pivot = crop.pivot_table(index="pressure_gpa", columns="window_label", values=feature_name, aggfunc="median")
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="viridis")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:g}" for v in pivot.index], fontsize=7)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
            ax.set_xlabel("2theta window")
            ax.set_ylabel("pressure (GPa)")
            ax.set_title(f"{event['event_id']}: pressure-window heatmap crop ({feature_name})")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            fig.tight_layout()
            fig.savefig(event_dir / "pressure_window_heatmap_crop.png", dpi=170)
            plt.close(fig)
            files["heatmap_crop"] = event_dir / "pressure_window_heatmap_crop.png"

    gids = [int(x) for x in re.findall(r"\d+", str(event.get("related_peak_group_ids", "")))]
    if gids:
        heat_dir = inputs.peak_dir / "per_peak_heatmaps"
        matches = []
        for gid in gids[:3]:
            matches.extend(sorted(heat_dir.glob(f"peak_group_{gid:03d}_*_correlation.png")))
        if matches:
            shutil.copy2(matches[0], event_dir / "relevant_peak_group_heatmap.png")
            files["peak_group_heatmap"] = event_dir / "relevant_peak_group_heatmap.png"

    raw_lines = []
    if not metrics.empty:
        for _, row in metrics.iterrows():
            raw_path = row.get("raw_2d_path", "")
            xy_path = row.get("xy_path", row.get("path", ""))
            if raw_path or xy_path:
                raw_lines.append(f"{row.get('frame_label', '')}\txy={xy_path}\traw2d={raw_path}")
    (event_dir / "raw_2d_review_paths.txt").write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""), encoding="utf-8")

    summary = textwrap.dedent(
        f"""\
        # {event['event_id']} - {event['event_type']}

        Initial interpretation: {event.get('initial_interpretation', '')}

        Why selected: {event.get('reason_selected', '')}

        Key numbers:
        - Cell: {event.get('cell', '')}
        - 2theta window: {fmt(event.get('related_window_start'))}-{fmt(event.get('related_window_end'))} deg
        - Median 2theta: {fmt(event.get('median_2theta'), 4)} deg
        - d / q: {fmt(event.get('observed_d'), 4)} A / {fmt(event.get('observed_q'), 4)} 1/A
        - Pressure range: {event.get('pressure_range', '')}
        - Related peak groups: {event.get('related_peak_group_ids', '')}
        - Static/background score: {fmt(event.get('static/background score if available'), 3)}

        Review rule: this event is a candidate. The map tells us where to look;
        the local .xy curve, raw 2D image, and reference/refinement checks decide
        whether it is sample-like, background-like, or unresolved.
        """
    )
    (event_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {name: norm_path(path) for name, path in files.items()}


def raw_inventory_from_features(features: pd.DataFrame) -> dict[tuple[str, float], str]:
    out = {}
    if features.empty or "raw_2d_path" not in features.columns:
        return out
    for _, row in features.dropna(subset=["raw_2d_path"]).iterrows():
        raw = str(row.get("raw_2d_path", ""))
        if raw:
            out[(str(row.get("cell", "")), round(safe_float(row.get("pressure_gpa")), 3))] = raw
    return out


def raw_2d_validation_index(
    catalog: pd.DataFrame,
    features: pd.DataFrame,
    raw_static: pd.DataFrame,
    auto_summary: pd.DataFrame,
) -> pd.DataFrame:
    raw_map = raw_inventory_from_features(features)
    rows = []
    high = catalog[catalog["manual_review_priority"].astype(str).str.lower().eq("high")]
    for _, event in high.iterrows():
        cell = str(event.get("cell", ""))
        p = pressure_from_text(event.get("strongest_frame", ""))
        if p is None:
            prange = re.findall(r"([0-9]+(?:\.[0-9]+)?)", str(event.get("pressure_range", "")))
            p = float(prange[0]) if prange else np.nan
        raw = raw_map.get((cell, round(p, 3)), "") if np.isfinite(p) else ""
        xy = ""
        if raw:
            local = features[(features["cell"] == cell) & np.isclose(features["pressure_gpa"], p)]
            if not local.empty:
                xy = str(local.iloc[0].get("xy_path", ""))
        spot_records = ""
        if raw and not auto_summary.empty:
            matched = auto_summary[auto_summary.astype(str).apply(lambda col: col.str.contains(re.escape(Path(raw).name), regex=True, na=False)).any(axis=1)]
            if not matched.empty:
                candidates = []
                for col in matched.columns:
                    if "spots" in col.lower() and matched.iloc[0].get(col, ""):
                        candidates.append(str(matched.iloc[0].get(col)))
                spot_records = ";".join(candidates[:4])
        raw_note, _ = raw2d_evidence(raw_static, pd.DataFrame(), cell, safe_float(event.get("median_2theta")))
        rows.append(
            {
                "event_id": event["event_id"],
                "frame_label": event.get("strongest_frame", ""),
                "source_xy_file": xy,
                "candidate_raw_TIFF_path": raw,
                "expected_2theta": event.get("median_2theta", ""),
                "expected_radius": "",
                "classification_exists": bool(raw_note),
                "relevant_spot_ring_csv_records": spot_records,
                "notes": raw_note if raw else "No reliable raw 2D match from pressure-window table.",
            }
        )
    return pd.DataFrame(rows)


def read_image_for_contact(path: str) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    try:
        img = plt.imread(path)
        arr = np.asarray(img, dtype=float)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.where(np.isfinite(arr), arr, np.nan)
        arr[np.abs(arr) > 1e7] = np.nan
        return arr
    except Exception:
        return None


def make_contact_sheet(raw_index: pd.DataFrame, out_path: Path) -> None:
    rows = raw_index.dropna(subset=["candidate_raw_TIFF_path"]).copy() if not raw_index.empty else pd.DataFrame()
    rows = rows[rows["candidate_raw_TIFF_path"].astype(str) != ""].drop_duplicates("candidate_raw_TIFF_path").head(8)
    n = max(len(rows), 1)
    cols = min(4, n)
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.6 * rows_n), squeeze=False)
    axes_flat = axes.ravel()
    if rows.empty:
        axes_flat[0].text(0.5, 0.5, "No raw TIFFs could be matched reliably.", ha="center", va="center")
        axes_flat[0].axis("off")
    else:
        for ax, (_, row) in zip(axes_flat, rows.iterrows()):
            img = read_image_for_contact(str(row["candidate_raw_TIFF_path"]))
            if img is None:
                ax.text(0.5, 0.5, "image load failed", ha="center", va="center")
                ax.axis("off")
                continue
            lo, hi = np.nanpercentile(img, [2, 99.5])
            show = np.clip(img, lo, hi)
            ax.imshow(show, cmap="gray", origin="upper")
            ax.set_title(f"{row['event_id']} | {row.get('frame_label', '')}\n2theta~{fmt(row.get('expected_2theta'), 3)} deg", fontsize=9)
            ax.axis("off")
        for ax in axes_flat[len(rows) :]:
            ax.axis("off")
    fig.suptitle("Manual raw-2D review contact sheet", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def overview_figures(
    out_dir: Path,
    catalog: pd.DataFrame,
    features: pd.DataFrame,
    clusters: pd.DataFrame,
    raw_index: pd.DataFrame,
) -> dict[str, Path]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = catalog["event_type"].value_counts()
    axes[0].barh(range(len(counts)), counts.values, color="tab:blue")
    axes[0].set_yticks(range(len(counts)))
    axes[0].set_yticklabels(counts.index, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Candidate events by category")
    pr = catalog["manual_review_priority"].value_counts()
    axes[1].bar(pr.index, pr.values, color=["tab:red" if p == "High" else "tab:orange" for p in pr.index])
    axes[1].set_title("Manual review priority")
    axes[1].set_ylabel("event count")
    fig.tight_layout()
    paths["candidate_event_overview"] = fig_dir / "candidate_event_overview.png"
    fig.savefig(paths["candidate_event_overview"], dpi=170)
    plt.close(fig)

    if not features.empty:
        for cell, local in features.groupby("cell"):
            pivot = local.pivot_table(index="pressure_gpa", columns="window_label", values="static_background_score", aggfunc="median")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 5.5))
            im = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="magma")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:g}" for v in pivot.index], fontsize=8)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
            ax.set_xlabel("2theta window")
            ax.set_ylabel("pressure (GPa)")
            ax.set_title(f"{cell}: event navigation map (static/background score with event markers)")
            for _, ev in catalog[catalog["cell"] == cell].iterrows():
                label = f"{fmt(ev.get('related_window_start'), 2)}-{fmt(ev.get('related_window_end'), 2)}"
                if label in pivot.columns:
                    ax.scatter([list(pivot.columns).index(label)], [len(pivot.index) - 1], s=60, facecolors="none", edgecolors="cyan")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            fig.tight_layout()
            # One combined path is requested; write the first cell to the canonical name and cells individually.
            cell_path = fig_dir / f"pressure_window_event_map_{cell}.png"
            fig.savefig(cell_path, dpi=170)
            if "pressure_window_event_map" not in paths:
                shutil.copy2(cell_path, fig_dir / "pressure_window_event_map.png")
                paths["pressure_window_event_map"] = fig_dir / "pressure_window_event_map.png"
            plt.close(fig)

    def bar_for_events(mask, path_name, title, value_col):
        sub = catalog[mask].copy().head(8)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        if sub.empty:
            ax.text(0.5, 0.5, "No events in this category.", ha="center", va="center")
            ax.axis("off")
        else:
            labels = sub["event_id"] + "\n" + sub["cell"].astype(str) + " " + sub["median_2theta"].map(lambda v: fmt(v, 2))
            vals = pd.to_numeric(sub[value_col], errors="coerce").fillna(0)
            ax.bar(range(len(sub)), vals, color="tab:green")
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(value_col)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        paths[path_name] = fig_dir / f"{path_name}.png"
        fig.savefig(paths[path_name], dpi=170)
        plt.close(fig)

    bar_for_events(catalog["event_type"].str.contains("shift|recurring", case=False, na=False), "top_shifting_events", "Top shifting / recurring candidates", "position_shift")
    bar_for_events(catalog["event_type"].str.contains("appearance|disappearance", case=False, na=False), "top_appearing_disappearing_events", "Appearance / disappearance candidates", "max_roi_area")
    bar_for_events(catalog["event_type"].str.contains("static", case=False, na=False), "likely_static_background_events", "Likely static/background candidates", "static/background score if available")
    bar_for_events(catalog["event_type"].str.contains("Tier-B", case=False, na=False), "tier_b_dependent_weak_events", "Tier-B-dependent weak trajectories", "Tier B count")

    fig, ax = plt.subplots(figsize=(10, 5))
    if clusters.empty:
        ax.text(0.5, 0.5, "No multi-peak co-varying clusters passed the simple grouping rule.", ha="center", va="center")
        ax.axis("off")
    else:
        top = clusters.drop_duplicates("cluster_id").sort_values("cluster_size", ascending=False).head(12)
        ax.barh(range(len(top)), top["cluster_size"], color="tab:purple")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["cluster_id"] + " | " + top["cluster_label"].astype(str), fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("peak groups in cluster")
        ax.set_title("Candidate co-varying peak-group clusters")
    fig.tight_layout()
    paths["co_varying_peak_group_clusters"] = fig_dir / "co_varying_peak_group_clusters.png"
    fig.savefig(paths["co_varying_peak_group_clusters"], dpi=170)
    plt.close(fig)

    make_contact_sheet(raw_index, fig_dir / "manual_review_contact_sheet.png")
    paths["manual_review_contact_sheet"] = fig_dir / "manual_review_contact_sheet.png"
    return paths


def write_manifest(
    out_dir: Path,
    inputs: Inputs,
    features: pd.DataFrame,
    window_summary: pd.DataFrame,
    peaks: pd.DataFrame,
    groups: pd.DataFrame,
    wavelength_a: float | None,
    wavelength_source: str,
    git_status: str,
) -> None:
    pressure_order = []
    if not features.empty:
        for cell, local in features.groupby("cell"):
            vals = sorted(pd.to_numeric(local["pressure_gpa"], errors="coerce").dropna().unique())
            pressure_order.append(f"{cell}: " + ", ".join(f"{v:g}" for v in vals))
    compression_note = "XDI pressure-window suite excludes decomp by default; peak-group input still contains decomp rows when present."
    window_width = ""
    window_step = ""
    if not window_summary.empty:
        widths = (pd.to_numeric(window_summary["window_end"], errors="coerce") - pd.to_numeric(window_summary["window_start"], errors="coerce")).dropna().unique()
        starts = sorted(pd.to_numeric(window_summary["window_start"], errors="coerce").dropna().unique())
        steps = np.diff(starts)
        window_width = ", ".join(f"{v:g}" for v in sorted(set(np.round(widths, 5))))
        window_step = ", ".join(f"{v:g}" for v in sorted(set(np.round(steps[steps > 0], 5)))) if len(steps) else ""
    rows = [
        ("input_suite_path", norm_path(inputs.xdi_suite)),
        ("input_peak_group_suite_path", norm_path(inputs.peak_suite)),
        ("pressure_window_map_folder", norm_path(inputs.xdi_suite / "heatmaps")),
        ("pressure_order_used", " | ".join(pressure_order)),
        ("compression_decompression_handling", compression_note),
        ("window_width_deg", window_width),
        ("window_step_deg", window_step),
        ("NCC_shift_settings", "read from XDI suite; max_shift_deg=0.35 deg in same_window_cross_pressure_index.csv when available"),
        ("ACF_settings", "read from XDI suite; within-frame all-window and nonoverlap ACF matrices used as navigation evidence"),
        ("peak_detector_version", "existing high_recall_scored_v2 Tier A/B/C tables; detector not tuned or rerun"),
        ("wavelength_A", fmt(wavelength_a, 5)),
        ("wavelength_source", wavelength_source),
        ("date_time", datetime.now().isoformat(timespec="seconds")),
        ("feature_rows", str(len(features))),
        ("peak_candidate_rows", str(len(peaks))),
        ("peak_groups", str(len(groups))),
    ]
    with (out_dir / "run_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        writer.writerows(rows)
    (out_dir / "git_status_short.txt").write_text(git_status + "\n", encoding="utf-8")


def write_state_summary(
    out_dir: Path,
    features: pd.DataFrame,
    peaks: pd.DataFrame,
    groups: pd.DataFrame,
    reference_files: pd.DataFrame,
) -> None:
    tier_cols = [c for c in peaks.columns if "tier" in c.lower()]
    feature_groups = {
        "ROI columns": [c for c in features.columns if "roi" in c.lower() or "area" in c.lower()],
        "NCC columns": [c for c in features.columns if "ncc" in c.lower() or "shift" in c.lower()],
        "ACF columns": [c for c in features.columns if "acf" in c.lower()],
        "FWHM columns": [c for c in features.columns if "fwhm" in c.lower()],
        "center/position columns": [c for c in features.columns if "position" in c.lower() or "center" in c.lower()],
    }
    raw_paths = features.get("raw_2d_path", pd.Series(dtype=str)).replace("", np.nan).dropna().astype(str).unique()
    rows = [
        ("number_of_frames", str(features[["cell", "frame_label"]].drop_duplicates().shape[0]) if not features.empty else "0"),
        ("number_of_pressure_window_features", str(len(features))),
        ("number_of_peak_groups", str(len(groups))),
        ("available_tier_columns", "; ".join(tier_cols)),
        ("available_raw_2d_image_paths", str(len(raw_paths))),
        ("available_reference_or_refinement_files", str(len(reference_files))),
    ]
    for label, cols in feature_groups.items():
        rows.append((label, "; ".join(cols)))
    with (out_dir / "analysis_state_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "summary"])
        writer.writerows(rows)


def markdown_image(path: Path, out_dir: Path, title: str) -> str:
    try:
        rel = path.relative_to(out_dir)
    except ValueError:
        rel = path
    return f"![{title}]({rel.as_posix()})"


def write_report(
    out_dir: Path,
    catalog: pd.DataFrame,
    assignments: pd.DataFrame,
    clusters: pd.DataFrame,
    raw_index: pd.DataFrame,
    static_every: pd.DataFrame,
    fig_paths: dict[str, Path],
    event_files: dict[str, dict],
    inputs: Inputs,
    wavelength_a: float | None,
    missing_notes: list[str],
) -> Path:
    high = catalog[catalog["manual_review_priority"].astype(str).str.lower().eq("high")]
    static_events = catalog[catalog["event_type"].astype(str).str.contains("static", case=False, na=False)]
    tier_b_events = catalog[catalog["event_type"].astype(str).str.contains("Tier-B", case=False, na=False)]
    sample_like = assignments[assignments["assignment_confidence"].isin(["strong_sample_candidate", "possible_sample_candidate"])]
    background_like = assignments[assignments["assignment_confidence"].isin(["possible_background", "likely_static_background", "likely_diamond_or_cell", "likely_gasket_or_medium"])]

    sections = []
    sections.append("# Candidate Event Validation Report\n")
    sections.append("## 1. Executive summary\n")
    sections.append(
        "This report turns the XDI-inspired maps into reviewable candidate events. "
        "The maps are used as a navigation layer: they point to pressure/window regions that deserve inspection. "
        "They do not prove phase identity by themselves.\n"
    )
    sections.append(
        f"- Input XDI suite: `{inputs.xdi_suite}`\n"
        f"- Input peak-group suite: `{inputs.peak_suite}`\n"
        f"- Candidate events selected: {len(catalog)}\n"
        f"- High-priority manual checks: {len(high)}\n"
        f"- Wavelength used for d/q: {fmt(wavelength_a, 5) if wavelength_a else 'missing'} A\n"
    )
    sections.append(markdown_image(fig_paths["candidate_event_overview"], out_dir, "Candidate event overview") + "\n")

    sections.append("## 2. Input suite and methodology\n")
    sections.append(
        "Each integrated `.xy` scan is treated as an I(2theta) curve. "
        "Each sliding 2theta window is treated as an ROI. Existing ROI, Tier A/B/C counts, NCC/shift, ACF, FWHM, and static-score columns are read from the completed XDI suite. "
        "The peak detector was not changed. Candidate events were then selected from multiple categories so the catalog is not biased toward only the strongest peaks.\n"
    )

    sections.append("## 3. Pressure ordering\n")
    if not catalog.empty:
        for cell in sorted(catalog["cell"].dropna().unique()):
            fs = read_csv_optional(inputs.xdi_suite / "tables" / "pressure_window_feature_table.csv")
            vals = sorted(pd.to_numeric(fs[fs["cell"] == cell]["pressure_gpa"], errors="coerce").dropna().unique()) if not fs.empty else []
            sections.append(f"- {cell}: " + ", ".join(f"{v:g} GPa" for v in vals) + "\n")
    sections.append("Decompression scans are not part of the pressure-window maps, but decompression detections in the peak-group table are used for the hysteresis candidate.\n")

    sections.append("## 4. Candidate event categories\n")
    for event_type, count in catalog["event_type"].value_counts().items():
        sections.append(f"- {event_type}: {count}\n")

    sections.append("## 5. Top sample-like pressure-dependent candidates\n")
    if sample_like.empty:
        sections.append("No event is assigned as a strong sample candidate from map evidence alone. Several remain possible sample-related features.\n")
    else:
        for _, row in sample_like.head(8).iterrows():
            sections.append(f"- {row.event_id}: {row.assignment_confidence}; {row.candidate_assignment}; notes: {row.notes}\n")
    sections.append(markdown_image(fig_paths["top_shifting_events"], out_dir, "Top shifting events") + "\n")

    sections.append("## 6. Appearing/disappearing features\n")
    app = catalog[catalog["event_type"].str.contains("appearance|disappearance", case=False, na=False)]
    for _, row in app.iterrows():
        sections.append(f"- {row.event_id}: {row.event_type}, {row.cell}, {fmt(row.median_2theta, 3)} deg. {row.reason_selected}\n")
    sections.append(markdown_image(fig_paths["top_appearing_disappearing_events"], out_dir, "Appearing/disappearing events") + "\n")

    sections.append("## 7. Systematically shifting features\n")
    shifting = catalog[catalog["event_type"].str.contains("shift|recurring", case=False, na=False)]
    for _, row in shifting.head(8).iterrows():
        sections.append(f"- {row.event_id}: {row.cell}, 2theta~{fmt(row.median_2theta, 3)} deg, shift span {fmt(row.position_shift, 3)} deg.\n")

    sections.append("## 8. FWHM / broadening anomalies\n")
    broad = catalog[catalog["event_type"].str.contains("FWHM|broadening", case=False, na=False)]
    for _, row in broad.iterrows():
        sections.append(f"- {row.event_id}: FWHM change {fmt(row.FWHM_change, 4)} deg; check if this is true broadening or overlap/background.\n")

    sections.append("## 9. NCC/ACF window-level structural changes\n")
    ncc = catalog[catalog["event_type"].str.contains("NCC|ACF", case=False, na=False)]
    for _, row in ncc.iterrows():
        sections.append(f"- {row.event_id}: {row.NCC_summary} {row.ACF_summary}\n")
    if "pressure_window_event_map" in fig_paths:
        sections.append(markdown_image(fig_paths["pressure_window_event_map"], out_dir, "Pressure-window event map") + "\n")

    sections.append("## 10. Tier-B-dependent weak signals\n")
    if tier_b_events.empty:
        sections.append("No Tier-B-dependent weak trajectory was selected.\n")
    else:
        for _, row in tier_b_events.iterrows():
            sections.append(f"- {row.event_id}: group(s) {row.related_peak_group_ids}, 2theta~{fmt(row.median_2theta, 3)} deg; needs manual confirmation.\n")
    sections.append(markdown_image(fig_paths["tier_b_dependent_weak_events"], out_dir, "Tier-B weak events") + "\n")

    sections.append("## 11. Likely static/background/diamond/gasket candidates\n")
    if background_like.empty:
        sections.append("No event can be confidently assigned to background solely from the maps.\n")
    else:
        for _, row in background_like.head(10).iterrows():
            sections.append(f"- {row.event_id}: {row.assignment_confidence}; {row.candidate_assignment}; raw-2D evidence: {row.raw_2d_evidence or 'none'}\n")
    sections.append(
        "Static rule used here: high zero-shift NCC, small best shift, small pressure slope, persistent signal, then raw-2D centroid confirmation. "
        "Correlation can nominate a peak/window; raw 2D is the authority before masking.\n"
    )
    if not static_every.empty:
        strict = static_every[static_every["candidate_kind"] == "strict_peak_group_all_compression_frames"]
        confirmed = static_every[static_every["candidate_kind"] == "raw2d_confirmed_static_spot"]
        sections.append(
            f"Strict all-compression-frame 1D peak-group candidates found: {len(strict)}. "
            f"Raw-2D confirmed static spots found: {len(confirmed)}.\n"
        )
        for _, row in static_every.head(12).iterrows():
            sections.append(f"- {row.candidate_kind}: {row.cell}, 2theta~{fmt(row.median_2theta, 4)} deg, notes: {row.notes}\n")
    sections.append(markdown_image(fig_paths["likely_static_background_events"], out_dir, "Static/background events") + "\n")

    sections.append("## 12. Reference assignment results\n")
    for conf, count in assignments["assignment_confidence"].value_counts().items():
        sections.append(f"- {conf}: {count}\n")
    sections.append("Assignments are intentionally cautious. A nearby reference line is not enough; pressure behavior and raw 2D evidence must agree.\n")

    sections.append("## 13. Raw 2D validation status\n")
    if raw_index.empty:
        sections.append("No high-priority event could be matched to raw 2D evidence from the available tables.\n")
    else:
        matched = raw_index["candidate_raw_TIFF_path"].astype(str).ne("").sum()
        sections.append(f"High-priority events in raw-image review index: {len(raw_index)}; matched raw TIFF paths: {matched}.\n")
    sections.append(markdown_image(fig_paths["manual_review_contact_sheet"], out_dir, "Manual review contact sheet") + "\n")

    sections.append("## 14. Co-varying peak-group clusters\n")
    if clusters.empty:
        sections.append("No multi-peak clusters passed the simple co-variation grouping rule.\n")
    else:
        top = clusters.drop_duplicates("cluster_id").sort_values("cluster_size", ascending=False).head(8)
        for _, row in top.iterrows():
            sections.append(f"- {row.cluster_id}: size {row.cluster_size}, {row.cluster_label}. Candidate correlated set only.\n")
    sections.append(markdown_image(fig_paths["co_varying_peak_group_clusters"], out_dir, "Co-varying clusters") + "\n")

    sections.append("## 15. Recommended manual checks\n")
    for _, row in high.head(10).iterrows():
        sections.append(f"- {row.event_id}: inspect local .xy, raw 2D at `{row.strongest_frame}`, and nearest reference/refinement lines.\n")

    sections.append("## 16. Main limitations\n")
    sections.extend(f"- {note}\n" for note in missing_notes)
    sections.append("- Integrated 1D curves can merge sample, diamond, gasket, pressure-medium, and detector features.\n")
    sections.append("- A single event is weaker than a co-varying set of peaks or a refinement/EOS trend.\n")
    sections.append("- Cell_14 static-spot classification remains underpowered compared with Cell_29.\n")

    sections.append("## 17. Suggested next scientific actions\n")
    sections.append("- Manually review the high-priority event folders first, especially local XY plots and raw TIFF contact sheets.\n")
    sections.append("- Promote only events with consistent local XY, raw-2D, and reference/refinement behavior into the paper narrative.\n")
    sections.append("- Use confirmed static/background candidates as mask candidates, then rerun physical XRD analysis only after raw-2D confirmation.\n")

    sections.append("## Event evidence snapshots\n")
    for _, row in catalog.head(8).iterrows():
        eid = row.event_id
        files = event_files.get(eid, {})
        if "local_xy" in files:
            sections.append(f"### {eid}: {row.event_type}\n")
            sections.append(markdown_image(Path(files["local_xy"]), out_dir, f"{eid} local XY") + "\n")

    report = out_dir / "candidate_event_validation_report.md"
    report.write_text("\n".join(sections), encoding="utf-8")
    return report


def add_pdf_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.95
    fig.text(0.08, y, title, fontsize=16, weight="bold", va="top")
    y -= 0.05
    for para in paragraphs:
        wrapped = textwrap.wrap(para, width=88)
        for line in wrapped:
            fig.text(0.08, y, line, fontsize=9.5, va="top")
            y -= 0.018
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                y = 0.95
        y -= 0.012
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_pdf_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")
    try:
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=14, pad=12)
    except Exception:
        ax.text(0.5, 0.5, f"Could not load {image_path}", ha="center", va="center")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(
    out_dir: Path,
    catalog: pd.DataFrame,
    assignments: pd.DataFrame,
    fig_paths: dict[str, Path],
    event_files: dict[str, dict],
    missing_notes: list[str],
) -> Path:
    pdf_path = out_dir / "candidate_event_validation_report.pdf"
    high = catalog[catalog["manual_review_priority"].astype(str).str.lower().eq("high")]
    with PdfPages(pdf_path) as pdf:
        add_pdf_text_page(
            pdf,
            "Candidate Event Validation Report",
            [
                f"Candidate events selected: {len(catalog)}.",
                f"High-priority manual checks: {len(high)}.",
                "Plain-language summary: the maps are a navigation system. They tell us which pressure and 2theta windows look interesting. They do not convict a peak as sample or background by themselves.",
                "Static-peak rule: high zero-shift NCC, small best shift, small pressure slope, persistent signal, then raw-2D centroid confirmation before masking.",
                "Allowed interpretation language: possible sample-related feature, likely static/background, candidate pressure-dependent event, unresolved but interesting.",
            ],
        )
        for key in [
            "candidate_event_overview",
            "pressure_window_event_map",
            "top_shifting_events",
            "top_appearing_disappearing_events",
            "likely_static_background_events",
            "tier_b_dependent_weak_events",
            "co_varying_peak_group_clusters",
            "manual_review_contact_sheet",
        ]:
            if key in fig_paths and Path(fig_paths[key]).exists():
                add_pdf_image_page(pdf, fig_paths[key], key.replace("_", " ").title())
        for _, row in catalog.head(8).iterrows():
            files = event_files.get(row.event_id, {})
            if "local_xy" in files:
                add_pdf_image_page(pdf, Path(files["local_xy"]), f"{row.event_id}: local XY pressure series")
        paragraphs = []
        for _, row in assignments.head(12).iterrows():
            paragraphs.append(
                f"{row.event_id}: {row.assignment_confidence}; assignment={row.candidate_assignment}; notes={row.notes}"
            )
        if missing_notes:
            paragraphs.extend(["Limitations:"] + missing_notes)
        add_pdf_text_page(pdf, "Reference Assignments And Limitations", paragraphs)
    return pdf_path


def cleanup_recommendation(out_dir: Path) -> pd.DataFrame:
    keep_exact = {
        "analysis_v2_20260701": "keep: raw-2D static/moving evidence and EOS/reference tables feed the current validation",
        "correlation_suite_20260621_high_recall_scored_v2": "keep: current peak-group/Tier A-B-C input suite",
        "final_interpretable_xrd_20260703": "keep: current final synthesis package",
        "xdi_pressure_window_maps_20260703": "keep: latest XDI pressure-window maps and this validation output",
        "auto_ring_filter_batch_poni_microz4p2_ringw3": "keep: auto-ring-filter output requested by user",
        "auto_ring_filter_batch_radius_groups_cell29_10deg_by_pressure_legend_outside": "keep: auto-ring-filter output requested by user and raw spot CSV links",
    }
    rows = []
    for path in sorted(Path("outputs").iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if name in keep_exact:
            action = "keep"
            reason = keep_exact[name]
        elif name.startswith("correlation_suite_") and name != "correlation_suite_20260621_high_recall_scored_v2":
            action = "delete_candidate"
            reason = "older correlation suite superseded by high_recall_scored_v2 and XDI validation"
        elif name.startswith("_validation") or name.startswith("diagnostics_"):
            action = "delete_candidate"
            reason = "older diagnostic/smoke output now summarized by current validation"
        elif name.startswith("peak_trajectories_"):
            action = "keep"
            reason = "keep for now: useful supporting trajectory source even if not final paper package"
        else:
            action = "review"
            reason = "not automatically classified"
        rows.append({"path": norm_path(path), "folder": name, "recommended_action": action, "reason": reason})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "output_cleanup_recommendation.csv", index=False)
    return df


def main() -> None:
    args = parse_args()
    xdi_suite = latest_xdi_suite(args.xdi_suite)
    peak_suite = args.peak_suite
    peak_dir = peak_dir_for_suite(peak_suite)
    inputs = Inputs(xdi_suite=xdi_suite, peak_suite=peak_suite, peak_dir=peak_dir, analysis_v2=args.analysis_v2)
    out_dir = unique_output_dir(xdi_suite / EVENT_DIR_NAME)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "events").mkdir(parents=True, exist_ok=True)

    features = read_csv_optional(xdi_suite / "tables" / "pressure_window_feature_table.csv")
    window_summary = read_csv_optional(xdi_suite / "tables" / "window_motion_summary.csv")
    peaks = prepare_peaks(read_csv_optional(peak_dir / "all_candidate_table.csv"))
    group_summary = read_csv_optional(peak_dir / "peak_group_summary.csv")
    groups = group_metrics(peaks, group_summary)

    wavelength_a, wavelength_source = read_wavelength_a(args.poni)
    known_refs = known_reference_lines(wavelength_a)
    reference_files = reference_file_inventory()
    refinement_refs = parse_refinement_reflections(wavelength_a)
    raw_static = read_csv_optional(args.analysis_v2 / "F_raw2d_spot_tracking" / "spot_static_verdicts.csv")
    class_table = read_csv_optional(args.analysis_v2 / "B_2d_static_diamond" / "classification_table.csv")
    peak_identification = read_csv_optional(args.analysis_v2 / "C_dspacing_eos" / "peak_identification.csv")
    auto_summary = read_csv_optional(Path("outputs/auto_ring_filter_batch_radius_groups_cell29_10deg_by_pressure_legend_outside/summary.csv"))

    reference_files.to_csv(out_dir / "tables" / "reference_file_inventory.csv", index=False)
    known_refs.to_csv(out_dir / "tables" / "known_background_reference_lines.csv", index=False)
    refinement_refs.to_csv(out_dir / "tables" / "refinement_predicted_reflections.csv", index=False)
    groups.to_csv(out_dir / "tables" / "peak_group_scientific_metrics.csv", index=False)

    catalog = select_events(inputs, features, window_summary, peaks, groups, raw_static, args.max_events)
    catalog = add_catalog_dq(catalog, wavelength_a)
    catalog.to_csv(out_dir / "candidate_event_catalog.csv", index=False)

    assignments = reference_assignments(catalog, known_refs, refinement_refs, peak_identification, raw_static, class_table)
    assignments.to_csv(out_dir / "event_reference_assignment.csv", index=False)

    clusters = build_covarying_clusters(peaks, groups)
    clusters.to_csv(out_dir / "co_varying_peak_group_clusters.csv", index=False)

    static_every = static_every_frame_candidates(peaks, features, raw_static, wavelength_a)
    static_every.to_csv(out_dir / "static_peak_candidates_every_frame.csv", index=False)

    raw_index = raw_2d_validation_index(catalog, features, raw_static, auto_summary)
    raw_index.to_csv(out_dir / "raw_2d_validation_index.csv", index=False)

    event_files = {}
    for _, event in catalog.iterrows():
        event_files[event["event_id"]] = create_event_package(
            event,
            out_dir,
            inputs,
            features,
            peaks,
            known_refs,
            wavelength_a,
        )
    pd.DataFrame(
        [{"event_id": eid, "artifact": name, "path": path} for eid, files in event_files.items() for name, path in files.items()]
    ).to_csv(out_dir / "event_artifact_index.csv", index=False)

    fig_paths = overview_figures(out_dir, catalog, features, clusters, raw_index)
    cleanup_df = cleanup_recommendation(out_dir)

    missing_notes = []
    if wavelength_a is None:
        missing_notes.append("Wavelength metadata was not found, so d-spacing and q columns are blank.")
    if refinement_refs.empty:
        missing_notes.append("No parseable refinement reflection list was found; assignment relies on known reference lines and existing identification tables.")
    if raw_index.empty or raw_index["candidate_raw_TIFF_path"].astype(str).eq("").all():
        missing_notes.append("Raw 2D image paths could not be reliably matched for high-priority events.")
    if raw_static.empty:
        missing_notes.append("Raw-2D static verdict table is missing; static calls must remain correlation-only.")

    report_md = write_report(
        out_dir,
        catalog,
        assignments,
        clusters,
        raw_index,
        static_every,
        fig_paths,
        event_files,
        inputs,
        wavelength_a,
        missing_notes,
    )
    report_pdf = write_pdf_report(out_dir, catalog, assignments, fig_paths, event_files, missing_notes)

    git_status = run_git_status()
    write_manifest(out_dir, inputs, features, window_summary, peaks, groups, wavelength_a, wavelength_source, git_status)
    write_state_summary(out_dir, features, peaks, groups, reference_files)

    summary_rows = [
        ("output_dir", norm_path(out_dir)),
        ("input_xdi_suite", norm_path(xdi_suite)),
        ("input_peak_suite", norm_path(peak_suite)),
        ("candidate_events", str(len(catalog))),
        ("event_categories", str(catalog["event_type"].nunique())),
        ("raw2d_review_rows", str(len(raw_index))),
        ("reference_assignments", str(len(assignments))),
        ("co_varying_cluster_rows", str(len(clusters))),
        ("static_every_frame_rows", str(len(static_every))),
        ("cleanup_delete_candidates", str((cleanup_df["recommended_action"] == "delete_candidate").sum())),
        ("report_md", norm_path(report_md)),
        ("report_pdf", norm_path(report_pdf)),
    ]
    with (out_dir / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        writer.writerows(summary_rows)

    print(f"Input XDI suite: {xdi_suite}")
    print(f"Output folder: {out_dir}")
    print(f"Candidate events: {len(catalog)}")
    print(f"Report: {report_md}")
    print(f"PDF: {report_pdf}")


if __name__ == "__main__":
    main()
