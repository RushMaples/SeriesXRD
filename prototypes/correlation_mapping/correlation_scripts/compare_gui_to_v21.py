#!/usr/bin/env python3
"""Auditable BulkXRD-GUI cross-check for uniform-correlation-v2.1.

This program is deliberately a *validation* layer.  It never changes peak
tracking, fitting, window definitions, thresholds, or any result file.  It
reads one or more BulkXRD analysis HDF5 files (the data behind GUI Peak map and
Pattern map), compares them with the frozen correlation outputs, and writes
plain CSV/JSON evidence.

Peak-map comparison
-------------------
Both GUI views requested by the user are reproduced:

``all_peaks``
    GUI Peak map with "Good peaks only" off, compared with every finite
    ``detected_peak_fits.csv`` row.

``good_only``
    GUI ``flag == 0`` rows, compared with v2/v2.1 ``state == reliable`` rows.

Within each exact ``channel + scan + pressure`` group, matches are one-to-one
(Hungarian assignment).  A pair is admissible only when its fitted profiles
overlap under this fixed, data-independent formula::

    |center_gui - center_corr| <= max(
        0.5 * (FWHM_gui + FWHM_corr),
        2 * GUI_radial_grid_step,
    )

The formula is reported verbatim in ``crosscheck_summary.json``.  It is a QC
matching rule, not a correlation-analysis parameter.

Window comparison
-----------------
For every v2.1 strict window, adjacent-pressure cells are extracted together
with support and scan-bootstrap confidence limits.  A rank-based
``low-similarity candidate`` is the lowest-similarity quartile *within that
window and family*.  It is not called a physical boundary.  A candidate is
``corroborated`` only when ACF-strict and direct-strict select the same pressure
interval; one-family-only selections remain unresolved.  No result is tuned to
make the calls agree.  The exact same 2theta window is sampled from each GUI
Pattern map and its direct/ACF Pearson values and simple peak-change diagnostics
are written to ``pattern_window_checks.csv``.

The fit/tungsten channel is always summarized from the correlation run.  If no
GUI fit HDF5 is supplied, that absence is explicit and the GUI verdict remains
partial rather than being silently treated as a pass.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


FLAG_DEFINITIONS: tuple[tuple[int, str], ...] = (
    (1, "low_amp"),
    (2, "bad_chi2"),
    (4, "center_drift"),
    (8, "width_bound"),
    (16, "no_converge"),
)
MATCH_FORMULA = (
    "abs(center_gui-center_correlation) <= "
    "max(0.5*(FWHM_gui+FWHM_correlation), 2*GUI_radial_grid_step)"
)
BOUNDARY_FORMULA = (
    "low-similarity candidate only: within each window/family, "
    "adjacent-pressure similarity <= finite Q25; corroborated only when "
    "ACF-strict and direct-strict select the exact same interval"
)
FIXED_VISUALIZATION_FILENAMES = {
    "all_area": "gui_peak_map_pressure_all_area.png",
    "good_area": "gui_peak_map_pressure_good_only_area.png",
    "all_fwhm": "gui_peak_map_pressure_all_fwhm.png",
    "good_fwhm": "gui_peak_map_pressure_good_only_fwhm.png",
    "pattern_clean": "gui_pattern_map_pressure_clean.png",
    "matched_overlay": "gui_v21_matched_peak_overlay_pressure.png",
}


@dataclass(frozen=True)
class GuiDataset:
    channel: str
    scan: str
    path: Path
    pattern_source: str
    unit: str
    wavelength_A: float
    radial_deg: np.ndarray
    radial_step_deg: float
    pressure_GPa: np.ndarray
    filenames: tuple[str, ...]
    excluded: np.ndarray
    patterns: np.ndarray
    peaks: pd.DataFrame


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


def _flag_reason(flag: int) -> str:
    value = int(flag)
    if value == 0:
        return "good"
    names = [name for bit, name in FLAG_DEFINITIONS if value & bit]
    unknown = value & ~sum(bit for bit, _ in FLAG_DEFINITIONS)
    if unknown:
        names.append(f"unknown_bits_{unknown}")
    return ";".join(names) if names else f"flag_{value}"


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _q_to_two_theta(q_A_inv: np.ndarray, wavelength_A: float) -> np.ndarray:
    q = np.asarray(q_A_inv, dtype=float)
    arg = q * float(wavelength_A) / (4.0 * np.pi)
    result = np.full(q.shape, np.nan, dtype=float)
    valid = np.isfinite(arg) & (np.abs(arg) <= 1.0)
    result[valid] = np.degrees(2.0 * np.arcsin(arg[valid]))
    return result


def _radial_as_two_theta(
    values: np.ndarray,
    fwhm: np.ndarray,
    unit: str,
    wavelength_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert common BulkXRD radial units and their widths to 2theta degrees."""

    center = np.asarray(values, dtype=float)
    width = np.asarray(fwhm, dtype=float)
    normalized = str(unit).strip().lower()
    if normalized in {"2th_deg", "2theta_deg", "2theta", "degrees", "deg"}:
        return center, width
    if normalized in {"2th_rad", "2theta_rad", "radians", "rad"}:
        return np.degrees(center), np.degrees(width)
    if normalized in {"q_a^-1", "q_a-1", "q_a", "q"}:
        if not np.isfinite(wavelength_A) or wavelength_A <= 0:
            raise ValueError("GUI q-axis requires a finite positive wavelength")
        lo = _q_to_two_theta(np.maximum(center - 0.5 * width, 0.0), wavelength_A)
        hi = _q_to_two_theta(center + 0.5 * width, wavelength_A)
        return _q_to_two_theta(center, wavelength_A), hi - lo
    if normalized in {"q_nm^-1", "q_nm-1", "q_nm"}:
        return _radial_as_two_theta(center * 0.1, width * 0.1, "q_A^-1", wavelength_A)
    raise ValueError(f"unsupported GUI radial unit {unit!r}")


def _read_pattern_source(h5: h5py.File, source: str) -> np.ndarray:
    wanted = str(source or "clean").strip().lower()
    if wanted == "residual":
        if "residual/clean" not in h5:
            raise ValueError("Pattern map source residual requested but /residual/clean is absent")
        return np.asarray(h5["residual/clean"][:], dtype=float)
    bg = h5.get("background")
    if bg is None or "clean" not in bg:
        raise ValueError("GUI analysis HDF5 has no /background/clean")
    clean = np.asarray(bg["clean"][:], dtype=float)
    if wanted == "clean":
        return clean
    if wanted == "baseline":
        return np.asarray(bg["baseline"][:], dtype=float)
    if wanted == "spot_residual":
        return np.asarray(bg["spot_residual"][:], dtype=float)
    if wanted == "robust":
        return clean + np.asarray(bg["baseline"][:], dtype=float)
    if wanted == "mean":
        return (
            clean
            + np.asarray(bg["baseline"][:], dtype=float)
            + np.asarray(bg["spot_residual"][:], dtype=float)
        )
    if wanted == "sigmaclip":
        if "sigmaclip_residual" not in bg:
            raise ValueError("Pattern map source sigmaclip is absent in GUI HDF5")
        return clean + np.asarray(bg["sigmaclip_residual"][:], dtype=float)
    raise ValueError(
        f"Pattern map source {wanted!r} is not supported by this read-only checker; "
        "use clean/robust/mean/baseline/spot_residual/sigmaclip/residual"
    )


def _gui_radial_grid_as_two_theta(
    radial: np.ndarray, unit: str, wavelength_A: float
) -> np.ndarray:
    zeros = np.zeros_like(radial, dtype=float)
    converted, _ = _radial_as_two_theta(radial, zeros, unit, wavelength_A)
    if not np.all(np.isfinite(converted)) or not np.all(np.diff(converted) > 0):
        raise ValueError("converted GUI radial grid is not finite and strictly increasing")
    return converted


def load_gui_dataset(row: pd.Series, dataset_index: int) -> GuiDataset:
    channel = str(row["channel"]).strip()
    scan = str(row["scan"]).strip()
    path = Path(str(row["analysis_h5"])).expanduser().resolve()
    source = str(row.get("pattern_source", "clean") or "clean").strip().lower()
    if not path.is_file():
        raise FileNotFoundError(f"GUI analysis HDF5 does not exist: {path}")

    with h5py.File(path, "r") as h5:
        unit = _decode(h5.attrs.get("unit", ""))
        wavelength = _safe_float(h5.attrs.get("wavelength", math.nan))
        if "radial" not in h5:
            raise ValueError(f"{path}: no /radial")
        raw_radial = np.asarray(h5["radial"][:], dtype=float)
        radial_deg = _gui_radial_grid_as_two_theta(raw_radial, unit, wavelength)
        step = float(np.median(np.diff(radial_deg)))
        patterns = _read_pattern_source(h5, source)
        frames = h5.get("frames")
        if frames is None or "pressure" not in frames:
            raise ValueError(f"{path}: no /frames/pressure; GUI pressure-axis check is impossible")
        pressures = np.asarray(frames["pressure"][:], dtype=float)
        if "filename" in frames:
            filenames = tuple(_decode(value) for value in frames["filename"][:])
        else:
            filenames = tuple(f"frame_{index}" for index in range(pressures.size))
        excluded = (
            np.asarray(frames["excluded"][:], dtype=bool)
            if "excluded" in frames
            else np.zeros(pressures.size, dtype=bool)
        )
        if patterns.shape != (pressures.size, radial_deg.size):
            raise ValueError(
                f"{path}: Pattern map shape {patterns.shape} does not match "
                f"frames/radial {(pressures.size, radial_deg.size)}"
            )

        peaks_group = h5.get("peaks")
        if peaks_group is None or "center" not in peaks_group:
            raise ValueError(f"{path}: no /peaks/center; run GUI Step 2 first")
        count = int(peaks_group["center"].shape[0])
        def peak_col(name: str, dtype: Any, default: Any) -> np.ndarray:
            if name in peaks_group:
                return np.asarray(peaks_group[name][:], dtype=dtype)
            return np.full(count, default, dtype=dtype)

        peak_frame = peak_col("frame", int, -1)
        center_raw = peak_col("center", float, math.nan)
        fwhm_raw = peak_col("fwhm", float, math.nan)
        center_deg, fwhm_deg = _radial_as_two_theta(
            center_raw, fwhm_raw, unit, wavelength
        )
        flags = peak_col("flag", int, 0)
        frame_pressure = np.full(count, np.nan, dtype=float)
        frame_name = np.full(count, "", dtype=object)
        frame_excluded = np.ones(count, dtype=bool)
        valid_frame = (peak_frame >= 0) & (peak_frame < pressures.size)
        frame_pressure[valid_frame] = pressures[peak_frame[valid_frame]]
        frame_name[valid_frame] = np.asarray(filenames, dtype=object)[peak_frame[valid_frame]]
        frame_excluded[valid_frame] = excluded[peak_frame[valid_frame]]
        prefix = f"gui{dataset_index:03d}"
        peaks = pd.DataFrame(
            {
                "gui_dataset_id": prefix,
                "gui_peak_id": [f"{prefix}|peak{index:07d}" for index in range(count)],
                "channel": channel,
                "scan": scan,
                "pressure_GPa": frame_pressure,
                "gui_frame": peak_frame,
                "gui_filename": frame_name,
                "gui_frame_excluded": frame_excluded.astype(int),
                "gui_center_native": center_raw,
                "gui_fwhm_native": fwhm_raw,
                "gui_two_theta_deg": center_deg,
                "gui_fwhm_two_theta_deg": fwhm_deg,
                "gui_amplitude": peak_col("amplitude", float, math.nan),
                "gui_area": peak_col("area", float, math.nan),
                "gui_chi2": peak_col("chi2", float, math.nan),
                "gui_flag": flags,
                "gui_good": (flags == 0).astype(int),
                "gui_rejected_reason": [_flag_reason(value) for value in flags],
                "gui_radial_step_deg": step,
                "gui_analysis_h5": str(path),
                "gui_pattern_source": source,
            }
        )

    return GuiDataset(
        channel=channel,
        scan=scan,
        path=path,
        pattern_source=source,
        unit=unit,
        wavelength_A=wavelength,
        radial_deg=radial_deg,
        radial_step_deg=step,
        pressure_GPa=pressures,
        filenames=filenames,
        excluded=excluded,
        patterns=patterns,
        peaks=peaks,
    )


def _load_gui_inventory(path: Path) -> list[GuiDataset]:
    inventory = pd.read_csv(path)
    required = {"channel", "scan", "analysis_h5"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"GUI inventory missing columns: {sorted(missing)}")
    if "pattern_source" not in inventory:
        inventory["pattern_source"] = "clean"
    datasets = [load_gui_dataset(row, index) for index, (_, row) in enumerate(inventory.iterrows())]
    keys = [(item.channel, item.scan) for item in datasets]
    if len(keys) != len(set(keys)):
        raise ValueError("GUI inventory has duplicate channel+scan entries")
    return datasets


def _require_run_profile(root: Path, expected_profile: str, *, role: str) -> dict[str, Any]:
    """Fail closed when a result directory is mislabeled or incomplete."""

    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{role} root has no run_manifest.json: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} run_manifest.json is unreadable: {manifest_path}") from exc
    actual = str(manifest.get("profile", "")).strip()
    if actual != expected_profile:
        raise ValueError(
            f"{role} profile mismatch: expected {expected_profile!r}, "
            f"found {actual!r} in {manifest_path}"
        )
    return manifest


def _load_correlation_peaks(root: Path, channel: str, label: str) -> pd.DataFrame:
    path = root / channel / "per_peak" / "detected_peak_fits.csv"
    if not path.is_file():
        return pd.DataFrame()
    peaks = pd.read_csv(path)
    needed = {
        "scan", "pressure_GPa", "peak_id", "state", "reason",
        "two_theta_deg", "fwhm_two_theta_deg", "raw_fitted_area",
    }
    missing = needed - set(peaks.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    peaks = peaks.copy()
    peaks.insert(0, "result_version", label)
    peaks.insert(1, "correlation_row_id", [f"{label}|{channel}|row{index:08d}" for index in range(len(peaks))])
    return peaks


def _candidate_gate(corr_fwhm: np.ndarray, gui_fwhm: np.ndarray, radial_step: np.ndarray) -> np.ndarray:
    return np.maximum(
        0.5 * (corr_fwhm[:, None] + gui_fwhm[None, :]),
        2.0 * radial_step[None, :],
    )


MATCH_COLUMNS = [
    "result_version", "match_view", "channel", "scan", "pressure_GPa",
    "match_status", "correlation_row_id", "correlation_frame", "correlation_peak_id",
    "correlation_state", "correlation_reason", "correlation_two_theta_deg",
    "correlation_fwhm_two_theta_deg", "correlation_raw_fitted_area",
    "correlation_relative_area", "gui_peak_id", "gui_frame", "gui_filename",
    "gui_good", "gui_flag", "gui_rejected_reason", "gui_two_theta_deg",
    "gui_fwhm_two_theta_deg", "gui_area", "position_delta_deg",
    "admissible_gate_deg", "position_delta_over_gate",
    "nearest_alternative_gui_delta_deg", "assignment_margin_to_next_gui_deg",
    "ambiguous_within_one_gui_step", "assignment_objective",
    "gui_over_correlation_fwhm_ratio", "gui_over_correlation_area_ratio",
    "matching_formula",
]


def _match_one_group(
    corr: pd.DataFrame,
    gui: pd.DataFrame,
    *,
    result_version: str,
    view: str,
    channel: str,
    scan: str,
    pressure: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    corr = corr.reset_index(drop=True)
    gui = gui.reset_index(drop=True)
    matched_corr: set[int] = set()
    matched_gui: set[int] = set()
    assignments: list[tuple[int, int, float, float, float, float, int]] = []
    if not corr.empty and not gui.empty:
        ccenter = pd.to_numeric(corr["two_theta_deg"], errors="coerce").to_numpy(float)
        gcenter = pd.to_numeric(gui["gui_two_theta_deg"], errors="coerce").to_numpy(float)
        cfwhm = pd.to_numeric(corr["fwhm_two_theta_deg"], errors="coerce").to_numpy(float)
        gfwhm = pd.to_numeric(gui["gui_fwhm_two_theta_deg"], errors="coerce").to_numpy(float)
        gstep = pd.to_numeric(gui["gui_radial_step_deg"], errors="coerce").to_numpy(float)
        delta = np.abs(ccenter[:, None] - gcenter[None, :])
        gate = _candidate_gate(cfwhm, gfwhm, gstep)
        admissible = (
            np.isfinite(delta) & np.isfinite(gate) & (gate > 0) & (delta <= gate)
        )
        cost = np.full(delta.shape, 1e9, dtype=float)
        # The gate decides admissibility.  Among admissible pairs, minimize the
        # absolute center difference so broad peaks are not artificially favored.
        cost[admissible] = delta[admissible]
        rr, cc = linear_sum_assignment(cost)
        for i, j in zip(rr, cc, strict=True):
            if cost[i, j] >= 1e8:
                continue
            matched_corr.add(int(i))
            matched_gui.add(int(j))
            alternative_indices = np.flatnonzero(admissible[i])
            alternative_indices = alternative_indices[alternative_indices != j]
            alternatives = delta[i, alternative_indices].astype(float)
            alternative = float(np.min(alternatives)) if alternatives.size else math.nan
            margin = alternative - float(delta[i, j]) if np.isfinite(alternative) else math.nan
            ambiguous = int(np.isfinite(margin) and margin <= float(gstep[j]))
            assignments.append(
                (
                    int(i), int(j), float(delta[i, j]), float(gate[i, j]),
                    alternative, margin, ambiguous,
                )
            )

    def base_record(
        status: str,
        crow: pd.Series | None,
        grow: pd.Series | None,
        *,
        alternative_delta: float = math.nan,
        assignment_margin: float = math.nan,
        ambiguous: Any = "",
    ) -> dict[str, Any]:
        delta = (
            abs(float(crow["two_theta_deg"]) - float(grow["gui_two_theta_deg"]))
            if crow is not None and grow is not None
            else math.nan
        )
        gate = (
            max(
                0.5 * (float(crow["fwhm_two_theta_deg"]) + float(grow["gui_fwhm_two_theta_deg"])),
                2.0 * float(grow["gui_radial_step_deg"]),
            )
            if crow is not None and grow is not None
            else math.nan
        )
        return {
            "result_version": result_version,
            "match_view": view,
            "channel": channel,
            "scan": scan,
            "pressure_GPa": pressure,
            "match_status": status,
            "correlation_row_id": "" if crow is None else crow["correlation_row_id"],
            "correlation_frame": "" if crow is None else crow.get("frame", ""),
            "correlation_peak_id": "" if crow is None else crow.get("peak_id", ""),
            "correlation_state": "" if crow is None else crow.get("state", ""),
            "correlation_reason": "" if crow is None else crow.get("reason", ""),
            "correlation_two_theta_deg": math.nan if crow is None else crow.get("two_theta_deg", math.nan),
            "correlation_fwhm_two_theta_deg": math.nan if crow is None else crow.get("fwhm_two_theta_deg", math.nan),
            "correlation_raw_fitted_area": math.nan if crow is None else crow.get("raw_fitted_area", math.nan),
            "correlation_relative_area": math.nan if crow is None else crow.get("relative_area", math.nan),
            "gui_peak_id": "" if grow is None else grow.get("gui_peak_id", ""),
            "gui_frame": "" if grow is None else grow.get("gui_frame", ""),
            "gui_filename": "" if grow is None else grow.get("gui_filename", ""),
            "gui_good": "" if grow is None else grow.get("gui_good", ""),
            "gui_flag": "" if grow is None else grow.get("gui_flag", ""),
            "gui_rejected_reason": "" if grow is None else grow.get("gui_rejected_reason", ""),
            "gui_two_theta_deg": math.nan if grow is None else grow.get("gui_two_theta_deg", math.nan),
            "gui_fwhm_two_theta_deg": math.nan if grow is None else grow.get("gui_fwhm_two_theta_deg", math.nan),
            "gui_area": math.nan if grow is None else grow.get("gui_area", math.nan),
            "position_delta_deg": delta,
            "admissible_gate_deg": gate,
            "position_delta_over_gate": delta / gate if np.isfinite(delta) and np.isfinite(gate) and gate > 0 else math.nan,
            "nearest_alternative_gui_delta_deg": alternative_delta,
            "assignment_margin_to_next_gui_deg": assignment_margin,
            "ambiguous_within_one_gui_step": ambiguous,
            "assignment_objective": "minimum absolute center difference after admissibility gate",
            "gui_over_correlation_fwhm_ratio": (
                float(grow["gui_fwhm_two_theta_deg"]) / float(crow["fwhm_two_theta_deg"])
                if crow is not None and grow is not None
                and np.isfinite(float(crow["fwhm_two_theta_deg"]))
                and np.isfinite(float(grow["gui_fwhm_two_theta_deg"]))
                and float(crow["fwhm_two_theta_deg"]) > 0
                else math.nan
            ),
            "gui_over_correlation_area_ratio": (
                float(grow["gui_area"]) / float(crow["raw_fitted_area"])
                if crow is not None and grow is not None
                and np.isfinite(float(crow["raw_fitted_area"]))
                and np.isfinite(float(grow["gui_area"]))
                and float(crow["raw_fitted_area"]) > 0
                else math.nan
            ),
            "matching_formula": MATCH_FORMULA,
        }

    for i, j, _, _, alternative, margin, ambiguous in assignments:
        rows.append(
            base_record(
                "matched", corr.iloc[i], gui.iloc[j],
                alternative_delta=alternative,
                assignment_margin=margin,
                ambiguous=ambiguous,
            )
        )
    for i in range(len(corr)):
        if i not in matched_corr:
            rows.append(base_record("correlation_only", corr.iloc[i], None))
    for j in range(len(gui)):
        if j not in matched_gui:
            rows.append(base_record("gui_only", None, gui.iloc[j]))
    return rows


def match_peak_tables(
    corr_peaks: pd.DataFrame,
    gui_peaks: pd.DataFrame,
    *,
    result_version: str,
) -> pd.DataFrame:
    if corr_peaks.empty or gui_peaks.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    result: list[dict[str, Any]] = []
    available = gui_peaks[["channel", "scan", "pressure_GPa"]].drop_duplicates()
    for _, group_key in available.iterrows():
        channel = str(group_key["channel"])
        scan = str(group_key["scan"])
        pressure = float(group_key["pressure_GPa"])
        corr_group = corr_peaks[
            (corr_peaks["scan"].astype(str) == scan)
            & np.isclose(pd.to_numeric(corr_peaks["pressure_GPa"], errors="coerce"), pressure, atol=1e-8, rtol=0)
        ]
        gui_group = gui_peaks[
            (gui_peaks["channel"].astype(str) == channel)
            & (gui_peaks["scan"].astype(str) == scan)
            & np.isclose(pd.to_numeric(gui_peaks["pressure_GPa"], errors="coerce"), pressure, atol=1e-8, rtol=0)
        ]
        for view in ("all_peaks", "good_only"):
            c = corr_group
            g = gui_group
            if view == "good_only":
                c = c[c["state"].astype(str).str.lower() == "reliable"]
                g = g[pd.to_numeric(g["gui_good"], errors="coerce") == 1]
            finite_c = np.isfinite(pd.to_numeric(c["two_theta_deg"], errors="coerce"))
            finite_g = np.isfinite(pd.to_numeric(g["gui_two_theta_deg"], errors="coerce"))
            c = c[finite_c]
            g = g[finite_g]
            result.extend(
                _match_one_group(
                    c, g, result_version=result_version, view=view,
                    channel=channel, scan=scan, pressure=pressure,
                )
            )
    return pd.DataFrame(result, columns=MATCH_COLUMNS)


def _summarize_matches(matches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "result_version", "match_view", "channel", "scan", "row_kind",
        "pressure_GPa", "correlation_candidates", "gui_candidates", "matched",
        "correlation_match_fraction", "gui_match_fraction", "median_position_delta_deg",
        "q90_position_delta_deg", "median_position_delta_over_gate",
        "median_gui_over_correlation_fwhm_ratio", "fwhm_spearman",
        "median_gui_over_correlation_area_ratio", "area_spearman",
    ]
    if matches.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_cols = ["result_version", "match_view", "channel", "scan"]
    for keys, group in matches.groupby(group_cols, dropna=False):
        for row_kind, pressure, subset in [
            ("overall", math.nan, group),
            *[("pressure", float(p), part) for p, part in group.groupby("pressure_GPa")],
        ]:
            n_corr = int((subset["match_status"] != "gui_only").sum())
            n_gui = int((subset["match_status"] != "correlation_only").sum())
            n_match = int((subset["match_status"] == "matched").sum())
            deltas = pd.to_numeric(
                subset.loc[subset["match_status"] == "matched", "position_delta_deg"],
                errors="coerce",
            ).dropna()
            ratios = pd.to_numeric(
                subset.loc[subset["match_status"] == "matched", "position_delta_over_gate"],
                errors="coerce",
            ).dropna()
            paired = subset[subset["match_status"] == "matched"].copy()
            fwhm_ratio = pd.to_numeric(
                paired["gui_over_correlation_fwhm_ratio"], errors="coerce"
            ).dropna()
            area_ratio = pd.to_numeric(
                paired["gui_over_correlation_area_ratio"], errors="coerce"
            ).dropna()
            fwhm_rho = math.nan
            area_rho = math.nan
            corr_fwhm = pd.to_numeric(paired["correlation_fwhm_two_theta_deg"], errors="coerce")
            gui_fwhm = pd.to_numeric(paired["gui_fwhm_two_theta_deg"], errors="coerce")
            valid_fwhm = np.isfinite(corr_fwhm) & np.isfinite(gui_fwhm)
            if int(valid_fwhm.sum()) >= 3 and corr_fwhm[valid_fwhm].nunique() > 1 and gui_fwhm[valid_fwhm].nunique() > 1:
                fwhm_rho = float(spearmanr(corr_fwhm[valid_fwhm], gui_fwhm[valid_fwhm]).statistic)
            corr_area = pd.to_numeric(paired["correlation_raw_fitted_area"], errors="coerce")
            gui_area = pd.to_numeric(paired["gui_area"], errors="coerce")
            valid_area = np.isfinite(corr_area) & np.isfinite(gui_area)
            if int(valid_area.sum()) >= 3 and corr_area[valid_area].nunique() > 1 and gui_area[valid_area].nunique() > 1:
                area_rho = float(spearmanr(corr_area[valid_area], gui_area[valid_area]).statistic)
            rows.append(
                {
                    "result_version": keys[0], "match_view": keys[1],
                    "channel": keys[2], "scan": keys[3], "row_kind": row_kind,
                    "pressure_GPa": pressure, "correlation_candidates": n_corr,
                    "gui_candidates": n_gui, "matched": n_match,
                    "correlation_match_fraction": n_match / n_corr if n_corr else math.nan,
                    "gui_match_fraction": n_match / n_gui if n_gui else math.nan,
                    "median_position_delta_deg": float(deltas.median()) if len(deltas) else math.nan,
                    "q90_position_delta_deg": float(deltas.quantile(0.9)) if len(deltas) else math.nan,
                    "median_position_delta_over_gate": float(ratios.median()) if len(ratios) else math.nan,
                    "median_gui_over_correlation_fwhm_ratio": float(fwhm_ratio.median()) if len(fwhm_ratio) else math.nan,
                    "fwhm_spearman": fwhm_rho,
                    "median_gui_over_correlation_area_ratio": float(area_ratio.median()) if len(area_ratio) else math.nan,
                    "area_spearman": area_rho,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _reason_crosstab(matches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "result_version", "match_view", "channel", "scan", "correlation_state",
        "correlation_reason", "gui_good", "gui_rejected_reason", "matched_count",
    ]
    if matches.empty:
        return pd.DataFrame(columns=columns)
    selected = matches[matches["match_status"] == "matched"].copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)
    return (
        selected.groupby(columns[:-1], dropna=False)
        .size().rename("matched_count").reset_index()[columns]
    )


TRACK_COLUMNS = [
    "result_version", "channel", "scan", "track_id", "track_official",
    "validation_scope",
    "segment_pressure_min_GPa", "segment_pressure_max_GPa", "segment_pressure_nodes",
    "v21_scan_present_points", "v21_scan_pressure_min_GPa", "v21_scan_pressure_max_GPa",
    "gui_matched_points", "gui_matched_pressure_min_GPa", "gui_matched_pressure_max_GPa",
    "v21_two_theta_slope_deg_per_GPa", "gui_two_theta_slope_deg_per_GPa",
    "slope_difference_deg_per_GPa", "slope_sign_agrees", "median_position_delta_deg",
]


def _linear_slope(x: Sequence[float], y: Sequence[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy)
    if np.count_nonzero(valid) < 2 or np.unique(xx[valid]).size < 2:
        return math.nan
    return float(np.polyfit(xx[valid], yy[valid], 1)[0])


def build_track_comparison(
    root: Path,
    channel: str,
    label: str,
    corr_peaks: pd.DataFrame,
    matches: pd.DataFrame,
    gui_scans: set[str],
) -> pd.DataFrame:
    observations_path = root / channel / "per_peak" / "peak_observations.csv"
    tracks_path = root / channel / "per_peak" / "canonical_tracks.csv"
    if not observations_path.is_file() or not tracks_path.is_file() or corr_peaks.empty:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    observations = pd.read_csv(observations_path)
    tracks = pd.read_csv(tracks_path)
    official = tracks[pd.to_numeric(tracks.get("official", 0), errors="coerce") == 1]
    if official.empty:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    good_matches = matches[
        (matches["result_version"] == label)
        & (matches["channel"] == channel)
        & (matches["match_view"] == "good_only")
        & (matches["match_status"] == "matched")
    ]
    corr_lookup = corr_peaks[[
        "scan", "pressure_GPa", "peak_id", "correlation_row_id", "two_theta_deg"
    ]].copy()
    corr_lookup["pressure_key"] = pd.to_numeric(corr_lookup["pressure_GPa"], errors="coerce").round(8)
    observations = observations.copy()
    observations["pressure_key"] = pd.to_numeric(observations["pressure_GPa"], errors="coerce").round(8)
    present = observations[
        (observations["track_official"] == 1)
        & (observations["state"].astype(str).str.lower() == "present")
        & observations["scan"].astype(str).isin(gui_scans)
    ].merge(
        corr_lookup,
        on=["scan", "pressure_key", "peak_id"],
        how="left",
        suffixes=("", "_fit"),
    )
    present = present.merge(
        good_matches[[
            "correlation_row_id", "gui_two_theta_deg", "position_delta_deg"
        ]],
        on="correlation_row_id",
        how="left",
    )
    track_meta = official.set_index("track_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for (scan, track_id), group in present.groupby(["scan", "track_id"], dropna=False):
        meta = track_meta.get(track_id, {})
        pressure = pd.to_numeric(group["pressure_GPa"], errors="coerce").to_numpy(float)
        vpos = pd.to_numeric(group["two_theta_deg"], errors="coerce").to_numpy(float)
        gpos = pd.to_numeric(group["gui_two_theta_deg"], errors="coerce").to_numpy(float)
        matched = np.isfinite(gpos)
        v_slope = _linear_slope(pressure[matched], vpos[matched])
        g_slope = _linear_slope(pressure[matched], gpos[matched])
        finite_pressure = pressure[np.isfinite(pressure)]
        matched_pressure = pressure[matched & np.isfinite(pressure)]
        deltas = pd.to_numeric(group["position_delta_deg"], errors="coerce").dropna()
        sign_agree: Any = ""
        if np.isfinite(v_slope) and np.isfinite(g_slope):
            sign_agree = int(np.sign(v_slope) == np.sign(g_slope))
        rows.append(
            {
                "result_version": label, "channel": channel, "scan": scan,
                "track_id": track_id, "track_official": 1,
                "validation_scope": "conditional_local_corroboration_on_correlation_track_identity",
                "segment_pressure_min_GPa": meta.get("pressure_min_GPa", math.nan),
                "segment_pressure_max_GPa": meta.get("pressure_max_GPa", math.nan),
                "segment_pressure_nodes": meta.get("pressure_nodes", math.nan),
                "v21_scan_present_points": int(np.count_nonzero(np.isfinite(vpos))),
                "v21_scan_pressure_min_GPa": float(np.min(finite_pressure)) if finite_pressure.size else math.nan,
                "v21_scan_pressure_max_GPa": float(np.max(finite_pressure)) if finite_pressure.size else math.nan,
                "gui_matched_points": int(np.count_nonzero(matched)),
                "gui_matched_pressure_min_GPa": float(np.min(matched_pressure)) if matched_pressure.size else math.nan,
                "gui_matched_pressure_max_GPa": float(np.max(matched_pressure)) if matched_pressure.size else math.nan,
                "v21_two_theta_slope_deg_per_GPa": v_slope,
                "gui_two_theta_slope_deg_per_GPa": g_slope,
                "slope_difference_deg_per_GPa": g_slope - v_slope if np.isfinite(v_slope) and np.isfinite(g_slope) else math.nan,
                "slope_sign_agrees": sign_agree,
                "median_position_delta_deg": float(deltas.median()) if len(deltas) else math.nan,
            }
        )
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def _matrix_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(path)
    if table.shape[1] < 2:
        raise ValueError(f"matrix CSV has no matrix columns: {path}")
    pressures = pd.to_numeric(table.iloc[:, 0], errors="coerce").to_numpy(float)
    matrix = table.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if matrix.shape != (pressures.size, pressures.size):
        raise ValueError(f"matrix is not square: {path} -> {matrix.shape}")
    return pressures, matrix


def _window_file(root: Path, family: str, subfolder: str, start: float, end: float, suffix: str) -> Path:
    return root / family / subfolder / f"window_{start:.4f}_{end:.4f}{suffix}"


BOUNDARY_COLUMNS = [
    "channel", "window_index", "start_deg", "end_deg", "interval_index",
    "pressure_low_GPa", "pressure_high_GPa", "pressure_gap_GPa",
    "acf_similarity", "acf_ci_low", "acf_ci_high", "acf_support",
    "direct_similarity", "direct_ci_low", "direct_ci_high", "direct_support",
    "acf_boundary_threshold_Q25", "direct_boundary_threshold_Q25",
    "acf_boundary_call", "direct_boundary_call", "same_boundary_call",
    "corroborated_low_similarity_candidate", "unresolved_one_family_candidate",
    "acf_call_has_direct_neighbor", "direct_call_has_acf_neighbor",
    "confidence_intervals_overlap", "boundary_formula",
]


def build_strict_boundary_table(result_root: Path, channel: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    across = result_root / channel / "across_frames"
    acf_summary_path = across / "acf_strict" / "window_summary.csv"
    direct_summary_path = across / "direct_strict" / "window_summary.csv"
    if not acf_summary_path.is_file() or not direct_summary_path.is_file():
        return pd.DataFrame(columns=BOUNDARY_COLUMNS), pd.DataFrame()
    acf_summary = pd.read_csv(acf_summary_path).sort_values("window_index")
    direct_summary = pd.read_csv(direct_summary_path).sort_values("window_index")
    if list(acf_summary["window_index"]) != list(direct_summary["window_index"]):
        raise ValueError(f"{channel}: ACF/direct strict window indices differ")
    rows: list[dict[str, Any]] = []
    for _, arow in acf_summary.iterrows():
        index = int(arow["window_index"])
        drow = direct_summary[direct_summary["window_index"] == index].iloc[0]
        start = float(arow["start_deg"])
        end = float(arow["end_deg"])
        if not np.isclose(start, float(drow["start_deg"]), atol=1e-10, rtol=0) or not np.isclose(end, float(drow["end_deg"]), atol=1e-10, rtol=0):
            raise ValueError(f"{channel} window {index}: ACF/direct boundaries differ")
        data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for family in ("acf_strict", "direct_strict"):
            data[family] = _matrix_csv(
                _window_file(across, family, "matrices", start, end, ".csv")
            )
            for kind, suffix in [
                ("support", "_support.csv"),
                ("ci_low", "_ci_low.csv"),
                ("ci_high", "_ci_high.csv"),
            ]:
                folder = "support_maps" if kind == "support" else "confidence_intervals"
                data[f"{family}_{kind}"] = _matrix_csv(
                    _window_file(across, family, folder, start, end, suffix)
                )
        pressures = data["acf_strict"][0]
        for key, (other_pressures, _) in data.items():
            if not np.allclose(pressures, other_pressures, atol=1e-10, rtol=0, equal_nan=True):
                raise ValueError(f"{channel} window {index}: pressure labels differ for {key}")
        acf_adj = np.array([data["acf_strict"][1][i, i + 1] for i in range(len(pressures) - 1)])
        direct_adj = np.array([data["direct_strict"][1][i, i + 1] for i in range(len(pressures) - 1)])
        acf_q25 = float(np.nanquantile(acf_adj, 0.25)) if np.any(np.isfinite(acf_adj)) else math.nan
        direct_q25 = float(np.nanquantile(direct_adj, 0.25)) if np.any(np.isfinite(direct_adj)) else math.nan
        acf_calls = np.isfinite(acf_adj) & (acf_adj <= acf_q25)
        direct_calls = np.isfinite(direct_adj) & (direct_adj <= direct_q25)
        for i in range(len(pressures) - 1):
            acf_lo = data["acf_strict_ci_low"][1][i, i + 1]
            acf_hi = data["acf_strict_ci_high"][1][i, i + 1]
            direct_lo = data["direct_strict_ci_low"][1][i, i + 1]
            direct_hi = data["direct_strict_ci_high"][1][i, i + 1]
            overlap: Any = ""
            if np.all(np.isfinite([acf_lo, acf_hi, direct_lo, direct_hi])):
                overlap = int(max(acf_lo, direct_lo) <= min(acf_hi, direct_hi))
            a_neighbor = int(
                (not acf_calls[i])
                or any(direct_calls[max(0, i - 1): min(len(direct_calls), i + 2)])
            )
            d_neighbor = int(
                (not direct_calls[i])
                or any(acf_calls[max(0, i - 1): min(len(acf_calls), i + 2)])
            )
            rows.append(
                {
                    "channel": channel, "window_index": index,
                    "start_deg": start, "end_deg": end, "interval_index": i,
                    "pressure_low_GPa": pressures[i], "pressure_high_GPa": pressures[i + 1],
                    "pressure_gap_GPa": pressures[i + 1] - pressures[i],
                    "acf_similarity": acf_adj[i], "acf_ci_low": acf_lo,
                    "acf_ci_high": acf_hi,
                    "acf_support": data["acf_strict_support"][1][i, i + 1],
                    "direct_similarity": direct_adj[i], "direct_ci_low": direct_lo,
                    "direct_ci_high": direct_hi,
                    "direct_support": data["direct_strict_support"][1][i, i + 1],
                    "acf_boundary_threshold_Q25": acf_q25,
                    "direct_boundary_threshold_Q25": direct_q25,
                    "acf_boundary_call": int(acf_calls[i]),
                    "direct_boundary_call": int(direct_calls[i]),
                    "same_boundary_call": int(acf_calls[i] == direct_calls[i]),
                    "corroborated_low_similarity_candidate": int(acf_calls[i] and direct_calls[i]),
                    "unresolved_one_family_candidate": int(acf_calls[i] != direct_calls[i]),
                    "acf_call_has_direct_neighbor": a_neighbor,
                    "direct_call_has_acf_neighbor": d_neighbor,
                    "confidence_intervals_overlap": overlap,
                    "boundary_formula": BOUNDARY_FORMULA,
                }
            )
    table = pd.DataFrame(rows, columns=BOUNDARY_COLUMNS)
    summaries: list[dict[str, Any]] = []
    for (ch, window), group in table.groupby(["channel", "window_index"]):
        a = pd.to_numeric(group["acf_similarity"], errors="coerce").to_numpy(float)
        d = pd.to_numeric(group["direct_similarity"], errors="coerce").to_numpy(float)
        valid = np.isfinite(a) & np.isfinite(d)
        rho = math.nan
        if np.count_nonzero(valid) >= 3 and np.unique(a[valid]).size > 1 and np.unique(d[valid]).size > 1:
            rho = float(spearmanr(1.0 - a[valid], 1.0 - d[valid]).statistic)
        ac = group["acf_boundary_call"].astype(bool).to_numpy()
        dc = group["direct_boundary_call"].astype(bool).to_numpy()
        union = int(np.count_nonzero(ac | dc))
        summaries.append(
            {
                "channel": ch, "window_index": window,
                "start_deg": group["start_deg"].iloc[0], "end_deg": group["end_deg"].iloc[0],
                "adjacent_intervals": len(group),
                "finite_both": int(np.count_nonzero(valid)),
                "boundary_score_spearman": rho,
                "exact_same_call_fraction": float(np.mean(ac == dc)) if len(group) else math.nan,
                "top_quartile_call_jaccard": int(np.count_nonzero(ac & dc)) / union if union else math.nan,
                "corroborated_candidate_count": int(np.count_nonzero(ac & dc)),
                "unresolved_candidate_count": int(np.count_nonzero(ac ^ dc)),
                "acf_calls_with_direct_neighbor_fraction": float(group.loc[group["acf_boundary_call"] == 1, "acf_call_has_direct_neighbor"].mean()) if np.any(ac) else math.nan,
                "direct_calls_with_acf_neighbor_fraction": float(group.loc[group["direct_boundary_call"] == 1, "direct_call_has_acf_neighbor"].mean()) if np.any(dc) else math.nan,
                "minimum_acf_support": pd.to_numeric(group["acf_support"], errors="coerce").min(),
                "minimum_direct_support": pd.to_numeric(group["direct_support"], errors="coerce").min(),
            }
        )
    return table, pd.DataFrame(summaries)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape or left.size < 3 or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return math.nan
    left = left - np.mean(left)
    right = right - np.mean(right)
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.clip(np.dot(left, right) / norm, -1.0, 1.0)) if norm > np.finfo(float).eps else math.nan


def _standardized_window(radial: np.ndarray, pattern: np.ndarray, start: float, end: float) -> np.ndarray | None:
    step = float(np.median(np.diff(radial)))
    count = max(3, int(math.floor((end - start) / step + 1e-12)) + 1)
    x = np.linspace(start, end, count)
    sampled = np.interp(x, radial, pattern, left=np.nan, right=np.nan)
    if not np.all(np.isfinite(sampled)):
        return None
    centered = sampled - float(np.mean(sampled))
    scale = float(np.std(centered))
    magnitude = max(float(np.max(np.abs(sampled))), 1.0)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps * magnitude:
        return None
    return centered / scale


def _acf_fingerprint(signal: np.ndarray | None) -> np.ndarray | None:
    if signal is None or signal.size < 3:
        return None
    n = signal.size
    transformed = np.fft.rfft(signal, n=2 * n)
    correlation = np.fft.irfft(transformed * np.conjugate(transformed), n=2 * n)[:n]
    if not np.isfinite(correlation[0]) or correlation[0] <= np.finfo(float).eps:
        return None
    raw = correlation[1:] / correlation[0]
    centered = raw - float(np.mean(raw))
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    return centered / scale


def _normalized_l1_redistribution(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    floor = min(float(np.nanmin(left)), float(np.nanmin(right)), 0.0)
    left = np.maximum(left - floor, 0.0)
    right = np.maximum(right - floor, 0.0)
    if left.sum() <= 0 or right.sum() <= 0:
        return math.nan
    return float(0.5 * np.sum(np.abs(left / left.sum() - right / right.sum())))


PATTERN_COLUMNS = [
    "channel", "scan", "gui_analysis_h5", "gui_pattern_source", "window_index",
    "start_deg", "end_deg", "pressure_low_GPa", "pressure_high_GPa",
    "acf_strict_aggregate_similarity", "direct_strict_aggregate_similarity",
    "acf_strict_support", "direct_strict_support", "acf_strict_ci_low",
    "acf_strict_ci_high", "direct_strict_ci_low", "direct_strict_ci_high",
    "acf_boundary_call", "direct_boundary_call", "same_boundary_call",
    "gui_direct_pearson_same_window", "gui_acf_pearson_same_window",
    "gui_normalized_L1_intensity_redistribution", "gui_positive_area_minmax_similarity",
    "gui_good_peak_count_low", "gui_good_peak_count_high", "gui_good_peak_count_change",
    "gui_good_peaks_unmatched_across_boundary", "gui_max_center_shift_in_mean_fwhm",
    "gui_unmatched_or_appearance_disappearance_candidate",
    "gui_peak_count_change_candidate",
]


def _find_frame_at_pressure(dataset: GuiDataset, pressure: float) -> int | None:
    locations = np.flatnonzero(np.isclose(dataset.pressure_GPa, pressure, atol=1e-8, rtol=0))
    if locations.size != 1:
        return None
    return int(locations[0])


def _gui_peak_change_metrics(low: pd.DataFrame, high: pd.DataFrame) -> dict[str, Any]:
    low = low[low["gui_good"] == 1].reset_index(drop=True)
    high = high[high["gui_good"] == 1].reset_index(drop=True)
    n_low, n_high = len(low), len(high)
    unmatched = n_low + n_high
    max_shift = math.nan
    if n_low and n_high:
        lcenter = low["gui_two_theta_deg"].to_numpy(float)
        hcenter = high["gui_two_theta_deg"].to_numpy(float)
        lfwhm = low["gui_fwhm_two_theta_deg"].to_numpy(float)
        hfwhm = high["gui_fwhm_two_theta_deg"].to_numpy(float)
        delta = np.abs(lcenter[:, None] - hcenter[None, :])
        denom = 0.5 * (lfwhm[:, None] + hfwhm[None, :])
        valid = np.isfinite(delta) & np.isfinite(denom) & (denom > 0) & (delta <= denom)
        cost = np.full(delta.shape, 1e9)
        cost[valid] = delta[valid] / denom[valid]
        rr, cc = linear_sum_assignment(cost)
        ratios = [float(cost[i, j]) for i, j in zip(rr, cc, strict=True) if cost[i, j] < 1e8]
        matched = len(ratios)
        unmatched = n_low + n_high - 2 * matched
        if ratios:
            max_shift = max(ratios)
    return {
        "gui_good_peak_count_low": n_low,
        "gui_good_peak_count_high": n_high,
        "gui_good_peak_count_change": n_high - n_low,
        "gui_good_peaks_unmatched_across_boundary": unmatched,
        "gui_max_center_shift_in_mean_fwhm": max_shift,
        "gui_unmatched_or_appearance_disappearance_candidate": int(unmatched > 0),
        "gui_peak_count_change_candidate": int(abs(n_high - n_low) > 0),
    }


def build_pattern_checks(
    boundaries: pd.DataFrame,
    datasets: Sequence[GuiDataset],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        subset = boundaries[boundaries["channel"] == dataset.channel]
        for _, boundary in subset.iterrows():
            p0 = float(boundary["pressure_low_GPa"])
            p1 = float(boundary["pressure_high_GPa"])
            i0 = _find_frame_at_pressure(dataset, p0)
            i1 = _find_frame_at_pressure(dataset, p1)
            if i0 is None or i1 is None:
                continue
            # Excluded GUI frames never contribute pattern evidence.
            if bool(dataset.excluded[i0]) or bool(dataset.excluded[i1]):
                continue
            start = float(boundary["start_deg"])
            end = float(boundary["end_deg"])
            s0 = _standardized_window(dataset.radial_deg, dataset.patterns[i0], start, end)
            s1 = _standardized_window(dataset.radial_deg, dataset.patterns[i1], start, end)
            f0 = _acf_fingerprint(s0)
            f1 = _acf_fingerprint(s1)
            mask = (dataset.radial_deg >= start) & (dataset.radial_deg <= end)
            raw0 = dataset.patterns[i0, mask]
            raw1 = dataset.patterns[i1, mask]
            positive0 = np.maximum(raw0, 0.0)
            positive1 = np.maximum(raw1, 0.0)
            area0 = float(np.trapezoid(positive0, dataset.radial_deg[mask])) if np.count_nonzero(mask) >= 2 else math.nan
            area1 = float(np.trapezoid(positive1, dataset.radial_deg[mask])) if np.count_nonzero(mask) >= 2 else math.nan
            area_similarity = (
                min(area0, area1) / max(area0, area1)
                if np.isfinite(area0) and np.isfinite(area1) and max(area0, area1) > 0
                else math.nan
            )
            peaks0 = dataset.peaks[
                np.isclose(dataset.peaks["pressure_GPa"], p0, atol=1e-8, rtol=0)
                & dataset.peaks["gui_two_theta_deg"].between(start, end, inclusive="both")
            ]
            peaks1 = dataset.peaks[
                np.isclose(dataset.peaks["pressure_GPa"], p1, atol=1e-8, rtol=0)
                & dataset.peaks["gui_two_theta_deg"].between(start, end, inclusive="both")
            ]
            record = {
                "channel": dataset.channel, "scan": dataset.scan,
                "gui_analysis_h5": str(dataset.path),
                "gui_pattern_source": dataset.pattern_source,
                "window_index": boundary["window_index"], "start_deg": start, "end_deg": end,
                "pressure_low_GPa": p0, "pressure_high_GPa": p1,
                "acf_strict_aggregate_similarity": boundary["acf_similarity"],
                "direct_strict_aggregate_similarity": boundary["direct_similarity"],
                "acf_strict_support": boundary["acf_support"],
                "direct_strict_support": boundary["direct_support"],
                "acf_strict_ci_low": boundary["acf_ci_low"],
                "acf_strict_ci_high": boundary["acf_ci_high"],
                "direct_strict_ci_low": boundary["direct_ci_low"],
                "direct_strict_ci_high": boundary["direct_ci_high"],
                "acf_boundary_call": boundary["acf_boundary_call"],
                "direct_boundary_call": boundary["direct_boundary_call"],
                "same_boundary_call": boundary["same_boundary_call"],
                "gui_direct_pearson_same_window": _pearson(s0, s1) if s0 is not None and s1 is not None else math.nan,
                "gui_acf_pearson_same_window": _pearson(f0, f1) if f0 is not None and f1 is not None else math.nan,
                "gui_normalized_L1_intensity_redistribution": _normalized_l1_redistribution(raw0, raw1),
                "gui_positive_area_minmax_similarity": area_similarity,
                **_gui_peak_change_metrics(peaks0, peaks1),
            }
            rows.append(record)
    return pd.DataFrame(rows, columns=PATTERN_COLUMNS)


def build_control_tables(result_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for channel in ("spots", "fit"):
        for family in ("acf_strict", "direct_strict"):
            path = result_root / channel / "across_frames" / family / "window_summary.csv"
            if path.is_file():
                data = pd.read_csv(path).copy()
                data["channel"] = channel
                data["family"] = family
                rows.append(data)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    all_windows = pd.concat(rows, ignore_index=True)
    keep = [
        "channel", "family", "window_index", "start_deg", "end_deg",
        "near_vs_far_auc", "auc_ci_low", "auc_ci_high", "near_supported_cells",
        "far_supported_cells", "auc_reason_if_na",
    ]
    window_auc = all_windows[[column for column in keep if column in all_windows]].copy()
    pivot = window_auc.pivot_table(
        index=["family", "window_index", "start_deg", "end_deg"],
        columns="channel", values="near_vs_far_auc", aggfunc="first",
    ).reset_index()
    if "spots" in pivot and "fit" in pivot:
        pivot["spots_minus_fit_auc"] = pivot["spots"] - pivot["fit"]
        pivot["fit_auc_higher"] = (pivot["fit"] > pivot["spots"]).astype(int)
    return window_auc, pivot


def build_control_boundary_agreement(boundaries: pd.DataFrame) -> pd.DataFrame:
    if boundaries.empty or not {"spots", "fit"}.issubset(set(boundaries["channel"])):
        return pd.DataFrame()
    key = ["window_index", "start_deg", "end_deg", "interval_index", "pressure_low_GPa", "pressure_high_GPa"]
    fields = ["acf_similarity", "direct_similarity", "acf_boundary_call", "direct_boundary_call"]
    spots = boundaries[boundaries["channel"] == "spots"][key + fields].rename(
        columns={field: f"spots_{field}" for field in fields}
    )
    fit = boundaries[boundaries["channel"] == "fit"][key + fields].rename(
        columns={field: f"fit_{field}" for field in fields}
    )
    result = spots.merge(fit, on=key, how="inner")
    result["same_acf_boundary_call_spots_fit"] = (
        result["spots_acf_boundary_call"] == result["fit_acf_boundary_call"]
    ).astype(int)
    result["same_direct_boundary_call_spots_fit"] = (
        result["spots_direct_boundary_call"] == result["fit_direct_boundary_call"]
    ).astype(int)
    result["shared_positive_acf_candidate_spots_fit"] = (
        (result["spots_acf_boundary_call"] == 1)
        & (result["fit_acf_boundary_call"] == 1)
    ).astype(int)
    result["either_positive_acf_candidate_spots_fit"] = (
        (result["spots_acf_boundary_call"] == 1)
        | (result["fit_acf_boundary_call"] == 1)
    ).astype(int)
    result["shared_positive_direct_candidate_spots_fit"] = (
        (result["spots_direct_boundary_call"] == 1)
        & (result["fit_direct_boundary_call"] == 1)
    ).astype(int)
    result["either_positive_direct_candidate_spots_fit"] = (
        (result["spots_direct_boundary_call"] == 1)
        | (result["fit_direct_boundary_call"] == 1)
    ).astype(int)
    return result


def _plot_peak_map(
    dataset: GuiDataset,
    output_path: Path,
    *,
    good_only: bool,
    color_by: str,
) -> dict[str, Any]:
    """Render the same pressure-x scatter encoded by the GUI Peak map tab."""

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    peaks = dataset.peaks.copy()
    if good_only:
        peaks = peaks[peaks["gui_good"] == 1]
    pressure = pd.to_numeric(peaks["pressure_GPa"], errors="coerce").to_numpy(float)
    center = pd.to_numeric(peaks["gui_two_theta_deg"], errors="coerce").to_numpy(float)
    value_column = "gui_area" if color_by == "area" else "gui_fwhm_two_theta_deg"
    values = pd.to_numeric(peaks[value_column], errors="coerce").to_numpy(float)
    coordinates_ok = np.isfinite(pressure) & np.isfinite(center)
    color_ok = np.isfinite(values) & ((values > 0) if color_by == "area" else True)
    valid = coordinates_ok & color_ok
    missing_color = coordinates_ok & ~color_ok

    fig, ax = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    if np.any(missing_color):
        ax.scatter(
            pressure[missing_color], center[missing_color],
            c="#a6a6a6", s=30, alpha=0.9, edgecolors="#404040", linewidths=0.35,
            label=f"missing/invalid {color_by}", zorder=2,
        )
    scatter = None
    if np.any(valid):
        norm: mcolors.Normalize | None = None
        if color_by == "area":
            positive = values[valid]
            vmin = float(np.min(positive))
            vmax = float(np.max(positive))
            norm = mcolors.LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin * (1.0 + 1e-12)))
        scatter = ax.scatter(
            pressure[valid], center[valid], c=values[valid], cmap="viridis", norm=norm,
            s=30, alpha=0.9, edgecolors="#303030", linewidths=0.35, zorder=3,
        )
        label = "fitted area (GUI units)" if color_by == "area" else "FWHM (2θ deg)"
        fig.colorbar(scatter, ax=ax, label=label)
    if not np.any(coordinates_ok):
        ax.text(
            0.5, 0.5, "No finite pressure/peak-position rows",
            transform=ax.transAxes, ha="center", va="center", color="#666666",
        )

    excluded_pressures = dataset.pressure_GPa[dataset.excluded & np.isfinite(dataset.pressure_GPa)]
    for index, pressure_value in enumerate(excluded_pressures):
        ax.axvline(
            float(pressure_value), color="#9e9e9e", linewidth=1.0,
            linestyle="--", alpha=0.8,
            label="excluded GUI frame" if index == 0 else None,
        )
    title_view = "Good peaks only (flag = 0)" if good_only else "All fitted peaks (rejected included)"
    ax.set_title(
        f"GUI Peak map — {dataset.channel}/{dataset.scan}\n"
        f"Pressure axis · {title_view} · color = {color_by}"
    )
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Peak center (2θ, deg)")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
    if np.any(missing_color) or excluded_pressures.size:
        ax.legend(loc="best", frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white", metadata={
        "Title": f"GUI pressure Peak map {title_view} color={color_by}",
        "Description": "Read-only visualization of BulkXRD analysis HDF5; no algorithm changes.",
    })
    plt.close(fig)
    return {
        "filename": output_path.name,
        "visualization": "gui_peak_map",
        "channel": dataset.channel,
        "scan": dataset.scan,
        "view": "good_only" if good_only else "all_peaks",
        "color_by": color_by,
        "plotted_points": int(np.count_nonzero(valid)),
        "missing_color_points": int(np.count_nonzero(missing_color)),
        "excluded_frames": int(np.count_nonzero(dataset.excluded)),
        "source": str(dataset.path),
    }


def _plot_pattern_map(dataset: GuiDataset, output_path: Path) -> dict[str, Any]:
    """Render the GUI clean Pattern map on its physical pressure axis."""

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finite_pressure = np.isfinite(dataset.pressure_GPa)
    order = np.argsort(dataset.pressure_GPa[finite_pressure], kind="stable")
    pressures = dataset.pressure_GPa[finite_pressure][order]
    patterns = dataset.patterns[finite_pressure][order].T.astype(float, copy=True)
    excluded = dataset.excluded[finite_pressure][order]
    if np.any(excluded):
        patterns[:, excluded] = np.nan
    positive = patterns[np.isfinite(patterns) & (patterns > 0)]
    vmin = float(np.percentile(positive, 5)) if positive.size else None
    vmax = float(np.percentile(positive, 99)) if positive.size else None
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#a6a6a6")

    fig, ax = plt.subplots(figsize=(10.0, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#a6a6a6")
    mesh = None
    if pressures.size >= 2:
        mesh = ax.pcolormesh(
            pressures, dataset.radial_deg, np.ma.masked_invalid(patterns),
            cmap=cmap, shading="nearest", vmin=vmin, vmax=vmax,
        )
    elif pressures.size == 1:
        extent = [pressures[0] - 0.5, pressures[0] + 0.5,
                  float(dataset.radial_deg.min()), float(dataset.radial_deg.max())]
        mesh = ax.imshow(
            np.ma.masked_invalid(patterns), aspect="auto", origin="lower", cmap=cmap,
            extent=extent, vmin=vmin, vmax=vmax, interpolation="nearest",
        )
    else:
        ax.text(
            0.5, 0.5, "No finite pressure frames",
            transform=ax.transAxes, ha="center", va="center", color="#4d4d4d",
        )
    if mesh is not None:
        fig.colorbar(mesh, ax=ax, label="GUI clean intensity")
    ax.set_title(
        f"GUI Pattern map — {dataset.channel}/{dataset.scan}\n"
        "Pressure axis · source = clean · gray = missing/excluded"
    )
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("2θ (deg)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white", metadata={
        "Title": "GUI clean Pattern map on pressure axis",
        "Description": "Gray indicates missing/nonfinite or excluded data; no algorithm changes.",
    })
    plt.close(fig)
    return {
        "filename": output_path.name,
        "visualization": "gui_pattern_map",
        "channel": dataset.channel,
        "scan": dataset.scan,
        "view": "pressure_x_clean",
        "color_by": "clean_intensity",
        "plotted_points": int(np.count_nonzero(np.isfinite(patterns))),
        "missing_color_points": int(np.count_nonzero(~np.isfinite(patterns))),
        "excluded_frames": int(np.count_nonzero(dataset.excluded)),
        "source": str(dataset.path),
    }


def _plot_v21_matched_overlay(
    dataset: GuiDataset,
    matches: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Overlay v2.1 reliable matches on the GUI good-only pressure Peak map."""

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    good_gui = dataset.peaks[dataset.peaks["gui_good"] == 1].copy()
    selected = matches[
        (matches["result_version"] == "uniform-correlation-v2.1")
        & (matches["match_view"] == "good_only")
        & (matches["match_status"] == "matched")
        & (matches["channel"] == dataset.channel)
        & (matches["scan"] == dataset.scan)
    ].copy()
    matched_gui_ids = set(selected["gui_peak_id"].astype(str))
    unmatched_gui = good_gui[~good_gui["gui_peak_id"].astype(str).isin(matched_gui_ids)]

    fig, ax = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    if not unmatched_gui.empty:
        ax.scatter(
            unmatched_gui["pressure_GPa"], unmatched_gui["gui_two_theta_deg"],
            s=25, facecolors="#bdbdbd", edgecolors="#595959", linewidths=0.35,
            alpha=0.8, label="GUI good, no admissible v2.1 match", zorder=1,
        )
    scatter = None
    if not selected.empty:
        ratios = pd.to_numeric(selected["position_delta_over_gate"], errors="coerce").to_numpy(float)
        finite = np.isfinite(ratios)
        for _, row in selected.iterrows():
            ax.plot(
                [row["pressure_GPa"], row["pressure_GPa"]],
                [row["gui_two_theta_deg"], row["correlation_two_theta_deg"]],
                color="#737373", linewidth=0.45, alpha=0.55, zorder=2,
            )
        ax.scatter(
            selected["pressure_GPa"], selected["gui_two_theta_deg"],
            s=30, facecolors="none", edgecolors="#4d4d4d", linewidths=0.55,
            label="GUI good (matched)", zorder=3,
        )
        if np.any(finite):
            scatter = ax.scatter(
                selected.loc[finite, "pressure_GPa"],
                selected.loc[finite, "correlation_two_theta_deg"],
                c=ratios[finite], cmap="viridis", vmin=0.0, vmax=1.0,
                s=35, marker="x", linewidths=0.9, label="v2.1 reliable match", zorder=4,
            )
            fig.colorbar(scatter, ax=ax, label="|Δcenter| / admissible gate")
    if selected.empty and unmatched_gui.empty:
        ax.text(
            0.5, 0.5, "No GUI-good/v2.1-reliable rows to overlay",
            transform=ax.transAxes, ha="center", va="center", color="#666666",
        )
    ax.set_title(
        f"GUI ↔ uniform-correlation-v2.1 peak overlay — {dataset.channel}/{dataset.scan}\n"
        "Pressure axis · gray = unmatched GUI good"
    )
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Peak center (2θ, deg)")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
    if not selected.empty or not unmatched_gui.empty:
        ax.legend(loc="best", frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white", metadata={
        "Title": "GUI to uniform-correlation-v2.1 matched peak overlay",
        "Description": "Gray marks unmatched GUI-good peaks; validation only.",
    })
    plt.close(fig)
    return {
        "filename": output_path.name,
        "visualization": "gui_v21_matched_peak_overlay",
        "channel": dataset.channel,
        "scan": dataset.scan,
        "view": "good_only_matched_overlay",
        "color_by": "position_delta_over_gate",
        "plotted_points": int(len(selected)),
        "missing_color_points": int(len(unmatched_gui)),
        "excluded_frames": int(np.count_nonzero(dataset.excluded)),
        "source": str(dataset.path),
    }


def write_gui_visualizations(
    output: Path,
    datasets: Sequence[GuiDataset],
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Write fixed-name GUI-equivalent evidence figures for the primary dataset."""

    columns = [
        "filename", "visualization", "channel", "scan", "view", "color_by",
        "plotted_points", "missing_color_points", "excluded_frames", "source",
    ]
    if not datasets:
        return pd.DataFrame(columns=columns)
    # The current official inventory contains exactly spots/scan048.  Fixed names
    # intentionally refer to this primary validation dataset.  Additional GUI
    # inventories remain in the numeric CSV audit and can be plotted in a later
    # inventory-specific extension without overwriting these evidence files.
    dataset = datasets[0]
    rows = [
        _plot_peak_map(dataset, output / FIXED_VISUALIZATION_FILENAMES["all_area"], good_only=False, color_by="area"),
        _plot_peak_map(dataset, output / FIXED_VISUALIZATION_FILENAMES["good_area"], good_only=True, color_by="area"),
        _plot_peak_map(dataset, output / FIXED_VISUALIZATION_FILENAMES["all_fwhm"], good_only=False, color_by="fwhm"),
        _plot_peak_map(dataset, output / FIXED_VISUALIZATION_FILENAMES["good_fwhm"], good_only=True, color_by="fwhm"),
        _plot_pattern_map(dataset, output / FIXED_VISUALIZATION_FILENAMES["pattern_clean"]),
        _plot_v21_matched_overlay(dataset, matches, output / FIXED_VISUALIZATION_FILENAMES["matched_overlay"]),
    ]
    return pd.DataFrame(rows, columns=columns)


def _fmt_number(value: Any, digits: int = 3) -> str:
    number = _safe_float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _overall_match_row(
    summaries: pd.DataFrame,
    result_version: str,
    view: str,
    channel: str = "spots",
) -> pd.Series | None:
    if summaries.empty:
        return None
    selected = summaries[
        (summaries["result_version"] == result_version)
        & (summaries["match_view"] == view)
        & (summaries["channel"] == channel)
        & (summaries["row_kind"] == "overall")
    ]
    return selected.iloc[0] if not selected.empty else None


def write_gui_crosscheck_report(
    output: Path,
    *,
    result_root: Path,
    legacy_root: Path | None,
    datasets: Sequence[GuiDataset],
    match_summaries: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    pattern_checks: pd.DataFrame,
    control_auc: pd.DataFrame,
    control_boundaries: pd.DataFrame,
    total_spots_scans: int,
    domain_comparison: dict[str, Any] | None = None,
) -> Path:
    """Write a concise, data-driven Chinese report beside the evidence files."""

    gui_spots = [item for item in datasets if item.channel == "spots"]
    gui_fit = [item for item in datasets if item.channel == "fit"]
    gui_spots_scans = len({item.scan for item in gui_spots})
    gui_total = sum(len(item.peaks) for item in gui_spots)
    gui_good = sum(int(item.peaks["gui_good"].sum()) for item in gui_spots)
    coverage = gui_spots_scans / total_spots_scans if total_spots_scans else math.nan
    domain_comparison = domain_comparison or {}

    main_good = _overall_match_row(
        match_summaries, "uniform-correlation-v2.1", "good_only"
    )
    main_all = _overall_match_row(
        match_summaries, "uniform-correlation-v2.1", "all_peaks"
    )
    legacy_good = _overall_match_row(
        match_summaries, "legacy-uniform-correlation-v2", "good_only"
    )

    def match_sentence(label: str, row: pd.Series | None) -> str:
        if row is None:
            return f"- {label}：没有可计算的匹配行。"
        return (
            f"- {label}：correlation 候选 {int(row['correlation_candidates'])} 个，"
            f"GUI 候选 {int(row['gui_candidates'])} 个，一对一匹配 {int(row['matched'])} 个；"
            f"峰位置中位差 {_fmt_number(row['median_position_delta_deg'], 6)}°，"
            f"90% 分位差 {_fmt_number(row['q90_position_delta_deg'], 6)}°；"
            f"GUI/correlation FWHM 中位比 {_fmt_number(row['median_gui_over_correlation_fwhm_ratio'])}，"
            f"FWHM Spearman {_fmt_number(row['fwhm_spearman'])}；"
            f"area 中位比 {_fmt_number(row['median_gui_over_correlation_area_ratio'])}，"
            f"area Spearman {_fmt_number(row['area_spearman'])}。"
        )

    boundary_lines: list[str] = []
    if not boundary_summary.empty:
        for channel in ("spots", "fit"):
            subset = boundary_summary[boundary_summary["channel"] == channel]
            if subset.empty:
                continue
            boundary_lines.append(
                f"- {channel}：26 个窗口中，ACF/direct 低相关排序 Spearman 中位数 "
                f"{_fmt_number(subset['boundary_score_spearman'].median())}，"
                f"全部格相同 call 比例中位数 {_fmt_number(subset['exact_same_call_fraction'].median())}，"
                f"正候选集合 Jaccard 中位数 {_fmt_number(subset['top_quartile_call_jaccard'].median())}；"
                f"每窗口共同支持候选中位数 "
                f"{_fmt_number(subset['corroborated_candidate_count'].median(), 1)}，"
                f"单-family unresolved 候选中位数 "
                f"{_fmt_number(subset['unresolved_candidate_count'].median(), 1)}；"
                f"相邻压力格的最低 support 为 "
                f"ACF {int(pd.to_numeric(subset['minimum_acf_support'], errors='coerce').min())}/"
                f"direct {int(pd.to_numeric(subset['minimum_direct_support'], errors='coerce').min())}。"
            )
    if not boundary_lines:
        boundary_lines.append("- 没有可读取的 strict ACF/direct 低相关候选表。")

    called_pattern = pattern_checks[
        (pattern_checks["acf_boundary_call"] == 1)
        | (pattern_checks["direct_boundary_call"] == 1)
    ] if not pattern_checks.empty else pd.DataFrame()
    if called_pattern.empty:
        pattern_sentence = "没有可计算的 GUI Pattern map 低相关候选窗口。"
    else:
        pattern_sentence = (
            f"在现有 GUI scan 覆盖到的 {len(called_pattern)} 个被任一 strict family 标记的"
            f"窗口×相邻压力单元中，GUI clean Pattern map 的 direct Pearson 中位数 "
            f"{_fmt_number(pd.to_numeric(called_pattern['gui_direct_pearson_same_window'], errors='coerce').median())}，"
            f"ACF Pearson 中位数 "
            f"{_fmt_number(pd.to_numeric(called_pattern['gui_acf_pearson_same_window'], errors='coerce').median())}，"
            f"归一化强度重分配中位数 "
            f"{_fmt_number(pd.to_numeric(called_pattern['gui_normalized_L1_intensity_redistribution'], errors='coerce').median())}；"
            f"{_fmt_number(100.0 * pd.to_numeric(called_pattern['gui_unmatched_or_appearance_disappearance_candidate'], errors='coerce').mean(), 1)}% "
            "伴随 GUI-good 峰计数变化或无法跨区间一对一匹配；这是候选诊断，不单独证明峰分裂。"
        )

    auc_values: dict[tuple[str, str], float] = {}
    if not control_auc.empty:
        for (channel, family), group in control_auc.groupby(["channel", "family"]):
            auc_values[(str(channel), str(family))] = _safe_float(
                pd.to_numeric(group["near_vs_far_auc"], errors="coerce").median()
            )
    control_boundary_sentence = "spots/fit 低相关候选比较不可计算。"
    if not control_boundaries.empty:
        acf_union = int(pd.to_numeric(control_boundaries["either_positive_acf_candidate_spots_fit"], errors="coerce").sum())
        direct_union = int(pd.to_numeric(control_boundaries["either_positive_direct_candidate_spots_fit"], errors="coerce").sum())
        acf_jaccard = (
            float(pd.to_numeric(control_boundaries["shared_positive_acf_candidate_spots_fit"], errors="coerce").sum()) / acf_union
            if acf_union else math.nan
        )
        direct_jaccard = (
            float(pd.to_numeric(control_boundaries["shared_positive_direct_candidate_spots_fit"], errors="coerce").sum()) / direct_union
            if direct_union else math.nan
        )
        control_boundary_sentence = (
            "spots 与 fit 的全部格相同 call 比例为："
            f"ACF {_fmt_number(pd.to_numeric(control_boundaries['same_acf_boundary_call_spots_fit'], errors='coerce').mean())}，"
            f"direct {_fmt_number(pd.to_numeric(control_boundaries['same_direct_boundary_call_spots_fit'], errors='coerce').mean())}；"
            f"更关键的正候选 Jaccard 为 ACF {_fmt_number(acf_jaccard)}、"
            f"direct {_fmt_number(direct_jaccard)}。"
        )

    spots_domain = domain_comparison.get("spots", {})
    domain_sentence = "没有可计算的 GUI/correlation 角度范围比较。"
    if spots_domain:
        domain_sentence = (
            f"GUI 角度范围为 {_fmt_number(spots_domain.get('gui_two_theta_min_deg'), 4)}–"
            f"{_fmt_number(spots_domain.get('gui_two_theta_max_deg'), 4)}°；"
            f"同一 GUI scan 中有 {int(spots_domain.get('correlation_reliable_peaks_outside_gui_domain', 0))} 个"
            " correlation reliable 峰落在 GUI 拟合范围外，这些不计作算法不一致。"
        )

    lines = [
        "# GUI × uniform-correlation-v2.1 交叉检查",
        "",
        "## 结论范围",
        "",
        f"当前 GUI 数据只覆盖 spots 的 {gui_spots_scans}/{total_spots_scans} 个 scan "
        f"（{_fmt_number(100.0 * coverage, 1)}%），即 `scan048` 的 19 个压力。"
        f"GUI Peak map 中共有 {gui_total} 个 fitted peaks，其中 {gui_good} 个为 good "
        f"（`flag == 0`）。fit/tungsten 的 GUI HDF5 数量为 {len(gui_fit)}。",
        "",
        "因此本检查只能作为局部、独立的 GUI 证据，不能代表 56 个 scan 已全部通过 GUI 验证。"
        "GUI 差异不会用于修改或放宽 v2.1。",
        "",
        "## Peak map：Good peaks only 开/关",
        "",
        match_sentence("v2.1，Good peaks only 开", main_good),
        match_sentence("v2.1，Good peaks only 关", main_all),
        match_sentence("legacy v2 参照，Good peaks only 开", legacy_good),
        f"- {domain_sentence}",
        "",
        "GUI 与 correlation 使用不同的接受/拒绝规则，因此 rejected reason 文本不必完全一致。"
        "这里更重视相同 scan+压力下的位置、area/FWHM趋势、持续压力范围与斜率。"
        "轨迹范围和斜率是以 correlation 的 segment identity 为条件的局部佐证，"
        "不是 GUI 独立重新追踪得到的第二套 track。",
        "",
        "![GUI Peak map：全部 peaks，area](gui_peak_map_pressure_all_area.png)",
        "",
        "![GUI Peak map：good only，area](gui_peak_map_pressure_good_only_area.png)",
        "",
        "![GUI Peak map：全部 peaks，FWHM](gui_peak_map_pressure_all_fwhm.png)",
        "",
        "![GUI Peak map：good only，FWHM](gui_peak_map_pressure_good_only_fwhm.png)",
        "",
        "![GUI 与 v2.1 匹配峰 overlay](gui_v21_matched_peak_overlay_pressure.png)",
        "",
        "## ACF-strict 与 direct-strict 低相关候选区间",
        "",
        *boundary_lines,
        "",
        "Q25 只用于找每个窗口、每个 family 的相对低相关候选，不是物理相边界。"
        "只有 ACF/direct 在同一相邻压力区间都选中时才叫 corroborated；"
        "只被一个 family 选中的全部列为 unresolved。表格同时保存各自的 support 与 "
        "95% scan-bootstrap CI；两种不同分数的 CI 数值重叠不作为一致性证据。",
        "",
        "## Pattern map 同一 2θ window",
        "",
        pattern_sentence,
        "",
        "这些连续量分别帮助判断低相关是否伴随峰消失/出现、移动/无法匹配，或强度重分配；"
        "单个 scan 的视觉变化不能单独证明 UOTe 相变。",
        "",
        "![GUI clean Pattern map](gui_pattern_map_pressure_clean.png)",
        "",
        "## fit/tungsten control",
        "",
        f"- spots strict-ACF AUC 中位数：{_fmt_number(auc_values.get(('spots', 'acf_strict')))}；"
        f"fit strict-ACF：{_fmt_number(auc_values.get(('fit', 'acf_strict')))}。",
        f"- spots direct-strict AUC 中位数：{_fmt_number(auc_values.get(('spots', 'direct_strict')))}；"
        f"fit direct-strict：{_fmt_number(auc_values.get(('fit', 'direct_strict')))}。",
        f"- {control_boundary_sentence}",
        "- 当前没有 fit/tungsten GUI HDF5，所以 fit control 是 correlation 内部对照，不是 GUI fit 对照。"
        "如果 spots 与 fit 在同一压力候选区间出现相同结构，应优先考虑压力标记物或系统效应，"
        "不能直接解释为 UOTe 特有变化。",
        "",
        "## 可审计来源",
        "",
        f"- v2.1：`{result_root}`",
        f"- legacy v2：`{legacy_root}`" if legacy_root else "- legacy v2：未提供",
        "- 数值明细：`peak_match_table.csv`、`track_range_slope_comparison.csv`、"
        "`strict_boundary_cells.csv`、`pattern_window_checks.csv`、`spots_fit_control_window_auc.csv`。",
        "",
    ]
    report_path = output / "GUI_CROSSCHECK_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _write_csv(path: Path, table: pd.DataFrame, columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if table.empty and columns is not None:
        table = pd.DataFrame(columns=list(columns))
    table.to_csv(path, index=False)


def _json_clean(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats to strict-JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _channel_scan_count(root: Path, channel: str) -> int:
    path = root / channel / "per_peak" / "detected_peak_fits.csv"
    if not path.is_file():
        return 0
    data = pd.read_csv(path, usecols=["scan"])
    return int(data["scan"].astype(str).nunique())


def run(args: argparse.Namespace) -> dict[str, Any]:
    result_root = Path(args.result_root).expanduser().resolve()
    legacy_root = Path(args.legacy_root).expanduser().resolve() if args.legacy_root else None
    result_manifest = _require_run_profile(
        result_root, "uniform-correlation-v2.1", role="result"
    )
    legacy_manifest: dict[str, Any] | None = None
    if legacy_root is not None:
        if legacy_root == result_root:
            raise ValueError("result-root and legacy-root must be different directories")
        legacy_manifest = _require_run_profile(
            legacy_root, "uniform-correlation-v2", role="legacy"
        )
    inventory_path = Path(args.gui_inventory).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_root / "validation" / "gui_crosscheck"
    )
    output.mkdir(parents=True, exist_ok=True)
    datasets = _load_gui_inventory(inventory_path)
    gui_peaks = pd.concat([item.peaks for item in datasets], ignore_index=True) if datasets else pd.DataFrame()
    _write_csv(output / "gui_peak_table.csv", gui_peaks)

    versions: list[tuple[str, Path]] = [("uniform-correlation-v2.1", result_root)]
    if legacy_root is not None:
        versions.append(("legacy-uniform-correlation-v2", legacy_root))
    all_matches: list[pd.DataFrame] = []
    all_tracks: list[pd.DataFrame] = []
    corr_peak_tables: dict[tuple[str, str], pd.DataFrame] = {}
    gui_channels = sorted(set(gui_peaks.get("channel", pd.Series(dtype=str)).astype(str)))
    for label, root in versions:
        for channel in gui_channels:
            corr = _load_correlation_peaks(root, channel, label)
            corr_peak_tables[(label, channel)] = corr
            matches = match_peak_tables(corr, gui_peaks[gui_peaks["channel"] == channel], result_version=label)
            all_matches.append(matches)
            all_tracks.append(
                build_track_comparison(
                    root, channel, label, corr, matches,
                    set(gui_peaks.loc[gui_peaks["channel"] == channel, "scan"].astype(str)),
                )
            )
    matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame(columns=MATCH_COLUMNS)
    tracks = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame(columns=TRACK_COLUMNS)
    summaries = _summarize_matches(matches)
    reasons = _reason_crosstab(matches)
    _write_csv(output / "peak_match_table.csv", matches, MATCH_COLUMNS)
    _write_csv(output / "peak_match_summary.csv", summaries)
    _write_csv(output / "rejection_reason_agreement.csv", reasons)
    _write_csv(output / "track_range_slope_comparison.csv", tracks, TRACK_COLUMNS)
    visualization_manifest = write_gui_visualizations(output, datasets, matches)
    _write_csv(output / "visualization_manifest.csv", visualization_manifest)

    boundary_tables: list[pd.DataFrame] = []
    boundary_summaries: list[pd.DataFrame] = []
    for channel in ("spots", "fit"):
        table, summary = build_strict_boundary_table(result_root, channel)
        boundary_tables.append(table)
        boundary_summaries.append(summary)
    boundaries = pd.concat(boundary_tables, ignore_index=True) if boundary_tables else pd.DataFrame(columns=BOUNDARY_COLUMNS)
    boundary_summary = pd.concat(boundary_summaries, ignore_index=True) if boundary_summaries else pd.DataFrame()
    _write_csv(output / "strict_boundary_cells.csv", boundaries, BOUNDARY_COLUMNS)
    _write_csv(output / "strict_boundary_agreement_summary.csv", boundary_summary)

    pattern_checks = build_pattern_checks(boundaries, datasets)
    _write_csv(output / "pattern_window_checks.csv", pattern_checks, PATTERN_COLUMNS)
    control_auc, control_pivot = build_control_tables(result_root)
    control_boundaries = build_control_boundary_agreement(boundaries)
    _write_csv(output / "spots_fit_control_window_auc.csv", control_auc)
    _write_csv(output / "spots_fit_control_auc_comparison.csv", control_pivot)
    _write_csv(output / "spots_fit_control_boundary_comparison.csv", control_boundaries)

    gui_scan_by_channel = {
        channel: sorted(set(gui_peaks.loc[gui_peaks["channel"] == channel, "scan"].astype(str)))
        for channel in gui_channels
    }
    domain_comparison: dict[str, Any] = {}
    for channel in gui_channels:
        channel_sets = [item for item in datasets if item.channel == channel]
        if not channel_sets:
            continue
        gui_min = min(float(np.nanmin(item.radial_deg)) for item in channel_sets)
        gui_max = max(float(np.nanmax(item.radial_deg)) for item in channel_sets)
        corr = corr_peak_tables.get(("uniform-correlation-v2.1", channel), pd.DataFrame()).copy()
        if not corr.empty:
            corr = corr[
                corr["scan"].astype(str).isin(gui_scan_by_channel.get(channel, []))
                & (corr["state"].astype(str).str.lower() == "reliable")
            ]
        centers = pd.to_numeric(corr.get("two_theta_deg", pd.Series(dtype=float)), errors="coerce")
        finite = centers[np.isfinite(centers)]
        domain_comparison[channel] = {
            "gui_two_theta_min_deg": gui_min,
            "gui_two_theta_max_deg": gui_max,
            "correlation_reliable_center_min_deg": float(finite.min()) if len(finite) else math.nan,
            "correlation_reliable_center_max_deg": float(finite.max()) if len(finite) else math.nan,
            "correlation_reliable_peaks_in_gui_scans": int(len(finite)),
            "correlation_reliable_peaks_outside_gui_domain": int(
                np.count_nonzero((finite.to_numpy(float) < gui_min) | (finite.to_numpy(float) > gui_max))
            ) if len(finite) else 0,
        }
    total_spots_scans = _channel_scan_count(result_root, "spots")
    gui_spots_scans = len(gui_scan_by_channel.get("spots", []))
    gui_fit_scans = len(gui_scan_by_channel.get("fit", []))
    report_path = write_gui_crosscheck_report(
        output,
        result_root=result_root,
        legacy_root=legacy_root,
        datasets=datasets,
        match_summaries=summaries,
        boundary_summary=boundary_summary,
        pattern_checks=pattern_checks,
        control_auc=control_auc,
        control_boundaries=control_boundaries,
        total_spots_scans=total_spots_scans,
        domain_comparison=domain_comparison,
    )
    main_good = summaries[
        (summaries["result_version"] == "uniform-correlation-v2.1")
        & (summaries["match_view"] == "good_only")
        & (summaries["row_kind"] == "overall")
    ] if not summaries.empty else pd.DataFrame()
    auc_medians: dict[str, Any] = {}
    if not control_auc.empty:
        for (channel, family), group in control_auc.groupby(["channel", "family"]):
            auc_medians[f"{channel}:{family}"] = _safe_float(
                pd.to_numeric(group["near_vs_far_auc"], errors="coerce").median()
            )
    summary = {
        "checker": "compare_gui_to_v21.py",
        "role": "read_only_validation_does_not_change_algorithms_or_results",
        "result_root": str(result_root),
        "legacy_root": str(legacy_root) if legacy_root else None,
        "result_profile_verified": result_manifest.get("profile"),
        "legacy_profile_verified": legacy_manifest.get("profile") if legacy_manifest else None,
        "gui_inventory": str(inventory_path),
        "output_dir": str(output),
        "matching_formula": MATCH_FORMULA,
        "boundary_formula": BOUNDARY_FORMULA,
        "gui_flag_definitions": {str(bit): name for bit, name in FLAG_DEFINITIONS},
        "gui_good_definition": "flag == 0",
        "correlation_good_definition": "state == reliable",
        "gui_datasets": [
            {
                "channel": item.channel, "scan": item.scan, "analysis_h5": str(item.path),
                "pattern_source": item.pattern_source, "unit": item.unit,
                "radial_step_deg": item.radial_step_deg,
                "n_frames": int(item.pressure_GPa.size), "n_peaks": int(len(item.peaks)),
                "n_good_peaks": int(item.peaks["gui_good"].sum()),
            }
            for item in datasets
        ],
        "coverage": {
            "spots_gui_scans": gui_spots_scans,
            "spots_correlation_scans": total_spots_scans,
            "spots_fraction": gui_spots_scans / total_spots_scans if total_spots_scans else math.nan,
            "fit_gui_scans": gui_fit_scans,
            "fit_gui_h5_missing": gui_fit_scans == 0,
        },
        "radial_domain_comparison": domain_comparison,
        "main_v21_good_only_match_summary": main_good.to_dict("records"),
        "strict_boundary_rows": int(len(boundaries)),
        "pattern_window_rows": int(len(pattern_checks)),
        "report_file": report_path.name,
        "visualization_files": visualization_manifest["filename"].tolist()
        if not visualization_manifest.empty else [],
        "median_near_far_auc": auc_medians,
        "verdict": (
            "partial_gui_coverage"
            if gui_spots_scans < total_spots_scans or gui_fit_scans == 0
            else "complete_gui_coverage"
        ),
        "interpretation_guardrails": [
            "GUI and uniform-correlation use independent peak-fit acceptance rules; rejected-reason text need not match exactly.",
            "Peak-map disagreement is reported, never used to loosen or retune v2.1.",
            "A structure shared by spots and fit/tungsten may be a pressure-marker or system effect, not UOTe-specific evidence.",
            "A scan048-only GUI result cannot validate an aggregate made from all scans.",
            "Q25 calls are rank-based low-similarity candidates, not physical phase boundaries.",
            "Only exact ACF/direct positive intersections are corroborated; one-family candidates remain unresolved.",
            "Cross-family CI numeric overlap is descriptive only because ACF and direct use different score scales.",
            "Track slope/range comparison is conditional on the correlation track identity, not independent GUI-only tracking.",
        ],
    }
    summary = _json_clean(summary)
    (output / "crosscheck_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, help="uniform-correlation-v2.1 run root")
    parser.add_argument("--legacy-root", help="optional frozen uniform-correlation-v2 root")
    parser.add_argument("--gui-inventory", required=True, help="CSV: channel,scan,analysis_h5[,pattern_source]")
    parser.add_argument("--output-dir", help="default: RESULT/validation/gui_crosscheck")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
