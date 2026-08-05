#!/usr/bin/env python3
"""Build robust powder peak tracks and exploratory pressure-transition evidence.

This analysis combines four powder ROI definitions:

* q-width factor c=0.60, log-squared transform
* q-width factor c=0.60, exp-squared transform
* q-width factor c=0.75, log-squared transform
* q-width factor c=0.75, exp-squared transform

Adjacent-pressure candidate peaks are first gated by |delta 2theta| < 0.06 deg.
For each definition the directional ROI scores are symmetrised as
sqrt(S_A_to_B * S_B_to_A).  A conservative edge score is the minimum across
the four definitions.  Dummy-augmented Hungarian matching then gives a global
one-to-one assignment while allowing births and deaths.

The script also summarises powder window correlations, builds exploratory
multi-indicator transition scores, and aligns those intervals to independent
single-crystal evidence where pressure coverage exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path("/Users/stanley/x-ray")
RESULTS_ROOT = REPO_ROOT / "correlations" / "results"
C60_ROOT = RESULTS_ROOT / "uote_nonlinear_squared_preprocessed_comparison_20260802"
C75_ROOT = RESULTS_ROOT / "uote_nonlinear_squared_qwidth075_comparison_20260803"
DEFAULT_OUTPUT = RESULTS_ROOT / "uote_robust_peak_tracks_transition_analysis_20260804"

DEFINITIONS = {
    "c60_log": (C60_ROOT, "log_squared"),
    "c60_exp": (C60_ROOT, "exp_squared"),
    "c75_log": (C75_ROOT, "log_squared"),
    "c75_exp": (C75_ROOT, "exp_squared"),
}

PRIMARY_THRESHOLD = 0.10
SENSITIVITY_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)
LOCATION_GATE_DEG = 0.06
EPS = 1e-12


def fmt_pressure(value: float) -> str:
    return f"{float(value):g}"


def threshold_tag(value: float) -> str:
    return f"t{int(round(value * 100)):03d}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def finite_summary(values: Iterable[float], prefix: str) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            f"{prefix}_n": 0,
            f"{prefix}_median": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_n": int(len(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def symmetric_log_change(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(abs(math.log((max(b, 0.0) + EPS) / (max(a, 0.0) + EPS))))


def empirical_percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    filled = numeric.fillna(numeric.max() + max(abs(numeric.max()), 1.0) * 1e-9)
    ranks = filled.rank(method="average")
    return (ranks - 1.0) / (len(ranks) - 1.0)


def load_powder_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    registries: dict[str, pd.DataFrame] = {}
    directed: dict[str, pd.DataFrame] = {}
    catalog_signature: pd.DataFrame | None = None

    for name, (root, mode) in DEFINITIONS.items():
        source = root / "_sources" / mode / "powder_roi"
        registry = pd.read_csv(source / "point_registry.csv")
        registry = registry.sort_values(["pressure_gpa", "local_peak_index"]).reset_index(drop=True)
        signature = registry[
            ["point_uid", "pressure_gpa", "local_peak_index", "q", "two_theta_deg"]
        ].reset_index(drop=True)
        if catalog_signature is None:
            catalog_signature = signature
        else:
            pd.testing.assert_frame_equal(catalog_signature, signature, check_exact=True)
        registries[name] = registry

        cols = [
            "anchor_point_uid",
            "target_point_uid",
            "anchor_pressure_gpa",
            "target_pressure_gpa",
            "anchor_local_peak_index",
            "target_local_peak_index",
            "anchor_two_theta_deg",
            "target_two_theta_deg",
            "location_similarity",
            "spots_absolute_anchor_ROI_iou",
            "supports_overlap",
            "zero_reason",
        ]
        pairs = pd.read_csv(source / "all_directed_cross_pressure_peak_pairs.csv.gz", usecols=cols)
        if pairs.duplicated(["anchor_point_uid", "target_point_uid"]).any():
            raise RuntimeError(f"Duplicate directed powder pair in {name}")
        directed[name] = pairs.set_index(["anchor_point_uid", "target_point_uid"], drop=False)

    return registries["c75_log"].copy(), registries, directed


def build_adjacent_candidate_table(
    registry: pd.DataFrame,
    directed: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    pressures = np.sort(registry["pressure_gpa"].unique())
    rows: list[dict[str, Any]] = []
    location_source = directed["c75_log"]

    for pressure_a, pressure_b in zip(pressures[:-1], pressures[1:]):
        peaks_a = registry.loc[registry["pressure_gpa"].eq(pressure_a)].copy()
        peaks_b = registry.loc[registry["pressure_gpa"].eq(pressure_b)].copy()
        for peak_a in peaks_a.itertuples(index=False):
            for peak_b in peaks_b.itertuples(index=False):
                key_ab = (peak_a.point_uid, peak_b.point_uid)
                key_ba = (peak_b.point_uid, peak_a.point_uid)
                loc_ab = location_source.loc[key_ab]
                location = float(loc_ab["location_similarity"])
                if not location > 0.0:
                    continue
                row: dict[str, Any] = {
                    "pressure_a_gpa": float(pressure_a),
                    "pressure_b_gpa": float(pressure_b),
                    "transition": f"{fmt_pressure(pressure_a)}->{fmt_pressure(pressure_b)}",
                    "point_uid_a": peak_a.point_uid,
                    "point_uid_b": peak_b.point_uid,
                    "local_peak_a": int(peak_a.local_peak_index),
                    "local_peak_b": int(peak_b.local_peak_index),
                    "q_a": float(peak_a.q),
                    "q_b": float(peak_b.q),
                    "two_theta_a_deg": float(peak_a.two_theta_deg),
                    "two_theta_b_deg": float(peak_b.two_theta_deg),
                    "delta_two_theta_deg": float(peak_b.two_theta_deg - peak_a.two_theta_deg),
                    "abs_delta_two_theta_deg": float(abs(peak_b.two_theta_deg - peak_a.two_theta_deg)),
                    "location_similarity": location,
                }
                mutual_scores: list[float] = []
                for definition, table in directed.items():
                    ab = table.loc[key_ab]
                    ba = table.loc[key_ba]
                    score_ab = float(ab["spots_absolute_anchor_ROI_iou"])
                    score_ba = float(ba["spots_absolute_anchor_ROI_iou"])
                    mutual = float(math.sqrt(max(score_ab, 0.0) * max(score_ba, 0.0)))
                    row[f"{definition}_a_to_b"] = score_ab
                    row[f"{definition}_b_to_a"] = score_ba
                    row[f"{definition}_mutual"] = mutual
                    mutual_scores.append(mutual)
                mutual_array = np.asarray(mutual_scores, dtype=float)
                row["robust_mutual_min"] = float(np.min(mutual_array))
                row["robust_mutual_geomean"] = (
                    float(np.prod(mutual_array) ** (1.0 / len(mutual_array)))
                    if np.all(mutual_array > 0)
                    else 0.0
                )
                row["mutual_range_across_definitions"] = float(np.max(mutual_array) - np.min(mutual_array))
                row["all_four_mutual_positive"] = int(np.all(mutual_array > 0.0))
                row["c75_positive_c60_zero"] = int(
                    ((row["c75_log_mutual"] > 0) and (row["c60_log_mutual"] == 0))
                    or ((row["c75_exp_mutual"] > 0) and (row["c60_exp_mutual"] == 0))
                )
                rows.append(row)

    candidates = pd.DataFrame(rows)
    candidates = candidates.sort_values(
        ["pressure_a_gpa", "local_peak_a", "local_peak_b"]
    ).reset_index(drop=True)
    return candidates


def solve_dummy_augmented_matching(
    transition_candidates: pd.DataFrame,
    peaks_a: pd.DataFrame,
    peaks_b: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> set[tuple[str, str]]:
    """Maximum-total-score one-to-one matching with zero-score dummy nodes."""
    peaks_a = peaks_a.sort_values("local_peak_index").reset_index(drop=True)
    peaks_b = peaks_b.sort_values("local_peak_index").reset_index(drop=True)
    n_a, n_b = len(peaks_a), len(peaks_b)
    weights = np.full((n_a + n_b, n_a + n_b), -1e9, dtype=float)
    weights[:n_a, n_b:] = 0.0
    weights[n_a:, :n_b] = 0.0
    weights[n_a:, n_b:] = 0.0
    a_lookup = {uid: idx for idx, uid in enumerate(peaks_a["point_uid"])}
    b_lookup = {uid: idx for idx, uid in enumerate(peaks_b["point_uid"])}

    for candidate in transition_candidates.itertuples(index=False):
        score = float(getattr(candidate, score_column))
        if score + 1e-15 < threshold:
            continue
        i = a_lookup[candidate.point_uid_a]
        j = b_lookup[candidate.point_uid_b]
        weights[i, j] = score + 1e-9 * float(candidate.location_similarity)

    row_ind, col_ind = linear_sum_assignment(-weights)
    selected: set[tuple[str, str]] = set()
    for i, j in zip(row_ind, col_ind):
        if i < n_a and j < n_b and weights[i, j] > 0.0:
            selected.add((str(peaks_a.loc[i, "point_uid"]), str(peaks_b.loc[j, "point_uid"])))
    return selected


def run_all_matchings(
    registry: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pressures = np.sort(registry["pressure_gpa"].unique())
    candidate_flags = candidates.copy()
    matching_summary_rows: list[dict[str, Any]] = []

    for threshold in SENSITIVITY_THRESHOLDS:
        tag = threshold_tag(threshold)
        candidate_flags[f"selected_robust_{tag}"] = 0
        for definition in DEFINITIONS:
            candidate_flags[f"selected_{definition}_{tag}"] = 0

    for pressure_a, pressure_b in zip(pressures[:-1], pressures[1:]):
        mask = candidate_flags["pressure_a_gpa"].eq(pressure_a) & candidate_flags[
            "pressure_b_gpa"
        ].eq(pressure_b)
        transition = candidate_flags.loc[mask]
        peaks_a = registry.loc[registry["pressure_gpa"].eq(pressure_a)]
        peaks_b = registry.loc[registry["pressure_gpa"].eq(pressure_b)]

        for threshold in SENSITIVITY_THRESHOLDS:
            tag = threshold_tag(threshold)
            robust_selected = solve_dummy_augmented_matching(
                transition, peaks_a, peaks_b, "robust_mutual_min", threshold
            )
            definition_sets: dict[str, set[tuple[str, str]]] = {}
            for definition in DEFINITIONS:
                definition_sets[definition] = solve_dummy_augmented_matching(
                    transition,
                    peaks_a,
                    peaks_b,
                    f"{definition}_mutual",
                    threshold,
                )
            strict_consensus = set.intersection(*definition_sets.values())

            pair_series = list(
                zip(candidate_flags.loc[mask, "point_uid_a"], candidate_flags.loc[mask, "point_uid_b"])
            )
            candidate_flags.loc[mask, f"selected_robust_{tag}"] = [
                int(pair in robust_selected) for pair in pair_series
            ]
            for definition, selected in definition_sets.items():
                candidate_flags.loc[mask, f"selected_{definition}_{tag}"] = [
                    int(pair in selected) for pair in pair_series
                ]

            matching_summary_rows.append(
                {
                    "pressure_a_gpa": float(pressure_a),
                    "pressure_b_gpa": float(pressure_b),
                    "transition": f"{fmt_pressure(pressure_a)}->{fmt_pressure(pressure_b)}",
                    "threshold": threshold,
                    "n_peaks_a": int(len(peaks_a)),
                    "n_peaks_b": int(len(peaks_b)),
                    "location_gated_candidates": int(len(transition)),
                    "robust_min_matches": int(len(robust_selected)),
                    "strict_four_assignment_consensus_matches": int(len(strict_consensus)),
                    **{
                        f"{definition}_matches": int(len(selected))
                        for definition, selected in definition_sets.items()
                    },
                }
            )

    primary_tag = threshold_tag(PRIMARY_THRESHOLD)
    individual_cols = [f"selected_{definition}_{primary_tag}" for definition in DEFINITIONS]
    candidate_flags["individual_assignment_agreement_at_primary"] = candidate_flags[
        individual_cols
    ].sum(axis=1)
    candidate_flags["strict_four_assignment_consensus_at_primary"] = (
        candidate_flags["individual_assignment_agreement_at_primary"].eq(4).astype(int)
    )
    candidate_flags["primary_reliable_edge"] = candidate_flags[
        f"selected_robust_{primary_tag}"
    ].astype(int)

    def edge_confidence(row: pd.Series) -> str:
        if not row["primary_reliable_edge"]:
            if row[f"selected_robust_{threshold_tag(0.05)}"]:
                return "sensitivity_only_0.05"
            return "not_selected"
        agreement = int(row["individual_assignment_agreement_at_primary"])
        if row[f"selected_robust_{threshold_tag(0.30)}"] and agreement == 4:
            return "high"
        if row[f"selected_robust_{threshold_tag(0.20)}"] and agreement >= 3:
            return "moderate"
        return "reliable_minimum"

    candidate_flags["edge_confidence"] = candidate_flags.apply(edge_confidence, axis=1)
    primary_edges = candidate_flags.loc[candidate_flags["primary_reliable_edge"].eq(1)].copy()
    primary_edges = primary_edges.sort_values(
        ["pressure_a_gpa", "local_peak_a", "local_peak_b"]
    ).reset_index(drop=True)
    return candidate_flags, primary_edges, pd.DataFrame(matching_summary_rows)


def build_tracks(
    registry: pd.DataFrame,
    primary_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    successor = dict(zip(primary_edges["point_uid_a"], primary_edges["point_uid_b"]))
    predecessor = dict(zip(primary_edges["point_uid_b"], primary_edges["point_uid_a"]))
    if len(successor) != len(primary_edges) or len(predecessor) != len(primary_edges):
        raise RuntimeError("Primary matching is not one-to-one")

    registry_by_uid = registry.set_index("point_uid", drop=False)
    ordered_uids = list(
        registry.sort_values(["pressure_gpa", "local_peak_index"])["point_uid"].astype(str)
    )
    starts = [uid for uid in ordered_uids if uid not in predecessor]
    node_to_track: dict[str, str] = {}
    track_meta: dict[str, dict[str, Any]] = {}
    track_rows: list[dict[str, Any]] = []

    for track_number, start_uid in enumerate(starts, start=1):
        track_id = f"RPT_{track_number:04d}"
        chain: list[str] = []
        uid = start_uid
        while uid:
            if uid in node_to_track:
                raise RuntimeError(f"Cycle or duplicate node in tracks at {uid}")
            node_to_track[uid] = track_id
            chain.append(uid)
            uid = successor.get(uid, "")
        chain_data = registry_by_uid.loc[chain]
        track_meta[track_id] = {
            "track_id": track_id,
            "n_pressure_points": int(len(chain)),
            "start_pressure_gpa": float(chain_data["pressure_gpa"].min()),
            "end_pressure_gpa": float(chain_data["pressure_gpa"].max()),
            "pressure_span_gpa": float(
                chain_data["pressure_gpa"].max() - chain_data["pressure_gpa"].min()
            ),
            "start_point_uid": chain[0],
            "end_point_uid": chain[-1],
            "point_uids": ";".join(chain),
        }

    if len(node_to_track) != len(registry):
        missing = set(registry["point_uid"].astype(str)) - set(node_to_track)
        raise RuntimeError(f"Tracks do not cover every node: {sorted(missing)[:5]}")

    for row in registry.sort_values(["pressure_gpa", "local_peak_index"]).itertuples(index=False):
        track_id = node_to_track[row.point_uid]
        meta = track_meta[track_id]
        event = "continuing"
        if row.point_uid == meta["start_point_uid"]:
            event = "initial" if float(row.pressure_gpa) == float(registry["pressure_gpa"].min()) else "birth"
        if row.point_uid == meta["end_point_uid"] and row.point_uid != meta["start_point_uid"]:
            event = "terminal" if float(row.pressure_gpa) == float(registry["pressure_gpa"].max()) else "death"
        if row.point_uid == meta["start_point_uid"] == meta["end_point_uid"]:
            event = "singleton"
        track_rows.append(
            {
                "track_id": track_id,
                "point_uid": row.point_uid,
                "pressure_gpa": float(row.pressure_gpa),
                "local_peak_index": int(row.local_peak_index),
                "q": float(row.q),
                "two_theta_deg": float(row.two_theta_deg),
                "source_table": row.source_table,
                "source_track": int(row.track),
                "event": event,
                "track_n_pressure_points": meta["n_pressure_points"],
                "track_start_pressure_gpa": meta["start_pressure_gpa"],
                "track_end_pressure_gpa": meta["end_pressure_gpa"],
            }
        )

    edge_table = primary_edges.copy()
    edge_table.insert(0, "track_id", edge_table["point_uid_a"].map(node_to_track))
    track_summary = pd.DataFrame(track_meta.values()).sort_values(
        ["n_pressure_points", "start_pressure_gpa", "track_id"], ascending=[False, True, True]
    )
    return pd.DataFrame(track_rows), edge_table, track_summary


def build_birth_death_table(
    registry: pd.DataFrame,
    primary_edges: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pressures = np.sort(registry["pressure_gpa"].unique())
    by_uid = registry.set_index("point_uid")
    for pressure_a, pressure_b in zip(pressures[:-1], pressures[1:]):
        peaks_a = registry.loc[registry["pressure_gpa"].eq(pressure_a)]
        peaks_b = registry.loc[registry["pressure_gpa"].eq(pressure_b)]
        edge_mask = primary_edges["pressure_a_gpa"].eq(pressure_a) & primary_edges[
            "pressure_b_gpa"
        ].eq(pressure_b)
        edges = primary_edges.loc[edge_mask]
        matched_a = set(edges["point_uid_a"])
        matched_b = set(edges["point_uid_b"])
        deaths = [uid for uid in peaks_a["point_uid"] if uid not in matched_a]
        births = [uid for uid in peaks_b["point_uid"] if uid not in matched_b]

        def describe(uids: list[str], column: str) -> str:
            if not uids:
                return ""
            values = by_uid.loc[uids, column]
            if column == "local_peak_index":
                return ";".join(str(int(value)) for value in values)
            return ";".join(str(value) for value in values)

        rows.append(
            {
                "pressure_a_gpa": float(pressure_a),
                "pressure_b_gpa": float(pressure_b),
                "transition": f"{fmt_pressure(pressure_a)}->{fmt_pressure(pressure_b)}",
                "n_peaks_a": int(len(peaks_a)),
                "n_peaks_b": int(len(peaks_b)),
                "reliable_matches": int(len(edges)),
                "match_fraction_symmetric": float(2 * len(edges) / (len(peaks_a) + len(peaks_b))),
                "survival_fraction_from_a": float(len(edges) / len(peaks_a)),
                "inheritance_fraction_into_b": float(len(edges) / len(peaks_b)),
                "deaths": int(len(deaths)),
                "births": int(len(births)),
                "death_point_uids": ";".join(deaths),
                "birth_point_uids": ";".join(births),
                "death_local_peak_indices": describe(deaths, "local_peak_index"),
                "birth_local_peak_indices": describe(births, "local_peak_index"),
            }
        )
    return pd.DataFrame(rows)


def pressure_matrix_value(path: Path, pressure_a: float, pressure_b: float) -> float:
    matrix = pd.read_csv(path)
    row_pressures = pd.to_numeric(matrix.iloc[:, 0], errors="coerce").to_numpy(float)
    row_index = int(np.nanargmin(np.abs(row_pressures - pressure_b)))
    if abs(row_pressures[row_index] - pressure_b) > 1e-7:
        raise KeyError(f"Pressure {pressure_b} not found in {path}")
    column_candidates = []
    for column in matrix.columns[1:]:
        try:
            column_candidates.append((abs(float(column) - pressure_a), column))
        except ValueError:
            continue
    if not column_candidates or min(column_candidates)[0] > 1e-7:
        raise KeyError(f"Pressure column {pressure_a} not found in {path}")
    column = min(column_candidates)[1]
    return safe_float(matrix.loc[row_index, column])


def window_across_values(
    root: Path,
    mode: str,
    sample: str,
    role: str,
    method: str,
    pressure_a: float,
    pressure_b: float,
) -> list[float]:
    directory = (
        root
        / mode
        / sample
        / "window_to_window_across_frames"
        / role
        / method
        / "matrices"
    )
    paths = sorted(directory.glob("window_*.csv"))
    return [pressure_matrix_value(path, pressure_a, pressure_b) for path in paths]


def lower_triangle_values(path: Path, selected_indices: list[int] | None = None) -> np.ndarray:
    matrix = pd.read_csv(path).iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if selected_indices is not None:
        matrix = matrix[np.ix_(selected_indices, selected_indices)]
    values = matrix[np.tril_indices(matrix.shape[0], k=-1)]
    return values[np.isfinite(values)]


def powder_window_metrics(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pressures = np.sort(registry["pressure_gpa"].unique())
    transition_rows: list[dict[str, Any]] = []
    within_rows: list[dict[str, Any]] = []
    nonoverlap_indices = [0, 5, 10, 15, 20, 25]

    for pressure in pressures:
        row: dict[str, Any] = {"pressure_gpa": float(pressure)}
        for mode_short, mode in (("log", "log_squared"), ("exp", "exp_squared")):
            for role in ("spots", "fit_control"):
                path = (
                    C60_ROOT
                    / mode
                    / "powder"
                    / "window_to_window_within_same_frame"
                    / role
                    / "by_pressure"
                    / "matrices"
                    / f"{fmt_pressure(pressure)}GPa.csv"
                )
                values = lower_triangle_values(path, nonoverlap_indices)
                row.update(finite_summary(values, f"within_{role}_{mode_short}_nonoverlap"))
        within_rows.append(row)
    within = pd.DataFrame(within_rows)

    for pressure_a, pressure_b in zip(pressures[:-1], pressures[1:]):
        row = {
            "pressure_a_gpa": float(pressure_a),
            "pressure_b_gpa": float(pressure_b),
            "transition": f"{fmt_pressure(pressure_a)}->{fmt_pressure(pressure_b)}",
        }
        for mode_short, mode in (("log", "log_squared"), ("exp", "exp_squared")):
            for role in ("spots", "fit_control"):
                for method in ("acf_strict", "direct_strict", "shift_tolerant_secondary"):
                    values = window_across_values(
                        C60_ROOT, mode, "powder", role, method, pressure_a, pressure_b
                    )
                    row.update(finite_summary(values, f"across_{role}_{method}_{mode_short}"))
        transition_rows.append(row)
    transitions = pd.DataFrame(transition_rows)

    for role in ("spots", "fit_control"):
        transitions[f"across_{role}_acf_robust_min"] = transitions[
            [f"across_{role}_acf_strict_log_median", f"across_{role}_acf_strict_exp_median"]
        ].min(axis=1)
    transitions["across_spots_direct_robust_min"] = transitions[
        ["across_spots_direct_strict_log_median", "across_spots_direct_strict_exp_median"]
    ].min(axis=1)

    within_lookup = within.set_index("pressure_gpa")
    for mode_short in ("log", "exp"):
        for role in ("spots", "fit_control"):
            source = f"within_{role}_{mode_short}_nonoverlap_median"
            transitions[f"{source}_a"] = transitions["pressure_a_gpa"].map(within_lookup[source])
            transitions[f"{source}_b"] = transitions["pressure_b_gpa"].map(within_lookup[source])
            transitions[f"{source}_abs_change"] = (
                transitions[f"{source}_b"] - transitions[f"{source}_a"]
            ).abs()
    transitions["within_spots_robust_abs_change"] = transitions[
        [
            "within_spots_log_nonoverlap_median_abs_change",
            "within_spots_exp_nonoverlap_median_abs_change",
        ]
    ].max(axis=1)
    transitions["within_spots_consensus_abs_change"] = transitions[
        [
            "within_spots_log_nonoverlap_median_abs_change",
            "within_spots_exp_nonoverlap_median_abs_change",
        ]
    ].min(axis=1)
    return transitions, within


def build_pressure_metrics(
    registry: pd.DataFrame,
    registries: dict[str, pd.DataFrame],
    primary_edges: pd.DataFrame,
    birth_death: pd.DataFrame,
    window_transitions: pd.DataFrame,
) -> pd.DataFrame:
    metrics = birth_death.merge(window_transitions, on=["pressure_a_gpa", "pressure_b_gpa", "transition"])

    edge_summary_rows: list[dict[str, Any]] = []
    for (pressure_a, pressure_b), edges in primary_edges.groupby(
        ["pressure_a_gpa", "pressure_b_gpa"], sort=True
    ):
        edge_summary_rows.append(
            {
                "pressure_a_gpa": float(pressure_a),
                "pressure_b_gpa": float(pressure_b),
                "median_robust_mutual_min": float(edges["robust_mutual_min"].median()),
                "median_robust_mutual_geomean": float(edges["robust_mutual_geomean"].median()),
                "median_abs_delta_two_theta_deg": float(edges["abs_delta_two_theta_deg"].median()),
                "max_abs_delta_two_theta_deg": float(edges["abs_delta_two_theta_deg"].max()),
                "strict_assignment_consensus_fraction": float(
                    edges["strict_four_assignment_consensus_at_primary"].mean()
                ),
            }
        )
    metrics = metrics.merge(
        pd.DataFrame(edge_summary_rows),
        on=["pressure_a_gpa", "pressure_b_gpa"],
        how="left",
    )

    pressure_summary = registry.groupby("pressure_gpa").agg(
        peak_count=("point_uid", "size"),
        catalog_peak_height_sum=("intensity", "sum"),
        catalog_area_sum=("area", "sum"),
        median_distinct_frames_per_peak=("distinct_frames", "median"),
        total_observations=("n_observations", "sum"),
    )
    for definition, definition_registry in registries.items():
        transformed = definition_registry.groupby("pressure_gpa")[
            "spots_absolute_integral_main"
        ].sum()
        pressure_summary[f"{definition}_transformed_integral_sum"] = transformed

    for side, pressure_column in (("a", "pressure_a_gpa"), ("b", "pressure_b_gpa")):
        for column in pressure_summary.columns:
            metrics[f"{column}_{side}"] = metrics[pressure_column].map(pressure_summary[column])

    metrics["peak_count_abs_log_change"] = [
        symmetric_log_change(a, b) for a, b in zip(metrics["n_peaks_a"], metrics["n_peaks_b"])
    ]
    metrics["catalog_peak_height_abs_log_change"] = [
        symmetric_log_change(a, b)
        for a, b in zip(metrics["catalog_peak_height_sum_a"], metrics["catalog_peak_height_sum_b"])
    ]
    metrics["catalog_area_abs_log_change"] = [
        symmetric_log_change(a, b)
        for a, b in zip(metrics["catalog_area_sum_a"], metrics["catalog_area_sum_b"])
    ]
    transformed_change_columns: list[str] = []
    for definition in DEFINITIONS:
        column = f"{definition}_transformed_integral_abs_log_change"
        transformed_change_columns.append(column)
        metrics[column] = [
            symmetric_log_change(a, b)
            for a, b in zip(
                metrics[f"{definition}_transformed_integral_sum_a"],
                metrics[f"{definition}_transformed_integral_sum_b"],
            )
        ]
    metrics["transformed_integral_change_robust_max"] = metrics[transformed_change_columns].max(axis=1)
    metrics["transformed_integral_change_median"] = metrics[transformed_change_columns].median(axis=1)
    metrics["transformed_integral_change_consensus_min"] = metrics[transformed_change_columns].min(axis=1)
    metrics["support_observation_abs_log_change"] = [
        symmetric_log_change(a, b)
        for a, b in zip(metrics["total_observations_a"], metrics["total_observations_b"])
    ]

    metrics["continuity_anomaly"] = 1.0 - metrics["match_fraction_symmetric"]
    metrics["birth_death_fraction"] = (
        metrics["births"] + metrics["deaths"]
    ) / (metrics["n_peaks_a"] + metrics["n_peaks_b"])
    # Transition candidates should require change to persist across definitions.  The
    # geometric mean is therefore used for ROI discontinuity, while the minimum
    # mutual score remains the strict acceptance score for individual track edges.
    metrics["roi_discontinuity_anomaly"] = 1.0 - metrics[
        "median_robust_mutual_geomean"
    ].fillna(0.0)
    metrics["location_shift_anomaly"] = metrics["median_abs_delta_two_theta_deg"].fillna(
        LOCATION_GATE_DEG
    )
    # An anomaly is robust only when both Log and Exp show it.  For a similarity
    # drop this is 1-max(similarity); for an absolute change it is min(change).
    metrics["across_window_consensus_similarity"] = metrics[
        ["across_spots_acf_strict_log_median", "across_spots_acf_strict_exp_median"]
    ].max(axis=1)
    metrics["across_window_anomaly"] = 1.0 - metrics["across_window_consensus_similarity"]
    metrics["within_window_change_anomaly"] = metrics["within_spots_consensus_abs_change"]
    metrics["peak_count_change_anomaly"] = metrics["peak_count_abs_log_change"]
    metrics["total_intensity_change_anomaly"] = metrics[
        "transformed_integral_change_consensus_min"
    ]

    anomaly_columns = [
        "continuity_anomaly",
        "roi_discontinuity_anomaly",
        "location_shift_anomaly",
        "across_window_anomaly",
        "within_window_change_anomaly",
        "peak_count_change_anomaly",
        "total_intensity_change_anomaly",
    ]
    percentile_columns: list[str] = []
    for column in anomaly_columns:
        percentile = f"{column}_percentile"
        percentile_columns.append(percentile)
        metrics[percentile] = empirical_percentile(metrics[column])
    metrics["indicator_count_top_quartile"] = metrics[percentile_columns].ge(0.75).sum(axis=1)
    metrics["combined_anomaly_score"] = metrics[percentile_columns].mean(axis=1)

    def candidate_confidence(row: pd.Series) -> str:
        count = int(row["indicator_count_top_quartile"])
        score = float(row["combined_anomaly_score"])
        if count >= 4 and score >= 0.70:
            return "high"
        if count >= 3 and score >= 0.60:
            return "medium"
        if count >= 2 and score >= 0.50:
            return "watch"
        return "background"

    metrics["powder_candidate_confidence"] = metrics.apply(candidate_confidence, axis=1)
    metrics["candidate_phase_interval"] = metrics["powder_candidate_confidence"].ne("background").astype(int)
    metrics["data_support_confound_flag"] = (
        metrics["support_observation_abs_log_change"].ge(
            metrics["support_observation_abs_log_change"].quantile(0.75)
        )
    ).astype(int)
    metrics["interpretation_note"] = ""
    first_mask = metrics["pressure_a_gpa"].eq(metrics["pressure_a_gpa"].min())
    metrics.loc[first_mask, "interpretation_note"] = (
        "first-series boundary; ROI and windows are anomalous, but observation support/acquisition "
        "differs and structural interpretation requires independent evidence"
    )
    return metrics


def single_frame_registry() -> pd.DataFrame:
    path = (
        C60_ROOT
        / "_sources"
        / "log_squared"
        / "single_roi"
        / "single_crystal"
        / "per_peak_all_frames"
        / "frame_registry.csv"
    )
    frames = pd.read_csv(path)
    frames = frames.loc[~frames["branch"].astype(str).str.contains("decomp", case=False, na=False)].copy()
    frames = frames.sort_values("pressure_GPa").reset_index(drop=True)
    return frames


def build_single_transition_metrics() -> pd.DataFrame:
    frames = single_frame_registry()
    features: dict[str, pd.DataFrame] = {}
    observations: dict[str, pd.DataFrame] = {}
    for short, mode in (("log", "log_squared"), ("exp", "exp_squared")):
        source = (
            C60_ROOT
            / "_sources"
            / mode
            / "single_roi"
            / "single_crystal"
            / "per_peak_all_frames"
        )
        features[short] = pd.read_csv(source / "frame_track_features.csv")
        observations[short] = pd.read_csv(source / "track_observations.csv")

    frame_for_pressure = dict(zip(frames["pressure_GPa"], frames["frame"]))
    selected_features: dict[str, pd.DataFrame] = {}
    for short, table in features.items():
        mask = [
            int(row.frame) == int(frame_for_pressure.get(float(row.pressure_GPa), -999))
            for row in table.itertuples(index=False)
        ]
        selected_features[short] = table.loc[mask].copy()

    log_features = selected_features["log"]
    pressure_values = np.sort(frames["pressure_GPa"].unique())
    rows: list[dict[str, Any]] = []

    intensity_sum_by_pressure: dict[float, float] = {}
    obs = observations["log"].copy()
    obs = obs.loc[
        [
            int(row.frame) == int(frame_for_pressure.get(float(row.pressure_GPa), -999))
            for row in obs.itertuples(index=False)
        ]
    ]
    intensity_column = "untransformed_normalized_intensity_counts_per_s_per_pixel"
    collapsed_intensity = obs.groupby(["pressure_GPa", "track"])[intensity_column].median()
    for pressure, group in collapsed_intensity.groupby(level=0):
        intensity_sum_by_pressure[float(pressure)] = float(group.sum())

    frame_lookup = frames.set_index("pressure_GPa")
    for pressure_a, pressure_b in zip(pressure_values[:-1], pressure_values[1:]):
        data_a = log_features.loc[log_features["pressure_GPa"].eq(pressure_a)].set_index("track")
        data_b = log_features.loc[log_features["pressure_GPa"].eq(pressure_b)].set_index("track")
        tracks_a, tracks_b = set(data_a.index), set(data_b.index)
        shared = sorted(tracks_a & tracks_b)
        row: dict[str, Any] = {
            "single_pressure_a_gpa": float(pressure_a),
            "single_pressure_b_gpa": float(pressure_b),
            "single_transition": f"{fmt_pressure(pressure_a)}->{fmt_pressure(pressure_b)}",
            "single_midpoint_gpa": float((pressure_a + pressure_b) / 2.0),
            "single_n_spots_a": int(len(tracks_a)),
            "single_n_spots_b": int(len(tracks_b)),
            "single_shared_tracks": int(len(shared)),
            "single_births": int(len(tracks_b - tracks_a)),
            "single_deaths": int(len(tracks_a - tracks_b)),
            "single_shared_fraction": float(2 * len(shared) / (len(tracks_a) + len(tracks_b))),
            "single_median_abs_delta_two_theta_deg": (
                float(np.median(np.abs(data_b.loc[shared, "two_theta_median_deg"].to_numpy() - data_a.loc[shared, "two_theta_median_deg"].to_numpy())))
                if shared
                else float("nan")
            ),
            "single_total_untransformed_intensity_a": intensity_sum_by_pressure.get(
                float(pressure_a), float("nan")
            ),
            "single_total_untransformed_intensity_b": intensity_sum_by_pressure.get(
                float(pressure_b), float("nan")
            ),
        }
        row["single_total_intensity_abs_log_change"] = symmetric_log_change(
            row["single_total_untransformed_intensity_a"],
            row["single_total_untransformed_intensity_b"],
        )
        row["single_peak_count_abs_log_change"] = symmetric_log_change(len(tracks_a), len(tracks_b))
        for short in ("log", "exp"):
            table = selected_features[short].set_index(["pressure_GPa", "track"])
            similarities: list[float] = []
            for track in shared:
                value_a = float(
                    table.loc[(pressure_a, track), "normalized_area_median_counts_per_s_per_pixel"]
                )
                value_b = float(
                    table.loc[(pressure_b, track), "normalized_area_median_counts_per_s_per_pixel"]
                )
                similarities.append(min(value_a, value_b) / max(value_a, value_b, EPS))
            row[f"single_roi_scalar_{short}_median"] = (
                float(np.median(similarities)) if similarities else float("nan")
            )
            across = window_across_values(
                C60_ROOT,
                f"{short}_squared",
                "single_crystal",
                "spots",
                "acf_strict",
                pressure_a,
                pressure_b,
            )
            row.update(finite_summary(across, f"single_across_spots_acf_{short}"))

        row["single_roi_scalar_robust_min"] = min(
            row["single_roi_scalar_log_median"], row["single_roi_scalar_exp_median"]
        )
        row["single_across_spots_acf_robust_min"] = min(
            row["single_across_spots_acf_log_median"],
            row["single_across_spots_acf_exp_median"],
        )
        frame_a = frame_lookup.loc[pressure_a]
        frame_b = frame_lookup.loc[pressure_b]
        same_orientation = str(frame_a["orientation"]) == str(frame_b["orientation"])
        same_branch = str(frame_a["branch"]) == str(frame_b["branch"])
        row["single_orientation_a"] = str(frame_a["orientation"])
        row["single_orientation_b"] = str(frame_b["orientation"])
        row["single_branch_a"] = str(frame_a["branch"])
        row["single_branch_b"] = str(frame_b["branch"])
        row["single_orientation_or_branch_confound"] = int(not (same_orientation and same_branch))
        rows.append(row)

    single = pd.DataFrame(rows)
    single["single_continuity_anomaly"] = 1.0 - single["single_shared_fraction"]
    single["single_roi_anomaly"] = 1.0 - single["single_roi_scalar_robust_min"]
    single["single_location_shift_anomaly"] = single["single_median_abs_delta_two_theta_deg"]
    single["single_window_anomaly"] = 1.0 - single["single_across_spots_acf_robust_min"]
    single_anomalies = [
        "single_continuity_anomaly",
        "single_roi_anomaly",
        "single_location_shift_anomaly",
        "single_window_anomaly",
        "single_peak_count_abs_log_change",
        "single_total_intensity_abs_log_change",
    ]
    single_percentiles = []
    for column in single_anomalies:
        percentile = f"{column}_percentile"
        single_percentiles.append(percentile)
        single[percentile] = empirical_percentile(single[column])
    single["single_indicator_count_top_quartile"] = single[single_percentiles].ge(0.75).sum(axis=1)
    single["single_combined_anomaly_score"] = single[single_percentiles].mean(axis=1)
    single["single_change_flag"] = (
        (single["single_indicator_count_top_quartile"] >= 2)
        & (single["single_combined_anomaly_score"] >= 0.55)
    ).astype(int)
    return single


def build_cross_evidence(
    powder_metrics: pd.DataFrame,
    single_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    single_min = float(single_metrics["single_pressure_a_gpa"].min())
    single_max = float(single_metrics["single_pressure_b_gpa"].max())
    single_columns = list(single_metrics.columns)

    for powder in powder_metrics.itertuples(index=False):
        midpoint = float((powder.pressure_a_gpa + powder.pressure_b_gpa) / 2.0)
        row = {column: getattr(powder, column) for column in powder_metrics.columns}
        row["powder_midpoint_gpa"] = midpoint
        if midpoint < single_min or midpoint > single_max:
            for column in single_columns:
                row[column] = np.nan
            row["single_pressure_coverage"] = "not_measured_at_powder_midpoint"
            row["single_midpoint_distance_gpa"] = np.nan
            if powder.powder_candidate_confidence != "background":
                row["cross_sample_evidence"] = "powder_candidate_single_not_measured"
            else:
                row["cross_sample_evidence"] = "single_not_measured"
        else:
            nearest_index = (single_metrics["single_midpoint_gpa"] - midpoint).abs().idxmin()
            nearest = single_metrics.loc[nearest_index]
            for column in single_columns:
                row[column] = nearest[column]
            row["single_pressure_coverage"] = "nearest_sampled_single_interval"
            row["single_midpoint_distance_gpa"] = float(
                abs(float(nearest["single_midpoint_gpa"]) - midpoint)
            )
            if powder.powder_candidate_confidence == "background":
                row["cross_sample_evidence"] = "not_a_powder_candidate"
            elif int(nearest["single_change_flag"]) == 1:
                if int(nearest["single_orientation_or_branch_confound"]) == 1:
                    if float(nearest["single_window_anomaly_percentile"]) >= 0.75:
                        row["cross_sample_evidence"] = "supporting_window_but_peak_metrics_orientation_confounded"
                    else:
                        row["cross_sample_evidence"] = "single_change_is_orientation_or_branch_confounded"
                else:
                    row["cross_sample_evidence"] = "concordant_powder_and_single_evidence"
            else:
                row["cross_sample_evidence"] = "no_clear_single_support_at_nearest_sampled_interval"

        powder_confidence = str(powder.powder_candidate_confidence)
        evidence = str(row["cross_sample_evidence"])
        support_confound = int(powder.data_support_confound_flag) == 1
        if powder_confidence == "background":
            row["overall_interpretive_confidence"] = "background"
        elif "concordant_powder_and_single" in evidence and not support_confound:
            row["overall_interpretive_confidence"] = (
                "high" if powder_confidence == "high" else "medium"
            )
        elif "supporting_window" in evidence:
            row["overall_interpretive_confidence"] = "medium_but_single_peak_metrics_confounded"
        elif "single_not_measured" in evidence:
            row["overall_interpretive_confidence"] = "medium_powder_only" if powder_confidence == "high" else "watch_powder_only"
        elif support_confound:
            row["overall_interpretive_confidence"] = "provisional_data_support_confound"
        elif powder_confidence == "high":
            row["overall_interpretive_confidence"] = "medium_powder_only_no_single_confirmation"
        else:
            row["overall_interpretive_confidence"] = "watch"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_tracks(
    registry: pd.DataFrame,
    track_nodes: pd.DataFrame,
    track_edges: pd.DataFrame,
    output_path: Path,
) -> None:
    pressures = np.sort(registry["pressure_gpa"].unique())
    pressure_to_x = {pressure: index for index, pressure in enumerate(pressures)}
    fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
    ax.scatter(
        track_nodes["pressure_gpa"].map(pressure_to_x),
        track_nodes["two_theta_deg"],
        s=15,
        c="#8a8f98",
        alpha=0.65,
        linewidths=0,
        zorder=2,
        label="all 280 pressure-level peaks",
    )
    segments = []
    scores = []
    for edge in track_edges.itertuples(index=False):
        segments.append(
            [
                (pressure_to_x[edge.pressure_a_gpa], edge.two_theta_a_deg),
                (pressure_to_x[edge.pressure_b_gpa], edge.two_theta_b_deg),
            ]
        )
        scores.append(edge.robust_mutual_min)
    norm = Normalize(vmin=PRIMARY_THRESHOLD, vmax=1.0)
    collection = LineCollection(segments, cmap="viridis", norm=norm, linewidths=2.0, alpha=0.9)
    collection.set_array(np.asarray(scores))
    ax.add_collection(collection)
    births = track_nodes.loc[track_nodes["event"].isin(["birth", "singleton"])]
    ax.scatter(
        births["pressure_gpa"].map(pressure_to_x),
        births["two_theta_deg"],
        s=34,
        facecolors="none",
        edgecolors="#d1495b",
        linewidths=0.9,
        zorder=3,
        label="birth / unmatched start",
    )
    colorbar = fig.colorbar(collection, ax=ax, pad=0.01)
    colorbar.set_label("Conservative mutual ROI = min across c=0.60/0.75 and Log/Exp")
    ax.set_xticks(range(len(pressures)), [fmt_pressure(value) for value in pressures], rotation=55, ha="right")
    ax.set_xlabel("Pressure level (GPa; equally spaced categorical axis)")
    ax.set_ylabel(r"Peak position $2\theta$ (degrees)")
    ax.set_title(
        "Robust powder peak tracks\n"
        r"$|\Delta2\theta|<0.06^\circ$, mutual ROI, global one-to-one matching, primary threshold $S_{robust}\geq0.10$"
    )
    ax.grid(axis="x", color="#e3e6ea", linewidth=0.7)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_transition_summary(metrics: pd.DataFrame, output_path: Path) -> None:
    labels = metrics["transition"].tolist()
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(5, 1, figsize=(19, 16), sharex=True, constrained_layout=True)

    axes[0].plot(x, metrics["match_fraction_symmetric"], marker="o", color="#1f77b4", label="matched fraction")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Reliable match\nfraction")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].bar(x - 0.18, metrics["births"], width=0.36, color="#2a9d8f", label="births")
    axes[1].bar(x + 0.18, metrics["deaths"], width=0.36, color="#e76f51", label="deaths")
    axes[1].set_ylabel("Peak count")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    axes[2].plot(x, metrics["median_robust_mutual_min"], marker="o", color="#6a4c93", label="median mutual ROI")
    axes[2].plot(
        x,
        metrics["across_window_consensus_similarity"],
        marker="s",
        color="#1982c4",
        label="across-window ACF (Log/Exp consensus)",
    )
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Similarity")
    axes[2].legend(frameon=False, ncol=2, loc="lower right")

    axes[3].plot(
        x,
        metrics["median_abs_delta_two_theta_deg"],
        marker="o",
        color="#f4a261",
        label=r"median $|\Delta2\theta|$",
    )
    axes[3].set_ylabel(r"$|\Delta2\theta|$\n(deg)")
    intensity_axis = axes[3].twinx()
    intensity_axis.plot(
        x,
        metrics["transformed_integral_change_consensus_min"],
        marker="^",
        color="#264653",
        label="total intensity change (four-definition minimum)",
    )
    intensity_axis.set_ylabel("max |log intensity ratio|")
    lines = axes[3].get_lines() + intensity_axis.get_lines()
    axes[3].legend(lines, [line.get_label() for line in lines], frameon=False, ncol=2, loc="upper right")

    confidence_color = {"high": "#b2182b", "medium": "#ef8a62", "watch": "#fddbc7", "background": "#b8c4ce"}
    colors = [confidence_color[value] for value in metrics["powder_candidate_confidence"]]
    axes[4].bar(x, metrics["combined_anomaly_score"], color=colors, edgecolor="#4d4d4d", linewidth=0.5)
    axes[4].plot(x, metrics["indicator_count_top_quartile"] / 7.0, color="#111111", marker="o", label="top-quartile indicator count / 7")
    axes[4].set_ylim(0, 1.05)
    axes[4].set_ylabel("Exploratory\nanomaly score")
    axes[4].legend(frameon=False, loc="upper right")

    for axis in axes:
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        for idx, confidence in enumerate(metrics["powder_candidate_confidence"]):
            if confidence != "background":
                axis.axvspan(idx - 0.48, idx + 0.48, color=confidence_color[confidence], alpha=0.07, zorder=0)
    axes[-1].set_xticks(x, labels, rotation=55, ha="right")
    axes[-1].set_xlabel("Adjacent powder pressure interval (GPa)")
    fig.suptitle(
        "Powder pressure-transition summary — robust four-definition peak matching and independent window evidence\n"
        "Candidate bands require coincident anomalies; they are not phase assignments",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_robustness(
    matching_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    output_path: Path,
) -> None:
    totals = matching_summary.groupby("threshold").agg(
        robust_min_matches=("robust_min_matches", "sum"),
        strict_consensus=("strict_four_assignment_consensus_matches", "sum"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    x = np.arange(len(totals))
    axes[0].bar(x - 0.18, totals["robust_min_matches"], width=0.36, color="#2a9d8f", label="conservative-min global match")
    axes[0].bar(x + 0.18, totals["strict_consensus"], width=0.36, color="#6a4c93", label="exact 4-assignment intersection")
    axes[0].set_xticks(x, [f"{value:.2f}" for value in totals.index])
    axes[0].set_xlabel("Mutual ROI threshold")
    axes[0].set_ylabel("Matched adjacent-pressure edges")
    axes[0].legend(frameon=False)
    axes[0].set_title("Threshold sensitivity")

    positive = candidates.loc[candidates["robust_mutual_min"] > 0, "robust_mutual_min"]
    axes[1].hist(positive, bins=np.linspace(0, 1, 31), color="#457b9d", alpha=0.85)
    for threshold, style in zip(SENSITIVITY_THRESHOLDS, [":", "-", "--", "-."]):
        axes[1].axvline(threshold, color="#d1495b", linestyle=style, linewidth=1.5, label=f"{threshold:.2f}")
    axes[1].set_xlabel("Worst-case mutual ROI across four definitions")
    axes[1].set_ylabel("Location-gated candidate pairs")
    axes[1].set_title("Candidate score distribution")
    axes[1].legend(title="threshold", frameon=False, ncol=2)
    fig.suptitle("Robustness audit: c=0.60 vs c=0.75 and Log vs Exp")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_cross_evidence(
    powder_metrics: pd.DataFrame,
    single_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False, constrained_layout=True)
    powder_mid = (powder_metrics["pressure_a_gpa"] + powder_metrics["pressure_b_gpa"]) / 2
    axes[0].plot(powder_mid, powder_metrics["combined_anomaly_score"], marker="o", color="#264653")
    candidate = powder_metrics["powder_candidate_confidence"].ne("background")
    axes[0].scatter(
        powder_mid[candidate],
        powder_metrics.loc[candidate, "combined_anomaly_score"],
        s=90,
        facecolors="none",
        edgecolors="#b2182b",
        linewidths=1.5,
        label="powder candidate",
    )
    axes[0].set_ylabel("Powder anomaly score")
    axes[0].set_xlabel("Powder interval midpoint (GPa)")
    axes[0].legend(frameon=False)
    axes[0].grid(color="#e5e7eb")

    clean = single_metrics["single_orientation_or_branch_confound"].eq(0)
    axes[1].plot(
        single_metrics["single_midpoint_gpa"],
        single_metrics["single_combined_anomaly_score"],
        color="#2a9d8f",
        marker="o",
    )
    axes[1].scatter(
        single_metrics.loc[~clean, "single_midpoint_gpa"],
        single_metrics.loc[~clean, "single_combined_anomaly_score"],
        marker="x",
        s=80,
        color="#e76f51",
        label="orientation/branch-confounded peak metrics",
    )
    axes[1].set_ylabel("Single-crystal anomaly score")
    axes[1].set_xlabel("Single-crystal interval midpoint (GPa)")
    axes[1].legend(frameon=False)
    axes[1].grid(color="#e5e7eb")
    fig.suptitle("Powder–single-crystal cross-evidence (different pressure grids)")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def robustness_audit(
    candidates: pd.DataFrame,
    matching_summary: pd.DataFrame,
) -> dict[str, Any]:
    directional_new_counts: dict[str, int] = {}
    for short in ("log", "exp"):
        c60_root, c60_mode = DEFINITIONS[f"c60_{short}"]
        c75_root, c75_mode = DEFINITIONS[f"c75_{short}"]
        columns = ["anchor_point_uid", "target_point_uid", "spots_absolute_anchor_ROI_iou"]
        c60 = pd.read_csv(
            c60_root / "_sources" / c60_mode / "powder_roi" / "all_directed_cross_pressure_peak_pairs.csv.gz",
            usecols=columns,
        ).rename(columns={"spots_absolute_anchor_ROI_iou": "c60"})
        c75 = pd.read_csv(
            c75_root / "_sources" / c75_mode / "powder_roi" / "all_directed_cross_pressure_peak_pairs.csv.gz",
            usecols=columns,
        ).rename(columns={"spots_absolute_anchor_ROI_iou": "c75"})
        merged = c60.merge(c75, on=["anchor_point_uid", "target_point_uid"], validate="one_to_one")
        directional_new_counts[short] = int(((merged["c60"] == 0) & (merged["c75"] > 0)).sum())

    window_hash_equal = True
    window_files_checked = 0
    for mode in ("log_squared", "exp_squared"):
        for relative in (
            Path(mode) / "powder" / "window_to_window_across_frames" / "spots" / "acf_strict" / "matrices",
            Path(mode) / "powder" / "window_to_window_within_same_frame" / "spots" / "by_pressure" / "matrices",
        ):
            files_60 = sorted((C60_ROOT / relative).glob("*.csv"))
            files_75 = sorted((C75_ROOT / relative).glob("*.csv"))
            if [path.name for path in files_60] != [path.name for path in files_75]:
                window_hash_equal = False
                continue
            for path_60, path_75 in zip(files_60, files_75):
                window_files_checked += 1
                if sha256(path_60) != sha256(path_75):
                    window_hash_equal = False

    threshold_totals = {
        f"{threshold:.2f}": {
            "robust_min_matches": int(
                matching_summary.loc[matching_summary["threshold"].eq(threshold), "robust_min_matches"].sum()
            ),
            "strict_four_assignment_consensus_matches": int(
                matching_summary.loc[
                    matching_summary["threshold"].eq(threshold),
                    "strict_four_assignment_consensus_matches",
                ].sum()
            ),
        }
        for threshold in SENSITIVITY_THRESHOLDS
    }
    return {
        "directional_zero_to_positive_c75_vs_c60": directional_new_counts,
        "adjacent_location_gated_candidates": int(len(candidates)),
        "adjacent_c75_positive_c60_zero_candidates": int(candidates["c75_positive_c60_zero"].sum()),
        "window_c60_c75_byte_identical": window_hash_equal,
        "window_files_hash_checked": window_files_checked,
        "qwidth_window_replication_note": (
            "c=0.60 and c=0.75 window files are identical by design; q-width is not counted as an "
            "independent window robustness replicate. Log and Exp remain distinct."
        ),
        "matching_threshold_totals": threshold_totals,
    }


def validate_outputs(
    registry: pd.DataFrame,
    candidates: pd.DataFrame,
    primary_edges: pd.DataFrame,
    track_nodes: pd.DataFrame,
    track_summary: pd.DataFrame,
    birth_death: pd.DataFrame,
    powder_metrics: pd.DataFrame,
    cross_evidence: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["catalog_has_280_peaks"] = len(registry) == 280
    checks["nineteen_powder_pressures"] = registry["pressure_gpa"].nunique() == 19
    checks["candidate_location_gate_respected"] = bool(
        (candidates["abs_delta_two_theta_deg"] < LOCATION_GATE_DEG + 1e-9).all()
    )
    checks["primary_edges_meet_worst_case_threshold"] = bool(
        (primary_edges["robust_mutual_min"] + 1e-12 >= PRIMARY_THRESHOLD).all()
    )
    checks["primary_edges_exclude_c75_only_new_pairs"] = not bool(
        primary_edges["c75_positive_c60_zero"].any()
    )
    checks["one_outgoing_edge_per_node"] = not primary_edges["point_uid_a"].duplicated().any()
    checks["one_incoming_edge_per_node"] = not primary_edges["point_uid_b"].duplicated().any()
    checks["tracks_cover_all_nodes_once"] = (
        len(track_nodes) == len(registry)
        and track_nodes["point_uid"].nunique() == len(registry)
        and int(track_summary["n_pressure_points"].sum()) == len(registry)
    )
    checks["birth_death_equations_hold"] = bool(
        (
            (birth_death["deaths"] == birth_death["n_peaks_a"] - birth_death["reliable_matches"])
            & (birth_death["births"] == birth_death["n_peaks_b"] - birth_death["reliable_matches"])
        ).all()
    )
    checks["eighteen_powder_transitions"] = len(powder_metrics) == 18
    checks["cross_evidence_has_every_powder_transition"] = len(cross_evidence) == 18
    required_plots = [
        output_root / "plots" / "reliable_peak_tracks.png",
        output_root / "plots" / "pressure_transition_summary.png",
        output_root / "plots" / "robustness_definition_comparison.png",
        output_root / "plots" / "powder_single_cross_evidence.png",
    ]
    checks["all_required_plots_exist"] = all(path.exists() and path.stat().st_size > 0 for path in required_plots)
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "catalog_peaks": int(len(registry)),
            "adjacent_location_candidates": int(len(candidates)),
            "primary_reliable_edges": int(len(primary_edges)),
            "tracks_total": int(len(track_summary)),
            "tracks_with_two_or_more_pressures": int((track_summary["n_pressure_points"] >= 2).sum()),
            "candidate_intervals": int(powder_metrics["candidate_phase_interval"].sum()),
        },
    }


def write_readme(
    output_root: Path,
    audit: dict[str, Any],
    validation: dict[str, Any],
    powder_metrics: pd.DataFrame,
    primary_edges: pd.DataFrame,
) -> None:
    candidates = powder_metrics.loc[powder_metrics["candidate_phase_interval"].eq(1)]
    candidate_lines = []
    for row in candidates.itertuples(index=False):
        candidate_lines.append(
            f"- **{row.transition} GPa** — {row.powder_candidate_confidence}; "
            f"score={row.combined_anomaly_score:.3f}, top-quartile indicators="
            f"{int(row.indicator_count_top_quartile)}/7, reliable matches={int(row.reliable_matches)}."
        )
    if not candidate_lines:
        candidate_lines = ["- No interval passed the exploratory multi-indicator candidate rule."]

    confidence_counts = primary_edges["edge_confidence"].value_counts().to_dict()
    content = f"""# Robust peak tracks and exploratory pressure-transition analysis

## Scope

This suite implements the first three requested steps and adds a cautious powder–single-crystal cross-check:

1. robustness screening across `c=0.60`, `c=0.75`, Log, and Exp;
2. adjacent-pressure one-to-one powder peak tracking with births/deaths;
3. multi-indicator candidate transition screening;
4. pressure-grid-aligned single-crystal evidence where measured.

It does **not** assign a crystal structure. Indexing, lattice refinement, EOS fitting, and structural-model comparison remain the crystallographic next step.

## Primary matching definition

- Candidate gate: `|delta 2theta| < 0.06 degrees` (location similarity > 0).
- For every definition: `S_mutual = sqrt(S_A_to_B * S_B_to_A)`.
- Conservative four-definition score: `S_robust = min(S_mutual[c60 Log], S_mutual[c60 Exp], S_mutual[c75 Log], S_mutual[c75 Exp])`.
- Primary reliable edge: dummy-augmented global one-to-one Hungarian assignment at `S_robust >= {PRIMARY_THRESHOLD:.2f}`.
- Sensitivity assignments: thresholds `{', '.join(f'{value:.2f}' for value in SENSITIVITY_THRESHOLDS)}`.

The conservative-min solve is the primary track because intersecting four independent Hungarian solutions is unstable under near-ties. Exact four-assignment agreement is retained as an audit/confidence column.

## Key robustness facts

- Original catalog: **280 pressure-level peaks**, **19 pressures**, based on the same 519 formal observations used previously.
- Location-gated adjacent-pressure candidates: **{audit['adjacent_location_gated_candidates']}**.
- Primary reliable edges: **{validation['counts']['primary_reliable_edges']}**.
- Edge confidence counts: `{json.dumps(confidence_counts, sort_keys=True)}`.
- `c=0.75` creates **{audit['directional_zero_to_positive_c75_vs_c60']['log']} Log** and **{audit['directional_zero_to_positive_c75_vs_c60']['exp']} Exp** new positive *directed* cells relative to `c=0.60` (the previously noted 584). They are not automatically accepted. The adjacent, location-compatible subset contains only **{audit['adjacent_c75_positive_c60_zero_candidates']}** candidates, and none can pass the primary worst-case rule because the corresponding `c=0.60` score is zero.
- Location is mathematically unchanged across c and transforms; it is one hard gate, not four independent votes.
- Powder window files are byte-identical between c=0.60 and c=0.75 by design (`{audit['window_c60_c75_byte_identical']}`); only Log versus Exp counts as distinct window robustness evidence.
- The 3.5→3.75 GPa boundary has no primary edge at 0.10. It is anomalous in ROI and window metrics, but it is also the first-series/acquisition-support boundary. It must not be called a phase transition without independent evidence.

## Exploratory candidate intervals

The score uses seven non-duplicated indicator families: peak continuity, median mutual ROI, peak-position shift, across-frame window ACF, within-frame window-coherence change, peak-count change, and transformed total-intensity change. For anomaly detection, Log/Exp window changes and the four transformed-intensity changes use their **minimum anomaly**, so a candidate is not promoted by only one preprocessing definition. Birth/death counts are reported but not double-counted with match fraction. `high`, `medium`, and `watch` mean multi-indicator statistical candidates, not confirmed phase transitions.

{chr(10).join(candidate_lines)}

## Main deliverables

- `plots/pressure_transition_summary.png` — requested transition summary.
- `plots/reliable_peak_tracks.png` — one-to-one powder tracks; edge colour is worst-case mutual ROI.
- `powder/reliable_peak_tracks.csv` — one row per peak node, including births/deaths/singletons.
- `powder/reliable_track_edges.csv` — accepted adjacent edges and all four directional/mutual scores.
- `powder/peak_birth_death_by_transition.csv` — exact birth/death peak IDs and local indices.
- `powder/pressure_transition_metrics.csv` — all transition indicators and anomaly percentiles.
- `powder/candidate_phase_transitions.csv` — non-background candidates only.
- `cross_sample/powder_single_cross_evidence.csv` — nearest sampled single-crystal interval and quality flags.
- `cross_sample/candidate_intervals_with_cross_evidence.csv` — candidate-only table with an overall interpretive confidence that accounts for coverage/confounds.
- `single_crystal/single_transition_metrics.csv` — independent single-crystal spot/window metrics.
- `robustness/adjacent_candidate_scores.csv` — all location-gated candidates, including sensitivity flags.
- `robustness/matching_threshold_summary.csv` — counts by interval and threshold.
- `VALIDATION_REPORT.json` — machine-checkable invariants.

## Interpretation limits

Single-crystal `c=0.60` and `c=0.75` products are identical because q-width c changes only the powder ROI. Single-crystal ROI is a masked-ellipse, exposure/pixel-normalized scalar similarity, not the powder 1D integrated IoU. Orientation/branch-changing intervals are explicitly flagged; whole-pattern window evidence is less affected because it uses both scans. Powder intervals above 12.8 GPa have no single-crystal pressure coverage and are marked `not measured`, not `no change`.
"""
    (output_root / "README.md").write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root: Path = args.output_root
    for subdir in ("plots", "powder", "single_crystal", "cross_sample", "robustness"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    registry, registries, directed = load_powder_inputs()
    candidates = build_adjacent_candidate_table(registry, directed)
    candidates, primary_edges, matching_summary = run_all_matchings(registry, candidates)
    track_nodes, track_edges, track_summary = build_tracks(registry, primary_edges)
    birth_death = build_birth_death_table(registry, primary_edges)
    window_transitions, within_pressure = powder_window_metrics(registry)
    powder_metrics = build_pressure_metrics(
        registry, registries, primary_edges, birth_death, window_transitions
    )
    single_metrics = build_single_transition_metrics()
    cross_evidence = build_cross_evidence(powder_metrics, single_metrics)
    audit = robustness_audit(candidates, matching_summary)

    candidates.to_csv(output_root / "robustness" / "adjacent_candidate_scores.csv", index=False)
    matching_summary.to_csv(output_root / "robustness" / "matching_threshold_summary.csv", index=False)
    write_json(output_root / "robustness" / "ROBUSTNESS_AUDIT.json", audit)
    track_nodes.to_csv(output_root / "powder" / "reliable_peak_tracks.csv", index=False)
    track_edges.to_csv(output_root / "powder" / "reliable_track_edges.csv", index=False)
    track_summary.to_csv(output_root / "powder" / "track_summary.csv", index=False)
    birth_death.to_csv(output_root / "powder" / "peak_birth_death_by_transition.csv", index=False)
    window_transitions.to_csv(output_root / "powder" / "window_transition_metrics.csv", index=False)
    within_pressure.to_csv(output_root / "powder" / "within_window_pressure_metrics.csv", index=False)
    powder_metrics.to_csv(output_root / "powder" / "pressure_transition_metrics.csv", index=False)
    powder_metrics.loc[powder_metrics["candidate_phase_interval"].eq(1)].to_csv(
        output_root / "powder" / "candidate_phase_transitions.csv", index=False
    )
    single_metrics.to_csv(output_root / "single_crystal" / "single_transition_metrics.csv", index=False)
    cross_evidence.to_csv(output_root / "cross_sample" / "powder_single_cross_evidence.csv", index=False)
    cross_evidence.loc[cross_evidence["candidate_phase_interval"].eq(1)].to_csv(
        output_root / "cross_sample" / "candidate_intervals_with_cross_evidence.csv", index=False
    )

    plot_tracks(registry, track_nodes, track_edges, output_root / "plots" / "reliable_peak_tracks.png")
    plot_transition_summary(powder_metrics, output_root / "plots" / "pressure_transition_summary.png")
    plot_robustness(matching_summary, candidates, output_root / "plots" / "robustness_definition_comparison.png")
    plot_cross_evidence(powder_metrics, single_metrics, output_root / "plots" / "powder_single_cross_evidence.png")

    validation = validate_outputs(
        registry,
        candidates,
        primary_edges,
        track_nodes,
        track_summary,
        birth_death,
        powder_metrics,
        cross_evidence,
        output_root,
    )
    write_json(output_root / "VALIDATION_REPORT.json", validation)
    write_readme(output_root, audit, validation, powder_metrics, primary_edges)
    if validation["status"] != "PASS":
        raise RuntimeError(f"Validation failed: {validation}")
    print(json.dumps({"output_root": str(output_root), **validation["counts"]}, indent=2))


if __name__ == "__main__":
    main()
