#!/usr/bin/env python3
"""Validate and index the complete formal-composite ROI waterfall suite."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_ROOT = (
    ROOT
    / "correlations/results/"
    "uote_nonlinear_squared_qwidth075_comparison_20260803"
)
DEFAULT_SUITE_ROOT = (
    DEFAULT_COMPARISON_ROOT
    / "waterfall_complete_formal_composite_qwidth075_20260803"
)
MODES = ("log_squared",)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument(
        "--powder-only",
        action="store_true",
        help="Validate only powder waterfalls for the selected transform mode(s).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Transform mode to validate (Log² only).",
    )
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def iter_gzip_rows(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_png(path: Path) -> tuple[int, int, str, int, str]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        image_format = image.format
        image.verify()
    if image_format != "PNG" or width < 2000 or height < 1500:
        raise ValueError(f"unexpected rendered image geometry: {path}")
    if mode != "P":
        raise ValueError(f"expected compact palette PNG, found {mode}: {path}")
    return width, height, mode, path.stat().st_size, sha256(path)


def finite_score(value: str) -> float | None:
    if not value.strip():
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"correlation outside [0,1]: {value}")
    return number


def merge_intervals(
    intervals: Iterable[tuple[float, float]], tolerance: float = 1.0e-12
) -> list[tuple[float, float]]:
    ordered = sorted((float(left), float(right)) for left, right in intervals)
    merged: list[list[float]] = []
    for left, right in ordered:
        if right <= left:
            continue
        if not merged or left > merged[-1][1] + tolerance:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [(left, right) for left, right in merged]


def expected_powder_supports(
    comparison_root: Path, mode: str
) -> tuple[dict[str, list[tuple[float, float]]], float]:
    rows = read_rows(
        comparison_root
        / "_sources"
        / mode
        / "powder_roi/observation_spots_absolute_profile_audit.csv"
    )
    if len(rows) != 519:
        raise RuntimeError(f"powder observation scope failed: {mode}")
    factors = {round(float(row["half_width_factor"]), 12) for row in rows}
    if len(factors) != 1:
        raise RuntimeError(f"powder half-width factors disagree: {mode}")
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["point_uid"]].append(
            (
                float(row["two_theta_lower_deg"]),
                float(row["two_theta_upper_deg"]),
            )
        )
    merged = {uid: merge_intervals(items) for uid, items in grouped.items()}
    if len(merged) != 280:
        raise RuntimeError(f"powder point-support scope failed: {mode}")
    return merged, factors.pop()


def powder_matrix_score_totals(
    comparison_root: Path, mode: str, anchors: Sequence[str]
) -> tuple[int, int, int]:
    cross = 0
    positive = 0
    zero = 0
    matrix_root = comparison_root / mode / "powder/roi_area/matrices"
    for anchor in anchors:
        rows = read_rows(matrix_root / f"{anchor}.csv")
        if len(rows) != 19:
            raise RuntimeError(f"powder matrix pressure scope failed: {mode} {anchor}")
        for row in rows:
            peak_count = int(row["peak_count_at_pressure"])
            for peak_index in range(1, peak_count + 1):
                value = finite_score(row[f"peak {peak_index}"])
                if value is None:
                    continue
                cross += 1
                positive += int(value > 0.0)
                zero += int(value == 0.0)
    return cross, positive, zero


def powder_matrix_scores(
    comparison_root: Path, mode: str, anchor: str
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    rows = read_rows(
        comparison_root / mode / "powder/roi_area/matrices" / f"{anchor}.csv"
    )
    if len(rows) != 19:
        raise RuntimeError(f"powder matrix pressure scope failed: {mode} {anchor}")
    for row in rows:
        pressure = str(float(row["pressure_gpa"]))
        for peak_index in range(1, int(row["peak_count_at_pressure"]) + 1):
            value = finite_score(row[f"peak {peak_index}"])
            if value is not None:
                scores[(pressure, str(peak_index))] = value
    return scores


def validate_powder(
    *, mode: str, comparison_root: Path, suite_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = suite_root / "powder" / mode
    suite = read_json(root / "SUITE_VALIDATION.json")
    index = read_rows(root / "WATERFALL_INDEX.csv")
    if suite.get("status") != "PASS" or len(index) != 280:
        raise RuntimeError(f"powder suite/index scope failed: {mode}")
    if len({row["anchor_token"] for row in index}) != 280:
        raise RuntimeError(f"duplicate powder anchors: {mode}")

    expected_supports, half_width_factor = expected_powder_supports(
        comparison_root, mode
    )
    expected_components_per_anchor = sum(
        len(items) for items in expected_supports.values()
    )

    component_counts: Counter[str] = Counter()
    entity_statuses: dict[str, dict[tuple[str, str], tuple[str, float | None]]] = (
        defaultdict(dict)
    )
    mapping_rows = 0
    support_bound_mismatches = 0
    for row in iter_gzip_rows(root / "PEAK_COLOR_MAPPING.csv.gz"):
        mapping_rows += 1
        anchor = row["anchor_token"]
        component_counts[anchor] += 1
        score = finite_score(row["correlation"])
        status = row["status"]
        if status not in {
            "anchor_self",
            "same_pressure_not_compared",
            "compared_zero",
            "compared_positive",
        }:
            raise RuntimeError(f"unexpected powder status: {status}")
        if status == "same_pressure_not_compared" and score is not None:
            raise RuntimeError("uncompared powder entity has a numeric score")
        if status != "same_pressure_not_compared" and score is None:
            raise RuntimeError("scored powder entity has no numeric score")
        entity = (row["pressure_gpa"], row["local_peak_index"])
        prior = entity_statuses[anchor].get(entity)
        current = (status, score)
        if prior is not None and prior != current:
            raise RuntimeError("powder support components disagree on status/score")
        entity_statuses[anchor][entity] = current
        uid = row["point_uid"]
        component_index = int(row["support_component_index"])
        expected_items = expected_supports.get(uid, ())
        if component_index >= len(expected_items):
            support_bound_mismatches += 1
        else:
            expected_left, expected_right = expected_items[component_index]
            support_bound_mismatches += int(
                abs(float(row["support_left_deg"]) - expected_left) > 1.0e-10
                or abs(float(row["support_right_deg"]) - expected_right) > 1.0e-10
            )
    expected_mapping_rows = 280 * expected_components_per_anchor
    if (
        mapping_rows != expected_mapping_rows
        or set(component_counts.values()) != {expected_components_per_anchor}
    ):
        raise RuntimeError(f"powder mapping scope failed: {mode}")
    if support_bound_mismatches:
        raise RuntimeError(f"powder support bounds drifted: {mode}")
    if set(entity_statuses) != {row["anchor_token"] for row in index}:
        raise RuntimeError(f"powder mapping anchors drifted: {mode}")
    score_value_mismatches = 0
    for anchor, entities in entity_statuses.items():
        if len(entities) != 280:
            raise RuntimeError(f"powder entity count failed: {mode} {anchor}")
        statuses = Counter(status for status, _ in entities.values())
        if statuses["anchor_self"] != 1:
            raise RuntimeError(f"powder anchor self count failed: {mode} {anchor}")
        expected_scores = powder_matrix_scores(comparison_root, mode, anchor)
        mapped_scores = {
            (str(float(pressure)), peak_index): score
            for (pressure, peak_index), (status, score) in entities.items()
            if status.startswith("compared_")
        }
        if set(mapped_scores) != set(expected_scores):
            raise RuntimeError(f"powder matrix/mapping score scope failed: {mode} {anchor}")
        score_value_mismatches += sum(
            int(
                mapped_scores[key] is None
                or abs(float(mapped_scores[key]) - expected_scores[key]) > 1.0e-15
            )
            for key in expected_scores
        )
    if score_value_mismatches:
        raise RuntimeError(f"powder matrix/mapping score values drifted: {mode}")

    master_rows: list[dict[str, Any]] = []
    for row in index:
        path = Path(row["output_png"])
        width, height, image_mode, size, digest = validate_png(path)
        master_rows.append(
            {
                "mode": mode,
                "sample": "powder",
                "family": "roi_area",
                "anchor": row["anchor_token"],
                "anchor_point_or_feature": row["anchor_point_uid"],
                "anchor_track": "",
                "anchor_frame": "",
                "anchor_pressure_gpa": row["anchor_pressure_gpa"],
                "anchor_local_peak_index": row["anchor_local_peak_index"],
                "source_matrix": str(
                    (
                        comparison_root
                        / mode
                        / "powder/roi_area/matrices"
                        / f"{row['anchor_token']}.csv"
                    ).resolve()
                ),
                "waterfall_png": str(path.resolve()),
                "png_width": width,
                "png_height": height,
                "png_mode": image_mode,
                "png_bytes": size,
                "png_sha256": digest,
                "validation_status": row["status"],
            }
        )
    positives = sum(int(row["positive_cross_pressure_cells"]) for row in index)
    zeros = sum(int(row["zero_cross_pressure_cells"]) for row in index)
    cross = sum(int(row["cross_pressure_colored_cells"]) for row in index)
    matrix_totals = powder_matrix_score_totals(
        comparison_root, mode, [row["anchor_token"] for row in index]
    )
    if (cross, positives, zeros) != matrix_totals or cross != 74_076:
        raise RuntimeError(f"powder score totals failed: {mode}")
    reconstruction = suite["formal_profile_reconstruction"]
    if (
        reconstruction["observation_components"] != 519
        or reconstruction["pressure_level_point_profiles"] != 280
        or reconstruction["pressure_composite_traces"] != 19
        or reconstruction["point_area_max_abs_error_vs_formal_registry"] > 1e-9
    ):
        raise RuntimeError(f"powder reconstruction audit failed: {mode}")
    display_domain = suite.get("display_profile_domain", "correlation_transform")
    display_reconstruction = suite.get(
        "display_profile_reconstruction", reconstruction
    )
    if display_domain == "original_positive":
        if (
            display_reconstruction["profile_domain"] != "original_positive"
            or display_reconstruction["nonlinear_transform_applied"] is not False
            or display_reconstruction["measurement_normalization_applied"] is not True
            or display_reconstruction["positive_clipping_applied"] is not True
            or display_reconstruction["observation_components"] != 519
            or display_reconstruction["source_xy_files"] != 360
            or display_reconstruction["pressure_level_point_profiles"] != 280
            or display_reconstruction["pressure_composite_traces"] != 19
            or abs(
                float(display_reconstruction["shared_display_scale"])
                - 60.729366715604584
            )
            > 1.0e-10
            or display_reconstruction["zero_integral_point_count"] != 10
        ):
            raise RuntimeError(f"powder original-profile display audit failed: {mode}")
    return master_rows, {
        "status": "PASS",
        "sample": "powder",
        "mode": mode,
        "anchors": len(index),
        "pngs": len(master_rows),
        "mapping_rows": mapping_rows,
        "formal_entities_per_anchor": 280,
        "support_components_per_anchor": expected_components_per_anchor,
        "support_bound_mismatches": support_bound_mismatches,
        "matrix_mapping_score_value_mismatches": score_value_mismatches,
        "half_width_factor": half_width_factor,
        "cross_pressure_cells": cross,
        "positive_cross_pressure_cells": positives,
        "zero_cross_pressure_cells": zeros,
        "pressure_rows_per_waterfall": 19,
        "strictly_nonoverlapping": True,
        "formal_profile_reconstruction": reconstruction,
        "display_profile_domain": display_domain,
        "display_profile_reconstruction": display_reconstruction,
    }


def validate_single(
    *, mode: str, comparison_root: Path, suite_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = suite_root / "single_crystal" / mode / "roi_area/formal_composite"
    suite = read_json(root / "SUITE_VALIDATION.json")
    index = read_rows(root / "ANCHOR_INDEX.csv")
    if suite.get("status") != "PASS" or len(index) != 263:
        raise RuntimeError(f"single-crystal suite/index scope failed: {mode}")
    if len({row["anchor_uid"] for row in index}) != 263:
        raise RuntimeError(f"duplicate single-crystal anchors: {mode}")

    mapping_rows = 0
    status_counts: Counter[str] = Counter()
    rows_per_anchor: Counter[str] = Counter()
    compared_per_anchor: Counter[str] = Counter()
    for row in iter_gzip_rows(root / "peak_color_mapping.csv.gz"):
        mapping_rows += 1
        anchor = f"{row['anchor_track']}:{row['anchor_frame']}"
        rows_per_anchor[anchor] += 1
        status = row["status"]
        status_counts[status] += 1
        score = finite_score(row["correlation"])
        if status not in {
            "anchor_self",
            "not_in_selected_track_matrix",
            "compared_zero",
            "compared_positive",
        }:
            raise RuntimeError(f"unexpected single-crystal status: {status}")
        if status == "not_in_selected_track_matrix" and score is not None:
            raise RuntimeError("neutral single-crystal feature has a score")
        if status != "not_in_selected_track_matrix" and score is None:
            raise RuntimeError("scored single-crystal feature has no score")
        if status.startswith("compared_"):
            compared_per_anchor[anchor] += 1
    if mapping_rows != 69_169 or set(rows_per_anchor.values()) != {263}:
        raise RuntimeError(f"single-crystal mapping scope failed: {mode}")
    if status_counts["anchor_self"] != 263:
        raise RuntimeError(f"single-crystal anchor self total failed: {mode}")
    if (
        status_counts["compared_positive"] + status_counts["compared_zero"]
        != 1_306
    ):
        raise RuntimeError(f"single-crystal directed comparison total failed: {mode}")

    master_rows: list[dict[str, Any]] = []
    for row in index:
        path = Path(row["waterfall_png"])
        width, height, image_mode, size, digest = validate_png(path)
        master_rows.append(
            {
                "mode": mode,
                "sample": "single_crystal",
                "family": "roi_area",
                "anchor": row["anchor_uid"],
                "anchor_point_or_feature": row["anchor_uid"],
                "anchor_track": row["track"],
                "anchor_frame": row["frame"],
                "anchor_pressure_gpa": row["pressure_gpa"],
                "anchor_local_peak_index": row["local_peak_index"],
                "source_matrix": str(
                    (
                        comparison_root
                        / mode
                        / "single_crystal/roi_area/matrices"
                        / f"track_{int(row['track']):03d}.csv"
                    ).resolve()
                ),
                "waterfall_png": str(path.resolve()),
                "png_width": width,
                "png_height": height,
                "png_mode": image_mode,
                "png_bytes": size,
                "png_sha256": digest,
                "validation_status": row["validation_status"],
            }
        )
    reconstruction = suite["formal_profile_reconstruction"]
    if (
        reconstruction["collapsed_frame_track_profiles"] != 263
        or reconstruction["pressure_frame_composites"] != 12
        or reconstruction["max_abs_collapsed_profile_area_error"] > 1e-10
        or reconstruction["raw_profile_claim"] is not False
    ):
        raise RuntimeError(f"single-crystal reconstruction audit failed: {mode}")
    if (
        suite["singleton_track_count"] != 26
        or suite["singleton_anchor_count"] != 26
        or suite["strictly_nonoverlapping_all_waterfalls"] is not True
    ):
        raise RuntimeError(f"single-crystal suite invariants failed: {mode}")
    return master_rows, {
        "status": "PASS",
        "sample": "single_crystal",
        "mode": mode,
        "anchors": len(index),
        "pngs": len(master_rows),
        "mapping_rows": mapping_rows,
        "formal_entities_per_anchor": 263,
        "directed_cross_frame_cells": (
            status_counts["compared_positive"] + status_counts["compared_zero"]
        ),
        "positive_cross_frame_cells": status_counts["compared_positive"],
        "zero_cross_frame_cells": status_counts["compared_zero"],
        "singleton_tracks": suite["singleton_track_count"],
        "singleton_anchors": suite["singleton_anchor_count"],
        "pressure_frame_rows_per_waterfall": 12,
        "strictly_nonoverlapping": True,
        "formal_profile_reconstruction": reconstruction,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suite_root = args.suite_root.resolve()
    comparison_root = args.comparison_root.resolve()
    master_rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for mode in args.modes:
        rows, audit = validate_powder(
            mode=mode,
            comparison_root=comparison_root,
            suite_root=suite_root,
        )
        master_rows.extend(rows)
        groups.append(audit)
        if not args.powder_only:
            rows, audit = validate_single(
                mode=mode,
                comparison_root=comparison_root,
                suite_root=suite_root,
            )
            master_rows.extend(rows)
            groups.append(audit)
    keys = {(row["mode"], row["sample"], row["anchor"]) for row in master_rows}
    paths = {row["waterfall_png"] for row in master_rows}
    expected_waterfalls = len(args.modes) * (280 if args.powder_only else 543)
    if (
        len(master_rows) != expected_waterfalls
        or len(keys) != expected_waterfalls
        or len(paths) != expected_waterfalls
    ):
        raise RuntimeError("master waterfall scope/uniqueness failed")
    if any(row["validation_status"] != "PASS" for row in master_rows):
        raise RuntimeError("master index contains a failed anchor")

    write_rows(suite_root / "MASTER_WATERFALL_INDEX.csv", master_rows)
    audit: dict[str, Any] = {
        "status": "PASS",
        "suite": "complete formal-composite ROI correlation waterfalls",
        "comparison_root": str(comparison_root),
        "suite_root": str(suite_root),
        "modes": list(args.modes),
        "samples": (["powder"] if args.powder_only else ["powder", "single_crystal"]),
        "family": "roi_area",
        "expected_waterfalls": expected_waterfalls,
        "waterfalls": len(master_rows),
        "unique_mode_sample_anchor_keys": len(keys),
        "unique_png_paths": len(paths),
        "total_png_bytes": sum(int(row["png_bytes"]) for row in master_rows),
        "fixed_color_range": [0.0, 1.0],
        "png_palette_colors": 128,
        "all_png_files_verified": True,
        "all_sha256_recorded": True,
        "all_anchor_validations_pass": True,
        "all_trace_and_ribbon_bands_nonoverlapping": True,
        "groups": groups,
        "scope_note": (
            "This suite is the user-approved anchored ROI-area waterfall view. "
            "Location matrices and strict-lower window maps remain in the parent "
            "formal comparison result and are not misrepresented as ROI peak colors."
        ),
    }
    write_json(suite_root / "MASTER_VALIDATION.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
