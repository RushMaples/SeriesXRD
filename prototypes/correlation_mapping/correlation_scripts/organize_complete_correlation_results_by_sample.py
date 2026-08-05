#!/usr/bin/env python3
"""Build a non-destructive, hash-verified UOTe correlation delivery view.

The source result suites remain untouched.  The delivery view has exactly two
top-level science directories (powder and single_crystal), and each science
directory has exactly the four correlation categories requested by the user.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


REQUIRED_CATEGORIES = (
    "roi_area",
    "location",
    "window_to_window_across_frames",
    "window_to_window_within_same_frame",
)


@dataclass(frozen=True)
class CopyRecord:
    sample: str
    category: str
    role: str
    source_path: str
    destination_path: str
    bytes: int
    source_sha256: str
    copied_sha256: str
    identical: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PackageBuilder:
    def __init__(self, build_root: Path) -> None:
        self.build_root = build_root
        self.records: list[CopyRecord] = []

    def copy_file(
        self,
        source: Path,
        destination: Path,
        *,
        sample: str,
        category: str,
        role: str,
    ) -> None:
        if source.name == ".DS_Store":
            return
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_sha = sha256_file(source)
        destination_sha = sha256_file(destination)
        self.records.append(
            CopyRecord(
                sample=sample,
                category=category,
                role=role,
                source_path=str(source.resolve()),
                destination_path=str(destination.relative_to(self.build_root)),
                bytes=destination.stat().st_size,
                source_sha256=source_sha,
                copied_sha256=destination_sha,
                identical=source_sha == destination_sha,
            )
        )

    def copy_tree(
        self,
        source: Path,
        destination: Path,
        *,
        sample: str,
        category: str,
        role: str,
        include: Callable[[Path], bool] | None = None,
    ) -> None:
        if not source.is_dir():
            raise NotADirectoryError(source)
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.name == ".DS_Store":
                continue
            relative = path.relative_to(source)
            if include is not None and not include(relative):
                continue
            self.copy_file(
                path,
                destination / relative,
                sample=sample,
                category=category,
                role=role,
            )

    def write_index(self) -> None:
        output = self.build_root / "PACKAGE_INDEX.csv"
        fieldnames = list(asdict(self.records[0]).keys())
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in sorted(
                self.records,
                key=lambda item: (
                    item.sample,
                    item.category,
                    item.destination_path,
                ),
            ):
                writer.writerow(asdict(record))


def copy_named_files(
    builder: PackageBuilder,
    source_root: Path,
    destination_root: Path,
    names: Iterable[str],
    *,
    sample: str,
    category: str,
    role: str,
) -> None:
    for name in names:
        builder.copy_file(
            source_root / name,
            destination_root / name,
            sample=sample,
            category=category,
            role=role,
        )


def copy_window_provenance(
    builder: PackageBuilder,
    v8_root: Path,
    destination: Path,
    *,
    sample: str,
    category: str,
) -> None:
    provenance = destination / "_provenance"
    copy_named_files(
        builder,
        v8_root,
        provenance,
        (
            "WINDOW_METHODS.md",
            "LOWER_TRIANGLE_METHODS.md",
            "window_lower_triangle_index.csv",
            "window_similarity_diagnostics.csv",
            "run_manifest.json",
            "validation_report.json",
            "RUN_COMPLETE.json",
        ),
        sample=sample,
        category=category,
        role="window_provenance",
    )
    builder.copy_tree(
        v8_root / "window_provenance",
        provenance / "integer_window_run",
        sample=sample,
        category=category,
        role="window_provenance",
    )


def copy_window_quicklooks(
    builder: PackageBuilder,
    v8_root: Path,
    destination: Path,
    *,
    sample: str,
    category: str,
    mode_token: str,
) -> None:
    prefix = "powder_" if sample == "powder" else "single_"

    def wanted(relative: Path) -> bool:
        name = relative.name
        return (
            name.startswith(prefix)
            and mode_token in name
            or name in {"quicklook_index.csv", "unique_lower_triangle_pairs.csv"}
        )

    builder.copy_tree(
        v8_root / "window_quicklooks",
        destination / "_quicklooks",
        sample=sample,
        category=category,
        role="quicklooks",
        include=wanted,
    )


def copy_role_audit_common(
    builder: PackageBuilder,
    role_root: Path,
    destination: Path,
    *,
    sample: str,
    category: str,
    role: str,
) -> None:
    for name in (
        "integer_window_validation.json",
        "window_definition.csv",
        "window_feature_validity.npz",
    ):
        builder.copy_file(
            role_root / name,
            destination / "_shared" / name,
            sample=sample,
            category=category,
            role=f"{role}_full_symmetric_audit",
        )


def build_powder(
    builder: PackageBuilder,
    v8_root: Path,
    destination: Path,
) -> None:
    roi_destination = destination / "roi_area"
    location_destination = destination / "location"
    across_destination = destination / "window_to_window_across_frames"
    within_destination = destination / "window_to_window_within_same_frame"

    builder.copy_tree(
        v8_root / "peak_maps" / "roi_spots_absolute_anchor_integrated_iou",
        roi_destination,
        sample="powder",
        category="roi_area",
        role="absolute_q_directional_anchor_integrated_iou",
    )
    builder.copy_tree(
        v8_root / "peak_maps" / "location",
        location_destination,
        sample="powder",
        category="location",
        role="pressure_level_anchor_maps",
    )

    shared_peak_metadata = (
        "anchor_map_index.csv",
        "frame_measurement_normalization.csv",
        "observation_assignment.csv",
        "point_registry.csv",
        "pressure_peak_grid.csv",
        "pressure_row_layout.csv",
    )
    roi_metadata = (
        "all_directed_cross_pressure_peak_pairs.csv.gz",
        "pressure_level_directional_similarity_matrices.npz",
        "observation_spots_absolute_profile_audit.csv",
        "qwidth_factor_optimization.csv",
    )
    for category, category_destination in (
        ("roi_area", roi_destination),
        ("location", location_destination),
    ):
        copy_named_files(
            builder,
            v8_root,
            category_destination / "_metadata",
            shared_peak_metadata,
            sample="powder",
            category=category,
            role="shared_peak_registry",
        )
        copy_named_files(
            builder,
            v8_root,
            category_destination / "_provenance",
            (
                "README.md",
                "run_manifest.json",
                "validation_report.json",
                "RUN_COMPLETE.json",
            ),
            sample="powder",
            category=category,
            role="peak_map_provenance",
        )
    copy_named_files(
        builder,
        v8_root,
        roi_destination / "_metadata",
        roi_metadata,
        sample="powder",
        category="roi_area",
        role="roi_area_payload",
    )
    copy_named_files(
        builder,
        v8_root,
        location_destination / "_provenance",
        (
            "unchanged_location_window_sha256.csv",
            "unchanged_payload_reuse_manifest.json",
        ),
        sample="powder",
        category="location",
        role="location_reuse_audit",
    )

    for role in ("spots", "fit_control"):
        builder.copy_tree(
            v8_root / "powder" / "windows" / role / "across_frames",
            across_destination / role,
            sample="powder",
            category="window_to_window_across_frames",
            role=role,
        )
        builder.copy_tree(
            v8_root / "powder" / "windows" / role / "within_frame",
            within_destination / role,
            sample="powder",
            category="window_to_window_within_same_frame",
            role=role,
        )
        audit_role_root = (
            v8_root / "window_full_symmetric_audit" / "powder" / role
        )
        builder.copy_tree(
            audit_role_root / "across_frames",
            across_destination / "_audit_full_symmetric" / role,
            sample="powder",
            category="window_to_window_across_frames",
            role=f"{role}_full_symmetric_audit",
        )
        builder.copy_tree(
            audit_role_root / "within_frame",
            within_destination / "_audit_full_symmetric" / role,
            sample="powder",
            category="window_to_window_within_same_frame",
            role=f"{role}_full_symmetric_audit",
        )
        copy_role_audit_common(
            builder,
            audit_role_root,
            across_destination / "_audit_full_symmetric" / role,
            sample="powder",
            category="window_to_window_across_frames",
            role=role,
        )
        copy_role_audit_common(
            builder,
            audit_role_root,
            within_destination / "_audit_full_symmetric" / role,
            sample="powder",
            category="window_to_window_within_same_frame",
            role=role,
        )

    copy_window_provenance(
        builder,
        v8_root,
        across_destination,
        sample="powder",
        category="window_to_window_across_frames",
    )
    copy_window_provenance(
        builder,
        v8_root,
        within_destination,
        sample="powder",
        category="window_to_window_within_same_frame",
    )
    copy_window_quicklooks(
        builder,
        v8_root,
        across_destination,
        sample="powder",
        category="window_to_window_across_frames",
        mode_token="across_frames",
    )
    copy_window_quicklooks(
        builder,
        v8_root,
        within_destination,
        sample="powder",
        category="window_to_window_within_same_frame",
        mode_token="within_frame",
    )


def copy_single_crystal_primary_peak_results(
    builder: PackageBuilder,
    legacy_root: Path,
    destination: Path,
) -> None:
    source = legacy_root / "single_crystal" / "per_peak_all_frames"
    roi_destination = destination / "roi_area"
    location_destination = destination / "location"

    builder.copy_tree(
        source / "normalized_area_heatmaps",
        roi_destination / "heatmaps",
        sample="single_crystal",
        category="roi_area",
        role="masked_2d_global_tracks",
    )
    builder.copy_tree(
        source / "normalized_area_matrices",
        roi_destination / "matrices",
        sample="single_crystal",
        category="roi_area",
        role="masked_2d_global_tracks",
    )
    copy_named_files(
        builder,
        source,
        roi_destination,
        (
            "aggregate_normalized_area_heatmap.png",
            "aggregate_normalized_area_matrix.csv",
            "HEATMAP_INDEX.md",
        ),
        sample="single_crystal",
        category="roi_area",
        role="masked_2d_global_tracks",
    )
    builder.copy_tree(
        source / "paired_heatmaps",
        roi_destination / "_joint_area_location_review" / "paired_heatmaps",
        sample="single_crystal",
        category="roi_area",
        role="joint_area_location_review",
    )
    builder.copy_tree(
        source / "gallery",
        roi_destination / "_joint_area_location_review" / "gallery",
        sample="single_crystal",
        category="roi_area",
        role="joint_area_location_review",
    )

    builder.copy_tree(
        source / "location_heatmaps",
        location_destination / "heatmaps",
        sample="single_crystal",
        category="location",
        role="masked_2d_global_tracks",
    )
    builder.copy_tree(
        source / "location_matrices",
        location_destination / "matrices",
        sample="single_crystal",
        category="location",
        role="masked_2d_global_tracks",
    )
    copy_named_files(
        builder,
        source,
        location_destination,
        (
            "aggregate_location_heatmap.png",
            "aggregate_location_matrix.csv",
            "HEATMAP_INDEX.md",
        ),
        sample="single_crystal",
        category="location",
        role="masked_2d_global_tracks",
    )
    builder.copy_tree(
        source / "trajectories",
        location_destination / "_trajectories",
        sample="single_crystal",
        category="location",
        role="track_trajectories",
    )

    shared_names = (
        "all_pair_scores.csv",
        "frame_registry.csv",
        "frame_track_features.csv",
        "per_track_matrices.npz",
        "track_observations.csv",
        "track_summary.csv",
    )
    provenance_names = (
        "REPORT.md",
        "run_manifest.json",
        "validation/validation_report.json",
        "validation/strict_lower_triangle_validation.json",
        "validation/single_roi_extraction_qc.csv",
        "inputs/single_frame_registry.csv",
    )
    for category, category_destination in (
        ("roi_area", roi_destination),
        ("location", location_destination),
    ):
        copy_named_files(
            builder,
            source,
            category_destination / "_metadata",
            shared_names,
            sample="single_crystal",
            category=category,
            role="masked_peak_metadata",
        )
        copy_named_files(
            builder,
            legacy_root,
            category_destination / "_provenance",
            provenance_names,
            sample="single_crystal",
            category=category,
            role="masked_2d_validation",
        )


def copy_single_crystal_conservative_supplement(
    builder: PackageBuilder,
    uniform_root: Path,
    destination: Path,
) -> None:
    per_peak = uniform_root / "spots" / "per_peak"
    for category, source_name in (("roi_area", "area"), ("location", "location")):
        supplement = destination / category / "_supplementary_uniform_v2_1"
        builder.copy_tree(
            per_peak / source_name,
            supplement / "results",
            sample="single_crystal",
            category=category,
            role="uniform_v2_1_one_unambiguous_segment",
        )
        copy_named_files(
            builder,
            per_peak,
            supplement / "_tracking_metadata",
            (
                "canonical_tracks.csv",
                "peak_observations.csv",
                "peak_summary.csv",
                "pressure_consensus_nodes.csv",
                "quarantined_nodes.csv",
                "selection_audit.csv",
                "selection_audit_summary.csv",
            ),
            sample="single_crystal",
            category=category,
            role="uniform_v2_1_tracking_metadata",
        )
        copy_named_files(
            builder,
            uniform_root,
            supplement / "_provenance",
            (
                "REPORT.md",
                "algorithm_config.json",
                "run_manifest.json",
                "validation/validation_report.json",
            ),
            sample="single_crystal",
            category=category,
            role="uniform_v2_1_validation",
        )


def build_single_crystal(
    builder: PackageBuilder,
    v8_root: Path,
    legacy_root: Path,
    uniform_root: Path,
    destination: Path,
) -> None:
    copy_single_crystal_primary_peak_results(builder, legacy_root, destination)
    copy_single_crystal_conservative_supplement(builder, uniform_root, destination)

    across_destination = destination / "window_to_window_across_frames"
    within_destination = destination / "window_to_window_within_same_frame"
    builder.copy_tree(
        v8_root / "single_crystal" / "windows" / "spots" / "across_frames",
        across_destination / "spots",
        sample="single_crystal",
        category="window_to_window_across_frames",
        role="spots",
    )
    builder.copy_tree(
        v8_root / "single_crystal" / "windows" / "spots" / "within_frame",
        within_destination / "spots",
        sample="single_crystal",
        category="window_to_window_within_same_frame",
        role="spots",
    )
    audit_role_root = (
        v8_root / "window_full_symmetric_audit" / "single_crystal" / "spots"
    )
    builder.copy_tree(
        audit_role_root / "across_frames",
        across_destination / "_audit_full_symmetric" / "spots",
        sample="single_crystal",
        category="window_to_window_across_frames",
        role="spots_full_symmetric_audit",
    )
    builder.copy_tree(
        audit_role_root / "within_frame",
        within_destination / "_audit_full_symmetric" / "spots",
        sample="single_crystal",
        category="window_to_window_within_same_frame",
        role="spots_full_symmetric_audit",
    )
    copy_role_audit_common(
        builder,
        audit_role_root,
        across_destination / "_audit_full_symmetric" / "spots",
        sample="single_crystal",
        category="window_to_window_across_frames",
        role="spots",
    )
    copy_role_audit_common(
        builder,
        audit_role_root,
        within_destination / "_audit_full_symmetric" / "spots",
        sample="single_crystal",
        category="window_to_window_within_same_frame",
        role="spots",
    )
    copy_window_provenance(
        builder,
        v8_root,
        across_destination,
        sample="single_crystal",
        category="window_to_window_across_frames",
    )
    copy_window_provenance(
        builder,
        v8_root,
        within_destination,
        sample="single_crystal",
        category="window_to_window_within_same_frame",
    )
    copy_window_quicklooks(
        builder,
        v8_root,
        across_destination,
        sample="single_crystal",
        category="window_to_window_across_frames",
        mode_token="across_frames",
    )
    copy_window_quicklooks(
        builder,
        v8_root,
        within_destination,
        sample="single_crystal",
        category="window_to_window_within_same_frame",
        mode_token="within_frame",
    )


def write_readmes(build_root: Path) -> None:
    (build_root / "README.md").write_text(
        """# UOTe complete correlation results, organized by sample

This is a non-destructive delivery view.  The source result suites remain
unchanged.  The only two science directories here are `powder` and
`single_crystal`; each contains exactly the four requested correlation
categories.

## Powder

- `roi_area`: 280 anchor maps over 19 pressures (3.5--50.7 GPa).  The score is
  the directional continuous integrated IoU on the anchor's absolute q
  support.  Disjoint peak supports are numeric zero; white is reserved for a
  missing local peak slot or the intentionally omitted anchor-pressure row.
- `location`: the corresponding 280 pressure-level anchor maps.
- `window_to_window_across_frames`: exact nominal windows 0--5, 1--6, ...,
  27--32 degrees.  `acf_strict` is primary; `direct_strict` is validation and
  `shift_tolerant_secondary` is secondary.  `fit_control` is the
  tungsten/background internal control, not an additional UOTe sample.
- `window_to_window_within_same_frame`: aggregate, by-pressure, and per-frame
  window-pair results.

## Single crystal

- Primary `roi_area` and `location` results use the 275 curated observations
  in `Single Crystal (Cell 29)/Masked/kept_obs.csv`: 12 actual masked frames,
  75 global tracks, and 653 unique comparable pairs for each metric.
  ROI area is sideband-background-subtracted raw 2D ROI excess divided by TIFF
  exposure and effective unmasked ROI pixels, followed by min/max similarity.
  Location is `clip(1 - abs(delta_2theta)/0.06 deg, 0, 1)`.
- These single-crystal maps are 12-by-12 same-global-track frame matrices.
  They are not presented as method-equivalent to the powder exhaustive
  anchor-versus-all-local-peaks layout.  A direct all-22-frame conversion was
  rejected because only 12 frames have curated masks/observations and several
  other raw frames contain saturated/int32-extreme artifacts.
- `_supplementary_uniform_v2_1` contains the independently validated,
  deliberately conservative one-segment subset.  It is supplementary because
  it does not represent all single-crystal peaks.
- Window results use the 22-frame compression ladder (11 pressures, two
  orientations) and exact nominal windows 0--5 through 18--23 degrees.

All user-facing window matrices contain only the strict lower triangle:
diagonal and duplicate upper triangle are blank.  Their complete symmetric
sources are retained under `_audit_full_symmetric`.

`PACKAGE_INDEX.csv` records every copied artifact and verifies its SHA256
against its source.  `VALIDATION_REPORT.json` gives package-level counts and
checks.
""",
        encoding="utf-8",
    )

    (build_root / "powder" / "README.md").write_text(
        """# Powder correlation results

The four subdirectories are the four requested result types.  ROI-area and
location maps use pressure rows ordered high to low and local peak numbers
assigned independently by increasing 2theta at each pressure.  The window
folders retain both sample `spots` and the separate `fit_control` diagnostic.
The primary visible window maps are strict lower triangles only.
""",
        encoding="utf-8",
    )
    (build_root / "single_crystal" / "README.md").write_text(
        """# Single-crystal correlation results

The four subdirectories are the four requested result types.  Primary
ROI-area and location maps use all available curated masked observations:
275 observations in 12 actual masked frames and 75 global tracks.  Missing
curated observations remain unknown rather than being converted to zero.
These are same-global-track frame matrices, not powder-style exhaustive
anchor-versus-all-local-peaks maps; the distinction is intentional and
recorded in the package-level README.
Window maps use the separate 22-frame, 11-pressure-by-two-orientation
compression ladder and are displayed as strict lower triangles only.
""",
        encoding="utf-8",
    )


def validate_package(
    builder: PackageBuilder,
    build_root: Path,
    *,
    v8_root: Path,
    legacy_root: Path,
    uniform_root: Path,
) -> dict[str, object]:
    expected_samples = ["powder", "single_crystal"]
    actual_sample_dirs = sorted(
        path.name for path in build_root.iterdir() if path.is_dir()
    )
    sample_categories: dict[str, list[str]] = {}
    for sample in expected_samples:
        sample_categories[sample] = sorted(
            path.name
            for path in (build_root / sample).iterdir()
            if path.is_dir()
        )

    mismatches = [
        record.destination_path
        for record in builder.records
        if not record.identical
    ]
    missing_copies = [
        record.destination_path
        for record in builder.records
        if not (build_root / record.destination_path).is_file()
    ]
    category_counts: dict[str, int] = {}
    for sample in expected_samples:
        for category in REQUIRED_CATEGORIES:
            key = f"{sample}/{category}"
            category_counts[key] = sum(
                1
                for path in (build_root / sample / category).rglob("*")
                if path.is_file() and path.name != ".DS_Store"
            )

    def load_passed(path: Path) -> bool:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            payload.get("passed")
            or str(payload.get("status", "")).upper() == "PASS"
            or payload.get("complete_required_run")
        )

    checks = {
        "exact_two_science_directories": actual_sample_dirs == expected_samples,
        "powder_exact_four_categories": sample_categories["powder"]
        == sorted(REQUIRED_CATEGORIES),
        "single_crystal_exact_four_categories": sample_categories[
            "single_crystal"
        ]
        == sorted(REQUIRED_CATEGORIES),
        "all_categories_nonempty": all(value > 0 for value in category_counts.values()),
        "all_copied_files_exist": not missing_copies,
        "all_copy_hashes_match": not mismatches,
        "no_ds_store": not any(
            path.name == ".DS_Store" for path in build_root.rglob("*")
        ),
        "powder_v8_validation_passed": load_passed(
            v8_root / "validation_report.json"
        ),
        "single_primary_validation_passed": load_passed(
            legacy_root / "validation" / "validation_report.json"
        ),
        "single_primary_strict_triangle_validation_passed": load_passed(
            legacy_root / "validation" / "strict_lower_triangle_validation.json"
        ),
        "single_supplement_validation_passed": load_passed(
            uniform_root / "validation" / "validation_report.json"
        ),
        "powder_roi_280_heatmaps": len(
            list((build_root / "powder" / "roi_area" / "heatmaps").glob("*.png"))
        )
        == 280,
        "powder_roi_280_matrices": len(
            list((build_root / "powder" / "roi_area" / "matrices").glob("*.csv"))
        )
        == 280,
        "powder_location_280_heatmaps": len(
            list((build_root / "powder" / "location" / "heatmaps").glob("*.png"))
        )
        == 280,
        "powder_location_280_matrices": len(
            list((build_root / "powder" / "location" / "matrices").glob("*.csv"))
        )
        == 280,
        "single_primary_roi_75_heatmaps": len(
            list(
                (
                    build_root / "single_crystal" / "roi_area" / "heatmaps"
                ).glob("*.png")
            )
        )
        == 75,
        "single_primary_roi_75_matrices": len(
            list(
                (
                    build_root / "single_crystal" / "roi_area" / "matrices"
                ).glob("*.csv")
            )
        )
        == 75,
        "single_primary_location_75_heatmaps": len(
            list(
                (
                    build_root / "single_crystal" / "location" / "heatmaps"
                ).glob("*.png")
            )
        )
        == 75,
        "single_primary_location_75_matrices": len(
            list(
                (
                    build_root / "single_crystal" / "location" / "matrices"
                ).glob("*.csv")
            )
        )
        == 75,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_roots": {
            "powder_and_integer_windows_v8": str(v8_root.resolve()),
            "single_crystal_primary_masked_2d": str(legacy_root.resolve()),
            "single_crystal_conservative_supplement": str(uniform_root.resolve()),
        },
        "checks": checks,
        "actual_science_directories": actual_sample_dirs,
        "sample_categories": sample_categories,
        "copied_artifact_count": len(builder.records),
        "copied_bytes": sum(record.bytes for record in builder.records),
        "category_file_counts_before_package_metadata": category_counts,
        "hash_mismatch_count": len(mismatches),
        "hash_mismatch_examples": mismatches[:20],
        "missing_copy_count": len(missing_copies),
        "missing_copy_examples": missing_copies[:20],
    }
    return report


def parse_args() -> argparse.Namespace:
    default_v8 = Path(
        "/Users/stanley/x-ray/correlations/results/"
        "uote_pressure_level_peak_spots_absolute_anchor_iou_"
        "integer_window_suite_20260730_v8"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-root", type=Path, default=default_v8)
    parser.add_argument(
        "--single-primary-root",
        type=Path,
        default=Path(
            "/Users/stanley/x-ray/correlations/results/"
            "uote_refinement_legacy_global_per_peak_"
            "strict_lower_triangle_20260716"
        ),
    )
    parser.add_argument(
        "--single-supplement-root",
        type=Path,
        default=Path(
            "/Users/stanley/x-ray/correlations/results/"
            "uote_single_crystal_correlations_uniform_v2_1_20260714"
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help=(
            "Default: <v8-root>/peak_maps/complete_correlation_results_by_sample"
        ),
    )
    parser.add_argument(
        "--finalize-existing-build",
        action="store_true",
        help=(
            "Re-run package validation and atomically finalize an already "
            "copied <destination>__building__ directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v8_root = args.v8_root.resolve()
    legacy_root = args.single_primary_root.resolve()
    uniform_root = args.single_supplement_root.resolve()
    destination = (
        args.destination.resolve()
        if args.destination is not None
        else v8_root
        / "peak_maps"
        / "complete_correlation_results_by_sample"
    )
    build_root = destination.with_name(destination.name + "__building__")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists; refusing to overwrite: {destination}"
        )
    if args.finalize_existing_build:
        if not build_root.is_dir():
            raise FileNotFoundError(
                f"No existing build directory to finalize: {build_root}"
            )
        builder = PackageBuilder(build_root)
        with (build_root / "PACKAGE_INDEX.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                builder.records.append(
                    CopyRecord(
                        sample=row["sample"],
                        category=row["category"],
                        role=row["role"],
                        source_path=row["source_path"],
                        destination_path=row["destination_path"],
                        bytes=int(row["bytes"]),
                        source_sha256=row["source_sha256"],
                        copied_sha256=row["copied_sha256"],
                        identical=row["identical"].strip().lower() == "true",
                    )
                )
        report = validate_package(
            builder,
            build_root,
            v8_root=v8_root,
            legacy_root=legacy_root,
            uniform_root=uniform_root,
        )
        (build_root / "VALIDATION_REPORT.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if report["status"] != "PASS":
            raise RuntimeError(
                "Package validation failed; inspect "
                f"{build_root / 'VALIDATION_REPORT.json'}"
            )
        (build_root / "RUN_COMPLETE.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "destination": str(destination),
                    "copied_artifact_count": len(builder.records),
                    "package_index": "PACKAGE_INDEX.csv",
                    "validation_report": "VALIDATION_REPORT.json",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        build_root.rename(destination)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nCreated: {destination}")
        return
    if build_root.exists():
        raise FileExistsError(
            f"Build directory already exists; inspect it before retrying: {build_root}"
        )

    for sample in ("powder", "single_crystal"):
        for category in REQUIRED_CATEGORIES:
            (build_root / sample / category).mkdir(parents=True, exist_ok=False)

    builder = PackageBuilder(build_root)
    build_powder(builder, v8_root, build_root / "powder")
    build_single_crystal(
        builder,
        v8_root,
        legacy_root,
        uniform_root,
        build_root / "single_crystal",
    )
    write_readmes(build_root)
    builder.write_index()
    report = validate_package(
        builder,
        build_root,
        v8_root=v8_root,
        legacy_root=legacy_root,
        uniform_root=uniform_root,
    )
    (build_root / "VALIDATION_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if report["status"] != "PASS":
        raise RuntimeError(
            "Package validation failed; inspect "
            f"{build_root / 'VALIDATION_REPORT.json'}"
        )
    (build_root / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "destination": str(destination),
                "copied_artifact_count": len(builder.records),
                "package_index": "PACKAGE_INDEX.csv",
                "validation_report": "VALIDATION_REPORT.json",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    build_root.rename(destination)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nCreated: {destination}")


if __name__ == "__main__":
    main()
