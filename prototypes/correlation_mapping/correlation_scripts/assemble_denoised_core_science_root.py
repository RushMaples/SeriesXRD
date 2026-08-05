#!/usr/bin/env python3
"""Assemble one compact denoised UOTe science root with hardlinks.

Only the validator-visible core is assembled:

* powder and single_crystal;
* roi_area and location primary heatmaps/matrices;
* window-to-window across-frame primary heatmaps/matrices; and
* window-to-window within-frame by-pressure and aggregate heatmaps/matrices.

No supplementary galleries, trajectories, confidence maps, or full symmetric
audit payloads are copied.  Location comes from the previous formal package.
All science payloads are hardlinks, so a same-filesystem assembly consumes no
additional data blocks.  A source/hash/inode manifest and completion marker
provide provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from validate_package_denoised_correlation_suites import (
    CATEGORIES,
    FORMAL_EXPECTATIONS,
    SCIENCE_SAMPLES,
    ExpectedCounts,
    validate_location_against_baseline,
    validate_one_suite,
)


@dataclass(frozen=True)
class LinkSpec:
    source: Path
    destination_relative: Path
    sample: str
    category: str
    role: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"no existing parent for {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def immediate_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == suffix
    )


def find_peak_pair_directories(
    source: Path,
    *,
    single_crystal: bool,
) -> tuple[Path, Path]:
    candidates: list[tuple[Path, Path]] = [
        (source / "matrices", source / "heatmaps"),
        (source / "roi_area" / "matrices", source / "roi_area" / "heatmaps"),
    ]
    if single_crystal:
        candidates.extend(
            [
                (
                    source / "normalized_area_matrices",
                    source / "normalized_area_heatmaps",
                ),
                (
                    source / "per_peak_all_frames" / "normalized_area_matrices",
                    source / "per_peak_all_frames" / "normalized_area_heatmaps",
                ),
            ]
        )
    for matrices, heatmaps in candidates:
        if matrices.is_dir() and heatmaps.is_dir():
            return matrices, heatmaps
    rendered = "\n".join(f"  {matrices} | {heatmaps}" for matrices, heatmaps in candidates)
    raise FileNotFoundError(
        f"could not find matrix/heatmap directories below {source}; tried:\n{rendered}"
    )


def peak_specs(
    source: Path,
    *,
    sample: str,
    category: str,
    expected_count: int,
    single_crystal_source: bool = False,
) -> list[LinkSpec]:
    matrices_dir, heatmaps_dir = find_peak_pair_directories(
        source, single_crystal=single_crystal_source
    )
    matrices = immediate_files(matrices_dir, ".csv")
    heatmaps = immediate_files(heatmaps_dir, ".png")
    if len(matrices) != expected_count or len(heatmaps) != expected_count:
        raise ValueError(
            f"{sample}/{category}: source has {len(matrices)} CSV and "
            f"{len(heatmaps)} PNG; expected {expected_count} each"
        )
    matrix_stems = {path.stem for path in matrices}
    heatmap_stems = {path.stem for path in heatmaps}
    if matrix_stems != heatmap_stems:
        raise ValueError(
            f"{sample}/{category}: source CSV/PNG stems are not one-to-one"
        )
    specs: list[LinkSpec] = []
    for path in matrices:
        specs.append(
            LinkSpec(
                source=path.resolve(),
                destination_relative=Path(sample) / category / "matrices" / path.name,
                sample=sample,
                category=category,
                role="primary_matrix",
            )
        )
    for path in heatmaps:
        specs.append(
            LinkSpec(
                source=path.resolve(),
                destination_relative=Path(sample) / category / "heatmaps" / path.name,
                sample=sample,
                category=category,
                role="primary_heatmap",
            )
        )
    return specs


def baseline_location_specs(
    baseline_root: Path,
    *,
    sample: str,
    expected_count: int,
) -> list[LinkSpec]:
    source = baseline_root / sample / "location"
    return peak_specs(
        source,
        sample=sample,
        category="location",
        expected_count=expected_count,
    )


def is_across_core(path: Path) -> bool:
    return path.suffix in {".csv", ".png"} and (
        (path.suffix == ".csv" and "matrices" in path.parts)
        or (path.suffix == ".png" and "heatmaps" in path.parts)
    ) and "_audit_full_symmetric" not in path.parts and (
        "one_minus_similarity_diagnostics" not in path.parts
    )


def is_within_core(path: Path) -> bool:
    if path.suffix == ".csv":
        return path.name == "matrix.csv" or "by_pressure/matrices" in path.as_posix()
    if path.suffix == ".png":
        return path.name == "heatmap.png" or "by_pressure/heatmaps" in path.as_posix()
    return False


def raw_window_role_root(
    window_root: Path,
    *,
    sample: str,
    channel: str,
    kind: str,
) -> Path:
    candidates = (
        window_root / sample / "windows" / channel / kind,
        window_root / sample / channel / kind,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    rendered = "\n".join(f"  {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"could not find {sample}/{channel}/{kind} below {window_root}; tried:\n{rendered}"
    )


def window_specs_from_final_layout(
    window_root: Path,
    *,
    sample: str,
    category: str,
) -> list[LinkSpec] | None:
    source_category = window_root / sample / category
    if not source_category.is_dir():
        return None
    predicate = (
        is_across_core
        if category == "window_to_window_across_frames"
        else is_within_core
    )
    specs: list[LinkSpec] = []
    for path in sorted(source_category.rglob("*")):
        if not path.is_file() or path.is_symlink() or not predicate(path):
            continue
        relative = path.relative_to(source_category)
        specs.append(
            LinkSpec(
                source=path.resolve(),
                destination_relative=Path(sample) / category / relative,
                sample=sample,
                category=category,
                role="window_core",
            )
        )
    return specs


def window_specs_from_raw_layout(
    window_root: Path,
    *,
    sample: str,
    category: str,
) -> list[LinkSpec]:
    kind = (
        "across_frames"
        if category == "window_to_window_across_frames"
        else "within_frame"
    )
    channels = ("fit_control", "spots") if sample == "powder" else ("spots",)
    predicate = is_across_core if kind == "across_frames" else is_within_core
    specs: list[LinkSpec] = []
    for channel in channels:
        source_role = raw_window_role_root(
            window_root,
            sample=sample,
            channel=channel,
            kind=kind,
        )
        for path in sorted(source_role.rglob("*")):
            if not path.is_file() or path.is_symlink() or not predicate(path):
                continue
            relative = path.relative_to(source_role)
            specs.append(
                LinkSpec(
                    source=path.resolve(),
                    destination_relative=Path(sample) / category / channel / relative,
                    sample=sample,
                    category=category,
                    role="window_core",
                )
            )
    return specs


def window_specs(
    window_root: Path,
    *,
    sample: str,
    category: str,
    expected_count: int,
) -> list[LinkSpec]:
    specs = window_specs_from_final_layout(
        window_root, sample=sample, category=category
    )
    if specs is None:
        specs = window_specs_from_raw_layout(
            window_root, sample=sample, category=category
        )
    csv_count = sum(spec.source.suffix == ".csv" for spec in specs)
    png_count = sum(spec.source.suffix == ".png" for spec in specs)
    if csv_count != expected_count or png_count != expected_count:
        raise ValueError(
            f"{sample}/{category}: selected {csv_count} CSV and {png_count} PNG; "
            f"expected {expected_count} each"
        )
    return specs


def build_specs(
    *,
    powder_roi_source: Path,
    single_roi_source: Path,
    window_root: Path,
    baseline_root: Path,
    expected: ExpectedCounts,
) -> list[LinkSpec]:
    specs: list[LinkSpec] = []
    specs.extend(
        peak_specs(
            powder_roi_source,
            sample="powder",
            category="roi_area",
            expected_count=expected.peak_maps[("powder", "roi_area")],
        )
    )
    specs.extend(
        peak_specs(
            single_roi_source,
            sample="single_crystal",
            category="roi_area",
            expected_count=expected.peak_maps[("single_crystal", "roi_area")],
            single_crystal_source=True,
        )
    )
    for sample in SCIENCE_SAMPLES:
        specs.extend(
            baseline_location_specs(
                baseline_root,
                sample=sample,
                expected_count=expected.peak_maps[(sample, "location")],
            )
        )
        specs.extend(
            window_specs(
                window_root,
                sample=sample,
                category="window_to_window_across_frames",
                expected_count=expected.across[sample],
            )
        )
        specs.extend(
            window_specs(
                window_root,
                sample=sample,
                category="window_to_window_within_same_frame",
                expected_count=expected.within[sample],
            )
        )
    return sorted(specs, key=lambda spec: spec.destination_relative.as_posix())


def preflight(
    specs: Sequence[LinkSpec],
    *,
    output_root: Path,
    expected: ExpectedCounts,
) -> dict[str, object]:
    errors: list[str] = []
    if output_root.exists():
        errors.append(f"output root already exists: {output_root}")
    output_parent = nearest_existing_parent(output_root.parent)
    output_device = output_parent.stat().st_dev
    destinations: set[str] = set()
    source_roots: set[str] = set()
    total_bytes = 0
    for spec in specs:
        destination_key = spec.destination_relative.as_posix()
        if destination_key in destinations:
            errors.append(f"duplicate destination: {destination_key}")
        destinations.add(destination_key)
        if not spec.source.is_file() or spec.source.is_symlink():
            errors.append(f"source is not a regular non-symlink file: {spec.source}")
            continue
        source_stat = spec.source.stat()
        total_bytes += source_stat.st_size
        source_roots.add(str(spec.source.parent))
        if source_stat.st_dev != output_device:
            errors.append(
                f"cross-filesystem hardlink impossible: {spec.source} device "
                f"{source_stat.st_dev}, output device {output_device}"
            )
        if is_within(output_root, spec.source.parent):
            errors.append(f"output root is inside a source directory: {spec.source.parent}")
    expected_science_files = (
        2 * sum(expected.peak_maps.values())
        + 2 * sum(expected.across.values())
        + 2 * sum(expected.within.values())
    )
    formal_expected_science_files = (
        2 * sum(FORMAL_EXPECTATIONS.peak_maps.values())
        + 2 * sum(FORMAL_EXPECTATIONS.across.values())
        + 2 * sum(FORMAL_EXPECTATIONS.within.values())
    )
    return {
        "passed": not errors,
        "errors": errors,
        "science_file_count": len(specs),
        "expected_science_file_count": expected_science_files,
        "formal_expected_science_file_count": formal_expected_science_files,
        "logical_bytes": total_bytes,
        "hardlinked_payload_bytes_avoided": total_bytes,
        "output_device": output_device,
        "source_directory_count": len(source_roots),
    }


def write_readme(root: Path, *, transform_label: str, manifest_rows: int) -> None:
    text = f"""# UOTe denoised core correlation suite: {transform_label}

This compact science root contains exactly `powder` and `single_crystal`, each
with `roi_area`, `location`, `window_to_window_across_frames`, and
`window_to_window_within_same_frame`.

All {manifest_rows} science payload files are hardlinks to their immutable
source results. No supplementary galleries or full-symmetric audit payloads
are included. Location is hardlinked from the previous formal baseline because
the intensity-only preprocessing does not change fixed peak coordinates.

`SOURCE_MANIFEST.csv` records each source, SHA256, device, inode, link count,
and destination. `ASSEMBLY_VALIDATION.json` records the independent core
validation. `RUN_COMPLETE.json` is written only after successful validation.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def write_manifest(root: Path, rows: Sequence[dict[str, object]]) -> Path:
    path = root / "SOURCE_MANIFEST.csv"
    fieldnames = (
        "sample",
        "category",
        "role",
        "destination_relative",
        "source_absolute",
        "bytes",
        "sha256",
        "source_device",
        "source_inode",
        "destination_device",
        "destination_inode",
        "same_inode",
        "source_link_count_after",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def assemble(
    *,
    output_root: Path,
    transform_label: str,
    powder_roi_source: Path,
    single_roi_source: Path,
    window_root: Path,
    baseline_root: Path,
    expected: ExpectedCounts,
    dry_run: bool,
) -> dict[str, object]:
    sources = {
        "powder_roi_source": powder_roi_source.resolve(),
        "single_roi_source": single_roi_source.resolve(),
        "window_root": window_root.resolve(),
        "baseline_root": baseline_root.resolve(),
    }
    output_root = output_root.resolve()
    specs = build_specs(
        powder_roi_source=sources["powder_roi_source"],
        single_roi_source=sources["single_roi_source"],
        window_root=sources["window_root"],
        baseline_root=sources["baseline_root"],
        expected=expected,
    )
    flight = preflight(specs, output_root=output_root, expected=expected)
    result: dict[str, object] = {
        "status": "DRY_RUN_PASS" if flight["passed"] and dry_run else "PREFLIGHT_FAIL",
        "transform_label": transform_label,
        "output_root": str(output_root),
        "sources": {key: str(value) for key, value in sources.items()},
        "preflight": flight,
    }
    if not flight["passed"]:
        raise RuntimeError(json.dumps(result, indent=2, ensure_ascii=False))
    if dry_run:
        return result

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.assembling-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir()
    try:
        for sample in SCIENCE_SAMPLES:
            for category in CATEGORIES:
                (staging / sample / category).mkdir(parents=True)

        manifest_rows: list[dict[str, object]] = []
        for spec in specs:
            destination = staging / spec.destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(spec.source, destination, follow_symlinks=False)
            source_stat = spec.source.stat()
            destination_stat = destination.stat()
            same_inode = (
                source_stat.st_dev == destination_stat.st_dev
                and source_stat.st_ino == destination_stat.st_ino
            )
            if not same_inode:
                raise RuntimeError(f"hardlink inode verification failed: {destination}")
            manifest_rows.append(
                {
                    "sample": spec.sample,
                    "category": spec.category,
                    "role": spec.role,
                    "destination_relative": spec.destination_relative.as_posix(),
                    "source_absolute": str(spec.source),
                    "bytes": source_stat.st_size,
                    "sha256": sha256_file(spec.source),
                    "source_device": source_stat.st_dev,
                    "source_inode": source_stat.st_ino,
                    "destination_device": destination_stat.st_dev,
                    "destination_inode": destination_stat.st_ino,
                    "same_inode": same_inode,
                    "source_link_count_after": source_stat.st_nlink,
                }
            )

        suite_validation = validate_one_suite(
            staging,
            label=transform_label,
            expected=expected,
            tolerance=1e-12,
        )
        location_validation = validate_location_against_baseline(
            staging,
            sources["baseline_root"],
            label=f"{transform_label}_location_vs_baseline",
            abs_tolerance=1e-12,
            rel_tolerance=1e-10,
        )
        validation = {
            "status": (
                "PASS"
                if suite_validation["status"] == "PASS"
                and location_validation["all_hashes_and_numerics_equal"]
                else "FAIL"
            ),
            "validated_at_utc": utc_now(),
            "suite": suite_validation,
            "location_vs_baseline": location_validation,
            "all_science_files_same_inode_as_source": all(
                bool(row["same_inode"]) for row in manifest_rows
            ),
        }
        if validation["status"] != "PASS":
            raise RuntimeError(
                "assembled core failed independent validation:\n"
                + json.dumps(validation, indent=2, ensure_ascii=False)
            )

        write_readme(
            staging,
            transform_label=transform_label,
            manifest_rows=len(manifest_rows),
        )
        manifest_path = write_manifest(staging, manifest_rows)
        validation_path = staging / "ASSEMBLY_VALIDATION.json"
        validation_path.write_text(
            json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        completion = {
            "status": "complete",
            "assembled_at_utc": utc_now(),
            "transform_label": transform_label,
            "science_files": len(manifest_rows),
            "logical_science_bytes": sum(int(row["bytes"]) for row in manifest_rows),
            "new_science_payload_blocks": 0,
            "hardlink_method": "os.link; same device and inode verified",
            "source_manifest": manifest_path.name,
            "source_manifest_sha256": sha256_file(manifest_path),
            "assembly_validation": validation_path.name,
            "assembly_validation_sha256": sha256_file(validation_path),
            "all_science_files_same_inode_as_source": True,
        }
        (staging / "RUN_COMPLETE.json").write_text(
            json.dumps(completion, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    result.update(
        {
            "status": "PASS",
            "science_files": len(specs),
            "manifest": str(output_root / "SOURCE_MANIFEST.csv"),
            "validation": str(output_root / "ASSEMBLY_VALIDATION.json"),
            "completion": str(output_root / "RUN_COMPLETE.json"),
        }
    )
    return result


def write_csv(path: Path, rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic-png\n")


def write_peak_source(root: Path, *, single: bool, value: float) -> None:
    matrices = root / ("normalized_area_matrices" if single else "matrices")
    heatmaps = root / ("normalized_area_heatmaps" if single else "heatmaps")
    if single:
        rows = [["row", "f0"], ["f0", value]]
    else:
        rows = [["pressure", "peak_count", "peak 1"], [3.5, 1, value]]
    write_csv(matrices / "map_000.csv", rows)
    write_png(heatmaps / "map_000.png")


def write_location_baseline(root: Path) -> None:
    for sample in SCIENCE_SAMPLES:
        destination = root / sample / "location"
        if sample == "powder":
            rows = [["pressure", "peak_count", "peak 1"], [3.5, 1, 0.8]]
        else:
            rows = [["row", "f0"], ["f0", 0.8]]
        write_csv(destination / "matrices" / "map_000.csv", rows)
        write_png(destination / "heatmaps" / "map_000.png")


def write_lower_matrix(path: Path, value: float) -> None:
    write_csv(path, [["row", "a", "b"], ["a", "", ""], ["b", value, ""]])


def write_window_source(root: Path) -> None:
    for sample in SCIENCE_SAMPLES:
        channels = ("fit_control", "spots") if sample == "powder" else ("spots",)
        for channel in channels:
            across = (
                root
                / sample
                / "windows"
                / channel
                / "across_frames"
                / "acf_strict"
                / "matrices"
                / "window_00_0_5.csv"
            )
            write_lower_matrix(across, 0.3)
            write_png(
                Path(str(across).replace("/matrices/", "/heatmaps/")).with_suffix(".png")
            )
            within = (
                root
                / sample
                / "windows"
                / channel
                / "within_frame"
                / "aggregate"
                / "matrix.csv"
            )
            write_lower_matrix(within, 0.4)
            write_png(within.with_name("heatmap.png"))

            # Production contains this supplementary 1-r rendering below the
            # powder fit-control ACF family.  It is not a fourth correlation
            # family and must stay out of the compact science package.
            if sample == "powder" and channel == "fit_control":
                diagnostic = (
                    root
                    / sample
                    / "windows"
                    / channel
                    / "across_frames"
                    / "acf_strict"
                    / "one_minus_similarity_diagnostics"
                    / "matrices"
                    / "window_00_0_5.csv"
                )
                write_lower_matrix(diagnostic, 0.7)
                write_png(
                    Path(str(diagnostic).replace("/matrices/", "/heatmaps/")).with_suffix(
                        ".png"
                    )
                )


def run_self_test() -> None:
    expected = ExpectedCounts(
        peak_maps={
            ("powder", "roi_area"): 1,
            ("powder", "location"): 1,
            ("single_crystal", "roi_area"): 1,
            ("single_crystal", "location"): 1,
        },
        across={"powder": 2, "single_crystal": 1},
        within={"powder": 2, "single_crystal": 1},
    )
    with tempfile.TemporaryDirectory(prefix="denoised-core-assembler-") as temporary:
        root = Path(temporary)
        powder_roi = root / "powder_roi"
        single_roi = root / "single_roi" / "per_peak_all_frames"
        windows = root / "windows_source"
        baseline = root / "baseline"
        output = root / "assembled"
        write_peak_source(powder_roi, single=False, value=0.3)
        write_peak_source(single_roi, single=True, value=0.4)
        write_window_source(windows)
        write_location_baseline(baseline)
        result = assemble(
            output_root=output,
            transform_label="synthetic_log_square",
            powder_roi_source=powder_roi,
            single_roi_source=single_roi,
            window_root=windows,
            baseline_root=baseline,
            expected=expected,
            dry_run=False,
        )
        if result["status"] != "PASS":
            raise AssertionError(json.dumps(result, indent=2))
        manifest_path = output / "SOURCE_MANIFEST.csv"
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 20 or not all(row["same_inode"] == "True" for row in rows):
            raise AssertionError("manifest did not record all 20 verified hardlinks")
        for row in rows:
            source_stat = Path(row["source_absolute"]).stat()
            destination_stat = (output / row["destination_relative"]).stat()
            if (source_stat.st_dev, source_stat.st_ino) != (
                destination_stat.st_dev,
                destination_stat.st_ino,
            ):
                raise AssertionError("science payload is not a hardlink")

        # The forthcoming window source may itself already use the final
        # user-facing category layout; exercise that discovery path too.
        final_layout_dry_run = assemble(
            output_root=root / "assembled_from_final_window_layout",
            transform_label="synthetic_final_layout",
            powder_roi_source=powder_roi,
            single_roi_source=single_roi,
            window_root=output,
            baseline_root=baseline,
            expected=expected,
            dry_run=True,
        )
        if final_layout_dry_run["status"] != "DRY_RUN_PASS":
            raise AssertionError("final-layout window source was not recognized")

        try:
            assemble(
                output_root=output,
                transform_label="must_refuse_overwrite",
                powder_roi_source=powder_roi,
                single_roi_source=single_roi,
                window_root=windows,
                baseline_root=baseline,
                expected=expected,
                dry_run=False,
            )
        except RuntimeError as error:
            if "already exists" not in str(error):
                raise
        else:
            raise AssertionError("assembler did not refuse existing output")
    print(
        "SELF-TEST PASS: raw/final window layouts recognized; supplementary "
        "1-r diagnostics excluded; 20 core files hardlinked and validated; "
        "overwrite refused"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble one compact denoised UOTe science root using hardlinks."
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--transform-label")
    parser.add_argument("--powder-roi-source", type=Path)
    parser.add_argument("--single-roi-source", type=Path)
    parser.add_argument("--window-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    required = {
        "--output-root": args.output_root,
        "--transform-label": args.transform_label,
        "--powder-roi-source": args.powder_roi_source,
        "--single-roi-source": args.single_roi_source,
        "--window-root": args.window_root,
        "--baseline-root": args.baseline_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit("required unless --self-test: " + ", ".join(missing))
    result = assemble(
        output_root=args.output_root,
        transform_label=args.transform_label,
        powder_roi_source=args.powder_roi_source,
        single_roi_source=args.single_roi_source,
        window_root=args.window_root,
        baseline_root=args.baseline_root,
        expected=FORMAL_EXPECTATIONS,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
