#!/usr/bin/env python3
"""Validate a completed ``uniform-correlation-v2.1`` result directory.

The validator deliberately knows only the public on-disk contract.  It does
not import the analysis runner, so a broken runner cannot accidentally make
its own output pass validation.  A deterministic ``RUN_COMPLETE.json`` marker
is written only after every required check succeeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from xml.etree import ElementTree

import numpy as np


PROFILE_ID = "uniform-correlation-v2.1"
DEFAULT_TOLERANCE = 1.0e-10
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FORMULA_ERROR_TOKENS = (
    "#REF!",
    "#DIV/0!",
    "#VALUE!",
    "#NAME?",
    "#N/A",
    "#NUM!",
    "#NULL!",
)
REQUIRED_SYNTHETIC_CASES = {
    "v2_upstream_config_equivalence",
    "continuous_peak",
    "local_ambiguity_split",
    "crossing_no_identity_switch",
    "missing_gap_bridge_and_cut",
    "no_cross_cut_interpolation",
    "unique_node_and_detection_assignment",
    "tracking_area_independence",
    "global_intensity_scale_invariance",
    "no_cross_scan_pairing",
    "missing_is_nan",
    "deterministic_permutation_pressure_reversal",
}


@dataclass
class ValidationState:
    """Accumulate named checks without aborting at the first failure."""

    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: Any = None) -> None:
        unique = name
        suffix = 2
        while unique in self.checks:
            unique = f"{name}_{suffix}"
            suffix += 1
        self.checks[unique] = bool(passed)
        if detail is not None:
            self.details[unique] = json_safe(detail)
        if not passed:
            if isinstance(detail, str):
                self.errors.append(f"{unique}: {detail}")
            else:
                self.errors.append(unique)


@dataclass(frozen=True)
class MatrixItem:
    source: Path
    key: str
    role: str
    values: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.source}:{self.key}" if self.key else str(self.source)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def declared_channels(run_dir: Path) -> tuple[str, ...]:
    """Read the channel contract; single-channel datasets are valid in v2.1."""

    try:
        manifest = load_json(run_dir / "run_manifest.json")
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ()
    values = manifest.get("channels", []) if isinstance(manifest, Mapping) else []
    channels = tuple(dict.fromkeys(str(item).strip().lower() for item in values if str(item).strip()))
    return channels


def recursive_values_for_keys(value: Any, aliases: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if normalized_name(str(key)) in aliases:
                found.append(item)
            found.extend(recursive_values_for_keys(item, aliases))
    elif isinstance(value, list):
        for item in value:
            found.extend(recursive_values_for_keys(item, aliases))
    return found


def contains_nonfinite_json(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(contains_nonfinite_json(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_nonfinite_json(item) for item in value)
    return False


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "present", "reliable", "detected", "valid"}:
        return True
    if text in {
        "0",
        "false",
        "no",
        "n",
        "absent",
        "unknown",
        "invalid",
        "missing",
        "unreliable",
        "out_of_range",
        "no_candidate",
        "invalid_pattern",
    }:
        return False
    return None


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def read_csv_matrix(path: Path) -> tuple[list[str], np.ndarray] | None:
    """Read a labelled square matrix, returning None for ordinary long tables."""

    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error):
        return None
    if len(rows) < 2 or len(rows[0]) < 2:
        return None
    labels = [cell.strip() for cell in rows[0][1:]]
    n = len(labels)
    if len(rows) != n + 1 or any(len(row) < n + 1 for row in rows[1:]):
        return None
    row_labels = [row[0].strip() for row in rows[1:]]
    if row_labels != labels:
        return None
    matrix = np.full((n, n), np.nan, dtype=float)
    for row_index, row in enumerate(rows[1:]):
        for col_index, cell in enumerate(row[1 : n + 1]):
            text = cell.strip()
            if not text or text.lower() in {"nan", "na", "n/a", "null", "none"}:
                continue
            try:
                matrix[row_index, col_index] = float(text)
            except ValueError:
                return None
    return labels, matrix


def role_for(path: Path, key: str = "") -> str:
    text = f"{path.as_posix().lower()} {normalized_name(key)}"
    key_name = normalized_name(key)
    if any(
        token in text
        for token in (
            "support",
            "availability",
            "available",
            "n_available",
            "n_both_present",
            "/n10",
            "/n01",
            "n_unknown",
        )
    ) or key_name == "available" or key_name.startswith("available_") or key_name.endswith("_available"):
        return "support"
    if key_name in {"n10", "n01", "n11", "n00"}:
        return "support"
    if "presence" in text or "jaccard" in text:
        return "presence"
    if (
        "/area/" in text
        or key_name == "area"
        or key_name.startswith("area_")
        or key_name.endswith("_area")
    ):
        return "area"
    if (
        "/location/" in text
        or key_name == "location"
        or key_name.startswith("location_")
        or key_name.endswith("_location")
    ):
        return "location"
    if any(token in text for token in ("across_frames", "within_frame", "acf_strict", "direct_strict", "shift_tolerant")):
        return "signed"
    return "unknown"


def iter_matrix_items(run_dir: Path) -> Iterator[MatrixItem]:
    for path in sorted(run_dir.rglob("*.csv")):
        parsed = read_csv_matrix(path)
        if parsed is None:
            continue
        _labels, matrix = parsed
        role = role_for(path)
        if role != "unknown":
            yield MatrixItem(path, "", role, matrix)

    for path in sorted(run_dir.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                for key in sorted(archive.files):
                    try:
                        array = np.asarray(archive[key])
                    except (TypeError, ValueError):
                        continue
                    role = role_for(path, key)
                    if role == "unknown" or array.ndim < 2 or array.shape[-1] != array.shape[-2]:
                        continue
                    if array.dtype.kind not in "biufc":
                        continue
                    yield MatrixItem(path, key, role, np.asarray(array, dtype=float))
        except (OSError, ValueError, zipfile.BadZipFile):
            continue


def finite_range_ok(array: np.ndarray, lower: float, upper: float, tolerance: float) -> bool:
    finite = np.asarray(array, dtype=float)[np.isfinite(array)]
    if finite.size == 0:
        return True
    return bool(np.min(finite) >= lower - tolerance and np.max(finite) <= upper + tolerance)


def symmetric_last_axes(array: np.ndarray, tolerance: float) -> tuple[bool, float]:
    transposed = np.swapaxes(array, -1, -2)
    finite_mismatch = np.isfinite(array) != np.isfinite(transposed)
    if np.any(finite_mismatch):
        return False, float("inf")
    mask = np.isfinite(array) & np.isfinite(transposed)
    if not np.any(mask):
        return True, 0.0
    difference = float(np.max(np.abs(array[mask] - transposed[mask])))
    return difference <= tolerance, difference


def validate_structure(run_dir: Path, state: ValidationState) -> None:
    required_files = [
        "algorithm_config.json",
        "run_manifest.json",
        "input_inventory.csv",
        "artifact_index.csv",
        "REPORT.md",
    ]
    for relative in required_files:
        state.add(f"structure_file_{normalized_name(relative)}", (run_dir / relative).is_file(), relative)

    workbooks = sorted(run_dir.glob("*.xlsx"))
    state.add("structure_workbook", len(workbooks) == 1, [path.name for path in workbooks])
    for directory in ("validation", "robustness", "comparison_to_v2"):
        state.add(f"structure_dir_{directory}", (run_dir / directory).is_dir(), directory)

    channel_directories = (
        "per_peak/area",
        "per_peak/location",
        "per_peak/presence",
        "per_peak/support",
        "per_peak/trajectories",
        "across_frames/acf_strict",
        "across_frames/direct_strict",
        "across_frames/shift_tolerant_secondary",
        "within_frame/all_windows",
        "within_frame/nonoverlap_control",
        "within_frame/by_pressure",
    )
    channel_files = (
        "per_peak/peak_observations.csv",
        "per_peak/canonical_tracks.csv",
        "per_peak/peak_summary.csv",
        "per_peak/per_peak_matrices.npz",
        "across_frames/across_frame_matrices.npz",
        "within_frame/within_frame_matrices.npz",
    )
    channels = declared_channels(run_dir)
    state.add("structure_declared_channels", bool(channels), channels)
    for channel in channels:
        state.add(f"structure_channel_{channel}", (run_dir / channel).is_dir(), channel)
        for relative in channel_directories:
            target = run_dir / channel / relative
            state.add(
                f"structure_{channel}_{normalized_name(relative)}",
                target.is_dir(),
                str(Path(channel) / relative),
            )
        for relative in channel_files:
            target = run_dir / channel / relative
            state.add(
                f"structure_{channel}_{normalized_name(relative)}_file",
                target.is_file(),
                str(Path(channel) / relative),
            )
        for audit_name in (
            "link_evidence.csv",
            "ambiguity_events.csv",
            "segment_lineage.csv",
            "quarantined_nodes.csv",
            "selection_audit.csv",
        ):
            target = run_dir / channel / "per_peak" / audit_name
            state.add(
                f"structure_{channel}_{normalized_name(audit_name)}",
                target.is_file(),
                str(target.relative_to(run_dir)),
            )


def validate_manifests(run_dir: Path, state: ValidationState) -> None:
    config_path = run_dir / "algorithm_config.json"
    manifest_path = run_dir / "run_manifest.json"
    inventory_path = run_dir / "input_inventory.csv"
    if not (config_path.is_file() and manifest_path.is_file()):
        return
    try:
        config = load_json(config_path)
        manifest = load_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        state.add("manifests_json_parse", False, str(error))
        return
    state.add("manifests_json_parse", True)
    state.add("manifests_config_finite", not contains_nonfinite_json(config))
    state.add("manifests_run_finite", not contains_nonfinite_json(manifest))

    profile_aliases = {"profile", "profile_id", "algorithm_profile", "algorithm_version"}
    profiles = [str(value) for value in recursive_values_for_keys(config, profile_aliases)]
    profiles.extend(str(value) for value in recursive_values_for_keys(manifest, profile_aliases))
    state.add("manifest_profile_frozen", PROFILE_ID in profiles, {"found": sorted(set(profiles))})

    seeds = recursive_values_for_keys(config, {"seed", "random_seed", "bootstrap_seed"})
    seeds.extend(recursive_values_for_keys(manifest, {"seed", "random_seed", "bootstrap_seed"}))
    numeric_seeds: list[int] = []
    for seed in seeds:
        try:
            numeric_seeds.append(int(seed))
        except (TypeError, ValueError):
            pass
    state.add("manifest_fixed_seed_zero", bool(numeric_seeds) and all(seed == 0 for seed in numeric_seeds), numeric_seeds)

    digest = sha256_file(config_path)
    recorded_digests = [
        str(value).lower()
        for value in recursive_values_for_keys(
            manifest,
            {"algorithm_config_sha256", "config_sha256", "algorithm_sha256", "profile_sha256"},
        )
    ]
    state.add(
        "manifest_config_sha256",
        digest in recorded_digests,
        {"computed": digest, "recorded": recorded_digests},
    )

    semantics = manifest.get("resolved_algorithm_semantics", {})
    recorded_semantic_sha = str(manifest.get("execution_semantics_sha256", "")).lower()
    computed_semantic_sha = ""
    if isinstance(semantics, Mapping) and semantics:
        encoded = json.dumps(
            json_safe(semantics),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        computed_semantic_sha = hashlib.sha256(encoded).hexdigest()
    state.add(
        "manifest_execution_semantics_sha256",
        bool(computed_semantic_sha)
        and SHA256_RE.fullmatch(recorded_semantic_sha) is not None
        and computed_semantic_sha == recorded_semantic_sha,
        {"computed": computed_semantic_sha, "recorded": recorded_semantic_sha},
    )
    binding_audit = manifest.get("config_binding_audit", {})
    binding_details: dict[str, Any] = {}
    binding_ok = True
    for family in (
        "peak_config",
        "window_config",
        "segmented_tracking_config",
        "tracking_policy",
    ):
        resolved_family = semantics.get(family, {}) if isinstance(semantics, Mapping) else {}
        audit_family = binding_audit.get(family, {}) if isinstance(binding_audit, Mapping) else {}
        resolved_keys = set(resolved_family) if isinstance(resolved_family, Mapping) else set()
        audit_keys = set(audit_family) if isinstance(audit_family, Mapping) else set()
        sources_nonempty = isinstance(audit_family, Mapping) and all(
            isinstance(source, str) and bool(source.strip()) for source in audit_family.values()
        )
        family_ok = bool(resolved_keys) and resolved_keys == audit_keys and sources_nonempty
        binding_ok = binding_ok and family_ok
        binding_details[family] = {
            "resolved_field_count": len(resolved_keys),
            "audited_field_count": len(audit_keys),
            "missing_audit_fields": sorted(resolved_keys - audit_keys),
            "unexpected_audit_fields": sorted(audit_keys - resolved_keys),
            "sources_nonempty": sources_nonempty,
        }
    state.add("manifest_all_config_fields_bound", binding_ok, binding_details)

    code_sha = manifest.get("environment", {}).get("code_sha256", {})
    binding_code_sha = code_sha.get("uniform_profile_binding_v21.py") if isinstance(code_sha, Mapping) else None
    state.add(
        "manifest_profile_binding_code_versioned",
        isinstance(binding_code_sha, str) and SHA256_RE.fullmatch(binding_code_sha) is not None,
        {"uniform_profile_binding_v21.py": binding_code_sha},
    )

    override_values = recursive_values_for_keys(manifest, {"overrides", "parameter_overrides", "cli_overrides"})
    nonempty_overrides = [value for value in override_values if value not in (None, {}, [], "")]
    statuses = [
        str(value).upper()
        for value in recursive_values_for_keys(manifest, {"status", "profile_status", "run_status"})
    ]
    override_ok = not nonempty_overrides or "EXPERIMENTAL" in statuses
    state.add("manifest_override_semantics", override_ok, {"overrides": nonempty_overrides, "statuses": statuses})

    if not inventory_path.is_file():
        return
    try:
        fields, rows = read_csv_rows(inventory_path)
    except (OSError, UnicodeError, csv.Error) as error:
        state.add("inventory_readable", False, str(error))
        return
    state.add("inventory_readable", bool(fields) and bool(rows), {"rows": len(rows), "columns": fields})
    normalized_fields = {normalized_name(field): field for field in fields}
    path_field = next(
        (normalized_fields[name] for name in ("path", "file_path", "source_path", "xy_path") if name in normalized_fields),
        None,
    )
    hash_field = next(
        (normalized_fields[name] for name in ("sha256", "file_sha256", "input_sha256") if name in normalized_fields),
        None,
    )
    state.add("inventory_path_and_sha_columns", path_field is not None and hash_field is not None)
    if path_field is None or hash_field is None:
        return
    inventory_paths = [row.get(path_field, "").strip() for row in rows]
    hashes = [row.get(hash_field, "").strip().lower() for row in rows]
    state.add("inventory_unique_paths", len(inventory_paths) == len(set(inventory_paths)))
    # Stable order is channel order from the manifest, then scan/pressure/frame.
    # Requiring a global lexical path sort would reject a deterministic
    # channel-block inventory (the official writer's intentional layout).
    channel_order = {
        str(channel): index for index, channel in enumerate(manifest.get("channels", []))
    } if isinstance(manifest, Mapping) else {}
    channel_field = normalized_fields.get("channel")
    scan_field = normalized_fields.get("scan")
    pressure_field = normalized_fields.get("pressure_gpa")
    frame_field = normalized_fields.get("frame")

    def inventory_sort_key(row: Mapping[str, str]) -> tuple[Any, ...]:
        channel = row.get(channel_field, "") if channel_field else ""
        scan = row.get(scan_field, "") if scan_field else ""
        try:
            pressure = float(row.get(pressure_field, "nan")) if pressure_field else math.nan
        except ValueError:
            pressure = math.inf
        try:
            frame = int(row.get(frame_field, "")) if frame_field else -1
        except ValueError:
            frame = -1
        path_value = row.get(path_field, "")
        return (channel_order.get(channel, len(channel_order)), channel, scan, pressure, frame, path_value)

    state.add("inventory_deterministic_order", rows == sorted(rows, key=inventory_sort_key))
    state.add("inventory_sha256_format", all(SHA256_RE.fullmatch(value) for value in hashes))
    missing: list[str] = []
    mismatched: list[str] = []
    for raw_path, expected in zip(inventory_paths, hashes):
        source = Path(raw_path).expanduser()
        if not source.is_absolute():
            source = run_dir / source
        if not source.is_file():
            missing.append(raw_path)
            continue
        if SHA256_RE.fullmatch(expected) and sha256_file(source) != expected:
            mismatched.append(raw_path)
    state.add("inventory_sources_exist", not missing, {"missing_count": len(missing), "examples": missing[:10]})
    state.add(
        "inventory_source_hashes_match",
        not mismatched,
        {"mismatch_count": len(mismatched), "examples": mismatched[:10]},
    )

    inventory_digest = sha256_file(inventory_path)
    recorded_inventory = [
        str(value).lower()
        for value in recursive_values_for_keys(manifest, {"input_inventory_sha256", "inventory_sha256"})
    ]
    state.add(
        "manifest_inventory_sha256_if_recorded",
        not recorded_inventory or inventory_digest in recorded_inventory,
        {"computed": inventory_digest, "recorded": recorded_inventory},
    )


def validate_official_acceptance_evidence(
    run_dir: Path, state: ValidationState, tolerance: float
) -> None:
    """Gate the official completion marker on the user's full acceptance contract."""

    manifest_path = run_dir / "run_manifest.json"
    config_path = run_dir / "algorithm_config.json"
    inventory_path = run_dir / "input_inventory.csv"
    try:
        manifest = load_json(manifest_path)
        config = load_json(config_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        state.add("official_acceptance_manifests_readable", False, str(error))
        return

    state.add("official_acceptance_manifests_readable", True)
    state.add(
        "official_profile_status",
        str(manifest.get("profile_status", "")).upper() == "OFFICIAL_FROZEN"
        and str(config.get("status", "")).upper() == "OFFICIAL_FROZEN",
        {
            "manifest_profile_status": manifest.get("profile_status"),
            "config_status": config.get("status"),
        },
    )
    channels = [str(item).strip().lower() for item in manifest.get("channels", [])]
    state.add(
        "official_declared_channels_present",
        bool(channels) and len(channels) == len(set(channels)),
        {"channels": channels},
    )
    state.add(
        "official_plots_written",
        manifest.get("plots_written") is True,
        {"plots_written": manifest.get("plots_written")},
    )
    try:
        wavelength = float(manifest.get("wavelength_A"))
    except (TypeError, ValueError):
        wavelength = math.nan
    state.add(
        "official_explicit_positive_wavelength",
        math.isfinite(wavelength) and wavelength > 0.0,
        {"wavelength_A": manifest.get("wavelength_A")},
    )
    state.add(
        "official_reference_tracks_posthoc_only",
        manifest.get("reference_tracks_role") in {"posthoc_annotation_only", "not_supplied"},
        {"reference_tracks_role": manifest.get("reference_tracks_role")},
    )

    inventory_rows: list[dict[str, str]] = []
    try:
        _fields, inventory_rows = read_csv_rows(inventory_path)
    except (OSError, UnicodeError, csv.Error):
        pass
    try:
        frame_count = int(manifest.get("frames", 0))
    except (TypeError, ValueError):
        frame_count = 0
    channel_counts: dict[str, int] = {}
    for row in inventory_rows:
        channel = str(row.get("channel", "")).strip().lower()
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
    expected_inventory_rows = frame_count * len(channels)
    state.add(
        "official_full_inventory_coverage",
        frame_count > 0
        and len(inventory_rows) == expected_inventory_rows
        and all(channel_counts.get(channel, 0) == frame_count for channel in channels),
        {
            "frames": frame_count,
            "inventory_rows": len(inventory_rows),
            "expected_rows": expected_inventory_rows,
            "channel_counts": channel_counts,
        },
    )

    synthetic_path = run_dir / "validation" / "synthetic_validation.json"
    state.add("official_synthetic_report_present", synthetic_path.is_file(), str(synthetic_path))
    synthetic: Mapping[str, Any] = {}
    if synthetic_path.is_file():
        try:
            loaded = load_json(synthetic_path)
            synthetic = loaded if isinstance(loaded, Mapping) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            synthetic = {}
    cases = synthetic.get("cases", {}) if isinstance(synthetic, Mapping) else {}
    case_passes = {
        case: parse_bool(cases.get(case)) if isinstance(cases, Mapping) else None
        for case in sorted(REQUIRED_SYNTHETIC_CASES)
    }
    state.add(
        "official_synthetic_cases_passed",
        parse_bool(synthetic.get("passed")) is True
        and all(case_passes.get(case) is True for case in REQUIRED_SYNTHETIC_CASES),
        {"cases": case_passes, "test_count": synthetic.get("test_count")},
    )

    reproducibility_path = run_dir / "validation" / "reproducibility_report.json"
    state.add(
        "official_reproducibility_report_present",
        reproducibility_path.is_file(),
        str(reproducibility_path),
    )
    reproducibility: Mapping[str, Any] = {}
    if reproducibility_path.is_file():
        try:
            loaded = load_json(reproducibility_path)
            reproducibility = loaded if isinstance(loaded, Mapping) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            reproducibility = {}
    details = reproducibility.get("details", {}) if isinstance(reproducibility, Mapping) else {}
    reproducibility_checks = (
        reproducibility.get("checks", {}) if isinstance(reproducibility, Mapping) else {}
    )
    required_reproducibility_checks = {
        "same_profile_sha256",
        "same_code_sha256",
        "same_input_inventory_sha256",
        "same_scientific_file_set",
        "complete_scientific_file_coverage",
        "all_scientific_values_reproducible",
        "max_abs_difference_within_tolerance",
    }
    try:
        reproducibility_tolerance = float(reproducibility.get("tolerance", math.inf))
        maximum_difference = float(details.get("max_abs_difference", math.inf))
        csv_files_compared = int(details.get("scientific_csv_files_compared", 0))
        npz_files_compared = int(details.get("scientific_npz_files_compared", 0))
        matched_files = int(details.get("matched_scientific_files", 0))
        expected_first = int(details.get("expected_scientific_files_first", -1))
        expected_second = int(details.get("expected_scientific_files_second", -1))
    except (TypeError, ValueError):
        reproducibility_tolerance = math.inf
        maximum_difference = math.inf
        csv_files_compared = 0
        npz_files_compared = 0
        matched_files = 0
        expected_first = -1
        expected_second = -1
    unmatched_files = details.get("unmatched_scientific_files", ["not_reported"])
    all_required_repro_checks = isinstance(reproducibility_checks, Mapping) and all(
        parse_bool(reproducibility_checks.get(name)) is True
        for name in required_reproducibility_checks
    )
    state.add(
        "official_full_rerun_reproducible",
        parse_bool(reproducibility.get("passed")) is True
        and all_required_repro_checks
        and reproducibility_tolerance <= DEFAULT_TOLERANCE
        and maximum_difference <= DEFAULT_TOLERANCE
        and csv_files_compared > 0
        and npz_files_compared > 0
        and matched_files == expected_first == expected_second
        and expected_first == csv_files_compared + npz_files_compared
        and unmatched_files == [],
        {
            "reported_tolerance": reproducibility_tolerance,
            "maximum_difference": maximum_difference,
            "csv_files_compared": csv_files_compared,
            "npz_files_compared": npz_files_compared,
            "matched_files": matched_files,
            "expected_first": expected_first,
            "expected_second": expected_second,
            "unmatched_files": unmatched_files,
            "required_checks": {
                name: reproducibility_checks.get(name)
                if isinstance(reproducibility_checks, Mapping)
                else None
                for name in sorted(required_reproducibility_checks)
            },
        },
    )

    legacy_path = run_dir / "comparison_to_v2" / "v2_integrity.json"
    state.add("official_v2_integrity_report_present", legacy_path.is_file(), str(legacy_path))
    legacy: Mapping[str, Any] = {}
    if legacy_path.is_file():
        try:
            loaded = load_json(legacy_path)
            legacy = loaded if isinstance(loaded, Mapping) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            legacy = {}
    files_before = legacy.get("files_before", legacy.get("files"))
    files_after = legacy.get("files_after")
    state.add(
        "official_v2_directory_unchanged",
        parse_bool(legacy.get("unchanged")) is True
        and legacy.get("sha256_before") == legacy.get("sha256_after")
        and files_before == files_after,
        {
            "files_before": files_before,
            "files_after": files_after,
            "sha256_before": legacy.get("sha256_before"),
            "sha256_after": legacy.get("sha256_after"),
        },
    )

    if str(manifest.get("input_mode", "")).lower() == "handoff":
        comparison_path = run_dir / "comparison_to_v2" / "across_within_comparison.json"
        state.add(
            "official_across_within_v2_comparison_present",
            comparison_path.is_file(),
            str(comparison_path),
        )
        comparison: Mapping[str, Any] = {}
        if comparison_path.is_file():
            try:
                loaded = load_json(comparison_path)
                comparison = loaded if isinstance(loaded, Mapping) else {}
            except (OSError, UnicodeError, json.JSONDecodeError):
                comparison = {}
        try:
            maximum = float(comparison.get("maximum_absolute_difference", math.inf))
        except (TypeError, ValueError):
            maximum = math.inf
        state.add(
            "official_across_within_match_v2",
            parse_bool(comparison.get("passed")) is True and maximum <= tolerance,
            {"maximum_absolute_difference": maximum, "tolerance": tolerance},
        )

        gui_dir = run_dir / "validation" / "gui_crosscheck"
        gui_required = {
            "crosscheck_summary.json",
            "gui_peak_table.csv",
            "peak_match_table.csv",
            "peak_match_summary.csv",
            "rejection_reason_agreement.csv",
            "track_range_slope_comparison.csv",
            "strict_boundary_cells.csv",
            "strict_boundary_agreement_summary.csv",
            "pattern_window_checks.csv",
            "spots_fit_control_window_auc.csv",
            "spots_fit_control_auc_comparison.csv",
            "spots_fit_control_boundary_comparison.csv",
            "gui_peak_map_pressure_all_area.png",
            "gui_peak_map_pressure_good_only_area.png",
            "gui_peak_map_pressure_all_fwhm.png",
            "gui_peak_map_pressure_good_only_fwhm.png",
            "gui_pattern_map_pressure_clean.png",
            "gui_v21_matched_peak_overlay_pressure.png",
            "visualization_manifest.csv",
            "GUI_CROSSCHECK_REPORT.md",
        }
        missing_gui = sorted(name for name in gui_required if not (gui_dir / name).is_file())
        state.add(
            "official_gui_crosscheck_artifacts_present",
            not missing_gui,
            {"directory": str(gui_dir), "missing": missing_gui},
        )
        gui_summary: Mapping[str, Any] = {}
        gui_summary_path = gui_dir / "crosscheck_summary.json"
        if gui_summary_path.is_file():
            try:
                loaded = load_json(gui_summary_path)
                gui_summary = loaded if isinstance(loaded, Mapping) else {}
            except (OSError, UnicodeError, json.JSONDecodeError):
                gui_summary = {}
        result_root_recorded = str(gui_summary.get("result_root", ""))
        legacy_root_recorded = str(gui_summary.get("legacy_root", ""))
        state.add(
            "official_gui_crosscheck_profiles_verified",
            gui_summary.get("result_profile_verified") == "uniform-correlation-v2.1"
            and gui_summary.get("legacy_profile_verified") == "uniform-correlation-v2"
            and result_root_recorded == str(run_dir.resolve())
            and bool(legacy_root_recorded)
            and legacy_root_recorded != result_root_recorded,
            {
                "result_profile_verified": gui_summary.get("result_profile_verified"),
                "legacy_profile_verified": gui_summary.get("legacy_profile_verified"),
                "result_root": result_root_recorded,
                "legacy_root": legacy_root_recorded,
            },
        )
        coverage = gui_summary.get("coverage", {}) if isinstance(gui_summary, Mapping) else {}
        try:
            gui_scans = int(coverage.get("spots_gui_scans", 0))
            correlation_scans = int(coverage.get("spots_correlation_scans", 0))
            fraction = float(coverage.get("spots_fraction", math.nan))
        except (TypeError, ValueError):
            gui_scans = 0
            correlation_scans = 0
            fraction = math.nan
        expected_scans = len(manifest.get("scans", []))
        expected_fraction = gui_scans / correlation_scans if correlation_scans > 0 else math.nan
        verdict = str(gui_summary.get("verdict", ""))
        state.add(
            "official_gui_crosscheck_coverage_truthfully_reported",
            gui_scans > 0
            and correlation_scans == expected_scans
            and math.isfinite(fraction)
            and math.isclose(fraction, expected_fraction, rel_tol=0.0, abs_tol=1.0e-15)
            and (
                (gui_scans < correlation_scans and verdict == "partial_gui_coverage")
                or (gui_scans == correlation_scans and verdict == "complete_gui_coverage")
            ),
            {
                "spots_gui_scans": gui_scans,
                "spots_correlation_scans": correlation_scans,
                "manifest_scans": expected_scans,
                "spots_fraction": fraction,
                "verdict": verdict,
                "fit_gui_h5_missing": coverage.get("fit_gui_h5_missing"),
            },
        )
        main_matches = gui_summary.get("main_v21_good_only_match_summary", [])
        valid_main_matches = [
            row
            for row in main_matches
            if isinstance(row, Mapping)
            and row.get("result_version") == "uniform-correlation-v2.1"
            and row.get("match_view") == "good_only"
        ] if isinstance(main_matches, list) else []
        state.add(
            "official_gui_crosscheck_v21_peak_matches_present",
            bool(valid_main_matches)
            and all(int(row.get("matched", 0)) > 0 for row in valid_main_matches),
            {"summaries": valid_main_matches},
        )
        auc_values = gui_summary.get("median_near_far_auc", {})
        required_auc = {
            "spots:acf_strict",
            "spots:direct_strict",
            "fit:acf_strict",
            "fit:direct_strict",
        }
        finite_auc = {}
        if isinstance(auc_values, Mapping):
            for key in required_auc:
                try:
                    finite_auc[key] = math.isfinite(float(auc_values.get(key)))
                except (TypeError, ValueError):
                    finite_auc[key] = False
        state.add(
            "official_gui_crosscheck_spots_fit_control_present",
            set(finite_auc) == required_auc and all(finite_auc.values()),
            {"finite_auc": finite_auc, "values": auc_values},
        )
        try:
            boundary_rows = int(gui_summary.get("strict_boundary_rows", 0))
            pattern_rows = int(gui_summary.get("pattern_window_rows", 0))
        except (TypeError, ValueError):
            boundary_rows = 0
            pattern_rows = 0
        state.add(
            "official_gui_crosscheck_pattern_boundary_support_ci_audited",
            boundary_rows > 0
            and pattern_rows > 0
            and "low-similarity candidate" in str(gui_summary.get("boundary_formula", ""))
            and "corroborated" in str(gui_summary.get("boundary_formula", "")),
            {
                "strict_boundary_rows": boundary_rows,
                "pattern_window_rows": pattern_rows,
                "boundary_formula": gui_summary.get("boundary_formula"),
            },
        )
        state.add(
            "official_gui_crosscheck_read_only_guardrail",
            str(gui_summary.get("role", "")).startswith("read_only_validation"),
            {"role": gui_summary.get("role")},
        )


def zero_official_track_evidence(channel_dir: Path) -> tuple[bool, dict[str, Any]]:
    """Return whether two independent CSV tables explicitly report zero official tracks."""

    canonical_path = channel_dir / "per_peak" / "canonical_tracks.csv"
    summary_path = channel_dir / "per_peak" / "peak_summary.csv"
    detail: dict[str, Any] = {
        "canonical_tracks": str(canonical_path),
        "peak_summary": str(summary_path),
        "canonical_rows": None,
        "official_rows": None,
        "summary_rows": None,
    }
    if not (canonical_path.is_file() and summary_path.is_file()):
        return False, detail
    try:
        canonical_fields, canonical_rows = read_csv_rows(canonical_path)
        summary_fields, summary_rows = read_csv_rows(summary_path)
    except (OSError, UnicodeError, csv.Error):
        return False, detail
    canonical_names = {normalized_name(field): field for field in canonical_fields}
    summary_names = {normalized_name(field): field for field in summary_fields}
    official_field = canonical_names.get("official")
    detail.update(
        {
            "canonical_rows": len(canonical_rows),
            "summary_rows": len(summary_rows),
            "canonical_has_track_id": "track_id" in canonical_names,
            "canonical_has_official": official_field is not None,
            "summary_has_track_id": "track_id" in summary_names,
        }
    )
    if (
        official_field is None
        or "track_id" not in canonical_names
        or "track_id" not in summary_names
    ):
        return False, detail
    official_values = [parse_bool(row.get(official_field)) for row in canonical_rows]
    if any(value is None for value in official_values):
        detail["invalid_official_values"] = True
        return False, detail
    official_count = sum(value is True for value in official_values)
    detail["official_rows"] = official_count
    return official_count == 0 and len(summary_rows) == 0, detail


def _finite_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _split_ids(value: Any) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(";") if item.strip())


def validate_v21_tracking_audits(run_dir: Path, state: ValidationState) -> None:
    """Validate the edge-level CUT contract independently from the runner."""

    allowed_cut_reasons = {
        "cut_one_way",
        "cut_low_margin",
        "cut_order_crossing",
        "cut_missing_too_long",
        "cut_outside_gate",
    }
    required_link_fields = {
        "edge_id",
        "first_consensus_id",
        "second_consensus_id",
        "first_pressure_GPa",
        "second_pressure_GPa",
        "first_pressure_index",
        "second_pressure_index",
        "first_q_A^-1",
        "second_q_A^-1",
        "forward_evaluated",
        "backward_evaluated",
        "forward_admissible",
        "backward_admissible",
        "forward_matched",
        "backward_matched",
        "forward_cost",
        "backward_cost",
        "forward_source_margin",
        "forward_target_margin",
        "backward_source_margin",
        "backward_target_margin",
        "missing_pressure_levels",
        "within_original_gate",
        "endpoint_quarantined",
        "order_crossing",
        "accepted",
        "cut_reason",
    }
    for channel in declared_channels(run_dir):
        root = run_dir / channel / "per_peak"
        try:
            link_fields, links = read_csv_rows(root / "link_evidence.csv")
            lineage_fields, lineage = read_csv_rows(root / "segment_lineage.csv")
            quarantine_fields, quarantine = read_csv_rows(root / "quarantined_nodes.csv")
            selection_fields, selection = read_csv_rows(root / "selection_audit.csv")
            summary_fields, selection_summary = read_csv_rows(
                root / "selection_audit_summary.csv"
            )
        except (OSError, UnicodeError, csv.Error) as error:
            state.add(f"v21_{channel}_audits_readable", False, str(error))
            continue
        state.add(f"v21_{channel}_audits_readable", True)
        missing_link_fields = sorted(required_link_fields - set(link_fields))
        state.add(
            f"v21_{channel}_link_schema_complete",
            not missing_link_fields,
            {"missing": missing_link_fields, "rows": len(links)},
        )

        link_failures: list[dict[str, Any]] = []
        for row in links:
            accepted = parse_bool(row.get("accepted")) is True
            forward = parse_bool(row.get("forward_matched")) is True
            backward = parse_bool(row.get("backward_matched")) is True
            quarantined = parse_bool(row.get("endpoint_quarantined")) is True
            crossing = parse_bool(row.get("order_crossing")) is True
            within_gate = parse_bool(row.get("within_original_gate")) is True
            missing = _finite_float(row.get("missing_pressure_levels"))
            reasons = set(_split_ids(row.get("cut_reason")))
            unknown_reasons = sorted(reasons - allowed_cut_reasons)
            margins = [
                _finite_float(row.get(name))
                for name in (
                    "forward_source_margin",
                    "forward_target_margin",
                    "backward_source_margin",
                    "backward_target_margin",
                )
            ]
            if unknown_reasons:
                link_failures.append({"edge": row.get("edge_id"), "unknown_reasons": unknown_reasons})
            if accepted and not (
                forward
                and backward
                and within_gate
                and not quarantined
                and not crossing
                and missing is not None
                and missing <= 2
                and all(value is not None and value >= 0.25 for value in margins)
                and not reasons
            ):
                link_failures.append({"edge": row.get("edge_id"), "reason": "accepted_edge_gate_failure"})
            if forward != backward and "cut_one_way" not in reasons:
                link_failures.append({"edge": row.get("edge_id"), "reason": "one_way_without_cut"})
            if crossing and (accepted or "cut_order_crossing" not in reasons):
                link_failures.append({"edge": row.get("edge_id"), "reason": "crossing_not_cut"})
            if missing is not None and missing > 2 and (
                accepted or "cut_missing_too_long" not in reasons
            ):
                link_failures.append({"edge": row.get("edge_id"), "reason": "long_gap_not_cut"})
        state.add(
            f"v21_{channel}_accepted_links_meet_frozen_rules",
            not link_failures,
            {"failure_count": len(link_failures), "examples": link_failures[:25]},
        )

        required_lineage = {
            "parent_track_id",
            "segment_id",
            "segment_index",
            "segment_count",
            "official",
            "pressure_nodes",
            "minimum_pressure_nodes",
            "node_ids",
            "boundary_unknown_pressure_indices",
        }
        state.add(
            f"v21_{channel}_lineage_schema_complete",
            required_lineage.issubset(lineage_fields),
            {"missing": sorted(required_lineage - set(lineage_fields))},
        )
        used_nodes: set[str] = set()
        duplicate_nodes: set[str] = set()
        id_failures: list[str] = []
        official_segments: list[dict[str, str]] = []
        for row in lineage:
            segment_id = str(row.get("segment_id", ""))
            if not re.fullmatch(r"radial_peak_\d{3}_segment_\d{2}", segment_id):
                id_failures.append(segment_id)
            for identifier in _split_ids(row.get("node_ids")):
                if identifier in used_nodes:
                    duplicate_nodes.add(identifier)
                used_nodes.add(identifier)
            if parse_bool(row.get("official")) is True:
                official_segments.append(row)
                nodes = _finite_float(row.get("pressure_nodes"))
                required = _finite_float(row.get("minimum_pressure_nodes"))
                if nodes is None or required is None or nodes < required:
                    id_failures.append(f"{segment_id}:support")
        state.add(
            f"v21_{channel}_segment_ids_and_node_uniqueness",
            not duplicate_nodes and not id_failures,
            {"duplicate_nodes": sorted(duplicate_nodes), "id_failures": id_failures[:25]},
        )

        required_quarantine = {"consensus_id", "pressure_GPa", "pressure_index", "reason"}
        required_selection = {
            "consensus_id",
            "q_A^-1",
            "fwhm_q_A^-1",
            "relative_area",
            "scan_support",
            "selection_status",
            "official_segment_ids",
        }
        state.add(
            f"v21_{channel}_quarantine_selection_schema_complete",
            required_quarantine.issubset(quarantine_fields)
            and required_selection.issubset(selection_fields)
            and bool(summary_fields),
            {
                "quarantine_missing": sorted(required_quarantine - set(quarantine_fields)),
                "selection_missing": sorted(required_selection - set(selection_fields)),
            },
        )
        quarantined_ids = {row.get("consensus_id", "") for row in quarantine}
        retained_quarantined = [
            row.get("consensus_id", "")
            for row in selection
            if row.get("consensus_id", "") in quarantined_ids
            and _split_ids(row.get("official_segment_ids"))
        ]
        summary_text = " ".join(
            str(value) for row in selection_summary for value in row.values()
        ).lower()
        state.add(
            f"v21_{channel}_quarantined_nodes_not_officially_retained",
            not retained_quarantined,
            retained_quarantined[:25],
        )
        state.add(
            f"v21_{channel}_selection_bias_summary_present",
            bool(selection)
            and bool(selection_summary)
            and "retained_fraction" in summary_text,
            {"selection_rows": len(selection), "summary_rows": len(selection_summary)},
        )

        official_ids = {str(row.get("segment_id", "")) for row in official_segments}
        segment_mask_failures: list[dict[str, Any]] = []
        for family in ("area", "location", "presence"):
            matrices = set(
                path.stem for path in (root / family / "matrices").glob("*.csv")
            )
            heatmaps = set(
                path.stem for path in (root / family / "heatmaps").glob("*.png")
            )
            supports = set(
                path.name.removesuffix("_support.csv")
                for path in (root / family / "support_maps").glob("*_support.csv")
            )
            state.add(
                f"v21_{channel}_{family}_artifacts_match_official_segments",
                matrices == official_ids and heatmaps == official_ids and supports == official_ids,
                {
                    "official": len(official_ids),
                    "matrices": len(matrices),
                    "heatmaps": len(heatmaps),
                    "supports": len(supports),
                },
            )
            if family in {"area", "location"}:
                rows_by_id = {
                    str(row.get("segment_id", "")): row for row in official_segments
                }
                for segment_id in sorted(official_ids):
                    parsed = read_csv_matrix(root / family / "matrices" / f"{segment_id}.csv")
                    if parsed is None:
                        segment_mask_failures.append(
                            {"segment": segment_id, "family": family, "reason": "matrix_unreadable"}
                        )
                        continue
                    labels, matrix = parsed
                    try:
                        pressure_values = np.asarray([float(item) for item in labels], dtype=float)
                    except ValueError:
                        segment_mask_failures.append(
                            {"segment": segment_id, "family": family, "reason": "pressure_labels"}
                        )
                        continue
                    segment_row = rows_by_id[segment_id]
                    lower = _finite_float(segment_row.get("pressure_min_GPa"))
                    upper = _finite_float(segment_row.get("pressure_max_GPa"))
                    if lower is None or upper is None:
                        segment_mask_failures.append(
                            {"segment": segment_id, "family": family, "reason": "range_missing"}
                        )
                        continue
                    outside = (pressure_values < lower - 1.0e-12) | (
                        pressure_values > upper + 1.0e-12
                    )
                    if np.any(np.isfinite(matrix[outside, :])) or np.any(
                        np.isfinite(matrix[:, outside])
                    ):
                        segment_mask_failures.append(
                            {"segment": segment_id, "family": family, "reason": "finite_outside_segment"}
                        )
                    for index_text in _split_ids(
                        segment_row.get("boundary_unknown_pressure_indices")
                    ):
                        try:
                            index = int(index_text)
                        except ValueError:
                            segment_mask_failures.append(
                                {"segment": segment_id, "family": family, "reason": "bad_boundary_index"}
                            )
                            continue
                        if 0 <= index < matrix.shape[0] and (
                            np.any(np.isfinite(matrix[index, :]))
                            or np.any(np.isfinite(matrix[:, index]))
                        ):
                            segment_mask_failures.append(
                                {"segment": segment_id, "family": family, "reason": f"finite_at_cut_{index}"}
                            )
        state.add(
            f"v21_{channel}_no_values_outside_segments_or_across_cuts",
            not segment_mask_failures,
            {"failures": segment_mask_failures[:25], "count": len(segment_mask_failures)},
        )


def validate_matrices(run_dir: Path, state: ValidationState, tolerance: float) -> list[MatrixItem]:
    items = list(iter_matrix_items(run_dir))
    roles = ("area", "location", "presence", "support", "signed")
    role_counts = {role: sum(item.role == role for item in items) for role in roles}
    zero_evidence: dict[str, tuple[bool, dict[str, Any]]] = {}
    channels = declared_channels(run_dir)
    for channel in channels:
        channel_dir = run_dir / channel
        zero_evidence[channel] = zero_official_track_evidence(channel_dir)
        channel_items = [
            item
            for item in items
            if item.source == channel_dir or channel_dir in item.source.parents
        ]
        for role in ("area", "location", "presence"):
            family_items = [item for item in channel_items if item.role == role]
            nonempty_count = sum(item.values.size > 0 for item in family_items)
            explicitly_zero, evidence_detail = zero_evidence[channel]
            state.add(
                f"matrix_family_{channel}_{role}_present_or_zero_tracks",
                explicitly_zero or nonempty_count > 0,
                {
                    "artifact_count": len(family_items),
                    "nonempty_matrix_count": nonempty_count,
                    "zero_official_tracks": explicitly_zero,
                    "track_evidence": evidence_detail,
                },
            )
        support_count = sum(item.role == "support" for item in channel_items)
        state.add(
            f"matrix_family_{channel}_support_present",
            support_count > 0,
            {
                "artifact_count": support_count,
                "zero_official_tracks": zero_evidence[channel][0],
            },
        )
    for role, count in role_counts.items():
        if role in {"area", "location", "presence"}:
            all_channels_explicitly_zero = bool(zero_evidence) and all(
                value[0] for value in zero_evidence.values()
            )
            nonempty_count = sum(
                item.role == role and item.values.size > 0 for item in items
            )
            passed = all_channels_explicitly_zero or nonempty_count > 0
            detail = {
                "artifact_count": count,
                "nonempty_matrix_count": nonempty_count,
                "all_channels_explicitly_zero": all_channels_explicitly_zero,
            }
        else:
            passed = count > 0
            detail = {"count": count}
        state.add(f"matrix_family_{role}_present", passed, detail)
    failures: list[dict[str, Any]] = []
    range_failures: list[str] = []
    support_failures: list[str] = []
    symmetry_checked = 0
    for item in items:
        # n10 and n01 are directional by definition; the pair is validated as
        # transposes below.  Every other plotted/count matrix is symmetric.
        directional_count = normalized_name(item.key) in {"n10", "n01"} or bool(
            re.search(r"_(?:n10|n01)$", normalized_name(item.source.stem))
        )
        if not directional_count:
            symmetry_checked += 1
            symmetric, difference = symmetric_last_axes(item.values, tolerance)
            if not symmetric:
                failures.append({"matrix": item.label, "max_abs_difference": difference})
        if item.role in {"area", "location", "presence"}:
            if not finite_range_ok(item.values, 0.0, 1.0, tolerance):
                range_failures.append(item.label)
        elif item.role == "signed":
            if not finite_range_ok(item.values, -1.0, 1.0, tolerance):
                range_failures.append(item.label)
        elif item.role == "support":
            finite = item.values[np.isfinite(item.values)]
            if finite.size and (
                np.min(finite) < -tolerance
                or not np.allclose(finite, np.rint(finite), rtol=0.0, atol=tolerance)
            ):
                support_failures.append(item.label)
    directional_failures: list[str] = []
    for path in sorted(run_dir.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "n10" in archive.files and "n01" in archive.files:
                    n10 = np.asarray(archive["n10"], dtype=float)
                    n01 = np.asarray(archive["n01"], dtype=float)
                    if n10.shape != n01.shape or not np.allclose(
                        n10,
                        np.swapaxes(n01, -1, -2),
                        rtol=0.0,
                        atol=tolerance,
                        equal_nan=True,
                    ):
                        directional_failures.append(str(path))
        except (OSError, ValueError, zipfile.BadZipFile):
            continue
    state.add(
        "matrices_symmetric",
        symmetry_checked > 0 and not failures,
        {
            "matrices_checked": symmetry_checked,
            "failure_count": len(failures),
            "failures": failures[:20],
        },
    )
    state.add("matrices_ranges", not range_failures, {"failures": range_failures[:20]})
    state.add("support_integer_nonnegative", not support_failures, {"failures": support_failures[:20]})
    state.add("directional_n10_n01_transpose", not directional_failures, {"failures": directional_failures})
    return items


def matching_aggregate_key(by_key: str, keys: Iterable[str]) -> str | None:
    available = set(keys)
    candidates: list[str] = []
    if by_key.endswith("_by_scan"):
        base = by_key[: -len("_by_scan")]
        candidates.extend((f"{base}_aggregate", f"aggregate_{base}", base))
    if by_key.endswith("_by_frame"):
        base = by_key[: -len("_by_frame")]
        candidates.extend((f"{base}_aggregate", "aggregate", f"aggregate_{base}", base))
    if by_key == "matrices_by_scan":
        candidates.extend(("aggregate", "aggregate_matrix"))
    if by_key == "matrices_by_frame":
        candidates.extend(("aggregate", "aggregate_matrix"))
    substitutions = {
        "strict_acf_by_scan": "acf_strict_aggregate",
        "acf_strict_by_scan": "acf_strict_aggregate",
        "direct_by_scan": "direct_strict_aggregate",
        "direct_strict_by_scan": "direct_strict_aggregate",
        "shift_by_scan": "shift_tolerant_aggregate",
        "shift_tolerant_by_scan": "shift_tolerant_aggregate",
    }
    if by_key in substitutions:
        candidates.insert(0, substitutions[by_key])
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def nanmedian_without_warning(array: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(array, axis=axis)


def best_aggregate_recomputation(
    by_values: np.ndarray,
    aggregate: np.ndarray,
    *,
    reducer: str = "median",
) -> tuple[float, int | None, int, bool]:
    """Return max error, selected axis, compared cells and NaN-mask compatibility."""

    candidates = [axis for axis in range(by_values.ndim) if by_values.shape[:axis] + by_values.shape[axis + 1 :] == aggregate.shape]
    best: tuple[float, int | None, int, bool] | None = None
    for axis in candidates:
        if reducer == "mean":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                recomputed = np.nanmean(by_values, axis=axis)
        else:
            recomputed = nanmedian_without_warning(by_values, axis=axis)
        finite_aggregate = np.isfinite(aggregate)
        finite_recomputed = np.isfinite(recomputed)
        # An aggregate may intentionally mask finite medians for insufficient
        # support, but it may never invent a value where all inputs are NaN.
        mask_compatible = not bool(np.any(finite_aggregate & ~finite_recomputed))
        overlap = finite_aggregate & finite_recomputed
        compared = int(np.count_nonzero(overlap))
        difference = float(np.max(np.abs(aggregate[overlap] - recomputed[overlap]))) if compared else 0.0
        candidate = (difference, axis, compared, mask_compatible)
        if best is None or (not best[3], best[0]) > (not candidate[3], candidate[0]):
            best = candidate
    return best if best is not None else (float("inf"), None, 0, False)


def validate_aggregate_recomputation(run_dir: Path, state: ValidationState, tolerance: float) -> None:
    results: list[dict[str, Any]] = []
    missing_aggregates: list[str] = []
    for path in sorted(run_dir.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                by_keys = sorted(
                    key
                    for key in keys
                    if key.endswith("_by_scan")
                    and not any(token in key for token in ("availability", "available", "support"))
                )
                # The within-frame product contains both frame- and scan-level
                # arrays; its official aggregate is defined from independent
                # scans, so matrices_by_frame must not be used here.
                for by_key in by_keys:
                    by_values = np.asarray(archive[by_key])
                    if by_values.dtype.kind not in "biufc" or by_values.ndim < 3:
                        continue
                    aggregate_key = matching_aggregate_key(by_key, keys)
                    if aggregate_key is None:
                        missing_aggregates.append(f"{path}:{by_key}")
                        continue
                    aggregate = np.asarray(archive[aggregate_key], dtype=float)
                    difference, axis, compared, mask_compatible = best_aggregate_recomputation(
                        np.asarray(by_values, dtype=float),
                        aggregate,
                        reducer="mean" if by_key == "presence_by_scan" else "median",
                    )
                    results.append(
                        {
                            "file": str(path),
                            "by_key": by_key,
                            "aggregate_key": aggregate_key,
                            "scan_or_frame_axis": axis,
                            "compared_cells": compared,
                            "max_abs_difference": difference,
                            "mask_compatible": mask_compatible,
                            "passed": mask_compatible and difference <= tolerance,
                        }
                    )
        except (OSError, ValueError, zipfile.BadZipFile):
            continue
    state.add("aggregate_pairs_discovered", bool(results), {"count": len(results), "missing": missing_aggregates})
    state.add("aggregate_counterparts_present", not missing_aggregates, {"missing": missing_aggregates})
    state.add(
        "aggregate_recomputation_within_tolerance",
        bool(results) and all(result["passed"] for result in results),
        {"tolerance": tolerance, "results": results},
    )


def csv_scan_pair_columns(fieldnames: Sequence[str]) -> tuple[str, str] | None:
    normalized = {normalized_name(field): field for field in fieldnames}
    alias_pairs = (
        ("scan_i", "scan_j"),
        ("scan_a", "scan_b"),
        ("scan_left", "scan_right"),
        ("source_scan", "target_scan"),
        ("first_scan", "second_scan"),
    )
    for first, second in alias_pairs:
        if first in normalized and second in normalized:
            return normalized[first], normalized[second]
    return None


def validate_no_cross_scan_pairs(run_dir: Path, state: ValidationState) -> None:
    checked = 0
    violations: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*.csv")):
        try:
            fields, rows = read_csv_rows(path)
        except (OSError, UnicodeError, csv.Error):
            continue
        pair = csv_scan_pair_columns(fields)
        if pair is None:
            continue
        checked += 1
        left, right = pair
        for row_number, row in enumerate(rows, start=2):
            first = row.get(left, "").strip()
            second = row.get(right, "").strip()
            if first and second and first != second:
                violations.append({"file": str(path), "row": row_number, "left": first, "right": second})
                if len(violations) >= 20:
                    break
    for path in sorted(run_dir.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                for left, right in (("scan_i", "scan_j"), ("scan_a", "scan_b"), ("scan_left", "scan_right")):
                    if left not in archive.files or right not in archive.files:
                        continue
                    checked += 1
                    first = np.asarray(archive[left]).astype(str)
                    second = np.asarray(archive[right]).astype(str)
                    if first.shape != second.shape or np.any(first != second):
                        violations.append({"file": str(path), "keys": [left, right]})
        except (OSError, ValueError, zipfile.BadZipFile):
            continue
    # Matrix arrays indexed by a single scan axis are also auditable evidence
    # that comparisons were not formed between scans.
    if checked == 0:
        for path in run_dir.rglob("*.npz"):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if "scan_names" in archive.files and any(key.endswith("_by_scan") for key in archive.files):
                        checked += 1
            except (OSError, ValueError, zipfile.BadZipFile):
                continue
    state.add("same_scan_pairing_auditable", checked > 0, {"sources_checked": checked})
    state.add("no_cross_scan_pairing", not violations, {"violations": violations})


def support_threshold(n_available: np.ndarray) -> np.ndarray:
    n = np.asarray(n_available, dtype=float)
    threshold = np.maximum(5.0, np.ceil(0.1 * n))
    return np.minimum(n, threshold)


def validate_missing_semantics(run_dir: Path, state: ValidationState, tolerance: float) -> None:
    observation_checks = 0
    observation_violations: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*/per_peak/peak_observations.csv")):
        try:
            fields, rows = read_csv_rows(path)
        except (OSError, UnicodeError, csv.Error):
            continue
        normalized = {normalized_name(field): field for field in fields}
        present_field = next((normalized[name] for name in ("reliable", "present", "is_reliable", "is_present") if name in normalized), None)
        status_field = next((normalized[name] for name in ("state", "status", "observation_state") if name in normalized), None)
        feature_fields = [
            normalized[name]
            for name in ("relative_area", "q", "q_center", "center_q", "fwhm_q")
            if name in normalized
        ]
        if not feature_fields or (present_field is None and status_field is None):
            continue
        observation_checks += 1
        for row_number, row in enumerate(rows, start=2):
            reliable = parse_bool(row.get(present_field)) if present_field else None
            if reliable is None and status_field:
                reliable = parse_bool(row.get(status_field))
            if reliable is not False:
                continue
            finite_features: list[str] = []
            for field in feature_fields:
                text = row.get(field, "").strip()
                if not text or text.lower() in {"nan", "na", "n/a", "null", "none"}:
                    continue
                try:
                    if math.isfinite(float(text)):
                        finite_features.append(field)
                except ValueError:
                    pass
            if finite_features:
                observation_violations.append(
                    {"file": str(path), "row": row_number, "finite_features": finite_features}
                )
                if len(observation_violations) >= 20:
                    break
    if observation_checks:
        state.add(
            "missing_observations_have_no_correlation_features",
            not observation_violations,
            {"tables": observation_checks, "violations": observation_violations},
        )
    else:
        state.add("missing_observation_tables_optional", True, "No compatible observation table was available")

    count_checks = 0
    count_failures: list[dict[str, Any]] = []
    mask_failures: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*/per_peak/per_peak_matrices.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                required_counts = {"n_available", "n_both_present", "n10", "n01", "n_unknown"}
                if not required_counts.issubset(keys):
                    continue
                count_checks += 1
                counts = {key: np.asarray(archive[key], dtype=float) for key in required_counts}
                shapes = {array.shape for array in counts.values()}
                if len(shapes) != 1:
                    count_failures.append({"file": str(path), "reason": "count shapes differ"})
                    continue
                n_available = counts["n_available"]
                accounted = counts["n_both_present"] + counts["n10"] + counts["n01"]
                if np.any(accounted > n_available + tolerance):
                    count_failures.append({"file": str(path), "reason": "present/mismatch counts exceed n_available"})
                if "required_support" in keys:
                    actual_required = np.asarray(archive["required_support"], dtype=float)
                    expected_required = support_threshold(n_available)
                    if actual_required.shape != expected_required.shape or not np.allclose(
                        actual_required,
                        expected_required,
                        rtol=0.0,
                        atol=tolerance,
                        equal_nan=False,
                    ):
                        count_failures.append({"file": str(path), "reason": "required_support does not follow S(N)"})
                denominator = counts["n_both_present"] + counts["n10"] + counts["n01"]
                expected_presence = np.divide(
                    counts["n_both_present"],
                    denominator,
                    out=np.full_like(denominator, np.nan),
                    where=denominator > 0,
                )
                presence_key = "presence_aggregate" if "presence_aggregate" in keys else (
                    "presence" if "presence" in keys else None
                )
                if presence_key is not None:
                    actual = np.asarray(archive[presence_key], dtype=float)
                    if actual.shape != expected_presence.shape:
                        count_failures.append({"file": str(path), "reason": "presence is not Jaccard from counts"})
                    else:
                        comparable = np.isfinite(expected_presence) | np.isfinite(actual)
                        if not np.allclose(
                            actual[comparable],
                            expected_presence[comparable],
                            rtol=0.0,
                            atol=tolerance,
                            equal_nan=True,
                        ):
                            count_failures.append({"file": str(path), "reason": "presence is not Jaccard from counts"})
                threshold = support_threshold(n_available)
                insufficient = counts["n_both_present"] < threshold
                for family in ("area", "location"):
                    key = f"{family}_aggregate" if f"{family}_aggregate" in keys else (
                        family if family in keys else None
                    )
                    if key is None:
                        continue
                    matrix = np.asarray(archive[key], dtype=float)
                    if matrix.shape != insufficient.shape or np.any(np.isfinite(matrix) & insufficient):
                        mask_failures.append({"file": str(path), "key": key})
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            count_failures.append({"file": str(path), "reason": str(error)})
    state.add("missing_count_arrays_present", count_checks > 0, {"files": count_checks})
    state.add("missing_presence_count_semantics", count_checks > 0 and not count_failures, {"failures": count_failures})
    state.add("missing_insufficient_support_is_nan", count_checks > 0 and not mask_failures, {"failures": mask_failures})


def resolve_report_reference(run_dir: Path, report_path: Path, reference: str) -> Path | None:
    reference = reference.strip().strip("<>")
    reference = reference.split("#", 1)[0].strip()
    if not reference or reference.startswith(("http://", "https://", "mailto:", "#")):
        return None
    # Markdown titles follow the destination after whitespace.  Paths used by
    # this project do not contain unescaped spaces.
    reference = reference.split()[0]
    candidate = Path(reference).expanduser()
    if candidate.name == "RUN_COMPLETE.json":
        return None
    if candidate.is_absolute():
        return candidate
    from_report = (report_path.parent / candidate).resolve()
    if from_report.exists():
        return from_report
    return (run_dir / candidate).resolve()


def validate_report_references(run_dir: Path, state: ValidationState) -> None:
    references: list[tuple[str, Path]] = []
    report_path = run_dir / "REPORT.md"
    if report_path.is_file():
        text = report_path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            resolved = resolve_report_reference(run_dir, report_path, match.group(1))
            if resolved is not None:
                references.append((match.group(1), resolved))

    index_path = run_dir / "artifact_index.csv"
    if index_path.is_file():
        try:
            fields, rows = read_csv_rows(index_path)
        except (OSError, UnicodeError, csv.Error):
            fields, rows = [], []
        path_fields = [
            field
            for field in fields
            if normalized_name(field) in {"path", "file", "file_path", "artifact", "artifact_path", "relative_path"}
        ]
        for row in rows:
            for field in path_fields:
                raw = row.get(field, "").strip()
                if raw:
                    resolved = resolve_report_reference(run_dir, report_path, raw)
                    if resolved is not None:
                        references.append((raw, resolved))
    missing = [{"reference": raw, "resolved": str(path)} for raw, path in references if not path.exists()]
    state.add("report_references_discovered", bool(references), {"count": len(references)})
    state.add("report_references_exist", not missing, {"missing": missing[:50], "count": len(missing)})


def workbook_formula_errors(path: Path) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not path.is_file() or not zipfile.is_zipfile(path):
        return [{"file": str(path), "error": "not a readable XLSX zip archive"}]
    try:
        with zipfile.ZipFile(path) as archive:
            worksheet_names = sorted(
                name for name in archive.namelist() if name.startswith("xl/worksheets/") and name.endswith(".xml")
            )
            for name in worksheet_names:
                root = ElementTree.fromstring(archive.read(name))
                for cell in root.findall(".//{*}c"):
                    address = cell.attrib.get("r", "?")
                    formula = cell.find("{*}f")
                    value = cell.find("{*}v")
                    formula_text = formula.text or "" if formula is not None else ""
                    value_text = value.text or "" if value is not None else ""
                    token = next(
                        (item for item in FORMULA_ERROR_TOKENS if item in formula_text.upper()),
                        None,
                    )
                    if cell.attrib.get("t") == "e":
                        token = value_text or token or "formula error"
                    if token:
                        errors.append({"sheet_xml": name, "cell": address, "error": token})
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as error:
        errors.append({"file": str(path), "error": str(error)})
    return errors


def validate_workbook(run_dir: Path, state: ValidationState) -> None:
    workbooks = sorted(run_dir.glob("*.xlsx"))
    state.add("workbook_single_report_present", len(workbooks) == 1, {"workbooks": [path.name for path in workbooks]})
    if not workbooks:
        return
    errors = workbook_formula_errors(workbooks[0])
    state.add("workbook_no_formula_errors", not errors, {"errors": errors[:50]})


def validate_artifact_hashes(run_dir: Path, state: ValidationState) -> None:
    index_path = run_dir / "artifact_index.csv"
    if not index_path.is_file():
        return
    try:
        fields, rows = read_csv_rows(index_path)
    except (OSError, UnicodeError, csv.Error) as error:
        state.add("artifact_index_readable", False, str(error))
        return
    normalized = {normalized_name(field): field for field in fields}
    path_field = next((normalized[name] for name in ("path", "file_path", "artifact_path", "relative_path") if name in normalized), None)
    hash_field = next((normalized[name] for name in ("sha256", "artifact_sha256", "file_sha256") if name in normalized), None)
    state.add("artifact_index_readable", bool(fields) and bool(rows), {"rows": len(rows)})
    if path_field is None or hash_field is None:
        state.add("artifact_index_hash_columns_optional", True, "No path/hash column pair; existence checked separately")
        return
    failures: list[dict[str, str]] = []
    checked = 0
    for row in rows:
        raw = row.get(path_field, "").strip()
        expected = row.get(hash_field, "").strip().lower()
        if not raw or not expected:
            continue
        checked += 1
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = run_dir / target
        if not target.is_file():
            failures.append({"path": raw, "reason": "missing"})
        elif not SHA256_RE.fullmatch(expected):
            failures.append({"path": raw, "reason": "malformed sha256"})
        elif sha256_file(target) != expected:
            failures.append({"path": raw, "reason": "sha256 mismatch"})
    state.add(
        "artifact_index_hashes_match",
        checked > 0 and not failures,
        {
            "artifacts_checked": checked,
            "failure_count": len(failures),
            "failures": failures[:50],
        },
    )


def validate_run(run_dir: Path | str, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """Run all validations without writing files and return a JSON-ready report."""

    root = Path(run_dir).expanduser().resolve()
    state = ValidationState()
    state.add("run_directory_exists", root.is_dir(), str(root))
    if root.is_dir():
        validate_structure(root, state)
        validate_manifests(root, state)
        validate_official_acceptance_evidence(root, state, tolerance)
        validate_v21_tracking_audits(root, state)
        validate_matrices(root, state, tolerance)
        validate_aggregate_recomputation(root, state, tolerance)
        validate_no_cross_scan_pairs(root, state)
        validate_missing_semantics(root, state, tolerance)
        validate_report_references(root, state)
        validate_workbook(root, state)
        validate_artifact_hashes(root, state)
    return {
        "validator": "validate_uniform_xy_correlations-v2.1",
        "profile": PROFILE_ID,
        "run_dir": str(root),
        "tolerance": tolerance,
        "checks": state.checks,
        "details": state.details,
        "errors": state.errors,
        "passed": bool(state.checks) and all(state.checks.values()),
    }


def write_validation_outputs(run_dir: Path | str, report: Mapping[str, Any]) -> tuple[Path, Path | None]:
    """Write the validation report and, only on success, completion marker."""

    root = Path(run_dir).expanduser().resolve()
    validation_dir = root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    report_path = validation_dir / "validation_report.json"
    report_path.write_bytes(canonical_json_bytes(report))

    marker_path = root / "RUN_COMPLETE.json"
    if not bool(report.get("passed")):
        if marker_path.exists():
            marker_path.unlink()
        return report_path, None

    manifest_path = root / "run_manifest.json"
    config_path = root / "algorithm_config.json"
    marker = {
        "profile": PROFILE_ID,
        "status": "complete",
        "all_validation_checks_passed": True,
        "check_count": len(report.get("checks", {})),
        "algorithm_config_sha256": sha256_file(config_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "validation_report_sha256": sha256_file(report_path),
    }
    temporary = marker_path.with_suffix(".json.tmp")
    temporary.write_bytes(canonical_json_bytes(marker))
    temporary.replace(marker_path)
    return report_path, marker_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed uniform-v2.1 result directory")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Validate and print the report without writing validation_report.json or RUN_COMPLETE.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_run(args.run_dir, tolerance=args.tolerance)
    if not args.no_write:
        report_path, marker_path = write_validation_outputs(args.run_dir, report)
        report = dict(report)
        report["validation_report_path"] = str(report_path)
        report["completion_marker_path"] = str(marker_path) if marker_path else None
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
