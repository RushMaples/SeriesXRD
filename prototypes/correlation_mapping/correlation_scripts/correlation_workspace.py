#!/usr/bin/env python3
"""Navigate and verify the organized UOTe correlation-code workspace.

This is a read-only front door. It does not replace the scientific programs;
it identifies the active entrypoints, checks the code catalog/inventory, and
reports the completion state of the current formal result roots.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
CORRELATIONS_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CORRELATIONS_ROOT.parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "uote-formal-qwidth075.json"
CATALOG_PATH = SCRIPT_DIR / "CODE_CATALOG.json"
INVENTORY_PATH = SCRIPT_DIR / "CODE_INVENTORY.csv"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_correlations_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    results_override = os.environ.get("CORRELATION_RESULTS_ROOT")
    if results_override and path.parts and path.parts[0] == "results":
        return Path(results_override).expanduser() / Path(*path.parts[1:])
    return CORRELATIONS_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "uotexrd-correlation-workspace-v3":
        raise ValueError(f"unsupported workspace config: {path}")
    return config


def catalog_groups() -> list[dict[str, Any]]:
    payload = read_json(CATALOG_PATH)
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"invalid code catalog: {CATALOG_PATH}")
    return groups


def command_catalog(args: argparse.Namespace) -> int:
    groups = catalog_groups()
    selected = [item for item in groups if args.group in (None, item["id"])]
    if not selected:
        raise ValueError(f"unknown catalog group: {args.group}")
    for group in selected:
        print(f"[{group['status']}] {group['id']} — {group['label']}")
        for name in group["files"]:
            print(f"  {name}")
    return 0


def _same_path(reported: Any, expected: Path) -> bool:
    if not isinstance(reported, str):
        return False
    candidate = Path(reported)
    if not candidate.is_absolute():
        candidate = CORRELATIONS_ROOT / candidate
    return candidate.resolve() == expected.resolve()


def _expected_primary_counts(config: dict[str, Any]) -> dict[str, int]:
    counts = config["expected_counts_per_transform"]
    powder = counts["powder"]
    single = counts["single_crystal"]
    return {
        "powder_roi_area": powder["roi_area"],
        "powder_location": powder["location"],
        "single_crystal_roi_area": single["roi_area"],
        "single_crystal_location": single["location"],
        "powder_across": powder["window_to_window_across_frames"],
        "single_crystal_across": single["window_to_window_across_frames"],
        "powder_within": powder["window_to_window_within_same_frame"],
        "single_crystal_within": single["window_to_window_within_same_frame"],
        "total_window_matrices": (
            powder["window_to_window_across_frames"]
            + powder["window_to_window_within_same_frame"]
            + single["window_to_window_across_frames"]
            + single["window_to_window_within_same_frame"]
        ),
    }


def _formal_completion_matches(value: dict[str, Any], config: dict[str, Any]) -> bool:
    counts = config["expected_counts_per_transform"]
    expected_science_files = 2 * sum(
        category_count
        for sample_counts in counts.values()
        for category_count in sample_counts.values()
    )
    return (
        value.get("status") == "complete"
        and str(value.get("transform_label", "")).startswith("log_squared")
        and value.get("science_files") == expected_science_files
        and value.get("all_science_files_same_inode_as_source") is True
    )


def _formal_validation_matches(value: dict[str, Any], config: dict[str, Any]) -> bool:
    result_paths = config["validated_results"]
    expected_roots = {
        "log_square": resolve_correlations_path(result_paths["all_peak_formal_suite"]),
        "baseline": resolve_correlations_path(result_paths["baseline_package"]),
    }
    expected_counts = _expected_primary_counts(config)
    suites = value.get("suites", {})
    report_roots = value.get("roots", {})
    checks = value.get("checks", {})
    return (
        value.get("status") == "PASS"
        and value.get("formal_expected_counts_per_transform") == expected_counts
        and bool(checks)
        and all(checks.values())
        and set(suites) == {"log_square"}
        and all(suites[name].get("status") == "PASS" for name in suites)
        and all(
            _same_path(report_roots.get(name), expected)
            for name, expected in expected_roots.items()
        )
    )


def _waterfall_validation_matches(
    value: dict[str, Any], config: dict[str, Any], *, original_profile: bool
) -> bool:
    result_paths = config["validated_results"]
    parameters = config["parameters"]
    scope = config["powder_scope"]
    anchors = config["expected_counts_per_transform"]["powder"]["roi_area"]
    modes = ["log_squared"] if original_profile else parameters["transform_modes"]
    domain = "original_positive" if original_profile else "correlation_transform"
    suite_key = (
        "log_original_profile_waterfalls" if original_profile else "transformed_waterfalls"
    )
    groups = value.get("groups", [])
    groups_by_mode = {
        group.get("mode"): group for group in groups if isinstance(group, dict)
    }
    group_scope_matches = len(groups) == len(modes) and set(groups_by_mode) == set(modes)
    for mode in modes:
        group = groups_by_mode.get(mode, {})
        reconstruction = group.get("display_profile_reconstruction", {})
        group_scope_matches = group_scope_matches and all(
            (
                group.get("status") == "PASS",
                group.get("sample") == "powder",
                group.get("display_profile_domain") == domain,
                group.get("half_width_factor")
                == parameters["powder_half_width_factor"],
                group.get("anchors") == anchors,
                group.get("formal_entities_per_anchor") == anchors,
                group.get("pngs") == anchors,
                group.get("cross_pressure_cells")
                == scope["directed_cross_pressure_cells"],
                group.get("positive_cross_pressure_cells")
                == scope["positive_roi_cells"],
                group.get("zero_cross_pressure_cells") == scope["exact_zero_roi_cells"],
                group.get("matrix_mapping_score_value_mismatches") == 0,
                group.get("support_bound_mismatches") == 0,
                group.get("strictly_nonoverlapping") is True,
                reconstruction.get("half_width_factor")
                == parameters["powder_half_width_factor"],
            )
        )
    expected_waterfalls = anchors * len(modes)
    return (
        value.get("status") == "PASS"
        and _same_path(
            value.get("comparison_root"),
            resolve_correlations_path(result_paths["powder_and_window_source_root"]),
        )
        and _same_path(
            value.get("suite_root"), resolve_correlations_path(result_paths[suite_key])
        )
        and value.get("modes") == modes
        and value.get("samples") == ["powder"]
        and value.get("expected_waterfalls") == expected_waterfalls
        and value.get("waterfalls") == expected_waterfalls
        and value.get("unique_png_paths") == expected_waterfalls
        and value.get("all_anchor_validations_pass") is True
        and value.get("all_png_files_verified") is True
        and value.get("all_sha256_recorded") is True
        and value.get("all_trace_and_ribbon_bands_nonoverlapping") is True
        and group_scope_matches
    )


def _marker_check(
    label: str,
    raw_path: str,
    predicate: Any,
) -> dict[str, Any]:
    path = resolve_correlations_path(raw_path)
    exists = path.is_file()
    value: dict[str, Any] = {}
    error: str | None = None
    if exists:
        try:
            value = read_json(path)
        except (OSError, ValueError, TypeError) as exc:
            error = str(exc)
    passed = False
    if exists and error is None:
        try:
            passed = bool(predicate(value))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            error = f"scope validation failed: {exc}"
    return {
        "label": label,
        "path": str(path),
        "exists": exists,
        "passed": passed,
        "reported_status": value.get("status"),
        "error": error,
    }


def _configured_path_check(
    label: str, configured: dict[str, str]
) -> dict[str, Any]:
    resolved = {name: resolve_correlations_path(path) for name, path in configured.items()}
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    return {
        "label": label,
        "path": str(CORRELATIONS_ROOT),
        "exists": not missing,
        "passed": not missing,
        "reported_status": "configured",
        "missing_paths": missing,
    }


def _completion_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    result_paths = config["validated_results"]
    return [
        _configured_path_check("configured active entrypoints", config["active_entrypoints"]),
        _configured_path_check(
            "required compatibility dependencies", config["required_legacy_dependencies"]
        ),
        _marker_check(
            "formal Log² all-peak package validation",
            result_paths["all_peak_formal_validation"],
            lambda value: value.get("status") == "PASS",
        ),
        _marker_check(
            "single-crystal 275-anchor Log² run",
            str(Path(result_paths["single_crystal_all_peak_log_suite"]) / "RUN_COMPLETE.json"),
            lambda value: value.get("status") == "PASS"
            and value.get("downstream_analysis", {}).get("peaks") == 275,
        ),
        _marker_check(
            "single-crystal original-XY waterfalls (Log² colors)",
            str(Path(result_paths["single_crystal_original_xy_waterfalls"]) / "SUITE_VALIDATION.json"),
            lambda value: value.get("status") == "PASS"
            and value.get("anchor_maps") == 275,
        ),
        _marker_check(
            "original-profile powder waterfalls (Log colors)",
            result_paths["log_original_profile_validation"],
            lambda value: _waterfall_validation_matches(
                value, config, original_profile=True
            ),
        ),
    ]


def command_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    checks = _completion_checks(config)
    payload = {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "config": str(args.config.resolve()),
        "active_parameters": config["parameters"],
        "checks": checks,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Current workflow: {config['description']}")
        print(
            "Powder support: "
            f"{config['parameters']['powder_support_formula']}"
        )
        for item in checks:
            marker = "PASS" if item["passed"] else "FAIL"
            print(f"[{marker}] {item['label']}: {item['path']}")
        print(f"Overall: {payload['status']}")
    return 0 if payload["status"] == "PASS" else 1


def _inventory_rows() -> list[dict[str, str]]:
    if not INVENTORY_PATH.is_file():
        return []
    with INVENTORY_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def local_imports(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def command_check_code(args: argparse.Namespace) -> int:
    groups = catalog_groups()
    catalog_files = [name for group in groups for name in group["files"]]
    actual_paths = sorted(SCRIPT_DIR.glob("*.py"))
    actual_files = [path.name for path in actual_paths]
    duplicate_catalog_entries = sorted(
        {name for name in catalog_files if catalog_files.count(name) > 1}
    )
    missing_from_catalog = sorted(set(actual_files) - set(catalog_files))
    missing_from_disk = sorted(set(catalog_files) - set(actual_files))

    parse_errors: list[str] = []
    unresolved_local_imports: list[str] = []
    local_modules = {path.stem for path in actual_paths}
    for path in actual_paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            parse_errors.append(f"{path.name}: {exc}")
            continue
        for module in local_imports(tree):
            if module.startswith("correlation_") and module not in local_modules:
                unresolved_local_imports.append(f"{path.name}: {module}")

    inventory_rows = _inventory_rows()
    inventory_missing: list[str] = []
    inventory_hash_mismatches: list[str] = []
    for row in inventory_rows:
        path = SCRIPT_DIR / row["path"]
        if not path.is_file():
            inventory_missing.append(row["path"])
        elif sha256(path) != row["sha256"]:
            inventory_hash_mismatches.append(row["path"])

    passed = not any(
        (
            duplicate_catalog_entries,
            missing_from_catalog,
            missing_from_disk,
            parse_errors,
            unresolved_local_imports,
            inventory_missing,
            inventory_hash_mismatches,
        )
    ) and bool(inventory_rows)
    payload = {
        "status": "PASS" if passed else "FAIL",
        "python_files": len(actual_files),
        "catalog_entries": len(catalog_files),
        "catalog_groups": len(groups),
        "inventory_entries": len(inventory_rows),
        "duplicate_catalog_entries": duplicate_catalog_entries,
        "missing_from_catalog": missing_from_catalog,
        "missing_from_disk": missing_from_disk,
        "parse_errors": parse_errors,
        "unresolved_local_imports": unresolved_local_imports,
        "inventory_missing": inventory_missing,
        "inventory_hash_mismatches": inventory_hash_mismatches,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


def command_commands(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result_paths = config["validated_results"]
    comparison = resolve_correlations_path(result_paths["powder_and_window_source_root"])
    original = resolve_correlations_path(
        result_paths["log_original_profile_waterfalls"]
    )
    package_validation = resolve_correlations_path(result_paths["all_peak_formal_validation"])
    commands = [
        [
            sys.executable,
            str(SCRIPT_DIR / "validate_package_denoised_correlation_suites.py"),
            "--log-root",
            str(resolve_correlations_path(result_paths["all_peak_formal_suite"])),
            "--baseline-root",
            str(resolve_correlations_path(result_paths["baseline_package"])),
            "--output-dir",
            str(package_validation.parent),
            "--dry-run",
        ],
        [
            sys.executable,
            str(SCRIPT_DIR / "validate_complete_formal_composite_waterfalls.py"),
            "--comparison-root",
            str(comparison),
            "--suite-root",
            str(original),
            "--powder-only",
            "--modes",
            "log_squared",
        ],
        [
            sys.executable,
            str(SCRIPT_DIR / "correlation_workspace.py"),
            "--config",
            str(args.config.resolve()),
            "status",
        ],
        [
            sys.executable,
            str(SCRIPT_DIR / "correlation_workspace.py"),
            "check-code",
        ],
    ]
    for command in commands:
        print(shlex.join(command))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="List categorized code files.")
    catalog.add_argument("--group", help="Optional catalog group id.")
    catalog.set_defaults(handler=command_catalog)

    status = subparsers.add_parser(
        "status", help="Check completion markers for the current formal results."
    )
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=command_status)

    check = subparsers.add_parser(
        "check-code", help="Verify catalog coverage, syntax, and SHA inventory."
    )
    check.set_defaults(handler=command_check_code)

    commands = subparsers.add_parser(
        "commands",
        help="Print resolved validation commands; validators refresh reports.",
    )
    commands.set_defaults(handler=command_commands)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
