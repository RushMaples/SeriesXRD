#!/usr/bin/env python3
"""Verify frozen v2 code/config and indexed results against a pre-v2.1 baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from uniform_correlation_io import directory_sha256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    file_failures: list[dict[str, str]] = []
    for relative, expected in baseline.get("v2_scientific_files", {}).items():
        target = workspace / relative
        if not target.is_file():
            file_failures.append({"path": relative, "reason": "missing"})
        else:
            actual = sha256(target)
            if actual != expected:
                file_failures.append(
                    {"path": relative, "reason": "sha256_mismatch", "expected": expected, "actual": actual}
                )

    result_key = "correlations/results/uote_xy_handoff2_correlations_uniform_v2_20260714"
    result_baseline = baseline.get("result_directories", {}).get(result_key, {})
    result_root = workspace / result_key
    count, digest = directory_sha256(result_root)
    full_tree_match = (
        count == int(result_baseline.get("file_count", -1))
        and digest == result_baseline.get("directory_sha256")
    )

    indexed_failures: list[dict[str, str]] = []
    index_path = result_root / "artifact_index.csv"
    indexed_count = 0
    if index_path.is_file():
        with index_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        indexed_count = len(rows)
        for row in rows:
            relative = row.get("relative_path", row.get("path", "")).strip()
            expected = row.get("sha256", "").strip().lower()
            target = result_root / relative
            if not relative or not target.is_file():
                indexed_failures.append({"path": relative, "reason": "missing"})
            elif sha256(target) != expected:
                indexed_failures.append({"path": relative, "reason": "sha256_mismatch"})
    else:
        indexed_failures.append({"path": str(index_path), "reason": "index_missing"})
    indexed_match = (
        not indexed_failures
        and indexed_count == int(result_baseline.get("indexed_artifact_count", -1))
        and sha256(index_path) == result_baseline.get("artifact_index_sha256")
    )

    passed = not file_failures and full_tree_match and indexed_match
    report = {
        "validator": "verify_uniform_v2_immutable-v1",
        "profile": "uniform-correlation-v2.1",
        "baseline": str(baseline_path),
        "v2_root": str(result_root),
        "passed": passed,
        "unchanged": passed,
        "files_before": result_baseline.get("file_count"),
        "files_after": count,
        "sha256_before": result_baseline.get("directory_sha256"),
        "sha256_after": digest,
        "checks": {
            "v2_code_and_config_hashes_unchanged": not file_failures,
            "v2_full_result_tree_unchanged": full_tree_match,
            "v2_indexed_artifacts_unchanged": indexed_match,
        },
        "details": {
            "v2_scientific_file_failures": file_failures,
            "indexed_artifact_count": indexed_count,
            "indexed_artifact_failures": indexed_failures[:50],
        },
        "errors": [item["path"] for item in file_failures + indexed_failures]
        + ([] if full_tree_match else ["v2 full result tree changed"]),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
