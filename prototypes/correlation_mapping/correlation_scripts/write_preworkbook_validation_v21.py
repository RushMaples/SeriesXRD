#!/usr/bin/env python3
"""Write a validation report for every check that can run before XLSX creation.

This stage exists only to break the intentional dependency cycle: the workbook
must show validated inputs, while the final validator must also inspect the
finished workbook.  It never writes ``RUN_COMPLETE.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_uniform_xy_correlations_v21 import validate_run


WORKBOOK_ONLY_CHECKS = {
    "structure_workbook",
    "workbook_single_report_present",
    "workbook_no_formula_errors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    full = validate_run(root, tolerance=args.tolerance)
    checks = {
        name: passed
        for name, passed in full.get("checks", {}).items()
        if name not in WORKBOOK_ONLY_CHECKS
    }
    details = {
        name: detail
        for name, detail in full.get("details", {}).items()
        if name not in WORKBOOK_ONLY_CHECKS
    }
    errors = [
        error
        for error in full.get("errors", [])
        if str(error).split(":", 1)[0] not in WORKBOOK_ONLY_CHECKS
    ]
    report = {
        "validator": "validate_uniform_xy_correlations-v2.1-pre-workbook",
        "profile": "uniform-correlation-v2.1",
        "stage": "pre-workbook",
        "run_dir": str(root),
        "tolerance": args.tolerance,
        "checks": checks,
        "details": details,
        "errors": errors,
        "passed": bool(checks) and all(checks.values()) and not errors,
        "excluded_until_final_stage": sorted(WORKBOOK_ONLY_CHECKS),
    }
    output = root / "validation" / "pre_workbook_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
