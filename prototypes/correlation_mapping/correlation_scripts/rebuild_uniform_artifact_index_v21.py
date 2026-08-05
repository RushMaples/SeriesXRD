#!/usr/bin/env python3
"""Rebuild a v2.1 run's stable artifact index.

The index intentionally omits files whose contents are written only after the
index is validated (the final validation report and completion marker), plus
Finder metadata and the workbook itself.  This avoids circular self-hashes
while keeping every scientific CSV, NPZ, plot, audit, and preview indexed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uniform_correlation_io import build_artifact_index, write_rows_csv


STATIC_EXCLUDES = {
    ".DS_Store",
    "artifact_index.csv",
    "RUN_COMPLETE.json",
    "validation/validation_report.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Run directory does not exist: {root}")

    excludes = set(STATIC_EXCLUDES)
    excludes.update(path.name for path in root.glob("*.xlsx"))
    rows = build_artifact_index(root, exclude=excludes)
    write_rows_csv(root / "artifact_index.csv", rows)
    print(f"indexed_artifacts={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
