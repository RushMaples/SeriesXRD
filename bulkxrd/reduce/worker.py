"""Worker subprocess for batch reduction.

The GUI/notebook launches this instead of running pyFAI in-process, mirroring
calib/worker.py: if pyFAI hard-crashes, the supervisor survives and reports
the return code and log.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import traceback

# Prepend conda Library/bin to PATH on Windows before any pyFAI import.
if sys.platform.startswith("win"):
    _prefix = Path(sys.executable).parent
    for _sub in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin"):
        _d = str(_prefix / _sub)
        if Path(_d).is_dir() and _d.lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

# Works both as a package module (python -m bulkxrd.reduce.worker) and as a
# directly-launched script (the GUI runs this file by path in a subprocess).
if __package__ in (None, ""):
    _pkg_parent = str(Path(__file__).resolve().parents[2])
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    from bulkxrd.core.config import read_json, write_json, print_status
    from bulkxrd.reduce.processing import reduce_dataset
else:
    from ..core.config import read_json, write_json, print_status
    from .processing import reduce_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    try:
        cfg = read_json(args.config)
        print_status("Reduction worker starting")
        manifest = reduce_dataset(cfg)
        write_json(args.output_json, manifest)
        print_status(f"Reduction worker completed -> {args.output_json}")
        return 0
    except Exception as e:
        print_status("Reduction worker failed: " + repr(e), "ERROR")
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
