"""Worker subprocess for risky pyFAI/matplotlib generation.

The GUI calls this script instead of running pyFAI inside the Tk process.
If pyFAI or a plotting backend hard-crashes, the GUI survives and reports the
worker return code/log.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import json
import traceback

# Prepend conda Library/bin to PATH on Windows before any pyFAI import.
# Required when the worker is launched without conda activate (direct python call).
if sys.platform.startswith("win"):
    _prefix = Path(sys.executable).parent
    for _sub in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin"):
        _d = str(_prefix / _sub)
        if Path(_d).is_dir() and _d.lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

# Works both as a package module (python -m seriesxrd.calib.worker) and as a
# directly-launched script (the GUI runs this file by path in a subprocess).
if __package__ in (None, ""):
    _pkg_parent = str(Path(__file__).resolve().parents[2])
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    from seriesxrd.core.config import read_json, write_json, print_status
    from seriesxrd.calib.processing import generate_qa_run, preview_cake_orientations
else:
    from ..core.config import read_json, write_json, print_status
    from .processing import generate_qa_run, preview_cake_orientations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--mode", default="qa", choices=["qa", "preview"])
    args = parser.parse_args()
    try:
        cfg = read_json(args.config)
        if args.mode == "preview":
            print_status("Worker starting cake orientation preview")
            result = preview_cake_orientations(cfg)
            write_json(args.output_json, result)
            print_status(f"Worker preview complete -> {args.output_json}")
            return 0
        print_status(f"Worker starting QA generation gen{args.generation:03d}")
        md = generate_qa_run(cfg, args.generation)
        write_json(args.output_json, md)
        print_status(f"Worker completed {md.get('generation')} -> {args.output_json}")
        return 0
    except Exception as e:
        print_status("Worker failed: " + repr(e), "ERROR")
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
