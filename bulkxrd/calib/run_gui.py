"""CLI entry point for the calibration review GUI.

Runnable three ways:
    bulkxrd-calib-gui --config <path>             (console script, pip install)
    python -m bulkxrd.calib.run_gui --config <path>
    python bulkxrd/calib/run_gui.py --config <path>
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# Works both as a package module and as a directly-launched script.
if __package__ in (None, ""):
    _pkg_parent = str(THIS_DIR.parents[1])
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    from bulkxrd.core.config import print_status
    from bulkxrd.calib.gui import run_app
else:
    from ..core.config import print_status
    from .gui import run_app


_CONFIG_SEARCH_PATHS = [
    # Workspace in the current working directory
    Path.cwd() / "calibration_session_config.json",
    # Same folder as this script (for standalone testing)
    THIS_DIR / "calibration_session_config.json",
]


def _auto_find_config() -> Path | None:
    for p in _CONFIG_SEARCH_PATHS:
        try:
            if p.exists():
                return p.resolve()
        except Exception:
            pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Bulk-XRD Calibration Review GUI")
    parser.add_argument("--config", default="", help="Path to calibration_session_config.json (optional — auto-found if omitted)")
    args = parser.parse_args()

    if args.config:
        cfg = Path(args.config).expanduser().resolve()
        if not cfg.exists():
            raise FileNotFoundError(f"Session config not found: {cfg}")
    else:
        cfg = _auto_find_config()
        if cfg is None:
            print("[ERROR] Could not auto-find calibration_session_config.json.", flush=True)
            print("Searched:", flush=True)
            for p in _CONFIG_SEARCH_PATHS:
                print(f"  {p}", flush=True)
            print("", flush=True)
            print("Run with:  python -m bulkxrd.calib.run_gui --config <path_to_config>", flush=True)
            return 1
        print_status(f"Auto-found config: {cfg}")

    print_status(f"Runner started with config {cfg}")
    rc = run_app(cfg)
    print_status(f"Runner finished with return code {rc}")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
