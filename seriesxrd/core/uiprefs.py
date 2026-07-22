"""Per-user UI preferences, deliberately separate from scientific sessions."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Mapping

from .config import read_json, write_json


DEFAULT_PREFS: dict[str, Any] = {"theme": "mocha"}


def prefs_path() -> Path:
    """Return the platform-native SeriesXRD UI-preference file path."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or
                    (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or
                    (Path.home() / ".config"))
    return base / "seriesxrd" / "ui.json"


def load_prefs(path: "str | Path | None" = None) -> dict[str, Any]:
    """Load preferences with stable defaults for missing/corrupt files."""
    result = dict(DEFAULT_PREFS)
    result.update(read_json(Path(path) if path is not None else prefs_path()))
    if result.get("theme") not in {"mocha", "latte"}:
        result["theme"] = DEFAULT_PREFS["theme"]
    return result


def save_prefs(
    values: Mapping[str, Any], path: "str | Path | None" = None,
) -> Path:
    """Merge and atomically save per-user UI preferences."""
    target = Path(path) if path is not None else prefs_path()
    prefs = load_prefs(target)
    prefs.update(dict(values))
    return write_json(target, prefs)

