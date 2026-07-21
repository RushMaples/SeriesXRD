"""Probe for a usable Tk display without risking the test process.

On headless Linux, Tk raises a catchable TclError. On headless macOS CI
runners, Tk (9.0, shipped with python.org 3.14) can SEGFAULT inside
TkpInit instead — an except clause never sees it and the whole pytest
process dies. So the probe constructs its Tk in a throwaway subprocess,
where a native crash is just a nonzero exit code.
"""
from __future__ import annotations

import subprocess
import sys

_PROBE = ("import tkinter\n"
          "r = tkinter.Tk()\n"
          "r.update_idletasks()\n"
          "r.destroy()\n"
          "print('TK-DISPLAY-OK')\n")
_cached: "bool | None" = None


def tk_display_available() -> bool:
    global _cached
    if _cached is None:
        try:
            proc = subprocess.run([sys.executable, "-c", _PROBE],
                                  capture_output=True, text=True, timeout=60)
            _cached = proc.returncode == 0 and "TK-DISPLAY-OK" in proc.stdout
        except Exception:
            _cached = False
    return _cached
