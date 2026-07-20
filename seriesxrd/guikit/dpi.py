"""Hi-DPI awareness helper (Windows only).

Call ``enable_hi_dpi()`` BEFORE creating a ``tk.Tk()`` root so the OS does not
scale the window blurrily.  The call is a no-op on non-Windows platforms.
"""
from __future__ import annotations


def enable_hi_dpi() -> None:
    """Set DPI awareness on Windows.  Safe no-op on Linux/macOS."""
    import os
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except Exception:
        pass
