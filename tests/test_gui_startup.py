"""Real-Tk GUI startup smoke test.

Constructs the unified application against a fresh workspace, pumps the
event loop once, and closes. Runs only where a display is available (CI
provides one via xvfb); the construction happens in a subprocess so a hang
or native crash fails the test instead of the test session.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _display_available() -> bool:
    try:
        import tkinter
        r = tkinter.Tk()
        r.destroy()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _display_available(),
                    reason="no display (run under xvfb-run to enable)")
def test_unified_app_constructs_and_closes():
    script = (
        "import sys\n"
        "from seriesxrd.app import SeriesXRDApp\n"
        "app = SeriesXRDApp(sys.argv[1])\n"
        "app.root.update_idletasks()\n"
        "app.root.update()\n"
        "app.root.destroy()\n"
        "print('GUI-STARTUP-OK')\n"
    )
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "workspace"
        proc = subprocess.run([sys.executable, "-c", script, str(ws)],
                              capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (
        f"GUI startup failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "GUI-STARTUP-OK" in proc.stdout


if __name__ == "__main__":
    test_unified_app_constructs_and_closes()
    print("OK")
