"""Launch helpers for Dioptas. Manual-control only; no fragile GUI automation."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import shlex
import subprocess
import sys


def build_dioptas_command(dioptas_command: str = "", dioptas_python: str = "", image_file: str = "", poni_file: str = "", mask_file: str = "") -> List[str]:
    # NOTE: The `dioptas` entry point only accepts a JSON *project* file as a
    # positional argument (or the literals "test"/"version"/"makeshortcut").
    # It has NO flags for image, PONI, or mask files — passing them as trailing
    # positionals causes Dioptas to try to open the image as a JSON project and
    # fail silently.  Files must be loaded manually inside the Dioptas UI; see
    # dioptas_manual_instructions() for human-readable guidance.
    if dioptas_command:
        cmd = shlex.split(dioptas_command)
    elif dioptas_python:
        cmd = [dioptas_python, "-m", "dioptas"]
    else:
        # Last resort: try python -m dioptas from the current interpreter.
        cmd = [sys.executable, "-m", "dioptas"]
    return cmd


def dioptas_manual_instructions(image_file: str = "", poni_file: str = "", mask_file: str = "") -> str:
    """Return human-readable instructions for loading files inside the Dioptas UI.

    Dioptas does not accept image, PONI, or mask files from the command line —
    only a JSON project file is accepted positionally.  This helper produces a
    message listing which files the user should load manually, omitting any
    paths that are empty.
    """
    lines = [
        "Dioptas does not accept these files from the command line.",
        "In Dioptas, load manually:",
    ]
    any_listed = False
    if image_file:
        lines.append(f"  Calibration → Load image: {image_file}")
        any_listed = True
    if poni_file:
        lines.append(f"  Calibration → Load calibration (.poni): {poni_file}")
        any_listed = True
    if mask_file:
        lines.append(f"  Mask → Load mask: {mask_file}")
        any_listed = True
    if not any_listed:
        return "No files specified — open Dioptas and load your files from the UI."
    return "\n".join(lines)


def launch_dioptas(dioptas_command: str = "", dioptas_python: str = "", image_file: str = "", poni_file: str = "", mask_file: str = "") -> subprocess.Popen:
    cmd = build_dioptas_command(dioptas_command, dioptas_python, image_file, poni_file, mask_file)
    print("[DIOPTAS] Launch command:", " ".join(map(str, cmd)), flush=True)
    # Redirect stdout/stderr to DEVNULL — reading from a PIPE that nothing drains
    # will deadlock Dioptas once the OS pipe buffer fills.
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[DIOPTAS] Started PID {proc.pid}", flush=True)
    if image_file or poni_file or mask_file:
        print(dioptas_manual_instructions(image_file, poni_file, mask_file), flush=True)
    return proc
