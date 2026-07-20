"""Environment and dependency checks for SeriesXRD workflows."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import importlib.util
import json
import os
import shutil
import subprocess
import sys

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "Pillow": "PIL",
    "matplotlib": "matplotlib",
    "pyFAI": "pyFAI",
    "fabio": "fabio",
}
OPTIONAL_IMPORTS = {
    "tifffile": "tifffile",
    "h5py": "h5py",
    "hdf5plugin": "hdf5plugin",
}

@dataclass
class DependencyStatus:
    python_exe: str
    required: Dict[str, bool]
    optional: Dict[str, bool]
    tkinter_ok: bool
    missing_required: List[str]
    missing_optional: List[str]
    conda_exe: str = ""
    conda_env_name: str = ""
    checked_in: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _spec_ok(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _same_interpreter(python_exe: str) -> bool:
    """Return True if python_exe resolves to the same executable as sys.executable."""
    if not python_exe:
        return True
    try:
        return Path(python_exe).resolve() == Path(sys.executable).resolve()
    except Exception:
        return False


def _check_deps_subprocess(python_exe: str) -> "Optional[Dict[str, bool]]":
    """Run a short script in python_exe that reports package availability as JSON.
    Returns a dict with all REQUIRED_IMPORTS and OPTIONAL_IMPORTS module names plus
    'tkinter', or None on failure."""
    all_mods = {**REQUIRED_IMPORTS, **OPTIONAL_IMPORTS, "tkinter": "tkinter"}
    checks = {pkg: mod for pkg, mod in all_mods.items()}
    script_lines = [
        "import importlib.util, json, sys",
        f"checks = {repr(checks)}",
        "result = {pkg: importlib.util.find_spec(mod) is not None for pkg, mod in checks.items()}",
        "print(json.dumps(result))",
    ]
    script = "; ".join(script_lines)
    try:
        proc = subprocess.run(
            [python_exe, "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout.strip())
    except Exception:
        return None


def check_dependencies(python_exe: str | None = None, conda_exe: str = "", conda_env_name: str = "") -> DependencyStatus:
    resolved_exe = python_exe or sys.executable
    checked_in = ""

    if _same_interpreter(resolved_exe):
        # Fast in-process check.
        required = {pkg: _spec_ok(mod) for pkg, mod in REQUIRED_IMPORTS.items()}
        optional = {pkg: _spec_ok(mod) for pkg, mod in OPTIONAL_IMPORTS.items()}
        try:
            import tkinter  # noqa: F401
            tkinter_ok = True
        except Exception:
            tkinter_ok = False
        checked_in = resolved_exe
    else:
        # Check in the target interpreter via subprocess.
        result = _check_deps_subprocess(resolved_exe)
        if result is None:
            # Subprocess failed — fall back to in-process check.
            required = {pkg: _spec_ok(mod) for pkg, mod in REQUIRED_IMPORTS.items()}
            optional = {pkg: _spec_ok(mod) for pkg, mod in OPTIONAL_IMPORTS.items()}
            try:
                import tkinter  # noqa: F401
                tkinter_ok = True
            except Exception:
                tkinter_ok = False
            checked_in = "current-process (fallback)"
        else:
            required   = {pkg: bool(result.get(pkg, False)) for pkg in REQUIRED_IMPORTS}
            optional   = {pkg: bool(result.get(pkg, False)) for pkg in OPTIONAL_IMPORTS}
            tkinter_ok = bool(result.get("tkinter", False))
            checked_in = resolved_exe

    missing_required = [pkg for pkg, ok in required.items() if not ok]
    if not tkinter_ok:
        missing_required.append("tkinter")
    missing_optional = [pkg for pkg, ok in optional.items() if not ok]
    return DependencyStatus(
        python_exe=resolved_exe,
        required=required,
        optional=optional,
        tkinter_ok=tkinter_ok,
        missing_required=missing_required,
        missing_optional=missing_optional,
        conda_exe=conda_exe or find_conda_exe(),
        conda_env_name=conda_env_name or os.environ.get("CONDA_DEFAULT_ENV", ""),
        checked_in=checked_in,
    )


def find_conda_exe() -> str:
    for name in ["mamba", "conda"]:
        found = shutil.which(name)
        if found:
            return found
    candidates = []
    if os.name == "nt":
        user = Path.home()
        candidates += [
            user / "miniforge3" / "Scripts" / "mamba.exe",
            user / "miniforge3" / "Scripts" / "conda.exe",
            user / "miniconda3" / "Scripts" / "conda.exe",
            Path("C:/ProgramData/miniforge3/Scripts/mamba.exe"),
            Path("C:/ProgramData/miniforge3/Scripts/conda.exe"),
        ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def _derive_conda_env_from_python(python_exe: str) -> str:
    """Try to derive conda env name from python_exe path.

    Conda environments live under <conda_root>/envs/<env_name>/bin/python.
    If the path matches that layout, return <env_name>; else return "".
    """
    try:
        parts = Path(python_exe).resolve().parts
        # Find "envs" directory and take the next component as the env name.
        for i, part in enumerate(parts):
            if part == "envs" and i + 1 < len(parts):
                return parts[i + 1]
    except Exception:
        pass
    return ""


def package_install_command(missing: List[str], python_exe: str, conda_exe: str = "", conda_env_name: str = "") -> List[str]:
    # Map import package names to conda-forge package names.
    conda_map = {
        "numpy": "numpy",
        "Pillow": "pillow",
        "matplotlib": "matplotlib",
        "pyFAI": "pyfai",
        "fabio": "fabio",
        "tifffile": "tifffile",
        "h5py": "h5py",
        "hdf5plugin": "hdf5plugin",
    }
    missing = [m for m in missing if m != "tkinter"]
    if conda_exe and conda_env_name:
        return [conda_exe, "install", "-n", conda_env_name, "-c", "conda-forge", "-y"] + [conda_map.get(m, m) for m in missing]
    if conda_exe:
        # Attempt to derive the env name from python_exe's path so we don't
        # accidentally install into the base environment.
        derived_env = _derive_conda_env_from_python(python_exe)
        if derived_env:
            return [conda_exe, "install", "-n", derived_env, "-c", "conda-forge", "-y"] + [conda_map.get(m, m) for m in missing]
        return [conda_exe, "install", "-c", "conda-forge", "-y"] + [conda_map.get(m, m) for m in missing]
    pip_map = {"Pillow": "pillow", "pyFAI": "pyFAI"}
    return [python_exe, "-m", "pip", "install"] + [pip_map.get(m, m) for m in missing]


def run_install_command(cmd: List[str], cwd: str | None = None) -> int:
    print("[INSTALL] Running:", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print("[INSTALL]", line.rstrip(), flush=True)
    return int(proc.wait())
