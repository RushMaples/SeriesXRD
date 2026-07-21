"""Run provenance: version, dependency, and input-identity records.

Every SeriesXRD artifact should say which SeriesXRD version wrote it, when,
from which inputs, and with what effective configuration — separately from
the artifact's *schema* version, which only changes when the file layout
changes. This module centralizes those records so the analysis HDF5, the
JSON manifests, and the GUI's "Copy run diagnostics" action all report the
same facts.

Layout written into an analysis HDF5:

    /provenance  attrs: seriesxrd_version, schema_version, tool, created_at,
                        python_version, platform, config_json,
                        dependencies_json, and per input <name>:
                        input_<name>_path, input_<name>_bytes,
                        input_<name>_mtime, input_<name>_sha256,
                        input_<name>_hash_kind
    /provenance/steps/<step>  attrs: tool, seriesxrd_version, schema_version,
                        created_at  (one per appending analysis step; the
                        step's own group carries its knob attrs)

Input hashing is honest about its cost tradeoff: files up to
``FULL_HASH_MAX_BYTES`` get a full SHA-256 (``hash_kind="sha256"``); larger
files get a SHA-256 over (size, first MiB, last MiB) recorded as
``hash_kind="sha256_head_tail"`` so a multi-GB reduced file does not stall
every run. Either kind changes whenever the file content it samples changes.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import VERSION, now_iso

# Full-file hash below this size; head/tail fingerprint above it.
FULL_HASH_MAX_BYTES = 64 * 1024 * 1024
_SAMPLE_BYTES = 1024 * 1024

# Package name -> import name, in reporting order. Optional extras are
# reported as "not installed" rather than omitted, so a diagnostics dump
# shows the whole picture.
_DEP_IMPORTS = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("h5py", "h5py"),
    ("pyFAI", "pyFAI"),
    ("fabio", "fabio"),
    ("matplotlib", "matplotlib"),
    ("Pillow", "PIL"),
    ("tifffile", "tifffile"),
    ("hdf5plugin", "hdf5plugin"),
    ("pymatgen", "pymatgen"),
    ("torch", "torch"),
)


def dependency_versions() -> Dict[str, str]:
    """Installed version of every dependency SeriesXRD can use.

    Missing optional packages report ``"not installed"``; a package that
    imports but exposes no version string reports ``"unknown"``.
    """
    out: Dict[str, str] = {"python": sys.version.split()[0]}
    for name, mod in _DEP_IMPORTS:
        try:
            m = __import__(mod)
        except Exception:
            out[name] = "not installed"
            continue
        ver = getattr(m, "__version__", None) or getattr(m, "version", None)
        # pyFAI/fabio expose `version` as a plain string attribute.
        out[name] = str(ver) if isinstance(ver, str) and ver else "unknown"
    return out


def file_fingerprint(path: "str | Path") -> Dict[str, Any]:
    """Identity record for an input file: path, size, mtime, content hash.

    See the module docstring for the full-hash vs head/tail tradeoff.
    Returns ``{"path", "bytes", "mtime", "sha256", "hash_kind"}``; on any
    I/O failure the hash fields are empty strings rather than raising, so
    provenance never blocks the run it documents.
    """
    p = Path(path)
    rec: Dict[str, Any] = {"path": str(p), "bytes": -1, "mtime": "",
                           "sha256": "", "hash_kind": ""}
    try:
        st = p.stat()
        rec["bytes"] = int(st.st_size)
        rec["mtime"] = datetime.fromtimestamp(
            st.st_mtime, tz=timezone.utc).isoformat()
        h = hashlib.sha256()
        with open(p, "rb") as f:
            if st.st_size <= FULL_HASH_MAX_BYTES:
                for chunk in iter(lambda: f.read(_SAMPLE_BYTES), b""):
                    h.update(chunk)
                rec["hash_kind"] = "sha256"
            else:
                h.update(str(st.st_size).encode())
                h.update(f.read(_SAMPLE_BYTES))
                f.seek(max(0, st.st_size - _SAMPLE_BYTES))
                h.update(f.read(_SAMPLE_BYTES))
                rec["hash_kind"] = "sha256_head_tail"
        rec["sha256"] = h.hexdigest()
    except Exception:
        pass
    return rec


def manifest_provenance(tool: str, schema_version: str) -> Dict[str, Any]:
    """The standard manifest header: who wrote this artifact, and when."""
    return {
        "tool": tool,
        "seriesxrd_version": VERSION,
        "schema_version": str(schema_version),
        "created_at": now_iso(),
    }


def _json_attr(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:
        return "{}"


def write_provenance(
    h5file,
    *,
    tool: str,
    schema_version: str,
    config: "Optional[Dict[str, Any]]" = None,
    inputs: "Optional[Dict[str, str | Path]]" = None,
) -> None:
    """Write the ``/provenance`` group of an open ``h5py.File``.

    Replaces any existing group — this records the run that *created* the
    file (appending steps use :func:`write_step_provenance` instead).
    """
    if "provenance" in h5file:
        del h5file["provenance"]
    g = h5file.create_group("provenance")
    g.attrs.update({
        "seriesxrd_version": VERSION,
        "schema_version": str(schema_version),
        "tool": tool,
        "created_at": now_iso(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config_json": _json_attr(config or {}),
        "dependencies_json": _json_attr(dependency_versions()),
    })
    for name, path in (inputs or {}).items():
        rec = file_fingerprint(path)
        g.attrs[f"input_{name}_path"] = rec["path"]
        g.attrs[f"input_{name}_bytes"] = int(rec["bytes"])
        g.attrs[f"input_{name}_mtime"] = rec["mtime"]
        g.attrs[f"input_{name}_sha256"] = rec["sha256"]
        g.attrs[f"input_{name}_hash_kind"] = rec["hash_kind"]


def write_step_provenance(h5file, step: str, *, tool: str,
                          schema_version: str) -> None:
    """Record an appending analysis step under ``/provenance/steps/<step>``.

    A re-run of the same step overwrites its own record. The step's knob
    attrs live on the step's data group (``/peaks``, ``/identify``, ...);
    this record pins down which SeriesXRD version wrote them and when.
    """
    try:
        steps = h5file.require_group("provenance").require_group("steps")
        if step in steps:
            del steps[step]
        g = steps.create_group(step)
        g.attrs.update({
            "tool": tool,
            "seriesxrd_version": VERSION,
            "schema_version": str(schema_version),
            "created_at": now_iso(),
        })
    except Exception:
        # Provenance must never fail the run it documents.
        pass


def provenance_report(analysis_h5: "Optional[str | Path]" = None) -> str:
    """Human-readable diagnostics block for support requests and the GUI's
    "Copy run diagnostics" action: SeriesXRD + dependency versions, platform,
    and — when an analysis file is given — its recorded provenance."""
    lines = [
        f"SeriesXRD {VERSION}",
        f"Python {sys.version.split()[0]} on {platform.platform()}",
        "",
        "Dependencies:",
    ]
    for name, ver in dependency_versions().items():
        if name == "python":
            continue
        lines.append(f"  {name}: {ver}")
    if analysis_h5:
        p = Path(analysis_h5)
        lines += ["", f"Analysis file: {p}"]
        try:
            import h5py  # type: ignore
            with h5py.File(str(p), "r") as h5:
                lines.append("Root attributes:")
                for k in sorted(h5.attrs):
                    lines.append(f"  {k}: {h5.attrs[k]!r}")
                if "provenance" in h5:
                    g = h5["provenance"]
                    lines.append("Recorded provenance:")
                    for k in sorted(g.attrs):
                        lines.append(f"  {k}: {g.attrs[k]!r}")
                    if "steps" in g:
                        for s in sorted(g["steps"]):
                            sa = g[f"steps/{s}"].attrs
                            lines.append(
                                f"  step {s}: seriesxrd "
                                f"{sa.get('seriesxrd_version', '?')} at "
                                f"{sa.get('created_at', '?')}")
        except Exception as exc:
            lines.append(f"  (unreadable: {exc})")
    return "\n".join(lines)
