"""The inter-stage handoff contract.

The calibration stage ends by writing ``calibration_handoff.json``
(see ``calib.processing.export_accepted_generation``). Every later stage
starts by loading that file through this module, so the contract is defined
in exactly one place. Dependency-light on purpose: stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import read_json

HANDOFF_FILENAME = "calibration_handoff.json"


@dataclass
class Handoff:
    """Validated view of an accepted-calibration handoff."""
    path: Path
    accepted_poni: Path
    accepted_mask_npz: Optional[Path]
    accepted_generation: str = ""
    accepted_folder: str = ""
    source_image: str = ""
    tool_version: str = ""
    created_at: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary_lines(self) -> List[str]:
        lines = [
            f"Handoff file:   {self.path}",
            f"Created:        {self.created_at or 'unknown'}",
            f"Tool version:   {self.tool_version or 'unknown'}",
            f"Generation:     {self.accepted_generation or 'unknown'}",
            f"Accepted PONI:  {self.accepted_poni}",
            f"Accepted mask:  {self.accepted_mask_npz or '(none — automatic mask only)'}",
            f"Source image:   {self.source_image or 'unknown'}",
        ]
        if self.problems:
            lines.append("PROBLEMS:")
            lines += [f"  - {p}" for p in self.problems]
        else:
            lines.append("Status:         OK")
        return lines


def _resolve_relative(base: Path, value: str) -> Path:
    """Resolve a handoff path; tolerate handoff folders that were moved."""
    p = Path(value).expanduser()
    if p.exists():
        return p.resolve()
    # The handoff JSON lives in <accepted_folder>/metadata/. If the accepted
    # folder was copied or moved wholesale, recover files by name relative
    # to it before giving up.
    candidate = base.parent.parent / "accepted_calibrations" / p.name
    if candidate.exists():
        return candidate.resolve()
    return p


def load_handoff(path: "str | Path") -> Handoff:
    """Load and validate a handoff JSON. Never raises on content problems;
    inspect ``Handoff.problems`` / ``Handoff.ok`` instead."""
    p = Path(path).expanduser().resolve()
    problems: List[str] = []
    data: Dict[str, Any] = {}
    if p.is_dir():
        p = p / HANDOFF_FILENAME
    if not p.exists():
        problems.append(f"Handoff file not found: {p}")
    else:
        data = read_json(p)
        if not data:
            problems.append(f"Handoff file is empty or not valid JSON: {p}")

    poni_str = str(data.get("accepted_poni", "") or "")
    mask_str = str(data.get("accepted_mask_npz", "") or "")
    poni = _resolve_relative(p, poni_str) if poni_str else Path("")
    mask = _resolve_relative(p, mask_str) if mask_str else None

    if not poni_str:
        problems.append("Handoff has no 'accepted_poni' entry — was a calibration accepted and exported?")
    elif not poni.is_file():
        problems.append(f"Accepted PONI file is missing on disk: {poni}")
    if mask is not None and not mask.is_file():
        problems.append(f"Accepted mask file is missing on disk: {mask}")
        mask = None

    verification = data.get("verification") or {}
    if verification and not verification.get("ok", True):
        problems.append(f"Calibration export reported missing items: {verification.get('missing')}")

    return Handoff(
        path=p,
        accepted_poni=poni,
        accepted_mask_npz=mask,
        accepted_generation=str(data.get("accepted_generation", "") or ""),
        accepted_folder=str(data.get("accepted_folder", "") or ""),
        source_image=str(data.get("source_image", "") or ""),
        tool_version=str(data.get("tool_version", "") or ""),
        created_at=str(data.get("created_at", "") or ""),
        raw=data,
        problems=problems,
    )


def find_latest_handoff(search_root: "str | Path") -> Optional[Path]:
    """Newest handoff JSON below a folder (accepted-calibrations root or metadata root)."""
    root = Path(search_root).expanduser()
    if not root.exists():
        return None
    matches = sorted(
        root.rglob(HANDOFF_FILENAME),
        key=lambda q: q.stat().st_mtime if q.exists() else 0,
        reverse=True,
    )
    return matches[0] if matches else None
