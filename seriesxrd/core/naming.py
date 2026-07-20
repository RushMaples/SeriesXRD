"""Consistent readable filenames for calibration generations."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict
from .config import safe_stem, now_timestamp, ensure_dir


def gen_label(index: int) -> str:
    return f"gen{int(index):03d}"


def generation_stem(index: int, kind: str, session_name: str = "calibration", timestamp: str | None = None) -> str:
    ts = timestamp or now_timestamp()
    return f"{gen_label(index)}_{safe_stem(kind)}_{safe_stem(session_name)}_{ts}"


def generation_paths(base_dirs: Dict[str, Path], index: int, session_name: str, timestamp: str | None = None) -> Dict[str, Path]:
    ts = timestamp or now_timestamp()
    glabel = gen_label(index)
    fig_dir = ensure_dir(base_dirs["figures"] / glabel)
    data_dir = ensure_dir(base_dirs["data"] / glabel)
    meta_dir = ensure_dir(base_dirs["metadata"] / glabel)
    def f(kind: str, ext: str) -> Path:
        return fig_dir / f"{generation_stem(index, kind, session_name, ts)}.{ext}"
    def d(kind: str, ext: str) -> Path:
        return data_dir / f"{generation_stem(index, kind, session_name, ts)}.{ext}"
    def m(kind: str, ext: str) -> Path:
        return meta_dir / f"{generation_stem(index, kind, session_name, ts)}.{ext}"
    return {
        "fig_dir": fig_dir,
        "data_dir": data_dir,
        "meta_dir": meta_dir,
        "raw_detector_png": f("raw_detector", "png"),
        "masked_detector_png": f("masked_detector", "png"),
        "mask_only_png": f("mask_only", "png"),
        "intensity_difference_png": f("intensity_difference_stacked", "png"),
        "intensity_normalized_png": f("intensity_normalized", "png"),
        "cake_png": f("cake", "png"),
        "coverage_png": f("coverage", "png"),
        "compilation_png": f("compilation_QA", "png"),
        "compilation_pdf": f("compilation_QA", "pdf"),
        "intensity_csv": d("intensity_vs_2theta", "csv"),
        "difference_csv": d("difference", "csv"),
        "coverage_csv": d("coverage", "csv"),
        "cake_npz": d("cake", "npz"),
        "mask_npz": d("mask", "npz"),
        "master_csv": d("master_data", "csv"),
        "report_txt": m("report", "txt"),
        "metadata_json": m("metadata", "json"),
    }


def next_available_path(path: "Path | str", *, is_dir: "bool | None" = None) -> Path:
    """Return a non-existing path, choosing max_existing_sibling_index + 1.

    Examples:
        filename        exists -> filename_001
        filename_002    exists with _001.._005 siblings -> filename_006
        filename002     exists with 001..005 siblings   -> filename006
    """
    p = Path(path)
    if not p.exists():
        return p
    parent = p.parent
    is_directory = p.is_dir() or (is_dir is True)
    if is_directory:
        stem = p.name
        suffix = ""
    else:
        stem = p.stem
        suffix = p.suffix
    # Detect trailing index: base [sep] digits
    m = re.match(r'^(.*?)([_-]?)(\d{1,6})$', stem)
    if m:
        base, sep, digits = m.group(1), m.group(2), m.group(3)
        width = len(digits)
    else:
        base = stem
        sep = "_"
        width = 3
    # Scan siblings for highest existing index with the same base pattern
    pat = re.compile(
        rf'^{re.escape(base)}{re.escape(sep)}(\d{{{width},}}){re.escape(suffix)}$',
        re.IGNORECASE,
    )
    max_idx = 0
    for sibling in parent.iterdir():
        sm = pat.match(sibling.name)
        if sm:
            try:
                max_idx = max(max_idx, int(sm.group(1)))
            except ValueError:
                pass
    next_idx = max(max_idx + 1, 1)
    new_name = f"{base}{sep}{next_idx:0{width}d}{suffix}"
    return parent / new_name
