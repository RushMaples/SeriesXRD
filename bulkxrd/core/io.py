"""I/O helpers for detector images and text data."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence
import csv
import numpy as np


def _resolve_float_dtype(arr: np.ndarray) -> np.ndarray:
    """Recover float32 data from detector TIFFs that store IEEE floats but omit
    the TIFF SampleFormat tag (e.g. Pilatus CdTe and other corrected /
    flat-fielded frames). Generic readers then default to integer and hand back
    int32/uint32 whose bytes are actually floats.

    The bytes are reinterpreted as float32, and that reading is kept only when
    BOTH hold: the integer reading contains values too large to be photon counts
    (>2**28), and the reinterpreted floats land in a physical magnitude range.
    Genuine integer counts reinterpreted as float collapse to denormals
    (~1e-40), and real per-pixel counts never approach 2**28, so neither
    condition fires on true integer images — this never corrupts real count data.
    """
    if arr.dtype not in (np.dtype("int32"), np.dtype("uint32")):
        return arr
    as_float = np.ascontiguousarray(arr).view(np.float32)
    finite = as_float[np.isfinite(as_float)]
    nonzero = finite[finite != 0]
    big = 1 << 28
    int_implausible = bool((arr > big).any() or (arr < -big).any())
    float_physical = nonzero.size > 0 and 1e-4 < float(np.median(np.abs(nonzero))) < 1e9
    if int_implausible and float_physical:
        print("[io] integer-typed TIFF holds float32 data (no SampleFormat tag) "
              "— reinterpreting as float32", flush=True)
        return as_float
    return arr


def read_detector_image(path: str | Path, flip_up_down: bool = False) -> np.ndarray:
    p = Path(path)
    errors = []
    raw = None
    try:
        import fabio  # type: ignore
        raw = np.asarray(fabio.open(str(p)).data)
    except Exception as e:
        errors.append(f"fabio: {e}")
    if raw is None:
        try:
            import tifffile  # type: ignore
            raw = np.asarray(tifffile.imread(str(p)))
        except Exception as e:
            errors.append(f"tifffile: {e}")
    if raw is None:
        try:
            from PIL import Image  # type: ignore
            raw = np.asarray(Image.open(p))
        except Exception as e:
            errors.append(f"PIL: {e}")
    if raw is None:
        raise RuntimeError("Could not read detector image. Tried fabio, tifffile, PIL. " + " | ".join(errors))
    
    arr = np.asarray(_resolve_float_dtype(raw), dtype=np.float32)
    if flip_up_down:
        arr = np.flipud(arr)

    return arr


def write_xy_csv(path: str | Path, x, y, x_name: str = "two_theta_deg", y_name: str = "intensity") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([x_name, y_name])
        for a, b in zip(np.asarray(x).ravel(), np.asarray(y).ravel()):
            w.writerow([float(a), float(b)])
    return p


def write_table_csv(path: str | Path, rows: Sequence[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()}) if rows else []
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return p
