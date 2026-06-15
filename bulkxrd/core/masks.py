"""Mask creation and persistence."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple
import numpy as np


def automatic_mask(data: np.ndarray, mask_negative=True, mask_zero=True, mask_nonfinite=True, saturated_threshold=None) -> np.ndarray:
    arr = np.asarray(data)
    mask = np.zeros(arr.shape, dtype=bool)
    if mask_nonfinite:
        mask |= ~np.isfinite(arr)
    if mask_negative:
        mask |= arr < 0
    if mask_zero:
        mask |= arr == 0
    if saturated_threshold not in (None, "", "None"):
        try:
            thr = float(saturated_threshold)
            mask |= arr >= thr
        except Exception:
            pass
    return mask


def polygon_to_mask(shape: Tuple[int, int], points: List[Tuple[float, float]]) -> np.ndarray:
    # Uses matplotlib.path if available; no hard import at module import time.
    from matplotlib.path import Path as MplPath  # type: ignore
    h, w = shape
    if len(points) < 3:
        return np.zeros(shape, dtype=bool)
    yy, xx = np.mgrid[:h, :w]
    coords = np.vstack((xx.ravel(), yy.ravel())).T
    path = MplPath(points)
    return path.contains_points(coords).reshape(shape)


def save_mask_npz(path: str | Path, mask: np.ndarray, metadata: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Inject orientation provenance keys without clobbering caller-provided values.
    meta: dict = dict(metadata) if metadata else {}
    meta.setdefault("origin", "upper")
    meta.setdefault("convention", "pyFAI (origin upper-left, True = masked pixel)")
    meta.setdefault("shape", list(mask.shape))
    np.savez_compressed(p, mask=np.asarray(mask, dtype=bool), metadata=meta)
    return p


def load_mask_npz(path: str | Path) -> np.ndarray:
    p = Path(path)
    with np.load(p, allow_pickle=True) as z:
        if "mask" in z:
            return np.asarray(z["mask"], dtype=bool)
        # fallback first array
        first = list(z.files)[0]
        return np.asarray(z[first], dtype=bool)


def load_mask_npz_with_metadata(path: str | Path) -> "tuple[np.ndarray, dict]":
    """Load a mask and its provenance metadata from an .npz file.

    Returns (mask_array, metadata_dict).  If the file has no 'metadata' array,
    an empty dict is returned so callers need not guard against KeyError.
    """
    p = Path(path)
    with np.load(p, allow_pickle=True) as z:
        if "mask" in z:
            mask = np.asarray(z["mask"], dtype=bool)
        else:
            first = list(z.files)[0]
            mask = np.asarray(z[first], dtype=bool)
        if "metadata" in z:
            raw_meta = z["metadata"]
            # numpy stores dict-like objects as 0-d object arrays
            try:
                meta = raw_meta.item() if hasattr(raw_meta, "item") else dict(raw_meta)
            except Exception:
                meta = {}
        else:
            meta = {}
    return mask, meta


def save_mask_preview_png(path: str | Path, mask: np.ndarray) -> Path:
    from PIL import Image  # type: ignore
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype("uint8"))
    img.save(p)
    return p
