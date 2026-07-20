"""Detector-image diagnostic: why does this file show no features?

Usage:
    seriesxrd-inspect <image_file>              (console script)
    python -m seriesxrd.core.inspect <image_file>

Prints the file's true format (from magic bytes, not the extension), how
each reader (fabio / tifffile / PIL) interprets it, and intensity
statistics, then states a verdict. Built for the recurring failure mode
where a detector TIFF is shared through chat/preview pipelines and arrives
re-encoded as an 8-bit preview, or displays black because of float/signed
data with bad display scaling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .config import sha256_file

# (magic bytes, human name, is_plausible_detector_format)
_MAGIC = [
    (b"II*\x00", "TIFF (little-endian)", True),
    (b"MM\x00*", "TIFF (big-endian)", True),
    (b"II+\x00", "BigTIFF", True),
    (b"\x89PNG", "PNG", False),
    (b"\xff\xd8\xff", "JPEG", False),
    (b"GIF8", "GIF", False),
    (b"BM", "BMP", False),
    (b"\x0a\x0a", "EDF (possible)", True),
    (b"###CBF", "CBF", True),
    (b"\x89HDF", "HDF5", True),
    (b"PK\x03\x04", "ZIP archive (npz/zip)", False),
    (b"<", "HTML/XML text — likely an error page, not an image", False),
    (b"{", "JSON text — not an image", False),
]


def _detect_magic(head: bytes) -> Tuple[str, Optional[bool]]:
    for magic, name, plausible in _MAGIC:
        if head.startswith(magic):
            return name, plausible
    if head[:4].isascii() and head[:1].isalpha():
        return "ASCII text (unrecognized)", False
    return "unknown", None


def _try_reader(name: str, fn) -> Tuple[str, Optional["object"], List[str]]:
    notes: List[str] = []
    try:
        arr = fn()
        return name, arr, notes
    except Exception as e:
        notes.append(f"{name}: FAILED — {e!r}")
        return name, None, notes


def _stats_lines(arr) -> List[str]:
    import numpy as np
    a = np.asarray(arr)
    lines = [f"  shape={a.shape}  dtype={a.dtype}"]
    if a.ndim == 3:
        lines.append(f"  NOTE: 3-dimensional ({a.shape[-1]} channels) — detector frames are 2D grayscale; "
                     "an RGB(A) image means the file was re-rendered for display.")
    flat = a.astype("float64", copy=False).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        lines.append("  all values non-finite")
        return lines
    q = np.percentile(finite, [0, 1, 50, 99, 100])
    lines.append(f"  min={q[0]:.6g}  p1={q[1]:.6g}  median={q[2]:.6g}  p99={q[3]:.6g}  max={q[4]:.6g}")
    lines.append(f"  zeros: {float(np.mean(finite == 0))*100:.1f}%   negatives: {float(np.mean(finite < 0))*100:.1f}%   "
                 f"unique values (sample): {min(np.unique(finite[:200000]).size, 100000)}")
    return lines


def inspect_image(path: "str | Path", verbose: bool = True) -> int:
    """Print a diagnostic report. Returns 0 if the file looks like a usable
    detector frame, 1 if it is definitely not, 2 if undetermined."""
    import numpy as np

    p = Path(path).expanduser()
    out: List[str] = []
    verdicts: List[str] = []
    rc = 2

    out.append(f"File:    {p}")
    if not p.is_file():
        print("\n".join(out))
        print("VERDICT: file does not exist.")
        return 1
    size = p.stat().st_size
    out.append(f"Size:    {size:,} bytes")
    out.append(f"SHA256:  {sha256_file(p)}")
    out.append("         (compare this hash with the sender's original — if it differs, the file changed in transfer)")

    with p.open("rb") as _f:
        head = _f.read(16)
    fmt, plausible = _detect_magic(head)
    out.append(f"Magic:   {head[:8]!r}  ->  {fmt}")
    ext = p.suffix.lower()
    if ext in (".tif", ".tiff") and "TIFF" not in fmt:
        verdicts.append(f"Extension is {ext} but content is {fmt}: the file was re-encoded or wrongly "
                        "saved (classic chat-app 'save preview' failure). Re-download the ORIGINAL file.")
        rc = 1
    if plausible is False and rc != 1:
        verdicts.append(f"{fmt} is not a raw detector format — this is a rendered/preview image.")
        rc = 1

    arrays = []
    def _fabio():
        import fabio
        img = fabio.open(str(p))
        out.append(f"fabio:   OK — class={type(img).__name__}, frames={getattr(img, 'nframes', 1)}")
        return img.data
    def _tifffile():
        import tifffile
        return tifffile.imread(str(p))
    def _pil():
        from PIL import Image
        im = Image.open(p)
        out.append(f"PIL:     OK — mode={im.mode}, size={im.size}")
        return np.asarray(im)

    for name, fn in (("fabio", _fabio), ("tifffile", _tifffile), ("PIL", _pil)):
        rname, arr, notes = _try_reader(name, fn)
        out += [f"{n}" for n in notes]
        if arr is not None:
            out.append(f"{rname} data:")
            out += _stats_lines(arr)
            arrays.append((rname, np.asarray(arr)))

    if not arrays:
        verdicts.append("No reader could open the file at all — it is corrupted or not an image.")
        rc = 1
    else:
        a = arrays[0][1]
        flat = a.astype("float64", copy=False).ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size and finite.max() == finite.min():
            verdicts.append("Image is perfectly uniform (constant value) — no detector signal present.")
            rc = 1
        elif a.dtype == np.uint8 or (finite.size and finite.max() <= 255 and ext in (".tif", ".tiff")):
            verdicts.append("8-bit dynamic range: detector TIFFs are 16/32-bit. This looks like a "
                            "re-encoded preview, not the original frame.")
            rc = 1
        elif finite.size:
            p99 = float(np.percentile(finite, 99))
            med = float(np.percentile(finite, 50))
            if p99 > med and rc == 2:
                verdicts.append("Data has real dynamic range — the file itself looks like a valid detector frame. "
                                "If it displays black, the problem is display scaling (try log scale or "
                                "percentile-clipped limits), not the file.")
                rc = 0

    if verbose:
        print("\n".join(out))
        print()
        for v in verdicts or ["Undetermined — see statistics above."]:
            print(f"VERDICT: {v}")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a detector image file that shows no features.")
    parser.add_argument("image", help="Path to the image file (TIFF/EDF/CBF/...)")
    args = parser.parse_args()
    return inspect_image(args.image)


if __name__ == "__main__":
    raise SystemExit(main())
