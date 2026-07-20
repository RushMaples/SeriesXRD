"""Read-only review of a reduced HDF5 file — the checkpoint between the
reduce and analysis stages.

Pure logic (h5py + numpy only): opens a ``reduced_*.h5`` written by
reduce.processing.reduce_dataset, summarizes its structure, samples a handful
of 1D patterns (and one cake if present), and flags obvious anomalies — so a
user can confirm the reduction looks sane before moving on to analysis. h5py is
imported lazily so this module imports without it installed.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


def _attr_str(v: Any) -> str:
    try:
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace")
        s = str(v)
        return s if len(s) <= 4000 else s[:4000] + " …(truncated)"
    except Exception:
        return repr(v)


def _decode(x: Any) -> str:
    return x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else str(x)


def _walk(h5obj, prefix: str = "", lines: "List[str] | None" = None) -> List[str]:
    import h5py  # type: ignore
    if lines is None:
        lines = []
    for key, item in h5obj.items():
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Group):
            lines.append(f"  {path}/  (group)")
            _walk(item, path, lines)
        else:
            lines.append(f"  {path}  shape={item.shape} dtype={item.dtype}")
    return lines


def _detect_anomalies(out: Dict[str, Any]) -> List[str]:
    a = list(out.get("anomalies", []))
    n = out.get("n_frames", 0)
    nf = out.get("n_failed", 0)
    if n and nf:
        pct = 100.0 * nf / n
        a.append(f"{nf}/{n} frames failed integration ({pct:.0f}%)"
                 + (" — check mask/geometry" if pct > 20 else ""))
    if out.get("radial") is None and out.get("patterns"):
        a.append("No radial axis dataset found — x-values unknown.")
    # A6: flag missing radial axis (written when all integration frames failed).
    radial_written_attr = out.get("attrs", {}).get("radial_written", "True")
    radial_written = str(radial_written_attr).lower() not in ("false", "0")
    radial = out.get("radial")
    if (not radial_written) or (radial is not None and np.all(radial == 0)):
        a.append("Radial axis missing (all integration frames failed)")
    bad: List[str] = []
    for pr in out.get("patterns", []):
        inten = np.asarray(pr["intensity"], dtype=float)
        finite = np.isfinite(inten)
        if not finite.any():
            bad.append(f"{pr['name']}: all-NaN pattern")
        elif np.nanmax(inten) <= 0:
            bad.append(f"{pr['name']}: all-zero/negative pattern")
        elif finite.mean() < 0.5:
            bad.append(f"{pr['name']}: >50% NaN bins")
    a.extend(bad[:10])
    if not a:
        a.append("No anomalies detected in the sampled patterns.")
    return a


def inspect_reduction(h5_path: "str | Path", max_patterns: int = 16) -> Dict[str, Any]:
    """Open a reduced HDF5 file and return a structure/sample/anomaly summary.

    Defensive: a missing file, missing h5py, or a partially-written file yields
    a result dict with anomalies rather than raising.
    """
    p = Path(h5_path).expanduser()
    out: Dict[str, Any] = {
        "path": str(p), "ok_to_read": False, "structure_lines": [], "attrs": {},
        "n_frames": 0, "n_ok": 0, "n_failed": 0, "failed_sample": [],
        "unit": "", "radial": None, "sample_indices": [], "patterns": [],
        "robust_present": False, "cake_present": False, "cake": None,
        "cake_radial": None, "cake_azimuthal": None, "cake_frame_index": None,
        "anomalies": [],
    }
    if not p.is_file():
        out["anomalies"].append(f"File does not exist: {p}")
        return out
    try:
        import h5py  # type: ignore
    except ImportError:
        out["anomalies"].append("h5py is not installed — cannot read reduced HDF5.")
        return out
    try:
        with h5py.File(str(p), "r") as h5:
            out["ok_to_read"] = True
            out["attrs"] = {k: _attr_str(v) for k, v in h5.attrs.items()}
            out["unit"] = str(h5.attrs.get("unit", ""))
            out["structure_lines"] = _walk(h5)

            frames = h5.get("frames")
            ok = None
            names = None
            if frames is not None and "ok" in frames:
                ok = np.asarray(frames["ok"][:], dtype=bool)
                out["n_frames"] = int(ok.size)
                out["n_ok"] = int(ok.sum())
                out["n_failed"] = int((~ok).sum())
            if frames is not None and "filename" in frames:
                try:
                    names = [_decode(x) for x in frames["filename"][:]]
                except Exception:
                    names = None
            if ok is not None and names is not None:
                out["failed_sample"] = [names[i] for i in np.where(~ok)[0][:20]]

            pat = h5.get("patterns")
            if pat is not None and "intensity" in pat:
                ds_int = pat["intensity"]
                total = int(ds_int.shape[0])
                if "radial" in pat:
                    out["radial"] = np.asarray(pat["radial"][:])
                idx_pool = np.where(ok)[0] if ok is not None else np.arange(total)
                if idx_pool.size == 0:
                    idx_pool = np.arange(total)
                k = min(max_patterns, idx_pool.size)
                sel = idx_pool[np.linspace(0, idx_pool.size - 1, k).astype(int)] if k > 0 else []
                out["sample_indices"] = [int(i) for i in sel]
                out["robust_present"] = "intensity_robust" in pat
                for i in sel:
                    inten = np.asarray(ds_int[int(i)])
                    nm = names[int(i)] if names and int(i) < len(names) else f"frame {int(i)}"
                    out["patterns"].append({"index": int(i), "name": nm, "intensity": inten})

            cakes = h5.get("cakes")
            if cakes is not None and "intensity" in cakes and cakes["intensity"].shape[0] > 0:
                out["cake_present"] = True
                out["cake"] = np.asarray(cakes["intensity"][0])
                if "radial" in cakes:
                    out["cake_radial"] = np.asarray(cakes["radial"][:])
                if "azimuthal" in cakes:
                    out["cake_azimuthal"] = np.asarray(cakes["azimuthal"][:])
                if "frame_index" in cakes:
                    fi = np.asarray(cakes["frame_index"][:])
                    out["cake_frame_index"] = int(fi[0]) if fi.size else None

            out["anomalies"] = _detect_anomalies(out)
    except Exception as e:
        out["anomalies"].append(f"Failed to read HDF5: {e!r}")
    return out


def gallery_frames(h5_path: "str | Path") -> Dict[str, Any]:
    """Per-frame metadata for the gallery matrix view (no heavy arrays).

    Returns ``{"ok_to_read", "previews_dir", "n_frames", "frames": [...],
    "error"}`` where each frame is ``{index, filename, ok, excluded,
    thumb}`` with ``thumb`` an absolute path (or "" if none). Thumbnails are
    NOT loaded here — the GUI loads only the visible ones.
    """
    p = Path(h5_path).expanduser()
    out: Dict[str, Any] = {"ok_to_read": False, "previews_dir": "", "n_frames": 0,
                           "frames": [], "error": ""}
    if not p.is_file():
        out["error"] = f"File does not exist: {p}"
        return out
    try:
        import h5py  # type: ignore
    except ImportError:
        out["error"] = "h5py is not installed."
        return out
    try:
        with h5py.File(str(p), "r") as h5:
            frames = h5.get("frames")
            if frames is None or "filename" not in frames:
                out["error"] = "No frames group in this HDF5."
                return out
            names = [_decode(x) for x in frames["filename"][:]]
            n = len(names)
            ok = np.asarray(frames["ok"][:], dtype=bool) if "ok" in frames else np.ones(n, bool)
            excl = np.asarray(frames["excluded"][:], dtype=bool) if "excluded" in frames else np.zeros(n, bool)
            thumbs = [_decode(x) for x in frames["thumb"][:]] if "thumb" in frames else [""] * n
            # previews_dir: prefer the stored attr, else infer beside the .h5.
            pv = frames.attrs.get("previews_dir") or h5.attrs.get("previews_dir") or ""
            pv = _decode(pv) if pv else ""
            if pv and not Path(pv).is_dir():
                # Handoff folder may have moved — look beside the .h5.
                guess = p.parent / (p.stem + "_previews")
                if guess.is_dir():
                    pv = str(guess)
            out["ok_to_read"] = True
            out["previews_dir"] = pv
            out["n_frames"] = n
            for i in range(n):
                t = thumbs[i] if i < len(thumbs) else ""
                tabs = str(Path(pv) / t) if (pv and t) else ""
                out["frames"].append({
                    "index": i, "filename": names[i],
                    "ok": bool(ok[i]), "excluded": bool(excl[i]), "thumb": tabs,
                })
    except Exception as e:
        out["error"] = f"Failed to read HDF5: {e!r}"
    return out


def set_excluded(h5_path: "str | Path", indices, excluded: bool) -> int:
    """Persist exclusion flags into ``frames/excluded`` (in place). Returns the
    number of frames updated. The original data is never removed — analysis
    reads the flag and skips excluded frames."""
    p = Path(h5_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    import h5py  # type: ignore
    idx = sorted({int(i) for i in indices})
    if not idx:
        return 0
    with h5py.File(str(p), "r+") as h5:
        frames = h5.get("frames")
        if frames is None or "excluded" not in frames:
            raise KeyError("frames/excluded dataset not present in this HDF5.")
        ds = frames["excluded"]
        for i in idx:
            if 0 <= i < ds.shape[0]:
                ds[i] = bool(excluded)
    return len(idx)


def review_reduction(h5_path: "str | Path") -> Dict[str, Any]:
    """Print the structure/anomaly report for a reduced HDF5 file and return the
    full review dict (radial axis, sampled patterns, cake sample, anomalies)."""
    review = inspect_reduction(h5_path)
    print(structure_report(review), flush=True)
    return review


def structure_report(review: Dict[str, Any]) -> str:
    """Human-readable text block for the GUI/notebook."""
    L: List[str] = [f"File: {review['path']}"]
    if not review.get("ok_to_read"):
        L.append("")
        L.append("ANOMALIES:")
        L += [f"  - {x}" for x in review.get("anomalies", [])]
        return "\n".join(L)
    L.append(f"Frames: {review['n_ok']} ok / {review['n_frames']} total"
             + (f"  ({review['n_failed']} failed)" if review['n_failed'] else ""))
    L.append(f"Unit: {review.get('unit') or '?'}    "
             f"Robust pattern: {'yes' if review['robust_present'] else 'no'}    "
             f"Cakes: {'yes' if review['cake_present'] else 'no'}")
    L.append("")
    L.append("HDF5 structure:")
    L += review.get("structure_lines", [])
    attrs = review.get("attrs", {})
    if attrs:
        L.append("")
        L.append("Root attributes:")
        for k, v in attrs.items():
            if k == "poni_text":
                continue
            L.append(f"  {k}: {v}")
    if review.get("failed_sample"):
        L.append("")
        L.append("Failed frames (sample):")
        L += [f"  {x}" for x in review["failed_sample"]]
    L.append("")
    L.append("Anomalies:")
    L += [f"  - {x}" for x in review.get("anomalies", [])]
    return "\n".join(L)
