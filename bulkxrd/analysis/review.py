"""Read-only review of an analysis HDF5 file — the QC surface for the analysis
stage (Step 1 background separation + Step 2 peak fitting).

Pure logic (h5py + numpy only): opens an ``*_analysis.h5`` written by
``analysis.background.run_background_separation`` (and optionally augmented with
``/peaks`` by ``analysis.peaks.run_peak_fitting``), summarizes its structure,
and exposes the arrays the GUI needs to plot — per-frame background traces,
fitted peaks, the contamination curve, and the peak map across the series.

The analysis file does NOT store the robust/mean patterns directly; it stores
``clean``, ``baseline`` and ``spot_residual`` per frame, from which the rest is
reconstructed losslessly:

    robust    = clean + baseline                 (spot-suppressed powder signal)
    mean      = robust + spot_residual           (azimuthal mean, with spots)
    hybrid    = clean + winsorized(spot_residual)  (default Step-2 fit source)
    sigmaclip = clean + sigmaclip_residual       (only if the reduce-side
                                                  trimmed-mean channel was stored)

h5py is imported lazily so this module imports without it installed.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
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


def _walk(h5obj, prefix: str = "", lines: "Optional[List[str]]" = None) -> List[str]:
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


# Peak columns appended by analysis.peaks.run_peak_fitting (flat/ragged layout).
_PEAK_COLS = ("center", "amplitude", "fwhm", "eta", "area", "chi2", "flag")


def _read_names(frames) -> "Optional[List[str]]":
    if frames is not None and "filename" in frames:
        try:
            return [_decode(x) for x in frames["filename"][:]]
        except Exception:
            return None
    return None


def inspect_analysis(h5_path: "str | Path") -> Dict[str, Any]:
    """Open an analysis HDF5 and return a structure/summary/anomaly dict.

    Defensive: a missing file, missing h5py, or a partially-written file yields
    a result dict with ``anomalies`` rather than raising.
    """
    p = Path(h5_path).expanduser()
    out: Dict[str, Any] = {
        "path": str(p), "ok_to_read": False, "structure_lines": [], "attrs": {},
        "unit": "", "radial": None, "n_frames": 0, "n_bins": 0,
        "source_reduced": "", "filenames": None,
        "has_background": False, "has_peaks": False,
        "has_residual": False, "has_unknowns": False,
        "contamination": None, "flagged": None,
        "n_peaks": 0, "n_good": 0, "n_flagged_peaks": 0,
        "n_residual_peaks": 0, "n_unknown_obs": 0,
        "n_unknown_tracks": 0, "n_unknown_clusters": 0,
        "peaks_per_frame_mean": 0.0, "peak_attrs": {},
        "residual_attrs": {}, "unknowns_attrs": {},
        "anomalies": [],
    }
    if not p.is_file():
        out["anomalies"].append(f"File does not exist: {p}")
        return out
    try:
        import h5py  # type: ignore
    except ImportError:
        out["anomalies"].append("h5py is not installed — cannot read analysis HDF5.")
        return out
    try:
        with h5py.File(str(p), "r") as h5:
            out["ok_to_read"] = True
            out["attrs"] = {k: _attr_str(v) for k, v in h5.attrs.items()}
            out["unit"] = str(h5.attrs.get("unit", ""))
            out["source_reduced"] = _attr_str(h5.attrs.get("source_reduced", ""))
            out["structure_lines"] = _walk(h5)

            if "radial" in h5:
                out["radial"] = np.asarray(h5["radial"][:])
                out["n_bins"] = int(out["radial"].size)

            frames = h5.get("frames")
            out["filenames"] = _read_names(frames)
            if frames is not None and "contamination" in frames:
                out["contamination"] = np.asarray(frames["contamination"][:], dtype=float)
            if frames is not None and "flagged" in frames:
                out["flagged"] = np.asarray(frames["flagged"][:], dtype=bool)

            bg = h5.get("background")
            if bg is not None and "clean" in bg:
                out["has_background"] = True
                shape = bg["clean"].shape
                out["n_frames"] = int(shape[0])
                if not out["n_bins"]:
                    out["n_bins"] = int(shape[1]) if len(shape) > 1 else 0

            pk = h5.get("peaks")
            if pk is not None and "center" in pk:
                out["has_peaks"] = True
                out["peak_attrs"] = {k: _attr_str(v) for k, v in pk.attrs.items()}
                flag = np.asarray(pk["flag"][:], dtype=int) if "flag" in pk else np.zeros(0, int)
                out["n_peaks"] = int(flag.size)
                out["n_good"] = int(np.sum(flag == 0))
                out["n_flagged_peaks"] = int(np.sum(flag != 0))
                if "counts" in pk and out["n_frames"]:
                    counts = np.asarray(pk["counts"][:], dtype=float)
                    out["peaks_per_frame_mean"] = float(counts.mean()) if counts.size else 0.0

            rg = h5.get("residual")
            if rg is not None:
                out["has_residual"] = "clean" in rg
                out["residual_attrs"] = {k: _attr_str(v) for k, v in rg.attrs.items()}
                rpk = rg.get("peaks")
                if rpk is not None and "center" in rpk:
                    out["n_residual_peaks"] = int(rpk["center"].shape[0])

            unk = h5.get("unknowns")
            if unk is not None:
                out["has_unknowns"] = True
                out["unknowns_attrs"] = {k: _attr_str(v) for k, v in unk.attrs.items()}
                obs = unk.get("obs")
                if obs is not None and "center" in obs:
                    out["n_unknown_obs"] = int(obs["center"].shape[0])
                tracks = unk.get("tracks")
                if tracks is not None and "id" in tracks:
                    out["n_unknown_tracks"] = int(tracks["id"].shape[0])
                clusters = unk.get("clusters")
                if clusters is not None and "id" in clusters:
                    out["n_unknown_clusters"] = int(clusters["id"].shape[0])

            out["anomalies"] = _detect_anomalies(out)
    except Exception as e:
        out["anomalies"].append(f"Failed to read HDF5: {e!r}")
    return out


def _detect_anomalies(out: Dict[str, Any]) -> List[str]:
    a: List[str] = []
    if not out.get("has_background"):
        a.append("No /background/clean — run Step 1 (background separation) first.")
    if out.get("radial") is None:
        a.append("No /radial axis — x-values unknown.")
    contam = out.get("contamination")
    if contam is not None and contam.size:
        hi = int(np.sum(contam > np.median(contam) + 5.0 * (np.median(np.abs(contam - np.median(contam))) or 1.0)))
        if hi:
            a.append(f"{hi} frame(s) have outlier diamond-contamination scores — inspect on the Review tab.")
    if not out.get("has_peaks"):
        a.append("No /peaks — run Step 2 (peak fitting) to populate the peak map.")
    elif out.get("n_peaks") and not out.get("n_good"):
        a.append("Peaks fitted but none passed (all flagged) — loosen min_snr / max_chi2.")
    if not a:
        a.append("No anomalies detected.")
    return a


def cake_for_frame(reduced_h5: "str | Path", frame_index: int) -> Dict[str, Any]:
    """The 2D cake (azimuth × radial) for ``frame_index``, read from a reduced
    file, or ``ok=False`` if none is stored.

    Cakes are NOT in the analysis file; they live in the *reduced* file under
    ``/cakes`` (typically a subset of frames, mapped by ``/cakes/frame_index``).
    Returns ``{ok, error, cake, radial, azimuthal, unit, frame_index}``.
    """
    out: Dict[str, Any] = {"ok": False, "error": "", "cake": None, "radial": None,
                           "azimuthal": None, "unit": "", "frame_index": int(frame_index)}
    p = Path(reduced_h5).expanduser()
    if not str(reduced_h5) or not p.is_file():
        out["error"] = "Source reduced file not found (cakes live there, not the analysis file)."
        return out
    try:
        import h5py  # type: ignore
        with h5py.File(str(p), "r") as h5:
            out["unit"] = _attr_str(h5.attrs.get("unit", ""))
            cakes = h5.get("cakes")
            if cakes is None or "intensity" not in cakes or cakes["intensity"].shape[0] == 0:
                out["error"] = "No cakes in the reduced file (re-run reduction with cakes enabled)."
                return out
            n_cakes = int(cakes["intensity"].shape[0])
            j = None
            if "frame_index" in cakes:
                fi = np.asarray(cakes["frame_index"][:])
                w = np.where(fi == int(frame_index))[0]
                j = int(w[0]) if w.size else None
            elif int(frame_index) < n_cakes:
                j = int(frame_index)
            if j is None:
                out["error"] = f"No cake stored for frame {frame_index}."
                return out
            out["cake"] = np.asarray(cakes["intensity"][j], dtype=float)
            if "radial" in cakes:
                out["radial"] = np.asarray(cakes["radial"][:], dtype=float)
            if "azimuthal" in cakes:
                out["azimuthal"] = np.asarray(cakes["azimuthal"][:], dtype=float)
            out["ok"] = True
    except Exception as e:
        out["error"] = f"Failed to read cake: {e!r}"
    return out


def frame_data(h5_path: "str | Path", frame_index: int) -> Dict[str, Any]:
    """Reconstruct everything needed to plot one frame.

    Returns ``{ok, error, index, filename, unit, radial, clean, baseline,
    spot_residual, robust, mean, hybrid, sigmaclip, residual, contamination,
    peaks, residual_peaks, unknown_obs}`` where each peak/unknown list contains
    per-frame rows from the corresponding HDF5 group. ``hybrid``/``sigmaclip``
    are None when their inputs are absent.
    """
    p = Path(h5_path).expanduser()
    out: Dict[str, Any] = {
        "ok": False, "error": "", "index": int(frame_index), "filename": "",
        "unit": "", "radial": None, "clean": None, "baseline": None,
        "spot_residual": None, "robust": None, "mean": None,
        "sigmaclip": None, "hybrid": None, "residual": None,
        "contamination": None, "peaks": [], "residual_peaks": [],
        "unknown_obs": [],
    }
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
            out["unit"] = str(h5.attrs.get("unit", ""))
            if "radial" in h5:
                out["radial"] = np.asarray(h5["radial"][:], dtype=float)
            bg = h5.get("background")
            if bg is None or "clean" not in bg:
                out["error"] = "No /background/clean in this file."
                return out
            n = bg["clean"].shape[0]
            i = int(frame_index)
            if not (0 <= i < n):
                out["error"] = f"Frame {i} out of range (0..{n - 1})."
                return out
            clean = np.asarray(bg["clean"][i], dtype=float)
            baseline = np.asarray(bg["baseline"][i], dtype=float) if "baseline" in bg else None
            spots = np.asarray(bg["spot_residual"][i], dtype=float) if "spot_residual" in bg else None
            sigres = np.asarray(bg["sigmaclip_residual"][i], dtype=float) if "sigmaclip_residual" in bg else None
            out["clean"] = clean
            out["baseline"] = baseline
            out["spot_residual"] = spots
            if baseline is not None:
                out["robust"] = clean + baseline                # robust = clean + baseline
                if spots is not None:
                    out["mean"] = out["robust"] + spots          # mean = robust + spot_residual
            if spots is not None:                                # hybrid fit source (default)
                from .peaks import winsorize_excess
                out["hybrid"] = clean + winsorize_excess(spots)
            if sigres is not None:                               # reduce-side trimmed mean
                out["sigmaclip"] = clean + sigres

            rg = h5.get("residual")
            if rg is not None and "clean" in rg:
                rclean = rg["clean"]
                if i < rclean.shape[0]:
                    out["residual"] = np.asarray(rclean[i], dtype=float)

            frames = h5.get("frames")
            names = _read_names(frames)
            if names is not None and i < len(names):
                out["filename"] = names[i]
            if frames is not None and "contamination" in frames:
                c = np.asarray(frames["contamination"][:], dtype=float)
                if i < c.size:
                    out["contamination"] = float(c[i])

            pk = h5.get("peaks")
            if pk is not None and "frame" in pk and "center" in pk:
                fr = np.asarray(pk["frame"][:], dtype=int)
                sel = np.where(fr == i)[0]
                if sel.size:
                    cols = {c: np.asarray(pk[c][:]) for c in _PEAK_COLS if c in pk}
                    for j in sel:
                        out["peaks"].append({c: (float(cols[c][j]) if c != "flag" else int(cols[c][j]))
                                             for c in cols})
            if rg is not None and "peaks" in rg:
                rpk = rg["peaks"]
                if "frame" in rpk and "center" in rpk:
                    fr = np.asarray(rpk["frame"][:], dtype=int)
                    sel = np.where(fr == i)[0]
                    if sel.size:
                        cols = {c: np.asarray(rpk[c][:]) for c in _PEAK_COLS if c in rpk}
                        for j in sel:
                            out["residual_peaks"].append({
                                c: (float(cols[c][j]) if c != "flag" else int(cols[c][j]))
                                for c in cols
                            })

            unk = h5.get("unknowns")
            obs = unk.get("obs") if unk is not None else None
            if obs is not None and "frame" in obs and "center" in obs:
                fr = np.asarray(obs["frame"][:], dtype=int)
                sel = np.where(fr == i)[0]
                if sel.size:
                    track = (np.asarray(obs["track"][:], dtype=int)
                             if "track" in obs else np.full(fr.size, -1, int))
                    center = np.asarray(obs["center"][:], dtype=float)
                    amp = (np.asarray(obs["amplitude"][:], dtype=float)
                           if "amplitude" in obs else np.full(fr.size, np.nan))
                    fwhm = (np.asarray(obs["fwhm"][:], dtype=float)
                            if "fwhm" in obs else np.full(fr.size, np.nan))
                    cluster_of_track: Dict[int, int] = {}
                    tr = unk.get("tracks") if unk is not None else None
                    if tr is not None and "id" in tr and "cluster" in tr:
                        ids = np.asarray(tr["id"][:], dtype=int)
                        clusters = np.asarray(tr["cluster"][:], dtype=int)
                        cluster_of_track = {
                            int(t): int(c) for t, c in zip(ids, clusters)
                        }
                    for j in sel:
                        tid = int(track[j])
                        out["unknown_obs"].append({
                            "track": tid,
                            "cluster": cluster_of_track.get(tid, -1),
                            "center": float(center[j]),
                            "amplitude": float(amp[j]),
                            "fwhm": float(fwhm[j]),
                        })
            out["ok"] = True
    except Exception as e:
        out["error"] = f"Failed to read HDF5: {e!r}"
    return out


def peak_map(h5_path: "str | Path", good_only: bool = False) -> Dict[str, Any]:
    """Flat peak table across the whole series, for the heatmap/scatter view.

    Returns ``{ok, error, n_frames, unit, frame, center, amplitude, fwhm, area,
    flag}`` (arrays). With ``good_only`` the flagged peaks (flag != 0) are
    dropped.
    """
    p = Path(h5_path).expanduser()
    out: Dict[str, Any] = {
        "ok": False, "error": "", "n_frames": 0, "unit": "",
        "frame": None, "center": None, "amplitude": None, "fwhm": None,
        "area": None, "flag": None,
    }
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
            out["unit"] = str(h5.attrs.get("unit", ""))
            bg = h5.get("background")
            if bg is not None and "clean" in bg:
                out["n_frames"] = int(bg["clean"].shape[0])
            pk = h5.get("peaks")
            if pk is None or "center" not in pk:
                out["error"] = "No /peaks — run Step 2 (peak fitting) first."
                return out
            frame = np.asarray(pk["frame"][:], dtype=int) if "frame" in pk else np.zeros(0, int)
            center = np.asarray(pk["center"][:], dtype=float)
            amplitude = np.asarray(pk["amplitude"][:], dtype=float) if "amplitude" in pk else np.zeros_like(center)
            fwhm = np.asarray(pk["fwhm"][:], dtype=float) if "fwhm" in pk else np.zeros_like(center)
            area = np.asarray(pk["area"][:], dtype=float) if "area" in pk else np.zeros_like(center)
            flag = np.asarray(pk["flag"][:], dtype=int) if "flag" in pk else np.zeros_like(center, int)
            if good_only:
                m = flag == 0
                frame, center, amplitude, fwhm, area, flag = (
                    frame[m], center[m], amplitude[m], fwhm[m], area[m], flag[m])
            out.update({"ok": True, "frame": frame, "center": center,
                        "amplitude": amplitude, "fwhm": fwhm, "area": area, "flag": flag})
    except Exception as e:
        out["error"] = f"Failed to read HDF5: {e!r}"
    return out


def identify_tracks(h5_path: "str | Path") -> Dict[str, Any]:
    """Per-phase Step-3a results for the pressure-vs-frame view.

    Returns ``{ok, error, unit, wavelength, p_min, p_max, n_frames, phases}``
    where ``phases`` is a list of ``{name, category, has_eos, pressure_model,
    prior_penalized, n_pred, pressure, score, confidence, recall, precision,
    n_matched, prior_penalty}`` (the array fields are all length n_frames).
    ``pressure_model`` is ``eos`` | ``axial_eos`` | ``no_eos`` (older files may
    carry the pre-rename ``ambient_only``; callers should treat it as ``no_eos``).
    """
    p = Path(h5_path).expanduser()
    out: Dict[str, Any] = {"ok": False, "error": "", "unit": "", "wavelength": 0.0,
                           "p_min": 0.0, "p_max": 0.0, "n_frames": 0, "phases": []}
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
            gid = h5.get("identify")
            if gid is None:
                out["error"] = "No /identify — run Step 3a (EOS phase matching) first."
                return out
            out["unit"] = str(gid.attrs.get("unit", ""))
            out["wavelength"] = float(gid.attrs.get("wavelength", 0.0) or 0.0)
            out["p_min"] = float(gid.attrs.get("p_min", 0.0) or 0.0)
            out["p_max"] = float(gid.attrs.get("p_max", 0.0) or 0.0)
            for key in gid:
                g = gid[key]
                if not hasattr(g, "keys") or "pressure" not in g:
                    continue
                rec = {
                    "name": str(g.attrs.get("name", key)),
                    "category": str(g.attrs.get("category", "")),
                    "has_eos": bool(g.attrs.get("has_eos", False)),
                    "pressure_model": str(g.attrs.get("pressure_model",
                        "eos" if g.attrs.get("has_eos", False) else "no_eos")),
                    "pressure_assumption": str(g.attrs.get("pressure_assumption", "")),
                    "prior_penalized": bool(g.attrs.get("prior_penalized", False)),
                    "n_pred": int(g.attrs.get("n_pred", 0)),
                    "pressure": np.asarray(g["pressure"][:], dtype=float),
                    "score": np.asarray(g["score"][:], dtype=float) if "score" in g else None,
                    "confidence": np.asarray(g["confidence"][:], dtype=float) if "confidence" in g else None,
                    "recall": np.asarray(g["recall"][:], dtype=float) if "recall" in g else None,
                    "precision": np.asarray(g["precision"][:], dtype=float) if "precision" in g else None,
                    "n_matched": np.asarray(g["n_matched"][:], dtype=int) if "n_matched" in g else None,
                    "prior_penalty": np.asarray(g["prior_penalty"][:], dtype=float) if "prior_penalty" in g else None,
                }
                out["n_frames"] = max(out["n_frames"], int(rec["pressure"].size))
                out["phases"].append(rec)
            out["phases"].sort(key=lambda r: r["name"].lower())
            out["ok"] = True
    except Exception as e:
        out["error"] = f"Failed to read HDF5: {e!r}"
    return out


def structure_report(review: Dict[str, Any]) -> str:
    """Human-readable text block for the GUI."""
    L: List[str] = [f"File: {review['path']}"]
    if not review.get("ok_to_read"):
        L.append("")
        L.append("ANOMALIES:")
        L += [f"  - {x}" for x in review.get("anomalies", [])]
        return "\n".join(L)
    L.append(f"Frames: {review['n_frames']}    Bins: {review['n_bins']}    "
             f"Unit: {review.get('unit') or '?'}")
    L.append(f"Step 1 background: {'yes' if review['has_background'] else 'no'}    "
             f"Step 2 peaks: {'yes' if review['has_peaks'] else 'no'}")
    L.append(f"Residual: {'yes' if review.get('has_residual') else 'no'}    "
             f"Unknowns: {'yes' if review.get('has_unknowns') else 'no'}")
    if review.get("has_peaks"):
        L.append(f"Peaks: {review['n_good']} good / {review['n_peaks']} total "
                 f"({review['n_flagged_peaks']} flagged)    "
                 f"mean {review['peaks_per_frame_mean']:.1f}/frame")
    if review.get("has_residual"):
        L.append(f"Residual peaks: {review.get('n_residual_peaks', 0)}")
    if review.get("has_unknowns"):
        L.append(f"Unknowns: {review.get('n_unknown_tracks', 0)} track(s), "
                 f"{review.get('n_unknown_clusters', 0)} cluster(s), "
                 f"{review.get('n_unknown_obs', 0)} observation(s)")
    if review.get("source_reduced"):
        L.append(f"Source reduced file: {review['source_reduced']}")
    L.append("")
    L.append("HDF5 structure:")
    L += review.get("structure_lines", [])
    attrs = review.get("attrs", {})
    if attrs:
        L.append("")
        L.append("Root attributes:")
        for k, v in attrs.items():
            L.append(f"  {k}: {v}")
    if review.get("peak_attrs"):
        L.append("")
        L.append("Peak-fit parameters:")
        for k, v in review["peak_attrs"].items():
            L.append(f"  {k}: {v}")
    if review.get("residual_attrs"):
        L.append("")
        L.append("Residual parameters:")
        for k, v in review["residual_attrs"].items():
            L.append(f"  {k}: {v}")
    if review.get("unknowns_attrs"):
        L.append("")
        L.append("Unknown-clustering parameters:")
        for k, v in review["unknowns_attrs"].items():
            L.append(f"  {k}: {v}")
    L.append("")
    L.append("Anomalies:")
    L += [f"  - {x}" for x in review.get("anomalies", [])]
    return "\n".join(L)


def review_analysis(h5_path: "str | Path") -> Dict[str, Any]:
    """Print the structure/anomaly report for an analysis HDF5 and return the
    full inspection dict."""
    review = inspect_analysis(h5_path)
    print(structure_report(review), flush=True)
    return review
