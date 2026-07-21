"""Semi-quantitative phase fractions from the Step-3a peak attribution.

Roadmap item "Quantitative phase fractions" (docs/roadmap.md, "Start with RIR
/ normalized-intensity fractions ... Rietveld-quality quantification comes
later, likely via the refinement export [docs/roadmap.md "Refinement
hand-off"] rather than an in-house Rietveld engine").

**These are SEMI-QUANTITATIVE intensity-share fractions, not a refinement
result.** For each frame, every good (unflagged) fitted peak that Step 3a-
removal (:mod:`analysis.residual`) attributed to a known phase (``/peaks/phase``
!= ``""``) contributes its integrated area (``/peaks/area``) to that phase's
share for that frame:

    fraction_p(frame) = A_p(frame) / sum_q A_q(frame)

summed over attributed peaks only — an unexplained (unattributed) peak
contributes to neither the numerator nor the denominator, so it does not
silently dilute the known phases' shares. A frame with no attributed peaks at
all gets an all-NaN row (there is nothing to apportion).

This is **not** a quantitative (weight/volume-fraction) measurement: texture
(preferred orientation), absorption/microabsorption, and structure-factor
differences between phases are **not corrected**. Two phases with equal molar
amounts but different diffracting power will not get equal fractions here.
Supplying a reference intensity ratio (RIR, ``I/Icor``) per phase via
``use_rir`` divides out that phase's relative diffracting power and improves
comparability across phases, but it is still an intensity-based estimate, not
Rietveld refinement. For a quantitative result, export the identified phases
and patterns and refine in a dedicated tool (see docs/roadmap.md "Refinement
hand-off").

Pure numpy + h5py (h5py imported lazily inside functions). No pymatgen, no GUI.

References: F. H. Chung, J. Appl. Cryst. 7 (1974) 519 (the matrix-flushing /
RIR quantification this approximates); C. R. Hubbard & R. L. Snyder, Powder
Diffr. 3 (1988) 74 (the RIR = I/Icor convention expected in ``use_rir``).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.config import VERSION
from ..core.provenance import manifest_provenance, write_step_provenance

SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _decode_str_array(raw) -> np.ndarray:
    """h5py variable-length utf-8 strings -> a numpy object array of ``str``."""
    return np.array(
        [s.decode("utf-8", "replace") if isinstance(s, (bytes, bytearray)) else str(s)
         for s in raw],
        dtype=object)


def _valid_rir(value: Any) -> bool:
    """True if ``value`` is a finite, positive RIR."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v > 0.0)


# ---------------------------------------------------------------------------
# Fraction computation
# ---------------------------------------------------------------------------

def phase_fractions(analysis_h5: "str | Path", *,
                    use_rir: "Optional[Dict[str, float]]" = None,
                    min_conf: "Optional[float]" = None) -> Dict[str, Any]:
    """Per-frame intensity-share (or RIR-weighted) phase fractions.

    Reads the Step-3a-removal peak attribution (``/peaks/phase``, written by
    :func:`analysis.residual.run_residual`) and the fitted peak areas
    (``/peaks/area``). For each frame, sums the areas of good peaks
    (``/peaks/flag == 0``) grouped by attributed phase, then normalizes each
    phase's sum by the frame's total attributed area. Peaks with an empty
    phase label (unattributed / unexplained) are excluded from both the
    numerator and the denominator. A frame with zero attributed area (no
    attributed peaks, or all excluded) gets an all-NaN row.

    ``use_rir``: optional ``{phase_name: RIR}`` (reference intensity ratio,
    ``I/Icor``). When given, each attributed peak's area is weighted
    ``A_p / RIR_p`` before summing/normalizing, so phases with very different
    diffracting power become more comparable. A phase named in the *result*
    but missing (or with a non-finite/non-positive value) from ``use_rir``
    falls back to ``RIR=1`` (i.e. unweighted) — this is reported per-phase in
    the returned ``rir_used`` dict so a caller can tell which phases actually
    got a correction. When ``use_rir`` is ``None`` the result is a pure
    intensity share (``method="intensity_share"``); otherwise
    ``method="rir"``.

    ``min_conf``: optional confidence gate. When given, a peak's phase
    attribution is only honoured in frames where that phase's Step-3a
    ``/identify/<phase>/confidence`` at that frame exceeds ``min_conf``;
    otherwise the peak is treated as unattributed for that frame (dropped
    from both numerator and denominator there). This only gates phases for
    which ``/identify`` confidence is actually present — if ``/identify`` (or
    a given phase's group in it) is absent, ``min_conf`` has no effect on that
    phase (there is nothing to gate on).

    Returns ``{ok, error, n_frames, phases, fractions, rir_used, method}``:
      ``phases``    list of attributed phase names (sorted), the column order
                    of ``fractions``
      ``fractions`` ``(n_frames, len(phases))`` float array; NaN where a
                    frame had no attributed peaks (or none survived
                    ``min_conf``)
      ``rir_used``  ``{phase_name: bool}`` — True if a valid RIR from
                    ``use_rir`` was applied to that phase, False if it fell
                    back to RIR=1 (including the whole intensity-share case)
    """
    import h5py  # type: ignore

    p = Path(analysis_h5).expanduser()
    method = "rir" if use_rir is not None else "intensity_share"
    out: Dict[str, Any] = {"ok": False, "error": "", "n_frames": 0, "phases": [],
                           "fractions": None, "rir_used": {}, "method": method}
    if not p.is_file():
        out["error"] = f"File does not exist: {p}"
        return out

    try:
        with h5py.File(str(p), "r") as h5:
            pk = h5.get("peaks")
            if pk is None or "frame" not in pk:
                out["error"] = "Analysis file lacks /peaks — run Step 2 (peak fitting) first."
                return out
            if "area" not in pk:
                out["error"] = ("Analysis file lacks /peaks/area — run Step 2 profile "
                                "fitting (integrated peak areas are required for fractions).")
                return out
            if "phase" not in pk:
                out["error"] = ("Analysis file lacks /peaks/phase — run Step 3a-removal "
                                "(analysis.residual.run_residual) first to attribute peaks "
                                "to identified phases.")
                return out

            frame = np.asarray(pk["frame"][:], dtype=int)
            area = np.asarray(pk["area"][:], dtype=float)
            flag = (np.asarray(pk["flag"][:], dtype=int) if "flag" in pk
                    else np.zeros(frame.size, dtype=int))
            phase = _decode_str_array(pk["phase"][:])

            frames_grp = h5.get("frames")
            if frames_grp is not None and "filename" in frames_grp:
                n = int(frames_grp["filename"].shape[0])
            elif frame.size:
                n = int(frame.max()) + 1
            else:
                n = 0

            good = (flag == 0)
            attributed = good & (phase != "") & np.isfinite(area)
            names = sorted({str(nm) for nm in phase[attributed]})
            n_phases = len(names)
            name_idx = {nm: i for i, nm in enumerate(names)}

            sel = np.nonzero(attributed)[0]
            frame_sel = frame[sel]
            phase_sel = phase[sel]
            area_sel = area[sel]
            phase_idx_sel = (np.array([name_idx[str(nm)] for nm in phase_sel], dtype=int)
                              if sel.size else np.zeros(0, dtype=int))

            rir_used = {nm: False for nm in names}
            weight_sel = area_sel.copy()
            if use_rir and sel.size:
                for nm in names:
                    if _valid_rir(use_rir.get(nm)):
                        rir_used[nm] = True
                rir_vec = np.array(
                    [float(use_rir.get(nm)) if _valid_rir(use_rir.get(nm)) else 1.0
                     for nm in phase_sel], dtype=float)
                weight_sel = area_sel / rir_vec

            if min_conf is not None and n_phases and n > 0 and sel.size:
                idg = h5.get("identify")
                # +inf sentinel = "no confidence data for this phase" -> never
                # gated; a phase that HAS /identify data uses its real value.
                conf_mat = np.full((n, n_phases), np.inf, dtype=float)
                if idg is not None:
                    from .identify import _h5_safe
                    for j, nm in enumerate(names):
                        g = idg.get(_h5_safe(nm))
                        if g is not None and "confidence" in g:
                            carr = np.asarray(g["confidence"][:], dtype=float)
                            m = min(n, carr.size)
                            conf_mat[:m, j] = carr[:m]
                in_range = (frame_sel >= 0) & (frame_sel < n)
                keep = np.zeros(sel.size, dtype=bool)
                keep[in_range] = (conf_mat[frame_sel[in_range], phase_idx_sel[in_range]]
                                  > float(min_conf))
                frame_sel = frame_sel[keep]
                phase_idx_sel = phase_idx_sel[keep]
                weight_sel = weight_sel[keep]

            sums = np.zeros((n, n_phases), dtype=float)
            if n_phases and n > 0 and frame_sel.size:
                in_range2 = (frame_sel >= 0) & (frame_sel < n)
                np.add.at(sums, (frame_sel[in_range2], phase_idx_sel[in_range2]),
                          weight_sel[in_range2])
            totals = sums.sum(axis=1) if n_phases else np.zeros(n)
            fractions = np.full((n, n_phases), np.nan, dtype=float)
            if n_phases:
                has_any = totals > 0
                fractions[has_any] = sums[has_any] / totals[has_any, None]

        out.update({"ok": True, "n_frames": n, "phases": names,
                   "fractions": fractions, "rir_used": rir_used, "method": method})
    except Exception as e:
        out["error"] = f"Failed to compute phase fractions: {e!r}"
    return out


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------

def _write_fractions(analysis_h5: "str | Path", names: Sequence[str],
                     fractions: np.ndarray, method: str) -> None:
    """Atomically write ``/fractions`` (``.tmp`` + ``os.replace``, via a
    ``shutil.copy2`` of the whole file), matching the write pattern used by
    ``frame_metadata.apply_to_analysis`` and ``residual.run_residual``.
    Replaces any existing ``/fractions`` group."""
    import h5py  # type: ignore

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    tmp = src.with_name(src.name + ".tmp")
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            if "fractions" in o:
                del o["fractions"]
            g = o.create_group("fractions")
            write_step_provenance(o, "fractions",
                                  tool="seriesxrd.analysis.fractions",
                                  schema_version=SCHEMA_VERSION)
            g.attrs["schema_version"] = SCHEMA_VERSION
            g.attrs["seriesxrd_version"] = VERSION
            g.attrs["method"] = str(method)
            g.attrs["schema"] = (
                "names (P,) str phase names, column order of fractions; "
                "fractions (N_frames, P) float64 per-frame share (intensity_share: "
                "peak-area share of Step-3a-attributed peaks; rir: RIR-weighted "
                "share). NaN row = frame had no attributed peaks. Semi-quantitative "
                "-- see analysis/fractions.py module docstring.")
            str_dtype = h5py.string_dtype(encoding="utf-8")
            g.create_dataset("names", data=np.asarray(list(names), dtype=object),
                             dtype=str_dtype)
            g.create_dataset("fractions", data=np.asarray(fractions, dtype="f8"))
        os.replace(tmp, src)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def run_fractions(analysis_h5: "str | Path", *,
                  use_rir: "Optional[Dict[str, float]]" = None,
                  write: bool = True,
                  present_frac: float = 0.05) -> Dict[str, Any]:
    """Compute :func:`phase_fractions` and (optionally) persist it to the file.

    Semi-quantitative intensity-share (or RIR-weighted) phase fractions —
    see the module docstring for what this is and is not. When ``write`` is
    True, atomically writes/replaces ``/fractions`` on ``analysis_h5``:

        /fractions  attrs: schema_version, method ("intensity_share"|"rir"), schema
        /fractions/names       (P,) str    phase names, column order below
        /fractions/fractions   (N, P) f8   per-frame share; NaN row = nothing attributed

    Re-running replaces the group (idempotent — no ``.tmp`` file is left
    behind either way, success or failure).

    Returns a manifest: ``{tool_version, source, ok, error, n_frames, phases,
    method, present_frac, rir_used, rir_missing, per_phase, written}`` where
    ``per_phase`` maps each phase name to ``{mean_fraction, max_fraction,
    n_frames_present}`` (``n_frames_present`` = number of frames whose
    fraction exceeds ``present_frac``, the "is this phase actually present in
    this frame" threshold) and ``rir_missing`` lists phases that fell back to
    RIR=1 because they (or a valid RIR value) were absent from ``use_rir``
    (empty unless ``method == "rir"``).
    """
    result = phase_fractions(analysis_h5, use_rir=use_rir)
    src = Path(analysis_h5).expanduser()
    method = result.get("method", "intensity_share")
    rir_used = dict(result.get("rir_used", {}))
    manifest: Dict[str, Any] = {
        **manifest_provenance("seriesxrd.analysis.fractions", SCHEMA_VERSION),
        "source": str(src),
        "ok": bool(result.get("ok")),
        "error": str(result.get("error", "")),
        "n_frames": int(result.get("n_frames", 0)),
        "phases": list(result.get("phases", [])),
        "method": method,
        "present_frac": float(present_frac),
        "rir_used": rir_used,
        "rir_missing": ([nm for nm, used in rir_used.items() if not used]
                        if method == "rir" else []),
        "per_phase": {},
        "written": False,
    }
    if not result.get("ok"):
        return manifest

    names = result["phases"]
    fractions = result["fractions"]
    for j, nm in enumerate(names):
        col = np.asarray(fractions[:, j], dtype=float)
        finite = col[np.isfinite(col)]
        manifest["per_phase"][nm] = {
            "mean_fraction": float(np.mean(finite)) if finite.size else float("nan"),
            "max_fraction": float(np.max(finite)) if finite.size else float("nan"),
            "n_frames_present": int(np.sum(col > float(present_frac))),
        }

    if write:
        _write_fractions(src, names, fractions, method)
        manifest["written"] = True
    return manifest
