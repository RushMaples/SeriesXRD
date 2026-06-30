"""Step 3a-removal: subtract identified phases, expose what is left.

This is the "remove and reveal" step of the categorization workflow — the
reason the package exists. Steps 1-2 clean the pattern and fit its peaks;
Step 3a (``identify.py``) decides *which* known phases are present in each
frame and at what pressure. This module closes the loop:

  1. For every frame, take the phases identified as present (confidence above a
     modest bar) and predict their reflection d-spacings at that frame's
     best-fit pressure (reusing the cached ambient reflections + the same
     ``predicted_d`` compression model as Step 3a — no pymatgen needed here).
  2. Attribute each fitted peak to the present phase whose predicted line it
     matches (nearest within ``rel_tol``). A peak that matches nothing known is
     *unexplained*.
  3. Reconstruct the explained peaks from their pseudo-Voigt fits and subtract
     them from ``clean`` → a **residual pattern** that holds only the
     unexplained signal: minor phases, the sample's weaker reflections, and
     candidate unknowns that the strong identified peaks were masking.
  4. Re-detect peaks on the residual so the unexplained features have their own
     peak list (the input to Step 3c unknown-clustering).

Appends ``/residual`` to the analysis HDF5 and a per-peak ``/peaks/phase``
attribution. Depends on numpy + scipy (via :mod:`peaks`); requires that
``identify.run_identification`` has already written ``/identify``.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .identify import radial_to_d, predicted_d, _parse_hkl, _h5_safe, DEFAULT_MIN_MATCHED
from .peaks import pseudo_voigt, fit_pattern
from .phases import Phase

SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Per-frame attribution + subtraction
# ---------------------------------------------------------------------------

def attribute_peaks(obs_d: np.ndarray,
                    phase_preds: "Dict[str, np.ndarray]",
                    rel_tol: float) -> "Tuple[List[str], np.ndarray]":
    """Assign each observed peak (d-spacing) to the best-matching present phase.

    ``phase_preds`` maps phase name -> predicted d-spacings (already scaled to
    the frame's pressure). A peak is attributed to the phase with the smallest
    *relative* gap, but only if that gap is within ``rel_tol`` (the same
    tolerance Step 3a scores with). Returns ``(labels, explained)`` where
    ``labels[i]`` is the phase name or ``""`` and ``explained`` is a bool mask.
    """
    n = obs_d.size
    labels = [""] * n
    explained = np.zeros(n, dtype=bool)
    if n == 0 or not phase_preds:
        return labels, explained
    best_rel = np.full(n, np.inf)
    for name, pred in phase_preds.items():
        pred = np.asarray(pred, float)
        pred = pred[np.isfinite(pred) & (pred > 0)]
        if pred.size == 0:
            continue
        # nearest predicted line for each observed peak
        gap = np.min(np.abs(obs_d[:, None] - pred[None, :]), axis=1)
        rel = gap / np.maximum(obs_d, 1e-12)
        take = rel < best_rel
        best_rel[take] = rel[take]
        for i in np.nonzero(take)[0]:
            labels[i] = name
    explained = best_rel < float(rel_tol)
    for i in range(n):
        if not explained[i]:
            labels[i] = ""
    return labels, explained


def subtract_peaks(radial: np.ndarray, clean: np.ndarray,
                   centers, amplitudes, fwhms, etas,
                   keep: np.ndarray) -> np.ndarray:
    """``clean`` minus the pseudo-Voigt reconstruction of the ``keep`` peaks."""
    model = np.zeros_like(radial, dtype=float)
    c = np.asarray(centers, float); a = np.asarray(amplitudes, float)
    w = np.asarray(fwhms, float); e = np.asarray(etas, float)
    for i in np.nonzero(np.asarray(keep, bool))[0]:
        if w[i] > 0 and np.isfinite(c[i]):
            model += pseudo_voigt(radial, c[i], a[i], w[i], e[i])
    return np.asarray(clean, float) - model


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------

def _phase_pred_d(phase: Phase, refl_d: np.ndarray, refl_hkl,
                  pressure: float) -> np.ndarray:
    """Predicted d-spacings for one phase at ``pressure`` using cached ambient
    reflections (so no pymatgen call). NaN pressure → ambient (0 GPa)."""
    P = 0.0 if not np.isfinite(pressure) else float(max(pressure, 0.0))
    hkls = [_parse_hkl(h) for h in refl_hkl] if refl_hkl is not None else None
    return predicted_d(phase, np.asarray(refl_d, float), hkls, P)


def _read_peaks(h5):
    pk = h5.get("peaks")
    if pk is None or "frame" not in pk:
        raise ValueError("Analysis file lacks /peaks — run Step 2 first.")
    frame = np.asarray(pk["frame"][:], dtype=int)
    out = {"frame": frame}
    for c in ("center", "amplitude", "fwhm", "eta", "flag"):
        if c in pk:
            out[c] = np.asarray(pk[c][:],
                                dtype=int if c == "flag" else float)
    out.setdefault("flag", np.zeros(frame.size, int))
    return out


def run_residual(
    analysis_h5: "str | Path",
    phases: "Sequence[Phase]",
    *,
    seen_conf: float = 0.5,
    rel_tol: float = 0.01,
    min_snr: float = 5.0,
    min_matched: int = DEFAULT_MIN_MATCHED,
    allow_sparse: bool = False,
    out_h5: "Optional[str | Path]" = None,
) -> Dict[str, Any]:
    """Subtract identified phases per frame and write the residual + attribution.

        /residual  attrs: schema_version, seen_conf, rel_tol, min_snr,
                          min_matched, allow_sparse, phases
        /residual/clean             (N, N_bins)  clean minus explained-phase peaks
        /residual/explained_counts  (N,) int     #peaks attributed to a known phase
        /residual/unexplained_counts(N,) int     #good peaks left unexplained
        /residual/peaks/counts      (N,) int     re-fitted on the residual
        /residual/peaks/frame       (Q,) int
        /residual/peaks/center      (Q,)         radial-axis position (fitted)
        /residual/peaks/amplitude   (Q,)         fitted peak height
        /residual/peaks/fwhm        (Q,)         fitted FWHM
        /peaks/phase                (P,) str      per fitted peak: phase name or ""

    A phase is only subtracted from a frame when it both clears ``seen_conf`` AND
    has ≥ ``min_matched`` one-to-one matched reflections there (read from
    ``/identify/<phase>/n_matched``); ``allow_sparse`` relaxes the evidence
    requirement for marker/sparse phases. This prevents a one- or two-line
    coincidence from being subtracted as a confidently-present phase. The
    residual is then re-fit with the Step-2 pseudo-Voigt pipeline (not raw
    local-maxima detection), so the surfaced unknowns carry real fitted profiles.

    ``phases`` are the Phase objects used in Step 3a (resolve via
    ``phases.load_library``). Requires ``/identify`` to already exist.
    """
    import h5py  # type: ignore

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    dst = Path(out_h5).expanduser().resolve() if out_h5 else src
    by_name = {p.name: p for p in phases}

    with h5py.File(str(src), "r") as h5:
        unit = str(h5.attrs.get("unit", ""))
        bg = h5.get("background")
        if bg is None or "clean" not in bg:
            raise ValueError("Analysis file lacks /background/clean — run Step 1 first.")
        clean = np.asarray(bg["clean"][:], dtype=float)
        radial = np.asarray(h5["radial"][:], dtype=float) if "radial" in h5 else \
            np.arange(clean.shape[1], dtype=float)
        idg = h5.get("identify")
        if idg is None:
            raise ValueError(
                "Analysis file lacks /identify — run Step 3a (phase matching) first.")
        wavelength = float(idg.attrs.get("wavelength", 0.0) or 0.0) or None
        # Per-phase confidence, pressure and cached ambient reflections.
        pinfo: Dict[str, Dict[str, Any]] = {}
        for name, ph in by_name.items():
            g = idg.get(_h5_safe(name))
            if g is None or "refl_d" not in g:
                continue
            pinfo[name] = {
                "phase": ph,
                "conf": np.asarray(g["confidence"][:], float),
                "press": np.asarray(g["pressure"][:], float),
                "n_matched": (np.asarray(g["n_matched"][:], int)
                              if "n_matched" in g else None),
                "refl_d": np.asarray(g["refl_d"][:], float),
                "refl_hkl": [s.decode() if isinstance(s, bytes) else str(s)
                             for s in g["refl_hkl"][:]] if "refl_hkl" in g else None,
            }
        pk = _read_peaks(h5)
        frames = h5.get("frames")
        excluded = (np.asarray(frames["excluded"][:], dtype=bool)
                    if frames is not None and "excluded" in frames else None)

    n, nb = clean.shape
    if excluded is None or excluded.size != n:
        excluded = np.zeros(n, dtype=bool)

    P = pk["frame"].size
    peak_phase = np.array([""] * P, dtype=object)
    explained_counts = np.zeros(n, dtype="i4")
    unexplained_counts = np.zeros(n, dtype="i4")
    residual = np.array(clean, dtype="f4")        # default: unchanged where nothing removed

    rd_counts = np.zeros(n, dtype="i4")
    rd_frame: List[int] = []
    rd_center: List[float] = []
    rd_amp: List[float] = []
    rd_fwhm: List[float] = []

    # Row offsets of each frame's peaks in the ragged /peaks arrays.
    order = np.argsort(pk["frame"], kind="stable")
    good_all = pk["flag"] == 0

    n_present_total = 0
    for i in range(n):
        cln = clean[i]
        if excluded[i] or not np.isfinite(cln).any():
            continue
        rows = order[pk["frame"][order] == i]
        good_rows = rows[good_all[rows]]
        if good_rows.size:
            centers = pk["center"][good_rows]
            obs_d = radial_to_d(centers, unit, wavelength)
            valid = np.isfinite(obs_d) & (obs_d > 0)
            # Phases present in this frame: confidence over the bar AND enough
            # one-to-one matched reflections (the evidence gate), unless the user
            # explicitly allows sparse/marker-only matches.
            preds: Dict[str, np.ndarray] = {}
            for name, info in pinfo.items():
                if info["conf"][i] <= float(seen_conf):
                    continue
                nm = info["n_matched"]
                if (not allow_sparse) and nm is not None and nm[i] < int(min_matched):
                    continue
                preds[name] = _phase_pred_d(info["phase"], info["refl_d"],
                                            info["refl_hkl"], info["press"][i])
            n_present_total += len(preds)
            labels, explained = attribute_peaks(np.where(valid, obs_d, np.inf),
                                                preds, rel_tol)
            explained &= valid
            for j, r in enumerate(good_rows):
                peak_phase[r] = labels[j]
            explained_counts[i] = int(explained.sum())
            unexplained_counts[i] = int(good_rows.size - explained.sum())
            # Subtract the explained peaks → residual pattern.
            residual[i] = subtract_peaks(
                radial, cln, centers, pk["amplitude"][good_rows],
                pk["fwhm"][good_rows], pk["eta"][good_rows], explained).astype("f4")

        # Re-fit the residual with the Step-2 pseudo-Voigt pipeline (not raw
        # local-maxima detection) so the surfaced unknowns carry real fitted
        # profiles — the proper input for Step 3c unknown-clustering. Keep only
        # good (unflagged) peaks.
        cands = fit_pattern(radial, residual[i], min_snr=min_snr, keep_flagged=False)
        rd_counts[i] = len(cands)
        for c in cands:
            rd_frame.append(i); rd_center.append(c["center"])
            rd_amp.append(c["amplitude"]); rd_fwhm.append(c["fwhm"])

    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            if "residual" in o:
                del o["residual"]
            g = o.create_group("residual")
            g.attrs.update({
                "schema_version": SCHEMA_VERSION, "seen_conf": float(seen_conf),
                "rel_tol": float(rel_tol), "min_snr": float(min_snr),
                "min_matched": int(min_matched), "allow_sparse": bool(allow_sparse),
                "phases": ", ".join(by_name),
            })
            g.create_dataset("clean", data=residual, compression="gzip", compression_opts=1)
            g.create_dataset("explained_counts", data=explained_counts)
            g.create_dataset("unexplained_counts", data=unexplained_counts)
            gp = g.create_group("peaks")
            gp.create_dataset("counts", data=rd_counts)
            gp.create_dataset("frame", data=np.asarray(rd_frame, dtype="i4"))
            gp.create_dataset("center", data=np.asarray(rd_center, dtype="f8"))
            gp.create_dataset("amplitude", data=np.asarray(rd_amp, dtype="f8"))
            gp.create_dataset("fwhm", data=np.asarray(rd_fwhm, dtype="f8"))
            # Per-fitted-peak attribution alongside /peaks.
            if "peaks" in o:
                if "phase" in o["peaks"]:
                    del o["peaks"]["phase"]
                o["peaks"].create_dataset(
                    "phase",
                    data=np.array([str(s) for s in peak_phase], dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"))
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    n_explained = int(explained_counts.sum())
    n_unexplained = int(unexplained_counts.sum())
    manifest = {
        "tool_version": SCHEMA_VERSION, "source": str(src), "out_h5": str(dst),
        "n_frames": int(n), "seen_conf": float(seen_conf), "rel_tol": float(rel_tol),
        "n_explained": n_explained, "n_unexplained": n_unexplained,
        "n_residual_peaks": int(len(rd_frame)),
        "phases": list(by_name),
    }
    print(f"[RESIDUAL] done -> {dst}  ({n_explained} peaks removed as known, "
          f"{n_unexplained} unexplained, {len(rd_frame)} residual peaks)", flush=True)
    return manifest
