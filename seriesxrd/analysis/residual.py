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
from .peaks import pseudo_voigt, fit_pattern, build_fit_source
from .phases import Phase

SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Per-frame attribution + subtraction
# ---------------------------------------------------------------------------

def attribute_peaks(obs_d: np.ndarray,
                    phase_preds: "Dict[str, np.ndarray]",
                    rel_tol: float) -> "Tuple[List[str], np.ndarray]":
    """Assign observed peaks to present phases by greedy ONE-TO-ONE matching.

    ``phase_preds`` maps phase name -> predicted d-spacings (already scaled to
    the frame's pressure). Candidate (observed peak, predicted line) pairs within
    ``rel_tol`` (the same tolerance Step 3a scores with) are consumed closest
    first, and **each predicted reflection may explain at most one observed
    peak** (and each observed peak at most one reflection). This mirrors the
    one-to-one matching identification uses: without it, one predicted line would
    label a whole cluster of nearby peaks as "explained", and subtraction would
    then erase real split/overlapped/unknown signal sitting beside a known line.
    Returns ``(labels, explained)`` where ``labels[i]`` is the phase name or
    ``""`` and ``explained`` is a bool mask.
    """
    obs_d = np.asarray(obs_d, float)
    n = obs_d.size
    labels = [""] * n
    explained = np.zeros(n, dtype=bool)
    if n == 0 or not phase_preds:
        return labels, explained
    tol = float(rel_tol)
    # All (relative gap, obs index, phase, line index) pairs within tolerance.
    cands: "List[Tuple[float, int, str, int]]" = []
    for name, pred in phase_preds.items():
        pred = np.asarray(pred, float)
        for j, pj in enumerate(pred):
            if not np.isfinite(pj) or pj <= 0:
                continue
            for i in range(n):
                oi = obs_d[i]
                if not np.isfinite(oi) or oi <= 0:
                    continue
                rel = abs(oi - pj) / oi
                if rel < tol:
                    cands.append((rel, i, name, j))
    cands.sort(key=lambda c: c[0])
    used_obs: set = set()
    used_line: set = set()                          # (phase, line index)
    for rel, i, name, j in cands:
        if i in used_obs or (name, j) in used_line:
            continue
        used_obs.add(i)
        used_line.add((name, j))
        labels[i] = name
        explained[i] = True
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
                  pressure: float, temperature: "Optional[float]" = None
                  ) -> np.ndarray:
    """Predicted d-spacings for one phase at ``pressure`` (and temperature, when
    known) using cached ambient reflections (so no pymatgen call). NaN pressure
    → ambient (0 GPa). Temperature goes through the same thermal-expansion seam
    Step 3a matched with, so subtraction happens at the identified lattice."""
    P = 0.0 if not np.isfinite(pressure) else float(max(pressure, 0.0))
    hkls = [_parse_hkl(h) for h in refl_hkl] if refl_hkl is not None else None
    return predicted_d(phase, np.asarray(refl_d, float), hkls, P, temperature)


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
    min_snr: "Optional[float]" = None,
    window_factor: "Optional[float]" = None,
    max_chi2: "Optional[float]" = None,
    min_prominence_snr: "Optional[float]" = None,
    edge_bins: "Optional[int]" = None,
    fit_min: "Optional[float]" = None,
    fit_max: "Optional[float]" = None,
    min_fwhm_bins: "Optional[float]" = None,
    local_baseline_bins: "Optional[int]" = None,
    min_matched: int = DEFAULT_MIN_MATCHED,
    allow_sparse: bool = False,
    out_h5: "Optional[str | Path]" = None,
) -> Dict[str, Any]:
    """Subtract identified phases per frame and write the residual + attribution.

        /residual  attrs: schema_version, seen_conf, rel_tol, peak-fit knobs,
                          min_matched, allow_sparse, source, phases
        /residual/clean             (N, N_bins)  fit source minus explained-phase
                                                 peaks (source = /peaks.attrs source)
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
        # Rebuild the SAME channel Step 2 fit the peaks on (auto → sigmaclip/hybrid
        # by default, not clean). Subtracting the fitted pseudo-Voigts from a
        # *different* channel would over-subtract or dig negative holes; the
        # amplitudes belong to this source.
        pkg = h5.get("peaks")
        want_source = str(pkg.attrs.get("source", "clean")) if pkg is not None else "clean"
        hybrid_spike_bins = int(pkg.attrs.get("hybrid_spike_bins", 5)) if pkg is not None else 5

        def _attr_float(name: str, fallback: "Optional[float]") -> "Optional[float]":
            val = pkg.attrs.get(name, fallback) if pkg is not None else fallback
            if val is None:
                return None
            try:
                f = float(val)
            except Exception:
                return fallback
            return f if np.isfinite(f) else None

        def _attr_int(name: str, fallback: "Optional[int]") -> "Optional[int]":
            val = pkg.attrs.get(name, fallback) if pkg is not None else fallback
            if val is None:
                return None
            try:
                return int(val)
            except Exception:
                return fallback

        r_min_snr = float(min_snr if min_snr is not None
                          else (_attr_float("min_snr", 5.0) or 5.0))
        r_window = float(window_factor if window_factor is not None
                         else (_attr_float("window_factor", 3.0) or 3.0))
        r_chi2 = float(max_chi2 if max_chi2 is not None
                       else (_attr_float("max_chi2", 25.0) or 25.0))
        r_prom = (float(min_prominence_snr) if min_prominence_snr is not None
                  else _attr_float("min_prominence_snr", None))
        r_edge = int(edge_bins if edge_bins is not None
                     else (_attr_int("edge_bins", 0) or 0))
        r_fit_min = (float(fit_min) if fit_min is not None
                     else _attr_float("fit_min", None))
        r_fit_max = (float(fit_max) if fit_max is not None
                     else _attr_float("fit_max", None))
        r_fwhm = float(min_fwhm_bins if min_fwhm_bins is not None
                       else (_attr_float("min_fwhm_bins", 0.0) or 0.0))
        # Step 2's local-baseline detrend is useful for weak-peak detection, but
        # applying it again during residual re-fitting is very expensive on large
        # frame stacks. Keep the residual pass undetrended unless a caller opts in.
        r_detrend = int(local_baseline_bins if local_baseline_bins is not None else 0)
        spot_res = np.asarray(bg["spot_residual"][:], dtype=float) if "spot_residual" in bg else None
        sc_res = np.asarray(bg["sigmaclip_residual"][:], dtype=float) if "sigmaclip_residual" in bg else None
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
        temperature = (np.asarray(frames["temperature"][:], dtype=float)
                       if frames is not None and "temperature" in frames else None)

    n, nb = clean.shape
    if excluded is None or excluded.size != n:
        excluded = np.zeros(n, dtype=bool)

    # The base we subtract from and re-fit: the recorded Step-2 fit source,
    # falling back to clean if a needed residual channel is missing.
    try:
        fit_src, used_source = build_fit_source(
            want_source, clean, spot_residual=spot_res, sigmaclip_residual=sc_res,
            hybrid_spike_bins=hybrid_spike_bins)
        fit_src = np.asarray(fit_src, dtype=float)
    except ValueError:
        fit_src, used_source = clean, "clean"

    P = pk["frame"].size
    peak_phase = np.array([""] * P, dtype=object)
    explained_counts = np.zeros(n, dtype="i4")
    unexplained_counts = np.zeros(n, dtype="i4")
    residual = np.array(fit_src, dtype="f4")      # default: unchanged where nothing removed

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
        cln = fit_src[i]
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
                T_i = (float(temperature[i]) if temperature is not None
                       and temperature.size == n and np.isfinite(temperature[i]) else None)
                preds[name] = _phase_pred_d(info["phase"], info["refl_d"],
                                            info["refl_hkl"], info["press"][i], T_i)
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
        cands = fit_pattern(
            radial,
            residual[i],
            min_snr=r_min_snr,
            window_factor=r_window,
            max_chi2=r_chi2,
            min_prominence_snr=r_prom,
            edge_bins=r_edge,
            fit_min=r_fit_min,
            fit_max=r_fit_max,
            min_fwhm_bins=r_fwhm,
            local_baseline_bins=r_detrend,
            keep_flagged=False,
        )
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
                "rel_tol": float(rel_tol), "min_snr": float(r_min_snr),
                "window_factor": float(r_window), "max_chi2": float(r_chi2),
                "min_prominence_snr": float(r_prom) if r_prom is not None else np.nan,
                "edge_bins": int(r_edge),
                "fit_min": float(r_fit_min) if r_fit_min is not None else np.nan,
                "fit_max": float(r_fit_max) if r_fit_max is not None else np.nan,
                "min_fwhm_bins": float(r_fwhm),
                "local_baseline_bins": int(r_detrend),
                "min_matched": int(min_matched), "allow_sparse": bool(allow_sparse),
                "source": str(used_source), "phases": ", ".join(by_name),
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
        "fit_source": str(used_source), "min_snr": float(r_min_snr),
        "min_prominence_snr": float(r_prom) if r_prom is not None else None,
        "window_factor": float(r_window), "max_chi2": float(r_chi2),
        "edge_bins": int(r_edge),
        "fit_min": float(r_fit_min) if r_fit_min is not None else None,
        "fit_max": float(r_fit_max) if r_fit_max is not None else None,
        "min_fwhm_bins": float(r_fwhm),
        "local_baseline_bins": int(r_detrend),
        "n_explained": n_explained, "n_unexplained": n_unexplained,
        "n_residual_peaks": int(len(rd_frame)),
        "phases": list(by_name),
    }
    print(f"[RESIDUAL] done -> {dst}  ({n_explained} peaks removed as known, "
          f"{n_unexplained} unexplained, {len(rd_frame)} residual peaks)", flush=True)
    return manifest
