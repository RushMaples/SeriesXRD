"""Step 3a of the analysis pipeline: deterministic EOS-based phase matching.

For each frame, the fitted peak list from Step 2 (good peaks only) is matched
against each candidate phase from the reference library (see ``phases.py``).
Pressure is the single fitting parameter:

  1. Simulate the phase's reflections once at ambient conditions (d-spacings +
     relative intensities + hkl) via pymatgen.
  2. Under (isotropic) compression every d-spacing scales by the same factor
     ``s(P) = (V(P)/V0)**(1/3)`` from the phase's Birch-Murnaghan EOS, so the
     predicted pattern at pressure P is just ``d0 * s(P)`` — no re-simulation.
  3. Convert the observed peak centers to d-spacing (wavelength-free for q-axis
     data; needs the wavelength for a 2theta axis) and score the overlap with an
     intensity-weighted Gaussian kernel on |d_pred - d_obs|.
  4. Optimize P (grid + local refine) to maximize the score → per-frame, per-
     phase best-fit pressure, match score, and a confidence (fraction of the
     in-window predicted lines that are actually present).

``run_identification`` drives a whole analysis HDF5 (the one Steps 1-2 wrote)
and appends an ``/identify`` group. Requires pymatgen for the simulation step
(optional dependency); raises an instructive error if it is missing.

Depends on numpy (+ scipy for the local refine, used if available). pymatgen is
reached only through ``phases.simulate_pattern``.
"""
from __future__ import annotations

import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phases import (Phase, simulate_pattern, compression_at_pressure,
                     pymatgen_available)
from .parallel import resolve_workers, chunk_ranges

SCHEMA_VERSION = "1"

# Wavelength used only to drive pymatgen's 2theta window when extracting the
# ambient reflection list. Small + wide window => all reflections down to
# ~0.14 A are captured; the d-spacings returned are wavelength-independent.
_SIM_WAVELENGTH = 0.2
_SIM_TWO_THETA_MAX = 90.0


# ---------------------------------------------------------------------------
# Axis → d-spacing
# ---------------------------------------------------------------------------

def radial_to_d(centers, unit: str, wavelength: "Optional[float]" = None) -> np.ndarray:
    """Convert peak centers on the reduced radial axis to d-spacing (Å).

    ``unit`` is the reduced/analysis ``unit`` attribute: ``q_A^-1``, ``q_nm^-1``,
    ``2th_deg`` or ``2th_rad``. The 2theta forms need ``wavelength`` (Å).
    """
    c = np.asarray(centers, dtype=float)
    u = (unit or "").strip().lower()
    if u in ("q_a^-1", "q_a-1", "q_a", "q"):          # q in Å^-1
        return 2.0 * np.pi / c
    if u in ("q_nm^-1", "q_nm-1", "q_nm"):            # q in nm^-1 → Å^-1 = /10
        return 2.0 * np.pi / (c * 0.1)
    if u in ("2th_deg", "2th_rad"):
        if wavelength is None or wavelength <= 0:
            raise ValueError(f"wavelength (Å) is required to convert {unit} to d-spacing.")
        theta = np.radians(c) / 2.0 if u == "2th_deg" else c / 2.0
        return float(wavelength) / (2.0 * np.sin(theta))
    raise ValueError(f"Unsupported unit for d-spacing conversion: {unit!r}")


def wavelength_from_reduced(reduced_path: "str | Path") -> "Optional[float]":
    """Best-effort read of the wavelength (Å) from a reduced file's PONI text."""
    try:
        import h5py  # type: ignore
        with h5py.File(str(reduced_path), "r") as h5:
            poni = h5.attrs.get("poni_text", "")
            if isinstance(poni, bytes):
                poni = poni.decode("utf-8", "replace")
        m = re.search(r"wavelength\s*:\s*([0-9eE.+-]+)", str(poni), re.IGNORECASE)
        if m:
            wl_m = float(m.group(1))           # pyFAI stores it in metres
            return wl_m * 1e10 if wl_m < 1e-6 else wl_m
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Reflections & pressure scaling
# ---------------------------------------------------------------------------

def phase_reflections(phase: Phase, *, max_reflections: int = 40,
                      min_rel_intensity: float = 0.01
                      ) -> "Tuple[np.ndarray, np.ndarray, List[str]]":
    """Ambient (d, weight, hkl) reflection list for a phase, strongest first.

    ``weight`` is the relative intensity normalised to its max. Keeps the
    strongest ``max_reflections`` lines above ``min_rel_intensity``. Requires
    pymatgen (via :func:`phases.simulate_pattern`).
    """
    pat = simulate_pattern(phase, _SIM_WAVELENGTH, two_theta_min=0.0,
                           two_theta_max=_SIM_TWO_THETA_MAX)
    d = np.array([p["d"] for p in pat], dtype=float)
    inten = np.array([p["intensity"] for p in pat], dtype=float)
    hkl = [p["hkl"] for p in pat]
    if d.size == 0:
        return d, inten, hkl
    w = inten / (inten.max() or 1.0)
    keep = w >= float(min_rel_intensity)
    d, w = d[keep], w[keep]
    hkl = [h for h, k in zip(hkl, keep) if k]
    order = np.argsort(w)[::-1][:max_reflections]
    return d[order], w[order], [hkl[i] for i in order]


def scale_at_pressure(phase: Phase, pressure: float) -> float:
    """Isotropic lattice scale factor ``s = (V(P)/V0)**(1/3)`` from the phase EOS
    (any supported type: BM2/BM3/BM4, Vinet, Murnaghan).

    Returns 1.0 at/below ambient or when the phase has no usable EOS."""
    if pressure <= 0 or not phase.has_eos():
        return 1.0
    return compression_at_pressure(phase.eos, float(pressure)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _nearest_gap(pred: np.ndarray, obs_sorted: np.ndarray) -> np.ndarray:
    """For each predicted d, distance to the nearest observed d (obs ascending)."""
    if obs_sorted.size == 0:
        return np.full(pred.shape, np.inf)
    idx = np.searchsorted(obs_sorted, pred)
    left = np.clip(idx - 1, 0, obs_sorted.size - 1)
    right = np.clip(idx, 0, obs_sorted.size - 1)
    return np.minimum(np.abs(pred - obs_sorted[left]), np.abs(pred - obs_sorted[right]))


def score_at_scale(obs_sorted: np.ndarray, d0: np.ndarray, weight: np.ndarray,
                   s: float, rel_tol: float,
                   strong_frac: float = 0.1) -> Tuple[float, int, float, float]:
    """Match score, #matched, recall, and precision for a lattice scale ``s``.

    ``score`` = Σ wᵢ·exp(−½(Δdᵢ/σᵢ)²), σᵢ = rel_tol·d_pred,ᵢ — the optimisation
    target for the pressure fit.

    Two complementary figures of merit, because neither alone survives DAC data:

    * ``recall`` — intensity-weighted fraction of the phase's *strong, observable*
      lines (≥ ``strong_frac`` of the strongest in-window line) that are matched.
      "Are the lines I'd expect to see present?" Good for simple/cubic phases;
      intrinsically low for a dense low-symmetry pattern where most strong lines
      go unobserved (texture, overlap, noise floor).
    * ``precision`` — fraction of the *observed* peaks lying in the predicted
      range that are explained by a predicted line. "Do my peaks belong to this
      phase?" High for the dominant/sample phase even when recall is low.

    A phase is taken as present if *either* is high (see ``fit_pressure_for_phase``).
    """
    pred = d0 * s
    sigma = np.maximum(rel_tol * pred, 1e-9)
    gap = _nearest_gap(pred, obs_sorted)
    kernel = np.exp(-0.5 * (gap / sigma) ** 2)
    score = float(np.sum(weight * kernel))
    n_matched = int(np.sum(gap < 2.0 * sigma))
    recall = 0.0
    precision = 0.0
    if obs_sorted.size and pred.size:
        span = 2.0 * float(sigma.max())
        # Recall over the strong in-window predicted lines.
        in_win = (pred >= obs_sorted[0] - span) & (pred <= obs_sorted[-1] + span)
        w_win = weight[in_win]
        if w_win.size:
            strong = w_win >= strong_frac * float(w_win.max())
            denom = float(np.sum(w_win[strong]))
            num = float(np.sum((weight * kernel)[in_win][strong]))
            recall = num / denom if denom > 0 else 0.0
        # Precision over observed peaks falling in the predicted d-range.
        pred_sorted = np.sort(pred)
        lo, hi = pred_sorted[0] - span, pred_sorted[-1] + span
        obs_in = obs_sorted[(obs_sorted >= lo) & (obs_sorted <= hi)]
        if obs_in.size:
            gap_obs = _nearest_gap(obs_in, pred_sorted)
            precision = float(np.mean(gap_obs < 2.0 * rel_tol * obs_in))
    return score, n_matched, recall, precision


def fit_pressure_for_phase(obs_d, phase: Phase,
                           refl: "Optional[Tuple[np.ndarray, np.ndarray, List[str]]]" = None,
                           *, p_min: float = 0.0, p_max: float = 100.0,
                           n_grid: int = 300, rel_tol: float = 0.01) -> Dict[str, Any]:
    """Best-fit pressure for one phase against one frame's observed d-spacings.

    Returns ``{pressure, score, confidence, n_matched, n_pred}``. With no EOS the
    phase is only scored at ambient (pressure 0). ``refl`` may be precomputed
    (see :func:`phase_reflections`) to avoid re-simulating across frames.
    """
    if refl is None:
        refl = phase_reflections(phase)
    d0, weight, _ = refl
    obs = np.sort(np.asarray(obs_d, dtype=float))
    obs = obs[np.isfinite(obs) & (obs > 0)]
    out = {"pressure": float("nan"), "score": 0.0, "confidence": 0.0,
           "recall": 0.0, "precision": 0.0, "n_matched": 0, "n_pred": int(d0.size)}
    if d0.size == 0 or obs.size == 0:
        out["pressure"] = 0.0
        return out

    def _record(p_val, s):
        score, nm, rec, prec = score_at_scale(obs, d0, weight, s, rel_tol)
        out.update({"pressure": p_val, "score": score, "n_matched": nm,
                    "recall": rec, "precision": prec,
                    "confidence": max(rec, prec)})  # present if EITHER is high

    if not phase.has_eos():
        _record(0.0, 1.0)
        return out

    Ps = np.linspace(float(p_min), float(p_max), int(max(n_grid, 2)))
    scores = np.array([score_at_scale(obs, d0, weight, scale_at_pressure(phase, P), rel_tol)[0]
                       for P in Ps])
    best = int(np.argmax(scores))
    p_opt = float(Ps[best])
    # Local refine within the bracketing grid cell.
    lo = float(Ps[max(best - 1, 0)])
    hi = float(Ps[min(best + 1, Ps.size - 1)])
    if hi > lo:
        try:
            from scipy.optimize import minimize_scalar  # lazy
            r = minimize_scalar(
                lambda P: -score_at_scale(obs, d0, weight, scale_at_pressure(phase, P), rel_tol)[0],
                bounds=(lo, hi), method="bounded")
            if r.success and -r.fun >= scores[best]:
                p_opt = float(r.x)
        except Exception:
            pass
    _record(p_opt, scale_at_pressure(phase, p_opt))
    return out


# ---------------------------------------------------------------------------
# Dataset driver (analysis HDF5 -> /identify appended)
# ---------------------------------------------------------------------------

def _h5_safe(name: str) -> str:
    return name.replace("/", "_")


def _identify_chunk(payload):
    """Worker: fit pressure for every phase over a contiguous chunk of frames.

    Reflections are precomputed in the parent and passed in, so the workers need
    no pymatgen. Excluded frames are left as NaN/zero.
    """
    phases, refls, obs_chunk, excluded_chunk, p_min, p_max, rel_tol = payload
    m = len(obs_chunk)
    res = {ph.name: {"pressure": np.full(m, np.nan, "f8"), "score": np.zeros(m, "f8"),
                     "confidence": np.zeros(m, "f8"), "recall": np.zeros(m, "f8"),
                     "precision": np.zeros(m, "f8"), "n_matched": np.zeros(m, "i4")}
           for ph in phases}
    for j in range(m):
        if excluded_chunk[j]:
            continue
        obs = obs_chunk[j]
        for ph, refl in zip(phases, refls):
            r = fit_pressure_for_phase(obs, ph, refl, p_min=p_min, p_max=p_max,
                                       rel_tol=rel_tol)
            rr = res[ph.name]
            rr["pressure"][j] = r["pressure"]; rr["score"][j] = r["score"]
            rr["confidence"][j] = r["confidence"]; rr["recall"][j] = r["recall"]
            rr["precision"][j] = r["precision"]; rr["n_matched"][j] = r["n_matched"]
    return res


def _read_good_peaks_by_frame(h5) -> "Tuple[int, List[np.ndarray]]":
    pk = h5.get("peaks")
    if pk is None or "center" not in pk or "frame" not in pk:
        raise ValueError("Analysis file lacks /peaks — run Step 2 (peak fitting) first.")
    frame = np.asarray(pk["frame"][:], dtype=int)
    center = np.asarray(pk["center"][:], dtype=float)
    flag = np.asarray(pk["flag"][:], dtype=int) if "flag" in pk else np.zeros_like(center, int)
    if "counts" in pk:
        n = int(np.asarray(pk["counts"][:]).size)
    else:
        bg = h5.get("background")
        n = int(bg["clean"].shape[0]) if bg is not None and "clean" in bg else (int(frame.max()) + 1 if frame.size else 0)
    good = flag == 0
    per_frame = [center[good & (frame == i)] for i in range(n)]
    return n, per_frame


def run_identification(
    analysis_h5: "str | Path",
    phases: "Sequence[Phase]",
    *,
    wavelength: "Optional[float]" = None,
    p_min: float = 0.0,
    p_max: float = 100.0,
    rel_tol: float = 0.01,
    out_h5: "Optional[str | Path]" = None,
    num_workers: int = 1,
) -> Dict[str, Any]:
    """Match every frame's good peaks against each candidate phase and store the
    per-frame best-fit pressure / score / confidence under ``/identify``.

        /identify  attrs: schema_version, unit, wavelength, p_min, p_max, rel_tol, phases
        /identify/<phase>/pressure    (N,)  best-fit pressure (GPa)
        /identify/<phase>/score       (N,)  match score
        /identify/<phase>/confidence  (N,)  matched-weight fraction in window
        /identify/<phase>/n_matched   (N,) int
                          attrs: name, n_pred, has_eos

    ``phases`` are Phase objects (resolve names via ``phases.load_library``).
    Requires pymatgen. Prints ``[IDENTIFY] <done> <total>`` progress.
    """
    import h5py  # type: ignore

    if not pymatgen_available():
        raise RuntimeError(
            "pymatgen is required for Step 3a phase matching (pip install pymatgen).")
    phases = [p for p in phases]
    if not phases:
        raise ValueError("No candidate phases supplied — enable some on the Phases tab.")

    import os
    import shutil

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    dst = Path(out_h5).expanduser().resolve() if out_h5 else src

    with h5py.File(str(src), "r") as h5:
        unit = str(h5.attrs.get("unit", ""))
        stored_wl = float(h5.attrs.get("wavelength", 0.0) or 0.0)
        source_reduced = h5.attrs.get("source_reduced", "")
        if isinstance(source_reduced, bytes):
            source_reduced = source_reduced.decode("utf-8", "replace")
        n, peaks_by_frame = _read_good_peaks_by_frame(h5)
        frames = h5.get("frames")
        excluded = (np.asarray(frames["excluded"][:], dtype=bool)
                    if frames is not None and "excluded" in frames else None)
    if excluded is None or excluded.size != n:
        excluded = np.zeros(n, dtype=bool)

    # Wavelength is only needed for a 2theta axis: prefer the value stored at
    # Step 1, then an explicit arg, then the reduced PONI.
    if wavelength is None and stored_wl > 0:
        wavelength = stored_wl
    if wavelength is None and unit.strip().lower() in ("2th_deg", "2th_rad"):
        wavelength = wavelength_from_reduced(source_reduced)
        if wavelength is None:
            raise ValueError(
                f"Data is on a {unit} axis but no wavelength was found. "
                "Re-run Step 1 (which stores it) or pass wavelength explicitly.")

    # Observed d per frame (convert once).
    obs_d_by_frame = [radial_to_d(c, unit, wavelength) if c.size else np.zeros(0)
                      for c in peaks_by_frame]

    workers = resolve_workers(num_workers)
    print(f"[IDENTIFY] {len(phases)} phase(s), {n} frames "
          f"({int(excluded.sum())} excluded), unit={unit or '?'} "
          f"P=[{p_min},{p_max}] GPa workers={workers}", flush=True)
    no_eos = [p.name for p in phases if not p.has_eos()]
    if no_eos:
        print(f"[IDENTIFY] WARNING: no Birch-Murnaghan EOS for {no_eos} — these "
              f"phases are evaluated at ambient only (pressure fixed at 0). Add "
              f"V0/K0/K0' on the Phases tab to fit pressure.", flush=True)

    # Reflections simulated once in the parent (needs pymatgen); workers only score.
    refl_cache = {ph.name: phase_reflections(ph) for ph in phases}
    refls = [refl_cache[ph.name] for ph in phases]
    results: Dict[str, Dict[str, np.ndarray]] = {
        ph.name: {"pressure": np.full(n, np.nan, "f8"), "score": np.zeros(n, "f8"),
                  "confidence": np.zeros(n, "f8"), "recall": np.zeros(n, "f8"),
                  "precision": np.zeros(n, "f8"), "n_matched": np.zeros(n, "i4")}
        for ph in phases}

    def _absorb(a, chunk_res):
        for name, rr in chunk_res.items():
            for k, v in rr.items():
                results[name][k][a:a + len(v)] = v

    if workers > 1 and n > 1:
        ranges = chunk_ranges(n, workers)
        payloads = [(phases, refls, obs_d_by_frame[a:b], excluded[a:b],
                     p_min, p_max, rel_tol) for a, b in ranges]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (a, b), chunk_res in zip(ranges, ex.map(_identify_chunk, payloads)):
                _absorb(a, chunk_res)
                done += (b - a)
                print(f"[IDENTIFY] {done} {n}", flush=True)
    else:
        _absorb(0, _identify_chunk((phases, refls, obs_d_by_frame, excluded,
                                    p_min, p_max, rel_tol)))
        print(f"[IDENTIFY] {n} {n}", flush=True)

    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            if "identify" in o:
                del o["identify"]
            gid = o.create_group("identify")
            gid.attrs.update({
                "schema_version": SCHEMA_VERSION, "unit": unit,
                "wavelength": float(wavelength) if wavelength else 0.0,
                "p_min": float(p_min), "p_max": float(p_max), "rel_tol": float(rel_tol),
                "phases": ", ".join(p.name for p in phases),
            })
            for ph in phases:
                g = gid.create_group(_h5_safe(ph.name))
                g.attrs.update({"name": ph.name, "n_pred": int(refl_cache[ph.name][0].size),
                                "has_eos": bool(ph.has_eos()), "category": ph.category})
                for k, v in results[ph.name].items():
                    g.create_dataset(k, data=v)
                # Cache the ambient reflection list so GUI overlays (reflection
                # tracks / phase layers) never need to re-run pymatgen — that
                # simulation, on the main thread, froze the app for low-symmetry
                # phases. d-spacings are wavelength-independent; tracks scale them
                # by the per-frame pressure.
                d0c, wc, hklc = refl_cache[ph.name]
                g.create_dataset("refl_d", data=np.asarray(d0c, "f8"))
                g.create_dataset("refl_w", data=np.asarray(wc, "f8"))
                g.create_dataset(
                    "refl_hkl",
                    data=np.array([str(h) for h in hklc], dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"))
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    # Per-phase summary. "Seen" uses a deliberately modest confidence bar (a
    # DAC pattern rarely shows every strong line); the richer stats below let the
    # user judge partial matches instead of collapsing them to a single 0.
    SEEN_CONF = 0.5
    summary = {}
    live = ~excluded
    for ph in phases:
        res = results[ph.name]
        conf = res["confidence"]
        nm = res["n_matched"]
        conf_live = conf[live] if live.any() else conf
        seen = (conf > SEEN_CONF) & live
        summary[ph.name] = {
            "mean_confidence": float(np.mean(conf_live)) if conf_live.size else 0.0,
            "max_confidence": float(np.max(conf_live)) if conf_live.size else 0.0,
            "max_recall": float(np.max(res["recall"][live])) if live.any() else 0.0,
            "max_precision": float(np.max(res["precision"][live])) if live.any() else 0.0,
            "seen_conf": SEEN_CONF,
            "n_frames_seen": int(np.sum(seen)),
            "n_frames_matched": int(np.sum((nm > 0) & live)),
            "max_matched": int(np.max(nm)) if nm.size else 0,
            "pressure_median": float(np.nanmedian(res["pressure"][seen])) if np.any(seen) else float("nan"),
            "has_eos": bool(ph.has_eos()),
        }

    manifest = {
        "tool_version": SCHEMA_VERSION, "source": str(src), "out_h5": str(dst),
        "n_frames": int(n), "unit": unit,
        "wavelength": float(wavelength) if wavelength else None,
        "p_min": float(p_min), "p_max": float(p_max), "rel_tol": float(rel_tol),
        "phases": [p.name for p in phases], "summary": summary,
    }
    print(f"[IDENTIFY] done -> {dst}", flush=True)
    return manifest
