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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phases import Phase, simulate_pattern, volume_at_pressure, pymatgen_available

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
        m = re.search(r"wavelength\s*:\s*([0-9eE.+-]+)", str(poni))
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
    """Isotropic lattice scale factor ``s = (V(P)/V0)**(1/3)`` from the phase EOS.

    Returns 1.0 at/below ambient or when the phase has no usable EOS."""
    if pressure <= 0 or not phase.has_eos():
        return 1.0
    e = phase.eos
    V = volume_at_pressure(float(pressure), float(e["V0"]), float(e["K0"]), float(e["K0p"]))
    return (V / float(e["V0"])) ** (1.0 / 3.0)


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
                   s: float, rel_tol: float) -> Tuple[float, int, float]:
    """Match score, #matched, and confidence for a given lattice scale ``s``.

    Score = Σ wᵢ·exp(−½(Δdᵢ/σᵢ)²) with σᵢ = rel_tol·d_pred,ᵢ (q-resolution makes
    a relative tolerance natural). Confidence is the matched-weight fraction over
    only the predicted lines whose position falls inside the observed d-range, so
    reflections that can't be seen don't penalise a real phase.
    """
    pred = d0 * s
    sigma = np.maximum(rel_tol * pred, 1e-9)
    gap = _nearest_gap(pred, obs_sorted)
    kernel = np.exp(-0.5 * (gap / sigma) ** 2)
    score = float(np.sum(weight * kernel))
    n_matched = int(np.sum(gap < 2.0 * sigma))
    if obs_sorted.size:
        lo, hi = obs_sorted[0] - 2 * sigma.max(), obs_sorted[-1] + 2 * sigma.max()
        in_win = (pred >= lo) & (pred <= hi)
        denom = float(np.sum(weight[in_win]))
        conf = float(np.sum((weight * kernel)[in_win]) / denom) if denom > 0 else 0.0
    else:
        conf = 0.0
    return score, n_matched, conf


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
           "n_matched": 0, "n_pred": int(d0.size)}
    if d0.size == 0 or obs.size == 0:
        out["pressure"] = 0.0
        return out

    if not phase.has_eos():
        score, nm, conf = score_at_scale(obs, d0, weight, 1.0, rel_tol)
        out.update({"pressure": 0.0, "score": score, "confidence": conf, "n_matched": nm})
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
    score, nm, conf = score_at_scale(obs, d0, weight, scale_at_pressure(phase, p_opt), rel_tol)
    out.update({"pressure": p_opt, "score": score, "confidence": conf, "n_matched": nm})
    return out


# ---------------------------------------------------------------------------
# Dataset driver (analysis HDF5 -> /identify appended)
# ---------------------------------------------------------------------------

def _h5_safe(name: str) -> str:
    return name.replace("/", "_")


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

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    dst = Path(out_h5).expanduser().resolve() if out_h5 else src
    if dst != src:
        import shutil
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    with h5py.File(str(dst), "r") as h5:
        unit = str(h5.attrs.get("unit", ""))
        source_reduced = h5.attrs.get("source_reduced", "")
        if isinstance(source_reduced, bytes):
            source_reduced = source_reduced.decode("utf-8", "replace")
        n, peaks_by_frame = _read_good_peaks_by_frame(h5)

    # Wavelength is only needed for a 2theta axis.
    if wavelength is None and unit.strip().lower() in ("2th_deg", "2th_rad"):
        wavelength = wavelength_from_reduced(source_reduced)
        if wavelength is None:
            raise ValueError(
                f"Data is on a {unit} axis but no wavelength was found (PONI text "
                "missing from the reduced file). Pass wavelength explicitly.")

    # Observed d per frame (convert once).
    obs_d_by_frame = [radial_to_d(c, unit, wavelength) if c.size else np.zeros(0)
                      for c in peaks_by_frame]

    print(f"[IDENTIFY] {len(phases)} phase(s), {n} frames, unit={unit or '?'} "
          f"P=[{p_min},{p_max}] GPa", flush=True)

    results: Dict[str, Dict[str, np.ndarray]] = {}
    refl_cache: Dict[str, Any] = {}
    for ph in phases:
        refl_cache[ph.name] = phase_reflections(ph)
        results[ph.name] = {
            "pressure": np.full(n, np.nan, "f8"), "score": np.zeros(n, "f8"),
            "confidence": np.zeros(n, "f8"), "n_matched": np.zeros(n, "i4"),
        }

    for i in range(n):
        obs = obs_d_by_frame[i]
        for ph in phases:
            r = fit_pressure_for_phase(obs, ph, refl_cache[ph.name],
                                       p_min=p_min, p_max=p_max, rel_tol=rel_tol)
            res = results[ph.name]
            res["pressure"][i] = r["pressure"]
            res["score"][i] = r["score"]
            res["confidence"][i] = r["confidence"]
            res["n_matched"][i] = r["n_matched"]
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[IDENTIFY] {i + 1} {n}", flush=True)

    with h5py.File(str(dst), "r+") as o:
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
            res = results[ph.name]
            for k, v in res.items():
                g.create_dataset(k, data=v)

    # Per-phase summary over frames where the phase was confidently matched.
    summary = {}
    for ph in phases:
        res = results[ph.name]
        conf = res["confidence"]
        seen = conf > 0.5
        summary[ph.name] = {
            "mean_confidence": float(np.mean(conf)) if n else 0.0,
            "n_frames_seen": int(np.sum(seen)),
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
