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
                     pymatgen_available, has_axial_eos, axial_scales)
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


def _parse_hkl(s) -> "Optional[Tuple[int, int, int]]":
    """Parse an hkl label like '(1, -1, 0)' or '1 1 1' to a tuple, or None."""
    nums = re.findall(r"-?\d+", str(s))
    return (int(nums[0]), int(nums[1]), int(nums[2])) if len(nums) >= 3 else None


def _d_from_lattice(H: np.ndarray, lattice: "Dict[str, Any]") -> np.ndarray:
    """d-spacings (Å) for hkl rows ``H`` (N×3) via the reciprocal metric tensor.

    General triclinic formula 1/d² = hᵀ G* h with G* = inv(G); valid for every
    crystal system, so anisotropic compression of a, b, c (and angles) is handled
    exactly."""
    L = lattice or {}
    a = float(L.get("a") or 0.0)
    b = float(L.get("b") or a) or a
    c = float(L.get("c") or a) or a
    al = math.radians(float(L.get("alpha", 90.0) or 90.0))
    be = math.radians(float(L.get("beta", 90.0) or 90.0))
    ga = math.radians(float(L.get("gamma", 90.0) or 90.0))
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    G = np.array([[a * a, a * b * cg, a * c * cb],
                  [a * b * cg, b * b, b * c * ca],
                  [a * c * cb, b * c * ca, c * c]], dtype=float)
    Gstar = np.linalg.inv(G)
    s = np.einsum("ni,ij,nj->n", H, Gstar, H)
    s = np.where(s > 0, s, np.nan)
    return 1.0 / np.sqrt(s)


def predicted_d(phase: Phase, d0: np.ndarray, hkls, P: float) -> np.ndarray:
    """Predicted d-spacings (Å) at pressure ``P``.

    Anisotropic when the phase has an axial EOS, a lattice, and parseable hkl:
    each reflection's d0 is scaled by the *ratio* of its compressed-lattice
    d-spacing to its ambient one (so it reduces exactly to d0 at P=0 and is
    independent of any small offset between the simulated d0 and the metric
    calculation). Otherwise the isotropic ``d0·s``."""
    d0 = np.asarray(d0, dtype=float)
    if (P > 0 and has_axial_eos(phase) and phase.lattice
            and hkls is not None and all(h is not None for h in hkls)):
        sa, sb, sc = axial_scales(phase, P)
        Lp = dict(phase.lattice)
        for k, s in (("a", sa), ("b", sb), ("c", sc)):
            if Lp.get(k):
                Lp[k] = float(Lp[k]) * s
        H = np.asarray(hkls, dtype=float)
        ratio = _d_from_lattice(H, Lp) / _d_from_lattice(H, phase.lattice)
        dd = d0 * ratio
        bad = ~np.isfinite(dd)
        if bad.any():
            dd[bad] = d0[bad] * scale_at_pressure(phase, P)
        return dd
    return d0 * scale_at_pressure(phase, P)


def _match_pairs(pred: np.ndarray, obs: np.ndarray, rel_tol: float,
                 factor: float = 2.0) -> "List[Tuple[int, int, float]]":
    """Greedy ONE-TO-ONE assignment of predicted lines to observed peaks.

    A predicted line and an observed peak can be paired only once: closest pairs
    (within ``factor·σᵢ``, σᵢ = rel_tol·predᵢ) are consumed first. This stops a
    single observed peak from "explaining" several predicted reflections (which
    inflated n_matched / recall and let wrong phases look present in DAC data).
    Returns ``[(i_pred, j_obs, gap)]``.
    """
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if pred.size == 0 or obs.size == 0:
        return []
    cand: "List[Tuple[float, int, int]]" = []
    for i, pi in enumerate(pred):
        if not np.isfinite(pi) or pi <= 0:
            continue
        tol = factor * max(rel_tol * pi, 1e-9)
        for j, oj in enumerate(obs):
            g = abs(pi - oj)
            if g < tol:
                cand.append((g, i, j))
    cand.sort()
    used_p: set = set()
    used_o: set = set()
    out: "List[Tuple[int, int, float]]" = []
    for g, i, j in cand:
        if i in used_p or j in used_o:
            continue
        used_p.add(i)
        used_o.add(j)
        out.append((i, j, g))
    return out


def _score_pred(obs_sorted: np.ndarray, pred: np.ndarray, weight: np.ndarray,
                rel_tol: float, strong_frac: float = 0.1
                ) -> Tuple[float, int, float, float]:
    """Match score, #matched, recall, precision for a predicted d-spacing array.

    ``score`` = Σ wᵢ·exp(−½(Δdᵢ/σᵢ)²), σᵢ = rel_tol·d_pred,ᵢ — a smooth
    optimisation target for the pressure fit (nearest-neighbour, so it stays
    differentiable as P sweeps).

    The reported *evidence* (n_matched, recall, precision), by contrast, uses a
    hard ONE-TO-ONE assignment so a wrong phase cannot reuse one peak many times:

    * ``recall`` — intensity-weighted fraction of the phase's *strong, observable*
      lines (≥ ``strong_frac`` of the strongest in-window line) that are matched
      one-to-one.
    * ``precision`` — fraction of the *observed* peaks in the predicted range that
      a predicted line claims (matched-pairs / observed-in-range, ≤ 1).
    """
    pred = np.asarray(pred, dtype=float)
    sigma = np.maximum(rel_tol * pred, 1e-9)
    gap = _nearest_gap(pred, obs_sorted)
    kernel = np.exp(-0.5 * (gap / sigma) ** 2)
    score = float(np.sum(weight * kernel))
    pairs = _match_pairs(pred, obs_sorted, rel_tol, factor=2.0)
    n_matched = len(pairs)
    matched_pred = np.zeros(pred.size, dtype=bool)
    for i, _j, _g in pairs:
        matched_pred[i] = True
    recall = 0.0
    precision = 0.0
    if obs_sorted.size and pred.size:
        span = 2.0 * float(sigma.max())
        # Recall over the strong in-window predicted lines (matched one-to-one).
        in_win = (pred >= obs_sorted[0] - span) & (pred <= obs_sorted[-1] + span)
        w_win = weight[in_win]
        if w_win.size:
            strong = w_win >= strong_frac * float(w_win.max())
            denom = float(np.sum(w_win[strong]))
            num = float(np.sum((weight * kernel * matched_pred)[in_win][strong]))
            recall = num / denom if denom > 0 else 0.0
        # Precision: matched pairs / observed peaks in the predicted d-range.
        pred_sorted = np.sort(pred)
        lo, hi = pred_sorted[0] - span, pred_sorted[-1] + span
        n_obs_in = int(np.sum((obs_sorted >= lo) & (obs_sorted <= hi)))
        if n_obs_in:
            precision = min(1.0, n_matched / float(n_obs_in))
    return score, n_matched, recall, precision


def score_at_scale(obs_sorted: np.ndarray, d0: np.ndarray, weight: np.ndarray,
                   s: float, rel_tol: float,
                   strong_frac: float = 0.1) -> Tuple[float, int, float, float]:
    """Isotropic convenience wrapper: score the predictions ``d0·s``."""
    return _score_pred(obs_sorted, np.asarray(d0, dtype=float) * s, weight,
                       rel_tol, strong_frac)


DEFAULT_MIN_MATCHED = 3
# Floor on the half-width of the pressure search/penalty window (GPa), so a
# near-zero metadata sigma can't collapse the window to nothing.
_MIN_PRESSURE_WINDOW = 0.3


def pressure_window_halfwidth(sigma: "Optional[float]", default_window: float,
                              sigma_k: float = 2.0) -> float:
    """Half-width (GPa) of the pressure prior window for one frame.

    Uses ``sigma_k·sigma`` when a per-frame pressure uncertainty is known,
    otherwise the ``default_window``; floored at :data:`_MIN_PRESSURE_WINDOW`."""
    if sigma is not None and np.isfinite(sigma) and sigma > 0:
        w = float(sigma_k) * float(sigma)
    else:
        w = float(default_window)
    return max(w, _MIN_PRESSURE_WINDOW)


def conservative_confidence(recall: float, precision: float, n_matched: int,
                            *, min_matched: int = DEFAULT_MIN_MATCHED,
                            prior_penalty: float = 1.0) -> float:
    """Combine the figures of merit into ONE conservative confidence in [0, 1].

    Replaces the old ``max(recall, precision)`` — which let a phase explaining a
    couple of the busiest peaks score 1.0 — with three multiplicative factors:

    * ``balanced`` = F1(recall, precision): demands the phase both shows its own
      strong lines *and* accounts for what is observed (neither alone suffices).
    * ``evidence`` = min(1, n_matched / min_matched): penalises matches resting on
      too few reflections (the DARA/RADAR-PD "minimum evidence" lesson).
    * ``prior_penalty`` ∈ (0, 1]: Gaussian falloff when the fitted pressure
      disagrees with the frame's metadata pressure (1.0 when no prior).
    """
    r = max(0.0, float(recall))
    p = max(0.0, float(precision))
    balanced = (2.0 * r * p / (r + p)) if (r + p) > 0 else 0.0
    mm = max(1, int(min_matched))
    evidence = min(1.0, max(0, int(n_matched)) / float(mm))
    return float(balanced * evidence * max(0.0, min(1.0, prior_penalty)))


def fit_pressure_for_phase(obs_d, phase: Phase,
                           refl: "Optional[Tuple[np.ndarray, np.ndarray, List[str]]]" = None,
                           *, p_min: float = 0.0, p_max: float = 100.0,
                           n_grid: int = 300, rel_tol: float = 0.01,
                           p_prior: "Optional[float]" = None,
                           p_window: "Optional[float]" = None,
                           min_matched: int = DEFAULT_MIN_MATCHED) -> Dict[str, Any]:
    """Best-fit pressure for one phase against one frame's observed d-spacings.

    Returns ``{pressure, score, confidence, recall, precision, n_matched,
    n_pred}``. Compression is anisotropic when the phase carries an ``axial_eos``
    (per-axis), else isotropic from the volume EOS; with neither, the phase is
    only scored at ambient (pressure 0). ``refl`` may be precomputed (see
    :func:`phase_reflections`) to avoid re-simulating across frames.

    Pressure prior — the key DAC fix. When ``p_prior`` (a frame's metadata
    pressure, GPa) is given, the search is *confined* to ``p_prior ± p_window``
    instead of the whole ``[p_min, p_max]`` range, so a wrong phase can no longer
    slide along pressure until a few lines happen to coincide. The fitted
    pressure is additionally weighed against the prior by a Gaussian penalty in
    :func:`conservative_confidence` (tolerance = ``p_window``). The prior is not
    applied to phases without an EOS (their pressure is fixed at ambient).
    """
    if refl is None:
        refl = phase_reflections(phase)
    d0, weight, hkl_raw = refl
    d0 = np.asarray(d0, dtype=float)
    weight = np.asarray(weight, dtype=float)
    hkls = [_parse_hkl(h) for h in hkl_raw] if hkl_raw is not None else None
    obs = np.sort(np.asarray(obs_d, dtype=float))
    obs = obs[np.isfinite(obs) & (obs > 0)]
    out = {"pressure": float("nan"), "score": 0.0, "confidence": 0.0,
           "recall": 0.0, "precision": 0.0, "n_matched": 0, "n_pred": int(d0.size)}
    if d0.size == 0 or obs.size == 0:
        out["pressure"] = 0.0
        return out

    has_prior = (p_prior is not None and np.isfinite(p_prior))
    has_eos = phase.has_eos() or has_axial_eos(phase)
    # The pressure-prior penalty applies whenever a frame pressure is known —
    # including for no-EOS phases. A no-EOS phase is scored at ambient (0 GPa);
    # on a genuinely high-pressure frame that disagrees with the prior, so its
    # confidence is dragged down rather than letting it falsely match as ambient.
    pen_prior = float(p_prior) if has_prior else None
    pen_tol = float(p_window) if (p_window and p_window > 0) else None

    def _score_P(P):
        return _score_pred(obs, predicted_d(phase, d0, hkls, P), weight, rel_tol)

    def _record(p_val):
        score, nm, rec, prec = _score_P(p_val)
        prior_pen = 1.0
        if pen_prior is not None and pen_tol:
            prior_pen = float(np.exp(-0.5 * ((p_val - pen_prior) / pen_tol) ** 2))
        out.update({"pressure": p_val, "score": score, "n_matched": nm,
                    "recall": rec, "precision": prec,
                    "confidence": conservative_confidence(
                        rec, prec, nm, min_matched=min_matched,
                        prior_penalty=prior_pen)})

    if not has_eos:
        _record(0.0)
        return out

    # Confine the search to the prior window when we have one.
    lo_b, hi_b = float(p_min), float(p_max)
    if has_prior and p_window and p_window > 0:
        lo_b = max(lo_b, float(p_prior) - float(p_window))
        hi_b = min(hi_b, float(p_prior) + float(p_window))
        if hi_b <= lo_b:                       # prior sits outside [p_min, p_max]
            lo_b = hi_b = min(max(float(p_prior), float(p_min)), float(p_max))

    if hi_b <= lo_b:
        _record(lo_b)
        return out

    Ps = np.linspace(lo_b, hi_b, int(max(n_grid, 2)))
    scores = np.array([_score_P(P)[0] for P in Ps])
    best = int(np.argmax(scores))
    p_opt = float(Ps[best])
    # Local refine within the bracketing grid cell.
    lo = float(Ps[max(best - 1, 0)])
    hi = float(Ps[min(best + 1, Ps.size - 1)])
    if hi > lo:
        try:
            from scipy.optimize import minimize_scalar  # lazy
            r = minimize_scalar(lambda P: -_score_P(P)[0],
                                bounds=(lo, hi), method="bounded")
            if r.success and -r.fun >= scores[best]:
                p_opt = float(r.x)
        except Exception:
            pass
    _record(p_opt)
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
    (phases, refls, obs_chunk, excluded_chunk, prior_chunk, window_chunk,
     p_min, p_max, rel_tol, min_matched) = payload
    m = len(obs_chunk)
    res = {ph.name: {"pressure": np.full(m, np.nan, "f8"), "score": np.zeros(m, "f8"),
                     "confidence": np.zeros(m, "f8"), "recall": np.zeros(m, "f8"),
                     "precision": np.zeros(m, "f8"), "n_matched": np.zeros(m, "i4")}
           for ph in phases}
    for j in range(m):
        if excluded_chunk[j]:
            continue
        obs = obs_chunk[j]
        pp = None
        if prior_chunk is not None and np.isfinite(prior_chunk[j]):
            pp = float(prior_chunk[j])
        pw = float(window_chunk[j]) if (window_chunk is not None and pp is not None) else None
        for ph, refl in zip(phases, refls):
            r = fit_pressure_for_phase(obs, ph, refl, p_min=p_min, p_max=p_max,
                                       rel_tol=rel_tol, p_prior=pp, p_window=pw,
                                       min_matched=min_matched)
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
    pressure_by_frame: "Optional[Sequence[float]]" = None,
    pressure_sigma_by_frame: "Optional[Sequence[float]]" = None,
    use_frame_pressure: bool = True,
    pressure_window: float = 2.0,
    pressure_sigma_k: float = 2.0,
    min_matched: int = DEFAULT_MIN_MATCHED,
    marker_prior: bool = False,
) -> Dict[str, Any]:
    """Match every frame's good peaks against each candidate phase and store the
    per-frame best-fit pressure / score / confidence under ``/identify``.

        /identify  attrs: schema_version, unit, wavelength, p_min, p_max, rel_tol,
                          phases, pressure_window, pressure_sigma_k, min_matched
        /identify/<phase>/pressure    (N,)  best-fit pressure (GPa)
        /identify/<phase>/score       (N,)  match score
        /identify/<phase>/confidence  (N,)  conservative confidence (see
                                            conservative_confidence)
        /identify/<phase>/recall, /precision (N,)
        /identify/<phase>/n_matched   (N,) int  one-to-one matched reflections
                          attrs: name, n_pred, has_eos, category

    Pressure prior (the DAC accuracy fix): if ``pressure_by_frame`` is given — or
    the analysis file already carries ``/frames/pressure`` (from filenames or a
    CSV import, see ``frame_metadata.py``) — each phase is fitted only within
    ``±window`` of that frame's pressure rather than the full ``[p_min, p_max]``.
    The window is ``pressure_sigma_k·σ`` where a per-frame ``pressure_sigma`` is
    known, else ``pressure_window`` (GPa). With ``marker_prior=True`` and no
    metadata pressure, a first pass over the marker-category phases estimates the
    per-frame pressure, which then primes the full pass.

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
    # Open-set mode sweeps the whole library, which can include phases with no
    # simulatable structure (e.g. a He pressure medium or a gasket alloy entry).
    # Drop them up front so one un-simulatable phase can't abort the whole run.
    no_struct = [p.name for p in phases if not p.has_structure()]
    phases = [p for p in phases if p.has_structure()]
    if no_struct:
        print(f"[IDENTIFY] skipped (no structure to simulate): {no_struct}", flush=True)
    if not phases:
        raise ValueError(
            "None of the selected phases have a simulatable structure "
            "(need a CIF, or space group + lattice + atoms). Add structure on the "
            "Phases tab, or enable phases that have one.")

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
        # Frame pressure prior (+ optional per-frame uncertainty) — written by
        # Step 1 from filenames, or by a frame_metadata CSV import. An explicit
        # pressure_by_frame argument overrides whatever is on disk.
        file_pressure = (np.asarray(frames["pressure"][:], dtype=float)
                         if frames is not None and "pressure" in frames else None)
        file_sigma = (np.asarray(frames["pressure_sigma"][:], dtype=float)
                      if frames is not None and "pressure_sigma" in frames else None)
    if excluded is None or excluded.size != n:
        excluded = np.zeros(n, dtype=bool)

    def _as_n(arr):
        if arr is None:
            return None
        a = np.asarray(arr, dtype=float)
        return a if a.size == n else None

    prior_pressure = _as_n(pressure_by_frame if pressure_by_frame is not None
                           else (file_pressure if use_frame_pressure else None))
    prior_sigma = _as_n(pressure_sigma_by_frame if pressure_sigma_by_frame is not None
                        else (file_sigma if use_frame_pressure else None))

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

    # Reflections simulated once in the parent (needs pymatgen); workers only
    # score. Guard each simulation so one phase that pymatgen chokes on (bad CIF,
    # exotic element) is skipped with a warning rather than killing the run.
    refl_cache = {}
    kept = []
    for ph in phases:
        try:
            refl_cache[ph.name] = phase_reflections(ph)
            kept.append(ph)
        except Exception as e:
            print(f"[IDENTIFY] skipped {ph.name!r}: simulation failed ({e})", flush=True)
    phases = kept
    if not phases:
        raise ValueError("No phases could be simulated for identification.")
    refls = [refl_cache[ph.name] for ph in phases]

    # Marker-derived prior: when no metadata pressure exists but the user asked
    # for it, fit only the pressure-marker phases first and adopt the best
    # marker's per-frame pressure as the prior that primes the full pass.
    if marker_prior and (prior_pressure is None or not np.any(np.isfinite(prior_pressure))):
        markers = [ph for ph in phases
                   if ph.category == "marker" and (ph.has_eos() or has_axial_eos(ph))]
        if markers:
            print(f"[IDENTIFY] marker-prior pass over {[m.name for m in markers]}", flush=True)
            est = np.full(n, np.nan, "f8")
            best_conf = np.zeros(n, "f8")
            for j in range(n):
                if excluded[j]:
                    continue
                obs = obs_d_by_frame[j]
                if obs.size == 0:
                    continue
                for m in markers:
                    r = fit_pressure_for_phase(obs, m, refl_cache[m.name], p_min=p_min,
                                               p_max=p_max, rel_tol=rel_tol,
                                               min_matched=min_matched)
                    if (r["n_matched"] >= min_matched and np.isfinite(r["pressure"])
                            and r["confidence"] > best_conf[j]):
                        best_conf[j] = r["confidence"]
                        est[j] = r["pressure"]
            if np.any(np.isfinite(est)):
                prior_pressure = est
                print(f"[IDENTIFY] marker prior set for "
                      f"{int(np.sum(np.isfinite(est)))}/{n} frames", flush=True)

    # Per-frame search/penalty window (half-width, GPa); None when no prior.
    n_prior = int(np.sum(np.isfinite(prior_pressure))) if prior_pressure is not None else 0
    if prior_pressure is not None and n_prior:
        windows = np.array([
            pressure_window_halfwidth(
                (prior_sigma[j] if (prior_sigma is not None and np.isfinite(prior_sigma[j]))
                 else None),
                pressure_window, pressure_sigma_k)
            for j in range(n)], dtype="f8")
        print(f"[IDENTIFY] pressure prior on {n_prior}/{n} frames "
              f"(window +/-{pressure_window} GPa or {pressure_sigma_k}*sigma)", flush=True)
        # Auto-widen the global search range to cover the metadata pressures (+window).
        # Otherwise a frame whose prior sits outside [p_min, p_max] would clamp its
        # search to the boundary while the prior penalty still measures against the
        # true prior — collapsing confidence to ~0 for an otherwise-correct phase.
        finite_prior = prior_pressure[np.isfinite(prior_pressure)]
        wmax = float(np.nanmax(windows))
        need_lo = max(0.0, float(finite_prior.min()) - wmax)
        need_hi = float(finite_prior.max()) + wmax
        if need_lo < p_min or need_hi > p_max:
            new_lo, new_hi = min(p_min, need_lo), max(p_max, need_hi)
            print(f"[IDENTIFY] WARNING: widening pressure range "
                  f"[{p_min:g},{p_max:g}] -> [{new_lo:g},{new_hi:g}] GPa to cover the "
                  f"frame-metadata pressures (+window).", flush=True)
            p_min, p_max = new_lo, new_hi
    else:
        prior_pressure = None
        windows = None

    results: Dict[str, Dict[str, np.ndarray]] = {
        ph.name: {"pressure": np.full(n, np.nan, "f8"), "score": np.zeros(n, "f8"),
                  "confidence": np.zeros(n, "f8"), "recall": np.zeros(n, "f8"),
                  "precision": np.zeros(n, "f8"), "n_matched": np.zeros(n, "i4")}
        for ph in phases}

    def _absorb(a, chunk_res):
        for name, rr in chunk_res.items():
            for k, v in rr.items():
                results[name][k][a:a + len(v)] = v

    def _slice(arr, a, b):
        return arr[a:b] if arr is not None else None

    if workers > 1 and n > 1:
        ranges = chunk_ranges(n, workers)
        payloads = [(phases, refls, obs_d_by_frame[a:b], excluded[a:b],
                     _slice(prior_pressure, a, b), _slice(windows, a, b),
                     p_min, p_max, rel_tol, min_matched) for a, b in ranges]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (a, b), chunk_res in zip(ranges, ex.map(_identify_chunk, payloads)):
                _absorb(a, chunk_res)
                done += (b - a)
                print(f"[IDENTIFY] {done} {n}", flush=True)
    else:
        _absorb(0, _identify_chunk((phases, refls, obs_d_by_frame, excluded,
                                    prior_pressure, windows,
                                    p_min, p_max, rel_tol, min_matched)))
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
                "pressure_window": float(pressure_window),
                "pressure_sigma_k": float(pressure_sigma_k),
                "min_matched": int(min_matched),
                "n_pressure_prior": int(n_prior),
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

    # Per-phase summary. "Seen" requires both a modest confidence bar (a DAC
    # pattern rarely shows every strong line) AND minimum reflection evidence
    # (≥ min_matched one-to-one matches), so a phase can't be called present off
    # one or two coincidental peaks. The richer stats let the user judge partial
    # matches instead of collapsing them to a single 0.
    SEEN_CONF = 0.5
    summary = {}
    live = ~excluded
    for ph in phases:
        res = results[ph.name]
        conf = res["confidence"]
        nm = res["n_matched"]
        conf_live = conf[live] if live.any() else conf
        seen = (conf > SEEN_CONF) & (nm >= int(min_matched)) & live
        summary[ph.name] = {
            "mean_confidence": float(np.mean(conf_live)) if conf_live.size else 0.0,
            "max_confidence": float(np.max(conf_live)) if conf_live.size else 0.0,
            "max_recall": float(np.max(res["recall"][live])) if live.any() else 0.0,
            "max_precision": float(np.max(res["precision"][live])) if live.any() else 0.0,
            "seen_conf": SEEN_CONF,
            "seen_min_matched": int(min_matched),
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
