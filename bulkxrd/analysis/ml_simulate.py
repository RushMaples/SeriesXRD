"""Step 3b, part 2 — pressure-conditioned simulator with DAC augmentations.

:mod:`mldata` already renders a *clean* single-phase pattern at a pressure. Real
diamond-anvil-cell data never looks like that, and a model trained on clean
patterns inherits a sim-to-real gap (the SimXRD-4M lesson: the simulator must
span the conditions the experiment actually shows). This module layers the
augmentations a DAC pattern exhibits onto that base, on the *same* shared d-grid
so simulated and experimental rows line up bin-for-bin:

  * EOS pressure shift   — peaks move as d0·s(P) (the physics prior itself).
  * mixtures             — several phases at once (sample + marker + gasket +
                           medium), the normal multi-phase DAC reality.
  * texture / missing    — azimuthally sparse rings: drop a fraction of peaks and
                           jitter relative intensities (orientation effects).
  * broadening variation — per-pattern FWHM and Lorentzian fraction (grain size /
                           strain / instrument).
  * d-offset drift       — a small global d-scale error (sample displacement /
                           calibration drift).
  * diamond spikes       — narrow spurious single-crystal peaks.
  * background humps      — smooth Compton / gasket / medium bumps.
  * truncation + noise   — a cut detector end and additive noise.

Multi-label by construction: each pattern's label is the multi-hot set of phases
present, so the same set trains a multi-phase classifier or the ranker's scorer.

Pure numpy; reflections come from :mod:`phases` (pymatgen) unless supplied, so the
augmentation maths is testable without pymatgen.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phases import Phase
from .identify import predicted_d, _parse_hkl, phase_reflections
from .peaks import pseudo_voigt
from .mldata import make_d_grid

Reflections = Tuple[np.ndarray, np.ndarray, list]   # (d0, weight, hkl)


# ---------------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    """Ranges for the per-pattern DAC augmentations (uniform unless noted).

    Defaults are deliberately moderate — enough variety to close the sim-to-real
    gap without drowning the signal. Tighten/loosen per instrument.
    """
    fwhm_d: Tuple[float, float] = (0.02, 0.06)        # peak FWHM in d (Å)
    eta: Tuple[float, float] = (0.2, 0.8)             # Lorentzian fraction
    drop_frac: Tuple[float, float] = (0.0, 0.4)       # fraction of a phase's peaks dropped
    intensity_jitter: float = 0.3                     # per-peak rel-intensity log-spread
    d_offset_frac: float = 0.003                      # global d-scale drift (relative)
    n_diamond_spikes: Tuple[int, int] = (0, 3)
    spike_amp: Tuple[float, float] = (0.2, 1.0)       # rel. to the max sample peak
    spike_fwhm_d: Tuple[float, float] = (0.008, 0.02)
    n_humps: Tuple[int, int] = (1, 3)
    hump_amp: Tuple[float, float] = (0.05, 0.35)
    hump_width_frac: Tuple[float, float] = (0.06, 0.25)   # of the d-range
    noise_sigma: Tuple[float, float] = (0.0, 0.05)
    truncate_frac: Tuple[float, float] = (0.0, 0.12)      # fraction cut from one end


def _u(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _ui(rng, lo_hi):
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


# ---------------------------------------------------------------------------
# Single-phase rendering with texture + broadening
# ---------------------------------------------------------------------------

def render_phase(d_grid: np.ndarray, centers: np.ndarray, weights: np.ndarray, *,
                 fwhm_d: float, eta: float, drop_frac: float = 0.0,
                 intensity_jitter: float = 0.0,
                 rng: "Optional[np.random.Generator]" = None) -> np.ndarray:
    """Render one phase's reflections (already positioned at ``centers``, Å) on
    ``d_grid`` with optional texture.

    ``drop_frac`` randomly removes that fraction of the reflections (azimuthally
    sparse rings below detection); ``intensity_jitter`` multiplies each surviving
    reflection by ``lognormal(0, jitter)`` (preferred-orientation intensity
    scatter). Peaks are pseudo-Voigts of width ``fwhm_d`` (Å) — the same profile
    the experimental peaks are fit with.
    """
    centers = np.asarray(centers, float)
    weights = np.asarray(weights, float).copy()
    rng = np.random.default_rng() if rng is None else rng
    keep = np.ones(centers.size, bool)
    if drop_frac > 0 and centers.size:
        n_drop = int(round(drop_frac * centers.size))
        if n_drop:
            keep[rng.choice(centers.size, size=min(n_drop, centers.size), replace=False)] = False
    if intensity_jitter > 0:
        weights = weights * rng.lognormal(0.0, intensity_jitter, size=weights.size)
    y = np.zeros_like(d_grid, dtype=float)
    for c, a, k in zip(centers, weights, keep):
        if k and d_grid[0] <= c <= d_grid[-1]:
            y += pseudo_voigt(d_grid, c, a, fwhm_d, eta)
    return y


# ---------------------------------------------------------------------------
# Whole-pattern augmentations
# ---------------------------------------------------------------------------

def add_background_humps(y: np.ndarray, d_grid: np.ndarray, cfg: AugmentConfig,
                         rng: np.random.Generator) -> np.ndarray:
    """Add smooth Gaussian humps (Compton / gasket / pressure-medium scattering)."""
    span = float(d_grid[-1] - d_grid[0]) or 1.0
    out = y.copy()
    scale = float(np.max(y)) or 1.0
    for _ in range(_ui(rng, cfg.n_humps)):
        c = _u(rng, (d_grid[0], d_grid[-1]))
        width = _u(rng, cfg.hump_width_frac) * span
        amp = _u(rng, cfg.hump_amp) * scale
        out = out + amp * np.exp(-0.5 * ((d_grid - c) / max(width, 1e-6)) ** 2)
    return out


def add_diamond_spikes(y: np.ndarray, d_grid: np.ndarray, cfg: AugmentConfig,
                       rng: np.random.Generator) -> np.ndarray:
    """Add narrow spurious single-crystal (diamond/gasket) spikes."""
    out = y.copy()
    scale = float(np.max(y)) or 1.0
    for _ in range(_ui(rng, cfg.n_diamond_spikes)):
        c = _u(rng, (d_grid[0], d_grid[-1]))
        out = out + _u(rng, cfg.spike_amp) * scale * pseudo_voigt(
            d_grid, c, 1.0, _u(rng, cfg.spike_fwhm_d), 1.0)
    return out


def apply_truncation(y: np.ndarray, cfg: AugmentConfig,
                     rng: np.random.Generator) -> np.ndarray:
    """Zero a random fraction of one end (detector truncation / beamstop)."""
    frac = _u(rng, cfg.truncate_frac)
    if frac <= 0:
        return y
    out = y.copy()
    k = int(frac * y.size)
    if k:
        if rng.random() < 0.5:
            out[:k] = 0.0
        else:
            out[-k:] = 0.0
    return out


def add_noise(y: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    sigma = _u(rng, cfg.noise_sigma)
    if sigma <= 0:
        return y
    scale = float(np.max(y)) or 1.0
    return y + rng.normal(0.0, sigma * scale, size=y.shape)


def _normalize(y: np.ndarray) -> np.ndarray:
    y = np.clip(np.nan_to_num(y, nan=0.0), 0.0, None)
    mx = float(np.max(y))
    return y / mx if mx > 0 else y


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _resolve_reflections(phases: "Sequence[Phase]",
                         reflections: "Optional[Dict[str, Reflections]]"
                         ) -> "Dict[str, Reflections]":
    if reflections is not None:
        return reflections
    return {ph.name: phase_reflections(ph) for ph in phases}


def simulate_augmented_pattern(
    phases_present: "Sequence[Phase]",
    pressures: "Sequence[float]",
    refls: "Dict[str, Reflections]",
    d_grid: np.ndarray,
    cfg: AugmentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """One augmented multi-phase pattern: render each present phase (with its own
    pressure, texture and broadening), then apply the shared whole-pattern
    augmentations (drift, humps, spikes, truncation, noise) and normalise."""
    fwhm = _u(rng, cfg.fwhm_d)
    eta = _u(rng, cfg.eta)
    drift = 1.0 + rng.uniform(-cfg.d_offset_frac, cfg.d_offset_frac)
    grid = d_grid / drift                              # global d-scale drift
    y = np.zeros_like(d_grid, dtype=float)
    for ph, P in zip(phases_present, pressures):
        d0, w, hkl = refls[ph.name]
        hkls = [_parse_hkl(h) for h in hkl] if hkl else None
        # Same anisotropic compression model as Step 3a (predicted_d), so an
        # axial-only phase shifts correctly instead of staying at ambient.
        centers = predicted_d(ph, np.asarray(d0, float), hkls, float(max(P, 0.0)))
        y += render_phase(grid, centers, w, fwhm_d=fwhm, eta=eta,
                          drop_frac=_u(rng, cfg.drop_frac),
                          intensity_jitter=cfg.intensity_jitter, rng=rng)
    y = add_background_humps(y, d_grid, cfg, rng)
    y = add_diamond_spikes(y, d_grid, cfg, rng)
    y = apply_truncation(y, cfg, rng)
    y = add_noise(y, cfg, rng)
    return _normalize(y).astype("f4")


def build_augmented_dataset(
    phases: "Sequence[Phase]",
    *,
    n_samples: int = 2000,
    max_phases_per_pattern: int = 1,
    pressures: "Optional[Sequence[float]]" = None,
    d_grid: "Optional[np.ndarray]" = None,
    cfg: "Optional[AugmentConfig]" = None,
    reflections: "Optional[Dict[str, Reflections]]" = None,
    seed: "Optional[int]" = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Build a DAC-augmented, pressure-conditioned training set.

    Each sample draws 1..``max_phases_per_pattern`` phases (mixtures when > 1),
    each at a random pressure from the grid (ambient only for no-EOS phases), and
    renders one augmented pattern. Returns ``(X (M, P) float32, Y (M, n_phases)
    multi-hot int8, phase_names, pressures (M, max_phases) with NaN padding)``.

    ``reflections`` (name -> (d0, weight, hkl)) bypasses pymatgen — supply it to
    simulate without the optional dependency.
    """
    grid = make_d_grid() if d_grid is None else np.asarray(d_grid, dtype=float)
    cfg = cfg or AugmentConfig()
    rng = np.random.default_rng(seed)
    phases = list(phases)
    names = [p.name for p in phases]
    idx_of = {p.name: j for j, p in enumerate(phases)}
    refls = _resolve_reflections(phases, reflections)
    if pressures is None:
        pressures = np.arange(0.0, 101.0, 5.0)
    pressures = np.asarray(pressures, float)
    k_max = max(1, min(int(max_phases_per_pattern), len(phases)))

    X = np.zeros((n_samples, grid.size), dtype="f4")
    Y = np.zeros((n_samples, len(phases)), dtype="i1")
    P_used = np.full((n_samples, k_max), np.nan, dtype="f4")
    for i in range(n_samples):
        k = 1 if k_max == 1 else int(rng.integers(1, k_max + 1))
        chosen = list(rng.choice(len(phases), size=k, replace=False))
        present = [phases[j] for j in chosen]
        ps = [float(rng.choice(pressures)) if (present[t].has_eos()
              or _has_axial(present[t])) else 0.0 for t in range(k)]
        X[i] = simulate_augmented_pattern(present, ps, refls, grid, cfg, rng)
        for j in chosen:
            Y[i, j] = 1
        P_used[i, :k] = ps
    return X, Y, names, P_used


def _has_axial(phase: Phase) -> bool:
    from .phases import has_axial_eos
    return has_axial_eos(phase)


def export_augmented_dataset(out_npz: "str | Path", phases: "Sequence[Phase]",
                             **kwargs) -> Dict[str, Any]:
    """Build :func:`build_augmented_dataset` and save it atomically as ``.npz``
    (``X``, ``Y``, ``phase_names``, ``pressures``, ``d_grid``)."""
    import os
    d_grid = kwargs.get("d_grid")
    grid = make_d_grid() if d_grid is None else np.asarray(d_grid, dtype=float)
    kwargs["d_grid"] = grid
    X, Y, names, P_used = build_augmented_dataset(phases, **kwargs)
    out = Path(out_npz).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, X=X, Y=Y, phase_names=np.array(names, dtype=object),
                        pressures=P_used, d_grid=grid)
    os.replace(tmp, out)
    man = {"out_npz": str(out), "n_samples": int(X.shape[0]), "n_points": int(grid.size),
           "n_phases": len(names), "phases": names,
           "multi_label": bool(kwargs.get("max_phases_per_pattern", 1) > 1)}
    print(f"[ML-SIM] augmented set: {X.shape[0]} patterns, {len(names)} phases -> {out}",
          flush=True)
    return man
