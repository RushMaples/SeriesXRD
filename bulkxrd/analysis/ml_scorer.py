"""Step 3b scorer seam — deterministic today, learned tomorrow, same interface.

:mod:`ml_rank` ranks candidate phases per frame; *how* a (measured pattern,
candidate phase, pressure) triple is turned into a similarity score is this
module's one job. The seam exists so a learned RADAR-PD-style scorer can slot in
behind the exact interface the production ranker already uses, without touching
the ranking loop, the HDF5 schema, or the verifier — and without making torch a
core dependency.

    scorer.score(meas, phase, refl, d_grid, pressure, pressure_grid=None)
        -> (best_score, best_pressure)

* :class:`CosineScorer` — the default and the trusted baseline. Simulates the
  candidate at the frame's pressure (or scans a coarse grid when unknown) with
  the same anisotropic ``predicted_d`` model Step 3a verifies with, and scores
  cosine similarity. Pure numpy; no optional dependencies.
* :class:`TorchScorer` — adapter for a trained model (``bulkxrd[ml]``). Loads a
  TorchScript module mapping a ``(2, P)`` stack of (measured, candidate)
  fingerprints to a scalar in [0, 1]. Import of torch is lazy and every missing
  prerequisite raises an instructive error at *construction*, never an
  ``ImportError`` crash mid-rank.
* :func:`make_scorer` — resolve a config spec (``"cosine"``, ``"torch:<path>"``,
  or a dict) to a scorer instance; the seam the worker/CLI can expose later.

``score_frame`` scores every candidate for one frame and exists for learned
scorers to override with a single batched forward pass; the default simply loops
``score`` so deterministic behaviour is unchanged.

Whatever scorer proposes, the deterministic Step-3a matcher still verifies —
the scorer can only ever shortlist, never accept.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phases import Phase, has_pressure_dof
from .mldata import simulate_training_pattern

Reflections = Tuple[np.ndarray, np.ndarray, list]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two non-negative spectra in [0, 1]."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class PhaseScorer:
    """Interface for frame-vs-candidate scoring (see module docstring).

    Subclasses implement :meth:`score`; :meth:`score_frame` may be overridden to
    batch all candidates of a frame in one pass. ``name`` is persisted as the
    ``method`` attr of ``/ml/candidates`` for provenance.
    """

    name = "abstract"

    def score(self, meas: np.ndarray, phase: Phase, refl: Reflections,
              d_grid: np.ndarray, pressure: "Optional[float]", *,
              pressure_grid: "Optional[np.ndarray]" = None) -> Tuple[float, float]:
        """Best (score, pressure) for one candidate against one measured frame.

        ``pressure`` is the frame's metadata prior (GPa) or None; with None and a
        pressure-capable phase, the score is maximised over ``pressure_grid``."""
        raise NotImplementedError

    def score_frame(self, meas: np.ndarray, phases: "Sequence[Phase]",
                    refls: "Sequence[Reflections]", d_grid: np.ndarray,
                    pressure: "Optional[float]", *,
                    pressure_grid: "Optional[np.ndarray]" = None
                    ) -> "List[Tuple[float, float]]":
        """(score, pressure) per candidate. Default: loop :meth:`score`."""
        return [self.score(meas, ph, rf, d_grid, pressure, pressure_grid=pressure_grid)
                for ph, rf in zip(phases, refls)]

    def _candidate_pressures(self, phase: Phase, pressure: "Optional[float]",
                             pressure_grid: "Optional[np.ndarray]") -> "List[float]":
        """The pressures a candidate is evaluated at: the frame prior when known,
        else a scan for pressure-capable phases, else ambient."""
        if pressure is not None and np.isfinite(pressure):
            return [float(pressure)]
        if has_pressure_dof(phase) and pressure_grid is not None and len(pressure_grid):
            return [float(p) for p in pressure_grid]
        return [0.0]


class CosineScorer(PhaseScorer):
    """The deterministic baseline: cosine of the measured pattern vs the phase
    simulated at the candidate pressure (same anisotropic ``predicted_d`` model
    the Step-3a verifier uses). Pure numpy."""

    name = "cosine"

    def __init__(self, fwhm_d: float = 0.03):
        self.fwhm_d = float(fwhm_d)

    def score(self, meas, phase, refl, d_grid, pressure, *, pressure_grid=None):
        best_s, best_p = 0.0, 0.0
        for i, P in enumerate(self._candidate_pressures(phase, pressure, pressure_grid)):
            sim = simulate_training_pattern(phase, P, d_grid, refl=refl,
                                            fwhm_d=self.fwhm_d)
            s = cosine_similarity(meas, sim)
            if i == 0 or s > best_s:
                best_s, best_p = s, P
        return best_s, best_p


class TorchScorer(PhaseScorer):
    """Adapter for a trained (RADAR-PD-style) scorer behind ``bulkxrd[ml]``.

    Contract: a TorchScript module taking a ``(1, 2, P)`` float32 tensor —
    channel 0 the measured fingerprint, channel 1 the candidate simulated at the
    trial pressure, both on the shared d-grid and preprocessed identically
    (clip-negative + max-normalised; see ``/ml/candidates`` attrs) — returning a
    scalar similarity in [0, 1]. Candidate simulation and pressure handling stay
    identical to the deterministic scorer, so the *only* thing the model replaces
    is the similarity function.
    """

    name = "torch"

    def __init__(self, model_path: "str | Path", *, fwhm_d: float = 0.03):
        try:
            import torch  # noqa: F401  (lazy; optional dependency)
        except ImportError as e:
            raise RuntimeError(
                "The learned scorer needs PyTorch, which is not installed. "
                "Install the optional extra with `pip install bulkxrd[ml]`, or use "
                "the default deterministic scorer (scorer='cosine').") from e
        p = Path(model_path).expanduser()
        if not p.is_file():
            raise RuntimeError(
                f"Learned-scorer model not found: {p}. Train/export a TorchScript "
                "model first, or use the default deterministic scorer "
                "(scorer='cosine').")
        import torch
        self._torch = torch
        self.model = torch.jit.load(str(p), map_location="cpu").eval()
        self.model_path = str(p)
        self.fwhm_d = float(fwhm_d)

    def score(self, meas, phase, refl, d_grid, pressure, *, pressure_grid=None):
        t = self._torch
        m = np.asarray(meas, dtype="f4")
        best_s, best_p = 0.0, 0.0
        for i, P in enumerate(self._candidate_pressures(phase, pressure, pressure_grid)):
            sim = simulate_training_pattern(phase, P, d_grid, refl=refl,
                                            fwhm_d=self.fwhm_d).astype("f4")
            x = t.from_numpy(np.stack([m, sim])[None, :, :])
            with t.no_grad():
                s = float(self.model(x).reshape(-1)[0])
            if i == 0 or s > best_s:
                best_s, best_p = s, P
        return best_s, best_p


def make_scorer(spec: "PhaseScorer | str | Dict[str, Any] | None" = None, *,
                fwhm_d: float = 0.03) -> PhaseScorer:
    """Resolve a scorer spec to an instance.

    ``None``/``"cosine"`` → :class:`CosineScorer` (the default);
    ``"torch:<model_path>"`` or ``{"kind": "torch", "model": <path>}`` →
    :class:`TorchScorer`; an existing :class:`PhaseScorer` passes through.
    Raises an instructive error (never a bare ImportError) when a learned scorer
    is requested without its prerequisites.
    """
    if spec is None:
        return CosineScorer(fwhm_d=fwhm_d)
    if isinstance(spec, PhaseScorer):
        return spec
    if isinstance(spec, str):
        s = spec.strip()
        if s in ("", "cosine"):
            return CosineScorer(fwhm_d=fwhm_d)
        if s.startswith("torch:"):
            return TorchScorer(s.split(":", 1)[1], fwhm_d=fwhm_d)
        if s == "torch":
            raise RuntimeError(
                "A learned scorer needs a model path: use 'torch:<model_path>' "
                "(or a {'kind': 'torch', 'model': ...} spec).")
        raise ValueError(f"Unknown scorer spec {spec!r} (use 'cosine' or 'torch:<path>').")
    if isinstance(spec, dict):
        kind = str(spec.get("kind", "cosine")).strip().lower()
        if kind == "cosine":
            return CosineScorer(fwhm_d=float(spec.get("fwhm_d", fwhm_d)))
        if kind == "torch":
            model = spec.get("model", "")
            if not model:
                raise RuntimeError(
                    "A learned scorer needs a model path: {'kind': 'torch', "
                    "'model': <path>}.")
            return TorchScorer(model, fwhm_d=float(spec.get("fwhm_d", fwhm_d)))
        raise ValueError(f"Unknown scorer kind {kind!r} (use 'cosine' or 'torch').")
    raise TypeError(f"Cannot resolve a scorer from {type(spec).__name__}.")
