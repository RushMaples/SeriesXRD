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

from .phases import Phase, has_pressure_dof, clamp_to_validity
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
        else a scan for pressure-capable phases, else ambient. Every pressure is
        clamped to the phase's validity ceiling (``eos['p_max']``) so a
        stability-limited entry is never simulated where it cannot exist."""
        if pressure is not None and np.isfinite(pressure):
            return [clamp_to_validity(phase, float(pressure))]
        if has_pressure_dof(phase) and pressure_grid is not None and len(pressure_grid):
            # Grid points above the ceiling collapse onto it (dedup via the set).
            ps = sorted({clamp_to_validity(phase, float(p)) for p in pressure_grid})
            return ps or [0.0]
        return [0.0]


class CosineScorer(PhaseScorer):
    """The deterministic baseline: full-pattern cosine of the measured pattern vs
    the phase simulated at the candidate pressure (same anisotropic
    ``predicted_d`` model the Step-3a verifier uses). Pure numpy.

    Note the cosine runs over the whole d-grid, not just the candidate's own
    lines — a minor phase in a busy mixture is diluted by everything else in the
    frame. That is why the default ranking source is the Step-3a *residual*
    (majors already removed), not the raw pattern.

    ``fwhm_q`` (Å⁻¹, q-constant instrument resolution) is the physical width
    model and takes precedence; ``fwhm_d`` (Å, constant in d) is the legacy
    fallback (see :func:`mldata.peak_fwhm_d`).
    """

    name = "cosine"

    def __init__(self, fwhm_d: float = 0.03, *, fwhm_q: "Optional[float]" = None):
        self.fwhm_d = float(fwhm_d)
        self.fwhm_q = float(fwhm_q) if fwhm_q else None

    def score(self, meas, phase, refl, d_grid, pressure, *, pressure_grid=None):
        best_s, best_p = 0.0, 0.0
        for i, P in enumerate(self._candidate_pressures(phase, pressure, pressure_grid)):
            sim = simulate_training_pattern(phase, P, d_grid, refl=refl,
                                            fwhm_d=self.fwhm_d, fwhm_q=self.fwhm_q)
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

    def __init__(self, model_path: "str | Path", *, fwhm_d: float = 0.03,
                 fwhm_q: "Optional[float]" = None, batch_size: int = 256):
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
        self.fwhm_q = float(fwhm_q) if fwhm_q else None
        self.batch_size = max(1, int(batch_size))

    def _simulate(self, phase, P, refl, d_grid) -> np.ndarray:
        return simulate_training_pattern(phase, P, d_grid, refl=refl,
                                         fwhm_d=self.fwhm_d,
                                         fwhm_q=self.fwhm_q).astype("f4")

    def _forward(self, pairs: np.ndarray) -> np.ndarray:
        """Model scores for a stack of (n, 2, P) pairs, chunked to batch_size."""
        t = self._torch
        out = np.zeros(pairs.shape[0], dtype="f4")
        with t.no_grad():
            for a in range(0, pairs.shape[0], self.batch_size):
                xb = t.from_numpy(pairs[a:a + self.batch_size])
                out[a:a + self.batch_size] = self.model(xb).reshape(-1).cpu().numpy()
        return out

    def score(self, meas, phase, refl, d_grid, pressure, *, pressure_grid=None):
        m = np.asarray(meas, dtype="f4")
        Ps = self._candidate_pressures(phase, pressure, pressure_grid)
        pairs = np.stack([np.stack([m, self._simulate(phase, P, refl, d_grid)])
                          for P in Ps])
        scores = self._forward(pairs)
        best = int(np.argmax(scores))
        return float(scores[best]), float(Ps[best])

    def score_frame(self, meas, phases, refls, d_grid, pressure, *,
                    pressure_grid=None):
        """All candidates of one frame in batched forward passes (one model call
        per ``batch_size`` pairs instead of one per candidate×pressure)."""
        m = np.asarray(meas, dtype="f4")
        rows: "List[np.ndarray]" = []
        owner: "List[Tuple[int, float]]" = []       # (candidate index, pressure)
        for ci, (ph, rf) in enumerate(zip(phases, refls)):
            for P in self._candidate_pressures(ph, pressure, pressure_grid):
                rows.append(np.stack([m, self._simulate(ph, P, rf, d_grid)]))
                owner.append((ci, float(P)))
        scores = self._forward(np.stack(rows)) if rows else np.zeros(0, "f4")
        best: "List[Tuple[float, float]]" = [(0.0, 0.0)] * len(phases)
        seen = [False] * len(phases)
        for (ci, P), s in zip(owner, scores):
            if not seen[ci] or s > best[ci][0]:
                best[ci] = (float(s), P)
                seen[ci] = True
        return best


def make_scorer(spec: "PhaseScorer | str | Dict[str, Any] | None" = None, *,
                fwhm_d: float = 0.03,
                fwhm_q: "Optional[float]" = None) -> PhaseScorer:
    """Resolve a scorer spec to an instance.

    ``None``/``"cosine"`` → :class:`CosineScorer` (the default);
    ``"torch:<model_path>"`` or ``{"kind": "torch", "model": <path>}`` →
    :class:`TorchScorer`; an existing :class:`PhaseScorer` passes through.
    ``fwhm_q`` (q-constant width, Å⁻¹) takes precedence over ``fwhm_d`` for
    candidate simulation. Raises an instructive error (never a bare ImportError)
    when a learned scorer is requested without its prerequisites.
    """
    if spec is None:
        return CosineScorer(fwhm_d=fwhm_d, fwhm_q=fwhm_q)
    if isinstance(spec, PhaseScorer):
        return spec
    if isinstance(spec, str):
        s = spec.strip()
        if s in ("", "cosine"):
            return CosineScorer(fwhm_d=fwhm_d, fwhm_q=fwhm_q)
        if s.startswith("torch:"):
            return TorchScorer(s.split(":", 1)[1], fwhm_d=fwhm_d, fwhm_q=fwhm_q)
        if s == "torch":
            raise RuntimeError(
                "A learned scorer needs a model path: use 'torch:<model_path>' "
                "(or a {'kind': 'torch', 'model': ...} spec).")
        raise ValueError(f"Unknown scorer spec {spec!r} (use 'cosine' or 'torch:<path>').")
    if isinstance(spec, dict):
        kind = str(spec.get("kind", "cosine")).strip().lower()
        kw = {"fwhm_d": float(spec.get("fwhm_d", fwhm_d)),
              "fwhm_q": spec.get("fwhm_q", fwhm_q)}
        if kind == "cosine":
            return CosineScorer(**kw)
        if kind == "torch":
            model = spec.get("model", "")
            if not model:
                raise RuntimeError(
                    "A learned scorer needs a model path: {'kind': 'torch', "
                    "'model': <path>}.")
            return TorchScorer(model, **kw)
        raise ValueError(f"Unknown scorer kind {kind!r} (use 'cosine' or 'torch').")
    raise TypeError(f"Cannot resolve a scorer from {type(spec).__name__}.")
