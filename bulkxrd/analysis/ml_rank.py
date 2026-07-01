"""Step 3b, part 3 — deterministic candidate ranker (ML proposes, physics verifies).

The seam this pipeline protects: a fast *proposer* shortlists which phases plausibly
explain a frame, and the deterministic Step-3a EOS matcher (:mod:`identify`) then
*verifies* each candidate by fitting it. This is the DARA / RADAR-PD lesson —
search-match (or ML) proposes a handful of hypotheses; physics-constrained
refinement decides. DARA's proposer is itself peak-matching scoring, so the
default ranker here is deliberately deterministic and dependency-free (no
torch). The similarity function itself lives behind the :mod:`ml_scorer` seam —
pass ``scorer=`` to :func:`rank_candidates` to swap in a learned RADAR-PD-style
scorer (``bulkxrd[ml]``) without moving the seam.

Per frame:
  1. Take the measured pattern on the shared d-grid — the **residual** by default
     (RADAR-PD ranks against what the known phases left behind, surfacing
     impurities/unknowns), or the Step-2 fit source.
  2. Simulate each library phase at the frame's **pressure** (the metadata prior),
     so candidates are already pressure-aligned — the analog of RADAR-PD's
     lattice-nudge. Where pressure is unknown, scan a coarse pressure grid and
     keep the phase's best.
  3. Score similarity (cosine over the candidate's support) and keep the top-K.

Writes ``/ml/candidates`` and returns the union of per-frame top-K names, which the
worker feeds to :func:`identify.run_identification` as the candidate set.

Pure numpy + h5py; pymatgen only to simulate reflections (cached once, injectable
for tests). Reuses :mod:`ml_features` and :mod:`mldata`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phases import Phase, pymatgen_available
from .identify import phase_reflections, _h5_safe
from .ml_features import frame_features
from .ml_scorer import PhaseScorer, CosineScorer, make_scorer

SCHEMA_VERSION = "1"
Reflections = Tuple[np.ndarray, np.ndarray, list]


def score_phase(meas: np.ndarray, phase: Phase, refl: Reflections, d_grid: np.ndarray,
                pressure: "Optional[float]", *, fwhm_d: float = 0.03,
                pressure_grid: "Optional[np.ndarray]" = None) -> Tuple[float, float]:
    """Best similarity of ``meas`` to ``phase`` and the pressure that achieved it.

    Back-compat wrapper over :class:`ml_scorer.CosineScorer` (the scoring seam —
    inject a different :class:`ml_scorer.PhaseScorer` into :func:`rank_candidates`
    to change how similarity is computed)."""
    return CosineScorer(fwhm_d=fwhm_d).score(meas, phase, refl, d_grid, pressure,
                                             pressure_grid=pressure_grid)


def rank_candidates(
    analysis_h5: "str | Path",
    phases: "Sequence[Phase]",
    *,
    source: str = "auto",
    top_k: int = 5,
    fwhm_d: float = 0.03,
    pressure_grid: "Optional[Sequence[float]]" = None,
    reflections: "Optional[Dict[str, Reflections]]" = None,
    scorer: "PhaseScorer | str | Dict[str, Any] | None" = None,
    out_h5: "Optional[str | Path]" = None,
) -> Dict[str, Any]:
    """Rank library phases per frame and write ``/ml/candidates``.

        /ml/candidates  attrs: schema_version, requested_source, source,
                        resolved_source, top_k, method, fwhm_d, phases,
                        clip_negative, normalize, n_points
        /ml/candidates/<phase>/score     (N,)  per-frame similarity (cosine)
        /ml/candidates/<phase>/pressure  (N,)  pressure the best score used
        /ml/candidates/topk_names        (N, top_k) str   ranked candidate names
        /ml/candidates/topk_score        (N, top_k)       their scores

    ``source="auto"`` ranks against ``/residual/clean`` when present (RADAR-PD —
    the leftover after known phases), else the Step-2 fit source. Three source
    attrs are recorded so a learned model can reproduce the exact preprocessing:
    ``requested_source`` (this argument, e.g. ``auto``), ``source`` (the rank
    level it mapped to, ``residual`` | ``fit``), and ``resolved_source`` (the
    actual channel the features came from, e.g. ``sigmaclip`` — ``fit`` resolves
    to whatever Step 2 recorded). Returns a manifest whose ``candidates`` is the
    union of per-frame top-K names — feed it to
    :func:`identify.run_identification` as the candidate set (ML proposes →
    physics verifies). pymatgen is required unless ``reflections`` is supplied.

    ``scorer`` is the similarity seam (:mod:`ml_scorer`): default the
    deterministic :class:`ml_scorer.CosineScorer`; pass a
    :class:`ml_scorer.PhaseScorer` instance or a spec (``"cosine"``,
    ``"torch:<model_path>"``) to swap in a learned scorer. Whatever the scorer
    proposes, Step 3a still verifies.
    """
    import h5py  # type: ignore

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    dst = Path(out_h5).expanduser().resolve() if out_h5 else src

    phases = [p for p in phases if p.has_structure() or p.name in (reflections or {})]
    if not phases:
        raise ValueError("No simulatable phases to rank (need a structure, or supply "
                         "reflections=).")

    # Choose the ranking source: residual if present, else the recorded fit source.
    requested = str(source or "auto")
    want = requested
    if requested == "auto":
        with h5py.File(str(src), "r") as h5:
            want = "residual" if (h5.get("residual") is not None
                                  and "clean" in h5["residual"]) else "fit"
    feats = frame_features(src, source=want, clip_negative=True)
    meas, grid, pressure = feats.X, feats.d_grid, feats.pressure
    n = feats.n_frames
    excluded = feats.excluded
    prep = feats.preprocessing()           # source/clip_negative/normalize/... for provenance

    # Reflections cached once in the parent (workers/scoring need no pymatgen).
    if reflections is None:
        if not pymatgen_available():
            raise RuntimeError("pymatgen is required to simulate phases for ranking "
                               "(pip install pymatgen), or pass reflections=.")
        reflections = {}
        kept = []
        for ph in phases:
            try:
                reflections[ph.name] = phase_reflections(ph)
                kept.append(ph)
            except Exception as e:
                print(f"[ML-RANK] skipped {ph.name!r}: simulation failed ({e})", flush=True)
        phases = kept
        if not phases:
            raise ValueError("No phases could be simulated for ranking.")

    pg = (np.asarray(pressure_grid, float) if pressure_grid is not None
          else np.arange(0.0, 101.0, 10.0))
    names = [p.name for p in phases]
    refls = [reflections[nm] for nm in names]
    score = {nm: np.zeros(n, "f8") for nm in names}
    pmat = {nm: np.full(n, np.nan, "f8") for nm in names}

    # The scoring seam: deterministic cosine unless an alternative is injected.
    the_scorer = make_scorer(scorer, fwhm_d=fwhm_d)
    print(f"[ML-RANK] {len(phases)} phase(s), {n} frames, source={want}, "
          f"scorer={the_scorer.name}, top_k={top_k}", flush=True)
    k = min(int(top_k), len(phases))
    topk_names = np.full((n, k), "", dtype=object)
    topk_score = np.zeros((n, k), "f8")
    for i in range(n):
        if excluded[i] or not np.isfinite(meas[i]).any() or meas[i].max() <= 0:
            continue
        P = pressure[i] if np.isfinite(pressure[i]) else None
        pairs = the_scorer.score_frame(meas[i], phases, refls, grid, P,
                                       pressure_grid=pg)
        row = []
        for nm, (s, p) in zip(names, pairs):
            score[nm][i] = s
            pmat[nm][i] = p
            row.append((s, nm))
        row.sort(key=lambda t: -t[0])
        for r in range(k):
            topk_score[i, r] = row[r][0]
            topk_names[i, r] = row[r][1]

    _write_candidates(src, dst, names, score, pmat, topk_names, topk_score,
                      requested, want, top_k, fwhm_d, prep, the_scorer.name)

    # Union of the per-frame top-K (live frames) — the shortlist for the verifier.
    live = ~excluded
    shortlist = sorted({nm for i in range(n) if live[i] for nm in topk_names[i] if nm})
    manifest = {
        "tool_version": SCHEMA_VERSION, "source": str(src), "out_h5": str(dst),
        "requested_source": requested, "ranking_source": want,
        "resolved_source": feats.source,
        "n_frames": int(n), "top_k": int(top_k),
        "phases": names, "candidates": shortlist,
    }
    print(f"[ML-RANK] done -> {dst}  (shortlist: {shortlist})", flush=True)
    return manifest


def _write_candidates(src, dst, names, score, pmat, topk_names, topk_score,
                      requested_source, source, top_k, fwhm_d, prep,
                      method: str = "cosine"):
    import os
    import shutil
    import h5py  # type: ignore
    tmp = Path(dst).with_name(Path(dst).name + ".tmp")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            if "ml" in o and "candidates" in o["ml"]:
                del o["ml"]["candidates"]
            gml = o.require_group("ml")
            g = gml.create_group("candidates")
            g.attrs.update({"schema_version": SCHEMA_VERSION,
                            # Source provenance, most abstract to most concrete:
                            # what was asked for, the rank level it mapped to,
                            # and the channel the features actually came from.
                            "requested_source": str(requested_source),
                            "source": str(source),
                            "resolved_source": str(prep.get("source", source)),
                            "top_k": int(top_k), "method": str(method),
                            "fwhm_d": float(fwhm_d), "phases": ", ".join(names),
                            # ML preprocessing provenance (same pipeline a model must use).
                            "clip_negative": bool(prep.get("clip_negative", True)),
                            "normalize": str(prep.get("normalize", "max")),
                            "n_points": int(prep.get("n_points", 0)),
                            "wavelength": float(prep.get("wavelength", 0.0))})
            for nm in names:
                gp = g.create_group(_h5_safe(nm))
                gp.attrs["name"] = nm
                gp.create_dataset("score", data=score[nm])
                gp.create_dataset("pressure", data=pmat[nm])
            sdt = h5py.string_dtype(encoding="utf-8")
            g.create_dataset("topk_names",
                             data=np.array([[str(x) for x in row] for row in topk_names],
                                           dtype=object), dtype=sdt)
            g.create_dataset("topk_score", data=np.asarray(topk_score, "f8"))
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def read_candidates(analysis_h5: "str | Path") -> Dict[str, Any]:
    """Read ``/ml/candidates`` back for the GUI / verifier.

    Returns ``{ok, error, requested_source, source, resolved_source, top_k,
    n_frames, phases, topk_names, topk_score, shortlist}`` (``phases`` maps
    name -> {score, pressure} arrays). Files written before the provenance split
    lack requested/resolved attrs; those fall back to ``source``."""
    import h5py  # type: ignore
    p = Path(analysis_h5).expanduser()
    out: Dict[str, Any] = {"ok": False, "error": "", "requested_source": "",
                           "source": "", "resolved_source": "", "top_k": 0,
                           "n_frames": 0, "phases": {}, "topk_names": None,
                           "topk_score": None, "shortlist": []}
    if not p.is_file():
        out["error"] = f"File not found: {p}"
        return out
    try:
        with h5py.File(str(p), "r") as h5:
            g = h5.get("ml/candidates")
            if g is None:
                out["error"] = "No /ml/candidates — run ML candidate ranking first."
                return out
            out["source"] = str(g.attrs.get("source", ""))
            out["requested_source"] = str(g.attrs.get("requested_source", out["source"]))
            out["resolved_source"] = str(g.attrs.get("resolved_source", out["source"]))
            out["top_k"] = int(g.attrs.get("top_k", 0))
            if "topk_names" in g:
                tn = [[x.decode() if isinstance(x, bytes) else str(x) for x in row]
                      for row in g["topk_names"][:]]
                out["topk_names"] = tn
                out["topk_score"] = np.asarray(g["topk_score"][:], float) if "topk_score" in g else None
                out["n_frames"] = len(tn)
                out["shortlist"] = sorted({nm for row in tn for nm in row if nm})
            for key in g:
                sub = g[key]
                if hasattr(sub, "attrs") and "score" in getattr(sub, "keys", lambda: [])():
                    out["phases"][str(sub.attrs.get("name", key))] = {
                        "score": np.asarray(sub["score"][:], float),
                        "pressure": np.asarray(sub["pressure"][:], float),
                    }
            out["ok"] = True
    except Exception as e:
        out["error"] = f"Failed to read candidates: {e!r}"
    return out
