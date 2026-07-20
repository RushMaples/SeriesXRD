"""Known-truth benchmark harness (``seriesxrd-benchmark``).

Ingests external labelled 1D powder patterns — RRUFF exports, opXRD dumps, any
XY text files — through the SAME pipeline preprocessing the experimental data
gets (synthetic reduced HDF5 → Step-1 SNIP → Step-2 peaks → Step-3b ranking),
then scores the ranker against the known labels:

    hit@1 / hit@K   is the true phase the top / a top-K candidate?
    MRR             mean reciprocal rank of the first true phase
    identify hits   (optional, needs pymatgen) does Step 3a verify the truth?

This is the VALIDATION GATE the training guide (docs/ml-training.md) requires: run it once with
the default cosine scorer to pin the deterministic baseline, and again with
``--ml-scorer torch:<model.pt>`` — a trained scorer is only promoted when it
beats the baseline on the same command. Because ingest reuses the pipeline's
own preprocessing, the benchmark numbers transfer to real runs (no separate
sim-to-real gap in the harness itself).

Labels come from a CSV (``filename,phases`` with ``;``-separated multi-phase
labels) whose names must resolve in the reference library — the benchmark
measures the *scorer*, so curate the library to contain the labelled phases
(e.g. imported from the same COD/AMCSD structures RRUFF references).

Pure numpy + h5py; pymatgen only for reflection simulation (injectable for
tests) and the optional identify pass.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phases import Phase
from .frame_metadata import _name_keys

Reflections = Tuple[np.ndarray, np.ndarray, list]

# Cu Kα1 — the overwhelming default for RRUFF/opXRD lab patterns.
CU_KA1 = 1.540598


# ---------------------------------------------------------------------------
# Pattern ingest
# ---------------------------------------------------------------------------

def read_xy_text(path: "str | Path") -> Dict[str, Any]:
    """Parse an XY pattern text file (RRUFF style) → ``{x, y, meta}``.

    Header lines starting with ``#`` may carry ``KEY=VALUE`` metadata (RRUFF
    uses ``##NAMES=...``, ``##WAVELENGTH=...``); data rows are ``x, y`` or
    ``x y``. Non-numeric trailing columns are ignored.
    """
    p = Path(path).expanduser()
    xs: List[float] = []
    ys: List[float] = []
    meta: Dict[str, str] = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if "=" in body:
                k, v = body.split("=", 1)
                meta[k.strip().lower()] = v.strip()
            continue
        parts = s.replace(",", " ").split()
        try:
            x, y = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        xs.append(x)
        ys.append(y)
    return {"x": np.asarray(xs, float), "y": np.asarray(ys, float), "meta": meta}


def load_labels_csv(path: "str | Path") -> Dict[str, List[str]]:
    """``filename,phases`` CSV → {filename: [phase, ...]} (``;``-separated)."""
    out: Dict[str, List[str]] = {}
    with Path(path).expanduser().open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header and header[0].strip().lower() not in ("filename", "file", "name"):
            fh.seek(0)
            reader = csv.reader(fh)
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                phases = [s.strip() for s in row[1].replace(";", ",").split(",")
                          if s.strip()]
                if phases:
                    out[row[0].strip()] = phases
    return out


def ingest_patterns(
    files: "Sequence[str | Path]",
    out_dir: "str | Path",
    *,
    unit: str = "2th_deg",
    wavelength: float = CU_KA1,
    peak_source: str = "auto",
    sensitivity: str = "normal",
) -> Dict[str, Any]:
    """XY files → synthetic reduced HDF5 → Step 1 + Step 2 → analysis HDF5.

    All patterns are resampled onto one common axis (union range, median step,
    NaN outside each file's measured window) so they share a ``/patterns``
    stack, then run through the REAL ``run_background_separation`` and
    ``run_peak_fitting`` — matched preprocessing is the whole point. There is
    no azimuthal dimension in 1D benchmark data, so ``intensity_robust`` =
    ``intensity`` (spot separation is a no-op, as it should be).

    Returns ``{analysis_h5, reduced_h5, files, n}``.
    """
    import h5py  # type: ignore
    from .background import run_background_separation
    from .peaks import run_peak_fitting

    files = [Path(f).expanduser() for f in files]
    if not files:
        raise ValueError("No pattern files to ingest.")
    pats = [read_xy_text(f) for f in files]
    good = [(f, p) for f, p in zip(files, pats) if p["x"].size >= 16]
    if not good:
        raise ValueError("No file yielded a readable XY pattern (>=16 points).")
    files = [f for f, _ in good]
    pats = [p for _, p in good]

    lo = min(float(p["x"].min()) for p in pats)
    hi = max(float(p["x"].max()) for p in pats)
    step = float(np.median([np.median(np.diff(p["x"])) for p in pats]))
    axis = np.arange(lo, hi + 0.5 * step, step)
    stack = np.full((len(pats), axis.size), np.nan, "f4")
    for i, p in enumerate(pats):
        order = np.argsort(p["x"])
        x, y = p["x"][order], p["y"][order]
        row = np.interp(axis, x, y, left=np.nan, right=np.nan)
        row[(axis < x[0]) | (axis > x[-1])] = np.nan
        stack[i] = row

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    reduced = out / "benchmark_reduced.h5"
    with h5py.File(str(reduced), "w") as h5:
        h5.attrs["unit"] = unit
        h5.attrs["poni_text"] = f"wavelength: {wavelength * 1e-10}"
        gp = h5.create_group("patterns")
        gp.create_dataset("intensity", data=stack)
        gp.create_dataset("intensity_robust", data=stack)   # no azimuth in 1D data
        gp.create_dataset("radial", data=axis)
        gf = h5.create_group("frames")
        gf.create_dataset("filename",
                          data=np.array([f.name for f in files], dtype=object),
                          dtype=h5py.string_dtype("utf-8"))
        gf.create_dataset("excluded", data=np.zeros(len(files), "?"))

    m1 = run_background_separation(reduced, out / "benchmark_analysis.h5")
    analysis = m1["out_h5"]
    run_peak_fitting(analysis, None, source=peak_source, sensitivity=sensitivity)
    return {"analysis_h5": str(analysis), "reduced_h5": str(reduced),
            "files": [str(f) for f in files], "n": len(files)}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _rank_metrics(names: "List[str]", scores: "Dict[str, np.ndarray]",
                  truth: "List[List[str]]", top_k: int) -> Dict[str, Any]:
    """hit@1 / hit@K / MRR over frames with at least one resolvable label."""
    hits1 = hitsk = 0
    rr: List[float] = []
    per_frame: List[Dict[str, Any]] = []
    n_scored = 0
    for i, true_names in enumerate(truth):
        wanted = [t for t in true_names if t in names]
        if not wanted:
            per_frame.append({"frame": i, "skipped": "label not in library"})
            continue
        n_scored += 1
        order = sorted(names, key=lambda nm: -float(scores[nm][i]))
        rank = min(order.index(t) for t in wanted) + 1
        hits1 += int(rank == 1)
        hitsk += int(rank <= top_k)
        rr.append(1.0 / rank)
        per_frame.append({"frame": i, "truth": wanted, "rank": rank,
                          "top1": order[0]})
    return {"n_scored": n_scored,
            "hit_at_1": hits1 / n_scored if n_scored else float("nan"),
            f"hit_at_{top_k}": hitsk / n_scored if n_scored else float("nan"),
            "mrr": float(np.mean(rr)) if rr else float("nan"),
            "per_frame": per_frame}


def run_benchmark(
    pattern_files: "Sequence[str | Path]",
    phases: "Sequence[Phase]",
    labels: "Dict[str, List[str]]",
    *,
    out_dir: "str | Path",
    unit: str = "2th_deg",
    wavelength: float = CU_KA1,
    top_k: int = 5,
    scorer=None,
    reflections: "Optional[Dict[str, Reflections]]" = None,
    run_identify: bool = True,
    rel_tol: float = 0.01,
) -> Dict[str, Any]:
    """Ingest → rank → (optionally identify) → scorecard vs the labels.

    ``scorer`` is the ml_scorer seam spec (None/"cosine" or "torch:<pt>") — run
    the SAME command twice to compare a trained scorer against the pinned
    cosine baseline. ``labels`` maps pattern filenames (any of full/base/stem)
    to library phase names. The report JSON lands in ``out_dir``.
    """
    from .ml_rank import rank_candidates
    from .identify import run_identification
    from .phases import pymatgen_available

    out = Path(out_dir).expanduser().resolve()
    ing = ingest_patterns(pattern_files, out, unit=unit, wavelength=wavelength)
    analysis = ing["analysis_h5"]

    # filename → truth, tolerant of path/extension differences.
    label_lut: Dict[str, List[str]] = {}
    for k, v in labels.items():
        for key in _name_keys(k):
            label_lut.setdefault(key, v)
    truth: List[List[str]] = []
    for f in ing["files"]:
        found: List[str] = []
        for key in _name_keys(f):
            if key in label_lut:
                found = label_lut[key]
                break
        truth.append(found)

    man_rank = rank_candidates(analysis, list(phases), source="fit",
                               top_k=top_k, scorer=scorer,
                               reflections=reflections)
    import h5py  # type: ignore
    with h5py.File(analysis, "r") as h5:
        g = h5["ml/candidates"]
        names = [str(g[k].attrs["name"]) for k in g
                 if hasattr(g[k], "attrs") and "name" in g[k].attrs]
        scores = {nm: None for nm in names}
        for k in g:
            sub = g[k]
            if hasattr(sub, "attrs") and "name" in sub.attrs:
                scores[str(sub.attrs["name"])] = np.asarray(sub["score"][:], float)
        method = str(g.attrs.get("method", "cosine"))

    report: Dict[str, Any] = {
        "n_patterns": ing["n"], "scorer": method, "top_k": int(top_k),
        "ranking_source": man_rank["ranking_source"],
        "fwhm_q": man_rank.get("fwhm_q"), "fwhm_q_poly": man_rank.get("fwhm_q_poly"),
        "rank": _rank_metrics(names, scores, truth, top_k),
    }

    if run_identify:
        if not pymatgen_available() and reflections is None:
            report["identify"] = {"skipped": "pymatgen not installed"}
        else:
            try:
                if reflections is not None:
                    # test/injection path: bypass the pymatgen requirement
                    from . import identify as _idf
                    saved = (_idf.pymatgen_available, _idf.phase_reflections)
                    _idf.pymatgen_available = lambda: True
                    _idf.phase_reflections = lambda ph, **kw: reflections[ph.name]
                    try:
                        run_identification(analysis, list(phases),
                                           rel_tol=rel_tol, use_frame_pressure=False)
                    finally:
                        _idf.pymatgen_available, _idf.phase_reflections = saved
                else:
                    run_identification(analysis, list(phases), rel_tol=rel_tol,
                                       use_frame_pressure=False)
                with h5py.File(analysis, "r") as h5:
                    gid = h5["identify"]
                    hits = n_lab = 0
                    for i, tr in enumerate(truth):
                        wanted = [t for t in tr
                                  if any(str(gid[k].attrs.get("name", k)) == t
                                         for k in gid)]
                        if not wanted:
                            continue
                        n_lab += 1
                        best_name, best_conf = "", -1.0
                        for k in gid:
                            sub = gid[k]
                            if not hasattr(sub, "attrs") or "confidence" not in sub:
                                continue
                            c = float(sub["confidence"][i])
                            if c > best_conf:
                                best_conf, best_name = c, str(sub.attrs.get("name", k))
                        hits += int(best_name in wanted and best_conf > 0.3)
                    report["identify"] = {
                        "n_scored": n_lab,
                        "top_confidence_hit_rate": hits / n_lab if n_lab else float("nan"),
                    }
            except Exception as e:
                report["identify"] = {"error": repr(e)}

    rp = out / "benchmark_report.json"
    rp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_json"] = str(rp)
    report["analysis_h5"] = analysis
    r = report["rank"]
    print(f"[BENCH] scorer={method}: hit@1={r['hit_at_1']:.3f} "
          f"hit@{top_k}={r[f'hit_at_{top_k}']:.3f} MRR={r['mrr']:.3f} "
          f"over {r['n_scored']} labelled patterns -> {rp}", flush=True)
    return report


# ---------------------------------------------------------------------------
# CLI  (seriesxrd-benchmark)
# ---------------------------------------------------------------------------

def main(argv: "Optional[List[str]]" = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="seriesxrd-benchmark",
        description="Score the candidate ranker against labelled external XY "
                    "patterns (RRUFF/opXRD exports). Run once for the cosine "
                    "baseline and once with --ml-scorer to gate a trained model.")
    p.add_argument("patterns", help="Directory of XY .txt files (or one file).")
    p.add_argument("--labels", required=True,
                   help="CSV: filename,phases (';'-separated library names).")
    p.add_argument("--workspace", default="", help="Workspace with the phase library.")
    p.add_argument("--out", default="benchmark_out", help="Output directory.")
    p.add_argument("--unit", default="2th_deg", help="Axis unit of the XY files.")
    p.add_argument("--wavelength", type=float, default=CU_KA1,
                   help=f"Å; default Cu Ka1 = {CU_KA1}.")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--ml-scorer", default="",
                   help="'cosine' (default) or 'torch:<model.pt>'.")
    p.add_argument("--no-identify", action="store_true",
                   help="Skip the Step-3a verify metrics (rank-only, no pymatgen).")
    args = p.parse_args(argv)

    from .phases import load_library
    root = Path(args.patterns).expanduser()
    files = sorted(root.rglob("*.txt")) + sorted(root.rglob("*.xy")) \
        if root.is_dir() else [root]
    if not files:
        print(f"[ERROR] no .txt/.xy patterns under {root}", flush=True)
        return 1
    lib = load_library(args.workspace or Path.cwd())
    pool = [ph for ph in lib.values() if ph.has_structure()]
    if not pool:
        print("[ERROR] no simulatable phases in the library.", flush=True)
        return 1
    labels = load_labels_csv(args.labels)
    if not labels:
        print(f"[ERROR] no usable rows in {args.labels}", flush=True)
        return 1
    try:
        run_benchmark(files, pool, labels, out_dir=args.out, unit=args.unit,
                      wavelength=args.wavelength, top_k=args.top_k,
                      scorer=(args.ml_scorer or None),
                      run_identify=not args.no_identify)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
