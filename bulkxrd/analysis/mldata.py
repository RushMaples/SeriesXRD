"""ML dataset export for Step 3b (compound identification by machine learning).

Turns the analysis HDF5 into model-ready tensors and, separately, synthesizes a
matching simulated + pressure-augmented training set through the *same*
preprocessing path — so there is no sim-to-real gap (the lesson from SimXRD-4M:
preprocess the training data exactly like the experimental data).

Representation choices follow the literature gathered for this pipeline:
  * Full resampled pattern on a fixed, wavelength-independent d-grid
    (SimXRD format: d ≈ 1.199–8.853 Å, 3501 points) — the primary input; models
    that keep peak position + relative intensity (no pooling) generalise best.
  * Optional extra channels (e.g. ``spot_residual``, the azimuthal mean−median
    texture signal) stacked alongside ``clean``.
  * Conditioning: wavelength, per-phase Step-3a pressure, candidate-phase
    multi-hot, and per-frame quality.
  * Weak multi-labels from the deterministic Step-3a match (confidence ≥ thr).

Exports a portable ``.npz`` (load anywhere, e.g. the WashU RIS GPU cluster).
Pure numpy (+ h5py lazy). Simulation reuses ``identify``/``phases`` (pymatgen).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phases import Phase
from .identify import radial_to_d, scale_at_pressure, phase_reflections
from .peaks import pseudo_voigt

# SimXRD-4M pattern format (Cao et al., ICLR 2025).
D_MIN, D_MAX, N_POINTS = 1.199, 8.853, 3501


def make_d_grid(d_min: float = D_MIN, d_max: float = D_MAX,
                n_points: int = N_POINTS) -> np.ndarray:
    """The fixed d-spacing grid (Å) shared by experimental and simulated data."""
    return np.linspace(float(d_min), float(d_max), int(n_points))


def resample_to_d(radial, intensity, unit: str, wavelength: "Optional[float]",
                  d_grid: np.ndarray) -> np.ndarray:
    """Resample one intensity row from the reduced radial axis onto ``d_grid``.

    Converts the radial axis to d-spacing, sorts ascending, and linearly
    interpolates onto the grid (0 outside the measured range).
    """
    d_axis = radial_to_d(np.asarray(radial, float), unit, wavelength)
    y = np.asarray(intensity, float)
    finite = np.isfinite(d_axis) & np.isfinite(y)
    d_axis, y = d_axis[finite], y[finite]
    order = np.argsort(d_axis)
    return np.interp(d_grid, d_axis[order], y[order], left=0.0, right=0.0)


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    mx = np.nanmax(X, axis=-1, keepdims=True)
    mx[mx <= 0] = 1.0
    return X / mx


# ---------------------------------------------------------------------------
# Experimental export
# ---------------------------------------------------------------------------

def _read_identify(h5) -> Dict[str, Dict[str, np.ndarray]]:
    gid = h5.get("identify")
    out: Dict[str, Dict[str, np.ndarray]] = {}
    if gid is None:
        return out
    for key in gid:
        g = gid[key]
        if not hasattr(g, "keys") or "pressure" not in g:
            continue
        name = str(g.attrs.get("name", key))
        out[name] = {
            "pressure": np.asarray(g["pressure"][:], dtype=float),
            "confidence": np.asarray(g["confidence"][:], dtype=float) if "confidence" in g else None,
        }
    return out


def _good_peaks_per_frame(h5, n: int) -> np.ndarray:
    pk = h5.get("peaks")
    if pk is None or "frame" not in pk:
        return np.zeros(n, "i4")
    frame = np.asarray(pk["frame"][:], dtype=int)
    flag = np.asarray(pk["flag"][:], dtype=int) if "flag" in pk else np.zeros_like(frame)
    counts = np.zeros(n, "i4")
    for f in frame[flag == 0]:
        if 0 <= f < n:
            counts[f] += 1
    return counts


def export_ml_dataset(
    analysis_h5: "str | Path",
    out_npz: "str | Path",
    *,
    channels: "Sequence[str]" = ("clean",),
    d_grid: "Optional[np.ndarray]" = None,
    conf_threshold: float = 0.5,
    wavelength: "Optional[float]" = None,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Export experimental frames as an ML-ready ``.npz``.

    Saved arrays: ``X`` (N, C, P) resampled patterns; ``d_grid`` (P,);
    ``channels`` (C,); ``frame_index`` (N,); ``y`` (N, n_phases) weak multi-labels
    and ``phase_names`` (n_phases,) from Step-3a (omitted if no /identify);
    ``pressure`` (N, n_phases); ``n_good_peaks`` (N,); ``contamination`` (N,) if
    present; plus scalar meta (``unit``, ``wavelength``, ``conf_threshold``).
    Returns a manifest dict.
    """
    import h5py  # type: ignore

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    grid = make_d_grid() if d_grid is None else np.asarray(d_grid, float)
    chans = [c for c in channels if c in ("clean", "baseline", "spot_residual")]
    if not chans:
        raise ValueError("No valid channels (choose from clean/baseline/spot_residual).")

    with h5py.File(str(src), "r") as h5:
        unit = str(h5.attrs.get("unit", ""))
        bg = h5.get("background")
        if bg is None or "clean" not in bg:
            raise ValueError("No /background/clean — run Step 1 first.")
        radial = np.asarray(h5["radial"][:], dtype=float)
        if wavelength is None and unit.strip().lower() in ("2th_deg", "2th_rad"):
            gid = h5.get("identify")
            wavelength = float(gid.attrs["wavelength"]) if gid is not None and gid.attrs.get("wavelength") else None
            if not wavelength:
                raise ValueError("2θ axis needs a wavelength (none stored); pass wavelength=.")
        stacks = {c: np.asarray(bg[c][:], dtype=float) for c in chans}
        n = stacks[chans[0]].shape[0]
        ident = _read_identify(h5)
        n_good = _good_peaks_per_frame(h5, n)
        frames = h5.get("frames")
        contamination = (np.asarray(frames["contamination"][:], dtype=float)
                         if frames is not None and "contamination" in frames else None)

    # Resample every frame × channel onto the d-grid.
    X = np.zeros((n, len(chans), grid.size), dtype="f4")
    for ci, c in enumerate(chans):
        st = stacks[c]
        for i in range(n):
            X[i, ci] = resample_to_d(radial, st[i], unit, wavelength, grid)
    if normalize:
        for ci in range(len(chans)):
            X[:, ci, :] = _normalize_rows(X[:, ci, :])

    save: Dict[str, Any] = {
        "X": X, "d_grid": grid, "channels": np.array(chans, dtype=object),
        "frame_index": np.arange(n), "unit": unit,
        "wavelength": float(wavelength) if wavelength else 0.0,
        "n_good_peaks": n_good, "conf_threshold": float(conf_threshold),
    }
    if contamination is not None:
        save["contamination"] = contamination
    phase_names: List[str] = sorted(ident.keys())
    if phase_names:
        y = np.zeros((n, len(phase_names)), dtype="i1")
        pressure = np.full((n, len(phase_names)), np.nan, dtype="f4")
        for j, name in enumerate(phase_names):
            conf = ident[name]["confidence"]
            if conf is not None:
                y[:, j] = (conf >= conf_threshold).astype("i1")
            pressure[:, j] = ident[name]["pressure"]
        save["y"] = y
        save["phase_names"] = np.array(phase_names, dtype=object)
        save["pressure"] = pressure

    out = Path(out_npz).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, **save)
    import os
    os.replace(tmp, out)

    manifest = {
        "out_npz": str(out), "n_frames": int(n), "n_channels": len(chans),
        "channels": chans, "n_points": int(grid.size), "unit": unit,
        "n_phases": len(phase_names), "phases": phase_names,
        "has_labels": bool(phase_names),
    }
    print(f"[MLDATA] exported {n} frames × {len(chans)} ch × {grid.size} pts -> {out}", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# Simulated + pressure-augmented training set (matched preprocessing)
# ---------------------------------------------------------------------------

def simulate_training_pattern(phase: Phase, pressure: float, d_grid: np.ndarray,
                              *, refl=None, fwhm_d: float = 0.03, eta: float = 0.5,
                              normalize: bool = True) -> np.ndarray:
    """A single-phase synthetic pattern on ``d_grid`` at ``pressure`` (GPa).

    Predicted reflections (d0·s(P)) are rendered as pseudo-Voigt peaks of width
    ``fwhm_d`` (in Å) weighted by relative intensity — the same profile the
    experimental peaks are fit with, so the two domains match.
    """
    if refl is None:
        refl = phase_reflections(phase)
    d0, w, _ = refl
    s = scale_at_pressure(phase, pressure)
    centers = d0 * s
    y = np.zeros_like(d_grid, dtype=float)
    for c, a in zip(centers, w):
        if d_grid[0] <= c <= d_grid[-1]:
            y += pseudo_voigt(d_grid, c, a, fwhm_d, eta)
    if normalize:
        mx = y.max()
        if mx > 0:
            y = y / mx
    return y


def build_simulated_dataset(
    phases: "Sequence[Phase]",
    *,
    pressures: "Optional[Sequence[float]]" = None,
    d_grid: "Optional[np.ndarray]" = None,
    fwhm_d: float = 0.03,
    eta: float = 0.5,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Single-phase pressure-augmented training set: scan each phase over the
    pressure grid → ``(X (M, P), y (M,) int label index, phase_names)``.

    Phases without an EOS are emitted at ambient only (one row). Requires
    pymatgen for the reflection simulation.
    """
    grid = make_d_grid() if d_grid is None else np.asarray(d_grid, float)
    if pressures is None:
        pressures = np.arange(0.0, 101.0, 5.0)
    pressures = np.asarray(pressures, float)
    names = [p.name for p in phases]
    rows: List[np.ndarray] = []
    labels: List[int] = []
    for j, ph in enumerate(phases):
        refl = phase_reflections(ph)
        ps = pressures if ph.has_eos() else np.array([0.0])
        for P in ps:
            rows.append(simulate_training_pattern(ph, float(P), grid, refl=refl,
                                                  fwhm_d=fwhm_d, eta=eta, normalize=normalize))
            labels.append(j)
    X = np.asarray(rows, dtype="f4") if rows else np.zeros((0, grid.size), "f4")
    return X, np.asarray(labels, dtype="i4"), names


def export_simulated_dataset(out_npz: "str | Path", phases: "Sequence[Phase]",
                             *, pressures=None, d_grid=None, fwhm_d: float = 0.03,
                             eta: float = 0.5) -> Dict[str, Any]:
    """Build and save the simulated training set as ``.npz`` (``X``, ``y``,
    ``phase_names``, ``d_grid``, ``pressures``). Requires pymatgen."""
    grid = make_d_grid() if d_grid is None else np.asarray(d_grid, float)
    X, y, names = build_simulated_dataset(phases, pressures=pressures, d_grid=grid,
                                          fwhm_d=fwhm_d, eta=eta)
    out = Path(out_npz).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, X=X, y=y, phase_names=np.array(names, dtype=object),
                        d_grid=grid, pressures=np.asarray(pressures if pressures is not None
                                                          else np.arange(0.0, 101.0, 5.0), float))
    import os
    os.replace(tmp, out)
    manifest = {"out_npz": str(out), "n_samples": int(X.shape[0]),
                "n_points": int(grid.size), "phases": names}
    print(f"[MLDATA] simulated set: {X.shape[0]} patterns ({len(names)} phases) -> {out}", flush=True)
    return manifest
