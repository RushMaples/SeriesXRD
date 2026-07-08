"""Refinement hand-off bundle (roadmap: continue in Rietveld software).

BulkXRD's Step 3 identifies phases and estimates per-frame pressure, but it is
not a Rietveld engine — a user who wants refined lattice parameters, site
occupancies, or texture needs to hand the result to GSAS-II (or similar) by
hand. This module packages everything that step needs into one directory:

  * ``patterns/frame_####.xy``    per-frame pattern on a 2theta(deg) axis
                                   (direct if the analysis file is already on
                                   2theta; converted from q via
                                   ``2theta = 2*asin(lambda*q/4*pi)`` when a
                                   wavelength is on file). Skipped when 2theta
                                   can't be derived (q axis, no wavelength).
  * ``patterns/frame_####_q.xy``  the SAME frame on its NATIVE radial axis,
                                   always written, so nothing is lost when the
                                   2theta file above was skipped.
  * ``phases/<name>.cif``         one CIF per identified phase, copied from
                                   the phase's own file or synthesized from
                                   its lattice/atoms/space_group (pymatgen).
                                   A phase that resolves to neither is
                                   recorded in ``phases_skipped`` — the whole
                                   export never fails for one bad phase.
  * ``instrument.instprm``        a minimal GSAS-II instrument-parameter file
                                   (only when a wavelength is known).
  * ``README.md``                 what's in the bundle + a runnable
                                   GSASIIscriptable snippet to load it.

References: B. H. Toby & R. B. Von Dreele, J. Appl. Cryst. 46 (2013) 544
(GSAS-II; the ``instrument.instprm`` here follows its documented CW powder
instrument-parameter file layout, with placeholder Caglioti U/V/W meant to be
refined — see G. Caglioti, A. Paoletti & F. P. Ricci, Nucl. Instrum. 3 (1958)
223 for the resolution form).

The pattern channel mirrors the Step-2 fit-source reconstruction used
elsewhere in the analysis stage (see ``heatmap._peaks_fit_source`` /
``heatmap.pattern_image``) so the exported pattern is exactly what Step 2 (and
therefore Step 3) actually saw — not a re-derived approximation.

Pure numpy; h5py and pymatgen are both imported lazily (pymatgen only when a
phase needs a CIF synthesized from lattice/atoms rather than copied).
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .heatmap import _peaks_fit_source
from .peaks import build_fit_source
from .phases import Phase, load_library, pymatgen_available, structure_from_phase

# q-axis unit spellings understood by identify.radial_to_d / heatmap — kept in
# sync so the same file is never treated as q by one module and something else
# by another.
_Q_A_UNITS = ("q_a^-1", "q_a-1", "q_a", "q")
_Q_NM_UNITS = ("q_nm^-1", "q_nm-1", "q_nm")
_TWOTH_UNITS = ("2th_deg", "2th_rad")


# ---------------------------------------------------------------------------
# Pattern channel reconstruction
# ---------------------------------------------------------------------------

def _pattern_source(h5, source: str) -> "Tuple[np.ndarray, str]":
    """(N_frames, N_bins) intensity for the requested channel, plus a resolved
    label for the file header.

    ``source="fit"`` (default) reconstructs exactly the Step-2 fit source —
    the same reconstruction ``heatmap._peaks_fit_source`` uses for the phase
    layers, so the exported pattern matches what Step 2/3 actually fit.
    ``"robust"`` mirrors ``heatmap.pattern_image``'s ``clean + baseline``
    reconstruction (the one channel :func:`peaks.build_fit_source` doesn't
    cover). Everything else (``clean``/``mean``/``hybrid``/``sigmaclip``/
    ``auto``) is delegated straight to :func:`peaks.build_fit_source`.
    """
    s = (source or "fit").strip().lower()
    if s == "fit":
        pk = h5.get("peaks")
        label = str(pk.attrs.get("source", "clean")) if pk is not None else "clean"
        return _peaks_fit_source(h5), f"fit:{label}"
    bg = h5.get("background")
    if bg is None or "clean" not in bg:
        raise ValueError("No /background/clean — run Step 1 first.")
    clean = np.asarray(bg["clean"][:], dtype=float)
    if s == "robust":  # = clean + baseline; not one of build_fit_source's channels
        if "baseline" not in bg:
            raise ValueError("/background/baseline not present.")
        return clean + np.asarray(bg["baseline"][:], dtype=float), "robust"
    spot = np.asarray(bg["spot_residual"][:], dtype=float) if "spot_residual" in bg else None
    sc = (np.asarray(bg["sigmaclip_residual"][:], dtype=float)
          if "sigmaclip_residual" in bg else None)
    data, resolved = build_fit_source(s, clean, spot_residual=spot, sigmaclip_residual=sc)
    return np.asarray(data, dtype=float), resolved


def _to_two_theta_deg(radial: np.ndarray, unit: str,
                      wavelength: "Optional[float]") -> "Optional[np.ndarray]":
    """2theta (deg) from the native radial axis, or ``None`` when it can't be
    derived (a q axis with no wavelength on file). Mirrors the unit handling
    of :func:`identify.radial_to_d` (same accepted unit spellings)."""
    u = (unit or "").strip().lower()
    r = np.asarray(radial, dtype=float)
    if u == "2th_deg":
        return r
    if u == "2th_rad":
        return np.degrees(r)
    if u in _Q_A_UNITS or u in _Q_NM_UNITS:
        if not wavelength or wavelength <= 0:
            return None
        q_a = r * 0.1 if u in _Q_NM_UNITS else r          # -> Angstrom^-1
        arg = np.clip(float(wavelength) * q_a / (4.0 * math.pi), -1.0, 1.0)
        return np.degrees(2.0 * np.arcsin(arg))
    return None


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _xy_header(*, axis: str, native_unit: str, wavelength: "Optional[float]",
              source_label: str, filename: str) -> str:
    wl = f"{float(wavelength):.6f}" if wavelength else "unknown"
    return (
        "BulkXRD refinement-export pattern\n"
        f"axis: {axis}\n"
        f"native_unit: {native_unit}\n"
        f"wavelength_A: {wl}\n"
        f"source_channel: {source_label}\n"
        f"original_filename: {filename}"
    )


def _write_xy(path: Path, x: np.ndarray, y: np.ndarray, *, header: str) -> None:
    np.savetxt(str(path), np.column_stack([np.asarray(x, float), np.asarray(y, float)]),
               header=header, comments="# ", fmt="%.8f")


_INSTPRM_HEADER = "#GSAS-II instrument parameter file; do not add/delete items!"


def _write_instprm(path: Path, wavelength: float) -> None:
    lines = [
        _INSTPRM_HEADER,
        "Type:PXC",
        f"Lam:{float(wavelength):.6f}",
        "Zero:0.0",
        "Polariz.:0.99",
        "U:2.0",
        "V:-2.0",
        "W:5.0",
        "X:0.0",
        "Y:0.0",
        "SH/L:0.002",
        "Azimuth:0.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(path: Path, *, unit: str, wavelength: "Optional[float]",
                  n_phases_written: int, n_phases_skipped: int) -> None:
    if wavelength:
        instprm_note = f"written — Lam = {float(wavelength):.6g} A"
        instprm_arg = '"instrument.instprm"'
    else:
        instprm_note = "NOT written — no wavelength on file; build one yourself before refining"
        instprm_arg = "None  # no instrument.instprm was written (wavelength unknown)"

    text = f"""# BulkXRD refinement hand-off bundle

This directory is a self-contained hand-off for Rietveld refinement (e.g. in
GSAS-II) of the phases BulkXRD identified in this dataset. Nothing past this
point depends on BulkXRD.

## Contents

- `patterns/frame_####.xy` -- two-column (2theta_deg, intensity) pattern per
  exported frame. Written only when 2theta is derivable: directly, if the
  analysis file's native axis is already 2theta; converted from q via
  `2theta = 2*asin(lambda*q/4*pi)` when a wavelength is on file.
- `patterns/frame_####_q.xy` -- the same frame on its NATIVE radial axis
  (unit: `{unit}`), always written, so nothing is lost if the 2theta file
  above wasn't produced.
- `phases/*.cif` -- {n_phases_written} phase CIF(s) written ({n_phases_skipped}
  skipped -- see `phases_skipped` in the export manifest for why).
- `instrument.instprm` -- {instprm_note}. Only `Lam` (wavelength) and `Zero`
  are real measured/derived values; **`U`/`V`/`W` are placeholder Caglioti
  peak-shape values, not measurements** -- refine them, don't trust them.

## Continue in GSAS-II (GSASIIscriptable)

```python
import glob
import GSASIIscriptable as G2sc

gpx = G2sc.G2Project(newgpx="refinement.gpx")
instprm = {instprm_arg}

for xy in sorted(glob.glob("patterns/frame_*.xy")):
    if xy.endswith("_q.xy"):
        continue  # native-axis copy; GSAS-II wants the 2theta file
    gpx.add_powder_histogram(xy, instprm)

for cif in sorted(glob.glob("phases/*.cif")):
    gpx.add_phase(cif, histograms=gpx.histograms())

for hist in gpx.histograms():
    hist.set_refinements({{"Background": {{"no. coeffs": 3, "refine": True}},
                          "Cell": True}})

gpx.save()
```

Start with background + cell, then bring in peak shape (`U`/`V`/`W`) and
atomic parameters once the pattern lines up -- the usual GSAS-II refinement
order.
"""
    path.write_text(text, encoding="utf-8")


def _safe_filename(name: str) -> str:
    keep = "-_.() "
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in str(name)).strip()
    return cleaned or "phase"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _write_peaks_csv(h5, csv_path: Path, frame_idx: "Sequence[int]",
                     filenames: "Sequence[str]") -> "Optional[int]":
    """One combined CSV of every fitted peak in ``frame_idx`` (flagged peaks
    included with their flag value; the ``phase`` column appears when the
    Step-3a-removal attribution exists). Returns the row count, or None when
    the file has no /peaks (Step 2 not run)."""
    import csv as _csv
    pk = h5.get("peaks")
    if pk is None or "center" not in pk:
        return None
    want = set(int(i) for i in frame_idx)
    pk_frame = np.asarray(pk["frame"][:], dtype=int)

    def _col(name, default=np.nan):
        if name in pk:
            return np.asarray(pk[name][:])
        return np.full(pk_frame.size, default)
    cols = {
        "center": _col("center"), "center_err": _col("center_err"),
        "amplitude": _col("amplitude"), "amplitude_err": _col("amplitude_err"),
        "fwhm": _col("fwhm"), "fwhm_err": _col("fwhm_err"),
        "eta": _col("eta"), "area": _col("area"), "chi2": _col("chi2"),
        "flag": (np.asarray(pk["flag"][:], dtype=int)
                 if "flag" in pk else np.zeros(pk_frame.size, int)),
    }
    phase_col = None
    if "phase" in pk:
        phase_col = [
            (s.decode("utf-8", "replace")
             if isinstance(s, (bytes, bytearray)) else str(s))
            for s in pk["phase"][:]]
    n_rows = 0
    with Path(csv_path).open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["frame", "filename"] + list(cols.keys())
                   + (["phase"] if phase_col is not None else []))
        for j in range(pk_frame.size):
            fi = int(pk_frame[j])
            if fi not in want:
                continue
            row = [fi, filenames[fi] if fi < len(filenames) else ""]
            row += [cols[k][j] for k in cols]
            if phase_col is not None:
                row.append(phase_col[j])
            w.writerow(row)
            n_rows += 1
    return n_rows


def export_frames(analysis_h5: "str | Path", out_dir: "str | Path", *,
                  frames: "Optional[Sequence[int]]" = None,
                  source: str = "fit", peaks: bool = True) -> Dict[str, Any]:
    """Export selected frames' REDUCTION/FIT patterns and peak-fitting results.

    The light sibling of :func:`export_refinement_bundle` for everyday
    "give me these frames" use: per selected frame a two-column ``.xy``
    pattern of the chosen channel (``frame_####_q.xy`` on the native axis
    always; ``frame_####.xy`` in 2θ additionally when the wavelength is
    known), and — with ``peaks=True`` — one combined ``peaks.csv`` holding
    every fitted peak of those frames (frame, filename, center ± esd,
    amplitude ± esd, fwhm ± esd, eta, area, chi2, flag, attributed phase).
    Flagged peaks are included with their flag value so nothing is silently
    dropped; filter on the ``flag`` column for good-only.

    ``source``: "fit" (what Step 2 actually fitted, default) or any of
    "clean"/"mean"/"hybrid"/"sigmaclip"/"auto"/"robust" — the reduction-side
    channels reconstructed exactly as the pipeline does. ``frames=None``
    exports every non-excluded frame. Returns a manifest.
    """
    import h5py  # type: ignore

    src = Path(analysis_h5).expanduser()
    out = Path(out_dir).expanduser()
    patterns_dir = out / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"n_frames": 0, "files_written": [],
                                "n_peaks": 0, "unit": "", "wavelength": None,
                                "source": source}

    with h5py.File(str(src), "r") as h5:
        unit = str(h5.attrs.get("unit", ""))
        wl_raw = float(h5.attrs.get("wavelength", 0.0) or 0.0)
        wavelength = wl_raw if wl_raw > 0 else None
        manifest["unit"] = unit
        manifest["wavelength"] = wavelength

        data, source_label = _pattern_source(h5, source)
        n_total = int(data.shape[0])
        radial = (np.asarray(h5["radial"][:], dtype=float) if "radial" in h5
                  else np.arange(data.shape[1], dtype=float))
        fr = h5.get("frames")
        filenames: "List[str]" = []
        if fr is not None and "filename" in fr:
            filenames = [
                (s.decode("utf-8", "replace") if isinstance(s, (bytes, bytearray)) else str(s))
                for s in fr["filename"][:]]
        excluded = (np.asarray(fr["excluded"][:], dtype=bool)
                    if (fr is not None and "excluded" in fr
                        and fr["excluded"].shape[0] == n_total)
                    else np.zeros(n_total, dtype=bool))
        if frames is None:
            frame_idx = [i for i in range(n_total) if not excluded[i]]
        else:
            frame_idx = sorted({int(i) for i in frames if 0 <= int(i) < n_total})

        for i in frame_idx:
            fname = filenames[i] if i < len(filenames) else ""
            p_native = patterns_dir / f"frame_{i:04d}_q.xy"
            _write_xy(p_native, radial, data[i], header=_xy_header(
                axis=unit or "native", native_unit=unit, wavelength=wavelength,
                source_label=source_label, filename=fname))
            manifest["files_written"].append(str(p_native))
            tth = _to_two_theta_deg(radial, unit, wavelength)
            if tth is not None:
                p_tth = patterns_dir / f"frame_{i:04d}.xy"
                _write_xy(p_tth, tth, data[i], header=_xy_header(
                    axis="2th_deg", native_unit=unit, wavelength=wavelength,
                    source_label=source_label, filename=fname))
                manifest["files_written"].append(str(p_tth))

        if peaks:
            csv_path = out / "peaks.csv"
            n_rows = _write_peaks_csv(h5, csv_path, frame_idx, filenames)
            if n_rows is not None:
                manifest["n_peaks"] = n_rows
                manifest["files_written"].append(str(csv_path))

    manifest["n_frames"] = len(frame_idx)
    print(f"[EXPORT] {manifest['n_frames']} frame pattern(s) "
          f"({source_label}) + {manifest['n_peaks']} peak row(s) -> {out}",
          flush=True)
    return manifest


def export_refinement_bundle(
    analysis_h5: "str | Path", out_dir: "str | Path", *,
    frames: "Optional[Sequence[int]]" = None,
    phases: "Optional[Sequence[str]]" = None,
    workspace: "Optional[str | Path]" = None,
    source: str = "fit",
) -> Dict[str, Any]:
    """Export a Rietveld-refinement hand-off bundle to ``out_dir``.

    ``frames`` — indices to export (default: every non-``/frames/excluded``
    frame). ``phases`` — phase names to resolve into CIFs (default: every
    phase present under ``/identify``). ``workspace`` — where
    :func:`phases.load_library` looks for the user phase library (default:
    ``analysis_h5``'s parent directory). ``source`` — the pattern channel;
    ``"fit"`` (default) reconstructs the Step-2 fit source, ``"robust"`` is
    ``clean + baseline``, and ``"clean"``/``"mean"``/``"hybrid"``/
    ``"sigmaclip"``/``"auto"`` go through :func:`peaks.build_fit_source`.

    Returns a manifest dict: ``{n_frames, files_written, phases_written,
    phases_skipped, wavelength, unit}``. Never raises for a single bad phase
    (recorded in ``phases_skipped`` instead) — only file-level problems with
    the analysis HDF5 itself (missing background, unreadable file) raise.
    """
    import h5py  # type: ignore

    src = Path(analysis_h5).expanduser()
    out = Path(out_dir).expanduser()
    patterns_dir = out / "patterns"
    phases_dir = out / "phases"
    patterns_dir.mkdir(parents=True, exist_ok=True)
    phases_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "n_frames": 0, "files_written": [], "phases_written": [],
        "phases_skipped": [], "wavelength": None, "unit": "",
    }

    with h5py.File(str(src), "r") as h5:
        unit = str(h5.attrs.get("unit", ""))
        wl_raw = float(h5.attrs.get("wavelength", 0.0) or 0.0)
        wavelength = wl_raw if wl_raw > 0 else None
        manifest["unit"] = unit
        manifest["wavelength"] = wavelength

        data, source_label = _pattern_source(h5, source)
        n_total = int(data.shape[0])
        radial = (np.asarray(h5["radial"][:], dtype=float) if "radial" in h5
                  else np.arange(data.shape[1], dtype=float))

        fr = h5.get("frames")
        filenames: "Optional[List[str]]" = None
        if fr is not None and "filename" in fr:
            raw_names = fr["filename"][:]
            filenames = [
                (s.decode("utf-8", "replace") if isinstance(s, (bytes, bytearray)) else str(s))
                for s in raw_names
            ]
        excluded = (np.asarray(fr["excluded"][:], dtype=bool)
                    if (fr is not None and "excluded" in fr and fr["excluded"].shape[0] == n_total)
                    else np.zeros(n_total, dtype=bool))

        if frames is None:
            frame_idx = [i for i in range(n_total) if not excluded[i]]
        else:
            frame_idx = [int(i) for i in frames if 0 <= int(i) < n_total]

        for i in frame_idx:
            y = data[i]
            fname = filenames[i] if (filenames is not None and i < len(filenames)) else ""

            native_path = patterns_dir / f"frame_{i:04d}_q.xy"
            _write_xy(native_path, radial, y, header=_xy_header(
                axis=unit or "native", native_unit=unit, wavelength=wavelength,
                source_label=source_label, filename=fname))
            manifest["files_written"].append(str(native_path))

            tth = _to_two_theta_deg(radial, unit, wavelength)
            if tth is not None:
                tth_path = patterns_dir / f"frame_{i:04d}.xy"
                _write_xy(tth_path, tth, y, header=_xy_header(
                    axis="2th_deg", native_unit=unit, wavelength=wavelength,
                    source_label=source_label, filename=fname))
                manifest["files_written"].append(str(tth_path))

        manifest["n_frames"] = len(frame_idx)

        if phases is None:
            names: List[str] = []
            gid = h5.get("identify")
            if gid is not None:
                for k in gid.keys():
                    g = gid[k]
                    nm = str(g.attrs.get("name", k)) if hasattr(g, "attrs") else k
                    names.append(nm)
            phase_names = names
        else:
            phase_names = [str(p) for p in phases]

    ws = Path(workspace).expanduser() if workspace is not None else src.parent
    lib = load_library(ws)
    have_pymatgen = pymatgen_available()

    for name in phase_names:
        ph: "Optional[Phase]" = lib.get(name)
        if ph is None:
            manifest["phases_skipped"].append(
                {"name": name, "reason": "not found in the phase library "
                                          f"(workspace={ws})"})
            continue

        cif_dst = phases_dir / f"{_safe_filename(name)}.cif"
        if ph.cif_path and Path(ph.cif_path).is_file():
            try:
                shutil.copy2(ph.cif_path, cif_dst)
                manifest["phases_written"].append({"name": name, "path": str(cif_dst)})
            except Exception as e:
                manifest["phases_skipped"].append(
                    {"name": name, "reason": f"failed to copy cif_path: {e!r}"})
            continue

        has_structure_data = bool(ph.lattice) and bool(ph.atoms) and bool(ph.space_group)
        if not has_structure_data:
            manifest["phases_skipped"].append(
                {"name": name, "reason": "no cif_path, and no lattice+atoms+space_group "
                                          "to synthesize one"})
            continue
        if not have_pymatgen:
            manifest["phases_skipped"].append(
                {"name": name, "reason": "has lattice+atoms+space_group but pymatgen "
                                          "is not installed to synthesize a CIF"})
            continue
        try:
            from pymatgen.io.cif import CifWriter  # type: ignore
            struct = structure_from_phase(ph)
            CifWriter(struct).write_file(str(cif_dst))
            manifest["phases_written"].append({"name": name, "path": str(cif_dst)})
        except Exception as e:
            manifest["phases_skipped"].append(
                {"name": name, "reason": f"CIF generation failed: {e!r}"})

    if wavelength:
        instprm_path = out / "instrument.instprm"
        _write_instprm(instprm_path, wavelength)
        manifest["files_written"].append(str(instprm_path))

    readme_path = out / "README.md"
    _write_readme(readme_path, unit=unit, wavelength=wavelength,
                 n_phases_written=len(manifest["phases_written"]),
                 n_phases_skipped=len(manifest["phases_skipped"]))
    manifest["files_written"].append(str(readme_path))

    return manifest


def main(argv: "list[str] | None" = None) -> int:
    """CLI: ``bulkxrd-export-refinement analysis.h5 out_dir [options]``."""
    import argparse
    p = argparse.ArgumentParser(
        prog="bulkxrd-export-refinement",
        description="Export a Rietveld hand-off bundle (patterns as .xy, phase "
                    "CIFs, GSAS-II instrument parameters, README) from an "
                    "analysis HDF5.")
    p.add_argument("analysis", help="Path to an *_analysis.h5.")
    p.add_argument("out_dir", help="Bundle output directory (created).")
    p.add_argument("--frames", default="",
                   help="Comma-separated frame indices (default: all "
                        "non-excluded frames).")
    p.add_argument("--phases", default="",
                   help="Comma-separated phase names (default: every phase "
                        "under /identify).")
    p.add_argument("--workspace", default="",
                   help="Workspace holding the user phase library "
                        "(default: beside the analysis file).")
    p.add_argument("--source", default="fit",
                   choices=["fit", "clean", "mean", "hybrid", "sigmaclip",
                            "auto", "robust"],
                   help="Pattern channel to export. Default fit (what Step 2 "
                        "actually fitted).")
    p.add_argument("--peaks", action="store_true",
                   help="Also write peaks.csv: every fitted peak of the "
                        "exported frames (center/amplitude/fwhm ± esd, eta, "
                        "area, chi2, flag, attributed phase).")
    args = p.parse_args(argv)
    frames = ([int(s) for s in args.frames.split(",") if s.strip()]
              if args.frames.strip() else None)
    names = ([s.strip() for s in args.phases.split(",") if s.strip()]
             if args.phases.strip() else None)
    try:
        man = export_refinement_bundle(
            args.analysis, args.out_dir, frames=frames, phases=names,
            workspace=(args.workspace or None), source=args.source)
        if args.peaks:
            import h5py  # type: ignore
            with h5py.File(str(Path(args.analysis).expanduser()), "r") as h5:
                fr = h5.get("frames")
                fnames = ([s.decode("utf-8", "replace")
                           if isinstance(s, (bytes, bytearray)) else str(s)
                           for s in fr["filename"][:]]
                          if fr is not None and "filename" in fr else [])
                n_total = (h5["background/clean"].shape[0]
                           if "background" in h5 else len(fnames))
                idx = (frames if frames is not None
                       else list(range(int(n_total))))
                n_rows = _write_peaks_csv(
                    h5, Path(args.out_dir) / "peaks.csv", idx, fnames)
            if n_rows is None:
                print("[EXPORT] --peaks skipped: no /peaks (run Step 2 first).",
                      flush=True)
            else:
                print(f"[EXPORT] peaks.csv: {n_rows} row(s)", flush=True)
    except (OSError, ValueError, KeyError) as e:
        print(f"[ERROR] {e}", flush=True)
        return 1
    print(f"[EXPORT] {man['n_frames']} frame(s), "
          f"{len(man['phases_written'])} phase CIF(s) -> {args.out_dir}",
          flush=True)
    for rec in man.get("phases_skipped") or []:
        print(f"[EXPORT] skipped phase {rec.get('name')!r}: "
              f"{rec.get('reason')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
