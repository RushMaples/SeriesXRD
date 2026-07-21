"""Williamson–Hall size/strain analysis from the Step-2 peak fits.

Classic W–H in q-space: the fitted peak width decomposes into a size term
(constant in q) and a microstrain term (linear in q),

    Δq(q) = 2πK/D  +  2ε·q

with D the volume-weighted crystallite size (Å), K the Scherrer shape factor
(~0.9), and ε the microstrain (Δd/d; NOTE conventions differ — the common
``4ε·sinθ`` form corresponds to a slope of 2ε on a q axis with this
definition). The Step-2 esd's (``fwhm_err``) weight the fit, so a few sharp
strong peaks aren't outvoted by noisy weak ones.

Instrument broadening: pass ``instrument_fwhm_q`` (a scalar Δq or a callable
fwhm_q(q), e.g. :func:`mldata.resolution_curve` fitted on a LaB6/CeO2 standard)
and it is removed in quadrature before fitting. WITHOUT it the numbers are
upper/lower bounds only (size underestimated, strain overestimated) and the
output is flagged ``instrument_corrected=False`` — don't put uncorrected values
in a paper.

References: G. K. Williamson & W. H. Hall, Acta Metall. 1 (1953) 22
(the size/strain decomposition); P. Scherrer, Nachr. Ges. Wiss. Goettingen
(1918) 98 (the K/D size term).

``williamson_hall`` returns per-frame results and (optionally) appends
``/microstructure`` to the analysis HDF5. Pure numpy + h5py.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..core.config import VERSION
from ..core.provenance import manifest_provenance, write_step_provenance

SCHEMA_VERSION = "1"
TWO_PI = 2.0 * np.pi


def _peaks_q_by_frame(analysis_h5: "str | Path"):
    """Per-peak (frame, q, dq, dq_err) for good peaks, converted to Å⁻¹."""
    import h5py  # type: ignore
    with h5py.File(str(Path(analysis_h5).expanduser()), "r") as h5:
        pk = h5.get("peaks")
        if pk is None or "fwhm" not in pk:
            raise ValueError("Analysis file lacks /peaks — run Step 2 first.")
        unit = str(h5.attrs.get("unit", "")).strip().lower()
        wl = float(h5.attrs.get("wavelength", 0.0) or 0.0)
        frame = np.asarray(pk["frame"][:], int)
        center = np.asarray(pk["center"][:], float)
        fwhm = np.asarray(pk["fwhm"][:], float)
        ferr = (np.asarray(pk["fwhm_err"][:], float) if "fwhm_err" in pk
                else np.full(fwhm.size, np.nan))
        flag = (np.asarray(pk["flag"][:], int) if "flag" in pk
                else np.zeros(fwhm.size, int))
        n = int(np.asarray(pk["counts"][:]).size) if "counts" in pk \
            else int(frame.max()) + 1 if frame.size else 0
    good = (flag == 0) & np.isfinite(fwhm) & (fwhm > 0)
    frame, center, fwhm, ferr = frame[good], center[good], fwhm[good], ferr[good]
    if unit in ("q_a^-1", "q_a-1", "q_a", "q"):
        q, dq, dqe = center, fwhm, ferr
    elif unit in ("q_nm^-1", "q_nm-1", "q_nm"):
        q, dq, dqe = center * 0.1, fwhm * 0.1, ferr * 0.1
    elif unit in ("2th_deg", "2th_rad") and wl > 0:
        tt = np.radians(center) if unit == "2th_deg" else center
        dtt = np.radians(fwhm) if unit == "2th_deg" else fwhm
        dte = np.radians(ferr) if unit == "2th_deg" else ferr
        q = (4.0 * np.pi / wl) * np.sin(tt / 2.0)
        conv = (2.0 * np.pi / wl) * np.cos(tt / 2.0)
        dq, dqe = conv * dtt, conv * dte
    else:
        raise ValueError(f"Cannot convert unit {unit!r} to q (wavelength known?).")
    return n, frame, q, dq, dqe


def williamson_hall(
    analysis_h5: "str | Path",
    *,
    k_shape: float = 0.9,
    instrument_fwhm_q=None,
    min_peaks: int = 5,
    write: bool = True,
    out_h5: "Optional[str | Path]" = None,
) -> Dict[str, Any]:
    """Per-frame W–H fit of the good Step-2 peaks.

    Returns a manifest with per-frame arrays ``size_A`` (Å; inf = no size
    broadening resolved, NaN = too few peaks), ``strain`` (Δd/d), their 1σ
    errors, ``r2`` and ``n_peaks``; optionally appends ``/microstructure`` to
    the file. Frames with fewer than ``min_peaks`` good peaks are NaN — a W–H
    line through 3 points is numerology.
    """
    src = Path(analysis_h5).expanduser().resolve()
    n, frame, q, dq, dqe = _peaks_q_by_frame(src)

    corrected = instrument_fwhm_q is not None
    if corrected:
        inst = (np.asarray(instrument_fwhm_q(q), float) if callable(instrument_fwhm_q)
                else np.full(q.size, float(instrument_fwhm_q)))
        with np.errstate(invalid="ignore"):
            dq = np.sqrt(np.clip(dq ** 2 - inst ** 2, 0.0, None))

    size = np.full(n, np.nan)
    size_err = np.full(n, np.nan)
    strain = np.full(n, np.nan)
    strain_err = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    n_used = np.zeros(n, "i4")
    for i in range(n):
        m = (frame == i) & np.isfinite(dq) & (dq > 0)
        if int(m.sum()) < int(min_peaks):
            continue
        x, y = q[m], dq[m]
        e = dqe[m]
        w = np.where(np.isfinite(e) & (e > 0), 1.0 / e, 1.0)
        w = w / w.max()
        M = np.column_stack([np.ones_like(x), x])
        coef = np.linalg.lstsq(M * w[:, None], y * w, rcond=None)[0]
        b0, b1 = float(coef[0]), float(coef[1])
        pred = M @ coef
        resid = (y - pred) * w
        dof = max(int(m.sum()) - 2, 1)
        s2 = float(np.sum(resid ** 2)) / dof
        try:
            cov = np.linalg.pinv((M * w[:, None]).T @ (M * w[:, None])) * s2
            e0, e1 = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
        except Exception:
            e0 = e1 = float("nan")
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2[i] = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else np.nan
        n_used[i] = int(m.sum())
        strain[i] = max(b1, 0.0) / 2.0
        strain_err[i] = e1 / 2.0
        if b0 > 0:
            size[i] = TWO_PI * float(k_shape) / b0
            size_err[i] = size[i] * e0 / b0
        else:
            size[i] = np.inf                      # no resolvable size broadening

    manifest: Dict[str, Any] = {
        **manifest_provenance("seriesxrd.analysis.microstructure", SCHEMA_VERSION),
        "k_shape": float(k_shape),
        "instrument_corrected": bool(corrected),
        "convention": "dq = 2*pi*K/D + 2*eps*q (eps = dd/d)",
        "n_frames": int(n),
        "size_A": size.tolist(), "size_err_A": size_err.tolist(),
        "strain": strain.tolist(), "strain_err": strain_err.tolist(),
        "r2": r2.tolist(), "n_peaks": n_used.tolist(),
    }
    if not corrected:
        manifest["warning"] = ("No instrument profile supplied — size is a lower "
                               "bound and strain an upper bound; measure a LaB6/"
                               "CeO2 standard and pass instrument_fwhm_q.")

    if write:
        import h5py  # type: ignore
        dst = Path(out_h5).expanduser().resolve() if out_h5 else src
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        try:
            with h5py.File(str(tmp), "r+") as o:
                if "microstructure" in o:
                    del o["microstructure"]
                g = o.create_group("microstructure")
                write_step_provenance(o, "microstructure",
                                      tool="seriesxrd.analysis.microstructure",
                                      schema_version=SCHEMA_VERSION)
                g.attrs.update({"schema_version": SCHEMA_VERSION,
                                "seriesxrd_version": VERSION,
                                "k_shape": float(k_shape),
                                "instrument_corrected": bool(corrected),
                                "convention": manifest["convention"]})
                for k, v in (("size_A", size), ("size_err_A", size_err),
                             ("strain", strain), ("strain_err", strain_err),
                             ("r2", r2), ("n_peaks", n_used)):
                    g.create_dataset(k, data=v)
            os.replace(tmp, dst)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        manifest["out_h5"] = str(dst)
    live = np.isfinite(strain)
    if live.any():
        print(f"[WH] {int(live.sum())}/{n} frames fit; median size="
              f"{np.nanmedian(size[live & np.isfinite(size)]):.0f} A, "
              f"median strain={np.nanmedian(strain[live]):.2e}"
              f"{' (UNCORRECTED for instrument)' if not corrected else ''}",
              flush=True)
    return manifest
