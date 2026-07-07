"""Batch reduction backend: apply an accepted calibration to a dataset.

Reads the calibration handoff (PONI + mask), walks a dataset folder, and
integrates every frame with pyFAI into 1D patterns (and optionally 2D
cakes), written to a single HDF5 file plus a JSON manifest.

Performance model: one ``AzimuthalIntegrator`` per worker process, created
once in the pool initializer — pyFAI caches its sparse matrix on the
integrator, so after the first frame each subsequent frame costs only the
re-binning. Heavy imports stay lazy, mirroring calib/processing.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import sys
import time

import numpy as np

from ..core.config import (
    VERSION, ensure_dir, write_json, sha256_file, now_iso, now_timestamp, safe_stem,
    output_base,
)
from ..core.handoff import load_handoff
from ..core.io import (read_detector_image, expand_frame_sources,
                       frame_display_name, harvest_stack_metadata,
                       is_h5_frame_spec)
from ..core.masks import load_mask_npz

DEFAULT_PATTERNS = "*.tif;*.tiff;*.edf;*.cbf;*.mar3450;*.h5"

# 2D cake method: CSR without pixel splitting — small sparse matrix, fast,
# low RAM (the bare "csr" string would default to bbox splitting, whose 2D
# matrix can exceed 18 GB on large detectors).
_CAKE_METHOD_2D = ("no", "csr", "cython")


def _ensure_conda_dlls() -> None:
    """Prepend conda Library/bin to PATH on Windows so pyFAI C extensions resolve."""
    if not sys.platform.startswith("win"):
        return
    prefix = Path(sys.executable).parent
    for subdir in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin"):
        dll_dir = prefix / subdir
        if dll_dir.is_dir():
            dll_str = str(dll_dir)
            if dll_str.lower() not in os.environ.get("PATH", "").lower():
                os.environ["PATH"] = dll_str + os.pathsep + os.environ.get("PATH", "")


def scan_dataset(dataset_dir: "str | Path", patterns: str = DEFAULT_PATTERNS,
                 recursive: bool = False) -> List[Path]:
    """List frame files matching semicolon-separated glob patterns, sorted by name."""
    root = Path(dataset_dir).expanduser()
    if not root.is_dir():
        return []
    found: "set[Path]" = set()
    for pat in (p.strip() for p in str(patterns or DEFAULT_PATTERNS).split(";")):
        if not pat:
            continue
        found.update(root.rglob(pat) if recursive else root.glob(pat))
    return sorted(p for p in found if p.is_file())


# ---------------------------------------------------------------------------
# Per-process integration state (multiprocessing pool initializer + task)
# ---------------------------------------------------------------------------
_W: Dict[str, Any] = {}  # populated per worker process by _pool_init


def _pool_init(poni_file: str, mask_file: str, settings: Dict[str, Any]) -> None:
    _ensure_conda_dlls()
    import pyFAI  # type: ignore
    ai = pyFAI.load(str(poni_file))
    mask = load_mask_npz(mask_file) if mask_file else None
    _W.update(ai=ai, mask=mask, settings=dict(settings))


def _render_thumbnail(out_png: str, cake, cake_radial, radial, intensity, unit: str) -> None:
    """Render a compact cake-over-1D preview PNG for one frame.

    Uses the Figure/FigureCanvasAgg API directly (no pyplot) so it is safe in
    worker processes and never touches a global matplotlib backend.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(2.4, 2.0), dpi=100)
    fig.patch.set_facecolor("#1e1e2e")
    ax_c = fig.add_axes([0.0, 0.42, 1.0, 0.58])   # cake fills the top
    ax_p = fig.add_axes([0.08, 0.04, 0.90, 0.34])  # 1D under it
    if cake is not None:
        c = np.array(cake, dtype=float)
        c[c <= 0] = np.nan
        extent = None
        if cake_radial is not None:
            r = np.asarray(cake_radial)
            extent = [float(r.min()), float(r.max()), -180, 180]
        vmin = np.nanpercentile(c, 5) if np.any(np.isfinite(c)) else None
        vmax = np.nanpercentile(c, 99) if np.any(np.isfinite(c)) else None
        
        ax_c.imshow(c, aspect="auto", origin="lower", cmap="magma", extent=extent, vmin=vmin, vmax=vmax)
    ax_c.set_xticks([]); ax_c.set_yticks([])
    if radial is not None and intensity is not None:
        ax_p.plot(np.asarray(radial), np.asarray(intensity, dtype=float), lw=0.7, color="#89b4fa")
    ax_p.set_facecolor("#11111b")
    ax_p.tick_params(colors="#6c7086", labelsize=5, length=2)
    for s in ax_p.spines.values():
        s.set_edgecolor("#45475a")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig).print_png(out_png)


def _named_params(fn) -> "set":
    """Explicitly named parameters of a callable. A bare ``**kwargs`` does NOT
    count: several pyFAI releases *log and ignore* unknown keyword arguments
    ("Got unknown argument quant_min ...") instead of raising TypeError, so the
    only way to know a quantile kwarg is honoured is to see it in the signature.
    """
    import inspect
    try:
        return {p.name for p in inspect.signature(fn).parameters.values()
                if p.kind != inspect.Parameter.VAR_KEYWORD}
    except (TypeError, ValueError):
        return set()


def _robust_integrate(ai, image, npt: int, mask, unit: str,
                      quant_halfwidth: float,
                      azimuth_range=None) -> "Tuple[Any, str]":
    """The spot-suppressed 'robust' 1D pattern; returns ``(result, estimator)``.

    A pure azimuthal MEDIAN of integer photon counts can only take integer /
    half-integer values, and because the median over hundreds of azimuthal
    pixels has almost no statistical noise, low-count patterns render as clean
    STAIRCASES (and ``clean = robust − baseline`` inherits the steps). Averaging
    a narrow quantile band around the median instead (default 45–55%) keeps the
    same outlier rejection — diamond spots occupy far less than 45% of the
    azimuth — but yields continuous intensities.

    Fallback chain (pyFAI version differences): ``medfilt1d_ng`` with
    ``quant_min/quant_max`` or a ``quantile=(lo, hi)`` tuple — dispatched by
    *signature inspection*, because pyFAI ignores unknown kwargs instead of
    raising → legacy ``medfilt1d`` with a percentile tuple → plain median
    (estimator ``median(band_unsupported)`` so the caller can warn).
    ``quant_halfwidth<=0`` requests the pure median explicitly.
    """
    h = float(quant_halfwidth or 0.0)
    lo = round(max(0.0, 0.5 - h), 6)
    hi = round(min(1.0, 0.5 + h), 6)
    extra = {"azimuth_range": azimuth_range} if azimuth_range else {}
    ng = getattr(ai, "medfilt1d_ng", None)
    if h > 0:
        if ng is not None:
            params = _named_params(ng)
            band = f"quantile_band({lo:.2f}-{hi:.2f})"
            if {"quant_min", "quant_max"} <= params:
                try:
                    return (ng(image, npt, mask=mask, unit=unit,
                               quant_min=lo, quant_max=hi, **extra), band)
                except TypeError:
                    pass
            elif "quantile" in params:
                try:
                    return (ng(image, npt, mask=mask, unit=unit,
                               quantile=(lo, hi), **extra), band)
                except TypeError:
                    pass
        mf = getattr(ai, "medfilt1d", None)
        if mf is not None:
            try:
                pct = (round(100.0 * lo, 4), round(100.0 * hi, 4))
                return (mf(image, npt, mask=mask, unit=unit, percentile=pct,
                           **extra),
                        f"percentile_band({pct[0]:.0f}-{pct[1]:.0f})")
            except TypeError:
                pass
    medfilt = ng or ai.medfilt1d
    est = "median" if h <= 0 else "median(band_unsupported)"
    return medfilt(image, npt, mask=mask, unit=unit, **extra), est


def _integrate_one(task: "Tuple[int, str, bool]") -> Dict[str, Any]:
    """Integrate a single frame. Runs inside a pool worker (or in-process)."""
    index, file_str, want_cake = task
    s = _W["settings"]
    ai, mask = _W["ai"], _W["mask"]
    t0 = time.time()
    try:
        image = read_detector_image(file_str)
        if mask is not None and mask.shape != image.shape:
            raise ValueError(f"mask shape {mask.shape} != image shape {image.shape}")
        az_range = s.get("azimuth_range")     # optional (min°, max°) sector
        res = ai.integrate1d(
            image, int(s["npt_1d"]), mask=mask, unit=s["unit"],
            method=s["method"], polarization_factor=s.get("polarization_factor"),
            azimuth_range=az_range,
        )
        out: Dict[str, Any] = {
            "index": index, "file": file_str, "ok": True, "error": "",
            "radial": np.asarray(res.radial), "intensity": np.asarray(res.intensity),
        }
        if s.get("robust_1d"):
            # Spot-suppressed channel: narrow quantile-band mean around the
            # azimuthal median (see _robust_integrate — a pure median staircases
            # on integer counts). Older pyFAI versions degrade gracefully.
            try:
                rres, est = _robust_integrate(
                    ai, image, int(s["npt_1d"]), mask, s["unit"],
                    float(s.get("robust_quant_halfwidth", 0.05) or 0.0),
                    azimuth_range=az_range)
                out["intensity_robust"] = np.asarray(rres.intensity)
                out["robust_estimator"] = est
            except Exception as e:
                out["robust_error"] = repr(e)
        if s.get("sigmaclip_1d"):
            # Azimuthal sigma-clipping = robust trimmed mean: iteratively reject
            # azimuthal bins that deviate from the per-radial mean (diamond
            # single-crystal spots) while KEEPING azimuthally-sparse real sample
            # intensity that the median would drop on a textured/incomplete ring.
            # error_model="azimuthal" derives each bin's variance from the
            # azimuthal spread itself. Degrades gracefully like robust above.
            try:
                sclip = getattr(ai, "sigma_clip_ng", None) or getattr(ai, "sigma_clip", None)
                if sclip is None:
                    raise AttributeError("AzimuthalIntegrator has no sigma_clip_ng/sigma_clip")
                thr = float(s.get("sigmaclip_thresh", 3.0) or 3.0)
                mit = int(s.get("sigmaclip_maxiter", 5) or 5)
                az_kw = {"azimuth_range": az_range} if az_range else {}
                try:
                    sres = sclip(image, int(s["npt_1d"]), mask=mask, unit=s["unit"],
                                 error_model="azimuthal", thres=thr, max_iter=mit,
                                 polarization_factor=s.get("polarization_factor"),
                                 **az_kw)
                except TypeError:
                    # Older signature: no error_model/polarization kwargs.
                    sres = sclip(image, int(s["npt_1d"]), mask=mask, unit=s["unit"],
                                 thres=thr, max_iter=mit)
                out["intensity_sigmaclip"] = np.asarray(sres.intensity)
            except Exception as e:
                out["sigmaclip_error"] = repr(e)
        if want_cake:
            cres = ai.integrate2d(
                image, int(s["npt_radial"]), int(s["npt_azimuthal"]),
                mask=mask, unit=s["unit"], method=_CAKE_METHOD_2D,
                correctSolidAngle=True,
            )
            out["cake"] = np.asarray(cres.intensity)
            out["cake_radial"] = np.asarray(cres.radial)
            out["cake_azimuthal"] = np.asarray(cres.azimuthal)
        # Per-frame gallery thumbnail: a low-res cake (computed here so every
        # frame gets one even when full cakes are sampled every Nth frame) plus
        # the 1D sparkline. Rendered in-worker so the gallery loads instantly.
        previews_dir = s.get("previews_dir")
        if previews_dir:
            try:
                tr, ta = int(s.get("thumb_radial", 180)), int(s.get("thumb_azimuthal", 90))
                if "cake" in out:
                    thumb_cake, thumb_r = out["cake"], out["cake_radial"]
                else:
                    tres = ai.integrate2d(image, tr, ta, mask=mask, unit=s["unit"],
                                          method=_CAKE_METHOD_2D, correctSolidAngle=True)
                    thumb_cake, thumb_r = np.asarray(tres.intensity), np.asarray(tres.radial)
                png = str(Path(previews_dir) / f"frame_{index:05d}.png")
                _render_thumbnail(png, thumb_cake, thumb_r, out["radial"], out["intensity"], s["unit"])
                out["thumb"] = f"frame_{index:05d}.png"
            except Exception as e:
                out["thumb_error"] = repr(e)
        out["seconds"] = time.time() - t0
        return out
    except Exception as e:
        return {"index": index, "file": file_str, "ok": False, "error": repr(e),
                "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# Top-level batch run (called by reduce/worker.py)
# ---------------------------------------------------------------------------

def _resolve_npt_1d(raw_npt, poni_file, first_image_file
                    ) -> "Tuple[int, Optional[int], str]":
    """Resolve the 1D bin count; returns ``(npt, suggested, mode)``.

    Blank / ``auto`` / ``0`` → the geometry-derived suggestion (pyFAI's rule of
    thumb: ~1 bin per pixel of maximum radial extent, computed exactly like the
    calibration stage's auto-fill). An explicit value is honoured, but a value
    well below the suggestion is warned about: under-sampling makes sharp peaks
    span only a couple of bins — patterns render as staircases and the
    pseudo-Voigt fits degrade (quantised centers, width-bound rejections).
    ``suggested`` is None when the geometry could not be read; ``mode`` is
    ``auto`` | ``explicit`` | ``fallback``.
    """
    raw = str(raw_npt if raw_npt is not None else "").strip().lower()
    explicit = raw not in ("", "auto", "0")
    suggested = None
    try:
        from ..calib.processing import suggest_integration_settings
        import pyFAI  # type: ignore
        ai = pyFAI.load(str(poni_file))
        shape = read_detector_image(str(first_image_file)).shape
        poni_info = {"pixel1": getattr(ai, "pixel1", None),
                     "pixel2": getattr(ai, "pixel2", None),
                     "poni1": getattr(ai, "poni1", None),
                     "poni2": getattr(ai, "poni2", None)}
        suggested = int(suggest_integration_settings(shape, poni_info)["npt_1d"])
    except Exception as e:
        print(f"[REDUCE] could not derive npt_1d from geometry ({e!r})", flush=True)

    if explicit:
        npt = int(float(raw))
        # Always show the comparison (a stale explicit value in an old workspace
        # config is indistinguishable from a deliberate choice otherwise).
        print(f"[REDUCE] npt_1d={npt} (explicit; geometry suggests "
              f"~{suggested if suggested else '?'})", flush=True)
        if suggested and npt < 0.7 * suggested:
            print(f"[REDUCE] WARNING: npt_1d={npt} is well below the geometric "
                  f"suggestion ~{suggested} (~1 bin/pixel of radial extent). "
                  f"Under-sampled peaks look stepped and fit poorly — leave "
                  f"'1D bins' blank to use the suggestion.", flush=True)
        return npt, suggested, "explicit"
    if suggested:
        print(f"[REDUCE] npt_1d auto -> {suggested} (from detector geometry). "
              f"NOTE: this is ~1 bin/pixel — if Step 2 later reports "
              f"undersampled peaks (median FWHM < 4 bins), re-reduce with its "
              f"recommended npt_1d.", flush=True)
        return suggested, suggested, "auto"
    print("[REDUCE] npt_1d fallback -> 1500 (geometry unreadable)", flush=True)
    return 1500, None, "fallback"


def _parse_azimuth_range(v):
    """'min,max' / (min, max) in degrees → tuple, else None (full azimuth)."""
    if v in (None, "", (), []):
        return None
    try:
        if isinstance(v, str):
            a, b = (float(x) for x in v.replace(";", ",").split(",")[:2])
        else:
            a, b = float(v[0]), float(v[1])
        return (a, b) if b > a else None
    except (TypeError, ValueError, IndexError):
        print(f"[REDUCE] WARNING: unparseable azimuth_range {v!r} — using "
              f"the full azimuth.", flush=True)
        return None


def build_settings(config: Dict[str, Any], npt_1d: int) -> Dict[str, Any]:
    """Integration settings shared by the batch reducer and the live watcher.

    One place resolves the config's integration knobs so the two entry points
    cannot drift apart.
    """
    return {
        "npt_1d": int(npt_1d),
        # Optional azimuthal sector (deg) applied to ALL 1D channels alike —
        # mean/robust/sigmaclip must see the same pixels or spot_residual =
        # mean − robust becomes meaningless. Cakes stay full-azimuth (they are
        # the waviness/texture diagnostic). Useful stopgap for wavy rings when
        # re-calibration isn't possible; needs a pyFAI whose robust integrators
        # accept azimuth_range.
        "azimuth_range": _parse_azimuth_range(config.get("azimuth_range", "")),
        # Default q: the analysis stage is designed around q (peak widths
        # ~constant in q; d-conversion is wavelength-free). 2th_deg is honoured
        # when a session explicitly selects it.
        "unit": config.get("unit", "q_A^-1") or "q_A^-1",
        "method": config.get("method", "csr") or "csr",
        "polarization_factor": float(config["polarization_factor"]) if str(config.get("polarization_factor", "")).strip() else None,
        "robust_1d": bool(config.get("robust_1d", True)),
        # Half-width of the azimuthal quantile band averaged for the robust
        # channel (0.05 -> 45-55%). 0 = pure median (staircases on low counts).
        "robust_quant_halfwidth": float(config.get("robust_quant_halfwidth", 0.05)
                                        if str(config.get("robust_quant_halfwidth", "")).strip() != ""
                                        else 0.05),
        "sigmaclip_1d": bool(config.get("sigmaclip_1d", True)),
        "sigmaclip_thresh": float(config.get("sigmaclip_thresh", 3.0) or 3.0),
        "sigmaclip_maxiter": int(config.get("sigmaclip_maxiter", 5) or 5),
        "npt_radial": int(config.get("npt_radial", 500) or 500),
        "npt_azimuthal": int(config.get("npt_azimuthal", 360) or 360),
    }


def reduce_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a full batch reduction. Returns the manifest (also written to disk).

    Progress lines ``[PROGRESS] <done> <total>`` go to stdout so a supervising
    GUI/notebook can render a progress bar.
    """
    handoff = load_handoff(config.get("handoff_file", ""))
    if not handoff.ok:
        raise ValueError("Invalid handoff: " + "; ".join(handoff.problems))

    files = scan_dataset(
        config.get("dataset_dir", ""),
        config.get("file_patterns", DEFAULT_PATTERNS),
        bool(config.get("recursive", False)),
    )
    if not files:
        raise FileNotFoundError(
            f"No frames found in {config.get('dataset_dir')!r} "
            f"matching {config.get('file_patterns', DEFAULT_PATTERNS)!r}"
        )
    # HDF5/NeXus stack containers (Eiger-style master files) expand into one
    # source per stored frame; plain image files pass through unchanged.
    files, n_stacks = expand_frame_sources(
        files, str(config.get("h5_data_path", "") or ""))
    if n_stacks:
        print(f"[REDUCE] expanded {n_stacks} HDF5 stack container(s) -> "
              f"{len(files)} frames total", flush=True)
    if not files:
        raise FileNotFoundError(
            "The matched files contained no readable frames (HDF5 containers "
            "were skipped — see the log; pass h5_data_path if the stack "
            "layout was not recognized).")

    npt_1d, npt_suggested, npt_mode = _resolve_npt_1d(
        config.get("npt_1d", ""), handoff.accepted_poni, files[0])
    settings = build_settings(config, npt_1d)
    save_cakes = bool(config.get("save_cakes", False))
    cake_every = max(1, int(config.get("cake_every", 1) or 1))
    make_thumbnails = bool(config.get("make_thumbnails", True))

    # Resolve worker count; track whether it was auto-selected.
    _cfg_workers = int(config.get("num_workers", 0) or 0)
    auto = not bool(_cfg_workers)
    num_workers = _cfg_workers or max((os.cpu_count() or 2) - 1, 1)
    num_workers = min(num_workers, len(files))

    session_name = safe_stem(config.get("session_name", "reduction"), default="reduction")
    out_root = ensure_dir(Path(config.get("processed_root") or output_base(config) / "data" / "processed")
                          / f"reduction_{session_name}")
    ts = now_timestamp()
    h5_path = out_root / f"reduced_{session_name}_{ts}.h5"
    manifest_path = out_root / f"reduced_{session_name}_{ts}.manifest.json"
    previews_dir = out_root / f"reduced_{session_name}_{ts}_previews"
    if make_thumbnails:
        ensure_dir(previews_dir)
        settings["previews_dir"] = str(previews_dir)
        settings["thumb_radial"] = int(config.get("thumb_radial", 180) or 180)
        settings["thumb_azimuthal"] = int(config.get("thumb_azimuthal", 90) or 90)

    try:
        import h5py  # type: ignore
    except ImportError as e:
        raise ImportError(
            "h5py is required for batch reduction output. Install with: "
            "conda install -c conda-forge h5py   (or: pip install h5py)"
        ) from e

    mask_file = str(handoff.accepted_mask_npz or "")
    tasks = [(i, str(f), save_cakes and (i % cake_every == 0)) for i, f in enumerate(files)]
    total = len(tasks)
    n_cakes = sum(1 for t in tasks if t[2])
    auto_tag = " (auto)" if auto else ""
    print(f"[REDUCE] {total} frames, {num_workers} workers{auto_tag}, cakes={'every %d' % cake_every if save_cakes else 'off'}", flush=True)

    failures: List[Dict[str, str]] = []
    robust_errors: List[str] = []
    robust_estimators: List[str] = []
    sigmaclip_errors: List[str] = []
    t_start = time.time()

    dataset_root = Path(config.get("dataset_dir", "")).expanduser().resolve()

    # A5: write to a temp path; rename to final path only on success.
    h5_tmp = out_root / (h5_path.name + ".tmp")

    try:
        with h5py.File(h5_tmp, "w") as h5:
            # SCHEMA: root-level version attr for forward-compatibility.
            h5.attrs["schema_version"] = "1"
            h5.attrs.update({
                "tool": "bulkxrd.reduce", "tool_version": VERSION, "created_at": now_iso(),
                "poni_file": str(handoff.accepted_poni), "poni_sha256": sha256_file(handoff.accepted_poni) or "",
                "mask_file": mask_file, "mask_sha256": sha256_file(mask_file) or "" if mask_file else "",
                "handoff_file": str(handoff.path), "unit": settings["unit"], "method": settings["method"],
                "dataset_dir": str(dataset_root),
                "npt_1d": int(settings["npt_1d"]),
                "npt_1d_suggested": int(npt_suggested or 0),
                "npt_1d_mode": npt_mode,
                "robust_quant_halfwidth": float(settings.get("robust_quant_halfwidth", 0.0)),
            })
            h5.attrs["poni_text"] = Path(handoff.accepted_poni).read_text(encoding="utf-8", errors="replace")
            g_pat = h5.create_group("patterns")
            ds_int = g_pat.create_dataset("intensity", shape=(total, settings["npt_1d"]), dtype="f4", fillvalue=np.nan)
            ds_rob = g_pat.create_dataset("intensity_robust", shape=(total, settings["npt_1d"]), dtype="f4", fillvalue=np.nan) if settings["robust_1d"] else None
            ds_sc = g_pat.create_dataset("intensity_sigmaclip", shape=(total, settings["npt_1d"]), dtype="f4", fillvalue=np.nan) if settings["sigmaclip_1d"] else None
            ds_rad = g_pat.create_dataset("radial", shape=(settings["npt_1d"],), dtype="f8")
            g_frames = h5.create_group("frames")
            g_frames.attrs["dataset_dir"] = str(dataset_root)
            str_dt = h5py.string_dtype(encoding="utf-8")
            ds_files = g_frames.create_dataset("filename", shape=(total,), dtype=str_dt)
            ds_ok = g_frames.create_dataset("ok", shape=(total,), dtype="?")
            ds_sec = g_frames.create_dataset("seconds", shape=(total,), dtype="f4")
            g_frames.create_dataset("excluded", shape=(total,), dtype="?", fillvalue=False)
            # SCHEMA: canonical 0-based frame ordering (filename sort is unreliable).
            g_frames.create_dataset("frame_index", data=np.arange(total, dtype="i8"))
            
            # THUMBNAILS: Integrate thumbnail tracking from the gallery branch
            ds_thumb = g_frames.create_dataset("thumb", shape=(total,), dtype=str_dt) if make_thumbnails else None
            if make_thumbnails:
                g_frames.attrs["previews_dir"] = str(previews_dir)

            # Reserved placeholders for future pressure/temperature/time-series metadata.
            g_frames.create_dataset("pressure",    shape=(total,), dtype="f8", fillvalue=np.nan)
            g_frames.create_dataset("temperature", shape=(total,), dtype="f8", fillvalue=np.nan)
            g_frames.create_dataset("timestamp",   shape=(total,), dtype=str_dt)
            for i, f in enumerate(files):
                ds_files[i] = frame_display_name(f, dataset_root)

            # NeXus per-frame metadata: stack containers often carry
            # timestamps, stage positions, and temperature alongside the
            # images — harvest them into /frames so nothing needs a sidecar
            # CSV. Plain image files are unaffected (their headers are read
            # on demand by frame_metadata.import_positions_from_headers).
            if any(is_h5_frame_spec(f) for f in files):
                meta = harvest_stack_metadata(
                    files,
                    timestamp_path=str(config.get("h5_timestamp_path", "") or ""),
                    pos_x_path=str(config.get("h5_pos_x_path", "") or ""),
                    pos_y_path=str(config.get("h5_pos_y_path", "") or ""),
                    temperature_path=str(config.get("h5_temperature_path", "") or ""))
                if any(meta["timestamp"]):
                    g_frames["timestamp"][...] = np.array(meta["timestamp"],
                                                          dtype=object)
                if np.any(np.isfinite(meta["temperature"])):
                    g_frames["temperature"][...] = meta["temperature"]
                for key in ("pos_x", "pos_y"):
                    if np.any(np.isfinite(meta[key])):
                        g_frames.create_dataset(key, data=meta[key])
                if meta["n_frames_with_meta"]:
                    manifest_nexus = {
                        "n_frames_with_meta": int(meta["n_frames_with_meta"]),
                        "timestamps": int(sum(bool(t) for t in meta["timestamp"])),
                        "positions": int(np.sum(np.isfinite(meta["pos_x"])
                                                | np.isfinite(meta["pos_y"]))),
                        "temperature": int(np.sum(np.isfinite(meta["temperature"]))),
                    }
                    print(f"[REDUCE] NeXus metadata harvested: "
                          f"{manifest_nexus['timestamps']} timestamp(s), "
                          f"{manifest_nexus['positions']} position(s), "
                          f"{manifest_nexus['temperature']} temperature(s)",
                          flush=True)
                else:
                    manifest_nexus = None
            else:
                manifest_nexus = None
            
            g_cake = h5.create_group("cakes") if save_cakes else None
            # ds_cake / ds_cake_idx created lazily once the cake shape is known;
            # maxshape=(None,...) allows resizing down after the loop.
            ds_cake = ds_cake_idx = None

            radial_written = False
            done = 0
            cake_slot = 0  # A1: O(1) counter instead of O(N) np.sum scan

            def _store(r: Dict[str, Any]) -> None:
                nonlocal radial_written, ds_cake, ds_cake_idx, done, cake_slot
                i = r["index"]
                ds_ok[i] = r["ok"]
                ds_sec[i] = r.get("seconds", 0.0)
                if not r["ok"]:
                    failures.append({"file": r["file"], "error": r["error"]})
                    print(f"[REDUCE] FAILED {Path(r['file']).name}: {r['error']}", flush=True)
                else:
                    ds_int[i] = r["intensity"]
                    if not radial_written:
                        ds_rad[:] = r["radial"]
                        radial_written = True
                    if ds_rob is not None and "intensity_robust" in r:
                        ds_rob[i] = r["intensity_robust"]
                    if ds_sc is not None and "intensity_sigmaclip" in r:
                        ds_sc[i] = r["intensity_sigmaclip"]

                    # THUMBNAILS: Save thumb path
                    if ds_thumb is not None and "thumb" in r:
                        ds_thumb[i] = r["thumb"]

                    # B3: collect robust_error messages (deduplicated).
                    if "robust_error" in r:
                        msg = r["robust_error"]
                        if msg not in robust_errors:
                            robust_errors.append(msg)
                    if "robust_estimator" in r and r["robust_estimator"] not in robust_estimators:
                        robust_estimators.append(r["robust_estimator"])
                    if "sigmaclip_error" in r:
                        msg = r["sigmaclip_error"]
                        if msg not in sigmaclip_errors:
                            sigmaclip_errors.append(msg)
                    if g_cake is not None and "cake" in r:
                        if ds_cake is None:
                            cshape = r["cake"].shape
                            # maxshape=(None,...) so we can resize down if frames fail.
                            ds_cake = g_cake.create_dataset(
                                "intensity", shape=(n_cakes, *cshape), dtype="f4",
                                chunks=(1, *cshape), compression="gzip", compression_opts=1,
                                maxshape=(None, *cshape),
                            )
                            ds_cake_idx = g_cake.create_dataset(
                                "frame_index", shape=(n_cakes,), dtype="i8", fillvalue=-1,
                                maxshape=(None,),
                            )
                            g_cake.create_dataset("radial", data=r["cake_radial"])
                            g_cake.create_dataset("azimuthal", data=r["cake_azimuthal"])
                        # A1: use the O(1) counter.
                        slot = cake_slot
                        ds_cake[slot] = r["cake"]
                        ds_cake_idx[slot] = i
                        cake_slot += 1
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"[PROGRESS] {done} {total}", flush=True)

            if num_workers <= 1:
                _pool_init(str(handoff.accepted_poni), mask_file, settings)
                for task in tasks:
                    _store(_integrate_one(task))
            else:
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor
                # "spawn" everywhere: fork-after-OpenMP (numpy/pyFAI loaded in the
                # parent) deadlocks on Linux, and Windows only has spawn anyway.
                with ProcessPoolExecutor(
                    max_workers=num_workers, initializer=_pool_init,
                    initargs=(str(handoff.accepted_poni), mask_file, settings),
                    mp_context=multiprocessing.get_context("spawn"),
                ) as pool:
                    for r in pool.map(_integrate_one, tasks, chunksize=4):
                        _store(r)

            # A6: record whether the radial axis was populated.
            h5.attrs["radial_written"] = bool(radial_written)

            # A1: resize cake datasets down to actually-written rows (removes
            # phantom -1/NaN rows when some caked frames failed).
            if g_cake is not None and ds_cake is not None and cake_slot < n_cakes:
                ds_cake.resize(cake_slot, axis=0)
                ds_cake_idx.resize((cake_slot,))

        # A5: all writes succeeded — atomically replace the final path.
        os.replace(h5_tmp, h5_path)

    except Exception:
        # A5: clean up the partial temp file so no zero-data stub is left.
        try:
            if h5_tmp.exists():
                h5_tmp.unlink()
        except OSError:
            pass
        raise

    elapsed = time.time() - t_start
    manifest: Dict[str, Any] = {
        "tool_version": VERSION,
        "created_at": now_iso(),
        "session_name": session_name,
        "handoff_file": str(handoff.path),
        "accepted_poni": str(handoff.accepted_poni),
        "accepted_generation": handoff.accepted_generation,
        "dataset_dir": str(config.get("dataset_dir", "")),
        "file_patterns": config.get("file_patterns", DEFAULT_PATTERNS),
        "n_frames": total,
        "n_failed": len(failures),
        "failures": failures[:50],
        "settings": {k: v for k, v in settings.items()},
        "npt_1d_suggested": npt_suggested,
        "npt_1d_mode": npt_mode,
        "save_cakes": save_cakes,
        "cake_every": cake_every,
        "make_thumbnails": make_thumbnails,
        "previews_dir": str(previews_dir) if make_thumbnails else "",
        "num_workers": num_workers,
        "elapsed_seconds": round(elapsed, 2),
        "h5_file": str(h5_path),
        "config": config,
    }
    if manifest_nexus:
        manifest["nexus_metadata"] = manifest_nexus
    if robust_estimators:
        manifest["robust_estimator"] = robust_estimators[0]
        if len(robust_estimators) > 1:   # mixed pyFAI fallbacks across workers
            manifest["robust_estimators_all"] = robust_estimators
        if any(e.startswith("median(band_unsupported") for e in robust_estimators):
            print("[REDUCE] WARNING: this pyFAI accepts none of the quantile-band "
                  "arguments (it logs 'Got unknown argument ...' and ignores them), "
                  "so the robust channel fell back to a PURE MEDIAN — quantized on "
                  "integer counts (staircase patterns at low intensity). Upgrade "
                  "pyFAI, or fit downstream on the sigma-clip / 'mean' source.",
                  flush=True)
    # B3: surface robust-pattern warnings in the manifest.
    if robust_errors:
        manifest["robust_warnings"] = robust_errors[:10]
        print(f"[REDUCE] WARNING: robust pattern unavailable: {robust_errors[0]}", flush=True)
    if sigmaclip_errors:
        manifest["sigmaclip_warnings"] = sigmaclip_errors[:10]
        print(f"[REDUCE] WARNING: sigma-clip pattern unavailable: {sigmaclip_errors[0]}", flush=True)

    # Geometry health check: when cakes were saved, fit the ring waviness on a
    # few of them right away. A large first harmonic means the sample sat off
    # the calibrant position — every 1D peak becomes a doublet, and it is far
    # cheaper to learn that now than after a full analysis run.
    if save_cakes and n_cakes:
        try:
            from .straighten import diagnose_reduced
            rep = diagnose_reduced(h5_path, max_cakes=3)
            summ = rep.get("summary") or {}
            if rep.get("ok") and summ:
                manifest["geometry_check"] = summ
                bin_w = 0.0
                try:
                    import h5py as _h5c
                    with _h5c.File(str(h5_path), "r") as _f:
                        r = np.asarray(_f["patterns/radial"][:], dtype=float)
                        bin_w = float((r.max() - r.min()) / max(r.size - 1, 1))
                except Exception:
                    pass
                split = float(summ.get("doublet_splitting", 0.0))
                if bin_w > 0 and split > 3.0 * bin_w:
                    off = summ.get("offset_mm")
                    off_txt = f" (implied sample offset ~{off:.3f} mm)" if off else ""
                    print(f"[REDUCE] WARNING: geometry check — ring waviness "
                          f"implies a 1D doublet splitting of {split:.4g} "
                          f"({split / bin_w:.1f} bins){off_txt}. Re-refine the "
                          f"geometry on a sample-position ring and re-reduce, "
                          f"or use the Review tab's straightening tools.",
                          flush=True)
                else:
                    print(f"[REDUCE] geometry check OK (waviness A1 = "
                          f"{summ.get('A1_median', float('nan')):.4g})", flush=True)
        except Exception as e:  # a diagnostic must never fail the reduction
            print(f"[REDUCE] geometry check skipped: {e!r}", flush=True)

    write_json(manifest_path, manifest)
    manifest["manifest_file"] = str(manifest_path)
    print(f"[REDUCE] done: {total - len(failures)}/{total} frames in {elapsed:.1f}s -> {h5_path}", flush=True)
    return manifest
