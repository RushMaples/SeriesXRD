"""Cake-space crystallite spot tracker — per-reflection d(P) curves.

The 1D powder pipeline is structurally blind to a sample that diffracts as
discrete grains — a coarse powder with few grains in the µm-scale DAC beam
(spotty rings; the common case) or an outright single crystal: the reflections
land in a handful of azimuth bins, so the azimuthal median rejects them and
the azimuthal mean dilutes them ~360×. But the saved cakes
(``/cakes/intensity``, one (azimuth × radial) image per frame) still hold them
as isolated blobs. This module recovers them:

  1. **Detection** (:func:`detect_spots`) — per cake, subtract the azimuthal
     median of each radial column (a powder ring lives at *every* azimuth, so
     it cancels; a single-crystal spot does not) and keep connected components
     of the excess above a per-column robust (MAD) noise floor. Azimuth is
     periodic — components are merged across the ±180° seam. Known powder
     lines can additionally be excluded by q-window: the fixed diamond-anvil
     reflections (analytic, no pymatgen) and any Step-3a-attributed peaks of
     the frame (``/peaks/phase``).
  2. **Pressure-point consolidation** (:func:`consolidate_spots`) — several
     frames sample the SAME pressure (a mass scan sweeps beam positions, and
     the crystal drifts across position indices as the cell compresses), so
     detections that agree in (azimuth, q) among all frames at one pressure
     are merged into one observation per reflection per pressure point.
  3. **Pressure-ladder linking** (:func:`link_spot_tracks`) — the consolidated
     points are linked across the sorted pressure ladder with the same
     gap-tolerant one-to-one greedy scheme as
     :func:`analysis.unknowns.link_tracks`, but on (azimuth, q) jointly: a
     reflection keeps ~constant azimuth (fixed crystal orientation) while q
     drifts smoothly with pressure (local-slope predictor, reused from
     ``unknowns``). Gap tolerance matters here — a still-image reflection is
     only visible while it satisfies the Ewald condition, so it appears in
     pressure BANDS with dead stretches between them. ``group_by="scan"``
     restricts consolidation+linking to ``scanNNN`` filename groups for
     datasets where each scan is an independent ladder (default ``"none"`` =
     one global ladder: same crystal seen from every beam position).

The driver (:func:`run_spot_tracking`) appends ``/spots`` (obs/, points/,
tracks/, groups/ — obs/tracks mirroring ``/unknowns``) with an atomic
tmp+``os.replace`` write. A track's points rows (each carrying pressure and d)
ARE its d(P) table, and each track carries ``d0`` — its d-spacing at the
lowest pressure — for matching against a calculated reflection list
(:func:`load_reflection_table` + :func:`match_tracks`), e.g. to identify the
(00l) reflections whose d *grows* with pressure (negative linear
compressibility of the c axis).

CLI: ``bulkxrd-spots`` (or ``python -m bulkxrd.analysis.spots``).
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .identify import radial_to_d
from .frame_metadata import extract_pressures
from .unknowns import _decode_text, _predict_center, _scan_label

SCHEMA_VERSION = "1"
GROUP_MODES = ("none", "scan")

#: Ambient diamond lattice constant (Å) — Occelli et al. (2003), matches the
#: bundled "Diamond (C)" phase entry.
DIAMOND_A0 = 3.56712


# ---------------------------------------------------------------------------
# Small circular-angle helpers (azimuth is periodic in degrees)
# ---------------------------------------------------------------------------

def circ_diff(a, b) -> np.ndarray:
    """Minimal absolute angular difference |a − b| in degrees, ∈ [0, 180]."""
    return np.abs((np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0)


def circ_mean(deg, weights=None) -> float:
    """Weighted circular mean of angles in degrees, in (−180, 180]."""
    ang = np.radians(np.asarray(deg, float))
    w = np.ones_like(ang) if weights is None else np.asarray(weights, float)
    return float(np.degrees(np.arctan2(np.sum(w * np.sin(ang)),
                                       np.sum(w * np.cos(ang)))))


# ---------------------------------------------------------------------------
# Diamond-anvil exclusion windows (analytic — cubic, no pymatgen)
# ---------------------------------------------------------------------------

def diamond_q_lines(q_max: float, a: float = DIAMOND_A0) -> np.ndarray:
    """Ambient q positions (Å⁻¹) of the allowed diamond-structure reflections.

    Fd-3m diamond selection rules: h,k,l all odd, or all even with
    h+k+l ≡ 0 (mod 4). Cubic, so d = a/√(h²+k²+l²) — fully analytic.
    """
    q_max = float(q_max)
    lines: set = set()
    n_max = int(np.ceil((a * q_max / (2.0 * np.pi)) ** 2))
    h_max = int(np.ceil(a * q_max / (2.0 * np.pi)))
    for h in range(h_max + 1):
        for k in range(h + 1):
            for l in range(k + 1):
                n = h * h + k * k + l * l
                if n == 0 or n > n_max:
                    continue
                parities = (h % 2, k % 2, l % 2)
                if all(parities):
                    allowed = True
                elif not any(parities):
                    allowed = (h + k + l) % 4 == 0
                else:
                    allowed = False
                if not allowed:
                    continue
                q = 2.0 * np.pi * np.sqrt(n) / a
                if q <= q_max:
                    lines.add(round(q, 9))
    return np.asarray(sorted(lines), dtype=float)


def diamond_q_windows(q_max: float, *, a: float = DIAMOND_A0,
                      rel_tol: float = 0.02,
                      max_compression: float = 0.04) -> List[Tuple[float, float]]:
    """(q_lo, q_hi) exclusion windows around each diamond line up to ``q_max``.

    Windows extend further UP in q (``max_compression``) than down, because the
    anvil lattice only ever shrinks under load (q grows ~1.2 %/10 GPa at the
    hydrostatic limit; the stressed culet reaches further — 0.04 covers ~50 GPa
    loads with margin).
    """
    qs = diamond_q_lines(q_max * (1.0 + rel_tol + max_compression) + 1e-9, a=a)
    return [(float(q * (1.0 - rel_tol)), float(q * (1.0 + rel_tol + max_compression)))
            for q in qs]


# ---------------------------------------------------------------------------
# Per-cake spot detection
# ---------------------------------------------------------------------------

def detect_spots(cake, radial, azimuthal, *,
                 min_snr: float = 6.0, min_intensity: float = 20.0,
                 min_pixels: int = 2, min_azim_samples: int = 30,
                 max_azim_extent: float = 45.0, max_radial_extent: int = 30,
                 exclude_q_windows: "Sequence[Tuple[float, float]]" = (),
                 exclude_peaks: "Optional[np.ndarray]" = None,
                 exclude_fwhm_mult: float = 2.0) -> List[Dict[str, float]]:
    """Detect single-crystal spots in one cake (azimuth × radial image).

    Per radial column the azimuthal median is the powder + smooth-background
    level (a ring occupies every azimuth bin); the excess above it is spot
    signal. Pixels with ``excess > max(min_snr·1.4826·MAD, min_intensity)``
    are grouped into 8-connected components (merged across the azimuth seam)
    and each component becomes one spot candidate.

    Filters: components smaller than ``min_pixels`` (zingers/hot pixels),
    wider than ``max_azim_extent`` degrees (textured-ring arcs, not spots),
    longer than ``max_radial_extent`` radial bins (streak artifacts), columns
    with fewer than ``min_azim_samples`` finite azimuth bins (no meaningful
    median), peak-q inside any ``exclude_q_windows`` (diamond lines), or
    within ``exclude_fwhm_mult × fwhm`` of an attributed powder peak
    (``exclude_peaks`` rows of (center, fwhm)).

    Returns spot dicts sorted by decreasing peak intensity:
    ``{q, azim, intensity, area, snr, q_width, azim_width, n_pixels}``
    (q/azim are excess-weighted centroids; widths are 2.355·RMS, floored at
    one bin).
    """
    c = np.asarray(cake, dtype=float)
    radial = np.asarray(radial, dtype=float)
    azim = np.asarray(azimuthal, dtype=float)
    if c.ndim != 2 or c.shape != (azim.size, radial.size):
        raise ValueError(f"cake shape {c.shape} does not match "
                         f"(n_azim={azim.size}, n_radial={radial.size})")
    n_az, n_rad = c.shape
    bin_r = float(np.median(np.abs(np.diff(radial)))) if n_rad > 1 else 1.0
    bin_a = float(np.median(np.abs(np.diff(azim)))) if n_az > 1 else 1.0

    finite = np.isfinite(c)
    n_fin = finite.sum(axis=0)
    ok_cols = n_fin >= max(int(min_azim_samples), 1)
    col_med = np.full(n_rad, np.nan)
    col_mad = np.full(n_rad, np.nan)
    if ok_cols.any():
        col_med[ok_cols] = np.nanmedian(c[:, ok_cols], axis=0)
    excess = c - col_med[None, :]
    if ok_cols.any():
        col_mad[ok_cols] = np.nanmedian(np.abs(excess[:, ok_cols]), axis=0)
    sigma = 1.4826 * col_mad
    thr = np.maximum(float(min_snr) * sigma, float(min_intensity))
    mask = finite & ok_cols[None, :]
    mask &= np.where(np.isfinite(excess), excess, -np.inf) > thr[None, :]

    lab, n_lab = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n_lab == 0:
        return []

    # Azimuth is periodic: merge components touching across the ±180° seam
    # (8-connectivity: last row's pixel r is adjacent to first row's r−1..r+1).
    parent = np.arange(n_lab + 1)

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n_az > 2:
        top, bot = lab[0, :], lab[n_az - 1, :]
        for r in np.nonzero(bot)[0]:
            for rr in (r - 1, r, r + 1):
                if 0 <= rr < n_rad and top[rr]:
                    parent[_find(int(bot[r]))] = _find(int(top[rr]))
    roots = np.array([_find(i) for i in range(n_lab + 1)])
    lab = roots[lab]

    windows = [(float(lo), float(hi)) for lo, hi in exclude_q_windows]
    excl = None
    if exclude_peaks is not None:
        excl = np.atleast_2d(np.asarray(exclude_peaks, dtype=float))
        if excl.size == 0:
            excl = None

    out: List[Dict[str, float]] = []
    for cid in np.unique(lab):
        if cid == 0:
            continue
        ai, ri = np.nonzero(lab == cid)
        if ai.size < int(min_pixels):
            continue
        vals = excess[ai, ri]
        k = int(np.argmax(vals))
        peak = float(vals[k])
        r_pk = int(ri[k])
        if int(ri.max() - ri.min()) + 1 > int(max_radial_extent):
            continue
        # Wrap-aware azimuthal extent: index offsets relative to the peak bin.
        rel = (ai - ai[k] + n_az // 2) % n_az - n_az // 2
        if (int(rel.max() - rel.min()) + 1) * bin_a > float(max_azim_extent):
            continue
        q_pk = float(radial[r_pk])
        if any(lo <= q_pk <= hi for lo, hi in windows):
            continue
        if excl is not None and np.any(
                np.abs(q_pk - excl[:, 0])
                <= np.maximum(float(exclude_fwhm_mult) * excl[:, 1], bin_r)):
            continue
        w = np.clip(vals, 0.0, None)
        wsum = float(w.sum()) or 1.0
        q_c = float(np.sum(w * radial[ri]) / wsum)
        az_c = circ_mean(azim[ai], weights=w)
        q_wid = max(2.3548 * float(np.sqrt(np.sum(w * (radial[ri] - q_c) ** 2) / wsum)),
                    bin_r)
        rel_deg = rel * bin_a
        rel_c = float(np.sum(w * rel_deg) / wsum)
        az_wid = max(2.3548 * float(np.sqrt(np.sum(w * (rel_deg - rel_c) ** 2) / wsum)),
                     bin_a)
        snr = peak / float(sigma[r_pk]) if np.isfinite(sigma[r_pk]) and sigma[r_pk] > 0 \
            else float("inf")
        out.append({"q": q_c, "azim": az_c, "intensity": peak,
                    "area": float(vals.sum()), "snr": float(snr),
                    "q_width": q_wid, "azim_width": az_wid,
                    "n_pixels": int(ai.size)})
    out.sort(key=lambda s: -s["intensity"])
    return out


# ---------------------------------------------------------------------------
# Pressure-point consolidation (all frames at one pressure → one obs per spot)
# ---------------------------------------------------------------------------

def consolidate_spots(q, azim, intensity, area, frame, *,
                      q_tol: float, azim_tol: float = 6.0) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Merge detections of ONE pressure point into per-reflection observations.

    Greedy, strongest first: a detection joins an existing consolidated spot
    when it is within ``q_tol`` of its running (intensity-weighted) q centroid
    and within ``azim_tol`` degrees of its circular-mean azimuth; otherwise it
    seeds a new spot. Returns ``(assignment, spots)`` — the consolidated-spot
    index of every input detection and one summary dict per spot
    (``{q, azim, q_width, azim_width, intensity, area, n_frames, best_frame}``,
    intensity = strongest member's peak excess, sorted by it).
    """
    q = np.asarray(q, float)
    azim = np.asarray(azim, float)
    intensity = np.asarray(intensity, float)
    area = np.asarray(area, float)
    frame = np.asarray(frame, int)
    order = np.argsort(-intensity)
    members: List[List[int]] = []
    cen_q: List[float] = []
    cen_az: List[float] = []
    for i in order:
        placed = False
        for s in range(len(members)):
            if (abs(q[i] - cen_q[s]) <= float(q_tol)
                    and circ_diff(azim[i], cen_az[s]) <= float(azim_tol)):
                members[s].append(int(i))
                idx = members[s]
                w = intensity[idx]
                cen_q[s] = float(np.sum(w * q[idx]) / np.sum(w))
                cen_az[s] = circ_mean(azim[idx], weights=w)
                placed = True
                break
        if not placed:
            members.append([int(i)])
            cen_q.append(float(q[i]))
            cen_az.append(float(azim[i]))
    assign = np.full(q.size, -1, dtype=int)
    spots: List[Dict[str, Any]] = []
    rank = sorted(range(len(members)), key=lambda s: -float(intensity[members[s]].max()))
    for new_id, s in enumerate(rank):
        idx = np.asarray(members[s], int)
        assign[idx] = new_id
        w = intensity[idx]
        best = int(idx[np.argmax(w)])
        spread = float(np.sqrt(np.sum(w * (q[idx] - cen_q[s]) ** 2) / np.sum(w)))
        az_spread = float(np.sqrt(np.sum(w * circ_diff(azim[idx], cen_az[s]) ** 2)
                                  / np.sum(w)))
        spots.append({"q": cen_q[s], "azim": cen_az[s],
                      "q_width": spread, "azim_width": az_spread,
                      "intensity": float(w.max()), "area": float(area[idx].sum()),
                      "n_frames": int(np.unique(frame[idx]).size),
                      "best_frame": int(frame[best])})
    return assign, spots


# ---------------------------------------------------------------------------
# Pressure-ladder linking (unknowns.link_tracks, but joint (azimuth, q))
# ---------------------------------------------------------------------------

def link_spot_tracks(ladder_pos, q, azim, intensity, *,
                     axis_values, max_gap: int = 3, min_track_points: int = 3,
                     link_q_rel: float = 0.05, link_q_floor: float = 0.0,
                     link_azim_tol: float = 8.0,
                     axis_predictor: bool = True) -> List[Dict[str, np.ndarray]]:
    """Chain consolidated spots into tracks along an ordered pressure ladder.

    Same gap-tolerant greedy one-to-one scheme as ``unknowns.link_tracks``,
    walking the ladder positions in order: a spot joins an open track when its
    q is within ``max(link_q_rel·q_pred, link_q_floor)`` of the track's
    predicted q (local d(P)-slope predictor, reused from ``unknowns``) AND its
    azimuth is within ``link_azim_tol`` degrees of the track's last azimuth
    (the crystal orientation is fixed; only slow drift is allowed). Candidates
    are ranked by the joint normalized distance, closest pair first. Tracks
    missing from more than ``max_gap`` consecutive ladder positions are
    retired — still-image reflections are only visible in pressure bands
    (Ewald condition), so gaps are normal. Tracks seen at fewer than
    ``min_track_points`` positions are dropped as noise.

    ``ladder_pos`` is each spot's 0-based ladder position; ``axis_values`` the
    per-position physical axis (pressure). Returns one dict per track:
    ``{spots, orders, centers, azims, amplitudes, axis}`` sorted by total
    intensity (``spots`` indexes the input arrays).
    """
    ladder_pos = np.asarray(ladder_pos, int)
    q = np.asarray(q, float)
    azim = np.asarray(azim, float)
    intensity = np.asarray(intensity, float)
    axis_values = np.asarray(axis_values, float)

    open_tracks: List[Dict[str, list]] = []
    done: List[Dict[str, list]] = []
    for pos in range(axis_values.size):
        axis_now = float(axis_values[pos])
        rows = np.nonzero(ladder_pos == pos)[0]
        still: List[Dict[str, list]] = []
        for t in open_tracks:
            if pos - int(t["orders"][-1]) > int(max_gap) + 1:
                done.append(t)
            else:
                still.append(t)
        open_tracks = still
        if rows.size == 0:
            continue
        cands: List[Tuple[float, int, int]] = []
        for ti, t in enumerate(open_tracks):
            q_pred = _predict_center(t, axis_now, use_axis_predictor=bool(axis_predictor))
            q_tol = max(float(link_q_rel) * abs(q_pred), float(link_q_floor), 1e-9)
            az_last = float(t["azims"][-1])
            for r in rows:
                dq = abs(q[r] - q_pred) / q_tol
                daz = float(circ_diff(azim[r], az_last)) / max(float(link_azim_tol), 1e-9)
                if dq <= 1.0 and daz <= 1.0:
                    cands.append((float(np.hypot(dq, daz)), ti, int(r)))
        cands.sort(key=lambda x: x[0])
        used_t: set = set()
        used_r: set = set()
        for g, ti, r in cands:
            if ti in used_t or r in used_r:
                continue
            used_t.add(ti)
            used_r.add(r)
            t = open_tracks[ti]
            t["spots"].append(int(r))
            t["orders"].append(pos)
            t["centers"].append(float(q[r]))
            t["azims"].append(float(azim[r]))
            t["amplitudes"].append(float(intensity[r]))
            t["axis"].append(axis_now)
        for r in rows:
            if int(r) not in used_r:
                open_tracks.append({"spots": [int(r)], "orders": [pos],
                                    "centers": [float(q[r])], "azims": [float(azim[r])],
                                    "amplitudes": [float(intensity[r])],
                                    "axis": [axis_now]})
    done.extend(open_tracks)
    kept = [{k: np.asarray(v) for k, v in t.items()}
            for t in done if len(t["spots"]) >= int(min_track_points)]
    kept.sort(key=lambda t: -float(np.sum(t["amplitudes"])))
    return kept


# ---------------------------------------------------------------------------
# Reflection-list matcher
# ---------------------------------------------------------------------------

def load_reflection_table(path: "str | Path") -> Dict[str, Any]:
    """Load a calculated reflection list from a whitespace table.

    Accepts either a plain two-column ``d intensity`` file or an hkl table
    like ``h k l d(Å) ... I ...`` (e.g. a VESTA/Mercury export). A header line
    is used to locate the ``d`` and ``I`` columns by name when present
    (unit-only tokens such as ``(Å)`` are ignored); otherwise: 2 columns →
    (d, I); ≥4 columns with integer first triple → hkl + d in column 3 and
    intensity in the last column whose maximum is ~100 (normalized I), else
    column 4. Returns ``{"d", "intensity", "hkl"}`` (hkl/intensity may be None).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Reflection file not found: {p}")
    header: List[str] = []
    rows: List[List[float]] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        try:
            vals = [float(t) for t in toks]
        except ValueError:
            if not rows and not header:
                # First non-numeric line = the header; drop unit-only tokens.
                header = [t for t in toks if not t.startswith("(")]
            continue
        rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric reflection rows found in {p}")
    ncols = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == ncols]
    data = np.asarray(rows, dtype=float)

    d_col = i_col = None
    hkl_cols: Optional[Tuple[int, int, int]] = None
    if header and len(header) == ncols:
        low = [t.lower() for t in header]
        for i, t in enumerate(low):
            if t == "d" or t.startswith("d("):
                d_col = i
            elif t in ("i", "int", "intensity", "irel", "i_rel", "i(%)"):
                i_col = i
        if all(t in low for t in ("h", "k", "l")):
            hkl_cols = (low.index("h"), low.index("k"), low.index("l"))
    if d_col is None:
        if ncols == 2:
            d_col, i_col = 0, 1
        elif ncols >= 4 and np.allclose(data[:, :3], np.round(data[:, :3])):
            hkl_cols = (0, 1, 2)
            d_col = 3
            if i_col is None:
                normed = [j for j in range(4, ncols)
                          if abs(float(np.nanmax(data[:, j])) - 100.0) < 1e-6]
                i_col = normed[-1] if normed else (4 if ncols > 4 else None)
        else:
            raise ValueError(
                f"Cannot locate the d column in {p} ({ncols} columns, "
                f"header={header or 'none'}) — expected 'd I' pairs or an "
                "'h k l d ... I' table.")
    return {"d": data[:, d_col],
            "intensity": data[:, i_col] if i_col is not None else None,
            "hkl": data[:, hkl_cols].astype(int) if hkl_cols is not None else None}


def match_tracks(d0, table: Dict[str, Any], *, rel_tol: float = 0.03,
                 top: int = 3) -> List[List[Dict[str, Any]]]:
    """Match per-track ``d0`` values against a loaded reflection table.

    For each d0, returns up to ``top`` reflections with |Δd|/d_calc ≤
    ``rel_tol``, closest first: ``{d_calc, delta_rel, intensity, hkl}``.
    """
    d_calc = np.asarray(table["d"], float)
    inten = table.get("intensity")
    hkl = table.get("hkl")
    out: List[List[Dict[str, Any]]] = []
    for d in np.atleast_1d(np.asarray(d0, float)):
        if not np.isfinite(d):
            out.append([])
            continue
        rel = np.abs(d - d_calc) / np.where(d_calc > 0, d_calc, np.inf)
        idx = np.nonzero(rel <= float(rel_tol))[0]
        idx = idx[np.argsort(rel[idx])][: int(top)]
        out.append([{"d_calc": float(d_calc[j]),
                     "delta_rel": float((d - d_calc[j]) / d_calc[j]),
                     "intensity": (float(inten[j]) if inten is not None else None),
                     "hkl": (tuple(int(v) for v in hkl[j]) if hkl is not None else None)}
                    for j in idx])
    return out


def load_spot_tracks(
    spots_h5: "str | Path",
    *,
    min_points: int = 1,
    match: "Optional[str | Path]" = None,
    match_tol: float = 0.03,
) -> Dict[str, Any]:
    """Read ``/spots`` into a plot-ready structure (GUI d(P) view).

    Returns ``{"ok": bool, "error": str?, "unit": str, "tracks": [...],
    "untracked": {"pressure", "d"}, "n_tracks_total": int}`` where each track
    dict carries pressure-sorted ``pressure``/``d``/``intensity`` arrays plus
    ``id``, ``group_label``, ``azim``, ``dd_dp``, ``d0``, ``n_points`` and —
    when a calculated reflection table is supplied via ``match`` — ``hkl`` (a
    label string, "" if nothing within ``match_tol``). Tracks with fewer than
    ``min_points`` pressure points are dropped.
    """
    import h5py  # type: ignore

    src = Path(spots_h5).expanduser().resolve()
    if not src.is_file():
        return {"ok": False, "error": f"file not found: {src}"}
    try:
        with h5py.File(str(src), "r") as h:
            g = h.get("spots")
            if g is None or "tracks" not in g:
                return {"ok": False,
                        "error": "no /spots group — run bulkxrd-spots first"}
            unit = str(g.attrs.get("unit", "q_A^-1"))
            gt, gp, gg = g["tracks"], g["points"], g["groups"]
            labels = [_decode_text(v) for v in gg["label"][:]]
            tracks = {k: np.asarray(gt[k][:]) for k in gt.keys()}
            points = {k: np.asarray(gp[k][:]) for k in
                      ("track", "pressure", "d", "intensity")}
    except Exception as e:                                  # unreadable file
        return {"ok": False, "error": str(e)}

    n_total = int(tracks["id"].size)
    hkl_labels = [""] * n_total
    if match and n_total:
        try:
            table = load_reflection_table(match)
            for i, m in enumerate(match_tracks(tracks["d0"], table,
                                               rel_tol=match_tol, top=1)):
                if m and m[0]["hkl"] is not None:
                    hkl_labels[i] = " ".join(str(v) for v in m[0]["hkl"])
                elif m:
                    hkl_labels[i] = f"d={m[0]['d_calc']:.3f}"
        except Exception:
            pass                                            # labels stay blank

    out_tracks: List[Dict[str, Any]] = []
    for i in range(n_total):
        if int(tracks["n_points"][i]) < int(min_points):
            continue
        rows = np.nonzero(points["track"] == int(tracks["id"][i]))[0]
        order = rows[np.argsort(points["pressure"][rows])]
        gid = int(tracks["group"][i])
        out_tracks.append({
            "id": int(tracks["id"][i]),
            "group_label": labels[gid] if 0 <= gid < len(labels) else "all",
            "pressure": points["pressure"][order].astype(float),
            "d": points["d"][order].astype(float),
            "intensity": points["intensity"][order].astype(float),
            "azim": float(tracks["azim"][i]),
            "dd_dp": float(tracks["dd_dp"][i]),
            "d0": float(tracks["d0"][i]),
            "n_points": int(tracks["n_points"][i]),
            "hkl": hkl_labels[i],
        })
    un = np.nonzero(points["track"] < 0)[0]
    return {"ok": True, "unit": unit, "n_tracks_total": n_total,
            "tracks": out_tracks,
            "untracked": {"pressure": points["pressure"][un].astype(float),
                          "d": points["d"][un].astype(float)}}


def export_ring_removed_cakes(
    reduced_h5: "str | Path",
    out_dir: "str | Path",
    frames: "Sequence[int]",
    *,
    write_png: bool = True,
    write_npy: bool = True,
    vmax_percentile: float = 99.5,
) -> Dict[str, Any]:
    """Export selected frames' cakes with the powder rings removed.

    Writes, per frame, the exact image the spot detector works on: the cake
    minus each radial column's azimuthal median (a ring lives at every
    azimuth, so the median IS the ring level; subtracting it leaves only the
    azimuthally-sparse crystallite spots). Outputs per frame:

        cake_ringless_f<frame>.png   quick-look image (radial × azimuth, robust
                                     contrast; needs matplotlib, else skipped)
        cake_ringless_f<frame>.npy   the raw excess array (azimuth × radial),
                                     NaN where the detector is masked

    plus once: ``cake_axes.npz`` (``radial`` q values + ``azimuthal`` degrees,
    to put physical axes on the .npy data). Returns a manifest dict.
    """
    import h5py  # type: ignore

    src = Path(reduced_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Reduced HDF5 not found: {src}")
    dest = Path(out_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    with h5py.File(str(src), "r") as h:
        cg = h.get("cakes")
        if cg is None or "intensity" not in cg:
            raise ValueError(f"No /cakes/intensity in {src} — re-run the "
                             "reduction with cake saving enabled.")
        radial = np.asarray(cg["radial"][:], float)
        azimuthal = np.asarray(cg["azimuthal"][:], float)
        cake_frames = (np.asarray(cg["frame_index"][:], int)
                       if "frame_index" in cg
                       else np.arange(cg["intensity"].shape[0]))
        idx_of = {int(f): j for j, f in enumerate(cake_frames)}
        np.savez(dest / "cake_axes.npz", radial=radial, azimuthal=azimuthal)

        fig_mod = None
        if write_png:
            try:
                import matplotlib
                matplotlib.use("Agg", force=False)
                from matplotlib.figure import Figure
                fig_mod = Figure
            except Exception:
                fig_mod = None                      # data-only export

        for fi in frames:
            fi = int(fi)
            j = idx_of.get(fi)
            if j is None:
                print(f"[SPOTS] frame {fi}: no cake saved — skipped", flush=True)
                continue
            cake = np.asarray(cg["intensity"][j], float)
            med = np.nanmedian(np.where(np.isfinite(cake), cake, np.nan), axis=0)
            excess = cake - med[None, :]
            if write_npy:
                np.save(dest / f"cake_ringless_f{fi:05d}.npy",
                        excess.astype("f4"))
                written.append(f"cake_ringless_f{fi:05d}.npy")
            if fig_mod is not None:
                fin = excess[np.isfinite(excess)]
                vmax = (float(np.percentile(fin, vmax_percentile))
                        if fin.size else 1.0) or 1.0
                fig = fig_mod(figsize=(9, 4), dpi=110, layout="constrained")
                ax = fig.add_subplot(1, 1, 1)
                im = ax.imshow(
                    np.where(np.isfinite(excess), excess, 0.0), origin="lower",
                    aspect="auto", cmap="magma", vmin=0.0, vmax=max(vmax, 1.0),
                    extent=(float(radial[0]), float(radial[-1]),
                            float(azimuthal[0]), float(azimuthal[-1])))
                fig.colorbar(im, ax=ax, label="counts above ring level")
                ax.set_xlabel("q (Å⁻¹)" if "q" in str(h.attrs.get("unit", "q"))
                              else "radial")
                ax.set_ylabel("azimuth (°)")
                ax.set_title(f"frame {fi} — rings removed "
                             f"(azimuthal-median subtracted)")
                fig.savefig(dest / f"cake_ringless_f{fi:05d}.png")
                written.append(f"cake_ringless_f{fi:05d}.png")

    print(f"[SPOTS] ring-removed cakes: {len(written)} file(s) -> {dest}",
          flush=True)
    return {"out_dir": str(dest), "files": written}


def export_spot_tracks(
    spots_h5: "str | Path",
    out_dir: "str | Path",
    *,
    match: "Optional[str | Path]" = None,
    match_tol: float = 0.03,
    include_observations: bool = False,
) -> Dict[str, Any]:
    """Export ``/spots`` as a group-handoff CSV bundle.

    Writes into ``out_dir`` (created if needed):

        spot_tracks.csv          one row per track: pressure span, d0, d range,
                                 dd/dP slope, azimuth (+ best matches against a
                                 calculated reflection table when ``match`` is
                                 given — the hkl assignment starting point)
        spot_track_points.csv    long-format d(P) tables — every consolidated
                                 pressure point of every track, ordered by
                                 (track, pressure). THE dataset for in-depth
                                 analysis (EOS fits, axis compressibilities).
        spot_untracked_points.csv  consolidated reflections that never linked
                                 into a track (visible at a single pressure
                                 band — e.g. an (00l) spot that only satisfies
                                 the Ewald condition once). Same columns.
        spot_observations.csv    every raw per-frame detection (optional,
                                 ``include_observations=True``) — recover
                                 which exact frames/cakes to inspect.
        README.txt               provenance: source files, every tracker knob,
                                 column glossary, match table used.

    ``match`` is a calculated-reflection file (see
    :func:`load_reflection_table`); matching uses each track's ``d0`` (its
    d-spacing at the lowest pressure). Returns a manifest dict.
    """
    import csv
    import h5py  # type: ignore

    src = Path(spots_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"HDF5 not found: {src}")
    dest = Path(out_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    table = load_reflection_table(match) if match else None

    with h5py.File(str(src), "r") as h:
        g = h.get("spots")
        if g is None or "tracks" not in g:
            raise ValueError(f"No /spots group in {src} — run bulkxrd-spots first.")
        attrs = {k: g.attrs[k] for k in g.attrs}
        gt, gp, gg = g["tracks"], g["points"], g["groups"]
        labels = [_decode_text(v) for v in gg["label"][:]]
        tracks = {k: np.asarray(gt[k][:]) for k in gt.keys()}
        points = {k: np.asarray(gp[k][:]) for k in gp.keys()}
        obs = ({k: np.asarray(g["obs"][k][:]) for k in g["obs"].keys()}
               if include_observations and "obs" in g else None)
        fr = h.get("frames")
        frame_names = ([_decode_text(v) for v in fr["filename"][:]]
                       if fr is not None and "filename" in fr else None)

    # Frame filenames: the group needs raw-file names, not just indices. A
    # standalone <stem>_spots.h5 has no /frames — fall back to the analysis /
    # reduced files recorded in the /spots provenance attrs.
    if frame_names is None:
        for cand in (str(attrs.get("analysis_h5", "") or ""),
                     str(attrs.get("source_reduced", "") or "")):
            if cand and Path(cand).is_file():
                try:
                    with h5py.File(cand, "r") as hh:
                        fr = hh.get("frames")
                        if fr is not None and "filename" in fr:
                            frame_names = [_decode_text(v) for v in fr["filename"][:]]
                            break
                except Exception:
                    continue

    def _fname(fi: int) -> str:
        return (frame_names[fi]
                if frame_names and 0 <= int(fi) < len(frame_names) else "")

    def _scan(fi: int) -> str:
        n = _fname(fi)
        return _scan_label(n) if n else ""

    n_tracks = int(tracks["id"].size)
    matches = (match_tracks(tracks["d0"], table, rel_tol=match_tol, top=1)
               if table is not None and n_tracks else [[] for _ in range(n_tracks)])

    def _lab(gid: int) -> str:
        return labels[gid] if 0 <= gid < len(labels) else "all"

    # --- spot_tracks.csv
    t_path = dest / "spot_tracks.csv"
    t_cols = ("track", "group", "n_points", "n_frames", "p_min_gpa", "p_max_gpa",
              "d0_A", "d_min_A", "d_max_A", "dd_dp_A_per_gpa", "azim_deg",
              "azim_spread_deg", "intensity_max", "best_frame", "best_frame_file")
    with open(t_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(t_cols) + (["match_hkl", "match_d_calc_A", "match_delta_pct",
                                    "match_intensity"] if table is not None else []))
        for i in range(n_tracks):
            row = [int(tracks["id"][i]), _lab(int(tracks["group"][i])),
                   int(tracks["n_points"][i]), int(tracks["n_frames"][i]),
                   round(float(tracks["p_min"][i]), 4), round(float(tracks["p_max"][i]), 4),
                   round(float(tracks["d0"][i]), 5), round(float(tracks["d_min"][i]), 5),
                   round(float(tracks["d_max"][i]), 5), round(float(tracks["dd_dp"][i]), 6),
                   round(float(tracks["azim"][i]), 2),
                   round(float(tracks["azim_spread"][i]), 2),
                   round(float(tracks["intensity_max"][i]), 1),
                   int(tracks["best_frame"][i]),
                   _fname(int(tracks["best_frame"][i]))]
            if table is not None:
                m = matches[i][0] if matches[i] else None
                row += ([("" if m["hkl"] is None else " ".join(str(v) for v in m["hkl"])),
                         round(m["d_calc"], 5), round(100 * m["delta_rel"], 3),
                         ("" if m["intensity"] is None else round(m["intensity"], 2))]
                        if m else ["", "", "", ""])
            w.writerow(row)

    # --- point tables (tracked, long format, ordered by track then pressure;
    #     untracked separately — single-band reflections still matter)
    p_cols = ("track", "group", "pressure_gpa", "d_A", "q", "azim_deg",
              "q_width", "azim_width_deg", "intensity", "area", "n_frames",
              "best_frame", "best_frame_file")

    def _point_row(j: int) -> list:
        return [int(points["track"][j]), _lab(int(points["group"][j])),
                round(float(points["pressure"][j]), 4), round(float(points["d"][j]), 5),
                round(float(points["q"][j]), 5), round(float(points["azim"][j]), 2),
                round(float(points["q_width"][j]), 5),
                round(float(points["azim_width"][j]), 2),
                round(float(points["intensity"][j]), 1),
                round(float(points["area"][j]), 1),
                int(points["n_frames"][j]), int(points["best_frame"][j]),
                _fname(int(points["best_frame"][j]))]

    tracked = np.nonzero(points["track"] >= 0)[0]
    order = tracked[np.lexsort((points["pressure"][tracked],
                                points["track"][tracked]))]
    pt_path = dest / "spot_track_points.csv"
    with open(pt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(p_cols)
        for j in order:
            w.writerow(_point_row(int(j)))

    untracked = np.nonzero(points["track"] < 0)[0]
    un_path = dest / "spot_untracked_points.csv"
    with open(un_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(p_cols)
        for j in untracked[np.argsort(points["pressure"][untracked])]:
            w.writerow(_point_row(int(j)))

    n_obs_written = 0
    if obs is not None:
        o_path = dest / "spot_observations.csv"
        o_cols = ("frame", "scan", "filename", "group", "point", "track",
                  "pressure_gpa", "d_A", "q", "azim_deg", "intensity", "area",
                  "snr", "q_width", "azim_width_deg", "n_pixels")
        with open(o_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(o_cols)
            for j in np.argsort(obs["frame"]):
                fi = int(obs["frame"][j])
                w.writerow([fi, _scan(fi), _fname(fi),
                            _lab(int(obs["group"][j])),
                            int(obs["point"][j]), int(obs["track"][j]),
                            round(float(obs["pressure"][j]), 4),
                            round(float(obs["d"][j]), 5), round(float(obs["q"][j]), 5),
                            round(float(obs["azim"][j]), 2),
                            round(float(obs["intensity"][j]), 1),
                            round(float(obs["area"][j]), 1),
                            round(float(obs["snr"][j]), 1),
                            round(float(obs["q_width"][j]), 5),
                            round(float(obs["azim_width"][j]), 2),
                            int(obs["n_pixels"][j])])
        n_obs_written = int(obs["frame"].size)

    # --- README with provenance + glossary
    knob_lines = "\n".join(f"  {k} = {attrs[k]!r}" for k in sorted(attrs))
    readme = f"""Crystallite spot-track export (bulkxrd)
=============================================

Source HDF5:      {src}
Exported:         spot_tracks.csv, spot_track_points.csv, spot_untracked_points.csv{
    ', spot_observations.csv' if obs is not None else ''}
Reflection table: {match or '(none — no hkl matching)'}
Match tolerance:  |d_obs - d_calc| / d_calc <= {match_tol} (on each track's d0)

What these are
--------------
Bragg reflections of individual crystallites, detected as azimuthal blobs in
the cake images (smooth powder rings cancel; the azimuthal median rejects
sparse spots, so the 1D patterns never see them). They arise whenever the
sample diffracts as discrete grains — a coarse-grained powder with few grains
in the micron-scale DAC beam (spotty rings; the common case), or a single
crystal. A spot's d-spacing comes from its ring radius alone, so d(P) is valid
either way, and several tracks sharing one hkl at different azimuths are
different grains reflecting the same plane family — independent repeats of the
same d(P). Detections at one pressure are consolidated across beam positions,
then linked along the pressure ladder into tracks: one track = one crystallite
reflection followed under compression. A track's rows in
spot_track_points.csv ARE its d(P) table.

Column glossary
---------------
track           track id (-1 in the untracked file = never linked)
pressure_gpa    frame metadata pressure of the consolidated point
d_A / q         d-spacing (Angstrom) / scattering vector at the point
azim_deg        detector azimuth of the blob (~constant per reflection: the
                grain orientation is fixed; a few degrees of drift under
                non-hydrostatic stress is grain rotation and does not affect d)
d0_A            track d at its LOWEST pressure (match against calculated
                ambient reflections)
dd_dp_A_per_gpa least-squares d-vs-P slope; POSITIVE = d grows under
                pressure (negative linear compressibility along that axis)
azim_spread_deg intensity-weighted azimuthal scatter of the track
best_frame      frame index of the most intense observation (which cake to
                inspect visually); best_frame_file is its raw-image filename
scan / filename (observations file) the scanNNN token and raw-image filename
                of each detection — filter by track to list every raw frame
                where that reflection is visible
intensity/area  blob peak height / integrated counts above the ring median
n_frames        frames contributing to the consolidated point

Untracked points (spot_untracked_points.csv) are reflections seen in only one
pressure band — a fixed grain only diffracts while the Ewald condition holds,
so short-lived spots are physical, not noise. Check them by eye at best_frame
before discarding.

Tracker parameters (as stored in /spots attrs)
----------------------------------------------
{knob_lines}
"""
    (dest / "README.txt").write_text(readme, encoding="utf-8")

    manifest = {"out_dir": str(dest), "n_tracks": n_tracks,
                "n_track_points": int(tracked.size),
                "n_untracked_points": int(untracked.size),
                "n_observations": n_obs_written,
                "matched": table is not None,
                "files": ["spot_tracks.csv", "spot_track_points.csv",
                          "spot_untracked_points.csv"]
                         + (["spot_observations.csv"] if obs is not None else [])
                         + ["README.txt"]}
    print(f"[SPOTS] exported {n_tracks} track(s) "
          f"({int(tracked.size)} points, {int(untracked.size)} untracked) "
          f"-> {dest}", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------

def _wavelength_A(attrs) -> Optional[float]:
    """Wavelength in Å from a reduced file's attrs (parsed out of poni_text)."""
    txt = attrs.get("poni_text")
    if txt:
        m = re.search(r"Wavelength:\s*([0-9.eE+-]+)", str(txt))
        if m:
            try:
                return float(m.group(1)) * 1e10   # PONI stores meters
            except ValueError:
                pass
    return None


def _attributed_peaks(analysis_h5: Path,
                      pressure: "Optional[np.ndarray]" = None) -> "Optional[Dict[int, np.ndarray]]":
    """Per-frame (center, fwhm) rows of Step-3a-attributed peaks, or None.

    When per-frame pressures are known, the attributed peaks are POOLED across
    all frames sharing a pressure: a powder ring sits at the same q in every
    frame of one pressure point, but Step 3a only attributes it in the frames
    where the phase cleared its evidence gate — without pooling, spots of a
    coarse-grained attributed phase (e.g. the W gasket) leak through in the
    frames that were not attributed.
    """
    import h5py  # type: ignore

    with h5py.File(str(analysis_h5), "r") as h:
        pk = h.get("peaks")
        if pk is None or "phase" not in pk:
            return None
        frame = np.asarray(pk["frame"][:], int)
        center = np.asarray(pk["center"][:], float)
        fwhm = np.asarray(pk["fwhm"][:], float)
        phase = np.asarray([_decode_text(v) for v in pk["phase"][:]])
    keep = phase != ""
    per: Dict[int, np.ndarray] = {}
    for fi in np.unique(frame[keep]):
        sel = keep & (frame == fi)
        per[int(fi)] = np.column_stack([center[sel], fwhm[sel]])
    if pressure is None or not np.any(np.isfinite(pressure)):
        return per
    pooled: Dict[float, List[np.ndarray]] = {}
    for fi, rows in per.items():
        if 0 <= fi < pressure.size and np.isfinite(pressure[fi]):
            pooled.setdefault(round(float(pressure[fi]), 9), []).append(rows)
    out: Dict[int, np.ndarray] = dict(per)
    for fi in range(pressure.size):
        if np.isfinite(pressure[fi]):
            group = pooled.get(round(float(pressure[fi]), 9))
            if group:
                out[int(fi)] = np.vstack(group)
    return out


def run_spot_tracking(
    reduced_h5: "str | Path",
    analysis_h5: "Optional[str | Path]" = None,
    *,
    out_h5: "Optional[str | Path]" = None,
    # detection
    min_snr: float = 6.0,
    min_intensity: float = 20.0,
    min_pixels: int = 2,
    min_azim_samples: int = 30,
    max_azim_extent: float = 45.0,
    max_radial_extent: int = 30,
    q_min: "Optional[float]" = None,
    q_max: "Optional[float]" = None,
    # exclusions
    exclude_diamond: bool = True,
    diamond_a: float = DIAMOND_A0,
    diamond_rel_tol: float = 0.02,
    diamond_max_compression: float = 0.04,
    exclude_attributed: bool = True,
    exclude_fwhm_mult: float = 2.0,
    exclude_d: "Optional[Sequence[float]]" = None,
    exclude_d_tol: float = 0.02,
    exclude_d_compression: float = 0.06,
    # consolidation + linking
    group_by: str = "none",
    q_tol: "Optional[float]" = None,
    azim_tol: float = 6.0,
    link_q_rel: float = 0.05,
    link_azim_tol: float = 8.0,
    max_gap: int = 3,
    min_track_points: int = 3,
    axis_predictor: bool = True,
    exclude_frames: "Optional[Sequence[int]]" = None,
) -> Dict[str, Any]:
    """Detect single-crystal spots in every cake, consolidate them per
    pressure point and link the points across the pressure ladder; append
    ``/spots`` to the target HDF5 (atomic tmp+``os.replace``).

    ``exclude_frames`` drops the listed frame indices from detection entirely
    (on top of ``/frames/excluded``/``ok``) — the seam for known-bad exposures,
    e.g. sweeps taken with a beam cover left on, whose foreign-material spots
    would otherwise form stationary fake tracks.

    All frames sharing one pressure (a mass scan sweeps beam positions over
    the same crystal) are consolidated into one observation per reflection,
    then the consolidated points are linked along the sorted pressure ladder.
    ``group_by="scan"`` runs an independent ladder per ``scanNNN`` filename
    group instead (datasets where scans are separate series).

        /spots  attrs: schema_version, source_reduced, unit, ladder,
                       pressure_source, group_by, n_obs, n_points, n_tracks,
                       n_groups + every knob
        /spots/obs/{frame,group,point,track,pressure,q,d,azim,intensity,area,
                    snr,q_width,azim_width,n_pixels}      every detection
        /spots/groups/{id,label,n_frames,n_obs,n_tracks}
        /spots/points/{id,group,order,track,pressure,q,d,azim,q_width,
                    azim_width,intensity,area,n_frames,best_frame}
                    one row per reflection per pressure point — a track's
                    rows (track == t, ordered by pressure) ARE its d(P) table
        /spots/tracks/{id,group,n_points,n_frames,p_min,p_max,q_first,q_last,
                    d0,d_min,d_max,dd_dp,azim,azim_spread,intensity_max,
                    best_frame}

    ``d0`` is each track's d-spacing at its lowest pressure (the value to
    match against a calculated ambient reflection list); ``dd_dp`` the
    least-squares d-vs-P slope (Å/GPa) — positive on a
    negative-linear-compressibility axis.

    The target is ``out_h5`` if given, else ``analysis_h5`` (appended, like
    the other analysis groups), else ``<reduced stem>_spots.h5`` next to the
    reduced file (the multi-GB reduced file itself is never rewritten).
    Frame pressures come from the analysis file when given, else from
    ``/frames/pressure``, else parsed from filenames. Frames without a finite
    pressure are detected but not consolidated/linked.
    """
    import h5py  # type: ignore

    group_key = (group_by or "none").strip().lower()
    if group_key not in GROUP_MODES:
        raise ValueError(f"group_by must be one of {GROUP_MODES}, got {group_by!r}")
    src = Path(reduced_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Reduced HDF5 not found: {src}")
    ana = Path(analysis_h5).expanduser().resolve() if analysis_h5 else None
    if ana is not None and not ana.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {ana}")
    if out_h5:
        dst = Path(out_h5).expanduser().resolve()
    elif ana is not None:
        dst = ana
    else:
        dst = src.with_name(src.stem + "_spots.h5")

    with h5py.File(str(src), "r") as h:
        cg = h.get("cakes")
        if cg is None or "intensity" not in cg:
            raise ValueError(f"No /cakes/intensity in {src} — re-run the reduction "
                             "with cake saving enabled (cakes for every frame).")
        radial = np.asarray(cg["radial"][:], float)
        azimuthal = np.asarray(cg["azimuthal"][:], float)
        cake_frames = (np.asarray(cg["frame_index"][:], int)
                       if "frame_index" in cg else np.arange(cg["intensity"].shape[0]))
        unit = str(h.attrs.get("unit", "q_A^-1"))
        wavelength = _wavelength_A(h.attrs)
        fr = h.get("frames")
        n_frames = int(fr["filename"].shape[0]) if fr is not None and "filename" in fr \
            else int(cake_frames.max()) + 1
        names = ([_decode_text(v) for v in fr["filename"][:]]
                 if fr is not None and "filename" in fr else [""] * n_frames)
        skip = np.zeros(n_frames, bool)
        if fr is not None:
            if "excluded" in fr:
                skip |= np.asarray(fr["excluded"][:], bool)
            if "ok" in fr:
                skip |= ~np.asarray(fr["ok"][:], bool)
        n_excluded_arg = 0
        if exclude_frames is not None:
            ef = np.asarray(sorted(set(int(f) for f in exclude_frames)), int)
            ef = ef[(ef >= 0) & (ef < n_frames)]
            skip[ef] = True
            n_excluded_arg = int(ef.size)
            if n_excluded_arg:
                print(f"[SPOTS] excluding {n_excluded_arg} listed frame(s) "
                      f"from detection", flush=True)

        # --- pressure prior: analysis file > reduced /frames/pressure > filenames
        pressure = np.full(n_frames, np.nan)
        pressure_source = "none"
        if ana is not None:
            with h5py.File(str(ana), "r") as ha:
                fa = ha.get("frames")
                if fa is not None and "pressure" in fa \
                        and fa["pressure"].shape[0] == n_frames:
                    p = np.asarray(fa["pressure"][:], float)
                    if np.any(np.isfinite(p)):
                        pressure, pressure_source = p, "analysis"
        if pressure_source == "none" and fr is not None and "pressure" in fr:
            p = np.asarray(fr["pressure"][:], float)
            if np.any(np.isfinite(p)):
                pressure, pressure_source = p, "reduced"
        if pressure_source == "none":
            p = extract_pressures(names)
            if np.any(np.isfinite(p)):
                pressure, pressure_source = p, "filename"

        # --- exclusion windows / attributed peaks
        rmask = np.ones(radial.size, bool)
        if q_min is not None:
            rmask &= radial >= float(q_min)
        if q_max is not None:
            rmask &= radial <= float(q_max)
        rsel = np.nonzero(rmask)[0]
        rad_use = radial[rsel]
        # Exclusion windows are assembled in q (Å⁻¹) then converted to the
        # radial-axis units. All extend further UP in q than down: lattices
        # only shrink under load (diamond anvil, gasket lines alike).
        is_q_axis = unit.lower().startswith("q")
        q_scale = 1.0
        if is_q_axis and "a^" not in unit.lower() \
                and unit.lower() not in ("q", "q_a"):
            q_scale = 0.1                       # q_nm^-1 axis → Å⁻¹
        if is_q_axis:
            q_max_A = float(rad_use.max()) * q_scale
        else:
            if not wavelength:
                raise ValueError("Diamond/d-line exclusion on a 2θ axis needs "
                                 "the wavelength (missing from poni_text).")
            q_max_A = 4.0 * np.pi / wavelength
        windows_A: List[Tuple[float, float]] = []
        if exclude_diamond:
            windows_A += diamond_q_windows(
                q_max_A, a=diamond_a, rel_tol=diamond_rel_tol,
                max_compression=diamond_max_compression)
        excl_d = [float(v) for v in (exclude_d or []) if float(v) > 0]
        for d_line in excl_d:
            q0 = 2.0 * np.pi / d_line
            windows_A.append((q0 * (1.0 - float(exclude_d_tol)),
                              q0 * (1.0 + float(exclude_d_tol)
                                    + float(exclude_d_compression))))
        windows: List[Tuple[float, float]] = []
        for lo_q, hi_q in windows_A:
            if is_q_axis:
                windows.append((lo_q / q_scale, hi_q / q_scale))
            else:
                tth = [2.0 * np.arcsin(np.clip(qq * wavelength / (4.0 * np.pi),
                                               -1.0, 1.0)) for qq in (lo_q, hi_q)]
                if unit.lower() == "2th_deg":
                    tth = [np.degrees(t) for t in tth]
                windows.append((float(tth[0]), float(tth[1])))
        attributed = None
        if ana is not None and exclude_attributed:
            attributed = _attributed_peaks(ana, pressure)

        bin_r = float(np.median(np.abs(np.diff(rad_use)))) if rad_use.size > 1 else 1.0
        if q_tol is None:
            q_tol = 3.0 * bin_r

        # --- per-cake detection
        obs: Dict[str, List[Any]] = {k: [] for k in
                                     ("frame", "q", "azim", "intensity", "area",
                                      "snr", "q_width", "azim_width", "n_pixels")}
        ds = cg["intensity"]
        n_cakes = int(ds.shape[0])
        for i in range(n_cakes):
            fi = int(cake_frames[i])
            if 0 <= fi < n_frames and skip[fi]:
                continue
            cake = np.asarray(ds[i], float)[:, rsel]
            spots = detect_spots(
                cake, rad_use, azimuthal,
                min_snr=min_snr, min_intensity=min_intensity,
                min_pixels=min_pixels, min_azim_samples=min_azim_samples,
                max_azim_extent=max_azim_extent,
                max_radial_extent=max_radial_extent,
                exclude_q_windows=windows,
                exclude_peaks=(attributed.get(fi) if attributed else None),
                exclude_fwhm_mult=exclude_fwhm_mult)
            for s in spots:
                obs["frame"].append(fi)
                for k in ("q", "azim", "intensity", "area", "snr",
                          "q_width", "azim_width", "n_pixels"):
                    obs[k].append(s[k])
            if (i + 1) % 200 == 0 or i + 1 == n_cakes:
                print(f"[SPOTS] detection {i + 1}/{n_cakes} cakes "
                      f"({len(obs['frame'])} spots so far)", flush=True)

    o_frame = np.asarray(obs["frame"], int)
    o_q = np.asarray(obs["q"], float)
    o_az = np.asarray(obs["azim"], float)
    o_int = np.asarray(obs["intensity"], float)
    o_area = np.asarray(obs["area"], float)

    # --- frame groups (one ladder per group; default = one global ladder)
    if group_key == "scan":
        glabels = [_scan_label(nm) for nm in names]
    else:
        glabels = ["all"] * n_frames
    label_ids: Dict[str, int] = {}
    for nm in glabels:
        if nm not in label_ids:
            label_ids[nm] = len(label_ids)
    frame_group = np.asarray([label_ids[nm] for nm in glabels], int)
    n_groups = len(label_ids)
    group_names = [label for label, _ in sorted(label_ids.items(), key=lambda kv: kv[1])]
    ladder = "pressure" if np.any(np.isfinite(pressure)) else "frame"
    obs_group = frame_group[o_frame] if o_frame.size else np.zeros(0, int)

    # --- consolidate per (group, pressure point) and link along each ladder
    obs_point = np.full(o_frame.size, -1, int)
    obs_track = np.full(o_frame.size, -1, int)
    points: Dict[str, List[Any]] = {k: [] for k in
                                    ("group", "order", "pressure", "q", "azim",
                                     "q_width", "azim_width", "intensity",
                                     "area", "n_frames", "best_frame")}
    all_tracks: List[Tuple[int, np.ndarray]] = []    # (group_id, point rows in order)
    for gid in range(n_groups):
        frames_in = np.nonzero(frame_group == gid)[0]
        if ladder == "pressure":
            pv = pressure[frames_in]
            pts = np.unique(np.round(pv[np.isfinite(pv)], 9))
            frame_pt = {int(f): int(np.searchsorted(pts, round(float(pressure[f]), 9)))
                        for f in frames_in if np.isfinite(pressure[f])}
            axis_vals = pts.astype(float)
        else:
            frame_pt = {int(f): i for i, f in enumerate(frames_in)}
            axis_vals = np.arange(frames_in.size, dtype=float)
        if not frame_pt:
            continue
        base = len(points["group"])
        pt_rows: List[int] = []
        for pos in range(axis_vals.size):
            fset = np.asarray([f for f, pp in frame_pt.items() if pp == pos], int)
            rows = np.nonzero((obs_group == gid) & np.isin(o_frame, fset))[0]
            if rows.size == 0:
                continue
            assign, cons = consolidate_spots(
                o_q[rows], o_az[rows], o_int[rows], o_area[rows], o_frame[rows],
                q_tol=float(q_tol), azim_tol=azim_tol)
            obs_point[rows] = assign + len(points["group"])
            for s in cons:
                pt_rows.append(len(points["group"]))
                points["group"].append(gid)
                points["order"].append(pos)
                points["pressure"].append(float(axis_vals[pos])
                                          if ladder == "pressure" else np.nan)
                for k in ("q", "azim", "q_width", "azim_width", "intensity",
                          "area", "n_frames", "best_frame"):
                    points[k].append(s[k])
        rows_g = np.asarray(pt_rows, int)
        if rows_g.size == 0:
            continue
        gq = np.asarray([points["q"][r] for r in rows_g], float)
        gaz = np.asarray([points["azim"][r] for r in rows_g], float)
        gint = np.asarray([points["intensity"][r] for r in rows_g], float)
        gpos = np.asarray([points["order"][r] for r in rows_g], int)
        tracks = link_spot_tracks(
            gpos, gq, gaz, gint, axis_values=axis_vals, max_gap=max_gap,
            min_track_points=min_track_points, link_q_rel=link_q_rel,
            link_q_floor=3.0 * bin_r, link_azim_tol=link_azim_tol,
            axis_predictor=axis_predictor)
        for t in tracks:
            rows_t = rows_g[np.asarray(t["spots"], int)]
            all_tracks.append((gid, rows_t))

    pt_track = np.full(len(points["group"]), -1, int)
    for ti, (gid, rows_t) in enumerate(all_tracks):
        pt_track[rows_t] = ti
    if o_frame.size:
        obs_track = np.where(obs_point >= 0, pt_track[np.clip(obs_point, 0, None)], -1)

    # --- per-track summary (a track's points rows ARE its d(P) table)
    pt_pressure = np.asarray(points["pressure"], float)
    pt_q = np.asarray(points["q"], float)
    pt_az = np.asarray(points["azim"], float)
    pt_int = np.asarray(points["intensity"], float)
    pt_d = radial_to_d(pt_q, unit, wavelength) if pt_q.size else np.zeros(0)
    summaries: List[Dict[str, Any]] = []
    for ti, (gid, rows_t) in enumerate(all_tracks):
        pv = pt_pressure[rows_t]
        dv = pt_d[rows_t]
        w = pt_int[rows_t]
        if ladder == "pressure":
            k0 = int(np.argmin(pv))
            p_lo, p_hi = float(np.min(pv)), float(np.max(pv))
        else:
            k0, p_lo, p_hi = 0, np.nan, np.nan
        dd_dp = np.nan
        if ladder == "pressure" and np.unique(pv).size >= 2:
            A = np.column_stack([pv, np.ones(pv.size)])
            dd_dp = float(np.linalg.lstsq(A, dv, rcond=None)[0][0])
        az_mean = circ_mean(pt_az[rows_t], weights=w)
        az_spread = float(np.sqrt(np.sum(w * circ_diff(pt_az[rows_t], az_mean) ** 2)
                                  / np.sum(w)))
        member_obs = np.nonzero(obs_track == ti)[0]
        summaries.append({
            "track": ti, "group": gid, "group_label": group_names[gid],
            "n_points": int(rows_t.size),
            "n_frames": int(np.unique(o_frame[member_obs]).size),
            "p_min": p_lo, "p_max": p_hi,
            "q_first": float(pt_q[rows_t[0]]), "q_last": float(pt_q[rows_t[-1]]),
            "d0": float(dv[k0]), "d_min": float(np.min(dv)),
            "d_max": float(np.max(dv)), "dd_dp": dd_dp,
            "azim": az_mean, "azim_spread": az_spread,
            "intensity_max": float(np.max(w)),
            "best_frame": int(points["best_frame"][rows_t[int(np.argmax(w))]])})

    # --- per-group summary
    grp_nframes = np.bincount(frame_group, minlength=n_groups)
    grp_nobs = (np.bincount(obs_group, minlength=n_groups)
                if o_frame.size else np.zeros(n_groups, int))
    grp_ntracks = np.zeros(n_groups, int)
    for gid, _ in all_tracks:
        grp_ntracks[gid] += 1

    # --- atomic write of /spots to the target file
    params = {"schema_version": SCHEMA_VERSION,
              "source_reduced": str(src),
              "analysis_h5": str(ana) if ana else "",
              "unit": unit, "ladder": ladder,
              "pressure_source": pressure_source,
              "group_by": group_key,
              "min_snr": float(min_snr), "min_intensity": float(min_intensity),
              "min_pixels": int(min_pixels),
              "min_azim_samples": int(min_azim_samples),
              "max_azim_extent": float(max_azim_extent),
              "max_radial_extent": int(max_radial_extent),
              "q_min": float(q_min) if q_min is not None else np.nan,
              "q_max": float(q_max) if q_max is not None else np.nan,
              "exclude_diamond": bool(exclude_diamond),
              "diamond_a": float(diamond_a),
              "diamond_rel_tol": float(diamond_rel_tol),
              "diamond_max_compression": float(diamond_max_compression),
              "exclude_attributed": bool(attributed is not None),
              "exclude_fwhm_mult": float(exclude_fwhm_mult),
              "exclude_d": np.asarray(excl_d, dtype="f8"),
              "exclude_d_tol": float(exclude_d_tol),
              "exclude_d_compression": float(exclude_d_compression),
              "q_tol": float(q_tol), "azim_tol": float(azim_tol),
              "link_q_rel": float(link_q_rel),
              "link_azim_tol": float(link_azim_tol),
              "max_gap": int(max_gap),
              "min_track_points": int(min_track_points),
              "axis_predictor": bool(axis_predictor),
              "n_excluded_frames": int(n_excluded_arg),
              "n_obs": int(o_frame.size), "n_points": len(points["group"]),
              "n_tracks": len(all_tracks), "n_groups": int(n_groups)}

    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        shutil.copy2(dst, tmp)
        mode = "r+"
    else:
        mode = "w"
    try:
        with h5py.File(str(tmp), mode) as o:
            if mode == "w":
                o.attrs.update({"schema_version": SCHEMA_VERSION,
                                "source_reduced": str(src), "unit": unit,
                                "tool": "bulkxrd.analysis.spots"})
            if "spots" in o:
                del o["spots"]
            g = o.create_group("spots")
            g.attrs.update(params)
            go = g.create_group("obs")
            go.create_dataset("frame", data=o_frame.astype("i4"))
            go.create_dataset("group", data=obs_group.astype("i4"))
            go.create_dataset("point", data=obs_point.astype("i4"))
            go.create_dataset("track", data=obs_track.astype("i4"))
            go.create_dataset("pressure", data=(pressure[o_frame] if o_frame.size
                                                else np.zeros(0)).astype("f8"))
            go.create_dataset("d", data=(radial_to_d(o_q, unit, wavelength)
                                         if o_frame.size else np.zeros(0)).astype("f8"))
            for k in ("q", "azim", "intensity", "area", "snr",
                      "q_width", "azim_width"):
                go.create_dataset(k, data=np.asarray(obs[k], "f8"))
            go.create_dataset("n_pixels", data=np.asarray(obs["n_pixels"], "i4"))
            gg = g.create_group("groups")
            gg.create_dataset("id", data=np.arange(n_groups, dtype="i4"))
            gg.create_dataset("label", data=np.asarray(group_names, dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
            gg.create_dataset("n_frames", data=grp_nframes.astype("i4"))
            gg.create_dataset("n_obs", data=grp_nobs.astype("i4"))
            gg.create_dataset("n_tracks", data=grp_ntracks.astype("i4"))
            gp = g.create_group("points")
            gp.create_dataset("id", data=np.arange(len(points["group"]), dtype="i4"))
            gp.create_dataset("track", data=pt_track.astype("i4"))
            gp.create_dataset("d", data=pt_d.astype("f8"))
            for k, dt in (("group", "i4"), ("order", "i4"), ("pressure", "f8"),
                          ("q", "f8"), ("azim", "f8"), ("q_width", "f8"),
                          ("azim_width", "f8"), ("intensity", "f8"),
                          ("area", "f8"), ("n_frames", "i4"),
                          ("best_frame", "i4")):
                gp.create_dataset(k, data=np.asarray(points[k], dtype=dt))
            gt = g.create_group("tracks")
            gt.create_dataset("id", data=np.arange(len(all_tracks), dtype="i4"))
            for k, dt in (("group", "i4"), ("n_points", "i4"), ("n_frames", "i4"),
                          ("p_min", "f8"), ("p_max", "f8"),
                          ("q_first", "f8"), ("q_last", "f8"), ("d0", "f8"),
                          ("d_min", "f8"), ("d_max", "f8"), ("dd_dp", "f8"),
                          ("azim", "f8"), ("azim_spread", "f8"),
                          ("intensity_max", "f8"), ("best_frame", "i4")):
                gt.create_dataset(k, data=np.asarray(
                    [s[k] for s in summaries], dtype=dt))
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    manifest = {"tool_version": SCHEMA_VERSION, "out_h5": str(dst),
                "unit": unit, "ladder": ladder,
                "pressure_source": pressure_source, "group_by": group_key,
                "n_obs": int(o_frame.size), "n_points": len(points["group"]),
                "n_tracks": len(all_tracks), "n_groups": int(n_groups),
                "tracks": summaries}
    print(f"[SPOTS] {o_frame.size} detections -> {len(points['group'])} "
          f"pressure-point spots -> {len(all_tracks)} track(s) "
          f"({ladder} ladder, group_by={group_key}, "
          f"pressure_source={pressure_source}) -> {dst}", flush=True)
    for s in summaries:
        print(f"[SPOTS]   track {s['track']}: azim {s['azim']:+7.1f}deg, "
              f"{s['n_points']} P-points ({s['n_frames']} frames), "
              f"P {s['p_min']:.1f}-{s['p_max']:.1f} GPa, d0 {s['d0']:.4f} A, "
              f"dd/dP {s['dd_dp']:+.5f} A/GPa, Imax {s['intensity_max']:.0f}",
              flush=True)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_matches(d0, azims, table: Dict[str, Any], *, rel_tol: float,
                   top: int) -> None:
    matches = match_tracks(d0, table, rel_tol=rel_tol, top=top)
    print(f"[SPOTS] matching {len(d0)} track d0's against {table['d'].size} "
          f"calculated reflections (rel_tol={rel_tol}):", flush=True)
    for ti, (d, az, mm) in enumerate(zip(d0, azims, matches)):
        if not mm:
            print(f"[SPOTS]   track {ti}: d0 {d:.4f} A (azim {az:+.1f}deg) "
                  "-> no match", flush=True)
            continue
        parts = []
        for m in mm:
            hkl = "".join(str(v) for v in m["hkl"]) if m["hkl"] else "?"
            it = f" I={m['intensity']:.1f}" if m["intensity"] is not None else ""
            parts.append(f"({hkl}) d={m['d_calc']:.4f} "
                         f"delta={100 * m['delta_rel']:+.2f}%{it}")
        print(f"[SPOTS]   track {ti}: d0 {d:.4f} A (azim {az:+.1f}deg) -> "
              + "; ".join(parts), flush=True)


def _parse_frame_list(spec: "Optional[str]") -> "Optional[List[int]]":
    """CLI frame-list: comma-separated indices, or '@file' with one per line."""
    if not spec:
        return None
    spec = spec.strip()
    if spec.startswith("@"):
        text = Path(spec[1:]).expanduser().read_text(encoding="utf-8")
        spec = ",".join(text.split())
    return [int(v) for v in spec.replace(",", " ").split()]


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="bulkxrd-spots",
        description="Cake-space single-crystal spot tracker: detect azimuthal "
                    "blobs in /cakes, consolidate them per pressure point, "
                    "link the points across the pressure ladder, append "
                    "/spots, and optionally match track d0's against a "
                    "calculated reflection list.")
    ap.add_argument("reduced", help="reduced HDF5 with /cakes (with --match-only: "
                                    "the file holding /spots)")
    ap.add_argument("--analysis", help="analysis HDF5 — supplies /frames/pressure "
                                       "and /peaks/phase exclusion; /spots is "
                                       "appended here when given")
    ap.add_argument("-o", "--out", help="output HDF5 (default: the analysis file, "
                                        "else <reduced>_spots.h5)")
    det = ap.add_argument_group("detection")
    det.add_argument("--min-snr", type=float, default=6.0)
    det.add_argument("--min-intensity", type=float, default=20.0,
                     help="absolute excess-counts floor (default 20)")
    det.add_argument("--min-pixels", type=int, default=2)
    det.add_argument("--min-azim-samples", type=int, default=30)
    det.add_argument("--max-azim-extent", type=float, default=45.0,
                     help="degrees; wider components are textured arcs, not spots")
    det.add_argument("--max-radial-extent", type=int, default=30,
                     help="radial bins; longer components are streak artifacts")
    det.add_argument("--q-min", type=float, default=None)
    det.add_argument("--q-max", type=float, default=None)
    exc = ap.add_argument_group("exclusions")
    exc.add_argument("--no-diamond", action="store_true",
                     help="keep spots on the diamond-anvil lines")
    exc.add_argument("--diamond-a", type=float, default=DIAMOND_A0)
    exc.add_argument("--diamond-tol", type=float, default=0.02)
    exc.add_argument("--diamond-compression", type=float, default=0.04)
    exc.add_argument("--no-attributed", action="store_true",
                     help="do not exclude Step-3a-attributed powder peaks")
    exc.add_argument("--exclude-fwhm-mult", type=float, default=2.0)
    exc.add_argument("--exclude-d",
                     help="comma-separated d-lines (Å) to exclude — e.g. the "
                          "gasket material's strongest reflections, whose "
                          "coarse grains make off-ring spots the attributed-"
                          "peak windows miss")
    exc.add_argument("--exclude-d-tol", type=float, default=0.02)
    exc.add_argument("--exclude-d-compression", type=float, default=0.06,
                     help="extra upward-in-q window growth for compression "
                          "of the excluded lines (default 0.06)")
    lnk = ap.add_argument_group("consolidation + linking")
    lnk.add_argument("--group-by", choices=GROUP_MODES, default="none",
                     help="'none' = one global pressure ladder (default); "
                          "'scan' = independent ladder per scanNNN group")
    lnk.add_argument("--q-tol", type=float, default=None,
                     help="within-pressure-point merge tolerance "
                          "(default 3 radial bins)")
    lnk.add_argument("--azim-tol", type=float, default=6.0)
    lnk.add_argument("--link-q-rel", type=float, default=0.05)
    lnk.add_argument("--link-azim-tol", type=float, default=8.0)
    lnk.add_argument("--max-gap", type=int, default=3,
                     help="max missing consecutive pressure points inside a "
                          "track (Ewald visibility bands; default 3)")
    lnk.add_argument("--min-track-points", type=int, default=3)
    lnk.add_argument("--exclude-frames", metavar="LIST",
                     help="frame indices to drop from detection: a comma list "
                          "(e.g. '3,7,12') or @FILE with one index per line — "
                          "for known-bad exposures (cover left on, etc.)")
    mat = ap.add_argument_group("matching")
    mat.add_argument("--match", help="calculated reflection file (d/I pairs or an "
                                     "'h k l d ... I' whitespace table)")
    mat.add_argument("--match-tol", type=float, default=0.03,
                     help="relative |delta d|/d tolerance (default 0.03)")
    mat.add_argument("--match-top", type=int, default=3)
    mat.add_argument("--match-only", action="store_true",
                     help="skip detection; read /spots from the given file and "
                          "just run the matcher")
    exp = ap.add_argument_group("export")
    exp.add_argument("--export", metavar="DIR",
                     help="write the group-handoff CSV bundle (spot_tracks.csv + "
                          "d(P) point tables + README) to DIR after tracking; "
                          "with --match-only, exports the existing /spots")
    exp.add_argument("--export-observations", action="store_true",
                     help="include every raw per-frame detection in the export "
                          "(spot_observations.csv)")
    exp.add_argument("--export-cakes", metavar="FRAMES",
                     help="comma-separated frame indices, or 'best' (each "
                          "track's best_frame): write those frames' cakes with "
                          "the powder rings removed (PNG + .npy) into "
                          "<--export DIR>/ringless (or next to the input)")
    args = ap.parse_args(argv)

    def _do_export_cakes(spots_path: Path, best_frames: "Sequence[int]",
                         reduced_path: "Optional[Path]" = None) -> None:
        """Resolve --export-cakes (frame list or 'best') and run the export."""
        spec = (args.export_cakes or "").strip().lower()
        if not spec:
            return
        frames = (sorted(set(int(f) for f in best_frames)) if spec == "best"
                  else [int(v) for v in spec.split(",") if v.strip()])
        if not frames:
            print("[SPOTS] --export-cakes: no frames to export.", flush=True)
            return
        red = reduced_path
        if red is None or not red.is_file():
            import h5py  # type: ignore
            with h5py.File(str(spots_path), "r") as h:
                red = Path(str(h["spots"].attrs.get("source_reduced", "")))
        if not red.is_file():
            print(f"[ERROR] --export-cakes: reduced file with /cakes not found "
                  f"({red}).", flush=True)
            return
        dest = (Path(args.export) / "ringless" if args.export
                else spots_path.with_name(spots_path.stem + "_ringless"))
        export_ring_removed_cakes(red, dest, frames)

    if args.match_only:
        if not (args.match or args.export or args.export_cakes):
            print("[ERROR] --match-only needs --match FILE, --export DIR "
                  "and/or --export-cakes.", flush=True)
            return 1
        import h5py  # type: ignore
        path = Path(args.reduced).expanduser().resolve()
        with h5py.File(str(path), "r") as h:
            if "spots" not in h or "tracks" not in h["spots"]:
                print(f"[ERROR] no /spots/tracks in {path} — run bulkxrd-spots "
                      "without --match-only first.", flush=True)
                return 1
            d0 = np.asarray(h["spots/tracks/d0"][:], float)
            az = np.asarray(h["spots/tracks/azim"][:], float)
            best = np.asarray(h["spots/tracks/best_frame"][:], int)
        if args.match:
            _print_matches(d0, az, load_reflection_table(args.match),
                           rel_tol=args.match_tol, top=args.match_top)
        if args.export:
            export_spot_tracks(path, args.export, match=args.match,
                               match_tol=args.match_tol,
                               include_observations=args.export_observations)
        _do_export_cakes(path, best.tolist())
        return 0

    try:
        manifest = run_spot_tracking(
            args.reduced, args.analysis, out_h5=args.out,
            min_snr=args.min_snr, min_intensity=args.min_intensity,
            min_pixels=args.min_pixels, min_azim_samples=args.min_azim_samples,
            max_azim_extent=args.max_azim_extent,
            max_radial_extent=args.max_radial_extent,
            q_min=args.q_min, q_max=args.q_max,
            exclude_diamond=not args.no_diamond, diamond_a=args.diamond_a,
            diamond_rel_tol=args.diamond_tol,
            diamond_max_compression=args.diamond_compression,
            exclude_attributed=not args.no_attributed,
            exclude_fwhm_mult=args.exclude_fwhm_mult,
            exclude_d=[float(v) for v in args.exclude_d.split(",") if v.strip()]
            if args.exclude_d else None,
            exclude_d_tol=args.exclude_d_tol,
            exclude_d_compression=args.exclude_d_compression,
            group_by=args.group_by, q_tol=args.q_tol, azim_tol=args.azim_tol,
            link_q_rel=args.link_q_rel, link_azim_tol=args.link_azim_tol,
            max_gap=args.max_gap, min_track_points=args.min_track_points,
            exclude_frames=_parse_frame_list(args.exclude_frames))
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", flush=True)
        return 1

    if args.match:
        d0 = np.asarray([s["d0"] for s in manifest["tracks"]], float)
        az = np.asarray([s["azim"] for s in manifest["tracks"]], float)
        _print_matches(d0, az, load_reflection_table(args.match),
                       rel_tol=args.match_tol, top=args.match_top)
    if args.export:
        export_spot_tracks(manifest["out_h5"], args.export, match=args.match,
                           match_tol=args.match_tol,
                           include_observations=args.export_observations)
    _do_export_cakes(Path(manifest["out_h5"]),
                     [int(s["best_frame"]) for s in manifest["tracks"]],
                     reduced_path=Path(args.reduced).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
