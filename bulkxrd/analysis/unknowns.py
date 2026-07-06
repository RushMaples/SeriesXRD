"""Step 3c — unknown-phase detection by co-occurrence clustering.

After Step 3a subtracts the identified phases, ``/residual/peaks`` holds the
re-fitted peaks nothing known explains. Real undiscovered phases do not appear
as isolated blips: their reflections form COHERENT TRACKS across the pressure
series (drifting smoothly as the lattice compresses) and those tracks appear,
disappear and move TOGETHER. This module turns that physics into the final
pipeline stage:

  1. **Track linking** — residual peaks are chained frame-to-frame into tracks
     (greedy one-to-one nearest-center linking with a width-scaled tolerance
     and a small frame-gap allowance, mirroring Step 2's seed propagation).
     Tracks seen in too few frames are noise and dropped.
  2. **Co-occurrence clustering** — tracks whose presence/absence patterns
     agree (Jaccard similarity of the frame sets) are merged single-link into
     clusters: "these reflections belong to the same unknown substance".
  3. **Fingerprinting** — each cluster reports its d-spacings at a reference
     frame (the frame where most of its tracks coexist), the set to search
     against COD/ICSD/MP or to flag as genuinely new. Appearance/disappearance
     frames of a cluster are phase-transition candidates.

Appends ``/unknowns`` to the analysis HDF5. Pure numpy + h5py; requires
``/residual`` (Step 3a) to exist. Consumed by the heatmap (per-cluster layers)
and by whoever hunts the unknowns.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .identify import radial_to_d

SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Track linking
# ---------------------------------------------------------------------------

def link_tracks(frames: np.ndarray, centers: np.ndarray, amplitudes: np.ndarray,
                fwhms: np.ndarray, *, n_frames: int,
                link_tol_fwhm: float = 1.5, max_gap: int = 2,
                min_track_frames: int = 3) -> "List[Dict[str, np.ndarray]]":
    """Chain per-frame residual peaks into cross-frame tracks.

    Greedy one-to-one linking, closest pair first: a peak joins an open track
    when it is within ``link_tol_fwhm × max(track width, peak width)`` of the
    track's last center and the track was last seen ≤ ``max_gap`` frames ago
    (weak reflections legitimately dip below SNR for a frame or two). Tracks
    observed in fewer than ``min_track_frames`` frames are discarded as noise.
    Returns one dict per kept track: ``{frames, centers, amplitudes, fwhms}``.
    """
    frames = np.asarray(frames, int)
    centers = np.asarray(centers, float)
    amplitudes = np.asarray(amplitudes, float)
    fwhms = np.asarray(fwhms, float)

    open_tracks: List[Dict[str, list]] = []
    done: List[Dict[str, list]] = []
    for fi in range(int(n_frames)):
        rows = np.nonzero(frames == fi)[0]
        # Retire tracks that fell too far behind. max_gap counts MISSING
        # frames: a track last seen at frame f may still link at frame
        # f + max_gap + 1.
        still: List[Dict[str, list]] = []
        for t in open_tracks:
            if fi - t["frames"][-1] > int(max_gap) + 1:
                done.append(t)
            else:
                still.append(t)
        open_tracks = still
        if rows.size == 0:
            continue
        # Candidate (gap, track, row) pairs within tolerance, closest first.
        cands: List[Tuple[float, int, int]] = []
        for ti, t in enumerate(open_tracks):
            c_last = t["centers"][-1]
            w_last = t["fwhms"][-1]
            for r in rows:
                tol = float(link_tol_fwhm) * max(w_last, fwhms[r], 1e-9)
                g = abs(centers[r] - c_last)
                if g <= tol:
                    cands.append((g, ti, int(r)))
        cands.sort(key=lambda x: x[0])
        used_t: set = set()
        used_r: set = set()
        for g, ti, r in cands:
            if ti in used_t or r in used_r:
                continue
            used_t.add(ti)
            used_r.add(r)
            t = open_tracks[ti]
            t["frames"].append(fi)
            t["centers"].append(float(centers[r]))
            t["amplitudes"].append(float(amplitudes[r]))
            t["fwhms"].append(float(fwhms[r]))
        for r in rows:
            if int(r) not in used_r:                 # unmatched peak → new track
                open_tracks.append({"frames": [fi],
                                    "centers": [float(centers[r])],
                                    "amplitudes": [float(amplitudes[r])],
                                    "fwhms": [float(fwhms[r])]})
    done.extend(open_tracks)
    kept = [{k: np.asarray(v) for k, v in t.items()}
            for t in done if len(t["frames"]) >= int(min_track_frames)]
    kept.sort(key=lambda t: -float(np.sum(t["amplitudes"])))
    return kept


# ---------------------------------------------------------------------------
# Co-occurrence clustering
# ---------------------------------------------------------------------------

def cluster_tracks(tracks: "List[Dict[str, np.ndarray]]", *,
                   jaccard_threshold: float = 0.6) -> np.ndarray:
    """Cluster ids (0-based) by single-link merging of tracks whose frame sets
    have Jaccard similarity ≥ ``jaccard_threshold`` — reflections of one phase
    appear and vanish together across the series."""
    n = len(tracks)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sets = [set(int(f) for f in t["frames"]) for t in tracks]
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            if union and inter / union >= float(jaccard_threshold):
                parent[find(i)] = find(j)
    roots: Dict[int, int] = {}
    out = np.zeros(n, int)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        out[i] = roots[r]
    return out


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------

def run_unknowns(
    analysis_h5: "str | Path",
    *,
    link_tol_fwhm: float = 1.5,
    max_gap: int = 2,
    min_track_frames: int = 3,
    jaccard_threshold: float = 0.6,
    out_h5: "Optional[str | Path]" = None,
) -> Dict[str, Any]:
    """Cluster the Step-3a residual peaks into candidate unknown phases and
    append ``/unknowns`` to the analysis HDF5.

        /unknowns  attrs: schema_version, link_tol_fwhm, max_gap,
                          min_track_frames, jaccard_threshold,
                          n_tracks, n_clusters
        /unknowns/obs/{track,frame,center,amplitude,fwhm}  (M,)  every kept
                          track observation (flat; ``track`` indexes tracks)
        /unknowns/tracks/{id,cluster,n_frames,first_frame,last_frame,
                          center_first,center_last}          (T,)
        /unknowns/clusters/{id,n_tracks,first_frame,last_frame,ref_frame} (C,)
        /unknowns/fingerprint/{cluster,d}                     (F,)  per-cluster
                          d-spacings at its reference frame — the set to search
                          against external databases

    The cluster boundaries (first/last frame) are phase-transition candidates.
    Returns a manifest with a per-cluster summary.
    """
    import h5py  # type: ignore

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    dst = Path(out_h5).expanduser().resolve() if out_h5 else src

    with h5py.File(str(src), "r") as h5:
        rg = h5.get("residual")
        if rg is None or "peaks" not in rg:
            raise ValueError("No /residual/peaks — run Step 3a (+ residual) first.")
        gp = rg["peaks"]
        frames = np.asarray(gp["frame"][:], int)
        centers = np.asarray(gp["center"][:], float)
        amps = np.asarray(gp["amplitude"][:], float)
        fwhms = np.asarray(gp["fwhm"][:], float)
        n_frames = int(np.asarray(gp["counts"][:]).size)
        unit = str(h5.attrs.get("unit", ""))
        wavelength = float(h5.attrs.get("wavelength", 0.0) or 0.0) or None

    tracks = link_tracks(frames, centers, amps, fwhms, n_frames=n_frames,
                         link_tol_fwhm=link_tol_fwhm, max_gap=max_gap,
                         min_track_frames=min_track_frames)
    cluster_of = (cluster_tracks(tracks, jaccard_threshold=jaccard_threshold)
                  if tracks else np.zeros(0, int))
    n_clusters = int(cluster_of.max()) + 1 if tracks else 0

    # Per-cluster reference frame (most member tracks present) + fingerprint.
    fp_cluster: List[int] = []
    fp_d: List[float] = []
    summaries: List[Dict[str, Any]] = []
    for c in range(n_clusters):
        members = [t for t, cl in zip(tracks, cluster_of) if cl == c]
        count = np.zeros(n_frames, int)
        for t in members:
            count[np.asarray(t["frames"], int)] += 1
        ref = int(np.argmax(count))
        ds: List[float] = []
        for t in members:
            hit = np.nonzero(np.asarray(t["frames"], int) == ref)[0]
            if hit.size:
                ds.append(float(t["centers"][hit[0]]))
        d_vals = sorted(radial_to_d(np.asarray(ds, float), unit, wavelength).tolist(),
                        reverse=True) if ds and unit else sorted(ds, reverse=True)
        fp_cluster += [c] * len(d_vals)
        fp_d += d_vals
        lo = min(int(t["frames"].min()) for t in members)
        hi = max(int(t["frames"].max()) for t in members)
        summaries.append({"cluster": c, "n_tracks": len(members),
                          "first_frame": lo, "last_frame": hi,
                          "ref_frame": ref,
                          "d_fingerprint": [round(v, 4) for v in d_vals]})

    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            if "unknowns" in o:
                del o["unknowns"]
            g = o.create_group("unknowns")
            g.attrs.update({"schema_version": SCHEMA_VERSION,
                            "link_tol_fwhm": float(link_tol_fwhm),
                            "max_gap": int(max_gap),
                            "min_track_frames": int(min_track_frames),
                            "jaccard_threshold": float(jaccard_threshold),
                            "n_tracks": len(tracks),
                            "n_clusters": n_clusters})
            go = g.create_group("obs")
            flat = {"track": [], "frame": [], "center": [],
                    "amplitude": [], "fwhm": []}
            for ti, t in enumerate(tracks):
                for j in range(t["frames"].size):
                    flat["track"].append(ti)
                    flat["frame"].append(int(t["frames"][j]))
                    flat["center"].append(float(t["centers"][j]))
                    flat["amplitude"].append(float(t["amplitudes"][j]))
                    flat["fwhm"].append(float(t["fwhms"][j]))
            for k, v in flat.items():
                go.create_dataset(k, data=np.asarray(
                    v, dtype=("i4" if k in ("track", "frame") else "f8")))
            gt = g.create_group("tracks")
            gt.create_dataset("id", data=np.arange(len(tracks), dtype="i4"))
            gt.create_dataset("cluster", data=cluster_of.astype("i4"))
            gt.create_dataset("n_frames",
                              data=np.array([t["frames"].size for t in tracks], "i4"))
            gt.create_dataset("first_frame",
                              data=np.array([t["frames"].min() if t["frames"].size else -1
                                             for t in tracks], "i4"))
            gt.create_dataset("last_frame",
                              data=np.array([t["frames"].max() if t["frames"].size else -1
                                             for t in tracks], "i4"))
            gt.create_dataset("center_first",
                              data=np.array([t["centers"][0] for t in tracks], "f8"))
            gt.create_dataset("center_last",
                              data=np.array([t["centers"][-1] for t in tracks], "f8"))
            gc = g.create_group("clusters")
            gc.create_dataset("id", data=np.arange(n_clusters, dtype="i4"))
            gc.create_dataset("n_tracks",
                              data=np.array([s["n_tracks"] for s in summaries], "i4"))
            gc.create_dataset("first_frame",
                              data=np.array([s["first_frame"] for s in summaries], "i4"))
            gc.create_dataset("last_frame",
                              data=np.array([s["last_frame"] for s in summaries], "i4"))
            gc.create_dataset("ref_frame",
                              data=np.array([s["ref_frame"] for s in summaries], "i4"))
            gf = g.create_group("fingerprint")
            gf.create_dataset("cluster", data=np.asarray(fp_cluster, "i4"))
            gf.create_dataset("d", data=np.asarray(fp_d, "f8"))
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    manifest = {"tool_version": SCHEMA_VERSION, "out_h5": str(dst),
                "n_residual_peaks": int(frames.size),
                "n_tracks": len(tracks), "n_clusters": n_clusters,
                "clusters": summaries}
    print(f"[UNKNOWNS] {frames.size} residual peaks -> {len(tracks)} coherent "
          f"track(s) -> {n_clusters} cluster(s) -> {dst}", flush=True)
    for s in summaries:
        print(f"[UNKNOWNS]   cluster {s['cluster']}: {s['n_tracks']} tracks, "
              f"frames {s['first_frame']}-{s['last_frame']}, "
              f"d = {s['d_fingerprint']}", flush=True)
    return manifest
