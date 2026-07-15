"""Cake-space single-crystal spot tracker (spots.py): synthetic cakes with
drifting blobs + a powder ring -> the tracker finds the blobs, consolidates
them per pressure point, links them across the pressure ladder, and ignores
the ring, the diamond line, and single-pixel zingers."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.analysis.spots import (
    circ_diff, circ_mean, diamond_q_lines, diamond_q_windows, detect_spots,
    consolidate_spots, link_spot_tracks, run_spot_tracking,
    load_reflection_table, match_tracks, export_spot_tracks, load_spot_tracks,
    export_ring_removed_cakes, DIAMOND_A0)

N_AZ, N_RAD = 120, 240
RADIAL = np.linspace(0.5, 6.0, N_RAD)          # q (A^-1)
AZIM = np.linspace(-180.0, 177.0, N_AZ)        # 3 deg bins
BIN_R = float(RADIAL[1] - RADIAL[0])

# Mirrors the UOTe mass-scan layout: one scanNNN = one beam position, and its
# frames step through the pressure ladder (scan001\P01_..._1p0GPa_...).
N_SCANS = 3
LADDER = [1.0, 4.0, 7.0, 10.0]
TOKENS = ["1p0GPa", "4p0GPa", "7p0GPa", "10p0GPa"]

# Blob A compresses normally (q grows with P); blob B is the NLC analog
# (q shrinks -> d grows). The diamond line stays put and must be excluded.
AZ_A, AZ_B = 40.0, -120.0


def q_a(p):
    return 2.00 + 0.010 * p


def q_b(p):
    return 3.60 - 0.008 * p


Q_DIAMOND = float(2.0 * np.pi * np.sqrt(3) / DIAMOND_A0)   # 111 line ~3.051
RING_Q = 2.60


def _add_blob(cake, q0, az0, amp, *, q_sig=1.2 * BIN_R, az_sig=4.0):
    dq = (RADIAL[None, :] - q0) / q_sig
    daz = circ_diff(AZIM, az0)[:, None] / az_sig
    cake += amp * np.exp(-0.5 * (dq ** 2 + daz ** 2))


def _make_cake(rng, scan, step):
    p = LADDER[step]
    cake = 10.0 + rng.normal(0.0, 2.0, (N_AZ, N_RAD))
    # Powder ring: present at EVERY azimuth -> azimuthal median removes it.
    cake += 80.0 * np.exp(-0.5 * ((RADIAL[None, :] - RING_Q) / (1.5 * BIN_R)) ** 2)
    _add_blob(cake, q_a(p), AZ_A, 300.0)
    # Blob B vanishes at one mid-ladder step of the last scan (per-scan gap;
    # the global ladder still sees it there via the other scans).
    if not (scan == N_SCANS - 1 and step == 2):
        _add_blob(cake, q_b(p), AZ_B, 200.0)
    # Diamond spot: every frame, fixed q/azim, very bright.
    _add_blob(cake, Q_DIAMOND, 70.0, 5000.0)
    # Zinger: one hot pixel in one frame only.
    if scan == 1 and step == 1:
        cake[10, 30] += 1000.0
    return cake


def _build_dataset():
    rng = np.random.default_rng(7)
    cakes, names = [], []
    for s in range(N_SCANS):
        for j in range(len(LADDER)):
            cakes.append(_make_cake(rng, s, j))
            names.append(f"scan{s + 1:03d}/P{j + 1:02d}_scan{s + 1:03d}_"
                         f"UOTe_{TOKENS[j]}_P1_{j:03d}.tif")
    return np.asarray(cakes, dtype=np.float32), names


def _write_reduced(path, cakes, names):
    import h5py
    n = cakes.shape[0]
    with h5py.File(path, "w") as h:
        h.attrs["unit"] = "q_A^-1"
        h.attrs["schema_version"] = "1"
        cg = h.create_group("cakes")
        cg.create_dataset("intensity", data=cakes)
        cg.create_dataset("radial", data=RADIAL)
        cg.create_dataset("azimuthal", data=AZIM)
        cg.create_dataset("frame_index", data=np.arange(n))
        fr = h.create_group("frames")
        fr.create_dataset("filename", data=np.asarray(names, dtype=object),
                          dtype=h5py.string_dtype(encoding="utf-8"))
        fr.create_dataset("excluded", data=np.zeros(n, bool))
        fr.create_dataset("ok", data=np.ones(n, bool))
        fr.create_dataset("pressure", data=np.full(n, np.nan))


def _test_helpers():
    assert abs(circ_diff(179.0, -179.0) - 2.0) < 1e-9
    assert abs(circ_mean([179.0, -179.0])) - 180.0 < 1e-6
    # Diamond selection rules: 111/220/311 allowed, 200/210 forbidden.
    q = diamond_q_lines(7.2)
    d = 2.0 * np.pi / q
    for d_ok in (2.0595, 1.2612, 1.0754):
        assert np.any(np.abs(d - d_ok) < 5e-3), (d_ok, d)
    assert not np.any(np.abs(d - DIAMOND_A0 / 2.0) < 5e-3), d      # (200)
    wins = diamond_q_windows(7.2)
    assert all(hi > lo for lo, hi in wins)
    # window is asymmetric: reaches further up (compression) than down.
    w111 = min(wins, key=lambda w: abs(0.5 * (w[0] + w[1]) - Q_DIAMOND))
    assert (w111[1] - Q_DIAMOND) > (Q_DIAMOND - w111[0])


def _test_detection():
    rng = np.random.default_rng(3)
    cake = _make_cake(rng, 0, 0)
    p0 = LADDER[0]
    spots = detect_spots(cake, RADIAL, AZIM, min_snr=6.0, min_intensity=20.0)
    qs = [s["q"] for s in spots]
    # Diamond, A and B found when no exclusions; the ring never is.
    assert any(abs(q - Q_DIAMOND) < 0.05 for q in qs), qs
    assert any(abs(q - q_a(p0)) < 0.05 for q in qs), qs
    assert any(abs(q - q_b(p0)) < 0.05 for q in qs), qs
    assert not any(abs(q - RING_Q) < 0.05 for q in qs), qs
    # Diamond window exclusion drops exactly the diamond spot.
    spots2 = detect_spots(cake, RADIAL, AZIM, min_snr=6.0, min_intensity=20.0,
                          exclude_q_windows=diamond_q_windows(RADIAL[-1]))
    qs2 = [s["q"] for s in spots2]
    assert not any(abs(q - Q_DIAMOND) < 0.05 for q in qs2), qs2
    assert any(abs(q - q_a(p0)) < 0.05 for q in qs2), qs2
    # Attributed-powder-peak exclusion drops a blob at the attributed q.
    spots3 = detect_spots(cake, RADIAL, AZIM, min_snr=6.0, min_intensity=20.0,
                          exclude_peaks=np.array([[q_a(p0), 2 * BIN_R]]))
    assert not any(abs(s["q"] - q_a(p0)) < 0.05 for s in spots3)
    # Zinger (single pixel) is dropped by min_pixels.
    z = 10.0 + rng.normal(0.0, 2.0, (N_AZ, N_RAD))
    z[10, 30] += 1000.0
    assert detect_spots(z, RADIAL, AZIM, min_pixels=2) == []
    zs = detect_spots(z, RADIAL, AZIM, min_pixels=1)
    assert len(zs) == 1 and zs[0]["n_pixels"] == 1
    # Azimuth wrap: a blob at the +/-180 deg seam is ONE spot, centroid ~180.
    wcake = 10.0 + rng.normal(0.0, 2.0, (N_AZ, N_RAD))
    _add_blob(wcake, 2.5, 180.0, 400.0)
    ws = [s for s in detect_spots(wcake, RADIAL, AZIM) if abs(s["q"] - 2.5) < 0.1]
    assert len(ws) == 1, ws
    assert circ_diff(ws[0]["azim"], 180.0) < 3.0, ws[0]["azim"]
    # A textured arc (wide in azimuth) is rejected as not-a-spot.
    acake = 10.0 + rng.normal(0.0, 2.0, (N_AZ, N_RAD))
    _add_blob(acake, 2.5, 0.0, 400.0, az_sig=40.0)
    assert all(abs(s["q"] - 2.5) > 0.1
               for s in detect_spots(acake, RADIAL, AZIM, max_azim_extent=45.0))


def _test_consolidation():
    # Two reflections seen from 2 beam positions at ONE pressure -> 2 spots.
    q = np.array([2.00, 2.01, 3.00, 2.995])
    az = np.array([40.0, 41.0, -120.0, -119.0])
    inten = np.array([300.0, 280.0, 200.0, 190.0])
    area = inten * 2
    frame = np.array([0, 1, 0, 1])
    assign, spots = consolidate_spots(q, az, inten, area, frame,
                                      q_tol=0.05, azim_tol=6.0)
    assert len(spots) == 2 and spots[0]["n_frames"] == 2
    assert assign[0] == assign[1] and assign[2] == assign[3]
    assert assign[0] != assign[2]
    # Same q but far azimuth stays separate (joint (q, azim) merge).
    assign2, spots2 = consolidate_spots(
        np.array([2.0, 2.0]), np.array([40.0, 150.0]),
        np.array([10.0, 9.0]), np.array([20.0, 18.0]), np.array([0, 1]),
        q_tol=0.05, azim_tol=6.0)
    assert len(spots2) == 2


def _test_linking():
    # Two reflections drifting in q at fixed azimuth -> one track each; a spot
    # at a different azimuth but matching q must NOT join (joint linking), and
    # a one-step gap is bridged (Ewald visibility bands).
    pos = np.array([0, 0, 1, 1, 2, 3, 3, 1])
    qq = np.array([2.00, 3.60, 2.04, 3.57, 2.07, 2.10, 3.52, 2.04])
    aa = np.array([40.0, -120.0, 40.5, -120.5, 41.0, 40.5, -119.5, 150.0])
    ii = np.array([300.0, 200.0, 290.0, 210.0, 280.0, 270.0, 205.0, 250.0])
    tracks = link_spot_tracks(pos, qq, aa, ii,
                              axis_values=np.array([1.0, 4.0, 7.0, 10.0]),
                              min_track_points=3, link_q_rel=0.05,
                              link_azim_tol=8.0, link_q_floor=0.02, max_gap=2)
    assert len(tracks) == 2, [t["centers"] for t in tracks]
    ta = next(t for t in tracks if circ_diff(circ_mean(t["azims"]), 40.0) < 3)
    tb = next(t for t in tracks if circ_diff(circ_mean(t["azims"]), -120.0) < 3)
    assert ta["spots"].size == 4
    # Track B skips ladder position 2 but still spans the whole ladder.
    assert tb["spots"].size == 3 and tb["axis"][-1] == 10.0, tb
    # The azim-150 imposter at matching q was never absorbed.
    assert 7 not in set(int(r) for r in ta["spots"]), ta["spots"]


def _test_matcher_parsing(tmp: Path):
    two = tmp / "two_col.txt"
    two.write_text("# d I\n7.4910 4.2\n3.7455 7.6\n2.8313 62.9\n")
    t2 = load_reflection_table(two)
    assert t2["hkl"] is None and t2["d"].size == 3

    # The user's format: header with unit token '(Å)' + h k l d ... I columns.
    hkl = tmp / "hkl_table.txt"
    hkl.write_text(
        "   h    k    l      d (Å)      F(real)      F(imag)"
        "          |F|         2θ          I    M ID(λ) Phase\n"
        "   0    0    1   7.491000    30.58     5.27      31.03"
        "    3.16    4.22    2     1     1\n"
        "   1    1    0   2.831256  -224.31   -14.41     224.77"
        "    8.37   62.85    4     1     1\n"
        "   1    0    2   2.735292   -12.09   207.21     207.56"
        "    8.67  100.00    8     1     1\n", encoding="utf-8")
    th = load_reflection_table(hkl)
    assert th["hkl"] is not None and tuple(th["hkl"][0]) == (0, 0, 1)
    assert abs(th["d"][1] - 2.831256) < 1e-9
    assert abs(th["intensity"][2] - 100.0) < 1e-9

    m = match_tracks([7.52, 1.0], th, rel_tol=0.03)
    assert m[0] and m[0][0]["hkl"] == (0, 0, 1)
    assert m[1] == []

    # Headerless hkl table: integer-triple heuristic + normalized-I column.
    nh = tmp / "no_header.txt"
    nh.write_text("0 0 1 7.4910 31.03 4.22\n1 1 0 2.8313 224.77 62.85\n"
                  "1 0 2 2.7353 207.56 100.00\n")
    tn = load_reflection_table(nh)
    assert tn["hkl"] is not None and abs(tn["intensity"][2] - 100.0) < 1e-9


def _test_end_to_end(tmp: Path):
    import h5py
    cakes, names = _build_dataset()
    reduced = tmp / "reduced.h5"
    _write_reduced(reduced, cakes, names)

    # Default: one global pressure ladder; positions consolidate per point.
    m = run_spot_tracking(reduced, min_snr=6.0, min_intensity=20.0,
                          min_track_points=3)
    out = Path(m["out_h5"])
    assert out.name == "reduced_spots.h5" and out.is_file()
    assert m["pressure_source"] == "filename" and m["ladder"] == "pressure"
    assert m["group_by"] == "none"
    assert m["n_tracks"] == 2, [
        (round(s["azim"], 1), round(s["d0"], 3)) for s in m["tracks"]]
    ta = next(s for s in m["tracks"] if circ_diff(s["azim"], AZ_A) < 3.0)
    tb = next(s for s in m["tracks"] if circ_diff(s["azim"], AZ_B) < 3.0)
    # A is seen at all 4 pressure points from all 3 positions; B misses one
    # frame but its pressure point survives via the other two positions.
    assert ta["n_points"] == 4 and ta["n_frames"] == 12, ta
    assert tb["n_points"] == 4 and tb["n_frames"] == 11, tb
    for s, qfun in ((ta, q_a), (tb, q_b)):
        assert s["p_min"] == 1.0 and s["p_max"] == 10.0, s
        assert abs(s["d0"] - 2.0 * np.pi / qfun(1.0)) < 0.02, s
    assert ta["dd_dp"] < -0.001 < 0.001 < tb["dd_dp"], (ta["dd_dp"], tb["dd_dp"])

    with h5py.File(out, "r") as h:
        g = h["spots"]
        assert g.attrs["n_tracks"] == 2 and g.attrs["ladder"] == "pressure"
        assert [v.decode() for v in g["groups/label"][:]] == ["all"]
        # A track's points rows ARE its d(P) table: d grows with P for B.
        tr = g["points/track"][:]
        d = g["points/d"][:]
        p = g["points/pressure"][:]
        rows = tr == int(tb["track"])
        dp = sorted(zip(p[rows], d[rows]))
        assert len(dp) == 4 and dp[0][1] < dp[-1][1], dp
        # Every observation belongs to a consolidated pressure point.
        assert np.all(g["obs/point"][:] >= 0)
        # No track anywhere near the diamond line.
        q_first = g["tracks/q_first"][:]
        assert not np.any(np.abs(q_first - Q_DIAMOND) < 0.1), q_first

    # Re-run appends atomically (replaces /spots, no .tmp left behind).
    m2 = run_spot_tracking(reduced, min_snr=6.0, min_intensity=20.0,
                           min_track_points=3)
    assert m2["n_tracks"] == 2
    assert not out.with_name(out.name + ".tmp").exists()

    # With diamond exclusion off, the anvil line becomes a (flat) track too.
    m3 = run_spot_tracking(reduced, exclude_diamond=False, min_track_points=3)
    assert m3["n_tracks"] == 3
    dia = min(m3["tracks"], key=lambda s: abs(s["q_first"] - Q_DIAMOND))
    assert abs(dia["q_first"] - Q_DIAMOND) < 0.05
    assert abs(dia["dd_dp"]) < 0.002, dia["dd_dp"]         # anvil doesn't move

    # Manual d-line exclusion (gasket lines): removing blob B's d kills its
    # track but not A's.
    m5 = run_spot_tracking(reduced, exclude_d=[2.0 * np.pi / q_b(1.0)],
                           min_track_points=3)
    assert m5["n_tracks"] == 1
    assert circ_diff(m5["tracks"][0]["azim"], AZ_A) < 3.0, m5["tracks"]

    # exclude_frames drops listed exposures from detection entirely (the
    # cover-left-on seam): excluding every frame yields no tracks.
    m6 = run_spot_tracking(reduced, exclude_frames=range(len(cakes)),
                           min_track_points=3)
    assert m6["n_tracks"] == 0 and m6["n_obs"] == 0, m6

    # group_by='scan': independent ladder per position -> 2 tracks per scan;
    # the last scan's B-track bridges its missing step (gap tolerance).
    m4 = run_spot_tracking(reduced, group_by="scan", min_track_points=3)
    assert m4["n_tracks"] == 2 * N_SCANS, [
        (s["group_label"], round(s["azim"], 1)) for s in m4["tracks"]]
    last = f"scan{N_SCANS:03d}"
    tb_gap = next(s for s in m4["tracks"]
                  if s["group_label"] == last and circ_diff(s["azim"], AZ_B) < 3)
    assert tb_gap["n_points"] == 3 and tb_gap["p_max"] == 10.0, tb_gap


def _test_export(tmp: Path):
    """CSV handoff bundle: track summary (+hkl matches), long-format d(P)
    tables, untracked points, README — from a tracked file."""
    import csv
    cakes, names = _build_dataset()
    reduced = tmp / "reduced_exp.h5"
    _write_reduced(reduced, cakes, names)
    m = run_spot_tracking(reduced, min_snr=6.0, min_intensity=20.0,
                          min_track_points=3)
    out_h5 = Path(m["out_h5"])

    # reflection table matching blob B's ambient d (the NLC analog)
    refl = tmp / "calc.txt"
    d_b = 2.0 * np.pi / q_b(1.0)
    refl.write_text(f"{d_b + 0.01:.4f} 100.0\n1.2345 50.0\n", encoding="utf-8")

    dest = tmp / "handoff"
    man = export_spot_tracks(out_h5, dest, match=refl, match_tol=0.03,
                             include_observations=True)
    assert man["n_tracks"] == 2 and man["n_untracked_points"] == 0, man
    for fn in ("spot_tracks.csv", "spot_track_points.csv",
               "spot_untracked_points.csv", "spot_observations.csv", "README.txt"):
        assert (dest / fn).is_file(), fn

    with open(dest / "spot_tracks.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    # raw-image traceability: best_frame_file resolves through the /spots
    # provenance attrs (the standalone spots file has no /frames itself),
    # and the observations ledger carries scan + filename per detection.
    assert all(r["best_frame_file"].endswith(".tif") for r in rows), rows
    with open(dest / "spot_observations.csv", newline="", encoding="utf-8") as f:
        obs_rows = list(csv.DictReader(f))
    assert obs_rows and all(o["filename"].endswith(".tif") for o in obs_rows)
    assert all(o["scan"].startswith("scan") for o in obs_rows)
    assert any(o["scan"] == "scan002" for o in obs_rows)
    # blob B's track matches the reflection table; A has no match within tol.
    rb = min(rows, key=lambda r: abs(float(r["d0_A"]) - d_b))
    ra = max(rows, key=lambda r: abs(float(r["d0_A"]) - d_b))
    assert rb["match_d_calc_A"] != "" and abs(
        float(rb["match_d_calc_A"]) - (d_b + 0.01)) < 1e-4, rb   # CSV rounds to 5 dp
    assert ra["match_d_calc_A"] == "", ra
    assert float(rb["dd_dp_A_per_gpa"]) > 0 > float(ra["dd_dp_A_per_gpa"])

    # the points file is a per-track d(P) table ordered by (track, pressure)
    with open(dest / "spot_track_points.csv", newline="", encoding="utf-8") as f:
        pts = list(csv.DictReader(f))
    assert len(pts) == 8                              # 2 tracks x 4 P points
    keys = [(int(r["track"]), float(r["pressure_gpa"])) for r in pts]
    assert keys == sorted(keys), keys
    # README carries provenance (source file + at least one tracker knob)
    txt = (dest / "README.txt").read_text(encoding="utf-8")
    assert str(out_h5) in txt and "min_track_points" in txt

    # --- the GUI-facing loader reads the same file into a plot-ready form
    d = load_spot_tracks(out_h5, min_points=1, match=refl)
    assert d["ok"] and d["n_tracks_total"] == 2 and len(d["tracks"]) == 2
    tB = min(d["tracks"], key=lambda t: abs(t["d0"] - d_b))
    assert tB["hkl"] != "" and tB["dd_dp"] > 0            # labeled, rising
    assert np.all(np.diff(tB["pressure"]) > 0)             # pressure-sorted
    assert tB["pressure"].size == tB["d"].size == tB["n_points"]
    # min_points filter drops everything when set above the track length
    d2 = load_spot_tracks(out_h5, min_points=99)
    assert d2["ok"] and d2["tracks"] == [] and d2["n_tracks_total"] == 2
    # graceful errors: missing file / no /spots group
    assert not load_spot_tracks(out_h5.with_name("nope.h5"))["ok"]

    # --- ring-removed cake export: the powder ring cancels, the blob survives
    rdest = tmp / "ringless"
    man2 = export_ring_removed_cakes(reduced, rdest, [0], write_png=False)
    assert "cake_ringless_f00000.npy" in man2["files"]
    ex = np.load(rdest / "cake_ringless_f00000.npy")
    axes = np.load(rdest / "cake_axes.npz")
    assert ex.shape == (AZIM.size, RADIAL.size)
    assert np.allclose(axes["radial"], RADIAL)
    kr = int(np.argmin(np.abs(RADIAL - RING_Q)))        # powder-ring column
    ka = int(np.argmin(np.abs(RADIAL - q_a(LADDER[0]))))  # blob A column
    ring_med = float(np.nanmedian(ex[:, kr]))
    assert abs(ring_med) < 1.0, ring_med                 # ring level removed
    assert np.nanmax(ex[:, ka]) > 50.0                   # crystallite spot kept


_PONI_100K = """poni_version: 2.1
Detector: Pilatus100k
Detector_config: {}
Distance: 0.1
Poni1: 0.01677
Poni2: 0.0418
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 4.133e-11
"""


def _test_export_masks(tmp: Path) -> None:
    """Keep-only masked re-integration: .xye with Poisson esds, no-coverage
    bins between the kept boxes omitted, exclude_d windows dropped."""
    import math
    import h5py
    import tifffile
    from bulkxrd.analysis.spots import export_spot_masks

    rawdir = tmp / "masks_raw"
    rawdir.mkdir()
    tifffile.imwrite(str(rawdir / "raw0.tif"),
                     np.full((195, 487), 100, dtype=np.int32))

    red = tmp / "masks_red.h5"
    with h5py.File(str(red), "w") as h:
        h.attrs["unit"] = "q_A^-1"
        h.attrs["poni_text"] = _PONI_100K
        pg = h.create_group("patterns")
        pg.create_dataset("radial", data=np.linspace(0.5, 6.0, 200))
        cg = h.create_group("cakes")
        cg.create_dataset("radial", data=RADIAL)
        cg.create_dataset("azimuthal", data=AZIM)
        fr = h.create_group("frames")
        fr.create_dataset("filename", data=np.array(["raw0.tif"], object),
                          dtype=h5py.string_dtype(encoding="utf-8"))
        fr.create_dataset("excluded", data=np.zeros(1, bool))

    spots = tmp / "masks_spots.h5"
    with h5py.File(str(spots), "w") as h:
        og = h.create_group("spots").create_group("obs")
        og.create_dataset("frame", data=np.array([0, 0], "i4"))
        og.create_dataset("q", data=np.array([3.0, 2.0]))
        og.create_dataset("d", data=2.0 * np.pi / np.array([3.0, 2.0]))
        og.create_dataset("azim", data=np.array([0.0, 90.0]))
        og.create_dataset("q_width", data=np.array([0.05, 0.05]))
        og.create_dataset("azim_width", data=np.array([20.0, 20.0]))
        og.create_dataset("track", data=np.array([0, 1], "i4"))

    out = tmp / "masks_out"
    man = export_spot_masks(red, spots, out, dataset_dir=rawdir)
    assert man["n_frames"] == 1, man
    assert (out / "frame_0000_mask.npy").is_file()
    xye = out / "frame_0000_masked_q.xye"
    assert xye.is_file(), man["files"]
    assert (out / "frame_0000_masked.xye").is_file()
    data = np.loadtxt(xye, ndmin=2)
    assert data.shape[1] == 3                      # x, intensity, esd
    assert np.all(data[:, 2] > 0)
    # only the two kept boxes contribute: no rows in the coverage gap
    assert data.shape[0] < 200
    assert np.any(np.abs(data[:, 0] - 3.0) < 0.1)  # spot 1 region present
    assert np.any(np.abs(data[:, 0] - 2.0) < 0.1)  # spot 2 region present
    assert not np.any((data[:, 0] > 2.3) & (data[:, 0] < 2.7))   # the gap

    # exclude_d drops the spot-2 window from the written pattern
    out2 = tmp / "masks_out_excl"
    export_spot_masks(red, spots, out2, dataset_dir=rawdir,
                      exclude_d=[math.pi])         # d = pi <-> q = 2.0
    data2 = np.loadtxt(out2 / "frame_0000_masked_q.xye", ndmin=2)
    assert not np.any((data2[:, 0] >= 2.0 * (1 - 0.028))
                      & (data2[:, 0] <= 2.0 * (1 + 0.028)))
    assert np.any(np.abs(data2[:, 0] - 3.0) < 0.1)  # spot 1 survives


def main() -> None:
    _test_helpers()
    _test_detection()
    _test_consolidation()
    _test_linking()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _test_matcher_parsing(tmp)
        _test_end_to_end(tmp)
        _test_export(tmp)
        _test_export_masks(tmp)
    print("test_spots: all assertions passed")


if __name__ == "__main__":
    main()
