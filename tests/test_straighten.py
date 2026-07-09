"""Cake waviness diagnosis + straightening (reduce/straighten.py).

A synthetic cake with rings displaced by A1·cos(φ−φ0) emulates the DAC
sample-off-calibrant-position geometry error: the naive azimuthal mean shows
double-horned peaks; the straightened collapse must recover single sharp rings
and the fit must recover the injected amplitude/phase (and, through a synthetic
reduced HDF5, the physical transverse offset in mm).
"""
import sys
import math
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.reduce.straighten import (ring_centroids, fit_waviness,
                                       straighten_cake, diagnose_reduced,
                                       straighten_reduced)

Q = np.linspace(0.5, 6.0, 600)
AZ = np.linspace(-180.0, 180.0, 72, endpoint=False)
A1, PHI1 = 0.020, 35.0                       # injected wobble (q units, deg)
RINGS = (2.5, 3.5, 4.6)
SIGMA = 0.012                                # ring radial width


def _wavy_cake(a1=A1, phi1=PHI1, a2=0.0):
    cake = np.zeros((AZ.size, Q.size))
    for j, phi in enumerate(AZ):
        shift = (a1 * math.cos(math.radians(phi - phi1))
                 + a2 * math.cos(2 * math.radians(phi)))
        for r0, amp in zip(RINGS, (100.0, 60.0, 40.0)):
            cake[j] += amp * np.exp(-0.5 * ((Q - r0 - shift) / SIGMA) ** 2)
    return cake


def _g(x, c, a, w):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


def test_fit_recovers_wobble():
    cake = _wavy_cake()
    cent = ring_centroids(cake, Q, 2.5, 0.08)
    assert np.isfinite(cent).sum() > 60
    f = fit_waviness(AZ, cent)
    assert f["ok"], f
    assert abs(f["r0"] - 2.5) < 0.002
    assert abs(f["A1"] - A1) < 0.002, f["A1"]
    assert abs((f["phi1_deg"] - PHI1 + 180) % 360 - 180) < 5, f["phi1_deg"]
    assert f["A2"] < 0.003                    # no injected tilt term
    # second harmonic recovered when injected
    cent2 = ring_centroids(_wavy_cake(a1=0.0, a2=0.015), Q, 2.5, 0.08)
    f2 = fit_waviness(AZ, cent2)
    assert f2["ok"] and abs(f2["A2"] - 0.015) < 0.002 and f2["A1"] < 0.003


def test_straighten_removes_double_horns():
    cake = _wavy_cake()
    res = straighten_cake(cake, Q, AZ)        # auto ring pick
    assert res["ok"], res.get("error")
    naive = np.nanmean(np.where(cake > 0, cake, np.nan), axis=0)
    fixed = res["intensity"]
    k = int(np.argmin(np.abs(Q - 2.5)))
    win = slice(k - 8, k + 9)
    # The wobble (A1 >> SIGMA?) splits the naive mean: its maximum near the ring
    # is depressed relative to the straightened one, and the straightened peak
    # is centred on the true r0 and narrower.
    assert np.nanmax(fixed[win]) > 1.3 * np.nanmax(naive[win])
    assert abs(Q[k - 8 + int(np.nanargmax(fixed[win]))] - 2.5) < 0.01

    def fwhm(y):
        y = np.nan_to_num(y[win], nan=0.0)
        half = y.max() / 2
        above = np.where(y > half)[0]
        return (above.max() - above.min()) * (Q[1] - Q[0]) if above.size else 0.0

    assert fwhm(fixed) < 0.7 * fwhm(naive), (fwhm(fixed), fwhm(naive))
    # every fitted ring reports the same wobble amplitude
    for f in res["fits"]:
        if f["ok"]:
            assert abs(f["A1"] - A1) < 0.003, (f["ring_r0"], f["A1"])


def test_diagnose_reduced_offset_mm():
    import h5py
    lam, dist = 0.4133, 0.3                   # Å, metres
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "red.h5"
        with h5py.File(str(p), "w") as f:
            f.attrs["unit"] = "q_A^-1"
            f.attrs["poni_text"] = (f"Distance: {dist}\n"
                                    f"Wavelength: {lam * 1e-10}\n")
            g = f.create_group("cakes")
            g.create_dataset("intensity", data=np.stack([_wavy_cake()] * 2))
            g.create_dataset("radial", data=Q)
            g.create_dataset("azimuthal", data=AZ)
            g.create_dataset("frame_index", data=np.array([0, 5]))
        rep = diagnose_reduced(p)
        assert rep["ok"], rep["error"]
        assert rep["n_cakes"] == 2 and rep["per_frame"][1]["frame"] == 5
        s = rep["summary"]
        assert abs(s["A1_median"] - A1) < 0.003
        assert abs(s["doublet_splitting"] - 2 * A1) < 0.006
        # offset = d2θ·D = (A1·λ/2π)·D
        expect_mm = (A1 * lam / (2 * math.pi)) * dist * 1000.0
        assert abs(s["offset_mm"] - expect_mm) < 0.15 * expect_mm, \
            (s["offset_mm"], expect_mm)
    # graceful when cakes are absent
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "nocake.h5"
        with h5py.File(str(p), "w") as f:
            f.attrs["unit"] = "q_A^-1"
        rep = diagnose_reduced(p)
        assert not rep["ok"] and "save_cakes" in rep["error"]


def test_straighten_reduced_writer():
    """straighten_reduced writes /patterns/intensity_straightened (+ per-frame
    waviness) atomically; frames without a saved cake stay NaN."""
    import h5py
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "red.h5"
        with h5py.File(str(p), "w") as f:
            f.attrs["unit"] = "q_A^-1"
            gp = f.create_group("patterns")
            gp.create_dataset("radial", data=Q)
            gp.create_dataset("intensity", data=np.zeros((3, Q.size), "f4"))
            g = f.create_group("cakes")
            g.create_dataset("intensity", data=np.stack([_wavy_cake()] * 2))
            g.create_dataset("radial", data=Q)
            g.create_dataset("azimuthal", data=AZ)
            g.create_dataset("frame_index", data=np.array([0, 2]))  # cake_every=2
        man = straighten_reduced(p)
        assert man["n_straightened"] == 2
        assert abs(man["A1_median"] - A1) < 0.003
        with h5py.File(str(p), "r") as f:
            st = f["patterns/intensity_straightened"][:]
            stm = f["patterns/intensity_straightened_robust"][:]   # median channel
            wa = f["frames/waviness_A1"][:]
            assert st.shape == (3, Q.size) and stm.shape == (3, Q.size)
            assert np.isnan(st[1]).all()            # no cake for frame 1
            assert np.isnan(stm[1]).all()
            k = int(np.argmin(np.abs(Q - 2.5)))
            for fr in (0, 2):
                for ch in (st, stm):
                    win = ch[fr][k - 8:k + 9]
                    assert np.nanmax(win) > 50      # sharp straightened ring
                    assert abs(Q[k - 8 + int(np.nanargmax(win))] - 2.5) < 0.01
            assert np.isnan(wa[1]) and abs(wa[0] - A1) < 0.003


def test_straighten_reduced_robust_suppresses_spots():
    """intensity_straightened_robust is the straightened MEDIAN, so a single-
    crystal spot that lifts the straightened MEAN must not lift it."""
    import h5py
    cake = _wavy_cake()
    ks = int(np.argmin(np.abs(Q - 3.0)))         # between rings (2.5, 3.5)
    cake[3, ks - 1:ks + 2] += 5000.0             # one bright azimuthal 'spot'
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "red.h5"
        with h5py.File(str(p), "w") as f:
            f.attrs["unit"] = "q_A^-1"
            gp = f.create_group("patterns")
            gp.create_dataset("radial", data=Q)
            gp.create_dataset("intensity", data=np.zeros((1, Q.size), "f4"))
            g = f.create_group("cakes")
            g.create_dataset("intensity", data=cake[None])
            g.create_dataset("radial", data=Q)
            g.create_dataset("azimuthal", data=AZ)
            g.create_dataset("frame_index", data=np.array([0]))
        straighten_reduced(p)
        with h5py.File(str(p), "r") as f:
            mean_ch = f["patterns/intensity_straightened"][0]
            med_ch = f["patterns/intensity_straightened_robust"][0]
    win = slice(ks - 4, ks + 5)
    # the spot lifts the mean between the rings but the median rejects it
    assert np.nanmax(mean_ch[win]) > np.nanmax(med_ch[win]) + 30


def test_background_straightened_source():
    """run_background_separation(robust_source='straightened') builds Step 1 on the
    de-waved median: a double-horned robust peak becomes a single clean peak, the
    reduce sigmaclip is skipped, and cake-less frames fall back to the ordinary
    median. Requesting it without the channel raises an instructive error."""
    import h5py
    from bulkxrd.analysis.background import run_background_separation
    q = np.linspace(1.0, 8.0, 800)
    n = 3
    bg = 50.0 + 30.0 * np.exp(-(q - 1.0) / 5.0)
    horns = bg + _g(q, 4.0 - 0.05, 200, 0.02) + _g(q, 4.0 + 0.05, 200, 0.02)
    single = bg + _g(q, 4.0, 380, 0.02)
    robust = np.tile(horns, (n, 1))
    straight_med = np.tile(single, (n, 1))
    spot = _g(q, 4.0, 1500, 0.015)
    mean = robust + spot
    straight_mean = straight_med + spot
    straight_med[2] = np.nan                     # frame 2 had no cake → fall back
    straight_mean[2] = np.nan
    kc = int(np.argmin(np.abs(q - 4.0)))
    w = slice(kc - 30, kc + 31)

    def _write(path, with_straight):
        with h5py.File(str(path), "w") as f:
            f.attrs["unit"] = "q_A^-1"
            gp = f.create_group("patterns")
            gp.create_dataset("radial", data=q)
            gp.create_dataset("intensity", data=mean.astype("f4"))
            gp.create_dataset("intensity_robust", data=robust.astype("f4"))
            if with_straight:
                gp.create_dataset("intensity_straightened", data=straight_mean.astype("f4"))
                gp.create_dataset("intensity_straightened_robust",
                                  data=straight_med.astype("f4"))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "red.h5"
        _write(p, with_straight=True)
        mr = run_background_separation(p, Path(td) / "an_robust.h5",
                                       robust_source="robust", max_half_window=60)
        ms = run_background_separation(p, Path(td) / "an_str.h5",
                                       robust_source="straightened", max_half_window=60)
        with h5py.File(ms["out_h5"], "r") as f:
            assert f.attrs["robust_source"] == "straightened"
            assert int(f.attrs["n_straightened"]) == 2         # frame 2 fell back
            assert not bool(f.attrs["has_sigmaclip"])          # skipped in this mode
            clean_s = f["background/clean"][0]
            clean_s2 = f["background/clean"][2]
        with h5py.File(mr["out_h5"], "r") as f:
            clean_r = f["background/clean"][0]
            clean_r2 = f["background/clean"][2]
        # straightened: the center is the maximum (one ring)
        assert clean_s[kc] >= np.max(clean_s[w]) - 1e-3
        # robust: the center dips between the two horns
        assert clean_r[kc] < 0.9 * np.max(clean_r[w])
        # frame 2 (no cake) is identical between the two modes — it fell back
        assert np.allclose(clean_s2, clean_r2, atol=1e-3)

    # Requesting straightened without the channel → instructive error.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "red_nostraight.h5"
        _write(p, with_straight=False)
        try:
            run_background_separation(p, Path(td) / "x.h5", robust_source="straightened")
            assert False, "expected ValueError for missing straightened channel"
        except ValueError as e:
            assert "straightened" in str(e).lower()


def main() -> None:
    test_fit_recovers_wobble()
    test_straighten_removes_double_horns()
    test_diagnose_reduced_offset_mm()
    test_straighten_reduced_writer()
    test_straighten_reduced_robust_suppresses_spots()
    test_background_straightened_source()
    print("STRAIGHTEN TEST OK")


if __name__ == "__main__":
    main()
