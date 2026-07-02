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
                                       straighten_cake, diagnose_reduced)

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


def main() -> None:
    test_fit_recovers_wobble()
    test_straighten_removes_double_horns()
    test_diagnose_reduced_offset_mm()
    print("STRAIGHTEN TEST OK")


if __name__ == "__main__":
    main()
