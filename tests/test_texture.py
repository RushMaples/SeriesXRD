"""Azimuthal texture analysis (reduce/texture.py).

Synthetic cakes with a known azimuthal intensity profile I(φ) emulate three
regimes a real Debye ring can be in: an ideal random powder (uniform I(φ)), a
textured/stressed sample (smooth 2-fold I(φ) modulation), and a coarse-grained
/ near-single-crystal sample (a few bright azimuthal spots on an otherwise weak
ring). ring_profile + texture_metrics must recover the injected numbers, and
run_texture must write them into a synthetic reduced HDF5's /texture group.
"""
import sys
import math
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seriesxrd.reduce.texture import ring_profile, texture_metrics, run_texture

Q = np.linspace(0.5, 6.0, 600)
AZ = np.linspace(-180.0, 180.0, 72, endpoint=False)
R0 = 2.5
SIGMA = 0.02
HALFWIDTH = 0.08


def _ring_cake(amp_at_phi, r0=R0, sigma=SIGMA):
    """Cake with a single ring at r0 whose amplitude varies with azimuth."""
    cake = np.zeros((AZ.size, Q.size))
    shape = np.exp(-0.5 * ((Q - r0) / sigma) ** 2)
    for j, phi in enumerate(AZ):
        cake[j] = amp_at_phi(phi) * shape
    return cake


def _uniform_texture_index():
    cake = _ring_cake(lambda phi: 100.0)
    prof = ring_profile(cake, Q, AZ, R0, HALFWIDTH)
    assert prof["ok"]
    assert np.isfinite(prof["intensity"]).sum() == AZ.size   # every row significant
    met = texture_metrics(prof["phi"], prof["intensity"])
    assert met["ok"]
    assert met["coverage"] == 1.0
    assert met["texture_index"] < 1e-6, met["texture_index"]
    assert met["po_amplitude"] < 1e-6, met["po_amplitude"]
    assert met["spotty_frac"] == 0.0
    return met["texture_index"]


def test_uniform_ring_is_texture_free():
    _uniform_texture_index()


def test_preferred_orientation_recovered():
    amp0, po_amp, po_phase = 100.0, 0.5, 30.0
    amp = lambda phi: amp0 * (1.0 + po_amp * math.cos(2 * math.radians(phi - po_phase)))
    cake = _ring_cake(amp)
    prof = ring_profile(cake, Q, AZ, R0, HALFWIDTH)
    assert prof["ok"]
    met = texture_metrics(prof["phi"], prof["intensity"])
    assert met["ok"]
    assert abs(met["po_amplitude"] - po_amp) < 0.1 * po_amp, met["po_amplitude"]
    dphi = (met["po_phase_deg"] - po_phase + 90) % 180 - 90
    assert abs(dphi) < 5.0, met["po_phase_deg"]
    uniform_texture_index = _uniform_texture_index()
    assert met["texture_index"] > uniform_texture_index


def test_spotty_ring_detected():
    weak, spike = 10.0, 200.0
    spot_idx = {5, 20, 40, 55}
    amps = [spike if j in spot_idx else weak for j in range(AZ.size)]
    cake = np.zeros((AZ.size, Q.size))
    shape = np.exp(-0.5 * ((Q - R0) / SIGMA) ** 2)
    for j, a in enumerate(amps):
        cake[j] = a * shape
    prof = ring_profile(cake, Q, AZ, R0, HALFWIDTH)
    assert prof["ok"]
    met = texture_metrics(prof["phi"], prof["intensity"])
    assert met["ok"]
    assert met["spotty_frac"] > 0.05, met["spotty_frac"]
    assert met["spotty_frac"] >= len(spot_idx) / AZ.size - 1e-9
    uniform_prof = ring_profile(_ring_cake(lambda phi: 100.0), Q, AZ, R0, HALFWIDTH)
    uniform_met = texture_metrics(uniform_prof["phi"], uniform_prof["intensity"])
    assert met["spotty_frac"] > uniform_met["spotty_frac"]


def _build_reduced_h5(path):
    import h5py
    cake_uniform = _ring_cake(lambda phi: 100.0)
    cake_textured = _ring_cake(
        lambda phi: 100.0 * (1.0 + 0.5 * math.cos(2 * math.radians(phi - 30.0))))
    with h5py.File(str(path), "w") as f:
        f.attrs["unit"] = "q_A^-1"
        g = f.create_group("cakes")
        g.create_dataset("intensity", data=np.stack([cake_uniform, cake_textured]))
        g.create_dataset("radial", data=Q)
        g.create_dataset("azimuthal", data=AZ)
        g.create_dataset("frame_index", data=np.array([0, 2]))  # cake_every=2


def test_run_texture_writer():
    import h5py
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "red.h5"
        _build_reduced_h5(p)

        man = run_texture(p, n_rings=2)
        assert man["n_cakes"] == 2 and man["n_rings"] == 2
        assert man["frame"] == [0, 2]                 # frame 1 (no cake) absent

        with h5py.File(str(p), "r") as f:
            gt = f["texture"]
            assert list(gt["frame"][:]) == [0, 2]
            assert gt["ring_r0"].shape == (2, 2)
            assert gt["texture_index"].shape == (2, 2)
            assert gt["po_amplitude"].shape == (2, 2)
            assert gt["po_phase_deg"].shape == (2, 2)
            assert gt["spotty_frac"].shape == (2, 2)
            assert gt["coverage"].shape == (2, 2)
            assert gt.attrs["n_rings"] == 2
            assert gt.attrs["unit"] == "q_A^-1"
            ring_r0_1 = gt["ring_r0"][:]
            texture_index_1 = gt["texture_index"][:]
            po_amplitude_1 = gt["po_amplitude"][:]
            # the strongest ring found is R0 for both cakes (only ring present)
            assert np.any(np.isclose(ring_r0_1, R0, atol=0.05))
            # cake 1 (textured) has higher texture_index / po_amplitude than
            # cake 0 (uniform) on the matching ring row
            assert np.nanmax(texture_index_1[1]) > np.nanmax(texture_index_1[0])
            assert np.nanmax(po_amplitude_1[1]) > np.nanmax(po_amplitude_1[0])

        # re-run replaces the group and is idempotent (deterministic inputs)
        run_texture(p, n_rings=2)
        with h5py.File(str(p), "r") as f:
            gt = f["texture"]
            np.testing.assert_allclose(gt["ring_r0"][:], ring_r0_1, equal_nan=True)
            np.testing.assert_allclose(gt["texture_index"][:], texture_index_1,
                                       equal_nan=True)
            np.testing.assert_allclose(gt["po_amplitude"][:], po_amplitude_1,
                                       equal_nan=True)


def test_run_texture_no_cakes_raises():
    import h5py
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "nocake.h5"
        with h5py.File(str(p), "w") as f:
            f.attrs["unit"] = "q_A^-1"
        try:
            run_texture(p)
            raise AssertionError("expected ValueError for missing /cakes")
        except ValueError as e:
            assert "save_cakes" in str(e)


def main() -> None:
    test_uniform_ring_is_texture_free()
    test_preferred_orientation_recovered()
    test_spotty_ring_detected()
    test_run_texture_writer()
    test_run_texture_no_cakes_raises()
    print("TEXTURE TEST OK")


if __name__ == "__main__":
    main()
