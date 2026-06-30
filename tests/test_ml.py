"""Step 3b scaffolding: frame features, DAC-augmented simulator, candidate ranker.

All three are exercised without pymatgen by injecting synthetic reflections
(built directly from a cubic metric, as in test_identify's anisotropic case), so
they run in CI where pymatgen is absent.
"""
import sys
import math
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.analysis import ml_features as mf
from bulkxrd.analysis import ml_simulate as ms
from bulkxrd.analysis import ml_rank as mr
from bulkxrd.analysis import identify as idf
from bulkxrd.analysis.peaks import pseudo_voigt
from bulkxrd.analysis.phases import Phase


def _au_refl():
    a0 = 4.078
    rows = [("111", 3, 100), ("200", 4, 46), ("220", 8, 26), ("311", 11, 28),
            ("222", 12, 8), ("400", 16, 4)]
    d0 = np.array([a0 / math.sqrt(s) for _, s, _ in rows])
    w = np.array([i for *_, i in rows], float) / 100.0
    return d0, w, [""] * len(d0)


def _write_analysis(path, *, pressure, with_residual=True, nb=1200):
    """Analysis HDF5 whose clean/residual holds Au peaks at `pressure` (q axis)."""
    import h5py
    d0, w, _ = _au_refl()
    au = Phase(name="Au", eos={"type": "BM3", "K0": 167, "K0p": 5.0})
    s = idf.scale_at_pressure(au, pressure)
    q = np.linspace(1.0, 7.0, nb)
    row = np.zeros(nb)
    for c, a in zip(2 * np.pi / (d0 * s), w):
        if q[0] <= c <= q[-1]:
            row += pseudo_voigt(q, c, a * 100, 0.05, 0.5)
    stack = np.stack([row, row]).astype("f4")
    with h5py.File(str(path), "w") as h:
        h.attrs["unit"] = "q_A^-1"
        h.create_dataset("radial", data=q)
        gb = h.create_group("background")
        gb.create_dataset("clean", data=stack)
        gb.create_dataset("spot_residual", data=np.zeros_like(stack))
        gb.create_dataset("baseline", data=np.zeros_like(stack))
        gp = h.create_group("peaks")
        gp.attrs["source"] = "clean"
        gp.create_dataset("frame", data=np.array([0, 1], "i4"))
        gp.create_dataset("flag", data=np.zeros(2, "i4"))
        gf = h.create_group("frames")
        gf.create_dataset("pressure", data=np.array([pressure, pressure]))
        gf.create_dataset("contamination", data=np.array([0.1, 0.2]))
        gf.create_dataset("excluded", data=np.zeros(2, "?"))
        if with_residual:
            h.create_group("residual").create_dataset("clean", data=stack)


def test_frame_features():
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "an.h5"
        _write_analysis(h5, pressure=30.0)
        ff = mf.frame_features(h5, source="fit")
        assert ff.X.shape == (2, ff.d_grid.size) and ff.d_grid.size == 3501
        assert ff.source == "clean"                  # recorded /peaks source
        assert np.allclose(ff.pressure, [30.0, 30.0])
        assert np.allclose(ff.contamination, [0.1, 0.2])
        assert ff.n_peaks.tolist() == [1, 1] and ff.excluded.tolist() == [False, False]
        assert abs(float(ff.X.max()) - 1.0) < 1e-6   # row-normalised
        # residual source resolves too
        ffr = mf.frame_features(h5, source="residual")
        assert ffr.source == "residual" and ffr.X.shape == (2, 3501)


def test_augmented_dataset():
    au = Phase(name="Au", eos={"type": "BM3", "K0": 167, "K0p": 5.0})
    si = Phase(name="Si", eos={"type": "BM3", "K0": 98, "K0p": 4.0})
    d0, w, hkl = _au_refl()
    refl = {"Au": (d0, w, hkl), "Si": (d0 * 1.1, w[::-1], hkl)}
    grid = ms.make_d_grid()

    X, Y, names, P = ms.build_augmented_dataset(
        [au, si], n_samples=24, max_phases_per_pattern=1, reflections=refl, seed=1)
    assert X.shape == (24, 3501) and Y.shape == (24, 2) and names == ["Au", "Si"]
    assert set(Y.sum(axis=1).tolist()) == {1}                 # single-label
    assert float(X.max()) <= 1.0 + 1e-6 and not np.allclose(X[0], X[1])

    # mixtures -> multi-label rows appear
    _, Y2, _, _ = ms.build_augmented_dataset(
        [au, si], n_samples=40, max_phases_per_pattern=2, reflections=refl, seed=2)
    assert int(Y2.sum(axis=1).max()) == 2

    # pressure conditioning shifts peaks (50 GPa compresses d)
    flat = ms.AugmentConfig(noise_sigma=(0, 0), n_humps=(0, 0), n_diamond_spikes=(0, 0),
                            drop_frac=(0, 0), intensity_jitter=0.0, truncate_frac=(0, 0),
                            d_offset_frac=0.0)
    y0 = ms.simulate_augmented_pattern([au], [0.0], refl, grid, flat, np.random.default_rng(0))
    y50 = ms.simulate_augmented_pattern([au], [50.0], refl, grid, flat, np.random.default_rng(0))
    assert grid[int(np.argmax(y50))] < grid[int(np.argmax(y0))]

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "aug.npz"
        man = ms.export_augmented_dataset(out, [au, si], n_samples=8, reflections=refl, seed=3)
        assert out.is_file() and man["n_samples"] == 8
        with np.load(out, allow_pickle=True) as z:
            assert z["X"].shape == (8, 3501) and z["Y"].shape == (8, 2)
            assert list(z["phase_names"]) == ["Au", "Si"]


def test_rank_candidates_and_shortlist():
    au = Phase(name="Au", eos={"type": "BM3", "K0": 167, "K0p": 5.0})
    decoy = Phase(name="Decoy", eos={"type": "BM3", "K0": 80, "K0p": 4.0})
    d0, w, hkl = _au_refl()
    refl = {"Au": (d0, w, hkl), "Decoy": (d0 * 1.12, w[::-1], hkl)}
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "an.h5"
        _write_analysis(h5, pressure=30.0)             # residual holds Au at 30 GPa
        man = mr.rank_candidates(h5, [au, decoy], reflections=refl, top_k=2, fwhm_d=0.05)
        assert man["ranking_source"] == "residual"     # auto picks the residual
        assert "Au" in man["candidates"]
        rc = mr.read_candidates(h5)
        assert rc["ok"] and rc["n_frames"] == 2
        # Au (correct phase at the metadata pressure) outranks the decoy.
        assert rc["phases"]["Au"]["score"][0] > rc["phases"]["Decoy"]["score"][0]
        assert rc["phases"]["Au"]["score"][0] > 0.8
        assert rc["topk_names"][0][0] == "Au"
        assert np.allclose(rc["phases"]["Au"]["pressure"], [30.0, 30.0])  # used the prior

        # No metadata pressure -> the ranker scans a coarse grid and still finds Au.
        import h5py
        with h5py.File(str(h5), "r+") as f:
            f["frames/pressure"][...] = np.nan
        man2 = mr.rank_candidates(h5, [au, decoy], reflections=refl, top_k=1, fwhm_d=0.05,
                                  pressure_grid=np.arange(0, 101, 10.0))
        assert man2["candidates"] == ["Au"]


def main() -> None:
    test_frame_features()
    test_augmented_dataset()
    test_rank_candidates_and_shortlist()
    print("ML TEST OK")


if __name__ == "__main__":
    main()
