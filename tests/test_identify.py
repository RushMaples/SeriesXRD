"""Step 3a deterministic EOS phase matching: axis conversion + pressure recovery.

The axis→d-spacing conversions are pure numpy and always tested. The matching /
pressure recovery and the end-to-end driver need pymatgen (for the reflection
simulation) and are skipped when it isn't installed.
"""
import sys
import math
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.analysis import identify as idf
from bulkxrd.analysis import phases as ph


def test_radial_to_d():
    # q (Å^-1): d = 2π/q
    q = np.array([1.0, 2.0, 3.1416])
    assert np.allclose(idf.radial_to_d(q, "q_A^-1"), 2 * np.pi / q)
    # q (nm^-1): 1 nm^-1 = 0.1 Å^-1
    qn = np.array([10.0, 20.0])
    assert np.allclose(idf.radial_to_d(qn, "q_nm^-1"), 2 * np.pi / (qn * 0.1))
    # 2θ (deg) with λ: d = λ / (2 sin θ)
    lam = 0.4133
    tt = np.array([10.0, 20.0])
    theta = np.radians(tt) / 2.0
    assert np.allclose(idf.radial_to_d(tt, "2th_deg", lam), lam / (2 * np.sin(theta)))
    # 2θ axis without wavelength must error.
    try:
        idf.radial_to_d(tt, "2th_deg")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_scale_monotonic():
    au = ph.load_bundled()["Au"]
    assert idf.scale_at_pressure(au, 0.0) == 1.0
    s50 = idf.scale_at_pressure(au, 50.0)
    s100 = idf.scale_at_pressure(au, 100.0)
    assert s100 < s50 < 1.0  # compresses, monotonically


def test_pressure_recovery():
    if not ph.pymatgen_available():
        print("  (pymatgen not installed — skipping pressure recovery)")
        return
    au = ph.load_bundled()["Au"]
    refl = idf.phase_reflections(au)
    d0, w, _ = refl
    assert d0.size >= 3

    for p_true in (20.0, 50.0, 120.0):
        s = idf.scale_at_pressure(au, p_true)
        # Observed = the strongest few reflections at the true pressure (+ jitter).
        rng = np.random.default_rng(int(p_true))
        obs_d = d0[:5] * s * (1.0 + rng.normal(0, 5e-4, 5))
        res = idf.fit_pressure_for_phase(obs_d, au, refl, p_min=0.0, p_max=200.0,
                                         rel_tol=0.01)
        assert abs(res["pressure"] - p_true) < 3.0, \
            f"recovered {res['pressure']:.1f} != {p_true}"
        assert res["confidence"] > 0.8 and res["n_matched"] >= 4


def _make_analysis_with_peaks(path, au, p_true, n_frames=4):
    """Minimal analysis HDF5 carrying a /peaks group with Au lines at p_true."""
    import h5py
    refl = idf.phase_reflections(au)
    d0, _, _ = refl
    s = idf.scale_at_pressure(au, p_true)
    centers_d = d0[:6] * s
    centers_q = 2 * np.pi / centers_d            # store on a q_A^-1 axis
    k = centers_q.size
    with h5py.File(str(path), "w") as h5:
        h5.attrs["unit"] = "q_A^-1"
        h5.attrs["source_reduced"] = "synthetic"
        gp = h5.create_group("peaks")
        gp.create_dataset("counts", data=np.full(n_frames, k, "i4"))
        gp.create_dataset("frame", data=np.repeat(np.arange(n_frames), k).astype("i4"))
        gp.create_dataset("center", data=np.tile(centers_q, n_frames).astype("f8"))
        gp.create_dataset("flag", data=np.zeros(n_frames * k, "i4"))


def test_skip_structureless_phase():
    """Open-set mode sweeps the whole library, which can include phases with no
    simulatable structure (e.g. He). They must be skipped, not crash the run."""
    if not ph.pymatgen_available():
        print("  (pymatgen not installed — skipping structureless test)")
        return
    au = ph.load_bundled()["Au"]
    he = ph.Phase(name="He")                    # no lattice/atoms → not simulatable
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "analysis.h5"
        _make_analysis_with_peaks(h5, au, p_true=60.0, n_frames=3)
        manifest = idf.run_identification(h5, [he, au], p_min=0.0, p_max=200.0)
        assert manifest["phases"] == ["Au"], manifest["phases"]


def test_run_identification():
    if not ph.pymatgen_available():
        print("  (pymatgen not installed — skipping run_identification)")
        return
    au = ph.load_bundled()["Au"]
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "analysis.h5"
        _make_analysis_with_peaks(h5, au, p_true=60.0, n_frames=4)
        manifest = idf.run_identification(h5, [au], p_min=0.0, p_max=200.0)
        assert manifest["steps"] if "steps" in manifest else True
        assert manifest["phases"] == ["Au"]
        from bulkxrd.analysis.review import identify_tracks
        tr = identify_tracks(h5)
        assert tr["ok"] and tr["n_frames"] == 4 and len(tr["phases"]) == 1
        rec = tr["phases"][0]
        assert rec["name"] == "Au"
        assert abs(float(np.median(rec["pressure"])) - 60.0) < 3.0
        assert np.all(rec["confidence"] > 0.8)
        # Reflections must be cached so GUI overlays need no pymatgen.
        import h5py
        with h5py.File(str(h5), "r") as f:
            g = f["identify"]["Au"]
            assert "refl_d" in g and g["refl_d"].size >= 3
        # reflection_tracks must read the cache (works regardless of pymatgen).
        from bulkxrd.analysis.heatmap import reflection_tracks
        tr2 = reflection_tracks(h5, au)
        assert tr2["ok"] and len(tr2["tracks"]) >= 3


def test_axial_eos_anisotropic():
    """Per-axis (anisotropic) compression: (00l) reflections compress more than
    (h00) when the c-axis is softer, and the fit recovers the pressure. No
    pymatgen needed — reflections are built directly."""
    # d-spacing from the lattice metric (cubic sanity: d = a/sqrt(h²+k²+l²)).
    Lc = {"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90, "beta": 90, "gamma": 90}
    H = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], float)
    assert np.allclose(idf._d_from_lattice(H, Lc),
                       [4.0, 4.0 / np.sqrt(2), 4.0 / np.sqrt(3)])

    # Tetragonal phase, c-axis softer (K0=100) than a-axis (K0=300).
    tet = ph.Phase(name="tet",
                   lattice={"a": 4.0, "b": 4.0, "c": 6.0,
                            "alpha": 90, "beta": 90, "gamma": 90},
                   axial_eos={"a": {"type": "BM3", "K0": 300, "K0p": 4},
                              "c": {"type": "BM3", "K0": 100, "K0p": 4}})
    assert ph.has_axial_eos(tet)
    sa, sb, sc = ph.axial_scales(tet, 10.0)
    assert sc < sa < 1.0 and abs(sa - sb) < 1e-9   # c compresses more; b inherits a

    # (001) shrinks more than (100) relative to ambient.
    dp = idf.predicted_d(tet, np.array([4.0, 6.0]), [(1, 0, 0), (0, 0, 1)], 10.0)
    assert (dp[1] / 6.0) < (dp[0] / 4.0) < 1.0

    # Build observed reflections at a known pressure and recover it.
    P_true = 8.0
    sa, sb, sc = ph.axial_scales(tet, P_true)
    Lp = {"a": 4 * sa, "b": 4 * sb, "c": 6 * sc, "alpha": 90, "beta": 90, "gamma": 90}
    Hf = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 2]], float)
    obs = idf._d_from_lattice(Hf, Lp)
    d0 = idf._d_from_lattice(Hf, tet.lattice)
    refl = (d0, np.ones(len(Hf)),
            ["(1, 0, 0)", "(0, 0, 1)", "(1, 0, 1)", "(1, 1, 0)", "(0, 0, 2)"])
    res = idf.fit_pressure_for_phase(obs, tet, refl, p_min=0, p_max=30, rel_tol=0.004)
    assert abs(res["pressure"] - P_true) < 1.5, res["pressure"]
    # An isotropic fit of the same anisotropic data is worse (sanity on the model).
    iso = ph.Phase(name="iso", lattice=tet.lattice,
                   eos={"type": "BM3", "K0": 180, "K0p": 4})
    res_iso = idf.fit_pressure_for_phase(obs, iso, refl, p_min=0, p_max=30, rel_tol=0.004)
    assert res["score"] >= res_iso["score"]


def test_sparse_observation_still_seen():
    """Regression: a DAC frame shows only a few of a phase's strong lines (the
    rest below the noise floor / overlapped). The phase must still register as
    'seen' with a finite pressure — the old confidence definition divided by
    EVERY predicted line, so a real phase scored ≈0 and reported P=nan."""
    if not ph.pymatgen_available():
        print("  (pymatgen not installed — skipping sparse-observation test)")
        return
    au = ph.load_bundled()["Au"]
    refl = idf.phase_reflections(au)
    d0, w, _ = refl
    p_true = 30.0
    s = idf.scale_at_pressure(au, p_true)
    # Observe only the 3 strongest reflections (sparse, like real DAC data).
    obs_d = d0[:3] * s
    res = idf.fit_pressure_for_phase(obs_d, au, refl, p_min=0.0, p_max=200.0,
                                     rel_tol=0.01)
    assert abs(res["pressure"] - p_true) < 4.0, res["pressure"]
    assert res["confidence"] > 0.3, f"sparse match should be seen, got {res['confidence']:.3f}"
    assert res["n_matched"] >= 3

    # End-to-end: the summary must surface it (n_frames_seen > 0, finite P).
    import h5py
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "analysis.h5"
        centers_q = 2 * np.pi / obs_d
        with h5py.File(str(h5), "w") as f:
            f.attrs["unit"] = "q_A^-1"
            f.attrs["source_reduced"] = "synthetic"
            gp = f.create_group("peaks")
            k = centers_q.size
            gp.create_dataset("counts", data=np.full(3, k, "i4"))
            gp.create_dataset("frame", data=np.repeat(np.arange(3), k).astype("i4"))
            gp.create_dataset("center", data=np.tile(centers_q, 3).astype("f8"))
            gp.create_dataset("flag", data=np.zeros(3 * k, "i4"))
        manifest = idf.run_identification(h5, [au], p_min=0.0, p_max=200.0)
        summ = manifest["summary"]["Au"]
        assert summ["n_frames_seen"] >= 1, summ
        assert summ["pressure_median"] == summ["pressure_median"], "pressure_median is NaN"


def main() -> None:
    test_radial_to_d()
    test_scale_monotonic()
    test_pressure_recovery()
    test_run_identification()
    test_skip_structureless_phase()
    test_axial_eos_anisotropic()
    test_sparse_observation_still_seen()
    print("IDENTIFY TEST OK")


if __name__ == "__main__":
    main()
