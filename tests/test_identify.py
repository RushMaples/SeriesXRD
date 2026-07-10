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


# ---------------------------------------------------------------------------
# Pressure-aware identification + hardened scoring (no pymatgen — synthetic FCC
# reflections built straight from the cubic metric, like test_axial_eos above).
# ---------------------------------------------------------------------------

def _synth_au():
    """An FCC-gold-like phase + its (d, weight, hkl) reflection list, no pymatgen."""
    a0 = 4.078
    rows = [("111", 3, 100), ("200", 4, 46), ("220", 8, 26), ("311", 11, 28),
            ("222", 12, 8), ("400", 16, 4), ("331", 19, 12), ("420", 20, 12),
            ("422", 24, 10), ("511", 27, 8)]
    d0 = np.array([a0 / math.sqrt(s) for _, s, _ in rows])
    w = np.array([i for *_, i in rows], float)
    w /= w.max()
    hkl = ["(%s, %s, %s)" % (h[0], h[1], h[2]) for h, _, _ in rows]
    au = ph.Phase(name="Au", category="marker",
                  eos={"type": "BM3", "K0": 167, "K0p": 5.0}, space_group="Fm-3m",
                  lattice={"a": a0, "b": a0, "c": a0, "alpha": 90, "beta": 90, "gamma": 90},
                  atoms=[{"element": "Au", "x": 0, "y": 0, "z": 0, "occ": 1.0}])
    return au, (d0, w, hkl)


def test_reflection_d_min_tracks_data_qrange():
    """Regression: the reflection d_min must follow the reduction's actual q-range,
    not the fixed 1.0 Å fallback. A short-wavelength run reaching q≈11.3 Å⁻¹ must
    request reflections down to d≈0.55 Å, so a phase's higher-order lines (e.g.
    tungsten 321/400/422 — which barely shift because W hardly compresses) are
    modelled instead of surfacing as false "unknown" clusters."""
    import h5py
    au, au_refl = _synth_au()
    q = np.linspace(0.02, 11.3, 1500)               # short-λ axis: d_min ≈ 0.556 Å
    expected = 2 * np.pi / q.max()
    with tempfile.TemporaryDirectory() as td:
        h5p = Path(td) / "an.h5"
        k, n_frames = 4, 2
        centers = 2 * np.pi / au_refl[0][:k]        # ambient Au lines as q
        with h5py.File(str(h5p), "w") as h5:
            h5.attrs["unit"] = "q_A^-1"
            h5.attrs["source_reduced"] = "synthetic"
            h5.create_dataset("radial", data=q)
            gp = h5.create_group("peaks")
            gp.create_dataset("counts", data=np.full(n_frames, k, "i4"))
            gp.create_dataset("frame", data=np.repeat(np.arange(n_frames), k).astype("i4"))
            gp.create_dataset("center", data=np.tile(centers, n_frames).astype("f8"))
            gp.create_dataset("flag", data=np.zeros(n_frames * k, "i4"))

        seen = {}
        real_avail, real_refl = idf.pymatgen_available, idf.phase_reflections
        idf.pymatgen_available = lambda: True

        def spy(phase, **kw):
            seen["d_min"] = kw.get("d_min")
            seen["max_reflections"] = kw.get("max_reflections")
            return au_refl
        idf.phase_reflections = spy
        try:
            idf.run_identification(h5p, [au], p_min=0, p_max=200, rel_tol=0.01)
        finally:
            idf.pymatgen_available, idf.phase_reflections = real_avail, real_refl

        # run_identification derived d_min from the axis (well below the 1.0 fallback)
        assert seen["d_min"] is not None
        assert abs(seen["d_min"] - expected) < 0.02, seen["d_min"]
        assert seen["d_min"] < 0.6
        # ...and scaled the strongest-N cap up with the reciprocal-space volume
        assert seen["max_reflections"] > idf._REFL_CAP_BASE
        assert seen["max_reflections"] == int(min(400, max(
            idf._REFL_CAP_BASE, round(idf._REFL_CAP_BASE * (1.0 / seen["d_min"]) ** 3))))
        # ...and recorded both as provenance
        with h5py.File(str(h5p), "r") as h5:
            assert abs(float(h5["identify"].attrs["sim_d_min"]) - expected) < 0.02
            assert int(h5["identify"].attrs["sim_max_refl"]) == seen["max_reflections"]


def test_reflection_d_min_falls_back_without_axis():
    """No /radial axis (or a low-q one) → keep the conservative 1.0 Å fallback
    rather than truncating anything the data actually reaches."""
    import h5py
    au, au_refl = _synth_au()
    with tempfile.TemporaryDirectory() as td:
        h5p = Path(td) / "an.h5"
        k, n_frames = 4, 2
        centers = 2 * np.pi / au_refl[0][:k]
        with h5py.File(str(h5p), "w") as h5:
            h5.attrs["unit"] = "q_A^-1"
            h5.attrs["source_reduced"] = "synthetic"
            gp = h5.create_group("peaks")            # deliberately no /radial
            gp.create_dataset("counts", data=np.full(n_frames, k, "i4"))
            gp.create_dataset("frame", data=np.repeat(np.arange(n_frames), k).astype("i4"))
            gp.create_dataset("center", data=np.tile(centers, n_frames).astype("f8"))
            gp.create_dataset("flag", data=np.zeros(n_frames * k, "i4"))

        seen = {}
        real_avail, real_refl = idf.pymatgen_available, idf.phase_reflections
        idf.pymatgen_available = lambda: True

        def spy(phase, **kw):
            seen["d_min"] = kw.get("d_min")
            seen["max_reflections"] = kw.get("max_reflections")
            return au_refl
        idf.phase_reflections = spy
        try:
            idf.run_identification(h5p, [au], p_min=0, p_max=200, rel_tol=0.01)
        finally:
            idf.pymatgen_available, idf.phase_reflections = real_avail, real_refl

        assert seen["d_min"] == idf._SIM_D_MIN_DEFAULT
        assert seen["max_reflections"] == idf._REFL_CAP_BASE   # no range → no scale-up


def test_conservative_confidence():
    cc = idf.conservative_confidence
    # Old max(recall,precision)=1.0 here; the balanced+evidence form is < that.
    assert cc(1.0, 1.0, 10, min_matched=3) == 1.0
    assert 0.0 < cc(0.2, 1.0, 10, min_matched=3) < 0.5          # imbalanced -> low
    # Evidence penalty: too few matched reflections drags it down.
    assert cc(1.0, 1.0, 1, min_matched=3) < cc(1.0, 1.0, 3, min_matched=3)
    assert cc(1.0, 1.0, 3, min_matched=3) == 1.0
    # Pressure-prior penalty multiplies in (0,1].
    assert cc(1.0, 1.0, 5, min_matched=3, prior_penalty=0.5) == 0.5
    assert cc(0.0, 0.0, 0, min_matched=3) == 0.0


def test_one_to_one_matching():
    # Two predicted lines crowd one observed peak: only ONE may claim it.
    pred = np.array([2.000, 2.001])
    obs = np.array([2.0005])
    pairs = idf._match_pairs(pred, obs, rel_tol=0.01)
    assert len(pairs) == 1, pairs
    # Two observed, two predicted -> two distinct pairs.
    pairs2 = idf._match_pairs(np.array([2.0, 3.0]), np.array([2.0, 3.0]), rel_tol=0.01)
    assert len(pairs2) == 2 and {p[0] for p in pairs2} == {0, 1}


def test_esd_weighted_matching():
    """Per-peak center esd's widen the match tolerance in quadrature: a noisy
    peak just outside the model tolerance can still match; a sharp one (or one
    with no esd information) cannot."""
    pred = np.array([2.0])
    obs = np.array([2.05])          # gap 0.05; model tol = 2·(0.01·2.0) = 0.04
    assert not idf._match_pairs(pred, obs, rel_tol=0.01)
    # esd 0.02 -> tol = 2·hypot(0.02, 0.02) ≈ 0.057 > 0.05: now a match.
    pairs = idf._match_pairs(pred, obs, rel_tol=0.01, obs_err=np.array([0.02]))
    assert len(pairs) == 1
    # NaN esd = no information -> model tolerance alone again.
    assert not idf._match_pairs(pred, obs, rel_tol=0.01, obs_err=np.array([np.nan]))
    # The smooth kernel widens too: same gap scores higher with the esd.
    s_no = idf._score_pred(obs, pred, np.array([1.0]), rel_tol=0.01)[0]
    s_esd = idf._score_pred(obs, pred, np.array([1.0]), rel_tol=0.01,
                            obs_err_sorted=np.array([0.02]))[0]
    assert s_esd > s_no


def test_esd_conversion_to_d():
    """radial_err_to_d_err: q axes via σ_d = d·σ_q/q; 2θ via d·cotθ·σ_2θ/2."""
    q = np.array([2.0, 4.0])
    sq = np.array([0.01, 0.02])
    d = 2 * np.pi / q
    out = idf.radial_err_to_d_err(q, sq, "q_A^-1")
    assert np.allclose(out, d * sq / q)
    tt = np.array([10.0])            # degrees
    stt = np.array([0.05])
    lam = 0.4
    dd = idf.radial_to_d(tt, "2th_deg", lam)
    theta = np.radians(tt) / 2
    expect = dd / np.tan(theta) * np.radians(stt) / 2
    assert np.allclose(idf.radial_err_to_d_err(tt, stt, "2th_deg", lam), expect)
    # No wavelength on a 2θ axis, or non-positive errs -> NaN (no esd info).
    assert np.isnan(idf.radial_err_to_d_err(tt, stt, "2th_deg")).all()
    assert np.isnan(idf.radial_err_to_d_err(q, np.zeros(2), "q_A^-1")).all()


def test_intensity_agreement():
    """Observed amplitudes tracking the predicted relative intensities raise the
    (soft) intensity factor; anti-correlated amplitudes lower it. k=0 disables."""
    au, refl = _synth_au()
    d0, w, _ = refl
    obs = d0 * idf.scale_at_pressure(au, 20.0)
    good = idf.fit_pressure_for_phase(obs, au, refl, p_min=0, p_max=200,
                                      obs_amp=w * 50.0, intensity_k=0.3)
    bad = idf.fit_pressure_for_phase(obs, au, refl, p_min=0, p_max=200,
                                     obs_amp=50.0 / (w + 0.05), intensity_k=0.3)
    assert good["intensity_corr"] > 0.95
    assert good["intensity_corr"] > bad["intensity_corr"]
    assert good["confidence"] > bad["confidence"]
    # Positions are unaffected: intensity only nudges the confidence.
    assert abs(good["pressure"] - bad["pressure"]) < 0.5
    # k=0 (or no amplitudes at all) leaves the classic confidence untouched.
    off = idf.fit_pressure_for_phase(obs, au, refl, p_min=0, p_max=200,
                                     obs_amp=50.0 / (w + 0.05), intensity_k=0.0)
    plain = idf.fit_pressure_for_phase(obs, au, refl, p_min=0, p_max=200)
    assert off["confidence"] == plain["confidence"]
    assert plain["intensity_corr"] != plain["intensity_corr"]      # NaN: no amps
    # conservative_confidence folds the factor in gently and ignores NaN.
    cc = idf.conservative_confidence
    assert cc(1.0, 1.0, 5, min_matched=3, intensity_corr=0.5, intensity_k=0.3) == 0.85
    assert cc(1.0, 1.0, 5, min_matched=3, intensity_corr=float("nan"),
              intensity_k=0.3) == 1.0


def test_thermal_expansion_seam():
    """Phase.thermal moves predicted lines with temperature — the ambient-
    pressure temperature-series analog of the EOS."""
    au, refl = _synth_au()
    au.thermal = {"alpha_v": 4.2e-5, "T0": 298.0}
    d0, w, hkl = refl
    s_hot = ph.thermal_scale(au, 898.0)              # (1+4.2e-5·600)^(1/3)
    assert abs(s_hot - (1 + 4.2e-5 * 600) ** (1 / 3)) < 1e-12 and s_hot > 1.005
    assert ph.thermal_scale(au, None) == 1.0
    assert ph.thermal_scale(ph.Phase(name="X"), 898.0) == 1.0
    hkls = [idf._parse_hkl(h) for h in hkl]
    assert np.allclose(idf.predicted_d(au, d0, hkls, 0.0, 898.0), d0 * s_hot)
    # Hot, ambient-pressure observation: with T the fit stays at ~0 GPa and
    # matches; without it, expansion can't be reached by compression at all.
    obs = d0 * s_hot
    hot = idf.fit_pressure_for_phase(obs, au, refl, p_min=0, p_max=50,
                                     rel_tol=0.003, temperature=898.0)
    cold = idf.fit_pressure_for_phase(obs, au, refl, p_min=0, p_max=50,
                                      rel_tol=0.003)
    assert hot["confidence"] > 0.8 and hot["pressure"] < 1.0
    assert cold["confidence"] < hot["confidence"]


def test_pressure_prior_confines_search():
    au, refl = _synth_au()
    d0, _, _ = refl
    obs = d0 * idf.scale_at_pressure(au, 20.0)        # observed at 20 GPa
    free = idf.fit_pressure_for_phase(obs, au, refl, p_min=0, p_max=200, rel_tol=0.01)
    assert abs(free["pressure"] - 20.0) < 1.0 and free["confidence"] > 0.8
    good = idf.fit_pressure_for_phase(obs, au, refl, p_min=0, p_max=200, rel_tol=0.01,
                                      p_prior=20.0, p_window=2.0)
    assert abs(good["pressure"] - 20.0) < 1.0 and good["confidence"] > 0.8
    # A wrong prior confines the fit far from the real pressure -> confidence collapses.
    bad = idf.fit_pressure_for_phase(obs, au, refl, p_min=0, p_max=200, rel_tol=0.01,
                                     p_prior=80.0, p_window=2.0)
    assert abs(bad["pressure"] - 80.0) <= 2.0
    assert bad["confidence"] < 0.3, bad["confidence"]


def test_ignore_prior_frees_search_and_penalty():
    """pressure_assumption='ignore_prior' exempts a phase from the prior
    ENTIRELY: the pressure search covers the full [p_min, p_max] (not just
    prior ± window) and no Gaussian penalty applies. This is the second-marker
    seam — gasket-flank / anvil-bridged marker metal sits tens of GPa away from
    the chamber pressure, and a prior-confined search could never reach it."""
    import dataclasses
    au, refl = _synth_au()
    d0, _, _ = refl
    obs = d0 * idf.scale_at_pressure(au, 20.0)        # material really at 20 GPa
    exempt = dataclasses.replace(au, name="Au (gradient)",
                                 pressure_assumption="ignore_prior")
    # Chamber prior says 80 GPa; the exempt phase must still find 20 GPa, at
    # full confidence (no prior penalty).
    got = idf.fit_pressure_for_phase(obs, exempt, refl, p_min=0, p_max=200,
                                     rel_tol=0.01, p_prior=80.0, p_window=2.0)
    assert abs(got["pressure"] - 20.0) < 1.0, got["pressure"]
    assert got["confidence"] > 0.8, got["confidence"]
    assert got["prior_penalty"] == 1.0, got["prior_penalty"]


def _run_identification_synthetic(tmp_h5, phases, refl_map, p_true, **kw):
    """Drive run_identification without pymatgen by patching the simulation."""
    import h5py
    real_avail, real_refl = idf.pymatgen_available, idf.phase_reflections
    idf.pymatgen_available = lambda: True
    idf.phase_reflections = lambda phase, **k: refl_map[phase.name]
    try:
        return idf.run_identification(tmp_h5, phases, p_min=0, p_max=200,
                                      rel_tol=0.01, **kw)
    finally:
        idf.pymatgen_available, idf.phase_reflections = real_avail, real_refl


def test_pressure_prior_rejects_decoy_end_to_end():
    """Metadata pressure on /frames rejects a decoy phase that, given free
    pressure, can otherwise pair up enough lines to look present."""
    import h5py
    au, au_refl = _synth_au()
    d0 = au_refl[0]
    decoy = ph.Phase(name="Decoy", category="sample",
                     eos={"type": "BM3", "K0": 80, "K0p": 4.0}, space_group="Fm-3m",
                     lattice={"a": 4.5, "b": 4.5, "c": 4.5, "alpha": 90, "beta": 90, "gamma": 90},
                     atoms=[{"element": "Si", "x": 0, "y": 0, "z": 0, "occ": 1.0}])
    decoy_refl = (np.array([4.5 / math.sqrt(s) for s in (3, 4, 8, 11, 12)]),
                  np.array([1., .8, .6, .5, .3]),
                  ["(1, 1, 1)", "(2, 0, 0)", "(2, 2, 0)", "(3, 1, 1)", "(2, 2, 2)"])
    P = [10.0, 20.0, 30.0]
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "an.h5"
        allq, frame, counts = [], [], []
        for i, p in enumerate(P):
            q = 2 * np.pi / (d0 * idf.scale_at_pressure(au, p))
            allq.extend(q); frame.extend([i] * len(q)); counts.append(len(q))
        with h5py.File(str(h5), "w") as f:
            f.attrs["unit"] = "q_A^-1"; f.attrs["source_reduced"] = "syn"
            gp = f.create_group("peaks")
            gp.create_dataset("counts", data=np.array(counts, "i4"))
            gp.create_dataset("frame", data=np.array(frame, "i4"))
            gp.create_dataset("center", data=np.array(allq, "f8"))
            gp.create_dataset("flag", data=np.zeros(len(allq), "i4"))
            gf = f.create_group("frames")
            gf.create_dataset("pressure", data=np.array(P))        # exact metadata prior
            gf.create_dataset("excluded", data=np.zeros(len(P), "?"))
        man = _run_identification_synthetic(
            h5, [au, decoy], {"Au": au_refl, "Decoy": decoy_refl}, P, pressure_window=2.0)
        assert man["summary"]["Au"]["n_frames_seen"] == 3
        assert man["summary"]["Decoy"]["n_frames_seen"] == 0, man["summary"]["Decoy"]
        with h5py.File(str(h5), "r") as f:
            assert np.allclose(np.round(f["identify/Au/pressure"][:]), P)
            assert f["identify"].attrs["n_pressure_prior"] == 3
            assert np.max(f["identify/Decoy/confidence"][:]) < 0.5

        # The summary's "seen" threshold follows the caller/GUI value rather
        # than a hard-coded 0.5. A threshold of 1.0 excludes every finite
        # confidence because the seen test is strict (> threshold).
        with h5py.File(str(h5), "r+") as f:
            if "identify" in f:
                del f["identify"]
        strict = _run_identification_synthetic(
            h5, [au], {"Au": au_refl}, P, pressure_window=2.0, seen_conf=1.0)
        assert strict["summary"]["Au"]["n_frames_seen"] == 0
        with h5py.File(str(h5), "r") as f:
            assert float(f["identify"].attrs["seen_conf"]) == 1.0

        # marker_prior path: same data, but no metadata pressure -> derive from Au.
        with h5py.File(str(h5), "r+") as f:
            f["frames/pressure"][...] = np.nan
            if "identify" in f:
                del f["identify"]
        man2 = _run_identification_synthetic(
            h5, [au, decoy], {"Au": au_refl, "Decoy": decoy_refl}, P,
            marker_prior=True, pressure_window=2.0)
        assert man2["summary"]["Au"]["n_frames_seen"] == 3
        assert man2["summary"]["Decoy"]["n_frames_seen"] == 0


def test_no_eos_penalized_and_range_auto_widens():
    """(a) A no-EOS phase (scored at ambient) is penalised on a high-pressure
    frame. (b) A prior outside [p_min, p_max] auto-widens the search so an
    otherwise-correct phase is still recovered instead of clamped to the edge."""
    import h5py
    au, au_refl = _synth_au()
    d0 = au_refl[0]
    no_eos = ph.Phase(name="NoEOS", category="sample", space_group="Fm-3m",
                      lattice=au.lattice, atoms=au.atoms)   # same lines, no EOS
    refl_map = {"Au": au_refl, "NoEOS": au_refl}

    def _one_frame_h5(path, p_true, prior):
        q = 2 * np.pi / (d0 * idf.scale_at_pressure(au, p_true))
        with h5py.File(str(path), "w") as f:
            f.attrs["unit"] = "q_A^-1"; f.attrs["source_reduced"] = "syn"
            gp = f.create_group("peaks")
            gp.create_dataset("counts", data=np.array([q.size], "i4"))
            gp.create_dataset("frame", data=np.zeros(q.size, "i4"))
            gp.create_dataset("center", data=q.astype("f8"))
            gp.create_dataset("flag", data=np.zeros(q.size, "i4"))
            gf = f.create_group("frames")
            gf.create_dataset("pressure", data=np.array([prior], "f8"))
            gf.create_dataset("excluded", data=np.zeros(1, "?"))

    with tempfile.TemporaryDirectory() as td:
        # (b-style penalty) frame genuinely at 30 GPa, prior 30.
        h = Path(td) / "noeos.h5"
        _one_frame_h5(h, p_true=30.0, prior=30.0)
        _run_identification_synthetic(h, [au, no_eos], refl_map, [30.0], pressure_window=2.0)
        with h5py.File(str(h), "r") as f:
            assert f["identify/Au/confidence"][0] > 0.8
            assert f["identify/NoEOS/confidence"][0] < 0.2, "no-EOS phase not penalised at 30 GPa"

        # (a) prior 150 GPa with p_max 100 -> auto-widen and recover.
        h2 = Path(td) / "oor.h5"
        _one_frame_h5(h2, p_true=150.0, prior=150.0)
        real_avail, real_refl = idf.pymatgen_available, idf.phase_reflections
        idf.pymatgen_available = lambda: True
        idf.phase_reflections = lambda phase, **k: refl_map[phase.name]
        try:
            idf.run_identification(h2, [au], p_min=0.0, p_max=100.0, rel_tol=0.01,
                                   pressure_window=2.0)
        finally:
            idf.pymatgen_available, idf.phase_reflections = real_avail, real_refl
        with h5py.File(str(h2), "r") as f:
            assert abs(f["identify/Au/pressure"][0] - 150.0) < 2.0, f["identify/Au/pressure"][0]
            assert f["identify/Au/confidence"][0] > 0.8
            assert f["identify"].attrs["p_max"] >= 150.0, "range not widened to cover prior"


def test_pressure_model_and_penalty_surfaced():
    """pressure_model (eos|axial_eos|no_eos), prior_penalty per frame, and
    prior_penalized are surfaced on /identify, the summary, and review.identify_tracks."""
    import h5py
    from bulkxrd.analysis.review import identify_tracks
    au, au_refl = _synth_au()
    d0 = au_refl[0]
    no_eos = ph.Phase(name="NoEOS", category="sample", space_group="Fm-3m",
                      lattice=au.lattice, atoms=au.atoms)
    tet = ph.Phase(name="Tet", lattice={"a": 4, "b": 4, "c": 6, "alpha": 90, "beta": 90, "gamma": 90},
                   axial_eos={"a": {"type": "BM3", "K0": 300, "K0p": 4},
                              "c": {"type": "BM3", "K0": 100, "K0p": 4}})
    assert idf.pressure_model(au) == "eos"
    assert idf.pressure_model(no_eos) == "no_eos"
    assert idf.pressure_model(tet) == "axial_eos"

    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "an.h5"
        q = 2 * np.pi / (d0 * idf.scale_at_pressure(au, 30.0))
        with h5py.File(str(h5), "w") as f:
            f.attrs["unit"] = "q_A^-1"; f.attrs["source_reduced"] = "syn"
            gp = f.create_group("peaks")
            gp.create_dataset("counts", data=np.array([q.size], "i4"))
            gp.create_dataset("frame", data=np.zeros(q.size, "i4"))
            gp.create_dataset("center", data=q.astype("f8"))
            gp.create_dataset("flag", data=np.zeros(q.size, "i4"))
            gf = f.create_group("frames")
            gf.create_dataset("pressure", data=np.array([30.0]))      # high-pressure frame
            gf.create_dataset("excluded", data=np.zeros(1, "?"))
        man = _run_identification_synthetic(h5, [au, no_eos],
                                            {"Au": au_refl, "NoEOS": au_refl}, [30.0],
                                            pressure_window=2.0)
        # Summary carries the model + penalty flags.
        assert man["summary"]["Au"]["pressure_model"] == "eos"
        assert man["summary"]["Au"]["prior_penalized"] is False
        assert man["summary"]["NoEOS"]["pressure_model"] == "no_eos"
        assert man["summary"]["NoEOS"]["prior_penalized"] is True
        assert man["summary"]["NoEOS"]["n_frames_penalized"] == 1
        # HDF5 attrs + per-frame prior_penalty dataset.
        with h5py.File(str(h5), "r") as f:
            assert str(f["identify/NoEOS"].attrs["pressure_model"]) == "no_eos"
            assert bool(f["identify/NoEOS"].attrs["prior_penalized"]) is True
            assert f["identify/NoEOS/prior_penalty"][0] < 0.1        # 0 GPa vs 30 GPa prior
            assert f["identify/Au/prior_penalty"][0] > 0.99
        # review surfaces them for the GUI.
        tr = identify_tracks(h5)
        by = {r["name"]: r for r in tr["phases"]}
        assert by["NoEOS"]["pressure_model"] == "no_eos" and by["NoEOS"]["prior_penalized"]
        assert by["Au"]["pressure_model"] == "eos"


def test_pressure_assumption_ignore_prior():
    """pressure_model renamed to no_eos; pressure_assumption resolves; a phase
    flagged ignore_prior is exempt from the pressure-prior penalty."""
    au, refl = _synth_au()
    no_eos = ph.Phase(name="NoEOS")
    ignore = ph.Phase(name="Ref", pressure_assumption="ignore_prior")
    assert idf.pressure_model(au) == "eos" and idf.pressure_model(no_eos) == "no_eos"
    assert idf.pressure_assumption(au) == "eos_based"
    assert idf.pressure_assumption(no_eos) == "eos_missing"          # default for no-EOS
    assert idf.pressure_assumption(ignore) == "ignore_prior"

    obs = refl[0]                                       # ambient lines, frame prior = 30
    pen = idf.fit_pressure_for_phase(obs, no_eos, refl, p_prior=30.0, p_window=2.0)
    ign = idf.fit_pressure_for_phase(obs, ignore, refl, p_prior=30.0, p_window=2.0)
    assert pen["confidence"] < 0.2 and pen["prior_penalty"] < 0.1   # penalised
    assert ign["confidence"] > 0.8 and ign["prior_penalty"] == 1.0  # exempt


def main() -> None:
    test_radial_to_d()
    test_scale_monotonic()
    test_pressure_recovery()
    test_run_identification()
    test_skip_structureless_phase()
    test_axial_eos_anisotropic()
    test_sparse_observation_still_seen()
    test_conservative_confidence()
    test_one_to_one_matching()
    test_esd_weighted_matching()
    test_esd_conversion_to_d()
    test_intensity_agreement()
    test_thermal_expansion_seam()
    test_pressure_prior_confines_search()
    test_pressure_prior_rejects_decoy_end_to_end()
    test_no_eos_penalized_and_range_auto_widens()
    test_pressure_model_and_penalty_surfaced()
    test_pressure_assumption_ignore_prior()
    print("IDENTIFY TEST OK")


if __name__ == "__main__":
    main()
