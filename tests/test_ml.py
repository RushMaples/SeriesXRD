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
        assert ff.clip_negative is True and ff.normalize == "max"
        assert ff.preprocessing()["source"] == "clean"
        # residual source resolves too
        ffr = mf.frame_features(h5, source="residual")
        assert ffr.source == "residual" and ffr.X.shape == (2, 3501)
        # every advertised source resolves (build_fit_source doesn't compose
        # robust/baseline/spot_residual — they must be read directly). _write_analysis
        # already wrote clean/baseline/spot_residual.
        for s in ("robust", "baseline", "spot_residual", "clean", "hybrid", "mean"):
            assert mf.frame_features(h5, source=s).X.shape == (2, 3501), s


def test_clip_negative():
    """The residual can go negative; clipping floors it so cosine ranking isn't
    distorted by holes the non-negative candidate fingerprints can't have."""
    import h5py
    q = np.linspace(1.0, 7.0, 400)
    clean = pseudo_voigt(q, 3.0, 100, 0.05, 0.5)[None, :].astype("f4")
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "an.h5"
        with h5py.File(str(h5), "w") as f:
            f.attrs["unit"] = "q_A^-1"; f.create_dataset("radial", data=q)
            gb = f.create_group("background"); gb.create_dataset("clean", data=clean)
            gp = f.create_group("peaks"); gp.attrs["source"] = "clean"
            gp.create_dataset("frame", data=np.array([0], "i4"))
            gp.create_dataset("flag", data=np.zeros(1, "i4"))
            f.create_group("frames").create_dataset("excluded", data=np.zeros(1, "?"))
            f.create_group("residual").create_dataset("clean", data=(clean - 30.0).astype("f4"))
        assert float(mf.frame_features(h5, source="residual", clip_negative=True).X.min()) >= 0.0
        assert float(mf.frame_features(h5, source="residual", clip_negative=False).X.min()) < 0.0


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
        assert man["requested_source"] == "auto" and man["resolved_source"] == "residual"
        assert "Au" in man["candidates"]
        rc = mr.read_candidates(h5)
        assert rc["ok"] and rc["n_frames"] == 2
        # Three-level source provenance persisted (requested -> rank level -> channel).
        assert rc["requested_source"] == "auto"
        assert rc["source"] == "residual" and rc["resolved_source"] == "residual"
        # 'fit' resolves to whatever Step 2 recorded (clean here) — a learned
        # model needs the resolved channel to reproduce the preprocessing.
        man_fit = mr.rank_candidates(h5, [au, decoy], reflections=refl, top_k=2,
                                     fwhm_d=0.05, source="fit")
        assert man_fit["requested_source"] == "fit"
        assert man_fit["resolved_source"] == "clean"
        rc_fit = mr.read_candidates(h5)
        assert rc_fit["source"] == "fit" and rc_fit["resolved_source"] == "clean"
        # restore the residual-ranked candidates for the assertions below
        man = mr.rank_candidates(h5, [au, decoy], reflections=refl, top_k=2, fwhm_d=0.05)
        rc = mr.read_candidates(h5)
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


def test_axial_ranking():
    """Regression: the ranker simulates with the anisotropic predicted_d, so an
    axial-only phase is matched at its true compressed positions — not frozen at
    ambient (which the old isotropic scale_at_pressure produced)."""
    import h5py
    Lc = {"a": 4.0, "b": 4.0, "c": 6.0, "alpha": 90, "beta": 90, "gamma": 90}
    H = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 0, 2], [2, 0, 0]], float)
    hkl = ["(1, 0, 0)", "(0, 0, 1)", "(1, 1, 0)", "(1, 0, 1)", "(0, 0, 2)", "(2, 0, 0)"]
    d0 = idf._d_from_lattice(H, Lc)
    w = np.ones(d0.size)
    tet = Phase(name="Tet", lattice=Lc,
                axial_eos={"a": {"type": "BM3", "K0": 300, "K0p": 4},
                           "c": {"type": "BM3", "K0": 100, "K0p": 4}})
    iso = Phase(name="Iso", lattice=Lc, eos={"type": "BM3", "K0": 180, "K0p": 4})
    refl = {"Tet": (d0, w, hkl), "Iso": (d0, w, hkl)}

    P = 20.0
    centers_d = idf.predicted_d(tet, d0, [idf._parse_hkl(h) for h in hkl], P)  # anisotropic obs
    q = np.linspace(1.0, 7.0, 1500)
    row = np.zeros(q.size)
    for c, a in zip(2 * np.pi / centers_d, w):
        if q[0] <= c <= q[-1]:
            row += pseudo_voigt(q, c, a * 100, 0.04, 0.5)
    stack = np.stack([row, row]).astype("f4")
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "an.h5"
        with h5py.File(str(h5), "w") as f:
            f.attrs["unit"] = "q_A^-1"; f.create_dataset("radial", data=q)
            gb = f.create_group("background"); gb.create_dataset("clean", data=stack)
            gp = f.create_group("peaks"); gp.attrs["source"] = "clean"
            gp.create_dataset("frame", data=np.array([0, 1], "i4"))
            gp.create_dataset("flag", data=np.zeros(2, "i4"))
            gf = f.create_group("frames")
            gf.create_dataset("pressure", data=np.array([P, P]))
            gf.create_dataset("excluded", data=np.zeros(2, "?"))
        man = mr.rank_candidates(h5, [tet, iso], reflections=refl, top_k=2, fwhm_d=0.04)
        rc = mr.read_candidates(h5)
        # The axial phase matches its anisotropically-compressed lines; the
        # isotropic competitor (same ambient lines) lands elsewhere at 20 GPa.
        assert rc["phases"]["Tet"]["score"][0] > 0.8, rc["phases"]["Tet"]["score"][0]
        assert rc["phases"]["Tet"]["score"][0] > rc["phases"]["Iso"]["score"][0]
        assert man["candidates"][0] if False else rc["topk_names"][0][0] == "Tet"


def test_worker_ml_rank_candidate_free():
    """--ml-rank / run_ml_rank must NOT require preselected candidates: it ranks
    the whole library and verifies the top-K."""
    import h5py
    from bulkxrd.analysis import worker as W
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "an.h5"
        with h5py.File(str(h5), "w") as f:
            f.attrs["unit"] = "q_A^-1"; f.attrs["source_reduced"] = "s"
            f.create_dataset("radial", data=np.linspace(1, 8, 10))
            f.create_group("background").create_dataset("clean", data=np.zeros((1, 10), "f4"))
            gp = f.create_group("peaks")
            for k in ("counts", "frame", "center", "flag"):
                gp.create_dataset(k, data=np.zeros(0 if k != "counts" else 1, "i4"))
            gf = f.create_group("frames")
            gf.create_dataset("pressure", data=np.array([20.0]))
            gf.create_dataset("excluded", data=np.zeros(1, "?"))
        lib = {"Au": Phase(name="Au", eos={"type": "BM3", "K0": 167, "K0p": 5.0}),
               "Re": Phase(name="Re")}
        saved = (W.load_library, W.pymatgen_available, W.rank_candidates,
                 W.run_identification, W.run_residual)
        cap = {}
        W.load_library = lambda ws: lib
        W.pymatgen_available = lambda: True
        W.rank_candidates = lambda path, pool, **k: {"candidates": ["Au"], "n_frames": 1}
        W.run_identification = lambda path, phases, **k: (
            cap.__setitem__("v", [p.name for p in phases])
            or {"out_h5": str(path), "summary": {}, "phases": [p.name for p in phases]})
        W.run_residual = lambda path, phases, **k: {"out_h5": str(path)}
        try:
            # No candidate_phases, identify_all off, run_ml_rank on -> must not raise.
            man = W.run_analysis({"analysis_h5_file": str(h5), "run_step1": False,
                                  "run_step2": False, "run_step3": True, "run_ml_rank": True})
            assert "ml_rank" in man["steps"] and cap["v"] == ["Au"]
            # And without ml-rank or candidates it still errors helpfully.
            try:
                W.run_analysis({"analysis_h5_file": str(h5), "run_step1": False,
                                "run_step2": False, "run_step3": True})
                assert False, "expected ValueError without candidates"
            except ValueError:
                pass
        finally:
            (W.load_library, W.pymatgen_available, W.rank_candidates,
             W.run_identification, W.run_residual) = saved


def test_scorer_seam():
    """The ml_scorer seam: default deterministic output is byte-identical with
    and without an explicitly injected CosineScorer; no torch is needed for the
    deterministic path; a learned scorer without its prerequisites raises a
    clear instructive error (never a bare ImportError crash); and candidate
    files written before the seam (no method/requested attrs) stay readable."""
    import h5py
    from bulkxrd.analysis import ml_scorer as msr

    # Deterministic path must not touch torch.
    assert "torch" not in sys.modules
    au = Phase(name="Au", eos={"type": "BM3", "K0": 167, "K0p": 5.0})
    decoy = Phase(name="Decoy", eos={"type": "BM3", "K0": 80, "K0p": 4.0})
    d0, w, hkl = _au_refl()
    refl = {"Au": (d0, w, hkl), "Decoy": (d0 * 1.12, w[::-1], hkl)}
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "an.h5"
        _write_analysis(h5, pressure=30.0)

        # 1) Default output unchanged by explicit seam use.
        mr.rank_candidates(h5, [au, decoy], reflections=refl, top_k=2, fwhm_d=0.05)
        rc_default = mr.read_candidates(h5)
        mr.rank_candidates(h5, [au, decoy], reflections=refl, top_k=2, fwhm_d=0.05,
                           scorer=msr.CosineScorer(fwhm_d=0.05))
        rc_injected = mr.read_candidates(h5)
        for nm in ("Au", "Decoy"):
            assert np.array_equal(rc_default["phases"][nm]["score"],
                                  rc_injected["phases"][nm]["score"]), nm
        assert rc_default["topk_names"] == rc_injected["topk_names"]
        with h5py.File(str(h5), "r") as f:
            assert str(f["ml/candidates"].attrs["method"]) == "cosine"

        # score_phase back-compat wrapper == the scorer it delegates to.
        m = mf.frame_features(h5, source="residual").X[0]
        grid = mf.frame_features(h5, source="residual").d_grid
        s_fn = mr.score_phase(m, au, refl["Au"], grid, 30.0, fwhm_d=0.05)
        s_cls = msr.CosineScorer(fwhm_d=0.05).score(m, au, refl["Au"], grid, 30.0)
        assert s_fn == s_cls

        # 4) Old files (pre-seam attrs) remain readable.
        with h5py.File(str(h5), "r+") as f:
            for a in ("method", "requested_source", "resolved_source"):
                if a in f["ml/candidates"].attrs:
                    del f["ml/candidates"].attrs[a]
        rc_old = mr.read_candidates(h5)
        assert rc_old["ok"] and rc_old["topk_names"] is not None
        assert rc_old["requested_source"] == rc_old["source"]  # fallback

    # 3) Learned scorer without prerequisites: instructive RuntimeError that
    #    points at the deterministic fallback — regardless of whether torch is
    #    installed (missing dep here; missing model file elsewhere).
    for bad in ("torch:/nonexistent/model.pt", "torch",
                {"kind": "torch"}, {"kind": "torch", "model": "/nonexistent/model.pt"}):
        try:
            msr.make_scorer(bad)
            assert False, f"expected RuntimeError for {bad!r}"
        except RuntimeError as e:
            assert "cosine" in str(e) or "model path" in str(e), str(e)
    try:
        msr.make_scorer("nonsense")
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert isinstance(msr.make_scorer(None), msr.CosineScorer)
    assert isinstance(msr.make_scorer("cosine"), msr.CosineScorer)


def main() -> None:
    test_frame_features()
    test_clip_negative()
    test_augmented_dataset()
    test_rank_candidates_and_shortlist()
    test_axial_ranking()
    test_worker_ml_rank_candidate_free()
    test_scorer_seam()
    print("ML TEST OK")


if __name__ == "__main__":
    main()
