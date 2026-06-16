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


def main() -> None:
    test_radial_to_d()
    test_scale_monotonic()
    test_pressure_recovery()
    test_run_identification()
    print("IDENTIFY TEST OK")


if __name__ == "__main__":
    main()
