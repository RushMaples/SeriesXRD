"""Step 3a-removal: phase subtraction → residual → unexplained re-detection."""
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.analysis.residual import (
    attribute_peaks, subtract_peaks, run_residual)
from bulkxrd.analysis.peaks import pseudo_voigt
from bulkxrd.analysis.phases import Phase


def _test_attribute_peaks():
    obs = np.array([2.094, 1.257, 3.5])           # d-spacings
    preds = {"A": np.array([2.090, 5.0]),         # matches obs[0]
             "B": np.array([1.258])}              # matches obs[1]
    labels, explained = attribute_peaks(obs, preds, rel_tol=0.01)
    assert labels == ["A", "B", ""], labels
    assert explained.tolist() == [True, True, False], explained
    # tighter tolerance drops the looser match
    _, exp2 = attribute_peaks(obs, preds, rel_tol=0.001)
    assert exp2.tolist() == [False, True, False], exp2


def _test_subtract_peaks():
    x = np.linspace(2.0, 8.0, 2000)
    clean = pseudo_voigt(x, 3.0, 100.0, 0.05, 0.5) + pseudo_voigt(x, 5.0, 40.0, 0.05, 0.5)
    keep = np.array([True, False])                # remove only the first peak
    res = subtract_peaks(x, clean, [3.0, 5.0], [100.0, 40.0],
                         [0.05, 0.05], [0.5, 0.5], keep)
    near3 = np.abs(x - 3.0) < 0.02
    near5 = np.abs(x - 5.0) < 0.02
    assert np.max(np.abs(res[near3])) < 1.0, "explained peak not removed"
    assert np.max(res[near5]) > 30.0, "unexplained peak should remain"


def _test_run_residual_end_to_end():
    import h5py
    from bulkxrd.analysis.identify import _h5_safe

    x = np.linspace(2.0, 8.0, 2000)               # q (Å^-1)
    clean = (pseudo_voigt(x, 3.0, 100.0, 0.05, 0.5)
             + pseudo_voigt(x, 5.0, 40.0, 0.05, 0.5))[None, :].astype("f4")
    # obs d-spacings: d = 2π/q  → q=3 → 2.0944, q=5 → 1.2566
    phase = Phase(name="KnownX")                  # no EOS → ambient predictions
    refl_d = np.array([2.0 * np.pi / 3.0])        # matches the q=3 peak only

    with tempfile.TemporaryDirectory() as td:
        h5path = Path(td) / "analysis.h5"
        with h5py.File(h5path, "w") as o:
            o.attrs["unit"] = "q_A^-1"
            o.create_dataset("radial", data=x)
            o.create_group("background").create_dataset("clean", data=clean)
            gp = o.create_group("peaks")
            gp.create_dataset("counts", data=np.array([2], "i4"))
            gp.create_dataset("frame", data=np.array([0, 0], "i4"))
            gp.create_dataset("center", data=np.array([3.0, 5.0], "f8"))
            gp.create_dataset("amplitude", data=np.array([100.0, 40.0], "f8"))
            gp.create_dataset("fwhm", data=np.array([0.05, 0.05], "f8"))
            gp.create_dataset("eta", data=np.array([0.5, 0.5], "f8"))
            gp.create_dataset("flag", data=np.array([0, 0], "i4"))
            idg = o.create_group("identify")
            idg.attrs["wavelength"] = 0.0
            g = idg.create_group(_h5_safe("KnownX"))
            g.create_dataset("confidence", data=np.array([1.0], "f8"))
            g.create_dataset("pressure", data=np.array([0.0], "f8"))
            g.create_dataset("refl_d", data=refl_d.astype("f8"))
            g.create_dataset("refl_hkl",
                             data=np.array(["(1, 1, 1)"], dtype=object),
                             dtype=h5py.string_dtype(encoding="utf-8"))

        m = run_residual(h5path, [phase], seen_conf=0.5, rel_tol=0.01, min_snr=5.0)
        assert m["n_explained"] == 1 and m["n_unexplained"] == 1, m

        with h5py.File(h5path, "r") as h5:
            res = np.asarray(h5["residual/clean"][0], float)
            phase_lbl = [s.decode() if isinstance(s, bytes) else str(s)
                         for s in h5["peaks/phase"][:]]
            ec = int(h5["residual/explained_counts"][0])
            uc = int(h5["residual/unexplained_counts"][0])
            rd_center = np.asarray(h5["residual/peaks/center"][:], float)
        # the q=3 peak is gone from the residual, the q=5 peak remains
        assert np.max(np.abs(res[np.abs(x - 3.0) < 0.02])) < 1.0, "explained not removed"
        assert np.max(res[np.abs(x - 5.0) < 0.02]) > 30.0, "unexplained removed in error"
        assert phase_lbl == ["KnownX", ""], phase_lbl
        assert ec == 1 and uc == 1
        # re-detection finds the surviving unexplained peak near q=5, not q=3
        assert any(abs(c - 5.0) < 0.05 for c in rd_center), rd_center
        assert not any(abs(c - 3.0) < 0.05 for c in rd_center), rd_center


def _build_residual_h5(path, n_matched):
    """One-frame analysis HDF5 whose /identify says KnownX is confident but only
    matched ``n_matched`` reflections (to exercise the evidence gate)."""
    import h5py
    from bulkxrd.analysis.identify import _h5_safe
    x = np.linspace(2.0, 8.0, 2000)
    clean = (pseudo_voigt(x, 3.0, 100.0, 0.05, 0.5)
             + pseudo_voigt(x, 5.0, 40.0, 0.05, 0.5))[None, :].astype("f4")
    with h5py.File(str(path), "w") as o:
        o.attrs["unit"] = "q_A^-1"
        o.create_dataset("radial", data=x)
        o.create_group("background").create_dataset("clean", data=clean)
        gp = o.create_group("peaks")
        gp.create_dataset("counts", data=np.array([2], "i4"))
        gp.create_dataset("frame", data=np.array([0, 0], "i4"))
        gp.create_dataset("center", data=np.array([3.0, 5.0], "f8"))
        gp.create_dataset("amplitude", data=np.array([100.0, 40.0], "f8"))
        gp.create_dataset("fwhm", data=np.array([0.05, 0.05], "f8"))
        gp.create_dataset("eta", data=np.array([0.5, 0.5], "f8"))
        gp.create_dataset("flag", data=np.array([0, 0], "i4"))
        idg = o.create_group("identify"); idg.attrs["wavelength"] = 0.0
        g = idg.create_group(_h5_safe("KnownX"))
        g.create_dataset("confidence", data=np.array([1.0], "f8"))
        g.create_dataset("pressure", data=np.array([0.0], "f8"))
        g.create_dataset("n_matched", data=np.array([n_matched], "i4"))
        g.create_dataset("refl_d", data=np.array([2.0 * np.pi / 3.0]))
        g.create_dataset("refl_hkl", data=np.array(["(1, 1, 1)"], dtype=object),
                         dtype=h5py.string_dtype(encoding="utf-8"))


def _test_evidence_gate():
    """A confident phase is only subtracted when it has >= min_matched matched
    reflections — unless allow_sparse overrides."""
    phase = Phase(name="KnownX")
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "an.h5"
        # n_matched=1 < min_matched=3 -> NOT subtracted
        _build_residual_h5(h5, n_matched=1)
        m = run_residual(h5, [phase], seen_conf=0.5, rel_tol=0.01, min_snr=5.0, min_matched=3)
        assert m["n_explained"] == 0, m
        # allow_sparse -> subtracted despite sparse evidence
        _build_residual_h5(h5, n_matched=1)
        m = run_residual(h5, [phase], seen_conf=0.5, rel_tol=0.01, min_snr=5.0,
                         min_matched=3, allow_sparse=True)
        assert m["n_explained"] == 1, m
        # n_matched=3 >= 3 -> subtracted
        _build_residual_h5(h5, n_matched=3)
        m = run_residual(h5, [phase], seen_conf=0.5, rel_tol=0.01, min_snr=5.0, min_matched=3)
        assert m["n_explained"] == 1, m


def _test_attribute_one_to_one():
    """One predicted line must not explain a whole cluster of observed peaks —
    only the nearest; the rest survive into the residual (split/overlap/unknown)."""
    obs = np.array([2.090, 2.095, 1.257])       # two peaks straddle one A line
    preds = {"A": np.array([2.091]), "B": np.array([1.258])}
    labels, explained = attribute_peaks(obs, preds, rel_tol=0.01)
    assert explained.tolist() == [True, False, True], (labels, explained)
    assert labels == ["A", "", "B"], labels
    # The nearer of the two (2.090, gap .001) wins; the farther (2.095) survives.


def _test_residual_uses_fit_source():
    """Peaks fit on sigmaclip must be subtracted from sigmaclip, not the median
    clean — otherwise an amplitude fit on the (taller) trimmed mean digs a deep
    negative hole in clean."""
    import h5py
    from bulkxrd.analysis.identify import _h5_safe
    x = np.linspace(2.0, 8.0, 2000)              # q (Å^-1)
    clean = pseudo_voigt(x, 3.0, 50.0, 0.05, 0.5)[None, :].astype("f4")
    sigmaclip = pseudo_voigt(x, 3.0, 100.0, 0.05, 0.5)[None, :].astype("f4")
    sc_resid = (sigmaclip - clean).astype("f4")  # sigmaclip = clean + residual
    phase = Phase(name="KnownX")
    with tempfile.TemporaryDirectory() as td:
        h5p = Path(td) / "an.h5"
        with h5py.File(h5p, "w") as o:
            o.attrs["unit"] = "q_A^-1"
            o.create_dataset("radial", data=x)
            gb = o.create_group("background")
            gb.create_dataset("clean", data=clean)
            gb.create_dataset("sigmaclip_residual", data=sc_resid)
            gp = o.create_group("peaks")
            gp.attrs["source"] = "sigmaclip"         # peaks were fit on sigmaclip
            gp.create_dataset("counts", data=np.array([1], "i4"))
            gp.create_dataset("frame", data=np.array([0], "i4"))
            gp.create_dataset("center", data=np.array([3.0], "f8"))
            gp.create_dataset("amplitude", data=np.array([100.0], "f8"))  # fit on sigmaclip
            gp.create_dataset("fwhm", data=np.array([0.05], "f8"))
            gp.create_dataset("eta", data=np.array([0.5], "f8"))
            gp.create_dataset("flag", data=np.array([0], "i4"))
            idg = o.create_group("identify"); idg.attrs["wavelength"] = 0.0
            g = idg.create_group(_h5_safe("KnownX"))
            g.create_dataset("confidence", data=np.array([1.0], "f8"))
            g.create_dataset("pressure", data=np.array([0.0], "f8"))
            g.create_dataset("n_matched", data=np.array([5], "i4"))
            g.create_dataset("refl_d", data=np.array([2.0 * np.pi / 3.0]))
            g.create_dataset("refl_hkl", data=np.array(["(1, 1, 1)"], dtype=object),
                             dtype=h5py.string_dtype(encoding="utf-8"))
        m = run_residual(h5p, [phase], seen_conf=0.5, rel_tol=0.01, min_snr=5.0, min_matched=3)
        assert m["fit_source"] == "sigmaclip", m
        with h5py.File(h5p, "r") as h5:
            res = np.asarray(h5["residual/clean"][0], float)
            assert str(h5["residual"].attrs["source"]) == "sigmaclip"
        near3 = np.abs(x - 3.0) < 0.02
        # Subtracting amp-100 from sigmaclip(=100) leaves ~0; subtracting from
        # clean(=50) would leave a ~-50 hole. Assert no deep negative hole.
        assert res[near3].min() > -5.0, f"negative hole -> wrong base: min={res[near3].min()}"
        assert np.max(np.abs(res[near3])) < 5.0, "explained peak not cleanly removed"


def _test_residual_refit_uses_peak_window():
    """Residual re-detection inherits Step-2's fit window from /peaks attrs."""
    import h5py
    from bulkxrd.analysis.identify import _h5_safe
    x = np.linspace(2.0, 8.0, 2000)
    clean = (pseudo_voigt(x, 3.0, 100.0, 0.05, 0.5)
             + pseudo_voigt(x, 7.0, 80.0, 0.05, 0.5))[None, :].astype("f4")
    phase = Phase(name="KnownX")
    with tempfile.TemporaryDirectory() as td:
        h5p = Path(td) / "an.h5"
        with h5py.File(h5p, "w") as o:
            o.attrs["unit"] = "q_A^-1"
            o.create_dataset("radial", data=x)
            o.create_group("background").create_dataset("clean", data=clean)
            gp = o.create_group("peaks")
            gp.attrs["fit_max"] = 6.0
            gp.attrs["min_snr"] = 5.0
            gp.attrs["min_prominence_snr"] = 2.0
            gp.attrs["edge_bins"] = 5
            gp.attrs["min_fwhm_bins"] = 2.0
            gp.attrs["local_baseline_bins"] = 81
            gp.create_dataset("counts", data=np.array([2], "i4"))
            gp.create_dataset("frame", data=np.array([0, 0], "i4"))
            gp.create_dataset("center", data=np.array([3.0, 7.0], "f8"))
            gp.create_dataset("amplitude", data=np.array([100.0, 80.0], "f8"))
            gp.create_dataset("fwhm", data=np.array([0.05, 0.05], "f8"))
            gp.create_dataset("eta", data=np.array([0.5, 0.5], "f8"))
            gp.create_dataset("flag", data=np.array([0, 0], "i4"))
            idg = o.create_group("identify"); idg.attrs["wavelength"] = 0.0
            g = idg.create_group(_h5_safe("KnownX"))
            g.create_dataset("confidence", data=np.array([1.0], "f8"))
            g.create_dataset("pressure", data=np.array([0.0], "f8"))
            g.create_dataset("n_matched", data=np.array([3], "i4"))
            g.create_dataset("refl_d", data=np.array([2.0 * np.pi / 3.0]))
            g.create_dataset("refl_hkl", data=np.array(["(1, 1, 1)"], dtype=object),
                             dtype=h5py.string_dtype(encoding="utf-8"))
        m = run_residual(h5p, [phase], seen_conf=0.5, rel_tol=0.01, min_matched=3)
        assert m["fit_max"] == 6.0, m
        assert m["local_baseline_bins"] == 0, m
        with h5py.File(h5p, "r") as h5:
            assert float(h5["residual"].attrs["fit_max"]) == 6.0
            assert int(h5["residual"].attrs["local_baseline_bins"]) == 0
            centers = np.asarray(h5["residual/peaks/center"][:], float)
        assert not any(abs(c - 7.0) < 0.1 for c in centers), centers


def test_residual_suite() -> None:
    _test_attribute_peaks()
    _test_attribute_one_to_one()
    _test_subtract_peaks()
    _test_run_residual_end_to_end()
    _test_evidence_gate()
    _test_residual_uses_fit_source()
    _test_residual_refit_uses_peak_window()


def main() -> None:
    test_residual_suite()
    print("RESIDUAL TEST OK")


if __name__ == "__main__":
    main()
