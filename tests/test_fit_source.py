"""Step-1→Step-2 selectable-fit-source pipeline (numpy + scipy + h5py).

Verifies that the reduce-side azimuthal sigma-clip channel is carried into the
analysis file as ``/background/sigmaclip_residual`` and that Step-2 can fit on a
``hybrid``/``sigmaclip`` source which recovers an azimuthally-sparse real peak
that the conservative median-based ``clean`` drops.
"""
import sys, shutil, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
from bulkxrd.analysis.peaks import pseudo_voigt, run_peak_fitting
from bulkxrd.analysis.background import run_background_separation
from bulkxrd.analysis.review import frame_data
from bulkxrd.analysis.heatmap import pattern_image


def _write_reduced(path, q, mean_s, robust_s, sigma_s):
    n = mean_s.shape[0]
    with h5py.File(path, "w") as h:
        h.attrs["unit"] = "q_A^-1"
        h.attrs["poni_text"] = "wavelength: 2.0e-11"
        gp = h.create_group("patterns")
        gp.create_dataset("intensity", data=mean_s.astype("f4"))
        gp.create_dataset("intensity_robust", data=robust_s.astype("f4"))
        if sigma_s is not None:
            gp.create_dataset("intensity_sigmaclip", data=sigma_s.astype("f4"))
        gp.create_dataset("radial", data=q)
        gf = h.create_group("frames")
        gf.create_dataset("filename",
                          data=np.array([f"f{i}.tif" for i in range(n)], dtype=object),
                          dtype=h5py.string_dtype("utf-8"))
        gf.create_dataset("excluded", data=np.zeros(n, "?"))


def _has_good_peak(path, center, tol=0.08):
    fd = frame_data(path, 0)
    return any(abs(p["center"] - center) < tol and p["flag"] == 0 for p in fd["peaks"])


def main() -> None:
    rng = np.random.default_rng(3)
    q = np.linspace(0.0, 6.27, 1500)
    N, nb = 4, q.size
    bg = 80 + 40 * np.exp(-q / 2.0)
    powder = (pseudo_voigt(q, 2.2, 300, 0.05, 0.4)
              + pseudo_voigt(q, 3.6, 240, 0.05, 0.4)
              + pseudo_voigt(q, 5.0, 360, 0.045, 0.3))
    spotty = pseudo_voigt(q, 4.3, 90, 0.06, 0.5)            # azimuthally-sparse REAL peak
    diamond = np.zeros_like(q); diamond[np.argmin(np.abs(q - 4.95))] += 4000.0
    robust = bg + powder                                   # median: drops spotty + diamond
    sigmaclip = robust + spotty                            # trimmed mean: keeps the real ring
    mean = sigmaclip + diamond                             # mean: also has the diamond spot
    mean_s = np.array([mean + rng.normal(0, 1.5, nb) for _ in range(N)])
    robust_s = np.array([robust + rng.normal(0, 1.5, nb) for _ in range(N)])
    sigma_s = np.array([sigmaclip + rng.normal(0, 1.5, nb) for _ in range(N)])

    td = Path(tempfile.mkdtemp())
    red = td / "reduced.h5"
    _write_reduced(red, q, mean_s, robust_s, sigma_s)

    # Step 1 carries the sigma-clip channel through as a residual on the median.
    m1 = run_background_separation(red, td / "an.h5", max_half_window=40, n_passes=1)
    ana = Path(m1["out_h5"])
    assert m1["has_sigmaclip"] is True
    with h5py.File(ana, "r") as h:
        assert "sigmaclip_residual" in h["background"]
        assert bool(h.attrs["has_sigmaclip"]) is True
        kreal = np.argmin(np.abs(q - 4.3))
        assert h["background/sigmaclip_residual"][0][kreal] > 0.5 * spotty[kreal]

    # Source selection: clean misses the sparse real peak, the rest recover it;
    # all keep the strong powder peak.
    seen = {}
    for src in ("clean", "hybrid", "sigmaclip", "auto"):
        a = td / f"an_{src}.h5"; shutil.copy2(ana, a)
        run_peak_fitting(a, None, source=src, sensitivity="normal", auto_range=True)
        seen[src] = (_has_good_peak(a, 3.6), _has_good_peak(a, 4.3))
        assert seen[src][0], f"{src}: lost the strong powder peak @3.6"
    assert seen["clean"][1] is False, "clean should miss the azimuthally-sparse peak"
    for src in ("hybrid", "sigmaclip", "auto"):
        assert seen[src][1] is True, f"{src} should recover the sparse real peak @4.3"

    # auto resolved to the reduce-side sigmaclip channel; provenance recorded.
    with h5py.File(td / "an_auto.h5", "r") as h:
        assert h["peaks"].attrs["source"] == "sigmaclip"
        assert h["peaks"].attrs["sensitivity"] == "normal"

    # Without a sigma-clip channel, Step 1 omits the residual and auto → hybrid.
    red2 = td / "reduced_nosc.h5"
    _write_reduced(red2, q, mean_s, robust_s, None)
    m1b = run_background_separation(red2, td / "an2.h5", max_half_window=40, n_passes=1)
    assert m1b["has_sigmaclip"] is False
    ana2 = Path(m1b["out_h5"])
    run_peak_fitting(ana2, None, source="auto", sensitivity="normal")
    with h5py.File(ana2, "r") as h:
        assert "sigmaclip_residual" not in h["background"]
        assert h["peaks"].attrs["source"] == "hybrid"
    assert _has_good_peak(ana2, 4.3), "hybrid (no sigmaclip) should still recover the peak"

    # The waterfall viewer can render every reconstructed source.
    for s in ("clean", "hybrid", "robust", "mean", "sigmaclip"):
        img = pattern_image(ana, source=s)
        assert img["ok"], (s, img["error"])
    # sigmaclip view errors cleanly when the channel is absent.
    assert not pattern_image(ana2, source="sigmaclip")["ok"]

    # A normal powder must NOT be flagged spotty (the diagnosis is data-driven,
    # not biased toward coarse-grained samples).
    with h5py.File(ana, "r") as h:
        assert not bool(h.attrs["spotty_sample"]), h.attrs["signal_frac_clean"]

    _test_spotty_sample_auto_source(td, q, rng)
    print("FIT SOURCE TEST OK")


def _test_spotty_sample_auto_source(td, q, rng):
    """Coarse-grained/near-single-crystal sample: ALL Bragg intensity lives in
    azimuthal spots, so the median AND sigma-clip channels hold only background.
    Step 1 must diagnose it (spotty_sample attr) and Step 2's auto source must
    fall through to 'mean' — with narrow peaks also triggering the measured
    undersampling feedback (npt recommendation)."""
    N, nb = 4, q.size
    bg = 60 + 30 * np.exp(-q / 2.0)
    sample = (pseudo_voigt(q, 2.1, 300, 0.012, 0.4)      # sharp: ~3 bins FWHM
              + pseudo_voigt(q, 2.9, 220, 0.012, 0.4)
              + pseudo_voigt(q, 4.1, 260, 0.012, 0.4))
    robust_s = np.array([bg + rng.normal(0, 1.5, nb) for _ in range(N)])
    sigma_s = np.array([bg + rng.normal(0, 1.5, nb) for _ in range(N)])
    mean_s = np.array([bg + sample + rng.normal(0, 1.5, nb) for _ in range(N)])
    red = td / "reduced_spotty.h5"
    _write_reduced(red, q, mean_s, robust_s, sigma_s)

    m1 = run_background_separation(red, td / "an_spotty.h5",
                                   max_half_window=40, n_passes=1)
    assert m1["spotty_sample"] is True, m1["signal_frac_clean"]
    ana = Path(m1["out_h5"])
    with h5py.File(ana, "r") as h:
        assert bool(h.attrs["spotty_sample"])
        assert float(h.attrs["signal_frac_clean"]) < 0.5

    m2 = run_peak_fitting(ana, None, source="auto", sensitivity="normal")
    with h5py.File(ana, "r") as h:
        assert h["peaks"].attrs["source"] == "mean", h["peaks"].attrs["source"]
    assert _has_good_peak(ana, 2.9), "auto->mean must recover the spotty sample"
    # Measured-sampling feedback: ~3-bin peaks -> concrete re-reduce advice.
    assert m2["median_fwhm_bins"] < 4.0
    assert m2["npt_recommended"] and m2["npt_recommended"] > q.size

    # An EXPLICIT source is never overridden by the diagnosis.
    a2 = td / "an_spotty_clean.h5"
    shutil.copy2(ana, a2)
    run_peak_fitting(a2, None, source="clean", sensitivity="normal")
    with h5py.File(a2, "r") as h:
        assert h["peaks"].attrs["source"] == "clean"


if __name__ == "__main__":
    main()
