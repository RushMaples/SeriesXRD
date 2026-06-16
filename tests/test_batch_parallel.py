"""Throughput + correctness: parallel==serial, excluded handling, wavelength
storage, atomic writes, and the headless batch CLI.

Steps 1-2 need numpy/h5py; Step 3a and the batch step-3 path additionally need
pymatgen (skipped when absent).
"""
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.analysis import phases as ph
from bulkxrd.analysis.background import run_background_separation
from bulkxrd.analysis.peaks import run_peak_fitting
from bulkxrd.analysis import identify as idf
from bulkxrd.analysis import batch


def _gauss(x, c, a, w):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


def _make_reduced(path, n=8, nb=1000, excluded_idx=(3,)):
    """Synthetic reduced HDF5 with strong peaks, a diamond spike, a PONI
    wavelength, and one excluded frame."""
    import h5py
    q = np.linspace(1.0, 7.0, nb)
    mean = np.zeros((n, nb), "f4")
    robust = np.zeros((n, nb), "f4")
    for i in range(n):
        shift = 0.01 * i
        bg = 50 + 20 * np.exp(-(q - 1) / 5.0)
        peaks = (_gauss(q, 2.5 - shift, 600, 0.02) + _gauss(q, 3.6 - shift, 500, 0.02)
                 + _gauss(q, 5.1 - shift, 550, 0.02))
        robust[i] = bg + peaks
        mean[i] = robust[i] + _gauss(q, 4.2, 3000, 0.02)   # diamond spike (MEAN only)
    excl = np.zeros(n, bool)
    for j in excluded_idx:
        excl[j] = True
    with h5py.File(str(path), "w") as h5:
        h5.attrs["unit"] = "q_A^-1"
        h5.attrs["poni_text"] = "Detector: Pilatus\nWavelength: 4.1300e-11\n"
        pat = h5.create_group("patterns")
        pat.create_dataset("intensity", data=mean)
        pat.create_dataset("intensity_robust", data=robust)
        pat.create_dataset("radial", data=q)
        fr = h5.create_group("frames")
        names = np.array([f"f_{i:03d}.tif" for i in range(n)], dtype=object)
        fr.create_dataset("filename", data=names, dtype=h5py.string_dtype(encoding="utf-8"))
        fr.create_dataset("excluded", data=excl)


def test_background_wavelength_excluded_and_parallel():
    import h5py
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        _make_reduced(red, n=8)
        a1 = Path(td) / "serial.h5"
        a2 = Path(td) / "par.h5"
        run_background_separation(red, a1, num_workers=1)
        run_background_separation(red, a2, num_workers=2)
        # No leftover temp files (atomic write).
        assert not a1.with_name(a1.name + ".tmp").exists()
        with h5py.File(str(a1), "r") as h, h5py.File(str(a2), "r") as g:
            # wavelength parsed from PONI (metres → Å) and stored.
            assert abs(float(h.attrs["wavelength"]) - 0.413) < 1e-3
            # excluded mask propagated.
            assert h["frames/excluded"][3] and not h["frames/excluded"][0]
            # parallel == serial, exactly.
            assert np.array_equal(h["background/clean"][:], g["background/clean"][:])
            assert np.allclose(h["frames/contamination"][:], g["frames/contamination"][:])


def _counts(path):
    import h5py
    with h5py.File(str(path), "r") as h:
        return np.asarray(h["peaks/counts"][:])


def test_peaks_excluded_atomic_and_parallel():
    import h5py
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        _make_reduced(red, n=8, excluded_idx=(3,))
        a = Path(td) / "a.h5"
        run_background_separation(red, a, num_workers=1)

        run_peak_fitting(a, None, num_workers=1)        # in place, atomic
        assert not a.with_name(a.name + ".tmp").exists()
        c_serial = _counts(a)
        # excluded frame fitted to zero peaks.
        assert c_serial[3] == 0
        # other frames found the 3 injected reflections.
        assert c_serial[0] >= 3

        # parallel run on a fresh copy → identical counts (strong peaks ⇒ seed-
        # independent, so chunk-boundary seed resets don't change the result).
        a2 = Path(td) / "a2.h5"
        run_background_separation(red, a2, num_workers=1)
        run_peak_fitting(a2, None, num_workers=2)
        assert np.array_equal(c_serial, _counts(a2))


def test_identify_excluded_and_parallel():
    if not ph.pymatgen_available():
        print("  (pymatgen not installed — skipping identify parallel/excluded)")
        return
    import h5py
    au = ph.load_bundled()["Au"]
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        _make_reduced(red, n=8, excluded_idx=(3,))
        a = Path(td) / "a.h5"
        run_background_separation(red, a, num_workers=1)
        run_peak_fitting(a, None, num_workers=1)

        idf.run_identification(a, [au], p_min=0.0, p_max=150.0, num_workers=1)
        assert not a.with_name(a.name + ".tmp").exists()
        with h5py.File(str(a), "r") as h:
            pr1 = np.asarray(h["identify/Au/pressure"][:])
        assert np.isnan(pr1[3])                  # excluded frame skipped
        assert np.isfinite(pr1[0])

        a2 = Path(td) / "a2.h5"
        run_background_separation(red, a2, num_workers=1)
        run_peak_fitting(a2, None, num_workers=1)
        idf.run_identification(a2, [au], p_min=0.0, p_max=150.0, num_workers=3)
        with h5py.File(str(a2), "r") as h:
            pr2 = np.asarray(h["identify/Au/pressure"][:])
        # parallel == serial on the non-excluded frames.
        ok = np.isfinite(pr1) & np.isfinite(pr2)
        assert ok.sum() >= 6 and np.allclose(pr1[ok], pr2[ok])


def test_batch_cli():
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        _make_reduced(red, n=6)
        out = Path(td) / "out.h5"
        steps = "123" if ph.pymatgen_available() else "12"
        argv = [str(red), "-o", str(out), "--steps", steps, "--workers", "2"]
        if ph.pymatgen_available():
            argv += ["--phases", "Au", "--workspace", td]
        rc = batch.main(argv)
        assert rc == 0 and out.is_file()
        import h5py
        with h5py.File(str(out), "r") as h:
            assert "background/clean" in h and "peaks" in h
            if ph.pymatgen_available():
                assert "identify" in h


def main() -> None:
    test_background_wavelength_excluded_and_parallel()
    test_peaks_excluded_atomic_and_parallel()
    test_identify_excluded_and_parallel()
    test_batch_cli()
    print("BATCH/PARALLEL TEST OK")


if __name__ == "__main__":
    main()
