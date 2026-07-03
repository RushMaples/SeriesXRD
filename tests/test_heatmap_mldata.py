"""Heatmap data layer + ML-dataset export.

Builds a small analysis HDF5 with background + peaks + a Step-3a /identify group
(Au at a known pressure), then exercises the waterfall image, reflection tracks,
per-phase layers, the experimental ML export, and the simulated training set.
The track/layer/export-label and simulation paths need pymatgen and are skipped
when it is absent (the core resampling/image paths are always tested).
"""
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.analysis import phases as ph
from bulkxrd.analysis import identify as idf
from bulkxrd.analysis import heatmap as hm
from bulkxrd.analysis import mldata as ml


def _build_analysis(path, au, p_true=60.0, n=4, nb=800):
    """Minimal analysis HDF5: radial + background/clean + peaks + identify(Au)."""
    import h5py
    q = np.linspace(1.0, 7.0, nb)
    refl = idf.phase_reflections(au) if ph.pymatgen_available() else None
    clean = np.zeros((n, nb), "f4")
    if refl is not None:
        d0, w, _ = refl
        s = idf.scale_at_pressure(au, p_true)
        centers_q = 2 * np.pi / (d0[:6] * s)
        from bulkxrd.analysis.peaks import pseudo_voigt
        for i in range(n):
            row = np.zeros(nb)
            for c, a in zip(centers_q, w[:6]):
                if q[0] <= c <= q[-1]:
                    row += pseudo_voigt(q, c, 100 * a, 0.03, 0.5)
            clean[i] = row
        centers = centers_q
    else:
        centers = np.array([2.0, 3.0, 4.0])
        for i in range(n):
            clean[i] = np.exp(-0.5 * ((q - 3.0) / 0.05) ** 2) * 100
    with h5py.File(str(path), "w") as h5:
        h5.attrs["unit"] = "q_A^-1"
        h5.attrs["source_reduced"] = "synthetic"
        h5.create_dataset("radial", data=q)
        gb = h5.create_group("background")
        gb.create_dataset("clean", data=clean)
        gb.create_dataset("baseline", data=np.zeros((n, nb), "f4"))
        gb.create_dataset("spot_residual", data=np.zeros((n, nb), "f4"))
        k = centers.size
        gp = h5.create_group("peaks")
        gp.create_dataset("counts", data=np.full(n, k, "i4"))
        gp.create_dataset("frame", data=np.repeat(np.arange(n), k).astype("i4"))
        gp.create_dataset("center", data=np.tile(centers, n).astype("f8"))
        gp.create_dataset("flag", data=np.zeros(n * k, "i4"))


def test_pattern_image():
    au = ph.load_bundled()["Au"]
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "a.h5"
        _build_analysis(h5, au)
        img = hm.pattern_image(h5, source="clean")
        assert img["ok"], img["error"]
        assert img["Z"].shape == (800, 4)          # (n_bins, n_frames)
        assert img["radial"].size == 800 and img["x"].size == 4
        # mean/robust reconstruct without error
        assert hm.pattern_image(h5, source="robust")["ok"]
        # unknown source rejected
        assert not hm.pattern_image(h5, source="nope")["ok"]
        # pressure axis needs Step-3a; absent here -> graceful error
        bad = hm.pattern_image(h5, x_axis="pressure", pressure_phase="Au")
        assert not bad["ok"]


def test_metadata_pressure_axis_and_anisotropic_tracks():
    """x_axis='pressure' from /frames/pressure (no phase needed), and reflection
    tracks driven by the anisotropic predicted_d (soft axis compresses more).
    Both run without pymatgen — tracks read the Step-3a reflection cache."""
    import h5py
    nb, n = 40, 5
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "a.h5"
        with h5py.File(str(h5), "w") as f:
            f.attrs["unit"] = "q_A^-1"
            f.create_dataset("radial", data=np.linspace(1.0, 8.0, nb))
            f.create_group("background").create_dataset(
                "clean", data=np.random.rand(n, nb).astype("f4"))
            gf = f.create_group("frames")
            gf.create_dataset("pressure", data=np.array([0.0, 5.0, 10.0, 15.0, 20.0]))
            # Cache reflections for a tetragonal phase with a soft c-axis.
            g = f.require_group("identify").require_group("tet")
            g.attrs["name"] = "tet"
            g.create_dataset("refl_d", data=np.array([4.0, 6.0]))     # (100), (001)
            g.create_dataset("refl_w", data=np.array([1.0, 1.0]))
            g.create_dataset("refl_hkl",
                             data=np.array(["(1, 0, 0)", "(0, 0, 1)"], dtype=object),
                             dtype=h5py.string_dtype(encoding="utf-8"))

        # Metadata pressure axis: no pressure_phase -> uses /frames/pressure.
        img = hm.pattern_image(h5, source="clean", x_axis="pressure")
        assert img["ok"], img["error"]
        assert np.allclose(img["x"], [0, 5, 10, 15, 20])
        assert "metadata" in img["x_label"]

        tet = ph.Phase(name="tet",
                       lattice={"a": 4.0, "b": 4.0, "c": 6.0,
                                "alpha": 90, "beta": 90, "gamma": 90},
                       axial_eos={"a": {"type": "BM3", "K0": 300, "K0p": 4},
                                  "c": {"type": "BM3", "K0": 100, "K0p": 4}})
        tr = hm.reflection_tracks(h5, tet)
        assert tr["ok"] and len(tr["tracks"]) == 2
        c100 = tr["tracks"][0]["centers"]      # (100) on the stiff a-axis
        c001 = tr["tracks"][1]["centers"]      # (001) on the soft c-axis
        # q rises with pressure for both; the soft c-axis reflection rises more.
        assert c100[-1] > c100[0] and c001[-1] > c001[0]
        assert (c001[-1] / c001[0]) > (c100[-1] / c100[0])


def test_resample():
    grid = ml.make_d_grid()
    assert grid.size == 3501 and abs(grid[0] - 1.199) < 1e-6 and abs(grid[-1] - 8.853) < 1e-6
    q = np.linspace(1.0, 7.0, 500)
    # a single q-peak at q0 → a peak near d0 = 2π/q0 on the grid
    q0 = 2.0
    y = np.exp(-0.5 * ((q - q0) / 0.02) ** 2)
    r = ml.resample_to_d(q, y, "q_A^-1", None, grid)
    d_peak = grid[int(np.argmax(r))]
    assert abs(d_peak - 2 * np.pi / q0) < 0.05


def test_tracks_layers_and_export():
    if not ph.pymatgen_available():
        print("  (pymatgen not installed — skipping tracks/layers/labelled export)")
        return
    au = ph.load_bundled()["Au"]
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "a.h5"
        _build_analysis(h5, au, p_true=60.0)
        idf.run_identification(h5, [au], p_min=0.0, p_max=200.0)

        tr = hm.reflection_tracks(h5, au)
        assert tr["ok"] and tr["tracks"]
        c0 = tr["tracks"][0]["centers"]
        assert np.isfinite(c0).all() and np.all(c0 > 0)

        layers = hm.phase_layers(h5, [au])
        assert layers["ok"] and layers["layers"]
        lay = layers["layers"][0]
        assert lay["name"] == "Au" and lay["intensity"].size == 4
        assert lay["intensity"].max() > 0  # the phase's windows capture signal

        out = Path(td) / "ml.npz"
        man = ml.export_ml_dataset(h5, out, channels=("clean", "spot_residual"))
        assert out.is_file() and man["n_channels"] == 2 and man["has_labels"]
        # Close the npz before the TemporaryDirectory is removed (Windows holds a
        # file handle open until the NpzFile is closed).
        with np.load(out, allow_pickle=True) as z:
            assert z["X"].shape == (4, 2, 3501)
            assert "y" in z and z["y"].shape[1] == 1 and z["y"].sum() > 0
            assert list(z["phase_names"]) == ["Au"]


def test_export_fit_and_residual_channels():
    """export_ml_dataset accepts every ml_features source — notably 'fit' (the
    channel Step 2 actually fit, resolved via /peaks.attrs) and 'residual' —
    and records the resolved channel names. Runs with or without pymatgen
    (the fixture simulates reflections only when pymatgen is present)."""
    import h5py
    au = ph.load_bundled()["Au"]
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "a.h5"
        _build_analysis(h5, au)
        with h5py.File(str(h5), "r+") as f:
            f["peaks"].attrs["source"] = "clean"
            clean = np.asarray(f["background/clean"][:])
            f.create_group("residual").create_dataset("clean", data=clean * 0.5)
        out = Path(td) / "ml.npz"
        man = ml.export_ml_dataset(h5, out, channels=("fit", "residual"))
        assert man["channels"] == ["fit", "residual"]
        assert man["resolved_channels"] == ["clean", "residual"]
        with np.load(out, allow_pickle=True) as z:
            assert z["X"].shape == (4, 2, 3501)
            assert list(z["resolved_channels"]) == ["clean", "residual"]
        # Unknown channels are rejected with the full menu in the error.
        try:
            ml.export_ml_dataset(h5, out, channels=("nope",))
            assert False, "expected ValueError"
        except ValueError as e:
            assert "fit" in str(e)


def test_simulated_dataset():
    if not ph.pymatgen_available():
        print("  (pymatgen not installed — skipping simulated dataset)")
        return
    au = ph.load_bundled()["Au"]
    X, y, names = ml.build_simulated_dataset([au], pressures=[0.0, 50.0, 100.0])
    assert X.shape == (3, 3501) and list(y) == [0, 0, 0] and names == ["Au"]
    # different pressures shift peaks → distinct rows
    assert not np.allclose(X[0], X[2])
    assert X.max() <= 1.0 + 1e-6 and X.min() >= 0.0


def test_simulated_dataset_axial_only_scans_pressure():
    """Regression: an AXIAL-only phase (per-axis EOS, no volume EOS) has a
    pressure degree of freedom, so the simulated training set must emit one row
    per pressure — checking only has_eos() pinned such phases at ambient. Runs
    without pymatgen by patching the reflection source."""
    assert not ph.has_pressure_dof(ph.Phase(name="none"))
    tet = ph.Phase(name="Tet",
                   lattice={"a": 4.0, "b": 4.0, "c": 6.0,
                            "alpha": 90, "beta": 90, "gamma": 90},
                   axial_eos={"a": {"type": "BM3", "K0": 300, "K0p": 4},
                              "c": {"type": "BM3", "K0": 100, "K0p": 4}})
    assert not tet.has_eos() and ph.has_pressure_dof(tet)
    refl = (np.array([4.0, 6.0]), np.array([1.0, 1.0]), ["(1, 0, 0)", "(0, 0, 1)"])
    noeos = ph.Phase(name="Ref")
    saved = ml.phase_reflections
    ml.phase_reflections = lambda p, **k: refl
    try:
        X, y, names = ml.build_simulated_dataset([tet, noeos],
                                                 pressures=[0.0, 10.0, 20.0])
    finally:
        ml.phase_reflections = saved
    # 3 pressure rows for the axial phase + 1 ambient row for the no-EOS one.
    assert list(y) == [0, 0, 0, 1], list(y)
    assert not np.allclose(X[0], X[2]), "axial-only phase did not move with pressure"


def main() -> None:
    test_pattern_image()
    test_metadata_pressure_axis_and_anisotropic_tracks()
    test_resample()
    test_tracks_layers_and_export()
    test_export_fit_and_residual_channels()
    test_simulated_dataset()
    test_simulated_dataset_axial_only_scans_pressure()
    print("HEATMAP/MLDATA TEST OK")


if __name__ == "__main__":
    main()
