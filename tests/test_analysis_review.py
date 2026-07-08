"""Analysis-stage review/worker integration (numpy + h5py; no pyFAI/Tk needed).

Builds a synthetic reduced HDF5, drives the analysis worker (Step 1 background +
Step 2 peaks) the same way the GUI's subprocess would, then exercises the
read-only review helpers the GUI plots from.
"""
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _gauss(x, c, a, w):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


def _make_reduced(path: Path, n_frames: int = 6, n_bins: int = 1200) -> None:
    import h5py
    q = np.linspace(1.0, 8.0, n_bins)
    mean = np.zeros((n_frames, n_bins), dtype="f4")
    robust = np.zeros((n_frames, n_bins), dtype="f4")
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        shift = 0.02 * i  # peaks drift as the lattice compresses
        bg = 60 + 30 * np.exp(-(q - 1) / 6.0)
        peaks = (_gauss(q, 2.5 - shift, 400, 0.012) + _gauss(q, 3.6 - shift, 260, 0.013)
                 + _gauss(q, 5.1 - shift, 520, 0.011))
        rob = bg + peaks + rng.normal(0, 2.0, n_bins)
        robust[i] = rob
        mean[i] = rob + _gauss(q, 4.0, 2500, 0.012)  # diamond spike (MEAN only)
    with h5py.File(str(path), "w") as h5:
        h5.attrs["unit"] = "q_A^-1"
        pat = h5.create_group("patterns")
        pat.create_dataset("intensity", data=mean)
        pat.create_dataset("intensity_robust", data=robust)
        pat.create_dataset("radial", data=q)
        fr = h5.create_group("frames")
        names = np.array([f"frame_{i:03d}.tif" for i in range(n_frames)], dtype=object)
        fr.create_dataset("filename", data=names, dtype=h5py.string_dtype(encoding="utf-8"))


def main() -> None:
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("ANALYSIS REVIEW TEST SKIPPED (h5py not installed)")
        return

    from bulkxrd.analysis.worker import run_analysis
    from bulkxrd.analysis.review import (
        inspect_analysis, frame_data, peak_map, structure_report)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        reduced = td / "reduced_test.h5"
        _make_reduced(reduced, n_frames=6, n_bins=1200)

        cfg = {
            "reduced_h5_file": str(reduced),
            "analysis_h5_file": "",
            "run_step1": True, "run_step2": True,
            "max_half_window": "50", "n_passes": "1", "use_lls": True,
            "contamination_threshold": "",
            "min_snr": "5.0", "window_factor": "3.0", "max_chi2": "25.0",
            "propagate_seeds": True,
        }
        manifest = run_analysis(cfg)
        assert manifest["steps"] == ["background", "peaks"], manifest["steps"]
        out = Path(manifest["analysis_h5_file"])
        assert out.is_file(), out

        # inspect_analysis: sees both steps, correct frame count, no fatal anomaly.
        rev = inspect_analysis(out)
        assert rev["ok_to_read"] and rev["has_background"] and rev["has_peaks"]
        assert rev["n_frames"] == 6 and rev["n_bins"] == 1200
        assert rev["n_good"] > 0, "no good peaks fitted"
        assert "q_A^-1" in rev["unit"]
        report = structure_report(rev)
        assert "Step 2 peaks: yes" in report and str(out) in report

        # frame_data: robust/mean reconstruction is lossless; peaks recovered.
        fd = frame_data(out, 0)
        assert fd["ok"] and fd["radial"] is not None
        recon = fd["clean"] + fd["baseline"]
        assert np.allclose(recon, fd["robust"], equal_nan=True)
        assert np.allclose(fd["robust"] + fd["spot_residual"], fd["mean"], equal_nan=True)
        assert fd["filename"] == "frame_000.tif"
        # the three injected reflections should each show up as a fitted peak
        centers = [p["center"] for p in fd["peaks"]]
        for c in (2.5, 3.6, 5.1):
            assert any(abs(pc - c) < 0.1 for pc in centers), f"missing peak near {c}: {centers}"

        # out-of-range frame is handled, not raised.
        assert not frame_data(out, 999)["ok"]

        # peak_map: good_only filters flagged peaks; arrays align.
        pm = peak_map(out, good_only=False)
        assert pm["ok"] and pm["n_frames"] == 6
        assert pm["frame"].shape == pm["center"].shape == pm["area"].shape
        pm_good = peak_map(out, good_only=True)
        assert pm_good["center"].size <= pm["center"].size
        assert np.all(pm_good["flag"] == 0)

    _test_cake_for_frame()
    _test_frame_data_residual_unknowns()
    print("ANALYSIS REVIEW TEST OK")


def _test_cake_for_frame():
    """cake_for_frame pulls the right cake by frame index from a reduced file."""
    import h5py
    from bulkxrd.analysis.review import cake_for_frame
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        n_rad, n_az = 50, 36
        cake0 = np.random.default_rng(1).random((n_az, n_rad)).astype("f4")
        cake5 = np.random.default_rng(2).random((n_az, n_rad)).astype("f4")
        with h5py.File(str(red), "w") as h5:
            h5.attrs["unit"] = "2th_deg"
            cg = h5.create_group("cakes")
            cg.create_dataset("intensity", data=np.stack([cake0, cake5]))
            cg.create_dataset("radial", data=np.linspace(2, 20, n_rad))
            cg.create_dataset("azimuthal", data=np.linspace(-180, 180, n_az))
            cg.create_dataset("frame_index", data=np.array([0, 5], dtype="i4"))
        ok = cake_for_frame(red, 5)
        assert ok["ok"] and ok["cake"].shape == (n_az, n_rad)
        assert np.allclose(ok["cake"], cake5)
        miss = cake_for_frame(red, 3)          # no cake stored for frame 3
        assert not miss["ok"] and "frame 3" in miss["error"]
        assert not cake_for_frame(Path(td) / "nope.h5", 0)["ok"]


def _test_frame_data_residual_unknowns():
    """frame_data exposes /residual/clean, /residual/peaks, and /unknowns/obs."""
    import h5py
    from bulkxrd.analysis.review import frame_data, inspect_analysis, structure_report
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "analysis.h5"
        radial = np.linspace(1.0, 5.0, 20)
        clean = np.vstack([radial, radial + 10]).astype("f4")
        with h5py.File(str(p), "w") as h:
            h.attrs["unit"] = "q_A^-1"
            h.create_dataset("radial", data=radial)
            bg = h.create_group("background")
            bg.create_dataset("clean", data=clean)
            bg.create_dataset("baseline", data=np.zeros_like(clean))
            bg.create_dataset("spot_residual", data=np.zeros_like(clean))
            rg = h.create_group("residual")
            rg.create_dataset("clean", data=clean * 0.5)
            rpk = rg.create_group("peaks")
            rpk.create_dataset("counts", data=np.array([1, 1], "i4"))
            rpk.create_dataset("frame", data=np.array([0, 1], "i4"))
            rpk.create_dataset("center", data=np.array([2.0, 3.0]))
            rpk.create_dataset("amplitude", data=np.array([12.0, 14.0]))
            rpk.create_dataset("fwhm", data=np.array([0.05, 0.06]))
            un = h.create_group("unknowns")
            obs = un.create_group("obs")
            obs.create_dataset("track", data=np.array([7], "i4"))
            obs.create_dataset("frame", data=np.array([1], "i4"))
            obs.create_dataset("center", data=np.array([3.0]))
            obs.create_dataset("amplitude", data=np.array([14.0]))
            obs.create_dataset("fwhm", data=np.array([0.06]))
            tr = un.create_group("tracks")
            tr.create_dataset("id", data=np.array([7], "i4"))
            tr.create_dataset("cluster", data=np.array([2], "i4"))
            cl = un.create_group("clusters")
            cl.create_dataset("id", data=np.array([2], "i4"))

        info = inspect_analysis(p)
        assert info["has_residual"] and info["has_unknowns"]
        assert info["n_residual_peaks"] == 2
        assert info["n_unknown_obs"] == 1
        assert "Residual: yes" in structure_report(info)

        fd = frame_data(p, 1)
        assert fd["ok"]
        assert np.allclose(fd["residual"], clean[1] * 0.5)
        assert fd["residual_peaks"][0]["center"] == 3.0
        assert fd["unknown_obs"][0]["track"] == 7
        assert fd["unknown_obs"][0]["cluster"] == 2


if __name__ == "__main__":
    main()
