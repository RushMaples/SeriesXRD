"""Step 3c unknown clustering + Williamson–Hall microstructure analysis."""
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
from bulkxrd.analysis.unknowns import link_tracks, cluster_tracks, run_unknowns
from bulkxrd.analysis.microstructure import williamson_hall

TWO_PI = 2 * np.pi


def _residual_file(path, n_frames=15):
    """Analysis HDF5 whose /residual/peaks holds two coherent 'phases':
    cluster A (2 tracks, frames 0-9, drifting up in q) and cluster B
    (2 tracks, frames 5-14), plus one-off noise peaks."""
    rng = np.random.default_rng(0)
    frames, centers, amps, fwhms = [], [], [], []

    def track(f0, f1, q0, drift):
        for f in range(f0, f1 + 1):
            frames.append(f)
            centers.append(q0 + drift * (f - f0) + rng.normal(0, 5e-4))
            amps.append(50.0)
            fwhms.append(0.02)

    track(0, 9, 2.00, +0.004)      # cluster A
    track(0, 9, 3.10, +0.006)
    track(5, 14, 2.60, +0.005)     # cluster B
    track(5, 14, 4.05, +0.008)
    for f in (2, 7, 11):           # isolated noise blips
        frames.append(f); centers.append(5.0 + f * 0.1)
        amps.append(8.0); fwhms.append(0.02)

    counts = np.bincount(np.asarray(frames), minlength=n_frames)
    with h5py.File(str(path), "w") as h:
        h.attrs["unit"] = "q_A^-1"
        rg = h.create_group("residual")
        gp = rg.create_group("peaks")
        gp.create_dataset("counts", data=counts.astype("i4"))
        gp.create_dataset("frame", data=np.asarray(frames, "i4"))
        gp.create_dataset("center", data=np.asarray(centers, "f8"))
        gp.create_dataset("amplitude", data=np.asarray(amps, "f8"))
        gp.create_dataset("fwhm", data=np.asarray(fwhms, "f8"))


def test_unknown_clustering():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _residual_file(p)
        man = run_unknowns(p, min_track_frames=3, jaccard_threshold=0.6)
        assert man["n_tracks"] == 4, man["n_tracks"]      # noise blips dropped
        assert man["n_clusters"] == 2, man["n_clusters"]
        for s in man["clusters"]:
            assert s["n_tracks"] == 2
            assert len(s["d_fingerprint"]) == 2           # both tracks at ref frame
        spans = sorted((s["first_frame"], s["last_frame"]) for s in man["clusters"])
        assert spans == [(0, 9), (5, 14)]                 # transition candidates
        with h5py.File(str(p), "r") as h:
            g = h["unknowns"]
            assert int(g.attrs["n_clusters"]) == 2
            assert g["obs/track"].shape[0] == 40          # 4 tracks x 10 frames
            cl = g["tracks/cluster"][:]
            assert len(set(cl.tolist())) == 2
            # fingerprint d = 2*pi/q of member centers at the reference frame
            fp_d = g["fingerprint/d"][:]
            assert np.all((fp_d > 1.0) & (fp_d < 4.0))
        # re-run replaces the group (idempotent)
        man2 = run_unknowns(p)
        assert man2["n_clusters"] == 2


def test_track_linking_gap_tolerance():
    """A track may skip up to max_gap frames (weak peak below SNR) without
    being split in two."""
    frames = np.array([0, 1, 4, 5])         # gap of 2 frames (2, 3 missing)
    centers = np.array([2.0, 2.004, 2.016, 2.02])
    amps = np.ones(4) * 10
    fwhms = np.ones(4) * 0.02
    tr = link_tracks(frames, centers, amps, fwhms, n_frames=6,
                     max_gap=2, min_track_frames=3)
    assert len(tr) == 1 and tr[0]["frames"].size == 4
    tr2 = link_tracks(frames, centers, amps, fwhms, n_frames=6,
                      max_gap=1, min_track_frames=2)
    assert len(tr2) == 2                    # gap too wide -> split


def test_williamson_hall():
    """Recover known size/strain: dq = 2*pi*K/D + 2*eps*q."""
    K, D, eps = 0.9, 400.0, 0.002
    n_frames, qs = 3, np.linspace(1.0, 5.5, 10)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        with h5py.File(str(p), "w") as h:
            h.attrs["unit"] = "q_A^-1"
            gp = h.create_group("peaks")
            frame = np.repeat(np.arange(n_frames), qs.size)
            q_all = np.tile(qs, n_frames)
            dq = TWO_PI * K / D + 2 * eps * q_all
            gp.create_dataset("counts", data=np.full(n_frames, qs.size, "i4"))
            gp.create_dataset("frame", data=frame.astype("i4"))
            gp.create_dataset("center", data=q_all)
            gp.create_dataset("fwhm", data=dq)
            gp.create_dataset("fwhm_err", data=np.full(q_all.size, 1e-4))
            gp.create_dataset("flag", data=np.zeros(q_all.size, "i4"))
        man = williamson_hall(p, k_shape=K, min_peaks=5)
        assert man["instrument_corrected"] is False and "warning" in man
        for i in range(n_frames):
            assert abs(man["size_A"][i] - D) < 0.05 * D, man["size_A"][i]
            assert abs(man["strain"][i] - eps) < 0.05 * eps, man["strain"][i]
            assert man["r2"][i] > 0.999
        with h5py.File(str(p), "r") as h:
            assert "microstructure" in h
            assert np.isfinite(h["microstructure/size_A"][:]).all()

        # instrument correction removes a constant-q broadening in quadrature
        inst = 0.008
        with h5py.File(str(p), "r+") as h:
            raw = h["peaks/fwhm"][:]
            h["peaks/fwhm"][...] = np.sqrt(raw**2 + inst**2)
        man2 = williamson_hall(p, k_shape=K, min_peaks=5, instrument_fwhm_q=inst,
                               write=False)
        assert man2["instrument_corrected"] is True
        assert abs(man2["size_A"][0] - D) < 0.05 * D
        assert abs(man2["strain"][0] - eps) < 0.06 * eps


def main() -> None:
    test_track_linking_gap_tolerance()
    test_unknown_clustering()
    test_williamson_hall()
    print("UNKNOWNS/WH TEST OK")


if __name__ == "__main__":
    main()
