"""Step 3 semi-quantitative phase fractions: intensity-share / RIR weighting
from the Step-3a-removal peak attribution (/peaks/phase, /peaks/area)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py

from seriesxrd.analysis.fractions import phase_fractions, run_fractions


def _fractions_file(path):
    """Analysis HDF5 with a /peaks list covering:
      frame 0: phase A area=30, phase B area=10 (good, attributed) plus a
               FLAGGED (bad) A peak with a huge area that must be excluded,
               plus an unattributed ("") peak excluded from the denominator
               -> fractions 0.75 / 0.25.
      frame 1: only unattributed ("") peaks -> an all-NaN fraction row.
      frame 2: phase A area=8, phase B area=8 -> an even 0.5 / 0.5 split.
    """
    frame = np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype="i4")
    center = np.array([1.0, 1.5, 2.0, 2.5, 1.0, 1.2, 1.0, 1.5], dtype="f8")
    area = np.array([30.0, 10.0, 1000.0, 5.0, 20.0, 15.0, 8.0, 8.0], dtype="f8")
    flag = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype="i4")
    phase = np.array(["A", "B", "A", "", "", "", "A", "B"], dtype=object)
    counts = np.bincount(frame, minlength=3).astype("i4")
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(str(path), "w") as h:
        h.attrs["unit"] = "q_A^-1"
        fr = h.create_group("frames")
        fr.create_dataset("filename",
                          data=np.array([f"f{i}.tif" for i in range(3)], dtype=object),
                          dtype=str_dtype)
        pk = h.create_group("peaks")
        pk.create_dataset("counts", data=counts)
        pk.create_dataset("frame", data=frame)
        pk.create_dataset("center", data=center)
        pk.create_dataset("area", data=area)
        pk.create_dataset("flag", data=flag)
        pk.create_dataset("phase", data=phase, dtype=str_dtype)


def test_intensity_share():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _fractions_file(p)
        res = phase_fractions(p)
        assert res["ok"], res["error"]
        assert res["n_frames"] == 3
        assert res["phases"] == ["A", "B"]
        assert res["method"] == "intensity_share"
        assert res["rir_used"] == {"A": False, "B": False}
        frac = res["fractions"]
        assert frac.shape == (3, 2)
        # frame 0: 30/(30+10)=0.75, 10/40=0.25 -- flagged + unattributed peaks excluded
        assert abs(frac[0, 0] - 0.75) < 1e-9, frac[0]
        assert abs(frac[0, 1] - 0.25) < 1e-9, frac[0]
        # frame 1: only unattributed peaks -> all-NaN row
        assert np.isnan(frac[1, 0]) and np.isnan(frac[1, 1]), frac[1]
        # frame 2: 8/16 each
        assert abs(frac[2, 0] - 0.5) < 1e-9, frac[2]
        assert abs(frac[2, 1] - 0.5) < 1e-9, frac[2]


def test_rir_changes_ratio():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _fractions_file(p)
        # B is missing from use_rir -> falls back to RIR=1.
        res = phase_fractions(p, use_rir={"A": 2.0})
        assert res["ok"], res["error"]
        assert res["method"] == "rir"
        assert res["rir_used"] == {"A": True, "B": False}
        frac = res["fractions"]
        # frame 0: weight_A=30/2=15, weight_B=10/1=10, denom=25 -> 0.6 / 0.4
        assert abs(frac[0, 0] - 0.6) < 1e-9, frac[0]
        assert abs(frac[0, 1] - 0.4) < 1e-9, frac[0]
        # frame 2: weight_A=8/2=4, weight_B=8/1=8, denom=12 -> 1/3, 2/3
        assert abs(frac[2, 0] - 1.0 / 3.0) < 1e-9, frac[2]
        assert abs(frac[2, 1] - 2.0 / 3.0) < 1e-9, frac[2]


def test_min_conf_gate():
    """A phase whose Step-3a confidence is below min_conf in a frame is
    dropped from that frame's numerator AND denominator -- like an
    unattributed peak -- instead of silently diluting the surviving phase."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(str(p), "w") as h:
            h.attrs["unit"] = "q_A^-1"
            pk = h.create_group("peaks")
            pk.create_dataset("counts", data=np.array([2], dtype="i4"))
            pk.create_dataset("frame", data=np.array([0, 0], dtype="i4"))
            pk.create_dataset("area", data=np.array([5.0, 5.0], dtype="f8"))
            pk.create_dataset("flag", data=np.array([0, 0], dtype="i4"))
            pk.create_dataset("phase", data=np.array(["A", "B"], dtype=object),
                              dtype=str_dtype)
            idg = h.create_group("identify")
            idg.create_group("A").create_dataset("confidence", data=np.array([0.9]))
            idg.create_group("B").create_dataset("confidence", data=np.array([0.3]))
        res = phase_fractions(p, min_conf=0.5)
        assert res["ok"], res["error"]
        frac = res["fractions"]
        ia, ib = res["phases"].index("A"), res["phases"].index("B")
        assert abs(frac[0, ia] - 1.0) < 1e-9, frac[0]
        assert abs(frac[0, ib] - 0.0) < 1e-9, frac[0]


def test_run_fractions_write_idempotent_atomic():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _fractions_file(p)
        man = run_fractions(p)
        assert man["ok"] and man["written"]
        assert man["phases"] == ["A", "B"]
        assert abs(man["per_phase"]["A"]["mean_fraction"] - 0.625) < 1e-9  # (0.75+0.5)/2
        assert man["per_phase"]["A"]["n_frames_present"] == 2   # frames 0, 2 (>0.05)
        assert man["rir_missing"] == []                          # not RIR mode
        with h5py.File(str(p), "r") as h:
            assert "fractions" in h
            g = h["fractions"]
            names = [s.decode() if isinstance(s, bytes) else s for s in g["names"][:]]
            assert names == ["A", "B"]
            assert g["fractions"].shape == (3, 2)
            assert str(g.attrs["method"]) == "intensity_share"
        assert not p.with_name(p.name + ".tmp").exists()

        # Re-run with RIR: replaces the group (idempotent), not additive.
        man2 = run_fractions(p, use_rir={"A": 2.0})
        assert man2["ok"] and man2["method"] == "rir"
        assert man2["rir_missing"] == ["B"]
        with h5py.File(str(p), "r") as h:
            g = h["fractions"]
            assert str(g.attrs["method"]) == "rir"
            frac2 = g["fractions"][:]
            assert frac2.shape == (3, 2)          # replaced, not duplicated
            assert abs(frac2[0, 0] - 0.6) < 1e-9
        assert not p.with_name(p.name + ".tmp").exists()


def main() -> None:
    test_intensity_share()
    test_rir_changes_ratio()
    test_min_conf_gate()
    test_run_fractions_write_idempotent_atomic()
    print("FRACTIONS TEST OK")


if __name__ == "__main__":
    main()
