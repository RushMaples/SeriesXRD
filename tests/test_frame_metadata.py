"""Frame-metadata seam: filename pressure parsing, CSV import, HDF5 apply.

Pure numpy + h5py (no pymatgen). Covers the pressure prior source that Step 3
consumes: parse from filenames/folders, import/override from CSV (keyed by frame
index or filename), and the atomic write into an analysis HDF5's /frames group.
"""
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.analysis import frame_metadata as fm


def test_parse_pressure_units():
    # GPa, decimal 'p', folder, kbar, Mbar, MPa, and non-matches.
    cases = {
        "UOTe-1GPa-001.tif": 1.0,
        "UOTe-1p5GPa_002.tif": 1.5,
        "UOTe-3p9GPa.tif": 3.9,
        "1 GPa/UOTe-005.tif": 1.0,
        "1 GPa\\UOTe-006.tif": 1.0,
        "scan_500MPa.tif": 0.5,
        "run-10kbar.cbf": 1.0,         # 10 kbar = 1.0 GPa
        "x-2Mbar.tif": 200.0,          # 2 Mbar = 200 GPa
        "sample_12.5GPa_cold.tif": 12.5,
        "UOTe-noP-001.tif": None,
        "20240101_run001.tif": None,   # date digits, no unit -> no match
    }
    for name, exp in cases.items():
        got = fm.parse_pressure_from_path(name)
        if exp is None:
            assert got is None, f"{name!r} -> {got}, expected None"
        else:
            assert got is not None and abs(got - exp) < 1e-9, f"{name!r} -> {got}, expected {exp}"


def test_extract_and_summary():
    names = ["a-1GPa.tif", "a-2GPa.tif", "a-noP.tif", "a-3GPa.tif"]
    pr = fm.extract_pressures(names)
    assert np.allclose(pr[[0, 1, 3]], [1.0, 2.0, 3.0]) and np.isnan(pr[2])
    s = fm.summarize_pressures(pr)
    assert s["n_frames"] == 4 and s["n_parsed"] == 3
    assert s["p_min"] == 1.0 and s["p_max"] == 3.0


def test_csv_by_frame_and_filename():
    names = ["d/UOTe-1GPa-001.tif", "UOTe-2p5GPa-002.tif", "UOTe-x-003.tif"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # keyed by frame, with sigma + temperature
        c1 = td / "byframe.csv"
        c1.write_text("frame,pressure_gpa,pressure_sigma_gpa,temperature_K\n"
                      "0,1.0,0.1,300\n1,2.5,0.2,305\n2,3.3,0.15,310\n")
        r = fm.read_pressure_csv(c1)
        assert r["ok"] and set(r["columns"]) >= {"frame", "pressure", "pressure_sigma", "temperature"}
        m = fm.map_csv_to_frames(r["rows"], names, len(names))
        assert np.allclose(m["pressure"], [1.0, 2.5, 3.3])
        assert np.allclose(m["pressure_sigma"], [0.1, 0.2, 0.15])
        assert np.allclose(m["temperature"], [300, 305, 310])

        # keyed by filename (basename match), pressure only
        c2 = td / "byname.csv"
        c2.write_text("filename,pressure\nUOTe-2p5GPa-002.tif,9.9\nUOTe-x-003.tif,8.8\n")
        r2 = fm.read_pressure_csv(c2)
        m2 = fm.map_csv_to_frames(r2["rows"], names, len(names))
        assert np.isnan(m2["pressure"][0]) and m2["pressure"][1] == 9.9 and m2["pressure"][2] == 8.8

        # a CSV without a pressure column is rejected
        c3 = td / "bad.csv"
        c3.write_text("frame,temperature_K\n0,300\n")
        assert not fm.read_pressure_csv(c3)["ok"]


def _make_analysis(path, names):
    import h5py
    n = len(names)
    with h5py.File(str(path), "w") as h:
        h.attrs["unit"] = "q_A^-1"
        h.create_group("background").create_dataset("clean", data=np.zeros((n, 8), "f4"))
        gf = h.create_group("frames")
        gf.create_dataset("filename", data=np.array(names, dtype=object),
                          dtype=h5py.string_dtype(encoding="utf-8"))
        gf.create_dataset("pressure", data=np.full(n, np.nan))   # placeholder, like reduce


def test_apply_roundtrip_and_partial_overwrite():
    import h5py
    names = ["1 GPa/UOTe-1GPa-001.tif", "UOTe-2p5GPa-002.tif", "UOTe-x-003.tif", "UOTe-4GPa-004.tif"]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _make_analysis(p, names)

        # extract from filenames
        man = fm.extract_to_analysis(p)
        assert man["n_pressure"] == 3
        md = fm.read_frame_metadata(p)
        assert np.allclose(md["pressure"][[0, 1, 3]], [1.0, 2.5, 4.0]) and np.isnan(md["pressure"][2])

        # CSV with sigma+temp over all frames
        csv = Path(td) / "c.csv"
        csv.write_text("frame,pressure_gpa,pressure_sigma_gpa,temperature_K\n"
                       "0,1.1,0.1,300\n1,2.6,0.2,305\n2,3.4,0.15,310\n3,4.1,0.2,300\n")
        fm.import_csv_to_analysis(p, csv)
        md = fm.read_frame_metadata(p)
        assert np.allclose(md["pressure"], [1.1, 2.6, 3.4, 4.1])
        assert np.allclose(md["pressure_sigma"], [0.1, 0.2, 0.15, 0.2])
        assert np.allclose(md["temperature"], [300, 305, 310, 300])

        # A later pressure-only CSV must NOT wipe the existing sigma channel
        # (apply only touches channels actually provided).
        csv2 = Path(td) / "c2.csv"
        csv2.write_text("frame,pressure_gpa\n0,5.0\n1,6.0\n2,7.0\n3,8.0\n")
        fm.import_csv_to_analysis(p, csv2)
        md = fm.read_frame_metadata(p)
        assert np.allclose(md["pressure"], [5, 6, 7, 8])
        assert np.allclose(md["pressure_sigma"], [0.1, 0.2, 0.15, 0.2]), "sigma should be preserved"

        # length mismatch is rejected
        try:
            fm.apply_to_analysis(p, pressure=[1.0, 2.0])
            assert False, "expected ValueError on length mismatch"
        except ValueError:
            pass


def test_background_step1_carries_metadata():
    """Step 1 copies temperature/timestamp and backfills pressure from filenames
    when the reduced placeholder is all-NaN."""
    import h5py
    from bulkxrd.analysis.background import run_background_separation
    names = ["UOTe-1GPa-001.tif", "UOTe-2GPa-002.tif", "UOTe-x-003.tif"]
    nb = 40
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "red.h5"
        rng = np.random.default_rng(0)
        mean = rng.normal(10, 1, (3, nb)).astype("f4")
        with h5py.File(str(red), "w") as h:
            h.attrs["unit"] = "q_A^-1"
            h.attrs["poni_text"] = "Wavelength: 4.0e-11"
            gp = h.create_group("patterns")
            gp.create_dataset("intensity", data=mean)
            gp.create_dataset("intensity_robust", data=mean.copy())
            gp.create_dataset("radial", data=np.linspace(1, 8, nb))
            gf = h.create_group("frames")
            gf.create_dataset("filename", data=np.array(names, dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
            gf.create_dataset("excluded", data=np.zeros(3, "?"))
            gf.create_dataset("pressure", data=np.full(3, np.nan))   # placeholder
            gf.create_dataset("temperature", data=np.array([300.0, 310.0, np.nan]))
            gf.create_dataset("timestamp", data=np.array(["t0", "t1", "t2"], dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
        out = Path(td) / "an.h5"
        man = run_background_separation(red, out)
        assert man["n_pressure"] == 2
        md = fm.read_frame_metadata(out)
        assert np.allclose(md["pressure"][[0, 1]], [1.0, 2.0]) and np.isnan(md["pressure"][2])
        assert np.allclose(md["temperature"][[0, 1]], [300.0, 310.0])
        assert md["timestamp"] == ["t0", "t1", "t2"]


def main() -> None:
    test_parse_pressure_units()
    test_extract_and_summary()
    test_csv_by_frame_and_filename()
    test_apply_roundtrip_and_partial_overwrite()
    test_background_step1_carries_metadata()
    print("FRAME METADATA TEST OK")


if __name__ == "__main__":
    main()
