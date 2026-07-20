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
from seriesxrd.analysis import frame_metadata as fm


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

        # a value-only CSV needs no pressure column (temperature series,
        # positions-only mapping runs) — but NO value column at all is rejected
        c3 = td / "temps.csv"
        c3.write_text("frame,temperature_K\n0,300\n")
        r3 = fm.read_pressure_csv(c3)
        assert r3["ok"] and r3["rows"][0]["temperature"] == 300.0
        c4 = td / "positions.csv"
        c4.write_text("frame,pos_x_mm,pos_y_mm\n0,1.0,2.0\n1,1.5,2.0\n")
        r4 = fm.read_pressure_csv(c4)
        m4 = fm.map_csv_to_frames(r4["rows"], names, len(names))
        assert m4["pos_x"][1] == 1.5 and m4["pos_y"][0] == 2.0
        c5 = td / "novalues.csv"
        c5.write_text("frame,notes\n0,hello\n")
        assert not fm.read_pressure_csv(c5)["ok"]


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


def test_partial_csv_merges_not_erases():
    """A correction sheet for a few frames must merge, not wipe the rest."""
    import h5py
    names = ["a-1GPa.tif", "b-2GPa.tif", "c-3GPa.tif", "d-4GPa.tif"]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _make_analysis(p, names)
        fm.extract_to_analysis(p)                       # [1,2,3,4] from filenames
        assert np.allclose(fm.read_frame_metadata(p)["pressure"], [1, 2, 3, 4])

        # partial CSV for only frame 1 -> merge: only frame 1 changes
        csv = Path(td) / "fix.csv"
        csv.write_text("frame,pressure_gpa\n1,99.0\n")
        man = fm.import_csv_to_analysis(p, csv)          # default merge
        assert man["n_mapped"] == 1
        assert np.allclose(fm.read_frame_metadata(p)["pressure"], [1, 99, 3, 4])

        # replace=True wipes everything not in the CSV to NaN
        fm.import_csv_to_analysis(p, csv, replace=True)
        pr = fm.read_frame_metadata(p)["pressure"]
        assert pr[1] == 99.0 and np.isnan(pr[0]) and np.isnan(pr[2]) and np.isnan(pr[3])


def test_background_step1_carries_metadata():
    """Step 1 copies temperature/timestamp and backfills pressure from filenames
    when the reduced placeholder is all-NaN."""
    import h5py
    from seriesxrd.analysis.background import run_background_separation
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


def _wavy_reduced(path, names, nb=40):
    import h5py
    rng = np.random.default_rng(0)
    n = len(names)
    mean = rng.normal(10, 1, (n, nb)).astype("f4")
    with h5py.File(str(path), "w") as h:
        h.attrs["unit"] = "q_A^-1"
        h.attrs["poni_text"] = "Wavelength: 4.0e-11"
        gp = h.create_group("patterns")
        gp.create_dataset("intensity", data=mean)
        gp.create_dataset("intensity_robust", data=mean.copy())
        gp.create_dataset("radial", data=np.linspace(1, 8, nb))
        gf = h.create_group("frames")
        gf.create_dataset("filename", data=np.array(names, dtype=object),
                          dtype=h5py.string_dtype(encoding="utf-8"))
        gf.create_dataset("excluded", data=np.zeros(n, "?"))
        gf.create_dataset("pressure", data=np.full(n, np.nan))


def test_user_edits_survive_reparse_and_step1():
    """The bug from the field: a mistyped filename token (50p7GPa for what
    should have been 5.27 GPa) was re-parsed on every Step-1 re-run,
    resurrecting the outlier a user had already corrected by hand. Manual
    edits are now marked user_edited, skipped by re-parsing, and carried
    forward through a Step-1 rebuild."""
    from seriesxrd.analysis.background import run_background_separation
    names = ["UOTe-1GPa-001.tif", "UOTe-50p7GPa-002.tif", "UOTe-3GPa-003.tif"]
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "red.h5"
        _wavy_reduced(red, names)
        out = Path(td) / "an.h5"
        run_background_separation(red, out)
        md = fm.read_frame_metadata(out)
        assert np.allclose(md["pressure"], [1.0, 50.7, 3.0])

        # Fix the outlier by hand, as the GUI's "Apply to selected" does.
        pr = md["pressure"].copy()
        pr[1] = 5.27
        fm.apply_to_analysis(out, pressure=pr, user_frames=[1])
        md = fm.read_frame_metadata(out)
        assert md["user_edited"].tolist() == [False, True, False]

        # A filename re-parse must not clobber the fix (other frames still parse).
        fm.extract_to_analysis(out)
        md = fm.read_frame_metadata(out)
        assert np.allclose(md["pressure"], [1.0, 5.27, 3.0])

        # A full Step-1 re-run rebuilds the file — the fix must be carried.
        run_background_separation(red, out)
        md = fm.read_frame_metadata(out)
        assert np.allclose(md["pressure"], [1.0, 5.27, 3.0]), md["pressure"]
        assert md["user_edited"].tolist() == [False, True, False]

        # replace=True is the explicit reset: re-parse everything, clear marks.
        fm.extract_to_analysis(out, replace=True)
        md = fm.read_frame_metadata(out)
        assert np.allclose(md["pressure"], [1.0, 50.7, 3.0])
        assert not md["user_edited"].any()


def test_csv_import_marks_user_frames():
    """CSV rows are deliberate human input: mapped frames get the user mark, so
    a later filename re-parse cannot overwrite them."""
    names = ["a-1GPa.tif", "b-2GPa.tif", "c-3GPa.tif"]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _make_analysis(p, names)
        fm.extract_to_analysis(p)
        csv = Path(td) / "fix.csv"
        csv.write_text("frame,pressure_gpa\n1,9.9\n")
        fm.import_csv_to_analysis(p, csv)
        md = fm.read_frame_metadata(p)
        assert md["user_edited"].tolist() == [False, True, False]
        fm.extract_to_analysis(p)                    # re-parse: keeps the CSV value
        assert np.allclose(fm.read_frame_metadata(p)["pressure"], [1.0, 9.9, 3.0])


def test_positions_csv_and_step1_carry():
    """Positions imported from a CSV land in /frames/pos_x+pos_y, are marked
    user-edited, and survive a Step-1 rebuild (the coordinate grid map needs
    them to persist like every other deliberate metadata input)."""
    from seriesxrd.analysis.background import run_background_separation
    names = ["m-001.tif", "m-002.tif", "m-003.tif", "m-004.tif"]
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "red.h5"
        _wavy_reduced(red, names)
        out = Path(td) / "an.h5"
        run_background_separation(red, out)
        csv = Path(td) / "pos.csv"
        csv.write_text("filename,pos_x_mm,pos_y_mm\n"
                       "m-001.tif,0.0,0.0\nm-002.tif,0.1,0.0\n"
                       "m-003.tif,0.0,0.1\nm-004.tif,0.1,0.1\n")
        man = fm.import_csv_to_analysis(out, csv)
        assert man["n_mapped"] == 4
        md = fm.read_frame_metadata(out)
        assert np.allclose(md["pos_x"], [0.0, 0.1, 0.0, 0.1])
        assert np.allclose(md["pos_y"], [0.0, 0.0, 0.1, 0.1])
        assert np.isnan(md["pressure"]).all()      # positions-only CSV: P untouched
        run_background_separation(red, out)        # rebuild
        md = fm.read_frame_metadata(out)
        assert np.allclose(md["pos_x"], [0.0, 0.1, 0.0, 0.1])
        assert np.allclose(md["pos_y"], [0.0, 0.0, 0.1, 0.1])


def test_positions_from_edf_headers():
    """Stage positions read from real EDF headers (direct key and the ESRF
    motor_mne/motor_pos pair convention), with the instructive failure path."""
    try:
        import fabio
        from fabio.edfimage import EdfImage
    except ImportError:
        print("  (fabio missing - header test skipped)")
        return
    from seriesxrd.analysis.frame_metadata import (
        import_positions_from_headers, frame_header_keys)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        names = []
        for i in range(3):
            f = td / f"map-{i:03d}.edf"
            img = EdfImage(data=np.zeros((4, 4), dtype=np.int32))
            img.header["samx"] = f"{0.1 * i:.3f}"                    # direct key
            img.header["motor_mne"] = "phi kappa samy"
            img.header["motor_pos"] = f"10.0 20.0 {0.2 * i:.3f}"     # paired key
            img.write(str(f))
            names.append(str(f))
        p = td / "an.h5"
        _make_analysis(p, names)
        man = import_positions_from_headers(p, "SAMX", "samy")       # case-insensitive
        assert man["n_mapped"] == 3
        md = fm.read_frame_metadata(p)
        assert np.allclose(md["pos_x"], [0.0, 0.1, 0.2])
        assert np.allclose(md["pos_y"], [0.0, 0.2, 0.4])
        assert md["user_edited"].all()
        keys = frame_header_keys(p)
        assert keys["ok"] and "samx" in keys["keys"] and "samy" in keys["keys"]
        try:
            import_positions_from_headers(p, "nope_x", "nope_y")
            assert False, "expected ValueError with available keys"
        except ValueError as e:
            assert "samx" in str(e)                                  # names the keys


def test_coordinate_grid_snapping():
    """coordinate_grid recovers the scan grid from jittered stage positions,
    independent of collection order (serpentine here)."""
    from seriesxrd.analysis.heatmap import coordinate_grid
    rng = np.random.default_rng(1)
    xs, ys, order = [], [], []
    fi = 0
    for r in range(3):                     # 3 rows x 4 cols, serpentine
        cols = range(4) if r % 2 == 0 else range(3, -1, -1)
        for c in cols:
            xs.append(0.05 * c + rng.normal(0, 0.002))   # jitter << pitch
            ys.append(0.05 * r + rng.normal(0, 0.002))
            order.append((r, c, fi))
            fi += 1
    cg = coordinate_grid(np.array(xs), np.array(ys))
    assert cg["ok"], cg["error"]
    g = cg["grid"]
    assert g.shape == (3, 4) and cg["fill_frac"] == 1.0 and cg["n_collisions"] == 0
    for r, c, k in order:
        assert g[r, c] == k, (r, c, k, g)
    # a missing frame leaves a hole, everything else still lands
    cg2 = coordinate_grid(np.array(xs[:-1]), np.array(ys[:-1]))
    assert cg2["ok"] and cg2["grid"].shape == (3, 4)
    assert int(np.sum(cg2["grid"] < 0)) == 1
    # exact (jitter-free) coordinates also cluster correctly
    cg3 = coordinate_grid(np.repeat([0.0, 1.0], 2), np.tile([0.0, 1.0], 2))
    assert cg3["ok"] and cg3["grid"].shape == (2, 2) and cg3["fill_frac"] == 1.0


def test_prior_range_offenders_names_the_outlier():
    """The identify range-widening warning names the frame(s) responsible and
    flags a value far off the series median as a likely metadata error."""
    from seriesxrd.analysis.identify import prior_range_offenders
    pr = np.array([1.0, 2.0, 50.7, 3.0, np.nan])
    w = np.full(5, 2.0)
    names = ["a.tif", "b.tif", "UOTe-50p7GPa-002.tif", "c.tif", "d.tif"]
    lines = prior_range_offenders(pr, w, 0.0, 15.0, names=names)
    assert len(lines) == 1, lines
    assert "frame 2" in lines[0] and "50p7GPa" in lines[0] and "50.7" in lines[0]
    assert "median" in lines[0]                       # the outlier hint fired
    # Low priors near 0 must NOT be flagged when p_min is 0 (the search's low
    # side clamps at 0, so they never actually widen the range).
    assert prior_range_offenders(np.array([1.0, 2.0]), np.full(2, 2.0),
                                 0.0, 15.0) == []
    # A genuinely out-of-range value without outlier character gets no hint.
    tight = prior_range_offenders(np.array([14.0, 15.5, 16.0]), np.full(3, 2.0),
                                  0.0, 15.0)
    assert tight and all("median" not in s for s in tight)


def main() -> None:
    test_parse_pressure_units()
    test_extract_and_summary()
    test_csv_by_frame_and_filename()
    test_apply_roundtrip_and_partial_overwrite()
    test_partial_csv_merges_not_erases()
    test_background_step1_carries_metadata()
    test_user_edits_survive_reparse_and_step1()
    test_csv_import_marks_user_frames()
    test_positions_csv_and_step1_carry()
    test_positions_from_edf_headers()
    test_coordinate_grid_snapping()
    test_prior_range_offenders_names_the_outlier()
    print("FRAME METADATA TEST OK")


if __name__ == "__main__":
    main()
