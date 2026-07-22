"""Step 3 refinement hand-off bundle export tests (analysis/refine_export.py)."""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py

from seriesxrd.analysis.refine_export import export_refinement_bundle
from seriesxrd.analysis.phases import Phase, save_user_phases

WAVELENGTH = 0.4066  # Angstrom
N_FRAMES = 5
N_BINS = 40


def _analysis_file(path) -> "tuple[np.ndarray, np.ndarray]":
    """Synthetic analysis HDF5: q axis, wavelength attr, background channels,
    frame filenames/excluded, and a minimal /identify group naming one phase."""
    q = np.linspace(0.5, 8.0, N_BINS)
    rng = np.random.default_rng(0)
    clean = np.zeros((N_FRAMES, N_BINS))
    baseline = np.zeros((N_FRAMES, N_BINS))
    spot_residual = np.zeros((N_FRAMES, N_BINS))
    for i in range(N_FRAMES):
        clean[i] = 100.0 * np.exp(-0.5 * ((q - 3.0) / 0.05) ** 2) + rng.normal(0, 0.5, N_BINS)
        baseline[i] = 2.0 + 0.1 * q
        spot_residual[i] = rng.normal(0, 0.2, N_BINS)

    excluded = np.zeros(N_FRAMES, dtype=bool)
    excluded[2] = True
    filenames = [f"frame_{i:04d}.tif" for i in range(N_FRAMES)]

    with h5py.File(str(path), "w") as h:
        h.attrs["unit"] = "q_A^-1"
        h.attrs["wavelength"] = WAVELENGTH
        h.create_dataset("radial", data=q)
        gb = h.create_group("background")
        gb.create_dataset("clean", data=clean)
        gb.create_dataset("baseline", data=baseline)
        gb.create_dataset("spot_residual", data=spot_residual)
        gf = h.create_group("frames")
        gf.create_dataset("filename", data=np.array(filenames, dtype=object),
                          dtype=h5py.string_dtype(encoding="utf-8"))
        gf.create_dataset("excluded", data=excluded)
        gid = h.create_group("identify")
        gph = gid.create_group("Fe")
        gph.attrs["name"] = "Fe"
    return q, clean


def _fe_phase() -> Phase:
    """Bcc iron: a=2.866 A cubic, space group Im-3m, one atom in the
    asymmetric unit (pymatgen's from_spacegroup expands the second bcc site).
    atoms format per phases.Phase: [{element,x,y,z,occ}]."""
    return Phase(
        name="Fe", space_group="Im-3m",
        lattice={"a": 2.866, "b": 2.866, "c": 2.866,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        atoms=[{"element": "Fe", "x": 0.0, "y": 0.0, "z": 0.0, "occ": 1.0}],
    )


def test_export_patterns_and_instprm():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        h5_path = td / "an.h5"
        q, clean = _analysis_file(h5_path)
        out_dir = td / "bundle"

        man = export_refinement_bundle(h5_path, out_dir)

        assert man["n_frames"] == N_FRAMES - 1, man["n_frames"]   # frame 2 excluded
        assert man["unit"] == "q_A^-1"
        assert man["wavelength"] == WAVELENGTH

        patterns_dir = out_dir / "patterns"
        for i in (0, 1, 3, 4):
            assert (patterns_dir / f"frame_{i:04d}_q.xy").is_file()
            assert (patterns_dir / f"frame_{i:04d}.xy").is_file()
        # excluded frame 2 is skipped by the default frame set
        assert not (patterns_dir / "frame_0002_q.xy").is_file()
        assert not (patterns_dir / "frame_0002.xy").is_file()
        assert str(patterns_dir / "frame_0000_q.xy") in man["files_written"]
        assert str(patterns_dir / "frame_0000.xy") in man["files_written"]

        # headers: '#'-prefixed, record unit/wavelength/source/filename
        header_text = (patterns_dir / "frame_0000_q.xy").read_text()
        header_lines = [ln for ln in header_text.splitlines() if ln.startswith("#")]
        assert len(header_lines) >= 5
        assert "native_unit: q_A^-1" in header_text
        assert f"wavelength_A: {WAVELENGTH:.6f}" in header_text
        assert "source_channel:" in header_text
        assert "original_filename: frame_0000.tif" in header_text

        # values: native file carries q as-is; fit source falls back to
        # 'clean' (no /peaks group in this synthetic file -> resolved 'clean')
        data_q = np.loadtxt(patterns_dir / "frame_0000_q.xy")
        assert np.allclose(data_q[:, 0], q, atol=1e-6)
        assert np.allclose(data_q[:, 1], clean[0], atol=1e-4)

        # check one point by hand: 2theta = 2*asin(lambda*q/(4*pi)), degrees
        data_tth = np.loadtxt(patterns_dir / "frame_0000.xy")
        idx = 10
        expected_tth = math.degrees(2.0 * math.asin(WAVELENGTH * q[idx] / (4.0 * math.pi)))
        assert abs(data_tth[idx, 0] - expected_tth) < 1e-5, (data_tth[idx, 0], expected_tth)
        assert np.allclose(data_tth[:, 1], clean[0], atol=1e-4)

        # instrument parameter file
        instprm_path = out_dir / "instrument.instprm"
        assert instprm_path.is_file()
        lines = instprm_path.read_text().splitlines()
        assert lines[0] == "#GSAS-II instrument parameter file; do not add/delete items!"
        lam_line = next(ln for ln in lines if ln.startswith("Lam:"))
        assert abs(float(lam_line.split(":", 1)[1]) - WAVELENGTH) < 1e-9
        assert "Type:PXC" in lines
        assert "Zero:0.0" in lines

        # README
        readme_path = out_dir / "README.md"
        assert readme_path.is_file()
        readme_text = readme_path.read_text()
        assert "GSASIIscriptable" in readme_text
        assert "U/V/W" in readme_text or "placeholder Caglioti" in readme_text
        assert "seriesxrd-import-gsas" in readme_text
        assert (out_dir / "export_seriesxrd_results.py").is_file()
        export_manifest = json.loads(
            (out_dir / "refinement_manifest.json").read_text())
        assert export_manifest["groups"][0]["frames"] == [0]
        assert export_manifest["groups"][0]["pattern"].endswith("frame_0000.xy")


def test_export_explicit_frames_bypass_excluded():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        h5_path = td / "an.h5"
        _analysis_file(h5_path)
        out_dir = td / "bundle"

        man = export_refinement_bundle(h5_path, out_dir, frames=[0, 2])
        assert man["n_frames"] == 2
        assert (out_dir / "patterns" / "frame_0002_q.xy").is_file()   # explicit -> not filtered


def test_export_phases_written_and_skipped():
    pytest.importorskip("pymatgen")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        h5_path = td / "an.h5"
        _analysis_file(h5_path)
        out_dir = td / "bundle"

        fe = _fe_phase()
        unknown = Phase(name="Unknown1")   # no cif_path, no lattice/atoms -> must be skipped
        save_user_phases(td, [fe, unknown])

        man = export_refinement_bundle(h5_path, out_dir, phases=["Fe", "Unknown1"], workspace=td)

        written = {d["name"]: d["path"] for d in man["phases_written"]}
        skipped = {d["name"] for d in man["phases_skipped"]}
        assert "Fe" in written, man
        assert "Unknown1" in skipped, man

        fe_cif = Path(written["Fe"])
        assert fe_cif.is_file()
        assert fe_cif == out_dir / "phases" / "Fe.cif"

        # pymatgen can re-read the generated CIF (parseable, right composition/cell)
        from pymatgen.core import Structure
        struct = Structure.from_file(str(fe_cif))
        assert struct.composition.reduced_formula == "Fe"
        assert abs(struct.lattice.a - 2.866) < 1e-3
        assert abs(struct.lattice.alpha - 90.0) < 1e-3


def test_default_phases_from_identify_group():
    pytest.importorskip("pymatgen")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        h5_path = td / "an.h5"
        _analysis_file(h5_path)     # /identify/Fe present
        out_dir = td / "bundle"

        save_user_phases(td, [_fe_phase()])
        man = export_refinement_bundle(h5_path, out_dir)   # phases=None
        names = {d["name"] for d in man["phases_written"]}
        assert names == {"Fe"}, man


def test_export_frames_patterns_and_peaks_csv():
    """The frame-selection export: chosen frames only, chosen channel, and a
    combined peaks.csv restricted to those frames (flagged rows kept with
    their flag; phase column present when the attribution exists)."""
    import csv
    from seriesxrd.analysis.refine_export import export_frames
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        an = td / "an.h5"
        _, clean = _analysis_file(an)
        # Add a Step-2 peaks table: 2 peaks on frame 0, 1 on frame 1 (flagged),
        # 1 on frame 3 — with the Step-3a attribution column.
        with h5py.File(str(an), "r+") as h:
            gp = h.create_group("peaks")
            gp.create_dataset("counts", data=np.array([2, 1, 0, 1, 0], "i4"))
            gp.create_dataset("frame", data=np.array([0, 0, 1, 3], "i4"))
            gp.create_dataset("center", data=np.array([2.0, 3.0, 2.5, 3.5]))
            gp.create_dataset("center_err", data=np.full(4, 0.01))
            gp.create_dataset("amplitude", data=np.array([50.0, 80.0, 20.0, 30.0]))
            gp.create_dataset("fwhm", data=np.full(4, 0.05))
            gp.create_dataset("eta", data=np.full(4, 0.3))
            gp.create_dataset("area", data=np.full(4, 4.0))
            gp.create_dataset("chi2", data=np.full(4, 1.1))
            gp.create_dataset("flag", data=np.array([0, 0, 2, 0], "i4"))
            gp.create_dataset("phase",
                              data=np.array(["Fe", "", "Fe", "Fe"], dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
            rg = h.create_group("residual")
            rg.create_dataset("clean", data=clean * 0.5)
            rpk = rg.create_group("peaks")
            rpk.create_dataset("counts", data=np.array([1, 1, 0, 1, 0], "i4"))
            rpk.create_dataset("frame", data=np.array([0, 1, 3], "i4"))
            rpk.create_dataset("center", data=np.array([2.1, 2.6, 3.6]))
            rpk.create_dataset("amplitude", data=np.array([10.0, 20.0, 30.0]))
            rpk.create_dataset("fwhm", data=np.array([0.04, 0.05, 0.06]))
            unk = h.create_group("unknowns")
            obs = unk.create_group("obs")
            obs.create_dataset("track", data=np.array([4], "i4"))
            obs.create_dataset("frame", data=np.array([1], "i4"))
            obs.create_dataset("center", data=np.array([2.6]))
            obs.create_dataset("amplitude", data=np.array([20.0]))
            obs.create_dataset("fwhm", data=np.array([0.05]))
            tr = unk.create_group("tracks")
            tr.create_dataset("id", data=np.array([4], "i4"))
            tr.create_dataset("cluster", data=np.array([9], "i4"))

        out = td / "sel"
        man = export_frames(an, out, frames=[0, 1], source="residual")
        assert man["n_frames"] == 2
        assert man["n_residual_peaks"] == 2
        assert man["n_unknown_obs"] == 1
        pats = sorted(p.name for p in (out / "patterns").iterdir())
        # native q + 2θ for exactly frames 0 and 1
        assert pats == ["frame_0000.xy", "frame_0000_q.xy",
                        "frame_0001.xy", "frame_0001_q.xy"], pats
        with (out / "peaks.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert man["n_peaks"] == 3 == len(rows)          # frame 3 excluded
        assert {r["frame"] for r in rows} == {"0", "1"}
        flagged = [r for r in rows if r["frame"] == "1"]
        assert flagged[0]["flag"] == "2"                  # kept, not dropped
        assert flagged[0]["phase"] == "Fe"
        assert rows[0]["filename"] == "frame_0000.tif"
        with (out / "residual_peaks.csv").open() as fh:
            rrows = list(csv.DictReader(fh))
        assert len(rrows) == 2
        assert {r["frame"] for r in rrows} == {"0", "1"}
        with (out / "unknowns.csv").open() as fh:
            urows = list(csv.DictReader(fh))
        assert len(urows) == 1 and urows[0]["cluster"] == "9"

        # channel provenance lands in the .xy header
        head = (out / "patterns" / "frame_0000_q.xy").read_text().splitlines()[:6]
        assert any("residual" in ln for ln in head), head
        data_q = np.loadtxt(out / "patterns" / "frame_0000_q.xy")
        assert np.allclose(data_q[:, 1], clean[0] * 0.5, atol=1e-4)

        # peaks=False -> no CSV; frames=None -> non-excluded frames only
        out2 = td / "all"
        man2 = export_frames(an, out2, peaks=False)
        assert man2["n_peaks"] == 0 and not (out2 / "peaks.csv").exists()
        assert man2["n_frames"] == N_FRAMES - 1           # one excluded frame


def test_write_xy_sigma_keep_and_nan():
    """3-column .xye writing, keep-mask row dropping, NaN-row dropping."""
    from seriesxrd.analysis.refine_export import _write_xy
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.xye"
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([10.0, np.nan, 30.0, 40.0])
        s = np.array([1.0, 1.0, 2.0, 2.0])
        keep = np.array([True, True, True, False])
        _write_xy(p, x, y, sigma=s, keep=keep, header="h")
        data = np.loadtxt(p, ndmin=2)
        # NaN row (x=2) and keep=False row (x=4) both dropped; 3 columns
        assert data.shape == (2, 3), data.shape
        assert np.allclose(data[:, 0], [1.0, 3.0])
        assert np.allclose(data[:, 2], [1.0, 2.0])


def test_exclude_windows_drop_vs_zero():
    """exclude_mode='drop' omits the window bins; 'zero' keeps the legacy
    full-grid zeroing."""
    from seriesxrd.analysis.refine_export import export_frames
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        an = td / "an.h5"
        q, clean = _analysis_file(an)
        d0 = 2.0 * math.pi / 3.0            # q window centered at 3.0 A^-1

        man = export_frames(an, td / "drop", frames=[0], source="clean",
                            peaks=False, exclude_d=[d0])
        assert man["exclude_mode"] == "drop"
        lo, hi = man["excluded_windows"][0][1:]
        data = np.loadtxt(td / "drop" / "patterns" / "frame_0000_q.xy")
        assert not np.any((data[:, 0] >= lo) & (data[:, 0] <= hi))
        assert data.shape[0] == N_BINS - int(np.sum((q >= lo) & (q <= hi)))

        man2 = export_frames(an, td / "zero", frames=[0], source="clean",
                             peaks=False, exclude_d=[d0], exclude_mode="zero")
        assert man2["exclude_mode"] == "zero"
        data2 = np.loadtxt(td / "zero" / "patterns" / "frame_0000_q.xy")
        assert data2.shape[0] == N_BINS
        sel = (data2[:, 0] >= lo) & (data2[:, 0] <= hi)
        assert sel.any() and np.allclose(data2[sel, 1], 0.0)


_PONI_100K = """poni_version: 2.1
Detector: Pilatus100k
Detector_config: {}
Distance: 0.1
Poni1: 0.01677
Poni2: 0.0418
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 4.133e-11
"""


def test_export_gsas_raw_pressure_groups():
    """Raw re-integration: per-pressure summing, Poisson esd column,
    no-coverage bins dropped, excluded windows dropped."""
    tifffile = pytest.importorskip("tifffile")
    from seriesxrd.analysis.refine_export import export_gsas_raw
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rawdir = td / "raw"
        rawdir.mkdir()
        img = np.full((195, 487), 100, dtype=np.int32)
        img[0, 0] = -1                       # Pilatus gap/defect marker
        for i in range(3):
            tifffile.imwrite(str(rawdir / f"f{i}.tif"), img)

        red = td / "red.h5"
        with h5py.File(str(red), "w") as h:
            h.attrs["unit"] = "q_A^-1"
            h.attrs["poni_text"] = _PONI_100K
            h.attrs["npt_1d"] = 150
            h.attrs["dataset_dir"] = str(rawdir)
            fr = h.create_group("frames")
            fr.create_dataset(
                "filename", data=np.array([f"f{i}.tif" for i in range(3)],
                                          dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"))
            fr.create_dataset("excluded", data=np.zeros(3, bool))
        an = td / "an.h5"
        with h5py.File(str(an), "w") as h:
            h.create_group("frames").create_dataset(
                "pressure", data=np.array([1.0, 1.0, 2.0]))

        out = td / "gsas"
        man = export_gsas_raw(red, out, analysis_h5=an,
                              group_by_pressure=True)
        assert man["n_groups"] == 2, man
        assert {g["label"] for g in man["groups"]} == {"1GPa", "2GPa"}
        assert next(g for g in man["groups"]
                    if g["label"] == "1GPa")["n_frames"] == 2
        for name in ("1GPa_q.xye", "1GPa.xye", "2GPa_q.xye", "2GPa.xye",
                     "instrument.instprm"):
            assert (out / name).is_file(), name

        d1 = np.loadtxt(out / "1GPa_q.xye", ndmin=2)
        d2 = np.loadtxt(out / "2GPa_q.xye", ndmin=2)
        assert d1.shape[1] == 3 and d2.shape[1] == 3
        assert np.all(d1[:, 2] > 0) and np.all(d2[:, 2] > 0)
        # two summed flat frames vs one: intensity ratio ~2
        ratio = np.median(d1[:, 1]) / np.median(d2[:, 1])
        assert abs(ratio - 2.0) < 0.05, ratio

        # excluded window bins are absent from the written pattern
        qc = float(np.median(d1[:, 0]))
        d0 = 2.0 * math.pi / qc
        out2 = td / "gsas_excl"
        export_gsas_raw(red, out2, frames=[0], exclude_d=[d0])
        dx = np.loadtxt(out2 / "frame_0000_q.xye", ndmin=2)
        lo, hi = qc * (1 - 0.028), qc * (1 + 0.028)
        assert not np.any((dx[:, 0] >= lo) & (dx[:, 0] <= hi))


def main() -> None:
    test_export_patterns_and_instprm()
    test_export_explicit_frames_bypass_excluded()
    test_export_phases_written_and_skipped()
    test_default_phases_from_identify_group()
    test_export_frames_patterns_and_peaks_csv()
    test_write_xy_sigma_keep_and_nan()
    test_exclude_windows_drop_vs_zero()
    test_export_gsas_raw_pressure_groups()
    print("REFINE EXPORT TEST OK")


if __name__ == "__main__":
    main()
