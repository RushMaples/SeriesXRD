"""GSAS-II sequential-refinement import and mapping tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import h5py
import numpy as np
import pytest

from seriesxrd.analysis.refine_import import (
    GSASII_EXPORT_HELPER,
    RESULT_SCHEMA,
    _read_gpx,
    import_gsasii_results,
)


def _analysis(path: Path, n: int = 4) -> None:
    with h5py.File(path, "w") as h5:
        frames = h5.create_group("frames")
        frames.create_dataset(
            "filename",
            data=np.asarray([f"sample_{i}.tif" for i in range(n)], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def _phase(weight, esd, a):
    return {
        "weight_fraction": weight,
        "weight_fraction_esd": esd,
        "cell": [a, a, a, 90.0, 90.0, 90.0, a ** 3],
        "cell_esd": [0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.1],
    }


def _results(path: Path) -> None:
    data = {
        "schema": RESULT_SCHEMA,
        "schema_version": "1",
        "histograms": [
            {
                "name": "PWDR frame_0000.xy Bank 1",
                "rwp": 4.2,
                "gof": 1.1,
                "converged": True,
                "phases": {"Iron": _phase(0.75, 0.02, 2.86),
                           "MgO": _phase(0.25, 0.02, 4.21)},
            },
            {
                "name": "PWDR combined.xye Bank 1",
                "rwp": 5.4,
                "gof": 1.3,
                "converged": False,
                "phases": {"Iron": _phase(0.6, 0.03, 2.82),
                           "MgO": _phase(0.4, 0.03, 4.17)},
            },
            {
                "name": "PWDR unrelated.xye Bank 1",
                "rwp": 9.0,
                "gof": 2.0,
                "converged": True,
                "phases": {"Iron": _phase(1.0, 0.1, 2.8)},
            },
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_import_maps_single_and_grouped_histograms_without_dac_metadata(tmp_path):
    analysis = tmp_path / "analysis.h5"
    results = tmp_path / "seriesxrd_refinement.json"
    export_manifest = tmp_path / "refinement_manifest.json"
    _analysis(analysis)
    _results(results)
    with h5py.File(analysis, "r+") as h5:
        screening = h5.create_group("fractions")
        screening.attrs["method"] = "intensity_share"
        screening.create_dataset("names", data=np.asarray(["screening"], dtype="S"))
        screening.create_dataset("fractions", data=np.asarray(
            [[0.1], [0.2], [0.3], [0.4]], dtype=float))
    export_manifest.write_text(json.dumps({
        "groups": [
            {"label": "frame_0000", "frames": [0],
             "pattern": "patterns/frame_0000.xy"},
            {"label": "combined", "frames": [2, 3],
             "pattern": "combined.xye"},
        ]
    }), encoding="utf-8")

    manifest = import_gsasii_results(
        analysis, results, export_manifest=export_manifest,
        phase_map={"Iron": "Fe"},
    )

    assert manifest["method"] == "rietveld_gsasii"
    assert manifest["mapped_frames"] == [0, 2, 3]
    assert manifest["unmapped_histograms"] == ["PWDR unrelated.xye Bank 1"]
    assert manifest["phases"] == ["Fe", "MgO"]
    assert manifest["warnings"] == []
    assert not analysis.with_name(analysis.name + ".tmp").exists()

    with h5py.File(analysis, "r") as h5:
        # The released screening-fraction group remains intact.
        assert h5["fractions"].attrs["method"] == "intensity_share"
        assert np.allclose(h5["fractions/fractions"][:, 0], [0.1, 0.2, 0.3, 0.4])
        fractions = h5["refinement/fractions"][:]
        esd = h5["refinement/fraction_esd"][:]
        assert h5["refinement"].attrs["fraction_method"] == "rietveld_gsasii"
        assert np.allclose(fractions[0], [0.75, 0.25])
        assert np.isnan(fractions[1]).all()
        assert np.allclose(fractions[2], [0.6, 0.4])
        assert np.allclose(fractions[3], fractions[2])
        assert np.allclose(esd[2], [0.03, 0.03])
        assert np.array_equal(h5["refinement/group_size"][:], [1, 0, 2, 2])
        assert np.array_equal(h5["refinement/converged"][:], [1, -1, 0, 0])
        assert np.allclose(h5["refinement/rwp"][[0, 2, 3]], [4.2, 5.4, 5.4])
        assert h5["refinement/cell"].shape == (4, 2, 7)
        hist = [x.decode() if isinstance(x, bytes) else str(x)
                for x in h5["refinement/source_histogram"][:]]
        assert hist[2] == hist[3] == "PWDR combined.xye Bank 1"
        # The importer must not invent a pressure/DAC dependency.
        assert "pressure" not in h5["frames"]


def test_frame_token_maps_without_export_manifest_and_dry_run_does_not_write(tmp_path):
    analysis = tmp_path / "analysis.h5"
    results = tmp_path / "seriesxrd_refinement.json"
    _analysis(analysis, n=1)
    _results(results)

    manifest = import_gsasii_results(analysis, results, write=False)
    assert manifest["mapped_frames"] == [0]
    assert manifest["written"] is False
    with h5py.File(analysis, "r") as h5:
        assert "fractions" not in h5
        assert "refinement" not in h5


def test_import_rejects_unmappable_or_wrong_schema_results(tmp_path):
    analysis = tmp_path / "analysis.h5"
    _analysis(analysis, n=1)
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        import_gsasii_results(analysis, non_object)

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema": "something-else",
                                 "schema_version": "1",
                                 "histograms": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported refinement JSON schema"):
        import_gsasii_results(analysis, wrong)

    unmapped = tmp_path / "unmapped.json"
    unmapped.write_text(json.dumps({
        "schema": RESULT_SCHEMA, "schema_version": "1",
        "histograms": [{"name": "PWDR sample", "phases": {
            "Fe": _phase(1.0, 0.1, 2.8)}}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="No GSAS-II histogram could be mapped"):
        import_gsasii_results(analysis, unmapped)


class _FakePhase:
    def __init__(self, name, phase_id):
        self.name = name
        self.id = phase_id
        self.data = {"pId": phase_id, "General": {"Name": name}}


class _FakeSeq:
    def __init__(self):
        self.phase = _FakePhase("Fe", 0)

    def histograms(self):
        return ["PWDR frame_0000.xy Bank 1"]

    def RefData(self, _hist):
        return ({
            # WgtFrac is a derived GSAS-II value. Keep a tempting raw Scale
            # here to ensure the adapter never substitutes it.
            "parmDict": {"0:7:Scale": 999.0},
            "depParmDict": {"0:7:WgtFrac": (1.0, 0.012)},
            "depSigDict": {"0:7:WgtFrac": (1.0, 0.012)},
            "Rvals": {"Rwp": 3.5, "GOF": 1.05, "converged": True},
        }, {"data": [{"hId": 7}]})

    def get_Variable(self, _hist, _var):
        return None

    def get_cell_and_esd(self, _phase, _hist):
        return ([2.86, 2.86, 2.86, 90, 90, 90, 23.4],
                [0.01, 0.01, 0.01, 0, 0, 0, 0.1], (0,))


class _FakeProject:
    def __init__(self, _path):
        self.seq = _FakeSeq()

    def seqref(self):
        return self.seq

    def phases(self):
        return [self.seq.phase]


def test_direct_gpx_adapter_uses_weight_fraction_not_scale(tmp_path):
    gpx = tmp_path / "refinement.gpx"
    gpx.write_bytes(b"fake")
    data = _read_gpx(gpx, g2_module=SimpleNamespace(G2Project=_FakeProject))
    row = data["histograms"][0]
    assert row["phases"]["Fe"]["weight_fraction"] == 1.0
    assert row["phases"]["Fe"]["weight_fraction_esd"] == 0.012
    assert row["rwp"] == 3.5
    assert row["phases"]["Fe"]["cell"][0] == 2.86


def test_standalone_gsas_helper_is_valid_python(monkeypatch, tmp_path):
    compile(GSASII_EXPORT_HELPER, "export_seriesxrd_results.py", "exec")
    assert "WgtFrac" in GSASII_EXPORT_HELPER
    assert "import seriesxrd" not in GSASII_EXPORT_HELPER

    fake_gsas_package = types.ModuleType("GSASII")
    fake_scriptable = types.ModuleType("GSASIIscriptable")
    fake_scriptable.G2Project = _FakeProject
    monkeypatch.setitem(sys.modules, "GSASII", fake_gsas_package)
    monkeypatch.setitem(sys.modules, "GSASIIscriptable", fake_scriptable)
    namespace = {"__name__": "seriesxrd_helper_test"}
    exec(GSASII_EXPORT_HELPER, namespace)
    gpx = tmp_path / "refinement.gpx"
    gpx.write_bytes(b"fake")
    data = namespace["extract"](gpx)
    row = data["histograms"][0]
    assert row["phases"]["Fe"]["weight_fraction"] == 1.0
    assert row["phases"]["Fe"]["weight_fraction_esd"] == 0.012
    assert row["converged"] is True
