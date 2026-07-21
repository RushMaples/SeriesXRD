"""Provenance records: /provenance group, version attrs, manifest headers.

Every analysis artifact must carry the actual SeriesXRD version separately
from the file-layout schema version, plus creation time, effective
configuration, dependency versions, and input identities.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py

from seriesxrd.core.config import VERSION
from seriesxrd.core.provenance import (dependency_versions, file_fingerprint,
                                       manifest_provenance, provenance_report)
from seriesxrd.analysis.background import run_background_separation
from seriesxrd.analysis.peaks import pseudo_voigt, run_peak_fitting


def _write_reduced(path, n=3, nb=400):
    q = np.linspace(0.5, 6.0, nb)
    rng = np.random.default_rng(7)
    base = 40.0 + 10.0 * np.exp(-q)
    sig = pseudo_voigt(q, 2.5, 80.0, 0.05, 0.4)
    mean = np.stack([base + sig + rng.normal(0, 0.5, nb) for _ in range(n)])
    with h5py.File(path, "w") as h:
        h.attrs["unit"] = "q_A^-1"
        h.attrs["poni_text"] = "wavelength: 2.0e-11"
        gp = h.create_group("patterns")
        gp.create_dataset("intensity", data=mean.astype("f4"))
        gp.create_dataset("intensity_robust", data=mean.astype("f4"))
        gp.create_dataset("radial", data=q)
        gf = h.create_group("frames")
        gf.create_dataset(
            "filename",
            data=np.array([f"f{i}_10GPa.tif" for i in range(n)], dtype=object),
            dtype=h5py.string_dtype("utf-8"))
        gf.create_dataset("excluded", data=np.zeros(n, "?"))


def test_dependency_versions():
    deps = dependency_versions()
    assert deps["numpy"] not in ("not installed", "unknown")
    assert deps["scipy"] not in ("not installed", "unknown")
    assert deps["h5py"] not in ("not installed", "unknown")
    assert "python" in deps


def test_file_fingerprint(tmp_path=None):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.bin"
        p.write_bytes(b"hello world")
        rec = file_fingerprint(p)
        assert rec["bytes"] == 11
        assert rec["hash_kind"] == "sha256"
        assert len(rec["sha256"]) == 64
        p.write_bytes(b"hello world!")
        assert file_fingerprint(p)["sha256"] != rec["sha256"]


def test_manifest_provenance_header():
    m = manifest_provenance("seriesxrd.analysis.test", "1")
    assert m["seriesxrd_version"] == VERSION
    assert m["schema_version"] == "1"
    assert m["tool"] == "seriesxrd.analysis.test"
    assert m["created_at"]


def test_analysis_file_provenance():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        red = td / "reduced.h5"
        _write_reduced(red)
        man = run_background_separation(red, td / "a.h5")
        # Manifest header: real version, schema separate.
        assert man["seriesxrd_version"] == VERSION
        assert man["schema_version"] == "1"
        assert "tool_version" not in man
        ana = td / "a.h5"
        with h5py.File(ana, "r") as h:
            assert h.attrs["seriesxrd_version"] == VERSION
            assert h.attrs["schema_version"] == "1"
            assert h.attrs["created_at"]
            g = h["provenance"]
            assert g.attrs["seriesxrd_version"] == VERSION
            cfg = json.loads(g.attrs["config_json"])
            assert cfg["max_half_window"] == 40
            deps = json.loads(g.attrs["dependencies_json"])
            assert "numpy" in deps
            assert g.attrs["input_reduced_path"] == str(red.resolve())
            assert len(g.attrs["input_reduced_sha256"]) == 64
        # An appending step records itself under /provenance/steps.
        man2 = run_peak_fitting(ana, None, source="clean")
        assert man2["seriesxrd_version"] == VERSION
        with h5py.File(ana, "r") as h:
            assert h["peaks"].attrs["seriesxrd_version"] == VERSION
            sp = h["provenance/steps/peaks"].attrs
            assert sp["seriesxrd_version"] == VERSION
            assert sp["created_at"]
        report = provenance_report(ana)
        assert f"SeriesXRD {VERSION}" in report
        assert "step peaks" in report


if __name__ == "__main__":
    test_dependency_versions()
    test_file_fingerprint()
    test_manifest_provenance_header()
    test_analysis_file_provenance()
    print("PROVENANCE TESTS OK")
