"""Known-truth benchmark harness: XY ingest through the real pipeline, ranking
scorecard vs labels, and the identify verify metrics — all without pymatgen via
injected reflections."""
import sys
import math
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.analysis.benchmark import (read_xy_text, load_labels_csv,
                                        ingest_patterns, run_benchmark, CU_KA1)
from bulkxrd.analysis.peaks import pseudo_voigt
from bulkxrd.analysis.phases import Phase


def _refl(a0, scale=1.0):
    rows = [(3, 100), (4, 46), (8, 26), (11, 28), (12, 8), (16, 4)]
    d0 = np.array([a0 * scale / math.sqrt(s) for s, _ in rows])
    w = np.array([i for _, i in rows], float) / 100.0
    return d0, w, [""] * len(d0)


def _xy_pattern(d0, w, *, lam=CU_KA1, noise=1.0, seed=0):
    """Synthetic RRUFF-style 2θ pattern from a reflection list."""
    rng = np.random.default_rng(seed)
    tt = np.arange(10.0, 80.0, 0.02)
    y = 50 + 30 * np.exp(-tt / 30.0)                       # smooth background
    for d, a in zip(d0, w):
        s = lam / (2 * d)
        if s >= 1:
            continue
        c = 2 * math.degrees(math.asin(s))
        if 11 < c < 79:
            y += pseudo_voigt(tt, c, 400 * a, 0.15, 0.4)
    return tt, y + rng.normal(0, noise, tt.size)


def test_read_xy_and_labels():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "sample.txt"
        f.write_text("##NAMES=Gold\n##WAVELENGTH=1.541838\n"
                     "10.0, 100.0\n10.02, 101.5\n# comment\n10.04 99.0\n",
                     encoding="utf-8")
        r = read_xy_text(f)
        assert r["x"].size == 3 and r["meta"]["names"] == "Gold"
        assert r["y"][1] == 101.5
        c = Path(td) / "labels.csv"
        c.write_text("filename,phases\nsample.txt,Au\nmix.txt,Au;NaCl-B1\n",
                     encoding="utf-8")
        lab = load_labels_csv(c)
        assert lab["sample.txt"] == ["Au"]
        assert lab["mix.txt"] == ["Au", "NaCl-B1"]


def test_benchmark_end_to_end():
    """Two labelled patterns, two-phase library: the true phase must rank #1 for
    each pattern (hit@1 = 1) with the cosine baseline, and the identify metrics
    must verify it. Runs the REAL Step-1/Step-2 preprocessing."""
    def _cubic(name, a0):
        # Real structure fields so run_identification keeps the phase (its
        # simulation is bypassed by the injected reflections anyway).
        return Phase(name=name, space_group="Fm-3m",
                     lattice={"a": a0, "b": a0, "c": a0,
                              "alpha": 90, "beta": 90, "gamma": 90},
                     atoms=[{"element": "Au", "x": 0, "y": 0, "z": 0, "occ": 1.0}])

    au = _cubic("AuLike", 4.078)
    other = _cubic("OtherLike", 4.078 * 1.13)
    refl = {"AuLike": _refl(4.078), "OtherLike": _refl(4.078, scale=1.13)}
    with tempfile.TemporaryDirectory() as td:
        files = []
        labels = {}
        for name, key, seed in (("pat_au.txt", "AuLike", 1),
                                ("pat_other.txt", "OtherLike", 2)):
            tt, y = _xy_pattern(*refl[key][:2], seed=seed)
            f = Path(td) / name
            f.write_text("\n".join(f"{a:.3f}, {b:.2f}" for a, b in zip(tt, y)),
                         encoding="utf-8")
            files.append(f)
            labels[name] = [key]

        rep = run_benchmark(files, [au, other], labels,
                            out_dir=Path(td) / "bench", top_k=2,
                            reflections=refl, run_identify=True)
        r = rep["rank"]
        assert r["n_scored"] == 2
        assert r["hit_at_1"] == 1.0 and r["hit_at_2"] == 1.0 and r["mrr"] == 1.0
        assert Path(rep["report_json"]).is_file()
        ide = rep["identify"]
        assert "error" not in ide, ide
        assert ide["n_scored"] == 2 and ide["top_confidence_hit_rate"] == 1.0
        # a label missing from the library is reported, not crashed on
        rep2 = run_benchmark(files, [au, other], {"pat_au.txt": ["AuLike"],
                                                  "pat_other.txt": ["NotInLib"]},
                             out_dir=Path(td) / "bench2", top_k=2,
                             reflections=refl, run_identify=False)
        assert rep2["rank"]["n_scored"] == 1


def test_ingest_common_axis():
    """Files with different 2θ ranges land on one common axis, NaN outside each
    file's measured window."""
    import h5py
    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / "a.txt"
        b = Path(td) / "b.txt"
        a.write_text("\n".join(f"{x:.2f} {100 + x:.2f}"
                               for x in np.arange(10, 40, 0.05)), encoding="utf-8")
        b.write_text("\n".join(f"{x:.2f} {200 + x:.2f}"
                               for x in np.arange(30, 70, 0.05)), encoding="utf-8")
        man = ingest_patterns([a, b], Path(td) / "out")
        with h5py.File(man["reduced_h5"], "r") as h5:
            q = h5["patterns/radial"][:]
            st = h5["patterns/intensity"][:]
            assert q.min() <= 10.01 and q.max() >= 69.9
            assert np.isnan(st[0][q > 41]).all()      # a has no data there
            assert np.isnan(st[1][q < 29]).all()      # b has no data there
            assert np.isfinite(st[0][(q > 11) & (q < 39)]).all()


def main() -> None:
    test_read_xy_and_labels()
    test_ingest_common_axis()
    test_benchmark_end_to_end()
    print("BENCHMARK TEST OK")


if __name__ == "__main__":
    main()
