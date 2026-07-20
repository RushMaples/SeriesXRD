"""CIF-corpus tooling: fetch (via a file:// base URL) and screening
(parse/size/dedupe with an injected parser — no pymatgen needed)."""
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seriesxrd.analysis.corpus import fetch_cod_cifs, screen_cifs


def test_fetch_file_url():
    with tempfile.TemporaryDirectory() as src_d, tempfile.TemporaryDirectory() as dst_d:
        src = Path(src_d)
        (src / "1000041.cif").write_text("data_a\n", encoding="utf-8")
        (src / "1000042.cif").write_text("data_b\n", encoding="utf-8")
        base = src.as_uri()                       # file:///...
        man = fetch_cod_cifs(dst_d, ["1000041", "1000042", "9999999"],
                             base_url=base, log=lambda *a, **k: None)
        assert man["n_fetched"] == 2 and len(man["failures"]) == 1
        assert man["failures"][0]["id"] == "9999999"
        assert (Path(dst_d) / "1000041.cif").read_text() == "data_a\n"
        # re-run: existing files are skipped, nothing re-downloaded
        man2 = fetch_cod_cifs(dst_d, ["1000041", "1000042"],
                              base_url=base, log=lambda *a, **k: None)
        assert man2["n_skipped"] == 2 and man2["n_fetched"] == 0


def test_screen_with_injected_parser():
    infos = {
        "ok1.cif": {"formula": "SiO2", "spacegroup": 154, "n_sites": 9, "volume": 113.0},
        "ok2.cif": {"formula": "MgO", "spacegroup": 225, "n_sites": 8, "volume": 74.7},
        "dup1.cif": {"formula": "SiO2", "spacegroup": 154, "n_sites": 9, "volume": 113.2},
        "big.cif": {"formula": "C300H500", "spacegroup": 1, "n_sites": 800, "volume": 9000.0},
    }

    def parser(p: Path):
        if p.name == "bad.cif":
            raise ValueError("unparseable")
        return infos[p.name]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for nm in list(infos) + ["bad.cif"]:
            (root / nm).write_text("data_x\n", encoding="utf-8")
        man = screen_cifs(root, max_sites=200, parser=parser,
                          log=lambda *a, **k: None)
        assert man["n_total"] == 5 and man["n_kept"] == 2
        assert man["n_parse_failed"] == 1 and man["n_too_big"] == 1
        assert man["n_duplicates"] == 1
        kept = {k["file"] for k in man["kept"]}
        # exactly ONE of the ok1/dup1 twins survives (first-seen wins), plus ok2
        assert "ok2.cif" in kept and len(kept & {"ok1.cif", "dup1.cif"}) == 1
        loser = ({"ok1.cif", "dup1.cif"} - kept).pop()
        # rejects moved out of the training folder; manifest written
        assert not (root / "bad.cif").exists()
        assert (root / "rejected" / "parse_failed" / "bad.cif").is_file()
        assert (root / "rejected" / "duplicate" / loser).is_file()
        assert (root / "rejected" / "too_big" / "big.cif").is_file()
        m2 = json.loads((root / "corpus_manifest.json").read_text())
        assert m2["n_kept"] == 2
        # re-screen after moving: only the kept files remain in scope
        man3 = screen_cifs(root, max_sites=200, parser=parser,
                           log=lambda *a, **k: None)
        assert man3["n_total"] == 2 and man3["n_kept"] == 2

    # missing pymatgen (default parser) raises the instructive error
    from seriesxrd.analysis import phases as ph
    if not ph.pymatgen_available():
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "x.cif").write_text("data_x\n", encoding="utf-8")
            try:
                screen_cifs(td, log=lambda *a, **k: None)
                assert False, "expected RuntimeError without pymatgen"
            except RuntimeError as e:
                assert "pymatgen" in str(e)


def main() -> None:
    test_fetch_file_url()
    test_screen_with_injected_parser()
    print("CORPUS TEST OK")


if __name__ == "__main__":
    main()
