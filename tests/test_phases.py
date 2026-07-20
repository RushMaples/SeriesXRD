"""Reference-phase library: registry, EOS math, and (optional) pymatgen paths.

The registry, merge/override semantics, and Birch-Murnaghan math are pure stdlib
and always tested. CIF parsing and pattern simulation are tested only when
pymatgen is installed (the feature degrades gracefully otherwise).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seriesxrd.analysis import phases as ph


def test_bundled():
    lib = ph.load_bundled()
    assert lib, "bundled baseline is empty"
    for name in ("Au", "Pt", "Cu", "MgO", "NaCl-B1", "Re", "Ne"):
        assert name in lib, f"missing bundled phase {name}"
    au = lib["Au"]
    assert au.builtin and au.category == "marker"
    assert au.has_structure() and au.has_eos()
    assert abs(au.eos["V0"] - au.lattice["a"] ** 3) < 0.1  # FCC conventional cell
    meta = ph.bundled_meta()
    assert "disclaimer" in meta


def test_user_override_and_merge():
    with tempfile.TemporaryDirectory() as td:
        # No user library yet -> merged == bundled.
        lib0 = ph.load_library(td)
        assert "Au" in lib0 and lib0["Au"].builtin

        # Add a brand-new user phase.
        custom = ph.Phase(name="MyStd", category="marker",
                          space_group="Fm-3m",
                          lattice={"a": 4.0, "b": 4.0, "c": 4.0,
                                   "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
                          eos={"type": "BM3", "V0": 64.0, "K0": 150.0, "K0p": 4.5})
        ph.upsert_user_phase(td, custom)
        lib1 = ph.load_library(td)
        assert "MyStd" in lib1 and not lib1["MyStd"].builtin

        # Override a bundled phase by name -> shadows, loses builtin flag.
        ph.upsert_user_phase(td, ph.Phase(name="Au", category="marker",
                                          eos={"type": "BM3", "V0": 67.8,
                                               "K0": 170.0, "K0p": 6.0}))
        lib2 = ph.load_library(td)
        assert lib2["Au"].eos["K0"] == 170.0 and not lib2["Au"].builtin

        # Removing a user entry works; removing a bundled (now-shadow) returns
        # True because the override lives in the user file.
        assert ph.remove_user_phase(td, "MyStd") is True
        assert "MyStd" not in ph.load_library(td)
        assert ph.remove_user_phase(td, "Au") is True
        assert ph.load_library(td)["Au"].builtin  # back to bundled
        # A purely bundled name that was never overridden can't be removed.
        assert ph.remove_user_phase(td, "Pt") is False

        # list_phases groups by category, returns Phase objects.
        names = [p.name for p in ph.list_phases(td)]
        assert "Au" in names and len(names) == len(ph.load_bundled())


def test_birch_murnaghan_roundtrip():
    V0, K0, K0p = 67.85, 167.0, 5.0
    assert abs(ph.volume_at_pressure(0.0, V0, K0, K0p) - V0) < 1e-6
    for P in (5.0, 30.0, 100.0, 300.0):
        V = ph.volume_at_pressure(P, V0, K0, K0p)
        assert 0 < V < V0, f"V not compressed at {P} GPa"
        P_back = ph.birch_murnaghan_pressure(V, V0, K0, K0p)
        assert abs(P_back - P) < 1e-3, f"roundtrip off at {P}: got {P_back}"


def test_eos_forms():
    """Each EOS form: P=0 at r=1, monotonic, and round-trips P→r→P."""
    K0, Kp = 160.0, 4.2
    for typ in ("BM2", "BM3", "BM4", "Vinet", "Murnaghan"):
        e = {"type": typ, "K0": K0, "K0p": Kp}
        if typ == "BM4":
            e["K0pp"] = -0.03
        assert abs(ph.eos_pressure(e, 1.0)) < 1e-9, f"{typ}: P!=0 at r=1"
        last = -1.0
        for r in (0.95, 0.9, 0.85, 0.8):
            p = ph.eos_pressure(e, r)
            assert p > last, f"{typ}: not monotonic"
            last = p
        for P in (2.0, 10.0, 40.0, 120.0):
            r = ph.compression_at_pressure(e, P)
            assert 0 < r < 1
            assert abs(ph.eos_pressure(e, r) - P) < 1e-3, f"{typ} roundtrip @ {P}"
    # Type normalisation + V0-independence of the scale factor.
    assert ph._eos_norm_type({"type": "birch-murnaghan"}) == "BM3"
    assert ph._eos_norm_type({"type": "vinet"}) == "Vinet"
    # has_eos requires a positive K0 (placeholder all-zeros is not usable).
    assert not ph.Phase(name="x", eos={"V0": 0, "K0": 0, "K0p": 0}).has_eos()
    assert ph.Phase(name="y", eos={"type": "Vinet", "K0": 160, "K0p": 4.2}).has_eos()


def test_compress_lattice():
    au = ph.load_bundled()["Au"]
    a0 = au.lattice["a"]
    at0 = ph.compress_lattice(au, 0.0)
    assert abs(at0["a"] - a0) < 1e-6
    at100 = ph.compress_lattice(au, 100.0)
    assert at100["a"] < a0  # compresses under pressure
    # isotropic: a/b/c scale by the same factor
    assert abs(at100["a"] - at100["b"]) < 1e-9


def test_pymatgen_paths():
    if not ph.pymatgen_available():
        print("  (pymatgen not installed — skipping CIF/simulate tests)")
        return
    # Simulate a bundled phase: should yield peaks with positive intensities.
    au = ph.load_bundled()["Au"]
    pat = ph.simulate_pattern(au, 0.4133, two_theta_min=0.0, two_theta_max=30.0)
    assert pat and all(p["intensity"] >= 0 for p in pat)
    assert pat == sorted(pat, key=lambda p: p["two_theta"])
    # Round-trip a CIF through import_cif using a generated CIF.
    from pymatgen.core import Lattice, Structure
    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.0782), ["Au"], [[0, 0, 0]])
    with tempfile.TemporaryDirectory() as td:
        cif = Path(td) / "au.cif"
        s.to(filename=str(cif))
        imported = ph.import_cif(td, cif, name="Au-test", category="marker")
        assert imported.space_group and imported.lattice.get("a")
        assert Path(imported.cif_path).is_file()
        assert "Au-test" in ph.load_library(td)


def test_signed_axial_expansivity():
    """axial_eos axes may carry beta = d(ln x)/dP (1/GPa): beta > 0 EXPANDS
    under pressure (negative linear compressibility — e.g. the UOTe c-axis,
    unrepresentable by a BM form whose K0 must be positive)."""
    import numpy as np
    from seriesxrd.analysis import identify as idf
    nlc = ph.Phase(name="NLC",
                   lattice={"a": 3.4, "b": 3.4, "c": 7.5,
                            "alpha": 90, "beta": 90, "gamma": 90},
                   eos={"type": "BM3", "K0": 50.0, "K0p": 4.0},
                   axial_eos={"c": {"beta": +1.5e-3}})
    assert ph.has_axial_eos(nlc) and ph.has_pressure_dof(nlc)
    sa, sb, sc = ph.axial_scales(nlc, 10.0)
    assert sc > 1.0, sc                       # c expands
    assert abs(sc - float(np.exp(1.5e-3 * 10))) < 1e-12
    assert sa < 1.0 and sb == sa              # a,b follow the volume EOS
    L = ph.compress_lattice(nlc, 10.0)
    assert L["c"] > 7.5 and L["a"] < 3.4
    # predicted_d moves an (00l) line UP in d with pressure, (h00) down.
    dp = idf.predicted_d(nlc, np.array([7.5, 3.4]), [(0, 0, 1), (1, 0, 0)], 10.0)
    assert dp[0] > 7.5 and dp[1] < 3.4
    # beta-only phase still counts as pressure-capable (no volume EOS at all).
    only = ph.Phase(name="OnlyBeta",
                    lattice={"a": 3, "b": 3, "c": 7,
                             "alpha": 90, "beta": 90, "gamma": 90},
                    axial_eos={"c": {"beta": -2e-3}})
    assert ph.has_pressure_dof(only) and not only.has_eos()
    assert ph.axial_scales(only, 5.0)[2] < 1.0    # negative beta contracts


def main() -> None:
    test_bundled()
    test_user_override_and_merge()
    test_birch_murnaghan_roundtrip()
    test_eos_forms()
    test_compress_lattice()
    test_signed_axial_expansivity()
    test_pymatgen_paths()
    print("PHASES TEST OK")


if __name__ == "__main__":
    main()
