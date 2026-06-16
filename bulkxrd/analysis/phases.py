"""Reference-phase library for compound identification (Step 3 prep).

A lab-customizable registry of candidate phases — pressure markers, gasket
materials, pressure media, and sample phases — each carrying the data Step 3a
needs: a Birch–Murnaghan equation of state (V0, K0, K0') and enough structure
(space group + lattice + asymmetric-unit atoms, or an imported CIF) to simulate
peak positions and relative intensities.

Two tiers, merged at load time:

  * BUNDLED baseline — ships with the package (``refdata/baseline_phases.json``),
    read-only, common high-pressure standards. Encoded as fact data (cell + EOS
    numbers), not CIF text, so there are no licensing constraints.
  * USER library — lives in the workspace (``reference_phases/``), writable.
    A user entry with the same ``name`` shadows the bundled one, so any lab can
    override "their" gold EOS or add an in-house standard.

Pure stdlib (json/pathlib/dataclasses/shutil/math). pymatgen is optional and
imported lazily — CIF parsing and pattern simulation degrade to an instructive
error when it is absent; the registry, EOS math, and the bundled baseline work
without it.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bundled baseline JSON ships beside this module.
_BUNDLED_JSON = Path(__file__).resolve().parent / "refdata" / "baseline_phases.json"

USER_LIB_DIRNAME = "reference_phases"
USER_JSON_NAME = "user_phases.json"
_CIF_SUBDIR = "cifs"

CATEGORIES = ("marker", "gasket", "medium", "sample", "other")


# ---------------------------------------------------------------------------
# Phase record
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    """One candidate phase. ``lattice`` is ``{a,b,c,alpha,beta,gamma}`` (Å, deg);
    ``atoms`` is the asymmetric unit ``[{element,x,y,z,occ}]`` (for intensity
    simulation); ``eos`` is ``{type:'BM3', V0, K0, K0p}`` with V0 in Å³ and K0 in
    GPa. ``builtin`` is set at load time (not persisted in the user file)."""
    name: str
    formula: str = ""
    category: str = "other"
    space_group: str = ""
    lattice: Dict[str, float] = field(default_factory=dict)
    atoms: List[Dict[str, Any]] = field(default_factory=list)
    eos: Dict[str, float] = field(default_factory=dict)
    axial_eos: Dict[str, Any] = field(default_factory=dict)
    amorphous: bool = False
    cif_path: str = ""
    source: str = ""
    notes: str = ""
    builtin: bool = False

    def to_dict(self, *, drop_builtin: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if drop_builtin:
            d.pop("builtin", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Phase":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def has_structure(self) -> bool:
        """True if there's enough to simulate (a CIF, or lattice + atoms)."""
        if self.cif_path and Path(self.cif_path).is_file():
            return True
        return bool(self.lattice) and bool(self.atoms) and bool(self.space_group)

    def has_eos(self) -> bool:
        e = self.eos or {}
        return all(k in e and e[k] is not None for k in ("V0", "K0", "K0p"))


# ---------------------------------------------------------------------------
# Loading / merging
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def load_bundled() -> "Dict[str, Phase]":
    """Load the packaged baseline phases (each flagged ``builtin=True``)."""
    data = _read_json(_BUNDLED_JSON) or {}
    phases = data.get("phases", []) if isinstance(data, dict) else []
    out: Dict[str, Phase] = {}
    for d in phases:
        try:
            p = Phase.from_dict(d)
            p.builtin = True
            out[p.name] = p
        except Exception:
            continue
    return out


def bundled_meta() -> Dict[str, Any]:
    """The ``_meta`` block of the bundled file (disclaimer/version), if any."""
    data = _read_json(_BUNDLED_JSON) or {}
    return data.get("_meta", {}) if isinstance(data, dict) else {}


def user_lib_dir(workspace: "str | Path") -> Path:
    return Path(workspace).expanduser().resolve() / USER_LIB_DIRNAME


def user_json_path(workspace: "str | Path") -> Path:
    return user_lib_dir(workspace) / USER_JSON_NAME


def load_user(workspace: "str | Path") -> "Dict[str, Phase]":
    """Load the workspace's user phases (empty dict if none)."""
    data = _read_json(user_json_path(workspace)) or {}
    phases = data.get("phases", []) if isinstance(data, dict) else []
    out: Dict[str, Phase] = {}
    for d in phases:
        try:
            p = Phase.from_dict(d)
            p.builtin = False
            out[p.name] = p
        except Exception:
            continue
    return out


def load_library(workspace: "str | Path") -> "Dict[str, Phase]":
    """Merged library: user entries shadow bundled ones with the same name."""
    lib = load_bundled()
    for name, p in load_user(workspace).items():
        p.builtin = False
        lib[name] = p
    return lib


def list_phases(workspace: "str | Path") -> "List[Phase]":
    """All phases sorted by (category order, name)."""
    order = {c: i for i, c in enumerate(CATEGORIES)}
    return sorted(load_library(workspace).values(),
                  key=lambda p: (order.get(p.category, len(order)), p.name.lower()))


# ---------------------------------------------------------------------------
# User-library mutation (atomic writes)
# ---------------------------------------------------------------------------

def save_user_phases(workspace: "str | Path", phases) -> Path:
    """Atomically write the user phase list (``.tmp`` + os.replace)."""
    import os
    d = user_lib_dir(workspace)
    d.mkdir(parents=True, exist_ok=True)
    path = user_json_path(workspace)
    payload = {"schema_version": "1",
               "phases": [p.to_dict(drop_builtin=True) for p in phases]}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def upsert_user_phase(workspace: "str | Path", phase: Phase) -> Path:
    """Add or replace a single user phase by name."""
    users = load_user(workspace)
    phase.builtin = False
    users[phase.name] = phase
    return save_user_phases(workspace, users.values())


def remove_user_phase(workspace: "str | Path", name: str) -> bool:
    """Remove a user phase by name. Returns False if it wasn't a user entry
    (bundled phases can't be deleted, only shadowed)."""
    users = load_user(workspace)
    if name not in users:
        return False
    cif = users[name].cif_path
    del users[name]
    save_user_phases(workspace, users.values())
    # Best-effort cleanup of an imported CIF that lives in our managed subdir.
    try:
        cp = Path(cif)
        if cif and cp.is_file() and cp.parent == user_lib_dir(workspace) / _CIF_SUBDIR:
            cp.unlink()
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# pymatgen-backed CIF parsing & simulation (optional, lazy)
# ---------------------------------------------------------------------------

def pymatgen_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("pymatgen") is not None


def _require_pymatgen():
    if not pymatgen_available():
        raise RuntimeError(
            "pymatgen is not installed — install it (pip install pymatgen) to "
            "parse CIFs and simulate patterns, or enter lattice/EOS by hand.")


def parse_cif(cif_path: "str | Path") -> Dict[str, Any]:
    """Extract ``{formula, space_group, lattice, atoms}`` from a CIF via pymatgen.

    ``atoms`` is the symmetrized asymmetric unit so a Structure can be rebuilt
    with :func:`structure_from_phase`. Raises an instructive error if pymatgen
    is unavailable or the CIF can't be parsed.
    """
    _require_pymatgen()
    from pymatgen.core import Structure  # type: ignore
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # type: ignore

    p = Path(cif_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"CIF not found: {p}")
    struct = Structure.from_file(str(p))
    sga = SpacegroupAnalyzer(struct)
    sym = sga.get_symmetrized_structure()
    lat = struct.lattice
    atoms: List[Dict[str, Any]] = []
    for group in sym.equivalent_sites:
        site = group[0]  # one representative per orbit = asymmetric unit
        atoms.append({
            "element": site.species_string,
            "x": float(site.frac_coords[0]),
            "y": float(site.frac_coords[1]),
            "z": float(site.frac_coords[2]),
            "occ": 1.0,
        })
    return {
        "formula": struct.composition.reduced_formula,
        "space_group": sga.get_space_group_symbol(),
        "lattice": {"a": float(lat.a), "b": float(lat.b), "c": float(lat.c),
                    "alpha": float(lat.alpha), "beta": float(lat.beta),
                    "gamma": float(lat.gamma)},
        "atoms": atoms,
    }


def import_cif(workspace: "str | Path", cif_path: "str | Path", *,
               name: "Optional[str]" = None, category: str = "other",
               eos: "Optional[Dict[str, float]]" = None,
               source: str = "", notes: str = "") -> Phase:
    """Copy a CIF into the workspace user library and register a phase from it.

    Parses the CIF with pymatgen if available (filling formula/space group/
    lattice/atoms); otherwise stores the file path and leaves those blank for
    manual entry. The phase is saved to the user library and returned.
    """
    src = Path(cif_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"CIF not found: {src}")
    cif_dir = user_lib_dir(workspace) / _CIF_SUBDIR
    cif_dir.mkdir(parents=True, exist_ok=True)
    dst = cif_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)

    phase = Phase(name=name or src.stem, category=category, cif_path=str(dst),
                  eos=dict(eos or {}), source=source, notes=notes)
    if pymatgen_available():
        try:
            meta = parse_cif(dst)
            phase.formula = meta["formula"]
            phase.space_group = meta["space_group"]
            phase.lattice = meta["lattice"]
            phase.atoms = meta["atoms"]
            if not name:
                phase.name = meta["formula"] or phase.name
        except Exception as e:  # keep the import; just couldn't auto-fill
            phase.notes = (phase.notes + f"\n[CIF parse failed: {e!r}]").strip()
    upsert_user_phase(workspace, phase)
    return phase


def structure_from_phase(phase: Phase):
    """Build a pymatgen Structure from a phase (CIF if present, else
    space group + lattice + asymmetric-unit atoms). pymatgen required."""
    _require_pymatgen()
    from pymatgen.core import Lattice, Structure  # type: ignore
    if phase.cif_path and Path(phase.cif_path).is_file():
        return Structure.from_file(phase.cif_path)
    if not (phase.lattice and phase.atoms and phase.space_group):
        raise ValueError(f"Phase {phase.name!r} lacks structure (need a CIF or "
                         "space group + lattice + atoms).")
    L = phase.lattice
    lattice = Lattice.from_parameters(L["a"], L["b"], L["c"],
                                      L["alpha"], L["beta"], L["gamma"])
    species = [a["element"] for a in phase.atoms]
    coords = [[a["x"], a["y"], a["z"]] for a in phase.atoms]
    return Structure.from_spacegroup(phase.space_group, lattice, species, coords)


def simulate_pattern(phase: Phase, wavelength_angstrom: float, *,
                     two_theta_min: float = 0.0, two_theta_max: float = 90.0
                     ) -> List[Dict[str, Any]]:
    """Simulate a powder pattern (peak positions + relative intensities).

    Returns ``[{two_theta, d, intensity, hkl}]`` sorted by 2θ. pymatgen required;
    raises an instructive error otherwise or if the phase has no structure.
    """
    _require_pymatgen()
    from pymatgen.analysis.diffraction.xrd import XRDCalculator  # type: ignore
    struct = structure_from_phase(phase)
    calc = XRDCalculator(wavelength=float(wavelength_angstrom))
    pat = calc.get_pattern(struct, two_theta_range=(two_theta_min, two_theta_max))
    out: List[Dict[str, Any]] = []
    for tt, inten, hkls, d in zip(pat.x, pat.y, pat.hkls, pat.d_hkls):
        hkl = ""
        try:
            hkl = str(hkls[0]["hkl"]) if hkls else ""
        except Exception:
            pass
        out.append({"two_theta": float(tt), "d": float(d),
                    "intensity": float(inten), "hkl": hkl})
    return out


# ---------------------------------------------------------------------------
# Birch–Murnaghan EOS (3rd order) — pure stdlib, central to Step 3a
# ---------------------------------------------------------------------------

def birch_murnaghan_pressure(V: float, V0: float, K0: float, K0p: float) -> float:
    """3rd-order Birch–Murnaghan pressure (GPa) at volume ``V`` (same units as V0)."""
    x = (V0 / V) ** (1.0 / 3.0)
    x2 = x * x
    return (1.5 * K0 * (x ** 7 - x ** 5)
            * (1.0 + 0.75 * (K0p - 4.0) * (x2 - 1.0)))


def volume_at_pressure(P: float, V0: float, K0: float, K0p: float,
                       *, tol: float = 1e-8, max_iter: int = 200) -> float:
    """Invert the 3rd-order BM EOS: volume (Å³) at pressure ``P`` (GPa).

    Bisection on V/V0 ∈ (lo, 1]; pure stdlib so this module stays import-light.
    """
    if P <= 0:
        return float(V0)
    lo, hi = 0.05, 1.0  # compression rarely exceeds ~20x; widen if needed
    # P decreases as V increases; find bracket where P(lo) > P > P(hi).
    f = lambda r: birch_murnaghan_pressure(V0 * r, V0, K0, K0p) - P
    flo, fhi = f(lo), f(hi)
    # Expand the lower bound if the target pressure is enormous.
    while flo < 0 and lo > 1e-4:
        lo *= 0.5
        flo = f(lo)
    if flo * fhi > 0:
        return float(V0 * lo)  # out of bracket — return the most-compressed bound
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol:
            return float(V0 * mid)
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return float(V0 * 0.5 * (lo + hi))


def compress_lattice(phase: Phase, pressure_gpa: float) -> Dict[str, float]:
    """Lattice parameters of ``phase`` at ``pressure_gpa`` via its BM EOS.

    Isotropic volume scaling (cube-root of V/V0 applied to a, b, c) unless an
    ``axial_eos`` is provided (not yet modeled — isotropic is used and a note is
    implied). Returns a ``{a,b,c,alpha,beta,gamma}`` dict. Raises if the phase
    has no usable EOS or lattice.
    """
    if not phase.has_eos():
        raise ValueError(f"Phase {phase.name!r} has no BM EOS (V0,K0,K0p).")
    if not phase.lattice:
        raise ValueError(f"Phase {phase.name!r} has no lattice to scale.")
    e = phase.eos
    V = volume_at_pressure(float(pressure_gpa), float(e["V0"]),
                           float(e["K0"]), float(e["K0p"]))
    scale = (V / float(e["V0"])) ** (1.0 / 3.0)
    L = dict(phase.lattice)
    for k in ("a", "b", "c"):
        if k in L and L[k]:
            L[k] = float(L[k]) * scale
    return L
