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
    GPa. ``eos`` may also carry ``p_max`` (GPa): the validity ceiling of this
    entry — usually a phase transition (e.g. NaCl B1→B2 near 30 GPa). Above it
    the phase is neither searched (identification caps its pressure window) nor
    simulated (training/ranking clamp to it), so an EOS is never extrapolated
    into a regime where the phase does not exist. ``builtin`` is set at load
    time (not persisted in the user file)."""
    name: str
    formula: str = ""
    category: str = "other"
    space_group: str = ""
    lattice: Dict[str, float] = field(default_factory=dict)
    atoms: List[Dict[str, Any]] = field(default_factory=list)
    eos: Dict[str, float] = field(default_factory=dict)
    axial_eos: Dict[str, Any] = field(default_factory=dict)
    # How to treat a phase that has no EOS under the pressure prior (Step 3a/3b):
    #   "" / "auto"        -> eos_missing (penalise off-ambient, flag uncertain)
    #   "ambient_reference"-> genuinely an ambient-only reference (penalise)
    #   "eos_missing"      -> we lack high-P EOS data (penalise, flag)
    #   "ignore_prior"     -> don't apply the pressure-prior penalty to this phase
    pressure_assumption: str = ""
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
        """True if there's a usable equation of state. Only a positive bulk
        modulus K0 is required (K0' defaults to 4; V0 cancels in the d-spacing
        scaling). Guards against placeholder all-zero EOS entries."""
        e = self.eos or {}
        try:
            return float(e.get("K0") or 0.0) > 0.0
        except (TypeError, ValueError):
            return False


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
# Equations of state — pure stdlib, central to Step 3a
# ---------------------------------------------------------------------------
#
# All forms are isothermal (~300 K) "cold" P–V EOS. They are written as a
# function of the compression ratio r = V/V0 so the absolute V0 cancels for the
# d-spacing scaling Step 3a needs (it only uses (V/V0)^(1/3)). Parameters:
#   K0   bulk modulus (GPa)            — required
#   K0p  pressure derivative K0'        — dimensionless (defaults to 4)
#   K0pp second derivative (1/GPa)      — BM4 only (defaults to 0)
# References for the functional forms:
#   Birch (1947) Phys. Rev. 71, 809   — Birch–Murnaghan 2nd/3rd/4th order
#   Vinet et al. (1987) J. Phys. C 19, L467 — universal (Vinet/Rose) EOS
#   Murnaghan (1944) PNAS 30, 244     — Murnaghan EOS

EOS_TYPES = ("BM2", "BM3", "BM4", "Vinet", "Murnaghan")


def _eos_norm_type(eos: "Dict[str, Any]") -> str:
    t = str((eos or {}).get("type", "BM3")).strip().lower().replace("-", "").replace(" ", "")
    if t in ("bm2", "birch2", "bm2nd"):
        return "BM2"
    if t in ("bm4", "birch4", "bm4th"):
        return "BM4"
    if t in ("vinet", "rose", "universal"):
        return "Vinet"
    if t in ("murnaghan", "murn"):
        return "Murnaghan"
    return "BM3"  # default / "bm", "bm3", "birchmurnaghan", unknown


def eos_pressure(eos: "Dict[str, Any]", r: float) -> float:
    """Pressure (GPa) at compression ``r = V/V0`` for any supported EOS dict.

    Supports BM2/BM3/BM4 (Birch–Murnaghan), Vinet, and Murnaghan. ``r`` ≤ 1 is
    compression; r = 1 gives P = 0. V0 is not needed (it cancels)."""
    if r <= 0:
        return float("inf")
    typ = _eos_norm_type(eos)
    K0 = float(eos["K0"])
    Kp = float(eos.get("K0p", 4.0) if eos.get("K0p") is not None else 4.0)
    if typ == "Murnaghan":
        return (K0 / Kp) * ((1.0 / r) ** Kp - 1.0)
    if typ == "Vinet":
        x = r ** (1.0 / 3.0)
        return 3.0 * K0 * (1.0 - x) / (x * x) * math.exp(1.5 * (Kp - 1.0) * (1.0 - x))
    # Birch–Murnaghan family, in Eulerian strain fE = ½[(V0/V)^(2/3) − 1].
    fE = 0.5 * ((1.0 / r) ** (2.0 / 3.0) - 1.0)
    pref = 3.0 * K0 * fE * (1.0 + 2.0 * fE) ** 2.5
    if typ == "BM2":
        return pref
    if typ == "BM4":
        Kpp = float(eos.get("K0pp", 0.0) or 0.0)
        term = (1.0 + 1.5 * (Kp - 4.0) * fE
                + 1.5 * (K0 * Kpp + (Kp - 4.0) * (Kp - 3.0) + 35.0 / 9.0) * fE * fE)
        return pref * term
    return pref * (1.0 + 1.5 * (Kp - 4.0) * fE)   # BM3


def compression_at_pressure(eos: "Dict[str, Any]", P: float,
                            *, tol: float = 1e-9, max_iter: int = 200) -> float:
    """Invert any supported EOS for the compression ratio ``r = V/V0`` at ``P``
    (GPa).

    Scans r from 1 downward for the FIRST (largest-r) crossing of P, then
    bisects. Taking the largest-r root keeps the *physical* branch even when a
    form goes non-monotonic at extreme compression (e.g. BM4 with negative K0''
    or BM3 with K0'<4), where a naive widening bracket would lock onto a spurious
    deep-compression root. Pure stdlib."""
    if P <= 0:
        return 1.0
    f = lambda r: eos_pressure(eos, r) - P     # f(1) = -P < 0; rises as r falls
    step = 0.0025
    r, lo = 1.0, None
    while r > 0.02:
        r2 = r - step
        if f(r2) >= 0.0:                        # crossing in [r2, r]
            lo, hi = r2, r
            break
        r = r2
    if lo is None:
        return 0.02                             # beyond the physical scan range
    for _ in range(max_iter):                   # bisect: f(lo) ≥ 0 > f(hi)
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol:
            return mid
        if fm > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def eos_summary(eos: "Dict[str, Any]") -> str:
    """Short human-readable EOS description, e.g. ``BM3: K0=167, K0'=5.0``."""
    e = eos or {}
    if not e.get("K0"):
        return "no EOS"
    typ = _eos_norm_type(e)
    parts = [f"K0={e.get('K0')}", f"K0'={e.get('K0p', 4.0)}"]
    if typ == "BM4" and e.get("K0pp") is not None:
        parts.append(f"K0''={e.get('K0pp')}")
    return f"{typ}: " + ", ".join(str(p) for p in parts)


# --- Birch–Murnaghan 3rd-order in absolute volume (back-compat wrappers) -----

def birch_murnaghan_pressure(V: float, V0: float, K0: float, K0p: float) -> float:
    """3rd-order Birch–Murnaghan pressure (GPa) at volume ``V`` (units of V0)."""
    return eos_pressure({"type": "BM3", "K0": K0, "K0p": K0p}, float(V) / float(V0))


def volume_at_pressure(P: float, V0: float, K0: float, K0p: float,
                       *, tol: float = 1e-8, max_iter: int = 200) -> float:
    """3rd-order BM volume (Å³) at pressure ``P`` (GPa) — back-compat wrapper."""
    r = compression_at_pressure({"type": "BM3", "K0": K0, "K0p": K0p}, float(P),
                                tol=tol, max_iter=max_iter)
    return float(V0) * r


def has_axial_eos(phase: Phase) -> bool:
    """True if the phase carries a per-axis (anisotropic) EOS for ≥1 axis."""
    ax = getattr(phase, "axial_eos", None) or {}
    return any(isinstance(ax.get(k), dict) and ax[k].get("K0") for k in ("a", "b", "c"))


def has_pressure_dof(phase: Phase) -> bool:
    """True if the phase has ANY pressure degree of freedom — a volume EOS or a
    per-axis (axial) EOS. The single predicate for "can this phase's peaks move
    with pressure?"; checking only ``has_eos()`` silently pins axial-only phases
    at ambient (identification, simulation, and ranking must all agree on this).
    """
    return phase.has_eos() or has_axial_eos(phase)


def valid_pressure_max(phase: Phase) -> float:
    """The pressure (GPa) up to which this phase entry is physically valid.

    Read from ``eos['p_max']`` (see :class:`Phase`) — typically the phase
    transition that ends the structure's stability field (NaCl-B1 → B2 near
    30 GPa, diamond-cubic Si → β-Sn near 11 GPa). ``inf`` when unset.
    Identification caps its pressure search here and the simulators clamp to
    it, so a stability-limited entry can't be fit or trained at pressures where
    the phase cannot exist.
    """
    try:
        v = float((phase.eos or {}).get("p_max") or 0.0)
    except (TypeError, ValueError):
        v = 0.0
    return v if v > 0 else float("inf")


def clamp_to_validity(phase: Phase, pressure: float) -> float:
    """``pressure`` clamped into the phase's validity range [0, p_max]."""
    return float(min(max(pressure, 0.0), valid_pressure_max(phase)))


def axial_scales(phase: Phase, pressure_gpa: float) -> "tuple":
    """Per-axis linear scale factors ``(sa, sb, sc)`` at ``pressure_gpa``.

    Each axis in ``axial_eos`` is parameterised on the *cubed* axis length (the
    PASCal/EosFit convention: fit an EOS to x³ vs P, giving an axial modulus),
    so the linear scale is ``compression(axial_eos[axis], P)**(1/3)``. Axes with
    no axial EOS fall back to the isotropic volume scale (and a symmetry-equal
    second axis, e.g. b=a for hexagonal/tetragonal, inherits the a-axis scale).
    """
    if pressure_gpa <= 0:
        return (1.0, 1.0, 1.0)
    ax = getattr(phase, "axial_eos", None) or {}
    iso = (compression_at_pressure(phase.eos, float(pressure_gpa)) ** (1.0 / 3.0)
           if phase.has_eos() else 1.0)

    def _axis(key: str, fallback: float) -> float:
        e = ax.get(key)
        if isinstance(e, dict) and e.get("K0"):
            return compression_at_pressure(e, float(pressure_gpa)) ** (1.0 / 3.0)
        return fallback

    sa = _axis("a", iso)
    L = phase.lattice or {}
    a = float(L.get("a") or 0.0)
    b = float(L.get("b") or 0.0)
    sb = _axis("b", sa if (a and b and abs(a - b) < 1e-6) else iso)
    sc = _axis("c", iso)
    return (sa, sb, sc)


def compress_lattice(phase: Phase, pressure_gpa: float) -> Dict[str, float]:
    """Lattice parameters of ``phase`` at ``pressure_gpa`` via its EOS.

    Uses the per-axis ``axial_eos`` where present (anisotropic compression);
    otherwise isotropic volume scaling (cube-root of V/V0 applied to a, b, c).
    Angles are held fixed. Returns a ``{a,b,c,alpha,beta,gamma}`` dict. Raises if
    the phase has neither a usable EOS nor an axial EOS, or no lattice.
    """
    if not has_pressure_dof(phase):
        raise ValueError(f"Phase {phase.name!r} has no usable EOS (need K0).")
    if not phase.lattice:
        raise ValueError(f"Phase {phase.name!r} has no lattice to scale.")
    sa, sb, sc = axial_scales(phase, float(pressure_gpa))
    L = dict(phase.lattice)
    for k, s in (("a", sa), ("b", sb), ("c", sc)):
        if k in L and L[k]:
            L[k] = float(L[k]) * s
    return L
