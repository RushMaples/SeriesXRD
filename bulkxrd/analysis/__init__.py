"""Pattern analysis stage.

Operates on the reduced 1D patterns from `bulkxrd.reduce`. The workflow
(see categorization.py) iteratively identifies, records, and removes each
feature in succession:

  Step 1 (implemented) — background-scattering isolation (background.py):
      diamond single-crystal spot residual (azimuthal mean - median) and the
      smooth/amorphous background (SNIP), leaving a clean powder pattern.
  Step 2 (implemented) — peak / profile fitting (peaks.py):
      pseudo-Voigt fit of every Bragg peak in the clean pattern -> center,
      amplitude, FWHM, eta, area, goodness-of-fit; seeds propagate across the
      frame series so a reflection keeps its identity.
  Step 3 (planned) — compound identification (deterministic EOS + ML).
"""
from __future__ import annotations

from .background import (
    snip_baseline,
    spot_residual,
    contamination_score,
    separate_background,
    run_background_separation,
)
from .peaks import (
    pseudo_voigt,
    pseudo_voigt_area,
    mad_sigma,
    detect_peaks,
    fit_pattern,
    fit_dataset,
    run_peak_fitting,
)
from .review import (
    inspect_analysis,
    frame_data,
    peak_map,
    structure_report,
    review_analysis,
)
from .session import seed_analysis_config, analysis_config_path
from .phases import (
    Phase,
    load_bundled,
    load_library,
    list_phases,
    upsert_user_phase,
    remove_user_phase,
    import_cif,
    parse_cif,
    simulate_pattern,
    pymatgen_available,
    birch_murnaghan_pressure,
    volume_at_pressure,
    compress_lattice,
)

__all__ = [
    "snip_baseline", "spot_residual", "contamination_score",
    "separate_background", "run_background_separation",
    "pseudo_voigt", "pseudo_voigt_area", "mad_sigma", "detect_peaks",
    "fit_pattern", "fit_dataset", "run_peak_fitting",
    "inspect_analysis", "frame_data", "peak_map", "structure_report",
    "review_analysis",
    "seed_analysis_config", "analysis_config_path",
    "Phase", "load_bundled", "load_library", "list_phases",
    "upsert_user_phase", "remove_user_phase", "import_cif", "parse_cif",
    "simulate_pattern", "pymatgen_available",
    "birch_murnaghan_pressure", "volume_at_pressure", "compress_lattice",
]
