"""Pattern analysis stage.

Operates on the reduced 1D patterns from `seriesxrd.reduce`. The workflow
(see categorization.py) iteratively identifies, records, and removes each
feature in succession:

  Step 1 (implemented) — background-scattering isolation (background.py):
      diamond single-crystal spot residual (azimuthal mean - median) and the
      smooth/amorphous background (SNIP), leaving a clean powder pattern.
  Step 2 (implemented) — peak / profile fitting (peaks.py):
      pseudo-Voigt fit of every Bragg peak in the clean pattern -> center,
      amplitude, FWHM, eta, area, goodness-of-fit; seeds propagate across the
      frame series so a reflection keeps its identity.
  Step 3 — pressure-aware identification, candidate ranking, and unknowns.
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
    build_fit_source,
    winsorize_excess,
    auto_fit_range,
    resolve_sensitivity,
    SENSITIVITY_PRESETS,
    FIT_SOURCES,
)
from .review import (
    inspect_analysis,
    frame_data,
    peak_map,
    identify_tracks,
    structure_report,
    review_analysis,
)
from .identify import (
    radial_to_d,
    phase_reflections,
    scale_at_pressure,
    fit_pressure_for_phase,
    run_identification,
    conservative_confidence,
    pressure_window_halfwidth,
    pressure_model,
    pressure_assumption,
)
from .frame_metadata import (
    parse_pressure,
    parse_pressure_from_path,
    extract_pressures,
    summarize_pressures,
    read_pressure_csv,
    map_csv_to_frames,
    read_frame_metadata,
    apply_to_analysis,
    extract_to_analysis,
    import_csv_to_analysis,
)
from .heatmap import pattern_image, reflection_tracks, phase_layers
from .mldata import (
    make_d_grid,
    resample_to_d,
    export_ml_dataset,
    simulate_training_pattern,
    build_simulated_dataset,
    export_simulated_dataset,
)
from .ml_features import FrameFeatures, frame_features
from .ml_simulate import (
    AugmentConfig,
    build_augmented_dataset,
    export_augmented_dataset,
)
from .ml_rank import rank_candidates, read_candidates, score_phase
from .ml_scorer import PhaseScorer, CosineScorer, TorchScorer, make_scorer
from .ml_train import generate_pairs, roc_auc
from .spots import (
    detect_spots,
    link_spot_tracks,
    run_spot_tracking,
    diamond_q_lines,
    diamond_q_windows,
    load_reflection_table,
    match_tracks,
)
from .refine_import import import_gsasii_results
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
    has_axial_eos,
    has_pressure_dof,
)

__all__ = [
    "snip_baseline", "spot_residual", "contamination_score",
    "separate_background", "run_background_separation",
    "pseudo_voigt", "pseudo_voigt_area", "mad_sigma", "detect_peaks",
    "fit_pattern", "fit_dataset", "run_peak_fitting",
    "build_fit_source", "winsorize_excess", "auto_fit_range",
    "resolve_sensitivity", "SENSITIVITY_PRESETS", "FIT_SOURCES",
    "inspect_analysis", "frame_data", "peak_map", "identify_tracks",
    "structure_report", "review_analysis",
    "radial_to_d", "phase_reflections", "scale_at_pressure",
    "fit_pressure_for_phase", "run_identification",
    "conservative_confidence", "pressure_window_halfwidth", "pressure_model",
    "pressure_assumption",
    "parse_pressure", "parse_pressure_from_path", "extract_pressures",
    "summarize_pressures", "read_pressure_csv", "map_csv_to_frames",
    "read_frame_metadata", "apply_to_analysis", "extract_to_analysis",
    "import_csv_to_analysis",
    "pattern_image", "reflection_tracks", "phase_layers",
    "make_d_grid", "resample_to_d", "export_ml_dataset",
    "simulate_training_pattern", "build_simulated_dataset",
    "export_simulated_dataset",
    "FrameFeatures", "frame_features",
    "AugmentConfig", "build_augmented_dataset", "export_augmented_dataset",
    "rank_candidates", "read_candidates", "score_phase",
    "PhaseScorer", "CosineScorer", "TorchScorer", "make_scorer",
    "generate_pairs", "roc_auc",
    "detect_spots", "link_spot_tracks",
    "run_spot_tracking", "diamond_q_lines", "diamond_q_windows",
    "load_reflection_table", "match_tracks",
    "import_gsasii_results",
    "seed_analysis_config", "analysis_config_path",
    "Phase", "load_bundled", "load_library", "list_phases",
    "upsert_user_phase", "remove_user_phase", "import_cif", "parse_cif",
    "simulate_pattern", "pymatgen_available",
    "birch_murnaghan_pressure", "volume_at_pressure", "compress_lattice",
    "has_axial_eos", "has_pressure_dof",
]
