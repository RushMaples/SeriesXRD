# Compatibility modules retained by the latest pipeline

The cleanup removed historical scripts that were not used by the latest
indexed results. A few filenames still contain old version labels because
current entrypoints import their functions.

## Powder ROI dependency chain

- `pressure_level_peak_spots_absolute_anchor_iou_correlations_v8.py`
- `pressure_level_peak_spots_qwidth_correlations_v7.py`
- `pressure_level_peak_correlations_v6.py`

The v8 runner imports v7 helpers, and v7 imports v6 helpers.

## Single-crystal dependency chain

- `run_refinement_legacy_correlations.py`
- `single_global_per_peak.py`
- `run_uote_xy_handoff_correlations.py`

These modules supply active single-crystal tracking and refinement functions.

## Uniform compatibility layer

The transformed integer-window workflow uses the frozen uniform-v2 core plus
selected additive v2.1 tracking/input adapters. Only modules reached by that
workflow remain.

These compatibility files should be refactored only after their used APIs have
moved and regression tests demonstrate numerical equivalence.
