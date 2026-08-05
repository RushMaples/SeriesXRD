# Correlation result index

Paths are relative to `correlations/`.

## Active validated results

| Result | Status | Purpose |
|---|---|---|
| `results/uote_nonlinear_squared_qwidth075_comparison_20260803` | ACTIVE / PASS | Complete Log² + Exp² formal package at powder c=0.75 |
| `results/uote_nonlinear_squared_qwidth075_comparison_20260803/waterfall_complete_formal_composite_qwidth075_20260803` | ACTIVE / PASS | 560 transformed-profile powder waterfalls |
| `results/uote_nonlinear_squared_qwidth075_comparison_20260803/waterfall_log_correlation_on_original_profiles_qwidth075_20260804` | ACTIVE / PASS | 280 Log correlation waterfalls shown on pre-Log positive profiles |
| `results/uote_robust_peak_tracks_transition_analysis_20260804` | ACTIVE EXPLORATORY / PASS | Robust tracks and transition evidence across c=0.60/0.75 and Log/Exp |

## Required baselines/provenance roots

| Result | Reason retained |
|---|---|
| `results/uote_nonlinear_squared_preprocessed_comparison_20260802` | c=0.60 Log/Exp comparison baseline |
| `results/uote_pressure_level_peak_spots_absolute_anchor_iou_integer_window_suite_20260730_v8` | Previous absolute-anchor formal source and location baseline |
| `results/uote_pressure_level_peak_spots_qwidth_iou_integer_window_suite_20260730_v7*` | v7 definition/history |
| `results/uote_pressure_level_peak_ellipse_iou_integer_window_suite_20260729_v6` | v6 pressure-level baseline and reused inputs |

## Safety

Many assembled payloads are hardlinks. Treat completed roots as immutable.
Write new analysis into a new result root, validate it, and update this index;
do not edit or reorganize payload files in place.
