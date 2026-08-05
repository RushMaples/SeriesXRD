# Correlation result index

Paths are relative to `correlation_mapping/`.

## Active Log² results

| Result | Status | Purpose |
|---|---|---|
| `results/uote_log_squared_all_peak_formal_20260805` | ACTIVE / PASS | Combined powder + 275-anchor single-crystal + window package |
| `results/uote_log_squared_all_peak_formal_validation_20260805` | ACTIVE / PASS | Independent hierarchy/count/range/triangle validation package |
| `results/uote_single_crystal_all_peak_log_squared_20260805` | ACTIVE / PASS | 275 all-peak ROI maps, 275 location maps, and 275 original-XY waterfalls |
| `results/uote_nonlinear_squared_qwidth075_comparison_20260803/waterfall_log_correlation_on_original_profiles_qwidth075_20260804` | ACTIVE / PASS | 280 Log² correlation waterfalls shown on pre-transform positive profiles |

## Required baseline

| Result | Reason retained |
|---|---|
| `results/uote_pressure_level_peak_spots_absolute_anchor_iou_integer_window_suite_20260730_v8/peak_maps/complete_correlation_results_by_sample` | Powder location-matrix baseline for independent validation |

## Safety

Many assembled payloads are hardlinks. Treat completed roots as immutable.
Write a new analysis into a new result root, validate it, and update this index;
do not edit or reorganize payload files in place.
