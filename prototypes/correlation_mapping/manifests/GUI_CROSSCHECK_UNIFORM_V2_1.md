# GUI cross-check for uniform-correlation-v2.1

This check is read-only. It does not change peak detection, tracking, fitting,
window correlations, thresholds, or any frozen v2 result.

## GUI data currently available

The inventory file `gui_crosscheck_uniform_v2_1_inventory.csv` currently has
one dataset:

- channel: `spots`
- scan: `scan048`
- pressure frames: 19
- GUI peak fits: 522 total, 260 good (`flag == 0`)
- Pattern map source: `clean`
- HDF5: `/Users/stanley/x-ray/correlations/bulkxrd_xy_gui/scan048_spots/benchmark_analysis.h5`

The powder correlation run contains 56 scans, so this is 1/56 scan coverage.
No GUI analysis HDF5 for the `fit`/tungsten channel is currently present.
Those facts must remain explicit in the final report.

## Final command

Run this after the Handoff2 v2.1 result directory exists:

```bash
python3 /Users/stanley/x-ray/correlations/correlation_scripts/compare_gui_to_v21.py \
  --result-root /Users/stanley/x-ray/correlations/results/uote_xy_handoff2_correlations_uniform_v2_1_20260714 \
  --legacy-root /Users/stanley/x-ray/correlations/results/uote_xy_handoff2_correlations_uniform_v2_20260714 \
  --gui-inventory /Users/stanley/x-ray/correlations/manifests/gui_crosscheck_uniform_v2_1_inventory.csv
```

The default output is:

`RESULT_ROOT/validation/gui_crosscheck/`

## Audit outputs

- `gui_peak_table.csv`: the GUI Peak map table, including area, FWHM, position,
  raw flag, decoded rejection reason, and the `flag == 0` good indicator.
- `peak_match_table.csv`: one-to-one pressure+scan matches against both v2.1
  (main) and frozen v2 (legacy reference), separately for all peaks and good
  peaks only.
- `peak_match_summary.csv`: coverage plus position, FWHM, and area agreement.
- `rejection_reason_agreement.csv`: GUI flag/reason versus correlation
  reliable/unknown reason.
- `track_range_slope_comparison.csv`: official-segment pressure range and
  two-theta-versus-pressure slope for GUI-covered scans. This is conditional
  local corroboration using the v2.1 segment identity, not an independent
  GUI-only trajectory reconstruction.
- `strict_boundary_cells.csv`: ACF-strict and direct-strict adjacent-pressure
  similarities with support and 95% confidence intervals. Q25 marks only
  rank-based low-similarity candidates, not physical phase boundaries.
- `strict_boundary_agreement_summary.csv`: rank and positive-candidate agreement
  between the two strict families. Only exact ACF/direct positive intersections
  are corroborated; one-family candidates remain unresolved.
- `pattern_window_checks.csv`: the exact same 2theta windows sampled from the
  GUI Pattern map, with GUI direct/ACF Pearson and peak/intensity-change facts.
- `spots_fit_control_window_auc.csv` and companion comparison tables: spots
  versus fit/tungsten control.
- `crosscheck_summary.json`: machine-readable coverage and guardrails.
- `gui_peak_map_pressure_all_area.png`: GUI-equivalent pressure Peak map,
  Good peaks only off, colored by area.
- `gui_peak_map_pressure_good_only_area.png`: pressure Peak map with
  `flag == 0`, colored by area.
- `gui_peak_map_pressure_all_fwhm.png`: GUI-equivalent pressure Peak map,
  Good peaks only off, colored by FWHM.
- `gui_peak_map_pressure_good_only_fwhm.png`: pressure Peak map with
  `flag == 0`, colored by FWHM.
- `gui_pattern_map_pressure_clean.png`: GUI-equivalent clean Pattern map on the
  physical pressure axis; gray denotes missing/nonfinite or excluded data.
- `gui_v21_matched_peak_overlay_pressure.png`: GUI-good points with the
  admissible v2.1 reliable matches overlaid; unmatched GUI-good points are gray.
- `visualization_manifest.csv`: source and plotted/missing point counts for all
  six evidence images.
- `GUI_CROSSCHECK_REPORT.md`: 简短中文解释，明确 1/56 coverage、Peak map
  good/all、position/area/FWHM、strict 边界、Pattern 变化和 fit control 的证据范围。

## Fixed QC matching rule

For a peak pair at the same channel, scan, and pressure:

```text
abs(center_gui - center_correlation)
<= max(0.5 * (FWHM_gui + FWHM_correlation),
       2 * GUI_radial_grid_step)
```

Admissible pairs are assigned one-to-one by the Hungarian algorithm while
minimizing absolute center difference. The script also records the nearest
alternative and flags assignments whose margin is within one GUI grid step.
This is a validation matching rule, not an analysis parameter, and GUI
differences are never used to loosen v2.1.

The script fails closed unless `result-root/run_manifest.json` says
`uniform-correlation-v2.1`, `legacy-root/run_manifest.json` says
`uniform-correlation-v2`, and the two roots are different directories.

GUI flags are decoded as a bitmask:

- 1: `low_amp`
- 2: `bad_chi2`
- 4: `center_drift`
- 8: `width_bound`
- 16: `no_converge`

Different rejected-reason text is not automatically a failure because the GUI
and uniform-correlation fitters use independent acceptance tests. Peak
position, area/FWHM trend, pressure persistence, and slope agreement are the
more useful cross-checks.
