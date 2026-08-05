# Legacy and compatibility policy

“Legacy” does not automatically mean “safe to delete.” This workspace has
three distinct kinds of older code.

## Required historical dependencies

- `pressure_level_peak_correlations_v6.py`
- `pressure_level_peak_spots_qwidth_correlations_v7.py`
- `run_refinement_legacy_correlations.py`
- `single_global_per_peak.py`

These are still imported by active v8 or single-crystal programs. They may be
refactored only after their used APIs have moved and regression tests prove
numerical equivalence.

## Frozen compatibility frameworks

Uniform-correlation v2 and v2.1 intentionally coexist. v2.1 reuses frozen v2
upstream analysis and adds segmented tracking/audit layers. Hash manifests
protect v2 immutability, so the two versions must not be “deduplicated” by
overwriting one with the other.

## Generic/older four-map workflow

`run_correlation_suite.py`, `run_uote_xy_handoff_correlations.py`,
`per_peak_correlation_maps.py`, and `window_autocorrelation_correlations.py`
belong to a frame-level four-map data model. They remain useful for generic XY
and BulkXRD compatibility, but they are not the current formal pressure-level
v8 definition.

## Historical visualization branches

- `generate_correlation_waterfalls.py`
- `generate_legacy_correlation_3d_waterfalls.py`
- `generate_peak_evolution_waterfalls.py`
- `generate_relationship_waterfalls.py`

These are presentation branches, not numerical source-of-truth programs.
Plotly is optional and required only for inline legacy 3D HTML export.

## Result archives

Old v6/v7/v8, c=0.60, handoff, uniform, and synthetic result roots should be
archived only after producing a checksum/dependency manifest. Several remain
active baselines or provenance sources. No result root should be deleted merely
because a newer visual presentation exists.
