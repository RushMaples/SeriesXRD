# Correlation code catalog

The authoritative per-file classification is
[`../CODE_CATALOG.json`](../CODE_CATALOG.json). It assigns every top-level
Python file to exactly one group.

| Group | Status | Purpose |
|---|---|---|
| `active_formal` | ACTIVE | Current q-width 0.75 UOTe ROI, windows, assembly, and waterfall generation |
| `required_legacy_dependencies` | REQUIRED_DEPENDENCY | v6/v7 and single-track implementations still imported by active code |
| `uniform_framework` | FROZEN_COMPATIBILITY | Frozen uniform-v2 plus additive v2.1 framework |
| `generic_four_map_and_bulkxrd` | SUPPORTED_GENERIC | Generic four-map workflows and BulkXRD adapters |
| `historical_visualizations` | LEGACY_PRESENTATION | Older 2D/3D/relationship presentation branches |
| `delivery_comparison_dashboard` | SUPPORTING | Packaging, comparison, workbook, and browsing tools |
| `exploratory_science` | EXPLORATORY | Static/moving/trajectory/transition secondary analyses |
| `validation` | VALIDATION | Independent delivery gates and synthetic checks |
| `tests` | TEST | Twenty deterministic contract/regression test modules |
| `workspace_maintenance` | MAINTENANCE | Read-only catalog/status/integrity front door |

## Why files remain physically flat

Many scripts import siblings by bare module name. The active v8 powder runner
imports v7, which imports v6. The active integer-window runner imports both the
frozen uniform-v2 core and v2.1 adapters. Historical result manifests also
store absolute script paths and some frozen baselines store exact code hashes.

Moving files before migrating those contracts would make the folder look
cleaner while reducing reproducibility. The current organization therefore
uses a catalog, docs, config, and integrity tool first. A future physical move
must preserve import compatibility and pass all hashes/tests before old paths
can be retired.

## Duplicate source warning

Several historical analysis files also exist under the workspace-level
`scripts/` directory. Most duplicate copies are byte-identical, but four core
four-map files have diverged. `correlation_scripts/` is the canonical source
for this correlation workspace; do not overwrite it mechanically from
`scripts/`.

## BulkXRD boundary

`correlations/BulkXRD` is a separately versioned upstream checkout used for
preprocessing/fitting compatibility. Its package and tests are not included in
this catalog and should remain a nested repository (or later become an
explicit submodule).
