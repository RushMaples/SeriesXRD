# Correlation scripts

This folder is intentionally limited to the latest correlation deliverables,
their recursive local dependencies, and direct regression coverage.

## Retained scope

There are 42 top-level Python files:

- 8 latest-result generation or validation entrypoints;
- 18 imported runtime dependencies;
- 1 additional uniform output validator used by retained tests;
- 14 regression-test modules;
- 1 read-only workspace/integrity tool.

The authoritative list is `CODE_CATALOG.json`; exact file hashes are recorded
in `CODE_INVENTORY.csv`.

## Latest result entrypoints

| Stage | Entrypoint |
|---|---|
| Powder ROI | `pressure_level_peak_spots_absolute_anchor_iou_correlations_v8.py` |
| Single-crystal ROI | `run_single_crystal_transformed_roi_correlations.py` |
| Across/within windows | `run_transformed_integer_window_correlations.py` |
| Package assembly | `assemble_denoised_core_science_root.py` |
| Package validation | `validate_package_denoised_correlation_suites.py` |
| Powder waterfalls | `generate_denoised_peak_correlation_waterfall.py` |
| Waterfall validation | `validate_complete_formal_composite_waterfalls.py` |
| Powder support audit | `audit_powder_qwidth_support_all_frames.py` |

The current frontend indexes only the Log² portion of the formal c=0.75 package
and two Log² powder waterfall displays. Alternative denoise experiments and
their cross-transform transition suite were removed together with the
Streamlit dashboard, single-crystal waterfall experiment, unshaded waterfall
branch, generic BulkXRD adapters, workbook builders, and historical
visualizations.

## Why older-looking modules remain

```text
powder v8
  └─ v7 helpers
       └─ v6 helpers

single-crystal ROI
  ├─ refinement helpers
  ├─ global-per-peak helpers
  └─ UOTe XY handoff helpers

transformed integer windows
  ├─ all-peak and integer-window modules
  └─ uniform v2/v2.1 core and input adapters
```

These files are imported by current entrypoints. Their names reflect algorithm
history, but they are runtime dependencies, not unused legacy code.

## Commands

Run from the parent `correlation_mapping/` directory:

```bash
python3 correlation_scripts/correlation_workspace.py status
python3 correlation_scripts/correlation_workspace.py catalog
python3 correlation_scripts/correlation_workspace.py check-code
python3 correlation_scripts/correlation_workspace.py commands

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s correlation_scripts -p 'test_*.py'
```

Commands that inspect completed research runs need a local `results/` tree.
The unit tests and code-integrity check do not.

## Requirements

- `requirements-core.txt`: NumPy, SciPy, Matplotlib, pandas, h5py, Pillow.
- `requirements-uote.txt`: core plus pyFAI and tifffile for the formal UOTe workflow.
- `requirements-dev.txt`: UOTe requirements plus pytest.

The React correlation explorer has its own dependencies in
`../correlation-explorer/package.json`.
